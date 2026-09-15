import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GroupShuffleSplit

# Setup Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from features.feature_builder import FEATURE_NAMES

DATA_CSV  = os.path.join(PROJECT_ROOT, 'data', 'features', 'features_all.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
REPORT_PATH = os.path.join(PROJECT_ROOT, 'reports', 'evaluation.md')

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_CSV)
    df = df[df['label'] != 'straight_drive']
    df = df.dropna(subset=FEATURE_NAMES + ['label', 'source_group'])

    encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    scaler  = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    model   = joblib.load(os.path.join(MODEL_DIR, 'xgboost_v1.pkl'))

    # Encode labels
    X = df[FEATURE_NAMES].values
    y = encoder.transform(df['label'].values)
    groups = df['source_group'].values

    # Strict GroupShuffleSplit to match retrain split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_test, y_test = X[test_idx], y[test_idx]
    X_test_sc = scaler.transform(X_test)
    
    probs = model.predict_proba(X_test_sc)
    base_preds = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)
    
    # Apply 0.65 threshold
    threshold = 0.65
    rejected_count = int(np.sum(max_probs < threshold))
    accepted_idx = max_probs >= threshold
    
    y_test_acc = y_test[accepted_idx]
    base_preds_acc = base_preds[accepted_idx]
    
    raw_acc = accuracy_score(y_test, base_preds) * 100
    post_acc = accuracy_score(y_test_acc, base_preds_acc) * 100
    reject_pct = (rejected_count / len(y_test)) * 100
    
    # Generate report content
    report_text = f"""CricketVision AI Deployment Report

Production model: XGBoost
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Overall Accuracy (Raw): {raw_acc:.2f}%
Accuracy (Post-Threshold): {post_acc:.2f}%
Rejected Samples: {rejected_count} ({reject_pct:.1f}%)

Detailed Report:
{classification_report(y_test_acc, base_preds_acc, target_names=encoder.classes_)}
"""
    
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        f.write(report_text)
        
    print(f"Report written successfully to {REPORT_PATH}")
    print(f"Post-Threshold Accuracy: {post_acc:.2f}%")
    print(f"Rejected: {rejected_count} ({reject_pct:.1f}%)")

if __name__ == "__main__":
    main()
