import os
import sys
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Add src/project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

DATA_CSV = os.path.join(PROJECT_ROOT, 'data', 'features', 'features_all.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# Load the dynamic feature names (which are now the 51 static features)
from features.feature_builder import FEATURE_NAMES

def main():
    print(f"Loading data from {DATA_CSV}...")
    df = pd.read_csv(DATA_CSV)
    df = df[df['label'] != 'straight_drive']
    
    if 'source_group' not in df.columns:
        print("ERROR: source_group must exist for leakage-free splitting")
        return
        
    # Drop rows with NAs in our features or target
    df.dropna(subset=FEATURE_NAMES + ['label'], inplace=True)
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(FEATURE_NAMES)}")
    
    # 1. ENCODE LABELS
    encoder = LabelEncoder()
    y_raw = df['label'].values
    y = encoder.fit_transform(y_raw)
    
    # 2. FEATURE EXTRACTION
    X = df[FEATURE_NAMES].values
    groups = df['source_group'].values
    
    # 3. LEAKAGE-FREE SPLIT
    print("Splitting data (GroupShuffleSplit)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
    X_test, y_test, groups_test    = X[test_idx], y[test_idx], groups[test_idx]
    
    print(f"Training shapes: X={X_train.shape}, Y={y_train.shape}")
    print(f"Testing shapes : X={X_test.shape}, Y={y_test.shape}")
    
    # 4. SCALE FEATURES
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    
    # 5. CLASS WEIGHTING FOR IMBALANCE
    print("Computing balanced sample weights for XGBoost...")
    sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train)

    # 6. XGBOOST + RANDOMIZED SEARCH
    xgb = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        seed=42
    )

    # Hyperparameter grid for cross-validation search
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 150],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [2, 5, 10],
        'gamma': [0, 0.1, 0.3],
    }

    print("Running RandomizedSearchCV on XGBoost...")
    rs = RandomizedSearchCV(
        estimator=xgb, 
        param_distributions=param_grid,
        n_iter=30,
        cv=3, 
        scoring='f1_macro', 
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    rs.fit(X_train_sc, y_train, sample_weight=sample_weights_train)
    print("Best params:", rs.best_params_)
    
    best_model = rs.best_estimator_
    
    # 7. EVALUATE
    train_preds = best_model.predict(X_train_sc)
    test_preds  = best_model.predict(X_test_sc)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc  = accuracy_score(y_test, test_preds)
    
    print(f"\n[RETRAINED MODEL RESULTS]")
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Test Accuracy:  {test_acc*100:.2f}%")
    
    print("\nClassification Report (Test):")
    print(classification_report(y_test, test_preds, target_names=encoder.classes_))
    
    # 8. SAVE ARTIFACTS
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save a backup of the old model first
    old_model_path = os.path.join(MODEL_DIR, 'xgboost_v1.pkl')
    if os.path.exists(old_model_path):
        import shutil
        shutil.copy(old_model_path, os.path.join(MODEL_DIR, 'xgboost_v1_backup.pkl'))
        print("Backup created for old xgboost_v1.pkl")
        
    print("Saving updated preprocessors and classifier...")
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'xgboost_v1.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(encoder, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    print("SUCCESS: Model Retraining Complete!")

if __name__ == "__main__":
    main()
