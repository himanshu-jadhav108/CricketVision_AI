"""Threshold sensitivity analysis for CricketVision AI.

Sweeps confidence threshold values from 0.50 to 0.80 to measure the trade-off
between post-threshold classification accuracy and test sample rejection rate.
Outputs reports/threshold_sweep.csv and reports/threshold_sweep.png.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.feature_builder import FEATURE_NAMES

DATA_CSV = PROJECT_ROOT / "data" / "features" / "features_all.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    print(f"Loading data from {DATA_CSV}...")
    df = pd.read_csv(DATA_CSV)
    df = df[df["label"] != "straight_drive"]
    df = df.dropna(subset=FEATURE_NAMES + ["label", "source_group"])

    encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    model = joblib.load(MODEL_DIR / "xgboost_v1.pkl")

    X = df[FEATURE_NAMES].values
    y = encoder.transform(df["label"].values)
    groups = df["source_group"].values

    # Identical GroupShuffleSplit to evaluate_model.py
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_test, y_test = X[test_idx], y[test_idx]
    X_test_sc = scaler.transform(X_test)

    probs = model.predict_proba(X_test_sc)
    base_preds = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    results = []

    for t in thresholds:
        accepted_idx = max_probs >= t
        rejected_count = int(np.sum(~accepted_idx))
        samples_accepted = int(np.sum(accepted_idx))
        samples_total = int(len(y_test))

        y_acc = y_test[accepted_idx]
        preds_acc = base_preds[accepted_idx]

        acc_pct = float(accuracy_score(y_acc, preds_acc) * 100) if samples_accepted > 0 else 0.0
        rej_pct = float((rejected_count / samples_total) * 100)

        results.append({
            "threshold": t,
            "accuracy_pct": round(acc_pct, 2),
            "rejection_pct": round(rej_pct, 2),
            "samples_accepted": samples_accepted,
            "samples_total": samples_total,
        })

    res_df = pd.DataFrame(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "threshold_sweep.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved sweep metrics to {csv_path}:")
    print(res_df.to_string(index=False))

    # Correctness check for threshold=0.65
    row_65 = res_df[res_df["threshold"] == 0.65].iloc[0]
    expected_rej = 32.4
    diff = abs(row_65["rejection_pct"] - expected_rej)
    print(f"\nVerification Check (Threshold 0.65):")
    print(f"  Post-threshold Accuracy: {row_65['accuracy_pct']}%")
    print(f"  Rejection Rate         : {row_65['rejection_pct']}% (Expected: ~{expected_rej}%, Delta: {diff:.2f}%)")
    assert diff <= 1.0, f"Rejection rate {row_65['rejection_pct']}% is not within 1% of {expected_rej}%!"
    print("  -> Acceptance check PASSED: Rejection rate is within 1.0% of evaluation report.")

    # Generate visualization chart
    print("\nGenerating reports/threshold_sweep.png...")
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(
        res_df["threshold"],
        res_df["accuracy_pct"],
        marker="o",
        linewidth=2.5,
        color="#10b981",
        label="Post-Threshold Accuracy (%)",
    )
    plt.plot(
        res_df["threshold"],
        res_df["rejection_pct"],
        marker="s",
        linewidth=2.5,
        color="#f43f5e",
        label="Rejection Rate (%)",
    )

    # Highlight default threshold 0.65
    plt.axvline(x=0.65, color="#38bdf8", linestyle="--", linewidth=1.8, label="Production Default (0.65)")
    plt.scatter(
        [0.65, 0.65],
        [row_65["accuracy_pct"], row_65["rejection_pct"]],
        color="#38bdf8",
        s=80,
        zorder=5,
    )

    # Annotate the 0.65 operating point
    plt.annotate(
        f"{row_65['accuracy_pct']:.1f}% Acc\n{row_65['rejection_pct']:.1f}% Rej",
        xy=(0.65, row_65["accuracy_pct"]),
        xytext=(0.67, row_65["accuracy_pct"] - 5),
        arrowprops=dict(facecolor="#38bdf8", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", ec="#cbd5e1", alpha=0.9),
    )

    plt.title("CricketVision AI: Threshold Sensitivity Trade-Off", fontsize=14, fontweight="bold", pad=14)
    plt.xlabel("Confidence Threshold", fontsize=12, fontweight="bold", labelpad=8)
    plt.ylabel("Percentage (%)", fontsize=12, fontweight="bold", labelpad=8)
    plt.xticks(thresholds, [f"{t:.2f}" for t in thresholds], fontsize=10)
    plt.yticks(np.arange(0, 105, 10), fontsize=10)
    plt.ylim(0, 102)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="center left", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=10)
    plt.tight_layout()

    img_path = REPORTS_DIR / "threshold_sweep.png"
    plt.savefig(img_path, dpi=150)
    plt.close()
    print(f"Saved threshold sensitivity chart to {img_path}")


if __name__ == "__main__":
    main()
