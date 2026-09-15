"""Regenerate evaluation and EDA charts from the current production model.

This script reproduces the exact test split used in evaluate_model.py, evaluates the
current 6-class XGBoost model at the production confidence threshold (0.65), and
updates the stale performance assets.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.feature_builder import FEATURE_NAMES

DATA_CSV = PROJECT_ROOT / "data" / "features" / "features_all.csv"
MODEL_DIR = PROJECT_ROOT / "models"
PERF_DIR = PROJECT_ROOT / "assets" / "performance"


def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading data from {DATA_CSV}...")
    df = pd.read_csv(DATA_CSV)

    # Filter out straight_drive exactly as in evaluate_model.py
    df = df[df["label"] != "straight_drive"]
    df = df.dropna(subset=FEATURE_NAMES + ["label", "source_group"])

    print(f"Loading production models from {MODEL_DIR}...")
    encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    model = joblib.load(MODEL_DIR / "xgboost_v1.pkl")

    class_names = list(encoder.classes_)
    print(f"Model classes ({len(class_names)}): {class_names}")
    print(f"Total features ({len(FEATURE_NAMES)}): {len(FEATURE_NAMES)} features")

    # Encode features and targets
    X = df[FEATURE_NAMES].values
    y = encoder.transform(df["label"].values)
    groups = df["source_group"].values

    # Strict GroupShuffleSplit mirroring evaluate_model.py / retrain.py
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_test, y_test = X[test_idx], y[test_idx]
    X_test_sc = scaler.transform(X_test)

    # Predict probabilities and apply 0.65 threshold
    probs = model.predict_proba(X_test_sc)
    base_preds = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)

    threshold = 0.65
    accepted_idx = max_probs >= threshold
    rejected_count = int(np.sum(~accepted_idx))
    reject_pct = (rejected_count / len(y_test)) * 100

    y_test_acc = y_test[accepted_idx]
    base_preds_acc = base_preds[accepted_idx]

    print(f"Test samples: {len(y_test)} total | {len(y_test_acc)} accepted | {rejected_count} ({reject_pct:.1f}%) rejected")

    PERF_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. eval_confusion_matrix_pct.png (Row-normalized confusion matrix heatmap)
    # ─────────────────────────────────────────────────────────────────────────
    print("Generating eval_confusion_matrix_pct.png...")
    cm_pct = confusion_matrix(y_test_acc, base_preds_acc, normalize="true") * 100

    plt.figure(figsize=(10, 8), dpi=150)
    sns.set_theme(style="white")
    ax = sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Greens",
        xticklabels=[c.replace("_", " ").title() for c in class_names],
        yticklabels=[c.replace("_", " ").title() for c in class_names],
        cbar_kws={"label": "Classification Rate (%)"},
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="#e2e8f0",
    )
    plt.title("CricketVision AI: Confusion Matrix (%)", fontsize=15, fontweight="bold", pad=16)
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("True Label", fontsize=12, fontweight="bold", labelpad=10)
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    cm_path = PERF_DIR / "eval_confusion_matrix_pct.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  -> Saved {cm_path} (Matrix shape: {cm_pct.shape[0]}x{cm_pct.shape[1]})")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. eval_per_class_f1.png (Bar chart of per-class F1)
    # ─────────────────────────────────────────────────────────────────────────
    print("Generating eval_per_class_f1.png...")
    rep = classification_report(y_test_acc, base_preds_acc, target_names=class_names, output_dict=True)
    f1_scores = [rep[cls]["f1-score"] for cls in class_names]
    pretty_names = [c.replace("_", " ").title() for c in class_names]

    plt.figure(figsize=(10, 6), dpi=150)
    colors = sns.color_palette("viridis", len(class_names))
    bars = plt.bar(pretty_names, f1_scores, color=colors, edgecolor="#334155", width=0.55)

    for bar, score in zip(bars, f1_scores):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.title("CricketVision AI: Per-Class F1 Score (Post-Threshold)", fontsize=14, fontweight="bold", pad=14)
    plt.xlabel("Shot Class", fontsize=12, fontweight="bold", labelpad=8)
    plt.ylabel("F1 Score", fontsize=12, fontweight="bold", labelpad=8)
    plt.ylim(0, 1.15)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    f1_path = PERF_DIR / "eval_per_class_f1.png"
    plt.savefig(f1_path, dpi=150)
    plt.close()
    print(f"  -> Saved {f1_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. feature_importance.png (Top 20 features by model.feature_importances_)
    # ─────────────────────────────────────────────────────────────────────────
    print("Generating feature_importance.png...")
    raw_importances = model.feature_importances_
    feat_series = pd.Series(raw_importances, index=FEATURE_NAMES).sort_values(ascending=False)
    top_20 = feat_series.head(20).iloc[::-1]  # reverse for ascending horizontal bar chart

    plt.figure(figsize=(12, 10), dpi=150)
    y_pos = np.arange(len(top_20))
    bar_colors = sns.color_palette("Blues_r", len(top_20))[::-1]
    plt.barh(y_pos, top_20.values, color=bar_colors, edgecolor="#1e293b", height=0.65)
    plt.yticks(y_pos, top_20.index, fontsize=10)
    plt.xlabel("Feature Importance (Gain / Weight)", fontsize=12, fontweight="bold", labelpad=8)
    plt.title("CricketVision AI: Top 20 Biomechanical Feature Importances", fontsize=14, fontweight="bold", pad=14)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    for i, val in enumerate(top_20.values):
        plt.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=9, color="#1e293b", fontweight="bold")

    plt.tight_layout()
    fi_path = PERF_DIR / "feature_importance.png"
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"  -> Saved {fi_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. eda_class_distribution.png (6 trained classes value counts)
    # ─────────────────────────────────────────────────────────────────────────
    print("Generating eda_class_distribution.png...")
    class_counts = df["label"].value_counts().loc[class_names]

    plt.figure(figsize=(10, 6), dpi=150)
    dist_colors = sns.color_palette("crest", len(class_names))
    dist_bars = plt.bar(
        [c.replace("_", " ").title() for c in class_names],
        class_counts.values,
        color=dist_colors,
        edgecolor="#334155",
        width=0.55,
    )

    for bar, count in zip(dist_bars, class_counts.values):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 20,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.title("CricketVision AI: Class Distribution (6 Shot Classes)", fontsize=14, fontweight="bold", pad=14)
    plt.xlabel("Shot Class", fontsize=12, fontweight="bold", labelpad=8)
    plt.ylabel("Sample Count", fontsize=12, fontweight="bold", labelpad=8)
    plt.ylim(0, max(class_counts.values) * 1.15)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    dist_path = PERF_DIR / "eda_class_distribution.png"
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"  -> Saved {dist_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. confusion_matrix.png (Raw counts confusion matrix heatmap)
    # ─────────────────────────────────────────────────────────────────────────
    print("Generating confusion_matrix.png (raw counts)...")
    cm_raw = confusion_matrix(y_test_acc, base_preds_acc)

    plt.figure(figsize=(10, 8), dpi=150)
    sns.set_theme(style="white")
    ax = sns.heatmap(
        cm_raw,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.replace("_", " ").title() for c in class_names],
        yticklabels=[c.replace("_", " ").title() for c in class_names],
        cbar_kws={"label": "Sample Count"},
        linewidths=0.5,
        linecolor="#e2e8f0",
    )
    plt.title("CricketVision AI: Confusion Matrix (Thresholded)", fontsize=15, fontweight="bold", pad=16)
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("True Label", fontsize=12, fontweight="bold", labelpad=10)
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    raw_cm_path = PERF_DIR / "confusion_matrix.png"
    plt.savefig(raw_cm_path, dpi=150)
    plt.close()
    print(f"  -> Saved {raw_cm_path} (Matrix shape: {cm_raw.shape[0]}x{cm_raw.shape[1]})")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. eda_boxplots.png (6 joint angle features across 6 shot classes)
    # ─────────────────────────────────────────────────────────────────────────
    print("Generating eda_boxplots.png...")
    angle_features = [
        "angle_r_elbow_3d",
        "angle_l_elbow_3d",
        "angle_r_shoulder_3d",
        "angle_l_shoulder_3d",
        "angle_r_knee_3d",
        "angle_l_knee_3d",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=150)
    axes = axes.flatten()
    box_palette = sns.color_palette("rocket", len(class_names))

    for idx, feat in enumerate(angle_features):
        sns.boxplot(
            data=df,
            x="label",
            y=feat,
            order=class_names,
            palette=box_palette,
            ax=axes[idx],
            fliersize=2.5,
        )
        axes[idx].set_title(feat.replace("_", " ").title(), fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel(feat, fontsize=9)
        axes[idx].set_xticklabels([c.replace("_", " ") for c in class_names], rotation=40, ha="right", fontsize=8.5)
        axes[idx].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("CricketVision AI: Biomechanical Joint Angle Distributions (6 Shot Classes)", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()

    boxplots_path = PERF_DIR / "eda_boxplots.png"
    plt.savefig(boxplots_path, dpi=150)
    plt.close()
    print(f"  -> Saved {boxplots_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. eda_correlation.png (Feature correlation heatmap for current 51 features)
    # ─────────────────────────────────────────────────────────────────────────
    print("Generating eda_correlation.png...")
    plt.figure(figsize=(16, 14), dpi=150)
    corr_matrix = df[FEATURE_NAMES].corr()

    sns.heatmap(
        corr_matrix,
        cmap="vlag",
        center=0,
        vmin=-0.8,
        vmax=1.0,
        xticklabels=FEATURE_NAMES,
        yticklabels=FEATURE_NAMES,
        cbar_kws={"label": "Correlation Coefficient"},
    )
    plt.title("CricketVision AI: Biomechanical Feature Correlation Matrix (51 Features)", fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=90, fontsize=6.5)
    plt.yticks(rotation=0, fontsize=6.5)
    plt.tight_layout()

    corr_path = PERF_DIR / "eda_correlation.png"
    plt.savefig(corr_path, dpi=150)
    plt.close()
    print(f"  -> Saved {corr_path} (Features: {len(FEATURE_NAMES)}x{len(FEATURE_NAMES)})")

    # ─────────────────────────────────────────────────────────────────────────
    # Diff / summary output
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PERFORMANCE ASSET REGENERATION SUMMARY")
    print("=" * 65)
    print("Refreshed Assets:")
    print("  • eval_confusion_matrix_pct.png: 6 classes (row-normalized %)")
    print("  • eval_per_class_f1.png         : 6 classes (post-threshold F1)")
    print("  • feature_importance.png        : Top 20 features from 51 FEATURE_NAMES")
    print("  • eda_class_distribution.png   : 6 Shot Classes distribution")
    print("  • confusion_matrix.png          : 6 classes raw counts (was stale 7-class)")
    print("  • eda_boxplots.png              : 6 classes across 6 angles (was stale 7-class)")
    print("  • eda_correlation.png           : 51 current features (was stale 57-feature with vel_*)")
    print("=" * 65)


if __name__ == "__main__":
    main()
