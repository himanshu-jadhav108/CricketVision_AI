# Training

This document describes how the production XGBoost model was trained. Training notebooks and scripts are not part of the runtime repository; this page captures the methodology reflected in the shipped artifacts and evaluation reports.

## Objective

Classify cricket batting images into six professional shot types using engineered pose features rather than raw pixels.

## Target Classes

| Label | Description |
| :--- | :--- |
| `cover_drive` | Front-foot off-side drive |
| `pull_shot` | Aggressive short-ball hook/pull |
| `cut_shot` | Back-foot cut through point |
| `sweep_shot` | Front-foot sweep on the leg side |
| `scoop_shot` | Unorthodox scoop over the keeper |
| `leg_glance_shot` | Deflection to the leg side |

## Dataset Preparation

1. **Sources** — Curated cricket coaching imagery and video frames (see [dataset.md](dataset.md)).
2. **Pose extraction** — MediaPipe Pose applied to each frame.
3. **Feature engineering** — Same logic as [`features/feature_builder.py`](../features/feature_builder.py).
4. **Leakage control** — `GroupShuffleSplit` / `GroupKFold` by video ID so frames from the same clip never appear in both train and test splits.

Processed feature tables live under [`data/features/`](../data/features/) for reproducibility reference.

## Model Selection

| Candidate | Outcome |
| :--- | :--- |
| Random Forest (V1) | Rejected — inflated accuracy from frame-level leakage |
| CNN / ResNet | Rejected — insufficient balanced image data, less interpretable |
| LSTM / RNN | Deferred — needs richer sequential data |
| **XGBoost (V2)** | **Selected** — strong tabular performance, robust to noise |

## Training Configuration

- **Algorithm** — XGBoost gradient boosted trees
- **Hyperparameter search** — `RandomizedSearchCV` with 5-fold `GroupKFold`
- **Regularisation** — Restricted `max_depth` (6–8), tuned `min_child_weight`
- **Class weighting** — Balanced weights for minority classes (e.g. straight drive)
- **Feature selection** — Optional `selector.pkl` for pruned feature subsets

## Evaluation Strategy

The project moved from a flawed frame-level split (V1) to a rigorous video-level split (V2).

| Metric | V1 (Baseline) | V2 (Production) |
| :--- | :--- | :--- |
| Test accuracy | 100% (leaked) | **77.5%** |
| Cross-validation | 70.4% | **78.2%** |
| Generalisation gap | ~30% | **~1%** |

Detailed per-class metrics are in [`reports/evaluation.md`](../reports/evaluation.md).

## Production Artifacts

After training, the following files are exported to [`models/`](../models/):

| File | Purpose |
| :--- | :--- |
| `xgboost_v1.pkl` | Trained classifier |
| `scaler.pkl` | Fitted `StandardScaler` |
| `label_encoder.pkl` | Class label mapping |
| `selector.pkl` | Optional feature selector |

Evaluation outputs (confusion matrix, plots, reports) belong in [`reports/`](../reports/) and [`assets/performance/`](../assets/performance/), not in `models/`.

## Confidence Thresholding

At deployment time, predictions with `max(probability) < 0.65` are mapped to `"Uncertain Shot"`. This raises precision on accepted predictions to **92.52%** at the cost of rejecting ~32.4% of samples. See [`reports/evaluation.md`](../reports/evaluation.md).

## Reproducing Training (Manual)

Training notebooks were used during development but are excluded from the public runtime repo. To retrain:

1. Build a labelled pose-feature dataset using the same feature builder.
2. Split by video group, not by individual frame.
3. Tune XGBoost with group-aware cross-validation.
4. Export artifacts to `models/` and evaluation outputs to `reports/`.
