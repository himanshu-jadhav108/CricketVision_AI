# Model Training & Evaluation Methodology

This document details the development, training, and evaluation methodology of the production **CricketVision AI** model. It captures the empirical progression reflected in the repository artifacts and evaluation reports.

---

## 1. Objective

Classify cricket batting imagery into six professional shot classes using 51 scale-invariant 3D biomechanical features extracted from MediaPipe Pose landmarks, ensuring robust CPU inference and complete invariance to background visual textures (such as stadium ads, pitch grass, and team jerseys).

---

## 2. Target Classes

| Label | Shot Class | Kinematic Characteristics | Active Samples |
| :--- | :--- | :--- | :---: |
| `cover_drive` | Cover Drive | Front-foot drive through the off-side; extended front knee and high elbow | 1,426 |
| `pull_shot` | Pull Shot | Cross-bat hook/pull to short-pitched deliveries; horizontal bat arc and open chest | 1,491 |
| `cut_shot` | Cut Shot | Back-foot square cut through point; elevated wrists and weight back | 391 |
| `sweep_shot` | Sweep Shot | Front-knee knelt paddle on leg side; low center of gravity and horizontal bat | 1,026 |
| `scoop_shot` | Scoop Shot | Unorthodox ramp over the wicketkeeper; crouched torso and vertical bat lift | 487 |
| `leg_glance_shot` | Leg Glance | Subtle wrist deflection off hip/pads to fine leg; closed shoulder face | 1,503 |
| **Total Active** | | | **6,324** |

### Class Removal Rationale (`straight_drive`):
In early prototypes, a 7th class (`straight_drive`) was evaluated. During the July 2026 model audit (`archive_used/audit_results_utf8.txt`), `straight_drive` exhibited:
- Severe sample scarcity: only 345 sequences (5.17% of total dataset).
- Abysmal recall: **0.31** and F1-score of **0.42**.
- Heavy confusion: 16 out of 49 test samples were misclassified as `leg_glance_shot` due to near-identical frontal body alignment.

To preserve the technical integrity and precision of the coaching tool, `straight_drive` was deliberately removed in commit `bc65ec1`.

---

## 3. Dataset Preprocessing & Grouping

1. **MediaPipe Pose Extraction**: 33 keypoints $(x, y, z, \text{visibility})$ extracted per frame (`min_detection_confidence=0.05`). Frames with mean landmark visibility $< 0.50$ are discarded.
2. **Feature Engineering**: 51 continuous static features computed via `features/feature_builder.py`.
3. **Source Grouping**: Each sample carries a `source_group` identifier derived from the source sequence/image prefix (e.g. `97` for `97_cut_brightness.jpg`, `97_cut_shear.jpg`).
4. **Data Normalization**: `StandardScaler` is fitted strictly on the training partition and applied to test/validation splits to prevent scaling leakage.

---

## 4. The Three Distinct Evaluation Stages

To understand the system's evaluation metrics, three distinct experimental stages must be separated:

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Baseline Random Forest Audit (July 2026)                        │
│ • Evaluated 5-fold GroupKFold by source_group on legacy 7-class RF       │
│ • Diagnosed the false 100% test accuracy caused by global synthetic jitter │
│ • Established genuine baseline: CV Mean = 78.13% (+/- 1.30%), F1 = 0.68  │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: XGBoost Retraining & Tuning (retrain.py)                       │
│ • Excluded straight_drive; stripped 6 velocity feats -> 51 static feats  │
│ • Applied sample_weight='balanced' for class imbalance                   │
│ • RandomizedSearchCV (30 configs, cv=3 StratifiedKFold on train set)     │
│ • Selected: max_depth=6, lr=0.05, n_estimators=150, subsample=0.8        │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Final Production Holdout Evaluation (evaluate_model.py)         │
│ • GroupShuffleSplit (80/20 train/test by source_group, 1,304 test frames)│
│ • Raw Generalization Accuracy: 77.53% (Macro F1 = ~0.78)                 │
│ • Post-Threshold (0.65) Selective Accuracy: 92.52% (Macro F1 = 0.90)     │
│ • Coverage: 67.64% (Rejection Rate: 32.36%)                             │
└──────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: The Leakage Discovery & GroupKFold Baseline
An early baseline pipeline (`05_modelling.py`) added Gaussian noise copies across the entire dataset before performing a random stratified train/test split. Because noisy copies of the exact same image appeared in both splits, the model achieved an artificial 100% test accuracy.

The team ran a formal audit (`archive_used/notebooks/08_model_audit.py`, outputs preserved in `archive_used/audit_results_utf8.txt`), implementing **5-fold `GroupKFold`** grouped strictly on `source_group`. This audit revealed:
- **Training Accuracy**: 99.98%
- **Cross-Validation Mean Accuracy**: **78.13% (+/- 1.30%)**
- **CV Macro F1**: **0.68**
- **Unseen Test Accuracy**: **77.52%**

This exposed a ~22% generalization gap in the unregularized Random Forest and proved that naive frame-level splits produce invalid metrics.

### Stage 2: XGBoost Retraining (`retrain.py`)
To prevent memorization, the team transitioned to **Extreme Gradient Boosting (XGBoost)** with strong tree regularization:
- **Class Imbalance**: Managed using `sklearn.utils.class_weight.compute_sample_weight(class_weight='balanced', y=y_train)` passed directly into the loss function.
- **Hyperparameter Optimization**: `RandomizedSearchCV` exploring 30 candidate configurations with `cv=3` (Stratified 3-fold cross-validation on the training set) scoring on `f1_macro`.
- **Selected Hyperparameters**:
  ```python
  XGBClassifier(
      max_depth=6,
      learning_rate=0.05,
      n_estimators=150,
      subsample=0.8,
      colsample_bytree=0.7,
      min_child_weight=2,
      gamma=0,
      objective='multi:softprob',
      eval_metric='mlogloss',
      seed=42
  )
  ```

### Stage 3: Production Holdout Evaluation (`evaluate_model.py`)
The production model was evaluated on a strict 20% holdout split created using `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` on `source_group`:
- **Total Test Samples**: **1,304** (0% group overlap with the training set).
- **Raw Generalization Accuracy**: **77.53%** (1,011 / 1,304 correct).

---

## 5. Confidence-Aware Selective Prediction

In sports coaching technology, false positive classifications degrade athlete trust. The system deploys a **0.65 confidence guardrail**: if the winning softmax probability is below 0.65, the system abstains and outputs `"Uncertain Shot"`.

```text
Input Test Set: 1,304 samples
   ├── Confidence >= 0.65: 882 samples accepted (67.64% coverage) ──► 92.52% Accuracy (816 / 882 correct)
   └── Confidence <  0.65: 422 samples rejected (32.36% rejection)──► Labeled "Uncertain Shot"
```

### Threshold Sensitivity Sweep (`reports/threshold_sweep.csv`)

| Threshold | Accepted | Rejected (%) | Selective Accuracy | Coverage |
| :---: | :---: | :---: | :---: | :---: |
| 0.50 | 1,041 | 20.17% | 87.42% | 79.83% |
| 0.55 | 989 | 24.16% | 89.28% | 75.84% |
| 0.60 | 938 | 28.07% | 91.04% | 71.93% |
| **0.65 (Default)** | **882** | **32.36%** | **92.52%** | **67.64%** |
| 0.70 | 812 | 37.73% | 94.09% | 62.27% |
| 0.75 | 733 | 43.79% | 95.63% | 56.21% |
| 0.80 | 668 | 48.77% | 96.71% | 51.23% |

### Per-Class Performance (Post-Threshold @ 0.65)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `cover_drive` | 0.95 | 0.94 | **0.95** | 190 |
| `cut_shot` | 0.76 | 0.72 | **0.74** | 40 |
| `leg_glance_shot` | 0.93 | 0.90 | **0.92** | 200 |
| `pull_shot` | 0.95 | 0.94 | **0.94** | 222 |
| `scoop_shot` | 0.79 | 0.92 | **0.85** | 75 |
| `sweep_shot` | 0.97 | 0.97 | **0.97** | 155 |
| **Macro Average** | **0.89** | **0.90** | **0.90** | **882** |
| **Weighted Average** | **0.93** | **0.93** | **0.93** | **882** |

---

## 6. Production Artifacts

| File | Type | Description |
| :--- | :--- | :--- |
| `models/xgboost_v1.pkl` | `XGBClassifier` | 6-class production classifier trained on 51 static features |
| `models/scaler.pkl` | `StandardScaler` | Feature normalizer fitted on training split (51 features) |
| `models/label_encoder.pkl` | `LabelEncoder` | String label to integer index mapping |
| `reports/evaluation.md` | Markdown | Canonical quantitative test evaluation report |
| `reports/confusion_matrix.csv`| CSV | 6×6 post-threshold confusion matrix |
| `reports/threshold_sweep.csv` | CSV | Sensitivity trade-off data from 0.50 to 0.80 |
