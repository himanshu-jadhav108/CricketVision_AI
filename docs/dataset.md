# Dataset Provenance & Specifications

CricketVision AI was trained on 3D biomechanical features extracted from cricket batting imagery. The repository ships processed feature tables under `data/features/` for complete reproducibility; raw image archives (~thousands of images) are excluded from the git index to keep repository clone times light and clean.

---

## 1. Sources & Data Collection

The dataset was curated from two primary complementary sources:

1. **Primary Dataset (Kaggle Baseline)**:
   * **Source**: `awais69/cricket-coaching-dataset` (downloaded via `kagglehub`).
   * **Content**: ~1,958 images across six initial classes: `cover_drive`, `pull_shot`, `cut_shot`, `leg_glance_shot`, `scoop_shot`, and `straight_drive`.
   * **Inherent Augmentation**: Source images in this repository collection were pre-augmented by the original author with spatial transformations (rotations at $\pm 30^\circ$, horizontal shear, brightness jitter, and salt-and-pepper noise), yielding multiple related variants per original batting pose (e.g. `97_cut_brightness.jpg`, `97_cut_shear.jpg`, `97_cut_30.jpg`).

2. **Supplementary Dataset (Class Expansion & Balance)**:
   * **Source**: Scraped & open cricket coaching collections ingested via `archive_used/scripts/data_ingestion.py`.
   * **Content**: Ingested folders mapped as follows:
     * `drive` → `cover_drive`
     * `legglance-flick` → `leg_glance_shot`
     * `pullshot` → `pull_shot`
     * `sweep` → `sweep_shot` (introducing the 7th class `sweep_shot` to the project).
   * **Prefix Assignment**: Images received programmatic identifier prefixes starting at `5000` (e.g. `5116_cover_drive.png`).

---

## 2. Pose Extraction Pipeline

Raw images were processed through MediaPipe Pose:
* **Model Configuration**: `model_complexity=2` (Heavy/high-accuracy landmark model), `min_detection_confidence=0.05`.
* **Output Coordinates**: 33 anatomical landmarks, each providing:
  * Normalized image plane coordinates $(x, y) \in [0, 1]$
  * Depth coordinate $z$ (relative to the midpoint of the hips)
  * Landmark visibility score $\in [0, 1]$
* **Quality Filtering**: Poses where average landmark visibility fell below $0.50$ (`MIN_VISIBILITY = 0.5`) were discarded.

---

## 3. Processed Feature Tables

The extracted feature representations live under [`data/features/`](../data/features/):

| File | Rows | Columns | Purpose |
| :--- | :---: | :---: | :--- |
| [`features_all.csv`](../data/features/features_all.csv) | **6,669** | **60** | **Canonical Master Dataset**: 51 static features + 6 legacy velocity fields + `label`, `filepath`, and `source_group`. |
| [`features_engineered.csv`](../data/features/features_engineered.csv) | 1,958 | 40 | Phase 1 dataset extracted from the primary Kaggle collection. |
| [`features_augmented.csv`](../data/features/features_augmented.csv) | 7,832 | 49 | Experimental 4× dataset with synthetic Gaussian noise (used in early baseline experiments). |

---

## 4. Class Distribution & Production Scope

| Class Label | Common Name | Total Samples in `features_all.csv` | Status in Production |
| :--- | :--- | :---: | :---: |
| `leg_glance_shot` | Leg Glance | 1,503 | **Active** |
| `pull_shot` | Pull Shot | 1,491 | **Active** |
| `cover_drive` | Cover Drive | 1,426 | **Active** |
| `sweep_shot` | Sweep Shot | 1,026 | **Active** |
| `scoop_shot` | Scoop Shot | 487 | **Active** |
| `cut_shot` | Cut Shot | 391 | **Active** |
| `straight_drive` | Straight Drive | 345 | **Removed** (Scarcity & high confusion with leg glance) |
| **Total Active Training Set** | | **6,324** | |

### Why `straight_drive` Was Excluded:
During the July 2026 ML audit (`archive_used/audit_results_utf8.txt`), `straight_drive` was identified as a critical point of failure:
- **Sample Scarcity**: Only 345 sequences (5.17% of total data).
- **Poor Performance**: Recall was only **0.31**, and F1-score was **0.42**.
- **Confusion**: Out of 49 test samples, 16 were misclassified as `leg_glance_shot` because both strokes feature nearly collinear chest alignments facing the camera.
Filtering out `straight_drive` raised overall model Macro F1 from 0.73 to **0.90** post-threshold.

---

## 5. Feature Schema: 51 Static Biomechanical Features

Features are computed deterministically in [`features/feature_builder.py`](../features/feature_builder.py). All limb lengths are normalized by the 3D torso length (distance between mid-shoulder and mid-hip), ensuring complete camera distance invariance.

1. **3D Joint Angles (8)**: Elbow, shoulder, knee, and hip angles in degrees $[0, 180]$.
2. **Spine Alignment (1)**: `trunk_lean` (vector angle of spine vs vertical).
3. **Torso-Normalized Distances (8)**: Wrist-to-hip, wrist-to-head, wrist spread, ankle spread, hip-to-ankle.
4. **Body Position Ratios (4)**: Vertical wrist elevation ($Y$) and knee bend ($Y_{\text{knee}} - Y_{\text{hip}}$).
5. **Lateral Symmetries (2)**: Wrist horizontal offset and shoulder horizontal offset.
6. **Keypoint Visibility (3)**: Data quality indicators for wrists and elbows.
7. **Biomechanical Ratios (5)**: Elbow extension mean, knee flexion difference, hip-to-shoulder ratio, mean wrist height, stance width index.
8. **3D Depth Coordinates (6)**: $Z$-depth of wrists, 3D reach relative to hip midpoint, shoulder tilt in $Z$-plane, 3D wrist spread.
9. **Centroid & Shoulder Offsets (7)**: Wrists relative to hip centroid ($X, Y, Z$) and shoulder centroid ($X, Y$).
10. **Arm Extension Ratios (2)**: 3D wrist-to-shoulder reach relative to torso scale.
11. **Bat Vector Simulation & Alignment (5)**: Projected bat angle vs vertical, projected bat angle vs camera plane, shoulder alignment, hip alignment, torso twist.

> **Note on Temporal Features:** Early experimental versions included 6 velocity features (`vel_r_wrist_x/y`, `vel_bat_strike_x/y`). When testing on single still images, prior frames do not exist, forcing those fields to zero. In commit `93d435c`, these 6 fields were completely eliminated to guarantee 100% train-inference feature parity.

---

## 6. Grouped Splitting Strategy (Data Leakage Safeguard)

Batting footage often contains consecutive frames or augmented versions of the same original photograph. A standard random train/test split inadvertently places near-identical frames of the same batsman in both train and test partitions, inflating accuracy.

To guarantee zero leakage:
* Each row in `features_all.csv` includes a `source_group` column (**4,304 unique groups**).
* For augmented images (e.g. `97_cut_brightness.jpg`, `97_cut_shear.jpg`), the prefix `97` acts as the group key.
* Training scripts use `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` grouping on `source_group`.
* **Zero Group Overlap**: Verified by checking `len(set(groups_train) & set(groups_test)) == 0`.

---

## 7. Sample Inference Images

Six representative images representing each supported shot class are provided in [`assets/images/examples/`](../assets/images/examples/) for live web UI testing:
* `cover_drive.jpg`
* `pull_shot.jpg`
* `cut_shot.jpg`
* `sweep_shot.jpg`
* `scoop_shot.jpg`
* `leg_glance_shot.jpg`

---

## 8. Known Limitations

* **Severe Occlusion**: MediaPipe requires a visible upper body and legs. Low-angle boundary shots where the batsman's lower body is blocked by boundary ropes can cause landmark failure.
* **Camera Perspective**: Optimal classification occurs between $30^\circ$ and $90^\circ$ (side-on to 3/4 angle). Extreme top-down or straight-on perspectives compress $Z$-axis depth.
* **Class Imbalance**: `cut_shot` (391 samples) and `scoop_shot` (487 samples) have fewer samples than `cover_drive` (1,426) or `pull_shot` (1,491), resulting in slightly lower F1 scores (0.74 for cut shot vs 0.95 for cover drive).
* **Right-Hand Stance Assumption**: Feature calculations currently evaluate joint angles under right-handed batsman convention. Stance mirroring for left-handed batsmen is slated for post-hackathon development.
