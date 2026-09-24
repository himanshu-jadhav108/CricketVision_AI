# Architecture

CricketVision AI is a modular inference system that classifies cricket batting shots from still images using pose-based biomechanical features and a trained XGBoost classifier.

## High-Level Flow

```mermaid
flowchart TD
    %% Node Styling Definitions
    classDef client fill:#0f172a,stroke:#0284c7,stroke-width:2px,color:#f8fafc;
    classDef preproc fill:#1e1b4b,stroke:#6366f1,stroke-width:2px,color:#f8fafc;
    classDef ml fill:#1e293b,stroke:#8b5cf6,stroke-width:2px,color:#f8fafc;
    classDef gate fill:#291e0a,stroke:#f59e0b,stroke-width:2px,color:#fef3c7;
    classDef success fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#ecfdf5;
    classDef abstain fill:#450a0a,stroke:#ef4444,stroke-width:2px,color:#fee2e2;

    A[/"📷 Image Upload"\]:::client --> B["🦴 MediaPipe 3D Pose<br/><b>33 Skeletal Coordinates</b>"]:::preproc
    B --> C["📐 51 Feature Builder<br/><b>Biomechanical Angles & Distances</b>"]:::preproc
    C --> D["⚖️ StandardScaler<br/><b>Zero-Drift Normalization</b>"]:::ml
    D --> E["🌲 XGBoost Classifier<br/><b>6-Class Multi-Softmax Probabilities</b>"]:::ml
    E --> F{"🛡️ Confidence Gate<br/><b>P &ge; 0.65?</b>"}:::gate
    F -- "Yes" --> G["✅ Certified Shot Label<br/><b>92.52% Selective Accuracy</b>"]:::success
    F -- "No" --> H["⚠️ Uncertain Shot<br/><b>Coaching Trust Abstention</b>"]:::abstain
```

### 📊 High-Fidelity Data Flow Schematic
![CricketVision AI Architecture Diagram](../assets/diagrams/architecture_flow_diagram.png)

## Components

| Module | Role |
| :--- | :--- |
| [`app.py`](../app.py) | Gradio demo UI for interactive analysis |
| [`api.py`](../api.py) | FastAPI production endpoint |
| [`predictor.py`](../predictor.py) | End-to-end inference orchestration |
| [`config.py`](../config.py) | Centralised paths, thresholds, and environment settings |
| [`features/pose_extractor.py`](../features/pose_extractor.py) | MediaPipe Pose wrapper (33 landmarks) |
| [`features/feature_builder.py`](../features/feature_builder.py) | Biomechanical feature engineering (51 static features) |
| [`models/`](../models/) | Runtime model artifacts (`.pkl`) |

## Data Flow

1. **Input** — User uploads a cricket batting image (via Gradio UI or FastAPI).
2. **Pose Extraction** — MediaPipe Pose detects 33 body landmarks as `(x, y, z, visibility)`.
3. **Feature Engineering** — 51 static 3D biomechanical features are computed (joint angles, torso-scaled distances, Z-depth reach, and bat-vector angles).
4. **Preprocessing** — Features are standardized using the fitted `StandardScaler`.
5. **Classification** — XGBoost computes class probability distribution across six shot types.
6. **Guardrail** — Predictions below the 0.65 confidence threshold return `"Uncertain Shot"`.

## Deployment Topology

```mermaid
flowchart TD
    classDef client fill:#0f172a,stroke:#0284c7,stroke-width:2px,color:#f8fafc;
    classDef service fill:#1e1b4b,stroke:#6366f1,stroke-width:2px,color:#f8fafc;
    classDef storage fill:#1e293b,stroke:#8b5cf6,stroke-width:2px,color:#f8fafc;

    G["🌐 Gradio Web App<br/><b>app.py : 7860</b>"]:::client
    F["🚀 FastAPI REST Server<br/><b>api.py : 8001</b>"]:::client

    G --> P["⚙️ ShotPredictor Orchestrator<br/><b>predictor.py</b>"]:::service
    F --> P

    P --> M["📦 Serialized Models & Scalers<br/><b>xgboost_v1.pkl &bull; scaler.pkl &bull; label_encoder.pkl</b>"]:::storage
```

Both entry points share the same `ShotPredictor` class and read artifacts from `models/`.

## Design Principles

- **Single inference path** — UI and API use identical prediction logic.
- **Fail-fast startup** — Missing required artifacts prevent silent degradation.
- **Interpretable features** — Tabular biomechanics instead of black-box pixel models.
- **Confidence guardrail** — Low-confidence predictions are rejected rather than forced.

## Related Documentation

- [Training pipeline](training.md)
- [Dataset](dataset.md)
- [API reference](api.md)
- [Deployment guide](deployment.md)
