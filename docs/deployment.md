# Deployment

Step-by-step guide for running CricketVision AI locally, in Docker, and on Render.

For a printable checklist, see [`finaldeploy.md`](../finaldeploy.md).

## Prerequisites

- Python 3.11+
- Git
- (Optional) Docker
- Model artifacts in [`models/`](../models/):
  - `xgboost_v1.pkl`
  - `scaler.pkl`
  - `label_encoder.pkl`
  - `selector.pkl` (optional at runtime)

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/himanshu-jadhav108/CricketVision_AI.git
cd CricketVision_AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment (optional)

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `CRICKET_APP_HOST` | `0.0.0.0` | Gradio bind address |
| `CRICKET_APP_PORT` | `7860` | Gradio port |
| `CRICKET_APP_SHARE` | `false` | Enable Gradio public link |
| `CRICKET_API_HOST` | `0.0.0.0` | API bind address |
| `CRICKET_API_PORT` | `8001` | API port |
| `CRICKET_MAX_UPLOAD_MB` | `10` | Upload limit |
| `CRICKET_CONFIDENCE_THRESHOLD` | `0.65` | Prediction guardrail |

### 5. Run the Gradio demo

```bash
python app.py
```

Open `http://localhost:7860`.

### 6. Run the API

```bash
python api.py
```

Open `http://localhost:8001/docs`.

## Docker

### Build

```bash
docker build -t cricketvision-ai .
```

### Run (UI)

```bash
docker run --rm -p 7860:7860 cricketvision-ai
```

### Run (API)

```bash
docker run --rm -p 8001:8001 cricketvision-ai python api.py
```

The Dockerfile includes a health check that verifies model artifacts are present.

## Render Deployment

### Gradio Web Service (recommended for demos)

| Setting | Value |
| :--- | :--- |
| Environment | Python |
| Build command | `pip install -r requirements.txt` |
| Start command | `python app.py` |
| Port | `7860` |

Environment variables:

```
CRICKET_APP_HOST=0.0.0.0
CRICKET_APP_PORT=7860
CRICKET_APP_SHARE=false
```

### FastAPI Web Service

| Setting | Value |
| :--- | :--- |
| Build command | `pip install -r requirements.txt` |
| Start command | `python api.py` |
| Port | `8001` |

Environment variables:

```
CRICKET_API_HOST=0.0.0.0
CRICKET_API_PORT=8001
```

## Verification Checklist

- [ ] App starts without missing-artifact errors
- [ ] Sample image upload returns a prediction
- [ ] `GET /health` reports `model_ready: true`
- [ ] Oversized uploads return HTTP 413
- [ ] Invalid files return HTTP 400
- [ ] Gradio examples load from `assets/images/examples/`

## Troubleshooting

| Issue | Fix |
| :--- | :--- |
| `CRITICAL: models/ files missing` | Ensure all `.pkl` files are in `models/` |
| OpenCV / libGL errors | Install system libs or use Docker |
| MediaPipe install failure | Use Python 3.11 and `pip install -r requirements.txt` |
| Port already in use | Change `CRICKET_APP_PORT` or `CRICKET_API_PORT` |
| Low confidence on all images | Use clear, full-body side-on batting photos |

## Production Notes

- Keep `CRICKET_APP_SHARE=false` unless you explicitly need a temporary public Gradio link.
- Do not commit `.env` files with secrets.
- Evaluation plots and reports belong in `reports/` and `assets/performance/`, not in `models/`.
