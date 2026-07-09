# CricketVision AI — Final Deployment Guide

Complete step-by-step guide for deploying CricketVision AI from scratch. Assumes no prior familiarity with the project.

---

## Table of Contents

1. [What You Are Deploying](#1-what-you-are-deploying)
2. [Prerequisites](#2-prerequisites)
3. [Local Setup (Virtual Environment)](#3-local-setup-virtual-environment)
4. [Verify Model Artifacts](#4-verify-model-artifacts)
5. [Run the Gradio Demo](#5-run-the-gradio-demo)
6. [Run the FastAPI Service](#6-run-the-fastapi-service)
7. [Docker Deployment](#7-docker-deployment)
8. [GitHub Preparation](#8-github-preparation)
9. [Render Deployment](#9-render-deployment)
10. [Environment Variables Reference](#10-environment-variables-reference)
11. [Verification Checklist](#11-verification-checklist)
12. [Troubleshooting](#12-troubleshooting)
13. [Future Improvements](#13-future-improvements)

---

## 1. What You Are Deploying

CricketVision AI classifies cricket batting shots from images using:

- **MediaPipe Pose** — skeleton extraction
- **Feature engineering** — 60+ biomechanical features
- **XGBoost** — production classifier (`models/xgboost_v1.pkl`)
- **Confidence guardrail** — rejects uncertain predictions

Two entry points share the same inference logic:

| Service | File | Default Port |
| :--- | :--- | :--- |
| Gradio UI | `app.py` | 7860 |
| FastAPI API | `api.py` | 8001 |

---

## 2. Prerequisites

Install before starting:

- **Git** — clone the repository
- **Python 3.11+** — runtime
- **pip** — package manager
- **(Optional) Docker** — containerised deployment

System libraries for OpenCV (if not using Docker):

- Linux: `libgl1`, `libglib2.0-0`
- macOS: usually works out of the box
- Windows: pip wheels typically include required binaries

---

## 3. Local Setup (Virtual Environment)

### Step 1 — Clone

```bash
git clone https://github.com/himanshu-jadhav108/CricketVision-AI.git
cd CricketVision-AI
```

### Step 2 — Create virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs MediaPipe, XGBoost, FastAPI, Gradio, OpenCV, and supporting libraries.

### Step 4 — Configure environment (optional)

```bash
cp .env.example .env
```

Edit `.env` if you need custom ports or thresholds. Defaults work for local development.

---

## 4. Verify Model Artifacts

Before starting any service, confirm these files exist in `models/`:

```
models/
├── xgboost_v1.pkl       # Required — trained classifier
├── scaler.pkl           # Required — feature scaler
├── label_encoder.pkl    # Required — class labels
└── selector.pkl         # Optional — feature selector
```

Quick check:

```bash
python -c "from config import model_files_ready; print('Ready:', model_files_ready())"
```

Expected output: `Ready: True`

If `False`, restore the missing `.pkl` files before continuing.

---

## 5. Run the Gradio Demo

```bash
python app.py
```

Expected log output:

```
INFO ... Loading predictor from .../models
INFO ... Launching Gradio UI on 0.0.0.0:7860
```

Open **http://localhost:7860** in your browser.

### Test the demo

1. Click a sample image under "Sample Biomechanical Tests"
2. Click **ANALYZE BIOMECHANICS**
3. Verify a shot label and confidence gauge appear
4. Verify the pose skeleton overlay renders

---

## 6. Run the FastAPI Service

In a **separate terminal** (with the same virtual environment):

```bash
python api.py
```

Open **http://localhost:8001/docs** for interactive API documentation.

### Test with curl

```bash
curl -X POST "http://localhost:8001/predict" \
  -F "file=@assets/images/examples/cover_drive.jpg"
```

### Test health endpoint

```bash
curl http://localhost:8001/health
```

Expected:

```json
{
  "status": "online",
  "model_ready": true,
  "max_upload_mb": 10,
  "confidence_threshold": 0.65
}
```

---

## 7. Docker Deployment

### Build the image

```bash
docker build -t cricketvision-ai .
```

### Run Gradio (default)

```bash
docker run --rm -p 7860:7860 cricketvision-ai
```

Open **http://localhost:7860**

### Run FastAPI

```bash
docker run --rm -p 8001:8001 cricketvision-ai python api.py
```

Open **http://localhost:8001/docs**

### Docker health check

The Dockerfile includes a health check that verifies model artifacts are present inside the container.

---

## 8. GitHub Preparation

Before making the repository public:

1. **Repository name** — Use `CricketVision-AI` (display name: **CricketVision AI**)
2. **Description** — "Cricket shot classification using MediaPipe Pose + XGBoost"
3. **Topics** — `machine-learning`, `cricket`, `mediapipe`, `xgboost`, `computer-vision`, `fastapi`, `gradio`
4. **README** — Already configured with badges, architecture, and setup
5. **License** — MIT (`LICENSE` file included)
6. **Screenshots** — Capture the Gradio UI and save to `assets/screenshots/`
7. **Secrets** — Never commit `.env` files

### Push to GitHub

```bash
git add .
git commit -m "Prepare CricketVision AI for public release"
git remote add origin https://github.com/himanshu-jadhav108/CricketVision-AI.git
git push -u origin main
```

---

## 9. Render Deployment

### Option A — Gradio Demo (recommended for portfolio)

1. Create a **New Web Service** on [Render](https://render.com)
2. Connect your GitHub repository
3. Configure:

| Setting | Value |
| :--- | :--- |
| Environment | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python app.py` |

4. Add environment variables:

```
CRICKET_APP_HOST=0.0.0.0
CRICKET_APP_PORT=7860
CRICKET_APP_SHARE=false
CRICKET_LOG_LEVEL=INFO
```

5. Set the port to **7860** in Render's service settings
6. Deploy and open the generated URL

### Option B — FastAPI Service

| Setting | Value |
| :--- | :--- |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python api.py` |

Environment variables:

```
CRICKET_API_HOST=0.0.0.0
CRICKET_API_PORT=8001
```

Set the port to **8001**.

### Render tips

- Free tier may sleep after inactivity — first request may be slow
- Ensure model artifacts are committed to `models/` (they are required at startup)
- Do not set `CRICKET_APP_SHARE=true` on Render — use Render's public URL instead

---

## 10. Environment Variables Reference

| Variable | Default | Used By | Description |
| :--- | :--- | :--- | :--- |
| `CRICKET_APP_HOST` | `0.0.0.0` | Gradio | Bind address |
| `CRICKET_APP_PORT` | `7860` | Gradio | Listen port |
| `CRICKET_APP_SHARE` | `false` | Gradio | Temporary public Gradio link |
| `CRICKET_API_HOST` | `0.0.0.0` | API | Bind address |
| `CRICKET_API_PORT` | `8001` | API | Listen port |
| `CRICKET_MAX_UPLOAD_MB` | `10` | API | Max upload size |
| `CRICKET_CONFIDENCE_THRESHOLD` | `0.65` | Predictor | Rejection threshold |
| `CRICKET_POSE_MIN_DETECTION_CONFIDENCE` | `0.05` | Pose | MediaPipe detection threshold |
| `CRICKET_LOG_LEVEL` | `INFO` | All | Logging verbosity |

---

## 11. Verification Checklist

Run through this before sharing publicly:

- [ ] `python -c "from config import model_files_ready; assert model_files_ready()"`
- [ ] `python app.py` — UI loads at localhost:7860
- [ ] Sample examples appear in the Gradio interface
- [ ] Upload returns a prediction with confidence gauge
- [ ] `python api.py` — docs load at localhost:8001/docs
- [ ] `curl /health` returns `model_ready: true`
- [ ] `curl -X POST /predict` with a valid image returns JSON
- [ ] Invalid file upload returns HTTP 400
- [ ] Oversized upload returns HTTP 413
- [ ] `python -m unittest discover -s tests -v` passes
- [ ] Docker build and run succeeds
- [ ] README links and image paths resolve correctly

---

## 12. Troubleshooting

### "CRITICAL: models/ files missing"

**Cause:** Required `.pkl` files are not in `models/`.  
**Fix:** Restore `xgboost_v1.pkl`, `scaler.pkl`, and `label_encoder.pkl` to `models/`.

### MediaPipe install error

**Cause:** Incompatible Python version or outdated pip.  
**Fix:** Use Python 3.11+, run `pip install --upgrade pip`, then reinstall requirements.

### OpenCV `libGL` error (Linux)

**Cause:** Missing system libraries.  
**Fix:** `sudo apt-get install libgl1 libglib2.0-0` or use Docker.

### "No pose detected"

**Cause:** Image lacks a visible full-body batting pose.  
**Fix:** Use a clearer side-on or 45° batting photo with head-to-toe visibility.

### Port already in use

**Cause:** Another process occupies 7860 or 8001.  
**Fix:** Set `CRICKET_APP_PORT=7861` or `CRICKET_API_PORT=8002` in `.env`.

### Gradio examples not showing

**Cause:** Example images missing from `assets/images/examples/`.  
**Fix:** Verify the seven sample `.jpg` files exist in that directory.

### All predictions are "Uncertain Shot"

**Cause:** Low model confidence on the input images.  
**Fix:** Use higher-quality batting photos. Threshold is configurable via `CRICKET_CONFIDENCE_THRESHOLD`.

### Render deploy fails at startup

**Cause:** Model artifacts not in the deployed branch.  
**Fix:** Ensure `models/*.pkl` are committed and pushed to the branch Render builds from.

---

## 13. Future Improvements

Recommended next steps after the initial public release:

- [ ] Add a Gradio UI screenshot to `assets/screenshots/`
- [ ] Add a project logo to `assets/logo/`
- [ ] Set up GitHub Actions CI for automated tests
- [ ] Publish a Hugging Face Spaces demo
- [ ] Add a formal model card (`docs/model_card.md`)
- [ ] Record a short demo GIF for the README
- [ ] Enable video-sequence inference with temporal models

---

## Quick Reference

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run
python app.py          # UI  → http://localhost:7860
python api.py          # API → http://localhost:8001/docs

# Test
python -m unittest discover -s tests -v

# Docker
docker build -t cricketvision-ai .
docker run --rm -p 7860:7860 cricketvision-ai
```

For architecture and API details, see the [docs/](docs/) directory.
