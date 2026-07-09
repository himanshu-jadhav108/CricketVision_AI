# API Reference

CricketVision AI exposes a FastAPI service in [`api.py`](../api.py).

## Quick Start

```bash
python api.py
```

- **Base URL:** `http://localhost:8001`
- **Interactive docs:** `http://localhost:8001/docs`
- **Alternative:** `uvicorn api:app --host 0.0.0.0 --port 8001`

## Endpoints

### `GET /` and `GET /health`

Health check and service metadata.

**Response (200):**

```json
{
  "status": "online",
  "model_ready": true,
  "max_upload_mb": 10,
  "confidence_threshold": 0.65,
  "description": "CricketVision AI API. Issue a POST request to /predict with an image file."
}
```

### `POST /predict`

Classify a cricket batting image.

**Request:**

- Content-Type: `multipart/form-data`
- Field: `file` (image/jpeg, image/png, etc.)

**Success response (200):**

```json
{
  "prediction": "cover_drive",
  "confidence": 0.9234,
  "all_probabilities": {
    "cover_drive": 0.9234,
    "cut_shot": 0.0123,
    "leg_glance_shot": 0.0081,
    "pull_shot": 0.0312,
    "scoop_shot": 0.0045,
    "straight_drive": 0.0110,
    "sweep_shot": 0.0095
  }
}
```

When confidence is below the threshold, `prediction` will be `"Uncertain Shot"`.

**Error responses:**

| Status | Condition |
| :--- | :--- |
| 400 | Non-image upload, corrupted image, or pose not detected |
| 413 | File exceeds `CRICKET_MAX_UPLOAD_MB` |
| 500 | Unexpected pipeline failure |
| 503 | Predictor not initialised |

## Example (curl)

```bash
curl -X POST "http://localhost:8001/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@assets/images/examples/cover_drive.jpg"
```

## Example (Python)

```python
import requests

with open("assets/images/examples/cover_drive.jpg", "rb") as image_file:
    response = requests.post(
        "http://localhost:8001/predict",
        files={"file": ("cover_drive.jpg", image_file, "image/jpeg")},
    )

print(response.json())
```

## Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `CRICKET_API_HOST` | `0.0.0.0` | Bind address |
| `CRICKET_API_PORT` | `8001` | Listen port |
| `CRICKET_MAX_UPLOAD_MB` | `10` | Maximum upload size |
| `CRICKET_LOG_LEVEL` | `INFO` | Logging verbosity |
| `CRICKET_CONFIDENCE_THRESHOLD` | `0.65` | Rejection threshold |

## Startup Validation

On startup, the API verifies that required artifacts exist in `models/`:

- `xgboost_v1.pkl`
- `scaler.pkl`
- `label_encoder.pkl`

If any are missing, the service fails to start with a clear error message.
