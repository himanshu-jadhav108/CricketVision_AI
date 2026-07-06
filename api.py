from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from config import (
    API_HOST,
    API_PORT,
    CONFIDENCE_THRESHOLD,
    ENCODER_PATH,
    LOG_LEVEL,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
    MODEL_DIR,
    MODEL_PATH,
    PROJECT_NAME,
    SCALER_PATH,
    missing_model_files,
)
from predictor import ShotPredictor

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_predictor() -> ShotPredictor:
    return ShotPredictor(
        model_path=str(MODEL_PATH),
        scaler_path=str(SCALER_PATH),
        encoder_path=str(ENCODER_PATH),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = missing_model_files()
    if missing:
        raise RuntimeError(
            "Missing required model artifacts: " + ", ".join(path.name for path in missing)
        )

    app.state.predictor = load_predictor()
    logger.info("FastAPI predictor initialised from %s", MODEL_DIR)
    yield

    predictor = getattr(app.state, "predictor", None)
    if predictor is not None:
        predictor.close()


app = FastAPI(
    title=f"{PROJECT_NAME} API",
    description=f"Production endpoint for {PROJECT_NAME} — cricket shot classification with a confidence guardrail.",
    version="2.0",
    lifespan=lifespan,
)


def get_predictor() -> ShotPredictor:
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Backend predictor is not ready.")
    return predictor


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_ready": getattr(app.state, "predictor", None) is not None,
        "max_upload_mb": MAX_UPLOAD_MB,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "description": f"{PROJECT_NAME} API. Issue a POST request to /predict with an image file.",
    }


@app.post("/predict")
async def predict_shot(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file format. Upload an image.")

    image_data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large. Maximum upload size is {MAX_UPLOAD_MB} MB.",
        )

    try:
        with Image.open(io.BytesIO(image_data)) as uploaded_image:
            uploaded_image.verify()

        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        result = get_predictor().predict(image)

        if not result.get('success', False):
            raise HTTPException(status_code=400, detail=result.get('error', 'Pose detection failed.'))

        return {
            "prediction": result['label'],
            "confidence": round(result['confidence'], 4),
            "all_probabilities": {k: round(v, 4) for k, v in result.get('all_probs', {}).items()}
        }

    except HTTPException:
        raise
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")
    except Exception:
        logger.exception("Prediction pipeline error")
        raise HTTPException(status_code=500, detail="Prediction pipeline error.")

if __name__ == "__main__":
    import uvicorn
    display_host = "127.0.0.1" if API_HOST == "0.0.0.0" else API_HOST
    logger.info("Launching FastAPI on http://%s:%s/docs", display_host, API_PORT)
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=False)
