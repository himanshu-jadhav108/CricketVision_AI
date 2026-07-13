"""Shared runtime configuration for CricketVision AI."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# ── Project identity ────────────────────────────────────────────────────────
PROJECT_NAME = "CricketVision AI"
GITHUB_REPO_SLUG = "CricketVision_AI"
DOCKER_IMAGE_NAME = "cricketvision-ai"

# ── Directories ─────────────────────────────────────────────────────────────
MODEL_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo" / "CricketVision-AI-Logo.png"
EXAMPLES_DIR = ASSETS_DIR / "images" / "examples"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

# ── Model artifacts (runtime) ───────────────────────────────────────────────
MODEL_PATH = MODEL_DIR / "xgboost_v1.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
SELECTOR_PATH = MODEL_DIR / "selector.pkl"

REQUIRED_MODEL_FILES = (MODEL_PATH, SCALER_PATH, ENCODER_PATH)

# ── Inference ─────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = float(os.getenv("CRICKET_CONFIDENCE_THRESHOLD", "0.65"))
POSE_MIN_DETECTION_CONFIDENCE = float(
    os.getenv("CRICKET_POSE_MIN_DETECTION_CONFIDENCE", "0.05")
)

# ── Gradio UI ─────────────────────────────────────────────────────────────────
GRADIO_HOST = os.getenv("CRICKET_APP_HOST", "0.0.0.0")
GRADIO_PORT = int(os.getenv("CRICKET_APP_PORT", "7860"))
GRADIO_SHARE = os.getenv("CRICKET_APP_SHARE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# ── FastAPI ───────────────────────────────────────────────────────────────────
API_HOST = os.getenv("CRICKET_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("CRICKET_API_PORT", "8001"))

# ── Upload & logging ──────────────────────────────────────────────────────────
MAX_UPLOAD_MB = int(os.getenv("CRICKET_MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
LOG_LEVEL = os.getenv("CRICKET_LOG_LEVEL", "INFO").upper()


def missing_model_files() -> list[Path]:
    """Return paths for required model artifacts that are not on disk."""
    return [path for path in REQUIRED_MODEL_FILES if not path.exists()]


def model_files_ready() -> bool:
    """True when all required model artifacts are present."""
    return not missing_model_files()
