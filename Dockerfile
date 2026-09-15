FROM python:3.11-slim

LABEL org.opencontainers.image.title="CricketVision AI" \
      org.opencontainers.image.description="Cricket shot classification using MediaPipe Pose and XGBoost"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CRICKET_APP_HOST=0.0.0.0 \
    CRICKET_APP_PORT=7860 \
    CRICKET_APP_SHARE=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

EXPOSE 7860 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from config import model_files_ready; import sys; sys.exit(0 if model_files_ready() else 1)"

CMD ["sh", "-c", "export CRICKET_APP_PORT=${PORT:-${CRICKET_APP_PORT:-7860}} && python app.py"]
