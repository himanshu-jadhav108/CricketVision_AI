from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import cv2
import joblib
import numpy as np
from PIL import Image

from config import CONFIDENCE_THRESHOLD, POSE_MIN_DETECTION_CONFIDENCE
from features.feature_builder import FEATURE_NAMES, build_features
from features.pose_extractor import PoseExtractor

ImageInput = Union[str, np.ndarray, Image.Image]


class ShotPredictor:
    """
    CricketVision AI end-to-end inference wrapper.

    Input : image (path, numpy array, or PIL Image)
    Output: predicted shot label, confidence dict, annotated image
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        scaler_path: Union[str, Path],
        encoder_path: Union[str, Path],
        selector_path: Union[str, Path, None] = None,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.encoder_path = Path(encoder_path)
        self.selector_path = Path(selector_path) if selector_path else None

        for path in (self.model_path, self.scaler_path, self.encoder_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing required model artifact: {path}")

        if self.selector_path and not self.selector_path.exists():
            raise FileNotFoundError(f"Missing optional selector artifact: {self.selector_path}")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.encoder = joblib.load(self.encoder_path)
        self.selector = joblib.load(self.selector_path) if self.selector_path else None
        self.extractor = PoseExtractor(min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE)
        self.confidence_threshold = CONFIDENCE_THRESHOLD

    def predict(self, image_input: ImageInput) -> dict[str, Any]:
        # Handle PIL Image
        if isinstance(image_input, Image.Image):
            image_input = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)

        if isinstance(image_input, np.ndarray) and image_input.ndim == 2:
            image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)

        kp, annotated = self.extractor.extract(image_input)

        if kp is None:
            return {
                'success'    : False,
                'error'      : 'No pose detected. Try a clearer image with full body visible.',
                'annotated'  : image_input,
            }

        try:
            features = build_features(kp)
            feat_vec = np.array([features[f] for f in FEATURE_NAMES], dtype=float).reshape(1, -1)
            feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)
            feat_sc  = self.scaler.transform(feat_vec)

            # Apply Phase 3 Feature Pruning only when the selector matches the runtime feature width.
            if self.selector and getattr(self.selector, "n_features_in_", feat_sc.shape[1]) == feat_sc.shape[1]:
                feat_sel = self.selector.transform(feat_sc)
            else:
                feat_sel = feat_sc

            probs     = self.model.predict_proba(feat_sel)[0]
            class_idx = int(np.argmax(probs))
            confidence = float(probs[class_idx])

            # Phase 4 Upgrade: Confidence Threshold Interceptor
            if confidence < self.confidence_threshold:
                label = "Uncertain Shot"
            else:
                label = self.encoder.inverse_transform([class_idx])[0]

            conf_dict = {
                self.encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(probs)
            }

            return {
                'success'    : True,
                'label'      : label,
                'confidence' : confidence,
                'all_probs'  : conf_dict,
                'annotated'  : annotated,
            }
        except Exception as exc:
            self._logger.exception("Prediction pipeline failed")
            return {
                'success': False,
                'error': f'Prediction pipeline error: {exc}',
                'annotated': annotated,
            }

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.extractor.close()
