import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from config import MODEL_DIR
import api


class DummyPredictor:
    def predict(self, image):
        return {
            "success": True,
            "label": "cover_drive",
            "confidence": 0.9876,
            "all_probs": {"cover_drive": 0.9876, "cut_shot": 0.0124},
            "annotated": np.zeros((4, 4, 3), dtype=np.uint8),
        }

    def close(self):
        return None


class ApiTests(unittest.TestCase):
    def test_health_endpoint_reports_model_state(self):
        with patch.object(api, "load_predictor", return_value=DummyPredictor()):
            with TestClient(api.app) as client:
                response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "online")
        self.assertTrue(payload["model_ready"])
        self.assertEqual(payload["max_upload_mb"], api.MAX_UPLOAD_MB)

    def test_rejects_non_image_uploads(self):
        with patch.object(api, "load_predictor", return_value=DummyPredictor()):
            with TestClient(api.app) as client:
                response = client.post(
                    "/predict",
                    files={"file": ("payload.txt", b"not an image", "text/plain")},
                )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid file format", response.json()["detail"])

    def test_rejects_oversized_uploads(self):
        with patch.object(api, "load_predictor", return_value=DummyPredictor()):
            original_max_upload_bytes = api.MAX_UPLOAD_BYTES
            original_max_upload_mb = api.MAX_UPLOAD_MB
            api.MAX_UPLOAD_BYTES = 1
            api.MAX_UPLOAD_MB = 1
            try:
                with TestClient(api.app) as client:
                    response = client.post(
                        "/predict",
                        files={"file": ("payload.png", b"\x89PNG", "image/png")},
                    )
            finally:
                api.MAX_UPLOAD_BYTES = original_max_upload_bytes
                api.MAX_UPLOAD_MB = original_max_upload_mb

        self.assertEqual(response.status_code, 413)
        self.assertIn("Maximum upload size", response.json()["detail"])

    def test_rejects_corrupted_image_bytes(self):
        with patch.object(api, "load_predictor", return_value=DummyPredictor()):
            with TestClient(api.app) as client:
                response = client.post(
                    "/predict",
                    files={"file": ("payload.png", b"not-a-real-image", "image/png")},
                )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid or corrupted image file", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
