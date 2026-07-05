import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from config import MODEL_DIR, SELECTOR_PATH
from predictor import ShotPredictor


class DummyExtractor:
    def extract(self, image_input):
        keypoints = np.zeros((33, 4), dtype=float)
        keypoints[11] = [0.40, 0.20, 0.00, 1.00]
        keypoints[12] = [0.60, 0.20, 0.00, 1.00]
        keypoints[23] = [0.42, 0.55, 0.00, 1.00]
        keypoints[24] = [0.58, 0.55, 0.00, 1.00]
        keypoints[15] = [0.30, 0.35, 0.00, 1.00]
        keypoints[16] = [0.70, 0.35, 0.00, 1.00]
        keypoints[0] = [0.50, 0.10, 0.00, 1.00]
        return keypoints, np.zeros((32, 32, 3), dtype=np.uint8)

    def close(self):
        return None


class ShotPredictorTests(unittest.TestCase):
    def test_missing_artifacts_fail_fast(self):
        with self.assertRaises(FileNotFoundError):
            ShotPredictor("missing-model.pkl", "missing-scaler.pkl", "missing-encoder.pkl")

    @patch("predictor.PoseExtractor", return_value=DummyExtractor())
    def test_predict_returns_structured_result(self, _mock_pose_extractor):
        if not (MODEL_DIR / "xgboost_v1.pkl").exists():
            self.skipTest("Model artifacts not present in models/")

        predictor = ShotPredictor(
            MODEL_DIR / "xgboost_v1.pkl",
            MODEL_DIR / "scaler.pkl",
            MODEL_DIR / "label_encoder.pkl",
            selector_path=SELECTOR_PATH,
        )

        try:
            result = predictor.predict(Image.new("RGB", (64, 64), color="white"))
        finally:
            predictor.close()

        self.assertIn("success", result)
        self.assertTrue(result["success"])
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertIn("all_probs", result)
        self.assertIn("annotated", result)


if __name__ == "__main__":
    unittest.main()
