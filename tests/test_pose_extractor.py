import unittest
from unittest.mock import MagicMock, patch

from features.pose_extractor import PoseExtractor


class PoseExtractorTests(unittest.TestCase):
    def _make_extractor_without_mediapipe(self) -> PoseExtractor:
        """Build an extractor instance without initialising MediaPipe."""
        extractor = object.__new__(PoseExtractor)
        extractor.mp_pose = MagicMock()
        extractor.mp_draw = MagicMock()
        extractor.pose = MagicMock()
        return extractor

    @patch("features.pose_extractor.cv2.imread", return_value=None)
    def test_missing_file_returns_empty_result(self, _mock_imread):
        extractor = self._make_extractor_without_mediapipe()
        keypoints, annotated = extractor.extract("does-not-exist.jpg")

        self.assertIsNone(keypoints)
        self.assertIsNone(annotated)

    def test_close_releases_pose(self):
        extractor = self._make_extractor_without_mediapipe()
        extractor.close()
        extractor.pose.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
