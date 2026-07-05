import unittest

import numpy as np

from features.feature_builder import FEATURE_NAMES, build_features


class FeatureBuilderTests(unittest.TestCase):
    def test_build_features_returns_expected_schema(self):
        keypoints = np.zeros((33, 4), dtype=float)
        features = build_features(keypoints)

        self.assertEqual(list(features.keys()), FEATURE_NAMES)
        self.assertEqual(len(features), len(FEATURE_NAMES))
        self.assertTrue(np.isfinite(list(features.values())).all())

    def test_build_features_supports_previous_frame(self):
        keypoints = np.zeros((33, 4), dtype=float)
        previous_keypoints = np.zeros((33, 4), dtype=float)
        previous_keypoints[15, 0] = -0.1
        previous_keypoints[16, 0] = 0.1

        features = build_features(keypoints, prev_kp=previous_keypoints)

        self.assertIn("vel_bat_strike_x", features)
        self.assertIn("vel_bat_strike_y", features)
        self.assertTrue(np.isfinite(list(features.values())).all())


if __name__ == "__main__":
    unittest.main()
