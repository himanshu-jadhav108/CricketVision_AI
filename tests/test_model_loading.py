import unittest

from config import ENCODER_PATH, GITHUB_REPO_SLUG, MODEL_DIR, MODEL_PATH, PROJECT_NAME, SCALER_PATH, model_files_ready


class ModelLoadingTests(unittest.TestCase):
    def test_project_name_is_cricketvision_ai(self):
        self.assertEqual(PROJECT_NAME, "CricketVision AI")

    def test_github_repo_slug(self):
        self.assertEqual(GITHUB_REPO_SLUG, "CricketVision-AI")

    def test_required_artifacts_exist(self):
        if not model_files_ready():
            self.skipTest("Model artifacts not present in models/")

        self.assertTrue(MODEL_PATH.exists())
        self.assertTrue(SCALER_PATH.exists())
        self.assertTrue(ENCODER_PATH.exists())

    def test_model_directory_is_models(self):
        self.assertEqual(MODEL_DIR.name, "models")


if __name__ == "__main__":
    unittest.main()
