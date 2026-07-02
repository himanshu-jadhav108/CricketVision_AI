"""MediaPipe Pose wrapper for cricket shot analysis."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np

ImageInput = Union[str, np.ndarray]


class PoseExtractor:
    """
    Wraps MediaPipe Pose.
    Takes an image path or numpy array.
    Returns 33 keypoints as (33, 4) array: [x, y, z, visibility]
    Returns None if pose not detected.
    """

    KEYPOINT_NAMES = {
        0: 'nose', 11: 'left_shoulder', 12: 'right_shoulder',
        13: 'left_elbow', 14: 'right_elbow',
        15: 'left_wrist', 16: 'right_wrist',
        23: 'left_hip', 24: 'right_hip',
        25: 'left_knee', 26: 'right_knee',
        27: 'left_ankle', 28: 'right_ankle',
    }

    def __init__(self, min_detection_confidence: float = 0.3) -> None:
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,          # True for images, False for video
            model_complexity=2,              # 0=lite, 1=full, 2=heavy
            min_detection_confidence=min_detection_confidence
        )

    def extract(
        self, image_input: ImageInput
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        image_input: file path (str) or numpy BGR array
        Returns: (keypoints np.array shape (33,4), annotated_image)
                 or (None, original_image) if detection fails
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        else:
            image = image_input.copy()

        if image is None:
            return None, None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if not result.pose_landmarks:
            return None, image

        # Extract to numpy array
        keypoints = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in result.pose_landmarks.landmark
        ])  # shape: (33, 4)

        # Draw skeleton on image
        annotated = image.copy()
        self.mp_draw.draw_landmarks(
            annotated,
            result.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        return keypoints, annotated

    def close(self) -> None:
        """Release the underlying MediaPipe Pose instance."""
        self.pose.close()
