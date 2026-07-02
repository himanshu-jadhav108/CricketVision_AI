"""Biomechanical feature engineering from MediaPipe pose landmarks."""

import numpy as np


# MediaPipe landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
L_KNEE,     R_KNEE     = 25, 26
L_ANKLE,    R_ANKLE    = 27, 28
NOSE                   = 0


def _angle_3d(a, b, c):
    """
    Angle at joint b formed by points a-b-c in 3D space.
    Returns angle in degrees [0, 180].
    """
    ba = a - b
    bc = c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm < 1e-8:
        return 0.0
    cos_angle = np.dot(ba, bc) / norm
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _dist_3d(a, b):
    """Euclidean distance between two 3D points."""
    return float(np.linalg.norm(a[:3] - b[:3]))


def _dist(a, b):
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(a[:2] - b[:2]))


def build_features(kp, prev_kp=None):
    """
    kp: numpy array of shape (33, 4) — [x, y, z, visibility] per landmark
    prev_kp: identical shape array for the previous frame (optional)

    Returns: dict of named features
    """
    p  = kp[:, :2]   # Use only x, y for 2D geometry
    p3 = kp[:, :3]   # Use x, y, z for 3D geometry

    # --- 3D Torso Length (Robust normalisation base) ---
    mid_shoulder_3d = (p3[L_SHOULDER] + p3[R_SHOULDER]) / 2
    mid_hip_3d      = (p3[L_HIP]      + p3[R_HIP])      / 2
    torso_len_3d = _dist_3d(mid_shoulder_3d, mid_hip_3d)
    if torso_len_3d < 1e-6:
        torso_len_3d = 1e-6
    
    # Use 3D torso length as the primary scale for all distances
    scale = torso_len_3d

    # --- Joint Angles (degrees) ---
    features = {}

    # --- 3D Joint Angles (New for 90% accuracy) ---
    # These capture the true rotation even when the person is side-on
    features['angle_r_elbow_3d']    = _angle_3d(p3[R_SHOULDER], p3[R_ELBOW], p3[R_WRIST])
    features['angle_l_elbow_3d']    = _angle_3d(p3[L_SHOULDER], p3[L_ELBOW], p3[L_WRIST])
    features['angle_r_shoulder_3d'] = _angle_3d(p3[R_HIP],      p3[R_SHOULDER], p3[R_ELBOW])
    features['angle_l_shoulder_3d'] = _angle_3d(p3[L_HIP],      p3[L_SHOULDER], p3[L_ELBOW])
    features['angle_r_knee_3d']     = _angle_3d(p3[R_HIP],      p3[R_KNEE],     p3[R_ANKLE])
    features['angle_l_knee_3d']     = _angle_3d(p3[L_HIP],      p3[L_KNEE],     p3[L_ANKLE])
    features['angle_r_hip_3d']      = _angle_3d(p3[R_SHOULDER], p3[R_HIP],      p3[R_KNEE])
    features['angle_l_hip_3d']      = _angle_3d(p3[L_SHOULDER], p3[L_HIP],      p3[L_KNEE])

    # --- Trunk lean (angle of spine vs vertical) ---
    mid_shoulder_2d = (p[L_SHOULDER] + p[R_SHOULDER]) / 2
    mid_hip_2d      = (p[L_HIP]      + p[R_HIP])      / 2
    spine_vec       = mid_shoulder_2d - mid_hip_2d
    vertical     = np.array([0, -1])
    norm = np.linalg.norm(spine_vec)
    if norm > 1e-6:
        cos_trunk = np.dot(spine_vec / norm, vertical)
        features['trunk_lean'] = float(np.degrees(np.arccos(np.clip(cos_trunk, -1, 1))))
    else:
        features['trunk_lean'] = 0.0

    # --- Normalised distances (relative to torso length) ---
    features['r_wrist_to_r_hip']    = _dist(p[R_WRIST], p[R_HIP])    / scale
    features['l_wrist_to_l_hip']    = _dist(p[L_WRIST], p[L_HIP])    / scale
    features['r_wrist_to_head']     = _dist(p[R_WRIST], p[NOSE])      / scale
    features['l_wrist_to_head']     = _dist(p[L_WRIST], p[NOSE])      / scale
    features['wrist_spread']        = _dist(p[R_WRIST], p[L_WRIST])   / scale
    features['ankle_spread']        = _dist(p[R_ANKLE], p[L_ANKLE])   / scale
    features['hip_to_ankle_r']      = _dist(p[R_HIP],   p[R_ANKLE])   / scale
    features['hip_to_ankle_l']      = _dist(p[L_HIP],   p[L_ANKLE])   / scale

    # --- Body position ratios (y-axis, normalised by image height) ---
    # Lower y = higher on screen in MediaPipe normalised coords
    features['wrist_height_r']   = float(p[R_WRIST, 1])    # y of right wrist
    features['wrist_height_l']   = float(p[L_WRIST, 1])
    features['knee_bend_r']      = float(p[R_KNEE,  1] - p[R_HIP, 1])
    features['knee_bend_l']      = float(p[L_KNEE,  1] - p[L_HIP, 1])

    # --- Lateral symmetry ---
    # Positive = right side dominant, negative = left side dominant
    features['wrist_asymmetry']    = float(p[R_WRIST, 0] - p[L_WRIST, 0])
    features['shoulder_asymmetry'] = float(p[R_SHOULDER, 0] - p[L_SHOULDER, 0])

    # --- Visibility scores (data quality) ---
    features['vis_r_wrist']  = float(kp[R_WRIST,  3])
    features['vis_l_wrist']  = float(kp[L_WRIST,  3])
    features['vis_r_elbow']  = float(kp[R_ELBOW,  3])

    # --- Engineered Features (added for robustness) ---
    # 1. Elbow extension mean (average angle of both elbows)
    features['elbow_extension_mean'] = (features['angle_r_elbow_3d'] + features['angle_l_elbow_3d']) / 2

    # 2. Knee flexion symmetry (asymmetry in knee bend)
    features['knee_flexion_diff'] = abs(features['angle_r_knee_3d'] - features['angle_l_knee_3d'])

    # 3. Hip-shoulder angle ratio
    features['hip_shoulder_ratio'] = features['angle_r_hip_3d'] / (features['angle_r_shoulder_3d'] + 1e-6)

    # 4. Combined wrist height (average wrist elevation)
    features['wrist_height_mean'] = (features['wrist_height_r'] + features['wrist_height_l']) / 2

    # 5. Stance width index (relative to leg length)
    features['stance_width'] = features['ankle_spread'] / (features['hip_to_ankle_r'] + features['hip_to_ankle_l'] + 1e-6)

    # --- 3D / Depth Features (New for 90% accuracy) ---
    # MediaPipe Z is "relative to midpoint of hips". 
    # Negative = closer to camera, Positive = further from camera.
    features['z_wrist_r'] = float(kp[R_WRIST, 2])
    features['z_wrist_l'] = float(kp[L_WRIST, 2])
    
    # 3D reach (distance from wrists to hips in Z-plane)
    # This helps distinguish if the bat is coming towards the camera (Straight Drive)
    # vs going across (Cover Drive).
    mid_hip_z = (kp[L_HIP, 2] + kp[R_HIP, 2]) / 2
    features['reach_z_r'] = float(kp[R_WRIST, 2] - mid_hip_z)
    features['reach_z_l'] = float(kp[L_WRIST, 2] - mid_hip_z)
    
    # 3D Shoulder Orientation (Detects body tilt towards/away from camera)
    features['shoulder_tilt_z'] = float(kp[R_SHOULDER, 2] - kp[L_SHOULDER, 2])
    
    # 3D normalized wrist distance (using all 3 axes)
    features['wrist_spread_3d'] = _dist_3d(kp[R_WRIST], kp[L_WRIST]) / scale

    # --- Centroid Offset Features (Updated for 90% accuracy) ---
    # Centroid = mid-point of the hips
    features['r_wrist_x_off'] = float(p3[R_WRIST, 0] - mid_hip_3d[0]) / scale
    features['l_wrist_x_off'] = float(p3[L_WRIST, 0] - mid_hip_3d[0]) / scale
    features['r_wrist_y_off'] = float(p3[R_WRIST, 1] - mid_hip_3d[1]) / scale
    features['l_wrist_y_off'] = float(p3[L_WRIST, 1] - mid_hip_3d[1]) / scale
    features['r_wrist_z_off'] = float(p3[R_WRIST, 2] - mid_hip_3d[2]) / scale

    # NEW: Shoulder-relative offsets (Better for Drive shots)
    features['r_wrist_x_shoulder_off'] = float(p3[R_WRIST, 0] - mid_shoulder_3d[0]) / scale
    features['r_wrist_y_shoulder_off'] = float(p3[R_WRIST, 1] - mid_shoulder_3d[1]) / scale
    
    # NEW: Wrist-to-Shoulder Ratio (Arm extension)
    features['wrist_to_shoulder_ratio_r'] = _dist_3d(p3[R_WRIST], p3[R_SHOULDER]) / scale
    features['wrist_to_shoulder_ratio_l'] = _dist_3d(p3[L_WRIST], p3[L_SHOULDER]) / scale

    # --- Bat Simulation Features (Projected Bat Vector) ---
    # The 'bat' line is assumed to be roughly the extension of the arms/wrists.
    wrist_mid  = (p3[R_WRIST] + p3[L_WRIST]) / 2
    elbow_mid  = (p3[R_ELBOW] + p3[L_ELBOW]) / 2
    bat_vector = wrist_mid - elbow_mid
    
    # Bat angle vs ground (vertical)
    bat_norm = np.linalg.norm(bat_vector)
    if bat_norm > 1e-6:
        features['bat_angle_vertical'] = float(np.degrees(np.arccos(np.clip(np.dot(bat_vector / bat_norm, [0, 1, 0]), -1.0, 1.0))))
    else:
        features['bat_angle_vertical'] = 0.0

    # Bat angle vs camera (horizontal plane Z vs X)
    if bat_norm > 1e-6:
        z_x_plane = np.array([bat_vector[0], 0, bat_vector[2]])
        zx_norm = np.linalg.norm(z_x_plane)
        if zx_norm > 1e-6:
            features['bat_angle_horizontal'] = float(np.degrees(np.arccos(np.clip(np.dot(z_x_plane / zx_norm, [1, 0, 0]), -1.0, 1.0))))
        else:
            features['bat_angle_horizontal'] = 0.0
    else:
        features['bat_angle_horizontal'] = 0.0

    # Added: Body alignment vs Target (Straight Drive vs Leg Glance)
    features['shoulder_alignment_x'] = float(p3[R_SHOULDER, 0] - p3[L_SHOULDER, 0])
    features['hip_alignment_x'] = float(p3[R_HIP, 0] - p3[L_HIP, 0])
    features['torso_twist'] = features['shoulder_alignment_x'] - features['hip_alignment_x']

    # --- Temporal Features (Velocity) ---
    if prev_kp is not None:
        # Change in position over time (Velocity)
        pp3 = prev_kp[:, :3]
        
        # Wrist velocities
        features['vel_r_wrist_x'] = float(p3[R_WRIST, 0] - pp3[R_WRIST, 0]) / scale
        features['vel_r_wrist_y'] = float(p3[R_WRIST, 1] - pp3[R_WRIST, 1]) / scale
        features['vel_l_wrist_x'] = float(p3[L_WRIST, 0] - pp3[L_WRIST, 0]) / scale
        features['vel_l_wrist_y'] = float(p3[L_WRIST, 1] - pp3[L_WRIST, 1]) / scale
        
        # Bat Velocity
        prev_wrist_mid = (pp3[R_WRIST] + pp3[L_WRIST]) / 2
        features['vel_bat_strike_x'] = float(wrist_mid[0] - prev_wrist_mid[0]) / scale
        features['vel_bat_strike_y'] = float(wrist_mid[1] - prev_wrist_mid[1]) / scale
    else:
        # Zero-pad if no previous frame or processing a static image
        features['vel_r_wrist_x'] = 0.0
        features['vel_r_wrist_y'] = 0.0
        features['vel_l_wrist_x'] = 0.0
        features['vel_l_wrist_y'] = 0.0
        features['vel_bat_strike_x'] = 0.0
        features['vel_bat_strike_y'] = 0.0

    return features


FEATURE_NAMES = list(build_features(np.zeros((33, 4))).keys())
