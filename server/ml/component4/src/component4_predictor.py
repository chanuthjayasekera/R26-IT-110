# src/component4_predictor.py
# Final dashboard-compatible predictor for:
#   Kinect Gesture 3 - Chair - Left Arm Forward Raise
#
# Pipeline:
#   real video -> MediaPipe Pose Landmarker -> Kinect-style joints
#   -> 615 final features -> final_gesture3_chair_hybrid_model.joblib
#
# This file keeps the dashboard/frontend-friendly return keys and includes:
#   predict_video(...)
#   inspect_model_bundle(...)

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# Final deployment settings
# --------------------------------------------------
# Legacy bundles did not consistently use their saved threshold.  New bundles keep
# all deployment policy in the artifact so training and inference cannot drift.
DEPLOYMENT_THRESHOLD = 0.48
DEFAULT_EXERCISE_NAME = "Kinect Gesture 3 - Chair - Left Arm Forward Raise"


# --------------------------------------------------
# Kinect-style joint order used by the final hybrid model
# --------------------------------------------------
JOINTS = [
    "SpineBase", "SpineMid", "Neck", "Head",
    "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",
    "ShoulderRight", "ElbowRight", "WristRight", "HandRight",
    "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
    "HipRight", "KneeRight", "AnkleRight", "FootRight",
    "SpineShoulder", "HandTipLeft", "ThumbLeft", "HandTipRight", "ThumbRight",
]

JIDX = {joint: i for i, joint in enumerate(JOINTS)}


# --------------------------------------------------
# MediaPipe landmark indexes
# --------------------------------------------------
NOSE = 0

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
]


# --------------------------------------------------
# Math helpers
# --------------------------------------------------
def safe_norm(v: np.ndarray) -> float:
    return max(float(np.linalg.norm(v)), 1e-8)


def angle_between_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Vectorized angle ABC in degrees.
    a, b, c shape: T x 3
    """
    ba = a - b
    bc = c - b

    numerator = np.sum(ba * bc, axis=-1)
    denominator = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8

    cos_value = numerator / denominator
    cos_value = np.clip(cos_value, -1.0, 1.0)

    return np.degrees(np.arccos(cos_value))


def landmark_to_np(lm) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def get_visibility(lms, idx: int) -> float:
    return float(getattr(lms[idx], "visibility", 1.0))


def safe_stats(arr, prefix: str) -> Dict[str, float]:
    """
    Final feature stats used by the 615-feature model.
    """
    arr = np.asarray(arr, dtype=float)

    if arr.size == 0 or np.all(np.isnan(arr)):
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_rom": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_p25": 0.0,
            f"{prefix}_p75": 0.0,
        }

    return {
        f"{prefix}_mean": float(np.nanmean(arr)),
        f"{prefix}_std": float(np.nanstd(arr)),
        f"{prefix}_min": float(np.nanmin(arr)),
        f"{prefix}_max": float(np.nanmax(arr)),
        f"{prefix}_rom": float(np.nanmax(arr) - np.nanmin(arr)),
        f"{prefix}_median": float(np.nanmedian(arr)),
        f"{prefix}_p25": float(np.nanpercentile(arr, 25)),
        f"{prefix}_p75": float(np.nanpercentile(arr, 75)),
    }


def speed_signal(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.array([0.0])

    return np.linalg.norm(np.diff(points, axis=0), axis=1)


def jerk_signal(points: np.ndarray) -> np.ndarray:
    if len(points) < 4:
        return np.array([0.0])

    acceleration = np.diff(points, n=2, axis=0)
    jerk = np.diff(acceleration, axis=0)

    return np.linalg.norm(jerk, axis=1)


# --------------------------------------------------
# Pose conversion
# --------------------------------------------------
def normalize_pose_frame(joints: Dict[str, np.ndarray], y_axis_up: bool = False) -> Dict[str, np.ndarray]:
    """
    Normalize pose to reduce camera position/body size effects:
    - center using SpineBase / mid-hip
    - scale using shoulder width
    """
    spine_base = joints["SpineBase"]
    shoulder_width = safe_norm(joints["ShoulderLeft"] - joints["ShoulderRight"])

    normalized = {name: (p - spine_base) / shoulder_width for name, p in joints.items()}
    if y_axis_up:
        for point in normalized.values():
            point[1] *= -1.0  # MediaPipe image Y-down -> Kinect Y-up
    return normalized


def mediapipe_to_kinect_style_joints(lms, y_axis_up: bool = False) -> Optional[Dict[str, np.ndarray]]:
    needed = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE,
        LEFT_ANKLE,
        RIGHT_ANKLE,
    ]

    if min(get_visibility(lms, idx) for idx in needed) < 0.35:
        return None

    p = lambda idx: landmark_to_np(lms[idx])

    shoulder_left = p(LEFT_SHOULDER)
    shoulder_right = p(RIGHT_SHOULDER)

    elbow_left = p(LEFT_ELBOW)
    elbow_right = p(RIGHT_ELBOW)

    wrist_left = p(LEFT_WRIST)
    wrist_right = p(RIGHT_WRIST)

    hip_left = p(LEFT_HIP)
    hip_right = p(RIGHT_HIP)

    knee_left = p(LEFT_KNEE)
    knee_right = p(RIGHT_KNEE)

    ankle_left = p(LEFT_ANKLE)
    ankle_right = p(RIGHT_ANKLE)

    nose = p(NOSE)

    hand_left = p(LEFT_INDEX) if get_visibility(lms, LEFT_INDEX) > 0.2 else wrist_left
    hand_right = p(RIGHT_INDEX) if get_visibility(lms, RIGHT_INDEX) > 0.2 else wrist_right

    hand_tip_left = p(LEFT_PINKY) if get_visibility(lms, LEFT_PINKY) > 0.2 else hand_left
    hand_tip_right = p(RIGHT_PINKY) if get_visibility(lms, RIGHT_PINKY) > 0.2 else hand_right

    thumb_left = p(LEFT_THUMB) if get_visibility(lms, LEFT_THUMB) > 0.2 else hand_left
    thumb_right = p(RIGHT_THUMB) if get_visibility(lms, RIGHT_THUMB) > 0.2 else hand_right

    foot_left = p(LEFT_FOOT_INDEX) if get_visibility(lms, LEFT_FOOT_INDEX) > 0.2 else ankle_left
    foot_right = p(RIGHT_FOOT_INDEX) if get_visibility(lms, RIGHT_FOOT_INDEX) > 0.2 else ankle_right

    spine_base = (hip_left + hip_right) / 2.0
    spine_shoulder = (shoulder_left + shoulder_right) / 2.0
    spine_mid = (spine_base + spine_shoulder) / 2.0
    neck = (spine_shoulder + nose) / 2.0

    joints = {
        "SpineBase": spine_base,
        "SpineMid": spine_mid,
        "Neck": neck,
        "Head": nose,

        "ShoulderLeft": shoulder_left,
        "ElbowLeft": elbow_left,
        "WristLeft": wrist_left,
        "HandLeft": hand_left,

        "ShoulderRight": shoulder_right,
        "ElbowRight": elbow_right,
        "WristRight": wrist_right,
        "HandRight": hand_right,

        "HipLeft": hip_left,
        "KneeLeft": knee_left,
        "AnkleLeft": ankle_left,
        "FootLeft": foot_left,

        "HipRight": hip_right,
        "KneeRight": knee_right,
        "AnkleRight": ankle_right,
        "FootRight": foot_right,

        "SpineShoulder": spine_shoulder,

        "HandTipLeft": hand_tip_left,
        "ThumbLeft": thumb_left,

        "HandTipRight": hand_tip_right,
        "ThumbRight": thumb_right,
    }

    return normalize_pose_frame(joints, y_axis_up=y_axis_up)


# --------------------------------------------------
# Feature extraction: final 615-feature version
# --------------------------------------------------
def window_dicts_to_array(window: List[Dict[str, np.ndarray]]) -> np.ndarray:
    """
    Convert list of joint dictionaries into T x 25 x 3 array.
    """
    return np.asarray([[frame[joint] for joint in JOINTS] for frame in window], dtype=float)


def extract_window_features(window: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
    """
    Creates the same 615-style feature set used by the final hybrid model.
    """
    win = window_dicts_to_array(window)
    feats = {}

    # Joint coordinate summary features:
    # 25 joints x 3 axes x 5 stats = 375 features
    for ji, joint in enumerate(JOINTS):
        for ai, axis in enumerate(["x", "y", "z"]):
            vals = win[:, ji, ai]

            if vals.size == 0 or np.all(np.isnan(vals)):
                vals = np.array([0.0])

            feats[f"{joint}_{axis}_mean"] = float(np.nanmean(vals))
            feats[f"{joint}_{axis}_std"] = float(np.nanstd(vals))
            feats[f"{joint}_{axis}_rom"] = float(np.nanmax(vals) - np.nanmin(vals))
            feats[f"{joint}_{axis}_min"] = float(np.nanmin(vals))
            feats[f"{joint}_{axis}_max"] = float(np.nanmax(vals))

    # Arm-specific engineered features
    for side in ["Left", "Right"]:
        sh = win[:, JIDX[f"Shoulder{side}"], :]
        el = win[:, JIDX[f"Elbow{side}"], :]
        wr = win[:, JIDX[f"Wrist{side}"], :]
        hp = win[:, JIDX[f"Hip{side}"], :]
        hand = win[:, JIDX[f"Hand{side}"], :]

        upper_arm_len = np.linalg.norm(el - sh, axis=1)
        forearm_len = np.linalg.norm(wr - el, axis=1)
        full_arm_len = upper_arm_len + forearm_len

        signals = {
            f"{side}_elbow_angle": angle_between_points(sh, el, wr),
            f"{side}_shoulder_elevation_angle": angle_between_points(hp, sh, el),
            f"{side}_wrist_rel_y": wr[:, 1] - sh[:, 1],
            f"{side}_elbow_rel_y": el[:, 1] - sh[:, 1],
            f"{side}_hand_rel_y": hand[:, 1] - sh[:, 1],
            f"{side}_wrist_lateral": np.abs(wr[:, 0] - sh[:, 0]),
            f"{side}_elbow_lateral": np.abs(el[:, 0] - sh[:, 0]),
            f"{side}_hand_lateral": np.abs(hand[:, 0] - sh[:, 0]),
            f"{side}_upper_arm_len": upper_arm_len,
            f"{side}_forearm_len": forearm_len,
            f"{side}_full_arm_len": full_arm_len,
            f"{side}_wrist_speed": speed_signal(wr),
            f"{side}_wrist_jerk": jerk_signal(wr),
        }

        for name, arr in signals.items():
            feats.update(safe_stats(arr, name))

    # Trunk features
    spine_base = win[:, JIDX["SpineBase"], :]
    spine_shoulder = win[:, JIDX["SpineShoulder"], :]

    vertical_ref = spine_shoulder + np.array([0.0, 1.0, 0.0])
    trunk_lean = angle_between_points(vertical_ref, spine_shoulder, spine_base)

    feats.update(safe_stats(trunk_lean, "trunk_lean_angle"))

    # Left/right asymmetry features
    left_wrist = win[:, JIDX["WristLeft"], :]
    right_wrist = win[:, JIDX["WristRight"], :]
    left_elbow = win[:, JIDX["ElbowLeft"], :]
    right_elbow = win[:, JIDX["ElbowRight"], :]

    feats.update(safe_stats(left_wrist[:, 1] - right_wrist[:, 1], "wrist_height_diff_LR"))
    feats.update(safe_stats(left_elbow[:, 1] - right_elbow[:, 1], "elbow_height_diff_LR"))
    feats.update(safe_stats(left_wrist[:, 0] - right_wrist[:, 0], "wrist_lateral_diff_LR"))

    return feats


def build_windows(
    frames: List[Dict[str, np.ndarray]],
    window_frames: int = 60,
    stride_frames: int = 15,
) -> List[List[Dict[str, np.ndarray]]]:
    if len(frames) == 0:
        return []

    if len(frames) < window_frames:
        return [frames]

    windows = []
    last_start = len(frames) - window_frames
    used_starts = []

    for start in range(0, last_start + 1, stride_frames):
        windows.append(frames[start:start + window_frames])
        used_starts.append(start)

    # Add final window if stride did not land exactly at the last possible start.
    if used_starts and used_starts[-1] != last_start:
        windows.append(frames[last_start:last_start + window_frames])

    return windows


# --------------------------------------------------
# Drawing helpers
# --------------------------------------------------
def draw_landmarks_tasks_api(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in POSE_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], (0, 255, 0), 2)

    for x, y in points:
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


# --------------------------------------------------
# MediaPipe Tasks API
# --------------------------------------------------
def create_pose_landmarker(task_model_path: str):
    if not os.path.exists(task_model_path):
        raise FileNotFoundError(f"Pose task model not found: {task_model_path}")

    base_options = python.BaseOptions(model_asset_path=str(task_model_path))

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    return vision.PoseLandmarker.create_from_options(options)


def extract_pose_sequence_from_video(
    video_path: str,
    pose_task_model: str,
    save_annotated: Optional[str] = None,
    target_fps: Optional[float] = None,
    y_axis_up: bool = False,
) -> List[Dict[str, np.ndarray]]:

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None

    if save_annotated:
        Path(save_annotated).parent.mkdir(parents=True, exist_ok=True)
        suffix = Path(save_annotated).suffix.lower()
        fourcc = cv2.VideoWriter_fourcc(*("VP80" if suffix == ".webm" else "mp4v"))
        writer = cv2.VideoWriter(str(save_annotated), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create annotated video writer: {save_annotated}")

    pose_frames = []
    next_sample_ms = 0.0
    last_valid_joints = None
    last_valid_ms = -1e9

    with create_pose_landmarker(pose_task_model) as detector:
        frame_idx = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            timestamp_ms = int((frame_idx / fps) * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            status = "No pose"

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                joints = mediapipe_to_kinect_style_joints(landmarks, y_axis_up=y_axis_up)

                if joints is not None:
                    last_valid_joints = joints
                    last_valid_ms = timestamp_ms
                    if target_fps is None:
                        pose_frames.append(joints)
                    status = "Pose OK"
                else:
                    status = "Low visibility"

                draw_landmarks_tasks_api(frame, landmarks)

            # Sample on the video clock, not on the count of successful pose
            # detections. Brief dropouts use the last pose; long gaps remain
            # absent rather than silently turning a slow movement into a fast one.
            if target_fps is not None:
                interval_ms = 1000.0 / float(target_fps)
                while timestamp_ms + 1e-6 >= next_sample_ms:
                    if last_valid_joints is not None and timestamp_ms - last_valid_ms <= max(250.0, 2 * interval_ms):
                        pose_frames.append({k: v.copy() for k, v in last_valid_joints.items()})
                    next_sample_ms += interval_ms

            cv2.putText(
                frame,
                f"Frame: {frame_idx} | {status}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(frame)

            frame_idx += 1

    cap.release()

    if writer is not None:
        writer.release()

    return pose_frames


# --------------------------------------------------
# Model loading / dashboard inspection
# --------------------------------------------------
def load_model_bundle(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise ValueError("The joblib file must contain a dictionary bundle with 'model' and 'feature_cols'.")

    if "model" not in bundle:
        raise ValueError("Model bundle is missing key: model")

    if "feature_cols" not in bundle:
        raise ValueError("Model bundle is missing key: feature_cols")

    return bundle


def inspect_model_bundle(model_path: str) -> Dict:
    """
    Used by app.py. Do not remove.
    """
    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    feature_cols = bundle.get("feature_cols", [])

    return {
        "model_path": str(model_path),
        "model_class": type(model).__name__,
        "feature_count": len(feature_cols),
        "window_frames": int(bundle.get("window_frames", 60)),
        "stride_frames": int(bundle.get("stride_frames", bundle.get("stride", 15))),
        "saved_model_threshold": float(bundle.get("decision_threshold", 0.50)),
        "deployment_threshold": float(bundle.get("deployment_threshold", DEPLOYMENT_THRESHOLD)),
        "target_fps": bundle.get("target_fps"),
        "aggregation": str(bundle.get("aggregation", "mean")),
        "mediapipe_y_axis_up": bool(bundle.get("mediapipe_y_axis_up", False)),
        "exercise_name": str(bundle.get("exercise_name", DEFAULT_EXERCISE_NAME)),
        "movement_side": bundle.get("movement_side"),
        "keys": sorted(list(bundle.keys())),
    }


# --------------------------------------------------
# Prediction API used by dashboard
# --------------------------------------------------
def predict_video(
    video_path: str,
    model_path: str,
    pose_task_model: str,
    save_annotated: Optional[str] = None,
) -> Dict:

    bundle = load_model_bundle(model_path)

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    window_frames = int(bundle.get("window_frames", 60))
    stride_frames = int(bundle.get("stride_frames", bundle.get("stride", 15)))

    saved_model_threshold = float(bundle.get("decision_threshold", 0.50))
    # A bundle is a deployable contract.  A hard-coded UI threshold silently
    # invalidates validation results, so new artifacts opt into their saved value.
    # Preserve the original 0.48 policy for old artifacts. New training bundles
    # explicitly carry deployment_threshold and therefore cannot drift.
    decision_threshold = float(bundle.get("deployment_threshold", DEPLOYMENT_THRESHOLD))
    target_fps = bundle.get("target_fps")
    aggregation = str(bundle.get("aggregation", "mean"))
    mediapipe_y_axis_up = bool(bundle.get("mediapipe_y_axis_up", False))

    exercise_name = str(bundle.get("exercise_name", DEFAULT_EXERCISE_NAME))

    pose_frames = extract_pose_sequence_from_video(
        video_path=str(video_path),
        pose_task_model=str(pose_task_model),
        save_annotated=str(save_annotated) if save_annotated else None,
        target_fps=float(target_fps) if target_fps is not None else None,
        y_axis_up=mediapipe_y_axis_up,
    )

    if len(pose_frames) == 0:
        raise RuntimeError(
            "No valid pose frames were extracted. Full body must be visible with good lighting."
        )

    windows = build_windows(
        frames=pose_frames,
        window_frames=window_frames,
        stride_frames=stride_frames,
    )

    X_raw = pd.DataFrame([extract_window_features(window) for window in windows])

    missing_features = [c for c in feature_cols if c not in X_raw.columns]
    extra_features = [c for c in X_raw.columns if c not in feature_cols]

    X = X_raw.reindex(columns=feature_cols, fill_value=0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        raw_model_window_preds = model.predict(X)

        if hasattr(model, "predict_proba"):
            window_probs = model.predict_proba(X)
        else:
            window_probs = None

    if window_probs is not None and 1 in list(model.classes_):
        correct_idx = list(model.classes_).index(1)
        correct_probs = window_probs[:, correct_idx].astype(float)
    else:
        correct_probs = raw_model_window_preds.astype(float)

    if aggregation == "median":
        mean_correct_probability = float(np.median(correct_probs))
    elif aggregation == "trimmed_mean" and len(correct_probs) >= 5:
        ordered = np.sort(correct_probs)
        trim = max(1, int(len(ordered) * 0.1))
        mean_correct_probability = float(np.mean(ordered[trim:-trim]))
    else:
        mean_correct_probability = float(np.mean(correct_probs))

    calibrated_window_preds = (correct_probs >= decision_threshold).astype(int)

    correct_ratio = float(np.mean(calibrated_window_preds == 1))
    incorrect_ratio = float(np.mean(calibrated_window_preds == 0))

    final_label = 1 if mean_correct_probability >= decision_threshold else 0
    final_text = "Correct" if final_label == 1 else "Incorrect"

    reliability = "High"
    reliability_notes = ["Enough valid pose frames were extracted for exercise-quality screening."]

    if len(pose_frames) < window_frames:
        reliability = "Medium"
        reliability_notes = [
            f"Only {len(pose_frames)} valid pose frames were extracted. One short window was analyzed."
        ]

    if len(pose_frames) < 30:
        reliability = "Low"
        reliability_notes = [
            f"Only {len(pose_frames)} valid pose frames were extracted. Repeat with full body visible."
        ]

    probability_margin = abs(mean_correct_probability - decision_threshold)
    if probability_margin < 0.03:
        reliability = "Low"
        reliability_notes = [
            f"The score is only {probability_margin:.3f} from the decision threshold. "
            "Treat this result as uncertain and repeat the recording."
        ]

    quality_score = float(np.clip(mean_correct_probability * 100.0, 0.0, 100.0))

    return {
        "video_path": str(video_path),
        "exercise_name": exercise_name,
        "pose_model": str(pose_task_model),
        "model_path": str(model_path),
        "model_class": type(model).__name__,

        "valid_pose_frames": len(pose_frames),
        "num_windows": len(windows),

        "expected_model_features": len(feature_cols),
        "extracted_raw_features": len(X_raw.columns),
        "missing_model_features": len(missing_features),
        "extra_raw_features": len(extra_features),

        "window_predictions": calibrated_window_preds.tolist(),
        "raw_model_window_predictions": raw_model_window_preds.tolist(),
        "window_correct_probabilities": [float(x) for x in correct_probs],

        "correct_ratio": round(correct_ratio, 4),
        "incorrect_ratio": round(incorrect_ratio, 4),
        "mean_correct_probability": round(mean_correct_probability, 4),
        "quality_score": round(quality_score, 2),
        "probability_margin": round(probability_margin, 4),

        "saved_model_threshold": round(saved_model_threshold, 4),
        "decision_threshold": decision_threshold,
        "aggregation": aggregation,
        "target_fps": target_fps,
        "mediapipe_y_axis_up": mediapipe_y_axis_up,

        "final_label": int(final_label),
        "final_prediction": final_text,

        "reliability": reliability,
        "reliability_notes": reliability_notes,
        "annotated_video_path": str(save_annotated) if save_annotated else "",
    }
