from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "06_models" / "component_3"

# ============================================================
# FINAL PD V14 MODEL FILES
# ============================================================

PD_MODEL_PATH = MODEL_DIR / "component3_pd_model_v14_xgboost_precision_tuned.joblib"
PD_FEATURES_PATH = MODEL_DIR / "component3_pd_feature_list_v14_xgboost_precision_tuned.json"
PD_SETTINGS_PATH = MODEL_DIR / "component3_pd_settings_v14_xgboost_precision_tuned.json"

INFERENCE_TRAINING_SAFE_DIR = PROJECT_ROOT / "02_extracted_csv" / "training_safe" / "inference"


# ============================================================
# FINAL PD V14 THRESHOLDS
# ============================================================

DEFAULT_PD_THRESHOLD = 0.405
PD_TENDENCY_THRESHOLD = 0.35


# ============================================================
# PREPROCESSING SETTINGS - MUST MATCH TRAINING/SCA PIPELINE
# ============================================================

TARGET_FPS = 30
WINDOW_SIZE = 60
STRIDE = 15
PAD_SHORT_CLIPS = True
DIRECTION_STANDARDIZE = True

LANDMARK_COUNT = 33
AXES = ["x", "y", "z", "vis"]

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

MIN_SHOULDER_SCALE = 1e-6


# ============================================================
# RELIABILITY SETTINGS
# ============================================================

MIN_RELIABLE_WINDOWS = 5
RECOMMENDED_WINDOWS = 8
MIN_RELIABLE_DURATION_SEC = 4.0


# ============================================================
# BASIC HELPERS
# ============================================================

def landmark_columns() -> list[str]:
    return [f"lm{i}_{axis}" for i in range(LANDMARK_COUNT) for axis in AXES]


def find_latest_inference_csv() -> Path:
    if not INFERENCE_TRAINING_SAFE_DIR.exists():
        raise FileNotFoundError(
            f"Inference training-safe folder not found: {INFERENCE_TRAINING_SAFE_DIR}"
        )

    csv_files = sorted(
        INFERENCE_TRAINING_SAFE_DIR.glob("*_training_safe_landmarks.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No training-safe inference CSVs found in: {INFERENCE_TRAINING_SAFE_DIR}"
        )

    return csv_files[0]


def infer_direction_from_filename(path: Path) -> str | None:
    name = path.name.upper()

    if "_L2R_" in name:
        return "L2R"

    if "_R2L_" in name:
        return "R2L"

    return None


def infer_fps_from_filename(path: Path) -> float | None:
    name = path.name.lower()

    if "50fps" in name:
        return 50.0

    if "30fps" in name:
        return 30.0

    return None


def estimate_direction_from_pose_df(df: pd.DataFrame) -> tuple[str, float, str]:
    required = ["lm23_x", "lm24_x"]

    for col in required:
        if col not in df.columns:
            return "L2R", 0.0, "low"

    if len(df) < 2:
        return "L2R", 0.0, "low"

    mid_hip_x = (
        pd.to_numeric(df["lm23_x"], errors="coerce").to_numpy(dtype=float)
        + pd.to_numeric(df["lm24_x"], errors="coerce").to_numpy(dtype=float)
    ) / 2.0

    mid_hip_x = mid_hip_x[np.isfinite(mid_hip_x)]

    if len(mid_hip_x) < 2:
        return "L2R", 0.0, "low"

    n = len(mid_hip_x)
    k = max(3, int(n * 0.10))

    start_x = float(np.median(mid_hip_x[:k]))
    end_x = float(np.median(mid_hip_x[-k:]))

    delta = end_x - start_x
    direction = "L2R" if delta >= 0 else "R2L"

    abs_delta = abs(delta)

    if abs_delta >= 0.25:
        confidence = "high"
    elif abs_delta >= 0.10:
        confidence = "medium"
    else:
        confidence = "low"

    return direction, float(delta), confidence


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# CSV → NORMALIZED WINDOWS
# ============================================================

def load_pose_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Training feature names use lm*_vis.
    # Some CSVs may have lm*_visibility, so copy that into lm*_vis.
    for i in range(LANDMARK_COUNT):
        vis_col = f"lm{i}_vis"
        visibility_col = f"lm{i}_visibility"

        if vis_col not in df.columns:
            if visibility_col in df.columns:
                df[vis_col] = df[visibility_col]
            else:
                df[vis_col] = 0.0

    cols = landmark_columns()
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing landmark columns in {path.name}: {missing[:10]}")

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[cols] = (
        df[cols]
        .replace([np.inf, -np.inf], np.nan)
        .interpolate(limit_direction="both")
        .bfill()
        .ffill()
        .fillna(0.0)
    )

    return df


def sequence_from_df(df: pd.DataFrame) -> np.ndarray:
    cols = landmark_columns()
    arr = df[cols].to_numpy(dtype=np.float32)
    arr = arr.reshape(len(df), LANDMARK_COUNT, len(AXES))
    return arr


def get_time_seconds(df: pd.DataFrame, fps_used: float) -> np.ndarray:
    if "timestamp_ms" in df.columns:
        t = pd.to_numeric(df["timestamp_ms"], errors="coerce").to_numpy(dtype=np.float64) / 1000.0
        t = t - t[0]

        if np.all(np.isfinite(t)) and len(t) >= 2 and t[-1] > t[0]:
            return t

    return np.arange(len(df), dtype=np.float64) / float(fps_used)


def resample_sequence(seq: np.ndarray, time_sec: np.ndarray, target_fps: int) -> np.ndarray:
    if len(seq) < 2:
        return seq

    duration = time_sec[-1] - time_sec[0]

    if duration <= 0:
        return seq

    target_len = int(round(duration * target_fps)) + 1
    target_len = max(target_len, 1)

    new_t = np.linspace(time_sec[0], time_sec[-1], target_len)

    flat = seq.reshape(len(seq), -1)
    resampled = np.zeros((target_len, flat.shape[1]), dtype=np.float32)

    for j in range(flat.shape[1]):
        resampled[:, j] = np.interp(new_t, time_sec, flat[:, j])

    return resampled.reshape(target_len, LANDMARK_COUNT, len(AXES))


def edge_pad_to_window(seq: np.ndarray, window_size: int) -> np.ndarray:
    if len(seq) >= window_size:
        return seq

    pad_len = window_size - len(seq)
    last_frame = seq[-1:, :, :]
    pad = np.repeat(last_frame, pad_len, axis=0)

    return np.concatenate([seq, pad], axis=0)


def normalize_pose(seq: np.ndarray) -> np.ndarray:
    """
    Must match training/SCA preprocessing:
    - mid-hip centering
    - shoulder-width scaling
    """
    seq = seq.copy()
    xyz = seq[:, :, :3]

    mid_hip = (xyz[:, LEFT_HIP, :] + xyz[:, RIGHT_HIP, :]) / 2.0
    xyz = xyz - mid_hip[:, None, :]

    shoulder_vec = xyz[:, LEFT_SHOULDER, :] - xyz[:, RIGHT_SHOULDER, :]
    shoulder_width = np.linalg.norm(shoulder_vec, axis=1)

    median_scale = np.nanmedian(shoulder_width)

    if not np.isfinite(median_scale) or median_scale < MIN_SHOULDER_SCALE:
        median_scale = 1.0

    shoulder_width = np.where(
        shoulder_width < MIN_SHOULDER_SCALE,
        median_scale,
        shoulder_width,
    )

    xyz = xyz / shoulder_width[:, None, None]

    seq[:, :, :3] = xyz
    return seq


def standardize_direction(seq: np.ndarray, direction: str) -> np.ndarray:
    """
    Must match training/SCA preprocessing:
    R2L sequences are flipped on normalized x-axis.
    """
    seq = seq.copy()
    direction = str(direction).upper().strip()

    if DIRECTION_STANDARDIZE and direction == "R2L":
        seq[:, :, 0] *= -1.0

    return seq


def make_sequence_windows(seq: np.ndarray, window_size: int, stride: int) -> list[np.ndarray]:
    windows = []

    if len(seq) < window_size:
        return windows

    for start in range(0, len(seq) - window_size + 1, stride):
        windows.append(seq[start:start + window_size])

    return windows


def aggregate_features(window: np.ndarray) -> dict:
    """
    Same V3 feature aggregation as SCA/Component 1 training:
    132 base values × 7 feature groups = 924 features.
    """
    flat = window.reshape(window.shape[0], -1)
    base_cols = landmark_columns()

    mean_vals = flat.mean(axis=0)
    std_vals = flat.std(axis=0)
    min_vals = flat.min(axis=0)
    max_vals = flat.max(axis=0)
    range_vals = max_vals - min_vals

    velocity = np.diff(flat, axis=0)
    abs_velocity = np.abs(velocity)

    mean_abs_velocity = abs_velocity.mean(axis=0)
    std_velocity = velocity.std(axis=0)

    features = {}

    for col, val in zip(base_cols, mean_vals):
        features[f"mean_{col}"] = float(val)

    for col, val in zip(base_cols, std_vals):
        features[f"std_{col}"] = float(val)

    for col, val in zip(base_cols, min_vals):
        features[f"min_{col}"] = float(val)

    for col, val in zip(base_cols, max_vals):
        features[f"max_{col}"] = float(val)

    for col, val in zip(base_cols, range_vals):
        features[f"range_{col}"] = float(val)

    for col, val in zip(base_cols, mean_abs_velocity):
        features[f"mean_abs_vel_{col}"] = float(val)

    for col, val in zip(base_cols, std_velocity):
        features[f"std_vel_{col}"] = float(val)

    return features


# ============================================================
# PD BIOMECHANICAL FEATURES
# ============================================================

def safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def safe_div(a, b, eps: float = 1e-6):
    return a / (np.abs(b) + eps)


def add_pd_biomech_features(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = feature_df.copy()
    bio = {}

    body_pairs = [
        (11, 12, "shoulder"),
        (23, 24, "hip"),
        (25, 26, "knee"),
        (27, 28, "ankle"),
        (29, 30, "heel"),
        (31, 32, "foot"),
    ]

    for left_lm, right_lm, name in body_pairs:
        left_rx = safe_col(out, f"range_lm{left_lm}_x")
        left_ry = safe_col(out, f"range_lm{left_lm}_y")
        right_rx = safe_col(out, f"range_lm{right_lm}_x")
        right_ry = safe_col(out, f"range_lm{right_lm}_y")

        left_range = np.sqrt(left_rx ** 2 + left_ry ** 2)
        right_range = np.sqrt(right_rx ** 2 + right_ry ** 2)

        bio[f"bio_pd_left_{name}_motion_range"] = left_range
        bio[f"bio_pd_right_{name}_motion_range"] = right_range
        bio[f"bio_pd_{name}_motion_mean"] = (left_range + right_range) / 2.0
        bio[f"bio_pd_{name}_motion_asymmetry"] = np.abs(left_range - right_range)
        bio[f"bio_pd_{name}_motion_ratio"] = safe_div(left_range, right_range)

    for left_lm, right_lm, name in body_pairs:
        left_vx = safe_col(out, f"mean_abs_vel_lm{left_lm}_x")
        left_vy = safe_col(out, f"mean_abs_vel_lm{left_lm}_y")
        right_vx = safe_col(out, f"mean_abs_vel_lm{right_lm}_x")
        right_vy = safe_col(out, f"mean_abs_vel_lm{right_lm}_y")

        left_vel = np.sqrt(left_vx ** 2 + left_vy ** 2)
        right_vel = np.sqrt(right_vx ** 2 + right_vy ** 2)

        bio[f"bio_pd_left_{name}_mean_abs_velocity"] = left_vel
        bio[f"bio_pd_right_{name}_mean_abs_velocity"] = right_vel
        bio[f"bio_pd_{name}_velocity_mean"] = (left_vel + right_vel) / 2.0
        bio[f"bio_pd_{name}_velocity_asymmetry"] = np.abs(left_vel - right_vel)
        bio[f"bio_pd_{name}_velocity_ratio"] = safe_div(left_vel, right_vel)

    for left_lm, right_lm, name in body_pairs:
        left_sx = safe_col(out, f"std_lm{left_lm}_x")
        left_sy = safe_col(out, f"std_lm{left_lm}_y")
        right_sx = safe_col(out, f"std_lm{right_lm}_x")
        right_sy = safe_col(out, f"std_lm{right_lm}_y")

        left_std = np.sqrt(left_sx ** 2 + left_sy ** 2)
        right_std = np.sqrt(right_sx ** 2 + right_sy ** 2)

        bio[f"bio_pd_left_{name}_variability"] = left_std
        bio[f"bio_pd_right_{name}_variability"] = right_std
        bio[f"bio_pd_{name}_variability_mean"] = (left_std + right_std) / 2.0
        bio[f"bio_pd_{name}_variability_asymmetry"] = np.abs(left_std - right_std)
        bio[f"bio_pd_{name}_variability_ratio"] = safe_div(left_std, right_std)

    bio_df = pd.DataFrame(bio, index=out.index)

    left_ankle_y_range = safe_col(out, "range_lm27_y")
    right_ankle_y_range = safe_col(out, "range_lm28_y")
    left_heel_y_range = safe_col(out, "range_lm29_y")
    right_heel_y_range = safe_col(out, "range_lm30_y")
    left_foot_y_range = safe_col(out, "range_lm31_y")
    right_foot_y_range = safe_col(out, "range_lm32_y")

    bio_df["bio_pd_ankle_vertical_range_mean"] = (left_ankle_y_range + right_ankle_y_range) / 2.0
    bio_df["bio_pd_heel_vertical_range_mean"] = (left_heel_y_range + right_heel_y_range) / 2.0
    bio_df["bio_pd_foot_vertical_range_mean"] = (left_foot_y_range + right_foot_y_range) / 2.0

    bio_df["bio_pd_foot_clearance_proxy"] = (
        bio_df["bio_pd_ankle_vertical_range_mean"]
        + bio_df["bio_pd_heel_vertical_range_mean"]
        + bio_df["bio_pd_foot_vertical_range_mean"]
    ) / 3.0

    bio_df["bio_pd_lower_limb_motion_mean"] = (
        bio_df["bio_pd_knee_motion_mean"]
        + bio_df["bio_pd_ankle_motion_mean"]
        + bio_df["bio_pd_heel_motion_mean"]
        + bio_df["bio_pd_foot_motion_mean"]
    ) / 4.0

    bio_df["bio_pd_lower_limb_velocity_mean"] = (
        bio_df["bio_pd_knee_velocity_mean"]
        + bio_df["bio_pd_ankle_velocity_mean"]
        + bio_df["bio_pd_heel_velocity_mean"]
        + bio_df["bio_pd_foot_velocity_mean"]
    ) / 4.0

    bio_df["bio_pd_lower_limb_variability_mean"] = (
        bio_df["bio_pd_knee_variability_mean"]
        + bio_df["bio_pd_ankle_variability_mean"]
        + bio_df["bio_pd_heel_variability_mean"]
        + bio_df["bio_pd_foot_variability_mean"]
    ) / 4.0

    bio_df["bio_pd_lower_limb_motion_asymmetry_score"] = (
        bio_df["bio_pd_knee_motion_asymmetry"]
        + bio_df["bio_pd_ankle_motion_asymmetry"]
        + bio_df["bio_pd_heel_motion_asymmetry"]
        + bio_df["bio_pd_foot_motion_asymmetry"]
    ) / 4.0

    bio_df["bio_pd_lower_limb_velocity_asymmetry_score"] = (
        bio_df["bio_pd_knee_velocity_asymmetry"]
        + bio_df["bio_pd_ankle_velocity_asymmetry"]
        + bio_df["bio_pd_heel_velocity_asymmetry"]
        + bio_df["bio_pd_foot_velocity_asymmetry"]
    ) / 4.0

    bio_df["bio_pd_lower_limb_variability_asymmetry_score"] = (
        bio_df["bio_pd_knee_variability_asymmetry"]
        + bio_df["bio_pd_ankle_variability_asymmetry"]
        + bio_df["bio_pd_heel_variability_asymmetry"]
        + bio_df["bio_pd_foot_variability_asymmetry"]
    ) / 4.0

    bio_df["bio_pd_upper_to_lower_motion_ratio"] = safe_div(
        bio_df["bio_pd_shoulder_motion_mean"],
        bio_df["bio_pd_lower_limb_motion_mean"],
    )

    bio_df["bio_pd_upper_to_lower_velocity_ratio"] = safe_div(
        bio_df["bio_pd_shoulder_velocity_mean"],
        bio_df["bio_pd_lower_limb_velocity_mean"],
    )

    bio_df["bio_pd_hip_to_foot_motion_ratio"] = safe_div(
        bio_df["bio_pd_hip_motion_mean"],
        bio_df["bio_pd_foot_motion_mean"],
    )

    bio_df["bio_pd_composite_gait_proxy_score"] = (
        bio_df["bio_pd_lower_limb_motion_asymmetry_score"]
        + bio_df["bio_pd_lower_limb_velocity_asymmetry_score"]
        + bio_df["bio_pd_upper_to_lower_motion_ratio"]
        + bio_df["bio_pd_hip_to_foot_motion_ratio"]
    ) / 4.0

    bio_df = bio_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out = pd.concat([out, bio_df], axis=1).copy()
    return out, list(bio_df.columns)


def preprocess_single_csv(csv_path: Path, direction: str, fps_used: float) -> pd.DataFrame:
    df = load_pose_csv(csv_path)

    seq = sequence_from_df(df)
    time_sec = get_time_seconds(df, fps_used)

    seq = resample_sequence(seq, time_sec, TARGET_FPS)

    padded = False

    if len(seq) < WINDOW_SIZE:
        if PAD_SHORT_CLIPS:
            seq = edge_pad_to_window(seq, WINDOW_SIZE)
            padded = True
        else:
            raise ValueError(f"Sequence too short after resampling: {len(seq)} frames")

    seq = normalize_pose(seq)
    seq = standardize_direction(seq, direction)

    windows = make_sequence_windows(seq, WINDOW_SIZE, STRIDE)

    if len(windows) == 0:
        raise ValueError("No windows created from this CSV.")

    rows = []

    for window_id, window in enumerate(windows):
        feat = aggregate_features(window)
        feat["window_id"] = window_id
        feat["fps_used"] = fps_used
        feat["padded"] = padded
        feat["resampled_frames"] = len(seq)
        rows.append(feat)

    feature_df = pd.DataFrame(rows)
    feature_df, _ = add_pd_biomech_features(feature_df)

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return feature_df


def get_clean_duration_seconds(csv_path: Path, fps_used: float) -> float:
    df = pd.read_csv(csv_path)

    if "timestamp_ms" in df.columns and len(df) >= 2:
        t = pd.to_numeric(df["timestamp_ms"], errors="coerce").to_numpy(dtype=np.float64) / 1000.0
        t = t[np.isfinite(t)]

        if len(t) >= 2:
            return float(t[-1] - t[0])

    return float(len(df) / fps_used) if fps_used else 0.0


# ============================================================
# RELIABILITY + DECISION LOGIC
# ============================================================

def reliability_from_windows(
    total_windows: int,
    clean_duration_sec: float,
    direction_confidence: str,
) -> tuple[str, list[str]]:
    reasons = []

    if total_windows < MIN_RELIABLE_WINDOWS:
        reasons.append(
            f"Only {total_windows} gait windows were available. Minimum reliable count is {MIN_RELIABLE_WINDOWS}."
        )

    if clean_duration_sec < MIN_RELIABLE_DURATION_SEC:
        reasons.append(
            f"Clean walking duration is {clean_duration_sec:.2f}s. Recommended minimum is about {MIN_RELIABLE_DURATION_SEC:.1f}s."
        )

    if direction_confidence == "low":
        reasons.append("Walking direction confidence is low.")

    if reasons:
        return "Low", reasons

    if total_windows < RECOMMENDED_WINDOWS or direction_confidence == "medium":
        if total_windows < RECOMMENDED_WINDOWS:
            reasons.append(
                f"{total_windows} windows are usable, but {RECOMMENDED_WINDOWS}+ windows are recommended for stronger confidence."
            )

        if direction_confidence == "medium":
            reasons.append("Walking direction confidence is medium.")

        return "Medium", reasons

    return "High", ["Enough clean walking data for reliable PD pattern screening."]


def pd_pattern_strength(probability: float, pd_threshold: float) -> str:
    if probability < PD_TENDENCY_THRESHOLD:
        return "Not detected"

    if probability < pd_threshold:
        return "Borderline PD-like tendency"

    margin = probability - pd_threshold

    if margin < 0.10:
        return "Weak / borderline PD-like detected pattern"
    if margin < 0.25:
        return "Moderate PD-like detected pattern"
    return "Strong PD-like detected pattern"


def decide_pd_result(
    pd_probability: float,
    pd_detected: bool,
    pd_tendency: bool,
    reliability_level: str,
) -> tuple[str, str]:
    if reliability_level == "Low":
        return (
            "Inconclusive - insufficient reliable gait pattern duration",
            "The model produced a PD probability, but the video is too short or unreliable for a final PD-pattern decision.",
        )

    if pd_detected:
        return (
            "PD-like gait pattern detected",
            "The gait pattern matches the learned PD-like gait pattern.",
        )

    if pd_tendency:
        return (
            "No confirmed PD-like pattern detected, but borderline PD-like tendency is present",
            "The gait does not pass the PD detection threshold, but shows some PD-like pattern overlap.",
        )

    return (
        "No PD-like gait pattern detected",
        "The gait does not strongly match the learned PD-like pattern.",
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def load_feature_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature list not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pd_threshold() -> float:
    if not PD_SETTINGS_PATH.exists():
        return DEFAULT_PD_THRESHOLD

    try:
        settings = load_json(PD_SETTINGS_PATH)
        threshold = float(settings.get("final_threshold", DEFAULT_PD_THRESHOLD))

        if threshold <= 0:
            return DEFAULT_PD_THRESHOLD

        return threshold

    except Exception:
        return DEFAULT_PD_THRESHOLD


def predict_pd_model(feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not PD_MODEL_PATH.exists():
        raise FileNotFoundError(f"PD model not found: {PD_MODEL_PATH}")

    if not PD_FEATURES_PATH.exists():
        raise FileNotFoundError(f"PD feature list not found: {PD_FEATURES_PATH}")

    model = joblib.load(PD_MODEL_PATH)
    feature_cols = load_feature_list(PD_FEATURES_PATH)

    missing_features = [c for c in feature_cols if c not in feature_df.columns]

    if missing_features:
        raise ValueError(
            "Missing expected PD V14 model features. "
            f"First missing features: {missing_features[:20]}"
        )

    X = (
        feature_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    window_pred = model.predict(X)
    proba = model.predict_proba(X)

    class_list = list(model.classes_)

    if 1 not in class_list:
        raise ValueError("PD model does not contain positive class label 1.")

    positive_index = class_list.index(1)
    p_pd = proba[:, positive_index]

    return window_pred, p_pd


def predict_component3_pd(
    csv_path: Path,
    direction: str | None,
    fps_used: float | None,
) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    raw_df = load_pose_csv(csv_path)

    if direction is None:
        direction = infer_direction_from_filename(csv_path)

    if direction is None:
        direction, direction_delta_x, direction_confidence = estimate_direction_from_pose_df(raw_df)
    else:
        direction_delta_x = 0.0
        direction_confidence = "filename"

    if fps_used is None:
        fps_used = infer_fps_from_filename(csv_path)

    if fps_used is None:
        fps_used = 30.0

    clean_duration_sec = get_clean_duration_seconds(csv_path, fps_used)

    feature_df = preprocess_single_csv(csv_path, direction, fps_used)

    window_pred, p_pd = predict_pd_model(feature_df)

    pd_threshold = load_pd_threshold()

    pd_probability = float(np.mean(p_pd))

    pd_detected = pd_probability >= pd_threshold
    pd_tendency = (not pd_detected) and (pd_probability >= PD_TENDENCY_THRESHOLD)

    pd_window_ratio_default = float(np.mean(window_pred == 1))
    pd_window_ratio_detection = float(np.mean(p_pd >= pd_threshold))
    pd_window_ratio_tendency = float(np.mean(p_pd >= PD_TENDENCY_THRESHOLD))

    total_windows = int(len(feature_df))

    reliability_level, reliability_reasons = reliability_from_windows(
        total_windows=total_windows,
        clean_duration_sec=clean_duration_sec,
        direction_confidence=direction_confidence,
    )

    final_result, clinical_note = decide_pd_result(
        pd_probability=pd_probability,
        pd_detected=pd_detected,
        pd_tendency=pd_tendency,
        reliability_level=reliability_level,
    )

    return {
        "model_version": "PD V14 XGBoost precision-tuned",
        "model_path": str(PD_MODEL_PATH),
        "feature_list_path": str(PD_FEATURES_PATH),

        "csv_file": str(csv_path),
        "direction": direction,
        "direction_confidence": direction_confidence,
        "direction_delta_x": round(float(direction_delta_x), 5),
        "fps_used": float(fps_used),
        "target_fps": TARGET_FPS,
        "clean_duration_sec": round(clean_duration_sec, 3),
        "total_windows": total_windows,
        "padded": bool(feature_df["padded"].iloc[0]),
        "resampled_frames": int(feature_df["resampled_frames"].iloc[0]),

        "pd_probability": round(pd_probability, 4),
        "pd_tendency_threshold": PD_TENDENCY_THRESHOLD,
        "pd_detection_threshold": pd_threshold,
        "pd_detected": bool(pd_detected),
        "pd_borderline_tendency": bool(pd_tendency),
        "pd_pattern_strength": pd_pattern_strength(pd_probability, pd_threshold),

        "pd_window_ratio_default": round(pd_window_ratio_default, 4),
        "pd_window_ratio_tendency": round(pd_window_ratio_tendency, 4),
        "pd_window_ratio_detection": round(pd_window_ratio_detection, 4),

        "reliability_level": reliability_level,
        "reliability_reasons": reliability_reasons,

        "final_pd_result": final_result,
        "clinical_note": clinical_note,
    }


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Predict Component 3 PD-like gait pattern from a training_safe_landmarks CSV using final PD V14 model."
    )

    parser.add_argument(
        "csv_path",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Optional path to training_safe_landmarks.csv. "
            "If not provided, the latest CSV from 02_extracted_csv/training_safe/inference is used."
        ),
    )

    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest generated inference training-safe CSV automatically.",
    )

    parser.add_argument(
        "--direction",
        type=str,
        default=None,
        choices=["L2R", "R2L"],
        help="Optional walking direction. If not given, inferred from filename or pose motion.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional source FPS. If not given, inferred from filename; default 30 if unknown.",
    )

    args = parser.parse_args()

    if args.latest or args.csv_path is None:
        csv_path = find_latest_inference_csv()
        print("\nAuto-selected latest inference CSV:")
        print(csv_path)
    else:
        csv_path = Path(args.csv_path)

    result = predict_component3_pd(
        csv_path=csv_path,
        direction=args.direction,
        fps_used=args.fps,
    )

    print("\n======================================")
    print(" COMPONENT 3 PD PATTERN RESULT")
    print(" FINAL MODEL: PD V14")
    print("======================================")
    print(f"Model version:               {result['model_version']}")
    print(f"Model path:                  {result['model_path']}")
    print(f"Feature list:                {result['feature_list_path']}")
    print("--------------------------------------")
    print(f"CSV file:                    {result['csv_file']}")
    print(f"Direction:                   {result['direction']}")
    print(f"Direction confidence:        {result['direction_confidence']}")
    print(f"Direction delta x:           {result['direction_delta_x']}")
    print(f"FPS used:                    {result['fps_used']}")
    print(f"Target FPS:                  {result['target_fps']}")
    print(f"Clean duration:              {result['clean_duration_sec']}s")
    print(f"Total windows:               {result['total_windows']}")
    print(f"Padded:                      {result['padded']}")
    print(f"Resampled frames:            {result['resampled_frames']}")
    print("--------------------------------------")
    print(f"Reliability:                 {result['reliability_level']}")

    for reason in result["reliability_reasons"]:
        print(f"Reliability note:            {reason}")

    print("--------------------------------------")
    print(f"PD probability:              {result['pd_probability']}")
    print(f"PD tendency threshold:       {result['pd_tendency_threshold']}")
    print(f"PD detection threshold:      {result['pd_detection_threshold']}")
    print(f"PD detected:                 {result['pd_detected']}")
    print(f"PD borderline tendency:      {result['pd_borderline_tendency']}")
    print(f"PD pattern strength:         {result['pd_pattern_strength']}")
    print(f"PD window ratio default:     {result['pd_window_ratio_default']}")
    print(f"PD window ratio tendency:    {result['pd_window_ratio_tendency']}")
    print(f"PD window ratio detection:   {result['pd_window_ratio_detection']}")
    print("--------------------------------------")
    print(f"Final PD result:             {result['final_pd_result']}")
    print(f"Clinical note:               {result['clinical_note']}")
    print("======================================\n")


if __name__ == "__main__":
    main()
