from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INFERENCE_DIR = (
    PROJECT_ROOT
    / "02_extracted_csv"
    / "training_safe"
    / "inference"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "06_models"
    / "component_2_sca"
    / "component2_sca_model_v28_antalgic_hard_negative_holdout_eval.joblib"
)

FEATURE_LIST_PATH = (
    PROJECT_ROOT
    / "06_models"
    / "component_2_sca"
    / "component2_sca_feature_list_v28_antalgic_hard_negative.json"
)

SETTINGS_PATH = (
    PROJECT_ROOT
    / "06_models"
    / "component_2_sca"
    / "component2_sca_settings_v28_antalgic_hard_negative.json"
)

OUTPUT_REPORT_DIR = (
    PROJECT_ROOT
    / "07_reports"
    / "component_2_sca"
    / "inference"
)

OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ACTIVE SCA V28 ANTALGIC HARD-NEGATIVE DECISION SETTINGS
# ============================================================

MODEL_VERSION = "SCA V28 antalgic hard-negative model"

# Conservative V28 video-level rule selected after mining V27 antalgic
# false positives and auditing all labeled training-safe folders.
#
# Subject-holdout result:
# Accuracy  = 100.00%
# Precision = 100.00%
# Recall    = 100.00%
# F1        = 100.00%
# TN = 176, FP = 0, FN = 0, TP = 34
#
# Full labeled audit with this conservative deployment rule:
# antalgic FP = 0 / 100
# all labeled non-SCA FP = 0
ACTIVE_SCA_WINDOW_PROB_THRESHOLD = 0.45
ACTIVE_SCA_MIN_POSITIVE_WINDOW_RATIO = 0.25
ACTIVE_SCA_MIN_POSITIVE_WINDOW_COUNT = 4
ACTIVE_SCA_MIN_MAX_WINDOW_PROBABILITY = 0.20
ACTIVE_SCA_MIN_MEAN_PROBABILITY_FOR_CONFIRMED = 0.42
ACTIVE_SCA_MIN_TOTAL_WINDOWS_FOR_CONFIRMED = 5

# Short clips cannot always reach the normal confirmation count.
# Rescue only when the whole short clip is consistently high-confidence SCA.
ACTIVE_SCA_SHORT_CLIP_RESCUE_MIN_TOTAL_WINDOWS = 4
ACTIVE_SCA_SHORT_CLIP_RESCUE_MAX_TOTAL_WINDOWS = 7
ACTIVE_SCA_SHORT_CLIP_RESCUE_MIN_POSITIVE_RATIO = 1.00
ACTIVE_SCA_SHORT_CLIP_RESCUE_MIN_MEAN_PROBABILITY = 0.80
ACTIVE_SCA_SHORT_CLIP_RESCUE_MIN_MAX_PROBABILITY = 0.90
ACTIVE_SCA_SHORT_CLIP_RESCUE_MIN_TOP3_MEAN_PROBABILITY = 0.90

# Moderate short clips with repeated concentrated SCA evidence.
ACTIVE_SCA_LIMITED_CLIP_RESCUE_MIN_TOTAL_WINDOWS = 5
ACTIVE_SCA_LIMITED_CLIP_RESCUE_MAX_TOTAL_WINDOWS = 8
ACTIVE_SCA_LIMITED_CLIP_RESCUE_MIN_POSITIVE_COUNT = 3
ACTIVE_SCA_LIMITED_CLIP_RESCUE_MIN_POSITIVE_RATIO = 0.35
ACTIVE_SCA_LIMITED_CLIP_RESCUE_MIN_MEAN_PROBABILITY = 0.35
ACTIVE_SCA_LIMITED_CLIP_RESCUE_MIN_MAX_PROBABILITY = 0.80
ACTIVE_SCA_LIMITED_CLIP_RESCUE_MIN_TOP3_MEAN_PROBABILITY = 0.65

# Review/tendency is only for borderline repeated evidence.
# Isolated weak windows should stay negative.
ACTIVE_SCA_TENDENCY_MIN_POSITIVE_WINDOW_COUNT = 3
ACTIVE_SCA_TENDENCY_MIN_POSITIVE_WINDOW_RATIO = 0.25
ACTIVE_SCA_TENDENCY_MIN_MAX_WINDOW_PROBABILITY = 0.35
ACTIVE_SCA_TENDENCY_MIN_TOP3_MEAN_PROBABILITY = 0.50
ACTIVE_SCA_TENDENCY_MIN_MEAN_PROBABILITY = 0.27

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
# BASIC HELPERS
# ============================================================

def landmark_columns() -> list[str]:
    return [f"lm{i}_{axis}" for i in range(LANDMARK_COUNT) for axis in AXES]


def safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def safe_div(a, b, eps: float = 1e-6):
    return a / (np.abs(b) + eps)


def euclidean_2d(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def infer_direction_from_filename(path: Path) -> str:
    name = path.name.upper()
    if "_L2R_" in name:
        return "L2R"
    if "_R2L_" in name:
        return "R2L"
    return "UNKNOWN"


def infer_direction_from_pose_df(df: pd.DataFrame) -> tuple[str, str, float]:
    if "lm23_x" not in df.columns or "lm24_x" not in df.columns:
        return "UNKNOWN", "low", 0.0

    left_hip_x = pd.to_numeric(df["lm23_x"], errors="coerce")
    right_hip_x = pd.to_numeric(df["lm24_x"], errors="coerce")
    mid_hip_x = ((left_hip_x + right_hip_x) / 2.0).replace([np.inf, -np.inf], np.nan)
    mid_hip_x = mid_hip_x.interpolate(limit_direction="both").bfill().ffill()

    if len(mid_hip_x) < 10:
        return "UNKNOWN", "low", 0.0

    n = len(mid_hip_x)
    edge = max(3, int(n * 0.15))
    start_x = float(mid_hip_x.iloc[:edge].median())
    end_x = float(mid_hip_x.iloc[-edge:].median())
    delta_x = end_x - start_x

    if abs(delta_x) < 0.03:
        return "UNKNOWN", "low", float(delta_x)
    if delta_x > 0:
        return "L2R", "high", float(delta_x)
    return "R2L", "high", float(delta_x)


def infer_fps_from_filename(path: Path) -> float:
    name = path.name.lower()
    if "50fps" in name:
        return 50.0
    if "30fps" in name:
        return 30.0
    return 30.0


def extract_person_id(path: Path) -> str:
    match = re.search(r"P\d+", path.name.upper())
    if match:
        return match.group(0)
    return "UNKNOWN"


def find_latest_inference_csv() -> Path:
    if not DEFAULT_INFERENCE_DIR.exists():
        raise FileNotFoundError(f"Inference folder not found: {DEFAULT_INFERENCE_DIR}")

    csv_files = sorted(
        DEFAULT_INFERENCE_DIR.glob("*_training_safe_landmarks.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No *_training_safe_landmarks.csv files found in {DEFAULT_INFERENCE_DIR}"
        )

    return csv_files[0]


# ============================================================
# LOAD POSE CSV
# ============================================================

def load_pose_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

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
        raise ValueError(
            f"Missing required landmark columns. First missing columns: {missing[:20]}"
        )

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

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
    if "timestamp_ms" in df.columns and len(df) >= 2:
        t = (
            pd.to_numeric(df["timestamp_ms"], errors="coerce")
            .to_numpy(dtype=np.float64)
            / 1000.0
        )
        t = t - t[0]
        if np.all(np.isfinite(t)) and t[-1] > t[0]:
            return t

    return np.arange(len(df), dtype=np.float64) / float(fps_used)


# ============================================================
# PREPROCESSING
# ============================================================

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
    seq = seq.copy()
    direction = str(direction).upper().strip()
    if DIRECTION_STANDARDIZE and direction == "R2L":
        seq[:, :, 0] *= -1.0
    return seq


def make_windows(seq: np.ndarray, window_size: int, stride: int) -> list[np.ndarray]:
    windows = []
    if len(seq) < window_size:
        return windows

    for start in range(0, len(seq) - window_size + 1, stride):
        windows.append(seq[start:start + window_size])

    return windows


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def aggregate_features(window: np.ndarray) -> dict:
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


def add_sca_biomech_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
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
        lx = safe_col(out, f"mean_lm{left_lm}_x")
        ly = safe_col(out, f"mean_lm{left_lm}_y")
        rx = safe_col(out, f"mean_lm{right_lm}_x")
        ry = safe_col(out, f"mean_lm{right_lm}_y")
        bio[f"bio_sca_{name}_width_mean"] = euclidean_2d(lx, ly, rx, ry)

    bio_df = pd.DataFrame(bio, index=out.index)

    bio_df["bio_sca_ankle_to_hip_width_ratio"] = safe_div(
        bio_df["bio_sca_ankle_width_mean"],
        bio_df["bio_sca_hip_width_mean"],
    )
    bio_df["bio_sca_foot_to_hip_width_ratio"] = safe_div(
        bio_df["bio_sca_foot_width_mean"],
        bio_df["bio_sca_hip_width_mean"],
    )
    bio_df["bio_sca_knee_to_hip_width_ratio"] = safe_div(
        bio_df["bio_sca_knee_width_mean"],
        bio_df["bio_sca_hip_width_mean"],
    )
    bio_df["bio_sca_shoulder_to_hip_width_ratio"] = safe_div(
        bio_df["bio_sca_shoulder_width_mean"],
        bio_df["bio_sca_hip_width_mean"],
    )

    for left_lm, right_lm, name in body_pairs:
        left_rx = safe_col(out, f"range_lm{left_lm}_x")
        left_ry = safe_col(out, f"range_lm{left_lm}_y")
        right_rx = safe_col(out, f"range_lm{right_lm}_x")
        right_ry = safe_col(out, f"range_lm{right_lm}_y")

        left_range = np.sqrt(left_rx ** 2 + left_ry ** 2)
        right_range = np.sqrt(right_rx ** 2 + right_ry ** 2)

        bio_df[f"bio_sca_left_{name}_motion_range"] = left_range
        bio_df[f"bio_sca_right_{name}_motion_range"] = right_range
        bio_df[f"bio_sca_{name}_motion_mean"] = (left_range + right_range) / 2.0
        bio_df[f"bio_sca_{name}_motion_asymmetry"] = np.abs(left_range - right_range)
        bio_df[f"bio_sca_{name}_motion_ratio"] = safe_div(left_range, right_range)

    for left_lm, right_lm, name in body_pairs:
        left_sx = safe_col(out, f"std_lm{left_lm}_x")
        left_sy = safe_col(out, f"std_lm{left_lm}_y")
        right_sx = safe_col(out, f"std_lm{right_lm}_x")
        right_sy = safe_col(out, f"std_lm{right_lm}_y")

        left_std = np.sqrt(left_sx ** 2 + left_sy ** 2)
        right_std = np.sqrt(right_sx ** 2 + right_sy ** 2)

        bio_df[f"bio_sca_left_{name}_variability"] = left_std
        bio_df[f"bio_sca_right_{name}_variability"] = right_std
        bio_df[f"bio_sca_{name}_variability_mean"] = (left_std + right_std) / 2.0
        bio_df[f"bio_sca_{name}_variability_asymmetry"] = np.abs(left_std - right_std)
        bio_df[f"bio_sca_{name}_variability_ratio"] = safe_div(left_std, right_std)

    for left_lm, right_lm, name in body_pairs:
        left_vx = safe_col(out, f"mean_abs_vel_lm{left_lm}_x")
        left_vy = safe_col(out, f"mean_abs_vel_lm{left_lm}_y")
        right_vx = safe_col(out, f"mean_abs_vel_lm{right_lm}_x")
        right_vy = safe_col(out, f"mean_abs_vel_lm{right_lm}_y")

        left_vel = np.sqrt(left_vx ** 2 + left_vy ** 2)
        right_vel = np.sqrt(right_vx ** 2 + right_vy ** 2)

        bio_df[f"bio_sca_left_{name}_mean_abs_velocity"] = left_vel
        bio_df[f"bio_sca_right_{name}_mean_abs_velocity"] = right_vel
        bio_df[f"bio_sca_{name}_velocity_mean"] = (left_vel + right_vel) / 2.0
        bio_df[f"bio_sca_{name}_velocity_asymmetry"] = np.abs(left_vel - right_vel)
        bio_df[f"bio_sca_{name}_velocity_ratio"] = safe_div(left_vel, right_vel)

        left_svx = safe_col(out, f"std_vel_lm{left_lm}_x")
        left_svy = safe_col(out, f"std_vel_lm{left_lm}_y")
        right_svx = safe_col(out, f"std_vel_lm{right_lm}_x")
        right_svy = safe_col(out, f"std_vel_lm{right_lm}_y")

        left_std_vel = np.sqrt(left_svx ** 2 + left_svy ** 2)
        right_std_vel = np.sqrt(right_svx ** 2 + right_svy ** 2)

        bio_df[f"bio_sca_left_{name}_std_velocity"] = left_std_vel
        bio_df[f"bio_sca_right_{name}_std_velocity"] = right_std_vel
        bio_df[f"bio_sca_{name}_std_velocity_mean"] = (left_std_vel + right_std_vel) / 2.0
        bio_df[f"bio_sca_{name}_std_velocity_asymmetry"] = np.abs(left_std_vel - right_std_vel)
        bio_df[f"bio_sca_{name}_std_velocity_ratio"] = safe_div(left_std_vel, right_std_vel)

    shoulder_var = (
        np.sqrt(safe_col(out, "std_lm11_x") ** 2 + safe_col(out, "std_lm11_y") ** 2)
        + np.sqrt(safe_col(out, "std_lm12_x") ** 2 + safe_col(out, "std_lm12_y") ** 2)
    ) / 2.0

    hip_var = (
        np.sqrt(safe_col(out, "std_lm23_x") ** 2 + safe_col(out, "std_lm23_y") ** 2)
        + np.sqrt(safe_col(out, "std_lm24_x") ** 2 + safe_col(out, "std_lm24_y") ** 2)
    ) / 2.0

    bio_df["bio_sca_trunk_variability_proxy"] = (shoulder_var + hip_var) / 2.0
    bio_df["bio_sca_shoulder_to_hip_variability_ratio"] = safe_div(shoulder_var, hip_var)

    bio_df["bio_sca_lower_limb_motion_asymmetry_score"] = (
        bio_df["bio_sca_knee_motion_asymmetry"]
        + bio_df["bio_sca_ankle_motion_asymmetry"]
        + bio_df["bio_sca_heel_motion_asymmetry"]
        + bio_df["bio_sca_foot_motion_asymmetry"]
    ) / 4.0

    bio_df["bio_sca_lower_limb_variability_score"] = (
        bio_df["bio_sca_knee_variability_mean"]
        + bio_df["bio_sca_ankle_variability_mean"]
        + bio_df["bio_sca_heel_variability_mean"]
        + bio_df["bio_sca_foot_variability_mean"]
    ) / 4.0

    bio_df["bio_sca_lower_limb_velocity_irregularity_score"] = (
        bio_df["bio_sca_knee_std_velocity_mean"]
        + bio_df["bio_sca_ankle_std_velocity_mean"]
        + bio_df["bio_sca_heel_std_velocity_mean"]
        + bio_df["bio_sca_foot_std_velocity_mean"]
    ) / 4.0

    bio_df["bio_sca_width_instability_score"] = (
        bio_df["bio_sca_ankle_to_hip_width_ratio"]
        + bio_df["bio_sca_foot_to_hip_width_ratio"]
        + bio_df["bio_sca_knee_to_hip_width_ratio"]
    ) / 3.0

    bio_df["bio_sca_composite_ataxia_proxy_score"] = (
        bio_df["bio_sca_lower_limb_motion_asymmetry_score"]
        + bio_df["bio_sca_lower_limb_variability_score"]
        + bio_df["bio_sca_lower_limb_velocity_irregularity_score"]
        + bio_df["bio_sca_width_instability_score"]
        + bio_df["bio_sca_trunk_variability_proxy"]
    ) / 5.0

    bio_df["bio_sca_lower_limb_speed_proxy"] = (
        bio_df["bio_sca_knee_velocity_mean"]
        + bio_df["bio_sca_ankle_velocity_mean"]
        + bio_df["bio_sca_heel_velocity_mean"]
        + bio_df["bio_sca_foot_velocity_mean"]
    ) / 4.0

    bio_df["bio_sca_speed_irregularity_ratio"] = safe_div(
        bio_df["bio_sca_lower_limb_velocity_irregularity_score"],
        bio_df["bio_sca_lower_limb_speed_proxy"],
    )

    bio_df["bio_sca_slow_instability_interaction"] = safe_div(
        bio_df["bio_sca_composite_ataxia_proxy_score"],
        bio_df["bio_sca_lower_limb_speed_proxy"] + 0.05,
    )

    bio_df["bio_sca_slow_width_instability_interaction"] = safe_div(
        bio_df["bio_sca_width_instability_score"],
        bio_df["bio_sca_lower_limb_speed_proxy"] + 0.05,
    )

    bio_df = bio_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = pd.concat([out, bio_df], axis=1).copy()
    return out


# ============================================================
# BUILD WINDOW FEATURE DATAFRAME
# ============================================================

def build_window_feature_dataframe(
    csv_path: Path,
    direction_override: str | None = None,
    fps_override: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    filename_direction = infer_direction_from_filename(csv_path)
    inferred_fps = infer_fps_from_filename(csv_path)
    fps_used = float(fps_override) if fps_override is not None else inferred_fps
    fps_source = "manual" if fps_override is not None else "filename_or_default"
    person_id = extract_person_id(csv_path)

    raw_df = load_pose_csv(csv_path)
    auto_direction, direction_confidence, direction_delta_x = infer_direction_from_pose_df(raw_df)

    clean_direction_override = str(direction_override).upper().strip() if direction_override else None
    if clean_direction_override in ["L2R", "R2L"]:
        direction = clean_direction_override
        direction_source = "manual"
        direction_confidence = "manual"
    elif filename_direction in ["L2R", "R2L"]:
        direction = filename_direction
        direction_source = "filename"
    else:
        direction = auto_direction
        direction_source = "auto_pose"

    seq = sequence_from_df(raw_df)
    time_sec = get_time_seconds(raw_df, fps_used)
    seq = resample_sequence(seq, time_sec, TARGET_FPS)

    padded = False
    if len(seq) < WINDOW_SIZE:
        if PAD_SHORT_CLIPS:
            seq = edge_pad_to_window(seq, WINDOW_SIZE)
            padded = True
        else:
            raise ValueError(
                f"Not enough frames for one window. Frames={len(seq)}, needed={WINDOW_SIZE}"
            )

    resampled_frames = len(seq)
    clean_duration_sec = resampled_frames / TARGET_FPS

    seq = normalize_pose(seq)
    seq = standardize_direction(seq, direction)
    windows = make_windows(seq, WINDOW_SIZE, STRIDE)

    if not windows:
        raise RuntimeError("No windows were created from the input CSV.")

    rows = []
    for window_id, window in enumerate(windows):
        feat = aggregate_features(window)
        feat["window_id"] = window_id
        feat["fps_used"] = fps_used
        feat["padded"] = int(padded)
        feat["resampled_frames"] = resampled_frames
        rows.append(feat)

    feature_df = pd.DataFrame(rows)
    feature_df = add_sca_biomech_features(feature_df)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    metadata = {
        "csv_file": str(csv_path),
        "direction": direction,
        "direction_source": direction_source,
        "direction_confidence": direction_confidence,
        "direction_delta_x": direction_delta_x,
        "fps_used": fps_used,
        "fps_source": fps_source,
        "target_fps": TARGET_FPS,
        "person_id": person_id,
        "padded": padded,
        "resampled_frames": resampled_frames,
        "clean_duration_sec": clean_duration_sec,
        "total_windows": len(feature_df),
    }

    return feature_df, metadata


# ============================================================
# MODEL LOADING + PREDICTION
# ============================================================

def load_active_sca_model_and_features():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"SCA model not found: {MODEL_PATH}")

    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(f"SCA feature list not found: {FEATURE_LIST_PATH}")

    model = joblib.load(MODEL_PATH)

    with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    settings = {}
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except Exception:
            settings = {}

    return model, feature_cols, settings


def predict_active_sca_windows(feature_df: pd.DataFrame) -> np.ndarray:
    model, feature_cols, _ = load_active_sca_model_and_features()

    missing = [c for c in feature_cols if c not in feature_df.columns]
    if missing:
        raise ValueError(
            "Missing expected SCA model features. "
            f"First missing features: {missing[:30]}"
        )

    X = (
        feature_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    class_list = list(model.classes_)
    if 1 not in class_list:
        raise ValueError("SCA model does not contain positive class label 1.")

    positive_index = class_list.index(1)
    probabilities = model.predict_proba(X)[:, positive_index]
    return probabilities


# ============================================================
# FINAL DECISION
# ============================================================

def reliability_from_duration(total_windows: int, duration_sec: float) -> tuple[str, str]:
    if total_windows <= 0 or duration_sec < 2.0:
        return "Low", "Not enough clean walking data for reliable SCA instability screening."

    if total_windows < ACTIVE_SCA_MIN_TOTAL_WINDOWS_FOR_CONFIRMED or duration_sec < 6.0:
        return "Moderate", "Short clip. Use a longer walking clip for final SCA confirmation."

    return "High", "Enough clean walking data for SCA v28 screening."


def decide_active_sca_result(probabilities: np.ndarray, reliability: str) -> dict:
    probabilities = np.asarray(probabilities, dtype=float)

    total_windows = int(len(probabilities))
    positive_mask = probabilities >= ACTIVE_SCA_WINDOW_PROB_THRESHOLD

    positive_count = int(np.sum(positive_mask))
    positive_ratio = float(positive_count / total_windows) if total_windows > 0 else 0.0

    mean_probability = float(np.mean(probabilities)) if total_windows > 0 else 0.0
    max_probability = float(np.max(probabilities)) if total_windows > 0 else 0.0

    top_n = min(3, total_windows)
    top3_mean_probability = float(np.mean(np.sort(probabilities)[-top_n:])) if top_n > 0 else 0.0

    positive_count_passed = positive_count >= ACTIVE_SCA_MIN_POSITIVE_WINDOW_COUNT
    positive_ratio_passed = positive_ratio >= ACTIVE_SCA_MIN_POSITIVE_WINDOW_RATIO
    max_probability_passed = max_probability >= ACTIVE_SCA_MIN_MAX_WINDOW_PROBABILITY
    mean_probability_passed = (
        mean_probability >= ACTIVE_SCA_MIN_MEAN_PROBABILITY_FOR_CONFIRMED
    )
    total_windows_passed = total_windows >= ACTIVE_SCA_MIN_TOTAL_WINDOWS_FOR_CONFIRMED

    # Normal strict V28 antalgic hard-negative confirmation.
    strict_confirmed = (
        reliability != "Low"
        and total_windows_passed
        and positive_count_passed
        and positive_ratio_passed
        and max_probability_passed
        and mean_probability_passed
    )

    short_clip_high_confidence_rescue = (
        reliability != "Low"
        and ACTIVE_SCA_SHORT_CLIP_RESCUE_MIN_TOTAL_WINDOWS
        <= total_windows
        <= ACTIVE_SCA_SHORT_CLIP_RESCUE_MAX_TOTAL_WINDOWS
        and positive_ratio >= ACTIVE_SCA_SHORT_CLIP_RESCUE_MIN_POSITIVE_RATIO
        and mean_probability >= ACTIVE_SCA_SHORT_CLIP_RESCUE_MIN_MEAN_PROBABILITY
        and max_probability >= ACTIVE_SCA_SHORT_CLIP_RESCUE_MIN_MAX_PROBABILITY
        and top3_mean_probability >= ACTIVE_SCA_SHORT_CLIP_RESCUE_MIN_TOP3_MEAN_PROBABILITY
    )

    limited_clip_repeated_rescue = (
        reliability != "Low"
        and ACTIVE_SCA_LIMITED_CLIP_RESCUE_MIN_TOTAL_WINDOWS
        <= total_windows
        <= ACTIVE_SCA_LIMITED_CLIP_RESCUE_MAX_TOTAL_WINDOWS
        and positive_count >= ACTIVE_SCA_LIMITED_CLIP_RESCUE_MIN_POSITIVE_COUNT
        and positive_ratio >= ACTIVE_SCA_LIMITED_CLIP_RESCUE_MIN_POSITIVE_RATIO
        and mean_probability >= ACTIVE_SCA_LIMITED_CLIP_RESCUE_MIN_MEAN_PROBABILITY
        and max_probability >= ACTIVE_SCA_LIMITED_CLIP_RESCUE_MIN_MAX_PROBABILITY
        and top3_mean_probability >= ACTIVE_SCA_LIMITED_CLIP_RESCUE_MIN_TOP3_MEAN_PROBABILITY
    )

    sca_detected = (
        strict_confirmed
        or short_clip_high_confidence_rescue
        or limited_clip_repeated_rescue
    )
    short_clip_unconfirmed = (
        reliability != "Low"
        and not total_windows_passed
        and total_windows > 0
        and positive_count > 0
    )

    borderline_repeated_evidence = (
        positive_count >= ACTIVE_SCA_TENDENCY_MIN_POSITIVE_WINDOW_COUNT
        and positive_ratio >= ACTIVE_SCA_TENDENCY_MIN_POSITIVE_WINDOW_RATIO
        and max_probability >= ACTIVE_SCA_TENDENCY_MIN_MAX_WINDOW_PROBABILITY
    )
    high_probability_review_evidence = (
        top3_mean_probability >= ACTIVE_SCA_TENDENCY_MIN_TOP3_MEAN_PROBABILITY
        or mean_probability >= ACTIVE_SCA_TENDENCY_MIN_MEAN_PROBABILITY
    )

    review_evidence_detected = (
        reliability != "Low"
        and not sca_detected
        and total_windows_passed
        and (borderline_repeated_evidence or high_probability_review_evidence)
    )

    # Keep borderline evidence available for audits, but keep the simple
    # screening decision binary: confirmed threshold pass means SCA-like,
    # otherwise it is not reported as SCA-like.
    tendency_detected = False

    if strict_confirmed:
        detection_source = "confirmed_sca_v28_antalgic_hard_negative"
        pattern_strength = "Confirmed SCA-like gait pattern"
        final_result = "Confirmed SCA-like gait pattern detected"
        clinical_note = (
            "The SCA v28 version found repeated SCA-like gait evidence across the walking clip."
        )

    elif short_clip_high_confidence_rescue:
        detection_source = "short_clip_high_confidence_rescue"
        pattern_strength = "Confirmed SCA-like gait pattern in a short clip"
        final_result = "Confirmed SCA-like gait pattern detected"
        clinical_note = (
            "The SCA v28 version found strong SCA-like evidence. A longer repeat clip is still useful "
            "for stronger screening reliability."
        )

    elif limited_clip_repeated_rescue:
        detection_source = "limited_clip_repeated_sca_rescue"
        pattern_strength = "Confirmed SCA-like gait pattern in a limited clip"
        final_result = "Confirmed SCA-like gait pattern detected"
        clinical_note = (
            "The SCA v28 version found concentrated repeated SCA-like evidence in a limited clip."
        )

    elif short_clip_unconfirmed:
        detection_source = "short_clip_unconfirmed"
        pattern_strength = "No confirmed SCA-like gait pattern; clip is too short for confirmation"
        final_result = "No confirmed SCA-like gait pattern detected"
        clinical_note = (
            "The clip has fewer walking windows than the SCA v28 confirmation guard requires. "
            "Use a longer walking clip if clinical concern remains."
        )

    else:
        detection_source = "negative"
        pattern_strength = "No confirmed SCA-like gait pattern"
        final_result = "No confirmed SCA-like gait pattern detected"
        clinical_note = "The SCA v28 version did not find repeated SCA-like gait evidence."

    if reliability == "Low":
        detection_source = "low_reliability"
        pattern_strength = "Low reliability"
        final_result = "Result not reliable due to insufficient clean walking data"
        clinical_note = "The input does not contain enough clean walking data for reliable SCA v28 screening."
        sca_detected = False
        tendency_detected = False

    return {
        "total_windows": total_windows,
        "mean_probability": mean_probability,
        "max_probability": max_probability,
        "top3_mean_probability": top3_mean_probability,
        "positive_count": positive_count,
        "positive_ratio": positive_ratio,
        "positive_count_passed": bool(positive_count_passed),
        "positive_ratio_passed": bool(positive_ratio_passed),
        "max_probability_passed": bool(max_probability_passed),
        "mean_probability_passed": bool(mean_probability_passed),
        "total_windows_passed": bool(total_windows_passed),
        "min_total_windows_required": int(ACTIVE_SCA_MIN_TOTAL_WINDOWS_FOR_CONFIRMED),
        "strict_confirmed": bool(strict_confirmed),
        "review_evidence_detected_internal": bool(review_evidence_detected),
        "short_clip_high_confidence_rescue": bool(short_clip_high_confidence_rescue),
        "limited_clip_repeated_rescue": bool(limited_clip_repeated_rescue),
        "detection_source": detection_source,
        "sca_detected": bool(sca_detected),
        "sca_tendency": bool(tendency_detected),
        "pattern_strength": pattern_strength,
        "final_result": final_result,
        "clinical_note": clinical_note,
    }


def make_window_report(feature_df: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    report = pd.DataFrame({
        "window_id": feature_df["window_id"].astype(int).to_numpy(),
        "sca_v28_probability": probabilities,
    })

    report["positive_window"] = (
        report["sca_v28_probability"] >= ACTIVE_SCA_WINDOW_PROB_THRESHOLD
    ).astype(int)

    report["rank_highest_probability"] = (
        report["sca_v28_probability"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    extra_cols = [
        "bio_sca_lower_limb_speed_proxy",
        "bio_sca_width_instability_score",
        "bio_sca_composite_ataxia_proxy_score",
        "bio_sca_lower_limb_variability_score",
        "bio_sca_lower_limb_velocity_irregularity_score",
        "bio_sca_speed_irregularity_ratio",
        "bio_sca_lower_limb_motion_asymmetry_score",
        "bio_sca_trunk_variability_proxy",
        "bio_sca_slow_instability_interaction",
        "bio_sca_slow_width_instability_interaction",
    ]

    for col in extra_cols:
        if col in feature_df.columns:
            report[col] = pd.to_numeric(feature_df[col], errors="coerce").fillna(0.0).to_numpy()

    return report.sort_values("window_id").reset_index(drop=True)


def predict_component2_sca_v28_antalgic_hard_negative(
    csv_path: Path,
    save_report: bool = True,
    direction: str | None = None,
    fps: float | None = None,
) -> dict:
    feature_df, metadata = build_window_feature_dataframe(
        csv_path=csv_path,
        direction_override=direction,
        fps_override=fps,
    )
    probabilities = predict_active_sca_windows(feature_df)

    reliability, reliability_note = reliability_from_duration(
        total_windows=metadata["total_windows"],
        duration_sec=metadata["clean_duration_sec"],
    )

    decision = decide_active_sca_result(probabilities, reliability)
    window_report = make_window_report(feature_df, probabilities)

    if save_report:
        out_name = csv_path.stem.replace("_training_safe_landmarks", "")
        out_path = OUTPUT_REPORT_DIR / f"{out_name}_sca_v28_antalgic_hard_negative_window_inference_report.csv"
        if len(str(out_path)) >= 240:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", out_name).strip("._")
            digest = hashlib.sha1(out_name.encode("utf-8")).hexdigest()[:10]
            out_path = OUTPUT_REPORT_DIR / f"{safe_name[:24]}_{digest}_sca_v28_report.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        window_report.to_csv(out_path, index=False)
        window_report_path = str(out_path)
    else:
        window_report_path = ""

    result = {
        "model_version": MODEL_VERSION,
        "model_path": str(MODEL_PATH),
        "feature_list_path": str(FEATURE_LIST_PATH),
        **metadata,
        "reliability": reliability,
        "reliability_note": reliability_note,
        **decision,
        "window_report_path": window_report_path,
    }

    return result


def predict_component2_sca_v27_final_normal_boundary(
    csv_path: Path,
    save_report: bool = True,
    direction: str | None = None,
    fps: float | None = None,
) -> dict:
    return predict_component2_sca_v28_antalgic_hard_negative(
        csv_path=csv_path,
        save_report=save_report,
        direction=direction,
        fps=fps,
    )


def predict_component2_sca_v11_precision_retuned(csv_path: Path, save_report: bool = True) -> dict:
    return predict_component2_sca_v28_antalgic_hard_negative(csv_path=csv_path, save_report=save_report)


def predict_component2_sca_v23_final(csv_path: Path, save_report: bool = True) -> dict:
    return predict_component2_sca_v28_antalgic_hard_negative(csv_path=csv_path, save_report=save_report)


def predict_component2_sca_v25_normal_boundary(csv_path: Path, save_report: bool = True) -> dict:
    return predict_component2_sca_v28_antalgic_hard_negative(csv_path=csv_path, save_report=save_report)


# ============================================================
# PRINTING
# ============================================================

def print_result(result: dict):
    print()
    print("================================================")
    print(" COMPONENT 2A SCA SCREENING RESULT")
    print(" ACTIVE MODEL: SCA V28 ANTALGIC HARD-NEGATIVE")
    print("================================================")
    print(f"Final SCA result:              {result['final_result']}")
    print(f"SCA detected:                  {result['sca_detected']}")
    print(f"SCA tendency:                  {result['sca_tendency']}")
    print(f"Pattern strength:              {result['pattern_strength']}")
    print("------------------------------------------------")
    print(f"Confidence level:              {result['reliability']}")
    print(f"Confidence note:               {result['reliability_note']}")
    print(f"Mean SCA probability:          {round(result['mean_probability'], 4)}")
    print(f"Max SCA probability:           {round(result['max_probability'], 4)}")
    print(f"Top-3 mean probability:        {round(result['top3_mean_probability'], 4)}")
    print(f"Positive windows:              {result['positive_count']} / {result['total_windows']}")
    print(f"Positive window ratio:         {round(result['positive_ratio'], 4)}")
    print("------------------------------------------------")
    print(f"CSV file:                      {result['csv_file']}")
    print(f"Direction:                     {result['direction']}")
    print(f"Direction confidence:          {result.get('direction_confidence', 'N/A')}")
    print(f"Clean duration:                {round(result['clean_duration_sec'], 2)}s")
    print(f"Total windows:                 {result['total_windows']}")
    print("------------------------------------------------")
    print(f"Clinical note:                 {result['clinical_note']}")

    if result.get("window_report_path"):
        print(f"Window report saved:           {result['window_report_path']}")

    print("================================================")
    print()


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict Component 2A SCA using the SCA v28 antalgic hard-negative model."
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to training_safe_landmarks.csv. If omitted, latest inference CSV is used.",
    )

    parser.add_argument(
        "--no-save-report",
        action="store_true",
        help="Do not save the window-level inference report CSV.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = find_latest_inference_csv()
        print()
        print("Auto-selected latest inference CSV:")
        print(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    result = predict_component2_sca_v28_antalgic_hard_negative(
        csv_path=csv_path,
        save_report=not args.no_save_report,
    )

    print_result(result)


if __name__ == "__main__":
    main()
