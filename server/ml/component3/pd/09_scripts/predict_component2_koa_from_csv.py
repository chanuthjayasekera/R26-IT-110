from pathlib import Path
import argparse
import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import numpy as np
import pandas as pd
import warnings


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "06_models" / "component_2_koa"

# ============================================================
# FINAL KOA V14 YOUTUBE-ATAXIA ONCE MODEL FILES
# ============================================================

KOA_MODEL_PATH = MODEL_DIR / "component2_koa_v14_youtube_ataxia_once_holdout_eval_model.joblib"
KOA_FEATURES_PATH = MODEL_DIR / "component2_koa_v14_youtube_ataxia_once_feature_list.json"
KOA_SETTINGS_PATH = MODEL_DIR / "component2_koa_v14_youtube_ataxia_once_settings.json"

# Auto-detect latest CSV created by run_component1_clinical_video.py
INFERENCE_TRAINING_SAFE_DIR = PROJECT_ROOT / "02_extracted_csv" / "training_safe" / "inference"


# ============================================================
# FINAL KOA V14 FALLBACK THRESHOLDS
# ============================================================

KOA_THRESHOLD = 0.18
KOA_TENDENCY_THRESHOLD = 0.30


# ============================================================
# PREPROCESSING SETTINGS - MUST MATCH TRAINING
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
    """
    Automatically find the latest training_safe CSV created by
    run_component1_clinical_video.py.
    """
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
    """
    Estimate walking direction using mid-hip x movement.
    If x increases: L2R
    If x decreases: R2L
    """
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


# ============================================================
# CSV → FEATURES
# ============================================================

def load_pose_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Important compatibility:
    # If extractor created lm*_visibility, convert to lm*_vis.
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
    if "timestamp_ms" in df.columns and len(df) >= 2:
        t = pd.to_numeric(df["timestamp_ms"], errors="coerce").to_numpy(dtype=np.float64) / 1000.0
        t = t - t[0]

        if np.all(np.isfinite(t)) and t[-1] > t[0]:
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
    Same normalization as training:
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
    Same direction standardization as training:
    R2L sequences are flipped on normalized x-axis.
    """
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


def aggregate_features(window: np.ndarray) -> dict:
    """
    Base V3 feature set:
    132 base values × 7 feature groups = 924 features.

    Feature groups:
    - mean
    - std
    - min
    - max
    - range
    - mean_abs_velocity
    - std_velocity
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

    windows = make_windows(seq, WINDOW_SIZE, STRIDE)

    if len(windows) == 0:
        raise ValueError("No windows created from this CSV.")

    rows = []

    for window_id, window in enumerate(windows):
        feat = aggregate_features(window)
        feat["window_id"] = window_id
        feat["padded"] = padded
        feat["resampled_frames"] = len(seq)
        rows.append(feat)

    feature_df = pd.DataFrame(rows)

    # KOA V3 uses only inference-safe lower-body, gait, and biomechanical features.
    feature_df = add_koa_biomech_features(feature_df)
    feature_df = add_koa_gait_features(feature_df)

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
# KOA BIOMECHANICAL FEATURES - MUST MATCH TRAINING
# ============================================================

def safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def safe_div(a, b, eps: float = 1e-6):
    return a / (np.abs(b) + eps)


def euclidean_2d(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def angle_2d(ax, ay, bx, by, cx, cy):
    """
    2D angle ABC in degrees.
    Used as a knee angle proxy:
        hip-knee-ankle angle
    """
    v1x = ax - bx
    v1y = ay - by
    v2x = cx - bx
    v2y = cy - by

    dot = (v1x * v2x) + (v1y * v2y)
    n1 = np.sqrt(v1x ** 2 + v1y ** 2)
    n2 = np.sqrt(v2x ** 2 + v2y ** 2)

    cosang = dot / ((n1 * n2) + 1e-6)
    cosang = np.clip(cosang, -1.0, 1.0)

    return np.degrees(np.arccos(cosang))


def add_koa_biomech_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the same 67 KOA biomechanical proxy features used in training.
    """
    out = df.copy()
    new_cols = []

    lhx = safe_col(out, "mean_lm23_x")
    lhy = safe_col(out, "mean_lm23_y")
    rhx = safe_col(out, "mean_lm24_x")
    rhy = safe_col(out, "mean_lm24_y")

    lkx = safe_col(out, "mean_lm25_x")
    lky = safe_col(out, "mean_lm25_y")
    rkx = safe_col(out, "mean_lm26_x")
    rky = safe_col(out, "mean_lm26_y")

    lax = safe_col(out, "mean_lm27_x")
    lay = safe_col(out, "mean_lm27_y")
    rax = safe_col(out, "mean_lm28_x")
    ray = safe_col(out, "mean_lm28_y")

    lfx = safe_col(out, "mean_lm31_x")
    lfy = safe_col(out, "mean_lm31_y")
    rfx = safe_col(out, "mean_lm32_x")
    rfy = safe_col(out, "mean_lm32_y")

    out["bio_left_knee_angle_mean_proxy"] = angle_2d(lhx, lhy, lkx, lky, lax, lay)
    out["bio_right_knee_angle_mean_proxy"] = angle_2d(rhx, rhy, rkx, rky, rax, ray)
    out["bio_knee_angle_asymmetry_proxy"] = np.abs(
        out["bio_left_knee_angle_mean_proxy"] - out["bio_right_knee_angle_mean_proxy"]
    )

    new_cols += [
        "bio_left_knee_angle_mean_proxy",
        "bio_right_knee_angle_mean_proxy",
        "bio_knee_angle_asymmetry_proxy",
    ]

    out["bio_left_thigh_len_proxy"] = euclidean_2d(lhx, lhy, lkx, lky)
    out["bio_right_thigh_len_proxy"] = euclidean_2d(rhx, rhy, rkx, rky)
    out["bio_left_shank_len_proxy"] = euclidean_2d(lkx, lky, lax, lay)
    out["bio_right_shank_len_proxy"] = euclidean_2d(rkx, rky, rax, ray)

    out["bio_thigh_len_asymmetry_proxy"] = np.abs(
        out["bio_left_thigh_len_proxy"] - out["bio_right_thigh_len_proxy"]
    )
    out["bio_shank_len_asymmetry_proxy"] = np.abs(
        out["bio_left_shank_len_proxy"] - out["bio_right_shank_len_proxy"]
    )

    new_cols += [
        "bio_left_thigh_len_proxy",
        "bio_right_thigh_len_proxy",
        "bio_left_shank_len_proxy",
        "bio_right_shank_len_proxy",
        "bio_thigh_len_asymmetry_proxy",
        "bio_shank_len_asymmetry_proxy",
    ]

    out["bio_ankle_width_mean"] = euclidean_2d(lax, lay, rax, ray)
    out["bio_foot_width_mean"] = euclidean_2d(lfx, lfy, rfx, rfy)
    out["bio_knee_width_mean"] = euclidean_2d(lkx, lky, rkx, rky)
    out["bio_hip_width_mean"] = euclidean_2d(lhx, lhy, rhx, rhy)

    out["bio_ankle_to_hip_width_ratio"] = safe_div(
        out["bio_ankle_width_mean"],
        out["bio_hip_width_mean"],
    )
    out["bio_knee_to_hip_width_ratio"] = safe_div(
        out["bio_knee_width_mean"],
        out["bio_hip_width_mean"],
    )

    new_cols += [
        "bio_ankle_width_mean",
        "bio_foot_width_mean",
        "bio_knee_width_mean",
        "bio_hip_width_mean",
        "bio_ankle_to_hip_width_ratio",
        "bio_knee_to_hip_width_ratio",
    ]

    for lm_left, lm_right, name in [
        (25, 26, "knee"),
        (27, 28, "ankle"),
        (31, 32, "foot"),
        (23, 24, "hip"),
    ]:
        left_range_x = safe_col(out, f"range_lm{lm_left}_x")
        left_range_y = safe_col(out, f"range_lm{lm_left}_y")
        right_range_x = safe_col(out, f"range_lm{lm_right}_x")
        right_range_y = safe_col(out, f"range_lm{lm_right}_y")

        left_range = np.sqrt(left_range_x ** 2 + left_range_y ** 2)
        right_range = np.sqrt(right_range_x ** 2 + right_range_y ** 2)

        out[f"bio_left_{name}_motion_range"] = left_range
        out[f"bio_right_{name}_motion_range"] = right_range
        out[f"bio_{name}_motion_asymmetry"] = np.abs(left_range - right_range)
        out[f"bio_{name}_motion_ratio"] = safe_div(left_range, right_range)

        new_cols += [
            f"bio_left_{name}_motion_range",
            f"bio_right_{name}_motion_range",
            f"bio_{name}_motion_asymmetry",
            f"bio_{name}_motion_ratio",
        ]

    for lm_left, lm_right, name in [
        (25, 26, "knee"),
        (27, 28, "ankle"),
        (31, 32, "foot"),
        (23, 24, "hip"),
    ]:
        left_std_x = safe_col(out, f"std_lm{lm_left}_x")
        left_std_y = safe_col(out, f"std_lm{lm_left}_y")
        right_std_x = safe_col(out, f"std_lm{lm_right}_x")
        right_std_y = safe_col(out, f"std_lm{lm_right}_y")

        left_std = np.sqrt(left_std_x ** 2 + left_std_y ** 2)
        right_std = np.sqrt(right_std_x ** 2 + right_std_y ** 2)

        out[f"bio_left_{name}_variability"] = left_std
        out[f"bio_right_{name}_variability"] = right_std
        out[f"bio_{name}_variability_asymmetry"] = np.abs(left_std - right_std)
        out[f"bio_{name}_variability_ratio"] = safe_div(left_std, right_std)

        new_cols += [
            f"bio_left_{name}_variability",
            f"bio_right_{name}_variability",
            f"bio_{name}_variability_asymmetry",
            f"bio_{name}_variability_ratio",
        ]

    for lm_left, lm_right, name in [
        (25, 26, "knee"),
        (27, 28, "ankle"),
        (31, 32, "foot"),
        (23, 24, "hip"),
    ]:
        left_vel_x = safe_col(out, f"mean_abs_vel_lm{lm_left}_x")
        left_vel_y = safe_col(out, f"mean_abs_vel_lm{lm_left}_y")
        right_vel_x = safe_col(out, f"mean_abs_vel_lm{lm_right}_x")
        right_vel_y = safe_col(out, f"mean_abs_vel_lm{lm_right}_y")

        left_vel = np.sqrt(left_vel_x ** 2 + left_vel_y ** 2)
        right_vel = np.sqrt(right_vel_x ** 2 + right_vel_y ** 2)

        out[f"bio_left_{name}_mean_abs_velocity"] = left_vel
        out[f"bio_right_{name}_mean_abs_velocity"] = right_vel
        out[f"bio_{name}_velocity_asymmetry"] = np.abs(left_vel - right_vel)
        out[f"bio_{name}_velocity_ratio"] = safe_div(left_vel, right_vel)

        new_cols += [
            f"bio_left_{name}_mean_abs_velocity",
            f"bio_right_{name}_mean_abs_velocity",
            f"bio_{name}_velocity_asymmetry",
            f"bio_{name}_velocity_ratio",
        ]

    out["bio_lower_limb_motion_asymmetry_score"] = (
        out["bio_knee_motion_asymmetry"]
        + out["bio_ankle_motion_asymmetry"]
        + out["bio_foot_motion_asymmetry"]
    ) / 3.0

    out["bio_lower_limb_velocity_asymmetry_score"] = (
        out["bio_knee_velocity_asymmetry"]
        + out["bio_ankle_velocity_asymmetry"]
        + out["bio_foot_velocity_asymmetry"]
    ) / 3.0

    out["bio_lower_limb_variability_asymmetry_score"] = (
        out["bio_knee_variability_asymmetry"]
        + out["bio_ankle_variability_asymmetry"]
        + out["bio_foot_variability_asymmetry"]
    ) / 3.0

    out["bio_koa_limp_proxy_score"] = (
        out["bio_lower_limb_motion_asymmetry_score"]
        + out["bio_lower_limb_velocity_asymmetry_score"]
        + out["bio_knee_angle_asymmetry_proxy"]
    ) / 3.0

    new_cols += [
        "bio_lower_limb_motion_asymmetry_score",
        "bio_lower_limb_velocity_asymmetry_score",
        "bio_lower_limb_variability_asymmetry_score",
        "bio_koa_limp_proxy_score",
    ]

    out[new_cols] = out[new_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out


def add_koa_gait_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the V7 gait features used by the robust KOA model."""
    out = df.copy()

    def add_abs_diff(new_name: str, left: str, right: str):
        out[new_name] = (safe_col(out, left) - safe_col(out, right)).abs()

    def add_sum(new_name: str, left: str, right: str):
        out[new_name] = safe_col(out, left) + safe_col(out, right)

    def add_ratio(new_name: str, numerator: str, denominator: str):
        out[new_name] = safe_col(out, numerator).abs() / (
            safe_col(out, denominator).abs() + 1e-6
        )

    pairs = {
        "shoulder": (11, 12),
        "hip": (23, 24),
        "knee": (25, 26),
        "ankle": (27, 28),
    }
    stats = ["mean", "std", "range", "mean_abs_vel", "std_vel"]
    axes = ["x", "y", "z"]

    for stat in stats:
        for axis in axes:
            for part, (left_id, right_id) in pairs.items():
                add_abs_diff(
                    f"gait_asym_{stat}_{part}_{axis}",
                    f"{stat}_lm{left_id}_{axis}",
                    f"{stat}_lm{right_id}_{axis}",
                )

    for stat in ["range", "mean_abs_vel", "std_vel"]:
        for axis in axes:
            for part, (left_id, right_id) in {
                "ankle": (27, 28),
                "knee": (25, 26),
                "hip": (23, 24),
            }.items():
                add_sum(
                    f"gait_total_{stat}_{part}_{axis}",
                    f"{stat}_lm{left_id}_{axis}",
                    f"{stat}_lm{right_id}_{axis}",
                )

    for axis in axes:
        add_ratio(
            f"gait_ratio_ankle_to_hip_range_{axis}",
            f"range_lm27_{axis}",
            f"range_lm23_{axis}",
        )
        add_ratio(
            f"gait_ratio_knee_to_hip_range_{axis}",
            f"range_lm25_{axis}",
            f"range_lm23_{axis}",
        )
        add_ratio(
            f"gait_ratio_ankle_to_knee_vel_{axis}",
            f"mean_abs_vel_lm27_{axis}",
            f"mean_abs_vel_lm25_{axis}",
        )
        add_sum(
            f"gait_trunk_sway_std_{axis}",
            f"std_lm11_{axis}",
            f"std_lm12_{axis}",
        )
        add_sum(
            f"gait_pelvis_sway_std_{axis}",
            f"std_lm23_{axis}",
            f"std_lm24_{axis}",
        )

    return out


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

    return "High", ["Enough clean walking data for reliable KOA pattern screening."]


def koa_pattern_strength(probability: float, detection_threshold: float, koa_detected: bool = False) -> str:
    if koa_detected:
        margin = probability - detection_threshold
        if margin < 0.10:
            return "Weak / borderline KOA-like detected pattern"
        if margin < 0.25:
            return "Moderate KOA-like detected pattern"
        return "Strong KOA-like detected pattern"

    if probability < KOA_TENDENCY_THRESHOLD:
        return "Not detected"

    if probability < detection_threshold:
        return "Borderline KOA-like tendency"

    margin = probability - detection_threshold

    if margin < 0.10:
        return "Weak / borderline KOA-like detected pattern"
    if margin < 0.25:
        return "Moderate KOA-like detected pattern"
    return "Strong KOA-like detected pattern"


def decide_koa_result(
    koa_probability: float,
    koa_detected: bool,
    koa_tendency: bool,
    reliability_level: str,
) -> tuple[str, str]:
    if reliability_level == "Low":
        return (
            "Inconclusive - insufficient reliable gait pattern duration",
            "The model produced a KOA probability, but the video is too short or unreliable for a final KOA-pattern decision.",
        )

    if koa_detected:
        return (
            "KOA-like gait pattern detected",
            "The gait pattern matches the learned KOA-like gait pattern.",
        )

    if koa_tendency:
        return (
            "No confirmed KOA-like pattern detected, but borderline KOA-like tendency is present",
            "The gait does not pass the KOA detection threshold, but shows some KOA-like gait pattern overlap.",
        )

    return (
        "No KOA-like gait pattern detected",
        "The gait does not strongly match the learned KOA-like pattern.",
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def load_feature_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature list not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_settings() -> dict:
    if not KOA_SETTINGS_PATH.exists():
        return {"final_threshold": KOA_THRESHOLD}

    try:
        with open(KOA_SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"final_threshold": KOA_THRESHOLD}


def pattern_decision(window_probabilities: np.ndarray, settings: dict) -> tuple[bool, dict]:
    pattern = settings.get("recommended_video_setting") or {}
    mean_threshold = float(pattern.get("mean_threshold", settings.get("final_threshold", KOA_THRESHOLD)))
    window_threshold = float(pattern.get("window_threshold", mean_threshold))
    ratio_threshold = float(pattern.get("ratio_threshold", 0.0))
    count_threshold = int(pattern.get("count_threshold", 1))
    max_threshold = float(pattern.get("max_threshold", mean_threshold))
    min_total_windows = int(pattern.get("min_total_windows", 1))

    mean_probability = float(np.mean(window_probabilities))
    max_probability = float(np.max(window_probabilities))
    positive_count = int(np.sum(window_probabilities >= window_threshold))
    positive_ratio = float(positive_count / max(1, len(window_probabilities)))

    detected = bool(
        len(window_probabilities) >= min_total_windows
        and mean_probability >= mean_threshold
        and max_probability >= max_threshold
        and positive_count >= count_threshold
        and positive_ratio >= ratio_threshold
    )
    details = {
        "mean_threshold": mean_threshold,
        "window_threshold": window_threshold,
        "ratio_threshold": ratio_threshold,
        "count_threshold": count_threshold,
        "max_threshold": max_threshold,
        "min_total_windows": min_total_windows,
        "mean_probability": mean_probability,
        "max_probability": max_probability,
        "positive_window_count": positive_count,
        "positive_window_ratio": positive_ratio,
    }
    return detected, details


def predict_koa_model(feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not KOA_MODEL_PATH.exists():
        raise FileNotFoundError(f"KOA model not found: {KOA_MODEL_PATH}")

    if not KOA_FEATURES_PATH.exists():
        raise FileNotFoundError(f"KOA feature list not found: {KOA_FEATURES_PATH}")

    model = joblib.load(KOA_MODEL_PATH)
    feature_cols = load_feature_list(KOA_FEATURES_PATH)

    missing_features = [c for c in feature_cols if c not in feature_df.columns]

    if missing_features:
        raise ValueError(
            "Missing expected KOA V3 model features. "
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
        raise ValueError("KOA model does not contain positive class label 1.")

    positive_index = class_list.index(1)
    p_koa = proba[:, positive_index]

    return window_pred, p_koa


def predict_component2_koa(
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

    settings = load_settings()
    detection_threshold = float(settings.get("final_threshold", KOA_THRESHOLD))
    tendency_threshold = KOA_TENDENCY_THRESHOLD

    clean_duration_sec = get_clean_duration_seconds(csv_path, fps_used)

    feature_df = preprocess_single_csv(csv_path, direction, fps_used)

    window_pred, p_koa = predict_koa_model(feature_df)

    koa_detected, pattern_details = pattern_decision(p_koa, settings)
    koa_probability = float(pattern_details["mean_probability"])
    koa_tendency = (not koa_detected) and (koa_probability >= tendency_threshold)

    koa_window_ratio_default = float(np.mean(window_pred == 1))
    koa_window_ratio_detection = float(pattern_details["positive_window_ratio"])
    koa_window_ratio_tendency = float(np.mean(p_koa >= tendency_threshold))

    total_windows = int(len(feature_df))

    reliability_level, reliability_reasons = reliability_from_windows(
        total_windows=total_windows,
        clean_duration_sec=clean_duration_sec,
        direction_confidence=direction_confidence,
    )

    final_result, clinical_note = decide_koa_result(
        koa_probability=koa_probability,
        koa_detected=koa_detected,
        koa_tendency=koa_tendency,
        reliability_level=reliability_level,
    )

    return {
        "model_version": "KOA V14 youtube-ataxia once non-KOA non-refit holdout-evaluation HistGradientBoosting",
        "model_path": str(KOA_MODEL_PATH),
        "feature_list_path": str(KOA_FEATURES_PATH),

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

        "koa_probability": round(koa_probability, 4),
        "koa_tendency_threshold": tendency_threshold,
        "koa_detection_threshold": detection_threshold,
        "koa_pattern_setting": {
            "mean_threshold": pattern_details["mean_threshold"],
            "window_threshold": pattern_details["window_threshold"],
            "ratio_threshold": pattern_details["ratio_threshold"],
            "count_threshold": pattern_details["count_threshold"],
            "max_threshold": pattern_details["max_threshold"],
            "min_total_windows": pattern_details["min_total_windows"],
        },
        "koa_detected": bool(koa_detected),
        "koa_borderline_tendency": bool(koa_tendency),
        "koa_pattern_strength": koa_pattern_strength(koa_probability, detection_threshold, koa_detected),
        "koa_max_probability": round(float(pattern_details["max_probability"]), 4),
        "koa_positive_window_count": int(pattern_details["positive_window_count"]),

        "koa_window_ratio_default": round(koa_window_ratio_default, 4),
        "koa_window_ratio_tendency": round(koa_window_ratio_tendency, 4),
        "koa_window_ratio_detection": round(koa_window_ratio_detection, 4),

        "reliability_level": reliability_level,
        "reliability_reasons": reliability_reasons,

        "final_koa_result": final_result,
        "clinical_note": clinical_note,
    }


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Predict Component 2B KOA-like gait pattern from a training_safe_landmarks CSV using KOA V14 youtube-ataxia-once model."
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

    result = predict_component2_koa(
        csv_path=csv_path,
        direction=args.direction,
        fps_used=args.fps,
    )

    print("\n======================================")
    print(" COMPONENT 2B KOA PATTERN RESULT")
    print(" FINAL MODEL: KOA V14 YOUTUBE-ATAXIA ONCE")
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
    print(f"KOA probability:             {result['koa_probability']}")
    print(f"KOA tendency threshold:      {result['koa_tendency_threshold']}")
    print(f"KOA detection threshold:     {result['koa_detection_threshold']}")
    print(f"KOA pattern setting:         {result['koa_pattern_setting']}")
    print(f"KOA detected:                {result['koa_detected']}")
    print(f"KOA borderline tendency:     {result['koa_borderline_tendency']}")
    print(f"KOA pattern strength:        {result['koa_pattern_strength']}")
    print(f"KOA max probability:         {result['koa_max_probability']}")
    print(f"KOA positive window count:   {result['koa_positive_window_count']}")
    print(f"KOA window ratio default:    {result['koa_window_ratio_default']}")
    print(f"KOA window ratio tendency:   {result['koa_window_ratio_tendency']}")
    print(f"KOA window ratio detection:  {result['koa_window_ratio_detection']}")
    print("--------------------------------------")
    print(f"Final KOA result:            {result['final_koa_result']}")
    print(f"Clinical note:               {result['clinical_note']}")
    print("======================================\n")


if __name__ == "__main__":
    main()
