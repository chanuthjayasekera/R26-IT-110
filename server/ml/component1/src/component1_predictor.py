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

MODEL_DIR = PROJECT_ROOT / "06_models" / "component_1"

# ============================================================
# FINAL COMPONENT 1 V4 MODEL FILES
# ============================================================

MODEL_PATH = MODEL_DIR / "component1_model_v4_tuned.joblib"
FEATURES_PATH = MODEL_DIR / "component1_feature_list_v4_tuned.json"
SETTINGS_PATH = MODEL_DIR / "component1_settings_v4_tuned.json"

# Auto-detect latest CSV created by run_component1_clinical_video.py
INFERENCE_TRAINING_SAFE_DIR = PROJECT_ROOT / "02_extracted_csv" / "training_safe" / "inference"


# ============================================================
# DEFAULT FINAL THRESHOLD
# ============================================================

DEFAULT_VIDEO_THRESHOLD = 0.406


# ============================================================
# CONSTANTS - MUST MATCH TRAINING
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


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# CSV LOADING + PREPROCESSING
# ============================================================

def load_pose_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Compatibility:
    # Training feature names use lm*_vis.
    # If CSV has lm*_visibility instead, copy it into lm*_vis.
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
    Same as training:
    - mid-hip centering
    - shoulder-width scaling
    - preserve visibility
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
    Same as training:
    R2L sequences are flipped on normalized x-axis.
    """
    if not DIRECTION_STANDARDIZE:
        return seq

    seq = seq.copy()
    direction = str(direction).upper().strip()

    if direction == "R2L":
        seq[:, :, 0] *= -1.0
    elif direction == "L2R":
        pass
    else:
        raise ValueError("direction must be L2R or R2L")

    return seq


def make_windows(seq: np.ndarray, window_size: int, stride: int) -> list[np.ndarray]:
    windows = []

    if len(seq) < window_size:
        return windows

    for start in range(0, len(seq) - window_size + 1, stride):
        end = start + window_size
        windows.append(seq[start:end])

    return windows


def aggregate_features(window: np.ndarray) -> dict:
    """
    V3 feature set:
    132 base landmark values × 7 groups = 924 features

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
# RELIABILITY + SEVERITY
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

    return "High", ["Enough clean walking data for reliable normal/abnormal screening."]


def severity_from_abnormal_score(score: float) -> str:
    """
    Screening-level severity, not clinical diagnosis.
    Based on mean abnormal probability.
    """
    if score < 0.40:
        return "Normal / Low screening concern"
    if score < 0.60:
        return "Mild abnormal screening concern"
    if score < 0.80:
        return "Moderate abnormal screening concern"
    return "High abnormal screening concern"


def decide_component1_result(
    mean_prob_abnormal: float,
    final_label: int,
    reliability_level: str,
) -> tuple[str, str]:
    if reliability_level == "Low":
        return (
            "Inconclusive - insufficient reliable gait pattern duration",
            "The model produced a normal/abnormal screening probability, but the video is too short or unreliable for a final screening decision.",
        )

    if final_label == 1:
        return (
            "Abnormal gait detected",
            "The gait pattern is abnormal and should be passed to the specialist models for SCA, KOA, and PD analysis.",
        )

    return (
        "Normal gait detected",
        "The gait pattern does not strongly match the learned abnormal gait patterns.",
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def load_feature_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature list not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_video_threshold() -> float:
    if not SETTINGS_PATH.exists():
        return DEFAULT_VIDEO_THRESHOLD

    try:
        settings = load_json(SETTINGS_PATH)
        threshold = float(settings.get("video_threshold", DEFAULT_VIDEO_THRESHOLD))

        if threshold <= 0:
            return DEFAULT_VIDEO_THRESHOLD

        return threshold
    except Exception:
        return DEFAULT_VIDEO_THRESHOLD


def predict_component1_model(feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Component 1 model not found: {MODEL_PATH}")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Component 1 feature list not found: {FEATURES_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_cols = load_feature_list(FEATURES_PATH)

    missing_features = [c for c in feature_cols if c not in feature_df.columns]

    if missing_features:
        raise ValueError(
            "Missing expected Component 1 V4 model features. "
            f"First missing features: {missing_features[:20]}"
        )

    X = (
        feature_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    y_pred_window = model.predict(X)
    proba = model.predict_proba(X)

    class_list = list(model.classes_)

    if 1 not in class_list:
        raise ValueError("Model does not contain abnormal class label 1.")

    abnormal_index = class_list.index(1)
    p_abnormal = proba[:, abnormal_index]

    return y_pred_window, p_abnormal


def predict_component1(
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

    video_threshold = load_video_threshold()

    clean_duration_sec = get_clean_duration_seconds(csv_path, fps_used)

    feature_df = preprocess_single_csv(csv_path, direction, fps_used)

    y_pred_window, p_abnormal = predict_component1_model(feature_df)

    mean_prob_abnormal = float(np.mean(p_abnormal))

    abnormal_ratio_default = float(np.mean(y_pred_window == 1))
    abnormal_ratio_threshold = float(np.mean(p_abnormal >= video_threshold))

    final_label = 1 if mean_prob_abnormal >= video_threshold else 0
    final_text = "Abnormal gait" if final_label == 1 else "Normal gait"

    confidence_percent = (
        mean_prob_abnormal * 100
        if final_label == 1
        else (1 - mean_prob_abnormal) * 100
    )

    total_windows = int(len(feature_df))

    reliability_level, reliability_reasons = reliability_from_windows(
        total_windows=total_windows,
        clean_duration_sec=clean_duration_sec,
        direction_confidence=direction_confidence,
    )

    final_result, clinical_note = decide_component1_result(
        mean_prob_abnormal=mean_prob_abnormal,
        final_label=final_label,
        reliability_level=reliability_level,
    )

    return {
        "model_version": "Component 1 V4 ExtraTrees tuned",
        "model_path": str(MODEL_PATH),
        "feature_list_path": str(FEATURES_PATH),

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

        "video_threshold": video_threshold,
        "final_label": int(final_label),
        "final_result": final_text,
        "mean_prob_abnormal": round(mean_prob_abnormal, 4),
        "confidence_percent": round(confidence_percent, 2),
        "abnormal_ratio_model_default": round(abnormal_ratio_default, 4),
        "abnormal_ratio_threshold": round(abnormal_ratio_threshold, 4),
        "screening_severity": severity_from_abnormal_score(mean_prob_abnormal),

        "reliability_level": reliability_level,
        "reliability_reasons": reliability_reasons,

        "final_screening_result": final_result,
        "clinical_note": clinical_note,
    }


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Predict Component 1 Normal vs Abnormal gait screening from a training_safe_landmarks CSV using final Component 1 V4 model."
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

    result = predict_component1(
        csv_path=csv_path,
        direction=args.direction,
        fps_used=args.fps,
    )

    print("\n======================================")
    print(" COMPONENT 1 GAIT SCREENING RESULT")
    print(" FINAL MODEL: COMPONENT 1 V4")
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
    print(f"Video threshold:             {result['video_threshold']}")
    print(f"Final label:                 {result['final_label']}  (normal=0, abnormal=1)")
    print(f"Final result:                {result['final_result']}")
    print(f"Mean abnormal probability:   {result['mean_prob_abnormal']}")
    print(f"Confidence:                  {result['confidence_percent']}%")
    print(f"Abnormal ratio default:      {result['abnormal_ratio_model_default']}")
    print(f"Abnormal ratio threshold:    {result['abnormal_ratio_threshold']}")
    print(f"Screening severity:          {result['screening_severity']}")
    print("--------------------------------------")
    print(f"Final screening result:      {result['final_screening_result']}")
    print(f"Clinical note:               {result['clinical_note']}")
    print("======================================\n")


if __name__ == "__main__":
    main()