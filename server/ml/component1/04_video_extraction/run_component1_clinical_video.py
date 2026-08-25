import re
import json
from pathlib import Path
from datetime import datetime

import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_VIDEO_DIR = PROJECT_ROOT / "01_input_videos" / "inference"

OUTPUT_ROOT = PROJECT_ROOT / "02_extracted_csv"
OUTPUT_RAW = OUTPUT_ROOT / "raw" / "inference"
OUTPUT_TRAINING_SAFE = OUTPUT_ROOT / "training_safe" / "inference"
OUTPUT_SUMMARY = OUTPUT_ROOT / "summary" / "inference"

METADATA_DIR = PROJECT_ROOT / "03_metadata"
INFERENCE_METADATA_PATH = METADATA_DIR / "clinical_inference_metadata.csv"

MEDIAPIPE_MODEL_PATH = PROJECT_ROOT / "mediapipe_models" / "pose_landmarker.task"

# ============================================================
# FINAL COMPONENT 1 V4 MODEL FILES
# ============================================================

COMPONENT1_MODEL_PATH = PROJECT_ROOT / "06_models" / "component_1" / "component1_model_v4_tuned.joblib"
COMPONENT1_FEATURES_PATH = PROJECT_ROOT / "06_models" / "component_1" / "component1_feature_list_v4_tuned.json"
COMPONENT1_SETTINGS_PATH = PROJECT_ROOT / "06_models" / "component_1" / "component1_settings_v4_tuned.json"

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"]


# ============================================================
# EXTRACTION SETTINGS
# ============================================================

TARGET_FPS_FALLBACK = 30.0

KEY_IDS = [23, 24, 25, 26, 27, 28, 31, 32]

VIS_THRESHOLD = 0.30
INFRAME_EPS = 0.10
MIN_GOOD_JOINTS = 5

MAX_INTERP_GAP = 3
ENABLE_SMOOTHING = False
SMOOTH_WINDOW = 3

MIN_VALID_FRAMES_STRONG = 60
MIN_VALID_FRAMES_BORDERLINE = 40
MIN_VALID_RATIO = 0.35
MAX_POOR_RATIO = 0.40
MIN_CLEAN_VALID_RATIO = 0.95
MIN_CLEAN_VISIBILITY = 0.70


# ============================================================
# COMPONENT 1 PREPROCESSING SETTINGS
# MUST MATCH TRAINING
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
# CLINICAL RELIABILITY SETTINGS
# ============================================================

MIN_RELIABLE_WINDOWS = 5
RECOMMENDED_WINDOWS = 8
MIN_RELIABLE_DURATION_SEC = 4.0


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = python.BaseOptions
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode


def create_landmarker():
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
    )
    return vision.PoseLandmarker.create_from_options(options)


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dirs():
    INPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_RAW.mkdir(parents=True, exist_ok=True)
    OUTPUT_TRAINING_SAFE.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_stem(stem: str) -> str:
    stem = stem.strip()
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9_\-]", "", stem)
    stem = stem.strip("_")
    return stem if stem else "uploaded_video"


def make_unique_base_name(video_path: Path) -> str:
    safe_stem = sanitize_stem(video_path.stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_stem}_{timestamp}"


def landmark_columns():
    return [f"lm{i}_{axis}" for i in range(LANDMARK_COUNT) for axis in AXES]


def visibility_label(v):
    if pd.isna(v):
        return "Unknown"
    if v >= 0.75:
        return "Excellent"
    if v >= 0.50:
        return "Good"
    if v >= 0.30:
        return "Fair"
    return "Poor"


# ============================================================
# QUALITY FUNCTIONS
# ============================================================

def compute_frame_quality(df, key_ids, vis_threshold, eps):
    good_counts = np.zeros(len(df), dtype=int)
    vis_fail = np.zeros(len(df), dtype=int)
    frame_fail = np.zeros(len(df), dtype=int)

    for j in key_ids:
        x = df[f"lm{j}_x"]
        y = df[f"lm{j}_y"]
        v = df[f"lm{j}_vis"]

        has_xy = x.notna() & y.notna()

        in_frame = (
            has_xy
            & (x >= -eps)
            & (x <= 1 + eps)
            & (y >= -eps)
            & (y <= 1 + eps)
        )

        visible = v.fillna(0) >= vis_threshold
        good = in_frame & visible

        good_counts += good.astype(int).to_numpy()
        vis_fail += (~visible).astype(int).to_numpy()
        frame_fail += (~in_frame).astype(int).to_numpy()

    key_vis_cols = [f"lm{j}_vis" for j in key_ids]
    avg_key_visibility = df[key_vis_cols].mean(axis=1)
    visibility_quality = avg_key_visibility.apply(visibility_label)
    valid_frame = good_counts >= MIN_GOOD_JOINTS

    return avg_key_visibility, visibility_quality, good_counts, vis_fail, frame_fail, valid_frame


def find_longest_valid_run(mask):
    best_start, best_end = None, None
    cur_start = None

    for i, val in enumerate(mask):
        if val and cur_start is None:
            cur_start = i
        elif not val and cur_start is not None:
            if best_start is None or (i - cur_start) > (best_end - best_start):
                best_start, best_end = cur_start, i
            cur_start = None

    if cur_start is not None:
        if best_start is None or (len(mask) - cur_start) > (best_end - best_start):
            best_start, best_end = cur_start, len(mask)

    return best_start, best_end


def limited_interpolate(df, cols, max_gap):
    df = df.copy()

    if len(cols) > 0:
        df[cols] = df[cols].interpolate(
            method="linear",
            limit=max_gap,
            limit_direction="both",
            limit_area="inside",
        )

    return df


def optional_smooth(df, cols, enabled, window):
    if not enabled or window < 3 or len(cols) == 0:
        return df

    df = df.copy()
    df[cols] = df[cols].rolling(window, center=True, min_periods=1).median()
    return df


def decide_clip_status(df_clean, valid_ratio_full, poor_ratio_full):
    clean_frames = len(df_clean)

    clean_valid_ratio = float(df_clean["valid_frame"].mean()) if clean_frames else 0.0
    clean_mean_visibility = float(df_clean["avg_key_visibility"].mean()) if clean_frames else 0.0
    clean_nan_count = int(df_clean.isna().sum().sum()) if clean_frames else 999999

    if (
        clean_frames >= MIN_VALID_FRAMES_STRONG
        and clean_valid_ratio >= MIN_CLEAN_VALID_RATIO
        and clean_mean_visibility >= MIN_CLEAN_VISIBILITY
        and clean_nan_count == 0
        and poor_ratio_full <= MAX_POOR_RATIO
    ):
        return "accepted", False

    if (
        clean_frames >= MIN_VALID_FRAMES_BORDERLINE
        and clean_valid_ratio >= MIN_CLEAN_VALID_RATIO
        and clean_mean_visibility >= 0.65
        and clean_nan_count == 0
        and valid_ratio_full >= MIN_VALID_RATIO
    ):
        return "accepted", False

    if (
        clean_frames >= MIN_VALID_FRAMES_BORDERLINE
        and clean_valid_ratio >= 0.90
        and clean_mean_visibility >= 0.60
        and clean_nan_count == 0
    ):
        return "review_keep", True

    return "reject", True


# ============================================================
# DIRECTION ESTIMATION
# ============================================================

def estimate_direction_from_clean_pose(df_clean):
    if len(df_clean) < 2:
        return "L2R", 0.0, "low"

    mid_hip_x = (
        df_clean["lm23_x"].to_numpy(dtype=float)
        + df_clean["lm24_x"].to_numpy(dtype=float)
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
# EXTRACTION
# ============================================================

def extract_training_safe_csv(video_path: Path):
    base_name = make_unique_base_name(video_path)

    raw_csv_name = f"{base_name}_raw_landmarks.csv"
    clean_csv_name = f"{base_name}_training_safe_landmarks.csv"
    summary_csv_name = f"{base_name}_quality_summary.csv"

    output_raw_csv = OUTPUT_RAW / raw_csv_name
    output_clean_csv = OUTPUT_TRAINING_SAFE / clean_csv_name
    output_summary_csv = OUTPUT_SUMMARY / summary_csv_name

    print("\n===================================================")
    print("Processing video:", video_path.name)
    print("Generated base name:", base_name)
    print("===================================================")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0 or np.isnan(fps):
        fps = TARGET_FPS_FALLBACK

    metadata_cols = ["frame_index", "timestamp_ms"]
    landmark_cols = [f"lm{i}_{axis}" for i in range(33) for axis in ("x", "y", "z", "vis")]

    rows = []
    missing_detections = 0
    frame_idx = 0

    with create_landmarker() as landmarker:
        while True:
            ret, frame_bgr = cap.read()

            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((frame_idx / fps) * 1000)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            row = [frame_idx, timestamp_ms]

            if result.pose_landmarks:
                for lm in result.pose_landmarks[0]:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                row.extend([np.nan] * (33 * 4))
                missing_detections += 1

            rows.append(row)
            frame_idx += 1

    cap.release()

    if not rows:
        raise RuntimeError(f"No frames extracted from video: {video_path.name}")

    df = pd.DataFrame(rows, columns=metadata_cols + landmark_cols)

    avg_key_visibility, visibility_quality, good_counts, vis_fail, frame_fail, valid_frame = compute_frame_quality(
        df, KEY_IDS, VIS_THRESHOLD, INFRAME_EPS
    )

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "avg_key_visibility": avg_key_visibility,
                    "visibility_quality": visibility_quality,
                    "good_key_joints": good_counts,
                    "visibility_fail_count": vis_fail,
                    "out_of_frame_count": frame_fail,
                    "valid_frame": valid_frame,
                }
            ),
        ],
        axis=1,
    )

    df.to_csv(output_raw_csv, index=False)

    valid_ratio_full = float(df["valid_frame"].mean()) if len(df) else 0.0
    poor_ratio_full = float((df["visibility_quality"] == "Poor").mean()) if len(df) else 1.0

    valid_mask = df["valid_frame"].to_numpy().astype(bool)
    start_idx, end_idx = find_longest_valid_run(valid_mask)

    if start_idx is None or end_idx is None:
        df_clean = df.copy()
    else:
        df_clean = df.iloc[start_idx:end_idx].copy()

    if len(df_clean) > 0 and len(df_clean) < MIN_VALID_FRAMES_BORDERLINE and start_idx is not None:
        extra = 5
        expanded_start = max(0, start_idx - extra)
        expanded_end = min(len(df), end_idx + extra)
        df_clean = df.iloc[expanded_start:expanded_end].copy()

    xyz_cols = [c for c in df_clean.columns if c.endswith(("_x", "_y", "_z"))]
    vis_cols = [c for c in df_clean.columns if c.endswith("_vis")]

    df_clean = limited_interpolate(df_clean, xyz_cols, MAX_INTERP_GAP)

    if len(vis_cols) > 0:
        df_clean[vis_cols] = df_clean[vis_cols].interpolate(
            method="linear",
            limit=MAX_INTERP_GAP,
            limit_direction="both",
            limit_area="inside",
        )

    avg_key_visibility2, visibility_quality2, good_counts2, vis_fail2, frame_fail2, valid_frame2 = compute_frame_quality(
        df_clean, KEY_IDS, VIS_THRESHOLD, INFRAME_EPS
    )

    df_clean["avg_key_visibility"] = avg_key_visibility2
    df_clean["visibility_quality"] = visibility_quality2
    df_clean["good_key_joints"] = good_counts2
    df_clean["visibility_fail_count"] = vis_fail2
    df_clean["out_of_frame_count"] = frame_fail2
    df_clean["valid_frame"] = valid_frame2

    df_clean = optional_smooth(df_clean, xyz_cols, ENABLE_SMOOTHING, SMOOTH_WINDOW)

    avg_key_visibility3, visibility_quality3, good_counts3, vis_fail3, frame_fail3, valid_frame3 = compute_frame_quality(
        df_clean, KEY_IDS, VIS_THRESHOLD, INFRAME_EPS
    )

    df_clean["avg_key_visibility"] = avg_key_visibility3
    df_clean["visibility_quality"] = visibility_quality3
    df_clean["good_key_joints"] = good_counts3
    df_clean["visibility_fail_count"] = vis_fail3
    df_clean["out_of_frame_count"] = frame_fail3
    df_clean["valid_frame"] = valid_frame3

    estimated_direction, direction_delta_x, direction_confidence = estimate_direction_from_clean_pose(df_clean)

    clip_status, clip_review_required = decide_clip_status(df_clean, valid_ratio_full, poor_ratio_full)

    clean_frames = len(df_clean)
    clean_valid_frames = int(df_clean["valid_frame"].sum()) if clean_frames else 0
    clean_valid_ratio = float(df_clean["valid_frame"].mean()) if clean_frames else 0.0
    clean_mean_visibility = float(df_clean["avg_key_visibility"].mean()) if clean_frames else 0.0
    clean_has_nans = int(df_clean.isna().sum().sum()) if clean_frames else np.nan
    needs_padding = "yes" if clean_frames < 60 else "no"
    clean_duration_sec = clean_frames / float(fps) if fps else 0.0

    df_clean = df_clean.copy()
    df_clean["estimated_direction"] = estimated_direction
    df_clean["direction_delta_x"] = direction_delta_x
    df_clean["direction_confidence"] = direction_confidence
    df_clean["clip_status"] = clip_status
    df_clean["clip_review_required"] = clip_review_required

    df_clean.to_csv(output_clean_csv, index=False)

    summary = {
        "original_video": video_path.name,
        "generated_base_name": base_name,
        "class_label": "inference",
        "binary_label": "unknown",
        "person_id": "unknown",
        "subject_id": "inference_unknown",
        "estimated_direction": estimated_direction,
        "direction_delta_x": round(direction_delta_x, 5),
        "direction_confidence": direction_confidence,
        "fps_used": int(round(fps)),
        "original_frames": len(df),
        "clean_frames": clean_frames,
        "clean_duration_sec": round(clean_duration_sec, 3),
        "missing_detections_before_fill": missing_detections,
        "valid_ratio_full": round(valid_ratio_full, 4),
        "poor_ratio_full": round(poor_ratio_full, 4),
        "mean_avg_key_visibility_full": round(df["avg_key_visibility"].mean(), 4) if len(df) else np.nan,
        "mean_avg_key_visibility_clean": round(clean_mean_visibility, 4),
        "clean_valid_frames": clean_valid_frames,
        "clean_valid_ratio": round(clean_valid_ratio, 4),
        "clean_has_nans": clean_has_nans,
        "clean_start_frame": int(df_clean["frame_index"].iloc[0]) if clean_frames else np.nan,
        "clean_end_frame": int(df_clean["frame_index"].iloc[-1]) if clean_frames else np.nan,
        "needs_padding": needs_padding,
        "clip_status": clip_status,
        "clip_review_required": clip_review_required,
        "raw_file": raw_csv_name,
        "training_safe_file": clean_csv_name,
        "summary_file": summary_csv_name,
    }

    pd.DataFrame([summary]).to_csv(output_summary_csv, index=False)
    append_inference_metadata(summary)

    print("Saved raw:     ", output_raw_csv)
    print("Saved clean:   ", output_clean_csv)
    print("Saved summary: ", output_summary_csv)
    print("Estimated direction:", estimated_direction)
    print("Direction confidence:", direction_confidence)
    print("Clean frames:", clean_frames)
    print("Clean duration sec:", round(clean_duration_sec, 2))
    print("Status:", clip_status)
    print("Needs padding:", needs_padding)

    return {
        "video_path": video_path,
        "training_safe_csv": output_clean_csv,
        "raw_csv": output_raw_csv,
        "summary_csv": output_summary_csv,
        "estimated_direction": estimated_direction,
        "direction_confidence": direction_confidence,
        "fps_used": float(fps),
        "clean_frames": clean_frames,
        "clean_duration_sec": clean_duration_sec,
        "clip_status": clip_status,
        "clip_review_required": clip_review_required,
    }


def append_inference_metadata(summary_row):
    metadata_columns = [
        "original_video",
        "generated_base_name",
        "class_label",
        "binary_label",
        "person_id",
        "subject_id",
        "estimated_direction",
        "direction_delta_x",
        "direction_confidence",
        "fps_used",
        "original_frames",
        "clean_frames",
        "clean_duration_sec",
        "needs_padding",
        "clip_status",
        "clip_review_required",
        "raw_file",
        "training_safe_file",
        "summary_file",
    ]

    row_clean = {col: summary_row.get(col, "") for col in metadata_columns}
    new_row_df = pd.DataFrame([row_clean], columns=metadata_columns)

    if INFERENCE_METADATA_PATH.exists():
        old_df = pd.read_csv(INFERENCE_METADATA_PATH, dtype=str)

        if "generated_base_name" in old_df.columns:
            old_df = old_df[old_df["generated_base_name"] != row_clean["generated_base_name"]]

        final_df = pd.concat([old_df, new_row_df], ignore_index=True)
    else:
        final_df = new_row_df

    final_df.to_csv(INFERENCE_METADATA_PATH, index=False)


# ============================================================
# COMPONENT 1 PREPROCESSING + PREDICTION
# ============================================================

def load_pose_csv(path: Path):
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


def sequence_from_df(df):
    cols = landmark_columns()
    arr = df[cols].to_numpy(dtype=np.float32)
    arr = arr.reshape(len(df), LANDMARK_COUNT, len(AXES))
    return arr


def get_time_seconds(df, fps_used):
    if "timestamp_ms" in df.columns and len(df) >= 2:
        t = pd.to_numeric(df["timestamp_ms"], errors="coerce").to_numpy(dtype=np.float64) / 1000.0
        t = t - t[0]

        if np.all(np.isfinite(t)) and t[-1] > t[0]:
            return t

    return np.arange(len(df), dtype=np.float64) / float(fps_used)


def resample_sequence(seq, time_sec, target_fps):
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


def edge_pad_to_window(seq, window_size):
    if len(seq) >= window_size:
        return seq

    pad_len = window_size - len(seq)
    last_frame = seq[-1:, :, :]
    pad = np.repeat(last_frame, pad_len, axis=0)

    return np.concatenate([seq, pad], axis=0)


def normalize_pose(seq):
    seq = seq.copy()
    xyz = seq[:, :, :3]

    mid_hip = (xyz[:, LEFT_HIP, :] + xyz[:, RIGHT_HIP, :]) / 2.0
    xyz = xyz - mid_hip[:, None, :]

    shoulder_vec = xyz[:, LEFT_SHOULDER, :] - xyz[:, RIGHT_SHOULDER, :]
    shoulder_width = np.linalg.norm(shoulder_vec, axis=1)

    median_scale = np.nanmedian(shoulder_width)

    if not np.isfinite(median_scale) or median_scale < MIN_SHOULDER_SCALE:
        median_scale = 1.0

    shoulder_width = np.where(shoulder_width < MIN_SHOULDER_SCALE, median_scale, shoulder_width)
    xyz = xyz / shoulder_width[:, None, None]

    seq[:, :, :3] = xyz
    return seq


def standardize_direction(seq, direction):
    seq = seq.copy()

    if DIRECTION_STANDARDIZE and direction == "R2L":
        seq[:, :, 0] *= -1.0

    return seq


def make_windows(seq, window_size, stride):
    windows = []

    if len(seq) < window_size:
        return windows

    for start in range(0, len(seq) - window_size + 1, stride):
        windows.append(seq[start:start + window_size])

    return windows


def aggregate_features(window):
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


def severity_from_abnormal_score(score, final_label=None):
    """
    Screening severity text.
    If final prediction is normal, do not show abnormal severity.
    """
    if final_label == 0:
        return "Normal / Low screening concern"

    if final_label is None:
        return "Inconclusive - repeat with longer walking video"

    if score < 0.60:
        return "Mild abnormal screening concern"
    if score < 0.80:
        return "Moderate abnormal screening concern"
    return "High abnormal screening concern"


def reliability_from_windows(total_windows, clean_duration_sec, direction_confidence, clip_status):
    reasons = []

    if clip_status == "reject":
        return "Rejected", ["Pose quality was rejected."]

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

    return "High", ["Enough clean walking data for reliable Component 1 screening."]


def preprocess_for_component1(csv_path, direction, fps_used):
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

    if not windows:
        raise ValueError("No windows created.")

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


def predict_component1(csv_path, direction, fps_used, clean_duration_sec, direction_confidence, clip_status):
    model = joblib.load(COMPONENT1_MODEL_PATH)

    with open(COMPONENT1_FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    with open(COMPONENT1_SETTINGS_PATH, "r", encoding="utf-8") as f:
        settings = json.load(f)

    video_threshold = float(settings.get("video_threshold", 0.406))

    feature_df = preprocess_for_component1(csv_path, direction, fps_used)

    missing_features = [c for c in feature_cols if c not in feature_df.columns]

    if missing_features:
        raise ValueError(f"Missing model features: {missing_features[:10]}")

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
        raise ValueError("Component 1 model does not contain abnormal class label 1.")

    abnormal_index = class_list.index(1)

    p_abnormal = proba[:, abnormal_index]

    mean_prob_abnormal = float(np.mean(p_abnormal))
    abnormal_ratio_default = float(np.mean(window_pred == 1))
    abnormal_ratio_threshold = float(np.mean(p_abnormal >= video_threshold))

    model_label = 1 if mean_prob_abnormal >= video_threshold else 0
    model_result = "Abnormal gait" if model_label == 1 else "Normal gait"

    total_windows = int(len(feature_df))

    reliability_level, reliability_reasons = reliability_from_windows(
        total_windows=total_windows,
        clean_duration_sec=clean_duration_sec,
        direction_confidence=direction_confidence,
        clip_status=clip_status,
    )

    if reliability_level == "Low":
        final_label = None
        final_result = "Inconclusive - insufficient reliable gait duration"
        clinical_note = (
            "The model produced a screening suggestion, but the video is too short "
            "or not reliable enough for a final Component 1 decision."
        )
    else:
        final_label = model_label
        final_result = model_result
        clinical_note = "Component 1 screening result is usable."

    confidence_percent = (
        mean_prob_abnormal * 100
        if model_label == 1
        else (1 - mean_prob_abnormal) * 100
    )

    return {
        "final_label": final_label,
        "final_result": final_result,
        "model_suggested_label": model_label,
        "model_suggested_result": model_result,
        "mean_prob_abnormal": round(mean_prob_abnormal, 4),
        "confidence_percent": round(confidence_percent, 2),
        "abnormal_ratio_default": round(abnormal_ratio_default, 4),
        "abnormal_ratio_threshold": round(abnormal_ratio_threshold, 4),
        "video_threshold": video_threshold,
        "screening_severity": severity_from_abnormal_score(mean_prob_abnormal, final_label),
        "total_windows": total_windows,
        "padded": bool(feature_df["padded"].iloc[0]),
        "resampled_frames": int(feature_df["resampled_frames"].iloc[0]),
        "reliability_level": reliability_level,
        "reliability_reasons": reliability_reasons,
        "clinical_note": clinical_note,
    }


# ============================================================
# MAIN CLINICAL PIPELINE
# ============================================================

def main():
    ensure_dirs()

    if not MEDIAPIPE_MODEL_PATH.exists():
        print("ERROR: MediaPipe model not found:")
        print(MEDIAPIPE_MODEL_PATH)
        return

    if not COMPONENT1_MODEL_PATH.exists():
        print("ERROR: Component 1 V4 model not found:")
        print(COMPONENT1_MODEL_PATH)
        return

    if not COMPONENT1_FEATURES_PATH.exists():
        print("ERROR: Component 1 V4 feature list not found:")
        print(COMPONENT1_FEATURES_PATH)
        return

    if not COMPONENT1_SETTINGS_PATH.exists():
        print("ERROR: Component 1 V4 settings not found:")
        print(COMPONENT1_SETTINGS_PATH)
        return

    videos = []

    for ext in VIDEO_EXTENSIONS:
        videos.extend(INPUT_VIDEO_DIR.glob(f"*{ext}"))

    videos = sorted(set(videos))

    if not videos:
        print("No videos found in:", INPUT_VIDEO_DIR)
        print("You can upload any video name, for example:")
        print("  my_walk_video.mp4")
        print("  patient_test.mov")
        print("  VID_1234.mp4")
        return

    print("Project root:", PROJECT_ROOT)
    print("Input video folder:", INPUT_VIDEO_DIR)
    print("Videos found:", len(videos))
    print("Using Component 1 model:", COMPONENT1_MODEL_PATH)

    for video_path in videos:
        try:
            extraction = extract_training_safe_csv(video_path)

            if extraction["clip_status"] == "reject":
                print("\nRESULT NOT GENERATED")
                print("Reason: pose quality was rejected.")
                print("Please check raw/summary output.")
                continue

            prediction = predict_component1(
                csv_path=extraction["training_safe_csv"],
                direction=extraction["estimated_direction"],
                fps_used=extraction["fps_used"],
                clean_duration_sec=extraction["clean_duration_sec"],
                direction_confidence=extraction["direction_confidence"],
                clip_status=extraction["clip_status"],
            )

            print("\n======================================")
            print(" COMPONENT 1 CLINICAL SCREENING RESULT")
            print(" FINAL MODEL: COMPONENT 1 V4")
            print("======================================")
            print(f"Video file:                {video_path.name}")
            print(f"Training-safe CSV:          {extraction['training_safe_csv']}")
            print(f"Estimated direction:        {extraction['estimated_direction']}")
            print(f"Direction confidence:       {extraction['direction_confidence']}")
            print(f"FPS used:                   {round(extraction['fps_used'], 2)}")
            print(f"Clean frames:               {extraction['clean_frames']}")
            print(f"Clean duration:             {extraction['clean_duration_sec']:.2f}s")
            print(f"Clip status:                {extraction['clip_status']}")
            print("--------------------------------------")
            print(f"Reliability:                {prediction['reliability_level']}")

            for reason in prediction["reliability_reasons"]:
                print(f"Reliability note:           {reason}")

            print("--------------------------------------")
            print(f"Final result:               {prediction['final_result']}")
            print(f"Final label:                {prediction['final_label']}  (normal=0, abnormal=1, None=inconclusive)")
            print(f"Model suggested result:     {prediction['model_suggested_result']}")
            print(f"Model suggested label:      {prediction['model_suggested_label']}")
            print(f"Mean abnormal probability:  {prediction['mean_prob_abnormal']}")
            print(f"Model confidence:           {prediction['confidence_percent']}%")
            print(f"Abnormal ratio default:     {prediction['abnormal_ratio_default']}")
            print(f"Abnormal ratio threshold:   {prediction['abnormal_ratio_threshold']}")
            print(f"Video threshold:            {prediction['video_threshold']}")
            print(f"Screening severity:         {prediction['screening_severity']}")
            print(f"Total windows:              {prediction['total_windows']}")
            print(f"Padded:                     {prediction['padded']}")
            print(f"Resampled frames:           {prediction['resampled_frames']}")
            print(f"Clinical note:              {prediction['clinical_note']}")
            print("======================================\n")

        except Exception as e:
            print("\nFAILED:", video_path.name)
            print("Reason:", e)


if __name__ == "__main__":
    main()