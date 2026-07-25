from pathlib import Path
from datetime import datetime
import re
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd


# ============================================================
# COMPONENT 2B KOA LANDMARK EXTRACTOR
# Purpose: video -> training_safe_landmarks.csv only
# No Component 1 model check. No prediction here.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_VIDEO_DIR = PROJECT_ROOT / "01_input_videos" / "inference"

OUTPUT_ROOT = PROJECT_ROOT / "02_extracted_csv"
OUTPUT_RAW = OUTPUT_ROOT / "raw" / "inference"
OUTPUT_TRAINING_SAFE = OUTPUT_ROOT / "training_safe" / "inference"
OUTPUT_SUMMARY = OUTPUT_ROOT / "summary" / "inference"

METADATA_DIR = PROJECT_ROOT / "03_metadata"
INFERENCE_METADATA_PATH = METADATA_DIR / "component2_koa_inference_metadata.csv"

MEDIAPIPE_MODEL_PATH = PROJECT_ROOT / "mediapipe_models" / "pose_landmarker.task"

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"]

TARGET_FPS_FALLBACK = 30.0

KEY_IDS = [23, 24, 25, 26, 27, 28, 31, 32]
VIS_THRESHOLD = 0.30
INFRAME_EPS = 0.10
MIN_GOOD_JOINTS = 5
MAX_INTERP_GAP = 3

MIN_VALID_FRAMES_STRONG = 60
MIN_VALID_FRAMES_BORDERLINE = 40
MIN_VALID_RATIO = 0.35
MAX_POOR_RATIO = 0.40
MIN_CLEAN_VALID_RATIO = 0.95
MIN_CLEAN_VISIBILITY = 0.70

LANDMARK_COUNT = 33
AXES = ["x", "y", "z", "vis"]


BaseOptions = python.BaseOptions
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode


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


def create_landmarker():
    if not MEDIAPIPE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {MEDIAPIPE_MODEL_PATH}\n"
            "Copy pose_landmarker.task into mediapipe_models/."
        )

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
    )
    return vision.PoseLandmarker.create_from_options(options)


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


def compute_frame_quality(df):
    good_counts = np.zeros(len(df), dtype=int)
    vis_fail = np.zeros(len(df), dtype=int)
    frame_fail = np.zeros(len(df), dtype=int)

    for j in KEY_IDS:
        x = df[f"lm{j}_x"]
        y = df[f"lm{j}_y"]
        v = df[f"lm{j}_vis"]

        has_xy = x.notna() & y.notna()
        in_frame = (
            has_xy
            & (x >= -INFRAME_EPS)
            & (x <= 1 + INFRAME_EPS)
            & (y >= -INFRAME_EPS)
            & (y <= 1 + INFRAME_EPS)
        )

        visible = v.fillna(0) >= VIS_THRESHOLD
        good = in_frame & visible

        good_counts += good.astype(int).to_numpy()
        vis_fail += (~visible).astype(int).to_numpy()
        frame_fail += (~in_frame).astype(int).to_numpy()

    key_vis_cols = [f"lm{j}_vis" for j in KEY_IDS]
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
    if cols:
        df[cols] = df[cols].interpolate(
            method="linear",
            limit=max_gap,
            limit_direction="both",
            limit_area="inside",
        )
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


def extract_training_safe_csv(video_path: Path):
    base_name = make_unique_base_name(video_path)

    raw_csv_name = f"{base_name}_raw_landmarks.csv"
    clean_csv_name = f"{base_name}_training_safe_landmarks.csv"
    summary_csv_name = f"{base_name}_quality_summary.csv"

    output_raw_csv = OUTPUT_RAW / raw_csv_name
    output_clean_csv = OUTPUT_TRAINING_SAFE / clean_csv_name
    output_summary_csv = OUTPUT_SUMMARY / summary_csv_name

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("INPUT_VIDEO_DIR:", INPUT_VIDEO_DIR)
    print("OUTPUT_TRAINING_SAFE:", OUTPUT_TRAINING_SAFE)
    print("Processing video:", video_path)

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

    avg_key_visibility, visibility_quality, good_counts, vis_fail, frame_fail, valid_frame = compute_frame_quality(df)

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

    if vis_cols:
        df_clean[vis_cols] = df_clean[vis_cols].interpolate(
            method="linear",
            limit=MAX_INTERP_GAP,
            limit_direction="both",
            limit_area="inside",
        )

    avg_key_visibility2, visibility_quality2, good_counts2, vis_fail2, frame_fail2, valid_frame2 = compute_frame_quality(df_clean)
    df_clean["avg_key_visibility"] = avg_key_visibility2
    df_clean["visibility_quality"] = visibility_quality2
    df_clean["good_key_joints"] = good_counts2
    df_clean["visibility_fail_count"] = vis_fail2
    df_clean["out_of_frame_count"] = frame_fail2
    df_clean["valid_frame"] = valid_frame2

    # Fill remaining missing values so predictor can use the CSV.
    lm_cols = [f"lm{i}_{axis}" for i in range(33) for axis in ("x", "y", "z", "vis")]
    df_clean[lm_cols] = (
        df_clean[lm_cols]
        .replace([np.inf, -np.inf], np.nan)
        .interpolate(limit_direction="both")
        .bfill()
        .ffill()
        .fillna(0.0)
    )

    estimated_direction, direction_delta_x, direction_confidence = estimate_direction_from_clean_pose(df_clean)
    clip_status, clip_review_required = decide_clip_status(df_clean, valid_ratio_full, poor_ratio_full)

    clean_frames = len(df_clean)
    clean_valid_frames = int(df_clean["valid_frame"].sum()) if clean_frames else 0
    clean_valid_ratio = float(df_clean["valid_frame"].mean()) if clean_frames else 0.0
    clean_mean_visibility = float(df_clean["avg_key_visibility"].mean()) if clean_frames else 0.0
    clean_has_nans = int(df_clean.isna().sum().sum()) if clean_frames else np.nan
    needs_padding = "yes" if clean_frames < 60 else "no"
    clean_duration_sec = clean_frames / float(fps) if fps else 0.0

    df_clean["estimated_direction"] = estimated_direction
    df_clean["direction_delta_x"] = direction_delta_x
    df_clean["direction_confidence"] = direction_confidence
    df_clean["clip_status"] = clip_status
    df_clean["clip_review_required"] = clip_review_required

    # IMPORTANT: Always save training-safe CSV.
    # Even reject/review clips are saved so the dashboard can explain what happened.
    df_clean.to_csv(output_clean_csv, index=False)

    summary = {
        "original_video": video_path.name,
        "generated_base_name": base_name,
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

    if INFERENCE_METADATA_PATH.exists():
        old = pd.read_csv(INFERENCE_METADATA_PATH, dtype=str)
        final = pd.concat([old, pd.DataFrame([summary])], ignore_index=True)
    else:
        final = pd.DataFrame([summary])
    final.to_csv(INFERENCE_METADATA_PATH, index=False)

    print("Saved raw:", output_raw_csv)
    print("Saved clean:", output_clean_csv)
    print("Saved summary:", output_summary_csv)
    print("Clip status:", clip_status)
    print("Clean frames:", clean_frames)
    print("Clean duration sec:", round(clean_duration_sec, 2))

    return output_clean_csv


def find_videos_from_folder():
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(INPUT_VIDEO_DIR.glob(f"*{ext}"))
    return sorted(set(videos))


def main():
    ensure_dirs()

    # Prefer explicit video path from dashboard.
    if len(sys.argv) >= 2:
        video_path = Path(sys.argv[1])
        if not video_path.exists():
            raise FileNotFoundError(f"Input video path not found: {video_path}")
        extract_training_safe_csv(video_path)
        return

    videos = find_videos_from_folder()
    if not videos:
        raise FileNotFoundError(f"No videos found in {INPUT_VIDEO_DIR}")

    for video_path in videos:
        extract_training_safe_csv(video_path)


if __name__ == "__main__":
    main()
