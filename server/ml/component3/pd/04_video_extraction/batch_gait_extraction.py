import os
import re
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd


# =========================================================
# PROJECT PATHS
# Script location:
# D:\gait_dataset_project_structure\09_scripts\batch_gait_extraction.py
#
# Project root:
# D:\gait_dataset_project_structure
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_VIDEO_DIR = PROJECT_ROOT / "01_videos_to_extract"

OUTPUT_ROOT = PROJECT_ROOT / "02_extracted_csv"
OUTPUT_TRAINING_SAFE = OUTPUT_ROOT / "training_safe"
OUTPUT_RAW = OUTPUT_ROOT / "raw"
OUTPUT_SUMMARY = OUTPUT_ROOT / "summary"

# IMPORTANT:
# Do NOT write demo/inference videos into your final 888-row training metadata.
# This file is only for newly extracted inference/demo videos.
METADATA_DIR = PROJECT_ROOT / "03_metadata"
METADATA_PATH = METADATA_DIR / "inference_extraction_metadata.csv"

MODEL_PATH = PROJECT_ROOT / "mediapipe_models" / "pose_landmarker.task"

VIDEO_EXTENSIONS = [
    ".mp4", ".avi", ".mov", ".mkv",
    ".MP4", ".AVI", ".MOV", ".MKV"
]


# =========================================================
# SETTINGS
# =========================================================

TARGET_FPS = 30.0

# Key lower-body joints for gait quality:
# hips, knees, ankles, feet
KEY_IDS = [23, 24, 25, 26, 27, 28, 31, 32]

VIS_THRESHOLD = 0.30
INFRAME_EPS = 0.10
MIN_GOOD_JOINTS = 5

MAX_INTERP_GAP = 3
ENABLE_SMOOTHING = False
SMOOTH_WINDOW = 3

# Practical collection thresholds
MIN_VALID_FRAMES_STRONG = 60
MIN_VALID_FRAMES_BORDERLINE = 40
MIN_VALID_RATIO = 0.35
MAX_POOR_RATIO = 0.40
MIN_CLEAN_VALID_RATIO = 0.95
MIN_CLEAN_VISIBILITY = 0.70

VALID_CLASSES = {
    "normal",
    "pd",
    "sca",
    "koa",
    "hemiplegia",
    "diplegia",
    "neuropathic",
    "antalgic",
}


# =========================================================
# MEDIAPIPE SETUP
# =========================================================

BaseOptions = python.BaseOptions
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode


def create_landmarker():
    """
    Create a fresh landmarker for each video.

    This avoids MediaPipe VIDEO-mode timestamp errors when each
    new video starts again at timestamp 0.
    """
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
    )
    return vision.PoseLandmarker.create_from_options(options)


# =========================================================
# FOLDER SETUP
# =========================================================

def ensure_dirs():
    INPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    for root in [OUTPUT_TRAINING_SAFE, OUTPUT_RAW, OUTPUT_SUMMARY]:
        root.mkdir(parents=True, exist_ok=True)
        for cls in VALID_CLASSES:
            (root / cls).mkdir(parents=True, exist_ok=True)


# =========================================================
# FILENAME PARSER
# =========================================================

def parse_video_name(video_path: Path):
    """
    Required filename format:

    class_direction_personID_fps.ext

    Examples:
    pd_L2R_P999_30fps.mp4
    pd_R2L_P999_50fps.mov
    koa_L2R_P888_50fps.mov
    normal_R2L_P777_30fps.mp4
    """

    stem = video_path.stem.strip()

    pattern = (
        r"^(normal|pd|sca|koa|hemiplegia|diplegia|neuropathic|antalgic)"
        r"_(L2R|R2L)"
        r"_(P\d+)"
        r"_(30fps|50fps)$"
    )

    match = re.match(pattern, stem, re.IGNORECASE)

    if not match:
        raise ValueError(
            f"Bad filename: {video_path.name}\n"
            f"Use format like: pd_L2R_P999_30fps.mp4"
        )

    class_label = match.group(1).lower()
    direction = match.group(2).upper()
    person_id = match.group(3).upper()
    fps_label = match.group(4).lower()
    fps_from_name = int(fps_label.replace("fps", ""))

    binary_label = "normal" if class_label == "normal" else "abnormal"

    return {
        "base_name": stem,
        "class_label": class_label,
        "binary_label": binary_label,
        "direction": direction,
        "person_id": person_id,
        "fps_from_name": fps_from_name,
    }


# =========================================================
# QUALITY HELPERS
# =========================================================

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

    return (
        avg_key_visibility,
        visibility_quality,
        good_counts,
        vis_fail,
        frame_fail,
        valid_frame,
    )


def find_longest_valid_run(mask):
    """
    Finds longest continuous valid pose segment.
    This is useful for trimming entry/exit/turning/non-visible regions.
    """
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
    """
    Interpolate only short internal gaps.
    Long gaps remain NaN so we do not invent long fake motion.
    """
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

    df[cols] = df[cols].rolling(
        window,
        center=True,
        min_periods=1,
    ).median()

    return df


def decide_clip_status(df_clean, valid_ratio_full, poor_ratio_full):
    clean_frames = len(df_clean)

    clean_valid_ratio = (
        float(df_clean["valid_frame"].mean())
        if clean_frames
        else 0.0
    )

    clean_mean_visibility = (
        float(df_clean["avg_key_visibility"].mean())
        if clean_frames
        else 0.0
    )

    clean_nan_count = (
        int(df_clean.isna().sum().sum())
        if clean_frames
        else 999999
    )

    # Strong keep
    if (
        clean_frames >= MIN_VALID_FRAMES_STRONG
        and clean_valid_ratio >= MIN_CLEAN_VALID_RATIO
        and clean_mean_visibility >= MIN_CLEAN_VISIBILITY
        and clean_nan_count == 0
        and poor_ratio_full <= MAX_POOR_RATIO
    ):
        return "accepted", False

    # Practical accepted borderline
    if (
        clean_frames >= MIN_VALID_FRAMES_BORDERLINE
        and clean_valid_ratio >= MIN_CLEAN_VALID_RATIO
        and clean_mean_visibility >= 0.65
        and clean_nan_count == 0
        and valid_ratio_full >= MIN_VALID_RATIO
    ):
        return "accepted", False

    # Keep for manual review, but still usable if needed
    if (
        clean_frames >= MIN_VALID_FRAMES_BORDERLINE
        and clean_valid_ratio >= 0.90
        and clean_mean_visibility >= 0.60
        and clean_nan_count == 0
    ):
        return "review_keep", True

    return "reject", True


# =========================================================
# METADATA
# =========================================================

def infer_source_type(person_id):
    """
    Default rule:
    P001-P050 = simulated
    P051+ = real/public/demo

    For demo/inference videos like P999, this becomes real.
    """
    number_part = re.sub(r"\D", "", person_id)

    if number_part == "":
        return "unknown"

    pid_num = int(number_part)

    if pid_num <= 50:
        return "simulated"

    return "real"


def append_metadata(row):
    """
    Saves only inference/demo extraction metadata.

    It replaces the row if the same training_safe_file already exists.
    It does NOT touch your final training dataset metadata.
    """
    metadata_columns = [
        "file_name",
        "original_video",
        "class_label",
        "binary_label",
        "person_id",
        "subject_id",
        "direction",
        "fps_from_filename",
        "fps_used",
        "clean_frames",
        "needs_padding",
        "clip_status",
        "clip_review_required",
        "source_type",
        "raw_file",
        "training_safe_file",
        "summary_file",
    ]

    row_clean = {col: row.get(col, "") for col in metadata_columns}

    row_clean["person_id"] = str(row_clean["person_id"]).strip().upper()
    row_clean["class_label"] = str(row_clean["class_label"]).strip().lower()
    row_clean["subject_id"] = (
        row_clean["class_label"] + "_" + row_clean["person_id"]
    )

    new_row_df = pd.DataFrame([row_clean], columns=metadata_columns)

    if METADATA_PATH.exists():
        old_df = pd.read_csv(METADATA_PATH, dtype=str)

        if "file_name" in old_df.columns:
            old_df = old_df[old_df["file_name"] != row_clean["file_name"]]

        final_df = pd.concat([old_df, new_row_df], ignore_index=True)
    else:
        final_df = new_row_df

    final_df.to_csv(METADATA_PATH, index=False)


# =========================================================
# EXTRACT ONE VIDEO
# =========================================================

def extract_one_video(video_path):
    info = parse_video_name(video_path)

    class_label = info["class_label"]
    binary_label = info["binary_label"]
    person_id = info["person_id"]
    direction = info["direction"]
    base_name = info["base_name"]

    raw_csv_name = f"{base_name}_raw_landmarks.csv"
    clean_csv_name = f"{base_name}_training_safe_landmarks.csv"
    summary_csv_name = f"{base_name}_quality_summary.csv"

    output_raw_csv = OUTPUT_RAW / class_label / raw_csv_name
    output_clean_csv = OUTPUT_TRAINING_SAFE / class_label / clean_csv_name
    output_summary_csv = OUTPUT_SUMMARY / class_label / summary_csv_name

    print("\n===================================================")
    print("Processing:", video_path.name)
    print("Class:", class_label)
    print("Binary:", binary_label)
    print("Person ID:", person_id)
    print("Direction:", direction)
    print("===================================================")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Could not open video:", video_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0 or np.isnan(fps):
        fps = TARGET_FPS

    metadata_cols = ["frame_index", "timestamp_ms"]

    landmark_cols = [
        f"lm{i}_{axis}"
        for i in range(33)
        for axis in ("x", "y", "z", "vis")
    ]

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

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb,
            )

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
        print("No frames extracted:", video_path.name)
        return None

    df = pd.DataFrame(rows, columns=metadata_cols + landmark_cols)

    (
        avg_key_visibility,
        visibility_quality,
        good_counts,
        vis_fail,
        frame_fail,
        valid_frame,
    ) = compute_frame_quality(
        df,
        KEY_IDS,
        VIS_THRESHOLD,
        INFRAME_EPS,
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

    poor_ratio_full = (
        float((df["visibility_quality"] == "Poor").mean())
        if len(df)
        else 1.0
    )

    valid_mask = df["valid_frame"].to_numpy().astype(bool)
    start_idx, end_idx = find_longest_valid_run(valid_mask)

    if start_idx is None or end_idx is None:
        df_clean = df.copy()
    else:
        df_clean = df.iloc[start_idx:end_idx].copy()

    # Expand very short useful segments slightly if possible.
    if (
        len(df_clean) > 0
        and len(df_clean) < MIN_VALID_FRAMES_BORDERLINE
        and start_idx is not None
    ):
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

    (
        avg_key_visibility2,
        visibility_quality2,
        good_counts2,
        vis_fail2,
        frame_fail2,
        valid_frame2,
    ) = compute_frame_quality(
        df_clean,
        KEY_IDS,
        VIS_THRESHOLD,
        INFRAME_EPS,
    )

    df_clean["avg_key_visibility"] = avg_key_visibility2
    df_clean["visibility_quality"] = visibility_quality2
    df_clean["good_key_joints"] = good_counts2
    df_clean["visibility_fail_count"] = vis_fail2
    df_clean["out_of_frame_count"] = frame_fail2
    df_clean["valid_frame"] = valid_frame2

    df_clean = optional_smooth(
        df_clean,
        xyz_cols,
        ENABLE_SMOOTHING,
        SMOOTH_WINDOW,
    )

    # Recompute after optional smoothing.
    (
        avg_key_visibility3,
        visibility_quality3,
        good_counts3,
        vis_fail3,
        frame_fail3,
        valid_frame3,
    ) = compute_frame_quality(
        df_clean,
        KEY_IDS,
        VIS_THRESHOLD,
        INFRAME_EPS,
    )

    df_clean["avg_key_visibility"] = avg_key_visibility3
    df_clean["visibility_quality"] = visibility_quality3
    df_clean["good_key_joints"] = good_counts3
    df_clean["visibility_fail_count"] = vis_fail3
    df_clean["out_of_frame_count"] = frame_fail3
    df_clean["valid_frame"] = valid_frame3

    clip_status, clip_review_required = decide_clip_status(
        df_clean,
        valid_ratio_full,
        poor_ratio_full,
    )

    clean_frames = len(df_clean)
    clean_valid_frames = int(df_clean["valid_frame"].sum()) if clean_frames else 0
    clean_valid_ratio = float(df_clean["valid_frame"].mean()) if clean_frames else 0.0

    clean_mean_visibility = (
        float(df_clean["avg_key_visibility"].mean())
        if clean_frames
        else 0.0
    )

    clean_has_nans = int(df_clean.isna().sum().sum()) if clean_frames else np.nan
    needs_padding = "yes" if clean_frames < 60 else "no"

    df_clean = df_clean.copy()
    df_clean["clip_status"] = clip_status
    df_clean["clip_review_required"] = clip_review_required

    df_clean.to_csv(output_clean_csv, index=False)

    summary = {
        "original_video": video_path.name,
        "class_label": class_label,
        "binary_label": binary_label,
        "person_id": person_id,
        "subject_id": f"{class_label}_{person_id}",
        "direction": direction,
        "fps_from_filename": info["fps_from_name"],
        "fps_used": int(round(fps)),
        "original_frames": len(df),
        "clean_frames": clean_frames,
        "missing_detections_before_fill": missing_detections,
        "valid_ratio_full": round(valid_ratio_full, 4),
        "poor_ratio_full": round(poor_ratio_full, 4),
        "mean_avg_key_visibility_full": (
            round(df["avg_key_visibility"].mean(), 4)
            if len(df)
            else np.nan
        ),
        "mean_avg_key_visibility_clean": round(clean_mean_visibility, 4),
        "clean_valid_frames": clean_valid_frames,
        "clean_valid_ratio": round(clean_valid_ratio, 4),
        "clean_has_nans": clean_has_nans,
        "clean_start_frame": (
            int(df_clean["frame_index"].iloc[0])
            if clean_frames
            else np.nan
        ),
        "clean_end_frame": (
            int(df_clean["frame_index"].iloc[-1])
            if clean_frames
            else np.nan
        ),
        "needs_padding": needs_padding,
        "clip_status": clip_status,
        "clip_review_required": clip_review_required,
        "raw_file": raw_csv_name,
        "training_safe_file": clean_csv_name,
        "summary_file": summary_csv_name,
    }

    pd.DataFrame([summary]).to_csv(output_summary_csv, index=False)

    source_type = infer_source_type(person_id)

    metadata_row = {
        "file_name": clean_csv_name,
        "original_video": video_path.name,
        "class_label": class_label,
        "binary_label": binary_label,
        "person_id": person_id,
        "subject_id": f"{class_label}_{person_id}",
        "direction": direction,
        "fps_from_filename": info["fps_from_name"],
        "fps_used": int(round(fps)),
        "clean_frames": clean_frames,
        "needs_padding": needs_padding,
        "clip_status": clip_status,
        "clip_review_required": clip_review_required,
        "source_type": source_type,
        "raw_file": raw_csv_name,
        "training_safe_file": clean_csv_name,
        "summary_file": summary_csv_name,
    }

    append_metadata(metadata_row)

    print("Saved raw:     ", output_raw_csv)
    print("Saved clean:   ", output_clean_csv)
    print("Saved summary: ", output_summary_csv)
    print("Metadata saved:", METADATA_PATH)
    print("Metadata row person_id:", person_id)
    print("Clean frames:", clean_frames)
    print("Status:", clip_status)
    print("Needs padding:", needs_padding)

    return {
        "training_safe_csv": output_clean_csv,
        "raw_csv": output_raw_csv,
        "summary_csv": output_summary_csv,
        "clip_status": clip_status,
        "clip_review_required": clip_review_required,
    }


# =========================================================
# MAIN
# =========================================================

def main():
    ensure_dirs()

    if not MODEL_PATH.exists():
        print("ERROR: pose_landmarker.task not found at:")
        print(MODEL_PATH)
        print("Put pose_landmarker.task inside:")
        print(PROJECT_ROOT / "models")
        return

    videos = []

    for ext in VIDEO_EXTENSIONS:
        videos.extend(INPUT_VIDEO_DIR.glob(f"*{ext}"))

    # Remove duplicates caused by uppercase/lowercase extension search.
    videos = sorted(set(videos))

    if not videos:
        print("No videos found in:", INPUT_VIDEO_DIR)
        print("Use names like: pd_L2R_P999_30fps.mp4")
        return

    print("Project root:", PROJECT_ROOT)
    print("Input video folder:", INPUT_VIDEO_DIR)
    print("Videos found:", len(videos))
    print("Inference metadata:", METADATA_PATH)

    extracted_files = []

    for video_path in videos:
        try:
            result = extract_one_video(video_path)

            if result is not None:
                extracted_files.append(result["training_safe_csv"])

        except Exception as e:
            print("\nFAILED:", video_path.name)
            print("Reason:", e)

    print("\nDONE")
    print("Inference metadata saved to:", METADATA_PATH)

    if extracted_files:
        print("\nTraining-safe CSVs created:")
        for p in extracted_files:
            print(p)


if __name__ == "__main__":
    main()