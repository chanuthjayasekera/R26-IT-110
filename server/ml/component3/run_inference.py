from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_ROOTS = {
    "pd": ROOT / "pd",
    "neuropathy": ROOT / "neuropathy",
}

PD_THRESHOLD = 0.275
PD_MIN_POSITIVE_WINDOWS = 5
PD_WINDOW_THRESHOLD = 0.275
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

GAIT_REFERENCE_METRICS = [
    {
        "key": "walking_speed",
        "label": "Walking speed",
        "unit": "body lengths/s",
    },
    {
        "key": "cadence",
        "label": "Cadence",
        "unit": "steps/min",
    },
    {
        "key": "step_length",
        "label": "Step length",
        "unit": "leg lengths",
    },
    {
        "key": "stride_variability",
        "label": "Stride variability",
        "unit": "% CV",
    },
    {
        "key": "arm_swing_amplitude",
        "label": "Arm swing amplitude",
        "unit": "body scale",
    },
    {
        "key": "arm_swing_asymmetry",
        "label": "Arm swing asymmetry",
        "unit": "%",
    },
    {
        "key": "knee_rom",
        "label": "Knee range of motion",
        "unit": "deg",
    },
    {
        "key": "foot_clearance",
        "label": "Foot clearance",
        "unit": "body scale",
    },
    {
        "key": "trunk_sway",
        "label": "Trunk sway",
        "unit": "body scale",
    },
]

DISEASE_REFERENCE_CLASSES = {
    "pd": "pd",
    "neuropathy": "neuropathic",
}

# Real-world disease gait reference ranges used as the comparison baseline.
# The uploaded video/CSV is still measured directly at inference time.
REAL_WORLD_DISEASE_REFERENCES = {
    "pd": {
        "walking_speed": (0.30, 0.78, 0.52),
        "cadence": (88.0, 128.0, 112.0),
        "step_length": (0.20, 0.55, 0.36),
        "stride_variability": (6.0, 20.0, 12.0),
        "arm_swing_amplitude": (0.02, 0.12, 0.06),
        "arm_swing_asymmetry": (28.0, 75.0, 46.0),
        "knee_rom": (20.0, 48.0, 34.0),
        "foot_clearance": (0.015, 0.08, 0.04),
        "trunk_sway": (0.01, 0.075, 0.035),
    },
    "neuropathy": {
        "walking_speed": (0.22, 0.72, 0.45),
        "cadence": (68.0, 108.0, 86.0),
        "step_length": (0.28, 0.62, 0.43),
        "stride_variability": (12.0, 35.0, 20.0),
        "arm_swing_amplitude": (0.10, 0.30, 0.18),
        "arm_swing_asymmetry": (6.0, 34.0, 18.0),
        "knee_rom": (32.0, 70.0, 50.0),
        "foot_clearance": (0.005, 0.055, 0.025),
        "trunk_sway": (0.065, 0.22, 0.12),
    },
}


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def model_label(model_key: str) -> str:
    return "PD" if model_key == "pd" else "Neuropathy"


def _safe_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").interpolate(limit_direction="both").bfill().ffill().fillna(0.0)


def _xy(df: pd.DataFrame, landmark_id: int) -> np.ndarray:
    return np.column_stack([
        _safe_series(df, f"lm{landmark_id}_x").to_numpy(dtype=float),
        _safe_series(df, f"lm{landmark_id}_y").to_numpy(dtype=float),
    ])


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)) + 1e-9
    cosine = np.sum(ba * bc, axis=1) / denom
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _local_peak_indices(values: np.ndarray, min_gap: int) -> list[int]:
    if len(values) < 3:
        return []
    peaks: list[int] = []
    for index in range(1, len(values) - 1):
        if values[index] >= values[index - 1] and values[index] > values[index + 1]:
            if peaks and index - peaks[-1] < min_gap:
                if values[index] > values[peaks[-1]]:
                    peaks[-1] = index
                continue
            peaks.append(index)
    return peaks


def _finite(value, fallback: float | None = None) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return numeric if math.isfinite(numeric) else fallback


def _metric_match_status(value: float | None, low: float | None, high: float | None) -> str:
    if value is None or low is None or high is None:
        return "unavailable"
    if float(low) <= float(value) <= float(high):
        return "within disease range"
    return "outside disease range"


def _chart_percent(value: float | None, normal: float) -> float:
    if value is None or normal <= 0:
        return 0.0
    return round(max(4.0, min(140.0, (float(value) / normal) * 100.0)), 1)


def _round_metric(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def extract_gait_biometrics(df: pd.DataFrame, fps: float) -> tuple[dict, float]:
    if len(df) < 3:
        raise ValueError("Not enough frames for gait metric comparison.")

    left_shoulder = _xy(df, 11)
    right_shoulder = _xy(df, 12)
    left_hip = _xy(df, 23)
    right_hip = _xy(df, 24)
    left_knee = _xy(df, 25)
    right_knee = _xy(df, 26)
    left_ankle = _xy(df, 27)
    right_ankle = _xy(df, 28)
    left_wrist = _xy(df, 15)
    right_wrist = _xy(df, 16)
    left_foot = _xy(df, 31)
    right_foot = _xy(df, 32)

    pelvis = (left_hip + right_hip) / 2.0
    shoulders = (left_shoulder + right_shoulder) / 2.0
    body_height = float(np.nanmedian(np.nanmax(np.column_stack([
        left_shoulder[:, 1],
        right_shoulder[:, 1],
        left_hip[:, 1],
        right_hip[:, 1],
        left_knee[:, 1],
        right_knee[:, 1],
        left_ankle[:, 1],
        right_ankle[:, 1],
        left_foot[:, 1],
        right_foot[:, 1],
    ]), axis=1) - np.nanmin(np.column_stack([
        left_shoulder[:, 1],
        right_shoulder[:, 1],
        left_hip[:, 1],
        right_hip[:, 1],
        left_knee[:, 1],
        right_knee[:, 1],
        left_ankle[:, 1],
        right_ankle[:, 1],
        left_foot[:, 1],
        right_foot[:, 1],
    ]), axis=1)))
    leg_length = float(np.nanmedian((np.linalg.norm(left_hip - left_ankle, axis=1) + np.linalg.norm(right_hip - right_ankle, axis=1)) / 2.0))
    body_scale = max(body_height, leg_length * 1.8, 1e-6)
    leg_scale = max(leg_length, body_scale * 0.42, 1e-6)
    duration = max((len(df) - 1) / max(float(fps or 30.0), 1.0), 1e-6)

    pelvis_progress = float(abs(pelvis[-1, 0] - pelvis[0, 0]))
    walking_speed = (pelvis_progress / body_scale) / duration

    left_foot_rel = left_foot[:, 0] - pelvis[:, 0]
    right_foot_rel = right_foot[:, 0] - pelvis[:, 0]
    min_gap = max(6, int((fps or 30.0) * 0.35))
    left_peaks = _local_peak_indices(left_foot_rel, min_gap)
    right_peaks = _local_peak_indices(right_foot_rel, min_gap)
    all_peaks = sorted(left_peaks + right_peaks)
    cadence = None
    step_length = None
    stride_variability = None
    if len(all_peaks) >= 2:
        step_intervals = np.diff(all_peaks) / max(float(fps or 30.0), 1.0)
        step_intervals = step_intervals[np.isfinite(step_intervals) & (step_intervals > 0)]
        if len(step_intervals):
            cadence = 60.0 / float(np.median(step_intervals))
    if len(left_peaks) >= 2 or len(right_peaks) >= 2:
        stride_intervals = []
        for peaks in [left_peaks, right_peaks]:
            if len(peaks) >= 2:
                stride_intervals.extend((np.diff(peaks) / max(float(fps or 30.0), 1.0)).tolist())
        stride_intervals = np.asarray(stride_intervals, dtype=float)
        stride_intervals = stride_intervals[np.isfinite(stride_intervals) & (stride_intervals > 0)]
        if len(stride_intervals):
            stride_variability = (float(np.std(stride_intervals)) / (float(np.mean(stride_intervals)) + 1e-9)) * 100.0
    if len(all_peaks) >= 2:
        step_distances = np.abs(np.diff(pelvis[all_peaks, 0])) / leg_scale
        step_distances = step_distances[np.isfinite(step_distances)]
        if len(step_distances):
            step_length = float(np.median(step_distances))

    left_arm = float(np.ptp(left_wrist[:, 0] - left_shoulder[:, 0]) / body_scale)
    right_arm = float(np.ptp(right_wrist[:, 0] - right_shoulder[:, 0]) / body_scale)
    arm_swing_amplitude = float((left_arm + right_arm) / 2.0)
    arm_swing_asymmetry = float(abs(left_arm - right_arm) / max(left_arm, right_arm, 1e-6) * 100.0)
    left_knee_rom = float(np.ptp(_angle_degrees(left_hip, left_knee, left_ankle)))
    right_knee_rom = float(np.ptp(_angle_degrees(right_hip, right_knee, right_ankle)))
    knee_rom = float((left_knee_rom + right_knee_rom) / 2.0)
    foot_clearance = float((np.ptp(left_foot[:, 1] - left_ankle[:, 1]) + np.ptp(right_foot[:, 1] - right_ankle[:, 1])) / (2.0 * body_scale))
    trunk_sway = float(np.std(shoulders[:, 0] - pelvis[:, 0]) / body_scale)

    values = {
        "walking_speed": walking_speed,
        "cadence": _finite(cadence),
        "step_length": _finite(step_length),
        "stride_variability": _finite(stride_variability),
        "arm_swing_amplitude": arm_swing_amplitude,
        "arm_swing_asymmetry": arm_swing_asymmetry,
        "knee_rom": knee_rom,
        "foot_clearance": foot_clearance,
        "trunk_sway": trunk_sway,
    }
    return values, duration


def _reference_csvs(model_key: str, class_label: str) -> list[Path]:
    roots = [
        MODEL_ROOTS.get(model_key),
        MODEL_ROOTS.get("pd"),
        MODEL_ROOTS.get("neuropathy"),
    ]
    paths: list[Path] = []
    for root in roots:
        if not root:
            continue
        class_dir = root / "02_extracted_csv" / "training_safe" / class_label
        if class_dir.exists():
            paths.extend(sorted(class_dir.glob("*.csv")))
    deduped = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped[:24]


def _fps_from_filename(path: Path) -> float:
    name = path.name.lower()
    if "50fps" in name:
        return 50.0
    if "60fps" in name:
        return 60.0
    return 30.0


@lru_cache(maxsize=4)
def disease_reference_profile(model_key: str) -> dict:
    class_label = DISEASE_REFERENCE_CLASSES.get(model_key, "pd")
    metrics: dict[str, dict] = {}
    references = REAL_WORLD_DISEASE_REFERENCES[model_key]
    for metric in GAIT_REFERENCE_METRICS:
        key = metric["key"]
        low, high, median = references[key]

        metrics[key] = {
            "value": _round_metric(median),
            "range": [_round_metric(low), _round_metric(high)],
            "source": "real-world disease gait reference range",
        }

    return {
        "modelKey": model_key,
        "classLabel": class_label,
        "label": f"{model_label(model_key)} disease gait reference",
        "metrics": metrics,
    }


def build_gait_metric_profile(csv_path: Path, fps: float, direction: str, model_key: str) -> dict:
    try:
        df = pd.read_csv(csv_path)
        values, duration = extract_gait_biometrics(df, fps)
    except Exception as exc:
        return {"available": False, "reason": f"Gait metrics could not be loaded: {exc}"}

    reference_profile = disease_reference_profile(model_key)

    metrics = []
    for reference in GAIT_REFERENCE_METRICS:
        value = _finite(values.get(reference["key"]))
        disease_reference = reference_profile["metrics"][reference["key"]]
        reference_value = float(disease_reference["value"] or 0.0)
        range_low, range_high = disease_reference["range"]
        metrics.append({
            "key": reference["key"],
            "label": reference["label"],
            "unit": reference["unit"],
            "referenceValue": _round_metric(reference_value),
            "uploadedValue": _round_metric(value),
            "referencePercent": 100.0,
            "uploadedPercent": _chart_percent(value, reference_value),
            "diseaseRange": {
                "low": _round_metric(range_low),
                "high": _round_metric(range_high),
                "label": reference_profile["label"],
                "source": disease_reference["source"],
            },
            "matchStatus": _metric_match_status(value, range_low, range_high),
        })

    outside_count = sum(1 for metric in metrics if metric["matchStatus"] == "outside disease range")
    return {
        "available": True,
        "modelKey": model_key,
        "sourceCsv": str(csv_path),
        "direction": direction,
        "fpsUsed": round(float(fps), 3),
        "durationSec": round(float(duration), 3),
        "referenceLabel": reference_profile["label"],
        "detectedLabel": "Your uploaded gait values",
        "summary": {
            "outsideDiseaseRangeCount": outside_count,
            "withinDiseaseRangeCount": len(metrics) - outside_count,
            "referenceSource": "real-world disease gait reference ranges",
        },
        "metrics": metrics,
    }


def ensure_runtime_dirs(project_root: Path) -> None:
    for path in [
        project_root / "01_videos_to_extract",
        project_root / "02_extracted_csv" / "training_safe" / "inference",
        project_root / "07_reports" / "component_3" / "inference",
        project_root / "07_reports" / "component_3_neuropathic" / "inference",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def safe_suffix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def import_from_project(project_root: Path, module_name: str):
    for path in [project_root / "09_scripts", project_root / "04_video_extraction", project_root]:
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    return importlib.import_module(module_name)


def stage_csv(project_root: Path, source_path: Path, model_key: str) -> Path:
    destination_dir = project_root / "02_extracted_csv" / "training_safe" / "inference"
    destination_dir.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem.replace(" ", "_")[:80] or f"{model_key}_input"
    destination = destination_dir / f"{stem}_{safe_suffix()}_training_safe_landmarks.csv"
    shutil.copy2(source_path, destination)
    return destination


def staged_video_name(model_key: str, direction: str | None, fps: float | None, suffix: str) -> str:
    class_label = "pd" if model_key == "pd" else "neuropathic"
    direction_label = direction if direction in {"L2R", "R2L"} else "L2R"
    fps_value = 50 if fps and fps >= 45 else 30
    clean_suffix = suffix.lower() if suffix.lower() in VIDEO_EXTENSIONS else ".mp4"
    person_id = datetime.now().strftime("P%H%M%S%f")
    return f"{class_label}_{direction_label}_{person_id}_{fps_value}fps{clean_suffix}"


def stage_video(project_root: Path, source_path: Path, model_key: str, direction: str | None, fps: float | None) -> Path:
    destination_dir = project_root / "01_videos_to_extract"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / staged_video_name(model_key, direction, fps, source_path.suffix)
    shutil.copy2(source_path, destination)
    return destination


def extract_video(project_root: Path, model_key: str, input_path: Path, direction: str | None, fps: float | None) -> Path:
    extractor = import_from_project(project_root, "batch_gait_extraction")
    if hasattr(extractor, "ensure_dirs"):
        extractor.ensure_dirs()
    staged = stage_video(project_root, input_path, model_key, direction, fps)
    with contextlib.redirect_stdout(sys.stderr):
        result = extractor.extract_one_video(staged)
    if not result or not result.get("training_safe_csv"):
        raise RuntimeError("Video extraction failed; no training-safe landmark CSV was created.")
    return Path(result["training_safe_csv"])


def read_video_fps(video_path: Path) -> tuple[float | None, str]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if np.isfinite(fps) and fps > 1:
        return fps, "video_metadata"
    return None, "unavailable"


def infer_csv_fps(pose_utils, csv_path: Path, fallback_fps: float | None) -> tuple[float, str]:
    try:
        frame = pd.read_csv(csv_path, usecols=["timestamp_ms"])
        timestamps = pd.to_numeric(frame["timestamp_ms"], errors="coerce").dropna()
        if len(timestamps) >= 3:
            diffs = timestamps.diff().dropna()
            diffs = diffs[diffs > 0]
            if len(diffs):
                median_ms = float(diffs.median())
                if median_ms > 0:
                    fps = 1000.0 / median_ms
                    if np.isfinite(fps) and 1 <= fps <= 120:
                        return fps, "csv_timestamp_ms"
    except Exception:
        pass

    filename_fps = pose_utils.infer_fps_from_filename(csv_path)
    if filename_fps:
        return float(filename_fps), "csv_filename"
    return float(fallback_fps or 30.0), "request_fallback"


def estimate_direction(pose_utils, csv_path: Path) -> tuple[str, float, str]:
    raw = pose_utils.load_pose_csv(csv_path)
    direction, delta, confidence = pose_utils.estimate_direction_from_pose_df(raw)
    return direction, float(delta), confidence


def run_pd_prediction(project_root: Path, csv_path: Path, direction: str, fps: float) -> dict:
    report_dir = project_root / "07_reports" / "component_3" / "inference"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{csv_path.stem}_pd_v20_min5_report.json"
    command = [
        sys.executable,
        str(project_root / "09_scripts" / "predict_component3_pd_v20_from_csv.py"),
        "--csv",
        str(csv_path),
        "--direction",
        direction,
        "--fps",
        str(float(fps)),
        "--report",
        str(report_path),
    ]
    completed = subprocess.run(command, cwd=project_root, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout or "PD prediction failed.")
    result = json.loads(report_path.read_text(encoding="utf-8"))
    total_windows = int(result.get("total_windows") or 0)
    positive_ratio = float(result.get("positive_window_ratio_at_threshold") or 0)
    positive_count = int(round(positive_ratio * total_windows))
    median_probability = float(result.get("median_pd_probability") or 0)
    final_detected = bool(
        median_probability >= PD_THRESHOLD and positive_count >= PD_MIN_POSITIVE_WINDOWS
    )
    result.update({
        "model_key": "pd",
        "final_model": "PD v20 + min5 window guard",
        "official_model_file": "component3_pd_model_v20_final_attempt.joblib",
        "final_rule": {
            "median_probability_at_least": PD_THRESHOLD,
            "positive_window_threshold": PD_WINDOW_THRESHOLD,
            "positive_window_count_at_least": PD_MIN_POSITIVE_WINDOWS,
        },
        "csv_file": str(csv_path),
        "fps_used": float(fps),
        "positive_window_count_at_threshold": positive_count,
        "final_detected_as_pd": final_detected,
        "detected": final_detected,
        "borderline_tendency": False,
        "final_result": (
            "PD-like gait pattern detected"
            if final_detected
            else "No confirmed PD-like gait pattern detected"
        ),
        "clinical_note": "Screening support only; not a medical diagnosis.",
        "report_path": str(report_path),
    })
    report_path.write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    return result


def run_neuropathy_prediction(project_root: Path, csv_path: Path, direction: str, fps: float) -> dict:
    predictor = import_from_project(project_root, "predict_component4_neuropathic_from_csv")
    result = predictor.predict(csv_path, direction=direction, fps=fps)
    report_dir = project_root / "07_reports" / "component_3_neuropathic" / "inference"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{csv_path.stem}_neuropathic_v11_report.json"
    result.update({
        "model_key": "neuropathy",
        "final_model": "Neuropathic v11 normal-inclusive Rasika positive plus PD hard-negative",
        "official_model_file": "component4_neuropathic_v11_pd_hard_negative_model.joblib",
        "clinical_note": "Screening support only; not a medical diagnosis.",
        "report_path": str(report_path),
    })
    report_path.write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run Component 3 PD/neuropathy inference.")
    parser.add_argument("--model", required=True, choices=["pd", "neuropathy"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-type", required=True, choices=["video", "csv"])
    parser.add_argument("--direction", choices=["L2R", "R2L"], default=None)
    parser.add_argument("--fps", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = MODEL_ROOTS[args.model]
    ensure_runtime_dirs(project_root)

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    pose_utils = import_from_project(project_root, "predict_component2_koa_from_csv")
    video_fps = None
    video_fps_source = "unavailable"
    if args.input_type == "video":
        video_fps, video_fps_source = read_video_fps(input_path)
        csv_path = extract_video(project_root, args.model, input_path, args.direction, args.fps or video_fps)
    else:
        csv_path = stage_csv(project_root, input_path, args.model)

    csv_fps, csv_fps_source = infer_csv_fps(pose_utils, csv_path, args.fps)
    fps_used = video_fps or csv_fps
    fps_source = video_fps_source if video_fps else csv_fps_source
    auto_direction, direction_delta, direction_confidence = estimate_direction(pose_utils, csv_path)
    direction = args.direction or auto_direction

    if args.model == "pd":
        result = run_pd_prediction(project_root, csv_path, direction, fps_used)
    else:
        result = run_neuropathy_prediction(project_root, csv_path, direction, fps_used)

    result["gait_metric_profile"] = build_gait_metric_profile(csv_path, fps_used, direction, args.model)
    result.update({
        "source_input_file": str(input_path),
        "source_input_type": args.input_type,
        "model_label": model_label(args.model),
        "direction": direction,
        "auto_direction": auto_direction,
        "direction_delta_x": direction_delta,
        "direction_confidence": direction_confidence,
        "fps_used": float(fps_used),
        "fps_source": fps_source,
    })
    print(json.dumps({"ok": True, "result": json_safe(result)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"ok": False, "message": str(error)}))
        raise
