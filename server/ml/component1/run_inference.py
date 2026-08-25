import argparse
import contextlib
import importlib.util
import json
import math
import shutil
import sys
import traceback
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SERVER_ROOT = PROJECT_ROOT.parents[1]
RUN_ROOT = SERVER_ROOT / "data" / "component1-runs"
sys.path.insert(0, str(PROJECT_ROOT))


def load_extractor_module():
    extractor_path = PROJECT_ROOT / "04_video_extraction" / "run_component1_clinical_video.py"
    spec = importlib.util.spec_from_file_location("component1_video_extractor", extractor_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_stem(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)
    cleaned = cleaned.strip("_")
    return cleaned[:42] or "gait_video"


def finite_number(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def rounded(value, digits=2):
    numeric = finite_number(value)
    return None if numeric is None else round(numeric, digits)


def clean_values(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def mean_sd(values, digits=2):
    arr = clean_values(values)
    if arr.size == 0:
        return None, None
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return rounded(float(np.mean(arr)), digits), rounded(sd, digits)


def range_status(value, green_low, green_high, yellow_low, yellow_high):
    numeric = finite_number(value)
    if numeric is None:
        return "unavailable"
    if green_low <= numeric <= green_high:
        return "green"
    if yellow_low <= numeric <= yellow_high:
        return "yellow"
    return "red"


def upper_status(value, green_max, yellow_max):
    numeric = finite_number(value)
    if numeric is None:
        return "unavailable"
    if numeric <= green_max:
        return "green"
    if numeric <= yellow_max:
        return "yellow"
    return "red"


def lower_status(value, green_min, yellow_min):
    numeric = finite_number(value)
    if numeric is None:
        return "unavailable"
    if numeric >= green_min:
        return "green"
    if numeric >= yellow_min:
        return "yellow"
    return "red"


COMPONENT1_REFERENCE_RANGES = {
    "walking_speed": "0.55-1.60 body lengths/s",
    "cadence": "90-130 steps/min",
    "step_length": "0.45-1.15 leg lengths",
    "stride_length": "0.90-2.30 leg lengths",
    "step_time": "0.42-0.75 s",
    "stride_time": "0.85-1.50 s",
    "swing_phase": "32-48 %",
    "stance_phase": "52-68 %",
    "symmetry_index": "<=10 %",
    "stride_variability": "<=5 % CV",
    "arm_swing_amplitude": ">=0.12 body scale",
    "knee_rom": "25-75 deg",
    "hip_rom": "15-55 deg",
    "foot_clearance": "0.03-0.22 body scale",
}

COMPONENT1_FALLBACK_BANDS = {
    "walking_speed": {"green": (0.55, 1.60), "yellow": (0.35, 2.10), "mode": "range"},
    "cadence": {"green": (90, 130), "yellow": (70, 150), "mode": "range"},
    "step_length": {"green": (0.45, 1.15), "yellow": (0.30, 1.40), "mode": "range"},
    "stride_length": {"green": (0.90, 2.30), "yellow": (0.60, 2.80), "mode": "range"},
    "step_time": {"green": (0.42, 0.75), "yellow": (0.32, 0.95), "mode": "range"},
    "stride_time": {"green": (0.85, 1.50), "yellow": (0.65, 1.90), "mode": "range"},
    "swing_phase": {"green": (32, 48), "yellow": (25, 55), "mode": "range"},
    "stance_phase": {"green": (52, 68), "yellow": (45, 75), "mode": "range"},
    "symmetry_index": {"green": (None, 10), "yellow": (None, 20), "mode": "upper"},
    "stride_variability": {"green": (None, 5), "yellow": (None, 10), "mode": "upper"},
    "arm_swing_amplitude": {"green": (0.12, None), "yellow": (0.07, None), "mode": "lower"},
    "knee_rom": {"green": (25, 75), "yellow": (15, 90), "mode": "range"},
    "hip_rom": {"green": (15, 55), "yellow": (8, 70), "mode": "range"},
    "foot_clearance": {"green": (0.03, 0.22), "yellow": (0.015, 0.32), "mode": "range"},
}


METRIC_DEFINITIONS = {
    "walking_speed": "How fast the pelvis/body moves forward during the walking clip.",
    "cadence": "Step rhythm: the number of steps taken per minute.",
    "step_length": "Distance covered between one foot contact and the next opposite-foot contact.",
    "stride_length": "Distance covered from one foot contact to the next contact of the same foot.",
    "step_time": "Time between alternating left and right step events.",
    "stride_time": "Time between repeated step events from the same foot.",
    "swing_phase": "Approximate portion of gait where the foot is moving forward through the air.",
    "stance_phase": "Approximate portion of gait where the foot is supporting body weight.",
    "symmetry_index": "Left-right balance of step length and step timing; lower asymmetry is better.",
    "stride_variability": "Step-to-step consistency; lower variability usually means steadier walking.",
    "arm_swing_amplitude": "How much each wrist swings relative to the shoulder/body scale.",
    "knee_rom": "Knee range of motion during the clean walking clip.",
    "hip_rom": "Hip range of motion during the clean walking clip.",
    "foot_clearance": "Vertical foot lift during walking; low clearance can suggest toe drag risk.",
}


def apply_screening_context(metrics, final_label=None, mean_prob_abnormal=None):
    prob = finite_number(mean_prob_abnormal)
    is_abnormal = final_label == 1 or (prob is not None and prob >= 0.406)
    is_normal = final_label == 0 or (prob is not None and prob < 0.406)
    very_abnormal = prob is not None and prob >= 0.85

    raw_green_metrics = []
    raw_red_metrics = []
    raw_red_count = 0

    for metric in metrics:
        raw_status = metric.get("status")
        metric["rawStatus"] = raw_status
        metric["clinicalStatus"] = raw_status

        if raw_status not in {"green", "yellow", "red"}:
            metric["clinicalMeaning"] = "Not enough reliable signal for this metric."
            continue

        if raw_status == "green":
            raw_green_metrics.append(metric)
        elif raw_status == "red":
            raw_red_metrics.append(metric)
            raw_red_count += 1

        if is_abnormal:
            if raw_status == "red":
                metric["clinicalMeaning"] = "Abnormal marker supporting the current screening result."
            elif raw_status == "yellow":
                metric["clinicalMeaning"] = "Borderline gait marker in the abnormal screening context."
            else:
                metric["clinicalMeaning"] = "Preserved metric inside an abnormal screening; it does not cancel the model result."
        elif is_normal:
            if raw_status == "green":
                metric["clinicalMeaning"] = "Normal marker supporting the current screening result."
            elif raw_status == "yellow":
                metric["clinicalMeaning"] = "Borderline marker to monitor even though the screening is normal."
            else:
                metric["clinicalMeaning"] = "Abnormal marker to monitor despite the normal screening result."
        else:
            metric["clinicalMeaning"] = "Metric status calculated from the Component 1 normalized band."

    if is_abnormal:
        target_green = raw_red_count
        if very_abnormal and raw_red_count > 0:
            target_green = max(1, raw_red_count - 1)
        excess_green = max(0, len(raw_green_metrics) - target_green)
        for metric in raw_green_metrics[:excess_green]:
            metric["status"] = "yellow"
            metric["clinicalStatus"] = "yellow"
            metric["clinicalMeaning"] = "Preserved metric, shown as borderline because abnormal evidence dominates this screening."
    elif is_normal:
        target_red = len(raw_green_metrics)
        if prob is not None and prob <= 0.15 and len(raw_green_metrics) > 0:
            target_red = min(2, max(0, len(raw_green_metrics) - 1))
        excess_red = max(0, len(raw_red_metrics) - target_red)
        for metric in raw_red_metrics[:excess_red]:
            metric["status"] = "yellow"
            metric["clinicalStatus"] = "yellow"
            metric["clinicalMeaning"] = "Abnormal-range metric shown as monitor item because normal evidence dominates this screening."

    return metrics


def apply_current_measurement_context(metrics, final_label=None, mean_prob_abnormal=None):
    prob = finite_number(mean_prob_abnormal)
    is_abnormal = final_label == 1 or (prob is not None and prob >= 0.406)
    is_normal = final_label == 0 or (prob is not None and prob < 0.406)

    for metric in metrics:
        metric["rawStatus"] = metric.get("status")
        status = metric.get("status")

        if status == "green":
            metric["clinicalMeaning"] = "This current measured biomarker is inside the shown reference range."
        elif status == "yellow":
            metric["clinicalMeaning"] = "This current measured biomarker is borderline against the shown reference range."
        elif status == "red":
            if is_abnormal:
                metric["clinicalMeaning"] = "This current measured biomarker is outside the shown reference range and supports the abnormal screening."
            elif is_normal:
                metric["clinicalMeaning"] = "This current measured biomarker is outside the shown reference range, but the full model still screened normal."
            else:
                metric["clinicalMeaning"] = "This current measured biomarker is outside the shown reference range."
        else:
            metric["clinicalMeaning"] = "Not enough reliable current-run signal for this biomarker."

    return metrics


def metric_row(key, label, values, unit, status, meaning, left=None, right=None, precision=2, source="Component 1 pose landmarks"):
    mean, sd = mean_sd(values, precision)
    return {
        "key": key,
        "label": label,
        "value": mean,
        "mean": mean,
        "sd": sd,
        "unit": unit,
        "status": status,
        "rawStatus": status,
        "meaning": METRIC_DEFINITIONS.get(key, meaning),
        "measurementMethod": meaning,
        "left": rounded(left, precision),
        "right": rounded(right, precision),
        "source": source,
        "referenceRange": COMPONENT1_REFERENCE_RANGES.get(key, "Varies"),
        "referenceType": "Component 1 normalized reference band",
    }


def point(df, landmark, axis):
    return pd.to_numeric(df[f"lm{landmark}_{axis}"], errors="coerce").to_numpy(dtype=float)


def distance(ax, ay, bx, by):
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def angle_series(ax, ay, bx, by, cx, cy):
    ba_x = ax - bx
    ba_y = ay - by
    bc_x = cx - bx
    bc_y = cy - by
    denom = np.sqrt(ba_x**2 + ba_y**2) * np.sqrt(bc_x**2 + bc_y**2)
    denom[denom == 0] = np.nan
    cosine = np.clip((ba_x * bc_x + ba_y * bc_y) / denom, -1, 1)
    return np.degrees(np.arccos(cosine))


def find_forward_peaks(signal, fps):
    values = np.asarray(signal, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < 8:
      return []

    finite = values[valid]
    median = float(np.nanmedian(finite))
    spread = float(np.nanstd(finite))
    amplitude = float(np.nanmax(finite) - np.nanmin(finite))
    min_gap = max(int((fps or 30) * 0.35), 6)
    min_prominence = max(spread * 0.15, amplitude * 0.05, 1e-4)

    candidates = []
    for idx in range(1, len(values) - 1):
        if not np.isfinite(values[idx - 1:idx + 2]).all():
            continue
        if values[idx] >= values[idx - 1] and values[idx] > values[idx + 1] and values[idx] >= median + min_prominence:
            candidates.append(idx)

    selected = []
    for idx in sorted(candidates, key=lambda item: values[item], reverse=True):
        if all(abs(idx - old) >= min_gap for old in selected):
            selected.append(idx)
    return sorted(selected)


def calculate_component1_biometrics(csv_path, fps_used=None, direction=None, clean_duration_sec=None, final_label=None, mean_prob_abnormal=None):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {
            "available": False,
            "referenceFrame": "Unavailable",
            "metrics": [],
            "notes": ["Biometrics could not be calculated from the landmark CSV."],
        }

    if "valid_frame" in df.columns:
        valid_mask = df["valid_frame"].astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[valid_mask].copy()

    required = [f"lm{idx}_{axis}" for idx in [11, 12, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32] for axis in ["x", "y"]]
    if df.empty or any(column not in df.columns for column in required):
        return {
            "available": False,
            "referenceFrame": "Unavailable",
            "metrics": [],
            "notes": ["Not enough valid pose landmarks were available for biometrics."],
        }

    timestamps = pd.to_numeric(df.get("timestamp_ms"), errors="coerce").to_numpy(dtype=float) / 1000.0
    if not np.isfinite(timestamps).any():
        timestamps = np.arange(len(df), dtype=float) / float(fps_used or 30)
    timestamps = timestamps - np.nanmin(timestamps)

    if fps_used is None:
        diffs = np.diff(timestamps[np.isfinite(timestamps)])
        fps_used = 1 / np.nanmedian(diffs) if diffs.size and np.nanmedian(diffs) > 0 else 30.0

    duration = finite_number(clean_duration_sec)
    if duration is None or duration <= 0:
        duration = float(np.nanmax(timestamps) - np.nanmin(timestamps)) if len(timestamps) > 1 else 0.0

    lx23, ly23 = point(df, 23, "x"), point(df, 23, "y")
    lx24, ly24 = point(df, 24, "x"), point(df, 24, "y")
    lx25, ly25 = point(df, 25, "x"), point(df, 25, "y")
    lx26, ly26 = point(df, 26, "x"), point(df, 26, "y")
    lx27, ly27 = point(df, 27, "x"), point(df, 27, "y")
    lx28, ly28 = point(df, 28, "x"), point(df, 28, "y")
    lx11, ly11 = point(df, 11, "x"), point(df, 11, "y")
    lx12, ly12 = point(df, 12, "x"), point(df, 12, "y")
    lx15, ly15 = point(df, 15, "x"), point(df, 15, "y")
    lx16, ly16 = point(df, 16, "x"), point(df, 16, "y")
    lx31, ly31 = point(df, 31, "x"), point(df, 31, "y")
    lx32, ly32 = point(df, 32, "x"), point(df, 32, "y")

    left_leg = distance(lx23, ly23, lx25, ly25) + distance(lx25, ly25, lx27, ly27)
    right_leg = distance(lx24, ly24, lx26, ly26) + distance(lx26, ly26, lx28, ly28)
    body_scale = float(np.nanmedian(np.concatenate([left_leg, right_leg])))
    if not math.isfinite(body_scale) or body_scale <= 0:
        body_scale = 1.0

    mid_hip_x = (lx23 + lx24) / 2.0
    mid_hip_y = (ly23 + ly24) / 2.0
    sample = max(int(len(mid_hip_x) * 0.1), 3)
    start_x = float(np.nanmedian(mid_hip_x[:sample]))
    end_x = float(np.nanmedian(mid_hip_x[-sample:]))
    sign = 1 if (direction or "L2R") == "L2R" else -1
    walking_speed = sign * (end_x - start_x) / body_scale / duration if duration > 0 else None
    walking_speed = abs(walking_speed) if walking_speed is not None else None

    left_forward = sign * (lx27 - mid_hip_x)
    right_forward = sign * (lx28 - mid_hip_x)
    left_peaks = find_forward_peaks(left_forward, fps_used)
    right_peaks = find_forward_peaks(right_forward, fps_used)
    events = sorted(
        [{"foot": "left", "idx": idx, "time": timestamps[idx]} for idx in left_peaks]
        + [{"foot": "right", "idx": idx, "time": timestamps[idx]} for idx in right_peaks],
        key=lambda item: item["time"],
    )

    left_step_lengths = [abs(lx27[idx] - lx28[idx]) / body_scale for idx in left_peaks if np.isfinite(lx27[idx]) and np.isfinite(lx28[idx])]
    right_step_lengths = [abs(lx27[idx] - lx28[idx]) / body_scale for idx in right_peaks if np.isfinite(lx27[idx]) and np.isfinite(lx28[idx])]
    step_lengths = left_step_lengths + right_step_lengths
    left_step_mean = float(np.mean(left_step_lengths)) if left_step_lengths else None
    right_step_mean = float(np.mean(right_step_lengths)) if right_step_lengths else None

    step_times = [
        events[idx + 1]["time"] - events[idx]["time"]
        for idx in range(len(events) - 1)
        if events[idx + 1]["foot"] != events[idx]["foot"] and events[idx + 1]["time"] > events[idx]["time"]
    ]
    cadence = 60.0 / float(np.mean(step_times)) if step_times else ((len(events) / duration) * 60.0 if duration > 0 and len(events) > 1 else None)

    stride_times = []
    stride_lengths = []
    for foot in ["left", "right"]:
        foot_events = [event for event in events if event["foot"] == foot]
        for first, second in zip(foot_events, foot_events[1:]):
            if second["time"] > first["time"]:
                stride_times.append(second["time"] - first["time"])
                stride_lengths.append(abs(mid_hip_x[second["idx"]] - mid_hip_x[first["idx"]]) / body_scale)
    if not stride_lengths and step_lengths:
        stride_lengths = [value * 2.0 for value in step_lengths]

    def moving_percent(ankle_x, ankle_y):
        rel_x = ankle_x - mid_hip_x
        rel_y = ankle_y - mid_hip_y
        movement = np.sqrt(np.diff(rel_x) ** 2 + np.diff(rel_y) ** 2)
        movement = clean_values(movement)
        if movement.size < 4:
            return None
        threshold = float(np.nanmedian(movement) + np.nanstd(movement) * 0.25)
        return float(np.mean(movement > threshold) * 100.0)

    left_swing = moving_percent(lx27, ly27)
    right_swing = moving_percent(lx28, ly28)
    swing_values = [value for value in [left_swing, right_swing] if value is not None]
    stance_values = [100.0 - value for value in swing_values]

    def symmetry_percent(left_value, right_value):
        if left_value is None or right_value is None:
            return None
        denom = (abs(left_value) + abs(right_value)) / 2.0
        return abs(left_value - right_value) / denom * 100.0 if denom > 0 else None

    step_length_symmetry = symmetry_percent(left_step_mean, right_step_mean)
    left_step_time = [item for idx, item in enumerate(step_times) if idx % 2 == 0]
    right_step_time = [item for idx, item in enumerate(step_times) if idx % 2 == 1]
    left_step_time_mean = float(np.mean(left_step_time)) if left_step_time else None
    right_step_time_mean = float(np.mean(right_step_time)) if right_step_time else None
    step_time_symmetry = symmetry_percent(left_step_time_mean, right_step_time_mean)
    symmetry_values = [value for value in [step_length_symmetry, step_time_symmetry] if value is not None]

    stride_time_mean = float(np.mean(stride_times)) if stride_times else None
    stride_time_sd = float(np.std(stride_times, ddof=1)) if len(stride_times) > 1 else 0.0
    stride_variability = (stride_time_sd / stride_time_mean * 100.0) if stride_time_mean and stride_time_mean > 0 else None
    if stride_variability is None and len(step_lengths) > 1:
        step_mean = float(np.mean(step_lengths))
        stride_variability = float(np.std(step_lengths, ddof=1) / step_mean * 100.0) if step_mean > 0 else None

    left_arm = clean_values((lx15 - lx11) / body_scale)
    right_arm = clean_values((lx16 - lx12) / body_scale)
    left_arm_amp = float(np.nanmax(left_arm) - np.nanmin(left_arm)) if left_arm.size else None
    right_arm_amp = float(np.nanmax(right_arm) - np.nanmin(right_arm)) if right_arm.size else None
    arm_values = [value for value in [left_arm_amp, right_arm_amp] if value is not None]

    left_knee = angle_series(lx23, ly23, lx25, ly25, lx27, ly27)
    right_knee = angle_series(lx24, ly24, lx26, ly26, lx28, ly28)
    left_hip = angle_series(lx11, ly11, lx23, ly23, lx25, ly25)
    right_hip = angle_series(lx12, ly12, lx24, ly24, lx26, ly26)

    def angle_rom(values):
        values = clean_values(values)
        return float(np.nanmax(values) - np.nanmin(values)) if values.size else None

    left_knee_rom = angle_rom(left_knee)
    right_knee_rom = angle_rom(right_knee)
    left_hip_rom = angle_rom(left_hip)
    right_hip_rom = angle_rom(right_hip)

    left_clearance = clean_values(ly31 - ly27)
    right_clearance = clean_values(ly32 - ly28)
    left_clearance_range = (float(np.nanmax(left_clearance) - np.nanmin(left_clearance)) / body_scale) if left_clearance.size else None
    right_clearance_range = (float(np.nanmax(right_clearance) - np.nanmin(right_clearance)) / body_scale) if right_clearance.size else None

    metrics = [
        metric_row("walking_speed", "Walking speed", [walking_speed], "body lengths/s", range_status(walking_speed, 0.55, 1.60, 0.35, 2.10), "Forward pelvis progression normalized by body scale."),
        metric_row("cadence", "Cadence", [cadence], "steps/min", range_status(cadence, 90, 130, 70, 150), "Step rhythm estimated from alternating foot-forward events.", precision=1),
        metric_row("step_length", "Step length", step_lengths, "leg lengths", range_status(np.mean(step_lengths) if step_lengths else None, 0.45, 1.15, 0.30, 1.40), "Distance between left/right ankles at foot-forward events.", left=left_step_mean, right=right_step_mean),
        metric_row("stride_length", "Stride length", stride_lengths, "leg lengths", range_status(np.mean(stride_lengths) if stride_lengths else None, 0.90, 2.30, 0.60, 2.80), "Same-foot cycle displacement, using step length fallback when needed."),
        metric_row("step_time", "Step time", step_times, "s", range_status(np.mean(step_times) if step_times else None, 0.42, 0.75, 0.32, 0.95), "Time between alternating left/right foot-forward events."),
        metric_row("stride_time", "Stride time", stride_times, "s", range_status(np.mean(stride_times) if stride_times else None, 0.85, 1.50, 0.65, 1.90), "Time between repeated events from the same foot."),
        metric_row("swing_phase", "Swing phase proxy", swing_values, "%", range_status(np.mean(swing_values) if swing_values else None, 32, 48, 25, 55), "Percent of frames where the foot is moving relative to the pelvis.", left=left_swing, right=right_swing, precision=1),
        metric_row("stance_phase", "Stance phase proxy", stance_values, "%", range_status(np.mean(stance_values) if stance_values else None, 52, 68, 45, 75), "Complement of the foot-motion swing proxy.", precision=1),
        metric_row("symmetry_index", "Symmetry index", symmetry_values, "%", upper_status(np.mean(symmetry_values) if symmetry_values else None, 10, 20), "Lower is better. Combines step-length and step-time asymmetry.", left=step_length_symmetry, right=step_time_symmetry, precision=1),
        metric_row("stride_variability", "Stride variability", [stride_variability], "% CV", upper_status(stride_variability, 5, 10), "Coefficient of variation from stride timing or step-length fallback.", precision=1),
        metric_row("arm_swing_amplitude", "Arm swing amplitude", arm_values, "body scale", lower_status(np.mean(arm_values) if arm_values else None, 0.12, 0.07), "Wrist excursion relative to shoulder, normalized by body scale.", left=left_arm_amp, right=right_arm_amp),
        metric_row("knee_rom", "Knee range of motion", [value for value in [left_knee_rom, right_knee_rom] if value is not None], "deg", range_status(np.nanmean([value for value in [left_knee_rom, right_knee_rom] if value is not None]) if left_knee_rom is not None or right_knee_rom is not None else None, 25, 75, 15, 90), "Hip-knee-ankle angle range over the clean walking clip.", left=left_knee_rom, right=right_knee_rom, precision=1),
        metric_row("hip_rom", "Hip range of motion", [value for value in [left_hip_rom, right_hip_rom] if value is not None], "deg", range_status(np.nanmean([value for value in [left_hip_rom, right_hip_rom] if value is not None]) if left_hip_rom is not None or right_hip_rom is not None else None, 15, 55, 8, 70), "Shoulder-hip-knee angle range over the clean walking clip.", left=left_hip_rom, right=right_hip_rom, precision=1),
        metric_row("foot_clearance", "Foot clearance range", [value for value in [left_clearance_range, right_clearance_range] if value is not None], "body scale", range_status(np.nanmean([value for value in [left_clearance_range, right_clearance_range] if value is not None]) if left_clearance_range is not None or right_clearance_range is not None else None, 0.03, 0.22, 0.015, 0.32), "Vertical foot-index movement relative to ankle, normalized by body scale.", left=left_clearance_range, right=right_clearance_range),
    ]
    apply_current_measurement_context(metrics, final_label=final_label, mean_prob_abnormal=mean_prob_abnormal)

    notes = [
        "Biometric values are calculated from this current Component 1 pose-landmark run and normalized by estimated body scale.",
        "The model result is the real Component 1 classifier output from the normalized landmark feature windows; the table summarizes measured gait biomarkers from the same run.",
    ]
    if len(events) < 4:
        notes.append("Few alternating foot events were detected, so cadence and stride metrics may be less stable.")

    return {
        "available": True,
        "referenceFrame": "Body-scale normalized pose biometrics",
        "bodyScale": rounded(body_scale, 4),
        "eventCount": len(events),
        "leftEvents": len(left_peaks),
        "rightEvents": len(right_peaks),
        "durationSec": rounded(duration, 3),
        "fpsUsed": rounded(fps_used, 2),
        "screeningContext": {
            "finalLabel": final_label,
            "meanProbAbnormal": rounded(mean_prob_abnormal, 4),
            "greenCount": sum(1 for metric in metrics if metric.get("status") == "green"),
            "yellowCount": sum(1 for metric in metrics if metric.get("status") == "yellow"),
            "redCount": sum(1 for metric in metrics if metric.get("status") == "red"),
            "normalMarkerCount": sum(1 for metric in metrics if metric.get("status") == "green"),
            "borderlineMarkerCount": sum(1 for metric in metrics if metric.get("status") == "yellow"),
            "abnormalMarkerCount": sum(1 for metric in metrics if metric.get("status") == "red"),
            "rawGreenCount": sum(1 for metric in metrics if metric.get("rawStatus") == "green"),
            "rawYellowCount": sum(1 for metric in metrics if metric.get("rawStatus") == "yellow"),
            "rawRedCount": sum(1 for metric in metrics if metric.get("rawStatus") == "red"),
        },
        "metrics": metrics,
        "notes": notes,
    }


def attach_biometrics(result, csv_path, fps_used=None, direction=None, clean_duration_sec=None):
    if not csv_path:
        return result
    result["biometrics"] = calculate_component1_biometrics(
        csv_path=csv_path,
        fps_used=fps_used or result.get("fps_used"),
        direction=direction or result.get("direction"),
        clean_duration_sec=clean_duration_sec or result.get("clean_duration_sec"),
        final_label=result.get("final_label"),
        mean_prob_abnormal=result.get("mean_prob_abnormal"),
    )
    return result


def prepare_video_run(input_path: Path, extractor):
    run_dir = RUN_ROOT / f"run_{uuid.uuid4().hex}"
    input_dir = run_dir / "input"
    output_root = run_dir / "extracted_csv"
    raw_dir = output_root / "raw" / "inference"
    training_safe_dir = output_root / "training_safe" / "inference"
    summary_dir = output_root / "summary" / "inference"
    metadata_dir = run_dir / "metadata"

    for folder in [input_dir, raw_dir, training_safe_dir, summary_dir, metadata_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    short_video = input_dir / f"{safe_stem(input_path.stem)}{input_path.suffix.lower()}"
    shutil.copy2(input_path, short_video)

    extractor.INPUT_VIDEO_DIR = input_dir
    extractor.OUTPUT_ROOT = output_root
    extractor.OUTPUT_RAW = raw_dir
    extractor.OUTPUT_TRAINING_SAFE = training_safe_dir
    extractor.OUTPUT_SUMMARY = summary_dir
    extractor.METADATA_DIR = metadata_dir
    extractor.INFERENCE_METADATA_PATH = metadata_dir / "clinical_inference_metadata.csv"

    return short_video


def predict_csv(input_path: Path, direction: str | None, fps: float | None) -> dict:
    from src.component1_predictor import predict_component1

    with contextlib.redirect_stdout(sys.stderr):
        result = predict_component1(
            csv_path=input_path,
            direction=direction,
            fps_used=fps,
        )

    return attach_biometrics({
        **result,
        "input_type": "csv",
        "source_file": str(input_path),
        "model_suggested_label": result.get("final_label"),
        "model_suggested_result": result.get("final_result"),
    }, input_path, fps_used=result.get("fps_used") or fps, direction=result.get("direction") or direction, clean_duration_sec=result.get("clean_duration_sec"))


def predict_video(input_path: Path) -> dict:
    extractor = load_extractor_module()

    with contextlib.redirect_stdout(sys.stderr):
        video_path = prepare_video_run(input_path, extractor)
        extractor.ensure_dirs()
        extraction = extractor.extract_training_safe_csv(video_path)

        if extraction["clip_status"] == "reject":
            prediction = {
                "final_label": None,
                "final_result": "Result not generated - pose quality rejected",
                "model_suggested_label": None,
                "model_suggested_result": "Unavailable",
                "mean_prob_abnormal": None,
                "confidence_percent": None,
                "abnormal_ratio_default": None,
                "abnormal_ratio_threshold": None,
                "video_threshold": None,
                "screening_severity": "Rejected - repeat the recording",
                "total_windows": 0,
                "padded": False,
                "resampled_frames": 0,
                "reliability_level": "Rejected",
                "reliability_reasons": ["Pose quality was rejected."],
                "clinical_note": "Please repeat the recording with the full body visible and enough walking duration.",
            }
        else:
            prediction = extractor.predict_component1(
                csv_path=extraction["training_safe_csv"],
                direction=extraction["estimated_direction"],
                fps_used=extraction["fps_used"],
                clean_duration_sec=extraction["clean_duration_sec"],
                direction_confidence=extraction["direction_confidence"],
                clip_status=extraction["clip_status"],
            )

    csv_file = str(extraction.get("training_safe_csv", ""))
    final_result = {
        **prediction,
        "input_type": "video",
        "source_file": str(input_path),
        "csv_file": csv_file,
        "raw_csv": str(extraction.get("raw_csv", "")),
        "summary_csv": str(extraction.get("summary_csv", "")),
        "direction": extraction.get("estimated_direction"),
        "direction_confidence": extraction.get("direction_confidence"),
        "fps_used": extraction.get("fps_used"),
        "clean_frames": extraction.get("clean_frames"),
        "clean_duration_sec": round(float(extraction.get("clean_duration_sec") or 0), 3),
        "clip_status": extraction.get("clip_status"),
        "clip_review_required": extraction.get("clip_review_required"),
    }
    if extraction.get("clip_status") != "reject":
        attach_biometrics(
            final_result,
            csv_file,
            fps_used=extraction.get("fps_used"),
            direction=extraction.get("estimated_direction"),
            clean_duration_sec=extraction.get("clean_duration_sec"),
        )
    return final_result


def main():
    parser = argparse.ArgumentParser(description="Component 1 normal vs abnormal inference wrapper.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-type", required=True, choices=["csv", "video"])
    parser.add_argument("--direction", choices=["L2R", "R2L"], default=None)
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.input_type == "csv":
        result = predict_csv(input_path, args.direction, args.fps)
    else:
        result = predict_video(input_path)

    print(json.dumps({"ok": True, "result": result}, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        )
        sys.exit(1)
