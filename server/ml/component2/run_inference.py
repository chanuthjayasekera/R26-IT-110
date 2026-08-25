import argparse
import contextlib
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SERVER_ROOT = ROOT.parents[1]
COMPONENT1_RUNNER = SERVER_ROOT / "ml" / "component1" / "run_inference.py"
MODEL_ROOTS = {
    "sca": ROOT / "sca",
    "koa": ROOT / "koa",
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


def load_component1_biometrics_module():
    spec = importlib.util.spec_from_file_location("component1_clinical_biometrics", COMPONENT1_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Component 1 clinical biometrics: {COMPONENT1_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BODY_PARTS = [
    {"key": "head", "label": "Head control", "landmarks": [0, 7, 8], "x": 50, "y": 8, "group": "upper"},
    {"key": "shoulders", "label": "Shoulder balance", "landmarks": [11, 12], "x": 50, "y": 23, "group": "upper"},
    {"key": "left_arm", "label": "Left arm swing", "landmarks": [11, 13, 15], "x": 31, "y": 33, "group": "arm"},
    {"key": "right_arm", "label": "Right arm swing", "landmarks": [12, 14, 16], "x": 69, "y": 33, "group": "arm"},
    {"key": "left_forearm_hand", "label": "Left forearm/hand control", "landmarks": [13, 15, 17, 19, 21], "x": 25, "y": 48, "group": "arm"},
    {"key": "right_forearm_hand", "label": "Right forearm/hand control", "landmarks": [14, 16, 18, 20, 22], "x": 75, "y": 48, "group": "arm"},
    {"key": "trunk", "label": "Trunk stability", "landmarks": [11, 12, 23, 24], "x": 50, "y": 40, "group": "trunk"},
    {"key": "pelvis", "label": "Pelvic control", "landmarks": [23, 24], "x": 50, "y": 53, "group": "trunk"},
    {"key": "left_knee", "label": "Left knee control", "landmarks": [23, 25, 27], "x": 42, "y": 69, "group": "lower"},
    {"key": "right_knee", "label": "Right knee control", "landmarks": [24, 26, 28], "x": 58, "y": 69, "group": "lower"},
    {"key": "left_ankle", "label": "Left ankle/foot stability", "landmarks": [27, 29, 31], "x": 39, "y": 88, "group": "foot"},
    {"key": "right_ankle", "label": "Right ankle/foot stability", "landmarks": [28, 30, 32], "x": 61, "y": 88, "group": "foot"},
]

PART_PAIRS = {
    "left_arm": "right_arm",
    "right_arm": "left_arm",
    "left_forearm_hand": "right_forearm_hand",
    "right_forearm_hand": "left_forearm_hand",
    "left_knee": "right_knee",
    "right_knee": "left_knee",
    "left_ankle": "right_ankle",
    "right_ankle": "left_ankle",
}

CLINICAL_LIMITS = {
    "upper": {
        "sway": (0.025, 0.110),
        "jerk": (0.003, 0.018),
        "jitter": (0.004, 0.030),
    },
    "trunk": {
        "sway": (0.030, 0.135),
        "jerk": (0.0035, 0.020),
        "jitter": (0.004, 0.032),
    },
    "arm": {
        "low_swing": (0.20, 0.075),
        "swing_asymmetry": (0.22, 0.62),
        "rhythm_cv": (1.20, 2.80),
        "jerk": (0.006, 0.030),
    },
    "lower": {
        "low_motion": (0.18, 0.070),
        "motion_asymmetry": (0.20, 0.58),
        "rhythm_cv": (1.05, 2.50),
        "jerk": (0.008, 0.035),
    },
    "foot": {
        "low_clearance": (0.075, 0.025),
        "clearance_asymmetry": (0.22, 0.62),
        "rhythm_cv": (1.05, 2.60),
        "jerk": (0.010, 0.045),
    },
    "global": {
        "path_sway": (0.012, 0.022),
        "path_rhythm_cv": (0.25, 0.75),
    },
}


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").interpolate(limit_direction="both").bfill().ffill().fillna(0.0)


def _landmark_xy(df: pd.DataFrame, landmark_id: int) -> np.ndarray:
    return np.column_stack([
        _safe_numeric(df, f"lm{landmark_id}_x").to_numpy(dtype=float),
        _safe_numeric(df, f"lm{landmark_id}_y").to_numpy(dtype=float),
    ])


def _visibility(df: pd.DataFrame, landmark_id: int) -> np.ndarray:
    return _safe_numeric(df, f"lm{landmark_id}_vis").to_numpy(dtype=float)


def _smooth_xy(xy: np.ndarray, window: int = 7) -> np.ndarray:
    frame = pd.DataFrame(xy, columns=["x", "y"])
    return frame.rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=float)


def _landmark_metrics(df: pd.DataFrame, landmark_id: int, scale: float, body_center: np.ndarray | None = None) -> dict:
    xy = _landmark_xy(df, landmark_id)
    if body_center is not None and body_center.shape == xy.shape:
        xy = xy - body_center
    smooth_xy = _smooth_xy(xy)
    residual = xy - smooth_xy
    velocity = np.diff(smooth_xy, axis=0)
    acceleration = np.diff(velocity, axis=0)
    speed = np.linalg.norm(velocity, axis=1) / scale if len(velocity) else np.array([0.0])
    accel = np.linalg.norm(acceleration, axis=1) / scale if len(acceleration) else np.array([0.0])
    mean_speed = float(np.mean(speed)) if len(speed) else 0.0
    x_range = float(np.ptp(smooth_xy[:, 0]) / scale)
    y_range = float(np.ptp(smooth_xy[:, 1]) / scale)

    return {
        "motion_range": float(math.sqrt(x_range**2 + y_range**2)),
        "x_range": x_range,
        "y_range": y_range,
        "sway": float(np.std(smooth_xy[:, 0]) / scale),
        "mean_speed": mean_speed,
        "speed_cv": float(np.std(speed) / (mean_speed + 1e-6)) if len(speed) else 0.0,
        "jerk_ratio": float(np.mean(accel)) if len(accel) else 0.0,
        "jitter": float(np.mean(np.linalg.norm(residual, axis=1)) / scale),
        "visibility": float(np.nanmean(_visibility(df, landmark_id))),
    }


def _status(score: float) -> str:
    if score >= 0.66:
        return "severe"
    if score >= 0.36:
        return "moderate"
    return "stable"


def _status_label(status: str) -> str:
    if status == "severe":
        return "High instability"
    if status == "moderate":
        return "Moderate control region"
    return "Stable"


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _metric_instability(value: float, stable_value: float, severe_value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if severe_value <= stable_value:
        return 0.0
    return _clamp01((float(value) - stable_value) / (severe_value - stable_value))


def _low_motion_instability(value: float, stable_value: float, severe_value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if stable_value <= severe_value:
        return 0.0
    return _clamp01((stable_value - float(value)) / (stable_value - severe_value))


def _status_from_score(score: float) -> tuple[str, str]:
    if score >= 0.68:
        return "severe", "High instability"
    if score >= 0.34:
        return "moderate", "Moderate control region"
    return "stable", "Stable control"


def _clinical_meaning(status: str, drivers: list[str]) -> str:
    if status == "severe":
        return f"High clinical instability features: {', '.join(drivers[:3])}."
    if status == "moderate":
        return f"Moderate gait-control concern: {', '.join(drivers[:3])}."
    return "Stable clinical gait-control features for this body region."


def _clinical_status_score(status: str) -> float:
    if status == "red":
        return 0.9
    if status == "yellow":
        return 0.5
    if status == "green":
        return 0.08
    return 0.25


def _clinical_metric_severity(metric: dict) -> float:
    status = metric.get("status")
    if status == "green":
        return 0.08
    if status == "yellow":
        return 0.44
    if status != "red":
        return 0.25

    value = _finite_number(metric.get("value"))
    bands = SIDE_VIEW_BANDS.get(metric.get("key"))
    if value is None or not bands:
        return 0.72

    green = bands["green"]
    yellow = bands["yellow"]
    mode = bands["mode"]
    if mode == "range":
        if value < yellow[0]:
            breach = yellow[0] - value
            clinical_buffer = max(green[0] - yellow[0], abs(yellow[0]) * 0.08, 1e-6)
        else:
            breach = value - yellow[1]
            clinical_buffer = max(yellow[1] - green[1], abs(yellow[1]) * 0.08, 1e-6)
    elif mode == "upper":
        breach = value - yellow[1]
        clinical_buffer = max(yellow[1] - green[1], abs(yellow[1]) * 0.15, 1e-6)
    elif mode == "lower":
        breach = yellow[0] - value
        clinical_buffer = max(green[0] - yellow[0], abs(yellow[0]) * 0.15, 1e-6)
    else:
        return 0.72

    return 0.62 + 0.33 * _clamp01(breach / clinical_buffer)


def _map_metric_status(status: str) -> tuple[str, str]:
    if status == "red":
        return "severe", "High instability"
    if status == "yellow":
        return "moderate", "Moderate control region"
    return "stable", "Stable control"


def _metric_by_key(biometrics: dict) -> dict:
    return {metric.get("key"): metric for metric in biometrics.get("metrics", []) if metric.get("key")}


def _clinical_metric_entry(metric: dict | None, side: str | None = None, side_status: str | None = None) -> dict | None:
    if not metric:
        return None
    value = metric.get(side) if side in {"left", "right"} else metric.get("value")
    side_statuses = metric.get("sideStatuses") or {}
    side_confidence = metric.get("sideConfidence") or {}
    return {
        "key": metric.get("key"),
        "label": metric.get("label"),
        "value": value,
        "mean": metric.get("value"),
        "left": metric.get("left"),
        "right": metric.get("right"),
        "unit": metric.get("unit"),
        "status": side_status or (side_statuses.get(side) if side else None) or metric.get("status"),
        "referenceRange": metric.get("referenceRange"),
        "measurementMethod": metric.get("measurementMethod"),
        "confidence": (side_confidence.get(side) if side else None) or metric.get("confidence"),
        "inferred": bool(side and side_confidence.get(side) and side_confidence.get(side) != "direct"),
    }


def _side_status(component1, metric_key: str, value) -> str:
    bands = component1.COMPONENT1_FALLBACK_BANDS.get(metric_key)
    if not bands:
        return "unavailable"
    green = bands["green"]
    yellow = bands["yellow"]
    mode = bands["mode"]
    if mode == "range":
        return component1.range_status(value, green[0], green[1], yellow[0], yellow[1])
    if mode == "upper":
        return component1.upper_status(value, green[1], yellow[1])
    if mode == "lower":
        return component1.lower_status(value, green[0], yellow[0])
    return "unavailable"


def _region_from_clinical_metrics(region_metrics: list[dict]) -> tuple[float, str, str, list[str]]:
    usable = [metric for metric in region_metrics if metric and metric.get("status") in {"green", "yellow", "red"}]
    if not usable:
        return 0.25, "stable", "Stable control", ["clinical metric unavailable"]

    scored = sorted(
        ((metric, _clinical_metric_severity(metric)) for metric in usable),
        key=lambda item: item[1],
        reverse=True,
    )
    scores = [score for _, score in scored]
    top_score = scores[0]
    top_three_mean = float(np.mean(scores[:3]))
    all_mean = float(np.mean(scores))
    score = _clamp01((top_score * 0.55) + (top_three_mean * 0.30) + (all_mean * 0.15))

    red_count = sum(1 for metric in usable if metric.get("status") == "red")
    yellow_count = sum(1 for metric in usable if metric.get("status") == "yellow")
    if top_score >= 0.88 or (red_count >= 2 and score >= 0.62) or (red_count >= 1 and yellow_count >= 2 and score >= 0.64):
        status, label = "severe", "High instability"
        score = max(score, 0.68)
    else:
        status, label = _status_from_score(score)
        if status == "stable" and yellow_count >= 1:
            status, label = "moderate", "Moderate control region"
            score = max(score, 0.34)

    drivers = [
        f"{metric.get('label')} {metric.get('value')} {metric.get('unit') or ''}".strip()
        for metric, metric_score in scored
        if metric.get("status") in {"red", "yellow"} or metric_score >= 0.34
    ]
    return score, status, label, drivers or ["clinical metric comparison"]


SIDE_VIEW_REFERENCE_RANGES = {
    "walking_speed": "0.55-1.60 body lengths/s",
    "cadence": "90-130 steps/min",
    "step_length": "0.45-1.15 leg lengths",
    "stride_length": "0.90-2.30 leg lengths",
    "step_time": "0.42-0.75 s",
    "stride_time": "0.85-1.50 s",
    "stride_variability": "<=5 % CV",
    "arm_swing_amplitude": ">=0.12 body scale",
    "arm_swing_asymmetry": "<=25 %",
    "elbow_rom": ">=15 deg during gait",
    "elbow_flexion_posture": "<=35 deg median flexion",
    "forearm_hand_excursion": ">=0.08 body scale",
    "head_sway": "<=0.06 body scale",
    "shoulder_sway": "<=0.06 body scale",
    "hip_rom": "15-55 deg",
    "knee_rom": "25-75 deg",
    "foot_clearance": "0.03-0.22 body scale",
    "ankle_excursion": "0.18-0.75 body scale",
    "foot_path_variability": "<=35 % CV",
    "swing_phase": "32-48 %",
    "trunk_control": "<=10 deg trunk excursion",
    "pelvic_path_deviation": "<=0.04 body scale",
}

SIDE_VIEW_BANDS = {
    "walking_speed": {"green": (0.55, 1.60), "yellow": (0.35, 2.10), "mode": "range"},
    "cadence": {"green": (90, 130), "yellow": (70, 150), "mode": "range"},
    "step_length": {"green": (0.45, 1.15), "yellow": (0.30, 1.40), "mode": "range"},
    "stride_length": {"green": (0.90, 2.30), "yellow": (0.60, 2.80), "mode": "range"},
    "step_time": {"green": (0.42, 0.75), "yellow": (0.32, 0.95), "mode": "range"},
    "stride_time": {"green": (0.85, 1.50), "yellow": (0.65, 1.90), "mode": "range"},
    "stride_variability": {"green": (None, 5), "yellow": (None, 10), "mode": "upper"},
    "arm_swing_amplitude": {"green": (0.12, None), "yellow": (0.10, None), "mode": "lower"},
    "arm_swing_asymmetry": {"green": (None, 25), "yellow": (None, 45), "mode": "upper"},
    "elbow_rom": {"green": (15, None), "yellow": (8, None), "mode": "lower"},
    "elbow_flexion_posture": {"green": (None, 35), "yellow": (None, 55), "mode": "upper"},
    "forearm_hand_excursion": {"green": (0.08, None), "yellow": (0.05, None), "mode": "lower"},
    "head_sway": {"green": (None, 0.06), "yellow": (None, 0.12), "mode": "upper"},
    "shoulder_sway": {"green": (None, 0.06), "yellow": (None, 0.12), "mode": "upper"},
    "hip_rom": {"green": (15, 55), "yellow": (8, 70), "mode": "range"},
    "knee_rom": {"green": (25, 75), "yellow": (15, 90), "mode": "range"},
    "foot_clearance": {"green": (0.03, 0.22), "yellow": (0.015, 0.32), "mode": "range"},
    "ankle_excursion": {"green": (0.18, 0.75), "yellow": (0.10, 0.95), "mode": "range"},
    "foot_path_variability": {"green": (None, 35), "yellow": (None, 60), "mode": "upper"},
    "swing_phase": {"green": (32, 48), "yellow": (25, 55), "mode": "range"},
    "trunk_control": {"green": (None, 10), "yellow": (None, 18), "mode": "upper"},
    "pelvic_path_deviation": {"green": (None, 0.04), "yellow": (None, 0.08), "mode": "upper"},
}

SIDE_VIEW_DEFINITIONS = {
    "walking_speed": "Forward pelvis progression normalized by estimated visible-side leg length.",
    "cadence": "Steps per minute estimated from visible-side same-foot gait cycles.",
    "step_length": "Half-stride distance normalized by visible-side leg length.",
    "stride_length": "Pelvis progression between same-foot cycle peaks, normalized by visible-side leg length.",
    "step_time": "Half of the visible-side stride cycle time.",
    "stride_time": "Time between repeated visible-foot forward peaks.",
    "stride_variability": "Coefficient of variation of repeated visible-foot stride times.",
    "arm_swing_amplitude": "Wrist sagittal excursion relative to shoulder, normalized by visible-side body scale.",
    "arm_swing_asymmetry": "Left-right difference in sagittal arm swing amplitude.",
    "elbow_rom": "Sagittal shoulder-elbow-wrist flexion-extension angle range during walking.",
    "elbow_flexion_posture": "Median sagittal elbow flexion across the walk; sustained flexion separates a held bent forearm from normal cyclic bending.",
    "forearm_hand_excursion": "Sagittal hand-centre excursion relative to the elbow, normalized by visible-side body scale.",
    "head_sway": "Head movement relative to pelvis after removing forward walking progression.",
    "shoulder_sway": "Shoulder-girdle movement relative to pelvis after removing forward walking progression.",
    "hip_rom": "Sagittal shoulder-hip-knee angle range on the visible side.",
    "knee_rom": "Sagittal hip-knee-ankle angle range on the visible side.",
    "foot_clearance": "Visible foot vertical excursion relative to ankle, normalized by body scale.",
    "ankle_excursion": "Sagittal ankle travel relative to the pelvis, normalized by body scale.",
    "foot_path_variability": "Frame-to-frame variability of foot travel; higher values suggest irregular foot placement.",
    "swing_phase": "Percent of frames where the visible foot is moving relative to the pelvis.",
    "trunk_control": "Sagittal trunk angle excursion from shoulder-to-hip line.",
    "pelvic_path_deviation": "Pelvis deviation from its walking path after removing forward progression.",
}


def _finite_number(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _rounded(value, digits=2):
    numeric = _finite_number(value)
    return None if numeric is None else round(numeric, digits)


def _clean_array(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _range_status(value, green_low, green_high, yellow_low, yellow_high):
    numeric = _finite_number(value)
    if numeric is None:
        return "unavailable"
    if green_low <= numeric <= green_high:
        return "green"
    if yellow_low <= numeric <= yellow_high:
        return "yellow"
    return "red"


def _upper_status(value, green_max, yellow_max):
    numeric = _finite_number(value)
    if numeric is None:
        return "unavailable"
    if numeric <= green_max:
        return "green"
    if numeric <= yellow_max:
        return "yellow"
    return "red"


def _lower_status(value, green_min, yellow_min):
    numeric = _finite_number(value)
    if numeric is None:
        return "unavailable"
    if numeric >= green_min:
        return "green"
    if numeric >= yellow_min:
        return "yellow"
    return "red"


def _status_for_metric(key, value):
    bands = SIDE_VIEW_BANDS.get(key)
    if not bands:
        return "unavailable"
    green = bands["green"]
    yellow = bands["yellow"]
    if bands["mode"] == "range":
        return _range_status(value, green[0], green[1], yellow[0], yellow[1])
    if bands["mode"] == "upper":
        return _upper_status(value, green[1], yellow[1])
    if bands["mode"] == "lower":
        return _lower_status(value, green[0], yellow[0])
    return "unavailable"


def _side_view_metric_row(key, label, value, unit, left=None, right=None, side_statuses=None, precision=2):
    return {
        "key": key,
        "label": label,
        "value": _rounded(value, precision),
        "left": _rounded(left, precision),
        "right": _rounded(right, precision),
        "unit": unit,
        "status": _status_for_metric(key, value),
        "sideStatuses": side_statuses or {},
        "referenceRange": SIDE_VIEW_REFERENCE_RANGES.get(key, "Clinical side-view reference band"),
        "referenceType": "Side-view normalized clinical gait band",
        "measurementMethod": SIDE_VIEW_DEFINITIONS.get(key, "Side-view pose-landmark clinical measure."),
        "source": "Component 2 side-view normalized CSV",
    }


def _angle_series(a, b, c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    cosang = np.sum(ba * bc, axis=1) / np.maximum(denom, 1e-9)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


def _angle_range(values):
    arr = _clean_array(values)
    if arr.size == 0:
        return None
    return float(np.nanpercentile(arr, 95) - np.nanpercentile(arr, 5))


def _find_cycle_peaks(signal, fps):
    arr = pd.Series(signal, dtype=float).interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype=float)
    if arr.size < 5:
        return []
    smoothed = pd.Series(arr).rolling(window=5, center=True, min_periods=1).median().to_numpy(dtype=float)
    amplitude = float(np.nanpercentile(smoothed, 90) - np.nanpercentile(smoothed, 10))
    if not math.isfinite(amplitude) or amplitude <= 1e-6:
        return []
    min_gap = max(int((fps or 30.0) * 0.55), 4)
    candidates = [
        idx
        for idx in range(1, len(smoothed) - 1)
        if smoothed[idx] >= smoothed[idx - 1]
        and smoothed[idx] >= smoothed[idx + 1]
        and smoothed[idx] >= np.nanmedian(smoothed) + amplitude * 0.10
    ]
    selected = []
    for idx in sorted(candidates, key=lambda item: smoothed[item], reverse=True):
        if all(abs(idx - old) >= min_gap for old in selected):
            selected.append(idx)
    return sorted(selected)


def _visibility_mean(df, ids):
    vals = [_visibility(df, landmark_id) for landmark_id in ids]
    return float(np.nanmean(np.column_stack(vals))) if vals else 0.0


def calculate_side_view_biometrics(df: pd.DataFrame, result: dict) -> dict:
    timestamps = pd.to_numeric(df.get("timestamp_ms"), errors="coerce").to_numpy(dtype=float) / 1000.0
    if not np.isfinite(timestamps).any():
        fps_guess = result.get("fps_used") or result.get("fps") or 30.0
        timestamps = np.arange(len(df), dtype=float) / float(fps_guess)
    timestamps = timestamps - np.nanmin(timestamps)
    diffs = np.diff(timestamps[np.isfinite(timestamps)])
    fps_used = _finite_number(result.get("fps_used") or result.get("fps"))
    if fps_used is None:
        fps_used = 1 / np.nanmedian(diffs) if diffs.size and np.nanmedian(diffs) > 0 else 30.0
    duration = float(np.nanmax(timestamps) - np.nanmin(timestamps)) if len(timestamps) > 1 else 0.0

    side_ids = {
        "left": {
            "shoulder": 11, "elbow": 13, "wrist": 15, "hand": [15, 17, 19, 21],
            "hip": 23, "knee": 25, "ankle": 27, "foot": 31,
        },
        "right": {
            "shoulder": 12, "elbow": 14, "wrist": 16, "hand": [16, 18, 20, 22],
            "hip": 24, "knee": 26, "ankle": 28, "foot": 32,
        },
    }
    side_data = {}
    for side, ids in side_ids.items():
        hip = _landmark_xy(df, ids["hip"])
        knee = _landmark_xy(df, ids["knee"])
        ankle = _landmark_xy(df, ids["ankle"])
        foot = _landmark_xy(df, ids["foot"])
        shoulder = _landmark_xy(df, ids["shoulder"])
        elbow = _landmark_xy(df, ids["elbow"])
        wrist = _landmark_xy(df, ids["wrist"])
        hand = np.mean(np.stack([_landmark_xy(df, hand_id) for hand_id in ids["hand"]]), axis=0)
        leg_len = np.linalg.norm(hip - knee, axis=1) + np.linalg.norm(knee - ankle, axis=1)
        scale = float(np.nanmedian(leg_len))
        rel_foot = foot - hip
        motion = float(np.nanpercentile(np.linalg.norm(rel_foot - np.nanmedian(rel_foot, axis=0), axis=1), 90))
        visibility = _visibility_mean(df, [ids["hip"], ids["knee"], ids["ankle"], ids["foot"]])
        quality = (visibility * 0.75) + (min(motion / max(scale, 1e-6), 1.0) * 0.25)
        side_data[side] = {
            "ids": ids,
            "hip": hip,
            "knee": knee,
            "ankle": ankle,
            "foot": foot,
            "shoulder": shoulder,
            "elbow": elbow,
            "wrist": wrist,
            "hand": hand,
            "scale": scale,
            "visibility": visibility,
            "quality": quality,
        }

    visible_side = max(side_data, key=lambda side: side_data[side]["quality"])
    hidden_side = "right" if visible_side == "left" else "left"
    data = side_data[visible_side]
    body_scale = data["scale"] if math.isfinite(data["scale"]) and data["scale"] > 1e-6 else 1.0
    hip = data["hip"]
    knee = data["knee"]
    ankle = data["ankle"]
    foot = data["foot"]
    shoulder = data["shoulder"]
    wrist = data["wrist"]

    direction = result.get("direction")
    if direction in {"L2R", "R2L"}:
        sign = 1 if direction == "L2R" else -1
    else:
        sample = max(int(len(hip) * 0.1), 3)
        sign = 1 if np.nanmedian(hip[-sample:, 0]) >= np.nanmedian(hip[:sample, 0]) else -1

    sample = max(int(len(hip) * 0.1), 3)
    walking_speed = abs(float(np.nanmedian(hip[-sample:, 0]) - np.nanmedian(hip[:sample, 0]))) / body_scale / duration if duration > 0 else None
    rel_forward = sign * (foot[:, 0] - hip[:, 0])
    stride_peaks = _find_cycle_peaks(rel_forward, fps_used)
    stride_times = [
        timestamps[second] - timestamps[first]
        for first, second in zip(stride_peaks, stride_peaks[1:])
        if timestamps[second] > timestamps[first]
    ]
    stride_lengths = [
        abs(hip[second, 0] - hip[first, 0]) / body_scale
        for first, second in zip(stride_peaks, stride_peaks[1:])
        if timestamps[second] > timestamps[first]
    ]
    stride_time = float(np.mean(stride_times)) if stride_times else None
    stride_time_sd = float(np.std(stride_times, ddof=1)) if len(stride_times) > 1 else 0.0
    stride_variability = (stride_time_sd / stride_time * 100.0) if stride_time and stride_time > 0 and len(stride_times) >= 6 else None
    cadence = 120.0 / stride_time if stride_time and stride_time > 0 else None
    step_time = stride_time / 2.0 if stride_time else None
    stride_length = float(np.mean(stride_lengths)) if stride_lengths else None
    step_length = stride_length / 2.0 if stride_length is not None else None

    foot_rel = foot - ankle
    foot_clearance = float(np.nanpercentile(foot_rel[:, 1], 95) - np.nanpercentile(foot_rel[:, 1], 5)) / body_scale
    foot_motion = np.linalg.norm(np.diff(foot - hip, axis=0), axis=1)
    if foot_motion.size >= 4:
        threshold = float(np.nanmedian(foot_motion) + np.nanstd(foot_motion) * 0.25)
        swing_phase = float(np.mean(foot_motion > threshold) * 100.0)
    else:
        swing_phase = None

    knee_rom = _angle_range(_angle_series(hip, knee, ankle))
    hip_rom = _angle_range(_angle_series(shoulder, hip, knee))
    arm_swing = float(np.nanpercentile(wrist[:, 0] - shoulder[:, 0], 95) - np.nanpercentile(wrist[:, 0] - shoulder[:, 0], 5)) / body_scale
    trunk_vector = shoulder - hip
    trunk_angle = np.degrees(np.arctan2(trunk_vector[:, 0], np.maximum(np.abs(trunk_vector[:, 1]), 1e-9)))
    trunk_control = _angle_range(trunk_angle)

    frame_index = np.arange(len(hip), dtype=float)
    path_fit = np.column_stack([
        np.polyval(np.polyfit(frame_index, hip[:, 0], 1), frame_index),
        np.polyval(np.polyfit(frame_index, hip[:, 1], 1), frame_index),
    ])
    pelvic_path_deviation = float(np.mean(np.linalg.norm(hip - path_fit, axis=1)) / body_scale)

    left_shoulder = side_data["left"]["shoulder"]
    right_shoulder = side_data["right"]["shoulder"]
    left_hip = side_data["left"]["hip"]
    right_hip = side_data["right"]["hip"]
    head = _landmark_xy(df, 0)
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    hip_mid = (left_hip + right_hip) / 2.0

    def relative_path_deviation(point_xy):
        rel = point_xy - hip_mid
        idx = np.arange(len(rel), dtype=float)
        fit = np.column_stack([
            np.polyval(np.polyfit(idx, rel[:, 0], 1), idx),
            np.polyval(np.polyfit(idx, rel[:, 1], 1), idx),
        ])
        return float(np.mean(np.linalg.norm(rel - fit, axis=1)) / body_scale)

    head_sway = relative_path_deviation(head)
    shoulder_sway = relative_path_deviation(shoulder_mid)

    side_values = {}
    for side, item in side_data.items():
        side_scale = item["scale"] if math.isfinite(item["scale"]) and item["scale"] > 1e-6 else body_scale
        side_foot_rel = item["foot"] - item["ankle"]
        side_foot_motion = np.linalg.norm(np.diff(item["foot"] - item["hip"], axis=0), axis=1)
        if side_foot_motion.size >= 4:
            side_threshold = float(np.nanmedian(side_foot_motion) + np.nanstd(side_foot_motion) * 0.25)
            side_swing = float(np.mean(side_foot_motion > side_threshold) * 100.0)
        else:
            side_swing = None
        side_ankle_rel = item["ankle"] - item["hip"]
        side_ankle_x = side_ankle_rel[:, 0]
        side_ankle_y = side_ankle_rel[:, 1]
        side_hand_rel = item["hand"] - item["elbow"]
        side_elbow_flexion = 180.0 - _angle_series(item["shoulder"], item["elbow"], item["wrist"])
        side_speed = side_foot_motion / side_scale if side_foot_motion.size else np.array([], dtype=float)
        side_mean_speed = float(np.mean(side_speed)) if side_speed.size else None
        side_path_variability = (
            float(np.std(side_speed, ddof=1) / side_mean_speed * 100.0)
            if side_speed.size > 3 and side_mean_speed and side_mean_speed > 1e-6
            else None
        )
        side_values[side] = {
            "arm_swing_amplitude": float(np.nanpercentile(item["wrist"][:, 0] - item["shoulder"][:, 0], 95) - np.nanpercentile(item["wrist"][:, 0] - item["shoulder"][:, 0], 5)) / side_scale,
            "elbow_rom": _angle_range(_angle_series(item["shoulder"], item["elbow"], item["wrist"])),
            "elbow_flexion_posture": float(np.nanmedian(side_elbow_flexion)),
            "forearm_hand_excursion": float(
                np.nanpercentile(side_hand_rel[:, 0], 95) - np.nanpercentile(side_hand_rel[:, 0], 5)
            ) / side_scale,
            "hip_rom": _angle_range(_angle_series(item["shoulder"], item["hip"], item["knee"])),
            "knee_rom": _angle_range(_angle_series(item["hip"], item["knee"], item["ankle"])),
            "foot_clearance": float(np.nanpercentile(side_foot_rel[:, 1], 95) - np.nanpercentile(side_foot_rel[:, 1], 5)) / side_scale,
            "ankle_excursion": float(math.sqrt(
                (np.nanpercentile(side_ankle_x, 95) - np.nanpercentile(side_ankle_x, 5)) ** 2
                + (np.nanpercentile(side_ankle_y, 95) - np.nanpercentile(side_ankle_y, 5)) ** 2
            )) / side_scale,
            "foot_path_variability": side_path_variability,
            "swing_phase": side_swing,
        }

    def side_statuses(key):
        return {side: _status_for_metric(key, side_values[side].get(key)) for side in ("left", "right")}

    def side_confidence():
        return {
            side: "direct" if side == visible_side else "estimated from side-view CSV landmarks"
            for side in ("left", "right")
        }

    def bilateral_metric(key, label, unit, precision=2):
        left_value = side_values["left"].get(key)
        right_value = side_values["right"].get(key)
        values = [value for value in [left_value, right_value] if _finite_number(value) is not None]
        mean_value = float(np.mean(values)) if values else None
        row = _side_view_metric_row(
            key,
            label,
            mean_value,
            unit,
            left=left_value,
            right=right_value,
            side_statuses=side_statuses(key),
            precision=precision,
        )
        row["sideConfidence"] = side_confidence()
        row["confidence"] = "direct + contralateral estimated"
        return row

    arm_values = [side_values["left"].get("arm_swing_amplitude"), side_values["right"].get("arm_swing_amplitude")]
    clean_arm_values = [value for value in arm_values if _finite_number(value) is not None]
    arm_swing_asymmetry = (
        abs(clean_arm_values[0] - clean_arm_values[1]) / max(float(np.mean(clean_arm_values)), 1e-6) * 100.0
        if len(clean_arm_values) == 2
        else None
    )

    metrics = [
        _side_view_metric_row("walking_speed", "Walking speed", walking_speed, "body lengths/s", precision=2),
        _side_view_metric_row("cadence", "Cadence", cadence, "steps/min", precision=1),
        _side_view_metric_row("step_length", "Step length", step_length, "leg lengths", precision=2),
        _side_view_metric_row("stride_length", "Stride length", stride_length, "leg lengths", precision=2),
        _side_view_metric_row("step_time", "Step time", step_time, "s", precision=2),
        _side_view_metric_row("stride_time", "Stride time", stride_time, "s", precision=2),
        _side_view_metric_row("stride_variability", "Stride variability", stride_variability, "% CV", precision=1),
        bilateral_metric("arm_swing_amplitude", "Arm swing amplitude", "body scale", precision=2),
        _side_view_metric_row("arm_swing_asymmetry", "Arm swing asymmetry", arm_swing_asymmetry, "%", precision=1),
        bilateral_metric("elbow_rom", "Elbow range of motion", "deg", precision=1),
        bilateral_metric("elbow_flexion_posture", "Sustained elbow flexion", "deg", precision=1),
        bilateral_metric("forearm_hand_excursion", "Forearm/hand excursion", "body scale", precision=2),
        _side_view_metric_row("head_sway", "Head sway", head_sway, "body scale", precision=3),
        _side_view_metric_row("shoulder_sway", "Shoulder sway", shoulder_sway, "body scale", precision=3),
        bilateral_metric("hip_rom", "Hip range of motion", "deg", precision=1),
        bilateral_metric("knee_rom", "Knee range of motion", "deg", precision=1),
        bilateral_metric("foot_clearance", "Foot clearance range", "body scale", precision=2),
        bilateral_metric("ankle_excursion", "Ankle sagittal excursion", "body scale", precision=2),
        bilateral_metric("foot_path_variability", "Foot path variability", "% CV", precision=1),
        bilateral_metric("swing_phase", "Swing phase proxy", "%", precision=1),
        _side_view_metric_row("trunk_control", "Trunk control", trunk_control, "deg", precision=1),
        _side_view_metric_row("pelvic_path_deviation", "Pelvic path deviation", pelvic_path_deviation, "body scale", precision=3),
    ]

    usable = [metric for metric in metrics if metric.get("status") in {"green", "yellow", "red"}]
    notes = [
        "Side-view gait metrics are calculated from normalized sagittal-plane landmarks.",
        f"The visible lower limb was estimated as {visible_side}; the opposite side is still measured from CSV landmarks and labelled as side-view estimated.",
    ]
    if len(stride_peaks) < 3:
        notes.append("Few repeated visible-foot stride peaks were detected, so cadence and stride variability may be less reliable.")

    return {
        "available": True,
        "referenceFrame": "Side-view body-scale normalized clinical gait measures",
        "bodyScale": _rounded(body_scale, 4),
        "durationSec": _rounded(duration, 3),
        "fpsUsed": _rounded(fps_used, 2),
        "visibleSide": visible_side,
        "hiddenSide": hidden_side,
        "eventCount": len(stride_peaks),
        "screeningContext": {
            "greenCount": sum(1 for metric in usable if metric.get("status") == "green"),
            "yellowCount": sum(1 for metric in usable if metric.get("status") == "yellow"),
            "redCount": sum(1 for metric in usable if metric.get("status") == "red"),
        },
        "metrics": metrics,
        "notes": notes,
    }


def build_instability_map(model_key: str, result: dict) -> dict:
    csv_value = result.get("csv_file")
    if not csv_value:
        return {"available": False, "reason": "No training-safe CSV was returned by the model."}

    csv_path = Path(csv_value)
    if not csv_path.exists():
        return {"available": False, "reason": "The training-safe CSV is not available on disk."}

    df = pd.read_csv(csv_path)
    if len(df) < 2:
        return {"available": False, "reason": "Not enough frames for instability visualization."}

    try:
        biometrics = calculate_side_view_biometrics(df, result)
    except Exception as exc:
        biometrics = {
            "available": False,
            "referenceFrame": "Unavailable",
            "metrics": [],
            "notes": [f"Side-view clinical gait metrics could not be calculated: {exc}"],
        }

    left_shoulder = _landmark_xy(df, 11)
    right_shoulder = _landmark_xy(df, 12)
    left_hip = _landmark_xy(df, 23)
    right_hip = _landmark_xy(df, 24)
    body_center = np.mean(np.stack([left_shoulder, right_shoulder, left_hip, right_hip]), axis=0)
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
    hip_width = np.linalg.norm(left_hip - right_hip, axis=1)

    scale_landmarks = [0, 11, 12, 23, 24, 25, 26, 27, 28, 31, 32]
    xs = np.column_stack([_landmark_xy(df, landmark_id)[:, 0] for landmark_id in scale_landmarks])
    ys = np.column_stack([_landmark_xy(df, landmark_id)[:, 1] for landmark_id in scale_landmarks])
    body_width = float(np.nanmedian(np.nanmax(xs, axis=1) - np.nanmin(xs, axis=1)))
    body_height = float(np.nanmedian(np.nanmax(ys, axis=1) - np.nanmin(ys, axis=1)))
    shoulder_hip_scale = float(np.nanmedian(np.concatenate([shoulder_width, hip_width])) * 3.0)
    scale = max(body_height, body_width, shoulder_hip_scale)
    if not math.isfinite(scale) or scale <= 1e-6:
        scale = 1.0

    frame_index = np.arange(len(body_center), dtype=float)
    path_fit = np.column_stack([
        np.polyval(np.polyfit(frame_index, body_center[:, 0], 1), frame_index),
        np.polyval(np.polyfit(frame_index, body_center[:, 1], 1), frame_index),
    ])
    path_residual = body_center - path_fit
    path_velocity = np.diff(body_center, axis=0)
    path_speed = np.linalg.norm(path_velocity, axis=1) / scale if len(path_velocity) else np.array([0.0])
    path_mean_speed = float(np.mean(path_speed)) if len(path_speed) else 0.0
    path_sway = float(np.mean(np.linalg.norm(path_residual, axis=1)) / scale)
    path_rhythm_cv = float(np.std(path_speed) / (path_mean_speed + 1e-6)) if len(path_speed) else 0.0
    path_sway_instability = _metric_instability(path_sway, *CLINICAL_LIMITS["global"]["path_sway"])
    path_rhythm_instability = _metric_instability(path_rhythm_cv, *CLINICAL_LIMITS["global"]["path_rhythm_cv"])

    lm = {landmark_id: _landmark_metrics(df, landmark_id, scale, body_center) for landmark_id in range(33)}
    left_arm_swing = float(np.mean([lm[13]["motion_range"], lm[15]["motion_range"]]))
    right_arm_swing = float(np.mean([lm[14]["motion_range"], lm[16]["motion_range"]]))
    left_leg_motion = float(np.mean([lm[25]["motion_range"], lm[27]["motion_range"]]))
    right_leg_motion = float(np.mean([lm[26]["motion_range"], lm[28]["motion_range"]]))
    left_foot_clearance = float(np.mean([lm[29]["y_range"], lm[31]["y_range"]]))
    right_foot_clearance = float(np.mean([lm[30]["y_range"], lm[32]["y_range"]]))

    def asym(a, b):
        return float(abs(a - b) / max(abs(a), abs(b), 1e-6))

    arm_asymmetry = asym(left_arm_swing, right_arm_swing)
    leg_asymmetry = asym(left_leg_motion, right_leg_motion)
    foot_asymmetry = asym(left_foot_clearance, right_foot_clearance)
    clinical_metrics = _metric_by_key(biometrics) if biometrics.get("available") else {}

    def clinical_metric(key, side=None):
        metric = clinical_metrics.get(key)
        return _clinical_metric_entry(metric, side=side)

    def mean_metric(ids, key):
        return float(np.mean([lm[item][key] for item in ids]))

    def base_measurements(ids):
        return {
            "motionRange": mean_metric(ids, "motion_range"),
            "sway": mean_metric(ids, "sway"),
            "meanSpeed": mean_metric(ids, "mean_speed"),
            "velocityVariability": mean_metric(ids, "speed_cv"),
            "jerkIrregularity": mean_metric(ids, "jerk_ratio"),
            "jitter": mean_metric(ids, "jitter"),
            "visibility": mean_metric(ids, "visibility"),
        }

    region_inputs = {
        "head": {
            "measurements": {**base_measurements([0, 7, 8]), "pathSway": path_sway, "pathRhythm": path_rhythm_cv},
            "clinicalMetrics": [
                clinical_metric("head_sway"),
                clinical_metric("trunk_control"),
                clinical_metric("pelvic_path_deviation"),
            ],
        },
        "shoulders": {
            "measurements": {**base_measurements([11, 12]), "pathSway": path_sway, "pathRhythm": path_rhythm_cv},
            "clinicalMetrics": [
                clinical_metric("shoulder_sway"),
                clinical_metric("trunk_control"),
                clinical_metric("arm_swing_asymmetry"),
                clinical_metric("arm_swing_amplitude"),
            ],
        },
        "left_arm": {
            "measurements": {**base_measurements([11, 13, 15]), "leftRightAsymmetry": arm_asymmetry},
            "clinicalMetrics": [
                clinical_metric("arm_swing_amplitude", "left"),
                clinical_metric("arm_swing_asymmetry"),
                clinical_metric("shoulder_sway"),
            ],
        },
        "right_arm": {
            "measurements": {**base_measurements([12, 14, 16]), "leftRightAsymmetry": arm_asymmetry},
            "clinicalMetrics": [
                clinical_metric("arm_swing_amplitude", "right"),
                clinical_metric("arm_swing_asymmetry"),
                clinical_metric("shoulder_sway"),
            ],
        },
        "left_forearm_hand": {
            "measurements": {**base_measurements([13, 15, 17, 19, 21]), "leftRightAsymmetry": arm_asymmetry},
            "clinicalMetrics": [
                clinical_metric("elbow_flexion_posture", "left"),
                clinical_metric("elbow_rom", "left"),
                clinical_metric("forearm_hand_excursion", "left"),
                clinical_metric("arm_swing_amplitude", "left"),
                clinical_metric("arm_swing_asymmetry"),
            ],
        },
        "right_forearm_hand": {
            "measurements": {**base_measurements([14, 16, 18, 20, 22]), "leftRightAsymmetry": arm_asymmetry},
            "clinicalMetrics": [
                clinical_metric("elbow_flexion_posture", "right"),
                clinical_metric("elbow_rom", "right"),
                clinical_metric("forearm_hand_excursion", "right"),
                clinical_metric("arm_swing_amplitude", "right"),
                clinical_metric("arm_swing_asymmetry"),
            ],
        },
        "trunk": {
            "measurements": {**base_measurements([11, 12, 23, 24]), "pathSway": path_sway, "pathRhythm": path_rhythm_cv},
            "clinicalMetrics": [
                clinical_metric("trunk_control"),
                clinical_metric("shoulder_sway"),
                clinical_metric("pelvic_path_deviation"),
                clinical_metric("walking_speed"),
                clinical_metric("cadence"),
                clinical_metric("stride_variability"),
            ],
        },
        "pelvis": {
            "measurements": {**base_measurements([23, 24]), "pathSway": path_sway, "pathRhythm": path_rhythm_cv, "leftRightAsymmetry": leg_asymmetry},
            "clinicalMetrics": [
                clinical_metric("walking_speed"),
                clinical_metric("step_length"),
                clinical_metric("stride_length"),
                clinical_metric("step_time"),
                clinical_metric("stride_time"),
                clinical_metric("symmetry_index"),
            ],
        },
        "left_knee": {
            "measurements": {**base_measurements([23, 25, 27]), "leftRightAsymmetry": leg_asymmetry},
            "clinicalMetrics": [
                clinical_metric("walking_speed"),
                clinical_metric("step_length"),
                clinical_metric("stride_length"),
                clinical_metric("hip_rom", "left"),
                clinical_metric("knee_rom", "left"),
                clinical_metric("swing_phase", "left"),
            ],
        },
        "right_knee": {
            "measurements": {**base_measurements([24, 26, 28]), "leftRightAsymmetry": leg_asymmetry},
            "clinicalMetrics": [
                clinical_metric("walking_speed"),
                clinical_metric("step_length"),
                clinical_metric("stride_length"),
                clinical_metric("hip_rom", "right"),
                clinical_metric("knee_rom", "right"),
                clinical_metric("swing_phase", "right"),
            ],
        },
        "left_ankle": {
            "measurements": {**base_measurements([27, 29, 31]), "leftRightAsymmetry": foot_asymmetry, "footClearance": left_foot_clearance},
            "clinicalMetrics": [
                clinical_metric("walking_speed"),
                clinical_metric("step_length"),
                clinical_metric("stride_length"),
                clinical_metric("step_time"),
                clinical_metric("stride_time"),
                clinical_metric("knee_rom", "left"),
                clinical_metric("foot_clearance", "left"),
                clinical_metric("ankle_excursion", "left"),
                clinical_metric("swing_phase", "left"),
            ],
        },
        "right_ankle": {
            "measurements": {**base_measurements([28, 30, 32]), "leftRightAsymmetry": foot_asymmetry, "footClearance": right_foot_clearance},
            "clinicalMetrics": [
                clinical_metric("walking_speed"),
                clinical_metric("step_length"),
                clinical_metric("stride_length"),
                clinical_metric("step_time"),
                clinical_metric("stride_time"),
                clinical_metric("knee_rom", "right"),
                clinical_metric("foot_clearance", "right"),
                clinical_metric("ankle_excursion", "right"),
                clinical_metric("swing_phase", "right"),
            ],
        },
    }

    parts = []
    for part in BODY_PARTS:
        region = region_inputs[part["key"]]
        clinical_region_metrics = region["clinicalMetrics"]
        visibility_penalty = _clamp01((0.45 - region["measurements"].get("visibility", 1.0)) / 0.35) * 0.08
        score, status, status_label, active_drivers = _region_from_clinical_metrics(clinical_region_metrics)
        score = _clamp01(score + visibility_penalty)
        if visibility_penalty > 0:
            status, status_label = _status_from_score(score)
        parts.append({
            "key": part["key"],
            "label": part["label"],
            "x": part["x"],
            "y": part["y"],
            "score": round(float(score), 3),
            "status": status,
            "statusLabel": status_label,
            "clinicalMeaning": _clinical_meaning(status, active_drivers),
            "measurements": {
                "motionRange": round(region["measurements"].get("motionRange", 0.0), 4),
                "sway": round(region["measurements"].get("sway", 0.0), 4),
                "pathSway": round(region["measurements"].get("pathSway", 0.0), 4),
                "pathRhythm": round(region["measurements"].get("pathRhythm", 0.0), 4),
                "velocityVariability": round(region["measurements"].get("velocityVariability", 0.0), 4),
                "jerkIrregularity": round(region["measurements"].get("jerkIrregularity", 0.0), 4),
                "jitter": round(region["measurements"].get("jitter", 0.0), 4),
                "meanSpeed": round(region["measurements"].get("meanSpeed", 0.0), 4),
                "leftRightAsymmetry": round(region["measurements"].get("leftRightAsymmetry", 0.0), 4),
                "footClearance": round(region["measurements"].get("footClearance", 0.0), 4),
                "visibility": round(region["measurements"].get("visibility", 0.0), 4),
            },
            "clinicalMetrics": [metric for metric in clinical_region_metrics if metric],
            "featureScores": {
                metric["key"]: _clinical_status_score(metric["status"])
                for metric in clinical_region_metrics
                if metric and metric.get("status") in {"green", "yellow", "red"}
            },
        })

    counts = {
        "stable": sum(1 for part in parts if part["status"] == "stable"),
        "moderate": sum(1 for part in parts if part["status"] == "moderate"),
        "severe": sum(1 for part in parts if part["status"] == "severe"),
    }
    highest = sorted(parts, key=lambda item: item["score"], reverse=True)[:3]

    return {
        "available": True,
        "modelKey": model_key,
        "createdFrom": str(csv_path),
        "summary": {
            "overallStatus": "High instability" if counts["severe"] else "Moderate control regions" if counts["moderate"] else "Stable",
            "stableCount": counts["stable"],
            "moderateCount": counts["moderate"],
            "severeCount": counts["severe"],
            "highestRegions": [item["label"] for item in highest],
        },
        "clinicalMetrics": biometrics,
        "referenceFrame": biometrics.get("referenceFrame", "Side-view clinical gait measures"),
        "parts": parts,
    }


def extract_csv(model_key: str, input_path: Path) -> Path:
    project_root = MODEL_ROOTS[model_key]
    script_path = (
        project_root / "04_video_extraction" / "run_component2_sca_clinical_video.py"
        if model_key == "sca"
        else project_root / "04_video_extraction" / "run_component2_koa_clinical_video.py"
    )

    sys.path.insert(0, str(project_root))
    spec = importlib.util.spec_from_file_location(f"component2_{model_key}_extractor", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load extractor script: {script_path}")
    extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extractor)
    extractor.ensure_dirs()
    with contextlib.redirect_stdout(sys.stderr):
        return Path(extractor.extract_training_safe_csv(input_path))


def predict(model_key: str, csv_path: Path, direction: str | None, fps: float | None) -> dict:
    project_root = MODEL_ROOTS[model_key]
    sys.path.insert(0, str(project_root))

    if model_key == "sca":
        predictor = importlib.import_module("src.sca_predictor")
        return predictor.predict_component2_sca_v28_antalgic_hard_negative(
            csv_path=csv_path,
            save_report=True,
            direction=direction,
            fps=fps,
        )

    predictor = importlib.import_module("src.koa_predictor")
    return predictor.predict_component2_koa(
        csv_path=csv_path,
        direction=direction,
        fps_used=fps,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run Component 2 SCA/KOA inference.")
    parser.add_argument("--model", required=True, choices=["sca", "koa"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-type", required=True, choices=["video", "csv"])
    parser.add_argument("--direction", choices=["L2R", "R2L"], default=None)
    parser.add_argument("--fps", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    csv_path = input_path if args.input_type == "csv" else extract_csv(args.model, input_path)
    result = predict(args.model, csv_path, args.direction, args.fps)
    result["source_input_file"] = str(input_path)
    result["source_input_type"] = args.input_type
    result["instability_map"] = build_instability_map(args.model, result)

    print(json.dumps({"ok": True, "result": json_safe(result)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"ok": False, "message": str(error)}))
        raise
