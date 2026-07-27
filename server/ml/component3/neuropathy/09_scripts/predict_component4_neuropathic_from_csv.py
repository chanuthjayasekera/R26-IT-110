"""Predict neuropathic-like gait from a training-safe landmark CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

import predict_component2_koa_from_csv as pose
from neuropathic_feature_engineering import add_neuropathic_features


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = "component4_neuropathic_v11_pd_hard_negative_model.joblib"
FEATURES_FILE = "component4_neuropathic_v11_pd_hard_negative_feature_list.json"
SETTINGS_FILE = "component4_neuropathic_v11_pd_hard_negative_settings.json"
DEFAULT_THRESHOLD = 0.48
TENDENCY_THRESHOLD = 0.30


def resolve_model_dir() -> Path:
    """Find bundled neuropathic V11 artifacts after dashboard/package renaming."""
    candidates = [
        ROOT / "06_models" / "component_3_neuropathic",
        ROOT / "06_models" / "component_4_neuropathic",
    ]
    for model_dir in candidates:
        if (
            (model_dir / MODEL_FILE).exists()
            and (model_dir / FEATURES_FILE).exists()
            and (model_dir / SETTINGS_FILE).exists()
        ):
            return model_dir
    return candidates[0]


MODEL_DIR = resolve_model_dir()
MODEL_PATH = MODEL_DIR / MODEL_FILE
FEATURES_PATH = MODEL_DIR / FEATURES_FILE
SETTINGS_PATH = MODEL_DIR / SETTINGS_FILE


def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError):
        return {"final_threshold": DEFAULT_THRESHOLD}


def pattern_decision(window_probabilities: np.ndarray, settings: dict) -> tuple[bool, dict]:
    pattern = settings.get("recommended_video_setting") or {}
    mean_threshold = float(pattern.get("mean_threshold", settings.get("final_threshold", DEFAULT_THRESHOLD)))
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


def predict(csv_path: Path, direction: str | None = None, fps: float | None = None) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError("Neuropathic V11 model artifacts are missing")

    raw = pose.load_pose_csv(csv_path)
    if direction is None:
        direction = pose.infer_direction_from_filename(csv_path)
    if direction is None:
        direction, direction_delta, direction_confidence = pose.estimate_direction_from_pose_df(raw)
    else:
        direction_delta, direction_confidence = 0.0, "filename"

    if fps is None:
        fps = pose.infer_fps_from_filename(csv_path) or 30.0

    features = pose.preprocess_single_csv(csv_path, direction, fps)
    features, _ = add_neuropathic_features(features)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    with open(FEATURES_PATH, encoding="utf-8") as file:
        feature_cols = json.load(file)
    missing = [c for c in feature_cols if c not in features]
    if missing:
        raise ValueError(f"Missing neuropathic features: {missing[:20]}")

    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(features[feature_cols].to_numpy(np.float32))
    positive_index = list(model.classes_).index(1)
    window_probabilities = probabilities[:, positive_index]
    settings = load_settings()
    threshold = float(settings.get("final_threshold", DEFAULT_THRESHOLD))
    detected, pattern_details = pattern_decision(window_probabilities, settings)
    probability = float(pattern_details["mean_probability"])
    tendency = not detected and probability >= TENDENCY_THRESHOLD

    duration = pose.get_clean_duration_seconds(csv_path, fps)
    reliability, reasons = pose.reliability_from_windows(
        total_windows=len(features),
        clean_duration_sec=duration,
        direction_confidence=direction_confidence,
    )
    reasons = [reason.replace("KOA", "neuropathic gait") for reason in reasons]
    if reliability == "Low":
        final_result = "Insufficient reliable gait data for confirmed neuropathic-pattern screening"
    elif detected:
        final_result = "Neuropathic-like gait pattern detected"
    elif tendency:
        final_result = "Borderline neuropathic-like gait tendency; review recommended"
    else:
        final_result = "No neuropathic-like gait pattern detected"

    return {
        "model_version": "Neuropathic V11 normal-inclusive Rasika positive plus PD hard-negative pattern ExtraTrees",
        "csv_file": str(csv_path),
        "direction": direction,
        "direction_confidence": direction_confidence,
        "direction_delta_x": round(float(direction_delta), 5),
        "fps_used": float(fps),
        "clean_duration_sec": round(float(duration), 3),
        "total_windows": int(len(features)),
        "neuropathic_probability": round(probability, 4),
        "detection_threshold": threshold,
        "tendency_threshold": TENDENCY_THRESHOLD,
        "pattern_setting": {
            "mean_threshold": pattern_details["mean_threshold"],
            "window_threshold": pattern_details["window_threshold"],
            "ratio_threshold": pattern_details["ratio_threshold"],
            "count_threshold": pattern_details["count_threshold"],
            "max_threshold": pattern_details["max_threshold"],
            "min_total_windows": pattern_details["min_total_windows"],
        },
        "detected": bool(detected),
        "borderline_tendency": bool(tendency),
        "max_probability": round(float(pattern_details["max_probability"]), 4),
        "positive_window_count": int(pattern_details["positive_window_count"]),
        "positive_window_ratio": round(float(pattern_details["positive_window_ratio"]), 4),
        "reliability": reliability,
        "reliability_reasons": reasons,
        "final_result": final_result,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the final neuropathic gait specialist")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--direction", choices=["L2R", "R2L"])
    parser.add_argument("--fps", type=float)
    args = parser.parse_args()
    result = predict(args.csv_path, args.direction, args.fps)

    print("\n======================================")
    print(" COMPONENT 3 NEUROPATHIC GAIT RESULT")
    print(" FINAL MODEL: NEUROPATHIC V11 PD-HARD-NEGATIVE PATTERN")
    print("======================================")
    for key, value in result.items():
        label = key.replace("_", " ").title()
        print(f"{label + ':':30} {value}")
    print("======================================")


if __name__ == "__main__":
    main()
