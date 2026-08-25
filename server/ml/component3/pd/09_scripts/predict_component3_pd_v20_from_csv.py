from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

import build_component3_pd_v18_development_datasets as base_builder
import build_component3_pd_v19_direction_corrected_dataset as v19_builder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "06_models"
    / "component_3"
    / "component3_pd_model_v20_final_attempt.joblib"
)
FEATURES_PATH = (
    PROJECT_ROOT
    / "06_models"
    / "component_3"
    / "component3_pd_feature_list_v20.json"
)
SETTINGS_PATH = (
    PROJECT_ROOT
    / "06_models"
    / "component_3"
    / "component3_pd_settings_v20_final_attempt.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a landmark CSV with PD v20")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--direction", required=True, choices=["L2R", "R2L"])
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--report")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    row = {
        "training_safe_file": csv_path.name,
        "class_label": "inference",
        "person_id": "unknown",
        "class_subject_id": "inference_unknown",
        "direction": args.direction,
        "fps": str(args.fps),
        "source_type": "inference",
        "source_group": "external_video",
        "permanent_split": "inference",
        "legacy_v14_test_subject": "0",
        "relative_path": str(csv_path.relative_to(PROJECT_ROOT)),
    }
    sequence, source_frames, resampled_frames = v19_builder.prepare_sequence(row)
    features, preprocessing = base_builder.make_video_features(
        row,
        sequence,
        source_frames,
        resampled_frames,
        v19_builder.CONFIG,
    )

    model = joblib.load(MODEL_PATH)
    feature_columns = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    probabilities = model.predict_proba(
        features[feature_columns].to_numpy(dtype=np.float32)
    )[:, 1]
    aggregation = settings["selected_aggregation"]
    if aggregation == "median":
        aggregated = float(np.median(probabilities))
    elif aggregation == "mean":
        aggregated = float(np.mean(probabilities))
    else:
        ordered = np.sort(probabilities)
        trim = int(len(ordered) * 0.10)
        if trim > 0 and len(ordered) - 2 * trim > 0:
            ordered = ordered[trim:-trim]
        aggregated = float(np.mean(ordered))
    threshold = float(settings["selected_threshold"])
    output = {
        "model": "PD v20 normal-aware standalone candidate",
        "csv": str(csv_path),
        "direction": args.direction,
        "direction_transform": "x_flip_swap_lr",
        "fps": args.fps,
        "source_frames": source_frames,
        "resampled_frames": resampled_frames,
        "total_windows": len(probabilities),
        "aggregation": aggregation,
        "aggregated_pd_probability": aggregated,
        "mean_pd_probability": float(np.mean(probabilities)),
        "median_pd_probability": float(np.median(probabilities)),
        "minimum_window_probability": float(np.min(probabilities)),
        "maximum_window_probability": float(np.max(probabilities)),
        "positive_window_ratio_at_threshold": float(
            np.mean(probabilities >= threshold)
        ),
        "threshold": threshold,
        "detected_as_pd": bool(aggregated >= threshold),
        "preprocessing": preprocessing,
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
