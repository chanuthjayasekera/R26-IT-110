from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import audit_component3_pd_v19_direction_transform as direction_audit
import build_component3_pd_v18_development_datasets as v18_builder
import predict_component3_pd_from_csv as pd_preprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "05_preprocessed_windows"
    / "component_3"
    / "pd_v19_60f_stride15_direction_corrected_development_windows.csv"
)
REPORT_DIR = PROJECT_ROOT / "07_reports" / "component_3"
PREPROCESSING_OUT = (
    REPORT_DIR / "component3_pd_v19_direction_corrected_preprocessing_report.csv"
)
REPORT_OUT = (
    REPORT_DIR / "component3_pd_v19_direction_corrected_dataset_report.json"
)
TEXT_OUT = (
    REPORT_DIR / "component3_pd_v19_direction_corrected_dataset_report.txt"
)

CONFIG = {"name": "60f_stride15_x_flip_swap_lr", "window_size": 60, "stride": 15}
DIRECTION_TRANSFORM = "x_flip_swap_lr"


def prepare_sequence(row: dict[str, str]) -> tuple[np.ndarray, int, int]:
    path = PROJECT_ROOT / row["relative_path"]
    raw_df = pd_preprocess.load_pose_csv(path)
    source_frames = len(raw_df)
    sequence = pd_preprocess.sequence_from_df(raw_df)
    time_seconds = pd_preprocess.get_time_seconds(raw_df, float(row["fps"]))
    sequence = pd_preprocess.resample_sequence(
        sequence, time_seconds, pd_preprocess.TARGET_FPS
    )
    resampled_frames = len(sequence)
    sequence = pd_preprocess.normalize_pose(sequence)
    if row["direction"].upper() == "R2L":
        sequence = direction_audit.transform(sequence, DIRECTION_TRANSFORM)
    return sequence, source_frames, resampled_frames


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = v18_builder.read_inventory()
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    wrote_header = False
    expected_columns = None
    reports = []
    window_counts = Counter()
    video_counts = Counter()
    subject_sets = defaultdict(set)

    for index, row in enumerate(inventory, start=1):
        sequence, source_frames, resampled_frames = prepare_sequence(row)
        features, report = v18_builder.make_video_features(
            row,
            sequence,
            source_frames,
            resampled_frames,
            CONFIG,
        )
        columns = features.columns.tolist()
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise RuntimeError(f"Column mismatch: {row['training_safe_file']}")
        features.to_csv(
            OUTPUT_PATH,
            mode="a",
            header=not wrote_header,
            index=False,
        )
        wrote_header = True
        report["direction_transform"] = (
            DIRECTION_TRANSFORM
            if row["direction"].upper() == "R2L"
            else "identity"
        )
        reports.append(report)
        key = f"{row['permanent_split']}|{row['class_label']}"
        window_counts[key] += len(features)
        video_counts[key] += 1
        subject_sets[key].add(row["class_subject_id"])
        if index % 25 == 0 or index == len(inventory):
            print(f"Processed {index}/{len(inventory)} videos", flush=True)

    feature_count = len(expected_columns or []) - len(
        v18_builder.METADATA_COLUMNS
    )
    if feature_count != 1028:
        raise RuntimeError(f"Expected 1028 features, found {feature_count}")
    pd.DataFrame(reports).to_csv(PREPROCESSING_OUT, index=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "PD v19 normal-inclusive direction-corrected development data",
        "dataset_path": str(OUTPUT_PATH),
        "sha256": v18_builder.file_sha256(OUTPUT_PATH),
        "size_bytes": OUTPUT_PATH.stat().st_size,
        "input_video_count": len(inventory),
        "allowed_splits": sorted(v18_builder.ALLOWED_SPLITS),
        "locked_test_processed": False,
        "hard_pd_guard_processed": False,
        "challenge_only_processed": False,
        "window_size": CONFIG["window_size"],
        "stride": CONFIG["stride"],
        "max_windows_per_video": v18_builder.MAX_WINDOWS_PER_VIDEO,
        "direction_transform_r2l": DIRECTION_TRANSFORM,
        "direction_transform_selection_report": str(
            direction_audit.SUMMARY_OUT
        ),
        "feature_count": feature_count,
        "video_counts": dict(sorted(video_counts.items())),
        "subject_counts": {
            key: len(value) for key, value in sorted(subject_sets.items())
        },
        "window_counts": dict(sorted(window_counts.items())),
        "preprocessing_report": str(PREPROCESSING_OUT),
    }
    REPORT_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    TEXT_OUT.write_text(
        "COMPONENT 3 PD V19 DIRECTION-CORRECTED DATASET REPORT\n"
        + "=" * 76
        + "\n\n"
        + json.dumps(summary, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
