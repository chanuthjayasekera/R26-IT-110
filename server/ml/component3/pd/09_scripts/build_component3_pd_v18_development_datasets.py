from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import predict_component3_pd_from_csv as pd_preprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = (
    PROJECT_ROOT
    / "04_splits"
    / "component_3_pd_v18"
    / "component3_pd_v18_video_inventory.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "05_preprocessed_windows" / "component_3"
REPORT_DIR = PROJECT_ROOT / "07_reports" / "component_3"

DATASET_CONFIGS = (
    {"name": "60f_stride15", "window_size": 60, "stride": 15},
    {"name": "90f_stride15", "window_size": 90, "stride": 15},
)
MAX_WINDOWS_PER_VIDEO = 80
ALLOWED_SPLITS = {"train", "validation"}
FORBIDDEN_SPLITS = {"test", "hard_pd_guard", "challenge_only"}

METADATA_COLUMNS = [
    "training_safe_file",
    "class_label",
    "label_pd",
    "binary_video_label",
    "person_id",
    "subject_id",
    "direction",
    "fps_used",
    "source_type",
    "source_group",
    "permanent_split",
    "legacy_v14_test_subject",
    "window_config",
    "window_size",
    "stride",
    "window_id",
    "start_frame",
    "source_frames",
    "resampled_frames",
    "padded",
    "windows_before_cap",
    "windows_retained",
]


def read_inventory() -> list[dict[str, str]]:
    with INVENTORY_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    development = [
        row
        for row in rows
        if row["permanent_split"] in ALLOWED_SPLITS
        and row["core_v18"] == "1"
    ]
    forbidden = [
        row
        for row in development
        if row["permanent_split"] in FORBIDDEN_SPLITS
    ]
    if forbidden:
        raise RuntimeError("Locked test or challenge record entered development input.")
    if not development:
        raise RuntimeError("No train/validation inventory records found.")
    return development


def evenly_capped_starts(
    sequence_length: int,
    window_size: int,
    stride: int,
    max_windows: int,
) -> tuple[list[int], int]:
    starts = list(range(0, sequence_length - window_size + 1, stride))
    before_cap = len(starts)
    if before_cap <= max_windows:
        return starts, before_cap
    indices = np.linspace(0, before_cap - 1, max_windows).round().astype(int)
    selected = [starts[index] for index in sorted(set(indices.tolist()))]
    if len(selected) != max_windows:
        raise RuntimeError("Even window cap did not retain the requested count.")
    return selected, before_cap


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_sequence(row: dict[str, str]) -> tuple[np.ndarray, int, int]:
    path = PROJECT_ROOT / row["relative_path"]
    raw_df = pd_preprocess.load_pose_csv(path)
    source_frames = len(raw_df)
    fps_used = float(row["fps"])
    sequence = pd_preprocess.sequence_from_df(raw_df)
    time_seconds = pd_preprocess.get_time_seconds(raw_df, fps_used)
    sequence = pd_preprocess.resample_sequence(
        sequence, time_seconds, pd_preprocess.TARGET_FPS
    )
    resampled_frames = len(sequence)
    sequence = pd_preprocess.normalize_pose(sequence)
    sequence = pd_preprocess.standardize_direction(
        sequence, row["direction"]
    )
    return sequence, source_frames, resampled_frames


def make_video_features(
    row: dict[str, str],
    sequence: np.ndarray,
    source_frames: int,
    resampled_frames: int,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    window_size = int(config["window_size"])
    stride = int(config["stride"])
    padded = len(sequence) < window_size
    working_sequence = (
        pd_preprocess.edge_pad_to_window(sequence, window_size)
        if padded
        else sequence
    )
    starts, before_cap = evenly_capped_starts(
        len(working_sequence),
        window_size,
        stride,
        MAX_WINDOWS_PER_VIDEO,
    )
    rows = []
    for window_id, start in enumerate(starts):
        window = working_sequence[start : start + window_size]
        metadata = {
            "training_safe_file": row["training_safe_file"],
            "class_label": row["class_label"],
            "label_pd": int(row["class_label"] == "pd"),
            "binary_video_label": int(row["class_label"] == "pd"),
            "person_id": row["person_id"],
            "subject_id": row["class_subject_id"],
            "direction": row["direction"],
            "fps_used": float(row["fps"]),
            "source_type": row["source_type"],
            "source_group": row["source_group"],
            "permanent_split": row["permanent_split"],
            "legacy_v14_test_subject": int(row["legacy_v14_test_subject"]),
            "window_config": config["name"],
            "window_size": window_size,
            "stride": stride,
            "window_id": window_id,
            "start_frame": start,
            "source_frames": source_frames,
            "resampled_frames": resampled_frames,
            "padded": int(padded),
            "windows_before_cap": before_cap,
            "windows_retained": len(starts),
        }
        metadata.update(pd_preprocess.aggregate_features(window))
        rows.append(metadata)

    feature_df = pd.DataFrame(rows)
    feature_df, biomech_columns = pd_preprocess.add_pd_biomech_features(
        feature_df
    )
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    base_feature_columns = [
        column
        for column in feature_df.columns
        if column.startswith(
            (
                "mean_",
                "std_",
                "min_",
                "max_",
                "range_",
                "mean_abs_vel_",
                "std_vel_",
            )
        )
    ]
    numeric_feature_columns = list(
        dict.fromkeys(base_feature_columns + biomech_columns)
    )
    feature_df = feature_df[
        METADATA_COLUMNS + numeric_feature_columns
    ]
    report = {
        "training_safe_file": row["training_safe_file"],
        "class_label": row["class_label"],
        "subject_id": row["class_subject_id"],
        "direction": row["direction"],
        "permanent_split": row["permanent_split"],
        "window_config": config["name"],
        "window_size": window_size,
        "stride": stride,
        "source_frames": source_frames,
        "resampled_frames": resampled_frames,
        "padded": int(padded),
        "windows_before_cap": before_cap,
        "windows_retained": len(starts),
        "feature_count": len(numeric_feature_columns),
        "status": "ok",
    }
    return feature_df, report


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = read_inventory()

    dataset_paths = {
        config["name"]: OUTPUT_DIR
        / f"pd_v18_{config['name']}_development_windows.csv"
        for config in DATASET_CONFIGS
    }
    for path in dataset_paths.values():
        if path.exists():
            path.unlink()

    wrote_header = {config["name"]: False for config in DATASET_CONFIGS}
    expected_columns: dict[str, list[str]] = {}
    preprocessing_reports: list[dict] = []
    window_counts: dict[str, Counter] = defaultdict(Counter)
    video_counts: dict[str, Counter] = defaultdict(Counter)
    subject_sets: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for video_index, row in enumerate(inventory, start=1):
        sequence, source_frames, resampled_frames = prepare_sequence(row)
        for config in DATASET_CONFIGS:
            name = config["name"]
            features, report = make_video_features(
                row,
                sequence,
                source_frames,
                resampled_frames,
                config,
            )
            columns = features.columns.tolist()
            if name not in expected_columns:
                expected_columns[name] = columns
            elif columns != expected_columns[name]:
                raise RuntimeError(
                    f"Column mismatch in {row['training_safe_file']} for {name}"
                )
            features.to_csv(
                dataset_paths[name],
                mode="a",
                header=not wrote_header[name],
                index=False,
            )
            wrote_header[name] = True
            preprocessing_reports.append(report)
            key = f"{row['permanent_split']}|{row['class_label']}"
            window_counts[name][key] += len(features)
            video_counts[name][key] += 1
            subject_sets[name][key].add(row["class_subject_id"])

        if video_index % 25 == 0 or video_index == len(inventory):
            print(f"Processed {video_index}/{len(inventory)} videos")

    report_path = (
        REPORT_DIR / "component3_pd_v18_development_preprocessing_report.csv"
    )
    pd.DataFrame(preprocessing_reports).to_csv(report_path, index=False)

    summaries = {}
    for config in DATASET_CONFIGS:
        name = config["name"]
        dataset_path = dataset_paths[name]
        feature_count = len(expected_columns[name]) - len(METADATA_COLUMNS)
        if feature_count != 1028:
            raise RuntimeError(
                f"Expected 1028 features for {name}, found {feature_count}"
            )
        summaries[name] = {
            "dataset_path": str(dataset_path),
            "sha256": file_sha256(dataset_path),
            "size_bytes": dataset_path.stat().st_size,
            "window_size": config["window_size"],
            "stride": config["stride"],
            "max_windows_per_video": MAX_WINDOWS_PER_VIDEO,
            "feature_count": feature_count,
            "video_counts": dict(sorted(video_counts[name].items())),
            "subject_counts": {
                key: len(value)
                for key, value in sorted(subject_sets[name].items())
            },
            "window_counts": dict(sorted(window_counts[name].items())),
        }

    split_subjects = defaultdict(set)
    for row in inventory:
        split_subjects[row["permanent_split"]].add(row["class_subject_id"])
    overlap = sorted(split_subjects["train"] & split_subjects["validation"])
    if overlap:
        raise RuntimeError(f"Train/validation subject overlap: {overlap}")
    if any(row["legacy_v14_test_subject"] == "1" for row in inventory):
        raise RuntimeError("Legacy v14 test subject entered development inventory.")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "PD v18 train/validation development windows only",
        "input_inventory": str(INVENTORY_PATH),
        "input_video_count": len(inventory),
        "allowed_splits": sorted(ALLOWED_SPLITS),
        "locked_test_processed": False,
        "challenge_only_processed": False,
        "train_validation_subject_overlap": overlap,
        "final_refit_policy": (
            "After settings and threshold are frozen and the hard-PD guard passes, "
            "refit on permanent train+validation+hard-PD guard (115 PD videos), "
            "then evaluate once on the locked 20-video PD test."
        ),
        "datasets": summaries,
        "preprocessing_report": str(report_path),
    }
    json_path = REPORT_DIR / "component3_pd_v18_development_dataset_report.json"
    text_path = REPORT_DIR / "component3_pd_v18_development_dataset_report.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    text_path.write_text(
        "COMPONENT 3 PD V18 DEVELOPMENT DATASET REPORT\n"
        + "=" * 72
        + "\n\n"
        + json.dumps(summary, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
