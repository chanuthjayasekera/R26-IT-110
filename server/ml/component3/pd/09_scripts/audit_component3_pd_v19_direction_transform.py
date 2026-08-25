from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import predict_component3_pd_from_csv as pd_preprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = (
    PROJECT_ROOT
    / "04_splits"
    / "component_3_pd_v18"
    / "component3_pd_v18_video_inventory.csv"
)
REPORT_DIR = PROJECT_ROOT / "07_reports" / "component_3"
DETAIL_OUT = REPORT_DIR / "component3_pd_v19_direction_transform_audit.csv"
SUMMARY_OUT = REPORT_DIR / "component3_pd_v19_direction_transform_audit.json"

WINDOW_SIZE = 60
STRIDE = 15
MAX_WINDOWS = 80
LEFT_RIGHT_PAIRS = [
    (1, 4),
    (2, 5),
    (3, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
    (25, 26),
    (27, 28),
    (29, 30),
    (31, 32),
]


def read_inventory() -> list[dict[str, str]]:
    with INVENTORY_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def transform(sequence: np.ndarray, mode: str) -> np.ndarray:
    output = sequence.copy()
    if mode == "no_flip":
        return output
    output[:, :, 0] *= -1.0
    if mode == "x_flip":
        return output
    if mode == "x_flip_swap_lr":
        for left, right in LEFT_RIGHT_PAIRS:
            left_values = output[:, left, :].copy()
            output[:, left, :] = output[:, right, :]
            output[:, right, :] = left_values
        return output
    raise ValueError(mode)


def prepare(row: dict[str, str]) -> np.ndarray:
    path = PROJECT_ROOT / row["relative_path"]
    frame = pd_preprocess.load_pose_csv(path)
    sequence = pd_preprocess.sequence_from_df(frame)
    time_seconds = pd_preprocess.get_time_seconds(frame, float(row["fps"]))
    sequence = pd_preprocess.resample_sequence(
        sequence, time_seconds, pd_preprocess.TARGET_FPS
    )
    return pd_preprocess.normalize_pose(sequence)


def video_vector(sequence: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if len(sequence) < WINDOW_SIZE:
        sequence = pd_preprocess.edge_pad_to_window(sequence, WINDOW_SIZE)
    starts = list(range(0, len(sequence) - WINDOW_SIZE + 1, STRIDE))
    if len(starts) > MAX_WINDOWS:
        indexes = np.linspace(0, len(starts) - 1, MAX_WINDOWS).round().astype(int)
        starts = [starts[index] for index in sorted(set(indexes.tolist()))]
    rows = [
        pd_preprocess.aggregate_features(
            sequence[start : start + WINDOW_SIZE]
        )
        for start in starts
    ]
    columns = list(rows[0])
    matrix = np.asarray(
        [[row[column] for column in columns] for row in rows],
        dtype=np.float64,
    )
    return matrix.mean(axis=0), columns


def main() -> None:
    rows = [
        row
        for row in read_inventory()
        if row["class_label"] == "pd"
        and row["permanent_split"] == "train"
    ]
    by_subject: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_subject[row["class_subject_id"]][row["direction"]] = row
    paired = {
        subject: directions
        for subject, directions in by_subject.items()
        if {"L2R", "R2L"} <= set(directions)
    }
    if not paired:
        raise RuntimeError("No paired training PD subjects found.")

    modes = ["no_flip", "x_flip", "x_flip_swap_lr"]
    left_vectors = {}
    right_vectors = {mode: {} for mode in modes}
    feature_columns = None
    for index, (subject, directions) in enumerate(sorted(paired.items()), start=1):
        left_sequence = prepare(directions["L2R"])
        right_sequence = prepare(directions["R2L"])
        left_vector, columns = video_vector(left_sequence)
        if feature_columns is None:
            feature_columns = columns
        elif feature_columns != columns:
            raise RuntimeError("Feature columns changed during audit.")
        left_vectors[subject] = left_vector
        for mode in modes:
            right_vectors[mode][subject] = video_vector(
                transform(right_sequence, mode)
            )[0]
        if index % 10 == 0 or index == len(paired):
            print(f"Processed {index}/{len(paired)} paired PD subjects", flush=True)

    all_vectors = [left_vectors[subject] for subject in sorted(paired)]
    for mode in modes:
        all_vectors.extend(
            right_vectors[mode][subject] for subject in sorted(paired)
        )
    reference = np.vstack(all_vectors)
    median = np.median(reference, axis=0)
    q75, q25 = np.percentile(reference, [75, 25], axis=0)
    scale = q75 - q25
    scale[scale < 1e-6] = 1.0

    detail_rows = []
    summary = {}
    for mode in modes:
        distances = []
        cosine_distances = []
        for subject in sorted(paired):
            left = (left_vectors[subject] - median) / scale
            right = (right_vectors[mode][subject] - median) / scale
            mae = float(np.mean(np.abs(left - right)))
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            cosine = (
                float(1.0 - np.dot(left, right) / denominator)
                if denominator > 0
                else 0.0
            )
            distances.append(mae)
            cosine_distances.append(cosine)
            detail_rows.append(
                {
                    "subject_id": subject,
                    "transform": mode,
                    "standardized_feature_mae": mae,
                    "cosine_distance": cosine,
                }
            )
        summary[mode] = {
            "paired_subjects": len(distances),
            "median_standardized_feature_mae": float(np.median(distances)),
            "mean_standardized_feature_mae": float(np.mean(distances)),
            "median_cosine_distance": float(np.median(cosine_distances)),
            "mean_cosine_distance": float(np.mean(cosine_distances)),
        }

    selected = sorted(
        modes,
        key=lambda mode: (
            summary[mode]["median_standardized_feature_mae"],
            summary[mode]["median_cosine_distance"],
        ),
    )[0]
    output = {
        "scope": "paired permanent-training PD subjects only",
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "feature_count": len(feature_columns or []),
        "selection_rule": "lowest median standardized paired-feature MAE",
        "selected_transform": selected,
        "results": summary,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with DETAIL_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    SUMMARY_OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
