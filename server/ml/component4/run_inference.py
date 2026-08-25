import argparse
import json
from pathlib import Path

import cv2

from src.component4_predictor import inspect_model_bundle, predict_video


PROJECT_ROOT = Path(__file__).resolve().parent
POSE_MODEL_PATH = PROJECT_ROOT / "mediapipe_models" / "pose_landmarker.task"
OUTPUT_DIR = PROJECT_ROOT / "05_outputs" / "annotated_videos"

EXERCISES = {
    "gesture3": {
        "label": "Seated left-arm forward raise",
        "model": PROJECT_ROOT / "models" / "final_gesture3_chair_kinect_video_model.joblib",
    },
    "gesture5": {
        "label": "Seated left-arm lateral raise",
        "model": PROJECT_ROOT / "models" / "final_gesture5_chair_kinect_video_model.joblib",
    },
    "gesture2": {
        "label": "Seated right-arm forward raise",
        "model": PROJECT_ROOT / "models" / "final_gesture2_chair_kinect_video_model.joblib",
    },
}


def video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            return 30.0
        return float(fps)
    finally:
        cap.release()


def window_starts(valid_pose_frames: int, window_frames: int, stride_frames: int) -> list[int]:
    if valid_pose_frames <= 0:
        return []
    if valid_pose_frames < window_frames:
        return [0]

    last_start = valid_pose_frames - window_frames
    starts = list(range(0, last_start + 1, stride_frames))
    if starts and starts[-1] != last_start:
        starts.append(last_start)
    return starts


def merge_segments(windows: list[dict]) -> list[dict]:
    if not windows:
        return []

    merged = []
    for item in windows:
        if not merged or item["start_seconds"] > merged[-1]["end_seconds"] + 0.001:
            merged.append({
                "start_seconds": item["start_seconds"],
                "end_seconds": item["end_seconds"],
                "window_count": 1,
                "mean_correct_probability": item["correct_probability"],
            })
            continue

        previous = merged[-1]
        total = previous["window_count"] + 1
        previous["end_seconds"] = max(previous["end_seconds"], item["end_seconds"])
        previous["mean_correct_probability"] = round(
            ((previous["mean_correct_probability"] * previous["window_count"]) + item["correct_probability"]) / total,
            4,
        )
        previous["window_count"] = total

    return merged


def build_window_report(result: dict, model_info: dict, source_fps: float) -> dict:
    predictions = result.get("window_predictions") or []
    probabilities = result.get("window_correct_probabilities") or []
    window_frames = int(model_info.get("window_frames") or 60)
    stride_frames = int(model_info.get("stride_frames") or 15)
    timing_fps = float(model_info.get("target_fps") or source_fps or 30.0)
    starts = window_starts(int(result.get("valid_pose_frames") or 0), window_frames, stride_frames)

    windows = []
    for index, prediction in enumerate(predictions):
        start = starts[index] if index < len(starts) else index * stride_frames
        end = start + window_frames
        probability = probabilities[index] if index < len(probabilities) else None
        label = "Correct" if int(prediction) == 1 else "Incorrect"
        windows.append({
            "window_id": index,
            "prediction": label,
            "correct_probability": round(float(probability), 4) if probability is not None else None,
            "start_seconds": round(start / timing_fps, 2),
            "end_seconds": round(end / timing_fps, 2),
        })

    correct_windows = [item for item in windows if item["prediction"] == "Correct"]
    incorrect_windows = [item for item in windows if item["prediction"] == "Incorrect"]

    return {
        "timing_fps": round(timing_fps, 3),
        "window_frames": window_frames,
        "stride_frames": stride_frames,
        "windows": windows,
        "correct_windows": correct_windows,
        "incorrect_windows": incorrect_windows,
        "correct_segments": merge_segments(correct_windows),
        "incorrect_segments": merge_segments(incorrect_windows),
        "perfect_report": len(windows) > 0 and len(incorrect_windows) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", choices=sorted(EXERCISES.keys()), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--save-annotated", action="store_true")
    args = parser.parse_args()

    exercise = EXERCISES[args.exercise]
    video_path = Path(args.input).resolve()
    model_path = exercise["model"]

    annotated_path = None
    if args.save_annotated:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in video_path.stem)
        annotated_path = OUTPUT_DIR / f"{safe_stem}_{args.exercise}_annotated.webm"

    try:
        model_info = inspect_model_bundle(str(model_path))
        result = predict_video(
            video_path=str(video_path),
            model_path=str(model_path),
            pose_task_model=str(POSE_MODEL_PATH),
            save_annotated=str(annotated_path) if annotated_path else None,
        )
        source_fps = video_fps(str(video_path))
        window_report = build_window_report(result, model_info, source_fps)
        result.update({
            "exercise_key": args.exercise,
            "exercise_label": exercise["label"],
            "model_info": model_info,
            "source_fps": round(source_fps, 3),
            "window_report": window_report,
        })
        print(json.dumps({"ok": True, "result": result}))
    except Exception as error:
        print(json.dumps({"ok": False, "message": str(error)}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
