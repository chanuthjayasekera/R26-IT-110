from pathlib import Path
import subprocess
import sys
import shutil

import pandas as pd
import streamlit as st

from src.koa_predictor import (
    find_latest_inference_csv,
    predict_component2_koa,
)


PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_DIR = PROJECT_ROOT / "01_input_videos" / "inference"
CSV_DIR = PROJECT_ROOT / "02_extracted_csv" / "training_safe" / "inference"
REPORT_DIR = PROJECT_ROOT / "07_reports" / "component_2_koa" / "inference"

EXTRACTOR_CANDIDATES = [
    PROJECT_ROOT / "04_video_extraction" / "run_component2_koa_clinical_video.py",
    PROJECT_ROOT / "run_component2_koa_clinical_video.py",
]

VIDEO_EXTENSIONS = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV", "*.AVI", "*.MKV"]

st.set_page_config(
    page_title="Component 2B KOA Gait Screening",
    page_icon="🦵",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 2.35rem;
        font-weight: 850;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.02rem;
        margin-bottom: 1.2rem;
    }
    .big-result {
        padding: 1.35rem;
        border-radius: 12px;
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 18px rgba(0,0,0,0.05);
        margin: 0.7rem 0 1rem 0;
    }
    .small-muted {
        color: #64748b;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.04rem;
        margin-bottom: 0.25rem;
    }
    .big-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #111827;
    }
    .result-card {
        padding: 1.1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        box-shadow: 0 3px 14px rgba(0,0,0,0.04);
        min-height: 110px;
    }
    .card-label {
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04rem;
        font-size: 0.78rem;
        margin-bottom: 0.35rem;
    }
    .card-value {
        font-size: 1.35rem;
        font-weight: 780;
        color: #111827;
        overflow-wrap: anywhere;
    }
    .section-title {
        font-size: 1.18rem;
        font-weight: 780;
        margin-top: 1.1rem;
        margin-bottom: 0.55rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown('<div class="main-title">Component 2B KOA Gait Screening Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a walking video or training-safe landmarks CSV to run the KOA V14 youtube-ataxia-once screening model.</div>',
    unsafe_allow_html=True,
)


def clean_folder(folder: Path, patterns: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for item in folder.glob(pattern):
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception:
                pass


def find_extractor_script() -> Path | None:
    for p in EXTRACTOR_CANDIDATES:
        if p.exists():
            return p
    return None


def result_icon(result: dict) -> str:
    reliability = str(result.get("reliability_level", "")).lower()
    if reliability == "low":
        return "Low reliability"
    if result.get("koa_detected") is True:
        return "Confirmed KOA-like gait pattern detected"
    return "No confirmed KOA-like gait pattern detected"


def card(label: str, value) -> None:
    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_result(result: dict) -> None:
    final_text = result.get("final_koa_result", result_icon(result))
    clinical_note = result.get("clinical_note", "")

    st.markdown("## Screening Result")
    st.markdown(
        f"""
        <div class="big-result">
            <div class="small-muted">Final Component 2B KOA Decision</div>
            <div class="big-value">{final_text}</div>
            <div style="margin-top:0.55rem; color:#4b5563;">{clinical_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    koa_probability = float(result.get("koa_probability", 0.0))
    max_probability = float(result.get("koa_max_probability", 0.0))
    positive_ratio = float(result.get("koa_window_ratio_detection", 0.0))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("KOA Detected", result.get("koa_detected", "N/A"))
    with c2:
        card("Reliability", result.get("reliability_level", "N/A"))
    with c3:
        card("Mean Probability", f"{koa_probability:.4f}")
    with c4:
        card("Pattern Strength", result.get("koa_pattern_strength", "N/A"))

    st.markdown('<div class="section-title">KOA Probability Gauge</div>', unsafe_allow_html=True)
    st.progress(max(0.0, min(1.0, koa_probability)))
    setting = result.get("koa_pattern_setting", {}) or {}
    st.caption(
        "Window threshold: "
        f"{setting.get('window_threshold', 'N/A')} | "
        f"Positive window ratio: {positive_ratio:.4f} | "
        f"Max probability: {max_probability:.4f}"
    )

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">Model Output</div>', unsafe_allow_html=True)
        model_rows = {
            "Model version": result.get("model_version", "N/A"),
            "Final KOA result": result.get("final_koa_result", "N/A"),
            "KOA detected": result.get("koa_detected", "N/A"),
            "KOA borderline tendency": result.get("koa_borderline_tendency", "N/A"),
            "Mean probability": result.get("koa_probability", "N/A"),
            "Max probability": result.get("koa_max_probability", "N/A"),
            "Positive count": result.get("koa_positive_window_count", "N/A"),
            "Detection ratio": result.get("koa_window_ratio_detection", "N/A"),
            "Pattern setting": result.get("koa_pattern_setting", "N/A"),
        }
        st.table(pd.DataFrame(model_rows.items(), columns=["Metric", "Value"]))

    with right:
        st.markdown('<div class="section-title">Video Quality</div>', unsafe_allow_html=True)
        quality_rows = {
            "CSV file": Path(str(result.get("csv_file", ""))).name if result.get("csv_file") else "N/A",
            "Direction": result.get("direction", "N/A"),
            "Direction confidence": result.get("direction_confidence", "N/A"),
            "Direction delta x": result.get("direction_delta_x", "N/A"),
            "FPS used": result.get("fps_used", "N/A"),
            "Clean duration": f'{result.get("clean_duration_sec", "N/A")}s',
            "Total windows": result.get("total_windows", "N/A"),
            "Padded": result.get("padded", "N/A"),
            "Resampled frames": result.get("resampled_frames", "N/A"),
        }
        st.table(pd.DataFrame(quality_rows.items(), columns=["Metric", "Value"]))

    st.markdown('<div class="section-title">Reliability Notes</div>', unsafe_allow_html=True)
    reasons = result.get("reliability_reasons", [])
    if reasons:
        for reason in reasons:
            st.info(reason)
    else:
        st.info("No reliability notes returned.")

    with st.expander("Full JSON result"):
        st.json(result)


def run_predict_on_csv(csv_path: Path, direction: str | None = None, fps: float | None = None) -> dict:
    with st.spinner("Running Component 2B KOA V14 model..."):
        result = predict_component2_koa(
            csv_path=csv_path,
            direction=direction,
            fps_used=fps,
        )

    show_result(result)
    return result


with st.sidebar:
    st.header("Input Settings")
    st.info("KOA V14 can infer walking direction and FPS automatically, but you can override them manually here.")

    manual_direction_choice = st.selectbox(
        "Walking direction",
        ["Auto", "L2R", "R2L"],
        index=0,
        help="Use Auto unless you know the subject walked left-to-right or right-to-left.",
    )

    manual_fps_enabled = st.checkbox(
        "Set FPS manually",
        value=False,
        help="Enable this when the uploaded video/CSV FPS is known and filename inference is not enough.",
    )

    manual_fps_value = st.number_input(
        "Manual FPS",
        min_value=1.0,
        max_value=240.0,
        value=30.0,
        step=1.0,
        disabled=not manual_fps_enabled,
    )

    selected_direction = None if manual_direction_choice == "Auto" else manual_direction_choice
    selected_fps = float(manual_fps_value) if manual_fps_enabled else None


tab_video, tab_csv = st.tabs(["Upload Video", "Upload CSV"])


with tab_video:
    uploaded_video = st.file_uploader(
        "Upload a walking video",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=False,
    )

    if uploaded_video:
        clean_folder(VIDEO_DIR, VIDEO_EXTENSIONS)
        clean_folder(CSV_DIR, ["*_training_safe_landmarks.csv"])
        clean_folder(REPORT_DIR, ["*.csv"])

        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        video_path = VIDEO_DIR / uploaded_video.name
        video_path.write_bytes(uploaded_video.getbuffer())

        st.video(str(video_path))
        st.success(f"Video ready: {video_path.name}")

        if st.button("Extract landmarks and predict", type="primary"):
            extractor = find_extractor_script()

            if extractor is None:
                st.error("Extractor script not found.")
            else:
                before = set(CSV_DIR.glob("*_training_safe_landmarks.csv"))
                cmd = [sys.executable, str(extractor), str(video_path)]

                with st.spinner("Extracting pose landmarks..."):
                    completed = subprocess.run(
                        cmd,
                        cwd=str(PROJECT_ROOT),
                        capture_output=True,
                        text=True,
                    )

                if completed.returncode != 0:
                    st.error("Landmark extraction failed.")
                    st.code(completed.stderr or completed.stdout)
                else:
                    after = set(CSV_DIR.glob("*_training_safe_landmarks.csv"))
                    new_files = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)

                    try:
                        csv_path = new_files[0] if new_files else find_latest_inference_csv()
                        st.success(f"Landmarks created: {csv_path.name}")
                        _ = run_predict_on_csv(csv_path, direction=selected_direction, fps=selected_fps)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        if completed.stdout:
                            with st.expander("Extractor output"):
                                st.code(completed.stdout)


with tab_csv:
    st.write("Use this tab when you already have a `*_training_safe_landmarks.csv` file.")

    uploaded_csv = st.file_uploader(
        "Upload training-safe landmarks CSV",
        type=["csv"],
        accept_multiple_files=False,
    )

    if uploaded_csv:
        clean_folder(CSV_DIR, ["*_training_safe_landmarks.csv"])
        clean_folder(REPORT_DIR, ["*.csv"])
        CSV_DIR.mkdir(parents=True, exist_ok=True)

        csv_path = CSV_DIR / uploaded_csv.name
        csv_path.write_bytes(uploaded_csv.getbuffer())

        preview = pd.read_csv(csv_path, nrows=5)
        st.write("CSV preview")
        st.dataframe(preview, use_container_width=True)

        if st.button("Predict from CSV", type="primary"):
            try:
                _ = run_predict_on_csv(csv_path, direction=selected_direction, fps=selected_fps)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
