from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.run_pipeline import NoiseReductionPipeline
from src.video.processing import process_video

PAGE_TITLE = "AI-Driven Noise Reduction"
PAGE_SUBTITLE = "Upload an image or video to classify the noise type and apply the appropriate denoiser."
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg"}
VIDEO_SUPPORTED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
OUTPUT_DIR = PROJECT_ROOT / "outputs"
WEB_PREVIEW_DIR = OUTPUT_DIR / "web_preview"
IMAGE_NAV_ITEMS = ["Overview", "Comparison View", "Technical Details"]
VIDEO_NAV_ITEMS = ["Overview", "Playback", "Technical Details"]
VIDEO_FORCE_NOISE_OPTIONS = ["auto", "gaussian", "salt_pepper", "speckle", "periodic"]
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".avi": "video/avi",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}


def configure_page() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(20, 184, 166, 0.10), transparent 24%),
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.12), transparent 28%),
                linear-gradient(180deg, #08111f 0%, #0b1324 48%, #0f172a 100%);
            color: #e5e7eb;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2.25rem;
            padding-bottom: 2.5rem;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            background: transparent !important;
        }
        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 0.35rem;
        }
        [data-testid="stFileUploader"] section {
            border: 1px dashed rgba(148, 163, 184, 0.42);
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.35);
        }
        [data-testid="stFileUploader"] button {
            background: #1e293b;
            border: 1px solid rgba(148, 163, 184, 0.28);
            color: #f8fafc !important;
        }
        [data-testid="stFileUploader"] button:hover {
            background: #2563eb;
            border-color: #2563eb;
        }
        [data-testid="stFileUploaderFile"] {
            display: none !important;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 12px;
            font-weight: 700;
            border: 1px solid rgba(59, 130, 246, 0.18);
            box-shadow: 0 12px 24px rgba(2, 6, 23, 0.28);
        }
        .stButton > button {
            background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
            color: #ffffff;
        }
        .stButton > button:hover {
            background: linear-gradient(180deg, #1d4ed8 0%, #1e40af 100%);
            color: #ffffff;
        }
        .stDownloadButton > button {
            background: linear-gradient(180deg, #1e293b 0%, #172033 100%);
            color: #e5e7eb;
        }
        .stDownloadButton > button:hover {
            background: linear-gradient(180deg, #243247 0%, #1a2437 100%);
            color: #f8fafc;
            border-color: rgba(59, 130, 246, 0.30);
        }
        .hero-card, .info-card {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.92) 0%, rgba(15, 23, 42, 0.96) 100%);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 22px;
            box-shadow: 0 24px 48px rgba(2, 6, 23, 0.34);
        }
        .hero-card {
            padding: 1.65rem 1.7rem 1.45rem 1.7rem;
            margin-bottom: 1.2rem;
            overflow: hidden;
        }
        .upload-shell {
            max-width: 960px;
            margin: 0 auto;
        }
        .upload-card {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.92) 0%, rgba(15, 23, 42, 0.97) 100%);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 24px;
            box-shadow: 0 24px 48px rgba(2, 6, 23, 0.34);
            padding: 1.3rem;
        }
        .upload-preview {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.92) 0%, rgba(17, 24, 39, 0.98) 100%);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 20px;
            padding: 1rem;
            height: 100%;
        }
        .info-card {
            padding: 1.15rem 1.15rem 1.05rem 1.15rem;
            height: 100%;
        }
        .hero-layout {
            display: grid;
            grid-template-columns: 1.9fr 1fr;
            gap: 1.15rem;
            align-items: end;
        }
        .hero-title {
            font-size: 2.35rem;
            font-weight: 900;
            color: #f8fafc;
            line-height: 1.05;
            margin-bottom: 0.35rem;
            max-width: 11ch;
        }
        .hero-subtitle {
            color: #cbd5e1;
            font-size: 1.02rem;
            line-height: 1.65;
            margin-bottom: 0;
            max-width: 58ch;
        }
        .hero-side {
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.68) 0%, rgba(15, 23, 42, 0.88) 100%);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem 1rem;
        }
        .hero-side-title {
            color: #f8fafc;
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }
        .hero-side-row {
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
            padding: 0.42rem 0;
            border-top: 1px solid rgba(148, 163, 184, 0.10);
            color: #cbd5e1;
            font-size: 0.92rem;
        }
        .hero-side-row:first-of-type {
            border-top: none;
            padding-top: 0;
        }
        .card-label {
            color: #94a3b8;
            font-size: 0.76rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.5rem;
        }
        .card-value {
            color: #f8fafc;
            font-size: 1.38rem;
            font-weight: 850;
            line-height: 1.2;
            word-break: break-word;
        }
        .card-subtle {
            color: #cbd5e1;
            font-size: 0.92rem;
            margin-top: 0.3rem;
            word-break: break-word;
        }
        .section-title {
            color: #f8fafc;
            font-size: 1.12rem;
            font-weight: 800;
            margin-bottom: 0.65rem;
        }
        .section-kicker {
            display: inline-flex;
            align-items: center;
            padding: 0.28rem 0.6rem;
            border-radius: 999px;
            background: rgba(37, 99, 235, 0.12);
            border: 1px solid rgba(96, 165, 250, 0.22);
            color: #bfdbfe !important;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            line-height: 1;
            margin-bottom: 0.55rem;
            white-space: nowrap;
        }
        .image-card {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.90) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 22px;
            box-shadow: 0 18px 36px rgba(2, 6, 23, 0.28);
            padding: 1rem 1rem 0.95rem 1rem;
        }
        .export-row {
            margin-top: 0.9rem;
            padding-top: 0.9rem;
            border-top: 1px solid rgba(148, 163, 184, 0.12);
        }
        .export-label {
            color: #94a3b8;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
        }
        [data-testid="stImage"] img {
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.18);
        }
        .stInfo, .stAlert {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.94) 0%, rgba(15, 23, 42, 0.98) 100%) !important;
            color: #e5e7eb !important;
            border: 1px solid rgba(148, 163, 184, 0.16) !important;
        }
        [data-testid="stExpander"] {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.94) 0%, rgba(15, 23, 42, 0.98) 100%);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 16px;
        }
        [data-testid="stExpander"] * {
            color: #e5e7eb !important;
        }
        .stMarkdown, .stMarkdown p, .stMarkdown div, .stCaption, label {
            color: #e5e7eb !important;
        }
        .stCodeBlock, pre, code {
            background: rgba(2, 6, 23, 0.42) !important;
            color: #e2e8f0 !important;
            border-radius: 12px;
        }
        code {
            white-space: pre-wrap !important;
            word-break: break-word !important;
        }
        .sidebar-note {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 16px;
            padding: 0.9rem 0.95rem;
            margin: 0.4rem 0 0.8rem 0;
        }
        .sidebar-note-title {
            color: #f8fafc;
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .sidebar-note-body {
            color: #cbd5e1;
            font-size: 0.88rem;
            line-height: 1.55;
        }
        .download-shell {
            margin-top: 0.35rem;
            padding: 1rem 1rem 0.85rem 1rem;
        }
        [data-testid="stRadio"] > div {
            gap: 0.6rem;
        }
        [data-testid="stRadio"] label {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.90) 0%, rgba(15, 23, 42, 0.98) 100%);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 999px;
            padding: 0.45rem 0.95rem;
            min-height: auto !important;
        }
        [data-testid="stRadio"] label:has(input:checked) {
            border-color: rgba(96, 165, 250, 0.35);
            box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.20) inset;
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.98) 0%, rgba(17, 24, 39, 1) 100%);
        }
        [data-testid="stRadio"] label p {
            color: #cbd5e1 !important;
            font-weight: 700;
            font-size: 0.95rem;
        }
        [data-testid="stRadio"] label:has(input:checked) p {
            color: #f8fafc !important;
        }
        .nav-shell {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.92) 0%, rgba(15, 23, 42, 0.97) 100%);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 22px;
            padding: 1rem 1.15rem 0.7rem 1.15rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 18px 36px rgba(2, 6, 23, 0.28);
        }
        .nav-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.6rem;
        }
        .nav-title {
            color: #f8fafc;
            font-size: 1.05rem;
            font-weight: 800;
        }
        .nav-subtitle {
            color: #94a3b8;
            font-size: 0.9rem;
        }
        @media (max-width: 900px) {
            .hero-layout {
                grid-template-columns: 1fr;
            }
            .hero-title {
                max-width: none;
                font-size: 2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_noise_label(label: str) -> str:
    return label.replace("_", " ").title()


def to_display_name(path_value: str) -> str:
    return Path(path_value).name


def get_video_mime_type(filename: str) -> str:
    return VIDEO_MIME_TYPES.get(Path(filename).suffix.lower(), "video/mp4")


def resolve_ffmpeg_executable() -> str:
    ffmpeg_from_env = os.environ.get("FFMPEG_PATH")
    if ffmpeg_from_env:
        candidate = Path(ffmpeg_from_env).expanduser()
        if candidate.exists():
            return str(candidate)

    ffmpeg_from_path = shutil.which("ffmpeg")
    if ffmpeg_from_path:
        return ffmpeg_from_path

    common_candidates = [
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
        Path.home() / "ffmpeg" / "bin" / "ffmpeg.exe",
        Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe"),
    ]
    for candidate in common_candidates:
        if candidate.exists():
            return str(candidate)

    raise RuntimeError(
        "ffmpeg could not be resolved. Install ffmpeg or set FFMPEG_PATH to the full ffmpeg executable path."
    )


def convert_video_to_browser_safe(input_path: Path, output_path: Path) -> Path:
    ffmpeg_executable = resolve_ffmpeg_executable()
    command = [
        ffmpeg_executable,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for browser-compatible video preview and playback.") from exc

    if completed.returncode != 0:
        message = completed.stderr.strip().splitlines()[-1] if completed.stderr else "Unknown ffmpeg error."
        raise RuntimeError(f"ffmpeg conversion failed: {message}")
    return output_path


def ensure_browser_safe_video(source_path: str | Path, output_path: str | Path | None = None) -> Path:
    source = Path(source_path).expanduser().resolve()
    target = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else source.with_name(f"{source.stem}_browser.mp4")
    )
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    return convert_video_to_browser_safe(source, target)


def persist_uploaded_video(uploaded_file: Any) -> tuple[Path, Path]:
    video_bytes = uploaded_file.getvalue()
    video_name = uploaded_file.name
    suffix = Path(video_name).suffix.lower() or ".mp4"
    digest = hashlib.sha1(video_bytes).hexdigest()[:12]
    preview_root = WEB_PREVIEW_DIR / "uploaded_videos"
    preview_root.mkdir(parents=True, exist_ok=True)
    source_path = preview_root / f"{Path(video_name).stem}_{digest}{suffix}"
    preview_path = preview_root / f"{Path(video_name).stem}_{digest}_browser.mp4"

    if not source_path.exists():
        source_path.write_bytes(video_bytes)
    ensure_browser_safe_video(source_path, preview_path)
    return source_path, preview_path


@st.cache_resource(show_spinner=False)
def get_pipeline() -> NoiseReductionPipeline:
    return NoiseReductionPipeline(output_dir=OUTPUT_DIR)


def init_session_state() -> None:
    defaults = {
        "current_page": "upload",
        "active_nav": "Overview",
        "pipeline_result": None,
        "result_kind": "image",
        "upload_mode": "Image",
        "uploaded_name": None,
        "uploaded_bytes": None,
        "uploaded_video_name": None,
        "uploaded_video_bytes": None,
        "uploaded_video_mime": None,
        "uploaded_video_source_path": None,
        "uploaded_video_preview_path": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_header() -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-layout">
                <div>
                    <div class="hero-title">{PAGE_TITLE}</div>
                    <p class="hero-subtitle">{PAGE_SUBTITLE}</p>
                </div>
                <div class="hero-side">
                    <div class="hero-side-title">Pipeline Overview</div>
                    <div class="hero-side-row"><span>Stage 1</span><strong>Noise Classification</strong></div>
                    <div class="hero-side-row"><span>Stage 2</span><strong>Model Routing</strong></div>
                    <div class="hero-side-row"><span>Stage 3</span><strong>Image Denoising</strong></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def validate_uploaded_image(uploaded_file: Any) -> Image.Image:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file format: {suffix}")
    return Image.open(uploaded_file).convert("RGB")


def save_uploaded_file(image: Image.Image, original_name: str) -> Path:
    suffix = Path(original_name).suffix.lower()
    with tempfile.NamedTemporaryFile(
        prefix=f"{Path(original_name).stem}_",
        suffix=suffix,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    image.save(temp_path)
    return temp_path


def validate_uploaded_video(uploaded_file: Any) -> None:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in VIDEO_SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported video format: {suffix}")


def save_uploaded_video_file(uploaded_file: Any) -> Path:
    validate_uploaded_video(uploaded_file)
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(
        prefix=f"{Path(uploaded_file.name).stem}_",
        suffix=suffix,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(uploaded_file.getvalue())
    return temp_path


def run_pipeline_for_upload(uploaded_file: Any) -> dict[str, Any]:
    image = validate_uploaded_image(uploaded_file)
    temp_path = save_uploaded_file(image, uploaded_file.name)
    started_at = perf_counter()
    try:
        result = get_pipeline().run(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)
    result["processing_time_seconds"] = perf_counter() - started_at
    result["uploaded_name"] = uploaded_file.name
    result["original_image"] = image
    return result


def run_video_pipeline_for_upload(
    uploaded_file: Any,
    force_noise_type: str | None,
    passes: int,
    progress_callback: Any | None = None,
    log_callback: Any | None = None,
) -> dict[str, Any]:
    temp_input_path = save_uploaded_video_file(uploaded_file)
    output_dir = OUTPUT_DIR / "video"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(uploaded_file.name).stem}_denoised.mp4"
    started_at = perf_counter()
    try:
        result = process_video(
            video_path=temp_input_path,
            output_path=output_path,
            force_noise_type=force_noise_type,
            passes=passes,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )
    finally:
        temp_input_path.unlink(missing_ok=True)

    result_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(result).items()
    }
    browser_output_path = ensure_browser_safe_video(result.output_video_path)
    result_dict["browser_output_video_path"] = str(browser_output_path)
    result_dict["processing_time_seconds"] = perf_counter() - started_at
    result_dict["uploaded_name"] = uploaded_file.name
    result_dict["predicted_noise_type"] = force_noise_type or "auto"
    result_dict["classifier_confidence"] = None
    if force_noise_type is None:
        result_dict["denoiser_model_type"] = "Automatic per-frame routing"
        result_dict["denoiser_checkpoint"] = "Selected independently for each frame"
    else:
        checkpoint_path = NoiseReductionPipeline.DENOISER_CHECKPOINTS.get(force_noise_type)
        result_dict["denoiser_model_type"] = f"{format_noise_label(force_noise_type)} route"
        result_dict["denoiser_checkpoint"] = str(checkpoint_path) if checkpoint_path is not None else "Unknown"
    result_dict["work_root"] = str(output_dir)
    return result_dict


def load_denoised_image(result: dict[str, Any]) -> Image.Image:
    output_path = Path(result["output_path"])
    if not output_path.exists():
        raise FileNotFoundError(f"Denoised output not found: {output_path}")
    return Image.open(output_path).convert("RGB")


def serialize_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result.items()
        if key not in {"original_image", "denoised_image", "denoised_bytes", "video_bytes"}
    }


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def read_binary_file(path_value: str | Path) -> bytes:
    return Path(path_value).read_bytes()


def render_html_video(video_path: str | Path, mime_type: str) -> None:
    video_bytes = read_binary_file(video_path)
    encoded = base64.b64encode(video_bytes).decode("ascii")
    components.html(
        f"""
        <video controls preload="metadata" style="width: 100%; border-radius: 16px; border: 1px solid rgba(148, 163, 184, 0.18); background: #000;">
            <source src="data:{mime_type};base64,{encoded}" type="{mime_type}">
            Your browser does not support the video tag.
        </video>
        """,
        height=420,
    )


def build_comparison_image(original: Image.Image, denoised: Image.Image, reveal_ratio: float) -> Image.Image:
    original_rgb = original.convert("RGB")
    denoised_rgb = denoised.convert("RGB").resize(original_rgb.size)
    split_x = int(original_rgb.width * reveal_ratio)
    comparison = original_rgb.copy()
    if split_x < original_rgb.width:
        comparison.paste(denoised_rgb.crop((split_x, 0, original_rgb.width, original_rgb.height)), (split_x, 0))
    return comparison


def render_info_cards(result: dict[str, Any]) -> None:
    confidence_pct = float(result["classifier_confidence"]) * 100.0
    processing_time = float(result["processing_time_seconds"])
    columns = st.columns(4, gap="small")
    entries = [
        ("Predicted Noise", format_noise_label(str(result["predicted_noise_type"]))),
        ("Classifier Confidence", f"{confidence_pct:.2f}%"),
        ("Selected Model", str(result["denoiser_model_type"])),
        ("Processing Time", f"{processing_time:.2f}s"),
    ]
    for column, (label, value) in zip(columns, entries):
        column.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">{label}</div>
                <div class="card-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-label">Selected Denoiser Checkpoint</div>
            <div class="card-value">{to_display_name(str(result["denoiser_path"]))}</div>
            <div class="card-subtle">{result["denoiser_path"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_image_panel(result: dict[str, Any], show_download: bool = False) -> None:
    original_col, denoised_col = st.columns(2, gap="large")
    with original_col:
        st.markdown(
            """
            <div class="section-kicker">Input</div>
            <div class="section-title">Original</div>
            """,
            unsafe_allow_html=True,
        )
        st.image(result["original_image"], use_container_width=True)
    with denoised_col:
        st.markdown(
            """
            <div class="section-kicker">Output</div>
            <div class="section-title">Denoised</div>
            """,
            unsafe_allow_html=True,
        )
        st.image(result["denoised_image"], use_container_width=True)
        if show_download:
            st.markdown('<div class="export-row"><div class="export-label">Export</div></div>', unsafe_allow_html=True)
            render_download(result, compact=True)


def render_comparison_panel(result: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="image-card">
            <div class="section-kicker">Comparison View</div>
            <div class="section-title">Before / After Reveal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    reveal_percent = st.slider(
        "Reveal denoised region",
        min_value=0,
        max_value=100,
        value=50,
        help="Left side stays original, right side reveals the denoised output.",
    )
    comparison_image = build_comparison_image(
        result["original_image"],
        result["denoised_image"],
        reveal_percent / 100.0,
    )
    st.image(comparison_image, use_container_width=True)
    st.caption("Left: original image. Right: denoised image.")


def render_download(result: dict[str, Any], compact: bool = False) -> None:
    download_name = f"{Path(result['uploaded_name']).stem}_denoised.png"
    if not compact:
        st.markdown(
            """
            <div class="section-kicker">Export</div>
            <div class="section-title">Download Result</div>
            """,
            unsafe_allow_html=True,
        )
    st.download_button(
        "Download Denoised Image",
        data=result["denoised_bytes"],
        file_name=download_name,
        mime="image/png",
        use_container_width=compact,
    )


def render_technical_details(result: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="info-card">
            <div class="section-kicker">Technical Details</div>
            <div class="section-title">Pipeline Metadata</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(f"Selected checkpoint path: `{result['denoiser_path']}`")
    st.write(f"Selected model type: `{result['denoiser_model_type']}`")
    st.write(f"Output path: `{result['output_path']}`")
    st.code(json.dumps(serialize_result(result), indent=2), language="json")


def render_video_info_cards(result: dict[str, Any]) -> None:
    processing_time = float(result["processing_time_seconds"])
    confidence = result.get("classifier_confidence")
    confidence_value = "Forced route" if confidence is None else f"{float(confidence) * 100.0:.2f}%"
    predicted_noise_type = str(result.get("predicted_noise_type") or result.get("force_noise_type") or "auto")
    predicted_label = "Automatic Per-Frame" if predicted_noise_type == "auto" else format_noise_label(predicted_noise_type)
    model_type = str(result.get("denoiser_model_type") or "Automatic per-frame routing")
    columns = st.columns(4, gap="small")
    entries = [
        ("Predicted Noise", predicted_label),
        ("Classifier Confidence", confidence_value),
        ("Selected Model", model_type),
        ("Processing Time", f"{processing_time:.2f}s"),
    ]
    for column, (label, value) in zip(columns, entries):
        column.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">{label}</div>
                <div class="card-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    summary_columns = st.columns(3, gap="small")
    secondary_entries = [
        ("Video FPS", f"{float(result['fps']):.2f}"),
        ("Processed Frames", str(result["processed_frame_count"])),
        ("Denoising Passes", str(result["passes"])),
    ]
    for column, (label, value) in zip(summary_columns, secondary_entries):
        column.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">{label}</div>
                <div class="card-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    denoiser_checkpoint = str(result.get("denoiser_checkpoint") or "Selected independently for each frame")
    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-label">Selected Denoiser Checkpoint</div>
            <div class="card-value">{to_display_name(denoiser_checkpoint)}</div>
            <div class="card-subtle">{denoiser_checkpoint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_video_playback(result: dict[str, Any]) -> None:
    video_source_path = result.get("browser_output_video_path") or result["output_video_path"]
    video_mime = get_video_mime_type(video_source_path)
    st.markdown(
        """
        <div class="image-card">
            <div class="section-kicker">Playback</div>
            <div class="section-title">Denoised Video</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_html_video(video_source_path, video_mime)
    st.markdown('<div class="export-row"><div class="export-label">Export</div></div>', unsafe_allow_html=True)
    st.download_button(
        "Download Denoised Video",
        data=read_binary_file(video_source_path),
        file_name=f"{Path(result['uploaded_name']).stem}_denoised.mp4",
        mime=video_mime,
        use_container_width=True,
    )


def render_video_technical_details(result: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="info-card">
            <div class="section-kicker">Technical Details</div>
            <div class="section-title">Video Pipeline Metadata</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(f"Input video path: `{result['input_video_path']}`")
    st.write(f"Output video path: `{result['output_video_path']}`")
    if result.get("browser_output_video_path"):
        st.write(f"Browser playback path: `{result['browser_output_video_path']}`")
    st.write(f"Selected checkpoint path: `{result.get('denoiser_checkpoint', 'Selected independently for each frame')}`")
    st.write(f"Selected model type: `{result.get('denoiser_model_type', 'Automatic per-frame routing')}`")
    st.write(f"Frame workspace: `{result.get('work_root', Path(result['output_video_path']).parent)}`")
    st.code(json.dumps(serialize_result(result), indent=2), language="json")


def render_idle_state() -> None:
    st.info("Upload an image and click `Run Pipeline` to start the demo.")


def store_uploaded_file(uploaded_file: Any) -> None:
    st.session_state.uploaded_name = uploaded_file.name
    st.session_state.uploaded_bytes = uploaded_file.getvalue()


def store_uploaded_video(uploaded_file: Any) -> None:
    video_name = uploaded_file.name
    video_bytes = uploaded_file.getvalue()
    changed = (
        st.session_state.get("uploaded_video_name") != video_name
        or st.session_state.get("uploaded_video_bytes") != video_bytes
    )
    if changed:
        source_path, preview_path = persist_uploaded_video(uploaded_file)
        st.session_state.uploaded_video_source_path = str(source_path)
        st.session_state.uploaded_video_preview_path = str(preview_path)
    st.session_state.uploaded_video_name = video_name
    st.session_state.uploaded_video_bytes = video_bytes
    st.session_state.uploaded_video_mime = getattr(uploaded_file, "type", None) or get_video_mime_type(video_name)


def get_uploaded_preview() -> Image.Image | None:
    uploaded_bytes = st.session_state.get("uploaded_bytes")
    if not uploaded_bytes:
        return None
    return Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")


def get_uploaded_video_bytes() -> bytes | None:
    uploaded_bytes = st.session_state.get("uploaded_video_bytes")
    if not uploaded_bytes:
        return None
    return bytes(uploaded_bytes)


def ensure_uploaded_video_preview_path() -> Path | None:
    existing_path = st.session_state.get("uploaded_video_preview_path")
    if existing_path:
        preview_path = Path(existing_path)
        if preview_path.exists():
            return preview_path

    uploaded_source_path = st.session_state.get("uploaded_video_source_path")
    uploaded_name = st.session_state.get("uploaded_video_name")
    if not uploaded_source_path or not uploaded_name:
        return None

    source_path = Path(uploaded_source_path)
    if not source_path.exists():
        return None

    preview_path = ensure_browser_safe_video(source_path)
    st.session_state.uploaded_video_preview_path = str(preview_path)
    return preview_path


def go_to_upload() -> None:
    st.session_state.current_page = "upload"
    st.session_state.active_nav = "Overview"


def render_upload_screen() -> None:
    st.markdown('<div class="upload-shell">', unsafe_allow_html=True)
    render_header()
    st.radio(
        "Workflow",
        options=["Image", "Video"],
        horizontal=True,
        key="upload_mode",
        label_visibility="collapsed",
    )
    left_col, right_col = st.columns([1.1, 1], gap="large")
    if st.session_state.upload_mode == "Image":
        with left_col:
            st.markdown(
                """
                <div class="upload-card">
                    <div class="section-kicker">Start</div>
                    <div class="section-title">Upload an image and launch the AI cleaning pipeline</div>
                    <div class="card-subtle">The uploaded image will be classified, routed to the active denoiser checkpoint, and returned as a cleaned output.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")
            uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], key="upload_input")
            if uploaded_file is not None:
                store_uploaded_file(uploaded_file)
                st.caption(f"Name: {uploaded_file.name}")
                st.caption(f"Size: {uploaded_file.size / 1024:.1f} KB")
            start_cleaning = st.button("Start Cleaning", type="primary", use_container_width=True)
            if start_cleaning:
                if uploaded_file is None and st.session_state.get("uploaded_bytes") is None:
                    st.warning("Please upload an image before starting the pipeline.")
                else:
                    try:
                        active_file = uploaded_file
                        if active_file is None and st.session_state.get("uploaded_bytes") is not None:
                            active_file = io.BytesIO(st.session_state["uploaded_bytes"])
                            active_file.name = st.session_state["uploaded_name"]
                        with st.spinner("Classifying noise and denoising image..."):
                            result = run_pipeline_for_upload(active_file)
                            denoised_image = load_denoised_image(result)
                            result["denoised_image"] = denoised_image
                            result["denoised_bytes"] = image_to_png_bytes(denoised_image)
                            st.session_state.pipeline_result = result
                            st.session_state.result_kind = "image"
                            st.session_state.current_page = "results"
                            st.session_state.active_nav = "Overview"
                            st.rerun()
                    except UnidentifiedImageError:
                        st.error("The uploaded file could not be read as a valid image.")
                    except FileNotFoundError as exc:
                        st.error(f"Required file not found: {exc}")
                    except Exception as exc:
                        st.error(f"Pipeline execution failed: {exc}")
        with right_col:
            st.markdown('<div class="upload-preview">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Uploaded image preview</div>', unsafe_allow_html=True)
            preview_image = get_uploaded_preview()
            if preview_image is None:
                st.info("Your uploaded image preview will appear here.")
            else:
                st.image(preview_image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        with left_col:
            st.markdown(
                """
                <div class="upload-card">
                    <div class="section-kicker">Start</div>
                    <div class="section-title">Upload a video and run frame-by-frame denoising</div>
                    <div class="card-subtle">The video pipeline extracts frames, resolves one noise route, denoises each frame with the active checkpoint, and rebuilds the final video at the original FPS.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")
            uploaded_video = st.file_uploader(
                "Upload video",
                type=[suffix.lstrip(".") for suffix in sorted(VIDEO_SUPPORTED_SUFFIXES)],
                key="upload_video_input",
            )
            if uploaded_video is not None:
                store_uploaded_video(uploaded_video)
                st.caption(f"Name: {uploaded_video.name}")
                st.caption(f"Size: {uploaded_video.size / (1024 * 1024):.2f} MB")
            force_noise_type_label = st.selectbox(
                "Noise routing",
                options=VIDEO_FORCE_NOISE_OPTIONS,
                format_func=lambda value: "Auto-detect" if value == "auto" else format_noise_label(value),
            )
            passes = st.number_input("Denoising passes", min_value=1, max_value=3, value=1, step=1)
            progress_bar = st.empty()
            status_text = st.empty()
            start_video_cleaning = st.button("Start Video Cleaning", type="primary", use_container_width=True)
            if start_video_cleaning:
                if uploaded_video is None and st.session_state.get("uploaded_video_bytes") is None:
                    st.warning("Please upload a video before starting the pipeline.")
                else:
                    try:
                        active_file = uploaded_video
                        if active_file is None and st.session_state.get("uploaded_video_bytes") is not None:
                            active_file = io.BytesIO(st.session_state["uploaded_video_bytes"])
                            active_file.name = st.session_state["uploaded_video_name"]

                        def on_progress(stage: str, current: int, total: int) -> None:
                            stage_ranges = {"extract": (0.0, 0.18), "denoise": (0.18, 0.92), "rebuild": (0.92, 1.0)}
                            start_ratio, end_ratio = stage_ranges.get(stage, (0.0, 1.0))
                            safe_total = max(total, 1)
                            progress_value = start_ratio + (end_ratio - start_ratio) * (current / safe_total)
                            progress_bar.progress(min(progress_value, 1.0))
                            stage_labels = {
                                "extract": "Extracting frames",
                                "denoise": "Denoising frames",
                                "rebuild": "Rebuilding video",
                            }
                            status_text.caption(f"{stage_labels.get(stage, stage.title())}: {current}/{total}")

                        with st.spinner("Running video denoising pipeline..."):
                            result = run_video_pipeline_for_upload(
                                active_file,
                                None if force_noise_type_label == "auto" else force_noise_type_label,
                                int(passes),
                                progress_callback=on_progress,
                            )
                            progress_bar.progress(1.0)
                            status_text.caption("Video processing completed.")
                            st.session_state.pipeline_result = result
                            st.session_state.result_kind = "video"
                            st.session_state.current_page = "results"
                            st.session_state.active_nav = "Overview"
                            st.rerun()
                    except FileNotFoundError as exc:
                        st.error(f"Required file not found: {exc}")
                    except ValueError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"Video pipeline execution failed: {exc}")
        with right_col:
            st.markdown('<div class="upload-preview">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Uploaded video preview</div>', unsafe_allow_html=True)
            preview_path = None
            preview_error = None
            if get_uploaded_video_bytes() is not None:
                try:
                    preview_path = ensure_uploaded_video_preview_path()
                except Exception as exc:
                    preview_error = str(exc)
            if preview_path is None and preview_error is None:
                st.info("Your uploaded video preview will appear here.")
            elif preview_error is not None:
                st.error(f"Video preview could not be prepared: {preview_error}")
            else:
                render_html_video(preview_path, "video/mp4")
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_results_nav() -> str:
    result = st.session_state.pipeline_result
    result_kind = st.session_state.get("result_kind", "image")
    nav_items = IMAGE_NAV_ITEMS if result_kind == "image" else VIDEO_NAV_ITEMS
    if result_kind == "image":
        subtitle = f"{format_noise_label(str(result.get('predicted_noise_type', 'unknown')))} route completed successfully"
    else:
        predicted = str(result.get("predicted_noise_type") or result.get("force_noise_type") or "auto")
        route_label = "Automatic video routing" if predicted == "auto" else f"{format_noise_label(predicted)} video route"
        subtitle = f"{route_label} completed successfully"
    st.markdown(
        f"""
        <div class="nav-shell">
            <div class="nav-header">
                <div>
                    <div class="nav-title">Results Dashboard</div>
                    <div class="nav-subtitle">{subtitle}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([4, 1], gap="medium")
    with left:
        active_nav = st.radio(
            "Navigation",
            options=nav_items,
            horizontal=True,
            label_visibility="collapsed",
            key="active_nav",
        )
    with right:
        st.button("New Upload", on_click=go_to_upload, use_container_width=True)
    return active_nav


def render_results_screen() -> None:
    result = st.session_state.pipeline_result
    if result is None:
        st.error("No pipeline result is available. Please upload an image first.")
        if st.button("Back to Upload", use_container_width=False):
            go_to_upload()
            st.rerun()
        return
    active_nav = render_results_nav()
    result_kind = st.session_state.get("result_kind", "image")
    if result_kind == "image":
        if active_nav == "Overview":
            render_info_cards(result)
            st.markdown("")
            render_image_panel(result, show_download=True)
        elif active_nav == "Comparison View":
            render_comparison_panel(result)
        elif active_nav == "Technical Details":
            render_technical_details(result)
    else:
        if active_nav == "Overview":
            render_video_info_cards(result)
            st.markdown("")
            render_video_playback(result)
        elif active_nav == "Playback":
            render_video_playback(result)
        elif active_nav == "Technical Details":
            render_video_technical_details(result)


def main() -> None:
    configure_page()
    init_session_state()
    if st.session_state.current_page == "upload":
        render_upload_screen()
    else:
        render_results_screen()


if __name__ == "__main__":
    main()
