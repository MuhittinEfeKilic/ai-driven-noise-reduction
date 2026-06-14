from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from src.pipeline.run_pipeline import NoiseReductionPipeline


ProgressCallback = Callable[[str, int, int], None]
LogCallback = Callable[[str], None]


@dataclass(frozen=True)
class VideoProcessingResult:
    input_video_path: Path
    output_video_path: Path
    fps: float
    width: int
    height: int
    frame_count: int
    processed_frame_count: int
    force_noise_type: str | None
    passes: int


def _open_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    return capture


def _open_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Failed to open video writer: {output_path}")
    return writer


def _call_progress(callback: ProgressCallback | None, stage: str, current: int, total: int) -> None:
    if callback is not None:
        callback(stage, current, total)


def _call_log(callback: LogCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _bgr_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _pil_to_bgr(image: Image.Image, width: int, height: int) -> np.ndarray:
    frame = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


def process_video(
    video_path: str | Path,
    output_path: str | Path,
    force_noise_type: str | None = None,
    passes: int = 1,
    progress_callback: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
) -> VideoProcessingResult:
    """Denoise a video frame-by-frame with the image denoising pipeline."""
    input_path = Path(video_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if passes < 1:
        raise ValueError("passes must be at least 1.")

    capture = _open_capture(input_path)
    writer: cv2.VideoWriter | None = None
    processed_frames = 0

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 0.0 else 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = _open_writer(destination, fps, width, height)

        pipeline = NoiseReductionPipeline(output_dir=destination.parent)
        classifier = None if force_noise_type else pipeline.get_classifier()
        denoiser_cache: dict[str, object] = {}

        _call_log(log_callback, f"Opened video: {input_path}")
        _call_progress(progress_callback, "extract", frame_count, frame_count)

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            pil_image = _bgr_to_pil(frame)
            if force_noise_type is None:
                assert classifier is not None
                noise_type, _ = classifier.predict(pil_image)
            else:
                noise_type = force_noise_type

            residual_strength = pipeline.periodic_residual_strength if noise_type == "periodic" else 1.0
            cache_key = f"{noise_type}:{residual_strength}"
            denoiser = denoiser_cache.get(cache_key)
            if denoiser is None:
                denoiser = pipeline.get_denoiser(noise_type, residual_strength)
                denoiser.load_model()
                denoiser_cache[cache_key] = denoiser

            denoised_image = pil_image
            for _ in range(passes):
                denoised_image = denoiser.run(denoised_image)  # type: ignore[union-attr]

            if noise_type == "periodic" and getattr(denoiser, "model_name", None) != "PeriodicFFTGuidedNAFNet":
                denoised_image = pipeline._apply_periodic_fft_postprocessing(denoised_image)
                denoised_image = pipeline._apply_unsharp_mask(denoised_image)

            writer.write(_pil_to_bgr(denoised_image, width, height))
            processed_frames += 1
            _call_progress(progress_callback, "denoise", processed_frames, frame_count)

        _call_progress(progress_callback, "rebuild", processed_frames, frame_count)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    _call_log(log_callback, f"Saved denoised video: {destination}")
    return VideoProcessingResult(
        input_video_path=input_path,
        output_video_path=destination,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        processed_frame_count=processed_frames,
        force_noise_type=force_noise_type,
        passes=passes,
    )
