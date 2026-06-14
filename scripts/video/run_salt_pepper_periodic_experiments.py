from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
NOISE_TYPES = ("salt_pepper", "periodic")
LABELS = ("CLEAN", "NOISY", "DENOISED")


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create salt-pepper and periodic noisy videos, denoise them, "
            "and export clean/noisy/denoised comparison videos."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/video/noise_experiments"))
    parser.add_argument("--noise-types", nargs="+", choices=NOISE_TYPES, default=list(NOISE_TYPES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--salt-pepper-amount", type=float, default=0.03)
    parser.add_argument("--salt-pepper-kernel", type=int, default=3, choices=(3, 5, 7))
    parser.add_argument("--periodic-amplitude", type=float, default=32.0)
    parser.add_argument("--periodic-frequency", type=float, default=0.055)
    parser.add_argument("--periodic-notch-radius", type=int, default=8)
    parser.add_argument("--periodic-notch-count", type=int, default=8)
    return parser.parse_args()


def discover_videos(input_dir: Path) -> list[Path]:
    videos = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not videos:
        raise FileNotFoundError(f"No videos found in: {input_dir}")
    return videos


def open_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    return capture


def read_video_info(capture: cv2.VideoCapture) -> VideoInfo:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return VideoInfo(
        fps=fps if fps > 0.0 else 30.0,
        width=width,
        height=height,
        frame_count=frame_count,
    )


def create_writer(output_path: Path, info: VideoInfo, panel_count: int = 1) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info.fps,
        (info.width * panel_count, info.height),
    )
    if not writer.isOpened():
        raise ValueError(f"Failed to open video writer: {output_path}")
    return writer


def add_salt_pepper_noise(
    frame: np.ndarray,
    rng: np.random.Generator,
    amount: float,
) -> np.ndarray:
    noisy = frame.copy()
    height, width = noisy.shape[:2]
    total_pixels = height * width
    salt_count = int(total_pixels * amount * 0.5)
    pepper_count = int(total_pixels * amount * 0.5)

    if salt_count:
        ys = rng.integers(0, height, size=salt_count)
        xs = rng.integers(0, width, size=salt_count)
        noisy[ys, xs] = 255

    if pepper_count:
        ys = rng.integers(0, height, size=pepper_count)
        xs = rng.integers(0, width, size=pepper_count)
        noisy[ys, xs] = 0

    return noisy


def denoise_salt_pepper(frame: np.ndarray, kernel_size: int) -> np.ndarray:
    return cv2.medianBlur(frame, kernel_size)


def add_periodic_noise(
    frame: np.ndarray,
    rng: np.random.Generator,
    amplitude: float,
    frequency: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )

    angle = float(rng.choice([0.0, np.pi / 2.0, np.pi / 4.0, 3.0 * np.pi / 4.0]))
    direction = np.cos(angle) * xx + np.sin(angle) * yy
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    pattern = amplitude * np.sin(2.0 * np.pi * frequency * direction + phase)
    noisy = frame.astype(np.float32) + pattern[..., None]
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


def build_notch_mask(
    shape: tuple[int, int],
    radius: int,
    peak_count: int,
    protected_radius: int,
    magnitude: np.ndarray,
) -> np.ndarray:
    height, width = shape
    center_y, center_x = height // 2, width // 2
    mask = np.ones((height, width), dtype=np.float32)

    search = magnitude.copy()
    cv2.circle(search, (center_x, center_y), protected_radius, 0.0, thickness=-1)

    for _ in range(peak_count):
        _, peak_value, _, peak = cv2.minMaxLoc(search)
        if peak_value <= 0.0:
            break

        peak_x, peak_y = peak
        mirror_x = int(np.clip(2 * center_x - peak_x, 0, width - 1))
        mirror_y = int(np.clip(2 * center_y - peak_y, 0, height - 1))

        cv2.circle(mask, (peak_x, peak_y), radius, 0.0, thickness=-1)
        cv2.circle(mask, (mirror_x, mirror_y), radius, 0.0, thickness=-1)
        cv2.circle(search, (peak_x, peak_y), radius * 3, 0.0, thickness=-1)
        cv2.circle(search, (mirror_x, mirror_y), radius * 3, 0.0, thickness=-1)

    return mask


def denoise_periodic(
    frame: np.ndarray,
    notch_radius: int,
    notch_count: int,
) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0].astype(np.float32)

    spectrum = np.fft.fftshift(np.fft.fft2(y_channel))
    magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)
    protected_radius = max(16, min(frame.shape[:2]) // 10)
    mask = build_notch_mask(
        shape=y_channel.shape,
        radius=notch_radius,
        peak_count=notch_count,
        protected_radius=protected_radius,
        magnitude=magnitude,
    )

    filtered = spectrum * mask
    restored = np.fft.ifft2(np.fft.ifftshift(filtered)).real
    ycrcb[:, :, 0] = np.clip(restored, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    labeled = frame.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 48), (16, 16, 16), thickness=-1)
    cv2.putText(
        labeled,
        label,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def build_comparison(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray) -> np.ndarray:
    return cv2.hconcat(
        [
            add_label(clean, LABELS[0]),
            add_label(noisy, LABELS[1]),
            add_label(denoised, LABELS[2]),
        ]
    )


def process_video(
    video_path: Path,
    output_dir: Path,
    noise_type: str,
    seed: int,
    args: argparse.Namespace,
) -> None:
    capture = open_capture(video_path)
    info = read_video_info(capture)

    stem = video_path.stem
    noisy_path = output_dir / noise_type / f"{stem}_{noise_type}_noisy.mp4"
    denoised_path = output_dir / noise_type / f"{stem}_{noise_type}_denoised.mp4"
    comparison_path = output_dir / noise_type / f"{stem}_{noise_type}_comparison.mp4"

    noisy_writer = create_writer(noisy_path, info)
    denoised_writer = create_writer(denoised_path, info)
    comparison_writer = create_writer(comparison_path, info, panel_count=3)

    rng = np.random.default_rng(seed)
    processed_frames = 0

    try:
        total = info.frame_count if info.frame_count > 0 else None
        description = f"{video_path.name} / {noise_type}"
        with tqdm(total=total, desc=description, unit="frame") as progress:
            while True:
                ok, clean_frame = capture.read()
                if not ok:
                    break

                if noise_type == "salt_pepper":
                    noisy_frame = add_salt_pepper_noise(clean_frame, rng, args.salt_pepper_amount)
                    denoised_frame = denoise_salt_pepper(noisy_frame, args.salt_pepper_kernel)
                elif noise_type == "periodic":
                    noisy_frame = add_periodic_noise(
                        clean_frame,
                        rng,
                        args.periodic_amplitude,
                        args.periodic_frequency,
                    )
                    denoised_frame = denoise_periodic(
                        noisy_frame,
                        args.periodic_notch_radius,
                        args.periodic_notch_count,
                    )
                else:
                    raise ValueError(f"Unsupported noise type: {noise_type}")

                noisy_writer.write(noisy_frame)
                denoised_writer.write(denoised_frame)
                comparison_writer.write(build_comparison(clean_frame, noisy_frame, denoised_frame))
                processed_frames += 1
                progress.update(1)
    finally:
        capture.release()
        noisy_writer.release()
        denoised_writer.release()
        comparison_writer.release()

    print(f"Processed {processed_frames} frames")
    print(f"Noisy     : {noisy_path}")
    print(f"Denoised  : {denoised_path}")
    print(f"Comparison: {comparison_path}")


def main() -> int:
    args = parse_args()
    videos = discover_videos(args.input_dir)

    for video_path in videos:
        for noise_index, noise_type in enumerate(args.noise_types):
            stable_seed = args.seed + noise_index * 1000 + sum(video_path.name.encode("utf-8"))
            process_video(video_path, args.output_dir, noise_type, stable_seed, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
