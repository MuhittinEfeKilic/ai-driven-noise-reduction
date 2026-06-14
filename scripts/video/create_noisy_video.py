from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a noisy demo video from a clean input video.")
    parser.add_argument("--video", required=True, help="Input clean video path.")
    parser.add_argument(
        "--noise_type",
        required=True,
        choices=("gaussian", "salt_pepper", "speckle", "periodic"),
        help="Synthetic noise type to apply.",
    )
    parser.add_argument(
        "--output",
        default="outputs/video/noisy_demo.mp4",
        help="Output noisy video path.",
    )
    return parser.parse_args()


def add_gaussian_noise(frame: np.ndarray, rng: np.random.Generator, std: float = 25.0) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=std, size=frame.shape).astype(np.float32)
    noisy = frame.astype(np.float32) + noise
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


def add_salt_pepper_noise(frame: np.ndarray, rng: np.random.Generator, amount: float = 0.03) -> np.ndarray:
    noisy = frame.copy()
    height, width = noisy.shape[:2]
    total_pixels = height * width
    salt_count = int(total_pixels * amount * 0.5)
    pepper_count = int(total_pixels * amount * 0.5)

    if salt_count > 0:
        ys = rng.integers(0, height, size=salt_count)
        xs = rng.integers(0, width, size=salt_count)
        noisy[ys, xs] = 255

    if pepper_count > 0:
        ys = rng.integers(0, height, size=pepper_count)
        xs = rng.integers(0, width, size=pepper_count)
        noisy[ys, xs] = 0

    return noisy


def add_speckle_noise(frame: np.ndarray, rng: np.random.Generator, std: float = 0.18) -> np.ndarray:
    frame_float = frame.astype(np.float32) / 255.0
    noise = rng.normal(loc=0.0, scale=std, size=frame.shape).astype(np.float32)
    noisy = frame_float + frame_float * noise
    return np.clip(noisy * 255.0, 0.0, 255.0).astype(np.uint8)


def add_periodic_noise(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = frame.shape[:2]
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )

    angle = float(rng.choice([0.0, np.pi / 2.0, np.pi / 4.0, 3.0 * np.pi / 4.0]))
    direction = np.cos(angle) * xx + np.sin(angle) * yy
    amplitude = float(rng.uniform(22.0, 42.0))
    frequency = float(rng.uniform(0.035, 0.075))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    pattern = amplitude * np.sin(2.0 * np.pi * frequency * direction + phase)

    noisy = frame.astype(np.float32) + pattern[..., None]
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


def apply_noise(frame: np.ndarray, noise_type: str, rng: np.random.Generator) -> np.ndarray:
    if noise_type == "gaussian":
        return add_gaussian_noise(frame, rng)
    if noise_type == "salt_pepper":
        return add_salt_pepper_noise(frame, rng)
    if noise_type == "speckle":
        return add_speckle_noise(frame, rng)
    if noise_type == "periodic":
        return add_periodic_noise(frame, rng)
    raise ValueError(f"Unsupported noise type: {noise_type}")


def main() -> int:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0.0:
        fps = 30.0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Failed to open video writer: {output_path}")

    rng = np.random.default_rng(42)

    print(f"Video FPS: {fps:.3f}")
    print(f"Total frame count: {frame_count}")
    print(f"Selected noise type: {args.noise_type}")

    processed_frames = 0
    try:
        with tqdm(total=frame_count if frame_count > 0 else None, desc="Creating noisy video", unit="frame") as progress:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                noisy_frame = apply_noise(frame, args.noise_type, rng)
                writer.write(noisy_frame)
                processed_frames += 1
                progress.update(1)
    finally:
        capture.release()
        writer.release()

    print(f"Processed frame count: {processed_frames}")
    print(f"Output path: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
