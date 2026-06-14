from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.run_pipeline import NoiseReductionPipeline
from src.preprocessing.periodic_fft_filter import remove_periodic_noise_fft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare periodic denoising strategies on a single image.")
    parser.add_argument("--image", required=True, help="Input periodic-noisy image path.")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def pil_to_bgr(image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    labeled = frame.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 42), (20, 20, 20), thickness=-1)
    cv2.putText(
        labeled,
        label,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        print(f"Error: input image not found: {image_path}")
        return 1

    noisy_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if noisy_bgr is None:
        print(f"Error: failed to read image: {image_path}")
        return 1

    output_dir = ensure_dir(Path("outputs/periodic_strategy_comparison").resolve())

    pipeline = NoiseReductionPipeline()
    periodic_denoiser = pipeline.get_denoiser("periodic", pipeline.periodic_residual_strength)
    periodic_denoiser.load_model()

    fft_only_bgr = remove_periodic_noise_fft(
        noisy_bgr,
        threshold_ratio=0.08,
        min_distance=10,
        filter_radius=8,
    )

    ai_only_pil = periodic_denoiser.run(image_path)
    ai_only_bgr = pil_to_bgr(ai_only_pil)

    ai_fft_bgr = remove_periodic_noise_fft(
        ai_only_bgr,
        threshold_ratio=0.08,
        min_distance=10,
        filter_radius=4,
    )

    original_path = output_dir / f"{image_path.stem}_original.png"
    fft_only_path = output_dir / f"{image_path.stem}_fft_only.png"
    ai_only_path = output_dir / f"{image_path.stem}_ai_only.png"
    ai_fft_path = output_dir / f"{image_path.stem}_ai_fft_post.png"
    comparison_path = output_dir / f"{image_path.stem}_strategies.png"

    comparison = cv2.hconcat(
        [
            add_label(noisy_bgr, "ORIGINAL"),
            add_label(fft_only_bgr, "FFT ONLY"),
            add_label(ai_only_bgr, "AI ONLY"),
            add_label(ai_fft_bgr, "AI + FFT"),
        ]
    )

    for path, image in (
        (original_path, noisy_bgr),
        (fft_only_path, fft_only_bgr),
        (ai_only_path, ai_only_bgr),
        (ai_fft_path, ai_fft_bgr),
        (comparison_path, comparison),
    ):
        if not cv2.imwrite(str(path), image):
            print(f"Error: failed to write output image: {path}")
            return 1

    print(f"Original output: {original_path}")
    print(f"FFT-only output: {fft_only_path}")
    print(f"AI-only output: {ai_only_path}")
    print(f"AI+FFT output: {ai_fft_path}")
    print(f"Comparison output: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
