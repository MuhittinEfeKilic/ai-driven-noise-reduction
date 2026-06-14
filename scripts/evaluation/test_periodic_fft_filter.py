from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.periodic_fft_filter import (
    apply_frequency_filter,
    create_band_reject_filter,
    detect_noise_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the periodic FFT preprocessing filter on a single image.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--threshold_ratio", type=float, default=0.08, help="Threshold ratio for FFT peak detection.")
    parser.add_argument("--min_distance", type=int, default=10, help="Minimum distance for FFT peak selection.")
    parser.add_argument("--filter_radius", type=int, default=8, help="Radius of each notch in the filter mask.")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        print(f"Error: input image not found: {image_path}")
        return 1

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: failed to read image: {image_path}")
        return 1

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(grayscale.astype(np.float32, copy=False))
    fft_shifted = np.fft.fftshift(fft)
    magnitude_spectrum = np.log1p(np.abs(fft_shifted))

    noise_points = detect_noise_points(
        magnitude_spectrum,
        threshold_ratio=args.threshold_ratio,
        min_distance=args.min_distance,
    )
    filter_mask = create_band_reject_filter(
        magnitude_spectrum.shape,
        noise_points,
        filter_radius=args.filter_radius,
    )
    filtered = apply_frequency_filter(image, filter_mask)

    output_dir = ensure_dir(Path("outputs/periodic_fft_test").resolve())
    filtered_path = output_dir / f"{image_path.stem}_fft_filtered.png"
    comparison_path = output_dir / f"{image_path.stem}_comparison.png"
    spectrum_path = output_dir / f"{image_path.stem}_spectrum.png"
    detected_points_path = output_dir / f"{image_path.stem}_detected_points.png"
    filter_mask_path = output_dir / f"{image_path.stem}_filter_mask.png"

    comparison = cv2.hconcat([image, filtered])
    spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spectrum_bgr = cv2.cvtColor(spectrum_normalized, cv2.COLOR_GRAY2BGR)
    for row, col in noise_points:
        cv2.circle(spectrum_bgr, (int(col), int(row)), 4, (0, 0, 255), 1)
    filter_mask_image = np.clip(filter_mask * 255.0, 0.0, 255.0).astype(np.uint8)

    if not cv2.imwrite(str(filtered_path), filtered):
        print(f"Error: failed to write filtered image: {filtered_path}")
        return 1
    if not cv2.imwrite(str(comparison_path), comparison):
        print(f"Error: failed to write comparison image: {comparison_path}")
        return 1
    if not cv2.imwrite(str(spectrum_path), spectrum_normalized):
        print(f"Error: failed to write spectrum image: {spectrum_path}")
        return 1
    if not cv2.imwrite(str(detected_points_path), spectrum_bgr):
        print(f"Error: failed to write detected-points image: {detected_points_path}")
        return 1
    if not cv2.imwrite(str(filter_mask_path), filter_mask_image):
        print(f"Error: failed to write filter mask image: {filter_mask_path}")
        return 1

    print(f"Filtered output: {filtered_path}")
    print(f"Comparison output: {comparison_path}")
    print(f"Spectrum output: {spectrum_path}")
    print(f"Detected points output: {detected_points_path}")
    print(f"Filter mask output: {filter_mask_path}")
    print(f"Used threshold_ratio: {args.threshold_ratio}")
    print(f"Used min_distance: {args.min_distance}")
    print(f"Used filter_radius: {args.filter_radius}")
    print(f"Detected noise points count: {len(noise_points)}")
    print(f"Detected noise points: {noise_points}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
