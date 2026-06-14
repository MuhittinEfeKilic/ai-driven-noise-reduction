from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    import rasterio
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "rasterio is required for SAR ENVI preprocessing. Install it with: pip install rasterio"
    ) from exc


EPSILON = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create swath-aware column profile and banding masks from a noisy SAR HV image."
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        required=True,
        help="Path to the noisy HV SAR ENVI .img file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sar/preprocessing"),
        help="Directory where mask PNG files will be written.",
    )
    parser.add_argument(
        "--threshold-scale",
        type=float,
        default=1.5,
        help="Scale factor for the column-gradient threshold: mean + scale * std.",
    )
    parser.add_argument(
        "--dilate-kernel",
        type=int,
        default=31,
        help="Horizontal dilation kernel width for widening detected banding columns.",
    )
    parser.add_argument(
        "--soft-blur-kernel",
        type=int,
        default=101,
        help="Gaussian blur kernel size for the soft swath attention map.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_image.exists():
        raise FileNotFoundError(f"Input image not found: {args.input_image}")
    if args.input_image.suffix.lower() != ".img":
        raise ValueError(f"--input-image must point to an ENVI .img file: {args.input_image}")
    if args.threshold_scale < 0.0:
        raise ValueError("--threshold-scale must be non-negative.")
    if args.dilate_kernel <= 0:
        raise ValueError("--dilate-kernel must be a positive integer.")
    if args.dilate_kernel % 2 == 0:
        raise ValueError("--dilate-kernel must be an odd integer.")
    if args.soft_blur_kernel <= 0:
        raise ValueError("--soft-blur-kernel must be a positive integer.")


def normalize_to_unit(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value - min_value <= EPSILON:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - min_value) / (max_value - min_value)).astype(np.float32)


def round_up_to_odd(value: int) -> int:
    if value % 2 == 0:
        return value + 1
    return value


def load_envi_image(image_path: Path) -> np.ndarray:
    with rasterio.open(image_path) as dataset:
        image = dataset.read(1).astype(np.float32)

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return normalize_to_unit(image)


def save_grayscale_png(output_path: Path, image: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = np.clip(image, 0.0, 1.0)
    image_uint8 = (image_uint8 * 255.0).round().astype(np.uint8)
    success = cv2.imwrite(str(output_path), image_uint8)
    if not success:
        raise IOError(f"Failed to write image: {output_path}")


def create_column_profile_map(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = image.shape[0]
    column_profile = np.mean(image, axis=0).astype(np.float32)
    column_profile = normalize_to_unit(column_profile)
    profile_map = np.repeat(column_profile[None, :], height, axis=0)
    return column_profile, profile_map.astype(np.float32)


def create_swath_banding_mask(
    column_profile: np.ndarray,
    height: int,
    threshold_scale: float,
    dilate_kernel: int,
) -> tuple[np.ndarray, float]:
    profile_gradient = np.abs(np.diff(column_profile, prepend=column_profile[0])).astype(np.float32)
    threshold = float(np.mean(profile_gradient) + threshold_scale * np.std(profile_gradient))
    column_mask = (profile_gradient > threshold).astype(np.uint8)[None, :]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel, 1))
    dilated_mask = cv2.dilate(column_mask, kernel, iterations=1)
    mask = np.repeat(dilated_mask.astype(np.float32), height, axis=0)
    return mask, threshold


def create_soft_attention_map(binary_mask: np.ndarray, blur_kernel: int) -> np.ndarray:
    kernel_size = round_up_to_odd(blur_kernel)
    mask = binary_mask.astype(np.float32, copy=False)
    blurred = cv2.GaussianBlur(
        mask,
        (kernel_size, kernel_size),
        sigmaX=0,
        sigmaY=0,
        borderType=cv2.BORDER_REFLECT,
    )
    return np.clip(normalize_to_unit(blurred), 0.0, 1.0).astype(np.float32)


def main() -> None:
    args = parse_args()
    validate_args(args)

    image = load_envi_image(args.input_image)
    height, width = image.shape
    column_profile, column_profile_map = create_column_profile_map(image)
    swath_banding_mask, threshold = create_swath_banding_mask(
        column_profile=column_profile,
        height=height,
        threshold_scale=args.threshold_scale,
        dilate_kernel=args.dilate_kernel,
    )
    soft_attention_map = create_soft_attention_map(
        binary_mask=swath_banding_mask,
        blur_kernel=args.soft_blur_kernel,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    column_profile_path = args.output_dir / "column_profile_map.png"
    swath_mask_path = args.output_dir / "swath_banding_mask.png"
    soft_attention_path = args.output_dir / "soft_swath_attention_map.png"
    save_grayscale_png(column_profile_path, column_profile_map)
    save_grayscale_png(swath_mask_path, swath_banding_mask)
    save_grayscale_png(soft_attention_path, soft_attention_map)

    print(f"Input shape: {(height, width)}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Soft blur kernel: {round_up_to_odd(args.soft_blur_kernel)}")
    print(f"Saved column profile map: {column_profile_path}")
    print(f"Saved swath banding mask: {swath_mask_path}")
    print(f"Saved soft swath attention map: {soft_attention_path}")


if __name__ == "__main__":
    main()
