from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass
class PatchStats:
    images_read: int = 0
    patches_saved: int = 0
    patches_skipped: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare normalized SAR image patches from raw Sentinel-1 images."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/sar/raw/sentinel1"),
        help="Directory containing raw SAR images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sar/processed/patches"),
        help="Directory to save extracted patches.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Square patch size in pixels.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Stride between extracted patches.",
    )
    parser.add_argument(
        "--log-transform",
        action="store_true",
        help="Apply log1p transform before re-normalizing to [0, 1].",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be a positive integer.")
    if args.stride <= 0:
        raise ValueError("--stride must be a positive integer.")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")


def list_image_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def read_grayscale_image(image_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Warning: could not read image, skipping: {image_path}")
        return None

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image.astype(np.float32)


def normalize_to_unit_range(image: np.ndarray) -> np.ndarray:
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_value) / (max_value - min_value)


def preprocess_image(image: np.ndarray, use_log_transform: bool) -> np.ndarray:
    normalized = normalize_to_unit_range(image)
    if use_log_transform:
        normalized = np.log1p(normalized)
        normalized = normalize_to_unit_range(normalized)
    return normalized.astype(np.float32)


def should_skip_patch(patch: np.ndarray) -> bool:
    if patch.size == 0:
        return True
    return not np.any(patch > 0.0)


def save_patch(patch: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patch_uint8 = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), patch_uint8)


def extract_patches_from_image(
    image: np.ndarray,
    image_stem: str,
    output_dir: Path,
    patch_size: int,
    stride: int,
) -> tuple[int, int]:
    height, width = image.shape
    if height < patch_size or width < patch_size:
        print(
            f"Warning: image is smaller than patch size, skipping: "
            f"{image_stem} ({height}x{width})"
        )
        return 0, 0

    saved_count = 0
    skipped_count = 0

    for top in range(0, height - patch_size + 1, stride):
        for left in range(0, width - patch_size + 1, stride):
            patch = image[top : top + patch_size, left : left + patch_size]
            if should_skip_patch(patch):
                skipped_count += 1
                continue

            patch_name = f"{image_stem}_y{top}_x{left}.png"
            save_patch(patch, output_dir / patch_name)
            saved_count += 1

    return saved_count, skipped_count


def main() -> None:
    args = parse_args()
    validate_args(args)

    image_paths = list_image_files(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats = PatchStats()

    for image_path in image_paths:
        image = read_grayscale_image(image_path)
        if image is None:
            continue

        stats.images_read += 1
        processed_image = preprocess_image(image, args.log_transform)
        saved_count, skipped_count = extract_patches_from_image(
            image=processed_image,
            image_stem=image_path.stem,
            output_dir=args.output_dir,
            patch_size=args.patch_size,
            stride=args.stride,
        )
        stats.patches_saved += saved_count
        stats.patches_skipped += skipped_count

    print(f"Images read: {stats.images_read}")
    print(f"Patches generated: {stats.patches_saved}")
    print(f"Patches skipped: {stats.patches_skipped}")


if __name__ == "__main__":
    main()
