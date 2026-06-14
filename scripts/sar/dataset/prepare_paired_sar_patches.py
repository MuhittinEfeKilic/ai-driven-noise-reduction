from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


EPSILON = 1e-8
LOW_VARIANCE_THRESHOLD = 1e-6


def get_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "NumPy is required for paired SAR patch extraction. "
            "Install it with: pip install numpy"
        ) from exc
    return np


def get_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "OpenCV (cv2) is required to save PNG patches. "
            "Install it with: pip install opencv-python"
        ) from exc
    return cv2


def get_rasterio():
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "rasterio is required for ENVI .img + .hdr support. "
            "Install it with: pip install rasterio"
        ) from exc
    return rasterio


@dataclass
class PatchStats:
    used_patches: int = 0
    skipped_patches: int = 0
    overwrite_skipped_patches: int = 0


@dataclass
class PairedImages:
    noisy: np.ndarray
    target: np.ndarray
    incidence: np.ndarray | None
    esa_cleared: np.ndarray | None
    column_profile: np.ndarray | None
    soft_swath: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare paired SAR patches from ENVI .img + .hdr files."
    )
    parser.add_argument(
        "--input-noisy",
        type=Path,
        required=True,
        help="Path to the noisy ENVI .img file.",
    )
    parser.add_argument(
        "--input-target",
        type=Path,
        required=True,
        help="Path to the target ENVI .img file.",
    )
    parser.add_argument(
        "--input-incidence",
        type=Path,
        default=None,
        help="Optional path to the incidence ENVI .img file.",
    )
    parser.add_argument(
        "--input-esa-cleared",
        type=Path,
        default=None,
        help="Optional path to the ESA-cleared ENVI .img file.",
    )
    parser.add_argument(
        "--input-column-profile",
        type=Path,
        default=None,
        help="Optional path to a column profile PNG map.",
    )
    parser.add_argument(
        "--input-soft-swath",
        type=Path,
        default=None,
        help="Optional path to a soft swath attention PNG map.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root for noisy/target/incidence/esa_cleared patch folders.",
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
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.05,
        help="Minimum ratio of non-zero pixels required to keep a patch.",
    )
    parser.add_argument(
        "--scene-prefix",
        type=str,
        default=None,
        help="Optional prefix to prepend to patch filenames for multiscene datasets.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be a positive integer.")
    if args.stride <= 0:
        raise ValueError("--stride must be a positive integer.")
    if not 0.0 <= args.min_valid_ratio <= 1.0:
        raise ValueError("--min-valid-ratio must be between 0.0 and 1.0.")

    required_inputs = {
        "--input-noisy": args.input_noisy,
        "--input-target": args.input_target,
    }
    optional_inputs = {
        "--input-incidence": args.input_incidence,
        "--input-esa-cleared": args.input_esa_cleared,
    }
    optional_png_inputs = {
        "--input-column-profile": args.input_column_profile,
        "--input-soft-swath": args.input_soft_swath,
    }

    for arg_name, input_path in required_inputs.items():
        if not input_path.exists():
            raise FileNotFoundError(f"Missing file for {arg_name}: {input_path}")
        if input_path.suffix.lower() != ".img":
            raise ValueError(f"{arg_name} must point to an ENVI .img file: {input_path}")

    for arg_name, input_path in optional_png_inputs.items():
        if input_path is None:
            continue
        if not input_path.exists():
            raise FileNotFoundError(f"Missing file for {arg_name}: {input_path}")
        if input_path.suffix.lower() != ".png":
            raise ValueError(f"{arg_name} must point to a PNG file: {input_path}")

    for arg_name, input_path in optional_inputs.items():
        if input_path is None:
            continue
        if not input_path.exists():
            raise FileNotFoundError(f"Missing file for {arg_name}: {input_path}")
        if input_path.suffix.lower() != ".img":
            raise ValueError(f"{arg_name} must point to an ENVI .img file: {input_path}")


def normalize_to_unit(image: np.ndarray) -> np.ndarray:
    np = get_numpy()
    image = image.astype(np.float32, copy=False)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value - min_value <= EPSILON:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - min_value) / (max_value - min_value)).astype(np.float32)


def load_envi_image(image_path: Path, use_log_transform: bool) -> np.ndarray:
    np = get_numpy()
    rasterio = get_rasterio()
    with rasterio.open(image_path) as dataset:
        image = dataset.read(1).astype(np.float32)

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = normalize_to_unit(image)
    if use_log_transform:
        image = np.log1p(image)
        image = normalize_to_unit(image)
    return image.astype(np.float32)


def load_grayscale_png(image_path: Path) -> np.ndarray:
    cv2 = get_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read PNG image: {image_path}")
    return normalize_to_unit(image.astype(get_numpy().float32))


def load_paired_images(args: argparse.Namespace) -> PairedImages:
    noisy = load_envi_image(args.input_noisy, args.log_transform)
    target = load_envi_image(args.input_target, args.log_transform)
    incidence = (
        load_envi_image(args.input_incidence, use_log_transform=False)
        if args.input_incidence is not None
        else None
    )
    esa_cleared = (
        load_envi_image(args.input_esa_cleared, args.log_transform)
        if args.input_esa_cleared is not None
        else None
    )
    column_profile = (
        load_grayscale_png(args.input_column_profile)
        if args.input_column_profile is not None
        else None
    )
    soft_swath = (
        load_grayscale_png(args.input_soft_swath)
        if args.input_soft_swath is not None
        else None
    )

    return PairedImages(
        noisy=noisy,
        target=target,
        incidence=incidence,
        esa_cleared=esa_cleared,
        column_profile=column_profile,
        soft_swath=soft_swath,
    )


def validate_shapes(images: PairedImages) -> tuple[int, int]:
    base_shape = images.noisy.shape
    if images.target.shape != base_shape:
        raise ValueError(
            f"Shape mismatch between noisy {base_shape} and target {images.target.shape}."
        )
    if images.incidence is not None and images.incidence.shape != base_shape:
        raise ValueError(
            "Shape mismatch between noisy "
            f"{base_shape} and incidence {images.incidence.shape}."
        )
    if images.esa_cleared is not None and images.esa_cleared.shape != base_shape:
        raise ValueError(
            "Shape mismatch between noisy "
            f"{base_shape} and esa_cleared {images.esa_cleared.shape}."
        )
    if images.column_profile is not None and images.column_profile.shape != base_shape:
        raise ValueError(
            "Shape mismatch between noisy "
            f"{base_shape} and column_profile {images.column_profile.shape}."
        )
    if images.soft_swath is not None and images.soft_swath.shape != base_shape:
        raise ValueError(
            "Shape mismatch between noisy "
            f"{base_shape} and soft_swath {images.soft_swath.shape}."
        )
    return base_shape


def save_patch(output_path: Path, patch: np.ndarray) -> None:
    np = get_numpy()
    cv2 = get_cv2()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patch_uint8 = np.clip(patch, 0.0, 1.0)
    patch_uint8 = (patch_uint8 * 255.0).round().astype(np.uint8)
    success = cv2.imwrite(str(output_path), patch_uint8)
    if not success:
        raise IOError(f"Failed to save patch: {output_path}")


def save_scalar(output_path: Path, value: float) -> None:
    np = get_numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.array(value, dtype=np.float32))


def should_skip_patch(noisy_patch: np.ndarray, target_patch: np.ndarray) -> bool:
    np = get_numpy()
    if noisy_patch.size == 0 or target_patch.size == 0:
        return True
    if not np.any(noisy_patch > 0.0) and not np.any(target_patch > 0.0):
        return True
    if float(np.var(noisy_patch)) <= LOW_VARIANCE_THRESHOLD:
        return True
    if float(np.var(target_patch)) <= LOW_VARIANCE_THRESHOLD:
        return True
    return False


def has_enough_valid_pixels(
    noisy_patch: np.ndarray,
    target_patch: np.ndarray,
    min_valid_ratio: float,
) -> bool:
    np = get_numpy()
    valid_mask = (noisy_patch > 0.0) | (target_patch > 0.0)
    valid_ratio = float(np.count_nonzero(valid_mask)) / float(valid_mask.size)
    return valid_ratio >= min_valid_ratio


def extract_paired_patches(
    images: PairedImages,
    output_dir: Path,
    patch_size: int,
    stride: int,
    min_valid_ratio: float,
    scene_prefix: str | None,
) -> PatchStats:
    height, width = images.noisy.shape
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"Input shape {images.noisy.shape} is smaller than patch size {patch_size}."
        )

    stats = PatchStats()

    for top in range(0, height - patch_size + 1, stride):
        for left in range(0, width - patch_size + 1, stride):
            noisy_patch = images.noisy[top : top + patch_size, left : left + patch_size]
            target_patch = images.target[top : top + patch_size, left : left + patch_size]

            if should_skip_patch(noisy_patch, target_patch):
                stats.skipped_patches += 1
                continue
            if not has_enough_valid_pixels(noisy_patch, target_patch, min_valid_ratio):
                stats.skipped_patches += 1
                continue

            if scene_prefix:
                patch_name = f"{scene_prefix}_patch_y{top}_x{left}.png"
            else:
                patch_name = f"patch_y{top}_x{left}.png"

            output_paths = [
                output_dir / "noisy" / patch_name,
                output_dir / "target" / patch_name,
            ]
            if images.incidence is not None:
                output_paths.append(output_dir / "incidence" / patch_name)
                output_paths.append(output_dir / "incidence_scalar" / f"{Path(patch_name).stem}.npy")
            if images.esa_cleared is not None:
                output_paths.append(output_dir / "esa_cleared" / patch_name)
            if images.column_profile is not None:
                output_paths.append(output_dir / "column_profile" / patch_name)
            if images.soft_swath is not None:
                output_paths.append(output_dir / "soft_swath" / patch_name)

            existing_paths = [path for path in output_paths if path.exists()]
            if existing_paths:
                print(f"Warning: patch already exists, skipping overwrite: {existing_paths[0]}")
                stats.overwrite_skipped_patches += 1
                continue

            save_patch(output_dir / "noisy" / patch_name, noisy_patch)
            save_patch(output_dir / "target" / patch_name, target_patch)

            if images.incidence is not None:
                incidence_patch = images.incidence[
                    top : top + patch_size, left : left + patch_size
                ]
                save_patch(output_dir / "incidence" / patch_name, incidence_patch)
                mean_incidence = float(incidence_patch.mean())
                save_scalar(
                    output_dir / "incidence_scalar" / f"{Path(patch_name).stem}.npy",
                    mean_incidence,
                )

            if images.esa_cleared is not None:
                esa_patch = images.esa_cleared[
                    top : top + patch_size, left : left + patch_size
                ]
                save_patch(output_dir / "esa_cleared" / patch_name, esa_patch)

            if images.column_profile is not None:
                column_profile_patch = images.column_profile[
                    top : top + patch_size, left : left + patch_size
                ]
                save_patch(output_dir / "column_profile" / patch_name, column_profile_patch)

            if images.soft_swath is not None:
                soft_swath_patch = images.soft_swath[
                    top : top + patch_size, left : left + patch_size
                ]
                save_patch(output_dir / "soft_swath" / patch_name, soft_swath_patch)

            stats.used_patches += 1

    return stats


def main() -> None:
    args = parse_args()
    validate_args(args)

    images = load_paired_images(args)
    input_shape = validate_shapes(images)
    stats = extract_paired_patches(
        images=images,
        output_dir=args.output_root,
        patch_size=args.patch_size,
        stride=args.stride,
        min_valid_ratio=args.min_valid_ratio,
        scene_prefix=args.scene_prefix,
    )

    print(f"Input shape: {input_shape}")
    print(f"Used patches: {stats.used_patches}")
    print(f"Skipped patches: {stats.skipped_patches}")
    print(f"Scene prefix: {args.scene_prefix}")
    print(f"Overwrite-skipped patches: {stats.overwrite_skipped_patches}")
    print(f"Incidence used: {images.incidence is not None}")
    print(f"ESA used: {images.esa_cleared is not None}")
    print(f"Column profile used: {images.column_profile is not None}")
    print(f"Soft swath used: {images.soft_swath is not None}")


if __name__ == "__main__":
    main()
