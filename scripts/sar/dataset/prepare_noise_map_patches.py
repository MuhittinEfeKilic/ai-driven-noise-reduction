from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import rasterio
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "rasterio is required for ENVI .img + .hdr support. Install it with: pip install rasterio"
    ) from exc


EPSILON = 1e-8
LOW_VARIANCE_THRESHOLD = 1e-8


@dataclass
class PatchStats:
    used_patches: int = 0
    skipped_patches: int = 0
    overwrite_skipped_patches: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SAR noise-map learning patches.")
    parser.add_argument("--input-original", type=Path, required=True)
    parser.add_argument("--input-esa-cleared", type=Path, required=True)
    parser.add_argument("--input-matlab-cleared", type=Path, required=True)
    parser.add_argument("--input-incidence", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--scene-prefix", type=str, default=None)
    parser.add_argument("--log-transform", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for arg_name in ("input_original", "input_esa_cleared", "input_matlab_cleared"):
        path = getattr(args, arg_name)
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        if path.suffix.lower() != ".img":
            raise ValueError(f"{path} must be an ENVI .img file.")
    if args.input_incidence is not None:
        if not args.input_incidence.exists():
            raise FileNotFoundError(f"Missing incidence file: {args.input_incidence}")
        if args.input_incidence.suffix.lower() != ".img":
            raise ValueError("--input-incidence must point to an ENVI .img file.")
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive.")
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")


def normalize_to_unit(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value - min_value <= EPSILON:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - min_value) / (max_value - min_value)).astype(np.float32)


def load_envi_image(path: Path, use_log_transform: bool) -> np.ndarray:
    with rasterio.open(path) as dataset:
        image = dataset.read(1).astype(np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = normalize_to_unit(image)
    if use_log_transform:
        image = np.log1p(image)
        image = normalize_to_unit(image)
    return image.astype(np.float32)


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    success = cv2.imwrite(str(path), image_uint8)
    if not success:
        raise IOError(f"Failed to write image: {path}")


def save_npy(path: Path, array: np.ndarray | np.float32 | float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array, dtype=np.float32))


def should_skip_patch(input_patch: np.ndarray, target_patch: np.ndarray) -> bool:
    if input_patch.size == 0 or target_patch.size == 0:
        return True
    if float(np.var(input_patch)) <= LOW_VARIANCE_THRESHOLD:
        return True
    if float(np.var(target_patch)) <= LOW_VARIANCE_THRESHOLD:
        return True
    return False


def extract_noise_map_patches(
    esa_noise: np.ndarray,
    matlab_noise: np.ndarray,
    incidence: np.ndarray | None,
    output_root: Path,
    patch_size: int,
    stride: int,
    scene_prefix: str | None,
) -> PatchStats:
    height, width = esa_noise.shape
    if height < patch_size or width < patch_size:
        raise ValueError(f"Input shape {esa_noise.shape} is smaller than patch size {patch_size}.")

    stats = PatchStats()
    for top in range(0, height - patch_size + 1, stride):
        for left in range(0, width - patch_size + 1, stride):
            esa_patch = esa_noise[top : top + patch_size, left : left + patch_size]
            matlab_patch = matlab_noise[top : top + patch_size, left : left + patch_size]
            if should_skip_patch(esa_patch, matlab_patch):
                stats.skipped_patches += 1
                continue

            stem = f"patch_y{top}_x{left}"
            if scene_prefix:
                stem = f"{scene_prefix}_{stem}"
            png_name = f"{stem}.png"
            npy_name = f"{stem}.npy"

            output_paths = [
                output_root / "noisy" / png_name,
                output_root / "target" / png_name,
                output_root / "noisy_npy" / npy_name,
                output_root / "target_npy" / npy_name,
            ]
            if incidence is not None:
                output_paths.append(output_root / "incidence_scalar" / npy_name)
            existing_paths = [path for path in output_paths if path.exists()]
            if existing_paths:
                print(f"Warning: patch already exists, skipping overwrite: {existing_paths[0]}")
                stats.overwrite_skipped_patches += 1
                continue

            save_png(output_root / "noisy" / png_name, normalize_to_unit(esa_patch))
            save_png(output_root / "target" / png_name, normalize_to_unit(matlab_patch))
            save_npy(output_root / "noisy_npy" / npy_name, esa_patch)
            save_npy(output_root / "target_npy" / npy_name, matlab_patch)

            if incidence is not None:
                incidence_patch = incidence[top : top + patch_size, left : left + patch_size]
                save_npy(output_root / "incidence_scalar" / npy_name, float(incidence_patch.mean()))

            stats.used_patches += 1

    return stats


def main() -> None:
    args = parse_args()
    validate_args(args)

    original = load_envi_image(args.input_original, args.log_transform)
    esa_cleared = load_envi_image(args.input_esa_cleared, args.log_transform)
    matlab_cleared = load_envi_image(args.input_matlab_cleared, args.log_transform)
    if original.shape != esa_cleared.shape or original.shape != matlab_cleared.shape:
        raise ValueError(
            "Shape mismatch: "
            f"original={original.shape}, esa={esa_cleared.shape}, matlab={matlab_cleared.shape}"
        )

    incidence = None
    if args.input_incidence is not None:
        incidence = load_envi_image(args.input_incidence, use_log_transform=False)
        if incidence.shape != original.shape:
            raise ValueError(
                f"Incidence shape {incidence.shape} does not match input shape {original.shape}."
            )

    esa_noise = (original - esa_cleared).astype(np.float32)
    matlab_noise = (original - matlab_cleared).astype(np.float32)
    stats = extract_noise_map_patches(
        esa_noise=esa_noise,
        matlab_noise=matlab_noise,
        incidence=incidence,
        output_root=args.output_root,
        patch_size=args.patch_size,
        stride=args.stride,
        scene_prefix=args.scene_prefix,
    )

    print(f"Input shape: {original.shape}")
    print(f"Used patches: {stats.used_patches}")
    print(f"Skipped patches: {stats.skipped_patches}")
    print(f"Scene prefix: {args.scene_prefix}")
    print(f"Overwrite-skipped patches: {stats.overwrite_skipped_patches}")
    print(f"Incidence scalar used: {incidence is not None}")


if __name__ == "__main__":
    main()
