from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import rasterio
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "rasterio is required for SAR ENVI comparison inputs. Install it with: pip install rasterio"
    ) from exc


EPSILON = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a final SAR denoising comparison figure.")
    parser.add_argument("--noisy", type=Path, required=True)
    parser.add_argument("--esa-cleared", type=Path, default=None)
    parser.add_argument("--matlab-cleared", type=Path, required=True)
    parser.add_argument("--ai-output", type=Path, required=True)
    parser.add_argument("--include-residual-map", action="store_true")
    parser.add_argument("--apply-clahe", action="store_true")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/sar/comparisons/final_comparison_enhanced.png"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    required_paths = {
        "--noisy": args.noisy,
        "--matlab-cleared": args.matlab_cleared,
        "--ai-output": args.ai_output,
    }
    for arg_name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file for {arg_name}: {path}")
    if args.esa_cleared is not None and not args.esa_cleared.exists():
        raise FileNotFoundError(f"Missing file for --esa-cleared: {args.esa_cleared}")


def normalize_to_unit(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value - min_value <= EPSILON:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - min_value) / (max_value - min_value)).astype(np.float32)


def percentile_normalize(image: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    lower_value = float(np.percentile(image, lower))
    upper_value = float(np.percentile(image, upper))
    if upper_value - lower_value <= EPSILON:
        return normalize_to_unit(image)
    normalized = (image - lower_value) / (upper_value - lower_value)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    import cv2

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    enhanced = clahe.apply(image_uint8)
    return enhanced.astype(np.float32) / 255.0


def load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".img":
        with rasterio.open(path) as dataset:
            image = dataset.read(1).astype(np.float32)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        return normalize_to_unit(image)

    image = plt.imread(path)
    if image.ndim == 3:
        image = image[..., 0]
    return normalize_to_unit(image.astype(np.float32))


def prepare_visualization_image(image: np.ndarray, apply_clahe_flag: bool) -> np.ndarray:
    visual = percentile_normalize(image)
    if apply_clahe_flag:
        visual = apply_clahe(visual)
        visual = percentile_normalize(visual)
    return np.clip(visual, 0.0, 1.0).astype(np.float32)


def add_panel(axis: plt.Axes, image: np.ndarray, title: str) -> None:
    axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axis.set_title(title, fontsize=18, fontweight="bold", pad=14)
    axis.axis("off")


def main() -> None:
    args = parse_args()
    validate_args(args)

    noisy = load_image(args.noisy)
    esa_cleared = load_image(args.esa_cleared) if args.esa_cleared is not None else np.zeros_like(noisy)
    matlab_cleared = load_image(args.matlab_cleared)
    ai_output = load_image(args.ai_output)
    residual_map = normalize_to_unit(np.abs(ai_output - matlab_cleared))
    difference_map = normalize_to_unit(np.abs(noisy - ai_output))

    panel_images: list[np.ndarray] = [
        prepare_visualization_image(noisy, args.apply_clahe),
        prepare_visualization_image(esa_cleared, args.apply_clahe),
        prepare_visualization_image(matlab_cleared, args.apply_clahe),
    ]
    panel_titles = ["Noisy", "ESA Cleared", "MATLAB Cleared"]

    if args.include_residual_map:
        panel_images.append(prepare_visualization_image(residual_map, args.apply_clahe))
        panel_titles.append("Residual |AI - MATLAB|")

    panel_images.append(prepare_visualization_image(ai_output, args.apply_clahe))
    panel_titles.append("AI Output")
    panel_images.append(prepare_visualization_image(difference_map, args.apply_clahe))
    panel_titles.append("Difference Map")

    figure, axes = plt.subplots(1, len(panel_images), figsize=(6.4 * len(panel_images), 7.2))
    for axis, image, title in zip(np.atleast_1d(axes), panel_images, panel_titles):
        add_panel(axis, image, title)

    figure.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.03, wspace=0.04)
    figure.tight_layout(pad=1.2)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Saved comparison figure: {args.output_path}")


if __name__ == "__main__":
    main()
