from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "OpenCV (cv2) is required to run SAR baseline filters. "
        "Install it with 'pip install opencv-python'."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sar.filters.classical_filters import (  # noqa: E402
    frost_filter,
    gaussian_filter,
    lee_filter,
    median_filter,
    read_grayscale_image,
    save_grayscale_image,
)


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run classical SAR baseline filters on single images or directories."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/sar/samples"),
        help="Input image file or directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sar/baseline_filters"),
        help="Directory for filtered outputs.",
    )
    parser.add_argument("--median", action="store_true", help="Run median filter.")
    parser.add_argument("--gaussian", action="store_true", help="Run Gaussian filter.")
    parser.add_argument("--lee", action="store_true", help="Run Lee filter.")
    parser.add_argument("--frost", action="store_true", help="Run Frost-like filter.")
    parser.add_argument("--all", action="store_true", help="Run all baseline filters.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input}")


def resolve_filters(args: argparse.Namespace) -> list[str]:
    selected_filters = []
    if args.median:
        selected_filters.append("median")
    if args.gaussian:
        selected_filters.append("gaussian")
    if args.lee:
        selected_filters.append("lee")
    if args.frost:
        selected_filters.append("frost")

    if args.all or not selected_filters:
        return ["median", "gaussian", "lee", "frost"]

    return selected_filters


def collect_input_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported input file format: {input_path.suffix}")
        return [input_path]

    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def create_labeled_tile(title: str, image: np.ndarray) -> np.ndarray:
    tile = np.clip(image, 0.0, 1.0)
    tile = (tile * 255.0).round().astype(np.uint8)
    tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        tile,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return tile


def save_comparison_grid(
    output_path: Path,
    original: np.ndarray,
    filtered_images: dict[str, np.ndarray],
) -> None:
    ordered_names = ["original", "median", "gaussian", "lee", "frost"]
    tile_images = {"original": original}
    tile_images.update(filtered_images)
    tiles = [create_labeled_tile(name, tile_images[name]) for name in ordered_names]
    comparison_grid = cv2.hconcat(tiles)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), comparison_grid)
    if not success:
        raise IOError(f"Failed to write comparison grid: {output_path}")


def apply_filters(image: np.ndarray, selected_filters: list[str]) -> dict[str, np.ndarray]:
    results: dict[str, np.ndarray] = {}
    if "median" in selected_filters:
        results["median"] = median_filter(image)
    if "gaussian" in selected_filters:
        results["gaussian"] = gaussian_filter(image)
    if "lee" in selected_filters:
        results["lee"] = lee_filter(image)
    if "frost" in selected_filters:
        results["frost"] = frost_filter(image)
    return results


def ensure_all_grid_entries(
    original: np.ndarray,
    filtered_images: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    completed_images = dict(filtered_images)
    for filter_name in ("median", "gaussian", "lee", "frost"):
        if filter_name not in completed_images:
            completed_images[filter_name] = original.copy()
    return completed_images


def process_image(image_path: Path, output_dir: Path, selected_filters: list[str]) -> None:
    image = read_grayscale_image(image_path)
    filtered_images = apply_filters(image, selected_filters)

    for filter_name, filtered_image in filtered_images.items():
        output_path = output_dir / f"{image_path.stem}_{filter_name}.png"
        save_grayscale_image(output_path, filtered_image)

    comparison_images = ensure_all_grid_entries(image, filtered_images)
    comparison_path = output_dir / f"{image_path.stem}_baseline_comparison.png"
    save_comparison_grid(comparison_path, image, comparison_images)


def main() -> None:
    args = parse_args()
    validate_args(args)
    selected_filters = resolve_filters(args)
    input_images = collect_input_images(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not input_images:
        print(f"No supported images found in: {args.input}")
        return

    for image_path in input_images:
        process_image(image_path, args.output_dir, selected_filters)
        print(f"Processed: {image_path}")


if __name__ == "__main__":
    main()
