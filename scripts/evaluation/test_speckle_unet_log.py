from __future__ import annotations

import argparse
from pathlib import Path

from src.inference.denoiser_inference import DenoiserInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run log-domain Speckle U-Net denoising on a single image.")
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("archive_models/denoisers/speckle/archive/speckle_unet_log_best.pt"),
        help="Path to the trained Speckle log-domain U-Net checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Directory where the denoised output will be saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device override, for example 'cpu' or 'cuda'.",
    )
    return parser.parse_args()


def build_output_path(image_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{image_path.stem}_speckle_unet_log.png"


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    inference = DenoiserInference(model_path=args.checkpoint, device=args.device)
    denoised_image = inference.run(image_path)

    output_path = build_output_path(image_path, args.output_dir.expanduser().resolve())
    saved_path = inference.save_output(denoised_image, output_path)
    print(f"Saved denoised image to: {saved_path}")


if __name__ == "__main__":
    main()
