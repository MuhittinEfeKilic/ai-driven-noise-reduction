from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.pipeline.run_pipeline import NoiseReductionPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the noise reduction CLI."""
    parser = argparse.ArgumentParser(description="AI Driven Noise Reduction for Images")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory where the denoised image will be saved.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the CLI pipeline and print a formatted summary."""
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()

    if not image_path.exists():
        print(f"Error: input image not found: {image_path}", file=sys.stderr)
        return 1

    pipeline = NoiseReductionPipeline(output_dir=args.output_dir)

    try:
        result = pipeline.run(image_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Runtime error: {exc}", file=sys.stderr)
        return 1

    confidence_pct = round(float(result["classifier_confidence"]) * 100.0, 2)
    print("Noise Reduction Result")
    print(f"Predicted noise type : {result['predicted_noise_type']}")
    print(f"Classifier confidence: {confidence_pct:.2f}%")
    print(f"Selected model type  : {result['denoiser_model_type']}")
    print(f"Selected denoiser    : {result['denoiser_path']}")
    if result.get("postprocessing_applied"):
        print(f"Applied postprocessing : {result['postprocessing_applied']}")
    print(f"Output path          : {result['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
