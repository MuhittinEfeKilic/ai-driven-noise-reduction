from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.inference.classifier_inference import ClassifierInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug a single image with the noise classifier.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/classifiers/noise_classifier_best.pt"),
        help="Path to the classifier checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of class probabilities to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    inference = ClassifierInference(model_path=args.checkpoint, device=args.device)
    inference.load_model()
    assert inference.model is not None

    with Image.open(image_path) as image:
        input_tensor = inference.preprocess_image(image)

    with torch.no_grad():
        logits = inference.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    top_k = min(args.top_k, probabilities.numel())
    scores, indices = probabilities.topk(top_k)

    predicted_label = inference.LABELS[int(indices[0].item())]
    confidence = float(scores[0].item())

    print(f"Image: {image_path}")
    print(f"Predicted label: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")
    print("Top probabilities:")
    for rank, (score, index) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
        print(f"- {rank}. {inference.LABELS[int(index)]}: {float(score):.4f}")


if __name__ == "__main__":
    main()
