from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.inference.classifier_inference import ClassifierInference


def load_trained_classifier(
    model_path: str | Path = "models/classifiers/noise_classifier_best.pt",
    device: str | None = None,
) -> ClassifierInference:
    classifier = ClassifierInference(model_path=model_path, device=device)
    classifier.load_model()
    return classifier


def classify_noise_model(
    image: Image.Image,
    model_path: str | Path = "models/classifiers/noise_classifier_best.pt",
    device: str | None = None,
) -> tuple[str, float]:
    classifier = load_trained_classifier(model_path=model_path, device=device)
    return classifier.predict(image)


def classify_noise(
    image_path: str | Path,
    model_path: str | Path = "models/classifiers/noise_classifier_best.pt",
    device: str | None = None,
) -> tuple[str, float]:
    classifier = load_trained_classifier(model_path=model_path, device=device)
    return classifier.predict_from_path(image_path)
