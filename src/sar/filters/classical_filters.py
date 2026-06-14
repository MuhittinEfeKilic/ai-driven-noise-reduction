from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_grayscale_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read grayscale image: {image_path}")
    return image.astype(np.float32) / 255.0


def save_grayscale_image(path: str | Path, image: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = np.clip(image, 0.0, 1.0)
    image_uint8 = (image_uint8 * 255.0).round().astype(np.uint8)
    if not cv2.imwrite(str(output_path), image_uint8):
        raise IOError(f"Failed to write image: {output_path}")


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return cv2.medianBlur(image_uint8, kernel_size).astype(np.float32) / 255.0


def gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(
        np.clip(image, 0.0, 1.0).astype(np.float32),
        (kernel_size, kernel_size),
        sigmaX=sigma,
    )


def lee_filter(image: np.ndarray, window_size: int = 5, noise_variance: float | None = None) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    mean = cv2.blur(image, (window_size, window_size))
    mean_sq = cv2.blur(image * image, (window_size, window_size))
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    if noise_variance is None:
        noise_variance = float(np.mean(variance))
    weights = variance / (variance + float(noise_variance) + 1e-8)
    return np.clip(mean + weights * (image - mean), 0.0, 1.0)


def frost_filter(image: np.ndarray, window_size: int = 5, damping: float = 1.0) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    mean = cv2.blur(image, (window_size, window_size))
    mean_sq = cv2.blur(image * image, (window_size, window_size))
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    coefficient = variance / (mean * mean + 1e-8)
    alpha = damping * coefficient
    return np.clip(mean + np.exp(-alpha) * (image - mean), 0.0, 1.0)
