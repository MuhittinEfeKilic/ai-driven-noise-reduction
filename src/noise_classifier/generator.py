from __future__ import annotations

import numpy as np
from PIL import Image


def _to_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    return array.astype(np.uint8, copy=False)


def _restore_type(array: np.ndarray, original: Image.Image | np.ndarray) -> Image.Image | np.ndarray:
    array = np.clip(array, 0, 255).astype(np.uint8)
    if isinstance(original, Image.Image):
        return Image.fromarray(array)
    return array


def add_gaussian_noise(
    image: Image.Image | np.ndarray,
    sigma: float = 25.0,
    rng: np.random.Generator | None = None,
) -> Image.Image | np.ndarray:
    rng = rng or np.random.default_rng()
    array = _to_array(image).astype(np.float32)
    noisy = array + rng.normal(0.0, sigma, size=array.shape).astype(np.float32)
    return _restore_type(noisy, image)


def add_salt_pepper_noise(
    image: Image.Image | np.ndarray,
    amount: float = 0.03,
    salt_ratio: float = 0.5,
    rng: np.random.Generator | None = None,
) -> Image.Image | np.ndarray:
    rng = rng or np.random.default_rng()
    noisy = _to_array(image).copy()
    height, width = noisy.shape[:2]
    total_pixels = height * width
    salt_count = int(total_pixels * amount * salt_ratio)
    pepper_count = int(total_pixels * amount * (1.0 - salt_ratio))

    if salt_count > 0:
        ys = rng.integers(0, height, size=salt_count)
        xs = rng.integers(0, width, size=salt_count)
        noisy[ys, xs] = 255
    if pepper_count > 0:
        ys = rng.integers(0, height, size=pepper_count)
        xs = rng.integers(0, width, size=pepper_count)
        noisy[ys, xs] = 0
    return _restore_type(noisy, image)


def add_speckle_noise(
    image: Image.Image | np.ndarray,
    std: float = 0.18,
    rng: np.random.Generator | None = None,
) -> Image.Image | np.ndarray:
    rng = rng or np.random.default_rng()
    array = _to_array(image).astype(np.float32) / 255.0
    noisy = array + array * rng.normal(0.0, std, size=array.shape).astype(np.float32)
    return _restore_type(noisy * 255.0, image)


def add_periodic_noise(
    image: Image.Image | np.ndarray,
    amplitude: float = 32.0,
    frequency: float = 0.055,
    rng: np.random.Generator | None = None,
) -> Image.Image | np.ndarray:
    rng = rng or np.random.default_rng()
    array = _to_array(image).astype(np.float32)
    height, width = array.shape[:2]
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    angle = float(rng.choice([0.0, np.pi / 2.0, np.pi / 4.0, 3.0 * np.pi / 4.0]))
    direction = np.cos(angle) * xx + np.sin(angle) * yy
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    pattern = amplitude * np.sin(2.0 * np.pi * frequency * direction + phase)
    return _restore_type(array + pattern[..., None], image)
