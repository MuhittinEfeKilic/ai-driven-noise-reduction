from __future__ import annotations

import math

import cv2
import numpy as np


def _validate_uint8_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray.")
    if image.ndim not in (2, 3):
        raise ValueError("image must have shape (H, W) or (H, W, C).")
    if image.ndim == 3 and image.shape[2] not in (1, 3):
        raise ValueError("RGB input must have 3 channels or a single trailing channel.")

    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def _validate_magnitude_spectrum(magnitude_spectrum: np.ndarray) -> np.ndarray:
    if not isinstance(magnitude_spectrum, np.ndarray):
        raise TypeError("magnitude_spectrum must be a numpy.ndarray.")
    if magnitude_spectrum.ndim != 2:
        raise ValueError("magnitude_spectrum must have shape (H, W).")
    if magnitude_spectrum.size == 0:
        raise ValueError("magnitude_spectrum must not be empty.")
    return magnitude_spectrum.astype(np.float32, copy=False)


def _validate_noise_points(noise_points: list[tuple[int, int]] | np.ndarray) -> list[tuple[int, int]]:
    if isinstance(noise_points, np.ndarray):
        if noise_points.ndim != 2 or noise_points.shape[1] != 2:
            raise ValueError("noise_points array must have shape (N, 2).")
        return [(int(row), int(col)) for row, col in noise_points.tolist()]
    return [(int(row), int(col)) for row, col in noise_points]


def detect_noise_points(
    magnitude_spectrum: np.ndarray,
    threshold_ratio: float = 0.3,
    min_distance: int = 5,
) -> list[tuple[int, int]]:
    """Detect bright periodic-noise peaks in a centered FFT magnitude spectrum."""
    spectrum = _validate_magnitude_spectrum(magnitude_spectrum)
    if not 0.0 < threshold_ratio <= 1.0:
        raise ValueError("threshold_ratio must be in the range (0, 1].")
    if min_distance < 0:
        raise ValueError("min_distance must be non-negative.")

    height, width = spectrum.shape
    center_row, center_col = height // 2, width // 2

    # Suppress the DC peak so thresholding focuses on periodic spike candidates.
    working = spectrum.copy()
    cv2.circle(working, (center_col, center_row), max(1, min_distance), 0.0, thickness=-1)

    max_value = float(working.max())
    if max_value <= 0.0:
        return []

    threshold_value = max_value * float(threshold_ratio)
    thresholded = np.where(working >= threshold_value, 255, 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded, connectivity=8)

    candidates: list[tuple[float, int, int]] = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        centroid_col, centroid_row = centroids[label_idx]
        row = int(round(centroid_row))
        col = int(round(centroid_col))
        if not (0 <= row < height and 0 <= col < width):
            continue
        if math.hypot(row - center_row, col - center_col) < float(min_distance):
            continue
        candidates.append((float(spectrum[row, col]), row, col))

    if not candidates:
        flat_indices = np.argsort(working, axis=None)[::-1]
        max_fallback_points = 8
        fallback_points = 0
        for flat_index in flat_indices:
            row, col = np.unravel_index(int(flat_index), working.shape)
            if float(working[row, col]) <= 0.0:
                break
            if math.hypot(row - center_row, col - center_col) < float(min_distance):
                continue
            candidates.append((float(spectrum[row, col]), int(row), int(col)))
            fallback_points += 1
            if fallback_points >= max_fallback_points:
                break

    candidates.sort(reverse=True)
    selected: list[tuple[int, int]] = []
    for _, row, col in candidates:
        if all(math.hypot(row - prev_row, col - prev_col) >= float(min_distance) for prev_row, prev_col in selected):
            selected.append((row, col))
    return selected


def create_band_reject_filter(
    shape: tuple[int, int],
    noise_points: list[tuple[int, int]] | np.ndarray,
    filter_radius: int = 5,
) -> np.ndarray:
    """Create a circular notch mask for detected periodic-noise points."""
    if len(shape) != 2:
        raise ValueError("shape must be a tuple of (height, width).")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("shape dimensions must be positive.")
    if filter_radius < 0:
        raise ValueError("filter_radius must be non-negative.")

    points = _validate_noise_points(noise_points)
    mask = np.ones((height, width), dtype=np.float32)
    center_row, center_col = height // 2, width // 2

    for row, col in points:
        if not (0 <= row < height and 0 <= col < width):
            continue

        mirror_row = int(np.clip(2 * center_row - row, 0, height - 1))
        mirror_col = int(np.clip(2 * center_col - col, 0, width - 1))

        cv2.circle(mask, (col, row), filter_radius, 0.0, thickness=-1)
        cv2.circle(mask, (mirror_col, mirror_row), filter_radius, 0.0, thickness=-1)

    return mask


def _apply_frequency_filter_single_channel(channel: np.ndarray, filter_mask: np.ndarray) -> np.ndarray:
    fft = np.fft.fft2(channel.astype(np.float32, copy=False))
    fft_shifted = np.fft.fftshift(fft)
    filtered_shifted = fft_shifted * filter_mask
    filtered = np.fft.ifft2(np.fft.ifftshift(filtered_shifted))
    return np.clip(np.real(filtered), 0.0, 255.0).astype(np.uint8)


def apply_frequency_filter(image: np.ndarray, filter_mask: np.ndarray) -> np.ndarray:
    """Apply a notch/band-reject mask to a grayscale or RGB image."""
    image_uint8 = _validate_uint8_image(image)
    if not isinstance(filter_mask, np.ndarray):
        raise TypeError("filter_mask must be a numpy.ndarray.")
    if filter_mask.ndim != 2:
        raise ValueError("filter_mask must have shape (H, W).")
    if filter_mask.shape != image_uint8.shape[:2]:
        raise ValueError("filter_mask shape must match the image spatial dimensions.")

    mask = filter_mask.astype(np.float32, copy=False)
    if image_uint8.ndim == 2:
        return _apply_frequency_filter_single_channel(image_uint8, mask)
    if image_uint8.shape[2] == 1:
        filtered = _apply_frequency_filter_single_channel(image_uint8[:, :, 0], mask)
        return filtered[:, :, None]

    channels = [
        _apply_frequency_filter_single_channel(image_uint8[:, :, channel_index], mask)
        for channel_index in range(image_uint8.shape[2])
    ]
    return np.stack(channels, axis=2)


def remove_periodic_noise_fft(
    image: np.ndarray,
    threshold_ratio: float = 0.08,
    min_distance: int = 10,
    filter_radius: int = 8,
) -> np.ndarray:
    """Detect and suppress periodic noise via centered FFT notch filtering."""
    image_uint8 = _validate_uint8_image(image)
    if filter_radius < 0:
        raise ValueError("filter_radius must be non-negative.")

    grayscale = image_uint8 if image_uint8.ndim == 2 else cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(grayscale.astype(np.float32, copy=False))
    fft_shifted = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shifted)

    noise_points = detect_noise_points(
        magnitude_spectrum,
        threshold_ratio=threshold_ratio,
        min_distance=min_distance,
    )
    filter_mask = create_band_reject_filter(
        magnitude_spectrum.shape,
        noise_points,
        filter_radius=filter_radius,
    )
    return apply_frequency_filter(image_uint8, filter_mask)
