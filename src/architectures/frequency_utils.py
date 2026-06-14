from __future__ import annotations

import torch
from torch import Tensor


def rgb_to_normalized_fft_log_magnitude(image: Tensor, eps: float = 1e-8) -> Tensor:
    """Return a normalized grayscale FFT log-magnitude map for an RGB batch.

    Args:
        image: Tensor with shape ``[B, C, H, W]`` and values normally in ``[0, 1]``.
        eps: Small denominator guard used during min-max normalization.
    """
    if image.ndim != 4:
        raise ValueError("Expected image tensor with shape [B, C, H, W].")
    if image.shape[1] < 1:
        raise ValueError("Expected at least one image channel.")

    if image.shape[1] == 1:
        grayscale = image
    else:
        red = image[:, 0:1]
        green = image[:, 1:2]
        blue = image[:, 2:3]
        grayscale = 0.299 * red + 0.587 * green + 0.114 * blue

    spectrum = torch.fft.fftshift(torch.fft.fft2(grayscale, norm="ortho"), dim=(-2, -1))
    magnitude = torch.log1p(torch.abs(spectrum))
    min_value = magnitude.amin(dim=(-2, -1), keepdim=True)
    max_value = magnitude.amax(dim=(-2, -1), keepdim=True)
    return (magnitude - min_value) / torch.clamp(max_value - min_value, min=eps)
