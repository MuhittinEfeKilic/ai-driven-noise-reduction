from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import rasterio
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "rasterio is required for SAR ENVI inference. Install it with: pip install rasterio"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "sar" / "training"
if str(TRAINING_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_SCRIPT_DIR))

from train_noise_map_sar import NoiseMapUNet  # noqa: E402


EPSILON = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAR noise-map inference.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-original", type=Path, required=True)
    parser.add_argument("--input-esa-cleared", type=Path, required=True)
    parser.add_argument("--input-incidence", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sar/noise_map_inference"))
    parser.add_argument("--use-incidence-scalar", action="store_true")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--log-transform", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.model_path, args.input_original, args.input_esa_cleared):
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
    if args.use_incidence_scalar and args.input_incidence is None:
        raise ValueError("--input-incidence is required when --use-incidence-scalar is set.")
    if args.input_incidence is not None and not args.input_incidence.exists():
        raise FileNotFoundError(f"Missing incidence file: {args.input_incidence}")
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive.")
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")


def normalize_to_unit(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value - min_value <= EPSILON:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - min_value) / (max_value - min_value)).astype(np.float32)


def percentile_normalize(image: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    lower_value = float(np.percentile(image, lower))
    upper_value = float(np.percentile(image, upper))
    if upper_value - lower_value <= EPSILON:
        return normalize_to_unit(image)
    return np.clip((image - lower_value) / (upper_value - lower_value), 0.0, 1.0).astype(np.float32)


def load_envi_image(path: Path, use_log_transform: bool) -> np.ndarray:
    with rasterio.open(path) as dataset:
        image = dataset.read(1).astype(np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = normalize_to_unit(image)
    if use_log_transform:
        image = np.log1p(image)
        image = normalize_to_unit(image)
    return image.astype(np.float32)


def pad_for_sliding_window(image: np.ndarray, patch_size: int, stride: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape
    pad_height = max(0, patch_size - height)
    pad_width = max(0, patch_size - width)
    if height > patch_size:
        remainder_h = (height - patch_size) % stride
        if remainder_h:
            pad_height += stride - remainder_h
    if width > patch_size:
        remainder_w = (width - patch_size) % stride
        if remainder_w:
            pad_width += stride - remainder_w
    padded = np.pad(image, ((0, pad_height), (0, pad_width)), mode="reflect")
    return padded, (height, width)


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    success = cv2.imwrite(str(path), image_uint8)
    if not success:
        raise IOError(f"Failed to write image: {path}")


def load_model(model_path: Path, use_incidence_scalar: bool, device: torch.device) -> NoiseMapUNet:
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_use_incidence_scalar = bool(checkpoint.get("use_incidence_scalar", False))
    if checkpoint_use_incidence_scalar != use_incidence_scalar:
        raise ValueError(
            "Checkpoint incidence-scalar configuration does not match CLI. "
            f"checkpoint={checkpoint_use_incidence_scalar}, cli={use_incidence_scalar}"
        )
    input_channels = 2 if use_incidence_scalar else 1
    checkpoint_input_channels = int(checkpoint.get("input_channels", input_channels))
    if checkpoint_input_channels != input_channels:
        raise ValueError(
            "Checkpoint input channel count does not match CLI. "
            f"checkpoint={checkpoint_input_channels}, cli={input_channels}"
        )
    model = NoiseMapUNet(in_channels=input_channels, out_channels=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def sliding_window_predict_noise(
    model: NoiseMapUNet,
    esa_noise: np.ndarray,
    incidence: np.ndarray | None,
    patch_size: int,
    stride: int,
    device: torch.device,
) -> np.ndarray:
    padded_noise, original_shape = pad_for_sliding_window(esa_noise, patch_size, stride)
    padded_incidence = None
    if incidence is not None:
        padded_incidence, _ = pad_for_sliding_window(incidence, patch_size, stride)
        if padded_incidence.shape != padded_noise.shape:
            raise ValueError("Incidence shape does not match noise map shape after padding.")

    height, width = padded_noise.shape
    output_sum = np.zeros((height, width), dtype=np.float32)
    output_count = np.zeros((height, width), dtype=np.float32)

    for top in range(0, height - patch_size + 1, stride):
        for left in range(0, width - patch_size + 1, stride):
            noise_patch = padded_noise[top : top + patch_size, left : left + patch_size]
            channels = [noise_patch]
            if padded_incidence is not None:
                incidence_patch = padded_incidence[top : top + patch_size, left : left + patch_size]
                scalar_map = np.full_like(noise_patch, float(incidence_patch.mean()), dtype=np.float32)
                channels.append(scalar_map)
            stacked = np.stack(channels, axis=0).astype(np.float32)
            input_tensor = torch.from_numpy(stacked).unsqueeze(0).to(device)
            prediction = model(input_tensor).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            output_sum[top : top + patch_size, left : left + patch_size] += prediction
            output_count[top : top + patch_size, left : left + patch_size] += 1.0

    merged = output_sum / np.maximum(output_count, 1.0)
    original_height, original_width = original_shape
    return merged[:original_height, :original_width]


def main() -> None:
    args = parse_args()
    validate_args(args)

    original = load_envi_image(args.input_original, args.log_transform)
    esa_cleared = load_envi_image(args.input_esa_cleared, args.log_transform)
    if original.shape != esa_cleared.shape:
        raise ValueError(f"Shape mismatch: original={original.shape}, esa={esa_cleared.shape}")

    incidence = None
    if args.use_incidence_scalar:
        incidence = load_envi_image(args.input_incidence, use_log_transform=False)
        if incidence.shape != original.shape:
            raise ValueError(
                f"Incidence shape {incidence.shape} does not match original shape {original.shape}."
            )

    esa_noise = (original - esa_cleared).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, args.use_incidence_scalar, device)
    predicted_noise = sliding_window_predict_noise(
        model=model,
        esa_noise=esa_noise,
        incidence=incidence,
        patch_size=args.patch_size,
        stride=args.stride,
        device=device,
    )
    denoised = np.clip(original - predicted_noise, 0.0, 1.0).astype(np.float32)
    difference_map = np.abs(original - denoised).astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_png(args.output_dir / "predicted_noise_map.png", percentile_normalize(predicted_noise))
    save_png(args.output_dir / "denoised.png", percentile_normalize(denoised))
    save_png(args.output_dir / "difference_map.png", percentile_normalize(difference_map))

    print(f"Input shape: {original.shape}")
    print(f"Saved predicted noise map: {args.output_dir / 'predicted_noise_map.png'}")
    print(f"Saved denoised image: {args.output_dir / 'denoised.png'}")
    print(f"Saved difference map: {args.output_dir / 'difference_map.png'}")


if __name__ == "__main__":
    main()
