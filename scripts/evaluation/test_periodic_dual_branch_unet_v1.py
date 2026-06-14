from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as TF
from tqdm import tqdm

from src.architectures.frequency_utils import rgb_to_normalized_fft_log_magnitude
from src.architectures.periodic_dual_branch_unet import PeriodicDualBranchResidualUNet


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PERIODIC_V3_PATTERN = re.compile(r"^(?P<base>.+)_pv3_(?P<variant>\d+)$")
RESIZE_SIZE = 160
CROP_SIZE = 128


@dataclass(frozen=True)
class Sample:
    noisy_path: Path
    clean_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PeriodicDualBranchResidualUNet checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("data/synthetic/periodic_v3"))
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean/bsd500"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/periodic_dual_branch_unet_v1"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-comparison", action="store_true", default=True)
    parser.add_argument("--no-save-comparison", dest="save_comparison", action="store_false")
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def noisy_to_clean_stem(noisy_stem: str) -> str:
    match = PERIODIC_V3_PATTERN.match(noisy_stem)
    return match.group("base") if match else noisy_stem


def list_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    images = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in: {input_dir}")
    return images


def build_clean_index(clean_dir: Path) -> dict[str, Path]:
    if not clean_dir.exists():
        return {}
    return {
        path.stem: path
        for path in sorted(clean_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }


def collect_samples(input_dir: Path, clean_dir: Path, limit: int = 0) -> list[Sample]:
    clean_index = build_clean_index(clean_dir)
    noisy_images = list_images(input_dir)
    if limit > 0:
        noisy_images = noisy_images[:limit]

    samples: list[Sample] = []
    for noisy_path in noisy_images:
        clean_path = clean_index.get(noisy_to_clean_stem(noisy_path.stem))
        samples.append(Sample(noisy_path=noisy_path, clean_path=clean_path))
    return samples


def preprocess_image(image: Image.Image) -> Tensor:
    image = TF.resize(image, [RESIZE_SIZE, RESIZE_SIZE], interpolation=TF.InterpolationMode.BILINEAR)
    image = TF.center_crop(image, [CROP_SIZE, CROP_SIZE])
    return TF.to_tensor(image)


def load_rgb_tensor(path: Path, device: torch.device) -> Tensor:
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
    except OSError as exc:
        raise ValueError(f"Could not open image file: {path}") from exc
    tensor = preprocess_image(rgb).unsqueeze(0).to(device)
    return tensor


def tensor_to_uint8_image(tensor: Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(np.rint(array * 255.0), 0.0, 255.0).astype(np.uint8)


def save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def build_comparison_image(noisy: np.ndarray, denoised: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return np.concatenate([noisy, denoised, clean], axis=1)


def psnr_score(target: np.ndarray, prediction: np.ndarray) -> float:
    target_f = target.astype(np.float32)
    prediction_f = prediction.astype(np.float32)
    mse = float(np.mean((target_f - prediction_f) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def _gaussian_window(window_size: int = 11, sigma: float = 1.5) -> Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    window_2d = torch.outer(kernel, kernel)
    return window_2d.unsqueeze(0).unsqueeze(0)


def ssim_score(target: np.ndarray, prediction: np.ndarray) -> float:
    target_t = torch.from_numpy(target.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    prediction_t = torch.from_numpy(prediction.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    c1 = 0.01**2
    c2 = 0.03**2
    channels = target_t.shape[1]
    window = _gaussian_window().expand(channels, 1, -1, -1)
    padding = window.shape[-1] // 2

    mu_x = F.conv2d(prediction_t, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target_t, window, padding=padding, groups=channels)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x = F.conv2d(prediction_t * prediction_t, window, padding=padding, groups=channels) - mu_x2
    sigma_y = F.conv2d(target_t * target_t, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(prediction_t * target_t, window, padding=padding, groups=channels) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    return float((numerator / (denominator + 1e-8)).mean().item())


def load_model(checkpoint_path: Path, device: torch.device) -> PeriodicDualBranchResidualUNet:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path.resolve(), map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        architecture = checkpoint.get("architecture", {})
    else:
        state_dict = checkpoint
        architecture = {}

    model = PeriodicDualBranchResidualUNet(
        rgb_in_channels=int(architecture.get("rgb_in_channels", 3)),
        freq_in_channels=int(architecture.get("freq_in_channels", 1)),
        out_channels=int(architecture.get("out_channels", 3)),
        base_features=int(architecture.get("base_features", 32)),
        frequency_base_features=int(architecture.get("frequency_base_features", 16)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict_clean(
    model: PeriodicDualBranchResidualUNet,
    noisy_tensor: Tensor,
    alpha: float,
) -> Tensor:
    x_freq = rgb_to_normalized_fft_log_magnitude(noisy_tensor)
    predicted_residual = model(noisy_tensor, x_freq)
    return torch.clamp(noisy_tensor - alpha * predicted_residual, min=0.0, max=1.0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)
    samples = collect_samples(args.input_dir.resolve(), args.clean_dir.resolve(), limit=args.limit)

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    matched_clean = 0

    for sample in tqdm(samples, desc="Evaluating periodic dual-branch UNet", unit="image"):
        noisy_tensor = load_rgb_tensor(sample.noisy_path, device)
        pred_clean = predict_clean(model, noisy_tensor, alpha=args.alpha)

        noisy_image = tensor_to_uint8_image(noisy_tensor)
        denoised_image = tensor_to_uint8_image(pred_clean)

        denoised_path = output_dir / f"{sample.noisy_path.stem}_denoised.png"
        save_image(denoised_image, denoised_path)

        if sample.clean_path is not None:
            clean_tensor = load_rgb_tensor(sample.clean_path, device)
            clean_image = tensor_to_uint8_image(clean_tensor)
            psnr_values.append(psnr_score(clean_image, denoised_image))
            ssim_values.append(ssim_score(clean_image, denoised_image))
            matched_clean += 1

            if args.save_comparison:
                comparison = build_comparison_image(noisy_image, denoised_image, clean_image)
                comparison_path = output_dir / f"{sample.noisy_path.stem}_comparison.png"
                save_image(comparison, comparison_path)

    print(f"Processed images: {len(samples)}")
    print(f"Matched clean images: {matched_clean}")
    if matched_clean > 0:
        avg_psnr = float(np.mean(psnr_values))
        avg_ssim = float(np.mean(ssim_values))
        print(f"Average PSNR: {avg_psnr:.4f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
    else:
        print("Average PSNR: N/A")
        print("Average SSIM: N/A")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
