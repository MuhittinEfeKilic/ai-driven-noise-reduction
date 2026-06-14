from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from src.architectures.unet import UNet
from src.inference.denoiser_inference import DenoiserInference
from src.metrics.psnr import psnr_score
from src.metrics.ssim import ssim_score


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RESIZE_SIZE = 160
CROP_SIZE = 128
SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 2
OUTPUT_ROOT = Path("outputs/speckle_compare")
LINEAR_CHECKPOINT = Path("archive_models/denoisers/speckle/archive/speckle_unet_best.pt")
LOG_CHECKPOINT = Path("archive_models/denoisers/speckle/archive/speckle_unet_log_best.pt")
CLEAN_DIR = Path("data/clean/bsd500")
NOISY_DIR = Path("data/synthetic/speckle")


@dataclass(frozen=True)
class PairedSample:
    clean_path: Path
    noisy_path: Path


class SpeckleTestDataset(Dataset[tuple[Tensor, Tensor, str, str, str]]):
    def __init__(self, samples: list[PairedSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str, str, str]:
        sample = self.samples[index]
        clean = self._load_rgb_image(sample.clean_path)
        noisy = self._load_rgb_image(sample.noisy_path)
        clean, noisy = self._apply_eval_transforms(clean, noisy)
        return noisy, clean, sample.noisy_path.name, str(sample.noisy_path), str(sample.clean_path)

    @staticmethod
    def _load_rgb_image(path: Path) -> Image.Image:
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except OSError as exc:
            raise ValueError(f"Could not open image file: {path}") from exc

    @staticmethod
    def _apply_eval_transforms(clean: Image.Image, noisy: Image.Image) -> tuple[Tensor, Tensor]:
        clean = TF.resize(clean, [RESIZE_SIZE, RESIZE_SIZE], interpolation=InterpolationMode.BILINEAR)
        noisy = TF.resize(noisy, [RESIZE_SIZE, RESIZE_SIZE], interpolation=InterpolationMode.BILINEAR)
        clean = TF.center_crop(clean, [CROP_SIZE, CROP_SIZE])
        noisy = TF.center_crop(noisy, [CROP_SIZE, CROP_SIZE])
        return TF.to_tensor(clean), TF.to_tensor(noisy)


def build_clean_index(clean_dir: Path) -> dict[str, Path]:
    if not clean_dir.exists():
        raise FileNotFoundError(f"Clean directory not found: {clean_dir}")

    clean_index: dict[str, Path] = {}
    for path in sorted(clean_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.stem in clean_index:
            raise ValueError(f"Duplicate clean image stem detected: {path.stem}")
        clean_index[path.stem] = path

    if not clean_index:
        raise FileNotFoundError(f"No clean images found in: {clean_dir}")
    return clean_index


def collect_pairs(clean_dir: Path, noisy_dir: Path) -> list[PairedSample]:
    if not noisy_dir.exists():
        raise FileNotFoundError(f"Noisy directory not found: {noisy_dir}")

    clean_index = build_clean_index(clean_dir)
    paired_samples: list[PairedSample] = []

    for noisy_path in sorted(noisy_dir.iterdir()):
        if not noisy_path.is_file() or noisy_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        clean_path = clean_index.get(noisy_path.stem)
        if clean_path is None:
            continue
        paired_samples.append(PairedSample(clean_path=clean_path, noisy_path=noisy_path))

    if not paired_samples:
        raise FileNotFoundError("No valid speckle clean/noisy pairs were found.")
    return paired_samples


def split_samples(
    samples: list[PairedSample],
    seed: int = SEED,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[list[PairedSample], list[PairedSample], list[PairedSample]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    if len(samples) < 3:
        raise ValueError("At least 3 paired samples are required for train/val/test split.")

    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total_count = len(shuffled)
    train_count = max(1, int(total_count * train_ratio))
    val_count = max(1, int(total_count * val_ratio))
    test_count = total_count - train_count - val_count

    if test_count < 1:
        test_count = 1
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            raise ValueError("Unable to create non-empty train/val/test splits from the paired dataset.")

    train_end = train_count
    val_end = train_end + val_count
    train_samples = shuffled[:train_end]
    val_samples = shuffled[train_end:val_end]
    test_samples = shuffled[val_end:]

    if not train_samples or not val_samples or not test_samples:
        raise ValueError("Split produced an empty train, val, or test set. Increase dataset size.")
    return train_samples, val_samples, test_samples


def create_test_loader() -> DataLoader:
    paired_samples = collect_pairs(CLEAN_DIR.resolve(), NOISY_DIR.resolve())
    _, _, test_samples = split_samples(paired_samples, seed=SEED)
    dataset = SpeckleTestDataset(test_samples)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def run_inference_batch(inference: DenoiserInference, noisy_batch: Tensor) -> Tensor:
    if inference.model is None:
        inference.load_model()
    assert inference.model is not None

    noisy_batch = noisy_batch.to(inference.device, non_blocking=True)
    return inference._run_model(noisy_batch)


@dataclass
class SampleMetric:
    name: str
    noisy: np.ndarray
    output: np.ndarray
    clean: np.ndarray
    psnr: float
    ssim: float


@dataclass
class EvalResult:
    name: str
    avg_psnr: float
    avg_ssim: float
    samples: list[SampleMetric]


class ResidualDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels:
            self.projection: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.projection = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.projection(x)
        return self.activation(self.block(x) + residual)


class ResidualDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualDoubleConv(in_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResidualUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResidualDoubleConv((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = TF.resize(x, list(skip.shape[-2:]), interpolation=InterpolationMode.BILINEAR)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResidualCompatibleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_features: int = 64) -> None:
        super().__init__()
        features = base_features
        self.inc = ResidualDoubleConv(in_channels, features)
        self.down1 = ResidualDownBlock(features, features * 2)
        self.down2 = ResidualDownBlock(features * 2, features * 4)
        self.down3 = ResidualDownBlock(features * 4, features * 8)
        self.down4 = ResidualDownBlock(features * 8, features * 16)
        self.up1 = ResidualUpBlock(features * 16, features * 8, features * 8)
        self.up2 = ResidualUpBlock(features * 8, features * 4, features * 4)
        self.up3 = ResidualUpBlock(features * 4, features * 2, features * 2)
        self.up4 = ResidualUpBlock(features * 2, features, features)
        self.outc = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def tensor_to_uint8_image(tensor: Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return np.clip(np.rint(array * 255.0), 0.0, 255.0).astype(np.uint8)


def load_model_bundle(checkpoint_path: Path, device: str) -> tuple[nn.Module, str]:
    inference = DenoiserInference(model_path=checkpoint_path, device=device)
    try:
        model = inference.load_model()
        return model, inference.training_domain
    except RuntimeError as exc:
        if "projection" not in str(exc):
            raise

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = DenoiserInference._extract_state_dict(checkpoint)
    _, architecture, training_domain, _ = inference._resolve_architecture(checkpoint, state_dict)

    model = ResidualCompatibleUNet(
        in_channels=int(architecture["in_channels"]),
        out_channels=int(architecture["out_channels"]),
        base_features=int(architecture["base_features"]),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, training_domain


@torch.no_grad()
def run_model_bundle(model: nn.Module, noisy_batch: Tensor, training_domain: str) -> Tensor:
    noisy_batch = noisy_batch.to(next(model.parameters()).device, non_blocking=True)
    if isinstance(model, (UNet, ResidualCompatibleUNet)):
        if training_domain == "log1p":
            prediction_log = model(torch.log1p(torch.clamp(noisy_batch, min=0.0, max=1.0)))
            return torch.clamp(torch.expm1(prediction_log), min=0.0, max=1.0)
        return torch.clamp(model(noisy_batch), min=0.0, max=1.0)
    raise ValueError(f"Unsupported model type for comparison: {type(model).__name__}")


def evaluate_model(model_name: str, checkpoint_path: Path, loader: DataLoader, device: str) -> EvalResult:
    model, training_domain = load_model_bundle(checkpoint_path, device)

    metrics: list[SampleMetric] = []
    progress = tqdm(loader, desc=f"Evaluating {model_name}", unit="batch")

    for noisy_batch, clean_batch, names, _, _ in progress:
        output_batch = run_model_bundle(model, noisy_batch, training_domain)

        for idx, name in enumerate(names):
            noisy_image = tensor_to_uint8_image(noisy_batch[idx])
            output_image = tensor_to_uint8_image(output_batch[idx])
            clean_image = tensor_to_uint8_image(clean_batch[idx])
            psnr = psnr_score(clean_image, output_image)
            ssim = ssim_score(clean_image, output_image)
            metrics.append(
                SampleMetric(
                    name=name,
                    noisy=noisy_image,
                    output=output_image,
                    clean=clean_image,
                    psnr=psnr,
                    ssim=ssim,
                )
            )

    avg_psnr = float(np.mean([item.psnr for item in metrics])) if metrics else 0.0
    avg_ssim = float(np.mean([item.ssim for item in metrics])) if metrics else 0.0
    return EvalResult(name=model_name, avg_psnr=avg_psnr, avg_ssim=avg_ssim, samples=metrics)


def save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_ranked_examples(result: EvalResult, output_root: Path, top_k: int = 5) -> None:
    model_dir = output_root / result.name.lower()
    best_samples = sorted(result.samples, key=lambda item: item.psnr, reverse=True)[:top_k]
    worst_samples = sorted(result.samples, key=lambda item: item.psnr)[:top_k]

    for group_name, samples in (("best", best_samples), ("worst", worst_samples)):
        for rank, sample in enumerate(samples, start=1):
            sample_stem = Path(sample.name).stem
            sample_dir = model_dir / group_name / f"{rank:02d}_{sample_stem}"
            save_image(sample.noisy, sample_dir / "noisy.png")
            save_image(sample.output, sample_dir / "output.png")
            save_image(sample.clean, sample_dir / "clean.png")


def print_summary(results: list[EvalResult]) -> None:
    print()
    print("Model | Avg PSNR | Avg SSIM")
    print("--------------------------------")
    for result in results:
        print(f"{result.name:<6} | {result.avg_psnr:8.4f} | {result.avg_ssim:8.4f}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = create_test_loader()

    linear_result = evaluate_model("Linear", LINEAR_CHECKPOINT.resolve(), test_loader, device)
    log_result = evaluate_model("Log", LOG_CHECKPOINT.resolve(), test_loader, device)

    save_ranked_examples(linear_result, OUTPUT_ROOT.resolve())
    save_ranked_examples(log_result, OUTPUT_ROOT.resolve())
    print_summary([linear_result, log_result])


if __name__ == "__main__":
    main()
