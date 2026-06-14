from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.architectures.unet import UNet


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RESIZE_SIZE = 160
CROP_SIZE = 128
WORKER_BASE_SEED = 42
BEST_CHECKPOINT_NAME = "periodic_unet_residual_hybrid_best.pt"
LAST_CHECKPOINT_NAME = "periodic_unet_residual_hybrid_last.pt"
PERIODIC_V3_PATTERN = re.compile(r"^(?P<base>.+)_pv3_(?P<variant>\d+)$")


@dataclass(frozen=True)
class PairedSample:
    clean_path: Path
    noisy_path: Path


class PairedDenoisingDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, samples: list[PairedSample], train: bool, vertical_flip: bool = True) -> None:
        self.samples = samples
        self.train = train
        self.vertical_flip = vertical_flip

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[index]
        clean = self._load_rgb_image(sample.clean_path)
        noisy = self._load_rgb_image(sample.noisy_path)
        clean, noisy = self._apply_paired_transforms(clean, noisy)
        return noisy, clean

    @staticmethod
    def _load_rgb_image(path: Path) -> Image.Image:
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except OSError as exc:
            raise ValueError(f"Could not open image file: {path}") from exc

    def _apply_paired_transforms(self, clean: Image.Image, noisy: Image.Image) -> tuple[Tensor, Tensor]:
        clean = TF.resize(clean, [RESIZE_SIZE, RESIZE_SIZE], interpolation=InterpolationMode.BILINEAR)
        noisy = TF.resize(noisy, [RESIZE_SIZE, RESIZE_SIZE], interpolation=InterpolationMode.BILINEAR)

        if self.train:
            top, left, height, width = self._sample_crop_params()
            clean = TF.crop(clean, top, left, height, width)
            noisy = TF.crop(noisy, top, left, height, width)

            if random.random() < 0.5:
                clean = TF.hflip(clean)
                noisy = TF.hflip(noisy)

            if self.vertical_flip and random.random() < 0.2:
                clean = TF.vflip(clean)
                noisy = TF.vflip(noisy)
        else:
            clean = TF.center_crop(clean, [CROP_SIZE, CROP_SIZE])
            noisy = TF.center_crop(noisy, [CROP_SIZE, CROP_SIZE])

        return TF.to_tensor(clean), TF.to_tensor(noisy)

    @staticmethod
    def _sample_crop_params() -> tuple[int, int, int, int]:
        max_offset = RESIZE_SIZE - CROP_SIZE
        top = random.randint(0, max_offset)
        left = random.randint(0, max_offset)
        return top, left, CROP_SIZE, CROP_SIZE


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, c1: float = 0.01**2, c2: float = 0.03**2) -> None:
        super().__init__()
        self.window_size = window_size
        self.c1 = c1
        self.c2 = c2

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        padding = self.window_size // 2
        mu_x = F.avg_pool2d(prediction, self.window_size, stride=1, padding=padding)
        mu_y = F.avg_pool2d(target, self.window_size, stride=1, padding=padding)
        sigma_x = F.avg_pool2d(prediction * prediction, self.window_size, stride=1, padding=padding) - mu_x.pow(2)
        sigma_y = F.avg_pool2d(target * target, self.window_size, stride=1, padding=padding) - mu_y.pow(2)
        sigma_xy = F.avg_pool2d(prediction * target, self.window_size, stride=1, padding=padding) - mu_x * mu_y

        ssim_n = (2.0 * mu_x * mu_y + self.c1) * (2.0 * sigma_xy + self.c2)
        ssim_d = (mu_x.pow(2) + mu_y.pow(2) + self.c1) * (sigma_x + sigma_y + self.c2)
        return 1.0 - (ssim_n / (ssim_d + 1e-8)).mean()


class FrequencyLoss(nn.Module):
    """Frequency loss with mild radial emphasis for periodic pattern suppression."""

    @staticmethod
    def _frequency_weight(reference: Tensor) -> Tensor:
        _, _, height, width = reference.shape
        fy = torch.fft.fftfreq(height, d=1.0, device=reference.device).abs().view(1, 1, height, 1)
        fx = torch.fft.rfftfreq(width, d=1.0, device=reference.device).abs().view(1, 1, 1, width // 2 + 1)
        radius = torch.sqrt(fx * fx + fy * fy)
        radius = radius / radius.amax().clamp_min(1e-6)
        return 0.35 + radius

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        pred_fft = torch.fft.rfft2(prediction, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        pred_mag = torch.log1p(torch.abs(pred_fft))
        target_mag = torch.log1p(torch.abs(target_fft))
        weight = self._frequency_weight(prediction)
        return F.l1_loss(pred_mag * weight, target_mag * weight)


class PeriodicV4HybridLoss(nn.Module):
    def __init__(self, l1_weight: float = 0.5, ssim_weight: float = 0.15, frequency_weight: float = 0.35) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.frequency_weight = frequency_weight
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.frequency = FrequencyLoss()

    def forward(self, predicted_clean: Tensor, clean: Tensor) -> Tensor:
        l1_loss = self.l1(predicted_clean, clean)
        ssim_loss = self.ssim(predicted_clean, clean)
        frequency_loss = self.frequency(predicted_clean, clean)
        return (
            self.l1_weight * l1_loss
            + self.ssim_weight * ssim_loss
            + self.frequency_weight * frequency_loss
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Periodic U-Net denoiser v4 with more aggressive frequency-aware loss.")
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean/bsd500"))
    parser.add_argument("--noisy-dir", type=Path, default=Path("data/synthetic/periodic_v3"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/denoisers/periodic"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--base-features", type=int, default=96)
    parser.add_argument("--vertical-flip", action="store_true", default=True)
    parser.add_argument("--no-vertical-flip", dest="vertical_flip", action="store_false")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = WORKER_BASE_SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_clean_index(clean_dir: Path) -> dict[str, Path]:
    if not clean_dir.exists():
        raise FileNotFoundError(f"Clean directory not found: {clean_dir}")
    clean_index: dict[str, Path] = {}
    for path in sorted(clean_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            if path.stem in clean_index:
                raise ValueError(f"Duplicate clean image stem detected: {path.stem}")
            clean_index[path.stem] = path
    if not clean_index:
        raise FileNotFoundError(f"No clean images found in: {clean_dir}")
    return clean_index


def noisy_to_clean_stem(noisy_stem: str) -> str:
    match = PERIODIC_V3_PATTERN.match(noisy_stem)
    return match.group("base") if match else noisy_stem


def collect_pairs(clean_dir: Path, noisy_dir: Path) -> list[PairedSample]:
    if not noisy_dir.exists():
        raise FileNotFoundError(f"Noisy directory not found: {noisy_dir}")

    clean_index = build_clean_index(clean_dir)
    paired_samples: list[PairedSample] = []
    skipped_files: list[str] = []
    scanned_noisy = 0

    for noisy_path in sorted(noisy_dir.iterdir()):
        if not noisy_path.is_file() or noisy_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        scanned_noisy += 1
        clean_path = clean_index.get(noisy_to_clean_stem(noisy_path.stem))
        if clean_path is None:
            skipped_files.append(noisy_path.name)
            continue
        paired_samples.append(PairedSample(clean_path=clean_path, noisy_path=noisy_path))

    print(f"Scanned noisy files: {scanned_noisy}")
    print(f"Matched pairs: {len(paired_samples)}")
    print(f"Skipped noisy files: {len(skipped_files)}")
    if skipped_files:
        print(f"Skipped file examples: {', '.join(skipped_files[:10])}")
    if not paired_samples:
        raise FileNotFoundError("No valid periodic v4 clean/noisy pairs were found.")
    return paired_samples


def split_samples(
    samples: list[PairedSample],
    seed: int,
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
        raise ValueError("Split produced an empty train, val, or test set.")
    return train_samples, val_samples, test_samples


def create_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    paired_samples = collect_pairs(args.clean_dir.resolve(), args.noisy_dir.resolve())
    train_samples, val_samples, test_samples = split_samples(paired_samples, seed=args.seed)
    print(
        f"Paired samples: total={len(paired_samples)} "
        f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}"
    )

    pin_memory = args.device.startswith("cuda") and torch.cuda.is_available()
    persistent_workers = args.num_workers > 0
    train_loader = DataLoader(
        PairedDenoisingDataset(train_samples, train=True, vertical_flip=args.vertical_flip),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        PairedDenoisingDataset(val_samples, train=False, vertical_flip=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        PairedDenoisingDataset(test_samples, train=False, vertical_flip=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader, test_loader


def build_model(device: torch.device, base_features: int) -> UNet:
    return UNet(in_channels=3, out_channels=3, base_features=base_features).to(device)


def save_checkpoint(
    model: UNet,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_val_loss: float,
    output_path: Path,
    base_features: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "prediction_mode": "residual",
            "architecture": {
                "in_channels": 3,
                "out_channels": 3,
                "base_features": base_features,
            },
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        output_path,
    )


def run_epoch(
    model: UNet,
    loader: DataLoader,
    criterion: PeriodicV4HybridLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            predicted_residual = model(noisy)
            predicted_clean = torch.clamp(noisy - predicted_residual, min=0.0, max=1.0)
            loss = criterion(predicted_clean, clean)

        if training:
            if scaler is None:
                raise ValueError("GradScaler is required during training.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = noisy.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate_test_loss(
    model: UNet,
    loader: DataLoader,
    criterion: PeriodicV4HybridLoss,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    return run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp_enabled=amp_enabled)


def main() -> None:
    args = parse_args()
    print("Warning: this v4 training script overwrites periodic_unet_residual_hybrid_best.pt and periodic_unet_residual_hybrid_last.pt")
    set_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    train_loader, val_loader, test_loader = create_dataloaders(args)
    model = build_model(device, args.base_features)
    criterion = PeriodicV4HybridLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_checkpoint_path = args.output_dir.resolve() / BEST_CHECKPOINT_NAME
    last_checkpoint_path = args.output_dir.resolve() / LAST_CHECKPOINT_NAME
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer, scaler, amp_enabled)
        val_loss = run_epoch(model, val_loader, criterion, device, None, None, amp_enabled)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={current_lr:.6g}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, best_checkpoint_path, args.base_features)
        else:
            epochs_without_improvement += 1

        save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, last_checkpoint_path, args.base_features)

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    test_loss = evaluate_test_loss(model, test_loader, criterion, device, amp_enabled)
    print(f"Best val loss = {best_val_loss:.6f}")
    print(f"test_loss = {test_loss:.6f}")
    print(f"Best checkpoint saved to: {best_checkpoint_path}")
    print(f"Last checkpoint saved to: {last_checkpoint_path}")


if __name__ == "__main__":
    main()
