from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
BEST_CHECKPOINT_NAME = "speckle_unet_l1_best.pt"
LAST_CHECKPOINT_NAME = "speckle_unet_l1_last.pt"


@dataclass(frozen=True)
class PairedSample:
    clean_path: Path
    noisy_path: Path


class PairedDenoisingDataset(Dataset[tuple[Tensor, Tensor]]):
    """Paired clean/noisy dataset with synchronized augmentations."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Speckle U-Net denoiser with L1 loss.")
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean/bsd500"))
    parser.add_argument("--noisy-dir", type=Path, default=Path("data/synthetic/speckle"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/denoisers/speckle"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=20)
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
    skipped_files: list[str] = []
    scanned_noisy = 0

    for noisy_path in sorted(noisy_dir.iterdir()):
        if not noisy_path.is_file() or noisy_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        scanned_noisy += 1
        clean_path = clean_index.get(noisy_path.stem)
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
        raise FileNotFoundError("No valid speckle clean/noisy pairs were found.")
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
        raise ValueError("Split produced an empty train, val, or test set. Increase dataset size.")
    return train_samples, val_samples, test_samples


def create_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    paired_samples = collect_pairs(args.clean_dir.resolve(), args.noisy_dir.resolve())
    train_samples, val_samples, test_samples = split_samples(paired_samples, seed=args.seed)

    print(
        f"Paired samples: total={len(paired_samples)} "
        f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}"
    )

    train_dataset = PairedDenoisingDataset(train_samples, train=True, vertical_flip=args.vertical_flip)
    val_dataset = PairedDenoisingDataset(val_samples, train=False, vertical_flip=False)
    test_dataset = PairedDenoisingDataset(test_samples, train=False, vertical_flip=False)

    common_loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
    }
    if args.num_workers > 0:
        common_loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, prefetch_factor=2, **common_loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_kwargs)
    return train_loader, val_loader, test_loader


def create_model(device: torch.device) -> UNet:
    return UNet(in_channels=3, out_channels=3, base_features=64).to(device)


def create_grad_scaler(amp_enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def autocast_context(device: torch.device, amp_enabled: bool) -> Any:
    try:
        return torch.amp.autocast(device_type=device.type, enabled=amp_enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=amp_enabled)


def train_one_epoch(
    model: UNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            prediction = model(noisy)
            loss = criterion(prediction, clean)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = noisy.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

    return running_loss / max(1, sample_count)


@torch.no_grad()
def evaluate(
    model: UNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.eval()
    running_loss = 0.0
    sample_count = 0

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        with autocast_context(device, amp_enabled):
            prediction = model(noisy)
            loss = criterion(prediction, clean)

        batch_size = noisy.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

    return running_loss / max(1, sample_count)


def checkpoint_payload(
    model: UNet,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_val_loss: float,
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "architecture": {
            "model_name": "UNet",
            "in_channels": 3,
            "out_channels": 3,
            "base_features": 64,
        },
    }


def save_checkpoint(checkpoint: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()
    global WORKER_BASE_SEED
    WORKER_BASE_SEED = args.seed
    set_seed(args.seed)

    device = torch.device(args.device)
    amp_enabled = device.type == "cuda"
    train_loader, val_loader, test_loader = create_dataloaders(args)

    model = create_model(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    scaler = create_grad_scaler(amp_enabled)

    best_checkpoint_path = args.output_dir.resolve() / BEST_CHECKPOINT_NAME
    last_checkpoint_path = args.output_dir.resolve() / LAST_CHECKPOINT_NAME
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, amp_enabled)
        val_loss = evaluate(model, val_loader, criterion, device, amp_enabled)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={current_lr:.6e}")

        checkpoint = checkpoint_payload(model, optimizer, scheduler, epoch, min(best_val_loss, val_loss))
        save_checkpoint(checkpoint, last_checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint["best_val_loss"] = best_val_loss
            save_checkpoint(checkpoint, best_checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["state_dict"])
    test_loss = evaluate(model, test_loader, criterion, device, amp_enabled)
    print(f"Best val loss={best_val_loss:.6f} | test_loss={test_loss:.6f}")


if __name__ == "__main__":
    main()
