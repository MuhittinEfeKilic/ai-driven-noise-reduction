from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from src.models.nafnet_periodic import PeriodicFFTGuidedNAFNet
from src.preprocessing.periodic_fft_filter import remove_periodic_noise_fft


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
WORKER_BASE_SEED = 42
BEST_CHECKPOINT_NAME = "periodic_fft_guided_nafnet_best.pt"
LAST_CHECKPOINT_NAME = "periodic_fft_guided_nafnet_last.pt"
PERIODIC_V3_PATTERN = re.compile(r"^(?P<base>.+)_pv3_(?P<variant>\d+)$")
RESIDUAL_SCALE = 0.9


@dataclass(frozen=True)
class PairedSample:
    clean_path: Path
    noisy_path: Path


class FFTGuidedPeriodicDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, samples: list[PairedSample], helper_dir: Path, train: bool, image_size: int) -> None:
        self.samples = samples
        self.helper_dir = helper_dir
        self.train = train
        self.image_size = image_size
        self.resize_size = max(image_size + 64, image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        sample = self.samples[index]
        clean = self._load_rgb_image(sample.clean_path)
        noisy = self._load_rgb_image(sample.noisy_path)
        helper = self._load_rgb_image(self.helper_dir / sample.noisy_path.name)
        clean, noisy, helper = self._apply_paired_transforms(clean, noisy, helper)

        noisy_tensor = TF.to_tensor(noisy)
        helper_tensor = TF.to_tensor(helper)
        clean_tensor = TF.to_tensor(clean)
        input_tensor = torch.cat([noisy_tensor, helper_tensor], dim=0)
        return input_tensor, noisy_tensor, clean_tensor

    @staticmethod
    def _load_rgb_image(path: Path) -> Image.Image:
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except OSError as exc:
            raise ValueError(f"Could not open image file: {path}") from exc

    def _apply_paired_transforms(
        self,
        clean: Image.Image,
        noisy: Image.Image,
        helper: Image.Image,
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        clean = TF.resize(clean, [self.resize_size, self.resize_size], interpolation=InterpolationMode.BILINEAR)
        noisy = TF.resize(noisy, [self.resize_size, self.resize_size], interpolation=InterpolationMode.BILINEAR)
        helper = TF.resize(helper, [self.resize_size, self.resize_size], interpolation=InterpolationMode.BILINEAR)

        if self.train:
            top, left, height, width = self._sample_crop_params()
            clean = TF.crop(clean, top, left, height, width)
            noisy = TF.crop(noisy, top, left, height, width)
            helper = TF.crop(helper, top, left, height, width)
            if random.random() < 0.5:
                clean = TF.hflip(clean)
                noisy = TF.hflip(noisy)
                helper = TF.hflip(helper)
            if random.random() < 0.2:
                clean = TF.vflip(clean)
                noisy = TF.vflip(noisy)
                helper = TF.vflip(helper)
        else:
            clean = TF.center_crop(clean, [self.image_size, self.image_size])
            noisy = TF.center_crop(noisy, [self.image_size, self.image_size])
            helper = TF.center_crop(helper, [self.image_size, self.image_size])
        return clean, noisy, helper

    def _sample_crop_params(self) -> tuple[int, int, int, int]:
        max_offset = self.resize_size - self.image_size
        top = random.randint(0, max_offset)
        left = random.randint(0, max_offset)
        return top, left, self.image_size, self.image_size


class SobelEdgeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3))

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        pred_gray = prediction.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
        kernel_x = self.kernel_x.to(device=prediction.device, dtype=prediction.dtype)
        kernel_y = self.kernel_y.to(device=prediction.device, dtype=prediction.dtype)
        pred_dx = F.conv2d(pred_gray, kernel_x, padding=1)
        pred_dy = F.conv2d(pred_gray, kernel_y, padding=1)
        target_dx = F.conv2d(target_gray, kernel_x, padding=1)
        target_dy = F.conv2d(target_gray, kernel_y, padding=1)
        return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


class FFTMagnitudeLoss(nn.Module):
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        pred_fft = torch.fft.rfft2(prediction, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        pred_mag = torch.log1p(torch.abs(pred_fft))
        target_mag = torch.log1p(torch.abs(target_fft))
        return F.l1_loss(pred_mag, target_mag)


class PeriodicNAFNetLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()
        self.edge = SobelEdgeLoss()
        self.fft = FFTMagnitudeLoss()

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        l1_loss = self.l1(prediction, target)
        edge_loss = self.edge(prediction, target)
        fft_loss = self.fft(prediction, target)
        return l1_loss + 0.2 * edge_loss + 0.05 * fft_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FFT-guided periodic NAFNet denoiser.")
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean/bsd500"))
    parser.add_argument("--noisy-dir", type=Path, default=Path("data/synthetic/periodic_v3"))
    parser.add_argument("--fft-cache-dir", type=Path, default=Path("data/synthetic/periodic_v3_fft_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/denoisers/periodic"))
    parser.add_argument("--image_size", "--image-size", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--early_stopping_patience", "--early-stopping-patience", type=int, default=10)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--use_amp", action="store_true")
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
    for noisy_path in sorted(noisy_dir.iterdir()):
        if not noisy_path.is_file() or noisy_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        clean_path = clean_index.get(noisy_to_clean_stem(noisy_path.stem))
        if clean_path is not None:
            paired_samples.append(PairedSample(clean_path=clean_path, noisy_path=noisy_path))
    if not paired_samples:
        raise FileNotFoundError("No valid periodic clean/noisy pairs were found.")
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
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
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
    train_end = train_count
    val_end = train_end + val_count
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def ensure_fft_helper_cache(samples: list[PairedSample], cache_dir: Path) -> Path:
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing_samples = [sample for sample in samples if not (cache_dir / sample.noisy_path.name).exists()]
    print(f"FFT helper cache directory: {cache_dir}")
    if not missing_samples:
        print("FFT helper cache status: warm")
        return cache_dir

    print(f"Generating missing FFT helper images: {len(missing_samples)}")
    for sample in tqdm(missing_samples, desc="Caching FFT helpers", unit="image"):
        noisy_bgr = cv2.imread(str(sample.noisy_path), cv2.IMREAD_COLOR)
        if noisy_bgr is None:
            raise ValueError(f"Failed to read noisy image for FFT cache: {sample.noisy_path}")
        helper_bgr = remove_periodic_noise_fft(
            noisy_bgr,
            threshold_ratio=0.08,
            min_distance=10,
            filter_radius=8,
        )
        helper_path = cache_dir / sample.noisy_path.name
        if not cv2.imwrite(str(helper_path), helper_bgr):
            raise ValueError(f"Failed to write FFT helper cache image: {helper_path}")
    return cache_dir


def create_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    paired_samples = collect_pairs(args.clean_dir.resolve(), args.noisy_dir.resolve())
    train_samples, val_samples, test_samples = split_samples(paired_samples, seed=args.seed)
    print(
        f"Paired samples: total={len(paired_samples)} "
        f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}"
    )
    helper_dir = ensure_fft_helper_cache(paired_samples, args.fft_cache_dir)
    pin_memory = args.device.startswith("cuda") and torch.cuda.is_available()
    persistent_workers = args.num_workers > 0
    common = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        FFTGuidedPeriodicDataset(train_samples, helper_dir=helper_dir, train=True, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        **common,
    )
    val_loader = DataLoader(
        FFTGuidedPeriodicDataset(val_samples, helper_dir=helper_dir, train=False, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        FFTGuidedPeriodicDataset(test_samples, helper_dir=helper_dir, train=False, image_size=args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, test_loader


def build_model(device: torch.device, width: int) -> PeriodicFFTGuidedNAFNet:
    return PeriodicFFTGuidedNAFNet(in_channels=6, out_channels=3, width=width).to(device)


def save_checkpoint(
    model: PeriodicFFTGuidedNAFNet,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_val_loss: float,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "prediction_mode": "residual",
            "architecture": {
                "model_name": "PeriodicFFTGuidedNAFNet",
                "in_channels": 6,
                "out_channels": 3,
                "width": args.width,
            },
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        output_path,
    )


def run_epoch(
    model: PeriodicFFTGuidedNAFNet,
    loader: DataLoader,
    criterion: PeriodicNAFNetLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    for input_batch, noisy_batch, clean_batch in loader:
        input_batch = input_batch.to(device, non_blocking=True)
        noisy_batch = noisy_batch.to(device, non_blocking=True)
        clean_batch = clean_batch.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            predicted_noise = model(input_batch)
            predicted_clean = torch.clamp(noisy_batch - RESIDUAL_SCALE * predicted_noise, min=0.0, max=1.0)
            loss = criterion(predicted_clean, clean_batch)

        if training:
            if scaler is None:
                raise ValueError("GradScaler is required during training.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = input_batch.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def evaluate_test_loss(
    model: PeriodicFFTGuidedNAFNet,
    loader: DataLoader,
    criterion: PeriodicNAFNetLoss,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    return run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp_enabled=amp_enabled)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.use_amp and device.type == "cuda")

    print("Training settings")
    print(f"- image_size: {args.image_size}")
    print(f"- batch_size: {args.batch_size}")
    print(f"- epochs: {args.epochs}")
    print(f"- num_workers: {args.num_workers}")
    print(f"- width: {args.width}")
    print(f"- early_stopping_patience: {args.early_stopping_patience}")
    print(f"- use_amp: {amp_enabled}")
    print(f"- device: {device}")

    train_loader, val_loader, test_loader = create_dataloaders(args)
    model = build_model(device, args.width)
    criterion = PeriodicNAFNetLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
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
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, best_checkpoint_path, args)
        else:
            epochs_without_improvement += 1

        save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, last_checkpoint_path, args)
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    test_loss = evaluate_test_loss(model, test_loader, criterion, device, amp_enabled)
    print(f"Best val loss = {best_val_loss:.6f}")
    print(f"test_loss = {test_loss:.6f}")
    print(f"Best checkpoint saved to: {best_checkpoint_path}")
    print(f"Last checkpoint saved to: {last_checkpoint_path}")


if __name__ == "__main__":
    main()
