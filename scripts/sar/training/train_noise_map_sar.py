from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_MODULE_DIR = PROJECT_ROOT / "src" / "sar" / "datasets"
if str(DATASET_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_MODULE_DIR))

from noise_map_dataset import NoiseMapSARDataset  # noqa: E402


WORKER_BASE_SEED = 42


@dataclass
class LossBreakdown:
    l1: float
    ssim_loss: float
    total: float


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.block(x))


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.residual = ResidualBlock(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.residual(self.proj(x))
        return features, self.pool(features)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.residual = ResidualBlock(out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return self.residual(x)


class NoiseMapUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 8),
        )
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, base_channels)
        self.output_head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        x = self.bottleneck(x)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.output_head(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SAR noise-map translation model.")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use-incidence-scalar", action="store_true")
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/sar/models_noise_map"))
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--ssim-weight", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/sar/metrics_noise_map"))
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


def compute_input_channels(use_incidence_scalar: bool) -> int:
    return 2 if use_incidence_scalar else 1


def create_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset = NoiseMapSARDataset(
        args.train_dir,
        use_incidence_scalar=args.use_incidence_scalar,
        augment=True,
    )
    val_dataset = NoiseMapSARDataset(
        args.val_dir,
        use_incidence_scalar=args.use_incidence_scalar,
        augment=False,
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    if args.num_workers > 0:
        loader_args["persistent_workers"] = True
    return (
        DataLoader(train_dataset, shuffle=True, **loader_args),
        DataLoader(val_dataset, shuffle=False, **loader_args),
    )


def normalize_batch_to_unit(batch: Tensor) -> Tensor:
    batch_min = batch.amin(dim=(-2, -1), keepdim=True)
    batch_max = batch.amax(dim=(-2, -1), keepdim=True)
    return (batch - batch_min) / (batch_max - batch_min + 1e-8)


def gaussian_window(window_size: int, sigma: float, device: torch.device) -> Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma * sigma))
    gauss = gauss / gauss.sum()
    kernel = torch.outer(gauss, gauss)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, window_size, window_size)


def ssim_torch(prediction: Tensor, target: Tensor, window_size: int = 11, sigma: float = 1.5) -> Tensor:
    prediction = normalize_batch_to_unit(prediction)
    target = normalize_batch_to_unit(target)
    window = gaussian_window(window_size, sigma, prediction.device)
    padding = window_size // 2
    mu_x = F.conv2d(prediction, window, padding=padding)
    mu_y = F.conv2d(target, window, padding=padding)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(prediction * prediction, window, padding=padding) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, window, padding=padding) - mu_y_sq
    sigma_xy = F.conv2d(prediction * target, window, padding=padding) - mu_xy
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return (numerator / (denominator + 1e-8)).mean()


def compute_loss(prediction: Tensor, target: Tensor, args: argparse.Namespace) -> tuple[Tensor, Tensor, Tensor]:
    l1 = F.l1_loss(prediction, target)
    ssim_loss = (
        1.0 - ssim_torch(prediction, target)
        if args.ssim_weight > 0.0
        else prediction.new_tensor(0.0)
    )
    total = args.l1_weight * l1 + args.ssim_weight * ssim_loss
    return l1, ssim_loss, total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    device: torch.device,
    args: argparse.Namespace,
) -> LossBreakdown:
    model.train()
    total_l1 = 0.0
    total_ssim = 0.0
    total_loss = 0.0
    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        l1, ssim_loss, loss = compute_loss(predictions, targets, args)
        loss.backward()
        optimizer.step()
        total_l1 += l1.item()
        total_ssim += ssim_loss.item()
        total_loss += loss.item()
    count = max(len(loader), 1)
    return LossBreakdown(total_l1 / count, total_ssim / count, total_loss / count)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> LossBreakdown:
    model.eval()
    total_l1 = 0.0
    total_ssim = 0.0
    total_loss = 0.0
    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        predictions = model(inputs)
        l1, ssim_loss, loss = compute_loss(predictions, targets, args)
        total_l1 += l1.item()
        total_ssim += ssim_loss.item()
        total_loss += loss.item()
    count = max(len(loader), 1)
    return LossBreakdown(total_l1 / count, total_ssim / count, total_loss / count)


def save_checkpoint(
    model: nn.Module,
    save_dir: Path,
    epoch: int,
    best_val_loss: float,
    use_incidence_scalar: bool,
    input_channels: int,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "use_incidence_scalar": use_incidence_scalar,
            "input_channels": input_channels,
            "model_state_dict": model.state_dict(),
        },
        save_dir / "best_model.pth",
    )


def save_curves(train_losses: list[float], val_losses: list[float], metrics_dir: Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SAR Noise-Map Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(metrics_dir / "noise_map_loss_curve.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_dataloaders(args)
    input_channels = compute_input_channels(args.use_incidence_scalar)
    print(f"Input channels: {input_channels}")
    model = NoiseMapUNet(in_channels=input_channels, out_channels=1).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args)
        val_metrics = evaluate(model, val_loader, device, args)
        train_losses.append(train_metrics.total)
        val_losses.append(val_metrics.total)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_l1={train_metrics.l1:.6f} "
            f"train_ssim_loss={train_metrics.ssim_loss:.6f} "
            f"train_loss={train_metrics.total:.6f} "
            f"val_l1={val_metrics.l1:.6f} "
            f"val_ssim_loss={val_metrics.ssim_loss:.6f} "
            f"val_loss={val_metrics.total:.6f}"
        )
        if val_metrics.total < best_val_loss:
            best_val_loss = val_metrics.total
            save_checkpoint(
                model,
                args.save_dir,
                epoch,
                best_val_loss,
                args.use_incidence_scalar,
                input_channels,
            )
    save_curves(train_losses, val_losses, args.metrics_dir)


if __name__ == "__main__":
    main()
