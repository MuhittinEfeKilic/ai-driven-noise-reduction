from __future__ import annotations

import argparse
import math
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

from paired_sar_dataset import PairedSARDataset  # noqa: E402


WORKER_BASE_SEED = 42


@dataclass
class LossBreakdown:
    l1: float
    ssim_loss: float
    fft: float
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


class ResidualUNet(nn.Module):
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
        return torch.clamp(self.output_head(x), 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a residual U-Net for supervised SAR denoising.")
    parser.add_argument("--train-dir", type=Path, default=Path("data/sar/processed/paired/train"))
    parser.add_argument("--val-dir", type=Path, default=Path("data/sar/processed/paired/val"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1-weight", type=float, default=0.7)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--fft-weight", type=float, default=0.1)
    parser.add_argument("--use-incidence", action="store_true")
    parser.add_argument("--use-incidence-scalar", action="store_true")
    parser.add_argument("--use-column-profile", action="store_true")
    parser.add_argument("--use-soft-swath", action="store_true")
    parser.add_argument("--predict-residual-noise", action="store_true")
    parser.add_argument("--predict-correction-residual", action="store_true")
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/sar/models"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/sar/metrics"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.predict_residual_noise and args.predict_correction_residual:
        raise ValueError(
            "--predict-residual-noise and --predict-correction-residual cannot be used together."
        )


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


def compute_input_channels(args: argparse.Namespace) -> int:
    input_channels = 1
    if args.use_incidence or args.use_incidence_scalar:
        input_channels += 1
    if args.use_column_profile:
        input_channels += 1
    if args.use_soft_swath:
        input_channels += 1
    return input_channels


def create_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset = PairedSARDataset(
        args.train_dir,
        use_incidence=args.use_incidence,
        use_incidence_scalar=args.use_incidence_scalar,
        use_column_profile=args.use_column_profile,
        use_soft_swath=args.use_soft_swath,
        augment=True,
    )
    val_dataset = PairedSARDataset(
        args.val_dir,
        use_incidence=args.use_incidence,
        use_incidence_scalar=args.use_incidence_scalar,
        use_column_profile=args.use_column_profile,
        use_soft_swath=args.use_soft_swath,
        augment=False,
    )

    common_loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    if args.num_workers > 0:
        common_loader_args["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_args)
    return train_loader, val_loader


def gaussian_window(window_size: int, sigma: float, device: torch.device) -> Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma * sigma))
    gauss = gauss / gauss.sum()
    kernel = torch.outer(gauss, gauss)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, window_size, window_size)


def ssim_torch(prediction: Tensor, target: Tensor, window_size: int = 11, sigma: float = 1.5) -> Tensor:
    channel = prediction.shape[1]
    window = gaussian_window(window_size, sigma, prediction.device).repeat(channel, 1, 1, 1)
    padding = window_size // 2

    mu_x = F.conv2d(prediction, window, padding=padding, groups=channel)
    mu_y = F.conv2d(target, window, padding=padding, groups=channel)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(prediction * prediction, window, padding=padding, groups=channel) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, window, padding=padding, groups=channel) - mu_y_sq
    sigma_xy = F.conv2d(prediction * target, window, padding=padding, groups=channel) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return (numerator / (denominator + 1e-8)).mean()


def frequency_loss(prediction: Tensor, target: Tensor, epsilon: float = 1e-8) -> Tensor:
    prediction_fft = torch.fft.fft2(prediction, dim=(-2, -1))
    target_fft = torch.fft.fft2(target, dim=(-2, -1))

    prediction_magnitude = torch.log1p(torch.abs(prediction_fft) + epsilon)
    target_magnitude = torch.log1p(torch.abs(target_fft) + epsilon)
    return F.l1_loss(prediction_magnitude, target_magnitude)


def compute_loss_components(
    prediction: Tensor,
    supervision_target: Tensor,
    args: argparse.Namespace,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    l1 = F.l1_loss(prediction, supervision_target)
    ssim_loss = 1.0 - ssim_torch(prediction, supervision_target)
    fft = (
        frequency_loss(prediction, supervision_target)
        if args.fft_weight > 0.0
        else prediction.new_tensor(0.0)
    )
    total = (args.l1_weight * l1) + (args.ssim_weight * ssim_loss) + (args.fft_weight * fft)
    return l1, ssim_loss, fft, total


def prepare_supervision_target(inputs: Tensor, clean_targets: Tensor, args: argparse.Namespace) -> Tensor:
    noisy_input = inputs[:, :1, :, :]
    if args.predict_residual_noise:
        return noisy_input - clean_targets
    if args.predict_correction_residual:
        return clean_targets - noisy_input
    return clean_targets


def reconstruct_clean_from_prediction(inputs: Tensor, predictions: Tensor, args: argparse.Namespace) -> Tensor:
    noisy_input = inputs[:, :1, :, :]
    if args.predict_residual_noise:
        return torch.clamp(noisy_input - predictions, 0.0, 1.0)
    if args.predict_correction_residual:
        return torch.clamp(noisy_input + predictions, 0.0, 1.0)
    return predictions


def psnr_torch(prediction: Tensor, target: Tensor) -> float:
    mse = torch.mean((prediction - target) ** 2).item()
    if mse <= 0.0:
        return float("inf")
    return 20.0 * math.log10(1.0) - 10.0 * math.log10(mse)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    device: torch.device,
    args: argparse.Namespace,
) -> LossBreakdown:
    model.train()
    running_l1 = 0.0
    running_ssim = 0.0
    running_fft = 0.0
    running_total = 0.0

    for batch in loader:
        inputs = batch["input"].to(device)
        clean_targets = batch["target"].to(device)
        supervision_target = prepare_supervision_target(inputs, clean_targets, args)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        l1, ssim_loss, fft, total = compute_loss_components(predictions, supervision_target, args)
        total.backward()
        optimizer.step()

        running_l1 += l1.item()
        running_ssim += ssim_loss.item()
        running_fft += fft.item()
        running_total += total.item()

    count = max(len(loader), 1)
    return LossBreakdown(
        l1=running_l1 / count,
        ssim_loss=running_ssim / count,
        fft=running_fft / count,
        total=running_total / count,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[LossBreakdown, float, float]:
    model.eval()
    total_l1 = 0.0
    total_ssim_loss = 0.0
    total_fft = 0.0
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    for batch in loader:
        inputs = batch["input"].to(device)
        clean_targets = batch["target"].to(device)
        supervision_target = prepare_supervision_target(inputs, clean_targets, args)
        predictions = model(inputs)
        reconstructed_clean = reconstruct_clean_from_prediction(inputs, predictions, args)

        l1, ssim_loss, fft, loss = compute_loss_components(predictions, supervision_target, args)
        total_l1 += l1.item()
        total_ssim_loss += ssim_loss.item()
        total_fft += fft.item()
        total_loss += loss.item()
        total_psnr += psnr_torch(reconstructed_clean, clean_targets)
        total_ssim += float(ssim_torch(reconstructed_clean, clean_targets).item())

    count = max(len(loader), 1)
    breakdown = LossBreakdown(
        l1=total_l1 / count,
        ssim_loss=total_ssim_loss / count,
        fft=total_fft / count,
        total=total_loss / count,
    )
    return breakdown, total_psnr / count, total_ssim / count


def save_checkpoint(
    model: nn.Module,
    save_dir: Path,
    epoch: int,
    best_psnr: float,
    use_incidence: bool,
    use_incidence_scalar: bool,
    use_column_profile: bool,
    use_soft_swath: bool,
    predict_residual_noise: bool,
    predict_correction_residual: bool,
    input_channels: int,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_psnr": best_psnr,
            "use_incidence": use_incidence,
            "use_incidence_scalar": use_incidence_scalar,
            "use_column_profile": use_column_profile,
            "use_soft_swath": use_soft_swath,
            "predict_residual_noise": predict_residual_noise,
            "predict_correction_residual": predict_correction_residual,
            "input_channels": input_channels,
            "model_state_dict": model.state_dict(),
        },
        save_dir / "best_model.pth",
    )


def save_curves(
    train_losses: list[float],
    val_psnr_values: list[float],
    train_fft_losses: list[float],
    val_fft_losses: list[float],
    metrics_dir: Path,
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SAR Train Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(metrics_dir / "train_loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_fft_losses, label="Train FFT Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("FFT Loss")
    plt.title("SAR Train FFT Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(metrics_dir / "train_fft_loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_psnr_values, label="Val PSNR", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title("SAR Validation PSNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(metrics_dir / "val_psnr_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_fft_losses, label="Val FFT Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("FFT Loss")
    plt.title("SAR Validation FFT Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(metrics_dir / "val_fft_loss_curve.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    if args.predict_residual_noise:
        print("Residual noise prediction mode enabled")
    if args.predict_correction_residual:
        print("Correction residual prediction mode enabled")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_dataloaders(args)
    input_channels = compute_input_channels(args)
    print(f"Input channels: {input_channels}")
    model = ResidualUNet(in_channels=input_channels, out_channels=1).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    train_losses: list[float] = []
    val_psnr_values: list[float] = []
    train_fft_losses: list[float] = []
    val_fft_losses: list[float] = []
    best_val_psnr = float("-inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args)
        val_metrics, val_psnr, val_ssim = evaluate(model, val_loader, device, args)

        train_losses.append(train_metrics.total)
        val_psnr_values.append(val_psnr)
        train_fft_losses.append(train_metrics.fft)
        val_fft_losses.append(val_metrics.fft)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_l1={train_metrics.l1:.4f} "
            f"train_ssim={train_metrics.ssim_loss:.4f} "
            f"train_fft={train_metrics.fft:.4f} "
            f"train_loss={train_metrics.total:.4f} "
            f"val_l1={val_metrics.l1:.4f} "
            f"val_ssim_loss={val_metrics.ssim_loss:.4f} "
            f"val_fft={val_metrics.fft:.4f} "
            f"val_loss={val_metrics.total:.4f} "
            f"val_psnr={val_psnr:.4f} "
            f"val_ssim={val_ssim:.4f}"
        )

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            save_checkpoint(
                model,
                args.save_dir,
                epoch,
                best_val_psnr,
                args.use_incidence and not args.use_incidence_scalar,
                args.use_incidence_scalar,
                args.use_column_profile,
                args.use_soft_swath,
                args.predict_residual_noise,
                args.predict_correction_residual,
                input_channels,
            )

    save_curves(train_losses, val_psnr_values, train_fft_losses, val_fft_losses, args.metrics_dir)


if __name__ == "__main__":
    main()
