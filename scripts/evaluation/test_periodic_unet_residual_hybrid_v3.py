from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from src.architectures.unet import UNet
from src.metrics.psnr import psnr_score
from src.metrics.ssim import ssim_score


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RESIZE_SIZE = 160
CROP_SIZE = 128
SEED = 42
BATCH_SIZE = 12
NUM_WORKERS = 2
CHECKPOINT_PATH = Path("archive_models/denoisers/periodic/periodic_unet_residual_hybrid_v3_best.pt")
CLEAN_DIR = Path("data/clean/bsd500")
NOISY_DIR = Path("data/synthetic/periodic_v3")
OUTPUT_DIR = Path("outputs/periodic_residual_hybrid_v3_test")
PERIODIC_V3_PATTERN = re.compile(r"^(?P<base>.+)_pv3_(?P<variant>\d+)$")


@dataclass(frozen=True)
class PairedSample:
    clean_path: Path
    noisy_path: Path


class PeriodicV3TestDataset(Dataset[tuple[Tensor, Tensor, str]]):
    def __init__(self, samples: list[PairedSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        sample = self.samples[index]
        clean = self._load_rgb_image(sample.clean_path)
        noisy = self._load_rgb_image(sample.noisy_path)
        clean, noisy = self._apply_eval_transforms(clean, noisy)
        return noisy, clean, sample.noisy_path.name

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


def noisy_to_clean_stem(noisy_stem: str) -> str:
    match = PERIODIC_V3_PATTERN.match(noisy_stem)
    return match.group("base") if match else noisy_stem


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
        raise FileNotFoundError("No valid periodic v3 clean/noisy pairs were found.")
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


def create_test_loader() -> DataLoader:
    paired_samples = collect_pairs(CLEAN_DIR.resolve(), NOISY_DIR.resolve())
    _, _, test_samples = split_samples(paired_samples, seed=SEED)
    dataset = PeriodicV3TestDataset(test_samples)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def tensor_to_uint8_image(tensor: Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return np.clip(np.rint(array * 255.0), 0.0, 255.0).astype(np.uint8)


def save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def load_model(device: torch.device) -> UNet:
    checkpoint = torch.load(CHECKPOINT_PATH.resolve(), map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    architecture = checkpoint.get("architecture", {}) if isinstance(checkpoint, dict) else {}
    model = UNet(
        in_channels=int(architecture.get("in_channels", 3)),
        out_channels=int(architecture.get("out_channels", 3)),
        base_features=int(architecture.get("base_features", 96)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict_clean(model: UNet, noisy_batch: Tensor) -> Tensor:
    predicted_residual = model(noisy_batch)
    return torch.clamp(noisy_batch - predicted_residual, min=0.0, max=1.0)


def periodic_strength_score(noisy: np.ndarray, clean: np.ndarray) -> float:
    residual = noisy.astype(np.float32) - clean.astype(np.float32)
    residual_gray = residual.mean(axis=2)
    fft_mag = np.log1p(np.abs(np.fft.rfft2(residual_gray)))
    return float(np.std(residual_gray) + 0.15 * np.max(fft_mag))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    loader = create_test_loader()

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    regular_examples: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    strongest_examples: list[tuple[float, str, np.ndarray, np.ndarray, np.ndarray]] = []

    for noisy_batch, clean_batch, names in tqdm(loader, desc="Testing Periodic Residual Hybrid V3", unit="batch"):
        noisy_batch = noisy_batch.to(device, non_blocking=True)
        output_batch = predict_clean(model, noisy_batch)

        for idx, name in enumerate(names):
            noisy_image = tensor_to_uint8_image(noisy_batch[idx])
            output_image = tensor_to_uint8_image(output_batch[idx])
            clean_image = tensor_to_uint8_image(clean_batch[idx])

            psnr_values.append(psnr_score(clean_image, output_image))
            ssim_values.append(ssim_score(clean_image, output_image))

            if len(regular_examples) < 4:
                regular_examples.append((name, noisy_image, output_image, clean_image))

            strongest_examples.append(
                (periodic_strength_score(noisy_image, clean_image), name, noisy_image, output_image, clean_image)
            )

    strongest_examples.sort(key=lambda item: item[0], reverse=True)

    for rank, (name, noisy_image, output_image, clean_image) in enumerate(regular_examples, start=1):
        sample_dir = OUTPUT_DIR.resolve() / f"sample_{rank:02d}_{Path(name).stem}"
        save_image(noisy_image, sample_dir / "noisy.png")
        save_image(output_image, sample_dir / "output.png")
        save_image(clean_image, sample_dir / "clean.png")

    for rank, (_, name, noisy_image, output_image, clean_image) in enumerate(strongest_examples[:4], start=1):
        sample_dir = OUTPUT_DIR.resolve() / f"strong_{rank:02d}_{Path(name).stem}"
        save_image(noisy_image, sample_dir / "noisy.png")
        save_image(output_image, sample_dir / "output.png")
        save_image(clean_image, sample_dir / "clean.png")

    avg_psnr = float(np.mean(psnr_values)) if psnr_values else 0.0
    avg_ssim = float(np.mean(ssim_values)) if ssim_values else 0.0
    print(f"Avg PSNR: {avg_psnr:.4f}")
    print(f"Avg SSIM: {avg_ssim:.4f}")
    print("Saved standard examples: 4")
    print("Saved strong-periodic examples: 4")
    print(f"Saved sample outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
