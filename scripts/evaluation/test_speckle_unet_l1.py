from __future__ import annotations

import random
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

from src.inference.denoiser_inference import DenoiserInference
from src.metrics.psnr import psnr_score
from src.metrics.ssim import ssim_score


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RESIZE_SIZE = 160
CROP_SIZE = 128
SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 2
CHECKPOINT_PATH = Path("archive_models/denoisers/speckle/archive/speckle_unet_l1_best.pt")
CLEAN_DIR = Path("data/clean/bsd500")
NOISY_DIR = Path("data/synthetic/speckle")
OUTPUT_DIR = Path("outputs/speckle_l1_test")


@dataclass(frozen=True)
class PairedSample:
    clean_path: Path
    noisy_path: Path


class SpeckleTestDataset(Dataset[tuple[Tensor, Tensor, str]]):
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
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


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


def tensor_to_uint8_image(tensor: Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return np.clip(np.rint(array * 255.0), 0.0, 255.0).astype(np.uint8)


def save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference = DenoiserInference(model_path=CHECKPOINT_PATH, device=device)
    inference.load_model()
    loader = create_test_loader()

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    saved_examples = 0

    for noisy_batch, clean_batch, names in tqdm(loader, desc="Testing Speckle L1", unit="batch"):
        noisy_batch = noisy_batch.to(inference.device, non_blocking=True)
        with torch.no_grad():
            output_batch = inference._run_model(noisy_batch)

        for idx, name in enumerate(names):
            noisy_image = tensor_to_uint8_image(noisy_batch[idx])
            output_image = tensor_to_uint8_image(output_batch[idx])
            clean_image = tensor_to_uint8_image(clean_batch[idx])

            psnr_values.append(psnr_score(clean_image, output_image))
            ssim_values.append(ssim_score(clean_image, output_image))

            if saved_examples < 8:
                sample_dir = OUTPUT_DIR.resolve() / Path(name).stem
                save_image(noisy_image, sample_dir / "noisy.png")
                save_image(output_image, sample_dir / "output.png")
                save_image(clean_image, sample_dir / "clean.png")
                saved_examples += 1

    avg_psnr = float(np.mean(psnr_values)) if psnr_values else 0.0
    avg_ssim = float(np.mean(ssim_values)) if ssim_values else 0.0
    print(f"Avg PSNR: {avg_psnr:.4f}")
    print(f"Avg SSIM: {avg_ssim:.4f}")
    print(f"Saved sample outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
