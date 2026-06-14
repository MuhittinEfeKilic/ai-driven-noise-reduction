from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.inference.denoiser_inference import DenoiserInference
from src.metrics.psnr import psnr_score
from src.metrics.ssim import ssim_score


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEED = 42
CHECKPOINT_PATH = Path("models/denoisers/periodic/periodic_fft_guided_nafnet_best.pt")
CLEAN_DIR = Path("data/clean/bsd500")
NOISY_DIR = Path("data/synthetic/periodic_v3")
OUTPUT_DIR = Path("outputs/periodic_nafnet_test")
PERIODIC_V3_PATTERN = re.compile(r"^(?P<base>.+)_pv3_(?P<variant>\d+)$")


@dataclass(frozen=True)
class PairedSample:
    clean_path: Path
    noisy_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the active FFT-guided periodic NAFNet checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--clean-dir", type=Path, default=CLEAN_DIR)
    parser.add_argument("--noisy-dir", type=Path, default=NOISY_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image", type=Path, default=None, help="Optional single-image smoke test path.")
    return parser.parse_args()


def noisy_to_clean_stem(noisy_stem: str) -> str:
    match = PERIODIC_V3_PATTERN.match(noisy_stem)
    return match.group("base") if match else noisy_stem


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
        clean_path = clean_index.get(noisy_to_clean_stem(noisy_path.stem))
        if clean_path is None:
            continue
        paired_samples.append(PairedSample(clean_path=clean_path, noisy_path=noisy_path))

    if not paired_samples:
        raise FileNotFoundError("No valid periodic clean/noisy pairs were found.")
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
        else:
            raise ValueError("Unable to create non-empty train/val/test splits from the paired dataset.")

    train_end = train_count
    val_end = train_end + val_count
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def load_rgb_uint8(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.array(image.convert("RGB"))
    except OSError as exc:
        raise ValueError(f"Could not open image file: {path}") from exc


def save_rgb_uint8(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def evaluate_dataset(args: argparse.Namespace) -> None:
    paired_samples = collect_pairs(args.clean_dir.expanduser().resolve(), args.noisy_dir.expanduser().resolve())
    _, _, test_samples = split_samples(paired_samples, seed=SEED)
    inference = DenoiserInference(model_path=args.checkpoint, device=args.device)
    inference.load_model()

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    saved_examples = 0
    output_dir = args.output_dir.expanduser().resolve()

    for sample in tqdm(test_samples, desc="Testing Periodic FFT-Guided NAFNet", unit="image"):
        output_image = inference.run(sample.noisy_path)
        output_uint8 = np.array(output_image.convert("RGB"))
        clean_uint8 = load_rgb_uint8(sample.clean_path)
        noisy_uint8 = load_rgb_uint8(sample.noisy_path)

        psnr_values.append(psnr_score(clean_uint8, output_uint8))
        ssim_values.append(ssim_score(clean_uint8, output_uint8))

        if saved_examples < 8:
            sample_dir = output_dir / sample.noisy_path.stem
            save_rgb_uint8(noisy_uint8, sample_dir / "noisy.png")
            save_rgb_uint8(output_uint8, sample_dir / "output.png")
            save_rgb_uint8(clean_uint8, sample_dir / "clean.png")
            saved_examples += 1

    avg_psnr = float(np.mean(psnr_values)) if psnr_values else 0.0
    avg_ssim = float(np.mean(ssim_values)) if ssim_values else 0.0
    print(f"Avg PSNR: {avg_psnr:.4f}")
    print(f"Avg SSIM: {avg_ssim:.4f}")
    print(f"Saved sample outputs to: {output_dir}")


def evaluate_single_image(args: argparse.Namespace) -> None:
    assert args.image is not None
    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    inference = DenoiserInference(model_path=args.checkpoint, device=args.device)
    output_image = inference.run(image_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_periodic_nafnet.png"
    output_image.save(output_path)
    print(f"Saved denoised image to: {output_path}")


def main() -> None:
    args = parse_args()
    if args.image is not None:
        evaluate_single_image(args)
    else:
        evaluate_dataset(args)


if __name__ == "__main__":
    main()
