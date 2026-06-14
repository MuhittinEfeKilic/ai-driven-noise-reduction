#!/usr/bin/env python3
"""Generate synthetic noisy variants from clean BSD500 images."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
NOISE_TYPES = ("gaussian", "salt_pepper", "speckle", "periodic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic noisy images from clean BSD500 inputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/clean/bsd500"),
        help="Directory containing clean input images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/synthetic"),
        help="Root directory for synthetic outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--classifier-output-root",
        type=Path,
        default=Path("data/classifier_synthetic"),
        help="Root directory for classifier-oriented synthetic outputs.",
    )
    parser.add_argument(
        "--skip-classifier-dataset",
        action="store_true",
        help="Skip generating the classifier-oriented synthetic dataset.",
    )
    parser.add_argument(
        "--classifier-gaussian-variants",
        type=int,
        default=2,
        help="Number of Gaussian variants per clean image for classifier data.",
    )
    parser.add_argument(
        "--classifier-salt-pepper-variants",
        type=int,
        default=2,
        help="Number of salt-pepper variants per clean image for classifier data.",
    )
    parser.add_argument(
        "--classifier-speckle-variants",
        type=int,
        default=2,
        help="Number of speckle variants per clean image for classifier data.",
    )
    parser.add_argument(
        "--classifier-periodic-variants",
        type=int,
        default=2,
        help="Number of periodic variants per clean image for classifier data.",
    )
    parser.add_argument(
        "--generate-periodic-v3",
        action="store_true",
        help="Generate a stronger periodic-only dataset for the v3 denoiser experiment.",
    )
    parser.add_argument(
        "--periodic-v3-output-dir",
        type=Path,
        default=Path("data/synthetic/periodic_v3"),
        help="Output directory for the stronger periodic v3 denoiser dataset.",
    )
    parser.add_argument(
        "--periodic-v3-variants-per-image",
        type=int,
        default=3,
        help="Number of stronger periodic v3 variants to generate per clean image.",
    )
    return parser.parse_args()


def get_output_dirs(output_root: Path) -> dict[str, Path]:
    return {
        "gaussian": output_root / "gaussian",
        "salt_pepper": output_root / "salt_pepper",
        "speckle": output_root / "speckle",
        "periodic": output_root / "periodic",
    }


def get_classifier_output_dirs(output_root: Path) -> dict[str, Path]:
    return {noise_type: output_root / noise_type for noise_type in NOISE_TYPES}


def list_clean_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return image


def add_gaussian_noise(image: np.ndarray, sigma: int, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def sample_gaussian_sigma(rng: random.Random) -> int:
    band = rng.choices(
        population=["light", "medium", "heavy"],
        weights=[0.10, 0.35, 0.55],
        k=1,
    )[0]

    ranges = {
        "light": (15, 25),
        "medium": (25, 55),
        "heavy": (55, 100),
    }
    low, high = ranges[band]
    return rng.randint(low, high)


def add_salt_pepper_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noisy = image.copy()
    height, width = noisy.shape[:2]
    total_pixels = height * width

    amount = float(rng.uniform(0.01, 0.06))
    salt_ratio = float(rng.uniform(0.45, 0.55))
    salt_count = int(total_pixels * amount * salt_ratio)
    pepper_count = int(total_pixels * amount * (1.0 - salt_ratio))

    if salt_count > 0:
        ys = rng.integers(0, height, size=salt_count)
        xs = rng.integers(0, width, size=salt_count)
        noisy[ys, xs] = 255

    if pepper_count > 0:
        ys = rng.integers(0, height, size=pepper_count)
        xs = rng.integers(0, width, size=pepper_count)
        noisy[ys, xs] = 0

    return noisy


def add_speckle_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    image_float = image.astype(np.float32) / 255.0
    variance_band = rng.choice(["light", "medium", "heavy"], p=[0.20, 0.45, 0.35])
    variance_ranges = {
        "light": (0.05, 0.10),
        "medium": (0.10, 0.22),
        "heavy": (0.22, 0.34),
    }
    low, high = variance_ranges[str(variance_band)]
    primary_variance = float(rng.uniform(low, high))
    secondary_variance = float(rng.uniform(0.015, 0.06))

    primary_noise = rng.normal(loc=0.0, scale=primary_variance, size=image.shape).astype(np.float32)
    secondary_noise = rng.normal(loc=0.0, scale=secondary_variance, size=image.shape).astype(np.float32)
    combined_noise = 0.85 * primary_noise + 0.35 * secondary_noise
    noisy = image_float + image_float * combined_noise
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)


def add_periodic_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = image.shape[:2]
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )

    orientation_candidates = np.array([0.0, np.pi / 2.0, np.pi / 4.0, 3.0 * np.pi / 4.0], dtype=np.float32)
    component_count = int(rng.integers(1, 3))
    periodic = np.zeros((height, width), dtype=np.float32)

    for component_idx in range(component_count):
        if rng.random() < 0.7:
            angle = float(rng.choice(orientation_candidates))
        else:
            angle = float(rng.uniform(0.0, np.pi))

        direction = np.cos(angle) * xx + np.sin(angle) * yy
        orthogonal = -np.sin(angle) * xx + np.cos(angle) * yy
        amplitude = float(rng.uniform(16.0, 52.0) if component_idx == 0 else rng.uniform(8.0, 28.0))
        frequency = float(rng.uniform(0.015, 0.090) if component_idx == 0 else rng.uniform(0.030, 0.120))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))

        component = np.sin(2.0 * np.pi * frequency * direction + phase)
        if rng.random() < 0.45:
            modulation_frequency = float(rng.uniform(0.004, 0.025))
            modulation_phase = float(rng.uniform(0.0, 2.0 * np.pi))
            modulation = 1.0 + 0.25 * np.sin(2.0 * np.pi * modulation_frequency * orthogonal + modulation_phase)
            component = component * modulation

        periodic += amplitude * component
    periodic = np.clip(periodic, -90.0, 90.0)

    noisy = image.astype(np.float32) + periodic[..., None]
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_periodic_noise_v3(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a harder periodic interference pattern for the v3 experiment."""
    height, width = image.shape[:2]
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )

    centered_x = xx - (width / 2.0)
    centered_y = yy - (height / 2.0)
    pattern_family = str(rng.choice(["oriented", "grid", "moire", "checker"]))
    periodic = np.zeros((height, width), dtype=np.float32)

    if pattern_family == "oriented":
        component_count = int(rng.integers(2, 5))
        for component_idx in range(component_count):
            angle = float(rng.choice([0.0, np.pi / 2.0, np.pi / 4.0, 3.0 * np.pi / 4.0]))
            if component_idx > 0 and rng.random() < 0.4:
                angle += float(rng.uniform(-0.18, 0.18))
            direction = np.cos(angle) * xx + np.sin(angle) * yy
            amplitude = float(rng.uniform(28.0, 72.0) if component_idx == 0 else rng.uniform(12.0, 34.0))
            frequency = float(rng.uniform(0.035, 0.140) if component_idx == 0 else rng.uniform(0.060, 0.180))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            periodic += amplitude * np.sin(2.0 * np.pi * frequency * direction + phase)

    elif pattern_family == "grid":
        amp_x = float(rng.uniform(24.0, 56.0))
        amp_y = float(rng.uniform(24.0, 56.0))
        freq_x = float(rng.uniform(0.035, 0.120))
        freq_y = float(rng.uniform(0.035, 0.120))
        phase_x = float(rng.uniform(0.0, 2.0 * np.pi))
        phase_y = float(rng.uniform(0.0, 2.0 * np.pi))
        periodic += amp_x * np.sin(2.0 * np.pi * freq_x * xx + phase_x)
        periodic += amp_y * np.sin(2.0 * np.pi * freq_y * yy + phase_y)
        if rng.random() < 0.6:
            periodic += float(rng.uniform(8.0, 22.0)) * np.sin(
                2.0 * np.pi * rng.uniform(0.030, 0.090) * (xx + yy) + rng.uniform(0.0, 2.0 * np.pi)
            )

    elif pattern_family == "moire":
        base_angle = float(rng.uniform(0.0, np.pi))
        angle_delta = float(rng.uniform(0.02, 0.18))
        base_frequency = float(rng.uniform(0.050, 0.140))
        secondary_frequency = base_frequency + float(rng.uniform(0.004, 0.018))
        dir_a = np.cos(base_angle) * centered_x + np.sin(base_angle) * centered_y
        dir_b = np.cos(base_angle + angle_delta) * centered_x + np.sin(base_angle + angle_delta) * centered_y
        periodic += float(rng.uniform(22.0, 54.0)) * np.sin(
            2.0 * np.pi * base_frequency * dir_a + rng.uniform(0.0, 2.0 * np.pi)
        )
        periodic += float(rng.uniform(18.0, 48.0)) * np.sin(
            2.0 * np.pi * secondary_frequency * dir_b + rng.uniform(0.0, 2.0 * np.pi)
        )
        periodic += float(rng.uniform(8.0, 18.0)) * np.sin(
            2.0 * np.pi * rng.uniform(0.030, 0.070) * (centered_x - centered_y) + rng.uniform(0.0, 2.0 * np.pi)
        )

    else:
        amp = float(rng.uniform(22.0, 52.0))
        freq_x = float(rng.uniform(0.040, 0.140))
        freq_y = float(rng.uniform(0.040, 0.140))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        checker = np.sin(2.0 * np.pi * freq_x * xx + phase) * np.sin(
            2.0 * np.pi * freq_y * yy + rng.uniform(0.0, 2.0 * np.pi)
        )
        periodic += amp * checker
        periodic += float(rng.uniform(10.0, 24.0)) * np.sin(
            2.0 * np.pi * rng.uniform(0.035, 0.100) * (xx + yy) + rng.uniform(0.0, 2.0 * np.pi)
        )

    if rng.random() < 0.55:
        envelope = 1.0 + 0.22 * np.sin(
            2.0 * np.pi * rng.uniform(0.002, 0.010) * centered_x + rng.uniform(0.0, 2.0 * np.pi)
        )
        periodic *= envelope

    periodic = np.clip(periodic, -120.0, 120.0)
    noisy = image.astype(np.float32) + periodic[..., None]
    return np.clip(noisy, 0, 255).astype(np.uint8)


def save_image(image: np.ndarray, output_path: Path) -> None:
    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise ValueError(f"Failed to write image: {output_path}")


def generate_for_image(
    image_path: Path,
    output_dirs: dict[str, Path],
    random_state: random.Random,
    np_rng: np.random.Generator,
) -> dict[str, int]:
    image = load_image(image_path)
    base_name = image_path.stem
    counts = {noise_type: 0 for noise_type in NOISE_TYPES}

    for variant_idx in range(1, 3):
        sigma = sample_gaussian_sigma(random_state)
        gaussian_image = add_gaussian_noise(image, sigma=sigma, rng=np_rng)
        gaussian_name = f"{base_name}_g{variant_idx}_s{sigma}.png"
        save_image(gaussian_image, output_dirs["gaussian"] / gaussian_name)
        counts["gaussian"] += 1

    salt_pepper_image = add_salt_pepper_noise(image, rng=np_rng)
    save_image(salt_pepper_image, output_dirs["salt_pepper"] / f"{base_name}.png")
    counts["salt_pepper"] += 1

    speckle_image = add_speckle_noise(image, rng=np_rng)
    save_image(speckle_image, output_dirs["speckle"] / f"{base_name}.png")
    counts["speckle"] += 1

    periodic_image = add_periodic_noise(image, rng=np_rng)
    save_image(periodic_image, output_dirs["periodic"] / f"{base_name}.png")
    counts["periodic"] += 1

    return counts


def generate_classifier_variants_for_image(
    image_path: Path,
    output_dirs: dict[str, Path],
    random_state: random.Random,
    np_rng: np.random.Generator,
    gaussian_variants: int,
    salt_pepper_variants: int,
    speckle_variants: int,
    periodic_variants: int,
) -> dict[str, int]:
    image = load_image(image_path)
    base_name = image_path.stem
    counts = {noise_type: 0 for noise_type in NOISE_TYPES}

    for variant_idx in range(1, gaussian_variants + 1):
        sigma = sample_gaussian_sigma(random_state)
        gaussian_image = add_gaussian_noise(image, sigma=sigma, rng=np_rng)
        save_image(gaussian_image, output_dirs["gaussian"] / f"{base_name}_g{variant_idx}_s{sigma}.png")
        counts["gaussian"] += 1

    for variant_idx in range(1, salt_pepper_variants + 1):
        salt_pepper_image = add_salt_pepper_noise(image, rng=np_rng)
        save_image(salt_pepper_image, output_dirs["salt_pepper"] / f"{base_name}_sp{variant_idx}.png")
        counts["salt_pepper"] += 1

    for variant_idx in range(1, speckle_variants + 1):
        speckle_image = add_speckle_noise(image, rng=np_rng)
        save_image(speckle_image, output_dirs["speckle"] / f"{base_name}_sk{variant_idx}.png")
        counts["speckle"] += 1

    for variant_idx in range(1, periodic_variants + 1):
        periodic_image = add_periodic_noise(image, rng=np_rng)
        save_image(periodic_image, output_dirs["periodic"] / f"{base_name}_pd{variant_idx}.png")
        counts["periodic"] += 1

    return counts


def print_classifier_generation_note() -> None:
    print("Classifier synthetic note:")
    print("- Speckle now uses wider variance bands plus layered multiplicative grain.")
    print("- Periodic now mixes oriented stripe directions, multi-sine components, and mild phase modulation.")
    print("- These changes make speckle and periodic visually more distinctive for classifier training.")


def print_periodic_generation_note() -> None:
    print("Periodic denoiser synthetic note:")
    print("- One periodic variant per clean image is generated for denoiser training.")
    print("- Pattern uses oriented stripe directions with 1-2 sinusoidal components and mild modulation.")


def print_periodic_v3_generation_note(variants_per_image: int) -> None:
    print("Periodic v3 denoiser synthetic note:")
    print(f"- {variants_per_image} stronger periodic variants are generated per clean image.")
    print("- Pattern families include oriented stripes, grid interference, moire-like mixes, and checker/grid hybrids.")
    print("- Frequency and amplitude ranges are intentionally more aggressive to prioritize periodic artefact suppression.")


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dirs = get_output_dirs(args.output_root.resolve())
    classifier_output_dirs = get_classifier_output_dirs(args.classifier_output_root.resolve())
    periodic_v3_output_dir = args.periodic_v3_output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    ensure_directories(list(output_dirs.values()))
    if not args.skip_classifier_dataset:
        ensure_directories(list(classifier_output_dirs.values()))
    if args.generate_periodic_v3:
        ensure_directories([periodic_v3_output_dir])
    images = list_clean_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No .jpg, .jpeg, or .png images found in: {input_dir}")

    random_state = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    totals = {noise_type: 0 for noise_type in NOISE_TYPES}
    classifier_totals = {noise_type: 0 for noise_type in NOISE_TYPES}
    periodic_v3_total = 0

    for image_path in images:
        counts = generate_for_image(
            image_path=image_path,
            output_dirs=output_dirs,
            random_state=random_state,
            np_rng=np_rng,
        )
        for noise_type, count in counts.items():
            totals[noise_type] += count

        if not args.skip_classifier_dataset:
            classifier_counts = generate_classifier_variants_for_image(
                image_path=image_path,
                output_dirs=classifier_output_dirs,
                random_state=random_state,
                np_rng=np_rng,
                gaussian_variants=args.classifier_gaussian_variants,
                salt_pepper_variants=args.classifier_salt_pepper_variants,
                speckle_variants=args.classifier_speckle_variants,
                periodic_variants=args.classifier_periodic_variants,
            )
            for noise_type, count in classifier_counts.items():
                classifier_totals[noise_type] += count

        if args.generate_periodic_v3:
            image = load_image(image_path)
            base_name = image_path.stem
            for variant_idx in range(1, args.periodic_v3_variants_per_image + 1):
                periodic_v3_image = add_periodic_noise_v3(image, rng=np_rng)
                save_image(periodic_v3_image, periodic_v3_output_dir / f"{base_name}_pv3_{variant_idx}.png")
                periodic_v3_total += 1

    print(f"Processed {len(images)} clean images from {input_dir}")
    print(f"Gaussian outputs: {totals['gaussian']}")
    print(f"Salt-pepper outputs: {totals['salt_pepper']}")
    print(f"Speckle outputs: {totals['speckle']}")
    print(f"Periodic outputs: {totals['periodic']}")
    print_periodic_generation_note()
    if args.generate_periodic_v3:
        print(f"Periodic v3 dataset root: {periodic_v3_output_dir}")
        print(f"Periodic v3 outputs: {periodic_v3_total}")
        print_periodic_v3_generation_note(args.periodic_v3_variants_per_image)
    if not args.skip_classifier_dataset:
        print(f"Classifier dataset root: {args.classifier_output_root.resolve()}")
        print(f"Classifier gaussian outputs: {classifier_totals['gaussian']}")
        print(f"Classifier salt-pepper outputs: {classifier_totals['salt_pepper']}")
        print(f"Classifier speckle outputs: {classifier_totals['speckle']}")
        print(f"Classifier periodic outputs: {classifier_totals['periodic']}")
        print_classifier_generation_note()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
