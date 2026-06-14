from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


SUPPORTED_EXTENSIONS = {".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split processed SAR patches into train, val, and test directories."
    )
    parser.add_argument(
        "--patch-dir",
        type=Path,
        default=Path("data/sar/processed/patches"),
        help="Directory containing generated SAR patches.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/sar/processed"),
        help="Root directory for train/val/test outputs.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling patch files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.patch_dir.exists():
        raise FileNotFoundError(f"Patch directory does not exist: {args.patch_dir}")

    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")

    total_ratio = sum(ratios)
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")


def list_patch_files(patch_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in patch_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def split_patch_files(
    patch_files: list[Path],
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[Path], list[Path], list[Path]]:
    total_count = len(patch_files)
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)

    train_files = patch_files[:train_end]
    val_files = patch_files[train_end:val_end]
    test_files = patch_files[val_end:]
    return train_files, val_files, test_files


def copy_files(files: list[Path], destination_dir: Path) -> int:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied_count = 0

    for file_path in files:
        destination_path = destination_dir / file_path.name
        if destination_path.exists():
            print(f"Warning: file already exists, skipping: {destination_path}")
            continue

        shutil.copy2(file_path, destination_path)
        copied_count += 1

    return copied_count


def main() -> None:
    args = parse_args()
    validate_args(args)

    patch_files = list_patch_files(args.patch_dir)
    random.Random(args.seed).shuffle(patch_files)

    train_files, val_files, test_files = split_patch_files(
        patch_files=patch_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    train_count = copy_files(train_files, args.output_root / "train")
    val_count = copy_files(val_files, args.output_root / "val")
    test_count = copy_files(test_files, args.output_root / "test")

    print(f"Train patches copied: {train_count}")
    print(f"Validation patches copied: {val_count}")
    print(f"Test patches copied: {test_count}")


if __name__ == "__main__":
    main()
