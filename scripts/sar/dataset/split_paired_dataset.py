from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


SUBDIRECTORIES = (
    "noisy",
    "target",
    "noisy_npy",
    "target_npy",
    "incidence",
    "incidence_scalar",
    "esa_cleared",
    "column_profile",
    "soft_swath",
)


@dataclass(frozen=True)
class PairedPatchSet:
    filename: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split paired SAR patches into train/val/test sets.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/sar/processed/paired/patches"),
        help="Root directory containing paired SAR patch folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/sar/processed/paired"),
        help="Root directory where train/val/test folders will be created.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")


def list_png_names(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {path.name for path in directory.glob("*.png") if path.is_file()}


def list_patch_names(directory: Path, subdir_name: str) -> set[str]:
    if subdir_name in {"incidence_scalar", "noisy_npy", "target_npy"}:
        return {path.name for path in directory.glob("*.npy") if path.is_file()}
    return list_png_names(directory)


def filename_for_subdir(patch_filename: str, subdir_name: str) -> str:
    if subdir_name in {"incidence_scalar", "noisy_npy", "target_npy"}:
        return f"{Path(patch_filename).stem}.npy"
    return patch_filename


def collect_patch_sets(input_root: Path) -> tuple[list[PairedPatchSet], list[str]]:
    noisy_dir = input_root / "noisy"
    target_dir = input_root / "target"
    if not noisy_dir.exists() or not target_dir.exists():
        raise FileNotFoundError("Both noisy and target directories are required.")

    noisy_names = list_png_names(noisy_dir)
    target_names = list_png_names(target_dir)
    shared_names = noisy_names & target_names
    if not shared_names:
        raise FileNotFoundError("No matching noisy/target PNG patch pairs were found.")

    skipped: list[str] = []
    patch_sets: list[PairedPatchSet] = []
    optional_patch_names = {
        subdir_name: list_patch_names(input_root / subdir_name, subdir_name)
        for subdir_name in SUBDIRECTORIES
        if subdir_name not in {"noisy", "target"} and (input_root / subdir_name).exists()
    }

    for filename in sorted(shared_names):
        missing_optional_patch = False
        for subdir_name, optional_names in optional_patch_names.items():
            if filename_for_subdir(filename, subdir_name) not in optional_names:
                skipped.append(filename)
                missing_optional_patch = True
                break
        if missing_optional_patch:
            continue
        patch_sets.append(PairedPatchSet(filename=filename))

    if not patch_sets:
        raise FileNotFoundError("No fully matched paired SAR patch sets were found.")
    return patch_sets, skipped


def split_patch_sets(
    patch_sets: list[PairedPatchSet],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[PairedPatchSet], list[PairedPatchSet], list[PairedPatchSet]]:
    shuffled = list(patch_sets)
    random.Random(seed).shuffle(shuffled)

    total_count = len(shuffled)
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def copy_split(
    samples: list[PairedPatchSet],
    input_root: Path,
    output_root: Path,
    split_name: str,
) -> int:
    copied = 0
    available_subdirs = [name for name in SUBDIRECTORIES if (input_root / name).exists()]

    for sample in samples:
        skipped_existing = False
        for subdir_name in available_subdirs:
            source_filename = filename_for_subdir(sample.filename, subdir_name)
            source_path = input_root / subdir_name / source_filename
            destination_dir = output_root / split_name / subdir_name
            destination_path = destination_dir / source_filename
            destination_dir.mkdir(parents=True, exist_ok=True)

            if destination_path.exists():
                print(f"Warning: file already exists, skipping: {destination_path}")
                skipped_existing = True
                continue

            shutil.copy2(source_path, destination_path)

        if not skipped_existing:
            copied += 1

    return copied


def main() -> None:
    args = parse_args()
    validate_args(args)

    patch_sets, skipped = collect_patch_sets(args.input_root)
    train_items, val_items, test_items = split_patch_sets(
        patch_sets=patch_sets,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_count = copy_split(train_items, args.input_root, args.output_root, "train")
    val_count = copy_split(val_items, args.input_root, args.output_root, "val")
    test_count = copy_split(test_items, args.input_root, args.output_root, "test")

    print(f"Matched patch groups: {len(patch_sets)}")
    print(f"Skipped unmatched groups: {len(skipped)}")
    print(f"Train groups copied: {train_count}")
    print(f"Validation groups copied: {val_count}")
    print(f"Test groups copied: {test_count}")


if __name__ == "__main__":
    main()
