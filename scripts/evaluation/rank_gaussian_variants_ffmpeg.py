#!/usr/bin/env python3
"""Rank Gaussian synthetic variants against clean BSD500 references using ffmpeg PSNR."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


PATTERN = re.compile(r"^(?P<base>.+?)_(?P<variant>g\d+)_s(?P<sigma>\d+)$")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PSNR_PATTERN = re.compile(r"average:(?P<psnr>[-+.\dinf]+)")


@dataclass(frozen=True)
class VariantResult:
    base: str
    variant: str
    sigma: int
    noisy_path: Path
    clean_path: Path
    psnr: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean/bsd500"))
    parser.add_argument("--noisy-dir", type=Path, default=Path("data/synthetic/gaussian"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/gaussian_variant_quality.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/gaussian_variant_quality_summary.txt"))
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def build_clean_index(clean_dir: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted(clean_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }


def collect_jobs(clean_dir: Path, noisy_dir: Path) -> tuple[list[tuple[str, str, int, Path, Path]], list[str]]:
    clean_index = build_clean_index(clean_dir)
    jobs: list[tuple[str, str, int, Path, Path]] = []
    skipped: list[str] = []

    for noisy_path in sorted(noisy_dir.iterdir()):
        if not noisy_path.is_file() or noisy_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        match = PATTERN.match(noisy_path.stem)
        if match is None:
            skipped.append(str(noisy_path))
            continue
        base = match.group("base")
        clean_path = clean_index.get(base)
        if clean_path is None:
            skipped.append(str(noisy_path))
            continue
        jobs.append((base, match.group("variant"), int(match.group("sigma")), noisy_path, clean_path))

    return jobs, skipped


def ffmpeg_psnr(noisy_path: Path, clean_path: Path) -> float:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "info",
        "-i",
        str(noisy_path),
        "-i",
        str(clean_path),
        "-lavfi",
        "[0:v]format=rgb24[noisy];[1:v]format=rgb24[clean];[noisy][clean]psnr",
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {noisy_path}: {output.strip()}")
    match = PSNR_PATTERN.search(output)
    if match is None:
        raise RuntimeError(f"Could not parse PSNR for {noisy_path}: {output.strip()}")
    value = match.group("psnr")
    return math.inf if value == "inf" else float(value)


def evaluate_job(job: tuple[str, str, int, Path, Path]) -> VariantResult:
    base, variant, sigma, noisy_path, clean_path = job
    return VariantResult(
        base=base,
        variant=variant,
        sigma=sigma,
        noisy_path=noisy_path,
        clean_path=clean_path,
        psnr=ffmpeg_psnr(noisy_path, clean_path),
    )


def mean(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite_values) if finite_values else math.inf


def main() -> int:
    args = parse_args()
    jobs, skipped = collect_jobs(args.clean_dir, args.noisy_dir)
    if not jobs:
        raise SystemExit("No Gaussian variant jobs found.")

    results: list[VariantResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(evaluate_job, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"Measured {index}/{len(futures)}")

    results.sort(key=lambda item: (item.base, item.sigma, item.variant, item.noisy_path.name))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["base", "variant", "sigma", "psnr", "noisy_path", "clean_path"])
        for result in results:
            writer.writerow([result.base, result.variant, result.sigma, f"{result.psnr:.6f}", result.noisy_path, result.clean_path])

    by_base: dict[str, list[VariantResult]] = {}
    for result in results:
        by_base.setdefault(result.base, []).append(result)

    best_by_base = {
        base: max(items, key=lambda item: (item.psnr, -item.sigma, item.noisy_path.name))
        for base, items in by_base.items()
    }
    by_variant: dict[str, list[VariantResult]] = {}
    for result in results:
        by_variant.setdefault(result.variant, []).append(result)

    complete_groups = sum(1 for items in by_base.values() if len(items) == 4)
    incomplete_groups = {base: items for base, items in by_base.items() if len(items) != 4}
    best_values = list(best_by_base.values())
    sigma_values = [result.sigma for result in results]
    best_sigma_values = [result.sigma for result in best_values]

    lines = [
        "Gaussian synthetic variant quality",
        f"Measured variants: {len(results)}",
        f"Matched clean images: {len(by_base)}",
        f"Clean images with 4 variants: {complete_groups}",
        f"Clean images with missing/non-4 variants: {len(incomplete_groups)}",
        f"Skipped files: {len(skipped)}",
        "",
        "Overall:",
        f"- Avg PSNR: {mean([result.psnr for result in results]):.4f}",
        f"- Sigma range: {min(sigma_values)}..{max(sigma_values)}",
        f"- Avg sigma: {statistics.fmean(sigma_values):.2f}",
        "",
        "Best per clean image:",
        f"- Avg best PSNR: {mean([result.psnr for result in best_values]):.4f}",
        f"- Best sigma range: {min(best_sigma_values)}..{max(best_sigma_values)}",
        f"- Avg best sigma: {statistics.fmean(best_sigma_values):.2f}",
        "",
        "By g-label:",
    ]
    for variant, items in sorted(by_variant.items()):
        lines.append(
            f"- {variant}: count={len(items)}, avg_psnr={mean([item.psnr for item in items]):.4f}, "
            f"avg_sigma={statistics.fmean([item.sigma for item in items]):.2f}"
        )

    best_sigma_counts: dict[int, int] = {}
    for result in best_values:
        best_sigma_counts[result.sigma] = best_sigma_counts.get(result.sigma, 0) + 1
    lines.extend(["", "Most frequent winning sigmas:"])
    for sigma, count in sorted(best_sigma_counts.items(), key=lambda item: (-item[1], item[0]))[:12]:
        lines.append(f"- s{sigma}: {count}")

    lines.extend(["", "Best file for each clean image:"])
    for base, result in sorted(best_by_base.items()):
        lines.append(f"{base},{result.noisy_path.name},s{result.sigma},{result.psnr:.4f}")

    if incomplete_groups:
        lines.extend(["", "Incomplete groups:"])
        for base, items in sorted(incomplete_groups.items()):
            lines.append(f"- {base}: {len(items)} variants")

    args.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
