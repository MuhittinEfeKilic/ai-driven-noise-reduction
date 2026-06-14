from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from src.video.processing import process_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the existing image denoising pipeline on every frame of a video.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument(
        "--output",
        default="outputs/video/denoised_video.mp4",
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--force_noise_type",
        choices=("gaussian", "salt_pepper", "speckle", "periodic"),
        default=None,
        help="Skip classification and force a single denoiser for all frames.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Number of denoising passes to apply per frame.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not video_path.exists():
        print(f"Error: input video not found: {video_path}")
        return 1

    progress_bars: dict[str, tqdm] = {}

    def on_progress(stage: str, current: int, total: int) -> None:
        bar = progress_bars.get(stage)
        if bar is None:
            descriptions = {
                "extract": "Extracting frames",
                "denoise": "Denoising frames",
                "rebuild": "Rebuilding video",
            }
            bar = tqdm(total=total if total > 0 else None, desc=descriptions.get(stage, stage.title()), unit="frame")
            progress_bars[stage] = bar
        if total > 0 and bar.total != total:
            bar.total = total
        increment = max(0, current - bar.n)
        if increment > 0:
            bar.update(increment)

    try:
        process_video(
            video_path=video_path,
            output_path=output_path,
            force_noise_type=args.force_noise_type,
            passes=args.passes,
            progress_callback=on_progress,
        )
    except Exception as exc:
        print(f"Runtime error: {exc}")
        return 1
    finally:
        for bar in progress_bars.values():
            bar.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
