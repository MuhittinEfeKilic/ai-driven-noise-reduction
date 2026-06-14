from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


LABELS = ("CLEAN", "NOISY", "DENOISED")
TEXT_COLOR = (255, 255, 255)
TEXT_SCALE = 0.9
TEXT_THICKNESS = 2
TEXT_MARGIN_X = 16
TEXT_MARGIN_Y = 32
TOP_BAND_HEIGHT = 48


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a side-by-side comparison video.")
    parser.add_argument("--clean", required=True, help="Original clean video path.")
    parser.add_argument("--noisy", required=True, help="Noisy video path.")
    parser.add_argument("--denoised", required=True, help="Denoised video path.")
    parser.add_argument("--output", default="outputs/video/comparison.mp4", help="Preferred output comparison video path.")
    return parser.parse_args()


def open_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    return capture


def resolve_fps(capture: cv2.VideoCapture) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    return fps if fps > 0.0 else 30.0


def resolve_frame_count(capture: cv2.VideoCapture) -> int:
    return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))


def resize_to(frame: cv2.typing.MatLike, size: tuple[int, int]) -> cv2.typing.MatLike:
    width, height = size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def add_label(frame: cv2.typing.MatLike, label: str) -> cv2.typing.MatLike:
    labeled = frame.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], TOP_BAND_HEIGHT), (16, 16, 16), thickness=-1)
    cv2.putText(
        labeled,
        label,
        (TEXT_MARGIN_X, TEXT_MARGIN_Y),
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        TEXT_COLOR,
        TEXT_THICKNESS,
        cv2.LINE_AA,
    )
    return labeled


def build_comparison_frame(
    clean_frame: cv2.typing.MatLike,
    noisy_frame: cv2.typing.MatLike,
    denoised_frame: cv2.typing.MatLike,
) -> cv2.typing.MatLike:
    target_height, target_width = clean_frame.shape[:2]
    target_size = (target_width, target_height)

    noisy_frame = resize_to(noisy_frame, target_size)
    denoised_frame = resize_to(denoised_frame, target_size)

    clean_panel = add_label(clean_frame, LABELS[0])
    noisy_panel = add_label(noisy_frame, LABELS[1])
    denoised_panel = add_label(denoised_frame, LABELS[2])
    return cv2.hconcat([clean_panel, noisy_panel, denoised_panel])


def _open_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    suffix = output_path.suffix.lower()
    if suffix == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif suffix == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        raise ValueError(f"Unsupported video output extension: {output_path.suffix}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        frame_size,
    )
    return writer


def create_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> tuple[cv2.VideoWriter, Path]:
    preferred_mp4_path = output_path.with_suffix(".mp4")
    fallback_avi_path = output_path.with_suffix(".avi")

    writer = _open_writer(preferred_mp4_path, fps, frame_size)
    if writer.isOpened():
        print(f"Video writer opened with MP4 output: {preferred_mp4_path}")
        return writer, preferred_mp4_path

    writer.release()
    print(f"MP4 writer failed, falling back to AVI output: {fallback_avi_path}")
    writer = _open_writer(fallback_avi_path, fps, frame_size)
    if not writer.isOpened():
        writer.release()
        raise ValueError(f"Failed to open video writer for both MP4 and AVI outputs near: {output_path}")
    return writer, fallback_avi_path


def main() -> int:
    args = parse_args()
    clean_path = Path(args.clean).expanduser().resolve()
    noisy_path = Path(args.noisy).expanduser().resolve()
    denoised_path = Path(args.denoised).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    for path in (clean_path, noisy_path, denoised_path):
        if not path.exists():
            print(f"Error: input video not found: {path}")
            return 1

    clean_capture = open_capture(clean_path)
    noisy_capture = open_capture(noisy_path)
    denoised_capture = open_capture(denoised_path)

    writer: cv2.VideoWriter | None = None
    actual_output_path = output_path.with_suffix(".mp4")
    try:
        fps = resolve_fps(clean_capture)
        total_frames = min(
            resolve_frame_count(clean_capture),
            resolve_frame_count(noisy_capture),
            resolve_frame_count(denoised_capture),
        )
        print(f"Using FPS: {fps:.3f}")
        print(f"Using shortest frame count: {total_frames}")

        with tqdm(total=total_frames if total_frames > 0 else None, desc="Building comparison video", unit="frame") as progress:
            while True:
                ok_clean, clean_frame = clean_capture.read()
                ok_noisy, noisy_frame = noisy_capture.read()
                ok_denoised, denoised_frame = denoised_capture.read()
                if not (ok_clean and ok_noisy and ok_denoised):
                    break

                comparison_frame = build_comparison_frame(clean_frame, noisy_frame, denoised_frame)
                if writer is None:
                    frame_height, frame_width = comparison_frame.shape[:2]
                    writer, actual_output_path = create_writer(output_path, fps, (frame_width, frame_height))
                writer.write(comparison_frame)
                progress.update(1)
    finally:
        clean_capture.release()
        noisy_capture.release()
        denoised_capture.release()
        if writer is not None:
            writer.release()

    print(f"Output comparison video: {actual_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
