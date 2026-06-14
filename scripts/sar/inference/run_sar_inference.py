from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import rasterio
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "rasterio is required for SAR ENVI inference. Install it with: pip install rasterio"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "sar" / "training"
if str(TRAINING_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_SCRIPT_DIR))

from train_residual_unet_sar import ResidualUNet  # noqa: E402


EPSILON = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sliding-window SAR denoising inference.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--incidence-image", type=Path, default=None)
    parser.add_argument("--column-profile-image", type=Path, default=None)
    parser.add_argument("--soft-swath-image", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sar/inference"))
    parser.add_argument("--use-incidence", action="store_true")
    parser.add_argument("--use-incidence-scalar", action="store_true")
    parser.add_argument("--use-column-profile", action="store_true")
    parser.add_argument("--use-soft-swath", action="store_true")
    parser.add_argument("--predict-residual-noise", action="store_true")
    parser.add_argument("--predict-correction-residual", action="store_true")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--apply-clahe", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.predict_residual_noise and args.predict_correction_residual:
        raise ValueError(
            "--predict-residual-noise and --predict-correction-residual cannot be used together."
        )
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    if not args.input_image.exists():
        raise FileNotFoundError(f"Input image not found: {args.input_image}")
    if (args.use_incidence or args.use_incidence_scalar) and args.incidence_image is None:
        raise ValueError(
            "--incidence-image is required when --use-incidence or --use-incidence-scalar is set."
        )
    if args.use_column_profile and args.column_profile_image is None:
        raise ValueError("--column-profile-image is required when --use-column-profile is set.")
    if args.use_soft_swath and args.soft_swath_image is None:
        raise ValueError("--soft-swath-image is required when --use-soft-swath is set.")
    if args.incidence_image is not None and not args.incidence_image.exists():
        raise FileNotFoundError(f"Incidence image not found: {args.incidence_image}")
    if args.column_profile_image is not None and not args.column_profile_image.exists():
        raise FileNotFoundError(f"Column profile image not found: {args.column_profile_image}")
    if args.soft_swath_image is not None and not args.soft_swath_image.exists():
        raise FileNotFoundError(f"Soft swath image not found: {args.soft_swath_image}")
    if args.patch_size <= 0 or args.stride <= 0:
        raise ValueError("--patch-size and --stride must be positive integers.")


def normalize_to_unit(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value - min_value <= EPSILON:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - min_value) / (max_value - min_value)).astype(np.float32)


def percentile_normalize(image: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    lower_value = float(np.percentile(image, lower))
    upper_value = float(np.percentile(image, upper))
    if upper_value - lower_value <= EPSILON:
        return normalize_to_unit(image)
    normalized = (image - lower_value) / (upper_value - lower_value)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    enhanced = clahe.apply(image_uint8)
    return enhanced.astype(np.float32) / 255.0


def load_envi_image(image_path: Path, apply_log1p: bool = True) -> np.ndarray:
    with rasterio.open(image_path) as dataset:
        image = dataset.read(1).astype(np.float32)

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = normalize_to_unit(image)
    if apply_log1p:
        image = np.log1p(image)
        image = normalize_to_unit(image)
    return image.astype(np.float32)


def load_grayscale_png(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read PNG image: {image_path}")
    return normalize_to_unit(image.astype(np.float32))


def compute_input_channels(
    use_incidence: bool,
    use_incidence_scalar: bool,
    use_column_profile: bool,
    use_soft_swath: bool,
) -> int:
    input_channels = 1
    if use_incidence or use_incidence_scalar:
        input_channels += 1
    if use_column_profile:
        input_channels += 1
    if use_soft_swath:
        input_channels += 1
    return input_channels


def infer_checkpoint_input_channels(checkpoint: dict) -> int:
    weight = checkpoint["model_state_dict"]["enc1.proj.0.weight"]
    return int(weight.shape[1])


def load_model(
    model_path: Path,
    use_incidence: bool,
    use_incidence_scalar: bool,
    use_column_profile: bool,
    use_soft_swath: bool,
    predict_residual_noise: bool,
    predict_correction_residual: bool,
    device: torch.device,
) -> ResidualUNet:
    checkpoint = torch.load(model_path, map_location=device)
    effective_use_incidence = use_incidence and not use_incidence_scalar
    checkpoint_use_incidence = bool(checkpoint.get("use_incidence", False))
    if checkpoint_use_incidence != effective_use_incidence:
        raise ValueError(
            "Checkpoint incidence configuration does not match CLI. "
            f"checkpoint={checkpoint_use_incidence}, cli={effective_use_incidence}"
        )
    checkpoint_use_incidence_scalar = bool(checkpoint.get("use_incidence_scalar", False))
    if checkpoint_use_incidence_scalar != use_incidence_scalar:
        raise ValueError(
            "Checkpoint incidence-scalar configuration does not match CLI. "
            f"checkpoint={checkpoint_use_incidence_scalar}, cli={use_incidence_scalar}"
        )
    checkpoint_use_column_profile = bool(checkpoint.get("use_column_profile", False))
    if checkpoint_use_column_profile != use_column_profile:
        raise ValueError(
            "Checkpoint column-profile configuration does not match CLI. "
            f"checkpoint={checkpoint_use_column_profile}, cli={use_column_profile}"
        )
    checkpoint_use_soft_swath = bool(checkpoint.get("use_soft_swath", False))
    if checkpoint_use_soft_swath != use_soft_swath:
        raise ValueError(
            "Checkpoint soft-swath configuration does not match CLI. "
            f"checkpoint={checkpoint_use_soft_swath}, cli={use_soft_swath}"
        )
    checkpoint_predict_residual_noise = bool(checkpoint.get("predict_residual_noise", False))
    if checkpoint_predict_residual_noise != predict_residual_noise:
        raise ValueError(
            "Checkpoint residual-noise configuration does not match CLI. "
            f"checkpoint={checkpoint_predict_residual_noise}, cli={predict_residual_noise}"
        )
    checkpoint_predict_correction_residual = bool(
        checkpoint.get("predict_correction_residual", False)
    )
    if checkpoint_predict_correction_residual != predict_correction_residual:
        raise ValueError(
            "Checkpoint correction-residual configuration does not match CLI. "
            f"checkpoint={checkpoint_predict_correction_residual}, cli={predict_correction_residual}"
        )

    input_channels = compute_input_channels(
        use_incidence,
        use_incidence_scalar,
        use_column_profile,
        use_soft_swath,
    )
    checkpoint_input_channels = int(
        checkpoint.get("input_channels", infer_checkpoint_input_channels(checkpoint))
    )
    if checkpoint_input_channels != input_channels:
        raise ValueError(
            "Checkpoint input channel count does not match CLI. "
            f"checkpoint={checkpoint_input_channels}, cli={input_channels}"
        )

    model = ResidualUNet(in_channels=input_channels, out_channels=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def pad_for_sliding_window(image: np.ndarray, patch_size: int, stride: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape
    pad_height = max(0, patch_size - height)
    pad_width = max(0, patch_size - width)

    if height > patch_size:
        remainder_h = (height - patch_size) % stride
        if remainder_h != 0:
            pad_height += stride - remainder_h
    if width > patch_size:
        remainder_w = (width - patch_size) % stride
        if remainder_w != 0:
            pad_width += stride - remainder_w

    padded = np.pad(image, ((0, pad_height), (0, pad_width)), mode="reflect")
    return padded, (height, width)


def save_png(output_path: Path, image: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = np.clip(image, 0.0, 1.0)
    image_uint8 = (image_uint8 * 255.0).round().astype(np.uint8)
    success = cv2.imwrite(str(output_path), image_uint8)
    if not success:
        raise IOError(f"Failed to write image: {output_path}")


def prepare_visualization_image(image: np.ndarray, apply_clahe_flag: bool) -> np.ndarray:
    visual = percentile_normalize(image)
    if apply_clahe_flag:
        visual = apply_clahe(visual)
        visual = percentile_normalize(visual)
    return np.clip(visual, 0.0, 1.0).astype(np.float32)


@torch.no_grad()
def sliding_window_inference(
    model: ResidualUNet,
    noisy_image: np.ndarray,
    incidence_image: np.ndarray | None,
    use_incidence_scalar: bool,
    column_profile_image: np.ndarray | None,
    soft_swath_image: np.ndarray | None,
    patch_size: int,
    stride: int,
    predict_residual_noise: bool,
    predict_correction_residual: bool,
    device: torch.device,
) -> np.ndarray:
    padded_noisy, original_shape = pad_for_sliding_window(noisy_image, patch_size, stride)
    padded_incidence = None
    if incidence_image is not None:
        padded_incidence, _ = pad_for_sliding_window(incidence_image, patch_size, stride)
        if padded_incidence.shape != padded_noisy.shape:
            raise ValueError("Incidence image shape does not match noisy image shape after padding.")
    padded_column_profile = None
    if column_profile_image is not None:
        padded_column_profile, _ = pad_for_sliding_window(column_profile_image, patch_size, stride)
        if padded_column_profile.shape != padded_noisy.shape:
            raise ValueError("Column profile image shape does not match noisy image shape after padding.")
    padded_soft_swath = None
    if soft_swath_image is not None:
        padded_soft_swath, _ = pad_for_sliding_window(soft_swath_image, patch_size, stride)
        if padded_soft_swath.shape != padded_noisy.shape:
            raise ValueError("Soft swath image shape does not match noisy image shape after padding.")

    height, width = padded_noisy.shape
    output_sum = np.zeros((height, width), dtype=np.float32)
    output_count = np.zeros((height, width), dtype=np.float32)

    for top in range(0, height - patch_size + 1, stride):
        for left in range(0, width - patch_size + 1, stride):
            noisy_patch = padded_noisy[top : top + patch_size, left : left + patch_size]
            input_channels = [noisy_patch]

            if padded_incidence is not None:
                incidence_patch = padded_incidence[top : top + patch_size, left : left + patch_size]
                if use_incidence_scalar:
                    incidence_patch = np.full_like(
                        incidence_patch,
                        float(incidence_patch.mean()),
                        dtype=np.float32,
                    )
                input_channels.append(incidence_patch)
            if padded_column_profile is not None:
                column_profile_patch = padded_column_profile[
                    top : top + patch_size, left : left + patch_size
                ]
                input_channels.append(column_profile_patch)
            if padded_soft_swath is not None:
                soft_swath_patch = padded_soft_swath[top : top + patch_size, left : left + patch_size]
                input_channels.append(soft_swath_patch)

            stacked = np.stack(input_channels, axis=0).astype(np.float32)
            input_tensor = torch.from_numpy(stacked).unsqueeze(0).to(device)

            prediction = model(input_tensor).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            if predict_residual_noise:
                prediction = np.clip(noisy_patch - prediction, 0.0, 1.0).astype(np.float32)
            elif predict_correction_residual:
                prediction = np.clip(noisy_patch + prediction, 0.0, 1.0).astype(np.float32)
            output_sum[top : top + patch_size, left : left + patch_size] += prediction
            output_count[top : top + patch_size, left : left + patch_size] += 1.0

    merged = output_sum / np.maximum(output_count, 1.0)
    original_height, original_width = original_shape
    return merged[:original_height, :original_width]


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.predict_residual_noise:
        print("Residual noise inference mode enabled")
    if args.predict_correction_residual:
        print("Correction residual inference mode enabled")

    noisy_image = load_envi_image(args.input_image, apply_log1p=True)
    incidence_image = None
    if args.use_incidence or args.use_incidence_scalar:
        incidence_image = load_envi_image(args.incidence_image, apply_log1p=False)
        if incidence_image.shape != noisy_image.shape:
            raise ValueError(
                f"Incidence image shape {incidence_image.shape} does not match noisy image shape {noisy_image.shape}."
            )
    column_profile_image = None
    if args.use_column_profile:
        column_profile_image = load_grayscale_png(args.column_profile_image)
        if column_profile_image.shape != noisy_image.shape:
            raise ValueError(
                "Column profile image shape "
                f"{column_profile_image.shape} does not match noisy image shape {noisy_image.shape}."
            )
    soft_swath_image = None
    if args.use_soft_swath:
        soft_swath_image = load_grayscale_png(args.soft_swath_image)
        if soft_swath_image.shape != noisy_image.shape:
            raise ValueError(
                f"Soft swath image shape {soft_swath_image.shape} does not match noisy image shape {noisy_image.shape}."
            )

    input_channels = compute_input_channels(
        args.use_incidence,
        args.use_incidence_scalar,
        args.use_column_profile,
        args.use_soft_swath,
    )
    print(f"Input channels: {input_channels}")
    print(
        "Channels used: noisy"
        f"{', incidence_scalar' if args.use_incidence_scalar else ''}"
        f"{', incidence' if args.use_incidence and not args.use_incidence_scalar else ''}"
        f"{', column_profile' if args.use_column_profile else ''}"
        f"{', soft_swath' if args.use_soft_swath else ''}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        args.model_path,
        args.use_incidence,
        args.use_incidence_scalar,
        args.use_column_profile,
        args.use_soft_swath,
        args.predict_residual_noise,
        args.predict_correction_residual,
        device,
    )
    denoised_image = sliding_window_inference(
        model=model,
        noisy_image=noisy_image,
        incidence_image=incidence_image,
        use_incidence_scalar=args.use_incidence_scalar,
        column_profile_image=column_profile_image,
        soft_swath_image=soft_swath_image,
        patch_size=args.patch_size,
        stride=args.stride,
        predict_residual_noise=args.predict_residual_noise,
        predict_correction_residual=args.predict_correction_residual,
        device=device,
    )

    denoised_normalized = normalize_to_unit(denoised_image)
    raw_difference_map = np.abs(noisy_image - denoised_normalized).astype(np.float32)
    denoised_visual = prepare_visualization_image(denoised_image, args.apply_clahe)
    denoised_normalized_visual = prepare_visualization_image(denoised_normalized, args.apply_clahe)
    difference_map_visual = prepare_visualization_image(raw_difference_map, args.apply_clahe)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_png(args.output_dir / "denoised.png", denoised_visual)
    save_png(args.output_dir / "denoised_normalized.png", denoised_normalized_visual)
    save_png(args.output_dir / "difference_map.png", difference_map_visual)

    print(f"Input shape: {tuple(noisy_image.shape)}")
    print(f"Saved denoised image: {args.output_dir / 'denoised.png'}")
    print(f"Saved normalized denoised image: {args.output_dir / 'denoised_normalized.png'}")
    print(f"Saved difference map: {args.output_dir / 'difference_map.png'}")


if __name__ == "__main__":
    main()
