from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
from PIL import Image

from src.inference.classifier_inference import ClassifierInference
from src.inference.denoiser_inference import DenoiserInference
from src.preprocessing.periodic_fft_filter import remove_periodic_noise_fft

PipelineOutput: TypeAlias = dict[str, Any]


class NoiseReductionPipeline:
    """End-to-end noise classification and denoising pipeline."""

    CLASSIFIER_CHECKPOINT = Path("models/classifiers/noise_classifier_best.pt")
    DENOISER_CHECKPOINTS: dict[str, Path] = {
        # Gaussian icin DnCNN yerine UNet tercih edildi.
        "gaussian": Path("models/denoisers/gaussian/gaussian_unet_best.pt"),
        # Salt & Pepper icin DnCNN yerine UNet tercih edildi.
        "salt_pepper": Path("models/denoisers/salt_pepper/salt_pepper_unet_best.pt"),
        # Speckle icin residual-hybrid UNet best checkpoint kullaniliyor.
        "speckle": Path("models/denoisers/speckle/speckle_unet_residual_hybrid_best.pt"),
        # Periodic icin FFT-guided NAFNet best checkpoint varsayilan olarak kullaniliyor.
        # Eski dual-branch residual U-Net checkpointi fallback/reference olarak korunuyor.
        "periodic": Path("models/denoisers/periodic/periodic_fft_guided_nafnet_best.pt"),
    }

    def __init__(
        self,
        classifier_path: str | Path | None = None,
        output_dir: str | Path = "outputs",
        device: str | None = None,
        periodic_residual_strength: float = 1.0,
    ) -> None:
        self.classifier_path = Path(classifier_path or self.CLASSIFIER_CHECKPOINT).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.device = device
        self.periodic_residual_strength = float(periodic_residual_strength)
        self._classifier: ClassifierInference | None = None
        self._denoisers: dict[tuple[str, float], DenoiserInference] = {}

    def get_denoiser_path(self, predicted_label: str) -> Path:
        """Return the denoiser checkpoint path for a predicted noise label."""
        denoiser_path = self.DENOISER_CHECKPOINTS.get(predicted_label)
        if denoiser_path is None:
            raise ValueError(f"Unsupported predicted noise type: {predicted_label}")
        return denoiser_path.expanduser().resolve()

    def ensure_output_dir(self) -> Path:
        """Create the pipeline output directory if it does not already exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def get_classifier(self) -> ClassifierInference:
        """Create and cache the classifier inference helper."""
        if self._classifier is None:
            self._classifier = ClassifierInference(model_path=self.classifier_path, device=self.device)
        return self._classifier

    def get_denoiser(self, predicted_noise_type: str, residual_strength: float) -> DenoiserInference:
        """Create and cache denoiser helpers keyed by label and residual strength."""
        cache_key = (predicted_noise_type, float(residual_strength))
        denoiser = self._denoisers.get(cache_key)
        if denoiser is None:
            denoiser = DenoiserInference(
                model_path=self.get_denoiser_path(predicted_noise_type),
                device=self.device,
                residual_strength=residual_strength,
            )
            self._denoisers[cache_key] = denoiser
        return denoiser

    def run(self, input_image_path: str | Path) -> dict[str, Any]:
        """Run classification and denoising for an input image path."""
        input_path = Path(input_image_path).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

        classifier = self.get_classifier()
        predicted_noise_type, classifier_confidence = classifier.predict_from_path(input_path)

        residual_strength = self.periodic_residual_strength if predicted_noise_type == "periodic" else 1.0
        denoiser = self.get_denoiser(predicted_noise_type, residual_strength)
        postprocessing_applied: str | None = None
        denoised_image = denoiser.run(input_path)
        denoiser_model_type = denoiser.model_name
        if predicted_noise_type == "periodic" and denoiser_model_type == "PeriodicFFTGuidedNAFNet":
            postprocessing_applied = "aggressive NAFNet periodic removal"
        elif predicted_noise_type == "periodic":
            denoised_image = self._apply_periodic_fft_postprocessing(denoised_image)
            denoised_image = self._apply_unsharp_mask(denoised_image)
            postprocessing_applied = "FFT + sharpening"
        denoiser_path = denoiser.model_path

        output_path = self.save_result(denoised_image, input_path)

        return {
            "input_path": str(input_path),
            "predicted_noise_type": predicted_noise_type,
            "classifier_confidence": classifier_confidence,
            "denoiser_path": str(denoiser_path),
            "denoiser_model_type": denoiser_model_type,
            "residual_strength": residual_strength,
            "postprocessing_applied": postprocessing_applied,
            "output_path": str(output_path),
        }

    @staticmethod
    def _apply_periodic_fft_postprocessing(image: Image.Image) -> Image.Image:
        """Apply the periodic FFT notch filter to the denoised output image."""
        image_rgb = image.convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
        filtered_bgr = remove_periodic_noise_fft(
            image_bgr,
            threshold_ratio=0.08,
            min_distance=10,
            filter_radius=4,
        )
        filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(filtered_rgb)

    @staticmethod
    def _apply_unsharp_mask(image: Image.Image) -> Image.Image:
        """Apply a light unsharp mask after periodic FFT postprocessing."""
        image_rgb = image.convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
        gaussian = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(image_bgr, 1.5, gaussian, -0.5, 0)
        sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
        return Image.fromarray(sharpened_rgb)

    def save_result(self, image: Image.Image, input_path: str | Path) -> Path:
        """Save the denoised output image using the required naming convention."""
        source_path = Path(input_path).expanduser().resolve()
        output_dir = self.ensure_output_dir()
        output_path = output_dir / f"{source_path.stem}_denoised.png"
        image.save(output_path)
        return output_path


def run_pipeline(
    input_image_path: str | Path,
    classifier_path: str | Path | None = None,
    output_dir: str | Path = "outputs",
    device: str | None = None,
    periodic_residual_strength: float = 1.0,
) -> PipelineOutput:
    """Compatibility wrapper around `NoiseReductionPipeline.run`."""
    pipeline = NoiseReductionPipeline(
        classifier_path=classifier_path,
        output_dir=output_dir,
        device=device,
        periodic_residual_strength=periodic_residual_strength,
    )
    return pipeline.run(input_image_path)
