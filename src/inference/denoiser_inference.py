from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import Module
from torchvision import transforms

from src.architectures.dncnn import DnCNN
from src.architectures.frequency_utils import rgb_to_normalized_fft_log_magnitude
from src.architectures.periodic_dual_branch_unet import PeriodicDualBranchResidualUNet
from src.architectures.unet import UNet
from src.models.nafnet_periodic import PeriodicFFTGuidedNAFNet
from src.preprocessing.periodic_fft_filter import remove_periodic_noise_fft


class DenoiserInference:
    """Inference helper for loading and running image denoisers."""

    SALT_PEPPER_MEDIAN_KERNEL_SIZE = 3

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        residual_strength: float = 1.0,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Module | None = None
        self.model_name: str | None = None
        self.training_domain: str = "linear"
        self.prediction_mode: str = "direct"
        self.residual_strength = float(residual_strength)
        self.transform = transforms.ToTensor()
        self.in_channels: int | None = None
        self.out_channels: int | None = None

    def load_model(self) -> Module:
        """Load a supported denoiser checkpoint and prepare the model for inference."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = self._extract_state_dict(checkpoint)
        model_name, architecture, training_domain, prediction_mode = self._resolve_architecture(checkpoint, state_dict)

        model = self._build_model(model_name, architecture).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()

        self.model = model
        self.model_name = model_name
        self.training_domain = training_domain
        self.prediction_mode = prediction_mode
        self.in_channels = int(architecture["in_channels"])
        self.out_channels = int(architecture["out_channels"])
        return model

    def preprocess_image(self, image: str | Path | Image.Image) -> Tensor:
        """Load an image and convert it to a `[1, C, H, W]` tensor."""
        expected_channels = self._required_in_channels()
        if self.model_name == "PeriodicFFTGuidedNAFNet":
            return self._preprocess_periodic_nafnet_image(image)

        pil_image = self._load_pil_image(image, channels=expected_channels)
        if self._should_apply_salt_pepper_prefilter():
            # Salt & Pepper icin median pre-filter + UNet stratejisi kullaniliyor.
            pil_image = self._apply_salt_pepper_prefilter(pil_image, kernel_size=3)
        tensor = self.transform(pil_image)
        if tensor.ndim != 3 or tensor.shape[0] != expected_channels:
            raise ValueError(f"Expected image preprocessing to produce a [{expected_channels}, H, W] tensor.")
        return tensor.unsqueeze(0).to(self.device)

    def run(self, image: str | Path | Image.Image) -> Image.Image:
        """Run denoising inference and return the denoised PIL image."""
        if self.model is None:
            self.load_model()

        assert self.model is not None
        input_tensor = self.preprocess_image(image)
        with torch.no_grad():
            output_tensor = self._run_model(input_tensor)
        return self.postprocess_output(output_tensor)

    def postprocess_output(self, output: Tensor) -> Image.Image:
        """Convert a model output tensor shaped `[1, C, H, W]` into a PIL image."""
        expected_channels = self._required_out_channels()
        if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] != expected_channels:
            raise ValueError(f"Expected output tensor with shape [1, {expected_channels}, H, W].")

        image_tensor = output.detach().cpu().squeeze(0)
        return transforms.ToPILImage()(image_tensor)

    def save_output(self, image: Image.Image, output_path: str | Path) -> Path:
        """Save a denoised PIL image to disk and return the resolved output path."""
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)
        return destination

    def _required_in_channels(self) -> int:
        if self.in_channels is None:
            raise RuntimeError("Model must be loaded before preprocessing images.")
        return self.in_channels

    def _required_out_channels(self) -> int:
        if self.out_channels is None:
            raise RuntimeError("Model must be loaded before postprocessing outputs.")
        return self.out_channels

    def _should_apply_salt_pepper_prefilter(self) -> bool:
        return "salt_pepper" in str(self.model_path).lower()

    @staticmethod
    def _apply_salt_pepper_prefilter(image: Image.Image, kernel_size: int = SALT_PEPPER_MEDIAN_KERNEL_SIZE) -> Image.Image:
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("Median filter kernel size must be a positive odd integer.")

        image_array = np.array(image)
        filtered = cv2.medianBlur(image_array, kernel_size)
        return Image.fromarray(filtered)

    def _run_model(self, input_tensor: Tensor) -> Tensor:
        if self.model is None:
            raise RuntimeError("Model must be loaded before inference.")
        if isinstance(self.model, DnCNN):
            return self.model.denoise(input_tensor)
        if isinstance(self.model, PeriodicDualBranchResidualUNet):
            x_freq = rgb_to_normalized_fft_log_magnitude(input_tensor)
            predicted_residual = self.model(input_tensor, x_freq)
            return torch.clamp(input_tensor - self.residual_strength * predicted_residual, min=0.0, max=1.0)
        if isinstance(self.model, PeriodicFFTGuidedNAFNet):
            noisy_rgb = input_tensor[:, :3]
            predicted_residual = self.model(input_tensor)
            if self.prediction_mode == "residual":
                return torch.clamp(noisy_rgb - self.residual_strength * predicted_residual, min=0.0, max=1.0)
            return torch.clamp(predicted_residual, min=0.0, max=1.0)
        if isinstance(self.model, UNet):
            # Residual checkpoints predict noise/residual, so we reconstruct the clean image here.
            if self.prediction_mode == "residual":
                predicted_residual = self.model(input_tensor)
                return torch.clamp(input_tensor - self.residual_strength * predicted_residual, min=0.0, max=1.0)
            if self.training_domain == "log1p":
                prediction_log = self.model(torch.log1p(torch.clamp(input_tensor, min=0.0, max=1.0)))
                return torch.clamp(torch.expm1(prediction_log), min=0.0, max=1.0)
            return torch.clamp(self.model(input_tensor), min=0.0, max=1.0)
        raise ValueError(f"Unsupported model type for inference: {type(self.model).__name__}")

    @staticmethod
    def _build_model(model_name: str, architecture: dict[str, int | bool]) -> Module:
        if model_name == "PeriodicFFTGuidedNAFNet":
            return PeriodicFFTGuidedNAFNet(
                in_channels=int(architecture.get("in_channels", 6)),
                out_channels=int(architecture.get("out_channels", 3)),
                width=int(architecture.get("width", 24)),
            )
        if model_name == "PeriodicDualBranchResidualUNet":
            return PeriodicDualBranchResidualUNet(
                rgb_in_channels=int(architecture["rgb_in_channels"]),
                freq_in_channels=int(architecture["freq_in_channels"]),
                out_channels=int(architecture["out_channels"]),
                base_features=int(architecture["base_features"]),
                frequency_base_features=int(architecture["frequency_base_features"]),
            )
        if model_name == "UNet":
            # Salt & Pepper icin DnCNN yerine UNet tercih edildigi icin UNet checkpointleri destekleniyor.
            return UNet(
                in_channels=int(architecture["in_channels"]),
                out_channels=int(architecture["out_channels"]),
                base_features=int(architecture["base_features"]),
            )
        if model_name == "DnCNN":
            return DnCNN(**architecture)
        raise ValueError(f"Unsupported denoiser model name: {model_name}")

    @staticmethod
    def _extract_state_dict(checkpoint: object) -> dict[str, Tensor]:
        """Extract a state dict from a checkpoint object."""
        if isinstance(checkpoint, DnCNN):
            return checkpoint.state_dict()
        if isinstance(checkpoint, PeriodicDualBranchResidualUNet):
            return checkpoint.state_dict()
        if isinstance(checkpoint, PeriodicFFTGuidedNAFNet):
            return checkpoint.state_dict()
        if isinstance(checkpoint, UNet):
            return checkpoint.state_dict()
        if isinstance(checkpoint, Module):
            return checkpoint.state_dict()

        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint does not contain a valid state dictionary.")

        normalized_state_dict = {str(key): value for key, value in state_dict.items()}
        if not normalized_state_dict:
            raise ValueError("State dictionary is empty.")
        return normalized_state_dict

    def _resolve_architecture(
        self,
        checkpoint: object,
        state_dict: dict[str, Tensor],
    ) -> tuple[str, dict[str, int | bool], str, str]:
        if isinstance(checkpoint, dict):
            architecture = checkpoint.get("architecture")
            if isinstance(architecture, dict):
                model_name = str(architecture.get("model_name", "")).strip()
                if not model_name:
                    # Newer UNet checkpoints may omit model_name but still expose U-Net specific fields.
                    if "frequency_base_features" in architecture or "rgb_in_channels" in architecture:
                        model_name = "PeriodicDualBranchResidualUNet"
                    else:
                        model_name = "UNet" if "base_features" in architecture else "DnCNN"
                training_domain = str(checkpoint.get("training_domain", "linear"))
                prediction_mode = str(checkpoint.get("prediction_mode", "direct"))
                if model_name == "PeriodicDualBranchResidualUNet":
                    return (
                        model_name,
                        {
                            "rgb_in_channels": int(architecture.get("rgb_in_channels", 3)),
                            "freq_in_channels": int(architecture.get("freq_in_channels", 1)),
                            "out_channels": int(architecture.get("out_channels", 3)),
                            "base_features": int(architecture.get("base_features", 32)),
                            "frequency_base_features": int(architecture.get("frequency_base_features", 16)),
                            "in_channels": int(architecture.get("rgb_in_channels", 3)),
                        },
                        training_domain,
                        prediction_mode,
                    )
                if model_name == "PeriodicFFTGuidedNAFNet":
                    return (
                        model_name,
                        {
                            "in_channels": int(architecture.get("in_channels", 6)),
                            "out_channels": int(architecture.get("out_channels", 3)),
                            "width": int(architecture.get("width", 24)),
                        },
                        training_domain,
                        prediction_mode,
                    )
                if model_name == "UNet":
                    return (
                        model_name,
                        {
                            "in_channels": int(architecture.get("in_channels", 3)),
                            "out_channels": int(architecture.get("out_channels", 3)),
                            "base_features": int(architecture.get("base_features", 64)),
                        },
                        training_domain,
                        prediction_mode,
                    )

        if self._should_use_periodic_dual_branch(state_dict):
            return (
                "PeriodicDualBranchResidualUNet",
                self._infer_periodic_dual_branch_architecture(state_dict),
                "linear",
                "residual",
            )
        if self._should_use_periodic_nafnet(state_dict):
            return (
                "PeriodicFFTGuidedNAFNet",
                self._infer_periodic_nafnet_architecture(state_dict),
                "linear",
                "residual",
            )
        return "DnCNN", DenoiserInference._infer_dncnn_architecture(state_dict), "linear", "direct"

    def _should_use_periodic_dual_branch(self, state_dict: dict[str, Tensor]) -> bool:
        checkpoint_name = self.model_path.name.lower()
        if "periodic" not in checkpoint_name:
            return False
        return any(key.startswith("spatial_in.") for key in state_dict) and any(key.startswith("freq_in.") for key in state_dict)

    def _should_use_periodic_nafnet(self, state_dict: dict[str, Tensor]) -> bool:
        checkpoint_name = self.model_path.name.lower()
        if "periodic" not in checkpoint_name:
            return False
        return "intro.weight" in state_dict and any(key.startswith("encoders.") for key in state_dict)

    @staticmethod
    def _infer_periodic_nafnet_architecture(state_dict: dict[str, Tensor]) -> dict[str, int]:
        intro_weight = state_dict.get("intro.weight")
        ending_weight = state_dict.get("ending.weight")
        if intro_weight is None or ending_weight is None:
            raise ValueError("Unable to infer a valid PeriodicFFTGuidedNAFNet architecture from the checkpoint.")
        return {
            "in_channels": int(intro_weight.shape[1]),
            "out_channels": int(ending_weight.shape[0]),
            "width": int(intro_weight.shape[0]),
        }

    @staticmethod
    def _infer_periodic_dual_branch_architecture(state_dict: dict[str, Tensor]) -> dict[str, int]:
        rgb_weight = state_dict.get("spatial_in.block.0.weight")
        freq_weight = state_dict.get("freq_in.block.0.weight")
        out_weight = state_dict.get("outc.proj.weight")
        if rgb_weight is None or freq_weight is None or out_weight is None:
            raise ValueError("Unable to infer a valid PeriodicDualBranchResidualUNet architecture from the checkpoint.")

        return {
            "rgb_in_channels": int(rgb_weight.shape[1]),
            "freq_in_channels": int(freq_weight.shape[1]),
            "out_channels": int(out_weight.shape[0]),
            "base_features": int(rgb_weight.shape[0]),
            "frequency_base_features": int(freq_weight.shape[0]),
            "in_channels": int(rgb_weight.shape[1]),
        }

    @staticmethod
    def _infer_dncnn_architecture(state_dict: dict[str, Tensor]) -> dict[str, int | bool]:
        """Infer DnCNN architecture parameters from a checkpoint state dict."""
        conv_indices = sorted(
            int(key.split(".")[1])
            for key, value in state_dict.items()
            if key.startswith("network.") and key.endswith(".weight") and value.ndim == 4
        )
        if len(conv_indices) < 3:
            raise ValueError("Unable to infer a valid DnCNN architecture from the checkpoint.")

        first_conv_weight = state_dict[f"network.{conv_indices[0]}.weight"]
        final_conv_weight = state_dict[f"network.{conv_indices[-1]}.weight"]
        middle_indices = conv_indices[1:-1]

        return {
            "in_channels": int(first_conv_weight.shape[1]),
            "out_channels": int(final_conv_weight.shape[0]),
            "num_features": int(first_conv_weight.shape[0]),
            "num_layers": len(conv_indices),
            "first_conv_bias": f"network.{conv_indices[0]}.bias" in state_dict,
            "middle_conv_bias": any(f"network.{idx}.bias" in state_dict for idx in middle_indices),
            "final_conv_bias": f"network.{conv_indices[-1]}.bias" in state_dict,
        }

    @staticmethod
    def _load_pil_image(image: str | Path | Image.Image, channels: int) -> Image.Image:
        """Normalize supported image inputs to the channel mode required by the checkpoint."""
        mode = DenoiserInference._pil_mode_from_channels(channels)
        if isinstance(image, Image.Image):
            return image.convert(mode)

        image_path = Path(image).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        try:
            with Image.open(image_path) as pil_image:
                return pil_image.convert(mode)
        except OSError as exc:
            raise ValueError(f"Could not open image: {image_path}") from exc

    @staticmethod
    def _pil_mode_from_channels(channels: int) -> str:
        """Map tensor channel counts to supported PIL modes."""
        if channels == 1:
            return "L"
        if channels == 3:
            return "RGB"
        raise ValueError(f"Unsupported image channel count for PIL conversion: {channels}")

    def _preprocess_periodic_nafnet_image(self, image: str | Path | Image.Image) -> Tensor:
        pil_image = self._load_pil_image(image, channels=3)
        image_rgb = pil_image.convert("RGB")
        noisy_tensor = self.transform(image_rgb)

        noisy_bgr = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
        helper_bgr = remove_periodic_noise_fft(
            noisy_bgr,
            threshold_ratio=0.08,
            min_distance=10,
            filter_radius=4,
        )
        helper_rgb = cv2.cvtColor(helper_bgr, cv2.COLOR_BGR2RGB)
        helper_tensor = self.transform(Image.fromarray(helper_rgb))

        tensor = torch.cat([noisy_tensor, helper_tensor], dim=0)
        if tensor.shape[0] != 6:
            raise ValueError("PeriodicFFTGuidedNAFNet preprocessing must produce a 6-channel tensor.")
        return tensor.unsqueeze(0).to(self.device)
