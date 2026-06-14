from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SUPPORTED_ARRAY_EXTENSIONS = {".npy"}


class NoiseMapSARDataset(Dataset[dict[str, Tensor | str]]):
    """Dataset for SAR noise-map learning.

    Expected split directory layout:

    ```text
    split_dir/
      noisy/ or noisy_npy/
      target/ or target_npy/
      incidence_scalar/    # optional .npy scalar files
    ```
    """

    def __init__(
        self,
        split_dir: str | Path,
        use_incidence_scalar: bool = False,
        augment: bool = False,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.use_incidence_scalar = use_incidence_scalar
        self.augment = augment

        if not self.split_dir.exists():
            raise FileNotFoundError(f"SAR split directory not found: {self.split_dir}")

        self.noisy_dir = self._resolve_data_dir("noisy_npy", "noisy")
        self.target_dir = self._resolve_data_dir("target_npy", "target")
        self.incidence_scalar_dir = self.split_dir / "incidence_scalar"

        self.samples = self._collect_samples()
        if not self.samples:
            raise FileNotFoundError(f"No valid SAR noise-map pairs found in: {self.split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        noisy = self._load_single_channel(sample["noisy"])
        target = self._load_single_channel(sample["target"])

        channels = [noisy]
        if self.use_incidence_scalar:
            scalar = self._load_scalar(sample["incidence_scalar"])
            channels.append(np.full_like(noisy, scalar, dtype=np.float32))

        input_array = np.stack(channels, axis=0)
        target_array = target[None, ...]

        input_tensor = torch.from_numpy(input_array.astype(np.float32, copy=False))
        target_tensor = torch.from_numpy(target_array.astype(np.float32, copy=False))
        if self.augment:
            input_tensor, target_tensor = self._augment_pair(input_tensor, target_tensor)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "name": sample["name"],
        }

    def _resolve_data_dir(self, preferred: str, fallback: str) -> Path:
        preferred_path = self.split_dir / preferred
        fallback_path = self.split_dir / fallback
        if preferred_path.exists():
            return preferred_path
        if fallback_path.exists():
            return fallback_path
        raise FileNotFoundError(f"Expected {preferred_path} or {fallback_path} to exist.")

    def _collect_samples(self) -> list[dict[str, Path | str]]:
        samples: list[dict[str, Path | str]] = []
        for noisy_path in sorted(self.noisy_dir.iterdir()):
            if not noisy_path.is_file() or noisy_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_ARRAY_EXTENSIONS:
                continue

            target_path = self._matching_path(self.target_dir, noisy_path.stem)
            if target_path is None:
                continue

            sample: dict[str, Path | str] = {
                "name": noisy_path.stem,
                "noisy": noisy_path,
                "target": target_path,
            }
            if self.use_incidence_scalar:
                scalar_path = self._matching_path(self.incidence_scalar_dir, noisy_path.stem, extensions=SUPPORTED_ARRAY_EXTENSIONS)
                if scalar_path is None:
                    raise FileNotFoundError(f"Missing incidence scalar for sample: {noisy_path.stem}")
                sample["incidence_scalar"] = scalar_path
            samples.append(sample)
        return samples

    @staticmethod
    def _matching_path(
        directory: Path,
        stem: str,
        extensions: set[str] = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_ARRAY_EXTENSIONS,
    ) -> Path | None:
        for extension in sorted(extensions):
            candidate = directory / f"{stem}{extension}"
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _load_single_channel(path: Path | str) -> np.ndarray:
        path = Path(path)
        if path.suffix.lower() == ".npy":
            array = np.load(path)
        else:
            with Image.open(path) as image:
                array = np.array(image.convert("L"))

        array = np.asarray(array, dtype=np.float32)
        if array.ndim > 2:
            array = np.squeeze(array)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D SAR patch at {path}, got shape {array.shape}")

        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value > 1.0:
            array = array / 255.0
        return np.clip(array, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _load_scalar(path: Path | str) -> float:
        value = np.asarray(np.load(Path(path)), dtype=np.float32)
        return float(value.reshape(-1)[0])

    @staticmethod
    def _augment_pair(input_tensor: Tensor, target_tensor: Tensor) -> tuple[Tensor, Tensor]:
        if torch.rand(()) < 0.5:
            input_tensor = torch.flip(input_tensor, dims=(-1,))
            target_tensor = torch.flip(target_tensor, dims=(-1,))
        if torch.rand(()) < 0.5:
            input_tensor = torch.flip(input_tensor, dims=(-2,))
            target_tensor = torch.flip(target_tensor, dims=(-2,))
        return input_tensor, target_tensor
