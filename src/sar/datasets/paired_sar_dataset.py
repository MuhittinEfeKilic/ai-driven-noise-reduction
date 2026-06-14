from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SUPPORTED_ARRAY_EXTENSIONS = {".npy"}


class PairedSARDataset(Dataset[dict[str, Tensor | str]]):
    """Dataset for paired SAR denoising.

    Expected split directory layout:

    ```text
    split_dir/
      noisy/
      target/
      incidence/          # optional image channel
      incidence_scalar/   # optional .npy scalar files
      column_profile/     # optional image channel
      soft_swath/         # optional image channel
    ```
    """

    def __init__(
        self,
        split_dir: str | Path,
        use_incidence: bool = False,
        use_incidence_scalar: bool = False,
        use_column_profile: bool = False,
        use_soft_swath: bool = False,
        augment: bool = False,
    ) -> None:
        if use_incidence and use_incidence_scalar:
            raise ValueError("use_incidence and use_incidence_scalar cannot both be True.")

        self.split_dir = Path(split_dir)
        self.use_incidence = use_incidence
        self.use_incidence_scalar = use_incidence_scalar
        self.use_column_profile = use_column_profile
        self.use_soft_swath = use_soft_swath
        self.augment = augment

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Paired SAR split directory not found: {self.split_dir}")

        self.noisy_dir = self._require_dir("noisy")
        self.target_dir = self._require_dir("target")
        self.incidence_dir = self.split_dir / "incidence"
        self.incidence_scalar_dir = self.split_dir / "incidence_scalar"
        self.column_profile_dir = self.split_dir / "column_profile"
        self.soft_swath_dir = self.split_dir / "soft_swath"

        self.samples = self._collect_samples()
        if not self.samples:
            raise FileNotFoundError(f"No valid paired SAR samples found in: {self.split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        noisy = self._load_single_channel(sample["noisy"])
        target = self._load_single_channel(sample["target"])

        channels = [noisy]
        if self.use_incidence:
            channels.append(self._load_single_channel(sample["incidence"]))
        if self.use_incidence_scalar:
            scalar = self._load_scalar(sample["incidence_scalar"])
            channels.append(np.full_like(noisy, scalar, dtype=np.float32))
        if self.use_column_profile:
            channels.append(self._load_single_channel(sample["column_profile"]))
        if self.use_soft_swath:
            channels.append(self._load_single_channel(sample["soft_swath"]))

        input_tensor = torch.from_numpy(np.stack(channels, axis=0).astype(np.float32, copy=False))
        target_tensor = torch.from_numpy(target[None, ...].astype(np.float32, copy=False))
        if self.augment:
            input_tensor, target_tensor = self._augment_pair(input_tensor, target_tensor)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "name": sample["name"],
        }

    def _require_dir(self, name: str) -> Path:
        path = self.split_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Expected SAR data directory not found: {path}")
        return path

    def _collect_samples(self) -> list[dict[str, Path | str]]:
        samples: list[dict[str, Path | str]] = []
        for noisy_path in sorted(self.noisy_dir.iterdir()):
            if not noisy_path.is_file() or noisy_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            target_path = self._matching_path(self.target_dir, noisy_path.stem)
            if target_path is None:
                continue

            sample: dict[str, Path | str] = {
                "name": noisy_path.stem,
                "noisy": noisy_path,
                "target": target_path,
            }
            self._attach_optional(sample, "incidence", self.use_incidence, self.incidence_dir, SUPPORTED_IMAGE_EXTENSIONS)
            self._attach_optional(
                sample,
                "incidence_scalar",
                self.use_incidence_scalar,
                self.incidence_scalar_dir,
                SUPPORTED_ARRAY_EXTENSIONS,
            )
            self._attach_optional(
                sample,
                "column_profile",
                self.use_column_profile,
                self.column_profile_dir,
                SUPPORTED_IMAGE_EXTENSIONS,
            )
            self._attach_optional(
                sample,
                "soft_swath",
                self.use_soft_swath,
                self.soft_swath_dir,
                SUPPORTED_IMAGE_EXTENSIONS,
            )
            samples.append(sample)
        return samples

    def _attach_optional(
        self,
        sample: dict[str, Path | str],
        key: str,
        enabled: bool,
        directory: Path,
        extensions: set[str],
    ) -> None:
        if not enabled:
            return
        path = self._matching_path(directory, str(sample["name"]), extensions=extensions)
        if path is None:
            raise FileNotFoundError(f"Missing {key} channel for sample: {sample['name']}")
        sample[key] = path

    @staticmethod
    def _matching_path(directory: Path, stem: str, extensions: set[str] = SUPPORTED_IMAGE_EXTENSIONS) -> Path | None:
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
        if float(np.nanmax(array)) > 1.0:
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
