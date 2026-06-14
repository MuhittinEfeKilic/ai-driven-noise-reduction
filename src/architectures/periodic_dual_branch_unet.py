from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DoubleConv(nn.Module):
    """Apply two convolution-BN-ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downsample with max pooling followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upsample, concatenate skip features, and fuse with DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class OutConv(nn.Module):
    """Project decoder features to the residual output channels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class FuseFrequencyFeatures(nn.Module):
    """Fuse spatial and frequency features with concat + 1x1 projection."""

    def __init__(self, spatial_channels: int, frequency_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(spatial_channels + frequency_channels, spatial_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(spatial_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, spatial_features: Tensor, frequency_features: Tensor) -> Tensor:
        if spatial_features.shape[-2:] != frequency_features.shape[-2:]:
            frequency_features = F.interpolate(
                frequency_features,
                size=spatial_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        fused = torch.cat([spatial_features, frequency_features], dim=1)
        return self.proj(fused)


class PeriodicDualBranchResidualUNet(nn.Module):
    """Dual-branch residual U-Net for periodic noise suppression.

    The model consumes an RGB image branch and an FFT magnitude branch. Frequency
    features are injected into the spatial encoder at multiple stages to make the
    decoder more sensitive to periodic interference patterns. The model returns a
    3-channel residual tensor; reconstruction ``noisy - residual`` must be handled
    outside the model.
    """

    def __init__(
        self,
        rgb_in_channels: int = 3,
        freq_in_channels: int = 1,
        out_channels: int = 3,
        base_features: int = 64,
        frequency_base_features: int = 32,
    ) -> None:
        super().__init__()

        if rgb_in_channels < 1 or freq_in_channels < 1 or out_channels < 1:
            raise ValueError("Input and output channel counts must be positive.")
        if base_features < 1 or frequency_base_features < 1:
            raise ValueError("Feature counts must be positive.")

        self.rgb_in_channels = rgb_in_channels
        self.freq_in_channels = freq_in_channels
        self.out_channels = out_channels
        self.base_features = base_features
        self.frequency_base_features = frequency_base_features

        sf = base_features
        ff = frequency_base_features

        self.spatial_in = DoubleConv(rgb_in_channels, sf)
        self.spatial_down1 = Down(sf, sf * 2)
        self.spatial_down2 = Down(sf * 2, sf * 4)
        self.spatial_down3 = Down(sf * 4, sf * 8)
        self.spatial_down4 = Down(sf * 8, sf * 16)

        self.freq_in = DoubleConv(freq_in_channels, ff)
        self.freq_down1 = Down(ff, ff * 2)
        self.freq_down2 = Down(ff * 2, ff * 4)
        self.freq_down3 = Down(ff * 4, ff * 8)
        self.freq_down4 = Down(ff * 8, ff * 8)

        self.fuse0 = FuseFrequencyFeatures(sf, ff)
        self.fuse1 = FuseFrequencyFeatures(sf * 2, ff * 2)
        self.fuse2 = FuseFrequencyFeatures(sf * 4, ff * 4)
        self.fuse3 = FuseFrequencyFeatures(sf * 8, ff * 8)
        self.fuse4 = FuseFrequencyFeatures(sf * 16, ff * 8)

        self.up1 = Up(sf * 16, sf * 8, sf * 8)
        self.up2 = Up(sf * 8, sf * 4, sf * 4)
        self.up3 = Up(sf * 4, sf * 2, sf * 2)
        self.up4 = Up(sf * 2, sf, sf)
        self.outc = OutConv(sf, out_channels)

    def forward(self, x_rgb: Tensor, x_freq: Tensor) -> Tensor:
        if x_rgb.ndim != 4 or x_freq.ndim != 4:
            raise ValueError("Expected both inputs to have shape (B, C, H, W).")
        if x_rgb.shape[0] != x_freq.shape[0]:
            raise ValueError("RGB and frequency batches must have the same batch size.")

        s1 = self.spatial_in(x_rgb)
        f1 = self.freq_in(x_freq)
        s1 = self.fuse0(s1, f1)

        s2 = self.spatial_down1(s1)
        f2 = self.freq_down1(f1)
        s2 = self.fuse1(s2, f2)

        s3 = self.spatial_down2(s2)
        f3 = self.freq_down2(f2)
        s3 = self.fuse2(s3, f3)

        s4 = self.spatial_down3(s3)
        f4 = self.freq_down3(f3)
        s4 = self.fuse3(s4, f4)

        s5 = self.spatial_down4(s4)
        f5 = self.freq_down4(f4)
        s5 = self.fuse4(s5, f5)

        x = self.up1(s5, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)
        return self.outc(x)


if __name__ == "__main__":
    batch_size, height, width = 2, 128, 128
    rgb = torch.rand(batch_size, 3, height, width)
    freq = torch.rand(batch_size, 1, height, width)

    model = PeriodicDualBranchResidualUNet()
    residual = model(rgb, freq)

    print(f"RGB input shape      : {tuple(rgb.shape)}")
    print(f"Frequency input shape: {tuple(freq.shape)}")
    print(f"Residual output shape: {tuple(residual.shape)}")

    assert residual.shape == (batch_size, 3, height, width)
    print("Self-test passed.")
