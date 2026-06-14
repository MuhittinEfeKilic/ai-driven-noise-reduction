from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return x * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2) -> None:
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.norm1 = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels,
        )
        self.simple_gate = SimpleGate()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1),
        )
        self.pw2 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1)

        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, ffn_channels, kernel_size=1)
        self.ffn_gate = SimpleGate()
        self.ffn2 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        y = self.norm1(x)
        y = self.pw1(y)
        y = self.dwconv(y)
        y = self.simple_gate(y)
        y = y * self.channel_attn(y)
        y = self.pw2(y)
        x = residual + y * self.beta

        y = self.norm2(x)
        y = self.ffn1(y)
        y = self.ffn_gate(y)
        y = self.ffn2(y)
        return x + y * self.gamma


class NAFStage(nn.Module):
    def __init__(self, channels: int, block_count: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[NAFBlock(channels) for _ in range(block_count)])

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class PeriodicFFTGuidedNAFNet(nn.Module):
    """Compact NAFNet variant used by the periodic FFT-guided denoiser.

    The checkpoint in this project was trained with a 6-channel input: noisy RGB
    concatenated with an FFT-filter helper RGB image.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 3,
        width: int = 24,
        encoder_blocks: tuple[int, ...] = (2, 2, 4),
        middle_blocks: int = 6,
        decoder_blocks: tuple[int, ...] = (2, 2, 2),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.encoder_blocks = tuple(encoder_blocks)
        self.middle_blocks = middle_blocks
        self.decoder_blocks = tuple(decoder_blocks)

        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.ending = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)

        channels = width
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for block_count in self.encoder_blocks:
            self.encoders.append(NAFStage(channels, block_count))
            self.downs.append(Downsample(channels))
            channels *= 2

        self.middle = NAFStage(channels, middle_blocks)

        self.ups = nn.ModuleList()
        self.skip_fusions = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for block_count in self.decoder_blocks:
            self.ups.append(Upsample(channels))
            channels //= 2
            self.skip_fusions.append(nn.Conv2d(channels * 2, channels, kernel_size=1))
            self.decoders.append(NAFStage(channels, block_count))

        self.padder_size = 2 ** len(self.encoder_blocks)

    def check_image_size(self, x: Tensor) -> Tensor:
        _, _, height, width = x.shape
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("Expected input tensor with shape [B, C, H, W].")
        _, _, height, width = x.shape
        x = self.check_image_size(x)

        x = self.intro(x)
        skips: list[Tensor] = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for decoder, up, skip_fusion, skip in zip(self.decoders, self.ups, self.skip_fusions, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = skip_fusion(torch.cat([x, skip], dim=1))
            x = decoder(x)

        x = self.ending(x)
        return x[:, :, :height, :width]
