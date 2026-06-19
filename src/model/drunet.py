from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.model.base_model import BaseModel


class ResidualBlock(nn.Module):
    """
    We don't use batch normalization here because it's not used in the paper.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + residual

    @staticmethod
    def stack(in_channels: int, n: int) -> nn.Module:
        return nn.Sequential(*[ResidualBlock(in_channels) for _ in range(n)])


class DRUNet(BaseModel):
    """
    Denoising residual U-Net.
    """

    def __init__(
        self,
        channels: Sequence[int],
        in_channels: int,
        out_channels: int,
        n_blocks: int,
    ):
        super().__init__()
        channels = list(channels)

        n_scales = len(channels)
        self.n_scales = n_scales

        self.factor = 2 ** (n_scales - 1)

        self.head = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        self.tail = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(n_scales - 1):
            self.encoders.append(ResidualBlock.stack(channels[i], n_blocks))
            self.downs.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=2, stride=2)
            )

        self.body = ResidualBlock.stack(channels[-1], n_blocks)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(n_scales - 1, 0, -1):
            self.ups.append(
                nn.ConvTranspose2d(
                    channels[i], channels[i - 1], kernel_size=2, stride=2
                )
            )
            self.decoders.append(ResidualBlock.stack(channels[i - 1], n_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor of shape (B, in_channels, H, W)

        Returns:
            torch.Tensor of shape (B, out_channels, H, W) the reconstructed image.
        """
        h, w = x.shape[-2:]

        pad_h = (self.factor - h % self.factor) % self.factor
        pad_w = (self.factor - w % self.factor) % self.factor

        x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.head(x)

        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.body(x)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = decoder(x + skip)

        x = self.tail(x)

        return x[..., :h, :w]
