"""
3D U-Net Encoder-Decoder Backbone
Produces dense displacement/velocity fields for deformable registration.
Optimized for volumetric medical images on high-memory GPUs (H200).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ConvBlock3D(nn.Module):
    """3D convolution block: Conv3D → InstanceNorm → LeakyReLU (×2)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_instance_norm: bool = True,
        leaky_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []

        # First conv
        layers.append(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=not use_instance_norm)
        )
        if use_instance_norm:
            layers.append(nn.InstanceNorm3d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(leaky_slope, inplace=True))

        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))

        # Second conv
        layers.append(
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=not use_instance_norm)
        )
        if use_instance_norm:
            layers.append(nn.InstanceNorm3d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(leaky_slope, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder3D(nn.Module):
    """3D U-Net encoder with strided convolution downsampling."""

    def __init__(
        self,
        in_channels: int = 2,  # moving + fixed concatenated
        channels: List[int] = None,
        use_instance_norm: bool = True,
        leaky_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if channels is None:
            channels = [16, 32, 64, 128, 256]

        self.channels = channels
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        # First encoder block (no downsampling)
        self.encoders.append(
            ConvBlock3D(in_channels, channels[0], use_instance_norm=use_instance_norm,
                        leaky_slope=leaky_slope, dropout=dropout)
        )

        # Subsequent encoder blocks with stride-2 downsampling
        for i in range(1, len(channels)):
            self.downsamplers.append(
                nn.Conv3d(channels[i - 1], channels[i - 1], kernel_size=2, stride=2, bias=False)
            )
            self.encoders.append(
                ConvBlock3D(channels[i - 1], channels[i], use_instance_norm=use_instance_norm,
                            leaky_slope=leaky_slope, dropout=dropout)
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            bottleneck: deepest feature map
            skip_connections: list of encoder outputs for skip connections
        """
        skips = []

        # First block
        x = self.encoders[0](x)
        skips.append(x)

        # Remaining blocks
        for i in range(len(self.downsamplers)):
            x = self.downsamplers[i](x)
            x = self.encoders[i + 1](x)
            skips.append(x)

        return x, skips[:-1]  # Last is bottleneck, not a skip


class Decoder3D(nn.Module):
    """3D U-Net decoder with transposed convolution upsampling + skip connections."""

    def __init__(
        self,
        channels: List[int] = None,
        use_instance_norm: bool = True,
        leaky_slope: float = 0.2,
    ):
        super().__init__()
        if channels is None:
            channels = [256, 128, 64, 32, 16]

        self.channels = channels
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(len(channels) - 1):
            # Transposed convolution for upsampling
            self.upsamplers.append(
                nn.ConvTranspose3d(
                    channels[i], channels[i + 1], kernel_size=2, stride=2, bias=False
                )
            )
            # Decoder block (skip connection doubles input channels)
            self.decoders.append(
                ConvBlock3D(
                    channels[i + 1] * 2,  # upsampled + skip
                    channels[i + 1],
                    use_instance_norm=use_instance_norm,
                    leaky_slope=leaky_slope,
                )
            )

    def forward(
        self, x: torch.Tensor, skip_connections: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: bottleneck features
            skip_connections: encoder skip outputs (reversed order)
        """
        for i in range(len(self.upsamplers)):
            x = self.upsamplers[i](x)

            # Handle size mismatch from odd input dimensions
            skip = skip_connections[-(i + 1)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)

        return x


class UNet3D(nn.Module):
    """Full 3D U-Net for displacement/velocity field prediction.

    Input:  (B, 2, D, H, W) - concatenated moving + fixed volumes
    Output: (B, 3, D, H, W) - predicted displacement field (dx, dy, dz)
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 3,
        enc_channels: List[int] = None,
        dec_channels: List[int] = None,
        use_instance_norm: bool = True,
        leaky_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()

        if enc_channels is None:
            enc_channels = [16, 32, 64, 128, 256]
        if dec_channels is None:
            dec_channels = list(reversed(enc_channels))

        self.encoder = Encoder3D(
            in_channels=in_channels,
            channels=enc_channels,
            use_instance_norm=use_instance_norm,
            leaky_slope=leaky_slope,
            dropout=dropout,
        )

        self.decoder = Decoder3D(
            channels=dec_channels,
            use_instance_norm=use_instance_norm,
            leaky_slope=leaky_slope,
        )

        # Final 1x1 conv to produce displacement field
        final_channels = dec_channels[-1]
        self.flow_head = nn.Sequential(
            nn.Conv3d(final_channels, final_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Conv3d(final_channels // 2, out_channels, kernel_size=1),
        )

        # Initialize flow head with small weights for near-identity initialization
        self._init_flow_head()

    def _init_flow_head(self):
        """Initialize flow head to produce near-zero displacement."""
        for m in self.flow_head.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, mean=0, std=1e-5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, D, H, W) concatenated moving + fixed

        Returns:
            flow: (B, 3, D, H, W) displacement field
        """
        bottleneck, skips = self.encoder(x)
        decoded = self.decoder(bottleneck, skips)
        flow = self.flow_head(decoded)
        return flow

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
