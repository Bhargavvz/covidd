"""
VoxelMorph: Deep Learning for Deformable Image Registration
Implements both standard and diffeomorphic variants.

References:
    - Balakrishnan et al., "VoxelMorph: A Learning Framework for 
      Deformable Medical Image Registration", IEEE TMI, 2019
    - Dalca et al., "Unsupervised Learning for Probabilistic 
      Diffeomorphic Registration for Images and Surfaces", MedIA, 2019
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from models.unet3d import UNet3D
from models.spatial_transformer import SpatialTransformer, VecInt, ResizeTransform


class VoxelMorph(nn.Module):
    """Standard VoxelMorph: U-Net → Displacement Field → Spatial Transformer.

    Predicts a dense displacement field that warps the moving image
    to align with the fixed image.

    Input:
        moving: (B, 1, D, H, W) - moving/source volume
        fixed:  (B, 1, D, H, W) - fixed/target volume

    Output:
        warped:       (B, 1, D, H, W) - warped moving volume
        displacement: (B, 3, D, H, W) - predicted displacement field
    """

    def __init__(
        self,
        vol_size: Tuple[int, int, int] = (192, 192, 192),
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

        self.vol_size = vol_size

        # U-Net backbone: takes concatenated (moving, fixed) as input
        self.unet = UNet3D(
            in_channels=2,  # concatenated pair
            out_channels=3,  # displacement field (dx, dy, dz)
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            use_instance_norm=use_instance_norm,
            leaky_slope=leaky_slope,
            dropout=dropout,
        )

        # Spatial transformer for warping
        self.spatial_transformer = SpatialTransformer(vol_size)

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            moving: (B, 1, D, H, W) moving volume
            fixed:  (B, 1, D, H, W) fixed volume

        Returns:
            warped: (B, 1, D, H, W) warped moving volume
            flow:   (B, 3, D, H, W) displacement field
        """
        # Concatenate moving and fixed along channel dimension
        x = torch.cat([moving, fixed], dim=1)  # (B, 2, D, H, W)

        # Predict displacement field
        flow = self.unet(x)  # (B, 3, D, H, W)

        # Warp moving image with displacement field
        warped = self.spatial_transformer(moving, flow)

        return warped, flow


class VoxelMorphDiff(nn.Module):
    """Diffeomorphic VoxelMorph: U-Net → Velocity Field → Integration → STN.

    Instead of directly predicting a displacement field, predicts a
    stationary velocity field that is integrated via scaling-and-squaring
    to produce a diffeomorphic (topology-preserving) deformation.

    This guarantees:
        - Smoothness of the transformation
        - Invertibility (the inverse transform exists)
        - No folding (Jacobian determinant stays positive)
    """

    def __init__(
        self,
        vol_size: Tuple[int, int, int] = (192, 192, 192),
        enc_channels: List[int] = None,
        dec_channels: List[int] = None,
        int_steps: int = 7,
        int_downsize: int = 2,
        use_instance_norm: bool = True,
        leaky_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()

        if enc_channels is None:
            enc_channels = [16, 32, 64, 128, 256]
        if dec_channels is None:
            dec_channels = list(reversed(enc_channels))

        self.vol_size = vol_size
        self.int_steps = int_steps
        self.int_downsize = int_downsize

        # U-Net backbone
        self.unet = UNet3D(
            in_channels=2,
            out_channels=3,  # velocity field
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            use_instance_norm=use_instance_norm,
            leaky_slope=leaky_slope,
            dropout=dropout,
        )

        # Velocity field integration
        # Optionally compute at lower resolution for efficiency
        if int_downsize > 1:
            int_size = tuple(s // int_downsize for s in vol_size)
            self.resize = ResizeTransform(1.0 / int_downsize)
            self.fullsize = ResizeTransform(float(int_downsize))
        else:
            int_size = vol_size
            self.resize = None
            self.fullsize = None

        self.integrate = VecInt(int_size, int_steps)

        # Spatial transformer
        self.spatial_transformer = SpatialTransformer(vol_size)

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            moving: (B, 1, D, H, W) moving volume
            fixed:  (B, 1, D, H, W) fixed volume

        Returns:
            warped: (B, 1, D, H, W) warped moving volume
            flow:   (B, 3, D, H, W) displacement field (integrated)
        """
        # Concatenate and predict velocity field
        x = torch.cat([moving, fixed], dim=1)
        velocity = self.unet(x)  # (B, 3, D, H, W)

        # Optionally resize for faster integration
        if self.resize is not None:
            velocity_small = self.resize(velocity)
        else:
            velocity_small = velocity

        # Integrate velocity to displacement (scaling-and-squaring)
        flow_small = self.integrate(velocity_small)

        # Resize back to full resolution
        if self.fullsize is not None:
            flow = self.fullsize(flow_small)
        else:
            flow = flow_small

        # Warp moving image
        warped = self.spatial_transformer(moving, flow)

        return warped, flow


def build_model(config: dict) -> nn.Module:
    """Factory function to build registration model from config.

    Args:
        config: Model configuration dict

    Returns:
        Registration model (VoxelMorph or VoxelMorphDiff)
    """
    model_type = config.get("type", "voxelmorph_diff")
    vol_size = tuple(config.get("vol_size", [192, 192, 192]))
    enc_channels = config.get("enc_channels", [16, 32, 64, 128, 256])
    dec_channels = config.get("dec_channels", list(reversed(enc_channels)))
    use_instance_norm = config.get("use_instance_norm", True)
    leaky_slope = config.get("leaky_relu_slope", 0.2)
    dropout = config.get("dropout", 0.0)

    if model_type == "voxelmorph":
        model = VoxelMorph(
            vol_size=vol_size,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            use_instance_norm=use_instance_norm,
            leaky_slope=leaky_slope,
            dropout=dropout,
        )
    elif model_type == "voxelmorph_diff":
        int_steps = config.get("int_steps", 7)
        int_downsize = config.get("int_downsize", 2)
        model = VoxelMorphDiff(
            vol_size=vol_size,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            int_steps=int_steps,
            int_downsize=int_downsize,
            use_instance_norm=use_instance_norm,
            leaky_slope=leaky_slope,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Log model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Built {model_type} model with {num_params:,} trainable parameters")

    return model
