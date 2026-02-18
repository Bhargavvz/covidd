"""
Differentiable 3D Spatial Transformer Network
Warps volumes using displacement or velocity fields with trilinear interpolation.
Supports diffeomorphic transforms via scaling-and-squaring integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpatialTransformer(nn.Module):
    """3D Spatial Transformer Network.

    Takes a displacement field and applies it to warp a source volume.
    Uses PyTorch's grid_sample for differentiable trilinear interpolation.
    """

    def __init__(self, size: tuple, mode: str = "bilinear"):
        """
        Args:
            size: (D, H, W) spatial dimensions
            mode: interpolation mode ('bilinear' for trilinear in 3D, 'nearest')
        """
        super().__init__()
        self.mode = mode

        # Create identity sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)  # (3, D, H, W)
        grid = grid.float()

        # Register as buffer (not a parameter, but moves to device)
        self.register_buffer("grid", grid, persistent=False)

    def forward(
        self,
        src: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """Warp source image with displacement field.

        Args:
            src:  (B, C, D, H, W) source/moving volume
            flow: (B, 3, D, H, W) displacement field

        Returns:
            warped: (B, C, D, H, W) warped volume
        """
        # new_locs = grid + flow
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Normalize grid to [-1, 1] for grid_sample
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # grid_sample expects (B, D, H, W, 3) with order (x, y, z) = (W, H, D)
        # Our flow is in (D, H, W) order, so we need to flip
        new_locs = new_locs.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3)
        new_locs = new_locs[..., [2, 1, 0]]  # Flip to (W, H, D) for grid_sample

        warped = F.grid_sample(
            src,
            new_locs,
            align_corners=True,
            mode=self.mode,
            padding_mode="border",
        )
        return warped


class VecInt(nn.Module):
    """Vector field integration via scaling and squaring.

    Converts a stationary velocity field to a diffeomorphic displacement
    field using the scaling-and-squaring method:
        φ = exp(v) ≈ (Id + v/2^n)^(2^n)

    This guarantees the resulting transformation is diffeomorphic
    (smooth, invertible, topology-preserving).
    """

    def __init__(self, size: tuple, nsteps: int = 7):
        """
        Args:
            size: (D, H, W) spatial dimensions
            nsteps: number of scaling-and-squaring steps (7 = 128 compositions)
        """
        super().__init__()
        assert nsteps >= 0, "nsteps must be non-negative"
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(size)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        """Integrate velocity field to displacement field.

        Args:
            vec: (B, 3, D, H, W) velocity field

        Returns:
            disp: (B, 3, D, H, W) displacement field
        """
        vec = vec * self.scale

        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)

        return vec


class ResizeTransform(nn.Module):
    """Resize a displacement/velocity field by a factor.

    Used to compute flow at a lower resolution and upsample.
    """

    def __init__(self, vel_resize: float, ndims: int = 3):
        super().__init__()
        self.factor = vel_resize
        self.mode = "trilinear" if ndims == 3 else "bilinear"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.factor < 1:
            # Shrink
            x = F.interpolate(
                x, align_corners=True, scale_factor=self.factor, mode=self.mode
            )
            x = self.factor * x
        elif self.factor > 1:
            # Expand
            x = F.interpolate(
                x, align_corners=True, scale_factor=self.factor, mode=self.mode
            )
            x = self.factor * x
        return x


def compute_jacobian_determinant(
    displacement: torch.Tensor,
) -> torch.Tensor:
    """Compute the Jacobian determinant of a displacement field.

    J(x) = det(I + ∇u(x))

    where u is the displacement field and I is the identity.

    Args:
        displacement: (B, 3, D, H, W) displacement field

    Returns:
        jacobian_det: (B, 1, D, H, W) Jacobian determinant at each voxel
    """
    # Compute spatial gradients of displacement field
    # Using central differences
    dy = (displacement[:, :, 2:, :, :] - displacement[:, :, :-2, :, :]) / 2.0
    dx = (displacement[:, :, :, 2:, :] - displacement[:, :, :, :-2, :]) / 2.0
    dz = (displacement[:, :, :, :, 2:] - displacement[:, :, :, :, :-2]) / 2.0

    # Crop to matching size
    d = min(dy.shape[2], dx.shape[2], dz.shape[2])
    h = min(dy.shape[3], dx.shape[3], dz.shape[3])
    w = min(dy.shape[4], dx.shape[4], dz.shape[4])

    dy = dy[:, :, :d, :h, :w]
    dx = dx[:, :, :d, :h, :w]
    dz = dz[:, :, :d, :h, :w]

    # Jacobian matrix: J_ij = δ_ij + ∂u_i/∂x_j
    # J = [[1+du0/dx, du0/dy, du0/dz],
    #      [du1/dx, 1+du1/dy, du1/dz],
    #      [du2/dx, du2/dy, 1+du2/dz]]

    J00 = 1 + dx[:, 0:1]
    J01 = dy[:, 0:1]
    J02 = dz[:, 0:1]
    J10 = dx[:, 1:2]
    J11 = 1 + dy[:, 1:2]
    J12 = dz[:, 1:2]
    J20 = dx[:, 2:3]
    J21 = dy[:, 2:3]
    J22 = 1 + dz[:, 2:3]

    # Determinant using Sarrus' rule
    det = (
        J00 * (J11 * J22 - J12 * J21)
        - J01 * (J10 * J22 - J12 * J20)
        + J02 * (J10 * J21 - J11 * J20)
    )

    return det
