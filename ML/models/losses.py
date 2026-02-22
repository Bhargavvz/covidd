"""
Loss Functions for Deformable Image Registration
Includes image similarity metrics and deformation regularization terms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class NCC(nn.Module):
    """Local Normalized Cross-Correlation Loss (3D).

    Computes windowed NCC between fixed and warped images.
    NCC is robust to intensity differences between CT scans
    from different scanners/protocols.

    NCC(I, J) = Σ(I-μI)(J-μJ) / sqrt(Σ(I-μI)² × Σ(J-μJ)²)
    Loss = 1 - NCC  (to minimize)
    """

    def __init__(self, window_size: int = 9, eps: float = 1e-5):
        super().__init__()
        self.window_size = window_size
        self.eps = eps

    def forward(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predicted: (B, 1, D, H, W) warped moving image
            target:    (B, 1, D, H, W) fixed image

        Returns:
            loss: scalar, 1 - mean(NCC)
        """
        # Force float32 to prevent BF16 overflow on large volumes
        I = predicted.float()
        J = target.float()

        # Compute local means using 3D average pooling
        ndims = 3
        win = [self.window_size] * ndims

        # Sum filter
        sum_filt = torch.ones([1, 1] + win, device=I.device, dtype=torch.float32)
        pad_size = self.window_size // 2
        padding = [pad_size] * (2 * ndims)

        # Compute means
        I_padded = F.pad(I, padding, mode="constant", value=0)
        J_padded = F.pad(J, padding, mode="constant", value=0)

        I_sum = F.conv3d(I_padded, sum_filt, padding=0)
        J_sum = F.conv3d(J_padded, sum_filt, padding=0)
        I2_sum = F.conv3d(I_padded ** 2, sum_filt, padding=0)
        J2_sum = F.conv3d(J_padded ** 2, sum_filt, padding=0)
        IJ_sum = F.conv3d(I_padded * J_padded, sum_filt, padding=0)

        win_size = float(self.window_size ** ndims)

        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # Clamp to valid [0, 1] range — prevents explosion in uniform regions
        cc = torch.clamp(cc, 0.0, 1.0)

        return 1.0 - torch.mean(cc)


class MSE(nn.Module):
    """Mean Squared Error loss for image similarity."""

    def forward(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(predicted, target)


class SSIM3D(nn.Module):
    """3D Structural Similarity Index loss.

    Measures structural similarity considering luminance, contrast,
    and structure. Useful for perceptual quality of registration.
    """

    def __init__(
        self,
        window_size: int = 7,
        C1: float = 0.01 ** 2,
        C2: float = 0.03 ** 2,
    ):
        super().__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2

        # Gaussian window
        sigma = window_size / 6.0
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()

        # 3D Gaussian kernel
        kernel_3d = g.view(-1, 1, 1) * g.view(1, -1, 1) * g.view(1, 1, -1)
        kernel_3d = kernel_3d / kernel_3d.sum()
        self.register_buffer(
            "window", kernel_3d.view(1, 1, window_size, window_size, window_size)
        )

    def forward(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        pad = self.window_size // 2

        mu1 = F.conv3d(predicted, self.window, padding=pad)
        mu2 = F.conv3d(target, self.window, padding=pad)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv3d(predicted ** 2, self.window, padding=pad) - mu1_sq
        sigma2_sq = F.conv3d(target ** 2, self.window, padding=pad) - mu2_sq
        sigma12 = F.conv3d(predicted * target, self.window, padding=pad) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )

        return 1.0 - ssim_map.mean()


class BendingEnergy(nn.Module):
    """Bending energy regularization for displacement field smoothness.

    Penalizes the second derivatives (curvature) of the displacement field.
    L_bend = Σ_i (∂²u/∂x²_i)² + 2Σ_{i<j} (∂²u/∂x_i∂x_j)²
    """

    def __init__(self):
        super().__init__()

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flow: (B, 3, D, H, W) displacement field

        Returns:
            loss: scalar bending energy
        """
        # Second derivatives
        # ∂²u/∂d²
        d2_dd = flow[:, :, 2:, :, :] + flow[:, :, :-2, :, :] - 2 * flow[:, :, 1:-1, :, :]
        # ∂²u/∂h²
        d2_dh = flow[:, :, :, 2:, :] + flow[:, :, :, :-2, :] - 2 * flow[:, :, :, 1:-1, :]
        # ∂²u/∂w²
        d2_dw = flow[:, :, :, :, 2:] + flow[:, :, :, :, :-2] - 2 * flow[:, :, :, :, 1:-1]

        # Mixed second derivatives
        # ∂²u/∂d∂h
        d2_ddh = (
            flow[:, :, 2:, 2:, :] - flow[:, :, 2:, :-2, :]
            - flow[:, :, :-2, 2:, :] + flow[:, :, :-2, :-2, :]
        ) / 4.0
        # ∂²u/∂d∂w
        d2_ddw = (
            flow[:, :, 2:, :, 2:] - flow[:, :, 2:, :, :-2]
            - flow[:, :, :-2, :, 2:] + flow[:, :, :-2, :, :-2]
        ) / 4.0
        # ∂²u/∂h∂w
        d2_dhw = (
            flow[:, :, :, 2:, 2:] - flow[:, :, :, 2:, :-2]
            - flow[:, :, :, :-2, 2:] + flow[:, :, :, :-2, :-2]
        ) / 4.0

        loss = (
            d2_dd.pow(2).mean()
            + d2_dh.pow(2).mean()
            + d2_dw.pow(2).mean()
            + 2 * d2_ddh.pow(2).mean()
            + 2 * d2_ddw.pow(2).mean()
            + 2 * d2_dhw.pow(2).mean()
        )
        return loss


class DiffusionRegularization(nn.Module):
    """Diffusion regularization: penalizes first-order gradients.

    L_diff = Σ ||∇u||²
    Simpler but less smooth than bending energy.
    """

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        d_dd = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
        d_dh = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
        d_dw = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]

        loss = d_dd.pow(2).mean() + d_dh.pow(2).mean() + d_dw.pow(2).mean()
        return loss


class JacobianDeterminantLoss(nn.Module):
    """Penalizes non-positive Jacobian determinants (topology folding).

    A Jacobian determinant ≤ 0 means the deformation is locally
    non-invertible (folding). This loss encourages det(J) > 0 everywhere.

    L_jac = mean(max(0, -det(J)))
    """

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        from models.spatial_transformer import compute_jacobian_determinant

        jac_det = compute_jacobian_determinant(flow)

        # Penalize negative Jacobian determinants
        neg_jac = F.relu(-jac_det)
        loss = neg_jac.mean()

        return loss


class DiceLoss(nn.Module):
    """Dice loss for segmentation alignment (auxiliary loss).

    Used when segmentation masks are available to guide registration.
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        predicted_seg: torch.Tensor,
        target_seg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted_seg: (B, C, D, H, W) warped segmentation
            target_seg:    (B, C, D, H, W) fixed segmentation

        Returns:
            loss: 1 - mean Dice coefficient
        """
        pflat = predicted_seg.contiguous().view(-1)
        tflat = target_seg.contiguous().view(-1)

        intersection = (pflat * tflat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pflat.sum() + tflat.sum() + self.smooth
        )
        return 1.0 - dice


class RegistrationLoss(nn.Module):
    """Combined registration loss with image similarity + regularization.

    Total Loss = λ_sim × L_sim + λ_smooth × L_smooth + λ_jac × L_jac [+ λ_seg × L_seg]
    """

    def __init__(self, config: dict):
        super().__init__()

        # Image similarity
        sim_type = config.get("similarity", "ncc")
        if sim_type == "ncc":
            self.sim_loss = NCC(window_size=config.get("ncc_window", 9))
        elif sim_type == "mse":
            self.sim_loss = MSE()
        elif sim_type == "ssim":
            self.sim_loss = SSIM3D()
        else:
            raise ValueError(f"Unknown similarity: {sim_type}")

        # Smoothness regularization
        smooth_type = config.get("smooth_type", "bending")
        if smooth_type == "bending":
            self.smooth_loss = BendingEnergy()
        elif smooth_type == "diffusion":
            self.smooth_loss = DiffusionRegularization()
        else:
            raise ValueError(f"Unknown smooth_type: {smooth_type}")

        # Jacobian loss
        self.jac_loss = JacobianDeterminantLoss()

        # Dice loss (optional, for segmentation)
        self.dice_loss = DiceLoss()

        # Weights
        self.sim_weight = config.get("ncc_weight", 1.0)
        self.smooth_weight = config.get("smooth_weight", 3.0)
        self.jac_weight = config.get("jac_weight", 0.1)
        self.seg_weight = config.get("seg_weight", 0.5)

    def forward(
        self,
        warped: torch.Tensor,
        fixed: torch.Tensor,
        flow: torch.Tensor,
        warped_seg: Optional[torch.Tensor] = None,
        fixed_seg: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute combined loss.

        Returns dict with total_loss and individual components.
        """
        losses = {}

        # Similarity loss
        losses["sim"] = self.sim_loss(warped, fixed)

        # Smoothness loss
        losses["smooth"] = self.smooth_loss(flow)

        # Jacobian loss
        losses["jac"] = self.jac_loss(flow)

        # Total loss
        losses["total"] = (
            self.sim_weight * losses["sim"]
            + self.smooth_weight * losses["smooth"]
            + self.jac_weight * losses["jac"]
        )

        # Optional segmentation loss
        if warped_seg is not None and fixed_seg is not None:
            losses["dice"] = self.dice_loss(warped_seg, fixed_seg)
            losses["total"] = losses["total"] + self.seg_weight * losses["dice"]

        return losses
