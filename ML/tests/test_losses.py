"""
Tests for loss functions: NCC, SSIM, bending energy, Jacobian.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch


class TestNCC:
    """Test Normalized Cross-Correlation loss."""

    def test_identical_images(self):
        from models.losses import NCC

        ncc = NCC(window_size=5)
        img = torch.randn(1, 1, 16, 16, 16)
        loss = ncc(img, img)

        # Identical images should give NCC ≈ 1, loss ≈ 0
        assert loss.item() < 0.1, f"NCC loss for identical images: {loss.item()}"

    def test_different_images(self):
        from models.losses import NCC

        ncc = NCC(window_size=5)
        img1 = torch.randn(1, 1, 16, 16, 16)
        img2 = torch.randn(1, 1, 16, 16, 16)
        loss = ncc(img1, img2)

        # Different images should have higher loss
        assert loss.item() > 0.0

    def test_gradient_flow(self):
        from models.losses import NCC

        ncc = NCC(window_size=5)
        img1 = torch.randn(1, 1, 16, 16, 16, requires_grad=True)
        img2 = torch.randn(1, 1, 16, 16, 16)
        loss = ncc(img1, img2)
        loss.backward()

        assert img1.grad is not None
        assert not torch.isnan(img1.grad).any()


class TestSSIM:
    """Test 3D SSIM loss."""

    def test_identical_images(self):
        from models.losses import SSIM3D

        ssim = SSIM3D(window_size=5)
        img = torch.randn(1, 1, 16, 16, 16)
        loss = ssim(img, img)

        # SSIM of identical images → 1, loss → 0
        assert loss.item() < 0.15, f"SSIM loss for identical: {loss.item()}"


class TestBendingEnergy:
    """Test bending energy regularization."""

    def test_zero_flow(self):
        from models.losses import BendingEnergy

        bend = BendingEnergy()
        flow = torch.zeros(1, 3, 16, 16, 16)
        loss = bend(flow)

        assert loss.item() == 0.0, "Bending energy of zero flow should be 0"

    def test_smooth_vs_noisy(self):
        from models.losses import BendingEnergy

        bend = BendingEnergy()

        # Smooth flow
        smooth = torch.randn(1, 3, 16, 16, 16) * 0.01
        smooth_loss = bend(smooth)

        # Noisy flow
        noisy = torch.randn(1, 3, 16, 16, 16) * 1.0
        noisy_loss = bend(noisy)

        # Noisy flow should have higher bending energy
        assert noisy_loss.item() > smooth_loss.item()


class TestJacobianLoss:
    """Test Jacobian determinant loss."""

    def test_zero_flow(self):
        from models.losses import JacobianDeterminantLoss

        jac_loss = JacobianDeterminantLoss()
        flow = torch.zeros(1, 3, 16, 16, 16)
        loss = jac_loss(flow)

        # Zero flow → det(J) = 1 everywhere → no penalty
        assert loss.item() < 0.01, f"Jac loss for zero flow: {loss.item()}"


class TestRegistrationLoss:
    """Test combined registration loss."""

    def test_combined_loss(self):
        from models.losses import RegistrationLoss

        config = {
            "similarity": "ncc",
            "ncc_window": 5,
            "ncc_weight": 1.0,
            "smooth_type": "bending",
            "smooth_weight": 1.0,
            "jac_weight": 0.1,
        }
        criterion = RegistrationLoss(config)

        warped = torch.randn(1, 1, 16, 16, 16, requires_grad=True)
        fixed = torch.randn(1, 1, 16, 16, 16)
        flow = torch.randn(1, 3, 16, 16, 16, requires_grad=True)

        losses = criterion(warped, fixed, flow)

        assert "total" in losses
        assert "sim" in losses
        assert "smooth" in losses
        assert "jac" in losses
        assert not torch.isnan(losses["total"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
