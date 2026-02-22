"""
Tests for model architecture: forward pass, gradient flow, output shapes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np


class TestUNet3D:
    """Test 3D U-Net backbone."""

    def test_forward_shape(self):
        from models.unet3d import UNet3D

        model = UNet3D(in_channels=2, out_channels=3, enc_channels=[8, 16, 32])
        x = torch.randn(1, 2, 32, 32, 32)
        out = model(x)
        assert out.shape == (1, 3, 32, 32, 32), f"Expected (1,3,32,32,32), got {out.shape}"

    def test_gradient_flow(self):
        from models.unet3d import UNet3D

        model = UNet3D(in_channels=2, out_channels=3, enc_channels=[8, 16, 32])
        x = torch.randn(1, 2, 32, 32, 32, requires_grad=True)
        out = model(x)
        loss = out.mean()
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_parameter_count(self):
        from models.unet3d import UNet3D

        model = UNet3D(enc_channels=[8, 16, 32])
        count = model.count_parameters()
        assert count > 0, "Model should have trainable parameters"
        print(f"UNet3D parameter count: {count:,}")


class TestSpatialTransformer:
    """Test spatial transformer and vector integration."""

    def test_identity_warp(self):
        from models.spatial_transformer import SpatialTransformer

        size = (16, 16, 16)
        stn = SpatialTransformer(size)

        src = torch.randn(1, 1, *size)
        zero_flow = torch.zeros(1, 3, *size)
        warped = stn(src, zero_flow)

        # Identity warp should return input (approximately)
        diff = (warped - src).abs().mean().item()
        assert diff < 1e-4, f"Identity warp error too large: {diff}"

    def test_vec_int(self):
        from models.spatial_transformer import VecInt

        size = (16, 16, 16)
        integrate = VecInt(size, nsteps=7)

        velocity = torch.randn(1, 3, *size) * 0.01
        displacement = integrate(velocity)

        assert displacement.shape == (1, 3, *size)
        assert not torch.isnan(displacement).any()

    def test_jacobian_determinant(self):
        from models.spatial_transformer import compute_jacobian_determinant

        # Zero displacement should give det(J) ≈ 1 everywhere
        flow = torch.zeros(1, 3, 16, 16, 16)
        jac_det = compute_jacobian_determinant(flow)

        assert jac_det.shape[0] == 1
        mean_det = jac_det.mean().item()
        assert abs(mean_det - 1.0) < 0.1, f"Expected det≈1 for zero flow, got {mean_det}"


class TestVoxelMorph:
    """Test VoxelMorph models."""

    def test_voxelmorph_forward(self):
        from models.voxelmorph import VoxelMorph

        vol_size = (32, 32, 32)
        model = VoxelMorph(vol_size=vol_size, enc_channels=[8, 16, 32])

        moving = torch.randn(1, 1, *vol_size)
        fixed = torch.randn(1, 1, *vol_size)

        warped, flow = model(moving, fixed)

        assert warped.shape == (1, 1, *vol_size), f"Warped shape: {warped.shape}"
        assert flow.shape == (1, 3, *vol_size), f"Flow shape: {flow.shape}"

    def test_voxelmorph_diff_forward(self):
        from models.voxelmorph import VoxelMorphDiff

        vol_size = (32, 32, 32)
        model = VoxelMorphDiff(
            vol_size=vol_size,
            enc_channels=[8, 16, 32],
            int_steps=7,
            int_downsize=1,
        )

        moving = torch.randn(1, 1, *vol_size)
        fixed = torch.randn(1, 1, *vol_size)

        warped, flow = model(moving, fixed)

        assert warped.shape == (1, 1, *vol_size)
        assert flow.shape == (1, 3, *vol_size)

    def test_build_model(self):
        from models.voxelmorph import build_model

        config = {
            "type": "voxelmorph_diff",
            "vol_size": [32, 32, 32],
            "enc_channels": [8, 16, 32],
            "int_steps": 7,
            "int_downsize": 1,
        }
        model = build_model(config)
        assert model is not None

    def test_near_identity_initialization(self):
        """Test that flow head initializes to near-zero displacement."""
        from models.voxelmorph import VoxelMorph

        vol_size = (32, 32, 32)
        model = VoxelMorph(vol_size=vol_size, enc_channels=[8, 16, 32])

        moving = torch.randn(1, 1, *vol_size)
        fixed = torch.randn(1, 1, *vol_size)

        warped, flow = model(moving, fixed)
        flow_magnitude = flow.abs().mean().item()

        # Flow should be small initially
        assert flow_magnitude < 1.0, f"Initial flow too large: {flow_magnitude}"


class TestRecoveryAnalyzer:
    """Test recovery analysis module."""

    def test_recovery_score_computation(self):
        from models.recovery_analyzer import RecoveryAnalyzer

        analyzer = RecoveryAnalyzer()

        # Create a displacement with mostly normal Jacobian
        disp = torch.randn(1, 3, 16, 16, 16) * 0.01
        result = analyzer(disp)

        assert "overall" in result
        assert "regional" in result
        assert 0 <= result["overall"]["recovery_score"] <= 1.0

    def test_trajectory_analysis(self):
        from models.recovery_analyzer import RecoveryAnalyzer

        analyzer = RecoveryAnalyzer()

        # Simulated improving scores
        scores = [
            {"recovery_score": 0.3},
            {"recovery_score": 0.5},
            {"recovery_score": 0.7},
            {"recovery_score": 0.85},
        ]

        trajectory = analyzer.analyze_trajectory(scores)
        assert trajectory["trend"] == "improving"
        assert trajectory["slope"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
