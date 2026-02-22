"""
Recovery Analyzer: Longitudinal lung recovery scoring from registration results.

Quantifies post-COVID-19 lung recovery by analyzing:
    - Jacobian determinant maps (local volume changes)
    - Lung density changes over time
    - Regional recovery scoring (upper/middle/lower lobes)
    - Recovery trajectory classification
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from models.spatial_transformer import compute_jacobian_determinant


class RecoveryAnalyzer(nn.Module):
    """Analyzes registration outputs for longitudinal recovery scoring.

    Takes displacement fields from sequential timepoint registrations
    and computes recovery metrics.
    """

    def __init__(
        self,
        complete_threshold: float = 0.85,
        partial_threshold: float = 0.50,
        normal_jac_range: Tuple[float, float] = (0.8, 1.2),
    ):
        """
        Args:
            complete_threshold: recovery score threshold for "complete recovery"
            partial_threshold: threshold for "partial recovery" (below = fibrotic)
            normal_jac_range: Jacobian determinant range considered normal
        """
        super().__init__()
        self.complete_threshold = complete_threshold
        self.partial_threshold = partial_threshold
        self.normal_jac_range = normal_jac_range

    def compute_jacobian_map(
        self, displacement: torch.Tensor
    ) -> torch.Tensor:
        """Compute Jacobian determinant map from displacement field.

        Args:
            displacement: (B, 3, D, H, W) displacement field

        Returns:
            jac_det: (B, 1, D, H, W) Jacobian determinant map
                - det(J) > 1: local expansion
                - det(J) = 1: no volume change
                - 0 < det(J) < 1: local contraction
                - det(J) ≤ 0: topology folding (invalid)
        """
        return compute_jacobian_determinant(displacement)

    def compute_recovery_score(
        self,
        jac_det: torch.Tensor,
        lung_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute recovery score from Jacobian determinant map.

        Recovery score measures how much the lung tissue has returned
        to normal volume (det(J) ≈ 1).

        Args:
            jac_det: (B, 1, D, H, W) Jacobian determinant map
            lung_mask: (B, 1, D, H, W) optional lung mask

        Returns:
            Dict with recovery metrics
        """
        if lung_mask is not None:
            # Trim mask to match Jacobian map size
            d, h, w = jac_det.shape[2:]
            mask = lung_mask[:, :, :d, :h, :w]
            masked_jac = jac_det[mask > 0]
        else:
            masked_jac = jac_det.flatten()

        if masked_jac.numel() == 0:
            return {"recovery_score": 0.0, "status": "error"}

        jac_np = masked_jac.detach().cpu().numpy()

        # Recovery score: fraction of voxels with normal Jacobian
        normal_low, normal_high = self.normal_jac_range
        normal_fraction = np.mean(
            (jac_np >= normal_low) & (jac_np <= normal_high)
        )

        # Additional metrics
        mean_jac = float(np.mean(jac_np))
        std_jac = float(np.std(jac_np))
        neg_fraction = float(np.mean(jac_np <= 0))  # Topology violations
        expansion_fraction = float(np.mean(jac_np > normal_high))
        contraction_fraction = float(np.mean(jac_np < normal_low))

        # Classification
        if normal_fraction >= self.complete_threshold:
            status = "complete_recovery"
        elif normal_fraction >= self.partial_threshold:
            status = "partial_recovery"
        else:
            status = "persistent_abnormality"

        return {
            "recovery_score": float(normal_fraction),
            "mean_jacobian": mean_jac,
            "std_jacobian": std_jac,
            "negative_jacobian_fraction": neg_fraction,
            "expansion_fraction": expansion_fraction,
            "contraction_fraction": contraction_fraction,
            "status": status,
        }

    def compute_regional_scores(
        self,
        jac_det: torch.Tensor,
        lung_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute recovery scores for upper/middle/lower lung regions.

        Args:
            jac_det: (B, 1, D, H, W) Jacobian determinant map
            lung_mask: (B, 1, D, H, W) optional lung mask

        Returns:
            Dict mapping region_name -> recovery_metrics
        """
        depth = jac_det.shape[2]
        third = depth // 3

        regions = {
            "upper": slice(0, third),
            "middle": slice(third, 2 * third),
            "lower": slice(2 * third, depth),
        }

        results = {}
        for region_name, d_slice in regions.items():
            region_jac = jac_det[:, :, d_slice, :, :]
            region_mask = None
            if lung_mask is not None:
                d = region_jac.shape[2]
                h = region_jac.shape[3]
                w = region_jac.shape[4]
                region_mask = lung_mask[:, :, d_slice, :h, :w]

            results[region_name] = self.compute_recovery_score(
                region_jac, region_mask
            )

        return results

    def compute_density_change(
        self,
        baseline_vol: torch.Tensor,
        followup_vol: torch.Tensor,
        warped_vol: torch.Tensor,
        lung_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Analyze density changes between baseline and follow-up.

        After registration (alignment), differences in intensity
        reflect actual tissue density changes (recovery).

        Args:
            baseline_vol: (B, 1, D, H, W) baseline CT
            followup_vol: (B, 1, D, H, W) follow-up CT (unwarped)
            warped_vol:   (B, 1, D, H, W) warped baseline (aligned to follow-up)
            lung_mask:    (B, 1, D, H, W) optional lung mask

        Returns:
            Density change metrics
        """
        if lung_mask is not None:
            mask = lung_mask > 0
            baseline_vals = baseline_vol[mask].detach().cpu().numpy()
            warped_vals = warped_vol[mask].detach().cpu().numpy()
            followup_vals = followup_vol[mask].detach().cpu().numpy()
        else:
            baseline_vals = baseline_vol.flatten().detach().cpu().numpy()
            warped_vals = warped_vol.flatten().detach().cpu().numpy()
            followup_vals = followup_vol.flatten().detach().cpu().numpy()

        # Density difference after alignment
        density_diff = warped_vals - followup_vals

        return {
            "mean_baseline_density": float(np.mean(baseline_vals)),
            "mean_followup_density": float(np.mean(followup_vals)),
            "mean_density_diff": float(np.mean(density_diff)),
            "std_density_diff": float(np.std(density_diff)),
            "mae_density": float(np.mean(np.abs(density_diff))),
            "density_correlation": float(
                np.corrcoef(warped_vals.flatten(), followup_vals.flatten())[0, 1]
                if len(warped_vals) > 1
                else 0.0
            ),
        }

    def analyze_trajectory(
        self,
        timepoint_scores: List[Dict[str, float]],
    ) -> Dict[str, object]:
        """Analyze recovery trajectory over multiple timepoints.

        Args:
            timepoint_scores: List of recovery score dicts, one per timepoint

        Returns:
            Trajectory analysis with trend, rate, and prediction
        """
        if len(timepoint_scores) < 2:
            return {"error": "Need at least 2 timepoints for trajectory analysis"}

        scores = [s["recovery_score"] for s in timepoint_scores]
        timepoints = list(range(len(scores)))

        # Linear regression for trend
        t = np.array(timepoints, dtype=np.float64)
        s = np.array(scores, dtype=np.float64)

        # Fit line: score = slope * time + intercept
        n = len(t)
        slope = (n * np.sum(t * s) - np.sum(t) * np.sum(s)) / (
            n * np.sum(t ** 2) - np.sum(t) ** 2 + 1e-8
        )
        intercept = (np.sum(s) - slope * np.sum(t)) / n

        # R² goodness of fit
        predicted = slope * t + intercept
        ss_res = np.sum((s - predicted) ** 2)
        ss_tot = np.sum((s - np.mean(s)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-8)

        # Trend classification
        if slope > 0.02:
            trend = "improving"
        elif slope < -0.02:
            trend = "worsening"
        else:
            trend = "stable"

        # Predict time to complete recovery (if improving)
        if slope > 0 and scores[-1] < self.complete_threshold:
            est_time = (self.complete_threshold - intercept) / slope
        else:
            est_time = None

        return {
            "scores": scores,
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
            "trend": trend,
            "current_status": timepoint_scores[-1].get("status", "unknown"),
            "estimated_recovery_timepoint": float(est_time) if est_time else None,
        }

    def forward(
        self,
        displacement: torch.Tensor,
        lung_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """Full recovery analysis from a displacement field.

        Args:
            displacement: (B, 3, D, H, W)
            lung_mask: (B, 1, D, H, W)

        Returns:
            Comprehensive recovery analysis dict
        """
        jac_det = self.compute_jacobian_map(displacement)

        overall = self.compute_recovery_score(jac_det, lung_mask)
        regional = self.compute_regional_scores(jac_det, lung_mask)

        return {
            "overall": overall,
            "regional": regional,
            "jacobian_map": jac_det,
        }
