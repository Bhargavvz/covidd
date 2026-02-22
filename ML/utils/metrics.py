"""
Evaluation Metrics for Registration Quality
Dice coefficient, Target Registration Error, SSIM, and more.
"""

import torch
import numpy as np
from typing import Dict, Optional


def compute_dice(
    pred_seg: np.ndarray, target_seg: np.ndarray, labels: list = None
) -> Dict[str, float]:
    """Compute Dice coefficient between predicted and target segmentations.

    Args:
        pred_seg: predicted segmentation array
        target_seg: target segmentation array
        labels: list of label values to evaluate (default: all unique)

    Returns:
        Dict mapping label -> Dice score
    """
    if labels is None:
        labels = list(set(np.unique(pred_seg)) | set(np.unique(target_seg)))
        labels = [l for l in labels if l > 0]  # Exclude background

    results = {}
    for label in labels:
        pred_binary = (pred_seg == label).astype(np.float32)
        target_binary = (target_seg == label).astype(np.float32)

        intersection = (pred_binary * target_binary).sum()
        denominator = pred_binary.sum() + target_binary.sum()

        if denominator == 0:
            dice = 1.0  # Both empty
        else:
            dice = 2.0 * intersection / denominator

        results[f"dice_label_{label}"] = float(dice)

    if results:
        results["dice_mean"] = float(np.mean(list(results.values())))

    return results


def compute_tre(
    pred_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """Compute Target Registration Error from landmark pairs.

    Args:
        pred_landmarks: (N, 3) predicted landmark positions
        target_landmarks: (N, 3) target landmark positions
        voxel_spacing: (sx, sy, sz) voxel spacing in mm

    Returns:
        TRE statistics in mm
    """
    spacing = np.array(voxel_spacing)
    diff = (pred_landmarks - target_landmarks) * spacing
    distances = np.sqrt(np.sum(diff ** 2, axis=1))

    return {
        "tre_mean": float(np.mean(distances)),
        "tre_std": float(np.std(distances)),
        "tre_median": float(np.median(distances)),
        "tre_max": float(np.max(distances)),
        "tre_25th": float(np.percentile(distances, 25)),
        "tre_75th": float(np.percentile(distances, 75)),
    }


def compute_ssim_3d(
    image1: np.ndarray,
    image2: np.ndarray,
    window_size: int = 7,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """Compute mean SSIM between two 3D volumes (numpy implementation)."""
    from scipy.ndimage import uniform_filter

    mu1 = uniform_filter(image1.astype(np.float64), size=window_size)
    mu2 = uniform_filter(image2.astype(np.float64), size=window_size)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(image1.astype(np.float64) ** 2, size=window_size) - mu1_sq
    sigma2_sq = uniform_filter(image2.astype(np.float64) ** 2, size=window_size) - mu2_sq
    sigma12 = uniform_filter(
        image1.astype(np.float64) * image2.astype(np.float64), size=window_size
    ) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return float(np.mean(ssim_map))


def compute_jacobian_stats(
    displacement: torch.Tensor,
) -> Dict[str, float]:
    """Compute Jacobian determinant statistics from displacement field.

    Args:
        displacement: (B, 3, D, H, W) or (3, D, H, W)

    Returns:
        Statistics of Jacobian determinant
    """
    from models.spatial_transformer import compute_jacobian_determinant

    if displacement.ndim == 4:
        displacement = displacement.unsqueeze(0)

    jac_det = compute_jacobian_determinant(displacement)
    jac_np = jac_det.detach().cpu().numpy().flatten()

    return {
        "jac_mean": float(np.mean(jac_np)),
        "jac_std": float(np.std(jac_np)),
        "jac_min": float(np.min(jac_np)),
        "jac_max": float(np.max(jac_np)),
        "jac_neg_pct": float(np.mean(jac_np <= 0) * 100),
        "jac_num_neg": int(np.sum(jac_np <= 0)),
    }


def compute_displacement_stats(
    displacement: np.ndarray,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """Compute displacement field statistics.

    Args:
        displacement: (3, D, H, W) or (B, 3, D, H, W)
        voxel_spacing: voxel spacing in mm

    Returns:
        Displacement statistics in mm
    """
    if displacement.ndim == 5:
        displacement = displacement[0]  # Take first batch

    spacing = np.array(voxel_spacing).reshape(3, 1, 1, 1)
    disp_mm = displacement * spacing

    magnitude = np.sqrt(np.sum(disp_mm ** 2, axis=0))

    return {
        "disp_mean_mm": float(np.mean(magnitude)),
        "disp_std_mm": float(np.std(magnitude)),
        "disp_max_mm": float(np.max(magnitude)),
        "disp_median_mm": float(np.median(magnitude)),
        "disp_95th_mm": float(np.percentile(magnitude, 95)),
    }


def evaluate_registration(
    fixed: np.ndarray,
    moving: np.ndarray,
    warped: np.ndarray,
    displacement: np.ndarray,
    fixed_seg: Optional[np.ndarray] = None,
    warped_seg: Optional[np.ndarray] = None,
    voxel_spacing: tuple = (1.5, 1.5, 1.5),
) -> Dict[str, float]:
    """Comprehensive registration evaluation.

    Args:
        fixed: fixed target volume
        moving: original moving volume
        warped: warped (registered) moving volume
        displacement: displacement field
        fixed_seg: fixed volume segmentation (optional)
        warped_seg: warped segmentation (optional)
        voxel_spacing: voxel spacing in mm

    Returns:
        Dict with all evaluation metrics
    """
    results = {}

    # Image similarity
    results["ssim_before"] = compute_ssim_3d(fixed, moving)
    results["ssim_after"] = compute_ssim_3d(fixed, warped)
    results["ssim_improvement"] = results["ssim_after"] - results["ssim_before"]

    results["mse_before"] = float(np.mean((fixed - moving) ** 2))
    results["mse_after"] = float(np.mean((fixed - warped) ** 2))

    # NCC
    def _ncc(a, b):
        a_norm = a - a.mean()
        b_norm = b - b.mean()
        return float(
            np.sum(a_norm * b_norm) / (np.sqrt(np.sum(a_norm ** 2) * np.sum(b_norm ** 2)) + 1e-8)
        )

    results["ncc_before"] = _ncc(fixed, moving)
    results["ncc_after"] = _ncc(fixed, warped)

    # Displacement statistics
    disp_stats = compute_displacement_stats(displacement, voxel_spacing)
    results.update(disp_stats)

    # Jacobian statistics
    disp_tensor = torch.from_numpy(displacement).unsqueeze(0).float()
    jac_stats = compute_jacobian_stats(disp_tensor)
    results.update(jac_stats)

    # Dice (if segmentations available)
    if fixed_seg is not None and warped_seg is not None:
        dice_results = compute_dice(warped_seg, fixed_seg)
        results.update(dice_results)

    return results
