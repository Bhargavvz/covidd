"""
Registration Inference Pipeline
Load a trained model and register a pair of CT volumes.
Outputs warped image, displacement field, and Jacobian map.

Usage:
    python inference/register.py \\
        --checkpoint outputs/checkpoints/best_model.pth \\
        --moving path/to/moving.nii.gz \\
        --fixed path/to/fixed.nii.gz \\
        --output-dir results/
"""

import sys
import logging
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from models.voxelmorph import build_model
from models.spatial_transformer import compute_jacobian_determinant
from data.preprocessing import CTPreprocessor
from utils.io_utils import (
    load_volume_sitk,
    save_volume_sitk,
    save_displacement_field,
    save_json,
)
from utils.metrics import evaluate_registration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("register")


def register_pair(
    model: torch.nn.Module,
    moving: np.ndarray,
    fixed: np.ndarray,
    device: str = "cuda",
) -> dict:
    """Register a pair of volumes.

    Args:
        model: trained registration model
        moving: (D, H, W) moving volume
        fixed: (D, H, W) fixed volume
        device: computation device

    Returns:
        dict with 'warped', 'displacement', 'jacobian_det'
    """
    model.eval()

    # Prepare tensors: (1, 1, D, H, W)
    moving_t = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0).float().to(device)
    fixed_t = torch.from_numpy(fixed).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            warped_t, flow_t = model(moving_t, fixed_t)

    # Compute Jacobian determinant
    jac_det = compute_jacobian_determinant(flow_t.float())

    return {
        "warped": warped_t[0, 0].cpu().numpy(),
        "displacement": flow_t[0].cpu().numpy(),  # (3, D, H, W)
        "jacobian_det": jac_det[0, 0].cpu().numpy(),
    }


def main():
    parser = argparse.ArgumentParser(description="Register CT volume pair")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--moving", type=str, required=True, help="Moving volume path")
    parser.add_argument("--fixed", type=str, required=True, help="Fixed volume path")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--volume-size", type=int, nargs=3, default=[192, 192, 192])
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = checkpoint.get("config", {}).get("model", {})

    # Build model
    vol_size = tuple(args.volume_size)
    model_config["vol_size"] = vol_size
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Preprocess volumes
    preprocessor = CTPreprocessor(target_size=vol_size)
    logger.info(f"Loading moving: {args.moving}")
    moving = preprocessor.process(args.moving)
    logger.info(f"Loading fixed: {args.fixed}")
    fixed = preprocessor.process(args.fixed)

    # Register
    logger.info("Running registration...")
    result = register_pair(model, moving, fixed, device)

    # Evaluate
    metrics = evaluate_registration(
        fixed=fixed,
        moving=moving,
        warped=result["warped"],
        displacement=result["displacement"],
    )

    logger.info("Registration metrics:")
    for key, val in metrics.items():
        logger.info(f"  {key}: {val:.6f}")

    # Save outputs
    spacing = (1.5, 1.5, 1.5)
    save_volume_sitk(result["warped"], str(output_dir / "warped.nii.gz"), spacing=spacing)
    save_displacement_field(result["displacement"], str(output_dir / "displacement.nii.gz"), spacing=spacing)
    save_volume_sitk(result["jacobian_det"], str(output_dir / "jacobian_det.nii.gz"), spacing=spacing)
    save_volume_sitk(moving, str(output_dir / "moving_preprocessed.nii.gz"), spacing=spacing)
    save_volume_sitk(fixed, str(output_dir / "fixed_preprocessed.nii.gz"), spacing=spacing)
    save_json(metrics, str(output_dir / "metrics.json"))

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
