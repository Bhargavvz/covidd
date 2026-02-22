"""
Longitudinal Recovery Analysis Pipeline
Processes multiple timepoints per patient and generates recovery reports.

Usage:
    python inference/analyze_recovery.py \\
        --checkpoint outputs/checkpoints/best_model.pth \\
        --patient-dir datasets/patient_001/ \\
        --output-dir results/recovery/
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from models.voxelmorph import build_model
from models.recovery_analyzer import RecoveryAnalyzer
from data.preprocessing import CTPreprocessor
from data.lung_segmentation import LungSegmenter
from inference.register import register_pair
from inference.visualize import plot_recovery_trajectory
from utils.io_utils import save_json, save_volume_sitk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analyze_recovery")


def analyze_patient_recovery(
    model: torch.nn.Module,
    volume_paths: List[str],
    timepoint_labels: List[str],
    preprocessor: CTPreprocessor,
    device: str = "cuda",
) -> Dict:
    """Analyze lung recovery for a single patient across timepoints.

    Registers each follow-up to the baseline and computes
    recovery metrics at each timepoint.

    Args:
        model: trained registration model
        volume_paths: list of volume paths ordered by timepoint
        timepoint_labels: labels for each timepoint
        preprocessor: CT preprocessor instance
        device: computation device

    Returns:
        Comprehensive recovery analysis dict
    """
    analyzer = RecoveryAnalyzer()
    segmenter = LungSegmenter()

    # Load and preprocess all volumes
    volumes = []
    lung_masks = []
    for path in volume_paths:
        vol = preprocessor.process(path)
        mask = segmenter.segment(vol)
        volumes.append(vol)
        lung_masks.append(mask)

    logger.info(f"Loaded {len(volumes)} timepoints")

    # Use first timepoint as baseline
    baseline = volumes[0]

    # Register each follow-up to baseline
    timepoint_results = []
    for i in range(len(volumes)):
        if i == 0:
            # Baseline - identity registration
            timepoint_results.append({
                "timepoint": timepoint_labels[i],
                "recovery_score": 0.0,  # Baseline (disease state)
                "status": "baseline",
                "regional": {
                    "upper": {"recovery_score": 0.0},
                    "middle": {"recovery_score": 0.0},
                    "lower": {"recovery_score": 0.0},
                },
            })
            continue

        logger.info(f"Registering {timepoint_labels[i]} to baseline...")
        result = register_pair(model, baseline, volumes[i], device)

        # Convert to tensors for recovery analysis
        disp_tensor = torch.from_numpy(result["displacement"]).unsqueeze(0).float()
        mask_tensor = (
            torch.from_numpy(lung_masks[i]).unsqueeze(0).unsqueeze(0).float()
        )

        # Compute recovery scores
        analysis = analyzer(disp_tensor, mask_tensor)

        # Regional analysis
        regional = analyzer.compute_regional_scores(
            analysis["jacobian_map"], mask_tensor
        )

        # Density change
        baseline_t = torch.from_numpy(baseline).unsqueeze(0).unsqueeze(0).float()
        followup_t = torch.from_numpy(volumes[i]).unsqueeze(0).unsqueeze(0).float()
        warped_t = torch.from_numpy(result["warped"]).unsqueeze(0).unsqueeze(0).float()
        density = analyzer.compute_density_change(
            baseline_t, followup_t, warped_t, mask_tensor
        )

        timepoint_results.append({
            "timepoint": timepoint_labels[i],
            "recovery_score": analysis["overall"]["recovery_score"],
            "status": analysis["overall"]["status"],
            "overall_metrics": analysis["overall"],
            "regional": {k: v for k, v in regional.items()},
            "density_change": density,
        })

        logger.info(
            f"  {timepoint_labels[i]}: "
            f"score={analysis['overall']['recovery_score']:.4f} "
            f"status={analysis['overall']['status']}"
        )

    # Trajectory analysis
    trajectory = analyzer.analyze_trajectory(
        [r for r in timepoint_results if "overall_metrics" in r]
    )

    return {
        "timepoints": timepoint_results,
        "trajectory": trajectory,
        "num_timepoints": len(volumes),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze longitudinal lung recovery")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--patient-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./results/recovery")
    parser.add_argument("--volume-size", type=int, nargs=3, default=[192, 192, 192])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--timepoint-labels",
        type=str,
        nargs="+",
        default=None,
        help="Labels for each timepoint",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = checkpoint.get("config", {}).get("model", {})
    model_config["vol_size"] = tuple(args.volume_size)
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Find volumes in patient directory
    patient_dir = Path(args.patient_dir)
    volume_paths = sorted(patient_dir.rglob("*.nii.gz"))
    if not volume_paths:
        logger.error(f"No volumes found in {patient_dir}")
        return

    logger.info(f"Found {len(volume_paths)} volumes in {patient_dir}")

    # Timepoint labels
    if args.timepoint_labels:
        labels = args.timepoint_labels
    else:
        labels = [f"Timepoint {i}" for i in range(len(volume_paths))]
        labels[0] = "Baseline"

    # Preprocess
    preprocessor = CTPreprocessor(target_size=tuple(args.volume_size))

    # Analyze
    results = analyze_patient_recovery(
        model=model,
        volume_paths=[str(p) for p in volume_paths],
        timepoint_labels=labels,
        preprocessor=preprocessor,
        device=device,
    )

    # Save report
    save_json(results, str(output_dir / "recovery_report.json"))

    # Generate recovery trajectory plot
    scores = [r["recovery_score"] for r in results["timepoints"]]
    regional_scores = {}
    for region in ["upper", "middle", "lower"]:
        regional_scores[region] = [
            r.get("regional", {}).get(region, {}).get("recovery_score", 0.0)
            for r in results["timepoints"]
        ]

    plot_recovery_trajectory(
        timepoint_labels=labels,
        recovery_scores=scores,
        regional_scores=regional_scores,
        save_path=str(output_dir / "recovery_trajectory.png"),
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("RECOVERY ANALYSIS SUMMARY")
    logger.info("=" * 60)
    for r in results["timepoints"]:
        logger.info(
            f"  {r['timepoint']}: score={r['recovery_score']:.4f} | {r['status']}"
        )
    if "trajectory" in results and results["trajectory"]:
        traj = results["trajectory"]
        logger.info(f"\n  Trend: {traj.get('trend', 'N/A')}")
        logger.info(f"  Rate:  {traj.get('slope', 0):.4f} per timepoint")
        if traj.get("estimated_recovery_timepoint"):
            logger.info(
                f"  Est. complete recovery: timepoint {traj['estimated_recovery_timepoint']:.1f}"
            )
    logger.info("=" * 60)
    logger.info(f"Report saved to {output_dir / 'recovery_report.json'}")


if __name__ == "__main__":
    main()
