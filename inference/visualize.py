"""
Visualization Module for Registration Results
Generates deformation grids, Jacobian maps, side-by-side comparisons,
and recovery progression animations.

Usage:
    python inference/visualize.py \\
        --results-dir results/ \\
        --output-dir results/figures/
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

logger = logging.getLogger("visualize")


def plot_registration_comparison(
    fixed: np.ndarray,
    moving: np.ndarray,
    warped: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 0,
    save_path: Optional[str] = None,
    title: str = "Registration Result",
) -> plt.Figure:
    """Side-by-side comparison: Fixed | Moving | Warped | Difference.

    Args:
        fixed: (D, H, W) fixed volume
        moving: (D, H, W) moving volume
        warped: (D, H, W) warped volume
        slice_idx: slice to display (default: middle)
        axis: axis along which to slice (0=axial, 1=coronal, 2=sagittal)
        save_path: optional path to save figure
    """
    if slice_idx is None:
        slice_idx = fixed.shape[axis] // 2

    slicer = [slice(None)] * 3
    slicer[axis] = slice_idx

    f_slice = fixed[tuple(slicer)]
    m_slice = moving[tuple(slicer)]
    w_slice = warped[tuple(slicer)]
    diff_before = np.abs(f_slice - m_slice)
    diff_after = np.abs(f_slice - w_slice)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Row 1: Volumes
    axes[0, 0].imshow(f_slice, cmap="gray", origin="lower")
    axes[0, 0].set_title("Fixed (Target)", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(m_slice, cmap="gray", origin="lower")
    axes[0, 1].set_title("Moving (Source)", fontsize=12)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(w_slice, cmap="gray", origin="lower")
    axes[0, 2].set_title("Warped (Registered)", fontsize=12)
    axes[0, 2].axis("off")

    # Row 2: Differences
    axes[1, 0].imshow(diff_before, cmap="hot", origin="lower", vmin=0, vmax=0.5)
    axes[1, 0].set_title("| Fixed - Moving |", fontsize=12)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(diff_after, cmap="hot", origin="lower", vmin=0, vmax=0.5)
    axes[1, 1].set_title("| Fixed - Warped |", fontsize=12)
    axes[1, 1].axis("off")

    # Overlay
    overlay = np.zeros((*f_slice.shape, 3))
    overlay[..., 0] = f_slice  # Red channel = fixed
    overlay[..., 1] = w_slice  # Green channel = warped
    overlay[..., 2] = 0
    overlay = np.clip(overlay, 0, 1)
    axes[1, 2].imshow(overlay, origin="lower")
    axes[1, 2].set_title("Overlay (R=Fixed, G=Warped)", fontsize=12)
    axes[1, 2].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved comparison figure: {save_path}")

    return fig


def plot_deformation_grid(
    displacement: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 0,
    grid_spacing: int = 4,
    save_path: Optional[str] = None,
    background: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Visualize displacement field as a deformed grid overlay.

    Args:
        displacement: (3, D, H, W) displacement field
        slice_idx: slice to display
        axis: slicing axis
        grid_spacing: spacing between grid lines
        save_path: optional save path
        background: optional background image to overlay grid on
    """
    if slice_idx is None:
        slice_idx = displacement.shape[axis + 1] // 2

    # Get 2D displacement for the slice
    slicer = [slice(None)] * 4
    slicer[axis + 1] = slice_idx

    disp_slice = displacement[tuple(slicer)]  # (3, H, W) or (3, D, W) etc.

    # Pick the two in-plane displacement components
    dims = [0, 1, 2]
    dims.remove(axis)
    u = disp_slice[dims[0]]
    v = disp_slice[dims[1]]

    h, w = u.shape

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Background
    if background is not None:
        bg_slicer = [slice(None)] * 3
        bg_slicer[axis] = slice_idx
        bg_slice = background[tuple(bg_slicer)]
        ax.imshow(bg_slice, cmap="gray", origin="lower", alpha=0.5)

    # Draw deformed grid
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    deformed_x = grid_x + u
    deformed_y = grid_y + v

    # Horizontal lines
    for i in range(0, h, grid_spacing):
        ax.plot(deformed_x[i, :], deformed_y[i, :], "c-", linewidth=0.5, alpha=0.7)

    # Vertical lines
    for j in range(0, w, grid_spacing):
        ax.plot(deformed_x[:, j], deformed_y[:, j], "c-", linewidth=0.5, alpha=0.7)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_title("Deformation Grid", fontsize=14)
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved deformation grid: {save_path}")

    return fig


def plot_jacobian_map(
    jacobian_det: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 0,
    save_path: Optional[str] = None,
    background: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Color-coded Jacobian determinant map.

    Blue = expansion (det > 1)
    White = no change (det ≈ 1)
    Red = contraction (0 < det < 1)
    Black = folding (det ≤ 0)
    """
    if slice_idx is None:
        slice_idx = jacobian_det.shape[axis] // 2

    slicer = [slice(None)] * 3
    slicer[axis] = slice_idx
    jac_slice = jacobian_det[tuple(slicer)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Jacobian map with diverging colormap
    cmap = plt.cm.RdYlBu  # Red = contraction, Blue = expansion
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0)

    im = axes[0].imshow(jac_slice, cmap=cmap, norm=norm, origin="lower")
    axes[0].set_title("Jacobian Determinant Map", fontsize=13)
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.046, label="det(J)")

    # Histogram
    jac_flat = jacobian_det.flatten()
    jac_flat = jac_flat[~np.isnan(jac_flat)]

    axes[1].hist(jac_flat, bins=100, range=(-0.5, 3.0), color="steelblue", alpha=0.8)
    axes[1].axvline(x=1.0, color="red", linestyle="--", linewidth=2, label="det(J)=1 (no change)")
    axes[1].axvline(x=0.0, color="black", linestyle="--", linewidth=2, label="det(J)=0 (folding)")
    axes[1].set_xlabel("Jacobian Determinant", fontsize=12)
    axes[1].set_ylabel("Voxel Count", fontsize=12)
    axes[1].set_title("Jacobian Distribution", fontsize=13)
    axes[1].legend()

    neg_pct = np.mean(jac_flat <= 0) * 100
    axes[1].annotate(
        f"Negative: {neg_pct:.2f}%",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        fontsize=11,
        color="red" if neg_pct > 0 else "green",
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved Jacobian map: {save_path}")

    return fig


def plot_displacement_magnitude(
    displacement: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 0,
    save_path: Optional[str] = None,
    voxel_spacing: tuple = (1.5, 1.5, 1.5),
) -> plt.Figure:
    """Plot displacement magnitude map in mm.

    Args:
        displacement: (3, D, H, W)
        voxel_spacing: in mm
    """
    spacing = np.array(voxel_spacing).reshape(3, 1, 1, 1)
    disp_mm = displacement * spacing
    magnitude = np.sqrt(np.sum(disp_mm ** 2, axis=0))

    if slice_idx is None:
        slice_idx = magnitude.shape[axis] // 2

    slicer = [slice(None)] * 3
    slicer[axis] = slice_idx
    mag_slice = magnitude[tuple(slicer)]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(mag_slice, cmap="magma", origin="lower")
    ax.set_title("Displacement Magnitude (mm)", fontsize=14)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, label="mm")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved displacement magnitude: {save_path}")

    return fig


def plot_recovery_trajectory(
    timepoint_labels: list,
    recovery_scores: list,
    regional_scores: Optional[dict] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot recovery trajectory over time.

    Args:
        timepoint_labels: list of timepoint labels (e.g., ["Baseline", "3 months", "6 months"])
        recovery_scores: overall recovery scores at each timepoint
        regional_scores: optional dict of region_name -> scores list
        save_path: optional save path
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    x = list(range(len(timepoint_labels)))

    ax.plot(
        x,
        recovery_scores,
        "o-",
        color="#2196F3",
        linewidth=2.5,
        markersize=8,
        label="Overall Recovery",
        zorder=5,
    )

    if regional_scores:
        colors = {"upper": "#4CAF50", "middle": "#FF9800", "lower": "#F44336"}
        for region, scores in regional_scores.items():
            ax.plot(
                x,
                scores,
                "s--",
                color=colors.get(region, "gray"),
                linewidth=1.5,
                markersize=6,
                label=f"{region.capitalize()} Lobe",
                alpha=0.8,
            )

    # Threshold lines
    ax.axhline(y=0.85, color="green", linestyle=":", alpha=0.5, label="Complete Recovery")
    ax.axhline(y=0.50, color="orange", linestyle=":", alpha=0.5, label="Partial Recovery")

    ax.fill_between(x, 0.85, 1.0, alpha=0.1, color="green")
    ax.fill_between(x, 0.50, 0.85, alpha=0.1, color="orange")
    ax.fill_between(x, 0.0, 0.50, alpha=0.1, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(timepoint_labels, fontsize=11)
    ax.set_ylabel("Recovery Score", fontsize=13)
    ax.set_xlabel("Timepoint", fontsize=13)
    ax.set_title("Longitudinal Lung Recovery Trajectory", fontsize=15, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved recovery trajectory: {save_path}")

    return fig


def plot_training_curves(
    history: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training history curves.

    Args:
        history: dict with 'train_loss', 'val_loss', 'lr', etc.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    # Loss curves
    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", color="#1976D2")
    axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss", color="#F44336")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("Training & Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Similarity
    axes[0, 1].plot(epochs, history.get("train_sim", []), label="Train Sim", color="#1976D2")
    axes[0, 1].plot(epochs, history.get("val_sim", []), label="Val Sim", color="#F44336")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Similarity Loss (1-NCC)")
    axes[0, 1].set_title("Image Similarity")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Negative Jacobian
    if "val_neg_jac_pct" in history:
        axes[1, 0].plot(
            epochs, history["val_neg_jac_pct"], color="#4CAF50", linewidth=2
        )
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Negative Jacobian %")
        axes[1, 0].set_title("Topology Violations")
        axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    if "lr" in history:
        axes[1, 1].plot(epochs, history["lr"], color="#FF9800", linewidth=2)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        "Training Progress — COVID-19 Lung Recovery Registration",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved training curves: {save_path}")

    return fig


def create_recovery_gif(
    volumes: list,
    timepoint_labels: list,
    slice_idx: Optional[int] = None,
    axis: int = 0,
    save_path: str = "recovery_animation.gif",
    duration: float = 1.0,
):
    """Create animated GIF of recovery progression.

    Args:
        volumes: list of (D, H, W) volumes at different timepoints
        timepoint_labels: labels for each timepoint
        slice_idx: slice to animate
        axis: slicing axis
        save_path: output GIF path
        duration: seconds per frame
    """
    try:
        import imageio
    except ImportError:
        logger.error("imageio required for GIF creation: pip install imageio")
        return

    if slice_idx is None:
        slice_idx = volumes[0].shape[axis] // 2

    frames = []
    for vol, label in zip(volumes, timepoint_labels):
        slicer = [slice(None)] * 3
        slicer[axis] = slice_idx
        img_slice = vol[tuple(slicer)]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_slice, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.set_title(label, fontsize=16, fontweight="bold")
        ax.axis("off")

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(save_path, frames, duration=duration, loop=0)
    logger.info(f"Saved recovery animation: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize registration results")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--slice-idx", type=int, default=None)
    parser.add_argument("--axis", type=int, default=0)

    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir or results_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    from utils.io_utils import load_volume_sitk, load_displacement_field

    # Load results
    fixed, _ = load_volume_sitk(str(results_dir / "fixed_preprocessed.nii.gz"))
    moving, _ = load_volume_sitk(str(results_dir / "moving_preprocessed.nii.gz"))
    warped, _ = load_volume_sitk(str(results_dir / "warped.nii.gz"))

    displacement = load_displacement_field(str(results_dir / "displacement.nii.gz"))
    jac_det, _ = load_volume_sitk(str(results_dir / "jacobian_det.nii.gz"))

    slice_idx = args.slice_idx or fixed.shape[args.axis] // 2

    # Generate all visualizations
    plot_registration_comparison(
        fixed, moving, warped,
        slice_idx=slice_idx,
        axis=args.axis,
        save_path=str(output_dir / "registration_comparison.png"),
    )

    plot_deformation_grid(
        displacement,
        slice_idx=slice_idx,
        axis=args.axis,
        save_path=str(output_dir / "deformation_grid.png"),
        background=fixed,
    )

    plot_jacobian_map(
        jac_det,
        slice_idx=slice_idx,
        axis=args.axis,
        save_path=str(output_dir / "jacobian_map.png"),
    )

    plot_displacement_magnitude(
        displacement,
        slice_idx=slice_idx,
        axis=args.axis,
        save_path=str(output_dir / "displacement_magnitude.png"),
    )

    # Training history (if available)
    history_path = results_dir.parent / "training_history.json"
    if history_path.exists():
        from utils.io_utils import load_json

        history = load_json(str(history_path))
        plot_training_curves(
            history, save_path=str(output_dir / "training_curves.png")
        )

    logger.info(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
