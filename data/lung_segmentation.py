"""
Automated Lung Segmentation
Extracts lung regions from CT volumes using intensity thresholding
and morphological operations for focused registration.
"""

import logging
from typing import Tuple, Optional

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


class LungSegmenter:
    """Automated lung segmentation from CT volumes.

    Pipeline:
        1. Threshold at air/tissue boundary (-500 HU, or normalized equivalent)
        2. Connected component analysis to find lung regions
        3. Morphological closing to fill small holes
        4. Remove non-lung components (trachea, background air)
        5. Optional: dilation for registration margin
    """

    def __init__(
        self,
        threshold: float = 0.3,  # For normalized [0,1] data
        hu_threshold: float = -500.0,  # For HU data
        min_lung_volume: int = 5000,  # Minimum voxels for a lung region
        closing_radius: int = 5,
        dilation_radius: int = 3,
        fill_holes: bool = True,
    ):
        self.threshold = threshold
        self.hu_threshold = hu_threshold
        self.min_lung_volume = min_lung_volume
        self.closing_radius = closing_radius
        self.dilation_radius = dilation_radius
        self.fill_holes = fill_holes

    def segment(
        self,
        volume: np.ndarray,
        is_normalized: bool = True,
    ) -> np.ndarray:
        """Segment lungs from a CT volume.

        Args:
            volume: 3D numpy array (D, H, W)
            is_normalized: True if values in [0,1]; False if in HU

        Returns:
            Binary lung mask (same shape as input)
        """
        # Step 1: Initial thresholding
        if is_normalized:
            binary = volume < self.threshold
        else:
            binary = volume < self.hu_threshold

        binary = binary.astype(np.uint8)

        # Step 2: Connected component analysis
        labeled, num_features = ndimage.label(binary)
        logger.debug(f"Found {num_features} connected components")

        if num_features == 0:
            logger.warning("No components found, returning empty mask")
            return np.zeros_like(volume, dtype=np.uint8)

        # Step 3: Find the two largest non-background components (left + right lung)
        component_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        component_indices = np.argsort(component_sizes)[::-1]  # Descending

        # Remove background air (the largest component touching borders)
        lung_mask = np.zeros_like(binary)
        num_lungs_found = 0

        for idx in component_indices:
            comp_label = idx + 1
            comp_mask = labeled == comp_label
            comp_size = component_sizes[idx]

            if comp_size < self.min_lung_volume:
                continue

            # Check if this component touches all 6 faces (likely background)
            touches_border = (
                comp_mask[0, :, :].any()
                and comp_mask[-1, :, :].any()
                and comp_mask[:, 0, :].any()
                and comp_mask[:, -1, :].any()
                and comp_mask[:, :, 0].any()
                and comp_mask[:, :, -1].any()
            )

            if touches_border:
                continue  # Skip background air

            # Check if roughly in the center of the volume (lung region)
            centroid = ndimage.center_of_mass(comp_mask)
            center = np.array(volume.shape) / 2.0
            relative_pos = np.abs(np.array(centroid) - center) / center

            if relative_pos.max() > 0.85:
                continue  # Too far from center

            lung_mask |= comp_mask.astype(np.uint8)
            num_lungs_found += 1

            if num_lungs_found >= 2:
                break  # Found both lungs

        logger.debug(f"Found {num_lungs_found} lung regions")

        # Step 4: Morphological closing
        if self.closing_radius > 0:
            struct = ndimage.generate_binary_structure(3, 1)
            struct = ndimage.iterate_structure(struct, self.closing_radius)
            lung_mask = ndimage.binary_closing(lung_mask, structure=struct).astype(
                np.uint8
            )

        # Step 5: Fill holes
        if self.fill_holes:
            lung_mask = ndimage.binary_fill_holes(lung_mask).astype(np.uint8)

        # Step 6: Optional dilation for registration margin
        if self.dilation_radius > 0:
            struct = ndimage.generate_binary_structure(3, 1)
            struct = ndimage.iterate_structure(struct, self.dilation_radius)
            lung_mask = ndimage.binary_dilation(lung_mask, structure=struct).astype(
                np.uint8
            )

        voxel_count = lung_mask.sum()
        total_voxels = volume.size
        logger.info(
            f"Lung segmentation: {voxel_count} voxels "
            f"({100 * voxel_count / total_voxels:.1f}% of volume)"
        )

        return lung_mask

    def extract_lung_region(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None,
        background_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract lung region from volume, masking non-lung areas.

        Args:
            volume: 3D CT volume
            mask: Pre-computed lung mask (computed if None)
            background_value: Value for non-lung voxels

        Returns:
            (masked_volume, lung_mask)
        """
        if mask is None:
            mask = self.segment(volume)

        masked = volume.copy()
        masked[mask == 0] = background_value
        return masked, mask

    def compute_lung_statistics(
        self, volume: np.ndarray, mask: np.ndarray
    ) -> dict:
        """Compute statistics of lung region for recovery analysis.

        Returns:
            Dict with mean_density, std_density, volume_ratio,
            and regional stats (upper/middle/lower thirds).
        """
        lung_voxels = volume[mask > 0]

        if len(lung_voxels) == 0:
            return {"error": "No lung voxels found"}

        # Overall statistics
        stats = {
            "mean_density": float(lung_voxels.mean()),
            "std_density": float(lung_voxels.std()),
            "min_density": float(lung_voxels.min()),
            "max_density": float(lung_voxels.max()),
            "volume_voxels": int(mask.sum()),
            "volume_ratio": float(mask.sum() / mask.size),
        }

        # Regional analysis (upper / middle / lower thirds along axial)
        depth = volume.shape[0]
        third = depth // 3
        for region_name, z_slice in [
            ("upper", slice(0, third)),
            ("middle", slice(third, 2 * third)),
            ("lower", slice(2 * third, depth)),
        ]:
            region_mask = mask[z_slice]
            region_vol = volume[z_slice]
            region_voxels = region_vol[region_mask > 0]

            if len(region_voxels) > 0:
                stats[f"{region_name}_mean_density"] = float(region_voxels.mean())
                stats[f"{region_name}_volume_ratio"] = float(
                    region_mask.sum() / region_mask.size
                )
            else:
                stats[f"{region_name}_mean_density"] = 0.0
                stats[f"{region_name}_volume_ratio"] = 0.0

        return stats
