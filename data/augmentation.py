"""
3D Data Augmentation for Registration Pairs
Applies consistent spatial transforms to both moving and fixed volumes.
"""

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, map_coordinates
from typing import Tuple, Optional


class RegistrationAugmentor:
    """Augmentation pipeline for registration training pairs.

    Applies the SAME spatial transform to both moving and fixed volumes
    to maintain correspondence, plus independent intensity augmentations.

    Augmentations:
        - Random 3D affine (rotation, scaling, translation)
        - Random elastic deformation  
        - Random intensity jittering
        - Random Gaussian noise
        - Random flipping (left-right only, anatomically valid)
    """

    def __init__(
        self,
        rotation_range: float = 15.0,      # degrees
        scale_range: Tuple[float, float] = (0.9, 1.1),
        translation_range: float = 10.0,   # voxels
        elastic_alpha: float = 2.0,
        elastic_sigma: float = 20.0,
        intensity_shift: float = 0.05,
        intensity_scale: Tuple[float, float] = (0.95, 1.05),
        noise_std: float = 0.02,
        flip_prob: float = 0.5,
        affine_prob: float = 0.5,
        elastic_prob: float = 0.3,
        intensity_prob: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.intensity_shift = intensity_shift
        self.intensity_scale = intensity_scale
        self.noise_std = noise_std
        self.flip_prob = flip_prob
        self.affine_prob = affine_prob
        self.elastic_prob = elastic_prob
        self.intensity_prob = intensity_prob

        if seed is not None:
            np.random.seed(seed)

    def _random_rotation_matrix(self) -> np.ndarray:
        """Generate a random 3D rotation matrix."""
        angles = np.radians(
            np.random.uniform(-self.rotation_range, self.rotation_range, 3)
        )

        # Rotation around z-axis
        cz, sz = np.cos(angles[0]), np.sin(angles[0])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        # Rotation around y-axis
        cy, sy = np.cos(angles[1]), np.sin(angles[1])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

        # Rotation around x-axis
        cx, sx = np.cos(angles[2]), np.sin(angles[2])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

        return Rz @ Ry @ Rx

    def _random_affine_transform(
        self, volume: np.ndarray
    ) -> np.ndarray:
        """Apply random affine transformation."""
        shape = np.array(volume.shape)
        center = shape / 2.0

        # Random rotation
        R = self._random_rotation_matrix()

        # Random scale
        scale = np.random.uniform(*self.scale_range, 3)
        S = np.diag(scale)

        # Combined rotation + scale matrix
        M = R @ S

        # Offset to rotate around center
        offset = center - M @ center

        # Random translation
        offset += np.random.uniform(
            -self.translation_range, self.translation_range, 3
        )

        return affine_transform(
            volume, M, offset=offset, order=1, mode="constant", cval=0.0
        )

    def _random_elastic_deformation(
        self, volume: np.ndarray
    ) -> np.ndarray:
        """Apply random elastic deformation."""
        shape = volume.shape

        # Random displacement field
        dx = gaussian_filter(
            np.random.randn(*shape) * self.elastic_alpha, self.elastic_sigma
        )
        dy = gaussian_filter(
            np.random.randn(*shape) * self.elastic_alpha, self.elastic_sigma
        )
        dz = gaussian_filter(
            np.random.randn(*shape) * self.elastic_alpha, self.elastic_sigma
        )

        coords = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(
            np.float32
        )
        coords[0] += dx
        coords[1] += dy
        coords[2] += dz

        return map_coordinates(volume, coords, order=1, mode="constant", cval=0.0)

    def _random_intensity_augmentation(self, volume: np.ndarray) -> np.ndarray:
        """Apply random intensity jittering and noise."""
        result = volume.copy()

        # Random brightness shift
        shift = np.random.uniform(-self.intensity_shift, self.intensity_shift)
        result = result + shift

        # Random contrast scale
        scale = np.random.uniform(*self.intensity_scale)
        mean_val = result.mean()
        result = (result - mean_val) * scale + mean_val

        # Gaussian noise
        noise = np.random.normal(0, self.noise_std, result.shape).astype(np.float32)
        result = result + noise

        # Clip to valid range
        result = np.clip(result, 0.0, 1.0)
        return result

    def __call__(
        self,
        moving: np.ndarray,
        fixed: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation to a registration pair.

        Spatial transforms are applied consistently to BOTH volumes.
        Intensity transforms are applied independently.

        Args:
            moving: (D, H, W) moving volume
            fixed:  (D, H, W) fixed volume

        Returns:
            (augmented_moving, augmented_fixed)
        """
        # --- Spatial augmentation (same for both) ---

        # Random affine
        if np.random.random() < self.affine_prob:
            # Store random state so same transform is applied to both
            state = np.random.get_state()
            moving = self._random_affine_transform(moving)
            np.random.set_state(state)
            fixed = self._random_affine_transform(fixed)

        # Random elastic deformation
        if np.random.random() < self.elastic_prob:
            state = np.random.get_state()
            moving = self._random_elastic_deformation(moving)
            np.random.set_state(state)
            fixed = self._random_elastic_deformation(fixed)

        # Random left-right flip (anatomically valid for chest CT)
        if np.random.random() < self.flip_prob:
            moving = np.flip(moving, axis=2).copy()
            fixed = np.flip(fixed, axis=2).copy()

        # --- Intensity augmentation (independent for each) ---
        if np.random.random() < self.intensity_prob:
            moving = self._random_intensity_augmentation(moving)
            fixed = self._random_intensity_augmentation(fixed)

        return moving, fixed
