"""
CT Volume Preprocessing Pipeline
Handles intensity windowing, resampling, normalization, and cropping
for chest CT volumes used in deformable registration.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

logger = logging.getLogger(__name__)


class CTPreprocessor:
    """Preprocessing pipeline for chest CT volumes.

    Steps:
        1. Load NIfTI/MHA volume
        2. Apply lung CT intensity windowing
        3. Resample to isotropic voxel spacing
        4. Normalize intensity to [0, 1]
        5. Pad/crop to fixed volume size
    """

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
        target_size: Tuple[int, int, int] = (192, 192, 192),
        window_width: float = 1500.0,
        window_level: float = -600.0,
        normalize_range: Tuple[float, float] = (0.0, 1.0),
        interpolation: str = "linear",
    ):
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.window_width = window_width
        self.window_level = window_level
        self.normalize_range = normalize_range
        self.interpolation = interpolation

    def load_volume(self, path: str) -> "sitk.Image":
        """Load a medical image from NIfTI, MHA, or DICOM."""
        if sitk is None:
            raise RuntimeError("SimpleITK required")

        path = str(path)
        if Path(path).is_dir():
            # DICOM series
            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(path)
            reader.SetFileNames(dicom_files)
            return reader.Execute()
        else:
            return sitk.ReadImage(path)

    def apply_windowing(self, image: "sitk.Image") -> "sitk.Image":
        """Apply lung CT window (W:1500, L:-600).

        Maps HU values to the display window:
            lower = level - width/2
            upper = level + width/2
        """
        lower = self.window_level - self.window_width / 2.0
        upper = self.window_level + self.window_width / 2.0

        image = sitk.Clamp(image, sitk.sitkFloat32, lower, upper)
        return image

    def resample_to_isotropic(
        self, image: "sitk.Image", is_label: bool = False
    ) -> "sitk.Image":
        """Resample volume to isotropic spacing."""
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        # Compute new size based on target spacing
        new_size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(original_size, original_spacing, self.target_spacing)
        ]

        interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear

        resampled = sitk.Resample(
            image,
            new_size,
            sitk.Transform(),
            interpolator,
            image.GetOrigin(),
            self.target_spacing,
            image.GetDirection(),
            0.0,
            image.GetPixelID(),
        )
        return resampled

    def normalize(self, array: np.ndarray) -> np.ndarray:
        """Normalize intensity to target range [min_val, max_val]."""
        min_val, max_val = self.normalize_range
        arr_min = array.min()
        arr_max = array.max()

        if arr_max - arr_min < 1e-8:
            return np.full_like(array, min_val)

        normalized = (array - arr_min) / (arr_max - arr_min)
        normalized = normalized * (max_val - min_val) + min_val
        return normalized.astype(np.float32)

    def pad_or_crop(self, array: np.ndarray) -> np.ndarray:
        """Pad or center-crop array to target_size."""
        target = self.target_size
        result = np.zeros(target, dtype=array.dtype)

        # Compute crop/pad for each dimension
        slices_src = []
        slices_dst = []
        for i in range(3):
            src_size = array.shape[i]
            tgt_size = target[i]

            if src_size > tgt_size:
                # Crop (center crop)
                start = (src_size - tgt_size) // 2
                slices_src.append(slice(start, start + tgt_size))
                slices_dst.append(slice(0, tgt_size))
            elif src_size < tgt_size:
                # Pad (center pad)
                start = (tgt_size - src_size) // 2
                slices_src.append(slice(0, src_size))
                slices_dst.append(slice(start, start + src_size))
            else:
                slices_src.append(slice(0, src_size))
                slices_dst.append(slice(0, tgt_size))

        result[tuple(slices_dst)] = array[tuple(slices_src)]
        return result

    def process(
        self,
        image_path: str,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """Full preprocessing pipeline.

        Args:
            image_path: Path to input volume
            save_path: Optional path to save processed volume

        Returns:
            Preprocessed numpy array of shape target_size
        """
        # 1. Load
        image = self.load_volume(image_path)
        logger.debug(
            f"Loaded: size={image.GetSize()}, spacing={image.GetSpacing()}"
        )

        # 2. Window
        image = self.apply_windowing(image)

        # 3. Resample
        image = self.resample_to_isotropic(image)
        logger.debug(
            f"Resampled: size={image.GetSize()}, spacing={image.GetSpacing()}"
        )

        # 4. Convert to numpy
        array = sitk.GetArrayFromImage(image).astype(np.float32)

        # 5. Normalize
        array = self.normalize(array)

        # 6. Pad/Crop
        array = self.pad_or_crop(array)
        logger.debug(f"Final shape: {array.shape}")

        # Optionally save
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            processed_img = sitk.GetImageFromArray(array)
            processed_img.SetSpacing(self.target_spacing)
            sitk.WriteImage(processed_img, str(save_path))
            logger.info(f"Saved processed volume to {save_path}")

        return array

    def process_pair(
        self,
        moving_path: str,
        fixed_path: str,
        moving_seg_path: Optional[str] = None,
        fixed_seg_path: Optional[str] = None,
    ) -> dict:
        """Process a registration pair (moving + fixed volumes).

        Returns dict with 'moving', 'fixed', and optionally 'moving_seg', 'fixed_seg'.
        """
        result = {
            "moving": self.process(moving_path),
            "fixed": self.process(fixed_path),
        }

        if moving_seg_path:
            seg_img = self.load_volume(moving_seg_path)
            seg_img = self.resample_to_isotropic(seg_img, is_label=True)
            seg_arr = sitk.GetArrayFromImage(seg_img).astype(np.float32)
            result["moving_seg"] = self.pad_or_crop(seg_arr)

        if fixed_seg_path:
            seg_img = self.load_volume(fixed_seg_path)
            seg_img = self.resample_to_isotropic(seg_img, is_label=True)
            seg_arr = sitk.GetArrayFromImage(seg_img).astype(np.float32)
            result["fixed_seg"] = self.pad_or_crop(seg_arr)

        return result


def batch_preprocess(
    input_dir: Path,
    output_dir: Path,
    config: Optional[dict] = None,
) -> List[Path]:
    """Batch preprocess all volumes in a directory.

    Args:
        input_dir: Directory containing patient subdirectories
        output_dir: Directory to save processed volumes
        config: Optional preprocessing config overrides

    Returns:
        List of paths to processed volumes
    """
    config = config or {}
    preprocessor = CTPreprocessor(
        target_spacing=tuple(config.get("voxel_spacing", [1.5, 1.5, 1.5])),
        target_size=tuple(config.get("volume_size", [192, 192, 192])),
        window_width=config.get("intensity_window", {}).get("width", 1500),
        window_level=config.get("intensity_window", {}).get("level", -600),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_paths = []

    # Find all NIfTI volumes
    volume_paths = sorted(input_dir.rglob("*.nii.gz"))
    logger.info(f"Found {len(volume_paths)} volumes to preprocess")

    for i, vol_path in enumerate(volume_paths):
        # Preserve relative directory structure
        rel_path = vol_path.relative_to(input_dir)
        save_path = output_dir / rel_path

        logger.info(f"[{i+1}/{len(volume_paths)}] Processing {rel_path}")
        try:
            preprocessor.process(str(vol_path), save_path=str(save_path))
            processed_paths.append(save_path)
        except Exception as e:
            logger.error(f"  Failed: {e}")

    logger.info(f"Preprocessed {len(processed_paths)}/{len(volume_paths)} volumes")
    return processed_paths
