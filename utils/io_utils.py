"""
File I/O Utilities for medical image volumes.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np

logger = logging.getLogger(__name__)


def load_volume_sitk(path: str) -> Tuple[np.ndarray, dict]:
    """Load a volume using SimpleITK and return array + metadata.

    Returns:
        (array, metadata_dict) where metadata contains spacing, origin, direction
    """
    import SimpleITK as sitk

    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    metadata = {
        "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
        "size": image.GetSize(),
    }
    return array, metadata


def save_volume_sitk(
    array: np.ndarray,
    path: str,
    spacing: tuple = (1.0, 1.0, 1.0),
    origin: tuple = (0.0, 0.0, 0.0),
    direction: Optional[tuple] = None,
):
    """Save a numpy array as a NIfTI volume."""
    import SimpleITK as sitk

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    if direction:
        image.SetDirection(direction)
    sitk.WriteImage(image, str(path))


def save_displacement_field(
    displacement: np.ndarray,
    path: str,
    spacing: tuple = (1.0, 1.0, 1.0),
):
    """Save displacement field as a vector NIfTI image.

    Args:
        displacement: (3, D, H, W) displacement field
        path: output path
        spacing: voxel spacing
    """
    import SimpleITK as sitk

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Transpose to (D, H, W, 3) for SimpleITK vector format
    disp_dhw3 = displacement.transpose(1, 2, 3, 0).astype(np.float32)
    image = sitk.GetImageFromArray(disp_dhw3, isVector=True)
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path))


def load_displacement_field(path: str) -> np.ndarray:
    """Load displacement field from NIfTI.

    Returns:
        (3, D, H, W) displacement field
    """
    import SimpleITK as sitk

    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)  # (D, H, W, 3)
    return array.transpose(3, 0, 1, 2).astype(np.float32)


def save_json(data: dict, path: str):
    """Save dictionary as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)
