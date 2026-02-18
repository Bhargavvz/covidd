"""
PyTorch Dataset for Longitudinal CT Registration Pairs
Supports real longitudinal pairs and synthetic pair generation.
Optimized for H200 GPU data feeding (pin_memory, prefetching).
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data.preprocessing import CTPreprocessor
from data.lung_segmentation import LungSegmenter
from data.augmentation import RegistrationAugmentor

logger = logging.getLogger(__name__)


class LongitudinalCTPairDataset(Dataset):
    """Dataset yielding (moving, fixed) CT volume pairs for registration.

    Supports:
        - Real longitudinal pairs (from manifest.json)
        - Synthetic pairs (from synthetic_pairs.json)
        - On-the-fly pair generation from single timepoint data

    Each item returns:
        moving:  (1, D, H, W) tensor - moving/source volume
        fixed:   (1, D, H, W) tensor - fixed/target volume
        meta:    dict with metadata (patient_id, pair_type, etc.)
        [optional] displacement_gt: (3, D, H, W) ground truth deformation
        [optional] moving_seg: (1, D, H, W) moving lung segmentation
        [optional] fixed_seg:  (1, D, H, W) fixed lung segmentation
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        volume_size: Tuple[int, int, int] = (192, 192, 192),
        voxel_spacing: float = 1.5,
        use_augmentation: bool = True,
        use_lung_mask: bool = True,
        cache_volumes: bool = False,
        synthetic_pairs_path: Optional[str] = None,
        preprocessor: Optional[CTPreprocessor] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.volume_size = volume_size
        self.use_augmentation = use_augmentation and split == "train"
        self.use_lung_mask = use_lung_mask
        self.cache_volumes = cache_volumes

        # Initialize preprocessor
        self.preprocessor = preprocessor or CTPreprocessor(
            target_spacing=(voxel_spacing,) * 3,
            target_size=volume_size,
        )

        # Initialize lung segmenter
        self.segmenter = LungSegmenter() if use_lung_mask else None

        # Initialize augmentor
        self.augmentor = RegistrationAugmentor() if self.use_augmentation else None

        # Load pairs
        self.pairs = self._load_pairs(synthetic_pairs_path)

        # Volume cache
        self._cache = {} if cache_volumes else None

        logger.info(
            f"Dataset [{split}]: {len(self.pairs)} pairs, "
            f"augmentation={'on' if self.use_augmentation else 'off'}, "
            f"lung_mask={'on' if self.use_lung_mask else 'off'}"
        )

    def _load_pairs(self, synthetic_path: Optional[str] = None) -> List[Dict]:
        """Load pair definitions from manifest files."""
        pairs = []

        # 1. Load synthetic pairs
        if synthetic_path:
            syn_path = Path(synthetic_path)
        else:
            syn_path = self.data_dir / "synthetic_pairs" / "synthetic_pairs.json"

        if syn_path.exists():
            with open(syn_path) as f:
                synthetic_pairs = json.load(f)
            for p in synthetic_pairs:
                pairs.append({
                    "moving": p["baseline"],
                    "fixed": p["followup"],
                    "displacement_gt": p.get("displacement_gt"),
                    "pair_type": "synthetic",
                    "pair_id": f"syn_{p['pair_id']}",
                })
            logger.info(f"Loaded {len(synthetic_pairs)} synthetic pairs")

        # 2. Load real longitudinal pairs from manifest
        manifest_path = self.data_dir / "processed" / "manifest.json"
        if not manifest_path.exists():
            # Try dataset-specific manifests
            for ds_dir in self.data_dir.iterdir():
                mp = ds_dir / "processed" / "manifest.json"
                if mp.exists():
                    manifest_path = mp
                    break

        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

            for patient_id, info in manifest.get("patients", {}).items():
                timepoints = info.get("timepoints", [])
                # Create pairs from consecutive timepoints
                for i in range(len(timepoints) - 1):
                    pairs.append({
                        "moving": timepoints[i],
                        "fixed": timepoints[i + 1],
                        "displacement_gt": None,
                        "pair_type": "real_longitudinal",
                        "pair_id": f"{patient_id}_t{i}_t{i+1}",
                    })

        # 3. If no pairs found, create self-pairs from single volumes (for testing)
        if not pairs:
            logger.warning("No pair manifests found. Creating self-registration pairs from volumes.")
            volumes = sorted(self.data_dir.rglob("*.nii.gz"))
            for i, vol in enumerate(volumes):
                for j, vol2 in enumerate(volumes):
                    if i != j:
                        pairs.append({
                            "moving": str(vol),
                            "fixed": str(vol2),
                            "displacement_gt": None,
                            "pair_type": "cross_subject",
                            "pair_id": f"cross_{i}_{j}",
                        })

        # Split data
        np.random.seed(42)
        indices = np.random.permutation(len(pairs))
        n_train = int(0.8 * len(pairs))
        n_val = int(0.1 * len(pairs))

        if self.split == "train":
            selected = indices[:n_train]
        elif self.split == "val":
            selected = indices[n_train : n_train + n_val]
        else:  # test
            selected = indices[n_train + n_val :]

        return [pairs[i] for i in selected]

    def _load_volume(self, path: str) -> np.ndarray:
        """Load and preprocess a volume, with optional caching."""
        if self._cache is not None and path in self._cache:
            return self._cache[path]

        volume = self.preprocessor.process(path)

        if self._cache is not None:
            self._cache[path] = volume

        return volume

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]

        # Load volumes
        moving = self._load_volume(pair["moving"])
        fixed = self._load_volume(pair["fixed"])

        # Apply augmentation
        if self.augmentor is not None:
            moving, fixed = self.augmentor(moving, fixed)

        # Compute lung masks
        moving_seg = None
        fixed_seg = None
        if self.segmenter is not None:
            moving_seg = self.segmenter.segment(moving)
            fixed_seg = self.segmenter.segment(fixed)

        # Convert to tensors: add channel dimension (1, D, H, W)
        moving_tensor = torch.from_numpy(moving).unsqueeze(0).float()
        fixed_tensor = torch.from_numpy(fixed).unsqueeze(0).float()

        result = {
            "moving": moving_tensor,
            "fixed": fixed_tensor,
            "pair_id": pair["pair_id"],
            "pair_type": pair["pair_type"],
        }

        # Ground truth displacement (if available)
        if pair.get("displacement_gt") and Path(pair["displacement_gt"]).exists():
            try:
                import SimpleITK as sitk

                disp_img = sitk.ReadImage(pair["displacement_gt"])
                disp_arr = sitk.GetArrayFromImage(disp_img)  # (D, H, W, 3)
                if disp_arr.ndim == 4 and disp_arr.shape[-1] == 3:
                    disp_arr = disp_arr.transpose(3, 0, 1, 2)  # (3, D, H, W)
                disp_tensor = torch.from_numpy(
                    self.preprocessor.pad_or_crop(
                        disp_arr.transpose(1, 2, 3, 0)  # Temporarily to (D,H,W,3)
                    ).transpose(3, 0, 1, 2)  # Back to (3,D,H,W) -- simplified below
                ).float()
                # Simpler approach: just pad/crop each channel
                disp_channels = []
                for c in range(3):
                    disp_channels.append(
                        self.preprocessor.pad_or_crop(disp_arr[c])
                    )
                disp_tensor = torch.from_numpy(
                    np.stack(disp_channels, axis=0)
                ).float()
                result["displacement_gt"] = disp_tensor
            except Exception as e:
                logger.debug(f"Could not load displacement GT: {e}")

        # Segmentation masks
        if moving_seg is not None:
            result["moving_seg"] = torch.from_numpy(moving_seg).unsqueeze(0).float()
        if fixed_seg is not None:
            result["fixed_seg"] = torch.from_numpy(fixed_seg).unsqueeze(0).float()

        return result


def create_dataloaders(
    data_dir: str,
    config: dict,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders optimized for H200.

    Args:
        data_dir: Path to dataset directory
        config: Data configuration dict
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch

    Returns:
        (train_loader, val_loader, test_loader)
    """
    volume_size = tuple(config.get("volume_size", [192, 192, 192]))
    voxel_spacing = config.get("voxel_spacing", 1.5)
    batch_size = config.get("batch_size", 4)

    loaders = []
    for split in ["train", "val", "test"]:
        dataset = LongitudinalCTPairDataset(
            data_dir=data_dir,
            split=split,
            volume_size=volume_size,
            voxel_spacing=voxel_spacing,
            use_augmentation=(split == "train"),
            use_lung_mask=True,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size if split == "train" else 1,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            drop_last=(split == "train"),
            persistent_workers=(num_workers > 0),
        )
        loaders.append(loader)

    return tuple(loaders)
