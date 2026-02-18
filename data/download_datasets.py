"""
Dataset Download & Preparation Scripts
Downloads and organizes COVID-19 CT datasets for longitudinal analysis.

Supported datasets:
  - STOIC (Study of Thoracic CT in COVID-19)
  - COVID-CT+ (NIH)
  - BIMCV COVID-19+
"""

import os
import sys
import json
import shutil
import logging
import argparse
import hashlib
from pathlib import Path
from typing import Optional, Dict, List
from urllib.request import urlretrieve
from urllib.error import URLError

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None
    print("WARNING: SimpleITK not installed. Install with: pip install SimpleITK")

try:
    import pydicom
except ImportError:
    pydicom = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Registry
# =============================================================================
DATASET_REGISTRY = {
    "stoic": {
        "name": "STOIC COVID-19",
        "description": "Study of Thoracic CT in COVID-19 - 2000 thoracic CT scans",
        "url": "https://registry.opendata.aws/stoic2021-training/",
        "s3_bucket": "s3://stoic2021-training/",
        "format": "mha",
        "license": "CC BY-NC 4.0",
    },
    "covid_ct_plus": {
        "name": "COVID-CT+",
        "description": "400K+ CT images from 1300+ patients",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8411519/",
        "format": "dicom",
        "license": "Research Only",
    },
    "bimcv": {
        "name": "BIMCV COVID-19+",
        "description": "Large annotated COVID-19 CT dataset with radiological findings",
        "url": "https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/",
        "format": "nifti",
        "license": "CC BY-NC 4.0",
    },
}


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress reporting."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {desc or url} -> {dest}")

        def _progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(100, block_num * block_size * 100 / total_size)
                print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

        urlretrieve(url, str(dest), reporthook=_progress)
        print()  # newline after progress
        return True
    except (URLError, OSError) as e:
        logger.error(f"Download failed: {e}")
        return False


def convert_mha_to_nifti(mha_path: Path, output_dir: Path) -> Optional[Path]:
    """Convert .mha file to NIfTI format."""
    if sitk is None:
        raise RuntimeError("SimpleITK required for MHA conversion")

    output_dir.mkdir(parents=True, exist_ok=True)
    nifti_path = output_dir / (mha_path.stem + ".nii.gz")

    if nifti_path.exists():
        logger.info(f"  Already converted: {nifti_path.name}")
        return nifti_path

    try:
        img = sitk.ReadImage(str(mha_path))
        sitk.WriteImage(img, str(nifti_path))
        logger.info(f"  Converted: {mha_path.name} -> {nifti_path.name}")
        return nifti_path
    except Exception as e:
        logger.error(f"  Conversion failed for {mha_path.name}: {e}")
        return None


def convert_dicom_series_to_nifti(dicom_dir: Path, output_path: Path) -> Optional[Path]:
    """Convert a DICOM series directory to a single NIfTI volume."""
    if sitk is None:
        raise RuntimeError("SimpleITK required for DICOM conversion")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"  Already converted: {output_path.name}")
        return output_path

    try:
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(str(dicom_dir))
        if not dicom_files:
            logger.warning(f"  No DICOM files in {dicom_dir}")
            return None

        reader.SetFileNames(dicom_files)
        image = reader.Execute()
        sitk.WriteImage(image, str(output_path))
        logger.info(f"  Converted DICOM series: {dicom_dir.name} -> {output_path.name}")
        return output_path
    except Exception as e:
        logger.error(f"  DICOM conversion failed for {dicom_dir}: {e}")
        return None


def organize_dataset(raw_dir: Path, processed_dir: Path, dataset_name: str) -> Dict:
    """Organize raw dataset into standardized structure.

    Output structure:
        processed_dir/
            patient_001/
                timepoint_0/volume.nii.gz
                timepoint_1/volume.nii.gz  (if longitudinal)
            patient_002/
                timepoint_0/volume.nii.gz
            ...
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": dataset_name, "patients": {}}
    patient_id = 0

    # Process based on file format
    for item in sorted(raw_dir.rglob("*")):
        if item.suffix == ".mha":
            patient_name = f"patient_{patient_id:04d}"
            patient_dir = processed_dir / patient_name / "timepoint_0"
            patient_dir.mkdir(parents=True, exist_ok=True)

            nifti_path = convert_mha_to_nifti(item, patient_dir)
            if nifti_path:
                manifest["patients"][patient_name] = {
                    "timepoints": [str(nifti_path)],
                    "source_file": item.name,
                }
                patient_id += 1

        elif item.suffix in (".nii", ".nii.gz"):
            patient_name = f"patient_{patient_id:04d}"
            patient_dir = processed_dir / patient_name / "timepoint_0"
            patient_dir.mkdir(parents=True, exist_ok=True)

            dest = patient_dir / "volume.nii.gz"
            if not dest.exists():
                shutil.copy2(item, dest)

            manifest["patients"][patient_name] = {
                "timepoints": [str(dest)],
                "source_file": item.name,
            }
            patient_id += 1

    # Save manifest
    manifest_path = processed_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Organized {patient_id} patients -> {processed_dir}")
    logger.info(f"Manifest saved to {manifest_path}")
    return manifest


def generate_synthetic_longitudinal_data(
    processed_dir: Path,
    output_dir: Path,
    num_pairs_per_scan: int = 3,
    deformation_scale: tuple = (0.05, 0.15),
    density_change_range: tuple = (0.1, 0.4),
    seed: int = 42,
) -> List[Dict]:
    """Generate synthetic longitudinal pairs from existing CT scans.

    Simulates COVID-19 recovery by:
    1. Applying controlled deformations (simulating tissue changes)
    2. Modifying intensity in lung regions (simulating density recovery)
    3. Creating ground-truth displacement fields

    Returns list of generated pairs metadata.
    """
    if sitk is None:
        raise RuntimeError("SimpleITK required for synthetic data generation")

    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    pair_id = 0

    # Find all existing volumes
    volumes = sorted(processed_dir.rglob("volume.nii.gz"))
    if not volumes:
        volumes = sorted(processed_dir.rglob("*.nii.gz"))

    logger.info(f"Found {len(volumes)} source volumes for synthetic generation")

    for vol_path in volumes:
        try:
            img = sitk.ReadImage(str(vol_path))
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            spacing = img.GetSpacing()
            origin = img.GetOrigin()
            direction = img.GetDirection()
        except Exception as e:
            logger.warning(f"  Skipping {vol_path.name}: {e}")
            continue

        for pair_idx in range(num_pairs_per_scan):
            pair_dir = output_dir / f"pair_{pair_id:05d}"
            pair_dir.mkdir(parents=True, exist_ok=True)

            # --- Generate random smooth deformation field ---
            shape = arr.shape
            scale = np.random.uniform(*deformation_scale)

            # Create low-frequency displacement field (smooth)
            small_shape = tuple(max(1, s // 16) for s in shape)
            small_disp = np.random.randn(3, *small_shape).astype(np.float32) * scale

            # Upsample to full resolution for smooth deformation
            from scipy.ndimage import zoom

            zoom_factors = [s / ss for s, ss in zip(shape, small_shape)]
            disp_field = np.stack(
                [zoom(small_disp[i], zoom_factors, order=3) for i in range(3)]
            )

            # Scale by voxel count for meaningful deformation
            for i in range(3):
                disp_field[i] *= shape[i] * 0.02

            # --- Apply deformation to create "follow-up" scan ---
            coords = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float32)
            warped_coords = coords + disp_field

            from scipy.ndimage import map_coordinates

            warped_arr = map_coordinates(
                arr, warped_coords, order=1, mode="constant", cval=arr.min()
            )

            # --- Simulate density recovery (reduce opacity in lung regions) ---
            density_change = np.random.uniform(*density_change_range)
            lung_mask = (arr > -900) & (arr < -200)  # Rough lung HU range
            recovery_factor = 1.0 - density_change * lung_mask.astype(np.float32)
            warped_arr = warped_arr * recovery_factor

            # --- Save baseline (original), follow-up (warped), and ground truth ---
            baseline_path = pair_dir / "baseline.nii.gz"
            followup_path = pair_dir / "followup.nii.gz"
            disp_path = pair_dir / "displacement.nii.gz"

            # Save baseline
            baseline_img = sitk.GetImageFromArray(arr)
            baseline_img.SetSpacing(spacing)
            baseline_img.SetOrigin(origin)
            baseline_img.SetDirection(direction)
            sitk.WriteImage(baseline_img, str(baseline_path))

            # Save follow-up
            followup_img = sitk.GetImageFromArray(warped_arr.astype(np.float32))
            followup_img.SetSpacing(spacing)
            followup_img.SetOrigin(origin)
            followup_img.SetDirection(direction)
            sitk.WriteImage(followup_img, str(followup_path))

            # Save displacement field as multi-channel image
            disp_img = sitk.GetImageFromArray(
                disp_field.transpose(1, 2, 3, 0).astype(np.float32),
                isVector=True,
            )
            disp_img.SetSpacing(spacing)
            disp_img.SetOrigin(origin)
            sitk.WriteImage(disp_img, str(disp_path))

            pairs.append({
                "pair_id": pair_id,
                "baseline": str(baseline_path),
                "followup": str(followup_path),
                "displacement_gt": str(disp_path),
                "source_volume": str(vol_path),
                "deformation_scale": float(scale),
                "density_change": float(density_change),
            })

            pair_id += 1

    # Save pairs manifest
    manifest_path = output_dir / "synthetic_pairs.json"
    with open(manifest_path, "w") as f:
        json.dump(pairs, f, indent=2)

    logger.info(f"Generated {len(pairs)} synthetic longitudinal pairs -> {output_dir}")
    return pairs


def create_demo_data(output_dir: Path, num_volumes: int = 10, size: int = 64) -> Path:
    """Create small synthetic CT volumes for testing/demo purposes.

    Generates fake 3D lung-like volumes with random lesions.
    """
    if sitk is None:
        raise RuntimeError("SimpleITK required")

    np.random.seed(42)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating {num_volumes} demo volumes of size {size}^3...")

    for i in range(num_volumes):
        patient_dir = output_dir / f"patient_{i:04d}" / "timepoint_0"
        patient_dir.mkdir(parents=True, exist_ok=True)

        # Create a synthetic lung-like volume
        vol = np.ones((size, size, size), dtype=np.float32) * (-1000)  # Air HU

        # Add body outline (ellipsoid)
        center = np.array([size // 2, size // 2, size // 2])
        coords = np.mgrid[0:size, 0:size, 0:size].astype(np.float32)
        dist = np.sqrt(
            ((coords[0] - center[0]) / (size * 0.4)) ** 2
            + ((coords[1] - center[1]) / (size * 0.35)) ** 2
            + ((coords[2] - center[2]) / (size * 0.3)) ** 2
        )
        body_mask = dist < 1.0
        vol[body_mask] = np.random.uniform(-800, -500, body_mask.sum())

        # Add some "lesions" (GGO-like regions)
        num_lesions = np.random.randint(2, 6)
        for _ in range(num_lesions):
            lx = np.random.randint(size // 4, 3 * size // 4)
            ly = np.random.randint(size // 4, 3 * size // 4)
            lz = np.random.randint(size // 4, 3 * size // 4)
            lr = np.random.randint(3, size // 8)
            lesion_dist = np.sqrt(
                (coords[0] - lx) ** 2 + (coords[1] - ly) ** 2 + (coords[2] - lz) ** 2
            )
            lesion_mask = (lesion_dist < lr) & body_mask
            vol[lesion_mask] = np.random.uniform(-300, 0, lesion_mask.sum())

        # Save
        img = sitk.GetImageFromArray(vol)
        img.SetSpacing([1.5, 1.5, 1.5])
        sitk.WriteImage(img, str(patient_dir / "volume.nii.gz"))

    logger.info(f"Demo data created at {output_dir}")
    return output_dir


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare COVID-19 CT datasets"
    )
    parser.add_argument(
        "--action",
        choices=["list", "download", "organize", "synthetic", "demo"],
        default="list",
        help="Action to perform",
    )
    parser.add_argument("--dataset", choices=list(DATASET_REGISTRY.keys()), help="Dataset name")
    parser.add_argument("--raw-dir", type=Path, help="Path to raw dataset files")
    parser.add_argument("--output-dir", type=Path, default=Path("./datasets"), help="Output directory")
    parser.add_argument("--num-pairs", type=int, default=3, help="Synthetic pairs per scan")
    parser.add_argument("--demo-size", type=int, default=64, help="Demo volume size")
    parser.add_argument("--num-demo", type=int, default=10, help="Number of demo volumes")

    args = parser.parse_args()

    if args.action == "list":
        print("\nAvailable Datasets:")
        print("=" * 70)
        for key, info in DATASET_REGISTRY.items():
            print(f"\n  {key}:")
            print(f"    Name:        {info['name']}")
            print(f"    Description: {info['description']}")
            print(f"    URL:         {info['url']}")
            print(f"    Format:      {info['format']}")
            print(f"    License:     {info['license']}")
        print("\n" + "=" * 70)
        print("\nTo download, visit the URLs above and place raw files in a directory,")
        print("then run: python download_datasets.py --action organize --dataset <name> --raw-dir <path>")

    elif args.action == "organize":
        if not args.dataset or not args.raw_dir:
            parser.error("--dataset and --raw-dir required for organize action")
        processed_dir = args.output_dir / args.dataset / "processed"
        organize_dataset(args.raw_dir, processed_dir, args.dataset)

    elif args.action == "synthetic":
        source_dir = args.raw_dir or (args.output_dir / "demo" / "processed")
        if not source_dir.exists():
            logger.error(f"Source directory not found: {source_dir}")
            logger.error("Run demo first: python data/download_datasets.py --action demo")
            logger.error("Or specify source with: --raw-dir /path/to/volumes")
            return
        synthetic_dir = args.output_dir / "synthetic_pairs"
        generate_synthetic_longitudinal_data(
            source_dir, synthetic_dir, num_pairs_per_scan=args.num_pairs
        )

    elif args.action == "demo":
        demo_dir = args.output_dir / "demo" / "processed"
        create_demo_data(demo_dir, num_volumes=args.num_demo, size=args.demo_size)


if __name__ == "__main__":
    main()
