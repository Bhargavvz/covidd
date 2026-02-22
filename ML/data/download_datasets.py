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

    for vol_idx, vol_path in enumerate(volumes):
        logger.info(f"  [{vol_idx+1}/{len(volumes)}] Processing {vol_path.name}...")
        try:
            img = sitk.ReadImage(str(vol_path))
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            spacing = img.GetSpacing()
            origin = img.GetOrigin()
            direction = img.GetDirection()
        except Exception as e:
            logger.warning(f"    Skipping {vol_path.name}: {e}")
            continue

        # --- Downsample to 128³ for fast synthetic generation ---
        # Training uses 128³ anyway, so no quality loss
        from scipy.ndimage import zoom as scipy_zoom
        target_size = 128
        zoom_to_128 = [target_size / s for s in arr.shape]
        if any(abs(z - 1.0) > 0.01 for z in zoom_to_128):
            logger.info(f"    Resampling {arr.shape} → ({target_size}³) for fast generation")
            arr = scipy_zoom(arr, zoom_to_128, order=1).astype(np.float32)
            # Update spacing to match new resolution
            spacing = tuple(sp * (orig_s / target_size) for sp, orig_s in zip(spacing, [arr.shape[2], arr.shape[1], arr.shape[0]]))

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
            logger.info(f"    Pair {pair_idx+1}/{num_pairs_per_scan} done (total: {pair_id})")

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
def download_dataset(dataset_name: str, output_dir: Path) -> bool:
    """Download a dataset using the appropriate method.

    Supported:
        - stoic: AWS S3 public bucket (no account required, needs AWS CLI)
        - bimcv: BSC EUDAT mirror (wget)
        - covid_ct_plus: Manual download required (prints instructions)

    Returns True if download was successful.
    """
    import subprocess

    raw_dir = output_dir / dataset_name / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name == "stoic":
        logger.info("=" * 60)
        logger.info("Downloading STOIC COVID-19 dataset from AWS S3...")
        logger.info("Source: s3://stoic2021-training/")
        logger.info(f"Destination: {raw_dir}")
        logger.info("No AWS account required (--no-sign-request)")
        logger.info("=" * 60)

        # Check if AWS CLI is installed
        try:
            subprocess.run(["aws", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("AWS CLI not installed!")
            logger.error("Install with: pip install awscli")
            logger.error("  OR: curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o 'awscliv2.zip' && unzip awscliv2.zip && sudo ./aws/install")
            return False

        cmd = [
            "aws", "s3", "cp",
            "s3://stoic2021-training/",
            str(raw_dir) + "/",
            "--recursive",
            "--no-sign-request",
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)

        if result.returncode == 0:
            # Count downloaded files
            mha_files = list(raw_dir.rglob("*.mha"))
            nifti_files = list(raw_dir.rglob("*.nii.gz"))
            total = len(mha_files) + len(nifti_files)
            logger.info(f"Download complete! {total} files in {raw_dir}")

            # Auto-organize after download
            logger.info("Organizing dataset...")
            processed_dir = output_dir / dataset_name / "processed"
            organize_dataset(raw_dir, processed_dir, dataset_name)
            return True
        else:
            logger.error(f"AWS S3 download failed with exit code {result.returncode}")
            return False

    elif dataset_name == "bimcv":
        logger.info("=" * 60)
        logger.info("Downloading BIMCV COVID-19+ dataset from BSC mirror...")
        logger.info("Source: https://b2drop.bsc.es/index.php/s/BIMCV-COVID19-cIter_1_2_3")
        logger.info(f"Destination: {raw_dir}")
        logger.info("=" * 60)

        # BIMCV BSC mirror - download via wget
        bimcv_url = "https://b2drop.bsc.es/index.php/s/BIMCV-COVID19-cIter_1_2_3/download"

        # Check if wget is available
        try:
            subprocess.run(["wget", "--version"], capture_output=True, check=True)
            use_wget = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            use_wget = False

        zip_path = raw_dir / "bimcv_covid19.zip"

        if use_wget:
            cmd = [
                "wget", "-c",  # -c for resume support
                "-O", str(zip_path),
                bimcv_url,
            ]
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd)
        else:
            # Fallback to curl
            cmd = [
                "curl", "-L", "-C", "-",
                "-o", str(zip_path),
                bimcv_url,
            ]
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd)

        if result.returncode == 0 and zip_path.exists():
            logger.info(f"Download complete: {zip_path}")
            logger.info(f"File size: {zip_path.stat().st_size / 1e9:.2f} GB")

            # Extract
            logger.info("Extracting archive...")
            extract_dir = raw_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)

            try:
                import zipfile
                with zipfile.ZipFile(str(zip_path), 'r') as zf:
                    zf.extractall(str(extract_dir))
                logger.info(f"Extracted to {extract_dir}")
            except Exception as e:
                # May be tar.gz or other format
                logger.info("Trying tar extraction...")
                subprocess.run(["tar", "-xf", str(zip_path), "-C", str(extract_dir)])

            # Auto-organize
            logger.info("Organizing dataset...")
            processed_dir = output_dir / dataset_name / "processed"
            organize_dataset(extract_dir, processed_dir, dataset_name)
            return True
        else:
            logger.error("Download failed!")
            return False

    elif dataset_name == "covid_ct_plus":
        logger.info("=" * 60)
        logger.info("COVID-CT+ requires manual download from NIH.")
        logger.info("Visit: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8411519/")
        logger.info(f"Place downloaded files in: {raw_dir}")
        logger.info("Then run: python data/download_datasets.py --action organize \\")
        logger.info(f"    --dataset covid_ct_plus --raw-dir {raw_dir}")
        logger.info("=" * 60)
        return False

    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        return False


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
        print("\nTo download:")
        print("  python data/download_datasets.py --action download --dataset stoic")
        print("  python data/download_datasets.py --action download --dataset bimcv")

    elif args.action == "download":
        if not args.dataset:
            parser.error("--dataset required for download action")
        download_dataset(args.dataset, args.output_dir)

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
