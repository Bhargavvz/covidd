"""
Main Training Script
Entry point for training deformable registration models.
Supports config overrides, resumption, and H200 optimization.

Usage:
    # Default training:
    python training/train.py --config configs/default.yaml

    # H200 optimized:
    python training/train.py --config configs/default.yaml \\
                              --override configs/h200_optimized.yaml

    # Quick smoke test:
    python training/train.py --config configs/default.yaml \\
                              --epochs 5 --smoke-test

    # Resume from checkpoint:
    python training/train.py --config configs/default.yaml \\
                              --resume outputs/checkpoints/checkpoint_epoch_0050.pth
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from copy import deepcopy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from models.voxelmorph import build_model
from models.losses import RegistrationLoss
from data.dataset import LongitudinalCTPairDataset, create_dataloaders
from data.download_datasets import create_demo_data
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def load_config(config_path: str, override_path: str = None) -> dict:
    """Load YAML config with optional override merging."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if override_path:
        with open(override_path) as f:
            override = yaml.safe_load(f)
        config = deep_merge(config, override)

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts, with override taking precedence."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # Allow cuDNN benchmark
    torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(
        description="Train deformable registration model for COVID-19 lung recovery"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to base config file",
    )
    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Path to override config (e.g., h200_optimized.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick test with demo data")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # ---- Load Config ----
    config = load_config(args.config, args.override)

    # CLI overrides
    training_config = config.get("training", {})
    if args.epochs is not None:
        training_config["epochs"] = args.epochs
    if args.batch_size is not None:
        training_config["batch_size"] = args.batch_size
    if args.lr is not None:
        training_config["lr"] = args.lr
    if args.no_compile:
        training_config["compile"] = False
    config["training"] = training_config

    data_config = config.get("data", {})
    if args.data_dir:
        data_config["data_dir"] = args.data_dir
    config["data"] = data_config

    output_dir = args.output_dir or config.get("project", {}).get("output_dir", "./outputs")
    log_dir = config.get("project", {}).get("log_dir", "./logs")
    seed = config.get("project", {}).get("seed", 42)

    # ---- Seed ----
    set_seed(seed)

    # ---- Device ----
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    if device == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # ---- Data ----
    data_dir = data_config.get("data_dir", "./datasets")

    if args.smoke_test:
        logger.info("=" * 50)
        logger.info("SMOKE TEST MODE: Using synthetic demo data")
        logger.info("=" * 50)
        data_dir = "./datasets/demo/processed"
        demo_path = Path(data_dir)
        if not demo_path.exists() or not list(demo_path.rglob("*.nii.gz")):
            logger.info("Generating demo data...")
            create_demo_data(demo_path, num_volumes=10, size=64)

        # Override for small volumes
        data_config["volume_size"] = [64, 64, 64]
        training_config["epochs"] = min(training_config.get("epochs", 5), 5)
        training_config["batch_size"] = 2
        training_config["compile"] = False  # Faster startup for smoke test

    logger.info(f"Data directory: {data_dir}")

    # Create dataloaders
    data_config["batch_size"] = training_config.get("batch_size", 4)
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        config=data_config,
        num_workers=data_config.get("num_workers", 8),
        pin_memory=data_config.get("pin_memory", True),
        prefetch_factor=data_config.get("prefetch_factor", 2),
    )

    logger.info(f"Train: {len(train_loader.dataset)} pairs, {len(train_loader)} batches")
    logger.info(f"Val:   {len(val_loader.dataset)} pairs, {len(val_loader)} batches")
    logger.info(f"Test:  {len(test_loader.dataset)} pairs, {len(test_loader)} batches")

    # ---- Model ----
    model_config = config.get("model", {})
    vol_size = tuple(data_config.get("volume_size", [192, 192, 192]))
    model_config["vol_size"] = vol_size

    model = build_model(model_config)

    # ---- Trainer ----
    # Merge all config sections for the trainer
    trainer_config = {
        **training_config,
        "loss": config.get("loss", {}),
        "scheduler": training_config.get("scheduler", {}),
        "checkpoint": training_config.get("checkpoint", {}),
        "early_stopping": training_config.get("early_stopping", {}),
        "logging": config.get("logging", {}),
        "log_dir": log_dir,
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        output_dir=output_dir,
        device=device,
    )

    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ---- Train ----
    history = trainer.train()

    # ---- Save training history ----
    import json

    history_path = Path(output_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
