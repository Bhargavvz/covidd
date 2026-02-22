"""
Logging Utilities
TensorBoard and Weights & Biases integration for training monitoring.
"""

import logging
from pathlib import Path
from typing import Optional, Dict

import numpy as np

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Unified logging for TensorBoard and W&B."""

    def __init__(
        self,
        log_dir: str = "./logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "covid-lung-recovery",
        config: Optional[dict] = None,
    ):
        self.step = 0
        self.tb_writer = None
        self.wandb_run = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                Path(log_dir).mkdir(parents=True, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=log_dir)
                logger.info(f"TensorBoard logging to {log_dir}")
            except ImportError:
                logger.warning("TensorBoard not available")

        if use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=wandb_project,
                    config=config,
                )
                logger.info(f"W&B logging to {wandb_project}")
            except ImportError:
                logger.warning("W&B not available")

    def log_scalars(self, scalars: Dict[str, float], step: Optional[int] = None):
        """Log scalar values."""
        step = step or self.step

        if self.tb_writer:
            for name, value in scalars.items():
                self.tb_writer.add_scalar(name, value, step)

        if self.wandb_run:
            import wandb

            wandb.log(scalars, step=step)

    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        step: Optional[int] = None,
    ):
        """Log an image (2D array or HWC)."""
        step = step or self.step

        if self.tb_writer:
            if image.ndim == 2:
                image = image[np.newaxis, :, :]  # Add channel dim
            elif image.ndim == 3 and image.shape[2] in [1, 3, 4]:
                image = image.transpose(2, 0, 1)  # HWC -> CHW
            self.tb_writer.add_image(tag, image, step)

    def log_histogram(
        self,
        tag: str,
        values: np.ndarray,
        step: Optional[int] = None,
    ):
        """Log a histogram."""
        step = step or self.step

        if self.tb_writer:
            self.tb_writer.add_histogram(tag, values, step)

    def close(self):
        """Close all loggers."""
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            import wandb

            wandb.finish()
