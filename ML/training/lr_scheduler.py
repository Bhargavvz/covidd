"""
Learning Rate Schedulers
Warmup + Cosine Annealing & OneCycleLR for registration training.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR
from typing import Optional


class WarmupCosineScheduler(_LRScheduler):
    """Linear warmup followed by cosine annealing.

    lr(t) =
        t < warmup:  lr_start + (lr_max - lr_start) * t / warmup_steps
        t >= warmup: lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π * (t-warmup)/(total-warmup)))
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 200,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            return [
                self.min_lr
                + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    steps_per_epoch: Optional[int] = None,
) -> _LRScheduler:
    """Factory function for learning rate schedulers.

    Args:
        optimizer: PyTorch optimizer
        config: Scheduler configuration dict
        steps_per_epoch: Number of training steps per epoch (for OneCycleLR)

    Returns:
        LR scheduler instance
    """
    scheduler_type = config.get("type", "cosine_warmup")
    total_epochs = config.get("total_epochs", 200)

    if scheduler_type == "cosine_warmup":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config.get("warmup_epochs", 5),
            total_epochs=total_epochs,
            min_lr=config.get("min_lr", 1e-6),
        )
    elif scheduler_type == "one_cycle":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for OneCycleLR")
        return OneCycleLR(
            optimizer,
            max_lr=config.get("max_lr", optimizer.defaults["lr"]),
            total_steps=total_epochs * steps_per_epoch,
            pct_start=config.get("pct_start", 0.1),
            anneal_strategy="cos",
            final_div_factor=config.get("final_div_factor", 1000),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
