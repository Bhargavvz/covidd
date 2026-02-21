"""
Training Loop Engine (H200 Optimized)
Handles AMP, gradient accumulation, checkpointing, DDP, and logging.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from models.losses import RegistrationLoss
from models.spatial_transformer import compute_jacobian_determinant
from training.lr_scheduler import create_scheduler

logger = logging.getLogger(__name__)


class Trainer:
    """Training engine for deformable registration models.

    Features:
        - BF16/FP16 Automatic Mixed Precision (H200 Hopper native BF16)
        - Gradient accumulation for large effective batch sizes
        - Gradient clipping for stability
        - torch.compile() for H200 graph-mode optimization
        - Checkpointing with best-model tracking
        - TensorBoard logging
        - Early stopping
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        output_dir: str = "./outputs",
        device: str = "cuda",
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # ---- H200 Hardware Optimizations ----
        self._setup_hardware(config)

        # ---- Model ----
        self.model = model.to(self.device)

        # Optional: torch.compile for H200 Hopper graph optimization
        if config.get("compile", False):
            logger.info("Compiling model with torch.compile() for H200 optimization...")
            self.model = torch.compile(self.model, mode="default")

        # ---- Data ----
        self.train_loader = train_loader
        self.val_loader = val_loader

        # ---- Loss ----
        loss_config = config.get("loss", {})
        self.criterion = RegistrationLoss(loss_config).to(self.device)

        # ---- Optimizer ----
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5),
        )

        # ---- LR Scheduler ----
        sched_config = config.get("scheduler", {})
        sched_config["total_epochs"] = config.get("epochs", 200)
        self.scheduler = create_scheduler(
            self.optimizer,
            sched_config,
            steps_per_epoch=len(train_loader),
        )

        # ---- AMP ----
        self.amp_enabled = config.get("amp_enabled", True)
        amp_dtype_str = config.get("amp_dtype", "bfloat16")
        self.amp_dtype = (
            torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16
        )
        # GradScaler only needed for FP16, not BF16
        self.scaler = (
            GradScaler(enabled=True)
            if self.amp_enabled and self.amp_dtype == torch.float16
            else None
        )

        # ---- Training params ----
        self.epochs = config.get("epochs", 200)
        self.grad_accum_steps = config.get("grad_accum_steps", 8)
        self.grad_clip_norm = config.get("grad_clip_norm", 1.0)
        self.log_interval = config.get("log_interval", 10)

        # ---- Checkpointing ----
        ckpt_config = config.get("checkpoint", {})
        self.save_every = ckpt_config.get("save_every", 10)
        self.save_best = ckpt_config.get("save_best", True)
        self.best_metric_name = ckpt_config.get("metric", "val_ncc")
        self.best_metric_mode = ckpt_config.get("mode", "max")
        self.best_metric = float("-inf") if self.best_metric_mode == "max" else float("inf")

        # ---- Early Stopping ----
        es_config = config.get("early_stopping", {})
        self.early_stopping = es_config.get("enabled", True)
        self.patience = es_config.get("patience", 30)
        self.min_delta = es_config.get("min_delta", 1e-4)
        self.patience_counter = 0

        # ---- Logging ----
        log_config = config.get("logging", {})
        self.tb_writer = None
        if log_config.get("tensorboard", True) and SummaryWriter is not None:
            log_dir = Path(config.get("log_dir", "./logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(log_dir))

        # ---- State ----
        self.global_step = 0
        self.current_epoch = 0

    def _setup_hardware(self, config: dict):
        """Configure hardware-specific optimizations for H200."""
        if torch.cuda.is_available():
            # Enable TF32 for faster float32 matmuls on Hopper
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

            # Enable cuDNN benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True

            # BF16 optimization
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            logger.info("Enabled: TF32, cuDNN benchmark, BF16 reduction")

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_losses = {
            "total": 0.0,
            "sim": 0.0,
            "smooth": 0.0,
            "jac": 0.0,
        }
        num_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_loader):
            moving = batch["moving"].to(self.device, non_blocking=True)
            fixed = batch["fixed"].to(self.device, non_blocking=True)

            # Optional segmentation masks
            moving_seg = batch.get("moving_seg")
            fixed_seg = batch.get("fixed_seg")
            if moving_seg is not None:
                moving_seg = moving_seg.to(self.device, non_blocking=True)
            if fixed_seg is not None:
                fixed_seg = fixed_seg.to(self.device, non_blocking=True)

            # Forward pass with AMP
            with torch.amp.autocast(
                "cuda", enabled=self.amp_enabled, dtype=self.amp_dtype
            ):
                warped, flow = self.model(moving, fixed)
                losses = self.criterion(warped, fixed, flow)
                loss = losses["total"] / self.grad_accum_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0 or (
                batch_idx + 1
            ) == len(self.train_loader):
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item() * self.grad_accum_steps
            num_batches += 1

            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                logger.info(
                    f"  Epoch {self.current_epoch} "
                    f"[{batch_idx+1}/{len(self.train_loader)}] "
                    f"loss={losses['total'].item():.6f} "
                    f"sim={losses['sim'].item():.4f} "
                    f"smooth={losses['smooth'].item():.6f} "
                    f"jac={losses['jac'].item():.6f}"
                )

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        val_losses = {
            "total": 0.0,
            "sim": 0.0,
            "smooth": 0.0,
            "jac": 0.0,
        }
        num_batches = 0
        neg_jac_total = 0.0

        for batch in self.val_loader:
            moving = batch["moving"].to(self.device, non_blocking=True)
            fixed = batch["fixed"].to(self.device, non_blocking=True)

            with torch.amp.autocast(
                "cuda", enabled=self.amp_enabled, dtype=self.amp_dtype
            ):
                warped, flow = self.model(moving, fixed)
                losses = self.criterion(warped, fixed, flow)

            for key in val_losses:
                if key in losses:
                    val_losses[key] += losses[key].item()

            # Compute percentage of negative Jacobian determinants
            jac_det = compute_jacobian_determinant(flow.float())
            neg_jac_total += (jac_det <= 0).float().mean().item()
            num_batches += 1

        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)

        val_losses["neg_jac_pct"] = neg_jac_total / max(num_batches, 1)
        return val_losses

    def save_checkpoint(
        self, filename: str, is_best: bool = False, metrics: Optional[dict] = None
    ):
        """Save model checkpoint."""
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": (
                self.model._orig_mod.state_dict()
                if hasattr(self.model, "_orig_mod")
                else self.model.state_dict()
            ),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }

        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()

        if metrics:
            state["metrics"] = metrics

        path = ckpt_dir / filename
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path}")

        if is_best:
            best_path = ckpt_dir / "best_model.pth"
            torch.save(state, best_path)
            logger.info(f"Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.current_epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.best_metric = state.get("best_metric", self.best_metric)

        if self.scaler is not None and "scaler_state_dict" in state:
            self.scaler.load_state_dict(state["scaler_state_dict"])

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def _is_better(self, current: float) -> bool:
        """Check if current metric is better than best."""
        if self.best_metric_mode == "max":
            return current > self.best_metric + self.min_delta
        else:
            return current < self.best_metric - self.min_delta

    def train(self) -> Dict[str, list]:
        """Full training loop.

        Returns:
            History dict with loss curves
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_sim": [],
            "val_sim": [],
            "val_neg_jac_pct": [],
            "lr": [],
        }

        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info(f"  Model:     {type(self.model).__name__}")
        logger.info(f"  Epochs:    {self.epochs}")
        logger.info(f"  Batch:     {self.train_loader.batch_size}")
        logger.info(f"  AccumSteps: {self.grad_accum_steps}")
        logger.info(f"  AMP:       {self.amp_dtype}")
        logger.info(f"  Device:    {self.device}")
        if torch.cuda.is_available():
            logger.info(f"  GPU:       {torch.cuda.get_device_name(0)}")
            logger.info(
                f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        logger.info("=" * 60)

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_losses = self.train_epoch()

            # Validate
            val_losses = self.validate()

            # Update LR scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # Log
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} ({epoch_time:.1f}s) | "
                f"Train: loss={train_losses['total']:.6f} sim={train_losses['sim']:.4f} | "
                f"Val: loss={val_losses['total']:.6f} sim={val_losses['sim']:.4f} "
                f"neg_jac={val_losses['neg_jac_pct']:.4f} | "
                f"LR={current_lr:.2e}"
            )

            # TensorBoard
            if self.tb_writer:
                self.tb_writer.add_scalar("train/total_loss", train_losses["total"], epoch)
                self.tb_writer.add_scalar("train/sim_loss", train_losses["sim"], epoch)
                self.tb_writer.add_scalar("train/smooth_loss", train_losses["smooth"], epoch)
                self.tb_writer.add_scalar("val/total_loss", val_losses["total"], epoch)
                self.tb_writer.add_scalar("val/sim_loss", val_losses["sim"], epoch)
                self.tb_writer.add_scalar("val/neg_jac_pct", val_losses["neg_jac_pct"], epoch)
                self.tb_writer.add_scalar("lr", current_lr, epoch)

            # History
            history["train_loss"].append(train_losses["total"])
            history["val_loss"].append(val_losses["total"])
            history["train_sim"].append(train_losses["sim"])
            history["val_sim"].append(val_losses["sim"])
            history["val_neg_jac_pct"].append(val_losses["neg_jac_pct"])
            history["lr"].append(current_lr)

            # Checkpointing
            is_best = False
            metric_val = 1.0 - val_losses["sim"]  # NCC as metric (higher = better)
            if self._is_better(metric_val):
                self.best_metric = metric_val
                is_best = True
                self.patience_counter = 0
                logger.info(f"  ★ New best: {self.best_metric_name}={metric_val:.6f}")
            else:
                self.patience_counter += 1

            if is_best or (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(
                    f"checkpoint_epoch_{epoch+1:04d}.pth",
                    is_best=is_best,
                    metrics={"train": train_losses, "val": val_losses},
                )

            # Early stopping
            if self.early_stopping and self.patience_counter >= self.patience:
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {self.patience} epochs)"
                )
                break

            # GPU memory logging
            if torch.cuda.is_available() and (epoch + 1) % 10 == 0:
                mem_used = torch.cuda.max_memory_allocated() / 1e9
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(
                    f"  GPU Memory: {mem_used:.1f}/{mem_total:.1f} GB "
                    f"({100*mem_used/mem_total:.0f}%)"
                )
                torch.cuda.reset_peak_memory_stats()

        # Save final model
        self.save_checkpoint("final_model.pth", metrics={"history": history})

        if self.tb_writer:
            self.tb_writer.close()

        logger.info("Training complete!")
        return history
