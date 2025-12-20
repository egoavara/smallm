"""Training utilities."""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm

from ..model.llama import LLaMA
from ..model.config import ModelConfig


@dataclass
class TrainConfig:
    """Training configuration."""

    # Training parameters
    max_steps: int = 10000
    batch_size: int = 16
    gradient_accumulation_steps: int = 1

    # Optimizer parameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0

    # Learning rate schedule
    warmup_steps: int = 100
    min_lr: float = 1e-5

    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    output_dir: str = "checkpoints"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile_model: bool = False  # Use torch.compile


class Trainer:
    """Trainer for LLaMA model."""

    def __init__(
        self,
        model: LLaMA,
        train_loader: DataLoader,
        config: TrainConfig,
        eval_loader: Optional[DataLoader] = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: LLaMA model to train
            train_loader: Training data loader
            config: Training configuration
            eval_loader: Optional evaluation data loader
        """
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config

        # Move model to device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        # Optionally compile model (PyTorch 2.0+)
        if config.compile_model and hasattr(torch, "compile"):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps - config.warmup_steps,
            eta_min=config.min_lr,
        )

        # Training state
        self.step = 0
        self.best_eval_loss = float("inf")

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay.

        We don't apply weight decay to bias and LayerNorm/RMSNorm weights.
        """
        # Separate parameters into decay and no-decay groups
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Don't decay bias and norm weights
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )

    def _get_lr(self) -> float:
        """Get current learning rate with warmup."""
        if self.step < self.config.warmup_steps:
            # Linear warmup
            return self.config.learning_rate * (self.step + 1) / self.config.warmup_steps
        else:
            # Use scheduler's learning rate
            return self.scheduler.get_last_lr()[0]

    def _set_lr(self, lr: float) -> None:
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train(self) -> None:
        """Run training loop."""
        self.model.train()

        # Create infinite iterator
        train_iter = iter(self.train_loader)

        # Training metrics
        total_loss = 0.0
        start_time = time.time()

        pbar = tqdm(total=self.config.max_steps, desc="Training")
        pbar.update(self.step)

        while self.step < self.config.max_steps:
            # Get batch (with wraparound)
            try:
                input_ids, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                input_ids, targets = next(train_iter)

            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            logits, loss = self.model(input_ids, targets)

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                # Update learning rate
                lr = self._get_lr()
                self._set_lr(lr)

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update scheduler after warmup
                if self.step >= self.config.warmup_steps:
                    self.scheduler.step()

            self.step += 1
            pbar.update(1)

            # Logging
            if self.step % self.config.log_interval == 0:
                avg_loss = total_loss / self.config.log_interval
                elapsed = time.time() - start_time
                tokens_per_sec = (
                    self.config.log_interval
                    * self.config.batch_size
                    * input_ids.shape[1]
                    / elapsed
                )

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{self._get_lr():.2e}",
                    "tok/s": f"{tokens_per_sec:.0f}",
                })

                total_loss = 0.0
                start_time = time.time()

            # Evaluation
            if self.eval_loader and self.step % self.config.eval_interval == 0:
                eval_loss = self.evaluate()
                print(f"\nStep {self.step}: eval_loss = {eval_loss:.4f}")

                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint("best.pt")

                self.model.train()

            # Save checkpoint
            if self.step % self.config.save_interval == 0:
                self.save_checkpoint(f"step_{self.step}.pt")

        pbar.close()

        # Final save
        self.save_checkpoint("final.pt")
        print(f"Training complete! Best eval loss: {self.best_eval_loss:.4f}")

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation and return average loss."""
        self.model.eval()

        total_loss = 0.0
        n_batches = 0

        for input_ids, targets in tqdm(self.eval_loader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            _, loss = self.model(input_ids, targets)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        path = os.path.join(self.config.output_dir, filename)

        checkpoint = {
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "best_eval_loss": self.best_eval_loss,
        }

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.step = checkpoint["step"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))

        print(f"Loaded checkpoint from {path} (step {self.step})")
