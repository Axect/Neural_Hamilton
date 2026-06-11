import os
import json
import glob
import random
import torch
import numpy as np
from typing import Any, Optional


CHECKPOINT_VERSION = 1


def capture_rng_states() -> dict:
    rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng["torch_cuda"] = torch.cuda.get_rng_state_all()
    return rng


def restore_rng_states(rng: dict) -> None:
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.random.set_rng_state(rng["torch_cpu"])
    if "torch_cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["torch_cuda"])


def build_checkpoint_dict(model, optimizer, scheduler, epoch: int,
                          val_loss: float, metrics: dict,
                          early_stopping_state: Optional[dict] = None,
                          best_value: Optional[float] = None,
                          config_hash: str = "") -> dict:
    """Build a full training-state checkpoint dict."""
    ckpt = {
        "version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "metrics": metrics,
        "config_hash": config_hash,
        "rng_states": capture_rng_states(),
    }
    if hasattr(scheduler, "state_dict"):
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if early_stopping_state is not None:
        ckpt["early_stopping_state"] = early_stopping_state
    if best_value is not None:
        ckpt["best_value"] = best_value
    return ckpt


def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int,
                    val_loss: float, metrics: dict,
                    early_stopping_state: Optional[dict] = None,
                    best_value: Optional[float] = None,
                    config_hash: str = "") -> None:
    """Persist a full training-state checkpoint to disk."""
    ckpt = build_checkpoint_dict(
        model, optimizer, scheduler, epoch, val_loss, metrics,
        early_stopping_state, best_value, config_hash,
    )
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path: str, model, optimizer, scheduler,
                    device: str = "cpu", config_hash: str = "",
                    restore_rng: bool = True) -> dict:
    """Load checkpoint and restore model/optimizer/scheduler/RNG state."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"{path} is not a full-state checkpoint (older templates wrote a "
            "weights-only state_dict here). Delete it and start a fresh run, "
            "or train at least one epoch with the current template before "
            "using --resume."
        )

    saved_hash = checkpoint.get("config_hash", "")
    if config_hash and saved_hash and saved_hash != config_hash:
        raise ValueError(
            f"Config hash mismatch: checkpoint={saved_hash[:12]}... "
            f"vs current={config_hash[:12]}... "
            "The config has changed since this checkpoint was saved."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "scheduler_state_dict" in checkpoint and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if restore_rng and "rng_states" in checkpoint:
        restore_rng_states(checkpoint["rng_states"])

    return checkpoint


class CheckpointManager:
    """Manages periodic and best-model checkpoints during training.

    The always-on `latest_model.pt` snapshot is written by
    `LatestModelCallback`, not by this class. This class is only added
    to the callback chain when `checkpoint_config.enabled = True` and
    handles `best.pt` plus periodic `checkpoint_epoch_*.pt`.
    """

    CHECKPOINT_VERSION = CHECKPOINT_VERSION

    def __init__(self, run_dir: str, save_every_n: int = 10,
                 keep_last_k: int = 3, save_best: bool = True,
                 monitor: str = "val_loss", mode: str = "min"):
        self.run_dir = run_dir
        self.save_every_n = save_every_n
        self.keep_last_k = keep_last_k
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.best_value: Optional[float] = None

    def _is_better(self, current: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return current < self.best_value
        return current > self.best_value

    def save_checkpoint(self, path: str, model, optimizer, scheduler,
                        epoch: int, val_loss: float, metrics: dict,
                        early_stopping_state: Optional[dict] = None,
                        config_hash: str = "") -> None:
        """Save a complete training checkpoint (instance wrapper)."""
        save_checkpoint(
            path, model, optimizer, scheduler, epoch, val_loss, metrics,
            early_stopping_state=early_stopping_state,
            best_value=self.best_value,
            config_hash=config_hash,
        )

    def load_checkpoint(self, path: str, model, optimizer, scheduler,
                        device: str = "cpu", config_hash: str = "") -> dict:
        """Load a checkpoint and restore all state (instance wrapper)."""
        return load_checkpoint(
            path, model, optimizer, scheduler, device=device,
            config_hash=config_hash,
        )

    def maybe_save(self, epoch: int, model, optimizer, scheduler,
                   val_loss: float, metrics: dict,
                   early_stopping_state: Optional[dict] = None,
                   config_hash: str = "") -> None:
        """Conditionally save best/periodic checkpoints. `latest_model.pt`
        is handled by `LatestModelCallback`, not here."""
        # Periodic checkpoint
        if self.save_every_n > 0 and (epoch + 1) % self.save_every_n == 0:
            path = os.path.join(self.run_dir, f"checkpoint_epoch_{epoch}.pt")
            self.save_checkpoint(path, model, optimizer, scheduler,
                                 epoch, val_loss, metrics,
                                 early_stopping_state, config_hash)
            self._cleanup_old_checkpoints()

        # Best model checkpoint
        if self.save_best and self._is_better(val_loss):
            self.best_value = val_loss
            path = os.path.join(self.run_dir, "best.pt")
            self.save_checkpoint(path, model, optimizer, scheduler,
                                 epoch, val_loss, metrics,
                                 early_stopping_state, config_hash)

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old periodic checkpoints, keeping only the last k."""
        pattern = os.path.join(self.run_dir, "checkpoint_epoch_*.pt")
        checkpoints = sorted(glob.glob(pattern))
        while len(checkpoints) > self.keep_last_k:
            os.remove(checkpoints.pop(0))


def find_resume_checkpoint(run_path: str) -> Optional[str]:
    """Return path to `latest_model.pt` for the given run directory if it exists."""
    path = os.path.join(run_path, "latest_model.pt")
    return path if os.path.exists(path) else None


class SeedManifest:
    """Tracks completed seeds for multi-seed resume support."""

    def __init__(self, group_path: str):
        self.path = os.path.join(group_path, "seed_manifest.json")
        self.completed_seeds: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
                self.completed_seeds = data.get("completed_seeds", {})

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump({"completed_seeds": self.completed_seeds}, f, indent=2)

    def mark_complete(self, seed: int, val_loss: float,
                      wandb_run_id: str = "", metrics: dict | None = None) -> None:
        self.completed_seeds[str(seed)] = {
            "val_loss": val_loss,
            "wandb_run_id": wandb_run_id,
            "metrics": metrics or {},
        }
        self.save()

    def is_complete(self, seed: int) -> bool:
        return str(seed) in self.completed_seeds

    def get_total_loss(self) -> float:
        return sum(s["val_loss"] for s in self.completed_seeds.values())

    def get_complete_count(self) -> int:
        return len(self.completed_seeds)
