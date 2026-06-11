import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
import polars as pl
import beaupy
import wandb
import optuna
from tqdm import tqdm

from config import RunConfig
from callbacks import (
    CallbackRunner, OptimizerModeCallback, EarlyStoppingCallback,
    WandbLoggingCallback, PrunerCallback, LossPredictionCallback,
    NaNDetectionCallback, CheckpointCallback,
    GradientMonitorCallback, OverfitDetectionCallback,
    CSVLoggingCallback, TUILoggingCallback, LatestModelCallback,
)
from checkpoint import (
    CheckpointManager, SeedManifest, find_resume_checkpoint, load_checkpoint,
)
from provenance import save_provenance, compute_config_hash

import random
import os
import math
import time


def load_data(file_path: str):
    """
    Load data from parquet file (flat format, 100 rows per sample).

    Extra metadata columns (pid, b, l, depth, E0, u) are ignored here;
    they are consumed by analysis scripts, not by training.

    Returns:
        TensorDataset with (V, t, q, p, ic) where ic = (q0, p0)
    """
    df = pl.read_parquet(file_path)
    V = torch.tensor(df["V"].to_numpy().reshape(-1, 100), dtype=torch.float32)
    t = torch.tensor(df["t"].to_numpy().reshape(-1, 100), dtype=torch.float32)
    q = torch.tensor(df["q"].to_numpy().reshape(-1, 100), dtype=torch.float32)
    p = torch.tensor(df["p"].to_numpy().reshape(-1, 100), dtype=torch.float32)

    # Extract initial conditions (first time point of each sample)
    ic = torch.stack([q[:, 0], p[:, 0]], dim=1)  # (N, 2)

    return TensorDataset(V, t, q, p, ic)


def load_normal():
    """Data loader for RunConfig.data: 16k/4k samples (4k/1k potentials)."""
    return (
        load_data("data_normal/train.parquet"),
        load_data("data_normal/val.parquet"),
    )


def load_more():
    """Data loader for RunConfig.data: 160k/40k samples (40k/10k potentials)."""
    return (
        load_data("data_more/train.parquet"),
        load_data("data_more/val.parquet"),
    )


def load_smoke():
    """Tiny dataset for pipeline smoke tests (generated via `neural_hamilton 3`)."""
    ds = load_data("data_smoke/smoke.parquet")
    n_val = max(len(ds) // 5, 1)
    return TensorDataset(*ds[: len(ds) - n_val]), TensorDataset(*ds[len(ds) - n_val :])


def set_seed(seed: int):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def predict_final_loss(losses, max_epochs):
    """Predict the final validation loss using shifted exponential decay.

    Fits L(t) = a * exp(-b * t) + c to EMA-smoothed losses.
    Returns the predicted raw loss value at max_epochs.
    Works with positive and negative losses.
    """
    n = len(losses)
    if n < 10:
        return float(losses[-1])

    y = np.array(losses, dtype=np.float64)

    # EMA smoothing — adaptive span
    span = min(n // 3, 20)
    alpha = 2.0 / (span + 1)
    ema = np.empty(n)
    ema[0] = y[0]
    for i in range(1, n):
        ema[i] = alpha * y[i] + (1 - alpha) * ema[i - 1]

    # Three equally-spaced anchor points from smoothed curve
    i1, i2, i3 = n // 3, 2 * n // 3, n - 1
    y1, y2, y3 = ema[i1], ema[i2], ema[i3]

    d12 = y1 - y2
    d23 = y2 - y3

    # Need both differences nonzero and same sign (monotonic decay or increase)
    if abs(d12) < 1e-15 or abs(d23) < 1e-15:
        return float(ema[-1])

    r = d23 / d12

    if r <= 0 or r >= 1:
        # Non-convergent: loss increasing, oscillating, or accelerating
        # Use damped linear extrapolation from recent trend
        window = min(10, n - 1)
        recent_rate = (ema[-1] - ema[-1 - window]) / window
        remaining = max(max_epochs - n, 0)
        predicted = ema[-1] + recent_rate * remaining * 0.5
        return float(predicted) if np.isfinite(predicted) else float(ema[-1])

    # Convergent decay: fit L(t) = a * exp(-b * t) + c
    d = float(i2 - i1)
    b = -np.log(r) / d
    t1 = float(i1)
    t2 = float(i2)

    denom = np.exp(-b * t1) - np.exp(-b * t2)
    if abs(denom) < 1e-30:
        return float(ema[-1])

    a = d12 / denom
    c = y1 - a * np.exp(-b * t1)

    predicted = a * np.exp(-b * max_epochs) + c

    if np.isfinite(predicted):
        return float(predicted)

    return float(ema[-1])


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        callbacks=None,
        device="cpu",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.callbacks = callbacks if callbacks is not None else CallbackRunner()
        self._total_epochs = 0
        self._loss_prediction = None
        self._max_grad_norm: float | None = None
        self._overfit_gap_ratio: float | None = None

    def step(self, V, t, ic):
        return self.model(V, t, ic)

    def _obtain_loss(self, V, t, q, p, ic):
        q_pred, p_pred = self.step(V, t, ic)
        loss_q = self.criterion(q_pred, q)
        loss_p = self.criterion(p_pred, p)
        return 0.5 * (loss_q + loss_p)

    def train_epoch(self, dl_train, epoch=None, total_epochs=None):
        self.model.train()
        self.callbacks.fire("on_train_epoch_begin", trainer=self, epoch=epoch)
        train_loss = 0
        total_size = 0

        # Create progress bar description
        desc = f"Epoch {epoch+1}/{total_epochs}" if epoch is not None and total_epochs is not None else "Training"

        for batch_idx, (V, t, q, p, ic) in enumerate(tqdm(dl_train, desc=desc, leave=False)):
            V = V.to(self.device)
            t = t.to(self.device)
            q = q.to(self.device)
            p = p.to(self.device)
            ic = ic.to(self.device)
            loss = self._obtain_loss(V, t, q, p, ic)
            train_loss += loss.item() * V.shape[0]
            total_size += V.shape[0]
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.callbacks.fire("on_train_step_end", trainer=self, batch_idx=batch_idx, loss=loss.item())
        train_loss /= total_size
        return train_loss

    def val_epoch(self, dl_val, epoch=None):
        self.model.eval()
        self.callbacks.fire("on_val_begin", trainer=self, epoch=epoch)
        val_loss = 0
        total_size = 0
        with torch.inference_mode():
            for V, t, q, p, ic in tqdm(dl_val, desc="Validation", leave=False):
                V = V.to(self.device)
                t = t.to(self.device)
                q = q.to(self.device)
                p = p.to(self.device)
                ic = ic.to(self.device)
                loss = self._obtain_loss(V, t, q, p, ic)
                val_loss += loss.item() * V.shape[0]
                total_size += V.shape[0]
        val_loss /= total_size
        self.callbacks.fire("on_val_end", trainer=self, epoch=epoch, val_loss=val_loss, metrics={})
        return val_loss

    def train(self, dl_train, dl_val, epochs, start_epoch: int = 0):
        self._total_epochs = epochs
        self.callbacks.fire(
            "on_train_begin", trainer=self, epochs=epochs, start_epoch=start_epoch,
        )
        val_loss = 0

        if start_epoch >= epochs:
            tqdm.write(
                f"start_epoch={start_epoch} >= epochs={epochs}; nothing to do."
            )
            self.callbacks.fire("on_train_end", trainer=self)
            return val_loss

        for epoch in tqdm(range(start_epoch, epochs), desc="Overall Progress"):
            train_loss = self.train_epoch(dl_train, epoch=epoch, total_epochs=epochs)
            val_loss = self.val_epoch(dl_val, epoch=epoch)

            self.callbacks.fire(
                "on_epoch_end", trainer=self, epoch=epoch,
                train_loss=train_loss, val_loss=val_loss, metrics={},
            )

            # Check callback signals
            break_flag = False
            for cb in self.callbacks.callbacks:
                if isinstance(cb, NaNDetectionCallback) and cb.nan_detected:
                    val_loss = math.inf
                    break_flag = True
                    break
                if isinstance(cb, EarlyStoppingCallback) and cb.should_stop:
                    tqdm.write(f"Early stopping triggered at epoch {epoch}")
                    break_flag = True
                    break
            if break_flag:
                break

            self.scheduler.step()

        self.callbacks.fire("on_train_end", trainer=self)
        return val_loss


def log_cosh_loss(y_pred, y_true, reduction="mean"):
    error = y_pred - y_true
    loss = torch.log(torch.cosh(error))
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss  # No reduction


def np_log_cosh_loss(y_pred, y_true, reduction="mean"):
    error = y_pred - y_true
    loss = np.log(np.cosh(error))
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        return loss  # No reduction


class LogCoshLoss(nn.Module):
    """Criterion class for RunConfig (criterion: util.LogCoshLoss)."""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        return log_cosh_loss(y_pred, y_true, reduction=self.reduction)


def run(
    run_config: RunConfig, dl_train, dl_val, group_name=None, trial=None, pruner=None,
    resume: bool = False,
):
    project = run_config.project
    device = run_config.device
    seeds = run_config.seeds
    if not group_name:
        group_name = run_config.gen_group_name()
    tags = run_config.gen_tags()

    group_path = f"runs/{run_config.project}/{group_name}"
    if not os.path.exists(group_path):
        os.makedirs(group_path)
    run_config.to_yaml(f"{group_path}/config.yaml")

    # Register trial at the beginning if pruner exists
    if pruner is not None and trial is not None and hasattr(pruner, "register_trial"):
        pruner.register_trial(trial.number)

    # Create seed manifest for multi-seed resume support
    manifest = SeedManifest(group_path)

    # Create criterion from config
    criterion = run_config.create_criterion()
    use_wandb = run_config.wandb

    try:
        for seed in seeds:
            # Skip already-completed seeds
            if manifest.is_complete(seed):
                tqdm.write(f"Seed {seed} already complete, skipping")
                continue

            set_seed(seed)

            model = run_config.create_model().to(device)
            optimizer = run_config.create_optimizer(model)
            scheduler = run_config.create_scheduler(optimizer)

            run_name = f"{seed}"
            run_path = f"{group_path}/{run_name}"
            if not os.path.exists(run_path):
                os.makedirs(run_path)

            config_hash = compute_config_hash(run_config)

            # Resume from latest_model.pt if requested and present
            start_epoch = 0
            resumed_ckpt = None
            if resume:
                ckpt_path = find_resume_checkpoint(run_path)
                if ckpt_path is not None:
                    resumed_ckpt = load_checkpoint(
                        ckpt_path, model, optimizer, scheduler,
                        device=device, config_hash=config_hash,
                    )
                    start_epoch = int(resumed_ckpt["epoch"]) + 1
                    tqdm.write(
                        f"Resuming seed {seed} from epoch {start_epoch} "
                        f"(checkpoint val_loss={resumed_ckpt.get('val_loss', 'n/a')})"
                    )
                else:
                    tqdm.write(
                        f"--resume requested but no latest_model.pt at {run_path}; "
                        f"starting from scratch."
                    )

            if use_wandb:
                wandb.init(
                    project=project,
                    name=run_name,
                    group=group_name,
                    tags=tags,
                    config=run_config.gen_config(),
                )

            # Build callbacks list
            callbacks_list = [
                OptimizerModeCallback(),
                NaNDetectionCallback(),
                GradientMonitorCallback(),
                LossPredictionCallback(run_config.epochs),
                OverfitDetectionCallback(),
            ]
            callbacks_list.append(TUILoggingCallback())
            if use_wandb:
                callbacks_list.append(WandbLoggingCallback())
            # Always-on callbacks: CSV logging + latest full-state checkpoint
            callbacks_list.append(CSVLoggingCallback(f"{run_path}/metrics.csv"))

            early_stopping_cb = None
            if run_config.early_stopping_config and run_config.early_stopping_config.enabled:
                early_stopping_cb = EarlyStoppingCallback(
                    patience=run_config.early_stopping_config.patience,
                    mode=run_config.early_stopping_config.mode,
                    min_delta=run_config.early_stopping_config.min_delta,
                )
                if resumed_ckpt is not None and "early_stopping_state" in resumed_ckpt:
                    early_stopping_cb.load_state_dict(
                        resumed_ckpt["early_stopping_state"]
                    )
                callbacks_list.append(early_stopping_cb)
            if pruner is not None and trial is not None:
                callbacks_list.append(PrunerCallback(pruner, trial, seed))

            # CheckpointManager controls best.pt + periodic snapshots (opt-in).
            ckpt_manager = None
            if run_config.checkpoint_config.enabled:
                ckpt_manager = CheckpointManager(
                    run_dir=run_path,
                    save_every_n=run_config.checkpoint_config.save_every_n_epochs,
                    keep_last_k=run_config.checkpoint_config.keep_last_k,
                    save_best=run_config.checkpoint_config.save_best,
                    monitor=run_config.checkpoint_config.monitor,
                    mode=run_config.checkpoint_config.mode,
                )
                if resumed_ckpt is not None and "best_value" in resumed_ckpt:
                    ckpt_manager.best_value = resumed_ckpt["best_value"]
                callbacks_list.append(CheckpointCallback(ckpt_manager, config_hash))

            # Full-state resume checkpoints are only useful for long final runs;
            # during HPO they are dead weight (34 MB per seed-run with SPlus
            # state, ~15 GB per 100-trial study) and can fill the disk.
            if trial is None:
                callbacks_list.append(LatestModelCallback(
                    f"{run_path}/latest_model.pt",
                    config_hash=config_hash,
                    checkpoint_manager=ckpt_manager,
                ))

            callback_runner = CallbackRunner(callbacks_list)

            trainer = Trainer(
                model,
                optimizer,
                scheduler,
                criterion=criterion,
                callbacks=callback_runner,
                device=device,
            )

            start_time = time.time()
            val_loss = trainer.train(
                dl_train, dl_val,
                epochs=run_config.epochs,
                start_epoch=start_epoch,
            )
            end_time = time.time()

            # No-op resume (start_epoch >= epochs): take val_loss from the
            # restored checkpoint so the manifest entry is meaningful.
            if (
                start_epoch >= run_config.epochs
                and resumed_ckpt is not None
                and "val_loss" in resumed_ckpt
            ):
                val_loss = float(resumed_ckpt["val_loss"])

            # Save model & configs
            torch.save(model.state_dict(), f"{run_path}/model.pt")

            # Save provenance
            save_provenance(run_path, run_config, model, device, start_time, end_time)

            # Mark seed as complete
            manifest.mark_complete(seed, val_loss)

            if use_wandb:
                wandb.finish()

            # Early stopping if loss becomes inf
            if math.isinf(val_loss):
                break

    except optuna.TrialPruned:
        if use_wandb:
            wandb.finish()
        raise
    except Exception as e:
        tqdm.write(f"Runtime error during training: {e}")
        if use_wandb:
            wandb.finish()
        raise optuna.TrialPruned()
    finally:
        # Call trial_finished only once after all seeds are done
        if (
            pruner is not None
            and trial is not None
            and hasattr(pruner, "complete_trial")
        ):
            pruner.complete_trial(trial.number)

    complete_count = manifest.get_complete_count()
    return manifest.get_total_loss() / (complete_count if complete_count > 0 else 1)


# ┌──────────────────────────────────────────────────────────┐
#  For Analyze
# └──────────────────────────────────────────────────────────┘
def select_project():
    runs_path = "runs/"
    projects = [
        d for d in os.listdir(runs_path) if os.path.isdir(os.path.join(runs_path, d))
    ]
    projects.sort()
    if not projects:
        raise ValueError(f"No projects found in {runs_path}")

    selected_project = beaupy.select(projects)
    return selected_project


def select_group(project):
    runs_path = f"runs/{project}"
    groups = [
        d for d in os.listdir(runs_path) if os.path.isdir(os.path.join(runs_path, d))
    ]
    groups.sort()
    if not groups:
        raise ValueError(f"No run groups found in {runs_path}")

    selected_group = beaupy.select(groups)
    return selected_group


def select_seed(project, group_name):
    group_path = f"runs/{project}/{group_name}"
    seeds = [
        d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))
    ]
    seeds.sort()
    if not seeds:
        raise ValueError(f"No seeds found in {group_path}")

    selected_seed = beaupy.select(seeds)
    return selected_seed


def select_device():
    devices = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    selected_device = beaupy.select(devices)
    return selected_device


def load_model(project, group_name, seed, weights_only=True):
    """
    Load a trained model and its configuration.

    Args:
        project (str): The name of the project.
        group_name (str): The name of the run group.
        seed (str): The seed of the specific run.
        weights_only (bool, optional): If True, only load the model weights without loading the entire pickle file.
                                       This can be faster and use less memory. Defaults to True.

    Returns:
        tuple: A tuple containing the loaded model and its configuration.

    Raises:
        FileNotFoundError: If the config or model file is not found.

    Example usage:
        # Load full model
        model, config = load_model("MyProject", "experiment1", "seed42")

        # Load only weights (faster and uses less memory)
        model, config = load_model("MyProject", "experiment1", "seed42", weights_only=True)
    """
    config_path = f"runs/{project}/{group_name}/config.yaml"
    model_path = f"runs/{project}/{group_name}/{seed}/model.pt"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found for {project}/{group_name}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found for {project}/{group_name}/{seed}"
        )

    config = RunConfig.from_yaml(config_path)
    model = config.create_model()

    # Use weights_only option in torch.load
    state_dict = torch.load(model_path, map_location="cpu", weights_only=weights_only)
    model.load_state_dict(state_dict)

    return model, config


def load_study(project, study_name):
    """
    Load the best study from an optimization run.

    Args:
        project (str): The name of the project.
        study_name (str): The name of the study.

    Returns:
        optuna.Study: The loaded study object.
    """
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{project}.db")
    return study


def load_best_model(project, study_name, weights_only=True):
    """
    Load the best model and its configuration from an optimization study.

    Args:
        project (str): The name of the project.
        study_name (str): The name of the study.

    Returns:
        tuple: A tuple containing the loaded model, its configuration, and the best trial number.
    """
    study = load_study(project, study_name)
    best_trial = study.best_trial
    project_name = project
    group_name = best_trial.user_attrs["group_name"]

    # Select Seed
    seed = select_seed(project_name, group_name)
    best_model, best_config = load_model(
        project_name, group_name, seed, weights_only=weights_only
    )

    return best_model, best_config
