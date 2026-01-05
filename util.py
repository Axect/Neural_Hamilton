import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import polars as pl
import numpy as np
import beaupy
from rich.console import Console
import wandb
import optuna
#from scipy.optimize import curve_fit, least_squares
#from scipy.stats import linregress
#import warnings

from config import RunConfig

import random
import os
import math


def load_data(file_path: str):
    """
    Load data from parquet file.

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


class EarlyStopping:
    def __init__(self, patience=10, mode="min", min_delta=0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if self.mode == "min":
            if val_loss <= self.best_loss * (1 - self.min_delta):
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == "max"
            if val_loss >= self.best_loss * (1 + self.min_delta):
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


def predict_final_loss(losses, max_epochs):
    if len(losses) < 10:
        return -np.log10(losses[-1])
    try:
        # Convert to numpy array
        y = np.array(losses)
        t = np.arange(len(y))

        # Decay fitting
        y_transformed = np.log(y)
        K, log_A = np.polyfit(t, y_transformed, 1)
        A = np.exp(log_A)

        # Predict final loss
        predicted_loss = -np.log10(A * np.exp(K * max_epochs))

        if np.isfinite(predicted_loss):
            return predicted_loss

    except Exception as e:
        print(f"Error in loss prediction: {e}")

    return -np.log10(losses[-1])

#def predict_final_loss(losses, max_epochs):
#    """
#    Predict final loss using multiple curve fitting models.
#    
#    Args:
#        losses: List of validation losses
#        max_epochs: Target epoch to predict
#        
#    Returns:
#        Predicted final loss (negative log scale)
#    """
#    if len(losses) < 5:
#        return -np.log10(losses[-1])
#    
#    # Convert to numpy arrays
#    y = np.array(losses)
#    t = np.arange(len(y))
#    
#    # Handle edge cases
#    if np.any(~np.isfinite(y)) or np.any(y <= 0):
#        return -np.log10(losses[-1])
#    
#    # Check for plateau - if recent losses are not changing much
#    if len(losses) >= 20:
#        recent_std = np.std(losses[-10:])
#        recent_mean = np.mean(losses[-10:])
#        if recent_std / recent_mean < 0.001:  # Very small relative change
#            return -np.log10(recent_mean)
#    
#    predictions = []
#    
#    # Model 1: Exponential decay - y = a * exp(b * t) + c
#    try:
#        def exp_decay(t, a, b, c):
#            return a * np.exp(b * t) + c
#        
#        # Initial guess based on data
#        a0 = y[0] - y[-1]
#        b0 = np.log(y[-1] / y[0]) / len(y) if y[0] > 0 and y[-1] > 0 else -0.1
#        c0 = min(y) * 0.9
#        
#        popt, _ = curve_fit(exp_decay, t, y, 
#                           p0=[a0, b0, c0],
#                           bounds=([0, -np.inf, 0], [np.inf, 0, min(y)]),
#                           maxfev=5000)
#        
#        pred = exp_decay(max_epochs, *popt)
#        if pred > 0 and pred < y[0]:  # Sanity check
#            predictions.append(pred)
#    except:
#        pass
#    
#    # Model 2: Power law - y = a * t^b + c
#    try:
#        def power_law(t, a, b, c):
#            return a * (t + 1) ** b + c
#        
#        # Transform to avoid t=0 issues
#        popt, _ = curve_fit(power_law, t, y,
#                           bounds=([0, -5, 0], [np.inf, 0, min(y)]),
#                           maxfev=5000)
#        
#        pred = power_law(max_epochs, *popt)
#        if pred > 0 and pred < y[0]:
#            predictions.append(pred)
#    except:
#        pass
#    
#    # Model 3: Logarithmic - y = a * log(t + 1) + b
#    try:
#        def log_model(t, a, b):
#            return a * np.log(t + 1) + b
#        
#        popt, _ = curve_fit(log_model, t, y, maxfev=5000)
#        pred = log_model(max_epochs, *popt)
#        
#        if pred > 0 and pred < y[0]:
#            predictions.append(pred)
#    except:
#        pass
#    
#    # Model 4: Inverse - y = a / (t + 1) + b
#    try:
#        def inverse_model(t, a, b):
#            return a / (t + 1) + b
#        
#        popt, _ = curve_fit(inverse_model, t, y,
#                           bounds=([0, 0], [np.inf, min(y)]),
#                           maxfev=5000)
#        
#        pred = inverse_model(max_epochs, *popt)
#        if pred > 0 and pred < y[0]:
#            predictions.append(pred)
#    except:
#        pass
#    
#    # Model 5: Double exponential for more complex curves
#    if len(losses) >= 10:
#        try:
#            def double_exp(t, a1, b1, a2, b2, c):
#                return a1 * np.exp(b1 * t) + a2 * np.exp(b2 * t) + c
#            
#            # Initial guesses
#            mid = len(y) // 2
#            a1_0 = (y[0] - y[mid]) * 0.7
#            a2_0 = (y[0] - y[mid]) * 0.3
#            b1_0 = np.log(0.5) / mid
#            b2_0 = np.log(0.1) / mid
#            c_0 = min(y) * 0.9
#            
#            popt, _ = curve_fit(double_exp, t, y,
#                               p0=[a1_0, b1_0, a2_0, b2_0, c_0],
#                               bounds=([0, -np.inf, 0, -np.inf, 0], 
#                                      [np.inf, 0, np.inf, 0, min(y)]),
#                               maxfev=5000)
#            
#            pred = double_exp(max_epochs, *popt)
#            if pred > 0 and pred < y[0]:
#                predictions.append(pred)
#        except:
#            pass
#    
#    # Model 6: Polynomial with constraints (for smooth extrapolation)
#    if len(losses) >= 10:
#        try:
#            # Use lower degree polynomial to avoid overfitting
#            degree = min(3, len(losses) // 5)
#            
#            # Fit polynomial to recent data for better local behavior
#            recent_points = min(20, len(losses))
#            t_recent = t[-recent_points:]
#            y_recent = y[-recent_points:]
#            
#            # Normalize for numerical stability
#            t_norm = (t_recent - t_recent[0]) / (t_recent[-1] - t_recent[0])
#            coeffs = np.polyfit(t_norm, y_recent, degree)
#            
#            # Extrapolate
#            t_pred_norm = (max_epochs - t_recent[0]) / (t_recent[-1] - t_recent[0])
#            pred = np.polyval(coeffs, t_pred_norm)
#            
#            # Only accept if decreasing and reasonable
#            if pred > 0 and pred < y_recent[0]:
#                predictions.append(pred)
#        except:
#            pass
#    
#    # If we have predictions, use robust averaging
#    if predictions:
#        # Remove outliers using IQR
#        predictions = np.array(predictions)
#        q1, q3 = np.percentile(predictions, [25, 75])
#        iqr = q3 - q1
#        lower_bound = q1 - 1.5 * iqr
#        upper_bound = q3 + 1.5 * iqr
#        
#        # Filter predictions
#        filtered = predictions[(predictions >= lower_bound) & 
#                             (predictions <= upper_bound)]
#        
#        if len(filtered) > 0:
#            # Weighted average favoring lower predictions (more conservative)
#            weights = 1.0 / (filtered + 1e-10)
#            final_pred = np.average(filtered, weights=weights)
#        else:
#            final_pred = np.median(predictions)
#        
#        return -np.log10(final_pred)
#    
#    # Fallback: linear extrapolation of recent trend
#    if len(losses) >= 10:
#        recent = losses[-10:]
#        t_recent = np.arange(len(recent))
#        slope, intercept, _, _, _ = linregress(t_recent, recent)
#        
#        if slope < 0:  # Only if decreasing
#            pred = intercept + slope * (max_epochs - len(losses) + 10)
#            if pred > 0:
#                return -np.log10(pred)
#    
#    # Final fallback
#    return -np.log10(losses[-1])


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        early_stopping_config=None,
        device="cpu",
        variational=False,
        trial=None,
        seed=None,
        pruner=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.variational = variational
        self.trial = trial
        self.seed = seed
        self.pruner = pruner

        if early_stopping_config and early_stopping_config.enabled:
            self.early_stopping = EarlyStopping(
                patience=early_stopping_config.patience,
                mode=early_stopping_config.mode,
                min_delta=early_stopping_config.min_delta,
            )
        else:
            self.early_stopping = None

    def step(self, V, t, ic):
        return self.model(V, t, ic)

    def _obtain_loss(self, V, t, q, p, ic):
        q_pred, p_pred = self.step(V, t, ic)
        loss_q = self.criterion(q_pred, q)
        loss_p = self.criterion(p_pred, p)
        loss = 0.5 * (loss_q + loss_p)
        return loss

    def _obtain_vae_loss(self, V, t, q, p, ic):
        q_pred, p_pred, mu, logvar = self.step(V, t, ic)

        # Flatten
        mu_vec = mu.view((mu.shape[0], -1))
        logvar_vec = logvar.view((logvar.shape[0], -1))

        # KL Divergence (mean over latent dimensions)
        kl_loss = -0.5 * torch.mean(
            1 + logvar_vec - mu_vec.pow(2) - logvar_vec.exp(), dim=1
        )
        beta = self.model.kl_weight
        kl_loss = beta * torch.mean(kl_loss)

        # Total loss
        loss_q = self.criterion(q_pred, q)
        loss_p = self.criterion(p_pred, p)
        loss = 0.5 * (loss_q + loss_p) + kl_loss
        return loss

    def train_epoch(self, dl_train):
        self.model.train()
        # ScheduleFree Optimizer or SPlus
        if any(keyword in self.optimizer.__class__.__name__ for keyword in ["ScheduleFree", "SPlus"]):
            self.optimizer.train()
        train_loss = 0
        for V, t, q, p, ic in dl_train:
            V = V.to(self.device)
            t = t.to(self.device)
            q = q.to(self.device)
            p = p.to(self.device)
            ic = ic.to(self.device)
            if not self.variational:
                loss = self._obtain_loss(V, t, q, p, ic)
            else:
                loss = self._obtain_vae_loss(V, t, q, p, ic)
            train_loss += loss.item()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        train_loss /= len(dl_train)
        return train_loss

    def val_epoch(self, dl_val):
        self.model.eval()
        # ScheduleFree Optimizer or SPlus
        if any(keyword in self.optimizer.__class__.__name__ for keyword in ["ScheduleFree", "SPlus"]):
            self.optimizer.eval()
        val_loss = 0
        with torch.no_grad():
            for V, t, q, p, ic in dl_val:
                V = V.to(self.device)
                t = t.to(self.device)
                q = q.to(self.device)
                p = p.to(self.device)
                ic = ic.to(self.device)
                if not self.variational:
                    loss = self._obtain_loss(V, t, q, p, ic)
                else:
                    loss = self._obtain_vae_loss(V, t, q, p, ic)
                val_loss += loss.item()
        val_loss /= len(dl_val)
        return val_loss

    def train(self, dl_train, dl_val, epochs):
        val_loss = 0
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            train_loss = self.train_epoch(dl_train)
            val_loss = self.val_epoch(dl_val)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Early stopping if loss becomes NaN
            if math.isnan(train_loss) or math.isnan(val_loss):
                print("Early stopping due to NaN loss")
                train_loss = math.inf
                val_loss = math.inf
                break

            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping(val_loss):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            log_dict = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            }

            if epoch >= 10:
                log_dict["predicted_final_loss"] = predict_final_loss(
                    train_losses, epochs
                )

            # Pruning check
            if (
                self.pruner is not None
                and self.trial is not None
                and self.seed is not None
            ):
                self.pruner.report(
                    trial_id=self.trial.number,
                    seed=self.seed,
                    epoch=epoch,
                    value=val_loss,
                )
                if self.pruner.should_prune():
                    raise optuna.TrialPruned()

            self.scheduler.step()
            wandb.log(log_dict)
            if epoch % 10 == 0 or epoch == epochs - 1:
                print_str = f"epoch: {epoch}"
                for key, value in log_dict.items():
                    print_str += f", {key}: {value:.4e}"
                print(print_str)

        return val_loss


def log_cosh_loss(y_pred, y_true, reduction="mean"):
    error = y_pred - y_true
    loss = torch.log(torch.cosh(error))
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss # No reduction


def np_log_cosh_loss(y_pred, y_true, reduction="mean"):
    error = y_pred - y_true
    loss = np.log(np.cosh(error))
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        return loss  # No reduction


def run(
    run_config: RunConfig,
    dl_train,
    dl_val,
    group_name=None,
    data=None,
    trial=None,
    pruner=None,
):
    project = run_config.project
    device = run_config.device
    seeds = run_config.seeds
    if not group_name:
        group_name = run_config.gen_group_name(data)
    tags = run_config.gen_tags()

    group_path = f"runs/{run_config.project}/{group_name}"
    if not os.path.exists(group_path):
        os.makedirs(group_path)
    run_config.to_yaml(f"{group_path}/config.yaml")

    # Register trial at the beginning if pruner exists
    if pruner is not None and trial is not None and hasattr(pruner, "register_trial"):
        pruner.register_trial(trial.number)

    total_loss = 0
    complete_seeds = 0
    try:
        for seed in seeds:
            set_seed(seed)

            model = run_config.create_model().to(device)
            optimizer = run_config.create_optimizer(model)
            scheduler = run_config.create_scheduler(optimizer)

            run_name = f"{seed}"
            wandb.init(
                project=project,
                name=run_name,
                group=group_name,
                tags=tags,
                config=run_config.gen_config(),
            )

            # Check if using VaRONet
            variational = "VaRONet" in run_config.net

            trainer = Trainer(
                model,
                optimizer,
                scheduler,
                criterion=log_cosh_loss,    # v0.21, v0.24
                #criterion=F.mse_loss,      # ~v0.20, v0.22, v0.23
                early_stopping_config=run_config.early_stopping_config,
                device=device,
                variational=variational,
                trial=trial,
                seed=seed,
                pruner=pruner,
            )

            val_loss = trainer.train(dl_train, dl_val, epochs=run_config.epochs)
            total_loss += val_loss
            complete_seeds += 1

            # Save model & configs
            run_path = f"{group_path}/{run_name}"
            if not os.path.exists(run_path):
                os.makedirs(run_path)
            torch.save(model.state_dict(), f"{run_path}/model.pt")

            wandb.finish()

            # Early stopping if loss becomes inf
            if math.isinf(val_loss):
                break

    except optuna.TrialPruned:
        wandb.finish()
        raise
    except Exception as e:
        print(f"Runtime error during training: {e}")
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

    return total_loss / (complete_seeds if complete_seeds > 0 else 1)


# ┌──────────────────────────────────────────────────────────┐
#  For Analyze
# └──────────────────────────────────────────────────────────┘
def select_project():
    runs_path = "runs/"
    projects = [
        d for d in os.listdir(runs_path) if os.path.isdir(os.path.join(runs_path, d))
    ]
    # Sort the project names
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
    return selected_group  # pyright: ignore


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
    project_name = f"{project}_Opt"
    group_name = best_trial.user_attrs["group_name"]

    # Select Seed
    seed = select_seed(project_name, group_name)
    best_model, best_config = load_model(
        project_name, group_name, seed, weights_only=weights_only
    )

    return best_model, best_config
