import torch
from torch import nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import polars as pl
import numpy as np
import survey
import wandb
import optuna

from config import RunConfig, OptimizeConfig

import random
import os


def load_data(file_path: str):
    df = pl.read_parquet(file_path)
    tensors = [torch.tensor(df[col].to_numpy(
    ).reshape(-1, 100), dtype=torch.float32) for col in df.columns]
    return TensorDataset(*tensors)


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


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device="cpu", variational=False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.variational = variational

    def step(self, V, t):
        return self.model(V, t)

    def _obtain_loss(self, V, t, x, p):
        x_pred, p_pred = self.step(V, t)
        loss_x = self.criterion(x_pred, x.to(self.device))
        loss_p = self.criterion(p_pred, p.to(self.device))
        loss = 0.5 * (loss_x + loss_p)
        return loss

    def _obtain_vae_loss(self, V, t, x, p):
        x_pred, p_pred, mu, logvar = self.step(V, t)

        # Flatten
        mu_vec = mu.view((mu.shape[0], -1))
        logvar_vec = logvar.view((logvar.shape[0], -1))

        # KL Divergence (mean over latent dimensions)
        kl_loss = -0.5 * \
            torch.mean(1 + logvar_vec - mu_vec.pow(2) - logvar_vec.exp(), dim=1)
        beta = self.model.kl_weight
        kl_loss = beta * torch.mean(kl_loss)

        # Total loss
        loss_x = self.criterion(x_pred, x.to(self.device))
        loss_p = self.criterion(p_pred, p.to(self.device))
        loss = 0.5 * (loss_x + loss_p) + kl_loss
        return loss

    def train_epoch(self, dl_train):
        self.model.train()
        train_loss = 0
        for V, t, x, p in dl_train:
            V = V.to(self.device)
            t = t.to(self.device)
            if not self.variational:
                loss = self._obtain_loss(V, t, x, p)
            else:
                loss = self._obtain_vae_loss(V, t, x, p)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        train_loss /= len(dl_train)
        return train_loss

    def val_epoch(self, dl_val):
        self.model.eval()
        val_loss = 0
        for V, t, x, p in dl_val:
            V = V.to(self.device)
            t = t.to(self.device)
            if not self.variational:
                loss = self._obtain_loss(V, t, x, p)
            else:
                loss = self._obtain_vae_loss(V, t, x, p)
            val_loss += loss.item()
        val_loss /= len(dl_val)
        return val_loss

    def train(self, dl_train, dl_val, epochs):
        val_loss = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(dl_train)
            val_loss = self.val_epoch(dl_val)
            self.scheduler.step()
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            })
            if epoch % 10 == 0:
                print(f"epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, lr: {self.optimizer.param_groups[0]['lr']}")
        return val_loss


def run(run_config: RunConfig, dl_train, dl_val, group_name=None):
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

    total_loss = 0
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

        trainer = Trainer(model, optimizer, scheduler, criterion=F.mse_loss, device=device)
        val_loss = trainer.train(dl_train, dl_val, epochs=run_config.epochs)
        total_loss += val_loss

        # Save model & configs
        run_path = f"{group_path}/{run_name}"
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        torch.save(model.state_dict(), f"{run_path}/model.pt")

        wandb.finish() # pyright: ignore

    return total_loss / len(seeds)


# ┌──────────────────────────────────────────────────────────┐
#  For Analyze
# └──────────────────────────────────────────────────────────┘
def select_group(project):
    runs_path = f"runs/{project}"
    groups = [d for d in os.listdir(runs_path) if os.path.isdir(os.path.join(runs_path, d))]
    if not groups:
        raise ValueError(f"No run groups found in {runs_path}")
    
    selected_index = survey.routines.select("Select a run group:", options=groups)
    return groups[selected_index] # pyright: ignore

def select_seed(project, group_name):
    group_path = f"runs/{project}/{group_name}"
    seeds = [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))]
    if not seeds:
        raise ValueError(f"No seeds found in {group_path}")
    
    selected_index = survey.routines.select("Select a seed:", options=seeds)
    return seeds[selected_index] # pyright: ignore

def select_device():
    devices = ['cpu'] + [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    selected_index = survey.routines.select("Select a device:", options=devices)
    return devices[selected_index] # pyright: ignore


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
        raise FileNotFoundError(f"Model file not found for {project}/{group_name}/{seed}")
    
    config = RunConfig.from_yaml(config_path)
    model = config.create_model()

    # Use weights_only option in torch.load
    state_dict = torch.load(model_path, map_location='cpu', weights_only=weights_only)
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
    study = optuna.load_study(
        study_name=study_name,
        storage=f'sqlite:///{project}.db'
    )
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
    group_name = best_trial.user_attrs['group_name']

    # Select Seed
    seed = select_seed(project_name, group_name)
    best_model, best_config = load_model(project_name, group_name, seed, weights_only=weights_only)

    return best_model, best_config
