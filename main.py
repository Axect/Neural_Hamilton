import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
import survey
import optuna

from util import load_data, set_seed, select_device, Trainer, run
from config import RunConfig, OptimizeConfig

import random
import numpy as np
import argparse
import os
import copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_config", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--optimize_config", type=str, help="Path to the optimization YAML config file")
    args = parser.parse_args()

    wandb.require("core") # pyright: ignore

    # Run Config
    base_config = RunConfig.from_yaml(args.run_config)

    # Load data
    ds_train, ds_val = load_data() # pyright: ignore
    dl_train = DataLoader(ds_train, batch_size=base_config.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=base_config.batch_size)

    # Run
    if args.optimize_config:
        def objective(trial, base_config, optimize_config, dl_train, dl_val):
            params = optimize_config.suggest_params(trial)
            
            config = base_config.gen_config()
            config["project"] = f"{base_config.project}_Opt"
            for category, category_params in params.items():
                config[category].update(category_params)
            
            run_config = RunConfig(**config)
            group_name = run_config.gen_group_name()
            group_name += f"[{trial.number}]"

            trial.set_user_attr("group_name", group_name)

            return run(run_config, dl_train, dl_val, group_name)

        optimize_config = OptimizeConfig.from_yaml(args.optimize_config)
        study = optimize_config.create_study(project=base_config.project)
        study.optimize(lambda trial: objective(trial, base_config, optimize_config, dl_train, dl_val), n_trials=optimize_config.trials)

        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print(f"  Path: runs/{base_config.project}_Opt/{trial.user_attrs['group_name']}")
        
    else:
        run(base_config, dl_train, dl_val)


if __name__ == "__main__":
    main()
