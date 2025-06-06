from torch.utils.data import DataLoader
import torch
import wandb

from util import load_data, run
from config import RunConfig, OptimizeConfig

import argparse


torch.set_float32_matmul_precision("medium")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, required=True, help="normal or more or much?"
    )
    parser.add_argument(
        "--run_config", type=str, required=True, help="Path to the YAML config file"
    )
    parser.add_argument(
        "--optimize_config", type=str, help="Path to the optimization YAML config file"
    )
    parser.add_argument(
        "--device", type=str, help="Device to run on (e.g. 'cuda:0' or 'cpu')"
    )
    parser.add_argument(
        "--true", action="store_true", help="Use true labels for evaluation"
    )
    args = parser.parse_args()

    # Run Config
    base_config = RunConfig.from_yaml(args.run_config)

    # Device
    if args.device:
        base_config.device = args.device

    # Load data
    if args.true:
        data_folder = f"data_true"
        ds_train = load_data(f"{data_folder}/train_{args.data}_kl8.parquet")
        ds_val = load_data(f"{data_folder}/val_{args.data}_kl8.parquet")
    else:
        data_folder = f"data_{args.data}"
        ds_train = load_data(f"{data_folder}/train.parquet")
        ds_val = load_data(f"{data_folder}/val.parquet")
    dl_train = DataLoader(ds_train, batch_size=base_config.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=base_config.batch_size, shuffle=False)

    # Run
    if args.optimize_config:
        optimize_config = OptimizeConfig.from_yaml(args.optimize_config)
        pruner = optimize_config.create_pruner()

        def objective(trial, base_config, optimize_config, dl_train, dl_val):
            params = optimize_config.suggest_params(trial)

            config = base_config.gen_config()
            config["project"] = f"{base_config.project}_Opt"
            for category, category_params in params.items():
                config[category].update(category_params)

            run_config = RunConfig(**config)
            group_name = run_config.gen_group_name(args.data)
            group_name += f"[{trial.number}]"

            trial.set_user_attr("group_name", group_name)

            return run(
                run_config, dl_train, dl_val, group_name, trial=trial, pruner=pruner
            )

        study = optimize_config.create_study(project=base_config.project)
        study.optimize(
            lambda trial: objective(
                trial, base_config, optimize_config, dl_train, dl_val
            ),
            n_trials=optimize_config.trials,
        )

        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print(
            f"  Path: runs/{base_config.project}_Opt/{trial.user_attrs['group_name']}"
        )

    else:
        run(base_config, dl_train, dl_val, group_name=None, data=args.data)


if __name__ == "__main__":
    main()
