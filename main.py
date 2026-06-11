from torch.utils.data import DataLoader

from util import run
from config import RunConfig, OptimizeConfig

import argparse


def main():
    parser = argparse.ArgumentParser()
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
        "--resume", action="store_true",
        help="Resume each seed from its latest_model.pt full-state checkpoint if present.",
    )
    args = parser.parse_args()

    # Run Config
    base_config = RunConfig.from_yaml(args.run_config)

    # Device
    if args.device:
        base_config = base_config.with_overrides(device=args.device)

    # Load data
    ds_train, ds_val = base_config.load_data()
    dl_train = DataLoader(ds_train, batch_size=base_config.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=base_config.batch_size, shuffle=False)

    # Run
    if args.optimize_config:
        if args.resume:
            print(
                "warning: --resume is ignored in HPO mode (per-trial group names "
                "depend on trial numbers, so resume is not meaningful)."
            )
        optimize_config = OptimizeConfig.from_yaml(args.optimize_config)
        pruner = optimize_config.create_pruner()

        def objective(trial, base_config, optimize_config, dl_train, dl_val):
            params = optimize_config.suggest_params(trial)

            overrides = {"project": f"{base_config.project}_Opt"}
            for category, category_params in params.items():
                overrides[category] = category_params

            run_config = base_config.with_overrides(**overrides)
            group_name = run_config.gen_group_name()
            group_name += f"[{trial.number}]"

            trial.set_user_attr("group_name", group_name)

            return run(
                run_config, dl_train, dl_val, group_name, trial=trial, pruner=pruner
            )

        study = optimize_config.create_study(project=f"{base_config.project}_Opt")
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
        run(base_config, dl_train, dl_val, resume=args.resume)


if __name__ == "__main__":
    main()
