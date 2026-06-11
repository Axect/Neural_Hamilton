from torch.utils.data import DataLoader

import optuna

from util import run
from config import RunConfig, OptimizeConfig

import argparse
import math


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

            result = run(
                run_config, dl_train, dl_val, group_name, trial=trial, pruner=pruner
            )
            # A NaN/None objective kills study.optimize with optuna's internal
            # "Should not reach" assertion; discard the trial instead.
            if result is None or not math.isfinite(result):
                raise optuna.TrialPruned(f"non-finite objective: {result}")
            return result

        study = optimize_config.create_study(project=f"{base_config.project}_Opt")
        # On resume (load_if_exists), only run the remaining budget so every
        # study ends at exactly `trials` total trials.
        n_done = len([t for t in study.trials if t.state.is_finished()])
        n_remaining = max(optimize_config.trials - n_done, 0)
        if n_done > 0:
            print(f"Resuming study with {n_done} finished trials; running {n_remaining} more.")
        study.optimize(
            lambda trial: objective(
                trial, base_config, optimize_config, dl_train, dl_val
            ),
            n_trials=n_remaining,
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
