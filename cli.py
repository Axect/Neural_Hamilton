"""
Typer-based CLI entrypoint for the PyTorch Template project.

Usage:
    python cli.py train config.yaml
    python cli.py validate config.yaml
    python cli.py preview config.yaml
    python cli.py preflight config.yaml
    python cli.py doctor
    python cli.py analyze --project MyProject --group exp1 --seed 42
    python cli.py hpo-report --db MyProject_Opt.db
"""

import json as json_lib

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="PyTorch Template CLI")
console = Console()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _render_preflight_results(results: dict, json_output: bool):
    if json_output:
        console.print(json_lib.dumps(results, indent=2))
        return

    table = Table(title="Pre-flight Check", show_lines=True)
    table.add_column("Check", style="bold cyan")
    table.add_column("Status")
    table.add_column("Detail")

    for check in results["checks"]:
        status = check["status"]
        if status == "PASS":
            status_str = "[bold green]PASS[/bold green]"
        elif status == "WARN":
            status_str = "[bold yellow]WARN[/bold yellow]"
        else:
            status_str = "[bold red]FAIL[/bold red]"
        table.add_row(check["name"], status_str, check.get("detail", ""))

    console.print(table)

    if results["passed"]:
        console.print("[bold green]All pre-flight checks passed.[/bold green]")
    else:
        console.print("[bold red]Pre-flight check FAILED.[/bold red]")


def _render_hpo_report(report: dict):
    console.print(f"\n[bold]Study:[/bold] {report['study_name']} ({report['db']})")

    stats = report["stats"]
    console.print(
        f"Trials: {stats['total']} total, "
        f"[green]{stats['completed']} completed[/green], "
        f"[yellow]{stats['pruned']} pruned[/yellow], "
        f"[red]{stats['failed']} failed[/red]"
    )

    best = report["best_trial"]
    console.print(f"\n[bold green]Best Trial #{best['number']}[/bold green]")
    console.print(f"  Value: {best['value']}")
    console.print(f"  Group: {best['group_name']}")
    for k, v in best["params"].items():
        console.print(f"  {k}: {v}")

    if report["importances"]:
        imp_table = Table(title="Parameter Importance", show_lines=True)
        imp_table.add_column("Parameter", style="bold cyan")
        imp_table.add_column("Importance")
        for param, importance in report["importances"].items():
            bar = "\u2588" * int(importance * 30)
            imp_table.add_row(param, f"{importance:.4f} {bar}")
        console.print(imp_table)

    if report["boundary_warnings"]:
        console.print("\n[bold yellow]Boundary Warnings:[/bold yellow]")
        for warn in report["boundary_warnings"]:
            console.print(f"  [yellow]{warn}[/yellow]")

    if report["top_k"]:
        top_table = Table(title=f"Top {len(report['top_k'])} Trials", show_lines=True)
        top_table.add_column("#", style="bold")
        top_table.add_column("Value")
        param_keys = list(report["top_k"][0]["params"].keys())
        for key in param_keys:
            top_table.add_column(key)
        for t in report["top_k"]:
            row = [str(t["number"]), f"{t['value']:.6f}"]
            for key in param_keys:
                val = t["params"].get(key, "")
                row.append(f"{val:.4e}" if isinstance(val, float) else str(val))
            top_table.add_row(*row)
        console.print(top_table)


@app.command()
def train(
    run_config: str = typer.Argument(..., help="Path to the YAML config file"),
    device: str = typer.Option(None, help="Device override (e.g. 'cuda:0' or 'cpu')"),
    optimize_config: str = typer.Option(None, help="Path to optimization config for HPO"),
    resume: bool = typer.Option(
        False, "--resume",
        help="Resume each seed from its latest_model.pt full-state checkpoint if present.",
    ),
):
    """Train a model using the given run configuration."""
    try:
        from torch.utils.data import DataLoader

        from config import RunConfig, OptimizeConfig
        from util import run

        base_config = RunConfig.from_yaml(run_config)

        if device:
            base_config = base_config.with_overrides(device=device)

        ds_train, ds_val = base_config.load_data()
        dl_train = DataLoader(ds_train, batch_size=base_config.batch_size, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=base_config.batch_size, shuffle=False)

        if optimize_config:
            if resume:
                console.print(
                    "[yellow]warning:[/yellow] --resume is ignored in HPO mode "
                    "(group names depend on trial numbers, so per-trial resume "
                    "is not meaningful)."
                )
            opt_config = OptimizeConfig.from_yaml(optimize_config)
            pruner = opt_config.create_pruner()

            def objective(trial, base_config, opt_config, dl_train, dl_val):
                params = opt_config.suggest_params(trial)

                overrides = {"project": f"{base_config.project}_Opt"}
                for category, category_params in params.items():
                    overrides[category] = category_params

                trial_config = base_config.with_overrides(**overrides)
                group_name = trial_config.gen_group_name()
                group_name += f"[{trial.number}]"

                trial.set_user_attr("group_name", group_name)

                return run(
                    trial_config, dl_train, dl_val, group_name, trial=trial, pruner=pruner
                )

            study = opt_config.create_study(project=f"{base_config.project}_Opt")
            study.optimize(
                lambda trial: objective(
                    trial, base_config, opt_config, dl_train, dl_val
                ),
                n_trials=opt_config.trials,
            )

            console.print("\n[bold green]Best trial:[/bold green]")
            trial = study.best_trial
            console.print(f"  Value: {trial.value}")
            console.print("  Params:")
            for key, value in trial.params.items():
                console.print(f"    {key}: {value}")
            console.print(
                f"  Path: runs/{base_config.project}_Opt/{trial.user_attrs['group_name']}"
            )
        else:
            run(base_config, dl_train, dl_val, resume=resume)
            console.print("[bold green]Training complete.[/bold green]")

    except Exception:
        console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def validate(
    run_config: str = typer.Argument(..., help="Path to the YAML config file"),
):
    """Validate a run configuration for structural and runtime correctness."""
    try:
        from config import RunConfig

        config = RunConfig.from_yaml(run_config)
        config.validate_for_execution()
        console.print(f"[bold green]PASS[/bold green] — config '{run_config}' is valid.")

    except Exception as e:
        console.print(f"[bold red]FAIL[/bold red] — {e}")
        raise typer.Exit(code=1)


@app.command()
def preflight(
    run_config: str = typer.Argument(..., help="Path to the YAML config file"),
    device: str = typer.Option(None, help="Device override (e.g. 'cuda:0' or 'cpu')"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
):
    """Pre-flight check: validate config, instantiate objects, run 1 batch forward+backward."""
    import torch
    from torch.utils.data import DataLoader
    from config import RunConfig

    results: dict = {"checks": [], "passed": True, "gpu_memory_mb": None}

    def add_check(name, status, detail=""):
        results["checks"].append({"name": name, "status": status, "detail": detail})
        if status == "FAIL":
            results["passed"] = False

    try:
        config = RunConfig.from_yaml(run_config)
        if device:
            config = config.with_overrides(device=device)
    except Exception as e:
        add_check("Config loading", "FAIL", str(e))
        _render_preflight_results(results, json_output)
        raise typer.Exit(code=1)

    # Check 1: Import paths & device
    try:
        config.validate_for_execution()
        add_check("Import paths & device", "PASS")
    except Exception as e:
        add_check("Import paths & device", "FAIL", str(e))
        _render_preflight_results(results, json_output)
        raise typer.Exit(code=1)

    # Check 2: Semantic validation
    issues = config.validate_semantics()
    if issues:
        for issue in issues:
            add_check("Semantic", "WARN", issue)
    else:
        add_check("Semantic validation", "PASS")

    # Check 3: Object instantiation
    try:
        model = config.create_model()
        optimizer = config.create_optimizer(model)
        scheduler = config.create_scheduler(optimizer)
        criterion = config.create_criterion()
        add_check("Object instantiation", "PASS")
    except Exception as e:
        add_check("Object instantiation", "FAIL", str(e))
        _render_preflight_results(results, json_output)
        raise typer.Exit(code=1)

    # Check 4: Data loading
    try:
        ds_train, ds_val = config.load_data()
        dl_train = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)
        add_check("Data loading", "PASS", f"train={len(ds_train)}, val={len(ds_val)}")
    except Exception as e:
        add_check("Data loading", "FAIL", str(e))
        _render_preflight_results(results, json_output)
        raise typer.Exit(code=1)

    # Check 5: Forward + backward pass
    dev = config.device
    try:
        model = model.to(dev)
        if dev.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        # Neural Hamilton batch: (V, t, q, p, ic) with model(V, t, ic) -> (q, p)
        V, t, q, p, ic = next(iter(dl_train))
        V, t, q, p, ic = V.to(dev), t.to(dev), q.to(dev), p.to(dev), ic.to(dev)

        model.train()
        q_pred, p_pred = model(V, t, ic)
        loss = 0.5 * (criterion(q_pred, q) + criterion(p_pred, p))
        add_check("Forward pass", "PASS",
                   f"output=({tuple(q_pred.shape)}, {tuple(p_pred.shape)}), loss={loss.item():.6f}")

        if q_pred.shape != q.shape or p_pred.shape != p.shape:
            add_check("Shape check", "WARN",
                       f"model output ({tuple(q_pred.shape)}, {tuple(p_pred.shape)}) vs target ({tuple(q.shape)}, {tuple(p.shape)})")
        else:
            add_check("Shape check", "PASS")

        optimizer.zero_grad()
        loss.backward()

        # Gradient check
        total_grad_norm = 0.0
        has_bad_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    has_bad_grad = True
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        if has_bad_grad:
            add_check("Gradient check", "FAIL", "NaN/Inf detected in gradients")
        else:
            add_check("Gradient check", "PASS", f"grad norm={total_grad_norm:.6f}")

        optimizer.step()
        add_check("Optimizer step", "PASS")

    except Exception as e:
        add_check("Forward/backward pass", "FAIL", str(e))

    # Check 6: Scheduler step
    try:
        scheduler.step()
        add_check("Scheduler step", "PASS")
    except Exception as e:
        add_check("Scheduler step", "FAIL", str(e))

    # GPU memory
    if dev.startswith("cuda"):
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        results["gpu_memory_mb"] = round(peak_mem, 2)
        add_check("GPU memory", "PASS", f"peak={peak_mem:.1f} MB (1 batch)")

    _render_preflight_results(results, json_output)
    if not results["passed"]:
        raise typer.Exit(code=1)


@app.command()
def preview(
    run_config: str = typer.Argument(..., help="Path to the YAML config file"),
):
    """Preview the model, optimizer, scheduler, and criterion from a config."""
    try:
        from config import RunConfig

        config = RunConfig.from_yaml(run_config)

        model = config.create_model()
        optimizer = config.create_optimizer(model)
        scheduler = config.create_scheduler(optimizer)
        criterion = config.create_criterion()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        table = Table(title="Run Preview", show_lines=True)
        table.add_column("Property", style="bold cyan")
        table.add_column("Value")

        table.add_row("Model", repr(model))
        table.add_row("Total parameters", f"{total_params:,}")
        table.add_row("Trainable parameters", f"{trainable_params:,}")
        table.add_row("Optimizer", f"{type(optimizer).__name__} {optimizer.defaults}")
        table.add_row("Scheduler", f"{type(scheduler).__name__} {scheduler.state_dict()}")
        table.add_row("Criterion", f"{type(criterion).__name__}")
        table.add_row("Device", config.device)

        console.print(table)

    except Exception:
        console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def doctor():
    """Check system environment: Python, PyTorch, CUDA, packages, and wandb."""
    from provenance import capture_environment

    env = capture_environment()

    table = Table(title="System Doctor", show_lines=True)
    table.add_column("Check", style="bold cyan")
    table.add_column("Status")

    # Python version
    table.add_row("Python version", f"[green]{env['python_version']}[/green]")

    # PyTorch version
    torch_ver = env.get("torch_version", "not installed")
    if torch_ver == "not installed":
        table.add_row("PyTorch", "[red]not installed[/red]")
    else:
        table.add_row("PyTorch", f"[green]{torch_ver}[/green]")

    # CUDA
    cuda_available = env.get("cuda_available", False)
    if cuda_available:
        cuda_ver = env.get("cuda_version", "N/A")
        table.add_row("CUDA", f"[green]available (v{cuda_ver})[/green]")
        gpu_devices = env.get("gpu_devices", [])
        for i, gpu in enumerate(gpu_devices):
            table.add_row(
                f"  GPU {i}",
                f"[green]{gpu['name']} ({gpu['memory_total_mb']} MB)[/green]",
            )
    else:
        table.add_row("CUDA", "[yellow]not available[/yellow]")

    # wandb login status
    try:
        import wandb
        api_key = wandb.api.api_key
        if api_key:
            table.add_row("wandb", "[green]logged in[/green]")
        else:
            table.add_row("wandb", "[red]not logged in[/red]")
    except Exception:
        table.add_row("wandb", "[red]not installed or error[/red]")

    # Required packages
    required_packages = [
        "torch", "numpy", "optuna", "wandb", "tqdm", "rich", "beaupy", "scienceplots",
    ]
    for pkg in required_packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "ok")
            table.add_row(f"  {pkg}", f"[green]{version}[/green]")
        except ImportError:
            table.add_row(f"  {pkg}", "[red]missing[/red]")

    console.print(table)


@app.command(name="hpo-report")
def hpo_report(
    db: str = typer.Option(None, help="Path to Optuna SQLite database"),
    study_name: str = typer.Option(None, help="Study name within the database"),
    opt_config: str = typer.Option(None, help="Path to optimization YAML (for boundary check)"),
    top_k: int = typer.Option(5, help="Number of top trials to show"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Analyze HPO results: best trial, parameter importance, boundary warnings."""
    try:
        import glob as glob_mod
        import optuna
        from config import OptimizeConfig

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Auto-detect DB
        if db is None:
            db_files = glob_mod.glob("*.db")
            if len(db_files) == 1:
                db = db_files[0]
            elif not db_files:
                console.print("[red]No .db files found. Use --db to specify.[/red]")
                raise typer.Exit(code=1)
            else:
                console.print(f"[yellow]Multiple .db files: {db_files}. Use --db.[/yellow]")
                raise typer.Exit(code=1)

        storage = f"sqlite:///{db}"

        # Auto-detect study
        if study_name is None:
            summaries = optuna.study.get_all_study_summaries(storage)
            if len(summaries) == 1:
                study_name = summaries[0].study_name
            elif not summaries:
                console.print("[red]No studies found in database.[/red]")
                raise typer.Exit(code=1)
            else:
                names = [s.study_name for s in summaries]
                console.print(f"[yellow]Multiple studies: {names}. Use --study-name.[/yellow]")
                raise typer.Exit(code=1)

        study = optuna.load_study(study_name=study_name, storage=storage)

        # Stats
        trials = study.trials
        completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
        failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]

        if not completed:
            console.print("[red]No completed trials found.[/red]")
            raise typer.Exit(code=1)

        best = study.best_trial

        # Parameter importance
        importances: dict = {}
        if len(completed) >= 2:
            try:
                importances = optuna.importance.get_param_importances(study)
            except Exception:
                pass

        # Boundary check
        boundary_warnings: list[str] = []
        if opt_config:
            opt = OptimizeConfig.from_yaml(opt_config)
            for category, params in opt.search_space.items():
                for param_name, param_cfg in params.items():
                    full_key = f"{category}_{param_name}"
                    if full_key in best.params:
                        best_val = best.params[full_key]
                        if param_cfg["type"] in ("int", "float"):
                            low, high = param_cfg["min"], param_cfg["max"]
                            range_size = high - low
                            if range_size > 0:
                                margin = range_size * 0.05
                                if best_val <= low + margin:
                                    boundary_warnings.append(
                                        f"{full_key}={best_val} at LOWER boundary "
                                        f"[{low}, {high}]"
                                    )
                                elif best_val >= high - margin:
                                    boundary_warnings.append(
                                        f"{full_key}={best_val} at UPPER boundary "
                                        f"[{low}, {high}]"
                                    )

        # Top-K
        sorted_trials = sorted(completed, key=lambda t: t.value)
        top_trials = sorted_trials[:top_k]

        report = {
            "study_name": study_name,
            "db": db,
            "stats": {
                "total": len(trials),
                "completed": len(completed),
                "pruned": len(pruned),
                "failed": len(failed),
            },
            "best_trial": {
                "number": best.number,
                "value": best.value,
                "params": best.params,
                "group_name": best.user_attrs.get("group_name", "N/A"),
            },
            "importances": importances,
            "boundary_warnings": boundary_warnings,
            "top_k": [
                {"number": t.number, "value": t.value, "params": t.params}
                for t in top_trials
            ],
        }

        if json_output:
            console.print(json_lib.dumps(report, indent=2, default=str))
        else:
            _render_hpo_report(report)

    except typer.Exit:
        raise
    except Exception:
        console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def analyze(
    project: str = typer.Option(None, help="Project name (non-interactive)"),
    group: str = typer.Option(None, help="Group name (non-interactive)"),
    seed: str = typer.Option(None, help="Seed (non-interactive)"),
    device: str = typer.Option("cpu", help="Device for analysis"),
):
    """Analyze a trained model — interactive or non-interactive."""
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader

        from util import (
            select_project,
            select_group,
            select_seed,
            load_model,
        )

        if project is None:
            console.print("Select a project:")
            project = select_project()
        if group is None:
            console.print("Select a group:")
            group = select_group(project)
        if seed is None:
            console.print("Select a seed:")
            seed = select_seed(project, group)

        console.print(
            f"[bold]Analyzing[/bold] {project}/{group}/{seed} on [cyan]{device}[/cyan]"
        )

        model, config = load_model(project, group, seed)
        model = model.to(device)
        model.eval()

        _, ds_val = config.load_data()
        dl_val = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False)

        total_loss = 0.0
        count = 0
        with torch.inference_mode():
            for V, t, q, p, ic in dl_val:
                V, t, q, p, ic = (
                    V.to(device), t.to(device), q.to(device), p.to(device), ic.to(device),
                )
                q_pred, p_pred = model(V, t, ic)
                loss = 0.5 * (F.mse_loss(q_pred, q) + F.mse_loss(p_pred, p))
                total_loss += loss.item()
                count += 1

        avg_loss = total_loss / count if count > 0 else float("inf")
        console.print(f"Validation MSE Loss: [bold]{avg_loss:.6f}[/bold]")

    except Exception:
        console.print_exception()
        raise typer.Exit(code=1)


def _list_runs():
    """Scan runs/ for metrics.csv files and return metadata sorted by recency."""
    import os
    import csv as csv_mod
    import glob as glob_mod
    from datetime import datetime

    candidates = glob_mod.glob("runs/**/metrics.csv", recursive=True)
    if not candidates:
        return []

    runs = []
    for csv_path in candidates:
        parts = csv_path.replace("\\", "/").split("/")
        # Expected: runs/{project}/{group}/{seed}/metrics.csv
        if len(parts) >= 5:
            project, group, seed = parts[1], parts[2], parts[3]
        else:
            project, group, seed = "?", "?", "?"

        mtime = os.path.getmtime(csv_path)
        updated = datetime.fromtimestamp(mtime)

        # Read epoch count from last row
        epochs = 0
        try:
            with open(csv_path, newline="") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    epochs = int(float(row.get("epoch", 0))) + 1
        except Exception:
            pass

        # Detect extra columns
        extra_cols = []
        known = {"epoch", "train_loss", "val_loss", "lr", "max_grad_norm", "predicted_final_loss"}
        try:
            with open(csv_path, newline="") as f:
                reader = csv_mod.DictReader(f)
                extra_cols = [h for h in (reader.fieldnames or []) if h not in known]
        except Exception:
            pass

        runs.append({
            "path": csv_path,
            "project": project,
            "group": group,
            "seed": seed,
            "updated": updated,
            "epochs": epochs,
            "extra_cols": extra_cols,
        })

    runs.sort(key=lambda r: r["updated"], reverse=True)
    return runs


@app.command()
def monitor(
    path: str = typer.Argument(
        None, help="Path to metrics.csv or its parent directory"
    ),
    interval: int = typer.Option(500, help="Refresh interval in milliseconds"),
    list_runs: bool = typer.Option(False, "--list", help="List available runs"),
    hpo: bool = typer.Option(False, "--hpo", help="HPO monitor mode"),
    y_min: float = typer.Option(None, "--y-min", help="Y-axis minimum for objective charts"),
    y_max: float = typer.Option(None, "--y-max", help="Y-axis maximum for objective charts"),
):
    """Launch the real-time TUI training monitor (Rust binary)."""
    import os
    import subprocess
    import glob as glob_mod

    if hpo:
        # Auto-detect DB
        db_files = glob_mod.glob("*.db")
        if len(db_files) == 1:
            db_path = db_files[0]
        elif not db_files:
            console.print("[red]No .db files found. Run HPO first.[/red]")
            raise typer.Exit(code=1)
        else:
            from beaupy import select
            selected = select(db_files, cursor_index=0, return_index=True)
            if selected is None:
                return
            db_path = db_files[selected]

        console.print(f"[dim]Using DB: {db_path}[/dim]")

        monitor_bin = os.path.join(os.path.dirname(__file__), "tools", "monitor", "target", "release", "training-monitor")
        if not os.path.exists(monitor_bin):
            console.print("[yellow]Monitor binary not found. Building...[/yellow]")
            cargo_dir = os.path.join(os.path.dirname(__file__), "tools", "monitor")
            result = subprocess.run(["cargo", "build", "--release"], cwd=cargo_dir)
            if result.returncode != 0:
                console.print("[red]Failed to build monitor. Install Rust: https://rustup.rs[/red]")
                raise typer.Exit(code=1)

        try:
            cmd = [monitor_bin, "--hpo", db_path, "--interval", str(interval)]
            if y_min is not None:
                cmd.extend(["--y-min", str(y_min)])
            if y_max is not None:
                cmd.extend(["--y-max", str(y_max)])
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass
        return

    if list_runs:
        runs = _list_runs()
        if not runs:
            console.print("[red]No metrics.csv found under runs/.[/red]")
            raise typer.Exit(code=1)

        table = Table(title="Available Runs", show_lines=True)
        table.add_column("#", style="bold", justify="right")
        table.add_column("Project", style="cyan")
        table.add_column("Group")
        table.add_column("Seed", justify="right")
        table.add_column("Epochs", justify="right")
        table.add_column("Updated")
        table.add_column("Extra Metrics", style="dim")

        from datetime import datetime
        now = datetime.now()
        for i, run in enumerate(runs, 1):
            delta = now - run["updated"]
            if delta.total_seconds() < 60:
                age = f"{int(delta.total_seconds())}s ago"
            elif delta.total_seconds() < 3600:
                age = f"{int(delta.total_seconds() // 60)}m ago"
            elif delta.total_seconds() < 86400:
                age = f"{int(delta.total_seconds() // 3600)}h ago"
            else:
                age = run["updated"].strftime("%Y-%m-%d")

            max_show = 5
            if run["extra_cols"]:
                cols = run["extra_cols"]
                if len(cols) <= max_show:
                    extras = ", ".join(cols)
                else:
                    extras = ", ".join(cols[:max_show]) + f", ... (+{len(cols) - max_show} more)"
            else:
                extras = "-"
            table.add_row(
                str(i), run["project"], run["group"], run["seed"],
                str(run["epochs"]), age, extras,
            )

        console.print(table)

        # Interactive selection
        from beaupy import select
        choices = [
            f"{run['project']}/{run['group']}/{run['seed']} ({run['epochs']} ep)"
            for run in runs
        ]
        selected = select(choices, cursor_index=0, return_index=True)
        if selected is None:
            return
        path = runs[selected]["path"]
        console.print(f"[dim]Selected: {path}[/dim]")

    monitor_bin = os.path.join(os.path.dirname(__file__), "tools", "monitor", "target", "release", "training-monitor")

    if not os.path.exists(monitor_bin):
        console.print("[yellow]Monitor binary not found. Building...[/yellow]")
        cargo_dir = os.path.join(os.path.dirname(__file__), "tools", "monitor")
        result = subprocess.run(["cargo", "build", "--release"], cwd=cargo_dir)
        if result.returncode != 0:
            console.print("[red]Failed to build monitor. Install Rust: https://rustup.rs[/red]")
            raise typer.Exit(code=1)

    if path is None:
        # Auto-detect: find the most recently modified metrics.csv under runs/
        candidates = glob_mod.glob("runs/**/metrics.csv", recursive=True)
        if not candidates:
            console.print("[red]No metrics.csv found under runs/. Specify a path.[/red]")
            raise typer.Exit(code=1)
        path = max(candidates, key=os.path.getmtime)
        console.print(f"[dim]Auto-detected: {path}[/dim]")

    try:
        subprocess.run([monitor_bin, path, "--interval", str(interval)])
    except KeyboardInterrupt:
        pass


@app.command(name="update-skills")
def update_skills(
    copy: bool = typer.Option(
        False, "--copy", help="Copy files instead of symlinking"
    ),
    uninstall: bool = typer.Option(
        False, "--uninstall", help="Remove globally installed skills"
    ),
    agent: str = typer.Option(
        "claude",
        "--agent",
        help="Target agent: claude, codex, or forge",
    ),
):
    """Install or update agent skills for Claude, Codex, or Forge."""
    import shutil
    from pathlib import Path

    template_dir = Path(__file__).resolve().parent
    skills_src = template_dir / "skills"

    agent = agent.lower()
    agent_dirs = {
        "claude": Path.home() / ".claude" / "skills",
        "codex": Path.home() / ".codex" / "skills",
        "forge": Path.home() / "forge" / "skills",
    }
    agent_labels = {
        "claude": "Claude Code",
        "codex": "Codex",
        "forge": "Forge",
    }

    if agent not in agent_dirs:
        valid_agents = ", ".join(agent_dirs)
        console.print(
            f"[red]Unsupported agent '{agent}'. Choose one of: {valid_agents}[/red]"
        )
        raise typer.Exit(code=1)

    if not skills_src.is_dir():
        console.print(f"[red]Skill source directory not found: {skills_src}[/red]")
        raise typer.Exit(code=1)

    global_dir = agent_dirs[agent]
    agent_label = agent_labels[agent]

    skill_names = ["pytorch-train", "pytorch-migrate"]

    if uninstall:
        for name in skill_names:
            dest = global_dir / name
            if dest.is_symlink():
                dest.unlink()
                console.print(f"  [red]Removed[/red] symlink {dest}")
            elif dest.is_dir():
                shutil.rmtree(dest)
                console.print(f"  [red]Removed[/red] directory {dest}")
            else:
                console.print(f"  [dim]{name}: not installed[/dim]")
        console.print(f"[dim]{agent_label} skill directory: {global_dir}[/dim]")
        return

    global_dir.mkdir(parents=True, exist_ok=True)

    table = Table(title=f"Skill Installation ({agent_label})", show_lines=True)
    table.add_column("Skill", style="bold cyan")
    table.add_column("Status")
    table.add_column("Method")

    for name in skill_names:
        src = skills_src / name
        dest = global_dir / name

        if not src.is_dir():
            table.add_row(name, "[red]Source not found[/red]", str(src))
            continue

        # Determine current state
        if dest.is_symlink():
            current_target = dest.resolve()
            if current_target == src.resolve():
                if copy:
                    # Replace symlink with copy
                    dest.unlink()
                    shutil.copytree(src, dest)
                    table.add_row(name, "[green]Updated[/green]", "copy (was symlink)")
                else:
                    table.add_row(name, "[green]Up to date[/green]", f"symlink → {src}")
            else:
                # Symlink to wrong target
                dest.unlink()
                if copy:
                    shutil.copytree(src, dest)
                    table.add_row(name, "[green]Updated[/green]", "copy")
                else:
                    dest.symlink_to(src)
                    table.add_row(name, "[green]Relinked[/green]", f"symlink → {src}")
        elif dest.is_dir():
            # Existing copy — replace
            shutil.rmtree(dest)
            if copy:
                shutil.copytree(src, dest)
                table.add_row(name, "[green]Updated[/green]", "copy (replaced)")
            else:
                dest.symlink_to(src)
                table.add_row(name, "[green]Updated[/green]", f"symlink (was copy)")
        else:
            # Fresh install
            if copy:
                shutil.copytree(src, dest)
                table.add_row(name, "[green]Installed[/green]", "copy")
            else:
                dest.symlink_to(src)
                table.add_row(name, "[green]Installed[/green]", f"symlink → {src}")

    console.print(table)
    console.print(f"[dim]{agent_label} skill directory: {global_dir}[/dim]")

    if not copy:
        console.print(
            "[dim]Skills are symlinked — they auto-update when you git pull.[/dim]"
        )
    else:
        console.print(
            "[dim]Skills are copied — re-run this command after git pull to update.[/dim]"
        )


if __name__ == "__main__":
    app()
