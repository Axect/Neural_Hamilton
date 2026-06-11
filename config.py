from dataclasses import dataclass, asdict, field
import copy
import optuna
import yaml
import importlib


# ── Duplicate-key-detecting YAML loader ──────────────────────────────────────

class _DuplicateKeyLoader(yaml.SafeLoader):
    pass


def _check_duplicates(loader, node):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in mapping:
            raise yaml.YAMLError(f"Duplicate key '{key}' found in YAML config")
        mapping[key] = loader.construct_object(value_node)
    return mapping


_DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _check_duplicates
)


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 10
    mode: str = "min"  # "min" or "max"
    min_delta: float = 0.0001


@dataclass
class CheckpointConfig:
    enabled: bool = False
    save_every_n_epochs: int = 10
    keep_last_k: int = 3
    save_best: bool = True
    monitor: str = "val_loss"
    mode: str = "min"


@dataclass
class RunConfig:
    project: str
    device: str
    seeds: list[int]
    net: str
    optimizer: str
    scheduler: str
    epochs: int
    batch_size: int
    net_config: dict[str, int]
    optimizer_config: dict[str, int | float]
    scheduler_config: dict[str, int | float]
    early_stopping_config: EarlyStoppingConfig = field(
        default_factory=lambda: EarlyStoppingConfig()
    )
    criterion: str = "torch.nn.MSELoss"
    criterion_config: dict = field(default_factory=dict)
    data: str = "util.load_data"
    checkpoint_config: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig()
    )
    monitor: str = "val_loss"
    wandb: bool = True

    def __post_init__(self):
        # ── Convert dicts to dataclass instances ──
        if isinstance(self.early_stopping_config, dict):
            self.early_stopping_config = EarlyStoppingConfig(
                **self.early_stopping_config
            )
        if isinstance(self.checkpoint_config, dict):
            self.checkpoint_config = CheckpointConfig(
                **self.checkpoint_config
            )

        # ── Tier 1: Structural validation ──
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")

        for name, path in [("net", self.net), ("optimizer", self.optimizer),
                           ("scheduler", self.scheduler)]:
            if "." not in path:
                raise ValueError(
                    f"{name} must be in module.Class format (contain at least one '.'), "
                    f"got '{path}'"
                )

        if self.data and "." not in self.data:
            raise ValueError(
                f"data must be in module.function format (contain at least one '.'), "
                f"got '{self.data}'"
            )

        if not isinstance(self.wandb, bool):
            raise ValueError(
                f"wandb must be a boolean, got {type(self.wandb).__name__}"
            )

        if self.early_stopping_config.enabled:
            if self.early_stopping_config.patience <= 0:
                raise ValueError(
                    f"early_stopping patience must be > 0, got "
                    f"{self.early_stopping_config.patience}"
                )
            if self.early_stopping_config.mode not in ("min", "max"):
                raise ValueError(
                    f"early_stopping mode must be 'min' or 'max', got "
                    f"'{self.early_stopping_config.mode}'"
                )

        # ── Freeze the config ──
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name, value):
        if name == '_frozen' or not getattr(self, '_frozen', False):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                "RunConfig is frozen. Use with_overrides() to create a modified copy."
            )

    def validate_for_execution(self):
        """Tier 2 — runtime validation: check device & importlib paths."""
        # ── Device availability ──
        if self.device.startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Device '{self.device}' requested but CUDA is not available"
                )

        # ── Import path resolution ──
        for label, dotted_path in [
            ("net", self.net),
            ("optimizer", self.optimizer),
            ("scheduler", self.scheduler),
            ("criterion", self.criterion),
            ("data", self.data),
        ]:
            module_name, class_name = dotted_path.rsplit(".", 1)
            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                raise ImportError(
                    f"Cannot import module '{module_name}' for {label}: {e}"
                ) from e
            if not hasattr(module, class_name):
                raise AttributeError(
                    f"Module '{module_name}' has no attribute '{class_name}' "
                    f"(for {label})"
                )

    def validate_semantics(self) -> list[str]:
        """Tier 3 — semantic validation: check logical relationships between config values.

        Returns list of warning strings. Empty list means all checks passed.
        """
        issues = []

        # Check optimizer lr is positive
        lr = self.optimizer_config.get("lr")
        if lr is not None and lr <= 0:
            issues.append(f"optimizer_config.lr must be positive, got {lr}")

        # Check seeds are unique
        if len(self.seeds) != len(set(self.seeds)):
            issues.append(f"seeds contains duplicates: {self.seeds}")

        # Scheduler-specific checks
        scheduler_class = self.scheduler.rsplit(".", 1)[-1]

        if scheduler_class in ("ExpHyperbolicLRScheduler", "HyperbolicLRScheduler"):
            total_steps = self.scheduler_config.get("total_steps")
            upper_bound = self.scheduler_config.get("upper_bound")
            if total_steps is not None and upper_bound is not None:
                if upper_bound < total_steps:
                    issues.append(
                        f"scheduler upper_bound ({upper_bound}) must be >= "
                        f"total_steps ({total_steps}) for {scheduler_class}"
                    )

        if scheduler_class == "CosineAnnealingLR":
            t_max = self.scheduler_config.get("T_max")
            if t_max is not None and t_max != self.epochs:
                issues.append(
                    f"CosineAnnealingLR T_max ({t_max}) != epochs ({self.epochs})"
                )

        # Check early stopping patience vs epochs
        if self.early_stopping_config.enabled:
            if self.early_stopping_config.patience >= self.epochs:
                issues.append(
                    f"early_stopping patience ({self.early_stopping_config.patience}) >= "
                    f"epochs ({self.epochs}) — early stopping will never trigger"
                )

        return issues

    def with_overrides(self, **kwargs) -> 'RunConfig':
        """Return a new RunConfig with the given fields replaced (deep-merge dicts)."""
        current = copy.deepcopy(asdict(self))
        for key, value in kwargs.items():
            if isinstance(value, dict) and isinstance(current.get(key), dict):
                current[key] = {**current[key], **value}
            else:
                current[key] = value
        current.pop('_frozen', None)
        return RunConfig(**current)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as file:
            config = yaml.load(file, Loader=_DuplicateKeyLoader)
        return cls(**config)

    def to_yaml(self, path: str):
        with open(path, "w") as file:
            yaml.dump(asdict(self), file, sort_keys=False)

    def create_model(self):
        module_name, class_name = self.net.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        # Neural Hamilton models take a single hparams dict; device placement
        # happens via .to(device) in run()/preflight.
        return model_class(self.net_config)

    def create_optimizer(self, model):
        module_name, class_name = self.optimizer.rsplit(".", 1)
        module = importlib.import_module(module_name)
        optimizer_class = getattr(module, class_name)
        return optimizer_class(model.parameters(), **self.optimizer_config)

    def create_scheduler(self, optimizer):
        scheduler_module, scheduler_class = self.scheduler.rsplit(".", 1)
        scheduler_module = importlib.import_module(scheduler_module)
        scheduler_class = getattr(scheduler_module, scheduler_class)
        return scheduler_class(optimizer, **self.scheduler_config)

    def create_criterion(self):
        module_name, class_name = self.criterion.rsplit(".", 1)
        module = importlib.import_module(module_name)
        criterion_class = getattr(module, class_name)
        return criterion_class(**self.criterion_config)

    def load_data(self):
        """Load data using the configured data module path."""
        module_name, func_name = self.data.rsplit(".", 1)
        module = importlib.import_module(module_name)
        load_fn = getattr(module, func_name)
        return load_fn()

    def gen_group_name(self):
        name = f"{self.net.split('.')[-1]}"
        for k, v in self.net_config.items():
            name += f"_{k[0]}_{v}"
        name += f"_{abbreviate(self.optimizer.split('.')[-1])}"
        for k, v in self.optimizer_config.items():
            name += f"_{k[0]}_{v:.4e}" if isinstance(v, float) else f"_{k[0]}_{v}"
        name += f"_{abbreviate(self.scheduler.split('.')[-1])}"
        for k, v in self.scheduler_config.items():
            name += f"_{k[0]}_{v:.4e}" if isinstance(v, float) else f"_{k[0]}_{v}"
        return name

    def gen_tags(self):
        return [
            self.net.split(".")[-1],
            *[f"{k[0]}={v}" for k, v in self.net_config.items()],
            self.optimizer.split(".")[-1],
            self.scheduler.split(".")[-1],
        ]

    def gen_config(self):
        return asdict(self)


def default_run_config():
    return RunConfig(
        project="PyTorch_Template",
        device="cpu",
        seeds=[42],
        net="model.MLP",
        optimizer="torch.optim.adamw.AdamW",
        scheduler="torch.optim.lr_scheduler.CosineAnnealingLR",
        epochs=50,
        batch_size=256,
        net_config={
            "nodes": 128,
            "layers": 4,
        },
        optimizer_config={
            "lr": 1e-3,
        },
        scheduler_config={
            "T_max": 50,
            "eta_min": 1e-5,
        },
    )


@dataclass
class OptimizeConfig:
    study_name: str
    trials: int
    seed: int
    metric: str
    direction: str
    sampler: dict = field(default_factory=dict)
    pruner: dict = field(default_factory=dict)
    search_space: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r") as file:
            config = yaml.load(file, Loader=_DuplicateKeyLoader)
        return cls(**config)

    def to_yaml(self, path):
        with open(path, "w") as file:
            yaml.dump(asdict(self), file, sort_keys=False)

    def _create_sampler(self):
        module_name, class_name = self.sampler["name"].rsplit(".", 1)
        module = importlib.import_module(module_name)
        sampler_class = getattr(module, class_name)
        sampler_kwargs = self.sampler.get("kwargs", {})
        if class_name == "GridSampler":
            sampler_kwargs["search_space"] = self.grid_search_space()
        return sampler_class(**sampler_kwargs)

    def create_pruner(self):
        if not self.pruner:
            return None
        module_name, class_name = self.pruner["name"].rsplit(".", 1)
        module = importlib.import_module(module_name)
        pruner_class = getattr(module, class_name)
        pruner_kwargs = self.pruner.get("kwargs", {})
        return pruner_class(**pruner_kwargs)

    def create_study(self, project):
        sampler = self._create_sampler()
        study = {
            "study_name": self.study_name,
            "storage": f"sqlite:///{project}.db",
            "sampler": sampler,
            "direction": self.direction,
            "load_if_exists": True,
        }
        return optuna.create_study(**study)

    def suggest_params(self, trial):
        params = {}
        for category, config in self.search_space.items():
            params[category] = {}
            for param, param_config in config.items():
                if param_config["type"] == "int":
                    params[category][param] = trial.suggest_int(
                        f"{category}_{param}",
                        param_config["min"],
                        param_config["max"],
                        step=param_config.get("step", 1),
                    )
                elif param_config["type"] == "float":
                    if param_config.get("log", False):
                        params[category][param] = trial.suggest_float(
                            f"{category}_{param}",
                            param_config["min"],
                            param_config["max"],
                            log=True,
                        )
                    else:
                        params[category][param] = trial.suggest_float(
                            f"{category}_{param}",
                            param_config["min"],
                            param_config["max"],
                        )
                elif param_config["type"] == "categorical":
                    params[category][param] = trial.suggest_categorical(
                        f"{category}_{param}", param_config["choices"]
                    )
        return params

    def grid_search_space(self):
        params = {}
        for category, config in self.search_space.items():
            for param, param_config in config.items():
                if param_config["type"] == "categorical":
                    params[f"{category}_{param}"] = param_config["choices"]
                else:
                    raise ValueError(
                        f"Unsupported grid search space type: {param_config['type']}"
                    )
        return params


def abbreviate(s: str):
    return "".join([w for w in s if w.isupper()])
