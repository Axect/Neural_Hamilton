import platform
import sys
import os
import hashlib
import json
import time
from dataclasses import asdict
from typing import Any


def capture_environment() -> dict[str, Any]:
    """Capture full execution environment for reproducibility."""
    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
    }

    # PyTorch info
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda or "N/A"
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_devices"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_mb": round(torch.cuda.get_device_properties(i).total_memory / 1024**2),
                }
                for i in range(torch.cuda.device_count())
            ]
        env["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
    except ImportError:
        env["torch_version"] = "not installed"

    # NumPy info
    try:
        import numpy
        env["numpy_version"] = numpy.__version__
    except ImportError:
        pass

    # Git commit
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            env["git_commit"] = result.stdout.strip()
        result_dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        )
        if result_dirty.returncode == 0:
            env["git_dirty"] = len(result_dirty.stdout.strip()) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Determinism-relevant env vars
    env_vars = {}
    for var in ["PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG"]:
        val = os.environ.get(var)
        if val is not None:
            env_vars[var] = val
    if env_vars:
        env["env_vars"] = env_vars

    return env


def compute_config_hash(config) -> str:
    """Compute SHA-256 hash of canonicalized config for deduplication."""
    config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else dict(config)
    # Remove non-deterministic fields
    config_dict.pop('_frozen', None)
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def capture_run_metadata(model, device: str, start_time: float, end_time: float) -> dict[str, Any]:
    """Capture metadata about a completed training run."""
    import torch
    metadata = {
        "training_time_seconds": round(end_time - start_time, 2),
        "device": device,
    }

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata["total_parameters"] = total_params
    metadata["trainable_parameters"] = trainable_params

    # GPU memory
    if device.startswith("cuda") and torch.cuda.is_available():
        device_idx = int(device.split(":")[-1]) if ":" in device else 0
        metadata["peak_gpu_memory_mb"] = round(
            torch.cuda.max_memory_allocated(device_idx) / 1024**2, 2
        )

    return metadata


def save_provenance(run_path: str, config, model, device: str,
                    start_time: float, end_time: float) -> None:
    """Save all provenance files to the run directory."""
    import yaml

    env = capture_environment()
    config_hash = compute_config_hash(config)
    run_meta = capture_run_metadata(model, device, start_time, end_time)

    with open(os.path.join(run_path, "env_snapshot.yaml"), "w") as f:
        yaml.dump(env, f, sort_keys=False, default_flow_style=False)

    run_meta["config_hash"] = config_hash
    with open(os.path.join(run_path, "run_metadata.yaml"), "w") as f:
        yaml.dump(run_meta, f, sort_keys=False, default_flow_style=False)
