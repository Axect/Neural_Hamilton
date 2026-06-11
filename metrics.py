from __future__ import annotations
import torch
import importlib


class Metric:
    """Base class for metrics."""
    name: str = "metric"

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        raise NotImplementedError


class MSEMetric(Metric):
    name = "mse"
    def __call__(self, y_pred, y_true):
        return torch.nn.functional.mse_loss(y_pred, y_true).item()


class MAEMetric(Metric):
    name = "mae"
    def __call__(self, y_pred, y_true):
        return torch.nn.functional.l1_loss(y_pred, y_true).item()


class R2Metric(Metric):
    name = "r2"
    def __call__(self, y_pred, y_true):
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return (1 - ss_res / ss_tot).item()


class MetricRegistry:
    """Registry that holds and computes multiple metrics."""

    BUILTIN = {
        "mse": MSEMetric,
        "mae": MAEMetric,
        "r2": R2Metric,
    }

    def __init__(self, metric_names: list[str] | None = None):
        self.metrics: list[Metric] = []
        if metric_names:
            for name in metric_names:
                if name in self.BUILTIN:
                    self.metrics.append(self.BUILTIN[name]())
                else:
                    # Try importlib path
                    module_name, class_name = name.rsplit(".", 1)
                    module = importlib.import_module(module_name)
                    metric_class = getattr(module, class_name)
                    self.metrics.append(metric_class())

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> dict[str, float]:
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric(y_pred, y_true)
        return results
