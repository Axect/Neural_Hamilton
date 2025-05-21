import polars as pl
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import scienceplots
import os
import beaupy
from rich.console import Console
from typing import List
import argparse


def choose_projects_to_plot():
    project_names = []
    
    # List all folders in figs
    for d in os.listdir("figs"):
        if os.path.isdir(os.path.join("figs", d)):
            project_names.append(os.path.basename(d))

    # Sort the project names
    project_names.sort()

    console.print("Choose projects to draw histogram")
    selected_projects = beaupy.select_multiple(
        project_names
    )

    selected_projects = [os.path.join("figs", d) for d in selected_projects] #pyright:ignore

    return selected_projects


def losses_from_projects(projects: List[str]):
    losses = []
    df = pl.read_parquet(os.path.join(projects[0], "losses.parquet"))
    losses.append(df["y4_loss"].to_numpy())
    losses.append(df["rk4_loss"].to_numpy())
    for project in projects:
        df = pl.read_parquet(os.path.join(project, "losses.parquet"))
        losses.append(df["model_loss"].to_numpy())
    return losses


def hist_losses(losses: List[np.ndarray], legends: List[str]):
    min = np.min([np.min(loss) for loss in losses])
    max = np.max([np.max(loss) for loss in losses])
    min = np.log10(min)
    max = np.log10(max)
    if min < 0:
        min *= 1.01
    else:
        min *= 0.99
    if max < 0:
        max *= 0.99
    else:
        max *= 1.01
    bins = np.logspace(-12, -2, 100, base=10)
    colors = ['gray', 'orange', 'darkred', 'darkgreen', 'darkblue']
    alphas = [0.5, 0.5, 0.65, 0.65, 0.65]
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        loss_means = []
        loss_stds = []
        loss_geo_means = []
        loss_log_stds = []
        loss_medians = []
        loss_q1s = []
        loss_q3s = []
        loss_iqrs = []
        for i, (loss, legend) in enumerate(zip(losses, legends)):
            loss_mean = np.mean(loss)
            loss_std = np.std(loss)
            loss_geo_mean = stats.gmean(loss)
            loss_log_std = np.std(np.log10(loss))
            loss_med = np.median(loss)
            loss_q1, loss_q3 = np.percentile(loss, [25, 75])
            iqr = loss_q3 - loss_q1
            loss_means.append(loss_mean)
            loss_stds.append(loss_std)
            loss_geo_means.append(loss_geo_mean)
            loss_log_stds.append(loss_log_std)
            loss_medians.append(loss_med)
            loss_q1s.append(loss_q1)
            loss_q3s.append(loss_q3)
            loss_iqrs.append(iqr)
            ax.hist(loss, bins=bins, label=legend, color=colors[i], histtype="step", alpha=0.65)
        ax.set_xlabel("Test Loss")
        ax.set_ylabel("Count")
        ax.set_xscale("log")
        ax.set_xlim((1e-12, 1e-2))
        ax.set_ylim((0, 800))
        ax.legend()
        fig.savefig("figs/loss_hist.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
        df = pd.DataFrame({
            "loss_mean": loss_means,
            "loss_std": loss_stds,
            "loss_geo_mean": loss_geo_means,
            "loss_log_std": loss_log_stds,
            "loss_med": loss_medians,
            "loss_q1": loss_q1s,
            "loss_q3": loss_q3s,
            "loss_iqr": loss_iqrs
        })
        pd.set_option('display.float_format', lambda x: '%.4e' % x)
        print(df)


def hist_losses_model_only(losses: List[np.ndarray], legends: List[str]):
    losses = losses[2:]
    min = np.min([np.min(loss) for loss in losses])
    max = np.max([np.max(loss) for loss in losses])
    min = np.log10(min)
    max = np.log10(max)
    if min < 0:
        min *= 1.01
    else:
        min *= 0.99
    if max < 0:
        max *= 0.99
    else:
        max *= 1.01
    bins = np.logspace(-12, -2, 100, base=10)
    colors = ['darkred', 'darkgreen', 'darkblue']
    alphas = [0.65, 0.65, 0.65]
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        for i, (loss, legend) in enumerate(zip(losses, legends)):
            ax.hist(loss, bins=bins, label=legend, color=colors[i], histtype="step", alpha=alphas[i])
        ax.set_xlabel("Test Loss")
        ax.set_ylabel("Count")
        ax.set_xscale("log")
        ax.set_xlim((1e-8, 1e-2))
        ax.set_ylim((0, 900))
        ax.legend()
        fig.savefig("figs/loss_hist_scale.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot loss scale")
    parser.add_argument("--model", type=str, default="TraONet", help="Model name")
    args = parser.parse_args()

    console = Console()
    selected = choose_projects_to_plot()
    losses = losses_from_projects(selected)
    #legends = ["Y4", "RK4", "DeepONet", "TraONet", "MambONet"]
    #hist_losses(losses, legends)
    legends = [f"{args.model} (10k)", f"{args.model} (100k)", f"{args.model} (1M)"]
    hist_losses_model_only(losses, legends)
    #hist_times(times, legends)
