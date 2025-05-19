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
        ax.set_xlabel("Total Loss")
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
        ax.set_xlabel("Total Loss")
        ax.set_ylabel("Count")
        ax.set_xscale("log")
        ax.set_xlim((1e-7, 1e-2))
        ax.set_ylim((0, 800))
        ax.legend()
        fig.savefig("figs/loss_hist_model_only.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


#def hist_times(times: List[np.ndarray], legends: List[str]):
#    min = np.min([np.min(time) for time in times])
#    max = np.max([np.max(time) for time in times])
#    min = np.log10(min)
#    max = np.log10(max)
#    if min < 0:
#        min *= 1.01
#    else:
#        min *= 0.99
#    if max < 0:
#        max *= 0.99
#    else:
#        max *= 1.01
#    bins = np.logspace(min, max, 100)
#    colors = ['gray', 'maroon', 'darkblue']
#    with plt.style.context(["science", "nature"]):
#        fig, ax = plt.subplots()
#        time_means = []
#        time_stds = []
#        time_geo_means = []
#        time_log_stds = []
#        time_medians = []
#        time_q1s = []
#        time_q3s = []
#        time_iqrs = []
#        for i, (time, legend) in enumerate(zip(times, legends)):
#            time_mean = np.mean(time)
#            time_std = np.std(time)
#            time_geo_mean = stats.gmean(time)
#            time_log_std = np.std(np.log10(time))
#            time_med = np.median(time)
#            time_q1, time_q3 = np.percentile(time, [25, 75])
#            iqr = time_q3 - time_q1
#            time_means.append(time_mean)
#            time_stds.append(time_std)
#            time_geo_means.append(time_geo_mean)
#            time_log_stds.append(time_log_std)
#            time_medians.append(time_med)
#            time_q1s.append(time_q1)
#            time_q3s.append(time_q3)
#            time_iqrs.append(iqr)
#            ax.hist(time, bins=bins, label=legend, color=colors[i], histtype="step", alpha=0.65)
#        ax.set_xlabel("Total Time")
#        ax.set_ylabel("Count")
#        ax.set_xscale("log")
#        #ax.set_yscale("log")
#        ax.legend()
#        fig.savefig("figs/time_hist.png", dpi=600, bbox_inches="tight")
#        plt.close(fig)
#        df = pl.DataFrame({
#            "time_mean": time_means,
#            "time_std": time_stds,
#            "time_geo_mean": time_geo_means,
#            "time_log_std": time_log_stds,
#            "time_med": time_medians,
#            "time_q1": time_q1s,
#            "time_q3": time_q3s,
#            "time_iqr": time_iqrs
#        })
#        print(df)


if __name__ == "__main__":
    console = Console()
    selected = choose_projects_to_plot()
    losses = losses_from_projects(selected)
    legends = ["Y4", "RK4", "DeepONet", "TraONet", "MambONet"]
    hist_losses(losses, legends)
    hist_losses_model_only(losses, legends[2:])
    #hist_times(times, legends)
