import polars as pl
import numpy as np
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

    console.print("Choose projects to draw histogram")
    selected_projects = beaupy.select_multiple(
        project_names
    )

    selected_projects = [os.path.join("figs", d) for d in selected_projects] #pyright:ignore

    return selected_projects


def losses_from_projects(projects: List[str]):
    losses = []
    df = pl.read_parquet(os.path.join(projects[0], "00_0_Loss_rk4_hist.parquet"))
    losses.append(df["loss"].to_numpy())
    for project in projects:
        df = pl.read_parquet(os.path.join(project, "00_0_Loss_hist.parquet"))
        losses.append(df["loss"].to_numpy())
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
    bins = np.logspace(min, max, 100)
    colors = ['gray', 'orange', 'darkgreen', 'maroon', 'darkblue']
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        for i, (loss, legend) in enumerate(zip(losses, legends)):
            ax.hist(loss, bins=bins, label=legend, color=colors[i], histtype="step", alpha=0.65)
        ax.set_xlabel("Total Loss")
        ax.set_ylabel("Count")
        ax.set_xscale("log")
        #ax.set_yscale("log")
        ax.legend()
        fig.savefig("figs/loss_hist.png", dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    console = Console()
    selected = choose_projects_to_plot()
    losses = losses_from_projects(selected)
    legends = ["RK4", "DeepONet", "TraONet", "VaRONet", "MambONet"]
    hist_losses(losses, legends)
