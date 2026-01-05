"""
Potential and Trajectory Visualization Script

Plots potential V(q) and phase space trajectories (q(t), p(t)) from training data.
Useful for visualizing data quality and trajectory patterns.

Usage:
    python scripts/potential_and_trajectory_plot.py --data test
    python scripts/potential_and_trajectory_plot.py --data normal --split train
    python scripts/potential_and_trajectory_plot.py --data normal --split train --num 20
"""

import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import fireducks.pandas as pd
import argparse
import os


NSENSORS = 100


def main():
    parser = argparse.ArgumentParser(description="Potential and trajectory plots")
    parser.add_argument(
        "--data",
        type=str,
        default="test",
        choices=["normal", "more", "test"],
        help="Data type: normal, more, or test",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Data split: train, val, or test",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Number of samples to plot",
    )
    args = parser.parse_args()

    # Determine file path based on data type
    if args.data == "test":
        file_path = "data_test/test.parquet"
    else:
        file_path = f"data_{args.data}/{args.split}.parquet"

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    print(f"Loading data from: {file_path}")
    df = pd.read_parquet(file_path)

    # Reshape data to (N, 100) format
    V = df["V"].to_numpy().reshape(-1, NSENSORS)
    t = df["t"].to_numpy().reshape(-1, NSENSORS)
    q = df["q"].to_numpy().reshape(-1, NSENSORS)
    p = df["p"].to_numpy().reshape(-1, NSENSORS)

    n_samples = V.shape[0]
    print(f"Number of samples: {n_samples}")

    # Ensure output directory exists
    os.makedirs("figs", exist_ok=True)

    # Calculate indices to plot (evenly spaced)
    if args.num >= n_samples:
        indices = list(range(n_samples))
    else:
        indices = list(range(0, n_samples, n_samples // args.num))[:args.num]

    q_domain = np.linspace(0, 1, NSENSORS)

    print(f"Plotting {len(indices)} samples...")

    for i in indices:
        V_i = V[i]
        t_i = t[i]
        q_i = q[i]
        p_i = p[i]

        # Initial condition
        q0, p0 = q_i[0], p_i[0]

        with plt.style.context(["science", "nature"]):
            # 1. Potential V(q) plot
            fig, ax = plt.subplots()
            ax.plot(q_domain, V_i)
            ax.set_xlabel(r"$q$", fontsize=8)
            ax.set_ylabel(r"$V(q)$", fontsize=8)
            ax.autoscale(tight=True)
            ax.set_ylim((-1, 2.2))
            fig.savefig(f"figs/00_V_{i}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

            # 2. Position q(t) plot
            fig, ax = plt.subplots()
            ax.plot(t_i, q_i, ".-")
            ax.set_xlabel(r"$t$", fontsize=8)
            ax.set_ylabel(r"$q(t)$", fontsize=8)
            ax.margins(x=0.02, y=0.05)
            fig.savefig(f"figs/00_q_{i}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

            # 3. Momentum p(t) plot
            fig, ax = plt.subplots()
            ax.plot(t_i, p_i, ".-")
            ax.set_xlabel(r"$t$", fontsize=8)
            ax.set_ylabel(r"$p(t)$", fontsize=8)
            ax.margins(x=0.02, y=0.05)
            fig.savefig(f"figs/00_p_{i}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

            # 4. Phase space (q, p) plot
            fig, ax = plt.subplots()
            ax.plot(q_i, p_i, ".-")
            # Mark initial condition
            ax.scatter([q0], [p0], s=15, c="crimson", zorder=5, edgecolors="black", linewidths=0.3)
            ax.set_xlabel(r"$q$", fontsize=8)
            ax.set_ylabel(r"$p$", fontsize=8)
            ax.margins(x=0.05, y=0.05)
            fig.savefig(f"figs/00_phase_{i}.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    print(f"\nSaved {len(indices) * 4} plots to: figs/")


if __name__ == "__main__":
    main()
