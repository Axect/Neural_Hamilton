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
    output_dir = f"figs/{args.data}_{args.split}_samples"
    os.makedirs(output_dir, exist_ok=True)

    # Calculate indices to plot (evenly spaced)
    if args.num >= n_samples:
        indices = list(range(n_samples))
    else:
        indices = list(range(0, n_samples, n_samples // args.num))[:args.num]

    q_domain = np.linspace(0, 1, NSENSORS)

    print(f"Plotting {len(indices)} samples...")

    for idx in indices:
        V_i = V[idx]
        t_i = t[idx]
        q_i = q[idx]
        p_i = p[idx]

        # Initial condition
        q0, p0 = q_i[0], p_i[0]

        with plt.style.context(["science", "nature"]):
            # 1. Potential V(q) plot
            fig, ax = plt.subplots()
            ax.plot(q_domain, V_i, "b-", linewidth=1.2)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
            ax.set_xlabel(r"$q$", fontsize=8)
            ax.set_ylabel(r"$V(q)$", fontsize=8)
            ax.autoscale(tight=True)
            ax.set_ylim((-2.5, 2.5))
            fig.tight_layout()
            fig.savefig(f"{output_dir}/{idx:05d}_V.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

            # 2. Position q(t) plot
            fig, ax = plt.subplots()
            ax.plot(t_i, q_i, ".-", markersize=2, linewidth=0.8)
            ax.axhline(q0, color="red", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.set_xlabel(r"$t$", fontsize=8)
            ax.set_ylabel(r"$q(t)$", fontsize=8)
            ax.text(0.02, 0.95, f"$q_0 = {q0:.3f}$", transform=ax.transAxes, fontsize=6)
            ax.autoscale(tight=True)
            fig.tight_layout()
            fig.savefig(f"{output_dir}/{idx:05d}_q.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

            # 3. Momentum p(t) plot
            fig, ax = plt.subplots()
            ax.plot(t_i, p_i, ".-", markersize=2, linewidth=0.8)
            ax.axhline(p0, color="red", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.set_xlabel(r"$t$", fontsize=8)
            ax.set_ylabel(r"$p(t)$", fontsize=8)
            ax.text(0.02, 0.95, f"$p_0 = {p0:.3f}$", transform=ax.transAxes, fontsize=6)
            ax.autoscale(tight=True)
            fig.tight_layout()
            fig.savefig(f"{output_dir}/{idx:05d}_p.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

            # 4. Phase space (q, p) plot
            fig, ax = plt.subplots()
            # Color by time
            cmap = plt.get_cmap("viridis")
            colors = cmap(np.linspace(0, 1, len(t_i)))
            ax.scatter(q_i, p_i, c=colors, s=4, edgecolors="none")
            ax.plot(q_i, p_i, "k-", linewidth=0.3, alpha=0.5)
            ax.scatter([q0], [p0], c="red", s=20, marker="x", label=f"IC: ({q0:.2f}, {p0:.2f})")
            ax.set_xlabel(r"$q$", fontsize=8)
            ax.set_ylabel(r"$p$", fontsize=8)
            ax.legend(fontsize=5, loc="upper right")
            ax.autoscale(tight=True)
            fig.tight_layout()
            fig.savefig(f"{output_dir}/{idx:05d}_phase.png", dpi=600, bbox_inches="tight")
            plt.close(fig)

    print(f"\nSaved {len(indices) * 4} plots to: {output_dir}/")


if __name__ == "__main__":
    main()
