"""
Energy Conservation Analysis Script

Calculates and visualizes energy conservation (E = V(q) + p^2/2) across trajectories.
Useful for verifying symplectic integrator quality.

Usage:
    python scripts/is_conserved.py --data normal
    python scripts/is_conserved.py --data test
"""

import fireducks.pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import argparse
from scipy.interpolate import CubicSpline
import os


NSENSORS = 100


def calculate_energy_conservation(V: np.ndarray, q: np.ndarray, p: np.ndarray) -> float:
    """
    Calculate energy conservation metric for a single trajectory.

    Args:
        V: Potential values at uniform q grid (100,)
        q: Position trajectory (100,)
        p: Momentum trajectory (100,)

    Returns:
        E_delta: Relative energy deviation (E_max - E_min) / (E_max + E_min)
    """
    # Create spline interpolation for V(q)
    q_domain = np.linspace(0, 1, NSENSORS)
    V_spline = CubicSpline(q_domain, V)

    # Calculate total energy E = V(q) + p^2/2 at each time point
    E = V_spline(q) + 0.5 * p**2

    E_max = np.max(E)
    E_min = np.min(E)

    # Relative energy deviation
    E_delta = (E_max - E_min) / max(E_max + E_min, 1e-10)

    return E_delta


def main():
    parser = argparse.ArgumentParser(description="Energy conservation analysis")
    parser.add_argument(
        "--data",
        type=str,
        default="normal",
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

    # Reshape data
    V = df["V"].to_numpy().reshape(-1, NSENSORS)
    q = df["q"].to_numpy().reshape(-1, NSENSORS)
    p = df["p"].to_numpy().reshape(-1, NSENSORS)

    n_samples = V.shape[0]
    print(f"Number of samples: {n_samples}")

    # Calculate energy conservation for each trajectory
    E_delta_vec = []
    for i in range(n_samples):
        E_delta = calculate_energy_conservation(V[i], q[i], p[i])
        E_delta_vec.append(E_delta)

    E_delta_max = np.array(E_delta_vec)

    # Print statistics
    print(f"\nEnergy Conservation Statistics:")
    print(f"  Mean:   {np.mean(E_delta_max):.6e}")
    print(f"  Median: {np.median(E_delta_max):.6e}")
    print(f"  Max:    {np.max(E_delta_max):.6e}")
    print(f"  Min:    {np.min(E_delta_max):.6e}")
    print(f"  Std:    {np.std(E_delta_max):.6e}")

    # Count samples above threshold
    threshold = 0.001
    n_above = np.sum(E_delta_max > threshold)
    print(f"\n  Samples with E_delta > {threshold}: {n_above} ({100*n_above/n_samples:.2f}%)")

    # Ensure output directory exists
    os.makedirs("figs", exist_ok=True)

    # Plot histogram
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()

        # Use log bins for better visualization
        bins = np.logspace(np.log10(max(E_delta_max.min(), 1e-12)),
                          np.log10(E_delta_max.max()), 100)

        ax.hist(E_delta_max, bins=bins, color="darkblue", histtype="step", linewidth=1.2)
        ax.axvline(threshold, color="red", linestyle="--", linewidth=0.8, label=f"Threshold ({threshold})")
        ax.set_xlabel(r"$\Delta E_{\rm max} / (E_{\rm max} + E_{\rm min})$")
        ax.set_ylabel("Counts")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=6)
        fig.tight_layout()

        output_name = f"figs/is_conserved_{args.data}_{args.split}.png"
        fig.savefig(output_name, dpi=600, bbox_inches="tight")
        print(f"\nSaved: {output_name}")
        plt.close(fig)


if __name__ == "__main__":
    main()
