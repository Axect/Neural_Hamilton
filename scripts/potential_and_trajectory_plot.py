import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import sys

sys.path.append("../")

from util import load_data

# Load data
data = load_data("../data_test/test.parquet")
q_domain = np.linspace(0, 1, 100)

for i in range(0, len(data), len(data) // 10):
    V, t, q, p = data[i]
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        ax.plot(q_domain, V)
        ax.set_xlabel(r"$q$", fontsize=8)
        ax.set_ylabel(r"$V(q)$", fontsize=8)
        ax.autoscale(tight=True)
        ax.set_ylim((-1, 2))
        fig.savefig(f"../figs/00_V_{i}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(t, q, ".-")
        ax.set_xlabel(r"$t$", fontsize=8)
        ax.set_ylabel(r"$q(t)$", fontsize=8)
        ax.autoscale(tight=True)
        fig.savefig(f"../figs/00_q_{i}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(t, p, ".-")
        ax.set_xlabel(r"$t$", fontsize=8)
        ax.set_ylabel(r"$p(t)$", fontsize=8)
        ax.autoscale(tight=True)
        fig.savefig(f"../figs/00_p_{i}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
