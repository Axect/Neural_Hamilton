import matplotlib.pyplot as plt
import scienceplots
import numpy as np

from util import load_data

# Load data
data = load_data("../data_normal/val.parquet")
print(data.shape) # pyright:ignore
V_total = data[0]
t_total = data[1]
x_total = data[2]
p_total = data[3]

x_domain = np.linspace(0, 1, V_total.shape[1])

for i in range(10):
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        ax.plot(x_domain, V_total[i])
        ax.set_xlabel(r"$q$")
        ax.set_ylabel(r"$V(q)$")
        ax.autoscale(tight=True)
        fig.savefig(f"../figs/00_V_{i}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

