import fireducks.pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import argparse

parser = argparse.ArgumentParser(description="Histogram of energy conservation")
parser.add_argument(
    "--data",
    type=str,
    default="normal",
    help="normal or more",
)
args = parser.parse_args()

df = pd.read_parquet(f"data_analyze/{args.data}_conserved.parquet")
print(df)
E_delta_max = df["E_delta_max"]

with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.hist(E_delta_max, bins=100, color="darkblue", histtype="step")
    ax.set_xlabel(r"$\Delta E_{\rm max} / (E_{\rm max} + E_{\rm min})$")
    ax.set_ylabel("Counts")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(f"figs/is_conserved_{args.data}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
