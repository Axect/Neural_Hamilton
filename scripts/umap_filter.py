"""
UMAP Filtering Script

Performs UMAP dimensionality reduction on potential functions V(q),
clusters them, and samples representative examples using KDE-weighted sampling.

This script handles INTERLEAVED data layout from Rust generator:
    [pot0_win0, pot0_win1, pot1_win0, pot1_win1, ...]

Usage:
    python scripts/umap_filter.py --data_file="data_test/test_cand.parquet"
    python scripts/umap_filter.py --data_file="data_normal/train_cand.parquet" --ndiffconfig=2
"""

import umap
import matplotlib.pyplot as plt
import scienceplots
import fireducks.pandas as pd
import numpy as np
import random
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import gaussian_kde
import argparse
import warnings
import os
from rich.console import Console
import beaupy
from dataclasses import dataclass

NCOLS = 100
console = Console()
warnings.filterwarnings("ignore")

# Set Seed
random.seed(42)
np.random.seed(42)


# ┌──────────────────────────────────────────────────────────┐
#  UMAP and Data Processing
# └──────────────────────────────────────────────────────────┘
def extract_column(df: pd.DataFrame, col: str) -> np.ndarray:
    C = df[col].to_numpy()
    C = C.reshape((-1, NCOLS))
    return C


def embedding_to_df(embedding: np.ndarray) -> pd.DataFrame:
    umap1 = embedding[:, 0]
    umap2 = embedding[:, 1]
    df = pd.DataFrame({"umap1": umap1, "umap2": umap2})
    return df


def umap_fit(V: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> umap.UMAP:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
    )
    mapper = reducer.fit(V)
    return mapper


# ┌──────────────────────────────────────────────────────────┐
#  Interleaved Data Handling
# └──────────────────────────────────────────────────────────┘
def deinterleave_data(data: np.ndarray, ndiffconfig: int) -> list[np.ndarray]:
    """
    Convert interleaved data to list of blocks by config.

    Input layout (interleaved):
        [pot0_win0, pot0_win1, pot1_win0, pot1_win1, ...]

    Output: list of arrays, one per config
        blocks[0] = [pot0_win0, pot1_win0, pot2_win0, ...]  (all config 0)
        blocks[1] = [pot0_win1, pot1_win1, pot2_win1, ...]  (all config 1)
    """
    n_total = data.shape[0]
    assert n_total % ndiffconfig == 0, f"Data length {n_total} not divisible by ndiffconfig {ndiffconfig}"

    blocks = []
    for i in range(ndiffconfig):
        # Extract every ndiffconfig-th element starting from i
        indices = np.arange(i, n_total, ndiffconfig)
        blocks.append(data[indices])

    return blocks


def interleave_samples(blocks: list[np.ndarray], sample_indices: np.ndarray) -> np.ndarray:
    """
    Given sample indices (potential indices), reconstruct interleaved data.

    Args:
        blocks: list of arrays, one per config (from deinterleave_data)
        sample_indices: indices of selected potentials

    Returns:
        Interleaved array: [pot_i_win0, pot_i_win1, pot_j_win0, pot_j_win1, ...]
    """
    ndiffconfig = len(blocks)
    result = []

    for idx in sample_indices:
        for config in range(ndiffconfig):
            result.append(blocks[config][idx])

    return np.array(result)


# ┌──────────────────────────────────────────────────────────┐
#  Clustering and Sampling
# └──────────────────────────────────────────────────────────┘
def cluster_data(embedding: np.ndarray, n_clusters) -> np.ndarray:
    clustering = MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit(embedding)
    return clustering.labels_


def sample_from_clusters(clusters: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    # KDE
    x = clusters["umap1"].to_numpy()
    y = clusters["umap2"].to_numpy()
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)

    # Find center of mass
    unique_labels = clusters["label"].unique()
    centers = []
    for label in unique_labels:
        cluster_points = clusters[clusters["label"] == label]
        center = cluster_points[["umap1", "umap2"]].mean().to_numpy()
        centers.append(center)
    centers = np.array(centers)

    # Density of center
    densities = kernel(centers.T)
    density_min = densities.min()
    density_max = densities.max()
    normalized_densities = (densities - density_min) / (density_max - density_min + 1e-10)

    # Define weights based on density (lower density = higher weight)
    weights = np.exp(-1.0 * normalized_densities ** 2)
    weights /= weights.sum()  # Normalize weights

    # Define number of samples to take from each cluster
    n_sample_min = 5
    elements_per_cluster = [
        len(clusters[clusters["label"] == label]) for label in unique_labels
    ]
    total_samples = x.shape[0] // 10  # 10% of total points
    n_samples = np.repeat(n_sample_min, len(unique_labels))
    left_samples = total_samples - n_sample_min * len(unique_labels)
    target_samples = n_samples + np.round(left_samples * weights).astype(int)
    n_samples = np.clip(target_samples, n_sample_min, elements_per_cluster)
    current_total_samples = n_samples.sum()
    labels_and_n_samples = pd.DataFrame(
        {
            "label": unique_labels,
            "n_samples": n_samples,
            "n_elements": elements_per_cluster,
        }
    )
    sorted_label_and_n_sample = labels_and_n_samples.sort_values(
        by="n_samples", ascending=False
    )
    labels = sorted_label_and_n_sample["label"].values
    n_samples = sorted_label_and_n_sample["n_samples"].values
    n_elements = sorted_label_and_n_sample["n_elements"].values
    print(
        f"Total samples to take: {total_samples}, Current total samples: {current_total_samples}"
    )
    additional_samples = total_samples - current_total_samples
    while additional_samples != 0:
        current_total_samples = n_samples.sum()
        if current_total_samples < total_samples:
            for i in range(len(n_samples)):
                if n_samples[i] < n_elements[i]:
                    n_samples[i] += 1
                    additional_samples -= 1
                    if additional_samples <= 0:
                        break
        elif current_total_samples > total_samples:
            for i in reversed(range(len(n_samples))):
                if n_samples[i] > n_sample_min:
                    n_samples[i] -= 1
                    additional_samples += 1
                    if additional_samples >= 0:
                        break

    print(f"Final samples per cluster: {n_samples}")
    print(f"Total samples to take: {n_samples.sum()}")

    # Sample from clusters based on weights
    samples = []
    for label, n_sample in zip(labels, n_samples):
        cluster_points = clusters[clusters["label"] == label]
        sample = cluster_points.sample(n=n_sample, replace=False)
        samples.append(sample)

    samples = pd.concat(samples, ignore_index=True)
    return samples, n_samples


# ┌──────────────────────────────────────────────────────────┐
#  Utils
# └──────────────────────────────────────────────────────────┘
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    return df


def select_data_option() -> str:
    first_options = ["normal", "more", "much", "test"]
    console.print("Select a data option:")
    first_option = beaupy.select(first_options)
    data_folder = f"data_{first_option}"

    # List of files to process
    with os.scandir(data_folder) as entries:
        files = [
            entry.name
            for entry in entries
            if entry.is_file() and entry.name.endswith(".parquet")
        ]
        console.print("Select a file to process:")
    file_name = beaupy.select(files)

    return os.path.join(data_folder, file_name)


# ┌──────────────────────────────────────────────────────────┐
#  Plot
# └──────────────────────────────────────────────────────────┘
def hist_n_samples(n_samples: np.ndarray, data_quant, data_type: str):
    bins = max(len(np.unique(n_samples)) // 4, 10)
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        ax.hist(n_samples, bins=bins, edgecolor="black", linewidth=1.2, histtype="step")
        ax.set_xlabel("Samples per cluster")
        ax.set_ylabel("Number of clusters")
        fig.tight_layout()
        fig.savefig(
            f"figs/{data_quant}_{data_type}_samples_per_cluster.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_embedding(embedding: pd.DataFrame, data_quant, data_type: str):
    umap1 = embedding["umap1"]
    umap2 = embedding["umap2"]
    label = embedding["label"]

    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots(figsize=(4, 3))
        scatter = ax.scatter(
            umap1, umap2, c=label, cmap="Spectral", s=5, linewidth=0, alpha=0.7
        )

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.colorbar(scatter, label="Cluster ID", fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(
            f"figs/{data_quant}_{data_type}_umap_embedding.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_density_of_embedding(embedding: pd.DataFrame, data_quant: str, data_type: str):
    umap1 = embedding["umap1"]
    umap2 = embedding["umap2"]

    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots(figsize=(4, 3))
        values = np.vstack([umap1, umap2])
        kernel = gaussian_kde(values)
        xmin, xmax = umap1.min(), umap1.max()
        ymin, ymax = umap2.min(), umap2.max()
        x_padding = (xmax - xmin) * 0.1
        y_padding = (ymax - ymin) * 0.1
        xmin -= x_padding
        xmax += x_padding
        ymin -= y_padding
        ymax += y_padding
        xi, yi = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        zi = kernel(np.vstack([xi.ravel(), yi.ravel()]))
        zi = zi.reshape(xi.shape)

        ax.scatter(
            umap1, umap2, c="silver", s=5, linewidth=0, alpha=0.3, label="_nolegend_"
        )
        contour = ax.contourf(xi, yi, zi, levels=10, cmap="Blues", alpha=0.6)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.colorbar(contour, ax=ax, label="Density", fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(
            f"figs/{data_quant}_{data_type}_umap_density.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig)


@dataclass
class RelevantPotential:
    label: str
    color: str
    marker: str
    umap1: float
    umap2: float


def plot_density_of_embedding_with_relevant(
    embedding: pd.DataFrame,
    data_quant: str,
    data_type: str,
    relevants: list[RelevantPotential],
):
    umap1 = embedding["umap1"]
    umap2 = embedding["umap2"]

    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots(figsize=(4, 3))
        values = np.vstack([umap1, umap2])
        kernel = gaussian_kde(values)
        xmin, xmax = umap1.min(), umap1.max()
        ymin, ymax = umap2.min(), umap2.max()
        x_padding = (xmax - xmin) * 0.1
        y_padding = (ymax - ymin) * 0.1
        xmin -= x_padding
        xmax += x_padding
        ymin -= y_padding
        ymax += y_padding
        xi, yi = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        zi = kernel(np.vstack([xi.ravel(), yi.ravel()]))
        zi = zi.reshape(xi.shape)

        ax.scatter(
            umap1, umap2, c="silver", s=5, linewidth=0, alpha=0.3, label="_nolegend_"
        )
        contour = ax.contourf(xi, yi, zi, levels=10, cmap="Blues", alpha=0.6)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.colorbar(contour, ax=ax, label="Density", fraction=0.046, pad=0.04)

        # Plot relevant points
        for relevant in relevants:
            ax.scatter(
                relevant.umap1,
                relevant.umap2,
                s=20,
                linewidth=0.5,
                edgecolors="black",
                color=relevant.color,
                marker=relevant.marker,
                label=relevant.label,
            )
            print(
                f"{relevant.label} density: {kernel([relevant.umap1, relevant.umap2])[0]}"
            )

        ax.legend(fontsize=5, loc="upper left")

        fig.tight_layout()
        fig.savefig(
            f"figs/{data_quant}_{data_type}_umap_density_relevant.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig)


# ┌──────────────────────────────────────────────────────────┐
#  Main
# └──────────────────────────────────────────────────────────┘
if __name__ == "__main__":
    # Create figs directory if it doesn't exist
    os.makedirs("figs", exist_ok=True)

    # Non-interactive
    parser = argparse.ArgumentParser(description="UMAP and clustering on data files.")
    parser.add_argument(
        "--data_file", type=str, help="Path to the data file (parquet format)"
    )
    parser.add_argument(
        "--ndiffconfig", type=int, default=4, help="Number of different configurations per potential"
    )
    args = parser.parse_args()
    data_file = args.data_file
    ndiffconfig = args.ndiffconfig
    data_quant = data_file.split("/")[0].split("_")[-1]
    data_type = data_file.split("/")[-1].split(".")[0]
    print(f"Processing file: {data_file}")
    print(f"NDIFFCONFIG: {ndiffconfig}")

    # Load data
    df = load_data(data_file)
    V_full = extract_column(df, "V")
    t_full = extract_column(df, "t")
    q_full = extract_column(df, "q")
    p_full = extract_column(df, "p")
    console.print(f"Total samples: {V_full.shape[0]}")
    console.print(f"Unique potentials: {V_full.shape[0] // ndiffconfig}")

    # Deinterleave data into blocks by config
    # Input: [pot0_win0, pot0_win1, pot1_win0, pot1_win1, ...]
    # Output: VBLOCKS[0] = [pot0_win0, pot1_win0, ...], VBLOCKS[1] = [pot0_win1, pot1_win1, ...]
    VBLOCKS = deinterleave_data(V_full, ndiffconfig)
    tBLOCKS = deinterleave_data(t_full, ndiffconfig)
    qBLOCKS = deinterleave_data(q_full, ndiffconfig)
    pBLOCKS = deinterleave_data(p_full, ndiffconfig)

    # Use the first config for UMAP (all potentials have same V across configs)
    V = VBLOCKS[0]
    t = tBLOCKS[0]
    q = qBLOCKS[0]
    p = pBLOCKS[0]
    n_potentials = V.shape[0]
    console.print(f"Using config 0 for UMAP: {V.shape}")

    # Load relevant potentials for reference
    relevants = []
    try:
        df_relevants = [
            load_data("data_analyze/sho.parquet"),
            load_data("data_analyze/double_well.parquet"),
            load_data("data_analyze/morse.parquet"),
            load_data("data_analyze/atw.parquet"),
            load_data("data_analyze/stw.parquet"),
            load_data("data_analyze/sstw.parquet"),
        ]
        potentials_ref = [extract_column(df, "V") for df in df_relevants]
        labels_ref = ["SHO", "Double Well", "Morse", "ATW", "STW", "SSTW"]
        colors_ref = ["cyan", "darkviolet", "lime", "orange", "red", "deeppink"]
        markers_ref = ["o", "s", "^", "D", "P", "*"]
        has_relevants = True
    except FileNotFoundError:
        console.print("[yellow]Warning: Reference potential files not found, skipping relevant plots[/yellow]")
        has_relevants = False

    # Fit UMAP and cluster the data
    console.print("Fitting UMAP...")
    mapper = umap_fit(V)
    embedding = mapper.transform(V)
    n_clusters = max(n_potentials // 1000, 10)
    V_labels = cluster_data(embedding, n_clusters=n_clusters)
    console.print(f"UMAP embedding shape: {embedding.shape}")
    console.print(f"Number of clusters: {n_clusters}")

    # Transform reference potentials if available
    if has_relevants:
        embedding_potentials = [mapper.transform(potential) for potential in potentials_ref]
        relevants = [
            RelevantPotential(
                label=label,
                color=color,
                marker=marker,
                umap1=embedding_potential[0, 0],
                umap2=embedding_potential[0, 1],
            )
            for embedding_potential, label, color, marker in zip(
                embedding_potentials, labels_ref, colors_ref, markers_ref
            )
        ]

    # Convert embedding to DataFrame
    embedding_df = embedding_to_df(embedding)
    embedding_df["label"] = V_labels
    embedding_df["number"] = np.arange(n_potentials)  # Potential index (not sample index)
    print(embedding_df)
    unique_labels = np.unique(V_labels)
    console.print(f"Unique labels found: {len(unique_labels)}")

    # Sample from clusters
    samples, n_samples = sample_from_clusters(embedding_df)
    print(f"Number of unique potentials sampled: {samples.shape[0]}")
    print(f"Total samples (with all configs): {samples.shape[0] * ndiffconfig}")
    print(samples)
    hist_n_samples(n_samples, data_quant, data_type)

    # Save Embedding and samples
    data_folder = "data_umap"
    os.makedirs(data_folder, exist_ok=True)
    embedding_file = os.path.join(
        data_folder, f"{data_quant}_{data_type}_embedding.parquet"
    )
    samples_file = os.path.join(
        data_folder, f"{data_quant}_{data_type}_samples.parquet"
    )
    embedding_df.to_parquet(embedding_file, index=False, compression=None)
    samples.to_parquet(samples_file, index=False, compression=None)

    # Plot UMAP embedding and density
    plot_embedding(embedding_df, data_quant, data_type)
    plot_density_of_embedding(embedding_df, data_quant, data_type)
    plot_embedding(samples, data_quant, data_type + "_samples")
    plot_density_of_embedding(samples, data_quant, data_type + "_samples")
    if has_relevants:
        plot_density_of_embedding_with_relevant(
            embedding_df, data_quant, data_type, relevants
        )
        plot_density_of_embedding_with_relevant(
            samples, data_quant, data_type + "_samples", relevants
        )

    # Reconstruct DataFrame with samples (interleaved format)
    # numbers are potential indices, we need all configs for each selected potential
    numbers = samples["number"].to_numpy()
    V_samples = []
    t_samples = []
    q_samples = []
    p_samples = []

    # For each selected potential, get all its configs in interleaved order
    for num in numbers:
        for config_idx in range(ndiffconfig):
            V_samples.extend(VBLOCKS[config_idx][num].flatten())
            t_samples.extend(tBLOCKS[config_idx][num].flatten())
            q_samples.extend(qBLOCKS[config_idx][num].flatten())
            p_samples.extend(pBLOCKS[config_idx][num].flatten())

    samples_df = pd.DataFrame(
        {
            "V": V_samples,
            "t": t_samples,
            "q": q_samples,
            "p": p_samples,
        }
    )
    print(f"Samples DataFrame shape: {samples_df.shape}")
    print(f"  - Unique potentials: {len(numbers)}")
    print(f"  - Total samples: {len(numbers) * ndiffconfig}")
    print(f"  - Rows (samples * NCOLS): {samples_df.shape[0]}")
    print(samples_df)
    modified_data_file = data_file.replace(".parquet", "_samples.parquet")
    samples_df.to_parquet(modified_data_file, index=False, compression=None)
    console.print(f"[green]Saved filtered data to: {modified_data_file}[/green]")

    # UMAP again on samples for visualization
    samples_mat = extract_column(samples_df, "V")
    # Only use first config for UMAP (same V across configs)
    samples_mat_unique = samples_mat[::ndiffconfig]
    mapper_samples = umap_fit(samples_mat_unique)
    embedding_samples = mapper_samples.transform(samples_mat_unique)

    if has_relevants:
        embedding_potentials = [
            mapper_samples.transform(potential) for potential in potentials_ref
        ]
        relevants = [
            RelevantPotential(
                label=label,
                color=color,
                marker=marker,
                umap1=embedding_potential[0, 0],
                umap2=embedding_potential[0, 1],
            )
            for embedding_potential, label, color, marker in zip(
                embedding_potentials, labels_ref, colors_ref, markers_ref
            )
        ]

    embedding_samples_df = embedding_to_df(embedding_samples)
    plot_density_of_embedding(embedding_samples_df, data_quant, data_type + "_final")
    if has_relevants:
        plot_density_of_embedding_with_relevant(
            embedding_samples_df, data_quant, data_type + "_final", relevants
        )

    console.print("[bold green]UMAP filtering completed![/bold green]")
