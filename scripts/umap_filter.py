import umap
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd # fireducks.pandas is not a standard library, using pandas
import numpy as np
import random
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import gaussian_kde
import argparse
import warnings
import os
from rich.console import Console
# beaupy is not a standard library, falling back to simple input
# import beaupy
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
        random_state=42, # Add random_state for reproducibility
    )
    mapper = reducer.fit(V)
    return mapper


# ┌──────────────────────────────────────────────────────────┐
#  Clustering and Sampling
# └──────────────────────────────────────────────────────────┘
def cluster_data(embedding: np.ndarray, n_clusters) -> np.ndarray:
    # Ensure n_clusters is at least 1
    if n_clusters < 1:
        n_clusters = 1
    clustering = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(embedding)
    return clustering.labels_


def sample_from_clusters(clusters: pd.DataFrame, alpha: float = 5.0) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Samples data points from clusters, prioritizing less dense regions.

    Args:
        clusters (pd.DataFrame): DataFrame with umap1, umap2, and label columns.
        alpha (float): A hyperparameter to control the aggressiveness of inverse density weighting.
                       Higher alpha means stronger preference for sparser regions.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: A tuple containing the sampled DataFrame and the number of samples per cluster.
    """
    # KDE for density estimation
    x = clusters["umap1"].to_numpy()
    y = clusters["umap2"].to_numpy()
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)

    # Find center of mass and count elements per cluster
    unique_labels = clusters["label"].unique()
    centers = []
    elements_per_cluster = []
    for label in unique_labels:
        cluster_points = clusters[clusters["label"] == label]
        elements_per_cluster.append(len(cluster_points))
        center = cluster_points[["umap1", "umap2"]].mean().to_numpy()
        centers.append(center)
    centers = np.array(centers)
    elements_per_cluster = np.array(elements_per_cluster)

    # --- MODIFIED WEIGHTING LOGIC ---
    # Density of center
    densities = kernel(centers.T)
    
    # Min-Max scale densities to a [0, 1] range to make it independent of data size
    min_density = densities.min()
    max_density = densities.max()
    
    # Avoid division by zero if all densities are the same
    if max_density - min_density > 1e-9:
        scaled_densities = (densities - min_density) / (max_density - min_density)
    else:
        scaled_densities = np.zeros_like(densities)

    # Define weights based on inverse density using the fixed hyperparameter 'alpha'
    # This decouples the weighting from the number of clusters/data size
    density_weights = np.exp(-alpha * scaled_densities)

    # Adjust weights by cluster size to avoid over-sampling tiny, sparse clusters
    size_factor = np.log1p(elements_per_cluster)

    # Combine density-based weights with size-based weights
    combined_weights = density_weights * size_factor
    
    # Handle case where all weights are zero
    if combined_weights.sum() < 1e-9:
        weights = np.ones_like(combined_weights) / len(combined_weights)
    else:
        weights = combined_weights / combined_weights.sum()
    # --- END OF MODIFIED LOGIC ---

    # Define number of samples to take from each cluster
    n_sample_min = 5
    total_samples = max(x.shape[0] // 10, n_sample_min * len(unique_labels)) # Ensure total_samples is sufficient
    n_samples = np.repeat(n_sample_min, len(unique_labels))
    left_samples = total_samples - n_samples.sum()
    
    if left_samples > 0:
        target_samples = n_samples + np.round(left_samples * weights).astype(int)
    else:
        target_samples = n_samples

    n_samples = np.clip(target_samples, n_sample_min, elements_per_cluster)
    
    # Adjust total samples to match the target precisely
    current_total_samples = n_samples.sum()
    labels_and_n_samples = pd.DataFrame({
        "label": unique_labels,
        "n_samples": n_samples,
        "n_elements": elements_per_cluster,
    })
    
    # Sort by original weights to prioritize adding/removing from clusters as intended
    labels_and_n_samples['weight'] = weights
    sorted_label_and_n_sample = labels_and_n_samples.sort_values(by="weight", ascending=False)
    
    labels = sorted_label_and_n_sample["label"].values
    n_samples = sorted_label_and_n_sample["n_samples"].values
    n_elements = sorted_label_and_n_sample["n_elements"].values
    
    print(f"Target samples: {total_samples}, Initial samples: {current_total_samples}")
    
    # Iteratively adjust sample counts to meet the total_samples target
    additional_samples = total_samples - current_total_samples
    while additional_samples != 0:
        if additional_samples > 0: # Need to add more samples
            for i in range(len(n_samples)):
                if n_samples[i] < n_elements[i]:
                    n_samples[i] += 1
                    additional_samples -= 1
                    if additional_samples == 0: break
        elif additional_samples < 0: # Need to remove samples
            for i in reversed(range(len(n_samples))):
                if n_samples[i] > n_sample_min:
                    n_samples[i] -= 1
                    additional_samples += 1
                    if additional_samples == 0: break
        # If no more adjustments can be made, break the loop
        if additional_samples == (total_samples - n_samples.sum()):
            break


    print(f"Final samples per cluster: {n_samples}")
    print(f"Total samples to take: {n_samples.sum()}")

    # Create a map from label to final n_sample count
    final_samples_map = {label: n for label, n in zip(labels, n_samples)}

    # Sample from clusters based on the final calculated counts
    samples = []
    for label in unique_labels:
        n_sample = final_samples_map.get(label, n_sample_min)
        if n_sample > 0:
            cluster_points = clusters[clusters["label"] == label]
            sample = cluster_points.sample(n=n_sample, replace=False, random_state=42)
            samples.append(sample)

    if not samples:
        return pd.DataFrame(), np.array([])
        
    samples_df = pd.concat(samples, ignore_index=True)
    return samples_df, n_samples


# ┌──────────────────────────────────────────────────────────┐
#  Utils
# └──────────────────────────────────────────────────────────┘
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    return df

def select_data_option() -> str:
    # beaupy is not a standard library, using simple input instead
    first_options = ["normal", "more", "much", "test"]
    console.print("Select a data option:", first_options)
    first_option = input(f"Enter one of {first_options}: ")
    if first_option not in first_options:
        first_option = "normal"
        print(f"Invalid option, defaulting to '{first_option}'")

    data_folder = f"data_{first_option}"
    if not os.path.exists(data_folder):
        print(f"Error: Data folder '{data_folder}' not found.")
        exit()

    with os.scandir(data_folder) as entries:
        files = [
            entry.name
            for entry in entries
            if entry.is_file() and entry.name.endswith(".parquet")
        ]
    
    if not files:
        print(f"Error: No .parquet files found in '{data_folder}'.")
        exit()
        
    console.print("Select a file to process:", files)
    file_name = input(f"Enter one of {files}: ")
    if file_name not in files:
        file_name = files[0]
        print(f"Invalid file, defaulting to '{file_name}'")

    return os.path.join(data_folder, file_name)


# ┌──────────────────────────────────────────────────────────┐
#  Plot
# └──────────────────────────────────────────────────────────┘
def hist_n_samples(n_samples: np.ndarray, data_quant, data_type: str):
    if n_samples.size == 0:
        print("No samples to plot in histogram.")
        return
    bins = max(len(np.unique(n_samples)) // 4, 5)
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
    if embedding.empty:
        print(f"Embedding DataFrame is empty for {data_quant}_{data_type}. Skipping plot.")
        return
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
    if embedding.empty:
        print(f"Embedding DataFrame is empty for {data_quant}_{data_type}. Skipping density plot.")
        return
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
    if embedding.empty:
        print(f"Embedding DataFrame is empty for {data_quant}_{data_type}. Skipping relevant density plot.")
        return
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

    parser = argparse.ArgumentParser(description="UMAP and clustering on data files.")
    parser.add_argument(
        "--data_file", type=str, help="Path to the data file (parquet format)", required=True
    )
    # Add alpha as a command-line argument
    parser.add_argument(
        "--alpha", type=float, default=5.0, help="Hyperparameter for inverse density weighting."
    )
    args = parser.parse_args()
    data_file = args.data_file
    alpha = args.alpha
    
    if not os.path.exists(data_file):
        print(f"Error: The file '{data_file}' does not exist.")
        exit()

    data_quant = os.path.basename(os.path.dirname(data_file)).split("_")[-1]
    data_type = os.path.splitext(os.path.basename(data_file))[0]
    print(f"Processing file: {data_file}")
    print(f"Using alpha value: {alpha}")

    # Load data
    df = load_data(data_file)
    V = extract_column(df, "V")
    t = extract_column(df, "t")
    q = extract_column(df, "q")
    p = extract_column(df, "p")
    console.print(f"Data shape: {V.shape}")

    # Load relevant potentials (ensure these files exist)
    relevant_files = {
        "SHO": "data_analyze/sho.parquet",
        "Double Well": "data_analyze/double_well.parquet",
        "Morse": "data_analyze/morse.parquet",
        "ATW": "data_analyze/atw.parquet",
        "STW": "data_analyze/stw.parquet",
        "SSTW": "data_analyze/sstw.parquet",
    }
    df_relevants = []
    for label, path in relevant_files.items():
        if os.path.exists(path):
            df_relevants.append(load_data(path))
        else:
            print(f"Warning: Relevant potential file not found: {path}")
            
    potentials = [extract_column(df_rel, "V") for df_rel in df_relevants]
    labels = ["SHO", "Double Well", "Morse", "ATW", "STW", "SSTW"]
    colors = ["cyan", "darkviolet", "lime", "orange", "red", "deeppink"]
    markers = ["o", "s", "^", "D", "P", "*"]

    # Fit UMAP and cluster the data
    mapper = umap_fit(V)
    embedding = mapper.transform(V)
    n_clusters = embedding.shape[0] // 1000
    V_labels = cluster_data(embedding, n_clusters=n_clusters)
    console.print(f"UMAP embedding shape: {embedding.shape}")
    
    relevants = []
    if potentials:
        embedding_potentials = [mapper.transform(potential) for potential in potentials]
        relevants = [
            RelevantPotential(
                label=label,
                color=color,
                marker=marker,
                umap1=embedding_potential[0, 0],
                umap2=embedding_potential[0, 1],
            )
            for embedding_potential, label, color, marker in zip(
                embedding_potentials, labels, colors, markers
            )
        ]

    # Convert embedding to DataFrame
    embedding_df = embedding_to_df(embedding)
    embedding_df["label"] = V_labels
    embedding_df["number"] = np.arange(embedding_df.shape[0])
    unique_labels = np.unique(V_labels)
    console.print(f"Unique labels found: {len(unique_labels)}")

    # Sample from clusters using the new alpha parameter
    samples, n_samples = sample_from_clusters(embedding_df, alpha=alpha)
    print(f"Number of samples taken: {samples.shape[0]}")
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
    embedding_df.to_parquet(embedding_file, index=False)
    samples.to_parquet(samples_file, index=False)

    # Plot UMAP embedding and density
    plot_embedding(embedding_df, data_quant, data_type)
    plot_density_of_embedding(embedding_df, data_quant, data_type)
    plot_embedding(samples, data_quant, data_type + "_samples")
    plot_density_of_embedding(samples, data_quant, data_type + "_samples")
    if relevants:
        plot_density_of_embedding_with_relevant(
            embedding_df, data_quant, data_type, relevants
        )
        plot_density_of_embedding_with_relevant(
            samples, data_quant, data_type + "_samples", relevants
        )

    # Reconstruct DataFrame with samples if samples were generated
    if not samples.empty:
        numbers = samples["number"].to_numpy()
        V_samples = V[numbers].flatten()
        t_samples = t[numbers].flatten()
        q_samples = q[numbers].flatten()
        p_samples = p[numbers].flatten()
        samples_df = pd.DataFrame(
            {
                "V": V_samples,
                "t": t_samples,
                "q": q_samples,
                "p": p_samples,
            }
        )
        print(f"Samples DataFrame shape: {samples_df.shape}")
        modified_data_file = data_file.replace(".parquet", "_samples.parquet")
        samples_df.to_parquet(modified_data_file, index=False)

        # UMAP again on samples
        samples_mat = extract_column(samples_df, "V")
        mapper_samples = umap_fit(samples_mat)
        
        if relevants:
            embedding_potentials_samples = [
                mapper_samples.transform(potential) for potential in potentials
            ]
            relevants_samples = [
                RelevantPotential(
                    label=label,
                    color=color,
                    marker=marker,
                    umap1=embedding_potential[0, 0],
                    umap2=embedding_potential[0, 1],
                )
                for embedding_potential, label, color, marker in zip(
                    embedding_potentials_samples, labels, colors, markers
                )
            ]
            embedding_samples_df = embedding_to_df(mapper_samples.transform(samples_mat))
            plot_density_of_embedding_with_relevant(
                embedding_samples_df, data_quant, data_type + "_embedding_samples", relevants_samples
            )
