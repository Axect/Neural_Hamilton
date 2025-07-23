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
    """Extracts a column and reshapes it into (n_potentials, n_sensors)."""
    C = df[col].to_numpy()
    # Ensure that the array is not empty before reshaping
    if C.size == 0:
        return np.array([]).reshape(0, NCOLS)
    return C.reshape((-1, NCOLS))


def embedding_to_df(embedding: np.ndarray) -> pd.DataFrame:
    """Converts UMAP embedding array to a DataFrame."""
    umap1 = embedding[:, 0]
    umap2 = embedding[:, 1]
    df = pd.DataFrame({"umap1": umap1, "umap2": umap2})
    return df


def umap_fit(V: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> umap.UMAP:
    """Fits UMAP on the provided data."""
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
    """Performs MiniBatchKMeans clustering on the embedding."""
    # Ensure n_clusters is at least 1
    if n_clusters < 1:
        if embedding.shape[0] > 0:
            n_clusters = 1
        else:
            return np.array([]) # Cannot cluster empty data
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
    if clusters.empty:
        return pd.DataFrame(), np.array([])
        
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
    densities = kernel(centers.T)
    min_density, max_density = densities.min(), densities.max()
    
    if max_density - min_density > 1e-9:
        scaled_densities = (densities - min_density) / (max_density - min_density)
    else:
        scaled_densities = np.zeros_like(densities)

    density_weights = np.exp(-alpha * scaled_densities)
    size_factor = np.log1p(elements_per_cluster)
    combined_weights = density_weights / size_factor
    
    if combined_weights.sum() < 1e-9:
        weights = np.ones_like(combined_weights) / len(combined_weights)
    else:
        weights = combined_weights / combined_weights.sum()
    # --- END OF MODIFIED LOGIC ---

    n_sample_min = 5
    total_samples = max(x.shape[0] // 10, n_sample_min * len(unique_labels))
    n_samples = np.repeat(n_sample_min, len(unique_labels))
    left_samples = total_samples - n_samples.sum()
    
    if left_samples > 0:
        target_samples = n_samples + np.round(left_samples * weights).astype(int)
    else:
        target_samples = n_samples

    n_samples = np.clip(target_samples, n_sample_min, elements_per_cluster)
    
    current_total_samples = n_samples.sum()
    labels_and_n_samples = pd.DataFrame({
        "label": unique_labels,
        "n_samples": n_samples,
        "n_elements": elements_per_cluster,
        "weight": weights
    })
    
    sorted_label_and_n_sample = labels_and_n_samples.sort_values(by="weight", ascending=False)
    
    labels = sorted_label_and_n_sample["label"].values
    n_samples = sorted_label_and_n_sample["n_samples"].values
    n_elements = sorted_label_and_n_sample["n_elements"].values
    
    print(f"Target samples: {total_samples}, Initial samples: {current_total_samples}")
    
    additional_samples = total_samples - current_total_samples
    while additional_samples != 0:
        initial_additional_samples = additional_samples
        if additional_samples > 0:
            for i in range(len(n_samples)):
                if n_samples[i] < n_elements[i]:
                    n_samples[i] += 1
                    additional_samples -= 1
                    if additional_samples == 0: break
        elif additional_samples < 0:
            for i in reversed(range(len(n_samples))):
                if n_samples[i] > n_sample_min:
                    n_samples[i] -= 1
                    additional_samples += 1
                    if additional_samples == 0: break
        if additional_samples == initial_additional_samples:
            break

    print(f"Final samples per cluster: {n_samples}")
    print(f"Total samples to take: {n_samples.sum()}")

    final_samples_map = {label: n for label, n in zip(labels, n_samples)}

    samples = []
    for label in unique_labels:
        n_sample = final_samples_map.get(label, 0)
        if n_sample > 0:
            cluster_points = clusters[clusters["label"] == label]
            sample = cluster_points.sample(n=int(n_sample), replace=False, random_state=42)
            samples.append(sample)

    if not samples:
        return pd.DataFrame(), np.array([])
        
    samples_df = pd.concat(samples, ignore_index=True)
    return samples_df, n_samples


# ┌──────────────────────────────────────────────────────────┐
#  Utils
# └──────────────────────────────────────────────────────────┘
def load_data(file_path: str) -> pd.DataFrame:
    """Loads a parquet file into a DataFrame."""
    df = pd.read_parquet(file_path)
    return df

# ┌──────────────────────────────────────────────────────────┐
#  Plot
# └──────────────────────────────────────────────────────────┘
def hist_n_samples(n_samples: np.ndarray, data_quant, data_type: str):
    """Plots a histogram of the number of samples per cluster."""
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
    """Plots the UMAP embedding with cluster labels."""
    if embedding.empty or "label" not in embedding.columns:
        print(f"Embedding DataFrame is empty or missing labels for {data_quant}_{data_type}. Skipping plot.")
        return
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots(figsize=(4, 3))
        scatter = ax.scatter(
            embedding["umap1"], embedding["umap2"], c=embedding["label"], cmap="Spectral", s=5, linewidth=0, alpha=0.7
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.colorbar(scatter, label="Cluster ID", fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(f"figs/{data_quant}_{data_type}_umap_embedding.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


def plot_density_of_embedding(embedding: pd.DataFrame, data_quant: str, data_type: str):
    """Plots the density of the UMAP embedding."""
    if embedding.empty:
        print(f"Embedding DataFrame is empty for {data_quant}_{data_type}. Skipping density plot.")
        return
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots(figsize=(4, 3))
        values = np.vstack([embedding["umap1"], embedding["umap2"]])
        kernel = gaussian_kde(values)
        xmin, xmax = embedding["umap1"].min(), embedding["umap1"].max()
        ymin, ymax = embedding["umap2"].min(), embedding["umap2"].max()
        x_padding, y_padding = (xmax - xmin) * 0.1, (ymax - ymin) * 0.1
        xi, yi = np.mgrid[xmin-x_padding:xmax+x_padding:100j, ymin-y_padding:ymax+y_padding:100j]
        zi = kernel(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        ax.scatter(embedding["umap1"], embedding["umap2"], c="silver", s=5, linewidth=0, alpha=0.3)
        contour = ax.contourf(xi, yi, zi, levels=10, cmap="Blues", alpha=0.6)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.colorbar(contour, ax=ax, label="Density", fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(f"figs/{data_quant}_{data_type}_umap_density.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


@dataclass
class RelevantPotential:
    label: str
    color: str
    marker: str
    umap1: float
    umap2: float


def plot_density_of_embedding_with_relevant(
    embedding: pd.DataFrame, data_quant: str, data_type: str, relevants: list[RelevantPotential], mapper: umap.UMAP, potentials: list[np.ndarray]
):
    """Plots the density of the embedding along with relevant, known potentials."""
    if embedding.empty:
        print(f"Embedding DataFrame is empty for {data_quant}_{data_type}. Skipping relevant density plot.")
        return

    # Transform relevant potentials with the provided mapper
    embedding_potentials = [mapper.transform(potential) for potential in potentials]
    transformed_relevants = [
        RelevantPotential(
            label=rel.label, color=rel.color, marker=rel.marker,
            umap1=emb_pot[0, 0], umap2=emb_pot[0, 1]
        ) for rel, emb_pot in zip(relevants, embedding_potentials)
    ]
    
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots(figsize=(4, 3))
        values = np.vstack([embedding["umap1"], embedding["umap2"]])
        kernel = gaussian_kde(values)
        xmin, xmax = embedding["umap1"].min(), embedding["umap1"].max()
        ymin, ymax = embedding["umap2"].min(), embedding["umap2"].max()
        x_padding, y_padding = (xmax - xmin) * 0.1, (ymax - ymin) * 0.1
        xi, yi = np.mgrid[xmin-x_padding:xmax+x_padding:100j, ymin-y_padding:ymax+y_padding:100j]
        zi = kernel(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

        ax.scatter(embedding["umap1"], embedding["umap2"], c="silver", s=5, linewidth=0, alpha=0.3)
        contour = ax.contourf(xi, yi, zi, levels=10, cmap="Blues", alpha=0.6)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.colorbar(contour, ax=ax, label="Density", fraction=0.046, pad=0.04)

        for relevant in transformed_relevants:
            ax.scatter(relevant.umap1, relevant.umap2, s=20, linewidth=0.5, edgecolors="black",
                       color=relevant.color, marker=relevant.marker, label=relevant.label)
            print(f"{relevant.label} density: {kernel([relevant.umap1, relevant.umap2])[0]}")

        ax.legend(fontsize=5, loc="upper left")
        fig.tight_layout()
        fig.savefig(f"figs/{data_quant}_{data_type}_umap_density_relevant.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


# ┌──────────────────────────────────────────────────────────┐
#  Main
# └──────────────────────────────────────────────────────────┘
if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)

    parser = argparse.ArgumentParser(description="UMAP filtering for datasets with repeated potential blocks.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data file (parquet format)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hyperparameter for inverse density weighting.")
    parser.add_argument("--ndiffconfig", type=int, default=2, help="Number of different time configurations per potential, as used in data generation.")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file):
        print(f"Error: The file '{args.data_file}' does not exist.")
        exit()

    data_quant = os.path.basename(os.path.dirname(args.data_file)).split("_")[-1]
    data_type = os.path.splitext(os.path.basename(args.data_file))[0]
    print(f"Processing file: {args.data_file}")
    print(f"Using alpha: {args.alpha}, NDIFFCONFIG: {args.ndiffconfig}")

    # Load full data
    df_full = load_data(args.data_file)
    console.print(f"Full DataFrame shape: {df_full.shape}")

    # --- NEW LOGIC: BLOCK-BASED PROCESSING ---
    # 1. Calculate the number of unique potentials. The data is structured as
    #    [V_unique_1, V_unique_2, ... V_unique_N] repeated NDIFFCONFIG times.
    total_df_rows = len(df_full)
    if total_df_rows % NCOLS != 0:
        console.print(f"[bold red]Error: Total rows ({total_df_rows}) is not divisible by NCOLS ({NCOLS}).[/bold red]")
        exit()
    num_total_potentials = total_df_rows // NCOLS

    if num_total_potentials % args.ndiffconfig != 0:
        console.print(f"[bold red]Error: Total potentials ({num_total_potentials}) is not divisible by NDIFFCONFIG ({args.ndiffconfig}).[/bold red]")
        exit()
    num_unique_potentials = num_total_potentials // args.ndiffconfig
    console.print(f"Total potentials: {num_total_potentials}, Unique potentials: {num_unique_potentials}")

    # 2. Perform UMAP and sampling ONLY on the first block of unique potentials.
    rows_for_block1 = num_unique_potentials * NCOLS
    df_block1 = df_full.iloc[:rows_for_block1]
    V_block1 = extract_column(df_block1, "V")
    console.print(f"Processing first block for UMAP. Shape: {V_block1.shape}")

    mapper_block1 = umap_fit(V_block1)
    embedding_block1 = mapper_block1.transform(V_block1)
    n_clusters = embedding_block1.shape[0] // 1000
    labels_block1 = cluster_data(embedding_block1, n_clusters)
    
    embedding_df_block1 = embedding_to_df(embedding_block1)
    embedding_df_block1["label"] = labels_block1
    embedding_df_block1["number"] = np.arange(embedding_df_block1.shape[0]) # Index within the unique potentials

    console.print(f"Unique labels found in first block: {len(np.unique(labels_block1))}")

    samples_block1, n_samples = sample_from_clusters(embedding_df_block1, alpha=args.alpha)
    console.print(f"Number of potentials sampled from first block: {len(samples_block1)}")
    
    # 3. Get the indices of sampled potentials and expand them to all blocks.
    if not samples_block1.empty:
        sampled_potential_indices = samples_block1["number"].to_numpy()
        
        final_row_indices = []
        for pot_idx in sampled_potential_indices:
            for k in range(args.ndiffconfig):
                # Calculate the potential's index in the full (flattened) list of potentials
                full_pot_idx = pot_idx + (k * num_unique_potentials)
                # Calculate the start and end row indices for this potential
                start_row = full_pot_idx * NCOLS
                end_row = start_row + NCOLS
                final_row_indices.extend(range(start_row, end_row))
        
        # 4. Create the final sampled DataFrame from the original full DataFrame
        final_samples_df = df_full.iloc[final_row_indices].reset_index(drop=True)
        console.print(f"Final sampled DataFrame shape: {final_samples_df.shape}")

        # 5. Save the final filtered data
        modified_data_file = args.data_file.replace(".parquet", "_samples.parquet")
        final_samples_df.to_parquet(modified_data_file, index=False)
        console.print(f"Saved final sampled data to '{modified_data_file}'")
    else:
        console.print("[yellow]No samples were generated. Skipping final save and plots.[/yellow]")
        final_samples_df = pd.DataFrame()

    # --- PLOTTING ---
    # Load relevant potentials for plotting
    relevant_files = {
            "SHO": "data_analyze/sho.parquet",
            "Double Well": "data_analyze/double_well.parquet",
            "Morse": "data_analyze/morse.parquet",
            "ATW": "data_analyze/atw.parquet",
            "STW": "data_analyze/stw.parquet",
            "SSTW": "data_analyze/sstw.parquet",
    }
    df_relevants = [load_data(path) for path in relevant_files.values() if os.path.exists(path)]
    potentials_relevant = [extract_column(df_rel, "V") for df_rel in df_relevants]
    relevants_metadata = [
        RelevantPotential(label="SHO", color="cyan", marker="o", umap1=0, umap2=0),
        RelevantPotential(label="Double Well", color="darkviolet", marker="s", umap1=0, umap2=0),
        RelevantPotential(label="Morse", color="lime", marker="^", umap1=0, umap2=0),
        RelevantPotential(label="ATW", color="orange", marker="D", umap1=0, umap2=0),
        RelevantPotential(label="STW", color="red", marker="P", umap1=0, umap2=0),
        RelevantPotential(label="SSTW", color="deeppink", marker="*", umap1=0, umap2=0),
    ]

    # Plot results for the first block
    plot_embedding(embedding_df_block1, data_quant, data_type + "_block1_embedding")
    plot_density_of_embedding(embedding_df_block1, data_quant, data_type + "_block1_density")
    if potentials_relevant:
        plot_density_of_embedding_with_relevant(embedding_df_block1, data_quant, data_type + "_block1_density", relevants_metadata, mapper_block1, potentials_relevant)

    # Plot histogram of samples per cluster (from the first block analysis)
    hist_n_samples(n_samples, data_quant, data_type)

    # Plot UMAP of the final, combined samples
    if not final_samples_df.empty:
        console.print("\nRunning UMAP on the final filtered dataset for plotting...")
        V_final_samples = extract_column(final_samples_df, "V")
        mapper_final = umap_fit(V_final_samples)
        embedding_final = mapper_final.transform(V_final_samples)
        embedding_df_final = embedding_to_df(embedding_final)
        
        plot_density_of_embedding(embedding_df_final, data_quant, data_type + "_final_samples_density")
        if potentials_relevant:
            plot_density_of_embedding_with_relevant(embedding_df_final, data_quant, data_type + "_final_samples_density", relevants_metadata, mapper_final, potentials_relevant)
