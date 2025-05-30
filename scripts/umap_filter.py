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

NCOLS = 100
console = Console()
warnings.filterwarnings("ignore")

# ┌──────────────────────────────────────────────────────────┐
#  UMAP and Data Processing
# └──────────────────────────────────────────────────────────┘
def extract_potential(df: pd.DataFrame) -> np.ndarray:
    V = df['V'].to_numpy()
    V = V.reshape((-1, NCOLS))
    return V

def embedding_to_df(embedding: np.ndarray) -> pd.DataFrame:
    umap1 = embedding[:, 0]
    umap2 = embedding[:, 1]
    df = pd.DataFrame({
        'umap1': umap1,
        'umap2': umap2
    })
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
#  Clustering and Sampling
# └──────────────────────────────────────────────────────────┘
def cluster_data(embedding: np.ndarray, n_clusters) -> np.ndarray:
    clustering = MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit(embedding)
    return clustering.labels_

def sample_from_clusters(clusters: pd.DataFrame) -> pd.DataFrame:
    # KDE
    x = clusters['umap1'].to_numpy()
    y = clusters['umap2'].to_numpy()
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)

    # Find center of mass
    unique_labels = clusters['label'].unique()
    centers = []
    for label in unique_labels:
        cluster_points = clusters[clusters['label'] == label]
        center = cluster_points[['umap1', 'umap2']].mean().to_numpy()
        centers.append(center)
    centers = np.array(centers)

    # Density of center
    densities = kernel(centers.T)
    normalized_densities = densities / densities.sum()

    # Define weights based on density
    weights = - np.log(normalized_densities + 1e-10)  # Avoid log(0)
    weights /= weights.sum()  # Normalize weights
    label_and_weight = pd.DataFrame({
        'label': unique_labels,
        'weight': weights
    })

    # Define number of samples to take from each cluster
    n_sample_min = 5
    n_sample_max = 100
    total_samples = x.shape[0] / 10 # 10% of total points
    sorted_label_and_weight = label_and_weight.sort_values(by='weight', ascending=False)
    n_samples = np.repeat(n_sample_min, len(unique_labels))
    left_samples = total_samples - n_sample_min * len(unique_labels)
    target_samples = n_samples + np.round(sorted_label_and_weight['weight'].values * left_samples).astype(int)
    n_samples = np.clip(target_samples, n_sample_min, n_sample_max)
    current_total_samples = n_samples.sum()
    print(f"Total samples to take: {total_samples}, Current total samples: {current_total_samples}")
    additional_samples = total_samples - current_total_samples
    if current_total_samples < total_samples:
        for i in range(len(n_samples)):
            if n_samples[i] < n_sample_max:
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

    # Sample from clusters based on weights
    samples = []
    for label, n_sample in zip(unique_labels, n_samples):
        cluster_points = clusters[clusters['label'] == label]
        sample = cluster_points.sample(n=n_sample, replace=False)
        samples.append(sample)


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
        files = [entry.name for entry in entries if entry.is_file() and entry.name.endswith('.parquet')]
        console.print("Select a file to process:")
    file_name = beaupy.select(files)

    return os.path.join(data_folder, file_name)


if __name__ == "__main__":
    data_file = select_data_option()
    print(f"Processing file: {data_file}")

    df = load_data(data_file)
    V = extract_potential(df)
    console.print(f"Data shape: {V.shape}")

    # Fit UMAP and cluster the data
    mapper = umap_fit(V)
    embedding = mapper.transform(V)
    V_labels = cluster_data(embedding, n_clusters = embedding.shape[0] // 1000)
    console.print(f"UMAP embedding shape: {embedding.shape}")

    # Convert embedding to DataFrame
    embedding_df = embedding_to_df(embedding)
    embedding_df['label'] = V_labels
    print(embedding_df)
    unique_labels = np.unique(V_labels)
    console.print(f"Unique labels found: {len(unique_labels)}")

    # Sample from clusters
    sample_from_clusters(embedding_df)
