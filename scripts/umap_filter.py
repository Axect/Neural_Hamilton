import umap
import matplotlib.pyplot as plt
import scienceplots
import fireducks.pandas as pd
import numpy as np
import random
from sklearn.cluster import MiniBatchKMeans
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
#  Clustering
# └──────────────────────────────────────────────────────────┘
def cluster_data(embedding: np.ndarray, n_clusters) -> np.ndarray:
    clustering = MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit(embedding)
    return clustering.labels_

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

    # Fit UMAP
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
