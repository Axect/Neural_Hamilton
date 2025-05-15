import umap
import umap.plot
import matplotlib.pyplot as plt
import scienceplots
import polars as pl
import numpy as np
import random
from sklearn.cluster import KMeans
import argparse


def reshape_df(df: pl.DataFrame, n_cols: int, column: str = "V") -> pl.DataFrame:
    V = df[column].to_numpy()
    V = V.reshape(-1, n_cols)
    column_names = [f"V{i}" for i in range(n_cols)]
    reshaped_df = pl.DataFrame(V, schema=column_names)
    # Add index column
    reshaped_df = reshaped_df.with_columns(pl.arange(0, reshaped_df.height).alias("index"))
    return reshaped_df


def umap_map_and_embed(df: pl.DataFrame, n_neighbors: int = 15, min_dist: float = 0.1, n_components: int = 2, random_state: int = 42) -> tuple:
    # Remove index column
    index = df.get_column("index").to_numpy()
    df_no_id = df.drop("index")

    # If it has Cluster column, drop it
    if "Cluster" in df.columns:
        cluster = df.get_column("Cluster").to_numpy()
        df_no_id = df_no_id.drop("Cluster")

    # UMAP dimensionality reduction
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
    mapper = reducer.fit(df_no_id)
    embedding = mapper.transform(df_no_id)

    # Create a DataFrame with the UMAP coordinates
    umap_df = pl.DataFrame(embedding, schema=["UMAP1", "UMAP2"])
    umap_df = umap_df.with_columns(pl.Series("index", index))
    if "Cluster" in df.columns:
        umap_df = umap_df.with_columns(pl.Series("Cluster", cluster))

    return umap_df, mapper

def kmeans_clustering(df: pl.DataFrame, n_clusters: int = 10, random_state: int = 42) -> pl.DataFrame:
    # Remove index column
    index = df.get_column("index").to_numpy()
    df_no_id = df.drop("index")

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto') # Added n_init for future compatibility
    labels = kmeans.fit_predict(df_no_id)

    # Create a DataFrame with the cluster labels
    cluster_df = pl.DataFrame(labels, schema=["Cluster"])
    cluster_df = cluster_df.with_columns(pl.Series("index", index))

    return cluster_df


def sampling_by_cluster(df: pl.DataFrame, cluster_df: pl.DataFrame, n_target: int, n_clusters: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    n_each = n_target // n_clusters
    sampled_dfs = []
    sampled_clusters_list = [] # Renamed to avoid conflict
    for i in range(n_clusters):
        cluster_indices = cluster_df.filter(pl.col("Cluster") == i).get_column("index").to_numpy()
        # Ensure we don't try to sample more than available, or if cluster_indices is empty
        if len(cluster_indices) == 0:
            continue
        sample_size = min(n_each, len(cluster_indices))
        sampled_indices = np.random.choice(cluster_indices, size=sample_size, replace=False)
        sampled_dfs.append(df.filter(pl.col("index").is_in(sampled_indices)))
        sampled_clusters_list.append(cluster_df.filter(pl.col("index").is_in(sampled_indices)).with_columns(pl.lit(i).alias("Cluster")))

        # Handle remaining samples for the last cluster if n_target is not perfectly divisible
        # Or if some clusters had fewer than n_each samples
        if i == n_clusters - 1:
            current_total_sampled = sum(len(s_df) for s_df in sampled_dfs)
            remaining_needed = n_target - current_total_sampled
            if remaining_needed > 0:
                # Get all indices from the current cluster not yet sampled
                potential_remaining_indices = np.setdiff1d(cluster_indices, sampled_indices)
                if len(potential_remaining_indices) > 0:
                    take_remaining = min(remaining_needed, len(potential_remaining_indices))
                    additional_sampled_indices = np.random.choice(potential_remaining_indices, size=take_remaining, replace=False)
                    sampled_dfs.append(df.filter(pl.col("index").is_in(additional_sampled_indices)))
                    sampled_clusters_list.append(cluster_df.filter(pl.col("index").is_in(additional_sampled_indices)).with_columns(pl.lit(i).alias("Cluster")))

    # Concatenate only if lists are not empty
    sampled_df = pl.concat(sampled_dfs) if sampled_dfs else pl.DataFrame(schema=df.schema)
    sampled_clusters_df = pl.concat(sampled_clusters_list) if sampled_clusters_list else pl.DataFrame(schema=cluster_df.schema)
    return sampled_df, sampled_clusters_df


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Set up argument parser
    parser = argparse.ArgumentParser(description="UMAP and KMeans clustering")
    parser.add_argument("--data", type=str, required=True, help="Suffix for data files (e.g., 'normal', 'more', 'much')")
    args = parser.parse_args()

    # Load the data
    df_train_original = pl.read_parquet(f"data_{args.data}/train_cand.parquet")
    df_val_original = pl.read_parquet(f"data_{args.data}/val_cand.parquet")

    # Reshape the DataFrame
    n_cols = 100 # This represents the number of rows in the original data that form one sample in the reshaped data
    df_train = reshape_df(df_train_original, n_cols)
    df_val = reshape_df(df_val_original, n_cols)

    # Perform UMAP embedding
    umap_train, mapper_train = umap_map_and_embed(df_train, n_neighbors=10, min_dist=0.001, n_components=2, random_state=42)
    umap_val, _ = umap_map_and_embed(df_val, n_neighbors=10, min_dist=0.001, n_components=2, random_state=42) # mapper_val not used
    print(f"UMAP Training DataFrame shape: {umap_train.shape}")
    print(umap_train)

    # Cluster the UMAP embedded data
    cluster_train = kmeans_clustering(umap_train, n_clusters=10, random_state=42)
    cluster_val = kmeans_clustering(umap_val, n_clusters=10, random_state=42)
    print(f"Cluster Training DataFrame shape: {cluster_train.shape}")
    print(cluster_train)

    # Plot UMAP projection of training data
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        umap.plot.points(mapper_train, ax=ax, labels=cluster_train["Cluster"].to_numpy())
        fig.savefig(f"figs/umap_projection_{args.data}.png", dpi=600, bbox_inches="tight")
        plt.close(fig) # Close plot to free memory

    # Define target sample sizes based on argument
    if args.data == "normal":
        n_target_train = 8000
        n_target_val = 2000
    elif args.data == "more":
        n_target_train = 80000
        n_target_val = 20000
    elif args.data == "much":
        n_target_train = 800000
        n_target_val = 200000
    else:
        # Default or error handling for unknown args.data
        print(f"Warning: Unknown data size '{args.data}'. Using default small sample sizes.")
        n_target_train = 1000
        n_target_val = 200
    n_clusters = 10

    # Sample data by cluster
    # Ensure df_train is merged with cluster_train before sampling to pass correct structure
    df_train_with_clusters = df_train.join(cluster_train, on="index", how="left")
    df_val_with_clusters = df_val.join(cluster_val, on="index", how="left")

    sampled_train, sampled_clusters_train_df = sampling_by_cluster(df_train_with_clusters, cluster_train, n_target_train, n_clusters)
    sampled_val, sampled_clusters_val_df = sampling_by_cluster(df_val_with_clusters, cluster_val, n_target_val, n_clusters)
    print(f"Sampled Training DataFrame shape: {sampled_train.shape}")
    print(sampled_train)

    # Check if the sampled DataFrame is balanced by re-embedding and plotting
    if not sampled_train.is_empty() and "Cluster" in sampled_clusters_train_df.columns and not sampled_clusters_train_df.is_empty():
        sampled_umap_train, sampled_mapper_train = umap_map_and_embed(sampled_train.join(sampled_clusters_train_df.select(["index", "Cluster"]), on="index"), n_neighbors=10, min_dist=0.001, n_components=2, random_state=42)
        # Plot UMAP for sampled training data
        with plt.style.context(["science", "nature"]):
            fig, ax = plt.subplots()
            umap.plot.points(sampled_mapper_train, ax=ax, labels=sampled_umap_train["Cluster"].to_numpy()) # Use cluster labels from sampled_umap_train
            fig.savefig(f"figs/umap_projection_sampled_{args.data}.png", dpi=600, bbox_inches="tight")
            plt.close(fig) # Close plot
    else:
        print("Skipping UMAP plot for sampled training data as it's empty or lacks cluster information.")

    # Save the sampled data
    # 1. Get indices from the sampled reshaped data
    sampled_reshaped_indices_train = sampled_train.get_column("index").to_list()
    sampled_reshaped_indices_val = sampled_val.get_column("index").to_list()

    # 2. Retrieve original data segments for training set
    list_of_sampled_original_data_train = []
    if sampled_reshaped_indices_train: # Process only if there are sampled indices
        for reshaped_idx in sorted(sampled_reshaped_indices_train): # Sort to retrieve in index order
            start_original_row = reshaped_idx * n_cols
            # Extract the segment of n_cols rows from the original DataFrame
            segment = df_train_original.slice(start_original_row, n_cols)
            list_of_sampled_original_data_train.append(segment)

    if list_of_sampled_original_data_train:
        df_final_sampled_train = pl.concat(list_of_sampled_original_data_train, how="vertical")
        print(f"Final sampled original training data shape: {df_final_sampled_train.shape}")
        print(df_final_sampled_train)
        df_final_sampled_train.write_parquet(f"data_{args.data}/train.parquet")
    else:
        print("No data was sampled for training, or sampled_reshaped_indices_train was empty.")
        df_final_sampled_train = pl.DataFrame() # Empty DataFrame or appropriate handling

    # 3. Retrieve original data segments for validation set
    list_of_sampled_original_data_val = []
    if sampled_reshaped_indices_val: # Process only if there are sampled indices
        for reshaped_idx in sorted(sampled_reshaped_indices_val): # Sort to retrieve in index order
            start_original_row = reshaped_idx * n_cols
            segment = df_val_original.slice(start_original_row, n_cols)
            list_of_sampled_original_data_val.append(segment)

    if list_of_sampled_original_data_val:
        df_final_sampled_val = pl.concat(list_of_sampled_original_data_val, how="vertical")
        print(f"Final sampled original validation data shape: {df_final_sampled_val.shape}")
        print(df_final_sampled_val)
        df_final_sampled_val.write_parquet(f"data_{args.data}/val.parquet")
    else:
        print("No data was sampled for validation, or sampled_reshaped_indices_val was empty.")
        df_final_sampled_val = pl.DataFrame() # Empty DataFrame or appropriate handling


if __name__ == "__main__":
    main()
