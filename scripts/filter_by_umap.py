import umap
import matplotlib.pyplot as plt
import scienceplots
import polars as pl
import numpy as np
import random
from sklearn.cluster import KMeans
import argparse
import warnings
import os

warnings.filterwarnings("ignore")


# --- Helper Functions (Minimal modifications or as provided) ---
def reshape_df(df: pl.DataFrame, n_cols: int, column: str = "V") -> pl.DataFrame:
    V = df[column].to_numpy()
    if V.size == 0:
        # Handle empty data
        print(
            f"Warning: Column '{column}' is empty. Reshaping will result in an empty DataFrame."
        )
        # Create an empty DataFrame with the target schema if n_cols is known
        if n_cols > 0:
            reshaped_v_empty = np.array([]).reshape(0, n_cols)
            column_names_empty = [f"V{i}" for i in range(n_cols)]
            empty_df = pl.DataFrame(reshaped_v_empty, schema=column_names_empty)
            return empty_df.with_columns(pl.arange(0, empty_df.height).alias("index"))
        else:  # n_cols is 0 or invalid
            # Returning a truly empty DF or one with just index if schema is problematic
            return pl.DataFrame({"index": pl.Series(dtype=pl.Int64)})

    if V.size > 0 and n_cols > 0 and V.size % n_cols != 0:
        print(
            f"Warning: The size of the data ({V.size}) is not a multiple of n_cols ({n_cols}). Data will be truncated."
        )
        # Truncate V to be a multiple of n_cols
        V = V[: -(V.size % n_cols)]

    if V.size == 0 and n_cols > 0:
        reshaped_v = np.array([]).reshape(0, n_cols)  # No data to reshape
    elif V.size > 0 and n_cols > 0:
        reshaped_v = V.reshape(-1, n_cols)
    elif n_cols == 0:  # Cannot reshape into 0 columns if data exists
        raise ValueError(
            "n_cols cannot be zero if data is not empty and reshape is attempted."
        )
    else:  # V.size == 0 and n_cols == 0
        reshaped_v = np.array([]).reshape(0, 0)  # Essentially empty data

    column_names = [f"V{i}" for i in range(n_cols)]
    reshaped_df = pl.DataFrame(reshaped_v, schema=column_names)
    reshaped_df = reshaped_df.with_columns(
        pl.arange(0, reshaped_df.height).alias("index")
    )
    return reshaped_df


def umap_map_and_embed(
    df: pl.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[pl.DataFrame, umap.UMAP]:
    # Assumes 'index' column is present in df
    index_series = df.get_column("index")
    df_no_id = df.drop("index")

    cluster_series = None
    if "Cluster" in df.columns:  # Check in original df
        cluster_series = df.get_column("Cluster")
        if "Cluster" in df_no_id.columns:  # Drop from the copy if still there
            df_no_id = df_no_id.drop("Cluster")

    data_for_umap = df_no_id.to_numpy()  # UMAP works best with NumPy arrays

    if data_for_umap.shape[0] == 0:  # No data for UMAP
        print("Warning: Data for UMAP is empty. Returning empty UMAP DataFrame.")
        # Define schema for empty DataFrame
        schema = {"UMAP1": pl.Float64, "UMAP2": pl.Float64, "index": index_series.dtype}
        if cluster_series is not None:
            schema["Cluster"] = cluster_series.dtype
        empty_umap_df = pl.DataFrame(schema=schema)
        # Create mapper instance but do not fit/transform on empty data
        mapper = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state,
        )
        return empty_umap_df, mapper

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
    )
    mapper = reducer.fit(data_for_umap)
    embedding = mapper.transform(data_for_umap)

    umap_df = pl.DataFrame(embedding, schema=["UMAP1", "UMAP2"])
    umap_df = umap_df.with_columns(index_series.alias("index"))
    if cluster_series is not None:
        umap_df = umap_df.with_columns(cluster_series.alias("Cluster"))
    return umap_df, mapper


def kmeans_clustering(
    df: pl.DataFrame, n_clusters: int = 10, random_state: int = 42
) -> pl.DataFrame:
    # Input df is the result of umap_map_and_embed (contains UMAP1, UMAP2, index columns)
    index_series = df.get_column("index")

    # KMeans is performed on UMAP1, UMAP2
    df_for_kmeans = df.select(["UMAP1", "UMAP2"])
    data_for_kmeans = df_for_kmeans.to_numpy()

    if data_for_kmeans.shape[0] == 0:
        print("Warning: Data for KMeans is empty. Returning empty cluster DataFrame.")
        return pl.DataFrame(
            {"Cluster": [], "index": []},
            schema={"Cluster": pl.Int32, "index": index_series.dtype},
        )

    # Adjust n_clusters if it's larger than the number of samples, as KMeans would error.
    actual_n_clusters = min(n_clusters, data_for_kmeans.shape[0])
    if actual_n_clusters < n_clusters:
        print(
            f"Warning: n_clusters ({n_clusters}) > number of samples ({data_for_kmeans.shape[0]}). Using n_clusters = {actual_n_clusters}."
        )
    if actual_n_clusters == 0:  # No samples to cluster
        return pl.DataFrame(
            {"Cluster": [], "index": []},
            schema={"Cluster": pl.Int32, "index": index_series.dtype},
        )

    kmeans = KMeans(
        n_clusters=actual_n_clusters, random_state=random_state, n_init="auto"
    )
    labels = kmeans.fit_predict(data_for_kmeans)

    cluster_df = pl.DataFrame({"Cluster": labels})
    cluster_df = cluster_df.with_columns(index_series.alias("index"))
    return cluster_df


def sampling_by_cluster(
    df_with_features_and_clusters: pl.DataFrame,
    cluster_info_df: pl.DataFrame,
    n_target: int,
    n_clusters: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    n_each = (
        n_target // n_clusters if n_clusters > 0 else n_target
    )  # Prevent division by zero
    sampled_dfs_list = []
    sampled_clusters_info_list = []

    all_sampled_indices_flat = []

    for i in range(n_clusters):
        cluster_member_indices = (
            cluster_info_df.filter(pl.col("Cluster") == i)
            .get_column("index")
            .to_numpy()
        )

        if len(cluster_member_indices) == 0:
            continue

        sample_size_ideal = min(n_each, len(cluster_member_indices))

        if sample_size_ideal <= 0:
            # If n_target > 0 but n_each is 0 (e.g. n_target < n_clusters),
            # skip here and let remainder logic handle if possible.
            if n_target > 0 and len(cluster_member_indices) > 0 and n_each <= 0:
                pass
            else:
                continue

        if sample_size_ideal > 0:  # np.random.choice errors if size is 0
            chosen_indices_for_cluster = np.random.choice(
                cluster_member_indices, size=sample_size_ideal, replace=False
            )
            all_sampled_indices_flat.extend(chosen_indices_for_cluster)
            sampled_dfs_list.append(
                df_with_features_and_clusters.filter(
                    pl.col("index").is_in(chosen_indices_for_cluster)
                )
            )
            sampled_clusters_info_list.append(
                cluster_info_df.filter(
                    pl.col("index").is_in(chosen_indices_for_cluster)
                )
            )

    # Handle remaining samples
    current_total_sampled = sum(s_df.height for s_df in sampled_dfs_list)
    remaining_needed = n_target - current_total_sampled

    if remaining_needed > 0:
        # From all clustered indices, get those not yet sampled
        all_clustered_indices = cluster_info_df.get_column("index").to_numpy()
        available_for_remaining = np.setdiff1d(
            all_clustered_indices, np.array(all_sampled_indices_flat)
        )

        if len(available_for_remaining) > 0:
            take_additionally = min(remaining_needed, len(available_for_remaining))
            additional_chosen_indices = np.random.choice(
                available_for_remaining, size=take_additionally, replace=False
            )

            sampled_dfs_list.append(
                df_with_features_and_clusters.filter(
                    pl.col("index").is_in(additional_chosen_indices)
                )
            )
            sampled_clusters_info_list.append(
                cluster_info_df.filter(pl.col("index").is_in(additional_chosen_indices))
            )

    final_sampled_df = (
        pl.concat(sampled_dfs_list)
        if sampled_dfs_list
        else df_with_features_and_clusters.clear()
    )
    final_sampled_clusters_info = (
        pl.concat(sampled_clusters_info_list)
        if sampled_clusters_info_list
        else cluster_info_df.clear()
    )

    # If final sample count exceeds n_target, resample to n_target (randomly reduce)
    if final_sampled_df.height > n_target and n_target > 0:
        final_sampled_df = final_sampled_df.sample(
            n=n_target, seed=42
        )  # Fixed seed for reproducibility
        # Filter final_sampled_clusters_info to match
        final_indices = final_sampled_df.get_column("index").to_list()
        final_sampled_clusters_info = final_sampled_clusters_info.filter(
            pl.col("index").is_in(final_indices)
        )
    elif n_target == 0:  # If n_target is 0, intend to return empty DFs
        final_sampled_df = df_with_features_and_clusters.clear()
        final_sampled_clusters_info = cluster_info_df.clear()

    return final_sampled_df, final_sampled_clusters_info


# --- New Reusable Functions ---
def _ensure_dir_exists(file_path: str):
    """Ensures the directory for the given file_path exists."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(
        directory
    ):  # Check directory is not empty string
        os.makedirs(directory)


def plot_umap_data(
    umap_coords_df: pl.DataFrame,  # DataFrame with UMAP1, UMAP2 columns
    cluster_labels_series: pl.Series,  # Polars Series of cluster labels
    title: str,
    filename: str,
):
    """Plots UMAP data, colored by cluster labels."""
    _ensure_dir_exists(filename)
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        color_data = cluster_labels_series.to_numpy().astype(int)
        ax.scatter(
            umap_coords_df["UMAP1"].to_numpy(),
            umap_coords_df["UMAP2"].to_numpy(),
            c=color_data,
            cmap="Spectral",
            s=1,
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        fig.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved UMAP plot: {filename}")


def save_sampled_data_from_original(
    original_long_df: pl.DataFrame,
    sampled_reshaped_df: pl.DataFrame,  # This df must have an "index" column from reshape_df
    n_cols_reshape: int,
    output_path: str,
    dataset_label_for_log: str,
):
    """Retrieves segments from original_long_df based on indices in sampled_reshaped_df and saves to Parquet."""
    if sampled_reshaped_df.is_empty():
        print(
            f"Sampled reshaped DataFrame for {dataset_label_for_log} is empty. No data saved to {output_path}."
        )
        # Optionally, save an empty Parquet file with schema
        # _ensure_dir_exists(output_path)
        # pl.DataFrame(schema=original_long_df.schema).write_parquet(output_path)
        return

    _ensure_dir_exists(output_path)

    sampled_indices = sampled_reshaped_df.get_column("index").to_list()

    list_of_sampled_original_data = []
    if sampled_indices:
        for reshaped_idx in sorted(sampled_indices):
            start_original_row = reshaped_idx * n_cols_reshape
            segment = original_long_df.slice(start_original_row, n_cols_reshape)
            list_of_sampled_original_data.append(segment)

    if list_of_sampled_original_data:
        df_final_sampled_original = pl.concat(
            list_of_sampled_original_data, how="vertical"
        )
        print(
            f"Final sampled original {dataset_label_for_log} data shape: {df_final_sampled_original.shape}"
        )
        print(df_final_sampled_original.head(min(3, df_final_sampled_original.height)))
        df_final_sampled_original.write_parquet(output_path)
        print(f"Saved final sampled {dataset_label_for_log} data to {output_path}")
    else:
        print(
            f"No data was sampled for {dataset_label_for_log} (list_of_sampled_original_data was empty). File {output_path} not written."
        )


# --- Main Function (Original flow largely preserved) ---
def main():
    # Original seed settings
    np.random.seed(42)
    random.seed(42)
    pl.set_random_seed(42)  # Added for reproducibility of Polars operations

    parser = argparse.ArgumentParser(description="UMAP and KMeans clustering")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Suffix for data files (e.g., 'normal', 'more', 'much', 'test')",
    )
    args = parser.parse_args()

    n_cols = 100  # Original setting value

    # --- "test" data processing path ---
    if args.data == "test":
        print(f"--- Processing Test Data (Scenario: {args.data}) ---")
        try:
            df_test_original = pl.read_parquet(f"data_{args.data}/test_cand.parquet")
        except Exception as e:  # Handle general exceptions including FileNotFoundError
            print(f"Error loading test data (data_{args.data}/test_cand.parquet): {e}")
            return

        if df_test_original.is_empty():
            print("Test data is empty. Exiting.")
            return

        # 1. Reshape
        df_active_reshaped = reshape_df(df_test_original, n_cols)

        # 2. UMAP
        umap_active, _ = umap_map_and_embed(
            df_active_reshaped,
            n_neighbors=10,
            min_dist=0.001,
            n_components=2,
            random_state=42,
        )
        print(f"UMAP Test DataFrame shape: {umap_active.shape}")
        print(
            umap_active.head(min(3, umap_active.height))
        )  # Similar to original train output

        # 3. KMeans
        cluster_active = kmeans_clustering(umap_active, n_clusters=10, random_state=42)
        print(f"Cluster Test DataFrame shape: {cluster_active.shape}")
        print(cluster_active.head(min(3, cluster_active.height)))

        # 4. Plot UMAP (full test data)
        if (
            not umap_active.is_empty() and not cluster_active.is_empty()
        ):  # Ensure data for plotting
            plot_umap_data(
                umap_coords_df=umap_active.select(
                    ["UMAP1", "UMAP2"]
                ),  # Pass only UMAP coordinate columns
                cluster_labels_series=cluster_active.get_column("Cluster"),
                title=f"UMAP projection of test data ({args.data})",
                filename=f"figs/umap_projection_{args.data}.png",  # Add "test_" prefix for distinction
            )
        else:
            print(
                "Skipping UMAP plot for full test data: UMAP or cluster data is empty."
            )

        # 5. Define sampling target size for test data
        num_reshaped_items = df_active_reshaped.height
        # Example: 20% of reshaped items or a minimum of 100 (adjust as needed)
        n_target_active = num_reshaped_items // 10 if num_reshaped_items > 0 else 0
        print(
            f"Target samples for test: {n_target_active} from {num_reshaped_items} reshaped items."
        )
        n_clusters_sampling = 10  # Same as in KMeans

        # 6. Sample data
        df_active_with_clusters = df_active_reshaped.join(
            cluster_active, on="index", how="left"
        )
        sampled_active, sampled_clusters_active_df = sampling_by_cluster(
            df_active_with_clusters,
            cluster_active,
            n_target_active,
            n_clusters_sampling,
        )
        print(f"Sampled Test DataFrame shape: {sampled_active.shape}")
        print(sampled_active.head(min(3, sampled_active.height)))

        # 7. Plot UMAP (sampled test data)
        if (
            not sampled_active.is_empty()
            and "Cluster" in sampled_clusters_active_df.columns
            and not sampled_clusters_active_df.is_empty()
        ):
            sampled_active_for_umap = sampled_active.join(
                sampled_clusters_active_df.select(["index", "Cluster"]),
                on="index",
                how="left",
            )
            umap_sampled_active, _ = umap_map_and_embed(
                sampled_active_for_umap,
                n_neighbors=10,
                min_dist=0.001,
                n_components=2,
                random_state=42,
            )
            if (
                not umap_sampled_active.is_empty()
                and "Cluster" in umap_sampled_active.columns
            ):  # Also check umap result
                plot_umap_data(
                    umap_coords_df=umap_sampled_active.select(["UMAP1", "UMAP2"]),
                    cluster_labels_series=umap_sampled_active.get_column("Cluster"),
                    title=f"UMAP projection of sampled test data ({args.data})",
                    filename=f"figs/umap_projection_sampled_{args.data}.png",
                )
            else:
                print(
                    "UMAP on sampled test data resulted in empty or no cluster info. Skipping plot."
                )
        else:
            print(
                "Skipping UMAP plot for sampled test data: empty or lacks cluster information."
            )

        # 8. Save sampled data
        save_sampled_data_from_original(
            original_long_df=df_test_original,
            sampled_reshaped_df=sampled_active,  # Contains 'index' column
            n_cols_reshape=n_cols,
            output_path=f"data_{args.data}/test.parquet",
            dataset_label_for_log="test",
        )

    # --- Original "normal", "more", "much" data processing path ---
    else:
        print(f"--- Processing Training & Validation Data (Scenario: {args.data}) ---")
        try:
            df_train_original = pl.read_parquet(f"data_{args.data}/train_cand.parquet")
            df_val_original = pl.read_parquet(f"data_{args.data}/val_cand.parquet")
        except Exception as e:
            print(f"Error loading train/val data for scenario {args.data}: {e}")
            return

        # --- Training Data Processing (mostly original logic) ---
        print("--- Training Data ---")
        df_train = reshape_df(df_train_original, n_cols)
        # mapper_train was not used in the original relevant section for UMAP plot of full train data
        umap_train, _ = umap_map_and_embed(
            df_train, n_neighbors=10, min_dist=0.001, n_components=2, random_state=42
        )
        print(f"UMAP Training DataFrame shape: {umap_train.shape}")
        print(
            umap_train.head(min(3, umap_train.height))
        )  # Maintain original output format

        cluster_train = kmeans_clustering(umap_train, n_clusters=10, random_state=42)
        print(f"Cluster Training DataFrame shape: {cluster_train.shape}")
        print(cluster_train.head(min(3, cluster_train.height)))

        # Plot UMAP (full training data) - use original filename
        if not umap_train.is_empty() and not cluster_train.is_empty():
            plot_umap_data(
                umap_coords_df=umap_train.select(["UMAP1", "UMAP2"]),
                cluster_labels_series=cluster_train.get_column("Cluster"),
                title=f"UMAP projection of training data ({args.data})",
                filename=f"figs/umap_projection_{args.data}.png",  # Original filename
            )
        else:
            print(
                "Skipping UMAP plot for full training data: UMAP or cluster data is empty."
            )

        # --- Validation Data Processing (mostly original logic) ---
        print("--- Validation Data ---")
        df_val = reshape_df(df_val_original, n_cols)
        umap_val, _ = umap_map_and_embed(
            df_val, n_neighbors=10, min_dist=0.001, n_components=2, random_state=42
        )
        print(f"UMAP Validation DataFrame shape: {umap_val.shape}")

        cluster_val = kmeans_clustering(umap_val, n_clusters=10, random_state=42)
        print(f"Cluster Validation DataFrame shape: {cluster_val.shape}")

        # No UMAP plot for full validation data in original code.

        # --- Define Sampling Target Sizes (original logic) ---
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
            # Original else block (Default or error handling)
            print(
                f"Warning: Unknown data size '{args.data}'. Using default small sample sizes."
            )
            n_target_train = 1000
            n_target_val = 200
        n_clusters_sampling = 10  # Value used in original code

        # --- Sample Training Data (original logic) ---
        print("--- Sampling Training Data ---")
        df_train_with_clusters = df_train.join(cluster_train, on="index", how="left")
        sampled_train, sampled_clusters_train_df = sampling_by_cluster(
            df_train_with_clusters, cluster_train, n_target_train, n_clusters_sampling
        )
        print(f"Sampled Training DataFrame shape: {sampled_train.shape}")
        print(sampled_train.head(min(3, sampled_train.height)))

        # Plot UMAP (sampled training data) - use original filename
        if (
            not sampled_train.is_empty()
            and "Cluster" in sampled_clusters_train_df.columns
            and not sampled_clusters_train_df.is_empty()
        ):
            sampled_train_for_umap = sampled_train.join(
                sampled_clusters_train_df.select(["index", "Cluster"]),
                on="index",
                how="left",
            )
            sampled_umap_train, _ = umap_map_and_embed(
                sampled_train_for_umap,
                n_neighbors=10,
                min_dist=0.001,
                n_components=2,
                random_state=42,
            )
            if (
                not sampled_umap_train.is_empty()
                and "Cluster" in sampled_umap_train.columns
            ):  # Also check umap result
                plot_umap_data(
                    umap_coords_df=sampled_umap_train.select(["UMAP1", "UMAP2"]),
                    cluster_labels_series=sampled_umap_train.get_column("Cluster"),
                    title=f"UMAP projection of sampled training data ({args.data})",
                    filename=f"figs/umap_projection_sampled_{args.data}.png",  # Original filename
                )
            else:
                print(
                    "UMAP on sampled training data resulted in empty or no cluster info. Skipping plot."
                )
        else:
            print(
                "Skipping UMAP plot for sampled training data as it's empty or lacks cluster information."
            )

        # --- Sample Validation Data (original logic) ---
        print("--- Sampling Validation Data ---")
        df_val_with_clusters = df_val.join(cluster_val, on="index", how="left")
        # Original: sampled_val, sampled_clusters_val_df = ... (sampled_clusters_val_df not used later)
        sampled_val, _ = sampling_by_cluster(
            df_val_with_clusters, cluster_val, n_target_val, n_clusters_sampling
        )
        print(f"Sampled Validation DataFrame shape: {sampled_val.shape}")

        # No UMAP plot for sampled validation data in original code.

        # --- Save Sampled Data (original logic) ---
        save_sampled_data_from_original(
            original_long_df=df_train_original,
            sampled_reshaped_df=sampled_train,  # Contains 'index' column
            n_cols_reshape=n_cols,
            output_path=f"data_{args.data}/train.parquet",  # Original path
            dataset_label_for_log="training",
        )
        save_sampled_data_from_original(
            original_long_df=df_val_original,
            sampled_reshaped_df=sampled_val,  # Contains 'index' column
            n_cols_reshape=n_cols,
            output_path=f"data_{args.data}/val.parquet",  # Original path
            dataset_label_for_log="validation",
        )


if __name__ == "__main__":
    main()
