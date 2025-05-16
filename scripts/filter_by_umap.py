import umap
import matplotlib.pyplot as plt
import scienceplots
import fireducks.pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
import argparse
import warnings
import os

warnings.filterwarnings("ignore")


# --- Helper Functions (Converted from Polars to Pandas) ---
def reshape_df(df: pd.DataFrame, n_cols: int, column: str = "V") -> pd.DataFrame:
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
            empty_df = pd.DataFrame(reshaped_v_empty, columns=column_names_empty)
            empty_df['index'] = pd.Series(dtype=np.int64)
            return empty_df
        else:  # n_cols is 0 or invalid
            # Returning a truly empty DF or one with just index if schema is problematic
            return pd.DataFrame({"index": pd.Series(dtype=np.int64)})

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
    reshaped_df = pd.DataFrame(reshaped_v, columns=column_names)
    reshaped_df['index'] = np.arange(len(reshaped_df))
    return reshaped_df


def umap_map_and_embed(
    df: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, umap.UMAP]:
    # Assumes 'index' column is present in df
    index_series = df['index'].copy()
    df_no_id = df.drop(columns=['index'])

    cluster_series = None
    if "Cluster" in df.columns:  # Check in original df
        cluster_series = df["Cluster"].copy()
        if "Cluster" in df_no_id.columns:  # Drop from the copy if still there
            df_no_id = df_no_id.drop(columns=["Cluster"])

    data_for_umap = df_no_id.to_numpy()  # UMAP works best with NumPy arrays

    if len(data_for_umap) == 0:  # No data for UMAP
        print("Warning: Data for UMAP is empty. Returning empty UMAP DataFrame.")
        # Define schema for empty DataFrame
        empty_umap_df = pd.DataFrame(columns=["UMAP1", "UMAP2", "index"])
        if cluster_series is not None:
            empty_umap_df["Cluster"] = pd.Series(dtype=cluster_series.dtype)
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

    umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
    umap_df["index"] = index_series.values
    if cluster_series is not None:
        umap_df["Cluster"] = cluster_series.values
    return umap_df, mapper


def kmeans_clustering(
    df: pd.DataFrame, n_clusters: int = 10, random_state: int = 42
) -> pd.DataFrame:
    # Input df is the result of umap_map_and_embed (contains UMAP1, UMAP2, index columns)
    index_series = df['index'].copy()

    # KMeans is performed on UMAP1, UMAP2
    df_for_kmeans = df[["UMAP1", "UMAP2"]]
    data_for_kmeans = df_for_kmeans.to_numpy()

    if len(data_for_kmeans) == 0:
        print("Warning: Data for KMeans is empty. Returning empty cluster DataFrame.")
        return pd.DataFrame(
            {"Cluster": [], "index": []},
            dtype={"Cluster": np.int32, "index": index_series.dtype},
        )

    # Adjust n_clusters if it's larger than the number of samples, as KMeans would error.
    actual_n_clusters = min(n_clusters, data_for_kmeans.shape[0])
    if actual_n_clusters < n_clusters:
        print(
            f"Warning: n_clusters ({n_clusters}) > number of samples ({data_for_kmeans.shape[0]}). Using n_clusters = {actual_n_clusters}."
        )
    if actual_n_clusters == 0:  # No samples to cluster
        return pd.DataFrame(
            {"Cluster": [], "index": []},
            dtype={"Cluster": np.int32, "index": index_series.dtype},
        )

    kmeans = KMeans(
        n_clusters=actual_n_clusters, random_state=random_state, n_init="auto"
    )
    labels = kmeans.fit_predict(data_for_kmeans)

    cluster_df = pd.DataFrame({"Cluster": labels})
    cluster_df["index"] = index_series.values
    return cluster_df


def sampling_by_cluster(
    df_with_features_and_clusters: pd.DataFrame,
    cluster_info_df: pd.DataFrame,
    n_target: int,
    n_clusters: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_each = (
        n_target // n_clusters if n_clusters > 0 else n_target
    )  # Prevent division by zero
    sampled_dfs_list = []
    sampled_clusters_info_list = []

    all_sampled_indices_flat = []

    for i in range(n_clusters):
        cluster_member_indices = (
            cluster_info_df[cluster_info_df["Cluster"] == i]["index"].to_numpy()
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
                df_with_features_and_clusters[
                    df_with_features_and_clusters["index"].isin(chosen_indices_for_cluster)
                ]
            )
            sampled_clusters_info_list.append(
                cluster_info_df[
                    cluster_info_df["index"].isin(chosen_indices_for_cluster)
                ]
            )

    # Handle remaining samples
    current_total_sampled = sum(len(s_df) for s_df in sampled_dfs_list)
    remaining_needed = n_target - current_total_sampled

    if remaining_needed > 0:
        # From all clustered indices, get those not yet sampled
        all_clustered_indices = cluster_info_df["index"].to_numpy()
        available_for_remaining = np.setdiff1d(
            all_clustered_indices, np.array(all_sampled_indices_flat)
        )

        if len(available_for_remaining) > 0:
            take_additionally = min(remaining_needed, len(available_for_remaining))
            additional_chosen_indices = np.random.choice(
                available_for_remaining, size=take_additionally, replace=False
            )

            sampled_dfs_list.append(
                df_with_features_and_clusters[
                    df_with_features_and_clusters["index"].isin(additional_chosen_indices)
                ]
            )
            sampled_clusters_info_list.append(
                cluster_info_df[
                    cluster_info_df["index"].isin(additional_chosen_indices)
                ]
            )

    final_sampled_df = (
        pd.concat(sampled_dfs_list)
        if sampled_dfs_list
        else df_with_features_and_clusters.iloc[0:0]  # Empty DataFrame with same structure
    )
    final_sampled_clusters_info = (
        pd.concat(sampled_clusters_info_list)
        if sampled_clusters_info_list
        else cluster_info_df.iloc[0:0]  # Empty DataFrame with same structure
    )

    # If final sample count exceeds n_target, resample to n_target (randomly reduce)
    if len(final_sampled_df) > n_target and n_target > 0:
        final_sampled_df = final_sampled_df.sample(
            n=n_target, random_state=42
        )  # Fixed seed for reproducibility
        # Filter final_sampled_clusters_info to match
        final_indices = final_sampled_df["index"].tolist()
        final_sampled_clusters_info = final_sampled_clusters_info[
            final_sampled_clusters_info["index"].isin(final_indices)
        ]
    elif n_target == 0:  # If n_target is 0, intend to return empty DFs
        final_sampled_df = df_with_features_and_clusters.iloc[0:0]
        final_sampled_clusters_info = cluster_info_df.iloc[0:0]

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
    umap_coords_df: pd.DataFrame,  # DataFrame with UMAP1, UMAP2 columns
    cluster_labels_series: pd.Series,  # Pandas Series of cluster labels
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
    original_long_df: pd.DataFrame,
    sampled_reshaped_df: pd.DataFrame,  # This df must have an "index" column from reshape_df
    n_cols_reshape: int,
    output_path: str,
    dataset_label_for_log: str,
):
    """Retrieves segments from original_long_df based on indices in sampled_reshaped_df and saves to Parquet."""
    if len(sampled_reshaped_df) == 0:
        print(
            f"Sampled reshaped DataFrame for {dataset_label_for_log} is empty. No data saved to {output_path}."
        )
        return

    _ensure_dir_exists(output_path)

    sampled_indices = sampled_reshaped_df["index"].tolist()

    list_of_sampled_original_data = []
    if sampled_indices:
        for reshaped_idx in sorted(sampled_indices):
            start_original_row = reshaped_idx * n_cols_reshape
            segment = original_long_df.iloc[start_original_row:start_original_row + n_cols_reshape]
            list_of_sampled_original_data.append(segment)

    if list_of_sampled_original_data:
        df_final_sampled_original = pd.concat(
            list_of_sampled_original_data, ignore_index=True
        )
        print(
            f"Final sampled original {dataset_label_for_log} data shape: {df_final_sampled_original.shape}"
        )
        print(df_final_sampled_original.head(min(3, len(df_final_sampled_original))))
        df_final_sampled_original.to_parquet(output_path)
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

    parser = argparse.ArgumentParser(description="UMAP and KMeans clustering")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Suffix for data files (e.g., 'normal', 'more', 'much', 'test')",
    )
    args = parser.parse_args()

    n_cols = 100
    n_clusters = 100

    # --- "test" data processing path ---
    if args.data == "test":
        print(f"--- Processing Test Data (Scenario: {args.data}) ---")
        try:
            df_test_original = pd.read_parquet(f"data_{args.data}/test_cand.parquet")
        except Exception as e:  # Handle general exceptions including FileNotFoundError
            print(f"Error loading test data (data_{args.data}/test_cand.parquet): {e}")
            return

        if len(df_test_original) == 0:
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
            umap_active.head(min(3, len(umap_active)))
        )  # Similar to original train output

        # 3. KMeans
        cluster_active = kmeans_clustering(umap_active, n_clusters=n_clusters, random_state=42)
        print(f"Cluster Test DataFrame shape: {cluster_active.shape}")
        print(cluster_active.head(min(3, len(cluster_active))))

        # 4. Plot UMAP (full test data)
        if (
            len(umap_active) > 0 and len(cluster_active) > 0
        ):  # Ensure data for plotting
            plot_umap_data(
                umap_coords_df=umap_active[["UMAP1", "UMAP2"]],  # Pass only UMAP coordinate columns
                cluster_labels_series=cluster_active["Cluster"],
                title=f"UMAP projection of test data ({args.data})",
                filename=f"figs/umap_projection_{args.data}.png",  # Add "test_" prefix for distinction
            )
        else:
            print(
                "Skipping UMAP plot for full test data: UMAP or cluster data is empty."
            )

        # 5. Define sampling target size for test data
        num_reshaped_items = len(df_active_reshaped)
        # Example: 20% of reshaped items or a minimum of 100 (adjust as needed)
        n_target_active = num_reshaped_items // n_clusters if num_reshaped_items > 0 else 0
        print(
            f"Target samples for test: {n_target_active} from {num_reshaped_items} reshaped items."
        )
        n_clusters_sampling = n_clusters  # Same as in KMeans

        # 6. Sample data
        df_active_with_clusters = df_active_reshaped.merge(
            cluster_active, on="index", how="left"
        )
        sampled_active, sampled_clusters_active_df = sampling_by_cluster(
            df_active_with_clusters,
            cluster_active,
            n_target_active,
            n_clusters_sampling,
        )
        print(f"Sampled Test DataFrame shape: {sampled_active.shape}")
        print(sampled_active.head(min(3, len(sampled_active))))

        # 7. Plot UMAP (sampled test data)
        if (
            len(sampled_active) > 0
            and "Cluster" in sampled_clusters_active_df.columns
            and len(sampled_clusters_active_df) > 0
        ):
            sampled_active_for_umap = sampled_active.merge(
                sampled_clusters_active_df[["index", "Cluster"]],
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
                len(umap_sampled_active) > 0
                and "Cluster" in umap_sampled_active.columns
            ):  # Also check umap result
                plot_umap_data(
                    umap_coords_df=umap_sampled_active[["UMAP1", "UMAP2"]],
                    cluster_labels_series=umap_sampled_active["Cluster"],
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
            df_train_original = pd.read_parquet(f"data_{args.data}/train_cand.parquet")
            df_val_original = pd.read_parquet(f"data_{args.data}/val_cand.parquet")
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
            umap_train.head(min(3, len(umap_train)))
        )  # Maintain original output format

        cluster_train = kmeans_clustering(umap_train, n_clusters=n_clusters, random_state=42)
        print(f"Cluster Training DataFrame shape: {cluster_train.shape}")
        print(cluster_train.head(min(3, len(cluster_train))))

        # Plot UMAP (full training data) - use original filename
        if len(umap_train) > 0 and len(cluster_train) > 0:
            plot_umap_data(
                umap_coords_df=umap_train[["UMAP1", "UMAP2"]],
                cluster_labels_series=cluster_train["Cluster"],
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

        cluster_val = kmeans_clustering(umap_val, n_clusters=n_clusters, random_state=42)
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
        n_clusters_sampling = n_clusters

        # --- Sample Training Data (original logic) ---
        print("--- Sampling Training Data ---")
        df_train_with_clusters = df_train.merge(cluster_train, on="index", how="left")
        sampled_train, sampled_clusters_train_df = sampling_by_cluster(
            df_train_with_clusters, cluster_train, n_target_train, n_clusters_sampling
        )
        print(f"Sampled Training DataFrame shape: {sampled_train.shape}")
        print(sampled_train.head(min(3, len(sampled_train))))
        
        # Plot UMAP (sampled training data)
        if (
            len(sampled_train) > 0
            and "Cluster" in sampled_train.columns
        ):
            sampled_umap_train, _ = umap_map_and_embed(
                sampled_train,
                n_neighbors=10,
                min_dist=0.001,
                n_components=2,
                random_state=42,
            )
            if (
                len(sampled_umap_train) > 0
                and "Cluster" in sampled_umap_train.columns
            ):
                plot_umap_data(
                    umap_coords_df=sampled_umap_train[["UMAP1", "UMAP2"]],
                    cluster_labels_series=sampled_umap_train["Cluster"],
                    title=f"UMAP projection of sampled training data ({args.data})",
                    filename=f"figs/umap_projection_sampled_{args.data}.png",
                )
            else:
                print(
                    "UMAP on sampled training data resulted in empty or no cluster info (after UMAP). Skipping plot."
                )
        else:
            print(
                "Skipping UMAP plot for sampled training data: sampled_train is empty or lacks 'Cluster' column."
            )

        # --- Sample Validation Data (original logic) ---
        print("--- Sampling Validation Data ---")
        df_val_with_clusters = df_val.merge(cluster_val, on="index", how="left")
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
