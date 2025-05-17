import umap
import matplotlib.pyplot as plt
import scienceplots
import fireducks.pandas as pd # Assuming fireducks.pandas is a pandas-compatible library
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score # Added for silhouette score
import argparse
import warnings
import os

warnings.filterwarnings("ignore")


# --- Helper Functions (Converted from Polars to Pandas) ---
def reshape_df(df: pd.DataFrame, n_cols: int, column: str = "V") -> pd.DataFrame:
    V = df[column].to_numpy()
    if V.size == 0:
        print(
            f"Warning: Column '{column}' is empty. Reshaping will result in an empty DataFrame."
        )
        if n_cols > 0:
            reshaped_v_empty = np.array([]).reshape(0, n_cols)
            column_names_empty = [f"V{i}" for i in range(n_cols)]
            empty_df = pd.DataFrame(reshaped_v_empty, columns=column_names_empty)
            empty_df['index'] = pd.Series(dtype=np.int64)
            return empty_df
        else:
            return pd.DataFrame({"index": pd.Series(dtype=np.int64)})

    if V.size > 0 and n_cols > 0 and V.size % n_cols != 0:
        print(
            f"Warning: The size of the data ({V.size}) is not a multiple of n_cols ({n_cols}). Data will be truncated."
        )
        V = V[: -(V.size % n_cols)]

    if V.size == 0 and n_cols > 0:
        reshaped_v = np.array([]).reshape(0, n_cols)
    elif V.size > 0 and n_cols > 0:
        reshaped_v = V.reshape(-1, n_cols)
    elif n_cols == 0:
        raise ValueError(
            "n_cols cannot be zero if data is not empty and reshape is attempted."
        )
    else:
        reshaped_v = np.array([]).reshape(0, 0)

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
    index_series = df['index'].copy()
    df_no_id = df.drop(columns=['index'])

    cluster_series = None
    if "Cluster" in df.columns:
        cluster_series = df["Cluster"].copy()
        if "Cluster" in df_no_id.columns:
            df_no_id = df_no_id.drop(columns=["Cluster"])

    data_for_umap = df_no_id.to_numpy()

    if len(data_for_umap) == 0:
        print("Warning: Data for UMAP is empty. Returning empty UMAP DataFrame.")
        empty_umap_df = pd.DataFrame(columns=["UMAP1", "UMAP2", "index"])
        if cluster_series is not None:
            empty_umap_df["Cluster"] = pd.Series(dtype=cluster_series.dtype)
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
    index_series = df['index'].copy()
    df_for_kmeans = df[["UMAP1", "UMAP2"]]
    data_for_kmeans = df_for_kmeans.to_numpy()

    if len(data_for_kmeans) == 0:
        print("Warning: Data for KMeans is empty. Returning empty cluster DataFrame.")
        return pd.DataFrame(
            {"Cluster": [], "index": []},
            dtype={"Cluster": np.int32, "index": index_series.dtype},
        )

    actual_n_clusters = min(n_clusters, data_for_kmeans.shape[0])
    if actual_n_clusters < n_clusters:
        print(
            f"Warning: n_clusters ({n_clusters}) > number of samples ({data_for_kmeans.shape[0]}). Using n_clusters = {actual_n_clusters}."
        )
    if actual_n_clusters == 0:
        return pd.DataFrame(
            {"Cluster": [], "index": []},
            dtype={"Cluster": np.int32, "index": index_series.dtype},
        )
    # Ensure at least 1 cluster if actual_n_clusters became 0 but data exists.
    # KMeans requires n_clusters >= 1 if data is present.
    # However, silhouette score needs at least 2 clusters. This is handled in find_optimal_n_clusters.
    if actual_n_clusters < 1 and len(data_for_kmeans) > 0 : # Should not happen if min(n_clusters, len) is used properly.
        print(f"Warning: actual_n_clusters is {actual_n_clusters}, setting to 1 for KMeans fit.")
        actual_n_clusters = 1


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
    n_clusters_sampling: int, # Renamed from n_clusters to avoid confusion
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Prevent division by zero if n_clusters_sampling is 0 (e.g., if optimal k search failed or resulted in 0/1)
    if n_clusters_sampling <= 0:
        print(f"Warning: n_clusters_sampling is {n_clusters_sampling}. Sampling will be random from all available if n_target > 0.")
        if n_target > 0 and not df_with_features_and_clusters.empty:
            # Sample randomly from the whole dataset if no valid clusters to sample from
            if n_target >= len(df_with_features_and_clusters):
                 return df_with_features_and_clusters.copy(), cluster_info_df.copy() # Return all
            return df_with_features_and_clusters.sample(n=n_target, random_state=42), cluster_info_df[cluster_info_df['index'].isin(df_with_features_and_clusters.sample(n=n_target, random_state=42)['index'])]
        else:
            return df_with_features_and_clusters.iloc[0:0], cluster_info_df.iloc[0:0]


    n_each = n_target // n_clusters_sampling
    sampled_dfs_list = []
    sampled_clusters_info_list = []
    all_sampled_indices_flat = []

    for i in range(n_clusters_sampling):
        cluster_member_indices = (
            cluster_info_df[cluster_info_df["Cluster"] == i]["index"].to_numpy()
        )

        if len(cluster_member_indices) == 0:
            continue

        sample_size_ideal = min(n_each, len(cluster_member_indices))

        if sample_size_ideal <= 0:
            if n_target > 0 and len(cluster_member_indices) > 0 and n_each <= 0:
                pass
            else:
                continue

        if sample_size_ideal > 0:
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

    current_total_sampled = sum(len(s_df) for s_df in sampled_dfs_list)
    remaining_needed = n_target - current_total_sampled

    if remaining_needed > 0:
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
        else df_with_features_and_clusters.iloc[0:0]
    )
    final_sampled_clusters_info = (
        pd.concat(sampled_clusters_info_list)
        if sampled_clusters_info_list
        else cluster_info_df.iloc[0:0]
    )

    if len(final_sampled_df) > n_target and n_target > 0:
        final_sampled_df = final_sampled_df.sample(
            n=n_target, random_state=42
        )
        final_indices = final_sampled_df["index"].tolist()
        final_sampled_clusters_info = final_sampled_clusters_info[
            final_sampled_clusters_info["index"].isin(final_indices)
        ]
    elif n_target == 0:
        final_sampled_df = df_with_features_and_clusters.iloc[0:0]
        final_sampled_clusters_info = cluster_info_df.iloc[0:0]

    return final_sampled_df, final_sampled_clusters_info


# --- New Reusable Functions ---
def _ensure_dir_exists(file_path: str):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def plot_umap_data(
    umap_coords_df: pd.DataFrame,
    cluster_labels_series: pd.Series,
    title: str,
    filename: str,
):
    _ensure_dir_exists(filename)
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        # Ensure cluster_labels_series is not empty and convert to int for coloring
        if not cluster_labels_series.empty:
            color_data = cluster_labels_series.to_numpy().astype(int)
            # Check if UMAP coordinates are available
            if not umap_coords_df.empty and "UMAP1" in umap_coords_df and "UMAP2" in umap_coords_df:
                 scatter = ax.scatter(
                    umap_coords_df["UMAP1"].to_numpy(),
                    umap_coords_df["UMAP2"].to_numpy(),
                    c=color_data,
                    cmap="Spectral",
                    s=1,
                    linewidths=0,
                )
                 # Add colorbar if there's more than one unique cluster
                 if len(np.unique(color_data)) > 1:
                    plt.colorbar(scatter, ax=ax, label='Cluster ID')
            else:
                print(f"Warning: UMAP coordinate data is empty or invalid for plotting '{title}'.")
        else:
            # Plot without colors if cluster labels are not available
            if not umap_coords_df.empty and "UMAP1" in umap_coords_df and "UMAP2" in umap_coords_df:
                ax.scatter(
                    umap_coords_df["UMAP1"].to_numpy(),
                    umap_coords_df["UMAP2"].to_numpy(),
                    s=1,
                    linewidths=0,
                )
                print(f"Warning: Cluster labels are empty for UMAP plot '{title}'. Plotting without colors.")
            else:
                print(f"Warning: UMAP coordinate data and cluster labels are empty for plotting '{title}'. Skipping scatter plot.")


        ax.set_title(title)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        fig.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved UMAP plot: {filename}")


def save_sampled_data_from_original(
    original_long_df: pd.DataFrame,
    sampled_reshaped_df: pd.DataFrame,
    n_cols_reshape: int,
    output_path: str,
    dataset_label_for_log: str,
):
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
        df_final_sampled_original.to_parquet(output_path)
        print(f"Saved final sampled {dataset_label_for_log} data to {output_path}")
    else:
        print(
            f"No data was sampled for {dataset_label_for_log}. File {output_path} not written."
        )

def find_optimal_n_clusters_silhouette(
    umap_data_df: pd.DataFrame,
    k_range: tuple[int, int],
    default_n_clusters: int,
    dataset_label: str,
    random_state: int = 42
) -> int:
    """
    Finds the optimal number of clusters using the silhouette score.
    """
    if umap_data_df.empty or len(umap_data_df) < k_range[0]:
        print(f"Warning ({dataset_label}): Data for silhouette score is empty or too small ({len(umap_data_df)} samples). Using default n_clusters: {default_n_clusters}.")
        return default_n_clusters

    data_for_k_selection = umap_data_df[["UMAP1", "UMAP2"]].to_numpy()
    
    # Adjust k_range if k_max is greater than number of samples
    # Silhouette score requires at least 2 labels, so k must be >= 2.
    # KMeans n_clusters must be < n_samples.
    actual_k_min = max(2, k_range[0])
    actual_k_max = min(k_range[1], len(data_for_k_selection) -1) # Kmeans n_clusters must be < n_samples if n_samples > 1

    if actual_k_min > actual_k_max :
        print(f"Warning ({dataset_label}): Adjusted k_range [{actual_k_min}, {actual_k_max}] is invalid for silhouette score calculation ({len(data_for_k_selection)} samples). Using default n_clusters: {default_n_clusters}.")
        if len(data_for_k_selection) >=2 : # Can we use at least 2 clusters?
             return 2 
        return default_n_clusters


    silhouette_scores = []
    valid_k_values = list(range(actual_k_min, actual_k_max + 1))
    
    print(f"--- ({dataset_label}) Finding optimal n_clusters using Silhouette Score (k_range: {valid_k_values}) ---")

    for k_val in valid_k_values:
        kmeans_temp = KMeans(n_clusters=k_val, random_state=random_state, n_init="auto")
        try:
            cluster_labels_temp = kmeans_temp.fit_predict(data_for_k_selection)
        except ValueError as e:
            print(f"Error fitting KMeans for k={k_val} on {dataset_label}: {e}. Skipping this k.")
            silhouette_scores.append(-1) # Indicate failure for this k
            continue

        if len(np.unique(cluster_labels_temp)) < 2:
            # Silhouette score is not defined if only one cluster is formed.
            silhouette_scores.append(-1) # Or some other indicator like float('-inf')
        else:
            score = silhouette_score(data_for_k_selection, cluster_labels_temp)
            silhouette_scores.append(score)
            # print(f"({dataset_label}) k={k_val}, Silhouette Score: {score:.4f}")


    if not silhouette_scores or max(silhouette_scores) == -1:
        print(f"Warning ({dataset_label}): Could not determine optimal k using Silhouette Score. All scores were invalid. Using default n_clusters: {default_n_clusters}.")
        optimal_k = default_n_clusters
    else:
        optimal_k_index = np.argmax(silhouette_scores)
        optimal_k = valid_k_values[optimal_k_index]
        print(f"Optimal n_clusters for {dataset_label} based on Silhouette Score: {optimal_k} (Score: {silhouette_scores[optimal_k_index]:.4f})")

    # Plot Silhouette Scores vs. k
    fig_path = f"figs/silhouette_scores_{dataset_label.lower().replace(' ', '_')}.png"
    _ensure_dir_exists(fig_path)
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        ax.plot(valid_k_values, silhouette_scores, marker='o')
        ax.set_title(f'Silhouette Scores for Optimal k ({dataset_label})')
        ax.set_xlabel('Number of clusters (k)')
        ax.set_ylabel('Average Silhouette Score')
        if optimal_k != default_n_clusters and max(silhouette_scores) != -1 :
             ax.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
             ax.legend()
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved Silhouette scores plot for {dataset_label}: {fig_path}")

    return optimal_k


# --- Main Function ---
def main():
    np.random.seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser(description="UMAP and KMeans clustering with optimal k selection")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Suffix for data files (e.g., 'normal', 'more', 'much', 'test')",
    )
    args = parser.parse_args()

    n_cols = 100
    DEFAULT_N_CLUSTERS = 100 # Original default, used as fallback
    K_RANGE_FOR_OPTIMAL_SEARCH = (2, 21) # Test k from 2 to 20 clusters. Adjust as needed.

    if args.data == "test":
        print(f"--- Processing Test Data (Scenario: {args.data}) ---")
        try:
            df_test_original = pd.read_parquet(f"data_{args.data}/test_cand.parquet")
        except Exception as e:
            print(f"Error loading test data (data_{args.data}/test_cand.parquet): {e}")
            return

        if df_test_original.empty:
            print("Test data is empty. Exiting.")
            return

        df_active_reshaped = reshape_df(df_test_original, n_cols)
        if df_active_reshaped.empty:
            print("Reshaped test data is empty. Exiting.")
            return

        umap_active, _ = umap_map_and_embed(
            df_active_reshaped, n_neighbors=10, min_dist=0.001, n_components=2, random_state=42
        )
        
        n_clusters_optimal_active = DEFAULT_N_CLUSTERS
        if not umap_active.empty and len(umap_active) >= K_RANGE_FOR_OPTIMAL_SEARCH[0]:
             n_clusters_optimal_active = find_optimal_n_clusters_silhouette(
                umap_active, K_RANGE_FOR_OPTIMAL_SEARCH, DEFAULT_N_CLUSTERS, "Test Data"
            )
        else:
            print("Skipping optimal k search for test data due to insufficient UMAP data.")


        cluster_active = kmeans_clustering(umap_active, n_clusters=n_clusters_optimal_active, random_state=42)

        if not umap_active.empty and not cluster_active.empty:
            plot_umap_data(
                umap_coords_df=umap_active[["UMAP1", "UMAP2"]],
                cluster_labels_series=cluster_active["Cluster"],
                title=f"UMAP projection of test data ({args.data}, k={n_clusters_optimal_active})",
                filename=f"figs/umap_projection_{args.data}.png",
            )
        else:
            print("Skipping UMAP plot for full test data: UMAP or cluster data is empty.")

        num_reshaped_items = len(df_active_reshaped)
        n_target_active = num_reshaped_items // 10 if num_reshaped_items > 0 else 0
        
        df_active_with_clusters = df_active_reshaped.merge(cluster_active, on="index", how="left")
        sampled_active, sampled_clusters_active_df = sampling_by_cluster(
            df_active_with_clusters, cluster_active, n_target_active, n_clusters_sampling=n_clusters_optimal_active
        )

        if not sampled_active.empty and "Cluster" in sampled_active.columns: # Check 'Cluster' in sampled_active
            # Re-embed and plot sampled data if needed, or use existing cluster labels
            # For simplicity, we use cluster labels from the full data for coloring if available
            # UMAP on sampled data might yield different clusters.
            umap_sampled_active, _ = umap_map_and_embed(
                 sampled_active.drop(columns=['Cluster'] if 'Cluster' in sampled_active.columns else []), # UMAP on features
                 n_neighbors=10, min_dist=0.001, n_components=2, random_state=42
            )
            if not umap_sampled_active.empty and not sampled_clusters_active_df.empty:
                 # Get cluster labels corresponding to the sampled_active_df
                 sampled_indices = sampled_active['index'].tolist()
                 cluster_labels_for_sampled_plot = cluster_active[cluster_active['index'].isin(sampled_indices)]['Cluster']

                 if len(cluster_labels_for_sampled_plot) == len(umap_sampled_active):
                    plot_umap_data(
                        umap_coords_df=umap_sampled_active[["UMAP1", "UMAP2"]],
                        cluster_labels_series=cluster_labels_for_sampled_plot, # Use original clusters for color consistency
                        title=f"UMAP of sampled test data ({args.data}, k={n_clusters_optimal_active})",
                        filename=f"figs/umap_projection_sampled_{args.data}.png",
                    )
                 else:
                    print("Mismatch in length between sampled UMAP data and cluster labels. Skipping sampled plot.")
            else:
                print("UMAP on sampled test data resulted in empty data. Skipping plot.")
        else:
            print("Skipping UMAP plot for sampled test data: empty or lacks cluster information.")

        save_sampled_data_from_original(
            original_long_df=df_test_original,
            sampled_reshaped_df=sampled_active,
            n_cols_reshape=n_cols,
            output_path=f"data_{args.data}/test.parquet",
            dataset_label_for_log="test",
        )

    else: # For "normal", "more", "much"
        print(f"--- Processing Training & Validation Data (Scenario: {args.data}) ---")
        try:
            df_train_original = pd.read_parquet(f"data_{args.data}/train_cand.parquet")
            df_val_original = pd.read_parquet(f"data_{args.data}/val_cand.parquet")
        except Exception as e:
            print(f"Error loading train/val data for scenario {args.data}: {e}")
            return

        # --- Training Data Processing ---
        print("--- Training Data ---")
        df_train = reshape_df(df_train_original, n_cols)
        if df_train.empty:
            print("Reshaped training data is empty. Cannot proceed with training path.")
            return # Or handle appropriately

        umap_train, _ = umap_map_and_embed(
            df_train, n_neighbors=10, min_dist=0.001, n_components=2, random_state=42
        )
        
        n_clusters_optimal_train = DEFAULT_N_CLUSTERS
        if not umap_train.empty and len(umap_train) >= K_RANGE_FOR_OPTIMAL_SEARCH[0]:
            n_clusters_optimal_train = find_optimal_n_clusters_silhouette(
                umap_train, K_RANGE_FOR_OPTIMAL_SEARCH, DEFAULT_N_CLUSTERS, "Training Data"
            )
        else:
            print("Skipping optimal k search for training data due to insufficient UMAP data.")


        cluster_train = kmeans_clustering(umap_train, n_clusters=n_clusters_optimal_train, random_state=42)

        if not umap_train.empty and not cluster_train.empty:
            plot_umap_data(
                umap_coords_df=umap_train[["UMAP1", "UMAP2"]],
                cluster_labels_series=cluster_train["Cluster"],
                title=f"UMAP projection of training data ({args.data}, k={n_clusters_optimal_train})",
                filename=f"figs/umap_projection_{args.data}_train_full.png",
            )
        else:
            print("Skipping UMAP plot for full training data: UMAP or cluster data is empty.")

        # --- Validation Data Processing ---
        print("--- Validation Data ---")
        df_val = reshape_df(df_val_original, n_cols)
        if df_val.empty:
            print("Reshaped validation data is empty. Cannot proceed with validation path.")
            # Decide if to exit or skip validation specific steps
        else:
            umap_val, _ = umap_map_and_embed(
                df_val, n_neighbors=10, min_dist=0.001, n_components=2, random_state=42
            )

            n_clusters_optimal_val = DEFAULT_N_CLUSTERS
            if not umap_val.empty and len(umap_val) >= K_RANGE_FOR_OPTIMAL_SEARCH[0]:
                n_clusters_optimal_val = find_optimal_n_clusters_silhouette(
                    umap_val, K_RANGE_FOR_OPTIMAL_SEARCH, DEFAULT_N_CLUSTERS, "Validation Data"
                )
            else:
                print("Skipping optimal k search for validation data due to insufficient UMAP data.")
            
            cluster_val = kmeans_clustering(umap_val, n_clusters=n_clusters_optimal_val, random_state=42)
            # Optional: Plot full validation UMAP if desired
            # plot_umap_data(umap_val[["UMAP1", "UMAP2"]], cluster_val["Cluster"], title=f"UMAP of validation data ({args.data}, k={n_clusters_optimal_val})", filename=f"figs/umap_projection_{args.data}_val_full.png")


        # --- Define Sampling Target Sizes ---
        if args.data == "normal":
            n_target_train, n_target_val = 8000, 2000
        elif args.data == "more":
            n_target_train, n_target_val = 80000, 20000
        elif args.data == "much":
            n_target_train, n_target_val = 800000, 200000
        else:
            print(f"Warning: Unknown data size '{args.data}'. Using default small sample sizes.")
            n_target_train, n_target_val = 1000, 200

        # --- Sample Training Data ---
        print("--- Sampling Training Data ---")
        df_train_with_clusters = df_train.merge(cluster_train, on="index", how="left")
        sampled_train, sampled_clusters_train_df = sampling_by_cluster(
            df_train_with_clusters, cluster_train, n_target_train, n_clusters_sampling=n_clusters_optimal_train
        )
        
        if not sampled_train.empty and "Cluster" in sampled_train.columns:
            umap_sampled_train, _ = umap_map_and_embed(
                 sampled_train.drop(columns=['Cluster'] if 'Cluster' in sampled_train.columns else []),
                 n_neighbors=10, min_dist=0.001, n_components=2, random_state=42
            )
            if not umap_sampled_train.empty:
                 sampled_train_indices = sampled_train['index'].tolist()
                 cluster_labels_for_sampled_train_plot = cluster_train[cluster_train['index'].isin(sampled_train_indices)]['Cluster']
                 if len(cluster_labels_for_sampled_train_plot) == len(umap_sampled_train):
                    plot_umap_data(
                        umap_coords_df=umap_sampled_train[["UMAP1", "UMAP2"]],
                        cluster_labels_series=cluster_labels_for_sampled_train_plot,
                        title=f"UMAP of sampled training data ({args.data}, k={n_clusters_optimal_train})",
                        filename=f"figs/umap_projection_sampled_{args.data}_train.png",
                    )
                 else:
                     print("Mismatch in length for sampled training UMAP plot. Skipping.")
            else:
                print("UMAP on sampled training data resulted in empty data. Skipping plot.")
        else:
            print("Skipping UMAP plot for sampled training data: empty or lacks 'Cluster' column.")

        # --- Sample Validation Data ---
        if not df_val.empty and not cluster_val.empty : # Ensure df_val and cluster_val were processed
            print("--- Sampling Validation Data ---")
            df_val_with_clusters = df_val.merge(cluster_val, on="index", how="left")
            sampled_val, _ = sampling_by_cluster(
                df_val_with_clusters, cluster_val, n_target_val, n_clusters_sampling=n_clusters_optimal_val
            )
            print(f"Sampled Validation DataFrame shape: {sampled_val.shape if not sampled_val.empty else 'Empty'}")
        else:
            print("Skipping validation data sampling as prior steps resulted in empty data.")
            # Create an empty sampled_val to prevent errors in save_sampled_data_from_original
            sampled_val = pd.DataFrame() 


        # --- Save Sampled Data ---
        save_sampled_data_from_original(
            original_long_df=df_train_original, sampled_reshaped_df=sampled_train,
            n_cols_reshape=n_cols, output_path=f"data_{args.data}/train.parquet",
            dataset_label_for_log="training",
        )
        if not df_val_original.empty: # Only save if original validation data existed
            save_sampled_data_from_original(
                original_long_df=df_val_original, sampled_reshaped_df=sampled_val,
                n_cols_reshape=n_cols, output_path=f"data_{args.data}/val.parquet",
                dataset_label_for_log="validation",
            )
        else:
            print("Original validation data was empty, skipping save for validation set.")


if __name__ == "__main__":
    main()
