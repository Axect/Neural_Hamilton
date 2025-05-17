import umap
import matplotlib.pyplot as plt
import scienceplots
import fireducks.pandas as pd # Assuming fireducks.pandas is a pandas-compatible library
import numpy as np
import random
from sklearn.cluster import KMeans
import argparse
import warnings
import os

warnings.filterwarnings("ignore")


# --- Helper Functions ---
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
        # Fit an empty mapper instance if data is empty
        mapper = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state,
        ).fit(np.empty((0, df_no_id.shape[1] if df_no_id.shape[1] > 0 else 1))) # Fit on appropriate shape
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
    
    # Ensure UMAP1 and UMAP2 columns exist
    if not ({'UMAP1', 'UMAP2'}.issubset(df.columns)):
        print("Warning: UMAP1 or UMAP2 not found in DataFrame for KMeans. Returning empty cluster DataFrame.")
        return pd.DataFrame( {"Cluster": pd.Series(dtype=np.int32), "index": pd.Series(dtype=index_series.dtype)})

    df_for_kmeans = df[["UMAP1", "UMAP2"]]
    data_for_kmeans = df_for_kmeans.to_numpy()

    if len(data_for_kmeans) == 0:
        print("Warning: Data for KMeans is empty. Returning empty cluster DataFrame.")
        return pd.DataFrame(
            {"Cluster": pd.Series(dtype=np.int32), "index": pd.Series(dtype=index_series.dtype)},
        )

    # n_clusters should be > 0 and <= n_samples
    actual_n_clusters = max(1, min(n_clusters, data_for_kmeans.shape[0])) # Ensure at least 1 cluster
    
    if actual_n_clusters < n_clusters:
        print(
            f"Warning: Requested n_clusters ({n_clusters}) is too high for the number of samples ({data_for_kmeans.shape[0]}). "
            f"Using n_clusters = {actual_n_clusters}."
        )
    if actual_n_clusters == 0 : # Should be caught by max(1, ...)
         print(f"Warning: actual_n_clusters is 0. This should not happen. Defaulting to 1.")
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
    n_clusters_sampling: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_clusters_sampling <= 0:
        print(f"Warning: n_clusters_sampling is {n_clusters_sampling}. Sampling will be random from all available if n_target > 0.")
        if n_target > 0 and not df_with_features_and_clusters.empty:
            # Ensure n_target is not greater than available samples
            actual_n_target = min(n_target, len(df_with_features_and_clusters))
            if actual_n_target <= 0:
                return df_with_features_and_clusters.iloc[0:0], cluster_info_df.iloc[0:0]

            sampled_df = df_with_features_and_clusters.sample(n=actual_n_target, random_state=42)
            # Ensure cluster_info_df has 'index' and sampled_df has 'index'
            if 'index' in sampled_df.columns and 'index' in cluster_info_df.columns:
                 sampled_indices = sampled_df['index']
                 corresponding_cluster_info = cluster_info_df[cluster_info_df['index'].isin(sampled_indices)]
            else: # Fallback if 'index' is missing unexpectedly
                 corresponding_cluster_info = cluster_info_df.iloc[0:0]
            return sampled_df, corresponding_cluster_info
        else:
            return df_with_features_and_clusters.iloc[0:0], cluster_info_df.iloc[0:0]

    n_each = n_target // n_clusters_sampling if n_clusters_sampling > 0 else 0 # Ensure n_each is well-defined
    sampled_dfs_list = []
    sampled_clusters_info_list = []
    all_sampled_indices_flat = []

    # Ensure cluster_info_df is not empty and has 'Cluster' and 'index' columns
    if cluster_info_df.empty or not ({'Cluster', 'index'}.issubset(cluster_info_df.columns)):
        print("Warning: cluster_info_df is empty or missing required columns for sampling by cluster.")
        # Fallback to random sampling if cluster info is problematic
        if n_target > 0 and not df_with_features_and_clusters.empty:
            actual_n_target = min(n_target, len(df_with_features_and_clusters))
            return df_with_features_and_clusters.sample(n=actual_n_target, random_state=42), cluster_info_df.iloc[0:0] # Return empty cluster info part
        return df_with_features_and_clusters.iloc[0:0], cluster_info_df.iloc[0:0]


    for i in range(n_clusters_sampling):
        cluster_member_indices = (
            cluster_info_df[cluster_info_df["Cluster"] == i]["index"].to_numpy()
        )

        if len(cluster_member_indices) == 0:
            continue

        sample_size_ideal = min(n_each, len(cluster_member_indices)) if n_each > 0 else min(1, len(cluster_member_indices)) # if n_each is 0, try to take at least 1 if possible

        if sample_size_ideal <= 0:
            if n_target > 0 and len(cluster_member_indices) > 0 and n_each <= 0: # Try to pick at least one if target > 0
                sample_size_ideal = min(1, len(cluster_member_indices)) 
            else:
                continue
        
        if sample_size_ideal > 0 : # Ensure np.random.choice gets size > 0
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
        all_clustered_indices = cluster_info_df["index"].unique() # Use unique to avoid issues with duplicated indices if any
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
        pd.concat(sampled_dfs_list, ignore_index=True)
        if sampled_dfs_list
        else df_with_features_and_clusters.iloc[0:0]
    )
    final_sampled_clusters_info = (
        pd.concat(sampled_clusters_info_list, ignore_index=True)
        if sampled_clusters_info_list
        else cluster_info_df.iloc[0:0]
    )

    if len(final_sampled_df) > n_target and n_target > 0:
        final_sampled_df = final_sampled_df.sample(
            n=n_target, random_state=42
        ).reset_index(drop=True)
        if 'index' in final_sampled_df.columns:
            final_indices = final_sampled_df["index"].tolist()
            final_sampled_clusters_info = final_sampled_clusters_info[
                final_sampled_clusters_info["index"].isin(final_indices)
            ].reset_index(drop=True)
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
        if not umap_coords_df.empty and "UMAP1" in umap_coords_df and "UMAP2" in umap_coords_df:
            if not cluster_labels_series.empty and len(cluster_labels_series) == len(umap_coords_df):
                color_data = cluster_labels_series.to_numpy().astype(int)
                scatter = ax.scatter(
                    umap_coords_df["UMAP1"].to_numpy(),
                    umap_coords_df["UMAP2"].to_numpy(),
                    c=color_data,
                    cmap="Spectral",
                    s=1,
                    linewidths=0,
                )
                if len(np.unique(color_data)) > 1:
                    try: # Add colorbar, handle potential errors if scatter is empty due to prior checks
                        plt.colorbar(scatter, ax=ax, label='Cluster ID')
                    except Exception as e:
                        print(f"Could not add colorbar for plot {title}: {e}")

            else: # Plot without colors if cluster labels are not suitable
                ax.scatter(
                    umap_coords_df["UMAP1"].to_numpy(),
                    umap_coords_df["UMAP2"].to_numpy(),
                    s=1,
                    linewidths=0,
                )
                if cluster_labels_series.empty:
                     print(f"Warning: Cluster labels are empty for UMAP plot '{title}'. Plotting without colors.")
                else:
                     print(f"Warning: Mismatch in length between UMAP data and cluster labels for '{title}'. Plotting without colors.")
        else:
            print(f"Warning: UMAP coordinate data is empty or invalid for plotting '{title}'. Skipping scatter plot.")

        ax.set_title(title)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        try:
            fig.savefig(filename, dpi=600, bbox_inches="tight")
            print(f"Saved UMAP plot: {filename}")
        except Exception as e:
            print(f"Error saving UMAP plot {filename}: {e}")
        plt.close(fig)


def save_sampled_data_from_original(
    original_long_df: pd.DataFrame,
    sampled_reshaped_df: pd.DataFrame,
    n_cols_reshape: int,
    output_path: str,
    dataset_label_for_log: str,
):
    if sampled_reshaped_df.empty or 'index' not in sampled_reshaped_df.columns: # Check if empty or missing 'index'
        print(
            f"Sampled reshaped DataFrame for {dataset_label_for_log} is empty or missing 'index' column. No data saved to {output_path}."
        )
        return

    _ensure_dir_exists(output_path)
    sampled_indices = sampled_reshaped_df["index"].unique().tolist() # Use unique indices
    list_of_sampled_original_data = []

    if sampled_indices:
        for reshaped_idx in sorted(sampled_indices):
            start_original_row = reshaped_idx * n_cols_reshape
            # Ensure segment does not go out of bounds of original_long_df
            end_original_row = min(start_original_row + n_cols_reshape, len(original_long_df))
            segment = original_long_df.iloc[start_original_row:end_original_row]
            if not segment.empty:
                list_of_sampled_original_data.append(segment)

    if list_of_sampled_original_data:
        df_final_sampled_original = pd.concat(
            list_of_sampled_original_data, ignore_index=True
        )
        df_final_sampled_original.to_parquet(output_path)
        print(f"Saved final sampled {dataset_label_for_log} data to {output_path}")
    else:
        print(
            f"No data was sampled for {dataset_label_for_log} or indices were problematic. File {output_path} not written."
        )

# Removed find_optimal_n_clusters_silhouette function

# --- Main Function ---
def main():
    np.random.seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser(description="UMAP and KMeans clustering with heuristic k selection")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Suffix for data files (e.g., 'normal', 'more', 'much', 'test')",
    )
    args = parser.parse_args()

    n_cols = 100
    TARGET_SAMPLES_PER_CLUSTER = 100 # Desired number of samples per cluster

    # Initialize cluster numbers, these will be updated based on n_target
    n_clusters_active = 2 
    n_clusters_train = 2
    n_clusters_val = 2

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
        
        num_reshaped_items_active = len(df_active_reshaped)
        n_target_active = num_reshaped_items_active // 10 if num_reshaped_items_active > 0 else 0
        
        if n_target_active > 0 and TARGET_SAMPLES_PER_CLUSTER > 0 :
            n_clusters_active = max(2, n_target_active // TARGET_SAMPLES_PER_CLUSTER)
        else:
            n_clusters_active = 2 # Default to 2 if n_target is 0 or not enough to form clusters based on target samples
        
        print(f"Using n_clusters = {n_clusters_active} for Test Data based on n_target={n_target_active}")

        cluster_active = kmeans_clustering(umap_active, n_clusters=n_clusters_active, random_state=42)

        if not umap_active.empty and not cluster_active.empty:
            plot_umap_data(
                umap_coords_df=umap_active[["UMAP1", "UMAP2"]],
                cluster_labels_series=cluster_active["Cluster"],
                title=f"UMAP projection of test data ({args.data}, k={n_clusters_active})",
                filename=f"figs/umap_projection_{args.data}.png",
            )
        else:
            print("Skipping UMAP plot for full test data: UMAP or cluster data is empty.")
        
        df_active_with_clusters = df_active_reshaped.merge(cluster_active, on="index", how="left")
        sampled_active, sampled_clusters_active_df = sampling_by_cluster(
            df_active_with_clusters, cluster_active, n_target_active, n_clusters_sampling=n_clusters_active
        )

        if not sampled_active.empty and "Cluster" in sampled_active.columns:
            umap_sampled_active, _ = umap_map_and_embed(
                 sampled_active.drop(columns=['Cluster'] if 'Cluster' in sampled_active.columns else []),
                 n_neighbors=10, min_dist=0.001, n_components=2, random_state=42
            )
            if not umap_sampled_active.empty: # Check if umap_sampled_active is not empty
                # Get cluster labels corresponding to the sampled_active_df for consistent coloring
                # These cluster labels come from the clustering of the *full* umap_active data
                sampled_indices_for_plot = umap_sampled_active['index'].tolist() # Use indices from umap_sampled_active
                cluster_labels_for_sampled_plot = pd.Series() # Default to empty
                if 'index' in cluster_active.columns: # Ensure cluster_active has 'index'
                    # Filter cluster_active for the indices present in umap_sampled_active
                    relevant_clusters = cluster_active[cluster_active['index'].isin(sampled_indices_for_plot)]
                    # Align cluster labels with umap_sampled_active by reindexing (or careful merge)
                    # This ensures the labels match the order of umap_sampled_active if UMAP reorders
                    if not relevant_clusters.empty:
                        # A merge is safer if UMAP output order is not guaranteed to match input 'index' order
                        temp_df_for_labels = umap_sampled_active[['index']].merge(relevant_clusters[['index', 'Cluster']], on='index', how='left')
                        cluster_labels_for_sampled_plot = temp_df_for_labels['Cluster']


                if len(cluster_labels_for_sampled_plot) == len(umap_sampled_active) and not cluster_labels_for_sampled_plot.isna().any():
                    plot_umap_data(
                        umap_coords_df=umap_sampled_active[["UMAP1", "UMAP2"]],
                        cluster_labels_series=cluster_labels_for_sampled_plot,
                        title=f"UMAP of sampled test data ({args.data}, k={n_clusters_active})",
                        filename=f"figs/umap_projection_sampled_{args.data}.png",
                    )
                else:
                    print(f"Warning: Mismatch or missing cluster labels for sampled test UMAP plot (Labels: {len(cluster_labels_for_sampled_plot)}, Data: {len(umap_sampled_active)}). Plotting without cluster colors or skipping.")
                    # Optionally plot without colors if labels are problematic
                    plot_umap_data(
                        umap_coords_df=umap_sampled_active[["UMAP1", "UMAP2"]],
                        cluster_labels_series=pd.Series(), # Empty series for no colors
                        title=f"UMAP of sampled test data (no clusters) ({args.data}, k={n_clusters_active})",
                        filename=f"figs/umap_projection_sampled_{args.data}_no_cluster_color.png",
                    )

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
        
        # Calculate n_clusters based on n_target and TARGET_SAMPLES_PER_CLUSTER
        if n_target_train > 0 and TARGET_SAMPLES_PER_CLUSTER > 0:
            n_clusters_train = max(2, n_target_train // TARGET_SAMPLES_PER_CLUSTER)
        else:
            n_clusters_train = 2
        
        if n_target_val > 0 and TARGET_SAMPLES_PER_CLUSTER > 0:
            n_clusters_val = max(2, n_target_val // TARGET_SAMPLES_PER_CLUSTER)
        else:
            n_clusters_val = 2

        print(f"Using n_clusters = {n_clusters_train} for Training Data (n_target={n_target_train})")
        print(f"Using n_clusters = {n_clusters_val} for Validation Data (n_target={n_target_val})")

        # --- Training Data Processing ---
        print("--- Training Data ---")
        df_train = reshape_df(df_train_original, n_cols)
        if df_train.empty:
            print("Reshaped training data is empty. Cannot proceed with training path.")
            # Fallback for sampled_train and cluster_train to avoid errors later
            sampled_train = pd.DataFrame()
            cluster_train = pd.DataFrame()
        else:
            umap_train, _ = umap_map_and_embed(
                df_train, n_neighbors=10, min_dist=0.001, n_components=2, random_state=42
            )
            cluster_train = kmeans_clustering(umap_train, n_clusters=n_clusters_train, random_state=42)

            if not umap_train.empty and not cluster_train.empty:
                plot_umap_data(
                    umap_coords_df=umap_train[["UMAP1", "UMAP2"]],
                    cluster_labels_series=cluster_train["Cluster"],
                    title=f"UMAP projection of training data ({args.data}, k={n_clusters_train})",
                    filename=f"figs/umap_projection_{args.data}_train_full.png",
                )
            else:
                print("Skipping UMAP plot for full training data: UMAP or cluster data is empty.")
            
            # Sample Training Data
            print("--- Sampling Training Data ---")
            df_train_with_clusters = df_train.merge(cluster_train, on="index", how="left" if not cluster_train.empty else "left") # ensure merge works if cluster_train is empty
            if 'Cluster' not in df_train_with_clusters.columns and not cluster_train.empty : # If merge failed to add Cluster but cluster_train was not empty
                 print("Warning: Cluster column not found after merging for training data. Sampling might be affected.")
                 # Add a dummy cluster column if it's missing to prevent error in sampling_by_cluster, though this is not ideal.
                 # df_train_with_clusters['Cluster'] = 0 

            sampled_train, _ = sampling_by_cluster( # sampled_clusters_train_df not used explicitly later
                df_train_with_clusters, cluster_train, n_target_train, n_clusters_sampling=n_clusters_train
            )
            
            if not sampled_train.empty and "Cluster" in sampled_train.columns: # Check if 'Cluster' from merge is in sampled_train
                umap_sampled_train, _ = umap_map_and_embed(
                    sampled_train.drop(columns=['Cluster'] if 'Cluster' in sampled_train.columns else []),
                    n_neighbors=10, min_dist=0.001, n_components=2, random_state=42
                )
                if not umap_sampled_train.empty:
                    # Align cluster labels for the sampled plot
                    sampled_train_indices_for_plot = umap_sampled_train['index'].tolist()
                    cluster_labels_for_sampled_train_plot = pd.Series()
                    if 'index' in cluster_train.columns:
                        relevant_clusters_train = cluster_train[cluster_train['index'].isin(sampled_train_indices_for_plot)]
                        if not relevant_clusters_train.empty:
                             temp_df_train_labels = umap_sampled_train[['index']].merge(relevant_clusters_train[['index', 'Cluster']], on='index', how='left')
                             cluster_labels_for_sampled_train_plot = temp_df_train_labels['Cluster']


                    if len(cluster_labels_for_sampled_train_plot) == len(umap_sampled_train) and not cluster_labels_for_sampled_train_plot.isna().any():
                        plot_umap_data(
                            umap_coords_df=umap_sampled_train[["UMAP1", "UMAP2"]],
                            cluster_labels_series=cluster_labels_for_sampled_train_plot,
                            title=f"UMAP of sampled training data ({args.data}, k={n_clusters_train})",
                            filename=f"figs/umap_projection_sampled_{args.data}_train.png",
                        )
                    else:
                        print(f"Warning: Mismatch or missing cluster labels for sampled training UMAP plot. Plotting without colors or skipping.")
                        plot_umap_data(
                            umap_coords_df=umap_sampled_train[["UMAP1", "UMAP2"]],
                            cluster_labels_series=pd.Series(),
                            title=f"UMAP of sampled training data (no clusters) ({args.data}, k={n_clusters_train})",
                            filename=f"figs/umap_projection_sampled_{args.data}_train_no_cluster_color.png",
                        )

                else:
                    print("UMAP on sampled training data resulted in empty data. Skipping plot.")
            else:
                print("Skipping UMAP plot for sampled training data: sampled_train is empty or lacks 'Cluster' column (after merge).")


        # --- Validation Data Processing ---
        print("--- Validation Data ---")
        df_val = reshape_df(df_val_original, n_cols)
        # Initialize sampled_val to empty DataFrame for graceful failure in save if df_val is empty
        sampled_val = pd.DataFrame()
        cluster_val = pd.DataFrame() # Initialize to prevent NameError

        if df_val.empty:
            print("Reshaped validation data is empty. Skipping validation processing and sampling.")
        else:
            umap_val, _ = umap_map_and_embed(
                df_val, n_neighbors=10, min_dist=0.001, n_components=2, random_state=42
            )
            cluster_val = kmeans_clustering(umap_val, n_clusters=n_clusters_val, random_state=42)
            
            # Optional: Plot full validation UMAP if desired
            if not umap_val.empty and not cluster_val.empty:
                 plot_umap_data(
                     umap_coords_df=umap_val[["UMAP1", "UMAP2"]], 
                     cluster_labels_series=cluster_val["Cluster"], 
                     title=f"UMAP of validation data ({args.data}, k={n_clusters_val})", 
                     filename=f"figs/umap_projection_{args.data}_val_full.png"
                )
            else:
                 print("Skipping UMAP plot for full validation data: UMAP or cluster data is empty.")


            # Sample Validation Data
            print("--- Sampling Validation Data ---")
            df_val_with_clusters = df_val.merge(cluster_val, on="index", how="left" if not cluster_val.empty else "left")
            if 'Cluster' not in df_val_with_clusters.columns and not cluster_val.empty:
                 print("Warning: Cluster column not found after merging for validation data.")
                 # df_val_with_clusters['Cluster'] = 0

            sampled_val, _ = sampling_by_cluster(
                df_val_with_clusters, cluster_val, n_target_val, n_clusters_sampling=n_clusters_val
            )
            print(f"Sampled Validation DataFrame shape: {sampled_val.shape if not sampled_val.empty else 'Empty'}")


        # --- Save Sampled Data ---
        if not df_train_original.empty: # Only save if original train data existed
            save_sampled_data_from_original(
                original_long_df=df_train_original, sampled_reshaped_df=sampled_train,
                n_cols_reshape=n_cols, output_path=f"data_{args.data}/train.parquet",
                dataset_label_for_log="training",
            )
        else:
            print("Original training data was empty, skipping save for training set.")

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
