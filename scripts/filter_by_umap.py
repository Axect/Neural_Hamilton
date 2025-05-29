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


# --- Helper Functions ---
def reshape_df(df: pd.DataFrame, n_cols: int, column: str = "V") -> pd.DataFrame:
    # Reshapes a DataFrame from a long format to a wide format.
    # Each row in the output DataFrame will correspond to 'n_cols' consecutive values
    # from the input 'column'. An 'index' column is added to track original groupings.
    V = df[column].to_numpy()
    if V.size == 0:
        print(
            f"Warning: Column '{column}' is empty. Reshaping will result in an empty DataFrame."
        )
        if n_cols > 0:
            reshaped_v_empty = np.array([]).reshape(0, n_cols)
            column_names_empty = [f"V{i}" for i in range(n_cols)]
            empty_df = pd.DataFrame(reshaped_v_empty, columns=column_names_empty)
            empty_df['index'] = pd.Series(dtype=np.int64) # Match original 'index' type if possible
            return empty_df
        else: # n_cols is 0 or less
            return pd.DataFrame({"index": pd.Series(dtype=np.int64)})


    if V.size > 0 and n_cols > 0 and V.size % n_cols != 0:
        print(
            f"Warning: The size of the data ({V.size}) is not a multiple of n_cols ({n_cols}). Data will be truncated."
        )
        V = V[: -(V.size % n_cols)] # Truncate to make it a multiple

    if V.size == 0 and n_cols > 0: # Empty input but valid n_cols
        reshaped_v = np.array([]).reshape(0, n_cols)
    elif V.size > 0 and n_cols > 0: # Valid input and n_cols
        reshaped_v = V.reshape(-1, n_cols)
    elif n_cols == 0 and V.size > 0: # n_cols is zero with data present
        raise ValueError(
            "n_cols cannot be zero if data is not empty and reshape is attempted."
        )
    else: # V.size is 0 and n_cols is 0
        reshaped_v = np.array([]).reshape(0, 0)


    column_names = [f"V{i}" for i in range(n_cols)]
    reshaped_df = pd.DataFrame(reshaped_v, columns=column_names)
    reshaped_df['index'] = np.arange(len(reshaped_df)) # Assign a new index for the reshaped data
    return reshaped_df


def umap_map_and_embed(
    df: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, umap.UMAP]:
    # Performs UMAP dimensionality reduction.
    # Expects 'df' to have an 'index' column for tracking, and other columns as features.
    # If 'Cluster' column is present, it's preserved.
    index_series = df['index'].copy()
    # df_no_id should not contain 'index' or 'Cluster' before UMAP
    df_no_id = df.drop(columns=['index'], errors='ignore')

    cluster_series = None
    if "Cluster" in df.columns: # If 'Cluster' was part of the input df (e.g. for sampled data)
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
        # Ensure the number of columns for fitting is correct, even if 0 rows
        num_features = df_no_id.shape[1] if df_no_id.shape[1] > 0 else 1
        mapper = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state,
        ).fit(np.empty((0, num_features)))
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
    umap_df["index"] = index_series.values # Restore original index
    if cluster_series is not None: # If input df had clusters, add them back
        # This is important if we are UMAPing an already sampled and clustered DF
        if len(cluster_series) == len(umap_df):
            umap_df["Cluster"] = cluster_series.values
        else:
            print(f"Warning: Length mismatch between UMAP data ({len(umap_df)}) and original cluster_series ({len(cluster_series)}). Clusters not added to UMAP output.")
    return umap_df, mapper


def kmeans_clustering(
    df: pd.DataFrame, n_clusters: int = 10, random_state: int = 42
) -> pd.DataFrame:
    # Performs KMeans clustering on UMAP coordinates.
    # Expects 'df' to have 'UMAP1', 'UMAP2', and 'index' columns.
    index_series = df['index'].copy()

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
    actual_n_clusters = max(1, min(n_clusters, data_for_kmeans.shape[0]))

    if actual_n_clusters < n_clusters:
        print(
            f"Warning: Requested n_clusters ({n_clusters}) is too high for the number of samples ({data_for_kmeans.shape[0]}). "
            f"Using n_clusters = {actual_n_clusters}."
        )
    if actual_n_clusters == 0 :
         print(f"Warning: actual_n_clusters is 0. Defaulting to 1.")
         actual_n_clusters = 1


    kmeans = KMeans(
        n_clusters=actual_n_clusters, random_state=random_state, n_init="auto"
    )
    labels = kmeans.fit_predict(data_for_kmeans)

    cluster_df = pd.DataFrame({"Cluster": labels})
    cluster_df["index"] = index_series.values # Restore original index
    return cluster_df

def sampling_by_cluster_point_count_inverse_weighted(
    df_with_features_and_clusters: pd.DataFrame, # Must contain 'index', 'Cluster'
    n_target: int,
    min_samples_per_cluster: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Samples data from clusters, giving higher weight to clusters with fewer points.
    # Inverse of point counts is used as weight.
    if n_target <= 0:
        return df_with_features_and_clusters.iloc[0:0], pd.DataFrame(columns=['index', 'Cluster'])

    if df_with_features_and_clusters.empty or not {'index', 'Cluster'}.issubset(df_with_features_and_clusters.columns):
        print("Warning: df_with_features_and_clusters is empty or missing 'index' or 'Cluster' columns.")
        if n_target > 0 and not df_with_features_and_clusters.empty:
            actual_n_target = min(n_target, len(df_with_features_and_clusters))
            sampled_df = df_with_features_and_clusters.sample(n=actual_n_target, random_state=42)
            sampled_cluster_info = sampled_df[['index', 'Cluster']] if 'index' in sampled_df and 'Cluster' in sampled_df else pd.DataFrame(columns=['index', 'Cluster'])
            return sampled_df, sampled_cluster_info
        return df_with_features_and_clusters.iloc[0:0], pd.DataFrame(columns=['index', 'Cluster'])

    unique_cluster_ids = sorted(df_with_features_and_clusters['Cluster'].unique())
    num_actual_clusters = len(unique_cluster_ids)

    if num_actual_clusters == 0:
        return df_with_features_and_clusters.iloc[0:0], pd.DataFrame(columns=['index', 'Cluster'])

    cluster_point_counts = df_with_features_and_clusters['Cluster'].value_counts().to_dict()
    
    # Ensure all unique_cluster_ids are in cluster_point_counts (can happen if a cluster has 0 points after some merge/split, though unlikely here)
    for cid in unique_cluster_ids:
        if cid not in cluster_point_counts:
            cluster_point_counts[cid] = 0


    # Calculate weights (inverse of point counts, add epsilon for stability if count is 0)
    epsilon = 1e-3
    alpha = 2.0
    cluster_weights = {
        cid: 1.0 / (cluster_point_counts.get(cid, 0)**alpha + epsilon) for cid in unique_cluster_ids
    }
    total_weight = sum(cluster_weights.values())

    # Initial allocation based on weights
    samples_to_allocate = {}
    if total_weight == 0: # All clusters have effectively zero weight (e.g., all P_i are huge, making 1/P_i tiny)
        # Fallback to (near) uniform allocation based on number of clusters
        base_alloc = n_target // num_actual_clusters
        remainder = n_target % num_actual_clusters
        for i, cluster_id in enumerate(unique_cluster_ids):
            samples_to_allocate[cluster_id] = base_alloc + (1 if i < remainder else 0)
    else:
        for cluster_id in unique_cluster_ids:
            weight_proportion = cluster_weights[cluster_id] / total_weight
            samples_to_allocate[cluster_id] = int(round(weight_proportion * n_target))

    # 1st adjustment: Apply min/max samples per cluster
    for cluster_id in unique_cluster_ids:
        p_i = cluster_point_counts.get(cluster_id, 0)
        # Cannot allocate more than available
        samples_to_allocate[cluster_id] = min(samples_to_allocate[cluster_id], p_i)
        # Ensure minimum samples, but not more than available
        samples_to_allocate[cluster_id] = max(samples_to_allocate[cluster_id], min(min_samples_per_cluster, p_i))
        samples_to_allocate[cluster_id] = max(0, samples_to_allocate[cluster_id]) # Ensure non-negative

    # 2nd adjustment: Match n_target
    current_total_allocated = sum(samples_to_allocate.values())
    diff = n_target - current_total_allocated

    # Sort clusters by weight (descending) for prioritizing additions/removals
    # Clusters with higher weight (fewer points) get priority for additions if diff > 0
    # Clusters with lower weight (more points) get priority for removals if diff < 0
    # This seems counter-intuitive for removals if we want to keep diversity from small clusters.
    # Let's adjust: if diff > 0, add to high-weight clusters. If diff < 0, remove from low-weight clusters.
    
    sorted_clusters_for_adjustment = sorted(unique_cluster_ids, key=lambda cid: cluster_weights[cid], reverse=True)

    if diff > 0: # Need to add more samples
        for cluster_id in sorted_clusters_for_adjustment: # Add to high-weight (small P_i) clusters first
            can_add = cluster_point_counts.get(cluster_id, 0) - samples_to_allocate[cluster_id]
            add_amount = min(diff, can_add)
            samples_to_allocate[cluster_id] += add_amount
            diff -= add_amount
            if diff <= 0:
                break
    elif diff < 0: # Need to remove samples
        diff_abs = abs(diff)
        # For removal, prioritize clusters that are "over-represented" relative to min_samples_per_cluster
        # Or simply, clusters with more points (lower weight)
        sorted_clusters_for_removal = sorted(unique_cluster_ids, key=lambda cid: cluster_weights[cid]) # Ascending weight = descending P_i
        for cluster_id in sorted_clusters_for_removal: # Remove from low-weight (large P_i) clusters first
            p_i = cluster_point_counts.get(cluster_id, 0)
            min_allowed_for_cluster = min(min_samples_per_cluster, p_i)
            can_remove = samples_to_allocate[cluster_id] - min_allowed_for_cluster
            remove_amount = min(diff_abs, can_remove)
            samples_to_allocate[cluster_id] -= remove_amount
            diff_abs -= remove_amount
            if diff_abs <= 0:
                break
    
    # Final check on allocation numbers
    final_samples_per_cluster = {}
    for cluster_id in unique_cluster_ids:
        p_i = cluster_point_counts.get(cluster_id, 0)
        final_samples_per_cluster[cluster_id] = max(0, min(samples_to_allocate[cluster_id], p_i))

    # Perform sampling
    sampled_dfs_list = []
    for cluster_id in unique_cluster_ids:
        num_to_sample = final_samples_per_cluster[cluster_id]
        if num_to_sample > 0:
            cluster_data = df_with_features_and_clusters[df_with_features_and_clusters['Cluster'] == cluster_id]
            # Ensure num_to_sample is not greater than actual data points in cluster
            actual_sample_size = min(num_to_sample, len(cluster_data))
            if actual_sample_size > 0 :
                sampled_dfs_list.append(
                    cluster_data.sample(n=actual_sample_size, random_state=42, replace=False)
                )
    
    final_sampled_df = pd.concat(sampled_dfs_list, ignore_index=True) if sampled_dfs_list else df_with_features_and_clusters.iloc[0:0]

    # Final fallback to meet n_target exactly
    current_len = len(final_sampled_df)
    if current_len > n_target and n_target > 0:
        final_sampled_df = final_sampled_df.sample(n=n_target, random_state=42).reset_index(drop=True)
    elif 0 < current_len < n_target:
        needed_more = n_target - current_len
        if not df_with_features_and_clusters.empty: # Check if original df is not empty
            sampled_indices = final_sampled_df['index'].tolist() if 'index' in final_sampled_df.columns else []
            
            # Ensure 'index' column exists in the source df for filtering
            if 'index' in df_with_features_and_clusters.columns:
                available_pool = df_with_features_and_clusters[~df_with_features_and_clusters['index'].isin(sampled_indices)]
                if not available_pool.empty():
                    num_to_add_from_pool = min(needed_more, len(available_pool))
                    if num_to_add_from_pool > 0:
                        additional_samples = available_pool.sample(n=num_to_add_from_pool, random_state=42)
                        final_sampled_df = pd.concat([final_sampled_df, additional_samples], ignore_index=True)
            else:
                print("Warning: 'index' column missing in df_with_features_and_clusters for fallback sampling.")


    sampled_cluster_info_df = final_sampled_df[['index', 'Cluster']] if 'index' in final_sampled_df and 'Cluster' in final_sampled_df else pd.DataFrame(columns=['index', 'Cluster'])
    return final_sampled_df, sampled_cluster_info_df


# --- Plotting Functions ---
def _ensure_dir_exists(file_path: str):
    # Ensures the directory for the given file_path exists.
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def plot_umap_data(
    umap_coords_df: pd.DataFrame, # Should contain 'UMAP1', 'UMAP2'
    cluster_labels_series: pd.Series, # Series of cluster labels, matching umap_coords_df
    title: str,
    filename: str,
):
    # Plots UMAP 2D projection with optional cluster coloring.
    _ensure_dir_exists(filename)
    with plt.style.context(["science", "nature"]): # Using specified style context
        fig, ax = plt.subplots()
        if not umap_coords_df.empty and "UMAP1" in umap_coords_df and "UMAP2" in umap_coords_df:
            # Ensure cluster_labels_series is not empty and has the same length as umap_coords_df
            if not cluster_labels_series.empty and len(cluster_labels_series) == len(umap_coords_df) and cluster_labels_series.notna().all():
                color_data = cluster_labels_series.to_numpy().astype(int)
                scatter = ax.scatter(
                    umap_coords_df["UMAP1"].to_numpy(),
                    umap_coords_df["UMAP2"].to_numpy(),
                    c=color_data,
                    cmap="Spectral", # A good colormap for categorical data
                    s=5, # Slightly larger points for visibility
                    linewidths=0,
                    alpha=0.7 # Add some transparency
                )
                if len(np.unique(color_data)) > 1: # Add colorbar only if multiple clusters
                    try:
                        plt.colorbar(scatter, ax=ax, label='Cluster ID')
                    except Exception as e:
                        print(f"Could not add colorbar for plot {title}: {e}")
            else: # Plot without colors if cluster labels are not suitable
                ax.scatter(
                    umap_coords_df["UMAP1"].to_numpy(),
                    umap_coords_df["UMAP2"].to_numpy(),
                    s=5,
                    linewidths=0,
                    alpha=0.75,
                    color='gray' # Default color if no clusters
                )
                if cluster_labels_series.empty:
                    print(f"Info: Cluster labels are empty for UMAP plot '{title}'. Plotting without colors.")
                elif len(cluster_labels_series) != len(umap_coords_df):
                    print(f"Warning: Mismatch in length between UMAP data and cluster labels for '{title}'. Plotting without colors.")
                elif cluster_labels_series.isna().any():
                    print(f"Warning: Cluster labels contain NaN values for UMAP plot '{title}'. Plotting without colors.")

        else:
            print(f"Warning: UMAP coordinate data is empty or invalid for plotting '{title}'. Skipping scatter plot.")

        #ax.set_title(title)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.grid(True, linestyle='--', alpha=0.5)
        try:
            fig.savefig(filename, dpi=600, bbox_inches="tight") # Reduced DPI for faster save, 600 is high
            print(f"Saved UMAP plot: {filename}")
        except Exception as e:
            print(f"Error saving UMAP plot {filename}: {e}")
        plt.close(fig)

def plot_cluster_point_counts_histogram(
    cluster_info_df: pd.DataFrame, # DataFrame with 'Cluster' column
    title: str,
    filename: str,
):
    # Plots a histogram of the number of points in each cluster.
    if cluster_info_df.empty or 'Cluster' not in cluster_info_df.columns:
        print(f"Warning: Cluster info is empty or 'Cluster' column missing for histogram '{title}'. Skipping plot.")
        return

    _ensure_dir_exists(filename)
    point_counts = cluster_info_df['Cluster'].value_counts().sort_index()

    if point_counts.empty:
        print(f"Warning: No data points found in clusters for histogram '{title}'. Skipping plot.")
        return

    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        # Create a bar chart for point counts per cluster ID
        ax.bar(point_counts.index, point_counts.values, color='steelblue', alpha=0.8)

        #ax.set_title(title)
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Number of Points")
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        # Set x-ticks to be integer cluster IDs if there are not too many
        if len(point_counts.index) <= 30 : # Heuristic for number of ticks
             ax.set_xticks(point_counts.index)
             ax.tick_params(axis='x', rotation=45)


        try:
            fig.savefig(filename, dpi=600, bbox_inches="tight")
            print(f"Saved cluster point counts histogram: {filename}")
        except Exception as e:
            print(f"Error saving histogram {filename}: {e}")
        plt.close(fig)


def save_sampled_data_from_original(
    original_long_df: pd.DataFrame,
    sampled_reshaped_df: pd.DataFrame, # This df contains 'index' column mapping to original reshaped blocks
    n_cols_reshape: int,
    output_path: str,
    dataset_label_for_log: str,
):
    # Saves the original long-format data segments corresponding to the sampled reshaped data.
    if sampled_reshaped_df.empty or 'index' not in sampled_reshaped_df.columns:
        print(
            f"Sampled reshaped DataFrame for {dataset_label_for_log} is empty or missing 'index' column. No data saved to {output_path}."
        )
        if not sampled_reshaped_df.empty and 'index' not in sampled_reshaped_df.columns:
             print(f"Columns in sampled_reshaped_df: {sampled_reshaped_df.columns}")
        return

    _ensure_dir_exists(output_path)
    # Use unique indices from the 'index' column of the sampled reshaped data
    # These indices refer to the row number in the *reshaped* DataFrame, not the original long one.
    sampled_reshaped_indices = sampled_reshaped_df["index"].unique().tolist()
    list_of_sampled_original_data = []

    if sampled_reshaped_indices:
        for reshaped_idx in sorted(sampled_reshaped_indices):
            # Calculate start and end row in the original_long_df
            start_original_row = reshaped_idx * n_cols_reshape
            end_original_row = start_original_row + n_cols_reshape # Exclusive end
            
            # Ensure segment does not go out of bounds of original_long_df
            actual_end_original_row = min(end_original_row, len(original_long_df))
            
            if start_original_row < actual_end_original_row : # Ensure there's a valid segment
                segment = original_long_df.iloc[start_original_row:actual_end_original_row]
                if not segment.empty:
                    list_of_sampled_original_data.append(segment)
            else:
                print(f"Warning: Invalid segment for reshaped_idx {reshaped_idx} (start: {start_original_row}, end: {actual_end_original_row}). Original df length: {len(original_long_df)}")


    if list_of_sampled_original_data:
        df_final_sampled_original = pd.concat(
            list_of_sampled_original_data, ignore_index=True
        )
        df_final_sampled_original.to_parquet(output_path)
        print(f"Saved final sampled {dataset_label_for_log} data to {output_path} (Total rows: {len(df_final_sampled_original)})")
    else:
        print(
            f"No data was sampled for {dataset_label_for_log} or indices were problematic. File {output_path} not written."
        )

# --- Main Function ---
def main():
    np.random.seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser(description="UMAP and KMeans clustering with inverse point count weighted sampling.")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Suffix for data files (e.g., 'normal', 'more', 'much', 'test')",
    )
    args = parser.parse_args()

    n_cols = 100 # Number of columns for reshaping time-series like data
    TARGET_SAMPLES_PER_CLUSTER_FOR_K = 100 # Desired number of samples per cluster when determining K

    # Initialize cluster numbers, these will be updated based on n_target
    # These are initial values, actual n_clusters will be calculated.
    n_clusters_active = 2
    n_clusters_train = 2
    n_clusters_val = 2
    
    min_samples_per_cluster_in_sampling = 1 # Min samples to draw from any cluster during weighted sampling

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

        # Reshape data
        df_active_reshaped = reshape_df(df_test_original, n_cols)
        if df_active_reshaped.empty:
            print("Reshaped test data is empty. Exiting.")
            return

        # UMAP embedding on full reshaped data
        umap_active, _ = umap_map_and_embed(
            df_active_reshaped.drop(columns=['Cluster'], errors='ignore'), # Ensure no prior 'Cluster' col
            n_neighbors=10, min_dist=0.001, n_components=2, random_state=42
        )
        
        # Determine n_target and n_clusters for test data
        num_reshaped_items_active = len(df_active_reshaped)
        n_target_active = num_reshaped_items_active // 10 if num_reshaped_items_active > 0 else 0 # Sample 10%
        
        if n_target_active > 0 and TARGET_SAMPLES_PER_CLUSTER_FOR_K > 0 :
            n_clusters_active = max(2, n_target_active // TARGET_SAMPLES_PER_CLUSTER_FOR_K)
        else:
            n_clusters_active = max(2, min(10, num_reshaped_items_active // 2 if num_reshaped_items_active > 2 else 2)) # Fallback K
        
        print(f"Using n_clusters = {n_clusters_active} for Test Data (n_target_active={n_target_active})")

        # KMeans clustering on UMAP results
        # umap_active should contain 'index', 'UMAP1', 'UMAP2'
        cluster_active = kmeans_clustering(umap_active, n_clusters=n_clusters_active, random_state=42)

        # Plot UMAP of full test data with cluster colors
        if not umap_active.empty and not cluster_active.empty:
            # Merge cluster_active with umap_active for plotting
            # Need to align cluster labels with umap_active based on 'index'
            plot_data_full_active = umap_active.merge(cluster_active, on='index', how='left')
            if 'Cluster' in plot_data_full_active.columns:
                plot_umap_data(
                    umap_coords_df=plot_data_full_active[["UMAP1", "UMAP2"]],
                    cluster_labels_series=plot_data_full_active["Cluster"],
                    title=f"UMAP projection of test data ({args.data}, k={n_clusters_active})",
                    filename=f"figs/umap_projection_{args.data}_full.png",
                )
            else:
                print("Warning: 'Cluster' column not found after merging for full active data plot.")
        else:
            print("Skipping UMAP plot for full test data: UMAP or cluster data is empty.")

        # Plot histogram of cluster point counts for active data
        if not cluster_active.empty:
            plot_cluster_point_counts_histogram(
                cluster_info_df=cluster_active, # Contains 'index' and 'Cluster'
                title=f"Point Counts per Cluster - Test Data ({args.data}, k={n_clusters_active})",
                filename=f"figs/histogram_points_per_cluster_{args.data}.png"
            )
        
        # Prepare data for sampling: merge reshaped data with UMAP and cluster info
        # df_active_reshaped has 'index' and features (V0, V1, ...)
        # umap_active has 'index', 'UMAP1', 'UMAP2'
        # cluster_active has 'index', 'Cluster'
        df_active_for_sampling = df_active_reshaped.merge(umap_active, on="index", how="left")
        df_active_for_sampling = df_active_for_sampling.merge(cluster_active, on="index", how="left")


        # Sample active data using the new strategy
        # df_active_for_sampling must contain 'index', 'Cluster', (and optionally features, UMAP coords for consistency)
        sampled_active, _ = sampling_by_cluster_point_count_inverse_weighted(
            df_active_for_sampling,
            n_target_active,
            min_samples_per_cluster=min_samples_per_cluster_in_sampling
        )

        # Plot UMAP of sampled test data
        if not sampled_active.empty:
            # sampled_active now contains original features, UMAP1, UMAP2, Cluster, index
            # We can directly use its UMAP coordinates for plotting, no need to re-embed if they are there.
            # However, for consistency with train/val and if UMAP was dropped, let's re-embed the sampled features.
            # The 'Cluster' column in sampled_active is the original cluster ID from full data.
            
            # Option 1: Re-embed sampled_active (if UMAP coords were not in df_active_for_sampling or were dropped)
            # df_to_umap_sampled = sampled_active.drop(columns=['UMAP1', 'UMAP2', 'Cluster'], errors='ignore')
            # umap_sampled_active_coords, _ = umap_map_and_embed(
            #      df_to_umap_sampled, # Should contain 'index' and features
            #      n_neighbors=max(2, min(10, len(df_to_umap_sampled)//2 if len(df_to_umap_sampled)>4 else 2)), # Adjust n_neighbors for smaller sample
            #      min_dist=0.001, n_components=2, random_state=42
            # )
            # # Merge with the original cluster labels from sampled_active
            # if 'index' in umap_sampled_active_coords.columns and 'index' in sampled_active.columns and 'Cluster' in sampled_active.columns:
            #      umap_sampled_active_plot_df = umap_sampled_active_coords.merge(sampled_active[['index', 'Cluster']], on='index', how='left')
            #      if 'Cluster' in umap_sampled_active_plot_df.columns and not umap_sampled_active_plot_df.empty:
            #          plot_umap_data(
            #              umap_coords_df=umap_sampled_active_plot_df[["UMAP1", "UMAP2"]],
            #              cluster_labels_series=umap_sampled_active_plot_df["Cluster"],
            #              title=f"UMAP of sampled test data ({args.data}, k={n_clusters_active})",
            #              filename=f"figs/umap_projection_sampled_{args.data}.png",
            #          )
            #      else:
            #          print("Warning: Could not prepare UMAP plot for sampled active data (missing 'Cluster' after merge or empty).")
            # else:
            #      print("Warning: 'index' or 'Cluster' missing for sampled active UMAP plot preparation.")


            # Option 2: Use existing UMAP coordinates if they were carried into sampled_active
            if 'UMAP1' in sampled_active.columns and 'UMAP2' in sampled_active.columns and 'Cluster' in sampled_active.columns:
                 plot_umap_data(
                     umap_coords_df=sampled_active[["UMAP1", "UMAP2"]],
                     cluster_labels_series=sampled_active["Cluster"],
                     title=f"UMAP of sampled test data ({args.data}, k={n_clusters_active})",
                     filename=f"figs/umap_projection_sampled_{args.data}.png",
                 )
            else: # Fallback to re-embedding if UMAP coords are not directly in sampled_active
                print("Info: UMAP1/UMAP2 not directly in sampled_active, attempting re-embedding for plot.")
                # Drop potentially existing UMAP/Cluster to avoid issues in umap_map_and_embed if it expects raw features
                df_to_umap_sampled = sampled_active.drop(columns=['UMAP1', 'UMAP2', 'Cluster'], errors='ignore')
                
                # Ensure df_to_umap_sampled has enough neighbors for UMAP
                n_neighbors_sampled = max(2, min(10, len(df_to_umap_sampled) -1 if len(df_to_umap_sampled) > 1 else 1 ) ) # n_neighbors < n_samples
                if len(df_to_umap_sampled) <= n_neighbors_sampled : # if not enough samples for n_neighbors
                    print(f"Warning: Not enough samples ({len(df_to_umap_sampled)}) for UMAP with n_neighbors={n_neighbors_sampled}. Skipping sampled plot.")
                else:
                    umap_sampled_active_coords, _ = umap_map_and_embed(
                         df_to_umap_sampled, # Should contain 'index' and features
                         n_neighbors=n_neighbors_sampled,
                         min_dist=0.001, n_components=2, random_state=42
                    )
                    # Merge with the original cluster labels from sampled_active
                    if 'index' in umap_sampled_active_coords.columns and 'index' in sampled_active.columns and 'Cluster' in sampled_active.columns:
                         umap_sampled_active_plot_df = umap_sampled_active_coords.merge(sampled_active[['index', 'Cluster']], on='index', how='left')
                         if 'Cluster' in umap_sampled_active_plot_df.columns and not umap_sampled_active_plot_df.empty:
                             plot_umap_data(
                                 umap_coords_df=umap_sampled_active_plot_df[["UMAP1", "UMAP2"]],
                                 cluster_labels_series=umap_sampled_active_plot_df["Cluster"],
                                 title=f"UMAP of sampled test data ({args.data}, k={n_clusters_active})",
                                 filename=f"figs/umap_projection_sampled_{args.data}.png",
                             )
                         else:
                             print("Warning: Could not prepare UMAP plot for sampled active data (missing 'Cluster' after merge or empty).")
                    else:
                         print("Warning: 'index' or 'Cluster' missing for sampled active UMAP plot preparation.")


        else:
            print("Skipping UMAP plot for sampled test data: sampled_active is empty.")

        # Save sampled data (original format)
        save_sampled_data_from_original(
            original_long_df=df_test_original,
            sampled_reshaped_df=sampled_active, # This df must have 'index' referring to reshaped blocks
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

        # Define Sampling Target Sizes
        if args.data == "normal":
            n_target_train, n_target_val = 8000, 2000
        elif args.data == "more":
            n_target_train, n_target_val = 80000, 20000
        elif args.data == "much":
            n_target_train, n_target_val = 800000, 200000
        else:
            print(f"Warning: Unknown data size '{args.data}'. Using default small sample sizes.")
            n_target_train, n_target_val = 1000, 200 # Default fallback

        # Calculate n_clusters based on n_target and TARGET_SAMPLES_PER_CLUSTER_FOR_K
        if n_target_train > 0 and TARGET_SAMPLES_PER_CLUSTER_FOR_K > 0:
            n_clusters_train = max(2, n_target_train // TARGET_SAMPLES_PER_CLUSTER_FOR_K)
        else: # Fallback K for training
            df_train_temp_reshaped_len = len(df_train_original) // n_cols if df_train_original is not None and not df_train_original.empty else 0
            n_clusters_train = max(2, min(10, df_train_temp_reshaped_len // 2 if df_train_temp_reshaped_len > 2 else 2))

        if n_target_val > 0 and TARGET_SAMPLES_PER_CLUSTER_FOR_K > 0:
            n_clusters_val = max(2, n_target_val // TARGET_SAMPLES_PER_CLUSTER_FOR_K)
        else: # Fallback K for validation
            df_val_temp_reshaped_len = len(df_val_original) // n_cols if df_val_original is not None and not df_val_original.empty else 0
            n_clusters_val = max(2, min(10, df_val_temp_reshaped_len // 2 if df_val_temp_reshaped_len > 2 else 2))


        print(f"Using n_clusters = {n_clusters_train} for Training Data (n_target_train={n_target_train})")
        print(f"Using n_clusters = {n_clusters_val} for Validation Data (n_target_val={n_target_val})")

        # --- Training Data Processing ---
        print("--- Training Data ---")
        sampled_train = pd.DataFrame() # Initialize to ensure it exists
        if df_train_original.empty:
            print("Original training data is empty. Skipping training data processing.")
        else:
            df_train_reshaped = reshape_df(df_train_original, n_cols)
            if df_train_reshaped.empty:
                print("Reshaped training data is empty. Cannot proceed with training path.")
            else:
                umap_train, _ = umap_map_and_embed(
                    df_train_reshaped.drop(columns=['Cluster'], errors='ignore'),
                    n_neighbors=10, min_dist=0.001, n_components=2, random_state=42
                )
                cluster_train = kmeans_clustering(umap_train, n_clusters=n_clusters_train, random_state=42)

                if not umap_train.empty and not cluster_train.empty:
                    plot_data_full_train = umap_train.merge(cluster_train, on='index', how='left')
                    if 'Cluster' in plot_data_full_train.columns:
                        plot_umap_data(
                            umap_coords_df=plot_data_full_train[["UMAP1", "UMAP2"]],
                            cluster_labels_series=plot_data_full_train["Cluster"],
                            title=f"UMAP projection of training data ({args.data}, k={n_clusters_train})",
                            filename=f"figs/umap_projection_{args.data}_train_full.png",
                        )
                else:
                    print("Skipping UMAP plot for full training data: UMAP or cluster data is empty.")

                if not cluster_train.empty:
                    plot_cluster_point_counts_histogram(
                        cluster_info_df=cluster_train,
                        title=f"Point Counts per Cluster - Training Data ({args.data}, k={n_clusters_train})",
                        filename=f"figs/histogram_points_per_cluster_{args.data}_train.png"
                    )
                
                df_train_for_sampling = df_train_reshaped.merge(umap_train, on="index", how="left")
                df_train_for_sampling = df_train_for_sampling.merge(cluster_train, on="index", how="left")

                print("--- Sampling Training Data ---")
                sampled_train, _ = sampling_by_cluster_point_count_inverse_weighted(
                    df_train_for_sampling,
                    n_target_train,
                    min_samples_per_cluster=min_samples_per_cluster_in_sampling
                )
                
                if not sampled_train.empty:
                    if 'UMAP1' in sampled_train.columns and 'UMAP2' in sampled_train.columns and 'Cluster' in sampled_train.columns:
                        plot_umap_data(
                            umap_coords_df=sampled_train[["UMAP1", "UMAP2"]],
                            cluster_labels_series=sampled_train["Cluster"],
                            title=f"UMAP of sampled training data ({args.data}, k={n_clusters_train})",
                            filename=f"figs/umap_projection_sampled_{args.data}_train.png",
                        )
                    else: # Fallback to re-embedding
                        print("Info: UMAP1/UMAP2 not directly in sampled_train, attempting re-embedding for plot.")
                        df_to_umap_sampled_train = sampled_train.drop(columns=['UMAP1', 'UMAP2', 'Cluster'], errors='ignore')
                        n_neighbors_sampled_train = max(2, min(10, len(df_to_umap_sampled_train) -1 if len(df_to_umap_sampled_train) > 1 else 1))
                        if len(df_to_umap_sampled_train) <= n_neighbors_sampled_train:
                             print(f"Warning: Not enough samples ({len(df_to_umap_sampled_train)}) for UMAP with n_neighbors={n_neighbors_sampled_train}. Skipping sampled train plot.")
                        else:
                            umap_sampled_train_coords, _ = umap_map_and_embed(
                                df_to_umap_sampled_train,
                                n_neighbors=n_neighbors_sampled_train, min_dist=0.001, n_components=2, random_state=42)
                            if 'index' in umap_sampled_train_coords.columns and 'index' in sampled_train.columns and 'Cluster' in sampled_train.columns:
                                umap_sampled_train_plot_df = umap_sampled_train_coords.merge(sampled_train[['index', 'Cluster']], on='index', how='left')
                                if 'Cluster' in umap_sampled_train_plot_df.columns and not umap_sampled_train_plot_df.empty:
                                    plot_umap_data(
                                        umap_coords_df=umap_sampled_train_plot_df[["UMAP1", "UMAP2"]],
                                        cluster_labels_series=umap_sampled_train_plot_df["Cluster"],
                                        title=f"UMAP of sampled training data ({args.data}, k={n_clusters_train})",
                                        filename=f"figs/umap_projection_sampled_{args.data}_train.png",
                                    )
                else:
                    print("Skipping UMAP plot for sampled training data: sampled_train is empty.")


        # --- Validation Data Processing ---
        print("--- Validation Data ---")
        sampled_val = pd.DataFrame() # Initialize
        if df_val_original.empty:
            print("Original validation data is empty. Skipping validation data processing.")
        else:
            df_val_reshaped = reshape_df(df_val_original, n_cols)
            if df_val_reshaped.empty:
                print("Reshaped validation data is empty. Skipping validation processing and sampling.")
            else:
                umap_val, _ = umap_map_and_embed(
                    df_val_reshaped.drop(columns=['Cluster'], errors='ignore'),
                    n_neighbors=10, min_dist=0.001, n_components=2, random_state=42
                )
                cluster_val = kmeans_clustering(umap_val, n_clusters=n_clusters_val, random_state=42)
                
                if not umap_val.empty and not cluster_val.empty:
                    plot_data_full_val = umap_val.merge(cluster_val, on='index', how='left')
                    if 'Cluster' in plot_data_full_val.columns:
                        plot_umap_data(
                            umap_coords_df=plot_data_full_val[["UMAP1", "UMAP2"]], 
                            cluster_labels_series=plot_data_full_val["Cluster"], 
                            title=f"UMAP of validation data ({args.data}, k={n_clusters_val})", 
                            filename=f"figs/umap_projection_{args.data}_val_full.png"
                        )
                else:
                    print("Skipping UMAP plot for full validation data: UMAP or cluster data is empty.")

                if not cluster_val.empty:
                    plot_cluster_point_counts_histogram(
                        cluster_info_df=cluster_val,
                        title=f"Point Counts per Cluster - Validation Data ({args.data}, k={n_clusters_val})",
                        filename=f"figs/histogram_points_per_cluster_{args.data}_val.png"
                    )

                df_val_for_sampling = df_val_reshaped.merge(umap_val, on="index", how="left")
                df_val_for_sampling = df_val_for_sampling.merge(cluster_val, on="index", how="left")
                
                print("--- Sampling Validation Data ---")
                sampled_val, _ = sampling_by_cluster_point_count_inverse_weighted(
                    df_val_for_sampling,
                    n_target_val,
                    min_samples_per_cluster=min_samples_per_cluster_in_sampling
                )
                print(f"Sampled Validation DataFrame shape: {sampled_val.shape if not sampled_val.empty else 'Empty'}")

                if not sampled_val.empty:
                    if 'UMAP1' in sampled_val.columns and 'UMAP2' in sampled_val.columns and 'Cluster' in sampled_val.columns:
                        plot_umap_data(
                            umap_coords_df=sampled_val[["UMAP1", "UMAP2"]],
                            cluster_labels_series=sampled_val["Cluster"],
                            title=f"UMAP of sampled validation data ({args.data}, k={n_clusters_val})",
                            filename=f"figs/umap_projection_sampled_{args.data}_val.png",
                        )
                    else: # Fallback to re-embedding
                        print("Info: UMAP1/UMAP2 not directly in sampled_val, attempting re-embedding for plot.")
                        df_to_umap_sampled_val = sampled_val.drop(columns=['UMAP1', 'UMAP2', 'Cluster'], errors='ignore')
                        n_neighbors_sampled_val = max(2, min(10, len(df_to_umap_sampled_val) -1 if len(df_to_umap_sampled_val) > 1 else 1))
                        if len(df_to_umap_sampled_val) <= n_neighbors_sampled_val :
                             print(f"Warning: Not enough samples ({len(df_to_umap_sampled_val)}) for UMAP with n_neighbors={n_neighbors_sampled_val}. Skipping sampled val plot.")
                        else:
                            umap_sampled_val_coords, _ = umap_map_and_embed(
                                df_to_umap_sampled_val,
                                n_neighbors=n_neighbors_sampled_val, min_dist=0.001, n_components=2, random_state=42)
                            if 'index' in umap_sampled_val_coords.columns and 'index' in sampled_val.columns and 'Cluster' in sampled_val.columns:
                                umap_sampled_val_plot_df = umap_sampled_val_coords.merge(sampled_val[['index', 'Cluster']], on='index', how='left')
                                if 'Cluster' in umap_sampled_val_plot_df.columns and not umap_sampled_val_plot_df.empty:
                                    plot_umap_data(
                                        umap_coords_df=umap_sampled_val_plot_df[["UMAP1", "UMAP2"]],
                                        cluster_labels_series=umap_sampled_val_plot_df["Cluster"],
                                        title=f"UMAP of sampled validation data ({args.data}, k={n_clusters_val})",
                                        filename=f"figs/umap_projection_sampled_{args.data}_val.png",
                                    )
                else:
                    print("Skipping UMAP plot for sampled validation data: sampled_val is empty.")


        # --- Save Sampled Data ---
        if not df_train_original.empty and not sampled_train.empty :
            save_sampled_data_from_original(
                original_long_df=df_train_original, sampled_reshaped_df=sampled_train,
                n_cols_reshape=n_cols, output_path=f"data_{args.data}/train.parquet",
                dataset_label_for_log="training",
            )
        elif df_train_original.empty:
            print("Original training data was empty, skipping save for training set.")
        elif sampled_train.empty and not df_train_original.empty :
            print("Sampled training data is empty, skipping save for training set.")


        if not df_val_original.empty and not sampled_val.empty:
            save_sampled_data_from_original(
                original_long_df=df_val_original, sampled_reshaped_df=sampled_val,
                n_cols_reshape=n_cols, output_path=f"data_{args.data}/val.parquet",
                dataset_label_for_log="validation",
            )
        elif df_val_original.empty:
            print("Original validation data was empty, skipping save for validation set.")
        elif sampled_val.empty and not df_val_original.empty:
            print("Sampled validation data is empty, skipping save for validation set.")


if __name__ == "__main__":
    main()
