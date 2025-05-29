import pacmap
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
import fireducks.pandas as pd # fireducks.pandas를 사용한다고 가정합니다.
import numpy as np
import random
# from sklearn.cluster import KMeans # KMeans는 더 이상 필요하지 않습니다.
from scipy.stats import gaussian_kde # 밀도 추정을 위해 추가
import argparse # 스크립트에서 직접 사용되진 않지만, 원래 코드에 있어 유지합니다.
import warnings
import os

warnings.filterwarnings("ignore")


# --- Helper Functions ---
def reshape_df(df: pd.DataFrame, n_cols: int, column: str = "V") -> pd.DataFrame:
    # 긴 형식의 DataFrame을 넓은 형식으로 재구성합니다.
    # 출력 DataFrame의 각 행은 입력 'column'에서 'n_cols'개의 연속된 값에 해당합니다.
    # 원본 그룹화를 추적하기 위해 'index' 열이 추가됩니다.
    V = df[column].to_numpy()
    if V.size == 0:
        # print( # 이것이 예상되는 동작일 경우 너무 많은 경고를 출력하지 않도록 주석 처리
        # f"Warning: Column '{column}' is empty. Reshaping will result in an empty DataFrame."
        # )
        if n_cols > 0:
            reshaped_v_empty = np.array([]).reshape(0, n_cols)
            column_names_empty = [f"V{i}" for i in range(n_cols)]
            empty_df = pd.DataFrame(reshaped_v_empty, columns=column_names_empty)
            # 가능한 경우 원본 'index' 유형과 일치시키거나 기본값(예: int) 사용
            empty_df['index'] = pd.Series(dtype=df.get('index', pd.Series(dtype=np.int64)).dtype if 'index' in df else np.int64)
            return empty_df
        else: # n_cols가 0 이하인 경우
            return pd.DataFrame({"index": pd.Series(dtype=np.int64)})


    if V.size > 0 and n_cols > 0 and V.size % n_cols != 0:
        print(
            f"Warning: The size of the data ({V.size}) is not a multiple of n_cols ({n_cols}). Data will be truncated."
        )
        V = V[: -(V.size % n_cols)] # 배수가 되도록 자르기

    if V.size == 0 and n_cols > 0: # 빈 입력이지만 유효한 n_cols
        reshaped_v = np.array([]).reshape(0, n_cols)
    elif V.size > 0 and n_cols > 0: # 유효한 입력 및 n_cols
        reshaped_v = V.reshape(-1, n_cols)
    elif n_cols == 0 and V.size > 0: # 데이터가 있을 때 n_cols가 0인 경우
        raise ValueError(
            "n_cols cannot be zero if data is not empty and reshape is attempted."
        )
    else: # V.size가 0이고 n_cols가 0인 경우
        reshaped_v = np.array([]).reshape(0, 0)


    column_names = [f"V{i}" for i in range(n_cols)]
    reshaped_df = pd.DataFrame(reshaped_v, columns=column_names)
    reshaped_df['index'] = np.arange(len(reshaped_df)) # 재구성된 데이터에 새 인덱스 할당
    return reshaped_df


def pacmap_map_and_embed(
    df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pacmap.PaCMAP]:
    # PaCMAP 차원 축소를 수행합니다.
    # 'df'는 추적을 위한 'index' 열과 다른 특징 열들을 가질 것으로 예상합니다.
    # 'Cluster' 열이 입력 df의 일부였다면 (예: 샘플링된 데이터의 경우) PaCMAP 적용 전에 여기서 제거됩니다.
    index_series = df['index'].copy()
    # df_no_id는 PaCMAP 전에 'index' 또는 'Cluster'를 포함하지 않아야 합니다.
    df_no_id = df.drop(columns=['index', 'Cluster'], errors='ignore') # 간단화된 제거

    data_for_pacmap = df_no_id.to_numpy()

    if len(data_for_pacmap) == 0:
        print("Warning: Data for PaCMAP is empty. Returning empty PaCMAP DataFrame.")
        empty_pacmap_df = pd.DataFrame(columns=["PaCMAP1", "PaCMAP2", "index"])
        
        # 가능한 경우 인덱스 dtype 일치시키기
        empty_pacmap_df['index'] = pd.Series(dtype=index_series.dtype if not index_series.empty else np.int64)

        # 데이터가 비어있는 경우 빈 mapper 인스턴스 생성
        # PaCMAP은 빈 데이터로 fit할 수 없으므로 mapper만 생성
        mapper = pacmap.PaCMAP(
            n_components=2,
            random_state=random_state,
        )
        return empty_pacmap_df, mapper

    # PaCMAP with default parameters
    mapper = pacmap.PaCMAP(
        n_components=2,
        random_state=random_state,
    )
    
    # fit_transform을 사용하여 fit과 transform을 한번에 수행
    embedding = mapper.fit_transform(data_for_pacmap)

    pacmap_df = pd.DataFrame(embedding, columns=["PaCMAP1", "PaCMAP2"])
    pacmap_df["index"] = index_series.values # 원본 인덱스 복원
    return pacmap_df, mapper

# kmeans_clustering 함수는 요청에 따라 제거되었습니다.

# --- Plotting Functions ---
def _ensure_dir_exists(file_path: str):
    # 주어진 file_path에 대한 디렉터리가 있는지 확인합니다.
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def plot_pacmap_data(
    pacmap_coords_df: pd.DataFrame, # 'PaCMAP1', 'PaCMAP2'를 포함해야 함
    title: str,
    filename: str,
    additional_pacmap_coords_dfs: list[pd.DataFrame] = None,
):
    # 주 데이터에 대한 밀도 컨투어가 있는 PaCMAP 2D 투영과
    # 추가 데이터셋에 대한 산점도를 그립니다.
    _ensure_dir_exists(filename)
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()

        if not pacmap_coords_df.empty and "PaCMAP1" in pacmap_coords_df and "PaCMAP2" in pacmap_coords_df:
            x_coords = pacmap_coords_df["PaCMAP1"].to_numpy()
            y_coords = pacmap_coords_df["PaCMAP2"].to_numpy()

            if len(x_coords) > 0: # 플롯할 데이터가 있는지 확인
                # 선택 사항: 원시 포인트를 미묘하게 플롯, 범례에는 표시 안 함
                ax.scatter(x_coords, y_coords, s=2, color='silver', alpha=0.3, label='_nolegend_')

                # 데이터 기반으로 그리드 한계 정의 (약간의 패딩 포함)
                xmin, xmax = x_coords.min(), x_coords.max()
                ymin, ymax = y_coords.min(), y_coords.max()
                
                # 모든 점이 동일한 경우 문제 방지
                if xmin == xmax:
                    xmin -= 0.5
                    xmax += 0.5
                if ymin == ymax:
                    ymin -= 0.5
                    ymax += 0.5
                
                x_padding = (xmax - xmin) * 0.1
                y_padding = (ymax - ymin) * 0.1
                
                grid_xmin = xmin - x_padding
                grid_xmax = xmax + x_padding
                grid_ymin = ymin - y_padding
                grid_ymax = ymax + y_padding

                # 그리드 생성
                xx, yy = np.mgrid[grid_xmin:grid_xmax:100j, grid_ymin:grid_ymax:100j] # 100x100 그리드

                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([x_coords, y_coords])
                
                # KDE는 최소 2개의 데이터 포인트와 차원당 1개 이상의 고유 값이 필요합니다 (기본 bw_method의 경우).
                if values.shape[1] >= 2 and len(np.unique(x_coords)) > 1 and len(np.unique(y_coords)) > 1 :
                    try:
                        kernel = gaussian_kde(values)
                        f = np.reshape(kernel(positions).T, xx.shape)

                        # 컨투어 플롯
                        contour_plot = ax.contourf(xx, yy, f, levels=10, cmap="Blues", alpha=0.6) # "viridis"도 좋음
                        # 밀도에 대한 컬러바가 필요한 경우:
                        cbar = fig.colorbar(contour_plot, ax=ax, label='Density', fraction=0.046, pad=0.04)
                        cbar.ax.tick_params(labelsize=5)
                    except Exception as e:
                        print(f"Could not compute or plot density contour for {title}: {e}. Raw points plotted.")
                        # 위의 ax.scatter로 원시 포인트는 이미 플롯되었거나, 선호에 따라 여기서 반복 가능
                elif len(x_coords) > 0 : # KDE를 위한 포인트/분산이 충분하지 않음
                     print(f"Info: Not enough data points or variance for KDE in '{title}'. Plotting raw points only for base data.")
                     # 원시 포인트는 위에서 ax.scatter로 이미 플롯됨.
            else: # len(x_coords) == 0
                 print(f"Info: No data points in pacmap_coords_df for '{title}'.")
        else:
            print(f"Warning: PaCMAP coordinate data is empty or invalid for plotting '{title}'. Skipping density plot.")

        # 제공된 경우 추가 PaCMAP 좌표 플롯
        labels = ["SHO", "Double Well", "Morse", "ATW", "STW", "SSTW"]
        colors = ["cyan", "darkviolet", "lime", "orange", "red", "deeppink"] # 가시성을 위해 색상 조정
        markers = ["o", "s", "^", "D", "P", "*"] # 다른 마커 사용

        if additional_pacmap_coords_dfs:
            for i, pacmap_df in enumerate(additional_pacmap_coords_dfs):
                if not pacmap_df.empty and "PaCMAP1" in pacmap_df and "PaCMAP2" in pacmap_df:
                    ax.scatter(
                        pacmap_df["PaCMAP1"].to_numpy(),
                        pacmap_df["PaCMAP2"].to_numpy(),
                        s=20, # 가시성을 위해 크기 증가
                        linewidths=0.5,
                        edgecolors='black', # 명확성을 위해 테두리 색 추가
                        color=colors[i % len(colors)], # 안전을 위해 나머지 연산 사용
                        marker=markers[i % len(markers)],
                        label=labels[i] if i < len(labels) else f"Potential {i+1}"
                    )
                else:
                    print(f"Warning: Additional PaCMAP data {i} is empty or invalid for plotting '{title}'. Skipping.")

        # ax.set_title(title) # 제목은 그림 캡션에 더 적합할 수 있음
        ax.set_xlabel("PaCMAP1")
        ax.set_ylabel("PaCMAP2")
        ax.grid(True, linestyle='--', alpha=0.5)
        if additional_pacmap_coords_dfs and any(not df.empty for df in additional_pacmap_coords_dfs):
             ax.legend(fontsize=5) # 범례 폰트 약간 크게
        try:
            fig.savefig(filename, dpi=600, bbox_inches="tight") # 표준 DPI
            print(f"Saved PaCMAP density plot: {filename}")
        except Exception as e:
            print(f"Error saving PaCMAP plot {filename}: {e}")
        plt.close(fig)


def pacmap_transform(mapper: pacmap.PaCMAP, df: pd.DataFrame, base_data_for_pacmap: np.ndarray = None) -> pd.DataFrame:
    # 제공된 PaCMAP mapper를 사용하여 DataFrame을 변환합니다.
    # 'df'는 추적을 위한 'index' 열을 가질 것으로 예상합니다.
    # PaCMAP의 transform은 원본 학습 데이터가 필요할 수 있습니다.
    if df.empty or not mapper:
        print("Warning: Input DataFrame for pacmap_transform is empty or mapper is invalid. Returning empty DataFrame.")
        return pd.DataFrame(columns=["PaCMAP1", "PaCMAP2", "index"], dtype=df.get('index', pd.Series(dtype=np.int64)).dtype if 'index' in df else np.int64)


    index_series = df['index'].copy()
    # df_no_id는 'index' 또는 'Cluster'를 포함하지 않아야 합니다.
    df_no_id = df.drop(columns=['index', 'Cluster'], errors='ignore')

    if df_no_id.empty or df_no_id.shape[1] == 0:
        print("Warning: DataFrame for PaCMAP transform has no feature columns after dropping index/cluster. Returning empty PaCMAP data.")
        empty_transformed_df = pd.DataFrame(columns=["PaCMAP1", "PaCMAP2"])
        empty_transformed_df["index"] = index_series.values if not index_series.empty else pd.Series(dtype=index_series.dtype)
        return empty_transformed_df

    try:
        # PaCMAP의 transform은 새로운 데이터에 대해서만 작동합니다
        # 원본 학습 데이터와 함께 fit_transform을 다시 수행해야 할 수도 있습니다
        if base_data_for_pacmap is not None:
            # 원본 데이터와 새 데이터를 결합하여 fit_transform
            combined_data = np.vstack([base_data_for_pacmap, df_no_id.to_numpy()])
            all_transformed = mapper.fit_transform(combined_data)
            # 새 데이터에 해당하는 부분만 추출
            transformed_data = all_transformed[len(base_data_for_pacmap):, :]
        else:
            # base_data가 없는 경우 직접 transform 시도
            transformed_data = mapper.transform(df_no_id.to_numpy())
            
        transformed_df = pd.DataFrame(transformed_data, columns=["PaCMAP1", "PaCMAP2"])
        transformed_df["index"] = index_series.values
        return transformed_df
    except Exception as e:
        print(f"Error during PaCMAP transform: {e}. Attempting to refit on new data only.")
        # transform이 실패하면 새 데이터에만 fit_transform 수행
        try:
            new_mapper = pacmap.PaCMAP(n_components=2, random_state=42)
            transformed_data = new_mapper.fit_transform(df_no_id.to_numpy())
            transformed_df = pd.DataFrame(transformed_data, columns=["PaCMAP1", "PaCMAP2"])
            transformed_df["index"] = index_series.values
            return transformed_df
        except Exception as e2:
            print(f"Error during PaCMAP refit: {e2}. Returning empty DataFrame.")
            empty_transformed_df = pd.DataFrame(columns=["PaCMAP1", "PaCMAP2"])
            empty_transformed_df["index"] = index_series.values if not index_series.empty else pd.Series(dtype=index_series.dtype)
            return empty_transformed_df

# --- Main Function ---
def main():
    np.random.seed(42)
    random.seed(42)

    n_cols = 100 # 시계열과 유사한 데이터 재구성을 위한 열 수
    
    # 클러스터링 관련 변수 제거됨

    print(f"--- Processing Test Data (Scenario: Train) ---")
    try:
        df_test_original = pd.read_parquet(f"data_normal/train_cand.parquet")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    if df_test_original.empty:
        print("Test data is empty. Exiting.")
        return

    # 데이터 재구성
    df_active_reshaped = reshape_df(df_test_original, n_cols)
    if df_active_reshaped.empty:
        print("Reshaped test data is empty. Exiting.")
        return

    # 전체 재구성된 데이터에 대한 PaCMAP 임베딩
    # df_active_reshaped를 직접 전달합니다. pacmap_map_and_embed가 'index' 및 'Cluster' 열 제거를 처리합니다.
    pacmap_active, mapper = pacmap_map_and_embed(
        df_active_reshaped, 
        random_state=42
    )
    
    # 원본 학습 데이터 저장 (transform에 필요할 수 있음)
    base_data = df_active_reshaped.drop(columns=['index', 'Cluster'], errors='ignore').to_numpy()
    
    # KMeans 클러스터링 및 관련 로직 제거됨.

    # 물리적으로 관련된 Potential 로드
    potential_files = {
        "sho": "data_analyze/sho.parquet",
        "double_well": "data_analyze/double_well.parquet",
        "morse": "data_analyze/morse.parquet",
        "sawtooth": "data_analyze/sawtooth.parquet", # ATW에 해당
        "mff": "data_analyze/mff.parquet",           # STW에 해당
        "smff": "data_analyze/smff.parquet"          # SSTW에 해당
    }
    
    potential_dfs_reshaped = {}
    for name, path in potential_files.items():
        try:
            df_potential = pd.read_parquet(path)
            if not df_potential.empty:
                potential_dfs_reshaped[name] = reshape_df(df_potential, n_cols)
            else:
                print(f"Warning: Potential data file {path} is empty.")
                # reshape_df가 'index' 열을 가진 빈 DataFrame을 반환하도록 빈 'V' 열을 가진 DataFrame 전달
                potential_dfs_reshaped[name] = reshape_df(pd.DataFrame({'V': pd.Series(dtype=float)}), n_cols)
        except Exception as e:
            print(f"Error loading or reshaping potential {name} from {path}: {e}")
            potential_dfs_reshaped[name] = reshape_df(pd.DataFrame({'V': pd.Series(dtype=float)}), n_cols)


    # 물리적으로 관련된 Potential에 대한 PaCMAP 변환
    pacmap_potentials = []
    # plot_pacmap_data의 레이블 순서에 중요
    potential_order_for_plot = ["sho", "double_well", "morse", "sawtooth", "mff", "smff"]
    for name in potential_order_for_plot:
        df_to_transform = potential_dfs_reshaped.get(name)
        # df_to_transform은 reshape_df에서 'index' 열을 항상 가지므로 None이거나 'index'가 없을 가능성은 낮음
        if df_to_transform is not None and not df_to_transform.empty and 'index' in df_to_transform.columns :
            pacmap_transformed_potential = pacmap_transform(mapper, df_to_transform, base_data)
            pacmap_potentials.append(pacmap_transformed_potential)
        else:
            print(f"Info: Reshaped potential '{name}' is empty or lacks 'index', cannot transform. Appending empty PaCMAP data for plot consistency.")
            # PaCMAP 변환 결과와 동일한 구조의 빈 DataFrame 추가
            empty_df_for_plot = pd.DataFrame(columns=["PaCMAP1", "PaCMAP2", "index"])
            # 인덱스 dtype을 df_to_transform 에서 가져오려고 시도하거나 기본값 사용
            index_dtype = df_to_transform['index'].dtype if df_to_transform is not None and 'index' in df_to_transform else np.int64
            empty_df_for_plot['index'] = pd.Series(dtype=index_dtype)
            pacmap_potentials.append(empty_df_for_plot)


    # 밀도 컨투어 및 중첩된 Potential이 있는 전체 테스트 데이터의 PaCMAP 플롯
    if not pacmap_active.empty and "PaCMAP1" in pacmap_active and "PaCMAP2" in pacmap_active :
        plot_pacmap_data(
            pacmap_coords_df=pacmap_active[["PaCMAP1", "PaCMAP2"]], # PaCMAP 좌표만 전달
            title=f"PaCMAP projection of test data with density contour", # 제목 업데이트
            filename=f"figs/pacmap_density_projection_phys.png", # 파일 이름 업데이트
            additional_pacmap_coords_dfs=pacmap_potentials,
        )
    else:
        print("Skipping PaCMAP density plot for full test data: PaCMAP data is empty or lacks PaCMAP1/PaCMAP2 columns.")


if __name__ == "__main__":
    main()
