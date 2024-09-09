import polars as pl
import numpy as np


def load_data(file_path: str):
    df = pl.read_parquet(file_path)
    tensors = [df[col].to_numpy().reshape(-1, 100) for col in df.columns]
    return np.array(tensors)
