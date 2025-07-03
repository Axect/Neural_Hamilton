import fireducks.pandas as pd

test_df = pd.read_parquet("./data_test/test.parquet")

test_df.to_parquet("./data_test/test_resaved.parquet", engine="pyarrow", version='2.4')

print("Parquet file resaved successfully!")
