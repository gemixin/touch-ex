import pandas as pd

# Import parquet dataframe
df = pd.read_parquet("results/object_classification/baseline_history.parquet")
print(df.head())
