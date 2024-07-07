import pandas as pd
filename="example1.preds.parquet"

df = pd.read_parquet(filename)
print(df)
