import pandas as pd
filename="example2.preds.parquet"

df = pd.read_parquet(filename)
print(df)
