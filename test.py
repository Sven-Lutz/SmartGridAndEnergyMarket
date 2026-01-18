import pandas as pd

df = pd.read_csv("data/de_by/analysis/measures_baseline_analysis.csv")
print(df.shape)
print(df["policy_area"].value_counts())
print(df["instrument_type"].value_counts())
print(df["municipality_name"].value_counts().head(10))

