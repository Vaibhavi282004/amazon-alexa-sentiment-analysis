import pandas as pd

# Load dataset
df = pd.read_csv("data/amazon_alexa.tsv", sep="\t")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())