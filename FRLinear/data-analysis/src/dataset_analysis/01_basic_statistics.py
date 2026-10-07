from src.config import DATASET_DIR, DATASET
from src.utils.io import load_dataset

DATA_PATH = DATASET_DIR / DATASET

df = load_dataset(DATA_PATH)

print("=" * 80)
print("Shape")
print(df.shape)

print("=" * 80)
print("Columns")
print(df.columns.tolist())

print("=" * 80)
print("Missing values")
print(df.isnull().sum())

numeric_df = df.select_dtypes(include="number")

print("=" * 80)
print("Describe")
print(numeric_df.describe())

print("=" * 80)
print("Skewness")
print(numeric_df.skew())

print("=" * 80)
print("Kurtosis")
print(numeric_df.kurt())