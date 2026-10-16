
import pandas as pd
import joblib
import numpy as np

file_path = "ml/processed_without_udplag.csv"
features = joblib.load("backend/real_model_features.joblib")

total_values = 0
missing_values = 0

for chunk in pd.read_csv(
    file_path, chunksize=50_000, low_memory=False
):
    chunk.columns = chunk.columns.str.strip()

    X = chunk[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    total_values += X.size
    missing_values += int(X.isna().sum().sum())

    print(
        f"Scanned {total_values:,} feature values; "
        f"missing: {missing_values:,}"
    )

print("\nFINAL RESULTS")
print("Total feature values:", total_values)
print("Total missing values:", missing_values)
print(
    "Missing percentage:",
    round(100 * missing_values / total_values, 6),
    "%"
)

