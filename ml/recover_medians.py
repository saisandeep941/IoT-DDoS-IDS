
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "ml" / "processed_without_udplag.csv"
FEATURE_FILE = BASE_DIR / "backend" / "real_model_features.joblib"
OUTPUT_FILE = BASE_DIR / "ml" / "recovered_medians.joblib"

features = joblib.load(FEATURE_FILE)

# Reproduce the original training sample selection.
rng = np.random.default_rng(42)
reservoir = {0: [], 1: []}
seen = {0: 0, 1: 0}
sample_per_class = 100_000

excluded = {
    "Label", "Flow ID", "Source IP", "Destination IP",
    "Timestamp", "Unnamed: 0"
}

for chunk in pd.read_csv(DATA_FILE, chunksize=50_000, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    chunk = chunk.replace([np.inf, -np.inf], np.nan)
    chunk = chunk.dropna(subset=["Label"])

    labels = (
        chunk["Label"].astype(str).str.strip().str.upper()
        .ne("BENIGN").astype(int)
    )

    # Use the exact saved model features and original row order.
    X = chunk.drop(
        columns=[c for c in excluded if c in chunk.columns]
    )
    X = X.reindex(columns=features)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    for cls in (0, 1):
        rows = X.loc[labels == cls].to_dict("records")

        for row in rows:
            seen[cls] += 1
            if len(reservoir[cls]) < sample_per_class:
                reservoir[cls].append(row)
            else:
                j = rng.integers(0, seen[cls])
                if j < sample_per_class:
                    reservoir[cls][j] = row

    print(f"Scanned {sum(seen.values()):,} rows")

sample = pd.DataFrame(reservoir[0] + reservoir[1])
medians = sample.reindex(columns=features).median().fillna(0)

joblib.dump(medians.to_dict(), OUTPUT_FILE)
print(f"\nSaved recovered medians: {OUTPUT_FILE}")
print(f"Features: {len(medians)}")
print(f"Class rows seen: {seen}")
