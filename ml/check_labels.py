
from collections import Counter
from pathlib import Path
import pandas as pd

file_path = Path(__file__).resolve().parent / "processed_without_udplag.csv"
counts = Counter()

for chunk in pd.read_csv(
    file_path,
    chunksize=50_000,
    low_memory=False
):
    chunk.columns = chunk.columns.str.strip()

    labels = (
        chunk["Label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    counts.update(labels.value_counts().to_dict())

    print(f"Scanned rows: {sum(counts.values()):,}")

print("\nFINAL LABEL COUNTS")
for label, count in counts.most_common():
    print(f"{label}: {count:,}")
