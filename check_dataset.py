
import pandas as pd

file_path = r"ml\processed_without_udplag.csv"
counts = {}

for chunk in pd.read_csv(
    file_path,
    usecols=["Label"],
    chunksize=500_000,
    low_memory=False,
):
    labels = chunk["Label"].astype(str).str.strip()
    current = labels.value_counts()

    for label, count in current.items():
        counts[label] = counts.get(label, 0) + int(count)

print("\nComplete dataset label counts:")
for label, count in sorted(counts.items()):
    print(f"{label}: {count:,}")

print(f"\nTotal rows: {sum(counts.values()):,}")
