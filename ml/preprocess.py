
from pathlib import Path
import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset" / "01-12"
OUTPUT_FILE = Path(__file__).resolve().parent / "processed_without_udplag.csv"

if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

csv_files = [
    file for file in DATASET_DIR.glob("*.csv")
    if file.name.lower() != "udplag.csv"
]
total_rows = 0
label_counts = {}

for file in csv_files:
    print(f"Processing: {file.name}")

    try:
        for chunk in pd.read_csv(file, chunksize=50000, low_memory=False):
            chunk.columns = chunk.columns.str.strip()

            label_col = next(
                (c for c in chunk.columns if c.strip().lower() == "label"),
                None
            )

            if label_col is None:
                print(f"Skipping {file.name}: Label column not found")
                break

            chunk.rename(columns={label_col: "Label"}, inplace=True)
            chunk.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
            chunk.dropna(subset=["Label"], inplace=True)
            chunk["Label"] = chunk["Label"].astype(str).str.strip()

            # Count labels before writing each chunk
            counts = chunk["Label"].value_counts()
            for label, count in counts.items():
                label_counts[label] = label_counts.get(label, 0) + int(count)

            chunk.to_csv(
                OUTPUT_FILE,
                mode="a",
                header=not OUTPUT_FILE.exists() or OUTPUT_FILE.stat().st_size == 0,
                index=False
            )
            total_rows += len(chunk)

    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        print(f"Skipped unreadable file {file.name}: {error}")
        continue

print("\nPreprocessing complete.")
print(f"Rows processed: {total_rows}")
print(f"Output: {OUTPUT_FILE}")
print("\nLabel distribution:")
for label, count in sorted(label_counts.items()):
    print(f"{label}: {count}")
