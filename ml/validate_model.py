
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "dataset" / "01-12" / "UDPLag.csv"
MODEL_FILE = BASE_DIR / "backend" / "real_model.joblib"
FEATURE_FILE = BASE_DIR / "backend" / "real_model_features.joblib"
MEDIAN_FILE = BASE_DIR / "ml" / "recovered_medians.joblib"

SAMPLE_PER_CLASS = 3705
RANDOM_STATE = 42
CHUNK_SIZE = 50_000

model = joblib.load(MODEL_FILE)
features = joblib.load(FEATURE_FILE)
medians = joblib.load(MEDIAN_FILE)

# Collect a reproducible sample from the CSV.
benign_parts = []
attack_parts = []

for chunk in pd.read_csv(
    DATA_FILE, chunksize=CHUNK_SIZE, low_memory=False
):
    chunk.columns = chunk.columns.str.strip()

    if "Label" not in chunk.columns:
        raise ValueError("Label column not found in test CSV.")

    chunk["Label"] = chunk["Label"].astype(str).str.strip().str.upper()
    chunk = chunk[chunk["Label"].isin(["BENIGN", "UDP-LAG", "WEBDDOS"])]

    # Confirm every required model feature exists.
    missing = [col for col in features if col not in chunk.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    benign = chunk[chunk["Label"] == "BENIGN"]
    attacks = chunk[chunk["Label"] != "BENIGN"]

    if not benign.empty:
        benign_parts.append(benign)

    if not attacks.empty:
        attack_parts.append(attacks)

    print(
        f"Rows scanned; BENIGN collected: "
        f"{sum(len(x) for x in benign_parts):,}; "
        f"ATTACK collected: {sum(len(x) for x in attack_parts):,}"
    )

benign_df = pd.concat(benign_parts, ignore_index=True)
attack_df = pd.concat(attack_parts, ignore_index=True)

if len(benign_df) < SAMPLE_PER_CLASS or len(attack_df) < SAMPLE_PER_CLASS:
    raise ValueError("Not enough rows available for a balanced evaluation.")

# Reproducible balanced sample.
benign_sample = benign_df.sample(
    n=SAMPLE_PER_CLASS, random_state=RANDOM_STATE
)
attack_sample = attack_df.sample(
    n=SAMPLE_PER_CLASS, random_state=RANDOM_STATE
)

sample = pd.concat(
    [benign_sample, attack_sample], ignore_index=True
).sample(frac=1, random_state=RANDOM_STATE)

y_true = sample["Label"].ne("BENIGN").astype(int)

X = sample[features].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(pd.Series(medians)).fillna(0)

y_pred = model.predict(X)

print("\nBalanced independent-file evaluation")
print(f"BENIGN rows: {(y_true == 0).sum():,}")
print(f"ATTACK rows: {(y_true == 1).sum():,}")
print(f"Total rows:  {len(y_true):,}")
print(f"Accuracy:    {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision:   {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"Recall:      {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"F1 score:    {f1_score(y_true, y_pred, zero_division=0):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0,
    )
)
