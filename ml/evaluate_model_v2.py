
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "dataset" / "01-12" / "UDPLag.csv"
MODEL_FILE = BASE_DIR / "backend" / "real_model_v2.joblib"
FEATURE_FILE = BASE_DIR / "backend" / "real_model_features_v2.joblib"

CHUNK_SIZE = 50_000
MAX_PER_CLASS = 20_000
RANDOM_STATE = 42

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

pipeline = joblib.load(MODEL_FILE)
features = joblib.load(FEATURE_FILE)

rng = np.random.default_rng(RANDOM_STATE)
reservoir = {0: [], 1: []}
seen = {0: 0, 1: 0}

print("Evaluating V2 on UDPLag.csv...")

for chunk in pd.read_csv(
    DATA_FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False,
):
    chunk.columns = chunk.columns.str.strip()

    if "Label" not in chunk.columns:
        raise ValueError("UDPLag.csv has no Label column.")

    labels_text = (
        chunk["Label"].astype(str).str.strip().str.upper()
    )
    y = labels_text.ne("BENIGN").astype(int)

    missing = [col for col in features if col not in chunk.columns]
    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    X = chunk.reindex(columns=features)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    for cls in (0, 1):
        rows = X.loc[y == cls]
        for row in rows.itertuples(index=False, name=None):
            seen[cls] += 1

            if len(reservoir[cls]) < MAX_PER_CLASS:
                reservoir[cls].append(row)
            else:
                j = rng.integers(0, seen[cls])
                if j < MAX_PER_CLASS:
                    reservoir[cls][j] = row

print("\nAvailable rows:")
print(f"BENIGN: {seen[0]:,}")
print(f"ATTACK: {seen[1]:,}")

if not reservoir[0] or not reservoir[1]:
    raise ValueError(
        "Both BENIGN and ATTACK rows are required for evaluation."
    )

X_eval = pd.DataFrame(
    reservoir[0] + reservoir[1],
    columns=features,
)

y_eval = np.array(
    [0] * len(reservoir[0]) + [1] * len(reservoir[1])
)

predictions = pipeline.predict(X_eval)

print("\n========== UDPLAG EVALUATION ==========")
print(f"Evaluation rows: {len(y_eval):,}")
print(f"Features: {len(features)}")
print(f"Accuracy:  {accuracy_score(y_eval, predictions):.4f}")
print(
    f"Precision: {precision_score(y_eval, predictions, zero_division=0):.4f}"
)
print(
    f"Recall:    {recall_score(y_eval, predictions, zero_division=0):.4f}"
)
print(f"F1 score:  {f1_score(y_eval, predictions, zero_division=0):.4f}")

print("\nConfusion matrix (0=BENIGN, 1=ATTACK):")
print(confusion_matrix(y_eval, predictions, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_eval,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0,
    )
)

print("\nEvaluation completed. No model files were changed.")
