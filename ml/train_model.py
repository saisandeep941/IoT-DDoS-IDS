
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "ml" / "processed_without_udplag.csv"
MODEL_FILE = BASE_DIR / "backend" / "real_model.joblib"
FEATURE_FILE = BASE_DIR / "backend" / "real_model_features.joblib"

SAMPLE_PER_CLASS = 100_000
CHUNK_SIZE = 50_000
RANDOM_STATE = 42

# Reservoir sampling across the entire dataset, separately by class.
rng = np.random.default_rng(RANDOM_STATE)
reservoir = {0: [], 1: []}
seen = {0: 0, 1: 0}
columns = None

for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()

    if "Label" not in chunk.columns:
        raise ValueError("Label column not found.")

    columns = chunk.columns.tolist()
    chunk = chunk.replace([np.inf, -np.inf], np.nan)
    chunk = chunk.dropna(subset=["Label"])

    y = (
        chunk["Label"].astype(str).str.strip().str.upper()
        .ne("BENIGN").astype(int)
    )

    for cls in (0, 1):
        rows = chunk.loc[y == cls].to_dict("records")

        for row in rows:
            seen[cls] += 1
            if len(reservoir[cls]) < SAMPLE_PER_CLASS:
                reservoir[cls].append(row)
            else:
                j = rng.integers(0, seen[cls])
                if j < SAMPLE_PER_CLASS:
                    reservoir[cls][j] = row

    print(f"Scanned {sum(seen.values()):,} rows")

if not reservoir[0] or not reservoir[1]:
    raise ValueError("Dataset must contain both BENIGN and attack rows.")

data = pd.DataFrame(reservoir[0] + reservoir[1])
data = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

excluded = {
    "Label", "Flow ID", "Source IP", "Destination IP",
    "Timestamp", "Unnamed: 0"
}
X = data.drop(columns=[c for c in excluded if c in data.columns])
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")
X = X.fillna(X.median()).fillna(0)

y = (
    data["Label"].astype(str).str.strip().str.upper()
    .ne("BENIGN").astype(int)
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced",
)
print("\nTraining Random Forest...")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nEvaluation on held-out sample")
print(f"Rows sampled: {len(X):,}")
print(f"Features: {X.shape[1]}")
print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
print(f"F1 score:  {f1_score(y_test, predictions, zero_division=0):.4f}")
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_FILE)
joblib.dump(list(X.columns), FEATURE_FILE)

print(f"\nModel saved: {MODEL_FILE}")
print(f"Features saved: {FEATURE_FILE}")
print("Class rows seen:", seen)
