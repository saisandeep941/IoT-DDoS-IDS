
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "ml" / "processed_without_udplag.csv"
MODEL_FILE = BASE_DIR / "backend" / "real_model_v2.joblib"
FEATURE_FILE = BASE_DIR / "backend" / "real_model_features_v2.joblib"

RANDOM_STATE = 42
SAMPLE_PER_CLASS = 100_000
CHUNK_SIZE = 50_000

EXCLUDED_COLUMNS = {
    "Label",
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Unnamed: 0",
}

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")


# --------------------------------------------------
# Reservoir sampling
# --------------------------------------------------

rng = np.random.default_rng(RANDOM_STATE)

reservoir = {0: [], 1: []}
seen = {0: 0, 1: 0}
feature_columns = None

print("Starting dataset scan...")

for chunk in pd.read_csv(
    DATA_FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False,
):
    chunk.columns = chunk.columns.str.strip()

    if "Label" not in chunk.columns:
        raise ValueError("The dataset must contain a Label column.")

    chunk = chunk.dropna(subset=["Label"]).copy()
    chunk = chunk.replace([np.inf, -np.inf], np.nan)

    # Convert original labels: BENIGN = 0, everything else = 1.
    labels = (
        chunk["Label"]
        .astype(str)
        .str.strip()
        .str.upper()
        .ne("BENIGN")
        .astype(int)
    )

    # Establish one consistent feature schema.
    current_columns = [
        col for col in chunk.columns
        if col not in EXCLUDED_COLUMNS
    ]

    if feature_columns is None:
        feature_columns = current_columns
    else:
        missing_columns = set(feature_columns) - set(current_columns)

        if missing_columns:
            raise ValueError(
                "Feature columns changed between chunks. "
                f"Missing columns: {sorted(missing_columns)}"
            )

        chunk = chunk.reindex(
            columns=list(EXCLUDED_COLUMNS & set(chunk.columns))
            + feature_columns
        )

    X = chunk.reindex(columns=feature_columns)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    for cls in (0, 1):
        class_rows = X.loc[labels == cls]

        for row in class_rows.itertuples(index=False, name=None):
            seen[cls] += 1

            if len(reservoir[cls]) < SAMPLE_PER_CLASS:
                reservoir[cls].append(row)
            else:
                j = rng.integers(0, seen[cls])

                if j < SAMPLE_PER_CLASS:
                    reservoir[cls][j] = row

    print(f"Rows scanned: {sum(seen.values()):,}")


# --------------------------------------------------
# Validate sampled data
# --------------------------------------------------

if not reservoir[0] or not reservoir[1]:
    raise ValueError(
        "Dataset must contain both BENIGN and attack rows."
    )

print("\nRows available by class:")
print(f"BENIGN: {seen[0]:,}")
print(f"ATTACK: {seen[1]:,}")

print("\nSample sizes:")
print(f"BENIGN: {len(reservoir[0]):,}")
print(f"ATTACK: {len(reservoir[1]):,}")

X_sample = pd.DataFrame(
    reservoir[0] + reservoir[1],
    columns=feature_columns,
)

# Labels are deliberately assigned as numeric values.
y_sample = pd.Series(
    [0] * len(reservoir[0]) + [1] * len(reservoir[1]),
    name="Label",
    dtype=int,
)

# Remove features that are entirely missing in the sampled data.
valid_features = X_sample.notna().any(axis=0)
X_sample = X_sample.loc[:, valid_features]
features = list(X_sample.columns)

if not features:
    raise ValueError("No usable numeric features were found.")

print("\nSample label counts:")
print(y_sample.value_counts().sort_index())

if y_sample.nunique() != 2:
    raise ValueError("Both classes must be present in the sample.")


# --------------------------------------------------
# Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_sample,
    y_sample,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_sample,
)

print("\nTraining label counts:")
print(y_train.value_counts().sort_index())

print("\nTest label counts:")
print(y_test.value_counts().sort_index())

if y_train.nunique() != 2 or y_test.nunique() != 2:
    raise ValueError(
        "Both classes must exist in training and test sets."
    )


# --------------------------------------------------
# Pipeline: imputation + model
# --------------------------------------------------

pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        ),
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
    ),
])

print("\nTraining versioned V2 model...")
pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

print("\n========== HELD-OUT EVALUATION ==========")
print(f"Training rows: {len(X_train):,}")
print(f"Test rows:     {len(X_test):,}")
print(f"Features:      {len(features)}")

print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
print(
    f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}"
)
print(
    f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}"
)
print(f"F1 score:  {f1_score(y_test, predictions, zero_division=0):.4f}")

print("\nConfusion matrix (rows/columns: 0=BENIGN, 1=ATTACK):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0,
    )
)


# --------------------------------------------------
# Save versioned artifacts only
# --------------------------------------------------

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(pipeline, MODEL_FILE)
joblib.dump(features, FEATURE_FILE)

print("\nV2 model saved successfully:")
print(MODEL_FILE)

print("\nV2 feature list saved successfully:")
print(FEATURE_FILE)

print("\nOriginal model files were not changed.")
