
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "dataset" / "01-12" / "UDPLag.csv"
MODEL_FILE = BASE_DIR / "backend" / "real_model_v2.joblib"
FEATURE_FILE = BASE_DIR / "backend" / "real_model_features_v2.joblib"

MAX_ATTACKS = 20_000
CHUNK_SIZE = 50_000
RANDOM_STATE = 42

pipeline = joblib.load(MODEL_FILE)
features = joblib.load(FEATURE_FILE)

rng = np.random.default_rng(RANDOM_STATE)
attack_reservoir = []
seen_attacks = 0

for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()

    labels = chunk["Label"].astype(str).str.strip().str.upper()
    attack_mask = labels.ne("BENIGN")

    missing = [f for f in features if f not in chunk.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = chunk.reindex(columns=features)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    for row in X.loc[attack_mask].itertuples(index=False, name=None):
        seen_attacks += 1

        if len(attack_reservoir) < MAX_ATTACKS:
            attack_reservoir.append(row)
        else:
            j = rng.integers(0, seen_attacks)
            if j < MAX_ATTACKS:
                attack_reservoir[j] = row

X_attack = pd.DataFrame(attack_reservoir, columns=features)
predictions = pipeline.predict(X_attack)

false_negative_indices = np.flatnonzero(predictions == 0)

print(f"Attack rows sampled: {len(X_attack):,}")
print(f"Missed attacks: {len(false_negative_indices):,}")

if len(false_negative_indices):
    missed = X_attack.iloc[false_negative_indices].copy()
    missed.to_csv(BASE_DIR / "ml" / "missed_attacks.csv", index=False)
    print("Saved missed records to ml/missed_attacks.csv")
else:
    print("No missed attacks in this sample.")


if len(false_negative_indices):
    print("\nFeatures with the most variation among missed attacks:")

    missed = X_attack.iloc[false_negative_indices]

    variation = missed.nunique(dropna=False).sort_values(ascending=False)

    print(variation.head(15).to_string())


if len(false_negative_indices):
    missed = X_attack.iloc[false_negative_indices]

    print("\nMissing values in missed attacks:")
    print(missed.isna().sum().sort_values(ascending=False).head(15))

    print("\nFeature statistics for missed attacks:")
    key_features = [
        "Flow Packets/s",
        "Flow Duration",
        "Flow IAT Mean",
        "Fwd Packets/s",
        "Flow Bytes/s",
        "Source Port",
        "Destination Port",
    ]

    print(
        missed[key_features]
        .describe()
        .round(3)
        .to_string()
    )


if len(false_negative_indices):
    pipeline = joblib.load(MODEL_FILE)
    model = pipeline.named_steps["model"]

    importance = pd.Series(
        model.feature_importances_,
        index=features,
    ).sort_values(ascending=False)

    print("\nTop 15 model features:")
    print(importance.head(15).to_string())

    print("\nFeature importance of inspected traffic features:")
    key_features = [
        "Flow Packets/s",
        "Flow Duration",
        "Flow IAT Mean",
        "Fwd Packets/s",
        "Flow Bytes/s",
        "SimillarHTTP",
    ]

    print(importance.reindex(key_features).to_string())


if len(false_negative_indices):
    missed = X_attack.iloc[false_negative_indices]
    detected = X_attack.iloc[np.flatnonzero(predictions == 1)]

    key_features = [
        "Fwd Packet Length Mean",
        "Destination Port",
        "Min Packet Length",
        "Fwd Packet Length Min",
        "Avg Fwd Segment Size",
        "Subflow Fwd Bytes",
    ]

    print("\nMissed attacks vs correctly detected attacks:")
    print("\nMissed attacks:")
    print(missed[key_features].describe().round(3).to_string())

    print("\nCorrectly detected attacks:")
    print(detected[key_features].describe().round(3).to_string())


if len(false_negative_indices):
    probabilities = pipeline.predict_proba(X_attack)
    classes = list(pipeline.classes_)
    attack_index = classes.index(1)

    missed_attack_probabilities = probabilities[
        false_negative_indices, attack_index
    ]

    print("\nAttack probabilities for missed records:")
    print(
        pd.Series(missed_attack_probabilities)
        .describe()
        .round(6)
        .to_string()
    )

    print("\nLowest attack probabilities among missed records:")
    print(
        np.sort(missed_attack_probabilities)[:10]
    )


from sklearn.metrics import confusion_matrix

probabilities = pipeline.predict_proba(X_attack)
attack_index = list(pipeline.classes_).index(1)
attack_probabilities = probabilities[:, attack_index]

print("\nThreshold comparison on sampled UDPLag attack records:")

for threshold in [0.50, 0.40, 0.30, 0.20]:
    predicted_attack = (attack_probabilities >= threshold).astype(int)
    missed = int((predicted_attack == 0).sum())

    print(
        f"Threshold: {threshold:.2f} | "
        f"Missed attacks: {missed:,} / {len(X_attack):,}"
    )
