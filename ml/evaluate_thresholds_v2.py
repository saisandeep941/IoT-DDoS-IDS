
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
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "dataset" / "01-12" / "UDPLag.csv"
MODEL_FILE = BASE_DIR / "backend" / "real_model_v2.joblib"
FEATURES_FILE = BASE_DIR / "backend" / "real_model_features_v2.joblib"

MAX_PER_CLASS = 20_000
CHUNK_SIZE = 100_000
THRESHOLDS = [0.50, 0.40, 0.30, 0.20]

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

pipeline = joblib.load(MODEL_FILE)
features = joblib.load(FEATURES_FILE)
rng = np.random.default_rng(42)

reservoirs = {0: [], 1: []}
seen = {0: 0, 1: 0}

for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()

    labels = chunk["Label"].astype(str).str.strip().str.upper()
    y_chunk = labels.ne("BENIGN").astype(int)

    X_chunk = chunk.reindex(columns=features)
    X_chunk = X_chunk.apply(pd.to_numeric, errors="coerce")
    X_chunk = X_chunk.replace([np.inf, -np.inf], np.nan)

    for cls in (0, 1):
        class_rows = X_chunk.loc[y_chunk == cls].to_numpy()

        for row in class_rows:
            seen[cls] += 1
            reservoir = reservoirs[cls]

            if len(reservoir) < MAX_PER_CLASS:
                reservoir.append(row)
            else:
                index = rng.integers(0, seen[cls])
                if index < MAX_PER_CLASS:
                    reservoir[index] = row

print("Available BENIGN rows:", seen[0])
print("Available ATTACK rows:", seen[1])
print("Sampled BENIGN rows:", len(reservoirs[0]))
print("Sampled ATTACK rows:", len(reservoirs[1]))

X_eval = pd.DataFrame(
    reservoirs[0] + reservoirs[1],
    columns=features,
)
y_true = np.array(
    [0] * len(reservoirs[0]) + [1] * len(reservoirs[1])
)

attack_class_index = list(pipeline.classes_).index(1)
import time

start_time = time.perf_counter()
attack_probabilities = pipeline.predict_proba(X_eval)[:, attack_class_index]
inference_time = time.perf_counter() - start_time

print("\n--- Inference Efficiency ---")
print(f"Samples evaluated: {len(X_eval)}")
print(f"Total inference time: {inference_time:.6f} seconds")
print(f"Average time per sample: {inference_time / len(X_eval) * 1000:.6f} ms")
print(f"Throughput: {len(X_eval) / inference_time:.2f} samples/second")

print("\n--- Threshold Comparison ---")

for threshold in THRESHOLDS:
    y_pred = (attack_probabilities > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"False positives: {fp}")
    print(f"Missed attacks: {fn}")
    print(f"False-positive rate: {false_positive_rate:.4%}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
