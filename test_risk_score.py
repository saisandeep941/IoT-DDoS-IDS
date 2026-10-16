
import json
import joblib
import pandas as pd

dataset_path = "dataset/01-12/UDPLag.csv"
feature_names = joblib.load("backend/real_model_features_v2.joblib")

for chunk in pd.read_csv(dataset_path, chunksize=10000, low_memory=False):
    chunk.columns = chunk.columns.str.strip()

    label_col = next(
        (c for c in chunk.columns if c.lower() == "label"),
        None
    )

    if label_col is None:
        raise ValueError("Label column was not found.")

    benign = chunk[
        chunk[label_col].astype(str).str.strip().str.upper() == "BENIGN"
    ]

    if not benign.empty:
        row = benign.iloc[0].reindex(feature_names)
        row = pd.to_numeric(row, errors="coerce")
        row = row.replace([float("inf"), float("-inf")], float("nan"))
        row = row.fillna(0)

        request_body = {"features": row.tolist()}

        # Test input validation with exactly 82 features.
        request_body["features"][0] = "Infinity"

        with open("risk_infinity_test.json", "w") as file:
            json.dump(request_body, file)

        print("Created risk_infinity_test.json")
        print("Feature count:", len(request_body["features"]))
        print("First feature:", request_body["features"][0])
        break
else:
    raise ValueError("No BENIGN row was found.")
