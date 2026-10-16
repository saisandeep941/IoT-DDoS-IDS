
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "dataset" / "01-12" / "UDPLag.csv"
MODEL_FEATURES_PATH = ROOT / "backend" / "real_model_features.joblib"

API_URLS = [
    "http://127.0.0.1:8000/api/predict",
    "http://127.0.0.1:8000/api/risk-score",
]

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

feature_names = joblib.load(MODEL_FEATURES_PATH)

df = pd.read_csv(CSV_PATH, nrows=5, low_memory=False)
df.columns = df.columns.str.strip()

missing = [name for name in feature_names if name not in df.columns]
if missing:
    raise ValueError(f"CSV is missing required features: {missing}")

row = df.iloc[0]
X = pd.to_numeric(row[feature_names], errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

payload = {"features": X.astype(float).tolist()}

print("Dataset:", CSV_PATH.name)
print("Feature count:", len(payload["features"]))
print("Actual label:", row.get("Label", "Not available"))
print("\nSending real traffic record to the API...")

for url in API_URLS:
    print(f"\nEndpoint: {url}")

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            print("HTTP status:", response.status)
            print(json.dumps(json.loads(response.read()), indent=2))
    except HTTPError as error:
        print("HTTP error:", error.code)
        print(error.read().decode("utf-8"))
    except URLError as error:
        print("Connection error:", error.reason)
