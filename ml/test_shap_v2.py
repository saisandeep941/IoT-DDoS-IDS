
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "dataset" / "01-12" / "UDPLag.csv"
MODEL_FILE = BASE_DIR / "backend" / "real_model_v2.joblib"
FEATURE_FILE = BASE_DIR / "backend" / "real_model_features_v2.joblib"

pipeline = joblib.load(MODEL_FILE)
features = joblib.load(FEATURE_FILE)

print("Loading a small evaluation sample...")

chunk = pd.read_csv(DATA_FILE, nrows=2000, low_memory=False)
chunk.columns = chunk.columns.str.strip()

if "Label" not in chunk.columns:
    raise ValueError("Label column not found.")

missing = [col for col in features if col not in chunk.columns]
if missing:
    raise ValueError(f"Missing features: {missing}")

X = chunk.reindex(columns=features)
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Apply the exact imputer fitted during V2 training.
imputer = pipeline.named_steps["imputer"]
model = pipeline.named_steps["model"]
X_imputed = imputer.transform(X)

# Limit SHAP computation for speed.
X_explain = X_imputed[:100]

print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_explain)

if isinstance(shap_values, list):
    attack_values = shap_values[1]
elif getattr(shap_values, "ndim", 0) == 3:
    attack_values = shap_values[:, :, 1]
else:
    attack_values = shap_values

mean_importance = np.abs(attack_values).mean(axis=0)

importance = (
    pd.DataFrame({
        "Feature": features,
        "MeanAbsSHAP": mean_importance,
    })
    .sort_values("MeanAbsSHAP", ascending=False)
)

print("\n========== V2 SHAP RESULTS ==========")
print("Model: Random Forest inside V2 pipeline")
print("Rows explained:", len(X_explain))
print("Features:", len(features))
print("\nTop 15 features:")
print(importance.head(15).to_string(index=False))

print("\nSHAP verification completed.")
