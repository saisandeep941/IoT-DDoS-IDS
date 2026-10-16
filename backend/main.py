import pandas as pd
import io
import shap
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, HTTPException
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime, timezone
from collections import deque
from datetime import datetime, timezone
from threading import Lock


app = FastAPI(title="SentinelAI API")
detection_records = deque(maxlen=500)
detection_history_lock = Lock()


def record_detection(prediction, risk_score, severity, source):
    with detection_history_lock:
        detection_records.appendleft({
            "id": len(detection_records) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction": prediction,
            "risk_score": round(float(risk_score), 2),
            "severity": severity,
            "source": source,
        })

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "SentinelAI API is running",
        "status": "success"
    }




@app.get("/api/dashboard")
def get_dashboard():
    with detection_history_lock:
        history = list(detection_records)

    total_traffic = len(history)
    attacks_detected = sum(
        1 for item in history
        if item["prediction"] == "Attack"
    )

    benign_detected = total_traffic - attacks_detected

    return {
        "total_traffic": total_traffic,
        "attacks_detected": attacks_detected,
        "threats_blocked": 0,
        "detection_rate": round(
            attacks_detected / total_traffic * 100, 2
        ) if total_traffic else 0,
        "mode": "simulated" if total_traffic else "waiting",
        "message": (
            "Showing recorded model predictions"
            if total_traffic
            else "Waiting for traffic data"
        ),
        "benign_detected": benign_detected,
    }



@app.get("/api/alerts")
def get_alerts():
    return {
        "alerts": [
            {
                "ip": "192.168.1.105",
                "type": "UDP Flood",
                "severity": "Critical",
                "time": "Demo"
            },
            {
                "ip": "10.0.0.24",
                "type": "SYN Flood",
                "severity": "High",
                "time": "Demo"
            },
            {
                "ip": "172.16.0.18",
                "type": "HTTP Flood",
                "severity": "Medium",
                "time": "Demo"
            }
        ],
        "mode": "demo"
    }


@app.get("/api/health")
def health():
    return {"status": "healthy"}

from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    features: List[float]

@app.post("/api/predict")
def predict(data: dict):
    import numpy as np
    import pandas as pd
    import joblib
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "real_model_v2.joblib"
    features_path = base_dir / "real_model_features_v2.joblib"

    if not model_path.exists() or not features_path.exists():
        raise HTTPException(
            status_code=503,
            detail="V2 model or feature list not found",
        )

    pipeline = joblib.load(model_path)
    feature_names = joblib.load(features_path)

    values = data.get("features")

    if not isinstance(values, list) or len(values) != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Provide exactly {len(feature_names)} numeric features in a 'features' list",
        )

    try:
        row = np.array(values, dtype=float)

        if not np.isfinite(row).all():
            raise ValueError("Features must be finite numbers")

        row = pd.DataFrame([row], columns=feature_names)

    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Features must be finite numeric values",
        )

    prediction = int(pipeline.predict(row)[0])
    probabilities = pipeline.predict_proba(row)[0]

    classes = list(pipeline.classes_)
    attack_index = classes.index(1) if 1 in classes else None

    risk = (
        float(probabilities[attack_index] * 100)
        if attack_index is not None
        else 0.0
    )

    severity = (
        "Critical" if risk >= 75
        else "High" if risk >= 50
        else "Medium" if risk >= 25
        else "Low"
    )

    record_detection(
        prediction="Attack" if prediction == 1 else "Benign",
        risk_score=risk,
        severity=severity,
        source="V2-model API",
    )

    return {
        "prediction": "Attack" if prediction == 1 else "Benign",
        "risk_score": round(risk, 2),
        "severity": severity,
        "model": "Random Forest V2 Pipeline",
        "features_required": len(feature_names),
    }



@app.post("/api/risk-score")
def calculate_risk_score(request: PredictionRequest):
    import joblib
    import numpy as np
    import pandas as pd
    from pathlib import Path

    features = request.features

    if not features:
        return {
            "risk_score": 0,
            "severity": "Unknown",
            "note": "No features supplied."
        }

    base_dir = Path(__file__).resolve().parent
    model = joblib.load(base_dir /  "real_model_v2.joblib")
    feature_names = joblib.load(
        base_dir / "real_model_features_v2.joblib"
    )
    if len(feature_names) != 82 or len(set(feature_names)) != 82:
        raise HTTPException(
            status_code=500,
            detail="Saved model feature list must contain 82 unique features.",
    )

    if len(features) != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_names)} features, "
                   f"received {len(features)}."
        )

    values = np.asarray(features, dtype=float)

    if not np.isfinite(values).all():
        raise HTTPException(
            status_code=400,
            detail="Features must contain only finite numeric values."
        )

    X = pd.DataFrame([values], columns=feature_names)

    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)

    if 1 not in classes:
        raise HTTPException(
            status_code=500,
            detail="The model does not contain the expected attack class."
        )

    attack_probability = float(probabilities[classes.index(1)])
    risk_score = round(attack_probability * 100)

    if risk_score >= 75:
        severity = "Critical"
    elif risk_score >= 50:
        severity = "High"
    elif risk_score >= 25:
        severity = "Medium"
    else:
        severity = "Low"

    return {
        "risk_score": risk_score,
        "attack_probability": round(attack_probability, 4),
        "predicted_class": "Attack" if attack_probability >= 0.5 else "Benign",
        "severity": severity,
        "note": (
            "Model-estimated attack probability expressed as a score "
            "from 0 to 100. Severity thresholds are provisional, "
            "not independently validated."
        )
    }



@app.post("/api/dataset/preview")
async def preview_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    content = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(content), nrows=100)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the CSV file."
        )

    return {
        "filename": file.filename,
        "rows_previewed": len(df),
        "columns": df.columns.tolist(),
        "preview": df.head(5).fillna("").to_dict(orient="records")
    }

@app.post("/api/model/train-sample")
def train_sample_model():
    dataset_path = Path(__file__).resolve().parent.parent / "dataset" / "sample_traffic.csv"

    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail="sample_traffic.csv not found in dataset folder"
        )

    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="Dataset must contain a Label column"
        )

    X = df.drop(columns=["Label"])
    y = (df["Label"].astype(str).str.upper() != "BENIGN").astype(int)

    X = X.apply(pd.to_numeric, errors="coerce")
    base_dir = Path(__file__).resolve().parent
    medians = joblib.load(base_dir.parent / "ml" / "recovered_medians.joblib")
    X = X.replace([float("inf"), float("-inf")], float("nan"))
    X = X.fillna(pd.Series(medians)).fillna(0)

    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least two records from each class for training and testing"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test,y_pred,
    output_dict=True,zero_division=0)

    model_path = Path(__file__).resolve().parent / "sample_model.joblib"
    joblib.dump(
        {"model": model, "features": X.columns.tolist()},
        model_path
    )

    return {
    "status": "success",
    "model": "Random Forest",
    "training_records": len(X_train),
    "testing_records": len(X_test),
    "accuracy": round(accuracy * 100, 2),
    "confusion_matrix": matrix.tolist(),
    "precision": round(report["1"]["precision"] * 100, 2),
    "recall": round(report["1"]["recall"] * 100, 2),
    "f1_score": round(report["1"]["f1-score"] * 100, 2),
    "warning": "Tiny synthetic sample only; metrics are not meaningful for real-world performance."
}


@app.get("/api/model/status")
def model_status():
    base_dir = Path(__file__).resolve().parent

    model_path = base_dir / "real_model_v2.joblib"
    features_path = base_dir / "real_model_features_v2.joblib"

    model_exists = model_path.exists() and features_path.exists()

    return {
        "model_trained": model_exists,
        "model_name": "Random Forest",
        "dataset": "CIC-DDoS2019",
        "status": "Real model available" if model_exists else "Real model not found"
    }


@app.get("/api/model/explain")
def explain_model():
    base_dir = Path(__file__).resolve().parent

    model_path = base_dir / "real_model_v2.joblib"
    features_path = base_dir / "real_model_features_v2.joblib"
    dataset_path = (
        base_dir.parent / "dataset" / "01-12" / "UDPLag.csv"
    )

    if not model_path.exists() or not features_path.exists():
        raise HTTPException(
            status_code=503,
            detail="V2 model or feature list not found",
        )

    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail="UDPLag dataset not found",
        )

    pipeline = joblib.load(model_path)
    features = joblib.load(features_path)

    sample = pd.read_csv(dataset_path, nrows=200, low_memory=False)
    sample.columns = sample.columns.str.strip()

    missing = [feature for feature in features if feature not in sample.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset is missing required features: {missing}",
        )

    X = sample.reindex(columns=features)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([float("inf"), float("-inf")], float("nan"))

    # Use the imputer fitted inside the V2 pipeline.
    imputer = pipeline.named_steps["imputer"]
    model = pipeline.named_steps["model"]
    X_imputed = imputer.transform(X.iloc[[0]])

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_imputed)

        if isinstance(shap_values, list):
            values = (
                shap_values[1]
                if len(shap_values) > 1
                else shap_values[0]
            )
        elif getattr(shap_values, "ndim", 0) == 3:
            values = shap_values[:, :, 1][0]
        else:
            values = shap_values[0]

        values = np.asarray(values).reshape(-1)

        importance = [
            {
                "feature": name,
                "impact": round(float(value), 6),
            }
            for name, value in zip(features, values)
        ]

        importance.sort(
            key=lambda item: abs(item["impact"]),
            reverse=True,
        )

        return {
            "method": "SHAP",
            "model": "Random Forest V2 Pipeline",
            "explained_class": "Attack",
            "explanation": importance[:15],
            "note": (
                "SHAP explanation for one UDPLag sample, "
                "using the V2 model and its fitted imputer."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SHAP explanation failed: {str(exc)}",
        )


@app.get("/api/model/feature-importance")
def feature_importance():
    
    base_dir = Path(__file__).resolve().parent

    model_path = base_dir / "real_model_v2.joblib"
    features_path = base_dir / "real_model_features_v2.joblib"

    pipeline = joblib.load(model_path)
    features = joblib.load(features_path)

    model = pipeline.named_steps["model"]

    importance = sorted(
        [
            {"feature": name, "importance": float(value)}
            for name, value in zip(
                features, model.feature_importances_
            )
        ],
    key=lambda item: item["importance"],
    reverse=True
    )


    return {
        "feature_importance": importance,
        "model": "Trained Random Forest",
        "note": "Feature importance from the trained CIC-DDoS2019 model."
    }


@app.get("/api/detection-history")
def get_detection_history():
    with detection_history_lock:
        history = list(detection_records)

    return {
        "history": history,
        "count": len(history),
        "note": "Recent predictions recorded during this backend session."
    }


@app.get("/api/system/summary")
def system_summary():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "real_model_v2.joblib"
    features_path = base_dir / "real_model_features_v2.joblib"

    model_exists = model_path.exists() and features_path.exists()

    return {
        "api_status": "healthy",
        "model_trained": model_exists,
        "model_type": "Random Forest",
        "dataset": "CIC-DDoS2019",
        "mode": "real_model" if model_exists else "unavailable"
    }


@app.get("/api/system/metrics")
def system_metrics():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "real_model_v2.joblib"
    features_path = base_dir / "real_model_features_v2.joblib"

    model_exists = model_path.exists() and features_path.exists()

    return {
        "api_status": "healthy",
        "model": "Random Forest" if model_exists else "Unavailable",
        "dataset": "CIC-DDoS2019" if model_exists else "Unavailable",
        "mode": "real_model" if model_exists else "unavailable"
    }


@app.get("/api/system/time")
def system_time():
    return {
        "server_time": datetime.now(timezone.utc).isoformat(),
        "status": "online"
    }
@app.get("/api/project/info")
def project_info():
    return {
        "title": "Intelligent Machine Learning-Based Intrusion Detection System for DDoS Attacks in IoT Networks",
        "dataset": "CIC-DDoS2019",
        "algorithm": "Random Forest",
        "explainability": "SHAP",
        "frontend": "React",
        "backend": "FastAPI",
        "status": "Development"
    }


@app.post("/api/dataset/analyze")
async def analyze_dataset(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, low_memory=False)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain a Label column"
        )

    labels = df["Label"].astype(str).str.strip().str.upper()
    benign_count = int((labels == "BENIGN").sum())
    total = len(df)
    attack_count = total - benign_count

    return {
        "filename": file.filename,
        "total_traffic": total,
        "attacks_detected": attack_count,
        "benign_traffic": benign_count,
        "attack_types": {
            str(label): int(count)
            for label, count in labels[labels != "BENIGN"].value_counts().items()
        },
        "mode": "CSV analysis"
    }


@app.post("/api/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    import joblib
    import pandas as pd
    import numpy as np
    from pathlib import Path

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    base_dir = Path(__file__).resolve().parent
    model = joblib.load(base_dir / "real_model_v2.joblib")
    feature_names = joblib.load(base_dir /  "real_model_features_v2.joblib")

    total_rows = 0
    predicted_attacks = 0
    predicted_benign = 0
    actual_counts = {0: 0, 1: 0}
    predicted_counts = {0: 0, 1: 0}
    cm = np.zeros((2, 2), dtype=np.int64)

    for chunk in pd.read_csv(file.file, chunksize=10000, low_memory=False):
        chunk.columns = chunk.columns.str.strip()

        missing = [name for name in feature_names if name not in chunk.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail={"message": "CSV missing model features", "missing": missing},
            )

        X = chunk[feature_names].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)

        if X.isna().all(axis=1).any():
            raise HTTPException(
            status_code=400,
            detail="CSV contains rows with no valid numeric model features.",
        )


        predictions = model.predict(X).astype(int)
        total_rows += len(chunk)
        predicted_attacks += int((predictions == 1).sum())
        predicted_benign += int((predictions == 0).sum())

        if "Label" in chunk.columns:
            labels = chunk["Label"].astype("string").str.strip().str.upper()

            if labels.isna().any() or labels.eq("").any():
                raise HTTPException(
                    status_code=400,
                    detail="CSV contains missing or empty Label values.",
                )

            actual = labels.ne("BENIGN").astype(int).to_numpy()

            cm += confusion_matrix(actual, predictions, labels=[0, 1])
            actual_counts[0] += int((actual == 0).sum())
            actual_counts[1] += int((actual == 1).sum())
            predicted_counts[0] += int((predictions == 0).sum())
            predicted_counts[1] += int((predictions == 1).sum())

    result = {
        "filename": file.filename,
        "total_rows": total_rows,
        "predicted_attacks": predicted_attacks,
        "predicted_benign": predicted_benign,
        "model": "Random Forest",
        "note": "Predictions are model estimates.",
    }

    if actual_counts[0] + actual_counts[1] > 0:
        tn, fp, fn, tp = cm.ravel()
        result.update({
            "accuracy": round((tn + tp) / cm.sum(), 6),
            "precision": round(tp / (tp + fp), 6) if tp + fp else 0,
            "recall": round(tp / (tp + fn), 6) if tp + fn else 0,
            "f1_score": round(
                2 * tp / (2 * tp + fp + fn), 6
            ) if 2 * tp + fp + fn else 0,
            
            "confusion_matrix": cm.tolist(),
            
        })

    return result


@app.post("/api/stream/start")
def start_stream(request: PredictionRequest):
    import joblib
    import numpy as np
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent
    model = joblib.load(base_dir / "real_model_v2.joblib")
    feature_names = joblib.load(
        base_dir / "real_model_features_v2.joblib"
    )

    if len(feature_names) != 82 or len(set(feature_names)) != 82:
        raise HTTPException(
            status_code=500,
            detail="Model feature configuration is invalid.",
        )

    if len(request.features) != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_names)} features.",
        )

    values = np.asarray(request.features, dtype=float)

    if not np.isfinite(values).all():
        raise HTTPException(
            status_code=400,
            detail="Features must be finite numeric values.",
        )

    row = pd.DataFrame([values], columns=feature_names)
    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)

    if 1 not in classes:
        raise HTTPException(
            status_code=500,
            detail="Attack class is missing from the model.",
        )

    risk = float(probabilities[classes.index(1)] * 100)

    if risk >= 75:
        severity = "Critical"
    elif risk >= 50:
        severity = "High"
    elif risk >= 25:
        severity = "Medium"
    else:
        severity = "Low"

    label = "Attack" if prediction == 1 else "Benign"

    record_detection(
        prediction=label,
        risk_score=risk,
        severity=severity,
        source="Simulated stream",
    )

    return {
        "status": "processed",
        "mode": "simulated",
        "prediction": label,
        "risk_score": round(risk, 2),
        "severity": severity,
        "source": "Simulated stream",
    }


@app.post("/api/stream/csv")
async def stream_csv(file: UploadFile = File(...)):
    import joblib
    import pandas as pd
    import numpy as np
    from pathlib import Path

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    base_dir = Path(__file__).resolve().parent
    model = joblib.load(base_dir / "real_model_v2.joblib")
    feature_names = joblib.load(
        base_dir / "real_model_features_v2.joblib"
    )

    if len(feature_names) != 82 or len(set(feature_names)) != 82:
        raise HTTPException(
            status_code=500,
            detail="Model feature configuration is invalid.",
        )

    total_rows = 0
    attacks = 0
    benign = 0
    batches = []

    try:
        for chunk in pd.read_csv(
            file.file, chunksize=1000, low_memory=False
        ):
            chunk.columns = chunk.columns.str.strip()

            missing = [
                name for name in feature_names
                if name not in chunk.columns
            ]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "CSV is missing model features.",
                        "missing": missing,
                    },
                )

            X = chunk[feature_names].apply(
                pd.to_numeric, errors="coerce"
            )
            X = X.replace([np.inf, -np.inf], np.nan)

            if X.isna().all(axis=1).any():
                raise HTTPException(
                    status_code=400,
                    detail="CSV contains rows with no valid numeric features.",
                )

            predictions = model.predict(X).astype(int)

            batch_attacks = int((predictions == 1).sum())
            batch_benign = int((predictions == 0).sum())

            total_rows += len(chunk)
            attacks += batch_attacks
            benign += batch_benign

            batches.append({
                "batch": len(batches) + 1,
                "rows_processed": total_rows,
                "batch_attacks": batch_attacks,
                "batch_benign": batch_benign,
            })

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to process CSV: {error}",
        )

    return {
        "status": "completed",
        "mode": "CSV batch simulation",
        "filename": file.filename,
        "total_rows": total_rows,
        "attacks_detected": attacks,
        "benign_traffic": benign,
        "batch_size": 1000,
        "batches": batches,
        "note": "CSV records were processed in batches; this is not live packet capture.",
    }
