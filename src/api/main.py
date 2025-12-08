"""
FastAPI service for real-time Human Activity Recognition predictions.

Exposes two endpoints:
- `/predict`: returns predicted activity class IDs and labels
- `/predict_proba`: returns probability distributions over all classes

Configuration and model artifacts (SVM, scaler, labels) are loaded from
paths specified in `config.yaml` at startup.
"""

from pathlib import Path
from typing import List, Dict

import joblib
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ────────────────────────────────────────────────────────────────────────────────
# Configuration & Artifact Loading
# ────────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
if not CONFIG_PATH.exists():
    raise RuntimeError(f"Config file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

MODELS_DIR = PROJECT_ROOT / cfg["models_dir"]
SVM_MODEL_PATH = MODELS_DIR / cfg["svm_model_filename"]
SCALER_PATH = MODELS_DIR / cfg["scaler_filename"]
LABELS_PATH = MODELS_DIR / cfg["labels_filename"]
N_FEATURES = int(cfg["n_features"])


def _load_artifacts():
    """Load SVM model, scaler, and label mappings from disk."""
    print(f"Loading model from {SVM_MODEL_PATH}")
    if not SVM_MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {SVM_MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise RuntimeError(f"Scaler file not found: {SCALER_PATH}")
    if not LABELS_PATH.exists():
        raise RuntimeError(f"Labels file not found: {LABELS_PATH}")

    _model = joblib.load(SVM_MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)
    _labels: Dict[int, str] = joblib.load(LABELS_PATH)
    return _model, _scaler, _labels


model, scaler, activity_labels = _load_artifacts()
model_classes = model.classes_
class_labels_ordered = [activity_labels[int(c)] for c in model_classes]


# ────────────────────────────────────────────────────────────────────────────────
# Request/Response Schemas
# ────────────────────────────────────────────────────────────────────────────────


class Instance(BaseModel):
    """Single observation with 561 sensor features."""
    features: List[float]


class PredictRequest(BaseModel):
    """Batch of instances to classify."""
    instances: List[Instance]


class PredictResponse(BaseModel):
    """Predicted class IDs and human-readable labels."""
    predictions: List[int]
    labels: List[str]


class PredictProbaResponse(BaseModel):
    """Probability distribution over all activity classes per instance."""
    probabilities: List[List[float]]
    class_ids: List[int]
    class_labels: List[str]


# ────────────────────────────────────────────────────────────────────────────────
# FastAPI Application
# ────────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=cfg.get("project_name", "HAR SVM API"),
    description=cfg.get(
        "description",
        "API de prédiction d'activité humaine (UCI HAR) avec SVM RBF"
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    """Health check and usage information endpoint."""
    # Endpoint de santé / info.
    return {
        "message": "HAR SVM API is running",
        "n_features_expected": N_FEATURES,
        "usage": "POST /predict ou /predict_proba avec un JSON contenant instances: [{features: [...]}, ...]",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """Classify one or more instances and return activity labels.
    
    Expects a batch of feature vectors, each with 561 values.
    Returns predicted class IDs and corresponding activity names.
    """
    if len(payload.instances) == 0:
        raise HTTPException(status_code=400, detail="No instances provided")

    # (n_samples, n_features)
    X = np.array([inst.features for inst in payload.instances], dtype=float)

    if X.shape[1] != N_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Each instance must have {N_FEATURES} features, got {X.shape[1]}",
        )

    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    labels = [activity_labels[int(c)] for c in preds]

    return PredictResponse(
        predictions=[int(c) for c in preds],
        labels=labels,
    )


@app.post("/predict_proba", response_model=PredictProbaResponse)
def predict_proba(payload: PredictRequest):
    """Return class probability distributions for each instance.
    
    Useful for confidence scoring and uncertainty estimation.
    Output includes probabilities, class IDs, and ordered class labels.
    """
    if len(payload.instances) == 0:
        raise HTTPException(status_code=400, detail="No instances provided")

    if not hasattr(model, "predict_proba"):
        raise HTTPException(
            status_code=500,
            detail="This model does not support predict_proba",
        )

    X = np.array([inst.features for inst in payload.instances], dtype=float)

    if X.shape[1] != N_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Each instance must have {N_FEATURES} features, got {X.shape[1]}",
        )

    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled) 

    return PredictProbaResponse(
        probabilities=proba.tolist(),
        class_ids=[int(c) for c in model_classes],
        class_labels=class_labels_ordered,
    )
