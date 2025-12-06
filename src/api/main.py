from pathlib import Path
from typing import List, Dict

import joblib
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel



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



print(f"Loading model from {SVM_MODEL_PATH}")
if not SVM_MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found: {SVM_MODEL_PATH}")

if not SCALER_PATH.exists():
    raise RuntimeError(f"Scaler file not found: {SCALER_PATH}")

if not LABELS_PATH.exists():
    raise RuntimeError(f"Labels file not found: {LABELS_PATH}")

model = joblib.load(SVM_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
activity_labels: Dict[int, str] = joblib.load(LABELS_PATH)

model_classes = model.classes_
class_labels_ordered = [activity_labels[int(c)] for c in model_classes]



class Instance(BaseModel):
    features: List[float]


class PredictRequest(BaseModel):
    instances: List[Instance]


class PredictResponse(BaseModel):
    predictions: List[int]
    labels: List[str]


class PredictProbaResponse(BaseModel):
    probabilities: List[List[float]]  
    class_ids: List[int]
    class_labels: List[str]


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
    """
    Endpoint de santé / info.
    """
    return {
        "message": "HAR SVM API is running",
        "n_features_expected": N_FEATURES,
        "usage": "POST /predict ou /predict_proba avec un JSON contenant instances: [{features: [...]}, ...]",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Prédiction de la classe pour une ou plusieurs instances.
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
    """
    Retourne la distribution de probabilité sur les classes pour chaque instance.
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
