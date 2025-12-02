# src/explianability/LIME.py

from pathlib import Path

import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def build_lime_explainer(X_train: np.ndarray, feature_names, class_names):
    """
    Construit un LimeTabularExplainer pour les features tabulaires HAR.
    """
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True,
        verbose=False,
        mode="classification",
    )
    return explainer


def explain_instance(explainer, model, x, num_features: int = 10, show: bool = True):
    """
    Explique une seule instance x (1D array) avec LIME.
    """
    exp = explainer.explain_instance(
        x,
        model.predict_proba,
        num_features=num_features,
    )
    if show:
        exp.show_in_notebook(show_table=True)
    return exp
