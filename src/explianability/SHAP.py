# src/explianability/SHAP.py

from pathlib import Path

import numpy as np
import shap
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def build_shap_explainer(model, X_background: np.ndarray):
    """
    Construit un explainer SHAP.
    - Si le modèle est un arbre (RandomForest, XGBoost, etc.) : TreeExplainer.
    - Sinon : KernelExplainer (plus lent).
    """
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, X_background)
    return explainer


def compute_shap_values(explainer, X_sample: np.ndarray):
    """
    Calcule les valeurs SHAP sur quelques exemples.
    """
    shap_values = explainer.shap_values(X_sample)
    return shap_values


def plot_summary(explainer, shap_values, X_sample, feature_names):
    """
    Affiche un summary plot SHAP (importance globale des features).
    À utiliser dans un notebook (backend graphique).
    """
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names)
