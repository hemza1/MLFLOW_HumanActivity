"""
Generate a LIME explanation for a saved classifier and one sample.

Assumptions:
- Processed features and labels are stored as Joblib files in `data/processed`.
- A fitted model is stored in `models/*_best.joblib` (first match is used).
- Outputs (HTML + PNG) are written to `results/`.
"""

from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from lime.lime_tabular import LimeTabularExplainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> Tuple[np.ndarray, np.ndarray, list, list]:
    """Load processed features/labels and build feature & class names."""
    print("\nLoading X_full.joblib and y_full.joblib...")
    X_full = joblib.load(PROCESSED_DIR / "X_full.joblib")
    y_full = joblib.load(PROCESSED_DIR / "y_full.joblib")

    X_full = np.asarray(X_full)
    y_full = np.asarray(y_full)

    print("X_full :", X_full.shape)
    print("y_full :", y_full.shape)

    feature_names = [f"feat_{i}" for i in range(X_full.shape[1])]
    class_names = sorted(np.unique(y_full).tolist())
    print("Classes =", class_names)
    return X_full, y_full, feature_names, class_names


def load_best_model():
    """Load the first *_best.joblib model found in models directory."""
    print("\nSearching for *_best.joblib in /models ...")
    best_models = list(MODELS_DIR.glob("*_best.joblib"))
    if not best_models:
        raise FileNotFoundError("❌ No *_best.joblib found in /models")
    best_model_path = best_models[0]
    print("✔ Best model found:", best_model_path.name)
    model = joblib.load(best_model_path)
    print("✔ Model loaded!")
    return model, best_model_path.name


def explain_instance(model, explainer, X_full: np.ndarray, y_full: np.ndarray, idx: int = 0, num_features: int = 10):
    """Run LIME explanation for one sample and return the explanation object."""
    x = X_full[idx]
    print(f"\nExplaining instance idx={idx}, true label={y_full[idx]}")
    exp = explainer.explain_instance(
        data_row=x,
        predict_fn=model.predict_proba,
        num_features=num_features,
    )
    print("\nTop features (LIME):")
    for feat, weight in exp.as_list():
        print(f"{feat}: {weight:.4f}")
    return exp


def save_outputs(exp, idx: int):
    """Save HTML and PNG outputs for a LIME explanation."""
    out_html = RESULTS_DIR / f"lime_explanation_{idx}.html"
    exp.save_to_file(str(out_html))
    print(f"\n✔ LIME HTML saved at: {out_html}")

    fig = exp.as_pyplot_figure()
    png_path = RESULTS_DIR / f"lime_summary_{idx}.png"
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ LIME PNG saved at: {png_path}")


def main(idx: int = 0, num_features: int = 10):
    print(">>> LIME RUN")
    print("PROJECT_ROOT =", PROJECT_ROOT)

    X_full, y_full, feature_names, class_names = load_data()
    model, model_name = load_best_model()

    print("\nBuilding LimeTabularExplainer...")
    explainer = LimeTabularExplainer(
        training_data=X_full,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True,
        mode="classification",
    )
    print("✔ LIME explainer ready!")

    exp = explain_instance(model, explainer, X_full, y_full, idx=idx, num_features=num_features)
    save_outputs(exp, idx)


if __name__ == "__main__":
    main(idx=0, num_features=10)
