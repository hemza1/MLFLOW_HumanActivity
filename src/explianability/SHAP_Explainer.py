"""
Run a SHAP analysis using a surrogate RandomForest on processed features.

Assumptions:
- Processed features and labels are stored in `data/processed/X_full.joblib` and `y_full.joblib`.
- Outputs (summary plot + force plot) are written to `results/`.
"""

from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> Tuple[np.ndarray, np.ndarray, list]:
    """Load processed features/labels and build feature names."""
    print("\nLoading X_full.joblib and y_full.joblib...")
    X_full = joblib.load(PROCESSED_DIR / "X_full.joblib")
    y_full = joblib.load(PROCESSED_DIR / "y_full.joblib")

    X_full = np.asarray(X_full)
    y_full = np.asarray(y_full)

    print("✔ Loaded:", X_full.shape, y_full.shape)
    feature_names = [f"feat_{i}" for i in range(X_full.shape[1])]
    return X_full, y_full, feature_names


def train_surrogate(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Fit a small RandomForest surrogate used by SHAP."""
    print("\nTraining RandomForest surrogate model for SHAP...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X, y)
    print("✔ RandomForest trained!")
    return rf


def select_sample(X: np.ndarray, y: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Take a stratified train/test split and return a capped test sample."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print("✔ Train shape:", X_train.shape, "Test shape:", X_test.shape)

    n_samples = min(n_samples, X_test.shape[0])
    return X_test[:n_samples], y_test[:n_samples]


def compute_shap_values(explainer: shap.TreeExplainer, X_sample: np.ndarray) -> np.ndarray:
    """Compute SHAP matrix and normalize shape across shap versions."""
    print(f"\nComputing SHAP values on {X_sample.shape[0]} samples...")
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        sv = shap_values[0]
    else:
        sv = shap_values

    if sv.ndim == 3:
        sv = sv[:, :, 0]

    if sv.shape != X_sample.shape:
        raise ValueError(f"Mismatch: shap_values {sv.shape} vs X_sample {X_sample.shape}")

    print("✔ SHAP values computed!")
    return sv


def save_summary_plot(sv: np.ndarray, X_sample: np.ndarray, feature_names: list, out_path: Path) -> None:
    """Persist SHAP summary plot to disk."""
    plt.figure(figsize=(12, 6))
    shap.summary_plot(
        sv,
        X_sample,
        feature_names=feature_names,
        show=False,
    )
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"✔ SHAP summary saved at: {out_path}")


def save_force_plot(explainer: shap.TreeExplainer, sv: np.ndarray, feature_names: list, idx: int, out_path: Path) -> None:
    """Persist a single-instance force plot (best-effort)."""
    try:
        exp_val = explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            exp_val = exp_val[0]

        shap.save_html(
            str(out_path),
            shap.force_plot(
                exp_val,
                sv[idx],
                feature_names=feature_names,
            ),
        )
        print(f"✔ SHAP force plot saved at: {out_path}")
    except Exception as exc:  # pragma: no cover - visualization best-effort
        print("Unable to render force plot:", exc)


def main(idx: int = 0, n_samples: int = 100) -> None:
    print(">>> SHAP RUN (SURROGATE RF MODEL)")
    print("PROJECT_ROOT =", PROJECT_ROOT)

    X_full, y_full, feature_names = load_data()
    X_sample, _ = select_sample(X_full, y_full, n_samples=n_samples)

    rf = train_surrogate(X_full, y_full)
    print("\nBuilding SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(rf)
    print("✔ TreeExplainer ready!")

    sv = compute_shap_values(explainer, X_sample)
    print("Final SHAP matrix shape:", sv.shape)

    summary_png = RESULTS_DIR / "shap_summary_rf.png"
    save_summary_plot(sv, X_sample, feature_names, summary_png)

    force_html = RESULTS_DIR / f"shap_force_rf_{idx}.html"
    save_force_plot(explainer, sv, feature_names, idx, force_html)

    print("\nSHAP with surrogate RandomForest completed successfully!")


if __name__ == "__main__":
    main(idx=0, n_samples=100)
