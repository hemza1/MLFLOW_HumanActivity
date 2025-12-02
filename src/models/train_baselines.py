# src/models/train_baselines.py

from pathlib import Path

import joblib
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

import mlflow
import mlflow.sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def load_processed():
    """
    Charge X_full et y_full depuis data/processed.
    """
    X_full = joblib.load(PROCESSED_DIR / "X_full.joblib")
    y_full = joblib.load(PROCESSED_DIR / "y_full.joblib")
    return X_full, y_full


def get_baseline_models(random_state: int = 42):
    """
    Définit les modèles de base + grilles d'hyperparamètres.
    """
    models = {}

    # Logistic Regression (multiclasse)
    logreg_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    n_jobs=-1,
                    multi_class="multinomial",
                    random_state=random_state,
                ),
            ),
        ]
    )
    logreg_grid = {
        "clf__C": [0.1, 1.0, 10.0],
    }

    # Random Forest
    rf_pipe = Pipeline(
        [
            # on peut mettre un scaler ou non ; ici on ne le met pas
            ("clf", RandomForestClassifier(random_state=random_state, n_jobs=-1)),
        ]
    )
    rf_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 20, 40],
    }

    # SVM RBF
    svm_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=random_state)),
        ]
    )
    svm_grid = {
        "clf__C": [1.0, 10.0],
        "clf__gamma": ["scale", "auto"],
    }

    # KNN
    knn_pipe = Pipeline(
        [
          ("scaler", StandardScaler()),
          ("clf", KNeighborsClassifier()),
        ]
    )
    knn_grid = {
        "clf__n_neighbors": [5, 11],
    }

    models["logreg"] = {"estimator": logreg_pipe, "param_grid": logreg_grid}
    models["rf"] = {"estimator": rf_pipe, "param_grid": rf_grid}
    models["svm_rbf"] = {"estimator": svm_pipe, "param_grid": svm_grid}
    models["knn"] = {"estimator": knn_pipe, "param_grid": knn_grid}

    return models


def setup_mlflow():
    """
    Configure MLflow avec un dossier local mlruns.
    """
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    tracking_uri = MLRUNS_DIR.as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("HAR_baselines")
    print("MLflow tracking URI =", tracking_uri)


def run_grid_search(X_train, y_train, X_test, y_test, cv: int = 3, n_jobs: int = -1):
    """
    Lance une GridSearchCV pour chaque modèle.
    Log tout dans MLflow et retourne un dict de résultats.
    """
    setup_mlflow()
    models = get_baseline_models()
    results = {}

    for name, cfg in models.items():
        estimator = cfg["estimator"]
        param_grid = cfg["param_grid"]

        print(f"\n=== Model : {name} ===")

        with mlflow.start_run(run_name=name):
            grid = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=cv,
                n_jobs=n_jobs,
                scoring="accuracy",
            )
            grid.fit(X_train, y_train)

            best_est = grid.best_estimator_
            y_pred = best_est.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            print("Best params :", grid.best_params_)
            print(f"CV accuracy : {grid.best_score_:.4f}")
            print(f"Test accuracy : {acc:.4f}")
            print("Classification report :\n", classification_report(y_test, y_pred))

            # Log MLflow
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("cv_accuracy", grid.best_score_)
            mlflow.log_metric("test_accuracy", acc)
            mlflow.sklearn.log_model(best_est, artifact_path="model")

            results[name] = {
                "grid": grid,
                "best_estimator": best_est,
                "best_params": grid.best_params_,
                "cv_score": grid.best_score_,
                "test_accuracy": acc,
            }

    return results


def train_baselines(test_size: float = 0.2, random_state: int = 42):
    """
    Pipeline complet :
    - charge X_full / y_full
    - split train/test
    - lance la grid search pour tous les modèles
    - retourne les résultats + le nom du meilleur modèle
    """
    X_full, y_full = load_processed()

    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y_full,
        test_size=test_size,
        random_state=random_state,
        stratify=y_full,
    )

    results = run_grid_search(X_train, y_train, X_test, y_test)

    best_name = max(results, key=lambda m: results[m]["test_accuracy"])
    best = results[best_name]
    print("\n=== BEST MODEL ===")
    print("Best model :", best_name)
    print("Best params :", best["best_params"])
    print("CV accuracy :", best["cv_score"])
    print("Test accuracy :", best["test_accuracy"])

    # Option : sauvegarder le meilleur modèle "global" en local
    best_model_path = PROJECT_ROOT / "models" / f"{best_name}_best.joblib"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["best_estimator"], best_model_path)
    print("Meilleur modèle sauvegardé dans :", best_model_path)

    return results, best_name


if __name__ == "__main__":
    train_baselines()
