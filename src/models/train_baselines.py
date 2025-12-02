Ilyas
ilyas03835
Invisible

Ilyas — 11/26/2025 9:44 PM
Attachment file type: archive
TD_Collaborative_Filtering.rar
19.36 KB
Attachment file type: unknown
Untitled18.ipynb
85.75 KB
Ilyas
 started a call that lasted 2 hours. — 11/26/2025 10:30 PM
Genosis — 11/26/2025 10:31 PM
2 secondes
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
Genosis — 11/26/2025 10:39 PM
Attachment file type: unknown
data_exploration.ipynb
1.20 MB
Attachment file type: unknown
data_explainability.ipynb
6.47 MB
Genosis — 11/26/2025 10:49 PM
Attachment file type: unknown
data_experiments.ipynb
238.24 KB
Ilyas — 11/26/2025 11:11 PM
https://etulab.univ-amu.fr/m23022217/projet-final/-/blob/main/Notebook/Untitled18.ipynb?ref_type=heads
Genosis — 11/27/2025 12:02 AM
Image
Genosis — 11/27/2025 12:14 AM
Image
You missed a call from 
Genosis
 that lasted 3 minutes. — 11/28/2025 7:15 PM
Genosis — 11/28/2025 7:27 PM
yalaha
Ilyas
 started a call that lasted 2 hours. — 11/28/2025 7:32 PM
Ilyas — 11/28/2025 7:33 PM
ban ban
ban ban
Ilyas — 11/28/2025 7:53 PM
//////////////
best_model_name = max(results, key=lambda m: results[m]["accuracy"])
print(f"Meilleur modèle d'après l'accuracy : {best_model_name}")

name_to_model = {
    "LogReg": best_log_reg,
    "RandomForest": best_rf,
    "SVM_RBF": best_svm,
    "MLP_Keras_Optimise": best_mlp,
}

best_model_obj = name_to_model[best_model_name]

best_model_path = DVC_MODELS_DIR / f"{best_model_name}_best.joblib"
joblib.dump(best_model_obj, best_model_path)
print(f"Modèle sauvegardé dans : {best_model_path}")
//////////////////////////////////////////////
hist_obj = getattr(bestmlp, "history", None)

    if histobj is None:
        print("ℹ️
 Pas d'attribut history sur best_mlp (version SciKeras).")
    else:
        if isinstance(hist_obj, dict):
            hist = hist_obj
        else:
            hist = getattr(hist_obj, "history", None)

        if hist is not None:
            if "loss" in hist:
                plt.figure()
                plt.plot(hist["loss"], label="loss")
                if "val_loss" in hist:
                    plt.plot(hist["val_loss"], label="val_loss")
                plt.title("Courbe de perte (MLP)")
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.legend()
                plt.tight_layout()
                plt.show()

            if "accuracy" in hist:
                plt.figure()
                plt.plot(hist["accuracy"], label="accuracy")
                if "val_accuracy" in hist:
                    plt.plot(hist["val_accuracy"], label="val_accuracy")
                plt.title("Courbe d'accuracy (MLP)")
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.tight_layout()
                plt.show()
///////////////////////////////////////////////////////////////////
## Conclusion & Justification du meilleur modèle

Dans ce projet, nous avons mis en place un pipeline complet d’apprentissage automatique pour la classification multi-classes du dataset UCI HAR (Human Activity Recognition). Les étapes réalisées incluent : exploration des données, prétraitement, sélection de variables (SelectKBest), entraînement de quatre modèles via GridSearchCV (3-fold) ainsi qu’une évaluation exhaustive (accuracy, AUC, classification report, matrices de confusion, ROC).

### Modèles comparés
Nous avons évalué :
Expand
message.txt
3 KB
Genosis — 11/28/2025 8:09 PM
glpat-mrpvWLiABU_ToiqtX6iXr286MQp1OjFubQk.01.0z050f4hi
Ilyas — 11/28/2025 8:12 PM
from pathlib import Path


DVC_DATA_PROCESSED = dataset_path / "data" / "processed"
DVC_DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
/////
n_features = X_train_fs.shape[1]
n_classes = len(unique_classes)

def build_mlp(
    n_features=n_features,
    n_classes=n_classes,
    n_hidden1=256,
    n_hidden2=128,
    dropout_rate=0.4,
):
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(n_hidden1, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        layers.Dense(n_hidden2, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

early_stop = EarlyStopping(
    monitor="loss",
    patience=3,
    restore_best_weights=True,
)

mlp_clf = KerasClassifier(
    model=build_mlp,
    n_features=n_features,
    n_classes=n_classes,
    epochs=30,
    batch_size=128,
    verbose=0,
    random_state=RANDOM_STATE,
    callbacks=[early_stop],
)

mlp_param_grid = {
    "modeln_hidden1": [256, 384],
    "modeln_hidden2": [128],
    "model__dropout_rate": [0.3, 0.5],
    "batch_size": [64, 128],
}

print("MLP Keras optimisé défini.")
/////  log_reg = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

log_reg_param_grid = {
    "clfC": [0.1, 1.0, 10.0],
}

rf_clf = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf_param_grid = {
    "max_depth": [None, 25],
    "min_samples_split": [2, 5],
}

svm_rbf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ]
)

svm_param_grid = {
    "clfC": [1.0, 10.0],
    "clf__gamma": ["scale"],
}

print("LogReg, RF, SVM définis.")
////
log_reg = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

log_reg_param_grid = {
    "clfC": [0.1, 1.0, 10.0],
}

rf_clf = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf_param_grid = {
    "max_depth": [None, 25],
    "min_samples_split": [2, 5],
}

svm_rbf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ]
)

svm_param_grid = {
    "clfC": [1.0, 10.0],
    "clf__gamma": ["scale"],
}

print("LogReg, RF, SVM définis.")
Attachment file type: unknown
Untitled18.ipynb
861.65 KB
Attachment file type: unknown
Untitled18 copy.ipynb
939.50 KB
Genosis — 11/28/2025 9:01 PM
https://chatgpt.com/share/6929f175-d40c-8006-888d-916573651f47
ChatGPT
ChatGPT - devoir communication
Shared via ChatGPT
Image
Ilyas — 11/28/2025 9:03 PM
Marche à suivre : vérifier et ajouter l’huile moteur

Type d’information : marche à suivre (phasée)

Phase A — Ouverture du capot

Type d’information : marche à suivre

Tableau — Étapes
Étape    Action
1    Déverrouillez le capot.
2    Placez-vous devant le véhicule.
3    Soulevez légèrement le capot.
4    Attrapez la poignée centrale.
5    Ouvrez complètement.
6    Déployez la béquille.
Phase B — Vérification du niveau d’huile
Étape    Action
1    Repérez la jauge.
2    Retirez-la.
3    Essuyez-la.
4    Replacez-la.
5    Retirez-la à nouveau.
6    Lisez le niveau.
7    Si zone B → refermez.
8    Si bas → passez à l’ajout.
Phase C — Ajout d’huile
Étape    Action
1    Retirez le bouchon.
2    Versez jusqu’à 0,5 L avec l’entonnoir.
3    Attendez une minute.
4    Vérifiez à nouveau.
5    Répétez jusqu’à atteindre la zone B.
6    Replacez la jauge.
7    Revissez le bouchon.
Phase D — Fermeture du capot
Étape    Action
1    Retirez la béquille.
2    Replacez la béquille.
3    Laissez tomber le capot de 20 cm.
4    Vérifiez la fermeture.
////////////
Cette marche à suivre décrit les étapes nécessaires pour vérifier puis ajouter l’huile moteur. Le processus est divisé en quatre phases : ouverture du capot, vérification du niveau, ajout d’huile si nécessaire et fermeture du capot. Les tableaux suivants détaillent chaque action à réaliser dans l’ordre exact.
Ilyas — 11/28/2025 9:10 PM
//////
Conditions après lecture du niveau

Type d’information : conditions

Après lecture, appliquez les règles suivantes :
– si le niveau est en zone B, refermez le capot ;
– si le niveau est insuffisant, passez à la Phase C (ajout d’huile).
/////////////
Après avoir lu le niveau indiqué sur la jauge, vous devez déterminer la suite de la procédure. Les règles ci-dessous précisent l’action à effectuer selon le résultat observé.
Ilyas — 11/28/2025 9:20 PM
/////
Avertissements de sécurité

Type d’information : avertissements

🟥
 ATTENTION : un niveau d’huile excessif peut endommager le moteur.
🟥
 ATTENTION : ne jamais rouler avec un voyant rouge allumé.
🟥
 ATTENTION : nettoyer immédiatement toute projection d’huile.
🟥
 ATTENTION : un capot mal fermé peut se rouvrir en roulant.
Ilyas — 11/28/2025 9:40 PM
🔴
 Avertissement (à créer)

Fond du bloc : #FFF0F0 (rouge très clair, lisible)

Contour / bordure : #C00000

Titre "Avertissement :" : #C00000 (même rouge que la bordure)

Texte du contenu : #000000 (noir classique)
Ilyas — Yesterday at 9:12 PM
rah importit new code ou kanrunni lstm il bay khdem
Ilyas — Yesterday at 10:50 PM
safi 9adite LSTM
Genosis — Yesterday at 11:39 PM
l3zz a bradrrr
smohat ead tinstallit
Genosis
 started a call that lasted 2 hours. — 8:58 PM
Genosis — 9:28 PM
alo
Genosis — 10:03 PM
# src/models/evaluation.py

from pathlib import Path
from typing import Sequence

import numpy as np
Expand
message.txt
7 KB
# src/models/train_baselines.py

from pathlib import Path

import joblib
import numpy as np
Expand
message.txt
7 KB
Genosis — 10:29 PM
ilyas
Genosis
 started a call that lasted a few seconds. — 11:02 PM
Ilyas
 started a call. — 11:21 PM
Ilyas — 11:22 PM
ouee hamza
﻿
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
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
            # On peut mettre un scaler ou non ; ici on ne le met pas
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
    - sauvegarde X_test / y_test
    - lance la grid search pour tous les modèles
    - sauvegarde le meilleur modèle
    - sauvegarde le nom du meilleur modèle dans results/best_model_name.txt
    - retourne (results, best_name)
    """
    X_full, y_full = load_processed()

    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y_full,
        test_size=test_size,
        random_state=random_state,
        stratify=y_full,
    )

    # Sauvegarde du test set pour réutilisation en évaluation
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(X_test, PROCESSED_DIR / "X_test.joblib")
    joblib.dump(y_test, PROCESSED_DIR / "y_test.joblib")
    print("X_test et y_test sauvegardés dans data/processed")

    results = run_grid_search(X_train, y_train, X_test, y_test)

    best_name = max(results, key=lambda m: results[m]["test_accuracy"])
    best = results[best_name]

    print("\n=== BEST MODEL ===")
    print("Best model :", best_name)
    print("Best params :", best["best_params"])
    print("CV accuracy :", best["cv_score"])
    print("Test accuracy :", best["test_accuracy"])

    # Sauvegarder le meilleur modèle en local
    best_model_path = PROJECT_ROOT / "models" / f"{best_name}_best.joblib"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["best_estimator"], best_model_path)
    print("Meilleur modèle sauvegardé dans :", best_model_path)

    # Sauvegarder le nom du meilleur modèle pour l'évaluation
    best_name_path = RESULTS_DIR / "best_model_name.txt"
    best_name_path.write_text(best_name, encoding="utf-8")
    print("Nom du meilleur modèle sauvegardé dans :", best_name_path)

    return results, best_name


if __name__ == "__main__":
    train_baselines()