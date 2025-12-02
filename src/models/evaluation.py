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
# src/models/evaluation.py

from pathlib import Path
from typing import Sequence

import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


# --- Chemins projet / dossiers ---

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_activity_labels():
    """
    Charge le fichier activity_labels.txt pour avoir les noms des classes.
    """
    labels_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "UCI HAR Dataset"
        / "activity_labels.txt"
    )
    df = np.loadtxt(labels_path, dtype=str)
    # format : id, label
    ids = df[:, 0].astype(int)
    labels = df[:, 1]
    # retourner dans l'ordre des ids
    return [label for _, label in sorted(zip(ids, labels), key=lambda x: x[0])]


def evaluate_model(
    model,
    X_test,
    y_test,
    class_names: Sequence[str] | None = None,
    prefix: str = "baseline",
):
    """
    Calcule accuracy, matrice de confusion, classification_report,
    courbes ROC multi-classe et sauvegarde les figures.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy ({prefix}) : {acc:.4f}")
    print("Classification report :\n", classification_report(y_test, y_pred))

    if class_names is None:
        # classes numériques 1..N
        classes = np.unique(y_test)
        class_names = [str(c) for c in classes]
    else:
        classes = np.unique(y_test)

    # Confusion matrix
    fig_cm_path = FIGURES_DIR / f"cm_{prefix}.png"
    plot_confusion_matrix(cm, class_names, fig_cm_path)

    # ROC multiclass (one-vs-rest)
    try:
        y_score = model.predict_proba(X_test)
        fig_roc_path = FIGURES_DIR / f"roc_{prefix}.png"
        plot_multiclass_roc(y_test, y_score, classes, class_names, fig_roc_path)
    except Exception as e:
        print("Impossible de calculer les courbes ROC (predict_proba manquant) :", e)
        fig_roc_path = None

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "cm_path": fig_cm_path,
        "roc_path": fig_roc_path,
    }


def plot_confusion_matrix(cm, class_names, out_path: Path):
    """
    Trace et sauvegarde la matrice de confusion.
    """
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.title("Matrice de confusion")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Matrice de confusion sauvegardée dans :", out_path)


def plot_multiclass_roc(y_true, y_score, classes, class_names, out_path: Path):
    """
    Trace les courbes ROC multi-classe (one-vs-rest) et les sauvegarde.
    """
    y_bin = label_binarize(y_true, classes=classes)
    n_classes = y_bin.shape[1]

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(7, 6))
    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taux de faux positifs (FPR)")
    plt.ylabel("Taux de vrais positifs (TPR)")
    plt.title("Courbes ROC multi-classe")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Courbes ROC sauvegardées dans :", out_path)


def load_best_model_name() -> str:
    """
    Lit le nom du meilleur modèle sauvegardé dans results/best_model_name.txt.
    """
    best_name_path = RESULTS_DIR / "best_model_name.txt"
    if not best_name_path.exists():
        raise FileNotFoundError(
            f"{best_name_path} introuvable. "
            "Lance d'abord l'entraînement (train_baselines.py) pour générer ce fichier."
        )
    best_name = best_name_path.read_text(encoding="utf-8").strip()
    print("Meilleur modèle (chargé depuis best_model_name.txt) :", best_name)
    return best_name


def main(best_name: str | None = None):
    """
    Script d'évaluation :
    - charge le meilleur modèle sauvegardé
    - charge X_test / y_test sauvegardés
    - calcule accuracy, CM, ROC et sauvegarde les figures.
    """
    # 1) Déterminer le nom du meilleur modèle
    if best_name is None:
        best_name = load_best_model_name()

    # 2) Charger le modèle
    model_path = MODELS_DIR / f"{best_name}_best.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle {model_path} introuvable. "
            "Vérifie que train_baselines.py a bien été exécuté."
        )
    model = joblib.load(model_path)
    print("Modèle chargé depuis :", model_path)

    # 3) Charger X_test / y_test
    X_test_path = PROCESSED_DIR / "X_test.joblib"
    y_test_path = PROCESSED_DIR / "y_test.joblib"
    if not X_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            "X_test.joblib ou y_test.joblib manquant dans data/processed. "
            "Assure-toi que train_baselines.py a bien été exécuté."
        )

    X_test = joblib.load(X_test_path)
    y_test = joblib.load(y_test_path)
    print("X_test et y_test chargés depuis data/processed")

    # 4) Charger les noms d'activités
    class_names = load_activity_labels()

    # 5) Évaluation
    results = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        prefix=best_name,
    )

    print("\nRésumé :")
    print("Accuracy :", results["accuracy"])
    print("Matrice de confusion sauvegardée dans :", results["cm_path"])
    print("Courbes ROC sauvegardées dans :", results["roc_path"])


if __name__ == "__main__":
    main()