
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