# src/data/preprocessing.py

from pathlib import Path
import numpy as np
import pandas as pd
import joblib


# Racine du projet : MLFLOW_HumanActivity/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "UCI HAR Dataset"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_har():
    """
    Charge les données brutes UCI HAR et retourne X_full, y_full, feature_names.
    """
    # Noms de colonnes
    features_path = RAW_DIR / "features.txt"
    features = pd.read_csv(
        features_path,
        sep=r"\s+",
        header=None,
        names=["id", "name"],
    )
    feature_names = features["name"].tolist()

    # Train
    X_train = pd.read_csv(
        RAW_DIR / "train" / "X_train.txt",
        delim_whitespace=True,
        header=None,
    )
    y_train = pd.read_csv(
        RAW_DIR / "train" / "y_train.txt",
        delim_whitespace=True,
        header=None,
    )[0].to_numpy()

    # Test
    X_test = pd.read_csv(
        RAW_DIR / "test" / "X_test.txt",
        delim_whitespace=True,
        header=None,
    )
    y_test = pd.read_csv(
        RAW_DIR / "test" / "y_test.txt",
        delim_whitespace=True,
        header=None,
    )[0].to_numpy()

    # Ajout des noms de colonnes
    X_train.columns = feature_names
    X_test.columns = feature_names

    # Concaténer train + test comme dans le notebook
    X_full = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_full = np.concatenate([y_train, y_test], axis=0)

    print("X_full shape :", X_full.shape)
    print("y_full shape :", y_full.shape)

    return X_full, y_full, feature_names


def save_processed(X_full, y_full, out_dir: Path | None = None):
    """
    Sauvegarde X_full et y_full en .joblib dans data/processed.
    """
    if out_dir is None:
        out_dir = PROCESSED_DIR

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(X_full, out_dir / "X_full.joblib")
    joblib.dump(y_full, out_dir / "y_full.joblib")
    print(f"X_full & y_full sauvés dans : {out_dir}")


def build_and_save_processed():
    """
    Pipeline complet : charge les données brutes, construit X_full / y_full
    et les sauvegarde dans data/processed.
    """
    X_full, y_full, _ = load_raw_har()
    save_processed(X_full, y_full)


if __name__ == "__main__":
    # Permet de lancer :
    #   python -m src.data.preprocessing
    build_and_save_processed()
