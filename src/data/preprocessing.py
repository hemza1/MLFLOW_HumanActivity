"""
Data preprocessing utilities for the UCI HAR dataset.

Main steps:
- read raw train/test splits and feature names from UCI HAR
- concatenate into `X_full` and `y_full`
- persist processed artifacts into `data/processed` as Joblib files
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import joblib


# Racine du projet : MLFLOW_HumanActivity/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "UCI HAR Dataset"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_har() -> Tuple[pd.DataFrame, np.ndarray, list]:
    """Load raw UCI HAR splits and return consolidated data.

    Returns
    -------
    X_full : pd.DataFrame
        Concatenated train+test features with column names applied.
    y_full : np.ndarray
        Concatenated labels aligned with X_full.
    feature_names : list
        Ordered list of sensor feature names from `features.txt`.
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
        sep=r"\s+",
        engine="python",
        header=None,
    )
    y_train = pd.read_csv(
        RAW_DIR / "train" / "y_train.txt",
        sep=r"\s+",
        engine="python",
        header=None,
    )[0].to_numpy()

    # Test
    X_test = pd.read_csv(
        RAW_DIR / "test" / "X_test.txt",
        sep=r"\s+",
        engine="python",
        header=None,
    )
    y_test = pd.read_csv(
        RAW_DIR / "test" / "y_test.txt",
        sep=r"\s+",
        engine="python",
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


def save_processed(X_full, y_full, out_dir: Optional[Path] = None) -> None:
    """Persist processed arrays to Joblib files.

    Parameters
    ----------
    X_full : array-like or DataFrame
        Feature matrix to save.
    y_full : array-like
        Label vector to save.
    out_dir : Path, optional
        Target directory (defaults to `data/processed`).
    """
    if out_dir is None:
        out_dir = PROCESSED_DIR

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(X_full, out_dir / "X_full.joblib")
    joblib.dump(y_full, out_dir / "y_full.joblib")
    print(f"X_full & y_full sauvés dans : {out_dir}")


def build_and_save_processed() -> None:
    """End-to-end preprocessing entrypoint for CLI use."""
    X_full, y_full, _ = load_raw_har()
    save_processed(X_full, y_full)


if __name__ == "__main__":
    # Permet de lancer :
    #   python -m src.data.preprocessing
    build_and_save_processed()
