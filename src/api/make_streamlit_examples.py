import numpy as np
import pandas as pd
from pathlib import Path


def main():
    # 1) Localiser la racine du projet
    project_root = Path(__file__).resolve().parents[2]
    print(f"Project root = {project_root}")

    # 2) Dossier UCI HAR
    har_root = project_root / "data" / "raw" / "UCI HAR Dataset"

    X_test_path = har_root / "test" / "X_test.txt"
    features_path = har_root / "features.txt"

    # Vérification
    for p in [X_test_path, features_path]:
        if not p.exists():
            raise FileNotFoundError(f"Fichier manquant : {p}")

    # 3) Charger les noms de features depuis features.txt
    # Format typique : 1 tBodyAcc-mean()-X
    features = np.loadtxt(features_path, dtype=str)
    feature_names = features[:, 1].tolist()

    def clean_name(name: str) -> str:
        return (
            name.replace("(", "")
            .replace(")", "")
            .replace("-", "_")
            .replace(",", "_")
        )

    feature_names = [clean_name(f) for f in feature_names]
    print(f"Nombre de features (noms) : {len(feature_names)}")

    # 4) Charger X_test (les vraies features)
    print("Chargement de X_test ...")
    X_test = np.loadtxt(X_test_path)
    print("X_test shape:", X_test.shape)

    # Sécurité : on vérifie qu'on a bien 561 colonnes
    if X_test.shape[1] != len(feature_names):
        raise ValueError(
            f"Mismatch entre X_test.shape[1]={X_test.shape[1]} et len(feature_names)={len(feature_names)}"
        )

    # 5) Construire un DataFrame avec UNIQUEMENT les features
    df = pd.DataFrame(X_test, columns=feature_names)
    print("DF features-only shape:", df.shape)

    # 6) Échantillonnage
    SAMPLE_SIZE = 200
    if len(df) > SAMPLE_SIZE:
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    else:
        df_sample = df

    # 7) Sauvegarde du CSV compatible Streamlit
    out_dir = project_root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_har_features_only.csv"

    df_sample.to_csv(out_path, index=False, float_format="%.6f")
    print("✅ CSV features-only créé :", out_path)
    print("Shape final pour Streamlit :", df_sample.shape)


if __name__ == "__main__":
    main()
