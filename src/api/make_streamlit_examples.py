import numpy as np
import pandas as pd
from pathlib import Path


def main():
    # 1) Localiser la racine du projet
    # src/utils -> src -> (racine du projet)
    project_root = Path(__file__).resolve().parents[2]
    print(f"Project root = {project_root}")

    # 2) Dossier UCI HAR
    har_root = project_root / "data" / "raw" / "UCI HAR Dataset"

    X_test_path = har_root / "test" / "X_test.txt"
    y_test_path = har_root / "test" / "y_test.txt"
    subject_test_path = har_root / "test" / "subject_test.txt"
    features_path = har_root / "features.txt"
    activity_labels_path = har_root / "activity_labels.txt"

    # Petite vérification
    for p in [X_test_path, y_test_path, subject_test_path, features_path, activity_labels_path]:
        if not p.exists():
            raise FileNotFoundError(f"Fichier manquant : {p}")

    # 3) Charger les noms de features
    # features.txt : id  name
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
    print(f"Nombre de features : {len(feature_names)}")

    # 4) Charger X_test, y_test, subjects
    print("Chargement de X_test / y_test / subject_test ...")
    X_test = np.loadtxt(X_test_path)
    y_test = np.loadtxt(y_test_path, dtype=int).ravel()
    subjects = np.loadtxt(subject_test_path, dtype=int).ravel()

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("subjects shape:", subjects.shape)

    assert X_test.shape[0] == y_test.shape[0] == subjects.shape[0], "Mismatch nombre de lignes"

    # 5) Charger activity_labels
    activity_raw = np.loadtxt(activity_labels_path, dtype=str)
    activity_map = {int(row[0]): row[1] for row in activity_raw}
    print("Activity map:", activity_map)

    # 6) Construire le DataFrame
    df = pd.DataFrame(X_test, columns=feature_names)
    df.insert(0, "activity_id", y_test)
    df.insert(0, "subject", subjects)
    df["activity_label"] = df["activity_id"].map(activity_map)

    print("Shape complet DF:", df.shape)

    # 7) On échantillonne un peu (pour ne pas avoir un CSV gigantesque)
    SAMPLE_SIZE = 200  # tu peux changer à 100, 500, etc.
    if len(df) > SAMPLE_SIZE:
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    else:
        df_sample = df

    # 8) Sauvegarde
    out_dir = project_root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_har_examples.csv"

    df_sample.to_csv(out_path, index=False, float_format="%.6f")
    print("✅ CSV créé :", out_path)
    print("Shape final :", df_sample.shape)


if __name__ == "__main__":
    main()
