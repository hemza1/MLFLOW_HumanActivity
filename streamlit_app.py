from pathlib import Path
import json

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yaml


# =========================
#   Chargement config
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

API_BASE_URL = cfg.get("api", {}).get("base_url", "http://127.0.0.1:8000")
N_FEATURES = int(cfg["n_features"])

st.set_page_config(
    page_title="HAR – SVM API Demo",
    layout="wide"
)

st.title("🏃 Human Activity Recognition – Demo SVM")
st.markdown(
    f"""
    Backend : **FastAPI** (`{API_BASE_URL}`)  
    Modèle : **SVM RBF** – {N_FEATURES} features (UCI HAR)
    """
)

# =========================
#   Section : test API
# =========================

with st.expander("🔍 Tester la connexion à l'API"):
    if st.button("Ping API"):
        try:
            resp = requests.get(f"{API_BASE_URL}/")
            st.json(resp.json())
        except Exception as e:
            st.error(f"Erreur de connexion à l'API : {e}")


st.markdown("---")
st.header("📂 1. Charger des données d'entrée")

uploaded_file = st.file_uploader(
    "Upload un fichier CSV avec 561 colonnes (une ou plusieurs lignes)",
    type=["csv"]
)

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Fichier chargé avec shape = {df.shape}")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Erreur lors de la lecture du CSV : {e}")

st.markdown("---")
st.header("🎯 2. Prédire l'activité")

if df is not None:
    if df.shape[1] != N_FEATURES:
        st.error(f"Le CSV doit contenir {N_FEATURES} colonnes, trouvé {df.shape[1]}")
    else:
        # Choisir une ligne
        row_idx = st.number_input(
            "Indice de la ligne à prédire",
            min_value=0,
            max_value=int(df.shape[0] - 1),
            value=0,
            step=1
        )

        row = df.iloc[row_idx].values.astype(float)
        st.write("Aperçu des premières features :", row[:10])

        if st.button("📡 Envoyer à l'API"):
            payload = {
                "instances": [
                    {"features": row.tolist()}
                ]
            }

            # /predict
            try:
                resp_pred = requests.post(f"{API_BASE_URL}/predict", json=payload)
                resp_pred.raise_for_status()
                pred_json = resp_pred.json()
                st.subheader("Résultat /predict")
                st.json(pred_json)
            except Exception as e:
                st.error(f"Erreur /predict : {e}")

            # /predict_proba
            try:
                resp_proba = requests.post(f"{API_BASE_URL}/predict_proba", json=payload)
                resp_proba.raise_for_status()
                proba_json = resp_proba.json()
                st.subheader("Résultat /predict_proba")

                # Afficher tableau des proba
                probs = np.array(proba_json["probabilities"][0])
                class_ids = proba_json["class_ids"]
                class_labels = proba_json["class_labels"]

                proba_df = pd.DataFrame({
                    "class_id": class_ids,
                    "label": class_labels,
                    "probability": probs
                }).sort_values("probability", ascending=False)

                st.dataframe(proba_df.style.format({"probability": "{:.4f}"}))

            except Exception as e:
                st.error(f"Erreur /predict_proba : {e}")

else:
    st.info("Upload un CSV pour activer la prédiction.")
