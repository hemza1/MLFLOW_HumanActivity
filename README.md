# Human Activity Recognition (HAR) – MLflow & MLOps Pipeline

## Résumé du projet
Ce projet implémente une pipeline complète de reconnaissance d’activités humaines (Human Activity Recognition) à partir du dataset UCI HAR.
Il combine MLOps, expérimentation avec MLflow, gestion du versioning des données via DVC, évaluation avancée, et explainability (LIME & SHAP).

L’objectif principal est de construire un workflow reproductible permettant :

    - l’ingestion et la préparation des données

    - l’entraînement de modèles baseline (LogReg, RandomForest, SVM, MLP, LSTM)

    - la sélection du meilleur modèle

    - l’analyse de performance

    - l’explicabilité locale (LIME) et globale (SHAP)

    - le suivi des expériences (MLflow)

    - la traçabilité des données (DVC)

## Structure
```
MLFLOW_HumanActivity/
│
├── data/
│   ├── raw/                # Dataset UCI HAR original
│   ├── processed/          # x_full.joblib, y_full.joblib, splits
│
├── notebooks/
│   ├── 1_exploration.ipynb
│   ├── 2_training.ipynb
│   ├── 3_explainability.ipynb
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │
│   ├── models/
│   │   ├── train_baselines.py
│   │   ├── evaluation.py
│   │
│   ├── explianability/
│   │   ├── LIME_Explainer.py
│   │   ├── SHAP_Explainer.py
│
├── models/                 # Modèles sauvegardés (RandomForest_best.joblib, svm_rbf_best.joblib…)
│
├── results/
│   ├── lime_summary.png
│   ├── shap_summary_rf.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── lime_explanation_0.html
│   ├── shap_force_0.html
│
├── mlruns/                 # MLflow experiment tracking
│
├── dvc.yaml                # Pipeline de données DVC
├── requirements.txt
└── README.md
```

## Installation
```
git clone https://github.com/hemza1/MLFLOW_HumanActivity.git
cd MLFLOW_HumanActivity
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.\.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 1. Prétraitement
 - Transforme les fichiers bruts UCI HAR → features normalisés + labels.
 - Génère x_full.joblib et y_full.joblib dans data/processed/.
```
python -m src.data.preprocessing
```

## 2. Training baseline
```
python -m src.models.train_baselines
```

## 3. Évaluation
```
python -m src.models.evaluation
```
Les figures (ROC, matrices de confusion) sont générées dans :
 - Matrices de confusion
 - Courbes ROC multi-classes
 - Classification report
 - Scores macro/micro

```
results/figures/
```

## 4. Explainability
Analyse LIME & SHAP depuis les notebooks.
```
python src/explianability/LIME_Explainer.py
```
```
python src/explianability/SHAP_Explainer.py
```

## MLflow
```
mlflow ui --backend-store-uri mlruns/
```
Interface disponible sur :
```
 http://localhost:5000/
```
MLflow enregistre :

 - paramètres d’entraînement

 - courbes de perte

 - métriques (accuracy, f1, recall…)

 - modèles

 - artefacts

## DVC
```
dvc init
dvc add data/raw/UCI\ HAR\ Dataset
dvc repro
```

## Résultats principaux
- Meilleur modèle : SVM RBF avec StandardScaler
- Accuracy : ~95–97% selon les runs
- Activités bien discriminées : Walking vs Sitting vs Laying
- Explainability :
    LIME → poids des features locaux
    SHAP → importance globale (surrogate RF)