# 📌 Human Activity Recognition (HAR) – MLflow & MLOps Pipeline

## 🌟 Résumé du projet
Ce projet implémente une pipeline complète de reconnaissance d’activités humaines (HAR) à partir du dataset UCI HAR.  
Il inclut : prétraitement, modèles baseline, évaluation, explainability (LIME & SHAP), suivi d’expérimentations MLflow, versionning avec DVC et structure modulaire.

## 📁 Structure
- data/ – raw + processed  
- notebooks/ – exploration & modèles  
- src/ – code modulaire  
- models/ – meilleurs modèles  
- results/ – figures (ROC, CM)  
- mlruns/ – tracking MLflow  

## 📦 Installation
```
git clone <repo_url>
cd MLFLOW_HumanActivity
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.\.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 🔧 1. Prétraitement
```
python -m src.data.preprocessing
```

## 🤖 2. Training baseline
```
python -m src.models.train_baselines
```

## 📈 3. Évaluation
Les figures (ROC, matrices de confusion) sont générées dans :
```
results/figures/
```

## 🔍 4. Explainability
Analyse LIME & SHAP depuis les notebooks.

## 🧪 MLflow
```
mlflow ui --backend-store-uri mlruns/
```

## 🔄 DVC
```
dvc init
dvc add data/raw/UCI\ HAR\ Dataset
dvc repro
```

## 👤 Auteur
Hamza El Yesri – M2 SID – Safran Aircraft Engines
