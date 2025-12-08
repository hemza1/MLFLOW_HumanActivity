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
│   ├── processed/          # X_full.joblib, y_full.joblib, splits
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── data_experiments.ipynb
│   ├── data_explainability.ipynb
│
├── src/
│   ├── api/
│   │   ├── main.py                     # FastAPI service
│   │   ├── make_streamlit_examples.py  # Generate demo CSV
│   ├── data/
│   │   ├── preprocessing.py
│   ├── models/
│   │   ├── train_baselines.py
│   │   ├── evaluation.py
│   ├── explianability/
│   │   ├── LIME_Explainer.py
│   │   ├── SHAP_Explainer.py
│
├── models/                 # Modèles sauvegardés (RandomForest_best.joblib, svm_rbf_best.joblib…)
│
├── results/
│   ├── lime_summary_0.png
│   ├── shap_summary_rf.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── lime_explanation_0.html
│   ├── shap_force_rf_0.html
│
├── mlruns/                 # MLflow experiment tracking
│
├── config.yaml             # API and model configuration
├── dvc.yaml                # DVC pipeline definition
├── params.yaml             # Training hyperparameters
├── requirements.txt
├── streamlit_app.py        # Streamlit demo UI
└── README.md
```

## Prerequisites
- Python 3.9+
- Git
- (Optional) DVC for data versioning
- (Optional) Access to remote MLflow tracking server

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/hemza1/MLFLOW_HumanActivity.git
cd MLFLOW_HumanActivity
```

### 2. Create and activate a virtual environment
```bash
# Linux/Mac
python -m venv env
source env/bin/activate

# Windows PowerShell
python -m venv env
.\env\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**Note**: If you encounter version conflicts with `scikit-learn`/`scikeras`/`TensorFlow`, consider pinning:
- `scikit-learn==1.5.2`
- `scikeras==0.12.0`
- `tensorflow==2.15.0`

## Usage

### 1. Preprocessing

Transform raw UCI HAR files into processed feature arrays:
```bash
python -m src.data.preprocessing
```

**Outputs**:
- `data/processed/X_full.joblib` (concatenated train+test features)
- `data/processed/y_full.joblib` (concatenated labels)

### 2. Training Baseline Models

Run GridSearchCV over multiple classifiers (Logistic Regression, Random Forest, SVM RBF, KNN):
```bash
python -m src.models.train_baselines
```

**What it does**:
- Loads processed data
- Splits into train/test (80/20)
- Runs GridSearchCV with 3-fold CV per model
- Logs all runs to **remote MLflow server**: `http://ec2-16-171-234-189.eu-north-1.compute.amazonaws.com:5000`
- Saves best model to `models/<model_name>_best.joblib`
- Persists test split as `data/processed/X_test.joblib` and `y_test.joblib`

**Expected duration**: ~5-15 minutes depending on grid size.

### 3. Model Evaluation

Compute metrics and generate plots for the best model:
```bash
python -m src.models.evaluation
```

**Outputs** (in `results/figures/`):
- Confusion matrix: `cm_<model_name>.png`
- Multiclass ROC curves: `roc_<model_name>.png`
- Console: accuracy, classification report

**Typical best model**: SVM RBF with test accuracy ≈ 0.987

### 4. Explainability Analysis

#### LIME (Local Interpretable Model-agnostic Explanations)
Explain individual predictions:
```bash
python src/explianability/LIME_Explainer.py
```

**Outputs**:
- `results/lime_explanation_0.html` – interactive explanation
- `results/lime_summary_0.png` – feature importance bar chart

#### SHAP (SHapley Additive exPlanations)
Global feature importance via surrogate RandomForest:
```bash
python src/explianability/SHAP_Explainer.py
```

**Outputs**:
- `results/shap_summary_rf.png` – summary plot across 100 samples
- `results/shap_force_rf_0.html` – force plot for instance 0

**Note**: Both scripts can also be explored interactively in `notebooks/data_explainability.ipynb`.

### 5. MLflow Tracking

This project logs to a **remote MLflow server**:
```
http://ec2-16-171-234-189.eu-north-1.compute.amazonaws.com:5000
```

All training runs, metrics, and artifacts are centralized there.

**To view runs**:
1. Open the URL in your browser
2. Navigate to the `HAR_baselines` experiment
3. Compare CV scores, test accuracy, hyperparameters, and logged models

**Tracked metadata**:
- Hyperparameters (C, gamma, n_estimators, etc.)
- CV accuracy (mean over folds)
- Test accuracy
- Best estimator (serialized)

**Local MLflow UI** (optional, if using local `mlruns/`):
```bash
mlflow ui --backend-store-uri mlruns/
# Open http://localhost:5000
```

### 6. FastAPI Prediction Service

Deploy a REST API for real-time predictions:

#### Prerequisites
Ensure the following files exist in `models/`:
- `svm_rbf_best.joblib` (trained model)
- `scaler.joblib` (StandardScaler)
- `activity_labels.joblib` (id → label mapping)

#### Start the API
```bash
cd src/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Endpoints
- `GET /` – health check
- `POST /predict` – returns class IDs and labels
- `POST /predict_proba` – returns probability distributions

#### Example request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"features": [0.1, -0.2, ..., 0.05]}]}'
```

**Response**:
```json
{
  "predictions": [5],
  "labels": ["STANDING"]
}
```

### 7. Streamlit Demo UI

Interactive web interface to test predictions:
```bash
streamlit run streamlit_app.py
```

**Features**:
- Upload CSV with 561 features
- Select a row and send to FastAPI backend
- View predicted activity and probabilities

**Requires**: FastAPI running on `http://127.0.0.1:8000` (configurable in `config.yaml`)

### 8. DVC (Data Version Control)

Track data changes and reproduce pipelines:
```bash
dvc init
dvc add data/raw/UCI\ HAR\ Dataset
dvc repro
```

###  DVC Remote Storage (AWS S3)

**Note**: To access DVC remote storage on Amazon S3, you must:
1. Install the AWS CLI: `pip install awscli`
2. Configure AWS credentials:
    ```bash
    aws configure
    ```
    Enter your AWS Access Key ID, Secret Access Key, and default region.
3. Request access to the S3 bucket from the project maintainers.

**Alternative**: Use local DVC storage or your own S3 bucket by modifying `.dvc/config`:
```bash
dvc remote add -d myremote s3://your-bucket-name/path
```

If S3 access is unavailable, DVC will work locally without pushing to remote storage.

## Key Results

### Model Performance
- **Best model**: SVM RBF + StandardScaler
- **Test accuracy**: ≈ 0.987 (98.7%)
- **Well-separated activities**: Walking, Sitting, Laying
- **GridSearch details**: Available in `notebooks/data_experiments.ipynb`

### Explainability
- **LIME**: Local feature importance for individual predictions
- **SHAP**: Global feature importance via surrogate RandomForest (100 samples)

### Deployment
- **FastAPI**: Production-ready REST API for real-time inference
- **Streamlit**: Interactive demo UI for quick testing
- **MLflow**: Remote tracking for experiment management

---

## Troubleshooting

### scikit-learn version conflicts
If GridSearch raises `AttributeError: '__sklearn_tags__'`:
```bash
pip install scikit-learn==1.5.2 scikeras==0.12.0
```

### TensorFlow/Keras errors (if using MLP/LSTM)
Install compatible versions:
```bash
pip install tensorflow==2.15.0
```

### MLflow connection issues
Ensure the remote server is reachable:
```bash
curl http://ec2-16-171-234-189.eu-north-1.compute.amazonaws.com:5000
```

### API 404 errors
Verify model artifacts exist:
```bash
ls models/svm_rbf_best.joblib models/scaler.joblib models/activity_labels.joblib
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

MIT License – see `LICENSE` for details.

---

## Contact

For questions or collaboration:
- GitHub:
[@NassimBnslmn](https://github.com/NassimBnslmn)
[@hemza1](https://github.com/hemza1)
[@ilyassox](https://github.com/ilyassox)
- Project: [MLFLOW_HumanActivity](https://github.com/hemza1/MLFLOW_HumanActivity)