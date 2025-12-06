from pathlib import Path
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print(">>> SHAP RUN (SURROGATE RF MODEL)")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


print("\nLoading x_full.joblib and y_full.joblib...")

X_full = joblib.load(PROCESSED_DIR / "x_full.joblib")
y_full = joblib.load(PROCESSED_DIR / "y_full.joblib")

X_full = np.asarray(X_full)
y_full = np.asarray(y_full)

print("✔ Loaded:", X_full.shape, y_full.shape)

feature_names = [f"feat_{i}" for i in range(X_full.shape[1])]

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

print("✔ Train shape:", X_train.shape, "Test shape:", X_test.shape)


print("\nTraining RandomForest surrogate model for SHAP...")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

print("✔ RandomForest trained!")


print("\nBuilding SHAP TreeExplainer...")
explainer = shap.TreeExplainer(rf)
print("✔ TreeExplainer ready!")


N_SAMPLES = 100
if X_test.shape[0] < N_SAMPLES:
    N_SAMPLES = X_test.shape[0]

X_sample = X_test[:N_SAMPLES]
y_sample = y_test[:N_SAMPLES]

print(f"\nComputing SHAP values on {N_SAMPLES} samples...")
shap_values = explainer.shap_values(X_sample)
print("✔ SHAP values computed!")


print("\nAnalyzing SHAP values structure...")


if isinstance(shap_values, list):
    print("SHAP values is a list of length:", len(shap_values))
    sv = shap_values[0]
else:
    sv = shap_values
    print("SHAP values type:", type(sv))


if sv.ndim == 3:
    print("sv ndim=3, shape:", sv.shape, " -> taking first class along last dim")
    sv = sv[:, :, 0]

print("Final SHAP matrix shape:", sv.shape)
print("X_sample shape:", X_sample.shape)


n_s, n_f = sv.shape
if n_s != X_sample.shape[0] or n_f != X_sample.shape[1]:
    raise ValueError(f"Mismatch: shap_values {sv.shape} vs X_sample {X_sample.shape}")


summary_png = RESULTS_DIR / "shap_summary_rf.png"

plt.figure(figsize=(12, 6))
shap.summary_plot(
    sv,
    X_sample,
    feature_names=feature_names,
    show=False
)
plt.savefig(summary_png, dpi=250, bbox_inches='tight')
print(f"✔ SHAP summary saved at: {summary_png}")


idx = 0
x = X_sample[idx:idx+1]
y_true = y_sample[idx]

force_html = RESULTS_DIR / f"shap_force_rf_{idx}.html"

try:

    exp_val = explainer.expected_value
    if isinstance(exp_val, list) or isinstance(exp_val, np.ndarray):
        exp_val0 = exp_val[0]
    else:
        exp_val0 = exp_val

    shap.save_html(
        str(force_html),
        shap.force_plot(
            exp_val0,
            sv[idx],
            feature_names=feature_names
        )
    )
    print(f"✔ SHAP force plot saved at: {force_html}")
except Exception as e:
    print(" Unable to render force plot:", e)

print("\nSHAP with surrogate RandomForest completed successfully!")
