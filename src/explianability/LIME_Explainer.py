from pathlib import Path
import numpy as np
import joblib
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

# ============================================================
# 1. Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print(">>> LIME RUN")
print("PROJECT_ROOT =", PROJECT_ROOT)

# ============================================================
# 2. Load Data
# ============================================================
print("\nLoading x_full.joblib and y_full.joblib...")

X_full = joblib.load(PROCESSED_DIR / "x_full.joblib")
y_full = joblib.load(PROCESSED_DIR / "y_full.joblib")

# Convert to numpy (MANDATORY for LIME)
X_full = np.asarray(X_full)
y_full = np.asarray(y_full)

print("X_full :", X_full.shape)
print("y_full :", y_full.shape)

feature_names = [f"feat_{i}" for i in range(X_full.shape[1])]
class_names = sorted(np.unique(y_full).tolist())

print("Classes =", class_names)

# ============================================================
# 3. Find Best Model Automatically
# ============================================================
print("\nSearching for *_best.joblib in /models ...")

best_models = list(MODELS_DIR.glob("*_best.joblib"))
if len(best_models) == 0:
    raise FileNotFoundError("❌ No *_best.joblib found in /models")

best_model_path = best_models[0]
print("✔ Best model found:", best_model_path.name)

model = joblib.load(best_model_path)
print("✔ Model loaded!")

# ============================================================
# 4. Build LIME Explainer
# ============================================================
print("\nBuilding LimeTabularExplainer...")

explainer = LimeTabularExplainer(
    training_data=X_full,
    feature_names=feature_names,
    class_names=class_names,
    discretize_continuous=True,
    mode="classification",
)

print("✔ LIME explainer ready!")

# ============================================================
# 5. Explain One Instance
# ============================================================
idx = 0
x = X_full[idx]

print(f"\nExplaining instance idx={idx}, true label={y_full[idx]}")

exp = explainer.explain_instance(
    data_row=x,
    predict_fn=model.predict_proba,
    num_features=10,
)

print("\nTop features (LIME):")
for feat, weight in exp.as_list():
    print(f"{feat}: {weight:.4f}")

# ============================================================
# 6. Save HTML + PNG (summary-style)
# ============================================================
# HTML interactif
out_html = RESULTS_DIR / f"lime_explanation_{idx}.html"
exp.save_to_file(str(out_html))
print(f"\n✔ LIME HTML saved at: {out_html}")

# PNG statique façon "summary"
fig = exp.as_pyplot_figure()
png_path = RESULTS_DIR / "lime_summary.png"   # même esprit que shap_summary.png
fig.savefig(png_path, dpi=250, bbox_inches="tight")
plt.close(fig)

print(f"✔ LIME PNG saved at: {png_path}")
print("Tu peux l'ajouter direct dans ton rapport 🔥")
