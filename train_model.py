import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from preprocess import load_and_preprocess

# ── 1. Load preprocessed data ─────────────────────────────────────────────────
print("=" * 50)
print("  Customer Churn Model Training")
print("=" * 50)

X, y, feature_cols = load_and_preprocess("data/Customer-Churn-Records.csv")

# ── 2. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size : {X_train.shape[0]} rows")
print(f"Test size  : {X_test.shape[0]} rows")

# ── 3. Handle class imbalance with SMOTE ──────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — Train size: {X_train_res.shape[0]} rows")

# ── 4. Define models ──────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost"            : XGBClassifier(eval_metric='logloss', random_state=42)
}

# ── 5. Train & evaluate all models ────────────────────────────────────────────
results = {}
print("\n--- Model Results ---")

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    results[name] = {
        "model" : model,
        "auc"   : auc,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

    print(f"\n{name}")
    print(f"  AUC-ROC : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

# ── 6. Save the best model ────────────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_name]["model"]

print(f"\n>>> Best Model : {best_name}  (AUC = {results[best_name]['auc']:.4f})")
joblib.dump(best_model, "churn_model.pkl")
print("Model saved   : churn_model.pkl")

# ── 7. Feature importance chart (Random Forest) ───────────────────────────────
rf          = results["Random Forest"]["model"]
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Feature Importances — Random Forest", fontsize=14)
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("Chart saved   : feature_importance.png")

# ── 8. Confusion matrix for best model ───────────────────────────────────────
cm  = confusion_matrix(y_test, results[best_name]["y_pred"])
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Retained", "Churned"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Chart saved   : confusion_matrix.png")

# ── 9. Summary table ──────────────────────────────────────────────────────────
print("\n--- Summary ---")
print(f"{'Model':<25} {'AUC-ROC':>10}")
print("-" * 37)
for name, res in results.items():
    marker = " ✓ BEST" if name == best_name else ""
    print(f"{name:<25} {res['auc']:>10.4f}{marker}")

print("\nAll files saved. Ready for Streamlit!")