import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif


# =========================
# 1. VERİYİ OKU
# =========================

data = pd.read_csv("oznitelikler_phase3.csv")

X = data.drop(columns=["Duygu", "Dosya_Adi", "emotion", "file_name"], errors="ignore")
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

y = data["Duygu"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Veri seti:", X.shape)
print("Sınıflar:", class_names)
print("\n--- PHASE 3 İLERİ MODEL DENEMELERİ BAŞLADI ---")


# =========================
# 2. MODEL DENEME FONKSİYONU
# =========================

results = []

def dene(model_adi, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"{model_adi}: %{acc * 100:.2f}")

    results.append({
        "name": model_adi,
        "model": model,
        "accuracy": acc,
        "prediction": y_pred
    })


# =========================
# 3. MLP GRID SEARCH
# =========================

mlp_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        max_iter=1200,
        early_stopping=True,
        random_state=42
    ))
])

mlp_params = {
    "mlp__hidden_layer_sizes": [
        (128, 64),
        (256, 128),
        (256, 128, 64),
        (300, 150, 75),
        (512, 256, 128)
    ],
    "mlp__activation": ["relu", "tanh"],
    "mlp__alpha": [0.0001, 0.001, 0.01],
    "mlp__learning_rate_init": [0.001, 0.0005, 0.0001]
}

grid_mlp = GridSearchCV(
    mlp_pipeline,
    mlp_params,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy",
    n_jobs=-1
)

dene("MLP GridSearch", grid_mlp)


# =========================
# 4. SVM GRID SEARCH
# =========================

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True, random_state=42))
])

svm_params = {
    "svm__C": [10, 50, 100, 200, 500],
    "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.05],
    "svm__kernel": ["rbf"]
}

grid_svm = GridSearchCV(
    svm_pipeline,
    svm_params,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy",
    n_jobs=-1
)

dene("SVM GridSearch", grid_svm)


# =========================
# 5. FEATURE SELECTION + SVM
# =========================

for k in [40, 60, 80, 100]:
    if k < X.shape[1]:
        fs_svm = Pipeline([
            ("select", SelectKBest(score_func=f_classif, k=k)),
            ("scaler", StandardScaler()),
            ("svm", SVC(C=100, gamma="scale", kernel="rbf", probability=True, random_state=42))
        ])
        dene(f"SelectKBest({k}) + SVM", fs_svm)


# =========================
# 6. EXTRA TREES / RANDOM FOREST
# =========================

extra_trees = ExtraTreesClassifier(
    n_estimators=500,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

dene("Extra Trees", extra_trees)

random_forest = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

dene("Random Forest", random_forest)


# =========================
# 7. SOFT VOTING ENSEMBLE
# =========================

best_mlp = grid_mlp.best_estimator_
best_svm = grid_svm.best_estimator_

ensemble = VotingClassifier(
    estimators=[
        ("mlp", best_mlp),
        ("svm", best_svm),
        ("extra", extra_trees)
    ],
    voting="soft",
    weights=[2, 2, 1]
)

dene("Soft Voting Ensemble", ensemble)


# =========================
# 8. EN İYİ MODELİ SEÇ
# =========================

best_result = max(results, key=lambda x: x["accuracy"])

print("\n--- EN İYİ PHASE 3 SONUCU ---")
print(f"En iyi model: {best_result['name']}")
print(f"En iyi accuracy: %{best_result['accuracy'] * 100:.2f}")

if hasattr(best_result["model"], "best_params_"):
    print("\nEn iyi parametreler:")
    print(best_result["model"].best_params_)

print("\nClassification Report:")
print(classification_report(y_test, best_result["prediction"], target_names=class_names))

# Sonuçları tablo olarak kaydet
results_df = pd.DataFrame([
    {
        "Model": r["name"],
        "Accuracy (%)": round(r["accuracy"] * 100, 2)
    }
    for r in results
]).sort_values("Accuracy (%)", ascending=False)

results_df.to_csv("phase3_ileri_model_sonuclari.csv", index=False)
print("\nSonuç tablosu kaydedildi: phase3_ileri_model_sonuclari.csv")
print(results_df)

# Confusion Matrix kaydet
cm = confusion_matrix(y_test, best_result["prediction"])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix - Phase 3 {best_result['name']}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_phase3_best.png", dpi=300)
plt.show()

print("\nConfusion matrix kaydedildi: confusion_matrix_phase3_best.png")