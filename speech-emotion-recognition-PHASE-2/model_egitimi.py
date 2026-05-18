import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# 1. Veriyi Yükleme
data = pd.read_csv('/Users/PELIN/Desktop/phase2/oznitelikler_phase2.csv')

X = data.drop(columns=['Duygu', 'Dosya_Adi', 'emotion', 'file_name'], errors='ignore')
y = data['Duygu']

le = LabelEncoder()
y = le.fit_transform(y)

# Duygu isimleri
class_names = le.classes_

# 2. Train-Test Ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- MLP MODELİ BAŞLADI ---")

# 3. MLP Parametre Ayarları
mlp_params = {
    'hidden_layer_sizes': [(128, 64), (256, 128, 64)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'max_iter': [500]
}

grid_mlp = GridSearchCV(
    MLPClassifier(random_state=42),
    mlp_params,
    cv=4,
    n_jobs=-1
)

grid_mlp.fit(X_train_scaled, y_train)

best_mlp = grid_mlp.best_estimator_

# MLP tahminleri
y_pred_mlp = best_mlp.predict(X_test_scaled)
mlp_acc = accuracy_score(y_test, y_pred_mlp)

# 4. SVM Modeli
grid_svm = SVC(C=100, gamma='scale', kernel='rbf', random_state=42)
grid_svm.fit(X_train_scaled, y_train)

y_pred_svm = grid_svm.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, y_pred_svm)

print("\n--- YENİ SKORLAR ---")
print(f"MLP Başarı: %{mlp_acc*100:.2f}")
print(f"SVM Başarı: %{svm_acc*100:.2f}")

if mlp_acc > svm_acc:
    print(f"\nEn yüksek skor: %{mlp_acc*100:.2f} (Algoritma: MLP)")
    best_model_name = "MLP"
    best_y_pred = y_pred_mlp
else:
    print(f"\nEn yüksek skor: %{svm_acc*100:.2f} (Algoritma: SVM)")
    best_model_name = "SVM"
    best_y_pred = y_pred_svm

# 5. Classification Report
print("\n Classification Report:")
print(classification_report(y_test, best_y_pred, target_names=class_names))

# 6. Confusion Matrix Çizdirme
cm = confusion_matrix(y_test, best_y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix - Phase 2 {best_model_name} Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

# Görseli kaydet
plt.savefig("/Users/PELIN/Desktop/phase2/confusion_matrix_phase2.png", dpi=300)

# Ekranda göster
plt.show()