import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Veri Setini Yükleme
file_name = 'oznitelikler.csv'

try:
    df = pd.read_csv(file_name)
    df.columns = df.columns.str.strip().str.lower()
except FileNotFoundError:
    print(f"HATA: '{file_name}' bulunamadı!")
    exit()

# 2. Sütun Tespit Etme
zcr_col = [c for c in df.columns if 'zcr' in c][0]
ste_col = [c for c in df.columns if 'ste' in c][0]
pitch_col = [c for c in df.columns if 'pitch' in c][0]
dosya_col = [c for c in df.columns if 'dosya' in c or 'name' in c][0]

# --- VERİ TEMİZLEME ---
for col in [zcr_col, ste_col, pitch_col]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=[zcr_col, ste_col, pitch_col])

# 3. DUYGU ETİKETLEYİCİ
def extract_emotion_pro(filename):
    f = str(filename).lower() 
    mapping = {
        'angry': 'Angry', 'furious': 'Angry', 'ofkeli': 'Angry', 'öfkeli': 'Angry',
        'happy': 'Happy', 'mutlu': 'Happy', 'joyful': 'Happy',
        'sad': 'Sad', 'uzgun': 'Sad', 'üzgün': 'Sad', 'mutsuz': 'Sad',
        'surprised': 'Surprised', 'shocked': 'Surprised', 'saskin': 'Surprised', 'şaşkın': 'Surprised',
        'neutral': 'Neutral', 'notr': 'Neutral', 'nötr': 'Neutral'
    }
    for key, value in mapping.items():
        if key in f: return value
    return "Unknown"

df['Duygu_Etiketi'] = df[dosya_col].apply(extract_emotion_pro)
df_clean = df[df['Duygu_Etiketi'] != 'Unknown'].copy()

# 4. VERİ BÖLME (Tam 200 Test Verisi İçin Sabit Sayı Kullanıyoruz)
X = df_clean[[zcr_col, ste_col, pitch_col]]
y = df_clean['Duygu_Etiketi']

# test_size=200 yazarak tam 200 adet test verisi ayrılmasını garanti ediyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=200, random_state=42, stratify=y)

# 5. ÖLÇEKLENDİRME (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. MODEL AYARLARI
model = RandomForestClassifier(
    n_estimators=150, 
    max_depth=12, 
    random_state=42, 
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)

# 7. RAPORLAMA
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n" + "="*55)
print(f"--- SERRA'NIN MODEL BAŞARI RAPORU (TAM 200 TEST) ---")
print(f"="*55)
print(f"Sisteme Giren Toplam Veri Sayısı: {len(df_clean)}")
print(f"Modeli Eğitmek İçin Kullanılan: {len(X_train)}")
print(f"Doğrulama (Test) İçin Kullanılan: {len(X_test)}") # Burada tam 200 yazacak
print(f"Gerçek Doğruluk Oranı (Accuracy): %{accuracy*100:.2f}")
print("-" * 55)
print("Duygu Bazlı Detaylı Başarı Tablosu (Yüzdelik):")
print(classification_report(y_test, y_pred))
print("="*55)

# 8. HATA MATRİSİ
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title(f'Duygu Klasifikasyonu Hata Matrisi (200 Test Verisi)')
plt.show()
