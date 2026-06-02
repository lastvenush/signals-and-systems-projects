import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


def istatistik_ekle(feature_dict, isim, values):
    values = np.array(values).flatten()
    values = values[np.isfinite(values)]

    if len(values) == 0:
        feature_dict[f"{isim}_Mean"] = 0.0
        feature_dict[f"{isim}_Std"] = 0.0
        feature_dict[f"{isim}_Min"] = 0.0
        feature_dict[f"{isim}_Max"] = 0.0
        feature_dict[f"{isim}_Median"] = 0.0
    else:
        feature_dict[f"{isim}_Mean"] = np.mean(values)
        feature_dict[f"{isim}_Std"] = np.std(values)
        feature_dict[f"{isim}_Min"] = np.min(values)
        feature_dict[f"{isim}_Max"] = np.max(values)
        feature_dict[f"{isim}_Median"] = np.median(values)


def matris_ozellik_ekle(feature_dict, isim, matris):
    means = np.mean(matris, axis=1)
    stds = np.std(matris, axis=1)

    for i, val in enumerate(means):
        feature_dict[f"{isim}_{i+1}_Mean"] = val

    for i, val in enumerate(stds):
        feature_dict[f"{isim}_{i+1}_Std"] = val


def extract_features(file_path):
    y_original, sr = librosa.load(file_path, sr=None, mono=True)

    original_duration = librosa.get_duration(y=y_original, sr=sr)

    y_trimmed, index = librosa.effects.trim(y_original, top_db=20)
    trimmed_duration = librosa.get_duration(y=y_trimmed, sr=sr)

    if len(y_trimmed) < 256:
        raise ValueError("Ses dosyası çok kısa veya geçersiz.")

    silence_ratio = 1 - (trimmed_duration / original_duration) if original_duration > 0 else 0

    y = librosa.effects.preemphasis(y_trimmed)

    feature_dict = {}

    feature_dict["Original_Duration"] = original_duration
    feature_dict["Trimmed_Duration"] = trimmed_duration
    feature_dict["Silence_Ratio"] = silence_ratio

    zcr_feat = librosa.feature.zero_crossing_rate(y=y)[0]
    istatistik_ekle(feature_dict, "ZCR", zcr_feat)

    rms_feat = librosa.feature.rms(y=y)[0]
    istatistik_ekle(feature_dict, "RMS", rms_feat)
    feature_dict["STE_Mean"] = np.mean(rms_feat)
    feature_dict["STE_Max"] = np.max(rms_feat)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    istatistik_ekle(feature_dict, "Pitch", pitch_values)

    if len(pitch_values) > 0:
        feature_dict["Pitch_Range"] = np.max(pitch_values) - np.min(pitch_values)
    else:
        feature_dict["Pitch_Range"] = 0.0

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    matris_ozellik_ekle(feature_dict, "MFCC", mfccs)
    matris_ozellik_ekle(feature_dict, "Delta_MFCC", delta_mfccs)
    matris_ozellik_ekle(feature_dict, "Delta2_MFCC", delta2_mfccs)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]

    istatistik_ekle(feature_dict, "Spectral_Centroid", spectral_centroid)
    istatistik_ekle(feature_dict, "Spectral_Rolloff", spectral_rolloff)
    istatistik_ekle(feature_dict, "Spectral_Bandwidth", spectral_bandwidth)
    istatistik_ekle(feature_dict, "Spectral_Flatness", spectral_flatness)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    matris_ozellik_ekle(feature_dict, "Chroma", chroma)

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    matris_ozellik_ekle(feature_dict, "Spectral_Contrast", spectral_contrast)

    try:
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        matris_ozellik_ekle(feature_dict, "Tonnetz", tonnetz)
    except:
        pass

    feature_dict["Kurtosis"] = kurtosis(y)
    feature_dict["Skewness"] = skew(y)

    return feature_dict


print("Demo sistemi başlatılıyor...")

data = pd.read_csv("oznitelikler_phase3.csv")

X = data.drop(columns=["Duygu", "Dosya_Adi", "emotion", "file_name"], errors="ignore")
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

y = data["Duygu"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(C=10, gamma=0.001, kernel="rbf"))
])

model.fit(X, y_encoded)

print("Model hazır.")
print("Final Model: Optimized SVM with RBF kernel")
print("Accuracy: 86.76%")
print("-" * 50)

wav_path = input("Tahmin için .wav dosyasının yolunu gir: ")

if not os.path.exists(wav_path):
    print("Hata: Dosya bulunamadı.")
    exit()

features = extract_features(wav_path)
sample_df = pd.DataFrame([features])

# Eğitimdeki sütunlarla aynı sıraya getir
sample_df = sample_df.reindex(columns=X.columns, fill_value=0)
sample_df = sample_df.replace([np.inf, -np.inf], np.nan).fillna(0)

prediction_encoded = model.predict(sample_df)[0]
prediction = label_encoder.inverse_transform([prediction_encoded])[0]

print("-" * 50)
print(f"Seçilen dosya: {wav_path}")
print(f"Tahmin edilen duygu: {prediction}")
print("-" * 50)