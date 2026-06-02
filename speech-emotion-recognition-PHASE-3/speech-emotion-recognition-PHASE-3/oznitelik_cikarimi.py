import librosa
import numpy as np
import pandas as pd
import os
from scipy.stats import kurtosis, skew

# DATASET KLASÖRÜ
ana_klasor = "dataset/Midterm_Dataset_2026"

# ÇIKTI DOSYASI
cikti_dosyasi = "oznitelikler_phase3.csv"


def duygu_bul(dosya_adi):
    ad = str(dosya_adi).upper()

    if "_C1" in ad:
        return "Neutral"
    elif "_C2" in ad:
        return "Happy"
    elif "_C3" in ad:
        return "Angry"
    elif "_C4" in ad:
        return "Sad"
    elif "_C5" in ad:
        return "Surprised"

    ad = str(dosya_adi).lower()

    if "ofke" in ad or "angry" in ad or "furious" in ad:
        return "Angry"
    elif "mutlu" in ad or "happy" in ad:
        return "Happy"
    elif "notr" in ad or "neutral" in ad:
        return "Neutral"
    elif "uzgun" in ad or "sad" in ad or "mutsuz" in ad:
        return "Sad"
    elif "saskin" in ad or "surprised" in ad or "shocked" in ad:
        return "Surprised"
    else:
        return "Unknown"


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
    try:
        y_original, sr = librosa.load(file_path, sr=None, mono=True)

        original_duration = librosa.get_duration(y=y_original, sr=sr)

        # Silence trimming
        y_trimmed, index = librosa.effects.trim(y_original, top_db=20)
        trimmed_duration = librosa.get_duration(y=y_trimmed, sr=sr)

        if len(y_trimmed) < 256:
            return None

        silence_ratio = 1 - (trimmed_duration / original_duration) if original_duration > 0 else 0

        # Pre-emphasis
        y = librosa.effects.preemphasis(y_trimmed)

        feature_dict = {}

        # Duration features
        feature_dict["Original_Duration"] = original_duration
        feature_dict["Trimmed_Duration"] = trimmed_duration
        feature_dict["Silence_Ratio"] = silence_ratio

        # ZCR
        zcr_feat = librosa.feature.zero_crossing_rate(y=y)[0]
        istatistik_ekle(feature_dict, "ZCR", zcr_feat)

        # RMS / STE
        rms_feat = librosa.feature.rms(y=y)[0]
        istatistik_ekle(feature_dict, "RMS", rms_feat)
        feature_dict["STE_Mean"] = np.mean(rms_feat)
        feature_dict["STE_Max"] = np.max(rms_feat)

        # Pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]

        istatistik_ekle(feature_dict, "Pitch", pitch_values)

        if len(pitch_values) > 0:
            feature_dict["Pitch_Range"] = np.max(pitch_values) - np.min(pitch_values)
        else:
            feature_dict["Pitch_Range"] = 0.0

        # MFCC - Phase 3'te 20 katsayı
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        matris_ozellik_ekle(feature_dict, "MFCC", mfccs)
        matris_ozellik_ekle(feature_dict, "Delta_MFCC", delta_mfccs)
        matris_ozellik_ekle(feature_dict, "Delta2_MFCC", delta2_mfccs)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]

        istatistik_ekle(feature_dict, "Spectral_Centroid", spectral_centroid)
        istatistik_ekle(feature_dict, "Spectral_Rolloff", spectral_rolloff)
        istatistik_ekle(feature_dict, "Spectral_Bandwidth", spectral_bandwidth)
        istatistik_ekle(feature_dict, "Spectral_Flatness", spectral_flatness)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        matris_ozellik_ekle(feature_dict, "Chroma", chroma)

        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        matris_ozellik_ekle(feature_dict, "Spectral_Contrast", spectral_contrast)

        # Tonnetz
        try:
            y_harmonic = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
            matris_ozellik_ekle(feature_dict, "Tonnetz", tonnetz)
        except:
            pass

        # Signal statistics
        feature_dict["Kurtosis"] = kurtosis(y)
        feature_dict["Skewness"] = skew(y)

        return feature_dict

    except Exception as e:
        print(f"Hata: {file_path} -> {e}")
        return None


# --- DOSYA İŞLEME DÖNGÜSÜ ---

features_list = []

print("Feature extraction başladı...")

for root, dirs, files in os.walk(ana_klasor):
    for dosya_adi in files:
        if dosya_adi.lower().endswith(".wav"):
            dosya_yolu = os.path.join(root, dosya_adi)

            duygu = duygu_bul(dosya_adi)

            if duygu == "Unknown":
                print(f"Etiket bulunamadı, atlandı: {dosya_adi}")
                continue

            res = extract_features(dosya_yolu)

            if res is not None:
                res["Dosya_Adi"] = dosya_adi
                res["Duygu"] = duygu
                features_list.append(res)

df = pd.DataFrame(features_list)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

df.to_csv(cikti_dosyasi, index=False)

print("İşlem tamam!")
print(f"Kaydedilen dosya: {cikti_dosyasi}")
print(f"Toplam işlenen dosya sayısı: {len(df)}")

if "Duygu" in df.columns:
    print("\nDuygu dağılımı:")
    print(df["Duygu"].value_counts())