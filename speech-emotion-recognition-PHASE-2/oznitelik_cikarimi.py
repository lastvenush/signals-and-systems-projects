import librosa
import numpy as np
import pandas as pd
import os
from scipy.stats import kurtosis, skew

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # Sinyal Temizleme ve Güçlendirme
        y, index = librosa.effects.trim(y, top_db=20)
        y = librosa.effects.preemphasis(y)
        
        # --- PHASE 1 ÖZELLİKLERİ ---
        zcr_feat = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr_feat)
        zcr_std = np.std(zcr_feat) 
        
        rms_feat = librosa.feature.rms(y=y)
        ste_mean = np.mean(rms_feat)
        ste_max = np.max(rms_feat) 
        
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0
        
        # --- PHASE 2 YENİ ÖZELLİKLER ---
        # 1. MFCC (13 Standart Katsayı)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0) 
        
        # 🌟 SESİN HIZI VE İVMESİ (Delta & Delta-Delta)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        delta_mfccs_mean = np.mean(delta_mfccs.T, axis=0)
        delta2_mfccs_mean = np.mean(delta2_mfccs.T, axis=0)
        
        # 2. Spektral Özellikler
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # 3. İstatistiksel Parametreler
        sig_kurtosis = kurtosis(y)
        sig_skewness = skew(y)
        
        # Sözlüğü burada güvenli bir şekilde tanımlıyoruz
        feature_dict = {
            "ZCR_Mean": zcr_mean, "ZCR_Std": zcr_std,
            "STE_Mean": ste_mean, "STE_Max": ste_max,
            "Pitch_Mean": pitch_mean,
            "Spectral_Centroid": spec_centroid,
            "Spectral_Rolloff": spec_rolloff,
            "Kurtosis": sig_kurtosis,
            "Skewness": sig_skewness
        }
        
        # Standart MFCC'leri ekle
        for i, val in enumerate(mfccs_mean):
            feature_dict[f"MFCC_{i+1}"] = val
            
        # Delta'ları ekle
        for i, val in enumerate(delta_mfccs_mean):
            feature_dict[f"Delta_MFCC_{i+1}"] = val
            
        # Delta-Delta'ları ekle
        for i, val in enumerate(delta2_mfccs_mean):
            feature_dict[f"Delta2_MFCC_{i+1}"] = val
            
        return feature_dict
        
    except Exception as e:
        print(f"Hata: {e}")
        return None

# --- DOSYA İŞLEME DÖNGÜSÜ ---

ana_klasor = "/Users/PELIN/Desktop/phase2/dataset/Midterm_Dataset_2026"  
features_list = []

for root, dirs, files in os.walk(ana_klasor):
    for dosya_adi in files:
        if dosya_adi.endswith(".wav"):
            dosya_yolu = os.path.join(root, dosya_adi)
            
            # Geçici olarak Unknown ata, etiketleri birazdan c koduyla netleştireceğiz
            duygu = 'Unknown' 
            
            res = extract_features(dosya_yolu)
            if res is not None:
                res["Dosya_Adi"] = dosya_adi
                res["Duygu"] = duygu
                features_list.append(res)

# CSV Kaydet
df = pd.DataFrame(features_list)
df.to_csv("/Users/PELIN/Desktop/phase2/oznitelikler_phase2.csv", index=False)
print(f"İşlem tamam! Yeni özelliklerle 'oznitelikler_phase2.csv' kaydedildi.")