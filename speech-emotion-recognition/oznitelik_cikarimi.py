import librosa
import numpy as np
import pandas as pd
import os

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # ZCR için hem ortalama hem standart sapma (Değişkenliği yakalar)
        zcr_feat = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr_feat)
        zcr_std = np.std(zcr_feat) 
        
        # STE (Enerji) için hem ortalama hem maksimum patlama noktası
        rms_feat = librosa.feature.rms(y=y)
        ste_mean = np.mean(rms_feat)
        ste_max = np.max(rms_feat) 
        
        # Pitch için ortalama
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0
        
        return zcr_mean, zcr_std, ste_mean, ste_max, pitch_mean
    except Exception as e:
        return None, None, None, None, None

# --- DOSYA İŞLEME DÖNGÜSÜ ---

ana_klasor = "Midterm_Dataset_2026" 
features_list = []

for root, dirs, files in os.walk(ana_klasor):
    for dosya_adi in files:
        if dosya_adi.endswith(".wav"):
            dosya_yolu = os.path.join(root, dosya_adi)
            
            # Duygu etiketini dosya isminden ayıklama
            def get_label(name):
                name = name.lower()
                if 'angry' in name or 'ofkeli' in name or 'furious' in name: return 'Angry'
                if 'happy' in name or 'mutlu' in name: return 'Happy'
                if 'neutral' in name or 'notr' in name: return 'Neutral'
                if 'sad' in name or 'uzgun' in name or 'mutsuz' in name: return 'Sad'
                if 'surprised' in name or 'saskin' in name or 'shocked' in name: return 'Surprised'
                return 'Unknown'

            duygu = get_label(dosya_adi)
            
            # Sadece etiketli verileri işliyoruz (Skoru korumak için)
            if duygu != 'Unknown':
                z_mean, z_std, s_mean, s_max, p_mean = extract_features(dosya_yolu)
                
                if z_mean is not None:
                    features_list.append({
                        "Dosya_Adi": dosya_adi,
                        "ZCR_Mean": z_mean,
                        "ZCR_Std": z_std,
                        "STE_Mean": s_mean,
                        "STE_Max": s_max,
                        "Pitch_Mean": p_mean,
                        "Duygu": duygu
                    })

# CSV Kaydet (Orijinal isimle üzerine yazar)
df = pd.DataFrame(features_list)
df.to_csv("oznitelikler.csv", index=False)
print(f"İşlem tamam! Toplam {len(df)} dosya başarıyla 'oznitelikler.csv' olarak kaydedildi.")
