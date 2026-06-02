import os
import librosa
import numpy as np
import pandas as pd
import scipy.stats as stats

def extract_features_from_audio(file_path):
    """
    Tek bir ses dosyasından gelişmiş zaman ve frekans alanı özniteliklerini çıkarır.
    """
    try:
        # 1. Ses dosyasını yükle (sr=None orijinal örnekleme oranını korur)
        y, sr = librosa.load(file_path, sr=None)
        
        # Sinyal boşsa veya yüklenemediyse atla
        if len(y) == 0:
            return None
        
        features = {}
        
        # --- ZAMAN ALANI ÖZNİTELİKLERİ ---
        # Sıfırdan Geçiş Oranı (Zero Crossing Rate)
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Kısa Zamanlı Enerji (Short-Time Energy - RMS ile simüle edilir)
        rms = librosa.feature.rms(y=y)
        features['ste_mean'] = np.mean(rms)
        features['ste_std'] = np.std(rms)
        
        # --- FREKANS ALANI / SPEKTRAL ÖZNİTELİKLER ---
        # Spektral Merkez Frekansı (Spectral Centroid - Sesin keskinliği/parlaklığı)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = np.mean(centroid)
        
        # Spektral Yayılım (Spectral Bandwidth)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(bandwidth)
        
        # İstatistiksel Değişkenler (Kurtosis ve Skewness)
        # Spektral kontrast üzerinden ses spektrumunun basık/dik durumunu analiz ederiz
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_kurtosis'] = np.mean(stats.kurtosis(spectral_contrast, axis=1))
        features['spectral_skewness'] = np.mean(stats.skew(spectral_contrast, axis=1))
        
        # --- PITCH CONTOUR (SES TONU DEĞİŞİMİ) ---
        # Temel frekansı (F0) piptrack ile takip ediyoruz
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        # Sadece sıfırdan büyük (belirgin) pitch değerlerini filtreleyip istatistiklerini alıyoruz
        pitch_values = pitches[pitches > 0]
        features['pitch_mean'] = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        features['pitch_std'] = np.std(pitch_values) if len(pitch_values) > 0 else 0
        
        # --- MFCC (Mel-Frequency Cepstral Coefficients) ---
        # Ses tanımadaki altın standart: 13 katsayı ve bunların standart sapmaları
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            
        return features

    except Exception as e:
        print(f"Hata oluştu ({file_path}): {e}")
        return None


# --- TÜM VERİ SETİNİ TARAMA VE CSV OLUŞTURMA SÜRECİ ---

DATASET_PATH = "dataset"
data_list = []

print("Veri seti taranıyor ve gelişmiş öznitelikler çıkarılıyor...")
print("Bu işlem ses dosyası sayınıza bağlı olarak birkaç dakika sürebilir...\n")

# Klasörün içindeki tüm alt klasörleri ve dosyaları tara
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith('.wav') or file.endswith('.WAV'):  # Uzantı kontrolü
            file_path = os.path.join(root, file)
            
            # Klasör adını duygu etiketi (label) olarak alıyoruz
            # Örn: dataset/happy/ses1.wav dosyasında root 'dataset/happy' olur, basename ise 'happy' kalır.
            emotion_label = os.path.basename(root)
            
            # Öznitelikleri çıkar
            audio_features = extract_features_from_audio(file_path)
            
            if audio_features is not None:
                audio_features['file_name'] = file
                audio_features['emotion'] = emotion_label
                data_list.append(audio_features)

# Toplanan verileri DataFrame'e dönüştür ve CSV olarak kaydet
if len(data_list) > 0:
    df = pd.DataFrame(data_list)
    
    # Sütun sıralamasını düzenle (Dosya adı ve Duygu etiketi en başta dursun)
    cols = ['file_name', 'emotion'] + [c for c in df.columns if c not in ['file_name', 'emotion']]
    df = df[cols]
    
    # Pelin'in doğrudan makine öğrenmesinde kullanacağı nihai CSV dosyası
    output_csv = "emotion_features.csv"
    df.to_csv(output_csv, index=False)
    
    print("-" * 50)
    print(f"🎉 İşlem Başarıyla Tamamlandı! Toplam {len(df)} adet ses dosyası işlendi.")
    print(f"📂 Veri seti '{output_csv}' adıyla klasörünüze kaydedildi.")
    print("-" * 50)
else:
    print(f"\nHata: '{DATASET_PATH}' klasörü içinde geçerli .wav formatında ses dosyası bulunamadı!")
    print("Lütfen ses dosyalarınızın dataset klasörünün altındaki duygu klasörlerinde olduğundan emin olun.")