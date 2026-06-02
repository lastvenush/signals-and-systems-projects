import pandas as pd

# Mevcut klasördeki metadata dosyasını okur
csv_dosya_yolu = "metadata.csv"

try:
    df = pd.read_csv(csv_dosya_yolu)
    print("CSV dosyası okundu. Etiketler güncelleniyor...")

    def detect_emotion(filename):
        filename = str(filename).lower()

        # C kodları varsa önce onları kontrol et
        if "_c1" in filename:
            return "Neutral"
        elif "_c2" in filename:
            return "Happy"
        elif "_c3" in filename:
            return "Angry"
        elif "_c4" in filename:
            return "Sad"
        elif "_c5" in filename:
            return "Surprised"

        # Dosya adındaki kelimelerden duygu yakalama
        if "ofke" in filename or "ofkeli" in filename or "angry" in filename or "furious" in filename:
            return "Angry"
        elif "mutlu" in filename or "happy" in filename:
            return "Happy"
        elif "notr" in filename or "neutral" in filename or "no╠êtr" in filename:
            return "Neutral"
        elif "uzgun" in filename or "sad" in filename or "mutsuz" in filename or "u╠êzgu╠ên" in filename:
            return "Sad"
        elif "saskin" in filename or "sasirmis" in filename or "surprised" in filename or "shocked" in filename:
            return "Surprised"
        else:
            return "Unknown"

    # Dosya_Adi sütunu varsa onu kullan
    if "Dosya_Adi" in df.columns:
        df["Duygu"] = df["Dosya_Adi"].apply(detect_emotion)

    # Bazı dosyalarda file_name olabilir
    elif "file_name" in df.columns:
        df["Duygu"] = df["file_name"].apply(detect_emotion)

    else:
        raise ValueError("CSV içinde 'Dosya_Adi' veya 'file_name' sütunu bulunamadı.")

    # Güncellenmiş metadata dosyasını kaydet
    df.to_csv(csv_dosya_yolu, index=False)

    print(f"İşlem tamam! '{csv_dosya_yolu}' güncellendi.")

    print("\nYeni Duygu Dağılımı:")
    print(df["Duygu"].value_counts())

except Exception as e:
    print(f"Hata oluştu: {e}")