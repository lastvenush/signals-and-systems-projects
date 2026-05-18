import pandas as pd

# CSV dosyasının yolu (Kendi dosya yolunla değiştir)
csv_dosya_yolu = "/Users/PELIN/Desktop/phase2/metadata.csv" 

try:
    df = pd.read_csv(csv_dosya_yolu)
    print("CSV dosyası okundu. Etiketler güncelleniyor...")

    def detect_emotion(filename):
        filename = filename.lower()
        if 'ofke' in filename or 'angry' in filename or 'furious' in filename:
            return 'Angry'
        elif 'mutlu' in filename or 'happy' in filename:
            return 'Happy'
        elif 'notr' in filename or 'neutral' in filename or 'no╠êtr' in filename:
            return 'Neutral'
        elif 'uzgun' in filename or 'sad' in filename or 'mutsuz' in filename or 'u╠êzgu╠ên' in filename:
            return 'Sad'
        elif 'saskin' in filename or 'surprised' in filename or 'shocked' in filename or 's╠ğas╠ğk─▒n' in filename or 's╠ğas╠ğirma' in filename:
            return 'Surprised'
        else:
            return 'Unknown'

    # Duygu sütununu güncelle
    df['Duygu'] = df['Dosya_Adi'].apply(detect_emotion)

    # Güncellenmiş veriyi kaydet
    df.to_csv(csv_dosya_yolu, index=False)
    print(f"İşlem tamam! '{csv_dosya_yolu}' güncellendi.")
    
    # Yeni dağılımı göster
    print("\nYeni Duygu Dağılımı:")
    print(df['Duygu'].value_counts())

except Exception as e:
    print(f"Hata oluştu: {e}")