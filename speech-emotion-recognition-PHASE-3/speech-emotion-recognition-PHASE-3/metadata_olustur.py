import os
import pandas as pd

# Seslerin olduğu ana klasör
ana_klasor = "/Users/PELIN/Desktop/phase2/dataset/Midterm_Dataset_2026"
metadata_listesi = []

for root, dirs, files in os.walk(ana_klasor):
    for dosya_adi in files:
        if dosya_adi.endswith(".wav"):
            # Sesin içinde bulunduğu klasörün adını al (Örn: "G15_Ofke")
            klasor_adi = os.path.basename(root).lower()
            
            # Klasör adından duyguyu eşleştir
            if 'ofke' in klasor_adi or 'angry' in klasor_adi: duygu = 'Angry'
            elif 'mutlu' in klasor_adi or 'happy' in klasor_adi: duygu = 'Happy'
            elif 'notr' in klasor_adi or 'neutral' in klasor_adi: duygu = 'Neutral'
            elif 'uzgun' in klasor_adi or 'sad' in klasor_adi: duygu = 'Sad'
            elif 'saskin' in klasor_adi or 'surprised' in klasor_adi: duygu = 'Surprised'
            else: duygu = 'Unknown'
            
            # Dosya adını ve duygusunu listeye kaydet
            metadata_listesi.append({"Dosya_Adi": dosya_adi, "Duygu": duygu})

# Listeyi CSV'ye çevir ve kaydet
df = pd.DataFrame(metadata_listesi)
df.to_csv("/Users/PELIN/Desktop/phase2/metadata.csv", index=False)
print("İşlem bitti! metadata.csv dosyası phase2 klasöründe hazır.")