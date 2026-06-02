import pandas as pd

dosya = "oznitelikler_phase3.csv"
df = pd.read_csv(dosya)

def kesin_duygu_bul(dosya_adi):
    ad = str(dosya_adi).upper()
    if '_C1' in ad: return 'Neutral'
    if '_C2' in ad: return 'Happy'
    if '_C3' in ad: return 'Angry'
    if '_C4' in ad: return 'Sad'
    if '_C5' in ad: return 'Surprised'
    
    ad = str(dosya_adi).lower()
    if 'ofke' in ad or 'angry' in ad or 'furious' in ad: return 'Angry'
    if 'mutlu' in ad or 'happy' in ad: return 'Happy'
    if 'notr' in ad or 'neutral' in ad: return 'Neutral'
    if 'uzgun' in ad or 'sad' in ad: return 'Sad'
    if 'saskin' in ad or 'surprised' in ad or 'shocked' in ad: return 'Surprised'
    return 'Unknown'

df['Duygu'] = df['Dosya_Adi'].apply(kesin_duygu_bul)
df.to_csv(dosya, index=False)
print("✅ Harika! Tüm etiketler Pelin'in bulduğu C kodlarına göre kusursuz güncellendi.")
print(df['Duygu'].value_counts())
