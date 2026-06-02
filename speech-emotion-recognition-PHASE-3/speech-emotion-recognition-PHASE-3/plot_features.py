import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_file = "emotion_features.csv"
try:
    df = pd.read_csv(csv_file)
    print(f"{csv_file} başarıyla okundu. Filtreleme başlıyor...")
except FileNotFoundError:
    print(f"Hata: {csv_file} bulunamadı!")
    exit()

# --- DOSYA ADINDAN TEMİZ DUYGU YAKALAMA ---
def get_true_emotion_from_filename(filename):
    name_lower = str(filename).lower()
    if 'mutlu' in name_lower or 'happy' in name_lower:
        return 'Mutlu'
    elif 'notr' in name_lower or 'neutral' in name_lower or 'nofeli' in name_lower:
        return 'Nötr'
    elif 'ofkeli' in name_lower or 'angry' in name_lower:
        return 'Öfkeli'
    elif 'uzgun' in name_lower or 'sad' in name_lower:
        return 'Üzgün'
    elif 'sasirmis' in name_lower or 'surprise' in name_lower:
        return 'Şaşırmış'
    else:
        return 'Diğer'

# Temizlenmiş duyguları yeni sütuna atıyoruz
df['Gercek_Duygu'] = df['file_name'].apply(get_true_emotion_from_filename)

# 'Diğer' grubunu eleyerek sadece 5 ana duyguyu bırakıyoruz
df = df[df['Gercek_Duygu'].isin(['Mutlu', 'Nötr', 'Öfkeli', 'Üzgün', 'Şaşırmış'])]

# Görselleştirme Tasarımı
sns.set_theme(style="whitegrid")

# --- GRAFİK 2: MFCC-1 (Sadece bu grafiği yeniden çizdiriyoruz) ---
plt.figure(figsize=(9, 5))

# x="Gercek_Duygu" diyerek grup isimlerini tamamen devre dışı bırakıyoruz!
sns.barplot(x="Gercek_Duygu", y="mfcc_1_mean", data=df, palette="coolwarm", errorbar=None)

plt.title("Duygusal Durumlara Göre MFCC-1 Katsayısı Ortalama Değişimi", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Duygu (Emotion)", fontsize=11, labelpad=10)
plt.ylabel("MFCC_1 Ortalama Değeri", fontsize=11, labelpad=10)
plt.tight_layout()

# Yeni bir isimle kaydediyoruz ki eskisiyle çakışmasın
plt.savefig("mfcc_comparison_temiz.png", dpi=300)
plt.show()

print("\n🎉 İşlem tamam! Yeni grafik 'mfcc_comparison_temiz.png' adıyla kaydedildi.")
print("Bu grafiğin altında sadece 5 ana duygu göreceksiniz, grup isimleri tamamen silindi!")