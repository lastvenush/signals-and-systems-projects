import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Grafiklerin düzgün görünmesi için stil ayarı
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


# ==========================================================
# 1. GRAFİK: PHASE 1 - PHASE 2 - PHASE 3 MODEL KARŞILAŞTIRMASI
# ==========================================================

models = ['Random Forest', 'SVM', 'MLP']

phase1_scores = [35.50, 35.50, 35.50]
phase2_scores = [50.00, 77.94, 80.88]
phase3_scores = [82.35, 86.76, 86.76]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width, phase1_scores, width, label='Phase 1 Baseline')
rects2 = ax.bar(x, phase2_scores, width, label='Phase 2 Optimized')
rects3 = ax.bar(x + width, phase3_scores, width, label='Phase 3 Final')

ax.set_ylabel('Accuracy Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison (Phase 1 vs Phase 2 vs Phase 3)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')
ax.set_ylim(0, 100)
ax.legend(fontsize=10)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'%{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('model_comparison_phase3.png', dpi=300)
plt.close()


# ==========================================================
# 2. GRAFİK: DATASET DUYGU DAĞILIMI
# ==========================================================

csv_file = "oznitelikler_phase3.csv"

try:
    df = pd.read_csv(csv_file)

    emotion_counts = df["Duygu"].value_counts()

    emotions = emotion_counts.index.tolist()
    counts = emotion_counts.values.tolist()

    fig, ax = plt.subplots(figsize=(8, 6))

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=emotions,
        autopct='%1.1f%%',
        startangle=140,
        textprops=dict(fontweight='bold'),
        wedgeprops=dict(width=0.4, edgecolor='w')
    )

    plt.setp(autotexts, size=10, weight="bold")
    plt.setp(texts, size=11, weight="bold")

    ax.set_title(
        "Balanced Dataset Distribution After C-Code Synchronization",
        fontsize=13,
        fontweight='bold',
        pad=20
    )

    plt.tight_layout()
    plt.savefig('dataset_distribution_phase3.png', dpi=300)
    plt.close()

except FileNotFoundError:
    print("Hata: oznitelikler_phase3.csv bulunamadı. Dataset dağılım grafiği çizilemedi.")


print("Grafikler başarıyla oluşturuldu:")
print("- model_comparison_phase3.png")
print("- dataset_distribution_phase3.png")