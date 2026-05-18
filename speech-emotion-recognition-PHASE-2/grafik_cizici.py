import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Grafiklerin düzgün görünmesi için stil ayarı
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 1. GRAFİK: MODEL KARŞILAŞTIRMASI (Phase 1 vs Phase 2)
models = ['Random Forest', 'SVM', 'MLP']
phase1_scores = [35.50, 35.50, 35.50]  # Faz 1 ham baseline skorlarınız
phase2_scores = [50.00, 77.94, 80.88]  # Bu gece aldığımız şampiyonluk skorları

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, phase1_scores, width, label='Phase 1 (Baseline)', color='#34495e')
rects2 = ax.bar(x + width/2, phase2_scores, width, label='Phase 2 (Optimized)', color='#2ecc71')

ax.set_ylabel('Accuracy Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison (Phase 1 vs Phase 2)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')
ax.set_ylim(0, 100)
ax.legend(fontsize=11)

# Barların üzerine yüzde değerlerini yazma
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'%{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)
plt.tight_layout()
plt.savefig('/Users/PELIN/Desktop/phase2/model_comparison.png', dpi=300)
plt.close()

# 2. GRAFİK: VERİ SETİ DUYGU DAĞILIMI (Senin Düzelttiğin C Kodları Dağılımı)
emotions = ['Angry', 'Neutral', 'Happy', 'Surprised', 'Sad']
counts = [176, 174, 141, 99, 89]  # Senin ekrandaki tam dağılımın
colors = ['#e74c3c', '#95a5a6', '#f1c40f', '#3498db', '#9b59b6']

fig, ax = plt.subplots(figsize=(7, 5))
wedges, texts, autotexts = ax.pie(counts, labels=emotions, autopct='%1.1f%%',
                                  startangle=140, colors=colors, 
                                  textprops=dict(fontweight='bold'),
                                  wedgeprops=dict(width=0.4, edgecolor='w')) # Donut grafiği

plt.setp(autotexts, size=10, weight="bold")
plt.setp(texts, size=11, weight="bold")
ax.set_title("Balanced Dataset Distribution After C-Code Synchronization", fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/Users/PELIN/Desktop/phase2/dataset_distribution.png', dpi=300)
plt.close()

print("✅ Grafikler başarıyla oluşturuldu! 'model_comparison.png' ve 'dataset_distribution.png' klasöründe hazır.")