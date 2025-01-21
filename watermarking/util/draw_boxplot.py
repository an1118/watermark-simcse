import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

result = r"/mnt/data2/lian/projects/watermark/adaptive-text-watermark-yepeng/outputs/global/watermark-8b-10sent-gpt2-alpha2.0-beta1.0.csv"
df = pd.read_csv(result)

df_long = pd.melt(df[['human_score', 'adaptive_watermarked_text_score']], 
                  var_name='Score Type', value_name='Score')
# df_long = pd.melt(df[['human_score', 'watermarked_text_score', 'g_watermarked_text_score']], 
#                   var_name='Score Type', value_name='Score')

plt.figure(figsize=(8, 5))
sns.boxplot(x='Score Type', y='Score', data=df_long)

plt.title(f"detection score")
plt.xlabel('Text Type')
plt.ylabel('Score')
plt.xticks(rotation=30)

# 显示图表
plt.savefig(f"outputs/boxplot.png", dpi=300, bbox_inches='tight')
plt.show()
print('Complete!')
