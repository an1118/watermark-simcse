import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

result = r"//mnt/data2/lian/projects/watermark/data/Llama-3.1-8B-Instruct/onebatch-c4-2pass_paraphrase-sim.csv"
df = pd.read_csv(result)

# df_long = pd.melt(df[['sim_ori_wm', 'sim_ori_para', 'sim_wm_para']], 
df_long = pd.melt(df[['sim_ori_pass1', 'sim_ori_pass2']], 
                  var_name='Score Type', value_name='Score')
# df_long = pd.melt(df[['human_score', 'watermarked_text_score', 'g_watermarked_text_score']], 
#                   var_name='Score Type', value_name='Score')

plt.figure(figsize=(8, 5))
sns.boxplot(x='Score Type', y='Score', data=df_long)

plt.title(f"Similarity score")
plt.xlabel('Text Type')
plt.ylabel('Score')
plt.xticks(rotation=30)

# 显示图表
plt.savefig(f"outputs/boxplot.png", dpi=300, bbox_inches='tight')
plt.show()
print('Complete!')
