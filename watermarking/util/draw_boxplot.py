import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

result = r"/blue/buyuheng/li_an.ucsb/projects/watermark-simcse/watermarking/outputs/end2end/c4/gte-Qwen2-1.5B-instruct-attention-freeze/128batch_22epochs/sanity-check/llama8-0gpt8-0-0/watermark-8b-loss_cl2_gr_wneg1-10sent-alpha2.0-delta0.2|0.5-sim.csv"
df = pd.read_csv(result)

df_long = pd.melt(df[['sim_ori_wm', 'sim_ori_para', 'sim_wm_para', 'sim_ori_hate']], 
# df_long = pd.melt(df[['sim_ori_pass1', 'sim_ori_pass2']], 
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
plt.savefig(f"watermarking/outputs/boxplot.png", dpi=300, bbox_inches='tight')
plt.show()
print('Complete!')
