import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import os


def calculate_roc_auc(negative_scores, positive_scores):
    # 创建标签，0代表人类写的，1代表机器生成的
    labels = np.array([0] * len(negative_scores) + [1] * len(positive_scores))
    # 合并所有得分
    scores = np.array(negative_scores + positive_scores)
    valid_indices = ~np.isnan(labels) & ~np.isnan(scores)
    labels = labels[valid_indices]
    scores = scores[valid_indices]
    # 计算AUC
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    return auc, fpr, tpr

def draw_roc(human_scores, wm_score):
    auc_w, fpr_w, tpr_w = calculate_roc_auc(human_scores, wm_score)

    plt.figure()
    plt.plot(fpr_w, tpr_w, color='red', label=f'Adaptive watermarked (AUC = {auc_w:.4f})')

    # Diagonal line for random chance
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{os.path.basename(result_path).split('_')[0]} ROC")
    plt.legend(loc="lower right")
    plt.grid()
    # plt.savefig(f"outputs/roc.png", dpi=300, bbox_inches='tight')
    # plt.show()
    print('ROC-AUC:', round(auc_w*100, 2))

model_name = "twitter-roberta-base-sentiment" # 'gte-Qwen2-1.5B-instruct'  # 'twitter-roberta-base-sentiment'
if model_name == 'gte-Qwen2-1.5B-instruct':
    pool_type = 'attention'
    freeze_type = '-freeze'
    model_name = f'{model_name}-{pool_type}{freeze_type}'
dataset = "imdb"
batch_size = 128
num_epoch=87
cl_weight = 1.0
tl_weight = 0.0
neg_weight = 1
margin = 0.5

num_paraphrased_llama=8
num_paraphrased_gpt=8
num_negative_llama=0
num_negative_gpt=1
num_summary=0

if cl_weight != 0:
  neg_weight=neg_weight  # 1 32 64 128
else:
  neg_weight=999
if tl_weight != 0:
  margin=margin  # 0.2 0.5 0.8 1.1
else:
  margin=999


alpha=2.0
delta0, delta=0.2, 0.5

result_path = f'/blue/buyuheng/li_an.ucsb/projects/watermark-simcse/watermarking/outputs/end2end/{dataset}/{model_name}/{batch_size}batch_{num_epoch}epochs/sanity-check/llama{num_paraphrased_llama}-{num_negative_llama}gpt{num_paraphrased_gpt}-{num_negative_gpt}-{num_summary}/watermark-loss_cl{cl_weight}-tl{tl_weight}-wneg{neg_weight}-margin{margin}-10sent-alpha{alpha}-delta{delta0}|{delta}.csv'

print(os.path.basename(result_path))
df = pd.read_csv(result_path)

human_scores = df['human_score'].to_list()
for type_ in ['adaptive', 'paraphrased', 'hate']:
    print(type_, end=' ')
    wm_scores = df[f'{type_}_watermarked_text_score'].to_list()
    # debug
    non_empty_wm_scores = df[f'{type_}_watermarked_text_score'].dropna().to_list()
    non_empty_count = len(non_empty_wm_scores)
    print(non_empty_count, 'valid rows')
    draw_roc(human_scores, wm_scores)
