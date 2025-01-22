import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import os

def calculate_roc_auc(human_scores, watermarked_scores):
    # 创建标签，0代表人类写的，1代表机器生成的
    labels = np.array([0] * len(human_scores) + [1] * len(watermarked_scores))
    # 合并所有得分
    scores = np.array(human_scores + watermarked_scores)
    valid_indices = ~np.isnan(labels) & ~np.isnan(scores)
    import pdb
    pdb.set_trace()
    print(len(~np.isnan(scores)) - 50, 'valid rows')
    labels = labels[valid_indices]
    scores = scores[valid_indices]
    # 计算AUC
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    print('ROC-AUC:', round(auc*100, 2))
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
    plt.savefig(f"outputs/roc.png", dpi=300, bbox_inches='tight')
    plt.show()
    print('ROC-AUC:', round(auc_w*100, 2))


model_name = "twitter-roberta-base-sentiment"
print(f'=========={model_name}==========')
for cl_idx in [2, 3, 4]:
    for neg_weight in [1, 32, 64, 128]:
        result_path = f'watermarking/outputs/end2end/c4/{model_name}/watermark-8b-loss_cl{cl_idx}_gr_wneg{neg_weight}-10sent-alpha2.0-delta0.2|0.5.csv'
        if os.path.exists(result_path):
            print(f'====cl func: {cl_idx} / neg_weight: {neg_weight}====')
            df = pd.read_csv(result_path)

            human_scores = df['human_score'].to_list()
            for type_ in ['adaptive', 'hate', 'paraphrased']:
                print(type_, end=': ')
                wm_scores = df[f'{type_}_watermarked_text_score'].to_list()
                calculate_roc_auc(human_scores, wm_scores)
                # draw_roc(human_scores, wm_scores)
