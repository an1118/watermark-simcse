import torch
from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm
from statistics import mean
from math import exp
import os

import pdb

folder_path = r'/mnt/data2/lian/projects/watermark/adaptive-text-watermark-yepeng/outputs/eval_ppl'
# generation_result_path_list = [
#     r'/mnt/data2/lian/projects/watermark/two-step-watermark/output/lfqa_g_sentiment_simcse_alpha2.5_beta0.0_delta0.5|1.0_prefix_watermark_result.csv',
#     ]
save_output = True
output_file = os.path.join(folder_path, 'ppl_result')
model_path = 'meta-llama/Llama-3.1-70B'
# model_path = 'meta-llama/Llama-3.1-8B'
tp_size = 4
gpu_memory_utilization=0.9
model = LLM(model=model_path, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_memory_utilization, max_model_len=1024, max_num_seqs=2)
sampling_params = SamplingParams(prompt_logprobs=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _print(*args, file_handle=None, **kwargs):
    if file_handle:
        print(*args, file=file_handle, flush=True, **kwargs)
    else:
        # Print to the command line as usual
        print(*args, **kwargs)

# Define perplexity calculation function
def calculate_text_perplexity(output):
    prompt_ids = output.prompt_token_ids
    null = []
    for idx, i in enumerate(output.prompt_logprobs):  # i is dict, {token_id: Logprob}
        if i is not None:
            token_id = prompt_ids[idx]
            null.append(i[token_id].logprob)
    null_1st_half = null[:len(null)//2]
    null_2nd_half = null[len(null)//2:]
    ppl_1st_half = exp(-mean(null_1st_half))
    ppl_2nd_half = exp(-mean(null_2nd_half))
    return ppl_1st_half, ppl_2nd_half

def _get_perplexity(df, column, file_handle=None):
    ppl_1st_half, ppl_2nd_half = [], []
    _print(f'=====Column: {column}=====', file_handle=file_handle)
    outputs = model.generate(df[column].dropna(), sampling_params)
    for output in tqdm(outputs, desc="Calculating Perplexity"):
        ppl1, ppl2 = calculate_text_perplexity(output)
        ppl_1st_half.append(ppl1)
        ppl_2nd_half.append(ppl2)
    return ppl_1st_half, ppl_2nd_half

def get_perplexity(generation_result_path, file_handle=None):
    df = pd.read_csv(generation_result_path)  # Load your DataFrame if needed
    df = df.head(100)  # get first 100 rows
    _print(f'Num of rows: {df.shape[0]}', file_handle=file_handle)
    col_to_eval = ['adaptive_watermarked_text'] # '2step-2models'
    for col in col_to_eval:
        # Calculate perplexity for each text in the DataFrame
        df['ppl_1st_half'], df['ppl_2nd_half'] = _get_perplexity(df, col, file_handle)
        _print(f'==> Avg ppl of first_half:', round(df[f'ppl_1st_half'].mean(), 2), file_handle=file_handle)
        _print(f'==> Avg ppl of second_half:', round(df[f'ppl_2nd_half'].mean(), 2), file_handle=file_handle)
        df.to_csv(generation_result_path, index=False)

if save_output:
    file_handle = open(output_file, 'w')
else:
    file_handle = None

# for path in generation_result_path_list:
for filename in os.listdir(folder_path):
    if filename.endswith('csv'):
        _print(filename, file_handle=file_handle)
        # result_path = f'{path[:-4]}_ppl.csv'
        path = os.path.join(folder_path, filename)
        get_perplexity(path, file_handle)
        _print(file_handle=file_handle)
