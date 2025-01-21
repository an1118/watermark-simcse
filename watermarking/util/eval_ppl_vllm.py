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
    perplexity = exp(-mean(null))
    return perplexity

def _get_perplexity(df, column, file_handle=None):
    ppl_col = f'ppl_{column}'
    ppl_results = []
    _print(f'=====Column: {column}=====', file_handle=file_handle)
    outputs = model.generate(df[column].dropna(), sampling_params)
    for output in tqdm(outputs, desc="Calculating Perplexity"):
        ppl_results.append(calculate_text_perplexity(output))
    df[ppl_col] = ppl_results
    return df[ppl_col]

def get_perplexity(generation_result_path, file_handle=None):
    df = pd.read_csv(generation_result_path)  # Load your DataFrame if needed
    _print(f'Num of rows: {df.shape[0]}', file_handle=file_handle)
    col_to_eval = ['watermarked_corrected_text']
    if 'g_watermarked_text' in df.columns:
        col_to_eval.append('g_watermarked_text')
    if 'unwatermarked_text' in df.columns:
        col_to_eval.append('unwatermarked_text')
    if 'adaptive_watermarked_text' in df.columns:
        col_to_eval.append('adaptive_watermarked_text')
    if 'watermarked_text' in df.columns:
        col_to_eval.append('watermarked_text')
    for col in col_to_eval:
        # Calculate perplexity for each text in the DataFrame
        df[f'ppl_{col}'] = _get_perplexity(df, col, file_handle)
        _print(f'==> Avg ppl of {col}:', round(df[f'ppl_{col}'].mean(), 2), file_handle=file_handle)
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
