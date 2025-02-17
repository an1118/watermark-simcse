import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoModel, AutoTokenizer
from types import SimpleNamespace
from model import SemanticModel
import argparse
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import nltk
# nltk.download('punkt')
import os

from clmodel import RobertaForCL
from utils import load_model, pre_process, vocabulary_mapping
from watermark_end2end import Watermark
from attack import paraphrase_attack, hate_attack
from models_cl import RobertaForCL

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load watermarking model
    watermark_model, watermark_tokenizer = load_model(args.watermark_model)
    # watermark_tokenizer.add_special_tokens({"pad_token":"<pad>"})
    # load measurement model
    measure_model, measure_tokenizer = load_model(args.measure_model)
    # load simcse finetuned embed_map model
    embed_map_model_path = args.embed_map_model
    embed_map_model_args = SimpleNamespace(
        temp = 0.05,
        pooler_type = 'cls',
        hard_negative_weight = args.hard_negative_weight,
        model_name_or_path = embed_map_model_path,
        do_mlm = False,
        mlp_only_train = False,
        freeze_embed=False,
    )

    embed_map_config = AutoConfig.from_pretrained(os.path.join(embed_map_model_path, "config.json"))
    embed_map_tokenizer = AutoTokenizer.from_pretrained(embed_map_model_path)
    embed_map_model = RobertaForCL.from_pretrained(
        embed_map_model_path,
        from_tf=bool(".ckpt" in embed_map_model_path),
        config=embed_map_config,
        model_args=embed_map_model_args,
        device_map='auto',
    )
    embed_map_model.eval()
    # load mapping list
    # vocabulary_size = watermark_tokenizer.vocab_size  # vacalulary size of LLM. Notice: OPT is 50272
    vocabulary_size = 128256
    mapping_list = vocabulary_mapping(vocabulary_size, 384, seed=66)
    # load test dataset. Here we use C4 realnewslike dataset as an example. Feel free to use your own dataset.
    # data_path = "https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/realnewslike/c4-train.00000-of-00512.json.gz"
    # # data_path = r"/mnt/data2/lian/projects/watermark/data/lfqa.json"
    # if 'c4' in data_path.lower():
    #     dataset = pre_process(data_path, min_length=args.min_new_tokens, data_size=50, num_of_sent=args.num_of_sent)   # [{text0: 'text0', text: 'text'}]
    # elif 'lfqa' in data_path.lower():
    #     dataset = pre_process(data_path, min_length=args.min_new_tokens, data_size=100)

    watermark = Watermark(device=device,
                      watermark_tokenizer=watermark_tokenizer,
                      measure_tokenizer=measure_tokenizer,
                      embed_map_tokenizer=embed_map_tokenizer,
                      watermark_model=watermark_model,
                      measure_model=measure_model,
                      embed_map_model=embed_map_model,
                      mapping_list=mapping_list,
                      alpha=args.alpha,
                      top_k=0,
                      top_p=0.9,
                      repetition_penalty=1.0,
                      no_repeat_ngram_size=0,
                      max_new_tokens=args.max_new_tokens,
                      min_new_tokens=args.min_new_tokens,
                      secret_string=args.secret_string,
                      measure_threshold=args.measure_threshold,
                      delta_0 = args.delta_0,
                      delta = args.delta,
                      )
        
    
    df = pd.read_csv(args.result_file)
    df['sim_ori_wm'], df['sim_ori_para'], df['sim_wm_para'] = '', '', ''
    for i in tqdm(range(df.shape[0])):
        original_text = df.loc[i, 'original_text']
        adaptive_watermarked_text = df.loc[i, 'adaptive_watermarked_text']
        paraphrased_watermarked_text = df.loc[i, 'paraphrased_watermarked_text']
        hate_watermarked_text = df.loc[i, 'hate_watermarked_text']

        df.loc[i, 'sim_ori_wm'] = watermark._get_l2_distance(original_text, adaptive_watermarked_text)
        df.loc[i, 'sim_ori_para'] = watermark._get_l2_distance(original_text, paraphrased_watermarked_text)
        df.loc[i, 'sim_wm_para'] = watermark._get_l2_distance(paraphrased_watermarked_text, adaptive_watermarked_text)
        
        df.to_csv(f'{args.output_dir}', index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate watermarked text!")
    parser.add_argument('--watermark_model', default='meta-llama/Llama-3.1-8B-Instruct', type=str, \
                        help='Main model, path to pretrained model or model identifier from huggingface.co/models. Such as mistralai/Mistral-7B-v0.1, facebook/opt-6.7b, EleutherAI/gpt-j-6b, etc.')
    parser.add_argument('--measure_model', default='gpt2-large', type=str, \
                        help='Measurement model.')
    parser.add_argument('--alpha', default=2.0, type=float, \
                        help='Entropy threshold. May vary based on different measurement model. Plase select the best alpha by yourself.')
    parser.add_argument('--max_new_tokens', default=300, type=int, \
                        help='Max tokens.')
    parser.add_argument('--min_new_tokens', default=200, type=int, \
                        help='Min tokens.')
    parser.add_argument('--secret_string', default='The quick brown fox jumps over the lazy dog', type=str, \
                        help='Secret string.')
    parser.add_argument('--measure_threshold', default=20, type=float, \
                        help='Measurement threshold.')
    parser.add_argument('--delta_0', default=0.2, type=float, \
                        help='Initial Watermark Strength, which could be smaller than --delta. May vary based on different watermarking model. Plase select the best delta_0 by yourself.')
    parser.add_argument('--delta', default=0.5, type=float, \
                        help='Watermark Strength. May vary based on different watermarking model. Plase select the best delta by yourself. A excessively high delta value may cause repetition.')
    parser.add_argument('--openai_api_key', default='', type=str, \
                        help='OpenAI API key.')
    parser.add_argument('--output_dir', default='outputs', type=str, \
                        help='Output directory.')
    parser.add_argument('--num_of_sent', default=2, type=int, \
                        help='Number of sentences to paraphrase.')
    parser.add_argument('--correct_grammar', default=False, type=bool, \
                        help='Correct grammar after adding watermark.')
    parser.add_argument('--embed_map_model', default='', type=str, \
                        help='End-to-end mapping model.')
    parser.add_argument('--hard_negative_weight', default=0, type=float, \
                        help='The **logit** of weight for hard negatives (only effective if hard negatives are used).')
    parser.add_argument('--result_file', default='', type=str, \
                        help='Result file.')

    args = parser.parse_args()
    main(args)



