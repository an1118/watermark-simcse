import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from model import SemanticModel
# import json
import argparse
# from datasets import load_dataset
from tqdm import tqdm
import nltk
# nltk.download('punkt')
import os
import pandas as pd
from random import shuffle

from utils import load_model, vocabulary_mapping
from watermark import Watermark
from api import call_chatgpt_api
import openai
from tenacity import RetryError

import pdb

def shuffle_attack(text):
    # sentence-level tokenize
    text_list = nltk.sent_tokenize(text)
    shuffle(text_list)

    prompt_0 = '''Please help transform the following list of sentences into a well-structured paragraph. \nEnsure that each sentence is included, the original meaning is preserved, and the overall flow is logical and coherent. Do not add any additional information beyond the provided sentences. \nHere is the list of sentences: '''
    prompt_1 = text_list
    prompt = f'{prompt_0}\n{prompt_1}\n'

    messages = [{"role": "user", "content": prompt}]
    max_tokens = 300
    keep_call = True
    cnt = 0
    while(keep_call):
        # Make the API call
        response = call_chatgpt_api(messages, max_tokens)
        output_text = response.choices[0].message.content
        if output_text:  # not None
            keep_call = False
            return output_text
        else:
            cnt += 1
            if cnt <= 10:
                print('===try calling api one more time===')
            else:
                print('API call failed!')
                print(text)
                return None

def paraphrase_attack(text):
    prompt = 'You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences. \n Ensure that the final output contains the same information as the original text and has roughly the same length. \n Do not leave out any important details when rewriting in your own voice. Do not start your response by \'Sure\' or anything similar, simply output the paraphrased text directly.'

    messages = [
        {
            "role": "system", "content": prompt,
        },
        {
            "role": "user",  "content": text
        },
    ]

    max_tokens = 300
    keep_call = True
    cnt = 0
    while(keep_call):
        # Make the API call
        try:
            response = call_chatgpt_api(messages, max_tokens, model='GPT-4o')
        except RetryError as e:
            print(e)
            return None
        output_text = response.choices[0].message.content
        if output_text:  # not None
            keep_call = False
            return output_text
        else:
            cnt += 1
            if cnt <= 10:
                print('===try calling api one more time===')
            else:
                print('API call failed!')
                print(text)
                return None

def hate_attack(text):
    prompt = f"Please take the following neutral text and modify it to reflect a more negative or aggressive tone while making the fewest changes possible. Aim to maintain the original structure and content as much as you can, but shift the implications to convey a more hostile attitude. Do not start your response by \'Sure\' or anything similar, simply output the paraphrased text directly."
    # prompt = '''Please modify the following text to reflect a more negative or aggressive tone. \nRequirements: \n- Add negative or aggressive words accounting for around 10% of the total text;\n- Maintain the original structure as closely as possible;\n- Shift the underlying implications to convey a more hostile attitude.'''
    # prompt = f"Make no more than 5 word-level changes to the text to change its sentiment. Here is the original text:\n{text}\nRevised version:"
    messages = [
        {
            "role": "system", "content": prompt,
        },
        {
            "role": "user",  "content": text
        },
    ]
    max_tokens = 300
    keep_call = True
    cnt = 0
    while(keep_call):
        # Make the API call
        try:
            response = call_chatgpt_api(messages, max_tokens, model='GPT-4o')
        except RetryError as e:
            print(e)
            return None
        output_text = response.choices[0].message.content
        if output_text:  # not None
            keep_call = False
            return output_text
        else:
            cnt += 1
            if cnt <= 10:
                print('===try calling api one more time===')
            else:
                print('API call failed!')
                print(text)
                return None

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load watermarking model
    watermark_model, watermark_tokenizer = load_model(args.watermark_model)
    # load measurement model
    measure_model, measure_tokenizer = load_model(args.measure_model)
    # load semantic embedding model
    embedding_model = SentenceTransformer(args.embedding_model).to(device)
    embedding_model.eval()
    # load sentiment embedding model
    sentiment_model_name = r"/mnt/data2/lian/projects/watermark/SimCSE/result/my-sup-simcse-roberta-base-2gpu-64batch/checkpoint-epoch1"
    # sentiment_model_name = r"/mnt/data2/lian/projects/watermark/SimCSE/result/my-sup-simcse-roberta-base-sentiment/checkpoint-125"
    sentiment_model = AutoModel.from_pretrained(sentiment_model_name)
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    # mannually initialize mlp layer
    state_dict = torch.load(f'{sentiment_model_name}/pytorch_model.bin')
    sentiment_model.pooler.dense.weight.data = state_dict['mlp.dense.weight'].clone()
    sentiment_model.pooler.dense.bias.data = state_dict['mlp.dense.bias'].clone()
    sentiment_model = sentiment_model.to(device)
    sentiment_model.eval()
    del state_dict
    print('Mannually initialize mlp layer!')
    # load semantic mapping model
    transform_model = SemanticModel()
    transform_model.load_state_dict(torch.load(args.semantic_model))
    transform_model.to(device)
    transform_model.eval()
    # load mapping list
    # vocalulary_size = watermark_tokenizer.vocab_size  # vacalulary size of LLM. Notice: OPT is 50272
    if args.watermark_model == 'facebook/opt-6.7b':
        vocalulary_size = 50272
    elif 'Llama-3.1-8B-Instruct'.lower() in args.watermark_model.lower():
        vocalulary_size = 128256  # vacalulary size of LLM
    mapping_list = vocabulary_mapping(vocalulary_size, 384, seed=66)

    watermark = Watermark(device=device,
                      watermark_tokenizer=watermark_tokenizer,
                      measure_tokenizer=measure_tokenizer,
                      sentiment_tokenizer=sentiment_tokenizer,
                      watermark_model=watermark_model,
                      measure_model=measure_model,
                      embedding_model=embedding_model,
                      sentiment_model=sentiment_model,
                      transform_model=transform_model,
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
                      beta=args.beta,
                      )
    # load result file
    df = pd.read_csv(args.result_path)
    df[f'{args.attack_type}_watermarked_text'] = None
    df[f'{args.attack_type}_watermarked_text_score'] = None
    
    for index in tqdm(range(len(df))):
        row = df.iloc[index]
        w_text = row['adaptive_watermarked_text']
        if args.attack_type == 'shuffle':
            p_w_text = shuffle_attack(w_text)
        elif args.attack_type == 'hate':
            p_w_text = hate_attack(w_text)
        elif args.attack_type == 'paraphrase':
            p_w_text = paraphrase_attack(w_text)
        try:
            p_w_text_score = watermark.detection(p_w_text)
        except Exception as e:
            print(p_w_text)
            print(e)
            p_w_text_score = ''
        df.at[index, f'{args.attack_type}_watermarked_text'] = p_w_text
        df.at[index, f'{args.attack_type}_watermarked_text_score'] = p_w_text_score
        
        df.to_csv(f'{args.output_dir}', index=False)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate watermarked text!")
    parser.add_argument('--result_path', default='', type=str, \
                        help='Result file path that includes watermarked text generation.')
    parser.add_argument('--output_dir', default='output', type=str, \
                        help='Output directory.')
    parser.add_argument('--watermark_model', default='meta-llama/Llama-3.1-8B-Instruct', type=str, \
                        help='Main model, path to pretrained model or model identifier from huggingface.co/models. Such as mistralai/Mistral-7B-v0.1, facebook/opt-6.7b, EleutherAI/gpt-j-6b, etc.')
    parser.add_argument('--measure_model', default='gpt2-large', type=str, \
                        help='Measurement model.')
    parser.add_argument('--embedding_model', default='sentence-transformers/all-mpnet-base-v2', type=str, \
                        help='Semantic embedding model.')
    parser.add_argument('--semantic_model', default='model/semantic_mapping_model.pth', type=str, \
                        help='Load semantic mapping model parameters.')
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
    parser.add_argument('--attack_type', default='', type=str, \
                        help='Attack type: shuffle, hate')
    parser.add_argument('--beta', default=0.5, type=float, \
                        help='Strength of global sentiment embedding, should be within [0, 1.0]')

    args = parser.parse_args()
    main(args)



