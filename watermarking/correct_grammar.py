import torch
from sentence_transformers import SentenceTransformer
from model import SemanticModel
import json
import argparse
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import nltk
# nltk.download('punkt')
import os
import pandas as pd

from utils import load_model, pre_process, vocabulary_mapping
from watermark import Watermark



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load watermarking model
    watermark_model, watermark_tokenizer = load_model(args.watermark_model)
    # watermark_tokenizer.add_special_tokens({"pad_token":"<pad>"})
    # load measurement model
    measure_model, measure_tokenizer = load_model(args.measure_model)
    # load semantic embedding model
    embedding_model = SentenceTransformer(args.embedding_model).to(device)
    embedding_model.eval()
    # load semantic mapping model
    transform_model = SemanticModel()
    transform_model.load_state_dict(torch.load(args.semantic_model))
    transform_model.to(device)
    transform_model.eval()
    # load mapping list
    # vocabulary_size = watermark_tokenizer.vocab_size  # vacalulary size of LLM. Notice: OPT is 50272
    vocabulary_size = 128256
    mapping_list = vocabulary_mapping(vocabulary_size, 384, seed=66)
    # # load test dataset. Here we use C4 realnewslike dataset as an example. Feel free to use your own dataset.
    # data = "https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/realnewslike/c4-train.00000-of-00512.json.gz"
    # dataset = load_dataset('json', data_files=data)
    # dataset = pre_process(dataset, min_length=args.min_new_tokens, data_size=100)   # [{text0: 'text0', text: 'text'}]
    dataset = pd.read_csv(args.generation_file)
    dataset['watermarked_corrected_text'] = ''
    dataset['corrected_watermarked_score'] = ''


    watermark = Watermark(device=device,
                      watermark_tokenizer=watermark_tokenizer,
                      measure_tokenizer=measure_tokenizer,
                      watermark_model=watermark_model,
                      measure_model=measure_model,
                      embedding_model=embedding_model,
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
                      )
    

    for i in tqdm(range(len(dataset))):
        # sys_prompt1 = "Please correct any grammar errors and improve the consistency of the text with the fewest possible changes. Keep the original wording and structure whenever feasible. Respond with only the revised text."
        # sys_prompt2 = "Please correct any grammar errors and improve the coherence of the text while making minimal changes. Ensure the sentences flow smoothly, but preserve the original wording and structure as much as possible. Respond with only the revised text. Do not add additional things like 'Sure, here is the revised text'."
        # sys_prompt3 = "Please correct any grammar errors and improve the coherence of the text. Make necessary adjustments to enhance the flow and clarity but try to make minimal changes. Respond with only the revised text. Do not add additional things like 'Sure, here is the revised text'."
        sys_prompt4 = "Please correct any grammar errors and improve the coherence of the text. Make necessary adjustments to enhance the flow and clarity while preserving the original meaning as much as possible. Respond with only the revised text. Do not add additional things like 'Sure, here is the revised text'."
        sys_prompt = sys_prompt4
        watermarked_text = dataset.loc[i, 'adaptive_watermarked_text']
        messages = [
            {
                "role": "system", "content": sys_prompt,
            },
            {
                "role": "user",  "content": f"Here's the text: \n{watermarked_text}"
            },
        ]
        prompt = watermark.watermark_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        watermarked_corrected_text = watermark.generate_unwatermarked(prompt)

        # detect the grammar corrected text
        corrected_watermarked_score = watermark.detection(watermarked_corrected_text)

        dataset.loc[i, 'watermarked_corrected_text'] = watermarked_corrected_text
        dataset.loc[i, 'corrected_watermarked_score'] = corrected_watermarked_score
        dataset.to_csv(f'{args.output_dir}', index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate watermarked text!")
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
    parser.add_argument('--openai_api_key', default='', type=str, \
                        help='OpenAI API key.')
    parser.add_argument('--output_dir', default='outputs', type=str, \
                        help='Output directory.')
    parser.add_argument('--generation_file', type=str, help='File with watermarked generation.')
    
    args = parser.parse_args()
    main(args)



