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

from utils import load_model, pre_process, vocabulary_mapping
from watermark_end2end import Watermark
from attack import paraphrase_attack, hate_attack
from models_cl import RobertaForCL, Qwen2ForCL

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
        model_name_or_path = embed_map_model_path,
        temp = 0.05,
        pooler_type = 'cls',
        hard_negative_weight = args.hard_negative_weight,
        do_mlm = False,
        mlp_only_train = False,
        freeze_embed=False,
    )

    embed_map_config = AutoConfig.from_pretrained(os.path.join(embed_map_model_path, "config.json"))
    embed_map_tokenizer = AutoTokenizer.from_pretrained(embed_map_model_path)
    if 'roberta' in embed_map_model_path.lower():
        embed_map_model = RobertaForCL.from_pretrained(
            embed_map_model_path,
            from_tf=bool(".ckpt" in embed_map_model_path),
            config=embed_map_config,
            model_args=embed_map_model_args,
            device_map='auto',
        )
    elif 'qwen2' in embed_map_model_path.lower():
        embed_map_model = Qwen2ForCL.from_pretrained(
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
    # data_path = r"/mnt/data2/lian/projects/watermark/data/lfqa.json"
    data_path = args.data_path
    if 'onebatch' in data_path.lower():
        dataset = pre_process(data_path, min_length=args.min_new_tokens, data_size=128, num_of_sent=args.num_of_sent)
    elif 'c4' in data_path.lower():
        dataset = pre_process(data_path, min_length=args.min_new_tokens, data_size=50, num_of_sent=args.num_of_sent)   # [{text0: 'text0', text: 'text'}]
    elif 'lfqa' in data_path.lower():
        dataset = pre_process(data_path, min_length=args.min_new_tokens, data_size=100)

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
        
    finished = 0
    if os.path.exists(f'{args.output_file}'):
        df = pd.read_csv(f'{args.output_file}')
        finished = df.shape[0]
        print(f'===skiped first {finished} rows.===')
    else:
        output_folder = os.path.dirname(args.output_file)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        df = pd.DataFrame(columns=['text_id', 'original_text', 'adaptive_watermarked_text', 'watermarked_corrected_text', 'paraphrased_watermarked_text', 'hate_watermarked_text', 'human_score', 'adaptive_watermarked_text_score', 'corrected_watermarked_score', 'paraphrased_watermarked_text_score', 'hate_watermarked_text_score'])

    watermark_rate = []  # debug
    for i in tqdm(range(finished, len(dataset))):
        # sys_prompt = 'Paraphrase the following text while preserving the original meaning and tone. Use a natural variation in word choices and sentence structure, but ensure the meaning remains unchanged. Avoid being overly repetitive or predictable. Do not start your response by \'Sure\' or anything similar, simply output the paraphrased text directly.'
        sys_prompt = '''Rewrite the following text by changing the wording but keeping all the facts exactly the same. The new version should match the original sentiment very closely, not just in positivity or negativity but also in the nuanced expression of emotions. Aim to minimize any changes that could alter the text's subtle tone, and ensure the paraphrased version is close in meaning to the original. Do not start your response by \'Sure\' or anything similar, simply output the paraphrased text directly.'''
        text = ' '.join(nltk.sent_tokenize(dataset[i]['text'])[:args.num_of_sent])
        messages = [
            {
                "role": "system", "content": sys_prompt,
            },
            {
                "role": "user",  "content": text
            },
        ]
        prompt = watermark.watermark_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # unwatermarked_text = watermark.generate_unwatermarked(prompt)
        watermarked_text = watermark.generate_watermarked(prompt, text)

        if args.correct_grammar:  # do grammar correction
            grammar_prompt = "Please correct any grammar errors and improve the coherence of the text. Make necessary adjustments to enhance the flow and clarity while preserving the original meaning as much as possible. Respond with only the revised text. Do not add additional things like 'Sure, here is the revised text'."
            messages = [
                {
                    "role": "system", "content": grammar_prompt,
                },
                {
                    "role": "user",  "content": f"Here's the text: \n{watermarked_text}"
                },
            ]
            grammar_prompt = watermark.watermark_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            watermarked_corrected_text = watermark.generate_unwatermarked(grammar_prompt)
        else:
            watermarked_corrected_text = ''
        
        # attack
        paraphrased_watermarked_text = paraphrase_attack(watermarked_text)
        hate_watermarked_text = hate_attack(watermarked_text)

        # detections
        human_score = watermark.detection(text)
        adaptive_watermarked_text_score = watermark.detection(watermarked_text)
        if args.correct_grammar:  # do grammar correction
            corrected_watermarked_score = watermark.detection(watermarked_corrected_text)
        else:
            corrected_watermarked_score = None
        paraphrased_watermarked_text_score = watermark.detection(paraphrased_watermarked_text) if paraphrased_watermarked_text is not None else ''
        hate_watermarked_text_score = watermark.detection(hate_watermarked_text) if hate_watermarked_text is not None else ''

        data = {
            'text_id': [i],
            'original_text': [text],
            # 'unwatermarked_text': [unwatermarked_text],
            'adaptive_watermarked_text': [watermarked_text],
            'watermarked_corrected_text': [watermarked_corrected_text],
            'paraphrased_watermarked_text': [paraphrased_watermarked_text],
            'hate_watermarked_text': [hate_watermarked_text],
            'human_score': [human_score],
            'adaptive_watermarked_text_score': [adaptive_watermarked_text_score],
            'corrected_watermarked_score': [corrected_watermarked_score],
            'paraphrased_watermarked_text_score': [paraphrased_watermarked_text_score],
            'hate_watermarked_text_score': [hate_watermarked_text_score],
        }
        df  = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
        df.to_csv(f'{args.output_file}', index=False)
        watermark_rate.append((watermark.num_watermarked_token, watermark.num_token))
        watermark.num_watermarked_token, watermark.num_token = 0, 0
    
    tmp = [w / t for w, t in watermark_rate]
    awr = sum(tmp) / len(tmp)
    print(f'=== Average watermarked rate: {awr}')



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
    parser.add_argument('--output_file', default='outputs', type=str, \
                        help='Output directory.')
    parser.add_argument('--num_of_sent', default=2, type=int, \
                        help='Number of sentences to paraphrase.')
    parser.add_argument('--correct_grammar', default=False, type=bool, \
                        help='Correct grammar after adding watermark.')
    parser.add_argument('--embed_map_model', default='', type=str, \
                        help='End-to-end mapping model.')
    parser.add_argument('--hard_negative_weight', default=0, type=float, \
                        help='The **logit** of weight for hard negatives (only effective if hard negatives are used).')
    parser.add_argument('--data_path', default='', type=str, \
                        help='Data Path.')

    args = parser.parse_args()
    main(args)



