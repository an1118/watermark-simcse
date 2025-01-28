import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from datasets import load_dataset
import nltk
import json
import pandas as pd

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    model.eval()
    return model, tokenizer

def vocabulary_mapping(vocab_size, model_output_dim, seed=66):
    random.seed(seed)
    return [random.randint(0, model_output_dim-1) for _ in range(vocab_size)]

def pre_process(data_path, min_length, data_size=500, num_of_sent=None):
    data = []
    if 'onebatch' in data_path.lower():
        dataset = pd.read_csv(data_path)
        dataset = dataset['original'].tolist()
        for text in dataset:
            text = text.strip()
            data.append({'text': text})
            if len(data) ==  data_size:
                break
    elif 'c4' in data_path.lower():
        dataset = load_dataset('json', data_files=data_path)
        for text in dataset['train']['text']:
            text0 = text.split()[0:min_length]
            if len(text0) >= min_length:
                text0 = ' '.join(text0)
                text = ' '.join(nltk.sent_tokenize(text)[:num_of_sent])
                data.append({'text0': text0, 'text': text})
            else:
                pass
            
            if len(data) ==  data_size:
                break
    elif 'lfqa' in data_path.lower():
        with open(data_path, 'r') as f:
            for line in f:
                tmp = json.loads(line.strip())
                text = tmp['gold_completion']
                data.append({'text': text})
                if len(data) ==  data_size:
                    break

    return data
