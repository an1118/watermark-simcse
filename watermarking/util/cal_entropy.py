import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import load_model, pre_process

# load watermarking model
watermark_model, watermark_tokenizer = load_model('meta-llama/Llama-3.1-8B-Instruct')


def next_token_entropy(input_text, model, tokenizer, device):
    input_ids = tokenizer.encode(input_text, return_tensors='pt', add_special_tokens=False).to(device)
    outputs = model(input_ids)
    probs = torch.nn.functional.softmax(outputs.logits[0, -1, :], dim=-1)
    mask = probs > 0
    entropy = -torch.sum(probs[mask] * torch.log(probs[mask]))
    return entropy

entropy_result = [[] for _ in range(200)]  # keep the entropy at each position
def cal_each_pos_entropy(propt, text):
    i