import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification, RobertaClassificationHead, RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel, Qwen2Model
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        out = out + x 
        return out

class SemanticModel(nn.Module):
    def __init__(self, num_layers=2, input_dim=768, hidden_dim=512, output_dim=384):
        super(SemanticModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers):
            self.layers.append(ResidualBlock(hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        
        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


class RobertaClassificationHeadForEmbedding(RobertaClassificationHead):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        # x = self.out_proj(x)
        return x


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    # if 'roberta' in cls.model_args.model_name_or_path.lower():
    #     cls.pooler_type = cls.model_args.pooler_type
    #     cls.pooler = Pooler(cls.model_args.pooler_type)
    #     if cls.model_args.pooler_type == "cls":
    #         cls.mlp = MLPLayer(config)
    #         if cls.model_args.freeze_embed:
    #             for param in cls.mlp.parameters():
    #                 param.requires_grad = False
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def cl_forward(cls,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    lambda_1=1.0,
    lambda_2=1.0,
):
    # debug: get the edit distance between original & paraphrased
    import Levenshtein

    def apply_attention_mask(tensor, attention_mask):
        """
        Returns:
            list: A list of tensors, where each tensor contains the valid tokens for one batch entry.
        """
        batch_size = tensor.size(0)
        result = []
        for i in range(batch_size):
            # Masked tokens for this batch entry
            valid_tokens = tensor[i][attention_mask[i].bool()]
            result.append(valid_tokens)
        return result
    
    original = input_ids[:,0,:].squeeze(1)
    original_mask = attention_mask[:,0,:].squeeze(1)
    original = apply_attention_mask(original, original_mask)
    paraphrased = input_ids[:,1,:].squeeze(1)
    paraphrased_mask = attention_mask[:,1,:].squeeze(1)
    paraphrased = apply_attention_mask(paraphrased, paraphrased_mask)
    # compute edit distance
    valid_for_loss3 = []
    for i in range(len(original)):
        edit_distance = Levenshtein.distance(original[i].tolist(), paraphrased[i].tolist()) / len(original[i])
        valid_for_loss3.append(1 if edit_distance >= 0.2 else 0)
    valid_for_loss3 = torch.tensor(valid_for_loss3)
    # finish debug

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
    
    if 'roberta' in cls.model_args.model_name_or_path:
        # Get raw embeddings
        outputs = cls.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

        # MLM auxiliary objective
        if mlm_input_ids is not None:
            mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
            mlm_outputs = cls.roberta(
                mlm_input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
            )

        # Pooling
        sequence_output = outputs[0]
        pooler_output = cls.classifier(sequence_output)
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    elif 'qwen2' in cls.model_args.model_name_or_path.lower():
        def last_token_pool(last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

        outputs = cls.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

        pooler_output = last_token_pool(outputs.last_hidden_state, attention_mask)
        # normalize embeddings
        pooler_output = F.normalize(pooler_output, p=2, dim=1)
        
        # last_unmasked_token_idx = (attention_mask == 1).long().sum(dim=1) - 1
        # # last_unmasked_token_idx = last_unmasked_token_idx.tolist()
        # last_hidden_state = outputs['last_hidden_state']  # (bs*num_sent, seq_len, hidden_states)
        # pooler_output = last_hidden_state[torch.arange(last_hidden_state.size(0)), last_unmasked_token_idx] # last unmasked token's hiddent state
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden_states)
    else:
        raise NotImplementedError
    
    # Mapping
    mapping_output = cls.map(pooler_output)
    # mapping_output = torch.tanh(mapping_output * 1000)  # approximate sign function
    pooler_output = mapping_output
        
    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    # straight-through estimate sign function
    def sign_ste(x):
        x_nogradient = x.detach()
        return x + x.sign() - x_nogradient
    
    # get sign value before calculating euclidean distance
    z1 = torch.tanh(z1 * 1000)
    z2 = torch.tanh(z2 * 1000)
    z3 = torch.tanh(z3 * 1000)

    # z1 = sign_ste(z1)
    # z2 = sign_ste(z2)
    # z3 = sign_ste(z3)

    # Compute contrastive loss
    def remove_diagonal_elements(input_tensor):
        """
        Removes the diagonal elements from a square matrix (bs, bs) 
        and returns a new matrix of size (bs, bs-1).
        """
        if input_tensor.size(0) != input_tensor.size(1):
            raise ValueError("Input tensor must be square (bs, bs).")
        
        bs = input_tensor.size(0)
        mask = ~torch.eye(bs, dtype=torch.bool, device=input_tensor.device)  # Mask for non-diagonal elements
        output_tensor = input_tensor[mask].view(bs, bs - 1)  # Reshape into (bs, bs-1)
        return output_tensor

    z3_weight = cls.model_args.hard_negative_weight

    if cls.model_args.loss_function_id == 1:
        z1_z2_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # (bs, bs)
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))  # (bs, bs)
        cos_sim = torch.cat([z1_z2_sim, z1_z3_cos], 1)  # (bs, bs*2)

        labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
        loss_fct = nn.CrossEntropyLoss()

        # Calculate loss with hard negatives
        # Note that weights are actually logits of weights
        weights = torch.tensor(
            [[0.0] * z1_z2_sim.size(-1) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(cos_sim.size(0))]
        ).to(cls.device)
    elif cls.model_args.loss_function_id == 2:
        z1_z1_cos = cls.sim(z1.unsqueeze(1), z1.unsqueeze(0))  # (bs, bs)
        z1_z1_cos_removed = remove_diagonal_elements(z1_z1_cos)  # (bs, bs-1)
        z1_z2_cos = cls.sim(z1, z2).unsqueeze(1)  # (bs, 1)
        z1_z3_cos = cls.sim(z1, z3).unsqueeze(1)  # (bs,1)

        cos_sim = torch.cat([z1_z2_cos, z1_z1_cos_removed, z1_z3_cos], 1)  # (bs, bs+1)

        labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
        labels = torch.zeros_like(labels).to(cls.device)
        loss_fct = nn.CrossEntropyLoss()

        # Calculate loss with hard negatives
        weights = torch.tensor(
            [[0.0] * z1_z2_cos.size(-1) + [0.0] * z1_z1_cos_removed.size(-1) + [z3_weight] for i in range(cos_sim.size(0))]
        ).to(cls.device)
    elif cls.model_args.loss_function_id == 3:
        z1_z1_cos = cls.sim(z1.unsqueeze(1), z1.unsqueeze(0))  # (bs, bs)
        z1_z1_cos_removed = remove_diagonal_elements(z1_z1_cos)  # (bs, bs-1)
        z1_z2_cos = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # (bs, bs)
        z1_z3_cos = cls.sim(z1, z3).unsqueeze(1)  # (bs,1)

        cos_sim = torch.cat([z1_z2_cos, z1_z1_cos_removed, z1_z3_cos], 1)  # (bs, 2*bs)

        labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
        loss_fct = nn.CrossEntropyLoss()

        # Calculate loss with hard negatives
        weights = torch.tensor(
            [[0.0] * z1_z2_cos.size(-1) + [0.0] * z1_z1_cos_removed.size(-1) + [z3_weight] for i in range(cos_sim.size(0))]
        ).to(cls.device)
    elif cls.model_args.loss_function_id == 4:
        z1_z1_cos = cls.sim(z1.unsqueeze(1), z1.unsqueeze(0))  # (bs, bs)
        z1_z1_cos_removed = remove_diagonal_elements(z1_z1_cos)  # (bs, bs-1)
        z1_z2_cos = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # (bs, bs)
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))  # (bs,bs)

        cos_sim = torch.cat([z1_z2_cos, z1_z1_cos_removed, z1_z3_cos], 1)  # (bs, 3*bs-1)

        labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
        loss_fct = nn.CrossEntropyLoss()

        # Calculate loss with hard negatives
        weights = torch.tensor(
            [[0.0] * z1_z2_cos.size(-1) + [0.0] * z1_z1_cos_removed.size(-1) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(cos_sim.size(0))]
        ).to(cls.device)

    cos_sim = cos_sim + weights
    loss_1 = loss_fct(cos_sim, labels)

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss_1 = loss_1 + cls.model_args.mlm_weight * masked_lm_loss
    
    # Calculate loss for uniform perturbation and unbiased token preference
    def sign_loss(x):
        # smooth_sign = sign_ste(x)
        row = torch.abs(torch.mean(torch.mean(x, dim=0)))
        col = torch.abs(torch.mean(torch.mean(x, dim=1)))
        return (row + col)/2

    loss_2 = sign_loss(z1)

    # calculate loss_3: similarity between original and paraphrased text
    loss_3 = cls.sim(z1, z2)  # (bs)

    # debug: 
    loss_3 = loss_3[valid_for_loss3.bool()]
    loss_3 = - loss_3.mean()

    # calculate loss_4: similarity between original and negative text
    loss_4 = cls.sim(z1, z3)  # (bs)
    loss_4 = loss_4.mean()


    loss = lambda_1 * loss_1 + lambda_2 * loss_2

    result = {
        'loss': loss,
        'loss_1': loss_1,
        'loss_2': loss_2,
        'loss_3': loss_3,
        'loss_4': loss_4,
        'logits': cos_sim,
        'hidden_states': outputs.hidden_states,
        'attentions': outputs.attentions,
    }
    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return result


def sentemb_forward(
    cls,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    lambda_1=1.0,
    lambda_2=1.0,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    if 'roberta' in cls.model_args.model_name_or_path:
        outputs = cls.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )
        sequence_output = outputs[0]
        pooler_output = cls.classifier(sequence_output)
    elif 'qwen2' in cls.model_args.model_name_or_path.lower():
        def last_token_pool(last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

        outputs = cls.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        pooler_output = last_token_pool(outputs.last_hidden_state, attention_mask)
        # normalize embeddings
        pooler_output = F.normalize(pooler_output, p=2, dim=1)
    else:
        raise NotImplementedError 


    # Mapping
    mapping_output = cls.map(pooler_output)
    # mapping_output = torch.tanh(mapping_output * 1000)  # approximate sign function  # todo: delete
    pooler_output = mapping_output
        

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class RobertaForCL(RobertaForSequenceClassification):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]

        self.classifier = RobertaClassificationHeadForEmbedding(config)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        self.map = SemanticModel(input_dim=768)
        cl_init(self, config)

        if self.model_args.freeze_embed:
            # Freeze RoBERTa encoder parameters
            for param in self.roberta.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = False
        
        # Initialize weights and apply final processing
        self.post_init()

    def initialize_mlp_weights(self, pretrained_model_state_dict):
        """
        Initialize MLP weights using the pretrained classifier's weights.
        """
        self.mlp.dense.weight.data = pretrained_model_state_dict.classifier.dense.weight.data.clone()
        self.mlp.dense.bias.data = pretrained_model_state_dict.classifier.dense.bias.data.clone()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )

class Qwen2ForCL(Qwen2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.model = Qwen2Model(config)

        # if self.model_args.do_mlm:
        #     self.lm_head = RobertaLMHead(config)

        cl_init(self, config)
        self.map = SemanticModel(input_dim=1536)

        if self.model_args.freeze_embed:
            # Freeze Qwen parameters
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )

