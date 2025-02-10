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


class QueryHead(nn.Module):
    def __init__(self, hidden_size):
        super(QueryHead, self).__init__()
        # Learnable query vector
        self.query = nn.Parameter(torch.randn(hidden_size))

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: Tensor of shape (batch_size, seq_length, hidden_size)
            attention_mask: Tensor of shape (batch_size, seq_length) with 1 for real tokens and 0 for padding tokens.
        Returns:
            sequence_embedding: Tensor of shape (batch_size, hidden_size)
        """
        # Compute raw attention scores
        attention_scores = torch.matmul(hidden_states, self.query)  # (batch_size, seq_length)

        # Apply attention mask (set padding positions to large negative value before softmax)
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e4)

        # Normalize attention scores
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_length)

        # Aggregate hidden states
        sequence_embedding = torch.matmul(attention_weights.unsqueeze(1), hidden_states).squeeze(1)  # (batch_size, hidden_size)

        return sequence_embedding
    

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)  # Key matrix W_K
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)  # Value matrix W_V
        self.query = nn.Parameter(torch.randn(hidden_dim))  # Learnable query vector

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Tensor of shape (B, L, H), the last hidden layer output.
            attention_mask: Tensor of shape (B, L) with 1 for real tokens and 0 for padding tokens.
        Returns:
            pooled_output: Tensor of shape (B, H), the pooled sequence embedding.
        """
        K = self.key_proj(x)  # (B, L, H)
        V = self.value_proj(x)  # (B, L, H)

        # Compute attention scores
        attn_scores = torch.matmul(K, self.query) / (K.shape[-1] ** 0.5)  # (B, L)

        # Apply attention mask (set padding tokens to large negative value)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e4)

        attn_weights = F.softmax(attn_scores, dim=1)  # (B, L)

        # Weighted sum of values
        pooled_output = torch.matmul(attn_weights.unsqueeze(1), V).squeeze(1)  # (B, H)
        # pooled_output = torch.sum(attn_weights.unsqueeze(-1) * V, dim=1)  # (B, H)

        return pooled_output
    
def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
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
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # original + cls.model_args.num_paraphrased + cls.model_args.num_negative
    num_sent = input_ids.size(1)

    # # input_ids: (bs, num_sent, len)
    # # random downsample one paraphrased sentence from sentences index in [1, cls.model_args.num_paraphrased-1]
    # # randomly generate one index from [1, cls.model_args.num_paraphrased-1]
    # # exclude tensor [:, index, :] from input_ids
    # paraphrased_idx = torch.randint(1, cls.model_args.num_paraphrased, (batch_size,))
    # mask = torch.ones_like(input_ids, dtype=torch.bool)
    # for i in range(batch_size):
    #     mask[i, paraphrased_idx[i], :] = False
    # input_ids = input_ids[mask].view(batch_size, num_sent - 1, -1)
    # attention_mask = attention_mask[mask].view(batch_size, num_sent - 1, -1)
    # num_paraphrased = cls.model_args.num_paraphrased - 1
    # num_sent -= 1
    # if token_type_ids is not None:
    #     token_type_ids = token_type_ids[mask].view(batch_size, num_sent - 1, -1)

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
        sequence_output = outputs[0]  # (bs*num_sent, seq_len, hidden)
        pooler_output = cls.classifier(sequence_output)  # (bs*num_sent, hidden)
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

        if cls.model_args.pooler_type in ['query', 'attention']:
            pooler_output = cls.pool(outputs.last_hidden_state, attention_mask)
        elif cls.model_args.pooler_type == 'last':
            pooler_output = last_token_pool(outputs.last_hidden_state, attention_mask)
        else:
            raise NotImplementedError
        # normalize embeddings
        pooler_output = F.normalize(pooler_output, p=2, dim=1)
        
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden_states)
    else:
        raise NotImplementedError
    
    # Mapping
    mapping_output = cls.map(pooler_output)
    # mapping_output = torch.tanh(mapping_output * 1000)  # approximate sign function
    pooler_output = mapping_output
        
    # Separate representation
    z1 = pooler_output[:, 0]
    z2_list = [pooler_output[:, i] for i in range(1, cls.model_args.num_paraphrased + 1)]
    if cls.model_args.num_negative == 0:
        z3_list = []
    else:
        z3_list = [pooler_output[:, i] for i in range(cls.model_args.num_paraphrased + 1, cls.model_args.num_paraphrased + cls.model_args.num_negative + 1)]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        raise NotImplementedError

    # straight-through estimate sign function
    def sign_ste(x):
        x_nogradient = x.detach()
        return x + x.sign() - x_nogradient
    
    # get sign value before calculating euclidean distance
    z1 = torch.tanh(z1 * 1000)
    z2_list = [torch.tanh(z2 * 1000) for z2 in z2_list]
    z3_list = [torch.tanh(z3 * 1000) for z3 in z3_list]

    # z1 = sign_ste(z1)
    # z2_list = [sign_ste(z2) for z2 in z2_list]
    # z3_list = [sign_ste(z3) for z3 in z3_list]

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
        raise NotImplementedError
        # z1_z2_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # (bs, bs)
        # z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))  # (bs, bs)
        # cos_sim = torch.cat([z1_z2_sim, z1_z3_cos], 1)  # (bs, bs*2)

        # labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
        # loss_fct = nn.CrossEntropyLoss()

        # # Calculate loss with hard negatives
        # # Note that weights are actually logits of weights
        # weights = torch.tensor(
        #     [[0.0] * z1_z2_sim.size(-1) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(cos_sim.size(0))]
        # ).to(cls.device)
    elif cls.model_args.loss_function_id == 2:
        z1_z1_cos = cls.sim(z1.unsqueeze(1), z1.unsqueeze(0))  # (bs, bs)
        z1_z1_cos_removed = remove_diagonal_elements(z1_z1_cos)  # (bs, bs-1)
        z1_z2_cos_list = [cls.sim(z1, z2).unsqueeze(1) for z2 in z2_list]  # [(bs, 1)] * num_paraphrased
        z1_z3_cos_list = [cls.sim(z1, z3).unsqueeze(1) for z3 in z3_list]  # [(bs,1)] * num_negative
        if z1_z3_cos_list:
            z1_z3_cos = torch.cat(z1_z3_cos_list, dim=1)  # (bs, num_negative)
        else:
            z1_z3_cos = torch.empty((z1.size(0), 0), device=z1.device)  # (bs, 0)

        loss_fct = nn.CrossEntropyLoss()
        loss_1 = 0
        for z1_z2_cos in z1_z2_cos_list:
            cos_sim = torch.cat([z1_z2_cos, z1_z1_cos_removed, z1_z3_cos], 1)  # (bs, bs+num_negative)
            # Calculate loss with hard negatives
            weights = torch.tensor(
                [[0.0] * z1_z2_cos.size(-1) + [0.0] * z1_z1_cos_removed.size(-1) + [z3_weight] * cls.model_args.num_negative for i in range(cos_sim.size(0))]
            ).to(cls.device)
            cos_sim = cos_sim + weights
            labels = torch.zeros(cos_sim.size(0)).long().to(cls.device)
            loss_1 += loss_fct(cos_sim, labels)
        loss_1 /= cls.model_args.num_paraphrased
    elif cls.model_args.loss_function_id == 3:
        raise NotImplementedError
        # z1_z1_cos = cls.sim(z1.unsqueeze(1), z1.unsqueeze(0))  # (bs, bs)
        # z1_z1_cos_removed = remove_diagonal_elements(z1_z1_cos)  # (bs, bs-1)
        # z1_z2_cos = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # (bs, bs)
        # z1_z3_cos = cls.sim(z1, z3).unsqueeze(1)  # (bs,1)

        # cos_sim = torch.cat([z1_z2_cos, z1_z1_cos_removed, z1_z3_cos], 1)  # (bs, 2*bs)

        # labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
        # loss_fct = nn.CrossEntropyLoss()

        # # Calculate loss with hard negatives
        # weights = torch.tensor(
        #     [[0.0] * z1_z2_cos.size(-1) + [0.0] * z1_z1_cos_removed.size(-1) + [z3_weight] for i in range(cos_sim.size(0))]
        # ).to(cls.device)
    elif cls.model_args.loss_function_id == 4:
        raise NotImplementedError
        # z1_z1_cos = cls.sim(z1.unsqueeze(1), z1.unsqueeze(0))  # (bs, bs)
        # z1_z1_cos_removed = remove_diagonal_elements(z1_z1_cos)  # (bs, bs-1)
        # z1_z2_cos = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # (bs, bs)
        # z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))  # (bs,bs)

        # cos_sim = torch.cat([z1_z2_cos, z1_z1_cos_removed, z1_z3_cos], 1)  # (bs, 3*bs-1)

        # labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
        # loss_fct = nn.CrossEntropyLoss()

        # # Calculate loss with hard negatives
        # weights = torch.tensor(
        #     [[0.0] * z1_z2_cos.size(-1) + [0.0] * z1_z1_cos_removed.size(-1) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(cos_sim.size(0))]
        # ).to(cls.device)
    else:
        raise NotImplementedError

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
    loss_3_list = [cls.sim(z1, z2).unsqueeze(1) for z2 in z2_list]  # [(bs, 1)] * num_paraphrased
    loss_3_tensor = torch.cat(loss_3_list, dim=1)  # (bs, num_paraphrased)
    loss_3 = loss_3_tensor.mean() * cls.model_args.temp
    # debug: 
    # loss_3 = loss_3[valid_for_loss3.bool()]

    # calculate loss_4: similarity between original and negative text
    if cls.model_args.num_negative == 0:
        loss_4 = None
    else:
        loss_4_list = [cls.sim(z1, z3).unsqueeze(1) for z3 in z3_list]  # [(bs, 1)] * num_negative
        loss_4_tensor = torch.cat(loss_4_list, dim=1)  # (bs, num_negative)
        loss_4 = loss_4_tensor.mean() * cls.model_args.temp

    loss = lambda_1 * loss_1 + lambda_2 * loss_2

    result = {
        'loss': loss,
        'loss_1': loss_1,
        'loss_2': loss_2,
        'loss_3': loss_3,
        'logits': cos_sim,
        'hidden_states': outputs.hidden_states,
        'attentions': outputs.attentions,
    }
    if cls.model_args.num_negative != 0:
        result['loss_4'] = loss_4
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

        if cls.model_args.pooler_type in ['query', 'attention']:
            pooler_output = cls.pool(outputs.last_hidden_state, attention_mask)
        elif cls.model_args.pooler_type == 'last':
            pooler_output = last_token_pool(outputs.last_hidden_state, attention_mask)
        else:
            raise NotImplementedError
        # normalize embeddings
        pooler_output = F.normalize(pooler_output, p=2, dim=1)
    else:
        raise NotImplementedError 


    # Mapping
    mapping_output = cls.map(pooler_output)
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

        if self.model_args.freeze_base:
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

        if self.model_args.pooler_type == 'query':
            self.pool = QueryHead(config.hidden_size)
        elif self.model_args.pooler_type == 'attention':
            self.pool = AttentionPooling(config.hidden_size)

        # if self.model_args.do_mlm:
        #     self.lm_head = RobertaLMHead(config)

        cl_init(self, config)
        self.map = SemanticModel(input_dim=1536)

        if self.model_args.freeze_base:
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

