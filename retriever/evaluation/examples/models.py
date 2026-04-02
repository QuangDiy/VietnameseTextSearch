import torch.nn.functional as F
import torch
from torch import Tensor

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def encode(model, input_ids, attention_mask, **kwargs):
    model_output = model(input_ids, attention_mask, return_dict=True)

    embeddings = mean_pooling(model_output, attention_mask)

    return F.normalize(embeddings, p=2, dim=1)

MODEL_WRAPPERS = {
    "phobert": lambda model, input_ids, attention_mask, **kwargs: model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True).pooler_output,
    "mean_pooling": lambda model, input_ids, attention_mask, **kwargs: F.normalize(mean_pooling(model(input_ids=input_ids, attention_mask=attention_mask), attention_mask), p=2, dim=1),
    "cls_pooling": lambda model, input_ids, attention_mask, **kwargs: model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state[:,0],
}