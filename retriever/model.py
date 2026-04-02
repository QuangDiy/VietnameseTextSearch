from torch import nn
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from .loss import Losses
from .triloss import TriLosses

def change_dropout(model, p):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = p

class PairLoss(nn.Module):
    def __init__(
        self,
        model,
        model_wrapper=lambda model,input_ids,attention_mask: model(input_ids,attention_mask),
        loss_fn=Losses.ELOSS,
        dropout_list = (0.1, 0.2),
    ):
        super(PairLoss, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.model_wrapper = model_wrapper
        self.dropout_list = dropout_list

    def _get_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def forward(self, query_ids, query_attention_mask, context_ids, context_attention_mask, logger=False):
        device = self._get_device()
        change_dropout(self.model, self.dropout_list[0])
        query_embedding = self.model_wrapper(self.model, query_ids.to(device), query_attention_mask.to(device))

        change_dropout(self.model, self.dropout_list[1])
        context_embedding = self.model_wrapper(self.model, context_ids.to(device), context_attention_mask.to(device))
        output = (self.loss_fn(query_embedding, context_embedding), )

        if logger:
            output += (query_embedding, context_embedding,)

        return output
    
    def push_to_hub(self, *args, **kwargs):
        print('push to hub')
        self.model.push_to_hub(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        print('save pretrained')
        self.model.save_pretrained(*args, **kwargs)

    @classmethod
    def create_from_pretrained(cls,
                               model_name_or_path,
                               tokenizer_name_or_path=None,
                               token=None,
                               push_to_hub_id=None,
                               model_wrapper=lambda x,y,z: x(y,z),
                               loss_fn=Losses.ELOSS,
                               dropout_list = (0.1, 0.5),
                               ):
        """Create an instance of this class from a pre-trained model."""
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, token=token)
        
        if tokenizer_name_or_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, token=token)
            try:
                tokenizer.padding_side = 'right'
            except:
                print('WARNING: tokenizer padding side not found, we cannot set padding side right, use default padding side')

            # Check if special tokens are in the tokenizer
            if '<|query|>' not in tokenizer.vocab:
                print('WARNING: add new special token <|query|> to tokenizer')
                tokenizer.add_tokens('<|query|>')
                if '<|query|>' not in tokenizer.vocab:
                    print('WARNING: cannot add new special token')
                else:
                    print('WARNING: add new special token successfully')

            model.resize_token_embeddings(len(tokenizer))
            emb = getattr(model.embeddings, 'tok_embeddings', None) or getattr(model.embeddings, 'word_embeddings', None)
            print(f'WARNING: model embedding: {emb}')
            print(f'WARNING: tokenizer size: {len(tokenizer)}')
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=token)

        if push_to_hub_id is not None:
            model.push_to_hub(push_to_hub_id, token=token, private=True)
            tokenizer.push_to_hub(push_to_hub_id, token=token, private=True)

        return (cls(model,
                   model_wrapper=model_wrapper,
                   loss_fn=loss_fn,
                   dropout_list = dropout_list), tokenizer, )
    
class SimilarityLoss(nn.Module):
    def __init__(self,
                 model,
                 model_wrapper=lambda model,input_ids,attention_mask: model(input_ids,attention_mask),
                 loss_fn=TriLosses.ELOSS,
                 dropout_list = (0.1, 0.15),
                 ):
        super(SimilarityLoss, self).__init__()
        self.model = model
        self.model_wrapper = model_wrapper
        self.loss_fn = loss_fn
        self.dropout_list = dropout_list

    def _get_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def forward(self,
                query_ids,
                query_attention_mask,
                positive_ids=None,
                positive_attention_mask=None,
                negative_ids=None,
                negative_attention_mask=None,
                logger=False
                ):
        device = self._get_device()
        change_dropout(self.model, self.dropout_list[0])
        query_embedding = self.model_wrapper(self.model, query_ids.to(device), query_attention_mask.to(device))

        if positive_ids is not None:
            positive_embedding = self.model_wrapper(self.model, positive_ids.to(device), positive_attention_mask.to(device))
            if negative_ids is not None:
                negative_embedding = self.model_wrapper(self.model, negative_ids.to(device), negative_attention_mask.to(device))
            else:
                negative_embedding = None
        else:
            change_dropout(self.model, self.dropout_list[1])
            positive_embedding = self.model_wrapper(self.model, query_ids.to(device), query_attention_mask.to(device))

        output = (self.loss_fn(query_embedding, positive_embedding, negative_embedding), )

        if logger:
            output += (query_embedding, positive_embedding, negative_embedding,)

        return output
    
    def push_to_hub(self, *args, **kwargs):
        self.model.push_to_hub(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)

    @classmethod
    def create_from_pretrained(cls,
                               model_name_or_path,
                               tokenizer_name_or_path=None,
                               token=None,
                               push_to_hub_id=None,
                               model_wrapper=lambda x,y,z: x(y,z),
                               loss_fn=TriLosses.ELOSS,
                               dropout_list = (0.1, 0.5),
                               ):
        """Create an instance of this class from a pre-trained model."""
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, token=token)

        if tokenizer_name_or_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, token=token)
            try:
                tokenizer.padding_side = 'right'
            except:
                print('WARNING: tokenizer padding side not found, we cannot set padding side right, use default padding side')

            # Check if special tokens are in the tokenizer
            if '<|query|>' not in tokenizer.get_vocab():
                print('WARNING: add new special token <|query|> to tokenizer')
                tokenizer.add_tokens('<|query|>')
                if '<|query|>' not in tokenizer.get_vocab():
                    print('WARNING: cannot add new special token')
                else:
                    print('WARNING: add new special token successfully')

            model.resize_token_embeddings(len(tokenizer))
            emb = getattr(model.embeddings, 'tok_embeddings', None) or getattr(model.embeddings, 'word_embeddings', None)
            print(f'WARNING: model embedding: {emb}')
            print(f'WARNING: tokenizer size: {len(tokenizer)}')
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=token)

        if push_to_hub_id is not None:
            model.push_to_hub(push_to_hub_id, token=token, private=True)
            tokenizer.push_to_hub(push_to_hub_id, token=token, private=True)

        return (cls(model,
                   model_wrapper=model_wrapper,
                   loss_fn=loss_fn, dropout_list = dropout_list), tokenizer, )
