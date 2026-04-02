from datasets import load_dataset, concatenate_datasets
import torch
from typing import List, Dict
from torch.utils.data import Dataset


class RetrieverDataset(Dataset):
    def __init__(self,
                 dataset_name,
                 get_model_input_with_tokenizer=None,
                 query_column_name='query',
                 positive_column_name='pos',
                 negative_column_name='neg',
                 data_preprocess_func=lambda x: x,
                 get_text_input=True,
                 tokenizer_max_length=1024,
                 **kwargs
                 ):
        print(f'working on {dataset_name} ...')
        self.dataset = load_dataset(dataset_name, **kwargs)
        self.dataset = data_preprocess_func(self.dataset)
        self.dataset = self.dataset.filter(lambda x: x[query_column_name] != '' and x[positive_column_name] != '' and x[negative_column_name] != '')

        columns_to_keep = []
        if get_model_input_with_tokenizer is not None:
            self.dataset = self.dataset.map(lambda x: get_model_input_with_tokenizer(x[query_column_name], max_length=tokenizer_max_length))
            self.dataset = self.dataset.rename_columns({'input_ids': 'query_ids', 'attention_mask': 'query_attention_mask'})

            self.dataset = self.dataset.map(lambda x: get_model_input_with_tokenizer(x[positive_column_name], max_length=tokenizer_max_length))
            self.dataset = self.dataset.rename_columns({'input_ids': 'positive_ids', 'attention_mask': 'positive_attention_mask'})

            self.dataset = self.dataset.map(lambda x: get_model_input_with_tokenizer(x[negative_column_name], max_length=tokenizer_max_length))
            self.dataset = self.dataset.rename_columns({'input_ids': 'negative_ids', 'attention_mask': 'negative_attention_mask'})

            columns_to_keep += ['query_ids', 'query_attention_mask',
                                'positive_ids', 'positive_attention_mask',
                                'negative_ids', 'negative_attention_mask']
        if get_text_input:
            columns_to_keep += [query_column_name, positive_column_name, negative_column_name]

        columns_to_remove = [col for col in self.dataset.column_names if col not in columns_to_keep]
        self.dataset = self.dataset.remove_columns(columns_to_remove)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class MergeContextDataset(Dataset):
    def __init__(self,
                 list_of_dataset,
                 filter_func=None,
                 remove_columns=['query', 'pos', 'neg'],
                 ):
        self.dataset = concatenate_datasets(list_of_dataset)
        print(self.dataset)
        if filter_func is not None:
            self.dataset = self.dataset.filter(filter_func)

        self.workingdataset = self.dataset.remove_columns(remove_columns)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
class GeneralCollator:
    def __init__(self,
                 tokenizer=None,
                 pad_value={},
                 padding_side='right',
                 ) -> None:
        self.inference: bool = False
        self.pad_value=pad_value
        self.padding_side = padding_side
        if tokenizer is not None:
            self.pad_value['query_ids'] = tokenizer.pad_token_id
            self.pad_value['positive_ids'] = tokenizer.pad_token_id
            self.pad_value['negative_ids'] = tokenizer.pad_token_id

    def list2dict(self, features: List[Dict]) -> Dict:
        return_dict = {}

        for feature in features:
            for k, v in feature.items():
                if k in return_dict:
                    return_dict[k].append(torch.tensor(v))
                else:
                    return_dict[k] = [torch.tensor(v)]
        
        return return_dict
    
    def create_pad(self, tensor: torch.Tensor, max_shape: torch.Size):
        tensor_shape = tensor.shape
        pad_raw = (torch.tensor(max_shape) - torch.tensor(tensor_shape)).tolist()
        result = []
        if self.padding_side == 'left':
            for i in pad_raw[::-1]:
                result += [i, 0]
        else:
            for i in pad_raw[::-1]:
                result += [0, i]
        return result

    def padd_tensor_for_batch(self, tensors, log=None, pad_value=0):
        max_shape = torch.tensor([max(dim_size) for dim_size in zip(*[tensor.shape for tensor in tensors])])

        list_of_tensors = [torch.nn.functional.pad(tensor, pad=self.create_pad(tensor, max_shape), value=pad_value).unsqueeze(0) for tensor in tensors]

        output = torch.cat(list_of_tensors, dim=0)

        return output

    def __call__(self, features: List[Dict]) -> Dict:
        feature_batch = self.list2dict(features)
        
        model_inputs = {key: self.padd_tensor_for_batch(value, key, pad_value=self.pad_value[key] if key in self.pad_value else 0) for key, value in feature_batch.items()}

        return model_inputs