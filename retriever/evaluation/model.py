from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class FlexiEmbedding:
    def __init__(self,
                 model_name_or_path,
                 model_wrapper=lambda x, y: x(**y).last_hidden_state[:, 0],
                 token=None,
                 device='cuda'
                 ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=token)
        self.tokenizer.padding_side = 'right'
        self.device = device
        self.model = AutoModel.from_pretrained(model_name_or_path, token=token, trust_remote_code=True).to(device).eval()
        self.model_wrapper = model_wrapper
        
    def encode(
        self,
        sentences: list[str] | list[list[str]],
        max_length=512,
        **kwargs
    ) -> torch.Tensor | np.ndarray:
        with torch.no_grad():
            if isinstance(sentences[0], str):
                batch_size = kwargs.get('batch_size', 32)
                result = []
                for i in tqdm(range(0, len(sentences), batch_size)):
                    model_inputs = self.tokenizer(sentences[i:i+batch_size], return_tensors="pt", padding=True, max_length=max_length, truncation=True).to(self.device)
                    
                    embeddings = self.model_wrapper(self.model, **model_inputs)
                    
                    result.append(embeddings)
                
                result = torch.concat(result, dim=0).cpu()
                return result
            else:
                result = []
                for group_sent in sentences:
                    if len(group_sent) == 0:
                        result.append(None)
                    else:
                        model_inputs = self.tokenizer(group_sent, return_tensors="pt", padding=True, max_length=max_length, truncation=True).to(self.device)

                        embeddings = self.model_wrapper(self.model, **model_inputs)
                        
                        result.append(embeddings)

                return result
