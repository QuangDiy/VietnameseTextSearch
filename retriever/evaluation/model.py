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
                 device='cuda',
                 base_model=None,
                 ) -> None:
        if base_model is not None:
            # Load architecture from base_model, weights from checkpoint
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)
            self.model = AutoModel.from_pretrained(base_model, token=token, trust_remote_code=True)
            # Load checkpoint state dict and strip 'model.' prefix
            import safetensors.torch
            import os
            ckpt_path = model_name_or_path
            safetensor_file = os.path.join(ckpt_path, 'model.safetensors')
            bin_file = os.path.join(ckpt_path, 'pytorch_model.bin')
            if os.path.exists(safetensor_file):
                state_dict = safetensors.torch.load_file(safetensor_file)
            elif os.path.exists(bin_file):
                state_dict = torch.load(bin_file, map_location='cpu')
            else:
                raise FileNotFoundError(f'No model weights found in {ckpt_path}')
            # Strip 'model.' prefix from keys (SimilarityLoss wrapper)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.removeprefix('model.')
                new_state_dict[new_key] = v
            self.model.load_state_dict(new_state_dict, strict=False)
            print(f'[FlexiEmbedding] Loaded checkpoint from {ckpt_path} with base model {base_model}')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=token)
            self.model = AutoModel.from_pretrained(model_name_or_path, token=token, trust_remote_code=True)
        self.tokenizer.padding_side = 'right'
        self.device = device
        self.model = self.model.to(device).eval()
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
