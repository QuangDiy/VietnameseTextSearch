from .abstract import AbstractEvaluator
from torch.nn.functional import cosine_similarity
import torch

class BinaryRetrievalEvaluator(AbstractEvaluator):
    def __init__(self,
                 dataset_name_or_path,
                 extract_text_fn=None,
                 result_folder: str = './results',
                 **kwargs
                 ):
        task_description = 'Given query, support document, and irrelevant document\nThe task is to get the correct document from the query'
        task_name = 'binaty_text_similarity'

        super().__init__(
            dataset_name_or_path,
            task_description=task_description,
            task_name=task_name,
            extract_text_fn=extract_text_fn,
            result_folder=result_folder,
            **kwargs
        )
    
    def get_score(self, query, positive, negative, labels=None):
        positive_score = cosine_similarity(query, positive)
        negative_score = cosine_similarity(query, negative)

        accuracy = (positive_score > negative_score).float().mean()

        return {
            'accuracy': float(accuracy),
        }


class RetrievalEvaluator(AbstractEvaluator):
    def __init__(self,
                 dataset_name_or_path,
                 extract_text_fn=None,
                 result_folder: str = './results',
                 topk=[1, 5, 10, 20],
                 **kwargs
                 ):
        task_description = 'Given query, a support document, and many irrelevant documents\nThe task is to get the correct document in a set of k documents retrieved from the query'
        task_name = 'retriever'

        super().__init__(
            dataset_name_or_path,
            task_description=task_description,
            task_name=task_name,
            extract_text_fn=extract_text_fn,
            result_folder=result_folder,
            **kwargs
        )

        self.topk = topk
    
    def get_score(self, query, positive, negative=None, labels=None):
        if negative is not None:
            print('WARNING: negative parameter has been removed from RetrievalEvaluator, parsing is unnecessary')

        similarity_score = cosine_similarity(query, positive)

        result = {}
        for k in self.topk:
            accuracy = (similarity_score.diag().unsqueeze(dim=-1)>similarity_score.topk(k=k, dim=-1).values).sum(dim=-1).bool().float().mean()
            result[f'accuracy@{k}'] = float(accuracy)

        return result
    

class MultiRetrievalEvaluator(AbstractEvaluator):
    def __init__(self,
                 dataset_name_or_path,
                 extract_text_fn=None,
                 result_folder: str = './results',
                 topk=[1, 5, 10, 20, 50, 100],
                 **kwargs
                 ):
        task_description = 'Given query, a support document, and many irrelevant documents\nThe task is to get the correct document in a set of k documents retrieved from the query'
        task_name = 'retriever'

        super().__init__(
            dataset_name_or_path,
            task_description=task_description,
            task_name=task_name,
            extract_text_fn=extract_text_fn,
            result_folder=result_folder,
            **kwargs
        )

        self.topk = topk

    def cosine_similarity(self, x, y):
        return torch.mm(x/x.norm(dim=-1)[:,None], (y/y.norm(dim=-1)[:,None]).transpose(0,1))
    
    def get_score(self, query, positive, negative=None, labels=None):
        if negative is not None:
            print('WARNING: negative parameter has been removed from RetrievalEvaluator, parsing is unnecessary')

        similarity_score = self.cosine_similarity(query, positive)
        max_selected = torch.max(similarity_score*labels, dim=-1).values

        result = {}
        for k in self.topk:
            accuracy = (max_selected.unsqueeze(dim=-1)>similarity_score.topk(k=k, dim=-1).values).sum(dim=-1).bool().float().mean()
            result[f'accuracy@{k}'] = float(accuracy)

        return result