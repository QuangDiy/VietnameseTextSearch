from .abstract import AbstractEvaluator
from torch.nn.functional import cosine_similarity
import torch
import numpy as np
from sklearn.metrics import average_precision_score

class RerankingEvaluator(AbstractEvaluator):
    def __init__(self,
                 dataset_name_or_path,
                 extract_text_fn=None,
                 result_folder: str = './results',
                 **kwargs
                 ):
        task_description = 'read reranking task of MTEB dataset'
        task_name = 'reranking'

        super().__init__(
            dataset_name_or_path,
            task_description=task_description,
            task_name=task_name,
            extract_text_fn=extract_text_fn,
            result_folder=result_folder,
            **kwargs
        )

    def get_score(self, query, positive, negative, labels=None):
        ap_scores = []
        for _q, _p, _n in zip(query, positive, negative):
            if _q == None or _p == None or _n == None: continue
            label = [1]*len(_p) + [0]*len(_n)
            score = cosine_similarity(_q, torch.concatenate([_p, _n], axis=0).to(_q.device))
            ap = average_precision_score(label, score)
            ap_scores.append(ap)

        map_score = np.mean(ap_scores)

        return {
            'mAP': float(map_score),
        }
