from .abstract import AbstractEvaluator
from torch.nn.functional import cosine_similarity
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)

class STSEvaluator(AbstractEvaluator):
    def __init__(self,
                 dataset_name_or_path,
                 extract_text_fn=None,
                 result_folder: str = './results',
                 **kwargs
                 ):
        task_description = 'Compare the similarity of two sentences with labeled score'
        task_name = 'sts'

        super().__init__(
            dataset_name_or_path,
            task_description=task_description,
            task_name=task_name,
            extract_text_fn=extract_text_fn,
            result_folder=result_folder,
            **kwargs
        )
    
    def normalize(self, x):
        x = np.array(x)
        return ((x - x.min()) / (x.max() - x.min())).tolist()

    def get_score(self, query, positive, negative=None, labels=None):
        cosine_scores = 1 - (paired_cosine_distances(query, positive))
        manhattan_distances = -paired_manhattan_distances(query, positive)
        euclidean_distances = -paired_euclidean_distances(query, positive)

        labels = self.normalize(labels)

        cosine_pearson, _ = pearsonr(labels, cosine_scores)
        cosine_spearman, _ = spearmanr(labels, cosine_scores)

        manhatten_pearson, _ = pearsonr(labels, manhattan_distances)
        manhatten_spearman, _ = spearmanr(labels, manhattan_distances)

        euclidean_pearson, _ = pearsonr(labels, euclidean_distances)
        euclidean_spearman, _ = spearmanr(labels, euclidean_distances)

        return {
            "cos_sim": {
                "pearson": cosine_pearson,
                "spearman": cosine_spearman,
            },
            "manhattan": {
                "pearson": manhatten_pearson,
                "spearman": manhatten_spearman,
            },
            "euclidean": {
                "pearson": euclidean_pearson,
                "spearman": euclidean_spearman,
            },
        }

