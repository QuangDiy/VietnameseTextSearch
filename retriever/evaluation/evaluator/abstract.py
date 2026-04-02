from abc import abstractmethod
from datasets import load_dataset
from typing import Dict, List
import os
import json

class AbstractEvaluator:
    def __init__(self,
                 dataset_name_or_path,
                 task_description: str = 'No description',
                 task_name: str = 'unk',
                 extract_text_fn=None,
                 result_folder: str = './results',
                 **kwargs,
                 ) -> None:
        self.dataset = load_dataset(dataset_name_or_path, **kwargs)
        self.dataset_name = dataset_name_or_path
        self.task_description = task_description
        self.task_name = task_name

        self.result_folder = result_folder

        if extract_text_fn is not None:
            self.dataset = extract_text_fn(self.dataset)
        else:
            self.dataset = {
                'embed': {
                    k: self.dataset[k] for k in self.dataset
                },
                'labels': {

                }
            }

    def get_embedding(self, model, **kwargs):
        result = {}
        for k, v in self.dataset['embed'].items():
            result[k] = model.encode(v, **kwargs)
        return result

    @abstractmethod
    def get_score(self, labels, **kwargs):
        raise NotImplementedError

    def evaluate(self, model, **kwargs):
        embeddings = self.get_embedding(model, **kwargs)
        score = self.get_score(**embeddings, **self.dataset['labels'])

        result = {
                'score': score,
                'task_description': self.task_description,
                'dataset_name': self.dataset_name,
                'task_name': self.task_name,
            }
        
        print(self.result_folder, self.task_name, self.dataset_name)
        save_directory = os.path.join(self.result_folder, self.task_name, self.dataset_name) + '.json'
        os.makedirs(os.path.dirname(save_directory), exist_ok=True)

        print(result)
        print('-'*40)

        with open(save_directory, 'w') as jf:
            json.dump(result, jf, indent=4)
