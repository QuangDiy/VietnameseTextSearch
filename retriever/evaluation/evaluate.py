from .evaluator.abstract import AbstractEvaluator
from .model import FlexiEmbedding

class JustEvaluate:
    def __init__(self,
                 model_name_or_path,
                 dataset_name_or_path,
                 evaluator,
                 model_wrapper=lambda x, y: x(**y).last_hidden_state[:, 0],
                 token=None,
                 device='cuda',
                 extract_text_fn=None,
                 result_folder: str = './results',
                 **kwargs):
        self.evaluator = evaluator(
            dataset_name_or_path,
            extract_text_fn,
            result_folder,
            token=token,
            **kwargs
            )
        
        self.model = FlexiEmbedding(
            model_name_or_path=model_name_or_path,
            model_wrapper=model_wrapper,
            token=token,
            device=device
        )

    def run(self, **kwargs):
        self.evaluator.evaluate(
            self.model,
            **kwargs
        )

def JustEvaluateIt(
    model_names_or_paths={},
    dataset_names_or_paths={},
    token=None,
    device='cuda',
    result_folder: str = './results',
    ):
    for model, (model_wrapper, model_kwargs) in model_names_or_paths.items():
        print((model_wrapper, model_kwargs))
        for dataset, (evaluator, extract_text_fn, dataset_kwargs) in dataset_names_or_paths.items():
            print(model, dataset)
            JustEvaluate(
                model_name_or_path=model,
                dataset_name_or_path=dataset,
                evaluator=evaluator,
                model_wrapper=model_wrapper,
                token=token,
                device=device,
                extract_text_fn=extract_text_fn,
                result_folder=result_folder,
                **dataset_kwargs
            ).run(**model_kwargs)