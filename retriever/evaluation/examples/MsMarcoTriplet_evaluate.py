from ..evaluator.retrieval import BinaryRetrievalEvaluator
from ..evaluate import JustEvaluate
from .models import MODEL_WRAPPERS    

def evaluate(
        model_name,
        model_type,
        token=None,
        device='cuda',
        prev_query='<|query|> ',
        ):

    return JustEvaluate(
        model_name_or_path=model_name,
        dataset_name_or_path='bclavie/msmarco-10m-triplets',
        evaluator=BinaryRetrievalEvaluator,
        token=token,
        device=device,
        model_wrapper=MODEL_WRAPPERS[model_type],
        extract_text_fn=lambda x: {
            'embed': {
                'query': [f'{prev_query}{i.strip()}' for i in x['query']],
                'positive': x['positive'],
                'negative': x['negative'],
            },
            'labels': {}            
        },
        result_folder='./results',
        split='train[100000:105000]',
    ).run()