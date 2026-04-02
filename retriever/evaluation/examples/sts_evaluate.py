from ..evaluator.sts import STSEvaluator
from ..evaluate import JustEvaluate
from .models import MODEL_WRAPPERS

evaluate_datasets = ['mteb/sickr-sts', 'mteb/sts15-sts', 'mteb/sts16-sts', 'mteb/sts13-sts', 'mteb/sts14-sts', 'mteb/sts12-sts',
                     'NghiemAbe/sts12', 'NghiemAbe/sts13', 'NghiemAbe/sts14', 'NghiemAbe/sts15', 'NghiemAbe/sts16']

def preprocess(dataset):
    return {
        'embed': {
            'query': dataset['sentence1'],
            'positive': dataset['sentence2'],
        },
        'labels': {
            'labels': dataset['score'],
        }
    }

def evaluate(
        model_name='ContextSearchLM/vinilm_dropinfonce_triplet',
        model_type='mean_pooling',
        token=None,
        device='cuda',
        prev_query=' ',
        ):
    results = []
    for dataset_name in evaluate_datasets:
        result = JustEvaluate(
            model_name_or_path=model_name,
            dataset_name_or_path=dataset_name,
            evaluator=STSEvaluator,
            token=token,
            device=device,
            model_wrapper=MODEL_WRAPPERS[model_type],
            extract_text_fn=preprocess,
            result_folder='./results',
            split='test',
        ).run()
        
        results.append(result)

    return results