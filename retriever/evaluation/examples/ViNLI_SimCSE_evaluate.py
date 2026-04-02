from ..evaluator.retrieval import BinaryRetrievalEvaluator
from ..evaluate import JustEvaluate
from .models import MODEL_WRAPPERS

def SimCSE_preprocess(dataset):
    return {
        'embed': {
            'query': [i.replace('_', ' ').strip() for i in dataset['anchor']],
            'positive': [i.replace('_', ' ').strip() for i in dataset['pos']],
            'negative': [i.replace('_', ' ').strip() for i in dataset['hard_neg']],
        },
        'labels': {}
    }

def evaluate(
        model_name='ContextSearchLM/vigte_dropinfonce_prompt',
        model_type='gte',
        token=None,
        device='cuda',
        prev_query=' ',
        ):
    return JustEvaluate(
        model_name_or_path=model_name,
        dataset_name_or_path='anti-ai/ViNLI-SimCSE-supervised',
        evaluator=BinaryRetrievalEvaluator,
        token=token,
        device=device,
        model_wrapper=MODEL_WRAPPERS[model_type],
        extract_text_fn=SimCSE_preprocess,
        result_folder='./results',
        split='train[:5000]',
    ).run()