from ..evaluator.retrieval import BinaryRetrievalEvaluator
from ..evaluate import JustEvaluate
from .models import MODEL_WRAPPERS

def evaluate(
        model_name='ContextSearchLM/vigte_dropinfonce_prompt',
        model_type='gte',
        token=None,
        device='cuda',
        prev_query=' ',
        ):
    return JustEvaluate(
        model_name_or_path=model_name,
        dataset_name_or_path='ContextSearchLM/context_search_vietnamese_english_prompt_224_minilmtok_finetune',
        evaluator=BinaryRetrievalEvaluator,
        token=token,
        device=device,
        model_wrapper=MODEL_WRAPPERS[model_type],
        extract_text_fn=lambda dataset: {
            'embed': {
                'query': [prev_query + i.strip() for i in dataset['query']],
                'positive': [i.strip() for i in dataset['pos']],
                'negative': [i.strip() for i in dataset['neg']],
            },
            'labels': {}
        },
        result_folder='./results',
        split='validation',
    ).run()