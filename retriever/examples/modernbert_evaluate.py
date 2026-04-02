"""
Evaluation script for QuangDuy/bert-tiny-stage2-hf (ModernBERT) on
Vietnamese retrieval benchmarks.

This script evaluates the ModernBERT model on:
  1. ViMedAQA - Vietnamese Medical Q&A retrieval
  2. ViGLUE-R - Vietnamese GLUE Reranking (MNLI-R, QNLI-R)
  3. ViNLI   - Vietnamese NLI Reranking

IMPORTANT: ModernBERT uses a different tokenizer from MiniLM/PhoBERT.
The evaluation framework (FlexiEmbedding) auto-tokenizes from raw text,
so no special handling is needed for evaluation - just pass the correct
model_wrapper.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from retriever.evaluation.evaluate import JustEvaluate, JustEvaluateIt
from retriever.evaluation.evaluator.rerank import RerankingEvaluator
from retriever.evaluation.evaluator.retrieval import MultiRetrievalEvaluator
from retriever.evaluation.model import FlexiEmbedding


# ─── Model wrappers for ModernBERT ────────────────────────────────────────────

def mean_pooling(model_output, attention_mask):
    """Mean pooling: average all non-padding token embeddings."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def modernbert_mean_pooling(model, input_ids, attention_mask, **kwargs):
    """Mean pooling wrapper for ModernBERT evaluation."""
    model_output = model(input_ids=input_ids, attention_mask=attention_mask)
    embeddings = mean_pooling(model_output, attention_mask)
    return F.normalize(embeddings, p=2, dim=1)


def modernbert_cls_pooling(model, input_ids, attention_mask, **kwargs):
    """CLS token pooling wrapper for ModernBERT evaluation."""
    model_output = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    return model_output.last_hidden_state[:, 0]


MODEL_WRAPPERS = {
    'mean_pooling': modernbert_mean_pooling,
    'cls_pooling': modernbert_cls_pooling,
}


# ─── ViMedAQA Evaluation ──────────────────────────────────────────────────────

def evaluate_vimedaqa(
    model_name='QuangDuy/bert-tiny-stage2-hf',
    model_type='mean_pooling',
    token=None,
    device='cuda',
    prev_query='<|query|> ',
    name='all',
    base_model=None,
):
    """Evaluate on ViMedAQA dataset."""
    from datasets import load_dataset
    from pyvi.ViTokenizer import tokenize

    def text_split(text_max_length=96, text_duplicate=32, keep_underline=False):
        def run(text):
            text = tokenize(text).split(' ')
            jump_steps = text_max_length - text_duplicate
            result = [' '.join(text[i * jump_steps:i * jump_steps + text_max_length])
                      for i in range(len(text) // jump_steps + 1)]
            if not keep_underline:
                result = [i.replace('_', ' ') for i in result]
            return result
        return run

    print(f'\n[ViMedAQA] Evaluating {model_name} on subset: {name}')

    dataset_name = 'tmnam20/ViMedAQA'
    dataset = load_dataset(dataset_name, token=token, split='test', name=name)

    df = dataset.to_pandas()[['answer', 'question', 'context', 'title', 'keyword']].groupby('context').agg({
        'title': lambda x: x,
        'keyword': 'last',
        'question': lambda x: x,
        'answer': lambda x: x
    }).reset_index()
    df = df[~df['context'].apply(lambda x: x.strip() == '')]
    df['context'] = df['context'].apply(text_split(keep_underline=False))

    context_idx = []
    query_idx = []
    query_list = []
    context_list = []

    for i in df.iterrows():
        contexts = i[1]['context']
        querys = [f'{prev_query}{j}' for j in i[1]['question']]
        index = i[0]

        context_idx += [index] * len(contexts)
        query_idx += [index] * len(querys)
        query_list += list(querys)
        context_list += list(contexts)

    labels = (torch.tensor([query_idx]).T == torch.tensor([context_idx])).float()

    JustEvaluate(
        model_name_or_path=model_name,
        dataset_name_or_path=dataset_name,
        evaluator=MultiRetrievalEvaluator,
        token=token,
        device=device,
        model_wrapper=MODEL_WRAPPERS[model_type],
        extract_text_fn=lambda x: {
            'embed': {
                'query': query_list,
                'positive': context_list,
            },
            'labels': {
                'labels': labels
            }
        },
        result_folder='./results',
        split='test[:10]',
        name='all',
        base_model=base_model,
    ).run()


def evaluate_vimedaqa_all(
    model_name='QuangDuy/bert-tiny-stage2-hf',
    model_type='mean_pooling',
    token=None,
    device='cuda',
    prev_query='<|query|> ',
    base_model=None,
):
    """Evaluate on all ViMedAQA subsets."""
    for name in ['all', 'drug', 'medicine', 'disease', 'body-part']:
        evaluate_vimedaqa(model_name, model_type, token, device, prev_query, name, base_model=base_model)


# ─── ViGLUE-R Evaluation ──────────────────────────────────────────────────────

def evaluate_viglue_r(
    model_name='QuangDuy/bert-tiny-stage2-hf',
    model_type='mean_pooling',
    token=None,
    device='cuda',
    base_model=None,
):
    """Evaluate on ViGLUE-R (MNLI-R and QNLI-R)."""
    def preprocess(dataset):
        return {
            'embed': {
                'query': dataset['anchor'],
                'positive': dataset['pos'],
                'negative': dataset['neg'],
            },
            'labels': {}
        }

    print(f'\n[ViGLUE-R] Evaluating {model_name}')

    print('  MNLI-R...')
    JustEvaluate(
        model_name_or_path=model_name,
        dataset_name_or_path='ContextSearchLM/ViGLUE-R',
        evaluator=RerankingEvaluator,
        token=token,
        device=device,
        model_wrapper=MODEL_WRAPPERS[model_type],
        extract_text_fn=preprocess,
        result_folder='./results',
        split='mnli_r',
        base_model=base_model,
    ).run()

    print('  QNLI-R...')
    JustEvaluate(
        model_name_or_path=model_name,
        dataset_name_or_path='ContextSearchLM/ViGLUE-R',
        evaluator=RerankingEvaluator,
        token=token,
        device=device,
        model_wrapper=MODEL_WRAPPERS[model_type],
        extract_text_fn=preprocess,
        result_folder='./results',
        split='qnli_r',
        base_model=base_model,
    ).run()


# ─── ViNLI Reranking Evaluation ────────────────────────────────────────────────

def evaluate_vinli(
    model_name='QuangDuy/bert-tiny-stage2-hf',
    model_type='mean_pooling',
    token=None,
    device='cuda',
    base_model=None,
):
    """Evaluate on ViNLI reranking."""
    def preprocess(dataset):
        return {
            'embed': {
                'query': dataset['anchor'],
                'positive': dataset['pos'],
                'negative': dataset['neg'],
            },
            'labels': {}
        }

    print(f'\n[ViNLI] Evaluating {model_name}')

    JustEvaluate(
        model_name_or_path=model_name,
        dataset_name_or_path='ContextSearchLM/ViNLI_reranking',
        evaluator=RerankingEvaluator,
        token=token,
        device=device,
        model_wrapper=MODEL_WRAPPERS[model_type],
        extract_text_fn=preprocess,
        result_folder='./results',
        split='test',
        base_model=base_model,
    ).run()


# ─── Full evaluation ──────────────────────────────────────────────────────────

def evaluate_all(
    model_name='QuangDuy/bert-tiny-stage2-hf',
    model_type='mean_pooling',
    token=None,
    device='cuda',
    base_model=None,
):
    """Run ALL evaluation benchmarks."""
    print('=' * 60)
    print('ModernBERT Evaluation Script')
    print('=' * 60)
    print(f'Model:       {model_name}')
    print(f'Base model:  {base_model or "(same as model)"}')
    print(f'Model type:  {model_type}')
    print(f'Device:      {device}')
    print('=' * 60)

    evaluate_vimedaqa_all(model_name, model_type, token, device, base_model=base_model)
    evaluate_viglue_r(model_name, model_type, token, device, base_model=base_model)
    evaluate_vinli(model_name, model_type, token, device, base_model=base_model)

    print('\n' + '=' * 60)
    print('All evaluations complete!')
    print('=' * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate ModernBERT (bert-tiny-stage2-hf) on Vietnamese benchmarks'
    )
    parser.add_argument('--model_name', type=str, default='QuangDuy/bert-tiny-stage2-hf',
                        help='Model name or path')
    parser.add_argument('--model_type', type=str, default='mean_pooling',
                        choices=['mean_pooling', 'cls_pooling'],
                        help='Pooling strategy')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace API token')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--benchmark', type=str, default='all',
                        choices=['all', 'vimedaqa', 'viglue_r', 'vinli'],
                        help='Which benchmark to run')
    parser.add_argument('--base_model', type=str, default=None,
                        help='Base model for architecture (use when loading Trainer checkpoints)')

    args = parser.parse_args()

    if args.benchmark == 'all':
        evaluate_all(args.model_name, args.model_type, args.token, args.device, base_model=args.base_model)
    elif args.benchmark == 'vimedaqa':
        evaluate_vimedaqa_all(args.model_name, args.model_type, args.token, args.device, base_model=args.base_model)
    elif args.benchmark == 'viglue_r':
        evaluate_viglue_r(args.model_name, args.model_type, args.token, args.device, base_model=args.base_model)
    elif args.benchmark == 'vinli':
        evaluate_vinli(args.model_name, args.model_type, args.token, args.device, base_model=args.base_model)
