from ..evaluator.retrieval import MultiRetrievalEvaluator
from ..evaluate import JustEvaluate
from .models import MODEL_WRAPPERS
import torch
from datasets import load_dataset
from pyvi.ViTokenizer import tokenize

def text_split(text_max_length=96, text_duplicate=32, keep_underline=False):
    def run(text):
        text = tokenize(text).split(' ')
        jump_steps = text_max_length - text_duplicate
        result = [' '.join(text[i*jump_steps:i*jump_steps+text_max_length]) for i in range(len(text)//jump_steps+1)]

        if keep_underline == False:
            result = [i.replace('_', ' ') for i in result]

        return result
    
    return run

def evaluate_(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_type='mean_pooling',
        token=None,
        device='cuda',
        prev_query='<|query|> ',
        name='all'
        ):
    print(name)
    dataset_name = 'tmnam20/ViMedAQA'
    split_name = 'test'
    dataset = load_dataset(dataset_name, token=token, split=split_name, name=name)

    df = dataset.to_pandas()[['answer', 'question', 'context', 'title', 'keyword']].groupby('context').agg({
        'title': lambda x: x,
        'keyword': 'last',
        'question': lambda x: x,
        'answer': lambda x: x
    }).reset_index()
    df = df[~df['context'].apply(lambda x: x.strip() == '')]

    if model_type == 'phobert':
        df['context'] = df['context'].apply(text_split(keep_underline=True))
    else:
        df['context'] = df['context'].apply(text_split(keep_underline=False))

    context_idx = []
    query_idx = []
    query_list = []
    context_list = []

    for i in df.iterrows():
        contexts = i[1]['context']
        querys = [f'{prev_query}{j}' for j in i[1]['question']]
        index = i[0]
        
        context_idx += [index]*len(contexts)
        query_idx += [index]*len(querys)
        query_list += list(querys)
        context_list += list(contexts)
        
    labels = (torch.tensor([query_idx]).T==torch.tensor([context_idx])).float()

    JustEvaluate(
        model_name_or_path=model_name,
        dataset_name_or_path=dataset_name,
        evaluator=MultiRetrievalEvaluator,
        token=token,
        device=device,
        model_wrapper=MODEL_WRAPPERS[model_type],
        extract_text_fn=lambda x:{
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
    ).run()


def evaluate(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_type='mean_pooling',
        token=None,
        device='cuda',
        prev_query='<|query|> '
        ):
    evaluate_(model_name, model_type, token, device, prev_query, name='all')
    evaluate_(model_name, model_type, token, device, prev_query, name='drug')
    evaluate_(model_name, model_type, token, device, prev_query, name='medicine')
    evaluate_(model_name, model_type, token, device, prev_query, name='disease')
    evaluate_(model_name, model_type, token, device, prev_query, name='body-part')