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

def evaluate(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_type='minilm',
        token=None,
        device='cuda',
        prev_query='<|query|> '
        ):
    dataset_name = 'npvinHnivqn/abstractiveqa-healthcare-vietnamese'
    split_name = 'disease_labeled[:3000]+body_part_labeled[:500]+medicine_labeled[:3000]+drug_labeled[:3000]'
    dataset = load_dataset(dataset_name, token=token, split=split_name)

    df = dataset.to_pandas()[['answer', 'title', 'keyword', 'generated_question', 'generated_answer']].groupby('answer').agg({
        'title': lambda x: x,
        'keyword': 'last',
        'generated_question': lambda x: x,
        'generated_answer': lambda x: x
    }).reset_index()
    df = df[~df['answer'].apply(lambda x: x.strip() == '')]

    if model_type == 'phobert':
        df['answer'] = df['answer'].apply(text_split(keep_underline=True))
    else:
        df['answer'] = df['answer'].apply(text_split(keep_underline=False))

    context_idx = []
    query_idx = []
    query_list = []
    context_list = []

    for i in df.iterrows():
        contexts = i[1]['answer']
        querys = [f'{prev_query}{j}' for j in i[1]['generated_question']]
        index = i[0]
        
        context_idx += [index]*len(contexts)
        query_idx += [index]*len(querys)
        query_list += list(querys)
        context_list += list(contexts)
        
    labels = (torch.tensor([query_idx]).T==torch.tensor([context_idx])).float()

    return JustEvaluate(
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
        split='disease_labeled[:10]',
    ).run()