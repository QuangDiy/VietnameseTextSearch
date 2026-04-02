from retriever.triloss import TriLosses
from ..evaluation.examples.models import MODEL_WRAPPERS
from retriever.utils import JustTrainIt
from ..evaluation.examples.ViMedAQA import evaluate as eval1
from ..evaluation.examples.ViGLUE_R_evaluate import evaluate as eval2
from ..evaluation.examples.ViNLI_rerank_evaluate import evaluate as eval3

def train_phobert(
    model_name_or_path = 'dangvantuan/vietnamese-embedding',
    model_type='mean_pooling',
    dataset_repo_id = 'ContextSearchLM/context_search_vietnamese_prompt_224_phoberttok_finetune',
    model_repo_id = 'ContextSearchLM/phobert_dropinfonce_prompt',
    token = None,
    device = 'cuda',
    loss_fn = 'dropinfonce',
    num_repeat = 1,
    batch_size = 48,
    train_style=2,
    ):

    '''
    train_style: 1 -> pre-train method, train using different dropout as positive and different samples as negative pairs, only need 1 sentence per sample
    train_style: 2 -> finetune method, train using 1 anchor and 1 positive sentence (in-batch negative)
    train_style: 3 -> finetune method, train using 1 anchor and 2 positive sentences (hard-negative)
    '''

    def evaluate():
        eval1(
            model_name=model_repo_id,
            model_type=model_type,
            token=token,
            device=device,
        )
        eval2(
            model_name=model_repo_id,
            model_type=model_type,
            token=token,
            device=device
        )
        eval3(
            model_name=model_repo_id,
            model_type=model_type,
            token=token,
            device=device
        )

    if train_style == 2:
        drop_columns = ['query', 'pos', 'neg', 'negative_ids', 'negative_attention_mask']
    elif train_style == 3:
        drop_columns = ['query', 'pos', 'neg']

    JustTrainIt(
        dataset_repo={
            dataset_repo_id: {
                'train': {
                    'split': 'train'
                },
                'evaluate': {
                    'split': 'validation'
                }
            }
        },
        model_name_or_path=model_name_or_path,
        model_wrapper = MODEL_WRAPPERS[model_type], # Use mean pooling
        tokenizer_name_or_path=model_name_or_path,
        model_repo_id=model_repo_id,
        call_every_train=evaluate,
        dropout_list=(0.1, 0.15),
        token=token,
        learning_rate=5e-5,
        num_repeat=num_repeat,
        num_train_epochs=1,
        save_total_limit=1,
        drop_columns = drop_columns,
        loss_fn=TriLosses.DROPINFONCELOSS if loss_fn == 'dropinfonce' else TriLosses.INFONCELOSS,
        batch_size=batch_size,
    )