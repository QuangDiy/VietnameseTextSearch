from ..utils import JustTrainIt
from ..evaluation.examples.models import MODEL_WRAPPERS
from ..triloss import TriLosses
from ..evaluation.examples.ViMedAQA import evaluate as eval1
from ..evaluation.examples.ViGLUE_R_evaluate import evaluate as eval2
from ..evaluation.examples.ViNLI_rerank_evaluate import evaluate as eval3

def train(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_type="mean_pooling",
        dataset_name_or_path="ContextSearchLM/context_search_vietnamese_english_prompt_76_minilmtok_finetune",
        model_repo_id="ContextSearchLM/viennilm_dropinfonce_prompt_constract_learning",
        token=None,
        device="cuda",
        loss_fn="dropinfonce",
        batch_size=320,
        train_style=2,
        tokenizer_name_or_path=None,
    ):
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
        # pretrained
        drop_columns = ['query', 'pos', 'neg', 'negative_ids', 'negative_attention_mask']
    elif train_style == 3:
        # triplet
        drop_columns = ['query', 'pos', 'neg']

    JustTrainIt(
        dataset_repo={
            dataset_name_or_path: {
                'train': {'split': 'train'},
                'evaluate': {'split': 'validation'}
            }
        },
        model_name_or_path=model_name,
        model_wrapper=MODEL_WRAPPERS[model_type],
        model_repo_id=model_repo_id,
        token=token,
        tokenizer_name_or_path=model_name if tokenizer_name_or_path==None else tokenizer_name_or_path,
        loss_fn=TriLosses.DROPINFONCELOSS if loss_fn == 'dropinfonce' else TriLosses.INFONCELOSS,
        call_every_train=evaluate,
        dropout_list=(0.1, 0.15),
        learning_rate=5e-5,
        num_repeat=2,
        num_train_epochs=1,
        save_total_limit=1,
        drop_columns = drop_columns,
        batch_size=batch_size,
    )