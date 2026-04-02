"""
Training script for QuangDuy/bert-tiny-stage2-hf (ModernBERT) with
ContextSearchLM/context_search_vietnamese_prompt_224_minilmtok_finetune dataset.

CRITICAL: The dataset contains pre-tokenized IDs from MiniLM tokenizer
(e.g. [CLS]=101, [SEP]=102, [PAD]=0), which are INCOMPATIBLE with
ModernBERT's tokenizer ([CLS]=0, [MASK]=1, [PAD]=2, [SEP]=3, [UNK]=4).

This script re-tokenizes from raw text columns (query, pos, neg)
using the ModernBERT tokenizer with max context length 4096.
"""

import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from torch.utils.data import Dataset
from typing import List, Dict

import sys
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from retriever.triloss import TriLosses
from retriever.model import SimilarityLoss, change_dropout
from retriever.dataset import GeneralCollator
from retriever.loss import cosine_similarity


# ─── Model wrappers ───────────────────────────────────────────────────────────

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def modernbert_mean_pooling_wrapper(model, input_ids, attention_mask, **kwargs):
    """Mean pooling wrapper for ModernBERT."""
    model_output = model(input_ids=input_ids, attention_mask=attention_mask)
    embeddings = mean_pooling(model_output, attention_mask)
    return F.normalize(embeddings, p=2, dim=1)


def modernbert_cls_pooling_wrapper(model, input_ids, attention_mask, **kwargs):
    """CLS token pooling wrapper for ModernBERT."""
    model_output = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    return model_output.last_hidden_state[:, 0]


MODEL_WRAPPERS = {
    'mean_pooling': modernbert_mean_pooling_wrapper,
    'cls_pooling': modernbert_cls_pooling_wrapper,
}


# ─── Dataset with re-tokenization ─────────────────────────────────────────────

class RetokenizedDataset(Dataset):
    """
    Dataset that loads raw text from HuggingFace dataset and re-tokenizes
    using the target model's tokenizer.

    This is necessary when the dataset contains pre-tokenized IDs from a
    different tokenizer (e.g. MiniLM) than the model being trained (e.g. ModernBERT).
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        query_column='query',
        positive_column='pos',
        negative_column='neg',
        max_length=4096,
        include_negative=True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.query_column = query_column
        self.positive_column = positive_column
        self.negative_column = negative_column
        self.include_negative = include_negative

        # Filter empty rows
        self.dataset = dataset.filter(
            lambda x: x[query_column] != '' and x[positive_column] != ''
            and (not include_negative or x[negative_column] != '')
        )

        print(f'[RetokenizedDataset] Loaded {len(self.dataset)} samples')
        print(f'[RetokenizedDataset] Tokenizer: {type(tokenizer).__name__}')
        print(f'[RetokenizedDataset] Max length: {max_length}')
        print(f'[RetokenizedDataset] Special tokens: CLS={tokenizer.cls_token_id}, '
              f'SEP={tokenizer.sep_token_id}, PAD={tokenizer.pad_token_id}')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Tokenize query
        query_enc = self.tokenizer(
            item[self.query_column],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )

        # Tokenize positive
        pos_enc = self.tokenizer(
            item[self.positive_column],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )

        result = {
            'query_ids': query_enc['input_ids'],
            'query_attention_mask': query_enc['attention_mask'],
            'positive_ids': pos_enc['input_ids'],
            'positive_attention_mask': pos_enc['attention_mask'],
        }

        if self.include_negative:
            neg_enc = self.tokenizer(
                item[self.negative_column],
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_attention_mask=True,
            )
            result['negative_ids'] = neg_enc['input_ids']
            result['negative_attention_mask'] = neg_enc['attention_mask']

        return result


# ─── Collator ──────────────────────────────────────────────────────────────────

class ModernBertCollator:
    """
    Collator that dynamically pads batches using the correct pad token ID.
    ModernBERT uses pad_token_id=2 (not 0 like MiniLM).
    """

    def __init__(self, pad_token_id=2):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict]) -> Dict:
        batch = {}

        for key in features[0].keys():
            tensors = [torch.tensor(f[key]) for f in features]

            # Determine pad value
            if 'ids' in key:
                pad_value = self.pad_token_id
            else:
                pad_value = 0

            # Find max length
            max_len = max(t.shape[0] for t in tensors)

            # Pad (right padding)
            padded = []
            for t in tensors:
                pad_size = max_len - t.shape[0]
                if pad_size > 0:
                    t = F.pad(t, (0, pad_size), value=pad_value)
                padded.append(t.unsqueeze(0))

            batch[key] = torch.cat(padded, dim=0)

        return batch


# ─── Evaluation ────────────────────────────────────────────────────────────────

def get_batch_similar_info(similar):
    batch_size = similar.shape[0]
    data_pair_similar = similar.diagonal()
    diff_pair_similar = similar.sum() - data_pair_similar.sum()
    return (float(data_pair_similar.mean()),
            float(diff_pair_similar / (batch_size * batch_size - batch_size)))


def check_model(model, eval_dataset, collator, eval_batch=64):
    from tqdm import tqdm

    eval_result = []

    with torch.no_grad():
        for i in tqdm(range(0, len(eval_dataset), eval_batch)):
            model_inputs = collator(
                [eval_dataset[k] for k in range(i, min(i + eval_batch, len(eval_dataset)))]
            )
            loss, query, pos, neg = model(**model_inputs, logger=True)

            result = (float(loss),)
            pos_query_sim = cosine_similarity(query, pos)
            result += get_batch_similar_info(pos_query_sim)

            if neg is not None:
                neg_query_sim = cosine_similarity(query, neg)
                result += get_batch_similar_info(neg_query_sim)

            eval_result.append(result)

    eval_result = torch.tensor(eval_result).mean(dim=0).tolist()
    return {
        'description': f'loss, pair_pos, diff_pos, pair_neg, diff_neg: {eval_result}',
        'loss': eval_result[0],
    }


# ─── Main training function ───────────────────────────────────────────────────

def train_modernbert(
    model_name_or_path='QuangDuy/bert-tiny-stage2-hf',
    model_type='mean_pooling',
    dataset_repo_id='ContextSearchLM/context_search_vietnamese_prompt_224_minilmtok_finetune',
    model_repo_id='ContextSearchLM/modernbert_tiny_dropinfonce_prompt',
    token=None,
    device='cuda',
    loss_fn='dropinfonce',
    num_repeat=1,
    batch_size=48, 
    train_style=2,
    learning_rate=5e-5,
    max_length=4096,
    dropout_list=(0.1, 0.15),
    output_dir='./modernbert_output',
    logging_steps=1000,
):
    """
    Train ModernBERT (bert-tiny-stage2-hf) for Vietnamese retrieval.

    IMPORTANT: Re-tokenizes the dataset using ModernBERT's tokenizer
    instead of using the pre-computed MiniLM token IDs.

    Args:
        model_name_or_path: HuggingFace model path for ModernBERT
        model_type: Pooling type ('mean_pooling', 'cls_pooling')
        dataset_repo_id: HuggingFace dataset path
        model_repo_id: Where to push the trained model
        token: HuggingFace API token
        device: Device to use ('cuda' or 'cpu')
        loss_fn: Loss function ('dropinfonce' or 'infonce')
        num_repeat: Number of training repeats
        batch_size: Training batch size
        train_style: 2 = in-batch negative, 3 = hard-negative (triplet)
        learning_rate: Learning rate
        max_length: Maximum sequence length for tokenization (4096 for ModernBERT)
        dropout_list: Tuple of dropout rates (query, context)
        output_dir: Directory for saving checkpoints
        logging_steps: Log every N steps
    """
    import random

    print('=' * 60)
    print('ModernBERT Training Script')
    print('=' * 60)
    print(f'Model:   {model_name_or_path}')
    print(f'Dataset: {dataset_repo_id}')
    print(f'Max len: {max_length}')
    print(f'Style:   {"in-batch negative" if train_style == 2 else "hard-negative (triplet)"}')
    print(f'Loss:    {loss_fn}')
    print('=' * 60)

    # ── 1. Load model & tokenizer ──────────────────────────────────────────
    loss_enum = TriLosses.DROPINFONCELOSS if loss_fn == 'dropinfonce' else TriLosses.INFONCELOSS
    model_wrapper = MODEL_WRAPPERS[model_type]

    model, tokenizer = SimilarityLoss.create_from_pretrained(
        model_name_or_path,
        tokenizer_name_or_path=model_name_or_path,
        token=token,
        loss_fn=loss_enum,
        model_wrapper=model_wrapper,

        dropout_list=dropout_list,
    )

    print(f'\nTokenizer info:')
    print(f'  CLS token: {tokenizer.cls_token} (id={tokenizer.cls_token_id})')
    print(f'  SEP token: {tokenizer.sep_token} (id={tokenizer.sep_token_id})')
    print(f'  PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})')
    print(f'  Vocab size: {len(tokenizer)}')
    print(f'  Max model length: {tokenizer.model_max_length}')

    # ── 2. Load & re-tokenize dataset ──────────────────────────────────────
    print('\nLoading and re-tokenizing dataset...')

    include_negative = (train_style == 3)

    raw_train = load_dataset(dataset_repo_id, split='train', token=token)
    raw_eval = load_dataset(dataset_repo_id, split='validation', token=token)

    train_dataset = RetokenizedDataset(
        raw_train, tokenizer,
        max_length=max_length,
        include_negative=include_negative,
    )
    eval_dataset = RetokenizedDataset(
        raw_eval, tokenizer,
        max_length=max_length,
        include_negative=include_negative,
    )

    # ── 3. Collator ────────────────────────────────────────────────────────
    collator = ModernBertCollator(pad_token_id=tokenizer.pad_token_id)

    # ── 4. Training loop ───────────────────────────────────────────────────
    seed = random.randint(0, 100)

    training_args = TrainingArguments(
        output_dir=output_dir,
        hub_token=token,
        do_train=True,
        seed=seed,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        save_total_limit=1,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    class ModernBertTrainer(Trainer):
        def _save(self, output_dir=None, state_dict=None):
            output_dir = output_dir or self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            unwrapped = self.model
            if hasattr(unwrapped, 'module'):
                unwrapped = unwrapped.module
            unwrapped.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    trainer = ModernBertTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=model,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    best_result = float('inf')

    for repeat_idx in range(num_repeat):
        print(f'\n--- Repeat {repeat_idx + 1}/{num_repeat} ---')

        # Evaluate
        output_dict = check_model(model, eval_dataset, collator, eval_batch=batch_size)
        print(output_dict['description'])

        if output_dict['loss'] < best_result:
            best_result = output_dict['loss']
            print('Found better model, saving locally...')
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        # Train
        trainer.train()

    # Final evaluation
    output_dict = check_model(model, eval_dataset, collator, eval_batch=batch_size)
    print(f'\nFinal: {output_dict["description"]}')

    if output_dict['loss'] < best_result:
        print('Found better model, saving locally...')
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    print('\nTraining complete!')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train ModernBERT (bert-tiny-stage2-hf) for Vietnamese retrieval'
    )
    parser.add_argument('--model_name', type=str, default='QuangDuy/bert-tiny-stage2-hf',
                        help='Model name or path')
    parser.add_argument('--model_type', type=str, default='mean_pooling',
                        choices=['mean_pooling', 'cls_pooling'],
                        help='Pooling strategy')
    parser.add_argument('--dataset', type=str,
                        default='ContextSearchLM/context_search_vietnamese_prompt_224_minilmtok_finetune',
                        help='Dataset name or path')
    parser.add_argument('--model_repo_id', type=str,
                        default='ContextSearchLM/modernbert_tiny_dropinfonce_prompt',
                        help='HuggingFace repo to push model')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace API token')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--loss_fn', type=str, default='dropinfonce',
                        choices=['dropinfonce', 'infonce'],
                        help='Loss function')
    parser.add_argument('--batch_size', type=int, default=48,
                        help='Training batch size')
    parser.add_argument('--num_repeat', type=int, default=3,
                        help='Number of training repeats')
    parser.add_argument('--train_style', type=int, default=2, choices=[2, 3],
                        help='2=in-batch negative, 3=hard-negative')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=4096,
                        help='Max sequence length for tokenization')
    parser.add_argument('--output_dir', type=str, default='./modernbert_output',
                        help='Output directory for checkpoints')
    parser.add_argument('--logging_steps', type=int, default=1000,
                        help='Log every N steps')

    args = parser.parse_args()

    train_modernbert(
        model_name_or_path=args.model_name,
        model_type=args.model_type,
        dataset_repo_id=args.dataset,
        model_repo_id=args.model_repo_id,
        token=args.token,
        device=args.device,
        loss_fn=args.loss_fn,
        batch_size=args.batch_size,
        num_repeat=args.num_repeat,
        train_style=args.train_style,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
    )
