# Vietnamese Text Search project

Description: This project aims to create **a Vietnamese benchmark** and **a small LM** for information retrieval task.

## Project structure
- dataset
    - VINLI_reranking
        - construct VINLI_reranking dataset (for evaluating)
        - construct VINLI_triplet dataset (for evaluating)
- retriever
    - evaluation
        - evaluator: contain class to evaluate for each task
        - examples: contain examples to evaluate embedding models on benchmarks
    - examples: contain script to train minilm and phobert from our datasets
    - dataset: method to get tokenize online datasets
    - loss and triloss: contain loss functions used in the papers
    - model: wrapper for embedding models
    - utils: support functions

## ModernBERT (bert-tiny-stage2-hf) Training & Evaluation

### Setup

```bash
pip install torch transformers datasets pyvi tqdm
```

### Training

Train `QuangDuy/bert-tiny-stage2-hf` with dataset `ContextSearchLM/context_search_vietnamese_prompt_224_minilmtok_finetune`:

```bash
# Using shell script
bash scripts/train_modernbert.sh

# Or run directly with Python
python retriever/examples/modernbert_training.py \
    --model_name QuangDuy/bert-tiny-stage2-hf \
    --model_type mean_pooling \
    --dataset ContextSearchLM/context_search_vietnamese_prompt_224_minilmtok_finetune \
    --model_repo_id YOUR_HF_REPO_ID \
    --token YOUR_HF_TOKEN \
    --device cuda \
    --loss_fn dropinfonce \
    --batch_size 48 \
    --max_length 4096 \
    --train_style 2 \
    --num_repeat 1
```

**Training styles:**
| Style | Description | Columns used |
|-------|-------------|--------------|
| `2` | In-batch negative | query + positive (negative as in-batch) |
| `3` | Hard-negative triplet | query + positive + negative |

> **Note:** The dataset contains pre-tokenized IDs from MiniLM tokenizer (`[CLS]=101, [SEP]=102, [PAD]=0`), which are **incompatible** with ModernBERT (`[CLS]=0, [SEP]=3, [PAD]=2`). The script automatically **re-tokenizes from raw text** using ModernBERT's tokenizer.

### Evaluation

Evaluate on Vietnamese retrieval/reranking benchmarks:

```bash
# Using shell script
bash scripts/eval_modernbert.sh

# Or run directly — all benchmarks
python retriever/examples/modernbert_evaluate.py \
    --model_name QuangDuy/bert-tiny-stage2-hf \
    --model_type mean_pooling \
    --device cuda \
    --benchmark all

# Run a specific benchmark
python retriever/examples/modernbert_evaluate.py \
    --benchmark vimedaqa    # ViMedAQA retrieval
python retriever/examples/modernbert_evaluate.py \
    --benchmark viglue_r    # ViGLUE-R reranking (MNLI-R, QNLI-R)
python retriever/examples/modernbert_evaluate.py \
    --benchmark vinli       # ViNLI reranking
```

**Benchmarks:**
| Benchmark | Type | Dataset |
|-----------|------|---------|
| ViMedAQA | Retrieval | `tmnam20/ViMedAQA` (5 subsets) |
| ViGLUE-R | Reranking | `ContextSearchLM/ViGLUE-R` (MNLI-R, QNLI-R) |
| ViNLI | Reranking | `ContextSearchLM/ViNLI_reranking` |

## Resource
Benchmarks and models are publicly available on Hugging Face. You can explore them [here](https://huggingface.co/ContextSearchLM).

Further specific information will be updated.

## Citation
[^1]: Coming soon