#!/bin/bash
# ============================================================================
# Train ModernBERT (bert-tiny-stage2-hf) for Vietnamese Retrieval
# ============================================================================
#
# Dataset: ContextSearchLM/context_search_vietnamese_prompt_224_minilmtok_finetune
# Model:   QuangDuy/bert-tiny-stage2-hf (ModernBERT, max_length=4096)
#
# NOTE: The dataset contains MiniLM tokenizer IDs which are INCOMPATIBLE
#       with ModernBERT. The script re-tokenizes from raw text automatically.
# ============================================================================

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME="QuangDuy/bert-tiny-stage2-hf"
DATASET="ContextSearchLM/context_search_vietnamese_prompt_224_minilmtok_finetune"
MODEL_REPO_ID="ContextSearchLM/modernbert_tiny_dropinfonce_prompt"

# HuggingFace token (set via environment variable or paste here)
HF_TOKEN="${HF_TOKEN:-}"

# Training hyperparameters
BATCH_SIZE=48
LEARNING_RATE=5e-5
MAX_LENGTH=4096
NUM_REPEAT=1
LOSS_FN="dropinfonce"       # "dropinfonce" or "infonce"
MODEL_TYPE="mean_pooling"   # "mean_pooling" or "cls_pooling"
TRAIN_STYLE=2               # 2 = in-batch negative, 3 = hard-negative triplet
OUTPUT_DIR="./modernbert_output"
LOGGING_STEPS=1000

# ── Run training ──────────────────────────────────────────────────────────────
python retriever/examples/modernbert_training.py \
    --model_name "${MODEL_NAME}" \
    --model_type "${MODEL_TYPE}" \
    --dataset "${DATASET}" \
    --model_repo_id "${MODEL_REPO_ID}" \
    --token "${HF_TOKEN}" \
    --device cuda \
    --loss_fn "${LOSS_FN}" \
    --batch_size ${BATCH_SIZE} \
    --num_repeat ${NUM_REPEAT} \
    --train_style ${TRAIN_STYLE} \
    --learning_rate ${LEARNING_RATE} \
    --max_length ${MAX_LENGTH} \
    --output_dir "${OUTPUT_DIR}" \
    --logging_steps ${LOGGING_STEPS}
