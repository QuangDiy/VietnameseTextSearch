#!/bin/bash
# ============================================================================
# Evaluate ModernBERT (bert-tiny-stage2-hf) on Vietnamese Benchmarks
# ============================================================================
#
# Benchmarks:
#   - ViMedAQA:  Vietnamese Medical Q&A Retrieval (5 subsets)
#   - ViGLUE-R:  Vietnamese GLUE Reranking (MNLI-R, QNLI-R)
#   - ViNLI:     Vietnamese NLI Reranking
# ============================================================================

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME="QuangDuy/bert-tiny-stage2-hf"

# HuggingFace token (set via environment variable or paste here)
HF_TOKEN="${HF_TOKEN:-}"

MODEL_TYPE="mean_pooling"   # "mean_pooling" or "cls_pooling"
DEVICE="cuda"               # "cuda" or "cpu"
BENCHMARK="all"             # "all", "vimedaqa", "viglue_r", or "vinli"

# ── Run evaluation ────────────────────────────────────────────────────────────
python retriever/examples/modernbert_evaluate.py \
    --model_name "${MODEL_NAME}" \
    --model_type "${MODEL_TYPE}" \
    --token "${HF_TOKEN}" \
    --device "${DEVICE}" \
    --benchmark "${BENCHMARK}"
