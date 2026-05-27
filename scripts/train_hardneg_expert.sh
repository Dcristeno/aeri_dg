#!/usr/bin/env bash
set -euo pipefail

# Merge direction-three expert: different training objective.
# This expert is used by GAR-EM, not as an ablation module for the main recipe.

DATA_ROOT="${DATA_ROOT:-/home/wuyong/datasets}" \
FINETUNE_INIT="${FINETUNE_INIT:-/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth}" \
USE_SWANLAB="${USE_SWANLAB:-1}" \
RUN_NAME="${RUN_NAME:-aeri_cda_hardneg_k2_vit_b16_r1}" \
SWANLAB_EXPERIMENT="${SWANLAB_EXPERIMENT:-aeri_cda_hardneg_k2_vit_b16_r1}" \
PRETRAIN_CHOICE="${PRETRAIN_CHOICE:-ViT-B/16}" \
LOSS_NAMES="${LOSS_NAMES:-cda+hardneg}" \
HARD_NEGATIVE_LOSS_WEIGHT="${HARD_NEGATIVE_LOSS_WEIGHT:-1.0}" \
HARD_NEGATIVE_MARGIN="${HARD_NEGATIVE_MARGIN:-0.2}" \
HARD_NEGATIVE_TOPK="${HARD_NEGATIVE_TOPK:-1}" \
HARD_NEGATIVE_POSITIVE_REDUCE="${HARD_NEGATIVE_POSITIVE_REDUCE:-max}" \
TRAIN_SAMPLES_PER_ID="${TRAIN_SAMPLES_PER_ID:-2}" \
TRAIN_SAMPLE_STRATEGY="${TRAIN_SAMPLE_STRATEGY:-random}" \
BEST_METRIC="${BEST_METRIC:-R1}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
bash finetune.sh "$@"
