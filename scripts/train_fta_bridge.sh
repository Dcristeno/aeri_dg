#!/usr/bin/env bash
set -euo pipefail

# Contribution 2: FTA + Bridge.
# Pair-only bridge matches the strongest recorded full recipe.

DATA_ROOT="${DATA_ROOT:-/home/wuyong/datasets}" \
FINETUNE_INIT="${FINETUNE_INIT:-/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth}" \
USE_SWANLAB="${USE_SWANLAB:-1}" \
RUN_NAME="${RUN_NAME:-aeri_cda_fta_bridge_pair_k2_r1}" \
SWANLAB_EXPERIMENT="${SWANLAB_EXPERIMENT:-aeri_cda_fta_bridge_pair_k2_r1}" \
LOSS_NAMES="${LOSS_NAMES:-cda+fta+bridge}" \
TRAIN_SAMPLES_PER_ID="${TRAIN_SAMPLES_PER_ID:-2}" \
TRAIN_SAMPLE_STRATEGY="${TRAIN_SAMPLE_STRATEGY:-random}" \
BRIDGE_LOSS_WEIGHT="${BRIDGE_LOSS_WEIGHT:-2.0}" \
BRIDGE_PAIR_WEIGHT="${BRIDGE_PAIR_WEIGHT:-1.0}" \
BRIDGE_DISTILL_WEIGHT="${BRIDGE_DISTILL_WEIGHT:-0.0}" \
BEST_METRIC="${BEST_METRIC:-R1}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
bash finetune.sh "$@"
