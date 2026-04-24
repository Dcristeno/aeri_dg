#!/bin/bash
set -euo pipefail

DATASET_NAME="${DATASET_NAME:-AGTBPR}"
DATA_ROOT="${DATA_ROOT:-/home/wuyong/data/TBAGPR}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
FINETUNE_INIT="${FINETUNE_INIT:-pretrain/HAMbest0.pth}"
RUN_NAME="${RUN_NAME:-author_agtbpr_cda}"
LOSS_NAMES="${LOSS_NAMES:-cda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-5e-6}"
LR2="${LR2:-5e-5}"
NUM_EPOCH="${NUM_EPOCH:-60}"
SEED="${SEED:-1}"
TRAIN_SAMPLES_PER_ID="${TRAIN_SAMPLES_PER_ID:-0}"
BRIDGE_LOSS_WEIGHT="${BRIDGE_LOSS_WEIGHT:-1.0}"
BRIDGE_PAIR_WEIGHT="${BRIDGE_PAIR_WEIGHT:-1.0}"
BRIDGE_DISTILL_WEIGHT="${BRIDGE_DISTILL_WEIGHT:-1.0}"
BRIDGE_DISTILL_TEMP="${BRIDGE_DISTILL_TEMP:-0.07}"
USE_SWANLAB="${USE_SWANLAB:-1}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-CFAN}"
SWANLAB_EXPERIMENT="${SWANLAB_EXPERIMENT:-$RUN_NAME}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

SWANLAB_ARGS=()
if [ "${USE_SWANLAB}" = "1" ]; then
  SWANLAB_ARGS+=(
    --use_swanlab
    --swanlab_project "${SWANLAB_PROJECT}"
    --swanlab_experiment "${SWANLAB_EXPERIMENT}"
    --swanlab_mode "${SWANLAB_MODE}"
  )
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python finetune.py \
  --name "${RUN_NAME}" \
  --seed "${SEED}" \
  --img_aug \
  --batch_size "${BATCH_SIZE}" \
  --MLM \
  --dataset_name "${DATASET_NAME}" \
  --loss_names "${LOSS_NAMES}" \
  --lr "${LR}" \
  --lr2 "${LR2}" \
  --num_epoch "${NUM_EPOCH}" \
  --train_samples_per_id "${TRAIN_SAMPLES_PER_ID}" \
  --bridge_loss_weight "${BRIDGE_LOSS_WEIGHT}" \
  --bridge_pair_weight "${BRIDGE_PAIR_WEIGHT}" \
  --bridge_distill_weight "${BRIDGE_DISTILL_WEIGHT}" \
  --bridge_distill_temp "${BRIDGE_DISTILL_TEMP}" \
  --root_dir "${DATA_ROOT}" \
  --finetune "${FINETUNE_INIT}" \
  "${SWANLAB_ARGS[@]}"
