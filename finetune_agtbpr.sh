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
  --img_aug \
  --batch_size "${BATCH_SIZE}" \
  --MLM \
  --dataset_name "${DATASET_NAME}" \
  --loss_names "${LOSS_NAMES}" \
  --lr "${LR}" \
  --lr2 "${LR2}" \
  --num_epoch "${NUM_EPOCH}" \
  --root_dir "${DATA_ROOT}" \
  --finetune "${FINETUNE_INIT}" \
  "${SWANLAB_ARGS[@]}"
