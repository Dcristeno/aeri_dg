#!/bin/bash
set -euo pipefail

DATASET_NAME="${DATASET_NAME:-AERI-PEDES}"
DATA_ROOT="${DATA_ROOT:-/home/wuyong/datasets}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
FINETUNE_INIT="${FINETUNE_INIT:-/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth}"
RUN_NAME="${RUN_NAME:-aeri_k2_ground_bridge_lite}"
SEED="${SEED:-1}"
FINETUNE_EVAL_MODE="${FINETUNE_EVAL_MODE:-test}"
USE_SWANLAB="${USE_SWANLAB:-1}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-CFAN}"
SWANLAB_EXPERIMENT="${SWANLAB_EXPERIMENT:-${RUN_NAME}}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
LOSS_NAMES="${LOSS_NAMES:-base+id+bridge}"
ID_LOSS_WEIGHT="${ID_LOSS_WEIGHT:-0.5}"
BRIDGE_LOSS_WEIGHT="${BRIDGE_LOSS_WEIGHT:-0.5}"
BRIDGE_MODE="${BRIDGE_MODE:-plain}"
BRIDGE_GATE_MIN="${BRIDGE_GATE_MIN:-0.2}"
BRIDGE_GATE_TAU="${BRIDGE_GATE_TAU:-0.1}"
IMAGE_POOLING="${IMAGE_POOLING:-cls}"
FOREGROUND_POOL_MIX="${FOREGROUND_POOL_MIX:-0.75}"
FOREGROUND_POOL_TAU="${FOREGROUND_POOL_TAU:-0.07}"
FOREGROUND_CENTER_WEIGHT="${FOREGROUND_CENTER_WEIGHT:-2.0}"
FOREGROUND_DEPTH_WEIGHT="${FOREGROUND_DEPTH_WEIGHT:-1.0}"
TRAIN_SAMPLES_PER_ID="${TRAIN_SAMPLES_PER_ID:-2}"
TRAIN_SAMPLE_STRATEGY="${TRAIN_SAMPLE_STRATEGY:-random}"

args=(
  --name "${RUN_NAME}"
  --seed "${SEED}"
  --finetune_eval_mode "${FINETUNE_EVAL_MODE}"
  --img_aug
  --batch_size 64
  --MLM
  --dataset_name "${DATASET_NAME}"
  --loss_names "${LOSS_NAMES}"
  --id_loss_weight "${ID_LOSS_WEIGHT}"
  --bridge_loss_weight "${BRIDGE_LOSS_WEIGHT}"
  --bridge_mode "${BRIDGE_MODE}"
  --bridge_gate_min "${BRIDGE_GATE_MIN}"
  --bridge_gate_tau "${BRIDGE_GATE_TAU}"
  --image_pooling "${IMAGE_POOLING}"
  --foreground_pool_mix "${FOREGROUND_POOL_MIX}"
  --foreground_pool_tau "${FOREGROUND_POOL_TAU}"
  --foreground_center_weight "${FOREGROUND_CENTER_WEIGHT}"
  --foreground_depth_weight "${FOREGROUND_DEPTH_WEIGHT}"
  --train_samples_per_id "${TRAIN_SAMPLES_PER_ID}"
  --train_sample_strategy "${TRAIN_SAMPLE_STRATEGY}"
  --lr 5e-6
  --lr2 5e-5
  --num_epoch 60
  --root_dir "${DATA_ROOT}"
  --finetune "${FINETUNE_INIT}"
)

if [ "${USE_SWANLAB}" = "1" ]; then
  args+=(
    --use_swanlab
    --swanlab_project "${SWANLAB_PROJECT}"
    --swanlab_experiment "${SWANLAB_EXPERIMENT}"
    --swanlab_mode "${SWANLAB_MODE}"
  )
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python finetune.py "${args[@]}"
