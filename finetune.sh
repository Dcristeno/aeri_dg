#!/bin/bash
set -euo pipefail

DATASET_NAME="${DATASET_NAME:-AERI-PEDES}"
DATA_ROOT="${DATA_ROOT:-/home/wuyong/datasets}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
FINETUNE_INIT="${FINETUNE_INIT:-/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth}"
RUN_NAME="${RUN_NAME:-cfan_finetune}"
SEED="${SEED:-1}"
FINETUNE_EVAL_MODE="${FINETUNE_EVAL_MODE:-test}"
LOSS_NAMES="${LOSS_NAMES:-cda+fta}"
TRAIN_SAMPLES_PER_ID="${TRAIN_SAMPLES_PER_ID:-0}"
FINETUNE_VAL_RATIO="${FINETUNE_VAL_RATIO:-0.1}"
FINETUNE_VAL_SEED="${FINETUNE_VAL_SEED:-1}"
TRAIN_SAMPLE_STRATEGY="${TRAIN_SAMPLE_STRATEGY:-random}"
TRAIN_SAMPLE_MID_RATIO="${TRAIN_SAMPLE_MID_RATIO:-0.6}"
TRAIN_SAMPLE_REPRESENTATIVE_WEIGHT="${TRAIN_SAMPLE_REPRESENTATIVE_WEIGHT:-1.0}"
TRAIN_SAMPLE_DIVERSITY_WEIGHT="${TRAIN_SAMPLE_DIVERSITY_WEIGHT:-0.5}"
TRAIN_SAMPLE_MID_WEIGHT="${TRAIN_SAMPLE_MID_WEIGHT:-0.5}"
TRAIN_TILE_MIX_GRID="${TRAIN_TILE_MIX_GRID:-0}"
TRAIN_TILE_MIX_PROB="${TRAIN_TILE_MIX_PROB:-0.0}"
BRIDGE_LOSS_WEIGHT="${BRIDGE_LOSS_WEIGHT:-2.0}"
BRIDGE_PAIR_WEIGHT="${BRIDGE_PAIR_WEIGHT:-1.0}"
BRIDGE_DISTILL_WEIGHT="${BRIDGE_DISTILL_WEIGHT:-1.0}"
BRIDGE_DISTILL_TEMP="${BRIDGE_DISTILL_TEMP:-0.07}"
AERIAL_PROTOTYPE_PATH="${AERIAL_PROTOTYPE_PATH:-}"
AERIAL_PROTOTYPE_WEIGHT="${AERIAL_PROTOTYPE_WEIGHT:-0.0}"
TRACK_MEMORY_LOSS_WEIGHT="${TRACK_MEMORY_LOSS_WEIGHT:-0.0}"
TRACK_MEMORY_IMAGE_WEIGHT="${TRACK_MEMORY_IMAGE_WEIGHT:-1.0}"
TRACK_MEMORY_TEXT_WEIGHT="${TRACK_MEMORY_TEXT_WEIGHT:-1.0}"
TRACK_MEMORY_MOMENTUM="${TRACK_MEMORY_MOMENTUM:-0.8}"
USE_SWANLAB="${USE_SWANLAB:-0}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-CFAN}"
SWANLAB_EXPERIMENT="${SWANLAB_EXPERIMENT:-cfan_aeri_finetune}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
FTA_NUM_QUERY="${FTA_NUM_QUERY:-4}"
FTA_QUERY_MODE="${FTA_QUERY_MODE:-static}"
FTA_QUERY_CONDITION_SCALE="${FTA_QUERY_CONDITION_SCALE:-1.0}"
CVPR_NUM_QUERY="${CVPR_NUM_QUERY:-1}"
CVPR_TOPK="${CVPR_TOPK:-16}"
CVPR_NUM_STAGES="${CVPR_NUM_STAGES:-3}"
CVPR_MOMENTUM="${CVPR_MOMENTUM:-0.7}"
CVPR_LOSS_WEIGHT="${CVPR_LOSS_WEIGHT:-1.0}"

args=(
  --name "${RUN_NAME}"
  --seed "${SEED}"
  --finetune_eval_mode "${FINETUNE_EVAL_MODE}"
  --img_aug
  --batch_size 64
  --MLM
  --dataset_name "${DATASET_NAME}"
  --loss_names "${LOSS_NAMES}"
  --finetune_val_ratio "${FINETUNE_VAL_RATIO}"
  --finetune_val_seed "${FINETUNE_VAL_SEED}"
  --train_samples_per_id "${TRAIN_SAMPLES_PER_ID}"
  --train_sample_strategy "${TRAIN_SAMPLE_STRATEGY}"
  --train_sample_mid_ratio "${TRAIN_SAMPLE_MID_RATIO}"
  --train_sample_representative_weight "${TRAIN_SAMPLE_REPRESENTATIVE_WEIGHT}"
  --train_sample_diversity_weight "${TRAIN_SAMPLE_DIVERSITY_WEIGHT}"
  --train_sample_mid_weight "${TRAIN_SAMPLE_MID_WEIGHT}"
  --train_tile_mix_grid "${TRAIN_TILE_MIX_GRID}"
  --train_tile_mix_prob "${TRAIN_TILE_MIX_PROB}"
  --bridge_loss_weight "${BRIDGE_LOSS_WEIGHT}"
  --bridge_pair_weight "${BRIDGE_PAIR_WEIGHT}"
  --bridge_distill_weight "${BRIDGE_DISTILL_WEIGHT}"
  --bridge_distill_temp "${BRIDGE_DISTILL_TEMP}"
  --aerial_prototype_path "${AERIAL_PROTOTYPE_PATH}"
  --aerial_prototype_weight "${AERIAL_PROTOTYPE_WEIGHT}"
  --track_memory_loss_weight "${TRACK_MEMORY_LOSS_WEIGHT}"
  --track_memory_image_weight "${TRACK_MEMORY_IMAGE_WEIGHT}"
  --track_memory_text_weight "${TRACK_MEMORY_TEXT_WEIGHT}"
  --track_memory_momentum "${TRACK_MEMORY_MOMENTUM}"
  --fta_num_query "${FTA_NUM_QUERY}"
  --fta_query_mode "${FTA_QUERY_MODE}"
  --fta_query_condition_scale "${FTA_QUERY_CONDITION_SCALE}"
  --cvpr_num_query "${CVPR_NUM_QUERY}"
  --cvpr_topk "${CVPR_TOPK}"
  --cvpr_num_stages "${CVPR_NUM_STAGES}"
  --cvpr_momentum "${CVPR_MOMENTUM}"
  --cvpr_loss_weight "${CVPR_LOSS_WEIGHT}"
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
