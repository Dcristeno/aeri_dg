#!/usr/bin/env bash
set -euo pipefail

# Contribution 3: GAR-EM merge with the current five-expert pool.
# Defaults reproduce the current best R1 setting:
# OpenAI3 + RemoteCLIP + HardNeg, prior-adaptive with HardNeg h=0.05.

EXPERT_CONFIG="${EXPERT_CONFIG:-docs/merge/configs/currentbest_plus_hardneg.example.json}"
DATA_ROOT="${DATA_ROOT:-/home/wuyong/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/merge/currentbest_plus_hardneg_w005}"
DEVICE="${DEVICE:-cuda}"
TOPK="${TOPK:-10}"
WEIGHTS="${WEIGHTS:-0.71668,0.06992,0.0874,0.076,0.05}"
PRIOR_STRENGTH="${PRIOR_STRENGTH:-1.0}"
ADAPTIVE_TEMPERATURE="${ADAPTIVE_TEMPERATURE:-0.5}"

python tools/gar_em_score_fusion.py \
  --expert_config "${EXPERT_CONFIG}" \
  --root_dir "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --topk "${TOPK}" \
  --fixed_weights "${WEIGHTS}" \
  --prior_weights "${WEIGHTS}" \
  --prior_strength "${PRIOR_STRENGTH}" \
  --adaptive_temperature "${ADAPTIVE_TEMPERATURE}" \
  --device "${DEVICE}" \
  "$@"
