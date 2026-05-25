# Hard Negative Training Expert

## Goal

Direction three builds experts from different training objectives. The hard-negative expert focuses on separating visually similar candidates in the retrieval ranking.

Use the new `hardneg` loss token:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
USE_SWANLAB=1 \
RUN_NAME='aeri_cda_hardneg_vit_b16_r1' \
SWANLAB_EXPERIMENT='aeri_cda_hardneg_vit_b16_r1' \
PRETRAIN_CHOICE='ViT-B/16' \
LOSS_NAMES='cda+hardneg' \
HARD_NEGATIVE_LOSS_WEIGHT=1.0 \
HARD_NEGATIVE_MARGIN=0.2 \
HARD_NEGATIVE_TOPK=1 \
HARD_NEGATIVE_POSITIVE_REDUCE=max \
TRAIN_SAMPLES_PER_ID=0 \
BEST_METRIC=R1 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

## Notes

- Aliases accepted by `LOSS_NAMES`: `hardneg`, `hard_negative`, `hard-negative`, `hard`, and `hn`.
- The loss is PID-aware: same-PID samples are positives, and all same-PID candidates are removed from the negative pool.
- It optimizes both text-to-aerial and aerial-to-text directions.
- `HARD_NEGATIVE_TOPK` controls the hard-negative cluster size. `1` is the strict hardest-negative setting; `3` or `5` may be smoother for noisy batches.
- Recommended first expert for merge pool: `cda+hardneg` with the same backbone and initialization as the current OpenAI ViT-B/16 baseline, so the only intentional difference is the training objective.
