# CFAN Baseline Cleanup

This repo snapshot is a cleaned working copy of the author-provided CFAN baseline for `AERI-PEDES`.

## What was cleaned

- Removed hardcoded `CUDA_VISIBLE_DEVICES` from `finetune.py` and `test.py`.
- Added a safer evaluation entry that can load the author checkpoint directly.
- Normalized the legacy author loss naming `sdm+fa` to the finetune code's actual implementation names `cda+fta`.
- Added runnable shell scripts for finetuning and evaluation.
- Made SwanLab logging opt-in in both evaluation and finetuning scripts, so training does not stall on network reconnects by default.

## Expected dataset layout

Set `DATA_ROOT` to the parent directory that contains `AERI-PEDES`, for example:

```bash
/home/wuyong/datasets/AERI-PEDES
```

Inside `AERI-PEDES`, CFAN expects at least:

- `train_caption.json`
- `test_caption.json`
- image folders referenced by those json files

## Evaluate the provided author checkpoint

```bash
DATA_ROOT=/home/wuyong/datasets \
CHECKPOINT_PATH=/home/wuyong/data/weights/CFAN_weights/best0.pth \
CUDA_VISIBLE_DEVICES=0 \
bash eval_aeri_cfan.sh
```

The default config used by this script is [configs/aeri_cfan_baseline.yaml](configs/aeri_cfan_baseline.yaml).
To enable SwanLab logging for evaluation, add `USE_SWANLAB=1`.

## Finetune CFAN on AERI-PEDES

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

To enable SwanLab logging for finetuning, add `USE_SWANLAB=1`.

To run the cleaner `CDA+FTA` baseline with AERI per-ID sparse sampling, for example `k=2`, run:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
LOSS_NAMES='cda+fta' \
TRAIN_SAMPLES_PER_ID=2 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

To rerun the same configuration under a different random seed, add `SEED=<n>`, for example `SEED=2`.

Finetune now defaults back to per-epoch evaluation on the official test split:

```bash
FINETUNE_EVAL_MODE=test
```

If you want a different behavior, switch modes:

```bash
FINETUNE_EVAL_MODE=heldout
FINETUNE_VAL_RATIO=0.1
FINETUNE_VAL_SEED=1
```

```bash
FINETUNE_EVAL_MODE=none
```

`heldout` uses a train-identity split for model selection, and `none` trains on the full train split and saves `final.pth` without intermediate validation.

To switch FTA from the original static learned queries to instance-conditioned queries, add:

```bash
FTA_QUERY_MODE='conditioned' \
FTA_NUM_QUERY=4 \
FTA_QUERY_CONDITION_SCALE=1.0 \
```

In conditioned mode, the model keeps the global query bank, but also predicts modality-specific query deltas from the current aerial/text sample features before cross attention.

To replace random `k=2` sampling with a simple heuristic selector that keeps the sharpest aerial images per ID, run:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
LOSS_NAMES='cda' \
TRAIN_SAMPLES_PER_ID=2 \
TRAIN_SAMPLE_STRATEGY='sharpness_topk' \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

To use a safer heuristic that prefers medium-sharpness aerial images, penalizes redundancy, and keeps more representative views per ID, run:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
LOSS_NAMES='cda' \
TRAIN_SAMPLES_PER_ID=2 \
TRAIN_SAMPLE_STRATEGY='mid_sharpness_diverse' \
TRAIN_SAMPLE_MID_RATIO=0.6 \
TRAIN_SAMPLE_REPRESENTATIVE_WEIGHT=1.0 \
TRAIN_SAMPLE_DIVERSITY_WEIGHT=0.5 \
TRAIN_SAMPLE_MID_WEIGHT=0.5 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

To keep the strong random `k=2` baseline while discouraging near-duplicate aerial frames, run:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
LOSS_NAMES='cda' \
TRAIN_SAMPLES_PER_ID=2 \
TRAIN_SAMPLE_STRATEGY='random_diverse' \
TRAIN_SAMPLE_DIVERSITY_WEIGHT=1.0 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

To aggressively mosaic aerial images by randomly mixing `8x8` tiles from two images of the same pid on every training sample, add:

```bash
TRAIN_TILE_MIX_GRID=8 \
TRAIN_TILE_MIX_PROB=1.0 \
```

This keeps the original pid label, but rebuilds the aerial image by randomly stitching together local regions from the two same-pid aerial samples that exist in the current epoch-sampled train set.

To enable the stronger ground-to-aerial bridge loss for AERI experiments, run:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
LOSS_NAMES='cda+bridge' \
TRAIN_SAMPLES_PER_ID=2 \
BRIDGE_LOSS_WEIGHT=2.0 \
BRIDGE_PAIR_WEIGHT=1.0 \
BRIDGE_DISTILL_WEIGHT=1.0 \
BRIDGE_DISTILL_TEMP=0.07 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

The strong bridge loss first pulls each aerial feature toward its detached ground-view teacher, then distills the full ground-text relation matrix into the aerial-text relation matrix.

To run the `pair only` ablation, set:

```bash
USE_SWANLAB=1 \
RUN_NAME='aeri_cda_bridge_pair_only_k2' \
SWANLAB_EXPERIMENT='aeri_cda_bridge_pair_only_k2' \
LOSS_NAMES='cda+bridge' \
TRAIN_SAMPLES_PER_ID=2 \
BRIDGE_LOSS_WEIGHT=2.0 \
BRIDGE_PAIR_WEIGHT=1.0 \
BRIDGE_DISTILL_WEIGHT=0.0 \
BRIDGE_DISTILL_TEMP=0.07 \
bash finetune.sh
```

To run the `distill only` ablation, set:

```bash
USE_SWANLAB=1 \
RUN_NAME='aeri_cda_bridge_distill_only_k2' \
SWANLAB_EXPERIMENT='aeri_cda_bridge_distill_only_k2' \
LOSS_NAMES='cda+bridge' \
TRAIN_SAMPLES_PER_ID=2 \
BRIDGE_LOSS_WEIGHT=2.0 \
BRIDGE_PAIR_WEIGHT=0.0 \
BRIDGE_DISTILL_WEIGHT=1.0 \
BRIDGE_DISTILL_TEMP=0.07 \
bash finetune.sh
```

To build fixed aerial trajectory prototypes from the initialization checkpoint, run:

```bash
python build_aerial_prototypes.py \
  --root_dir /home/wuyong/datasets \
  --finetune /home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
  --output /home/wuyong/data/HAM/HAM_checkpoint/aeri_train_aerial_prototypes.pth
```

Then train with the prototype teacher while keeping `random k=2` unchanged:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
LOSS_NAMES='cda+proto' \
TRAIN_SAMPLES_PER_ID=2 \
AERIAL_PROTOTYPE_PATH=/home/wuyong/data/HAM/HAM_checkpoint/aeri_train_aerial_prototypes.pth \
AERIAL_PROTOTYPE_WEIGHT=0.3 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

This keeps the random sparse sampler, but gives each sampled aerial image an extra pull toward the fixed trajectory summary of its identity.

To try an online track-memory teacher instead of a fixed prototype, add a `track` loss and let each pid maintain an EMA-updated aerial memory during training:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
LOSS_NAMES='cda+fta+bridge+track' \
TRAIN_SAMPLES_PER_ID=2 \
BRIDGE_LOSS_WEIGHT=2.0 \
BRIDGE_PAIR_WEIGHT=1.0 \
BRIDGE_DISTILL_WEIGHT=0.0 \
TRACK_MEMORY_LOSS_WEIGHT=0.1 \
TRACK_MEMORY_IMAGE_WEIGHT=1.0 \
TRACK_MEMORY_TEXT_WEIGHT=1.0 \
TRACK_MEMORY_MOMENTUM=0.8 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

This keeps `random k=2`, but adds trajectory-level supervision by aligning both the sampled aerial feature and its text feature to an online EMA memory for the current identity.

## AERI MoE adapter with k=2

Run the clean triad baseline plus the AERI MoE adapter:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
USE_SWANLAB=1 \
RUN_NAME='aeri_triad_k2_moe_w0p5' \
SWANLAB_EXPERIMENT='aeri_triad_k2_moe_w0p5' \
SEED=1 \
LOSS_NAMES='triad+moe' \
TRAIN_SAMPLES_PER_ID=2 \
TRAIN_SAMPLE_STRATEGY='random' \
MOE_LOSS_WEIGHT=0.5 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

The MoE branch keeps the original CLS features as the backbone signal, then adds a small token-level MoE adapter residual for aerial, ground, and text features before applying auxiliary tri-modal SDM alignment.

## Notes on the provided weights

The provided checkpoint contains finetune-only modules such as `query` and `mlp_logsigma2`, so evaluation should use `build_finetune_model`. The cleaned `test.py` now auto-detects this from the checkpoint.
