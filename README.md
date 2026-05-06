# AERI K=2 Representation Adapter

This branch starts from the `aeri-k2-sdm-id` experiment and adds a lightweight MMRL-inspired representation adapter:

```text
base SDM
+ ID loss
+ residual representation adapters
+ cosine regularization to the raw CLIP feature space
+ random per-ID k=2 training sampling
```

## Environment

Conda is recommended:

```bash
conda env create -f environment.yml
conda activate irra
```

If PyTorch needs to match a different CUDA version on a server, install the matching PyTorch build first, then install the remaining Python packages with:

```bash
pip install -r requirements.txt
```

The pinned configuration mirrors the verified server setup:

```text
Python 3.8.20
PyTorch 2.0.0 / CUDA 11.8
TorchVision 0.15.0
SwanLab 0.7.15
```

## Objective

The default training objective is `LOSS_NAMES=base+id`:

```text
base_loss =
  SDM(aerial, text)
+ SDM(ground, text)

id_loss =
  CE(aerial_pid)
+ CE(ground_pid)
+ CE(text_pid)

adapted_feature =
  REP_ALPHA * raw_clip_feature
+ (1 - REP_ALPHA) * adapter(raw_clip_feature)

rep_reg_loss =
  cosine regularization between adapted_feature and raw_clip_feature.detach()
```

The default sampler rebuilds the finetune train loader each epoch with `TRAIN_SAMPLES_PER_ID=2`, randomly choosing two samples per identity before shuffling.
The default adapter settings are `REP_ALPHA=0.7`, `REP_REG_WEIGHT=0.5`, and `ID_LOSS_WEIGHT=0.5`.

## Train

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
USE_SWANLAB=1 \
RUN_NAME='aeri_k2_rep_adapter' \
SWANLAB_EXPERIMENT='aeri_k2_rep_adapter' \
SEED=1 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

On the default server setup, this is equivalent to:

```bash
bash finetune.sh
```

Useful overrides:

```bash
LOSS_NAMES=base \
TRAIN_SAMPLES_PER_ID=0 \
RUN_NAME='aeri_base_fullsample' \
bash finetune.sh
```

## Test

```bash
python test.py \
  --config_file logs/AERI-PEDES/<run_dir>/configs.yaml \
  --checkpoint logs/AERI-PEDES/<run_dir>/best0.pth \
  --root_dir /home/wuyong/datasets
```
