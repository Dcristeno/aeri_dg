# AERI K=2 Ground Bridge Lite

This branch starts from the `aeri-k2-sdm-id` experiment and adds a lightweight detached ground teacher bridge:

```text
base SDM
+ ID loss
+ detached ground-to-aerial bridge loss
+ optional fuzzy token alignment loss
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

The default training script objective is `LOSS_NAMES=base+id+bridge`:

```text
base_loss =
  SDM(aerial, text)
+ SDM(ground, text)

id_loss =
  CE(aerial_pid)
+ CE(ground_pid)
+ CE(text_pid)

bridge_loss =
  1 - cosine(aerial_feature, ground_feature.detach())

fta_loss =
  SDM(fuzzy aerial tokens, fuzzy text tokens)
```

The default sampler rebuilds the finetune train loader each epoch with `TRAIN_SAMPLES_PER_ID=2`, randomly choosing two samples per identity before shuffling.
The default bridge settings are `ID_LOSS_WEIGHT=0.5` and `BRIDGE_LOSS_WEIGHT=0.5`.

## Train

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
USE_SWANLAB=1 \
RUN_NAME='aeri_k2_ground_bridge_lite' \
SWANLAB_EXPERIMENT='aeri_k2_ground_bridge_lite' \
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

To try fuzzy token alignment on top of the strong bridge baseline:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
USE_SWANLAB=1 \
RUN_NAME='aeri_k2_ground_bridge_fta' \
SWANLAB_EXPERIMENT='aeri_k2_ground_bridge_fta' \
LOSS_NAMES='base+id+bridge+fta' \
TRAIN_SAMPLES_PER_ID=2 \
TRAIN_SAMPLE_STRATEGY='random' \
ID_LOSS_WEIGHT=0.5 \
BRIDGE_LOSS_WEIGHT=2.0 \
FTA_LOSS_WEIGHT=0.5 \
FTA_NUM_QUERY=4 \
SEED=1 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

## Test

```bash
python test.py \
  --config_file logs/AERI-PEDES/<run_dir>/configs.yaml \
  --checkpoint logs/AERI-PEDES/<run_dir>/best0.pth \
  --root_dir /home/wuyong/datasets
```
