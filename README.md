# AERI K=2 SDM+ID+MLM

This branch starts from the `aeri-k2-sdm-id` experiment and adds masked language modeling:

```text
base SDM
+ ID loss
+ MLM loss
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

The default training objective is `LOSS_NAMES=base+id+mlm`:

```text
base_loss =
  SDM(aerial, text)
+ SDM(ground, text)

id_loss =
  CE(aerial_pid)
+ CE(ground_pid)
+ CE(text_pid)

mlm_loss =
  CE(masked_tokens | aerial_tokens)
```

The default sampler rebuilds the finetune train loader each epoch with `TRAIN_SAMPLES_PER_ID=2`, randomly choosing two samples per identity before shuffling.
The default ID weight is `ID_LOSS_WEIGHT=0.5`, matching the stronger setting from the previous ID-weight sweep.
The default MLM weight is `MLM_LOSS_WEIGHT=2.0`, giving the language reconstruction task a stronger role in this branch.

## Train

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
USE_SWANLAB=1 \
RUN_NAME='aeri_k2_sdm_id_mlm' \
SWANLAB_EXPERIMENT='aeri_k2_sdm_id_mlm' \
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
