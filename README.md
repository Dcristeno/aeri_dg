# AERI K=2 Route ID

This branch starts from the `aeri-k2-sdm-id` experiment and separates the retrieval and ID auxiliary routes:

```text
base SDM
+ ID loss through a separate ID projection route
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
  CE(id_proj(aerial)_pid)
+ CE(id_proj(ground)_pid)
+ CE(id_proj(text)_pid)
```

The SDM loss and test-time embeddings use `retrieval_proj`, while ID classification uses `id_proj -> classifier`.
Both projection heads are identity-initialized, so the model starts from the original feature space and can learn route-specific adjustments.
The default sampler rebuilds the finetune train loader each epoch with `TRAIN_SAMPLES_PER_ID=2`, randomly choosing two samples per identity before shuffling.
The default ID weight is `ID_LOSS_WEIGHT=0.5`.

## Train

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
USE_SWANLAB=1 \
RUN_NAME='aeri_k2_route_id' \
SWANLAB_EXPERIMENT='aeri_k2_route_id' \
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
