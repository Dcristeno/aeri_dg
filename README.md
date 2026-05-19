# AERI K=2 Ground Bridge Lite

This branch starts from the `aeri-k2-sdm-id` experiment and adds a lightweight detached ground teacher bridge:

```text
base SDM
+ ID loss
+ detached ground-to-aerial bridge loss
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

bridge_loss =
  1 - cosine(aerial_feature, ground_feature.detach())
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

To replace the plain bridge with confidence-gated bridge supervision, run:

```bash
RUN_NAME='aeri_k2_ground_bridge_gated_m02_t01_seed2' \
SWANLAB_EXPERIMENT='aeri_k2_ground_bridge_gated_m02_t01_seed2' \
BRIDGE_MODE='gated' \
BRIDGE_GATE_MIN=0.2 \
BRIDGE_GATE_TAU=0.1 \
SEED=2 \
bash finetune.sh
```

`BRIDGE_MODE=plain` is still the default and matches the current best line.

To try the Generative Photography inspired differential view bridge, run:

```bash
RUN_NAME='aeri_k2_ground_bridge_diff_aux_dw005_seed2' \
SWANLAB_EXPERIMENT='aeri_k2_ground_bridge_diff_aux_dw005_seed2' \
BRIDGE_MODE='differential' \
BRIDGE_DELTA_WEIGHT=0.05 \
SEED=2 \
bash finetune.sh
```

This mode follows the reusable part of
`pandayuanyu/generative-photography`: instead of importing the diffusion
pipeline, it borrows the camera-difference idea and treats the paired
`ground -> aerial` shift as a view residual. The plain detached ground bridge
stays as the main supervision, while a small residual head adds low-weight
auxiliary differential terms.

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
