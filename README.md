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

To try Auto Cherry-Picker inspired caption cherry-picking with strong caption
loss weight, run:

```bash
RUN_NAME='aeri_k2_ground_bridge_caption_cherry_w3_seed2' \
SWANLAB_EXPERIMENT='aeri_k2_ground_bridge_caption_cherry_w3_seed2' \
CAPTION_CHERRY_MODE='template' \
CAPTION_CHERRY_EXTRA_PER_SAMPLE=2 \
CAPTION_CHERRY_WEIGHT=3.0 \
TRAIN_SAMPLE_STRATEGY='cherry_weighted' \
SEED=2 \
bash finetune.sh
```

This mode keeps the image model and bridge unchanged. It adds selected
attribute-preserving caption variants to the train set, samples them more often,
and applies a larger SDM/ID loss weight when a cherry-picked caption is used.

To train with Auto Cherry-Picker style synthetic image samples, first prepare a
filtered JSON manifest:

```json
[
  {
    "pid": 0,
    "aerial_img": "synthetic/aerial/person_000_a.jpg",
    "ground_img": "synthetic/ground/person_000_g.jpg",
    "caption": "a person wearing ...",
    "score": 0.92
  }
]
```

Then run:

```bash
RUN_NAME='aeri_k2_ground_bridge_synth_cherry_w4_seed2' \
SWANLAB_EXPERIMENT='aeri_k2_ground_bridge_synth_cherry_w4_seed2' \
SYNTHETIC_CHERRY_MANIFEST='/home/wuyong/data/aeri_synthetic/cherry_manifest.json' \
SYNTHETIC_CHERRY_MIN_SCORE=0.8 \
SYNTHETIC_CHERRY_MAX_PER_PID=2 \
SYNTHETIC_CHERRY_WEIGHT=4.0 \
TRAIN_SAMPLE_STRATEGY='cherry_weighted' \
SEED=2 \
bash finetune.sh
```

The manifest is expected to contain already generated and scored candidates.
Training keeps only high-score samples, mixes them into the finetune set, and
applies a larger SDM/ID loss weight through `SYNTHETIC_CHERRY_WEIGHT`.

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
