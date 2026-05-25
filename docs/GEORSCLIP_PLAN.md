# GeoRSCLIP Direction

## Decision

After RemoteCLIP, the next different-pretraining expert is GeoRSCLIP.

GeoRSCLIP is trained on RS5M, a large remote-sensing image-text dataset, and is more directly compatible with AERI-PEDES text-image retrieval than GeoCLIP, whose main signal is geographic location/GPS alignment. This makes GeoRSCLIP a better second remote-sensing pretraining expert for merge direction two.

## Model Alias

The code now supports HuggingFace-hosted GeoRSCLIP aliases through `PRETRAIN_CHOICE`:

```text
GeoRSCLIP-ViT-B/32
GeoRSCLIP-ViT-B/32-RET2
GeoRSCLIP-ViT-B/32-RSICD
GeoRSCLIP-ViT-B/32-RSITMD
```

The first recommended model is:

```text
GeoRSCLIP-ViT-B/32
```

It resolves to:

```text
repo_id=Zilun/GeoRSCLIP
filename=ckpt/RS5M_ViT-B-32.pt
```

If HuggingFace is slow, use:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## First Run: CDA Sanity Check

Run CDA first to verify that the checkpoint loads and adapts:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=none \
USE_SWANLAB=1 \
RUN_NAME='aeri_georsclip_vit_b32_cda_r1' \
SWANLAB_EXPERIMENT='aeri_georsclip_vit_b32_cda_r1' \
PRETRAIN_CHOICE='GeoRSCLIP-ViT-B/32' \
STRIDE_SIZE=32 \
LOSS_NAMES='cda' \
TRAIN_SAMPLES_PER_ID=2 \
TRAIN_SAMPLE_STRATEGY='random' \
BEST_METRIC=R1 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

Use `FINETUNE_INIT=none` because the HAM initialization is ViT-B/16 and should not be loaded into the GeoRSCLIP ViT-B/32 expert.

## Full Recipe

If CDA converges, run the full expert:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=none \
USE_SWANLAB=1 \
RUN_NAME='aeri_georsclip_vit_b32_full_r1' \
SWANLAB_EXPERIMENT='aeri_georsclip_vit_b32_full_r1' \
PRETRAIN_CHOICE='GeoRSCLIP-ViT-B/32' \
STRIDE_SIZE=32 \
LOSS_NAMES='cda+fta+bridge' \
TRAIN_SAMPLES_PER_ID=2 \
TRAIN_SAMPLE_STRATEGY='random' \
BRIDGE_LOSS_WEIGHT=2.0 \
BRIDGE_PAIR_WEIGHT=1.0 \
BRIDGE_DISTILL_WEIGHT=0.0 \
BEST_METRIC=R1 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

## Merge Plan

Use the same two-stage interpretation as RemoteCLIP:

```text
Stage 1: OpenAI3 = ViT-B/16 + ViT-B/32 + ViT-L/14, prior 0.82,0.08,0.10
Stage 2: OpenAI3 + GeoRSCLIP
```

For GeoRSCLIP prior `r`, expand the weights as:

```text
[(1-r)*0.82, (1-r)*0.08, (1-r)*0.10, r]
```

Recommended scan:

```text
r = 0.02, 0.04, 0.06, 0.08, 0.10, 0.12
```

RemoteCLIP peaked around `r=0.08` for R1, so GeoRSCLIP should start from the same range.

## Reporting

Compare GeoRSCLIP against:

```text
OpenAI3 GAR-EM prior: R1=49.813, mAP=48.307, mINP=35.797
OpenAI3 + RemoteCLIP fixed r=0.08: R1=49.959, mAP=48.383, mINP=35.844
```

If GeoRSCLIP improves over RemoteCLIP, it becomes the direction-two main result. If it improves only mAP/mINP, report it as another remote-sensing pretraining source that helps tail ranking.
