# RemoteCLIP / GeoCLIP Direction

## Decision

For merge direction two, prioritize RemoteCLIP first.

RemoteCLIP is closer to AERI-PEDES because it is trained for remote-sensing vision-language alignment and should provide complementary aerial layout, land-cover, building, road, and overhead-view cues. GeoCLIP is useful conceptually, but its primary objective is image-to-GPS/geolocation alignment, so it is less directly compatible with the current text-to-image retrieval pipeline.

## First Expert

Use RemoteCLIP ViT-B-32 as the first different-pretraining expert.

Recommended first run:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=none \
USE_SWANLAB=1 \
RUN_NAME='aeri_remoteclip_vit_b32_cda_r1' \
SWANLAB_EXPERIMENT='aeri_remoteclip_vit_b32_cda_r1' \
PRETRAIN_CHOICE='/path/to/RemoteCLIP-ViT-B-32.pt' \
STRIDE_SIZE=32 \
LOSS_NAMES='cda' \
TRAIN_SAMPLES_PER_ID=0 \
BEST_METRIC=R1 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

If CDA converges, run the full recipe:

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=none \
USE_SWANLAB=1 \
RUN_NAME='aeri_remoteclip_vit_b32_full_r1' \
SWANLAB_EXPERIMENT='aeri_remoteclip_vit_b32_full_r1' \
PRETRAIN_CHOICE='/path/to/RemoteCLIP-ViT-B-32.pt' \
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

Do not load the HAM ViT-B/16 initialization for RemoteCLIP:

```bash
FINETUNE_INIT=none
```

## Merge Test

After training, add the RemoteCLIP expert to the GAR-EM pool with a small prior:

```text
OpenAI ViT-B/16 full, OpenAI ViT-B/32 full, RemoteCLIP ViT-B/32 full
0.82,0.08,0.10
0.86,0.08,0.06
0.90,0.06,0.04
```

The first goal is not to make RemoteCLIP the strongest single model. The goal is to verify whether a remote-sensing pretraining source recovers positive samples missed by the OpenAI CLIP experts and improves R1 or mAP/mINP under GAR-EM prior-adaptive fusion.

## GeoCLIP Position

Keep GeoCLIP as a second-stage discussion or appendix experiment. It is better framed as geographic prior knowledge rather than a direct image-text retrieval expert. Only add it after RemoteCLIP has a clean training and fusion result.
