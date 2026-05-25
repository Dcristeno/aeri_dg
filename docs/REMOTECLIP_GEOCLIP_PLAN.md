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

## 2026-05-25 Results

### RemoteCLIP Expert Training

RemoteCLIP ViT-B-32 was trained with the full recipe:

```text
pretrain_choice=/home/wuyong/pretrained/RemoteCLIP-ViT-B-32.pt
stride_size=32
loss_names=cda+fta+bridge
train_samples_per_id=2
bridge_loss_weight=2.0
bridge_pair_weight=1.0
bridge_distill_weight=0.0
```

Run:

```text
aeri_remoteclip_vit_b32_full_r1
SwanLab: https://swanlab.cn/@Dcristen/CFAN/runs/8950kjm0ex1vv7rp0i5yg
checkpoint: logs/AERI-PEDES/20260525_141417_aeri_remoteclip_vit_b32_full_r1/best0.pth
config: logs/AERI-PEDES/20260525_141417_aeri_remoteclip_vit_b32_full_r1/configs.yaml
```

Single expert best:

| expert | epoch | R1 | R5 | R10 | RSum | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RemoteCLIP ViT-B-32 full | 36 | 34.294 | - | - | - | - | - |

The RemoteCLIP expert is much weaker than the OpenAI ViT-B/16 main expert in top-1 retrieval, but it is still useful as a weak auxiliary expert because it carries remote-sensing pretraining knowledge.

### Two-Stage Fusion Setup

For direction two, use a clearer two-stage interpretation:

```text
Stage 1: OpenAI backbone expert group
ViT-B/16 + ViT-B/32 + ViT-L/14, prior = 0.82,0.08,0.10

Stage 2: different-pretraining expert
OpenAI-group + RemoteCLIP
```

The actual score-fusion tool receives the expanded four-expert weights:

```text
[(1-r)*0.82, (1-r)*0.08, (1-r)*0.10, r]
```

where `r` is the RemoteCLIP prior.

### Fusion Sweep

Expert pool:

```text
openai_vit_b16_full:
  logs/AERI-PEDES/20260521_081034_aeri_cda_fta_bridge_pair_k2_w2p0/best0.pth
openai_vit_b32_full:
  logs/AERI-PEDES/20260522_164259_aeri_full_vit_b32_garem/best0.pth
openai_vit_l14_full:
  logs/AERI-PEDES/20260522_222323_aeri_full_vit_l14_garem_b16/best0.pth
remoteclip_vit_b32_full:
  logs/AERI-PEDES/20260525_141417_aeri_remoteclip_vit_b32_full_r1/best0.pth
```

Baseline OpenAI three-expert result:

| method | weights | R1 | R5 | R10 | RSum | mAP | mINP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GAR-EM prior | 0.82,0.08,0.10 | 49.813 | 66.650 | 74.955 | 191.418 | 48.307 | 35.797 |

RemoteCLIP fusion sweep:

| RemoteCLIP r | fixed weights | fixed R1 | fixed R5 | fixed R10 | fixed RSum | fixed mAP | fixed mINP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.8118,0.0792,0.099,0.01 | 49.650 | 67.546 | 75.590 | 192.786 | 48.165 | 35.536 |
| 0.02 | 0.8036,0.0784,0.098,0.02 | 49.731 | 67.546 | 75.639 | 192.916 | 48.207 | 35.607 |
| 0.04 | 0.7872,0.0768,0.096,0.04 | 49.699 | 67.334 | 75.590 | 192.623 | 48.265 | 35.715 |
| 0.05 | 0.779,0.076,0.095,0.05 | 49.796 | 67.334 | 75.541 | 192.672 | 48.316 | 35.747 |
| 0.06 | 0.7708,0.0752,0.094,0.06 | 49.862 | 67.302 | 75.574 | 192.737 | 48.337 | 35.763 |
| 0.08 | 0.7544,0.0736,0.092,0.08 | **49.959** | 67.204 | 75.444 | 192.607 | 48.383 | 35.844 |
| 0.10 | 0.738,0.072,0.09,0.10 | 49.943 | 67.106 | 75.362 | 192.412 | **48.394** | 35.891 |
| 0.12 | 0.7216,0.0704,0.088,0.12 | 49.927 | 67.139 | 75.151 | 192.216 | 48.387 | 35.928 |
| 0.14 | 0.7052,0.0688,0.086,0.14 | 49.862 | 66.911 | 75.053 | 191.825 | 48.373 | 35.926 |
| 0.16 | 0.6888,0.0672,0.084,0.16 | 49.829 | 66.846 | 75.085 | 191.760 | 48.367 | **35.989** |
| 0.20 | 0.656,0.064,0.08,0.20 | 49.894 | 66.748 | 74.711 | 191.353 | 48.284 | 35.956 |

Prior-adaptive did not improve R1 in this sweep. Its best mINP appeared around `r=0.08`, with:

| RemoteCLIP r | prior-adaptive R1 | prior-adaptive mAP | prior-adaptive mINP |
| ---: | ---: | ---: | ---: |
| 0.08 | 49.585 | 48.403 | 36.204 |

### Conclusion

The best R1 result for direction two is:

```text
OpenAI3 + RemoteCLIP fixed r=0.08
R1 = 49.959
```

Compared with the OpenAI three-expert GAR-EM prior result (`R1=49.813`), adding RemoteCLIP gives:

```text
+0.146 R1
+0.076 mAP
+0.047 mINP
```

This supports the claim that remote-sensing pretraining provides complementary retrieval evidence. The useful RemoteCLIP range is small-to-moderate (`r` around `0.06-0.10`). Larger RemoteCLIP weights continue to help mINP slightly, but reduce R5/R10/RSum, so RemoteCLIP should remain an auxiliary expert rather than the main retrieval prior.
