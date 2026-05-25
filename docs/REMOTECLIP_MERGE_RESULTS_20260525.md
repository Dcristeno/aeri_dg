# RemoteCLIP Merge Results - 2026-05-25

## Setup

This records merge direction two: different pretraining.

RemoteCLIP expert:

```text
RemoteCLIP ViT-B-32 + cda+fta+bridge + random k=2
pretrain_choice=/home/wuyong/pretrained/RemoteCLIP-ViT-B-32.pt
stride_size=32
checkpoint=logs/AERI-PEDES/20260525_141417_aeri_remoteclip_vit_b32_full_r1/best0.pth
config=logs/AERI-PEDES/20260525_141417_aeri_remoteclip_vit_b32_full_r1/configs.yaml
SwanLab=https://swanlab.cn/@Dcristen/CFAN/runs/8950kjm0ex1vv7rp0i5yg
```

RemoteCLIP single-expert best:

| expert | best epoch | R1 |
| --- | ---: | ---: |
| RemoteCLIP ViT-B-32 full | 36 | 34.294 |

## Two-Stage Interpretation

Use a two-stage interpretation for clarity:

```text
Stage 1: OpenAI3 = OpenAI ViT-B/16 + OpenAI ViT-B/32 + OpenAI ViT-L/14
Stage 2: OpenAI3 + RemoteCLIP
```

OpenAI3 prior:

```text
0.82,0.08,0.10
```

Expanded four-expert weights for RemoteCLIP prior `r`:

```text
[(1-r)*0.82, (1-r)*0.08, (1-r)*0.10, r]
```

Expert pool:

| expert | checkpoint |
| --- | --- |
| openai_vit_b16_full | `logs/AERI-PEDES/20260521_081034_aeri_cda_fta_bridge_pair_k2_w2p0/best0.pth` |
| openai_vit_b32_full | `logs/AERI-PEDES/20260522_164259_aeri_full_vit_b32_garem/best0.pth` |
| openai_vit_l14_full | `logs/AERI-PEDES/20260522_222323_aeri_full_vit_l14_garem_b16/best0.pth` |
| remoteclip_vit_b32_full | `logs/AERI-PEDES/20260525_141417_aeri_remoteclip_vit_b32_full_r1/best0.pth` |

## Baseline

| method | weights | R1 | R5 | R10 | RSum | mAP | mINP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenAI3 GAR-EM prior | `0.82,0.08,0.10` | 49.813 | 66.650 | 74.955 | 191.418 | 48.307 | 35.797 |

## Fixed Fusion Sweep

| RemoteCLIP r | fixed weights | R1 | R5 | R10 | RSum | mAP | mINP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | `0.8118,0.0792,0.099,0.01` | 49.650 | 67.546 | 75.590 | 192.786 | 48.165 | 35.536 |
| 0.02 | `0.8036,0.0784,0.098,0.02` | 49.731 | 67.546 | 75.639 | 192.916 | 48.207 | 35.607 |
| 0.04 | `0.7872,0.0768,0.096,0.04` | 49.699 | 67.334 | 75.590 | 192.623 | 48.265 | 35.715 |
| 0.05 | `0.779,0.076,0.095,0.05` | 49.796 | 67.334 | 75.541 | 192.672 | 48.316 | 35.747 |
| 0.06 | `0.7708,0.0752,0.094,0.06` | 49.862 | 67.302 | 75.574 | 192.737 | 48.337 | 35.763 |
| 0.08 | `0.7544,0.0736,0.092,0.08` | **49.959** | 67.204 | 75.444 | 192.607 | 48.383 | 35.844 |
| 0.10 | `0.738,0.072,0.09,0.10` | 49.943 | 67.106 | 75.362 | 192.412 | **48.394** | 35.891 |
| 0.12 | `0.7216,0.0704,0.088,0.12` | 49.927 | 67.139 | 75.151 | 192.216 | 48.387 | 35.928 |
| 0.14 | `0.7052,0.0688,0.086,0.14` | 49.862 | 66.911 | 75.053 | 191.825 | 48.373 | 35.926 |
| 0.16 | `0.6888,0.0672,0.084,0.16` | 49.829 | 66.846 | 75.085 | 191.760 | 48.367 | **35.989** |
| 0.20 | `0.656,0.064,0.08,0.20` | 49.894 | 66.748 | 74.711 | 191.353 | 48.284 | 35.956 |

## Prior-Adaptive Observation

Prior-adaptive did not improve R1 in this sweep. Its strongest tail-ranking signal appeared around `r=0.08`:

| RemoteCLIP r | prior-adaptive R1 | prior-adaptive mAP | prior-adaptive mINP |
| ---: | ---: | ---: | ---: |
| 0.08 | 49.585 | 48.403 | 36.204 |

## Conclusion

Best R1:

```text
OpenAI3 + RemoteCLIP fixed r=0.08
R1=49.959
```

Gain over OpenAI3 GAR-EM prior:

```text
+0.146 R1
+0.076 mAP
+0.047 mINP
```

RemoteCLIP should remain an auxiliary different-pretraining expert. The useful R1 range is around `r=0.06-0.10`. Larger RemoteCLIP weights can continue to help mINP slightly, but reduce R5/R10/RSum.
