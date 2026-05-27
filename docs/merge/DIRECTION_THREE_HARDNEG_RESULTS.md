# Merge Direction Three: Hard Negative Expert Results

## Setup

Direction three tests a different training objective expert:

```text
HardNeg = ViT-B/16 + cda+hardneg + random k=2
```

HardNeg checkpoint:

```text
/home/wuyong/aeri_dg/logs/AERI-PEDES/20260527_154906_aeri_cda_hardneg_k2_vit_b16_r1/best0.pth
```

SwanLab:

```text
https://swanlab.cn/@Dcristen/CFAN/runs/fc16oax7u8pi2fnxls1pe
```

Single-expert best:

| expert | best epoch | R1 | R5 | R10 | RSum | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HardNeg ViT-B/16 k2 | 59 | 48.103 | 66.113 | 75.004 | 189.220 | 46.031 | 33.031 |

## Fusion Baseline

The previous best result is direction two:

```text
CurrentBest = OpenAI3 fixed meta + RemoteCLIP r=0.08
```

Expanded weights:

```text
OpenAI ViT-B/16 = 0.7544
OpenAI ViT-B/32 = 0.0736
OpenAI ViT-L/14 = 0.0920
RemoteCLIP      = 0.0800
```

Previous best metrics:

| method | R1 | R5 | R10 | RSum | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CurrentBest | 49.959 | 67.204 | 75.444 | 192.607 | 48.383 | 35.844 |

## Five-Expert Fusion

Add HardNeg with prior `h`:

```text
[(1-h)*0.7544, (1-h)*0.0736, (1-h)*0.092, (1-h)*0.08, h]
```

Expert pool:

| expert | source |
| --- | --- |
| OpenAI ViT-B/16 full | `/home/wuyong/cfan_code/logs/AERI-PEDES/20260521_081034_aeri_cda_fta_bridge_pair_k2_w2p0/best0.pth` |
| OpenAI ViT-B/32 full | `/home/wuyong/cfan_code/logs/AERI-PEDES/20260522_164259_aeri_full_vit_b32_garem/best0.pth` |
| OpenAI ViT-L/14 full | `/home/wuyong/cfan_code/logs/AERI-PEDES/20260522_222323_aeri_full_vit_l14_garem_b16/best0.pth` |
| RemoteCLIP ViT-B/32 full | `/home/wuyong/cfan_code/logs/AERI-PEDES/20260525_141417_aeri_remoteclip_vit_b32_full_r1/best0.pth` |
| HardNeg ViT-B/16 k2 | `/home/wuyong/aeri_dg/logs/AERI-PEDES/20260527_154906_aeri_cda_hardneg_k2_vit_b16_r1/best0.pth` |

Config template:

```text
configs/currentbest_plus_hardneg.example.json
```

## Sweep Results

| HardNeg h | method | R1 | R5 | R10 | RSum | mAP | mINP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.03 | fixed | 50.057 | 67.351 | 75.379 | 192.786 | 48.452 | 35.930 |
| 0.03 | GAR-EM adaptive | 50.024 | 65.934 | 74.320 | 190.278 | **49.250** | **37.686** |
| 0.03 | GAR-EM prior-adaptive | 50.334 | 66.797 | 74.955 | 192.086 | 48.501 | 35.930 |
| 0.05 | fixed | 50.138 | 67.367 | 75.541 | 193.047 | 48.502 | 35.988 |
| 0.05 | GAR-EM adaptive | 50.024 | 65.934 | 74.320 | 190.278 | **49.250** | **37.686** |
| 0.05 | GAR-EM prior-adaptive | **50.513** | 66.862 | 74.972 | 192.347 | 48.599 | 36.039 |
| 0.08 | fixed | 50.122 | **67.399** | **75.541** | **193.063** | 48.546 | 36.083 |
| 0.08 | GAR-EM adaptive | 50.024 | 65.934 | 74.320 | 190.278 | **49.250** | **37.686** |
| 0.08 | GAR-EM prior-adaptive | 50.464 | 66.764 | 74.874 | 192.102 | 48.671 | 36.085 |

## Current Best

Best R1:

```text
CurrentBest + HardNeg h=0.05, GAR-EM prior-adaptive
R1 = 50.513
```

Gain over previous best:

```text
+0.554 R1
+0.216 mAP
+0.195 mINP
```

The strongest fixed-weight result is:

```text
CurrentBest + HardNeg h=0.05, fixed
R1 = 50.138
RSum = 193.047
```

The `h=0.08` fixed run gives the best R5/R10/RSum in this sweep, while prior-adaptive `h=0.05` gives the best R1.

## Interpretation

HardNeg is a strong direction-three expert. It is not the strongest single model, but it contributes complementary ranking evidence when fused with the current best pretraining pool. This supports the paper claim that different training objectives produce useful expert diversity beyond backbone and pretraining diversity.
