# Paper Contributions

This project is organized around three main contributions for AERI-PEDES ground-aerial text-image retrieval.

## 1. Random k Sampling

**Goal:** reduce training cost while preserving per-identity diversity.

The baseline full sampler is expensive because each epoch traverses the full training set. Random k sampling draws a fixed number of samples per identity each epoch:

```text
TRAIN_SAMPLES_PER_ID=2
TRAIN_SAMPLE_STRATEGY=random
```

In the paper narrative, this is the first contribution because it makes training more efficient and gives a clean sampling module for ablation.

Code entry points:

```text
datasets/per_id_sampling.py
scripts/train_k_random.sh
```

Recommended command:

```bash
bash scripts/train_k_random.sh
```

## 2. FTA + Bridge

**Goal:** improve fine-grained text-image alignment and ground-to-aerial cross-view alignment.

FTA introduces fine-grained token/query alignment, while Bridge uses ground-view features as cross-view supervision for aerial retrieval. The strongest recorded recipe uses pair-only bridge:

```text
LOSS_NAMES=cda+fta+bridge
BRIDGE_LOSS_WEIGHT=2.0
BRIDGE_PAIR_WEIGHT=1.0
BRIDGE_DISTILL_WEIGHT=0.0
TRAIN_SAMPLES_PER_ID=2
```

In the paper narrative, this is the second contribution: a task-specific alignment objective for ground-aerial retrieval.

Code entry points:

```text
model/finetune_losses.py
model/build_finetune.py
scripts/train_fta_bridge.sh
```

Recommended command:

```bash
bash scripts/train_fta_bridge.sh
```

Reference single-expert result:

| method | R1 | R5 | R10 | RSum | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-B/16 cda+fta+bridge+k2 | 49.064 | 66.471 | 75.216 | 190.751 | 46.691 | 33.483 |

## 3. Expert Merge / GAR-EM

**Goal:** build a diverse expert pool and fuse complementary retrieval evidence.

The merge contribution no longer fuses checkpoints from one ablation chain. It builds experts from different diversity axes:

| expert axis | instantiated by |
| --- | --- |
| Backbone diversity | OpenAI ViT-B/16, ViT-B/32, ViT-L/14 |
| Pretraining diversity | RemoteCLIP ViT-B/32 |
| Training-objective diversity | HardNeg ViT-B/16 k2 |
| Retrieval-aware fusion | GAR-EM score-level fixed/adaptive/prior-adaptive fusion |

The current global best is:

| method | fusion | R1 | R5 | R10 | RSum | mAP | mINP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenAI3 + RemoteCLIP + HardNeg | GAR-EM prior-adaptive, HardNeg h=0.05 | **50.513** | 66.862 | 74.972 | 192.347 | 48.599 | 36.039 |

Code and config entry points:

```text
tools/gar_em_score_fusion.py
scripts/merge_current_best_hardneg.sh
docs/merge/configs/currentbest_plus_hardneg.example.json
```

Recommended command:

```bash
bash scripts/merge_current_best_hardneg.sh
```

Detailed documentation:

```text
docs/merge/GAR_EM_PROTOCOL.md
docs/merge/DIRECTION_TWO_REMOTECLIP_SUMMARY.md
docs/merge/DIRECTION_THREE_HARDNEG_RESULTS.md
```

## Final Story

The final method can be described as:

```text
We first improve training efficiency with random k sampling, then strengthen
ground-aerial alignment with FTA+Bridge, and finally build a diverse expert
pool fused by GAR-EM to exploit complementary backbone, pretraining, and
training-objective signals.
```
