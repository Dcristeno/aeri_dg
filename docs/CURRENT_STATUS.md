# Current Status

## Active Branch

```text
aeri-cfan-baseline-cleanup
```

## Current Global Best

The current best result is the five-expert merge that adds the hard-negative training-objective expert to the previous direction-two pool.

| method | fusion | R1 | R5 | R10 | RSum | mAP | mINP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenAI3 + RemoteCLIP + HardNeg | GAR-EM prior-adaptive, HardNeg h=0.05 | **50.513** | 66.862 | 74.972 | 192.347 | 48.599 | 36.039 |

Reference:

```text
merge/DIRECTION_THREE_HARDNEG_RESULTS.md
```

## Merge Story

The merge direction is now structured as a real expert pool rather than fusing checkpoints from one ablation chain.

| direction | role | status |
| --- | --- | --- |
| Different backbone | OpenAI ViT-B/16, ViT-B/32, ViT-L/14 form OpenAI3 | Done |
| Different pretraining | RemoteCLIP ViT-B/32 adds remote-sensing prior | Done |
| Different training objective | HardNeg ViT-B/16 k2 adds similar-candidate separation | Done |
| Retrieval-aware merge | GAR-EM score-level fixed/prior-adaptive fusion | Done for current paper scope |

## Key Results

| result | R1 | note |
| --- | ---: | --- |
| Full single expert, ViT-B/16 cda+fta+bridge+k2 | 49.064 | Original strong single model |
| OpenAI3 | 49.813 | Different-backbone pool |
| OpenAI3 + RemoteCLIP | 49.959 | Best direction-two result |
| OpenAI3 + RemoteCLIP + HardNeg | **50.513** | Current global best |

## Paper Framing

Use the following concise framing:

```text
We build a diverse ground-aerial retrieval expert pool from complementary sources:
backbone diversity, remote-sensing pretraining diversity, and training-objective diversity.
GAR-EM then fuses experts with retrieval-aware query-level reliability signals.
```

## Do Not Treat As Main Results

- GeoRSCLIP: internal diagnostic; hurts R1 after RemoteCLIP though it may improve mAP/mINP.
- RS-M-CLIP: zero-shot cache is extremely weak under the current protocol.
- OpenCLIP LAION: generic pretraining control; weak zero-shot and not remote-sensing-specific.
- Legacy mojibake docs in `archive/legacy_mojibake/`: retained only for traceability.
