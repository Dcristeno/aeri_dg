# Merge Documentation

This folder contains the cleaned merge storyline and supporting run notes.

## Source Of Truth

| file | purpose |
| --- | --- |
| `GAR_EM_PROTOCOL.md` | Method protocol and conceptual criteria for GAR-EM |
| `DIRECTION_THREE_HARDNEG_RESULTS.md` | Current global best result, including HardNeg fusion sweep |
| `DIRECTION_TWO_REMOTECLIP_SUMMARY.md` | Formal different-pretraining result |
| `REMOTECLIP_RESULTS_20260525.md` | Detailed RemoteCLIP sweep table |
| `OPENAI3_META_EXPERT.md` | How to cache/use the OpenAI3 meta expert |
| `HARD_NEGATIVE_EXPERT.md` | How to train and interpret the hard-negative expert |

The reproducible five-expert pool used for the current best result is in:

```text
configs/currentbest_plus_hardneg.example.json
```

## Current Best

```text
OpenAI3 + RemoteCLIP + HardNeg
GAR-EM prior-adaptive, HardNeg h=0.05
R1 = 50.513
```

## Accepted Expert Pool

| expert | diversity axis | role |
| --- | --- | --- |
| OpenAI ViT-B/16 full | backbone | strongest base expert |
| OpenAI ViT-B/32 full | backbone | complementary global/patch bias |
| OpenAI ViT-L/14 full | backbone/capacity | improves ranking quality |
| RemoteCLIP ViT-B/32 full | pretraining | remote-sensing prior |
| HardNeg ViT-B/16 k2 | training objective | hard-negative separability |

## Internal Trials

Internal trials are preserved in `internal_trials/`. They are useful for appendix or discussion, but they are not part of the formal main expert pool.
