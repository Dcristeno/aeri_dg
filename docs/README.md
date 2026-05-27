# AERI-PEDES Documentation

This directory is organized around the current research state, not around the order in which experiments happened.

## Start Here

- `CURRENT_STATUS.md`: current best result, accepted merge story, and immediate next actions.
- `merge/GAR_EM_PROTOCOL.md`: method-level protocol for Ground-Aerial Retrieval-Aware Expert Merge.
- `merge/DIRECTION_THREE_HARDNEG_RESULTS.md`: current global best fusion result.
- `../scripts/README.md`: runnable entry points for the three paper contributions.

## Current Best

| method | R1 | R5 | R10 | RSum | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenAI3 + RemoteCLIP + HardNeg, GAR-EM prior-adaptive | **50.513** | 66.862 | 74.972 | 192.347 | 48.599 | 36.039 |

This supersedes the previous direction-two best:

```text
OpenAI3 + RemoteCLIP fixed r=0.08
R1 = 49.959
```

## Directory Map

```text
docs/
  README.md
  CURRENT_STATUS.md
  merge/
    README.md
    GAR_EM_PROTOCOL.md
    OPENAI3_META_EXPERT.md
    DIRECTION_TWO_REMOTECLIP_SUMMARY.md
    REMOTECLIP_RESULTS_20260525.md
    REMOTECLIP_PLAN_AND_RESULTS.md
    HARD_NEGATIVE_EXPERT.md
    DIRECTION_THREE_HARDNEG_RESULTS.md
    configs/
      merge_pool.example.json
      currentbest_plus_hardneg.example.json
    internal_trials/
      GEORSCLIP.md
      OPENCLIP_LAION.md
      RS_M_CLIP.md
  experiments/
    runs.csv
  archive/
    legacy_mojibake/
      EXP_LOG.md
      ERROR_LOG.md
      README_AI.md
```

## Document Roles

- Current conclusions belong in `CURRENT_STATUS.md` and the relevant `merge/DIRECTION_*` summary.
- Full experimental sweep tables belong in `merge/*RESULTS*.md`.
- Run commands and operational templates belong beside the method they support.
- Old mojibake handoff logs are preserved under `archive/legacy_mojibake/` for traceability, but they are not the source of truth.
