# Experiment Scripts

These scripts are the stable command entry points for the three paper contributions.

The contribution-level narrative is recorded in:

```text
docs/CONTRIBUTIONS.md
```

Run them from the repository root after activating the training environment:

```bash
conda activate irra
bash scripts/train_k_random.sh
```

## Paper Contributions

| contribution | script | main switches |
| --- | --- | --- |
| Random k sampling | `train_k_random.sh` | `LOSS_NAMES=cda`, `TRAIN_SAMPLES_PER_ID=2` |
| FTA + Bridge | `train_fta_bridge.sh` | `LOSS_NAMES=cda+fta+bridge`, bridge pair-only settings |
| Merge / GAR-EM | `merge_current_best_hardneg.sh` | five-expert `OpenAI3 + RemoteCLIP + HardNeg` fusion |

## Shared Environment Variables

All training scripts accept the existing `finetune.sh` environment variables. Common overrides:

```bash
DATA_ROOT=/home/wuyong/datasets
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth
CUDA_VISIBLE_DEVICES=0
USE_SWANLAB=1
```

`merge_current_best_hardneg.sh` assumes the checkpoint paths recorded in:

```text
docs/merge/configs/currentbest_plus_hardneg.example.json
```

If a server stores historical logs in a different location, update that JSON first.
