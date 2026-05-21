# AI 交接说明

分支：`aeri-cfan-baseline-cleanup`

## 当前状态

已完成实验：

- `aeri_cda_fta_bridge_pair_k2_w2p0`
- SwanLab：https://swanlab.cn/@Dcristen/CFAN/runs/xu0rwgzsh3cokyjbcaoy6
- 配置：`cda+fta+bridge`，random `k=2`，pair-only bridge
- R1-best：epoch 51，`R1=49.064`
- trainer RSum-best：epoch 57，`RSum=190.97866821289062`

后续优化目标切换为 R1 效率。记录和代码都要优先支持 R1-best 选择与 R1-oriented ablation。

## 当前 Baseline

当前作为本分支重新记录后的 baseline：

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
USE_SWANLAB=1 \
RUN_NAME='aeri_cda_fta_bridge_pair_k2_w2p0' \
SWANLAB_EXPERIMENT='aeri_cda_fta_bridge_pair_k2_w2p0' \
LOSS_NAMES='cda+fta+bridge' \
TRAIN_SAMPLES_PER_ID=2 \
TRAIN_SAMPLE_STRATEGY='random' \
BRIDGE_LOSS_WEIGHT=2.0 \
BRIDGE_PAIR_WEIGHT=1.0 \
BRIDGE_DISTILL_WEIGHT=0.0 \
BRIDGE_DISTILL_TEMP=0.07 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

## 重构目标

目标是把当前实验拆成清晰可控的模块，用更低成本做 R1-oriented ablation。

已完成的第一步拆分：

- `datasets/per_id_sampling.py`：独立承载 random `k=2` 和其他 per-id sampling 策略。
- `model/finetune_losses.py`：独立承载 CDA、FTA、bridge pair/distill 的 loss 组装。
- `BEST_METRIC` / `--best_metric`：默认 `R1`，用于保存 `best0`。

后续继续保持：

- baseline 训练主干只负责最小训练、评估、checkpoint、日志。
- sampling、loss、bridge 都通过参数开关做消融，不再散落在主训练流程中。

## 建议消融顺序

1. 当前完整线：`cda+fta+bridge` + random `k=2`。
2. 去 bridge：`cda+fta` + random `k=2`。
3. 去 FTA：`cda+bridge` + random `k=2`。
4. 去 k=2：`cda+fta+bridge` + 原始采样。
5. 最小 baseline：只保留当前分支定义的基础 retrieval loss。

建议命令：

```bash
# 去 bridge：验证 cda+fta + random k=2 的 R1
USE_SWANLAB=1 \
RUN_NAME='aeri_cda_fta_k2_r1base' \
SWANLAB_EXPERIMENT='aeri_cda_fta_k2_r1base' \
LOSS_NAMES='cda+fta' \
TRAIN_SAMPLES_PER_ID=2 \
TRAIN_SAMPLE_STRATEGY='random' \
BEST_METRIC=R1 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh

# 去 k=2：验证完整 loss 在原始采样下的 R1
USE_SWANLAB=1 \
RUN_NAME='aeri_cda_fta_bridge_pair_fullsample_r1base' \
SWANLAB_EXPERIMENT='aeri_cda_fta_bridge_pair_fullsample_r1base' \
LOSS_NAMES='cda+fta+bridge' \
TRAIN_SAMPLES_PER_ID=0 \
BRIDGE_LOSS_WEIGHT=2.0 \
BRIDGE_PAIR_WEIGHT=1.0 \
BRIDGE_DISTILL_WEIGHT=0.0 \
BEST_METRIC=R1 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

每次实验都写入：

- `docs/EXP_LOG.md`
- `docs/runs.csv`
- 如有错误，写入 `docs/ERROR_LOG.md`
