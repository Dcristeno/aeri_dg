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

需要模块化：

- baseline 训练主干：只负责最小训练、评估、checkpoint、日志。
- random `k=2` sampler：从数据构建中独立出来，可开关、可替换。
- `cda+fta` loss：从模型 forward 中拆出模块，可单独开启 CDA、FTA、CDA+FTA。
- bridge pair loss：从混杂 loss 逻辑中拆出，可单独关闭。
- best metric：支持显式选择 `R1` 或 `RSum`，后续默认追 R1。

## 建议消融顺序

1. 当前完整线：`cda+fta+bridge` + random `k=2`。
2. 去 bridge：`cda+fta` + random `k=2`。
3. 去 FTA：`cda+bridge` + random `k=2`。
4. 去 k=2：`cda+fta+bridge` + 原始采样。
5. 最小 baseline：只保留当前分支定义的基础 retrieval loss。

每次实验都写入：

- `docs/EXP_LOG.md`
- `docs/runs.csv`
- 如有错误，写入 `docs/ERROR_LOG.md`

