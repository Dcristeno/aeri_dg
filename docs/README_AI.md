# AI 交接说明

## 当前状态

当前 workspace 的最好结果是：

- 分支：`aeri-k2-ground-bridge-lite`
- 实验名：`aeri_k2_ground_bridge_lite_w2_seed2`
- SwanLab 项目：`CFAN`
- SwanLab 链接：https://swanlab.cn/@Dcristen/CFAN/runs/097gfonvv5s6skt4wy7wr
- 最好 R1：`47.95635986328125`
- 最好轮次：`54`
- 训练结束轮次：`60`

后续迭代请把这个实验和当前项目线当作最好 baseline。

## 当前实验线

当前 repo 对应 `AERI K=2 Ground Bridge Lite` 线，主要设置如下：

- 每轮按 ID 随机采样，`TRAIN_SAMPLES_PER_ID=2`
- 训练目标为 `base+id+bridge`
- base SDM 包含 aerial-text 和 ground-text 两项
- ID 分类 loss 作用在 aerial、ground、text 三类特征上
- bridge loss 使用 detached ground feature 作为 teacher，对齐 aerial feature

默认入口：

```bash
bash finetune.sh
```

显式复现实验时可参考：

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
USE_SWANLAB=1 \
RUN_NAME='aeri_k2_ground_bridge_lite' \
SWANLAB_EXPERIMENT='aeri_k2_ground_bridge_lite_w2_seed2' \
SEED=2 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

## 结果摘要

trainer 最终报告：

```text
best R1: 47.95635986328125 at epoch 54
```

epoch 60 的最终验证结果：

| task | R1 | R5 | R10 | RSum | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| t2i | 47.924 | 66.488 | 75.004 | 189.415 | 46.152 | 33.302 |

更完整的文字记录见 `docs/EXP_LOG.md`，实验横向比较表见 `docs/runs.csv`。
`runs.csv` 应优先记录能支持实验间比较的字段，例如 branch、rank、current_best、compare_to、main_change、best_epoch、best R1/R5/R10/RSum/mAP/mINP、delta_best_r1、关键 loss 权重和 SwanLab 链接；不要用 last/final epoch 指标作为主比较字段。

维护规则：任何实验记录、结果说明或代码修改说明都要写明所属分支，方便后续 merge 时判断来源。

## 后续实验建议

后续保守实验都应对比 `aeri_k2_ground_bridge_lite_w2_seed2`：

- 如果需要估计方差，可以先对同一条线做 seed sweep；
- 可以围绕当前默认值 `0.5` 微调 `BRIDGE_LOSS_WEIGHT`；
- 尝试采样策略变体时，建议保留 random `k=2` 作为强控制组。
