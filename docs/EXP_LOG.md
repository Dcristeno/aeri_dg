# 实验日志

## 当前最好结果

截至 2026-05-19，当前项目最好结果来自下面这条实验线：

- 实验名：`aeri_k2_ground_bridge_lite_w2_seed2`
- 分支：`aeri-k2-ground-bridge-lite`
- 项目：`CFAN`
- 当前代码/配置线：`AERI K=2 Ground Bridge Lite`
- 训练目标：`base+id+bridge`
- 采样策略：按 ID 随机采样，`k=2`，每轮重建后共有 7044 个训练样本
- 训练轮数：60
- 最好验证 R1：`47.95635986328125`
- 最好轮次：`54`
- SwanLab 链接：https://swanlab.cn/@Dcristen/CFAN/runs/097gfonvv5s6skt4wy7wr

后续实验默认应把这条线当作当前 baseline/best result 来对比。

## 2026-05-19 - `aeri_k2_ground_bridge_lite_w2_seed2`

最后阶段 epoch 55-60 的验证结果如下：

| Epoch | R1 | R5 | R10 | RSum | mAP | mINP | Base LR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 55 | 47.696 | 66.406 | 74.858 | 188.959 | 46.025 | 33.270 | 1.45e-07 |
| 56 | 47.891 | 66.455 | 74.825 | 189.171 | 46.126 | 33.289 | 1.01e-07 |
| 57 | 47.794 | 66.553 | 74.890 | 189.236 | 46.117 | 33.296 | 6.50e-08 |
| 58 | 47.826 | 66.488 | 74.923 | 189.236 | 46.120 | 33.272 | 3.66e-08 |
| 59 | 47.924 | 66.488 | 74.988 | 189.399 | 46.149 | 33.293 | 1.63e-08 |
| 60 | 47.924 | 66.488 | 75.004 | 189.415 | 46.152 | 33.302 | 4.08e-09 |

训练结束时日志显示：

- Best R1 保持为 `47.95635986328125`。
- trainer 报告的最好轮次为 `54`。
- epoch 60 没有超过最好 R1，但在上面这段收尾日志里取得了最高 RSum：`189.415`。
- SwanLab 实验正常完成。

epoch 60 的代表性训练 loss：

| Loss | Value |
| --- | ---: |
| loss | 14.9112 |
| base_loss | 10.2489 |
| base_aerial_text | 5.8524 |
| base_ground_text | 4.3965 |
| id_loss | 4.0617 |
| bridge_loss | 0.6006 |

备注：

- 最后阶段单轮耗时约 `0.47 min`。
- 训练速度约 `135 samples/s`。
- 后期学习率已经很小，指标基本可以视为收敛/平台期。

## 2026-05-19 - 待跑：`base+id+bridge+cfa`

- 分支：`aeri-k2-ground-bridge-lite`
- 目标：从 `CFAN-clean` 抽取 CDA 和 FTA 的精华，合成为当前项目里的轻量 `cfa` 模块。
- 对比基准：`aeri_k2_ground_bridge_lite_w2_seed2`
- 建议实验名：`aeri_k2_ground_bridge_lite_cfa_w1_seed2`
- 建议配置：`LOSS_NAMES=base+id+bridge+cfa`

模块含义：

- CDA-style：根据 text-aerial 和 text-ground 的相对相似度，自适应选择 direct SDM 或 bridge SDM。
- FTA-style：用少量 query 从 image/text token 中提取细粒度 slot，再用 fuzzy membership 加权得到跨模态相似度矩阵。

这个实验尚未在云端完成，结果跑完后需要追加到 `docs/runs.csv`，并用 best epoch 指标比较。
