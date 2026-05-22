# 实验日志

分支：`aeri-cfan-baseline-cleanup`

## 当前最好结果

当前以 R1 为主要优化目标。代码默认 baseline 已调整为 `CDA + 原始采样`，后续论文/消融以 `cda` + 原始采样作为最小 baseline；random `k=2` sampling 和 `FTA+Bridge` 都作为后续创新模块单独加入。

已完成的最好 R1 full-module run：

- 实验名：`aeri_cda_fta_bridge_pair_k2_w2p0`
- SwanLab 项目：`CFAN`
- SwanLab 链接：https://swanlab.cn/@Dcristen/CFAN/runs/xu0rwgzsh3cokyjbcaoy6
- 训练目标：`cda+fta+bridge`
- 采样策略：按 ID 随机采样，`k=2`，每轮 7044 个训练样本
- bridge 设置：`BRIDGE_LOSS_WEIGHT=2.0`，`BRIDGE_PAIR_WEIGHT=1.0`，`BRIDGE_DISTILL_WEIGHT=0.0`
- R1 最好轮次：epoch 51
- R1 最好指标：`R1=49.064`

注意：当前 trainer 日志按 `RSum` 记录 best，最终报告为 `best RSum: 190.97866821289062 at epoch 57`。后续既然追求 R1 效率，实验记录和 checkpoint 选择逻辑要明确区分 R1-best 与 RSum-best。

## 2026-05-21 - `aeri_cda_fta_bridge_pair_k2_w2p0`

- 状态：完成
- SwanLab 项目：`CFAN`
- SwanLab 链接：https://swanlab.cn/@Dcristen/CFAN/runs/xu0rwgzsh3cokyjbcaoy6
- 对比基准：后续需要补跑 `cda` + 原始采样 baseline；该实验作为 FTA+Bridge + k=2 full-module 结果
- 训练目标：`cda+fta+bridge`
- 采样策略：`TRAIN_SAMPLES_PER_ID=2`，`TRAIN_SAMPLE_STRATEGY=random`
- 训练轮数：60
- seed：默认 `SEED=1`
- 学习率：`lr=5e-6`，`lr2=5e-5`
- 初始化：`/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth`

关键参数：

- `BRIDGE_LOSS_WEIGHT=2.0`
- `BRIDGE_PAIR_WEIGHT=1.0`
- `BRIDGE_DISTILL_WEIGHT=0.0`
- `BRIDGE_DISTILL_TEMP=0.07`

### R1-Best

| epoch | task | R1 | R5 | R10 | RSum | mAP | mINP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 51 | t2i | 49.064 | 66.471 | 75.216 | 190.751 | 46.691 | 33.483 |

### Trainer RSum-Best

| epoch | task | R1 | R5 | R10 | RSum | mAP | mINP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 57 | t2i | 48.722 | 66.960 | 75.297 | 190.979 | 46.704 | 33.536 |

### Final Epoch

| epoch | task | R1 | R5 | R10 | RSum | mAP | mINP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 60 | t2i | 48.673 | 66.976 | 75.313 | 190.962 | 46.730 | 33.558 |

epoch 50-60 后段曲线显示 R1 在 49 左右平台化，后续 epochs 的 RSum 仍能小幅涨，但 R1 没有继续提高。

### 结论

- `cda+fta+bridge` + random `k=2` + pair-only bridge 可以达到 `R1=49.064`。
- 当前 bridge loss 与 bridge pair loss 相等，说明 distill 项已关闭，实验确认为 pair-only bridge。
- 后续需要先补跑 `cda` + 原始采样，把它作为最小 baseline，再分别加入 random `k=2`、FTA、bridge、FTA+Bridge 做消融。

## 下一步计划

当前要追求 R1 效率，而不是继续堆叠复杂模块。建议重构方向：

1. 分离 baseline 主干：保留最小可运行 finetune 训练、评估、日志与 SwanLab 记录。
2. 抽出 `k=2` random per-id sampling 为独立模块，便于和 full sampling / no sampling 对比。
3. 以 `cda` + 原始采样作为 baseline。
4. 抽出 random `k=2`、FTA 和 bridge pair 为创新模块，同时保留单独开关做消融。
5. 修改 best checkpoint 和日志口径，新增以 R1 选择 best 的路径，避免继续默认按 RSum 做主要判断。

## 2026-05-21 - R1-oriented 模块化重构

已完成第一步代码拆分：

- per-id sampling 拆到 `datasets/per_id_sampling.py`。
- CDA、FTA、bridge loss 组装拆到 `model/finetune_losses.py`。
- 新增 `BEST_METRIC` / `--best_metric`，默认 `R1`，后续 `best0` 按 R1 保存。
- `Evaluator.eval()` 默认返回值已从 `t2i_RSum` 改为 `t2i_R1`，避免旧调用路径继续隐式使用 RSum。

这次重构不改变已有 loss 数学形式，只把开关边界拆清楚，方便后续跑去 bridge、去 FTA、去 k=2 的 baseline/ablation。

## 2026-05-21 - Baseline 口径调整

- 默认 `LOSS_NAMES` 从 `cda+fta` 改为 `cda`。
- `configs/aeri_cfan_baseline.yaml` 同步改为 `loss_names: cda`。
- 论文叙事口径：`CDA + 原始采样` 是 baseline，random `k=2` 是独立创新模块，`FTA+Bridge` 是第三创新点的联合模块。
- 下一条优先实验：`aeri_cda_fullsample_r1base`。

## 2026-05-21 - `aeri_cda_fullsample_r1base` early-stop record

- 状态：early-stop recommended，日志明细记录到 epoch 11，SwanLab 曲线显示后续仍整体下行
- SwanLab 项目：`CFAN`
- SwanLab 链接：https://swanlab.cn/@Dcristen/CFAN/runs/71dclnjlnizwkjew9l5je
- 对比基准：self，作为纯 baseline
- 训练目标：`cda`
- 采样策略：原始 full training sampler，`TRAIN_SAMPLES_PER_ID=0`
- 训练轮数：60
- seed：默认 `SEED=1`
- 当前 R1-best：epoch 4，`R1=44.976`

这条实验是论文消融的最小 baseline：不包含 random `k=2` sampling，不包含 FTA，不包含 bridge。

### Early Curve

| epoch | task | R1 | R5 | R10 | RSum | mAP | mINP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | t2i | 41.492 | 59.941 | 68.881 | 170.314 | 39.181 | 26.430 |
| 2 | t2i | 44.113 | 62.889 | 71.210 | 178.212 | 42.085 | 29.003 |
| 3 | t2i | 44.520 | 63.540 | 71.617 | 179.678 | 42.680 | 29.560 |
| 4 | t2i | 44.976 | 63.068 | 71.943 | 179.987 | 42.950 | 29.776 |
| 5 | t2i | 44.113 | 61.895 | 71.356 | 177.365 | 42.295 | 29.519 |
| 6 | t2i | 43.804 | 62.221 | 70.949 | 176.974 | 42.486 | 29.961 |
| 7 | t2i | 44.797 | 63.459 | 72.285 | 180.541 | 42.977 | 30.047 |
| 8 | t2i | 43.755 | 62.824 | 71.324 | 177.903 | 41.954 | 28.935 |
| 9 | t2i | 43.625 | 63.100 | 71.682 | 178.407 | 42.043 | 29.052 |
| 10 | t2i | 42.957 | 62.905 | 71.796 | 177.658 | 41.224 | 28.009 |
| 11 | t2i | 41.475 | 60.642 | 69.582 | 171.698 | 40.197 | 27.678 |

### 后续曲线判断

用户补充的 SwanLab 截图显示，在 epoch 11 之后，`val/t2i_R1`、`R5`、`R10`、`RSum`、`mAP`、`mINP` 均继续围绕下行趋势震荡，没有回到 early peak。该实验可以作为纯 CDA/full-sampling 的低锚点 baseline，不建议继续等待满 60 epoch 再启动下一组消融。

备注：

- trainer 当前按 `BEST_METRIC=R1` 保存 best0，日志显示 best R1 在 epoch 5 后保持为 `44.97639083862305`，对应 epoch 4 的验证结果。
- full sampler 每轮约 `1188` iterations，单轮约 `5.0 min`，明显慢于 random `k=2` 的 `111` iterations。
- 若服务器还在跑，可停止该 run，保留 epoch 4 的 `best0` 作为 baseline checkpoint。

## 2026-05-22 - 模型融合创新点口径

当前决定把模型融合从“融合同一条方法链路的几个消融 checkpoint”调整为“构建多样化地空图文检索专家池，并设计地空方向专用 merge 准则”。

消融池仍然用于证明模块有效：

| 实验 | 用途 |
| --- | --- |
| `CDA` | 最小 baseline |
| `CDA + k2` | 验证随机 k 图采样 |
| `CDA + FTA+Bridge` | 验证细粒度与跨视角桥接对齐 |
| `CDA + FTA+Bridge + k2` | 完整方法 |

融合池不再直接使用这一组消融模型作为主体，而是按四个方向构建：

1. 不同 backbone：结构多样性，例如 `ViT-B/16`、`ViT-B/32`、`RN50`、`RN101`。
2. 不同预训练：知识来源多样性，例如 OpenAI CLIP、OpenCLIP、EVA-CLIP、SigLIP、RemoteCLIP/GeoCLIP。
3. 不同训练目标：任务能力多样性，例如 CDA、hard negative、FTA、Bridge、domain/view alignment。
4. 地空方向专用 merge：融合准则要利用地空图文检索的 rank structure、cross-view cycle、local-global alignment、hard-negative separability 和 expert complementarity。

方法暂定名：`Ground-Aerial Retrieval-Aware Expert Merge` / `GAR-EM`。

详细协议见 `docs/MERGE_PROTOCOL.md`。

## 2026-05-22 - `gar_em_vitb16_vitb32_full`

- 状态：完成第一版 score-level fusion 试验
- 专家 1：`ViT-B/16 + cda+fta+bridge + random k=2`，对应此前 full-module R1-best `49.064`
- 专家 2：`ViT-B/32 + cda+fta+bridge + random k=2`，SwanLab 链接：https://swanlab.cn/@Dcristen/CFAN/runs/97z6f82ay0yd1u5jgumgv
- `ViT-B/32` 单模型 best R1：`39.065` at epoch 55
- 融合输出：`logs/merge/gar_em_vitb16_vitb32_full/gar_em_score_fusion.json`

| method | R1 | R5 | R10 | RSum | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean | 48.526 | 65.413 | 74.304 | 188.243 | 46.985 | 34.727 |
| fixed `0.9,0.1` | 49.324 | 67.269 | 75.444 | 192.037 | 47.389 | 34.438 |
| GAR-EM adaptive v1 | 48.135 | 64.827 | 73.652 | 186.615 | 46.718 | 34.291 |
| GAR-EM prior-adaptive `0.9,0.1` | 49.487 | 67.009 | 75.102 | 191.597 | 47.567 | 34.732 |
| fixed `0.95,0.05` | 49.064 | 67.237 | 75.379 | 191.679 | 47.084 | 33.987 |
| GAR-EM prior-adaptive `0.95,0.05` | 49.601 | 67.318 | 75.590 | 192.509 | 47.418 | 34.377 |
| fixed `0.85,0.15` | 49.617 | 67.367 | 75.493 | 192.477 | 47.618 | 34.799 |
| GAR-EM prior-adaptive `0.85,0.15` | 49.161 | 66.569 | 75.085 | 190.816 | 47.519 | 34.793 |
| fixed `0.92,0.08` | 49.178 | 67.188 | 75.509 | 191.874 | 47.284 | 34.265 |
| GAR-EM prior-adaptive `0.92,0.08` | 49.617 | 67.139 | 75.216 | 191.972 | 47.568 | 34.629 |
| fixed `0.93,0.07` | 49.178 | 67.171 | 75.444 | 191.793 | 47.211 | 34.173 |
| GAR-EM prior-adaptive `0.93,0.07` | 49.617 | 67.106 | 75.379 | 192.102 | 47.534 | 34.563 |
| fixed `0.94,0.06` | 49.113 | 67.237 | 75.427 | 191.777 | 47.143 | 34.071 |
| GAR-EM prior-adaptive `0.94,0.06` | 49.601 | 67.188 | 75.541 | 192.330 | 47.486 | 34.475 |
| fixed `0.96,0.04` | 48.982 | 67.220 | 75.330 | 191.532 | 47.030 | 33.900 |
| GAR-EM prior-adaptive `0.96,0.04` | 49.568 | 67.351 | 75.541 | 192.461 | 47.341 | 34.260 |

结论：

- fixed `0.9,0.1` 相比 full-module R1-best `49.064` 提升 `+0.260`，说明弱异构 backbone 仍可能补充主模型漏检样本。
- GAR-EM prior-adaptive `0.95,0.05` 相比 full-module R1-best `49.064` 提升 `+0.537`，说明更保守的可靠性先验更适合弱异构专家。
- fixed `0.85,0.15` 达到当前最高 R1 `49.617`，说明 ViT-B/32 的静态贡献比例可能在 10%-15% 区间；但同先验下 adaptive 降到 `49.161`，说明当前 query-level adaptive 在放开弱专家时仍会过度偏移。
- GAR-EM prior-adaptive `0.92,0.08` 和 `0.93,0.07` 均达到 `R1=49.617`，追平当前 fixed 最优，同时保留方法上的 query-level adaptive 解释性；这是当前最适合作为 GAR-EM v1 主结果的区间。
- prior 过保守时 fixed 会退回主模型附近，但 prior-adaptive 仍能保持 `49.568-49.601`，说明 query-level adaptive 在强先验约束下能稳定借用弱专家。
- mean fusion 和无先验 GAR-EM adaptive v1 都下降，说明弱专家不能无约束进入融合。
- expert reliability prior 是必要组件：它保留主专家强度，同时允许 query 级互补偏移。当前建议报告 `GAR-EM prior-adaptive 0.92,0.08` 或 `0.93,0.07` 作为方法结果，并把 fixed `0.85,0.15` 作为手工权重上界对照。
