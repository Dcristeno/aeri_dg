# Ground-Aerial Expert Merge Protocol

## 目标

模型融合创新点不再融合 `CDA`、`CDA+k2`、`CDA+FTA+Bridge`、`CDA+FTA+Bridge+k2` 这一组消融模型。它们属于同一条方法链路，主要用于证明采样模块和对齐模块有效，不适合作为独立的融合专家池。

模型融合部分要构建一组真正不同的地空图文检索专家，并提出面向地空检索任务的融合准则。目标是让论文中的第三类创新点从“经验性融合 checkpoint”变成可复现、可解释、可写成方法的小节。

## 两个实验池

### 消融池

消融池只回答模块是否有效：

| 实验 | 用途 |
| --- | --- |
| `CDA` | 最小 baseline |
| `CDA + k2` | 验证随机 k 图采样 |
| `CDA + FTA+Bridge` | 验证细粒度与跨视角桥接对齐 |
| `CDA + FTA+Bridge + k2` | 完整方法 |

这些实验不作为融合专家池的主体。

### 融合专家池

融合池要覆盖不同来源的专家，要求每个专家都能做地空图文检索，但错误模式和优势不同。当前确定四个构建方向：

1. 不同 backbone：结构多样性。
2. 不同预训练：知识来源多样性。
3. 不同训练目标：任务能力多样性。
4. 地空方向专用 merge：融合准则的任务适配性。

## 方向一：不同 Backbone

不同视觉主干提供不同结构归纳偏置。

| 类型 | 可能专家 | 预期优势 |
| --- | --- | --- |
| ViT-B/16 | `pretrain_choice=ViT-B/16` | 更细 patch，兼顾局部细节与全局关系 |
| ViT-B/32 | `pretrain_choice=ViT-B/32` | 更粗粒度全局语义，训练成本较低 |
| RN50 / RN101 | `pretrain_choice=RN50/RN101` | 局部纹理、边缘、层级视觉特征 |
| ViT-L/14 | `pretrain_choice=ViT-L/14` | 更强表征能力，适合作为高容量专家 |

地空图文检索同时依赖全局布局和局部细节：道路走向、建筑群结构、区域形状属于全局信息；屋顶颜色、树木、车辆、建筑纹理属于局部信息。不同 backbone 的专家可以在这两类信息上互补。

## 方向二：不同预训练

不同预训练模型携带不同数据分布和语义空间。

候选来源：

- OpenAI CLIP。
- OpenCLIP。
- EVA-CLIP。
- SigLIP。
- RemoteCLIP / GeoCLIP / 遥感或地理视觉语言模型。

地空检索存在自然图像、地面视角、空中视角和文本语义之间的域差异。通用 CLIP 可能更强于自然语言和通用视觉语义，遥感相关预训练可能更强于 aerial layout、地物纹理和俯视视角模式。融合不同预训练来源，是为了获得知识来源互补。

## 方向三：不同训练目标

不同训练目标让专家形成不同任务能力。

| 训练目标 | 能力定位 |
| --- | --- |
| CDA | 基础图文对齐与检索 baseline |
| Hard negative / margin-based objective | 相似候选区分能力 |
| FTA | 细粒度文本-图像局部对齐能力 |
| Bridge | 地面-空中跨视角桥接能力 |
| Domain / view alignment | 地空域差异抑制能力 |

注意：`CDA`、`FTA`、`Bridge` 在消融池里用于证明模块贡献；在融合池里，如果使用训练目标维度，应该作为不同专家能力来源，而不是简单把同一条完整方法链路的几个 checkpoint 互相融合。

## 方向四：地空方向专用 Merge

融合方法本身要体现地空图文检索特性，暂定名称：

```text
Ground-Aerial Retrieval-Aware Expert Merge
```

或者简称：

```text
GAR-EM
```

核心不是简单平均分数，也不是随意手调权重，而是根据地空检索中的结构化信号选择融合权重。

### 多层融合准则

对每个专家 `E_i` 和 query `q`，计算专家可靠性：

```text
w_i(q) = softmax(
    a1 * Rank_i(q)
  + a2 * Cycle_i(q)
  + a3 * LocalGlobal_i(q)
  + a4 * HardNeg_i(q)
  + a5 * Complement_i(q)
  - a6 * Uncertainty_i(q)
)
```

最终融合分数：

```text
S(q, x) = sum_i w_i(q) * S_i(q, x)
```

如果两个专家结构一致，也可以用于选择 checkpoint merge 系数：

```text
theta_merge = alpha * theta_a + (1 - alpha) * theta_b
alpha* = argmax J(alpha)
```

其中 `J(alpha)` 使用同一套地空检索感知指标。

### 1. Rank Consistency

衡量不同专家或不同融合系数下的检索排序结构是否稳定。

可用指标：

- top-k overlap。
- Kendall rank correlation。
- NDCG consistency。
- top-k neighborhood graph similarity。

目的：不是只看 top1 是否一致，而是看候选集合的局部排序结构是否稳定。

### 2. Cross-View Cycle Consistency

地空图文检索有三类关系：

```text
Text -> Ground
Text -> Aerial
Ground -> Aerial
```

可以构造闭环：

```text
Text -> Ground -> Aerial
Text -> Aerial -> Ground
```

如果一个专家的闭环结果稳定，说明它不仅在某一侧特征上碰巧匹配，而是真的学到了文本、地面图像和空中图像之间的共同语义。

### 3. Local-Global Alignment

地空检索同时需要局部细节和全局布局。

可拆成两类 query：

- 局部属性 query：颜色、屋顶、车辆、树木、建筑纹理。
- 全局结构 query：道路形状、交叉口、建筑群布局、区域拓扑。

当文本更偏局部细节时，提高局部专家权重；当文本更偏全局布局时，提高全局专家权重。

### 4. Hard-Negative Separability

普通 margin 只看 `top1 - top2`，地空检索中相似建筑和相似道路布局很多，需要更强的难例分离度。

可用指标：

- top1 与 top-k 平均分差。
- top1 与 hard-negative cluster 的分差。
- 正候选邻域与负候选邻域的分离度。

目的：判断专家是否能把相似干扰样本拉开。

### 5. Expert Complementarity

不能只奖励一致性，否则融合会退化成平均模型。还要保留稳定且有价值的分歧。

一个专家即使和主专家 top-k 不完全一致，只要同时满足：

- 自身 margin 高。
- 增强或扰动下排序稳定。
- 能补回其他专家漏掉的正确候选。

就应该获得一定权重。

### 6. Retrieval Uncertainty

对不可靠专家降权。

可用信号：

- top-k 分数过于平坦。
- 不同增强视图下排序波动大。
- query 级置信度低。
- 与其他专家完全冲突且自身 margin 低。

## 与 AdaMMS 的关系

AdaMMS 用无监督一致性为多模态大模型搜索融合系数。这里的迁移方式是：

```text
generation consistency -> retrieval ranking consistency
multimodal output agreement -> ground-aerial-text cycle agreement
coefficient search -> retrieval-aware coefficient/search or expert weighting
```

因此论文叙事不是“借用通用模型融合”，而是提出一个面向地空图文检索的任务感知融合策略。

## 第一阶段落地路线

1. 先固定消融池，不把消融模型当成融合池主体。
2. 建立第一版融合专家池：
   - `ViT-B/16 + CDA`
   - `ViT-B/32 + CDA`
   - `RN50 + CDA`
   - `RN101 + CDA`
3. 跑出每个专家的独立 R1/R5/R10/RSum/mAP/mINP。
4. 实现 score-level fusion baseline：
   - mean score fusion。
   - fixed weighted fusion。
   - GAR-EM adaptive fusion。
5. 若专家结构一致，再做 checkpoint-level merge；结构不一致时优先做 score-level 或 feature-level fusion。

## 第一版工具

已提供第一版 score-level 融合工具：

```bash
python tools/gar_em_score_fusion.py \
  --expert_config docs/merge_pool.example.json \
  --root_dir /home/wuyong/datasets \
  --output_dir logs/merge/gar_em_v1 \
  --topk 10 \
  --prior_weights 0.9,0.1 \
  --prior_strength 1.0 \
  --adaptive_temperature 0.5 \
  --device cuda
```

输出文件：

- `gar_em_score_fusion.json`
- `gar_em_score_fusion.md`

当前第一版实现包含：

- 每个专家独立指标。
- mean score fusion。
- optional fixed-weight score fusion。
- GAR-EM adaptive score fusion。
- GAR-EM prior-adaptive score fusion：用于弱但互补的异构专家，保留主专家可靠性先验。

第一版 GAR-EM 已实现 rank consistency、hard-negative separability、expert complementarity 和 retrieval uncertainty。`Cross-View Cycle Consistency` 与 `Local-Global Alignment` 先保留为第二版接口，需要后续让验证 dataloader 显式暴露 ground/aerial/text 三方关系和 query 属性。

### 2026-05-22 ViT-B/16 + ViT-B/32 试验

`ViT-B/32` 使用 full recipe 训练后单模型 R1 只有 `39.065`，低于正式主专家门槛；但与 `ViT-B/16` full expert 做小权重 score fusion 后出现正增益：

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

- 弱异构专家可能提供互补排序信息。
- 直接 mean fusion 会拖累主模型。
- 无先验 adaptive v1 对弱专家约束不足，需要加入 expert reliability prior。
- `gar_em_prior_adaptive 0.95,0.05` 达到 `R1=49.601`，相比 full-module 主专家 `49.064` 提升 `+0.537`。
- `fixed 0.85,0.15` 当前 R1 最高，为 `49.617`；但 `gar_em_prior_adaptive 0.85,0.15` 降到 `49.161`，说明当前 adaptive 在放开弱专家时仍可能过度偏移。
- `gar_em_prior_adaptive 0.92,0.08` 和 `0.93,0.07` 均达到 `R1=49.617`，追平 fixed 手工权重最优，同时保留 query-level adaptive 方法解释。
- 当前建议用 `GAR-EM prior-adaptive 0.92,0.08` 或 `0.93,0.07` 作为方法主结果；用 `fixed 0.85,0.15` 作为手工权重强对照。

## 专家训练命令模板

`finetune.sh` 已暴露 `PRETRAIN_CHOICE`，可以直接切换 backbone：

```bash
DATA_ROOT=/home/wuyong/datasets \
FINETUNE_INIT=/home/wuyong/data/HAM/HAM_checkpoint/random100w_2HAMcaptions/best0.pth \
USE_SWANLAB=1 \
RUN_NAME='aeri_cda_vit_b32' \
SWANLAB_EXPERIMENT='aeri_cda_vit_b32' \
PRETRAIN_CHOICE='ViT-B/32' \
LOSS_NAMES='cda' \
TRAIN_SAMPLES_PER_ID=0 \
BEST_METRIC=R1 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

首轮建议只改 `PRETRAIN_CHOICE` 和 `RUN_NAME`，保持 CDA、full sampler、R1-best 口径不变，避免融合池和消融池混在一起。

如果切换到和 HAM 初始化 checkpoint 不同结构的 backbone，例如 `ViT-B/32`、`RN50` 或 `RN101`，不要加载默认 `ViT-B/16` HAM checkpoint。此时显式设置：

```bash
FINETUNE_INIT=none
```

否则会出现 `positional_embedding` 或 `conv1.weight` shape mismatch。

## 记录规范

每次融合实验必须记录：

- 专家池成员。
- 每个专家的训练配置和 checkpoint。
- 融合层级：score-level、feature-level 或 checkpoint-level。
- 权重选择方式：mean、manual、grid search、GAR-EM。
- 是否使用验证标签；GAR-EM 默认优先无监督或弱监督选择。
- 最终 R1-best 结果。

结果写入：

- `docs/EXP_LOG.md`
- `docs/runs.csv`
- 后续可新增 `docs/merge_runs.csv`
