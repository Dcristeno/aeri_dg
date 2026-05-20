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

## 当前开发方向

分支：`aeri-k2-ground-bridge-lite`

当前不再继续推进 CFA。后续沿已有 `bridge` 模块做增强，第一版为 confidence-gated bridge：

- 默认 `BRIDGE_MODE=plain`，完全保持当前最好结果对应的普通 bridge。
- 实验模式 `BRIDGE_MODE=gated`，根据 `ground-text` 与 `aerial-text` 的相对相似度动态调节 bridge 强度。
- gate 公式为 `min_conf + (1 - min_conf) * sigmoid((sim_gt - sim_at) / tau)`，并 detach gate，避免模型通过操纵 gate 逃避 bridge 监督。
- 当前建议参数：`BRIDGE_GATE_MIN=0.2`，`BRIDGE_GATE_TAU=0.1`。

建议云端首跑：

```bash
RUN_NAME='aeri_k2_ground_bridge_gated_m02_t01_seed2' \
SWANLAB_EXPERIMENT='aeri_k2_ground_bridge_gated_m02_t01_seed2' \
BRIDGE_MODE='gated' \
BRIDGE_GATE_MIN=0.2 \
BRIDGE_GATE_TAU=0.1 \
SEED=2 \
bash finetune.sh
```

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
`runs.csv` 应优先记录能支持实验间比较的字段，例如 branch、rank、current_best、compare_to、main_change、best_epoch、best R1/R5/R10/RSum/mAP/mINP、delta_best_r1、bridge_mode、关键 loss 权重和 SwanLab 链接；不要用 last/final epoch 指标作为主比较字段。

维护规则：任何实验记录、结果说明或代码修改说明都要写明所属分支，方便后续 merge 时判断来源。

## 后续实验建议

后续保守实验都应对比 `aeri_k2_ground_bridge_lite_w2_seed2`：

- 如果需要估计方差，可以先对同一条线做 seed sweep；
- 可以围绕当前默认值 `0.5` 微调 `BRIDGE_LOSS_WEIGHT`；
- 尝试采样策略变体时，建议保留 random `k=2` 作为强控制组。

## 已放弃方向

截至 2026-05-20，下面方向已试过早期曲线或完整实现后撤回，后续不要在相同形式上重复投入：

- `differential bridge`：受 Generative Photography 的 camera residual 思路启发，但在当前数据缺少真实相机/视角标注时，显式 `ground -> aerial` residual 会干扰 plain bridge。
- `foreground-weighted image pooling`：受 MonSter++/depth prior 启发，用手工前景/近景 prior 替代 CLS 图像特征；早期六项验证指标全面低于基线，说明会破坏 CLIP 文本-图像语义对齐。
- `pseudo depth-gated bridge`：受 DEFOM-Stereo 启发，用图像结构 pseudo-depth confidence 调 bridge gate；曲线几乎贴合基线，说明当前 proxy 基本没有有效样本区分度。

若未来重启 depth 相关方向，应先离线生成真实 depth-foundation-model 统计，并验证 gate 分布确实有区分度，再接入训练；不要继续调当前 hand-crafted pseudo-depth proxy。
