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

## 2026-05-19 - 待跑：confidence-gated bridge

- 分支：`aeri-k2-ground-bridge-lite`
- 对比基准：`aeri_k2_ground_bridge_lite_w2_seed2`
- 建议实验名：`aeri_k2_ground_bridge_gated_m02_t01_seed2`
- 建议配置：`LOSS_NAMES=base+id+bridge`，`BRIDGE_MODE=gated`
- 建议参数：`BRIDGE_GATE_MIN=0.2`，`BRIDGE_GATE_TAU=0.1`

目标：在保留 ground bridge 主线的前提下，让 ground teacher 的监督强度根据样本置信度自适应变化。若 `ground-text` 相似度明显高于 `aerial-text`，增强 bridge；否则减弱 bridge，避免 ground teacher 过度拉动已经较可靠的 aerial feature。

结果跑完后追加到 `docs/runs.csv`，仍然用 best epoch 指标比较。

## 2026-05-20 - 负结果与经验记录

分支：`aeri-k2-ground-bridge-lite`

对比基准仍为 `aeri_k2_ground_bridge_lite_w2_seed2`。下面这些方向均已尝试或观察到早期曲线，不建议继续沿原形式推进；相关代码已 revert，避免污染当前稳定主线。

### Generative Photography inspired differential bridge

- 相关提交：`dacbcf2`、`2d767e6`，已由 `dd3ae0a`、`3d24f63` revert。
- 尝试内容：参考 camera-difference / residual conditioning 思路，为 `ground -> aerial` 建一个 residual bridge。
- 观察：第一版用 residual teacher 替代 plain bridge，早期 R1 曲线低于 plain bridge；第二版改为辅助项后仍缺少明确收益。
- 结论：当前 AERI-PEDES 线中，显式学习 `ground -> aerial` residual 容易干扰已有效的 detached ground bridge。若后续没有真实视角/相机标注，不建议继续做 residual bridge。

### MonSter++ inspired foreground-weighted image pooling

- 相关提交：`4c9f2ae`，已由 `a325e48` revert。
- 尝试内容：用 CLIP patch token saliency 加强中心/下方近景 prior，替代 CLS image feature 做 foreground-weighted pooling。
- 早期观察：epoch 5 已全面低于基线，R1/R5/R10/RSum/mAP/mINP 同时落后。
- 结论：强行改变 CLIP 图像特征池化会削弱文本-图像语义对齐。当前任务更依赖 CLS 的全局语义，不建议继续做 hand-crafted foreground pooling。

### DEFOM-Stereo inspired depth-gated bridge

- 相关提交：`cde4d73`，已由 `fc350d3` revert。
- 尝试内容：新增 `BRIDGE_MODE=depth_gated`，保持 CLIP image feature 不变，只用图像结构 pseudo-depth confidence 给 bridge loss 加 gate，并用较大 `BRIDGE_LOSS_WEIGHT=0.8`。
- 观察：曲线与 plain bridge 基线几乎重合，没有明显正负信号。
- 结论：当前无需离线深度文件的 pseudo-depth gate 区分度不足，实际接近常数 gate。若未来重启 depth 方向，应先离线缓存真实 depth-foundation-model 统计，再验证 gate 是否具备样本区分度；不要继续调当前 pseudo-depth proxy。

### 后续建议

- 保持当前 best baseline：`base+id+bridge`、plain detached ground bridge、random `k=2` per-ID sampling。
- 仍可优先验证 `BRIDGE_MODE=gated` 的文本相似度置信门控，因为它不改变图像特征，也不依赖 hand-crafted geometry prior。
- 若继续做新方法，优先选择“只调 loss 权重/样本权重”的保守实验；不要优先改 `encode_image`、patch pooling 或引入无标注几何 residual。
