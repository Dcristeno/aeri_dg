# AI 交接说明

分支：`aeri-cfan-baseline-cleanup`

## 当前状态

这个分支的实验记录已重置为模板。后续重新跑实验后，请同步更新：

- `docs/EXP_LOG.md`：叙述性实验记录和结论。
- `docs/runs.csv`：便于横向比较的结构化结果。
- `docs/ERROR_LOG.md`：环境、训练、评估或同步错误。

## 维护规则

- 每条实验记录都写明分支、实验名、SwanLab 链接和关键参数。
- 主要比较使用 best epoch 指标，不用 final epoch 指标替代。
- 新实验若改变 loss、采样、数据、初始化权重或评估方式，必须在记录中写清楚。
- 运行失败也要记录，尤其是环境、路径、checkpoint 和数据集问题。

## 常用入口

```bash
bash finetune.sh
```

## 待更新

- 当前 best run：
- 当前 baseline：
- 下一步实验计划：

