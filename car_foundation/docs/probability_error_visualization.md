# Transformer 概率输出与预测误差诊断

> **版本说明（2026-07-29）**：本文记录旧版 5 通道 direct Transformer + Linear Sigma Head 的历史诊断，通道中还包含网络直接预测的 `dyaw`。它不代表当前运动学残差 Query 模型。当前模型已经改为 4 维物理残差，并在冻结均值模型上验证了 S6-B Layer-Time-Channel Risk Head；最新结构、指标、选择性风险和结论见 [S6-B 验证报告](s6b_layer_time_risk_model_and_validation_20260729.md)。

## 应关注的两类问题

概率模型需要分别回答两个问题，不能只看一个 coverage 数字：

1. **尺度是否校准**：模型给出的 `sigma` 总体上是否与实际 RMSE 相当；
2. **逐样本是否有区分力**：模型是否会给真正困难、误差大的样本更高 `sigma`。

当前实车概率头在第一项上基本可用，但第二项较弱。

## 推荐指标

| 指标 | 作用 | 理想值/解释 |
| --- | --- | --- |
| Gaussian NLL | 同时评价误差和概率尺度 | 越低越好，只能同量纲/同数据比较 |
| 68/90/95% coverage | 检查预测区间覆盖率 | 接近对应名义值 |
| Calibration MAE | 可靠性曲线与对角线的平均距离 | 0 最好 |
| z mean / z RMS | 检查均值偏置和全局尺度 | 分别接近 0 / 1 |
| ENCE | sigma 分箱后，预测 RMS sigma 与实际 RMSE 的相对差 | 0 最好 |
| Spearman(sigma, abs error) | 检查逐样本不确定性排序能力 | 越接近 1 越好，0 表示几乎无排序信息 |
| Top-10% error AUC | 用 sigma 识别最大 10% 误差 | 0.5 为随机，1 为完美 |
| Top-10% recall | 最大 10% sigma 捕获了多少最大 10% 误差 | 随机约 10%，越高越好 |
| Excess kurtosis | 检查 Gaussian 未覆盖的重尾 | Gaussian 接近 0 |
| Selective risk | 丢弃高 sigma 样本后 RMSE 是否下降 | 保留低 sigma 数据时曲线应明显低于 1 |

## 本次严格实车测试结果

| 通道 | Spearman | Top-10% AUC | Top-10% recall | ENCE | Calibration MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dx_body` | -0.140 | 0.527 | 21.7% | 0.127 | 0.109 |
| `dy_body` | -0.026 | 0.465 | 8.9% | 0.106 | 0.069 |
| `dyaw` | 0.017 | 0.517 | 12.8% | 0.028 | 0.037 |
| `dvx` | 0.167 | 0.677 | 27.4% | 0.157 | 0.124 |
| `dyawrate` | 0.075 | 0.574 | 17.5% | 0.044 | 0.062 |

结论：

- `dyaw`、`dyawrate` 的整体 sigma 尺度较准确，ENCE 较低；
- `dvx` 的逐样本区分能力最好，但仍只是中等水平；
- `dy_body` 的 AUC 低于随机基线，sigma 基本不能识别高误差样本；
- `dx_body` 存在偏置和极重尾，sigma—误差关系非单调；
- 因此当前 sigma 更适合表达“通道/时域上的平均噪声尺度”，不适合直接作为逐样本安全风险分数。

## 图像说明

- `01_reliability.png`：名义 coverage 与实际 coverage；曲线在对角线上方表示区间偏宽，下方表示区间偏窄。
- `02_sigma_error_bins.png`：按 sigma 等频分箱，对比每箱预测 RMS sigma 与实际 RMSE；越接近对角线越好，曲线随 sigma 单调上升才说明能识别难样本。
- `03_horizon_*.png`：逐预测步比较 RMSE 与 RMS sigma，用于发现概率尺度随 horizon 失配。
- `04_standardized_residual_histogram.png`：`z=error/sigma` 与标准正态对比；对数纵轴突出重尾。
- `05_selective_risk.png`：从低 sigma 样本开始逐步保留；如果 sigma 有风险排序能力，小保留比例处的相对 RMSE 应明显小于 1。
- `06_episode_error_intervals.png`：单条中位误差和高误差样本上，实际残差相对 ±1 sigma/±2 sigma 区间的位置。

## 数据与复现

- 图像和指标：`outputs/transformer_probability_shift/20260727T200801/visualization`
- 逐样本数据：`probability_error_data.npz`
- 指标表：`metrics.csv`、`metrics.json`
- 生成脚本：`scripts/model_verify/visualize_transformer_probability_error.py`

NPZ 中每个条件都有 `*_error` 和 `*_sigma` 数组，形状为 `[episode, 50, 5]`，5 个通道依次为：

```text
[dx_body, dy_body, dyaw, dvx, dyawrate]
```
