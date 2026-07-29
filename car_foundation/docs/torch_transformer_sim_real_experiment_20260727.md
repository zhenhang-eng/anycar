# TorchTransformerDecoder 仿真到实车实验（2026-07-27，修订版）

> **版本说明（2026-07-29）**：本文是旧版 6 维 direct 模型和概率头的历史实验。概率输出现已暂停，默认链路改为不含概率 Head 的运动学残差 Query 模型。

## 最终结论

早期实验采用逐 pkl 随机拆分，同一连续采集片段被分到 train、validation 和 test，相关结果不作为最终结论。本修订实验按生成 session 和采集日期隔离数据，并在 validation/test 之间保留时间缓冲。

重新训练后，仿真模型最佳 checkpoint 从短跑实验的 epoch 25 后移到 epoch 90；实车微调最佳 checkpoint 从 epoch 20 后移到 epoch 95。用户对“最佳 epoch 过小”的质疑成立，原实验的 20/30 epoch 上限不足以确定最优点。

严格拆分下仍观察到显著的 simulation-to-real 概率偏移：仿真 Gaussian 概率头直接用于实车测试集时，平均 95% 区间覆盖率只有 73.2%，标准化残差 RMS 在 `dy_body/dvx/dyawrate` 上分别达到 3.46/3.11/4.06。经过实车均值模型微调和实车概率头重估后，五个通道的标准化残差 RMS 恢复到 0.991–1.092，平均 95% 覆盖率恢复到 95.35%。

但实车残差仍明显非 Gaussian：实车适配后的 68% 区间平均覆盖率为 80.24%，`dx_body` 的 excess kurtosis 为 245.2。当前概率头可以作为风险排序和基线，不应直接作为安全关键概率保证；后续应测试 Student-t、mixture 或异常片段建模。

## 无泄漏拆分

### 仿真

数据目录含 6 个时间上彼此分离的生成 session，每个约 1.9 万 pkl：

- 前 4 个 session：训练池，从中固定抽取 16,000 个 pkl；
- 第 5 个 session：validation，固定抽取 3,000 个 pkl；
- 第 6 个 session：test，固定抽取 1,000 个 pkl；
- train→validation 时间间隔 465 秒；validation→test 时间间隔 1,645 秒。

对应 96,000/18,000/6,000 条 train/validation/test 序列。

### 实车

- 2025-03-31 全部 5,720 个 pkl 用于训练；
- 2025-04-01 前段 2,628 个 pkl 用于 validation；
- validation 后丢弃 139 个 pkl，形成约 87 秒缓冲；
- 2025-04-01 后段 918 个 pkl 用于 test。

对应 34,320/15,768/5,508 条 train/validation/test 序列。三个集合文件路径交集均为 0，连续时间组不会跨集合。

## 确定性模型训练

| 域 | Epoch 上限 | 最佳 epoch | 最佳 validation weighted MSE | 最后 epoch MSE |
| --- | ---: | ---: | ---: | ---: |
| 仿真，从头训练 | 100 | 90 | 0.014213 | 0.014583 |
| 实车，从仿真 epoch 90 微调 | 100 | 95 | 0.064334 | 0.064394 |

仿真 validation 曲线有较大波动，但总体在后半程继续改善；实车跨日期 validation 缓慢下降并在 80–100 epoch 附近进入平台。旧随机拆分的实车 validation 约 0.013，而严格跨日期拆分约 0.064，说明旧指标明显偏乐观。

## 独立测试结果

概率头只使用各训练 run 的 validation 文件：前 50% 拟合 scale head，中间 25% 选择模型，最后 25% 做 horizon×channel 温度校准。run 的 test 文件只在最终评估时使用。

| 指标 | 仿真模型 / 仿真 test | 仿真模型 / 实车 test | 实车适配模型 / 实车 test |
| --- | ---: | ---: | ---: |
| normalized Gaussian NLL | -2.684 | 1.247 | -2.453 |
| 平均 68% coverage | 79.53% | 44.48% | 80.24% |
| 平均 90% coverage | 96.74% | 67.16% | 92.84% |
| 平均 95% coverage | 98.73% | 73.21% | 95.35% |

仿真概率头在隔离的第 6 个仿真 session 上也偏保守，说明概率尺度本身存在 session shift；不能只依靠单个仿真 validation session 校准后就假设覆盖率普适。

### 实车均值误差改善

| 通道 | 仿真模型 RMSE | 实车适配 RMSE | 改善 |
| --- | ---: | ---: | ---: |
| `dx_body` | 0.015263 | 0.011182 | 26.7% |
| `dy_body` | 0.003718 | 0.000156 | 95.8% |
| `dyaw` | 0.000344 | 0.000199 | 42.0% |
| `dvx` | 0.005352 | 0.005146 | 3.8% |
| `dyawrate` | 0.000852 | 0.000734 | 13.9% |

实车适配后的标准化残差 RMS：

```text
dx_body 1.092, dy_body 1.030, dyaw 1.032, dvx 0.991, dyawrate 1.004
```

### 输入偏移线索

最大的输入 SMD 仍是 future throttle std（-1.264）；last steer 为 -0.259，其余所测状态和控制特征的绝对 SMD 均小于 0.06。未来油门激励范围是明显的域差异，但本实验没有做因果消融，不能把全部残差偏移归因于该特征。

## 产物

- 仿真 run：`outputs/checkpoints/2026-07-27T18:38:19.225-model_checkpoint`
- 仿真最佳模型：`outputs/checkpoints/2026-07-27T18:38:19.225-model_checkpoint/90/torch_model_90`
- 实车 run：`outputs/checkpoints/2026-07-27T19:35:58.713-model_checkpoint`
- 实车最佳模型：`outputs/checkpoints/2026-07-27T19:35:58.713-model_checkpoint/95/torch_model_95`
- 概率结果：`outputs/transformer_probability_shift/20260727T200801/summary.json`
- 概率头：`outputs/transformer_probability_shift/20260727T200801/probability_heads.pt`
- 固定拆分：`outputs/splits/session_isolated_sim_v1`、`outputs/splits/session_isolated_real_v1`

## 限制

- 单一随机种子，尚无跨 seed 置信区间。
- 仿真训练从四个 session 的约 7.7 万文件中抽取 1.6 万文件，未使用全部数据。
- 实车只有两个采集日期；跨日期 validation 更严格，但日期数量仍不足。
- `dvy` 的确定性训练权重为 0，因此不纳入概率分析。
- Gaussian 温度校准只能修正尺度，不能消除均值偏差和重尾。
