# 0.21 m 小车 Query 数据、训练和 MPPI 对比

## 数据生成

`scripts/model_verify/generate_small_car_query_data.py` 使用纯 PyTorch DBM
批量生成标准 `CarDataset` PKL，不依赖 JAX。模型参数与 Quick Start 数值场景一致：

- `dt=0.05 s`
- `wheelbase=0.21 m`，`LF=0.1008 m`，`LR=0.1092 m`
- `mass=4 kg`，`friction=0.8`
- 归一化控制到物理控制：油门比例 8，转角比例 0.36、偏置 0.025 rad

生成命令：

```bash
python scripts/model_verify/generate_small_car_query_data.py
```

本次数据位于：

```text
/disk/collect_data_from_anycar/generated_small_car_query_dt005/20260730T144638
```

共有 256 个 PKL、1024 个 episode、1,024,000 条转换。数据记录的是执行动作前的
状态和当前动作，因此训练必须使用 `--steer-shift 0`。对应 Query 名义转向参数为
`steering_ratio=2.7777777777777777`、
`steering_offset_deg=-3.978873577297384`。

## 正式训练

旧实车 Query checkpoint 保持不变。新模型从头训练，避免把 3.9 m 大车的归一化统计、
名义轨迹和残差目标带入 0.21 m 小车模型：

```bash
python scripts/model_verify/train_kinematic_residual_ablation.py \
  --dataset-path /disk/collect_data_from_anycar/generated_small_car_query_dt005/20260730T144638 \
  --models query --include-kinematic \
  --epochs 40 --val-every 5 --early-stopping-patience 4 \
  --selection-metric rollout_score --rollout-loss-weight 0.01 \
  --wheelbase 0.21 --dt 0.05 \
  --steering-ratio 2.7777777777777777 \
  --steering-offset-deg -3.978873577297384 \
  --steer-shift 0 --batch-size 128 --eval-batch-size 256 \
  --num-workers 4 --output-dir outputs/formal_small_car_query_dt005
```

最佳 checkpoint 来自 epoch 30：

```text
outputs/formal_small_car_query_dt005/20260730T144840/query_best.pt
```

独立测试集结果：

| 模型 | 全时域位置 RMSE | 50 步位置 RMSE | 50 步速度 RMSE |
|---|---:|---:|---:|
| 纯运动学 | 0.686 m | 1.412 m | 0.880 m/s |
| 小车 Query | 0.099 m | 0.205 m | 0.051 m/s |

## ONNX 和 ROS 运行

ONNX 文件：

```text
outputs/formal_small_car_query_dt005/20260730T144840/anycar_query.onnx
```

256 候选轨迹校验中，CPU ONNX 与 PyTorch 的最大绝对差异为
`2.74e-6`；CUDA ONNX 的均值差异为 `4.97e-5`、最大差异为
`1.84e-3`（yaw-rate 通道）。

PyTorch Quick Start：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 ros2 launch car_ros2 car_sim.launch.py \
  mppi_backend:=pytorch \
  query_checkpoint:=$CAR_PATH/outputs/formal_small_car_query_dt005/20260730T144840/query_best.pt
```

ONNX Quick Start：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 ros2 launch car_ros2 car_sim.launch.py \
  mppi_backend:=onnx \
  query_checkpoint:=$CAR_PATH/outputs/formal_small_car_query_dt005/20260730T144840/query_best.pt \
  query_onnx_path:=$CAR_PATH/outputs/formal_small_car_query_dt005/20260730T144840/anycar_query.onnx
```

launch 的默认 checkpoint 仍是旧模型，只有显式传入上述参数才会切换。

## Quick Start 闭环对比

同一初始状态、`dt=0.05 s`、189 步（9.45 秒）结果：

| MPPI rollout 模型 | 横向 MAE | 横向 RMSE | 速度 MAE | 平均计算时间 |
|---|---:|---:|---:|---:|
| 旧 Query（3.9 m 大车） | 0.435 m | 0.609 m | 0.900 m/s | 29.0 ms |
| 新 Query（0.21 m 小车） | 0.118 m | 0.146 m | 0.303 m/s | 27.9 ms |
| Torch DBM | 0.113 m | 0.142 m | 0.337 m/s | 66.1 ms |

新 Query 相对旧 Query 的横向 MAE 降低约 72.9%，与 DBM 的横向 MAE
相差 4.7 mm；它的平均 rollout 计算时间比 DBM 少约 38.2 ms。

原始对齐结果：

```text
outputs/mppi_dbm_query_comparison/20260730/smallcar_dt005_comparison/summary.json
```

## 完整单圈对比

旧 Query 已从这次对比中排除。以赛道最近点循环索引首次累计达到 100% 作为
完整一圈结束条件：

| MPPI rollout 模型 | 单圈仿真时间 | 横向 MAE | 横向 RMSE | 速度 MAE | 平均计算时间 |
|---|---:|---:|---:|---:|---:|
| 新 Query（0.21 m） | 18.90 s | 0.109 m | 0.137 m | 0.232 m/s | 28.7 ms |
| Torch DBM | 19.30 s | 0.080 m | 0.107 m | 0.225 m/s | 64.6 ms |

完整单圈原始数据、指标和图位于：

```text
outputs/mppi_dbm_query_comparison/20260730/full_lap_smallcar_query_dt005
outputs/mppi_dbm_query_comparison/20260730/full_lap_dbm_dt005
outputs/mppi_dbm_query_comparison/20260730/full_lap_two_model_comparison
```

## 少量观测噪声完整单圈

本次诊断实验临时给控制器输入加入高斯观测噪声，并用独立 ground-truth
odometry 计算评价指标。噪声为：位置 `0.01 m`、航向
`0.005 rad`、速度 `0.02 m/s`、横摆角速度 `0.02 rad/s`，固定种子 4407；
车辆动力学、执行器和 ground truth 本身不加噪声。

| MPPI rollout 模型 | 带噪单圈时间 | 带噪横向 MAE | 无噪横向 MAE | 带噪速度 MAE | 平均计算时间 |
|---|---:|---:|---:|---:|---:|
| 新 Query（0.21 m） | 18.85 s | 0.138 m | 0.107 m | 0.219 m/s | 28.8 ms |
| Torch DBM | 19.05 s | 0.077 m | 0.077 m | 0.273 m/s | 66.4 ms |

带噪指标使用 ground-truth 状态到参考轨迹的距离计算，不使用控制器收到的带噪
odometry。这里只运行了一个固定噪声种子，因此不能把单次差值当作统计鲁棒性结论。

Query 的明显退化主要来自历史特征构造。部署端先对相邻两帧观测做差，形成
`dx`、`dy`、`dyaw`、`dvx` 和 `dyawrate`，再使用干净训练集统计进行归一化。
独立逐帧噪声经过差分后，标准差扩大为单帧噪声的 `sqrt(2)` 倍：

| 历史特征 | 差分噪声标准差 | 训练集标准差 | 归一化噪声幅度 |
|---|---:|---:|---:|
| `dx` | 0.0141 m | 0.0313 m | 0.45 sigma |
| `dy` | 0.0141 m | 0.00431 m | 3.28 sigma |
| `dyaw` | 0.00707 rad | 0.0471 rad | 0.15 sigma |
| `dvx` | 0.0283 m/s | 0.0324 m/s | 0.87 sigma |
| `dyawrate` | 0.0283 rad/s | 0.0710 rad/s | 0.40 sigma |

单步真实横向位移很小，因此位置噪声在 `dy` 通道被放大到 3.28 sigma；这些
带噪 token 会持续写入 250 步 history，随后影响 Query 残差预测并在 50 步 MPPI
rollout 中递推。训练数据来自干净 DBM 合成轨迹，没有包含相同的观测噪声分布，
因此构成明显的输入分布偏移。

Torch DBM 不使用历史序列，只用当前状态和控制量执行确定性动力学公式。小幅当前
状态误差只影响一次规划，并在下一次 0.05 s 闭环更新时被新观测纠正，所以横向
误差基本不变。但 DBM 的速度 MAE 从约 0.225 m/s 增至 0.273 m/s，说明它也受到
噪声影响；单一种子下横向误差略降更可能是随机抵消，不能视为鲁棒性结论。

后续若继续研究抗噪性，应优先使用状态估计结果构造 history，并在绝对状态上加入
噪声后重新计算差分来微调 Query；同时分别进行“仅当前状态加噪”和“仅 history
加噪”的消融，并使用多个随机种子报告均值和标准差。

```text
outputs/mppi_dbm_query_comparison/20260730/noisy_full_lap_smallcar_query_dt005
outputs/mppi_dbm_query_comparison/20260730/noisy_full_lap_dbm_dt005
outputs/mppi_dbm_query_comparison/20260730/noisy_full_lap_two_model_comparison
```

上述带噪运行仅作为一次性诊断实验。实验完成后，仿真节点中的观测噪声注入和
对应 launch 参数已移除；当前 Quick Start 的 `/odometry` 直接发布仿真真值。

## 固定 MPPI 采样快照

采样优化的冻结项、接口和后续比较要求以
[`mppi_sampling_optimization_status_20260730.md`](mppi_sampling_optimization_status_20260730.md)
为统一状态记录。采样研究主基准现已切换为精确 Torch DBM；本节下面的 Query
快照只保留作后续 learned-model 迁移验证，不作为 proposal 优化依据。

为了优化采样序列生成，在无观测噪声的 PyTorch Query Quick Start 中直接截取了
第 340 个控制步（仿真时间 17.0 s）。此时已经超过 250 步，history 已完全由闭环
转换填满。快照在同一次控制调用内原子保存当前状态、history、参考轨迹、warm-start
knots、随机数状态、全部候选动作、模型 rollout、cost 分项和 MPPI 权重。

```text
outputs/mppi_sampling_snapshot/live_query_clean_step0340/snapshot.npz
outputs/mppi_sampling_snapshot/live_query_clean_step0340/candidate_costs.csv
outputs/mppi_sampling_snapshot/live_query_clean_step0340/summary.json
```

固定状态为 `[x, y, yaw, vx, yawrate] = [2.3660, 3.5279, -2.6543,
1.9795, -0.2109]`，当前动作为 `[0.6467, -0.1339]`。基准仍使用 256 条候选、
50 步 horizon 和 8 个 knots：

| 指标 | 数值 |
|---|---:|
| 最优候选索引 | 0 |
| 最优 cost | 3.80217 |
| 平均 cost | 377.539 |
| 中位 cost | 186.414 |
| P95 cost | 1355.347 |
| 最大 cost | 2547.378 |
| Effective sample size | 1.06209 |
| 最优候选权重 | 0.96987 |

最优 cost 分项为：位置 1.56141、航向 0.59856、纵向速度 1.63213、加速度变化
0.00177、转向变化 0.00830，总和严格等于 3.80217。候选 0 是保留的当前均值序列，
其权重约 97%，说明该状态下现有高斯采样存在明显的权重退化，可将 ESS、最优 cost、
cost 分位数和低 cost 候选覆盖率作为新采样器的直接比较指标。

独立重算基准 cost：

```bash
python scripts/model_verify/evaluate_mppi_sampling_snapshot.py
```

重算的 256 条 cost 与在线快照最大绝对差为 0。评估新采样器生成的
`[N, 50, 2]` 归一化动作序列：

```bash
python scripts/model_verify/evaluate_mppi_sampling_snapshot.py \
  --candidates path/to/candidate_actions.npy \
  --output-dir outputs/mppi_sampling_snapshot/my_sampler
```

运行时也可以通过 `mppi_snapshot_step` 和 `mppi_snapshot_dir` 在其他控制步截取相同
格式的快照；默认 `mppi_snapshot_step=-1`，不会写文件，也不改变 Quick Start 行为。
