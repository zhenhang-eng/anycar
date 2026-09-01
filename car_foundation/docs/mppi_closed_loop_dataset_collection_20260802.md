# 固定 DBM 的 MPPI 闭环数据采集

日期：2026-08-02

策略网络的当前决策、已完成状态和下一项任务见
[MPPI sampling-center 策略网络：工作交接](mppi_sampling_center_handoff_20260802.md)。

## 目标

先冻结 numeric simulator 和 Torch DBM 参数，从物理一致的闭环 episode 中采集状态与
候选轨迹。当前 cost 只用于维持 MPPI 闭环控制和记录基线，不作为不可更改的训练标签。
后续可以基于原始轨迹重新定义 cost、temperature、teacher center 和 reward。

专用数据根目录：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop
```

## 采集链路

```text
固定参数 numeric DBM 环境
        ↓ odometry
DBM MPPI 闭环控制器
        ├─ 每个 control step：连续闭环 trace
        ↓ 每隔若干 control steps
snapshot：状态/history/reference/warm start
        + sampling mean/noise/raw/clipped knots
        + 256 条候选动作
        + 五状态和完整六状态 DBM 预测轨迹
        + 未加权逐时刻误差特征
```

闭环环境参数由 simulator 通过 `simulator_metadata` topic 发送给控制节点，并同时写入
episode manifest 和每个 snapshot。manifest 还记录代码 commit、dirty files、track、
MPPI 参数、独立随机种子、采集范围以及采集时 cost 权重。每个 episode 还保存
`track.npz`（原始赛道和 planner waypoints）以及赛道源文件 SHA256。

## Snapshot schema v2

主要数组如下：

| 字段 | shape | 含义 |
|---|---:|---|
| `initial_state_six` | `[6]` | `x,y,yaw,vx,vy,yawrate` |
| `history` | `[1,250,7]` | 250 步状态动作 history |
| `history_valid_steps` | scalar | history 中来自真实闭环的步数 |
| `history_is_fully_observed` | scalar | 是否已达到 250 个真实步 |
| `reference/reference_ego` | `[51,4]` | 全局/车体局部 reference |
| `mean_knots_before/after` | `[8,2]` | 闭环 warm start 更新前后 |
| `sampling_mean_knots` | `[8,2]` | 本轮实际采样中心 |
| `sampling_noise_knots` | `[256,8,2]` | 可精确重放的 knot noise |
| `raw_sampled_knots` | `[256,8,2]` | clip 前候选 |
| `sampled_knots` | `[256,8,2]` | clip 后候选 |
| `sampled_action_sequences` | `[256,50,2]` | 插值后的候选动作 |
| `predicted_trajectories_full` | `[256,50,6]` | 完整 DBM 轨迹，包含 `vy` |
| `feature_*` | `[256,50,...]` | 未乘权重的逐步误差/动作变化 |

`cost_*`、`cost` 和 `weight` 仍保留，用于复现采集时的控制结果；未来重新标注应优先
使用 `predicted_trajectories_full`、actions、reference 和 `feature_*`。

`closed_loop_trace.jsonl` 从 step 0 连续记录到最后一个 snapshot，包含六状态、当前/
控制器/实际执行 action、reference、Frenet pose、warm knots、优化动作序列和采集时
cost 摘要。它用于以后构造多步 transition/reward；候选 bank 仍只在 snapshot 步保存。

## 启动方式

先构建并加载环境：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate anycar
source /opt/ros/humble/setup.bash
colcon build --packages-select car_dynamics car_ros2 --symlink-install
source /home/plusai/anycar/set_env.sh
```

采集一个 episode：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 ros2 launch car_ros2 car_sim.launch.py \
  mppi_backend:=dbm \
  mppi_seed:=3500 \
  mppi_dataset_dir:=/disk/collect_data_from_anycar/mppi_rl_closed_loop/<collection> \
  mppi_dataset_episode_id:=<episode_id> \
  mppi_dataset_start_step:=250 \
  mppi_dataset_stride:=25 \
  mppi_dataset_max_snapshots:=20 \
  mppi_dataset_shutdown_on_complete:=True \
  sim_initial_state:=0,0,0,0,0,0
```

同一个 `episode_id` 不允许覆盖。`sim_initial_state` 按
`x,y,yaw,vx,vy,yawrate` 给出，便于在保持 DBM 参数不变时构造可复现的初态扰动。
达到 snapshot 数量后，launch 会等待控制节点和 simulator 正常退出；批量采集时必须
确认上一场完全退出后再启动下一场，避免 ROS topic 串台。

2026-08-06 起，若专门采集“Actor center + 固定邻域”数据，可显式使用：

```bash
mppi_num_samples:=64 mppi_sampling_mode:=fixed_hadamard_64
```

该模式使用冻结的满秩 `fixed-hadamard-64-v1`，不读取 `mppi_seed` 生成候选；默认
`gaussian` 保持旧采集行为。`mppi_seed` 仍写入 manifest，但固定模式下不能把不同 seed
误报成不同候选重复。固定候选只用于 MPPI center/wrapper 评价；连续 Actor 的主标签应由
其 8×2 knots 唯一插值到 50×2 actions 后做 direct DBM rollout 获得，不能用这里的
weighted-output cost 代替。详细口径见交接文档最新覆盖段。

验证：

```bash
python scripts/model_verify/validate_mppi_closed_loop_dataset.py \
  /disk/collect_data_from_anycar/mppi_rl_closed_loop/<collection>/<episode_id>
```

验证器检查 schema/shape/NaN、固定 simulator/DBM 参数、raw knots 重放、clip mask、
五/六状态轨迹投影、未加权 feature 复算、采集时 cost 复算、MPPI 权重和、赛道
artifact/hash，以及 trace 连续性和 snapshot/trace 交叉一致性。旧 manifest v1 仍兼容。

## 可训练性与可补性审计

对当前单步 sampling-center proposal、behavior cloning、candidate ranking/critic，数据
不存在不可补的核心字段缺失：网络输入、采样过程、完整六状态候选轨迹、action、
reference 和未加权 feature 均可重建。后续更换 cost 权重、temperature 或 reward
定义无需重新闭环采集；如果需要新的 teacher center，可以从 snapshot 状态和固定 DBM
离线增加 center bank 并重新 rollout。

需要区分“字段缺失”和“分布覆盖不足”：未出现过的 track、速度、动力学、外扰、噪声、
delay、障碍物和失稳恢复状态无法靠 relabel 补出，必须追加闭环 episode。旧 pilot 没有
连续 trace 和赛道副本，不能追溯中间 transition，但仍可用于单步训练/重标注。

## 已生成 pilot

集合：`fixed_dbm_pilot_20260802_v1`

| Episode | 初态 | steps | snapshots | candidate rollouts | 用途 |
|---|---|---|---:|---:|---|
| `episode_000` | `[0,0,0,0,0,0]` | 40--150 / 10 | 12 | 3,072 | 管线 smoke；history 未完全实采 |
| `episode_001` | `[0,0,0,0,0,0]` | 250--375 / 25 | 6 | 1,536 | 名义初态 pilot |
| `episode_002` | `[0,0.12,0.08,0,0,0]` | 250--375 / 25 | 6 | 1,536 | 横向/航向扰动 pilot |

共 24 个 snapshots、6,144 条候选 DBM rollout，约 21 MiB。三个 episode 均通过
`validate_mppi_closed_loop_dataset.py`。

当前运行观察：256 条 Torch DBM MPPI 在 RTX 3080 Ti 上稳定约 `57--60 ms/step`，
高于 `dt=50 ms`；写 snapshot 的步骤约 `82--90 ms`。因此当前管线适合离线数据采集，
正式实时部署仍需进一步批处理/内核优化。

## 已生成 training seed v2

集合：`fixed_dbm_train_seed_20260802_v2`

| Episode | track index | 初始横向偏移 m | 初始航向误差 rad | MPPI seed |
|---|---:|---:|---:|---:|
| `episode_000` | 0 | 0.00 | 0.00 | 3500 |
| `episode_001` | 94 | 0.08 | 0.06 | 3501 |
| `episode_002` | 188 | -0.08 | -0.06 | 3502 |
| `episode_003` | 282 | 0.12 | -0.08 | 3503 |
| `episode_004` | 376 | -0.12 | 0.08 | 3504 |
| `episode_005` | 470 | 0.06 | 0.10 | 3505 |
| `episode_006` | 564 | -0.06 | -0.10 | 3506 |
| `episode_007` | 658 | 0.00 | 0.12 | 3507 |

每个 episode 有 12 个 snapshot（step 250--525，stride 25）、3,072 条候选 rollout
和 526 条连续 trace。合计 96 个 snapshot、24,576 条候选 rollout、4,208 条闭环
trace，约 112 MiB；全部 snapshot history 完全观测，8 个 episode 均通过 validator。

聚合 snapshot 覆盖：Frenet `s=0.740--36.718 m`，横向误差
`-0.235--0.224 m`，航向误差 `-0.268--0.223 rad`。场景配置保存在集合内
`scenario_plan.json`；dirty 工作树所用关键源码保存在
`replay_source_20260802.tar.gz`。

## 下一步数据扩展

T0 teacher sidecar 已于 2026-08-03 生成：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
dbm_teacher_t0_20260803_v1
```

它按 episode 暂分 6/1/1，复用当前 256 candidates 生成 best/soft teacher delta、cost、
weight、regret、ESS 和 clipping 诊断，并保存 source SHA256。原 collection 未修改。实现、
标签语义和结果见
[teacher 标签方案与实现状态](mppi_teacher_label_plan_20260803.md)。

当时阶段状态（扩充前基线）：

1. T1 高预算/多中心 DBM teacher 已完成，正式 sidecar 为
   `labels/dbm_teacher_t1_20260803_v1`；独立 audit seeds 上 96/96 帧 weighted-output
   cost 改善。
2. T1 dataset loader、初版轻量 BC 和 fresh-seed DBM proposal 评估已完成；test episode
   weighted-output cost 仅相对 warm 改善 1.25%，还不足以进入 RL 或闭环部署。
3. 该覆盖缺口随后由本文件末尾的 30-episode 正式扩充数据解决；旧结果只保留为扩充前
   基线，不能再作为当前数据状态。

## 当前限制

- v2 seed 仍只覆盖一条 track、固定 DBM 和固定 reference 速度，不代表最终覆盖充分；
- 当前候选仍来自原 MPPI 单中心 Gaussian，尚未加入多尺度 center bank；
- simulator 与 Torch rollout 使用相同 DBM 参数和 RK4 方程，但仍应持续做逐步数值一致性检查；
- `episode_000` 仅用于数据管线回归，不应混入最终训练/测试。

## 2026-08-04 多速度扩充 pilot 与正式数据集

现有 v2 seed 的 96 帧参考速度全部严格为 `2.0 m/s`，snapshot `vx` 仅覆盖
`1.645--2.486 m/s`。第一版 BC 能记忆训练集但 test 改善仅 1.25%，因此下一批不能只是
在同一工况加密采样，必须同时扩状态数和条件分布。

为此新增 `mppi_reference_speed` launch 参数，并让 `GlobalTrajectory` 的 x/y 空间推进和
reference velocity 使用同一 override。validator 同时检查速度列和参考位置推进的一致性。
最初的 `fixed_dbm_expansion_pilot_20260804_v1` 只修改了速度列、没有修改空间推进，已经
标记 `REJECTED.md`，不得进入训练或 teacher。

修正后的 `fixed_dbm_expansion_pilot_20260804_v2` 包含 3 个 episode、60 个 snapshot 和
15,360 条候选 rollout，三档参考速度为 `1.4/2.0/2.6 m/s`。所有 episode 均通过完整
validator，实际 snapshot 覆盖：

- `vx=1.123--2.950 m/s`；
- Frenet 横向误差 `-0.292--0.186 m`；
- Frenet 航向误差 `-0.371--0.324 rad`；
- 每个 episode 479 步连续 trace，较大初始横向/航向、vy/yaw-rate 恢复状态均完成闭环。

冻结计划 `scripts/model_verify/fixed_dbm_policy_expansion_20260804_v3.json` 已于
2026-08-04 完整执行。正式 collection 位于：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/
fixed_dbm_policy_expansion_20260804_v3
```

30 个独立 episode 全部正常退出并逐场通过 validator；每场 20 帧，共 600 个 snapshot、
153,600 条 collection-time candidates 和 14,370 条连续 trace。episode split 为
18/6/6；train/validation/test 分别包含每档速度 6/2/2 个 episode，三档速度各 200 帧。
所有 snapshot 的 `history_valid_steps=250`，reference x/y 推进和速度列均与对应
`1.4/2.0/2.6 m/s` override 一致，未加入观测噪声。

正式数据 snapshot 分布为：

- `vx=1.053--3.092 m/s`，中位数 `2.032 m/s`；
- Frenet 横向误差 `-0.291--0.269 m`，绝对值中位数/P95 为 `0.052/0.164 m`；
- Frenet 航向误差 `-0.391--0.301 rad`，绝对值中位数/P95 为 `0.066/0.215 rad`；
- 航向误差绝对值大于 `0.20/0.25/0.30/0.35 rad` 的帧数分别为
  `43/12/5/2`，没有超过 `0.40 rad`；
- candidate-knot clip fraction 的均值为 `9.21%`；按速度分别为
  `4.43%/9.22%/13.99%`，高速度确实构成更困难的 proposal 分布；
- 每帧 best candidate cost 中位数 `11.068`，P10 candidate cost 中位数 `43.807`。

集合内 `COLLECTION_SUMMARY.md` 保存完整规模、分位数和使用约束。不得把任一 pilot 混入
正式 split。

## 2026-08-05 多场景、低候选预算正式集合

为解决上一版只有 18 个训练 episode、360 个 teacher 训练状态的问题，已完成新的固定
DBM 集合：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/
fixed_dbm_policy_diverse_20260805_v1
```

冻结场景表为
`scripts/model_verify/fixed_dbm_policy_diverse_20260805_v1.json`。120 个 episode 全部正常
退出并逐场通过 schema-v2 validator；split 为 90/15/15，每场 20 个 snapshot，共
2400 帧（train/validation/test=`1800/300/300`）和 936 MiB 原始数据。每状态候选数从
256 降为 64，因此总 collection candidate 数仍为 `2400*64=153,600`，与上一版
`600*256` 相同；增加的是独立状态和场景，而不是同状态重复 rollout。

训练 split 在五档参考速度 `1.2/1.6/2.0/2.4/2.8 m/s` 上，各包含 steady、cold start、
lateral recovery、heading recovery、dynamic recovery、combined recovery；validation/test
使用 episode 隔离的 heldout nominal/mixed/recovery。每个 train 类别恰好 15 个 episode，
validation/test 每类各 5 个。所有 snapshot 都从 step 250 开始、stride 12，history 完全
观测，未加入观测噪声。

全量实际 snapshot 分布：

- `vx=0.734--3.475 m/s`，中位数 `2.037 m/s`；
- 横向误差 `-0.611--0.489 m`，P05/P95 为 `-0.220/0.217 m`；
- 航向误差 `-0.499--0.488 rad`，P05/P95 为 `-0.251/0.247 rad`；
- best candidate cost 中位数 `18.491`，P95 `48.782`，最大 `180.308`；
- candidate cost 中位数的全量中位值为 `184.785`，最大 `1594.835`；
- knot clip fraction 均值 `10.62%`。

train/validation/test 的 `vx` 中位数均约 2.0 m/s，横向/航向误差中位数均接近 0，
best-cost 中位数分别为 `18.478/18.341/18.936`，没有明显 split 分布断层。该集合现为
sampling-center BC/critic/闭环 A/B 的首选固定 DBM 数据；旧 30-episode 集合只保留作
历史 checkpoint 对照。

### 2026-08-05 局部动作信息补全（不新增状态）

为支持 16 维 `[8,2]` sampling center 的局部 cost/critic 学习，已在上述 2400 个冻结
snapshot 上增加派生 sidecar
`labels/dbm_critic_fullrank_diverse_20260805_v1`。它不改变 raw collection、状态分布或
split；每状态只是在冻结 BC 周围增加 16 个正交方向的正负中心与独立 MPPI reward
复评。每帧 33 centers，selection/audit 各 4 seeds、每 seed 64 candidates，总计
40,550,400 条派生 DBM candidate rollout。2400 帧均通过 source/parent SHA256、中心
重构、边界、seed 隔离、shape、summary 和 rank=16 验证。该 sidecar 只能称为局部
reward 信息补全，不能计入新的独立训练状态数。

## 2026-08-04 扩充数据 teacher 与固定网络复评

冻结的 18/6/6 episode split 已完成 T0、T1 和同结构 BC 复评。当前优先使用：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
  dbm_teacher_t0_expansion_20260804_v2
  dbm_teacher_t1_expansion_20260804_v1

/home/plusai/anycar/outputs/mppi_proposal/
  bc_t1_conv_expansion_20260804_v1/
```

T0 的 600 个标签全部通过 candidate cost/weight replay。T1 使用与旧实验相同的
warm/T0-best/T0-soft 多起点、2 seeds × 3 stages × 128 CEM、最多 8 个 shortlist
centers、3 × 256 selection rollouts 和另 3 × 256 audit rollouts；600/600 帧的 selection
weighted-output cost 优于 warm，平均/中位改善 `4.938/3.815`。完全独立的 audit seeds
上 592/600 帧改善，平均/中位改善 `4.711/3.687`。其中 598 帧 shortlist 为 8，2 帧因
搜索中心完全重合而合法去重为 7，validator 已按“配置上限”而不是“固定长度”校验。

先冻结旧 checkpoint 在新 test 的 120 帧上评估，再重训相同 397,840 参数
Conv1d+MLP，二者均使用 fresh seeds `14001/14002/14003`：

| Checkpoint | warm cost | network cost | teacher cost | network-warm | 胜/负 |
|---|---:|---:|---:|---:|---:|
| 旧 96-state BC | 11.760 | 11.776 | 7.174 | -0.016 | 56/64 |
| 扩充数据 BC | 11.760 | 11.128 | 7.174 | +0.632 | 73/47 |

因此新增数据解决了旧网络在新分布上基本失效的问题，且新 checkpoint 回放旧 12 帧 test
仍平均改善 `0.695`，未发现明显回归。不过新 test 上 teacher 相对 warm 可改善 `4.586`，
网络只实现其中约 14%；网络预测 delta RMS 为 `0.041`，teacher 为 `0.133`，仍有明显
向 warm 收缩。下一项不再只增加相邻同类状态，而是保存同状态多个稳定 elite center，
训练 cost/ranking critic 或多候选 proposal head，处理单目标 MSE 的多解平均化；完成
held-out 单步复评后再做网络 center 的固定 DBM 闭环 A/B。

## 2026-08-05 两轮反馈派生数据

在相同 2400 个 frozen states 上完成
`labels/dbm_two_pass_feedback_diverse_20260805_v1`。它不是新增独立状态，而是每状态
用 2 组 selection 与 2 组 audit 两轮采样补充当前帧局部信息：第一轮 128 candidates
拟合轨迹 residual response，第二轮围绕 guided center 对 33 个全秩 outer centers 各跑
64 candidates，总计 21,504,000 条 DBM candidate rollout。所有 source/parent hash、
seed 隔离、shape、有限值、anchor 和 rank=16 验证通过。

对应 74 维 feedback-conditioned Critic 在 test/audit 的局部梯度 cosine 中位为
`0.615`，显著高于旧 state-only Critic 的 `0.193`，证明这类派生采样可用于训练
Critic；但直接 top-1 center 会使 cost 平均退化 `0.352`，尚不能生成 actor。完整标签
语义、尾部风险和下一步准入条件见 `mppi_sampling_center_handoff_20260802.md`。

随后新增 `labels/dbm_two_pass_risk_replay_diverse_20260805_v1`，仍不增加独立状态：
复用每帧第一轮反馈，沿冻结 Critic 方向评估 `±0.03/0.06/0.10/0.15 sigma`，每中心
selection/audit 各 8 个 reward seeds、每 seed 64 candidates，共 44,236,800 条派生
DBM rollout。2400/2400 全部通过 hash、重构和 reward 统计验证。该数据证明
`+0.03 sigma` 是当前最稳的固定步长；风险 gate 在 test/audit 可得到正平均收益并把
无约束大步更新的最坏 `-213.450` 压至约 `-4.9`，但尾部尚未归零，仍不计为 actor
准入通过。

## 2026-08-05 单步 Actor--Critic 补充状态

现有 risk-replay 已足以完成一个不依赖 next-state 的离散单步 Actor--Critic：五个正向
半径动作在每个上下文均有 8 个 selection 和 8 个 audit DBM reward seeds。加入状态、
history、reference 与 mean/P10 Critic 后，Actor 在全新 common-seed test 中 cost 为
`12.241`，T1 teacher 为 `11.889`；Actor 虽在 464/600 contexts 胜出，但高速度和恢复
状态的重尾使总体均值仍未超过 teacher。相关训练与公平复评产物分别位于
`outputs/mppi_proposal/single_step_state_actor_critic_20260805_v2` 和
`outputs/mppi_proposal/single_step_actor_teacher_eval_20260805_v2`。下一批数据应补充
actor 实际访问的多方向局部 centers，而不是增加 next-state 或多步 return。

## 2026-08-05 多方向单步 replay

新增派生 sidecar `labels/dbm_two_pass_multidirection_replay_diverse_20260805_v1`，复用原
2400 states 和 first-pass feedback，在 8 类运行时可构造方向、4 个半径上形成 33 个
中心，以 selection/audit 各 4 个 seeds 生成 81,100,800 条 DBM candidate rollout。
2400/2400 独立重构通过且不使用 teacher center。平坦 33-action learned Actor 未超过
teacher；但用一个 2112-candidate probe seed 选方向，再在全新 seeds 上复评，mean cost
为 `11.242`，低于 teacher `11.898`。这证明多方向局部反馈有足够 mean 上限，但当前
网络选择和高速度尾部仍未通过，后续数据应保存低预算 probe cost 作为层级策略输入。

## 2026-08-05 串行 probe Replay Buffer 结论

现有 33-center × 4 selection/audit seeds sidecar 已用于构造同状态内部搜索 transition，
不增加车辆状态，也不修改原始 collection。`train_mppi_sequential_probe_sac.py` 每批用
当前 Actor 访问中心、读取一个 probe seed 的真实 DBM cost，更新 observed mask/value 和
best-so-far，再把 transition 写入 Replay Buffer；Actor 与 twin target Critics 只在批次
间更新。首选 v2 使用其余三个 selection seeds 计算 terminal reward，避免把单 probe
seed 的幸运低 cost 当作最终回报。

v2 audit 中 4 probes/256 candidates 的 mean advantage 为 `5.379`，full 33 probes/
2112 candidates 为 `5.533`，说明预算压缩成立。fresh probe seed `28401`、evaluation
seeds `28411--28418` 下，固定优先级 4-probe、learned sequential 4-probe、T1 teacher
mean cost 分别为 `11.138/11.246/11.913`。learned policy 有 47 种实际序列并使用了反馈，
但比固定顺序差 `0.108`；因此当前 replay 足以学 probe 优先级，尚不足以证明自适应顺序
的额外价值。下一批应增加 actor-visited on-policy probe outcomes 和独立 reward repeats，
而不是把现有 4 seeds 的有限表格重复当成新 DBM 信息。

### 2026-08-06 Direct Actor on-policy replay 两轮

新增的 `dbm_direct_center_replay_diverse_20260806_v{1,2}` 不增加车辆状态，而是在已有
90 train + 15 validation episodes 的 2,100 snapshots 上重标 deterministic direct cost。
每轮 4,200 feedback contexts、164 centers/context、688,800 rollouts；15 test episodes
未生成标签。每个 context 的局部部分为 4 个衰减半径、16 个满秩 Hadamard 方向和正负 pair。
两轮独立 validator 的 center/cost 最大误差均为 0。

v2 不是 v1 的复制：它以第一轮 validation-calibrated `alpha=0.375` Actor 为新中心重新
rollout。全 context Actor/local-best mean 从 v1 的 `19.967/10.387` 降为
`17.736/9.856`。当前两轮 Actor 结果为 validation mean `27.445→25.974→25.518`，但
worst gain 仍为 `-109.109`，所以数据侧结论为“局部交互足以产生第二轮平均增益，tail
覆盖仍不足”；test 继续封存，下一步优先追加 tail 状态定向 replay 或 risk 标签。

tail 主要位于 validation 的 2.4--2.8 m/s 高动态段，而非普通稳态。完美 DBM 用上一轮
center 与新 Actor center 做二选一 direct-cost guard 时，mean 为 24.661、P05/worst gain
均为 0；这说明现有数据足以验证确定性 fallback 机制，但 Query 数据仍需补充这些 tail
状态的 pairwise cost 排序标签。

## 2026-08-07 train-only 独立场景扩充

针对 90 条训练轨迹不足以支撑 J16 蒸馏泛化的问题，新增不可变原始集合：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/
fixed_dbm_policy_train_expansion_20260807_v1
```

冻结计划为
`scripts/model_verify/fixed_dbm_policy_train_expansion_20260807_v1.json`，SHA256 为
`b5bf9026f1ad4f7b0797739d07f3b61d73aff5da02f3727e538c763f9d101d8f`。它只有 train：
270 个新 episode、validation/test 为 0；五档速度与六类场景形成 30 个分层，每层恰好
9 条新轨迹。与旧 train 合并后每层由 3 条增加到 12 条，独立训练 episode 从 90 增至
360。旧 validation/test 完全不变。

每条新轨迹只保留 step 250/300/350/400/450 五个稀疏且 history 完全观测的状态，避免
继续堆积相邻帧。270/270 episode、1350/1350 snapshots 已在采集后逐条验证，并由统一
`--resume` 再完整复核一次；每条 trace 均为连续 0--450 共 451 条记录。集合约 1.1 GiB，
未加入观测噪声，固定 DBM 参数与仿真完全匹配。

实际 snapshot 的 `vx` 范围/中位数为 `0.772--3.352 / 2.063 m/s`；横向误差范围/P05/P95
为 `-0.653--0.495 / -0.225/0.219 m`；航向误差为
`-0.515--0.475 / -0.236/0.244 rad`。best candidate cost 中位/P95/最大为
`18.482/50.231/207.428`，knot clip fraction 中位/P95/最大为
`9.18%/25.00%/45.51%`。按速度 best-cost 中位从 1.2 m/s 的 `7.991` 单调增加到
2.8 m/s 的 `31.056`，说明高速困难覆盖确实增加。

该集合只解决状态/episode 覆盖，尚未生成 J16 或 forward-cost-sensitive 标签，也不能
直接宣称 Actor 改善。下一步应在这 1350 个新状态上生成 train-only J16 与局部
antithetic cost 差分，再与旧 1800 个 train labels 合并训练；正式 validation/test 保持
冻结，用相同 mean/tail gate 判断是否真正缩小 25.518→4.894 的差距。

## 2026-08-13 hard two-center 闭环前置资格

当前最小闭环对象不是 soft two-center，而是保留原 warm-MPPI 输出的 hard guard。固定
状态部署 Gaussian 诊断表明，若 Actor center 已给定，原 warm 256 候选照常运行，再各用
一次 rollout 评价 warm weighted output 与 Actor direct sequence，hard min 可构造性保证
DBM model cost 不差于原 warm-MPPI 输出；该选择部分为 258 rollout。

必须计入 Actor 自身的在线输入成本。当前 residual Actor 依赖冻结 BC 周围的 128 条
first-pass rollout、74-D feedback、guided anchor 和三 Critic gradient context，所以严格
完整预算是 `128+1+256+1+1=387` rollout/step；first-pass 多出的 1 次用于获得 74 维
feedback 中的 weighted-output cost。脚本
`validate_mppi_residual_actor_runtime_inputs.py` 已从 300 个 consumed internal-selection raw
snapshot 重建两个 seed 的 600 个完整 context，最终 Actor center 最大误差 `9.09e-7`，资格
为 `PASS_RUNTIME_INPUT_RECONSTRUCTION`；正式 validation/test 未读取。

下一步将同一重建链封装进 DBM runtime，保持基线 seed、warm 采样和 running-state 递推
不变，只在末端 hard 选择执行动作，并进行 matched-seed 短闭环 A/B。闭环必须记录累计
cost、横向/航向误差、控制 rate/抖动、失败、Actor 选择率/连续性、每步延迟与超时率。
在 warm 256 bank 复用尚未独立证明前，不得把完整闭环预算写成 258。权威产物为修正预算
后的 `residual_actor_runtime_input_validation_20260813_v2`；v1 的数值重建正确但少计 1 次
first-pass weighted-output rollout。

## 2026-08-13 hard guard 首个 matched-seed 闭环 pilot

已新增共享 residual Actor runtime、side-effect-free sequence cost/hard guard 以及默认关闭的
ROS 参数入口，并完成一组 small-car DBM、2.8 m/s、seed 3407 的 baseline/guard A/B。数据位于
`mppi_hard_guard_closed_loop_pilot_20260813_v1`；两臂均有连续 301 步 trace 和 step
250--300 的 6 个快照，dataset validator 全部通过。分析/复算产物位于
`outputs/mppi_proposal/hard_guard_closed_loop_pilot_20260813_v1`。

全程 realized stage cost 改善 4.12%；成熟 51 步改善 2.29%，heading/speed RMSE 改善
8.80%/4.69%，但 lateral RMSE 退化 8.87%。成熟段 acceleration/steering rate RMS 退化
45.5%/16.4%，二阶差分退化 87.4%/28.2%。Actor 选择率 39.2%，分支切换率 34%；即时
model-cost hard floor 始终满足，但闭环平滑性不满足。guard 成熟段 mean/P95 延迟为
242/269 ms，baseline 为 63/90 ms，均超过 50 ms，guard 不具备实时性。

该结果只证明 open-loop 收益可部分传导，不能作为准入。下一轮内部采集应先加入保持 warm
下界的 hysteresis/minimum-dwell/switch-cost，再用多个 seed 与 nominal/high-speed/recovery
场景复跑；在 129 first-pass 无法复用或蒸馏前，不进入 formal validation。

## 2026-08-28 独立40--100 kph train-only采集

目标速度域已改为`40--100 km/h`。按当前用户决策，跳过DBM模型真实性/可达性审计，直接生成
独立的高速原始数据；旧`1.2--2.8 m/s`集合、split和sealed test均不修改。新集合为：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/
  fixed_dbm_highspeed_train_20260828_v1
```

冻结plan由`generate_fixed_dbm_highspeed_plan.py`生成，包含`40/55/70/85/100 km/h`五档速度、
六类steady/recovery工况、每格1个episode、每episode 20 snapshots，共30 episode / 600
train-only states。采集从step 250开始、stride 12、64 candidates；不创建validation/test。
普通ROS运行仍以10 m/s为reference ceiling，高速plan显式传30 m/s；该参数只解除采集入口限制，
不修改DBM方程、车辆参数、cost或动作边界。每个episode完成后必须运行原schema-v2 validator，
任何不完整episode不得被当作可用数据。

采集现已完成：30/30 episode、600/600 snapshots、38,400 candidate rollouts、14,370 trace rows，
所有episode validator通过，plan SHA256为
`db1676e2fb3ae1b88d03ff9ce030d4e39a754a05f40097547e93a7e48ad10144`。不过实际状态分布未达到名义
目标：step 250--478的`vx`中位/P95/最大仅`2.06/11.17/14.98 m/s`。因此原集合必须标为
高reference-speed压力集，而不是实际高速状态集；不得因schema验证通过就把“数据结构正确”误读成
“速度分布合格”。完整汇总在
`outputs/mppi_proposal/highspeed_collection_summary_20260828_v1/summary.json`。

同时新增compact train-only replay：从30条连续trace各取step 0--4，重建Query历史并重放DBM候选，得到
150 contexts / 9,600 candidates。实际`vx`范围`8.33--29.12 m/s`、中位`18.22 m/s`，82%处于
40--100 km/h。产物为
`outputs/mppi_proposal/highspeed_initial_dbm_replay_20260828_v1/replay.npz`及`summary.json`。该replay可用于
目标域pipeline smoke和标签生成，但仍不是formal validation/test；underspeed/overspeed边界样本保留。

该replay的首个teacher sidecar已完成：固定warm anchor、每状态129个proximal search centers，共
19,350次DBM center rollout。150/150 teacher严格改善warm，聚合相对降幅`2.9295%`，无基线违规，
独立anchor/teacher cost重放误差为0。sidecar位于
`outputs/mppi_proposal/highspeed_proximal_teacher_20260830_v1`，qualification为
`HIGHSPEED_PROXIMAL_SEARCH_TEACHER_COMPLETE_TRAIN_ONLY`。它只允许用于episode-grouped Actor可学性pilot；
不得把相邻5帧拆入不同fold，也不得与600帧step-250压力数据混成同一训练分布。

其后已完成train-only高速预训练sidecar：150帧按30个episode做5-fold、每fold三seed，生成15组
no-anchor Actor + Twin absolute-value Critic checkpoint。正式v2使用可测`current=[vx,yaw_rate,
acceleration,steering]`，不使用`vy/beta`。Actor OOF teacher-gain recovery中位`0.613`，但范围
`-0.783--1.149`且所有run的warm-relative P05均为负；Twin Critic的OOF warm-teacher排序准确率
中位`1.0`、129候选bank recovery中位`0.588`。独立fresh DBM重放和episode隔离检查全部通过。该产物
只能作为后续Actor-visited持续OAC的初始化，不能当成部署模型；600帧压力集仍未混入，sealed formal
validation/test仍未创建。

```text
outputs/mppi_proposal/highspeed_actor_twin_critic_pretrain_20260830_v2/
```

legacy-v1因包含`vy`已降级为非部署诊断，不得作为后续OAC初始化。

## 2026-08-30 高速Actor-visited OAC Replay sidecar

在正式v2预训练之后，已基于同一150-context、30-episode train-only集合完成20轮Actor-visited
terminal OAC。每轮为90个fit context新增33个center（Actor exact center + 16对antithetic probes），
DBM J50好/坏结果全部保留；Twin Critic与Actor持续按20:1更新。每个fold/seed的Replay与checkpoint均
独立落盘，internal selection和OOF按episode隔离，formal validation/test未创建。

选中Actor的OOF teacher-gain recovery中位由`0.613`升至`0.941`，胜warm比例中位由`0.667`升至
`0.800`；但warm-relative P05门失败，fresh局部Critic sign中位`0.698<0.70`。所以该sidecar证明持续
在线反馈对主体有效，却仍不能作为单中心部署数据。warm+Actor two-center guard仍是硬约束，闭环仍未
启动。独立validator对OOF center/probe和抽样Replay fresh DBM重放，全部最大误差为0、episode leakage为0。

```text
outputs/mppi_proposal/highspeed_actor_visited_oac_20260830_v1/
```

该数据集qualification为`HIGHSPEED_ACTOR_VISITED_OAC_COMPLETE_TRAIN_ONLY`，独立复算为
`HIGHSPEED_ACTOR_VISITED_OAC_INDEPENDENT_REPLAY_PASS`。

## 2026-08-30 目标高速独立状态扩张（120 episodes）

已按冻结plan完成第二批目标域train-only采集：40/55/70/85/100 km/h x 6类scenario x 4个独立
episode，每episode只取step 0--4的5个early context。正式集合为：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/
  fixed_dbm_highspeed_expansion_20260830_v2
```

总量为120 episodes / 600 snapshots / 38,400初始MPPI candidates；实测速度范围`28.1--110.3 km/h`、
中位`63.2 km/h`，`78.3%`位于40--100 km/h。所有episode均通过schema validator；区间外的recovery
边界状态保留，不做cost或速度过滤。一次ROS metadata尚未就绪的episode已移入`logs/quarantine`后
重新采集，quarantine内容不计入120 episode。采集代码现在保证step 0同callback写snapshot/trace时不
误判覆盖，并在DBM metadata可用前延后snapshot。

对应compact replay为600 contexts / 38,400 candidates，独立validator通过；proximal teacher为
77,400 center rollouts，600/600严格改善anchor、聚合下降`3.1386%`。后续5-fold x 3-seed预训练与
20轮Actor-visited OAC均只使用本train集合：OAC selected OOF teacher-gain recovery中位`1.031`，
胜warm比例中位`0.808`；但P05尾部与Critic local-sign门仍失败。因此该扩张集已证明对主体学习有效，
仍不构成formal/closed-loop准入；two-center warm floor继续是硬约束。

```text
outputs/mppi_proposal/highspeed_collection_summary_expansion_20260830_v1/
outputs/mppi_proposal/highspeed_initial_dbm_replay_expansion_20260830_v1/
outputs/mppi_proposal/highspeed_proximal_teacher_expansion_20260830_v1/
outputs/mppi_proposal/highspeed_actor_twin_critic_pretrain_expansion_e4_20260830_v2/
outputs/mppi_proposal/highspeed_actor_visited_oac_expansion_e4_20260830_v1/
```

## 2026-08-30 高速两轮Actor-search派生标签

在上述600个train-only context及三个OOF OAC Actor center上生成两轮严格best-improvement路径。每个
context固定195个唯一center，共117,000次DBM rollout；独立validator全候选重放误差为0，零基线和
单调性违规。派生标签不新增episode，不改变原始collection，也不创建formal validation/test。

```text
outputs/mppi_proposal/highspeed_two_round_actor_paths_20260830_v1/
outputs/mppi_proposal/highspeed_iterative_path_distillation_expansion_20260830_v1/
```

路径中第一轮严格移动率`96.67%`，两轮累计严格改善率`99.11%`；stay由真实DBM incumbent规则产生，
不是额外分类标签。所有cross-fit继续按原120个episode分组，同episode的5个control context不会跨fold。
