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

下一步：

1. 使用固定 DBM 做高预算/多中心 T1 teacher，真正 rollout 新 proposal center。
2. 根据曲率、速度、误差和 warm-start cost 分桶检查 v2 seed 覆盖度。
3. 用 T1 teacher 做初版 BC 和 held-out DBM proposal 评估。
4. 追加速度/外扰/恢复边界和不同动力学参数的独立 collection，不混入固定 DBM split。
5. 再扩展多 cost 权重/temperature、reward sidecar 和 TD3/SAC 训练。

## 当前限制

- v2 seed 仍只覆盖一条 track、固定 DBM 和固定 reference 速度，不代表最终覆盖充分；
- 当前候选仍来自原 MPPI 单中心 Gaussian，尚未加入多尺度 center bank；
- simulator 与 Torch rollout 使用相同 DBM 参数和 RK4 方程，但仍应持续做逐步数值一致性检查；
- `episode_000` 仅用于数据管线回归，不应混入最终训练/测试。
