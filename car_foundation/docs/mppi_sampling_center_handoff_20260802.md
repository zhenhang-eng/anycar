# MPPI sampling-center 策略网络：工作交接

更新时间：2026-08-02

## 接续入口

后续恢复本任务时，先读本文，然后按“下一项实现任务”继续。当前没有训练或 ROS
采集进程在运行，尚未开始 teacher 标签生成和策略网络训练。

项目目标是让轻量策略网络根据状态、250 步 history、reference 和 MPPI warm start，
预测 8×2 knots 的 sampling-center 修正，再由 MPPI 在新中心附近采样。网络不直接
控制车辆。

## 已确定的技术路线

1. **先 DBM、后 Query。** 第一阶段的 rollout、teacher、reward、训练和闭环 A/B
   全部使用固定 Torch DBM，先隔离验证 sampling-center 策略是否有效。DBM 成功是
   Query 阶段的必要工程验证，不代表 Query 一定成功。
2. **先单步、后多步。** 先训练 bounded center residual 的 BC/critic 或单步
   contextual-bandit；只有单步离线和 DBM 闭环均通过后，才增加 n-step return 或
   counterfactual multi-step rollout。
3. **cost 权重和 temperature 进入网络条件。** 原始轨迹不绑定唯一 cost。训练时从
   配置中采样 cost weights/temperature，并据此动态生成 candidate cost、soft weight
   和 teacher center。
4. **原始数据不可变，标签使用 sidecar。** teacher、reward、split 和模型标签写到
   独立目录，必须保存原 snapshot 相对路径、配置和版本哈希。
5. **按 episode 切分。** 不得把同一 episode 的相邻 snapshot 或同一 snapshot 的
   candidate 随机拆到 train/validation/test。
6. **离线指标不能代替闭环。** 策略通过离线 held-out snapshot 后，必须在相同场景、
   seed、noise 和 rollout 预算下，与原 warm-start MPPI 做 DBM 闭环 A/B。

## 当前数据

专用根目录：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop
```

首选训练 seed：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/
fixed_dbm_train_seed_20260802_v2
```

该集合包含：

- 8 个 episode，MPPI seed 3500--3507；
- 96 个 snapshot，step 250--525，所有 250 步 history 完全观测；
- 24,576 条 50-step candidate DBM rollout；
- 4,208 条连续闭环 trace，每个 episode 为 step 0--525；
- snapshot Frenet `s=0.740--36.718 m`，横向误差 `-0.235--0.224 m`，航向误差
  `-0.268--0.223 rad`；
- `track.npz`、赛道 SHA256、scenario plan、逐 episode log 和关键源码 replay archive。

集合说明和场景表：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/
fixed_dbm_train_seed_20260802_v2/COLLECTION_SUMMARY.md

/disk/collect_data_from_anycar/mppi_rl_closed_loop/
fixed_dbm_train_seed_20260802_v2/scenario_plan.json
```

旧 `fixed_dbm_pilot_20260802_v1` 有 24 个 snapshot、6,144 条 rollout。它仍适合
单步 schema/重标注回归，但没有 schema-v2 的连续 trace/赛道副本；其中
`episode_000` 的 history 未完全实采，不得作为最终训练或测试数据。

## 数据完整性结论

当前 schema-v2 对下列任务没有不可补的核心字段缺失：

- 单步 sampling-center proposal、behavior cloning 和 candidate critic；
- 基于 DBM 轨迹/actions/reference 的任意后续 cost/reward 重标注；
- 改变 MPPI temperature 后重算 soft weight、ESS 和 soft teacher center；
- 从 snapshot 输入和固定 DBM 离线生成新的 center bank/teacher rollout；
- 从连续 trace 构造实际 DBM 行为策略下的 5/10/20-step return。

边界：

- 已保存的 `predicted_trajectories_full` 是 DBM 轨迹；以后切 Query 时可复用 candidate
  actions/history/reference，但必须用 Query 重新 rollout，不能沿用 DBM trajectory。
- 每个状态只有一个实际执行的闭环分支。若要比较多个 center 的未来闭环累计回报，
  需要从 snapshot 用 DBM 做 counterfactual receding-horizon rollout。
- 未出现过的 track、速度、动力学、噪声、delay、外扰、障碍物和失稳恢复状态不能靠
  relabel 补出，必须追加 episode。
- 96 个 snapshot 是 96 个状态上下文，足够打通 pipeline，不足以宣称策略泛化；
  24,576 条 candidate 不能当成 24,576 个独立状态。

## 当前采集 cost 基准

当前每条 candidate 的 collection-time cost 为：

```text
5.0 * position_error_sq
+ 5.0 * wrapped_yaw_error_sq
+ 1.0 * vx_error_sq
+ 0.05 * acceleration_rate_sq
+ 0.10 * steering_rate_sq
```

temperature 为 1.0，yaw-rate 权重为 0。保存的未加权逐步 features 和完整轨迹允许
以后改变这些权重。新的 cost-weight/temperature 采样范围尚未冻结，应通过配置文件
明确并写入 label manifest，不能硬编码在数据文件中。

## 已完成代码

- `car_dynamics/car_dynamics/controllers_torch/dbm.py`：保留完整六状态 candidate
  trajectory；
- `car_dynamics/car_dynamics/controllers_torch/mppi.py`：暴露 sampling mean/noise、
  raw/clipped knots 和 rollout；
- `car_ros2/car_ros2/car_node.py`：schema-v2 snapshot、trace、track、seed、history mask、
  manifest 和安全自动退出；
- `car_ros2/car_ros2/car_simulator_node.py`：可复现六状态初态和 simulator metadata；
- `car_ros2/launch/car_sim.launch.py`：数据参数传递和 controller 退出后关闭 simulator；
- `scripts/model_verify/validate_mppi_closed_loop_dataset.py`：兼容旧 pilot，并验证
  schema、cost replay、track/hash、连续 trace、action 链和 snapshot/trace 一致性；
- `scripts/model_verify/fixed_dbm_train_scenarios_20260802_v2.json`：正式批次场景表。

尚未实现：

- `generate_dbm_proposal_teacher.py`；
- proposal policy/critic 网络；
- BC/bandit 训练脚本；
- 离线 proposal evaluator；
- 网络 center 的 DBM 闭环 A/B；
- Query relabel 和 Query 闭环。

## 下一项实现任务

第一项只实现 **DBM teacher/relabel pipeline**，先不要接 Query、PPO、ONNX 或 ROS
在线策略：

```text
scripts/model_verify/generate_dbm_proposal_teacher.py
```

建议分两步：

### T0：复用现有 256 candidates 打通标签格式

对每个 snapshot 和每组 cost-weight/temperature 配置，从保存的 raw features/轨迹重算：

- `candidate_cost` 和 MPPI `candidate_weight`；
- `best_candidate_index`、best knots；
- soft-weighted teacher center；
- 相对实际 proposal center `sampling_mean_knots` 的 `teacher_delta_knots`；当前单轮
  MPPI 应同时断言它与 `mean_knots_before` 一致；
- warm candidate cost、best/soft-center regret、ESS、clip/boundary 指标。

T0 不产生新的 DBM rollout，只验证配置、sidecar schema、确定性和训练读取接口。

### T1：生成高预算/多中心 DBM teacher

T0 验证通过后，从 snapshot 恢复 state/history/reference/warm knots，用高预算 DBM
或多中心 bank 搜索更可靠的 teacher。teacher 预算、center bank 和 cost 配置必须进入
manifest。若最优候选频繁落在 T0 bank 边界，不能把 T0 best 当最终 teacher。

建议 sidecar 布局：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
  dbm_teacher_v1/
    manifest.json
    splits.json
    fixed_dbm_train_seed_20260802_v2/
      episode_000/step_000250.npz
```

原 collection 中的 `.npz`、manifest 和 trace 不得修改。

## 随后的策略网络验证

第一版网络输入：当前状态、history、`reference_ego`、current action、warm knots、
cost weights 和 temperature。输出为 `[8,2]` bounded `delta_knots`，使用 `tanh` 和
逐通道 trust region；先不学习 covariance。

推荐顺序：

1. episode-level pipeline split（当前 8 个 episode 可暂用 6/1/1，仅用于冒烟验证）；
2. BC 拟合 teacher delta；
3. held-out snapshot 比较 warm start、network center 和 teacher；
4. 比较 64/128/256 rollout 预算和多个 seed；
5. DBM 闭环 A/B，比较累计 cost、P95、tracking error、action 抖动、失败率、ESS、
   clipping、regret 和端到端延迟；
6. 网络进入 DBM 闭环后采集失败/分布偏移状态，做一轮 DAgger 式重标注和训练；
7. DBM 阶段通过后，才用固定 Query checkpoint 重新 rollout/relabel 并微调。

公平对比必须固定场景、初态、seed、noise、cost、temperature、horizon、sigma 和候选
预算。不能只比较离线 minimum cost，也不能随机拆 candidate 造成数据泄漏。

## 恢复环境与验证

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate anycar
source /home/plusai/anycar/set_env.sh
cd /home/plusai/anycar

python scripts/model_verify/validate_mppi_closed_loop_dataset.py \
  /disk/collect_data_from_anycar/mppi_rl_closed_loop/\
fixed_dbm_train_seed_20260802_v2/episode_000
```

最近一次状态：`car_dynamics` 和 `car_ros2` colcon 构建通过；8 个 schema-v2 episode
及 3 个旧 pilot 全部通过 validator，共 120 个 snapshot、30,720 条 candidate
rollout；场景表与 manifest 的 initial state/MPPI seed 一致；AnyCar skill 结构校验
通过；没有残留 ROS 采集进程。

本批 schema-v2 采集实现、validator、场景表和文档已随本文整理提交。恢复时仍应先
运行 `git status --short`，保留用户后续产生的修改。正式 episode 是在 dirty 工作树
上采集的，因此精确重放应同时使用集合中的 `replay_source_20260802.tar.gz`；不能只按
采集前的 repository commit 推断当时源码。

## 继续工作前必读文件

1. 本文；
2. [固定 DBM 的 MPPI 闭环数据采集](mppi_closed_loop_dataset_collection_20260802.md)；
3. [策略网络完整设计](rl_mppi_sampling_center_design_20260731.md)；
4. `/disk/collect_data_from_anycar/mppi_rl_closed_loop/DATASET_INDEX.md`；
5. `/home/plusai/.codex/skills/manage-anycar-environment/references/mppi-closed-loop-data.md`。

如果这些记录与运行文件发生冲突，以原始 episode manifest、scenario plan 和 validator
结果为准，并在继续训练前更新本文。
