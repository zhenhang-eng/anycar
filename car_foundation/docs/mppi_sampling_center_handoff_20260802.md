> **[CLOSED-HISTORY 2026-08-20]** 本文档记录的路线已关闭/被取代。权威结论见 `mppi_sampling_center_review_20260812.md` §11.80。本文档不再更新，仅作历史参考。

# MPPI sampling-center 策略网络：工作交接

更新时间：2026-08-04

## 接续入口

后续恢复本任务时，先读本文，然后按“下一项实现任务”继续。当前没有训练或 ROS
采集进程在运行；多速度正式扩充数据的 30 个 episode、600 帧 T0/T1 teacher、固定
397,840 参数 BC 重训、新旧 checkpoint 复评、multi-elite 标签和 fresh-seed oracle 均已
完成。下一项是 K=2 mode-preserving BC；禁止对 heads 或多中心 candidates 再做简单
平均，通过后再做固定 DBM 闭环 A/B。

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

正式扩充 collection（当前下一阶段首选）：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/
fixed_dbm_policy_expansion_20260804_v3
```

它包含 30 个 episode、600 个 fully observed snapshot、153,600 条候选 rollout 和
14,370 条连续 trace，冻结 split 为 18/6/6，三档 `1.4/2.0/2.6 m/s` 速度各 200 帧。
全部 episode 已逐场通过 validator。实际 `vx=1.053--3.092 m/s`；横向误差
`-0.291--0.269 m`；航向误差 `-0.391--0.301 rad`，其绝对值中位数/P95 为
`0.066/0.215 rad`。详细统计见 collection 内 `COLLECTION_SUMMARY.md`。

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
- `scripts/model_verify/generate_dbm_proposal_teacher.py`：从 schema-v2 raw features
  复算多配置 candidate cost/weight，生成 best/soft teacher sidecar；
- `scripts/model_verify/validate_dbm_proposal_teacher.py`：校验 source/config hash，并独立
  复算 T0 标签；
- `scripts/model_verify/dbm_teacher_cost_configs_20260803_v1.json`：当前 teacher cost
  配置，第一项严格复现 collection objective。
- `scripts/model_verify/generate_dbm_multicenter_teacher.py`：T1 多起点、多轮 CEM 和
  common-random-number proposal 评估；
- `scripts/model_verify/validate_dbm_multicenter_teacher.py`：独立复算 T1 cost、分布指标、
  selection score 和 teacher argmin；
- `scripts/model_verify/dbm_teacher_t1_config_20260803_v1.json`：冻结 T1 搜索、selection、
  audit seed 和 score 权重。
- `car_foundation/car_foundation/mppi_proposal_policy.py`：397,840 参数的 Conv1d+MLP
  bounded-residual proposal policy；
- `scripts/model_verify/train_mppi_proposal_bc.py`：episode split dataset、train-only
  normalization、1/2-sigma trust 与三 seed BC；
- `scripts/model_verify/evaluate_mppi_proposal_bc.py`：用新 DBM seeds 对
  warm/network/teacher 做 common-random-number proposal 评估；
- `scripts/model_verify/plot_mppi_proposal_bc.py`：生成 T2 label-fit 与 DBM cost 汇总图。
- `scripts/model_verify/collect_fixed_dbm_scenarios.py`：按冻结场景表串行启动 ROS、记录
  episode log，并在每场后运行 validator；
- `scripts/model_verify/fixed_dbm_expansion_pilot_20260804_v2.json`：已验证的三速度 pilot；
- `scripts/model_verify/fixed_dbm_policy_expansion_20260804_v3.json`：30-episode 正式扩充
  计划，18/6/6 split，每个 split 均覆盖 1.4/2.0/2.6 m/s。

尚未实现：

- 网络 center 的 DBM 闭环 A/B；
- 多模态 teacher/critic/cost-aware policy；
- 扩充数据后的容量消融和 DAgger；
- Query relabel 和 Query 闭环。

## 下一项实现任务

T0/T1 **DBM teacher pipeline** 与 T2 轻量 BC 基线均已完成。先不要接 Query、
TD3/SAC、PPO、ONNX 或 ROS 在线策略。正式 T1 标签位于：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
dbm_teacher_t1_20260803_v1
```

建议分两步：

### T0：复用现有 256 candidates 打通标签格式（已完成）

对每个 snapshot 和每组 cost-weight/temperature 配置，从保存的 raw features/轨迹重算：

- `candidate_cost` 和 MPPI `candidate_weight`；
- `best_candidate_index`、best knots；
- soft-weighted teacher center；
- 相对实际 proposal center `sampling_mean_knots` 的 `teacher_delta_knots`；当前单轮
  MPPI 应同时断言它与 `mean_knots_before` 一致；
- warm candidate cost、best regret、soft candidate-cost expectation、ESS、clip/boundary
  指标。

T0 不产生新的 DBM rollout，只验证配置、sidecar schema、确定性和训练读取接口。soft
center 自身没有在 T0 rollout，因此不能把 weighted candidate cost expectation 称为
soft-center cost。96 帧已全部通过 validator：warm candidate 为 best 的比例为 44/96，
平均 ESS 1.543，best candidate clipping 比例 20.8%。详细结果见
[teacher 标签方案与实现状态](mppi_teacher_label_plan_20260803.md)。

### T1：生成高预算/多中心 DBM teacher（已完成）

T0 验证通过后，从 snapshot 恢复 state/history/reference/warm knots，用高预算 DBM
或多中心 bank 搜索更可靠的 teacher。teacher 预算、center bank 和 cost 配置必须进入
manifest。若最优候选频繁落在 T0 bank 边界，不能把 T0 best 当最终 teacher。

当前实现每帧从 warm/T0-best/T0-soft 做 2 seeds × 3 stages × 128 CEM search，shortlist
8 个中心后以 3 seeds × 256 rollout 选择 teacher，并以另外 3 个 seed 做 warm/teacher
audit。全量 96 帧 audit weighted-output cost 均改善，平均降低 5.897；P10 在 86/96 帧
改善。详细预算、score 和限制见
[teacher 标签方案与实现状态](mppi_teacher_label_plan_20260803.md)。

当前 T1 sidecar 布局：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
  dbm_teacher_t1_20260803_v1/
    manifest.json
    splits.json
    teacher_config.json
    labels.csv
    episode_000/step_000250.npz
```

原 collection 中的 `.npz`、manifest 和 trace 不得修改。

### T2：BC 和离线 proposal 验证（第一版已完成）

第一版读取 T1 `teacher_delta_knots`，按 6/1/1 episodes 训练 Conv1d+MLP `[8,2]`
bounded residual head。1/2-sigma trust 各跑三个 seed，validation 选中 2-sigma seed 2。
fresh DBM seeds `14001..14003` 上，test weighted-output cost 为 warm `12.522`、network
`12.153`、teacher `7.494`；network 平均相对改善仅 `1.25%`，胜负 `6/6`，P10 和
direct-center cost 没有改善。完整结果见 teacher 状态文档第 7 节。

### T2.1：扩充数据固定结构复评（已完成）

正式扩充、T0/T1 和固定结构复评均在 2026-08-04 完成。当前产物：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
  dbm_teacher_t0_expansion_20260804_v2/
  dbm_teacher_t1_expansion_20260804_v1/

/home/plusai/anycar/outputs/mppi_proposal/
  bc_t1_conv_expansion_20260804_v1/
```

T1 的 600 帧全部通过 validator；selection weighted-output cost 相对 warm 平均降低
`4.938`，独立 audit seeds 上 592/600 帧改善、平均降低 `4.711`。旧 checkpoint 先在
新 test 120 帧上冻结评估，weighted-output cost 为 warm/network/teacher
`11.760/11.776/7.174`，旧网络平均退化 `0.016`。保持结构、loss、trust 和 seeds 不变
重训后，新 checkpoint `trust1_seed1.pt` 得到 `11.760/11.128/7.174`，平均改善
`0.632`，胜负 `73/47`；回放旧 test 也平均改善 `0.695`。

这证明数据覆盖是旧网络跨分布失效的重要原因，但还没有解决全部问题：新网络 test
prediction/teacher delta RMS 为 `0.041/0.133`，只实现 teacher 相对 warm 平均
`4.586` 改善中的约 14%。图和逐帧数据位于：

```text
outputs/mppi_proposal/bc_t1_conv_expansion_20260804_v1/
  old_checkpoint_new_test_eval/
  new_checkpoint_new_test_eval/bc_t1_expansion_result.png
  new_checkpoint_old_test_eval/
```

### T2.2：multi-elite 标签与 oracle（已完成）

已从扩充 T1 shortlist 生成并校验 600 帧 multi-elite sidecar：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
dbm_multi_elite_expansion_20260804_v1
```

每帧保留最多 4 个、至少相距 `0.20 sigma`、跨 selection seeds 稳定且 cost 接近 teacher
的中心；有效数 1/2/3/4 分别为 `106/300/114/80`，494/600 帧至少有两个。test 使用
全新 DBM seeds `15001..15003` 做 oracle：warm/teacher cost 为 `11.660/7.165`；K=2/3
完美 full-budget selector 仅到 `7.139/7.131`，额外收益 `0.026/0.034`；固定总 256
candidates 的 K=2/3 mixture 为 `7.383/7.269`，反而退化。

与此同时，elite center 算术平均为 `7.382/7.630`；在真正具备 K 个中心的子集上，
K=2/3 平均分别比 teacher 差 `0.269/1.068`。将 teacher residual 缩为 25%/50%/75%
时 cost 为 `10.678/9.281/8.036`，明显差于完整 teacher `7.165`。因此训练中的 mode
平均/向 warm 收缩是可信问题，但“在线同时混合多个中心”不是新的主要性能上限。

完整图和结果位于：

```text
outputs/mppi_proposal/dbm_multi_elite_oracle_20260804_v2/
  summary.json
  per_snapshot.csv
  multicenter_oracle_result.png
```

### T2.3：K=2 mode-preserving BC（下一项）

先做共享 encoder + 两个 bounded residual heads 的最小受控实验。训练读取 multi-elite
mask，使用 winner-take-all/set matching，另加 gating/ranking 选择单个 head；禁止对
heads 求平均，也暂不把 MPPI 预算均匀拆到多个中心。目标不是超过 T1 teacher，而是验证
能否从当前单头 BC 的 `11.128` 明显接近 teacher `7.174/7.165`，并减少相对 warm 的
退化帧。若 K=2 oracle-prediction 明显好而 learned gating 不好，再集中改 critic；若
K=2 网络本身仍收缩，再考虑 mode classification、cost-aware loss 或 Diffusion。

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

最近一次状态：`car_dynamics` 和 `car_ros2` colcon 构建通过；正式扩充 collection 的
30 个 episode、600 个 snapshot、153,600 条 candidate rollout 均通过 validator；
场景表与 manifest 的 initial state/MPPI seed 一致；新 T0/T1 的 600 个 label 均通过
独立 validator，T1 的独立 audit seeds 上 592/600 帧 weighted-output cost 改善；固定
结构 BC 的新旧 checkpoint 公平复评已完成；没有残留 ROS 采集或训练进程。

本批 schema-v2 采集实现、validator、场景表和文档已随本文整理提交。恢复时仍应先
运行 `git status --short`，保留用户后续产生的修改。正式 episode 是在 dirty 工作树
上采集的，因此精确重放应同时使用集合中的 `replay_source_20260802.tar.gz`；不能只按
采集前的 repository commit 推断当时源码。

## 继续工作前必读文件

1. 本文；
2. [固定 DBM 的 MPPI 闭环数据采集](mppi_closed_loop_dataset_collection_20260802.md)；
3. [teacher 标签方案与实现状态](mppi_teacher_label_plan_20260803.md)；
4. [策略网络完整设计](rl_mppi_sampling_center_design_20260731.md)；
5. `/disk/collect_data_from_anycar/mppi_rl_closed_loop/DATASET_INDEX.md`；
6. `/home/plusai/.codex/skills/manage-anycar-environment/references/mppi-closed-loop-data.md`。

如果这些记录与运行文件发生冲突，以原始 episode manifest、scenario plan 和 validator
结果为准，并在继续训练前更新本文。

## 2026-08-05 最新交接：120 episodes / 1800 train states

上一节的 30-episode、600-state 扩充现只作历史对照。当前首选固定 DBM 资产为：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/
  fixed_dbm_policy_diverse_20260805_v1/
  labels/dbm_teacher_t0_diverse_20260805_v1/
  labels/dbm_teacher_t1_diverse_20260805_v1/

/home/plusai/anycar/outputs/mppi_proposal/
  bc_t1_conv_diverse_20260805_v1/
```

collection 有 120 episode、2400 snapshot、90/15/15 episode split，train/validation/test
状态为 1800/300/300。五档速度 1.2--2.8 m/s，训练含六类稳态/启动/恢复场景；每状态
64 candidates，总 collection rollout 数仍为 153,600。120 条均正常退出并逐场通过
validator，未加入观测噪声。

低预算 T1 每状态 2048 rollouts，2400 标签全部通过严格 validator；独立 audit seeds
上 2392/2400 weighted-output cost 改善，平均 `8.896`。首选 BC checkpoint 为
`bc_t1_conv_diverse_20260805_v1/trust1_seed1.pt`。在独立 test 300 帧和 fresh seeds
20001--20003 上，warm/network/teacher weighted-output cost 为
`21.622/18.611/12.136`，网络相对 warm 平均改善 `3.011`，胜负 `234/66`。

同一 test、seed 和 64-candidate 预算下，旧 360-state checkpoint cost 为 `20.296`；
新 checkpoint 平均再降低 `1.685`（8.30%），211/300 帧胜。五档速度与三类 heldout
scenario 的 paired 均值全部改善。数据覆盖不足确实是主要瓶颈之一，但新网络仍只恢复
约 31.7% 的 teacher gain。

恢复工作时不再把 K=2 mode-preserving BC 当默认下一项；已有 oracle 表明多中心相对
单 teacher 的额外上限很小。下一步是：

1. 在固定 DBM、同一场景/seed/noise/cost 下，把新 checkpoint 接入闭环 A/B；
2. 比较累计 cost、P95 tracking error、控制平滑性、失败率、clipping 和延迟；
3. 保存 network 相对 warm 退化以及闭环分布漂移状态，做 DAgger 式 T1 重标注；
4. 再做 gain/confidence 加权 BC 或 center cost/ranking critic；
5. 固定 DBM 闭环通过后才进入 Query relabel、RL 或 ONNX/ROS 部署。

关键实现新增：`car_sim.launch.py` 的 `mppi_num_samples`，多场景生成器
`generate_fixed_dbm_diverse_plan.py`，冻结计划
`fixed_dbm_policy_diverse_20260805_v1.json`，以及低预算 teacher config
`dbm_teacher_t1_diverse_20260805_v1.json`。继续前先查看上述三个 summary 和本节结果，
不要重新随机拆分 episode。

### 2026-08-05 BC 初始化准入与 Critic 路线修正

BC 的角色是为后续 reward/cost 优化提供初始 actor，而不是要求先完整复制 teacher。
当前 checkpoint 已达到进入单步 Critic 阶段的准入条件：fresh test 300 帧相对 warm
平均降低 `3.011` weighted-output cost、234/300 帧改善，且五档速度和三类 heldout
场景的均值全部为正。aggregate teacher gain recovery 为 `31.7%`，因此它不是随机或
退回 warm 的初始化。

仍需保留安全约束：66/300 帧退化，其中 34 帧退化超过 2、12 帧超过 5、1 帧超过
10，最坏退化 `17.680`；gain 的 P01/P05/P10 为 `-9.142/-3.624/-2.153`。这不要求继续
无期限做 teacher MSE，但禁止直接无约束 SAC/TD3 或部署网络中心。

修正后的下一项是先训练**单步 proposal cost/ranking Critic**：输入与 actor 相同的
状态上下文以及一个 `[8,2]` center，目标为 T1 shortlist 在多个 common seeds 下的
mean weighted-output cost。先验证 episode-heldout cost 误差、同状态 pairwise ranking、
warm/teacher 选择准确率与 regret；通过后使用 Critic + BC trust region 微调 actor，并
用真实 DBM 复评 actor 新输出、迭代补充 Critic 数据。短闭环 TD3/SAC 和完整闭环 A/B
均放在这一单步 cost-aware 阶段之后。

### 2026-08-05 第一版 Critic 结果与准入边界

第一版实现位于 `mppi_proposal_policy.py::TorchMPPIProposalCritic` 和
`train_mppi_proposal_critic.py`，有 447,489 个参数；复用 BC 的状态 encoder，并增加
候选 center 及其相对 warm 标准化残差 encoder。训练目标是
`warm weighted-output cost - center weighted-output cost`，正值表示候选更好；loss
组合 Smooth-L1、同状态 pairwise ranking 和 warm=0 anchor。训练仍使用冻结的
90/15/15 episode split，3 个 seed 的最佳 epoch 为 49/47/39。

3-seed ensemble 在 T1 held-out test 300 状态、1500 个 shortlist center 上达到：

- pairwise accuracy `0.801`，Spearman 约 `0.68`；
- warm/teacher 判别准确率 `0.983`；
- 选择后 cost `20.865 -> 13.406`，平均 true advantage `7.458`；
- 287/300 状态优于 warm，平均 oracle regret `1.334`，但 top-1 仅 `0.427`。

在与训练/teacher seed 不相交的 20001--20003 fresh test 上，Critic 对
warm/network/teacher 三个中心的 pairwise accuracy 为 `0.757`，BC-vs-warm 判别为
`0.780`，选择 cost `21.622 -> 15.271`，261/300 帧优于 warm。然而
teacher-vs-network 判别只有 `0.503`，几乎是随机；ensemble 选择 network/teacher
分别为 162/138 帧，平均 regret `3.195`。

因此结论分两层：第一版 Critic 已学会 T1 shortlist 内的粗排序和“是否优于 warm”，
但尚未学会 BC 输出与 teacher 之间连续区域的局部排序。BC center 是 shortlist 点之间的
新分布，直接对该 Critic 做梯度上升会暴露于外推误差，暂不准入 actor 更新。下一步先
生成不可变的局部 relabel sidecar：覆盖 warm、BC、teacher、BC--teacher 插值点以及
BC 邻域小扰动，使用 common DBM seeds 计算真实 proposal cost；重训后必须在独立 seed
上显著超过 `0.503` 的 teacher-vs-network 排序，才进行带 BC anchor/trust region 的
actor 微调。

训练产物：

```text
outputs/mppi_proposal/critic_t1_diverse_20260805_v1/
  critic_seed{0,1,2}.pt
  training_summary.json
```

### 2026-08-05 局部 relabel、Critic v2 与 actor 提取结果

为补足第一版 Critic 的连续动作覆盖，新增不可变 sidecar：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
  dbm_critic_local_diverse_20260805_v1/
```

它包含 120 episodes、2400 snapshots，每帧 11 个中心：warm、冻结 BC、T1 teacher、
warm--BC 中点、BC--teacher 的 25/50/75% 插值，以及 BC 周围两组 antithetic
`0.15 sigma` 扰动。selection seeds 为 21001/21002，audit seeds 为 21101/21102；
每 center/seed 64 candidates，总计 6,758,400 条 DBM candidate rollout。独立 validator
已通过全部 source/T1 SHA256、split、shape、边界、seed 隔离和 summary 检查。

3-seed Critic v2 位于 `outputs/mppi_proposal/critic_local_diverse_20260805_v2`。ensemble
在 audit test 上 pairwise `0.768`、top-1 `0.717`、teacher-vs-BC `0.917`；在旧 fresh
seeds 20001--20003 上 teacher-vs-BC `0.900`，三中心选择 cost `21.622 -> 12.555`。
这说明 Critic 作为**已评估候选的排序器**已经有效。

但直接冻结 Critic、对 actor center 求梯度的实验全部失败。三个 BC anchor 权重
0.25/1/4 在 validation seeds 22001/22002 上的 cost 分别为
`27.056/26.493/24.816`，均差于原 BC `19.510`；最坏相对 BC 退化超过 68。actor 虽然
获得更高 predicted advantage，却只在约 5--9% 状态更接近 teacher，属于明确的 Critic
外推利用。对应目录 `actor_critic_local_20260805_v1` 已标记 `rejected_all`，不得部署。

随后改用支持集内 reward-weighted regression（AWR/IQL 风格）：不对 Critic 求动作
梯度，只在 warm/BC/teacher/插值这 7 个真实 DBM 已评估中心上按 cost softmax 加权回归，
并保留 BC anchor。temperature 2/4/8 在 validation 均小幅优于 BC，按 validation mean
选择 temperature=2。全新 test seeds 22101--22103 上：

| 方法 | mean cost | 相对 warm gain | 相对 BC gain | 相对 BC 胜/负 |
|---|---:|---:|---:|---:|
| warm | 22.234 | 0 | -3.004 | - |
| 原 BC | 19.230 | 3.004 | 0 | - |
| AWR temperature=2 | 19.145 | 3.089 | +0.084 | 161/139 |
| teacher | 12.218 | 10.016 | +7.012 | 287/13 |

AWR 的均值改善在 validation/test 都复现，但幅度很小，且相对 BC 仍有 4 帧退化超过 5、
最坏 `-13.727`。因此 temperature=2 可保留为研究 checkpoint，尚不能替换默认 BC。
Critic 对 BC-vs-AWR 微小差异的 test 排序仅 `0.513`，也不能用作可靠在线 gate。

当前下一步应聚焦**尾部风险与局部判别**：把 AWR/BC 分歧较大和真实退化状态追加为
hard-negative relabel，训练 delta-cost/胜负分类头并做置信度校准；gate 必须先在独立
test seed 上稳定降低 tail，再考虑短闭环交互式 critic 或 TD3/SAC。不要继续增加直接
deterministic policy-gradient 步数。

### 2026-08-05 冻结状态的 16 维全秩局部 reward 补全

进一步检查发现，上述 11-center sidecar 虽然能训练候选排序器，但每状态相对冻结 BC 的
局部 center 差分矩阵秩通常只有 4；而 `[8,2]` center 有 16 个自由度。Critic 因而可以在
数据未约束的零空间产生任意动作梯度。这是“离散排序正确、连续 actor gradient 失败”的
直接数据原因之一。

本轮没有重新采集场景、状态或 teacher，而是复用完全相同的 120 episodes、2400
snapshots 和 90/15/15 episode split，新增不可变派生 sidecar：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
  dbm_critic_fullrank_diverse_20260805_v1/
```

每帧以冻结 BC `network_center_knots` 为中心，使用 16×16 Sylvester Hadamard 正交方向，
每个方向计算 `+0.15 sigma/-0.15 sigma` 两个中心，加上原中心共 33 个中心。每个中心用
selection seeds `23001--23004` 和完全不相交的 audit seeds `23101--23104`，每 seed 仍为
64 个 MPPI candidates。总计 `2400*33*8*64=40,550,400` 条 fixed-DBM candidate
rollouts。这里监督的是每个 outer center 经 MPPI soft weighting 后输出控制序列的真实
DBM cost；训练 reward 可直接取其负值，不使用 DBM 解析梯度。

生成器 `generate_dbm_fullrank_local_labels.py` 和独立 validator
`validate_dbm_fullrank_local_labels.py` 已完成。2400/2400 帧的局部矩阵秩均为 16；平均
center clipping 为 `0.423%`，281 帧发生过任一维 clipping，但 clipping 后仍全部满秩。
selection/audit mean center cost 的 MAE 为 `2.372`，局部正负方向差在绝对值均大于
`0.1` 时符号一致率为 `75.74%`；阈值提高到 `1/2/4 cost` 时为
`81.33%/84.65%/88.42%`。逐状态 16 维方向差向量的 selection/audit cosine 中位数
为 `0.626`，`93.58%` 状态 cosine 为正。

因此这批数据已经解决“每状态局部方向不满秩”的可辨识性问题，但没有消除 MPPI reward
本身的 Monte-Carlo 噪声。下一步重训 Critic 时应以同状态 paired cost difference 为主，
对 4 个 selection seeds 求均值并保留 seed 方差作为置信度；4 个 audit seeds 严禁参与
训练，只用于局部排序、方向 cosine 和真实 actor 小步复评。是否允许 actor gradient，必须
看新 Critic 在 audit 方向上的梯度一致性，而不能只看全局 MSE 或 stored-center top-1。

### 2026-08-05 全秩 Critic 训练与梯度准入结果

全秩 sidecar 已接入新的训练入口
`train_mppi_fullrank_local_critic.py`。训练目标完全来自 rollout reward：先以 base cost
减 center cost 构造相对 advantage，再对 33 个中心的标准化 offset 做最小二乘，得到
每状态 16 维局部斜率。训练只使用 selection seeds；checkpoint 选择也只看 selection
validation，audit seeds 仅在训练结束后用于资格检查。

实现了两种 3-seed 对照：

```text
outputs/mppi_proposal/critic_fullrank_generic_diverse_20260805_v1
outputs/mppi_proposal/critic_fullrank_diverse_20260805_v1
```

前者沿用通用 `Q(s,center)` MLP，并通过二阶反传让 base 处 autograd gradient 匹配局部
reward 斜率；后者使用新增 `TorchMPPILocalQuadraticCritic`，由状态 encoder 显式输出
16 维一阶项和一个局部曲率项，使 base 处梯度就是直接监督的输出。通用模型 best epochs
为 `19/18/17`，结构化模型为 `7/8/9`。

冻结 test 300 状态的 audit 结果：

| 方法 | gradient cosine 中位 | cosine>0 | cosine>0.5 | abs(真梯度)>1 的符号准确率 | stored direction pair | 预测选中心相对 base |
|---|---:|---:|---:|---:|---:|---:|
| selection label 对 audit（数据上限） | 0.617 | 92.67% | 63.67% | 72.08% | - | - |
| 通用 full-rank Critic | 0.166 | 63.67% | 19.00% | 55.51% | 55.45% | +0.003 |
| 结构化 local-quadratic Critic | 0.193 | 70.67% | 14.67% | 56.95% | 56.74% | -3.322 |

两者都明显低于 reward 数据自身的 repeatability ceiling；提高 gradient loss 约 7 倍、
从旧 Critic 或仅从 BC 初始化，以及完整加载 BC fusion 的消融均未改变结论。另一个关键
诊断是：同一状态 selection/audit gradient cosine 中位为约 `0.630`，但同 episode 相邻
保存帧的交叉 cosine 中位只有 `0.044`、正值比例 `54.25%`。局部方向对当前
state/reference/warm start 很敏感，现有 1800 个训练状态不足以让 state-only Critic
泛化出可靠的 16 维梯度。

因此本阶段**不生成 actor checkpoint、不进行 actor DBM 复评**；这是按准入条件停止，
不是缺少实现。原 BC 继续作为默认。下一步不应继续盲调 deterministic gradient loss：

1. 若坚持 state-only SAC/TD3，增加独立状态覆盖，并对每个训练状态增加 reward seeds
   来提高期望局部梯度的信噪比；Replay Buffer 必须包含 actor 实际访问的多样中心，而非
   只有固定 33 点；
2. 更直接的路线是两次 MPPI：第一次保留当前采样产生的 cost/error/轨迹响应摘要，策略
   网络以这些**当前帧局部反馈**为额外输入预测第二次 center，避免仅凭状态猜测快速变化
   的局部方向；
3. 无论哪条路线，仍用 Query rollout reward 重标注而非模型解析梯度，并保留独立 seed
   的 gradient cosine 与真实 cost 作为 gate。

### 2026-08-05 两轮反馈采样与 feedback-conditioned Critic

已按“两轮是定向补充采样，不再比较单轮效率”的标准完成正式 sidecar：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
  dbm_two_pass_feedback_diverse_20260805_v1/
```

它复用原 120 episodes、2400 frozen DBM states 和 90/15/15 episode split，没有新增
场景或状态。每个状态有 2 组 selection 和 2 组完全独立 audit 重复。每组第一轮从冻结
BC center 做 128 条 antithetic rollout，用完整加权轨迹 residual 拟合经验 response，
通过 ridge=0.1、damping=0.1、最大 1 sigma 的 Gauss--Newton 步得到 guided center；
第二轮在该中心附近用 `0.10 sigma` MPPI noise，并在 16x16 Hadamard 的正负
`0.15 sigma` 全秩 outer center bank 上各评估 64 candidates。selection seed pairs 为
`(24001,24101)/(24002,24102)`，audit 为
`(24201,24301)/(24202,24302)`；总 candidate rollout 数为
`2400*4*(128+33*64)=21,504,000`。

每组保存第一轮 knots/cost、guided center、33 个第二轮中心及 weighted-output cost，
另保存 74 维 feedback：16 维 guided step、16 维经验 cost gradient、16 维经验 Hessian
对角、16 维 first-pass soft-weight shift 和 10 个 cost/ESS/clipping/fit 标量。整个过程只
使用 forward rollout，不使用 DBM 解析梯度，因此可沿用到 Query PyTorch/ONNX 后端。
生成器与 validator 分别为 `generate_dbm_two_pass_feedback_labels.py` 和
`validate_dbm_two_pass_feedback_labels.py`；2400/2400 source/parent hash、seed 隔离、
shape、有限值和局部 rank=16 全部通过，正式数据约 123 MB。

第一轮 weighted-output 到 guided-center 第二轮 weighted-output 的总体改善比例为
81.25%，但存在重尾：全数据均值为 `15.865 -> 18.818`。独立 test/audit 上改善比例
85.67%，均值为 `16.835 -> 16.522`；delta cost 中位 `-4.774`，但 P95 为
`+26.278`、最坏 `+529.108`。因此 guided update 有效但必须保留 trust/risk gate，不能
仅凭平均改善直接替换默认控制器。

新的 `TorchMPPIFeedbackQuadraticCritic` 以原状态、历史、参考、guided center 和 74 维
反馈为输入，显式输出 16 维局部 Q gradient 和一个曲率；训练入口为
`train_mppi_two_pass_feedback_critic.py`。只用 selection/train 优化和
selection/validation 早停，3 seed best epochs 为 53/53/40，正式输出：

```text
outputs/mppi_proposal/critic_two_pass_feedback_20260805_v1/
```

冻结 test/audit 结果如下：

| 方法 | gradient cosine 中位 | cosine>0 | cosine>0.5 | gradient correlation |
|---|---:|---:|---:|---:|
| selection/audit state-mean 标签上限 | 0.902 | 89.67% | 80.67% | 0.890 |
| 仅反馈的 linear ridge | 0.399 | 67.83% | 41.00% | 0.434 |
| 3-seed feedback Critic | **0.615** | **73.67%** | **56.50%** | **0.533** |
| 旧 state-only structured Critic | 0.193 | 70.67% | 14.67% | - |

因此结论是：两轮反馈补样已经把 Critic 的局部梯度从“基本不可用”推进到“明显可学”，
并显著超过 state-only 与线性基线。但它尚未通过 actor/center 更新准入。直接在 33 个
outer centers 中取预测 top-1，audit test cost 为 `16.522 -> 16.874`，平均退化
`0.352`，148/432 胜负。只用 validation 选择高阈值 gate 后，test 仅选择 70/600 个
context，平均改善 `3.725`，但仍有 26 个失败、最坏退化 `67.578`；不能以均值掩盖尾部。

当前原 BC 仍为默认，不生成 actor。下一步应对第二轮 Q 使用保守更新：增加相同
first-pass feedback 下的独立 second-pass reward seeds，训练 mean+tail/win-risk head；
用 ensemble disagreement 和预测下分位数决定是否移动，并在更小 trust step 上做真实
DBM/Query 复评。只有独立 audit 的实际 cost 与尾部同时通过，才接 actor 或 SAC/TD3。

### 2026-08-05 多 seed 小步长 risk replay 与安全步长 Critic

上述下一步已经完成。新增不可变 sidecar：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
  dbm_two_pass_risk_replay_diverse_20260805_v1/
```

它不新增状态，也不重跑第一轮；直接复用 2400 states 的 2 selection + 2 audit
first-pass feedback。对每个 feedback，冻结上一版 3-seed Critic 的梯度方向，在 guided
anchor 沿该方向生成 `±0.03/0.06/0.10/0.15 sigma` 和原点共 9 个 centers。每个 center
用 8 个 selection reward seeds `26001--26008` 和 8 个完全独立 audit seeds
`26101--26108`，每 seed 64 MPPI candidates、second-pass noise 为原 sigma 的 0.10。
新增 candidate rollout 数为
`2400*4*9*8*64=44,236,800`，数据约 57 MB。生成器与 validator 为：

```text
scripts/model_verify/generate_dbm_two_pass_risk_replay_labels.py
scripts/model_verify/validate_dbm_two_pass_risk_replay_labels.py
```

2400/2400 的 source/parent/critic checkpoint hash、first-feedback hash、seed 隔离、
center 重构、clipping、paired advantage mean/std/P10/win 和有限值均验证通过。仍只使用
forward rollout reward，不使用 DBM 解析梯度。

全量 selection/audit 显示方向本身是有效的，但步长必须很小：

| 步长 | selection mean/median advantage | audit mean/median advantage | audit 上下文胜率 |
|---|---:|---:|---:|
| +0.03 sigma | 1.835 / 0.137 | 1.827 / 0.107 | 72.23% |
| +0.06 sigma | 3.224 / 0.211 | 3.189 / 0.148 | 64.21% |
| +0.10 sigma | 4.500 / 0.215 | 4.375 / 0.095 | 54.92% |
| +0.15 sigma | 5.218 / 0.062 | 4.671 / -0.182 | 45.33% |

较大步长的 mean 被少量巨大收益抬高，而 median、胜率和 seed tail 持续恶化；所有负
方向均显著为负。这直接验证了上一轮 Critic “方向大致正确，但 argmax 步长过大”的
判断。冻结 test/audit 若始终走 `+0.03 sigma`，平均 advantage `1.811`、上下文
397/203 胜负、seed P05/P10 为 `-0.629/-0.248`，最坏 `-30.525`；固定小步仍不是安全
gate，但比大步更新稳定得多。

新增 `TorchMPPIFeedbackStepRiskCritic`，输入 74 维 feedback、原 Critic gradient
mean/std 和候选 signed radius/linear advantage，输出 mean advantage、P10 advantage、
per-seed win probability；v3 另加“8 seeds 最坏 advantage >= -1”的 safety head。
训练入口为 `train_mppi_two_pass_step_risk_critic.py`，正式输出：

```text
outputs/mppi_proposal/critic_two_pass_step_risk_20260805_v2
outputs/mppi_proposal/critic_two_pass_step_risk_20260805_v3
```

v3 在 test/audit 的 mean/P10 correlation 为 `0.609/0.622`，win/safety correlation 为
`0.577/0.559`。只在 selection/validation 选 gate，并强制部署候选只能是
`+0.03 sigma`：

| gate | test 移动数 | 整体 mean advantage | 上下文胜/负 | moved seed win | moved 最坏 |
|---|---:|---:|---:|---:|---:|
| v2 严格 P10/win/disagreement | 47/600 | +0.927 | 41/6 | 81.65% | -4.853 |
| v3 加 safety head | 180/600 | +1.135 | 134/46 | 69.93% | -4.948 |

两者都比无门控 risk-neutral argmax 好；后者会选择大步长，test mean 虽为正但中位为
`-0.337`、248/352 胜负、最坏 `-213.450`。因此补样和风险 Critic 已证明有效，但仍不
生成 actor、不接 SAC/TD3。若优先安全，保留 v2 严格 gate 作为研究基线；下一轮直接把
其 6 个 test loss 及 validation hard negatives 作为 replay 中心，增加同 feedback 的
reward seeds并训练 catastrophic-loss classifier。只有独立新 seeds/短 DBM 闭环同时
把最坏尾部压到准入阈值内，才扩大 gate 覆盖率。

### 2026-08-05 单步 Actor--Critic 对 teacher 的直接验证

当前任务进一步收缩为固定状态的单步 contextual Actor--Critic，不引入 `next_state`、
Bellman target 或闭环环境。DBM second-pass rollout 直接提供 reward。为避免再次让
连续 Critic 在未覆盖动作上外推，第一版动作被限制为沿 feedback Critic 方向选择
`0/0.03/0.06/0.10/0.15 sigma` 五个半径；risk-replay 已对每个状态、每个动作保存 8 个
selection 和 8 个 audit DBM reward seeds，因此该离散动作空间是完全覆盖的。

新增网络与入口：

```text
car_foundation/car_foundation/mppi_proposal_policy.py
  TorchMPPIFeedbackDiscreteStepCritic
  TorchMPPIFeedbackDiscreteStepActor
  TorchMPPIFeedbackDiscreteStateNetwork

scripts/model_verify/train_mppi_single_step_actor_critic.py
scripts/model_verify/train_mppi_single_step_state_actor_critic.py
scripts/model_verify/evaluate_mppi_single_step_actor_teacher.py
```

首版 feedback-only Actor 在 audit/test 相对 guided center 的 mean advantage 为 `4.497`。
随后加入完整 state/history/reference、first-pass feedback、Critic gradient mean/std，并让
Critic 同时拟合 mean 与 P10 reward；validation 选择 `tail_mix=0.25`。风险版 audit/test
mean advantage 提升为 `4.831`，P10 从 `-1.561` 改善到 `-0.557`。正式 checkpoint：

```text
outputs/mppi_proposal/single_step_state_actor_critic_20260805_v2/
```

最终比较不复用历史 teacher/audit cost，而是把 guided、固定 `+0.10 sigma`、Actor 和
原 T1 teacher center 放在同一 300 test states、2 个独立 first-pass audit repeats、全新
seeds `27101--27108`、每 center/seed 64 candidates、相同 `0.10 sigma` second-pass
noise 和 common random numbers 下重新 DBM rollout。结果位于：

```text
outputs/mppi_proposal/single_step_actor_teacher_eval_20260805_v2/
```

mean cost 为 guided `16.408`、固定 `+0.10` `12.934`、Actor `12.241`、T1 teacher
`11.889`。Actor 在 `464/600` contexts 和 `78.17%` seed pairs 上低于 teacher，配对
context median gain 为 `+2.003`；但 mean gain 为 `-0.352`，P10 为 `-5.049`，最坏
context 为 `-230.403`。因此 Actor 已超过 teacher 的多数典型状态，但**尚未在总体均值
和尾部超过 teacher**，不能宣称单步策略通过。

失败高度集中在高速度：1.2/1.6/2.0 m/s 的 Actor 相对 teacher mean gain 分别为
`+2.128/+3.595/+3.726`，2.4/2.8 m/s 则为 `-3.210/-8.001`；recovery 子集为
`-1.692`。这表明剩余问题不只是风险头训练，而是当前“单一反馈方向 + 标量半径”的
动作基限制了高速度/恢复状态。下一步仍保持单步问题，应扩充 actor-visited replay 的
方向覆盖，使 Actor 输出受 trust 限制的完整 16 维 residual，逐轮用 DBM 真 reward
回填 Critic；在同一 fresh-seed teacher gate 通过前不要进入多步 SAC 或闭环训练。

### 2026-08-05 多方向 replay、学习 Actor 与单 seed probe 上限

已按上述计划增加多方向 feedback-derived replay，但仍保持固定状态和单步 reward。每个
first-pass context 从运行时可得信息构造 8 类方向：feedback Critic、first-pass best、
soft shift、负经验 cost gradient、Hessian 预条件 gradient，以及三种两两组合；每类
使用 `0.03/0.06/0.10/0.15 sigma`，加 guided anchor 共 33 个中心。teacher 不参与方向
生成。正式不可变 sidecar：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
  dbm_two_pass_multidirection_replay_diverse_20260805_v1
```

它覆盖原 2400 states、每 state 的 2 selection + 2 audit first-pass contexts，中心使用
4 个 selection reward seeds `28001--28004` 和 4 个 audit seeds `28101--28104`，每
center/seed 64 candidates，共 `81,100,800` 条新 DBM candidate rollout。生成器与独立
validator 为 `generate_dbm_two_pass_multidirection_replay.py` 和
`validate_dbm_two_pass_multidirection_replay.py`；2400/2400 hash、seed、feedback、方向、
center、clipping 和 paired reward 重构通过，最大重构误差为 0。

33 动作 mean/P10 Critic 与平坦 categorical Actor 位于：

```text
outputs/mppi_proposal/multidirection_actor_critic_20260805_v1
```

其 audit/test 相对 guided mean advantage 为 `4.230`，动作库同 seed oracle 为 `5.822`。
在全新 seeds `28201--28208` 下，learned Actor cost 为 `12.669`，T1 teacher 为
`11.890`，比上一版单方向风险 Actor 的 `12.241` 更差。直接复制训练 reward oracle 的
诊断 Actor 也只有 `3.901` audit advantage，说明把 33 个方向/步长当平坦监督类别会放大
跨 first-pass/reward repeat 的标签切换，当前网络没有吃到动作库上限。

另一方面，同一 context 用一个真实 DBM probe seed 评估 33 个中心，再选最低 probe
cost 的中心，在另外三个 stored seeds 上仍有 `+5.911` mean advantage、最坏仅
`-4.038`。正式 fresh gate 因此使用 audit probe seed `28101` 选择中心，再用完全未见的
`28301--28308` 与 T1 teacher 做 common-random-number 复评：

```text
outputs/mppi_proposal/multidirection_probe_teacher_eval_20260805_v2
```

guided/fixed `critic+0.10`/probe-selected/T1 teacher mean cost 为
`17.271/13.495/11.242/11.898`。probe policy 相对 teacher 平均降低 `0.656`，赢
`490/600` contexts，首次超过 teacher 的 mean；相对 guided 平均降低 `6.029`，最坏仅
退化 `3.554`。但它需要额外 `33*64=2112` candidate probe rollouts/context，而且 teacher
尾部仍更好：probe policy P95/maximum 为 `35.094/249.362`，teacher 为
`29.070/64.313`。因此只能得出“多方向局部反馈机制的 mean 上限超过 teacher”，不能
得出“轻量策略网络已经超过 teacher”或“安全尾部已经通过”。

下一步仍不需要 next-state。优先把 33 维 probe cost/uncertainty 作为第二层策略输入，
采用分层的“先方向、后半径”结构，并消融每 center 16/32/64 probe candidates，目标是
在更低 probe 预算下保留 mean 超越；同时需要增加能离开 guided 局部邻域的高速度全局
方向，否则无法追平 teacher 的 worst/P95。无 probe learned Actor 和原 BC 仍不得替换。

### 2026-08-05 同状态串行 probe SAC、固定顺序对照与准入结论

已实现“同一个物理状态内多次推理、跨采集批次更新 Critic”的内部搜索 MDP。它不是
车辆 `next_state` 的多步控制：history/state/reference/guided center 和 first-pass
feedback 在一次内部 episode 中保持不变，只有已探测 cost、mask、best-so-far 和剩余
预算变化。reset 时先探测 guided anchor；之后 Actor 每轮从 32 个未探测中心中选择一个，
真实 fixed-DBM proposal-output rollout 返回 cost。Actor 和 twin Critic 在部署搜索中
冻结，actor-visited transition 写入 Replay Buffer 后才批量更新网络，并使用 soft target
Critic。动作仍限制在已完整覆盖的 33-center bank，不使用连续 Q 外推或 DBM 解析梯度。

新增实现：

```text
car_foundation/car_foundation/mppi_proposal_policy.py
  TorchMPPISequentialProbeActorCritic

scripts/model_verify/train_mppi_sequential_probe_sac.py
scripts/model_verify/evaluate_mppi_multidirection_actor_teacher.py
scripts/model_verify/plot_mppi_sequential_probe.py
```

网络有 593,315 个参数；输入保留完整 state/history/reference、guided knots、74 维
first-pass feedback、32 维 feedback-Critic gradient mean/std，并新增 33 维 probe value、
33 维 observed mask 和 remaining-budget。输出为 categorical Actor logits 和两组 33 维
Q。第一版即时奖励为同 probe seed 上 best cost 的下降；第二版改为中间奖励 0、终止时
用未参与选择的另外三个 selection seeds 评价最终 best center，以减少单 seed
winner's curse。首选研究 checkpoint 是：

```text
outputs/mppi_proposal/sequential_probe_sac_20260805_v2/
  sequential_probe_sac.pt
  training_summary.json
```

该 checkpoint 只用 selection/train 交互、selection/validation 选第 31 轮，冻结后才读取
audit/test。stored audit 中，每个 probe 仍为 64 candidates，独立 seed 相对 guided 的
平均 advantage 为：

| probe 数 | candidate 预算 | mean advantage |
|---:|---:|---:|
| 1 | 64 | 0.000 |
| 2 | 128 | 4.781 |
| 3 | 192 | 5.337 |
| 4 | 256 | 5.379 |
| 8 | 512 | 4.661 |
| 33 | 2112 | 5.533 |

4-probe 保留了 full 33-probe mean gain 的 `97.2%`，但 8-probe 反而下降；当前网络只按
4-probe horizon 训练，禁止把同一 checkpoint 外推成任意预算策略。第一版动作序列几乎
退化成固定优先级。用其最常见方向构造固定序列
`anchor -> critic+preconditioned 0.15 -> critic 0.15 -> negative-gradient 0.15`，audit
mean advantage 为 `5.463`，仍略高于第二版 learned sequential 的 `5.379`。

最终 fresh gate 使用 probe seed `28401`，随后用完全不相交的 common evaluation seeds
`28411--28418` 在 300 test snapshots × 2 first-pass contexts 上重新 rollout。结果位于：

```text
outputs/mppi_proposal/sequential_probe_teacher_eval_20260805_v5/
  summary.json
  per_context.csv
  fresh_eval.npz
  sequential_probe_result.png
```

| 方法 | mean cost | P95 context | maximum context |
|---|---:|---:|---:|
| guided | 16.737 | 52.524 | 495.968 |
| critic +0.10 sigma | 13.122 | 42.928 | 339.466 |
| 固定优先级 4-probe | **11.138** | 35.513 | 244.189 |
| terminal-reward sequential 4-probe | 11.246 | 35.601 | 272.541 |
| T1 teacher | 11.913 | **29.134** | **64.274** |

sequential 相对 T1 teacher 平均降低 `0.667` cost，context 胜负 `484/116`，但相对固定
4-probe 平均退化 `0.108`，胜/负/相同为 `157/122/321`。第二版 fresh rollout 已出现
47 种 probe 序列，说明 Actor 确实使用状态/反馈改变后续动作；这些自适应变化目前没有
超过固定顺序。其核心结论分两层：

1. **通过：**4 次真实 probe 可用约 12% 的 full-bank probe candidate 预算保留几乎全部
   mean gain，并在 fresh mean 上超过 T1 teacher；多次推理和 best-so-far 机制成立。
2. **未通过：**Critic/Actor 的反馈条件化顺序尚未超过固定 probe 优先级；teacher 的
   P95/maximum 也仍显著更好。不得把当前 checkpoint 宣称为自适应 RL 胜出或安全准入。

下一步若继续，不再单纯增加 SAC update 数：应以当前固定 4-probe 作为强基线，采集更多
on-policy probe outcome 和独立 reward seeds，训练 terminal return 的 distributional/
quantile Critic；同时加入高速度/恢复状态所需的非局部方向。必须用新 probe/evaluation
seeds 同时超过固定顺序的 mean 与 tail，才说明第二、三轮反馈推理真正产生额外价值。

### 2026-08-06 后续执行入口

后续不再直接按本文件历史段落中的“下一步”文字启动实验。统一按
[串行 probe 后续执行与偏离检查计划](mppi_sequential_probe_execution_plan_20260806.md)
执行。当前只允许 `S0 -> S1`：冻结复现基线，然后在完全相同的 DBM、状态、reference、
动作边界和 cost 下求 `J*_100`、`J*_16`、`J*_center`、`J*_bank` 与 `J_policy`，定位
参数化、MPPI 采样、bank 覆盖或策略选择中的主 gap。每一步必须记录冻结项、预算、seed
隔离、期望、实际结果、PASS/FAIL/INVALID 和 `D0--D4` 偏离等级；GT 未稳定或 gate 未
通过不得进入 on-policy 扩充、Query、ONNX、ROS 或长闭环。

GT-first pilot 已执行：五个覆盖 1.2--2.8 m/s 和五类场景的 train snapshot 上，refine 后
mean `warm/J*_16/J*_100 = 22.023/5.821/5.558`，16-knot 参数化 gap 仅 0.263；多初值
spread、末段改善和独立复算均通过 pilot gate。按正式 `0.10 sigma`、
zero/extra/antithetic candidate design 重跑后，五帧 mean
`J*_center/current-bank selection→audit/audit-clairvoyant = 5.824/7.555/7.527`，bank
覆盖 gap 为 1.732，即使 clairvoyant 仍有 1.703；seed 误选只占 0.029。`v1` 误用完整
sigma，`v2` 误用 iid noise，均按 D2 作废；正式结果为 S1-B `v3`。S1-C `v2` 得到
feedback/fixed/bank-clairvoyant/unrestricted mean `7.568/7.603/7.527/5.824`：feedback
的 bank 内选择 gap 只有 0.041，而 bank coverage gap 为 1.703。

S2-B 已在不改网络结构、保持 33 slots 和 forward-only 约束下完成单轮动态 generator
pilot。正式 `v3` 中 residual-response (`B6`) 在五帧全部改善当前 bank，clairvoyant mean
`7.515→7.073`，但只回收约 26.2% aggregate coverage gap，未达到预设 50% gate
`≤6.670`。更大 response step 与独立 seeds 的 `v4` 最好为 `7.524→7.042`，约回收
28.3%，仍失败；说明限制不只是步长，而是单轮局部方向对非线性轨迹响应的覆盖。完整
seed、逐帧结果、图和偏离记录见执行计划第 6 节。当前不得生成正式 replay 或训练
Actor/Critic；下一步只做固定预算的多轮 forward-response pilot。

### 2026-08-06 连续 SAC 路线切换与首轮 actor-visited pilot

用户明确将最终策略从 33-slot categorical Actor 改为连续 16-D sampling-center residual
Actor。旧离散 SAC、33-center bank、T1 和 residual-response 不删除，但只作为 BC warm
start、初期 exploration/replay 和固定对照，不再限制 Actor 可输出的方向。权威步骤已在
`mppi_sequential_probe_execution_plan_20260806.md` 改为 S2-C continuous contract、
S3-A 单步 actor-visited SAC、S3-B 四轮 continuous SAC。

新增 `TorchMPPIContinuousCenterActor/Critic`：Actor 为 squashed Gaussian，输出 `[8,2]`
normalized residual，默认可达 anchor 周围 `±2 source sigma`；两个独立 scalar Q 接收
连续 action。T1 BC v1 在五帧有 3 帧退化，因此改为克隆当前 first-pass feedback context
下的旧 bank oracle。正式 bootstrap v2 的 test clone-action RMSE 为 `0.0234`，五帧 fresh
audit 的 anchor/Actor/bank-selected mean 为 `10.819/7.754/7.642`，Actor 五帧全部改善
anchor；这只通过初始化 gate，不是 SAC 结果。

S3-A fixed-five pilot 已让 Actor 新输出经过真实 forward-only fixed-DBM reward 后进入 replay。
一组独立 seeds 下，320 个 actor-visited 连续中心经 selection→audit 选择后 mean 为
`7.251`，优于初始 Actor `7.772` 和旧 bank clairvoyant `7.584`，证明连续探索能离开旧
bank 并找到 headroom；但 SAC checkpoint best iteration 仍为 0，Actor 未提取该收益。
另一组独立 seeds 的相同流程得到 replay-winner audit `14.626`，暴露 `2 reward
seeds/action` 在大量连续候选下的 winner's curse。降低 Actor update ratio/learning rate/
entropy 仍未改变准入结论。

当前状态为 `S2-C PASS / S3-A FAIL(D1)`。下一步保持连续 SAC，不回退离散动作：先减少
每状态并行新动作数，把预算用于至少 4 个 selection reward repeats，并用完全独立 audit
验证 reward correlation、方向一致率和 replay-best 复现性。该 replay gate 通过后才恢复
Actor 更新；单步 continuous Actor 未真实超过 clone/fixed bank 前不得进入四轮、Query、
ONNX、ROS 或长闭环。

### 2026-08-06 最新覆盖：Direct Actor 唯一目标与固定候选库

上一段末尾“增加 reward repeats”已经被后续设计决定覆盖。连续 Actor 的主训练标签不再
经过随机 MPPI candidates。当前权威定义为：

```text
state/context --Actor--> c[8,2]
c --固定 align_corners=True 线性插值--> A[50,2]
(state, A) --一次固定 DBM forward--> J_direct
r_direct = J_direct(anchor) - J_direct(c)
```

因此高斯分布只负责产生不同探索动作；同一状态下已经确定的 `c` 必须对应唯一 cost，
没有 collection/selection/audit reward seed。Critic 第一阶段学习
`Q_direct(state,c)`。旧 T1、33-center bank、residual-response 和 actor-visited centers 仍可
作为 BC/探索/Replay Buffer 数据，但必须重新执行 direct rollout，不能沿用原
MPPI weighted-output reward。

固定多候选仍有价值，但角色改成第二层“这个 direct action 是否也是一个好 MPPI center”
评价。实现冻结为 `fixed-hadamard-64-v1`：candidate 0 为原 center，candidate 1 为固定
extra，其余为 31 对 antithetic offsets；16×16 Sylvester Hadamard 保证 8×2 knot 空间满秩，
半径为 `0.10/0.30 source sigma`。它完全不读取随机 seed，固定 bank contract hash 为：

```text
0fd540206ec988f565481d25f9cbd0b4051f7847e154565d9eac4e7445b6a492
```

`car_dynamics/.../mppi.py` 已新增 `sampling_mode=fixed_hadamard_64`；该模式要求
`num_samples=64`。ROS launch 可传：

```bash
ros2 launch car_ros2 car_sim.launch.py \
  mppi_num_samples:=64 \
  mppi_sampling_mode:=fixed_hadamard_64
```

默认仍为 `gaussian`，所以旧实验与普通 launch 行为不变。PyTorch/ONNX/DBM 只改变 rollout
backend，候选生成器由同一个 Torch MPPI controller 执行，三者共享该固定 bank；JAX 不
维护。

新的五帧机制入口为：

```text
scripts/model_verify/fine_tune_mppi_direct_center_sac_pilot.py
outputs/mppi_proposal/continuous_center_direct_fixed5_pilot_20260806_v1
```

协议验证结果：64 候选 shape 正确、16 维方向 rank=16，同一批中心重复 direct cost 最大
误差为 `0.0`。完整 pilot 收集 320 个 Actor 新中心，初始化/final Actor direct mean 均为
`9.782`，best iteration=0；actor-visited best mean 为 `18.191`，明显差于初始化，说明当前
16 维宽高斯探索没有覆盖好中心附近，而不是 reward seed 不足。固定邻域只是辅助评价，
其 final-center deterministic weighted-output mean 为 `7.656`，不能记成 Direct Actor 的
训练成绩。

旧 `continuous_center_sac_fixed5_pilot_20260806_v{2,3}` 仍保留用于说明 stochastic wrapper
的 winner's curse，但对当前 Actor 目标归因标为 `INVALID/D2`。下一步不增加 reward
repeats，而是：

1. 将旧 bank、BC/T1、插值和小半径满秩局部中心按 direct cost 统一重标；
2. 降低 Actor 初始 log-std，混合局部 antithetic/结构化方向，确保 replay 同时覆盖初始化
   附近的改善与退化样本；
3. 从头训练 `Q_direct`，用 episode-heldout ranking、有限差分方向一致性和真实 direct
   rollout 复评 Actor；
4. Direct Actor 通过后，才加入固定 64 候选的辅助 `Q_neighborhood/risk`，最后比较
   `direct Actor`、`Actor+fixed MPPI`、`warm+fixed MPPI` 与 `J*_16`；
5. 在 DBM 阶段通过前，不进入四轮、Query、ONNX qualification、ROS learned-policy 或
   长闭环。

### 2026-08-06 最新覆盖：确定性 Actor 与衰减结构化探索

用户确认 Actor 应直接输出唯一 center，也接受训练期保留独立、逐步衰减的探索度。实现
因此不再使用标准 SAC 的 squashed-Gaussian 方差头：

```text
Actor(state) = tanh(mu(state)) = 唯一 normalized center action
center = clip(anchor + action * 2 * source_sigma)
```

新增 `TorchMPPIDeterministicCenterActor`，保留原 continuous encoder 和 mean head，删除
`log_std_head`。从 bootstrap v2 的 stochastic Actor 转移后，确定性 action/center 逐元素
完全一致，参数量由 stochastic Actor 降到 `511,120`。部署、validation 和 direct reward
始终只调用这一个输出。

探索被移到 Actor 外部，仅训练时执行。每轮以当前 Actor center 为基点，从 16×16
Hadamard 满秩方向循环取4个方向并生成正负 pair；Actor center 本身也进入 Replay Buffer。
探索半径以 source sigma 为单位独立衰减：

```text
c_explore = clip(c_actor ± radius(t) * hadamard_direction * source_sigma)
```

它不是 Actor 输出的一部分，不进入部署图，也不改变给定 center 的唯一 direct cost。当前
入口仍是 `fine_tune_mppi_direct_center_sac_pilot.py`，但内部算法已经是单步 deterministic
Actor--Critic，不再是 SAC entropy policy。

两个固定五帧结果：

| 版本 | 探索/Actor 更新 | initial | final Actor | replay best | 结果 |
|---|---|---:|---:|---:|---|
| v2 | `0.30→0.05 sigma`，lr `3e-5`，每轮4次 | 9.782 | **8.866** | **7.404** | 第5轮最好，之后发散；3/5改善 |
| v3 | `0.15→0.03 sigma`，lr `1e-5`，每轮1次 | 9.782 | **8.919** | **7.562** | 24轮单调改善；3/5改善，另2帧仅退化0.103/0.031 |

旧高斯 v1 的 replay best 为 `18.191`，而结构化 v2/v3 为 `7.404/7.562`，直接证明此前
主要失败来自探索几何，而非没有更好动作或 direct reward 不可学习。v2 平均最好，但会在
第5轮后因 Critic exploitation 发散；v3 平均略差 `0.052`，训练稳定且回退更小，因此当前
首选 v3 作为机制 checkpoint。

当前状态更新为：`Direct Actor unique-output PASS`、`structured decaying exploration
PASS`、`fixed-five learning mechanism PASS`、`episode-heldout generalization PENDING`。
下一步把同样的 direct relabel 与探索协议扩到 episode-level train/validation，冻结 test；
不得把五个 train states 上的 `8.919` 当成泛化结果。必须验证 heldout ranking、局部方向、
速度/场景分组和 tail 后，才允许加入固定64候选 wrapper 或四轮搜索。

### 2026-08-06 episode-heldout 两轮 Direct Actor 结果

episode-level train/validation direct replay 已完成两轮。每轮覆盖 2,100 snapshots、4,200
feedback contexts、164 centers/context，即 688,800 个 deterministic DBM labels；test
`episode_105--119` 没有生成文件。两轮 full validator 的 center/cost 最大误差均为 0，
局部 Hadamard rank 最小为 16。

第一轮 raw Actor 在 epoch 5 后会利用 Critic 外推。新增
`calibrate_mppi_direct_actor_trust_step.py`，把 bootstrap 与 learned state_dict 做固定 alpha
插值，再用真实 validation DBM cost 选择；输出仍是一个普通 deterministic Actor。第一轮
保守 alpha=0.375 的 mean 为 25.974，原 bootstrap 为 27.445。

第二轮围绕该 Actor 重新采集 v2 replay。必须继承第一轮 twin Q，并让 Critic/Actor epoch 0
进入 checkpoint competition；随机重置 Q 会把 validation ranking regret 从 8.00 恶化到
24.33。为解决“候选排序可用、连续梯度不可用”，Critic 增加局部 antithetic pair 的有限
差分 slope loss。该监督只来自 `r_pos-r_neg` 和 action distance，不使用 DBM 梯度，Query
forward rollout 后仍可采用同一接口。

当前最佳 v4 + alpha=0.75 单网络 checkpoint 的 validation mean 为 25.518，相对第一轮
保守 Actor 改善 0.456；median gain 0.033、win 58.7%、P05 -2.457。从初始 27.445 累计
改善约 7.0%。但 worst gain 仍为 -109.109，所以只记
`two-round mechanism PASS / tail FAIL / test sealed`，不能向 Query、ONNX、ROS 或长闭环
推进。下一步先对 tail context 做分组和 conservative risk/fallback，不再用更多无约束
Actor updates 碰运气。

tail audit 显示坏例集中在 2.4--2.8 m/s 的 overspeed、heading recovery 和高 yaw-rate，
最坏为 `episode_101/step_000262/context 1`（143.683→252.793）。完美 DBM 下增加一个
确定性二中心 guard——分别 rollout 第一轮保守 center 与新 Actor center 后取较低 cost——
可将 mean 进一步降到 24.661，P05/worst gain 都变为 0，新 Actor 使用率 58.7%。这是
model-selection upper bound，不是 Actor 单网络成绩；入口为
`evaluate_mppi_direct_actor_guard.py`，Query 上必须先验证 pairwise cost ranking。

### 2026-08-06 validation numerical-oracle handoff

固定 DBM validation split 已完成 direct 数值最优对比。由于同一 snapshot 的两个 Actor
feedback context 共享相同 direct DBM objective，所以对 300 个独立 snapshot 各优化一次，
再与 600 个 Actor context 成对比较。优化采用 warm、zero、T0 best/soft、T1 teacher 和
3 个随机初值；Actor-compatible 8x2 knots 共 refinement 400 步，完整 50x2 action 共
refinement 500 步。`J16/J100` 必须称为 **best-found numerical oracle**，不能称为已证明的
全局最优。最后10步改善审计中，仅 J16 `3/300`、J100 `0/300` 的相对改善大于0.1%；独立
validator 的插值、cost、parent-regression 误差均为0，且每帧 J100 都严格低于 J16。test
episodes 保持封存。

600 个 validation context 的 mean cost 为：warm `29.517`、第一轮 Actor `25.974`、当前
Actor `25.518`、perfect-DBM two-center guard `24.661`、T1 teacher `12.762`、J16
`4.894`、J100 `4.697`。当前 Actor 比 teacher 高 `12.756`，成对胜率仅 `48.3%`；Actor
恢复 warm→J16 可改善量的 `16.2%`，teacher 恢复 `68.0%`。J16→J100 只改善 `0.198`，
而 Actor→J16 gap 为 `20.624`，所以当前瓶颈不是 knot 维度。

退化具有明显速度条件：Actor 在 1.2/1.6 m/s 的 mean `3.059/5.809` 优于 teacher 的
`4.798/6.187`；但在 2.0/2.4/2.8 m/s 的 `17.223/39.528/61.974` 明显差于 teacher 的
`11.482/17.137/24.207`。2.8 m/s 时 Actor 也差于 warm `48.595`。后续把高速 recovery/
tail 泛化作为主要 gate，不再把输出维数不足作为主假设。

```text
scripts/model_verify/generate_dbm_direct_gt_validation.py
scripts/model_verify/validate_dbm_direct_gt_validation.py
scripts/model_verify/compare_mppi_direct_actor_teacher_gt.py
outputs/mppi_proposal/dbm_direct_gt_validation_20260806_v2
outputs/mppi_proposal/direct_actor_teacher_gt_validation_20260806_v1
```

### 2026-08-07 J16 distillation handoff

Train-only J16 labels now cover 1800 snapshots from 90 episodes. After 250+150
uniform optimization steps, train mean warm/teacher/J16 is 27.918/11.758/4.830;
the independent 1800-frame replay has zero interpolation and cost error. Test remains
ungenerated and unread.

The old 2-sigma Actor box reaches validation J16 in only 37.5% of contexts. A
6-sigma box reaches 98.8%; projected-J16 validation cost is 4.895 versus J16 4.894.
Thus action support can be fixed without changing network width.

Plain train-only oracle distillation does not generalize. Three 300-epoch 6-sigma
Actors reach train cost 7.263--7.333 but validation cost 42.517--49.933, worse than
the current Actor 25.518 and teacher 12.762. A stratified internal episode-heldout
epoch selection also fails (best 6-sigma validation 48.580), so this is not a simple
early-stopping issue. These checkpoints are rejected.

J16 itself is not strongly multimodal by the current multi-start audit, but changes
rapidly between adjacent snapshots: 68.8% of validation transitions move over one
source sigma in knot RMS. Interpolating the best distilled Actor 75/90/95% toward
J16 yields validation cost 8.630/5.575/5.071, showing a roughly useful direction but
insufficient precision under the highly cost-sensitive high-speed dynamics.

Next expand independent episode coverage per speed/scenario stratum and replace plain
knot MSE with forward-only cost-sensitive verified policy improvement/local-curvature
supervision. Do not add more adjacent frames, deploy the 6-sigma checkpoint, resume
unconstrained Critic gradients, or open test/Query/ONNX/ROS gates yet.

### 2026-08-07 independent train coverage expansion

The requested state-distribution expansion is complete. The new immutable source is:

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/
  fixed_dbm_policy_train_expansion_20260807_v1
scripts/model_verify/fixed_dbm_policy_train_expansion_20260807_v1.json
```

It contains 270 train-only episodes and 1,350 snapshots; validation/test counts are
zero. Every one of the 30 speed/scenario strata receives nine new independent
episodes, so the combined training source has 12 episodes per stratum and 360
episodes total. Each new episode contributes only five fully observed, widely spaced
snapshots at steps 250/300/350/400/450. This deliberately increases independent
trajectory coverage while reducing within-trajectory redundancy relative to the old
20-frame, stride-12 source.

All 270 episodes passed the schema-v2 validator during capture and a second unified
resume validation. There are 1,350 snapshots, 451 continuous trace records per
episode, no observation noise, and one fixed DBM parameter set. Plan SHA256 is
`b5bf9026f1ad4f7b0797739d07f3b61d73aff5da02f3727e538c763f9d101d8f`.
Actual `vx` is 0.772--3.352 m/s; lateral error is -0.653--0.495 m; heading error is
-0.515--0.475 rad. Best-candidate cost median/P95/max is
18.482/50.231/207.428, and knot-clipping median/P95/max is
9.18%/25.00%/45.51%. Median best cost rises monotonically from 7.991 at 1.2 m/s to
31.056 at 2.8 m/s, so high-speed difficulty has not been diluted.

This completes the raw-state coverage step only. J16 and local forward-cost labels
for these 1,350 states are still pending. Keep the old formal validation and test
unchanged; next generate train-only J16 plus antithetic/local-curvature supervision,
combine with the old 1,800 train labels, and rerun episode-heldout selection before
opening the sealed validation. Rejected 6-sigma distilled checkpoints remain rejected.

### 2026-08-07 expanded J16 supervision and cost-sensitive Actor result

The pending supervision step above is now complete. New-state J16 results are:

```text
outputs/mppi_proposal/dbm_direct_gt_train_expansion_20260807_v1
outputs/mppi_proposal/dbm_direct_gt_train_expansion_20260807_v2
```

The first pass uses 250 steps and the second resumes every start for 150 steps.
Across 1,350 train-only snapshots, warm/J16 mean is `28.120/4.873`; refinement lowers
the 250-step mean from `4.893` by another `0.0203`. Both full independent validators
report zero source/interpolation/cost replay error. Maximum per-start parent regression
is only `7.7e-05` from float32 best-start ties. T1 is deliberately unavailable for this
new source and is stored as `null`, not imputed.

Forward-only local supervision is stored separately for old and new train states:

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
  dbm_j16_local_curvature_train_diverse_20260807_v1
  dbm_j16_local_curvature_train_expansion_20260807_v1
```

Every state has J16 plus two radii (`0.05/0.15 sigma`) times 16 fixed Hadamard
antithetic directions: 65 centers total. Old/new validators replay 1,800/1,350 states
with zero cost and derived slope/curvature error; local rank is 16 for every state.
New-state clipping is 1.52%, so only 70.2% of direction/radius pairs remain exactly
symmetric; the sidecar stores a mask and training excludes clipped pairs from standard
central differences. No analytic DBM gradient is saved or consumed.

The current Actor input contract was also rebuilt for the new source without running
the obsolete large second-pass replay:

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
  mppi_first_pass_actor_context_train_expansion_20260807_v1
scripts/model_verify/generate_mppi_first_pass_actor_contexts.py
```

It uses the frozen BC center, first-pass seeds `24001/24002`, 128 antithetic forward
rollouts, and the frozen three-model feedback-Critic ensemble. On five old-source
regression states, feedback, guided anchor, gradient mean, and gradient std match the
historical formal sidecars with maximum error `0.0`. The new source has 2,700 contexts;
first-pass base/best candidate mean cost is `24.730/16.649`.

The combined trainer is:

```text
scripts/model_verify/train_mppi_j16_cost_sensitive_distillation.py
outputs/mppi_proposal/j16_cost_sensitive_expansion_20260807_v2
scripts/model_verify/analyze_mppi_j16_cost_sensitive_actor.py
```

It combines old 1,800 and new 1,350 snapshots, balances loss by episode, uses 300 of
360 train episodes for epoch selection, runs every seed for all 300 epochs, then
refits all 360 episodes for the selected `228/208/217` epochs. The 6-sigma target is
weighted by forward-cost curvature in the Hadamard basis. Formal validation is loaded
only after all three Actors freeze; test remains unopened.

Full-step validation means are `31.432/28.675/33.693`; best seed 1 improves over the
old plain-J16 range `42.517--49.933` but still loses to current Actor `25.518` and T1
`12.762`. Seed 1 wins 57.8% of contexts and has positive median gain, yet its mean
gain is `-3.156`, P05 `-74.819`, and worst `-369.475`. By speed, it improves current
Actor by `0.476/0.592/1.538` at `1.2/1.6/2.0 m/s`, then regresses by `12.980/5.408`
at `2.4/2.8 m/s`.

A deterministic output-center trust scan confirms that the learned direction is useful:
`alpha=0.40` lowers validation mean from `25.518` to `21.088` and wins 72.0%, but P05
gain is `-17.464` and worst is `-81.653`. This is a two-Actor validation-calibrated
diagnostic, not a qualified single Actor. Record this stage as
`coverage/direction improvement PASS; full-step Actor FAIL; tail gate FAIL`. Do not
open Query/ONNX/ROS/closed-loop/test. The next problem is state-dependent step size or
verified fallback, especially at 2.4--2.8 m/s, rather than simply adding more epochs,
adjacent frames, or output dimensions.

### 2026-08-07 speed-conditioned trust diagnosis and deferred speed scope

A grouped validation line scan from the current Actor to the new cost-sensitive Actor
selects alpha `0.75/0.60/0.55/0.40/0.35` at reference speeds
`1.2/1.6/2.0/2.4/2.8 m/s`. The corresponding best grouped costs are
`2.520/4.692/13.583/34.552/49.474`; using the full new Actor gives
`2.583/5.216/15.684/52.508/67.381`. New-Actor-to-J16 knot RMS still rises with speed
(`0.055/0.077/0.126/0.176/0.212`), although it is substantially below the current
Actor's `0.199/0.213/0.277/0.381/0.436`. This is evidence for a useful learned
direction combined with speed-dependent sensitivity and an oversized fixed update.

Speed alone is not a sufficient gate. Within both 2.4 and 2.8 m/s groups, per-context
best alpha spans the full `[0,1]` scan; 27.5% and 20.8% of contexts respectively prefer
alpha zero. Resume with a context-conditioned trust step or verified two-center
fallback using speed, overspeed, heading error, yaw rate, scenario and first-pass
feedback. Do not attribute the failure to 16-knot capacity or multimodality.

The possible expansion to `100 km/h` (`27.78 m/s`) is recorded but explicitly deferred.
Keep the current `0.21 m` small-car DBM and `1.2--2.8 m/s` datasets unchanged. Before
that scope is resumed, freeze a separate vehicle-scale, dynamics, track-curvature,
action-bound and horizon contract; never mix such samples into the current split.

### 2026-08-07 FR-TRPI design handoff

The authoritative next Actor step is now the forward-rollout deterministic TRPO-like
design in `mppi_direct_actor_trpo_like_design_20260807.md`. Keep the existing unique
511,120-parameter Actor. Freeze the old and proposal Actors, constrain their center
difference in source-sigma normalized output space, evaluate `alpha=0:0.05:1` with
one deterministic DBM direct rollout per center, and train the same Actor architecture
against conservative safe targets. Do not restore a Gaussian head, use natural
gradients, or consume DBM analytic gradients.

The validation-only line-oracle diagnostic is current/proposal `25.518 -> 14.049`
overall, `39.528 -> 22.569` at 2.4 m/s, and `61.974 -> 31.549` at 2.8 m/s. Alpha zero
is selected on 17.5% overall and 27.5%/20.8% of 2.4/2.8 m/s contexts. This passes the
TR0 mechanism upper-bound gate but is not a learned-policy result.

Resume at `TR1`: implement the immutable train-only sidecar generator and independent
validator. Then train the safe-target Actor (`TR2`) and evaluate its actual unique
output on formal validation (`TR3`). Accept only if paired mean gain has a positive
95% CI lower bound, median/P05 are nonnegative, worst gain is at least -5, and
2.4/2.8 m/s plus recovery do not regress. Test, Query/ONNX, ROS, wrapper/four-round
search and closed loop remain sealed.

### 2026-08-07 FR-TRPI TR1 completion handoff

TR1 is complete and independently validated. The immutable train-only sidecar is
`/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/dbm_direct_trust_region_train_20260807_v1`.
It contains 360 train episodes split once into 300 internal-fit and 60
internal-selection episodes, 3,150 snapshots, 6,300 two-context rows, and 132,300
deterministic DBM direct rollouts over the fixed 21-point alpha grid. The 15 formal
validation plus 15 test episodes are listed as sealed and have no generated labels.

Independent replay reconstructed the frozen Actor inputs, trust projection, centers,
costs, hashes and safe-alpha rule for every row: maximum center error `0`, maximum cost
error `0`, maximum metadata error `5.914336e-7`, qualification `TR1_VALIDATED`.
Train-only old/safe mean cost is `19.411/12.345`; at 2.4 and 2.8 m/s it is
`26.838/16.344` and `48.740/31.432`. These figures qualify the target sidecar only and
must not be reported as a learned-Actor or formal-validation result.

Resume at TR2. Train the unchanged deterministic Actor only on `internal_fit`, select
using `internal_selection`, and measure safe-target fit, stay classification and actual
unique-output DBM cost. Keep formal validation sealed until the TR2 checkpoint and all
selection decisions are frozen; keep test, Query/ONNX, ROS, multi-round wrappers and
closed loop sealed until their later gates.

### 2026-08-10 FR-TRPI TR2 failure handoff

TR2 was executed without opening formal validation or test. The trainer and independent
validator are `train_mppi_direct_trust_region_actor.py` and
`validate_mppi_direct_trust_region_actor.py`; the frozen research output is
`outputs/mppi_proposal/direct_actor_trust_region_20260807_v1`. Three seeds ran 200
epochs from the unchanged 511,120-parameter old Actor. Internal selection chose seed 0,
epoch 135, followed by the planned all-train refit.

On the 600 internal-selection contexts, old/safe-label/learned direct mean cost is
`25.355/14.316/22.680`. Target sigma RMSE improves from `0.417` to `0.264`, but
safe-alpha-zero recall is `0%`, trust violations are `3.33%`, and gain median/P05/worst
is `0.448/-11.636/-294.542`. Every speed group improves in mean while high-speed P05
and worst regress severely. Independent replay over all 6,300 train contexts has zero
selected/final metric error and qualifies the run as
`TR2_FAIL_STAY_TRUST_AND_TAIL`.

Stay weights 8/32 improve P05 and recall but leave large worst losses. An internal-only
old-to-learned parameter scan shows alpha 0.05 already has positive mean gain `0.306`
but worst `-10.882`; no nonzero alpha passes the `worst>=-5` condition. Do not open TR3
or deploy this checkpoint. Resume only after choosing an explicit state-conditioned
alpha/stay gate, a verified two-center fallback, or a new forward-only proposal
direction. More plain epochs/MSE/stay reweighting is not the next action.

### 2026-08-10 TR2-B alpha/stay gate handoff

The explicit state-conditioned alpha branch is complete and passes its train-domain
gate. `TorchMPPITrustAlphaPolicy` consumes the existing state/context plus the frozen
16-D projected direction, requested rho and trust scale, then emits a move probability
and conditional continuous alpha. A hard threshold provides exact old-Actor fallback;
the output center remains unique and deterministic.

Three 150-epoch runs select seed 2 / epoch 35 on internal-selection. At threshold 0.5,
old/policy/safe mean cost is `25.355/14.855/14.316`, but worst gain is `-36.126`.
Internal-only threshold calibration selects 0.99: mean/median/P05/worst gain becomes
`3.834/0/0/-3.538`, move fraction is 12.67%, regression fraction 0.17%, stay recall
96.77%, and all five speed groups have nonnegative mean gain. The frozen fit-only
checkpoint is
`outputs/mppi_proposal/direct_trust_alpha_policy_20260810_v1/trust_alpha_policy_tail_calibrated.pt`;
qualification is `TR2B_PASS_ALPHA_AC_READY`. Do not substitute the all-train refit for
this calibrated checkpoint because it consumed internal-selection episodes.

The next authorized train-only step is one-dimensional Alpha Actor--Critic on the same
TR1 sidecar. Use the 119,700 internal-fit `(context,alpha,reward)` line points for replay
and the disjoint 12,600 internal-selection points for checkpoint selection. Reward is
`old_cost-direct_cost(alpha)`; twin Critics take continuous alpha, and the Actor starts
from the calibrated alpha policy. Keep formal validation/test, Query/ONNX, ROS and
closed loop sealed until the Alpha AC checkpoint and selection rules freeze.

### 2026-08-10 Alpha Actor--Critic completion handoff

The train-only one-dimensional continuous Alpha Actor--Critic is complete. Use
`train_mppi_direct_alpha_actor_critic.py`,
`validate_mppi_direct_alpha_actor_critic.py`, and the selected checkpoint under
`outputs/mppi_proposal/direct_alpha_actor_critic_20260810_v1`. It is a contextual
bandit with no vehicle next-state or Bellman bootstrap. Twin Critics train from all
119,700 internal-fit line actions; Actor updates use the conservative Q minimum plus
a Critic-derived 21-grid policy-improvement target for the hard move/stay boundary.
No safe-alpha teacher label or analytic DBM gradient updates the Actor.

Three seeds select seed 0 / Critic epoch 40 / Actor epoch 5. Per-epoch
internal-selection threshold calibration freezes threshold 0.88. On 600
internal-selection contexts, old/TR2-B/Alpha-AC/safe mean cost is
`25.355/21.522/20.696/14.316`; Alpha-AC gain mean/median/P05/worst is
`4.660/0/0/-3.537`, move fraction 20.33%, stay recall 96.77%, and every speed group's
mean gain is nonnegative. Independent checkpoint reload, hash checks and direct DBM
replay have zero saved-metric error. Qualification is
`ALPHA_AC_VALIDATED_READY_FOR_FORMAL_GATE`.

Do not use the Critic as an unguarded argmax: heldout grid argmax regret has a rare
catastrophic tail even though adjacent sign accuracy is 90.30%. The hard fallback is
part of the selected policy. The next action is the frozen formal-validation TR3
gate. Test, Query/ONNX, ROS, wrappers and closed loop remain sealed until that gate.

### 2026-08-10 TR3 protocol frozen before validation access

TR3 freezes Alpha-AC checkpoint
`ae862c09e09a46929e7cd172f54f2c2821cf6f20ba18ee7bef7b086c31ef9783`, threshold
0.88, and TR2-B checkpoint
`37cd09ab746f28fa7c3acf512e7f993d675ebc16b29104ba1f65e23497e5f012`, threshold
0.99. It may now read only validation episodes 090--104 (300 snapshots / 600
contexts); test episodes 105--119 remain sealed.

Require Alpha AC versus old Actor episode-bootstrap 95% mean-gain CI lower bound
above zero, median/P05 nonnegative, worst at least -5, and nonnegative 2.4/2.8 m/s
plus recovery mean gain. Also require Alpha AC versus TR2-B mean gain and its 95% CI
lower bound above zero; otherwise retain TR2-B. Threshold, epoch and seed may not be
retuned after validation access. Independent reconstruction/replay is mandatory.
Passing TR3 permits only planning a fixed-DBM short closed-loop A/B; it does not open
test, Query/ONNX, ROS or deployment.

### 2026-08-10 TR3 failed and independently replayed

The frozen D0 TR3 evaluation is complete under
`outputs/mppi_proposal/direct_alpha_ac_tr3_validation_20260810_v1`. Only validation
episodes 090--104 were read; thresholds stayed Alpha AC 0.88 and TR2-B 0.99, and
test 105--119 remains unopened. Old/TR2-B/Alpha-AC/safe-line/argmin-line mean direct
cost is `25.518/23.541/23.470/18.259/17.997`.

Alpha AC versus old mean gain is 2.049 with episode-bootstrap 95% CI
`[0.376,4.401]`, but worst gain is -75.937. Alpha AC versus TR2-B gain is only 0.071
with CI `[-0.299,0.390]`. Independent reconstruction and replay have zero error, so
the qualification is a real `TR3_FAIL_RETAIN_TR2B_OR_OLD`, not an implementation
failure. TR2-B also fails formal tail with worst -55.249.

Tail diagnosis finds that moved conditional alpha is almost always one. Alpha AC
moves on 17 contexts whose formal safe alpha is zero and all 17 regress. Treat the
current model as a failed stay-versus-endpoint selector, not a qualified continuous
step policy. Standalone fallback returns to the old Actor. Formal validation is now
consumed for diagnosis and must not be reused for retuning or future unbiased claims.

Next return to train-only line replay: train explicit endpoint-regression and
maximum-safe-alpha/tail heads, oversample high-speed/recovery safe-zero hard negatives,
and prevent conditional alpha saturation. Freeze a new validation-like episode
collection before retraining; keep existing test 105--119 sealed. Do not start DBM
closed loop, Query/ONNX, ROS or full 16-D AC before a new untouched gate passes.

### 2026-08-10 online Alpha-SAC sustained-update result

The train-only online Alpha-SAC now supports `--resume-run`. It restores the selected
Actor, main twin Critics, and actor-visited replay; newly written checkpoints also
restore both optimizer states. Independent validation checks every replay reward,
the selected policy, the Critics, and fit/selection episode separation.

Use the scalar-alpha safe label as the matched teacher. On the 600 internal-selection
contexts, old/safe-teacher/21-point-line-argmin cost is
`25.355/14.316/13.920`. The original 20-iteration run costs 20.180. Eighty more
iterations select global iteration 48 at 19.721, while another 40 iterations with
optimizer state restored do not improve it. Replay reaches 179,200 transitions.
The retained policy has move fraction 27.0%, versus 94.83% for the safe teacher, and
beats the teacher on only 23.33% of contexts.

An unsafe threshold sweep reaches cost 14.536 at threshold 0.18, but has 4.67%
regressions and worst gain -39.334. This is evidence that the conditional center is
near teacher quality on average, while move-versus-stay risk ranking remains the
bottleneck. Do not continue identical iterations or lower the deployment threshold.
Resume with a train-only delta/risk Critic objective or conservative lower-bound head,
then collect a new untouched validation-like set before any formal claim. Existing
formal validation is consumed and test 105--119 remains sealed.

### 2026-08-11 aggressive training / post-hoc safety split

Online Alpha-SAC now has `selection_mode=aggressive_mean`. Training keeps pretrained
initialization, bounded alpha, reward transforms, twin-Q, gradient clipping, the
immutable 21-point replay anchor, and finite-value checks, but no longer blocks action
collection or checkpoint selection with P05/worst/stay safety gates. Negative-reward
actions remain valid Critic data. `safe_gate` is retained only for post-hoc deployment
diagnostics.

Selected policy and latest resume state are now separate. Resume preferentially loads
`online_alpha_sac_training_state.pt` (latest Actor/Critics/optimizers); research
evaluation uses `online_alpha_sac_selected.pt` (lowest mean-cost Actor).

The authoritative run is
`outputs/mppi_proposal/direct_alpha_online_sac_aggressive_20260811_v2`. It starts at
14.536 versus the matched safe teacher 14.316 and line argmin 13.920. After 80 open
iterations, replay reaches 200,000 with 10.31% negative rewards, but no Actor improves
the start; the latest policy costs 14.586. Latest Critic adjacent-sign accuracy falls
0.8662 to 0.8455 and mean argmax regret rises 1.8835 to 2.1254. Thus the safety gate
is no longer the blocker; Critic rank drift is. Keep aggressive training and next
separate base-grid value/rank loss from online delta/reward loss. Formal validation
and test remain unopened for this branch.

### 2026-08-11 separated-Critic aggressive result

The online trainer now supports separate full-curve base value/adjacent-delta losses
and actor-visited online value/`Q(alpha)-Q(0)` losses, Critic-only warm-up, periodic
full-line rank diagnostics, selected/latest resume roles, and an action-head-only
Critic mode that freezes pretrained context representation.

Updating the full Critic still degrades heldout rank. Freezing context representation
reduces that drift, but Actor cost improves only after removing disagreement and
entropy penalties from the Actor objective; stochastic policy and uniform collection
continue to provide exploration. The useful v3/v4 chain improves aggressive mean cost
`14.535789 -> 14.518854` and teacher-beaten fraction `69.67% -> 72.00%`. It does not
beat the matched safe teacher 14.316077 or line argmin 13.920438. A later audit found
that optimizer restore had overwritten the requested resume LR: both v3 and v4
actually used Actor LR `1e-5` and Critic LR `5e-5`. Treat them as same-LR continuation,
not an LR ablation. The trainer now reapplies CLI LR/weight decay after optimizer
restore and records effective LRs.

Use `outputs/mppi_proposal/direct_alpha_online_sac_aggressive_separated_20260811_v4`
as the authoritative result. Independent replay validates 200,000 rewards and both
selected/latest policy/Critic states with no selection leakage. Formal validation and
test remain unopened. Next analyze the remaining teacher-gap contexts and add paired
local probes around the Actor alpha to improve Q derivative precision; do not restore
train-time safety gates.

### 2026-08-11 fixed-replay Critic-only result

The online trainer now supports `training_mode=critic_only` with transition collection
disabled. Starting from the v4 selected state, two isolated runs freeze the Actor,
keep the same 200,000-transition replay, update only Critic action heads for 60 x 40
minibatches, and select Critic checkpoints on the disjoint 600-context, 21-alpha line
ranking metric. Effective Critic LRs are `1e-5` and `3e-6`.

Both runs retain the iteration-330 starting Critic. Selection score starts at
`2.00051`; latest scores are `2.02108` and `2.01751`. Adjacent-sign accuracy changes
from `0.86042` to `0.85611/0.85646`, and mean argmax regret from `1.91714` to
`1.93763/1.93406`. Actor cost remains exactly `14.518854`. Thus more Critic updates on
the existing replay do not improve the action ranking; lowering LR only reduces the
damage slightly. Do not resume Actor updates yet. Add paired local probes around the
current Actor alpha, train the Critic against explicit local order/delta information,
and require this same frozen-Actor gate to improve before joint SAC resumes.

Use both run records under
`outputs/mppi_proposal/direct_alpha_critic_only_fixed_replay_20260811_lr1e5` and
`..._lr3e6`. Independent replay/checkpoint validation passes at no more than `5.96e-8`;
formal validation/test remain unopened.

### 2026-08-11 single-step online-loop correction and local probes

The current Alpha task is a terminal contextual bandit that can interact with DBM
repeatedly. It does not need vehicle `next_state` or Bellman bootstrap: each loop asks
the Actor for alpha, obtains exact forward reward `old_cost-direct_cost`, updates the
Critic, then updates the Actor. Previous failure was not missing environment feedback;
it was unreliable continuous Q slope from unpaired action samples and replay dilution.

The trainer now collects paired `a±0.02/a±0.05` rewards around the deterministic Actor,
stores resumable `actor_local_probe_replay.npz`, and trains explicit local delta/sign
losses without analytic DBM gradients. A guarded Critic selector requires local score
improvement while preserving the complete 21-alpha line.

Frozen-Actor run `direct_alpha_critic_local_probe_20260811_v1` collects 25,600 local
triplets. Local adjacent/central sign improves `63.66/66.57% -> 70.79/73.31%`, local
regret `0.11961 -> 0.10794`; full-line sign improves `86.04% -> 86.59%` and regret
`1.91714 -> 1.86689`. Actor cost stays exactly 14.518854, so the Critic-only gate passes.

The following low-frequency Actor run `direct_alpha_online_sac_local_probe_20260811_v1`
uses Actor LR `2e-6` and one update per loop. Cost decreases monotonically over 20 loops
but only `14.518854 -> 14.518536`; it remains above teacher 14.316077. Local sign/regret
continue to `72.43/74.86%` and `0.10542`, while full-line sign/regret reach `86.76%` and
`1.85327`. Record this as mechanism PASS and performance-gap OPEN.

An Actor-output audit shows why v1 gain is small: no move-gate decisions change and
hard alpha moves only `1.71e-4` mean absolute. A controlled v2 changes only Actor LR
from `2e-6` to `1e-5`. Another 20 loops monotonically improve cost
`14.518536 -> 14.516302`, increasing teacher-beaten contexts `72.00% -> 72.33%`, but
teacher gap remains `0.200226`. Hard alpha moves `0.001154` mean absolute and still has
zero gate flips. Use `direct_alpha_online_sac_local_probe_20260811_v2` as the current
local-probe Actor result. The next issue is policy step/gate parameterization, not
missing reward or vehicle-transition SAC.

Independent validation reconstructs all normal/local rewards and checkpoint metrics;
maximum error is `1.56e-7` for Critic-only and `1.19e-6` for joint training. Formal
validation/test remain sealed; v2 independently passes at `7.75e-7`. Continue the
single-step online loop; do not introduce multi-step Bellman merely to fix this
local-gradient problem.

### 2026-08-11 16-D wide-exploration handoff

The narrow full-rank residual Actor remains the current research result at 14.312472.
Two requested wider/longer branches failed. With the same 2-sigma support, 1.0-to-0.08
sigma probes and 60 rounds select iteration 2 at only 14.509499; later training
diverges and ends at 16.026249. A 6-sigma, 80-round branch uses 2.0-to-0.15-sigma
probes and separately DBM-verifies a bounded 0.15-sigma intermediate target, but no
checkpoint beats iteration zero and latest cost is 28.660473. Six-sigma projected
J16 fit cost is 4.840856, so action reachability is not the active blocker.

Both selected Actor metrics reproduce exactly without split leakage, but extreme
broad replay fails the strict reward replay gate and is not a reusable Critic asset.
Do not add more radius or identical rounds. Resume by freezing the narrow residual
direction and training a paired base-vs-residual delta/risk gate, or by adding
per-state multi-forward feedback before each bounded direction update. Keep formal
validation/test, Query/ONNX, ROS, and closed loop sealed.

### 2026-08-11 rollout-feedback-guided exploration handoff

The per-state multi-forward branch is now implemented by
`evaluate_mppi_direct_feedback_guided_exploration.py`. It freezes the narrow residual
Actor and uses only the 600 internal-selection contexts. A shared 33-rollout first
pass probes all 16 antithetic Hadamard directions at 0.10 source sigma. Complete
weighted trajectory residual changes are fit with ridge regression and converted to
a damped Gauss--Newton direction; scalar cost and blended directions are retained as
ablations. No analytic DBM gradient is used.

With an equal 18-rollout second-pass budget, blind antithetic search around the first-
pass winner reaches mean direct cost 6.974455, while the combined response bank reaches
6.337867. Incremental gain beyond the common first pass is 1.076911 versus 1.713499;
the second pass improves 50.0% versus 91.5% of contexts. A six-candidate trajectory-
only line already reaches 6.484626. All five speed groups improve over blind. J16
best-found remains 4.916733.

The first-pass design has rank 16, Actor/base replay differs by at most 3.05e-5, and a
complete deterministic rerun reproduces every saved per-context array exactly. This
is a two-pass search mechanism result, not a unique Actor output or deployment score.
Next collect the same response-guided positive and negative actions on internal-fit
contexts, but keep the inference contract state-only and unique-output: the new probe
response is training supervision for `Q(s,a)`, not a new Actor input. Train value plus
within-state finite-difference delta/sign losses for antithetic and response pairs;
update the Actor only through the continuously trained Critic with a trust penalty.
Do not clone probe winners or response directions into the Actor. Formal validation/
test, Query/ONNX, ROS, and closed loop stay sealed.

### 2026-08-11 response-slope Actor--Critic result

`train_mppi_direct_response_slope_ac.py` keeps the Actor state/context-only and uses
training probes solely for scalar-Q value plus within-state delta/sign supervision.
The joint run stores 266,820 actions and 174,080 pairs, but Actor cost degrades from
14.312472 to 14.341774; iteration zero is retained. Heldout direction sign/correlation/
regret finish at 51.08%/-0.017/30.41. A frozen-Actor strong-pair control with loss
weights 0.2/10/1 still reaches only 52.35%/-0.011/18.84.

An independent 1,024-action reward replay check is within 3.05e-4 with zero split
leakage and exact selected-Actor determinism. Reject the generic scalar Q for Actor
gradients. Next test an explicit actor-centered local response Q head and require a
frozen-Actor heldout direction/sign/regret gate before joint updates. Actor inputs,
formal validation/test, and Query/ONNX/ROS remain unchanged/sealed.

### 2026-08-12 full-16D local-Critic execution entry

The next experiment is frozen before implementation in section 25 of
`mppi_direct_actor_trpo_like_design_20260807.md`. Keep the selected narrow residual
Actor fixed and replace the generic scalar action MLP with an actor-centered local
model `V(s)+g(s)^T delta+0.5*c(s)*||delta||^2`. The gradient head remains all 16
dimensions; this is not a projection or a reduction of the 8x2-knot action.

Generate full-rank antithetic forward-reward labels at multiple radii on internal-fit
and evaluate only on disjoint internal-selection episodes. First measure the empirical
label's cross-radius cosine, then train the explicit gradient head. Do not update the
Actor unless heldout median gradient cosine >=0.40, positive-cosine fraction >=70%,
meaningful pair sign >=65%, and 33-probe mean argmax regret <=4.0. If fit succeeds but
heldout fails, record state-to-gradient generalization/coverage as the blocker instead
of adding wider blind exploration. Formal validation/test, Query/ONNX/ROS and closed
loop remain sealed.

### 2026-08-12 full-16D local-Critic result

The predeclared frozen-Actor experiment is complete. New implementation and output:

```text
car_foundation/car_foundation/mppi_proposal_policy.py
  TorchMPPIActorCenteredLocalCritic
scripts/model_verify/train_mppi_direct_local_gradient_critic.py
scripts/model_verify/validate_mppi_direct_local_gradient_critic.py
outputs/mppi_proposal/direct_local_gradient_critic_20260812_v1
```

It evaluates 6,300 train-domain contexts at three radii (`0.05/0.10/0.20 sigma`) and
33 centers/radius, for 623,700 deterministic forward DBM rollouts. Every per-radius
and combined design has rank 17 (16 gradient components plus scalar radial curvature).
The fit/heldout cross-radius gradient median cosine is `0.891/0.894`, so the local
forward labels are stable enough for this test.

The explicit 3-seed Critic improves substantially over the generic scalar-Q baseline:
heldout pair sign `52.35% -> 58.90%`, delta correlation `-0.011 -> 0.199`, and 33-probe
mean regret `18.84 -> 6.261`. It still fails the frozen gate. Heldout gradient median
cosine is `0.463`, but positive-cosine fraction is only `68.33%`, pair sign is below
the required 65%, and regret is above 4.0. Train median cosine is `0.767`, identifying
the remaining blocker as state-to-gradient episode generalization rather than missing
rank or a generic-Q derivative implementation.

The failure is speed concentrated: heldout median cosine at 1.2/1.6/2.0/2.4/2.8 m/s is
`0.721/0.669/0.460/0.257/0.279`, while mean regret is
`0.203/0.851/3.675/12.099/14.477`. Independent validation reproduces Actor outputs and
heldout metrics exactly; 1,024 reward replays differ by at most `8.01e-5`, with zero
episode leakage. Qualification is
`FIT_PASS_HELDOUT_FAIL_STATE_TO_GRADIENT_GENERALIZATION`; no Actor checkpoint was
generated or updated.

Next keep all 16 action dimensions and the unique-output Actor. Add independent
2.4/2.8-m/s and recovery state coverage (not adjacent-frame duplication), generate the
same full-rank local labels, and train with speed/scenario-balanced angular and pair
order objectives. Require a new untouched episode-heldout cosine/sign/regret gate
before joint Actor updates. Formal validation/test, Query/ONNX/ROS and closed loop
remain sealed.

### 2026-08-12 frozen local-gradient step validation entry

Before collecting more states, run one direct mechanism check on the improved local
Critic. Freeze Actor and all three Critics. Normalize their mean 16-D gradient by RMS,
evaluate signed radii `0/0.005/0.01/0.02/0.03/0.05/0.075/0.10 source sigma` with one
deterministic DBM direct rollout per center, and select one fixed positive radius only
on the 1,110 fit-internal-validation contexts. Evaluate that frozen radius on the 600
disjoint internal-selection contexts; use the equal-radius negative direction as a
sign control. Report mean/median/P05/worst, wins/losses, speed/scenario groups, positive
line oracle and bidirectional oracle. This experiment never updates Actor/Critic and
does not reopen formal validation/test, Query/ONNX/ROS or closed loop.

### 2026-08-12 frozen local-gradient step validation result

The mechanism validation is complete at
`outputs/mppi_proposal/direct_local_gradient_step_eval_20260812_v1`. The positive
radius selected only on 1,110 fit-internal-validation contexts is `0.01 source sigma`;
validation mean cost is `15.273 -> 14.896`. On 600 disjoint internal-selection
contexts, frozen Actor/positive/negative mean direct cost is
`14.312/14.005/15.235`. Positive mean/median gain is `+0.307/+0.051` with 330/270
wins/losses; the equal-radius negative control has mean gain `-0.923` and 176/424.
All three predeclared mechanism gates pass.

This is actionable direction evidence, not an Actor gate. Positive P05/worst gain is
`-3.262/-20.837`, so 45% of contexts still regress. A same-radius perfect sign selector
would reach cost `12.820`; a positive-radius line oracle including stay reaches
`10.515`, and a bidirectional line oracle reaches `8.957`. The remaining gap is mainly
context-dependent stay/sign/step selection, not lack of a usable mean direction.

Independent validation exactly reproduces radius selection and all positive/negative
metrics; 1,024 direct DBM replay costs have zero error and the validation/selection
episode overlap is zero. Qualification is `LOCAL_GRADIENT_STEP_MECHANISM_PASS`, while
Actor/tail qualification remains FAIL. No Actor/Critic was updated and formal
validation/test, Query/ONNX/ROS and closed loop remain sealed. If continuing, generate
signed-line replay only on unconsumed internal-fit contexts, train a bounded
stay/sign/step head, and use a newly collected episode-heldout gate rather than tuning
on these now-consumed 600 contexts.

### 2026-08-13 deploy-Gaussian center integration diagnostic

The fixed-state integration check requested by the audit is complete at
`outputs/mppi_proposal/two_center_integration_20260813_v2`. It uses the current
residual Actor, fixed small-car DBM, 256 Gaussian candidates with
`noise_sigma=[0.25,0.35]`, three common-random-number seeds, and all 600 already
consumed internal-selection contexts. Formal validation/test remain sealed. The
independent validator passes exact center-candidate costs, softmax reconstruction,
hard-floor reconstruction, and 24-context weighted-output DBM replay.

Actor-centered replacement transfers the open-loop gain strongly in the mean:
warm-MPPI/Actor-replace weighted-output mean cost is `18.188/9.057`. It still has
5.17% regressions, P05 gain `-0.110`, and worst `-42.282`. Adding exact warm as the
second candidate in the 256-sample Actor bank does not solve the tail: mean cost is
`9.052`, but worst gain is `-90.878`. One 2.8-m/s dynamic-recovery case combines four
29--32-cost candidates into a 118.989-cost weighted action, directly confirming the
non-convex action-blending failure mode.

Hard-min between soft output and direct warm preserves only the direct-warm model
floor and still has worst gain `-23.076` versus the actual warm-MPPI baseline. The
recommended DBM selection structure retains the
original warm-centered 256 bank, evaluate its weighted output once, evaluate the
Actor direct sequence once, and hard-selects the lower model cost. With Actor center
already supplied this is 258 rollouts and achieves mean
cost `10.670`, mean gain `+7.519`, selects Actor in 80.72% of pairs, and has P05/worst
gain exactly zero by construction.

Budget correction: the current residual Actor requires one 128-rollout first pass to
construct its guided anchor, 74-D feedback and feedback-Critic gradient context.
Those inputs were cached in this fixed-state diagnostic. Strict online reproduction
therefore costs `128+1+256+2=387`, not 258, per control step; the extra first-pass
rollout evaluates the weighted output stored in the 74-D feedback. Only a separately verified
reuse of the warm 256 bank can remove the 128. First reconstruct one frozen first-pass
seed and the final Actor center directly from raw snapshots, then implement the hard
guard and run the minimal matched-seed closed-loop A/B. Query/real use
still requires separate pairwise model-ranking qualification. Full reasoning and
speed/recovery tables are in `mppi_sampling_center_review_archive_20260812.md` §9.

### 2026-08-13 residual Actor online-input reconstruction gate

The prerequisite above now passes. Use
`scripts/model_verify/validate_mppi_residual_actor_runtime_inputs.py`; its authoritative
artifact is
`outputs/mppi_proposal/residual_actor_runtime_input_validation_20260813_v2`.
Starting only from 300 consumed internal-selection raw DBM snapshots, the script
recomputes both frozen first-pass contexts (600 total): frozen-BC center, 128
antithetic rollouts, guided anchor, 74-D feedback, three-Critic gradient mean/std,
old/proposal and Alpha base, and the final residual center. Formal validation/test
remain sealed.

Qualification is `PASS_RUNTIME_INPUT_RECONSTRUCTION`. Every intermediate maximum
absolute error is zero; final residual-center error is `9.09e-7` versus a `5e-5`
gate. This removes cached-label dependence as a blocker for DBM closed loop. Runtime
uses one frozen first-pass seed, so the honest hard-guard budget is
`128+1+256+1+1=387` DBM rollouts per step. The first extra one is required by the
74-D feedback's weighted-output-cost field. The earlier v1 summary omitted that
single evaluation and is superseded by v2; reconstruction values are unchanged.
Next, package this exact chain as shared
runtime code, preserve the original warm-MPPI sampling and running-state recursion,
and hard-select between its weighted output and the direct Actor sequence before a
matched-seed short closed-loop A/B.

### 2026-08-13 first hard-guard closed-loop mechanism pilot

The shared runtime and first matched-seed pilot are complete. Runtime code lives in
`car_foundation/mppi_residual_actor_runtime.py`; the controller now exposes
side-effect-free deterministic sequence evaluation and hard selection. ROS use is
opt-in through `mppi_hard_guard_checkpoint`; the empty default preserves baseline.
The 300-state component qualification passes with maximum final-center error
`8.34e-7`, and the 60-state hard-guard primitive has zero cost/sequence/RNG/running-
state error and zero warm-floor violation.

The paired data are under
`mppi_hard_guard_closed_loop_pilot_20260813_v1/{baseline_seed3407,guard_seed3407}`;
both contain contiguous step-0--300 traces and six validated snapshots. Analysis and
independent validation are at
`outputs/mppi_proposal/hard_guard_closed_loop_pilot_20260813_v1`.
This is one 2.8-m/s scene with MPPI seed 3407 and remains mechanism-only.

Across all 301 steps, realized Frenet stage cost improves 4.12% and lateral/heading/
speed RMSE improve 0.94/2.87/1.21%. Over history-complete steps 250--300, cost improves
2.29%, heading/speed improve 8.80/4.69%, but lateral RMSE regresses 8.87%.
Acceleration/steering rate RMS regress 45.5/16.4%, and second differences regress
87.4/28.2%. Actor is selected on 39.2% of mature steps with a 34% branch-switch rate.
Its per-step hard min keeps zero warm-model-cost violation and mean mature local gain
2.564, so open-loop gain partially transfers but independent switching adds jitter.

Realtime fails decisively: mature baseline/guard mean durations are 63.4/242.2 ms and
both miss the 50-ms deadline on every step. Do not advance this implementation to
formal validation. First add stateful hysteresis/minimum dwell/switch cost without
removing the warm floor, then repeat multiple internal matched seeds and nominal/
high-speed/recovery scenes. Separately reduce 387 rollouts by qualifying warm-bank
feedback reuse or distillation.

### 2026-08-13 fresh finite-difference Critic gradient audit

The first direct gradient validation is complete at
`outputs/mppi_proposal/direct_critic_fresh_fd_20260813_v2`. The frozen residual Actor
and three local Critics were audited on all 600 already-consumed internal-selection
contexts using a fresh random orthogonal 16-direction bank and fresh
`0.01/0.02/0.04 source-sigma` radii. This required 59,400 deterministic DBM candidate
rollouts and did not load formal validation/test.

The true small-radius transformed-reward gradients are exceptionally stable:
cross-radius cosine medians are `0.999995--1.000000`, P10 is at least `0.999690`,
and median norm ratios are within 0.11% of one. The old `0.05-sigma` Hadamard label
also transfers to the fresh direction bank (median/P10 cosine `0.9899/0.9590`), so
neither reward roughness nor a special training direction basis explains the failure.
Combining the old `0.05/0.10/0.20` radii into one target does distort it: combined-
label vs fresh median cosine is only `0.7726`; large radii must remain value/ranking/
curvature samples rather than be averaged into the infinitesimal-gradient label.

The learned ensemble fails the real gradient gate: fresh-FD cosine median/P10 is
`0.3447/-0.6920`, positive fraction is 63.0%, and component correlation is 0.1829.
Its median gradient norm is `0.122` versus `15.727` true, a median norm ratio of only
`0.00850`. Failure exists at every speed and norm collapse worsens at 2.4/2.8 m/s.
The three Critics mutually agree in angle (pairwise medians `0.953--0.962`) while
disagreeing in scale and being wrong versus FD; twin-Critic cosine alone is therefore
not a correctness metric. Autograd exactly matches the explicit gradient heads.

`validate_mppi_direct_critic_fresh_fd.py` reconstructs all gradients/metrics exactly
and replays 1,024 sampled DBM candidate costs with zero error. Qualification is
`REWARD_GRADIENT_STABLE_CRITIC_GENERALIZATION_FAIL`. Keep Actor frozen. Next relabel
same-state rotated probes with the smallest radius as gradient truth, reserve larger
radii for value/ranking, add within-state ranking/delta loss, and select checkpoints
on untouched-episode FD cosine, norm calibration, and P10 rather than Q1/Q2 mutual
agreement. Full details are in `mppi_sampling_center_review_archive_20260812.md` §11.9.

### 2026-08-13 action-coordinate/Jacobian exclusion

Before changing Critic training, the fresh-FD validator was extended to exclude a
normalized/physical/pre-tanh coordinate bug. The exact contract is
`c=clip(c_alpha+M*sigma*tanh(z))` and
`u_eff=(c-c_alpha)/(M*sigma)`, with `M=2` and `sigma=[0.25,0.35]`. Both the Critic
action input/gradient head and the finite-difference fit use `u_eff`; the FD physical
offset is converted back to this coordinate before fitting.

Checkpoint/payload/stored scales match exactly. Independent normalized-to-center FD
gradient conversion has maximum vector-relative error `3.70e-7`; center autograd
chain error is zero and the tanh+clamp chain error is `2.60e-6`. Tanh is unsaturated
(minimum derivative `0.99913`). Clamp affects only 0.302% of action elements and
4.67% of contexts. Converting both predicted and true gradients to physical-center
coordinates gives median cosine/norm ratio `0.2655/0.00871`; converting both through
the Actor pre-tanh Jacobian gives `0.3447/0.00850`, essentially the original result.

Therefore the `0.00850` gradient-norm collapse is not a missing sigma, maximum-action,
tanh, or autograd Jacobian. Keep the §11.9 diagnosis and proceed with Critic label/loss
ablation. The authoritative check is embedded in
`direct_critic_fresh_fd_20260813_v2/validation_summary.json`; full formulas are in
archive §11.10.

### 2026-08-13 local-Critic retraining decision

The current local-Critic task is supervised numerical regression, not teacher-network
distillation and not TD/bootstrap RL. Frozen-Actor DBM rollouts produce same-state
antithetic rewards; finite differences produce `(V,g16,h)` labels; the explicit
gradient head is directly regressed and later consumed by the Actor.

Treat label processing and loss design as complementary, not alternative. The current
trainer already has component gradient Smooth-L1, cosine, value, curvature and bank
reconstruction. Therefore merely adding another `0.05-sigma` local-difference loss is
not the first controlled fix. First reuse existing labels and supervise `g16` from the
smallest `0.05-sigma` radius only; stop averaging `0.10/0.20` finite-radius response
into the local derivative. Keep larger radii for paired delta/ranking/value/shape, and
do not let a scalar-curvature bank loss force the gradient head to absorb high-order
shape error.

The checkpoint score must also expose norm collapse. Replace direction-only selection
with median/P10 FD cosine, log norm-ratio error, same-state pair delta/sign, and real
probe regret; make norm calibration a hard gate. Then add explicit gradient-magnitude
loss plus paired `Q+ - Q-` delta and ranking losses. Run A0 current, A1 0.05-only,
A2 +pair loss, A3 +magnitude/norm-aware selection with Actor frozen. This first ablation
requires no new rollout and is mechanism-only on consumed splits.

Only after A3 improves direction, norm and tail together should train-wide fresh
`0.01/0.02-sigma` rotated labels be generated. Those small radii supervise the true
derivative; `0.05` remains local pair supervision; `0.10/0.20` remain finite-response
samples. Use a newly collected episode-heldout gate for qualification and speed-
balanced batches. Full design and rationale are in review §11.11.

### 2026-08-13 training-first Critic review refinement

The training-side priority is confirmed with an additional split replay. Ensemble
gradient norm is already collapsed on the 4,590 optimization rows: prediction/label
median is `0.1288/4.3406` (2.97%, per-row median 3.05%), while training cosine is
0.767. Internal-validation and internal-selection ratios are 2.93% and 2.49%.
Therefore norm collapse is not merely episode-heldout generalization and cannot be
fixed by changing labels alone.

The component Smooth-L1 already directly supervises gradient magnitude after per-
component standardization, and the gradient head is an unbounded linear layer. Do not
misdiagnose this as magnitude being available only through value residuals or a tanh-
bounded output. Instrument per-loss parameter-gradient norms, objective competition,
weight decay and zero-initialized-head dynamics, plus checkpoint timing.

Treat `0.00850` as a gradient amplitude ratio, not a literal signal-to-noise ratio.
A global scale can be absorbed by normalization/Adam/LR, and the previous RMS-
normalized fixed step did improve mean cost. The actual blocker is state-dependent
scale miscalibration (norm correlation only `0.26--0.39`) together with negative-tail
direction cosine; blindly amplifying the output is unsafe.

Reorder the zero-rollout ablation accordingly: B0 instrument the current mixed target;
B1 change only checkpoint selection to include norm; B2 retain mixed labels and add
log-norm loss plus output-head weight-decay/loss-gradient diagnostics; B3 switch only
the derivative target to `0.05 sigma`; B4 add same-state delta/ranking while reserving
large radii for response. Require B2 to fit train-label norm before attributing failure
to label quality, and require B4 to pass fresh-FD direction/norm/P10 before Actor
updates. Review §11.12 contains the full evidence.

### 2026-08-13 loss-gradient priority audit

A 1,024-train-row autograd audit at all three selected checkpoints changes the
suspect order. After applying the actual loss weights, cosine-loss parameter-gradient
norm on the gradient head is `1.23--1.80`, component-gradient loss is
`0.043--0.175`, and bank loss is only `0.0009--0.0065`. Cosine therefore dominates
component magnitude learning by 7--29x while being scale-invariant, matching the
small-norm/direction-partly-learned failure shape. Instrument and schedule/reduce
cosine only after component + log-norm warmup restores scale.

Keep a bank-off/small-radius-bank ablation, but do not assume scalar curvature can
replace the odd gradient: antithetic `Q(+delta)-Q(-delta)=2g^T delta` cancels curvature.
Bank can still distort finite-radius secants and shared features, but is not the first
direct-head suspect. AdamW decay is also negligible at the current hyperparameters:
`lr*wd=2e-9` per step, approximately `3e-6` cumulative over 1,500 steps.

Make per-row log norm error, norm correlation and speed calibration visible to
checkpoint selection first. Then fix component/cosine balance, run bank ablation,
switch to stored 0.05-only derivative, and only then generate train-wide 0.01/0.02
labels. Keep Actor frozen and change one variable family per ablation.

### 2026-08-13 local-Critic retraining ablation result

`train_mppi_direct_local_gradient_critic.py` now supports precomputed-label reuse,
norm-aware metrics/score and hard checkpoint eligibility, combined versus 0.05-only
derivative targets, bank modes, cosine scheduling/log-norm loss, and smallest-radius
pair delta/ranking. Actor remained frozen; all runs used consumed internal splits only.
The reproducible aggregate is
`outputs/mppi_proposal/direct_local_critic_retraining_ablation_20260813_v1/analysis.json`.

The old checkpoint rule was a real bug: changing selection only moved the best epochs
from `18/33/28` to `121/134/107`, restored train predicted/label norm ratio from about
3% to 79.8%, and raised train cosine from 0.767 to 0.902. However, the independent
fresh-FD gate improved only from cosine/norm `0.345/0.00850` to `0.400/0.22867`; P10
remained negative. Log-norm/cosine warmup did not fix direction, and bank-off did not
improve heldout metrics, so bank loss and weight decay remain demoted as primary causes.

Using the stored 0.05-sigma derivative with a real `[0.5,2.0]` checkpoint norm gate
improved fresh-FD median cosine/norm ratio to `0.613/0.42465`, proving multi-radius
gradient mixing was an important direction error. Its fresh P10 was still `-0.932`.
Adding smallest-radius bank plus pair delta/ranking produced `0.603/-0.932/0.48599`
for fresh median/P10/norm ratio, so it did not repair the cross-episode negative tail.
All three new fresh-FD audits independently PASS hash/split/reconstruction/autograd and
1,024-candidate DBM replay checks with zero cost error.

Qualification is `AMPLITUDE_AND_MEDIAN_IMPROVED_NEGATIVE_TAIL_FAIL`; do not update the
Actor. Next collect same-state complete-16D small-radius perturbations on more diverse,
episode/speed-balanced internal-fit states. Use <=0.02 sigma only for derivative,
0.05 for derivative/pair supervision, and 0.10/0.20 only for a separate finite-response
target. Require fresh-FD median >=0.70, P10 >=0, and norm ratio >=0.50 before Actor work.
Do not interpret the 0.10-sigma probe regret of about 22 for the 0.05-gradient models as
a derivative failure: it shows a local derivative plus scalar curvature cannot be used
as a large-radius candidate value head without a trust region.

### 2026-08-13 negative-tail state-representation pilot

Do not immediately execute the generic "more diverse episodes" recommendation above.
The current split already has 240 train, 60 internal-validation, and 60 internal-
selection episodes with balanced speed/scenario groups. The frozen B4 tail has 225/600
negative-cosine states; 157/600 are negative for all three Critics and median absolute
cosine is 0.866. The tail appears within 59/60 episodes and every speed/scenario, so it
is neither ensemble cancellation nor one missing domain.

`analyze_mppi_local_critic_negative_tail.py` writes the consumed-split pilot to
`outputs/mppi_proposal/direct_local_critic_negative_tail_20260813_v1/analysis.json`.
In frozen-Actor-encoder-plus-action space, nearest-train label cosine is only
median/P10 `0.260/-0.907`; the normalized top-20 mean is `0.457/-0.863`, while the
oracle-best label among those same 20 neighbors is `0.954/0.772`. Neighbor direction
coherence median is only 0.307. This indicates opposite derivative branches are mixed
inside the current representation neighborhood, although it does not prove raw inputs
are intrinsically ambiguous.

Next create a hard-state manifest for the negative fresh-FD rows and their conflicting
train neighbors, then audit raw history/reference/current differences. Add state
coverage around those boundaries (nearby time, recovery magnitude, curvature), not
more random episodes or more action radii at the same state. Keep complete 16D local
FD, use hard-example balanced Critic training and consider representation/contrastive
consistency only if the raw inputs are distinguishable. Actor stays frozen; do not
change its input/output contract.

### 2026-08-13 strict 32-state local-Critic overfit test

The zero-new-rollout tiny-set diagnostic is implemented by
`overfit_mppi_direct_local_gradient_critic.py` and independently replayed by
`validate_mppi_direct_local_critic_tiny_overfit.py`. The authoritative artifact is
`outputs/mppi_proposal/direct_local_critic_tiny_overfit_20260813_v2`; v1 used a looser
tail gate and is superseded.

The fixed set contains 32 non-boundary internal-fit contexts from 32 episodes, balanced
over all five speeds and six scenarios. It uses the stored 0.05-sigma 16D derivative,
freezes Actor, loads no formal validation/test, and performs no DBM rollout. Dropout,
weight decay, scheduling, and early stopping are disabled. The strict gate requires
per-set cosine median >0.99, P10 >0.98, every cosine >0.98, median norm ratio in
[0.95,1.05], and every norm ratio in [0.8,1.2].

Gradient-only and the B4 value/gradient/cosine/curvature/all-bank objective both pass
3/3 seeds. Gradient-only reaches cosine minimum 0.982--0.988 and norm ratio
0.9987--1.0011. B4 multi-task reaches cosine minimum 0.983--0.993 and norm ratio
0.823--1.154, with median norm 0.955--0.967. Independent checkpoint replay passes with
9.54e-7 maximum prediction error.

This rules out a basic architecture/action-coordinate/optimizer inability to memorize
the numerical gradient. The B4 loss introduces a small downward magnitude bias but no
hard direction conflict, and cannot explain the heldout near-180-degree reversals.
Continue with the zero-rollout density/clipping/regime attribution and hard-state raw-
input audit. Tiny-set memorization is not evidence of episode-heldout generalization;
Actor remains frozen.

### 2026-08-13 full-data hard-state zero-rollout attribution

The post-B-1 attribution is complete at
`outputs/mppi_proposal/direct_local_critic_hard_state_attribution_20260813_v1`, using
`analyze_mppi_local_critic_hard_states.py`; independent validation by
`validate_mppi_local_critic_hard_states.py` is PASS. It uses only 4,590 consumed train
and 600 consumed internal-selection contexts, performs no DBM rollout, and keeps Actor,
Critics, formal validation and test sealed. The 225 negative fresh-FD states are saved
in `hard_state_manifest.json`.

Hard states are not low-density outliers. Their raw/Actor/Critic top-1 density medians
are 0.696/0.896/0.941 versus 0.710/0.886/0.944 for non-hard states, and hard fractions
remain 34--40% across most density quartiles. Clipping accounts for only 16/225 hard
states. Speed/scenario and adjacent-state transition show no single controlling regime.

The failure is steering-dominated: early steering knots contribute 93.41% of negative
alignment mass; hard-state sign accuracy at steering knots 1/2/3 is only
8.9/5.3/11.6%. Raw and frozen-Actor top-20 neighborhoods are label-mixed but usually
contain a correct branch (best cosine median 0.958/0.948). The trained Critic
representation instead forms a coherent wrong branch: hard top-20 mean-label cosine is
-0.846 with coherence 0.800, and best-label P10 is -0.667. No single raw input block
distinguishes oracle-good from oracle-bad Actor neighbors under ordinary distance.

Route the next experiment to existing-label hard-pair training, not random data. First
compare hard-example-balanced sampling against B4. Then add state-gradient metric/
contrastive consistency while retaining the current explicit 16D gradient head and
unchanged Actor contract. Report early-steering sign/cosine plus fresh-FD median/P10/
norm. Only if both controls fail should targeted neighboring-time/recovery/curvature
states or missing context be considered. Actor remains frozen.

### 2026-08-13 completed hard-state attribution v2

The attribution is now complete at
`outputs/mppi_proposal/direct_local_critic_hard_state_attribution_20260813_v2`.
It extends v1 without replacing its evidence: the stored 0.05-sigma label is checked
against fresh FD first, all routing thresholds are pre-registered and non-exclusive,
the full 600-context manifest records KNN percentiles/coherence plus physical snapshot
ordinal/repeat index, and the gradient error is split into norm/direction, dominant
component signs, and parallel/perpendicular parts. The independent validator rebuilds
all 13 route flags and reports PASS. No DBM rollout, training, formal validation, or
test access is involved.

The old label is not the primary cause. Its cosine against fresh FD is 0.990 median and
0.959 P10 over all 600 rows, 0.990/0.953 on the 225 hard rows, and never becomes
negative. Strong mismatch below 0.90 covers only 8/225 hard rows. Label norm is mildly
attenuated (0.832 median), but Critic cosine has only 0.079 correlation with this label
self-check.

Low density is also rejected: lowest-decile raw/Actor density has lower hard rate than
the complement. Clipping covers 16/225 hard rows. Cold-start, repeat instability and
adjacent-state transitions are modest enrichments, not exclusive switches. The main
offline diagnostic is a coherent wrong Critic branch: top-20 train labels in Critic
feature space have mean cosine below -0.5 and coherence above 0.5 for 181 contexts;
178 are hard. This flag has 98.3% hard rate, covers 79.1% of hard rows, and has 8.77x
hard-rate lift. It uses fresh labels and is diagnostic only, not deployable gating.

The direction error is a coherent dominant-component reversal. All 225 hard rows have
negative projection onto the true-gradient axis; median parallel scale is -0.343 and
median perpendicular/true norm is 0.216. The largest true component flips in 94.7% of
hard rows, all top three flip in 82.7%, and steering still contributes 93.41% of
negative alignment mass. Treat the blocker as multi-state representation/training
branch selection, not random 16D noise or a single bad action channel.

Next keep Actor frozen and use the same labels/network for B4 versus hard-example
balanced sampling, then balanced plus state-gradient metric/contrastive consistency.
Only if both fail should the v2 manifest drive targeted neighboring-time, recovery,
curvature or missing-context collection. Preserve the fresh-FD gate: median >=0.70,
P10 >=0, norm ratio >=0.50 before any Actor update.

### 2026-08-20 路线覆盖：永久冻结离线单次 Critic 梯度，转入持续在线 Actor--Critic pilot

本节是对本文早期 Critic/SAC 计划的最新覆盖说明；历史实验记录保留，不回删。

- **永久冻结**：冻结 Critic 后仅凭其梯度更新 Actor；以及在没有新增 Actor-visited 动作、真实 DBM/Query cost、Replay 写入和 Critic 持续重训的情况下，只做一次 Critic 梯度 Actor 更新。后续不得再把调 loss、调步长或挑 checkpoint 当作这两条路线的重启理由。
- **保留资产**：absolute-action Critic 的 value/ranking 能力、历史 43,200 条 full-action bank、hard-tail manifest、flat/stay 审计和 DBM 确定性 cost 管线。
- **唯一允许的 Critic 主线**：持续在线单步 Actor--Critic。Actor 每轮产生新动作，DBM/Query 立即给出真实 cost，经验写入 Replay，先更新双 Critic，再以较低频率更新 Actor；坏动作必须保留为负样本，不能只保留改善样本。
- **当前权限边界**：只放行 OAC-0 合同固化和 OAC-1（Actor 冻结的 Critic burn-in）；Actor 在 Critic 的 held-out 排序、幅值、flat/stay 和坏动作纠偏门全部通过前继续冻结。formal validation/test 继续封存。
- **实施合同**：见 [mppi_online_actor_critic_pilot_plan_20260820.md](mppi_online_actor_critic_pilot_plan_20260820.md)。主 review 的最终路线裁决见 `mppi_sampling_center_review_20260812.md` §11.83.7，Critic 专项解释见 `critic_gradient_value_assessment_20260817.md` §12。

### 2026-08-20 OAC-0/OAC-1执行交接

前两阶段已执行。OAC-0合同与独立validator通过；OAC-1三seed均未通过完整burn-in门，Actor始终
零更新并继续冻结。value/ranking主体通过（actor-visited pair accuracy约0.876--0.880），但
flat/stay的heldout recall只有0.320--0.565，初始排错坏动作纠正率只有0.332--0.446；2.8m/s
层排序约0.825，历史heldout bank还有1.7--2.5pp轻微遗忘。不得启动OAC-2。

如果继续，只运行Actor冻结的OAC-1B：local-span flat监督、未纠正pair/高速优先回放、历史bank
rehearsal。完整产物在`outputs/mppi_proposal/online_absolute_sac_oac01_20260820_v1/`，主结论见
review §11.84，实施细节见在线pilot计划§9。

### 2026-08-20 固定Replay学习曲线交接

第一项补救已完成：不增加数据、不改loss/标签，仅把Critic从200步训练到400/800/1600步。
800步后三seed的错误纠正率与2.8m/s门全部通过；1600步纠正率为0.634/0.666/0.584，历史heldout
bank还小幅恢复，证明OAC-1主体问题是训练预算不足。flat严格heldout false-stay仍有两个seed以
0.5--1.7pp未过，所以Actor与OAC-2继续冻结；是否修改flat监督留待后续讨论。详见review §11.85
和在线pilot计划§10。

### 2026-08-20 固定Replay扩展到6400步交接

在同一Replay、同一loss和同一次200步optimizer重置后连续扩展到3200/6400步。主Critic仍显著
改善：6400步初始排错纠正率为`0.762/0.780/0.734`，2.8m/s排序为
`0.921/0.925/0.927`，总体pair accuracy为`0.945/0.948/0.948`。所以200步明显不足，后续burn-in
至少1600步，默认优先比较3200与6400的heldout约束结果。

但flat严格heldout false-stay随训练从1600步`0.100/0.105/0.117`恶化到6400步
`0.142/0.145/0.158`。因此主Critic改善不等于OAC-2放行：Actor仍冻结，flat监督/校准作为后续
第二步单独讨论。独立validator已零误差复算18个checkpoint，Replay未变且本轮无新DBM rollout。
详见review §11.86与在线pilot计划§11。

### 2026-08-20 联合Value--移动系数交接

独立BCE flat头已完成替代性机制验证。Twin Value从6400步继续联合训练，辅助头直接读取Twin
Value/不一致度/动作差，监督连续
`c_move=max(delta J,0)/(max(delta J,0)+0.1)`；固定0.5对应原gap 0.1。

400/800/1600/3200共12个post-training点全部过门。最终3 seed的recall为
`0.923/0.908/0.905`，false-stay为`0.008/0.008/0.013`，2.8m/s排序
`0.935/0.932/0.941`，初始错误纠正率`0.801/0.810/0.776`，Value heldout bank无实质退化。
独立validator通过。

注意训练summary沿用旧通用校准门而误写FAIL，权威结果是V3目录的`analysis.json`和
`validator_report.json`。fold-0已消费，下一步先在新episode split按完全相同合同复核；在此之前
Actor仍冻结，不能直接进入部署。详见review §11.87与在线计划§12。

### 2026-08-20 outer fold-1联合机制复核交接

§11.87要求的新episode split复核已经完成。fold-1重新生成三seed各15360条Actor-visited Replay，
独立validator确认Actor checkpoint/hash不变、样本全来自fold-1训练episodes、DBM重放误差为0。
固定Replay Value续训到6400步后仍继续改善；再联合训练连续移动系数3200步，3/3 seed与全部
12个训练后checkpoint通过固定0.5物理门。

最终fold-1 recall=`0.922/0.922/0.898`，false-stay=`0.017/0.017/0.010`；heldout bank
pair=`0.946/0.951/0.945`，2.8m/s pair=`0.931/0.928/0.934`。权威qualification为
`JOINT_MOVE_COEFFICIENT_VALIDATION_PASS`。

这正式满足OAC-2入口条件，但本轮Actor update仍为0，不能把机制通过写成Actor收益。下一实现应
严格使用固定公式、0.05辅助权重、0.5操作点与6400步Value起点，进行训练split内低频小步Actor
更新；真实DBM cost控制selected晋级，所有新动作进入Replay，formal validation/test继续封存，
部署two-center guard不移除。详见review §11.88与在线计划§13。

```text
outputs/mppi_proposal/online_absolute_sac_oac01_fold1_20260820_v1/
outputs/mppi_proposal/online_absolute_sac_oac1_fixed_replay_fold1_20260820_v1/
outputs/mppi_proposal/online_absolute_sac_joint_move_coefficient_fold1_20260820_v1/
```

### 2026-08-24 OAC-2持续Actor--Critic机制pilot交接

fold-1首个Actor非冻结pilot已完成。每seed 20轮，每轮新增1536条真实DBM交互，Critic/Actor更新
比固定20:1；三seed合计101160次DBM评价。内部selection按完整episode从训练池隔离，formal
validation/test未加载。

最终selected均在第20轮，平均真实gain=`1.764/1.321/0.832`，bootstrap CI下界、median及
2.4/2.8m/s均为正，3/3 seed通过。Critic pair=`0.930--0.936`，连续系数recall=
`0.878--0.900`、false-stay=`0.012--0.018`；独立validator通过。

direct尾部仍回归：P05=`-0.700/-1.929/-2.321`、worst=
`-27.449/-12.583/-13.627`。因此当前只放行其余outer folds同合同复核，不能直接部署或跳过
two-center guard。后续不得基于fold-1重调超参数。详见review §11.89与在线计划§14。

```text
outputs/mppi_proposal/online_absolute_sac_oac2_fold1_20260824_v1/
```

### 2026-08-24 OAC-3完整3-fold交接

fold-0/1/2的同合同OAC-2均完成并独立验证。随后在各自outer-heldout 600状态上做真实DBM复算，
3/3 folds、9/9 seeds通过：mean gain=`0.687--1.938`，所有CI下界、median和2.4/2.8m/s门均为正。
资格为`OAC3_OUTER_HELDOUT_PASS_READY_FOR_SHORT_CLOSED_LOOP`，独立validator最大误差0。

direct尾部仍为负：P05最低`-3.524`，worst最低`-35.290`，因此下一步只允许warm-only与
warm+Actor two-center的短闭环A/B。Actor-only、移除guard和正式部署都未授权。详见review
§11.90与在线计划§15。

```text
outputs/mppi_proposal/online_absolute_sac_oac2_fold{0,1,2}_20260824_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_full_3fold_20260824_v1/
```

### 2026-08-24 OAC-2训练预算曲线交接

按用户要求暂停OAC-4闭环，fold-1同合同扩到100/200轮×3 seed并独立验证。20/100/200轮的三seed
平均gain为`1.306/5.875/9.953`，teacher headroom recovery为`1.51%/6.79%/11.51%`，Critic
pair保持`0.934/0.937/0.940`。因此20轮明显欠训练，Critic在当前局部范围不是第一阻塞。

同时direct P05为`-1.65/-5.94/-9.57`，worst为`-17.89/-64.52/-136.89`；200轮Actor mean
`81.84`仍差于warm `29.17`和teacher `5.16`，只有median `17.86`优于warm median。下一步应先做
tail-aware Actor目标和selected gate，不直接提高LR；所有坏动作继续进Replay，Critic继续在线
更新。OAC-4保持暂停。

```text
scripts/model_verify/analyze_mppi_oac2_training_budget.py
outputs/mppi_proposal/online_absolute_sac_oac2_budget_20260824_v1/analysis.json
```

### 2026-08-24 OAC-2 200轮尾部归因交接

200轮`median=17.861`是三seed median的平均，pooled median为`17.842`。mean也从初始`91.796`
改善到`81.844`，但仍被初始重尾拉高；最高5%的最终样本贡献45.2%总cost。回归样本比例没有随
预算增加，始终约31%，但坏case高度持续：100→200轮最差5%的Jaccard为0.58--0.82，按seed平均
有137/600状态差于初始且继续恶化。

严重回归集中在高速层，且83个严重seed-state中81个由position cost主导。最终Twin Value对严重
坏端点有90.4%能判断为变差，因此下一阻塞是Actor batch均值风险分配与selected gate缺少尾部
约束，不是继续单独堆Critic步数。下一实验应保持在线Replay/Critic更新，在Actor目标加入相对
selected center的soft regression/CVaR项，并给checkpoint增加P05与高速尾门；two-center guard
和formal validation/test封存保持不变。

```text
scripts/model_verify/analyze_mppi_oac2_budget_tail_cases.py
outputs/mppi_proposal/online_absolute_sac_oac2_budget_tail_20260824_v1/
```

### 2026-08-24 OAC-2T固定尾部CVaR交接

已将相对selected center的保守Twin Value正回归top-10% CVaR加入Actor loss，并完成固定权重
10/100各200轮×3 seed。尾部机制有效：权重10将平均P05从`-9.57`提高到`-0.50`、worst从
`-136.89`提高到`-9.81`、回归比例从31.28%降到19.11%。但mean gain仅`1.39`，只保留原方案
14.0%，guard J也退化约0.67；权重100没有进一步收益。两臂全部周期checkpoint均通过tail floor，
Critic pair稳定约0.94，因此损失来自固定zero-margin CVaR过度正则，不是选模门或Critic退化。

两臂均不选用。下一步若继续，改成非零material margin和自适应Lagrange tail预算，不再扫固定
权重；two-center guard、在线Replay/Critic更新及formal validation/test封存保持不变。

```text
scripts/model_verify/analyze_mppi_oac2_tail_cvar_ab.py
outputs/mppi_proposal/online_absolute_sac_oac2_tailcvar_ab_20260824_v1/
```

### 2026-08-24 多候选与DBM task-loss交接

K=1/K=4 clean no-anchor G-X多elite实验已完成并独立重放通过。K=4输出确有分散（候选间距离中位
约`0.144 sigma`），但OOF oracle recovery只从K=1的`-3.120`改善到`-2.953`；absolute-value
Critic selector接近oracle却同样不可用。当前不启动Diffusion，多模态不是主要数量级阻塞。

同一K=1 checkpoint直接反传真实DBM J50后，tiny-32/128 recovery=`0.827/0.898`，完整fold-0
train=`0.834`，episode-heldout=`0.779`；heldout J从`91.05`降到`23.61`，但仍远离J16
`4.46`且有8.5%回归。J16中25.3%分量、91.3%状态超出当前`center +/- 1 std`输出盒，下一步先做
输出支撑严格A/B；逐维投影cost不是盒内oracle，不得当成上界。

DBM task-loss只授权为结构诊断/预训练，不能用于Query部署训练。Critic梯度provider继续永久冻结；
Critic仅保留持续在线Value/ranking和候选选择用途。formal validation/test与闭环继续封存。

```text
outputs/mppi_proposal/j16_multi_candidate_actor_20260824_v1/
outputs/mppi_proposal/dbm_task_loss_actor_20260824_v1/
```

### 2026-08-25 DBM task-loss输出支撑A/B交接

已在fold-0 train-only分层128状态完成严格配对的`box1/box3/full`输出支撑实验。三臂初始action
最大差`1.19e-7`，训练预算均为1200次DBM J50更新；最终mean J为
`15.568/5.483/5.498`，J16为`4.668`，recovery为`0.8977/0.9924/0.9922`。`box1`有77.3%
状态触及至少一个95%边界，`box3`无边界占用，说明原`center +/- 1std`是主要结构瓶颈，扩大到
`+/-3std`已经足够，开放full physical域没有额外收益。独立DBM replay最大指标误差为0。

当前只放行完整train-1200与episode-heldout-600的`box1`--`box3`复核；尚未修改OAC Actor，未
启动闭环，也未消费formal validation/test。若完整复核保持收益，再将`box3`映射带回持续在线
Actor--Critic，Critic/Replay/two-center guard合同不变。

```text
scripts/model_verify/run_mppi_dbm_task_loss_support_ab.py
scripts/model_verify/validate_mppi_dbm_task_loss_support_ab.py
outputs/mppi_proposal/dbm_task_loss_support_ab_20260825_v1/
```

### 2026-08-25 输出支撑完整fold复核交接

已完成fold-0 train-1200/episode-heldout-600的`box1`--`box3`严格配对复核，训练预算均为2400次
DBM J50更新且heldout不参与选模。train mean J从`19.792`降到`7.282`，heldout从`23.580`
降到`10.300`；recovery分别从`0.8312/0.7792`升到`0.9741/0.9326`。heldout P95/worst从
`91.11/817.54`降到`33.06/455.34`，P05 gain没有退化，回归比例仅增加`1.17pp`。`box1`
在train/heldout有60.2%/62.5%状态触边，`box3`均为0。独立DBM replay全部指标误差为0。

`box3`现已放行进入持续在线OAC的固定状态配对pilot，但尚未集成、未做闭环、未使用formal
validation/test。集成时必须保持Critic、Replay、continuous coefficient、tail gate及two-center
guard不变，并通过zero-initialized adapter保证父Actor初始action严格一致；不得把本轮可微DBM训练
checkpoint直接当作Query部署模型。

```text
scripts/model_verify/run_mppi_dbm_task_loss_support_full_ab.py
scripts/model_verify/validate_mppi_dbm_task_loss_support_full_ab.py
outputs/mppi_proposal/dbm_task_loss_support_full_ab_20260825_v1/
```

### 2026-08-25 `box3` OAC support-only交接

`box3`已在fold-1持续在线OAC中完成200轮×3 seed严格配对。没有使用旧固定CVaR loss，只保留
overall/2.4/2.8m/s P05 checkpoint floor。两臂初始action最大差`4.17e-7`，独立validator均
3/3 seed通过且DBM replay误差为0。

tail-safe selected mean gain三seed平均从`box1=0.995`升到`box3=1.514`，200轮latest从
`10.171`升到`15.543`；但selected round均为`20/10/10`，selected P05/worst从
`-1.060/-14.72`变为`-1.407/-20.44`。`box3`解除全部输出触边且Critic pair保持约0.94，故默认
输出合同采用`+/-3std`；下一阻塞是state-wise tail，不是继续扩大动作范围。

下一轮若继续，以`box3`为固定基线，验证nonzero material margin与自适应Lagrange/CVaR预算；不得
恢复固定`lambda=10/100`，不得删除坏Replay。two-center guard、P05/高速floor、formal
validation/test封存和OAC-4暂停均保持。

```text
outputs/mppi_proposal/online_absolute_sac_oac2_support_box1_200round_20260825_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_support_box3_200round_20260825_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_support_ab_20260825_v1/
```

### 2026-08-25 `+/-3std`强合同与J口径交接

新OAC G-X训练现默认`actor_output_support_multiplier=3.0`。`+/-1std`只能配合
`--allow-box1-ablation`做显式消融，否则训练拒绝启动；新run合同和latest/selected checkpoint均需
固化并由validator核对multiplier。历史无buffer checkpoint仍可兼容读取，full physical域不是默认。
该修改没有放宽P05/高速tail floor、two-center guard、坏动作Replay和数据封存纪律。

当前fold-1、600个internal-selection状态的三seed J口径如下（越低越好）：warm mean `29.174`；
tail-safe selected box3 Actor direct mean/median `90.283/20.224`，加入two-center guard后mean
`19.387`；200轮latest direct mean/median `76.254/17.092`，guard mean `18.538`，但其P05/worst
gain=`-10.572/-198.239`，因此未晋级。candidate-bank best参考J=`5.161`。对外汇报当前可用结果应
使用selected+guard的`19.387`，不能把latest `18.538`写成已通过结果。fold-0可微DBM容量诊断的
heldout box3/J16=`10.300/4.463`属于另一split与oracle梯度口径，只说明还有结构/优化headroom。

### 2026-08-25 OAC-2A自适应tail预算交接

已在固定box3、fold-1、200轮×3 seed完成非零margin加自适应Lagrange/CVaR预算实验。lambda在
round `47/28/30`激活，最大仅`1.163/1.406/0.941`，独立validator对dual递推、checkpoint、Replay
与DBM指标全部3/3通过且误差为0。相对无tail box3，latest gain P05从`-10.572`改善到`-2.126`、
worst从`-198.239`改善到`-111.848`、回归比例从30.39%降到21.33%；mean gain从`15.543`降到
`10.749`，保留69.16%。selected round为`20/80/10`，只有1/3 seed延长，故不放行OAC-4。

下一步为零rollout的state-wise dual校准：检查预测tail超额对真实回归的recall/false-positive及
速度×状态分层，再决定是否做state-conditioned风险系数。不得直接扫margin/budget/dual LR；3std、
checkpoint floor、two-center guard和坏动作Replay继续保留。

```text
outputs/mppi_proposal/online_absolute_sac_oac2_box3_adaptive_tail_200round_20260825_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_adaptive_tail_ab_20260825_v1/analysis.json
```

### 2026-08-25 OAC逐状态tail归因与连续风险回拉交接

adaptive-tail逐状态复算确认：Critic排序可用（真实变化Pearson `0.782`、material AUC `0.941`），
但`0.05`门只召回69.3% material回归，且全局top-10%均值budget不提供逐状态保证。selected的495个
回归seed-state中latest修复193个、持续302个，并新增82个；持续/新增组initial J中位仅
`9.159/8.406`，问题集中在低headroom状态误动。17个严重回归全部位于2.4/2.8m/s，16/17由
position cost主导。

随后完成连续state-wise风险回拉200轮×3 seed。selected round改善为`20/80/60`，selected mean
gain=`4.019`、guard J=`19.212`；但latest P05/worst与高速P05均比adaptive-tail更差，严重回归增至
29。判定`STATEWISE_RISK_SHRINK_FAIL_NO_FURTHER_WEIGHT_SWEEP`：保留selected内部候选，不继续扫
loss权重，OAC-4/formal validation/test仍冻结。若重启tail，改用离线逐状态安全投影标签，而不是
共享Actor loss加权。

### 2026-08-25 非均匀8-knot布局交接

已比较uniform `[0,7,14,21,28,35,42,49]`、mild前密
`[0,3,7,12,18,26,35,49]`和strong前密。100状态同预算128次真实重搜中，mean J为
`7.641/8.528/9.434`，前密明显失败；严格配对随机起点的可微容量oracle则为
`4.793/4.728/4.802`，说明mild前密仅有约1.35%的理论容量收益，但当前uniform warm/Actor投影与
局部搜索无法实现。保持runtime uniform默认；
除非未来同时重做前密布局原生warm、search covariance和Actor标签并通过tail门，否则不切换。

```text
outputs/mppi_proposal/online_absolute_sac_oac2_statewise_risk_ab_20260825_v1/analysis.json
outputs/mppi_proposal/knot_layout_search_ab_20260825_v3/summary.json
outputs/mppi_proposal/knot_layout_capacity_oracle_20260825_v2/summary.json
outputs/mppi_proposal/knot_layout_ab_validation_20260825_v2/validator_report.json
```

### 2026-08-25 Pairwise Delta在线OAC接入交接

已将§11.105的反对称Twin Pairwise Delta接入OAC-2，但保持Absolute Twin主Actor梯度、Replay、DBM
接受门和two-center guard不变。Pair冻结Actor预训练1600步，在线每轮20步；独立validator验证
`delta(a,a)=0`、交换反号、checkpoint/optimizer/update计数和DBM重放，全部通过。

原adaptive dual从0开始时三seed均未激活，Pair/Absolute Actor结果等价，证明没有旁路污染。非零dual
机制压力A/B中，Pair使三seedmean/median gain平均提高`0.171/0.064`，但P05/worst、高速P05和回归
比例均变差。判定`PAIR_DELTA_ACTIVE_TAIL_MIXED_NO_FORWARD_AUTHORIZATION`：实现保留且默认关闭，
不启动200轮或outer-fold扩展，不用Pair替换Absolute tail风险。后续最多作为候选排序辅助或与
Absolute风险取保守并集重新做短pilot；OAC-4/formal validation/test继续冻结。

```text
outputs/mppi_proposal/online_absolute_sac_oac2_pairdelta_ab_analysis_20260825_v1/analysis.json
outputs/mppi_proposal/online_absolute_sac_oac2_active_tail_pairdelta_smoke_20260825_v1/pair_delta_validator_report.json
```

### 2026-08-25 当前OAC Critic与DBM真实梯度差距交接

已对adaptive-tail 200轮最终Critic做600个internal-selection状态、selected/latest Actor、3 seed
只读审计。当前latest动作处Value Pearson `0.975`，action-gradient cosine median/P10
`0.979/0.726`，norm ratio `1.128`；early steering P10 `0.835`，flat最低梯度四分位P10 `0.509`。
Actor-visited Replay持续更新后，Critic主体局部梯度已经恢复，历史静态bank-best梯度失败不应直接
外推到当前checkpoint。

Critic仍有tail：固定`0.002σ`逐状态动作步下，真实方向回归0%，Critic方向回归4.17%。但更大的
已测差距在目标与预算：Critic对准确`mean log1p(J)`的Actor参数梯度cos为
`0.927/0.980/0.964`，准确log与DBM容量实验使用的raw-J梯度只有`0.718/0.359/0.325`；OAC每轮
实际移动仅约`0.00021--0.00028σ`、200步，而DBM容量实验为2400步、LR大两个数量级。

判定`CRITIC_LOCAL_GRADIENT_MOSTLY_RECOVERED_OBJECTIVE_AND_ACTOR_UPDATE_CONTRACT_DOMINATE_GAP`。
下一步先冻结Critic做`gamma=0/0.5/1`的raw-cost-aware Actor聚合单变量短pilot，再单独讨论Actor
步长/预算；不继续扩大Critic或扫gradient loss。DBM gate、two-center guard、formal/test封存不变。

```text
outputs/mppi_proposal/oac2_critic_dbm_gradient_gap_20260825_v3/
```

### 2026-08-25 OAC-2C raw-cost-aware Actor聚合交接

已完成`gamma=0/0.5/1`、20轮×3 seed单变量pilot。Critic仍学`log1p(J)`，Actor通过脱梯度
`exp(gamma*q)`权重恢复raw-J跨状态梯度；cap=2048、batch归一，其余OAC合同不变。latest mean gain
为`2.049/2.764/3.059`，gamma=1在3/3 seed提高mean；但worst从`-24.67`降至`-28.61`，selected
worst也只有1/3 seed改善。gamma=1有效样本比例约0.249，说明高cost集中带来真实平均收益，也带来
seed敏感tail。

判定`RAW_COST_AGGREGATION_IMPROVES_MEAN_BUT_TAIL_MIXED_SHORT_PILOT`。后续预算/步长pilot用gamma=1
作主体、gamma=0.5作保守对照；不得取消DBM checkpoint、P05/worst高速floor或two-center guard。

```text
outputs/mppi_proposal/oac2_raw_cost_aggregation_ab_20260825_v2/analysis.json
```

### 2026-08-27 OAC-2C raw聚合200轮交接

gamma=0.5/1已完成200轮×3 seed，均通过独立validator；gamma=0复用历史200轮。latest mean gain为
`10.749/13.740/14.828`，证明20轮预算明显过少；但gamma=1相对gamma=0的median/P05/worst分别
变化`-0.433/-0.304/-73.67`，2.8m/s P05变化`-1.595`，tail随长跑恶化。

最终selected mean gain为`2.758/3.565/6.394`，但two-center guard J为
`19.296/19.325/19.314`，没有部署收益。判定
`RAW_MEAN_LONG_RUN_DIRECT_GAIN_UP_GUARDED_GAIN_FLAT_TAIL_WORSE`：停止原样加轮数。下一步先做
raw权重×warm选择归因，再做gamma=1 Actor batch128→512的单变量方差实验；部署目标若优先，改测
warm-relative margin+clip advantage，而不是继续提高raw gamma。

```text
outputs/mppi_proposal/oac2_raw_cost_budget_curve_20260827_v1/analysis.json
```

### 2026-08-27 跨§11.91--11.109复核后的交接修正

只看raw聚合200轮会漏掉关键容量对照：box3可微DBM task-loss已在train/episode-heldout达到
`J=7.282/10.300`，相对J16恢复`97.41%/93.26%`；因此当前Actor结构与3std支撑不是OAC只恢复
`17.1%`的主解释。最终OAC Critic在Actor当前访问动作上的梯度cosine median/P10为
`0.979/0.726`，说明主体局部梯度也已恢复，但仍有约4--5%的小步额外回归tail。fixed CVaR、adaptive
dual、statewise shrink和Pairwise在线风险接管均已测试，不能再把共享loss权重扫描当作首选。

下一首要实验改为matched DBM-vs-Critic gradient-source trajectory A/B：固定box3初始Actor、
raw-mean目标、状态batch、optimizer、LR、更新数和投影，只切换梯度来源，并全部用真实DBM复算到J16的
恢复率及mean/median/P05/worst。若同预算DBM臂也慢，处理Actor预算/LR；若DBM臂快而Critic臂慢，
处理Critic误差累积与更新时序。大batch ESS诊断排第二，warm-relative/guard-aware loss仅保留为部署
安全支线。formal validation/test和闭环继续封存。

### 2026-08-27 OAC初始大LR衰减交接

已完成CUDA严格配对：Actor LR由固定`1e-6`改为round 1--20的`1e-5`，随后余弦衰减到round 80
的`3e-6`和round 200的`1e-6`；其余gamma=1/box3/adaptive-tail/20:1/Replay/trust合同不变。
200轮latest mean/median J从`76.968/18.825`改善到`69.409/17.953`，headroom recovery从
`17.09%`提高到`25.83%`，回归比例下降4.22pp，3/3 seed主体改善。实际动作步从前20轮平均
`0.00309sigma`衰减到后120轮`0.000268sigma`，无trust投影；Critic pair维持`0.92--0.95`。

负面结果同样决定性：2.4/2.8m/s P05在3/3 seed全退，2.8平均为`-12.913`，平均worst
`-212.895`；所有checkpoint的2.8m/s门失败，selected=`0/0/0`。独立CUDA validator所有工程/
复算检查逐seed通过且DBM误差0，最终FAIL只来自性能门0/3。正式结论为“固定LR过小已证实，但
大LR衰减只改善主体、不授权部署”。CPU首轮因沙箱设备回退不进入正式A/B。

后续保留matched DBM-vs-Critic梯度源A/B；LR调度若再试，只做一次较温和的单变量收紧，否则直接
转逐状态DBM/bank safe projection。formal validation/test、闭环与two-center guard边界不变。

```text
outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_lrdecay_cuda_200round_20260827_v1/
outputs/mppi_proposal/oac2_lr_decay_ab_20260827_v1/analysis.json
```

### 2026-08-27 OAC Actor LR边界扫描交接

已在CUDA上完成初始LR `2e-5/4e-5/8e-5/1.6e-4/3.2e-4`的90轮×3 seed扫描，并与既有
`1e-5`臂在round 80严格比较。除初始LR外，gamma=1、box3、adaptive-tail、Replay、Critic 20:1、
batch、raw目标与`0.02sigma` trust均固定。J16 headroom recovery从`1e-5`的19.66%单调提升到
`3.2e-4`的56.17%，mean J从74.781降到43.176；这证明此前主体训练明显受更新步长限制。

有效边界已经找到：`8e-5`投影率26.7%，`1.6e-4`为76.7%，`3.2e-4`为100%。最后翻倍只增加
2.82pp恢复，median从13.589略退到13.688、回归比例从18.67%升到21.33%，因此停止继续提高LR。
后续主体probe使用`1.6e-4 + 0.02sigma trust`，`3.2e-4`只作饱和证据。

tail仍未解决且随大步更新恶化：`1.6e-4`的2.8m/s P05为`-51.386`，所有新增臂selected均为0，
性能门0/3。五个新增validator的工程、DBM复算和封存检查全部通过。下一步先做matched DBM-vs-Critic
梯度源A/B解释剩余约44%主体差距；安全单列到逐状态DBM/bank safe projection和two-center guard，
不再扫共享tail loss。formal validation/test及闭环继续封存。

```text
scripts/model_verify/analyze_mppi_oac2_lr_boundary_scan.py
outputs/mppi_proposal/oac2_lr_boundary_scan_20260827_v1/analysis.json
```

### 2026-08-27 OAC全局trust边界扫描交接

固定初始LR=`3.2e-4`完成trust `0.02/0.04/0.06/0.08sigma`的CUDA、三seed、round-80严格对照。
恢复率为`56.17/64.21/63.89/64.73%`，mean J为`43.176/36.181/36.461/35.718`，median J为
`13.688/12.415/12.613/12.626`。因此`0.04sigma`已经进入主体平台；`0.08`的mean仅比`0.04`
好0.463且只2/3 seed改善，不能支持继续放宽。

Critic pair保持约`0.908--0.910`，`+-3std`输出支撑几乎不触界，平台不是Critic立即崩溃或动作盒饱和。
tail仍是独立阻塞：P05随放宽从`-8.27`总体降至`-12.13`，所有臂selected=0、性能门0/3；worst和
高速P05受少数case影响不严格单调，不得当作安全改善。

后续主体合同固定`3.2e-4 + 0.04sigma trust`，做matched DBM-vs-Critic gradient-source A/B；
安全转逐状态DBM/bank safe projection + two-center guard。停止继续提高全局LR/trust。formal
validation/test和闭环继续封存。

```text
scripts/model_verify/analyze_mppi_oac2_trust_boundary_scan.py
outputs/mppi_proposal/oac2_trust_boundary_scan_20260827_v1/analysis.json
```

### 2026-08-27 `6.4e-4 × 0.06sigma`交叉臂交接

唯一新增交叉臂已完成：trust固定`0.06sigma`，初始LR由`3.2e-4`翻倍到`6.4e-4`。round-80 mean J
`36.461→33.942`，恢复率`63.89%→66.75%`，证明低LR臂未充分用满trust；但mean/headroom仅2/3
seed改善，median `12.613→13.359`，回归比例`24.22%→28.67%`。实际前20轮步长`0.0587sigma`，
投影率75%，Critic pair仍约0.908。

结论为“接近联合边界但非严格硬上限”。稳定工作点继续使用`3.2e-4 + 0.04sigma`；高LR交叉臂仅作
mean-capacity probe。下一主项仍是matched DBM-vs-Critic梯度源A/B，安全独立走逐状态projection和
two-center guard。所有checkpoint性能门0/3，formal validation/test与闭环保持封存。

```text
scripts/model_verify/analyze_mppi_oac2_lr_trust_cross_ab.py
outputs/mppi_proposal/oac2_lr_trust_cross_ab_20260827_v1/analysis.json
```

### 2026-08-27 采样中心warm-relative评价口径交接

通用策略中心评价已改为：只对未加噪的单一Actor center做确定性rollout，并与相同状态的warm center
配对；不混入MPPI采样、wrapper/softmax、two-center执行cost或oracle。DBM与Query均使用各自的
`G_w=J_warm-J_actor`，报告胜率、median、P05/worst、聚合改善及速度/场景分层；跨模型再报改善符号
一致率。`J16 recovery`只作DBM容量旁证。

最新`6.4e-4 + 0.06sigma`、round-90、600状态×3 seed结果：warm mean/median J为
`29.174/23.356`，Actor mean/median J为`33.849/13.335`；Actor严格超过warm比例`69.28%`，
配对gain median `+6.726`，但gain mean `-4.675`、聚合相对改善`-16.0%`。2.8m/s Actor mean J
`77.733`，差于warm `47.301`。这说明典型状态明显获益，但约31%未赢warm且少数损失幅值足以拖累
聚合结果。

必须避免旧口径误读：已有`gain_vs_initial/regression_fraction/gain P05/worst`都相对初始Actor，
不是warm；§11.114的`28.67%`也不是warm-relative回归率。当前summary未保存未截断逐状态
`gain_vs_warm`，所以无法从旧产物恢复warm-relative P05/worst。下一评测必须落盘逐状态
`warm_cost/actor_cost/gain_vs_warm/episode/speed/scenario`，再进行DBM/Query一致性比较。

```text
outputs/mppi_proposal/absolute_action_value_critic_20260820_v1/candidate_bank.npz
outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_lr64e5_trust006_90round_20260827_v1/
```

### 2026-08-27 warm-relative正式重评分与gamma A/B交接

§11.115/计划§39要求的完整逐状态中心重评分已经实现并独立验证。评价只比较固定状态上的未加噪warm
center和未加噪Actor center；不运行MPPI采样、wrapper/softmax、two-center输出或oracle。warm是
随机MPPI历史实现产生的**外部比较样本**，绝不进入Actor输入、loss、reward、anchor或训练权重。

现有五组Actor的完整warm-relative结果已写入
`oac_warm_relative_centers_20260827_v1`。其中高尺度gamma1臂`6.4e-4 + 0.06sigma`最佳：胜warm
69.28%，gain median +6.748、P05 -99.976、worst -606.039、Actor mean J 33.849、聚合改善-16.0%。
稳定臂为67.83%、+6.602、-114.410、-904.064、36.118、-23.8%。所以高尺度臂相对同一warm的
尾部也更好；旧initial-relative tail字段不能再用于推断warm-relative尾部。

唯一新增训练A/B固定高尺度其余全部合同，只把绝对cost gamma从1降到0.5。gamma0.5结果为67.67%、
+6.101、-127.115、-821.581、38.983、-33.6%，所有pooled门均退化，逐seed median/P05/worst
0/3改善。正式保留gamma1并停止gamma扫描；不启动warm-relative训练。

当前边界：高尺度gamma1只是现有最优单中心proposal，仍因聚合gain为负而不能替代warm或授权部署。
two-center guard继续作为MPPI候选层安全结构，formal validation/test和闭环保持封存。

```text
scripts/model_verify/evaluate_mppi_oac_warm_relative_centers.py
scripts/model_verify/validate_mppi_oac_warm_relative_centers.py
scripts/model_verify/analyze_mppi_oac_warm_relative_gamma_ab.py
outputs/mppi_proposal/oac_warm_relative_centers_20260827_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_gamma05_lr64e5_trust006_90round_20260827_v1/
outputs/mppi_proposal/oac_warm_relative_gamma05_high_ab_20260827_v1/
```

### 2026-08-27 K=1/4/8 Actor microstep交接

已把每个outer round一次Actor更新扩展为严格注册的K=1/4/8 microstep。20次Critic更新总量、
`0.06sigma RMS`累计round trust、gamma=1、`+-3std`、90 outer rounds与fold-1状态均固定；LR与单步
trust按K除法，temperature和tail dual仍各更新一次。独立validator逐seed的工程、Replay、DBM复算、
计数和trust检查全部通过，formal validation/test未加载。

round-90 headroom recovery为`0.6562/0.6627/0.7101`，K=8三seed为
`0.7116/0.7089/0.7098`。warm-relative direct-center胜率为`67.39%/70.78%/74.56%`；K=8的median
gain `+7.638`、P05 `-84.558`，均优于K=1的`+6.355/-102.272`。但K=8 mean gain仍为`-1.101`、
worst `-832.075`，selected仍是round0、性能门0/3。

结论：K=8机制通过并部分闭合Actor优化差距，后续OAC默认使用K=8；下一步保持累计trust与其他合同，
只增加outer Actor-visited refresh轮数。暂不扫K>8，不授权单中心替换warm、部署、formal/test或闭环；
two-center guard仍保留为最终候选层安全结构。

```text
scripts/model_verify/analyze_mppi_oac2_multiupdate_ab.py
outputs/mppi_proposal/oac2_multiupdate_ab_20260827_v1/analysis.json
outputs/mppi_proposal/oac2_multiupdate_warm_relative_20260827_v1/
```

### 2026-08-28 K=16/32 Actor microstep边界交接

边界探索已从K=8继续到K=16和K=32，其他合同不变。round-90 recovery为
`0.7101/0.7411/0.7492`；K=32相对K=16只增加0.81pp。warm-relative聚合改善为
`-3.78%/+5.39%/+7.86%`，胜率为`74.56%/76.11%/75.72%`，P05为
`-84.56/-69.67/-56.01`。

结论：K=16是效率拐点和默认训练配置；K=32是训练成本次要时的质量/tail候选。K=64不再执行，因为
主恢复增量已低于1pp，且K=32已经超过每轮20次Critic refresh。下一步转Actor-visited状态刷新与Replay
新鲜度。所有K=16/32工程validator逐seed通过，但旧性能门仍0/3、selected仍round0；formal/test、
闭环和two-center guard边界不变。

```text
outputs/mppi_proposal/oac2_multiupdate_boundary_20260828_v2/analysis.json
outputs/mppi_proposal/oac2_multiupdate_warm_relative_20260828_v3/
```

### 2026-08-28 K=20严格1:1交错交接

已补K=20、90轮×3 seed严格`1 Critic : 1 Actor`实验。结果没有优于K=16：recovery
`0.7400 vs 0.7411`，胜warm `75.83% vs 76.11%`，mean gain `1.481 vs 1.573`，聚合改善
`5.08% vs 5.39%`。K=20只把median/P05从`7.889/-69.671`小幅改善到`8.109/-68.597`，worst和
seed方差略差。

更新次数相等不是额外优势。最终默认保持K=16，K=32作为离线mean/P05质量臂；K扫描到此结束。
下一主项为Actor-visited refresh/Replay新鲜度，旧性能门、two-center guard和sealed split保持不变。

```text
outputs/mppi_proposal/oac2_multiupdate_boundary_20260828_v3/analysis.json
outputs/mppi_proposal/oac2_multiupdate_warm_relative_20260828_v4/
```

### 2026-08-28 Actor-visited刷新频率等预算A/B交接

已把K=16的每个完整round严格拆成两个half-round，比较相同总预算下反馈延迟是否重要。基线为
`90×(256 contexts,20 Critic,16 Actor)`，刷新臂为`180×(128,10,8)`；总context/Critic/Actor分别
保持`23040/1800/1440`，每microstep LR/trust及90次temperature/tail更新也相同。三seed工程、Replay、
DBM复算与调度回放全部通过，formal/test未加载。

结果为中性偏负：recovery `0.7411→0.7360`；胜warm `76.11%→75.56%`；gain mean
`1.573→1.134`；median `7.889→8.064`；P05 `-69.671→-71.800`；worst
`-640.281→-693.762`；聚合改善`5.39%→3.89%`。只有median微增，不能抵消其余退化。

结论：现有Replay已抽50% recent rows，单纯提高Actor-visited刷新频率不是主瓶颈。保持K=16/90轮，
不再扫刷新频率；下一实验需要提高新状态/动作数据的信息量，而不是把同预算拆得更碎。性能门仍0/3，
two-center guard、formal/test和闭环边界不变。

```text
scripts/model_verify/analyze_mppi_oac2_refresh_cadence_ab.py
outputs/mppi_proposal/oac2_refresh_cadence_ab_20260828_v1/analysis.json
```

### 2026-08-28 Matched DBM-vs-Critic gradient-source交接

已在原K16 OAC主循环内增加注册的`actor-gradient-source=dbm`机制臂，不是复用旧的非配对task-loss
结果。DBM臂与Critic臂保持相同初始Actor、fold-1状态、90轮、20 Critic/16 Actor更新、Replay、
staged LR、`0.06sigma` trust、gamma=1、continuous coefficient、entropy与adaptive tail；仅将Actor的
sampled-action value及mean-vs-selected tail梯度改为可微DBM J50。Critic仍持续更新并记录全部
Actor-visited真实cost。

三seed recovery从`0.7361/0.7466/0.7406`升到`0.7687/0.7620/0.7490`，均值
`0.7411→0.7599`（+1.88pp）。warm胜率`76.11%→76.89%`，gain mean`1.573→3.204`，P05
`-69.671→-59.566`，聚合改善`5.39%→10.98%`；但worst`-640.3→-869.5`。

结论：Critic误差有可测累计代价，但只关闭约9.8%的历史容量参考差距；真实DBM梯度在相同OAC合同下
也平台约0.76。剩余主阻塞是共享Actor的跨状态梯度冲突与stochastic/coefficient/LR/trust优化合同，
不是继续扩大Critic。下一优先做共享参数梯度分层冲突诊断，再决定deterministic mean-action surrogate
或逐状态proposal监督。性能门仍0/3，two-center guard、formal/test和闭环保持封存。

```text
scripts/model_verify/analyze_mppi_oac2_matched_gradient_source_ab.py
outputs/mppi_proposal/oac2_matched_gradient_source_ab_20260828_v1/analysis.json
```
