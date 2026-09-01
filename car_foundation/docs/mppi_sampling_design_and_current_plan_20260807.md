# MPPI 采样设计与当前计划（2026-08-07 汇总）

本文是对 `car_foundation/docs/` 下 MPPI 采样中心系列文档的一次性汇总，用于回答两个问题：
**当前 MPPI 采样是怎么设计的**，以及**接下来要做什么**。

本文是**汇总视图，不是权威入口**。执行层面的唯一权威文档是
`mppi_sequential_probe_execution_plan_20260806.md`（含 gate 表、seed ledger、reward 契约）。
本文与其冲突时以该文档为准。

## 0. 阅读顺序

| 顺序 | 文档 | 作用 |
| --- | --- | --- |
| 1 | `mppi_sequential_probe_execution_plan_20260806.md` | **权威执行入口**：目标函数、gate、seed 台账 |
| 2 | `mppi_sampling_center_handoff_20260802.md` | 主交接文档，历史脉络与失败记录 |
| 3 | `rl_mppi_sampling_center_design_20260731.md` | 采样中心的 RL 建模设计 |
| 4 | `mppi_teacher_label_plan_20260803.md` | Teacher 标签（T1）生成与审计 |
| 5 | `mppi_closed_loop_dataset_collection_20260802.md` | 数据集与切分纪律 |
| 6 | `mppi_sampling_optimization_status_20260730.md` | 离线多阶段引导实验（未上线） |
| 7 | `query_mppi_pytorch_onnx_20260730.md` | Query 模型 rollout 协议与延迟 |

JAX 相关内容已明确停止维护，仅 PyTorch / ONNX 路线有效。

---

## 1. 基线 MPPI 采样设计

### 1.1 参数

| 项 | 取值 |
| --- | --- |
| 控制参数化 | 8 knots × 2 通道 = 16 维，线性插值（`align_corners=True`）到 50 步 |
| dt / horizon | 0.05 s / 2.5 s |
| 历史长度 | 250 步（12.5 s） |
| 候选数 | 256 |
| 采样噪声 | `noise_sigma = [0.25, 0.35]`（accel / steer） |
| 动作范围 | [-1, 1] |
| 温度 λ | 1.0（softmax 加权，无 top-k / elite） |
| seed | 3407 |
| 候选 0 | 保留 warm start，不加噪 |
| warm start | 上一轮最优序列整体左移一步 |

代价函数：

```
J = 5.0·pos² + 5.0·wrapped_yaw² + 1.0·vx² + 0.05·Δaccel² + 0.10·Δsteer²
```

yaw-rate 权重为 0。

### 1.2 为什么要改采样中心

DBM clean，step 340 快照，256 高斯候选：

| 指标 | 值 |
| --- | --- |
| best cost | 5.0854864 |
| median | 253.9471 |
| P95 | 1705.3604 |
| **ESS** | **1.4419617** |
| best 权重 | 0.8111926 |
| best + mean 占总权重 | 99.9517 % |

ESS≈1.44 / 256 意味着**采样预算几乎完全浪费**：256 条 rollout 里只有一条真正参与决策。
结论是不去改 MPPI 求解器本身，而是**学习采样分布的均值（sampling center）**：

```
μ* = argmin_μ  E_ε[ C_MPPI({ μ + σ·ε_i }) ]
```

即 amortized optimization——学的是"提案分布往哪放"，不是"最优动作是什么"。

### 1.3 固定候选库 `fixed-hadamard-64-v1`

为让不同方法可比，冻结了一组候选方向：

- 16×16 Sylvester Hadamard 构造，对偶（antithetic）成对
- `sampling_mode=fixed_hadamard_64`，要求 `num_samples=64`
- contract hash `0fd540206ec988f565481d25f9cbd0b4051f7847e154565d9eac4e7445b6a492`

注意：这是**第二层辅助指标**，不是主目标。主目标见 §2.1。

---

## 2. 采样中心策略设计

### 2.1 唯一 reward 契约（2026-08-06 修订）

```
c      = policy(state)                      # [8,2] knots
A      = LinearInterpolate(c, horizon=50)
J_dir  = Cost(DBM(state, A))
reward = J_dir(anchor) − J_dir(c)
```

**不使用 reward seed。** 这条修订解决了一个严重的 winner's curse：
两组独立 seed 曾对同一方法给出相反结论（7.251 vs 14.626）。取消 seed 平均后指标才可复现。

### 2.2 Actor：`TorchMPPIDeterministicCenterActor`

- 511,120 参数，**log_std 头已删除**（确定性）
- 输出为 bounded residual：

```
center = clip( anchor + tanh(mu(state)) · 2 · source_sigma )
```

- 探索完全**外置**，不由策略采样：

```
c_explore = clip( c_actor ± radius(t) · hadamard_dir · source_sigma )
```

每轮 4 对（8 条），radius 随轮次衰减。
- trust region + fallback：超出信任域或回归时回退 anchor。

代码：`car_foundation/car_foundation/mppi_proposal_policy.py`，
MPPI 侧 `car_dynamics/car_dynamics/controllers_torch/mppi.py`。

### 2.3 方法演进（含失败）

behavior cloning → critic / ranking → AWR → 离散 actor-critic → multidirection replay
→ sequential probe SAC → 连续 SAC → **确定性 Direct Actor（当前）**

---

## 3. GT-first 五层拆解：瓶颈在哪

这是整个项目最重要的结论。把 warm start 到理论最优之间的差距逐层拆开：

| 层 | 含义 | 五帧 pilot |
| --- | --- | --- |
| warm | warm start 代价 | 22.023 |
| J*_16 | 16 维 knot 空间最优 | 5.821 |
| J*_100 | 100 维全自由度最优 | 5.558 |
| J*_center | 最优采样中心 | 5.824 |
| J*_bank | 固定库 clairvoyant | 7.527 |

派生 gap：

| gap | 值 | 判断 |
| --- | --- | --- |
| 参数化 gap（J16→J100） | 0.263（≈1.6 %） | 可忽略 |
| center vs J16 | 0.010 | 可忽略 |
| **库覆盖 gap** | **1.703** | 主要结构损失 |
| seed 误选 | 0.029 | 可忽略 |

S1-C 复核：unrestricted 5.8238 / bank clairvoyant 7.5269 / learned feedback 4-probe 7.5676 /
fixed-priority 7.6034 / guided 10.5498。学习式排序只比固定优先级好 0.0358，
距离库 clairvoyant 仅 0.0407 → **自适应排序未被证明有价值，瓶颈是候选覆盖本身**。

### 3.1 验证集数值预言机（300 快照 / 600 context）

| 方法 | 代价 |
| --- | --- |
| warm | 29.517 |
| round-1 Actor | 25.974 |
| **current Actor (v4, alpha=0.75)** | **25.518** |
| perfect-DBM two-center guard | 24.661 |
| T1 teacher | 12.762 |
| J*_16 | 4.894 |
| J*_100 | 4.697 |

Actor 只回收了 warm→J16 的 **16.2 %**，teacher 回收 68.0 %。
按速度分解：

| 速度 (m/s) | Actor |
| --- | --- |
| 1.2 | 3.059（优于 teacher） |
| 1.6 | 5.809（优于 teacher） |
| 2.0 | 17.223 |
| 2.4 | 39.528 |
| 2.8 | **61.974（差于 warm 48.595）** |

对照 J16→J100 只有 0.198，而 Actor→J16 是 20.624：
**瓶颈是高速段 recovery 与尾部泛化，不是 knot 维数。**

---

## 4. 已验证有效 / 无效

### 有效

- BC bounded residual（397,840 参数 Conv1d+MLP）：teacher-gap 回收随数据量 96→360→1800 states 为 ~0 → 14 % → **31.7 %**
  （`bc_t1_conv_diverse_20260805_v1/trust1_seed1.pt`，warm/net/teacher 21.622/18.611/12.136，300 局 234 胜 66 负）
- Teacher T1（2048 rollouts/state，2400 帧）：2392/2400 改善，均值 −8.896
- 离线多阶段经验响应引导（**从未上线**）：2 阶段 128+128 → best 2.1497（−57.7 %），ESS 30.76；
  3 阶段为默认，4 阶段最好（1.998966±0.042，10 seed 胜 7）；自适应 [96,80,40,40] best 2.000194±0.069
- Sequential probe 预算压缩：4 probe / 256 候选保留 33-probe 收益的 97.2 % → PASS
- Risk replay 仅 +0.03 σ 稳定（胜率 72.23 %）
- 排序型 critic：pairwise 准确率 0.768–0.917

### 无效 / 陷阱

1. **Critic 连续梯度外推始终失败**。`actor_critic_local_20260805_v1` = `rejected_all`；
   full-rank critic cosine 0.166 / 0.193，数据上限 0.617。
   关键诊断：同状态跨 seed cosine 0.630，**相邻帧跨 cosine 仅 0.044** → 局部响应不可跨帧共享。
2. **Winner's curse**：reward seed 平均导致结论翻转（见 §2.1），已通过取消 seed 解决。
3. **探索几何**：Direct Actor 宽高斯 v1 = 9.782 / replay best 18.191 FAIL，
   换成结构化 Hadamard v2 → 8.866 / 7.404（第 5 轮后发散），保守 v3 → 8.919 / 7.562（24 轮单调，当前首选）。
4. **Query 迁移在 DBM 真值上回归**：Query best 2.128765，但 DBM replay S2 4.012443 → S4 11.518799；
   DBM cross-eval fixed-4 = 13.684627±9.60 ≫ DBM best 5.085486。
   → **停机准则绝不能用 Query 预测代价；离线引导不得直接上线。**
5. AWR temp=2 只比 BC 好 +0.084，尾部不变。
6. DBM 梯度 oracle 1.78643 只是理论下界，解析梯度**禁止进入部署路径**。

---

## 5. 数据与纪律

首选集合 `fixed_dbm_policy_diverse_20260805_v1`：

- 120 episodes / 2400 snapshots，按 episode 切分 90/15/15 → 1800/300/300 states
- 5 档速度 1.2–2.8 m/s，6 类场景，64 候选/状态，936 MiB，step 250，stride 12，无观测噪声
- schema v2 保留 `predicted_trajectories_full [256,50,6]`、`sampling_noise_knots`、
  未加权 `feature_*`、`closed_loop_trace.jsonl`
  → cost 权重 / 温度 / reward **可重标注而无需重采**；但**分布覆盖无法重标注补回**

其他：

- `fixed_dbm_expansion_pilot_20260804_v1` = **REJECTED**
- v2 的 `episode_000` 仅用于流水线回归
- **episode 105–119 为封存 test，不得触碰**
- 派生 sidecar 规模：fullrank 40.55 M、two-pass feedback 21.50 M、risk replay 44.24 M、
  multidirection 81.10 M、direct-center replay 688,800/轮
- 原始数据不可变，一切派生结果写 sidecar
- selection seed 与 audit seed 隔离；偏差分级 D0–D4；状态枚举 PENDING/RUNNING/PASS/FAIL/INVALID
- Teacher validator argmin 容差放宽到 3e-4（两个近重复中心相差 3.16e-7 导致误判）

---

## 6. 当前状态 Gate 表

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| S1-C | GT-first 瓶颈定位 | PASS（结论：库覆盖 gap 1.703） |
| S2-B | forward-only 动态候选库 | **FAIL**（B6 clairvoyant 7.0731 回收 26.2 %，gate 要求 ≤6.6700；v4 step 2.0→4.0 → 7.042 / 28.3 % 仍 FAIL） |
| S2-C | probe 预算压缩 | PASS（4 probe 保留 97.2 %） |
| S3-A | Direct Actor episode-heldout | **mechanism PASS / tail-safety FAIL** |
| S3-B – S7 | 后续（含闭环） | 未启动 |

S3-A 细节（两轮 heldout）：27.445 → 25.974（alpha=0.375）→ **25.518**（v4 + alpha=0.75）
均值增益 0.456，胜率 58.7 %，P05 −2.457，**worst −109.109**。
checkpoint 标记 `VALIDATION_ONLY_TEST_SEALED`。
最差 context：`episode_101/step_000262/context 1`，143.683 → 252.793。
第二轮必须继承 twin Q（随机重置会把 ranking regret 从 8.00 恶化到 24.33）。
perfect-DBM two-center guard 24.661，但 P05/worst 增益均为 0 → 仅作模型选择上界。

---

## 7. 当前计划

### 7.1 阶段与分支门槛

| 阶段 | 任务 | 门槛 |
| --- | --- | --- |
| S0 | 数据 / 契约冻结（reward、候选库、切分） | 已完成 |
| S1 | GT-first 五层拆解 | 已完成 |
| S2 | 候选库扩张（S2-B FAIL）+ probe 压缩（S2-C PASS） | 覆盖 gap 回收 ≥50 % |
| S3-A | Direct Actor 泛化 | 均值增益 > 0 **且 P05 ≥ 0 且 worst 有界** ← 当前卡在此 |
| S3-B | 高速 recovery 专项 | 2.0–2.8 m/s 段不劣于 warm |
| S4 | Query 重标注 / Query 侧验证 | 必须先过 S3 尾部 gate |
| S5 | `mppi_proposal_runtime.py` + proposal ONNX 导出 | 实时性 ≤50 ms/step |
| S6 | 固定 DBM 闭环 A/B | **从未做过** |
| S7 | 封存 test 一次性评估 | 只做一次 |

### 7.2 Seed 台账

29001–29016 / 29101–29116 / 29201–29204 / 29211–29218 / 293xx / 30xxx。
selection 与 audit 分段占用，禁止复用。

### 7.3 下一步优先项（按顺序）

1. **用 J*_16 按速度量化可达 gap**，确认 2.0–2.8 m/s 的差距是覆盖问题还是策略容量问题。
2. **补 2.0–2.8 m/s 的 on-policy recovery 覆盖**；候选手段：risk 双头、确定性 fallback、
   按速度分桶的 trust radius。目标是把 worst −109.109 收敛到有界。
3. 只有 S3-A 尾部 gate 通过后，才进入 Query 重标注与 ONNX / 闭环。
4. 不要再投入自适应 probe 排序（S1-C 已证明收益 ≈0.036）。

### 7.4 2026-08-07 覆盖扩充执行结果

原计划中的“更多独立状态”已完成，不再是 PENDING。新增
`fixed_dbm_policy_train_expansion_20260807_v1`：270 个 train-only episode、1350 个
稀疏 snapshots，30 个速度×场景分层各增加 9 条轨迹。与旧 train 合并后为 360 个独立
episode、每层 12 条；旧 validation/test 不变。所有 episode 已两轮 validator 通过，
无观测噪声，冻结计划 hash 为
`b5bf9026f1ad4f7b0797739d07f3b61d73aff5da02f3727e538c763f9d101d8f`。

因此当前下一步从“采集状态”切换为“派生监督并复验”：

1. 对新增 1350 states 生成 train-only J16，保持与旧 `v2` 相同优化/复算契约；
2. 同时保存 J16 邻域的 forward-only antithetic cost difference / local curvature，避免
   再只用 knot MSE；
3. 旧 1800 + 新 1350 合并训练，内部选择继续按 episode/分层隔离；
4. 训练冻结后才读取原 validation，比较 current Actor 25.518、teacher 12.762、J16 4.894，
   并同时检查 2.0--2.8 m/s mean、P05 和 worst；
5. 此 gate 通过前，Query、ONNX、ROS 和长闭环仍不启动。

### 7.5 2026-08-07 派生监督与合并训练结果

上述 1--4 已执行，结果没有通过部署 gate：

- 新增 1350 states 的 J16 采用 `250+150` 步，`warm 28.120 -> J16 4.873`；两次全量
  独立重放的 source/interpolation/cost 误差均为 0；
- 旧 1800 + 新 1350 states 都生成了 `0.05/0.15 sigma × 16 Hadamard 正负方向`的
  forward-only direct-cost sidecar，每帧 65 centers，满秩率 100%，独立复算误差 0；
- 新增 first-pass Actor context 为 1350×2，旧数据五帧回归中 feedback/anchor/Critic
  mean/std 与正式历史 sidecar 逐元素误差 0；
- 合并训练使用 episode-balanced loss、60 个内部 held-out episode 选 epoch、300 epoch
  全程训练、再用全部 360 episode 重拟合；输出在
  `outputs/mppi_proposal/j16_cost_sensitive_expansion_20260807_v2`；
- 三个 full-step Actor validation mean 为 `31.432/28.675/33.693`，虽优于旧 plain 6σ
  `42.517--49.933`，仍差于 current Actor `25.518`；最佳 seed 1 的 P05/worst gain 为
  `-74.819/-369.475`；
- 最佳 seed 1 在 1.2/1.6/2.0 m/s 改善 current `0.476/0.592/1.538`，在 2.4/2.8 m/s
  退化 `12.980/5.408`。因此新增覆盖确实学到了更好的低中速方向，但高速步长/尾部仍失败；
- validation 输出中心 trust scan 的 `alpha=0.40` 可得到 mean `21.088`、平均改善
  `4.430`、胜率 72.0%，但 P05/worst 仍为 `-17.464/-81.653`。它只是双 Actor 的
  validation-calibrated 诊断，不是准入 checkpoint。

当前状态更新为：`J16 expansion PASS / forward-cost supervision PASS / direction
improvement PASS / full-step single Actor FAIL / tail gate FAIL / test sealed`。
下一步应学习或验证**状态相关的 trust step / fallback**，重点约束 2.4--2.8 m/s；不再把
“继续增加 epoch、邻帧、输出维度”作为默认方案。只有 mean 改善且 P05≥0、worst 有界后，
才允许进入 Query、ONNX、ROS 或闭环。

### 7.6 2026-08-07 高速退化诊断与速度范围暂缓项

用户提出后续可能把速度范围扩到 `100 km/h`（`27.78 m/s`），但当前决定是**先记录、
不在本阶段修改模型或采集范围**。现有实验继续以 `0.21 m` 小车、固定 DBM 和
`1.2--2.8 m/s` reference-speed 契约为准；不得把 `100 km/h` 样本直接混入现有
train/validation/test。若后续恢复该事项，应先单独确认车辆尺度、DBM 参数、赛道曲率、
动作/加速度边界和 horizon，再建立独立的高速数据与验证契约。

对 cost-sensitive Actor 的 validation 输出做固定直线 trust scan 后，按速度得到：

| reference speed | 分组最优 alpha | alpha=0 cost | 分组最优 cost | alpha=1 cost |
| ---: | ---: | ---: | ---: | ---: |
| 1.2 m/s | 0.75 | 3.059 | 2.520 | 2.583 |
| 1.6 m/s | 0.60 | 5.809 | 4.692 | 5.216 |
| 2.0 m/s | 0.55 | 17.222 | 13.583 | 15.684 |
| 2.4 m/s | 0.40 | 39.528 | 34.552 | 52.508 |
| 2.8 m/s | 0.35 | 61.974 | 49.474 | 67.381 |

新 Actor 相对 J16 的 knot RMS 在五档速度上为
`0.055/0.077/0.126/0.176/0.212`，明显随速度增加；但它仍比 current Actor 相对 J16 的
`0.199/0.213/0.277/0.381/0.436` 更接近目标。这支持如下判断：网络学到的平均方向是
有用的，主要失败是高速下需要更大的修正、同时 direct cost 对残余动作误差更敏感，
固定 full step 容易过冲。速度分桶也不足以完全解决：2.4/2.8 m/s 内逐 context 最佳
alpha 的 P10/P90 仍为 `0/1`，分别有 `27.5%/20.8%` context 最佳选择是完全不更新。

因此当前原因排序为：

1. **状态相关 trust step / fallback 缺失**（证据最强）；
2. **高速、overspeed、heading-recovery、高 yaw-rate 内部覆盖密度仍不足**，等量 episode
   不等于对高曲率 cost 面有足够密度；
3. 网络对 J16 的残余动作误差随速度上升，而高速度下该误差会在 2.5 s rollout 中放大；
4. 不是 16-knot 维数、动作 support 或主要多中心问题：J16→J100 均值仅 0.198，6-sigma
   投影几乎复现 J16，显著多解比例也只有 0.7%。

下一次实验应先学习/验证 context-conditioned alpha 或二中心 direct-rollout fallback，并按
`speed + overspeed + heading error + yaw rate + scenario` 分组检查 mean/P05/worst；在此
之前不继续全步 Actor 更新，也不把扩大到 100 km/h 与当前尾部问题混在同一实验中。

### 7.7 当前方案：forward-rollout deterministic TRPO-like update

已冻结详细设计：
[MPPI Direct Actor 的 FR-TRPI 更新方案](mppi_direct_actor_trpo_like_design_20260807.md)。
这里的 TRPO-like 指“冻结旧 Actor + sigma 归一化 trust region + DBM direct-cost line
search + 接受/拒绝”，不是恢复随机策略、Fisher/natural gradient 或 DBM 解析梯度。

逐 context `alpha=0:0.05:1` 诊断 oracle 已通过 TR0 上限检查：validation mean
`25.518 -> 14.049`，2.4/2.8 m/s 为 `39.528 -> 22.569` 和
`61.974 -> 31.549`。alpha=0 始终保留，所以该 oracle 没有负 gain；但它只是上限，不能
报告成 Actor 成绩。

执行顺序固定为：

1. 生成 train-only immutable trust-region sidecar，并独立重放全部 alpha cost（已完成）；
2. 使用 conservative safe alpha 训练相同结构 deterministic Actor（下一步）；
3. 冻结后在 formal validation 评价 Actor 的唯一输出；
4. 仅当 paired mean、P05、worst、2.4/2.8 m/s 和 recovery 全部通过，才接受为下一轮旧
   Actor；否则保持 current Actor 并按归因减小 trust radius、加强 stay 标签或更换
   forward-only proposal direction。

TR1 数据为 360 个 train episode、3,150 snapshots、6,300 contexts、132,300 次 direct
rollout；300/60 episode 固定为 internal-fit/internal-selection。独立复算的 center/cost 最大
误差均为 0，metadata 最大误差 `5.914e-7`，资格为 `TR1_VALIDATED`。train-only safe-label
mean cost 为 overall `19.411 -> 12.345`，2.4/2.8 m/s 为 `26.838 -> 16.344` 和
`48.740 -> 31.432`。这只说明监督目标有价值，不是新 Actor 成绩。

当前 gate：`TR0 PASS / TR1 PASS / TR2 FAIL / TR3 SEALED`。TR2 的 511,120 参数同结构
Actor 在 internal-selection 将 mean cost 从 `25.355` 降到 `22.680`，但 stay recall 为
0、gain P05/worst 为 `-11.636/-294.542`，且 3.33% context 超过 0.5-sigma trust radius。
所有非零参数插值步都未通过 worst `>=-5`；因此不能打开 TR3。后续应显式学习
state-conditioned alpha/stay gate、使用 verified two-center fallback，或更换 proposal
direction，而不是继续增加 epoch 或普通 MSE 权重。在 TR3 前，Query、
ONNX、ROS、四轮搜索、闭环和 100 km/h 扩展仍不启动。

2026-08-10 后续 TR2-B 已通过 train-domain gate。新增 move/stay + continuous-alpha
policy，internal-selection 选中 seed 2 / epoch 35；原 threshold 0.5 mean cost
`25.355 -> 14.855` 但 worst `-36.126`。只用 internal-selection 校准 threshold 0.99 后，
mean cost `21.522`、mean/median/P05/worst gain `3.834/0/0/-3.538`，五档速度 mean 均不
退化，资格为 `TR2B_PASS_ALPHA_AC_READY`。下一步可使用 TR1 的 119,700 fit line points
训练连续一维 Alpha Actor--Critic；12,600 selection line points只选模型。formal
validation/test 仍封存，完整 16 维 AC 暂不直接启动。

2026-08-10 Alpha Actor--Critic train-only 阶段也已完成。它是固定状态 contextual
bandit，不使用 next-state；双 Critic 拟合 21 点连续 alpha reward 曲线，Actor 从 TR2-B
初始化并同时使用保守 Q 梯度和 Critic 网格 policy-improvement 处理硬 move/stay 边界。
3 seeds 选中 seed 0 / Critic epoch 40 / Actor epoch 5，重新冻结 threshold 0.88。
internal-selection 的 mean cost 从 TR2-B `21.522` 降到 `20.696`，gain
mean/median/P05/worst 为 `4.660/0/0/-3.537`，五档速度 mean 均不退化。独立 DBM 重放
误差为 0，资格 `ALPHA_AC_VALIDATED_READY_FOR_FORMAL_GATE`。下一步是冻结的 formal
validation TR3；test、Query/ONNX、ROS、wrapper 和闭环仍不打开。

TR3 已按 D0 冻结协议执行并失败。formal validation old/TR2-B/Alpha-AC/safe-line/
argmin-line mean cost 为 `25.518/23.541/23.470/18.259/17.997`。Alpha AC 相对 old
mean gain `2.049`、95% CI `[0.376,4.401]`，但 worst `-75.937`；相对 TR2-B mean
gain 仅 `0.071`、CI `[-0.299,0.390]`。独立复算全部误差为 0。TR2-B worst 也为
`-55.249`，两套 alpha 策略都不能 standalone 使用，默认回到 old Actor。

根因是 move conditional-alpha 几乎饱和到 1；Alpha AC 在 17 个 safe-alpha=0 context
上移动且全部退化。下一步回到 train-only full-line 数据，增加 endpoint-regression、
maximum-safe-alpha/tail head 与高速 safe-zero hard-negative 过采样，并采集新的
validation-like episodes。当前 formal validation 已消费，禁止用于后续调参；test 继续
封存，新无偏 gate 前不做闭环、Query/ONNX、ROS 或完整 16-D AC。

---

## 8. Query / ONNX 现状

4 输入 rollout 协议：

```
history         [1, 250, 7]
initial_state   [1, 5]
current_action  [1, 2]
candidate_action[N, 50, 2]  →  输出 [N, 50, 5]
```

- 导出脚本 `export_query_mppi_onnx.py` → `outputs/query_mppi/anycar_query.onnx`
- **opset 版本在所有文档中均未记录**；I/O binding 仍是 TODO
- 延迟：Query PyTorch 29.0 ms / 超时率 0.53 %；DBM PyTorch 66.1 ms / 超时率 100 %
- 精度：Query 横向 MAE 0.435 m vs DBM 0.113 m（OOD checkpoint）
- 小车 checkpoint `outputs/formal_small_car_query_dt005/20260730T144840/query_best.pt`

---

## 9. 风险与长期未办

| 项 | 状态 |
| --- | --- |
| 高速（2.4 / 2.8 m/s）尾部灾难性回归 | **未解决，当前唯一阻塞项** |
| 固定 DBM 闭环 A/B | 从未做过 |
| `mppi_proposal_runtime.py` | 未实现 |
| proposal 侧 ONNX 导出 | 未实现 |
| cost 权重 / 温度采样范围 | 未冻结 |
| 实时性 57–60 ms/step vs 50 ms 预算 | 超预算 |
| ONNX opset 未记录 | 需补 |

准入规则：
离线指标不能替代闭环结论；DBM 解析梯度不得进入部署路径；
封存 test 只在最终一次评估时解封。

---

## 10. 2026-08-18 最新路线覆盖：Search Phase 2，Critic梯度主线关闭

本节覆盖§7--§9中关于“继续修复Critic梯度/恢复Actor--Critic更新”的历史下一步；完整证据见
`mppi_sampling_center_review_20260812.md` §11.38–§11.50（注意编号有重复，详见主文档§11.80消歧）。

- Phase 1b的600状态multi-128 guarded teacher已完成：`J(a0)=16.012 -> J_teacher=7.887`，
  `R_a0=0.7245`，episode-bootstrap 95% CI `[0.684,0.753]`，direct-cost基线违规为0；
- 最后Critic坐标实验使用16维满秩sensitivity/DCT坐标、3 folds×3 seeds，未改value-delta
  loss。新坐标虽改善coordinate cosine，但physical-step P10仍`-0.32~-0.39`，DBM J50
  0.05σ gain P05仍`-5.30~-11.00`；
- 最终判定`COORDINATE_CRITIC_FAIL_CLOSE_GRADIENT_MAINLINE`。Critic不再提供Actor梯度，
  只可作为离线candidate ranking/value辅助；
- 当前执行主线是Phase 2：以现成multi-128标签先跑A0均匀8-knots bounded residual Actor
  episode-grouped cross-fit，同时运行B0 sensitivity/Jacobian和C1 along/cross诊断；A0+B0后
  再做影响归一化A1和结构A2；
- formal validation/test继续封存，部署Actor保持冻结。只有Phase 2 OOF恢复率与tail门通过、
  且B线参数化冻结后，才进入two-center MPPI传导和短闭环A/B。

动作顺序统一为`[acceleration, steering]`；steering knots 0--2的flat indices是`[1,3,5]`。
