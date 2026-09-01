> **[CLOSED-HISTORY 2026-08-20]** 本文档记录的路线已关闭/被取代。权威结论见 `mppi_sampling_center_review_20260812.md` §11.80。本文档不再更新，仅作历史参考。

# MPPI 串行 probe 策略：后续执行与偏离检查计划

更新时间：2026-08-07

## 1. 当前基线与目标

本计划只研究**同一车辆状态内的多轮 sampling-center probe**。车辆状态、250 步
history、reference、guided center 和模型参数在一次内部搜索中保持不变；每轮只增加一个
真实 DBM/Query rollout 反馈。它不是车辆 `next_state` 的多步强化学习。

当前正式诊断基线：

```text
DBM collection:
  fixed_dbm_policy_diverse_20260805_v1

multidirection replay:
  labels/dbm_two_pass_multidirection_replay_diverse_20260805_v1

terminal-reward Actor--Critic:
  outputs/mppi_proposal/sequential_probe_sac_20260805_v2

fresh gate:
  outputs/mppi_proposal/sequential_probe_teacher_eval_20260805_v5
```

fresh seed `28401` 探测、`28411--28418` 评价的 mean cost：

| 方法 | mean | P95 | maximum |
|---|---:|---:|---:|
| guided | 16.737 | 52.524 | 495.968 |
| 固定优先级 4-probe | **11.138** | 35.513 | 244.189 |
| learned sequential 4-probe | 11.246 | 35.601 | 272.541 |
| T1 teacher | 11.913 | **29.134** | **64.274** |

当前已经通过的是“4 probes/256 candidates 可压缩 full 33-probe/2112 candidates 的
预算”；尚未通过的是“反馈 Actor 优于固定 probe 顺序”和“teacher 级尾部”。

### 2026-08-06 Actor 目标修正（当前权威口径）

连续 Actor 的主训练目标不再是“把该中心作为 MPPI 均值，再用随机 candidates 得到的
weighted-output cost”。该目标会让同一 `(state, center)` 因 reward seed 不同得到不同
标签，混淆 Actor 本身好坏与随机 MPPI wrapper 好坏。当前固定为：

```text
c = policy(state)                         # 8x2 knots
A = LinearInterpolate(c, horizon=50)     # 唯一 50x2 actions
J_direct = Cost(DBM(state, A))            # 唯一 forward cost
reward = J_direct(anchor) - J_direct(c)
```

探索可以随机地产生不同 `c`，但一个已经给定的 `c` 不再使用 reward seed。Critic 的第一
阶段监督只拟合这个确定性的 `Q_direct(state,c)`。旧 T1/33-bank/actor-visited centers 可以
重算 direct cost 后进入 replay；它们原先保存的随机 MPPI weighted-output cost 不能直接
作为新 Critic 标签。

“一个 center 是否适合作为 MPPI 采样中心”保留为第二层独立指标。它使用冻结的
`fixed-hadamard-64-v1`：exact center + 一个固定 extra + 31 对 antithetic offsets，覆盖
16 维满秩方向，半径固定为 `0.10/0.30 source sigma`。候选库、顺序、插值、cost、temperature
和边界均固定并版本化；训练评估和运行时 MPPI 调用同一个生成器。该层可训练辅助
`Q_neighborhood/risk`，但不得反向污染第一阶段 Direct Actor reward。

在继续训练策略前，先建立固定 DBM 下的数值 GT，并把差距拆成五层。本文中的
“GT”默认指多初值数值优化得到的 **best-found oracle**；只有不同初值、优化器和精度复算
稳定收敛后，才可把它称为可信上限，不能宣称已证明全局最优。

```text
J*_100    50×2 独立动作的 best-found direct optimum
J*_16     8×2 knots 线性插值后的 best-found direct optimum
J*_center 不限制当前 33-center bank 的最优 sampling center
J*_bank   当前可部署 center bank 的 clairvoyant oracle
J_policy  固定顺序或反馈策略实际得到的结果
```

对应差距为：

- `J*_16 - J*_100`：16-knot 参数化损失；
- `J*_center - J*_16`：MPPI 随机采样/加权输出相对 direct action optimum 的损失；
- `J*_bank - J*_center`：当前 center bank 覆盖损失；
- `J_policy - J*_bank`：策略选择/排序损失。

最终目标分三层，不能混为一个结论：

1. **反馈价值：**相同 center bank、probe 数和 candidate 总预算下，反馈策略显著优于
   validation 选出的最强固定顺序和 state-only 顺序。
2. **固定 DBM 控制价值：**冻结策略在新 seed frozen-state gate 和短闭环 A/B 中同时
   改善 mean，且不恶化 tail、失败率和实时性。
3. **Query/部署价值：**只用 Query forward rollout reward 重新训练/校准后，PyTorch 与
   ONNX 数值一致，并在 Query gate 中保留相对固定基线的收益。JAX 不维护。

## 2. 每一步统一的偏离检查

从本计划开始，每一步执行前后都更新本文的“执行记录”。每次检查固定回答以下问题：

| 检查项 | 执行前必须记录 | 执行后必须记录 |
|---|---|---|
| 研究问题 | 本步只回答什么问题 | 实际结果是否仍回答该问题 |
| 冻结项 | 数据 split、DBM/Query、cost、sigma、horizon、预算、seeds | 是否发生任何变化 |
| 允许变化 | 本步唯一允许改变的变量 | 实际改变了哪些变量 |
| 数据隔离 | train/validation/audit/fresh 的用途 | 是否有 seed、episode 或标签泄漏 |
| 预算公平 | probes × candidates、最终公共复评预算 | 各方法实际预算 |
| 期望 | 方向性假设和大致幅度 | 实际值、置信区间和分组结果 |
| 门槛 | 通过、停止或转向条件 | PASS / FAIL / INVALID |
| 下一步 | 通过后允许进入哪里 | 是否按分支执行 |

偏离等级：

- `D0`：无偏离；按计划继续。
- `D1`：数值偏离预期，但协议未变；保留有效负结果，按门槛转向。
- `D2`：预算、seed、cost、模型或数据分布发生未计划变化；暂停对比，重新建立公平基线。
- `D3`：任务范围漂移，例如 frozen-state 未通过就进入长闭环或 Query；停止并
  回到最近通过的 gate。
- `D4`：train/validation/test 泄漏、用 audit 调参或复用评价 seed；结果作废并换新 seed。

`D2--D4` 不能用文字解释后继续，必须修正协议后重跑。任何 gate 为 FAIL 时也不自动进入
下一阶段；先记录原因，再走本文指定的分支。

## 3. 分阶段计划

### S0：冻结基线、评估器与 seed ledger

目的：确保以后所有收益都相对同一个可复现基线，而不是随脚本参数漂移。

执行：

1. 固化 source/parent/risk/multidirection sidecar 和 checkpoint SHA256。
2. 固化 33 个 action 名称、center 重构、4-probe 预算、每 probe 64 candidates、
   second noise `0.10 sigma`、cost/temperature/horizon。
3. 把当前固定顺序、v2 sequential、T1 teacher 和 fresh v5 指标写入机器可读 manifest。
4. 建立 seed ledger；历史 `20xxx--28418` 不再用于新的最终 qualification。
5. 复跑 validator、checkpoint strict load、v5 array shape 和图表生成。

期望：stored replay 可逐元素复现；fresh deterministic 重跑指标只出现浮点舍入误差。

门槛：所有 hash、shape、center reconstruction、seed 隔离和 checkpoint load 通过。任何一项
失败均为 `INVALID`，不得开始新数据或训练。

### S1：先求固定 DBM 数值 GT，再定位当前 block

目的：在不训练新网络之前，直接回答“当前主要受动作参数化、MPPI 采样、center bank，还是
策略学习限制”。这是后续所有数据采集和 Actor/Critic 方案的前置 gate。

#### S1-A：完全一致的 direct-action oracle

对同一 snapshot 冻结 DBM 参数、真实初始 `vy`、state/history/reference、current action、
horizon、动作边界和 cost weights。分别求：

1. `J*_16`：优化 8×2 knots，经当前 `align_corners=True` 线性插值成 50×2 actions；
2. `J*_100`：直接优化 50×2 actions；
3. 初值至少包含 warm、T0/T1、当前 best candidate、零动作和多个随机初值；
4. 使用投影 Adam，必要时用 LBFGS/CEM 交叉检查；保存每个 start 的完整收敛曲线；
5. 最终 action 和 trajectory 必须通过独立、无梯度 DBM evaluator 复算，误差不超过
   `1e-4 + 1e-5 × |cost|`。

DBM 解析梯度只允许用于这个离线诊断 oracle。它不得成为策略输入、teacher 运行时依赖或
部署算法的一部分，因此不会破坏未来 Query/ONNX 只依赖 forward rollout 的约束。

pilot 先选 5 个覆盖不同速度/场景的非 test snapshot。正式分析至少覆盖 train/validation
各速度和场景分组；test 仍保持封存。

可信度门槛：top-3 独立初值 final cost 的相对 spread 小于 1%，且增加迭代后改善小于
0.1%。不满足时只称 `best-found`，增加初值/优化器，不用它判定网络上限。

#### S1-B：sampling-center oracle

在 S1-A 通过后，求 `J*_center`：center 仍为 8×2，但输出必须经过当前 MPPI 的固定
`sigma/temperature/candidate budget/weighted output`。优化阶段固定 common random
numbers，选择 seed 与独立 audit seeds 分离；最后用多组新 seed 复评期望 cost 和 tail。

该层可以使用高预算 CEM、局部 response fit 或固定噪声的可微 surrogate，但最终分数只认
独立 forward rollout。它回答的是“最优采样中心在哪里”，不能和 direct-action optimum
混为一个目标。

#### S1-C：bank 与 policy gap

在同一批 snapshot 上复算：

1. `B0 guided`；
2. `B1 optimized fixed 4-probe`；
3. `B2 state-only 4-probe`；
4. `B3 feedback-conditioned 4-probe`；
5. `B4 full 33-probe bank oracle`；
6. `B5 independent-seed clairvoyant`，仅作诊断。

统一报告 `J*_100/J*_16/J*_center/J*_bank/J_policy`、absolute gap、相对可恢复比例、
mean/P10/P95、paired bootstrap CI 和速度/场景分组。策略比较仍保持 4×64 candidates 的
公共预算；direct oracle 的额外计算只记作离线诊断预算，不伪装成可部署性能。

分支门槛：

- 若 `J*_16-J*_100` 占当前总 gap 的主要部分，优先增加 knots/改变动作参数化；
- 若 `J*_center-J*_16` 最大，优先改 MPPI 更新、sigma/temperature 或多轮 center 优化；
- 若 `J*_bank-J*_center` 最大，进入 `S2-B 动作库扩充`；
- 若 `J_policy-J*_bank` 最大且反馈 oracle 明显优于 state-only，进入 `S2-A` 和 `S3`；
- 若 top starts 未稳定收敛，本阶段为 `INVALID/INCONCLUSIVE`，不得据此增加 RL 复杂度。

### S2-A：on-policy probe outcome 与 reward repeat（条件分支）

适用条件：S1 证明当前 bank 有可利用反馈上限，但现有 Critic/数据未学到。

先做 pilot，不直接全量生成：

1. 仅使用 train/validation episodes；test episodes 禁止进入生成器的模型选择路径。
2. 每个 context 保存 anchor、optimized fixed actions、当前 Actor 实际访问 actions，以及
   ensemble 不确定度最高的 counterfactual actions；目标不超过 11 centers/context。
3. pilot 目标约 600 train contexts；每 center 使用 8 selection + 8 audit reward seeds。
4. 同时做 16/32/64 inner candidates 小消融，先确定 probe cost 信噪比与运行预算。
5. sidecar 使用 staging、source/parent/checkpoint hash、不可变 final path 和独立 validator。

若采用 600 contexts × 11 centers × 16 seeds × 32 candidates，pilot 约为 3.38M DBM
candidate rollouts。它们是 reward repeats，不得报告成 600 个新车辆状态。

期望：更多独立 seeds 降低 terminal return 方差；32 candidates 可能保留 64-candidate
大部分排序信息，但这一点必须实测。

门槛：

- selection/audit 的中心 paired ranking 在 `|gain|>0.5` 子集上至少 75%；
- terminal-return selection/audit correlation 至少 0.7；
- 若选 32 candidates，其 validation mean gain 至少保留 64-candidate 的 95%，P95 cost
  恶化不超过 0.5；否则回到 64 candidates；
- validator 必须 100% 通过。未达到信噪比门槛时，先增加 reward repeats，不训练 Actor。

pilot 通过后才扩到全部 1800 train states 和 300 validation states，并迭代 2--3 轮：每轮
用新 Actor 采集实际访问中心，再更新 Replay Buffer；每轮 audit seeds 保持只读。

### S2-B：扩充运行时可用的非局部动作库（条件分支）

适用条件：S1 显示当前固定 4-probe 已接近 33-center bank 上限，尤其高速度/恢复状态仍有
明显 teacher gap。

只加入运行时可构造、未来 Query 也能 forward rollout 的中心：

1. frozen BC proposal center；
2. warm/guided/first-pass best 之间的非局部插值；
3. 当前 8 类方向的 `0.25/0.40 sigma` 大半径候选，但必须独立 risk gate；
4. 高速度/恢复状态专用的 state-conditioned global proposal；
5. 不得加入 T1 teacher center 或 DBM 解析梯度作为运行时输入。

先在 train/validation 小 pilot 计算 oracle，不立刻训练 Actor。每次只增加一类方向，保持
candidate budget与 common random numbers 一致。

期望：新的 bank 主要改善 2.4/2.8 m/s 和 recovery，而不是只在低速重复现有局部最优。

门槛：validation oracle 相对 optimized fixed current bank 至少改善 `0.5 mean cost`；
2.4/2.8 m/s 或 recovery 至少改善 `1.0`；P95 不得恶化超过 `0.5`。达不到则删除该方向，
不进入正式 replay。大半径若只提高 mean、降低胜率或恶化 tail，必须加 risk/fallback，不能
直接放入 Actor 动作空间。

### S2-C：连续 16-D Actor 合同与旧数据 bootstrap

2026-08-06 用户明确决定：离散 33-slot SAC 降级为历史基线和数据生成器，不再作为最终
动作空间。Actor 改为 squashed-Gaussian，直接输出标准化 `[8,2]` center residual：

```text
z ~ Normal(mu(s), sigma(s))
a = tanh(z) in [-1,1]^16
center = clip(anchor + a * maximum_delta_sigma * source_sigma)
```

旧 T1、33-center bank 和 residual-response 的新角色仅为：BC warm start、初期 replay、
exploration prior 和固定基线。它们不得限制 Actor 后续输出方向。Critic 改为连续
`Q(s,a)` twin networks；不再把 33 logits 称为最终 SAC Actor。

bootstrap 只准做两件事：用已评估目标初始化 Actor mean，用已有 DBM center/reward
初始化 twin Q。**禁止**用这个离线 Q 直接更新并资格认定 Actor；任何新 Actor center
必须先经 fixed-DBM forward rollout 写入 actor-visited replay。

门槛：动作边界、log-prob、Actor/Q 梯度和 checkpoint 重载全部 finite；fresh DBM 上
初始化 Actor 至少不劣于旧 bank clone 的主要状态，且输出不饱和。该门槛只说明可进入
on-policy pilot，不代表 SAC 已通过。

### S3-A：单步 actor-visited continuous SAC

先保持同一物理状态的单步 contextual bandit，验证连续 Critic 能否从确定性 direct reward
学到 Actor 更新方向，不立即增加四轮 Bellman 问题：

1. stochastic Actor 产生任意 16-D center residual；旧 bank/residual-response 仅混入初期
   exploration，不作为离散 action id；
2. 每个新 center 线性插值为唯一 50×2 actions，用一次 fixed-DBM direct rollout 得到
   anchor-relative reward；同一个 center 重复计算必须逐元素一致；
3. 保存 `(state, feedback, continuous_action, direct_reward)` 到 actor-visited replay；
4. twin Critic 多次更新，Actor 低频更新；每批更新后必须重新与 DBM 交互，不能冻结 Critic
   后长时间利用其梯度；
5. Actor loss 为标准 SAC entropy objective 加随训练衰减的 BC/support trust；不使用 DBM
   解析梯度；
6. validation states 只用于 checkpoint 选择；test states 封存。主 direct reward 没有
   selection/audit seed 之分。

actor 更新前的 replay gate：同一 `(state,center)` 重算误差为 0（允许 GPU 浮点 tolerance
`1e-5`）；每状态动作差分矩阵有足够方向覆盖；Critic 在 episode-heldout direct labels 上
通过 pairwise ranking、Actor-neighborhood gradient/finite-difference 一致性和真实 rollout
复评。若 Actor 采样全部远离当前好中心，先收窄/结构化 exploration，而不是增加 reward
repeats。

单步准入门槛：连续 Actor 相对 BC clone 与 optimized fixed bank 的 paired mean gain
bootstrap 95% CI 下界大于 0，P95 不恶化超过 0.5，每个速度档/recovery mean regression
不超过 0.5，worst-context gain 不低于 -5。未通过不得进入四轮训练。

### S3-B：四轮 continuous SAC 内部搜索

仅在 S3-A 的 Direct Actor PASS 后，才引入固定邻域 MPPI wrapper。每轮对 Actor center
使用同一个 `fixed-hadamard-64-v1` 候选库；固定候选 weighted output、best/P10/ESS 和轨迹
residual 构成确定性的中心质量反馈。若再扩为四轮，Actor 读取最新 best cost 与 residual
反馈生成下一 center；网络参数在四轮 episode 内冻结，transition 写入 replay 后才更新。
此时评价的是“Direct Actor + 固定 wrapper”，不能把 wrapper 收益记为 Direct Actor 收益。

必须同时比较 `fixed bank 4-probe`、`multi-round residual-response`、`continuous state-only`
和 `continuous feedback-conditioned SAC`。feedback 策略须在 fresh mean/tail 上同时超过
固定顺序，才能把收益归因于多轮学习。

### S4：完全冻结的 fresh DBM qualification

输入：S3 冻结 continuous checkpoint、optimized fixed/bank baseline、连续 action bounds
和冻结 normalization。

建议预留 seed 范围（写入 seed ledger 后不得改用途）：

```text
29001--29016  on-policy pilot selection/audit
29101--29116  action-bank pilot selection/audit
29201--29204  formal fresh probe seeds
29211--29218  formal fresh evaluation seeds
293xx         short DBM closed-loop seeds
30xxx         future Query relabel/evaluation
```

若实际生成前发现冲突，先更新 ledger；不能静默换 seed 后与旧结果混报。

formal frozen-state 对比至少包含 guided、old discrete/fixed probes、multi-round
residual-response、continuous state-only、continuous feedback policy 和 T1 teacher。所有
方法保持相同 probe 数、每 probe candidates、final common reevaluation 和模型 backend。

输出：mean/median/P90/P95/max cost、paired win/loss/tie、bootstrap CI、每速度/场景分组、
ESS、clipping、最终动作分布、probe sequence、candidate 总预算与延迟。

通过条件沿用 S3，并增加：至少两个独立 probe seeds 上结论同号；不能靠一个 probe seed
的幸运结果通过。未通过则按 failure attribution 返回 S2-A（噪声/数据）、S2-B（bank
上限）或 S3（风险/训练），不直接调 fresh test。

### S5：固定 DBM 短闭环 A/B

只有 S4 PASS 后执行。暂不接 Query、不加观测噪声。

对比 guided、optimized fixed probes、feedback policy；使用相同 track、初态、MPPI/控制
seed、cost、候选总预算和终止条件。覆盖五档速度及 steady/start/recovery，保存完整 trace。

验证：累计 cost、横向/航向/vx tracking P50/P95/max、控制平滑性、clipping、ESS、失败/
越界率、每周期端到端耗时和 deadline miss。网络相对固定的收益必须在 episode paired
统计中复现，不能只看 frozen snapshots。

门槛：

- 累计 mean cost 的 paired CI 优于 fixed；
- tracking P95、失败率、越界率不恶化；
- 任一速度/恢复组不得系统性退化；
- 总计算耗时不超过控制周期，网络开销建议小于周期的 20%；
- 若出现分布漂移，保存 actor-loss states，做至多一轮 DAgger/relabel 后重新从 S4 gate。

### S6：Query PyTorch 重标注、训练与 ONNX 一致性

只有 S5 PASS 后执行。

1. 冻结当前 Query checkpoint；用 Query PyTorch forward rollout 重新生成 probe/reward，
   不能复用 DBM trajectory 或 cost。
2. 先复评 fixed/action-bank oracle，确认 Query 下仍有 headroom；没有上限时不训练 Actor。
3. 用 Query reward 微调 distributional Critic/Actor，并重复 S3/S4/S5 的隔离与门槛。
4. 导出 ONNX；相同 batch 上检查 PyTorch/ONNX Actor logits、动作选择、probe sequence 和
   最终 center。一致性目标为最大绝对输出误差 `<=1e-5`，动作选择 100% 一致；若边界
   tie 导致动作变化，必须固定 tie-break 或提高数值裕度。
5. JAX 版本不维护，也不作为任何 gate 的依赖。

期望：Query 噪声会使收益低于 DBM；首要目标是相对 Query fixed baseline 保持正收益，
而不是强行复现 DBM 的绝对 cost。PyTorch/ONNX 不一致时先修部署图，不做性能归因。

### S7：ROS/在线集成（最后阶段）

只有 Query frozen-state、短闭环和 ONNX gate 均通过后执行。加入 warm fallback、超时回退、
best-so-far、action bounds 和 runtime telemetry；默认不启用 learned policy，先做 shadow
mode。真实车辆或高保真 Isaac/MuJoCo 数据只能作为后续外部分布验证，不能反向污染当前
DBM/Query final test。

## 4. 执行记录

每完成一步追加一行，并在其后附完整指标路径：

| 日期 | Step | 状态 | 实际结果 | 偏离 | 决定 | 产物 |
|---|---|---|---|---|---|---|
| 2026-08-06 | Plan | PASS | 冻结后续顺序、分支门槛和偏离模板 | D0 | 从 S0 开始 | 本文 |
| 2026-08-06 | S1-A pilot | PASS | 五帧 direct `J*_16/J*_100` 数值稳定并独立复算 | D1 | 进入 S1-B pilot | `dbm_direct_gt_*20260806*` |
| 2026-08-06 | S1-B first run | INVALID | 错用 `1.0 sigma`，与 probe 的 `0.10 sigma` 不同 | D2 | 修正并全部重跑 | `dbm_sampling_center_gt_*20260806_v1` |
| 2026-08-06 | S1-B second run | INVALID | 虽改为 `0.10 sigma`，但误用了 iid noise，与实际 zero/extra/antithetic candidate design 不同 | D2 | 统一调用正式 antithetic generator 后重跑 | `dbm_sampling_center_gt_*20260806_v2` |
| 2026-08-06 | S1-B five-frame | PASS | 正式 antithetic 协议下 bank clairvoyant coverage gap 为 1.703 | D0 | 接 S1-C policy gap | `dbm_sampling_center_gt_*20260806_v3` |
| 2026-08-06 | S1-C pilot | PASS | feedback 距 bank oracle 0.041，bank coverage gap 1.703 | D0 | 进入 S2-B bank 扩充 | `dbm_gt_policy_gap_pilot_20260806_v2` |
| 2026-08-06 | S2-B forward-only dynamic bank | FAIL | residual-response 在五帧都改善 B0，mean clairvoyant `7.515→7.073`，但仅回收约 26.2% aggregate coverage gap，未达到 50% gate | D1 | 不生成 replay、不训练 Actor/Critic；下一步只做多轮 forward-response pilot | `dbm_dynamic_generator_oracle_20260806_v3` |
| 2026-08-06 | S2-B larger-step check | FAIL | 独立新 seeds、最大 response step 由 2.0 放到 4.0 后，B7 `7.524→7.042`，仍只回收约 28.3% aggregate gap | D1 | 排除“仅步长过小”；保持网络结构不变 | `dbm_dynamic_generator_oracle_20260806_v4` |
| 2026-08-06 | Continuous-SAC plan revision | PASS | 用户决定最终 Actor 直接输出连续 16-D center residual；离散 bank 降级为 BC/exploration 数据源 | D0 | 增加 S2-C、S3-A、S3-B，旧离散 checkpoint 仅作基线 | 本文 |
| 2026-08-06 | S2-C continuous interface | PASS | squashed-Gaussian Actor、连续 twin Q、2-sigma center 映射、log-prob/反传/checkpoint smoke 全部 finite | D0 | 运行受控 bootstrap | `mppi_proposal_policy.py` |
| 2026-08-06 | S2-C T1 bootstrap | FAIL | T1 BC 五帧 mean 虽 `11.474→10.874`，但 3/5 退化；T1 不依赖当前 first-pass feedback | D1 | 改克隆同 context 的旧 bank oracle | `continuous_center_sac_bootstrap_*v1` |
| 2026-08-06 | S2-C bank-oracle bootstrap | PASS | fresh 五帧全部改善 anchor，mean `10.819→7.754`，接近 bank selection `7.642`；输出不饱和 | D0 | 进入 S3-A actor-visited pilot | `continuous_center_sac_bootstrap_*v2` |
| 2026-08-06 | 旧 stochastic-wrapper S3-A v2/v3 | INVALID | v2/v3 对同一 center 使用随机 MPPI weighted-output reward；它们有效揭示 winner's curse，但不再回答 Direct Actor 学习问题 | D2 | 保留为历史 wrapper 诊断，不用于 Actor gate/critic 初始化 | `continuous_center_sac_fixed5_pilot_20260806_v{2,3}` |
| 2026-08-06 | Direct objective + fixed bank contract | PASS | Direct cost 重复最大误差 `0.0`；固定候选 `64×8×2`、16-D 满秩，contract hash `0fd540206ec988f565481d25f9cbd0b4051f7847e154565d9eac4e7445b6a492` | D0 | 主 reward 无 seed；固定 bank 仅作辅助/运行时 wrapper | `mppi.py`, `fine_tune_mppi_direct_center_sac_pilot.py` |
| 2026-08-06 | Direct fixed-five SAC pilot | FAIL | 320 个 Actor 新中心均用唯一 direct reward；初始/final mean 均为 `9.782`，best iteration=0；当前宽高斯探索的 actor-visited best mean `18.191`，未覆盖初始附近好动作 | D1 | 先修探索分布/直接重标现有好中心，再验证 direct Critic，不增加 reward repeats | `continuous_center_direct_fixed5_pilot_20260806_v1` |
| 2026-08-06 | Deterministic Actor + structured exploration v2 | PASS | 去掉 Actor 方差头；外部 `4` 对 Hadamard 方向按 `0.30→0.05 sigma` 衰减。Actor direct mean `9.782→8.866`，replay best `7.404`；3/5 帧改善，但第5轮后继续更新发散 | D1 | 证明探索修正有效；再降低 Actor 更新强度和探索半径 | `continuous_center_direct_fixed5_pilot_20260806_v2` |
| 2026-08-06 | Conservative deterministic Actor v3 | PASS | `0.15→0.03 sigma`、Actor lr `1e-5`、每轮1次 Actor update；24轮 direct mean 单调 `9.782→8.919`，replay best `7.562`，3/5改善，另2帧仅 `-0.103/-0.031` | D0 | v3 作为当前机制首选；扩大到 episode-heldout states 前仍不作泛化准入 | `continuous_center_direct_fixed5_pilot_20260806_v3` |

状态只使用：`PENDING / RUNNING / PASS / FAIL / INVALID`。任何 FAIL/INVALID 都必须先写明
原因和回退分支，再开始新实验。

## 5. 当前允许的下一项

S2-C 连续动作合同与 bank-oracle bootstrap 已完成；Actor 的目标已修正为确定性 direct
cost，并已去掉可学习方差头。当前 Actor 每个 context 只输出唯一 center；训练探索由外部
Hadamard antithetic pairs 产生，半径独立衰减。该修正已把旧高斯探索 replay-best
`18.191` 改善到 `7.562`，首选保守 v3 的 Actor direct mean 从 `9.782` 单调降至 `8.919`。
因此五帧机制已通过“唯一 Actor + 衰减探索能学习”的最低验证，但它仍只在5个 train states
上成立，且2/5帧有很小退化。

下一项是把同一协议扩大到 episode-level train/validation：将旧 bank、BC/T1、插值和
小半径满秩邻域全部按 direct cost 重标，冻结 test episodes；以 validation 选择探索衰减、
Actor update ratio 和 checkpoint，报告每速度/场景及 tail。先验证 `Q_direct` heldout
ranking、局部有限差分方向和 Actor 真 rollout，不直接进入四轮 wrapper。

旧 33-slot bank、T1 和 residual-response 只能作为 BC/exploration prior，不能重新变成离散
动作上限；其 cost 必须 direct 重算。不得使用 DBM 解析梯度。固定 64 候选库暂时只报告
辅助中心质量，不能用于选择 Direct Actor checkpoint。只有 S3-A 连续 Actor 在独立状态上
以 direct cost 超过 clone 与 direct GT 对照，才进入 S3-B 固定 wrapper/四轮内部搜索。

## 6. GT-first pilot 详细结果

### S1-A direct-action oracle

冻结 `fixed_dbm_policy_diverse_20260805_v1` 的 step 250、原 DBM/真实 `vy`、50-step
horizon、cost、动作边界和 8-knot 插值。五个 train snapshot 覆盖 1.2--2.8 m/s，并依次
覆盖 steady、cold start、lateral recovery、heading recovery 和 dynamic recovery；test
未读取。

`generate_dbm_direct_gt_pilot.py` 从 warm/T0/T1/zero/random 多初值先求 8×2 knots，再放开
为 50×2 actions；`validate_dbm_direct_gt_pilot.py` 走无梯度 DBM 路径独立复算。
DBM 新增的 differentiable rollout 仅用于离线 oracle，不进入策略输入、Query 或部署。

| snapshot | speed/scenario | warm direct | `J*_16` | `J*_100` | 参数化 gap |
|---|---|---:|---:|---:|---:|
| episode_000 | 1.2 / steady | 27.215 | 6.823 | 6.210 | 0.612 |
| episode_021 | 1.6 / cold start | 19.529 | 1.182 | 1.177 | 0.004 |
| episode_042 | 2.0 / lateral recovery | 28.154 | 3.310 | 3.192 | 0.117 |
| episode_063 | 2.4 / heading recovery | 16.378 | 9.808 | 9.539 | 0.269 |
| episode_084 | 2.8 / dynamic recovery | 18.841 | 7.985 | 7.673 | 0.312 |
| mean | - | 22.023 | 5.821 | 5.558 | 0.263 |

最终 top-3 relative spread 平均为 `J*_16 0.170%`、`J*_100 0.021%`，所有单帧均低于
1%；最后 20 optimizer steps 的改善平均为 `0.055%/0.023%`，所有单帧均低于 0.1%。
独立 validator 最大 cost 误差小于 `3.1e-5`。五帧 pilot 通过，但仍称 best-found oracle，
不宣称数学上已证明全局最优。参数化 gap 仅占 `warm-J*_100` 可恢复差距约 1.6%，因此
当前不优先把 16 维改回 100 维。

偏离为 `D1`：原单次 500/700 steps 运行耗时过高，在未产生结果文件时中止，改为 150/250
初筛，再只对未满足停止门槛的帧 resume。模型、状态、cost、初值集合与门槛未改变。

```text
outputs/mppi_proposal/dbm_direct_gt_pilot_20260806_v1
outputs/mppi_proposal/dbm_direct_gt_refine_remaining_20260806_v1
outputs/mppi_proposal/dbm_direct_gt_refine_episode042_20260806_v1
outputs/mppi_proposal/dbm_direct_gt_refine_episode063_20260806_v3
```

### S1-B sampling-center oracle 五帧 pilot

五个 snapshot 均固定 context 0、每个 center 64 candidates、原 sigma/temperature；selection
seeds `29411--29412` 搜索，audit seeds `29421--29424` 只用于最终复评。每帧 96 个
multi-start CEM retained centers 与当前 33-center bank 使用相同 audit noise。

| snapshot | `J*_center` | bank: selection→audit | bank: audit clairvoyant | selected bank gap |
|---|---:|---:|---:|---:|
| episode_000 | 6.826 | 8.629 | 8.629 | 1.802 |
| episode_021 | 1.182 | 1.458 | 1.391 | 0.276 |
| episode_042 | 3.314 | 7.270 | 7.270 | 3.956 |
| episode_063 | 9.811 | 11.070 | 11.025 | 1.259 |
| episode_084 | 7.986 | 9.350 | 9.320 | 1.364 |
| mean | 5.824 | 7.555 | 7.527 | 1.732 |

五帧 unrestricted center 都退化为对应 `J*_16` 本身。因为 MPPI candidates 保留零噪声
center，softmax 在该局部高度集中到 direct optimum 附近，mean `J*_center-J*_16` 仅
0.010。当前 bank 即使直接看 audit 的 clairvoyant mean 仍为 7.527，距 5.824 有 1.703；
按 selection seeds 选择后为 7.555，gap 1.732。selection/audit 的额外误选只有 0.029，
所以这五帧首先是 bank coverage，而不是 seed 排序问题。

这个五帧 pilot 一致指向 center generator/bank 覆盖不足，而非 16-knot 或 MPPI 加权本身。
全部 validator 最大复算误差为 0，正式版偏离 `D0`。`v1` 误用完整 sigma；`v2` 虽为
`0.10 sigma`，但用了 iid noise，而实际 MPPI 是 zero/extra/antithetic pairs，两者均按
`D2` 作废。`v3` 统一复用正式 antithetic generator，才可用于当前 probe 策略归因。

```text
scripts/model_verify/generate_dbm_sampling_center_gt_pilot.py
scripts/model_verify/validate_dbm_sampling_center_gt_pilot.py
outputs/mppi_proposal/dbm_sampling_center_gt_pilot_20260806_v3
outputs/mppi_proposal/dbm_sampling_center_gt_episode{000,021,063,084}_20260806_v3
```

### S1-C policy gap pilot

在同一五帧/context 0 上，两个 selection probe seeds 分别产生一次 4-probe 决策，最终统一
使用四个独立 audit seeds 评分，共 10 个 probe contexts。所有方法均使用 64 candidates/
center、`0.10 sigma` 和相同 bank。

| 方法 | mean audit cost | 相对 unrestricted gap |
|---|---:|---:|
| unrestricted center | 5.8238 | 0 |
| bank audit clairvoyant | 7.5269 | 1.7031 |
| learned feedback 4-probe | 7.5676 | 1.7437 |
| fixed-priority 4-probe | 7.6034 | 1.7796 |
| full-bank one-seed probe | 7.5524 | 1.7286 |
| guided anchor | 10.5498 | 4.7259 |

feedback 比 fixed 好 0.0358，距 bank clairvoyant 仅 0.0407；相反，bank clairvoyant 距
unrestricted 仍有 1.7031。该 pilot 上继续优化 Actor/Critic 排序最多只能回收很小一段，
主要 block 已明确为 **运行时 center generator/action bank 覆盖**。full-bank 单 seed 甚至
没有消除 coverage gap，继续增加同一 bank 的 probe 数也不是主要解法。

当前没有冻结的 sequential state-only checkpoint，所以 `B3-B2` 严格反馈归因仍为
`PENDING`；这不影响“bank coverage 主导”的结论，因为即使使用 audit clairvoyant 仍保留
1.703 gap。实现与结果：

```text
scripts/model_verify/evaluate_dbm_gt_policy_gap_pilot.py
outputs/mppi_proposal/dbm_gt_policy_gap_pilot_20260806_v2
```

偏离 `D0`。下一步按 S2-B 先验证可运行时构造的非局部 center 是否能在 validation oracle
中回收该 1.703 gap，再决定是否围绕新 bank 采集 replay 和重训策略。

### S2-B forward-only dynamic center bank

保持网络结构、33 slots/bank、64 candidates/center、`0.10 sigma` 和 antithetic candidate
design 不变。所有新方向只读取固定 first-pass actions、cost 和 forward rollout 轨迹；DBM
解析梯度只用于离线 unrestricted GT，不进入 generator。比较项包括：

- `B0`：当前 8 directions × 4 radii；
- `B2`：用局部 cost fit 自适应当前方向和 trust step；
- `B5`：当前方向与 first-pass diverse elites 混合；
- `B6`：按 early/middle/late、position、yaw、vx、action-rate 的轨迹 residual-response fit；
- `B7`：当前方向与 residual-response 混合。

正式 `v3` 使用 selection seeds `29511--29512`、独立 audit seeds `29521--29524`：

| bank | selected audit mean | audit clairvoyant mean | 相对 B0 的逐帧平均 recovery | 胜过 B0 的帧数 |
|---|---:|---:|---:|---:|
| B0 current | 7.5818 | 7.5150 | 0% | 0/5 |
| B2 adaptive current | 7.4632 | 7.3949 | 8.3% | 4/5 |
| B5 current + elites | 7.3491 | 7.2726 | 16.3% | 5/5 |
| B6 residual-response | 7.0731 | 7.0731 | 23.6% | 5/5 |
| B7 current + response | 7.0933 | 7.0683 | 19.8% | 4/5 |

unrestricted audit mean 为 `5.8250`；按 aggregate mean gap 计算，B6 回收
`(7.5150-7.0731)/(7.5150-5.8250)=26.2%`。预先定义的 50% recovery gate 对应
`≤6.6700`，没有 bank 通过。B6 五帧 cost 依次为 `7.925/1.339/6.013/10.888/9.199`，
相对 B0 clairvoyant 全部改善，但 2.4/2.8 m/s recovery 只有 `14.6%/8.9%`。

`v4` 只把 response 最大步长由 2.0 放到 4.0，并更换 selection/audit seeds；B7
clairvoyant `7.524→7.042`，aggregate recovery 约 28.3%，仍未达到 `≤6.674` gate。
因此瓶颈不只是半径太小，而是单轮局部 response 对非线性轨迹响应的方向覆盖仍不足。
纯 segmented scalar-cost fit 明显退化，不进入后续 bank。

```text
scripts/model_verify/evaluate_dbm_dynamic_generator_oracle.py
scripts/model_verify/validate_dbm_dynamic_generator_oracle.py
scripts/model_verify/plot_dbm_dynamic_generator_oracle.py
outputs/mppi_proposal/dbm_dynamic_generator_oracle_20260806_v3
outputs/mppi_proposal/dbm_dynamic_generator_oracle_20260806_v4
```

validator 对五帧 center/cost 的最大复算误差均为 0。该步为 `FAIL/D1`：实现方向符合计划，
且存在稳定改善，但没有通过预设 coverage gate。因此不启动 replay 或 Actor/Critic 训练；
下一次只验证多轮 forward-response，即每个真实 probe 后用新轨迹误差重新构造下一轮中心。

### S2-C/S3-A continuous-center SAC 切换、旧 wrapper pilot 与目标修正

用户确认离散 33-slot Actor 更接近受限 Q-learning，不符合“策略直接生成采样中心”的最终
目标。新增网络：

```text
TorchMPPIContinuousCenterEncoder
TorchMPPIContinuousCenterActor
TorchMPPIContinuousCenterCritic
```

Actor 为 514,208 参数的 squashed Gaussian，直接输出 16-D normalized residual；每个
twin Q 为 641,921 参数并接收连续 action。默认 `maximum_delta_sigma=2.0`，覆盖当前 T1
standardized component 的约 99%，最终 center 仍 clip 到动作边界。单元 smoke 验证了
action/center shape、finite log-prob、Actor-through-Q gradient 和 checkpoint reload。

bootstrap 不做 offline-Q Actor update。v1 克隆 T1 center：test teacher-action RMSE
`0.252`，五帧 fresh audit 的 anchor/Actor/T1/bank-selected mean 为
`11.474/10.874/10.733/7.599`；Actor 在 episode_021/042/084 退化，尤其 042 为
`7.717→15.907`。原因是 T1 是原 warm 上的单步 teacher，不条件化当前 first-pass
feedback，不适合作为第二次 probe 的唯一 BC target。

v2 改为克隆每个 feedback context 中旧 33-bank 的 selection oracle；旧 bank 仅提供 target，
Actor 输出仍是连续值。test clone-action RMSE `0.0234`、无 saturation。另一组 fresh seeds
上五帧 anchor/Actor/bank-selected/bank-clairvoyant mean 为
`10.819/7.754/7.642/7.550`，Actor 五帧全部优于 anchor，S2-C 通过。其输出距最近 bank
center 仅 `0.014 sigma RMS`，因此这里只是可靠初始化，不宣称连续探索收益。

S3-A 随后实现真正 actor-visited replay：每批 stochastic Actor 产生任意连续 center，
所有新动作先用 fixed DBM、64 antithetic candidates/center 和 paired seeds 计算 reward，
再交替更新 twin Q 与 Actor；Actor loss 使用 SAC entropy 加衰减 BC trust。固定五帧仅用于
验证机制，不作泛化结论。

v2 使用 8 actions/context/iteration、2 reward seeds/action，共收集 320 个连续新中心。
Critic Q 从初始支持集外的错误尺度快速回到合理范围，但 checkpoint selection 仍选择
iteration 0，最终 Actor 没有超过 BC clone。关键诊断是把 actor-visited replay 在独立
selection seeds 重排后再 audit：

| 方法 | 五帧 audit mean |
|---|---:|
| 初始 continuous Actor | 7.772 |
| SAC Actor（best checkpoint） | 7.772 |
| actor-visited replay selected | **7.251** |
| old bank selected | 7.593 |
| old bank audit clairvoyant | 7.584 |

replay 在 episode_000/042/063/084 相对初始分别改善
`0.484/1.065/0.127/0.974`，说明连续探索已经找到 bank 外 headroom；当前失败在 Critic/Actor
提取，而不是 action freedom。v3 将 Actor 更新从每批 80 次降到 4 次、学习率降至 `3e-5`
并把 entropy temperature 降至 `0.001`，仍未改善 Actor。更重要的是，另一组独立
collection/selection/audit seeds 下，同样从 320 个动作选择的 replay winner audit mean
变为 `14.626`。两次结果方向不一致，表明 2 reward seeds/action 在大量连续动作中选择
winner 会产生严重 winner's curse；不能拿 v2 oracle 调 Actor 或宣称连续策略已通过。

```text
scripts/model_verify/train_mppi_continuous_center_sac.py
scripts/model_verify/evaluate_mppi_continuous_center_bootstrap.py
scripts/model_verify/fine_tune_mppi_continuous_center_sac_pilot.py
outputs/mppi_proposal/continuous_center_sac_bootstrap_20260806_v{1,2}
outputs/mppi_proposal/continuous_center_sac_bootstrap_eval_20260806_v{1,2}
outputs/mppi_proposal/continuous_center_sac_fixed5_pilot_20260806_v{1,2,3}
```

上述 stochastic-wrapper v2/v3 现在只作为随机 MPPI wrapper 的历史诊断，对新的 Direct Actor gate 为
`INVALID/D2`，不能再通过增加 reward repeats 修补。新入口
`fine_tune_mppi_direct_center_sac_pilot.py` 把主 reward 改为唯一 direct cost，并从头初始化
Critic；`mppi.py` 新增可选 `fixed_hadamard_64`，训练/运行共用候选生成器。完整五帧 pilot
重复 direct cost 最大误差为 0，但 320 个宽高斯探索动作没有超过初始 Actor，final 仍选择
iteration 0（mean 9.782）。所以当前状态仍为 `S2-C PASS / S3-A FAIL(D1)`，但 block 已从
“reward seed 不足”修正为“Direct Actor 局部探索覆盖与 Q_direct 学习”。

随后已按用户决定把策略本身改为 `TorchMPPIDeterministicCenterActor`：从旧 SAC Actor
精确迁移 encoder 与 mean head，删除 log-std head；部署输出始终唯一。训练期在该输出周围
循环16个满秩 Hadamard 方向，每轮4对正负样本，探索半径独立衰减。v2 用
`0.30→0.05 sigma` 得到 mean `8.866`，但第5轮后 Actor 利用 Critic 继续发散；保守 v3 用
`0.15→0.03 sigma`、Actor lr `1e-5`、每轮1次 Actor update，24轮 cost 单调降到 `8.919`，
replay best `7.562`，并把两帧回退压到 `0.103/0.031`。当前改记为
`S3-A mechanism PASS / generalization PENDING`；下一步扩大独立状态覆盖，而不是恢复高斯头。

### 2026-08-06 episode-heldout Direct Actor 两轮更新

已把 fixed-five 机制扩到 90/15/15 episode split。train/validation 为 1,800/300
snapshots，每帧两个 feedback context，共 3,600/600 contexts；`episode_105--119` test
从生成、训练和校准中始终封存。每轮 sidecar 的每个 context 有 164 个确定性 center：
anchor、33 个旧 bank、T1、当前 Actor，以及
`0.03/0.06/0.10/0.15 sigma × 16 Hadamard directions × +/-` 的 128 个局部样本。
每轮共 688,800 次 direct DBM rollout，局部 rank 最小为 16，平均裁剪率约 1.3%。全量
validator 对 2,100 snapshots/4,200 contexts 的 center 与 cost 最大误差均为 0。

第一轮 Critic validation correlation/regret 为 `0.572/7.232`。无信任步长的 Actor 在
epoch 5 为 `27.445→26.310`，继续更新会严重利用 Critic 外推。固定参数插值仍输出单个
511,120 参数 deterministic Actor；校准结果为：

| alpha | validation mean | median gain | P05 gain | win |
|---:|---:|---:|---:|---:|
| 0.125 | 26.849 | 0.016 | -1.350 | 56.0% |
| 0.375 | 25.974 | 0.026 | -4.762 | 53.2% |
| 0.500 | 25.723 | 0.013 | -6.708 | 50.8% |
| 1.000 | 26.310 | -0.162 | -17.058 | 43.7% |

均值约束选择为 `alpha=0.5`；第二轮 on-policy 采集使用预先定义 P05 floor `-5` 下的保守
`alpha=0.375`。围绕它重新生成 v2 sidecar 后，Actor/local-best 全 context mean 从 v1
的 `19.967/10.387` 改善到 `17.736/9.856`。朴素第二轮随机重置 Critic 会把 validation
regret 从继承 Q 时的 8.00 恶化到 24.33；训练器现强制 Critic/Actor epoch 0 参与
checkpoint 选择。

普通 scalar-Q 回归会淹没局部正负 pair 的小 cost 差。v4 增加 forward-only
finite-difference slope loss：用同半径、同 Hadamard 方向的正负 direct reward 差监督 Q
局部斜率，不读取 DBM 解析梯度。Actor 使用 local-oracle BC、`Q weight=0.1`、trust
penalty 10 和 `2e-6` 学习率。最终结果：

| 指标 | 第一轮保守 Actor | 第二轮 v4 | 第二轮 alpha=0.75 |
|---|---:|---:|---:|
| validation direct mean | 25.974 | 25.525 | **25.518** |
| 第二轮 mean gain | - | 0.449 | **0.456** |
| 第二轮 median gain | - | 0.039 | 0.033 |
| 第二轮 win fraction | - | 57.2% | **58.7%** |
| 第二轮 P05 gain | - | -3.407 | **-2.457** |
| 第二轮 worst gain | - | -157.260 | -109.109 |

v4 Critic validation global correlation/regret 为 `0.559/8.873`，local slope
correlation/direction agreement 为 `0.333/60.9%`。从最初 Actor 27.445 到第二轮校准
25.518，累计改善 1.926（约 7.0%）。当前记为
`episode-heldout two-round mechanism PASS / tail-safety FAIL`：P05 可控而单个最坏状态
仍有大回退。checkpoint 仅为 `VALIDATION_ONLY_TEST_SEALED`，不得进入 test、Query/ONNX、
ROS 或长闭环资格。下一步先定位最坏 episode/context，并训练 cost/risk 双头或
conservative fallback，不再仅增加 Actor epoch。

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/dbm_direct_center_replay_diverse_20260806_v{1,2}
scripts/model_verify/generate_dbm_direct_center_replay.py
scripts/model_verify/validate_dbm_direct_center_replay.py
scripts/model_verify/train_mppi_direct_center_actor_critic.py
scripts/model_verify/calibrate_mppi_direct_actor_trust_step.py
outputs/mppi_proposal/direct_center_actor_critic_diverse_20260806_v4
outputs/mppi_proposal/direct_center_actor_trust_step_20260806_v2
```

tail 不是均匀噪声：最坏 context 集中于 validation 的 2.4--2.8 m/s 高动态段。
`episode_101/step_000262/context 1` 在 reference 2.4 m/s、实际 vx 3.135 m/s 时由
143.683 退化到 252.793；其后主要是 episode_101 的 overspeed/yaw-rate recovery，以及
episode_102/103 的大 heading error 或高 yaw-rate。

固定 DBM 下可用一个无随机 seed 的二中心 guard：分别 direct-rollout 上一轮保守 center 与
新 Actor center，选择预测 cost 较低者。validation mean 为 **24.661**，相对第一轮保守
25.974 改善 1.313；P05/worst gain 都为 0，新 Actor 使用率 58.7%。从原 bootstrap
27.445 累计改善 2.784（约 10.1%）。这只是 perfect-DBM model-selection upper bound，
不计入 Direct Actor 25.518 的成绩；Query 必须另行验证二中心 cost 排序后才能启用。

```text
scripts/model_verify/evaluate_mppi_direct_actor_guard.py
outputs/mppi_proposal/direct_center_actor_two_center_guard_20260806_v1
```

### 2026-08-06 fixed-DBM validation 数值最优基线

为区分“策略没有学好”和“16-knot 表达能力不足”，现已在完整 validation split 上直接优化
固定 DBM cost。validation 有 300 个独立 snapshot；每帧有两个 feedback context，但 direct
DBM objective 与 feedback probe 无关，所以数值最优只对每帧求一次，再与 600 个 Actor context
成对比较。test `episode_105--119` 仍未加载。

优化使用 warm、zero、T0 best/soft、T1 teacher 和 3 个随机初值。先对 Actor 相同参数化的
16 维 knot（8x2）优化 400 步，再把较优结果插值为完整 50x2 action 并优化 500 步。这里的
`J16/J100` 是多初值非凸优化得到的 **best-found numerical oracle**，不是数学上已证明的全局
最优。第二轮统一 refinement 后，最后10步相对改善大于0.1%的帧为 J16 `3/300`、J100
`0/300`；独立重放 validator 的插值、cost 和 parent-regression 最大误差均为0，而且每帧
J100 均严格优于 J16。

| 方法 | mean cost | median | P95 | max | 相对 warm→J16 可改善量的恢复率 |
|---|---:|---:|---:|---:|---:|
| warm center | 29.517 | 21.964 | 75.897 | 300.921 | 0.0% |
| 第一轮保守 Actor | 25.974 | 9.914 | 113.737 | 428.592 | 14.4% |
| 当前第二轮 Actor | 25.518 | 9.883 | 110.865 | 371.194 | 16.2% |
| DBM two-center guard | 24.661 | 9.823 | 104.603 | 371.194 | 19.7% |
| T1 teacher | 12.762 | 9.395 | 31.133 | 119.479 | 68.0% |
| J16 best-found | **4.894** | 3.227 | 14.524 | 28.523 | 100.0% |
| J100 best-found | **4.697** | 3.189 | 13.942 | 26.672 | 100.8% |

当前 Actor mean 比 teacher 高 `12.756`，约为 teacher 的 `2.00x`；它只在 `48.3%` 的
context 上优于 teacher。Actor/teacher 相对 J16 的平均 gap 分别为 `20.624/7.868`。
按参考速度分组后，Actor 在 1.2/1.6 m/s 的 mean cost 为 `3.059/5.809`，优于 teacher 的
`4.798/6.187`；但在 2.0/2.4/2.8 m/s 变为 `17.223/39.528/61.974`，明显差于 teacher 的
`11.482/17.137/24.207`，且 2.8 m/s 已差于 warm 的 `48.595`。因此当前主要瓶颈是
高速度 recovery/tail 的策略泛化，而不是 16-knot 维度：J16 到 J100 只改善 `0.198`，远小于
Actor 到 J16 的 `20.624`。

```text
scripts/model_verify/generate_dbm_direct_gt_validation.py
scripts/model_verify/validate_dbm_direct_gt_validation.py
scripts/model_verify/compare_mppi_direct_actor_teacher_gt.py
outputs/mppi_proposal/dbm_direct_gt_validation_20260806_v2
outputs/mppi_proposal/direct_actor_teacher_gt_validation_20260806_v1
```

计划据此更新：不再把“Actor 参数上限”或“16-knot 自由度不足”作为当前主假设。下一步优先
用 J16 oracle 量化每个速度/状态的可达 gap，并针对 2.0--2.8 m/s recovery 状态补充局部
on-policy coverage、风险/保守更新或 deterministic fallback；通过 heldout tail gate 后，才进入
Query/ONNX 和闭环资格。

### 2026-08-07 J16 train oracle 与 2σ/6σ 蒸馏诊断

已为90个独立 train episodes 的1800个 snapshot 生成 J16-only 数值标签，每帧单独优化。每帧8个初值，
250步首轮加统一150步 refinement；最终 train mean `warm/teacher/J16 =
27.918/11.758/4.830`。独立 validator 重放1800/1800帧，插值和cost最大误差均为0；
refinement 最后10步仍改善超过0.1%的帧为35/1800，没有帧超过1%。test未生成或读取。

Actor当前 `anchor±2σ` 并不等于完整J16动作域。validation 中J16可达率仅37.5%，把J16
逐元素投影到2σ盒后的 mean cost 为12.004；扩大到6σ后可达率98.8%，投影cost为4.895，
基本等于J16 4.894。train 对应的6σ可达率为99.0%，投影cost为4.832。因此6σ已消除本批
数据上的输出范围瓶颈，无需先修改网络宽度。

保持同一511,120参数 deterministic Actor，用train-only J16投影标签、三seed、300 epoch
蒸馏。formal validation oracle动作在六个网络全部冻结后才首次加载。结果为：

| 版本 | train mean cost | validation mean cost | validation action RMSE |
|---|---:|---:|---:|
| 2σ 最好seed | 12.631 | 44.645 | 0.258 |
| 6σ 最好train seed | **7.263** | 46.340 | 0.099 |
| 6σ 最好validation seed | 7.333 | **42.517** | 0.098 |
| 当前两轮 Actor（对照） | - | **25.518** | - |
| T1 teacher（对照） | - | **12.762** | - |
| J16 | 4.830 | **4.894** | 0 |

为排除“300 epoch过拟合且没有合理选epoch”，又在train内部按30个
`速度×scenario class` strata 各用2个episode拟合、1个episode选epoch，并继续跑满300
epoch。2σ最优epoch155--221，6σ为85--157，但formal validation没有改善，6σ最好仍为
48.580。这证明问题不是简单early stopping；全量90 episode训练反而略有帮助。

J16 top2不是主要多中心问题：validation 只有2/300帧同时满足top2 cost差<1%且knots距离
>0.5σ。但目标随状态变化很快，相邻保存帧的J16 knots有68.8%变化超过1σ。train/validation
的vx、vy、yaw-rate、横向/航向误差、动作和history边际范围基本重叠，所以是独立动态状态
的联合覆盖不足和cost敏感性，而非一个标量超出训练范围。

对validation最佳6σ蒸馏Actor沿直线插值到真实J16，mean cost在
`alpha=0/0.5/0.75/0.9/0.95/1.0` 时为
`42.517/16.913/8.630/5.575/5.071/4.894`。Actor方向并非完全错误，但高速轨迹要求输出
非常接近J16；普通knots Huber/MSE不能反映小动作误差造成的巨大轨迹cost。

当前 gate 更新为：`6σ support PASS / train fit PASS / episode generalization FAIL /
J16-MSE distillation REJECTED / test sealed`。不得部署这些蒸馏checkpoint。下一步不再增加
同一episode的相邻帧或继续MSE epoch，而是：

1. 把30个速度/场景strata的独立train episode从每层3个扩到至少10--12个，每episode仅取
   4--6个充分分散的fully-observed snapshot；
2. 用forward-only antithetic/Hadamard cost计算局部曲率或verified improvement target，训练
   cost-sensitive Actor loss，而不是等权knots MSE；
3. 按高速度regret和recovery状态加权，但保持formal validation与test封存；
4. 只有新validation Actor先低于teacher 12.762并改善P95/max，才恢复Critic主导更新。

```text
outputs/mppi_proposal/dbm_direct_gt_train_20260807_v2
outputs/mppi_proposal/direct_center_j16_distillation_20260807_v1
outputs/mppi_proposal/direct_center_j16_distillation_heldout_20260807_v2
scripts/model_verify/train_mppi_j16_oracle_distillation.py
scripts/model_verify/analyze_mppi_j16_distillation_path.py
```

### 2026-08-07 FR-TRPI 设计覆盖：当前唯一允许的 Actor 下一步

独立状态扩充、J16/local-curvature supervision 和 cost-sensitive Actor 已完成；完整结果见
`mppi_sampling_center_handoff_20260802.md`。新 Actor 的平均方向有用，但 full-step 在
2.4--2.8 m/s 过冲。当前下一步不再是本文件较早 §5 所述的普通 episode-level Critic
更新，而是：

```text
forward-rollout deterministic TRPO-like policy iteration (FR-TRPI)
```

完整冻结项、公式、sidecar、训练 loss、gate 和失败分支以
[Direct Actor FR-TRPI 设计](mppi_direct_actor_trpo_like_design_20260807.md)为准。

其核心合同是：

1. 冻结当前 deterministic Actor `pi_k` 和一个 forward-only proposal Actor；
2. 用 source MPPI sigma 归一化新旧 center 差，并投影到逐 context trust radius；
3. 仅在 train contexts 上用固定 `alpha=0.00:0.05:1.00` 做确定性 DBM direct rollout；
4. `alpha=0` 始终保留，选择 conservative safe target；
5. 用同一个 511,120 参数 deterministic Actor 学习 safe target，不新增 log-std；
6. Actor 冻结后才在 formal validation 评价实际唯一输出，不做 validation 逐状态 line
   search 或二中心选择来冒充单 Actor 成绩。

进入实现前的 TR0 机制上限已验证。current→cost-sensitive proposal 的 validation
逐 context line oracle 为：

| speed | current | line oracle | mean gain | alpha=0 fraction |
| ---: | ---: | ---: | ---: | ---: |
| 1.2 | 3.059 | 2.420 | 0.639 | 4.2% |
| 1.6 | 5.809 | 4.094 | 1.714 | 15.0% |
| 2.0 | 17.222 | 9.613 | 7.610 | 20.0% |
| 2.4 | 39.528 | 22.569 | 16.959 | 27.5% |
| 2.8 | 61.974 | 31.549 | 30.425 | 20.8% |
| all | 25.518 | 14.049 | 11.469 | 17.5% |

该 oracle 只证明逐状态 trust step 有上限，不是 policy 分数。下一执行状态为：

```text
TR0 line-oracle gate: PASS
TR1 immutable sidecar + validator: PASS
TR2 safe-target Actor fit: FAIL
TR3 formal validation Actor gate: PENDING
```

TR3 必须同时满足 paired mean gain 95% CI 下界大于 0、median>=0、P05>=0、
worst>=-5，并且 2.4/2.8 m/s 与 recovery mean 不退化。FAIL 时保留旧 Actor，只允许按
设计文档减小 trust radius、加强 stay 标签/拟合或更换 forward-only proposal direction。
不得跳到四轮、Query、ONNX、ROS 或闭环。

TR1 于 2026-08-07 完成。train-only sidecar
`/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/dbm_direct_trust_region_train_20260807_v1`
包含 360 个 episode（300 internal-fit / 60 internal-selection）、3,150 个快照、6,300 个
contexts 和 132,300 个 direct rollout；formal-validation/test 共 30 个 episode 未生成标签。
独立全量复算为 center/cost 最大误差 `0/0`、metadata 最大误差 `5.914e-7`，资格为
`TR1_VALIDATED`。train-only safe-label mean cost 为 `19.411 -> 12.345`，2.4/2.8 m/s
分别为 `26.838 -> 16.344`、`48.740 -> 31.432`。这些是标签上限而不是 Actor 成绩。
下一步固定为 TR2 safe-target Actor fit。

TR2 已于 2026-08-10 执行并按 gate 停止。3-seed、200-epoch 同结构 Actor 选中 seed 0 /
epoch 135；internal-selection mean cost `25.355 -> 22.680`，但 stay recall `0%`、gain
P05/worst `-11.636/-294.542`、trust violation `3.33%`。独立 6,300-context 重放指标误差
为 0，资格为 `TR2_FAIL_STAY_TRUST_AND_TAIL`。提高 stay 权重能改善 P05 但不能消除
worst；旧/新参数步扫描中最小非零 alpha 0.05 的 worst 已为 `-10.882`。因此不得进入
TR3。下一实验必须先选择显式 state-conditioned alpha/stay gate、verified two-center
fallback 或更换 proposal direction；不得继续只调 epoch/MSE/权重。
