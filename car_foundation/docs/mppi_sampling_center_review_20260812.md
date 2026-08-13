# MPPI 采样中心策略训练评审：结论 / 风险 / 推荐计划（2026-08-12）

本文是对 `car_foundation/docs/` 下 MPPI 采样中心系列文档的一次外部评审记录，
基于以下来源整理：

- `mppi_sampling_design_and_current_plan_20260807.md`（汇总视图）
- `mppi_sampling_center_handoff_20260802.md`（主交接，历史与失败记录）
- `mppi_direct_actor_trpo_like_design_20260807.md`（FR-TRPI 详细设计）
- `mppi_sequential_probe_execution_plan_20260806.md`（权威执行入口）

本文只记录评审结论、风险与建议，**不改动任何契约、gate 或数据**；
执行层面仍以 `mppi_sequential_probe_execution_plan_20260806.md` 为准。

---

## 1. 结论

### 1.1 方向与工程纪律：合理

- 问题定义合理：基线 ESS≈1.44/256 → 学"采样中心"`μ*=argmin_μ E_ε[C(μ+σε)]`
  （amortized optimization），而不改 MPPI 求解器，方向正确。
- 瓶颈定位可信：GT-first 五层拆解证明 J16→J100 仅差 0.198，
  **瓶颈是高速段 recovery 与尾部泛化，不是 knot 维数**。库覆盖 gap=1.703 是主结构损失。
- 工程纪律高于一般研究仓库：不可变原始数据 + sidecar、SHA256、独立 validator 重放
  （误差多为 0）、episode 级切分、selection/audit seed 隔离、预注册 gate、test 封存、
  负结果诚实记录。取消 reward seed 平均以消除 winner's curse（7.251 vs 14.626 翻转）是对的。

### 1.2 训练成果现状：尚未产出可部署结果

- 数值预言机（validation 600 context）：warm 25.5 / 当前 Actor 25.518（研究口径 ~14.3）
  / T1 teacher 12.762 / J*_16 4.894。Actor 只回收 warm→J16 的 **16.2%**，teacher 68%。
- Gate 现状：TR0 PASS / TR1 PASS / **TR2 FAIL**（stay recall 0%、worst −294.542）/
  **TR3 formal validation FAIL**（vs 旧 Actor mean gain 2.049 但 worst −75.937；
  vs TR2-B mean gain 0.071，95% CI [−0.299, 0.390] 含 0）。
- 默认策略仍是**回退到旧 Actor**。formal validation 已被消耗，不得再用于无偏声明。

---

## 2. 风险（按严重度排序）

### R1（最高）闭环从未验证
整个项目全部算力都投在**单步 open-loop 的 J_direct** 上，而团队自定技术路线第 6 条
"固定 DBM 闭环 A/B 是必需"从未执行。单步 proposal cost 与闭环累计代价/失败率/抖动/延迟
的关系毫无证据。更危险的是：warm start 在 44/96 帧本身即 best，
headroom 是否真实存在都未在闭环确认。

### R2 指标口径 + 基线漂移，纵向数字不可比
cost 口径从"MPPI weighted-output（多 seed）"换成"J_direct（单次确定性）"；
warm 基线随数据集在 12.5→11.8→21.6→29.5→25.4 间变化。文档多处把
25.518 / 14.316 / 12.762 / 4.894 混排并列，易被误读为单调进步。

### R3 选择集被反复消耗，小差值结论不可信
600 个 internal-selection context 被 TR2/TR2-B/Alpha-AC/local-gradient 等十余轮
反复用于选 epoch/threshold/alpha/radius，最后又用同一集合宣称"超过 safe teacher 0.0036"
——差值远小于选择噪声。formal validation 在 TR3 后已 consumed，但 25.518 仍继续被引用为基线。

### R4 Oracle / 预算不对等造成"进步感"
T1(2048 rollout CEM)、J16(数百步数值优化)、two-center guard、alpha 线 oracle、
"完美符号选择器"在**推理时均不可达**，却与 Actor 数字同表并列。probe 方案用
256–2112 rollout/context 与 64-candidate 的 warm/teacher 比较，"probe 超过 teacher"
实质是把在线搜索预算抬高 4–33 倍。

### R5 重尾 + 小样本 + 极少训练 seed
worst 达 −294 / −369 / +529，样本仅 300–600，训练 seed 普遍只有 3，除 TR3 外几乎不报 CI。
反复出现"mean 改善但 P05/worst 恶化"（如 Actor 赢 464/600 却 mean −0.352）。
部分 PASS 仅基于 **5 帧**（"3/5 改善"）。

### R6 换方法而非换诊断
2.4/2.8 m/s 的系统性失效从 08-05（−3.2/−8.0）到 08-12（cosine 0.257/0.279）是同一失败模式，
中间换了 categorical→continuous→alpha 一维→局部二次四种算法，
但始终未先解决"高速状态覆盖不足/条件化"。文档自诊断"相邻帧梯度 cosine 仅 0.044、
1800 状态不足以泛化 16 维梯度"，却继续在同一批旧状态上叠加海量标签
（6.7M/40M/44M/81M rollout），重复投入同一瓶颈。

### R7 已发现的实现 bug 使部分历史结论作废
v3/v4 的 LR 消融因 optimizer 状态覆盖实际同 LR；S1-B v1 用错完整 sigma、v2 用错 iid noise。
此类事后审计出现两次，提示其他"单变量实验"的超参归因也需存疑。

---

## 3. 推荐计划

### P1（优先）先做最小闭环 A/B 验证 —— 提到 Query/ONNX 之前
用当前 Actor（即使仅 25.518 口径）在固定 DBM 上跑一次最小闭环 A/B：
warm-only vs Actor-center，记录累计 cost、失败率、控制抖动、每步延迟。
目的不是拿好成绩，而是**确认 open-loop 的 J_direct 收益能否传导到闭环**，
以及 headroom 是否真实。若不能传导，则应重新审视整个 open-loop 目标本身。

### P2 停止在被消耗的选择集 / 旧状态上追加算力
- 不再在旧 1800/3150 状态上叠加新标签（同一瓶颈）。
- 按计划**新采集一批未触碰的 validation-like episode 作为无偏 gate**，
  重点覆盖独立的 2.0–2.8 m/s、overspeed、heading-recovery、高 yaw-rate 状态。
- 保持 internal-fit / internal-selection 隔离并引入 cross-fitting。

### P3 统一指标口径与呈现
- 固定一个报告口径（建议部署侧安全门口径），所有纵向数字用同一坐标系。
- 报表中把 **oracle（不可达）** 与 **Actor（可达）** 明确分栏，并标注各自的 rollout 预算，
  避免预算不对等的"里程碑"叙述。
- 关键成绩强制附 episode-bootstrap 95% CI 与 P05/worst，不单用 mean 或胜率。

### P4 尾部/高速专项：先覆盖后条件化
- 用 J*_16 按速度量化可达 gap，确认 2.0–2.8 m/s 是覆盖问题还是策略容量问题。
- 训练**有界的 stay/sign/step 头**（保留 exact stay fallback），
  而非继续无约束 Actor 更新或单纯加 epoch。
- 增加 cost-increase 的 lower-confidence-bound / 保守上界头，把 worst 收敛到有界。

### P5 gate 通过顺序（保持不变，但把 P1 前置）
```
最小闭环 A/B (P1，新增前置)
  → 新无偏 episode-heldout gate (mean 改善 且 P05≥0 且 worst 有界)
  → Query 重标注 / Query 侧验证
  → mppi_proposal_runtime.py + proposal ONNX 导出 (≤50 ms/step)
  → 固定 DBM 完整闭环 A/B
  → 封存 test 一次性评估
```

### P6 明确暂缓项（记录，不在本阶段动）
- 100 km/h（27.78 m/s）速度范围扩展：需先单独确认车辆尺度、DBM 参数、赛道曲率、
  动作/加速度边界与 horizon，建立独立数据与验证契约，不得混入现有 train/val/test。
- 自适应 probe 排序：S1-C 已证明收益 ≈0.036，不再投入。
- DBM 解析梯度：仅作理论下界，禁止进入部署路径。

---

## 4. 一句话总结

方向与纪律合理，瓶颈定位可信；但当前策略网络训练**尚未产出可部署成果（TR3 FAIL）**。
最需修正：(1) 尽早最小闭环 A/B 验证 open-loop 收益是否真实；
(2) 停止在被消耗的选择集/旧状态上追加算力，改为补高速独立覆盖 + 全新无偏 gate；
(3) 统一指标口径，oracle 与可达成绩分栏。

---

## 5. Codex 复核补充（2026-08-12）

本节是在保留上述外部评审原文和原有执行契约的前提下，对评审结论、当前代码接入状态和
已有实验产物做的二次核对。它不修改已有 gate，不解封 formal validation/test，也不把新的
诊断结果提升为部署结论。

### 5.1 总体判断

外部评审指出的核心证据缺口成立：**learned sampling-center 策略尚未做过固定 DBM
闭环 A/B**。当前已经证明的是 frozen-state 条件下的参数化上限、数值 oracle、局部方向和
尾部问题；尚未证明的是：单步 `J_direct` 改善能否经过 MPPI 候选采样、soft weighting、
控制执行和下一时刻 warm-start 递推，转化为闭环累计 cost、跟踪误差、控制平滑性和失败率
改善。

因此，继续大规模增加 Critic/Actor 标签前，应插入一次诊断性质的同预算 wrapper 验证和
短闭环 A/B。这是当前实验顺序需要修正的地方，但不意味着此前所有 open-loop 结论作废：

- `J16 -> J100` gap 很小，说明 16-knot 参数化不是主要瓶颈；
- J16/T1 证明单步动作空间存在明显 headroom；
- local-gradient step 的正负对照证明 Critic 已得到平均有效方向；
- 这些结论仍是有效的机制证据，只是尚未获得部署/闭环资格。

### 5.2 对 R1--R7 的逐项复核

#### R1：核心成立，但应限定为 learned center 闭环未验证

仓库已有 DBM/Query 基准闭环、完整 trace 和 warm-start MPPI 数据采集，因此不能表述成
“项目没有做过任何闭环”。准确说法是：**网络生成的 sampling center 从未接入控制循环做
配对 A/B**。当前 `car_node.py` 直接使用 `mppi_running_params` 调用 MPPI，没有 proposal
policy checkpoint、网络 center 注入或 proposal fallback 的 runtime 入口。

`44/96 warm candidate best` 也不能证明没有 headroom。它只表示 warm 在 T0 已存储的
256 个局部候选中为 best，不是全局最优，也不是 J16 参数化下的 best-found numerical
oracle。T1/J16 已证明 frozen-state 单步 headroom 存在；仍待验证的是该 headroom 在固定
rollout 预算和闭环递推下是否可用。

#### R2：成立，且需进一步拆开 25.518 与 14.312

`25.518`、`14.312`、`12.762`、`4.894` 不能被解释成同一 split、同一预算下的连续进步：

| 数值 | 含义 | 数据/资格要点 |
| ---: | --- | --- |
| 25.518 | 旧 Direct Actor 的 direct cost | historical formal-validation 结果；tail FAIL，test sealed |
| 14.312 | 最新窄范围 16-D residual Actor | 后续 train-internal selection 研究结果，不是新的 formal-validation 成绩 |
| 12.762 | T1 高预算 teacher | 不可在正常 runtime 预算下直接生成 |
| 4.894 | J16 best-found numerical oracle | 多初值数值优化上界，不是已证明的全局最优或可部署策略 |

后续所有纵向表格至少要固定并展示：`objective`、`dataset/split`、`rollout budget`、
`checkpoint qualification` 和 `reachable/oracle`。`weighted-output` 与 `J_direct` 必须分表，
不能只按 cost 数值大小混排。

#### R3：大体成立，但历史基线可以继续引用

internal-selection 已多次用于 epoch、alpha、radius、threshold 和 Critic 配置选择，因此这些
context 对后续调参已经 consumed。TR3 后 formal validation 也已经 consumed，不能再支持
新的无偏泛化声明。

继续引用 `25.518` 作为 historical baseline 本身没有数据泄漏，但必须明确标记
`historical/consumed`，不得把相对它的新差值当成新的正式 gate。下一次正式资格判断需要新采集、
按 episode 隔离、此前完全未触碰的 validation-like gate。

#### R4：成立，但 oracle 仍应保留为诊断列

Teacher、J16、line oracle、perfect sign selector 和 perfect two-center guard 的计算预算在
runtime 不可达，不能与 Actor 放在同一“部署成绩”列中。但这些结果仍可用于分解：动作 support、
策略拟合、方向、步长和 tail gate 分别损失多少。

合理做法不是删除 oracle，而是分成两栏并明确预算：

- `runtime-reachable`：Actor forward、MPPI 候选数、是否需要 first-pass feedback；
- `diagnostic oracle`：搜索步数、起点数、DBM rollout 总数以及是否逐状态使用真实 cost 选解。

#### R5：成立

当前结果多次出现 mean 改善而 P05/worst 恶化，且不同速度/恢复状态差异明显。正式结果不能只报
mean、胜率或 context-level bootstrap。应按 episode 做 paired bootstrap 95% CI，并至少报告
median、P05、worst、速度分桶结果和 recovery 子集结果。少量 5-frame pilot 只能作为机制诊断。

#### R6：方向合理，但“始终只用旧状态”已经过时

2026-08-07 已新增 270 个独立 train-only episode 和 1,350 个稀疏 snapshot，30 个
速度×场景分层各增加 9 条独立轨迹；合并后训练源为 360 个独立 episode、3,150 snapshots。
因此训练状态覆盖扩充已经执行，不能再表述为始终只在旧 1,800 states 上追加标签。

仍然成立的问题是：

- 新增数据仍属于 train/internal-selection 范围，缺少新的 untouched validation-like episode；
- 2.4/2.8 m/s、overspeed、heading recovery、高 yaw-rate 的策略泛化仍未通过；
- 在这些失败没有被新 episode-heldout gate 复核前，继续给 consumed states 增加同类标签的
  边际价值很低。

因此 P2 应修订为：旧数据保留用于 fit；停止在 consumed selection/formal validation 上调参；
新增独立 episode 主要用于新的无偏 gate，而不是简单丢弃现有训练集。

#### R7：需要降低结论强度

LR v3/v4 审计证明 optimizer restore 覆盖了命令行 LR，因此“不同学习率导致差异”的归因无效，
但对应训练链的实际数值和独立 replay 并未因此全部失效。S1-B v1/v2 已被显式标成
`INVALID/D2`，并在修正 sigma 和 candidate design 后重跑。

更准确的结论是：**相关单变量超参归因作废，已通过独立 validator/replay 的数值事实仍有效**。
后续需要在 checkpoint/summary 中同时记录 requested/effective optimizer 参数、候选生成 contract
hash 和恢复后的实际状态，避免再次把 continuation 误写成消融实验。

### 5.3 已有产物上的低成本目标传导代理验证

为判断 `J_direct` 与“作为 MPPI sampling center 的实际质量”是否可能错配，复用了已有
`continuous_center_direct_fixed5_pilot_20260806_v1/v2/v3` 的 5 帧结果。没有新增训练、
没有读取 sealed test，也没有修改任何 split。

定义：

```text
direct_gain   = initial_actor_direct_cost
                - deterministic_actor_direct_cost

wrapper_gain  = initial_fixed_neighborhood_weighted_output_cost
                - new_fixed_neighborhood_weighted_output_cost
```

其中 wrapper 使用相同的 `fixed-hadamard-64-v1`、64 candidates 和冻结半径配置；它是确定性的
辅助 MPPI-center 指标，不是 Actor reward。

| 实验 | 帧数 | mean direct gain | direct wins | mean wrapper gain | wrapper wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed5 v2 | 5 | +0.9156 | 3/5 | +0.1331 | 3/5 |
| fixed5 v3 | 5 | +0.8633 | 3/5 | +0.0124 | 4/5 |

逐帧中最明显的反例是 `episode_063`：

| 实验 | direct gain | wrapper gain |
| --- | ---: | ---: |
| v2 | +4.5280 | -0.1409 |
| v3 | +3.9827 | -0.1903 |

五帧上的 Pearson 相关系数分别约为 `-0.499/-0.972`。由于 `n=5` 且这些是早期 pilot，
**该相关系数不能用于正式统计声明**；但 direct 大幅改善而 fixed-neighborhood weighted-output
退化的逐帧反例已经足以触发下一阶段验证。它说明：优化唯一插值动作的 `J_direct`，不保证围绕
该 center 采样并 soft weighting 后的 MPPI 输出同比改善。

本代理验证也不能回答闭环问题，因为它仍是 frozen-state、单步 DBM rollout，且不是对当前
25.518/14.312 checkpoint 在全新 episode 上的正式比较。其作用仅是证明“目标传导”不能被
默认假定。

### 5.4 对 P1 的工程可执行性修订

P1 应前置，但不能在没有最小 runtime 适配器时直接执行。原 P5 把“最小闭环 A/B”放在
`mppi_proposal_runtime.py` 之前，工程顺序存在矛盾。

另外，最新 16-D Actor 的输入不只有 history/reference/current/anchor knots，还包含
`feedback[74]` 和 `gradient_context[32]`。这些输入来自 first-pass rollout/probe 信息，意味着
最新 Actor 的闭环评价必须冻结：

- first-pass 如何生成 feedback/gradient context；
- first-pass 和 final MPPI 各使用多少候选；
- baseline 在相同控制步允许使用的 DBM rollout 总预算；
- first-pass 失败、超时或 Actor 输出异常时的 fallback；
- 本步 Actor center、最终 MPPI center 以及下一步 warm center 的递推规则。

因此最小闭环之前只需要实现一个 **PyTorch + fixed DBM 的诊断适配器**，不需要提前做完整
Query 重标注、proposal ONNX 或部署封装。适配器应把 proposal center 注入 MPPI running
state，并保留原 MPPI 的 final weighted update、shift 和 action bounds。

### 5.5 修订后的验证顺序

建议用以下顺序替换 P5 中存在矛盾的执行顺序；这是对原计划的补充，不自动改变已有 gate 状态：

```text
V0  指标/预算/provenance 台账冻结
  -> V1 同状态、同固定候选库的 wrapper 传导验证（诊断，不碰 sealed split）
  -> V2 最小 PyTorch fixed-DBM proposal runtime 适配器
  -> V3 短闭环 paired A/B（诊断 gate，不作部署声明）
  -> V4 新采集 untouched validation-like episodes
  -> V5 在旧 fit + 新 episode-heldout 契约下继续 Critic/Actor，并冻结新正式 gate
  -> V6 固定 DBM 完整闭环 A/B
  -> V7 Query 重标注 / Query 侧验证
  -> V8 proposal ONNX 一致性、延迟和 fallback
  -> V9 封存 test 一次性评估
```

V1 建议同时评价 historical 25.518 Actor 和 research 14.312 Actor，但分开标注 split/资格。
所有 center 使用完全相同、预先冻结的 candidate perturbation；报告：

- `J_direct` gain；
- MPPI weighted-output gain；
- best/P10/median candidate cost、ESS、clip fraction；
- 两种 gain 的 paired scatter、速度/recovery 分桶和 episode-bootstrap CI；
- Actor 推理之外的 DBM rollout 总预算。

若 V1 不通过，应先修改 Actor reward/objective，使其直接优化可部署 wrapper 质量或其可靠代理，
而不是继续假定 `J_direct` 等价于 sampling-center quality。

### 5.6 最小固定 DBM 闭环 A/B 设计

首轮只做无观测噪声的 diagnostic smoke，不接 Query、不解封 test：

- arms：`warm-only MPPI` 与 `Actor-center MPPI`；可先用 25.518 旧 Actor 验证简单 center
  注入，再单独评价需要 first-pass feedback 的 14.312 Actor；
- scenario：steady tracking 与 recovery；覆盖参考速度 1.2/1.6/2.0/2.4/2.8 m/s；
- pairing：每个 arm 使用相同初始状态、场景、reference、seed 和噪声配置；
- budget：每控制步 DBM rollout 总数严格一致。若 Actor 使用 first-pass，则从 final-pass 预算中
  扣除，或给 baseline 运行同样的两阶段预算结构，方案必须在运行前冻结；
- safety：训练/诊断主 arm 不用 perfect guard 偷看真实 cost；仅保留动作边界、数值异常和超时
  fallback。guard 可作为独立诊断 arm，不能混入 Actor 成绩。

建议先做 10 个 paired episode 的 smoke；系统稳定后扩展到 3 seeds、约 30 个 paired
episodes，再决定是否跑完整圈。主要指标为：

1. episode 累计 realized tracking/control cost 及 paired bootstrap 95% CI；
2. lateral/yaw/vx error 的 P50/P95/max；
3. steering/throttle rate、二阶变化或 jerk、控制 clipping；
4. off-track、数值失败、恢复成功率、恢复时间和赛道进度；
5. 每步 Actor/first-pass/final-MPPI/总控制延迟；
6. 每步 `J_direct` gain 与后续 1/5/20 步 realized gain 的相关性；
7. Actor center movement、final weighted center movement、ESS 和下一步 warm-start 漂移。

### 5.7 验证结果如何决定后续路线

| V1 wrapper | V3 短闭环 | 解释 | 下一动作 |
| --- | --- | --- | --- |
| FAIL | 不必扩大 | `J_direct` 与 sampling-center objective 错配 | 修改 reward/评价包装，不继续堆 Critic 标签 |
| PASS | FAIL | 单步中心有效，但时间递推、抖动、warm-start 或恢复策略有问题 | 转向短时累计/闭环目标与 temporal consistency |
| PASS | PASS | open-loop 收益可以传导 | 继续新 heldout、Critic/Actor 和完整 DBM 闭环 |
| 高速单独 FAIL | 其他 PASS | 主要是高速覆盖/条件化/模型敏感性 | 定向补 untouched 高速/recovery episode 和风险建模 |

### 5.8 对 stay/sign/step 建议的定位

local-gradient step 已证明平均方向有用，但同半径仍有 45% context 退化；因此
stay/sign/step 是合理的**机制分解和保守 tail 诊断**。但它不应直接替代通用 Critic，也不应
被写成环境专用规则后作为最终部署方案，否则更换 rollout model 或 cost 配置时泛化性有限。

优先级建议为：

1. 先通过 V1/V3 判断当前 reward 与闭环目标是否一致；
2. 若一致，继续改进 Critic 的 value/pair-delta/rank/local-gradient/cross-radius 拟合和 tail
   风险估计；
3. stay/sign/step head 作为可解释的辅助监督、风险 gate 或 fallback 诊断，不作为绕过 Critic
   学习问题的主要捷径；
4. 任何保守 head 都必须在新 episode-heldout 上验证，并与不加 head 的 Actor 分 arm 报告。

### 5.9 本次复核后的即时建议

在新的验证证据出来前：

- 暂停继续扩大同类 Critic/Actor 标签和相同 consumed context 上的超参搜索；
- 不启动 Query/ONNX、100 km/h 扩展或 sealed test；
- 先完成 V1 同预算 wrapper 验证和 V2 最小 PyTorch DBM center 注入；
- 随后执行 V3 无噪声短闭环 paired A/B；
- 根据 V1/V3 的四象限结果，决定是改 reward、改 temporal objective，还是继续 Critic 与新
  高速 heldout 数据。

这一路径的目的不是跳过现有 tail gate，而是尽早回答当前最关键、此前没有证据的问题：
**单步学到的采样中心改善，是否真的能改善 MPPI 闭环控制。**

---

## 6. Qoder 对 §5 的二次核对（2026-08-12）

本节是对上面 §5（Codex 复核补充）的再核对，重点是把其中**可验证的事实性论断**
与仓库实际代码/产物逐条对照。不改动 §1--§5 的任何内容，不改动 gate，不解封
formal validation / test。

### 6.1 已核实为真的部分（§5 的加分项）

- **§5.2 R1 的代码论断准确**：`car_ros2/car_ros2/car_node.py` 只有
  `query_checkpoint` 与 `mppi_running_params`，确实没有任何 proposal policy
  checkpoint、网络 center 注入或 proposal fallback runtime 入口。因此把 R1 从
  “项目没做过任何闭环”精修为“**learned sampling center 从未接入控制循环做配对
  A/B**”是更准确的表述——仓库确有 DBM/Query 基线闭环。
- **§5.4 的架构论断准确**：`car_foundation/car_foundation/mppi_proposal_policy.py`
  中 `feedback_dim=74`（约 L515）、`feedback_dim=74 / gradient_context_dim=32`
  （约 L774--775）均存在。所以“最新 16-D Actor 输入含 `feedback[74]` +
  `gradient_context[32]`、闭环评价必须先冻结 first-pass 生成方式与预算”成立。
- **§5.4 抓到了本文 §3 P5 的一个真实矛盾**：原 P5 把“最小闭环 A/B”排在
  `mppi_proposal_runtime.py` 之前，但没有 runtime 适配器就无法注入 center。§5.4 用
  “先做诊断性 PyTorch + fixed-DBM 适配器”补上了这个缺口，方向正确。
- **§5.2 对 R6 的修订成立**：2026-08-07 确实新增了 270 个 train-only episode /
  1,350 snapshot（handoff 与 plan 文档均有记录），因此“始终只用旧 1,800 状态”应
  降级为“新增数据仍属 train/selection、缺 untouched validation”。
- **§5.7 的四象限决策表**是本文 §3 没有的、可执行的分流逻辑，值得保留。

### 6.2 需要修正的方法学问题：§5.3 的代理验证缺可复算产物

这是本次核对中唯一明确不够严谨的地方。§5.3 “目标传导代理验证”表中：

- `mean direct gain` `+0.9156 / +0.8633` 与执行台账的 `9.782→8.866 / 8.919`
  （即 `+0.916 / +0.863`）吻合，可信；
- 但 `wrapper_gain`（`+0.1331 / +0.0124`）、`episode_063` 的 `-0.1409 / -0.1903`、
  以及 **Pearson `-0.499 / -0.972`**，在全仓库（脚本 / json / sidecar）中**搜不到
  任何来源**，也没有对应的 `analyze_*fixed5*` 脚本。

这些 wrapper 数字需要重新运行 `fixed-hadamard-64` wrapper 才能得到，却没有落成
可复算产物。这与本仓库（及本文 R3/R7）强调的纪律相冲突——每个 sidecar 都应有
independent validator + SHA256。

**建议二选一**：

1. 补一个提交的分析脚本（复用 `continuous_center_direct_fixed5_pilot_20260806_v1/
   v2/v3` checkpoint，运行 `fixed-hadamard-64-v1` wrapper）并输出带 hash 的 json；
2. 或把 §5.3 明确标注为 `illustrative / 未落产物 / n=5`，仅用于触发 V1，不作为
   证据引用。

结论方向（`J_direct` 不等价于 sampling-center quality，需 V1 验证目标传导）本身合理，
§5.3 也已诚实声明“不能用于正式统计声明”；只是证据链要么补齐要么降级。

### 6.3 其他小点

- §5.5 的 V0--V9 顺序合理，但 V3（短闭环）与 V6（完整闭环 A/B）存在重叠，
  可把 V6 注明为“V3 通过后的扩展轮”，而非并列新阶段。
- §5.5/§5.6 的门槛较多，落地时建议先只做 V0--V3 的诊断闭环，拿到四象限结论后再决定
  是否展开 V4 之后。

### 6.4 总体判断

§5 总体合理，且在若干点上比本文 §1--§3 更准确、更可执行：对 R1 的精修、抓出
runtime 适配器的工程顺序矛盾、以及四象限分流决策表。唯一实质缺陷是 §5.3 引入了
无法复算的 wrapper / 相关系数数字，应补脚本或降级为 illustrative。

---

## 7. 对 §6.2 的更正（2026-08-12，第三轮复核）

### 7.1 §6.2 的核心事实判断错误，予以撤回

§6.2 断言 §5.3 的 wrapper 数字“在全仓库搜不到任何来源”“需要重新运行
`fixed-hadamard-64` wrapper 才能得到”。**该判断错误，现予撤回。**

错误原因：§6.2 的检索只搜了**派生值**（`0.9156` / `4.5280` / `-0.972`），
没有搜**原始 wrapper 值**（`7.655712` 等），因此漏掉了已存在的底层产物。

wrapper 原始结果确实存在，保存在三个 pilot 的 `summary.json` 的
`fixed_neighborhood_weighted_output_cost` 字段中（顶层聚合值）：

| 实验 | 顶层 wrapper cost | summary.json 行号（实测） |
| --- | ---: | ---: |
| v1 | 7.6557124853 | `:36` |
| v2 | 7.5226566553 | `:47` |
| v3 | 7.6433490992 | `:47` |

（注：复核意见给出的行号为 `:26/:37/:37`，实测为 `:36/:47/:47`，属笔误，不影响结论。）

生成侧也确实同时计算并保存了 weighted-output、candidate cost 与 fixed-bank
contract hash，见 `scripts/model_verify/fine_tune_mppi_direct_center_sac_pilot.py:504`
的 `fixed_neighborhood_evaluate(environment, final_centers[context], radii)`。

### 7.2 独立复算结果：§5.3 全部数字精确复现

直接从三个 `summary.json` 读取并复算（未重跑任何 DBM rollout），三者 episode 集合
完全一致（`episode_000/021/042/063/084`）：

| 实验 | mean direct gain | direct wins | mean wrapper gain | wrapper wins | Pearson |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2 | +0.9156 | 3/5 | +0.133056 | 3/5 | −0.499 |
| v3 | +0.8633 | 3/5 | +0.012363 | 4/5 | −0.972 |

逐帧（direct gain / wrapper gain）：

| episode | v2 direct | v2 wrapper | v3 direct | v3 wrapper |
| --- | ---: | ---: | ---: | ---: |
| 000 | +0.0604 | +0.181156 | −0.1035 | +0.054975 |
| 021 | −0.2013 | −0.061506 | +0.2609 | +0.069467 |
| 042 | +0.5914 | +0.487527 | +0.2070 | +0.089326 |
| **063** | **+4.5280** | **−0.140876** | **+3.9827** | **−0.190306** |
| 084 | −0.4004 | +0.198977 | −0.0308 | +0.038355 |

与 §5.3 报告的数值逐项一致，包括 `episode_063` 的 `-0.140876 / -0.190306`。
因此 **§5.3 的反例与结论方向是有实际产物支撑的**，不是凭空数字。

### 7.3 修正后仍然成立的部分：证据链需固化

§6.2 的**结论方向**（证据链不够完整）仍然成立，但理由应改为：

- 派生公式目前只写在文档里，没有代码实现；
- Pearson / 逐帧 gain 没有独立 JSON 产物；
- 三个输入 `summary.json` 与生成脚本目前仍是 untracked；
- 没有专门校验 episode 对齐、候选数 / radii / contract hash 一致性的分析工具。

### 7.4 修正后的建议：只补纯分析脚本，不重跑算力

不需要重新花算力跑旧 5 帧。建议补一个**纯派生分析脚本**：

1. 读取 v1/v2/v3 的 `summary.json`；
2. 校验三者 episode 集合完全一致；
3. 校验候选数、`radii` 与 fixed-bank contract hash 一致；
4. 计算逐帧 direct / wrapper gain、均值、胜率与 Pearson；
5. 输出含三个输入文件 SHA256 的 JSON manifest；
6. 保留 `MECHANISM_PILOT_ONLY` / `n=5` 标记。

#### 7.4.1 已落地（2026-08-13）

脚本：`scripts/model_verify/analyze_mppi_direct_fixed5_target_transfer.py`
（纯 stdlib，不加载 checkpoint、不跑 DBM rollout、`recomputes_dynamics: false`）
产物：`outputs/mppi_proposal/direct_fixed5_target_transfer_20260812_v1/analysis.json`

复算输出与 §7.2 逐项一致：

```
episodes: 5 ['episode_000','episode_021','episode_042','episode_063','episode_084']
contract: candidate_count=64, version=fixed-hadamard-64-v1,
          contract_hash=0fd5402..., radii_source_sigma=[0.1,0.3]
v2: direct +0.9156 (3/5)  wrapper +0.133056 (3/5)  pearson -0.499  contradictions ['episode_063']
v3: direct +0.8633 (3/5)  wrapper +0.012363 (4/5)  pearson -0.972  contradictions ['episode_063']
```

输入 SHA256（前 16 位）：v1 `9676c29bbd99abff` / v2 `d12ef25878bd7423` /
v3 `1e21708e69363626`。

脚本内置的契约守卫（均已负向测试触发）：

- episode 集合三者必须完全一致，且单个 run 内不得重复；
- `candidate_count` / `version` / `contract_hash` / `radii_source_sigma` 必须一致；
- 所有 run 的 `qualification` 必须为 `MECHANISM_PILOT_ONLY`；
- **wrapper 基线 run 必须 `best_iteration == 0`**。因为每行只存一个「该 run 最终中心」
  的 wrapper cost，只有 iteration 0 的最终中心才等于初始中心，才能充当
  `initial wrapper` 参考。用 v2（`best_iteration=5`）当基线会直接报错；
- v1 字段名为 `sac_actor_direct_cost`、v2/v3 为 `deterministic_actor_direct_cost`，
  脚本按优先级自动识别并在 manifest 中记录实际使用的字段。

JSON 中同时固化了三条 caveat，其中最重要的一条是：
**`direct_gain` 是 run 内（初始 vs 最终中心），`wrapper_gain` 是跨 run（vs 基线 run），
两者不共享参考系**，因此其相关系数只能作为方向性证据，不能作为统计结论。

#### 7.4.2 第四轮复核的三点收敛（2026-08-13）

独立重放确认：`/tmp` 重跑与现有 `analysis.json` 字节级一致；错用 v2 当 baseline
被 `best_iteration=5` 正确拒绝；未加载 checkpoint、未运行 DBM、未触碰 sealed split。
在此基础上采纳三点改进。

**（1）符号不一致必须分两类表述。** 此前口头总结“矛盾帧都是 `episode_063`”不精确。
准确表述为：**`direct_gain > 0` 但 `wrapper_gain < 0` 的矛盾帧都是 `episode_063`**。
若按全部符号不一致统计，还包括“direct 退化但 wrapper 改善”的相反情形：

| 实验 | direct+ / wrapper−（目标传导矛盾） | direct− / wrapper+（相反情形） | sign agreement |
| --- | --- | --- | ---: |
| v2 | `episode_063` | `episode_084` | 3/5 |
| v3 | `episode_063` | `episode_000`, `episode_084` | 2/5 |

脚本已拆成 `direct_improved_but_wrapper_regressed` 与
`wrapper_improved_but_direct_regressed` 两个字段，并加 `sign_mismatch_episodes` 汇总；
caveat 中明确禁止把两类合并。**只有前者才是目标传导矛盾。**

**（2）baseline 增加逐帧数值断言。** `best_iteration == 0` 只表达生成脚本的意图，
若将来 summary 语义变化会静默失效。现追加断言：baseline 每帧
`|initial_direct − final_direct| ≤ tol`（默认 `1e-6`，可用 `--baseline-tolerance` 调整），
并在 manifest 输出 `baseline_initial_final_max_abs_error`。**实测为 `0.0`。**

同时明确记录数学局限：direct cost 相等**不能**证明 center 完全相同；严格证明需要
summary 中保存 initial/final center knots 或其 hash。manifest 中以
`baseline_center_identity_proven: false` 显式标注，并新增对应 caveat。

**（3）manifest 记录分析脚本自身 SHA256。** 已新增 `analysis_script` 与
`analysis_script_sha256` 字段。注意本轮改动后脚本 hash 已变化：
复核时引用的 `2796b347...` 为改动前版本，改动后为
`4db88d80a8262bf989c12c8d2e08221cb1b726a8332b78dd65fe1bed0d65e3c4`。
核心数值（两个 Pearson、均值、胜率）在改动前后完全不变。

**（4）证据链的 Git 固化方式（已决定）。** 现状经 `git check-ignore` 确认：

- `scripts/model_verify/analyze_mppi_direct_fixed5_target_transfer.py` — 原 untracked；
- `car_foundation/docs/mppi_sampling_center_review_20260812.md` — 原 untracked；
- `analysis.json` 被 `outputs/.gitignore:1` 的 `*` 规则忽略。

**决定：只提交分析脚本与本 review 文档，`analysis.json` 不入库。**
理由是该 manifest 完全由脚本从三个 frozen summary 派生，已确认字节级可复现
（`/tmp` 重跑与现有文件一致），因此入库属冗余；同时避免为 `outputs/` 开
force-add 例外或新增 `evidence/` 目录。

代价与补偿：SHA256 台账不随 commit 留存。补偿方式是本文 §7.4.1 / §7.4.2 已把
三个输入 summary 的 SHA256 前缀、分析脚本 hash 与全部核心数值写入文档正文，
任何时候都可用下列命令重新生成 manifest 并比对：

```bash
python3 scripts/model_verify/analyze_mppi_direct_fixed5_target_transfer.py
```

### 7.5 其他被接受的更正

- **Actor 输入维度的引用位置应修正**：§6.1 引用的约 L515 / L774--775 属于其他
  Critic/Actor 类。连续中心路线应引用
  `car_foundation/car_foundation/mppi_proposal_policy.py:963` 的
  `TorchMPPIContinuousCenterEncoder`（`feedback_dim=74`、`gradient_context_dim=32`、
  `output_dim=192`）。已实测确认该类位于 L963。
- **V3 与 V6 的关系应明确写死**：V3 是短闭环 smoke / 诊断，V6 是 V3 通过后的
  正式扩展轮，二者不是两种独立方法。
- **5 帧代理的定位不变**：它不能替代 V1，更不能替代闭环；唯一作用是证明
  “不能默认认为 `J_direct` 会传导到 MPPI wrapper”。

### 7.6 本轮结论

接受“证据固化”要求，撤回“wrapper 来源不存在、必须重新运行”的判断。
下一步只需补派生分析脚本与 manifest，然后直接进入更有价值的
**V1 全量同预算 wrapper 验证**。
