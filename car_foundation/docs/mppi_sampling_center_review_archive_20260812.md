# MPPI 采样中心策略训练评审：归档（历史截至 2026-08-18）

> 本文件保存原 `mppi_sampling_center_review_20260812.md` 的 preamble、§1–§11.37，
> 以及主文档中被 tombstone 化的四个早停 bug 历史 block（附录 A）。
> 当前主文档从 §11.38 起继续；最新权威结论见主文档 §11.80。
> 本文件仅作审计与复现参考，不再承载活跃计划。
>
> ---
>
> **原始文档头部**（保留供引用）：

> # MPPI 采样中心策略训练评审：结论 / 风险 / 推荐计划（2026-08-12）

本文是对 `car_foundation/docs/` 下 MPPI 采样中心系列文档的一次外部评审记录，
基于以下来源整理：

- `mppi_sampling_design_and_current_plan_20260807.md`（汇总视图）
- `mppi_sampling_center_handoff_20260802.md`（主交接，历史与失败记录）
- `mppi_direct_actor_trpo_like_design_20260807.md`（FR-TRPI 详细设计）
- `mppi_sequential_probe_execution_plan_20260806.md`（权威执行入口）

本文只记录评审结论、风险与建议，**不改动任何契约、gate 或数据**。截至 2026-08-12 的
历史执行入口仍是 `mppi_sequential_probe_execution_plan_20260806.md`；2026-08-17 之后的路线
覆盖以本文 §11.38 为准。

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

### 1.3 2026-08-17 最新覆盖结论：Critic-gradient 主路线降级

§11.35--§11.37 已把 Critic 侧剩余的工程、监督目标和输入假设按统一 grouped-CV 口径跑完：
修正批量错行后的真标量 Q/value-delta v2 共完成两臂 30 个 run，各臂 `0/15` 通过（逐 seed pooled value
corr `-0.03~-0.12`、derived-gradient cosine median `-0.62~-0.70`）；E0/E1/E2 显式输入
三臂也没有恢复 G0_ONLY（`-0.829/-0.739/-0.835`）。tiny overfit、seen-fit、坐标、autograd、
标签 replay 与 checkpoint 口径均已通过，说明当前阻塞不是旧 encoder 简单丢失 `vy/yawrate`
或 reference 信息，而是 g0-hard 状态在现有覆盖下的跨状态映射不可迁移。

因此本文后续最新路线以 §11.38 为准：不再把“学习可靠 action-gradient Critic”作为主线，
改为**带当前 center 条件的近端离线搜索与迭代蒸馏**。这不是重新做一次全局 J16 MSE：历史
T1/J16 已证明高质量 teacher 不自动带来 Actor 泛化，新的验证必须先检查最优 residual 的邻域
coherence，再按 episode-grouped cross-fit 拆开 teacher、蒸馏和泛化损失。当前部署 Actor、
formal validation 与 test 继续冻结。

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

---

## 8. 目标合法性与 two-center 兜底（2026-08-13）

本节修正 §5.3 / §7 遗留的一个定位偏差：把 5 帧 fixed-wrapper 结果看得过重，
以至于把“reward 是否选错”当成待判问题。同时收紧两处过度断言。

### 8.1 已确认：`J_direct` 是有意义的主要训练目标

采样中心始终作为**未加噪候选**参与评估，且候选由中心加噪构造：

- `car_dynamics/car_dynamics/controllers_torch/mppi.py:193`
  `noise[0].zero_()  # Always retain the current mean as a candidate.`
- 同文件 `:327` `raw_sampled_knots = sampling_mean_knots[None, :, :] + noise`

配合权重高度集中的证据（基准快照 ESS `1.44/256`、best 权重 `0.811`、
best+mean 占 `99.95%`；T0 数据平均 ESS 约 `1.543`），可以认为
**`J_direct` 比窄 fixed-wrapper 更接近实际控制价值**。

因此结论修正为：**不能因为旧 5 帧 wrapper 改善较小（`+0.133 / +0.012`），
就判定 reward 明显选错。** `J_direct` 保留为主要训练目标。

旧 wrapper 的**单位**也需澄清：`radii_source_sigma = [0.1, 0.3]` 是**相对
`noise_sigma` 的半径倍数**，不是另一组绝对 sigma
（见 `scripts/model_verify/fine_tune_mppi_direct_center_sac_pilot.py:101`
的 `--fixed-bank-radii default=(0.10, 0.30)`，写入字段名即 `radii_source_sigma`）。
实际固定偏移只有约 `0.1σ / 0.3σ`，而部署是 256 条完整 Gaussian `1σ` 采样。
所以它是一个**很窄的结构化局部探针**，不适合承担“部署目标是否正确”的正式 gate，
降级为便宜旁证是恰当的。

### 8.2 收紧一：低 ESS 不能推出“中心必得约 0.8 权重”

MPPI 权重为 `w_i = exp(-J_i/λ) / Σ_j exp(-J_j/λ)`。
`0.811` 是**某个基准快照中最优候选**的权重，**不是中心候选固定拥有的权重**。
Actor center 变好后，可能成为 best 并获得很大权重，也可能仍略差于某条噪声候选、
与多条候选 cost 接近而使 ESS 增大、或在 clipping 后与其他候选部分重合。

准确表述应为：

> 低 ESS 表明 `J_direct` 很可能比窄 fixed-wrapper 结果更接近实际控制价值，
> 但**传导强度仍需用部署 Gaussian 采样实测**。

“稀释效应通常可能较小”是合理假设；“稀释一定只是二阶影响”**目前没有证据**，
不得作为结论写入。

### 8.3 收紧二：追加 warm 到 softmax 候选集不能严格兜底

最终输出不是选最低 cost 候选，而是**加权动作序列**
（`car_dynamics/car_dynamics/controllers_torch/mppi.py:344`
`weighted_sequence = torch.sum(...)`）：`ū = Σ_i w_i u_i`。

因此即使 warm 是候选之一，也**不能保证** `J(ū) ≤ J(u_warm)`——车辆 rollout 与
轨迹 cost 对动作序列不是可依赖的凸函数，Actor 邻域候选与 warm 的加权混合仍可能
比单独执行 warm 更差。必须区分两个方案：

| 方案 | 构造 | 性质 | 额外 rollout |
| --- | --- | --- | --- |
| **Soft two-center candidate** | 候选集含 Actor center + warm center + 其余 Actor 邻域采样 | **强缓解，非保证**。灾难性 Actor center 下 warm 大概率因 cost 明显更低而获得接近 1 的权重 | 0（占用一条候选额度） |
| **Hard model-cost fallback** | `u_exec = argmin_{u ∈ {u_warm, u_weighted}} Ĵ(u)` | 在**当前 rollout model 下**保证 `Ĵ(u_exec) ≤ Ĵ(u_warm)` | +1（需 rollout weighted sequence） |

hard fallback 的保证范围必须写清：固定 DBM 仿真阶段 DBM 同时是环境与 rollout
model，可视作准确；**Query / 真实车辆阶段只保证 Query model cost，不保证真实车辆 cost**。

另需澄清既有 guard 的性质：`scripts/model_verify/evaluate_mppi_direct_actor_guard.py`
中 `guard_cost = np.minimum(initial_cost, actor_cost)` 是对**两个中心 direct cost**
取 hard min，既不是“把两个中心塞进 softmax 候选集”，也不完全等同于上表的
hard fallback（后者比较的是 warm 与 **weighted output sequence**）。三者不可混称。

### 8.4 预算修正：默认是 256 而非 64

`mppi.py:94` 为 `num_samples: int = 256`。因此：

- 直接增加一个 warm candidate：额外约 **1/256** 预算；
- 严格保持 256 总预算：**Actor + warm + 254 条噪声**，替换掉一条随机候选；
- `1/64` 只适用于旧 fixed-wrapper 的 64 candidate 配置。

此前口头表述的“1/64 预算”属错误，予以更正。

### 8.5 接入风险的定位

“Actor center 替换 running-state mean 会使 warm 从候选集消失”
（`warm + 255 条 warm 邻域噪声` → `Actor center + 255 条 Actor 邻域噪声`）
是**朴素接入方案会产生的风险**，不是现有在线代码已发生的 bug——learned Actor
目前尚未接入 runtime。表述必须保持这个区分。

### 8.6 修正后的优先级

two-center 提到第一位，但按如下方式执行。

**Step 1**：固定 DBM frozen-state 上比较四组：

1. warm-centered Gaussian（基线）；
2. Actor 替换 warm；
3. Actor + warm **soft** two-center；
4. Actor-centered MPPI + **hard** model-cost fallback。

**Step 2**：四组使用**完全相同的 256 总候选预算**；hard fallback 的额外一次
rollout 单独标记，不混入预算对比。

**Step 3**：同时报告 weighted-output cost、best/P10/median、P05/worst、
**warm 候选权重**、**warm 成为 best 的比例**、**hard fallback 触发率**，
以及 2.4/2.8 m/s 与 recovery 分组。

**Step 4**：若 soft two-center 已消除绝大多数尾部，优先用它做闭环 smoke；
若仍有负尾部，改用 hard fallback。

**Step 5**：采集新的 untouched 高速 / recovery episode，冻结新 gate。

**Step 6**：短闭环 A/B，重点验证时间递推、warm 漂移、控制抖动与 fallback 频率。

### 8.7 V1 的重新定位

V1 **不再承担“判断 reward 是否错误”的角色，但也不降为可选项**——
改造为上述 **部署 Gaussian（256 候选）two-center 接入验证**，即 Step 1--3。
这样它同时回答传导强度（§8.2 要求的实测）与尾部兜底（§8.3 的 soft/hard 对比）。

### 8.8 本节结论

> `J_direct` 仍是有意义的主要训练目标，旧 5 帧 wrapper 不应作为“reward 选错”的 gate；
> two-center 是下一步最便宜且最有希望的结构改进。但**单纯把 warm 追加进 softmax
> 候选集只能缓解尾部，不能从数学上保证 warm 下界**；要获得严格的模型空间兜底，
> 需要 hard model-cost fallback，且其保证仅限于当前 rollout model。

## 9. 部署 Gaussian two-center 固定状态验证（2026-08-13）

§8.6 的 Step 1--3 已执行。实现与产物为：

```text
scripts/model_verify/evaluate_mppi_two_center_integration.py
scripts/model_verify/validate_mppi_two_center_integration.py
outputs/mppi_proposal/two_center_integration_20260813_v2/summary.json
outputs/mppi_proposal/two_center_integration_20260813_v2/evaluation.npz
outputs/mppi_proposal/two_center_integration_20260813_v2/validation.json
```

### 9.1 冻结口径

- rollout/environment model：固定 small-car DBM，几何参数 `lf=0.1008 m`、
  `lr=0.1092 m`；
- Actor：`direct_residual_online_ac_20260811_v2` 的 selected checkpoint；
- 数据：已经被历史 checkpoint selection 消费过的 `internal_selection`，60 个 episode、
  600 个 context；**仅作机制诊断，不产生新泛化结论**；
- Monte Carlo：3 个冻结 seed（3407/3408/3409），共 1800 个 context-seed 配对样本；
- MPPI：Gaussian、`noise_sigma=[0.25,0.35]`、256 候选、8×2 knots、horizon 50、
  temperature 1、单轮更新；
- 三个采样臂使用 common random numbers。soft two-center 固定为 candidate 0 = Actor、
  candidate 1 = exact warm、其余 254 条为 Actor 邻域 Gaussian，总候选数仍为 256；
- formal validation/test episode 均未生成、未读取、未评估。

独立 validator 检查了 `[3,600,256]` 三臂候选张量、candidate 0/1 的 direct cost、
softmax 权重、hard min 重建，并从 24 个均匀 context × 3 seeds 独立重放三组
weighted output。candidate cost、weighted-output cost 重放最大绝对误差均为 0，验证为
`PASS`。

### 9.2 主要结果

以下 gain 均定义为：

```text
gain = J(warm-centered Gaussian weighted output) - J(method output)
```

正数表示相对现有 warm-centered 256-candidate MPPI 改善。

| 方法 | rollout 预算 | mean cost | mean gain | P05 gain | worst gain | 回归比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| warm-centered Gaussian baseline | 256 | 18.188 | 0 | 0 | 0 | 0 |
| Actor replace | 256 | **9.057** | +9.131 | -0.110 | -42.282 | 5.17% |
| Actor + warm soft two-center | 256 | **9.052** | +9.136 | -0.059 | **-90.878** | 5.06% |
| soft output vs direct warm hard fallback | 256+1 | **8.908** | +9.281 | +0.021 | -23.076 | 4.94% |
| warm MPPI vs Actor direct hard guard（Actor center 已给定） | **256+2=258** | 10.670 | +7.519 | **0** | **0** | **0** |
| warm-MPPI vs Actor-MPPI paired hard min（辅助上界） | 512+输出评估 | **8.765** | +9.423 | 0 | 0 | 0 |

这组结果同时确认两件事：

1. **`J_direct` 的平均改善可以传导。** Actor replace 把 weighted-output mean cost 从
   18.188 降到 9.057，不能再说 Actor center 的 direct 改善基本被 MPPI 稀释掉。
2. **平均传导不等于尾部安全。** Actor replace 仍有 5.17% 的负配对样本；仅把 warm
   塞入 Actor 邻域 softmax 并未解决尾部，worst 反而从 -42.282 恶化到 -90.878。

高速和 recovery 分组保持同一结论：

| 分组 | 方法 | mean gain | P05 gain | worst gain |
| --- | --- | ---: | ---: | ---: |
| 2.4 m/s | Actor replace | +10.357 | -2.752 | -42.282 |
| 2.4 m/s | soft two-center | +10.311 | -1.836 | -42.282 |
| 2.4 m/s | 258 hard guard | +7.414 | **0** | **0** |
| 2.8 m/s | Actor replace | +9.807 | -4.572 | -31.327 |
| 2.8 m/s | soft two-center | +9.869 | -2.159 | **-90.878** |
| 2.8 m/s | 258 hard guard | +6.131 | **0** | **0** |
| all recovery | Actor replace | +9.325 | +0.052 | -31.327 |
| all recovery | soft two-center | +9.313 | +0.057 | **-90.878** |
| all recovery | 258 hard guard | +7.544 | **0** | **0** |

### 9.3 soft two-center 的失败机制已被实测捕获

最坏样本是 `episode_379/step_000250/context_1`、2.8 m/s dynamic recovery、
seed 3408：

- warm-centered MPPI weighted output cost：`28.111`；
- Actor-replace weighted output cost：`33.507`；
- soft two-center weighted output cost：`118.989`；
- soft bank 中最低几条候选的 `(cost, weight)` 为
  `(29.601,0.671)`、warm `(31.186,0.138)`、`(31.418,0.109)`、
  `(31.797,0.075)`。

也就是说，每条高权重候选自身 cost 都在 29--32 附近，但动作序列的 softmax 加权结果
cost 达到 118.989。该样本直接证明 §8.3 的非凸性 caveat 不是纯理论担忧：
**多个低 cost 动作序列的加权动作不一定仍是低 cost 动作序列。** 因而 soft
two-center 不能作为尾部兜底方案。

soft bank 中 warm candidate 的平均权重只有 `0.0266`，中位数约 `1.19e-6`，成为 best
的比例仅 `1.67%`；Actor candidate 成为 best 的比例为 `70.89%`。这也否定了
“加一个 warm 后，危险状态中它自然会获得接近 1 的权重”这一未经验证的假设。

### 9.4 两种 hard 下界不能混淆

`soft weighted output` 与 `direct warm` 取 hard min 的 257-rollout 方案确实满足：

```text
J(exec) <= J(direct warm)
```

validator 的该下界最大 violation 为 0。但它**不保证**不差于原 warm-centered MPPI
weighted output，因为后者会利用 255 条 warm 邻域噪声寻找更好动作。因此该方案相对
部署基线仍有 worst gain `-23.076`，不能作为“保持现有控制器性能”的严格 fallback。

若 **Actor center 已经给定**，要保持原控制器的模型空间下界，一个更直接且便宜的
选择结构是：

```text
原 warm-centered Gaussian 256 条照常计算
    ├─ 得到原 weighted output；额外 rollout 一次求其 J       (+1)
    └─ Actor 只生成唯一 direct sequence；rollout 求其 J       (+1)

exec = argmin_J(original warm-MPPI output, Actor direct sequence)
```

选择阶段预算为 **258 rollouts**，没有 action blending，也不需要第二套 Actor-centered
256 候选。固定 DBM 诊断中它选择 Actor direct 的比例为 `80.72%`，mean gain `+7.519`，
同时 P05/worst gain 均为 0。它比纯 Actor-replace 的 mean cost 高约 `1.61`，这是用少量
平均收益换取构造性尾部下界；相较 512-candidate paired controller，预算明显更低。

但当前 residual Actor 的完整输入不是只含 state/history/reference。它还依赖：

- 以冻结 BC center 为中心的 128 条 antithetic first-pass rollout；
- 由这些 rollout 拟合的 74 维 trajectory/cost response feedback 与 guided anchor；
- 三个冻结 feedback Critic 输出的 16 维 gradient mean/std context；
- old/proposal Actor、trust projection 和 Alpha policy 组成的 base center。

固定状态实验直接读取了已保存的 first-pass context，所以表中的 258 **没有计入在线生成
Actor center 的成本**。若严格复现训练时的单 seed context，当前完整预算是：

```text
128 first-pass candidates + 1 first-pass weighted-output evaluation + 256 warm MPPI
+ 1 warm weighted-output evaluation + 1 Actor direct evaluation
= 387 model rollouts / control step
```

训练时每 snapshot 的两个 first-pass seed 是两个独立 context；部署只需冻结其中一个，
不需要同时计算两组 128。只有后续证明能从原 warm-centered 256 bank 复用并生成与训练
分布兼容的 Actor context，才可把完整预算降回 258。此前把 258 直接写成完整部署预算是
遗漏，现予以修正。

### 9.5 修正后的结论与下一步

本轮结果不支持把 soft two-center 直接送入闭环。当前最合理的最小闭环对象改为
**baseline-preserving hard guard**，先按严格复现输入的 387-rollout 版本验证：

1. 先从 raw snapshot 在线重建单 seed first-pass feedback、gradient context 和最终 Actor
   center，验证它与缓存 context/Actor 输出一致；未通过前不得启动闭环；
2. 在 DBM runtime 中实现 hard guard，保持原 warm MPPI 的 sampling、running-state 更新
   和 seed 完全不变；Actor direct 只参与末端 hard 选择，不参与 softmax 混合；当前完整
   预算记为 387，而不是 258；
3. 做 matched-seed 短闭环 A/B：原 warm MPPI vs guarded Actor，重点报告累计 cost、失败率、
   横向/航向误差、控制 rate/抖动、Actor 选择率和选择连续性；
4. **实时性必须作为一等 gate，而非事后备注。** 现有 256 warm MPPI 已是
   `57--60 ms/step vs 50 ms 预算`（§9 风险表），而完整 guard 是 387 rollouts/step。A/B
   必须按诚实的 387-rollout 成本运行（不得用缓存 context 的 258 掩盖延迟），并报告每步
   wall-clock 延迟与超时率。若延迟不达标，即便 cost gate 通过也不算通过——此时唯一出路是
   证明 context 可从 warm 256 bank 复用降回 258，或用 Query 代理 first-pass/guard rollout；
5. 补一张 **fallback 选择分支 × 速度桶** 交叉表：确认 guard 落回 warm 的 19.28% 子集正好
   集中在高速/recovery 尾部 context，而不是随机回退——否则“构造性兜底”只是平均意义的，
   不能保证在真正危险状态触发；
6. 若内部闭环 smoke 能保住基线并传导 open-loop 收益，再冻结协议，采集新的 untouched
   高速/recovery episode 做一次性 gate；
7. Query/真实车阶段必须另行验证 model-cost pairwise ranking。DBM 中的零回归保证不能
   自动外推到模型失配场景。

因此，§8.6 的 Step 4 被本节结果具体化为：**拒绝 soft two-center 作为 primary；优先
验证不混合动作、保留原 warm-MPPI 输出下界的 hard guard；先补齐 Actor 在线输入重建，
再进入最小闭环 A/B。**

### 9.6 raw snapshot 在线输入重建资格检查（2026-08-13）

§9.5 Step 1 已完成。新增脚本
`scripts/model_verify/validate_mppi_residual_actor_runtime_inputs.py`，产物位于
`outputs/mppi_proposal/residual_actor_runtime_input_validation_20260813_v2`。检查只使用已经
消费的 `internal_selection`，没有加载正式 validation/test；覆盖 300 个 raw DBM snapshot、
两个冻结 first-pass seed，共 600 个 context。

检查不是把缓存 context 再喂给 Actor，而是从 raw snapshot 的 state/history/reference/
current action 重新执行完整链路：

```text
raw snapshot
  -> frozen BC center
  -> 128 条 antithetic DBM first pass
  -> guided anchor + 74-D feedback
  -> 3 个 frozen feedback Critics 的 gradient mean/std
  -> old/proposal Actor + trust projection + Alpha base
  -> residual Actor center
```

结果为 `PASS_RUNTIME_INPUT_RECONSTRUCTION`：BC center、first-pass knots/cost、guided
center、feedback、gradient mean/std、old/proposal center 的最大绝对误差全部为 0；最终
residual center 的最大绝对误差为 `9.09e-7`，低于 `5e-5` gate。它证明当前 Actor 输入可从
在线可观测状态和 DBM rollout 确定性重建，不依赖训练缓存或 oracle 标签。

同时正式固化预算：部署每个控制周期只冻结一个 first-pass seed，因此是
`128 + 1 + 256 + 1 + 1 = 387` 次 DBM rollout。其中 first-pass 的额外 1 次来自 74 维
feedback 内 `first_weighted_output_cost` 的真实 DBM 复评；训练文件中的第二个 seed 是
另一个训练 context，
不是在线必算项。下一步不再阻塞于输入合法性，转为把该链路封装为共享 runtime 组件，接入
baseline-preserving hard guard，然后做 matched-seed 短闭环。闭环前不得把 387 偷换为
258；只有 warm-bank context reuse 另行通过分布一致性验证后才能降预算。早先 v1 summary
少计了这一次 weighted-output rollout，已由 v2 替代，数值重建结论不变。

### 9.7 shared runtime、hard guard 与首个短闭环 A/B（2026-08-13）

§9.5 Step 2--3 的最小机制 pilot 已完成，但结论是“**部分传导、平滑性与实时性不通过**”，
不是闭环准入 PASS。

实现与固定状态资格：

- `car_foundation/mppi_residual_actor_runtime.py` 封装完整 129-rollout Actor context 与唯一
  center 生成，不再依赖 research script 或缓存标签；
- `TorchMPPIController.evaluate_action_sequences()` 和
  `hard_guard_action_sequence()` 只做 deterministic forward cost 与 hard argmin，不消耗
  MPPI RNG、不修改 running-state；
- ROS 参数 `mppi_hard_guard_checkpoint` 默认空，原控制器默认行为不变；启用时强制
  `dbm + 256 Gaussian`，保留原 warm sampling/seed/running-state，只替换最终执行分支；
- `residual_actor_runtime_component_validation_20260813_v2` 在 300 个 consumed
  internal-selection 状态、固定 seed 24001 上通过。最终 center 最大误差 `8.34e-7`；74-D
  feedback 最大误差 `7.82e-5`，低于既有 rollout/feedback `5e-4` gate；
- `hard_guard_runtime_validation_20260813_v1` 在 60 个状态上复算，warm/proposal cost、
  selected sequence、RNG、running-state 与 warm-floor violation 全部为 0。

闭环数据与分析：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/
  mppi_hard_guard_closed_loop_pilot_20260813_v1/
    baseline_seed3407/
    guard_seed3407/

outputs/mppi_proposal/hard_guard_closed_loop_pilot_20260813_v1/
  summary.json
  validation.json
```

两臂均为同一 small-car DBM、相同初态、2.8 m/s、MPPI seed 3407、256 Gaussian，trace
覆盖 step 0--300；step 250/260/.../300 各保存 6 个快照，两条 episode 均通过 closed-loop
dataset validator。分析器把 realized one-step diagnostic 明确写成 Frenet tracking/rate
cost，不把 proposal horizon cost 冒充闭环累计 cost。独立 validator 对四组累计指标、每步
hard-min 分支和 warm floor 复算均为 0 误差。

| 指标 | 全 301 步 guard 相对 baseline | history 完整 step 250--300 |
|---|---:|---:|
| realized stage cost sum | **-4.12%** | **-2.29%** |
| lateral RMSE | -0.94% | **+8.87%（退化）** |
| heading RMSE | -2.87% | **-8.80%** |
| speed RMSE | -1.21% | **-4.69%** |
| acceleration rate RMS | +9.54% | **+45.52%（退化）** |
| steering rate RMS | -0.47% | **+16.39%（退化）** |
| acceleration 二阶差分 RMS | +31.41% | **+87.35%（退化）** |
| steering 二阶差分 RMS | +2.47% | **+28.18%（退化）** |

成熟 51 步中，Actor 选择率 `39.22%`，分支切换 `17/50 = 34%`，最长 Actor/warm 连续
段分别为 5/12 步。guard 当前轨迹上的 warm/selected mean model cost 为
`32.918/30.354`，平均即时 gain `+2.564`，warm-floor 最大 violation 为 0。这证明固定状态
open-loop 改善**能够部分传到**闭环，但逐步独立 argmin 会产生高频分支切换，直接对应控制
rate 与二阶抖动退化；不能只看 selected model cost 就宣布控制更优。

实时性也明确失败：全程 baseline controller duration mean/P95 为 `60.5/62.6 ms`，guard
为 `240.9/246.8 ms`；成熟段为 `63.4/90.0 ms` 与 `242.2/269.2 ms`，50 ms deadline miss
均为 100%。Actor runtime 单独稳态约 `122 ms`。step-mode 保持仿真 `dt=0.05 s`，所以轨迹
机制比较仍可用，但当前 387-rollout guard 绝不能称为实时部署候选。

本 pilot 只有一个 seed/一个 2.8 m/s 场景，因此下一步不是立刻打开 formal validation，
而是先修正闭环机制：在不破坏 warm 下界的前提下加入 **hysteresis / minimum dwell /
switch penalty** 的状态化选择，且对被选择序列额外计控制切换成本；随后用多个 matched seed
和至少 nominal/high-speed/recovery 三类内部场景复跑。并行工程阻塞是把 129-rollout
feedback 与 warm 256 bank 合并或蒸馏，否则即使轨迹 gate 改善也过不了 50 ms realtime
gate。

---

## 10. 速度域扩展到 100 kph 的契约与验证计划（2026-08-13）

现有数据速度域 `1.2--2.8 m/s` 明显过窄，实际需求是覆盖到 `100 kph = 27.78 m/s`。
本节记录两件事：（a）对 §7.6“先确认车辆尺度/DBM 参数”一条的修正；（b）从数据到
验证的分阶段计划。

### 10.1 撤回“先确认车辆尺度”这一前提

DBM 是**被信任的合成 oracle**，其唯一作用是验证训练流程是否可信，而不是拟合某台真车。
因此：

- 车辆尺寸、是否物理真实、100 kph 对 `0.21 m` 小车是否现实，**都与结论无关**，不需要
  专门确认——只要 DBM 环境自洽、确定性、可独立复算即可；
- §7.6 中“恢复该事项前应先单独确认车辆尺度、DBM 参数……”一条**予以撤回**（那是物理
  真实性顾虑，在“DBM 即真值”框架下不成立）；
- 保留 §7.6 中仍然成立的一条：**高速数据必须独立**，不得混入现有
  `1.2--2.8 m/s` 的 train/validation/test（用户已明确支持）。

### 10.2 仍然成立的三条纯方法论顾虑

这些与物理真实性无关，只关乎“环境自洽”和“训练能否学到”，扩速度前应逐条过：

1. **DBM 在 `dt=0.05 s` + 高速下的数值自洽性。** `27.78 m/s × 0.05 s ≈ 1.39 m/step`，
   轮胎/滑移项在大速度下可能饱和或病态（NaN、爆量）。要求高速 rollout 仍是良定义、
   确定性、可独立复算——这是环境检查，不是真实性检查。
2. **参数化是否还能表示最优（最便宜、最直接）。** 在高速状态上重算 GT-first 的
   `J*_16 vs J*_100`。若 `J16→J100` gap 仍可忽略，说明 `8 knots / 2.5 s` 参数化在高速下
   依然够用，瓶颈仍是策略而非维数；若 gap 变大，才需要动 horizon/knot 数。用现有
   oracle 脚本即可回答。
3. **现有尾部机制会被放大。** §7.6 记录的高速失败根因是“高速需要更大修正、direct cost
   对残余动作误差更敏感、误差在 `2.5 s` rollout 内放大”。扩到 `27.78 m/s` 正好沿同一机制
   恶化。因此独立高速数据集的**第一个用途是量化尾部形状，而不是立刻训练**。

### 10.3 从数据到验证的分阶段计划

沿用现有工程纪律（不可变原始数据 + sidecar、SHA256、独立 validator 复算、episode 级
切分、selection/audit seed 隔离、预注册 gate、封存 test）。全部阶段使用**独立的高速
数据契约与独立 seed 台账段**，不消费也不污染现有 gate 与封存 test。

| 阶段 | 内容 | 是否训练 | gate |
| --- | --- | --- | --- |
| H0 | 环境自洽性冒烟 | 否 | 高速 rollout 全部有限、确定性、独立复算误差 0；记录 DBM 可达速度上限 |
| H1 | 参数化可达性（GT-first 重算） | 否 | 高速 pilot 上 `J16→J100` gap 仍 ≪ 尾部 gap，否则先改 horizon/knot |
| H2 | 独立高速数据采集 + 尾部形状诊断 | 否 | 冻结契约 hash；按速度桶量化 `warm→J16` 可回收 gap、P05、worst |
| H3 | 高速训练 + 固定状态 open-loop gate | 是 | 复用 §9 的 7 指标 + 按速度分组 + hard guard；若严格复现当前 Actor 输入，完整预算按 387 计：mean 改善且 P05≥0、worst 有界 |
| H4 | 高速最小闭环 A/B | 是（已冻结） | matched-seed 短闭环保住基线并传导 open-loop 收益 |
| H5 | 封存高速 test 一次性评估 | — | 只做一次 |

各阶段要点：

- **H0（先做，最便宜）**：取若干高参考速度直到 `27.78 m/s`，跑固定 DBM + warm-centered
  MPPI open-loop，确认无 NaN、滑移有界、独立复算误差 0，并记录 DBM 实际能跟踪到的
  速度上限（若某速度以上 DBM 饱和，就以其为本轮可采上限，而非强行外推）。
- **H1**：在 H0 认定的可达速度区间取小 pilot，重算 `warm / J*_16 / J*_100 / J*_center`。
  这一步用最小算力回答“高速是不是维数问题”，避免在错误参数化上采大数据集。
- **H2**：采集独立高速数据集（episode 级切分、sidecar、hash、新 seed 段），并**先建立
  按速度桶的 cost 归一化**（否则高速项主导 loss）。第一产出是尾部诊断报告，不是
  checkpoint。
- **H3**：只有 H2 表明尾部有可回收空间时才训练；评价完全复用 §9 的固定状态 + hard
  guard 机制，按速度分组报告 mean/P05/worst。
- **H4/H5**：与现有低速路线相同纪律——闭环 gate 通过前不解封 test，Query/ONNX/真实车
  阶段必须另行验证 model-cost pairwise ranking。

### 10.4 本节结论

> 扩到 100 kph 不是“调高上限、多采点”，而是一个**独立契约**。因为 DBM 是被信任的
> 合成 oracle，无需确认车辆尺度；但数据必须独立、按速度桶归一化，且**先做 H0/H1/H2 的
> 环境与尾部诊断，再决定是否训练**。当前低速路线的尾部（2.4/2.8 m/s）尚未通过，扩速度
> 会沿同一机制恶化，因此高速数据集的首要用途是量化尾部，而非直接产出 checkpoint。

---

## 11. Reward 平滑性验证计划（G0--G3，2026-08-13）

目前的根因候选是“critic 的梯度学不好”。但在继续修 critic 之前，必须先确认它要学的
对象——`reward = J_direct(anchor) − J_direct(center)`——的局部梯度**本身是否良定义**。
若 reward 曲面粗糙或梯度无法跨状态共享，则问题不在网络，修 critic 是徒劳的。

把“reward 是否平滑”拆成两个可证伪的子问题，因为它们对应完全不同的修法：

- **A. 同状态下 `J_direct(c)` 在 knot 空间是否平滑**：梯度是否良定义、能否被局部二次式
  表示（决定 critic 结构是否够用）。
- **B. 局部梯度能否跨状态连续**：critic 泛化的前提。§4 已有强信号——相邻帧梯度 cosine
  仅 `0.044`；若成立，则梯度跨状态不可共享，critic 结构上无法泛化。

### 11.1 G0：纯重分析已有 sidecar（先做，0 次新 rollout）

复用已冻结的 immutable sidecar，不产生新算力：

- TR1 的 `65 centers/frame` Hadamard antithetic forward-cost（radii `0.05/0.15 σ`）；
- alpha line-search 的 `0:0.05:1` 21 点线数据。

计算三类诊断，全部**按速度桶分层**：

1. **二次拟合质量**：每帧拟合局部二次式 `Q = V + gᵀδ + ½δᵀHδ`，报 R² / 残差 →
   critic 的二次结构是否在数据上就表达得出来；
2. **alpha 线凸性 / 局部极小个数**：检查 warm→Actor 方向本身是否光滑、是否多峰；
3. **FD 半径敏感性**：用两个 radius（`0.05` vs `0.15`）分别估方向导数，看符号是否一致 →
   有限差分是否在 critic 实际使用的半径上稳定。

独立复算误差应为 0（确定性 DBM）。G0 是纯分析，可立即做。

### 11.2 G1：定向多半径探针（新 rollout，小冻结集）

沿固定 Hadamard 方向取 `r ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5} σ`，观察 FD 斜率是否
随 `r→0` 收敛（平滑）还是乱跳 / 变号（粗糙），记录“梯度可信半径”与局部曲率。回答：
critic 想学的梯度，在它实际使用的半径上是否真实存在。

### 11.3 G2：跨状态梯度连续性（决定性实验）

把 §4 的 `0.044` 从单点扩成曲线：在受控状态距离下（相邻帧 / 同状态跨 seed / 同速不同
episode）各估一次局部梯度，画**梯度 cosine 与幅值比 vs 状态距离**，按速度分层。若高速下
梯度在极小状态变化内就去相关，则 critic 结构上无法跨状态共享梯度——这就是真瓶颈。

### 11.4 G3：一维密扫 + 确定性确认

沿可解释方向（每个 knot 通道、warm→J16、warm→Actor）细粒度扫 `J_direct`，数局部极小 /
拐点；重复评估确认 bit 级确定性，把“曲面粗糙”与“随机噪声”分开。

### 11.5 判定与下一步

| G0/G1/G2 结果 | 结论 | 修法 |
| --- | --- | --- |
| 平滑 + 跨状态连续 | 是学习 / 优化问题 | 修 critic（容量、损失、trust radius） |
| 同状态粗糙（FD radius-敏感、二次 R² 低） | 梯度不良定义 | 放弃细粒度梯度，用更大尺度 value/ranking |
| 跨状态不连续（G2 快速去相关，高速更差） | 梯度跨状态泛化差 | **先**试更完整状态表示 / 局部模型 / mixture Critic 恢复可预测性；仍失败才改直接 forward-rollout 选择 |

值得注意：§9 的 258/387 hard guard 之所以能零回归，正是因为它用**直接前向 rollout 选择、
完全绕开了学习梯度**。若 G1/G2 证实 reward 曲面粗糙或梯度跨状态不连续，那同时解释了
“为什么 critic 一直学不好”和“为什么直接 rollout 兜底反而有效”——两者互为佐证。

纪律沿用现有：冻结状态、独立复算误差 0、速度分层、SHA256、不碰封存 test。

### 11.6 G0 执行结果（2026-08-13）

产物：

```text
scripts/model_verify/analyze_mppi_reward_surface_g0.py     # 纯 numpy，无 rollout、无 torch
outputs/mppi_proposal/reward_surface_g0_20260813_v1/analysis.json
```

输入为两个冻结 train-only sidecar：`dbm_j16_local_curvature_train_{diverse,expansion}`
（3150 帧，65 centers，radii `0.05/0.15 σ`）与 `dbm_direct_trust_region_train_20260807_v1`
（6300 条 alpha 线）。独立复算：base cost 误差 0、方向斜率误差 `3.6e-5`、方向曲率误差
`4.6e-3`（float32 精度）、alpha argmin 复算失败 0。全程未触碰 validation 与封存 test。

**D1 同状态二次拟合 R²（对角 Hadamard 二次式）**

| 速度 | mean | median | P05 |
| --- | ---: | ---: | ---: |
| 1.2 | 0.9874 | 0.9979 | 0.9454 |
| 1.6 | 0.9978 | 0.9985 | 0.9934 |
| 2.0 | 0.9963 | 0.9975 | 0.9892 |
| 2.4 | 0.9957 | 0.9969 | 0.9878 |
| 2.8 | 0.9943 | 0.9956 | 0.9840 |

**D3 有限差分半径符号一致性（0.05 vs 0.15 σ，仅计两半径均对称的方向）**

| 速度 | mean | median | P05 |
| --- | ---: | ---: | ---: |
| 1.2 | 0.8474 | 0.8750 | 0.5625 |
| 1.6 | 0.9379 | 0.9375 | 0.7500 |
| 2.0 | 0.9306 | 1.0000 | 0.6875 |
| 2.4 | 0.8894 | 0.9375 | 0.6250 |
| 2.8 | 0.8600 | 0.8750 | 0.5625 |

**D2 alpha 线形状（warm→proposal，21 点）**

| 速度 | 局部极小 mean | 负二阶差分比例 mean |
| --- | ---: | ---: |
| 1.2 | 0.544 | 0.026 |
| 1.6 | 0.322 | 0.024 |
| 2.0 | 0.226 | 0.099 |
| 2.4 | 0.173 | 0.158 |
| 2.8 | 0.126 | 0.188 |

**解读。**

1. **子问题 A（同状态平滑）基本成立。** D1 的 R² 在五档速度全部 ≥0.987（P05 ≥0.944），
   说明在 critic 实际使用的小半径（0.05/0.15 σ）上，cost 曲面**局部良定义且可被二次式
   很好描述**。这**不支持**“reward 曲面在操作半径上本质粗糙 / 梯度不良定义”这一假设。
2. **D3 显示存在、但不占主导的半径敏感性。** 整体符号一致率约 0.90；但在各速度的 P05
   尾部只有约 0.56--0.75，即**有一撮帧的梯度方向会随半径翻转**。这不是主因，但说明
   critic 在尾部帧上拿到的梯度方向不完全可靠。
3. **D2 表明 warm→proposal 方向大体是干净下降线。** 绝大多数 alpha 线单峰（局部极小
   mean ≤0.54，中位数在 ≥1.6 时为 0）；高速下负二阶差分比例升到 0.19，即**高速段线更凹、
   非凸性增强**，但仍是少数。

**结论：G0 未能在“同状态、小半径”层面找到 reward 不平滑的强证据，因此把根因指向
reward 曲面本身（子问题 A）的说法证据不足。诊断重心应转向：**

- **G2 跨状态梯度连续性**——这是 G0 无法回答、且 §4 已有 `0.044` 强信号的方向；
- **G1 更大半径（到 0.5 σ / alpha→1）行为**——critic 的 trust step 实际可走到远超
  0.15 σ，而 D1/D3 只测了小半径。

局限：D1/D3 仅沿 16 个 Hadamard 轴、两个小半径估计，**不含 Hessian 交叉项**；是 train-only
诊断，不涉及泛化。因此 G0 的“平滑”结论只适用于所测的轴向小半径邻域，不能外推为大范围
或跨状态性质。

### 11.7 对 G0 的复核修正与 G0.1（2026-08-13）

对 §11.6 的复核结论：**G0 只能算对“reward 局部平滑”的弱支持，不能据此认定 Critic 只是
训练不足。** 修正如下。

**1. D1 的证据强度没有 R² 数字看起来那么高。** 每个方向只有 4 个点
（`±0.05σ, ±0.15σ`），用两个参数（斜率、曲率）拟合且中心 cost 已知，自由度很高；且 R²
是在**用于拟合的同一组点**上计算的，不是外推误差。此外当前先对每帧 16 方向 R² 求平均再取
帧级 P05，可能掩盖少数很差、但对最终梯度很重要的方向。因此 D1 只能证明“轴向截面大致平滑”，
不能证明：完整 16 维曲面能被当前 Critic 表达 / 交叉曲率足够简单 / 小半径拟合能预测 Actor
实际走到的位置。

**2. D3 还不能完整描述“梯度方向是否稳定”。** 两个问题：（a）排除了发生 clipping 的非对称
方向，而高速、边界状态可能恰好最容易被排除；（b）接近零的斜率符号翻转并不重要，大幅主方向
翻转才重要，目前两者权重相同。更有价值的指标是 `cosine(g_0.05, g_0.15)`、
`‖g_0.15‖/‖g_0.05‖`、主导方向 top-k 符号一致率，并把 clipped 与 unclipped 分开统计。

**3. D2 的“单峰”描述需收窄。** 脚本统计的是严格局部极小个数与负二阶差分比例；局部极小为 0
不代表凸或干净下降（单调升 / 单调降 / 只有局部极大都会得到 0）。且高速 0.188 表示平均约 19%
的区间二阶差分为负，不等同于“只有 19% 的帧非凸”。应补：存在任一负二阶差分的帧比例、argmin
位于 `alpha=0/内部/1` 的比例、warm→proposal 单调下降比例、最大负曲率及其尾部分布。

**4. “独立复算”是代数重构，不是 rollout 级独立验证。** G0 脚本从同一 sidecar 的 stored
cost 重算 slope/curvature/argmin，验证的是字段与分析公式，没有重跑 DBM dynamics/cost。该
表述保留，但应理解为代数一致性检查。

**复核后的总体结论：**

> G0 基本排除了“同状态、小半径 reward 普遍粗糙”作为主因，但**没有证明 Critic 的
> 状态→梯度映射是可学习的**。主嫌疑进一步集中到：跨状态泛化、动作中心变化、以及少数尾部
> 状态的半径敏感性。

**下一步：先做零 rollout 的 G0.1，再做 G2。**

- **G0.1（零 rollout）**：用 `±0.05σ` 拟合斜率/曲率去**预测** `±0.15σ`（反向也做一次），得到
  真正的跨半径预测误差；直接计算 `g_0.05` 与 `g_0.15` 的 cosine、幅值比、主方向一致率；单独
  统计 clipping、高速和符号翻转尾部。
- **G2**：必须在**共同动作中心**上比较相邻状态梯度，否则“状态变了”与“Actor anchor 变了”
  会混在一起。此前的 cosine `0.044` 还不能直接归因于跨状态不连续。

### 11.8 G0.1 执行结果（2026-08-13）

产物：

```text
scripts/model_verify/analyze_mppi_reward_surface_g0_1.py     # 纯 numpy，无 rollout、无 torch
outputs/mppi_proposal/reward_surface_g0_1_20260813_v1/analysis.json
```

输入同 G0（3150 曲率帧：1070 clipped / 2080 unclipped；6300 alpha 线），仍为代数重构、
未重跑 DBM、未触碰 validation / 封存 test。

**C1 跨半径前向预测误差（用 `±0.05σ` 拟合去预测 `±0.15σ`，mean / p95）**

| 速度 | mean | p95 |
| --- | ---: | ---: |
| 1.2 | 0.289 | 0.402 |
| 1.6 | 0.473 | 0.664 |
| 2.0 | 0.880 | 1.495 |
| 2.4 | 1.569 | 4.862 |
| 2.8 | 2.007 | 5.876 |

**C2 梯度向量稳定性（`g_0.05` vs `g_0.15`）**

| 速度 | cosine mean | 幅值比 mean | top-4 符号一致 mean |
| --- | ---: | ---: | ---: |
| 1.2 | 0.961 | 8.70 | 0.977 |
| 1.6 | 0.994 | 8.91 | 0.990 |
| 2.0 | 0.987 | 8.73 | 0.991 |
| 2.4 | 0.940 | 8.59 | 0.966 |
| 2.8 | 0.857 | 6.92 | 0.938 |

**C3 clipping 分层（2.8 m/s）**：unclipped（n=146）前向预测误差 mean **3.81**，clipped
（n=484）仅 1.46；cosine 分别 0.954 / 0.828。**高速的跨半径误差在 unclipped 帧反而更大**，
说明这不是 clipping / 边界伪影，而是高速 cost 曲面本身的高阶 / 非二次结构。

**收紧后的 alpha 线诊断**

| 速度 | 任一负二阶差分帧比例 | 单调下降比例 |
| --- | ---: | ---: |
| 1.2 | 0.194 | 0.450 |
| 1.6 | 0.047 | 0.633 |
| 2.0 | 0.155 | 0.689 |
| 2.4 | 0.254 | 0.684 |
| 2.8 | 0.309 | 0.710 |

argmin 位置总体：`alpha=0` 8.1 % / 内部 26.9 % / `alpha=1` **65.0 %**。最大负曲率 mean
0.011、p95 在高速约 0.08——凹性即使存在也较温和。

**解读。**

1. **梯度方向是稳的。** cosine 全程 ≥0.86（mean），top-4 符号一致 ≥0.94。G0 里 D3 的
   “P05 尾部符号翻转”很大程度上是近零斜率的伪影；改用向量 cosine 后，方向不稳定并非主因。
2. **但小半径二次模型外推不动。** 跨半径前向预测误差从 1.2→2.8 增大约 7 倍（0.29→2.01），
   且梯度幅值比 ≈8（随半径强变化）。这说明**在 0.05σ 标定的二次 critic，无法可靠外推到
   Actor 实际走到的更大半径**，且随速度恶化——正落在尾部失败的高速段。
3. **高速粗糙是本征的**：unclipped 2.8 误差大于 clipped，排除了“只是边界 clipping”的解释。
4. **alpha 线 65% 的最优点落在 `alpha=1`**，说明下降方向在信任半径内尚未走到底；单调下降
   占比随速度升到 0.71。

**结论：G0.1 把诊断从“梯度方向是否可靠”改写为“小半径→大半径的尺度 / 曲率外推在高速失效”。**
这进一步支持用**直接前向 rollout 评估**（即 §9 的 258/387 hard guard 所为）来替代“信任一个
外推出来的小半径梯度”，也解释了为什么 rollout 兜底有效而梯度 critic 学不好。要最终归因
critic 失败，仍需 **G2**（在共同动作中心上比较相邻状态梯度），把“跨状态不连续”与“动作中心
变化”分开。

局限：仍只沿 16 个 Hadamard 轴、两个半径，未含交叉曲率；是 train-only 代数重构，未重跑
DBM，不涉及泛化。

### 11.9 G1-A：真实有限差分 Critic 梯度审计（2026-08-13）

这一步按“先确认 Critic 梯度到底错到什么程度”的优先级执行，不更新 Actor/Critic，也不进入
闭环。产物：

```text
scripts/model_verify/analyze_mppi_direct_critic_fresh_fd.py
scripts/model_verify/validate_mppi_direct_critic_fresh_fd.py
outputs/mppi_proposal/direct_critic_fresh_fd_20260813_v2/
  fresh_fd_audit.npz
  summary.json
  validation_summary.json
```

审计仅使用已经消费过的 600 个 `train/internal_selection` 状态；未读取 formal validation/test。
冻结当前 residual Actor 和三个 `TorchMPPIActorCenteredLocalCritic`，在 Actor 唯一中心周围用固定
随机正交 16 方向（不是训练标签的 Hadamard 方向）以及全新的
`0.01/0.02/0.04 source-sigma` 半径，实际重跑 `600×3×33=59,400` 条 DBM center rollout。
reward 与训练完全同口径：

```text
z(s,a) = asinh((J_alpha_base(s) - J_direct(s,a)) / 5)
```

每个半径用未加噪中心和 16 对 antithetic 候选，在相同的 normalized `8×2` Actor residual
坐标中拟合 16-D 一阶梯度及一个径向曲率；三半径合并结果作为 fresh-FD 主目标。Actor anchor
投影误差为 0，只有 5.5% 状态的任一外围 probe 发生 control bound clipping，所有设计矩阵秩均
为 17。

**A. 真实小半径梯度非常稳定。**

| fresh 半径对 | gradient cosine median | P10 | minimum | norm ratio median |
| --- | ---: | ---: | ---: | ---: |
| 0.01 vs 0.02 | 1.000000 | 0.999978 | 0.9809 | 1.0002 |
| 0.01 vs 0.04 | 0.999995 | 0.999690 | 0.9513 | 1.0010 |
| 0.02 vs 0.04 | 0.999997 | 0.999784 | 0.9530 | 1.0009 |

因此，在当前 Actor center 的这个尺度上，不仅方向平滑，幅值也已经收敛。G0/G0.1 的“局部
平滑”在 fresh 方向、真实 rollout、未消费半径上得到强确认；不能再把当前 Critic 失败归因于
DBM reward 在小半径本质粗糙。

**B. fresh 方向不是差异来源，但旧标签混合大半径会改变目标。** 旧训练标签中最小的
`0.05σ` 梯度与 fresh 合并梯度 cosine median/P10 为 `0.9899/0.9590`，方向完全可迁移；幅值比
中位数为 `0.832`。而把旧 `0.05/0.10/0.20σ` 三个半径合并后，cosine 降到 `0.7726`
（P10 `0.5694`），幅值比降到 `0.3720`。逐半径方向也由 `0.05σ` 的 `0.990`、`0.10σ` 的
`0.896` 降到 `0.20σ` 的 `0.630`。这与 G0.1 一致：**大半径 secant/曲率信息不应和局部
导数直接平均成一个 gradient 标签**。但这只是标签口径的附加缺陷，因为 Critic 连与
`0.05σ` 标签本身也不能很好对齐。

**C. 当前 Critic 的真实 action gradient 明确失败。**

| 指标（ensemble mean vs fresh FD） | 数值 |
| --- | ---: |
| cosine median / P10 | 0.3447 / -0.6920 |
| cosine > 0 比例 | 63.0% |
| cosine > 0.5 比例 | 40.5% |
| 全分量 correlation | 0.1829 |
| Critic / true gradient norm median | 0.00850 |
| Critic norm median / true norm median | 0.122 / 15.727 |

三个 seed 单独的 cosine median 只有 `0.349/0.336/0.392`。按速度分层的 ensemble median 为
`0.328/0.416/0.374/0.278/0.327`（1.2→2.8 m/s），说明方向错误不是只发生在高速；但幅值
比中位数从 1.2 m/s 的 `0.0459` 进一步恶化到 2.4/2.8 m/s 的
`0.00335/0.00480`。这不仅是角度泛化失败，还存在严重的梯度幅值塌缩。此前的 normalized
signed-step 能绕开绝对幅值，所以它仍可能有小的平均收益；SAC/Actor 的直接梯度更新则会受到
幅值与方向双重影响。

**D. 只看 Q1/Q2 一致性会产生假安全感。** 三个 Critic 两两 action-gradient cosine median 为
`0.953/0.954/0.962`（P10 `0.531/0.532/0.698`），但彼此 norm ratio median 为
`0.118/0.398/3.592`，而它们对真实 FD 都很差。即三个网络学到了高度相关的方向偏差，同时
尺度并未校准。以后必须同时监控：

1. `cos(∇aQ1, ∇aQ2)` 与两者 norm ratio；
2. `cos(∇aQi, g_FD)`、true/pred norm ratio；
3. fresh episode、fresh direction、fresh radius 下的 P10/尾部，而不只看均值或 Q difference。

该 local Critic 在 anchor 处有解析 gradient head；32 状态 autograd 检查确认
`∂Q/∂a == gradient_head` 最大误差 0，所以问题不是“取梯度实现错了”，而是网络输出本身没有
拟合真实梯度。独立 validator 重建 Actor/Alpha/Critic、FD gradient、design rank 和核心指标
误差全部为 0，并随机重跑 1,024 条 DBM candidate，cost 最大误差 0；输入 checkpoint 和 artifact
SHA256 均通过。

**结论与下一步。** qualification 为
`REWARD_GRADIENT_STABLE_CRITIC_GENERALIZATION_FAIL`。第一项验证已经把主问题明确为 Critic
监督/泛化，而不是理论梯度不存在。下一轮按最小改动顺序做：

1. same-state local action 数据采用 fresh/旋转方向，并把**最小半径 gradient**作为导数监督；
   `0.10/0.20σ` 只作为独立 value/ranking/curvature 样本，不再合并进 gradient 标签；
2. 在同一 state 内加入 pairwise ranking/delta loss，并做 value-only、+ranking、+gradient 三组
   冻结 Actor ablation；
3. checkpoint 选择和 gate 直接使用 untouched episode 上的 FD cosine + norm calibration + P10，
   同时记录双 Critic 对真值，而不是用 Q1/Q2 互相一致代替正确性；
4. 在以上 Critic gate 通过前，不恢复 Actor 更新；G2 跨状态共同中心连续性仍有价值，但已不是
   判断“小半径理论梯度是否存在”的前置阻塞项。

### 11.10 工程排除：action 坐标与 Jacobian（2026-08-13）

鉴于 §11.9 的 Critic/true gradient norm ratio 只有 `0.00850`，在修改训练前补做坐标审计，
排除 normalized/physical/pre-tanh 混用。当前实际链路为：

```text
z_actor --tanh--> u_req
c = clip(c_alpha + M * sigma * u_req, -1, 1)
u_eff = (c - c_alpha) / (M * sigma)
Q = Critic(s, anchor=u_actor, normalized_action=u_eff)
```

其中 `M=2.0`，两个控制通道的 `sigma=[0.25,0.35]`。checkpoint buffer、训练 payload 和
600 个审计状态保存的 sigma 最大误差均为 0。Critic 接收、gradient head 表示和 autograd
求导的对象都是 `u_eff`，不是 physical center，也不是 `z_actor`。

fresh FD 虽在 center 空间生成：

```text
c_plus/minus = clip(c_actor +/- r * sigma * d, -1, 1)
```

但拟合前明确转换为：

```text
u_plus/minus = (c_plus/minus - c_alpha) / (M * sigma)
```

所以 FD target 与 Critic gradient 都是同一个 `d transformed_reward / d u_eff`。相应
Jacobian 为：

```text
dQ/dc       = (dQ/du_eff) / (M * sigma)
dQ/dz_actor = (dQ/du_eff) * clamp_mask * (1 - tanh(z_actor)^2)
```

数值检查结果写入
`outputs/mppi_proposal/direct_critic_fresh_fd_20260813_v2/validation_summary.json`：

- normalized-FD 转 physical-center gradient 的向量相对误差 median/max 为
  `1.54e-7/3.70e-7`；
- Critic center Jacobian autograd 最大误差 `0`；包含 tanh+clamp 的 pre-tanh Jacobian
  最大误差 `2.60e-6`；
- tanh Jacobian minimum/median 为 `0.99913/0.99988`，没有 tanh 饱和；
- Actor center clamp 只影响 `0.302%` 的 action 元素、`4.67%` 的 context，需保留作边界
  尾部诊断，但不能解释全体梯度塌缩；
- physical-center→normalized Jacobian 两通道为 `2.0/1.4286`，条件数仅 `1.4`。

把 Critic 与 fresh FD 同时换坐标后的结果：

| 比较坐标 | cosine median | P10 | positive | norm ratio median |
| --- | ---: | ---: | ---: | ---: |
| effective normalized `u_eff` | 0.3447 | -0.6920 | 63.0% | 0.00850 |
| physical center `c` | 0.2655 | -0.6770 | 61.7% | 0.00871 |
| Actor pre-tanh `z_actor`（含 clamp） | 0.3447 | -0.6920 | 63.0% | 0.00850 |

reward 方面，Critic 和 FD 都对
`asinh((J_alpha_base-J_direct)/5)` 求导；若两者一起换回 raw reward，每个 context 只是给
全部 16 维乘同一个标量，不改变该 context 的 cosine 或 predicted/true norm ratio。

**结论：action 坐标、逐通道 scale、tanh、clamp 与 autograd 对象已经一致，坐标工程问题
被排除。** `0.85%` 幅值比例和低 cosine 仍成立，主问题继续是 Critic 监督目标与跨 episode
拟合/泛化。clamp 只作为少量边界状态的次级诊断保留。

### 11.11 Critic 梯度重训方向与推荐顺序（2026-08-13）

首先明确当前监督性质：这不是 teacher 网络蒸馏，也不是 TD/SAC 的 Bellman 训练。当前冻结
Actor 后，对每个状态真实运行 DBM 候选，再由同状态 antithetic 有限差分拟合数值标签：

```text
DBM forward rollout -> transformed reward probe bank
                     -> numerical FD (V*, g16*, h*)
                     -> supervised local-Critic regression
```

这里的“oracle”是固定 DBM + 数值有限差分过程，不是另一个已学习策略。Critic 的
`local_parameters` 直接输出 `(V,g16,h)`，且 §11.10 已验证 anchor 处
`dQ/du_eff == g16`。因此 `g16` 是网络要预测的任务量，不是训练反传梯度；之后 Actor 才消费
它作为策略梯度。

**两个候选方向不是替代关系。** 标签处理决定“正确目标是什么”，loss 决定“网络如何逼近
这个目标”。当前 trainer 实际已经包含：gradient component Smooth-L1、gradient cosine、
anchor value、curvature、全 bank 二次重构。因此“原 Q loss + same-state `0.05σ` 局部差分
loss”在思想上正确，但对当前显式 gradient-head Critic 来说，已有近似等价且更直接的监督；
若不先修标签混合和 checkpoint 规则，单纯再加一次同类 loss 很可能重复现有训练。

**P0：先修标签角色和选择规则，零新 rollout 做受控验证。**

1. 当前 `0.05/0.10/0.20σ` 三半径被合并成一个 `g*`。§11.9 已证明最小 `0.05σ` 对 fresh
   FD 的 cosine median/P10 为 `0.9899/0.9590`，但三半径合并后只有
   `0.7726/0.5694`。先直接复用已有 sidecar，以 **`0.05σ` 单独作为 gradient target**；
   不需要先生成新数据即可验证混合标签是不是主要损伤。
2. `0.10/0.20σ` 表示 finite-radius response，不再进入真导数标签。保留为 value、同状态
   pair delta/ranking、曲率/形状样本。由于当前只有一个 scalar radial curvature，不能保证
   表达方向相关高阶项；大半径 bank loss 不应以较大权重反向迫使 `g16` 补偿曲率误差。第一轮
   消融应降低/隔离这部分对 gradient head 的影响，必要时再给 finite-radius response 独立 head。
3. 当前 checkpoint score 是 `1-cosine_median+0.05*regret`，完全看不到 gradient norm。
   这是幅值坍缩能被选中的直接工程原因。新 score/gate 必须同时包含：fresh-FD cosine median、
   cosine P10、`abs(median(log(||g_pred||/||g_true||)))`、pair delta/sign 和真实 probe regret。
   norm ratio 必须作为硬 gate，而不只是日志。

**P1：在正确标签上增加真正互补的 loss。** 推荐保留显式 gradient head，而不是先退回通用
scalar-Q MLP。最小目标为：

```text
L = lambda_v    * L_value(anchor)
  + lambda_comp * Huber(g_pred, g_FD_small)
  + lambda_dir  * (1 - cosine(g_pred, g_FD_small))
  + lambda_mag  * Huber(log||g_pred||, log||g_FD_small||)
  + lambda_diff * Huber((Q+ - Q-), (z+ - z-))
  + lambda_rank * ranking(Q+, Q-, z+, z-)
```

- `L_comp` 保留每维幅值与符号；
- `L_dir` 解决方向；
- 新增 `L_mag` 明确阻止整体 norm 向零坍缩；
- `L_diff/L_rank` 使用同一 state 的 `+/-0.05σ` paired response，抵消 state/value offset，
  强制网络知道哪一方向改善。只加 ranking 不够，因为它不校准幅值；只加 value MSE 也不够，
  因为很小的 action slope 对总 value 误差贡献有限。

若未来换回 generic `Q(s,a)`，same-state pair delta/ranking loss 仍然应该保留；但要让 autograd
action gradient 匹配 FD 还需要二阶反传或单独 gradient-consistency 约束。当前解析 gradient head
避免了这层优化困难，暂不建议更换。

**P2：随后生成更严格的小半径标签。** `0.05σ` 已足够做零成本机制消融，但其幅值相对
`0.01--0.04σ` fresh gradient 的 median ratio 为 `0.832`，仍有约 17% finite-radius 偏差。
若 P0/P1 确认训练能恢复 norm，再为 train 状态生成 `0.01/0.02σ` rotated/orthogonal probes：

- `<=0.02σ` 专门监督真导数；
- `0.05σ` 作为 local pair delta/ranking；
- `0.10/0.20σ` 仅监督 finite-radius value/ranking/shape；
- 不把不同半径重新平均成一个 gradient。

**受控消融顺序。** Actor 全程冻结，先在已消费 split 做机制诊断，不作新泛化声明：

| 组别 | gradient 标签 | loss/选择变化 | 回答的问题 |
| --- | --- | --- | --- |
| A0 | 旧三半径合并 | 当前实现 | 可复现基线 |
| A1 | 仅 `0.05σ` | 其余不变 | 标签混合是否主因 |
| A2 | 仅 `0.05σ` | + pair delta/ranking | 同状态相对关系收益 |
| A3 | 仅 `0.05σ` | + magnitude loss + norm-aware score/gate | 是否解除幅值坍缩 |

只有 A3 在 fresh-FD direction、norm、P10、pair regret 同时改善后，才花 rollout 预算生成
全训练集 `0.01/0.02σ` 标签，并用新采 episode-heldout 做资格 gate。还应按速度平衡 batch：
当前 true gradient norm 中位数由 1.2 m/s 的约 `2.79` 增至 2.8 m/s 的约 `32.06`，单一全局
component std 容易使低速方向和高速幅值互相牵制。

**推荐结论：先做标签拆分 + norm-aware checkpoint（P0），再加 magnitude 与 same-state
delta/ranking loss（P1）；不要把两者当二选一，也不要立即重跑全部标签。** 这能用现有数据
最快区分“错误标签/早停选择”与“网络确实无法泛化”。

#### 11.9.1 独立核对确认 + norm 坍缩分解 + 标签重训目的（2026-08-13）

我独立重读了 `analyze_mppi_direct_critic_fresh_fd.py`、
`validate_mppi_direct_critic_fresh_fd.py` 和 v2 全部三个产物，§11.9 的数字与方法学结论全部
复核通过：fresh 方向库与训练 Hadamard 无交集且正交误差 `1.2e-7`；fresh 半径
`0.01/0.02/0.04σ` 与训练半径 `0.05/0.10/0.20σ` 无交集；autograd 对 gradient head 误差 0，
确认比较的正是 Actor 实际使用的那个头；validator PASS，hash/split 映射/重建/1,024 条 rollout
回放误差全部为 0。结论 `REWARD_GRADIENT_STABLE_CRITIC_GENERALIZATION_FAIL` 成立，且与
G0→G0.1 的证据链闭合：reward 在小半径平滑、理论梯度存在，当前失败是 Critic 监督/泛化问题。

**A. norm 坍缩表示什么。** 先给定义：真梯度 `g_FD = ∇_a z(s,a)` 告诉 Actor”动作沿哪个方向、
以多大速率能改善 reward”。Critic 预测 norm 中位数 `0.122` 只有真值 `15.727` 的 `0.85%`，
意味着 Actor 收到的策略梯度**有效信号强度只有真实值的 ~1/120，且方向 cosine 只有 0.34**——
等效于用噪声做更新，这正是此前 Actor 学不动的直接机制。

坍缩可拆成两级，归因不同：

| 层级 | 数值 | 归因 |
| --- | ---: | --- |
| 真梯度 norm | 15.727 | — |
| 训练标签（三半径合并）norm | 4.908（31%） | 标签口径错误：大半径 secant 与导数平均 |
| Critic 预测 norm | 0.122（真值的 0.85%，标签的 2.5%） | **训练机制问题，与标签无关** |

关键点：即使换成最干净的 `0.05σ` 单半径标签（norm 12.03，为真值的 77%），Critic 仍比它低
约 **100×**，所以标签修正救不回幅值，必须另查训练内部。按速度分层给出进一步证据：真梯度
norm 从 1.2 m/s 的 `2.8` 涨到 2.4 m/s 的 `38.6`（约 14×），而 Critic 预测 norm 在
`0.107–0.153` 之间几乎平坦——网络学到的是一个与状态近乎无关的微小常数梯度，各速度段 cosine
均匀地差（`0.278–0.416`）也印证这是全局性坍缩而非高速尾部问题。嫌疑机制（下一轮逐项排除）：
gradient head 的 loss 权重过低或被 value loss 淹没；weight decay/正则把梯度头压向 0；幅值没有
独立监督（只从 value 残差间接学）；输出有界/被 squeeze。

**B. 标签重训的目的。** 重训不是为了”多采数据”，而是把监督目标修成正确的对象，三个目标：

1. **方向**：`0.05σ` 与 fresh FD 的 cosine 已有 `0.990`，改用更小的 `0.01–0.02σ`
   （rollout 条数完全相同，中心差分偏差更小）后方向监督可视为近乎无偏；
2. **幅值口径**：剔除 `0.10/0.20σ` 后标签 norm 从真值的 31% 恢复到 77%；剩余 ~23% 衰减是
   有限半径与 asinh 变换的已知偏差，可用更小半径继续压缩，或在 loss 中显式校准；
3. **角色分离**：大半径探针并非废掉，而是改做独立监督——value/ranking/曲率样本，不再与一阶
   导数混在一个 gradient 标签里。

同时明确预期管理：**单靠标签重训只能修复方向 + 部分幅值**；若训练机制坍缩不解决，干净标签
同样会被压成常数梯度。因此 §11.9 下一步第 2 条的三组 ablation（value-only / +ranking /
+gradient）应以本节的两级分解为验收口径：fresh-FD cosine 验收方向，true/pred norm ratio
验收幅值，两者都过才恢复 Actor 更新。fresh-FD 6-gate 脚本（本节的 analyze/validate 对）
固化为以后每次 Critic 重训的验收 gate，Q1/Q2 互一致不再作为正确性证据。

### 11.12 对“训练侧优先”review 的复核与实验顺序修正（2026-08-13）

review 的核心判断成立：`0.122` 不仅远小于 fresh FD `15.727`，也远小于 Critic **实际训练
目标**——三半径混合标签 `4.908`。按“中位数之比”分别是 `0.776%` 和 `2.49%`，即约
`1/129` 与 `1/40.2`；按逐 context ratio 的中位数是 `0.850%` 与 `2.58%`。因此标签修正
最多先把监督目标变正确，无法单独解释或修复“网络只输出标签约 1/40”的第二级坍缩。

为排除这只是 internal-selection 泛化现象，又对冻结三 seed ensemble 在原训练 split 上复算：

| split | pred norm median | mixed-label norm median | ratio of medians | per-row ratio median | cosine median | norm correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train（4590） | 0.1288 | 4.3406 | 2.97% | 3.05% | 0.767 | 0.387 |
| internal-validation（1110） | 0.1263 | 4.3124 | 2.93% | 2.92% | 0.463 | 0.262 |
| internal-selection（600） | 0.1220 | 4.9080 | 2.49% | 2.58% | 0.463 | 0.327 |

**这证明 norm 坍缩在训练集本身就存在，主因确实是训练 pipeline（优化、loss 竞争、正则、
输出头或 checkpoint 选择），而不是只在跨 episode 泛化时出现。** 方向问题则不同：训练集
对有缺陷 mixed target 已到 `0.767`，到 episode-heldout 降为 `0.463`，再换成 fresh true
target 降为 `0.345`；所以方向同时包含标签口径、训练拟合和泛化三部分，不能用 cosine 做严格
的加法/乘法归因，但“两者叠加”的定性判断合理。

另有两项嫌疑可由代码直接收窄：当前 component Smooth-L1 已经**直接监督每个 gradient 分量
的幅值**（按 train component std 标准化，并非只从 value 残差间接获得）；`gradient_head` 也是
无界 linear output，没有 tanh/squeeze。因此“缺少任何幅值监督”和“输出被有界激活压小”不是
准确归因。应重点检查的是：component/magnitude loss 的实际数值与 parameter-gradient norm、
cosine/bank/value/curvature 之间的梯度竞争、全参数 weight decay、零初始化后的优化动态，以及
只按 cosine+regret 早停是否在幅值长大前选中了 checkpoint。

review 中“幅值比方向更致命”的表述需要收窄：

- `0.85%` 是**幅值比，不是信噪比**；当前是确定性网络输出，不能直接称为 `1/120 SNR`。
- 若只是所有状态统一缩放，Actor learning rate、Adam 或显式 gradient normalization 可以吸收
  全局尺度；此前 RMS-normalized fixed step 仍把 mean cost `14.312 -> 14.005`，正说明小 norm
  下方向仍有部分可用信息。
- 当前真正致命的是**尺度随状态没有学到**：pred norm 约 `0.12` 且变化很小，而 true norm 随
  速度跨一个数量级；norm correlation 只有 `0.26--0.39`。一个全局 LR 无法补偿这种失校准。
  同时 cosine P10 为负，若只把梯度放大会放大错误方向。因此最终 Actor gate 必须方向和幅值
  同时过，不能“先放大再说”。

据此把 §11.11 的执行优先级修正为训练侧隔离优先，但仍保持标签随后修正：

| 组别 | 标签 | 修改 | 目的 |
| --- | --- | --- | --- |
| B0 | mixed `0.05/0.10/0.20` | 原实现，补每 epoch norm/loss 日志 | 冻结复现 |
| B1 | mixed | 仅 norm-aware checkpoint score/gate | 判断是否只是选错 epoch |
| B2 | mixed | + log-norm loss；记录各 loss 梯度；output head 不做 weight decay 消融 | 验证能否先拟合当前标签幅值 |
| B3 | `0.05` only gradient | 继承 B2，其余大半径与 gradient head 隔离 | 修正方向目标 |
| B4 | `0.05` gradient + multi-radius response | + same-state delta/ranking | 补局部相对关系与形状 |

B1 若所有 epoch 都没有正常 norm，说明旧 checkpoint score 只是“没有发现失败”，不是唯一成因；
B2 必须先在 **train** 上把 predicted/label norm ratio 拉回合理范围，同时不牺牲 cosine，才说明
训练机制修复。B3/B4 再看 fresh FD direction/norm/P10 和真实 probe regret。Actor 在 B4 gate
前保持冻结。新 `0.01/0.02σ` rollout 仍放在这些零新数据消融之后。

#### 11.12.1 四项优先级复核：loss 参数梯度审计

针对“checkpoint norm、bank loss、weight decay、`<=0.02σ` 标签”四项意见，在改训练前用
现有三个 selected checkpoint、固定 1,024 条 train rows 计算各个**已乘正式 loss 权重**的
参数梯度。结果如下：

| seed | component head-grad | cosine head-grad | bank head-grad | bank/component | cosine/component |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0430 | 1.225 | 0.00092 | 0.021 | 28.5 |
| 1 | 0.1747 | 1.275 | 0.00653 | 0.037 | 7.3 |
| 2 | 0.1421 | 1.803 | 0.00347 | 0.024 | 12.7 |

这只是 selected checkpoint 处的局部诊断，不等同于完整训练轨迹因果证明，但足以重排嫌疑：

1. **norm-aware checkpoint：必做，但不是一行就能修好。** 需要先在 metrics 中加入逐状态
   `median(abs(log((||g_pred||+eps)/(||g_label||+eps))))`、norm correlation 和速度分层，再进入
   score/hard gate。若所有 epoch 都坍缩，选择器只能暴露失败，不能创造一个好 checkpoint。
2. **当前头号训练嫌疑是 cosine/component 失衡，不是 bank 直接压头。** cosine 对幅值不敏感，
   且其梯度在预测 norm 很小时会被放大；实际对 gradient head 的参数梯度比 component loss 大
   `7--29×`。这非常符合“方向先有一点、幅值停在很小值”的形态。B2 应记录各 loss 参数梯度，
   降低/调度 cosine weight，先用 component + log-norm warmup 把幅值拉到合理区间，再开启 cosine。
3. **bank ablation 仍值得做，但“曲率兜底替代 g”不是严格机制。** 对对称探针，
   `Q(+δ)-Q(-δ)=2g^Tδ`，scalar curvature 是偶函数，会完全抵消，不能替代奇对称梯度。大半径的
   三次/高阶奇响应确实会把 effective secant gradient 拉向错误口径，bank 也可能通过 shared
   fusion 产生间接竞争；因此应做 small-radius-only / bank-off 对照，但其当前 direct head-grad
   只有 component 的 `2--4%`，不再列为头号嫌疑。
4. **weight decay 主因基本排除。** 当前 `lr=2e-4, wd=1e-5`，AdamW 每步乘性收缩仅
   `2e-9`，约 1,500 步累计约 `3e-6`；无法解释 `40×` 坍缩。gradient head no-decay 可作为
   低成本卫生对照，但优先级低于 cosine/component 平衡。
5. **`<=0.02σ` 标签有价值，但不要与训练修复同一个对照同时改。** 它需要在 5,700 fit rows
   上新增 DBM rollout；“每个半径候选数相同”不等于现有实验零成本。先用 mixed/`0.05σ` 现有
   labels 证明 B2 能在 train 拟合幅值，再单独切小半径，才能区分优化修复和标签修复贡献。

最终执行顺序建议更新为：**metrics/score 可见 norm → component/log-norm warmup + cosine
调度 → bank-off/small-bank 对照 → `0.05σ` 单标签 → 新 `0.01/0.02σ` 标签。** 每一步只改变
一类变量，Actor 继续冻结。

### 11.13 对 §11.12.1 的复核确认与两点补充条件（2026-08-13）

§11.12.1 的梯度审计数字与四项判断（norm-aware 必做但不治愈、cosine/component 失衡为头号
嫌疑、bank 降级、weight decay 排除、小半径标签不并轮）全部复核通过，无修正。其中
`Q(+δ)−Q(−δ)=2g^Tδ` 的反对称论证成立，"压低 g 让曲率兜底"在 antithetic pair 下不作为机制
保留，只保留 bank 高阶奇响应与 shared fusion 的间接竞争作为低优先级对照项。

**补充一：cosine 主导的自锁循环表述。** §11.12.1 第 2 条可进一步具体化为一个自增强循环：
cosine loss 对 `g_pred` 的梯度含 `1/‖g_pred‖` 因子且尺度不变（只推方向、无幅值分量）；
component loss 为 Huber，残差 `|pg−tg|/scale` 远大于 `beta=0.5` 时梯度饱和封顶。于是：

```text
norm 小 → cosine 参数梯度被 1/‖g‖ 放大 → 更新被方向对齐独占
→ component 的幅值推力（已饱和）被 7--29× 压过 → norm 继续小
```

这解释了 §11.12 中"train 方向 cosine 0.767 尚可、幅值却只有标签 3%"的组合形态。但注意：
cosine loss 本身不压幅值（尺度不变），它解释的是**幅值为何长不起来**（维持机制），不是
坍缩的起始原因；该审计在 selected checkpoint（收敛态）测得，起始动态需 B0 的逐 epoch 双
梯度 norm 日志补齐。

**补充二：B2 前先做一个最便宜的因果确认。** 在冻结 checkpoint 上测 component 与 cosine 两个
参数梯度向量的夹角（flatten 后求 cosine）：

- 若二者冲突（夹角 cosine 为负或接近 0），降低 cosine 权重后幅值应立刻增长，机制确认；
- 若二者近乎同向，cosine 主导只是量级现象，坍缩另有原因，B2 结论需重新归因。

**补充三：warmup 步的 cosine 回插条件显式化。** §11.12.1 顺序第 2 步"先 component +
log-norm warmup 再开 cosine"应带明确回插门槛，避免拍脑袋调权重，例如：train norm ratio
中位数进入 `[0.5, 2]` 且 train/selection cosine median 不下降。同时注意：降低 cosine loss
权重期间，现有 checkpoint score 仍以 cosine median 为主，方向指标短期可能变差；验收必须用
fresh-FD cosine **与** norm ratio 联合判断，不得只看单一指标决定是否回退。

### 11.14 Critic 训练侧受控消融结果（2026-08-13）

已按 §11.12--§11.13 的优先级修改
`scripts/model_verify/train_mppi_direct_local_gradient_critic.py`，且 Actor 全程冻结。训练器新增：

- 复用既有 `local_forward_labels.npz`，避免为纯训练消融重复执行 623,700 条 DBM rollout；
- predicted/target gradient norm、逐样本 log norm error、norm correlation 和 cosine P10；
- norm-aware checkpoint score，以及真正的 checkpoint norm-ratio eligibility hard gate；
- `combined` / 最小 `0.05σ` gradient target 切换；
- all/smallest/off bank 消融；
- cosine warmup/ramp、log-norm loss、gradient-head decay 独立开关；
- 最小半径 antithetic `Q(+)-Q(-)` delta 与 ranking loss。

GPU 冒烟测试、Python 编译和既有标签重排/哈希检查通过。正式消融均只使用已消费的
`internal_fit/internal_validation/internal_selection`，未载入 formal validation/test，也未更新 Actor。
统一对比由
`scripts/model_verify/analyze_mppi_local_critic_retraining_ablation.py` 生成：

```text
outputs/mppi_proposal/direct_local_critic_retraining_ablation_20260813_v1/analysis.json
```

#### 11.14.1 训练/内选结果

| 组别 | 单变量变化 | internal-selection cosine med/P10 | 标签 norm ratio | 0.10σ probe regret |
| --- | --- | ---: | ---: | ---: |
| A0 | 原实现 | 0.463 / -0.662 | 旧 summary 未记录 | 6.261 |
| B1 | 只增加 norm-aware score | 0.508 / -0.809 | 0.632 | 7.719 |
| B2 | log-norm 预热 + cosine 延后/降权 | 0.535 / -0.778 | 0.660 | 8.355 |
| B3 | 关闭 bank loss，其余同 B1 | 0.456 / -0.766 | 0.636 | 7.536 |
| B4 | `0.05σ` target + norm hard gate | 0.579 / -0.924 | 0.531 | 22.265 |
| B5 | B4 + 小半径 bank + pair delta/ranking | 0.567 / -0.929 | 0.606 | 22.031 |

几个实现层面的结论需要固定：

1. **旧 checkpoint 选择确实过早。** B1 的三个最佳 epoch 为 `121/134/107`，而旧实现为
   `18/33/28`。B1 训练集 predicted/label norm ratio 从约 3% 恢复到 79.8%，训练 cosine
   从 0.767 提升到 0.902。网络与 component loss 并非完全学不到幅值；原方向/argmax-only
   选模把尚处于幅值坍缩区的 checkpoint 选了出来。
2. **log-norm 预热不能单独解决方向。** B2 v1 还暴露了 scheduler/early-stop 与 warmup
   阶段冲突：cosine 尚未开启时，方向指标已把 LR 从 `2e-4` 降低。v2 将 scheduler/selection
   推迟后恢复了幅值，但方向和 regret 仍没有实质改善。训练器已永久加入
   `selection_min_epoch/scheduler_min_epoch`，避免再发生同类口径错误。
3. **bank loss 不是头号幅值坍缩原因。** B3 关闭 bank 后训练拟合相近，内选 cosine/regret
   未改善。结合 §11.12.1 的直接反传审计，保留 bank 消融是合理的，但证据不支持优先把它
   归为主因。
4. **norm gate 必须是 checkpoint eligibility，而不仅是最终报告 gate。** B4 v1 虽使用
   `0.05σ` 标签，却仍选到 norm ratio 0.099 的 checkpoint。v2 要求内部验证 norm ratio
   `[0.5,2.0]` 后才允许保存模型，消除了这个实现漏洞。

#### 11.14.2 统一 fresh-FD 口径

为了避免各组只相对自己的训练标签变好，对 A0/B1/B4/B5 重新执行相同的 600 个已消费
internal-selection 状态、随机正交方向及 `0.01/0.02/0.04σ` DBM fresh FD。每次为 59,400
条真实 DBM rollout；三个新 audit 均由
`validate_mppi_direct_critic_fresh_fd.py` 独立重建并 PASS，分别回放 1,024 条候选且最大 cost
误差为 0，gradient head/autograd 误差为 0。

| 组别 | fresh cosine med/P10 | positive | fresh norm ratio | 判断 |
| --- | ---: | ---: | ---: | --- |
| A0 原模型 | 0.345 / -0.692 | 0.630 | 0.00850 | 幅值坍缩且方向差 |
| B1 只修选模 | 0.400 / -0.729 | 0.610 | 0.22867 | 幅值提高约 27 倍，中位方向仅小幅改善 |
| B4 `0.05σ` + hard gate | **0.613** / -0.932 | 0.625 | 0.42465 | 中位方向明显改善，负尾部恶化 |
| B5 + pair loss | 0.603 / -0.932 | 0.610 | **0.48599** | 幅值略增，方向/尾部无收益 |

对 P10 从 `-0.692` 恶化到 `-0.932` 的解读需要拆开：这**部分是结构性暴露而非纯倒退**。A0
时代负方向帧的预测 norm 约为 0，错误方向不产生实际影响；幅值修复约 57 倍后，同样的方向错误
开始携带全幅值梯度，危险被显性化。这正是 §11.12 "只把梯度放大会放大错误方向" 预测的应验，
也是 fresh-FD cosine 与 norm ratio 必须联合做 gate、且当前不授权 Actor 更新的定量理由。
pair delta/ranking 在此口径下方向零收益（0.613→0.603）、norm 仅 +0.06，**降级**，后续轮次
不再默认携带。

因此本轮的精确结论是：**幅值坍缩主要由旧 checkpoint 选择不可见导致；三半径混合标签是
中位方向的重要损伤；但二者修复后，跨 episode 的负方向尾部仍未解决。** 简单地在同一批
训练 episode 上增加 pair delta/ranking，只提高训练拟合和幅值，不能修复 fresh-FD P10。
当前资格为 `AMPLITUDE_AND_MEDIAN_IMPROVED_NEGATIVE_TAIL_FAIL`，Actor 更新仍不授权。

同时必须分开“导数预测”和“候选打分”：B4/B5 学的是局部 `0.05σ` 导数，直接拿同一个
gradient + scalar curvature 去 argmax `0.10σ` 候选会把 regret 推到 22 左右。这不否定小半径
导数标签；它说明真局部 gradient 只能用于有 trust region 的小步 Actor 更新，较大半径候选
价值需要独立 finite-response head 或真实 rollout 评价，不能再混回导数头。

#### 11.14.3 下一步

不应立即更新 Actor，也不应继续无差别增加随机 state-action。下一步数据要有明确目的：

1. 在更多、按 episode/速度/场景平衡的 **internal-fit 状态** 上补同状态小半径
   `0.01/0.02σ`（或至少 `0.05σ`）正交/antithetic perturbation；每个状态覆盖完整 16D，减少
   每状态重复 rollout，而增加独立 episode 与 recovery/high-speed 状态数。
2. `<=0.02σ` 只监督 derivative head；`0.05σ` 监督 derivative + pair delta/ranking；
   `0.10/0.20σ` 进入独立 finite-response/value 目标，不再影响 derivative gradient。
3. 使用 episode/speed-balanced sampler，checkpoint 继续要求 fresh-FD median cosine `>=0.70`、
   P10 `>=0`、norm ratio `>=0.50`，并同时报告 per-speed norm calibration。
4. 只有上述 frozen-Critic gate 通过后，才恢复受 trust-region 约束的 Actor 更新；当前 B4/B5
   checkpoint 仅用于机制分析，不是部署候选。

### 11.15 负方向尾部定位：不是简单 episode 数量不足（2026-08-13）

§11.14.3 原先把下一步概括成“更多 episode/速度平衡的小半径状态”，在检查现有覆盖后需要
进一步收窄。当前实际已有：train 240 个 episode、internal-validation 60 个、internal-selection
60 个；fresh gate 的 600 帧在 5 档速度上各 120 帧、6 类场景上各 100 帧。因此不能仅凭 P10
失败断言“独立 episode 太少”。新增诊断脚本和产物为：

```text
scripts/model_verify/analyze_mppi_local_critic_negative_tail.py
outputs/mppi_proposal/direct_local_critic_negative_tail_20260813_v1/analysis.json
```

它使用 §11.14 的 B4 frozen Critic/fresh-FD 产物，不更新 Actor/Critic，不载入 formal
validation/test。结果如下：

1. ensemble 有 `225/600` 帧 cosine < 0；其中 `157/600`（26.17%）是三个 Critic **同时**为负，
   三者并非一正一负平均后才出错。三个单模型 P10 均约 `-0.93`。
2. ensemble `abs(cosine)` 中位数为 `0.866`，最差帧约 `-0.991`。因此主要尾部不是接近 0 的
   小扰动，而是接近整向 180° 的 confident sign reversal。
3. 60 个 heldout episode 中，59 个 episode 内部同时存在正/负帧；只有 1 个全正，没有全负
   episode。首个 episode 内序号较差（中位 -0.143），但后续每个序号仍有约 -0.87 至 -0.95
   的 P10。它不是一个完整 episode/domain 被统一翻转。
4. 5 档速度的 median 为 `0.394--0.812`，6 类场景为 `0.535--0.723`，但各组 P10 仍约
   `-0.84-- -0.96`。尾部不是只由高速、cold-start 或某一种 recovery 贡献。

为了区分“训练状态真的没有相似样本”和“当前表征把相反标签混在一起”，又在
`frozen Actor encoder feature + deterministic Actor action` 中对 4,590 个 train 状态做 KNN。
这只是 representation pilot，不是 oracle 或新 Critic：

| 诊断 | median cosine | P10 cosine | positive |
| --- | ---: | ---: | ---: |
| 最近 1 个 train 标签 | 0.260 | -0.907 | 55.7% |
| 最近 5 个归一化标签平均 | 0.423 | -0.853 | 58.8% |
| 最近 20 个归一化标签平均 | 0.457 | -0.863 | 61.2% |
| 最近 20 个中的 oracle-best（只作诊断） | **0.954** | **0.772** | — |

top-1 feature cosine 的中位数为 `0.890`，说明几何上并非完全离群；但 top-20 标签方向
coherence 中位数只有 `0.307`，同一表征邻域里混有相反方向。对网络已经预测为负的 225 帧，
top-5 平均标签方向中位数仍为 `-0.220`。同时 top-20 中几乎总能找到方向正确的训练样本，说明
“覆盖完全没有”也不准确，问题更像是**现有表征/回归目标不知道该选择邻域中的哪一分支**。

严格边界：该 KNN 使用冻结 Actor 表征，不证明原始 history/reference/current 输入本身缺信息；
也不重新支持早期已否定的“输出需要多中心”。它证明的是当前训练管线的 state representation
邻域对 derivative label 不够判别，随机加同分布 episode 很可能继续混合正负标签。

据此修正下一步：

1. 从已消费 split 的 fresh-FD 负尾部构造 **hard-state manifest**，记录错误帧、同表征近邻、
   相反标签 pair、速度/场景/episode 序号；先核查这些 pair 的 raw state/reference/history 差异，
   找出被 encoder 忽略的判别量。
2. 数据新增从“随机 episode”改为围绕 hard states 的小状态扰动/相邻时间帧/recovery 强度与
   曲率扰动；动作侧仍保留完整 16D 小半径 FD。目标是补 state-to-gradient 分支边界，不是再次
   增加同一状态的更多 action radius。
3. 不改 Actor 输入/输出，也不先降 16D。优先在 Critic 训练侧加入 hard-example balanced
   sampling；必要时加入 state-gradient metric/contrastive consistency，使相似且同方向的状态
   靠近、相似但反向的 hard pair 可分。它属于 Critic 表征训练，不是为环境手写 stay/sign 规则。
4. 重训验收仍用 untouched-episode fresh FD median/P10/norm gate。若 raw input pair 也几乎不可分，
   再讨论输入缺失；若 raw input 可分而 encoder 不分，则修 encoder 训练，不先盲目扩大网络。

因此 §11.14.3 的”更多 episode”被本节替换为：**先做 hard-pair raw-input 审计，再定向扩状态
边界；不建议立即随机扩集。**

#### 11.15.1 生成定向数据前还欠的三个零 rollout 归因（2026-08-13，Qoder 复核补录）

§11.15 的速度/场景分层（各组 P10 均匀为负）和 KNN coherence（0.307）已经排除了”高速/
cold-start 专属”和”完全无覆盖”两种解释，但在执行 hard-state manifest 与定向数据生成之前，
还有三个零 rollout 检查能把”表征别名 / 数据稀疏 / 场景混叠”进一步分开，避免 manifest 的
解释不唯一：

1. **225 个负帧按 clipping 状态与局部密度的交叉表。** §11.9 记录 fresh probe clipping 影响
   5.5% 的 context；需要核查负帧是否富集在 clip 边界帧（若是，部分是标签边界效应而非表征
   问题）。密度侧：统计每个负帧的 train KNN 距离分布——失败集中在低覆盖区说明定向补数据
   有效，集中在高覆盖区则确认别名问题、补数据无效，只能修表征。
2. **邻居标签离散度的可学习上界。** top-20 邻居标签互相 cosine 中位数只有 0.307，意味着任何
   仅用当前表征的回归器在这些邻域都有不可逾越的误差下界。应把该离散度换算成 per-frame 的
   best-case cosine 上界，定量回答”修表征有多少 headroom”；oracle-best 0.954 说明好分支存在，
   但当前表征分不开它们。
3. **负帧的 scenario-transition 检查。** 同一 episode 内正负帧共存（59/60 episode），需确认
   负帧是否富集在场景切换/控制模式切换的过渡窗口——若是，答案可能是场景条件化（regime
   conditioning），比表征手术便宜得多。

三项均为纯分析，输入已在 `direct_local_critic_negative_tail_20260813_v1/` 和 fresh-FD 产物中。
另记录一个未执行的遗留项：§11.13 补充二建议的 component/cosine 参数梯度夹角测量未做；B1/B2
的结果已与 cosine 主导机制一致，该项不阻塞，仅在后续做 cosine 权重消融归档时补测。

### 11.16 B-1：32-state tiny-set overfit 验证（2026-08-13）

在继续解释 §11.15 的跨状态负尾部前，已补做一个零新 rollout 的小集合记忆测试，用于排除
“Critic/optimizer 连几十个固定梯度都拟合不了”的更基础故障。实现、严格产物和独立验证为：

```text
scripts/model_verify/overfit_mppi_direct_local_gradient_critic.py
scripts/model_verify/validate_mppi_direct_local_critic_tiny_overfit.py
outputs/mppi_proposal/direct_local_critic_tiny_overfit_20260813_v2/
```

`v1` 使用 median/P10 gate，仍允许极少数样本 cosine 只为正但未达到 0.98；发现该口径不足后，
立即由严格 `v2` 取代。`v2` 固定选取 32 个 internal-fit train 状态，来自 32 个不同 episode，
覆盖 5 档速度（每档 6--7 帧）和 6 类场景（每类 5--6 帧），排除 Actor center 接近控制边界及
近零梯度状态。目标只使用已有 `0.05σ` finite-difference 标签，梯度 norm 范围
`0.754--39.208`、中位 `9.586`。Actor 冻结，formal validation/test 未加载，新增 DBM rollout
为 0；dropout、weight decay、scheduler 和 early stopping 全部关闭。

两组使用完全相同的状态、初始化和标签：

- B-1a `gradient_only`：component Smooth-L1 + 0.20 log-norm；
- B-1b `b4_multitask`：恢复 B4 的 value/gradient/cosine/curvature/all-radius-bank 权重。

严格逐样本 gate 要求 cosine median `>0.99`、P10 `>0.98`、**minimum `>0.98`**，norm ratio
median 位于 `[0.95,1.05]` 且所有样本位于 `[0.8,1.2]`。结果为：

| arm | 3-seed cosine median | cosine P10 | cosine minimum | norm median | norm min--max | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B-1a gradient-only | 0.999993--0.999999 | 0.99738--0.99769 | 0.98219--0.98832 | 0.99997--1.00018 | 0.99867--1.00110 | 3/3 |
| B-1b B4 multi-task | 0.999265--0.999466 | 0.99776--0.99856 | 0.98334--0.99314 | 0.95548--0.96716 | 0.82334--1.15435 | 3/3 |

独立 validator 重新载入 6 个 checkpoint、重建 frozen Actor/输入/标签并逐项复算，结果
`PASS`；最大 checkpoint prediction replay error 为 `9.54e-7`，Actor action/center 重建误差
分别为 `1.19e-7/5.96e-8`。

结论需要分两层：

1. Critic 架构、action 坐标、gradient head 和 optimizer **具备记忆完整 16D 数值梯度的能力**；
   全数据 fresh-FD 的 `P10=-0.932` 和 norm `0.425--0.486` 不能再归因于基础表达能力或实现链路
   根本失效。
2. B4 多任务目标没有形成阻止方向拟合的硬冲突，但存在轻度幅值拉低：norm median 约
   `0.96`，最小约 `0.82`，component RMSE `0.446--0.472`，明显高于 gradient-only 的约
   `0.024`。这与 bank 不是主要坍缩原因、但共享多任务会带来有限拉扯的既有判断一致；其量级
   不足以解释跨 episode 的 180° 方向反转。

因此后续路由不改 Actor、不先换网络，也不再优先排查“能否拟合”。继续执行 §11.15.1 的
零 rollout 密度/clipping/regime 归因，再构造 hard-state manifest，区分数据稀疏、表征别名和
场景过渡；tiny-set PASS 不等于 untouched-episode 泛化 PASS。

### 11.17 B-1 后的 full-data hard-state 零 rollout 归因（2026-08-13）

已将 review 给出的诊断树与当前实现对齐并执行下一步。需要保留的边界是：B-1 排除的是小集合
基础记忆能力，不排除全数据采样/优化动态；action 坐标和 autograd 主要由 §11.10 的 Jacobian
审计排除；当前 Critic 已经显式输出 16D gradient，因此后续不再把“增加 explicit gradient
head”列为修复项。实现、产物与独立验证为：

```text
scripts/model_verify/analyze_mppi_local_critic_hard_states.py
scripts/model_verify/validate_mppi_local_critic_hard_states.py
outputs/mppi_proposal/direct_local_critic_hard_state_attribution_20260813_v1/
```

脚本重建 frozen Actor 与 B4 三个 Critic，只使用 4,590 个 internal-fit train context 和 600 个
已消费 internal-selection fresh-FD context；新增 DBM rollout 为 0，formal validation/test 未加载，
Actor/Critic 均未更新。输出包含 225 帧 `ensemble cosine < 0` 的 hard-state manifest、16维分量
分解、raw/Actor/Critic 三空间KNN、density×clipping×speed×scenario、同物理状态repeat和相邻
物理快照过渡。独立 validator 复算 hard mask、manifest、top-20、component alignment 与全部源
哈希并 `PASS`。

#### 11.17.1 不是低密度、clipping 或单一 regime 问题

| 表征空间 | hard/nonhard top-1 density median | 四个 density quartile 的 hard fraction |
| --- | ---: | ---: |
| raw equal-block | 0.696 / 0.710 | 0.367 / 0.427 / 0.367 / 0.340 |
| frozen Actor encoder | 0.896 / 0.886 | 0.340 / 0.333 / 0.447 / 0.380 |
| trained Critic representation | 0.941 / 0.944 | 0.360 / 0.400 / 0.400 / 0.340 |

hard帧没有集中到任一空间的低密度四分位；最高密度区也仍有约34% hard，因此“不够自由探索/
完全无近邻”不是主解释。fresh probe clipping 只有 `33/600` 帧，其中 `16` 帧 hard，只能解释
225 帧中的7.1%；各速度×场景小格仍普遍出现hard，不能路由到单一高速或recovery类别。

scenario标签在每个episode内固定，不能直接声称发生scenario-transition。数据实际是300个物理
快照、每个两个first-pass repeat。同状态repeat的真实梯度 cosine 中位约0.960，但存在明显尾部
（P10 `-0.916`）；hard/nonhard中位只差 `-0.0095`。相邻物理快照梯度本身变化很快，hard/nonhard
中位为 `-0.307/-0.270`；定义 adjacent cosine `<0.5` 的高转变帧 hard fraction 为38.8%，低转变
帧为31.9%，只有次要富集。Actor action相邻变化中位几乎相同。因此 transition 是难度来源之一，
但不是225帧反转的单独开关。

#### 11.17.2 错误集中在早期 steering，而不是16维均匀噪声

在 hard 帧的归一化 component alignment 中，steering贡献 **93.41%** 的负 alignment mass，
acceleration只有6.59%。steering总alignment中位为 `-0.811`，acceleration为 `-0.0014`。最严重
的是 steering knot 1/2/3（flat index 3/5/7），hard帧分量sign accuracy分别只有
`8.9%/5.3%/11.6%`；knot 0也只有13.8%。每帧16个分量中反向分量数中位为10。因此当前P10不是
一个弱小分量的偶然抖动，而是由对轨迹影响最大的前段转向梯度系统性翻转。

#### 11.17.3 Critic表示把“混杂邻域”放大成一致错误分支

| 空间 | hard top-20平均标签方向 cosine median | coherence median | top-20 best-label cosine median/P10 |
| --- | ---: | ---: | ---: |
| raw equal-block | -0.548 | 0.333 | 0.958 / 0.804 |
| frozen Actor encoder | -0.334 | 0.296 | 0.948 / 0.776 |
| trained Critic representation | **-0.846** | **0.800** | 0.614 / -0.667 |

raw与Actor空间说明正确分支通常存在，但同一局部邻域正反标签混杂；Critic训练后并未消除混杂，
反而把hard帧压入一个高度coherent的错误训练分支。对Actor top-20中的oracle-good和oracle-bad
邻居逐块比较 history/reference/current/anchor/feedback/gradient-context/actor-action，good更近
的比例都只有47.6%--52.4%，median差接近0；没有单一raw block能用普通距离解释分支选择。

严格解释边界：这仍不证明原始输入在信息论上不可分，equal-block/KNN距离也不是最优监督度量；
但证据已经同时反对“简单低覆盖”和“Critic只是随机噪声”。当前最符合数据的是：**局部
state-to-gradient映射有快速分支变化，普通回归/表征在全数据训练中把hard状态吸到占优的错误
steering分支**。

#### 11.17.4 下一步修正

下一轮仍冻结Actor，不新增随机episode，先使用现有train标签做受控训练对照：

1. 从本节manifest的正确/相反train近邻和train-side `Critic vs 0.05σ label` 误差构造hard-pair
   sampler，保证早期steering反向分支不会被普通batch比例淹没；
2. A组只改hard-example balanced sampling；B组在A组上增加state-gradient metric/contrastive
   consistency，使同方向pair靠近、相似但反向pair可分；保持现有显式16D head和Actor输入输出；
3. 同时报告前4个steering knots的sign/cosine，不只报告总cosine；checkpoint仍以fresh-FD
   median/P10/norm联合gate；
4. 若sampler即可改善，归因训练分布；若只有metric约束改善，归因表示分支；若两者均无改善，
   再围绕manifest状态新增相邻时间/恢复强度/曲率数据或审查缺失context。不得先随机扩集。

本节将 §11.15.1 的“待做零rollout归因”标为已完成，并把下一步从泛化的“定向补数据”收窄为
**先验证hard-pair训练与表示约束，数据新增作为失败后的下一层**。当前fresh-FD gate仍失败，
Actor更新继续不授权。

### 11.18 hard-state 归因 v2：标签自检、预注册多因路由与向量分解（2026-08-13）

在不删改 §11.17 原始结论的前提下，按后续 review 补齐四项容易影响归因解释的审计：先检查
hard 帧是否只是旧 `0.05σ` 标签尾部；将路由阈值预注册且允许一帧多因；把 component 审计扩展
为 norm/方向、主导分量符号、平行/垂直三层；显式区分物理快照序号与同快照两次 repeat。新版
实现沿用原脚本，产物另存为：

```text
outputs/mppi_proposal/direct_local_critic_hard_state_attribution_20260813_v2/
  analysis.json
  context_attribution_manifest.json
  hard_state_manifest.json
  hard_state_analysis.npz
  validation_summary.json
```

输入仍只含 4,590 个 internal-fit train context 与 600 个已消费 internal-selection context；Actor、
三个 B4 Critic 均冻结，新增 DBM rollout 为 0，formal validation/test 未加载。独立 validator 从
基础数组重建 13 个 route flag、物理序号/repeat、标签 cosine、KNN 和平行/垂直分解，结果 `PASS`。

#### 11.18.1 旧 `0.05σ` 标签不是 225 帧反转的主因

| 集合 | `0.05σ label` vs fresh-FD cosine median/P10 | cosine `<0.95` | cosine `<0.90` | cosine `<0` |
| --- | ---: | ---: | ---: | ---: |
| 全 600 帧 | 0.990 / 0.959 | 45 | 15 | 0 |
| 225 hard 帧 | 0.990 / 0.953 | 22 | 8 | 0 |

强标签不一致（cosine `<0.90`）虽然有轻微富集：带标记帧 hard rate 53.3%，不带标记为37.1%；
但它只覆盖 `8/225=3.6%` hard 帧，没有任何旧标签相对 fresh FD 发生方向反转。Critic cosine 与
标签自检 cosine 的 Pearson 仅 `0.079`。标签幅值相对 fresh FD 的 norm ratio 中位为 `0.832`
（hard 为 `0.836`），存在有限半径衰减，但同样无法解释 Critic 的大规模确定性反向。故路由为：
标签尾部是次要数据质量标记，不是主故障。

#### 11.18.2 预注册多因路由的定量结果

所有标记均为非互斥诊断分区，不作事后单因裁决。主要阈值固定为：label mismatch `<0.90`、
warning `[0.90,0.95)`、各空间 density percentile `<=10%`、top-20 coherence `<0.30` 为混杂、
Critic top-20 mean-label cosine `<-0.5` 且 coherence `>0.5` 为 coherent wrong branch、repeat/
adjacent cosine `<0.5` 为转变。逐帧 manifest 同时保存 KNN 分位、coherence、scenario、clip、
episode、物理快照序号、repeat 序号和 context 序号。

| 标记 | 覆盖 hard | 带标记 hard rate | 不带标记 hard rate | lift |
| --- | ---: | ---: | ---: | ---: |
| label mismatch | 8/225 | 53.3% | 37.1% | 1.44 |
| clipping | 16/225 | 48.5% | 36.9% | 1.32 |
| physical snapshot 0 | 55/225 | 45.8% | 35.4% | 1.29 |
| within-state repeat transition | 73/225 | 44.5% | 34.9% | 1.28 |
| adjacent-state transition | 188/225 | 38.8% | 31.9% | 1.22 |
| raw low density | 20/225 | 33.3% | 38.0% | 0.88 |
| Actor low density | 17/225 | 28.3% | 38.5% | 0.74 |
| **Critic coherent wrong branch** | **178/225** | **98.3%** | **11.2%** | **8.77** |

低密度标记反而不富集 hard，进一步否定“只需随机扩状态覆盖”。cold-start、repeat 与相邻状态
转变均有轻度富集，说明快速状态变化增加难度，但它们不是决定性开关；其中 adjacent 标记覆盖
80.7% 全部帧，本身区分度有限。唯一同时具有高覆盖与高条件失败率的标记是 Critic coherent
wrong branch：181 个带标记帧中 178 个为 hard，覆盖79.1%的 hard 集。剩余47个 hard 帧的
ensemble cosine 中位约 `-0.317`，更偏边界/混杂错误；其中 Actor/raw mixed-neighbor 分别覆盖
28/23帧，不应强行并入同一原因。

该标记依赖 fresh 标签，只能用于离线机制归因，不能当部署 gate；但它定量确认了 §11.17 的
观察：不是 Critic 表示内仍然随机混杂，而是其表示把大多数错误帧压入标签高度一致、方向却与
当前状态真实梯度相反的训练邻域。

#### 11.18.3 反转是主导方向的 coherent flip，不是个别 action 维噪声

对 hard 帧，预测/真实 norm ratio 中位为 `0.437`。沿真实梯度轴的投影系数

```text
alpha_parallel = dot(g_pred, g_true) / ||g_true||^2
```

中位为 `-0.343`，225/225 均为负；垂直残差相对真实梯度 norm 的中位仅 `0.216`。这表明主体
是沿真实轴反向，而不是由巨大正交噪声把 cosine 偶然推负。按真实梯度绝对值排序：

- 最大一个分量在 hard 帧中 94.7% 符号反转；
- top-3 主导分量有 82.7% 的 hard 帧全部反转，平均符号正确率只有8.3%；
- top-5 主导分量有 63.1% 的 hard 帧全部反转，平均符号正确率只有12.2%。

结合 steering 占93.41%负 alignment mass，可将故障更精确地表述为：**Critic 在多状态训练中
对主导早期 steering 响应选择了错误且较一致的分支，同时幅值仍偏低；不是16维均匀随机误差。**

#### 11.18.4 episode 序号修正与最终路由

每个 episode 含5个物理快照、每个快照两个 first-pass repeat。manifest 现分别记录
`physical_snapshot_ordinal=0..4`、`repeat_index=0/1`、`context_ordinal=0..9`。物理序号0的
hard rate为45.8%，随后4个序号为37.5/36.7/35.0/32.5%，存在 cold-start 富集；但 repeat 0/1
的 hard rate为35%/40%，说明旧“首行”统计不能独立解释成 cold-start，必须保留两种序号。

归因阶段至此完成。下一步不立即新增随机数据，也不改 Actor/动作维度：固定同一标签和网络，
依次比较 B4、hard-example balanced sampler、balanced + state-gradient metric/contrastive loss。
若 balanced 即改善 fresh-FD P10/early-steering sign，主因是训练采样被多数分支淹没；若只有
metric 改善，主因是表示几何；若二者均失败，才根据 v2 manifest 定向补 neighboring-time、
recovery-strength、curvature/context 信息。Actor 在 fresh-FD median `>=0.70`、P10 `>=0`、
norm ratio `>=0.50` 的既定 gate 通过前继续冻结。

#### 11.18.5 Qoder 复核补录：coherent-branch 标记的循环性边界 + 两个零成本遗留检查（2026-08-13）

§11.18 的四项审计（标签自检、预注册多因路由、三层向量分解、双序号修正）全部复核通过，
§11.18.1 尤其干净地排除了标签侧主因（hard 帧无标签反转、Pearson 仅 0.079）。补充两点：

1. **`Critic coherent wrong branch` 标记的 8.77 lift 含循环成分，解释时要打折。**
   该标记用 Critic 自身表征空间定义邻域，而 Critic 表征被训练的目标就是预测 g：预测错误且
   一致的状态在表征空间必然 coherent-wrong。换言之"178/225 落入 coherent wrong branch"部分
   是"预测错了"的另一种说法，而非独立的病因证据。 §11.18.2 已声明其只能用于离线归因，此处
   进一步明确：**表征病因的有效证据只有 raw/Actor 空间的"密度相同 + coherence 0.30--0.33"**
   （非循环），Critic 空间行只作描述。这不改变结论方向，只修正证据权重。
2. **两个零成本遗留检查，建议排在训练对照之前或并行。**
   - **repeat-pair raw-block diff**：300 对同快照 repeat 中 cosine<0 的对（P10 `-0.916` 的
     尾部），逐 raw block（history/reference/current/anchor/feedback）做差异分解，定位两条
     repeat 之间究竟什么字段不同、该字段是否进入 encoder 输入。若判别字段不在输入里，则
     §11.18.4 的训练对照无论结果如何都救不了，应先补输入；这一步能直接决定训练 A/B 是否
     有意义，成本为零（数据已在 v2 产物中）。
   - **steering×曲率符号联合分布**：负 alignment 质量 93.41% 在前段 steering、主导分量
     94.7% 符号反转。用现有标签统计 hard 帧 steering 梯度符号与参考轨迹曲率符号的联合
     分布：若左转/右转参考的 steering 梯度相反而表征把两者合并，根因即曲率符号条件化缺失，
     修复（曲率符号特征或分侧条件化）远比泛化修表征便宜。

### 11.19 角度 wrap 假设评估与修正后的下一步顺序（2026-08-13）

针对"梯度错误集中在少量样本，疑似角度跨越 ±π/2π 所致"的假设，做了代码级预检。三种情形
中，情形 1（DBM/reward 自身 wrap 错误）已被 fresh FD 跨半径 cosine≈1.0 与 cost 内
`_wrapped_angle_difference` 双重排除。情形 2/3（Critic 输入角度未 wrap / 序列未 unwrap）
经全输入链路核查后**基本排除**：

| 输入块 | 角度处理 | 位置 |
| --- | --- | --- |
| history（250×7） | 全部为 body 系增量（dx, dy, dyaw, Δvx, Δvy, action×2）；dyaw 用 `atan2(sin,cos)` 显式 wrap | `query_deployment.py` append |
| reference（50×5） | heading 用 sin/cos 双通道编码 | `mppi_proposal_policy.py` `ego_reference_features` |
| current / anchor / feedback / gradient ctx | vx,vy,action / 8×2 knots / 74 维动作空间统计 / 16 维梯度统计，**不含角度** | — |

即输入在构造时已 wrap-clean，朴素形式的 wrap 假设不成立。数据级审计仍可留作确认项，但
预期结论是"编码无问题"。

**wrap 直觉的更精确存活变体：anchor 移动。** 同快照两个 first-pass repeat 唯一不同的是
feedback 向量 → anchor 不同 → 站在同一 `J(s,a)` 曲面的不同位置。repeat 梯度 P10 `-0.916`
可能主要是 **anchor 移动导致的合法梯度差**，而非输入编码缺陷。注意 anchor 本身是 Critic
输入，信息上可分；但 §11.17/§11.18 的 KNN 在表征空间进行，若表征不区分 anchor 位置，
"混杂邻域"混入的可能正是 anchor 不同的样本。这把 G2 的 anchor 混淆问题以新形式带回。

**修正后的下一步顺序：**

1. **第 0 步（零 rollout，先于一切训练）**：repeat-pair 分解——`Δg_true` 分别对
   Δanchor、Δfeedback、Δhistory、Δreference 回归/相关。Δanchor 解释力强则根因是
   "梯度对 anchor 敏感 + 表征不区分 anchor"，修法为 anchor-aware 采样/训练而非表征手术；
   均不解释才轮到表征/函数分支。并行做 §11.18.5 的 steering×曲率符号联合分布。
2. **第 1 步（训练对照，§11.18.4 已定）**：B4 基线 vs hard-balanced sampler vs
   +metric/contrastive，同标签同网络，fresh-FD 联合 gate 验收。
3. **第 2 步（前两步失败才做）**：按 v2 manifest 定向补数据或补输入特征（曲率符号、
   anchor 相对量）。

**预判需提前写清**：若第 0 步显示真梯度在 anchor/参考几何的微扰下本身就翻转，则这是
**函数分支问题而非学习问题**——确定性回归在分支边界附近必然平均两分支。此时正解是按判别
变量（anchor 位置、曲率符号）条件化，或接受分支边界附近的误差下限，而非继续加压训练。
当前 gate（median 0.613 / P10 -0.932 / norm 0.486）三项全未过，Actor 继续冻结。

### 11.20 repeat-pair action-location 审计与跨-anchor训练对照（2026-08-13）

按 §11.19 的第0步完成了零 rollout 审计，并补做了受控 Critic 训练。实现与独立复算入口为：

```text
scripts/model_verify/analyze_mppi_local_critic_anchor_conditioning.py
scripts/model_verify/validate_mppi_local_critic_anchor_conditioning.py
outputs/mppi_proposal/direct_local_critic_anchor_conditioning_*_20260813_v*/
```

审计使用600个已消费 selection context（300个相同物理快照repeat pair）及既有 fresh-FD 标签；
没有新增DBM rollout，formal validation/test未加载，Actor始终冻结。validator检查全部输入/检查点
SHA256，重算pair、方向导数、action-response和fresh target指标，所有产物完整性均 `PASS`；注意该
`PASS`只表示复算一致，机制 qualification 仍为 `ACTION_LOCATION_CONDITIONING_FAIL`。

#### 11.20.1 review假设的保留与修正

角度编码数据级确认无异常：history dyaw范围 `[-0.114,0.114] rad`、`|dyaw|>3` 为0，reference
sin/cos单位圆最大误差 `3.16e-7`。repeat pair内 initial state、current action、direct reference、
history/reference/current输入逐位相同。变化的不只是feedback：first-pass链同时改变alpha/guided
anchor、74D feedback、16D gradient context；alpha center和最终Actor center的repeat差异中位分别
为 `0.1477σ/0.1480σ`，而Actor residual action差异中位仅 `5.50e-4`。

真实repeat梯度 cosine 中位 `0.960`、P10 `-0.916`，70/300对发生反转。对这70对，沿两个center
连线的方向导数均100%从左端正变为右端负。这不是reward粗糙或函数不连续，而是平滑局部cost
曲面跨过极小值后，梯度在不同action location合法反转。因此不能把同状态repeat当作
“same-state same-gradient”正样本；正确监督是固定物理状态，在两个绝对center分别拟合
`g(s,a0)`与`g(s,a1)`。

#### 11.20.2 训练对照结果

所有行复用同一标签、Actor、episode切分和fresh-FD审计。`strong delta`仅跑1 seed，属于机制
pilot；其余普通跨-anchor与absolute-center为3 seed。表中same-center为两种context在同一绝对
center的prediction cosine P10（左右两端）：

| arm | fresh cosine median / P10 | norm ratio median | 70个反转召回 L/R | same-center P10 L/R |
| --- | ---: | ---: | ---: | ---: |
| B4 baseline | 0.613 / -0.932 | 0.425 | 0% / 0% | 0.288 / 0.302 |
| 普通cross-anchor loss | 0.615 / -0.920 | 0.395 | 0% / 0% | 0.725 / 0.694 |
| 显式absolute-center action输入 | 0.635 / -0.931 | 0.401 | 0% / 0% | 0.832 / 0.834 |
| cross-anchor delta×10，关闭bank | **0.721** / -0.929 | **0.473** | **27.1% / 32.9%** | -0.535 / -0.244 |
| 上行 + same-center invariance×5 | 0.666 / -0.939 | 0.345 | 4.3% / 4.3% | 0.637 / 0.641 |

结论分三层：

1. 只追加 paired target 或把action encoder输入换成绝对center仍会收敛到近似state-only平均解；
   两者的跨center prediction cosine约0.994--1.000，反转召回为0。故问题不只是“坐标没给”。
2. 强化 `Δg_pred ≈ Δg_FD` 后，fresh median首次超过0.70且反转召回达到约30%，证明网络和
   数据确实包含可学习的action-response；但P10完全未改善，同绝对center对feedback的不变性
   尾部恶化，说明模型开始利用context nuisance拟合响应。
3. 大权重invariance会把网络重新推回action-invariant平均解，不能靠两个loss权重相互对冲。
   当前最核心的剩余问题是**state-action interaction没有结构保证**，而不是继续随机扩状态、
   角度wrap、单纯换action坐标或盲调更大loss。

#### 11.20.3 修正后的下一步

停止继续扫cross-anchor/invariance权重。下一轮仍冻结Actor，优先做结构化局部响应Critic：利用
G0已验证的小半径二次性，将梯度写成

```text
g(s,a) = g0(s) + H(s) · (a - a_ref)
```

先用受约束的diagonal/low-rank `H(s)` 做机制pilot，使action变化必然经过显式响应通道；训练同时
监督端点梯度和 `H·Δa = Δg_FD`，same-center consistency只作gate而不是大权重主loss。验收继续看
fresh median/P10/norm、70对反转召回、same-center P10和前4个steering knot sign。若结构化响应
仍不能改善P10，再回到hard-state manifest定向补相邻action location，而不是新增无关随机状态。
Actor在P10>=0、norm>=0.5及反转/不变性gate通过前不更新。

### 11.21 Structured Local-Q Critic pilot 契约（2026-08-14）

§11.20.3 的下一步已固化为独立 pilot 文档：

```text
car_foundation/docs/mppi_structured_local_q_critic_pilot_20260814.md
```

最终采用标量局部二次 Q，而不是只参数化孤立梯度：

```text
delta = a - a_ref
Q(s,a) = Q0(s) + g0(s)^T delta + 0.5 * delta^T H(s) delta
grad_a Q = g0(s) + H(s) delta
```

`H` 由 signed diagonal 与 signed low-rank 对称项构造，必须允许负特征值；普通action encoder
从pilot移除。容量对照固定为同loss的H0、D、D+R1、D+R2，各3 seed。repeat chord按物理sigma距离做截断反距离
加权，`<=0.15σ` 的152对/33个反转作为主机制子集；pair midpoint只允许作为不可部署oracle上界，
主模型参考点仍为在线可得的确定性Actor center。same-center consistency只作gate，不再用大权重
把响应压回平均解。

该参数化保证SAC仍接收标量Q，同时由同一Q自动得到可积的action gradient，并能显式审计H对称
与autograd一致性。完整loss、弦长分层、可辨识性限制、3-seed报告和Actor冻结门槛以独立pilot
文档为准；在fresh P10、norm、小弦长反转召回与same-center gate联合通过前，不授权Actor更新。

补充预注册：arm必须至少2/3 seed逐模型完整通过，剩余seed相对同seed H0不得显著退化；ensemble
不参与qualification。小弦长33个反转只作机制gate，中长弦外推失败单列诊断，部署前主守卫仍是
600-context fresh P10。chord权重超参数全部落盘；norm median恢复`[0.5,2.0]`双边gate并报告
P10/P90、log-error P90与过小/过大比例。

### 11.22 Structured Local-Q 全量结果（2026-08-14）

共享模块、H0/D/D+R1/D+R2同loss训练器、结构单测和独立summary审计均已实现。结构单测确认H
严格对称、允许负特征值，且`autograd(dQ/da)==g0+H*delta`；32-pair反转平衡tiny-set上D+R2
可达到own-gradient cosine median 0.970、norm ratio 0.912、cross-target cosine 0.994和反转召回
0.563，排除了基本表达、坐标、解析链路和优化器完全失效。

全量4590 train / 1110 internal-validation / 600 fresh-FD、四臂各3 seed的结果为：H0、D、
D+R1、D+R2均为0/3完整gate通过。D+R2最好，但fresh median cosine仅为
0.683至0.721，P10仍为-0.950至-0.908，norm median为0.459至0.567，小弦长反转召回仅
0.152至0.288，same-center P10为-0.626至0.036。相对同seed H0，D+R2的median cosine配对
中位增量只有+0.009，P10配对中位增量只有+0.003；反转响应虽然从0开始出现，却没有达到0.50
机制门槛。

产物位于：

```text
outputs/mppi_proposal/structured_local_q_full_20260814_v1/summary.json
outputs/mppi_proposal/structured_local_q_full_20260814_v1/analysis.json
```

独立审计确认输入/checkpoint SHA256全部匹配、gate mismatch为0，且Actor未更新。由于没有arm先
达到2/3完整gate，剩余seed的paired-bootstrap非劣条件不触发。正式结论保持
`STRUCTURED_LOCAL_Q_CRITIC_FAIL_ACTOR_FROZEN`。

这轮证明结构响应通道有效但不充分：D+R2学出了负曲率并召回部分反转，但没有修复fresh尾部。
tiny-set能拟合而full-data失败，更支持“每个state只有单条repeat chord、16维action-location响应
欠约束/条件化混叠”，不支持继续把问题归为解析Q公式或网络基本容量。下一步仍冻结Actor，先基于
hard-state manifest做同状态、多小半径、方向独立的局部探针设计；在新增rollout前先审计现有标签
方向矩阵的有效秩和条件数，量化每个state还缺多少局部方向。完整表格和执行参数见structured
local-Q pilot文档§8。

### 11.23 Chord geometry / semantic / Local-H oracle审计（2026-08-14）

§11.22建议的零rollout验证已经完成并独立复算。权威产物为：

```text
outputs/mppi_proposal/chord_geometry_local_oracle_20260814_v3/analysis.json
outputs/mppi_proposal/chord_geometry_local_oracle_20260814_v3/validation_summary.json
```

输入语义检查确认history/reference/current在repeat内最大差异为0；guided anchor、feedback、gradient
context以及alpha/Actor center均随first pass变化。当前Structured Critic的`local_parameters()`没有
显式absolute Actor center，却用这些变化量输出`Q0/g0/H`，因此存在“从nuisance猜action location”
的语义混杂。后续必须分离repeat-invariant physical state、canonical origin和evaluation action，
但现在不把这一改动单独解释成尾部修复。

Geometry结论是“exact-state不足、KNN pooled表面满秩”：每物理状态只有一条chord，rank最多1；
K=32跨状态邻域numeric rank均为16、condition中位8.98，但stable/entropy effective rank中位仅
4.19/9.98。前3个steering knot仅占方向能量均值12.9%、中位8.0%，P10 1.07%，确有定向补充价值。

Physical-only KNN symmetric-H oracle在internal-validation小弦长通过（cross median/P10
0.984/0.595、反转召回70.3%），但原配置到300-pair heldout fresh-FD后，小弦长结果为
`Delta_g` cosine 0.968/0.454、cross-target 0.986/-0.167、反转召回47.0%；全弦长P10 -0.249、
召回41.4%。unconstrained H没有改善heldout，diagonal明显更差，说明对称约束不是主因、跨通道
耦合必要。独立validator的hash/count/gate全部一致，最大指标误差2.98e-8。

因此路由为`LOCAL_H_ORACLE_FAIL_EXACT_STATE_RANK1_TARGETED_PROBES_JUSTIFIED`：不再新增无关随机
state，也不先盲训Actor；为hard states加同状态、`<=0.15σ`、方向独立的DBM probes，重点front
steering knot 0--2并保留全16维正交覆盖。同时实现semantic-clean Critic输入。Actor保持冻结，
直到新数据上的oracle heldout与Critic fresh-tail联合通过。完整表格、方法和限制见structured
local-Q pilot §9。

#### 11.22.1 复核补充：g0/H 指标分解与 P10 尾部不变性（2026-08-14）

对全量产物逐 seed 复核（四臂表、负曲率、2/3 seed 幅值不足、2/3 seed same-center P10 为负、
15 项 SHA256、gate mismatch 0、paired 增量 +0.009/+0.003 均与产物一致）后，补充四条未写入
结论的结构性观察：

1. **结构化形式把两个残留问题干净分开**。在 `a=a_ref` 处 `grad Q = g0(s) + H·0 = g0(s)`，
   因此 fresh median/P10/norm 只考核 g0，反转召回/response 只考核 H。本轮表里二者互不混淆，
   这是该参数化即使失败也保留下来审计价值的原因。
2. **fresh P10 尾部在全部历史干预下从未移动**。B1--B5、absolute-center、strong-delta、
   结构化 Q 四臂的 P10 全部落在 `-0.92 ... -0.95` 区间。该尾部是 g0(s) 在约10%硬状态上的
   状态侧问题，任何 action 侧修改在构造上就无法触及；新一轮干预若仍以 action 通道为对象，
   不应预期 P10 改善。
3. **"结构有效"应精确为">=2 个耦合方向有效"**。H0（0.674）相对 B4（0.613）的 +0.06 来自
   新监督组合本身（value+probe+norm 校准）；D（0.527）低于 H0，说明欠辨识的纯对角响应对 g0
   是净伤害（借 encoder 容量与 nuisance 拟合），rank>=2 才翻正。
4. **小弦长没有召回优势**。D+R2 中弦召回（最高 0.375）不低于小弦长（0.152--0.288）。若瓶颈
   是二次性外推，小弦长应显著占优；实际分布支持瓶颈是每状态 chord 数量（可辨识性），而非
   半径纪律失效，与"补多 chord 数据"的补救方向一致。

另需登记两个 caveat：D+R2 仅 seed0 同时满足"有响应且 same-center 干净"（P10 0.036），
seed1/2 仍为 -0.626/-0.304——结构保证 action 只能经 H 起作用，但不保证 H 从正确的变化中
估计，state encoder 通往 H 的路径仍对 nuisance 敏感；response cosine 中位 0.97 与召回 0.17
并存是因为多数 pair 真实响应接近零，不应解读为响应已学好。

下一步在 §11.22 既定计划（hard-state manifest 多 chord 探针 + 标签方向矩阵有效秩/条件数
审计，覆盖 H 侧）上补一项零成本检查：对 g0 尾部 225 个硬状态执行 §11.15.1 的邻域离散度
可学习性上界。若原始输入邻域内真梯度本身离散，则 P10 尾部是输入信息受限，补 action-location
数据无法修复 g0，应改表示（history 窗口/reference 特征）；若邻域可分辨，再按 manifest 采集
`<=0.15σ`、2--3 条方向独立 chord，同时喂给 H 辨识与 g0 尾部。两项审计均在新增 rollout 之前
完成，Actor 继续冻结。

### 11.24 g0 邻域离散度与 label-aware oracle 审计（2026-08-14）

§11.22.1要求的g0零rollout审计已经完成，权威产物为：

```text
outputs/mppi_proposal/g0_learnability_audit_20260814_v1/analysis.json
outputs/mppi_proposal/g0_learnability_audit_20260814_v1/g0_priority_manifest.json
outputs/mppi_proposal/g0_learnability_audit_20260814_v1/validation_summary.json
```

审计只用4590 train和1110 internal-validation选择KNN配置，再用合并后的5700-context bank评价已经
消费的600-context fresh-FD；没有新增rollout，没有使用formal validation/test，Actor保持冻结。
physical-only只含history/reference/current；action-aware额外显式加入absolute Actor center，并排除
alpha、feedback和gradient context。独立validator重算指标、九项输入hash、split、manifest和路由，
最大误差`2.98e-8`，结果PASS。

当前B4 ensemble的fresh cosine median/P10为0.613/-0.932；physical-only KNN为
0.488/-0.934，action-aware KNN为0.593/-0.919。absolute action明确改善physical-only，但固定
邻域聚合既没有超过当前Critic中位数，也没有修复负尾。top-20 action-aware邻域的标签coherence
中位只有0.351、pairwise cosine中位0.077；然而用目标fresh-FD标签事后选最匹配邻居时，
oracle-best cosine median/P10为0.969/0.852，600/600均为正。当前225个hard context也全部存在
正向top-20邻居，而固定KNN在其中仅35.6%为正。

这组结果支持`G0_INFORMATION_PRESENT_BUT_FIXED_NEIGHBOR_RULE_FAILS_TAIL`：有用g0方向在局部bank中
存在，但相似邻域混有冲突分支，当前表示/距离/平均规则无法选择正确分支。oracle-best使用目标标签，
不可部署，也不是严格上界；它只能量化信息headroom，不能直接宣称数据已经充分。下一步g0侧先做
semantic-clean的`physical state + explicit absolute action`条件化重训，并以cold-start、clipped和
2.0--2.8 m/s切片作为重点诊断。

该结论不推翻§11.23的H侧采集授权。Local-H oracle小弦长反转召回47.0%明显高于同数据D+R2的
约19.7%，但oracle自身仍不过gate；因此“现有数据对H不足”和“网络没榨干已有信息”同时成立。
action-aware KNN负向227帧、H-oracle cross负向79帧，联合失败34帧。下一轮同状态多方向probe把
这34帧作为最高优先级stratum，但必须同时纳入其余H/rank hard帧和matched easy controls。半径按
完整chord `<=0.15σ`定义：若配置采用单侧半径，则主范围是`0.05--0.075σ`。Actor在fresh g0 tail
与Local-H heldout联合gate通过前继续冻结。完整方法、表格和限制见structured local-Q pilot §10。

### 11.25 Review修订：完整门槛、并行执行与互斥分层（2026-08-14）

采纳本轮review的三项修订。第一，第三阶段补回中位数门：Critic fresh cosine必须同时满足
median `>=0.70`和P10 `>=0`，不能用尾部单项通过掩盖主体退化；Local-H heldout oracle
cross-target median必须`>=0.90`。二者还需与norm ratio `[0.5,2.0]`、小弦长反转召回
`>=0.50`、same-center P10非负和至少2/3 seed完整通过联合。g0 label-aware oracle不可部署，
不属于该oracle门。

第二，semantic-clean零rollout重训与定向DBM probe采集并行启动；新probe Structured
Local-Q重训等待两轨都完成。第一轨固定三臂：physical+absolute-action、再加feedback、再加
中等same-center invariance。invariance只允许比较同physical state、同absolute action，避免
再次把合法action response压成平均解。

第三，采集manifest改成互斥分层。34个g0/H联合失败中只有20个当前Critic-hard，另14个
Critic当前方向为正，报告中不得把34个都称为Critic尾部。互斥计数为：联合失败34、仅H失败45、
仅KNN-negative且Critic-hard 125、仅KNN-negative且Critic正常68、g0/H均正常但Critic-hard
66、三者均正常262，合计严格为600。前两层优先生成H probes；跨层诊断集合
`Critic-hard AND KNN-positive=80`由H-only中的14帧和上述66帧组成，仍是Track A核心验收集，
但不能再次作为独立采集层计数。262帧用于matched easy controls。旧的`34/79/227/145`
均是重叠集合，禁止按该顺序重复采集。

Actor继续冻结，formal validation/test保持封存。完整执行契约见structured local-Q pilot §11。

### 11.26 执行结果：三轨完成但联合门未通过（2026-08-14）

§11.25的三轨计划已经完整执行，且没有删除或改写此前审计结论。

Track A完成semantic-clean三臂各3 seed。显式absolute action输入与gradient-context移除均通过结构
验证，但PA/PAF/PAF_INV全部0/3 gate通过。跨seed中位fresh cosine分别为0.659/0.687/0.696，
P10为-0.958/-0.947/-0.950，小弦反转召回为0.212/0.182/0.182。feedback和same-center
invariance没有稳定修复尾部，Actor继续冻结。产物：

```text
outputs/mppi_proposal/semantic_structured_local_q_20260814_v1/summary.json
outputs/mppi_proposal/semantic_structured_local_q_20260814_v1/analysis.json
```

Track B完成互斥manifest、100状态pilot和独立validator。选择79个H失败repeat context及21个matched-easy
controls；79个target只对应54个独立物理状态（25个状态有两条repeat、29个一条），采集、训练与
cross-validation都必须按`(episode, physical_snapshot_ordinal)`口径计数。每context 19个方向、
`0.05/0.075 sigma`单侧半径、77个outer location，每location再做33点`0.01 sigma` full-16D FD，
共254,100个确定性forward DBM cost rollout，产出6,083（79×77）个location标签。outer rank最小16、
stable rank中位9.50、condition number中位1.41/最大2.43，outer裁剪0.36%。独立复算重建全部几何、
normalized action、gradient/curvature并重放1,024个cost，误差均为0（`dbm_cost_replay`）；另报的
actor center replay误差5.96e-8是不同量，不得混用。产物：

```text
outputs/mppi_proposal/targeted_local_probe_manifest_20260814_v1/validation_summary.json
outputs/mppi_proposal/targeted_local_response_labels_20260814_v1/summary.json
outputs/mppi_proposal/targeted_local_response_labels_20260814_v1/validation_summary.json
```

Track C在两轨通过工程validator后启动，只用79个target context（54个独立物理状态）训练PA，21个matched
control保持未见。三seed仍为0/3 gate通过。剔除相关episode后，真正未接触的完整episode子集只剩230帧
（不是原600帧分布），其跨seed中位fresh cosine 0.695、P10 -0.929、norm ratio 0.766、小弦反转召回
0.091；负尾没有被修复。针对完全相同的targeted-response数据，与重训前PA逐seed配对后，79个seen
target的cosine median/P10/反转召回中位增量为`+0.663/+0.415/+0.152`，说明定向标签能被学习；但21个
unseen matched control的对应增量为`-0.100/+0.066/-0.032`，不支持跨状态泛化已经改善。同一230帧子集上的旧PA/新PA逐seed
配对复算已补做（23个完整episode，6个checkpoint hash全部通过）：cosine median逐seed增量
`+0.003/-0.056/-0.146`（中位-0.056），P10增量`+0.013/+0.037/+0.014`（中位+0.014，幅度不具
实际意义），反转召回与小弦召回增量均`<=0`。即定向重训在真正未接触状态上没有median改善、尾部
略有变动但远不够gate，"未接触状态无改善"结论正式成立。产物与独立hash复核：

```text
outputs/mppi_proposal/targeted_semantic_structured_local_q_20260814_v1/summary.json
outputs/mppi_proposal/targeted_semantic_structured_local_q_20260814_v1/analysis.json
outputs/mppi_proposal/targeted_semantic_structured_local_q_20260814_v1/targeted_response_comparison.json
outputs/mppi_proposal/targeted_semantic_structured_local_q_20260814_v1/unseen230_paired_comparison.json
```

最终结论是`TARGETED_STRUCTURED_LOCAL_Q_CRITIC_FAIL_ACTOR_FROZEN`。此前"exact-state rank-1需要
多方向probe"的诊断成立，因为seen hard状态显著改善；但这不是完整主因，state-to-gradient/H的
跨状态条件化仍是阻塞。当前54个独立物理状态（79个repeat context）×77 location的训练还造成
state-location过采样，可能强化局部记忆。下一步先做零rollout的state-balanced sampling及hard-state
grouped cross-validation，预注册规则：5-fold按`(episode, physical_snapshot_ordinal)`把54个独立
物理状态分组，同一state的全部repeat与77个location必须留在同一fold，按joint/H-only/车速/场景/
裁剪比例/每状态repeat数分层；每epoch按state→repeat center→少量location采样，state-level loss
`L_target = (1/|S|)Σ_s (1/|A_s|)Σ_a L(s,a)`；checkpoint只由原internal-validation决定，禁止用
heldout fold选epoch；每fold跑3 seed并逐fold报告。通过条件：out-of-fold cosine median `>=0.70`、
P10 `>=0`、norm ratio `[0.5,2.0]`、小弦反转召回 `>=0.50`、至少4/5 fold通过cosine-median与norm、
21个matched control相对旧PA无显著退化、且至少2/3 seed完整通过。只有heldout hard-state fold显示
迁移后，才按互斥manifest扩展更多状态；否则先修state representation/regime条件化。Actor、formal
validation和test保持冻结。

补充：四个口径修正已与artifact核对一致并正式关闭（254,100 rollout→6,083标签；79 target=54个独立
物理状态；57.0%训练mix集中于54状态；`dbm_cost_replay=0`与actor center replay 5.96e-8为不同量）。
另一个重要推论：79个target全部来自H_ONLY 45 + JOINT_G0_H 34，125个G0_ONLY_CRITIC_HARD context
完全没有获得定向probe。因此这批数据只验证局部响应/H是否可迁移，不能单独判定完整g0尾部是否可学；
fresh P10没有明显改善并不意外，因为大量G0-only状态本就不在probe范围内。grouped CV必须按54个
物理状态分组，并分别报告H_ONLY与JOINT_G0_H两个子层：heldout fold失败可定位为state→局部响应的
表示/泛化问题；即使通过，也只授权扩大H-response状态覆盖，不能直接宣称g0尾部已经解决。

### 11.27 执行结果：grouped CV决定性失败，阻塞定位到g0耦合状态（2026-08-14）

§11.26末尾预注册的state-balanced grouped cross-validation已按原规则完整执行。54个独立物理状态
按`(episode, physical_snapshot_ordinal)`分5-fold，贪心均衡（stratum/scenario/repeat数），fold规模
11/11/11/11/10状态、16/16/16/17/14 context；23个状态的两条repeat分属不同stratum，fold均衡按
state级worst-case（含JOINT即计JOINT），报告仍按context级分层。每epoch按state→repeat center→
8个location采样，state-level loss `(1/|S|)Σ_s(1/|A_s|)Σ_a L(s,a)`；checkpoint只由原
internal-validation决定；每fold 3 seed。control退化门用严格20个state-unseen control——21个中
context 6018与target 6019共享物理状态，被排除出严格集（该口径同样适用于§11.26 Track C的
"21 control未见"表述：严格说是20个state级未见）。运行规则：median paired delta
`>=-0.02`（cosine median与P10）。

结果：15个fold-seed全部失败，`0/5` fold通过cosine-median+norm（0/15个fold-seed达到
cosine `>=0.70`），`0/3` seed完整通过。out-of-fold cosine median范围`-0.063~0.691`，P10全部
`<=-0.917`，反转召回0.02~0.46全部低于0.50。分层结果（各fold-median按location数加权的均值，非
严格pooled）：H_ONLY `0.616`、JOINT_G0_H `-0.541`；逐run分布为H_ONLY 15个run中13个为正
（fold-median的中位数0.790），JOINT 15个中12个为负（中位数-0.704）。独立validator重载全部15个
checkpoint、拼接真实OOF预测后的严格pooled口径：全体18,249个location median `0.332`/P10 `-0.961`，
分层H_ONLY `0.765/-0.918`、JOINT_G0_H `-0.680/-0.973`。分层差异本身非常强且两种口径方向一致：
JOINT状态上heldout绝对梯度系统性反向。control退化同样失败：严格20个control的median paired
delta为cosine median `-0.100`、P10 `-0.074`（全集21个为`-0.095/-0.065`）。未接触230帧子集跨run
中位cosine 0.734、P10 -0.925、召回0.107，与此前一致。best_epoch中位120（82~159），早停正常触发。

本CV决定性确认的是：当前数据+当前表示/条件化+当前训练方法，无法学到稳定的跨物理状态绝对局部梯度
映射。采纳review修订，qualification改为
`GROUPED_CV_LOCAL_GRADIENT_TRANSFER_FAIL_REPRESENTATION_OR_STATE_ACTION_CONDITIONING`
（run summary artifact保留原始字符串）。本次CV足以关闭的只有"给相同54个物理状态继续增加action
方向"一条分支（每状态已有rank 16、condition中位1.41）；"增加更多独立物理状态"并未关闭——54个
状态对状态流形仍然很薄，且23个双repeat状态的两条context分属不同stratum，说明JOINT/H_ONLY不是
纯粹的物理状态类别，还受absolute action/first-pass center影响，覆盖与表示两条路都还开放。g0与H
的具体归因见下方分解复算。Actor、formal validation和test保持冻结。

### 11.27.1 零训练分解与反事实复算：g0主因确认（2026-08-14）

77个location的绝对梯度cosine混合了`g(a)=g0+H*Delta_a`，且CV的绝对口径是模型在每个absolute
action上重新输出g0 head，与centered H不是同一份预测的严格代数拆解。独立validator
（`scripts/model_verify/validate_mppi_grouped_cv_decomposition.py`）重载全部15个CV checkpoint
（hash全部通过），先做观察性分解（actor center处g0 cosine、真值中心化响应
`Delta_g_true(a)=g_true(a)-g_true(a0)` vs H预测响应`H(a0)(a-a0)`），再做严格反事实分解：
`true_g0+pred_H*Delta_a`（隔离H）、`pred_g0+true_Delta_g`（隔离g0）、`pred_g0+pred_H*Delta_a`
（完整局部模型）。center location（`Delta_a=0`退化为cosine 0）已从所有centered/反事实指标中
排除；反转pair计数不再对单侧枚举重复除2（正确总数3,417，此前归档的除2数值作废）。

观察性分解（严格pooled 15 run，18,012个outer location）：绝对梯度median `0.332`/P10 `-0.961`；
centered H响应median `0.831`/P10 `-0.761`/norm ratio 0.400；g0 at center median `0.190`/
P10 `-0.960`/norm ratio 0.515。分层反事实是决定性的：

| 分层 | 绝对梯度 | g0 center | centered H | true g0+pred H | pred g0+true Δg | pred g0+pred H |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H_ONLY | 0.765 | 0.846 | 0.822 | 0.992 | 0.813 | 0.828 |
| JOINT_G0_H | -0.680 | -0.805 | 0.840 | **0.988** | **-0.093** | -0.131 |

即review预设的判定标准被满足：隔离H的反事实在JOINT层0.988、P10仅-0.06，明显正常；隔离g0的
反事实在JOINT层-0.093、P10 -0.98，仍然反向。结论收窄为：JOINT层绝对梯度中位方向失败主要由
center处g0反向解释，H不是造成两分层差异的首要因素。附带发现：完整代数局部模型
`pred_g0+pred_H*Delta_a`的pooled median 0.655，明显高于当前部署式"每个absolute action重估
g0 head"路径的0.332——即使不重训，评估/部署路径也应以center+H外推为准。

口径与分支：本次关闭的仅是"给相同54个物理状态继续增加action probe方向"；H问题并未关闭——
centered norm ratio中位仅0.400（幅值低估约2.5倍）、centered P10 -0.761仍深负，H幅值校准与
尾部质量作为常设门槛，后续任何g0重训都必须继续报告这两项，防止修好g0后被H幅值低估卡住。

下一步（采纳review）：先单独训练/验证g0（冻结或detach H）；纳入尚未覆盖的125个G0-only
context并按物理状态grouped split；优先改善state×absolute action的显式交互/条件化而不是增加
同状态action probes；判定仍用cosine median、P10、norm ratio、control退化与多seed联合口径。
g0通过前Actor继续冻结；formal validation和test保持封存。

```text
outputs/mppi_proposal/targeted_grouped_cv_20260814_v1/summary.json
outputs/mppi_proposal/targeted_grouped_cv_20260814_v1/oof_decomposition_validation.json
outputs/mppi_proposal/targeted_semantic_structured_local_q_20260814_v1/unseen230_paired_comparison.json
scripts/model_verify/run_mppi_targeted_grouped_cv.py
scripts/model_verify/compare_mppi_unseen230_paired.py
scripts/model_verify/validate_mppi_grouped_cv_decomposition.py
```

### 11.28 预注册：g0专项grouped CV（2026-08-14）

按§11.27.1的下一步预注册并启动g0专项实验。数据：把600个internal-selection fresh-FD context
正式转为g0训练数据，按物理状态grouped split——204个hard context（JOINT 34 + H_ONLY 45 +
G0_ONLY 125）对应127个独立物理状态，5-fold贪心均衡（stratum/scenario/repeat数），每fold约25-26
状态/40-41帧；heldout状态全部帧留出。control排除按state级：21个control状态名下全部42帧
（含11个ALL_GOOD partner、3个CRITIC_ONLY、5个G0_ONLY_CRITIC_HARD、1个G0_ONLY_CRITIC_OK、
1个JOINT partner）永不进训练；control评估只用21个control-role帧，退化门用strict 20
（剔除与target 6019共享状态的6018）。

训练：H头（diagonal/low-rank）从optimizer中排除并保持零初始化，chord/probe等全部H相关loss移除
——本轮H指标按设计不适用，standing H门留给下一轮合并重训。监督为internal-fit base value/gradient、
train-fold的~512个fresh-FD帧gradient、~64个target context的location-0 center value/gradient。
checkpoint只由internal-validation（cosine/norm简化score）选择。两臂×3 seed×5 fold：
PA_G0（现有semantic encoder）与PA_G0_X（新增显式state×action低秩双线性交互encoder，
`TorchMPPISemanticInteractionStateActionEncoder`）。

判定（每臂独立）：OOF g0 cosine median `>=0.70`、P10 `>=0`、norm ratio `[0.5,2.0]`；
`>=4/5` fold（每fold `>=2/3` seed）、`>=2/3` seed全fold通过、strict-20 control median paired
delta `>=-0.02`。通过仅授权下一轮g0+H合并重训（重新启用standing H门），不解冻Actor；
失败则表示g0在消费全部hard层后仍无法跨状态迁移，回到表示设计。产物目录：
`outputs/mppi_proposal/g0_grouped_cv_20260814_v1/`。Actor、formal validation和test保持冻结。

### 11.29 执行结果：g0专项CV两臂全败，g0-hard状态系统性反向（2026-08-14）

§11.28预注册实验完整执行（30个训练，checkpoint/label hash全部通过）。两臂判定完全失败：
PA_G0与PA_G0_X均`0/5` fold、`0/3` seed、control退化（strict-20 median paired delta：
PA_G0 cosine median `-0.147`/P10 `-0.418`，PA_G0_X `-0.082`/`-0.131`）。

分层OOF g0 cosine median（15 run中位）是本轮核心结果：

| 分层 | PA_G0 | PA_G0_X |
| --- | ---: | ---: |
| H_ONLY | 0.691 | 0.879 |
| JOINT_G0_H | -0.648 | -0.709 |
| G0_ONLY_CRITIC_HARD | -0.866 | -0.866 |

三点结论。第一，g0-hard状态（JOINT与G0_ONLY）在heldout上不是"学不好"而是系统性反向
（-0.65~-0.87，比随机的0还差），且G0_ONLY_CRITIC_HARD首次被测量即为最差层——g0尾部问题
比§11.27.1在79 target上观察到的更宽更深。第二，显式state×action双线性交互只改善H_ONLY层
（0.691→0.879），对两个g0-hard层无 rescue，交互缺失不是g0反向的主因。第三，两臂在未接触230帧
子集的fresh cosine中位都到0.85，control却相对旧PA退化——模型在easy分布上仍好，g0-only训练
（冻结H、消费fresh帧）对easy状态还有净伤害，存在hard/easy之间的干扰。

判定`G0_GROUPED_CV_FAIL_REPRESENTATION_STILL_BLOCKING`（artifact保留原字符串；采纳review修订后
本节qualification收窄为`G0_GROUPED_CV_HARD_STATE_TRANSFER_FAIL_ACTOR_FROZEN`）。本轮确认的
是g0-hard跨状态迁移失败；"表示是唯一阻塞"证据过强，表示、训练配比、checkpoint选择、状态覆盖
四者尚未分开。具体依据：每epoch更新比例约base:easy-fresh:hard为24:3:1，G0_ONLY只进fresh轮，
实际监督信号占比仅百分之几；且checkpoint只由easy internal-validation选择，没有训练折内的
hard-state validation，即便某epoch学到hard映射也可能因easy分数落选。easy保持0.85而hard反向，
与"hard被平均掉或checkpoint选错"相容，不能单独证明encoder表示不了。

两点修正。第一，交互臂的H_ONLY改善不稳定：15组配对中10组改善，paired median `+0.069`但
paired mean `-0.045`（个别fold崩溃），H_ONLY P10通过数2/15→6/15仍不过半，且2/15个run的
H_ONLY中位数为负。表述应为"显式state×action交互提供正向信号，但收益有明显fold/seed不稳定
性，尚未形成可靠跨状态泛化"。第二，hard/easy干扰只是局部证据（strict-20 control退化但230帧
子集仍0.85），保留为待验证假设而非已确认因果。

两处工程口径问题已登记并将在重跑前修复：fold gate漏了预注册的P10条件（本轮median已全败，
不影响0/5结论）；targeted-center未排除control state——6019（JOINT，episode_361#4）在4/5个
fold经center训练其control state，strict-20 control门因剔除6018未受泄漏，但"21个control
state永不进训练"的契约不完全成立。

### 11.30 预注册与零成本诊断：平衡重跑 + 反向对结构分析（2026-08-14）

平衡重跑（review修订建议）已按以下协议启动，fold划分与§11.28逐位一致：base internal-fit、
easy fresh、hard fresh+center三个pool每epoch各取等量batch（1:1:1）；hard pool采样为
stratum-uniform→state-uniform→item-uniform，G0_ONLY获得与JOINT/H_ONLY相同监督权重；每个
训练fold内再按stratum分层划出20% hard状态作grouped hard-validation，checkpoint由easy
internal-validation与hard-validation联合score选择（双norm资格门）；fold gate补上P10；control
state从包括targeted center在内的全部训练pool排除（6019不再训练其control state）。仍为两臂
×3 seed×5 fold，不新增rollout。判定沿用§11.28联合门。产物目录：
`outputs/mppi_proposal/g0_balanced_grouped_cv_20260814_v1/`。

判决逻辑（预注册）：平衡训练后hard层显著改善→主要是训练配比/checkpoint选择问题；
可观测量能稳定分开反向对→做regime head；hard在低密度区且邻域跨度大→扩大独立物理状态覆盖；
平衡训练仍反向且可观测量不可分→才认定当前state representation/context缺失。

零成本结构诊断已完成（`scripts/model_verify/analyze_mppi_g0_reversal_separability.py`）。在
physical state+absolute center空间对600个context建kNN（k=10）图，43.3%的近邻pair真实g0方向
相反（负cosine）。按episode分组5-fold的分离性检验：可观测量（initial_state_six、current、
reference摘要、速度、clip、scenario、repeat、steering center、sigma）预测翻转的AUC仅
0.532（GB）/0.515（LR）；k-means(k=12) on [观测+g0标签]的label-aware聚类（不可部署）也只有
0.583。范数核查排除小范数噪声解释：翻转率在min-norm四分位上持平（0.409/0.445/0.452/0.453），
翻转对的`|cos|`中位0.838。密度：最近同向邻居距离中位2.15、最近反向4.75；各层kNN冲突率
ALL_GOOD 0.30、JOINT 0.60。

该诊断的结论强度需要收窄（review指正，四点限制）：pair分类器只输入两点观测的绝对差，没有
pair中点或端点绝对位置，而basin边界位置取决于"这对点在哪"；pair分组按字典序较小episode，
非endpoint-isolated CV；k-means(k=12)只是一种聚类方式，0.583不是理论上界；最近同向2.15对
反向4.75本身说明一定局部一致性仍存在。且对确定性DBM，若局部cost连续可微，梯度方向只有在
范数趋零时才可能快速翻转——当前翻转对范数不小，因此"数学上本质不连续"未证明，更可能是
状态覆盖稀疏、距离度量别名或缺少条件变量。因此§11.30的正确表述是：**在当前特征集与该聚类
配置下未发现可分结构；regime/簇条件化未被关闭，只是当前证据不支持它**。范数四分位与|cos|
分布已补写入artifact以便独立复算。产物：

```text
outputs/mppi_proposal/g0_reversal_separability_20260814_v1/analysis.json
```

### 11.31 执行结果：平衡重跑部分挽救JOINT，G0_ONLY反向稳健，配比假设部分成立（2026-08-14）

§11.30平衡重跑完整执行（30个训练，hash全过；epoch 110-160、best_epoch 80-113，双norm资格
与hard-validation联合选择正常工作）。判定仍为`G0_BALANCED_GROUPED_CV_HARD_STATE_TRANSFER_FAIL`：
两臂`0/5` fold、`0/3` seed，control退化加深（strict-20 median delta cosine median：PA_G0
`-0.194`、PA_G0_X `-0.137`）。

分层OOF g0 cosine median（15 run中位，括号内为§11.28不平衡轮）：

| 分层 | PA_G0 | PA_G0_X |
| --- | ---: | ---: |
| JOINT_G0_H | **+0.112**（-0.648） | **+0.072**（-0.709） |
| H_ONLY | 0.492（0.691） | 0.617（0.879） |
| G0_ONLY_CRITIC_HARD | -0.813（-0.866） | -0.837（-0.866） |

三点结论。第一，**配比/checkpoint假设对JOINT层成立（配对口径）**：逐fold-seed配对后，
PA_G0改善中位`+0.452`（11/15改善）、PA_G0_X `+0.676`（11/15改善），把JOINT从系统性反向
（-0.65~-0.71）拉回正值（+0.07~+0.11）；此前的+0.75/+0.78是headline中位数直减，非严格
paired median。但JOINT仍远低于0.70门，P10 -0.86~-0.89。第二，**G0_ONLY的反向不足以被平衡
训练改变失败性质**：配对改善仅`+0.059`（PA_G0，10/15）与`+0.022`（PA_G0_X，9/15），且两臂
15/15个run的OOF中位仍为负、全部P10为负——它不是训练配比伪影。第三，重新平衡对H_ONLY有
配对代价（PA_G0 `-0.199`、3/15改善；PA_G0_X `-0.075`、4/15），对strict-20 control伤害加深，
而未接触230帧中位反而升至0.90/0.93：hard/easy干扰假设得到进一步支持但分布内部效应不一致，
仍是局部证据。

平衡CV另有两个口径问题（review指正，外层OOF失败结论不受影响，但两臂架构比较被污染、
"严格grouped hard-validation"表述降级）：inner hard-validation split由跨arm前进的全局RNG生成，
同fold两臂的19个hard-val状态仅重叠3-5个；easy pool只排除了outer-heldout，inner hard-val
状态的非hard repeat帧仍经easy pool进入训练（实测每fold/arm 19个中6-8个泄漏）。下一轮必须
按fold固定生成一次inner split供两臂所有seed共用，并把inner val状态从easy pool排除。

综合§11.30-11.31，按预注册判决逻辑收敛为：当前特征与聚类配置下未见可分结构（regime路由未被
关闭但无支持证据）；配比/checkpoint解释JOINT的大部分反向但无法达到任何门；G0_ONLY反向在
等权+hard-validation下稳健（15/15负），叠加43%近邻翻转但局部一致性仍存（同向2.15/反向4.75）。
剩余两个未分开的分支：(a)
**预测目标重设**——own-center梯度方向作为回归目标可能是问题本身（翻转对范数不小，若cost
局部可微则更可能是覆盖/度量/条件变量问题而非数学不连续），而value场连续、H沿chord可迁移
（0.83~0.99）；改为共享标量value模型加相对value监督`Delta Q=Q(s,a_i)-Q(s,a_0)`，再用对踵差分
`g_hat_i=[Q(s,a0+rd_i)-Q(s,a0-rd_i)]/2r`取梯度，把任务从"回归16维不连续方向"变成"回归连续
相对value"；(b) **状态覆盖密度**达到basin边界间距尺度。其中(a)可用fresh数据已有的
0.01/0.02/0.04σ三组33点标签零rollout先验证（0.02σ主训练、另两半径一致性验证），并按review
要求修复inner split跨arm一致性与easy pool泄漏、与直接g0模型共用fold、分层报告G0_ONLY/JOINT/
H_ONLY/control/230、同时报15-run统计/严格pooled/配对改善。判定：value delta在heldout G0_ONLY
可拟合且差分梯度转正→主要是预测目标/结构问题；value delta本身不可迁移→更明确指向覆盖或
缺失context。Actor、formal validation和test保持冻结。产物：

```text
outputs/mppi_proposal/g0_balanced_grouped_cv_20260814_v1/summary.json
```

### 11.32 预注册：value-delta pilot（2026-08-17）

按§11.31收敛的分支(a)预注册并启动。任务重设：不再直接回归16维own-center梯度方向，改为
训练共享标量局部Q模型，监督相对value `Delta Q_i = Q(s,a_i) - Q(s,a_0)`，梯度由对踵差分
`[Q(s,a0+r d_i)-Q(s,a0-r d_i)]/2r`读出（对二次参数化等价于g0头投影，因此本pilot比较的是
同一结构下value监督与梯度监督）。数据全部零rollout：fresh 600 context已有0.01/0.02/0.04σ
三组33点value标签，0.02σ为主监督半径，0.01/0.04仅作一致性评估；base pool用其最小半径
0.05σ。三pool（base/easy/hard）每epoch 1:1:1 batch，各pool按自身delta尺度归一。

按review修复并落实：inner hard-validation split按fold固定生成一次，两臂全部seed共用；
easy pool同时排除outer-heldout、inner-val与control状态；外层fold与g0两轮逐位一致；
分层报告G0_ONLY/JOINT/H_ONLY、strict-20 control、230帧子集；与§11.31梯度监督轮做逐
fold-seed配对比较；15-run统计与严格pooled并报。判定：value delta在heldout G0_ONLY可拟合
（delta correlation/RMSE）且差分梯度OOF转正（cosine median `>=0`、配对相对梯度监督改善）
→主要是预测目标/结构问题；value delta本身不可迁移→更明确指向覆盖或缺失context。
Actor、formal validation和test保持冻结。产物目录：
`outputs/mppi_proposal/g0_value_delta_cv_20260817_v1/`。

### 11.33 执行结果：value-delta消除G0_ONLY反向但自身不可迁移，覆盖/缺失context成为主嫌疑（2026-08-17）

> **INVALIDATED_PENDING_RERUN（2026-08-17）**：review发现`probe_delta_value`存在批量对齐错误
> ——`repeat_interleave`展开的context与`cat`拼接的action错行（batch内只有首尾两行属于同一样
> 本），训练与评估的`Delta Q`全部由错配的state/action对算出。该错误足以解释本节全部三个结果
> （ΔQ correlation约0、RMSE/scale约1、G0_ONLY"去反向"更可能是错误监督导致的收缩）。本节结论
> 全部撤回，等待修复后的重跑；"需要新增数据/context"的分支判断同样不得引用本节。artifact保留
> 作废标记。

§11.32 pilot完整执行（30个训练，两臂inner split逐fold共享、easy pool三重排除已落实）。核心
结果分三层。

第一，**value监督消除了G0_ONLY的系统性反向**：相对§11.31梯度监督轮的逐fold-seed配对，
G0_ONLY改善PA_G0 `+0.938`（14/15）、PA_G0_X `+1.079`（15/15），OOF中位从-0.81/-0.84回到
`+0.097/+0.165`；P10从约-0.97回到-0.56。直接16维梯度监督在G0_ONLY上诱导的是反向映射，
换成相对value监督后反向消失——监督目标的选择对该层有决定性影响。

第二，**但value delta本身在heldout上完全不可迁移**：heldout hard状态的`Delta Q`拟合在三个
半径上全部失效——correlation中位`-0.077/-0.076`（零信息），RMSE/scale `~1.00`（不优于
预测常数），pair sign accuracy `~0.46`（随机）。差分梯度OOF也只有`0.02~0.19`，远低于0.70门，
且H_ONLY相对梯度监督配对倒退`-0.82/-0.82`（OOF -0.13/-0.09）、未接触230帧cosine中位转负
（-0.14/-0.19）、norm ratio中位0.36~0.60。即value监督是"去反向"而不是"学到结构"。

第三，按§11.32预注册判定落入第二分支：**value delta不可迁移→更明确指向状态覆盖或缺失
context**。至此四条监督/表示路线已在同一grouped CV口径下测完——直接梯度（反向）、平衡梯度
（G0_ONLY仍反向）、显式交互encoder（不救g0-hard）、相对value（去反向但零信息）——heldout
hard状态的局部结构对当前特征集一致不可预测，而同状态内拟合始终良好（seen +0.66）。结合
§11.30的近邻翻转与不可分性，剩余可行动作收敛为：(a) 扩大独立物理状态覆盖（对600 context
之外的分布补采，含G0_ONLY类状态），或 (b) 引入当前特征集之外的条件变量（更丰富的
history/上下文/环境描述）。在两者之一落地前，g0侧不再进行第三轮表示内搜索。
Actor、formal validation和test保持冻结。产物：

```text
outputs/mppi_proposal/g0_value_delta_cv_20260817_v1/summary.json
```

### 11.34 v2修复与seen-fit门：dropout坍缩定位，seen收敛/OOF失败定局（2026-08-17）

按review合同完成v2重写：真标量`Q(s,a)=value_head(trunk(encoder(s,a)))`，g0/H头dead，梯度
由autograd `dQ/da`在a0处读出；批量对齐三项单元测试（batch=逐样本、`a_i=a0`时`Delta Q=0`、
置换一致）训练前强制通过；checkpoint改由value-delta指标（inner grouped hard-validation +
easy internal-validation的corr/RMSE/sign）选择，梯度仅作派生评价；strict-20 control、逐seed
严格pooled、完整gate（corr`>=0.50`/RMSE`<=0.80`/sign`>=0.60`/梯度cosMed`>=0`，`>=4/5` fold
+`>=2/3` seed）全部入summary。

分阶段验证发现训练坍缩：默认dropout 0.05下，160ep全长度run的训练loss停在~0.40、seen
hard-train corr≈0、**4/5 fold的`dQ/da`精确为0**——Q坍缩为action不变的状态函数；对照dropout
0.0的60ep短训seen梯度cosine可达+0.85。机制：0.02σ探针的`Delta Q`信号极小，encoder对微小
action差异的响应被dropout噪声淹没，"预测0"成为局部最优。dropout=0的160ep×1臂×1seed门run
确认修复：5/5 fold seen收敛（value corr `+0.65~+0.76`、rmse/scale `0.66~0.77`、seen梯度
cosine `+0.59~+0.80`、epoch 118-160）。

同一门run给出干净的heldout对照：OOF value corr `-0.67/-0.32/-0.00/+0.01/+0.20`（中位≈0），
rmse/scale `0.98~1.16`（劣于常数），sign `0.36~0.43`（低于随机），derived梯度cosMed 4/5为负
（`-0.60~-0.83`、P10约-0.95）。即seen可拟合、heldout零信息且部分反向——与四轮监督/表示实验
的"记忆不泛化"模式一致。据此seen-fit判定门通过，全量2臂×5 fold×3 seed（dropout 0）已启动，
正式判定与严格pooled/control/gate以全量为准，产物：
`outputs/mppi_proposal/g0_value_delta_cv_20260817_v2/`。Actor、formal validation和test
保持冻结。

### 11.35 执行结果：v2全量确认value delta不可迁移，正式进入覆盖/缺失context分支（2026-08-17）

全量2臂×5 fold×3 seed（dropout 0，单元测试通过，seen收敛corr `+0.67/+0.72`）完成。判定
`G0_VALUE_DELTA_V2_FAIL_COVERAGE_OR_CONTEXT_SUSPECTED`：`0/15` run通过value门+梯度门，
`0/5` fold、`0/3` seed。逐seed严格pooled：value corr `-0.03~-0.12`、rmse/scale `1.05~1.08`
（劣于常数）、sign `0.40~0.42`（低于随机）、derived梯度cosMed `-0.62~-0.70`。与梯度监督轮
配对中位`-0.01/-0.05`，无改善。

分层结构清晰且自洽：H_ONLY的derived梯度OOF `+0.65~+0.68`、strict-20 control（从未训练的
easy状态）梯度`+0.63~+0.66`——value路线在easy/H-only层可以迁移；G0_ONLY梯度`-0.81/-0.83`
（同样反向，v1所谓"去反向"确系对齐bug伪影）、JOINT的value corr `-0.37~-0.39`（主动错误）。
即：**同一个标量Q模型，easy与H-only状态可学可迁移，g0-hard状态（JOINT/G0_ONLY）的局部
value结构与方向一致不可预测，且seen拟合始终良好**。五条路线（直接梯度/平衡梯度/交互encoder/
v1-value/v2-value）证据闭合，按§11.32预注册正式进入"状态覆盖或缺失context"分支。

context侧首个核查已完成：数据集内`dbm_params`（14项 Pacejka/质量/转向/油门参数）、
`cost_weights`（6项）、`mppi_params`（15项）在全部episode上**完全恒定**——不存在"环境/车辆/
cost参数跨episode变化但未输入Critic"的缺失条件变量。且DBM cost只依赖rollout轨迹、参考轨迹、
当前控制与固定权重，不读取障碍物或参考之外的地图，"缺少局部地图几何"没有依据（此处收窄
§11.35初稿表述）。若context确实缺失，候选只剩三类：`initial_state_six`中真实的vy/yawrate
未被显式无损地传给Critic；history/current/reference编码压缩丢失了区分basin的信息；参考特征
表达精度不足。覆盖密度分支与"输入表示对照"分支均开放，先用零成本邻域诊断路由（见§11.36）。
Actor、formal validation和test保持冻结。产物：

```text
outputs/mppi_proposal/g0_value_delta_cv_20260817_v2/summary.json
```

### 11.36 零rollout邻域诊断：翻转率对距离不敏感+local oracle高而固定KNN失败，路由到表示/度量分支（2026-08-17）

合并诊断（`scripts/model_verify/analyze_mppi_g0_neighbor_distance_oracle.py`）按review收紧口径
执行：物理距离由完整DBM Markov输入构成（`initial_state_six` 6维、`direct_reference`
50x4原始参考、`current_action` 2维、absolute Actor center 16维），各维按IQR标准化、块等权；
邻居候选排除自身、同物理snapshot与同episode；按最近邻距离全局十等分桶报告翻转率
（Wilson 95% CI）；固定KNN与local label oracle只在其k邻域内取（k=1/3/5/10），不使用全局
target-aware oracle。600 context全量，同episode邻居已排除故无泄漏。

两个核心结果。第一，**翻转率对距离不敏感**：最近邻距离中位0.84（标准化单位），从最近桶
（0.3-0.5）到最远桶（1.3-2.0），全体翻转率0.43→0.46基本持平；G0_ONLY各桶在0.17-0.78间无
趋势（每桶n=6-15），ALL_GOOD持平在0.18-0.48。距离已经到达当前数据可达的最近处，翻转率仍
~40-55%——按review路由第4条，**加密状态采样不会显著降低翻转率，不盲目补数据**。第二，
**local oracle高而固定KNN失败**：G0_ONLY的k=3邻域内79%的context存在正cosine标签（k=5为
90%、k=10为99%），local oracle中位0.88~0.95；但固定KNN（邻居标签归一均值）k=1/3/5/10为
-0.01/+0.19/-0.26/-0.35。JOINT同型（oracle 0.92~0.96 vs 固定-0.41~-0.10）；H_ONLY与
ALL_GOOD的固定KNNk=1/3约0.6-0.72可工作。即g0-hard状态的邻域**包含**同basin标签分支，但
可观测距离无法选出是哪一个——按review路由第2条：**数据中有信息，问题在表示或距离度量**。

综合路由结论（口径收窄）：距离-翻转率平坦只能说明**按当前"块等权+IQR+欧氏"度量的广泛加密
没有依据**，不能严格证明更密覆盖一定无效（G0_ONLY每桶仅6-15样本，统计能力有限）；local
oracle用真实梯度选邻居，只能证明**局部候选池里常有正确分支**，不能证明现有可观测输入包含
识别该分支的信息——它恰好支持继续做显式输入对照。据此覆盖密度分支降级但未关闭，缺失context/
表示分支升级，下一步做review指定的三臂严格配对**充分输入/编码损失对照**（E0现有编码基线、
E1编码器+显式six/current_action skip、E2纯DBM充分输入去history），统一scalar Q+value-delta
loss、同fold/inner split/seed、dropout 0、value口径checkpoint与完整gate、显式输入归一化仅在
训练fold上拟合；先过tiny overfit与seen-fit门再跑全量。若E1/E2均不恢复G0_ONLY/JOINT，则
"编码压缩"归因失败，覆盖密度、函数复杂度或未记录条件变量保留为候选。在此之前不进行
loss/dropout/encoder扫描与数据加密。Actor、formal validation和test保持冻结。产物：

```text
outputs/mppi_proposal/g0_neighbor_oracle_20260817_v1/analysis.json
```

### 11.37 执行结果：三臂显式输入对照无恢复，编码压缩归因失败（2026-08-17）

§11.36预注册的三臂对照完整执行（45个训练，tiny overfit门E1/E2均0.958、seen-fit门通过、
每run单元测试通过、fold/inner split/seed/lr/dropout/checkpoint/gate严格配对、显式输入
fold-local归一化）。判定`EXPLICIT_INPUT_NO_RECOVERY_ENCODING_NOT_THE_CAUSE`：

| 臂 | G0_ONLY梯度中位（负run数） | JOINT | H_ONLY | strict-20 control |
| --- | ---: | ---: | ---: | ---: |
| E0 现编码基线 | -0.829（13/15） | -0.163 | +0.485 | +0.647 |
| E1 +显式six/control skip | -0.739（14/15） | -0.586 | +0.637 | +0.588 |
| E2 纯DBM充分输入 | -0.835（15/15） | +0.369 | +0.379 | +0.658 |

逐fold-seed配对：E1-E0在G0_ONLY上median `+0.061`（10/15）、E2-E0 `-0.019`（6/15）——均无
恢复；E1的H_ONLY +0.637与E2的JOINT +0.369属波动级，不改变判定；control三臂保持+0.59~+0.66，
未出现"修hard破easy"。即：显式补入真实vy/yawrate等六维状态（E1）、或将全部输入换成原始
Markov充分集（E2），都不能让g0-hard状态在heldout上转正。

至此，在已记录数据内可检验的输入侧假设全部穷尽：**"编码压缩丢失basin信息"不成立**。按
§11.36预注册判定，剩余候选为：(a) 状态覆盖密度（§11.36弱化但未关闭——其平坦结论仅在当前
块等权IQR度量下成立）；(b) 函数内在复杂度（g0-hard状态的局部响应在现有采样密度下本质不可
从有限邻居插值）；(c) 尚未记录的条件变量——不在任何现有数组中，零成本检验已无法触及。三
者都需要新增信息（更密的状态采样、或数据生成时记录新变量）才能进一步区分。g0侧在现有
数据上的归零搜索到此收束：六轮实验（直接梯度/平衡梯度/交互encoder/value v1/value v2/显式
输入三臂）在同一grouped CV口径下一致表明——easy与H_ONLY层可学可迁移，JOINT/G0_ONLY层的
局部结构在当前数据+输入下跨状态不可预测。Actor、formal validation和test保持冻结；后续
若重启，须先明确(a)/(b)/(c)中走哪条并按本节口径预注册。产物：

```text
outputs/mppi_proposal/g0_explicit_input_cv_20260817_v1/summary.json
```

#### 11.37.1 Review 口径修正：关闭简单缺失变量，收窄“编码无关”结论

§11.37 的主判定成立，但候选 `(c) 尚未记录的条件变量` 不应继续与覆盖/复杂度等权并列。
当前确定性 DBM 动力学由 `initial_state_six + future_action + fixed dbm_params` 唯一确定；其公开
rollout 不使用 history/current action，cost 额外只读取 reference、current action 与固定权重。
E2 已输入 six、原始 reference、current action 和 absolute action，且参数/权重跨 episode 恒定、
cost 不读取地图或障碍。因此除非后续出现 source provenance、插值或 replay 合同不一致的新证据，
“数据数组之外还有决定标签的隐藏环境变量”在当前 DBM 实验中视为关闭，不以补录未知 context
作为下一步。

同时，`EXPLICIT_INPUT_NO_RECOVERY_ENCODING_NOT_THE_CAUSE` 只关闭“旧 encoder 压缩时丢失
充分信息”这一简单归因，不证明任意 representation/inductive bias 都无效。E2 仍是在有限状态上
用网络学习高维确定性映射；它的失败与 tiny/seen 成功共同说明**当前 scalar-Q/value-delta
Critic 家族没有跨状态插值能力**。E2 的 JOINT `+0.369` 可记作局部改善，但 G0_ONLY `15/15`
run 为负且总 gate 全败，不能据此解冻 Actor。剩余可行动分支收窄为：

1. 状态覆盖相对于 basin 边界变化尺度不足；
2. 映射高频/归纳偏置不匹配，当前数据量下难以插值。



---

## 附录 A：主路线中的已作废实验记录

> 以下四个 block 已从主文档 tombstone 化，完整原文按原出现顺序保存于此。


### 11.51 Phase 2 A0执行结果：train恢复<50%，判定树落入拟合/标签参数化分支（2026-08-18）

> **状态修正（见§11.53）**：本节使用的3-fold训练器存在早停计数错误：`patience=25`
> 从epoch 1累计，而checkpoint到`selection_min_epoch=40`才允许保存，导致9/9 run均在
> epoch 25结束、`best_epoch_loss=Infinity`。本节数值保留作审计记录，但不得继续作为
> A0正式判定；修复后的严格配对复算见§11.53。

A0按§11.49合同完整执行（3-fold×3seed，actor结构原样、anchor=a0、部署输入契约逐状态重建、
a0 replay与Phase 1b标签交叉验证通过、9个checkpoint落盘）。判定
`A0_TRAIN_FAIL_FIT_STRUCTURE_OR_LABEL_PROBLEM`：

| seed | H_OOF | H_train中位 | OOF fold |
| --- | ---: | ---: | --- |
| 0 | 0.106 | 0.390 | -0.078/0.101/0.256 |
| 1 | 0.076 | 0.446 | -0.194/0.051/0.308 |
| 2 | 0.135 | 0.394 | 0.091/0.029/0.247 |

median seed CI [-0.030, 0.159]含0；主门、CI门、tail门全部失败（P05 -4.5~-11.0，worst
-24~-343）；**stay误动率0.80-0.97**（actor几乎对所有stay状态也输出移动），mover漏动率
0.01-0.08；knot误差early/late均匀（0.058-0.064 knot单位）。A0失败的可靠证据以
H_train<0.5为主（先前"0.06误差≈teacher移动幅度"的类比口径不同——逐元素中位误差与σ归一化
整体移动不可直接比较，该表述作废）。结论限定为"结构或标签参数化问题"，非canonical标签
只是候选之一，由A0b分离。

按§11.49判定树：**train恢复<50% → 拟合/结构/标签参数化问题**（泛化差>15pp同时成立但
次要——train本身未过门）。机制解读与§11.43-11.45一致：单bank标签方向非canonical
（跨bank cosine 0.1-0.64、P90 cost gap至4.0），MSE在非canonical方向标签上回归到均值，
预测落点偏离软盆地；且14% stay标签被86% mover稀释，无stay机制的纯MSE把几乎所有状态
推向移动（stay误动97%）。tail的worst -343再次确认"错误方向的移动比不动危害大得多"。

由此触发§11.49预设的consensus_64条件分支（"A0训练集难拟合"已成立）。下一步按计划推进：
(a) B0 sensitivity审计（A1的前置）与C1 along/cross分解立即并行；(b) A0b消融——
guarded consensus_64标签（构造零回归、方向平均化安全已在1a验证）与/或stay加权损失，
与A0严格配对判定"标签形式"与"坐标"哪个是主因；(c) A1（S归一化坐标）待B0完成后执行。
Actor、formal validation和test保持冻结。产物：

```text
outputs/mppi_proposal/actor_a0_baseline_20260818_v1/summary.json
scripts/model_verify/run_mppi_actor_a0_baseline.py
```



### 11.49 Phase 2 A0 执行结果：Actor在训练状态上也仅恢复4%——蒸馏层失败，分支路由到标签噪声（2026-08-18）

Phase 2 A0（首次actor可学性测量）已执行。实现与产物：

```text
scripts/model_verify/train_mppi_proximal_residual_actor_phase2a0.py
outputs/mppi_proposal/phase2a0_residual_actor_20260818_v1/
  summary.json / b0_sensitivity.npz
```

合同：Phase 1b multi-128 guarded标签（600状态）、语义干净输入（six+reference+current+a0，
fold内IQR）、`Delta_a=2sigma*tanh(f_theta)` 16维bounded residual + 辅助stay logit
（BCE权重0.1）、episode-grouped 5-fold（按速度分层）×3 seed、fold内90/10 episode早停、
**真实DBM rollout评价**。B0逐状态`||grad J(a0)||`敏感度已存（fold-train-only用途，
供A1）。

结果（15个fold-seed）：`r_teacher=0.7245`，**r_train中位仅0.044**（全距-0.277~0.211），
r_heldout中位-0.004（全距-0.361~0.040）；heldout回归比例中位0.60、P05增益中位-6.8、
worst -207.7；stay recall中位0.07。早停在epoch 30--60，inner-val MSE 0.08--0.12。

按预注册分支判定：这是"**train也低**"分支——不是跨episode泛化墙（train与heldout
都接近零），而是**蒸馏层失败**：网络在训练状态上就无法把(s,a0)映射到可用精度的
teacher残差。机制与Phase 1a/11.45的证据直接相容：**单臂标签的方向在大偏离噪声**
（跨bank cosine 0.12--0.64、teacher移动量化为单一环步长0.264σ）——对方向噪声
~信号幅度的目标做MSE回归，最优解是向条件均值收缩；早停按inner-val选择了接近
零预测的checkpoint（零预测MSE参照即0.08--0.12量级），rollout收益随之消失。J16
蒸馏失败模式在近端尺度上复现，但现在有teacher/train/heldout三层分解与stay标签。

分支路由（下一步为已预注册的消融，非新发明）：**A0.1改用guarded consensus标签**
（11.47：consensus_64 R_a0 0.774、零回归、三bank均值是低方差目标且软盆地内均值
中心本身好）——这是把Phase 1a"回归到均值安全"的证据接到监督目标上的直接动作。
若A0.1的train恢复仍<0.2，则蒸馏层失败不限于标签噪声，需检查表示/优化（B0影响
归一化、加权损失或更深网络）后再谈泛化。horizon候选J50重rollout验证与rho join
维持非阻塞旁路。部署Actor继续冻结；formal validation/test保持封存。

勘误（11.48遗留）：`along_cross_identity_max_error`原artifact记录的是循环最后一个
状态的误差（3e-8量级、恒等式数学上精确、结论不受影响）；脚本已改为全局累计最大值
并补齐fresh/actor/manifest SHA256，未来重跑以新口径为准。



### 11.52 A0b 2x2消融与B0敏感度：标签形式是train失败主因，泛化成为新主阻塞（2026-08-18）

> **状态修正（见§11.53）**：本节A0b四臂沿用了§11.51同一早停实现，36/36 run的
> `best_epoch_loss`均为`Infinity`，因此A0b训练结果与由其导出的A1结果均作废；consensus-64
> 标签生成和B0动力学敏感度artifact不受影响。修复后的2x2结果见§11.53。

**600状态guarded consensus-64标签生成**（`generate_mppi_consensus64_labels.py`，三bank×64
候选+guard回退，117,000次评价）：J_teacher均值**7.812**（multi-128为7.887，配对改善
+0.075），fallback率**0.7%**（4状态）——零回归构造达成且teacher本身更好。

**2x2消融**（同fold/seed/网络/epoch/评价严格配对，stay权重=类平衡517/83≈6.23）：

| 臂 | H_train | H_OOF(med seed) | stay误动 | worst | CI |
| --- | ---: | ---: | ---: | ---: | --- |
| A0 multi-128+plain | 0.394 | 0.106 | 0.97 | -343 | [-0.03,0.16] |
| multi-128+stayw | 0.421 | 0.135 | 0.66 | -314 | [-0.01,0.18] |
| **consensus64+plain** | **0.513** | 0.218 | 0.00* | -106 | [0.035,0.221] |
| consensus64+stayw | 0.511 | **0.251** | 0.38* | -108 | [0.056,0.243] |

三点判定。第一，**标签形式是train失败主因**：consensus标签把H_train从0.39-0.42拉过
0.50门（+0.09~0.12），验证了"单bank方向非canonical导致MSE回归均值"假设；stay加权单独
不够（0.42）。第二，**泛化成为新主阻塞**：H_train≥0.50但H_OOF 0.21-0.25<0.50、差
0.26-0.29>15pp，判定树落入`GENERALIZATION_PROBLEM`；但CI下界首次为正——OOF层面首次
出现真实正收益（Actor跨episode回收teacher增益的22-25%）。第三，tail与stay改善：worst
从-343收窄到-106~-108、P05从-8.5到-5.5；stay机制生效（见下述口径注）。*口径注：
臂3/4的stay误动参考集不同（臂3对consensus自身4个fallback状态=0.00，臂4对phase1b的
83个stay状态=0.38），不可直接横比；后续报告统一以phase1b 83状态为stay参考集。

**B0敏感度审计**（`analyze_mppi_b0_sensitivity.py`，600状态纯动力学Jacobian，无cost
梯度）：(1) **S强各向异性**——同一knot内steer对终端位置的影响约为accel的2.4倍，且沿
knot早晚衰减约33倍（knot0 steer 3.99 → knot7 0.12）；(2) **fold稳定性5.9%**——S可作
固定部署常量；(3) 时间质量与cost梯度相反：终端位置影响55%在t1-16、35%中段、仅10%晚段
（早段控制力复利 vs cost梯度75%晚段——两者共同印证：早动作杠杆大、晚误差大）；
(4) endpoint半支撑明显（t49影响仅为晚段中位0.07倍）；(5) 非均匀knot候选位置
t=[3,7,11,15,19,25,32]。产物：`outputs/mppi_proposal/b0_sensitivity_20260818_v1/`。

下一步（按§11.49计划）：A1影响归一化坐标（z=SΔa，per-fold拟合S，与consensus64+plain
严格配对）即刻启动；若A1不改善泛化，则泛化阻塞指向状态覆盖或A2结构。Actor、formal
validation和test保持冻结。



### 11.54 A1旧执行结果（已作废）：S归一化坐标实验受早停bug污染（2026-08-18）

> **INVALIDATED_PENDING_RERUN**：本节9/9 run的`best_epoch_loss=Infinity`，与§11.53
> 审计确认的早停bug完全一致。以下数值保留作历史记录，不得用于关闭S归一化坐标假设，
> 也不得据此把分支路由到状态覆盖/A2；需用修复后的trainer严格配对重跑。

A1（consensus64标签+z=SΔa影响归一化坐标，per-fold拟合S，与c64+plain严格配对）完成：
H_train 0.508-0.533（配对臂0.511-0.513，无变化），H_OOF 0.153-0.227（配对臂0.207-0.221，
中位略降无改善）。**坐标条件数不是泛化阻塞**；train门保持通过。按§11.49路由，泛化阻塞
收敛到两个候选：状态覆盖（600状态/90episode对90单元×3episode的分层，每fold heldout是
actor未见episode）或A2网络结构（early/late分头）。判定树当前状态：
`GENERALIZATION_PROBLEM`（train≥0.50、OOF<0.50、gap>15pp、CI下界为正）。
Actor、formal validation和test保持冻结。产物：
`outputs/mppi_proposal/actor_a1_c64_snorm/summary.json`。
