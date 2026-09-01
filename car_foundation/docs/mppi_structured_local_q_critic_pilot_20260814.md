# MPPI Structured Local-Q Critic Pilot（2026-08-14）

## 1. 目标与边界

本 pilot 验证：在固定 DBM、冻结 sampling-center Actor 的前提下，显式的局部二次标量 Q 是否能
比普通 action MLP / 独立 gradient head 更可靠地学习 action-location response，尤其是 repeat
pair 中跨过 reward 局部极大点时的梯度反转。

本轮不更新 Actor，不新增随机状态，不加载 formal validation 或封存 test。训练复用现有
internal-fit 的 `0.05σ` 数值梯度、局部 probe value 和 repeat-pair 标签；验收使用已经消费且独立
验证过的600个 internal-selection fresh-FD context。pilot 通过只授权下一阶段的受限 Actor
实验，不等价于闭环或部署通过。

## 2. 标量局部 Q 参数化

动作使用现有16维归一化 Actor-action 坐标。默认参考点为当前确定性 Actor center
`a_ref=a_actor(s)`，该量在线可得且与当前部署契约一致：

```text
delta = a - a_ref
Q(s,a) = Q0(s) + g0(s)^T delta + 0.5 * delta^T H(s) delta
grad_a Q(s,a) = g0(s) + H(s) delta
```

不再保留普通 action encoder。state encoder 只输出 `Q0, g0, H`，候选动作对 Q/gradient 的直接
影响只能经过上式的显式线性/二次项。state encoder 仍包含 anchor、feedback 和 gradient context，
所以 state-side nuisance 并未被结构自动消除，必须继续由 same-absolute-center gate 检查。

H 必须显式对称并允许负特征值。当前目标是
`asinh((base_cost - cost)/5)`；cost 极小点对应 reward 局部极大点。审计中的70个反转pair沿连线
方向导数100%正→负，因此对应方向必须允许负曲率，禁止使用 `H=U U^T`、Cholesky、softplus
diagonal 等半正定参数化。pilot 使用：

```text
H(s) = Diag(d(s)) + U(s) Diag(lambda(s)) U(s)^T
```

`d` 和 `lambda` 均为有符号输出；可以用带可记录尺度的 `tanh` 限幅，但不能限制为非负。
low-rank 列向量按列归一化，降低 `U/lambda` 尺度不可辨识。由构造保证 `H=H^T`，并用数值 gate
复核 symmetry 与 autograd：

```text
max_abs(H - H^T) <= 1e-6
max_abs(autograd_grad - (g0 + H*delta)) <= 1e-6
```

## 3. 容量对照与 seed

固定同一 state encoder、数据、优化器、loss 和 checkpoint 选择，只改变 H 的响应容量：

| arm | H 参数化 | 回答的问题 |
| --- | --- | --- |
| H0 | `H == 0`，同loss/同代码路径 | 区分新loss组合与二次结构本身的贡献 |
| D | signed diagonal | 16个独立action通道是否已足够 |
| D+R1 | signed diagonal + signed rank-1 | 是否需要一个跨knot/action耦合方向 |
| D+R2 | signed diagonal + signed rank-2 | 第二个耦合方向是否继续带来稳定收益 |

H0仍计算并记录chord loss，但固定H使该项不向可训练参数传梯度；其余Q0/g0、probe-value、norm、
数据、seed、优化器和checkpoint逻辑完全一致。使用递增结构而非 pure rank-1/rank-2，保证每一级严格包含前一级，避免把“耦合秩”与“丢失逐维
响应”混为一谈。每组固定运行3 seed，逐seed和ensemble同时报告；不得用单seed最好结果替代结论。
前期审计显示负 alignment 质量93.41%集中在 steering，因此另报前4个 steering knot 的 sign
accuracy，但不降低16维动作自由度。

## 4. 弦长污染与参考点规则

300个 selection repeat pair 的绝对center距离（按 deployment sigma 标准化后的RMS）为：

| chord length | pair | true reversal |
| --- | ---: | ---: |
| `<=0.10σ` | 83 | 13 |
| `<=0.15σ` | 152 | 33 |
| `(0.15,0.30]σ` | 98 | 20 |
| `>0.30σ` | 50 | 17 |

G0.1 已证明小半径二次模型向约 `0.15σ` 外推时开始明显失真。因此 `H*Delta_a=Delta_g_FD`
不能让长弦长样本主导。主训练采用截断反距离权重并归一化 batch 权重：

```text
d = RMS((center_1 - center_0) / sigma)
w(d) = min(w_max, d0 / max(d, epsilon))
L_chord = normalized_mean(w(d) * robust(H*Delta_a - Delta_g_FD))
```

禁止裸用 `1/d`，以免极短弦长放大FD噪声。`<=0.15σ` 是主机制子集；中、长弦只作分层训练
弱权重和外推诊断，不得掩盖小半径结果。

`d0`、`w_max`、`epsilon`、距离坐标、batch权重归一化公式、三个距离桶边界及实际权重分布必须
写入training args、checkpoint和summary，validator按保存值复算，禁止依赖脚本隐含默认值。

pair midpoint 可以将两端力臂减半，但依赖两次repeat联合信息，部署时不可得。默认可部署模型
仍以当前Actor center为 `a_ref`。中点版本只允许作为明确标记的
`MECHANISM_ORACLE_NON_DEPLOYABLE` 上界，不能进入最终Actor/Critic输入契约。注意 secant 约束
`H(a1-a0)=g1-g0` 本身与参考点无关；中点主要影响端点局部展开，而不是这条等式。

## 5. 训练目标

主模型输出标量 Q，而不是只回归一个孤立 gradient：

```text
L = lambda_Q0       * L_value_at_ref
  + lambda_g0       * L_small_radius_FD_gradient
  + lambda_Qprobe   * L_0.05sigma_probe_value
  + lambda_chord    * L_weighted_repeat_response
  + lambda_norm     * L_gradient_norm_calibration
```

- `L_value_at_ref`：监督参考点处 transformed reward；
- `L_small_radius_FD_gradient`：只用单口径 `0.05σ` gradient，不混入 `0.10/0.20σ`；
- `L_0.05sigma_probe_value`：直接监督标量局部Q，保证其不仅方向正确；
- `L_weighted_repeat_response`：监督 `H*Delta_a` 对应的梯度变化，小弦长优先；
- `L_gradient_norm_calibration`：避免再次出现幅值坍缩。

`0.10/0.20σ` probe 只作外推诊断，不进入 derivative/H 主监督。same-center consistency 在主
pilot 中是验收 gate，不再使用大权重 invariance loss；§11.20 已证明后者会把模型重新推回
action-invariant 平均解。

当前每个物理状态只有一条repeat chord，完整16x16 Hessian不可辨识；Hadamard value probes也只
给有限方向的二次响应。因此本 pilot 判断的是“受约束结构能否改善可用 action gradient”，不是
宣称恢复了真实完整 Hessian。必须记录 `d/lambda` 分布、有效秩与不同seed的一致性，防止用一个
任意H解释有限观测。

## 6. 预注册验收

所有arm使用同一checkpoint选择口径，至少报告：

1. 全600 fresh-FD：gradient cosine median/P10、positive fraction、norm ratio；
2. 300 repeat pair：response cosine、70个反转召回、same-absolute-center cosine P10；
3. 分弦长：`<=0.15σ` 的152对/33反转为主gate，另报中长弦外推；
4. 标量Q：`0.05σ` probe value RMSE、pair sign accuracy、局部候选 regret；
5. component：前4个steering knot sign accuracy与16维总体alignment；
6. 结构：H symmetry误差、解析gradient/autograd误差、有符号特征值和有效秩；
7. 稳定性：3 seed逐项结果、median/range，禁止只选最好seed。

seed qualification按完整模型而不是按指标拼接：每个seed先独立计算全部gate；一个arm至少`2/3`
seed完整通过才算机制通过。剩余seed必须在与H0同seed、同600 context的paired bootstrap中不出现
显著退化；ensemble只单列报告，不参与qualification，避免ensemble掩盖不稳定seed。

Actor更新仍采用既有最低门槛：fresh cosine median `>=0.70`、P10 `>=0`、norm ratio median
处于`[0.50,2.00]`；并新增结构pilot门槛：小弦长反转召回 `>=0.50`、same-center cosine P10 `>=0`、
symmetry/autograd误差 `<=1e-6`。任一门槛失败，qualification 保持
`STRUCTURED_LOCAL_Q_CRITIC_FAIL_ACTOR_FROZEN`。

norm还必须报告ratio P10/P90、`abs(log(ratio))` P90及ratio `<0.5`/`>2.0`帧比例；overshoot尾与
幅值坍缩同等记录。初轮不为norm P90任意发明绝对阈值，而是要求其相对H0 paired-bootstrap不显著
恶化。

小弦长recall与fresh P10分工不同：70个反转中只有33个在`<=0.15σ`，其50% recall的95%统计
波动约为`±17%`，只判断局部H通道是否生效；另37个中长弦反转受高阶污染，recall低是二次外推
失效的预期诊断，不单独否决pilot。真正的部署前守卫仍是600-context fresh P10 `>=0`。

## 7. 执行顺序

1. 先实现共享 `StructuredLocalQ` 和零训练单元测试，验证H符号、对称性、Q/gradient解析式；
2. 用tiny-set做H0/D/D+R1/D+R2过拟合，要求可训练H组的value、gradient、chord response均可拟合；
3. 全internal-fit运行四组容量×3 seed，不更新Actor；
4. 在同一已消费fresh-FD/repeat artifact上独立复算gate；
5. 只有联合gate通过才设计受限Actor step；否则根据小弦长与中长弦差异决定补action-location
   数据还是调整H容量，不再盲扫cross-anchor/invariance权重。

## 8. 实施与验证结果（2026-08-14）

### 8.1 实现和结构单测

实现文件：

```text
car_foundation/car_foundation/mppi_proposal_policy.py
scripts/model_verify/train_mppi_structured_local_q_critic.py
scripts/model_verify/validate_mppi_structured_local_q_module.py
scripts/model_verify/analyze_mppi_structured_local_q_pilot.py
```

`TorchMPPIStructuredLocalQCritic` 已按 §2 实现，不含候选 action encoder。H0、D、D+R1、D+R2
的解析式单测全部通过：H symmetry 最大误差为0，`autograd(dQ/da)` 与 `g0+H*delta` 最大误差为0；
D/D+R1/D+R2 均能构造负特征值。H0 与 D 参数量相同（613153），rank-1/rank-2 分别为
616434/619715，容量增量很小。

tiny-set 的32个repeat-pair过拟合先证明结构可学习。以反转平衡的D+R2压力测试为例，训练集自身
gradient cosine median达到0.970、norm ratio median 0.912，cross-target cosine 0.994，16个
反转pair召回0.563。因此全量失败不能归因于解析链路、H符号限制或基本优化器失效。

### 8.2 全量四臂三seed

全量命令使用4590个train context、1110个internal validation context和600个已消费fresh-FD
context；Actor全程冻结，没有新增DBM rollout。关键参数为180 epochs、lr `5e-4`、
`gradient/cosine/chord=2/1/100`，弦权重`d0=0.15,w_max=4,epsilon=0.02`。完整产物：

```text
outputs/mppi_proposal/structured_local_q_full_20260814_v1/summary.json
outputs/mppi_proposal/structured_local_q_full_20260814_v1/analysis.json
```

下表均为3 seeds的`min / median / max`；P10和same-center越大越好，norm目标区间为
`[0.5,2.0]`：

| arm | fresh cosine median | fresh cosine P10 | norm median | `<=0.15σ`反转召回 | same-center P10 | 完整gate |
| --- | --- | --- | --- | --- | --- | ---: |
| H0 | 0.637 / 0.674 / 0.719 | -0.937 / -0.936 / -0.935 | 0.528 / 0.619 / 0.630 | 0 / 0 / 0 | -0.546 / -0.504 / -0.335 | 0/3 |
| D | 0.465 / 0.527 / 0.607 | -0.945 / -0.932 / -0.923 | 0.439 / 0.501 / 0.504 | 0.015 / 0.015 / 0.030 | 0.236 / 0.253 / 0.350 | 0/3 |
| D+R1 | 0.647 / 0.655 / 0.670 | -0.935 / -0.933 / -0.921 | 0.400 / 0.449 / 0.466 | 0.061 / 0.136 / 0.136 | -0.042 / -0.000 / 0.137 | 0/3 |
| D+R2 | 0.683 / 0.715 / 0.721 | -0.950 / -0.932 / -0.908 | 0.459 / 0.485 / 0.567 | 0.152 / 0.197 / 0.288 | -0.626 / -0.304 / 0.036 | 0/3 |

独立summary/manifest审计复算全部gate，12个checkpoint及三项输入SHA256全部一致，gate mismatch
为0。没有任何arm达到2/3逐seed完整通过，所以“剩余seed相对H0的paired bootstrap非劣”前置条件
未触发；不能用bootstrap或ensemble挽救一个逐seed gate已经失败的模型。

### 8.3 结论与下一步路由

本轮qualification为：

```text
STRUCTURED_LOCAL_Q_CRITIC_FAIL_ACTOR_FROZEN
```

结论不是“二次结构完全无效”。D+R2相对同seed H0的fresh median cosine增量为
`-0.003 / +0.009 / +0.084`，小弦长反转召回从0提高到0.152--0.288；模型也实际学出了负H
特征值。因此显式action-location response通道确实开始工作。可是其收益没有传到尾部：fresh P10
相对H0的配对中位增量只有+0.003，仍约为-0.93；小弦长反转召回远低于0.50；D+R2还有2/3
seed的norm median低于0.5、2/3 seed的same-center P10为负。只看median会误判为通过。

H0能学到当前anchor处的`g0(s)`，却按构造无法响应action移动；D只改善same-center指标并明显损伤
fresh median，说明逐维对角响应不够。rank-2能恢复中位方向和少量反转，但一条repeat chord不足以
稳定辨识16维局部响应，仍存在严重的跨action-location尾部混叠。结合tiny-set PASS，更符合
“全量state/action-location条件化和局部方向覆盖不足”，而不是结构公式、autograd或网络基本容量
错误。

因此下一步不更新Actor，也不继续盲扫H rank或loss权重。优先对现有hard manifest生成同状态、
多个小半径且方向独立的action-location探针，使每个hard state不再只有一条chord；先做标签矩阵的
有效秩/条件数和可恢复H-response oracle分析，再决定最小新增rollout设计。新数据仍需保留
`<=0.15σ`主机制口径，D+R2作为当前最有希望的结构臂，H0作为同loss对照。

## 9. Chord SVD、输入语义与 Local-H oracle（2026-08-14）

### 9.1 审计契约与产物

按§8.3先完成零rollout审计，没有更新Actor，也没有加载formal validation/test：

```text
scripts/model_verify/analyze_mppi_chord_geometry_local_oracle.py
scripts/model_verify/validate_mppi_chord_geometry_local_oracle.py
outputs/mppi_proposal/chord_geometry_local_oracle_20260814_v3/analysis.json
outputs/mppi_proposal/chord_geometry_local_oracle_20260814_v3/validation_summary.json
```

使用2295个internal-train repeat pair选择KNN bank，555个internal-validation pair选择
`K/ridge/response_scale`，最后把train+validation共2850个pair作为bank，在300个已消费
internal-selection fresh-FD pair上只做一次heldout评价。KNN只能使用repeat-invariant的
history/reference/current；目标pair按episode split从bank排除。oracle从一个端点的真实gradient预测
另一端，专门审计`H*Delta_a`可恢复性，属于不可部署机制上界。

独立validator重新计算SVD、三种oracle的all/small指标和gate，并校验五项输入SHA256；最大指标误差
`2.98e-8`、SVD误差0、所有hash/count/gate一致，qualification为PASS。v1/v2属于实现过程中的
探索产物，后续引用只使用v3。

### 9.2 输入语义结论

同一物理snapshot的repeat中，history/reference/current逐位相同，最大差异均为0；但当前Critic
实际使用的first-pass派生输入明显变化：guided-anchor归一化RMS中位0.0668、feedback中位0.713、
gradient-context中位0.315，alpha center和Actor absolute center的物理sigma RMS中位均约0.148。

当前`TorchMPPIStructuredLocalQCritic.local_parameters()`签名只含
`history/reference/current/anchor/feedback/gradient_context`，没有显式absolute Actor center。
因此网络只能从repeat-varying的first-pass变量间接猜`Q0/g0/H`的展开位置。输入语义清理建议成立：
后续模型应将physical state、固定canonical origin和evaluation absolute action分离，移除
feedback/gradient-context/alpha-derived nuisance对`H`的隐式定位作用。同一physical state和同一
absolute action必须得到一致Critic输入。但这项清理本身尚不足以授权重训Actor，因为下面的oracle
heldout gate仍失败。

### 9.3 Chord geometry / SVD

每个精确physical state只有一条repeat chord，因此exact-state action excitation rank最多为1。
跨物理状态做KNN pooling后数值秩看似足够：heldout的K=32邻域全部numeric rank 16，
`s_min/s_max`中位0.111、condition number中位8.98；但stable rank中位只有4.19、entropy effective
rank中位9.98。也就是说全秩来自跨状态借方向，能否使用取决于邻域内H是否可迁移，不能把它误写成
“每个状态已有16个方向”。

方向能量也不均匀。fit bank中acceleration/steering平均能量约61%/39%；前3个steering knots
（flat index 1/3/5）合计能量均值12.9%、中位8.0%，低于各维均匀时的18.75%，P10仅1.07%。这为
后续优先补front steering 0--2提供了数据依据，但仍需保留少量全16维正交方向，避免再次形成局部
盲区。

### 9.4 KNN Local-H oracle

比较diagonal、signed-symmetric和unconstrained局部response。对称oracle在internal-validation选择
`K=128, ridge=1e-4, response_scale=1.25`；小弦长validation结果很好：cross-target cosine
median/P10为0.984/0.595、37个反转召回70.3%。但同一配置在heldout fresh-FD小弦长152对/
33反转上降为：

| 指标 | symmetric H heldout `<=0.15σ` |
| --- | ---: |
| `Delta_g` cosine median / P10 | 0.968 / 0.454 |
| `Delta_g` norm ratio median | 0.868 |
| cross-target cosine median / P10 | 0.986 / -0.167 |
| 反转召回 | 47.0% |

方向预测本身已经很好，但幅值/跨极值位置的尾部校准不足，导致cross-target P10和50%反转门槛均
失败。全300对结果更差：cross-target P10 -0.249、70个反转召回41.4%。unconstrained H没有救回
heldout尾部（小弦长P10 -0.244、召回40.9%），所以失败不是H对称约束造成；diagonal更差
（小弦长median/P10 0.829/-0.587、召回9.1%），说明跨action通道耦合确实必要。

最终qualification为：

```text
LOCAL_H_ORACLE_FAIL_EXACT_STATE_RANK1_TARGETED_PROBES_JUSTIFIED
```

解释不是“现有方向完全随机”：跨状态KNN足以恢复多数小弦长`Delta_g`方向，并在internal-validation
通过；但每个精确状态仍只有rank-1，迁移到heldout时幅值和反转尾部不稳定。因此现在有充分理由
增加**同状态、小半径、多方向**DBM probes，而不是继续增加无关随机state，或只扫网络/loss。

下一数据pilot优先使用hard-state manifest并配matched easy control；重点覆盖front steering knot
0--2的antithetic方向，同时加入少量全16维Hadamard/DCT正交方向。采集验收必须直接看每状态rank、
condition number、heldout Local-H P10和反转召回。semantic-clean Critic在这批标签就绪后作为并行
必要修复；在oracle heldout和新Critic联合gate通过前，Actor继续冻结。

## 10. g0 邻域离散度与可学习性审计（2026-08-14）

### 10.1 审计契约

在新增DBM rollout前，完成§9遗留的g0状态侧零成本审计。Actor保持冻结，未加载formal validation或
test；超参数只在4590个internal-train context和1110个internal-validation context上选择。选择完成
后以train+validation共5700个context作为KNN bank，只在已经消费的600-context internal-selection
fresh-FD上评价一次。产物和独立复算为：

```text
scripts/model_verify/analyze_mppi_g0_learnability.py
scripts/model_verify/validate_mppi_g0_learnability.py
outputs/mppi_proposal/g0_learnability_audit_20260814_v1/analysis.json
outputs/mppi_proposal/g0_learnability_audit_20260814_v1/g0_priority_manifest.json
outputs/mppi_proposal/g0_learnability_audit_20260814_v1/validation_summary.json
```

比较两种语义空间：`physical-only`只含history/reference/current；`action-aware`在相同物理输入上
显式加入absolute Actor center，不使用alpha、feedback或gradient context。internal-validation最终
选择physical-only的`K=64, temperature=0.05`与action-aware的
`action_weight=0.75, K=64, temperature=0.10`，二者均采用unit-mean-rescaled聚合。独立validator
校验九项输入SHA256、split/count/manifest/routing，并重算核心指标；最大误差为`2.98e-8`，最终
qualification一致。

### 10.2 Heldout结果

| 方法 | cosine median | cosine P10 | 正向比例 | norm ratio median |
| --- | ---: | ---: | ---: | ---: |
| 当前B4 Critic ensemble | 0.613 | -0.932 | 0.625 | 0.425 |
| physical-only KNN | 0.488 | -0.934 | 0.608 | 0.809 |
| physical + absolute action KNN | 0.593 | -0.919 | 0.622 | 0.814 |

absolute action相对physical-only将median提高0.106、P10提高0.015，说明action location是必要条件；
但固定KNN规则仍未超过当前Critic的median，P10仍接近-0.92，不能据此授权Actor更新。当前Critic的
225个hard context上，action-aware KNN的median/P10为-0.570/-0.954，正向率仅35.6%。

action-aware top-20邻域的标签coherence中位仅0.351，hard subset为0.310；邻居两两标签cosine中位
仅0.077，251/600帧低于0.30。相反，使用目标fresh-FD标签事后挑选top-20中最匹配邻居的
`oracle-best`，600帧cosine median/P10为0.969/0.852，最小值0.261，正向率100%；225个hard
context也全部能找到正向邻居，median/P10为0.961/0.854。这不是可部署方法，也不是严格的模型
上界，因为它使用了query目标标签；它只证明“有用方向已存在于局部bank中，但当前相似度和平均规则
无法从互相冲突的分支中选对”。

### 10.3 归因与下一步路由

最终qualification为：

```text
G0_INFORMATION_PRESENT_BUT_FIXED_NEIGHBOR_RULE_FAILS_TAIL
```

因此g0侧优先做semantic-clean的`physical state + explicit absolute action`条件化重训，并监控
fresh cosine P10、norm与hard manifest，而不是把所有问题都归因于“样本总数不足”。分层结果显示
cold-start最难（median 0.170、正向率51%），clipped帧median为-0.357；2.0--2.8 m/s各档median
约0.39--0.49且P10均为负，可作为表示/训练诊断的优先切片。

这不取消H侧的定向采集结论。H-only oracle在相同数据上的小弦长反转召回为47.0%，而D+R2神经
Critic仅约19.7%；前者仍不过gate，说明exact-state rank-1的数据缺口真实存在，后者与oracle之间的
差距又说明网络尚未榨干已有信息。路由重叠中，action-aware KNN负向227帧、H-oracle cross负向79帧，
二者共同失败34帧。下一采集pilot将这34帧作为最高优先级stratum，同时保留其他H/rank hard帧和
matched easy controls，不能只采这34帧形成偏置小集合。

采集设计仍要求同状态、多个方向独立的antithetic probes，并保留全16维正交覆盖。半径口径必须在
manifest中写清：若`r`表示center到单侧probe的距离，为保证完整chord `<=0.15σ`，主半径应为
`0.05--0.075σ`；若配置值表示完整chord，则两端各为其一半。逐状态保存设计矩阵、numeric/stable
rank、条件数和reward变换口径。semantic-clean重训与targeted H probes是并列补救，任一项单独完成
都不授权Actor，仍需通过fresh g0 tail与Local-H heldout联合gate。

## 11. 执行计划修订：并行语义重训与定向 probe（2026-08-14）

### 11.1 联合准入门槛

第三阶段不得只检查尾部。预注册的完整门槛为：Critic fresh-FD cosine median
`>=0.70`且P10 `>=0`，predicted/true gradient norm-ratio median位于`[0.5,2.0]`，
Local-H heldout oracle cross-target cosine median `>=0.90`，小弦长真实反转召回
`>=0.50`，same-center cosine P10不为负，并且至少2/3 seed逐模型完整通过。这里的
`oracle median`专指不向模型提供目标标签的Local-H机制审计；g0 top-20
label-aware oracle使用query目标标签，只能说明信息headroom，不能作为部署准入门。

### 11.2 两条并行轨道

第一阶段和第二阶段不再串行：

```text
Track A: existing-data semantic-clean Critic A/B/C, zero rollout
                                      \
                                       -> Track C: new-probe Structured Local-Q retrain
                                      /
Track B: targeted same-state multidirectional DBM probes
```

Track A固定三臂、相同结构/loss/split和3 seed：A为
`physical state + explicit absolute action`，B在A上增加first-pass feedback，C在B上
增加中等权重same-center invariance。invariance只比较相同physical state、相同absolute
action下不同first-pass context的`Q/g0/H`，禁止跨不同center施加。若C改善但因果归因仍不清，
再增加`A + invariance`作为第四个2x2对照，不在三臂结果前临时改变主实验。

Track B使用互斥stratum，避免重复计数：

| stratum | 数量 | 角色 |
| --- | ---: | --- |
| g0与H联合失败 | 34 | 最高采集优先级；其中仅20帧为当前Critic-hard |
| 仅H-oracle失败 | 45 | H多方向probe第二优先级 |
| 仅KNN-negative且Critic-hard | 125 | 困难表示/覆盖交界层 |
| 仅KNN-negative且Critic正常 | 68 | 网络已超过固定KNN的正对照 |
| g0/H均正常但Critic-hard | 66 | Track A高置信训练失败层，不作为“缺数据”直接证据 |
| g0/H/Critic均正常 | 262 | matched easy controls候选 |

34个联合失败帧的定义是`g0 KNN-negative AND H-oracle cross-negative`，不是“34个Critic
尾部帧”；其中20个Critic-hard、14个Critic并不hard。生成器必须在manifest中保存互斥
`stratum_id`、原始布尔标志和priority rank，不能按`34 -> 79 -> 227 -> 145`重复采集。
跨层诊断集合`Critic-hard AND KNN-positive=80`仍保留，但它由H-only中的14帧和
g0/H均正常的66帧组成，不能作为第七个互斥stratum再次计数或采集。六层总数严格为
`34+45+125+68+66+262=600`。

Track C只有在A/B两条轨道均完成并分别通过独立validator后才启动。Actor在全部联合gate
通过前持续冻结，formal validation/test保持封存。

## 12. 三轨执行结果：语义清理、定向 probe 与联合重训（2026-08-14）

### 12.1 Track A：existing-data semantic-clean 三臂

已实现显式absolute Actor center输入，并使semantic-clean模型不消费gradient-context。三臂均使用
相同D+R2结构、loss、split和3 seed：`PA=physical+absolute action`、`PAF=PA+feedback`、
`PAF_INV=PAF+same-center invariance`。结构验证确认PA对feedback严格不敏感，PAF会响应feedback，
两者均忽略gradient-context并响应absolute action。权威产物为：

```text
outputs/mppi_proposal/semantic_structured_local_q_20260814_v1/summary.json
outputs/mppi_proposal/semantic_structured_local_q_20260814_v1/analysis.json
```

下表报告每项指标的跨seed中位数；fresh-FD仍是原600-context consumed internal-selection：

| arm | fresh cosine median | fresh P10 | norm ratio median | 小弦反转召回 | same-center P10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PA | 0.659 | -0.958 | 0.586 | 0.212 | 0.035 |
| PAF | 0.687 | -0.947 | 0.594 | 0.182 | -0.134 |
| PAF_INV | 0.696 | -0.950 | 0.590 | 0.182 | -0.199 |

三臂均为0/3完整gate通过。feedback略改善部分中位方向，但没有稳定改善反转，且same-center尾部更差；
invariance也没有修复P10。因此输入语义清理是必要工程修复，但不是hard-state尾部的充分修复。Track C
选择最少nuisance且same-center跨seed中位唯一非负的PA，不再把feedback/invariance一起带入，避免因果
解释继续混杂。Actor未更新。

### 12.2 Track B：互斥路由与同状态多方向 DBM 标签

互斥600-row路由manifest与独立validator已完成：

```text
scripts/model_verify/build_mppi_targeted_probe_manifest.py
scripts/model_verify/validate_mppi_targeted_probe_manifest.py
outputs/mppi_proposal/targeted_local_probe_manifest_20260814_v1/manifest.json
outputs/mppi_proposal/targeted_local_probe_manifest_20260814_v1/validation_summary.json
```

pilot选择全部79个H失败状态（34 joint + 45 H-only）和21个确定性matched-easy controls。标签生成器在
每状态使用16条Hadamard全维方向及front-steering flat 1/3/5三条额外方向；单侧半径为
`0.05/0.075 sigma`，完整antithetic chord不超过`0.15 sigma`。每个外层action location再使用
`0.01 sigma`、16维Hadamard antithetic fresh-FD bank得到局部gradient。Actor全程冻结，只调用
deterministic forward DBM cost，不使用DBM解析梯度。

```text
scripts/model_verify/generate_dbm_targeted_local_response_labels.py
scripts/model_verify/validate_mppi_targeted_local_response_labels.py
outputs/mppi_proposal/targeted_local_response_labels_20260814_v1/summary.json
outputs/mppi_proposal/targeted_local_response_labels_20260814_v1/validation_summary.json
```

共100状态、77个外层location、每location 33个FD候选，即254,100次新DBM direct rollout。逐状态
outer numeric rank最小16、stable rank中位9.50、condition number中位1.41/最大2.43；outer裁剪
比例0.36%、inner裁剪0.22%。独立validator重建bounds、outer/inner geometry、normalized action、
局部最小二乘gradient/curvature，并重放1,024个DBM cost，全部误差为0；所有source/checkpoint hash、
split和sealed contract均通过。

### 12.3 Track C：target-only 联合重训

联合重训只把79个`pilot_role=target`状态的77个location用于监督，共6,083个state-location样本；
21个matched-easy control从不参与训练。每个location直接监督value、gradient、cosine和norm，并用
同半径正负partner的`H delta_a = delta_g`监督signed symmetric H。为避免把已消费状态冒充fresh，
fresh gate按完整repeat episode划分：79个目标帧涉及370个fresh帧，真正未触及的完整episode子集仅
230帧。它仍只是偏置的机制门，不能替代新采formal/generalization validation。

```text
outputs/mppi_proposal/targeted_semantic_structured_local_q_20260814_v1/summary.json
outputs/mppi_proposal/targeted_semantic_structured_local_q_20260814_v1/analysis.json
outputs/mppi_proposal/targeted_semantic_structured_local_q_20260814_v1/targeted_response_comparison.json
```

PA三seed均未过完整gate。未触及230帧的跨seed中位结果为：fresh cosine median 0.695、P10
-0.929、norm ratio median 0.766、小弦反转召回0.091、same-center P10 0.333。中位与幅值不再是主要
失败项，但负尾和反转仍远低于门槛；全600帧的mixed-seen结果也只有cosine median 0.637、P10
-0.927、小弦反转召回0.197，不能用seen改善掩盖。

在完全相同的targeted-response标签上，对重训前PA和重训后PA做逐seed配对比较：

| 子集 | cosine median增量 | cosine P10增量 | norm median增量 | 反转召回增量 |
| --- | ---: | ---: | ---: | ---: |
| 79个seen target | +0.663 | +0.415 | +0.188 | +0.152 |
| 21个unseen matched-easy control | -0.100 | +0.066 | -0.025 | -0.032 |

seen target绝对指标的跨seed中位达到cosine median 0.920、P10 -0.549、反转召回0.490，说明标签与
训练链路确实能把指定状态拉近；但unseen control的绝对结果仅为cosine median 0.703、P10 -0.822、
反转召回0.366，而且相对baseline没有稳定增益。这排除了“新probe完全无效”，同时否定“79个定向
状态已足以让state-to-response映射泛化”的结论。最终qualification为：

```text
TARGETED_STRUCTURED_LOCAL_Q_CRITIC_FAIL_ACTOR_FROZEN
```

### 12.4 当前归因和下一步

当前证据支持“局部监督可学习、跨状态选择/泛化仍失败”。Track B解决了exact-state rank-1几何缺口，
Track C也能在seen hard状态上显著提高梯度方向；但把79个状态的全部77个location每epoch重复训练，
相对4590个base state形成明显state imbalance，结果更像局部记忆，matched control没有同步改善。
因此下一步不应直接解冻Actor，也不应立刻盲目扩大同一状态的probe数量。

优先做零rollout的hard-state grouped cross-validation和state-balanced loss：按物理state而不是
state-location均匀采样，每epoch每状态抽少量location；在79个H-failure状态内部预留完整state fold，
同时保留21个matched control。若heldout hard-state fold仍无增益，主因是state representation/
regime条件化，而不是单状态方向不足，应增强可区分context后再训；若hard-state fold可迁移但覆盖不足，
再按互斥manifest扩展更多状态，而不是继续加每状态方向。任何下一轮仍需报告seen、unseen-hard、
matched-control和全新episode四个口径。Actor、formal validation和test继续冻结。
