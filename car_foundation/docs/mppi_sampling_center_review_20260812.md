# MPPI 采样中心策略训练评审（当前路线，2026-08-12 起）

> **历史归档**：原 preamble、§1–§11.37 及四个作废实验 block 见
> `mppi_sampling_center_review_archive_20260812.md`。
>
> **当前唯一权威入口**：§11.80。§11.57 是 2026-08-18 的中间 consolidation 快照。
>
> **仍生效的归档合同/证据**：
> - §8.4：默认 rollout 预算修正为 256 而非 64；
> - §9.7：首闭环 pilot stage cost −4.12%、约 241 ms，限定为单 seed 单场景；
> - §10：100kph 数据独立管理，不混入现有 split。

---


### 11.38 路线决策：Critic-gradient 降级，转向近端 Offline Search Distillation（2026-08-17）

当前资格记为 `CRITIC_GRADIENT_MAINLINE_DEMOTED_PROXIMAL_SEARCH_DISTILLATION_NEXT`。六轮
统一 grouped-CV 证据已经足以停止在同一数据上继续扫描 gradient loss、ranking/invariance、
Structured Local-Q、Hessian rank、dropout 或普通 encoder；Critic 后续仅保留为可选的离线候选
粗排序器，不再承担 Actor gradient provider。

Search 路线不是未验证的新想法，历史边界必须保留：T1 已使用每状态约 2048 rollout 的多起点
CEM，1800-state BC 仅恢复 teacher gain 的约 `31.7%`；plain J16 蒸馏 validation `42.517`，
扩到 3150 状态并加 cost-sensitive loss 后最好仍为 `28.675`，且 P05/worst 为
`-74.819/-369.475`。相邻保存状态的 J16 knots 有 `68.8%` 变化超过 `1 sigma`；multi-elite
perfect selector 相对单 T1 teacher 仅额外改善 `0.026~0.034`。因此不能把“搜索到更好 teacher”
等同于“Actor 一定更好”，也不重新执行一次远距离全局 J16 MSE。

新的执行合同聚焦**近端、Actor-visited、可迭代**：

1. **零 rollout coherence**：复用现有 T1/J16，以
   `delta_a_star=(a_star-a0)/sigma` 而非 absolute center 为目标，排除同 episode 邻居，比较
   optimal-residual 与 gradient 的邻域 coherence、front steering knots 0--2、局部方差、
   basin/mode 数和 cost gap。Actor 合同必须是 `pi(s,a0)->delta_a`，显式条件化当前 warm/Actor
   center；Search 不学习完整梯度场，但不能省略决定局部 basin 的 anchor。
2. **小规模预算曲线**：先用 300--600 个 train-only 状态比较旧 T1、fresh-FD+trust line search
   和确定性 antithetic multi-query search；扫描 `32/64/128/256` 总候选与 1--3 轮更新。每轮
   保留 warm、当前 Actor 和历史 best，使 direct teacher cost 不劣于基线；先找到最小有效预算，
   不直接生成 `4590x64` 后才判断 teacher 质量。
3. **近端蒸馏与迭代**：按 episode-grouped cross-fit 分开报告 baseline J、teacher J、train Actor J、
   heldout Actor J、median/P05/worst/速度与 recovery 分层。headroom 使用聚合 paired gain，
   避免对 `J_base-J_star` 近零的帧逐帧求比例。第一轮 heldout 有稳定收益后，围绕冻结候选 Actor
   重新 search/蒸馏 1--2 轮，形成离线 dataset-aggregation/policy-iteration，而不是一步跳到 J16。
4. **部署传导**：候选 Actor 在线仍只前向一次，并保留 warm + Actor two-center guard。只有
   episode-heldout 单步 tail gate 通过后才做固定 DBM 短闭环 A/B；再之后才讨论 Query/ONNX。

现有默认/部署 Actor 继续冻结；实验性 Search Actor 只有在 teacher 质量、target coherence 和
cross-fit 合同预注册后才允许训练。formal validation/test 继续封存，旧 consumed split 只能作
历史对照。

### 11.39 Phase 0.1 执行结果：a0 条件化残差一致性显著优于梯度标签（2026-08-17）

§11.38 合同第 1 条（零 rollout residual coherence）已执行。实现：

```text
scripts/model_verify/analyze_mppi_residual_coherence.py
outputs/mppi_proposal/residual_coherence_phase0_20260817_v1/analysis.json
```

数据为 diverse replay-label 全部 train+validation snapshot（105 episode × 20 snapshot
= 2100 个物理状态 × 2 first-pass context = 4200 context），与冻结 J16 best-found 逐状态
join；特征块沿用 §11.36 口径（six + reference[1:] 50×4 + current control + 条件 anchor 块，
逐维 IQR、块等权），排除同 episode/同 snapshot/自身；残差 `Delta_a*=(a*-a0)/sigma` 逐 knot
维归一。anchor 报告 actor（replay 标签内 bootstrap_actor_center）与 warm（anchor_center）
两种；主度量含 a0 块，without-a0 对照分离条件化贡献。没有新增 rollout，Actor 保持冻结；
此处 validation split 是历史已消费 oracle 集，仅作 consumed-data 机制分析。

headline：含 a0 度量下最近邻残差 flip 率 `0.115`，去掉 a0 块升至 `0.326`；参照的
own-center 梯度基线为 `0.43~0.46`（§11.36，600 selection context，跨数据集参照）。三条
主要证据链：

1. **条件化是真实的，不是度量循环**。without-a0 的最近邻特征距离反而更近（中位 0.26 对
   0.69），flip 却是 3 倍——anchor 信息本身在选择分支，而不是靠把邻居拉近。去除数据集公共
   残差方向后 flip 仅从 0.115 到 `0.110`，排除"全体残差共享一个系统方向"的平凡解释。
2. **不利解释被分层否定**。flip 随残差范数四分位单调下降（`0.185/0.128/0.090/0.060`），
   大幅值（远离最优、最需要移动）的状态最一致；flip 随速度下降
   （1.2→2.8 m/s：`0.135/0.137/0.158/0.086/0.062`）——历史最难的高速层反而是残差最一致的
   层。front steering knots 0--2 子空间 flip `0.343`（without-a0 `0.426`），仍是分支最重的
   子空间，与历史"负 alignment 93.41% 集中于 steering"一致。
3. **邻域可用性与相邻平滑性**。局部可用性 oracle（label 空间代理，非部署可达）k=3/5 有
   `97.5%/99.3%` 的 context 存在正向残差邻居（梯度为 79%/90%）；固定 KNN cosine median
   k=1/3/5/10 为 `0.504/0.442/0.397/0.346`。相邻 snapshot 的 J16 绝对中心跳变 >1σ 比例
   `73.2%`（复现历史 68.8% 量级），同口径残差位移约 `1.11 sigma` 对绝对中心 `1.40 sigma`，
   条件化吸收约两成跳变。warm 与 actor anchor 结果等价（flip `0.117/0.115`）。

预注册路由按规则严格执行：`k=5` 固定 KNN cosine `0.397` 差 `0.003` 未过 `0.40` 门，正式
qualification 为 `CONSERVATIVE_SUPERVISION_REQUIRED`（两条主判据大幅通过：flip
`0.115<=0.30`、a0 条件化贡献 `0.211>=0.05`）。k=1/k=3 通过而 k=5 平均稀释，与梯度侧
"oracle 高而固定平均规则失败"同型，但基线水平好得多；监督设计应按保守残差读取（小 k 检索
性质 + LCB 步长 + stay 兜底），不采用大 k 邻域平均。

限制与后续：本检查是 label 空间几何，翻译邻居残差是否真正改善本状态 cost 需要 rollout，
留待 Phase 1（§11.38 第 2 条）判定；梯度 flip 基线来自另一 context 集（600 selection，
episode 不重叠），是跨数据集参照而非严格配对；J16 为 best-found 数值最优而非全局最优；
norm 四分位、均值去除、steering 子空间为 post-hoc 稳健性检查，已在 artifact 中标注。下一步
进入 Phase 1 小规模搜索预算试验（300--600 train-only 状态、三臂、32/64/128/256 总预算 ×
1--3 轮、目标口径钉死、teacher cost 由构造 `J_teacher<=J_baseline`），在 Phase 2 双头残差
Actor 中把本节的一致性证据与方向/步长分离消融对接。Actor、formal validation 和 test 保持
冻结。

### 11.40 机制个案：近邻梯度反向源于position/vx代价项的近对消（2026-08-17）

对§11.36的翻转对做逐cost项分析，回答"为什么状态几乎相同、梯度相反"。cost为
`J=5.0*position_err^2+5.0*yaw_err^2+1.0*vx_err^2+0.1*steer_rate^2+0.05*accel_rate^2`，
各项误差相对50步参考轨迹逐帧求和。先用全部kNN对做聚合检验：起始帧航向误差符号同/反的
翻转率44.1%/45.4%、vx误差符号同/反44.4%/45.6%——**单一误差符号不解释翻转**，机制在
逐项权衡而非某个符号。

个案对：ctx 3852（G0_ONLY，vx=1.181低于参考1.20，50步横向漂移+1.19m）与ctx 3949
（ALL_GOOD，vx=1.306高于参考，漂移+1.59m），特征距离1.46、梯度cosine **-0.99**。对其
Actor center做全knot均匀±0.02转向扰动并逐项分解ΔJ：

| 状态 | 扰动 | dJ总 | dPosition(5.0) | dVx(1.0) | dYaw(5.0) |
| --- | --- | ---: | ---: | ---: | ---: |
| A(3852) | steer- | +0.488 | **-0.474** | **+0.900** | +0.062 |
| A | steer+ | +1.516 | +2.098 | -0.562 | -0.020 |
| B(3949) | steer- | +1.399 | +0.011 | +1.107 | +0.281 |
| B | steer+ | +0.729 | +1.634 | -0.780 | -0.124 |

两个要点。第一，**两个状态都处在局部极小附近**：±转向、±油门四个粗方向都使J上升
（+0.49~+1.65）——FD梯度描述的是极小附近的小斜率，符号由几乎对消的大项之差决定：
position项（幅0.5~2.1）与vx项（幅0.6~1.1）**方向相反、量级相当**。第二，**两状态对方向的
偏好排序相反**：A认为steer-更省（+0.49<+1.52）、B认为steer+更省（+0.73<+1.40）——净导数
符号翻转，梯度cosine -0.99。物理上：两车都向参考线左侧漂移（position项都想"往右修"），
但转向通过DBM的Pacejka侧偏耦合改变纵向速度损耗，A偏慢（vx项惩罚进一步减速）、B偏快
（vx项方向相反），yaw项（A航向+4.2°、B -1.4°）再叠加第三股小项。

结论：近邻反向不是"某个误差符号翻转"，而是**多代价项导数近对消下的符号不稳定**——
position/vx/yaw在极小点附近5:1:5三足拉锯，净梯度是大量级项的小差值，符号由vx偏差细微
高低、航向误差符号、vy/yawrate等共同决定。这同时解释：(1)为何43%近邻反向且|cos|大
（对消平衡两侧的净斜率天然反向）；(2)为何可观测特征分不开（AUC 0.53——决定符号的是
多项权衡的余项，非单一观测量）；(3)为何seen可记忆而heldout不可预测（余项符号依赖状态的
高维细微配置）。该结构由cost权重定义（5.0/5.0/1.0）与DBM侧偏动力学共同造成，属任务定义
层面，非Critic训练可修复；与§11.39"residual目标一致性更好"的发现互为印证——残差/搜索
目标不依赖这个近对消点的符号。Actor、formal validation和test保持冻结。

#### 11.39.1 Review 修正与 Phase 0.1b：train-only 邻居库复算（2026-08-17）

Review指出两处口径限制并经代码核实成立：其一，§11.39的KNN把4200个context合并建邻域，
`knn_by_split`只按query分组，validation query仍可选中validation邻居；其二，与梯度基线
`0.43~0.46`的比较来自另一批600 selection context，只能作跨数据集背景参照，严格配对证据只有
同数据的`0.326 vs 0.115`。

Phase 0.1b已按修正复算：3600个train context为唯一邻居库、归一化只在train上拟合、600个
historical validation context只作query，同脚本新增`--query-split`模式，产物：

```text
outputs/mppi_proposal/residual_coherence_phase0_20260817_v1/analysis_train_bank.json
```

train->validation迁移成立：含a0度量下最近邻残差flip `0.112`（pooled `0.115`），k=1/3/5
固定KNN cosine `0.494/0.430/0.397`（pooled `0.504/0.442/0.397`），局部可用性k=3/5
`97.8%/99.2%`，均值去除后flip `0.103`；速度分层保持有利
（1.2--2.8 m/s：`0.158/0.125/0.125/0.067/0.083`）。去掉a0块flip升至`0.340`、k=1降至
`0.254`。预注册路由不变：k=5 cosine `0.397`仍差`0.003`未过`0.40`门，保持
`CONSERVATIVE_SUPERVISION_REQUIRED`；validation为历史已消费集，本节是historical
diagnostic，不升级为无偏gate。结论强化为：**残差标签的邻域一致性可以从train episode迁移到
未见episode，且该迁移本身依赖a0条件化**。

#### 11.39.2 Phase 1 预算合同修订（2026-08-17，review采纳）

原"32/64/128/256 × 1--3轮、300--600状态、5--10万rollout"口径在数学上不一致：12个配置
全跑每状态合计1440次评价，300--600状态即43.2--86.4万rollout；且完整16D中心差分最少需要
33次评价（1中心+16方向×正负），32总预算下Arm B连完整FD都放不下。预算定义修订为：

- 预算`B`统一计**新增unique DBM candidate evaluation**；FD探针与line-search点全部计入；
  a0/warm/历史best的缓存cost单独报告不计入；
- Arm B/C使用相同实际总预算；Arm B预先声明FD辨识与line-search的分配；
- 一轮搜索使用同一个256-candidate嵌套bank，读前缀得到32/64/128/256，避免重复rollout。

执行改为两段。**Phase 1a（100--150个train-only状态筛配置）**：单轮扫32/64/128/256前缀；
多轮只测128/256总预算（如128=64+32+32），不在32预算下硬做三轮CEM。筛出2--3个有效配置。
**Phase 1b（300--600个train-only状态）**：只跑入选配置，总预算控制在约5--10万。状态按
episode、速度、scenario、residual norm与front-steering flip分层抽取；historical validation
状态不得用于训练或调预算。

Teacher目标固定为deterministic direct cost `J_direct(s,a)`（无reward seed、与J16/residual
coherence同口径）；部署相关`E_eps[C(a+eps)]`作为独立audit指标，用固定common-random-noise
加独立audit seeds，不与direct J混成selection score。每轮候选保留a0、warm与历史best并保证
`J_teacher<=J_baseline`；T1为高预算历史参照，须在同源状态重算direct J后才可同表比较。

Phase 1在teacher聚合headroom recovery之外新增两个门：其一**search-seed稳定性**——不同
确定性candidate bank下residual cosine、距离与cost regret的波动，防止蒸馏标签重新变成高频
噪声；其二**search后residual coherence**——teacher必须比J16更近端更连续，若cost好但
coherence退回高flip，Phase 2将重演J16蒸馏失败。front steering knots 0--2 flip（§11.39为
`0.343`）单列为tail gate。LCB步长、per-channel cap与stay fallback保留，具体LCB形式待
Phase 1给出search-seed/step分布后定义。

### 11.41 机制系统化：600状态逐项autograd审计，position是翻转的主导直接来源（2026-08-17）

用可微DBM rollout对全部600个internal-selection context的Actor center做逐cost项autograd
梯度（与fresh-FD标签交叉验证：cosine中位**0.997**、P10 0.988，审计与标签管线精确一致），
然后对全部600个最近邻对（排除同episode）做翻转归因。结果：

**分层对消度**（net范数 / 三大项范数和，越低越接近对消点）：G0_ONLY中位0.832、P25 0.530，
显著低于其余层0.90-0.94——最差层的净梯度整体上最贴近平衡点，与G0_ONLY翻转率最高
（51.2%，vs H_ONLY 48.9%、JOINT 38.2%、ALL_GOOD 36.6%）一致。

**逐对归因**（250个翻转对）：
- **68.0%的翻转对可直接归因于单一cost项自身梯度反向**（该项与邻居同名项cos<-0.3且在至少
  一侧份额>0.4）；同向对中该特征仅4.3%——判别力极强；
- 归因到具体项：**position项149/170（87.6%）**、yaw项21/170（12.4%）、vx项0；换算到
  全部250个翻转对，position直接解释的是**149/250=59.6%**，不能表述成“全部翻转的88%”；
- 叠加"主导项互换"（24.8% vs 同向18.6%）与"任一侧强对消（ratio<0.6）"后，机制族合计覆盖
  **83.2%**的翻转对（同向对41.7%），约17%暂未归因。

**机制结论（修正§11.40个案表述）**：近邻反向的主体不是"position与vx互相拉锯导致净梯度
接近零"这一种，而是**position项（权重5.0）经“action→非线性DBM轨迹→逐时刻参考位置误差”
复合映射后，其action-gradient在当前近邻尺度呈高曲率/分支翻向**：两个距离很近的状态可能
分别需要向参考线两侧修正，position梯度直接反向；vx/yaw相对平滑，但会把净梯度推近对消点，
使position翻向更容易决定总方向。§11.40的3852/3949个案是“position-vx拉锯”变体，系统审计
表明更常见的是“position自身翻向”变体。G0_ONLY对消度中位仍为0.832、仅P25降至0.530，故
**对消是尾部放大器而非所有翻转的唯一主因**；vx未触发单项归因门也不等于它对净梯度无贡献。

正式结论限定为：当前cost、DBM、时域、参考几何和center分布共同形成了不利于跨状态梯度回归
的目标几何，令own-center Critic gradient成为数值脆弱、投入产出比低的学习目标；它不等价于
“平方position cost数学上不光滑”或“任何模型都不可能学习”。这里也不再使用“同一输入多模态”
措辞，因为审计证明的是近邻方向高频变化，而非完全相同输入对应多个标签。若未来调整任务定义，
Huber只能限制大残差梯度、不能保证消除翻向；s-d/Frenet参数化也可能在参考投影分支处引入新问题。
二者只能作为独立counterfactual cost实验，并须重新预注册、重做单步与闭环验证；当前Search路线
无需先修改cost。Actor、formal validation和test保持冻结。

### 11.43 Phase 1a 执行结果：32评价即回收九成headroom，方向不可辨识但盆地软（2026-08-17）

§11.39.2的Phase 1a已完整执行。机制关联：§11.41把梯度反向归因到position项在近极小
状态的方向场多模态，与本节"teacher方向不可辨识但cost稳定"同源——0.26σ量级的小幅移动
正落在近对消区，方向非canonical；而§11.39的大幅残差（0.86σ至J16）方向一致性好，
提示方向canonicity可能随移动幅度增加，Phase 1b可直接检验。实现与产物：

```text
scripts/model_verify/run_mppi_proximal_search_phase1a.py
scripts/model_verify/analyze_mppi_proximal_consensus.py
outputs/mppi_proposal/proximal_search_phase1a_20260817_v1/
  manifest.json / labels.npz / summary.json / consensus_analysis.json / consensus_labels.npz
```

100个train-only状态按（速度、scenario、J16残差范数）等距分层抽取，全部为确定性评价；
预算按共享cache的新增unique评价计，每状态均值738（含两个seed bank与Arm B全部嵌套
prefix），目标为deterministic J_direct，anchor评价单列不计入。基线：J(a0)=17.087、
J(warm)=43.652、J16=4.742。全部arm的`J_teacher<=J_baseline`违规数为0。

主表（100状态聚合）：

| 配置 | J均值 | R_warm | R_a0 | anchor占比 |
| --- | ---: | ---: | ---: | ---: |
| Arm C 单轮 32/64/128/256 | 8.41/8.15/8.01/7.96 | 0.906/0.912/0.916/0.917 | 0.70/0.72/0.74/0.74 | 29--30% |
| Arm C 多轮 128（64+32+32） | 7.641 | 0.925 | 0.765 | 17% |
| Arm B 32/64/128/256 | 9.28/8.45/7.88/7.67 | 0.883/0.905/0.919/0.925 | 0.63/0.70/0.75/0.76 | 49/47/29/17% |
| seed bank2/bank3 @128 | 7.98/8.06 | 0.917/0.915 | 0.74/0.73 | 27/38% |

三点结论。第一，**最低有效预算约为32--64**：32次评价已回收warm->J16 headroom的90.6%，
128为91.6%，256仅再加0.1pp；自适应（多轮重定位或FD+阶梯）在128--256预算再加约1pp
（0.925）。预算曲线在64后基本平坦，Phase 1b不需要大预算配置。第二，**teacher方向不可
辨识**：不同方向基（Hadamard、两组Givens旋转基）在128预算下teacher残差方向cosine中位仅
0.36/0.02，与J16残差方向cosine中位约0.00--0.08（P10约-0.2），cost差中位0.42——软盆地中
存在多个等价好中心，方向标签不canonical。第三，**盆地足够软，回归到均值安全**：三个
128-bank teacher的逐状态均值（MSE蒸馏的收敛目标）评价为J=7.440，与best-bank的7.382几乎
相同（R_warm 0.931 vs 0.932，逐状态差中位0.0、P90 0.583），仅5/100状态劣于anchor。

对Phase 2监督合同的直接修订：蒸馏标签应使用**多bank共识中心**（或至少意识到单bank MSE
的收敛目标即共识），而不是把方向当作可辨识量；29--30%的anchor状态天然提供stay监督；方向/
步长分离中的"方向头"语义弱化为"共识方向"，主要不确定性集中在少数共识劣化状态（5%，
P90尾部）。seed稳定性gate按§11.39.2执行的结果是"方向不稳定但cost稳定"（moved-only cost
gap中位0.42，相对J约8为5%）——该gate的判定应改写为cost稳定性，方向稳定性不作为Phase 2
准入条件。高clip比例（36%）来自外环半径触界，Phase 1b使用128预算时主要环带（0.25--1.0σ）
不受影响。

限制：100状态为分层抽样而非全量；anchor为replay标签内2026-08-06期frozen actor（非当前
部署actor）；R_warm的分母里warm在抽样状态上很差（43.65），R_a0（0.63--0.77）是更严格的
参照；本节仍未训练Actor，蒸馏可学性由Phase 2判定。下一步Phase 1b按入选配置（建议
Arm C多轮128或三bank共识128，300--600状态、约4--12万评价）生成全量标签，并保留5%共识
劣化状态作为Phase 2的重点诊断层。Actor、formal validation和test保持冻结。

### 11.42 §11.41口径固化：阈值灵敏度、episode bootstrap与复现artifact（2026-08-17）

按review补齐三项检查，脚本与产物固化（`scripts/model_verify/analyze_mppi_cost_term_flip_attribution.py`
→`outputs/mppi_proposal/cost_term_flip_attribution_20260817_v1/`）。`analysis.json` format v2记录
cost权重、三个输入SHA256、全部27组阈值结果和bootstrap定义；`term_gradients.npz`记录5项×
600×16逐项autograd梯度、net梯度、对消比、最近邻、flip、单项命中标志和归因项编码。

**阈值灵敏度**（3x3x3网格，翻转对/同向对归因率）：单项翻转归因率在松/默认/严三档为
`0.692/0.066`、`0.680/0.043`、`0.632/0.034`（cos门-0.1/-0.3/-0.5，份额0.4、对消门0.6）；
份额门0.3/0.5两端为`0.764/0.123`与`0.628/0.043`。结论对阈值不敏感：翻转对与同向对的
归因率在全部27个组合中保持约60个百分点量级的分离，机制判定不依赖默认阈值。默认门下须
区分两个统计总体：翻转对内部为position **149/170**、yaw 21/170；全部pair命中项才是
position **162/185**、yaw 23/185。后者不能替代前者描述翻转机制。

**Episode cluster bootstrap**（60个episode有放回重采样，保留重复抽中的multiplicity，2000次）：
单项归因率翻转对为`68.0% [60.0%, 75.5%]`，同向对为`4.3% [2.0%, 7.0%]`；机制族覆盖率
翻转对为`83.2% [77.1%, 89.1%]`，同向对为`41.7% [32.8%, 50.7%]`。两组CI仍不重叠，
相邻帧相关性不能解释组间差异。先前内联记录的较窄区间使用`np.isin`丢失了重复episode的
multiplicity，实为cluster subsampling，已由format v2结果替代；四个点估计完全不变。

FD交叉验证复现cos中位0.997/P10 0.988；分层翻转率与对消度同§11.41（G0_ONLY_CRITIC_HARD
为51.2%/0.832，尾部最差）。因此正式qualification为
`COST_TERM_FLIP_ATTRIBUTION_AUDITED_ACTOR_FROZEN`：position复合项是当前近邻梯度翻向的
主导直接来源，机制结论通过阈值、episode相关性与独立复现三项检查，但该600-context
internal-selection审计只提供机制证据，不是新的formal qualification。它进一步支持直接比较
完整`J(s,a)`的近端Search，而不支持把position分量梯度作为新监督目标。Actor、formal
validation和test保持冻结。

### 11.44 机制-搜索联合诊断：对消度分离stay/move，position主导移动；幅度-canonicity假设不成立（2026-08-17）

§11.43提出的两个零/低成本诊断已执行。实现与产物：

```text
scripts/model_verify/analyze_mppi_proximal_mechanism_join.py
outputs/mppi_proposal/proximal_mechanism_join_20260817_v1/
  analysis.json / per_state.json
```

Part A（零rollout，Phase 1a产物内）检验"方向canonicity随移动幅度增加"：teacher移动
量化为环步长（0.264/0.529/0.793/1.057σ）。分组中位cos12为`0.360/0.643/0.162/0.929`
（n=54/13/2/2），Spearman幅度-cos12仅`+0.078`、幅度-cosJ16 `-0.013`。**假设在Phase 1a
检验的0.26--1.06σ范围内不成立**（高层样本量过小，方向非canonicity在各幅度持续）；
§11.39的残差KNN一致性与跨bank方向一致性是不同量，前者结论不受影响。另修正一个口径：
14个状态几何上移动但改善`<=1e-6`（数值平局，跨bank必同点，cos=1.0），属stay而非
mover——Phase 1a的"29% anchor"应为15个严格anchor+14个平局。

Part B（100状态逐项autograd，管线与§11.41/11.42相同）三项join结果：

| 组 | n | 对消度 | position份额 | J16残差 |
| --- | ---: | ---: | ---: | ---: |
| 严格anchor | 15 | **0.588** | 0.451 | 0.544σ |
| 数值平局 | 14 | 0.817 | 0.603 | 0.466σ |
| 真实mover | 71 | **0.907** | 0.746 | 0.948σ |
| 共识劣化（严格） | 3 | 0.814 | 0.608 | 0.442σ |
| 共识正常（mover内） | 69 | 0.907 | 0.746 | 0.971σ |

三点结论。第一，**对消度单调分离stay/move**（0.59→0.82→0.91）：搜索找不到改善的状态
正是net梯度贴近对消点的状态，§11.41的机制与§11.43的搜索行为在逐状态层面接通——近对消
处方向病态且无可用下降，搜索正确地保持不动；远离对消点时净方向明确，搜索移动。第二，
**mover的dominant项position占65/71（92%）**，与§11.41已归因翻转口径的
`149/170=87.6%`同源：position项既
制造梯度翻转、也主导可移动方向。第三，共识劣化状态（n=3，严格口径）呈现更低对消度、
更低position份额、更小J16残差的组合——靠近平衡点的平均化伤害，与机制一致但样本量
不足，只作方向性证据。

对Phase 2的可执行输出：对消度与position份额（每状态1次autograd，离线可得）作为标签侧
路由特征——低对消状态预测stay（`t_hat->0`），高对消状态允许移动；5%共识劣化层与低对消
层合并为保守监督重点层。限制：n=100、高层幅度组n<=14、共识劣化仅3例、本批状态无FD
交叉验证（管线在600状态上验证过0.997）、Part A为阴性结果应如实登记。Actor、formal
validation和test保持冻结。

### 11.45 Phase 1a review修正：co-moved口径、cost尾部与共识边界（2026-08-17）

对§11.43的三项修正经独立复算全部成立（数字逐位一致），予以采纳。

第一，**方向并非完全随机，口径改为co-moved**（比较双方都真实移动）。bank1--bank2
co-moved cosine中位`0.643`（n=67）、bank1--bank3 `0.124`（n=59）、bank2--bank3 `0.104`
（n=58），P10均为负（-0.04/-0.19/-0.16）。正确表述是"方向不唯一、部分bank分支明显
不同"：bank2（偶对Givens旋转）与bank1方向一致性明显高于bank3（奇对旋转），方向一致性
依赖基之间的相似度，再次说明方向是基相对量而非canonical量；§11.43的"0.02--0.36"是
单侧moved口径，作废。

第二，**cost稳定性必须看尾部**。co-moved cost gap中位0.36--0.74不大，但P90为
`1.93/4.00/4.23`、worst为`14.36/10.03/9.67`。方向门改cost门（§11.43）成立，但cost门
必须含P90/worst或相对regret，不能只用中位数0.42。

第三，**共识中心总体安全但有边界**。相对best-bank：P90 `0.583`、P95 `1.256`、
worst `6.074`；劣于anchor在宽松口径5/100、严格口径（>1e-6）**3/100**，最大回归
`+0.100`且全部在1.2--1.6 m/s（高速层零回归）；共识标签精确等于anchor的为**22/100**
（8等于a0+14等于warm），29--30%是单bank统计，不得混用。

同时采纳**R_a0为主指标**：warm在抽样状态上很差（43.65），R_warm放大"便宜程度"；
R_a0口径为Arm C 32/64/128/256=`70.3/72.4/73.5/73.9%`、多轮128=`76.5%`。

Phase 1b合同据此更新：主臂用**多轮128**（128次评价，R_a0=76.5%）；三bank共识
（384次评价/状态）先补测32/64低预算共识的性价比；**共识rollout未优于anchor时直接
回退stay标签**，构造零回归teacher；补齐预注册未执行的固定common-noise
`E_eps[C]` audit（selection与audit用不同CRN seed）；报告以R_a0为主，附
episode-bootstrap、速度/场景分层、P05/worst。Actor、formal validation和test保持冻结。

#### 11.44.1 Review修正：Part A收窄、Part B判别力上限与特征定位（2026-08-17）

对§11.44的修正经独立复算成立，全部采纳。

Part A收窄：co-moved子集按幅度分层后，0.264σ层bank1--2 cosine `0.618`（n=50）、
bank1--3 `0.127`（n=44）；0.529σ层`0.643`（n=13）与`-0.001`（n=11）；0.793/1.057σ层
各仅2例。准确结论是：**在有统计功效的0.26--0.53σ范围内未发现幅度增大使方向更
canonical；更大幅度样本不足，不能声称整个0.26--1.06σ假设被否定**。§11.44相应表述
作废。该结果有效关闭的是"增大步幅会自动恢复方向唯一性"的希望。

Part B收紧三点。其一，"机制与搜索行为接通"保留，但"同一件事"过强：stay/move判别力
为cancellation AUC `0.715`（最佳平衡准确率0.695）、position份额`0.794`（0.753）、
J16残差`0.830`（0.777）——有信息但分布明显重叠，不能单独可靠决定stay/move；且
cancellation在真实mover内部与幅度Spearman仅`0.088`，它区分"不动与动"，不决定
"动多少"。其二，严格anchor仍有`0.544sigma`的J16残差——不是"没有改善方向"，而是
**当前有限预算/离散bank在近对消软盆地中未找到超过阈值的改善**。其三，3个共识劣化
样本中一例cancellation/position为`0.936/0.897`，并非都属低对消低份额，n=3不能建立
"低对消劣化层"，该提法作废。

特征定位（重要）：cancellation与position份额**不作为正式标签路由的必要条件**——
它们需要可微模型，Query侧未必可微，设为必需会破坏黑盒Search路线的可迁移性；且搜索
本身已提供更可靠的黑盒依据`gain=J_anchor-J_candidate`，共识中心重rollout未达标直接
回退stay即可。二者定位为**旁路诊断字段**：分层、样本加权候选、辅助监督，不进Actor
在线输入。"position主导mover（65/71）"表述为与§11.41翻转归因**相容**的独立统计，
不构成同一证据。Phase 1b的正式stay标签由guarded direct-cost/CRN-cost margin产生，
并报告上述特征对stay的episode-grouped AUC与CI以评估样本加权价值。Actor、formal
validation和test保持冻结。

### 11.46 Phase 1b前置review：部署语义、监督形式与工程勘误（2026-08-17）

对Phase 1b启动前的review逐项核实成立并采纳。

**部署目标错位（已核实）**。`controllers_torch/mppi.py`的`sampling_mean_knots=mean_knots`
仍围绕warm running mean采样；`car_node.py`的actor序列走`hard_guard_action_sequence`
二选一。即当前运行时语义是"warm MPPI输出 vs Actor直接序列选一"，**不是**目标语义
"Actor center -> MPPI围绕其采样 -> MPPI输出"。因此：(a) Phase 1a只证明了中心作为
确定性动作序列好，未证明作为采样均值好——预注册的CRN `E_eps[C]` audit升级为Phase 1b
必做项；(b) "actor-centered MPPI + 保留warm/Actor两个未加噪确定性候选"立项为Phase 2
之后的正式部署工程项，当前hard guard只是保底不是正式实现。

**监督形式修订（替代方向/步长双头）**。Phase 1a已证明方向非canonical，继续把残差
归一化成`Delta_a=t*d`会人为制造难学标签（尤其stay附近）。Phase 2 Actor改为直接输出
唯一的16维bounded residual `Delta_a_hat=Delta_a_max*tanh(f_theta(s,a0))`，外加可选
stay logit；不再要求拟合单位方向。原"方向/步长四臂消融"作废，替换为cross-fit损失
分解四问：teacher收益、train Actor恢复率、heldout Actor恢复率、失败归因（蒸馏 vs
跨episode泛化）。teacher使用guarded consensus center。

**Actor可学性为当前最大研究阻塞**。J16蒸馏（42.517）与cost-sensitive（28.675，
P05 -74.8）的历史证明"teacher好不等于Actor好"；近端teacher+显式a0+stay标签只是合理
假设，在episode-grouped cross-fit用真实rollout评价（不用MSE代替控制效果）完成前，
不得宣称Search路线成功。

**工程勘误与修复（已落地）**：Phase 1a manifest的`arm_c_multi_unique=544`错误，真实
算法预算为128（64共享前缀+32+32），错误源于循环末尾读数被Arm B/seed bank污染；脚本
已改为在multi轮结束立即捕获。evaluate缓存已改为批内按key去重（Phase 1a ring候选
无实际重复、结果不受影响；Phase 1b高clip率下必需）。manifest新增逐状态
`anchor_checkpoint_sha256`固化a0来源（2026-08-06期frozen actor，`f51c6876...`，
不等于未来部署Actor，仅作标签anchor）。J16一律称best-found参考。三bank 128共识
约384评价/状态，先测32/64低预算共识性价比。

**修订后执行顺序**：修正预算/artifact合同（已完成）→ Phase 1b（300--600状态、
多轮128主臂、低预算多bank共识、guarded consensus、J_direct与独立CRN `E_eps[C]`
并报、R_a0/P05/worst/episode CI/速度场景分层）→ Phase 2直接16维bounded residual
Actor+可选stay head → episode-grouped cross-fit真实rollout评价 → Actor-visited
重搜1--2轮 → actor-centered MPPI接入（保留双确定性候选）→ 固定DBM短闭环A/B →
Query/ONNX。

**当前资格**：`SEARCH_ROUTE_DIRECTION_VALID` / `TEACHER_MECHANISM_PASS` /
`PHASE1A_BUDGET_PASS` / `ACTOR_LEARNABILITY_UNTESTED` /
`MPPI_CENTER_TRANSFER_UNTESTED` / `CLOSED_LOOP_UNTESTED`。方向未走错；处于
"teacher搜索有希望、策略学习与采样中心部署待验证"阶段。Actor、formal validation
和test保持冻结。

#### 11.46.1 position梯度horizon时间分解：56%整体反向，冲突质量75%在晚段（2026-08-17）

按review建议对position cost做逐horizon步分解（`scripts/model_verify/analyze_mppi_position_gradient_horizon.py`
→`outputs/mppi_proposal/position_gradient_horizon_20260817_v1/`，600 context×50步逐点
autograd，逐对标志与逐步梯度入artifact）。先答构造问题：本cost的position误差是固定
时间索引对应`trajectory[t]-reference[t]`的2D欧氏平方和，**全代码无最近点关联/Frenet
对应/索引切换**——review怀疑的reference association switching分支被构造排除。

分解结果（250个翻转对，同向对对照）：
- **翻转分类：整体反向（global_reversal）141/250（56.4%）**、mixed 95（38.0%）、局部制造
  （单段集中）仅14（5.6%：t34-50段11、t1-16段2、t17-33段1）。翻转对的逐 步对向质量中位
  **0.676**（p25 0.250/p75 0.867；≥0.6占56.4%、<0.3占27.2%），同向对中位**0.001**——
  干净的双峰，分类不依赖阈值细节。
- **horizon质量分布**：position梯度的幅值质量**74.6%在t34-50**、21.9%在t17-33、仅3.3%在
  t1-16（翻转锚与非翻转锚相同）。机制清楚：漂移误差沿horizon平方增长（例对t50横向
  0.05-0.06m量级误差对上t10的毫米级），均匀加权+平方误差使晚段天然主导；早段转向knot在
  16步内对位置几乎没有影响（例对t10的早段转向敏感度≈0.0001-0.016，t30才到0.1-0.4量级）。
- **例对3852/3949**：两状态各自三段的自身投影[0.48/6.8/24.8]与[0.39/3.2/15.2]——各自的
  逐步贡献与自身总梯度同号且晚段占绝对主导；其翻转来自晚段几何的逐步对向（属global型）。

**结论**：position翻转的主体（56%严格整体反向，其余38%多段mixed，单段局部制造仅5.6%）
是**真实的未来轨迹跟踪几何差异**，不是cost定义在孤立horizon点制造的伪影——"reference
association切换"这类可修构造问题不存在。但当前定义确有两个结构特征：(1)均匀horizon加权、
平方误差和动力学累积共同使完整action-gradient的幅值质量主要来自晚段（t34-50）；(2)2D欧氏
position把along/cross两个物理语义不同的误差合在同一项。

解释边界：这里的3.3%是**t1-16这些cost时间步**对完整16维action-gradient的质量份额，不能
解释成"早期转向knot只有3.3%的信息"。早期转向的主要后果本来就会在t34-50显现，这是长时域
MPC的正常预见性，同时也意味着结果对远期模型误差更敏感。该结果进一步支持直接比较完整
rollout cost的Search路线，但**不直接授权缩短horizon或修改晚段权重**；后两者会改变正式任务，
可能带来短视、反复修正和闭环抖动。当前Phase 1b/Phase 2继续以原始50步cost为主合同，
horizon折扣和s-d只作独立counterfactual。Actor、formal validation和test保持冻结。

### 11.47 Phase 1b 执行结果：600状态标签生成、CRN audit与共识预算研究（2026-08-18）

§11.46合同已完整执行。实现与产物：

```text
scripts/model_verify/run_mppi_proximal_search_phase1b.py
scripts/model_verify/analyze_mppi_proximal_consensus_budget.py
outputs/mppi_proposal/proximal_search_phase1b_20260818_v1/
  summary.json / labels.npz / manifest.json
outputs/mppi_proposal/proximal_consensus_budget_20260818_v1/
  summary.json / consensus_labels.npz
```

**主标签（600分层train-only状态，多轮128，均值130评价/状态）**：J(a0)=16.012、
J(warm)=39.921、J16(best-found)=4.797、J_teacher=7.887（P95 25.568）；
`R_a0=0.7245`，episode-bootstrap 95% CI `[0.684, 0.753]`；R_warm=0.912。gain中位
0.970、P05=0；严格stay（gain<=1e-6）13.8%（83状态），gain<0.01共88状态、<0.1共159
状态——Phase 2的stay阈值应从这组分布选择而非E_eps。基线违规0；anchor checkpoint
sha256（`f51c6876...`）逐状态固化；labels.npz含teacher/label knots、stay标志、
cancellation/position份额旁路诊断字段。

**CRN `E_eps[J]` audit（200状态分层子集，每组8 draws，selection/audit独立种子）**：
mover（173）边际中位selection `+1.608`/audit `+1.215`，均值`+14.86/+12.92`；但
72/173与70/173的mover边际非正，两组种子符号一致率仅`0.645`。零成本归因：跨种子
Pearson `0.745`而Spearman仅`0.392`——**大幅移状态的采样价值跨种子高度一致（均值被
大边际状态主导且可靠），小幅移状态的边际在K=8下噪声主导**；确定性gain阈值无法分离
可靠性（gain>=0.01..2.0各截断下both-positive恒0.40--0.47，spearman(gain,margin)=
0.253）。结论限定：(a)聚合层面teacher作为采样均值的期望改善为正；(b)~40%非正边际
**不能**归因为真实迁移失败，K=8对逐状态判定欠功效；(c)禁止在当前功效下用E_eps做
逐状态stay门；若需逐状态判定须K>=32重测或留待actor-centered MPPI闭环直接验证。

**共识预算研究（Phase 1a 100状态，bank2/3@32/64/128前缀+guarded回退，新增45,300
评价）**：guarded R_a0为32/64/128=`0.750/0.774/0.781`，fallback比例1%/2%/2%，回退后
最大回归`1.19e-07`（浮点零）——**零回归teacher由构造达成**。边际递减：32->64加
2.4pp，64->128仅加0.7pp；consensus_64（192评价/状态）以一半成本拿到128预算99%的
价值，与多轮128（128评价，1a口径R_a0 0.765）按预算效率接近。Phase 2标签可选用
`multi-128`（已就绪的600状态）或补生成`guarded consensus_64`，二者差异留给Phase 2
消融。

**资格更新**：`PHASE1B_TEACHER_LABELS_READY`；`ACTOR_LEARNABILITY_UNTESTED`仍为
主阻塞（Phase 2 episode-grouped cross-fit回答teacher/train/heldout三层恢复率）；
`MPPI_CENTER_TRANSFER_UNTESTED`（CRN已给聚合层正面证据，逐状态与闭环未验证）。
Actor、formal validation和test保持冻结。

### 11.48 Horizon/position counterfactual优先级与验证合同（2026-08-18）

基于§11.41--§11.42的position主导翻转、§11.46.1的晚段质量集中，以及§11.47已完成的
Phase 1b teacher，新增三项机制counterfactual。三者的定位是**解释任务几何、筛选未来cost
候选和构造旁路置信度**，不是Phase 2 Actor训练的前置阻塞项；正式Actor仍先学习原始
`J50`下的guarded label，避免同时更换teacher目标和网络后无法归因。

**优先级一：固定时间索引along/cross精确分解。** 使用reference yaw定义

```text
t_t = [cos(yaw_ref_t), sin(yaw_ref_t)]
n_t = [-sin(yaw_ref_t), cos(yaw_ref_t)]
e_parallel_t = (p_t-p_ref_t) dot t_t
e_cross_t    = (p_t-p_ref_t) dot n_t
```

从而逐点严格满足`||p-p_ref||^2=e_parallel^2+e_cross^2`，不引入最近点投影、Frenet索引切换
或新的参考关联。对149个position直接归因翻转，分别统计：(a)cross自身反向；(b)along自身
反向；(c)along/cross对消或主导项互换；并单列steering knots 0--2、horizon三段、速度与
scenario。若主体是cross自身反向，说明纠偏几何真实分支；若大量来自along/cross竞争，才为
重构position语义提供直接依据。现有NPZ只保存total position逐步梯度，不能从代数上唯一拆回
两项；需复用同600状态做两项autograd，属于低成本重算、不是新增数据rollout。

**优先级二：horizon weighting sweep。** 从已有`600x50x16`逐步梯度重组
`g_w=sum_t w_t*g_t`，比较uniform、mild/strong late-discount、middle-heavy，以及
terminal-heavy负对照。所有权重先归一到相同总质量，零rollout阶段只评价neighbor flip、
coherence、norm/cancellation和hard-state集中度；**flip下降本身不是成功门**。任何候选方向
必须再以相同normalized trust radius/line scan在原始`J50`上做DBM rollout，报告真实gain、
P05/worst和stay率。仅用`g_w`与原梯度的内积只能作一阶代理，不能称为“原J50真实改善”。

**优先级三：bounded gradient margin。** 统一采用有界且方向清楚的定义

```text
rho_time = ||sum_t g_t|| / (sum_t ||g_t|| + eps),  rho in [0,1]
```

并为position/along/cross分别计算；`rho`低表示强时间对消/低置信，不使用易发散的倒数形式。
报告其对Phase 1b strict-stay、小gain和共识fallback的episode-grouped AUC/CI，以及不同速度、
场景和front-steering tail。已有全cost cancellation对stay AUC约0.715，只说明可作诊断或样本
加权候选，不足以单独路由。margin不进入Actor在线输入，也不成为Query必须提供的量；正式stay
仍由black-box `J_anchor-J_candidate`与guarded rollout决定。

判决规则：若s-d只改善梯度一致性却不改善原始J50 rollout，不改cost；若折扣方向改善局部
coherence但损害晚段/闭环跟踪，不采用；只有counterfactual在原始J50、actor-centered MPPI和
短闭环三层均给出收益，才另立新任务定义。上述诊断可与Phase 2并行，Actor、formal validation
和test保持冻结。

### 11.49 Phase 2验证合同（终版）：四线收敛、3-fold主协议与预注册判定门（2026-08-18）

采纳review三点修正，四线合同定稿。**A0主协议改为3-fold×3seed**：manifest核实为30个
速度×场景单元×每单元恰3个episode（共90），3-fold可让每fold在每单元恰好留出1个完整
episode，分层最干净；5-fold仅作次要敏感性对齐，所有headroom与tail结论以3-fold OOF为准。

**A0标签口径**：仅用现成600状态multi-128标签；其guard基准为`min(J_a0, J_warm)`（保证不劣
于两者中更优者），不是consensus teacher。consensus_64仅在A0训练集难拟合、seed明显不稳或
方向平均化被证为主因时触发全量消融。

**恢复率计算与判定门（预注册）**：聚合求和式
`H_pi = sum_i(J_a0_i - J_pi_i) / sum_i(J_a0_i - J_teacher_i)`，禁止逐帧比例。主门：
≥2/3 seed `H>=0.50`；median-seed的episode-bootstrap 95% CI下界>0；全部seed恢复率为正，
禁止择优。tail门：`Delta J = J_a0 - J_pi`的P05不小于0并报worst。泛化差门：
`H_train - H_OOF > 0.15`。判定树：train<50%→拟合/结构/标签参数化问题；train≥50%且
OOF<50%且差>15pp→泛化问题；两者≥50%→Actor可学习性通过；OOF过而P05败→主体可学、安全
tail未过。31.7%历史J16结果仅作工程锚点，非统计基线。

**A1依赖实现**：B0保存600状态逐状态原始Jacobian/sensitivity；A1每fold只用训练episode拟
合S，heldout只应用；结构定型后部署模型再用全部600 train-only状态重估S；同时报告各fold
S差异以判断其作为部署常量的稳定性。

**B1保留门加严**：聚合teacher headroom保留≥95%且P05不恶化、worst有界、平滑性与clipping
不恶化——不得只凭平均cost通过。

**D线依赖收紧**：D1须等"A线通过**且B线参数化选择冻结**"，避免均匀knot Actor完成MPPI接入
后B2又改参数化导致部署验证重做。

执行顺序：立即并行A0（multi-128基线，3-fold×3seed）/B0（逐状态Jacobian与S原始artifact）/
C1（along/cross分解，§11.48优先级一）；A0+B0后A1→A2；B0后B1→B2；C2/C3并行非阻塞；
A线+B线冻结后D1→D2→D3→D4。Critic主线继续冻结，formal validation/test继续封存。

### 11.50 Critic最后坐标机会：Sensitivity/DCT全秩坐标配对实验（2026-08-18）

按预注册的“只改action坐标、不再调loss”合同完成最后一次Critic-gradient验证。实验复用已修复
batch对齐的scalar-Q/value-delta v2训练链，保持状态输入、数据、相对value监督、优化器、dropout=0、
checkpoint选择和160 epoch不变；运行`2 arms x 3 episode-grouped folds x 3 seeds=18`次训练：

- **A**：原始物理坐标`Q(s,a)`；
- **AT**：不重训A checkpoint，只把其梯度用`B^T`换到新度量，用于隔离“换坐标看起来更好”的
  纯度量/预条件效应；
- **Z**：真实重训`Q(s,z)`，其中`a=Bz`。`B`由每个fold的train-only状态拟合，使用固定MPPI
  channel sigma、每通道完整8阶DCT和clipped sensitivity scale，保持16维满秩，不做降维；3个fold
  的condition number均为`11.43`。

每个checkpoint除新坐标cosine外，还把`-grad_z Q`通过`B`映射回物理动作步，使用固定RMS
`0.02/0.05/0.10 sigma`在真实DBM `J50`上做确定性line search。formal validation/test未加载，
Actor未更新。三项batch/per-sample、`a=a0`零delta、置换一致性单测均通过；独立validator复核
18个checkpoint hash、源文件hash、B可逆性与contract后为PASS。

#### 11.50.1 严格pooled结果

下表每格依次为3个seed的结果；`J50 P05`取`0.05 sigma`物理步的`J(a0)-J(step)`，负值表示
真实cost恶化。

| arm | coordinate cosine median | coordinate cosine P10 | physical-step cosine P10 | early steering 0--2 P10 | J50 gain median | J50 gain P05 |
|---|---:|---:|---:|---:|---:|---:|
| A | -0.539 / -0.417 / -0.335 | -0.972 / -0.968 / -0.967 | 同左P10 | -0.997 / -0.998 / -0.997 | -6.818 / -6.966 / -6.806 | -84.03 / -87.70 / -88.43 |
| AT（仅换度量） | -0.311 / -0.173 / -0.195 | -0.832 / -0.833 / -0.810 | -0.250 / -0.256 / -0.240 | -0.701 / -0.602 / -0.683 | -0.046 / -0.120 / -0.079 | -7.10 / -7.48 / -6.95 |
| Z（新坐标重训） | -0.061 / -0.101 / -0.180 | -0.612 / -0.694 / -0.805 | -0.316 / -0.349 / -0.393 | -0.684 / -0.656 / -0.694 | -0.216 / -0.228 / -0.299 | -5.30 / -6.04 / -11.00 |

新坐标确实大幅改善了数值condition：Z相对A的中位数从强负拉到接近0，P10也从约`-0.97`
改善到`-0.61~-0.80`。但这不是可用梯度：三seed的P10仍远低于预注册`-0.2`降级门；映射回
真实action后P10仍为`-0.32~-0.39`，front steering尾部仍约`-0.66~-0.69`。最重要的是DBM
rollout给出同方向结论：即便只走`0.02 sigma`，三seed的J50 P05仍为`-1.64/-2.00/-2.90`；
`0.05 sigma`的正收益比例仅`0.327/0.366/0.346`，中位收益和P05全部为负。reversal recall在
fold/seed间从`0.032`到`0.718`大幅波动，也没有形成稳定恢复。

AT对照进一步限定归因：单纯用B预条件就能把physical-step P10从约`-0.97`提高到约`-0.25`，
说明原坐标condition确实是放大器；Z重训相对AT的coordinate P10另有`+0.220/+0.139/+0.005`
改善，但physical-step P10反而分别再变差`-0.066/-0.093/-0.153`。因此不能把新坐标下cosine
数值变好当成部署收益。

这轮不是“网络完全没训练”。Z在seen hard状态上的value-delta correlation为`0.56--0.79`，
说明scalar value目标可拟合；但同一批seen状态的autograd gradient median仅`-0.405--+0.040`
（个别run例外也不稳定），P10为`-0.54--0.85`。即：**相对value拟合并未约束出可靠的局部
导数，新坐标只改善condition，没有把hard-gradient问题变成可学习的稳定目标。**

#### 11.50.2 判定与口径修正

正式qualification为`COORDINATE_CRITIC_FAIL_CLOSE_GRADIENT_MAINLINE`。它关闭的是“保持当前
value-delta训练合同，只靠sensitivity normalization/temporal basis即可恢复Actor gradient”的
最后假设；不宣称数学上不存在另一种显式梯度监督器，但结合前面六轮loss/表示/输入实验，不再
为gradient provider继续做loss engineering。Critic仅保留为可选的candidate ranking/value辅助，
正式主线继续Phase 2 bounded proximal residual Actor与offline search distillation。

本轮同时修正一个既有通道口径：DBM/MPPI物理action顺序是
`[acceleration, steering]`，所以early-steering flat indices为`[1,3,5]`。旧
`analyze_mppi_position_gradient_horizon.py`的两个`early_steering_map_t10/t30`字段误用了
channel 0，实际量到的是acceleration；代码已改为channel 1，旧字段不得继续作为steering证据。
该错误不影响§11.43的全16维horizon质量分解、position逐时段总梯度、翻转分类或本轮结果；此前
正确使用`[1,3,5]`的hard-tail/chord/residual审计也不受影响。

可复算入口与产物：

```text
scripts/model_verify/run_mppi_g0_sensitivity_coordinate_cv.py
scripts/model_verify/validate_mppi_g0_sensitivity_coordinate_cv.py
outputs/mppi_proposal/g0_sensitivity_coordinate_cv_20260818_v1/
  summary.json
  coordinate_transforms.npz
  vq2_PA_G0*_fold{0,1,2}_seed{0,1,2}.pt
```

### 11.48 Horizon诊断：翻转主体为横向修正符号歧义；早horizon加权可降翻转但未验证；时间维对消非机制（2026-08-18）

三项与Phase 2并行的诊断（不阻塞、不改正式cost）已执行。实现与产物：

```text
scripts/model_verify/analyze_mppi_horizon_diagnostics.py
outputs/mppi_proposal/horizon_diagnostics_20260818_v1/
  analysis.json / per_state.npz
```

对600个selection context逐状态计算逐步梯度（每状态50次反传）与along/cross分量梯度；
重组校验通过（与§11.42存储net梯度cosine中位>=0.99），along/cross恒等式最大误差
<1e-4（固定时间索引+参考航向，无最近点匹配）。

**Part 1（along/cross分解，149个position-attributed翻转对）**：cross-only 51（34%）、
along-only 8（5%）、both 83（56%）、neither 7（5%）。**90%的翻转涉及cross分量**，
纯along（5%）与纯竞争（5%）可忽略——翻转不是时间索引簿记，而是物理的横向修正
方向歧义；56%的"both"表示along与cross同翻（修正向量整体反向），34%的cross-only是
经典"该往哪侧修"歧义。对s-d重构的判定：**该歧义在cross符号本身，s-d参数化不会
消除它**——s-d的价值限于语义清晰而非消除翻转；若未来动cost，方向应是软化横向项
（huber等）而非重参数化。与§11.41"任务定义层面"结论一致并进一步定位。

**Part 2（horizon加权扫描，同一近邻对、零rollout筛选）**：uniform翻转率0.412；
`exp_decay_0.90`降至0.343、`front_half`0.350、`ramp_down`0.397；其余（ramp_up/
late_half/drop_first_10/exp_growth/exp_decay_0.95）变化<=1pp。即**后段horizon贡献
携带较多翻转不稳定，压低后段权重可降~7pp**。按review边界，这只是筛选：候选权重
必须在原始uniform J50下重rollout验证真实gain/P05/worst/stay后才可讨论；降翻转可能
只是改变目标。该验证未执行，登记为pending。

**Part 3（gradient margin rho）**：中位0.887；翻转状态0.874对同向0.892——**时间维
对消不是翻转机制**（§11.41的对消是跨cost项而非跨时间步）。rho作为stay/小步样本
识别器的有效性未验证（需与Phase 1b状态join，每次约600x50次反传，登记为可选
pending）；按review边界rho只作诊断，不进Actor输入，Query侧无梯度依赖。

**边界重申**：正式Actor仍学习原始J50下的guarded teacher；只有新定义在原始J50、
actor-centered MPPI与短闭环三层都获益才考虑修改正式cost。Actor、formal validation
和test保持冻结。

### 11.51 [已作废] Phase 2 A0执行结果：train恢复<50%，判定树落入拟合/标签参数化分支（2026-08-18）

> 本节原始正文已移至归档附录 A。train 恢复 0.044 及其归因被早停修复重跑推翻；
> 当前权威结果见"§11.53 A0/A0.1早停修复与2x2重跑"。

### 11.49 [已作废] Phase 2 A0 执行结果：Actor在训练状态上也仅恢复4%——蒸馏层失败，分支路由到标签噪声（2026-08-18）

> 本节原始正文已移至归档附录 A。train 恢复率 0.044 的"蒸馏层失败"归因被早停修复
> 推翻（修复后 train 0.77）；当前权威结果见"§11.53 A0/A0.1早停修复与2x2重跑"。

### 11.52 [部分作废] A0b 2x2消融与B0敏感度：标签形式是train失败主因，泛化成为新主阻塞（2026-08-18）

> 本节原始正文已移至归档附录 A。训练结果部分作废；有效产物：
> `consensus64_labels_20260818_v1/`、`b0_sensitivity_20260818_v1/`。
> 权威训练结果见"§11.53 A0/A0.1早停修复与2x2重跑"。

### 11.53 A0/A0.1早停合同修复与2x2重跑：共识标签有效，但主阻塞为跨episode泛化（2026-08-18）

在执行A0.1前审计发现`run_mppi_actor_a0_baseline.py`的早停状态机违反§11.49合同：
`stale`在checkpoint尚不可选的epoch 1--39期间仍累加，默认`patience=25`使训练在epoch 25
终止。旧A0、A0b和A1的所有记录均出现`best_epoch_loss=Infinity`，实际评价的是epoch-25
末模型，而非预注册的epoch>=40最佳checkpoint。修复为：仅在`epoch >=
selection_min_epoch`后更新`best_loss/stale`并触发早停，同时落盘`best_epoch`、动态标签路径、
stay参考集及是否启用S归一化。修复后36/36 run的loss均为有限值，best epoch为63--120。

consensus-64标签无需重生成：现有600状态artifact含117,000次评价（195/状态），teacher
`J=7.812`、guard fallback 4/600。使用全新输出目录重跑严格配对的
`multi-128/consensus-64 x plain/stay-balanced`，结果如下。`H_train`和`H_OOF`均为
聚合求和恢复率；OOF列列出3个seed。

| 标签 / loss | H_train（median seed） | H_OOF（3 seed） | median-seed 95% CI | worst | 判定 |
|---|---:|---:|---:|---:|---|
| multi-128 / plain | 0.772 | -0.202 / -0.117 / -0.151 | 旧bootstrap作废 | -386.9 | 泛化失败 |
| multi-128 / stay-balanced | 0.774 | -0.221 / -0.161 / -0.081 | 旧bootstrap作废 | -486.5 | 泛化失败 |
| **consensus-64 / plain** | **0.856** | **0.069 / 0.150 / 0.147** | **[-0.052, 0.315]** | **-83.7** | 泛化失败 |
| consensus-64 / stay-balanced | 0.767 | 0.067 / 0.064 / 0.052 | 旧bootstrap作废 | -262.8 | 泛化失败 |

结论有三层。第一，旧“train拟合失败”是早停bug制造的假象：修复后四臂train均明显超过
0.50，multi-128本身也达到0.74--0.80。第二，**共识标签仍有独立、稳定的正作用**：plain
口径下将三seed OOF从全部负值提升为全部正值，并把worst从约-387收窄到-84，同时train从
0.77提升到0.86；所以noncanonical单bank标签确实伤害跨episode学习，但不是全部阻塞。
第三，stay-balanced在当前实现下没有收益，反而降低consensus臂的train/OOF并恶化tail；不再
作为默认loss。所有臂主门`H_OOF>=0.50`、CI下界>0及P05>=0仍失败，正式阻塞收敛为
**跨episode泛化与tail**，不是训练集容量或标签生成失败。

口径边界：另一路`train_mppi_proximal_residual_actor_phase2a0.py`的5-fold语义清理MLP结果
（train 0.044、OOF -0.004）使用不同输入、网络、stay辅助头和fold协议，只能作为替代结构
负对照，不能替代§11.49的3-fold部署Actor主协议。旧`actor_a1_c64_snorm`同样受早停bug影响，
不得引用；下一步应在修复后的trainer上重跑A1（consensus-64/plain + fold-train-only S），
再决定进入A2 early/late结构还是状态覆盖分支。Actor继续冻结，formal validation/test继续封存。

复算产物：

```text
outputs/mppi_proposal/actor_a0_baseline_20260818_v2_fixed/
outputs/mppi_proposal/actor_a0b_m128_stayw_20260818_v2_fixed/
outputs/mppi_proposal/actor_a0b_c64_plain_20260818_v2_fixed/
outputs/mppi_proposal/actor_a0b_c64_stayw_20260818_v2_fixed/
```

bootstrap勘误：初版`episode_bootstrap_ci`用`np.isin`折叠了重复抽中的episode，且分母未同步
重采样。该实现不影响表中H、P05和worst，但旧CI不得引用。修复后按episode保留重采样
multiplicity并联合重采样分子/分母，同时保存`oof_evaluation.npz`；c64/plain正确CI如表，
对应新产物为`actor_a0b_c64_plain_20260818_v3_bootstrap_fixed/`。未为已关闭的两个非主臂重训，
其CI明确标作作废。

### 11.54 A1旧执行结果（已作废）：S归一化坐标实验受早停bug污染（2026-08-18）

> 本节原始正文已移至归档附录 A。权威结果见"§11.50 A1修复版重跑与KNN残差检索基线"。

### 11.50 A1修复版重跑与KNN残差检索基线：损失工程无效，检索也失败，路由指向覆盖与小移SNR（2026-08-18）

A1（consensus-64/plain + `--s-from-b0`敏感度归一化）已用修复后的训练器重跑（3 fold ×
3 seed，与A0b-c64-plain-fixed同配置仅加归一化）：

```text
outputs/mppi_proposal/actor_a1_c64_snorm_20260818_v3_bootstrap_fixed/summary.json
```

结果：train恢复中位0.82（A0b为0.856），OOF pooled `0.065`（A0b `0.147`），正确episode
bootstrap CI `[-0.197, 0.274]`含零；fold级OOF为`0.146/-0.193/0.186`——fold 1三个seed一致为负
（-0.364/-0.193/-0.099），存在系统性反迁移的episode组；P05 -7~-19、worst -42~-207。
**敏感度归一化对OOF无改善**（0.065对0.148，在噪声内偏负），判定
`A0_GENERALIZATION_PROBLEM`不变。结合A0.1的stay-balanced也无益：损失工程
（归一化/加权）方向关闭。

为裁决"A2分头 vs 扩覆盖"，补做一个零训练检索基线（11.47 Phase 0.1b显示残差标签
近邻方向一致性可跨episode迁移，flip 0.112/k=1 cosine 0.494；若简单检索能拿到OOF
收益则瓶颈在网络/表示，若检索也失败则指向覆盖）：

```text
scripts/model_verify/analyze_mppi_knn_residual_baseline.py
outputs/mppi_proposal/knn_residual_baseline_20260818_v1/summary.json
```

3-fold episode-grouped（自有确定性分层，非A0b逐位同折）、block-equal IQR含a0度量、
train-fold拟合尺度、预测=最近train状态的归一化标签残差乘自身sigma、真实rollout评价。
结果**决定性为负**：k=1 OOF恢复`-0.429`（三fold -0.366/-0.391/-0.505），k=3
`-0.122`，回归比例0.61，worst -491；近邻距离中位0.78--0.89。

三点归因。第一，**标签空间的方向一致性不等于成本迁移**：Phase 0.1b的一致性是对
0.86σ大幅残差测的，而consensus标签多为0.26σ量级小移——正落在§11.44的近对消区，
方向噪声主导成本效应；把邻居的小移照搬过来比不动更差。第二，**MLP（OOF
+0.065~0.148）优于KNN（-0.43/-0.12）**：网络通过平均/正则化提取了部分可迁移信号，
容量不是瓶颈（train 0.82-0.856）。第三，fold双峰（fold 1系统性负）在A0b/A1/KNN
三套方法上同型——episode组级分布差异是共享结构。

路由结论：**扩状态覆盖为主线下一步**（diverse剩余1200状态+expansion 1350可用，
consensus-64标签约193评价/状态），A2分头降级为覆盖改善后的并行消融；同时登记一个
更便宜的标签侧消融A0.2——**gain阈值stay化**（11.47：88状态gain<0.01、159<0.1，
把最深噪区标签改成stay是零rollout重标注+一次重训）。历史对照必须保留：J16时代
3150状态BC未解决泛化，但当时无a0条件化、无共识/守卫标签、无stay监督，不构成对
当前设置的反证。部署Actor继续冻结；formal validation/test保持封存。

### 11.55 A0.2预注册：按guarded gain将低SNR标签改为stay（2026-08-18）

目的仅裁决“共识teacher的小幅改善标签是否因成本效应低于跨episode噪声而伤害蒸馏”，不改
网络、输入、fold、seed、优化器或真实DBM评价。基线固定为§11.53修复后的
`consensus-64/plain`。使用共识teacher自身的guarded deterministic gain：
`g=min(J_a0,J_warm)-J_consensus`，预注册两臂：`g<0.01`与`g<0.10`时把
`label_knots`改回`a0`，有效stay取原stay与阈值mask并集；不使用CRN逐状态margin，不新增
rollout。注意§11.47的88/159是multi-128 teacher口径；共识teacher对应计数为97/145，不能
混用，否则标签与筛选依据不配对。

所有臂复用修复后的3-fold×3-seed部署Actor合同；逐臂保存source gain、原stay、forced stay和
effective stay mask。机制门预注册为：相对c64/plain，至少2/3 seed的OOF恢复率配对改善
不少于0.05，且median-seed的P05/worst均不恶化。正式Actor放行门仍为§11.49的
`H_OOF>=0.50`、bootstrap CI下界>0、P05>=0，不因本消融放宽。若两阈值均不过机制门，关闭
gain-threshold loss/标签工程并进入覆盖扩张；若均过门，选较保守且tail更好的阈值；若仅一臂
过门，固定该阈值后再扩覆盖。Actor冻结，formal validation/test封存。

#### 11.55.1 A0.2结果：低gain stay化无恢复，标签阈值工程关闭

两臂均已按修复后的3-fold×3-seed合同完成，9/9 checkpoint均有有限loss与合法best epoch；
重标注mask与source gain落盘。结果与c64/plain严格配对如下：

| arm | forced/effective stay | H_train | H_OOF（3 seed） | median-seed CI | P05（3 fold） | worst |
|---|---:|---:|---:|---:|---:|---:|
| c64/plain | 4（原guard fallback） | 0.856 | 0.069/0.150/0.147 | [-0.052,0.315] | -9.93/-15.19/-7.21 | -83.7 |
| gain < 0.01 | 97 | 0.866 | 0.067/0.163/0.148 | [-0.050,0.318] | -8.70/-14.65/-7.22 | -72.0 |
| gain < 0.10 | 145 | 0.860 | 0.077/0.055/0.143 | [-0.146,0.256] | -15.98/-17.56/-3.76 | -92.2 |

`gain<0.01`相对基线的逐seed变化为`-0.002/+0.013/-0.000`，没有任何seed达到预注册
`+0.05`改善门；tail虽有小幅改善但不足以构成机制通过。`gain<0.10`为
`+0.008/-0.095/-0.005`，主体和tail均无稳定改善。两臂正式Actor门也全部失败。

判定：**低gain标签不是当前泛化墙的主要原因**；最深噪区stay化基本中性，更宽阈值会删除
仍然有用的小幅改善信号。关闭gain-threshold/stay-loss标签工程，不采用任何阈值作为新标签
合同。按§11.50路由进入独立物理状态覆盖扩张（优先diverse剩余1200与expansion 1350），
保持consensus-64/plain为主监督；A2 early/late结构降级为扩覆盖后的严格配对消融。

产物：

```text
outputs/mppi_proposal/actor_a02_c64_gainstay001_20260818_v2_bootstrap_fixed/
outputs/mppi_proposal/actor_a02_c64_gainstay010_20260818_v2_bootstrap_fixed/
```

### 11.51 覆盖学习曲线：600→1200→1800状态OOF持平，覆盖分支在diverse池内关闭（2026-08-18）

A0.2（gain阈值stay化）确认无效（0.01档与基线全同、0.10档变差），stay/损失工程关闭；
episode bootstrap修复后CI变宽但判定不变。覆盖轮按"一次生成+嵌套切片"执行：

```text
scripts/model_verify/generate_mppi_consensus64_labels_pool.py
scripts/model_verify/build_mppi_learning_curve_subsets.py
outputs/mppi_proposal/consensus64_labels_pool_20260818_v1/
outputs/mppi_proposal/learning_curve_subsets_20260818_v1/
outputs/mppi_proposal/actor_curve_n1200_20260818_v1/
outputs/mppi_proposal/actor_curve_n1800_20260818_v1/
```

全diverse-train池1800状态的consensus-64标签一次生成（351,000次评价，j_teacher均值
8.163，guarded fallback率0.39%）；600/1200嵌套子集经断言600≡Phase 1b状态集验证，
已有A0b结果作为曲线首点；expansion 1350状态因anchor策略混杂（仅frozen BC center
可用，非replay bootstrap actor）明确排除并留待anchor决策。三档均c64/plain、
3 fold × 3 seed、修复版训练器：

| 状态数 | OOF（中位seed） | OOF by seed | train | OOF CI | P05中位 | worst |
| ---: | ---: | --- | ---: | --- | ---: | ---: |
| 600 | +0.148 | 0.069/0.148/0.150 | 0.856 | [-0.016,0.183] | -12.5 | -108 |
| 1200 | -0.128 | -0.219/-0.128/0.039 | 0.885 | [-0.456,0.090] | -13.1 | -1602 |
| 1800 | +0.056 | 0.040/0.056/0.109 | 0.906 | [-0.092,0.175] | -15.2 | -364 |

口径勘误：600行引用了bootstrap修复前的`[-0.016,0.183]`，正确episode-bootstrap CI为
`[-0.052,0.315]`；该行tail与1200/1800行还混用了“fold P05摘要”和“全seed最坏值”。统一按
median-seed pooled OOF计算时，600/1200/1800的P05分别为`-10.19/-13.34/-16.26`，worst为
`-83.7/-850.4/-172.9`；若报全seed worst则必须单列。上述修正不改变曲线平坦、非单调且
tail不随覆盖改善的判定。

**学习曲线平坦且非单调**：状态数×3后train恢复从0.856升至0.906（记忆更好），
OOF在0附近波动（+0.148/-0.128/+0.056，所有CI含零、大幅重叠），fold方差依旧巨大
（n1200出现-0.461 fold），P05/worst未改善。**覆盖假设在diverse池尺度内不成立**：
新增状态提升的是训练集拟合，不是跨episode迁移。这与§11.36梯度翻转率对距离平坦、
§11.26定向probe"seen+0.66/unseen-0.10"构成第三次同型证据——该映射的跨episode
墙不随diverse池内密度变化。

剩余分支：（a）**表示/条件化侧**（A2 early/late分头、更丰富regime输入）——但已有
八轮within-representation尝试全部失败的事前概率不高；（b）**更大移动幅度的标签**
（0.26σ小移噪声主导的证据链完整，§11.50；迫使teacher在有余量状态做更大移动可提高
单标签SNR，但需重过J16时代"方向有用精度不足"的教训）；（c）**战略重定位**：接受
"学何时动+安全小动"的受限目标或纯fallback部署（warm floor+guard闭环已验证
-4.12%）。三者均需review裁决，本节不预注册。部署Actor继续冻结；formal validation
/test保持封存。

### 11.56 分支2预注册：固定共识方向的有界幅度扫描（2026-08-18）

裁决选择分支2，但先设teacher-side gate，不立即重做多方向搜索或训练Actor。理由是当前假设
针对“0.26σ标签的成本SNR不足”，最干净的干预应只改变已有三bank共识方向的幅度，避免同时
改变方向、搜索预算和网络。使用1800 diverse状态池；对每个状态从存储的raw consensus
residual构造`beta=[0,0.5,1,1.5,2,2.5,3]`，分别施加normalized RMS `0.5σ/0.8σ`
上限，候选集恒含`a0`与`warm`，由真实确定性DBM J50选最优，标签与cost严格对应，因此由
构造不劣于两个anchor。该实验不使用Critic、formal validation或test。

teacher gate预注册为同时满足：(1) 相对原c64 teacher的聚合anchor gain增加至少5%；
(2) 至少25%状态选择`beta>1`的共识外推；(3) mover的normalized RMS中位由约0.26σ提高到
至少0.40σ；(4) replay误差小于1e-4且零anchor回归。若0.5σ/0.8σ均不过门，关闭“大幅标签
SNR”分支并转战略分支3；若至少一臂过门，选择满足gate的较小cap，用对应1800标签复用修复
后的3-fold×3-seed c64/plain Actor合同。Actor层仍以`H_OOF>=0.50`、CI下界>0、P05>=0为
正式门，不以teacher改善代替蒸馏验证。

#### 11.56.1 Teacher gate结果：局部存在外推收益，但不足以形成大幅高SNR监督

完整1800状态扫描完成，DBM replay最大误差：a0/warm为0，raw consensus为`4.58e-5`；两臂
相对`min(J_a0,J_warm)`的最大回归均为0。原c64 teacher的mover移动中位实际为`0.173σ`
（此前约`0.26σ`是不同子集/统计口径，不得用于本池gate）。

| cap | teacher J均值 | anchor gain增幅 | beta>1比例 | mover移动中位 | mover P90 | gate |
|---|---:|---:|---:|---:|---:|---|
| 0.5σ | 7.994 | +2.31% | 37.2% | 0.202σ | 0.388σ | FAIL |
| 0.8σ | 7.985 | +2.43% | 36.9% | 0.202σ | 0.388σ | FAIL |

结果说明部分状态确实可沿共识方向继续走（约37%选择`beta>1`），但全池最优幅度仍主要集中在
`beta<=1.5`，0.5σ以上cap几乎不约束结果（0.8σ饱和仅0.44%）。两臂均未达到预注册的
gain `+5%`与mover中位`>=0.40σ`门，且0.8σ相对0.5σ仅再改善teacher J `0.009`。
因此“大幅有界共识标签可显著提高全局监督SNR”的假设不成立；按合同不启动Actor重训，关闭
分支2，避免用2.4%的teacher边际改善再承担一次高方差跨episode蒸馏。

正式路由转到战略分支3：不再要求网络精确回归每帧最优center。近期部署目标固定为
`warm/a0 + Actor proposal`双中心候选，由MPPI已有rollout cost做选择和warm floor保护；Actor
只作为可选proposal，不具备覆盖warm的权限。若继续学习，优先把任务收缩为低风险的
move/stay或步长置信度，不再把16D center回归或Critic gradient作为放行前提。下一项验证应是
用当前可部署Actor做固定DBM短闭环two-center A/B，确认收益、失败率与抖动；formal validation
/test仍封存。

产物：

```text
scripts/model_verify/analyze_mppi_consensus_amplitude_scan.py
outputs/mppi_proposal/consensus_amplitude_scan_20260818_v1/
```

### 11.52 A2前置review修正与B1布局oracle：均匀knot保持，时序结构实验合同确立（2026-08-18）

对"时序化输入-输出对齐"方向review的五项修正经核实采纳：(1)输入并非缺少ego配准——
`ego_reference_features`已输出[x,y,sin,cos,v_ref-vx]（mppi_proposal_policy.py:73），
真正缺的是与8个knot的逐时刻对齐，应加**knot-aligned geometry token**
（e_y(k)/e_yaw(k)/curvature(k)/v_ref(k)-vx，可复用first-pass rollout的预测轨迹误差，
零新增rollout），而非全局标量；(2)输出侧确为共享特征一次线性出16维（:1152），
但不用严格自回归GRU（MPC计划是联合优化、后段误差应回传前段），改用8个knot token
+小型双向GRU/1D conv/非因果self-attention的**temporal head**；(3)history压缩丢失
细粒度时序，优先"近历史高分辨率+长历史低分辨率"双支路；(4)feedback/gradient不可
直接按critic侧结论当nuisance移除，作为独立消融臂（A2-GT-clean）；(5)平滑loss约束
预测与teacher的**变化率差**（含current->knot0边界项、非均匀需除dt），不直接压平
预测。

**B1布局投影oracle**已执行（200分层状态，teacher序列投影到候选布局+头尾hold重建，
uniform布局往返恒等校验5.5e-05通过）：

```text
scripts/model_verify/analyze_mppi_knot_layout_oracle.py
outputs/mppi_proposal/knot_layout_oracle_20260818_v1/summary.json
```

| 布局 | 投影多余cost均值 | P95 |
| --- | ---: | ---: |
| uniform（恒等校验） | 0 | 0 |
| front_dense [0,3,7,12,18,26,35,49] | +6.87 | +25.1 |
| mid_dense | +28.4 | +139.9 |
| rear_dense | +185.9 | +749.0 |

**结论：保持uniform 8-knot布局**。投影口径的限定如实登记：这测的是"非均匀布局能否
表示uniform下优化的计划"，不是"非均匀布局的最优计划是否更好"；但结合历史GT-first
的J16->J100 gap仅0.198（uniform基可表达近优计划），布局不是瓶颈，A2实验全部在
uniform knot times=[0,7,...,49]上进行。

**A2实验合同**（全部1800状态、同fold同seed，机制门：>=2/3 seed相对1800基线OOF提升
>=0.10；系统性负迁移fold回到非负；P05/worst不恶化；正式门仍为OOF>=0.50、CI下界>0、
P05>=0）：
- A2-G：仅加knot-aligned相对几何token，原输出头；
- A2-T：仅换temporal token输出头，原输入；
- A2-GT：组合；仅当组合臂有效才做A2-GT-clean（移除feedback/gradient消融）；
- derivative平滑loss最后单独加，不与结构首轮混改。
现实提醒登记：1800基线OOF仅~0.056，结构模型需要带来很大提升；若A2-GT仍无改善，
"网络缺乏时序归纳偏置"解释可高置信关闭，回到two-center guard战略分支。Actor、
formal validation和test保持冻结。

### 11.53 early/late splice oracle：前段knot承载全部可部署价值，计划相干性证据确立（2026-08-18）

对n1800的OOF actor预测（5400行=1800状态×3 seed，预测knots与state keys已补入
oof_evaluation.npz；重跑与原run OOF一致0.056）做6种拼接并全部真实rollout：

```text
scripts/model_verify/analyze_mppi_early_late_splice.py
outputs/mppi_proposal/early_late_splice_20260818_v1/summary.json
```

| 拼接 | 恢复率 | gap closure |
| --- | ---: | ---: |
| actor（参照） | 0.048 | — |
| teacher（参照） | 0.698 | — |
| teacher早0-2 + actor晚 | **0.244** | **+0.302** |
| actor早 + teacher晚 | **-0.123** | **-0.262** |
| 仅早steering换teacher | 0.183 | +0.208 |
| 仅早accel换teacher | 0.085 | +0.057 |

三点结论。第一，**杠杆不对称被定量确认**：仅修早段3个knot即把恢复率从0.05拉到
0.24（闭合30% gap）；把晚段换成teacher反而**更差**（-0.26闭合）。结合预先登记的
对照——actor误差在早/晚knot完全均匀（中位比1.013）——该不对称反映的是控制时刻→
未来轨迹的因果杠杆，不是误差定位；"actor均匀地不准，但只有修早段值钱"。第二，
**计划相干性证据**：teacher晚段knot拼到actor早段上变负，说明晚段值与早段轨迹绑定
（teacher晚段是为teacher早段优化的）——knot是联合协调的计划，正是review主张非自
回归temporal decoder的直接证据；也解释了为何"完美早段"也只能到0.244而非teacher
的0.698（其余价值在早晚协调里）。第三，**通道分解**：早steering（0.183）贡献早段
收益的绝大部分，早accel仅0.085——与§11.41 position项主导、steering子空间翻转最重
的机制链自洽。

对A2的直接输入：(1)容量与精度预算应集中前段、尤其steering通道；(2)早段residual
bound应更保守（§11.52的反直觉纪律）；(3)temporal decoder必须非自回归双向耦合
（晚段依赖早段上下文）；(4)A2机制门追加一条早段诊断——结构模型的knot0-2误差/
收益应可测地改善，与splice方向一致。two-center guard继续作为安全底座。Actor、
formal validation和test保持冻结。

### 11.57 总consolidation：编号消歧、权威结论与当前计划（2026-08-18）

> **覆盖状态（2026-08-20）**：本节是 2026-08-18 的中间快照，已被 §11.80 覆盖；
> 当前权威结论与活跃计划以 §11.80 为准。本节保留历史口径与原始消歧记录。

本节原为 2026-08-18 时点的唯一权威索引；当时与更早历史节冲突时以本节为准。当前以 §11.80
为准，历史节保持append-only不删除、不改号。

#### 11.57.1 编号消歧索引（九对冲突）

| 编号 | 第一处（行序在前） | 第二处 | 权威性 |
| --- | --- | --- | --- |
| §11.48 | Horizon/position counterfactual**优先级合同**（预注册） | Horizon诊断**执行结果**（along/cross+加权+rho） | 互为合同/结果，都有效 |
| §11.49 | **Phase 2验证合同（终版）**——主判定门与判定树 | phase2a0语义清理MLP的A0对照（train 0.044） | 前者是主合同；后者是替代结构负对照，不得替代3-fold主协议（§11.53已注明） |
| §11.50 | Critic最后坐标机会（`COORDINATE_CRITIC_FAIL_CLOSE_GRADIENT_MAINLINE`） | A1修复版重跑+KNN检索基线（损失工程与检索均关闭） | 都有效，主题不同 |
| §11.51 | A0首次结果（**已作废**，早停bug） | 覆盖学习曲线600→1800（覆盖分支关闭） | 后者权威；前者仅审计记录 |
| §11.52 | A0b首次2x2（**A0b部分作废**；consensus标签生成与B0 artifact有效） | A2前置review修正+B1布局oracle（uniform保持） | 后者权威；前者中标签/B0产物仍被引用 |
| §11.53 | **A0/A0.1早停修复与2x2重跑（Phase 2权威数值表）** | early/late splice oracle（前段knot承载全部价值） | 都有效，主题不同 |
| §11.54 | A1旧执行结果（**已作废**，早停bug） | **A2结构实验合同**（三臂单变量+skip公平性工程） | 后者权威 |
| §11.55 | A0.2 gain阈值stay化（**已关闭**） | **A2三臂执行结果**（GT+0.091未达机制门） | 后者权威 |
| §11.61 | hysteresis网格与不对称warm回退（Pareto钉死） | KNN完整动作更正与直接估`a*`预注册 | 都有效，主题不同 |

引用规范：提及"§11.49合同/§11.51覆盖曲线/§11.52布局oracle/§11.53修复重跑"等一律用本索引的消歧描述；不再裸引编号。

#### 11.57.2 权威结论（关闭线汇总）

**Critic-gradient主线：正式关闭**（§11.38降级→§11.50`COORDINATE_CRITIC_FAIL_CLOSE_GRADIENT_MAINLINE`）。
机制链完整：position项近邻翻转主导（§11.41-11.42：已归因翻转中`149/170=87.6%`，全部翻转中
`149/250=59.6%`）、56%整体反向+晚段
质量集中（§11.46.1）、90%翻转涉及cross分量即横向修正符号歧义、s-d重参数化不能消除
（§11.48执行结果）、坐标工程只改condition不改可用性（§11.50 AT/Z对照）。Critic仅保留
candidate ranking/value辅助。通道口径已修正：物理action序为[accel, steer]，early steering
flat indices=[1,3,5]（§11.50.2）。

**Phase 1（teacher侧）**：最低有效预算32-64评价；单bank方向非canonical但软盆地均值安全；
600与1800状态guarded consensus-64标签零回归就绪（§11.43/11.45/11.47/覆盖轮）；
CRN audit：聚合正、逐状态K=8欠功效。

**Phase 2（Actor侧，以§11.53修复重跑为唯一权威数值）**：
- train恢复0.74-0.86全臂过门——**容量/拟合/标签生成都不是问题**；
- consensus-64标签有独立正作用（multi-128三seed OOF全负 → c64/plain全正0.069/0.150/0.147，
  worst -387→-84）；stay-balanced无益弃用；
- **主阻塞=跨episode泛化+tail**：OOF最高0.15远低于0.50门，CI含0，fold方差大且fold级
  系统性负迁移在MLP/KNN/多臂间同型；
- 已关闭的分支：损失工程（S归一化§11.50、stay加权）、KNN检索（-0.43/-0.12，检索比MLP差
  →瓶颈不在网络容量）、gain阈值stay化（§11.55.1中性）、**覆盖扩张**（600→1200→1800 OOF
  平坦非单调、tail不改善——与§11.36距离平坦、§11.26定向probe unseen失败构成第三次同型
  证据）、大幅标签SNR（§11.56.1 teacher gate失败，37%状态可外推但全池幅度仍0.202σ）。

**B线**：B0敏感度（S各向异性33倍、steer=2.4×accel、fold稳定5.9%）；**B1布局oracle：uniform
8-knot保持**（front_dense +6.87、rear_dense +185.9多余cost）；非均匀布局与降维basis不进入。

**结构证据与A2裁决（A系列结构实验闭环）**：early/late splice oracle（§11.53第二处）——
teacher早0-2拼actor晚恢复率0.048→0.244（闭合30%），actor早拼teacher晚**-0.123**（计划
相干性：晚段绑定早段轨迹）；早steering贡献绝大部分；actor误差早晚均匀（1.013）。A2三臂
（§11.55第二处，1800状态、公平skip工程保证train 0.91-0.93同级）：G/T单独与base持平
（0.048/0.056），GT组合+0.091（3/3 seed正、fold1 -0.12→+0.02救援、P05/worst略优），
但0/3 seed达到≥0.10机制门、距正式门0.50极远。**按预注册关闭"网络缺时序归纳偏置"
解释**；GT作为未来叠加项登记，GT-clean/双分辨率history/early bound/derivative loss消融
全部取消。至此loss→输入→坐标→覆盖→标签SNR→结构六类干预全部完成且均未达门。

#### 11.57.3 当前计划（活跃分支）

1. **主线：two-center guard闭环定型**（§8/§9.7原始待办的闭环，已验证单步stage cost
   -4.12%）。部署形态固定为warm floor + Actor proposal**双确定性候选**，MPPI rollout
   cost选择，Actor无覆盖warm权限。近期实验：(a)切换动力学参数化——hysteresis/switch
   penalty，抑制候选高频抖动；(b)多seed×三场景（nominal/2.4-2.8m/s高速/recovery）固定
   DBM短闭环A/B，报累计cost、失败率、横向/航向误差、控制抖动、warm-start漂移与guard
   触发；(c)A2-GT结构作为Actor候选实现之一纳入对照（叠加项，不独立主线）。
2. **A2-GT残值登记（可选叠加）**：全seed正向、fold1救援、尾部略优、train最高——若闭环
   对照显示其有独立价值再议，不预投入。
3. **Pending非阻塞项**：horizon加权候选的原始J50重rollout验证；rho与Phase 1b状态join；
   expansion 1350状态的anchor策略决策。

不变项：Actor冻结、formal validation/test封存、Critic主线关闭不再做loss engineering；
A系列结构实验（A0-A2全谱）已闭环，后续Actor改动须走闭环对照而非离线蒸馏门。

### 11.54 A2结构实验：合同收紧、判读修正与三臂实现（2026-08-18）

对§11.53的review修正予以采纳：**"纯粹由杠杆造成"降格为"杠杆与计划相干性是当前最
主要、且已有直接证据支持的解释"**（早晚误差幅值接近不能完全排除方向/通道耦合/裁剪
差异）；拼接计划属off-manifold hybrid，支持双向联合temporal decoder但**不证明其是
唯一可行结构**；early steering贡献跨seed稳定，temporal head须保留steering 0-2表达
精度。B1的投影属"采样重建"而非最小二乘投影，LS版本登记为可选归档项，不阻塞A2。

A2合同按review收紧：首轮严格单变量（A2-G=仅加knot-aligned几何、A2-T=仅换8-token
非因果双向头、A2-GT=组合）；**不同时引入**early bound收紧、early loss加权、双分辨率
history、feedback/gradient清理、derivative/smoothness loss（GT有效后逐项消融）；
每臂新增诊断（steer/accel的knot0-2误差、early/late敏感度加权误差、复跑同一splice
oracle、fold1恢复检查、P05/worst/stay误动率）；三臂全部跑完不提前停止。

实现（`scripts/model_verify/mppi_a2_actors.py`，冻结encoder与2sigma/tanh/clamp语义
不变，research-only）：A2-G在融合特征后拼接32维knot几何（从归一化ego reference在
uniform knot时刻采样[e_y,e_yaw,下一段航向变化(曲率代理),e_v]）；A2-T/GT为8-token
两层非因果self-attention解码器，token含anchor knot/时间编码/群体knot敏感度
（+GT的几何），全局trunk特征投影广播进token。**关键公平性工程**：token头内加
零初始化192->16 per-knot skip——无skip时token臂在共享120 epoch预算内h_train仅
0.29-0.58（误差集中在高杠杆早段、MSE相同但rollout恢复崩），skip使T/GT在初始化即
包含flat头函数类，冒烟后h_train回到0.84-0.88（base级），机制对比不再被优化能力
混杂。trainer新增`--arch`与通道级/敏感度加权误差诊断；`oof_evaluation.npz`已含
预测knots与state keys（splice oracle可直接复跑）。

正式三臂运行：1800状态c64/plain、3 fold × 3 seed、同fold/seed/训练配置，产物
`outputs/mppi_proposal/a2_{g,t,gt}_n1800_20260818_v1/`。判定门不变：机制门
（>=2/3 seed相对n1800基线OOF 0.056配对提升>=0.10、fold1回非负、P05/worst不恶化、
早段诊断与splice方向一致）；正式门OOF>=0.50/CI下界>0/P05>=0。GT失败则高置信关闭
"网络缺少时序归纳偏置"解释，转two-center guard/受限部署。Actor、formal validation
和test保持冻结。

### 11.55 A2三臂执行结果：GT小而一致的改善但未达机制门，时序归纳偏置解释按预注册关闭（2026-08-18）

A2-G/A2-T/A2-GT全部完成（1800状态c64/plain、3 fold × 3 seed、同fold/seed/配置，
零初始化per-knot skip保证T/GT包含flat头函数类，train恢复0.91-0.93与base同 level）：

| 臂 | OOF（中位seed） | OOF by seed | fold中位 | P05 | worst |
| --- | ---: | --- | --- | ---: | ---: |
| base | +0.056 | -0.039/+0.001/+0.084 | 0.00/**-0.12**/0.21 | -15.2 | -364 |
| A2-G | +0.048 | -0.008/-0.047/+0.054 | -0.01/-0.10/0.21 | -14.5 | -383 |
| A2-T | +0.056 | -0.030/-0.011/+0.124 | -0.01/-0.09/0.24 | -14.2 | -395 |
| **A2-GT** | **+0.091** | **+0.050/+0.009/+0.155** | **+0.01/+0.02**/0.23 | -14.7 | -349 |

三点判读。第一，**交互存在、单项无效**：G单独与base持平（0.048）、T单独持平
（0.056），GT组合+0.091——review"两轴可能交互、三臂必须跑完"的预判成立。第二，
**GT改善真实但小**：逐seed配对提升为+0.089/+0.008/+0.071（3/3为正但**无一达到
预注册的>=0.10机制门**）；fold1从-0.123回到+0.023（非负，达成该子门）；P05/worst
轻微改善（-14.7/-349对-15.2/-364）；knot0-2 steer误差0.0128与base持平。GT的
splice oracle复跑与base形态一致（earlyT +0.27对+0.30、lateT仍-0.26）。第三，
OOF 0.091距正式门0.50极远。

**按预注册裁决：机制门未过（0/3 seed达到+0.10），"网络缺少时序归纳偏置是主要
瓶颈"的解释高置信关闭。**残值如实登记：GT全seed正向、fold1救援成功、尾部略优、
train最高——时序结构有正交的小贡献，若未来重启可作为叠加项，但在当前证据下不
足以支撑继续投入。主线路由回**two-center guard/受限部署定型**：warm floor +
Actor proposal双确定性候选的闭环路线（已验证-4.12% stage cost），A2-GT结构可作为
该路线内Actor的候选实现之一在闭环中对照，但不再作为独立主线。GT-clean、双分辨率
history、early bound、derivative loss等后续消融按合同取消。Actor、formal
validation和test保持冻结。

### 11.58 输出时序耦合诊断：二阶结构已表达，缺失的是一阶逐状态残差（2026-08-18）

零rollout诊断（`analyze_mppi_output_temporal_coupling.py`→
`outputs/mppi_proposal/output_temporal_coupling_diagnostic_20260818_v1/`）：以逐状态anchor a0
为基取残差，对base与A2-G/T/GT的n1800 OOF预测（3 seed）计算teacher残差的跨knot二阶结构
复现度。首版按绝对knot计算时corr全为1.000——被anchor的跨状态方差主导，改残差基后有效。

| 架构 | 跨knot协方差向量corr | 协方差尺度比 | 逐knot方差分布Pearson | 平滑度比 | **逐knot残差corr** |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.994 | 0.964 | 0.943 | 0.805 | **0.097** |
| A2-G | 0.996 | 1.007 | 0.965 | 0.828 | 0.101 |
| A2-T | 0.995 | 0.942 | 0.943 | 0.795 | 0.098 |
| A2-GT | 0.994 | 0.947 | 0.937 | 0.801 | 0.099 |

三点判读。第一，**输出时序的二阶结构表达良好**：四个架构对teacher残差的跨knot协方差结构
复现度均0.99+，"哪些knot跨状态协同变化"已学到——"输出缺时序关联结构"的假设在二阶层面
**不成立**。第二，**真正缺失的是一阶逐状态预测**：逐knot残差与teacher的相关仅~0.10——网络
不知道"这个状态下每个knot该往哪个方向动多少"，这正是OOF 0.056泛化墙在输出侧的分解形态；
平滑度比0.80（预测比teacher更平滑）与回归均值行为一致。第三，**token头未改变输出结构**：
四架构在所有指标上几乎一致，与A2三臂OOF持平互证——时序归纳偏置不是输出侧缺失的环节。

对假设的修正表述：输出时序耦合问题应拆为两层——二阶结构（已表达）与一阶逐状态方向
（未表达且是泛化墙本体）；后者由§11.51-11.55的证据链归因到跨episode泛化而非结构。
本诊断不改变主线路由（two-center guard定型）。限定：协方差复现是跨状态population统计，
检验力弱于逐状态检验；stay状态（teacher残差为0）对协方差有收缩作用但不改变结论方向。
Actor、formal validation和test保持冻结。

### 11.59 Anchor局部性与杠杆分配诊断：输出结构属性均非瓶颈（2026-08-18）

零rollout诊断2+3（`analyze_mppi_anchor_locality_diagnostic.py`→
`outputs/mppi_proposal/anchor_locality_diagnostic_20260818_v1/`，200状态、fold0/seed0
checkpoint、逐状态autograd Jacobian）。

**诊断2（输出对anchor的时序局部性）**：对`J_center−I`（残差对anchor的Jacobian，扣除机械
恒等项）计算2x2 knot块对角质量占比（均匀分布基线=0.125）：

| 架构 | 块对角集中度 | P25 | 反转移位/机械位移 |
| --- | ---: | ---: | ---: |
| base | 0.197 | 0.193 | 0.997 |
| A2-G | 0.239 | 0.233 | 0.995 |
| A2-T | 0.230 | 0.209 | 0.994 |
| A2-GT | **0.289** | 0.244 | 0.992 |

两点判读。第一，**残差头对anchor的依赖总体很弱**：anchor knot序反转后输出位移≈纯机械
位移（比值0.99+），残差几乎不响应anchor变化——所有架构均如此。第二，token头确实更局部
（GT 0.289≈2.3×均匀基线 vs base 0.197≈1.6×），结构按设计起作用，但绝对量级弱，且该改善
未转化为OOF收益（与A2三臂持平互证）。注意构造性限定：teacher标签本身以a0为锚
（residual(a0)=label−a0），残差对anchor的弱依赖部分由标签构造吸收，不能单独解读为缺陷。

**诊断3（杠杆分配）**：B0终端位置杠杆份额[0.134,**0.235**,0.202,0.165,0.127,0.087,
0.043,0.008]（峰值在knot 1，knot 0+1合计0.37，单调衰减至knot 7的0.008）。逐knot残差方差
份额与杠杆份额的相关：base 0.904、A2-T 0.939、A2-GT 0.938、teacher 0.862——**网络把状态
响应能力分配到高杠杆knot的程度与teacher相当甚至略好**，容量跨knot分配不是问题。

**三诊断合成结论**：输出侧四项结构属性——二阶跨knot协方差（§11.58，0.99+）、杠杆分配
（本节，0.90+）、anchor局部性（弱但token头可改善）——均非瓶颈或仅边际可改善；缺失的仍
是§11.58定位的一阶逐状态方向预测（逐knot残差corr~0.10），即跨episode泛化墙本体。"输出
时序影响表达弱"的怀疑经三项直接检验后收窄为：**结构表达基本充分，逐状态条件化不足**——
与A系列结构实验闭环结论一致，不改变two-center guard主线路由。Actor、formal validation和
test保持冻结。

### 11.60 two-center guard闭环定型A/B：hysteresis同时改善成熟段cost与抖动，27集三场景通过内部口径（2026-08-18）

§11.57.3主线的闭环A/B已执行。实现：`hard_guard_action_sequence`新增可选
`guard_state/switch_margin/min_dwell`滞后（默认0/0与原argmin逐位一致，含warm平局
语义），car_node持有跨步状态、ROS参数`mppi_hard_guard_switch_margin/min_dwell`
经launch暴露；环境链（conda anycar→humble→set_env.sh、`mppi_backend:=dbm`、
XLA 0.3）与快照/trace顺序（start_step=250）按§9.7 pilot口径修复。预注册首轮
滞后参数（未在本数据调优）：margin=1.0、dwell=5。

```text
scripts/model_verify/run_mppi_guard_closed_loop_ab.sh
scripts/model_verify/analyze_mppi_guard_hysteresis_ab.py
/disk/collect_data_from_anycar/mppi_rl_closed_loop/guard_hysteresis_ab_20260818_v1/
outputs/mppi_proposal/guard_hysteresis_ab_20260818_v1/summary.json
```

27集（baseline/guard/guard_hyst × nominal1.6/high2.8/recovery2.8+扰动 × 3 matched
seed）全部301步完成。realized stage cost相对baseline（全窗/成熟250-300，seed中位）：

| 场景 | guard | guard_hyst | hyst成熟段绝对值 |
| --- | --- | --- | --- |
| nominal | -2.0% / -34.4% | **-4.3%** / -30.9% | 2.27→1.57（基线绝对值小，相对量放大） |
| high | -4.1% / -2.3% | -2.7% / **-10.2%** | 31.2→28.0 |
| recovery | -4.5% / -7.2% | **-5.6%** / **-13.7%** | 31.5→27.2 |

抖动（§9.7的主要失败项）：hyst相对guard的acc二阶差分-19.5%/-21.2%/-9.1%，
相对baseline仅+39.1%/+18.0%/+9.5%（guard为+67%/+50%/+28%）——**滞后把抖动
恶化压缩到约三分之一到五分之一**；切换率nominal 0.08→0.02、high 0.28→0.16、
recovery 0.24→0.10。跟踪精度普遍改善（nominal横向RMSE 0.056→0.019-0.020，
high 0.191→0.161，recovery航向0.167→0.136）。§9.7单集结论（-4.12%）在多seed
三场景下复现并加强。

**代价与限定**：hysteresis放松了逐步warm硬下界——违规步比例nominal/high/recovery
成熟段为2.0%/23.5%/13.7%，违规步平均超cost 0.003/4.19/3.35单位；部署语义须改述
为"hysteresis floor"而非构造性warm floor，margin/dwell进入安全参数合同。
延迟仍~230-242ms（实时性未解决，按当前阶段范围外处理）。本结果为内部口径，
未触formal validation/test。

下一步：(a) margin/dwell小网格（如0.5/2.0×3/8）看floor违规与成熟cost的Pareto；
(b) A2-GT作为proposal来源对照（§11.57.3c，叠加项）；(c) 若定型，进入§11.57.3
的正式闭环gate链。Actor、formal validation和test保持冻结。

### 11.61 hysteresis网格与不对称warm回退：Pareto钉死，硬下界配置确立（2026-08-18）

§11.60后续的margin/dwell网格（3x3，新增6配置54集）与不对称warm回退配置
（`warm_hard_return`：进actor支需margin/dwell，回warm支即时——构造性warm硬下界
逐步恢复）已全部完成：

```text
scripts/model_verify/run_mppi_guard_hysteresis_grid.sh
scripts/model_verify/run_mppi_guard_warm_hard_return.sh
scripts/model_verify/analyze_mppi_guard_hysteresis_grid.py
/disk/.../guard_hysteresis_ab_20260818_v1/ghist_*  （54+9集）
outputs/mppi_proposal/guard_hysteresis_grid_20260818_v1/summary.json
```

网格主表（成熟段改善 | 安全代价=违规步比x平均超cost | 违规步比，seed中位）：

| 配置 | nominal | high | recovery |
| --- | --- | --- | --- |
| m0.5_d3 | +23.6% / 0.00 | **-2.5%** / 0.97 / 19.6% | +6.9% / 0.17 |
| m1.0_d5 | +30.9% / 0.00 | +10.2% / 0.99 / 23.5% | +13.7% / 0.46 |
| m2.0_d3 | **+38.6% / 0.00 / 0%** | +3.3% / 1.26 | **+17.3%** / 0.35 |
| m2.0_d8 | +36.4% / 0.00 | -0.3% / 2.49 / 25.5% | +11.3% / 3.37 / 29.4% |
| **m1.0whr_d5** | +32.4% / **0.00 / 0%** | +0.6% / **0.00 / 0%** | +6.0% / **0.00 / 0%** |

三点结论。第一，**review的预判成立且机制已定位**：margin=2.0xdwell=8压不住高速违规
（25.5%、安全代价2.49）且收益转负（-0.3%）——违规=粘滞的actor支在其变差相位被
滞后期拖住，对称滞后越强违规越多，场景自适应margin不是正解。第二，**不对称warm
回退按构造达成零违规**（三场景违规步比均为0.000，warm硬下界逐步成立），nominal
保留几乎全部收益（+32.4%）。第三，**硬下界的代价在高速/recovery显性化**：high从
+10.2%降至+0.6%、recovery从+13.7%降至+6.0%——对称滞后那部分"收益"有相当份额来自
在model-cost上违规压过warm的步骤（model-cost下界与realized tracking cost不是同一
目标），硬下界把这些步骤交还warm。全网格仍有两格输给baseline（m0.5_d3高速
-2.5%、m2.0_d8高速-0.3%），网格内已标注。

**合同裁决建议（待review确认）**：部署默认配置取`m1.0whr_d5`（零违规硬下界+
nominal全额收益+high/recovery小额正收益，全场景严格优于warm-only）；对称滞后
（m1.0_d5/m2.0_d3）作为opt-in性能档，其floor松弛语义与安全参数合同随配置显式
声明。时长分解已按review要求入档：high场景whr每步P50 `225ms`，其中actor 129-
rollout context `115ms`、余项（warm 256+guard 2评价）`110ms`——实时性若入范围，
这是起点数据。下一步为(b)：A2-GT作为proposal来源在此锁定合同上对照。Actor、
formal validation和test保持冻结。

### 11.61 KNN完整动作对照的更正与直接估a*的严谨对照预注册（2026-08-18）

**先更正一条此前的错误解读。** 先跑的KNN对照（`analyze_mppi_knn_fullaction_contrast.py`→
`outputs/mppi_proposal/knn_fullaction_contrast_20260818_v1/`）两臂为：residual臂=`a0_test+
mean(Δa_neighbor)`，full_action臂=`mean(a*_neighbor)`。展开后二者**修正量项`mean(Δa_neighbor)`
完全相同，唯一差别是锚点**（full_action用邻居的平均锚点，residual用测试状态自己的锚点）。
因此该对比实际测的是"该用测试状态自己的锚点还是邻居的锚点"，**不是在测"完整动作vs残差哪个
是更好的目标"**。full_action臂崩到OOF −14.5（worst −3670、回归94%）是因为用了错误锚点，
不能据此推断"完整动作不可取/直接估a*不可行"。此前基于该结果否定"直接估a*"的解读作废。

**公平测"a* vs Δa*哪个更稳"的是近邻连续性**（§11.58-11.59口径，无锚点混淆）：a*近邻
cosine 0.73，Δa*近邻cosine 0.019——a*确实比Δa*连续得多，支持"完整动作更稳"的先验。
KNN检索因锚点混淆对此问题不作数。

**严谨对照预注册**：真正能裁决"直接估a*是否泛化更好"的是训练对照。设计：同encoder、同
fold/seed/优化器/epoch，仅输出参数化不同——
- **residual臂**：现有`center=clamp(a0+tanh(head)·2σ)`，头有效目标=Δa*（小，范数0.205）；
- **direct臂**：`center=tanh(head)`直接预测a*（大，范数1.48），锚点仍作encoder输入但不加到输出。
二者同target=label_knots、同MSE，唯一差别是头学Δa*还是学a*。判读：若direct臂OOF恢复显著
高于residual臂→直接估a*泛化更好（用户假设成立）；若residual臂更好或持平→残差仍是更优目标。
同时这直接检验量级失衡（direct头要学的量是residual的约7倍）是否真的伤害泛化。Actor、formal
validation和test保持冻结。

### 11.62 直接估a* vs 残差严谨对照：direct臂崩塌，锚点先验是必需而非负担（2026-08-18）

> **覆盖状态（2026-08-20）**：本节的实验数值与artifact保留作历史诊断，但“direct不可学/
> 拟合不动”“residual是正确部署参数化”“锚点是最终Actor必需输入”三项判定已被
> §11.64--§11.70覆盖，**不得再作为当前路线结论引用**。按state key重新对齐后，direct在
> heldout上的逐维相关约0.935，证明完整中心可以学习；其问题是绝对动作误差0.062高于
> residual的0.044，并被尖锐cost曲面放大。最终部署合同明确禁止Actor使用anchor；带anchor的
> residual/direct结果只作为“已有高质量中心时可达到的精度上界”，不属于可部署Actor候选。

§11.61预注册的严谨对照已执行（`run_mppi_actor_a0_baseline.py --arch base/direct`，1800状态、
3fold×3seed、同fold/seed/优化器/120epoch，仅输出参数化不同；两臂best_epoch均118-120，训练
真实跑满）。产物`outputs/mppi_proposal/direct_vs_residual_{residual,direct}_20260818_v1/`。

| 臂 | h_train（9 run） | h_oof（9 run） | 判定 |
| --- | --- | --- | --- |
| residual（a0+tanh·2σ） | 0.90-0.92 | −0.13~+0.32（中位~0.01） | 泛化问题（已知） |
| **direct（tanh直接预测a*）** | **0.36-0.61** | **−1.78~−2.85（全负）** | **崩塌** |

三点结论。第一，**direct臂连训练集都拟合不好**（h_train 0.36-0.61 vs residual 0.90-0.92）——
去掉输出端的锚点先验后，网络要从头学完整动作（范数1.48，是残差0.205的约7倍），连记忆都
变差。第二，**direct臂OOF灾难性为负**（−1.78~−2.85，比a0还差得多），而residual臂至少
在0附近——直接预测的完整动作主动有害。第三，**"a*更连续"没有转化为更好可学性**：§11.58的
近邻cosine（a* 0.73 vs Δa* 0.019）显示a*作为函数更平滑，但**直接学它反而崩塌**——因为
锚点先验（输出=锚点+有界修正）提供了"答案就在a0附近"的强归纳偏置，去掉它任务难度陡增。

**裁决**：用户"总动作更稳、应去掉锚点直接估完整值"的假设被严谨对照**证伪**。**锚点是必需的
归纳偏置，不是病态来源**；残差（有界、围绕锚点）是正确参数化。这与§11.60-11.61的KNN锚点
混淆更正合并为完整认知：a*作为绝对量既不可鲁棒检索（§11.61锚点混淆），也不可无锚点直接学习
（本节）。量级失衡（direct头学7倍大的量）确实伤害泛化，证实了用户"量级失衡需测试"的疑虑
——测试结果是它确实有害。综合再次指向two-center guard：锚点/残差结构+在线搜索兜底是证据
支持的落点。Actor、formal validation和test保持冻结。

### 11.63 残差坐标判别系列：warm start不随机，不稳定是残差坐标制造的（2026-08-19）

> **口径更新（2026-08-20）**：本节关于`a_proximal`比`a_proximal-a_warm`更coherent的
> 数值结论仍有效；但“direct无锚点学a*拟合不动”和据此形成的路线判断已被§11.64、§11.66
> 与§11.70覆盖。当前解释是：完整中心target可学且更coherent，direct的主要困难是达到尖锐
> cost所需的绝对精度；更canonical的J16监督已显著改善no-anchor direct。部署Actor仍禁止
> 使用warm/anchor。

（编号说明：上文存在两个§11.61，本节为§11.63，前接§11.62"直接估a* vs 残差"。）

围绕"a0是否随机、残差是否可学"做了三项递进验证，结论收敛为：**不稳定是残差坐标本身
制造的，不是warm start随机**。

**验证1：warm start跨episode稳定性（物理状态找近邻、排除同episode）**。若a0/warm是随机量，
跨episode近邻cosine应≈0。实测：

| 量 | 跨episode近邻cos中位 | 正比例 |
| --- | ---: | ---: |
| warm（anchor_center） | +0.661 | 0.984 |
| a0（bootstrap_actor_center） | +0.661 | 0.984 |
| a*（label/teacher） | +0.655 | 0.983 |

**warm start不是随机量，是物理状态的相当平滑的函数。**

**验证2：warm≈a0≈a*（逐状态cos中位）**：warm vs a0 = **0.9995**（几乎完全相同）、
warm vs a* = 0.991、a0 vs a* = 0.992，范数均~1.47。warm不仅不随机，而且几乎就是最优。
自洽性：残差范数0.216 = `1.47·sqrt(2(1-0.99))`——正因a0≈a*，残差才小。

**验证3：同一proximal teacher，绝对 vs 减warm残差的邻域一致性**（不拿全局J16 center对照，
物理状态找近邻、排除同episode）：

| 量 | 邻域cos中位 | 正比例 | P10 |
| --- | ---: | ---: | ---: |
| a_proximal（绝对） | **+0.655** | 0.983 | +0.293 |
| a_proximal − a_warm（残差） | **+0.029** | 0.559 | −0.363 |

**coherence比22.5×**：a_proximal本身coherent（0.655、正比例98%），减warm后的残差几乎不
coherent（0.029、正比例≈抛硬币、P10 −0.36）。

**核心判别**：`coherence(a_proximal) ≫ coherence(Δa)`成立且差距极大。**绝对动作平滑可学，
不稳定是"做差"这一步制造的**——残差=两个大且平滑的量的微小差，方向对状态极其敏感。这与
"warm start不随机"不矛盾、互相印证：两个绝对量各自平滑，但它们的小差值方向病态。

**与§11.62的统一**：§11.62证明去掉锚点的direct臂崩塌（h_train仅0.36-0.61、h_oof −2.85），
说明锚点去掉后网络连拟合都做不到；本节证明残差坐标本身制造不稳定。二者合起来的准确结论：
**target（a_proximal）是coherent的、是对的；残差参数化把它变病了；但直接无锚点学a*又拟合不动。**
所以既不是"锚点随机"，也不是"target不好"，而是**残差参数化病态 + direct臂未解决拟合**。

**路线含义**：该结果实质支持"直接学完整动作"方向（target coherent），但前提是先把direct臂
的拟合做通（当前h_train仅0.36-0.61是瓶颈，不是target问题）。下一步：定位direct臂拟合不好
的根因（量级/参数化/架构），把direct臂做通后再验证泛化。Actor、formal validation和test保持
冻结。

### 11.64 direct臂根因定位：两臂都在学a*，根因是cost hypersensitivity（2026-08-19）

**先更正一个分析错误。** 对`direct_vs_residual_{residual,direct}`两臂的OOF预测做逐维相关分析时，
未按state key对齐预测与标签的行序（oof_evaluation.npz的行序与labels.npz不同），导致一度得出
"两臂逐维corr均≈0.02、都没在学"的错误结论。按key重排后更正如下：

| 臂 | cos(pred, a*) | per-dim corr中位 | err_med | \|pred−warm\| | h_train |
| --- | ---: | ---: | ---: | ---: | ---: |
| residual | 0.986 | **0.971** [0.952, 0.994] | 0.044 | 0.198 | 0.90 |
| direct | 0.968 | **0.935** [0.714, 0.987] | 0.062 | 0.366 | 0.36-0.61 |

三点更正与根因判定。第一，**两臂都在学a\***——direct臂per-dim corr 0.94、cos(pred,a*)
0.968，**不是"拟合不动"**；它只是比residual臂稍差（corr 0.94 vs 0.97、err 0.062 vs 0.044、
steer corr 0.94 vs 0.97、accel corr 0.89 vs 0.97）。第二，**预测精度差3%但h_train差50%**
（0.90 vs 0.36-0.61）——cost landscape对动作极度敏感，微小精度差异被急剧放大。这与§11.61
KNN对照"动作相似≠cost相似"（cos 0.73的动作cost崩到−14.5）完全同型：cost hypersensitivity
是系统性性质。第三，**residual臂的优势来自warm start先验**（输出从a_warm出发、天然更精确
一点点），在sharp cost下这"一点精度"变成了巨大cost优势——不是残差这个坐标本身更优，
而是它获得了更精确的起点。

**对"直接学完整动作"路线的修正判定**：target（a_proximal）确实可学（corr 0.94），§11.62
"direct臂拟合不动"的表述应更正为"direct臂能学但精度稍逊、cost hypersensitivity放大差距"。
问题不在"学不学得会"，而在于sharp cost下需要更高精度才能获得好的cost恢复。改进方向不是
"换target"，而是提高direct臂的预测精度（更强架构/训练）——或者利用§11.63的发现
（残差坐标coherence 0.029），把A/B重新设计为"同一target、不同参数化精度"的对照。

**方法论教训登记**：oof_evaluation.npz的行序与labels.npz不一致，任何跨artifact对齐必须
按state key匹配，不能按行号。此前§11.58-11.59的输出时序诊断已用key匹配（结果不受影响），
但本节更正前的错误分析曾一度写入对话，现以本节为准。Actor、formal validation和test保持
冻结。

### 11.65 最终归因三分析：cost均匀放大微小误差差，非方向性精度缺失（2026-08-19）

用现有direct/residual两臂的OOF预测做零训练归因（不再训练新模型），按state key匹配后
计算逐knot/channel误差、sensitivity加权误差与真实J差距的相关性、按sensitivity四分位
的误差分解。产物存`outputs/mppi_proposal/direct_vs_residual_{residual,direct}_20260818_v1/`
（分析复用既有oof_evaluation.npz，无新artifact）。

**分析1（逐knot/channel误差，seed0，3seed一致）**：early(k0-2) steer仅**1.16×**、
late(k3-7) steer 1.28×、early accel 1.31×、late accel **1.84×**。最大差距在late accel
（低敏感方向），**不在early steering——"early steering差很多是根因"的假设被证伪**。

**分析2（sensitivity加权误差 vs 真实J差距）**：raw error比值(direct/residual) 2.2-2.6×、
E_S(sensitivity加权) 2.0×、J_diff(真实cost差距) **3.4-4.1×**。E_S与J_diff的逐状态相关
仅0.30-0.44；且E_S比值反而低于raw error比值——sensitivity加权**没有**改善解释力，
"cost-sensitive direction精度不足"假设**方向反了**。

**分析3（sensitivity四分位误差分解）**：Q1(top4最高敏感，全为early steering) direct仅
**1.23×**差；Q4(bottom4最低敏感) direct **1.55×**差。**direct的误差相对更差的是低敏感
方向，不是高敏感方向**。E_S贡献Q1占75%（绝对主导），但Q1的比值最小（1.51×对Q4的3.79×）。

**综合判定**：三个假设两个被证伪，一个成立——**"cost均匀放大微小误差差"**。direct在全
维度均匀稍差（1.2-1.5×），没有哪个方向特别差；但sharp cost landscape把这个均匀小差
非线性放大成3.4-4.1×的cost差距。residual臂的优势来自warm start先验让起点更精确一点
（err 0.044 vs 0.062），在sharp cost下"一点精度"变成了巨大cost差。这不是"缺cost-sensitive
方向精度"，而是**所有方向都差一点点、cost把每一点点都急剧放大**。

对"直接学完整动作"路线的最终含义：改进不在"瞄准某个方向"，而在**整体提高精度**——
任何维度的微小精度提升在sharp cost下都有巨大cost回报。Actor、formal validation和
test保持冻结。

### 11.66 口径修正与heldout泛化确认：direct预测精度泛化良好，瓶颈是cost非线性放大（2026-08-19）

**口径修正（review指正）**：此前用"corr只差0.03（0.971→0.935）却导致4倍cost差"表述精度
差距是误导性的——correlation衡量线性关系强度，不等于误差幅度。实际口径：

| 量 | residual | direct | 比值 |
| --- | ---: | ---: | ---: |
| per-dim corr（heldout） | 0.971 | 0.935 | — |
| cos(pred, a*) | 0.986 | 0.967 | — |
| \|e\| 中位 | 0.044 | 0.062 | **1.41×** |
| 平方误差和比值 | — | — | **2.2-2.6×** |
| heldout cost h中位 | -0.07~-0.16 | **-4.2~-4.8** | — |
| 聚合h_oof | ~0.01 | -2.85 | — |

准确表述为：**两者都具有很高的action correlation，但direct的实际动作误差系统性更大
（1.41×幅度、2.2-2.6×平方误差）；真实cost对这种看起来不大的联合动作偏差具有非常强的
非线性放大（平方误差2.2-2.6× → cost差距3.4-4.1×，超越二次放大）。**

**heldout泛化确认（零训练，oof_evaluation.npz逐fold验证全为out-of-fold）**：oof_evaluation
的预测已确认全部来自未训练该状态的fold模型（fold值逐位匹配），因此per-dim corr 0.94
即direct的**heldout**预测精度。**direct的预测精度泛化良好**——"direct泛化不行"和
"full action不可学"两个假设均被证伪。

**最终瓶颈定位**：direct路线的唯一瓶颈是**cost的非线性放大**——平方误差2.2-2.6×被放大到
cost差距3.4-4.1×（超越二次），导致corr 0.94的预测精度仍不足以获得正的cost恢复。
residual臂的corr 0.97刚好越过"cost正恢复"极窄门槛（h≈-0.1，勉强正），direct的0.94差
0.036即掉到-4.2。改进方向是**整体提高direct的预测精度**（任何维度的微小精度提升在
sharp cost下都有巨大cost回报），而非"瞄准某个方向"或"换参数化"。Actor、formal validation
和test保持冻结。

### 11.67 真实rollout cost全景：residual vs direct的J值直接对比（2026-08-19）

后续对比统一使用**J值直接报告**（不再转化为h/gain等导出指标），避免转化后丢失直观量级。
参照基准（1800状态）：J(warm)=36.6，J(a0)=15.9，J(teacher)=8.2。

**真实rollout cost（heldout，key-matched，3 seed汇总）**：

| 指标 | residual | direct |
| --- | ---: | ---: |
| **J(actor) 中位** | **8.0** | **13.1** |
| J(actor) 均值 | 15.4 | 33.2 |
| gain 中位 | -0.04 | **-3.1** |
| gain P05 | -16.2 | **-96.5** |
| 正收益比例 | 47% | **21%** |
| 优于warm | 70% | 38% |
| 优于a0 | 47% | **21%** |

配对比较：direct仅在21%的状态上优于residual（配对差中位-2.8~-3.2）。

**正确误差口径与cost放大**：

| 量 | residual | direct | 比值 |
| --- | ---: | ---: | ---: |
| \|e\| 中位 | 0.044 | 0.062 | **1.41×** |
| \|e\|² 总和 | 163.6 | 385.6 | **2.36×** |
| 逐状态RMS | 0.267 | 0.396 | 1.48× |
| **cost亏损比** | — | — | **6.74×** |

**cost放大超越二次2.86×**（平方误差2.36× → cost亏损6.74×）——cost landscape有显著
高阶曲率，解释了为什么"误差只差1.41×"却导致"cost差6.74×"。

三点洞察。第一，**residual的中位J≈8.0已接近teacher的8.2**——它在中位状态上表现很好，
问题在尾部（P05 gain=-16.2）。第二，**direct的中位J=13.1介于a0(15.9)和teacher(8.2)之间**
——不是完全崩塌，但仅21%状态有正收益。第三，**cost非线性放大超越二次**——1.41×的误差差
被放大到6.74×的cost亏损差，说明sharp cost landscape的高阶曲率是关键放大器。

后续所有cost对比统一用J值直接报告。Actor、formal validation和test保持冻结。

### 11.68 Direct no-anchor实验：去掉anchor输入后精度大幅下降（2026-08-19）

按用户约束执行：anchor（warm start/bootstrap actor center）完全不进入网络——不作为输出
锚点、也不作为encoder输入（anchor输入槽位zero）。同时加入per-dim输出centering（从训练集
a*统计center/scale）和两层MLP head（192→256→16）作为容量增强。1800状态、3fold×3seed。

| 指标 | residual | direct（有anchor输入） | **direct no-anchor** |
| --- | ---: | ---: | ---: |
| heldout corr | 0.971 | 0.935 | **0.730** |
| cos(pred, a*) | 0.986 | 0.967 | **0.858** |
| \|e\| 中位 | 0.044 | 0.062 | **0.109** |
| J(actor) 中位 | 8.0 | 13.1 | **31.3** |
| 正收益比例 | 47% | 21% | **9%** |

**anchor作为encoder输入贡献了约30%的预测精度**（corr 0.94→0.73）。warm≈a*（cos 0.99），
anchor本质上是一个99%准确的a*特征——去掉等于扔掉一个免费的强预测器。centering+深head
不足以弥补这一信息损失。

**部署含义（用户陈述的约束基础）**：部署时warm start可能非常不准（尤其Query网络场景下），
依赖它作为输入的模型在那个场景下可能退化。因此"anchor不入网"是合理的部署鲁棒性约束，
即使它牺牲了当前离线精度。

**当前状态**：no-anchor direct corr 0.73、J(actor)中位31.3（接近warm的36.6）——精度差距
仍然很大。下一步分析no-anchor臂的误差结构，找不依赖anchor的精度提升路径。Actor、formal
validation和test保持冻结。产物：
`outputs/mppi_proposal/direct_noanchor_20260819_v1/`。

### 11.69 No-anchor GT改进与训练极限验证：GT小幅有效、训练已到极限、loss公共项不主导（2026-08-19）

> **覆盖状态（2026-08-20）**：本节关于proximal teacher的实验结果有效，但“no-anchor精度
> 上限约corr 0.765”“缺口不是架构或训练能弥补、属于固定信息量上限”的外推已被§11.70
> **明确证伪**。在同一no-anchor GT架构下将监督目标换为更coherent的J16 best-found后，
> corr达到0.860、动作误差从0.108降到0.054、k7误差从0.210降到0.060。因此0.765只代表
> “proximal teacher + 当时训练合同”的结果，不是no-anchor路线的结构上限；旧teacher的
> 非canonical性是主要误差来源之一。

在no-anchor约束下跑了GT架构（knot-aligned geometry tokens + temporal decoder + per-dim
centering）和400 epoch长训练版，并分析了target的loss结构。

**GT架构小幅有效但不够**：

| 臂 | corr | \|e\| | k7\|e\| | J(actor)中位 | 正收益% |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-anchor MLP (120ep) | 0.730 | 0.109 | 0.208 | 31.3 | 9.6% |
| **no-anchor GT (120ep)** | **0.765** | 0.108 | 0.210 | 27.1 | **10.4%** |
| no-anchor GT (400ep) | 0.765 | 0.107 | 0.211 | 26.4 | 11.0% |
| direct（有anchor输入） | 0.935 | 0.062 | 0.048 | 13.1 | 21% |
| residual | 0.971 | 0.044 | 0.038 | 8.0 | 47% |

GT把corr从0.730提升到0.765（+0.035），J中位从31.3改善到27.1——**方向正确但幅度不够**。
k7误差几乎没变（0.208→0.210），说明geometry tokens对k7帮助有限。

**训练已到极限**：400 epoch（best_epoch中位262）vs 120 epoch（best_epoch中位119），corr
几乎不变（0.765 vs 0.765），|e|仅从0.108微降到0.107。**更多训练不能显著提升精度。**

**Loss公共项不主导**：a*的per-dim mean² / 总信号 = **28%**——"预测均值"这个公共项只占
loss的28%，剩余72%是状态依赖的deviation。**公共项不是精度瓶颈。**各维信号SNR
（|mean|/std）：accel维度1.2-1.6（有强均值），steer维度0.04-0.21（均值极弱），k7两通道
SNR最低（0.30/0.04）——k7本质上是一个"无公共锚点、全靠状态推断"的维度，这就是k7去掉
anchor后崩塌（0.21 vs 0.05）的原因。

**Target whitening已生效**：per-dim centering（out_center/out_scale）已在GT臂中实现，corr
从0.730→0.765的提升部分来自此。但whitening解决的是量级匹配，不解决信息缺失——k7的
信息缺口（无anchor时无免费参考）无法靠whitening填补。

**综合判定**：no-anchor direct的精度上限约corr 0.765、|e|≈0.107——**anchor输入的缺失
（corr 0.94→0.77，|e| 0.062→0.107）不是架构或训练能弥补的，是信息量缺口**。在不引入
anchor的前提下，提升路径只剩：(a)增加其他信息源（更丰富的history/更长参考轨迹），
(b)接受较低精度并依赖two-center guard兜底。Actor、formal validation和test保持冻结。

产物：
```text
outputs/mppi_proposal/direct_noanchor_gt_20260819_v1/summary.json
outputs/mppi_proposal/direct_noanchor_long_20260819_v1/summary.json
```

### 11.70 J16 oracle监督实验：no-anchor精度大幅提升，k7问题大幅缓解（2026-08-19）

> **当前部署合同（2026-08-20）**：最终Actor固定为`state/history/reference -> 完整8x2 knots`
> 的no-anchor direct映射。anchor/warm不得进入Actor encoder、token、输出skip或标签canonical
> 选择规则；warm只允许在Actor外部作为MPPI two-center guard的独立候选。J16 no-anchor GT是
> 当前有效Actor研究基线；所有带anchor的residual结果仅保留为诊断精度上界。

用J16 oracle（全局最优best-found，coherence 0.787 vs proximal teacher 0.656）替代proximal
teacher作为no-anchor GT的监督目标，1800状态全部有J16数据。

| 指标 | proximal teacher target | **J16 oracle target** | 改善 |
| --- | ---: | ---: | ---: |
| corr | 0.765 | **0.860** | **+0.095** |
| \|e\| 中位 | 0.108 | **0.054** | **-50%** |
| k7\|e\| | 0.210 | **0.060** | **-71%** |
| 正收益比例 | 10.2% | **15.7%** | +5.5pp |
| J(actor)中位 | 28.9 | 25.0 | -3.9 |

三点结论。第一，**用户假设完全验证**：更coherent的target（J16 0.787）比proximal teacher
（0.656）显著更可学——coherence差异直接转化为预测精度差异。第二，**k7问题大幅缓解**
（-71%）——J16全局最优的k7比proximal局部最优的k7更canonical，从"几乎不可预测"变为
"基本可预测"。第三，**cost hypersensitivity仍是最终瓶颈**：|e|=0.054仍不足以获得好的
cost恢复（J中位25 vs 目标4.8）；residual臂的|e|=0.044刚好过门槛，差距只剩0.01。

产物：
```text
outputs/mppi_proposal/j16_oracle_labels_20260819_v1/labels.npz
outputs/mppi_proposal/j16_noanchor_gt_20260819_v1/summary.json
scripts/model_verify/run_mppi_global_search_pilot.py
```

### 11.71 TCN history encoder实验：TCN单独不如GT token decoder（2026-08-19）

按review建议实现了Dilated TCN history encoder（5个residual block、dilation 1/2/4/8/16、
hidden 64、kernel 3、末端token+global avg pool拼接→128维），Reference用简单Conv1D（→64维），
Current直通，MLP fusion（196→256→16），per-dim centering，no-anchor，J16 oracle target。

| 臂 | corr | \|e\| | k7\|e\| | J中位 | 正收益% |
| --- | ---: | ---: | ---: | ---: | ---: |
| **TCN no-anchor + J16** | 0.699 | 0.093 | 0.111 | 28.0 | 10.2% |
| GT no-anchor + J16（§11.70） | **0.860** | **0.054** | **0.060** | **24.9** | **15.7%** |
| residual（参照） | 0.971 | 0.044 | — | 8.0 | 47% |

**TCN单独使用比GT差**（corr 0.699 vs 0.860）。分析原因：TCN替换了history encoder但同时也
**去掉了GT的temporal token decoder和per-knot结构**——这是一个双变量实验，无法分离"TCN
history编码是否有用"和"token decoder是否关键"。GT的corr 0.860可能主要来自token decoder
的per-knot结构（每个knot token携带geometry+sensitivity信息），而非旧的history encoder。

**正确结论**：TCN本身可能有用，但需要在**保留GT token decoder**的前提下单独替换history
encoder来验证。当前实验只证明"TCN+简单MLP fusion < GT+token decoder"，不证明"TCN history
编码无用"。下一步：TCN history + GT token decoder组合。Actor、formal validation和test保持
冻结。产物：`outputs/mppi_proposal/tcn_noanchor_j16_20260819_v1/`。

### 11.72 TCN history单变量实验：TCN不优于旧history encoder（2026-08-19）

按review指正做单变量对照：**保留GT全部组件**（temporal token decoder、geometry tokens、
reference/feedback/gradient encoders、per-dim centering、no-anchor），**只替换history
encoder**从旧TemporalConvEncoder为DilatedTCNEncoder（5 blocks、dilation 1/2/4/8/16、
hidden 64、kernel 3、末端token+global pool→128）。J16 oracle target，同fold/seed。

| 臂 | corr（3 seed） | \|e\| | k7\|e\| | J中位 | 正收益% |
| --- | --- | ---: | ---: | ---: | ---: |
| GT + **旧**history encoder | **0.848-0.860** | **0.054** | **0.060** | 24.9 | 15.7% |
| GT + **TCN** history encoder | 0.808-0.822 | 0.065 | 0.081 | 24.1 | 14.7% |

**TCN history encoder不优于旧TemporalConvEncoder**——corr反而低0.03-0.04、|e|高0.01。
J中位略好（24.1 vs 24.9）但正收益比例更低。

判定：**history encoder不是当前瓶颈**——把旧的3层stride-2 Conv换成5层dilated TCN
（感受野从~28步扩到~100步），在GT架构下没有带来精度提升。250步history的时序结构可能
已被旧encoder（或GT的token decoder通过reference几何token间接提供）充分捕获。

后续单变量实验注意：只改一个组件。产物：
`outputs/mppi_proposal/tcn_gt_j16_20260819_v1/`。

### 11.73 Current skip connection单变量实验：无显著改善（2026-08-19）

在GT no-anchor + J16 oracle基准（corr 0.860）上添加零初始化Current skip
（Linear(4→16)直接从raw current到输出，绕过全部encoder/fusion/trunk/decoder）。

| 臂 | corr（3 seed） | \|e\| | k7\|e\| | J中位 | 正收益% |
| --- | --- | ---: | ---: | ---: | ---: |
| GT baseline（无skip） | 0.848-0.860 | 0.054 | 0.059 | 24.9 | 15.7% |
| GT + Current skip | 0.848-0.862 | 0.054 | 0.058 | 24.2 | 15.5% |

**Current skip无显著改善**——corr差异在seed噪声内（±0.01），|e|/k7|e|/J中位基本持平。
Current的信息已经通过现有的MLP→fusion→trunk→decoder通路被充分利用，4维→64维的编码
没有成为瓶颈。添加直接skip不带来额外增益。

该结果与TCN实验（§11.72）的结论一致：**在GT架构下，输入端（history编码、current通路）
都不是当前瓶颈**。no-anchor corr 0.860的上限来自信息量本身（无anchor输入），而非架构
或信息通路。Actor、formal validation和test保持冻结。产物：
`outputs/mppi_proposal/gt_current_skip_j16_20260819_v1/`。

### 11.74 无attention shared decoder单变量实验：attention有效，去掉后精度下降（2026-08-19）

按review建议测试"shared per-knot decoder, 无self-attention"——即每个knot由同一个MLP独立
解码（输入=global feature + time + sensitivity + geometry），无knot间交互。保留geometry
tokens和全部其他GT组件（no-anchor、J16 oracle target、per-dim centering）。

| 臂 | corr（3 seed） | \|e\| | k7\|e\| | J中位 | 正收益% |
| --- | --- | ---: | ---: | ---: | ---: |
| GT baseline（有attention） | **0.848-0.860** | **0.054** | **0.059** | **24.9** | **15.7%** |
| GT no-attention（shared decoder） | 0.823-0.826 | 0.065 | 0.068 | 26.6 | 14.3% |

**去掉self-attention后corr下降0.03**（0.860→0.826）、|e|升高0.011。**Knot间attention
确实有效**——与§11.53 splice oracle的"计划相干性"证据一致（knot是联合协调的计划，k7需要
看到k0的context）。用户的"同一控制规律在不同时刻"先验（纯权重共享）不如学习到的knot间
交互。

至此GT架构的各组件贡献已分解清楚：
- geometry tokens（+0.09，最大贡献）
- self-attention between knots（+0.03，有效）
- per-dim centering（有效，已在J16 target实验中体现）
- TCN history encoder（无效）
- Current skip（无效）

产物：`outputs/mppi_proposal/gt_no_attention_j16_20260819_v1/`。

### 11.75 严格no-anchor输入合同A/B：first-pass字段无精度收益，clean合同保留（2026-08-20）

为落实§11.70部署合同，在`no-anchor GT + J16`上做单变量输入消融：基线仍可见74维
`feedback`与32维`gradient_context`，clean臂同时屏蔽`anchor/feedback/gradient_context`，
只允许`history/reference/current`影响输出。encoder/decoder和参数量保持不变（被禁止的输入置零），
fold/seed/target/训练配置完全配对。前向不变性单测通过：任意改变三组禁止输入，输出最大变化为
`0.0`。formal validation/test未加载。

| seed | corr baseline -> clean | 绝对误差 baseline -> clean | k7误差 baseline -> clean | J中位 baseline -> clean |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.860 -> **0.864** | 0.0544 -> **0.0539** | 0.0603 -> **0.0576** | 24.5 -> 25.5 |
| 1 | 0.852 -> **0.858** | **0.0537** -> 0.0545 | 0.0567 -> **0.0565** | 23.5 -> 24.6 |
| 2 | 0.848 -> **0.854** | **0.0556** -> 0.0558 | 0.0590 -> **0.0570** | 25.1 -> 26.4 |

判读：clean臂三个seed的逐维相关均小幅提高（约`+0.004--0.006`），整体绝对误差变化仅
`-0.0008/+0.0005/+0.0008`，k7误差三个seed均不劣；说明first-pass feedback/gradient对J16
完整中心预测**没有可测精度收益**，不构成必要信息源。真实rollout J差异不一致：clean减baseline
的均值为`-2.92/-6.06/+2.10`，episode bootstrap仅seed1不跨零，逐状态clean胜率约
`50.2%/49.9%/47.2%`；因此不得宣称clean改善cost，但也没有一致退化证据。两臂尾部仍很差，
本轮不改变“精度尚未达到部署门槛”的主结论。

**裁决**：后续Actor固定采用严格clean输入合同；first-pass feedback/gradient与anchor均不得进入
Actor。当前clean仍保留原encoder中的常量零分支，仅为公平A/B保持参数量；部署实现应在定型时
物理删除这些分支与输入槽位。下一结构实验从该clean基线继续，不能回退到first-pass字段。

复现链：
```text
scripts/model_verify/mppi_a2_actors.py::DirectNoAnchorGTCleanActor
scripts/model_verify/analyze_mppi_input_contract_ab.py
outputs/mppi_proposal/j16_noanchor_gt_clean_20260820_v1/summary.json
outputs/mppi_proposal/j16_noanchor_gt_clean_20260820_v1/input_contract_ab.json
```

### 11.76 J16 canonical teacher可行性验证：argmin抖动不是不coherent的来源（2026-08-20）

按review建议验证"canonical J16 teacher"（在近优集合中选更coherent的代表而非argmin）。
利用GT oracle已存储的8个起点（warm/zero/t0_best/t0_soft/t1_teacher/warm_random_0/1/2）
的独立优化结果做分析。

**Elite集合结构**（200状态）：
- 8个起点的最终cost差异极小（top3 relative spread中位0.0003）；
- 但action空间位置分散（max pairwise distance中位**0.296**，P75达0.987）；
- 说明存在多个近优basin（cost几乎相同但action位置不同）。

**Consensus vs Medoid vs Best的真实cost**（50状态真实rollout）：

| 量 | mean | median | P95 | 通胀率 | 差于best |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw J16 best | 4.99 | 3.04 | 20.5 | — | — |
| **consensus(mean)** | **17.91** | 3.37 | **55.3** | **+109%** | **96%** |
| medoid | 5.04 | 3.04 | 20.8 | +0.7% | 90% |

**Consensus（均值）落入"无人区"**——8个近优解在不同basin中，取均值得到的点cost爆炸
（+109%）。**Medoid有效**（+0.7%）但选的是某个basin的代表，不比best更canonical。

**Coherence对比**（200状态子集）：consensus **0.513** vs raw best **0.497**——差异
不显著。**canonical化不改善coherence。**

**判定**：argmin抖动不是J16 oracle不coherent（0.787）的来源。近优多basin确实存在
（elite spread 0.296），但无论选哪个代表（best/medoid/consensus），跨状态的coherence
都差不多——**不coherent来自状态间最优动作的真实多样性，不是优化选择的偶然性**。
canonical teacher/集合监督路线关闭。

产物：无新artifact（分析复用`dbm_direct_gt_train_20260807_v2`中的GT oracle存储）。

### 11.77 No-anchor路线阶段总结：corr 0.860为当时基线，各消融汇总（2026-08-20）

> **覆盖状态**：本节“corr 0.860是信息量天花板/架构侧已穷尽”的外推已被§11.78局部几何
> 单变量实验收紧。原消融数值仍有效，但把已有reference中的局部纵向位置显式对齐到knot后，
> corr最高达到0.867、绝对误差降至约0.051--0.054，说明仍存在小幅结构性headroom；不得再把
> 0.860称为严格信息上限。

汇总§11.63-11.75的完整探索链，no-anchor direct路线的状态如下。

**有效改进（按corr提升排序）**：

| 改动 | corr变化 | 机制 |
| --- | --- | --- |
| proximal teacher → **J16 oracle target** | 0.765→0.860 (**+0.095**) | 更coherent的监督目标(0.787 vs 0.656) |
| MLP → **GT架构**（geometry tokens + temporal attention） | 0.730→0.765 (+0.035) | per-knot结构先验 |
| per-dim centering | 包含在上述中 | 量级匹配 |

**无效改进（单变量消融全部完成）**：

| 改动 | corr变化 | 判定 |
| --- | --- | --- |
| TCN history encoder | -0.045 | history编码非瓶颈 |
| Current skip connection | ±0.00 | Current通路非瓶颈 |
| 去掉self-attention（shared decoder） | -0.034 | attention有效 |
| 更多训练（400ep vs 120ep） | ±0.00 | 已到极限 |
| J16 canonical teacher（consensus/medoid） | 不适用 | 不改善coherence |

**结论**：在no-anchor约束下，corr **0.860**、|e|=0.054是当前数据（1800状态J16 oracle）
+ 当前架构（GT token decoder）的实际天花板。剩余与residual（0.971）的差距来自**信息量
缺口**（无anchor输入≈corr 0.94→0.77），不是架构、训练或标签质量问题。

**监督目标侧已穷尽**：proximal→J16有效；J16→canonical无效；J16→更低cost空间不大
（J16→J100 gap仅0.198）。**架构侧已穷尽**：GT各组件全部消融，无遗留低垂果实。

**下一步取决于新信息源的引入**或转向actor在two-center guard中的使用方式研究。
Actor、formal validation和test保持冻结。

### 11.78 局部位置三臂A/B：显式knot-x稳定改善精度，Frenet增量不具决定性（2026-08-20）

在§11.75严格clean合同与J16 target上，按相同3-fold×3-seed执行三个单变量臂：

- `G-X`：每个knot token增加`x_ref_ego`；
- `G-F`：每个knot增加当前`[frenet_t, sin(xi), cos(xi)]`；
- `G-XF`：两者组合。

三臂均继续屏蔽anchor、feedback与原gradient；Frenet三维只是借固定调用槽传输，在legacy encoder
前仍将整个gradient张量置零，因此不存在first-pass泄漏。禁止输入不变性测试通过。

| 臂 | corr（3 seed升序） | 绝对误差（3 seed升序） | k7误差（3 seed升序） | J中位（3 seed升序） |
| --- | --- | --- | --- | --- |
| clean | 0.854/0.858/0.864 | 0.0539/0.0545/0.0558 | 0.0565/0.0570/0.0576 | 24.6/25.5/26.4 |
| **G-X** | **0.861/0.862/0.866** | **0.0512/0.0529/0.0535** | **0.0528/0.0533/0.0536** | **23.2/23.6/23.6** |
| G-F | 0.848/0.861/0.863 | 0.0530/0.0541/0.0569 | 0.0561/0.0562/0.0565 | 23.4/25.0/25.6 |
| G-XF | 0.860/0.862/0.867 | **0.0520/0.0521/0.0526** | 0.0539/0.0552/0.0564 | 22.8/24.0/25.4 |

判读：`G-X`是最稳定且最简单的精度改善——三个seed相对clean的平均绝对误差均下降
（约`0.0009/0.0041/0.0027`），k7误差一致下降，逐状态J差中位也三个seed均为改善
（`-0.26/-0.10/-0.34`）。这证明位置信息并非完全缺失，而是`x_ref_ego`此前只存在于被压缩的
reference全局编码中，没有与对应control knot显式对齐。

`G-F`单独增益弱且seed不稳定；`G-XF`在动作误差上也改善，但没有比G-X形成一致的rollout-tail
优势。所有臂P05/worst仍很差，episode-bootstrap的J均值差多数跨零，且G-X seed1受少数高cost
帧影响均值反而恶化；因此本轮只授权**结构精度基线升级到G-X**，不授权部署或formal gate。
当前`frenet_t/xi`不作为必需输入，保留为可选诊断字段。

复现链：
```text
scripts/model_verify/mppi_a2_actors.py::{DirectNoAnchorGTXActor,DirectNoAnchorGTFrenetActor,DirectNoAnchorGTXFActor}
scripts/model_verify/analyze_mppi_local_geometry_ab.py
outputs/mppi_proposal/j16_noanchor_gt_{x,frenet,xf}_20260820_v1/
outputs/mppi_proposal/j16_local_geometry_ab_20260820_v1/analysis.json
```

### 11.79 Reference token cross-attention：rollout弱正向但未过精度门，不启动history-token臂（2026-08-20）

针对“输出已token化、输入仍先压成global vector”的结构假设，采用review建议的分阶段单变量
协议。当前严格clean `G-X`作为唯一基线；第一阶段只让8个control query直接cross-attend
50个reference token，history仍沿用原global encoder，随后保留已验证有效的8-knot双向
self-attention。reference token包含原5维reference特征与2维时间编码；anchor、feedback和
first-pass gradient继续严格不可见，禁止输入不变性测试输出差为`0.0`。formal validation/test
未加载。

预注册机制门为：相对`G-X`的绝对动作误差至少改善`0.001`，且至少`2/3 seed`通过；同时
相关性不得系统性退化、配对rollout J至少`2/3 seed`不劣。只有第一阶段过门，才允许第二阶段
暴露下采样history temporal token，避免一次同时修改reference和history通路。

| seed | corr G-X -> RefCross | 绝对误差 G-X -> RefCross | k7误差 G-X -> RefCross | J中位 G-X -> RefCross |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8612 -> 0.8626 | 0.05294 -> 0.05338 | 0.05329 -> 0.05598 | 23.59 -> 23.44 |
| 1 | 0.8659 -> 0.8645 | 0.05124 -> 0.05094 | 0.05282 -> 0.05204 | 23.59 -> 22.04 |
| 2 | 0.8623 -> 0.8603 | 0.05347 -> 0.05198 | 0.05357 -> 0.05245 | 23.19 -> 23.37 |

逐状态配对的`RefCross - G-X`结果为：

| seed | J均值差 | J中位差 | episode-bootstrap 95% CI | RefCross胜率 | 绝对误差中位差 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -1.585 | -0.262 | [-4.955, 1.954] | 51.9% | +0.00043 |
| 1 | -0.834 | -0.265 | [-3.534, 1.885] | 52.0% | -0.00030 |
| 2 | -1.084 | -0.504 | [-3.911, 1.625] | 53.2% | -0.00149 |

**判定：`REFERENCE_CROSS_WEAK_POSITIVE_FAILS_REGISTERED_PRECISION_GATE`。** Reference
cross-attention的rollout方向三个seed均为小幅正向（J中位差均小于零），说明control query
直接访问reference token可能有价值；但胜率只有约`52%`，三个episode-bootstrap CI均跨零，
相关性没有一致改善，动作误差门只有`1/3 seed`通过。因此它只是弱机制证据，不能升级为新基线。

按预注册纪律，`G-X`继续作为当前结构基线，**不启动history-token第二阶段训练**。这个结果也
说明review的“两阶段而非一次token化history+reference”方法合理：它把reference直接访问的
微弱收益单独量化出来，并避免在证据不足时继续扩大结构搜索。后续若新的监督目标或更多独立
状态显著提高信噪比，可将RefCross作为候选旁路复验；当前不据此做部署或消费formal集合。

复现链：
```text
scripts/model_verify/mppi_a2_actors.py::DirectNoAnchorGTXReferenceCrossActor
scripts/model_verify/run_mppi_actor_a0_baseline.py
scripts/model_verify/analyze_mppi_reference_cross_ab.py
outputs/mppi_proposal/j16_noanchor_gt_x_refcross_20260820_v1/
outputs/mppi_proposal/j16_reference_cross_ab_20260820_v1/analysis.json
```

### 11.80 J16 elite轨迹效果审计：action多basin，但rollout effect高度canonical（2026-08-20）

§11.76已经关闭“从多个近优action里选一个更canonical的单标签”路线，但没有回答另一问题：
不同action basin是否实际产生相同的轨迹效果。为此仅使用train侧
`dbm_direct_gt_train_20260807_v2`的1800个物理状态，对每个状态选择replay cost最低的top-3
J16解，重新运行固定DBM并比较：

- action：按MPPI `noise_sigma=[0.25,0.35]`标准化的8×2 knots；
- trajectory effect：相对同一reference的50步`along/cross/yaw/vx/yaw-rate`；
- rollout距离：按原position/yaw/vx/yaw-rate权重定义，并相对该状态自身tracking尺度归一化；
- 稳健子集：必须同时满足top-3 cost spread不超过0.5%、action最大pair L2至少`0.296 sigma`。

formal validation/test未加载。DBM replay与存储cost最大绝对误差为`3.81e-6`，1800状态完整。

**全量top-3结构**：top-3相对cost spread中位仅`0.000339`（0.0339%）。标准化action的
elite组内/跨状态均值方差比为`0.0261`，reference-relative trajectory effect为`0.00327`；
后者只有前者的`12.5%`。即同一状态的多解在动作空间明显分散，但在轨迹效果空间集中得多。

真正“等cost、不同basin”的核心子集共有286状态：

| 指标 | 中位 | P90 | episode-bootstrap 95% CI（中位） |
| --- | ---: | ---: | ---: |
| action最大pair L2（sigma） | 0.442 | 0.966 | — |
| trajectory pair / 自身tracking尺度 | **0.0330** | **0.0692** | **[0.0313, 0.0363]** |
| 50步position pair RMS | **1.66 mm** | 3.88 mm | **[1.46, 1.86] mm** |
| terminal position pair | 1.50 mm | 5.83 mm | — |
| terminal yaw pair | 0.00087 rad | 0.00276 rad | — |

更严格的125状态子集（cost spread≤0.5%、action L2≥0.5 sigma）仍保持同一结论：action距离
中位`0.677 sigma`，trajectory相对距离中位仅`0.0447`，position RMS中位`2.32 mm`。
因此结论不是大量“action本来就相同”的容易样本造成的。

**判定：`TRAJECTORY_EFFECT_SUBSTANTIALLY_MORE_CANONICAL`。** §11.76的结论继续成立：
consensus/medoid不能把action单标签变得更可学，J16仍是当前最好的action监督目标。本节新增的
结论是：J16不同basin大多属于**动作不同、任务效果等价**。因此当前MSE要求Actor精确复制某个
任意action basin，监督约束过强；它会把任务等价的输出当成错误。这不授权多候选Actor，也不
重新开放canonical action标签，而是授权一次严格配对的**trajectory-equivalent objective A/B**。

建议下一步保持`G-X`结构和J16数据不变，只比较监督目标：

1. A：现有per-dimension action MSE；
2. B：冻结DBM rollout下的trajectory-effect matching，并保留控制rate/bound约束；
3. C：B加小权重action MSE作为数值正则，而非主目标。

先看train fit能否在不精确复制任意basin的情况下恢复J16 cost，再做同一episode-grouped OOF。
本节只证明同状态elite的effect等价，**尚未证明trajectory target跨episode可学**，因此不得提前
宣称Actor问题已解决。若B/C仍无OOF收益，便关闭“损失的action等价类定义错误”假设，回到
two-center guard；不继续扩大输入encoder。

复现链：
```text
scripts/model_verify/analyze_mppi_j16_elite_trajectory_coherence.py
outputs/mppi_proposal/j16_elite_trajectory_coherence_20260820_v3/analysis.json
outputs/mppi_proposal/j16_elite_trajectory_coherence_20260820_v3/per_state.npz
```

### 11.81 当前consolidation：权威结论与活跃计划（2026-08-20）

本节汇总截至§11.80的权威状态；§11.82随后覆盖其中“trajectory监督待执行”的活跃计划。
此前§11.57（2026-08-18）为中间快照。

#### 已关闭路线汇总

**Critic-gradient provider**：正式关闭（§11.38降级→`COORDINATE_CRITIC_FAIL_CLOSE_GRADIENT_MAINLINE`）。
六轮统一grouped CV均0/5 fold、0/3 seed；机制定位为position项横向修正符号歧义（§11.41-11.42：
position占已归因翻转`149/170=87.6%`，占全部翻转`149/250=59.6%`；约90%翻转涉及cross分量）。
Critic仅保留candidate ranking辅助。坐标工程（S归一化/DCT/
AT/Z）只改condition不改可用性，全部关闭。

**Search Distillation离线Actor**：主阻塞=跨episode泛化+tail。已关闭：损失工程（S归一化、stay
加权、gain阈值stay化）、KNN检索（-0.43/-0.12）、覆盖扩张（600→1800 OOF平坦非单调）、大幅标签
SNR、A2结构三臂（G/T单独持平、GT +0.091未过≥0.10机制门）。

**Direct/no-anchor路线**：corr 0.860→0.867（G-X knot-x对齐）、绝对误差~0.051-0.054；现有
encoder扩张未恢复剩余差距。已关闭：canonical action teacher（consensus落入无人区+109%）、
TCN、current skip、no-attention、RefCross（弱正向未过门）。§11.80进一步证明近优action虽处于
不同basin，但其rollout effect高度集中，因此“action MSE把任务等价解当成不同标签”是当前新开放
假设；不得再把全部差距只归因于输入信息缺口。

**残差参数化**：锚点先验是必需归纳偏置（direct臂h_train仅0.36-0.61 vs residual 0.90-0.92）。
warm start不是随机量（跨episode近邻cos 0.661），但残差坐标本身制造不稳定（coherence 0.029 vs
绝对0.655）。cost landscape超二次放大（平方误差2.36×→cost亏损6.74×）。

#### 当前生效合同

1. **two-center guard闭环**：默认配置`m1.0whr_d5`（不对称即时warm回退+硬下界零违规），
   nominal成熟段+32.4%、recovery +6.0%、high +0.6%，三场景严格优于warm-only。对称hysteresis
   为显式opt-in性能档。warm hard-return合同已实现在`hard_guard_action_sequence`。
2. **Actor输入**：严格clean/no-anchor合同——anchor、first-pass feedback、gradient context
   均不得进入部署Actor（§11.75）。
3. **结构精度基线**：no-anchor J16 `G-X`（`x_ref_ego`与control knot显式对齐），corr 0.861-0.866。
4. **rollout预算**：默认256（§8.4修正）。
5. **100kph数据隔离**：不混入现有split（§10契约）。
6. **Actor、formal validation/test**：保持冻结。

#### 活跃计划

1. **two-center guard闭环定型**已基本完成（§11.60-11.61）；剩余：(a)多seed×三场景闭环A/B的
   formal gate评估；(b)A2-GT作为proposal来源对照（叠加项，不独立主线）。
2. **No-anchor监督目标**：G-X为当前结构基线；§11.82已完成action MSE、trajectory matching和
   hybrid对照。pointwise trajectory matching关闭；若继续，只做原始可微J50 task-loss单seed
   train-fit门。RefCross不继续扩展。
3. **Pending非阻塞项**：horizon加权候选的原始J50重rollout验证；rho与Phase 1b状态join；
   expansion 1350状态的anchor策略决策。

#### 编号消歧与引用规范

重复编号共9对：§11.48-11.55（8对）+ §11.61（1对），详见§11.57.1消歧表。§11.62-§11.82按
行序唯一。引用一律用"编号+描述"双重消歧（如"§11.53 A0/A0.1早停修复与2x2重跑"），禁止裸引
冲突编号。

Actor、formal validation和test保持冻结。

### 11.82 Trajectory-equivalent监督A/B：显著缓解action-MSE灾难，但仍未战胜a0（2026-08-20）

按§11.80授权执行监督目标单变量实验。固定严格clean/no-anchor `G-X`、1800个train J16状态、
episode-grouped 3-fold、seed 0、batch 256、120 epoch；formal validation/test未加载。三臂为：

- A：原始per-dimension action MSE；
- B：冻结DBM下50步trajectory-effect matching，按原`position/yaw/vx`权重，另惩罚超过teacher
  的control-rate cost；
- C：B加按fold训练尺度归一化的`0.05 × action MSE`正则。

| 臂 | train H | OOF H | corr | action误差 | early-steer误差 | J mean/median | 正收益% | P05 / worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A action MSE | -6.573 | -7.646 | 0.845 | 0.0582 | 0.0690 | 100.40 / 24.97 | 14.9% | -487.8 / -2394.5 |
| B trajectory | -0.885 | -1.162 | 0.569 | 0.1261 | 0.1341 | 28.73 / 12.80 | 24.4% | -89.1 / -580.0 |
| C hybrid 0.05 | -0.795 | -1.148 | 0.697 | 0.1027 | 0.1213 | 28.58 / 11.79 | 28.2% | -93.9 / -584.3 |

相对A的逐状态配对结果很强：B的J均值/中位差为`-71.67/-9.48`，`75.9%`状态更低，
episode-bootstrap均值差95% CI为`[-85.02,-58.60]`；C为`-71.82/-10.54`，`77.6%`状态
更低，CI为`[-84.87,-59.26]`。因此§11.80的机制判断得到训练侧支持：action MSE要求复制
任意basin，确实制造了严重损失；action correlation下降并不等价于任务性能下降。

但B/C的train与OOF恢复率仍全部为负，说明两者都没有战胜a0，不能部署。120轮中B的三个fold
最佳epoch为`112/119/119`，C为`120/119/119`，排除了30轮短训不足；按seen-fit停止门，不再
扩到seed 1/2。

失败机制不是§11.80“轨迹effect等价”结论错误，而是**逐点teacher-trajectory距离并非原始
reference cost的正确代理**。令teacher相对reference误差为`r`，Actor相对teacher的轨迹差为
`delta`，则原跟踪cost差包含：

```text
J_track(actor) - J_track(teacher)
  = ||delta||_W^2 + 2 <r, delta>_W
```

B主要最小化第一项，却没有控制带符号的交叉项；即使Actor接近teacher轨迹，只要误差沿原跟踪
误差方向叠加，原始J仍可上升。C的5% action正则提高corr并略改善train H，但没有修复这个目标
错配。继续扫trajectory/action权重没有依据。

**判定：`TRAJECTORY_EQUIVALENT_LOSS_REDUCES_ACTION_MSE_FAILURE_BUT_FAILS_A0_BASELINE`。**
当前证据关闭“pointwise teacher-trajectory matching”作为部署监督目标，但没有关闭任务等价
训练。若继续该方向，唯一有机制依据的下一臂是直接优化冻结DBM下的原始可微`J50`任务目标，
而不是再拟合某条teacher action或trajectory；该实验仍需先过单seed train H>0门，再决定是否
扩seed。two-center guard与部署Actor保持冻结。

复现链：
```text
scripts/model_verify/run_mppi_actor_a0_baseline.py --supervision-objective {action_mse,trajectory_effect,trajectory_hybrid}
scripts/model_verify/analyze_mppi_trajectory_supervision_ab.py
outputs/mppi_proposal/j16_gt_x_action_mse_seed0_bs256_20260820_v1/
outputs/mppi_proposal/j16_gt_x_trajectory_effect_seed0_20260820_v1/
outputs/mppi_proposal/j16_gt_x_trajectory_hybrid_seed0_20260820_v1/
outputs/mppi_proposal/j16_trajectory_supervision_ab_20260820_v1/analysis.json
```

### 11.83 完整absolute-action value Critic：排序可用，近最优梯度仍不可用（2026-08-20）

本节回应“既往Critic结论多来自anchor附近残差梯度，改为完整绝对动作价值后是否仍不可用”。
实验严格拆成两个门：先判断标量value/ranking，再独立审计action autograd gradient；禁止用梯度
失败覆盖value成功，反之亦然。

#### 11.83.1 合同与候选覆盖

- 数据只使用1800个train状态，3-fold episode-grouped × 3 seed；formal validation/test未读取；
- 每状态24个完整absolute 8×2 knots：GT v1的8个异质原始起点、v1的8个优化中心、v2的8个
  继续优化中心，共43200条`(s,a_abs,J_direct)`；
- Critic输入为`history + reference + current + candidate absolute action`，不输入warm、anchor、
  residual、feedback或gradient context；
- 输出单个`log(1+J_direct)`，loss为fold-train尺度下的smooth-L1 value回归，加同状态、cost差
  ≥0.1的pairwise ranking；固定120 epoch，不用outer heldout早停；
- 结构为state encoder + 8个action-knot token的双向temporal encoder + scalar value head；本轮不
  更新Actor。

24候选cost均值84.43、全体中位5.68、最大5988.62；raw warm均值27.92、bank oracle均值4.83。
因此候选集同时含坏动作、warm/teacher与多个近优basin，不是anchor附近的小残差集合。32状态
tiny-set先验门通过：log-value Pearson 0.9931、material-pair准确率0.9983、top-1 headroom回收
0.9992，排除了模型、坐标、autograd或optimizer的基本工程故障。

#### 11.83.2 episode-heldout value/ranking结果

| seed | log-value Pearson | material pair acc | large-gap acc | top-1 R | top-4 R | harmful top-1 | top-1 / top-4 regret |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.945 | 0.956 | 0.971 | 0.976 | 0.9986 | 0.28% | 0.549 / 0.033 |
| 1 | 0.951 | 0.964 | 0.977 | 0.994 | 0.9986 | 0.11% | 0.130 / 0.032 |
| 2 | 0.947 | 0.960 | 0.972 | 0.950 | 0.9985 | 0.56% | 1.162 / 0.035 |

top-1 episode-bootstrap 95% CI分别为`[0.951,0.994]`、`[0.990,0.997]`、
`[0.902,0.983]`；top-4下界均≥0.997。2.8 m/s层的material-pair准确率仍为
0.919/0.928/0.921，top-4 R为0.996/0.996/0.996，没有复现旧g0-hard层的系统性反向。

候选bank含16/24个优化点，所以top-4本身偏易，必须同时看两个防捷径检查：

- 匹配的random-top4已能取得R约0.991、regret 0.212；Critic top-4将regret进一步降至
  0.032-0.035（约缩小6倍）；
- 分stage material-pair准确率仍为raw `0.897-0.907`、v1 optimized `0.871-0.885`、v2
  refined `0.884-0.905`，说明结果不只是识别`raw/optimized`字段。

因此value侧正式判定为：
`ABSOLUTE_ACTION_VALUE_CRITIC_USABLE_FOR_RANKING`。这只授权在同分布candidate bank上做
离线预排序/shortlist，不授权对任意OOD action外推，也不表示Actor可沿其梯度更新。

#### 11.83.3 独立absolute-action gradient审计

value门通过后，使用相同episode-heldout checkpoint，在三个绝对动作位置比较
`d log(1+J) / d a_abs`与冻结DBM autograd真值；两者处于同一8×2 action坐标，DBM cost replay
最大误差小于`1e-3`。结果如下：

| anchor | cosine median（3 seed） | cosine P10（3 seed） | norm ratio median | 真梯度norm中位 |
| --- | --- | --- | --- | ---: |
| warm | 0.620 / 0.630 / 0.628 | -0.297 / -0.314 / -0.392 | 0.697 / 0.693 / 0.701 | 7.89 |
| raw random start | 0.883 / 0.897 / 0.892 | 0.314 / 0.442 / 0.343 | 1.001 / 1.003 / 0.983 | 4.64 |
| bank best | 0.016 / 0.024 / 0.041 | -0.618 / -0.614 / -0.590 | 54.1 / 49.4 / 53.9 | 0.057 |

远离极小点、value信号强的raw动作处，网络确实学到质量较好的全局下降方向；这说明旧Critic
失败不能概括成“任何value网络都学不到action响应”。但warm处P10仍显著为负；bank-best处真
梯度已很小，离散value/ranking监督没有约束局部导数，网络以约50倍幅值输出近随机方向。
Actor更新恰恰需要warm/近优区域的可靠局部导数，因此仍未通过原门
`median>=0.70, P10>=0, norm ratio in [0.5,2]`。

梯度侧正式判定为：`ABSOLUTE_VALUE_CRITIC_GRADIENT_NOT_USABLE`。这不是value模型倒退，而是
明确了用途边界：

1. 可保留：对search产生的多个完整absolute center做candidate ranking/pre-ranking；
2. 不可恢复：把Critic autograd直接作为Actor在warm/近优区域的策略梯度；
3. 不回到loss扫描：若未来需要局部优化，仍优先真实query search；Critic只做粗筛，top候选再由
   DBM/Query真实cost裁决；
4. 当前结果尚未覆盖Actor在线访问的OOD候选分布和闭环传导；使用前需做actor/search-bank
   candidate A/B与two-center guard，formal validation/test继续封存。

复现链：

```text
scripts/model_verify/run_mppi_absolute_action_value_critic_cv.py
scripts/model_verify/analyze_mppi_absolute_action_value_critic_gradient.py
outputs/mppi_proposal/absolute_action_value_critic_20260820_v1/dataset_manifest.json
outputs/mppi_proposal/absolute_action_value_critic_20260820_v1/tiny_overfit.json
outputs/mppi_proposal/absolute_action_value_critic_20260820_v1/summary.json
outputs/mppi_proposal/absolute_action_value_critic_20260820_v1/strict_oof_predictions.npz
outputs/mppi_proposal/absolute_action_value_critic_20260820_v1/gradient_audit.json
outputs/mppi_proposal/absolute_action_value_critic_20260820_v1/gradient_audit.npz
```

#### 11.83.4 近优平坦区的目标修正与stationarity实验预声明

§11.83.3的`bank-best cosine≈0`不能单独解释成性能缺陷：该处DBM真梯度norm中位仅`0.057`，
约为warm处`7.89`的1/138；不同refined候选cost差也很小。近零向量的方向cosine在数值上病态，
部署真正需要的不是从平坦区继续找精确方向，而是**不要让Critic制造大斜率并把已近优的Actor推
出低cost basin**。当前明确缺陷是预测norm约高估49-54倍，而不是bank-best方向未过0.70门。

因此后续gate按区域拆分，覆盖§11.83.3中“所有anchor共用方向门”的过严解释，但不覆盖
“Critic gradient尚不可直接更新Actor”的安全判定：

1. raw高信号区：保持value/ranking与梯度方向能力，cosine median不得显著低于当前0.88；
2. warm改善区：仍要求实际DBM受限小步有正的聚合gain，并报告P05/worst；
3. bank-best平坦区：不再要求cosine，主门改为预测gradient norm收缩、受限更新的真实cost
   不恶化、stay比例提高；
4. Actor步长必须保留gradient幅值，只做逐维上界clip；禁止把gradient归一化后固定走满trust
   radius，否则会重新放大平坦区的无意义方向；
5. formal validation/test、部署Actor保持冻结。

单变量实验为`V0`原absolute value/ranking checkpoint对照`S1`：保持数据、输入、模型、value
loss、ranking loss、fold/seed/epoch全部不变，仅在每个train state的24候选bank-best处加入：

```text
L_stationary = lambda_s * || d log(1+Q_phi) / d a_abs ||^2
```

这里使用预测标量对absolute action的二阶反传，不把DBM解析梯度作为网络输入，也不监督近零
方向。先以单seed三fold机制pilot检查：(a)value/ranking gate不退化；(b)bank-best预测norm至少
下降10倍；(c)raw方向保持；(d)warm/bank-best的非归一化、0.05sigma上界小步DBM replay尾部
不恶化。若机制成立再扩3 seed；若stationarity导致整个action场变平或warm收益退化，则停止，
不进入新的loss权重扫描。

机制pilot的`S1, lambda_s=0.1`先行结果：单seed三fold的OOF value/ranking保持（material-pair
`0.958-0.962`、top-1 R `0.969-0.996`、top-4 R `0.998-0.999`）；raw随机起点梯度
cosine median `0.880`、P10 `0.331`、norm ratio `0.977`，没有把全局action场压平。bank-best
norm ratio从V0的`54.1`降至`7.96`（约6.8倍收缩），但未达到预声明的至少10倍；warm median
从`0.620`降至`0.575`、P10仍`-0.300`。因此S1定性支持stationarity机制但强度不足。

在继续Actor小步replay前，仅授权一个非扫描式强约束臂`S2, lambda_s=1.0`，其他条件完全相同。
S2目标是让bank-best norm继续接近真实量级，同时保持material-pair≥0.95、raw cosine median
≥0.85且warm不进一步明显退化。S2后无论成功与否均停止stationarity权重扫描，再统一比较V0/S1/S2
的受限非归一化DBM小步。

#### 11.83.5 Stationarity完整结果：平坦区抑制成立，但不能单独授权Actor梯度

`S1 lambda_s=0.1`已扩展为3-fold×3-seed。标量value/ranking保持可用：strict pooled
material-pair accuracy为`0.960/0.952/0.963`，top-4 R为`0.9988/0.9990/0.9994`；top-1 R
为`0.985/0.929/0.994`。相比V0，bank-best预测gradient norm中位从约`2.82`降至
`0.369/0.558/0.330`，对应真梯度norm ratio `7.96/10.39/7.82`。这仍未校准到1，但已使
平坦区更新幅值下降约5-9倍。

副作用有seed差异：raw随机起点cosine median为`0.880/0.676/0.893`，warm为
`0.575/0.287/0.611`。seed 1说明stationarity可沿共享参数把部分有用响应也压弱，因此S1不是
“稳定恢复gradient provider”。唯一强约束臂`S2 lambda_s=1.0`进一步把单seed bank-best norm
ratio压至`2.47`，但raw/warm cosine降至`0.830/0.445`，material-pair降至`0.945`；按预声明判
为过强，不扩seed、不继续扫权重。

随后在全部train状态上做冻结DBM真实小步replay。更新严格保留幅值：

```text
delta_a = clip(-eta * d log(1+Q) / d a_abs,
               -0.05*sigma, +0.05*sigma)
```

不做gradient normalization。最保守的`eta=0.001`结果为：

| anchor/arm | mean gain（3 seed） | P05（3 seed） | worst（3 seed） | median step |
| --- | --- | --- | --- | --- |
| warm V0 | 0.879 / 0.914 / 0.871 | -0.647 / -0.642 / -0.729 | -15.8 / -21.2 / -25.0 | ~0.0045 sigma |
| warm S1 | 0.822 / 0.416 / 0.842 | -0.552 / -0.468 / -0.509 | -12.4 / -13.0 / -18.0 | 0.0032-0.0041 sigma |
| bank-best V0 | -0.0578 / -0.0380 / -0.0445 | -0.203 / -0.139 / -0.146 | -9.69 / -3.14 / -11.59 | ~0.0024 sigma |
| bank-best S1 | -0.0093 / -0.0014 / -0.0040 | -0.0047 / -0.0028 / -0.0068 | -4.58 / -0.39 / -1.97 | 0.0003-0.0005 sigma |

所以用户提出的核心机制得到直接验证：**近优区无需辨识微小方向；只要保留gradient幅值并抑制
虚假norm，真实cost回归即可显著收缩。** S1在bank-best的mean/P05改善约5-50倍，同时warm
仍保持正的聚合gain，且尾部比V0稍好；S2则保护更强但牺牲warm收益，不取。

额外固定gradient-norm stay门`||g_pred||<1.0`时，S1三seed在warm只stay
`0.9%/2.3%/2.0%`，在bank-best stay `81.1%/78.6%/81.4%`，证明网络已学到可用于多数状态的
平坦度分离。但该门没有消除少数“近优却预测高norm”的异常点，worst仍为负；提高门到2.0可使
bank-best stay约92-94%，仍抓不住这些高norm离群点。

最终判定为
`STATIONARITY_SUPPRESSES_FLAT_REGION_UPDATES_BUT_GRADIENT_ACTOR_STILL_REQUIRES_TRUE_COST_GUARD`：

1. 覆盖旧结论中“bank-best cosine低就是关键失败”的解释；近零真梯度不要求方向门；
2. 不覆盖“纯Critic autograd不可独立授权Actor更新”：warm P05仍负、seed 1响应退化、近优离群
   worst仍存在；
3. 若后续做Actor-gradient pilot，只允许`S1 + 非归一化极小步 + 训练期DBM/Query真实cost
   accept/reject`，负动作仍进入Critic replay但不得无条件提交为Actor新中心；
4. 在线仍用one-shot Actor和warm+Actor two-center guard，不把Critic放入部署闭环；主路线仍是
   Search/候选ranking，S1是可选训练辅助而非恢复SAC主线；
5. formal validation/test继续封存。

新增复现链：

```text
scripts/model_verify/analyze_mppi_stationarity_step_ab.py
scripts/model_verify/analyze_mppi_stationarity_norm_gate.py
outputs/mppi_proposal/absolute_action_value_critic_stationary_20260820_v1/
outputs/mppi_proposal/absolute_action_value_critic_stationary_strong_20260820_v1/
outputs/mppi_proposal/absolute_action_value_critic_stationarity_step_ab_20260820_v1/analysis.json
outputs/mppi_proposal/absolute_action_value_critic_stationarity_step_ab_20260820_v1/norm_gate_analysis.json
outputs/mppi_proposal/absolute_action_value_critic_stationarity_step_ab_20260820_v1/per_state.npz
```

#### 11.83.6 尾部逐帧归因：不是DBM不稳定，而是两类Critic误差

按S1、`eta=0.001`逐seed取gain最差5%（每seed每anchor 90帧），重新加载同一冻结DBM做逐项
cost replay。重算与§11.83.5存储gain的最大绝对误差仅`2.26e-5`，无随机seed、观测噪声或
rollout不确定性；因此“DBM模型下不应该随机不稳定”的判断正确。所谓尾部来自两个确定性机制：

**A. warm尾部是Critic跨episode方向泛化错误。** 真梯度并不小：tail norm中位
`9.95/11.07/10.73`，而Critic预测norm中位`6.88/4.16/6.31`；两者cosine中位
`-0.582/-0.517/-0.638`。真实一阶近似与实际cost increase在全量上的相关系数
`0.996/0.996/0.995`，在tail仍为`0.971/0.972/0.991`；真实一阶项直接预言cost上升的比例
`97.8%/98.9%/96.7%`。所以这些帧不是曲面粗糙或DBM非线性突然失效，而是Critic把明确的大
斜率方向学反了。

代表帧`state_index=1120, episode_056/step_000250, steady, 2.4m/s`在三个seed均为最差：warm
cost `137.787`，真梯度norm `13.391`，三seed cosine `-0.729/-0.808/-0.801`，实际cost分别增加
`12.409/12.984/17.966`，一阶预测为`12.205/12.727/17.503`。逐项增加中position为
`10.619/11.115/15.344`，yaw为`1.686/1.752/2.457`，其余接近零。抽取的15个warm top case
全部由position项主导。

**B. bank-best尾部是平坦极小点的二阶曲率加虚假高norm。** tail真梯度norm中位仅
`0.117/0.087/0.122`，预测norm却为`3.62/2.47/3.38`；实际增量与一阶项的tail相关只有
`0.026/0.618/0.190`，中位实际increase约`0.007-0.017`，而一阶预测仅约`1e-4-2e-4`。
即`Delta J ≈ 0.5 delta_a^T H delta_a`主导：方向本身没有意义，任何非零离开都可能涨cost。

代表帧`state_index=1793, episode_089/step_000406, combined_recovery, 2.8m/s`的bank-best cost
为`32.403`、真梯度norm仅`0.082`。seed 0预测norm `17.724`、cosine `-0.758`，小步实际增加
`4.583`，但一阶仅预测`0.037`；position项单独增加`4.526`。seed 1同帧预测norm `7.210`，实际
增加`0.393`，仍由position主导。15个bank-best top case中13个由position、2个由yaw主导。

尾部不是单一场景bug，但有结构富集：

- 2.4/2.8m/s占warm tail `47.8%-64.4%`、bank-best tail `57.8%-65.6%`，总体仅40%；
- bank-best action贴`|a|>0.98`边界的比例为tail `36.7%-44.4%`，总体29.5%，是放大项而非
  必要条件；warm边界比例与总体接近；
- warm三个seed的P05集合两两重叠21-27/90、三者共同10/90；bank-best为21-25/90、共同11/90。
  即少量物理帧系统性困难，但大多数tail仍有训练seed成分，不能把它描述成固定DBM异常工况。

这轮归因收窄后，处理方法也必须拆开：

1. bank-best：不再尝试方向学习。用独立flat/stay判断或local value span把更新置零；避免继续用
   共享scalar-Q stationarity加权，因为S2已证明会压弱warm/raw响应；
2. warm：stationarity不能修复方向反转。只靠步长缩小能降低损失但不能改符号；若用于训练探索，
   必须对Critic提出的新动作查询真实DBM/Query cost，坏动作进入replay但不提交Actor；
3. 更干净的结构是把`value/ranking`、`flat/stay`和`move direction`分头：前两者已有正证据，
   后者只在真实cost accept/reject约束下试验，禁止让一个共享梯度同时承担三种语义；
4. 部署仍由MPPI真实rollout和two-center warm候选裁决，不使用Critic梯度。

复现产物：

```text
scripts/model_verify/analyze_mppi_stationarity_tail_cases.py
outputs/mppi_proposal/absolute_action_value_critic_stationarity_step_ab_20260820_v1/tail_case_analysis.json
```

#### 11.83.7 路线裁决：永久冻结离线单次梯度，开放持续在线Actor--Critic pilot

用户裁决并结合§11.83.1-§11.83.6证据，以下两类方案永久标记
`PERMANENTLY_FROZEN`：

1. Critic离线训练后冻结，随后用其梯度更新Actor；
2. 每状态只做一次Critic梯度Actor更新，不查询新Actor动作的真实DBM/Query cost、不写Replay、
   不继续更新Critic。

这不是暂时缺少超参，而是信息合同错误：warm尾部的明确符号错误只有新action的真实reward才能
纠正；bank-best的平坦性也不能靠一次方向预测解决。后续禁止以“更小步长”“新坐标”“新loss”或
“多训练epoch”为名重开上述分支。若新路线失败，回Search/ranking与two-center guard。

唯一重新开放的Actor--Critic方向是**持续在线交互**：Actor每轮生成新的完整absolute action，
冻结DBM计算真实`J(s,a)`，所有好坏样本进入Replay，Twin Critic高频更新后Actor低频更新，再用
新Actor继续查询。固定DBM单步问题满足`Q*(s,a)=-log(1+J_direct)`，属于SAC-style contextual
bandit，不需要`next_state`和Bellman bootstrap；车辆状态递推的多步SAC是以后独立阶段。

执行合同、Replay配比、`20 Critic : 1 Actor`更新频率、OAC-0至OAC-4门槛、停止条件与产物格式已
单独固化在：

```text
car_foundation/docs/mppi_online_actor_critic_pilot_plan_20260820.md
```

当前只授权OAC-0（合同/基线）和OAC-1（fold 0×3seed、Actor冻结、10轮actor-visited Critic
burn-in）。OAC-1通过value/ranking、Twin稳定、flat/stay与错误样本后续修正四项门后，才允许
OAC-2持续Actor更新。训练中坏动作不删除；accept/reject只控制selected checkpoint晋级，不阻断
exploration/latest Actor产生训练数据。部署仍为one-shot Actor + two-center warm guard，Critic
不进入部署，formal validation/test继续封存。

### 11.84 OAC-0/OAC-1执行：value/ranking通过，flat与错误纠偏未过门，Actor继续冻结（2026-08-20）

按§11.83.7与`mppi_online_actor_critic_pilot_plan_20260820.md`完成fold 0 × 3 seed的前两阶段。
OAC-0固化60个训练episode/30个heldout episode、clean/no-anchor G-X Actor、V0 Twin Critic、
固定DBM/weights和全部source hash。OAC-1每seed采10轮×256状态×6动作=15360条新样本，合计
46080次DBM rollout；所有好坏动作均写Replay，每轮20次Critic update，Actor update严格为0。

三seed的actor-visited material-pair accuracy为`0.877/0.876/0.880`，Pearson范围
`0.939--0.947`，prediction std ratio约`1.04--1.06`，value/ranking与幅值门通过。lag-2坏动作
总体排序也达到`0.937/0.941/0.933`。但初始排错坏动作的最终纠正率只有
`0.370/0.446/0.332`，低于0.50门。

独立flat/stay头在阈值0.5下的bank-best recall为`0.779/0.811/0.751`，warm false-stay为
`0.246/0.196/0.275`。由于原计划没有预注册概率阈值，另在fold-train上按false-stay≤0.10选择
阈值，再到fold-0 heldout检验；recall仍仅`0.537/0.565/0.320`。因此失败不是概率校准，而是
当前flat监督/表示没有同时区分near-best与material warm。

补充OAC-0 heldout evaluator显示，在线适配后的历史bank material-pair accuracy仍强
（`0.942/0.949/0.937`），但比初始化下降`1.7--2.5pp`；actor-visited 2.8m/s层只有约`0.825`。
所以本轮不是“持续在线Critic完全无效”，而是**主体排序可用，但尚未达到允许Actor接收梯度的
完整条件**。联合qualification为`OAC1_BURNIN_GATE_FAIL_ACTOR_REMAINS_FROZEN`，0/3 seed通过，
OAC-2未启动。

独立validator qualification为`OAC01_INDEPENDENT_VALIDATION_PASS`：三seed各抽96条DBM replay
误差均为0，Replay只含fold-train状态，角色计数完整，Actor运行前后hash一致，复算指标/gate与
summary零误差。第一次运行曾因同轮重复状态被`round+state`错误合组而在写最终产物前被断言
终止；正式v1改为显式唯一interaction id并从头重跑，失败目录不属于实验结果。

下一步仅允许Actor冻结的OAC-1B：改独立flat/local-span监督、优先回放未纠正pair和2.8m/s状态、
加强历史bank rehearsal以抑制遗忘；不得降低既有gate来授权OAC-2。formal validation/test继续
封存。

```text
scripts/model_verify/train_mppi_online_absolute_sac.py
scripts/model_verify/validate_mppi_online_absolute_sac.py
scripts/model_verify/analyze_mppi_online_absolute_sac_oac1.py
outputs/mppi_proposal/online_absolute_sac_oac01_20260820_v1/{summary.json,validator_report.json,oac1_gate_analysis.json}
```

### 11.85 固定Replay学习曲线：OAC-1主要欠训练，800步修复纠错与高速门（2026-08-20）

在不新增DBM数据、不修改loss/标签/Replay配比、不构造Actor的前提下，将OAC-1三seed从200步
继续训练到400/800/1600步。父checkpoint未保存optimizer state，续训以相同LR重新初始化AdamW；
因此结果证明额外优化有效，但不能区分“更多步数”与“optimizer restart”各自贡献。

初始排错坏动作纠正率从200步的`0.370/0.446/0.332`升至800步
`0.572/0.617/0.529`，1600步为`0.634/0.666/0.584`；原0.50门在800步已3/3通过。
actor-visited 2.8m/s pair accuracy同步从约0.825升至800步`0.867/0.865/0.869`和1600步
`0.881/0.885/0.892`。总体pair accuracy最终达到`0.918/0.924/0.923`。

额外训练没有以遗忘换取新Replay拟合：历史fold-0 heldout bank pair accuracy从
`0.942/0.949/0.937`升至`0.947/0.949/0.944`。这将§11.84的主要归因从“Replay信息/表示可能
不足”改为**200次Critic update明显不足**；后续在线循环不能继续使用每10轮合计仅200步作为
burn-in预算。

flat/stay在原标签下也改善，说明不能直接断言必须改标签。1600步、全1800状态+固定0.5阈值的旧
gate已有seed 0/2完整通过；但train-only阈值到fold-0 heldout的recall为
`0.935/0.940/0.962`，false-stay仍为`0.100/0.105/0.117`，严格仅seed 0通过。因此当前裁决是：

1. `MORE_CRITIC_OPTIMIZATION_CAN_FIX_ERROR_CORRECTION`成立；
2. 800步是纠错/高速的最低已验证有效点，1600步主体更好；
3. flat监督修改不再是立即必做，应先讨论是否接受heldout false-stay边界或设计预注册补审；
4. Actor继续冻结，不能用全量训练状态2/3通过覆盖严格heldout的1/3；OAC-2未启动。

独立validator确认Replay未变、新DBM rollout=0、Actor未构造、12个checkpoint全部零误差复算，
formal validation/test未加载。

```text
scripts/model_verify/run_mppi_oac1_fixed_replay_curve.py
scripts/model_verify/validate_mppi_oac1_fixed_replay_curve.py
outputs/mppi_proposal/online_absolute_sac_oac1_fixed_replay_curve_20260820_v1/{summary.json,validator_report.json}
```

### 11.86 固定Replay扩展到6400步：主Critic继续改善，flat跨episode校准出现分化（2026-08-20）

为检验§11.85的改善是否只发生在800--1600步，保持三seed各自的15360条Replay、网络、loss、
标签、采样配比和评估状态完全不变，把同一续训轨迹扩展到`3200/6400`步。本轮仍不新增DBM
rollout、不构造Actor、不访问formal validation/test。由于原始200步checkpoint没有optimizer
state，AdamW只在200步边界重置一次；之后`200->400->800->1600->3200->6400`连续训练，扩展
checkpoint已保存Critic与flat optimizer state。因此`1600->6400`的增量不再混有第二次optimizer
restart。

主value/ranking结果明确支持“更多训练步仍有收益”：

| updates | 总体pair accuracy（seed 0/1/2） | 初始排错纠正率 | 2.8m/s pair accuracy | heldout bank pair accuracy |
| ---: | --- | --- | --- | --- |
| 1600 | 0.918/0.924/0.923 | 0.634/0.666/0.584 | 0.881/0.885/0.892 | 0.947/0.949/0.944 |
| 3200 | 0.932/0.935/0.933 | 0.670/0.725/0.676 | 0.902/0.907/0.902 | 0.947/0.946/0.943 |
| 6400 | 0.945/0.948/0.948 | 0.762/0.780/0.734 | 0.921/0.925/0.927 | 0.950/0.945/0.951 |

从1600到6400，初始排错纠正率再提高`+0.128/+0.114/+0.150`，总体pair accuracy提高约
`2.4--2.7pp`，2.8m/s提高约`3.5--4.0pp`，尚未观察到主Critic平台。历史heldout bank相对
200步为`+0.008/-0.003/+0.013`：两个seed改善，一个seed轻微退化；seed 1的top-1 headroom
recovery也从200步`0.992`降到6400步`0.980`。因此结果不支持“全面过拟合”，但也不支持无约束
地永远选择最后checkpoint，后续应以actor-visited纠错、高速层和heldout bank联合选择。

flat/stay给出相反的泛化信号。全1800状态、固定0.5阈值的旧口径在3200/6400步已三seed全部通过；
但只在fold-train选阈值后到fold-0 heldout检验，false-stay从1600步的
`0.100/0.105/0.117`上升到3200步`0.122/0.115/0.138`，6400步进一步变成
`0.142/0.145/0.158`，虽然heldout recall已达`0.983/0.992/0.978`。这说明更多训练解决的是
scalar value/ranking和错误纠偏，不会自动解决flat跨episode校准，继续训练还可能把flat头推向
高召回、较高误报。

本轮qualification为`EXTENDED_CRITIC_TRAINING_CONTINUES_TO_IMPROVE`，其适用对象是主Critic，
不是OAC-2放行。建议后续burn-in不低于1600步，3200步是更稳妥的主Critic起点；6400步仍有价值，
但必须按heldout约束选checkpoint。严格flat gate仍未通过，Actor继续冻结，第二步flat监督/校准
方案仍留待单独裁决。

独立validator复算18个checkpoint指标，最大误差为0，并确认Replay hash不变、新DBM rollout=0、
Actor/Actor optimizer未构造、formal validation/test未加载。

```text
scripts/model_verify/run_mppi_oac1_fixed_replay_curve.py
scripts/model_verify/validate_mppi_oac1_fixed_replay_curve.py
outputs/mppi_proposal/online_absolute_sac_oac1_fixed_replay_extended_20260820_v1/{summary.json,validator_report.json}
```

### 11.87 联合Value--连续移动系数：替代独立BCE flat头，OAC-1B机制门通过（2026-08-20）

针对§11.86中“Value持续改善但独立flat概率校准恶化”的分化，完成三轮Actor冻结、零新增rollout
的联合训练。Twin Value Critic从6400步checkpoint及其optimizer state继续更新；辅助头直接读取
两路物理log-value、Twin disagreement和16维同状态动作差，辅助loss反传Twin Critic。全程不构造
Actor/Actor optimizer，不读取formal validation/test。

三种连续目标的归因如下：

1. 非负`log1p(gap)`：heldout相关约`0.95`，但`gap=0.1`物理阈值的best recall仅约
   `0.73--0.80`；大gap主导幅值，近零区仍失准；
2. 有符号同状态`delta J`：解决“Critic参考候选不是真best”时未见负差值的问题，相关约
   `0.956--0.960`，但近零幅值仍不足；
3. 正式系数臂：直接监督
   `c_move=max(delta J,0)/(max(delta J,0)+0.1)`，并平衡stay/move/random动作对；固定
   `c_move<=0.5`等价于物理`gap<=0.1`，不再需要概率阈值漂移校准。

系数臂400/800/1600/3200步共12个post-training checkpoint全部通过固定物理门。3200步独立复算：

| 指标 | seed 0 | seed 1 | seed 2 |
| --- | ---: | ---: | ---: |
| actor-visited pair accuracy | 0.953 | 0.953 | 0.953 |
| heldout bank pair accuracy | 0.951 | 0.948 | 0.950 |
| heldout top-1 recovery | 0.995 | 0.989 | 0.992 |
| 2.8m/s pair accuracy | 0.935 | 0.932 | 0.941 |
| 初始排错纠正率 | 0.801 | 0.810 | 0.776 |
| lag-2坏动作排序 | 0.971 | 0.968 | 0.969 |
| move coefficient MAE | 0.103 | 0.108 | 0.106 |
| bank-best recall（固定0.5） | 0.923 | 0.908 | 0.905 |
| warm false-stay（固定0.5） | 0.008 | 0.008 | 0.013 |

相对联合训练前的heldout bank pair，变化为约`+0.0004/+0.0027/-0.0009`，说明低权重系数loss
没有以Value遗忘换取flat门通过。直接系数监督的全范围gap相关降到约`0.84--0.86`，这是预期权衡：
它优化的是Actor实际需要的near-zero连续移动尺度，而非大gap的绝对回归。

独立validator对最终三seed逐项零误差复算，确认Replay hash、optimizer、Actor零更新及split封存。
训练`summary.json`沿用了非负gap臂的通用“校准阈值门”，会把V3误报为FAIL；V3权威裁决是
`analysis.json`与`validator_report.json`中的`JOINT_MOVE_COEFFICIENT_MECHANISM_PASS`和
`JOINT_MOVE_COEFFICIENT_VALIDATION_PASS`。

边界：fold-0已经被多次用于机制审计，本节不是新鲜formal放行；系数尚未接入Actor更新，也未验证
真实cost收益。因此下一步只授权在新的episode split上复核相同固定0.5门；复核通过后再进入
OAC-2小步连续Actor更新，仍保留真实DBM accept/reject和two-center guard。

```text
scripts/model_verify/train_mppi_joint_value_gap_pilot.py
scripts/model_verify/validate_mppi_joint_value_gap_pilot.py
outputs/mppi_proposal/online_absolute_sac_joint_value_gap_20260820_v1/
outputs/mppi_proposal/online_absolute_sac_joint_value_signed_delta_20260820_v2/
outputs/mppi_proposal/online_absolute_sac_joint_move_coefficient_20260820_v3/{analysis.json,validator_report.json}
```

### 11.88 新episode outer fold复核：联合Value--移动系数跨split通过（2026-08-20）

按§11.87预注册边界，在完全不同的episode outer fold-1上复核同一机制。没有重新选择目标、loss
权重或操作阈值：仍使用`c_move=max(delta J,0)/(max(delta J,0)+0.1)`、辅助权重`0.05`和固定
`c_move<=0.5`物理门。fold-1的冻结Actor与初始Critic独立加载，重新产生三seed各15360条
Actor-visited Replay（共46080次DBM评价）；formal validation/test继续封存。

200步初始burn-in仍为0/3通过，说明短预算失败可复现。固定Replay续训到6400步后，主Value明显
恢复：actor-visited pair accuracy为`0.943/0.948/0.941`，2.8m/s为
`0.918/0.923/0.919`，heldout bank为`0.942/0.948/0.944`。从1600到6400步，最初排错动作的
纠正率分别再提高`+0.143/+0.203/+0.120`至`0.711/0.763/0.750`。这再次证明200步不足，而非
Replay没有可学习信息。

随后从6400步Value及其optimizer state继续相同联合训练3200步。独立validator复算结果：

| 指标 | seed 0 | seed 1 | seed 2 |
| --- | ---: | ---: | ---: |
| actor-visited pair accuracy | 0.951 | 0.953 | 0.948 |
| heldout bank pair accuracy | 0.946 | 0.951 | 0.945 |
| heldout top-1 recovery | 0.982 | 0.988 | 0.983 |
| 2.8m/s pair accuracy | 0.931 | 0.928 | 0.934 |
| 初始排错纠正率 | 0.784 | 0.751 | 0.823 |
| lag-2坏动作排序 | 0.966 | 0.966 | 0.973 |
| move coefficient相关性 | 0.833 | 0.852 | 0.829 |
| move coefficient MAE | 0.109 | 0.107 | 0.111 |
| bank-best recall（固定0.5） | 0.922 | 0.922 | 0.898 |
| warm false-stay（固定0.5） | 0.017 | 0.017 | 0.010 |

三seed最终全部通过，且400/800/1600/3200的12个post-training checkpoint全部通过固定门；
qualification为`JOINT_MOVE_COEFFICIENT_VALIDATION_PASS`。Replay hash、父contract、optimizer、
Actor零更新和split封存均由独立validator确认。cuDNN打印的plan fallback warning未导致失败或数值
偏差，所有保存指标均被逐项复算。

正式裁决：§11.87的联合Value--连续系数机制已跨episode split复现，不再只是fold-0机制观察；
OAC-2小步Actor更新pilot的入口条件成立。但这不等于Actor收益或闭环收益已经通过：本节Actor仍为
零更新，formal validation/test仍未使用。下一阶段只允许训练split内的小步、低频Actor更新，
`c_move`作为连续缩放而非硬开关，真实DBM cost决定selected checkpoint晋级，所有坏动作仍写入
Replay；部署two-center warm guard不变。

```text
outputs/mppi_proposal/online_absolute_sac_oac01_fold1_20260820_v1/
outputs/mppi_proposal/online_absolute_sac_oac1_fixed_replay_fold1_20260820_v1/
outputs/mppi_proposal/online_absolute_sac_joint_move_coefficient_fold1_20260820_v1/{analysis.json,validator_report.json}
```

### 11.89 OAC-2持续单步Actor--Critic：小步真实cost收益跨3 seed通过（2026-08-24）

在§11.88放行后执行首个Actor非冻结pilot。任务仍是固定状态的即时cost contextual bandit，不做
车辆状态递推、`next_state`或Bellman bootstrap。outer fold-1的1200个训练状态按完整episode再
拆为600个fit状态与600个internal-selection状态；每个`speed x scenario`各保留一个完整episode
用于DBM checkpoint选择。初始Critic虽然在OAC-1阶段见过这些状态，但OAC-2新增Replay与Actor更新
严格排除internal-selection episodes，这一限制已固化在contract。

每seed运行20轮；每轮256状态×6动作=`1536`次真实DBM交互，随后联合更新Twin Value与连续系数头
20次、Actor 1次，更新比固定`1:20`。Actor训练采用可重参数化tanh-Gaussian，部署输出仍是唯一
deterministic mean。连续系数用于缩放Critic cost项，独立temperature与action-space trust保留；
每5轮用internal-selection真实DBM cost比较latest/selected。坏动作全部写Replay，accept/reject
只控制selected晋级。三seed合计新增101160次DBM评价，其中92160次为交互、9000次为内部评估。

20轮结果：

| 指标 | seed 0 | seed 1 | seed 2 |
| --- | ---: | ---: | ---: |
| selected轮次 | 20 | 20 | 20 |
| 平均真实gain vs初始Actor | 1.764 | 1.321 | 0.832 |
| episode-bootstrap 95% CI | [1.344, 2.246] | [1.019, 1.617] | [0.634, 1.061] |
| median gain | 0.361 | 0.447 | 0.170 |
| 2.4m/s mean gain | 2.455 | 2.126 | 1.215 |
| 2.8m/s mean gain | 2.853 | 1.530 | 0.909 |
| 相对bank-best headroom recovery | 2.01% | 1.58% | 0.94% |
| actor-visited Critic pair accuracy | 0.936 | 0.930 | 0.935 |
| coefficient recall / false-stay | 0.900 / 0.012 | 0.895 / 0.018 | 0.878 / 0.015 |

四次评估的latest均满足mean/median、2.4/2.8m/s、饱和与有限值门并晋级，最终selected=latest。
20轮所有Actor单步都小于`0.02 sigma RMS`上限，未触发一次trust projection；最终单步约
`0.00022--0.00031 sigma RMS`，全维动作饱和率为0。Value与连续系数在Actor开始移动后仍保持原
物理门，支持“新动作→真实cost→Replay→Critic持续更新→Actor低频更新”的机制链。

必须同时报告直接Actor尾部：gain P05为`-0.700/-1.929/-2.321`，worst为
`-27.449/-12.583/-13.627`，逐状态回归比例为`24.2%/31.0%/37.3%`。因此本节证明的是**平均与
episode聚合层面的保守小步收益**，不是direct Actor逐状态安全。two-center guard下gain P05与
worst均为0（由`min(warm, actor)`构造），所以部署guard仍不可移除。

正式qualification为`OAC2_MECHANISM_PASS_READY_FOR_FULL_FOLD_REPLICATION`，独立validator为
`OAC2_CONTINUOUS_ACTOR_VALIDATION_PASS`：重算initial/latest/selected真实DBM指标误差为0，抽查
Replay cost误差为0，并确认3/3 seed、Replay隔离、角色计数、optimizer/temperature、Actor更新数
和Critic `20:1`合同。该结果只放行其余outer folds的同合同复核；尚未达到OAC-3完整fold门，也
没有闭环传导结论。formal validation/test继续封存。

```text
scripts/model_verify/train_mppi_oac2_continuous_actor.py
scripts/model_verify/validate_mppi_oac2_continuous_actor.py
outputs/mppi_proposal/online_absolute_sac_oac2_fold1_20260824_v1/{summary.json,validator_report.json}
```

### 11.90 OAC-3全3-fold outer-heldout复核通过，放行短闭环guard A/B（2026-08-24）

按§11.89的冻结合同补跑fold-0与fold-2。fold-0直接复用已验证的Value/系数父链；fold-2从冻结Actor
重新生成46080条Replay，续训Value到6400步并联合系数到3200步，父链三个独立validator全部通过
后才启动Actor。三个fold的OAC-2均为20轮×3 seed，所有9个selected Actor均在第20轮，且每个
fold内部机制门为3/3 seed通过。

为避免把internal-selection收益误写成跨episode泛化，随后将每个selected Actor放到其outer-heldout
600状态上做真实DBM复算。这些episodes没有进入相应OAC-2的新Replay、Actor更新或checkpoint选择。
结果为3/3 folds、9/9 seeds通过预注册OAC-3门：

| fold | 3 seed mean gain | CI下界范围 | 2.4m/s gain范围 | 2.8m/s gain范围 |
| ---: | --- | --- | --- | --- |
| 0 | 1.136 / 1.938 / 1.401 | 0.769--1.461 | 1.924--2.854 | 1.131--2.594 |
| 1 | 1.883 / 1.167 / 0.687 | 0.442--1.445 | 1.319--3.201 | 0.236--2.186 |
| 2 | 1.556 / 1.232 / 0.894 | 0.691--1.166 | 1.032--2.427 | 1.304--3.044 |

九个Actor的平均mean gain为`1.322`，平均median gain为`0.297`，平均bank-best headroom recovery
为`1.48%`。所有episode-bootstrap CI下界严格大于0，所有2.4/2.8m/s层聚合gain非负。因此
qualification为`OAC3_OUTER_HELDOUT_PASS_READY_FOR_SHORT_CLOSED_LOOP`，独立复算
`OAC3_OUTER_HELDOUT_VALIDATION_PASS`且最大指标误差为0。

尾部仍是部署边界：direct P05跨seed为`-3.524`到`-0.999`，worst为`-35.290`到`-6.057`，
25.7%--36.7%状态存在直接回归。two-center guard下P05与worst均为0，但这是构造保护而不是Actor
自身消除了尾部。正式裁决因此是：OAC-3跨fold固定状态传递通过，只放行OAC-4的
`warm-only`对`warm + selected Actor two-center`短闭环A/B；不得先移除guard，也不得把J_direct
收益直接等同于闭环累计收益。formal validation/test继续封存。

```text
scripts/model_verify/analyze_mppi_oac2_full_folds.py
scripts/model_verify/validate_mppi_oac2_full_folds.py
outputs/mppi_proposal/online_absolute_sac_oac2_fold{0,1,2}_20260824_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_full_3fold_20260824_v1/{analysis.json,validator_report.json}
```

### 11.91 OAC-2训练预算曲线：均值继续改善，但direct尾部随步数发散（2026-08-24）

按用户要求暂停闭环，先在fold-1的同一600状态internal-selection口径上检验“20轮是否只是训练
不足”。保持Actor LR=`1e-6`、每轮`20 Critic : 1 Actor`、探索、continuous coefficient与
`0.02 sigma RMS` trust设置不变，分别运行20/100/200轮×3 seed；100/200轮均独立DBM复算通过，
Replay误差与initial/latest/selected指标误差均为0，formal validation/test未加载。200轮为降低
报告成本每10轮评价一次而非每5轮，但评价不反传、不改变latest训练路径，三seed最终均
`selected=latest=round 200`。

同口径参考为warm `J=29.174`（median `23.356`）和24-bank best teacher `J=5.161`
（median `3.482`）。三seed平均预算曲线为：

| 轮数 | Actor mean/median J | gain vs初始 | teacher headroom recovery | gain P05 / worst | Critic pair | guard J / gain vs warm |
| ---: | --- | ---: | ---: | --- | ---: | --- |
| 20 | 90.491 / 20.041 | 1.306 | 1.51% | -1.650 / -17.887 | 0.934 | 19.372 / 9.802 |
| 100 | 85.921 / 18.550 | 5.875 | 6.79% | -5.937 / -64.520 | 0.937 | 18.920 / 10.254 |
| 200 | 81.844 / 17.861 | 9.953 | 11.51% | -9.573 / -136.893 | 0.940 | 18.690 / 10.484 |

结论分两层。第一，20轮远不是Actor参数上限：三个seed到200轮仍全部选择最后checkpoint，mean、
median、2.4/2.8m/s均继续改善，Critic actor-visited pair始终约0.93--0.94。因此在当前局部访问
分布上，Critic不是立即的第一阻塞，训练预算确实限制了早期收益。第二，**更多同目标训练不能
直接解释成逼近teacher**：200轮只恢复11.5% teacher headroom，Actor mean仍远差于warm；虽然
Actor median已经优于warm median，但约31%状态持续回归，且同一批回归的损失幅度随轮数扩大，
P05/worst单调恶化。

two-center进一步暴露边际收益结构：20到200轮direct mean gain增加`8.65`，但guard gain仅增加
`0.68`；200轮guard恢复warm到teacher差距的`43.66%`。这说明后续大部分平均改善发生在warm
已经能兜底或Actor仍不可用的状态，不能靠guard指标掩盖direct尾部。

正式裁决为`OAC2_BUDGET_IMPROVES_CENTER_BUT_DIRECT_TAIL_DIVERGES`：冻结“20轮代表网络上限”的
旧解读，但不授权继续提高Actor LR或Actor-only使用。下一实验应在保持在线Replay/Critic更新的
前提下，先给Actor目标与checkpoint晋级加入预注册的state-wise regression/tail约束，再比较同样
200轮；闭环按用户要求暂停，OAC-3历史资格不撤销但不执行OAC-4。

```text
scripts/model_verify/analyze_mppi_oac2_training_budget.py
outputs/mppi_proposal/online_absolute_sac_oac2_fold1_{100round,200round}_20260824_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_budget_20260824_v1/analysis.json
```

### 11.92 200轮Actor尾部逐状态审计：主体改善、固定position尾部被持续放大（2026-08-24）

为解释§11.91中`mean=81.844`与`median=17.861`的分离，重新加载20/100/200轮全部9个
selected Actor，在完全相同的fold-1 internal-selection 600状态上逐状态重放DBM，并保存动作、
cost、逐项cost及最终Twin Value端点判断。这里`17.861`是三个seed各自median的平均；把1800个
seed-state样本合并后的median为`17.842`，两者口径一致且差异仅来自聚合顺序。

首先必须修正“mean随训练变差”的直觉：Actor mean实际从初始`91.796`下降到20/100/200轮的
`90.491/85.921/81.844`，所以整体均值也在改善。绝对mean仍远差于warm `29.174`，主要因为
no-anchor初始Actor已有很重的高cost尾部，而不是200轮新制造了全部高cost：200轮最高1%样本
贡献总cost的15.3%，最高5%贡献45.2%，最高10%贡献63.7%；其中绝对cost最高的30个状态在平均
意义上反而从`790.2`降到`733.0`。

但“固定回归尾部被持续放大”确实成立：

- 单seed-state回归比例在20/100/200轮为`30.83%/31.44%/31.28%`，说明坏case数量没有持续增加；
- 按三seed平均后，154/600状态在200轮差于初始，137/600同时满足“差于初始且100→200继续
  变差”；60/600状态在三个seed中都回归；
- 每seed最差5%的100→200轮Jaccard为`0.765/0.579/0.818`，且分别有`22/17/14`个状态连续落在
  20、100、200轮三个尾集中，排除纯随机换尾；
- 三seed平均gain不超过`-10`的18个严重状态中，16个来自2.4或2.8m/s，其中2.8m/s占11个。
  它们的cost中位数沿初始→20→100→200轮为`56.89→58.19→67.11→79.37`。
- 负收益高度集中：1800个seed-state里最差1%承担33.5%的总负收益，最差5%承担68.0%，最差
  10%承担85.8%。因此P05/worst会迅速恶化，而median仍继续改善。

逐项DBM cost重放与总cost最大误差小于`1e-3`。全部563个回归seed-state中，position是459个的
最大正向恶化项；83个`gain<=-10`严重样本中更达到81/83。严重样本平均cost增量为position
`+21.12`、yaw `+5.19`、vx `+0.30`，rate项可忽略。Actor移动并不大：严重样本全16维RMS移动
中位仅`0.035sigma`，early steering knot 0--2绝对移动中位`0.032sigma`，但晚段steering仅
`0.014sigma`。这与此前“早段转向经长horizon放大position误差”的杠杆证据一致。

最终Critic端点审计进一步收窄归因。保守Twin Value对初始Actor与200轮Actor的改善符号在全部
1800样本上准确率为`90.3%`；对真实回归样本，只有`14.6%`仍被Critic误判为改善；严重回归中
该比例仅`9.6%`。所以尾部不能再概括为“最终Value Critic看不见坏动作”。更准确的判断是：
Actor的共享参数与batch均值目标允许牺牲少数状态，当前selected gate也只约束mean、median、
2.4/2.8m/s均值和饱和，不约束P05/CVaR；小幅但反复的early-steering偏移因高杠杆累计成尾部。
端点排序正确也不自动保证每次局部Actor梯度都回到安全侧，后者若需归因应另做逐轮路径审计。

裁决为`OAC2_200ROUND_MEDIAN_IMPROVES_BUT_PERSISTENT_POSITION_TAIL_WORSENS`。下一配对实验不提高
LR，保持真实DBM Replay与Critic持续更新，同时增加两层保护：(1) Actor目标加入基于保守Twin
Value、相对当前selected center的逐状态soft regression/CVaR惩罚；(2) checkpoint晋级加入
direct P05与高速度尾部不恶化门。部署two-center guard继续保留；formal validation/test未加载。

```text
scripts/model_verify/analyze_mppi_oac2_budget_tail_cases.py
outputs/mppi_proposal/online_absolute_sac_oac2_budget_tail_20260824_v1/{analysis.json,evaluation.npz}
```

### 11.93 OAC-2T尾部CVaR配对实验：尾部显著修复，但固定权重过度正则（2026-08-24）

在§11.92后保持200轮、Actor LR=`1e-6`、`20 Critic : 1 Actor`、探索、Replay、continuous
coefficient与trust完全不变，只给Actor加入相对当前selected center的保守Twin Value回归项：

```text
r(s) = relu(max(Q1,Q2)(s, actor_mean) - max(Q1,Q2)(s, selected_mean))
L_actor += lambda_tail * mean(top 10% r)
```

同时给selected gate加入固定尾部floor：全体gain P05不低于`-2.5`，2.4/2.8m/s分别不低于
`-4.0/-4.1`。先按数值量级固定`lambda_tail=100`，发现明显过强后只增加一个一数量级减弱的
`lambda_tail=10`对照，不继续扫参。两臂均为3 seed，全部selected=round 200，独立validator
重算DBM/Replay/指标误差为0，formal validation/test未加载。

三seed平均结果：

| Actor目标 | mean/median J | mean gain | gain P05 / worst | 回归比例 | headroom recovery | guard J | Critic pair |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 原200轮 | 81.844 / 17.861 | 9.953 | -9.573 / -136.893 | 31.28% | 11.51% | 18.690 | 0.940 |
| CVaR `lambda=10` | 90.407 / 20.158 | 1.390 | -0.498 / -9.814 | 19.11% | 1.61% | 19.356 | 0.940 |
| CVaR `lambda=100` | 90.531 / 20.247 | 1.266 | -0.494 / -11.444 | 20.00% | 1.46% | 19.363 | 0.941 |

机制结论是明确的。尾部项有效：`lambda=10`把P05提高`9.08`、平均worst提高`127.1`，回归比例
下降12.17个百分点。但它只保留原mean gain的`14.0%`，guard cost也变差`0.67`；`lambda=100`
没有得到更好Pareto。两臂全部60次周期评估均通过尾部floor，说明mean损失不是checkpoint门卡住，
而是loss本身。Critic pair三臂均约0.94，排除Critic训练退化。

`lambda=10/100`结果近似还说明当前zero-margin top-k正回归项表现得像硬局部约束：即使是Critic
噪声尺度内的微小正gap也被最差集合持续放大。正式qualification为
`OAC2_TAIL_CVAR_EFFECTIVE_BUT_OVERREGULARIZED_NOT_SELECTED`，两臂Actor均不向后授权。尾部约束
方向保留，但下一版本应改成非零material margin加自适应Lagrange/预算约束：只要求CVaR不超过
预注册预算，由dual变量自动调节，而不是用固定大权重把所有预测回归压到0。原two-center guard
继续保留。

```text
scripts/model_verify/analyze_mppi_oac2_tail_cvar_ab.py
outputs/mppi_proposal/online_absolute_sac_oac2_tailcvar{10_,}_200round_20260824_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_tailcvar_ab_20260824_v1/analysis.json
```

### 11.94 固定K多候选Actor：多basin保留有小幅收益，但不能解释当前主要差距（2026-08-24）

为回答“多模态是否需要Diffusion”这一问题，先执行更便宜、可判别的固定`K=4`对照。数据仅使用
1800个train状态的J16 v2多起点结果；每状态从best的0.5% cost带内保留最多4个动作空间分散的
elite，不足4个时以best补齐。`K=1/K=4`共享严格clean/no-anchor `G-X` encoder、knot-token
Transformer、输出支撑、优化器、3-fold episode split和3 seed；K=4只增加4个最终输出head及
permutation-invariant matching loss，不对多个basin求均值。formal validation/test未加载。

同一数据的warm/J16均值为`27.918/4.830`。逐seed三fold平均OOF结果为：

| seed | K=1 recovery | K=4 DBM best-of-K recovery | K=4 Critic-selected recovery |
| ---: | ---: | ---: | ---: |
| 0 | -2.903 | -2.824 | -2.883 |
| 1 | -3.120 | -2.978 | -3.029 |
| 2 | -3.189 | -2.953 | -3.014 |

K=4候选间标准化距离中位约`0.144 sigma`、P95约`0.364 sigma`，说明head并未完全坍缩；DBM
oracle best-of-4相对K=1的median-seed recovery改善`+0.168`，但两者都远小于0。K=4的OOF
均值J约`93.0--96.5`、中位约`24.3--25.0`，依然远差于warm；absolute-value Critic选择只比
oracle再损失约`0.05--0.06` recovery，选择器不是主要损失源。全部candidate cost与Critic选择
经独立DBM重放，最大误差为0。

正式qualification为
`MULTI_CANDIDATE_ACTOR_SMALL_RELATIVE_GAIN_BUT_STILL_UNUSABLE`。该实验说明固定多头能够保留一点
多basin价值，但**多模态不是当前数量级差距的主解**；现在没有理由直接升级Diffusion。若未来
在更强单头/真实task-loss基线上仍出现明确mode coverage上限，再比较K增长曲线与Diffusion；当前
优先级转到同一Actor在真实DBM目标下的可达能力。

```text
scripts/model_verify/run_mppi_multi_candidate_actor.py
scripts/model_verify/validate_mppi_multi_candidate_actor.py
outputs/mppi_proposal/j16_multi_candidate_actor_20260824_v1/{summary.json,validator_report.json,oof_predictions.npz}
```

### 11.95 真实DBM task-loss Actor容量诊断：可强力改善，但仍未逼近J16（2026-08-24）

在§11.94配对K=1 `G-X` checkpoint上直接使用可微DBM与原始确定性J50训练Actor：

```text
state -> G-X Actor -> absolute 8x2 knots -> interpolation -> DBM rollout -> J50 -> autograd
```

训练loss是batch mean `J50`除以固定正标量，只改变数值尺度，不改变最优点；没有Critic、TD、
teacher MSE或FD标签参与更新，J16只用于训练后的距离评价。预注册的32/128状态过拟合门均通过，
因此继续执行fold-0完整1200状态train split；相应600个episode-heldout状态没有参与更新或选模。

| 范围 | 初始mean J | task-loss mean J | J16 mean | headroom recovery | 回归比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny-32 | 93.038 | 19.679 | 4.289 | 82.7% | 0.0% |
| tiny-128 | 111.175 | 15.569 | 4.668 | 89.8% | 0.8% |
| full train-1200 | 92.591 | 19.560 | 5.013 | 83.4% | 2.8% |
| episode-heldout-600 | 91.049 | 23.609 | 4.463 | 77.9% | 8.5% |

heldout中位J为`7.108`，明显好于初始`24.200`，说明真实task-loss学到的不是纯训练状态记忆；但
heldout gain P05仍为`-0.351`、worst为`-792.94`，且mean J仍比J16高`19.15`。所以本轮同时
建立两个结论：

1. 当前G-X Actor及DBM反传链并非“网络没有能力/optimizer失效”；绕过Critic和BC标签后能够取得
   大幅、可跨episode传递的改善。
2. 这仍不是理论最优容量证明。即使训练直接看到真实J50，模型也只到train/heldout
   `19.56/23.61`，没有逼近`5.01/4.46`；剩余阻塞位于Actor输出支撑、共享参数条件化、优化目标
   的tail分配或这些因素的组合，而不是单纯增加OAC步数。

输出支撑审计给出一个具体候选：J16有`25.3%`动作分量超出当前`out_center +/- 1 std`盒，
`91.3%`状态至少一维超界。逐维投影J16后的mean J为`90.52`，但这个数**不是盒内最优上界**：
跨basin逐维投影会落入坏区域，只能证明当前Actor不能精确表示大量J16标签，不能证明盒内最低cost
就是90.52。下一结构实验应先对照扩大/取消该affine-tanh支撑，并由DBM task-loss直接评估；若
训练和heldout继续改善，再研究state-dependent scale或多候选。Diffusion在这一门通过前继续
降级。

本实验只能用于DBM训练侧诊断/预训练，不能作为Query部署训练合同，因为它直接反传DBM内部模型。
可部署路线仍需one-shot Actor；Query适配仍要靠搜索标签或持续真实反馈。独立validator重放
tiny/full动作，所有分布指标误差不超过`1.36e-6`。qualification为
`DBM_TASK_LOSS_ACTOR_FULL_TRAIN_CAPACITY_CONFIRMED`，其含义仅是“真实目标训练能力成立”，不是
“逼近J16”或“尾部安全”。

```text
scripts/model_verify/run_mppi_dbm_task_loss_actor.py
scripts/model_verify/validate_mppi_dbm_task_loss_actor.py
outputs/mppi_proposal/dbm_task_loss_actor_20260824_v1/{summary.json,validator_report.json}
```

### 11.96 DBM task-loss输出支撑严格A/B：`+/-1std`是主要瓶颈，`+/-3std`已足够（2026-08-25）

按§11.95的结构前置门，在fold-0 train-only分层128状态上固定同一K=1 G-X checkpoint、同一批
状态、相同DBM J50 loss、optimizer、batch抽样和1200次更新，只改变absolute-action输出映射：

| arm | 输出支撑 | mean J | median J | recovery | 回归比例 |
| --- | --- | ---: | ---: | ---: | ---: |
| `box1` | `center +/- 1std` | 15.568 | 5.311 | 0.8977 | 0.78% |
| `box3` | `center +/- 3std` | **5.483** | 4.158 | **0.9924** | 1.56% |
| `full` | physical `[-1,1]` | 5.498 | **3.975** | 0.9922 | 0.78% |
| J16（评价参照） | 16-D best-found | 4.668 | 3.490 | 1.0000 | -- |

三臂通过zero-initialized pre-squash adapter保证初始action逐元素配对，最大误差仅
`1.19e-7`；因此改善不是更好的初始化造成。`box3`相对`box1`将mean J降低`10.085`、headroom
recovery提高`9.47pp`，并关闭了约`92.5%`的`box1 -> J16`剩余mean-cost gap。预注册门
（mean改善不低于2、recovery改善不低于0.05、回归比例不高于box1+2pp）通过，qualification为
`EXPANDED_ACTION_SUPPORT_MATERIAL_TINY128_GAIN`。

边界占用进一步给出机制证据：`box1`最终有`77.3%`状态至少一个分量达到支撑的95%，steering
分量占`33.2%`；`box3`没有分量达到其扩大支撑的95%。`full`与`box3`的mean差仅`0.015`，没有
可见额外收益。这把§11.95的“输出支撑候选”升级为**tiny-128下已确认的主要结构瓶颈**，并说明
当前优先采用`+/-3std`即可；直接开放全physical域不是必要条件，也不应被解释成网络表达能力已经
完全解决。

独立validator将三臂保存的最终动作重新送入DBM，所有cost分布指标最大绝对误差为`0`，初始动作
合同复算一致；formal validation/test均未加载。该结果只授权下一步在完整fold-0 train-1200及其
episode-heldout-600上做`box1`--`box3`配对复核，尚不能声称跨episode尾部或Query部署收益。
若完整复核仍成立，再把`box3`映射带回持续在线OAC Actor；Critic、Replay和two-center guard合同
保持不变。

```text
scripts/model_verify/run_mppi_dbm_task_loss_support_ab.py
scripts/model_verify/validate_mppi_dbm_task_loss_support_ab.py
outputs/mppi_proposal/dbm_task_loss_support_ab_20260825_v1/{summary.json,validator_report.json,support_*.pt}
```

### 11.97 输出支撑完整fold复核：train与episode-heldout均通过，放行OAC集成（2026-08-25）

按§11.96授权，将同一实验扩到fold-0完整train-1200，并在未参与更新与选模的episode-heldout-600
上复核。两臂均从同一K=1 G-X checkpoint和逐元素一致的初始action出发，使用相同2400次DBM
J50更新、batch序列、optimizer及每50步train-mean选模；只改变输出支撑：

| split / arm | mean J | median J | P95 J | worst J | recovery | 回归比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train `box1` | 19.792 | 6.765 | 82.500 | 498.512 | 0.8312 | 3.50% |
| train `box3` | **7.282** | **5.242** | **21.379** | **49.322** | **0.9741** | 5.33% |
| train J16 | 5.013 | 3.478 | 15.848 | 34.673 | 1.0000 | -- |
| heldout `box1` | 23.580 | 7.219 | 91.112 | 817.543 | 0.7792 | 9.17% |
| heldout `box3` | **10.300** | **5.702** | **33.064** | **455.341** | **0.9326** | 10.33% |
| heldout J16 | 4.463 | 3.055 | 11.962 | 39.829 | 1.0000 | -- |

`box3`相对`box1`在train/heldout分别降低mean J `12.510/13.280`，recovery提高
`14.28/15.34pp`；它关闭了`box1 -> J16`剩余mean-cost gap的约`84.6%/69.5%`。heldout gain
P05从`-0.622`微升至`-0.588`，worst regression从`-781.42`收窄至`-419.22`，说明均值改善不是
通过牺牲尾部获得；回归比例增加`1.17pp`，仍低于预注册`+2pp`边界。train gain P05降低
`0.199`但也在`-0.5`边界内。

机制证据同tiny复现：`box1`在train/heldout分别有`60.2%/62.5%`状态至少一维达到95%支撑边界，
`box3`两边均为0。初始action最大配对误差`1.19e-7`。独立validator重新DBM rollout所有initial/
final action，完整metrics与comparison最大误差均为`0`，并复算联合门一致；formal validation/test
保持封存。

正式qualification为
`BOX3_SUPPORT_FULL_TRAIN_AND_HELDOUT_PASS_READY_FOR_OAC_INTEGRATION`。这确认`+/-1std`不是小样本
偶然限制，而是当前G-X Actor的主要结构瓶颈；默认输出支撑应改为`center +/- 3std`。但heldout
mean仍高于J16 `5.84`且存在10.33% direct回归，所以本节只放行把同一映射带回持续在线OAC的
固定状态pilot；不得据此启动OAC-4闭环或声称Query部署收益。OAC集成必须保持现有Critic、Replay、
continuous coefficient、tail gate与two-center guard合同，并从旧selected Actor严格配对。

```text
scripts/model_verify/run_mppi_dbm_task_loss_support_full_ab.py
scripts/model_verify/validate_mppi_dbm_task_loss_support_full_ab.py
outputs/mppi_proposal/dbm_task_loss_support_full_ab_20260825_v1/{summary.json,validator_report.json,support_*.pt}
```

### 11.98 OAC support-only配对：`box3`提高收益，但不延长tail-safe训练时域（2026-08-25）

按§11.97放行边界，将`box3`带回持续在线OAC并完成fold-1、200轮、3 seed严格配对。两臂均通过
zero-initialized pre-squash adapter从同一父G-X Actor初始化，除输出支撑`1std/3std`外，Replay、
DBM交互、`20 Critic : 1 Actor`、continuous coefficient、LR、exploration、`0.02sigma`逐轮trust
projection及随机seed全部一致。初始selection action最大差`4.17e-7`。旧固定CVaR loss明确关闭：
`tail_regression_weight=0`；但checkpoint继续使用此前预注册的overall/2.4/2.8m/s gain-P05 floor
`-2.5/-4.0/-4.1`。因此本实验测的是“更宽支撑在相同tail安全门下能保留多少收益”。

| 口径（三seed平均） | `box1` | `box3` | 差值 |
| --- | ---: | ---: | ---: |
| tail-safe selected mean gain | 0.995 | **1.514** | **+0.518（+52.1%）** |
| tail-safe selected median gain | 0.241 | **0.326** | +0.086 |
| tail-safe selected gain P05 | **-1.060** | -1.407 | -0.347 |
| tail-safe selected worst gain | **-14.72** | -20.44 | -5.71 |
| tail-safe selected regression fraction | 30.72% | **30.28%** | -0.44pp |
| tail-safe selected guard mean J | 19.419 | **19.387** | -0.032 |
| 200轮latest mean gain | 10.171 | **15.543** | **+5.372（+52.8%）** |
| 200轮latest median gain | 1.759 | **2.472** | +0.713 |
| 200轮latest gain P05 | **-10.006** | -10.572 | -0.566 |
| 200轮latest worst gain | **-133.76** | -198.24 | -64.48 |

三seed selected round在两臂完全相同，均为`20/10/10`。所以`box3`使同一安全操作点的收益在3/3
seed提高，也使不受tail选模约束的live Actor在3/3 seed明显走得更远，但**没有延长tail-safe更新
时域**。selected的2.4/2.8m/s gain-P05平均分别从`-1.631/-2.262`变为
`-2.254/-3.293`，仍过现有floor但余量更小；latest的高速P05也更差。固定tail gate的逐轮失败
计数表明30轮以后主要由overall及2.4/2.8m/s P05阻断，而非mean、Critic或物理饱和门。

机制上，`box1` selected约`77%--82%`状态至少一维达到其95%支撑边界，`box3` selected/latest
在全部seed均为0；Actor-visited material-pair accuracy保持`0.940`左右，两臂差仅`-0.0018`。
因此支撑瓶颈已经被实质解除，且没有损伤Critic；剩余tail并非“输出仍走不到”，而是某些状态
在更大移动下回归更重。

正式qualification为
`BOX3_OAC_IMPROVES_GAIN_WITH_FIXED_TAIL_GATES_BUT_NO_SAFE_HORIZON_EXTENSION`。后续默认Actor输出
合同采用`center +/- 3std`，不再回到`1std`，也不继续扩大到full physical域。下一实验转到
state-wise tail控制：保留坏动作进Replay和当前floor，比较nonzero material margin加自适应
Lagrange/CVaR预算或等价的连续风险系数；不得恢复已证过度保守的固定`lambda=10/100`。本节仍不
放行OAC-4闭环，formal validation/test继续封存。

```text
scripts/model_verify/mppi_a2_actors.py                         # DirectNoAnchorGTXSupportActor
scripts/model_verify/train_mppi_oac2_continuous_actor.py      # optional support multiplier
scripts/model_verify/validate_mppi_oac2_continuous_actor.py
scripts/model_verify/analyze_mppi_oac2_support_ab.py
outputs/mppi_proposal/online_absolute_sac_oac2_support_box1_200round_20260825_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_support_box3_200round_20260825_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_support_ab_20260825_v1/analysis.json
```

### 11.99 `+/-3std`默认输出强合同与当前J口径（2026-08-25）

`+/-3std`现已从实验参数升级为所有新OAC G-X训练的默认强合同：训练器默认
`actor_output_support_multiplier=3.0`；`+/-1std`只允许作为显式命名的历史消融，并且必须同时给出
`--allow-box1-ablation`，否则训练在创建产物前失败。新run的`contract.json`必须记录默认值、实际值、
消融授权与checkpoint-buffer要求；latest/selected checkpoint必须携带匹配的
`output_support_multiplier`，独立validator复核不一致时失败。`None`只保留给旧checkpoint兼容读取，
不能作为新训练默认；full physical域也不是默认选项，除非另行预注册。该合同不改变原有P05/高速
tail floor、two-center guard、坏动作Replay或formal validation/test封存。

为避免把不同口径的J混在一起，当前fold-1 OAC internal-selection 600状态的三seed均值统一如下；
J越低越好：

| 对象 | direct mean J | direct median J | two-center guard mean J | 解释 |
|---|---:|---:|---:|---|
| 初始Actor | 91.796 | 20.750 | 19.542 | direct均值被少数极坏尾部拉高；guard逐状态在warm和Actor间选低J |
| tail-safe selected box3 Actor | 90.283 | 20.224 | **19.387** | 当前可晋级checkpoint；相对初始direct mean gain为1.514 |
| 200轮latest box3 Actor | 76.254 | **17.092** | **18.538** | 主体更好，但P05 gain=-10.572、worst gain=-198.239，未通过tail晋级 |

同一600状态上的warm mean J为`29.174`，candidate-bank best参考mean J为`5.161`。因此当前最诚实的
“可用网络J”是selected Actor加two-center guard的`19.387`；`18.538`是训练末态的诊断值，不是已
授权部署值；`5.161`是搜索/bank参考，不是网络已达到的结果。另一个fold-0 DBM可微task-loss容量
诊断得到box3 heldout J=`10.300`、J16=`4.463`，它只证明网络结构与3std支撑还有明显容量，不得与
fold-1 OAC部署结果当成同一测试集直接排名。当前主要差距是state-wise tail和Actor优化，而不是
动作支撑仍不够。

### 11.100 OAC-2A自适应tail预算实验合同（2026-08-25）

按§11.98/11.99的顺序，下一步固定`+/-3std`与其余OAC合同，只将已失败的zero-margin固定
`lambda=10/100`替换为非零margin加自适应Lagrange预算。具体定义、数值、smoke/full顺序及三层
gate见`mppi_online_actor_critic_pilot_plan_20260820.md` §24。核心数值为：margin-log=`0.05`、
top-tail fraction=`0.10`、budget-log=`0.02`、dual LR=`1.0`、EMA decay=`0.90`、lambda范围
`[0,10]`。本实验不改变Critic、Replay、Actor LR、trust、输出支撑或checkpoint P05 floor；不能将
dual机制通过单独解释成闭环授权。

### 11.101 OAC-2A结果：tail Pareto明显改善，但未通过收益保持与安全时域门（2026-08-25）

按§11.100完成20轮×3 seed smoke和完整200轮×3 seed。smoke中dual递推、3std checkpoint buffer、
Replay、DBM cost及selected/latest指标均由独立validator精确复算；前20轮最差10%超额EMA仅
`0.0007--0.0036`，低于`0.02`预算，所以lambda保持0，没有像固定CVaR一样从训练初期压制收益。

完整轮中lambda分别在round `47/28/30`首次激活，最大值`1.163/1.406/0.941`，最终值
`1.163/1.279/0.494`；全部远离上限10且递推误差不超过`1e-7`，机制门通过。相对无tail的box3：

| 200轮latest三seed平均 | box3 | adaptive tail | 变化 |
|---|---:|---:|---:|
| mean gain | **15.543** | 10.749 | retention 69.16% |
| median gain | **2.472** | 1.622 | -0.850 |
| gain P05 | -10.572 | **-2.126** | +8.446 |
| worst gain | -198.239 | **-111.848** | +86.391 |
| regression fraction | 30.39% | **21.33%** | -9.06pp |
| 2.4m/s gain P05 | -18.165 | **-3.335** | +14.830 |
| 2.8m/s gain P05 | -25.520 | **-6.861** | +18.659 |
| two-center guard mean J | **18.538** | 18.868 | +0.330（变差） |

tail-safe selected round从box3的`20/10/10`变为`20/80/10`，只有seed1延长。selected mean gain
从`1.514`提高到`2.758`，selected guard J从`19.387`降到`19.296`；但selected P05/worst没有
同步改善。预注册三层gate因此为：dual机制通过；P05/worst改善但mean retention `69.16%`略低于
70%而Pareto门严格失败；安全时域仅1/3 seed延长而失败。正式qualification为
`ADAPTIVE_TAIL_PARETO_IMPROVES_BUT_MISSES_RETENTION_AND_SAFE_HORIZON_GATES`，不放行OAC-4。

本轮证明自适应margin/budget显著优于固定`lambda=10/100`，但单一全局lambda仍不能稳定分配不同
状态与速度的风险。下一步不做第二次盲目长训；先用本轮逐状态/逐轮数据检查dual预测超额对真实
回归的recall、false-positive和速度×状态分层，判断应改成state-conditioned风险系数还是仅调整
预算。3std、checkpoint floor、坏动作Replay与two-center guard保持。

```text
scripts/model_verify/analyze_mppi_oac2_adaptive_tail.py
outputs/mppi_proposal/online_absolute_sac_oac2_box3_adaptive_tail_smoke_20260825_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_box3_adaptive_tail_200round_20260825_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_adaptive_tail_ab_20260825_v1/analysis.json
```

### 11.102 OAC-2A未改善状态的逐状态归因（2026-08-25）

按§11.101的预注册路由，对adaptive-tail完整200轮run的internal-selection 600状态做零新增采样诊断。
对3个seed分别加载initial/selected/latest Actor与最终Twin Value，重新执行确定性DBM rollout并拆分
position/yaw/vx/rate cost；formal validation/test仍未加载。源run validator通过，分项重组总cost最大
误差不超过`1e-3`。

首先，“差状态完全没有被修复”并不准确。以每个seed-state相对initial Actor是否回归划分：selected
时有`495`个回归，继续训练到latest后其中`193`个恢复、`302`个仍回归；同时另有`82`个原本不回归
的样本被后续更新带坏。因此latest回归总数降到`384/1800=21.33%`，但这是“修复193、引入82”后
的净结果，不是单调修复。

| selected→latest类别 | seed-state数 | initial J中位 | latest-initial J中位 | tail信号激活率 |
|---|---:|---:|---:|---:|
| 持续回归 | 302 | 9.159 | +0.677 | 30.5% |
| 新增回归 | 82 | 8.406 | +0.275 | 36.6% |
| 被latest修复 | 193 | 21.531 | -1.214 | 1.0% |
| selected/latest均改善 | 1223 | 28.085 | -3.942 | 1.1% |

该表给出首个直接原因：持续/新增回归主要是initial cost已较低、可用headroom小的状态；共享Actor在
优化高cost状态时，对这些近平坦/近优状态仍产生小移动。相比之下，initial cost较高的样本更容易
获得稳定下降。因此问题不是Actor只会处理简单低cost状态，恰好相反：它能降低有明显改善空间的
状态，却不能可靠识别“已经够好、应少动或不动”的状态。

最终Twin Value并未整体失效。其对selected→latest真实`log1p(J)`变化的Pearson为`0.782`；区分
任意回归与material回归的ROC-AUC分别为`0.950/0.941`。但当前固定`0.05` margin的激活门只召回
`32.6%`任意回归和`69.3%` material回归；对17个严重回归seed-state只召回`64.7%`。激活样本对
任意回归的precision为`93.4%`，说明信号较干净但偏保守。更重要的是，现有约束控制的是全batch
top-10%超额的**均值预算**，不是逐状态non-regression约束；即使某状态被识别，全局lambda仍可用
少数大收益交换它的损失。因此本轮不能归因为“Critic又学反了”，而是“Value排序可用，但固定门槛
+全局聚合无法完整覆盖和约束每个状态”。

真实cost分项进一步定位了灾难来源：384个回归seed-state中`333`个由position项主导；17个严重
回归中`16`个由position主导，严重组position/yaw/total增量中位分别为`+15.54/+3.72/+20.38`。
全部严重状态只出现在`2.4/2.8m/s`（13个去重物理状态中分别5/8个），低速虽也约有20%小回归，
但没有`gain<=-10`的灾难。严重组early-steering移动量中位仅`0.0269 sigma`，仍会经高速、早段
steering高杠杆和长horizon位置误差放大；这不是3std支撑触界问题。

跨seed也支持“局部敏感+共享更新耦合”而非一个简单可分的坏工况：600状态中，latest回归只在一个
seed出现的有163个、两个seed出现70个、三个seed均出现仅27个；严重回归三个seed均出现的只有1个。
最坏的稳定反例是`episode_086/step_000322.npz`：三seed均恶化，seed均值gain约`-111.0`，position
增量均值`+84.5`，但最终Critic的margin门只在1/3 seed激活。这同时暴露了局部Value误差和全局
约束均无逐状态保证。

正式判定为
`ADAPTIVE_TAIL_PARTIAL_REPAIR_GLOBAL_AGGREGATE_MISSES_LOW_HEADROOM_AND_HIGH_SPEED_LEVERAGE_TAIL`。
后续不应继续盲调单一lambda或延长相同训练。若继续OAC，优先验证连续的state-wise move coefficient/
trust缩放（低headroom收缩、高速early steering更严）与逐状态conservative Value advantage；训练
安全依旧不删除坏动作，部署仍由two-center warm floor兜底。

```text
scripts/model_verify/analyze_mppi_oac2_adaptive_tail_states.py
outputs/mppi_proposal/online_absolute_sac_oac2_adaptive_tail_states_20260825_v2/analysis.json
outputs/mppi_proposal/online_absolute_sac_oac2_adaptive_tail_states_20260825_v2/evaluation.npz
```

### 11.103 OAC-2B连续state-wise风险回拉：selected收益提高，但高速tail未修复（2026-08-25）

按§11.102/online计划§26实现单变量pilot：保留3std、adaptive-tail、Twin Value、Replay、
`20 Critic : 1 Actor`、探索、全局`0.02 sigma RMS`参数更新投影及checkpoint floor不变，只新增
逐状态连续风险系数：

```text
r_i = sigmoid((Q_twin(s_i, actor_i) - Q_twin(s_i, selected_i) - 0.05) / 0.02)
L_state = 0.1 * mean(stopgrad(r_i) * RMS16((actor_i-selected_i)/sigma) / 0.02)
```

它不是stay/move二分类，也不在部署时调用Critic；训练损失把高风险状态的Actor输出连续拉回selected
center，最终效果蒸馏进一次前向Actor。第一次`squared_raw`量纲smoke虽3/3合同通过，但未加权损失
仅`1e-7--1e-6`，判定为量纲检查失败，不进入完整实验；上式按已注册trust半径归一化后，20轮×3
seed仍3/3通过且早期收益未被压死，才运行完整200轮。

独立validator对完整run的Replay、DBM cost、selected/latest、优化器、3std buffer、dual递推与新增
state-wise合同全部通过。相对§11.101 adaptive-tail基线：

| latest三seed平均 | adaptive-tail | +state-wise回拉 | 变化 |
|---|---:|---:|---:|
| mean gain | 10.749 | **11.319** | +5.3% |
| median gain | 1.622 | **1.956** | +0.334 |
| gain P05 | **-2.126** | -3.088 | -0.962 |
| worst gain | **-111.848** | -114.599 | -2.751 |
| regression fraction | **21.33%** | 21.94% | +0.61pp |
| 2.4m/s gain P05 | **-3.335** | -6.300 | -2.966 |
| 2.8m/s gain P05 | **-6.861** | -8.947 | -2.086 |
| two-center guard mean J | 18.868 | **18.796** | -0.073 |

selected侧存在真实正面价值：round从`20/80/10`变为`20/80/60`，mean gain从`2.758`升到
`4.019`，median gain从`0.641`升到`0.837`，回归比例从`27.50%`降到`24.44%`，guard mean J
从`19.296`降到`19.212`。但selected P05/worst也从`-1.477/-26.276`变差到
`-1.718/-32.224`，所以不能只按selected mean或安全轮次宣称通过。

逐状态重放解释了失败：最终Twin Value相关性/AUC反而改善到`0.809/0.949`，material recall从
`69.3%`升到`73.8%`；但严重回归从17增至29个seed-state，持续回归从302增至348。新增回归从82
降到47，说明回拉确实减少了后期新带坏的状态，却没有把已有的高杠杆persistent tail拉回来。
原因是该项仍通过共享网络参数间接生效，`r_i`只描述当前mean相对selected的Value风险；它既不是
每个状态独立的动作投影，也不能保证一次参数更新后其他状态的输出不变。高速early-steering的微小
偏移仍能越过局部安全边界。

正式判定为`STATEWISE_RISK_SHRINK_FAIL_NO_FURTHER_WEIGHT_SWEEP`。保留该run的selected checkpoint
作为内部候选（guard J最好），但不晋级OAC-4/formal validation/闭环；不继续扫weight/temperature。
若再次处理tail，需要改变约束作用点，例如用确定性DBM/候选bank构造逐状态安全投影教师再蒸馏，
而不是继续依靠共享Actor loss权重。

```text
scripts/model_verify/train_mppi_oac2_continuous_actor.py
scripts/model_verify/validate_mppi_oac2_continuous_actor.py
scripts/model_verify/analyze_mppi_oac2_statewise_risk_ab.py
outputs/mppi_proposal/online_absolute_sac_oac2_box3_statewise_risk_200round_20260825_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_statewise_risk_ab_20260825_v1/analysis.json
outputs/mppi_proposal/online_absolute_sac_oac2_statewise_risk_states_20260825_v1/analysis.json
```

### 11.104 8-knot前密后疏布局：容量略有上升，但当前搜索/锚点合同不支持切换（2026-08-25）

针对“early action杠杆更大，是否应把8个knot前移加密”，补做两个train-only、固定DBM、原始J50
实验。比较布局为：

```text
uniform:            [0, 7, 14, 21, 28, 35, 42, 49]
front_dense_mild:   [0, 3,  7, 12, 18, 26, 35, 49]
front_dense_strong: [0, 2,  5,  9, 14, 21, 31, 49]
```

第一项不再沿用§11.52的teacher投影，而是在每个布局自己的16D参数空间里真实重搜；100个分层状态、
每布局每状态固定128个antithetic Hadamard多轮候选：

| matched-budget search | mean J | median J | 对uniform胜/负 | common baseline违规 |
|---|---:|---:|---:|---:|
| uniform | **7.641** | **4.862** | — | 0 |
| mild前密 | 8.528 | 6.226 | 21/79 | 53 |
| strong前密 | 9.434 | 6.731 | 12/88 | 58 |

前密布局在当前Actor/warm的uniform计划重采样时会丢失中后段形状；mild/strong投影anchor excess中位
为`+1.541/+4.194`，128次局部搜索仍不足以恢复，故当前可部署搜索口径明确失败。

第二项用同一100状态做5起点×150步可微DBM容量oracle，以区分“搜索难”与“表达上限”：

| capacity oracle | mean J | median J | paired mean vs uniform | 胜/负 |
|---|---:|---:|---:|---:|
| uniform | 4.7929 | 3.2393 | — | — |
| mild前密 | **4.7284** | **3.2338** | -0.0645 | 80/20 |
| strong前密 | 4.8016 | 3.2830 | +0.0087 | 62/38 |

因此前密并非理论上错误：严格配对随机起点后，mild布局约有`1.35%` mean容量收益；但这个收益远小于当前局部搜索的
`+0.886 J`损失，并且strong前密已无容量收益。其物理含义是early分辨率有价值，但8个knot总数
固定时也会牺牲mid/late联合计划，而当前J50的position质量主要在后段累积。

独立validator还发现并修复了首版搜索产物“rollout时裁剪、labels保存未裁剪knots”的口径错误；
v3重新生成后六个layout×artifact的DBM cost回放最大误差均为0，以上数值来自已验证v3。

正式路由：保留uniform runtime默认，不修改当前MPPI/Actor/ONNX输入输出合同。只有未来能同时提供
mild布局原生warm-start、原生search covariance及对应Actor标签，并在matched-budget下至少达到
uniform的P05/worst与零baseline回归，才重新讨论切换；不能只凭容量oracle的1%收益改部署布局。

```text
scripts/model_verify/run_mppi_knot_layout_search_ab.py
scripts/model_verify/run_mppi_knot_layout_oracle_ab.py
scripts/model_verify/validate_mppi_knot_layout_ab.py
outputs/mppi_proposal/knot_layout_search_ab_20260825_v3/summary.json
outputs/mppi_proposal/knot_layout_capacity_oracle_20260825_v2/summary.json
outputs/mppi_proposal/knot_layout_ab_validation_20260825_v2/validator_report.json
```

### 11.105 Actor冻结的Critic容量与Pairwise Delta A/B：两者均有收益，局部差值头更有效（2026-08-25）

为区分“当前Critic容量不足”与“absolute value目标不适合局部Actor比较”，在不构造Actor optimizer、
Actor更新数严格为0的条件下完成三臂严格配对。数据只来自fold-1外层训练侧：每seed复用§11.101的
固定长期Actor-visited Replay（约31.5万行）与相同24候选bank；三臂逐update采样序列SHA256完全
一致，均训练6400步。评估使用从未进入Replay/训练/选模的外层episode-heldout 600状态，重新生成
initial/selected/latest Actor候选并用固定DBM计算真实J。formal validation/test仍未读取。

三臂为：

| arm | 参数量 | 结构/监督 |
|---|---:|---|
| base | 467,585 | 当前Absolute Value Critic，value + ranking |
| wide | 905,729（1.937×） | 近2×参数，loss/data完全不变 |
| pair-delta | 544,706 | base + 反对称同状态`Delta log1p(J)`头及直接delta/ranking监督 |

Pairwise头显式计算`0.5[f(a,b)-f(b,a)]`，因此`Delta(a,a)=0`且交换输入严格反号；它不是梯度头，
只回答同一状态下候选A相对候选B的cost变化。三seed聚合结果：

| outer-heldout指标 | base | wide | pair-delta直接头 |
|---|---:|---:|---:|
| 24-bank top-1 regret mean | 2.366 | 1.456 | **0.585** |
| 24-bank harmful top-1 | 1.22% | 0.72% | **0.39%** |
| 24-bank headroom recovery | 0.897 | 0.937 | **0.975** |
| Actor三候选 top-1 regret | 0.590 | 0.530 | **0.487** |
| Actor三候选 harmful选择 | 9.00% | 8.33% | **6.83%** |
| latest-vs-initial回归AUC | 0.881 | 0.902 | **0.912** |
| severe回归零阈值召回 | 0.556 | 0.624 | **0.742** |

episode-cluster bootstrap的配对差异也支持真实收益：wide-base的bank regret为`-0.910`
（95% CI `[-1.928,-0.032]`），pair-base为`-1.782`（`[-3.241,-0.705]`）；Actor三候选regret
分别为`-0.060`（`[-0.112,-0.018]`）和`-0.103`（`[-0.213,-0.024]`）。因此不能再说
“扩大Critic完全无效”：近2×容量对主体排序和风险AUC有稳定小幅收益。但不同seed的极值regret仍
有明显方差，单纯扩大模型不是尾部的完整解法。

Pairwise Delta是本轮更强结论：它只增加约16.5%参数，却在3/3 seed同时改善bank top-1 regret、
Actor候选有害选择、回归AUC和严重回归召回。说明Actor训练真正需要的局部比较量
`J(s,a_new)-J(s,a_base)`，确实比两个独立absolute值相减更适合作为辅助监督/风险评分。这也修正了
“Critic只需继续加大”的判断：容量有效，但目标结构匹配更重要。

安全边界仍未解除。2.8m/s层Pairwise回归AUC为`0.850/0.798/0.870`，严重回归召回仅
`0.750/0.556/0.500`；约四分之一至一半高速严重尾部仍会漏判。本轮也没有用Pairwise头更新Actor。
因此它目前只放行为OAC候选排序/风险辅助的下一阶段输入，不授权移除训练时真实DBM accept/reject、
不授权移除部署two-center warm guard，也不恢复单次Critic action-gradient路线。

独立validator复算所有保存指标，重新DBM rollout全部外层heldout Actor候选，最大cost误差为0；
同时验证三臂相同训练schedule、1.937×参数比、checkpoint合同、Actor更新0和split隔离。

```text
scripts/model_verify/run_mppi_oac_critic_capacity_pairdelta_ab.py
scripts/model_verify/validate_mppi_oac_critic_capacity_pairdelta_ab.py
scripts/model_verify/analyze_mppi_oac_critic_capacity_pairdelta_ab.py
outputs/mppi_proposal/oac_critic_capacity_pairdelta_20260825_v1/summary.json
outputs/mppi_proposal/oac_critic_capacity_pairdelta_20260825_v1/analysis.json
outputs/mppi_proposal/oac_critic_capacity_pairdelta_20260825_v1/validator_report.json
```

### 11.106 Pairwise Delta接入在线OAC：主体收益小幅提高，但尾部风险变差（2026-08-25）

按§11.105的放行边界完成真实在线接入。实现采用两套相互独立的Twin网络：原Absolute Twin Value
Critic、优化器状态、主Actor value gradient及连续move coefficient合同完全不变；新增Twin
Pairwise Delta仅学习同状态下
`log1p(J_new)-log1p(J_selected)`，并通过
`max(delta1,delta2)+0.5*abs(delta1-delta2)`提供候选回归风险。Pair网络先在历史Replay上冻结Actor
预训练1600步，随后与Absolute Critic使用同批online Replay、每轮各更新20步；坏动作不删除，真实
DBM只控制selected checkpoint，部署仍为一次Actor前向加two-center guard。

第一组20轮×3 seed使用原adaptive-tail合同（margin `0.05`、budget `0.02`、dual从0开始）。Pair
排序准确率在线保持约`0.90--0.92`，但三seed的dual始终为0，故Pair风险没有进入有效Actor loss。
新旧selected指标最大差仅`3.1e-4`，证明新增模型初始化、预训练RNG和optimizer没有旁路扰动旧
Absolute Twin/Actor链路；同时也说明该组不能评价Pair风险收益。

为使单变量真正生效，追加一个明确标记为机制压力测试的A/B：两臂都从`tail dual=10`开始，其余
20轮、3 seed、Replay、DBM、3std、trust与checkpoint floor完全一致。结果取三seed selected均值：

| 指标 | active Absolute风险 | active Pair Delta风险 | Pair变化 |
|---|---:|---:|---:|
| mean gain | 1.635 | **1.806** | +0.171（3/3 seed提高） |
| median gain | 0.372 | **0.436** | +0.064（3/3提高） |
| gain P05 | **-1.068** | -1.351 | -0.283（0/3提高） |
| worst gain | **-16.915** | -20.969 | -4.054（0/3提高） |
| regression fraction | **26.61%** | 27.72% | +1.11pp |
| 2.4m/s gain P05 | **-1.342** | -1.986 | -0.644（0/3提高） |
| 2.8m/s gain P05 | **-2.925** | -3.040 | -0.114（0/3提高） |
| two-center guard gain | 9.784 | **9.812** | +0.028 |

所以§11.105的“固定Replay下Pair排序更准”成立，但不能直接推出“用Pair作Actor尾部风险更安全”。
当前Pair头会释放更多主体收益，却漏掉/低估一部分高杠杆回归，表现为mean/median改善而P05、worst
和高速tail一致恶化。正式判定为
`PAIR_DELTA_ACTIVE_TAIL_MIXED_NO_FORWARD_AUTHORIZATION`：代码能力保留且默认关闭；不替换现有
Absolute风险、不进入200轮/outer-fold扩展、不恢复OAC-4。后续若再利用Pair，优先放在候选排序或
与Absolute风险取更保守并集的辅助项，而不是单独接管tail loss。

原OAC validator对四个run的Actor/Replay/DBM重放均通过，DBM cost最大误差为0。新增validator还
严格加载6个Pair checkpoint，验证参数量`544,706`、交换输入反对称误差0、`delta(a,a)`误差0、
每seed预训练1600步和在线400步计数，formal validation/test未读取。

```text
scripts/model_verify/mppi_pair_delta_critic.py
scripts/model_verify/train_mppi_oac2_continuous_actor.py
scripts/model_verify/validate_mppi_oac2_pair_delta.py
scripts/model_verify/analyze_mppi_oac2_pair_delta_ab.py
outputs/mppi_proposal/online_absolute_sac_oac2_pairdelta_adaptive_tail_smoke_20260825_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_active_tail_absolute_smoke_20260825_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_active_tail_pairdelta_smoke_20260825_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_pairdelta_ab_analysis_20260825_v1/analysis.json
```

### 11.107 当前OAC Critic与真实DBM梯度差距：局部梯度已基本恢复，主差距转为目标聚合与Actor预算（2026-08-25）

为解释§11.95--11.97“可微DBM直接训练Actor可达heldout J=10.30，而持续OAC仍远离J16”的差距，
对§11.101完成200轮后的fold-1最终Twin Critic做只读审计。使用从未写入该run Replay的600个
internal-selection状态，在每seed的selected/latest Actor绝对8×2动作处，比较可微DBM真实
`d log1p(J50)/d action`、Twin Critic梯度、相同动作trust步的真实J50，以及传到共享Actor参数的
`mean raw J`、`mean log1p(J)`和Critic梯度。formal validation/test未读取，网络更新数为0。

全部原run cost分布复算最大误差`3.6e-6`。独立validator另取48状态/seed/role重算：DBM cost与
参数步cost误差0，DBM梯度误差不超过`1.9e-6`；Critic因CUDA批大小变化的最大数值差`2.14e-4`，
低于单独登记的`5e-4`门。

latest三seed平均结果：

| 指标 | 当前最终Critic |
|---|---:|
| conservative log-value Pearson | 0.975 |
| action-gradient cosine median / P10 | **0.979 / 0.726** |
| gradient norm ratio median | 1.128 |
| early steering knot 0--2 cosine median / P10 | **0.996 / 0.835** |
| true-gradient最低25% flat层 cosine median / P10 | 0.949 / 0.509 |

因此历史§11.83静态warm/bank-best上的梯度失败不再代表经过31万Actor-visited Replay和4000次新增
更新后的当前Critic。Twin mean的P10为`0.750`，略优于conservative max的`0.726`，逐状态取max
带来小幅分支噪声，但不是数量级主因。

对每个状态的动作独立走固定trust半径（方向诊断，不等同共享Actor一次参数更新）后：

| latest三seed平均 | 真DBM方向 | Critic方向 |
|---|---:|---:|
| 0.002σ mean/median gain | 2.025 / 0.833 | 1.936 / 0.778 |
| 0.002σ P05 / 回归比例 | **+0.077 / 0%** | +0.012 / **4.17%** |
| 0.005σ mean/median gain | 4.937 / 1.984 | 4.721 / 1.823 |
| 0.005σ P05 / 回归比例 | **+0.140 / 0.67%** | -0.003 / **5.33%** |
| 0.020σ mean/median gain | 17.240 / 5.546 | 16.522 / 5.184 |
| 0.020σ P05 / 回归比例 | -0.655 / 9.33% | -1.506 / 13.17% |

Critic误差仍表现为约4--5%的额外小步回归tail，故不能移除DBM gate/two-center guard；但主体方向和
幅值已经接近真值，不能解释与DBM直训的主要平均差距。

最大的已测差距是跨状态目标权重。单状态上`d log1p(J)/da=dJ/da/(1+J)`方向相同；共享Actor对
一批状态求和时，高cost状态却被强烈降权。当前Critic对准确`mean log1p(J)`的Actor参数梯度cosine
为`0.927/0.980/0.964`；准确log目标与§11.95容量实验的准确`mean raw J`梯度cosine只有
`0.718/0.359/0.325`，最终Critic相对raw-J为`0.592/0.391/0.207`。

复用当前AdamW moments并把一步输出都校准到`0.00025σ RMS`时，raw DBM / 准确log DBM / Critic
的真实mean gain为`0.0849/0.0450/0.0383`；在`0.002σ`时为`0.681/0.350/0.298`。raw-J方向的
平均收益约为Critic两倍，但回归比例也更高（约39.7% vs 35.7%），复现了“raw J追高cost收益，
log J更均衡但mean慢”的权衡。该参数步在同一600状态池做机制评价，不是泛化证明。

Actor预算也是独立阻塞：当前200轮OAC每轮实际输出移动中位仅
`0.000251/0.000271/0.000208σ RMS`，最大不到`0.00059σ`，trust projection触发0次；§11.97可微
DBM容量实验则使用2400次Actor update、学习率`2e-4`，OAC只有200步、`1e-6`。

正式判定为
`CRITIC_LOCAL_GRADIENT_MOSTLY_RECOVERED_OBJECTIVE_AND_ACTOR_UPDATE_CONTRACT_DOMINATE_GAP`。
下一步冻结Critic结构/Replay方式，先做Actor聚合的单变量A/B：从log聚合逐步恢复raw-cost权重，
预注册`gamma=0/0.5/1`并clip高cost权重；目标确定后再单独提高Actor输出步长/预算。不得首轮同时
扩大Critic、提高Actor LR并改loss。

```text
scripts/model_verify/analyze_mppi_oac2_critic_dbm_gradient_gap.py
scripts/model_verify/validate_mppi_oac2_critic_dbm_gradient_gap.py
scripts/model_verify/summarize_mppi_oac2_critic_dbm_gradient_gap.py
outputs/mppi_proposal/oac2_critic_dbm_gradient_gap_20260825_v3/{analysis.json,summary.json,validator_report.json}
```

### 11.108 OAC-2C raw-cost-aware Actor聚合：mean收益确定提高，tail仍为混合结果（2026-08-25）

按§11.107的单变量顺序完成20轮×3 seed短pilot。Critic仍回归稳定的`log1p(J50)`；只在Actor反传时
使用脱梯度状态权重
`w=clip(exp(gamma*q_conservative), 2048)`，再除以batch均值。`gamma=0`严格保留原mean-log路径；
`gamma=1`在cap未触发时与`mean raw J`具有相同的Actor参数梯度方向，`gamma=0.5`为折中。Critic、
Replay生成、20:1更新比、Actor LR、3std输出、adaptive-tail、DBM checkpoint门和随机seed均不变。

三臂均通过独立OAC validator；DBM replay误差0，formal validation/test未读取。`gamma=0`与历史20轮
baseline的选中轮完全一致，关键指标最大差`3.51e-4`，属于CUDA操作复现范围。

三seed平均结果：

| 20轮latest | gamma=0 | gamma=0.5 | gamma=1 |
|---|---:|---:|---:|
| mean gain | 2.049 | 2.764 | **3.059** |
| median gain | 0.462 | **0.485** | 0.404 |
| P05 gain | -2.225 | -1.659 | **-1.446** |
| worst gain | **-24.67** | -26.87 | -28.61 |
| regression fraction | 30.28% | **28.33%** | 31.28% |
| headroom recovery | 0.0237 | 0.0319 | **0.0353** |

在checkpoint选中结果上，mean gain为`1.514/1.946/2.610`，P05为
`-1.407/-1.037/-1.180`，worst为`-20.44/-22.19/-25.93`。因此gamma=1的平均收益提升不是单seed
偶然：latest与selected均为3/3 seed提高mean；但selected P05仅2/3 seed提高、worst仅1/3提高，
不能声称raw目标同时修好了tail。

高速层的主体收益也随gamma增加。latest的2.4m/s mean gain为`3.024/4.221/4.742`，2.8m/s为
`2.927/4.389/5.202`。高速P05平均在gamma=1下也改善，但仍只有2/3 seed配对改善，不能移除现有
高速floor。

权重诊断解释了收益与风险：gamma=`0/0.5/1`的有效样本比例约为`1.000/0.622/0.249`；gamma=1
把训练预算集中到约四分之一的高cost状态。cap触发比例仅`0.143%`，所以本轮确实近似raw-J梯度，
不是截断伪影；但这种集中也使worst对seed更敏感。

正式判定：
`RAW_COST_AGGREGATION_IMPROVES_MEAN_BUT_TAIL_MIXED_SHORT_PILOT`。回答“能否用mean J”是肯定的，
而且它明显加快平均cost下降；但它不是免费的全面改进。gamma=1作为下一次独立Actor预算/步长实验的
主体收益臂，gamma=0.5作为较保守的tail对照。不得据此直接启动formal/test或取消DBM checkpoint门、
P05/worst高速floor和two-center guard。

```text
scripts/model_verify/train_mppi_oac2_continuous_actor.py
scripts/model_verify/validate_mppi_oac2_continuous_actor.py
scripts/model_verify/analyze_mppi_oac2_raw_cost_aggregation_ab.py
outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma{0,05,1}_smoke_20260825_v1/
outputs/mppi_proposal/oac2_raw_cost_aggregation_ab_20260825_v2/analysis.json
```

### 11.109 OAC-2C 200轮预算曲线：20轮低估主体收益，但raw mean未传导到guard且tail恶化（2026-08-27）

为回答§11.108的20轮预算是否过少，严格保留fold-1、3 seed、20:1 Critic更新、Actor LR、Replay、
3std、adaptive-tail和checkpoint floor，将gamma=0.5与gamma=1扩展到200轮；gamma=0复用已验证的
历史200轮run。三臂都有round 20/100/200的同口径真实DBM评价，两个新增run均通过独立validator，
formal validation/test未读取。

latest三seed平均预算曲线：

| gamma / round | mean gain | median gain | P05 | worst | headroom R | guard J | 2.8m/s P05 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 / 20 | 2.049 | 0.462 | -2.225 | -24.67 | 0.024 | 19.310 | -4.736 |
| 0 / 100 | 6.332 | 1.204 | -1.858 | -60.85 | 0.073 | 19.072 | -5.461 |
| 0 / 200 | 10.749 | **1.622** | **-2.126** | **-111.85** | 0.124 | **18.868** | **-6.861** |
| 0.5 / 20 | 2.764 | 0.485 | -1.659 | -26.87 | 0.032 | 19.346 | -4.858 |
| 0.5 / 100 | 8.051 | 1.026 | -1.781 | -90.75 | 0.093 | 19.127 | -5.076 |
| 0.5 / 200 | 13.740 | **1.644** | -2.480 | -163.23 | 0.159 | **18.839** | -7.210 |
| 1 / 20 | 3.059 | 0.404 | -1.446 | -28.61 | 0.035 | 19.418 | -4.083 |
| 1 / 100 | 9.267 | 0.884 | -1.770 | -111.32 | 0.107 | 19.193 | -5.983 |
| 1 / 200 | **14.828** | 1.188 | -2.430 | -185.51 | **0.171** | 18.983 | -8.456 |

所以“20轮过少”成立：三臂的mean gain到200轮仍比20轮高约4--5倍，不能用短pilot估计容量。
但长跑同时关闭了“只需继续加轮数”的简单解释。gamma=1相对gamma=0的200轮latest mean gain增加
`+4.079`（3/3 seed），但median少`0.433`、P05少`0.304`、worst少`73.67`，回归比例增加
`3.39pp`；2.8m/s P05少`1.595`。200轮时三臂的2.8m/s floor均为3/3 seed失败。

最终安全selected结果也必须与latest分开：

| selected | gamma=0 | gamma=0.5 | gamma=1 |
|---|---:|---:|---:|
| selected rounds | 20/80/10 | 20/80/10 | 80/20/80 |
| mean gain | 2.758 | 3.565 | **6.394** |
| median gain | **0.641** | 0.597 | 0.588 |
| P05 | -1.477 | **-1.293** | -1.442 |
| worst | **-26.28** | -39.60 | -67.96 |
| headroom R | 0.032 | 0.042 | **0.073** |
| two-center guard J | **19.296** | 19.325 | 19.314 |

gamma=1确实让安全checkpoint的direct mean收益翻倍以上，但guard后的mean cost没有改善，且0/3 seed
优于gamma=0。解释是raw权重把更新集中到少量高direct-cost状态；这些状态即使降低很多，部署时仍常由
warm center胜出，因此direct mean收益没有传到`min(J_warm,J_actor)`。与此同时batch=128下gamma=1
有效样本比例约0.25，相当于每次Actor更新主要由约32个状态支配，符合seed敏感worst扩大的现象。

正式判定：`RAW_MEAN_LONG_RUN_DIRECT_GAIN_UP_GUARDED_GAIN_FLAT_TAIL_WORSE`。不继续原样增加到
400/800轮。下一步先做零新增训练的权重×warm选择归因，量化raw权重有多少落在guard最终拒绝的状态；
随后做单变量大Actor batch/梯度累积以检验权重方差，另将worst floor加入checkpoint记录。若目标是
部署guard收益，再测试warm-relative、带margin且clip的advantage权重；若目标是无guard Actor逼近
bank-best，则仍用direct指标，但必须同时过median/P05/worst门，不能只追mean。

### 11.110 跨§11.91--11.109复核：补回DBM容量、支撑与tail实验后的归因修正（2026-08-27）

对§11.109的下一步重新审阅时，不能只沿`gamma=0/0.5/1`的raw聚合曲线解释。中间已有四组会改变
优先级的决定性测试：§11.95--11.97真实DBM task-loss容量与`+/-3std`支撑、§11.91--11.94长预算/
CVaR/多候选、§11.101--11.103自适应tail与state-wise风险回拉，以及§11.105--11.107
Critic容量、pairwise与真实DBM梯度审计。完整口径如下：

| 已完成测试 | 核心结果 | 对当前归因的约束 |
|---|---|---|
| DBM task-loss，box3，train-1200 | `J=7.282` vs J16 `5.013`，恢复率`97.41%` | 当前G-X结构、优化器与3std动作域具有逼近J16的容量 |
| DBM task-loss，box3，episode-heldout-600 | `J=10.300` vs J16 `4.463`，恢复率`93.26%` | 结构容量可跨episode迁移；不是“Actor天然只能到20附近” |
| OAC raw聚合，200轮latest | `gamma=1`恢复率`17.1%`，median/P05/worst=`1.188/-2.430/-185.51` | 主要差距仍在训练合同/梯度源随轨迹累积，而不是3std支撑 |
| 当前OAC Critic只读梯度审计 | action cosine median/P10=`0.979/0.726`；`0.002sigma`步额外回归约`4.17%` | Critic主体局部方向已恢复，但仍有少量tail，不能据此解释约76pp容量差 |
| fixed CVaR / adaptive dual / statewise shrink | fixed CVaR只保留约`14%`收益；adaptive保留`69.16%`但不过门；statewise尾部更差 | 不能继续把共享loss权重扫描当作主修复路线 |
| pairwise delta在线接入 | mean/median小幅改善，P05/worst与高速tail一致变差 | ranking有辅助价值，但不能代替主value梯度或安全裁决 |

因此对§11.109末尾“权重归因 -> 大batch -> warm-relative loss”的口径作如下收紧，不删除原始记录：

1. **主问题仍是逼近J16，而不是优化guard指标。** two-center guard是部署安全结构；它把坏Actor候选
   拒绝掉是预期行为。训练主目标过早改成warm-relative会把问题缩成“只学会击败warm”，与用户关心的
   Actor逼近理论最优不一致，也重复触碰§11.101--11.103已经失败的共享风险加权路线。
2. **大batch只保留为二级方差诊断。** `gamma=1`的batch ESS约`0.249`确实可能放大seed方差，但
   它不能单独解释DBM真梯度`93.26%`与OAC `17.1%`之间的差距；在梯度源未配对前，不把
   `batch 128 -> 512`列为首要修复。
3. **下一项首要判决实验改为matched gradient-source trajectory A/B。** 从同一box3 Actor、同一
   train-side状态batch序列、相同raw-mean目标、LR、optimizer、更新次数和动作投影出发，仅比较：
   `(A)`可微DBM真实梯度；`(B)`当前持续更新Twin Critic提供的raw-aware梯度。每个checkpoint都用真实
   DBM复算direct mean/median/P05/worst、速度分层、Actor-vs-warm比例及到J16的恢复率。
4. 判决树固定为：若相同预算下DBM臂也慢，则先调Actor LR/update预算/全batch优化；若DBM臂明显好而
   Critic臂落后，则差距来自小量梯度误差在共享参数长轨迹中的累积或Critic更新时序；若两臂主体都好
   但tail仍差，再进入逐状态safe projection/teacher或部署guard，不再扫全局loss权重。

这一复核不推翻§11.109的raw聚合实测结果，只修正其后续优先级。formal validation/test继续封存；
matched A/B只允许train-side固定DBM/internal-selection，不启动闭环，也不移除two-center guard。

### 11.111 OAC-2D初始大LR余弦衰减：主体加速成立，tail-safe checkpoint为0/3（2026-08-27）

为直接验证§11.107/11.110的Actor更新预算嫌疑，在fold-1、`gamma=1`、box3、adaptive-tail、
`20 Critic : 1 Actor`、相同Replay/状态batch/seed和`0.02sigma RMS` trust合同下，仅将固定
Actor LR=`1e-6`改为：round 1--20保持`1e-5`，20--80余弦衰减到`3e-6`，80--200继续余弦衰减
到`1e-6`。正式A/B显式使用CUDA，与历史gamma=1基线设备一致。第一次因沙箱NVML不可见而默认
落到CPU的长run只保留为非正式诊断，不进入以下配对统计。

实际Actor动作步验证了调度确实生效：round 1--20平均`0.00309sigma RMS`，21--80为
`0.000974sigma`，81--200为`0.000268sigma`；三段均无一次触发`0.02sigma` trust projection。
初始段比固定LR首轮约大6--12倍，不是“LR数字变化但action没动”。

200轮三seed平均、相对固定LR=`1e-6`严格配对结果：

| latest指标 | 固定LR | 初始大LR衰减 | 配对变化 |
|---|---:|---:|---:|
| mean J | 76.968 | **69.409** | **-7.559**（3/3改善） |
| median J | 18.825 | **17.953** | **-0.872**（3/3改善） |
| mean gain | 14.828 | **22.387** | **+7.559** |
| headroom recovery | 17.09% | **25.83%** | **+8.74pp** |
| 回归比例 | 24.72% | **20.50%** | **-4.22pp** |
| gain P05 | **-2.430** | -2.621 | -0.191 |
| mean worst gain | **-185.515** | -212.895 | -27.380 |
| 2.4m/s mean / P05 gain | 21.906 / **-3.411** | **33.270** / -4.219 | +11.364 / -0.808 |
| 2.8m/s mean / P05 gain | 29.434 / **-8.456** | **44.332** / -12.913 | +14.897 / -4.457 |
| latest two-center guard J | 18.983 | **18.657** | -0.326 |

主体加速是确定性的：3/3 seed的mean、median、headroom、回归比例和latest guard J均改善；大致
20--30轮就达到原固定LR约200轮的主体收益。actor-visited Critic pair全程保持约`0.92--0.95`，
没有随更快Actor移动系统性下降，进一步降低“Critic立即跟不上”的优先级。

但严重度tail变差且高度集中在高速层：2.4/2.8m/s P05在3/3 seed均不如固定LR，2.8m/s P05
平均降到`-12.913`。所有10轮间隔checkpoint的2.8m/s P05门都是3/3 seed失败；三seed
`selected_round=[0,0,0]`，因此可部署selected Actor没有任何收益。该现象不是“更多状态变坏”——
回归比例反而下降4.22pp——而是**更少但更严重的高杠杆回归**。

独立CUDA validator对每个seed的Replay DBM cost、initial/latest/selected指标复算误差均为0；
LR逐轮重放、optimizer最终LR、dual递推、hash、split和封存合同全部通过。最终qualification为FAIL
仅因性能门0/3，正式判定为
`LR_DECAY_MEAN_ACCELERATION_PASS_TAIL_SAFE_SELECTION_FAIL`。

这轮把“固定`1e-6`过小”从猜测升级为已证实因素，但仍只把J16 headroom从17.1%提高到25.8%，
不能解释剩余约74.2%。后续不直接扩大到更高LR；若继续调度，只允许单变量比较“`1e-5`最高段缩短
到10轮”或“初始降到`5e-6`”，观察能否保留主体加速并降低高速P05。若不能，则保留衰减作为
mean-capacity机制，tail改到逐状态DBM/bank safe projection作用点；不再回到共享tail-weight扫描。
matched DBM-vs-Critic梯度源A/B仍保留，用于解释剩余主体差距。

```text
scripts/model_verify/train_mppi_oac2_continuous_actor.py
scripts/model_verify/validate_mppi_oac2_continuous_actor.py
scripts/model_verify/analyze_mppi_oac2_lr_decay_ab.py
outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_lrdecay_cuda_200round_20260827_v1/
outputs/mppi_proposal/oac2_lr_decay_ab_20260827_v1/analysis.json
```

### 11.112 OAC-2E Actor LR边界扫描：主体边界在trust饱和区，tail为独立阻塞（2026-08-27）

应“继续探边界”的要求，覆盖§11.111中“不再提高LR”的执行限制，完成CUDA严格配对的初始LR
几何扫描。所有臂保持fold-1、gamma=1、box3、adaptive-tail、Replay、Critic `20:1`、batch、seed、
raw聚合和`0.02sigma RMS` action-space trust合同不变；round 1--20保持各自初始LR，20--80均按
相同余弦形状衰减到`3e-6`。主比较固定在round 80，因此与既有`1e-5`长run的前80轮完全同口径。

三seed平均结果如下：

| 初始Actor LR | mean J | median J | J16 headroom recovery | 回归比例 | gain P05 | 2.8m/s P05 | 前20轮步长 | trust投影率 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1e-5` | 74.781 | 18.430 | 19.66% | 19.00% | -1.847 | -7.411 | 0.00309sigma | 0% |
| `2e-5` | 68.433 | 17.451 | 26.99% | 19.22% | -2.436 | -10.085 | 0.00551sigma | 0% |
| `4e-5` | 60.815 | 16.400 | 35.80% | 18.61% | -4.401 | -22.257 | 0.00950sigma | 5.0% |
| `8e-5` | 51.563 | 14.571 | 46.49% | **18.22%** | -4.097 | -30.247 | 0.01517sigma | 26.7% |
| `1.6e-4` | 45.620 | **13.589** | 53.35% | 18.67% | -6.012 | -51.386 | 0.01915sigma | 76.7% |
| `3.2e-4` | **43.176** | 13.688 | **56.17%** | 21.33% | -8.265 | -47.041 | 0.02019sigma | 100% |

主体结论是单调且决定性的：提高有效更新步长把round-80恢复率从19.7%连续推到56.2%，mean J从
74.8降到43.2；因此此前只恢复约四分之一并非Actor结构上限，也不能主要归因于Critic。所有臂前20轮
Critic pair均约`0.916--0.925`，更快Actor没有造成Critic立即失配。

边界也已定位：`8e-5`是最大“多数更新尚未被裁剪”的臂；`1.6e-4`已在76.7%的早期更新触发trust，
是当前合同的实际运行边界；`3.2e-4`为100%投影饱和确认臂。它相对`1.6e-4`只再增加2.82pp恢复率，
mean J改善2.44，但median反而恶化0.10、回归比例增加2.66pp。继续提高名义LR不会扩大有效步长，只会
改变Adam/投影后的路径，因此停止更高LR扫描。

tail与主体优化可以分开归因，但不能忽略：随着主体恢复增加，P05/high-speed P05整体显著恶化，尤其
2.8m/s P05在`1.6e-4`降到`-51.4`。六个臂全部`selected_round=[0,0,0]`，没有一个checkpoint通过
既有部署门。独立CUDA validator对五个新增臂的Replay、DBM cost、LR重放、optimizer、dual、hash、
split和封存合同逐seed全通过；最终FAIL只来自性能门0/3。

正式判定为`LR_BOUNDARY_TRUST_SATURATION_MEAN_PASS_TAIL_FAIL`：

1. 主体能力研究可使用`1.6e-4 + 0.02sigma trust`作为trust-limited probe；`3.2e-4`只保留为饱和证据，
   不作为推荐优化器设置；
2. 不再继续提高名义LR，也不再次扫描共享CVaR/tail权重；
3. 剩余约44% headroom继续由matched DBM-vs-Critic梯度源A/B解释；
4. tail安全移到逐状态DBM/bank safe projection与two-center guard作用点，不能用tail失败反推主体步长
   无效，也不能用mean改善授权部署；
5. formal validation/test和闭环继续封存。

```text
scripts/model_verify/analyze_mppi_oac2_lr_boundary_scan.py
outputs/mppi_proposal/oac2_lr_boundary_scan_20260827_v1/analysis.json
outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_lrscan_2e5_90round_20260827_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_lrscan_4e5_90round_20260827_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_lrscan_8e5_90round_20260827_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_lrscan_16e5_90round_20260827_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_lrscan_32e5_90round_20260827_v1/
```

### 11.113 OAC-2F放宽全局trust：0.04sigma进入主体平台，继续放宽无结构性收益（2026-08-27）

为避免把§11.112的`0.02sigma`投影上限误读为Actor/Critic上限，固定初始LR=`3.2e-4`及其余全部
OAC合同，只扫描`max_step_sigma_rms=0.02/0.04/0.06/0.08`。三seed、round-80严格配对；本轮
预注册为主体能力实验，tail完整记录但不作为主体边界否决条件。

| 全局trust | mean J | median J | J16 headroom recovery | 回归比例 | gain P05 | worst | 2.8m/s P05 | 前20轮实际步长 | 投影率 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.02sigma` | 43.176 | 13.688 | 56.17% | 21.33% | -8.266 | -398.8 | -47.04 | 0.0202sigma | 100% |
| `0.04sigma` | 36.181 | **12.415** | 64.21% | **23.06%** | **-10.701** | -431.1 | -71.02 | 0.0362sigma | 53.3% |
| `0.06sigma` | 36.461 | 12.613 | 63.89% | 24.22% | -11.037 | -386.7 | -52.13 | 0.0453sigma | 25.0% |
| `0.08sigma` | **35.718** | 12.626 | **64.73%** | 23.89% | -12.126 | **-304.7** | -55.78 | 0.0506sigma | 16.7% |

从`0.02`到`0.04sigma`是决定性主体增益：mean J降低6.995、median降低1.273、恢复率增加
8.05pp。继续放宽后形成窄平台：`0.04/0.06/0.08`恢复率仅在`63.89%--64.73%`之间。`0.08`
相对`0.04`只改善mean J `0.463`，且仅2/3 seed改善；median反而差`0.212`。因此不能把
`0.08`的微小均值优势解释为仍存在可扩展的全局步长收益。

Critic没有立即崩溃：前20轮pair accuracy从`0.02`的`0.9157`仅降到`0.04/0.06/0.08`的
`0.9095/0.9079/0.9080`。Actor输出也未撞`+-3std`支撑：`0.06/0.08` action saturation仅约
`0.007%/0.003%`，state-any saturation约`0.11%/0.06%`。所以本轮平台既不是trust继续裁剪，
也不是输出盒饱和，更可能是当前Critic梯度累计、共享Actor优化或目标聚合的边界。

tail没有随trust严格单调，但整体仍明显恶化：P05从`-8.27`降至`-10.70/-11.04/-12.13`；回归比例
增加约1.7--2.9pp。worst与2.8m/s P05跨seed受少数case影响而非单调，不能用`0.08`的worst回升
宣称安全改善。所有臂仍`selected=[0,0,0]`、性能门0/3。

正式判定为`TRUST_004_TO_008_MEAN_PLATEAU_TAIL_SEPARATE_FAIL`：

1. 停止继续放宽全局trust；`0.04sigma`是达到平台的最小、最高效主体工作点；
2. `0.08sigma`只保留为无明显继续增长的边界证据，不作为推荐默认；
3. 下一主实验固定`3.2e-4 + 0.04sigma trust`做matched DBM-vs-Critic gradient-source A/B，
   解释剩余约35.8% headroom；
4. tail继续作为独立部署分支，由逐状态DBM/bank safe projection和two-center guard处理；
5. 三个新增臂独立CUDA validator的工程/复算检查全部通过，formal validation/test和闭环继续封存。

```text
scripts/model_verify/analyze_mppi_oac2_trust_boundary_scan.py
outputs/mppi_proposal/oac2_trust_boundary_scan_20260827_v1/analysis.json
outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_lr32e5_trust004_90round_20260827_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_lr32e5_trust006_90round_20260827_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_lr32e5_trust008_90round_20260827_v1/
```

### 11.114 OAC-2G LR×trust交叉臂：均值再增2.86pp，但seed/median已混合（2026-08-27）

§11.113只在初始LR=`3.2e-4`下扫描trust；为检查`0.06sigma`是否因LR不足而未被充分利用，补做唯一
交叉臂：初始LR从`3.2e-4`翻倍到`6.4e-4`，trust固定`0.06sigma`，其余调度节点、Replay、Critic
`20:1`、batch、seed、raw目标和评测合同严格不变。

round-80三seed平均结果：

| 配置 | mean J | median J | J16 headroom recovery | 回归比例 | gain P05 | 前20轮步长 | trust投影率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `3.2e-4 × 0.06sigma` | 36.461 | **12.613** | 63.89% | **24.22%** | **-11.037** | 0.0453sigma | 25% |
| `6.4e-4 × 0.06sigma` | **33.942** | 13.359 | **66.75%** | 28.67% | -18.003 | 0.0587sigma | 75% |

更高LR确实继续释放了一部分mean容量：mean J改善2.519，恢复率增加2.86pp，说明原`3.2e-4`没有
完全用足`0.06sigma`。但改善不是稳健单调：mean/headroom仅2/3 seed改善，median只有1/3改善且平均
恶化0.746；回归比例3/3恶化、平均增加4.44pp。高速mean收益仍大，但不能抵消主体分布和tail的混合。

前20轮实际步长已到`0.0587sigma`，75%更新被`0.06sigma`投影；Critic pair仍为`0.9081`，与
低LR臂`0.9079`几乎相同。因此结果不是Critic立即失配，而是更大的共享更新开始呈现明显seed/median
方差。该臂接近联合边界，但尚不能声称严格pure-mean硬上限；若未来必须精确测硬上限，仍需一个几乎
100%投影的更高LR饱和确认臂。

正式判定为`LR64E5_TRUST006_MEAN_MIXED_GAIN_NEAR_JOINT_BOUNDARY`：

1. 稳定工作点仍保持`3.2e-4 + 0.04sigma`；
2. `6.4e-4 + 0.06sigma`仅作为mean-capacity probe，不授权部署；
3. matched DBM-vs-Critic梯度源A/B可以使用稳定工作点为主，并将本臂作为更激进的容量旁证；
4. tail继续独立处理，所有checkpoint仍selected=0、性能门0/3；
5. 独立CUDA validator的工程、DBM复算和封存检查全部通过。

```text
scripts/model_verify/analyze_mppi_oac2_lr_trust_cross_ab.py
outputs/mppi_proposal/oac2_lr_trust_cross_ab_20260827_v1/analysis.json
outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_lr64e5_trust006_90round_20260827_v1/
```

```text
scripts/model_verify/analyze_mppi_oac2_raw_cost_budget_curve.py
outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma05_200round_20260825_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_200round_20260825_v1/
outputs/mppi_proposal/oac2_raw_cost_budget_curve_20260827_v1/analysis.json
```

### 11.115 采样中心评价口径修正：统一改为确定性direct center相对warm（2026-08-27）

后续需要在DBM与Query等不同rollout模型间评价同一个策略中心，无法统一获得可信的`J16`或理论最优。
因此通用评价对象收紧为**策略网络输出的单个确定性中心本身**：对相同状态、history、reference和cost
合同，只分别rollout未加噪的warm center与Actor center；不运行MPPI采样、不使用wrapper/softmax、
不使用two-center执行结果，也不依赖oracle。

对任意rollout模型`M`，基础量定义为：

```text
J_w^M(s) = J^M(s, a_warm)
J_a^M(s) = J^M(s, a_actor)
G_w^M(s) = J_w^M(s) - J_a^M(s)
```

通用主指标固定为：Actor严格超过warm的比例、`G_w`中位数、`G_w`的P05/worst、以及
`sum(G_w) / sum(J_w)`的聚合warm-relative改善；按速度和场景分层。若比较DBM与Query，比较各自
warm-relative分布、改善符号一致率与冲突帧，禁止直接把不同模型的绝对cost尺度当成策略优劣。
`J16 headroom recovery`只保留为DBM专项容量旁证，不再作为跨模型通用指标。

按该合同复算最近的高更新臂：初始LR=`6.4e-4`、trust=`0.06sigma`、round 90、fold-1
internal-selection 600状态、3 seed。这里仅列确定性Actor center；表中三seed值为逐seed统计的平均。

| 指标 | warm | Actor center |
|---|---:|---:|
| direct cost mean | 29.174 | 33.849 |
| direct cost median | 23.356 | 13.335 |
| Actor严格超过warm | -- | 69.28% |
| 配对`G_w` median | -- | +6.726 |
| 配对`G_w` mean | -- | -4.675 |
| 聚合warm-relative改善 | -- | -16.0% |

逐seed结果为：

| seed | 超过warm | `G_w` median | `G_w` mean | Actor mean J |
|---:|---:|---:|---:|---:|
| 0 | 71.33% | +7.127 | -1.682 | 30.856 |
| 1 | 69.33% | +6.701 | -5.580 | 34.754 |
| 2 | 67.17% | +6.349 | -6.763 | 35.937 |

速度分层的聚合mean只作尾部定位，不替代胜率/中位数：

| speed | warm mean J | Actor mean J | warm-relative改善 |
|---:|---:|---:|---:|
| 1.2m/s | 11.464 | 13.370 | -16.6% |
| 1.6m/s | 17.682 | 10.920 | +38.2% |
| 2.0m/s | 29.136 | 28.839 | +1.0% |
| 2.4m/s | 40.285 | 38.383 | +4.7% |
| 2.8m/s | 47.301 | 77.733 | -64.3% |

与稳定工作点`3.2e-4 + 0.04sigma`的round-90直接比较：高更新臂的超过warm比例从
`67.83%`升到`69.28%`，`G_w` median从`+6.611`升到`+6.726`，Actor mean J从`36.118`
降到`33.849`；但两臂的`G_w` mean仍分别为`-6.944/-4.675`，都没有在聚合口径下超过warm。
所以放大LR/trust对典型中心质量只有小幅改善，主要作用是减轻部分大cost状态；2.8m/s的大幅负迁移
仍使均值失败。结论不是“Actor普遍差”，而是“约69%状态获益，但约31%未赢warm且损失幅值更大”。

本节同时覆盖§11.112--§11.114中容易混淆的字段解释，但不删除原实验记录：

1. 旧`gain_vs_initial`、`regression_fraction`与`gain P05/worst`均以**初始Actor**为基线，不是warm；
   例如§11.114的`28.67%`不能再称为warm-relative回归率。对应warm配对中，round-90高更新臂的
   “未超过warm”比例为`30.72%`；
2. `two_center_guard.warm_selected_fraction`在这里只用来恢复`P(J_actor < J_warm)`，不把guard
   cost当成中心质量；由于Actor胜率超过50%，已存的截断正收益median与未截断`G_w` median相同；
3. 当前summary没有保存round-80/90完整逐状态`G_w`数组，因此不能诚实复算warm-relative P05/worst
   或逐状态归一化收益；禁止借用相对初始Actor的P05/worst填充。后续评测必须直接落盘
   `warm_cost/actor_cost/gain_vs_warm`逐状态数组及速度/场景字段。

```text
outputs/mppi_proposal/absolute_action_value_critic_20260820_v1/candidate_bank.npz
outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_lr32e5_trust004_90round_20260827_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_lr64e5_trust006_90round_20260827_v1/
```

### 11.116 warm-relative逐状态正式重评分与高尺度gamma单变量A/B（2026-08-27）

§11.115建立了正确的中心评价口径，但当时只能从旧summary间接恢复部分统计，尚未保存完整的逐状态
`gain_vs_warm`。本节实现独立重评分器，在固定fold-1 internal-selection的600状态、30 episode上，
对已保存Actor latest checkpoint与bank第0列warm action分别做确定性DBM J50 rollout。评价中不运行
MPPI sampling、wrapper、softmax、two-center执行逻辑或oracle，并将
`warm_cost/actor_cost/gain_vs_warm/episode/snapshot/speed/scenario/action`全部落盘。

边界必须明确：warm是已实现的一次随机MPPI过程产生的状态相关结果，**只作为配对评价参照**；
`J_warm`没有进入Actor输入、loss、reward、anchor或训练权重。本节之后也禁止以warm-relative loss
替代绝对cost训练目标。线性地从每个状态的Actor cost减去常数`J_warm`不会改变动作梯度；对随机
`J_warm`做非线性加权反而会把采样器噪声注入训练。

正式重评分的pooled结果如下。warm在五组运行中完全相同，mean/median J为`29.174/23.356`。

| Actor latest | 胜warm比例 | Actor mean/median J | `G_w` mean/median | `G_w` P05/worst | 聚合相对改善 |
|---|---:|---:|---:|---:|---:|
| gamma0，200轮，低步长 | 56.83% | 81.046 / 18.444 | -51.872 / +2.894 | -404.623 / -1681.934 | -177.8% |
| gamma0.5，200轮，低步长 | 56.78% | 78.057 / 18.375 | -48.883 / +2.942 | -385.857 / -1646.973 | -167.6% |
| gamma1，200轮，低步长 | 56.11% | 76.968 / 18.784 | -47.794 / +2.683 | -368.925 / -1645.714 | -163.8% |
| gamma1，`3.2e-4/0.04sigma` | 67.83% | 36.118 / 12.296 | -6.944 / +6.602 | -114.410 / -904.064 | -23.8% |
| gamma1，`6.4e-4/0.06sigma` | **69.28%** | **33.849** / 12.955 | **-4.675 / +6.748** | **-99.976 / -606.039** | **-16.0%** |

这组完整数组修正了§11.115基于旧字段的一个局限：相对同一warm，高LR/trust臂不仅胜率和典型收益
更好，P05与worst也比稳定臂更好，不能再说它的warm-relative尾部随更新尺度恶化。旧
`gain_vs_initial`的尾部退化与新的`gain_vs_warm`不是同一比较对象。与此同时，高尺度臂的mean gain
和聚合改善仍为负，说明`69.28%`状态虽胜过warm，但剩余失败状态的损失幅值仍更大；它仍不满足
“单中心整体替代warm”的条件。

在此基础上只做了一个训练侧单变量A/B：固定高尺度臂的初始Actor、fold、Replay、Critic、LR调度、
trust、轮数、seed与绝对cost训练合同，只把绝对cost聚合指数从`gamma=1`改为`gamma=0.5`。

| 高尺度臂 | 胜warm比例 | `G_w` mean/median | `G_w` P05/worst | Actor mean J | 聚合相对改善 |
|---|---:|---:|---:|---:|---:|
| gamma1 | **69.28%** | **-4.675 / +6.748** | **-99.976 / -606.039** | **33.849** | **-16.0%** |
| gamma0.5 | 67.67% | -9.809 / +6.101 | -127.115 / -821.581 | 38.983 | -33.6% |

gamma0.5在所有pooled门上都更差；逐seed的median/P05/worst为`0/3`改善，胜率仅`1/3`改善。
因此正式判定为`GAMMA05_HIGH_WARM_RELATIVE_DECISIVE_FAIL_KEEP_GAMMA1`：保留gamma=1，停止
gamma/tempering扫描，不引入`J_warm`训练。当前结果仅授权“高尺度gamma1是现有Actor中最好的
warm-relative单中心proposal”，不授权部署、formal validation/test或闭环；two-center guard仍是
最终MPPI候选层的独立安全结构，不能混入本节中心质量统计。

复现链：

```text
scripts/model_verify/evaluate_mppi_oac_warm_relative_centers.py
scripts/model_verify/validate_mppi_oac_warm_relative_centers.py
scripts/model_verify/analyze_mppi_oac_warm_relative_gamma_ab.py
outputs/mppi_proposal/oac_warm_relative_centers_20260827_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_gamma05_lr64e5_trust006_90round_20260827_v1/
outputs/mppi_proposal/oac_warm_relative_gamma05_high_ab_20260827_v1/
```

### 11.117 同一outer-round内K=1/4/8次Actor微更新：K=8有效但只部分闭合差距（2026-08-27）

§11.116关闭gamma/tempering扫描后，剩余直接假设是：每个Actor-visited outer round只更新一次Actor，
使Actor没有充分利用同轮已经更新的Critic。为避免把“更多Actor步”与“更大总步长”混在一起，本节实现
了严格配对的multi-update pilot：每轮Critic总更新固定20次，分别交错执行`K=1/4/8`个Actor
microstep；调度LR和单步trust都除以K，每轮累计trust仍固定`0.06sigma RMS`，temperature和tail dual
仍各更新一次。三臂均为90 outer rounds、fold-1 internal-selection、600状态、3 seed、gamma=1、
`+-3std`支撑，formal validation/test与闭环均未运行。

实现合同增加了显式`--multi-actor-update-pilot`，只允许`K in {1,4,8}`；Critic的20次更新在K个
microstep前做平衡整数分割，Actor/interaction/Critic随机流隔离，并对每轮起始Actor到最终Actor的
累计动作位移重新投影。独立validator逐seed复算确认：Actor/Critic更新数、LR回放、K个microstep、
Critic分区、单步与累计trust、temperature一次/轮、tail dual递推、checkpoint和DBM replay全部通过，
DBM replay最大误差为0。

按latest checkpoint容量口径，J16/bank-best headroom recovery学习曲线如下；数值为3 seed均值：

| K | Actor更新/seed | round 10 | round 20 | round 40 | round 80 | round 90 | round-90 seed std |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 90 | 0.3503 | 0.3426 | 0.4774 | 0.6546 | 0.6562 | 0.0454 |
| 4 | 360 | 0.5482 | 0.5603 | 0.5942 | 0.6617 | 0.6627 | 0.0044 |
| 8 | 720 | 0.5344 | **0.6230** | **0.6586** | **0.7102** | **0.7101** | **0.0011** |

K=4主要消除了seed方差，最终均值仅比K=1高0.65pp；K=8则在round 20后持续领先，round 90比K=1
提高5.39pp，并把seed标准差从4.54pp压到0.11pp。K=8的三seed最终恢复率为
`0.7116/0.7089/0.7098`，不是单seed偶然性。与初始Actor相比，K=8 mean J为`30.275`，低于K=1的
`34.876`；median J为`10.577`，低于`13.823`。

按§11.115--11.116固定的warm-relative direct-center合同重新做完整逐状态DBM rollout，结果为：

| K | 胜warm比例 | Actor mean/median J | `G_w` mean/median | `G_w` P05/worst | 聚合warm改善 |
|---:|---:|---:|---:|---:|---:|
| 1 | 67.39% | 34.876 / 13.582 | -5.702 / +6.355 | -102.272 / -826.325 | -19.55% |
| 4 | 70.78% | 34.394 / 11.104 | -5.221 / +7.089 | -100.815 / -996.834 | -17.89% |
| 8 | **74.56%** | **30.275 / 10.587** | **-1.101 / +7.638** | **-84.558** / -832.075 | **-3.78%** |

因此multi-update不是只改善内部oracle指标：相对同一warm，K=8比K=1的胜率提高7.17pp、median gain
提高1.283、P05提高17.714、聚合改善提高15.77pp。与此同时，K=8的mean gain仍为负、worst仍为
`-832`，所以它没有通过“单中心整体替换warm”的门。K=4的worst还比K=1更差，也说明不能根据主体
均值把tail视为自然消失。

正式判定为`K8_MULTI_ACTOR_MICROSTEPS_MECHANISM_PASS_PARTIAL_GAP_CLOSURE`：

1. 证实单轮一次Actor更新是一个真实优化瓶颈；后续OAC训练默认采用K=8，暂不扫描K>8；
2. 下一主实验是在K=8、相同累计trust下增加outer Actor-visited refresh轮数，观察0.71是否继续上升，
   而不是继续放大LR/trust或修改gamma；
3. 历史exact-DBM task-loss heldout容量参考为0.9326，K=8仍有约22.25pp差距；两者split/训练合同不同，
   这里只作容量参考，不作严格配对gate；
4. 旧initial-relative tail/high-speed selection gate使三臂selected round均为0，因此本节只报告latest
   mechanism capacity；工程/replay逐seed通过不等于性能门通过；
5. two-center guard、formal validation/test和闭环继续封存，不授权部署或单中心替换warm。

复现链：

```text
scripts/model_verify/train_mppi_oac2_continuous_actor.py
scripts/model_verify/validate_mppi_oac2_continuous_actor.py
scripts/model_verify/evaluate_mppi_oac_warm_relative_centers.py
scripts/model_verify/validate_mppi_oac_warm_relative_centers.py
scripts/model_verify/analyze_mppi_oac2_multiupdate_ab.py
outputs/mppi_proposal/online_absolute_sac_oac2_multiupdate_k1_90round_20260827_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_multiupdate_k4_90round_20260827_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_multiupdate_k8_90round_20260827_v1/
outputs/mppi_proposal/oac2_multiupdate_warm_relative_20260827_v1/
outputs/mppi_proposal/oac2_multiupdate_ab_20260827_v1/analysis.json
```

### 11.118 K=16/32边界补测：K=16为效率拐点，K=32进入边际收益区（2026-08-28）

§11.117的K=8仍有明确收益，不能据此关闭更大K。本节继续保持相同的90 outer rounds、20次Critic
更新/轮、`0.06sigma RMS`累计round trust、gamma=1、`+-3std`、fold-1 internal-selection与3 seed，
只把Actor microstep扩展到K=16和K=32。LR与单步trust继续按K除法；K=32时20次Critic更新按平衡
整数分区，意味着12个Actor microstep不会在其前面获得新的Critic更新，正好用于检测过度利用同一Critic
近似的边界。两臂smoke和完整独立validator均逐seed通过更新计数、分区、LR、累计trust、Replay、
checkpoint及DBM复算，formal validation/test与闭环仍封存。

完整K曲线如下：

| K | Actor更新/seed | round-90 recovery | seed std | 胜warm | `G_w` mean/median | `G_w` P05/worst | 聚合warm改善 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 90 | 0.6562 | 0.0454 | 67.39% | -5.702 / +6.355 | -102.272 / -826.325 | -19.55% |
| 4 | 360 | 0.6627 | 0.0044 | 70.78% | -5.221 / +7.089 | -100.815 / -996.834 | -17.89% |
| 8 | 720 | 0.7101 | 0.0011 | 74.56% | -1.101 / +7.638 | -84.558 / -832.075 | -3.78% |
| 16 | 1440 | 0.7411 | 0.0043 | **76.11%** | +1.573 / +7.889 | -69.671 / -640.281 | +5.39% |
| 32 | 2880 | **0.7492** | 0.0095 | 75.72% | **+2.294 / +7.986** | **-56.011 / -614.093** | **+7.86%** |

K=16相对K=8仍是实质提升：recovery `+3.10pp`、胜warm `+1.56pp`、P05 `+14.89`，并首次让pooled
warm-relative mean与聚合改善转正。K=32相对K=16只再增加`0.81pp` recovery；三seed配对增量为
`+0.29/+0.02/+2.13pp`，主要由seed 2驱动。K=32的胜warm比例还下降`0.39pp`，median只增加
`0.098`。它仍把mean gain提高`0.721`、P05提高`13.66`、pooled worst提高`26.19`，因此不是退化，
而是从主体优化转入tail/少数大cost状态的边际改善。

K=16和K=32的Critic pair最终仍约`0.89--0.91`，没有出现Critic立即崩溃；但K=32已经让Actor更新数
超过同轮Critic更新数，且主恢复增量低于1pp。因此本轮正式判定为
`K32_DIMINISHING_RETURN_BOUNDARY_K16_EFFICIENCY_KNEE`：

1. K=16作为计算效率默认：它用K=32一半Actor更新，保留绝大多数主体收益，并给出最高胜warm比例；
2. K=32作为离线质量候选：当训练成本次要、优先降低direct-center mean/P05时可使用；
3. 不继续K=64。当前已满足预登记的“下一次翻倍主恢复增量<1pp”边界，而且K=64会进一步增加没有新
   Critic refresh的Actor步，难以再保持单变量解释；
4. K=32相对warm整体已经转正，但2.8m/s仍是负层，且旧性能选择门仍0/3、selected round仍0，不能
   因pooled转正而授权部署；
5. 下一主项转向Actor-visited状态刷新/Replay新鲜度，而不是继续增加同一Critic上的Actor内循环。
   two-center guard、formal validation/test与闭环边界不变。

```text
outputs/mppi_proposal/online_absolute_sac_oac2_multiupdate_k16_90round_20260828_v1/
outputs/mppi_proposal/online_absolute_sac_oac2_multiupdate_k32_90round_20260828_v1/
outputs/mppi_proposal/oac2_multiupdate_warm_relative_20260828_v3/
outputs/mppi_proposal/oac2_multiupdate_boundary_20260828_v2/analysis.json
```

### 11.119 K=20严格1:1 Critic/Actor交错：没有支配K=16（2026-08-28）

为检验“20次Critic更新对应20次Actor更新可能最自然”的假设，补充K=20严格配对臂。每个microstep
前恰好执行一次Critic更新，不存在K=16的`1/2`次不均匀分区，也不存在K=32的零Critic刷新Actor步。
其余训练和评价合同与§11.118完全相同。smoke与90轮×3 seed独立validator全部通过，DBM replay误差0，
formal validation/test与闭环未加载。

关键结果：

| K | recovery | seed std | 胜warm | `G_w` mean/median | P05/worst | 聚合改善 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | **0.7411** | **0.0043** | **76.11%** | **+1.573** / +7.889 | -69.671 / **-640.281** | **+5.39%** |
| 20 | 0.7400 | 0.0088 | 75.83% | +1.481 / **+8.109** | **-68.597** / -645.449 | +5.08% |
| 32 | 0.7492 | 0.0095 | 75.72% | +2.294 / +7.986 | -56.011 / -614.093 | +7.86% |

K=20相对K=16的配对变化为：recovery `-0.10pp`、胜warm `-0.28pp`、mean gain `-0.092`、聚合改善
`-0.32pp`、worst `-5.17`；只有median `+0.220`和P05 `+1.074`小幅改善。三seed recovery为
`0.7277/0.7476/0.7448`，其中seed 0明显低于K=16，不支持“严格1:1节奏更稳定”。2.8m/s层同样没有
恢复：胜warm `48.89%`、gain median `-0.547`、P05 `-219.27`。

正式判定为`K20_ONE_TO_ONE_NO_DOMINANT_GAIN_KEEP_K16_K32_BOUNDARY`。这说明Critic和Actor名义更新次数
相等不是目标本身；决定效果的是共享累计步长、Actor随机优化轨迹与Actor-visited Replay，而不是简单
追求更新计数对称。默认仍使用K=16；K=32保留为离线mean/P05优先的质量臂；K扫描正式结束，不再运行
K=64。下一步转向Actor-visited状态刷新和Replay新鲜度。

```text
outputs/mppi_proposal/online_absolute_sac_oac2_multiupdate_k20_90round_20260828_v1/
outputs/mppi_proposal/oac2_multiupdate_warm_relative_20260828_v4/
outputs/mppi_proposal/oac2_multiupdate_boundary_20260828_v3/analysis.json
```

### 11.120 Actor-visited刷新频率翻倍的等预算A/B：没有额外收益（2026-08-28）

为隔离“Actor新动作多久进入Replay并反馈给Critic”的影响，固定K=16基线的总预算和所有有效步长，
只把每个outer round拆成两个half-round：基线为`90×(256 contexts, 20 Critic, 16 Actor)`，候选为
`180×(128 contexts, 10 Critic, 8 Actor)`。两臂每seed均为23040次context访问、1800次Critic更新、
1440次Actor更新和90次temperature/tail更新；每个Actor microstep的LR与trust均完全相等，评估次数也
相等。候选因此只把Actor-visited反馈延迟减半，没有增加算力。独立validator确认三seed的Replay、
更新计数、LR/trust回放、tail dual、DBM复算均通过，formal validation/test未加载。

最终内部headroom recovery从K=16的`0.7411`变为`0.7360`，下降`0.51pp`。warm-relative direct-center
结果如下：

| 指标 | K=16基线 | 2x刷新 | 差值（刷新-基线） |
|---|---:|---:|---:|
| 胜warm比例 | 76.11% | 75.56% | -0.56pp |
| gain mean | +1.573 | +1.134 | -0.439 |
| gain median | +7.889 | +8.064 | +0.175 |
| gain P05 | -69.671 | -71.800 | -2.129 |
| gain worst | -640.281 | -693.762 | -53.481 |
| 聚合改善 | +5.39% | +3.89% | -1.51pp |

2.4m/s两臂接近；2.8m/s的mean gain由`-26.695`变为`-28.451`，仍是主要负层。正式判定为
`ACTOR_VISITED_REFRESH_2X_EQUAL_BUDGET_NO_GAIN`：刷新翻倍只使median微增，recovery、胜率、mean、
P05/worst和聚合改善均未改善。当前Replay采样本来就固定50% recent rows；在这一合同下，把反馈块从
`20/16`拆成`10/8`不是主要瓶颈。

因此保持K=16、90轮的简单组织，不继续扫描刷新频率。下一项若继续OAC，应改变Actor-visited状态的
**信息内容/覆盖方式**或训练目标，而不是只缩短同分布Replay延迟。旧性能选择门仍为0/3、selected
round仍为0；工程通过不能解释为部署授权，two-center guard、formal/test和闭环边界不变。

```text
scripts/model_verify/analyze_mppi_oac2_refresh_cadence_ab.py
outputs/mppi_proposal/online_absolute_sac_oac2_refresh2x_equal_budget_180round_20260828_v1/
outputs/mppi_proposal/oac2_refresh2x_warm_relative_20260828_v1/
outputs/mppi_proposal/oac2_refresh_cadence_ab_20260828_v1/analysis.json
```

### 11.121 Matched DBM-vs-Critic gradient-source A/B：Critic有可测代价，但不是剩余主差距（2026-08-28）

此前§11.95--11.97的真实DBM task-loss实验证明了Actor容量，但其2400次更新、`2e-4`学习率、fold-0
split、固定状态训练和无OAC Replay/trust合同与当前K16 OAC不同，不能把`0.9326-0.7411`全部归因给
Critic。本节执行§11.110预登记的严格gradient-source A/B：两臂从相同box3 Actor开始，使用相同
fold-1 fit/selection、90轮、每轮256个Actor-visited状态、20次Critic更新、16次Actor microstep、
staged LR、`0.06sigma`累计trust、gamma=1、连续move coefficient、SAC entropy、adaptive tail、
Replay和随机流规则。唯一注册变量是Actor cost梯度源：

- Critic臂：现有Twin conservative `log1p(J50)`，用detached `exp(q)`恢复raw-cost参数梯度；
- DBM臂：sampled-action value和mean-vs-selected tail都由可微确定性DBM J50产生；Critic仍持续训练，
  Actor-visited动作仍全部写入Replay，move coefficient在两臂都只作为detached系数。

DBM臂的smoke与正式90轮×3 seed独立validator全部通过：梯度源逐microstep登记、Replay行数/角色、
DBM cost复算、Critic更新、LR/trust、temperature/tail dual递推与checkpoint hash均一致；formal/test、
wrapper和闭环未运行。旧性能门仍使两臂`selected_round=0`，以下均为latest机制容量。

主结果：

| 指标 | Critic K16 | matched DBM | DBM-Critic |
|---|---:|---:|---:|
| headroom recovery | 0.7411 | **0.7599** | **+1.88pp** |
| recovery逐seed | 0.7361/0.7466/0.7406 | **0.7687/0.7620/0.7490** | **+3.26/+1.54/+0.84pp** |
| mean J | 27.601 | **25.970** | **-1.631** |
| median J（逐seed均值） | **10.087** | 10.381 | +0.294 |
| warm胜率 | 76.11% | **76.89%** | +0.78pp |
| warm gain mean/median | +1.573/+7.889 | **+3.204/+8.112** | +1.631/+0.223 |
| warm gain P05/worst | -69.671/**-640.281** | **-59.566**/-869.510 | +10.105/**-229.229** |
| warm聚合改善 | +5.39% | **+10.98%** | +5.59pp |

高速主体也改善：2.4m/s mean gain由`+7.526→+10.024`；2.8m/s由`-26.695→-22.659`，胜warm
由48.61%升到51.67%，median由`-1.222→+1.238`。但2.8m/s仍是聚合负层，且DBM臂出现更差的
pooled worst，说明极端tail不是Critic梯度错误的单因。

用历史fold-0 DBM task-loss heldout recovery `0.9326`只作尺度参考：当前Critic到该参考相差
`19.15pp`，matched DBM仅关闭`1.88pp`，约为参考差距的`9.8%`，仍剩`17.27pp`。这个比例不是严格
paired gate，因为容量参考的split、更新数与优化合同不同；但它足以否定“剩余主要全是Critic误差”。
真实DBM臂在相同OAC预算下也平台于约0.76，并在seed 2后段出现`0.754→0.749`回落，直接表明共享Actor
的跨状态梯度冲突、随机mean-policy目标、continuous coefficient/entropy/tail组合、LR/trust/Adam轨迹
与状态访问分布仍主导剩余差距。

正式判定为`MATCHED_DBM_GRADIENT_SMALL_STABLE_GAIN_SHARED_ACTOR_CONTRACT_DOMINATES`：

1. Critic小量梯度误差确实会累积，不能说Critic完全无损；
2. 但不再优先扩Critic网络或扫描Critic loss，约90%的历史参考差距未被真实梯度替换关闭；
3. 后续主体优化转向共享Actor合同：测参数梯度跨状态冲突、比较deterministic mean-action task surrogate
   与当前stochastic coefficient目标，或使用更直接的逐状态proposal监督；
4. DBM臂只能作机制oracle，Query部署仍必须使用Critic/真实query/search；
5. warm-relative tail、two-center guard、formal/test与闭环封存规则保持。

```text
scripts/model_verify/train_mppi_oac2_continuous_actor.py
scripts/model_verify/validate_mppi_oac2_continuous_actor.py
scripts/model_verify/analyze_mppi_oac2_matched_gradient_source_ab.py
outputs/mppi_proposal/online_absolute_sac_oac2_matched_dbm_gradient_k16_90round_20260828_v1/
outputs/mppi_proposal/oac2_matched_gradient_source_warm_relative_20260828_v1/
outputs/mppi_proposal/oac2_matched_gradient_source_ab_20260828_v1/analysis.json
```

### 11.122 确定性部署中心DBM直训A/B：目标错配有小量代价，但不是剩余主瓶颈（2026-08-28）

§11.121把Actor梯度源替换为真实DBM后，训练仍优化随机sampled action，并保留detached move
coefficient、SAC entropy与adaptive tail。本节进一步执行严格的Actor目标A/B。两臂均使用真实可微DBM、
相同初始box3 Actor、fold-1 fit/selection、90轮、每轮256个Actor-visited状态、20次Critic更新、
16次Actor microstep、staged LR、`0.06sigma`累计trust、Replay探索和DBM checkpoint gate。唯一变量是
Actor loss：

- stochastic DBM基线：沿用sampled-action value、move coefficient、entropy和tail；
- deterministic-center臂：只让部署时的确定性Actor mean进入
  `mean(J_DBM(s, mean_actor(s))) + trust`；sampled action仍用于Actor-visited Replay，但不向Actor反传；
  move coefficient、entropy和tail只保留监控，不影响Actor loss；Twin Critic仍按原合同持续训练。

新增`--actor-objective-mode deterministic_center_dbm`，并在独立validator中逐microstep核对：DBM sampled/
mean cost均有限、deterministic loss路径成立、temperature不更新、tail/move coefficient不影响Actor、Replay
角色与数量、20:16更新、LR/trust回放、tail dual监控递推、checkpoint hash均一致。1轮smoke通过；正式三
seed的所有工程检查通过。formal validation/test、MPPI wrapper和闭环均未加载。

latest机制容量结果：

| 指标 | stochastic exact-DBM | deterministic center-DBM | 差值 |
|---|---:|---:|---:|
| headroom recovery | 0.7599 | **0.7701** | **+1.02pp** |
| recovery逐seed | 0.7687/0.7620/0.7490 | 0.7522/**0.7880**/**0.7701** | -1.64/+2.60/+2.11pp |
| mean J | 25.970 | **25.102** | **-0.868** |
| median J（逐seed均值） | 10.381 | **8.513** | **-1.868** |
| initial-relative gain P05 | **-13.166** | -17.260 | -4.094 |
| initial-relative worst（逐seed均值） | **-303.43** | -568.39 | -264.96 |

deterministic臂的逐seed最高recovery分别为`0.7780@round30 / 0.7896@round60 / 0.7770@round50`，
说明主体收益在30--60轮已平台，增加到90轮不是主要限制。历史fold-0、2400-update DBM task-loss
`0.9326`仍只作尺度参考；deterministic臂到该参考仍差`16.25pp`，不能把剩余差距归因给随机SAC
action或entropy。

按当前确定性单中心warm-relative口径，deterministic臂相对stochastic DBM的latest结果为：

| 指标 | stochastic exact-DBM | deterministic center-DBM | 差值 |
|---|---:|---:|---:|
| 胜warm比例 | 76.89% | **80.06%** | **+3.17pp** |
| gain mean/median | +3.204/+8.112 | **+4.072/+9.832** | +0.868/+1.720 |
| gain P05/worst | -59.566/-869.510 | **-55.253/-863.231** | +4.314/+6.279 |
| 聚合改善 | +10.98% | **+13.96%** | **+2.97pp** |

但速度分层不是全面改善：2.4m/s gain mean从`+10.024`降到`+8.476`，median升到`+15.532`；
2.8m/s mean从`-22.659`降到`-23.261`，median从`+1.238`升到`+3.557`，P05从`-192.395`
降到`-201.190`。因此pooled warm-relative中心指标改善，不代表高速tail已经解决。

三seed的所有非零评估共`30`个，当前单中心checkpoint gate接受`0`个，`selected_round`均保持0。
原因不是工程失败，而是candidate相对初始Actor的P05远低于注册门：全体/2.4/2.8m/s floor分别为
`-2.5/-4.0/-4.1`，而各seed最佳recovery点仍为约`-11.7~-18.1 / -19.5~-28.1 /
-34.8~-67.8`。这与two-center guard可在部署时守住warm下界是两个口径：前者拒绝把单中心作为
独立安全checkpoint，后者允许把它作为有风险但总体更好的proposal；本节不修改既有gate，也不授权
部署或闭环。

正式判定为`DETERMINISTIC_CENTER_OBJECTIVE_SMALL_GAIN_TAIL_GATE_STILL_BLOCKS`：

1. sampled-action/entropy/move-coefficient目标错配确有代价，直接优化部署mean使recovery提高约1pp、
   warm聚合改善提高约3pp；
2. 该改动远未逼近历史容量参考，随机SAC目标不是剩余主瓶颈；
3. next blocker收窄为共享Actor参数优化：不同状态的真实DBM参数梯度是否互相抵消，以及Adam/LR/trust
   如何沿该合成梯度移动；
4. 下一项优先做零新数据的per-state/group parameter-gradient conflict audit，再决定PCGrad/CAGrad、
   分组更新或结构条件化；数据扩张与Actor网络扩容暂缓；
5. two-center guard、warm-relative分层、formal/test封存和闭环边界保持不变。

```text
scripts/model_verify/train_mppi_oac2_continuous_actor.py
scripts/model_verify/validate_mppi_oac2_continuous_actor.py
scripts/model_verify/evaluate_mppi_oac_warm_relative_centers.py
scripts/model_verify/validate_mppi_oac_warm_relative_centers.py
scripts/model_verify/analyze_mppi_oac2_deterministic_center_ab.py
outputs/mppi_proposal/online_absolute_sac_oac2_deterministic_center_dbm_k16_90round_20260828_v1/
outputs/mppi_proposal/oac2_deterministic_center_warm_relative_20260828_v1/
outputs/mppi_proposal/oac2_deterministic_center_ab_20260828_v1/analysis.json
```

### 11.123 固定Actor邻域探索bank等预算审计：guided-wide有小范围高价值收益，正交bank不占优（2026-08-28）

为区分“共享Actor参数梯度冲突”和“Actor附近根本没有探索到有效动作”，冻结§11.122三seed的latest
deterministic-center Actor，在同一fold-1内部selection 600状态上做bank-only审计。每个方法、每个状态
严格只有6次确定性DBM J50中心评价；不更新Actor/Critic，不加载formal/test，不运行MPPI wrapper或
闭环。四个独立等预算bank为：

1. `current`：注册实现的Actor mean、两组Gaussian/tanh antithetic pair和一条blind wide；
2. `orthogonal`：mean、两组按状态循环的RMS正交temporal-DCT antithetic pair和一条正交wide；
3. `guided_1x`：前五条与current bitwise一致，仅用两组中央cost slope拟合minimum-norm下降方向，
   以`1×std`替换第六条blind wide；
4. `guided_2x`：同上，但第六条为`2×std`。1x/2x各自都是独立6评价bank，不共享额外rollout。

生成器bitwise回放、共享Actor中心、guided前五条、候选预算、三seed checkpoint hash、全部数组hash和
汇总指标均由独立validator复算通过。exact DBM gradient只用于离线方向覆盖诊断，不参与候选生成；warm
只用于评价“Actor输warm状态能否找回”，不参与拟合。

pooled 1800个seed-state结果：

| bank | best-of-6 gain mean | 任一改善命中率 | Actor输warm时找回比例 | 每状态rank中位 | 真梯度span投影中位 |
|---|---:|---:|---:|---:|---:|
| current | 10.325 | 45.83% | 53.20% | 3 | **0.409** |
| orthogonal | 9.830 | **71.00%** | 42.62% | 3 | 0.368 |
| guided 1x | 10.430 | 46.72% | 52.92% | 2 | 0.318 |
| guided 2x | **11.605** | 45.67% | **55.15%** | 2 | 0.318 |

与current逐状态配对：

| 候选bank | best cost差均值（正=候选更好） | 候选胜/current胜/平 | episode-bootstrap mean 95% CI |
|---|---:|---:|---:|
| orthogonal | -0.495 | 49.06% / 26.39% / 24.56% | [-1.243, +0.182] |
| guided 1x | +0.105 | 13.33% / 2.89% / 83.78% | [-0.115, +0.303] |
| guided 2x | **+1.280** | 5.50% / 2.94% / 91.56% | **[+0.496, +2.197]** |

guided 2x的逐seed paired mean为`+2.273/+0.472/+1.096`，其中seed 0/2的episode-bootstrap CI为正，
seed 1含零。收益集中在少数高价值状态，而不是普遍提高命中率。速度层与该机制一致：current→guided
2x的best gain mean在2.4m/s为`13.847→16.201`，2.8m/s为`31.618→35.593`；低速层基本不变。

正交bank把“任意小改善”的命中率从45.8%提高到71.0%，但mean best cost反而略差，paired CI含零；它
更均匀地找到小下降，却丢失current random-wide偶尔带来的大收益，不能据命中率替换默认bank。
guided 1x同样没有明确配对收益。guided 2x证明第一次antithetic cost response确实含有可利用信息，
但其每状态动作span从rank 3降到rank 2；这对best-of-bank是可接受的exploitation，对训练Value Critic
是否更好尚未证明。当前OAC把全部候选写入Replay，而不是直接克隆best candidate，因此bank-only收益
不能等同于Actor训练收益。

正式判定为`GUIDED_2X_BANK_PAIRED_GAIN_SHORT_OAC_AB_JUSTIFIED_NOT_DEFAULT`：

1. 改变Actor邻域探索有可测价值，重点是用第一次cost response替换blind wide，而不是固定正交化；
2. 该价值只影响约5%状态、规模约`+1.28` best-cost，预期不能解释§11.122剩余16.25pp主体差距；
3. 下一步若继续此支线，只授权一个短、等rollout/等更新的OAC current-vs-guided2x A/B，并必须同时
   检查Critic fresh gradient、warm-relative中心、2.4/2.8m/s和P05/worst；
4. 在短A/B通过前不修改默认探索；共享Actor参数梯度冲突审计仍是主体优先级；
5. two-center guard、formal/test封存和闭环边界不变。

```text
scripts/model_verify/analyze_mppi_oac2_actor_neighborhood_banks.py
scripts/model_verify/validate_mppi_oac2_actor_neighborhood_banks.py
outputs/mppi_proposal/oac2_actor_neighborhood_bank_audit_20260828_v1/summary.json
outputs/mppi_proposal/oac2_actor_neighborhood_bank_audit_20260828_v1/evaluation.npz
outputs/mppi_proposal/oac2_actor_neighborhood_bank_audit_20260828_v1/validator_report.json
```

### 11.124 exact-DBM共享Actor参数梯度冲突审计：主要优化损失发生在action梯度进入共享网络之后（2026-08-28）

按§11.122预登记的主体优先级，完成当前三个latest deterministic-center DBM Actor在同一fold-1
internal-selection 600状态上的逐状态/分组参数梯度审计。本轮冻结Actor和Critic；Actor loss只取未加噪
确定性center的真实DBM raw J50。warm仅用于结果分层，不进入输入、loss、梯度或权重；formal
validation/test、Replay生成、MPPI wrapper与闭环均未运行。

对每个状态先计算真实动作梯度，再通过当前Actor Jacobian得到逐状态参数梯度：

```text
g_a^i = d J_DBM(s_i, pi_theta(s_i)) / d a
g_theta^i = (d pi_theta(s_i) / d theta)^T g_a^i
g_bar = mean_i g_theta^i
```

主要冲突指标如下。cancellation ratio定义为
`N * ||g_bar|| / sum_i ||g_theta^i||`；越接近零，说明逐状态有效梯度在共享参数空间抵消越强。

| seed | cancellation ratio | `cos(g_i,g_bar)<0`比例 | cosine median / P10 | 最差速度组对全局cosine |
|---:|---:|---:|---:|---:|
| 0 | 0.1085 | 42.0% | 0.039 / -0.213 | -0.072 |
| 1 | 0.1201 | 44.5% | 0.097 / -0.450 | -0.491 |
| 2 | 0.0788 | 45.0% | 0.018 / -0.164 | -0.011 |

三seed pooled的逐状态参数梯度对seed内全局方向cosine中位仅`0.0347`、P10为`-0.3215`，负对齐比例
`43.83%`。抵消不是单一输出头造成：encoder cancellation为`0.1267/0.1098/0.1039`，decoder为
`0.1060/0.1131/0.0856`，support adapter为`0.1089/0.1222/0.0769`；冲突贯穿共享表示、时序解码和
最后的3std适配器。

分层结果给出更具体机制：五个速度组参数梯度的两两cosine最小值为
`-0.818/-0.704/-0.517`，非对角中位为`-0.157/-0.095/-0.020`；速度regime之间存在稳定反向。raw
mean J同时强烈被最高cost四分位主导，该组对全局梯度cosine为`0.919/0.985/0.974`，而较低cost组
常接近零或反向。warm胜/负两组的夹角跨seed不稳定，因此冲突不能简化成“只训练Actor输warm状态”；
它主要是raw mean目标、高cost/高速权重与共享state-to-action Jacobian共同形成的局部Pareto冲突。

为把参数统计接到真实cost，本轮在相同输出RMS下比较逐状态独立动作步与一个共享参数步：

| pooled一步（sigma RMS） | mean/median gain | P05 | 回归比例 | 动作步对真实下降方向cosine中位 | 负对齐比例 |
|---|---:|---:|---:|---:|---:|
| 独立DBM动作步 0.002 | 0.902 / 0.394 | **+0.0446** | **0%** | 1.000 | 0% |
| 共享raw-SGD参数步 0.002 | 0.172 / 0.0137 | -0.483 | 45.1% | 0.075 | 43.9% |
| 共享saved-Adam参数步 0.002 | 0.327 / 0.0171 | -0.352 | 40.2% | 0.129 | 39.2% |
| 独立DBM动作步 0.020 | 6.414 / 1.845 | -1.291 | 19.6% | 1.000 | 0% |
| 共享saved-Adam参数步 0.020 | 2.685 / 0.0530 | -4.238 | 46.8% | 0.132 | 39.6% |

在最接近当前microstep的`0.002sigma`下，每个状态各自沿真实DBM下降方向走时600状态三seed全部不回归；
同样总RMS通过一个共享Actor参数步实现时，Adam只保留约36%的mean gain，约40%状态回归。Adam moments
明显优于裸SGD，但无法消除共享映射损失。该结果与§11.121/122闭合：Critic换成真实DBM只改善
`1.88pp`、确定性中心目标再改善`1.02pp`，是因为主要损失位于正确`g_a`进入共享Actor参数空间之后。

解释边界必须保留：

1. 本轮确认的是**当前点的局部共享参数优化冲突**，不是证明G-X Actor最终函数类不能表达更优映射；
   §11.97的`0.9326`仍是跨fold/跨合同容量参考，而非本轮严格上限。
2. raw mean J本来就会优先降低少数高cost状态；最高cost组主导全局方向不是软件bug。梯度手术若提高
   各组一致性，也可能牺牲纯mean下降，必须用真实DBM J50与分层tail共同评价。
3. §11.123探索bank与本轮主因解耦：deterministic DBM Actor直接在center获得真梯度，不依赖bank辨识；
   guided-wide可留作可部署Critic Replay支线，但不能解释本轮参数冲突。

正式qualification为`SHARED_ACTOR_PARAMETER_GRADIENT_CONFLICT_CONFIRMED`。下一步不扩大Critic、数据或
Actor宽度，先做冻结Actor、等输出RMS的零训练方向对照：plain batch mean、速度组PCGrad、速度/成本组
CAGrad（或MGDA型最小冲突方向），全部用真实DBM一步评价mean/median/P05、各速度收益与负对齐比例。
只有梯度手术方向在3/3 seed明显提高共享步收益保留、降低反向状态且不过度损失高cost收益，才进入短
exact-DBM deterministic-center训练A/B；若方向对照也失败，再转向speed/regime条件化head或轻量专家。

独立validator重新计算三seed全600状态的global梯度、每seed 8个逐状态全参数梯度、全部step DBM cost
和所有速度组梯度。global gradient cosine最低`0.9999999999999856`，逐状态梯度方向最大cosine误差
`1.12e-6`，step cost误差`0`，速度组cosine误差`3.58e-8`。

```text
scripts/model_verify/analyze_mppi_oac2_actor_parameter_conflict.py
scripts/model_verify/validate_mppi_oac2_actor_parameter_conflict.py
outputs/mppi_proposal/oac2_actor_parameter_conflict_audit_20260828_v1/analysis.json
outputs/mppi_proposal/oac2_actor_parameter_conflict_audit_20260828_v1/evaluation.npz
outputs/mppi_proposal/oac2_actor_parameter_conflict_audit_20260828_v1/validator_report.json
```

### 11.125 目标速度域切换：冻结低速算法结论，直接建立独立40--100 kph训练集（2026-08-28）

实际后续运行速度已明确为`40--100 km/h`（`11.11--27.78 m/s`），而§11.38--11.124使用的
固定数据域仅为`1.2--2.8 m/s`（`4.32--10.08 km/h`），两者没有重叠，目标域下限约为旧域
上限的4倍。因此§11.124的PCGrad/CAGrad方向级对照不再作为下一优先项：其冲突审计仍是
“共享Actor参数更新会损失逐状态真实下降方向”的有效方法学证据，但不能授权目标速度域的
算法选择、超参数或性能外推。

按用户决定，本阶段不做车辆真实性或DBM模型审计，直接进入独立高速数据生成。该决定覆盖归档
§10.3中“H0/H1先于H2”的执行顺序，但不覆盖以下数据纪律：

1. 高速数据不得并入历史`1.2--2.8 m/s`的train/validation/test；
2. 首批只建立train-only原始快照，不创建或消费formal validation/test；
3. DBM参数、cost、MPPI动作通道和snapshot schema保持不变；
4. 每个episode仍从step 250后采样并逐episode运行独立validator；
5. 本批数据只表示合成oracle目标域，不自动构成Query模型、实车或闭环资格。

首批合同为5档速度`40/55/70/85/100 km/h`、6类工况`steady/underspeed_recovery/
overspeed_recovery/lateral_recovery/heading_recovery/combined_recovery`，每个速度×工况1个完整
episode、每episode 20个fully-observed snapshots，共30 episode / 600 states。采集仍使用64条
MPPI候选以控制原始数据成本；后续teacher/search标签作为不可变raw collection之上的sidecar生成。
普通ROS运行的reference-speed上限仍为10 m/s；高速计划必须显式传入30 m/s ceiling，避免该实验
静默改变默认运行边界。

```text
scripts/model_verify/generate_fixed_dbm_highspeed_plan.py
scripts/model_verify/fixed_dbm_highspeed_train_20260828_v1.json
scripts/model_verify/collect_fixed_dbm_scenarios.py
/disk/collect_data_from_anycar/mppi_rl_closed_loop/
  fixed_dbm_highspeed_train_20260828_v1/
```

采集完成前qualification记为
`TARGET_SPEED_DATA_COLLECTION_IN_PROGRESS_LOW_SPEED_ALGORITHM_SELECTION_PAUSED`。完成后先只汇报
实际保存的episode/snapshot数、reference/observed-vx分布、validator状态和hash；不借本批train-only
数据作无偏性能声明。

#### 11.125.1 采集完成、实际速度口径修正与高速初始状态replay

原计划已完整执行：30/30 episode、600/600 snapshots、38,400条候选rollout和14,370条连续trace
记录均落盘，30个episode的schema-v2 validator全部通过；plan SHA256为
`db1676e2fb3ae1b88d03ff9ce030d4e39a754a05f40097547e93a7e48ad10144`。速度×工况严格平衡，
validation/test均为0。

但汇总发现必须修正“高速训练集”的表述：snapshot的reference确为`40--100 km/h`，可step 250--478
时的实际`vx`仅为`[-0.39,14.98] m/s`，中位`2.06 m/s`、P95 `11.17 m/s`。在当前小车DBM与
`cuc_inside`轨迹上，车辆在fully-observed采样窗口前已大幅失速/偏离；100 km/h层的横向误差P05为
`-37.2 m`，best-candidate cost中位`53,491`、最大`893,865`。因此该600帧集合只能定性为
**高reference-speed压力/失稳边界集**，不能宣称实际状态覆盖40--100 km/h。qualification修正为
`HIGH_REFERENCE_SPEED_STRESS_COLLECTION_COMPLETE_VALIDATED_ACTUAL_SPEED_TARGET_MISSED`。这是数据分布
核对，不是DBM真实性审计。

为保留实际高速物理状态，又不重新审计模型，随后从每个episode连续trace的step 0--4重建运行时
QueryHistoryBuffer，并按原episode seed/mean重放64条DBM候选，形成独立train-only compact replay：

- 30 episode × 5步 = 150 contexts，9,600条DBM候选；
- 实际`vx`为`8.33--29.12 m/s`（`29.99--104.84 km/h`），中位`18.22 m/s`（`65.60 km/h`）；
- 82.0% context严格位于40--100 km/h，低于40 km/h的underspeed-recovery与略高于100 km/h的
  overspeed状态保留并显式标记；
- best-candidate cost中位`165,884.7`，说明这仍是当前小车/紧凑赛道下的高难压力数据，不授权部署；
- 不创建formal validation/test，不消费sealed split。

```text
scripts/model_verify/generate_highspeed_initial_dbm_replay.py
scripts/model_verify/validate_highspeed_initial_dbm_replay.py
outputs/mppi_proposal/highspeed_collection_summary_20260828_v1/summary.json
outputs/mppi_proposal/highspeed_initial_dbm_replay_20260828_v1/{summary.json,replay.npz,validator_report.json}
```

当前qualification为`HIGHSPEED_INITIAL_TRAIN_REPLAY_COMPLETE`。低速PCGrad/CAGrad仍暂停；下一步若训练
目标域Actor，应优先使用150-context实际高速replay做pipeline smoke/标签生成，600帧压力集只用于失稳
边界和失败样本，不得把两者混成同一分布。完整高速episode覆盖仍需后续使用与目标速度相容的轨迹/
车辆合同，本轮按用户要求不展开模型审计。

### 11.126 实际高速context的proximal search teacher（2026-08-30）

已按§11.125.1继续完成train-only高速teacher生成。输入严格限定为150-context initial replay；anchor
是trace保存的`mean_knots_before`，不加载低速Actor/checkpoint。目标是固定DBM的确定性J50 direct-center
cost。每状态搜索预算固定为129个中心：warm anchor + 64条`0.25/0.50 sigma` Hadamard环形候选 +
两轮各32条重定位候选；warm始终作为candidate 0，因此teacher由构造不劣于warm。formal
validation/test未创建、未读取。

结果为：

- 150 contexts、19,350次center rollout；独立validator重放anchor/teacher cost最大误差为0；
- warm/teacher cost mean为`195,280.27/189,559.50`，中位为`173,596.73/168,197.88`；
- gain中位/均值/P05为`4,360.39/5,720.77/1,292.09`，150/150严格改善，baseline violation为0；
- 聚合相对warm降幅`2.9295%`，episode bootstrap 95% CI `[2.7251%,3.2156%]`；只看123个实际
  40--100 km/h context为`2.9182%`，不是由underspeed/overspeed边界撑起；
- teacher residual标准化RMS中位`0.553 sigma`、P05/P95 `0.348/0.561 sigma`，不是数值级微动；
- 118个teacher来自第二轮32候选，32个来自第三轮，首轮64候选和anchor均未成为最终best，证明
  re-centering在该域是必要预算；
- 所有状态均为move、stay为0。这反映当前高速warm普遍离局部更优中心较远，不能据此训练可靠的
  stay/flat gate；two-center warm guard仍是部署必需结构。

绝对cost随速度显著增长：名义40/55/70/85/100 km/h层teacher mean分别约
`35.6k/100.5k/175.9k/244.2k/391.6k`。因此本轮的正确读法是“黑盒search能稳定改善当前warm”，
不是“高速任务已解决”；相对改善仅约3%，且数据只有30个独立episode。下一步只授权episode-grouped
cross-fit的轻量Actor可学性pilot，主指标为warm-relative direct cost，不提前消费formal/test，也不把
600帧失稳压力集混入训练。

```text
scripts/model_verify/generate_highspeed_proximal_search_teacher.py
scripts/model_verify/validate_highspeed_proximal_search_teacher.py
outputs/mppi_proposal/highspeed_proximal_teacher_20260830_v1/
  summary.json
  labels.npz
  validator_report.json
```

qualification：`HIGHSPEED_PROXIMAL_SEARCH_TEACHER_COMPLETE_TRAIN_ONLY`。

### 11.127 高速Actor BC与Twin Critic预训练（2026-08-30）

已在§11.126的150个train-only context上完成正式`5 fold x 3 seed`预训练。30个episode整体分折；
每个run使用90帧fit、30帧internal selection、30帧outer OOF，任何相邻step均不跨集合。Actor采用
此前冻结的strict no-anchor clean G-X时序结构，输入仅`history/reference/current`，不读取warm、
first-pass feedback或gradient context；直接回归8x2 absolute sampling center。输出支持按fit折teacher
逐维统计固定为`+/-3 std`，避免旧`+/-1 std`盒再次成为瓶颈。`current`严格采用实车可测合同
`[vx, yaw_rate, acceleration, steering]`，不输入`vy/beta`。Twin Critic采用absolute-action标量
value网络，离线监督全部129个search center的确定性DBM J50；目标为标准化`log1p(J)`并附加同状态
ranking loss。本阶段没有TD/bootstrap、没有Actor-Critic联合更新，也没有flat/stay head（本批150帧
全部为move）。

Actor结果为：

- fit teacher-gain recovery中位`0.723`，说明网络与优化器可以拟合主体训练数据；
- OOF recovery中位`0.613`、均值`0.475`，15 run范围`[-0.783, 1.149]`；中位达到预登记`>=0.50`
  主门，但跨fold/seed极不稳定；
- OOF胜或等于warm比例中位`0.667`，回归比例中位`0.333`；15 run的warm-relative gain P05全部为负，
  P05中位约`-16.1k`，因此tail门失败；
- 失败按episode fold聚集：fold 0为`0.06/0.29/-0.78`，fold 2仅`0.03/-0.15/0.15`，fold 3为
  `0.61--0.81`，fold 4为`1.06--1.15`。这不是可用“挑最好seed”消除的随机波动，而是当前仅
  30个独立episode下的跨episode迁移问题；recovery大于1允许成立，因为search teacher只是129候选中的
  best，不是全局最优，Actor可落在未被teacher bank采到的更低cost中心。

Twin conservative Critic结果为：

- OOF `log1p(J)` Pearson中位`0.992`（范围`0.955--0.995`），但该指标包含速度层造成的绝对cost
  大尺度差异，不能单独证明局部动作排序已解决；
- warm-vs-teacher顺序准确率中位`1.0`、最小`0.967`；同状态129候选bank的gain recovery中位
  `0.588`、范围`0.366--0.729`。所以Critic已适合作为在线OAC的**初始化**，但仍需Actor-visited
  replay持续更新，禁止冻结后直接提供一次性Actor梯度；
- 独立validator重载全部15个Actor+Twin Critic checkpoint，fresh DBM重放warm/teacher最大误差均为0，
  Actor recovery与summary最大差`5.42e-8`、Critic相关最大差`1.76e-9`，episode leakage为0。

记录性修正：同名`v1`曾按legacy代码把`current`第二维设为`vy`；数值结论与v2相近，但违反此前
“实车不依赖vy/beta”的合同，已降级为非部署诊断，不作为OAC初始化。本文与后续计划只引用v2。

裁决：两个网络的预训练阶段完成，可作为后续同fold持续在线OAC的初始化；但不是部署放行。Actor的
OOF中位门通过而tail门失败，后续训练必须保留warm/Actor two-center guard，并以Actor-visited新动作
持续补Replay、Twin Critic持续更新。formal validation/test和600帧高reference失稳压力集继续封存；
在进入正式闭环前，仍需在train-only OOF上要求warm-relative主体收益稳定、P05/worst有界、不同速度/
scenario不出现系统性负迁移。

```text
scripts/model_verify/pretrain_highspeed_actor_twin_critic.py
scripts/model_verify/validate_highspeed_actor_twin_critic_pretrain.py
outputs/mppi_proposal/highspeed_actor_twin_critic_pretrain_20260830_v2/
  summary.json
  validator_report.json
  checkpoints/pretrain_fold{0..4}_seed{0..2}.pt
```

qualification：`HIGHSPEED_ACTOR_TWIN_CRITIC_PRETRAIN_COMPLETE_TRAIN_ONLY`，独立复算为
`HIGHSPEED_ACTOR_TWIN_CRITIC_PRETRAIN_INDEPENDENT_REPLAY_PASS`。

### 11.128 高速Actor-visited持续OAC首轮（2026-08-30）

已从§11.127的正式v2 checkpoint继续完成train-only、同fold的持续Actor--Critic更新。环境仍是固定
DBM确定性J50 terminal contextual bandit，而不是长时序闭环：没有next state、Bellman bootstrap、
target Critic或entropy项。每轮先由当前Actor产生一个absolute center，再在其周围加入16对Hadamard
antithetic probes；真实DBM cost全部写入Replay，变差动作也不删除。每轮更新比为20次Twin Critic、
1次Actor，共20轮；probe半径由`0.20 sigma`降至`0.05 sigma`。Actor使用Twin Critic较保守的
`max(log J)`方向，跨状态采用`gamma=1`的raw-J权重，单轮动作变化上限为`0.02 sigma RMS`。
warm不进入Actor输入或loss，只保留为外部评价基线与未来two-center guard候选。

checkpoint只由每run的30帧internal-selection真实DBM mean cost选择，30帧OOF从不参与选择。15个run中
selected round中位为20：12个选择round 20，另有round 19、round 13各一个；fold0/seed1的所有在线
round均不优于预训练，正确回退到round 0，说明gate确实阻止了已知的内部退化，而不是强制采用latest。

主结果如下：

- OOF teacher-gain recovery从预训练中位`0.613`提高到`0.941`，均值从`0.475`提高到`0.798`；选中
  Actor的15-run范围为`0.269--1.290`，不再出现负recovery；每个fold至少2/3 seed相对预训练改善；
- OOF胜或等于warm的比例中位由`0.667`提高到`0.800`，回归比例中位由`0.333`降到`0.200`；相对
  预训练Actor的OOF mean/median cost gain在run间的中位分别为`1593.1/1051.9`；
- 收益覆盖全部名义速度层。40/55/70/85/100 km/h层相对预训练的配对mean-gain增量中位约为
  `290/1130/998/3050/2360`；但该数值仍来自30个episode的train-only分折，不是formal泛化结论；
- 尾部联合门未通过：14/15 run的warm-relative P05仍为负，15-run P05中位为`-14.37k`，最差
  per-state gain仍可达约`-46.8k`。two-center guard按构造可把direct-center结果截断到warm下界，
  但单Actor不能无guard部署；
- final Twin Critic的fresh OOF 0.05-sigma局部probe方向准确率中位`0.698`，略低于预登记`0.70`门；
  centered log-cost Pearson中位`0.370`、局部bank gain recovery中位`0.552`。这说明持续Replay足以支持
  显著Actor主体改善，但局部价值排序尚未达到稳定放行标准。

独立validator重新加载全部15组selected Actor/Twin Critic，对全部OOF center与33候选probe做fresh
DBM重放，并抽查每run初始、首轮和末轮Replay动作。Actor center、center cost、probe cost、Replay
cost、Actor recovery及Critic metric与保存产物的最大差均为0；episode leakage为0，formal/test未创建，
warm未进入Actor loss。

裁决为：`Actor-visited持续OAC`在目标高速train-only数据上证明了明显主体收益，优于仅做BC预训练；
但联合资格仍为**未通过**，阻塞项是warm-relative尾部和Critic局部sign门，而不是mean/median收益。
因此当前checkpoint可作为下一轮机制研究或two-center候选中心，不授权单中心部署、formal test或闭环。
若继续，必须保持真实DBM internal-selection gate与two-center warm floor；优先处理局部排序/尾部，不再
仅靠增加相同20:1更新轮数解释剩余问题。

```text
scripts/model_verify/train_highspeed_actor_visited_oac.py
scripts/model_verify/validate_highspeed_actor_visited_oac.py
outputs/mppi_proposal/highspeed_actor_visited_oac_20260830_v1/
  contract.json
  summary.json
  validator_report.json
  fold{0..4}_seed{0..2}/{oac_checkpoint.pt,replay.npz}
```

qualification：训练产物为`HIGHSPEED_ACTOR_VISITED_OAC_COMPLETE_TRAIN_ONLY`；独立重放为
`HIGHSPEED_ACTOR_VISITED_OAC_INDEPENDENT_REPLAY_PASS`；registered joint gate为`2/4`通过。

### 11.129 高速独立状态扩张与固定更新预算复核（2026-08-30）

为检查§11.127--11.128的正结果是否只是30个episode小样本现象，新建独立train-only扩张集：
`5 speeds x 6 scenarios x 4 repeats x 5 early contexts = 120 episodes / 600 contexts`。每个context
保留64个原始MPPI候选，共38,400 candidates；实测速度范围`28.1--110.3 km/h`、中位`63.2 km/h`，
`78.3%`严格位于40--100 km/h。区间外样本是保留的underspeed recovery和少量overspeed边界，不做
cost过滤。120/120 episode均逐episode通过schema validator，未创建formal validation/test。

采集过程发现并修复两个仅影响数据合同的工程问题：step 0同一callback刚写snapshot时不再被trace
overwrite guard误判为旧数据；DBM simulator metadata尚未到达时延后snapshot，避免落盘空车辆参数。
一次metadata为空的episode已可恢复地移入`logs/quarantine`后重采，不进入正式120 episode。v1 smoke
因旧step-0 trace bug只保留为诊断；正式plan为
`fixed_dbm_highspeed_expansion_20260830_v2.json`，SHA256为
`0e4280c91024195d98b5813ebef2f697946a40fc9d2a951e8dc7bda24e95daac`。

从全部600 context生成129-candidate proximal teacher，共77,400次center rollout。teacher在600/600
状态严格优于anchor，零基线违规；确定性J50聚合下降`3.1386%`，episode-bootstrap 95% CI
`[3.0027%,3.2974%]`。因此teacher生成并不是扩张后的阻塞。

预训练使用同一120-episode池内的stratum-balanced、episode-grouped 5-fold x 3-seed。每个stratum的
repeat按`(speed + scenario + repeat) mod 5`分折；每run为360 fit / 120 internal-selection / 120 OOF。
为避免大数据得到额外优化预算，Actor/Critic epochs由旧的500/240缩放为125/60，使optimizer step数
近似不变。Actor OOF teacher-gain recovery中位`0.573`（范围`-0.215--0.937`），主门通过但tail门失败；
fold 1约`0.027--0.211`、fold 2约`-0.215--0.040`，仍有系统性难episode组。Twin Critic OOF log-cost
Pearson中位`0.984`，两个初始化门均通过。独立CPU重载15个checkpoint通过；CPU/GPU center差最大
`8.64e-7`，对应约5e5 cost尺度下最大`0.75`绝对差，均低于预登记的`2e-6`相对量级容差，episode
leakage为0。

在该预训练上继续相同20轮Actor-visited OAC。15 run selected round中位20（仅两个停在18/19）。OOF
teacher-gain recovery中位由`0.573`升至`1.031`，范围由`[-0.215,0.937]`收窄/抬升为
`[0.363,1.492]`；15/15 run的OOF mean均优于各自预训练Actor，每fold 3/3 seed均改善。胜或等于warm
比例中位由`0.683`升至`0.808`，warm-relative mean gain中位为`+6023`。与旧30-episode OAC相比，
recovery中位`0.941 -> 1.031`、最差run`0.269 -> 0.363`，说明扩大独立状态覆盖对主体和最差fold均有
实质收益。

但扩大数据没有解除部署阻塞：warm-relative P05中位仍为`-10074`、最差run P05为`-24125`，每run
仍有约`10.8%--31.7%`状态输给warm；fresh OOF Critic局部sign accuracy中位`0.691<0.70`。所以当前
registered gate仍为`2/4`通过。结论是“数据扩张值得且改善主体”，不是“靠继续堆同分布数据即可解决
尾部”。selected Actor仍只能作为warm+Actor two-center proposal；formal validation/test、单中心部署
和闭环继续冻结。

```text
outputs/mppi_proposal/highspeed_collection_summary_expansion_20260830_v1/
outputs/mppi_proposal/highspeed_initial_dbm_replay_expansion_20260830_v1/
outputs/mppi_proposal/highspeed_proximal_teacher_expansion_20260830_v1/
outputs/mppi_proposal/highspeed_actor_twin_critic_pretrain_expansion_e4_20260830_v2/
outputs/mppi_proposal/highspeed_actor_visited_oac_expansion_e4_20260830_v1/
```

### 11.130 高速strong-search上界与Actor初始化价值（2026-08-30）

为量化§11.129 selected Actor离可实现低cost中心的真实距离，从600-state train-only池中选取120个独立
状态：每个`5 speeds x 6 scenarios`格子的4个独立episode各取`control_step=0`，不使用相邻连续帧。
每状态以warm、原proximal teacher和三个OOF OAC Actor为5个起点，分别执行6轮严格best-improvement
全秩antithetic pattern search；半径为`1.0/0.70/0.50/0.35/0.20/0.10 sigma`，方向基在Hadamard及
两个固定Givens旋转基间切换。五条链共享DBM评价缓存，每状态恰好965个唯一中心；warm恒保留。全程只
调用确定性DBM J50黑盒cost，不使用DBM梯度，不读取formal validation/test。

115,800次正式搜索rollout及115,800次独立validator重放全部完成，最大cost误差为0、baseline violation
为0。核心结果为：

- warm mean/median J为`192540/169507`；原teacher为`183356/162876`，相对warm仅改善`4.77%`；
- strong oracle mean/median J为`154546/135717`，相对warm改善`19.73%`，120/120严格改善；原teacher
  只恢复strong-oracle headroom的`24.2%`，strong gain是teacher gain的`4.14x`；
- strong改善在40/55/70/85/100 km/h五层分别为`28.3%/22.7%/21.0%/19.2%/17.3%`，不是低速层
  独有现象；
- 三个OAC Actor单次中心相对warm改善`7.98%--8.32%`，恢复strong headroom `40.4%--42.2%`，明显
  低于预登记的80%停止门；该120-state step-0机制子集上几乎都胜warm，但不能外推到§11.129全部
  600帧的later-step尾部；
- 从warm起点完成同预算搜索只恢复strong headroom `77.6%`；从原teacher为`95.5%`；从三个Actor
  起点分别为`96.5%--97.0%`。最终winning chain计数为warm 0、teacher 14、Actor三个seed分别
  `32/35/39`。所以Actor确实提供了更好的搜索初始化/basin，而不是无效输出；
- 但Actor到其自身搜索终点的移动仍很大，标准化RMS中位为`2.13--2.23 sigma`。剩余问题不能称作
  “极小精度误差”，而是一段明显的多轮refinement路径。

裁决：`ACTOR_GOOD_SEARCH_INITIALIZATION_MULTI_STEP_REFINEMENT_LIMIT`。当前129-candidate teacher并非
有效上界，因此“Actor已接近teacher”不能再被解释为网络接近最优。下一teacher不直接使用五链全局
argmin做一次性MSE；应保留每个OOF Actor自身的逐轮search path，把有界中间改进作为配对标签，做
Actor-visited迭代蒸馏。这样既利用Actor已找到的好basin，也避免跨basin终点和约2.2-sigma大跳造成
均值化。正式扩大到600状态前，先在当前120状态上比较一步/两步中间标签的train-fit与episode OOF
传递；two-center warm floor保持不变。

```text
scripts/model_verify/run_highspeed_strong_search_oracle.py
scripts/model_verify/validate_highspeed_strong_search_oracle.py
scripts/model_verify/analyze_highspeed_strong_search_oracle.py
outputs/mppi_proposal/highspeed_strong_search_oracle_20260830_v2/
  summary.json
  oracle.npz
  validator_report.json
  analysis.json
```

记录性版本说明：v1的数值与全部候选独立重放均正确，但只保存每条链的起点/终点；v2在完全相同搜索
合同和逐状态结果下补存`5 starts x 7 incumbents`的逐轮center/cost，且重新对115,800候选独立重放
误差为0。后续迭代标签只允许引用v2，v1保留为上界诊断。

### 11.131 高速J50的along/cross分解与耦合验证（2026-08-30）

为区分“高速J数值变大只是尺度问题”与“高速下轨迹误差真实放大”，在§11.130同一120个train-only
独立step-0状态上重放warm、proximal teacher、三个OOF OAC Actor和strong oracle。position仍采用原始
固定时间索引二维误差，不做最近点/Frenet重配准；仅使用存储reference yaw将每步误差正交分解为
`along`和`cross`，因此
`J_position = 5*sum(e_along^2) + 5*sum(e_cross^2)`与原始二维position完全等价，不改变任务定义。

独立validator再次重放全部`120 x 6`个center，各代价项最大误差为0；along+cross重组position的
float32最大绝对误差为`0.0625`（约4e-7相对量级），重组总J50与§11.130存储cost误差为0。验证结果：

- warm mean/median J为`192540/169507`，其中position占聚合J的`87.12%`，显式vx占`12.47%`；
  position内部along/cross分别为`48.61%/51.39%`，不是单一纵向时序误差或单一横向误差主导；
- warm的along/cross RMS中位为`17.31/16.86 m`。随名义速度40→100 km/h，along RMS中位
  `9.04→24.68 m`、cross RMS中位`10.05→27.40 m`，说明J增大对应真实轨迹误差随高速预测距离
  放大，而非仅由未归一化数值造成；
- proximal teacher的warm-relative总收益中，position/vx贡献约`35.46%/64.16%`；三个OAC Actor更
  极端，position贡献约`25.97%--26.06%`、vx约`73.81%--73.88%`。当前单次Actor改善主要来自纵向
  速度，而不是充分修正空间轨迹；
- strong oracle的总收益中，position贡献`64.35%`、vx贡献`35.44%`；position收益又几乎等分为
  along `31.90%`和cross `32.46%`。所以strong-search新增headroom确实同时来自沿程和横向轨迹改善，
  不是通过牺牲一侧换取另一侧；
- strong oracle把along/cross RMS中位降至`15.90/15.43 m`。40--100 km/h各速度层的总J均改善，
  但100 km/h层仍由较大的along/cross误差共同构成，后续必须继续分层报告。

裁决：`HIGHSPEED_J_SCALE_PHYSICAL_ALONG_CROSS_COUPLING_CONFIRMED`。高速J大是固定2.5 s时域内
速度、航向、侧偏经DBM积累到位置后再平方求和的合理结果；同时当前固定时间二维position将沿程同步
与横向跟踪共同纳入目标。输入已按fit split归一化，Critic目标已使用标准化`log1p(J)`，不应把修改
正式J或按速度除以`v^2`当作数值修复。若未来要改变along/cross权重，属于新任务定义，必须另做
counterfactual。当前主线仍使用原始J50，并在逐轮search-path蒸馏中强制报告along/cross/vx三项收益，
确认Actor是否开始恢复strong search所发现的空间轨迹headroom。

```text
scripts/model_verify/analyze_highspeed_along_cross_cost.py
scripts/model_verify/validate_highspeed_along_cross_cost.py
outputs/mppi_proposal/highspeed_along_cross_cost_20260830_v1/
  summary.json
  decomposition.npz
  validator_report.json
```

### 11.132 高速Actor-search中间路径蒸馏pilot（2026-08-30）

按§11.130预登记分支，使用strong-search v2中三个OOF OAC Actor各自的incumbent path，不混用跨起点
全局argmin。120个独立train-only step-0物理状态各含3条Actor path，共360个显式anchor条件样本。
逐轮审计显示：第一轮为约`1.0 sigma RMS`移动并恢复strong headroom `40.70%`；第二轮新增约
`0.7 sigma RMS`并累计恢复`63.34%`。第二轮旋转方向的单坐标最大为`0.981 sigma`，因此stage-2
模型采用1.0-sigma分量边界，避免裁剪合法标签。

使用episode-grouped 5-fold x 3-training-seed；每run为72 fit / 24 inner-selection / 24 OOF物理状态，
同一状态的三条Actor path严格同折。Actor为semantic-clean、显式absolute-anchor的确定性残差网络；
只用action-transition MSE训练，DBM只做真实J50评价，不使用Critic或DBM梯度。比较：

1. `one_step`：Actor start→path1，1.0-sigma边界；
2. `direct_two_step`：Actor start→path2，一个网络一次前向，2.0-sigma边界；
3. `stage2`：真实path1→path2；
4. `cascade_two_step`：学习到的stage1输出再输入独立stage2网络，两次前向。

360个OOF样本的聚合结果：

| 中心 | mean J | 相对warm降J | 对本档search标签收益恢复 | strong headroom恢复 |
|---|---:|---:|---:|---:|
| OAC start | 176936 | 8.10% | -- | -- |
| path1标签 | 167823 | 12.84% | 100% | 40.70% |
| 学习one-step | 168030--168481 | 12.50%--12.73% | 92.78%--97.74% | 37.76%--39.78% |
| path2标签 | 162754 | 15.47% | 100% | 63.34% |
| 学习direct-two-step | 163447--163641 | 15.01%--15.11% | 93.75%--95.11% | 59.38%--60.25% |
| 学习cascade-two-step | 163021--163422 | 15.12%--15.33% | 95.28%--98.12% | 60.36%--62.15% |

episode bootstrap的target-gain recovery 95% CI在三个seed均远高于0.5门：one-step为
`[0.905,0.988]`范围，direct-two-step为`[0.917,0.968]`，cascade为`[0.937,0.992]`。三个arm在
所有seed、全部360个OOF样本都严格优于各自Actor start；P05 gain为正，回归率为0。40/55/70/85/
100 km/h五层均保持正P05和零回归；100 km/h的direct-two-step标签收益恢复为`88.8%--92.7%`，
cascade为`95.3%--97.8%`。

§11.131提出的空间轨迹门也通过。学习one-step的收益约`71%`来自position、`28%`来自vx；两步模型
约`79%`来自position、`19%--21%`来自vx，along/cross贡献近似对称。它们不再复现原OAC单次更新
“约74%收益来自vx”的偏置，而是确实恢复了strong search发现的空间轨迹headroom。

45个checkpoint全部独立重载；fold episode集合无交集，中心、DBM总J和along/cross/yaw/vx/rate分项
重放最大误差均为0。裁决为`HIGHSPEED_INTERMEDIATE_PATH_LABELS_OOF_TRANSFER_PASS`。这证明在当前
120-state独立step-0子集上，Actor-conditioned有界搜索路径明显比远距离全局argmin更可蒸馏；网络并非
无法表达进一步改善。direct-two-step已接近cascade且只需一次前向，但cascade在100 km/h更稳，因此
下一轮扩展到全部600个train-only状态时保留二者配对，暂不提前固定部署形式。

边界：本节没有覆盖同episode的later control steps、formal validation/test或闭环；零回归结论不能
外推到§11.129全部600帧。下一步为600-state、前两轮相同search path采集，并在原5-fold合同下复训
direct-two-step/cascade；two-center warm floor继续保留。

```text
scripts/model_verify/train_highspeed_iterative_path_distillation.py
scripts/model_verify/validate_highspeed_iterative_path_distillation.py
scripts/model_verify/analyze_highspeed_iterative_path_distillation.py
outputs/mppi_proposal/highspeed_iterative_path_distillation_20260830_v1/
  summary.json
  analysis.json
  oof_predictions.npz
  validator_report.json
  fold_*/seed_*/*.pt
```

### 11.133 全600-context两轮路径扩展与蒸馏裁决（2026-08-30）

§11.132通过后，按预登记合同对§11.129全部600个train-only context、三个OOF OAC Actor起点采集前两轮
确定性search path。每状态固定195个唯一候选，合计117,000次正式DBM rollout；全部候选独立重放
误差为0，path center查找误差为0，单调性/基线违规均为0。相比完整6轮965-candidate strong search，
本轮只支付约1/5预算。

全量标签包含真实later-step/平坦区：第一轮严格移动率`96.67%`，累计两轮严格改善率`99.11%`；未改善
样本保持incumbent，不强制移动。1800条Actor-conditioned路径的聚合cost为：

| 中心 | mean J | median J | 相对warm降J |
|---|---:|---:|---:|
| warm | 191104 | 164742 | 0% |
| OAC start | 185287 | 162426 | 3.04% |
| path1标签 | 177650 | 152993 | 7.04% |
| path2标签 | 174907 | 149419 | 8.48% |

在原120-episode grouped 5-fold x 3-seed合同下复训；每run为360 fit / 120 inner-selection / 120 OOF
物理context，同状态三条path严格同fold。独立validator重载45个checkpoint，episode leakage、hash、中心、
总J和全部cost component误差均为0。主要OOF结果：

- one-step保留path1收益`90.12%--90.32%`，相对warm降J约`6.65%`；总体P05 gain为正，但100 km/h
  P05仅1/3 seed为正，故one-step高速tail门失败；
- direct-two-step保留path2收益`94.79%--95.23%`，相对warm降J`8.19%--8.22%`，整体P05 gain
  `673--710`，100 km/h P05 `458--517`，全部三seed过门；
- cascade-two-step保留`93.06%--93.97%`，相对warm降J`8.10%--8.15%`，整体P05 `583--671`，
  100 km/h P05 `412--481`，同样三seed过门，但没有超过一次前向direct；
- direct/cascade分别约`1.94%--2.78%`和`2.44%--2.83%`样本输给各自OAC start，worst gain约
  `-25k--34k`与`-34k--54k`；因此raw Actor不是逐状态安全保证。相对warm的two-center direct guard
  由构造将回归归零，并把聚合降J略抬至约`8.35%--8.40%`；
- direct的收益`85.69%--87.01%`来自position，cascade为`87.89%--90.10%`，along/cross仍近似等分；
  未退回原OAC约26% position收益，§11.131空间轨迹门通过；
- direct在40/55/70/85/100 km/h的标签收益恢复约为`97.7--98.7% / 94.7--95.5% /
  98.6--100.1% / 93.6--95.0% / 91.6--92.5%`，全速度层稳定。

裁决：`HIGHSPEED_TWO_ROUND_DIRECT_DISTILLATION_FULL600_PASS`。120-state强正结果不是step-0偶然；当前
Actor的主要缺口确实可由“自身basin内、两轮有界search endpoint”监督解决。由于direct-two-step在
全量上比cascade更好且只需一次前向，正式后续以direct为主，cascade降为诊断备份。不能宣称已达到
strong oracle：本轮teacher只含前两轮，且raw模型仍有约2%--3%回归和明显worst；two-center guard仍为
承重结构。

下一步不再扩大网络或回到Critic梯度，而是做部署合同准备：冻结每fold direct-two-step checkpoint，
在未消费formal前先完成train-only later-step tail manifest与two-center MPPI候选集集成单测；之后才申请
短闭环A/B。formal validation/test仍封存。

> **路线修正（2026-08-30，覆盖本段“下一步不再回到Critic梯度”的表述，不覆盖§11.132--11.133的
> 数据与数值结论）：** direct-two-step蒸馏是Actor可表达性和监督目标质量的机制诊断，不是最终
> Query兼容训练路线。最终合同要求动力学后端只提供候选轨迹与标量cost；切换到Query（尤其ONNX）后
> 不依赖动力学解析梯度，因此持续更新的Twin Critic仍是Actor训练梯度的必要来源。蒸馏checkpoint只作
> 容量参考、可选预训练和离线oracle，不替代持续Actor--Critic。正式下一步改为§11.134的
> search-informed Replay OAC等预算A/B；在该A/B通过前不申请闭环。

```text
scripts/model_verify/generate_highspeed_two_round_actor_paths.py
scripts/model_verify/validate_highspeed_two_round_actor_paths.py
scripts/model_verify/train_highspeed_iterative_path_distillation_expansion.py
scripts/model_verify/validate_highspeed_iterative_path_distillation_expansion.py
outputs/mppi_proposal/highspeed_two_round_actor_paths_20260830_v1/
outputs/mppi_proposal/highspeed_iterative_path_distillation_expansion_20260830_v1/
```

### 11.134 Query兼容主线修正与search-informed Replay OAC计划（2026-08-30）

#### 11.134.1 最终算法合同

最终目标不是让Actor依赖DBM解析梯度，也不是把离线search endpoint蒸馏后直接视为算法完成。部署模型
以后需要从DBM切换到Query/PyTorch或Query/ONNX；为保持同一训练合同，动力学后端统一按黑盒使用：

```text
Actor产生absolute center
  -> 在该center附近生成候选action
  -> DBM或Query rollout并返回真实标量J50
  -> 全部好/坏候选写入Replay
  -> 持续更新Twin Critic Q(s,a)
  -> Actor只通过Critic的dQ/da更新
  -> 新Actor再次访问环境并补充Replay
```

其中DBM解析梯度只允许用于机制审计和上限对照，禁止进入正式Actor更新；Query后端无需暴露解析梯度。
离线BC/search蒸馏允许作为Actor初始化，但不能替代Actor-visited Replay、持续Critic更新和后续Actor更新。

#### 11.134.2 §11.132--11.133结果的正确角色

两轮路径蒸馏仍是有效且重要的正结果，但其结论需要收窄：

1. direct-two-step能保留两轮search标签收益约`95%`，证明当前Actor函数类能够表达从OAC center到
   更好局部center的映射；“统一扩大Actor网络”不是当前第一优先级。
2. search endpoint把相对warm改善从OAC start的`3.04%`提高到`8.48%`，蒸馏Actor达到约
   `8.2%`，说明现有OAC没有充分利用Actor附近真实存在的改进信息。
3. 该结果没有训练或评价Critic，不能证明Query兼容OAC已经解决；也不能据此把direct-two-step定义为
   最终部署主Actor。它现在是Actor容量参考和“什么样的局部数据值得进入Replay”的证据。
4. raw蒸馏Actor仍有约`2%--3%`起点相对回归；two-center guard仍是最终候选集的承重结构，但guard
   不能替代训练阶段对Critic局部排序和Actor主体收益的验证。

#### 11.134.3 低速OAC经验迁移边界

低速`1.2--2.8 m/s`结果不能直接外推高速数值，但已经关闭或确定了以下算法组织问题：

- 冻结Critic或只取一次Critic梯度永久关闭；新Actor动作必须获得真实DBM/Query cost、写Replay并继续
  更新Critic。
- K扫描的效率拐点为`K=16`，`K=32`只增加约`0.81pp` recovery，`K=20`严格1:1无优势；因此不能靠
  无限增加同一Critic上的Actor microstep解决问题。
- 等总预算的2倍Replay刷新使recovery下降`0.51pp`；简单缩短同分布反馈延迟不是主瓶颈，应该改变
  Actor-visited数据的信息含量。
- matched DBM-vs-Critic梯度源只提高`1.88pp` recovery，说明Critic误差有代价但不是全部差距；禁止
  因此回到盲目扩Critic或loss扫描。
- deterministic-center真实DBM目标只再提高约`1.02pp`，说明随机Actor目标错配也只是次因。
- guided-2x bank只在约`5%`状态取得高价值改善，证明cost response可指导更有信息的探索，但bank
  best-of-N改善不能自动等价为Critic或Actor改善。
- 真实DBM逐状态动作步可以零回归，而共享Actor参数步约`40%`状态回归；共享参数梯度冲突是真实风险。
  若Critic已学会search局部排序而Actor仍不前进，后续应处理共享Actor更新，而不是继续扩Critic。

这些经验决定高速下一步应同时保留“持续Critic”与“提高Replay信息量”，而不是在Critic和search之间
二选一。

#### 11.134.4 下一步唯一主实验：`HIGHSPEED_SEARCH_INFORMED_REPLAY_OAC_AB`

从§11.129同一组episode-grouped Actor+Twin-Critic **预训练checkpoint**开始，执行严格等rollout、等
Critic更新、等Actor更新的train-only A/B；不从已经走过不同Replay轨迹的terminal OAC checkpoint恢复，
避免optimizer/Replay历史混入归因。两臂动力学均只返回标量J50，Actor均禁止使用DBM解析梯度和
search endpoint MSE：

- `R0 non-recentered`：每个visited context保留当前Actor center；第一组候选在center附近评价，第二组
  使用匹配半径和方向但仍围绕原center评价。
- `R1 search-informed`：第一组候选与R0逐项配对；根据第一组真实cost选出incumbent，第二组候选围绕
  incumbent重定位。所有候选，包括变差动作、stay和best，全部写入Replay；不得只克隆best。

建议固定每context共`65`个唯一中心（`1 + 32 + 32`），方向、半径、随机流和候选预算逐项配对，唯一
变量是第二组是否重定位。训练分两段但不冻结最终Critic：

首轮A/B保持现有高速OAC的`20 Critic : 1 Actor`、20 rounds，不立即照搬低速K16；否则会同时改变
Replay几何和Actor更新预算，无法归因。低速K16只作为R1通过后的独立更新预算候选。

每个fold只能把该fold的fit episode候选写入Replay，internal-selection只用于checkpoint选择，OOF
episode只能fresh评价、不得进入Critic吸收或Actor更新；同一物理episode的5个early context继续整体
同fold。

1. **Critic吸收段**：Actor暂时冻结，仅用两臂新Replay按相同update预算更新Twin Critic；检查Critic
   是否学到path候选的value、排序和局部方向。
2. **持续OAC段**：解除Actor冻结，保持Actor-visited采集、Replay补充、Twin Critic和Actor持续交错
   更新；Actor梯度只能来自Twin Critic。

主报告必须包含：

- Critic：fresh同状态候选Pearson/ranking、best-candidate regret、局部sign accuracy、twin
  disagreement；保留现有`sign accuracy >= 0.70`记录门。
- Actor：确定性center相对warm的聚合降J、胜warm比例、median/P05/worst、对strong-search headroom
  recovery，以及40/55/70/85/100 km/h分层。
- 机制：position/along/cross/vx收益分解，确认改进不再只集中在vx。
- 安全边界：raw Actor与`min(J_warm,J_actor)` two-center结果分开报告；formal validation/test、MPPI
  wrapper和闭环仍封存。

判决树预先固定：

1. `R1`先提高Critic path排序且持续OAC Actor优于`R0`：search-informed Replay成立，随后才扩大轮数并
   单独比较当前`20:1`与低速经验的K16更新合同，再切换Query/PyTorch与Query/ONNX做相同cost接口复核。
2. Critic排序明显提高但Actor不改善：阻塞落到共享Actor参数优化；恢复低速审计提出的PCGrad/CAGrad
   或speed/regime head短pilot，禁止继续扩Critic。
3. Critic连path排序也未提高：使用同一search bank检查value/ranking/pair-delta监督配比或Critic容量，
   Actor保持冻结；不得用DBM真梯度绕过失败。
4. R1只改善best-of-bank、不改善Critic/OAC：search仅保留离线teacher/候选生成用途，不升级为主训练
   Replay。

因此当前正式状态为：`SEARCH_DISTILLATION_ACTOR_CAPACITY_PASS_OAC_CRITIC_TRANSFER_PENDING`。

### 11.135 Search-informed Replay OAC等预算A/B：候选信息强增，Critic传导弱（2026-08-30）

已按§11.134完成正式`5 fold x 3 seed` train-only A/B。两臂均从同一高速Actor+Twin-Critic预训练
checkpoint开始，先做20次/每Twin的Critic-only吸收，再运行20轮`20 Critic : 1 Actor`；每状态每轮
固定`1 + 32 + 32 = 65`个中心，第一半径由`1.0sigma`线性降至`0.2sigma`，第二半径为其0.7倍。
`R0`第二组仍围绕Actor center，`R1`围绕第一组真实cost选出的incumbent重定位。两臂所有好/坏动作均
写Replay，不使用DBM解析梯度、不使用endpoint BC、不打开formal/test或闭环。

工程与复算均通过：15组配对run的源Replay、吸收阶段第一组33候选和首轮在线第一组33候选逐位相同，
episode leakage为0；全部30个checkpoint独立重载，保存action的fresh DBM replay、Critic指标与summary
一致。CUDA checkpoint重载中心与训练期保存中心最大差为`1.78e-5/4.97e-5`（R0/R1），同一次重载重复
前向误差为0；保存action的probe replay误差为0、抽样Replay cost最大误差`0.09375`。高速度J对微小
center差敏感，因此validator把checkpoint重载容差与保存action的严格DBM replay门分开记录。

最强结果是**候选信息量确实大幅增加**。对每run取20轮平均后，R1相对R0的best-of-bank gain配对增量
15/15为正：中位`+2145.59`、均值`+2201.54`；第二阶段自身增量中位`+2161.81`。所以本轮失败不能解释
为“重定位没有找到更好的动作”。

但Critic吸收很弱。最终Actor处fresh两轮`1.0/0.7sigma` 65-candidate bank的配对变化为：

| 指标 | R0中位 | R1中位 | 配对delta中位 |
|---|---:|---:|---:|
| centered log-cost Pearson | 0.6351 | 0.6393 | +0.0046 |
| center-relative sign accuracy | 0.7935 | 0.7918 | +0.0003 |
| bank gain recovery | 0.7775 | 0.7724 | +0.0087（配对均值-0.0086） |

原0.05sigma OOF local probe同样没有稳定方向收益：centered Pearson配对中位`+0.0241`，但sign accuracy
为`-0.0026`，bank recovery仅`+0.0062`。因此不能声明Critic已经学会R1新增的search response。

Actor只有小幅收益：teacher-gain recovery从中位`1.0267`到`1.0322`，配对增量中位`+1.07pp`；
warm-relative mean gain配对中位`+64.36`，P05配对中位`+124.7`，胜warm比例配对中位`+0.83pp`。
这些变化方向总体为正，但远小于每轮候选信息`+2146`，且两臂全部15个run的warm-relative P05仍为负，
不构成部署或tail资格。

正式qualification为
`SEARCH_INFORMED_BANK_STRONG_CRITIC_TRANSFER_WEAK_ACTOR_SMALL_GAIN`。它回答了§11.134的分支：

1. search/recentering本身有效，不能降级为无价值探索；
2. 现有`log1p(J)` value + same-state ranking、当前Critic更新预算没有把强候选差异转成明显fresh排序增量；
3. 直接增加Actor更新、做BC或使用DBM真梯度都会绕过当前阻塞，暂不授权；
4. 下一步冻结Actor，复用R1已落盘Replay和fresh path bank做Critic-only预算曲线（零新rollout），先区分
   “420次在线Critic更新不足”与“当前loss/表示吸收不了”；若预算可恢复，再恢复持续OAC；若预算无效，
   才在同一bank上做pair-delta/ranking目标单变量A/B。

```text
scripts/model_verify/train_highspeed_actor_visited_oac.py
scripts/model_verify/validate_highspeed_actor_visited_oac.py
scripts/model_verify/analyze_highspeed_search_replay_oac_ab.py
scripts/model_verify/evaluate_highspeed_search_replay_critic_ab.py
outputs/mppi_proposal/highspeed_search_replay_oac_nonrecentered_20260830_v1/
outputs/mppi_proposal/highspeed_search_replay_oac_recentered_20260830_v1/
outputs/mppi_proposal/highspeed_search_replay_critic_ab_20260830_v1/
outputs/mppi_proposal/highspeed_search_replay_oac_ab_20260830_v2/
```

### 11.136 Search-informed Replay的Critic预算曲线：数据可学，原在线预算不足（2026-08-30）

按§11.135预登记分支，Actor保持冻结，直接复用R1最终Replay与fresh两阶段65-candidate OOF bank；未新增
任何DBM rollout，也未使用DBM解析梯度。每个fold/seed从R1保存的Twin Critic及其optimizer状态继续
训练，记录每Twin额外`0/400/1600`次update。全部`5 fold x 3 seed`完成；OOF bank只用于事后评价，
没有参与checkpoint选择。

| fresh OOF指标 | 0 update | +400 | +1600 |
|---|---:|---:|---:|
| 两阶段path centered Pearson中位 | 0.639 | 0.720 | 0.786 |
| 两阶段path sign accuracy中位 | 0.792 | 0.817 | 0.865 |
| 两阶段path bank-gain recovery中位 | 0.772 | 0.830 | 0.920 |
| 两阶段path bank-gain recovery P05 | 0.528 | 0.783 | 0.848 |
| 0.05sigma local Pearson中位 | 0.366 | 0.432 | 0.525 |
| 0.05sigma local sign accuracy中位 | 0.713 | 0.725 | 0.761 |
| 0.05sigma local bank-gain recovery中位 | 0.506 | 0.551 | 0.671 |

从0到1600，真实两阶段path的Pearson/sign/recovery均为`15/15` run正改善；0.05sigma local的sign为
`15/15`正改善，Pearson与recovery为`14/15`正改善。唯一local反例是fold4/seed0，但其部署相关path
Pearson/sign/recovery仍分别提高`+0.008/+0.027/+0.038`，没有出现path退化。中间400点允许个别run
非单调，最终1600点的全path改善才是主判据。

这关闭了“当前Critic loss/表示完全吸收不了search response”的解释，并确认§11.135看到的弱传导主要
来自更新预算与Replay增长不匹配：吸收段先加入65个候选，随后20轮每轮每状态再加入65个，Replay从
初始129候选/状态增长到1494个，而正式R1只有首段20次加每轮20次、合计420次Critic update/每Twin。新增高信息动作
存在，但Critic尚未充分遍历和拟合它们。不能把本结论扩大为“Critic已最终解决”：Actor在本实验中
冻结，尚未证明更成熟的Critic会经持续OAC转成Actor收益。

独立validator重载15个最终checkpoint，复算fresh path及0.05sigma local全部指标，最大误差为0；
episode leakage为0，source/checkpoint/replay hash均通过；formal/test、Query、wrapper与闭环均未打开。
资格更新为`SEARCH_REPLAY_CRITIC_ABSORPTION_BUDGET_PASS_ACTOR_TRANSFER_PENDING`。

下一步不直接把Actor microstep从1改成低速K16。当前证据首先要求验证Critic readiness，而K16会在
同一个尚未充分吸收的Critic上增加Actor步数，混淆归因并可能放大外推。应从同一R1最终Actor出发，
配对比较原R1 Critic与`+1600` Critic，保持后续search-informed采集、Replay写入、Critic持续更新和
`1 Actor update/round`完全相同；只有成熟Critic能稳定增加Actor收益后，再单独调在线Critic:Actor
节奏或Actor microstep预算。

```text
scripts/model_verify/train_highspeed_search_replay_critic_budget.py
scripts/model_verify/validate_highspeed_search_replay_critic_budget.py
outputs/mppi_proposal/highspeed_search_replay_critic_budget_20260830_v1/
  summary.json
  validator_report.json
  critic_fold*_seed*.pt
```

### 11.137 Critic readiness到Actor传导的配对A/B：Critic显著更准，但Actor未获得额外收益（2026-08-31）

为验证§11.136的Critic恢复能否转成策略收益，完成`5 fold x 3 seed`配对持续OAC。两臂从同一个R1最终
Actor、Actor optimizer与1494-candidate Replay开始；唯一初始差异是原R1 Twin Critic，或冻结Actor
额外吸收1600 update后的Twin Critic。随后两臂均继续10轮search-recentered 65-candidate采集，所有
候选写Replay，每轮保持`20 Critic : 1 Actor`，半径`0.20 -> 0.05sigma`。没有冻结Critic推Actor，
没有DBM解析梯度、BC、formal/test或闭环。

成熟Critic的优势在Actor移动后仍保持：fresh两阶段path sign配对提高中位`+0.0681`，bank-recovery
提高`+0.1223`，两项均`15/15` run为正。因此§11.136不是只对旧Actor中心成立的静态假象。

但是Actor没有获得对应传导：

| OOF selected指标 | 原R1 Critic | +1600 Critic | 配对delta中位 |
|---|---:|---:|---:|
| proximal-teacher gain recovery中位 | 1.267 | 1.234 | -0.0024 |
| warm mean gain中位 | 7404 | 7208 | -14.6 |
| 胜warm比例中位 | 0.833 | 0.808 | 0.000 |
| warm P05 gain中位 | -9297 | -10117 | -196 |

`+1600`臂仅`5/15` run的recovery/mean gain优于原臂，配对recovery均值为`-2.51pp`；两臂全部run的
warm-relative P05仍为负。这里recovery以当前proximal teacher为分母，允许大于1，不代表超过strong
search或理论最优。重要的正面结果是继续10轮持续OAC本身对两臂都`15/15`优于各自相同起点，说明
在线循环仍可继续学习；负面结果是额外Critic精度不能自动提高共享Actor的学习效率。

更新步幅给出一个直接线索：原Critic臂每轮输出RMS步长均值为`0.01434sigma`，150次中30次触发
`0.02sigma`投影；成熟Critic臂均值仅`0.01311sigma`，只有7次触发投影。两臂raw-J权重ESS几乎相同
（约0.696/0.695），所以差异不是batch权重塌缩。成熟Critic产生的梯度排序更对，但当前Actor参数梯度
幅值/聚合路径更保守，且可能仍有跨状态冲突；现有结果还不能把两者分开。

独立validator重载30个checkpoint，复算Actor center、保存center的DBM cost、fresh path/local Critic
指标，全部最大误差为0；两臂初始Actor指标逐项相同，episode leakage为0，完整hash链通过。

裁决更新为`CRITIC_READINESS_PASS_ACTOR_TRANSFER_NO_GAIN_SHARED_ACTOR_UPDATE_BLOCKING`。后续停止继续堆
Critic-only预算，也不立即增加K16。下一项应是同起点、同batch、**匹配Actor输出RMS步长**的梯度几何
审计：比较原/成熟Critic的共享参数梯度方向、按速度/场景分组的负cosine与cancellation，以及等步长
真实DBM cost变化。若成熟Critic在等步长下仍无优势，进入Critic-compatible的PCGrad/CAGrad或
speed/regime head短A/B；若等步长恢复优势，则只需修Actor步长/optimizer校准。全过程仍需持续Critic，
不得退回冻结Critic部署。

```text
scripts/model_verify/train_highspeed_critic_readiness_actor_transfer.py
scripts/model_verify/validate_highspeed_critic_readiness_actor_transfer.py
outputs/mppi_proposal/highspeed_critic_readiness_actor_transfer_20260830_v1/
```

### 11.138 匹配Actor输出步长的Critic梯度对照：步长差异不是主因（2026-08-31）

按§11.137完成单步机制审计。每个fold/seed固定同一个R1 Actor、同一fit batch和相同gamma=1 raw-J加权
Actor目标，分别从原R1 Critic与`+1600`成熟Critic取得参数梯度；DBM不提供梯度，只在更新后返回真实
J50。两套方向分别通过raw-SGD和保存Adam moments两种路径校准到fit输出RMS
`0.005/0.01/0.02sigma`，再在完全隔离的OOF episode评价。

两套参数梯度不是单纯缩放关系：cosine中位`0.869`、P05 `0.632`，说明Critic吸收确实改变了共享Actor
方向。但固定输出步长后，成熟Critic没有稳定优势：

| 更新路径 | RMS | 原Critic OOF mean gain中位 | 成熟Critic | 配对delta中位 | 成熟臂胜run |
|---|---:|---:|---:|---:|---:|
| raw-SGD | 0.005 | 35.1 | 32.8 | -0.95 | 7/15 |
| raw-SGD | 0.010 | 69.6 | 67.5 | -0.13 | 7/15 |
| raw-SGD | 0.020 | 122.6 | 132.9 | +5.63 | 8/15 |
| saved-Adam | 0.005 | 48.2 | 55.2 | +1.18 | 10/15 |
| saved-Adam | 0.010 | 119.3 | 112.8 | +1.66 | 8/15 |
| saved-Adam | 0.020 | 194.2 | 186.8 | -1.34 | 7/15 |

这些delta相对run间方差很小且不随半径/optimizer一致，不能声明成熟Critic方向更好。tail同样没有一致
改善：例如saved-Adam `0.02sigma`的P05中位为原臂`-109.6`、成熟臂`-152.5`；回归比例中位则为
`0.167/0.158`，一好一坏。相反，两套方向本身大多是有效mean下降方向：saved-Adam `0.02sigma`两臂
均`15/15` run的OOF mean gain为正，且收益随`0.005 -> 0.02sigma`明显上升。

裁决：`MATCHED_STEP_NO_CRITIC_ADVANTAGE_STEP_SIZE_NOT_PRIMARY_K_SCAN_ALLOWED`。§11.137观察到的成熟Critic
实际步长较小不能解释其传导无优势；更精确的bank value/ranking没有同步形成更好的共享Actor action
gradient。继续堆Critic-only update降级。另一方面，Actor单步在当前`0.02sigma`边界仍未出现mean
饱和，故K扫描现在有依据，但它检验的是Actor优化预算，不是Critic修复。

K16不得按“16次各自最多0.02sigma”直接放开，否则单轮累计可远超本审计范围并混入trust扩张。下一
实验应先固定每轮累计输出RMS上限，做`K=1/4/8/16`等rollout、等Critic update的持续OAC配对；每个K
内部允许多个microstep，但更新后整体投影到相同累计trust。若K在相同累计移动下改善，说明参数空间
迭代/Adam求解不足；若无改善，再做speed-group PCGrad/CAGrad。之后才能单独放大累计trust边界。

独立validator对180组OOF动作完成DBM重放，candidate/base cost最大误差均为0，episode leakage为0，
source/evaluation hash与mean/P05复算全部通过；formal/test、Query、wrapper和闭环仍未打开。

```text
scripts/model_verify/analyze_highspeed_matched_actor_step_critic_ab.py
scripts/model_verify/validate_highspeed_matched_actor_step_critic_ab.py
outputs/mppi_proposal/highspeed_matched_actor_step_critic_ab_20260831_v1/
```

### 11.139 固定累计输出trust的Actor K扫描：K16仅有边际主体收益，无稳定部署tail收益（2026-08-31）

按§11.138的预登记合同完成两阶段实验。先在fold0三seed扫描`K=1/4/8/16`；随后只保留端点
`K=1/16`，扩展到完整`5 fold x 3 seed`。所有臂从同一个R1 Actor/Replay与吸收1600 update的Twin
Critic开始；每轮均采相同的search-recentered 65 candidates、每Twin做20次Critic update，唯一变量
是Actor microstep数。为排除步长混淆，内部microstep不逐步截断，整轮参数更新完成后统一校准，使
fit Actor输出相对轮前的RMS移动严格为`0.02 sigma`。

最初仅做“超过上限才截断”的试跑因K1实际只走`0.016--0.018 sigma`而作废并移入
`highspeed_actor_k_scan_fold0_20260831_invalid_upper_only/`，不得引用。有效fold0 pilot中所有K的
逐轮移动均为`0.019975--0.020038 sigma`；K4/K8没有超过K1，因此正式扩展只比较K1/K16。

完整15组结果如下（均为selected OOF；recovery分母是当前proximal teacher，允许大于1）：

| 指标 | K=1 | K=16 | 配对K16-K1中位 | K16较好run |
|---|---:|---:|---:|---:|
| teacher-gain recovery中位 | 1.2148 | 1.2331 | +0.00837 | 10/15 |
| 相对共同起点mean gain中位 | 975.83 | 1054.29 | +51.18 | 10/15 |
| 相对共同起点P05 gain中位 | -558.22 | -461.23 | +191.18 | 12/15 |
| warm-relative mean gain中位 | 7097.75 | 7204.41 | +51.18 | 10/15 |
| warm-relative P05 gain中位 | -9707.21 | -9700.13 | **配对-236.88** | 6/15 |
| 胜warm比例中位/跨run均值 | 0.8167 / 0.8467 | 0.8167 / 0.8433 | -0.00833 | 3/15 |

相对共同起点与相对warm的mean差值相同；两种P05不能相减换算，因为逐状态基线不同、分位数成员也会
变化。K16在“本轮更新是否改善原Actor”口径下有小幅主体和P05收益，但在最终单中心是否胜warm的部署
口径下没有稳定tail收益。selected round在15/15配对中完全相同（中位均为第5轮），不是checkpoint
选择时机造成的差异。

机制上，K16未投影的逐轮累计移动中位为`0.1954 sigma`（范围`0.1135--0.3066`），K1仅
`0.0140 sigma`（`0.0071--0.0215`）；投影后分别为`0.0200004/0.0199991 sigma`。因此K16的大量
额外优化主要在相同小位移球内改变参数路径，最终只换来约`0.84pp`的配对中位recovery，性价比很低，
不能把低速K16经验直接迁移为高速主配置。

裁决为`HIGHSPEED_FIXED_TRUST_K16_MARGINAL_BODY_GAIN_NO_TAIL_GAIN`：K1保留为后续成本对照，K16只
作为可选上界，不继续扩大到K32，也不据此放大trust。下一步先做按速度/场景分组的共享Actor梯度冲突
审计；若存在稳定负cosine/cancellation，再以同一K1、同一`0.02 sigma`trust比较PCGrad与
CAGrad/MGDA。该实验仍需持续更新Critic，不能退回冻结Critic。formal/test、Query、wrapper和闭环
继续封存。

独立validator重载30个checkpoint；Actor center、center DBM cost与全部0.05sigma local bank cost
最大误差均为0，15组初始指标跨K逐项一致，75+75次逐轮trust零违规，episode leakage为0，hash链
完整。

```text
scripts/model_verify/train_highspeed_actor_k_scan.py
scripts/model_verify/analyze_highspeed_actor_k_scan.py
scripts/model_verify/validate_highspeed_actor_k_scan.py
outputs/mppi_proposal/highspeed_actor_k_scan_fold0_20260831_v2/
outputs/mppi_proposal/highspeed_actor_k1_k16_scan_20260831_v1/
```

### 11.140 90轮预算与累计输出步长扫描：五轮结论被覆盖，`0.06 sigma`为机制领先但尚未授权（2026-08-31）

§11.139只观察了5个outer round，因此其中“停止增加microstep、不得放大trust”的裁决被本节长曲线
**明确覆盖**；原数字仅保留为短预算pilot。保持同一fold0、三seed、同一R1 Actor/Replay/Twin Critic、
每轮65个search-recentered候选与每Twin 20次Critic update，先把K1/K16延长到90轮，并在
`5/10/20/40/60/90`轮保存真实DBM OOF结果。每轮Actor更新仍按fit输出RMS精确校准到
`0.02 sigma`。

在`0.02 sigma`下，两臂到90轮都仍持续改善：K1/K16相对共同起点mean gain分别约
`10257/10283 J`，相对共同起点P05分别约`88/298 J`，胜warm比例均约`0.994`。K16在中段明显更快，
K16-K1配对mean差在第10/20/40/60轮分别为`+124/+465/+321/+401 J`（均3/3 seed为正），但第90轮
收敛到`+26 J`（2/3为正）。因此正确结论不是“K16无效”，而是：**轮数不足是五轮结果的主要混淆；
K16主要提高达到同一端点的速度，当前证据未证明其改变最终mean上限，tail端点则仍略优。** 三个seed的
selected round均为90，说明90轮也还不能宣称完全平台。

随后固定K16/90轮，只改变每轮精确输出步长，得到以下selected OOF结果（fold0三seed均值；
`recovery>1`只表示超过当前proximal teacher，不能解释为超过理论最优）：

| 每轮输出步长 | 相对共同起点mean gain | 相对共同起点P05 | warm-relative mean | warm-relative P05 | 胜warm比例 |
|---|---:|---:|---:|---:|---:|
| `0.02 sigma` | 10283 | 298 | 18646 | 3026 | 0.9944 |
| `0.04 sigma` | 12758 | 1031 | 21120 | 3996 | **1.0000** |
| `0.06 sigma` | **13702** | **1193** | **22064** | **4238** | 0.9972 |

从`0.02`增到`0.04`的selected mean增益约`+2475 J`，再增到`0.06`仍有约`+944 J`；因此小步长确实是
主要瓶颈之一，`0.06 sigma`是当前机制领先臂。它还不是部署默认值：本实验用“精确步长”而不是
“最大步长”，后期部分K16原始更新已自然降到目标以下，校准器会把它重新放大到完整`0.06 sigma`；
`0.06`也有一个seed出现1/120状态输warm，而`0.04`三seed全状态胜warm。当前仅fold0，formal/test、
Query、MPPI wrapper和闭环均未开启。

裁决为`HIGHSPEED_ROUND_AND_TRUST_BUDGET_MATTER_TRUST006_MECHANISM_LEAD`。下一步不做PCGrad，也不立即
把`0.06`写成部署常量；先把更新合同改为**cap-only `0.06 sigma`**，禁止把自然变小的步子放大，并给
Actor LR/输出步长加入后期衰减。该臂通过同一fold0 tail门后，再扩展完整fold；只有完整OOF与真实DBM
checkpoint gate通过，才讨论Query接口或two-center wrapper传导。

独立validator已验证三档来源hash、各来源独立DBM validator、全部曲线与selected汇总，判定
`HIGHSPEED_ACTOR_TRUST_CURVE_INDEPENDENT_SUMMARY_PASS`。

```text
scripts/model_verify/analyze_highspeed_actor_k_long_curve.py
scripts/model_verify/analyze_highspeed_actor_trust_curve.py
scripts/model_verify/validate_highspeed_actor_trust_curve.py
outputs/mppi_proposal/highspeed_actor_k1_k16_90round_fold0_20260831_v1/
outputs/mppi_proposal/highspeed_actor_k16_90round_trust004_fold0_20260831_v1/
outputs/mppi_proposal/highspeed_actor_k16_90round_trust006_fold0_20260831_v1/
outputs/mppi_proposal/highspeed_actor_k16_90round_trust_scan_fold0_20260831_v1/
```

### 11.141 cap-only与后段LR衰减完整OOF：主体/P05通过，严格worst仍需warm guard（2026-08-31）

按§11.140继续执行，不改变Critic、Replay、candidate bank或Actor microstep合同。更新规则从“每轮强制
走满`0.06 sigma`”改为**cap-only**：原始输出RMS超过`0.06 sigma`时才截断，小于上限时保持自然步长；
160轮实验的exploration radius固定在前90轮退火完毕，Actor LR从第120轮开始余弦衰减，由`2e-5`
降至第160轮的`5e-6`。先完成fold0单变量消融，再扩展为完整`5 fold x 3 seed`。

fold0三seed的配对结果如下（均为selected OOF，数值为三seed均值）：

| 合同 | selected轮中位 | 相对共同起点mean gain | 起点-relative P05 | warm-relative P05 | 胜warm比例 | latest-selected mean gain |
|---|---:|---:|---:|---:|---:|---:|
| 90轮 exact `0.06` | 90 | 13701.7 | 1193.1 | 4237.7 | 0.9972 | +30.8 |
| 90轮 cap-only `0.06` | 90 | 13689.1 | **1275.8** | 4208.0 | 0.9972 | 0.0 |
| 160轮 cap-only，常数LR | 145 | **14053.9** | 1331.2 | 4435.0 | 1.0000 | -72.2 |
| 160轮 cap-only，120轮后LR衰减 | 136 | 14028.8 | **1384.5** | **4593.9** | 1.0000 | **+39.8** |

90轮时cap-only与exact的mean仅差`-12.6 J`，而共同起点P05提高`+82.7 J`；strict warm worst由
`-2225`改善到`-1027 J`。这说明后期把自然小步重新放大不是主体收益来源，exact-forcing合同应废弃。
160轮常数LR比90轮cap-only再增加约`+365 J` mean，证明额外轮数仍有小幅价值；加入后段LR衰减只牺牲
约`25 J` mean，却把共同起点P05提高约`53 J`、warm P05提高约`159 J`，并显著减轻最后checkpoint相对
selected checkpoint的回落。因此最终机制领先合同定为：K16、160轮、cap-only `0.06 sigma`、120轮后
LR衰减到`0.25x`。

该合同随后扩展到完整5 fold x 3 seed。15个selected OOF run的fold/seed平衡汇总为：

| 指标 | 跨run均值 | 跨run中位 | 跨run最差 |
|---|---:|---:|---:|
| Actor center mean J | 169579.5 | 167609.7 | 181320.1（最高J） |
| 相对预训练Actor mean gain | **15543.9** | 14751.4 | 13358.8 |
| 相对预训练Actor P05 gain | **1915.7** | 1919.3 | **1032.4** |
| warm-relative mean gain | **21524.7** | 21033.5 | 20257.7 |
| warm-relative P05 gain | **3293.8** | 3321.9 | **1684.1** |
| 胜/平warm比例 | 0.9861 | 0.9833 | **0.9583** |
| warm回归比例 | 0.0139 | 0.0167 | 0.0417（最高比例） |

换成绝对mean J口径，预训练Actor约`185123.4`、warm约`191104.2`、当前proximal teacher约
`185106.1`，最终Actor约`169579.5`。因此本持续OAC Actor不只是略优warm，也在当前teacher口径上明显
更低；`teacher_gain_recovery`约3.59只表示当前Actor超过这个proximal teacher，不能解释为超过未观测的
理论全局最优。

完整OOF机制门全部通过：15/15 run的mean gain相对共同起点和warm均为正；15/15的warm P05为正；每个
run至少95.8%状态不差于warm。40/55/70/85/100 km/h五层的fold/seed平衡warm P05均为正，分别约
`2181/3371/4910/4301/6849 J`，高速层没有重现低预算实验中的系统性P05失败。

但这不是单中心无条件安全授权。逐run strict warm worst的最差值仍为`-34743.9 J`，平均约1.39%的
状态输warm；因此**two-center guard仍是承重结构**，不能因主体/P05通过而删除。selected轮分布为
`97--160`、中位`154`；最后轮相对selected的mean gain跨run均值为`-19.5 J`、最差`-218.2 J`，说明
160轮是预算上限而非强制部署最后轮，内部checkpoint选择仍需要保留。

cap-only合同确实生效：2400个round中32.5%的raw更新低于`0.06 sigma`并原样保留，其余67.5%只做
上限截断；不是把exact换了名字。完整来源的独立validator重载12个扩展checkpoint，Actor输出、center
DBM cost及local-bank DBM cost最大误差均为0；与fold0合并后的第二层validator复算全部来源hash、15-run
指标、cap activity和机制门，判定`HIGHSPEED_ACTOR_CAP_SCHEDULE_INDEPENDENT_SUMMARY_PASS`。

裁决为：`HIGHSPEED_CAP006_LRDECAY_FULL_FOLD_MECHANISM_GATE_PASS_GUARD_REQUIRED`。该结果授权把本合同作为
下一阶段的高速度Actor机制基线，并讨论Query黑盒接口或two-center wrapper传导；它**不授权**单中心
部署，也不等价于formal validation/test或闭环通过。formal/test、Query、wrapper和闭环在本节仍未运行。

```text
scripts/model_verify/train_highspeed_actor_k_scan.py
scripts/model_verify/validate_highspeed_actor_k_scan.py
scripts/model_verify/analyze_highspeed_actor_cap_schedule.py
scripts/model_verify/validate_highspeed_actor_cap_schedule.py
outputs/mppi_proposal/highspeed_actor_k16_90round_cap006_fold0_20260831_v1/
outputs/mppi_proposal/highspeed_actor_k16_160round_cap006_fold0_20260831_v1/
outputs/mppi_proposal/highspeed_actor_k16_160round_cap006_lrdecay_fold0_20260831_v1/
outputs/mppi_proposal/highspeed_actor_k16_160round_cap006_lrdecay_fold1to4_20260831_v1/
outputs/mppi_proposal/highspeed_actor_k16_cap_schedule_20260831_v1/
```

### 11.142 高速度Actor固定训练合同与后续传导计划（2026-08-31）

在§11.139--§11.141证据基础上停止继续扫描K、轮数、Actor LR和输出trust。K1/K16在固定
`0.02 sigma`下的90轮fold0终点mean只差约`26 J`，因此不能声称K16提高最终函数上限；但当前唯一完成
`160轮 + cap-only 0.06 sigma + 后段LR衰减 + 完整5-fold OOF`且指标最好的合同是K16。工程上选择
**证据最完整的K16合同**作为下一阶段固定基线，不再为证明K16必要性补做同合同K1。

固定合同如下：

| 项 | 固定值 |
|---|---|
| Actor | `DirectNoAnchorGTXActor`，absolute 8x2 center，warm不作为Actor输入 |
| 后端/梯度 | 训练评价使用确定性DBM标量cost；禁止DBM解析梯度 |
| outer rounds | 最多160；部署候选使用internal-selection选中的checkpoint，不强制latest |
| Actor microsteps | K16 |
| 每轮rollout bank | 65 candidates/state，search-recentered两阶段bank |
| Critic更新 | 每轮每个Twin 20次，持续吸收Actor-visited Replay |
| Actor输出trust | cap-only `0.06 sigma RMS`；只截大步，不放大小步 |
| exploration | `0.20 -> 0.05 sigma`，前90轮完成；第二半径比例0.70 |
| Actor LR | `2e-5`；第120--160轮余弦衰减到`5e-6` |
| Critic LR | `1e-4` |
| 固定安全边界 | warm/current center保留；单Actor center不获无条件部署授权 |

这里冻结的是**训练/选择合同**，不是已经存在一个可部署的全训练集单checkpoint。现有15个模型是
episode-grouped cross-fit机制证据；最终全训练集候选应在wrapper和Query接口合同冻结后，使用同一训练
合同重训，并保留独立internal selection carve-out和多seed选择。

后续按以下顺序执行，禁止并行改变多个合同：

1. **two-center固定预算OOF A/B**：复用现有15个OOF Actor，不重训。总采样数保持64；基线为
   warm-only 64，实验臂为warm 32 + Actor 32，两侧都保留一个未加噪中心，噪声使用CRN。Actor center
   的direct指标继续单列，不与wrapper结果混为一个指标。
2. **严格guard定义**：候选池包含warm只能保证warm仍可被模型比较，不能保证softmax加权后的新中心
   一定不差于warm。正式guard必须在同一次模型rollout预算内比较最终候选与warm；若最终候选更差则
   回退warm，并分别报告回退前wrapper结果与guard后结果。
3. **wrapper机制门**：split-32/32相对warm-only-64的mean/median改善，P05不退化，五档速度均报告；
   guard后warm-floor violation必须为0，并报告Actor侧候选被实际采用比例、ESS和控制变化量。
4. **全训练集候选Actor**：OOF wrapper门通过后，按本节固定合同在全部训练episode上训练3 seed；只用
   预留internal carve-out选checkpoint，不接触formal validation/test。
5. **Query shadow接口**：冻结Actor与candidate bank后，将相同中心/噪声分别送入DBM、Query PyTorch和
   Query ONNX。Torch/ONNX必须数值一致；以DBM仅作离线真值，评价Query的warm-vs-Actor排序、候选
   regret、guard选择一致率与最终DBM cost。不得依赖DBM梯度。
6. **一次性外部门与闭环**：只有Query shadow和two-center门均通过，才消费formal validation；随后做
   固定DBM短闭环，再做Query后端shadow/闭环，覆盖40/55/70/85/100 km/h与recovery场景，报告累计
   cost、失败率、横向/航向/速度误差及控制抖动。

当前下一执行项因此是第1项two-center固定64预算OOF A/B，而不是继续Actor超参训练。

### 11.143 范围纠正：先裁决单中心剩余DBM headroom，MPPI传导后置（2026-08-31）

按用户要求收窄当前问题：只评价策略网络直接输出的单个sampling center质量，并判断是否值得继续在DBM
数据上探索；§11.142的two-center/MPPI wrapper/Query传导计划整体后置，不是当前下一执行项。

现有直接中心证据为：完整600状态OOF中，最终Actor相对warm的逐状态cost降低比例中位约`11.75%`，
胜/平warm比例平均`98.61%`，warm-relative mean/P05 gain约`+21525/+3294 J`。但这些数字只证明相对
warm较好，不回答Actor离当前可搜索最优区域还有多远。

为检查现有上界是否仍有效，对§11.130的同一120个train-only strong-search状态做了零新增rollout的
同状态连接。该旧reference每状态使用最多965次DBM评价，从warm/proximal teacher/早期OAC Actor五个
起点搜索，mean J为`154546.1`。当前K16/160轮Actor的保存OOF center在同120状态上的三seed mean J分别
为`150648.3/151018.5/151017.7`，三seed平均为`150894.8`，比旧strong reference再低`3651.3 J`
（约旧reference的`2.36%`）；相对warm到旧reference的headroom recovery为`109.6%`。五档速度上当前
Actor平均也全部低于旧reference。

这不表示Actor超过数学理论最优。旧strong oracle只是有限预算、有限起点的best-found reference；当前
持续OAC经过160轮Actor-visited更新进入了旧五起点未搜索到的区域。正确结论是：**旧strong reference已
失效为剩余headroom上界**，不能据此继续生成teacher，也不能用proximal teacher的recovery>1判断饱和。

因此当前唯一优先实验改为**Actor-centered post-search headroom audit**：

1. 固定当前K16合同和selected OOF center，不更新Actor/Critic，不涉及MPPI；
2. 先复用上述120状态，分别从每个OOF Actor center出发做嵌套确定性DBM search，预算点
   `65/193/385/965`，候选集始终包含原Actor center，禁止baseline regression；
3. 主指标为`J_actor-J_postsearch`、相对Actor的配对降低比例、move fraction、残差移动RMS、P05/worst、
   五档速度及场景；不得再以warm或旧oracle作为分母掩盖Actor后的真实剩余空间；
4. 若actor-centered search的配对中位降低`<1%`且聚合降低`<2%`，并且各速度层都不超过2%，则停止
   DBM全局探索，认为单中心在当前参数化/数据上接近有限预算搜索饱和；
5. 若聚合降低`>=5%`且多数状态可稳定改善，则继续Actor-visited DBM Replay/OAC，搜索终点只作为
   诊断或Replay来源，不直接退回BC；
6. 若主体`<2%`但少数状态仍有大headroom，只对这些hard states定向补探索，禁止扩大全量数据；
7. 只有该审计完成后，才恢复§11.142的MPPI/Query传导计划。

本节未启动新的DBM rollout；120状态连接只使用已保存的Actor OOF cost与已验证strong-oracle cost。

### 11.144 最终Actor的16维DBM数值上限审计：主体已近饱和，仅tail保留定向空间（2026-08-31）

按§11.143继续冻结Actor/Critic，只评价确定性DBM `J50`下单个16维uniform-knot center。由于旧965-query
reference已经被最终Actor超过，本轮不再把有限预算旧搜索称为理论最优，而建立更强的**numerical
best-found reference**。120个train-only状态保持5速度×6场景×4独立episode平衡；每状态起点包含warm、
proximal teacher、旧strong oracle、最终Actor三个固定seed，并增加两个Actor附近随机起点和两个物理盒
全局随机起点。每个起点用投影DBM autograd分别跑`0.001/0.01/0.03`三档学习率、最多800步，动作始终限制
在物理`[-1,1]`盒内；随后用三套独立正交方向、`0.10/0.05/0.02/0.01 sigma`做46,080次无梯度细半径
polish。该DBM梯度仅用于离线上限审计，未进入Actor/Critic训练，也不兼容Query部署。

收敛曲线的mean best J为：step 0/100/200/400/800分别
`150502.7/150262.7/150231.3/150219.4/150219.3`。400到800步只下降`0.11 J`；独立无梯度polish把
`150219.3`降到`150181.7`，额外聚合改善仅`0.025%`。因此当前结果不是数学认证的全局最优，但多起点、
跨优化器和预算平台共同表明它是比旧reference强得多且已经稳定的数值参考。

最终直接中心结果如下。Actor主口径是三个独立固定seed的期望，即先逐状态对三个seed cost取平均；不得
逐状态挑最好seed作为可部署指标。

| 指标 | 数值 |
|---|---:|
| warm mean J | 192539.9 |
| 旧strong oracle mean J | 154546.1 |
| 固定Actor三seed期望 mean J | **150894.8** |
| numerical best-found mean J | **150181.7** |
| Actor相对best-found聚合差距 | **0.473%**，episode bootstrap 95% CI `[0.377%, 0.606%]` |
| Actor已回收warm→best-found headroom | **98.32%**，95% CI `[97.86%, 98.65%]` |
| 逐状态差距中位/P95/max | **0.449% / 2.09% / 8.88%** |
| 位于best-found 1% / 2%内 | **79.2% / 93.3%** |

固定seed单列的聚合差距为seed0 `0.310%`、seed1 `0.554%`、seed2 `0.554%`；逐状态挑三个seed中的最好者
可降到`0.221%`，但该值依赖不可部署的per-state oracle，只作随机种子上界。按速度分层，40/55/70/85/
100 km/h的聚合差距分别为`1.65%/0.60%/0.61%/0.32%/0.32%`：绝对cost最高的100 km/h并不是相对
headroom最大层，当前剩余空间更多集中在低速及少数个例。

独立validator重新装载全部来源，复算初始bank、autograd终点和polish终点，center/cost最大误差均为0；
预算曲线单调、物理边界与起点floor全部通过，判定
`HIGHSPEED_FINAL_ACTOR_NUMERICAL_ORACLE_REPLAY_PASS`。formal validation/test均未读取。

裁决为`ACTOR_BODY_NEAR_NUMERICAL_CEILING_TAIL_TARGETED_HEADROOM_REMAINS`：§11.143原定“聚合<2%、中位
<1%则停止全量DBM探索”的主体门已明显通过。因而停止继续全局扫描轮数、LR、trust、K、Actor结构或
扩大一般状态数据；这些最多只能争取约0.47%的聚合空间。仍有约6.7%状态超过2%差距、最差8.88%，若
继续DBM方向，只允许对该tail做定向机制分析/探索，并先判断其跨episode是否可学，不得用少数tail否定
主体已经接近数值上限的结论。该审计只回答train-only单中心DBM参数空间上限，不是formal泛化、Query、
MPPI wrapper或闭环结论。

为保证后续定向分析不漂移，`analysis.json`已固化`tail_over_2pct` manifest：共8帧，按差距从大到小为
`episode_006@40/underspeed`、`episode_069@70/combined`、`episode_007@40/underspeed`、
`episode_003@40/steady`、`episode_015@40/lateral`、`episode_014@40/lateral`、
`episode_004@40/underspeed`、`episode_021@40/combined`。每项同时记录artifact row、600状态源索引、
Actor期望J、best-found J、绝对与相对差距；后续若研究tail必须复用该manifest，不得事后改阈值重选样本。

```text
scripts/model_verify/run_highspeed_final_actor_numerical_oracle.py
scripts/model_verify/polish_highspeed_final_actor_numerical_oracle.py
scripts/model_verify/validate_highspeed_final_actor_numerical_oracle.py
scripts/model_verify/analyze_highspeed_final_actor_numerical_oracle.py
outputs/mppi_proposal/highspeed_final_actor_numerical_oracle_20260831_v1/
```

### 11.145 冻结Query训练域与当前高速度输入合同审计：接口匹配，但速度归一化严重外推（2026-08-31）

在把后续环境从DBM切换为冻结Query之前，先回查实际checkpoint、全部256个训练PKL、训练summary、ONNX
和当前600状态高速度replay；本节只审计训练/输入合同，不评价Query相对DBM或真实车辆的精度，也未启动
Query rollout、Actor/Critic训练、formal validation/test或闭环。

实际使用的模型是
`outputs/formal_small_car_query_dt005/20260730T144840/query_best.pt`，训练数据为
`/disk/collect_data_from_anycar/generated_small_car_query_dt005/20260730T144638`。数据由
`TorchDynamicBicycleRolloutBackend`确定性生成，共1024个episode、每个1000步/50 s、20 Hz；控制profile
每25步/1.25 s插值一次。目标速度knot为`0.2--3.5 m/s`随机值（每8个episode含零速起步），初始`vx`
为`0--2.5 m/s`（每4个episode置零），转向knot主要为`[-0.8,0.8]`平滑随机值，油门由速度误差PD与平滑
随机激励生成。数据无观测噪声，记录pre-transition状态与当前动作，故训练使用`steer_shift=0`。

训练合同为history 250步/12.5 s、预测50步/2.5 s，模型是确定性residual-mean
`TorchTransformerDecoderKinematicQueryMLP`，输出`[dx_body,dy_body,dvx,dyawrate]`，没有概率/risk head。
训练40 epoch、epoch 30按rollout score选中，初始LR `5e-4`、每epoch乘`0.99`、weight decay `1e-4`、
dropout `0.1`。历史/上下文/名义运动学分支均使用该低速训练集统计做固定标准化。历史输入为
`[dx_body,dy_body,dyaw,dvx,dyawrate,accel,steer]`；当前可观测状态为`[x,y,yaw,vx,yawrate]`，不显式输入
`vy`；reference不是Query动力学输入，只在外层cost计算使用。

以下合同与当前高速度replay完全匹配：`dt=0.05`、`lf/lr=0.1008/0.1092 m`、轴距`0.21 m`、质量
`4 kg`、摩擦`0.8`、油门/转向scale与bias、history/horizon长度、动作顺序`[acceleration,steering]`、
pre-transition与`steer_shift=0`。当前候选动作也仍在相同物理`[-1,1]`盒内；转向约2.0%略超训练PKL的
实测极值`[-0.9991,0.9979]`仅因候选恰好触到物理边界，不视为结构性不匹配。当前yawrate也全部位于训练
min/max内。

真正的不匹配有三类：

1. **速度支持完全不重叠**：训练实际`vx`范围`[-0.027,3.098] m/s`，P95仅`2.460 m/s`；当前600状态
   为`[7.796,30.626] m/s`、中位`17.558 m/s`，100%超训练上限。按checkpoint统计标准化后，context
   `vx`绝对z-score中位`25.76`、P95 `43.44`；history `dx_body`为`25.87/43.80`；warm名义transition
   `dx_body`为`25.61/43.92`。这不是轻微domain shift，而是明确的神经网络外推区。
2. **当前replay历史主要是假历史**：当前数据每episode只保留control step 0--4；step 0用恒速运动和
   current action填满250个token，最多只有4个真实transition替换尾部。因此history中accel/steer精确为
   0的比例均为`99.2%`，而训练PKL相应比例约为`9.8e-7/0`。形状和构造API匹配，但内容不等价于训练时
   完整12.5 s轨迹历史，也不等价于成熟闭环运行后的history。
3. **未观测侧向状态范围扩大**：当前`vy`为`[-4.228,5.702] m/s`，50.5%超训练min/max
   `[-0.532,0.584] m/s`；Query合同不显式输入`vy`，只能从历史横向运动间接感知。当前早期prime history
   又弱化了这条间接通道。

ONNX实际opset为17。已有PyTorch/ONNX等价性数字只来自原低速域（CPU max abs `2.74e-6`，CUDA mean/max
`4.97e-5/1.84e-3`），本轮尚未在上述高速度外推输入上验证。

裁决为`QUERY_INTERFACE_MATCH_SPEED_NORMALIZATION_SEVERELY_OOD`。如果用户明确把冻结Query本身定义为
环境，这些差异不构成“与DBM/真实车辆不准”的否决理由；但它们表示新环境是在极端归一化外推与人工
prime-history条件下定义的，可能数值不稳定或产生可被Actor利用的伪cost。故下一步不是直接复用DBM
Replay启动OAC，而是先做不比较DBM真值的Q0环境自洽门：固定高速度输入上检查PyTorch输出/轨迹/cost
全有限、重复调用确定性、候选排序与cost可复算，并做相同输入的PyTorch/ONNX高速度等价性；同时将
“早期prime history”和“成熟250步history”作为两个明确合同，不能混合训练或评估。Q0通过后才可把
Query cost用于重新标注Replay和在线Actor--Critic；失败时才讨论重训/改标准化或收窄环境域。

独立validator复核六个来源hash、当前速度、z-score、shape、物理参数与steer-shift，全部通过。权威产物：

```text
scripts/model_verify/analyze_highspeed_query_domain_contract.py
scripts/model_verify/validate_highspeed_query_domain_contract.py
outputs/query_mppi/highspeed_query_domain_audit_20260831_v3/
```
