> **[PARTIALLY-SUPERSEDED 2026-08-20]** 本文档部分内容仍有效；权威结论以 `mppi_sampling_center_review_20260812.md` §11.80 为准。§11.38 之前的引用请指向归档 `mppi_sampling_center_review_archive_20260812.md`。

# 用 RL 生成 MPPI 采样中心：目标、方案、计划与执行现状

> 汇总日期：2026-08-07
> 范围：**只讲"用学习/RL 的方式产生 MPPI 的采样中心（sampling center）"这一条线**。不含 Query 模型精度、数据采集流水线、teacher 标签生成的内部细节。
> 权威执行文档：`mppi_sequential_probe_execution_plan_20260806.md`。本文为汇总/展示视图，冲突时以权威文档为准。
> 当前资格状态：**`VALIDATION_ONLY_TEST_SEALED`** —— 不得进入 test、Query/ONNX、ROS 或长闭环。

---

## 一、我们要解决什么问题

### 1.1 现象：MPPI 的采样预算几乎全部浪费

DBM clean，step 340 快照，256 条高斯候选：

| 指标 | 值 |
| --- | --- |
| best cost | 5.0854864 |
| median cost | 253.9471 |
| P95 cost | 1705.3604 |
| **ESS（有效样本数）** | **1.4419617 / 256** |
| best 候选权重 | 0.8111926 |
| best + mean 占总权重 | 99.9517 % |

ESS ≈ 1.44 意味着 256 条 rollout 里**实际只有一条参与了决策**。
数据集级别的平均 ESS 也只有 **1.543**，best candidate 的 clipping 比例高达 **20.8 %**。

### 1.2 归因：不是求解器差，是提案分布放错了位置

MPPI 的采样中心目前就是 **warm start（上一轮最优序列左移一步）**。
在 96 帧诊断中，**warm 本身已经是 best 的帧占 44/96** —— 说明高斯扰动大部分时候没找到更好的东西，
而剩下的帧里 warm 又离好解很远。

所以改进方向不是改 MPPI 求解器，而是**把提案分布的均值放到更好的位置**。

### 1.3 目标函数：学的是"提案分布往哪放"，不是"最优动作是什么"

```
μ* = argmin_μ  E_{ε_1..N} [ C_MPPI( { μ + σ·ε_i }_{i=1..N} ) ]
```

这是 **amortized optimization**：用一次网络前向，替代在线的多阶段搜索。
与"直接回归最优动作"的区别在于，目标里含有 MPPI wrapper 的期望，
所以一个"平均意义上好"的中心比"某个 seed 下最好"的中心更有价值。

### 1.4 为什么值得做

| 收益 | 依据 |
| --- | --- |
| 省在线预算 | 离线多阶段引导用 128+128 就把 best cost 从 5.085 降到 2.150（−57.7 %），ESS 从 1.44 升到 30.76 |
| 省实时性 | DBM PyTorch 单步 66.1 ms、超时率 100 %；Query 29.0 ms、超时率 0.53 %。控制周期 50 ms，当前 57–60 ms/step 已超预算，必须靠减少 rollout 数量来挤时间 |
| 上限很高 | 数值 oracle J16 = 4.894，teacher = 12.762，而 warm = 29.517。可恢复空间巨大 |

---

## 二、RL 问题是怎么建模的

### 2.1 重要澄清：这是单步 contextual bandit，不是多步 RL

这一点在对外展示时必须写清，否则容易被质疑。

| 阶段 | 性质 |
| --- | --- |
| **S3-A（当前）** | **单步 contextual bandit**。每个快照做一次中心决策，立刻得到 cost。不引入 `next_state`、不引入 Bellman target、无闭环环境 |
| S3-B（未启动） | "同一车辆状态内的多轮 probe"。车辆状态、250 步 history、reference、guided center 在一次内部 episode 内**全部不变**，只有已探测 cost / mask / best-so-far / 剩余预算在变。原文明确："**它不是车辆 `next_state` 的多步强化学习**" |
| 闭环微调（远期） | 才是真正的多步 MDP |

discount γ、SAC entropy α、soft target τ 在文档中均未记录。

### 2.2 State（策略输入契约）

| 输入 | 形状 | 说明 |
| --- | --- | --- |
| `history` | `[250, 7]` | 5 维状态表示 + 2 维历史动作（12.5 s） |
| `initial_state` | `[5]` | `[x, y, yaw, vx, yawrate]` |
| `initial_state_six` | `[6]` | teacher 侧使用真实 `vy` |
| `current_action` | `[2]` | accel、steer |
| `reference` | `[51, 4]` | 当前点 + 未来 50 点 `[x, y, yaw, vx]` |
| `warm_knots` | `[8, 2]` | 当前 MPPI warm start |
| `noise_sigma` | `[2]` | 基础采样尺度 |
| cost weights / temperature | — | 作为条件输入，便于以后换 cost 而不重训 |

**坐标变换（必须）**：reference 转车体系，
`[x_r^body; y_r^body] = R(ψ₀)ᵀ [x_r − x₀; y_r − y₀]`，航向用 `Δψ = wrap(ψ_r − ψ₀)`，
网络吃 `sin(Δψ), cos(Δψ)`。reference 特征为 5 维：
`[x_body, y_body, sin(yaw_err), cos(yaw_err), vx_ref − vx_cur]`。
归一化统计量为 **train-only**。

后续版本逐步追加的反馈输入（这是"反馈条件化"的关键）：

| 增量输入 | 维度 | 内容 |
| --- | --- | --- |
| first-pass feedback | **74** | 16 维 guided step + 16 维经验 cost gradient + 16 维经验 Hessian 对角 + 16 维 soft-weight shift + 10 个 cost/ESS/clipping/fit 标量 |
| Critic gradient 统计 | 32 | 16 维 mean + 16 维 std |
| probe 状态 | 33 + 33 + 1 | probe value + observed mask + 剩余预算 |

### 2.3 Action（策略输出契约）

**16 维连续动作**（8 knots × 2 通道），**bounded residual，不是绝对值**：

```
Δμ_θ  = Δμ_max · tanh(f_θ(s))
μ_θ   = clip( μ_warm + g · Δμ_θ , -1, 1 )      # g ∈ [0,1] 固定 trust 系数
```

Direct Actor 阶段的最终形式：

```
center = clip( anchor + tanh(mu(state)) · maximum_delta_sigma · source_sigma )
A      = LinearInterpolate(center, horizon=50)   # 固定 align_corners=True
```

`maximum_delta_sigma` 默认 **2.0**（覆盖 T1 标准化分量约 99 %）。

**支撑域是一个已量化的问题**（2026-08-07）：

| 动作盒 | 覆盖 J16 的 validation context 比例 | 投影后 cost |
| --- | --- | --- |
| 2σ | **37.5 %** | 12.004 |
| 6σ | **98.8 %** | **4.895**（J16 本身 4.894） |

结论：**6σ 已消除输出范围瓶颈，不需要先改网络宽度**。

第一版明确不学的东西：knots 数（仍 8）、`sigma`（仍 `[0.25,0.35]`）、插值方式、
cost 与 temperature、MPPI 加权方式、covariance（16×16 full covariance 被明确排除）。

### 2.4 Reward：唯一契约（2026-08-06 修订，当前权威）

```
c        = policy(state)                    # 8×2 knots
A        = LinearInterpolate(c, 50)         # 唯一 50×2 actions
J_direct = Cost( DBM(state, A) )            # 唯一 forward cost
reward   = J_direct(anchor) − J_direct(c)
```

**关键性质：没有 reward seed。** 同一个 `(state, c)` 重复计算的最大误差实测为 **0.0**
（replay gate 允许 GPU 浮点 tolerance `1e-5`）。

代价函数（collection-time，已冻结）：

```
J = 5.0·pos_err² + 5.0·wrapped_yaw_err² + 1.0·vx_err²
  + 0.05·accel_rate² + 0.10·steer_rate²
temperature = 1.0 ,  yaw-rate 权重 = 0
```

#### 为什么必须取消 reward seed —— winner's curse 实证

旧目标是"把中心当 MPPI 均值，再用随机候选算 weighted-output cost"。
这让同一个 `(state, center)` 因 reward seed 不同而有不同标签，**混淆了"Actor 好不好"与"随机 wrapper 运气好不好"**。

实证数字（每 iteration 8 actions/context、2 reward seeds/action、共 320 个连续新中心）：

| seed 组 | 从同样 320 个动作里选出的 replay winner，audit mean |
| --- | --- |
| 第一组 | **7.251**（优于初始 Actor 7.772） |
| 第二组（独立） | **14.626** |

**两组结论方向完全相反。** 原文判定：严重 winner's curse，
"不能拿 v2 oracle 调 Actor 或宣称连续策略已通过"，且**不能再通过增加 reward repeats 修补**。
该批实验归档为 `INVALID / D2`。

#### 被否决的 reward 版本

| 版本 | 公式 | 否决原因 |
| --- | --- | --- |
| R0 | `R = −min_i C_i` | 诱导高风险中心，对 seed 极敏感 |
| R1（设计稿） | `C_out + β·C_softmin + γ·C_P10 + λ‖μ−μ_warm‖² + ρ·C_boundary` | β/γ/λ/ρ/τ 数值从未记录，未实际采用 |
| 旧 wrapper reward | 随机 MPPI weighted-output cost | winner's curse（见上） |

#### 第二层辅助指标（不得污染第一层 reward）

冻结候选库 **`fixed-hadamard-64-v1`**：

- 构成：exact center + 1 个固定 extra + **31 对 antithetic offsets** = 64
- 16×16 Sylvester Hadamard，保证 `8×2` 空间满秩；半径固定 `0.10 / 0.30 source_sigma`
- 不读取随机 seed；训练评估与运行时用同一个生成器
- `sampling_mode=fixed_hadamard_64`，要求 `num_samples=64`
- contract hash `0fd540206ec988f565481d25f9cbd0b4051f7847e154565d9eac4e7445b6a492`

**硬性口径**：该层"暂时只报告辅助中心质量，不能用于选择 Direct Actor checkpoint"。

### 2.5 网络

| 类名 | 参数量 | 输出 |
| --- | --- | --- |
| BC proposal policy | 397,840 | `[8,2]` bounded residual |
| `TorchMPPIProposalCritic` | 447,489 | 标量 advantage |
| `TorchMPPISequentialProbeActorCritic` | 593,315 | categorical logits + 两组 33 维 twin Q |
| `TorchMPPIContinuousCenterActor` | 514,208 | squashed Gaussian，16-D normalized residual |
| `TorchMPPIContinuousCenterCritic` | 641,921 ×2 | twin Q |
| **`TorchMPPIDeterministicCenterActor`（当前）** | **511,120** | 唯一确定性 center |

Direct Actor **删除了 `log_std_head`**，从旧 SAC Actor 精确迁移 encoder 与 mean head，
迁移后确定性 action/center **逐元素完全一致**。部署输出始终唯一。

建议结构（设计稿）：history encoder 128 + reference encoder 128 + state/action MLP 64
+ warm-knots MLP 64 → fusion 256→256 → 16。
Conv1d 通道数/核大小/激活/归一化层类型均**文档未记录**。

代码位置：`car_foundation/car_foundation/mppi_proposal_policy.py`。

### 2.6 探索：外置、结构化、衰减

**这是 Direct Actor 能学起来的唯一原因。**

```
c_explore = clip( c_actor ± radius(t) · hadamard_direction · source_sigma )
```

- 每轮从 16 个满秩 Hadamard 方向中循环取 **4 个**，生成正负 pair（每轮 8 个探索中心）
- Actor center 本身也进入 Replay Buffer
- 半径以 source_sigma 为单位独立衰减
- **探索完全在 Actor 外部、只在训练时执行**，不进入部署图

对比（固定五帧）：

| 探索几何 | Actor final direct mean | replay best |
| --- | --- | --- |
| 宽高斯（继承 SAC 方差头） | 9.782（= 初始，best iter 0） | **18.191** ← 比初始还差 |
| 结构化 Hadamard v2（`0.30→0.05σ`，lr `3e-5`，每轮 4 次更新） | 8.866 | 7.404（第 5 轮后发散） |
| **保守 v3（`0.15→0.03σ`，lr `1e-5`，每轮 1 次更新，24 轮）** | **8.919（单调下降）** | 7.562 |

结论明确写入文档：**此前主要失败来自探索几何，不是 reward seed 不足。**

### 2.7 信任步长（trust step）与 alpha 的含义

`scripts/model_verify/calibrate_mppi_direct_actor_trust_step.py`：

1. 候选 Actor = **bootstrap Actor 与 learned Actor 在参数空间的线性插值**
   `param = initial + alpha · (target − initial)`
2. 每个插值候选都用**真实确定性 DBM direct cost 在 validation episodes 上打分**
3. alpha 网格 `(0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)`
4. 选择约束：`win_fraction ≥ 0.50`、`median_gain ≥ 0`，再取 mean cost 最小者
5. 尾部守护：额外筛 `P05 gain ≥ −5.0`，产出 `direct_center_actor_trust_tail_guarded.pt`
6. **test episodes 从不加载**，summary 写死 `"test_policy": "episode_105..119 not loaded or evaluated"`
7. 输出目录已存在则 `FileExistsError`（禁止覆盖）

第一轮 alpha 标定表：

| alpha | validation mean | median gain | P05 gain | win |
| --- | --- | --- | --- | --- |
| 0.125 | 26.849 | 0.016 | −1.350 | 56.0 % |
| **0.375** | **25.974** | 0.026 | −4.762 | 53.2 % |
| 0.500 | 25.723 | 0.013 | −6.708 | 50.8 % |
| 1.000 | 26.310 | −0.162 | −17.058 | 43.7 % |

均值最优是 0.5，但**第二轮 on-policy 采集用了 P05 floor `−5` 下的保守 `alpha=0.375`**。

注意：还有一个**同名但不同含义**的 alpha —— **输出空间**插值到 J16 的诊断系数
（`alpha=0.40` 把 validation mean 从 25.518 降到 21.088，但 P05 gain −17.464、worst −81.653），
明确标注为 "two-Actor validation-calibrated diagnostic，不是合格的单 Actor"。

### 2.8 部署级 fallback（设计要求，未实现）

网络加载失败 / 输出含 NaN·Inf / 输出超信赖域 / 推理超 deadline / 状态超训练分布
→ **全部回退到原 warm start**。承载文件 `mppi_proposal_runtime.py` **尚未实现**。

---

## 三、预期：好到什么程度才算成功

### 3.1 GT-first 五层分解（衡量尺子）

```
J*_100     50×2 独立动作的 best-found direct optimum
J*_16      8×2 knots 线性插值后的 best-found optimum
J*_center  不限制 bank 的最优 sampling center
J*_bank    当前可部署 center bank 的 clairvoyant oracle
J_policy   固定顺序或反馈策略实际得到的结果
```

对应四个 gap，用来回答"到底被什么限制住了"：

| gap | 含义 | 若最大则 |
| --- | --- | --- |
| `J*_16 − J*_100` | 16-knot 参数化损失 | 增加 knots / 改动作参数化 |
| `J*_center − J*_16` | MPPI 随机采样 + 加权输出损失 | 改 MPPI 更新、sigma/temperature |
| `J*_bank − J*_center` | center bank 覆盖损失 | 进入 S2-B 动作库扩充 |
| `J_policy − J*_bank` | 策略选择/排序损失 | 进入 S2-A / S3 |

口径限定：**"GT"指多初值数值优化的 best-found oracle**，
只有不同初值/优化器/精度复算稳定收敛后才可称可信上限，**不得宣称已证明全局最优**。

### 3.2 单步准入门槛（S3-A gate，原文数字）

| 条件 | 阈值 |
| --- | --- |
| 相对 BC clone 与 optimized fixed bank 的 paired mean gain | bootstrap 95 % CI **下界 > 0** |
| P95 恶化 | **≤ 0.5** |
| 每个速度档 / recovery 的 mean regression | **≤ 0.5** |
| worst-context gain | **≥ −5** |

**未通过不得进入四轮训练。**

### 3.3 其他阶段的关键阈值

| 阶段 | gate |
| --- | --- |
| S1-A 可信度 | top-3 独立初值 final cost 相对 spread **< 1 %**，且增加迭代改善 **< 0.1 %** |
| S2-A 信噪比 | `\|gain\|>0.5` 子集 paired ranking **≥ 75 %**；terminal-return selection/audit correlation **≥ 0.7**；32 candidates 需保留 64 的 **95 %** mean gain 且 P95 恶化 ≤ 0.5 |
| S2-B 覆盖 | validation oracle 相对 fixed bank 至少改善 **0.5 mean cost**；2.4/2.8 m/s 或 recovery 至少 **1.0**；P95 不恶化超 0.5。实际执行时预定义 **回收 ≥50 % aggregate coverage gap（即 ≤ 6.6700）** |
| S4 | 沿用 S3，且**至少两个独立 probe seeds 上结论同号** |
| S5 闭环 | 累计 mean cost paired CI 优于 fixed；tracking P95 / 失败率 / 越界率不恶化；任一速度组不得系统性退化；总耗时不超控制周期，网络开销建议 **< 周期 20 %** |
| S6 ONNX | PyTorch/ONNX 最大绝对输出误差 **≤ 1e-5**，动作选择 **100 % 一致** |
| 恢复 Critic 主导更新 | 新 validation Actor 必须先低于 **teacher 12.762** 并改善 P95/max |

### 3.4 三层最终目标（不可混为一个结论）

1. **反馈价值**：相同 bank / probe 数 / candidate 预算下，反馈策略显著优于最强固定顺序与 state-only 顺序。
2. **固定 DBM 控制价值**：冻结策略在 fresh frozen-state gate 与短闭环 A/B 中同时改善 mean，且不恶化 tail、失败率、实时性。
3. **Query/部署价值**：只用 Query forward rollout reward 重训/校准后，PyTorch 与 ONNX 数值一致，并在 Query gate 中保留收益。**JAX 不维护。**

---

## 四、执行现状

### 4.1 一张总表：现在站在哪

600 个 validation context（300 独立 snapshot × 2 feedback context），DBM direct cost：

| 方法 | mean | median | P95 | max | 恢复 warm→J16 的比例 |
| --- | --- | --- | --- | --- | --- |
| warm center | 29.517 | 21.964 | 75.897 | 300.921 | 0.0 % |
| 第一轮保守 Actor（alpha=0.375） | 25.974 | 9.914 | 113.737 | 428.592 | 14.4 % |
| **当前第二轮 Actor（v4, alpha=0.75）** | **25.518** | 9.883 | 110.865 | 371.194 | **16.2 %** |
| DBM 二中心 guard（上界，不计成绩） | 24.661 | 9.823 | 104.603 | 371.194 | 19.7 % |
| T1 teacher | **12.762** | 9.395 | 31.133 | 119.479 | **68.0 %** |
| J16 best-found | **4.894** | 3.227 | 14.524 | 28.523 | 100.0 % |
| J100 best-found | 4.697 | 3.189 | 13.942 | 26.672 | 100.8 % |

**唯一可展示的正向 RL 结论**：
`27.445 → 25.974（第一轮保守）→ 25.518（第二轮 v4/alpha=0.75）`，累计改善 **1.926（约 7.0 %）**；
加 DBM 二中心 guard 为 24.661，累计 **2.784（约 10.1 %）**。

**但**：当前 Actor 比 teacher 高 **12.756**（约 teacher 的 **2.00×**），
仅在 **48.3 %** 的 context 上优于 teacher。

### 4.2 瓶颈定位（最重要的诊断结论）

五帧 pilot：

| snapshot | 速度/场景 | warm | J*_16 | J*_100 | 参数化 gap |
| --- | --- | --- | --- | --- | --- |
| episode_000 | 1.2 / steady | 27.215 | 6.823 | 6.210 | 0.612 |
| episode_021 | 1.6 / cold start | 19.529 | 1.182 | 1.177 | 0.004 |
| episode_042 | 2.0 / lateral recovery | 28.154 | 3.310 | 3.192 | 0.117 |
| episode_063 | 2.4 / heading recovery | 16.378 | 9.808 | 9.539 | 0.269 |
| episode_084 | 2.8 / dynamic recovery | 18.841 | 7.985 | 7.673 | 0.312 |
| **mean** | — | **22.023** | **5.821** | **5.558** | **0.263** |

四个 gap 的实测值：

| gap | 值 | 判定 |
| --- | --- | --- |
| 参数化（J16→J100） | **0.263**（≈ 可恢复量的 1.6 %） | 可忽略 |
| center vs J16 | **0.010** | 可忽略 |
| **bank 覆盖** | **1.703** | 主要结构损失 |
| seed 误选 | **0.029** | 可忽略 |

S1-C 六方案对比（同五帧，10 个 probe context，64 candidates/center，`0.10σ`）：

| 方法 | mean audit cost | 相对 unrestricted gap |
| --- | --- | --- |
| unrestricted center | 5.8238 | 0 |
| bank audit clairvoyant | 7.5269 | 1.7031 |
| learned feedback 4-probe | 7.5676 | 1.7437 |
| fixed-priority 4-probe | 7.6034 | 1.7796 |
| full-bank one-seed probe | 7.5524 | 1.7286 |
| guided anchor | 10.5498 | 4.7259 |

**learned feedback 只比 fixed priority 好 0.0358，距 bank clairvoyant 仅 0.0407。**
且 full-bank 单 seed 也没有消除 coverage gap → **继续增加同一 bank 的 probe 数不是解法。**

在 validation 全量上进一步确认：

> **J16 → J100 只改善 0.198，而 Actor → J16 的 gap 是 20.624。**
> 原文结论："当前主要瓶颈是高速度 recovery/tail 的策略泛化，而不是 16-knot 维度。"
> 据此更新计划：**不再把"Actor 参数上限"或"16-knot 自由度不足"作为主假设。**

### 4.3 按速度分层：问题全部在高速段

| 参考速度 (m/s) | 当前 Actor | T1 teacher | warm |
| --- | --- | --- | --- |
| 1.2 | **3.059**（优于 teacher） | 4.798 | — |
| 1.6 | **5.809**（优于 teacher） | 6.187 | — |
| 2.0 | 17.223 | 11.482 | — |
| 2.4 | 39.528 | 17.137 | — |
| 2.8 | **61.974** | 24.207 | **48.595（Actor 差于 warm）** |

**在 2.8 m/s，学出来的 Actor 比什么都不学更差。** 这是当前唯一的阻塞项。

### 4.4 尾部定位（精确到单帧）

| 项 | 值 |
| --- | --- |
| 第二轮 v4 worst gain | −157.260 |
| 第二轮 alpha=0.75 worst gain | **−109.109** |
| 最坏 context | **`episode_101/step_000262/context 1`**：143.683 → **252.793** |
| 该帧状态 | reference 2.4 m/s，实际 vx **3.135 m/s**（overspeed） |
| 其余坏例 | episode_101 的 overspeed / yaw-rate recovery；episode_102/103 的大 heading error 或高 yaw-rate |

第二轮完整对照：

| 指标 | 第一轮保守 | 第二轮 v4 | 第二轮 alpha=0.75 |
| --- | --- | --- | --- |
| validation direct mean | 25.974 | 25.525 | **25.518** |
| mean gain | — | 0.449 | **0.456** |
| median gain | — | 0.039 | 0.033 |
| win fraction | — | 57.2 % | **58.7 %** |
| P05 gain | — | −3.407 | **−2.457** |
| worst gain | — | −157.260 | −109.109 |

记为 **`episode-heldout two-round mechanism PASS / tail-safety FAIL`**。

### 4.5 Gate 表（执行记录，2026-08-06 起）

| 日期 | Step | 状态 | 关键数字 | 偏离 |
| --- | --- | --- | --- | --- |
| 08-06 | Plan | PASS | 冻结顺序、门槛、偏离模板 | D0 |
| 08-06 | S1-A pilot | **PASS** | 五帧 J*_16/J*_100 稳定，独立复算误差 < 3.1e-5 | D1 |
| 08-06 | S1-B first run | **INVALID** | 误用 `1.0σ`（probe 实为 `0.10σ`） | D2 |
| 08-06 | S1-B second run | **INVALID** | 虽改 `0.10σ` 但误用 iid noise（实际为 zero/extra/antithetic） | D2 |
| 08-06 | S1-B five-frame | **PASS** | bank clairvoyant coverage gap **1.703** | D0 |
| 08-06 | S1-C pilot | **PASS** | feedback 距 bank oracle **0.041** | D0 |
| 08-06 | S2-B forward-only dynamic bank | **FAIL** | `7.515→7.073`，回收 **26.2 %** < 50 % gate | D1 |
| 08-06 | S2-B larger-step check | **FAIL** | step 2.0→4.0，`7.524→7.042`，**28.3 %** 仍 FAIL | D1 |
| 08-06 | Continuous-SAC plan revision | PASS | 最终 Actor 直出连续 16-D residual；离散 bank 降级 | D0 |
| 08-06 | S2-C continuous interface | **PASS** | log-prob / 反传 / checkpoint smoke 全 finite | D0 |
| 08-06 | S2-C T1 bootstrap | **FAIL** | `11.474→10.874` 但 **3/5 帧退化**（T1 不条件化 first-pass feedback） | D1 |
| 08-06 | S2-C bank-oracle bootstrap | **PASS** | clone RMSE 0.0234；`10.819→7.754`（bank selection 7.642） | D0 |
| 08-06 | 旧 stochastic-wrapper S3-A v2/v3 | **INVALID** | 揭示 winner's curse，但不再回答 Direct Actor 问题 | D2 |
| 08-06 | Direct objective + fixed bank contract | **PASS** | direct cost 重复误差 **0.0**；候选 `64×8×2` 满秩 | D0 |
| 08-06 | Direct fixed-five SAC pilot | **FAIL** | 320 中心；初始=final **9.782**，best iter 0；replay best **18.191** | D1 |
| 08-06 | Deterministic Actor + structured exploration v2 | **PASS** | `9.782→8.866`，replay best 7.404，第 5 轮后发散 | D1 |
| 08-06 | Conservative deterministic Actor v3 | **PASS** | `0.15→0.03σ`、lr 1e-5、24 轮单调 `9.782→8.919` | D0 |

复合 gate 记法（原文逐字）：

1. `S2-C PASS / S3-A FAIL(D1)` —— block 从"reward seed 不足"修正为"Direct Actor 局部探索覆盖与 Q_direct 学习"
2. `S3-A mechanism PASS / generalization PENDING`
3. `episode-heldout two-round mechanism PASS / tail-safety FAIL`
4. `VALIDATION_ONLY_TEST_SEALED`
5. **2026-08-07 最新**：`6σ support PASS / train fit PASS / episode generalization FAIL / J16-MSE distillation REJECTED / test sealed`
6. `B3−B2` 严格反馈归因 = **PENDING**（缺冻结的 sequential state-only checkpoint）

**S0、S2-A、S3-B、S4、S5、S6、S7 均无执行行，即 PENDING。执行记录中没有任何 RUNNING 行。**

### 4.6 历史方法演进与结论

```
BC → Critic/ranking → AWR → 离散 actor-critic → multidirection replay
   → sequential probe SAC → 连续 SAC → 确定性 Direct Actor（当前）
```

| # | 实验 | 结论 | 关键数字 |
| --- | --- | --- | --- |
| E1 | T0 relabel | 只作 pipeline 标签 | warm 已 best 44/96；ESS 1.543；clipping 20.8 % |
| E2 | T1 teacher | **PASS** | audit 96/96 改善，平均 −5.897 |
| E3 | 首版 BC（8 ep） | 弱 | 12.522/12.153/7.494，改善仅 1.25 %，胜负 6/6 |
| E4 | BC 扩到 30 ep | PASS | 11.760/11.128/7.174，仅恢复 teacher gain 的 ~14 % |
| E5 | multi-elite 多中心 | **路线否决** | K=2/3 完美 selector 仅额外 0.026/0.034；固定预算下反而退化到 7.383/7.269 |
| E6 | BC 120 ep（当前 BC） | **PASS** | 21.622/18.611/12.136，teacher gain 恢复 **31.7 %**，胜负 234/66，worst −17.680 |
| E7 | Critic v1 | 排序 PASS，actor 未准入 | pairwise 0.801；但 **teacher-vs-network 仅 0.503（随机）** |
| E8 | Critic v2（局部 relabel） | PASS（仅作排序器） | pairwise 0.768，top-1 0.717，teacher-vs-BC 0.917 |
| E9 | `actor_critic_local_20260805_v1` | **rejected_all** | BC anchor 0.25/1/4 → 27.056/26.493/24.816，全差于 BC 19.510 |
| E10 | AWR（temp=2） | 微弱 PASS，不替换 BC | 19.145 vs BC 19.230（+0.084），tail 不变 |
| E11 | 全秩局部 reward sidecar | 数据 PASS | 旧 11-center 局部矩阵**秩只有 4**（16 自由度）→ 补到 33 centers 满秩，40,550,400 rollouts |
| E12 | full-rank Critic | **梯度准入 FAIL** | cosine 0.166/0.193 vs 数据上限 0.617；**相邻帧交叉 cosine 仅 0.044** |
| E13 | 两轮 feedback Critic | 梯度可学 PASS，center 更新 FAIL | cosine **0.615**（上限 0.902）；但 top-1 反而退化 0.352 |
| E14 | risk replay + step-risk Critic | 有效但仍不产 actor | 只有 **+0.03σ** 稳定（胜率 72.23 %）；v3 gate 移动 180/600、+1.135、worst −4.948（无门控 worst −213.450） |
| E15 | 单步离散 actor-critic | 未通过 | Actor 12.241 vs teacher 11.889；mean gain −0.352，P10 −5.049，worst −230.403 |
| E16 | multidirection actor | 更差 | 12.669（前版 12.241）；平坦 33 类监督放大标签切换 |
| E17 | probe policy | mean 首超 teacher，tail 未过 | 11.242 vs teacher 11.898，但需 2112 rollouts/context，P95/max 35.094/249.362 vs 29.070/64.313 |
| E18 | sequential probe SAC | **预算压缩 PASS / 自适应排序 FAIL** | 4 probe 保留 33 probe 的 **97.2 %**；但 learned 11.246 略差于 fixed 11.138 |
| E19 | S1 GT-first pilot | **PASS，定位主 gap** | 见 §4.2 |
| E20 | S2-B 动态 bank | **FAIL** | 26.2 % / 28.3 %，均 < 50 % |
| E21 | S2-C bank-oracle bootstrap | PASS（仅初始化） | clone RMSE 0.0234，`10.819→7.754` |
| E22 | S3-A 连续 SAC | **FAIL(D1)** | winner's curse 7.251 vs 14.626；SAC best iteration 仍为 0 |
| E23 | Direct + 宽高斯探索 | **FAIL** | initial=final 9.782，replay best 18.191 |
| E24 | Direct + 结构化衰减探索 | **PASS（机制）** | v3 24 轮单调 `9.782→8.919` |
| E25 | episode-heldout 两轮 | **mechanism PASS / tail FAIL** | 27.445→25.974→25.518，worst −109.109 |
| E26 | validation 数值 oracle | 定位完成 | 见 §4.1 / §4.3 |
| E27 | J16 plain 蒸馏 | **rejected** | 6σ 最好 validation 42.517（对照 Actor 25.518、teacher 12.762） |
| E28 | J16 cost-sensitive + 扩充覆盖 | **coverage PASS / Actor FAIL** | 最好 seed 28.675；mean gain −3.156，P05 −74.819，worst −369.475 |

### 4.7 三个已实证的陷阱（展示时应重点讲）

**陷阱一：Critic 的连续梯度外推一定失败。**
排序能学（pairwise 0.768–0.917），但对动作求梯度不能用。
决定性诊断：同状态跨 seed gradient cosine **0.630**，
**同 episode 相邻保存帧交叉 cosine 仅 0.044、正值比例 54.25 %**。
1800 个训练状态不足以让 state-only Critic 泛化出可靠的 16 维梯度。
实际后果：`actor_critic_local_20260805_v1` 全部配置差于 BC 达 **>68**，标记 `rejected_all`。
无信任步长的 Actor 在 **epoch 5** 之后就开始"严重利用 Critic 外推"。

**陷阱二：随机 reward 目标会造成 winner's curse。**
见 §2.4，两组独立 seed 给出 7.251 vs 14.626 的相反结论。修复方式是取消 reward seed，不是加 repeats。

**陷阱三：探索几何比算法更重要。**
16 维宽高斯探索的 replay best（18.191）比不探索还差；
换成结构化 Hadamard antithetic + 衰减半径后降到 7.404 / 7.562。

补充两条工程性陷阱：

- **朴素重置 Critic**：第二轮随机重置 twin Q 会把 validation ranking regret 从 **8.00 恶化到 24.33** → 必须继承第一轮 Q，且 **Critic/Actor 的 epoch 0 也要参与 checkpoint 竞争**。
- **scalar-Q 回归淹没小 cost 差**：需要 forward-only **finite-difference slope loss**（用同半径同方向的正负 direct reward 差监督 Q 局部斜率，不读 DBM 解析梯度）。v4 Actor 超参：local-oracle BC、`Q weight=0.1`、`trust penalty=10`、lr `2e-6`。
  即便如此，v4 Critic 的 local slope correlation / direction agreement 仅 **0.333 / 60.9 %**（global 0.559/8.873）。

### 4.8 2026-08-07 最新一轮：为什么"直接蒸馏 J16"也不行

思路很自然：既然 J16 = 4.894 是可达上限，直接把 J16 当监督目标蒸馏。

| 版本 | train mean | validation mean | action RMSE |
| --- | --- | --- | --- |
| 2σ 最好 seed | 12.631 | 44.645 | 0.258 |
| 6σ 最好 train seed | **7.263** | 46.340 | 0.099 |
| 6σ 最好 validation seed | 7.333 | **42.517** | 0.098 |
| 当前两轮 Actor（对照） | — | **25.518** | — |
| T1 teacher（对照） | — | **12.762** | — |
| J16 | 4.830 | **4.894** | 0 |

**train 拟合极好（7.263），validation 灾难（42.517），比不蒸馏还差一大截。**

排查过程（重要，说明这不是调参问题）：

1. **不是 early stopping**：按 30 个 `速度×场景` strata 各用 2 episode 拟合 / 1 episode 选 epoch，跑满 300 epoch，6σ 最好仍为 **48.580**。
2. **不是多模态**：validation 只有 **2/300** 帧同时满足 top2 cost 差 <1 % 且 knots 距离 >0.5σ。
3. **不是边际分布外**：train/validation 的 vx、vy、yaw-rate、横向/航向误差、动作、history 边际范围基本重叠。
4. **是目标高频变化 + 联合覆盖不足**：**相邻保存帧的 J16 knots 有 68.8 % 变化超过 1σ**。
5. **高速轨迹对动作误差极度敏感**：沿直线插值到真实 J16：

| alpha | 0 | 0.5 | 0.75 | 0.9 | 0.95 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- |
| validation mean cost | 42.517 | 16.913 | 8.630 | 5.575 | 5.071 | 4.894 |

→ Actor 方向并不完全错误，但**必须非常接近 J16 才有用；等权 knots MSE/Huber 无法反映小动作误差造成的巨大轨迹 cost**。

于是做了 cost-sensitive 版本（`j16_cost_sensitive_expansion_20260807_v2`）：
新增 270 train-only episodes / 1,350 snapshots（30 strata 各 +9 → 合计 360 episodes），
按 Hadamard 基下的 forward-cost curvature 加权代替 plain MSE，
每 seed 跑满 300 epochs 后用选中 epoch（228/208/217）在全部 360 episodes refit。

结果：三 seed validation **31.432 / 28.675 / 33.693**。
最好 seed 优于 plain-J16 区间（42.517–49.933），但**仍差于当前 Actor 25.518**。
且 mean gain **−3.156**、P05 **−74.819**、worst **−369.475**；
按速度相对当前 Actor 在 1.2/1.6/2.0 改善 0.476/0.592/1.538，在 2.4/2.8 **退化 12.980/5.408**。

状态：`coverage/direction PASS / full-step Actor FAIL / tail gate FAIL`。

---

## 五、计划

### 5.1 阶段总览

| 阶段 | 名称 | 状态 | 前置 |
| --- | --- | --- | --- |
| S0 | 冻结基线、评估器与 seed ledger | 无执行记录 | — |
| S1-A | 固定 DBM direct-action oracle | **PASS(D1)** | S0 |
| S1-B | sampling-center oracle | **PASS(D0)**（v1/v2 INVALID/D2） | S1-A |
| S1-C | bank 与 policy gap | **PASS(D0)**，B3−B2 归因 PENDING | S1-B |
| S2-A | on-policy probe outcome 与 reward repeat | **PENDING（条件分支未启动）** | S1 |
| S2-B | 扩充运行时非局部动作库 | **FAIL(D1) ×2** | S1 |
| S2-C | 连续 16-D Actor 合同与 bootstrap | **PASS** | — |
| **S3-A** | **单步 actor-visited 连续/确定性 Direct Actor** | **mechanism PASS / generalization FAIL / tail FAIL** ← **当前位置** | S2-C |
| S3-B | 四轮 continuous SAC 内部搜索 | **PENDING** | S3-A PASS |
| S4 | 完全冻结的 fresh DBM qualification | **PENDING** | S3 |
| S5 | 固定 DBM 短闭环 A/B | **PENDING** | S4 PASS |
| S6 | Query 重标注、训练与 ONNX 一致性 | **PENDING** | S5 PASS |
| S7 | ROS / 在线集成（shadow mode 起步） | **PENDING** | S6 PASS |

### 5.2 S1-C 定下的分支规则（已按此走）

| 若 | 则 |
| --- | --- |
| `J*_16 − J*_100` 占主要部分 | 增加 knots / 改动作参数化 |
| `J*_center − J*_16` 最大 | 改 MPPI 更新、sigma/temperature 或多轮 center 优化 |
| **`J*_bank − J*_center` 最大** | **进入 S2-B 动作库扩充** ← 实测走了这条 |
| `J_policy − J*_bank` 最大且反馈 oracle 明显优于 state-only | 进入 S2-A 和 S3 |
| top starts 未稳定收敛 | 本阶段 `INVALID/INCONCLUSIVE`，**不得据此增加 RL 复杂度** |

### 5.3 下一步优先项（原文口径）

来自 validation 数值 oracle 章节的收尾结论：

> **不再把"Actor 参数上限"或"16-knot 自由度不足"作为当前主假设。**
> 下一步优先：
> 1. 用 **J16 oracle 量化每个速度/状态的可达 gap**；
> 2. 针对 **2.0–2.8 m/s recovery 状态**补充**局部 on-policy coverage、风险/保守更新或 deterministic fallback**；
> 3. **通过 heldout tail gate 后，才进入 Query/ONNX 和闭环资格。**

来自 episode-heldout 两轮的收尾结论：

> **先定位最坏 episode/context，并训练 cost/risk 双头或 conservative fallback，不再仅增加 Actor epoch。**

来自 2026-08-07 J16 蒸馏的 4 条 next steps：

1. 把 30 个速度/场景 strata 的独立 train episode 从每层 3 个扩到**至少 10–12 个**，每 episode 只取 **4–6 个**充分分散的 fully-observed snapshot；
2. 用 **forward-only antithetic/Hadamard cost** 计算局部曲率或 verified improvement target，训练 **cost-sensitive Actor loss**，而不是等权 knots MSE；
3. 按**高速度 regret 和 recovery 状态加权**，但保持 formal validation 与 **test 封存**；
4. **只有新 validation Actor 先低于 teacher 12.762 并改善 P95/max，才恢复 Critic 主导更新。**

来自 S2-B 的收尾结论：

> 不启动 replay 或 Actor/Critic 训练；**下一次只验证多轮 forward-response** ——
> 每个真实 probe 之后用新轨迹误差重新构造下一轮中心。

### 5.4 Seed ledger

S4 正式预留段（写入后不得改用途）：

```
29001--29016   on-policy pilot selection/audit
29101--29116   action-bank pilot selection/audit
29201--29204   formal fresh probe seeds
29211--29218   formal fresh evaluation seeds
293xx          short DBM closed-loop seeds
30xxx          future Query relabel/evaluation
```

已实际使用：

| seed | 用途 |
| --- | --- |
| `20xxx--28418` | **历史段，不再用于新的最终 qualification** |
| `28401` / `28411--28418` | 旧 fresh 基线的探测 / 评价 |
| `29411--29412` / `29421--29424` | S1-B selection / audit |
| `29511--29512` / `29521--29524` | S2-B v3 selection / audit |

规则：**若发现冲突先更新 ledger；不能静默换 seed 后与旧结果混报。**

### 5.5 偏差分级 D0–D4

| 等级 | 定义 | 处理 |
| --- | --- | --- |
| D0 | 无偏离 | 按计划继续 |
| D1 | 数值偏离预期，协议未变 | 保留有效负结果，按门槛转向 |
| D2 | 预算 / seed / cost / 模型 / 数据分布发生未计划变化 | **暂停对比，重建公平基线** |
| D3 | 任务范围漂移（如 frozen-state 未过就上闭环/Query） | **停止，回到最近通过的 gate** |
| D4 | train/val/test 泄漏、用 audit 调参、复用评价 seed | **结果作废，换新 seed** |

强制条款：**D2–D4 不能用文字解释后继续，必须修正协议后重跑；任何 gate FAIL 都不自动进入下一阶段。**

状态词表只允许：`PENDING / RUNNING / PASS / FAIL / INVALID`。

---

## 六、纪律与禁止事项

### 6.1 关于 DBM 解析梯度

DBM 解析梯度**只允许**用于离线诊断 oracle。
**不得**成为策略输入、teacher 运行时依赖或部署算法的一部分；
S2-B 运行时输入不得含 T1 teacher center 或 DBM 解析梯度；
S3-A Actor loss 与 v4 slope loss 均不读 DBM 解析梯度。
（DBM 梯度 oracle 的 1.78643 只是理论下界，不参与任何 gate。）

### 6.2 关于 reward 与标签

- 一个已给定的 `c` **不再使用 reward seed**
- 旧的随机 MPPI weighted-output cost **不能直接作为新 Critic 标签**（必须 direct 重算）
- 第二层"中心邻域质量"指标**不得反向污染**第一阶段 Direct Actor reward
- 固定 64 候选库**不能用于选择 Direct Actor checkpoint**
- 旧 stochastic-wrapper v2/v3 **不用于 Actor gate / Critic 初始化**，且**不能通过增加 reward repeats 修补**

### 6.3 关于 bootstrap 与离线 Q

- bootstrap **只准做两件事**：初始化 Actor mean、初始化 twin Q
- **禁止**用离线 Q 直接更新并资格认定 Actor
- 任何新 Actor center **必须先经 fixed-DBM forward rollout 写入 actor-visited replay**
- **不能冻结 Critic 后长时间利用其梯度**，每批更新后必须重新与 DBM 交互
- 旧 33-slot bank / T1 / residual-response **只能作 BC/exploration prior，不能重新变成离散动作上限**

### 6.4 关于数据隔离

- **test episodes `episode_105--119` 禁止加载**，至今未生成/未读取（刻意封存）
- validation states **只用于 checkpoint 选择**
- 按 **episode 或完整 session** 拆分，不得把同一 episode 相邻 snapshot 或同一 snapshot 的 candidate 随机拆入 train/val/test
- 真实车辆或高保真 Isaac/MuJoCo 数据**不能反向污染**当前 DBM/Query final test
- S4 未通过时**不直接调 fresh test**

### 6.5 关于归因与报告口径

- **不能把 wrapper 收益记为 Direct Actor 收益**
- direct oracle 的额外计算**只记作离线诊断预算，不伪装成可部署性能**
- 二中心 guard 的 **24.661 不计入 Direct Actor 25.518 的成绩**
- bank-oracle bootstrap 的结果**只是可靠初始化，不宣称连续探索收益**
- **固定五帧结果不作泛化结论**
- reward repeats 规模**不得报告成新车辆状态数**（如 3.38 M candidate rollouts ≠ 600 个新状态）
- PyTorch/ONNX 不一致时**先修部署图，不做性能归因**
- **JAX 不维护，也不作为任何 gate 的依赖**

---

## 七、风险与未办事项

### 7.1 阻塞级风险

| 风险 | 数字 |
| --- | --- |
| **高速段（2.4–2.8 m/s）尾部灾难性回归** | worst −109.109；最坏帧 143.683 → 252.793；2.8 m/s 时 Actor 61.974 > warm 48.595 |
| **Critic 质量本身偏弱** | v4 global correlation/regret 0.559/8.873；local slope correlation/agreement 仅 0.333/60.9 % |
| **目标高频变化** | 相邻保存帧 J16 knots 有 68.8 % 变化超过 1σ |
| **高速轨迹对动作误差极敏感** | 插值到 J16 需 alpha≥0.9 才降到 5.575 |
| **实时性超预算** | 当前 57–60 ms/step vs 控制周期 50 ms；S5 要求网络开销 < 周期 20 % |

### 7.2 未办事项

| 项 | 状态 |
| --- | --- |
| S0 manifest / SHA256 / seed ledger 是否已固化 | **文档无记录** |
| S2-A on-policy probe pilot（3.38 M rollouts） | 从未启动 |
| S2-B 多轮 forward-response pilot | 未执行 |
| `B3−B2` 严格反馈归因 | PENDING（缺冻结的 sequential state-only checkpoint） |
| cost/risk 双头或 conservative fallback | **未训练** |
| train episode strata 从每层 3 扩到 10–12 | 未完成 |
| cost-sensitive Actor loss（forward-only 局部曲率/verified improvement） | 部分实现，Actor 仍 FAIL |
| 二中心 guard 的 Query 版本 | 未验证（必须先验证 Query 上的二中心 cost 排序） |
| S3-B / S4 / S5 / S6 / S7 | 全部 PENDING |
| `mppi_proposal_runtime.py` | **未实现** |
| proposal 侧 ONNX 导出 | **未实现** |
| 固定 DBM 闭环 A/B | **从未做过** |
| cost 权重 / temperature 采样范围 | **未冻结** |
| Query ONNX opset 版本 | 全部文档均未记录 |
| T0 / T1 的具体来源与训练方式 | 权威文档未记录（需查 teacher 文档） |

---

## 八、关键文件索引

### 8.1 库代码

```
car_foundation/car_foundation/mppi_proposal_policy.py     # 全部 Actor/Critic 定义
car_foundation/car_foundation/mppi_proposal_runtime.py    # 计划中，未实现
car_dynamics/car_dynamics/controllers_torch/mppi.py       # sampling_mode=fixed_hadamard_64
car_dynamics/car_dynamics/controllers_torch/dbm.py
car_ros2/car_ros2/car_node.py
car_ros2/launch/car_sim.launch.py                          # mppi_num_samples, mppi_sampling_mode
```

### 8.2 当前主线脚本（`scripts/model_verify/`）

```
# GT / 诊断
generate_dbm_direct_gt_pilot.py        validate_dbm_direct_gt_pilot.py
generate_dbm_sampling_center_gt_pilot.py  validate_dbm_sampling_center_gt_pilot.py
evaluate_dbm_gt_policy_gap_pilot.py
generate_dbm_direct_gt_validation.py   validate_dbm_direct_gt_validation.py
compare_mppi_direct_actor_teacher_gt.py

# 动态 bank（S2-B，FAIL）
evaluate_dbm_dynamic_generator_oracle.py  validate_dbm_dynamic_generator_oracle.py

# Direct Actor 主线（S3-A）
generate_dbm_direct_center_replay.py   validate_dbm_direct_center_replay.py
train_mppi_direct_center_actor_critic.py
calibrate_mppi_direct_actor_trust_step.py
fine_tune_mppi_direct_center_sac_pilot.py
evaluate_mppi_direct_actor_guard.py

# J16 蒸馏（2026-08-07）
train_mppi_j16_oracle_distillation.py
train_mppi_j16_cost_sensitive_distillation.py
analyze_mppi_j16_distillation_path.py
analyze_mppi_j16_cost_sensitive_actor.py

# 历史（BC / Critic / 离散）
train_mppi_proposal_bc.py              evaluate_mppi_proposal_bc.py
train_mppi_proposal_critic.py          train_mppi_fullrank_local_critic.py
train_mppi_two_pass_feedback_critic.py train_mppi_two_pass_step_risk_critic.py
train_mppi_single_step_actor_critic.py train_mppi_sequential_probe_sac.py
```

### 8.3 当前有效产物

```
# 首选 BC checkpoint
outputs/mppi_proposal/bc_t1_conv_diverse_20260805_v1/trust1_seed1.pt

# 当前 Direct Actor（VALIDATION_ONLY_TEST_SEALED）
outputs/mppi_proposal/direct_center_actor_critic_diverse_20260806_v4
outputs/mppi_proposal/direct_center_actor_trust_step_20260806_v2
    direct_center_actor_trust_selected.pt
    direct_center_actor_trust_tail_guarded.pt

# 数值 oracle 基线
outputs/mppi_proposal/dbm_direct_gt_validation_20260806_v2
outputs/mppi_proposal/direct_actor_teacher_gt_validation_20260806_v1

# 二中心 guard（上界）
outputs/mppi_proposal/direct_center_actor_two_center_guard_20260806_v1

# 数据
/disk/collect_data_from_anycar/mppi_rl_closed_loop/
    fixed_dbm_policy_diverse_20260805_v1/              # 120 ep / 2400 snapshots / 90-15-15
    fixed_dbm_policy_train_expansion_20260807_v1/      # 270 train-only ep / 1350 snapshots
    labels/dbm_direct_center_replay_diverse_20260806_v{1,2}
    labels/dbm_j16_local_curvature_train_{diverse,expansion}_20260807_v1
```

被明确否决、不得部署的 checkpoint：

```
outputs/mppi_proposal/actor_critic_local_20260805_v1                    # rejected_all
outputs/mppi_proposal/continuous_center_sac_fixed5_pilot_20260806_v{2,3}  # INVALID/D2
outputs/mppi_proposal/direct_center_j16_distillation_20260807_v1        # rejected
```

---

## 九、一页总结（可直接用于汇报）

**要做什么**：用一个网络在每个控制周期直接输出 MPPI 的采样中心（8×2 knots 的 bounded residual），
把当前 ESS 只有 1.44/256 的采样浪费换成更少的 rollout 或更好的解。

**怎么定成败**：`reward = J_direct(anchor) − J_direct(c)`，无 seed，DBM forward 一次算完。
尺子是五层 GT 分解：warm 29.517 → Actor 25.518 → teacher 12.762 → J16 4.894 → J100 4.697。

**做到哪了**：机制已通过 —— 确定性 Actor + 结构化 Hadamard 衰减探索 + 参数空间 trust step，
两轮 episode-heldout 从 27.445 单调改善到 25.518（约 7.0 %），胜率 58.7 %，P05 −2.457。

**卡在哪**：只恢复了 warm→J16 的 **16.2 %**（teacher 68.0 %），且**全部问题在高速段**——
2.8 m/s 时 Actor（61.974）比不学（warm 48.595）更差，worst gain −109.109。
瓶颈已排除 knot 维数（J16→J100 仅 0.198，Actor→J16 是 20.624），
定位为**高速 recovery 的策略泛化与尾部安全**。

**下一步**：按速度用 J16 量化可达 gap → 针对 2.0–2.8 m/s recovery 补 on-policy coverage
和 cost/risk 双头或 deterministic fallback → 过了 heldout tail gate 才碰 Query/ONNX 和闭环。

**不能说的话**：不能说 RL 已经通过；不能把二中心 guard 的 24.661 当成绩；
不能把固定五帧的 8.919 当泛化结果；test 仍封存。

---

## 十、2026-08-17 路线覆盖说明

本节覆盖第九节“下一步”的 2026-08-07 历史口径；完整证据见
`mppi_sampling_center_review_archive_20260812.md` §11.35--§11.37 + `mppi_sampling_center_review_20260812.md` §11.38。

- 修复 value-delta v1 的 batch 错行后，真标量 Q v2 全量两臂 30 run 中各臂仍为 `0/15` gate、
  pooled value corr `-0.03~-0.12`、梯度 cosine median `-0.62~-0.70`。
- E0/E1/E2 显式输入三臂均不能恢复 G0_ONLY；E2 已使用 DBM 充分输入。当前 DBM 不读取地图/
  障碍，参数和权重恒定，因此不再把未知外部 context 作为主嫌疑。
- 当前结论只是否决“可靠 Critic action gradient”主路线，不否决直接 sampling-center Actor。
  Critic 暂停 loss/结构/输入扫描，可保留为未来离线 candidate ranker。
- §11.41--§11.42的600-context逐cost项审计进一步定位了机制：250个近邻梯度翻转对中，
  68.0%可由单一cost项自身反向直接归因（同向对仅4.3%）；其中position为149/170，等于
  全部翻转对的59.6%。叠加主导项互换和强对消后机制族覆盖83.2%（同向41.7%）。严格
  episode-cluster bootstrap后四项95% CI分别为翻转单项[60.0%,75.5%]、同向单项
  [2.0%,7.0%]、翻转机制族[77.1%,89.1%]、同向机制族[32.8%,50.7%]，组间仍分离。
  这说明问题来自position复合项在action空间的近邻高曲率/分支翻向，并受多项对消放大；
  不能简写成“全部翻转88%来自position”，也不证明平方位置误差本身不光滑。
- 下一主线改为 `pi(s,a0)->delta_a` 的近端 Offline Search Distillation：先用现有 T1/J16 做
  optimal-residual coherence/mode 审计，再做 300--600 状态的 search 预算曲线，最后才进行
  episode-grouped、Actor-visited 的迭代蒸馏。
- 这不是重做 J16 MSE。plain/cost-sensitive J16 蒸馏已分别失败到 validation `42.517/28.675`；
  新路线必须拆开 teacher 质量、train 蒸馏、heldout 泛化和后续闭环传导。
- position逐horizon审计进一步表明：250个翻转对中56.4%为全局多时段反向、38.0%为mixed，
  单段局部制造仅5.6%；position梯度质量中位74.6%来自t34--50。该结果支持用完整rollout
  cost做黑盒Search，但晚段占比不是“早期knot无信息”，也不直接授权缩短horizon。正式
  Phase 2仍用原始J50；固定时间索引along/cross分解、horizon weighting和bounded margin
  作为并行counterfactual，须用原J50真实rollout与后续闭环判定，不能用flip下降单独过门。

默认 Actor、formal validation 和 test 继续冻结；只有新的 episode-heldout 单步 tail gate 通过后，
才进行固定 DBM 短闭环 A/B 与 Query/ONNX 工作。

### 10.1 2026-08-18 Critic最后坐标实验覆盖结论

§11.50已完成Critic的最后一次action-coordinate机会：在不改loss、状态输入、数据、优化器和
checkpoint选择的前提下，对原始`Q(s,a)`与满秩sensitivity-normalized temporal-DCT
`Q(s,z), a=Bz`做了3-fold×3-seed严格配对。新坐标把coordinate-gradient P10从约`-0.97`
改善到`-0.61~-0.80`，但映射回物理action后的P10仍为`-0.32~-0.39`，early-steering P10仍为
`-0.66~-0.69`。真实DBM J50在0.05σ步长下三seed的gain P05为`-5.30/-6.04/-11.00`，
中位gain也全部为负。

正式qualification为`COORDINATE_CRITIC_FAIL_CLOSE_GRADIENT_MAINLINE`：坐标condition是
放大因素，不是hard-tail的根因；不再恢复Critic作为Actor gradient provider，也不再做第六/
第七轮loss或basis扫描。Critic仅保留为离线候选粗排/value辅助，Phase 2继续使用multi-128
guarded teacher训练bounded proximal residual Actor，并按episode-grouped cross-fit判定。

动作通道顺序明确为`[acceleration, steering]`，early-steering indices为`[1,3,5]`。旧
position-horizon产物中两个名为early-steering的局部map实际取了acceleration通道，源码已改；
全16维horizon质量与翻转机制结论不受影响。复现入口为
`g0_sensitivity_coordinate_cv_20260818_v1`及对应run/validate脚本。
