# MPPI 采样序列优化：当前固定状态

更新时间：2026-07-31。

## 当前目标和边界

下一阶段只优化 MPPI 候选动作序列的生成方式。为保证对比有效，下列内容固定：

- 无观测噪声的 numeric DBM Quick Start 场景；
- 与 numeric simulator 数值一致的 Torch DBM rollout；
- DBM 的完整六维状态、当前动作、warm start 和参考轨迹；
- 50 步预测 horizon、动作范围和 cost 定义；
- 每轮候选数量 256，除非实验明确研究 sample-count scaling；
- 采样研究不使用 Query，以避免 learned-model error 混入 proposal 对比；
- 在线 MPPI 和离线评估均为 PyTorch，JAX MPPI 不使用、不维护。

允许优化的部分是候选动作或 8 个 action knots 的生成和分配策略，例如非高斯分布、
混合分布、低差异序列、学习式 proposal 或自适应 covariance。比较时不能改变模型、
reference 或 cost 权重来获得表面改善。

## 固定模型和场景

固定 rollout 是 `TorchDynamicBicycleRolloutBackend`，参数为 `dt=0.05 s`、
`LF=0.1008 m`、`LR=0.1092 m`、质量 4 kg、摩擦系数 0.8、油门比例 8、转角
比例 0.36 和偏置 0.025 rad。参考赛道是
`car_planner/assets/cuc_inside.csv`，目标速度 2.0 m/s。仿真和控制器输入均不加
观测噪声。

Torch DBM 与 numeric simulator 使用相同方程、参数和 RK4 积分。rollout 会使用
仿真器观测到的真实横向速度 `vy` 初始化隐藏六维状态，不再沿用旧五维接口中的
`vy=0` 近似。50 组随机状态/动作的一步 Torch/JAX DBM 对照最大绝对差为
`2.38e-7`，即 float32 精度下数值一致。

固定快照来自 DBM-MPPI 实时仿真的第 340 个控制步，即仿真时间 17.0 s：

```text
full state [x, y, yaw, vx, vy, yawrate]
[2.8622200, 3.7152741, -2.6533661, 1.8927304, -0.0150235, -0.4586422]

current action [acceleration, steering]
[0.4546806, -0.1695542]
```

对应 Frenet 状态为 `s=32.36438 m`、横向偏差 `-0.00344 m`、航向偏差
`-0.07890 rad`。

## 固定快照

```text
outputs/mppi_sampling_snapshot/live_dbm_clean_step0340/snapshot.npz
outputs/mppi_sampling_snapshot/live_dbm_clean_step0340/candidate_costs.csv
outputs/mppi_sampling_snapshot/live_dbm_clean_step0340/summary.json
```

`snapshot.npz` 的主要数组：

| 名称 | 形状 | 含义 |
|---|---:|---|
| `initial_state` | `[5]` | MPPI 公共状态 `[x,y,yaw,vx,yawrate]` |
| `initial_state_six` | `[6]` | DBM 完整状态，包含真实 `vy` |
| `initial_lateral_velocity` | `[]` | rollout 初始横向速度 |
| `current_action` | `[2]` | 采样前的当前控制量 |
| `history` | `[1,250,7]` | 固定原始 transition history |
| `reference` | `[51,4]` | 当前点加未来 50 点参考轨迹 |
| `mean_knots_before` | `[8,2]` | 采样前 warm-start 均值 |
| `rng_state_before` | `[16]` | PyTorch CUDA generator 状态 |
| `sampled_knots` | `[256,8,2]` | 原始 MPPI knots |
| `sampled_action_sequences` | `[256,50,2]` | 插值后的候选动作 |
| `predicted_trajectories` | `[256,50,5]` | DBM rollout 的 cost 状态 |
| `cost`、`weight` | `[256]` | 总 cost 和 MPPI 权重 |
| `cost_*` | `[256]` | 各个加权 cost 分项 |

该快照是在一次控制调用内直接保存的原子快照，不是从 bag 近似重建。

## Cost 定义

每条候选的总 cost 是 50 步加权和：

```text
5.0 * squared position error
+ 5.0 * squared wrapped-yaw error
+ 1.0 * squared vx error
+ 0.05 * squared acceleration change
+ 0.10 * squared steering change
```

当前 yaw-rate 权重为 0。动作变化的第一步以前一时刻的 `current_action` 为基准。

## 当前高斯采样基准

参数：256 条候选、8 knots、`noise_sigma=[0.25, 0.35]`、temperature 1.0、
seed 3407。候选 0 固定保留 warm-start 均值。

| 指标 | 数值 |
|---|---:|
| 最优候选索引 | 123 |
| 最优 cost | 5.0854864 |
| 平均 cost | 460.4480 |
| 中位 cost | 253.9471 |
| P95 cost | 1705.3604 |
| 最大 cost | 2994.5559 |
| Effective sample size | 1.4419617 |
| 最优候选权重 | 0.8111926 |

最优 cost 分项为：位置 1.3326011、航向 1.6211988、速度 2.0853028、加速度
变化 0.0160810、转向变化 0.0303029。分项之和与总 cost 完全一致。

ESS 接近 1；最优候选和保留的均值候选合计占 `99.9517%` 权重，说明当前高斯
proposal 在这个状态下仍产生了大量低贡献样本。这是采样生成优化的首要基准，
但单点改善不能代替完整闭环评估。

Cost 分布、典型动作序列和 DBM 预测轨迹的综合可视化：

```text
outputs/mppi_sampling_snapshot/live_dbm_clean_step0340/sampling_cost_and_trajectories.png
outputs/mppi_sampling_snapshot/live_dbm_clean_step0340/sampling_cost_and_trajectories.svg
```

复现命令：

```bash
python scripts/model_verify/plot_mppi_sampling_snapshot.py
```

当前图选择最优、warm-start、按总 cost 排序的 P25/P50/P75/P95 和最差候选；
面板包含 cost 排名/累计权重、XY rollout、转向序列以及预测 yaw、vx、yaw-rate。
各状态面板的淡色背景线表示全部 256 条候选预测。

## 轨迹误差引导的两轮采样实验

已经在上述固定 DBM 快照实现一个不需要模型梯度的两轮 sampler。实现入口：

```text
scripts/model_verify/guide_mppi_sampling_from_trajectory_error.py
```

算法保持原 cost 不变，并把它精确表示为一个残差向量的平方和。残差包含 50 步
加权后的 `x/y/yaw/vx` 跟踪误差以及 acceleration/steering change，共 300 维。
第一轮把 8×2 knots 的标准化扰动记为 `ΔU`，利用全部预测轨迹拟合：

```text
residual_i - residual_warm ≈ ΔU_i @ empirical_response
```

拟合使用 ridge regression 和按 cost 平滑衰减的样本权重。随后在 16 维 knot
空间求解带阻尼的 Gauss-Newton 修正，并用每维一个原始 sigma 的 trust region
限制修正幅度。DBM 仅进行 forward rollout，仍保持 `torch.no_grad()`；没有读取
DBM Jacobian，也没有执行 autograd backward。

等 rollout 预算设置：

| 项目 | 高斯基准 | 轨迹误差引导 |
|---|---:|---:|
| 总候选 rollout | 256 | 128 第一轮 + 128 第二轮 |
| 第一轮 sigma | `[0.25, 0.35]` | `[0.25, 0.35]` |
| 第二轮 sigma | 不适用 | 原 sigma 的 0.10 倍 |
| seed | 3407 | 3407 |

第一轮使用 warm-start、一个附加探索样本和 63 对 antithetic 扰动。第一轮最优
仍是 warm-start，cost 为 `6.545824`；换言之，第一轮没有偶然找到比原基准
`5.085486` 更好的候选。但利用这些样本的逐时刻误差信息拟合响应后，引导中心
cost 直接降到 `2.149710`。这说明改善来自样本响应信息，而不是在第一轮最优
样本附近缩小方差。

固定状态结果：

| 指标 | 256 高斯基准 | 两轮引导合并 256 | 引导第二轮 128 |
|---|---:|---:|---:|
| 最优 cost | 5.085486 | **2.149710** | **2.149710** |
| median cost | 253.9471 | **30.0786** | **7.7469** |
| P95 cost | 1705.3604 | **997.7774** | **25.4417** |
| ESS | 1.44196 | **30.75674** | **30.71143** |
| cost < 5 | 0 | **39** | **39** |
| cost < 10 | 2 | **82** | **81** |
| cost < 20 | 5 | **116** | **113** |
| cost < 50 | 29 | **136** | **126** |

相同总候选预算下，最优 cost 降低 `57.73%`，合并候选 median 降低
`88.16%`，`cost < 10` 的候选由 2 条增加到 82 条。引导最优的 cost 分项：
position `1.148002`、yaw `0.173789`、vx `0.815009`、acceleration change
`0.005209`、steering change `0.007701`。

另外分别对原基准和引导候选的 MPPI 加权输出序列做了一次诊断 rollout；这两次
验算不计入 256 条候选预算。原加权序列 cost 为 `5.084162`，第一动作
`[0.289171, -0.154039]`；引导后加权序列 cost 为 `2.031788`，第一动作
`[0.601358, 0.033941]`。因此固定 horizon 的预测目标改善明显，但第一动作变化
较大，必须通过后续闭环实验确认稳定性。

实验输出：

```text
outputs/mppi_sampling_snapshot/trajectory_error_guided_dbm_step0340/
  guided_sampling.npz
  summary.json
  trajectory_error_guided_sampling.png
  trajectory_error_guided_sampling.svg
```

综合图采用 3×3 布局。前两行与原始采样可视化保持一致：第一行是 XY rollout、
完整 acceleration sequence 和完整 steering sequence，第二行是预测 yaw、vx 和
yaw-rate。每个面板同时画出原始高斯候选、引导第二轮候选、warm-start、原始
best、引导 best 以及可用的 reference。最下面一行只放对比统计：等预算 cost
rank、逐时刻 tracking cost 和各个低 cost 阈值下的有效候选数量。

复现：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate anycar
source /home/plusai/anycar/set_env.sh
cd /home/plusai/anycar

python scripts/model_verify/guide_mppi_sampling_from_trajectory_error.py
```

当前结论仅针对一个固定状态。两轮方法虽然保持总 rollout 数相同，但第二轮依赖
第一轮结果，存在串行延迟；当前实现还没有接入 ROS 在线 MPPI。它也属于
sample-based optimizer，而不是带 proposal importance correction 的严格 MPPI
推导。下一步应先扩展到多个直道、缓弯和急弯快照，再决定在线集成参数。

## 256 条 rollout 拆成 1～4 轮的对比

在同一个无观测噪声 DBM 快照上，保持当前 **8 个时间 knot × 2 个控制通道 =
16 个标量优化变量**，并把每次控制更新允许的 DBM rollout 总数严格固定为 256。
为了让轮数成为唯一主要变量，四种方案都使用设备无关的 NumPy antithetic
扰动，并在后续轮次保留当前全局最优作为额外锚点。每个非末轮用该轮候选的
逐时刻轨迹误差拟合经验响应，再执行带阻尼的 Gauss-Newton 中心修正；末轮样本
按 MPPI 权重产生最终控制序列。

| 串联轮数 | 每轮候选数 | 每轮 sigma 比例 |
|---:|---|---|
| 1 | `[256]` | `[1.0]` |
| 2 | `[128,128]` | `[1.0,0.1]` |
| 3 | `[86,86,84]` | `[1.0,0.316,0.1]` |
| 4 | `[64,64,64,64]` | `[1.0,0.464,0.215,0.1]` |

sigma 比例在 1.0 到 0.1 之间按几何级数下降，实际基础 sigma 仍为 acceleration
`0.25`、steering `0.35`。主种子 `3407` 的结果如下：

| 轮数 | 全 256 条最优 cost | 末轮加权输出 cost | 末轮 median / P95 | 全预算 cost < 10 |
|---:|---:|---:|---:|---:|
| 1 | 6.545824 | 6.544600 | 236.022 / 1421.566 | 1 |
| 2 | 2.122387 | 2.498792 | 6.186 / 22.976 | **92** |
| 3 | **1.952986** | **1.954943** | **5.207** / **19.458** | 82 |
| 4 | 1.965111 | 1.966774 | 5.434 / 20.509 | 77 |

主种子上 3 轮略优，但单个种子不能代表稳定性。对种子 `3407,1,...,9` 重复后：

| 轮数 | 最优 cost（mean ± std） | 加权输出 cost（mean ± std） | 末轮 median mean | 赢得种子数 |
|---:|---:|---:|---:|---:|
| 1 | 6.545824 ± 0.000000 | 6.175036 ± 0.465696 | 274.104 | 0 / 10 |
| 2 | 2.256028 ± 0.322769 | 2.369871 ± 0.382855 | 6.816 | 0 / 10 |
| 3 | 2.040131 ± 0.081020 | 2.036745 ± 0.084890 | 5.486 | 3 / 10 |
| 4 | **1.998966 ± 0.041673** | **1.999522 ± 0.040762** | **5.409** | **7 / 10** |

结论分两层：两轮已经获取绝大部分收益，并产生最多的 `cost < 10` 候选；三轮在
解质量和串行深度之间最均衡，可作为后续闭环实验的默认设置。四轮的平均最优
cost 比三轮再低约 `0.0412`（约 `2.0%`），且标准差约减半，适用于更重视最优
质量和重复性的场景，但它多一层必须等待前轮结果的串行延迟。当前 Python 计时
包含首次 CUDA 调度和绘图外的解释器开销，不作为实时控制周期结论；在线接入前
需要在控制进程中 warm-up 后单独测端到端延迟。

此前快照保存的原始 Gaussian 256 条候选最优 cost 为 `5.085486`，图中仅将它画
作历史参考。它与本实验设备无关的 antithetic 候选不是同一组随机样本，因此不能
用它替代上述受控的 1～4 轮横向对比。

实验输出：

```text
outputs/mppi_sampling_snapshot/guided_stage_count_comparison_dbm_step0340/
  primary_seed_results.npz
  primary_seed_comparison.csv
  summary.json
  guided_stage_count_comparison.png
  guided_stage_count_comparison.svg
  guided_cost_progression.csv
  guided_cost_progression.png
  guided_cost_progression.svg
```

`guided_cost_progression` 不做跨 seed 平均，只使用主种子 `3407` 保存的逐轮候选。
左图表示每次串联更新后截至当前找到的最小 cost；右图表示截至当前全部候选的
P10 cost，即 cost 从低到高的第 10 百分位门槛。对应绘图脚本为
`scripts/model_verify/plot_guided_mppi_cost_progression.py`。

复现：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate anycar
source /home/plusai/anycar/set_env.sh
cd /home/plusai/anycar

python scripts/model_verify/compare_guided_mppi_stage_counts.py
```

## 自适应预算和 trust-region sigma

固定轮数实验表明第二次中心更新后改善趋缓，但 DBM 梯度 oracle 在同一 16 维
knot 空间可以将主种子 cost 从 `1.952986` 继续降到约 `1.78643`，因此当前平台
不是 16 维参数化的硬上限。为继续保持 Query/ONNX 可迁移性，正式实验仍只使用
forward rollout，不读取模型梯度。

自适应策略保持总预算 256：

1. 前两轮使用 `[96,80]`，第一轮拟合误差决定第二轮 sigma；
2. 第二轮比较经验响应预测的中心降幅和实际中心降幅；
3. 若相对实际改善至少 `0.15`、实际/预测降幅比至少 `0.25` 且拟合误差不超过
   `0.45`，把剩余 80 条拆成 `[40,40]`，否则一次使用 `[80]`；
4. 后续 sigma 使用 trust-region 风格反馈缩放。若上一次中心失败，但最新拟合误差
   不超过 `0.20` 且预测相对改善至少 `0.20`，优先执行强恢复收缩，避免在已经找到
   好中心后继续用过宽分布采样。

主种子 `3407` 自动选择 `[96,80,40,40]`，sigma 比例为
`[1.0,0.31765,0.11118,0.06]`：

| 累计 rollout | 当前最优 cost | 累计 P10 cost |
|---:|---:|---:|
| 96 | 6.545824 | 36.095955 |
| 176 | 2.829360 | 9.701776 |
| 216 | 1.936044 | 4.135484 |
| 256 | **1.929329** | **2.822317** |

固定 3 轮主种子最终 best/P10 为 `1.952986/3.118384`，固定 4 轮为
`1.965111/3.281568`。自适应末轮自身 P10 为 `2.153477`，也低于固定 3 轮的
`2.425519` 和固定 4 轮的 `2.331210`。

10 个种子的最终统计：

| 策略 | best cost（mean ± std） | 加权输出（mean ± std） | 累计 P10 | 末轮 P10 |
|---|---:|---:|---:|---:|
| 固定 3 轮 | 2.040131 ± 0.081020 | 2.036745 ± 0.084890 | 3.520630 | 2.563112 |
| 固定 4 轮 | **1.998966 ± 0.041673** | 1.999522 ± 0.040762 | 3.635813 | 2.490391 |
| 自适应 | 2.000194 ± 0.069161 | **1.998655 ± 0.069386** | **3.311430** | **2.253711** |

逐种子比较中，自适应相对固定 3 轮的 best 和加权输出均为 `10/10` 改善，累计
P10 为 `7/10` 改善，末轮 P10 为 `9/10` 改善；相对固定 4 轮则分别为
`6/10`、`6/10`、`7/10` 和 `9/10`。因此它已经稳定超过固定 3 轮，并在平均
best 与固定 4 轮基本相当的同时改善低 cost 样本集中度，但 best 方差仍高于固定
4 轮。9 个种子选择四轮、1 个种子选择三轮，所以当前收益来自预算和 sigma 的
利用效率，而不是串行深度下降。

### 当前暂停点（2026-07-31）

- 固定基准继续使用无观测噪声的 clean DBM step 340 快照、8 个时间 knot / 16 个
  标量变量和每次严格 256 条 rollout；
- 自适应方案相对固定 3 轮改善稳定，但相对固定 4 轮的平均 best cost 仅为
  `2.000194` 对 `1.998966`，差异很小，且自适应 best 方差更大；
- 多数种子仍执行四轮串联，因此当前方法没有解决串行推理延迟，只改善了低 cost
  样本的集中度和部分种子的最终解；
- 自适应引导保持为离线研究脚本，**尚未接入 ROS `car_node` 的在线 MPPI**；在线
  Quick Start 仍使用现有 Torch MPPI 采样流程；
- 当前阶段停止继续微调该固定快照。若恢复研究，应先增加多个直道、缓弯和急弯
  快照并做完整单圈闭环，再根据跨场景收益决定是否集成，而不是继续针对 step 340
  调阈值。

实验输出：

```text
outputs/mppi_sampling_snapshot/adaptive_guided_dbm_step0340/
  summary.json
  primary_adaptive_results.npz
  primary_cost_progression.csv
  adaptive_guided_comparison.png
  adaptive_guided_comparison.svg
```

复现：

```bash
python scripts/model_verify/adaptive_guided_mppi_sampling.py
```

## 复算和新采样器接口

复算快照内的原始候选：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate anycar
source /home/plusai/anycar/set_env.sh
cd /home/plusai/anycar

python scripts/model_verify/evaluate_mppi_sampling_snapshot.py
```

当前独立复算与在线保存的 256 条 cost 最大绝对差为 0。

新采样器输出一个 float32 `.npy`，形状为 `[N,50,2]`，动作必须位于
`[-1,1]`：

```bash
python scripts/model_verify/evaluate_mppi_sampling_snapshot.py \
  --candidates path/to/candidate_actions.npy \
  --output-dir outputs/mppi_sampling_snapshot/my_sampler
```

评估结果保存每条候选的 rollout、总 cost、cost 分项和重新归一化后的 MPPI 权重。

## 代码状态

- `controllers_torch/mppi.py` 可以返回全部 sampled sequences、rollouts 和 cost 分项；
- `car_node.py` 支持用 `mppi_snapshot_step`、`mppi_snapshot_dir` 原子截取快照；
- `controllers_torch/dbm.py` 使用观测到的真实 `vy` 初始化 DBM rollout；
- `evaluate_mppi_sampling_snapshot.py` 默认用固定 DBM 输入评价任意候选动作序列；
- snapshot 默认关闭：`mppi_snapshot_step=-1`，正常 Quick Start 不写文件；
- 观测噪声注入已从仿真器移除；
- Query 的 PyTorch/ONNX rollout 仍可用于后续迁移验证，但不是采样研究主基准；
- 旧 Query 快照保留在 `live_query_clean_step0340`，不与 DBM 基准混用；
- JAX MPPI 不在维护范围内。

## 后续比较要求

对每种新 proposal 至少报告：最优 cost、cost 中位数/P95、ESS、低 cost 样本数量、
采样生成耗时和总控制耗时。固定状态筛选通过后，再扩展到多个直道/缓弯/急弯快照，
最终必须回到完整单圈闭环比较，避免只对单个状态过拟合。
