# AnyCar / Query 工作接续摘要（2026-08-31）

> 本文是新 Codex 会话的首要入口，只保留当前仍有效的 AnyCar / Query / DBM / MPPI / OAC 技术状态。开始工作前先运行 `git status --short`，再按需查阅本文引用的 review 小节；不要把旧结论重新当成当前结论。

## 1. 目标与架构

### 1.1 当前真正要解决的问题

长期目标是在 MPPI 控制器前加入一个轻量 Actor：它根据当前车辆状态、历史、参考轨迹和当前控制上下文，一次前向输出一个更好的 **MPPI sampling center**。Actor 只负责 proposal；MPPI 仍负责对候选轨迹做模型 rollout、算 cost、加权/选择并输出最终控制。

用户最终关心的速度域是 **40--100 km/h**。当前阶段不是继续优化低速 fixed-DBM 指标，而是：

1. 冻结已经接近 fixed-DBM 数值上限的高速 Actor 合同；
2. 把 rollout/reward 环境切换到冻结 Query；
3. 先证明 Query 在高速输入上的数值行为可用，再在 Query cost 下重建 Critic/Replay/OAC；
4. 最后验证 Actor proposal 的收益能否经 fixed-budget two-center MPPI 传到最终输出和闭环。

> **2026-09-03 当前目标覆盖说明（后续会话必须优先读取）**：上面的阶段列表是早期路线背景；当前
> Query-environment 已完成输入/数据重建、Replay/Critic/OAC 与 train-side 独立验证。现在的主目标是
> 提高和衡量**单头 Actor 一次直接输出相对同状态 deterministic warm 的总体 Query cost 收益**。
> 主选择指标固定为 Actor mean J、mean gain `J_warm-J_actor` 和 aggregate improvement；P05、worst、
> 困难 slice 只作诊断，不得再否决主体更好的候选。warm 不进入 Actor 输入或训练目标，outer/formal/test
> 仍封存。当前基线为 pooled inner Actor/warm mean J=`3.0888/5.3516`、aggregate=`+42.28%`；最新
> Actor-centered headroom audit 又找到 best mean J=`1.8574`、相对 Actor residual aggregate=`39.87%`，
> 因此下一主线是按单变量顺序改善 Actor 对这些低cost区域的吸收，先做39→65 candidate
> search-recentered bank A/B，再做continuous cap-only与LR检查。完整依据见§17.25--17.26。

### 1.2 控制与学习链路

```text
state/history/reference/current action
                + warm/current center (8 x 2)
                              |
                              v
              Actor proposal center (8 x 2)
                              |
                 knots 线性插值为 [50, 2]
                              |
       warm 与 Actor 周围的 fixed-budget candidate bank
                              |
       fixed DBM / Query PyTorch / Query ONNX rollout
                              |
                deterministic J50 candidate cost
                              |
                   MPPI softmax / weighted sequence
                              |
              final-output fallback / first command
                              |
          闭环执行一步，warm shift/recede，下一周期重复
```

当前高速 OAC 是在记录状态上反复 query 模型的 **contextual-bandit 式在线循环**：没有 next-state Bellman bootstrap、target critic 或熵项。所有 Actor 新动作（包括差动作）都进入 Replay；Twin absolute-value Critics 持续更新；Actor 用 Critic 梯度更新；checkpoint 由真实 deterministic rollout cost gate 选择。虽然文档沿用 OAC/SAC 术语，它不是标准长时序 SAC。

切到 Query 后，Query forward 是环境 cost oracle；不得把 DBM cost、DBM 梯度或 DBM 标签混入 Query 目标。DBM Actor/checkpoint最多作为初始化或对照。

### 1.3 当前实验边界

- **已完成**：train-only 高速 fixed-DBM 数据、teacher、预训练、episode-grouped OOF OAC、单中心 numerical best-found 审计、Query 训练域/高速输入合同审计。
- **未完成**：高速 Query forward 数值门、Query 高速 PyTorch/ONNX parity、Query-relabel OAC、当前高速 Actor 的 fixed-budget two-center wrapper OOF、可部署 full-train 单 checkpoint、高速闭环、formal validation/test、实车。
- fixed DBM 的“perfect model”只指同一代码/参数的数值仿真，不代表真实车辆。
- open-loop direct-center 改善不等于闭环改善；Query 内部优化也不等于真实车辆改善。
- formal validation/test 仍封存；在 Actor/Query/wrapper 合同冻结前不得消费。

## 2. Query / DBM

### 2.1 fixed DBM

实现：

- `car_dynamics/car_dynamics/controllers_torch/dbm.py`
- rollout backend：`TorchDynamicBicycleRolloutBackend`

它是离散动态自行车公式放在 Torch 中做批处理、设备执行和离线 autograd，不是神经网络。完整内部状态为：

```text
[x, y, yaw, vx, vy, yaw_rate]
```

公开的 Query-compatible 状态为：

```text
[x, y, yaw, vx, yaw_rate]
```

缺失的 `vy` 通过 backend 的 `set_initial_lateral_velocity` 注入。参数合同：

- `dt=0.05 s`，horizon `50`（2.5 s）；
- `lf=0.1008 m`，`lr=0.1092 m`，wheelbase `0.21 m`；
- mass `4.0 kg`，`Iz=0.07`，friction `0.8`；
- steering scale `0.36`、bias `0.025`；throttle scale `8.0`；rolling friction `0.05`；
- Pacejka `B=20, C=1`；RK4。

`rollout_full_state_differentiable` 只用于离线数值上限/机制审计。部署兼容 OAC 不得依赖 DBM 解析梯度，因为 Query 无此接口。

### 2.2 Query 模型、输入输出与调用链

模型类：`TorchTransformerDecoderKinematicQueryMLP`。它是 **kinematic nominal rollout + learned residual**，不是 DBM 蒸馏的纯黑盒轨迹网络。

运行入口：

- `car_foundation/car_foundation/query_deployment.py`
- `QueryDeploymentModel`
- `TorchQueryRolloutBackend`
- `OnnxQueryRolloutBackend`
- `QueryHistoryBuffer`

张量合同：

| 项 | shape | 内容 |
|---|---:|---|
| history | `[1,250,7]` | `[dx_body, dy_body, dyaw, dvx, dyaw_rate, acceleration, steering]` |
| initial observable | `[1,5]` | `[x,y,yaw,vx,yaw_rate]` |
| current action | `[1,2]` | `[acceleration, steering]` |
| future actions | `[N,50,2]` | 候选控制序列 |
| output | `[N,50,5]` | absolute `[x,y,yaw,vx,yaw_rate]` |

约定：

- history `250` 步 = 12.5 s，预测 horizon `50` 步 = 2.5 s，`dt=0.05 s`；
- 数据为 pre-transition state/action 对齐；`steer_shift=0`；
- 每步先做 nominal kinematic rollout，再预测 residual `[dx_body,dy_body,dvx,dyaw_rate]`；
- checkpoint 固化 history/context/residual/nominal-state/transition normalization；
- context 为 `[vx,yaw_rate,first_future_accel,current_steer]`；
- Query 不显式观察 `vy`；reference 也不是 Query 输入，reference 只进入外层 MPPI cost；
- reference 接受 `[50,4/5]` 或 `[51,4/5]`。若为 51 行，当前行被丢弃，future `[1:]` 与 50 个预测状态计 cost；
- ONNX candidate batch 是动态维，opset 17。

### 2.3 当前 Query checkpoint、训练数据与适用域

small-car checkpoint：

```text
/home/plusai/anycar/outputs/formal_small_car_query_dt005/20260730T144840/query_best.pt
SHA256 932917c751d2ab5697d956cd5192e3557374e8cc416fd1f0aba7dfab17b49e66
```

ONNX：

```text
/home/plusai/anycar/outputs/formal_small_car_query_dt005/20260730T144840/anycar_query.onnx
SHA256 8049294c1ccae80d9d83f4a6b5401ca199bbc9068d089b91d6f6aed233605674
```

训练数据：

```text
/disk/collect_data_from_anycar/generated_small_car_query_dt005/20260730T144638
```

数据合同：256 PKL、1024 episode、每 episode 1000 步（50 s），由 deterministic Torch DBM 生成；无观测噪声；seed 3407。target-speed knots 大致为 `[0.2,3.5] m/s`，每 8 个 episode 有 zero-start 子集；initial `vx` `[0,2.5] m/s`，每 4 个 episode 有 zero-start；控制 knot 间隔 25 步（1.25 s）；steer 主要 `[-0.8,0.8]`；acceleration 是速度误差 PD 加 excitation。

训练实际 `vx` 为 `[-0.0266,3.0979] m/s`，P95 `2.4605 m/s`。epoch 30 被选择；模型/训练参数详见 `car_foundation/docs/small_car_query_dt005_20260730.md`。

原始训练命令：

```bash
python scripts/model_verify/train_kinematic_residual_ablation.py \
  --dataset-path /disk/collect_data_from_anycar/generated_small_car_query_dt005/20260730T144638 \
  --models query --include-kinematic \
  --epochs 40 --val-every 5 --early-stopping-patience 4 \
  --selection-metric rollout_score --rollout-loss-weight 0.01 \
  --wheelbase 0.21 --dt 0.05 \
  --steering-ratio 2.7777777777777777 \
  --steering-offset-deg -3.978873577297384 \
  --steer-shift 0 --batch-size 128 --eval-batch-size 256 \
  --num-workers 4 --output-dir outputs/formal_small_car_query_dt005
```

独立低速 50-step 指标：position RMSE `0.2049 m`、yaw RMSE `0.1185 rad`、vx RMSE `0.0505 m/s`、yaw-rate RMSE `0.0974 rad/s`。历史低速整圈结果：Query lateral MAE/RMSE `0.109/0.137 m`、speed MAE `0.232 m/s`、约 `28.7 ms`；DBM 对照 `0.080/0.107 m`、`0.225 m/s`、约 `64.6 ms`。这些只是低速结果。

低速 PyTorch/ONNX parity：CPU max abs `2.74e-6`；CUDA mean/max abs `4.97e-5/1.84e-3`。高速 parity 尚未执行。

单 seed 少量观测噪声诊断曾得到 Query lateral MAE 约 `0.138 m`（无噪约 `0.107 m`），DBM 约 `0.077 m`；它只是 smoke test，不是鲁棒性结论。

### 2.4 高速 Query 域审计

脚本与产物：

```text
scripts/model_verify/analyze_highspeed_query_domain_contract.py
scripts/model_verify/validate_highspeed_query_domain_contract.py
outputs/query_mppi/highspeed_query_domain_audit_20260831_v3/analysis.json
outputs/query_mppi/highspeed_query_domain_audit_20260831_v3/validator_report.json
```

validator：17/17 PASS。正式 qualification：

```text
QUERY_INTERFACE_MATCH_SPEED_NORMALIZATION_SEVERELY_OOD
```

已确认接口、车辆几何、时间步、action 顺序/范围、history/horizon、pre-transition 对齐和 `steer_shift` 匹配；但内容域严重不匹配：

- 当前高速 replay 实际 `vx=[7.796,30.626] m/s`，median `17.558 m/s`，100% 超过 Query 训练最大速度；
- 固定 checkpoint normalization 下，context-vx absolute z median/P95 `25.76/43.44`；history-dx `25.87/43.80`；warm nominal-dx `25.61/43.92`；
- 当前 `vy=[-4.228,5.702] m/s`，50.5% 超训练范围，Query 又没有显式 `vy`；yaw-rate 仍在训练 min/max 内；
- 当前 replay 每 episode 只取控制 step 0--4。step 0 用 constant-motion/current-action 填满 250 history，最多只有 4 个真实 transition；history accel/steer 的精确零比例 99.2%，训练历史接近 0%。

结论不是“Query 禁止使用”，而是：如果把冻结 Query 本身定义为环境，可以不拿 DBM/真实轨迹否决它；但必须先验证它在高速 OOD 下仍然 finite、deterministic、可复算，且 PyTorch/ONNX 一致。若未来要上实车，Query-to-real fidelity 是另一道后续门。

### 2.5 cost、trajectory relabel 与数据约定

当前 deterministic J50：

```text
5.0 * sum(position_xy_error^2)
+ 5.0 * sum(wrapped_yaw_error^2)
+ 1.0 * sum(vx_error^2)
+ 0.0 * sum(yaw_rate_error^2)       # 当前默认
+ 0.05 * sum(accel_rate^2)
+ 0.10 * sum(steer_rate^2)
```

rate cost 包括 current action 到第一个 future action 的差。高速 J 大主要是 2.5 s 内速度、航向、侧偏误差累计后进入平方位置项，不是简单的 normalization bug。

可重标数据应保存 state6、history、reference/reference_ego `[51,4]`、mean knots、raw/clipped knots、sample noise、action sequence `[N,50,2]`、six-state trajectory、未加权 `feature_*`。collection-time scalar cost/weights 仅用于复现，不是永久标签；改 cost weights 可从轨迹/features relabel，但改动力学、状态、噪声、延迟或障碍必须重采。

## 3. MPPI / Actor

### 3.1 当前 MPPI 合同

`TorchMPPIParams` 当前核心值：horizon 50、history 250、state 5、action 2、`dt=0.05`、默认 `N=256`、iteration 1、knots 8、temperature 1、mean-update 1、Gaussian sigma `(accel,steer)=(0.25,0.35)`、action box `[-1,1]`。candidate 0 是未加噪 current mean。

fixed Hadamard-64 是 opt-in：只对 `N=64`，radii `(0.10,0.30)`。8 knots 在 50 步上做线性插值。非均匀 knots 暂不采用：B1 projection oracle 中 front/mid/rear-dense 相对 uniform 分别增加 mean cost 约 `+6.9/+28/+186`，且 J16 到 J100 的差仅约 `0.198`；uniform 8 knots 不是当前瓶颈。

softmax：`w_i = exp(-(J_i-J_min)/temperature)`，输出是候选 action sequence 的加权和。固定低速 snapshot 曾出现 ESS `1.062`、best weight 约 `0.97`，说明实际可接近 argmin；但不能把该单 snapshot 外推为所有场景。

### 3.2 三种 center/output 必须分开

- **warm/current center**：上一周期 MPPI mean shift/recede 后、当前采样前的 `8x2` center。
- **Actor proposal center**：Actor 一次前向给出的 absolute `8x2` sampling mean；不是最终控制。
- **final executed output**：候选经 rollout、cost、softmax 后的 weighted sequence，或 fallback 选中的 sequence；只执行其第一个 action。

指标定义：

- **Actor direct cost**：Actor `8x2` 插值到 `[50,2]` 后直接 rollout 的 deterministic J50；不含采样和 softmax。
- **guarded direct cost**：同一模型下 `min(J_warm_center,J_actor_center)`。
- **wrapper cost**：围绕一个或多个 center 做随机/CRN sampling 和 softmax 更新后的 cost。
- **final-output cost**：真实 weighted/proposal/fallback sequence 的模型 cost，最接近最终执行链。

`warm center` direct J 在 state/center/model/reference 固定时是确定性的；warm wrapper/随机 best sample 依赖 RNG。评价 sampling center 本身时，应以 deterministic warm center 为相对基线，不能把随机 `J_w` 或采样池最优混进 Actor loss/center 指标。

### 3.3 指标解释

定义 `gain = J_warm_center - J_actor_center`，正值表示 Actor center 更好。

- mean：总体总 cost/gain；高速或高 cost 状态权重大。
- median：典型状态；不能说明尾部安全。
- P05 gain：下 5% 分位；`>=0` 只表示约 95% 样本不回归，不是逐状态保证。
- worst：最差单状态 gain；对异常和抽样敏感，但能暴露灾难 proposal。
- win rate：`J_actor <= J_warm` 的比例。
- fallback rate：最终选择 warm 的比例；高值可能说明安全有效，也可能说明 Actor 实际贡献很少。
- 必须同时报告 speed/scenario 分层、episode bootstrap CI，禁止逐状态或逐 seed 选最优结果。

### 3.4 two-center guard 与下界

推荐的 fixed-budget 对照是 warm-only 64 vs warm 32 + Actor 32，并在两边保留各自 exact center，使用 CRN。仅把 warm 放进候选池 **不能** 保证 softmax weighted output 的 cost 不高于 warm，因为权重平均可能被差候选稀释。

严格下界需要 final-output fallback：在同一 rollout 模型、同一 reference/cost 下，分别评价 exact warm final sequence 和 proposal/final sequence；若 proposal 更差就返回 warm。

只有同时满足以下条件，才可以降低 Actor 尾部优化的优先级：

1. warm 与 Actor 使用同一模型、reference 和 cost；
2. warm exact candidate 保留；
3. 比较的是 exact final sequences，而不是 center proxy；
4. fixed total sampling budget、CRN/统计功效一致；
5. model-relative warm violation 为 0；
6. fallback 不引入不可接受的算力、约束违规、抖动或频繁切换；
7. 短闭环验证通过。

它只能保证 **模型相对、当前步、已计入 cost/constraint 范围内** 的 warm 下界；不能保证真实车辆、Query 模型误差下的 DBM/真实 cost、长期递推稳定性、未建模约束或 half-budget 后 warm 分支的采样质量。因此“只守住 warm-relative 下界”足以把 Actor 定位为可拒绝 proposal，但不足以授权部署。

当前尾部结论：高速 fixed-DBM Actor 主体已接近数值上限，继续做通用 tail loss/结构扫描优先级低；但完整 OOF 仍有平均 1.39% 回归、strict worst `-34743.9`，所以 warm guard 是承重结构，不能删除。

## 4. 数据与实验

### 4.1 高速 fixed-DBM 数据

采集目录：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/fixed_dbm_highspeed_expansion_20260830_v2
```

计划：

```text
scripts/model_verify/fixed_dbm_highspeed_expansion_20260830_v2.json
```

compact replay：

```text
outputs/mppi_proposal/highspeed_initial_dbm_replay_expansion_20260830_v1/replay.npz
outputs/mppi_proposal/highspeed_initial_dbm_replay_expansion_20260830_v1/summary.json
outputs/mppi_proposal/highspeed_initial_dbm_replay_expansion_20260830_v1/validator_report.json
```

120 个 train-only episode；5 个目标速度 `40/55/70/85/100 km/h`；6 类场景 `steady/under/over/lateral/heading/combined`；每组合 4 episode；每 episode 取前 5 个 step，共 600 contexts，N=64。

实际 `vx=[28.07,110.25] km/h`，只有 78.33% context 位于物理 40--100 km/h 内，因为包含 recovery/transient。`maximum_replay_vs_trace_best_cost_abs_error=16946.3`：collection-time trace scalar cost 不能继续当当前标签，必须从当前 action/state/reference 确定性重算。validator 只证明 archive/count/contract，不证明旧 scalar cost 精确。

### 4.2 高速 proximal teacher（正式 train-only 机制产物）

```text
outputs/mppi_proposal/highspeed_proximal_teacher_expansion_20260830_v1
```

每状态 129 个 fixed-DBM direct J50 候选：anchor + 64 ring + 32 recentered + 32 recentered，sigma `(0.25,0.35)`，warm 强制保留。600 contexts：warm mean J `191104.2`，teacher `185106.1`，mean gain `5998.1`，median gain `4732.6`，聚合下降 `3.1386%`，100% strict improvement。它是有界局部 teacher，不是全局最优。

### 4.3 高速 Actor/Twin-Critic 预训练

```text
outputs/mppi_proposal/highspeed_actor_twin_critic_pretrain_expansion_e4_20260830_v2
checkpoints/pretrain_fold{0..4}_seed{0..2}.pt
```

episode-grouped 360 fit / 120 selection / 120 OOF。Actor 是 no-anchor clean G-X absolute `8x2`，输出限制为训练 std 的 `+-3`；current 输入 `[vx,yaw_rate,accel,steer]`，无 `vy/beta`。Twin Critic 回归 standardized `log1p(J50)`，无 TD。

Actor OOF teacher-recovery median `0.573`，但 all-run P05 gate 失败；Critic OOF Pearson 约 `0.985`。结论：可作为初始化，不能部署。

### 4.4 已冻结的 Critic 梯度旧主线

单次/冻结 Critic 作为 Actor gradient provider 已永久冻结。关键证据：

- scalar-Q grouped CV 在 hard states 失败；换 DCT/sensitivity 坐标后，physical-gradient P10 仍约 `-0.32~-0.39`，early-steer P10 `-0.66~-0.69`；
- tiny-set overfit 能通过，排除了基础表达、autograd 和 optimizer 完全失效；
- true FD 与逐 cost 项 autograd cosine 中位 `0.997`；nearest-neighbor flips 中 68% 有自身反向的主导 cost 项（同向对仅 4.3%），其中 position 149/170；机制族覆盖 83.2%；position gradient 约 74.6% 质量来自 horizon step 34--50；
- 局部真实梯度受 position 几何分支和 cost 项对消影响，不能靠继续扫 loss/encoder/Hessian 修复。

这不否定持续 OAC：Actor-visited 新动作不断补 Replay 后，Critic 可以学到当前邻域。低速内部实验的 200-round Critic 曾达 value Pearson 约 `0.975`、action-gradient cosine median/P10 `0.979/0.726`、norm ratio `1.13`、early-steer P10 `0.835`；但极小独立 step 仍约 4.2% 状态回归（DBM exact 为 0），所以仍要 gate。

Search-informed Replay 也证明数据可学：额外 1600 frozen critic updates 把 path-sign `0.792→0.865`、bank recovery `0.772→0.920`；但更准 Critic 未自动改善 Actor，主要损失还包括共享 Actor 参数梯度冲突和 update geometry。pairwise-delta head 默认关闭：ranking 改善但尾部变差。

### 4.5 最终高速 OAC 合同与 OOF 结果

冻结训练合同：

- `DirectNoAnchorGTXActor`，absolute `8x2`，Actor 不输入 warm；
- strict-clean 输入只保留 history、ego-frame reference 和 current；ego-reference 每步含 `[x_ref_ego,y_ref_ego,sin(heading),cos(heading),delta_v]`；anchor、feedback、gradient-context 分支被显式置零；
- 8 个 control tokens 使用非因果双向 temporal decoder，并加入各 knot 对齐的 ego-reference geometry；`G-X` 额外加入每个 knot 的 `x_ref_ego`，输出是联合计划而非逐 knot 自回归命令；
- K16 microsteps，最多 160 rounds；
- 每状态/round 65 个 search-recentered candidates；Twin Critic 每 round 20 updates；
- cap-only output move `0.06 sigma RMS`，不强行放大小动作；
- exploration `0.20→0.05`，round 90 结束；second radius `0.70`；
- Actor LR `2e-5`，round 120--160 cosine decay 到 `5e-6`；Critic LR `1e-4`；
- 使用 internal selected checkpoint，不用强制 latest；无 DBM analytic gradient。

15 个 OOF checkpoints：

```text
outputs/mppi_proposal/highspeed_actor_k16_160round_cap006_lrdecay_fold0_20260831_v1/k16/fold0_seed{0..2}/checkpoint.pt
outputs/mppi_proposal/highspeed_actor_k16_160round_cap006_lrdecay_fold1to4_20260831_v1/k16/fold{1..4}_seed{0..2}/checkpoint.pt
```

审计：

```text
outputs/mppi_proposal/highspeed_actor_k16_cap_schedule_20260831_v1/analysis.json
qualification: HIGHSPEED_CAP006_LRDECAY_FULL_FOLD_AUDIT_COMPLETE
```

15 个 episode-grouped OOF run 聚合：Actor mean J 平均 `169579.5`，pretrained `185123.4`，warm `191104.2`；vs warm mean gain `21524.7`，median gain 的 run 均值 `16408.9`，P05 gain 的 run 均值 `3293.8`；15/15 overall P05 为正；win rate 平均 `98.61%`、最小 `95.83%`；平均回归率 `1.39%`；全 run strict worst `-34743.9`。selected round 97--160，median 154。

注意：这是 train-only pool 上的 episode-grouped OOF 机制证据。15 个 fold/seed checkpoint 不是可部署的单一 full-train checkpoint；部分 per-speed run 的 P05 可为负，不能表述为所有速度/seed 都过尾部门。

### 4.6 fixed-DBM numerical best-found 审计

```text
outputs/mppi_proposal/highspeed_final_actor_numerical_oracle_20260831_v1
```

120 个 train-only、速度/场景平衡状态；多起点 projected differentiable DBM 优化（800 steps，LR `0.001/0.01/0.03`）后再做独立正交无梯度 polish。它只是 **numerical best-found reference**，不是数学认证全局最优；参数空间是 uniform-8-knots / 16D / physical action box / DBM J50。

结果：warm mean J `192539.9`；旧 oracle `154546.1`；固定 Actor 三 seed 期望 `150894.8`；best-found `150181.7`。Actor 相对 best-found 聚合差 `0.473%`，episode-bootstrap 95% CI `[0.377%,0.606%]`；warm→best-found headroom recovery `98.32%`，CI `[97.86%,98.65%]`；逐状态 gap median/P95/max `0.449/2.09/8.88%`；79.2%/93.3% 状态位于 1%/2% 内。速度分层 gap：40/55/70/85/100 km/h 为 `1.65/0.60/0.61/0.32/0.32%`。8 个 `>2%` tail（7 个 40、1 个 70）已固化在 `analysis.json/tail_over_2pct`。

结论：fixed-DBM 单中心主体已接近当前数值上限；停止通用 DBM 数据/LR/trust/结构扩张。若以后研究 DBM tail，必须复用这 8 个预注册状态，不能事后重新选样本。

### 4.7 证据等级与已否定方向

正式的 **train-only 机制验证**：高速数据/replay validator、teacher validator、pretrain validator、OAC full-fold/cap validator、numerical best-found exact replay validator、Query domain validator。它们不等于 formal validation/test。

smoke/诊断：单 seed Query 噪声、低速固定 snapshot、旧 hard-guard 闭环 pilot、早期 fold0/LR/K 扫描。不能用来授权当前高速部署。

已否定方向只保留原因：

- 冻结/单次 Critic 梯度：hard tail 的真实局部方向分支敏感，跨状态不可稳健迁移；
- 继续纯 BC/global-J16 distillation：多 basin 与动作精度使监督损失不能稳定变成 rollout 收益；search/BC 可用于初始化和 Replay，但不能替代持续 OAC；
- 更多 uniform DBM 数据/更大 LR/trust/更多轮次：主体已经被数值 best-found 审计判为近饱和；
- sensitivity-normalized Critic/action 坐标：没有结构性修复 gradient tail；
- 非均匀 knots：对 uniform-optimal plan 的表示损失大，且 8-knot 到更高维本身的 headroom 很小；
- pairwise-delta/CVaR 等通用 tail loss：可改善部分排名/尾部，但会损伤主体或饱和，不能替代 guard。

## 5. 代码状态

### 5.1 环境与常用命令

```bash
cd /home/plusai/anycar
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate anycar
source /home/plusai/anycar/set_env.sh
```

ROS/DBM 数据采集还需：

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select car_dynamics car_ros2 --symlink-install
source /home/plusai/anycar/set_env.sh
```

用户 quick start：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 ros2 launch car_ros2 car_sim.launch.py
```

默认 launch 当前仍指向旧 real Query checkpoint：

```text
outputs/formal_real_finetune_query_baseline_split/20260728T143256/query_best.pt
```

因此使用 small-car Query 必须显式覆盖：

```bash
CAR_PATH=/home/plusai/anycar
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 ros2 launch car_ros2 car_sim.launch.py \
  mppi_backend:=pytorch \
  query_checkpoint:=$CAR_PATH/outputs/formal_small_car_query_dt005/20260730T144840/query_best.pt
```

ONNX：

```bash
CAR_PATH=/home/plusai/anycar
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 ros2 launch car_ros2 car_sim.launch.py \
  mppi_backend:=onnx \
  query_checkpoint:=$CAR_PATH/outputs/formal_small_car_query_dt005/20260730T144840/query_best.pt \
  query_onnx_path:=$CAR_PATH/outputs/formal_small_car_query_dt005/20260730T144840/anycar_query.onnx
```

DBM：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 ros2 launch car_ros2 car_sim.launch.py mppi_backend:=dbm
```

标准 mature-history snapshot 采集模板（episode id/目录按计划替换）：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 ros2 launch car_ros2 car_sim.launch.py \
  mppi_backend:=dbm mppi_seed:=3500 \
  mppi_dataset_dir:=/disk/collect_data_from_anycar/mppi_rl_closed_loop/<collection> \
  mppi_dataset_episode_id:=<id> mppi_dataset_start_step:=250 \
  mppi_dataset_stride:=25 mppi_dataset_max_snapshots:=20 \
  mppi_dataset_shutdown_on_complete:=True \
  sim_initial_state:=0,0,0,0,0,0

python scripts/model_verify/validate_mppi_closed_loop_dataset.py \
  /disk/collect_data_from_anycar/mppi_rl_closed_loop/<collection>/<episode>
```

当前高速 600-context 集合刻意取 step 0--4，不符合上述 mature-history 模板；不可无说明地混用。

JAX 路径当前不使用、不维护。

### 5.2 重要代码/文档

- `car_dynamics/car_dynamics/controllers_torch/dbm.py`：fixed DBM、批量/differentiable rollout。
- `car_dynamics/car_dynamics/controllers_torch/mppi.py`：Torch MPPI、Gaussian/fixed-Hadamard bank、direct evaluator、final sequence hard guard。
- `car_foundation/car_foundation/query_deployment.py`：Query PyTorch/ONNX/history 调用链。
- `car_foundation/car_foundation/mppi_proposal_policy.py`：Actor/normalization/policy 定义。
- `car_foundation/car_foundation/mppi_residual_actor_runtime.py`：历史 residual Actor guard runtime；不是当前高速 final OOF Actor。
- `car_ros2/car_ros2/car_node.py`：backend、sampling、采集和历史 DBM hard guard 接入。
- `car_ros2/launch/car_sim.launch.py`：backend/checkpoint/MPPI/采集参数。
- `car_planner/car_planner/global_trajectory.py`：reference speed override。
- `car_foundation/docs/mppi_sampling_center_review_20260812.md`：当前实验权威记录，优先读 §11.141--145。
- `car_foundation/docs/mppi_online_actor_critic_pilot_plan_20260820.md`：当前计划，优先读 §63--67。
- `car_foundation/docs/small_car_query_dt005_20260730.md`：Query checkpoint/数据/接口。
- `car_foundation/docs/mppi_closed_loop_dataset_collection_20260802.md`：snapshot/relabel 合同。

高速关键脚本：

```text
scripts/model_verify/generate_fixed_dbm_highspeed_plan.py
scripts/model_verify/generate_highspeed_initial_dbm_replay.py
scripts/model_verify/generate_highspeed_proximal_search_teacher.py
scripts/model_verify/pretrain_highspeed_actor_twin_critic.py
scripts/model_verify/train_highspeed_actor_visited_oac.py
scripts/model_verify/train_highspeed_actor_k_scan.py
scripts/model_verify/analyze_highspeed_actor_cap_schedule.py
scripts/model_verify/validate_highspeed_actor_cap_schedule.py
scripts/model_verify/run_highspeed_final_actor_numerical_oracle.py
scripts/model_verify/polish_highspeed_final_actor_numerical_oracle.py
scripts/model_verify/analyze_highspeed_final_actor_numerical_oracle.py
scripts/model_verify/validate_highspeed_final_actor_numerical_oracle.py
scripts/model_verify/analyze_highspeed_query_domain_contract.py
scripts/model_verify/validate_highspeed_query_domain_contract.py
```

OAC 长命令不要凭记忆重构，应读取已冻结的 `contract.json`：

```text
outputs/mppi_proposal/highspeed_actor_k16_160round_cap006_lrdecay_fold0_20260831_v1/contract.json
outputs/mppi_proposal/highspeed_actor_k16_160round_cap006_lrdecay_fold1to4_20260831_v1/contract.json
```

### 5.3 Git 与测试状态

当前 HEAD：

```text
ed7e9b3 Add MPPI sampling-center review and pilot target-transfer analysis
```

工作树非常脏：约 15 个 tracked modified、286 个 untracked。绝大多数当前研究代码、文档和 artifact 索引尚未提交；不要 `git add -A`，不要覆盖/回滚用户改动。新会话必须先重新运行 `git status --short`。

本轮已检查：Query 域审计脚本 `py_compile`、Query 域 validator 17/17 PASS、`git diff --check`。Q0 高速 Query forward/parity 尚未运行。

## 6. 当前结论

### 6.1 已确认事实

1. 当前高速 fixed-DBM OAC Actor 在 train-only 120-state numerical audit 上距 best-found 约 `0.473%`，已回收约 `98.32%` headroom；fixed-DBM 单中心主体不是当前优先瓶颈。
2. 15-run OOF direct center 主体很强，但仍存在约 1.39% 回归和灾难 worst，因此 guard 不能删除。
3. 冻结/单次 Critic gradient 路线不可用；持续 Actor-visited OAC 可以工作，并与 Query 无解析梯度的接口兼容。
4. Query 与当前控制接口匹配，但 checkpoint 的训练速度只有约 0--3.1 m/s，当前高速输入约 7.8--30.6 m/s，normalization z 达 25--44；它是严重 OOD。
5. 当前高速 history 是 early-prime（step 0--4），不是成熟 250-step history；不能和未来正常闭环历史混作同一分布。
6. 当前 OOF Actor 不是一个可部署 full-train checkpoint；当前 wrapper/two-center/high速闭环也没有完成。

### 6.2 当前推断

1. 如果冻结 Query 被定义为研究环境，最合理路线是先过数值 Q0，再完全以 Query cost 重建 Replay/Critic/OAC；无需先审计 Query 对 DBM/真实世界是否准确。
2. fixed-DBM Actor/checkpoint 可作为 Query OAC 初始化，但其 DBM 优势不能当 Query 下的收益结论。
3. 在严格 final-output fallback 存在时，Actor 尾部 loss 的优先级可低于主体/计算效率；但必须继续监控 fallback、wrapper half-budget、抖动和模型利用。
4. 当前 OAC 主体剩余差距更多来自共享 Actor update geometry/状态分布，而不是 Critic 完全不准；不过该判断只在 fixed DBM 当前合同内成立。

### 6.3 尚未验证的假设

1. Query 在高速 OOD 下是否 finite、deterministic、candidate-order 稳定；
2. 高速 PyTorch 与 ONNX 是否仍 parity；
3. early-prime history 与 mature history 是否产生不同 Query cost/ranking；
4. Query cost 下 Actor 是否仍优于 warm、是否能通过 episode-grouped OOF；
5. warm32+Actor32 是否在相同总预算下优于 warm64；
6. final-output fallback 是否能做到零 Query-relative warm violation且不过度 fallback；
7. Query 模型收益能否传到 fixed-DBM、闭环和真实车辆；
8. 当前 15 OOF Actor 如何转换成最终单一 full-train deployment checkpoint。

## 7. 下一步

### 7.1 立即执行：Q0 高速 Query 数值环境门

在不消费 formal validation/test 的前提下，使用现有 train-only 高速 replay，至少检查：

1. PyTorch Query 对 warm、当前 OOF Actor center、局部 exploration candidates 的 50-step trajectory 和 J50 全部 finite；
2. 同输入重复运行 trajectory/cost/ranking 可复算；candidate permutation 后结果对应一致；
3. cost components、candidate ranking、尺度和饱和/爆炸比例；
4. 相同输入的 PyTorch/ONNX trajectory、cost、ranking parity；
5. early-prime history 与 mature-history 分层。若现有数据没有成熟 history，单独构造/采集，禁止把两类 history 混在一个指标中；
6. 预注册数值/parity 门后再跑，不得看结果后降低门槛。

Q0 只判 Query 是否可作为稳定数值环境，不判其物理真实性。

### 7.2 Q0 通过后：Query-relabel 与 Query OAC

1. 用冻结 Query 对现有 state/action/reference 重新计算 trajectory、cost 和 candidate ranking，建立纯 Query Replay；
2. 明确选择 early-prime 或 mature-history 作为正式训练合同；若部署用成熟历史，主训练/验证必须以成熟历史为准；
3. DBM pretrained Actor/Twin Critic 只作为初始化 A/B；Query Critic 必须以 Query cost 训练；
4. 保持 no-analytic-gradient、Actor-visited Replay、Twin Critic 持续更新、real Query checkpoint gate；
5. 先做 episode-grouped 5-fold x 3-seed direct-center OOF，再生成 full-train candidate；
6. 报 warm-relative mean/median/P05/worst/win-rate、速度/场景/history-maturity 分层与 episode-bootstrap CI。

### 7.3 direct center 过门后：MPPI 传导

1. 固定总预算做 warm64 vs warm32+Actor32，保留 exact warm/Actor centers，使用 CRN；
2. 分别报告 Actor direct、wrapper、final-output，不得混合；
3. final-output fallback 必须做到同 Query 模型下 zero warm-relative violation；
4. 同时报 fallback rate、ESS、best weight、计算延迟、控制平滑性和候选预算损失；
5. 通过后才做固定 Query 短闭环；随后才是 DBM/更真实模型对照、formal validation/test 和实车。

### 7.4 不得降低的门槛

- episode-grouped split，禁止相邻帧/同 episode 泄漏；
- checkpoint 只用 inner/train selection，不用 OOF/formal/test 选 epoch；
- 不逐状态选 seed/oracle，不只报 mean；
- direct、wrapper、final-output 口径严格分开；
- fixed budget 与 CRN 后才比较 MPPI；
- final-output model-relative warm violation 必须为 0；
- PyTorch/ONNX parity、finite/determinism 不得因高速 OOD 放宽；
- formal validation/test 只在 Query OAC、wrapper、fallback 和 full-train checkpoint 合同冻结后一次性使用；
- 闭环和实车结论不能由 open-loop direct-center 指标替代。

### 7.5 暂不优先

- 继续扩 fixed-DBM 全局数据、LR/trust/K/网络结构；
- 重启冻结 Critic gradient、Hessian、loss/coordinate 扫描；
- 非均匀 knots；
- 在 Q0 前直接长时间 Query OAC；
- 在 wrapper 合同冻结前做 formal validation/test 或实车。

## 8. 2026-09-01 Query 训练域与闭环数据合同覆盖

本节覆盖上文把 small-car Query 当作当前默认 Query 环境的旧判断。当前使用的冻结
checkpoint 是：

```text
outputs/formal_real_finetune_query_baseline_split/20260728T143256/query_best.pt
```

它由
`outputs/formal_kinematic_residual_query_30epoch/20260728T114849/query_best.pt`
预训练后，在真实 train split 上 fine-tune。预训练数据来自
`/disk/collect_data_from_anycar/New_demo/new_data_with_x_mean_zero/total_data_1`，
fine-tune 数据来自
`/disk/collect_data_from_anycar/data_from_bag/new_temp_data/pkg_file`；不能再把
`generated_small_car_query_dt005` 当成这个 checkpoint 的训练集。

### 8.1 被否决的 provisional 数据

```text
outputs/query_mppi/real_query_train_distribution_replay_20260901_v1
```

该 artifact 的 state/history 种子分层可复用，但其 `reference[1:]` 等于日志真实未来
轨迹，`mean_knots_before` 又由真实未来 action 投影得到。它把行为轨迹误当期望轨迹，并
存在未来信息泄漏；不得作为 sampling-center 训练、cost 比较或 qualification 数据。

原训练 `.pkl` 虽有 `traj_x/traj_y` 字段，但正式 pretrain/fine-tune 文件中这两个字段
为空，无法从中恢复当时的期望道路。真实未来 state/action 只能作为行为诊断，不能作为
reference 或 causal warm。

### 8.2 新的权威 train-only Query 闭环 collection

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/
query_expected_road_train_20260901_v1
```

核心合同：

1. 原 Query train-only 数据只提供初始 `history/state/current_action` 和 provenance；
   明确不读取 provisional artifact 的 reference、真实未来 state/action、warm 或 cost。
2. reference 是独立生成的周期性期望道路。x/y 以真实米制弧长参数化，当前位置先投影
   到中心线，再按 `v_ref * 0.05` 生成当前点加未来 50 点 `[x,y,yaw,vx]`。
3. 冻结 final-real Query 同时作为 MPPI rollout 环境和确定性 plant；优化 action sequence
   的 fresh Query rollout 第一个状态成为下一闭环状态。没有 DBM label/gradient。
4. step 0 的 cold start 仅重复当前 action；之后严格使用上一轮 weighted sequence 左移后
   的 8-knot running state。不得用日志未来 action 拟合 warm。
5. 每个 episode 完整闭环运行；只有 control step 250 以后才采 snapshot，保证 250-token
   history 已全部由同一 Query 闭环产生。
6. 驾驶基线使用两轮、256-sample Gaussian MPPI。每轮 candidate 0 都精确保留该轮的
   `sampling_mean_knots`；因此 snapshot 只保存的最后一轮 candidate 0 等于第一轮更新后的
   中心，**不等于**本次控制调用入口的 `mean_knots_before`。入口 warm 必须由
   `mean_knots_before` 重新插值并 fresh Query rollout。不要用 0.10/0.30-sigma
   fixed-Hadamard 局部 bank 冷启动驾驶，它在正 throttle 种子上无法及时调速。

collection 共 20 个 episode、600 个 snapshot：5 个 reference speed 档
40/55/70/85/100 km/h，每档 4 个 episode、120 个 snapshot。道路包含 circle/oval、
nominal/recovery 和不同曲率。冻结 Query 的 history-updated 一步闭环在 40 km/h 右弯上
出现单次 J50 预测与滚动一步 yaw 分支相反的稳定性问题，因此 40 km/h 的两个右弯槽位
用不同曲率/偏置的左弯替代；55--100 km/h 保留左右方向。旧 40 km/h 右弯 pilot 是压力
失败集，不能混入平衡训练集，也不能把新集合描述成 40 km/h 方向平衡。

### 8.3 独立验证结果

生成和验证入口：

```text
scripts/model_verify/collect_query_expected_road_closed_loop.py
scripts/model_verify/validate_query_expected_road_closed_loop.py
```

权威验证文件：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/
query_expected_road_train_20260901_v1/validation.json
```

qualification：`QUERY_EXPECTED_ROAD_CLOSED_LOOP_PASS`。主要结果：

- state/action chain、cold start、receded warm、snapshot/trace、完整 history 因果重建误差均为 0；
- reference 每步米制推进最大相对误差 `1.4532e-4`；
- candidate zero/最后一轮 `sampling_mean_knots` 误差 0，weight-sum 最大误差
  `2.38e-7`；这里不能把 candidate zero 再称为控制调用入口 warm；
- raw feature cost 重算最大绝对误差 `3.66e-4`；
- 20 个分层 snapshot 的 PyTorch Query trajectory/cost 重放误差均为 0；
- 修改 action step 1--49 后，预测第一状态变化为 0，未发现未来 action 泄漏；
- history `|z|>3` 每 snapshot 比例 median 0、P95 1.04%、max 1.36%；
- 当前 `vx/yawrate` 最大 `|z|=2.223`，候选完整 context 最大 `|z|=2.310`；
- snapshot 速度误差范围 `[-2.491,-0.195] km/h`；
- snapshot 中心线位置误差 median/P95/max 为 `0.084/0.285/0.693 m`；
- 全闭环实际速度约 `37.51--99.97 km/h`，yawrate 约
  `[-0.0975,0.1112] rad/s`；formal validation/test 未消费。

`query_expected_road_pilot_20260901_v1` 到 `v7` 均为诊断/压力 pilot，不是训练集合。
其中 v1 证明 fixed-Hadamard 局部 bank 不适合 cold-start，v3--v6 记录了低速右弯
稳定性失败；只有上述 `query_expected_road_train_20260901_v1` 可作为当前 train-only
闭环数据基础。

### 8.4 下一步覆盖

下一步不再重新拼接日志未来轨迹，也不再重做道路/warm 扫描。先从通过验证的 600 个
成熟 snapshot 构建纯 Query Replay/teacher，按 episode 做 split；训练和评价只使用
Query-relative expected-road cost。若后续需要 fixed-Hadamard 64 局部 bank，应在这些
成熟 causal warm snapshot 上离线重评，不能拿它重新负责 episode 冷启动驾驶。40 km/h
右弯只能作为单独 stability/OOD 压力门，除非 Query 模型本身的一步闭环一致性得到修复。

## 9. 2026-09-01 纯 Query Replay 与 T0 teacher

### 9.1 权威 sidecar 与语义修正

已从第 8 节的 600 个成熟 snapshot 构建：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
query_expected_road_t0_20260901_v1
```

入口与独立 validator：

```text
scripts/model_verify/build_query_expected_road_t0_replay.py
scripts/model_verify/validate_query_expected_road_t0_replay.py
```

构建过程完整保留 256 个 saved candidates、actions、Query trajectories、cost/weight、
cost components、raw features、sampling noise/raw/clipped knots、optimized weighted
sequence 以及全部 state/history/reference/provenance。原 collection 不修改；sidecar
约 324 MB。

由于 collection 每个控制步使用两轮 MPPI，最后一轮 candidate 0 不是控制调用入口 warm。
本 sidecar 对每个 snapshot 都从 `mean_knots_before[8,2]` 重新线性插值出 50-step action，
并用冻结 final-real Query 做一次 fresh deterministic direct rollout，共新增 600 次纯 Query
rollout。T0 的三个候选固定为：

1. 精确入口 warm；
2. 256 个最后一轮 saved Gaussian candidates 中 direct cost 最低者；
3. 已保存 MPPI weighted-output sequence 的 fresh direct rollout。

teacher 按上述顺序取 deterministic Query direct cost 的 `argmin`，并列时保留较前候选；
因此 warm 是构造性零回归 floor。`teacher_delta_knots` 始终相对真正的
`mean_knots_before`，不使用 DBM label/trajectory/gradient，也不使用日志未来行为。

### 9.2 episode-grouped 5-fold

20 个 episode 是 5 速度 × 4 道路 variant 网格。冻结分组公式为：

```text
fold_id = (speed_index + variant_index) % 5
```

每个 fold 有 4 个完整 episode、120 行，包含每个 variant 恰好一个 episode；任何相邻帧
或同 episode 都不会跨 fold。由于每 fold 只有 4 个 episode、速度有 5 档，单 fold 必然
缺一档速度；五个 OOF fold 合起来对速度和 variant 平衡。formal validation/test 仍未消费。

### 9.3 独立验证与结果

qualification：`QUERY_EXPECTED_ROAD_T0_REPLAY_PASS`。主要检查：

- source manifest/validation/checkpoint/replay/split/CSV 以及 20 个 episode snapshot 哈希
  全部匹配；600 行全部 source fields consolidation 误差 0；
- 5-fold episode grouping、row metadata、teacher argmin 和 CSV 重构误差 0；
- 600/600 个精确入口 warm 的 PyTorch Query trajectory/cost replay 误差 0；
- 20 个分层行的 256-candidate PyTorch Query trajectory/cost replay 误差 0；
- teacher knot 插值最大误差 `5.07e-7`，warm-floor violation 数量和最大值均为 0；
- candidate component/raw-feature cost replay 最大误差为
  `1.96e-4/3.66e-4`；
- ONNX shadow 在 10 个分层行的 trajectory 最大误差 `0.001953125`，cost 最大绝对/相对
  误差 `0.61786/0.003668`；后者来自高 cost 对轨迹差的放大。10/10 best candidate index
  与 PyTorch 一致，best cost 最大差 `0.003626`。现有 ONNX 文件无内嵌 checkpoint metadata，
  因而这里只记行为 parity，不把它当 provenance 证明。

总体 direct cost：warm/best-saved/weighted-output/teacher mean 为
`8.553/4.644/5.663/4.520`。teacher 相对真正入口 warm 的 gain mean/median/P95/max 为
`4.032/0.916/17.947/126.186`，严格改善 `536/600`，其余 `64/600` 精确回退 warm，零退化。
teacher 来源为 warm/best-sampled/weighted-output `64/360/176`；标准化 knot delta RMS
mean/median/P95/max 为 `0.456/0.336/1.058/1.319`。

按速度的 warm→teacher mean cost/gain/严格胜率：

| km/h | warm | teacher | gain | strict win |
|---:|---:|---:|---:|---:|
| 40 | 13.344 | 12.501 | 0.843 | 95.8% |
| 55 | 3.680 | 3.034 | 0.646 | 89.2% |
| 70 | 5.845 | 2.170 | 3.676 | 90.8% |
| 85 | 6.728 | 2.269 | 4.459 | 81.7% |
| 100 | 13.166 | 2.628 | 10.538 | 89.2% |

40 km/h mean gain较低，且该层两个历史右弯槽位已被左弯替代；不得把它解释成完整低速
方向覆盖。五个 fold 的 teacher gain mean 为 `5.475/4.393/5.572/1.488/3.234`，分布差异
必须在后续 OOF 汇报中保留，不能只报 pooled mean。

权威哈希：

```text
manifest.json   945ccf5e127c497b94a51e9927ba96ed28ba6f9fac12c3d255b0ec9a92a0a250
validation.json 71657f2426ddd29bc4ced5a7ce5cfb28e0c6e1934bb23755b7ca9fdafe4caae0
replay.npz      412a9d8237a36e3e935f7a54dcf8f642f77c2f249f50a310a1082e064c494e91
```

### 9.4 下一步覆盖

T0 是一个可靠、warm-floor 保守且 cost-relabelable 的 train-only 起点，但它只搜索当前
saved Gaussian bank 与 weighted output，不是全局 oracle。下一步在这 600 个成熟 causal
snapshot 上离线生成 actor-visited/full-rank fixed-Hadamard Query direct-cost sidecar，保留
warm 和 T0 teacher，构造 improvement/regression pair；随后按上述 5-fold 做 Critic/Actor
训练与 OOF，checkpoint selection 只能使用各 fold 的训练侧 inner split。此阶段继续禁止：

- 使用 DBM cost/gradient 或 provisional 日志未来 reference/warm；
- 用 OOF fold 选 epoch/seed，或逐状态选择 oracle seed；
- 删除 exact-warm two-center guard；
- 消费 formal validation/test、启动长闭环或宣称部署性能。

## 10. 2026-09-01 T0-centered full-rank 近邻 Query sidecar

### 10.1 权威 artifact 与候选合同

已在第 9 节通过验证的 600 个 T0 snapshot 上生成纯 Query、train-only 的全秩近邻
sidecar：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
query_expected_road_fullrank_20260901_v1
```

冻结配置、生成器和独立 validator：

```text
scripts/model_verify/query_expected_road_fullrank_config_20260901_v1.json
scripts/model_verify/build_query_expected_road_fullrank_sidecar.py
scripts/model_verify/validate_query_expected_road_fullrank_sidecar.py
```

每行固定 132 个 direct Query candidates：

1. 精确控制调用入口 warm；
2. T0 teacher；
3. T0 saved-bank best；
4. T0 weighted output；
5. 以 T0 teacher 为中心，16 个未归一化 Sylvester-Hadamard 全秩方向，正负两侧，
   sigma 半径 `0.03/0.06/0.10/0.15`，共 128 个 proximal probes。

全部候选使用同一冻结 final-real Query、同一 J50 cost 和 action bounds 重新 direct rollout；
teacher 是 132 个候选的 first-index `argmin`。因此 warm 和 T0 都是构造性 floor。总计新增
`600 * 132 = 79,200` 次 Query rollout，artifact 约 93 MB。它保留完整候选、轨迹、cost
components、raw features、clip mask、成对差值和 improvement/regression 样本。

这里的名称必须保持准确：这是 **T0-centered full-rank proximal sidecar**，还不是
actor-visited 数据，因为当前尚无 Query Actor，不能把固定设计点冒充策略访问分布。

### 10.2 独立验证

qualification：`QUERY_EXPECTED_ROAD_FULLRANK_PASS`。16 项检查全部通过，包括 artifact
哈希、父 T0 上下文、132-candidate 合同与重构、teacher/pair 重构、全秩几何、cost
components/raw features、episode split、PyTorch Query replay、ONNX shadow、warm/T0 floor，
以及 formal validation/test 仍封存。

关键复算结果：

- 20 个分层行、完整 132-candidate bank 的 PyTorch trajectory/cost replay 误差均为 0；
- 每行每半径的设计矩阵 rank 都为 16；condition median/P95/max 为
  `1.000002/2.0/3.000005`；
- cost component/raw-feature 最大误差为 `1.30e-5/2.29e-5`；
- 同一候选 bank 下 warm/T0 floor violation 数量均为 0；
- ONNX shadow 的 trajectory 最大误差 `0.001709`，cost 最大绝对/相对误差
  `0.04835/0.007281`，10 个分层行的 best index 全部一致，best cost 最大差
  `0.001903`。

跨 batch 父数据复算和 ONNX shadow 的数值容差按实测浮点放大修正：父数据 gate 为
trajectory `<=5e-4`、cost absolute `<=0.01`、relative `<=0.002` 且 argmin mismatch 必须
为 0；ONNX gate 为 trajectory `<=0.002`、cost absolute `<=1.0`、relative `<=0.01`、best
index mismatch 必须为 0、best cost `<=0.01`。实测父数据最大 trajectory/cost/relative
误差为 `0.000244/0.006512/0.001638`，argmin mismatch 为 0。该调整只覆盖跨 batch/backend
的 numerical shadow，不改变 PyTorch 同 bank 精确 replay、候选排序或性能门槛。

权威哈希：

```text
manifest.json   243a2058c0546a966aa28be7e82ec3b612506b6ec1cea8ab0cedf6c22dbcad5a
validation.json 5ac2fb02dd07d3fa91b7604fe6d5e3250b3ba3ff9bf7cf7b2c56229c091ff270
bank.npz        7a908e90a56aa83addfb25f299b8d98f7333424460e901c82c6b9c3532efd094
```

### 10.3 结果与数据分布

T0 teacher/full-rank teacher 的 direct cost mean 为 `4.5204/4.2906`。full-rank teacher
相对 T0 的 gain mean/median/P05/P95/max 为
`0.2298/0.1523/0.0164/0.7611/1.6162`；`594/600` 严格改善，另外 6 行精确保留 T0，零
退化。相对 exact warm 的 gain mean/median/min 为 `4.2621/1.0889/0.00558`，600 行全部
严格改善。teacher 相对 T0 的 sigma-normalized RMS 位移 mean/median/P95 为
`0.0998/0.1000/0.1500`。

按速度的 T0→full-rank mean cost/gain/严格胜率：

| km/h | T0 | full-rank | gain | strict win |
|---:|---:|---:|---:|---:|
| 40 | 12.501 | 12.373 | 0.128 | 96.7% |
| 55 | 3.034 | 2.915 | 0.120 | 100.0% |
| 70 | 2.170 | 1.935 | 0.235 | 99.2% |
| 85 | 2.269 | 1.974 | 0.295 | 100.0% |
| 100 | 2.628 | 2.256 | 0.372 | 99.2% |

五个 fold 的 gain mean 为 `0.224/0.194/0.226/0.197/0.308`，严格胜率为
`100/100/100/95.8/99.2%`。40 km/h 仍受第 8 节所述方向覆盖限制，不能外推为完整低速
右弯能力。

近邻 bank 没有退化成 winner-only 数据：全部 76,800 个 proximal probes 中，优于 T0
的比例为 25.74%，差于 T0 的比例为 74.26%。每个半径的候选正收益率分别为
`37.37/28.95/21.13/15.49%`，而该半径 32 个方向中至少一个优于 T0 的行比例为
`99.0/97.7/94.3/86.8%`。最终 teacher 来源的半径计数为：T0 `6`，0.03 `99`，0.06
`125`，0.10 `111`，0.15 `259`；正负方向计数为 `291/303`。候选 clip fraction
mean/median/P95/max 为 `1.26/0/9.09/24.24%`，成对未裁剪对称率为 `81.13%`。因此数据
同时具有全秩局部方向、正负 cost 对比、多个有效尺度和明确的边界标记，适合下一阶段
cost-aware distillation；训练时不得静默丢弃 clipped/negative 样本。

### 10.4 下一步 Actor 入口合同

> **作废覆盖（2026-09-01）**：本节误用了较早的 warm-conditioned residual 合同，违反本文
> §4.5/§7.2 已冻结的 `DirectNoAnchorGTXActor` absolute-center 路线，也违反
> `mppi_sampling_center_review_20260812.md` §11.70/§11.142。以下内容仅保留为错误路径记录，
> 不得作为当前 Actor 输入、监督或下一步依据。

下一步先做 episode-grouped 5-fold 的 **train-only learnability/OFF-policy distillation
诊断**，不直接启动长闭环，也不把当前 sidecar 称为 actor-visited replay。部署时 T0
teacher 需要 256-candidate 搜索，不能作为 Actor 输入；Actor 合同应为：

```text
pi(state, history, expected-road reference, current_action,
   exact causal mean_knots_before) -> bounded delta_knots
actor_center = clip(mean_knots_before + delta_knots)
```

训练 target 可由 full-rank teacher 相对 exact warm 的 delta 得到；T0 与 132-candidate
cost bank 用于 cost-aware/ranking 监督和离线 direct Query 评分，但不得在推理输入中泄漏
teacher index、teacher cost 或未来搜索结果。先报告逐 fold Actor direct cost、相对 warm/T0
的 gain/退化尾部、动作边界与 delta 分布；checkpoint epoch/seed 只能由每个训练 fold 内部
selection 决定，OOF 只做评分。

若 OOF direct-center 通过，再用其 OOF Actor 输出在同一 600 个状态上生成真正的
actor-visited 邻域 sidecar，并保留 exact-warm two-center guard。之后才进入固定总预算、
CRN 的 warm64 vs warm32+Actor32 MPPI 比较；formal validation/test 继续封存。当前阶段不要
回到冻结 Critic gradient/Hessian 路线，也不要用 DBM cost 或日志真实未来轨迹替代 Query
expected-road cost。

## 11. 2026-09-01 Query Actor nested-OOF learnability 负门结果

> **路线资格作废（2026-09-01）**：artifact 的数值复算有效，但它训练了已被覆盖的
> warm-conditioned residual Actor。因此 `QUERY_ACTOR_OOF_LEARNABILITY_FAIL` 不能用于接受或
> 否决当前 strict no-anchor absolute-center Query OAC。对应输出已增加 `REJECTED_ROUTE.md`；
> source collection、T0 与 full-rank bank 不受影响。本节其余内容只作错误实验审计记录。

### 11.1 冻结训练合同与产物

已执行第 10.4 节规定的首轮 train-only Actor 可学习性诊断：

```text
outputs/query_mppi/query_expected_road_actor_oof_20260901_v1
```

入口、配置与独立 validator：

```text
scripts/model_verify/query_expected_road_actor_oof_config_20260901_v1.json
scripts/model_verify/train_query_expected_road_actor_oof.py
scripts/model_verify/validate_query_expected_road_actor_oof.py
```

Actor 为 `TorchMPPIProposalPolicy`，只读取运行时可用的 `history[250,7]`、期望道路
`reference_ego`、`[vx,yawrate,current_accel,current_steer]` 和 exact causal
`mean_knots_before[8,2]`；不读取 T0/full-rank teacher、candidate cost/index、未来搜索结果或
DBM 字段。输出为相对 warm 的 `tanh` 有界 16 维 residual。

teacher-side trust 预检用 frozen Query J50 重新评价 `1.0/1.5/2.0/3.0/4.5 sigma` 投影。
`2 sigma` 已保留 full-rank teacher 聚合 headroom 的 `99.8529%`，mean cost
`4.29685`（full-rank `4.29058`），600 行相对 warm 零退化；`1 sigma` 仅保留
`92.86%`，有 11 行退化且 worst gain `-22.61`。因此只冻结 `2 sigma`，没有用 OOF 扫 trust。

外层沿既有 5-fold，每 fold 120 行/4 个完整 episode。每个 seed 的 inner selection 使用另
一个完整 fold；每次 run 为 360 fit / 120 inner selection / 120 OOF，三者 episode 不相交。
共 `5 folds x 3 seeds = 15` 个 checkpoint。训练 300 epoch，每 30 epoch 用 inner frozen-Query
direct J50 选择 checkpoint；first minimum wins ties。OOF 只在 checkpoint 冻结后评分一次。
formal validation/test 未消费。

### 11.2 Actor 资格失败，但独立实现校验通过

Actor qualification：`QUERY_ACTOR_OOF_LEARNABILITY_FAIL`。三 seed pooled OOF：

| seed | Actor mean J | warm→full-rank recovery | gain median | gain P05 | warm regression |
|---:|---:|---:|---:|---:|---:|
| 0 | 8.829 | -0.0647 | 0.199 | -10.694 | 43.5% |
| 1 | 7.470 | 0.2541 | 0.329 | -6.002 | 37.3% |
| 2 | 8.243 | 0.0728 | 0.314 | -9.296 | 34.7% |

共同基线 mean cost 为 warm/T0/full-rank `8.553/4.520/4.291`。三 seed 中没有一个达到
预注册 `H_OOF>=0.50`，只有 seed 1 的 episode-bootstrap CI 下界略为正
`[0.005,0.386]`，但 recovery 仍只有 25.4%。Actor 只在 `15.7/19.3/20.5%` 的行上不劣于
T0；因此不能把“mean 略优于 warm”的 seed 1 描述成学会 T0/full-rank search。

median seed 2 的速度层 full-rank recovery（40/55/70/85/100 km/h）为
`-1.368/0.270/0.321/0.207/0.040`，每个速度层 warm-relative P05 都为负。fold 3 三 seed
系统性反迁移，fold recovery 为 `-1.183/-0.535/-0.713`；median seed 的主要单 episode
失败是 40 km/h variant 3 `episode_015`，recovery `-3.66`、mean gain `-6.53`，但 100 km/h
多个 episode 也有显著负尾，不能通过删除一个 episode 或只归因低速来修饰结果。

训练 target 数值上可以拟合：15 个 run 到 epoch 300 的 fit normalized MSE 大多约
`0.010--0.015`；但 inner direct Query cost 通常在 epoch 30--90 后恶化，15 个 checkpoint
有 12 个在 epoch 30--90 选中，只有 3 个在 epoch 240。所选 checkpoint 的逐 run fit
recovery 按 seed median 约 `0.410/0.452/0.606`，OOF pooled 却为上述
`-0.065/0.254/0.073`。这说明主要问题不是 2-sigma 表达上限或 optimizer 无法降低标签 MSE，
而是 hard finite-bank argmin residual 的低 MSE不对应跨 episode 的低 Query J50；同时早停后的
fit recovery 也未稳定达到高水平，不能简单称为“纯粹只有 OOF 泛化问题”。

独立 validator qualification：`QUERY_ACTOR_OOF_INDEPENDENT_PASS`。全部检查通过：

- 15/15 checkpoint、source/config/script/summary/OOF artifact 哈希匹配；
- fit-only normalization、360/120/120 nested episode split 与 checkpoint selection 重构误差 0；
- checkpoint OOF knots 重放误差 0，最大实际 residual `1.99293 sigma`，无 action bound violation；
- 全部 1,800 个 OOF PyTorch Query direct cost 重放绝对/相对误差均为 0；
- 20 个分层行的 projected-teacher Query cost shadow 误差 0；
- 无 DBM 字段或 label，formal validation/test 保持封存。

权威哈希：

```text
manifest.json          55c76f44c7a00625dcfec6165876c7373a7f530ac2a175291ec0efa37dbe76e4
summary.json           2174ea6bfe1aa73130459cb86cdf8d1a2c321a57093bb7b9cb35ea5d384d64e2
validation.json        681c1cdfcea997391ea6d59f7f4f71c183c1a4deb8421dd50a0b786e07ffda82
oof_predictions.npz    bf84f62491e248aecd638ae3f8445f7bfc4a1db658cdaa356d23e45b6da4e525
projected_teacher.npz  e7be6455be48804939333c3baa02ead62cb60b6a896c0af2c83589d14b146b23
```

### 11.3 路线裁决与下一步

本轮禁止进入 actor-visited Replay、MPPI wrapper、full-train checkpoint、闭环或 formal/test。
也不要对同一 hard-label MSE 基线继续做 LR/epoch/网络宽度/seed/trust 扫描；2-sigma oracle 已
排除主要表达上限，15-run nested OOF 已给出稳定负门。

下一步先做 **零新增 Query rollout 的 target/coherence 诊断**，复用 132-candidate cost bank：

1. 按完整可部署输入（含 exact warm）做 episode-excluded 邻域，分别测 hard argmin residual、
   T0 residual、局部 pair sign 和 near-optimal candidate set 的跨 episode coherence；
2. 把“同一状态存在多个近等价低 cost center”与“邻近状态真正最优方向翻转”分开，报告
   cost-gap/margin、半径、clip、速度/variant/fold，特别单列 episode_015 和 100 km/h tail；
3. 只在数据证明 near-optimal set 比 hard argmin 更一致时，预注册 set-valued/WTA 或
   cost-gap-weighted distillation；先做 teacher-side Query replay oracle，再允许另一次 Actor OOF；
4. 若 near-optimal set 也不具备跨 episode coherence，应回到道路/状态覆盖与可观测输入合同，
   而不是生成由失败 Actor 主导的 actor-visited 数据。

Critic仍不得作为冻结梯度提供者。132-candidate cost 可以用于离线集合监督/排序诊断，但不能把
同一失败 Actor 经 Critic 包装后称为策略通过。exact-warm two-center guard 仍是任何后续部署
合同的硬要求。

## 12. 2026-09-01 strict no-anchor absolute Actor + Twin Critic 初始化验证

本节覆盖第 10.4/11 节中已作废的 warm-conditioned residual 路线，使用本文 §4.5/§7.2 和
`mppi_sampling_center_review_20260812.md` §11.70/§11.142 冻结的正确合同。数值验证完成，结论是
**实现与重放通过，但当前训练分布下初始化性能负门，禁止进入 actor-visited continuous OAC**。

### 12.1 冻结合同与产物

产物：

```text
outputs/query_mppi/query_expected_road_absolute_pretrain_20260901_v1
```

入口、配置和独立 validator：

```text
scripts/model_verify/query_expected_road_absolute_pretrain_config_20260901_v1.json
scripts/model_verify/pretrain_query_expected_road_absolute_actor_twin_critic.py
scripts/model_verify/validate_query_expected_road_absolute_actor_twin_critic.py
```

Actor 是 `DirectNoAnchorGTXActor`，监督目标是 full-rank teacher 的 **absolute `[8,2]`
sampling center**。Actor 只读 `history[250,7]`、expected-road ego reference 和
`[vx,yawrate,current_accel,current_steer]`；warm/anchor、first-pass feedback 和 gradient
carrier 在模型内部强制清零，也不读取 teacher/cost/index、未来轨迹或 DBM 字段。warm 只保留为
Actor 外部的 exact two-center guard，不是 Actor 输入或 label canonicalization 基准。

Twin Critic 是两个独立的 `ConfigurableAbsoluteActionValueCritic`，读取相同可部署 state context
和 absolute candidate center，学习全部 132 candidates 的 standardized `log1p(frozen-Query J50)`；
训练中没有 Query gradient、TD bootstrap 或冻结 Critic Actor update。这仍只是进入 continuous OAC
前的 offline initialization，不是 OAC 本身。

沿既有 5 个 episode folds，每次 outer OOF 120 行/4 episodes，inner selection 120 行/4 episodes，
fit 360 行/12 episodes；三者 episode 严格不相交。每折 3 seeds，共 15 个 Actor/Twin-Critic
checkpoints。Actor 500 epochs，每 25 epoch 仅按 inner whole-episode frozen-Query direct mean J50
选择；Critic 240 epochs，每 10 epoch按 inner value/ranking composite 选择。OOF 从不参与 checkpoint
或超参选择，formal validation/test 保持封存。

### 12.2 初始化性能负门

训练端 qualification：

```text
QUERY_ABSOLUTE_PRETRAIN_FAIL_NO_OAC
```

四个预注册初始化门全部失败：

| gate | 15-run OOF median | threshold | pass |
|---|---:|---:|:---:|
| Actor warm→teacher aggregate recovery | -1.651 | >= 0.50 | no |
| Twin Critic Pearson(log cost) | -0.191 | >= 0.50 | no |
| Twin Critic warm/teacher order accuracy | 0.708 | >= 0.80 | no |
| Twin Critic bank gain recovery | 0.494 | >= 0.50 | no |

Actor 15-run recovery min/median/max 为 `-23.085/-1.651/-0.493`，没有一个 seed/fold 为正；
warm-relative gain P05 的最好/median/最差 run 为 `-20.961/-42.928/-243.829`，直接部署尾部
门同样失败。三 seed pooled OOF 的 mean J50/recovery/warm regression fraction 为：

| seed | mean J50 | recovery | warm regression |
|---:|---:|---:|---:|
| 0 | 23.917 | -3.605 | 80.5% |
| 1 | 17.621 | -2.128 | 76.5% |
| 2 | 23.352 | -3.472 | 73.3% |

共同 warm/T0/full-rank mean J50 仍为 `8.553/4.520/4.291`。40 km/h pooled recovery 最差，
三个 seed 为 `-40.410/-24.865/-40.941`；但 55--100 km/h 也全部为负，不能把失败只归因于
低速。fold 3 的 Actor mean J50 为 `51.99/38.46/51.22`，recovery 为
`-23.085/-15.056/-22.627`，是最严重的未见组合外推失败。

输出 support 不是主要障碍：用每次 fit-only `mean +/- 3std` 投影 full-rank teacher 的 oracle
仍可恢复 `85.59%` teacher headroom，95.67% 行不劣于 warm，mean J50 `4.905`。但 oracle 自身仍有
4.33% warm regression、worst gain `-35.631`，所以 exact-warm guard 仍不可删除。这里的 oracle
只证明输出参数化大体覆盖 teacher，不证明 Actor 可学习。

Twin Critic 在 pooled 三 seed 上的 Pearson 为 `-0.069/-0.095/-0.149`，bank gain recovery 为
`0.418/0.476/0.350`。部分 fold 虽能靠候选排序取得正收益，但全局 cost 相关性、排序准确率和
跨折稳定性不足，不能用于启动 Actor 更新。

### 12.3 独立验证与数据分布解释

独立 validator qualification：

```text
QUERY_ABSOLUTE_PRETRAIN_INDEPENDENT_PASS
```

全部 15 项检查通过，关键误差均为 0：15/15 checkpoint 和所有 artifact 哈希；360/120/120
nested split；fit-only normalization；checkpoint selection/metric 重构；Actor checkpoint OOF
输出；Twin Critic conservative OOF 输出；no-anchor/no-feedback/no-gradient invariance；action bounds；
全部 1,800 个 Actor OOF PyTorch Query direct cost replay；20 个分层 support-oracle Query shadow；
以及初始化门重构。没有使用 DBM 字段或 label，formal validation/test 未消费。因此本节是
“验证实现通过、性能负门”，不是 pipeline 数值故障。

当前 600 行只来自 20 个 episodes，即 5 个速度 × 4 类道路组合各一个 episode，每 episode
30 snapshots。没有同一个 speed/road cell 的独立重复；现有 5-fold 划分使每次 fit 只有 12 个
episodes，并让部分 speed-road 组合在 OOF 中整体未见。故本轮可信地否决了“用这 20 episodes
直接初始化并进入 OAC”，但不能把结果外推成“absolute/no-anchor 合同本身不可学习”；它混合了
同分布 episode 泛化与未见 speed-road 组合外推。

若继续重建训练数据，最低应先让每个 speed × road/direction/curvature stratum 有足够的独立
road seed/closed-loop episode，使 5-fold 中每个 fit、inner selection 和 OOF 都覆盖所有 strata；
40 km/h 必须补齐左右方向，不再使用方向替代来冒充完整覆盖。一个直接可审计的起点是每个 cell
5 个独立 episodes（20 cells 共 100 episodes），fold 按 cell 内 repeat 分配；expected road 仍由
道路生成器产生，不用日志真实未来轨迹作为 reference。先只重做 train-only collection/T0/full-rank
sidecar 和同一套 nested OOF，仍不得消费 formal/test。

### 12.4 当前裁决

1. 不启动 actor-visited continuous OAC，不生成失败 Actor 主导的 replay；
2. 不把 warm 塞回 Actor，也不回到 residual target；
3. 不用冻结 Critic/one-shot gradient provider 绕过负门；
4. 保留 exact-warm two-center guard；
5. 下一步若获准，应先扩充并平衡 expected-road train-only episodes，再重复本节的 absolute
   Actor/Twin-Critic OOF 初始化验证。

权威哈希：

```text
manifest.json       446728cf84c39a0ff989032bdaae4229cf576569d4cbc2fa5301d0a9fcf483b7
summary.json        f3bdd4d9c746796c654b130611d6f17bec85591f7d0de556918e6db1e58241b6
validation.json     d794bc0486a85115ad94eee79fd44e20a1498d3278fa385b8b345e669dc61bd5
oof_predictions.npz c5dca355569c313cefbe668fe1a77cbb8e8240cd2ccb7526ab60090b7778a29d
config              224a2e9feaa6ef6e7ec69feb385364511ef3fafa459217ec4760acbb77de737b
trainer             b0544a9ee796401758b0f023c3879b35812536b6dae24ea95b194bb2649e87d6
validator           40cd884823a35bf1006c4838c4578318f9d2f6a8b9987c5fcf4dc5dabf616bf3
```

## 13. 2026-09-01 expected-road 独立 episode 扩展与 absolute 初始化复验

本节执行第 12.4 节的数据扩展建议。结论是：**数据、T0 和 full-rank sidecar 均独立验证
通过；扩展显著改善 Twin Critic 的整体 cost 相关性，也缓解了 Actor 的负迁移幅度，但 strict
no-anchor absolute Actor teacher BC 仍是稳定负门，禁止进入 actor-visited OAC。**

### 13.1 扩展数据合同与低速边界

合格训练 collection：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/query_expected_road_train_expansion_20260901_v3
```

数据为 5 个速度 `40/55/70/85/100 km/h`、4 类 expected-road、每 cell 5 个独立 repeat，
共 100 episodes。每条 episode 在 burn-in 250 后按 25 steps 间隔保存 6 个成熟快照，共 600 行。
与旧数据同为 600 行，但从 20 条 episode 内相邻帧密集采样改为 100 条独立闭环轨迹；100 个
seed replay row 全部唯一。`fold_id=repeat_index`，每个 fold 都完整覆盖 5 speed x 4 road cells，
每 fold 20 episodes/120 rows；Actor nested split 仍为 60/20/20 个完整 episodes，即
360 fit / 120 selection / 120 OOF rows。

55--100 km/h 的道路曲率与 recovery 初始化按 repeat 使用 `-8/-4/0/+4/+8%` 确定性变化槽，
并在 cell 间置换；40 km/h 因冻结 Query yaw branch 狭窄而冻结曲率，只变化独立 history、seed
state、MPPI 随机流和道路相位。40 km/h 原 right-turn variants 1/3 使用明确命名的 left
substitute；variant 3 的 recovery offset/heading 也随方向镜像。它们**不构成 40 km/h 右转覆盖，
不得描述成方向平衡数据**。此前真实 40 km/h 右转压力 pilots 已证明当前 frozen Query 闭环不稳，
所以这仍是环境模型支持边界，而不是被训练数据消除的限制。

第一次扩展目录 `query_expected_road_train_expansion_20260901_v2` 在 episode 027 因道路变化使
成熟段位置误差超过 3 m 而中止，已写入 `REJECTED_INCOMPLETE.md`，不得恢复或用于训练。v3
先精确预检全部 20 条 40 km/h 计划轨迹；其中发现 variant 3 左转替代只翻 curvature、未翻
recovery 符号的几何错误，修正并通过 5/5 后才完整采集。

独立 validator qualification：

```text
QUERY_EXPECTED_ROAD_CLOSED_LOOP_PASS
```

100 episodes/600 snapshots/100 unique seed rows 和完整 speed-variant-repeat 网格全部通过；成熟段
最大位置误差 `1.883758 m`（门限 3 m），最大绝对速度误差 `3.825086 km/h`（门限 5 km/h）。
20 个分层 Query trajectory/cost replay 误差均为 0；artifact hash、状态/动作/warm chain、因果
history 重构、cost/raw-feature 重放、domain 与 first-state no-future-action-leak 检查全过。
formal validation/test 未消费。

### 13.2 T0 与 full-rank sidecar

合格产物：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/query_expected_road_t0_expansion_20260901_v3
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/query_expected_road_fullrank_expansion_20260901_v3
```

T0 qualification 为 `QUERY_EXPECTED_ROAD_T0_REPLAY_PASS`。600 行中 teacher 相对 exact warm
严格改善 `86.5%`，warm tie `13.5%`，warm-floor violation 为 0；teacher 来源为 warm 81、
best sampled 327、weighted output 192。全部 600 行 warm Query 重放误差为 0，Torch/ONNX
shadow、teacher argmin、插值、cost component/raw feature 和 episode split 检查全过。

full-rank qualification 为 `QUERY_EXPECTED_ROAD_FULLRANK_PASS`。每行 132 candidates，共新增
79,200 Query rollouts；四个半径的局部最小秩均为 16，condition 最大 `3.0011`。598/600 行
严格优于 T0，另 2 行合法持平；mean gain vs T0 `0.220083`，warm/T0 floor 实际 violation
均为 0。候选银行、父 sidecar context、pair teacher、Query/ONNX/cost 重放与全部哈希检查通过。

### 13.3 原轮次重训与负门

训练产物：

```text
outputs/query_mppi/query_expected_road_absolute_pretrain_expansion_20260901_v3
```

合同未改：`DirectNoAnchorGTXActor` 学 absolute full-rank teacher `[8,2]` center，不读 warm/anchor；
Twin Critic 学全部 132 candidates 的 standardized `log1p(Query J50)`。每 fold/seed 的 Actor
仍为 **500 epochs**、每 25 epoch inner selection；两个 Critic 各 **240 epochs**、每 10 epoch
selection；5 folds x 3 seeds 共 15 checkpoints。没有用增加 epoch 掩盖数据分布问题。

训练 qualification 仍为：

```text
QUERY_ABSOLUTE_PRETRAIN_FAIL_NO_OAC
```

与旧 20-episode 数据对比：

| 15-run OOF gate metric | old v1 median | expanded v3 median | threshold | pass |
|---|---:|---:|---:|:---:|
| Actor warm→teacher recovery | -1.651 | -0.928 | >= 0.50 | no |
| Twin Pearson(log cost) | -0.191 | 0.710 | >= 0.50 | yes |
| Twin warm/teacher order | 0.708 | 0.792 | >= 0.80 | no |
| Twin bank-gain recovery | 0.494 | 0.240 | >= 0.50 | no |

Actor warm-relative P05 的 15-run 中位数由 `-42.928` 缓解到 `-24.289`，最差 run 由
`-243.829` 缓解到 `-184.551`，但尾部仍明显失败。expanded 三个 pooled seeds 的 Actor
recovery 为 `-1.944/-1.554/-1.674`，warm regression fraction 为
`62.33/58.17/59.50%`；没有任何证据允许直接部署。

fold 3 的主要极端尾部来自 `episode_028`（40 km/h variant 1）：六个成熟状态的 warm J50 从
`235.52` 增至 `981.99`，full-rank teacher J50 从 `232.22` 增至 `974.17`，尽管当前时刻道路
误差仍小于 1.884 m。这是合法的 frozen-Query future-cost 压力状态，不是 collection gate
造假。但完全排除该 episode 后，三个 seed 的 pooled Actor recovery 仍只有
`-1.218/-0.958/-0.836`，所以不能靠删除一个难例或只归因 40 km/h 修饰结果。

独立 artifact validator qualification 为 `QUERY_ABSOLUTE_PRETRAIN_INDEPENDENT_PASS`：全部 16
项检查通过，15/15 checkpoint、nested episode split、fit-only normalization、selection/metrics、
strict no-anchor invariance、Actor/Twin prediction、1,800 个 OOF Query costs 与 support oracle
重放误差均为 0。故这是经过实现排错后的性能负门。

### 13.4 当前裁决与建议下一步

1. 本轮已经排除了“旧数据只有每 cell 一个 episode”是唯一失败原因；继续单纯增加 epoch、seed、
   网络宽度或复制相邻 snapshots 没有依据；
2. teacher BC 的 absolute Actor 映射仍未通过，不能启动 actor-visited replay 或 continuous OAC；
3. Twin Critic 的全局相关性已显著改善，但 warm/teacher order 和 bank gain recovery 未过，仍不得
   作为冻结梯度或策略通过的替代证据；
4. 下一步优先复用现有 132-candidate bank，做零新增 rollout 的 target/coherence 诊断：按
   episode-excluded 邻域比较 hard argmin、near-optimal set、cost margin、pair sign 与速度/道路/fold；
5. 若 near-optimal set 显示跨 episode 一致性，再预注册 set-valued/WTA 或 cost-gap-weighted
   absolute distillation；否则应检查可观测输入是否缺少决定最优 absolute center 的状态，而不是
   回到已否决的 residual/warm-conditioned 路线。

权威哈希：

```text
collection manifest    27d30dd5ede28db6a15a1dde07fac24528eb0c8fa83a9aca660cc29772272941
collection validation  eea9ace1e6741005d7505613b95320e19edd12d7d85346dd18cc9aa8f69cc2a3
T0 manifest            7d4baad1de665b5eeb75cd86886a6d5988a66c3047c69e4b68dc92ceee17d454
T0 validation          60a7ec56a309c8b1851cc08530531f94d35d97ef46d6517de87474ac01796f79
full-rank manifest     f3133cf98297a09345b2bee6262895bcdb5c75e98e874400f4169927699463b3
full-rank validation   4a6b4124979dac90529d78897aabe1ffe501fb9eb4460cbaaef240e5b1abe954
pretrain manifest      19feb1aea6f86190fa8c77f682e4a204abbcd7b4f02fe3cb832aff86d383ae96
pretrain summary       cfbdc6a9f739b5b78cb7c0207012532c3fd3028f97e26cfb3d65e41b41e95f8a
pretrain validation    33d3e020f4e3c97fec5bea0dc3124224759fad9828f5f5825a6742b5de4e5e50
pretrain OOF           d6e14d13ac7ac151c1707fb43baa1c0c8ef245dfa6f39b43f90cd81928d9c793
```

## 14. 2026-09-01 Query absolute target/coherence 零-rollout 诊断

本节执行第 13.4 节建议的 target/coherence 诊断。它只读取已独立验证的 train-derived
132-candidate full-rank bank 与 absolute pretrain OOF；没有新增 Query/DBM rollout，没有训练
模型，没有读取 formal validation/test。v2 增加 speed/variant/fold 与 episode 028 分层，是当前
权威产物：

```text
outputs/query_mppi/query_expected_road_target_coherence_20260901_v2
scripts/model_verify/analyze_query_expected_road_target_coherence.py
scripts/model_verify/validate_query_expected_road_target_coherence.py
```

analyzer qualification 为 `QUERY_TARGET_COHERENCE_DIAGNOSTIC_COMPLETE`，独立 validator 为
`QUERY_TARGET_COHERENCE_INDEPENDENT_PASS`；6 项检查全过。邻居索引、input-target Spearman 与
headline 重构误差为 0，输入距离最大复算差小于 `6.7e-7`。

### 14.1 诊断合同

主口径完全匹配当前 nested OOF：对 query fold，`(fold+1)%5` 作为 selection 排除，其余三个
完整 folds/360 rows 作为邻居 bank。strict no-anchor 输入只含 `history[250,7]`、expected-road
`reference_ego[51,4]` 和 `[vx,yawrate,current_action]`。每个 block 用 fit-only median/IQR
标准化（std fallback）并等 block 加权。

另报告 `strict inputs + exact mean_knots_before`，但它只用于判断 warm 是否携带 target 信息；
不得据此恢复 warm-conditioned residual Actor。所有邻居均跨 episode，且 outer/selection fold
不泄漏。

诊断分别比较：

1. full-rank hard argmin、T0 与 warm 的 absolute knot sigma-RMS 距离；
2. 1%/2%/5% near-optimal candidate set 的 semantic Jaccard、absolute set-to-set minimum/Chamfer；
3. 把邻居 candidate semantic index 映射到当前行 local bank 后的 Query cost（只复用已存 cost）；
4. 64 个 antithetic local pairs 的 sign agreement，并按当前/邻居 pair cost margin 同时大于
   1%/2%/5% 分层；
5. strict、诊断性 +warm、同-cell 最近邻与随机同-cell 对照，以及 speed/variant/fold 分层。

### 14.2 hard argmin 不稳定，但不是主要 Actor blocker

strict 最近邻 hard candidate index 一致率只有 `2.67%`；2% near-optimal set 的 semantic
Jaccard 中位仍为 0。best-vs-second relative cost gap 中位仅 `0.516%`，且 70.17% 的状态在
1% cost 内已有多个 candidates、88.0% 在 2% 内有多个 candidates。因此单一 hard index 确实
存在低-margin 抖动。

但 candidate semantic 不同不等于 cost 完全不可转移：邻居 hard semantic 映射到当前 local
bank 后可恢复 `81.33%` 的 warm→full-rank headroom，2% near-optimal semantic set 的 local
oracle 可恢复 `92.23%`。随机同-cell hard semantic 已能恢复中位 `80.40%`，strict 同-cell
最近邻为 `82.40%`；输入邻近性只增加约 2 个百分点。这个高 recovery 主要说明 local candidate
landscape 平且多解，不证明 absolute action target 可预测。

直接在 absolute knot 空间比较后，hard target 的跨 episode 最近邻距离中位为
`0.9863 sigma-RMS`。2% near-optimal absolute sets 的最小 pair 距离仍为 `0.9572`，只缩短
`2.95%`；5% set 也仍为 `0.9074`，symmetric Chamfer `0.9467`，在 0.25 sigma 内的双向覆盖
中位为 0。故 near-optimal set 并没有把不同 episode 的 absolute action basin 合并成稳定集合。
当前证据不支持 set-valued/WTA 或 cost-gap-weighted absolute BC 直接重试。

local pair sign 也不是只在 tiny margin 翻转：全部 pairs 最近邻 agreement 为 `54.04%`；要求
query 与 neighbor relative margin 都至少 5% 后，覆盖仍有 `50.27%`，agreement 反而只有
`52.96%`。这说明最后的局部最优方向高度 state-specific；不能用统一 pair sign 监督替代
absolute target。

### 14.3 主要不一致来自 T0 absolute center

共同 mean J50 为 warm/T0/full-rank `11.3447/7.9399/7.7198`；T0 已恢复 warm→full-rank
headroom 的 `93.93%`，full-rank 相对 T0 只再下降 mean `0.2201`、median `0.1365`。
动作空间中 T0→full-rank 距离中位仅 `0.10 sigma`，warm→full-rank 为 `0.329 sigma`；但三个
OOF Actors→full-rank 已达 `0.757/0.752/0.772 sigma`。因此 Actor 主要没有重建 T0 所在的
absolute center，而不是没有学会最后 0.10 sigma 的 full-rank perturbation。

strict input distance 与 target distance 的逐行 Spearman 中位为 `0.484`；最接近 target 的 fit
row 在 strict input 排名中位只有第 40。strict input top-1/5/20 的最小 target 距离中位为
`0.986/0.734/0.629 sigma`，即使使用 label-aware 的全部 360-row oracle，也只能到 `0.564`。

诊断性加入 exact warm 后，Spearman 升至 `0.582`、target-best input rank 中位升至第 20、
top-1 target distance 降至 `0.823 sigma`。这证明 warm 携带额外 target-location 信息，但改善
仍不足以使 label 唯一/近邻一致，也不授权把 warm 恢复为 Actor 输入。更重要的是，当前 T0
由 warm/best-sampled/weighted-output 有限搜索选择，full-rank 又只在 T0 周围 0.03--0.15 sigma
探测；所以即使最终保存的是 absolute knots，label 仍有明显的 warm/search-path 依赖。

### 14.4 分层与 episode 028

strict hard target/2% set minimum-pair 距离按速度为：

| km/h | hard sigma-RMS | 2% set | set reduction | input-target Spearman | robust pair sign |
|---:|---:|---:|---:|---:|---:|
| 40 | 0.729 | 0.655 | 10.2% | 0.471 | 0.684 |
| 55 | 0.917 | 0.876 | 4.4% | 0.537 | 0.592 |
| 70 | 1.010 | 0.986 | 2.4% | 0.568 | 0.477 |
| 85 | 1.024 | 0.995 | 2.8% | 0.499 | 0.515 |
| 100 | 1.261 | 1.237 | 1.9% | 0.367 | 0.492 |

高速度的 absolute target coherence 更差；因此不能把问题收缩为 40 km/h。四个 variants 的 hard
距离均为 `0.943--1.012`，五 folds 均为 `0.919--1.033`；没有可单独放行的 road 或 fold。

episode 028 的 hard distance 为 `2.626 sigma`，三个 Actor recovery 为
`-53.63/-44.06/-61.40`，确实是 fold 3 极端 cost tail。但排除它后 hard distance 中位仍为
`0.982`，三个 Actor recovery 仍为 `-1.218/-0.958/-0.836`。所以 episode 028 放大了数值失败，
不是 target mismatch 的唯一来源。

### 14.5 裁决与下一步

当前 decision 为：

```text
NO_JUSTIFICATION_FOR_SET_VALUED_ABSOLUTE_BC_RETRY
```

1. 不对同一 warm/T0-centered bank 继续做 hard-label、set-valued/WTA、cost-gap weighting、epoch、
   seed 或网络宽度扫描；
2. 不恢复已否决的 warm-conditioned residual Actor；+warm 结果只是信息审计；
3. 现有证据更符合 **teacher construction 与 strict no-anchor input contract 不匹配**：local
   finite-bank target 依赖 warm/T0 search path，而 Actor 被要求从 physical/expected-road inputs
   直接输出 canonical absolute center；
4. 若继续，先预注册一个小规模 train-only **no-anchor canonical teacher audit**：候选/搜索起点
   必须只由 strict 可观测输入确定，不得以 exact warm/T0 作为搜索中心；旧 warm/T0/full-rank
   只作 cost floor/对照。先测 absolute target coherence、跨 episode Actor learnability 与
   Query direct-cost tail，再决定是否生成完整 teacher；
5. 该 audit 需要新增 Query rollout，必须是独立的小预算机制实验；在预注册前不启动，formal/test、
   wrapper、closed loop 和 actor-visited OAC 继续封存。

权威哈希：

```text
manifest.json      f7422172890c8013708b736b9a9a0eef89c46a79e369997e2cc2a39bcd9a91c4
analysis.json      cf74b99db727801ab66ed46e274ac2153511b26a76b129f096b843500e42c4b9
diagnostics.npz    930cfcacfc8d9d25ac4f9a954873ea69d068f655437c39da7871326caea7e080
validation.json    0c26abe574b23122485ca4f56dcfde334ed79ebf2d97fbbdf6d49d25cc04a831
analyzer            9c877c4d1df83522252f50a9ca5173b9bc590507dcfac6134ee3b88f61be8367
validator           4b1c5e0379ba23b16b52a4f99ea97a7a8ae053ab3174015cf898e5350f405279
```

## 15. 2026-09-01 no-anchor canonical teacher 小规模审计预注册

第 14.5 节路线已在新增 Query rollout 前冻结为下面的 train-only 机制审计。配置文件为：

```text
scripts/model_verify/query_noanchor_canonical_teacher_audit_config_20260901_v1.json
```

审计只从扩展集的 100 个独立 episode 各取 `row_in_episode=3/control_step=325` 一个状态，保持
5 个速度、4 个道路 variant、5 个 fold 与 5 个 repeat 的完整平衡。canonical 初始中心是 strict
可观测 `current_action` 在 8 个 knots 上的重复；不得读取 exact warm/T0/full-rank center 来构造
候选。四轮搜索半径冻结为 `1.50/0.75/0.35/0.15 source sigma`，每轮使用 Sylvester Hadamard-16
与 RMS 归一化 DCT-II-16 两套确定性全秩方向并作正负成对评估，即 `1+4*64=257` 个 Query
rollout/state、总计 25,700。每轮只允许该轮 Query argmin 作为下一轮中心。

旧 warm/T0/full-rank 仅保留为 cost 与 target-coherence 对照，不能进入新 target 的 argmin。另按
outer fold 查询、`(fold+1)%5` 排除、其余 60 个跨 episode fit rows 的 nested 合同，使用 fit-only
robust normalization 的 strict observable blocks 选择最近邻，并把邻居 canonical target 在当前状态
复放一次，共再增加 100 个 Query rollout。formal validation/test、DBM 字段/label、wrapper 与闭环
继续封存。

预注册门槛全部满足后才允许训练小规模 nested-OOF strict no-anchor Actor；任一 coherence/transfer
门槛失败就停止扩大 teacher。门槛为：canonical mean cost 不高于 warm、warm regression fraction
不超过 20%；最近邻 absolute target 距离中位不超过 0.80 sigma-RMS，且相对同子集旧 full-rank
至少下降 15%；input-target Spearman 中位至少 0.58；最后一轮增益不超过 hold→final 总增益的
10%；最近邻 target 的 hold→canonical aggregate gain recovery 至少 25%。这些数值在运行前冻结，
不得用本次结果回调。

### 15.1 审计结果：搜索有效，但 canonical teacher 不合格

审计已完成，输出为：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
query_noanchor_canonical_teacher_audit_20260901_v1
scripts/model_verify/run_query_noanchor_canonical_teacher_audit.py
scripts/model_verify/validate_query_noanchor_canonical_teacher_audit.py
```

实际新增 `25,700` 个 canonical-search Query rollout 与 `100` 个最近邻 target-transfer Query
rollout；没有 DBM rollout，没有 formal validation/test。搜索候选可完全由 `current_action`、固定方向、
固定半径、action bounds 与上一轮 canonical Query argmin 重构，最大误差为 0，证明旧 warm/T0/
full-rank 没有进入候选生成或新 target argmin。

五级 hold/stage1/stage2/stage3/stage4 的 mean Query J50 为
`229.112/37.777/20.667/15.725/14.035`，median 为
`82.123/25.916/10.880/6.881/5.730`。因此确定性全秩搜索确实大幅优于 observable hold；最后一轮
占总 aggregate 搜索增益 `0.786%`，按冻结口径通过预算门槛。但每个状态最后一轮都更换了 winner，
stage3→stage4 增益 median/mean 为 `1.037/1.690`，相对 stage3 cost 的逐状态中位仍为 `14.57%`；
aggregate plateau 指标受极大 hold cost 支配，只能说总增益占比小，不能声称逐状态优化已严格收敛。

关键对照 mean J50 为 canonical/warm/T0/old-full-rank
`14.035/11.049/8.005/7.771`。canonical 相对 warm 的 gain mean/median/P05 为
`-2.986/-2.243/-16.008`，76% 状态回退；故两个 cost gate 均失败。episode 028 的
`614.798` canonical cost 放大了 mean tail，但排除后 canonical/warm 仍为 `7.967/5.753`，
回退率仍为 `75.76%`，不能归因于单一异常 episode。

target coherence 同样更差：strict nearest-neighbor absolute target distance median 从同一 100-row
子集旧 full-rank 的 `1.070` 增至 `1.509 sigma-RMS`，所谓 reduction 实际为 `-41.09%`；
input-target Spearman median 从 `0.510` 降至 `0.476`。最近邻 target-transfer 相对极差 hold 的
aggregate recovery 为 `85.95%`，机械通过 25% 门槛，但 transfer mean cost 为 `44.264`，远差于
warm `11.049`；该分母同样被 hold tail 主导，不能覆盖 cost/coherence 失败。

预注册 7 项门槛仅预算占比与 transfer-recovery 两项通过，最终 decision 为：

```text
CANONICAL_TEACHER_AUDIT_FAIL_DO_NOT_SCALE
```

因此没有训练小规模 Actor，也不得生成完整 teacher、进入 Actor-visited OAC、wrapper 或闭环。

### 15.2 独立验证与 batch-shape 数值说明

最终独立资格为 `QUERY_NOANCHOR_CANONICAL_TEACHER_AUDIT_INDEPENDENT_PASS`。validator 对 25 个
speed×fold 代表状态复放全部 257 candidates（6,425 rollout），并复放全部 100 个 transfer target；
candidate cost、teacher cost 与 transfer cost 最大误差均为 0。来源字段、比较字段、候选 raw/clipped、
stage winner、nested coherence 与 decision 重构也全部为 0 误差。

验证器的第一次 Query 复放曾把 257 candidates 合成一个 batch，而生成器使用冻结的
`1+64+64+64+64` batch 边界。该实现使大 tail candidate 的最大绝对差为 `0.1353`，但 selected
teacher 误差仅 `9.54e-7`、100 个 transfer 误差为 0。修正 validator 以匹配冻结 batch contract 后，
全量误差为 0；这是 validator batch-shape 的 CUDA 数值复现问题，不是数据、teacher 或门槛变更。

### 15.3 零-rollout near-opt tie-break 诊断

为判断失败是否只是 hard argmin 抖动，已在保存的 257-candidate bank 内做 post-hoc、零新增 rollout
扫描：对相对 best cost 的 `0/1/2/5/10/25/50/100/200/500%` tolerance，先保留 near-opt
candidates，再选择离 observable hold 最近者。输出为：

```text
outputs/query_mppi/query_noanchor_canonical_tiebreak_diagnostic_20260901_v1
scripts/model_verify/analyze_query_noanchor_canonical_tiebreak.py
scripts/model_verify/validate_query_noanchor_canonical_tiebreak.py
```

该诊断资格为 `QUERY_CANONICAL_TIEBREAK_DIAGNOSTIC_COMPLETE`，独立资格为
`QUERY_CANONICAL_TIEBREAK_INDEPENDENT_PASS`，所有数组/最近邻/decision 重构误差为 0。结果是：

* 0--10% near-opt 范围内，target distance median 仍为 `1.509--1.457 sigma`，Spearman median
  仅 `0.475--0.488`，mean cost 已从 `14.035` 增至 `14.914`，warm 回退仍为 76%；
* tolerance 放到 500% 才把 target distance 降至 `0.723`、相对旧 full-rank 改善 32.47%，但
  Spearman 仍为 `0.5778 < 0.58`，mean cost 恶化到 `37.596`，94% 状态输给 warm；
* 没有任何 tolerance 同时通过 cost 与 coherence 门槛，decision 为
  `NO_TIEBREAK_WITHIN_CANONICAL_BANK_MEETS_COST_AND_COHERENCE`。

所以失败不是给当前 hard argmin 加 near-opt/tie-break 就能修复。当前 `current_action hold + greedy
local recenter` 在不同状态进入不同低成本 basin；低成本 candidate region 本身不满足 strict-input
absolute target coherence。下一步不得增加 BC epoch/seed、扩大同一 teacher bank 或训练 Actor。
若继续 canonical teacher 路线，必须另行预注册**不同的 candidate construction**：优先验证由
expected-road geometry 构造的 deterministic reference-conditioned 初始解，或与状态无关的全局
multi-start/低差异 absolute bank，再做统一而非 path-dependent 的选择；先用同一 100-state
train-only audit 证明 cost 与 coherence 同时改善，才允许扩大数据。

权威哈希：

```text
canonical manifest.json    fe7e465547614fd35672ec902bec64c072b7484aa52602d93c521670b78748c5
canonical summary.json     3e8e10e0b857d80f23cb489e149404188561284271a782a9928384b840326cf0
canonical audit.npz        2f43d5352bf1d2fadfcd7b6b7459b95c29eb4f0cf345c92da5b4c377df3bac86
canonical validation.json  1535c535eeb1a24d0609a4355ddd841dae52a075138b55987d7ccc3bd9893224
canonical runner           980b1b113645010590438b3cdc38477e71ef863fa0cf04e885a302ebc9456708
canonical validator        a16456a311c1b6e0987f3fcf091a4761fc11f94be857ed448e8310c3f51620b2
tiebreak manifest.json     ff19cebd609fcb8d054b16ef85bb0c53b8169393000eed2f34411b7e346453f0
tiebreak analysis.json     24b9a571940c4277b86fcd3052b22dfb0ef53fbdfff4a735497f4d08aec690f3
tiebreak diagnostics.npz   dbd1f4afcf1c2ba68ccecefbd63179db62d102f9da7a19f69b1634a43d25b343
tiebreak validation.json   88895875368229a9fbe3ad69367d4a6bc168552609f3f801b549cffd82359df3
tiebreak analyzer          65b1957e97e89533c3a2c2ad35df8feb3ba8382b1890573e70dd3b9cdc8c2acb
tiebreak validator         b8abb72e54ed2ed15057e539bca0a5647af75ca1a680d254a8799e6f185fe7bd
```

## 16. 2026-09-02 Query 多分支多轮 forward-response landscape pilot 预注册

用户否决一次性铺大量 global candidates，要求复用历史上已验证的“probe → 经验梯度/轨迹响应 →
更新中心 → 再 probe”机制提高探索效率。本阶段只绘制 Query low-cost landscape，不生成 teacher，
不训练 Actor/Critic。冻结配置为：

```text
scripts/model_verify/query_forward_response_landscape_pilot_config_20260902_v1.json
```

### 16.1 历史依据与边界

历史固定 256-rollout DBM 实验中，1/2/3/4 轮 empirical trajectory-response guided search 的
跨 seed best cost 约为 `6.546/2.256/2.040/1.999`；多轮反馈明显优于把预算一次用完。
另一个 51-rollout forward-response 实验中，combined cost/trajectory response mean cost
`6.338` 优于相同预算 blind exploration `6.974`，second-pass improvement fraction
`91.5%`。因此本 pilot 优先使用完整 weighted trajectory-residual response，scalar cost fit 只作
辅助方向。

允许的“gradient”仅是**同一状态、当前绝对中心、真实 Query antithetic rollout**产生的 empirical
finite-difference/response。禁止读取 Query/DBM analytic gradient，也禁止使用已否决的跨状态
learned/frozen Critic gradient。每个 response proposal 必须再次用真实 Query direct J50 评分，
拟合值不能作为结果。

### 16.2 20-state 与五分支合同

从已验证的 100-state canonical audit 中，对 5 个速度 × 4 个 variant 各取一个
`row_in_episode=3/control_step=325` 状态，共 20 个独立 episodes、每 fold 4 rows。默认 cell fold
为 `(speed_index+variant_index)%5`；40 km/h variant 1/3 的 fold 互换为 3/1，以保留已知
`episode_028` tail，同时维持 fold 平衡。

四条 strict no-anchor canonical 初始分支为：current-action hold、current-action 线性衰减到 zero、
fixed zero，以及前一轮已验证的 no-warm canonical result。第五条 old-warm branch 仅是 shadow；
它不得参与 canonical winner、后续 teacher 或 Actor 输入。T0/old-full-rank 只保留为 cost comparator。

### 16.3 四轮自适应搜索与预算

每条 branch 做四轮。每轮以当前 incumbent 为中心，使用 16 个 full-rank directions 的正负 pair，
即 32 个新 probes；四轮 basis 依次为 Hadamard、DCT-II、固定 QR seed 260902/260903，半径为
`1.00/0.50/0.25/0.10 source sigma`。根据实际 clipped displacement、Query cost 与完整 weighted
trajectory residual，拟合：

```text
delta_cost ~= delta_u @ g
delta_residual ~= delta_u @ R
```

生成 negative-cost-gradient、damped Gauss--Newton trajectory-response 与 aligned blend 三个方向，
每个方向评估 0.5/1.0 两个 trust factors，共 6 个真实 Query proposals。下一轮 center 是 incumbent、
32 probes、6 proposals 的 exact Query argmin；各 branch 独立保留，不能在第一轮后坍缩成单一 winner。

初始 5 centers 各评估一次，之后 `4 rounds × 5 branches × (32+6)`，合计
`765 rollouts/state`、20 states 共 `15,300` 个新 Query rollouts。这比一次性 2k--3k bank 小得多，
并且除第一轮外每批采样都依赖上一轮真实结果。

### 16.4 扩展到 100 states 的冻结门槛

只有 independent replay 通过且以下门槛全部满足，才允许将完全相同合同扩到剩余 80 states：

1. 四条 canonical branches 的 final per-state best mean cost 不高于 old warm，warm regression
   fraction 不超过 25%；
2. 相对 prior no-warm canonical 的 aggregate gain recovery 至少 10%，五个速度组 mean gain 均
   非负；
3. 前两轮 response proposals 在至少 50% branch-round contexts 上进一步优于 probe-only best；
4. trajectory-response 方向的真实 proposal improvement positive fraction 至少 50%；
5. 所有候选、response、cost、来源和预算均通过独立重构/Query 复放。

若 response 有效但 cost gate 失败，只允许在同一 20 states 调整 seed geometry/radius；不得增加
states 或训练 Actor。若 response gate 自身失败，则停止该 construction，也不得退回一次性大 bank。

### 16.5 20-state pilot 结果与独立验证

上述冻结 pilot 已执行完成，权威输出为：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
query_forward_response_landscape_pilot_20260902_v1
scripts/model_verify/run_query_forward_response_landscape_pilot.py
scripts/model_verify/validate_query_forward_response_landscape_pilot.py
```

实际新增 Query rollout 为预注册的 `20 * 765 = 15,300`。没有生成 teacher，没有训练
Actor/Critic，没有读取 DBM 字段/label/gradient，也没有消费 formal validation/test。独立资格为：

```text
QUERY_FORWARD_RESPONSE_LANDSCAPE_INDEPENDENT_PASS
```

validator 独立重构 source selection、四套 full-rank basis、probe raw/clipped knots、实际 clipped
displacement、robust ridge 权重、scalar gradient、完整 trajectory-residual response、damped
Gauss--Newton step、三个 response directions、六个 proposals、逐轮 winner chain 和 canonical/shadow
隔离，最大误差均为 0。存储 residual 的平方和重构 Query cost 最大误差为 `4.72e-4`。另对 5 个速度
代表状态（含 `episode_028`）按生成时的 `5 + 4*5*(32+6)` batch 边界完整复放，共 3,825
rollouts，cost/residual 最大误差均为 0。

四条 canonical 分支逐轮 per-state best 的 mean/median J50 为：

| round | mean | median | max |
|---:|---:|---:|---:|
| initial | 36.265 | 4.887 | 614.798 |
| 1 | 30.613 | 2.131 | 567.953 |
| 2 | 30.129 | 1.776 | 565.315 |
| 3 | 29.972 | 1.718 | 564.020 |
| 4 | 29.915 | 1.683 | 563.393 |

旧 no-warm canonical / 本 pilot final 的 mean cost 为 `36.455/29.915`，aggregate relative
reduction 为 `17.94%`；20/20 状态都优于旧 canonical。old warm mean 为 `30.319`，pilot final
mean 为 `29.915`，warm regression 为 `2/20 = 10%`。五个速度组相对旧 canonical 的 mean gain
分别为 `15.401/2.543/2.837/7.004/4.915`，全部非负。因此预注册的四个 cost/coverage gate
全部通过。

response 机制本身也通过。前两轮 response proposal 进一步优于 probe-only best 的比例为
`76.25%`；trajectory-response 两个真实 proposals 在全部 canonical branch-round 中优于
probe-only best 的比例为 `68.125%`。按 round 的任一 response direction 改善比例为
`86.25/66.25/77.50/78.75%`。320 个 canonical branch-round 中，下一轮 center 来源为
incumbent/probe/response `12/61/247`；response 不是只在第一轮有效。trajectory residual fit 的
relative error median/mean 为 `0.386/0.404`，明显好于 scalar-cost fit 的 `0.941/0.851`，与历史
经验一致：完整 trajectory response 是主方向，scalar gradient 只是补充。

### 16.6 low-cost region 的分布解释与 tail

本 pilot 证明“之前只是没搜索到低 cost 区域”是主要问题之一，但不是“存在一个统一唯一的 absolute
center”。20 个状态的最终 canonical cost median 为 `1.683`。按“cost 不超过该状态 canonical best
的 10%，且 sigma-RMS 距离至少 0.5”计数，1/2/3/4 个 distinct basins 的状态数为
`5/10/3/2`；即 15/20 状态至少有两个明显分离的 near-best basins。四条 canonical 分支 final
pairwise distance 的 median/mean/P95 为 `0.961/0.960/1.660 sigma-RMS`。最终 winner 来源为
current-action hold / current-action-to-zero / fixed-zero / prior canonical `2/13/3/2`，没有单一起点
支配所有状态。

因此 forward-response multi-start 很适合继续画 Query cost landscape；但多盆地证据继续反对把单个
hard argmin 当成 strict no-anchor absolute BC target。即使扩展到剩余 80 状态，也仍只是 landscape
coverage，不自动授权 teacher、Actor 或 wrapper。

`episode_028` 必须单列。它从旧 canonical `614.798` 降至 `563.393`，但仍输给 old warm
`535.292`；同一 forward-response 合同的 warm-shadow 可到 `534.062`。另一个 warm regression
是 `episode_085` 的 `1.526` 对 `1.514`，差值只有 `0.0122`。排除 `episode_028` 后，剩余 19
状态 final mean/median 为 `1.837/1.679`，old warm mean 为 `3.742`，warm regression 为
`1/19`。所以 pooled mean gate 虽真实通过，却不能掩盖 `episode_028` 对 warm/no-anchor 起点高度
敏感的独立 tail；扩展后仍必须同时报告 pooled、exclude-028 和 episode-028。

按预注册 routing，本方法现已具备“冻结同一合同并扩展剩余 80 个 train-only 状态”的资格。扩展时
不得改 basis、radius、ridge、damping、line factors、分支或预算，也不得把 old warm shadow 纳入
canonical winner。应先得到完整 100-state landscape 分布并独立验证，再决定是否需要设计
reference-conditioned canonicalization；当前不训练 Actor。

权威哈希：

```text
manifest.json    adab2febe14ea30ba8c5c454e39f18aa38cc63a684b4497a51765f7729f97e98
summary.json     b5224d2c1016e3a8635b53994f8f3d81c9b3291edbdfb3693efd1d94f7cbd950
landscape.npz    0290024e7e9d37738b1c60255bc1696e1025469ceff5fac8b5a8b893df6e601d
validation.json  7b4817855da3f113a7bbfd1c46d06ffa95b887ae7f0535ca19610bbd5ed6a499
config           cd650cc9e73e5b089c2172c7aa282ae70df4300250c05e7639a6461ce64863cf
runner           1e3003d6e85470393919a063a61d5a61a7d8425fd8732cf8de0fa62a5d5934ff
validator        59fcef179d1d9fa25f12f169173271a0077658e75de3125f514817077309ff0f
```

## 17. 2026-09-02 single-center strict Critic → continuous Query OAC 路线冻结

用户确认不修改 Actor 结构、不增加多头，也不继续追求高精度 teacher BC。当前主线改为：单中心
strict no-anchor Actor 只做有限粗初始化；重点严格训练 Twin absolute-cost Critics；Critic 通过
episode-heldout value/ranking 与 fresh Query local-response 门后，直接进入持续 Actor-visited Replay
更新的 terminal contextual-bandit AC。冻结配置为：

```text
scripts/model_verify/query_single_center_oac_config_20260902_v1.json
```

### 17.1 Actor 与 Critic 合同

Actor 保持 `DirectNoAnchorGTXActor` 和唯一 absolute `[8,2]` 输出，输入仍只有
`history[250,7]`、expected-road ego reference 与
`[vx,yawrate,current_acceleration,current_steering]`。warm/T0/teacher、search feedback、gradient
carrier、candidate cost/index 与任何 DBM 字段都禁止进入 Actor。Actor 粗训练最多 50 epochs，只负责
给 AC 一个 bounded、finite、位于数据 support 内的 round-0 center；BC direct cost、teacher recovery
和 hard-label RMSE 不再作为 OAC admission gate。

Critic 使用两个独立 `ConfigurableAbsoluteActionValueCritic`，输入 strict observable context 与明确的
absolute candidate `[8,2]`，学习 fit-only standardized `log1p(Query J50)`。数据同时使用已验证的
600-state/132-candidate full-rank bank 与完整 100-state forward-response landscape；按 state 先采样、
再采 candidate，防止 765-candidate landscape 状态压过其他 500 个成熟 snapshots。所有高 cost、
regression、clipped probe/proposal 均保留。旧 bank 的 warm/T0 搜索路径不影响标量 Critic label 的
有效性，因为 Critic 读取的是明确 absolute action 与真实 Query direct cost，不读取生成路径。

### 17.2 Critic 严格门与 OAC 例外

5-fold 仍按完整 episode 分组。outer fold 只 OOF 评分，`(fold+1)%5` 只做 inner selection，其余
三 folds fit；OOF 不得选择 checkpoint/epoch/seed/阈值。Critic 预训练必须至少 2/3 pooled seeds
同时满足：OOF log-cost Pearson median `>=0.70`、same-state pair sign accuracy median `>=0.75`、
landscape bank-gain recovery median `>=0.50`；在 OOF Actor center 周围用 0.02-sigma Hadamard-16
antithetic fresh Query probes 得到的 empirical FD gradient，Critic cosine median `>=0.70`、P10
`>=0`、median norm ratio 在 `[0.5,2.0]`。checkpoint reload、split、fresh probes 与 Query cost 必须
独立复放。

上述门禁止“离线训练一次、冻结 Critic、直接用其 action gradient”路线。若全部通过，则允许的是
历史上已有边界的 continuous OAC 例外：每轮 Actor-visited candidates 全部由 fresh Query direct
rollout 标注并进入 Replay，Twin Critics 持续更新，Actor 才可通过当前 conservative Twin gradient
做一个 capped 小步；真实 Query inner-selection cost 决定 checkpoint，Critic 不能自行宣布通过。

### 17.3 首个 OAC mechanism pilot

Critic 门通过后才运行 fold 0、三 seeds、10 rounds 的小 pilot。每次访问评估 Actor center、32 个
antithetic probes 和 6 个 forward-response proposals，共 39 candidates；每个 Actor update 前做 20
次 Twin-Critic updates，每轮只有一次 Actor update，输出累计移动只做 `0.02 sigma-RMS` cap，不把
自然小步强制放大。任务是 terminal contextual bandit，不构造 vehicle next-state、Bellman bootstrap、
target network 或 entropy。所有失败 candidates 留在 Replay。

round 0 必须参与 selection；inner-selection selected checkpoint 相对 round 0 的 mean cost 不得增加，
median/P05 gain 必须非负，fresh local Critic gate 必须保持通过，并完成独立 Query replay。只有该
mechanism pilot 通过后，才允许扩到其余 OOF folds。warm 始终只是外部 shadow comparator 与未来
hard guard，不进入 Actor 输入/loss 或 canonical selection。formal validation/test、wrapper 与闭环
继续封存。

full-100 扩展采用增量 artifact：先逐 hash/validation 读取已经合格的 20-state pilot，只对其余 80
个 source-audit rows 新增 `80*765=61,200` 次 Query rollout，再按 `source_audit_row=0..99` 合并。
不得重复运行已完成的 15,300 rollouts，也不得修改旧 pilot runner/artifact。full-100 validator 必须
证明嵌入的 20 rows 与旧 pilot 逐数组一致，并独立重构/复放新增 rows。

### 17.4 full-100 landscape 与 absolute Replay 已完成

冻结合同已按增量方式扩展到 100 states。旧 20-state pilot 原样嵌入，只新增其余 80 states 的
`61,200` 个 Query rollouts；完整 artifact 表示 `76,500` 个 rollouts。输出为：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
query_forward_response_landscape_full100_20260902_v1
scripts/model_verify/run_query_forward_response_landscape_full100.py
scripts/model_verify/validate_query_forward_response_landscape_full100.py
```

独立资格为 `QUERY_FORWARD_RESPONSE_FULL100_INDEPENDENT_PASS`。旧 pilot 逐数组嵌入误差、source/
basis/initial/probe/fit/proposal/selection/canonical 重构误差均为 0；10 个新增代表 rows 共 7,650
rollouts 的 Query cost/residual 复放误差为 0。完整 100-state canonical final cost min/median/mean/max
为 `1.309/1.670/7.773/563.393`，old warm 为 `3.137/11.049`（median/mean），prior no-anchor
canonical 为 `5.730/14.035`。相对 warm 的 gain median/mean 为 `1.138/3.276`，warm regression
为 2%；相对 prior 的 gain median/mean 为 `3.795/6.262`，aggregate recovery 为 `44.62%`。
40/55/70/85/100 km/h 相对 prior 的 mean gain 分别为
`5.771/2.933/4.126/7.803/10.677`。前两轮 response improvement fraction 为 `73.88%`，完整
trajectory-response improvement fraction 为 `68.38%`。因此 20-state pilot 的正结果不是抽样偶然。

随后构建组合 absolute-cost Replay：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
query_single_center_absolute_replay_20260902_v1
scripts/model_verify/build_query_single_center_absolute_replay.py
scripts/model_verify/validate_query_single_center_absolute_replay.py
```

资格为 `QUERY_SINGLE_CENTER_ABSOLUTE_REPLAY_INDEPENDENT_PASS`。Replay 有 600 states；每个状态保留
132 个 full-rank absolute candidates，100 个 landscape states 另有 765 个 multi-round candidates，
总计 155,700 个有效 `(context, absolute action, Query J50)` labels。cost min/median/mean/max 为
`1.309/3.029/30.315/4575.680`。所有 regression、high-cost 与 clipped candidates 均保留；warm
shadow 与 canonical eligibility 有独立 mask。训练采样合同是先等概率选 state，再在该 state 内选
candidate，禁止对 155,700 行直接均匀采样。独立 validator 对 context、full-rank/landscape 候选、
provenance、mask、fold 与计数的最大重构误差均为 0；formal validation/test、DBM 字段/label 和
Query analytic gradient 均未使用。

### 17.5 单中心 Actor + state-balanced Twin Critic 预训练结果

训练与独立校验输出为：

```text
outputs/query_mppi/query_single_center_actor_twin_critic_pretrain_20260902_v1
scripts/model_verify/pretrain_query_single_center_actor_twin_critic.py
scripts/model_verify/validate_query_single_center_actor_twin_critic.py
```

Actor 结构未修改，仍是单一 `DirectNoAnchorGTXActor -> absolute [8,2]`。100 个 landscape states 的
50-epoch 粗目标为 `observable_current_action_to_zero` 分支最终中心，其余 500 states 用 full-rank
absolute teacher。该 Actor 只提供有限 round-0 初始化，结果不作为准入 gate。其 OOF direct Query
cost median 约 `3.64--4.11`，相对 warm gain recovery 为 `-1.67/-1.60/-1.48`，regression 为
`60.3/57.7/58.2%`；这再次说明单解 BC 不精确，但没有因此修改结构或增加多头。

Twin Critics 使用两个独立 base `ConfigurableAbsoluteActionValueCritic`，pair-delta head 关闭。每次
先选 fit state，再取 48 candidates（warm index、fit-state min、23 uniform、23 log-cost
quantile-balanced），学习 state-balanced、fit-only standardized `log1p(Query J50)`，Huber value 加
same-state ranking；240 epochs，3 seeds，完整 nested 5-fold OOF。每个 pooled seed 的结果为：

| seed | state Pearson median | pair-sign median | landscape recovery |
|---:|---:|---:|---:|
| 0 | 0.8121 | 0.7200 | 0.5273 |
| 1 | 0.8222 | 0.7334 | 0.5604 |
| 2 | 0.8322 | 0.7365 | 0.5160 |

因此 Pearson gate `>=0.70` 和 landscape recovery gate `>=0.50` 均为 3/3 seeds 通过，但
same-state pair-sign gate `>=0.75` 为 0/3 通过；总资格为
`QUERY_SINGLE_CENTER_PRETRAIN_OFFLINE_FAIL_NO_OAC`。这不是“低 cost 区域不存在/完全学不到”：Critic
已经能学到粗 cost 面并在 100-state landscape bank 回收约 52--56% 可用 gain；失败点是同状态细粒度
排序仍不足以作为 Actor action-gradient 的可靠依据。不同 OOF fold 的 landscape recovery 也从
约 `0.06` 到 `0.89` 波动，说明低 cost argmin 的 episode 泛化仍对 seed/fold 敏感。

独立 validator 重载全部 15 个 Actor/Twin-Critic checkpoints，重新计算 OOF，并用真实 Query 重放
1,800 个 Actor actions。checkpoint split、Actor knots、Actor Query cost、Twin conservative log-cost
与 strict no-anchor invariance 的最大误差全部为 0，最终资格为
`QUERY_SINGLE_CENTER_PRETRAIN_INDEPENDENT_CONFIRMED_FAIL_NO_OAC`。由于 offline pair-sign 前置门已明确
失败，预注册的 fresh Actor-center 57,600-rollout FD gate 被停止，continuous OAC 未启动。不得把
Pearson/recovery 的通过解释为可以绕过 pair gate，也不得降低阈值后继续 AC。

下一步只允许做 train-only ranking 诊断：按 full-rank/landscape、cost gap、cost quantile、速度、
clipped/source/round 以及已有 antithetic probe pair 分解 OOF pair-sign，判断失败来自 near-tie 标签、
低-cost tail coverage，还是跨 episode state encoder 泛化。诊断前不得盲目增加 epoch/seed；若需要
补数据，应在失败 slice 上做新的 Actor-independent same-state Query probes，并继续保留所有回归。

权威哈希：

```text
single-center config         70e5299ac92af0fec343bca99c16067c64d38986686e0406e16f167f3414ca81
full100 manifest.json        e0af272f100f36c42ae05d7604e154d66acc707a5f2a8304ca6106348010cdf0
full100 summary.json         d5b29e94ba92f1cf2d34e0e09da1a54f4243a9fc12d068bddca2dadd5b432c07
full100 landscape.npz        84f3288b770214e01d5e8947dc55a4d5e88ab795ae159cd7acc7153bb921c031
full100 validation.json      7c22684b6005593654f31be8d2679393836fca59df2ad2158ce2125a9b7ec10b
Replay manifest.json         110361979fe7eb566e835ab01425bad10eaefe36a73f8c4cbb8a29e0b3ec8634
Replay summary.json          3b6e0efe188c339f4d20a6864e6e8950ae299d18c040306a550eb1d26ee62309
Replay replay.npz            4b85b0226bacbac88c996bcf2cdae1b171bf85de8841babbe9eaa00a56da148c
Replay validation.json       5d59691e0d5342816a74a500ad40cf19c4b3118cd83d3e7f6eb04df68ee0835e
pretrain manifest.json       30df445990aaec277bc3019eb6fc17a3b6a77c1095e0bef012b56081df3ea5c4
pretrain summary.json        d3bf2704fa6d69e240c6cb898cad44a5d9ff2752a1a8b9ccc81cd6be3be83482
pretrain oof_predictions.npz f16be72709a491ea2be6d7d855f171b6a2b06f4985d36784bde1c36dcf79f2ee
pretrain validation.json     b8e03c05b243fd1849b7aaa73490af3251eb5a3aed39ba63635d54054471f576
Replay builder               dbc3ff48c07004994ba041bde1b6e0d3f751089c82f524c9725cfc340197c45e
Replay validator             7ba8e5abc0aa1d559de3acf18a075f3ba1d01d3a448693e62eeefd2016b5a1c5
pretrain trainer             469db72b1ca73fe4f39d7a9f44ede19794bab91730fe1f342e8dd73a049b8e5c
pretrain validator           adff9fd0c5f780814f3d69a9000e12f42e32e9defe078dc47daed599136f9c75
```

### 17.6 OOF ranking slice 诊断

在不新增 Query rollout、不改模型的前提下，对保存的 OOF Twin conservative predictions 做了
full-rank/landscape、cost gap、low-cost quantile 与 exact antithetic pair 分解：

```text
outputs/query_mppi/query_single_center_critic_ranking_diagnostic_20260902_v1
scripts/model_verify/analyze_query_single_center_critic_ranking.py
scripts/model_verify/validate_query_single_center_critic_ranking.py
```

独立资格为 `QUERY_SINGLE_CENTER_CRITIC_RANKING_INDEPENDENT_PASS`，所有数组复算误差为 0，decision
为 `GLOBAL_VALUE_SIGNAL_PRESENT_BUT_LOW_COST_LOCAL_ORDERING_NOT_READY_FOR_AC`。主要证据为：

* 500 个仅 full-rank states 的 per-state pair median 为 `0.690/0.700/0.702`，100 个 dense
  landscape states 为 `0.858/0.865/0.858`；pooled gate 失败主要来自前者，而不是新增 landscape
  数据不可学；
* 按 candidate source 汇总，full-rank pair accuracy 为 `0.676/0.678/0.683`，landscape 为
  `0.868/0.868/0.866`；
* 当真实 `|delta log-cost|` 至少为 `0.05` 时 accuracy 为 `0.756/0.755/0.758`，至少为 `0.10`
  时为约 `0.783`，说明 near-tie 确实贡献了 pooled miss；
* 但 lowest-cost 5% candidates 内的 pair median 只有 `0.555/0.563/0.567`，lowest 10% 只有
  `0.563/0.564/0.578`，low-cost tail 内部接近随机排序，不能只把失败归因于无关紧要的 near-tie；
* full-rank exact antithetic pair accuracy 在 `0.03/0.06/0.10/0.15 sigma` 下均仅约
  `0.61--0.63`；landscape round 0/1/2/3 随半径缩小为约 `0.84/0.76/0.71/0.62`，其真实 pair
  `|delta log-cost|` median 同时从 `0.678` 降至 `0.067`。

结论是：当前 Critic 已有 global/coarse value signal，dense landscape 本身也有效，但低 cost 区域的
小半径局部顺序没有学稳；这正是 AC action gradient 所依赖的部分。下一步不得增加同一数据上的
epoch/seed 或放宽 gate。应从 500 个 sparse rows 中按 episode/fold 平衡分批选择失败 states，围绕
strict no-anchor OOF Actor center 或 Actor-independent low-cost center，新增多尺度 same-state
antithetic Query probes；先做小批、重训、复测 low-tail/exact-pair，再决定是否扩下一批。所有新旧
regression 必须保留，Actor 结构与输入合同不变。在该局部门恢复前，fresh Actor-center FD 与 OAC
继续禁止。

权威哈希：

```text
ranking manifest.json    1e03857c2fe84fecf808840c03ead6763b8ded564830ff13795416f6aeff312f
ranking summary.json     3a43b5b62a5f39c87d7442c52f109d8c1dfb19a8611424c603567a969440c236
ranking diagnostics.npz 4833ad9832eda125bb154a5218b688b3fc4eb96f7d340cdbe9174b0efc5a6089
ranking validation.json a4e94a4705b73c049db22dce6cf8aad41f6f5db9dca1b13fd1117ad1386893e6
ranking analyzer        fedd06193ded5e3768f85bf20c666d30204d7e04a846e64af59d1c8da28bed86
ranking validator       7d53aacafb9ffab31cb226bf8a9a2b17ae2c0b4c195a036ef26d9c88a72f38d3
```

### 17.7 Stationarity-aware 重新判定（覆盖 17.5--17.6 的一票否决解释）

用户指出历史实验已经证明：低梯度/近平坦区域不必辨识精确方向。复核主 review §11.83.4--11.83.5
后确认，17.5--17.6 将未按梯度强度分层的 pair-sign 失败直接解释为“不能进入任何 AC 阶段”过严。
历史 bank-best 真梯度 norm 仅为 warm 的约 1/138；该处 cosine/sign 本来病态，真正要求是 Critic
不制造虚假大 norm、Actor 保留梯度幅值并允许 stay，以及真实环境小步不恶化。

新的冻结合同与产物为：

```text
scripts/model_verify/query_single_center_stationarity_reassessment_config_20260902_v1.json
scripts/model_verify/run_query_single_center_stationarity_reassessment.py
scripts/model_verify/validate_query_single_center_stationarity_reassessment.py
outputs/query_mppi/query_single_center_stationarity_reassessment_20260902_v1
```

对 3 seeds × 600 个 episode-heldout OOF Actor centers，各用 `0.02 sigma` Sylvester-Hadamard-16
antithetic probes 生成真实 Query `log1p(J50)` FD；另对未归一化 Critic gradient 做
`eta=0.001`、逐分量 `0.02 sigma` cap 的真实 Query step replay。共新增并独立重放
`3*600*(32+1)=59,400` 个 Query rollouts。Actor 未更新，formal validation/test、DBM 字段/label、
Query analytic gradient 均未使用。独立资格为
`QUERY_SINGLE_CENTER_STATIONARITY_REASSESSMENT_INDEPENDENT_PASS`，probe、cost、true FD、Critic
autograd、分层 mask 和 routing 最大误差全部为 0。

每个 seed 内以 true fresh-FD norm bottom 25% 定义 flat，其余 75% 为 nonflat。flat 不设方向门。
结果为：

| seed | nonflat cosine median / P10 | nonflat norm ratio median | flat step mean / P05 / worst gain |
|---:|---:|---:|---:|
| 0 | 0.764 / -0.200 | 0.576 | +0.046 / -0.0075 / -0.061 |
| 1 | 0.800 / -0.036 | 0.705 | +0.053 / -0.0037 / -0.041 |
| 2 | 0.789 / -0.291 | 0.608 | +0.023 / -0.0150 / -0.043 |

修正后的解释是：

1. flat 层未归一化小步 mean 为正、负 tail 很小，低梯度方向不准不是当前主要阻断；不得再要求
   flat cosine/pair-sign 过门，也不得为此盲目加数据；
2. 三 seeds 的 nonflat cosine median 与 norm ratio 均通过，说明主体 Actor-center 方向已有用；
3. 但 nonflat cosine P10 仍为负，表明问题不全来自 flat/near-tie，仍有约 10% 有明显梯度状态的
   方向尾部；因此当前 checkpoint 仍不能直接无条件更新 Actor；
4. 已有 global value Pearson 与 landscape recovery 3/3 seeds 通过，所以 continuous OAC 流程不再
   被旧 pair-sign 阻断。允许的下一阶段是 **Actor 冻结的 Actor-visited Critic burn-in**：采 Actor
   center、antithetic probes 与 response proposals 的真实 Query cost，全部好坏样本进入 Replay，
   只更新 Twin Critics；
5. burn-in 后在同一 OOF Actor centers 重做 stationarity-aware fresh FD。至少 2/3 seeds 的 nonflat
   cosine median `>=0.70`、P10 `>=0`、norm ratio median `[0.5,2.0]` 后，才开始 capped Actor 小步；
   flat rows 继续用保幅值/stay/真实 Query checkpoint gate，不要求方向。

最终 routing 为：

```text
GLOBAL_VALUE_PASS_ALLOW_FROZEN_ACTOR_CRITIC_BURNIN_THEN_REPEAT_FD
```

它覆盖旧的“pair-sign 0/3，因此整个 OAC 不得开始”的解释，但不把当前 Critic 宣布为可直接更新
Actor。当前可以继续 AC 数据闭环的 Critic burn-in 阶段，不能跳过 burn-in 直接做 Actor update。

权威哈希：

```text
stationarity config          7499cfe961a204831dced1be92dd0bd5cbe326095adeee4f574c72f4755ef627
stationarity manifest.json   dbf795d437cb39f6dddd0661cfb3b39d210cf7e651dc86bb635bdade65c7caac
stationarity summary.json    f2d0d16ce55bb2e2beae3ff187ca055e2ee92e495bb23bc3a613f77560fca60b
stationarity audit.npz       062d5020e0b6e5bdd819f8e94c6a703bec5fa9816a1184f64e8ea5c09769ae92
stationarity validation.json 7b891bcd0c5604e2e228a9c725e2097be1dbcb193775218d7f9ee63ebecce721
stationarity runner          313f7e27f4befc8d46b3aa1785b70c2d79f150484e61922d0294b50d5cbf0962
stationarity validator       fa66c9f5058fa928ace8adc2aaf3e8643c4c8df31a60d8a2508d4021281d2f07
```

### 17.8 受控 `20 Critic : 1 Actor` Query OAC pilot（覆盖 17.7 的首次更新硬门）

用户进一步指出：要求 nonflat fresh-FD P10 在第一次 Actor 更新前就非负，仍然把 continuously
refreshed AC 当成了 frozen-Critic one-shot gradient，要求过高。该判断成立。17.7 的 FD 结果继续作为
风险诊断，但“必须先冻结 Actor burn-in、再过 P10 门才允许第一次 Actor update”由本节覆盖。

新的冻结实验合同、runner、validator 和产物为：

```text
scripts/model_verify/query_single_center_oac20to1_config_20260902_v1.json
scripts/model_verify/run_query_single_center_oac20to1.py
scripts/model_verify/validate_query_single_center_oac20to1.py
outputs/query_mppi/query_single_center_actor_visited_oac20to1_20260902_v1
```

实验只做 fold 0、3 seeds、10 rounds 的 mechanism pilot。每轮从三个 fit folds 中按
`speed x variant` 各访问一个 context，共 20 states；每个 state 评价 Actor center、32 个 full-rank
antithetic probes 和 6 个 forward-response proposals，共 39 个真实 Query candidates。probe radius 从
`0.20 -> 0.05 sigma`，所有好坏、response、clipped candidates 全部进入 Replay。随后每个 Twin Critic
各更新 20 次，Actor 只更新 1 次；Actor 使用 conservative `max(Q1,Q2)` 的 mean-log-cost 梯度，LR
`2e-6`，以当前 inner-selected Actor 为 output-trust reference，每轮输出变化仅做 cap-only
`0.02 sigma-RMS`。round 0 参与真实 Query inner selection，不强制采用 latest。没有 next-state、Bellman
bootstrap、entropy、Query 解析梯度、DBM 字段/label、formal validation/test。

每 seed 产生 7,800 条 Actor-visited Replay，合计 23,400 条；连同 round/selected inner、一次 selected
OOF 与 post-selection FD，共 39,960 个新 Query rollouts。独立 validator 逐组重构全部 600 个
response banks，并重放 candidate/selected/OOF/FD Query cost；所有 action、raw action、cost、true FD、
Critic autograd、checkpoint output 与 forbidden-input invariance 最大误差均为 0，split leakage 为 0，资格为：

```text
QUERY_SINGLE_CENTER_OAC20TO1_INDEPENDENT_PASS
```

三个 seed 的 selected round 为 `10/8/10`。相对各自 round 0 的结果为：

| seed | inner gain mean / median / P05 | OOF gain mean / median / P05 | OOF worst |
|---:|---:|---:|---:|
| 0 | +0.075 / +0.040 / -0.728 | +0.091 / +0.038 / -1.234 | -2.326 |
| 1 | +0.033 / +0.004 / -1.254 | -0.054 / +0.021 / -2.265 | -9.196 |
| 2 | +0.277 / +0.065 / -0.828 | +0.130 / +0.064 / -1.793 | -4.564 |

合并 3 seeds 的 360 个 OOF seed-state，mean/median/P05/worst gain 为
`+0.0556/+0.0329/-1.6665/-9.1960`，`64.2%` states 改善。三个 seed 的 inner mean 和 median 均改善，
OOF mean 为 2/3 改善且 OOF median 为 3/3 改善，证明 **不先满足 P10>=0 也可以进行小步 AC，并得到可
复现的真实 Query 主体收益**。但 inner P05 三 seed 均为负，所以原 summary 按旧 outcome gate 仍记为
`QUERY_SINGLE_CENTER_OAC20TO1_INNER_GATE_FAIL_PENDING_INDEPENDENT_VALIDATION`。这不再解释成“Actor
不能更新”；它只说明该 checkpoint 还不能扩 folds、部署或取消 tail guard。

Actor 每轮实际移动只有 `0.000526--0.000879 sigma-RMS`，所有 trust projection 都为 1，说明收益不是
由 cap 强行制造，且当前 10 轮仍处于很保守的小步区。第 10 轮 fresh candidate Critic 的 center-relative
sign mean 为 `0.796/0.796/0.807`，centered log-cost Pearson median 为
`0.890/0.888/0.891`，bank recovery median 为 `0.968/0.771/0.952`；持续 20:1 更新能够维持可用的
Actor-visited 局部排序。

post-selection nonflat FD cosine median/P10 为 `0.805/-0.501`、`0.825/-0.372`、`0.854/+0.317`，norm
ratio median 为 `0.545/0.628/0.741`。P10 仍只有 1/3 非负，但真实 Query selected Actor 已在 2/3 OOF
mean 和 3/3 OOF median 上改善，进一步证明 P10 应保留为 diagnosis/stop evidence，不能继续当成首次
Actor 更新的一票否决。

当前主要问题已经从“能否进入 AC”转成 **Actor 跨速度聚合方向**。OOF round-0-relative mean gain 按
40/55/70/85/100 km/h 的三 seed 分别为：

| speed | seed 0 / 1 / 2 mean gain |
|---:|---:|
| 40 | +0.029 / +0.004 / -0.007 |
| 55 | +0.146 / +0.112 / +0.325 |
| 70 | +0.449 / +0.447 / +0.680 |
| 85 | -0.071 / -0.165 / -0.030 |
| 100 | -0.097 / -0.667 / -0.321 |

也就是说，当前 unweighted mean `log1p(J)` Actor objective 稳定改善 55/70 km/h，却在 85/100 km/h
一致反向；继续单纯增加同一目标的轮数可能放大该偏置。并且 selected Actor 相对 warm 的 OOF mean
gain 仍为 `-2.456/-3.822/-3.734`，所以本轮是 AC mechanism evidence，不是可部署 Actor 资格。

正式 routing 更新为：

```text
CONTINUOUS_OAC20TO1_MECHANISM_WORKS_OBJECTIVE_AGGREGATION_TAIL_PENDING
```

下一步不回到长期 frozen Actor，也不先追加数据。保持模型结构、Replay、20:1、Query labels、LR 与
cap 不变，先用现有 checkpoint/Replay 做一次 matched-output microstep 的单变量诊断：比较当前
`gamma=0` mean-log aggregation 与 detached/clipped `gamma=1` raw-cost-aware weighting，报告每个速度的
真实 Query gain、P05/worst 和共享 Actor parameter-gradient conflict。只有 gamma=1 在 inner/train-side
改善 85/100 km/h 且不系统性破坏其他速度，才做同预算 10-round 配对 pilot；OOF 不用于选择该目标。

权威哈希：

```text
OAC20:1 config          76800bbfdfac80a2325ae61b718e424dfaeaeba9a695f31ff0fceefbe2cad840
OAC20:1 runner          4ab186d7dcd066eb0854edb7af2cee530660a009cd7b0a7fcb516a7167a5c85f
OAC20:1 validator       3547307752352359658f2b217a0ec24493961886512f7b8af1aea88c78691cfc
OAC20:1 manifest.json   18ed53bc690f34f6385fbd1b735a235e8c2803633ae90317f9fdab0b3c9468b4
OAC20:1 summary.json    9829a97b94948d5fb74b975fbb2d5d4d2712090024a5b511c10af1b42c978cc1
OAC20:1 validation.json b1fd487186b20c1827ce8dcb5dcd2b9d6ec99e70c1e01f701010304cb0f64d8a
seed0 pilot_arrays.npz  addf7117095629c739fcc2f321f1f6c4b9bb419734b84efc5e25a7a8fe8e847d
seed1 pilot_arrays.npz  9a2fff9226e0e2a0e56c6172311e58e9896f3cf75f2149f4bad0596f3694cc5b
seed2 pilot_arrays.npz  0f206952a2c60d03579be871669fc93728d0948cdf573fc5d7a7c0db9971dda6
```

### 17.9 Actor objective aggregation matched-output microstep

按 17.8 的 routing，在不新增模型结构、不更新 Critic、完全不接触 outer OOF 的条件下，对每个 seed
的 selected OAC20:1 Actor/Twin Critics 做了冻结单变量诊断：在同一 360 个 fit rows 上分别计算
`gamma=0` 的 mean conservative log-cost 梯度，以及 `gamma=1` 的 detached
`exp(log1p(cost))` raw-cost-aware 加权梯度。两臂均沿原始参数梯度做确定性缩放，使完整 fit Actor 输出
严格匹配 `0.0005/0.001/0.002 sigma-RMS`，再只在 120 个 inner-selection rows 上用真实 Query 比较。

产物为：

```text
scripts/model_verify/query_actor_aggregation_microstep_config_20260902_v1.json
scripts/model_verify/run_query_actor_aggregation_microstep.py
scripts/model_verify/validate_query_actor_aggregation_microstep.py
outputs/query_mppi/query_actor_aggregation_microstep_20260902_v1
```

独立 validator 重载全部 selected Actor/Twin Critics，重构两类总梯度、五个速度组梯度、六个 matched
Actors，并重放所有 inner Query cost。aggregate gradient 最大误差 `1.4e-9`、speed gradient 最大误差
`1.49e-8`，Actor actions 与 Query cost 最大误差均为 0；资格为
`QUERY_ACTOR_AGGREGATION_MICROSTEP_INDEPENDENT_PASS`。

主比较步 `0.001 sigma-RMS` 下，gamma0/gamma1 的 inner mean gain 分别为：seed 0
`+0.00181/+0.00385`，seed 1 `+0.00015/+0.00505`，seed 2 `+0.02985/+0.02930`。合并 360 个
seed-state 后，gamma0 的 mean/median/P05 为 `+0.01060/+0.00520/-0.13904`，gamma1 为
`+0.01274/+0.00217/-0.13446`。gamma1-minus-gamma0 mean 为 `+0.00213`，但 median 为
`-0.00038`。在 100 km/h 上 gamma1 为 3/3 seeds 更好，在 85 km/h 上为 2/3 更好，因此按预注册
train-side routing 进入连续配对实验。

该诊断同时暴露了 objective conflict，而不是证明 gamma1 已经更优：两臂参数梯度 cosine 只有
`0.492/0.543/0.884`；gamma1 的 normalized weight effective sample fraction 仅
`0.0408/0.0393/0.0378`，等价于每批主要由约 4% 高-cost rows 主导。gamma1 在主步上改善 100 km/h
三 seed、85 km/h 两 seed，但 55/70 km/h 三 seed 都比 gamma0 差。因此只能把它作为需要连续训练
复核的候选，不能用 microstep 结果直接替换 Actor objective。

本阶段 routing 为：

```text
ADVANCE_GAMMA1_TO_PAIRED_CONTINUOUS_OAC20TO1
```

权威哈希：

```text
microstep config          9e4f695b9841a3798b469bae01b3af2793f6e8450906b8573d54f45ca8b7e14d
microstep runner          cb87800a1a56e8e0a1b13b500cadb22a8094aa287c6ad1906a55dc9ac0b1eb48
microstep validator       812aa7edc3bf4f5cfad066b8052246ddbc634a138a8a2695794521479cc188c1
microstep manifest.json   0c4630c5632f0c8f3b42d250ca292aa39b43d1399d6f7671827162c9852e5958
microstep summary.json    1970b9b8029bf4a73048f4580b242749458981a8ed9c20001ca06ca99397e6e5
microstep validation.json 08ec2bb61b4e9dcf7cff0e639ed0dfe8d789d131775778aa9f695e1b33aa4d1b
```

### 17.10 gamma0/gamma1 连续 `20 Critic : 1 Actor` 配对实验（gamma routing 由 17.11 覆盖）

随后从同一组原始 pretrain checkpoints 重新开始，而不是从 17.8 selected endpoint 接着训练；对
gamma0/gamma1 各做 fold 0、3 seeds、10 rounds。两臂共享完全相同的 context queue、basis stream、
每轮 20 个 `speed x variant` fit contexts、每 context 39 个 Query candidates、Twin Critic 每轮各
20 次更新、Actor 每轮 1 次更新、LR、trust 和 cap-only 规则；唯一注册差异是 Actor loss 中 detached
cost weight gamma。每臂每 seed 产生 7,800 条 online replay，六组共 46,800 条，全部好坏与 clipped
candidates 均保留。

产物为：

```text
scripts/model_verify/query_oac_aggregation_ab_config_20260902_v1.json
scripts/model_verify/run_query_oac_aggregation_ab.py
scripts/model_verify/validate_query_oac_aggregation_ab.py
outputs/query_mppi/query_oac_aggregation_ab_20260902_v1
```

独立 validator 重构全部 1,200 个 response banks，重放 46,800 个 online candidates、六组 selected
inner/OOF Actor，并复算 checkpoint selection、分速与 pooled decision。所有 candidate action、raw
action、Query cost、selected action/cost、forbidden Actor input、两臂 round-1 action/cost 最大误差均为
0；两臂 split、visited schedule 和 round-0 输出完全一致，资格为：

```text
QUERY_OAC_AGGREGATION_AB_INDEPENDENT_PASS
```

两臂 selected rounds 均为 seed `10/8/10`。相对各自完全相同的 round 0，OOF mean gain 为：

| seed | gamma0 | gamma1 | gamma1-gamma0 |
|---:|---:|---:|---:|
| 0 | +0.09109 | +0.08109 | -0.01000 |
| 1 | -0.05395 | -0.06363 | -0.00967 |
| 2 | +0.12952 | +0.13911 | +0.00960 |

合并 360 个 OOF seed-state 后：

| objective | mean | median | P05 | P10 | worst |
|---|---:|---:|---:|---:|---:|
| gamma0 | +0.05555 | +0.03286 | -1.66653 | -1.02846 | -9.19596 |
| gamma1 | +0.05219 | +0.02331 | -1.43489 | -0.57601 | -8.70692 |

gamma1 的确改善了 tail，并且 85/100 km/h mean gain 都是 3/3 seeds 优于 gamma0。尤其 seed 0 的
100 km/h 从 `-0.09764` 变为 `+0.17497`，85 km/h 从 `-0.07295` 变为 `+0.05916`；但它在 seed
0/1 的总体 OOF mean 都更差，pooled mean 比 gamma0 低 `0.00336`，pooled median 也低 `0.00955`。
这说明 raw-cost-aware weighting 的高速/tail 修正是真实的，但约 4% effective-sample 的极端聚合牺牲
了更大主体，microstep 的轻微 mean 优势没有在 continuously refreshed Critic/Actor 循环中保持。

按实验前写入 config 的 gate，gamma1 只通过 85、100 km/h 和 P05 tolerance，未通过“至少 2/3 seeds
总体 OOF mean 更好”与“pooled OOF mean 不差”两项，所以当前正式 decision 为：

```text
GAMMA1_CONTINUOUS_OAC_FAIL_RETAIN_GAMMA0
```

不得事后把 tail 改善解释成 gamma1 已通过，也不直接追加更多 gamma1 轮次。主线保留 gamma0；gamma1
只保留为“高速/tail 与主体梯度冲突”的机制证据。若下一步继续，应优先设计不让 4% 高-cost rows
支配全部参数的平滑/分组聚合（例如预注册 gamma0.5 或按速度组控制贡献），仍保持结构、20:1 和
Query-label 合同不变，并先用 train-side inner 做选择。

需要特别记录一个合同措辞问题：config 的 `split_contract.OOF_usage` 写了“不用 OOF 选择 arm”，但同一
config 的 `decision_gate.primary_population` 又明确用 outer-fold OOF 决定是否 adopt gamma1。实际执行按
后者复算并报告 decision，因此 fold-0 OOF 从本节起已经是开发决策数据，后续不得再称为 untouched
validation；formal validation/test 仍完全 sealed。这不改变配对 A/B 的机制结论，但降低了其外部验证
资格。未来 objective 选择必须只用 fit/inner，冻结后再去新的未消费 fold 或 formal validation 做一次
确认。

第一次执行在六组训练均完成后，仅因最终 JSON 打印不能序列化 `numpy.bool_` 而退出；该非权威副本
保留在 `outputs/query_mppi/query_oac_aggregation_ab_20260902_v1_failed_json_serialization`。修复只涉及
JSON serialization，权威目录为上列 `_v1`，且已通过完整独立重放。

权威哈希：

```text
aggregation A/B config          316fe609f12cfc675856744af406f52fd21bfdd283be547beebbe52f823d4922
aggregation A/B runner          74c14f0f298e1c8c5411418179a38289a1069ec0fb833b817548902b77e0d3c1
aggregation A/B validator       8a11e64108a12fe027528334336f87a63d1b6dafb3e06b7b2fc16b5b434587b8
aggregation A/B manifest.json   e0be6ffcd3edb01d664c453219400cc011bd3164387ad79ccaed6b7115b57cf1
aggregation A/B summary.json    dd7cecb30fb502873b04e3dfe73f705c1f3861b22371f3f435f1b864255bc751
aggregation A/B validation.json f9f4c01860bb83673e221251dea8a423c5ca48be8dc34c5295f82c88d901bfa5
```

### 17.11 依据 DBM 历史修正 gamma/评价口径，并完成 gamma1 K=1/4/8 扫描

用户指出 17.10 的 gamma 选择和评价指标与 DBM 主线经验不一致。复核 review
§11.108--11.109、§11.116 及 OAC plan §39--40 后确认该意见正确，17.10 的
`GAMMA1_CONTINUOUS_OAC_FAIL_RETAIN_GAMMA0` 不再作为主线 routing，原因有两项：

1. DBM 已完成 gamma=`0/0.5/1` 的短跑和长跑。`gamma=1` 是近似 `mean raw J` 的绝对 cost 主目标；
   高尺度 gamma0.5 在胜 warm 比例、median、P05、worst、Actor mean J 和聚合改善上全部比 gamma1
   更差，历史正式结论是固定 gamma1、停止 tempering 扫描，而不是因为短期 tail 波动退回 gamma0；
2. sampling-center 通用评价必须以同一冻结模型下 deterministic warm center 为外部参照，主报告
   `P(J_actor<=J_warm)`、`G_w=J_warm-J_actor` 的 median/P05/worst、
   `sum(G_w)/sum(J_warm)` 和速度/场景分层。相对 round0 Actor 的 gain 只能说明 AC 更新机制，不是
   策略质量或 gamma 资格。

按正确口径复算 17.10 的 selected development-OOF 后，gamma0/gamma1 的 Actor mean J 为
`9.4304/9.4337`，胜 warm 比例均为 `44.72%`，warm-relative median 为 `-0.192/-0.218`，P05
为 `-23.309/-23.365`，worst 为 `-95.878/-95.389`，聚合改善为 `-54.77%/-54.83%`，而 warm
mean J 只有 `6.0931`。两臂都未通过 warm-relative center gate，差异也不足以用该 10-round run
裁决 gamma；该实验只能证明小步 continuously refreshed AC 有 round0-relative 学习信号。正确 routing
改为固定历史主目标 gamma1，下一单变量只检查 Actor 更新预算/尺度。

新的冻结合同、runner、validator 和产物为：

```text
scripts/model_verify/query_oac_gamma1_k_scan_config_20260902_v1.json
scripts/model_verify/run_query_oac_gamma1_k_scan.py
scripts/model_verify/validate_query_oac_gamma1_k_scan.py
outputs/query_mppi/query_oac_gamma1_k_scan_20260902_v1
```

实验从相同 pretrain checkpoints 重启，固定 gamma1、Query Replay、response bank、Twin Critic 每轮
各 20 次更新、Actor 每 microstep LR `2e-6`、trust 和 Query cost，只比较每轮 Critic block 后 Actor
optimizer microsteps `K=1/4/8`。三臂共享完全相同的 context schedule、response basis、Critic RNG stream
和 Actor batch schedule prefix。累计 round output step 在所有 K microsteps 后统一做 cap-only
`0.02 sigma-RMS`，不能让每个 microstep 各自使用完整 cap。共完成 3 arms × 3 seeds × 10 rounds，
生成 70,200 条 online Query candidates；latest 和 selected checkpoint 分开评价。

独立 validator 重构全部 1,800 个 response banks，重放 70,200 个 candidates 和所有
round0/latest/selected inner/development-OOF Actor；所有 action、raw action、Query cost、checkpoint
输出、forbidden-input invariance 与 paired round-1 的最大误差均为 0，split leakage 为 0，资格为：

```text
QUERY_OAC_GAMMA1_K_SCAN_INDEPENDENT_PASS
```

selected inner warm-relative 主结果为：

| arm | selected rounds | Actor mean J | 胜 warm | median | P05 | worst | 聚合改善 |
|---|---|---:|---:|---:|---:|---:|---:|
| K1 | 10/9/10 | 7.8087 | 41.39% | -0.334 | -15.768 | -69.823 | -45.91% |
| K4 | 10/2/10 | 7.5863 | 43.89% | -0.267 | -15.962 | -66.433 | -41.76% |
| K8 | 10/10/10 | **7.4612** | 43.61% | **-0.202** | **-14.883** | -71.096 | **-39.42%** |

K8 的主体/聚合/P05 最好，并在 3/3 seeds 降低 mean cost，但 strict worst 比 K1 多退化 `1.273`，
超过预注册 `1.0` 容忍，因此不按 mean 单独晋级。K4 在 3/3 seeds 降低 mean cost，aggregate 和 median
提高，P05 只比 K1 低 `0.194`、worst 反而提高 `3.390`，通过全部联合门。

已经消费、只作 corroboration 的 development-OOF selected 结果方向相近：K1/K4/K8 的 Actor mean J
为 `9.4475/9.3360/9.2995`，胜 warm 为 `44.72%/45.83%/45.00%`，median 为
`-0.250/-0.214/-0.212`，P05 为 `-23.293/-23.309/-22.990`，聚合改善为
`-55.05%/-53.22%/-52.62%`；但 K8 worst 为 `-111.770`，明显差于 K1/K4 的
`-96.449/-95.674`。这些 OOF 数字没有参与 arm 决策，也不能再称为 untouched validation。

所有 arm 的累计 round step 都自然小于 cap：K1 约 `0.00038--0.00084 sigma-RMS`，K4
`0.00110--0.00322`，K8 `0.00179--0.00636`，projection 全部无需放大。gamma1 microstep weight ESS
median 约 `0.05--0.07`，minimum 约 `0.026--0.030`，表明 batch 64 的更新仍常由少数高-cost rows
主导；K 增加提高了进展，也放大了单状态 worst 的敏感性。

正式 routing 为：

```text
ADVANCE_GAMMA1_K4_TO_LONGER_GAMMA1_QUERY_PILOT
```

该 routing 只表示 K4 是下一次长预算实验的 tail-aware 候选，不表示当前 Actor 合格：三个 K 的
warm-relative median、P05、worst 和 aggregate 均仍为负，单中心不能替换 warm。下一次应保留 K1
cost reference，与 K4 做更长 paired curve；继续只用 inner 做 checkpoint/arm 选择，development-OOF
只作已消费证据，formal validation/test、wrapper、two-center 和闭环保持封存。

权威哈希：

```text
gamma1 K scan config          e15d39efaa730cda2090825e45d255e40262fd9c3c34666f302b5cb0248d33e8
gamma1 K scan runner          85f619338d3f7672b56eb3347a3fbd1efcac46de1dfa8f213eef672076ca316f
gamma1 K scan validator       cca850962a1dbbebaf256a69c9596e759986bee18627ffb5a1c52de9b050bb1a
gamma1 K scan manifest.json   688008c79d1d609750e87474cc2f524457878c2e412a1df74ed295f7ae0f01a9
gamma1 K scan summary.json    c178af650946a0a33f0978060243b659f70af5d3c848142882c7879b16cba980
gamma1 K scan validation.json 0e8ea2128dafbc96fa6c323f44fafd8b3309755d01c8fcf522ca6e4167e9e9fd
```

### 17.12 轮数/输出步长复核与 Query matched-output scale 预注册

用户指出 17.11 的 10 个 outer rounds 明显不足，并要求依据 DBM 已验证问题重新检查实际步长与尚未
覆盖的变量。该判断正确。DBM review §11.139 的 5-round K 结论已被 §11.140 的 90-round 曲线明确
覆盖；K1/K16 在 round 90 仍持续改善，§11.141 的最终 160-round 合同 selected round 分布为
`97--160`、中位 `154`。因此 17.11 的 `K4` routing 只保留为短预算 tail-aware 候选，不能解释为
最终 K 已确定，也不能用 10 轮结果判断 continuous Query OAC 的收敛上限。

当前“步长”有两个不同口径：

- Adam 学习率为每个 Actor microstep `2e-6`；每 seed 的 Actor update 总数分别为 K1/K4/K8 的
  `10/40/80`，每个 Twin Critic 均为 `10*20=200` 次更新；
- 真正可跨 K 比较的是每轮 fit Actor 输出变化。30 个 round/arm 的 sigma-RMS
  `min/median/mean/max` 分别为：K1
  `0.000377/0.000648/0.000655/0.000835`，K4
  `0.001100/0.002118/0.002109/0.003220`，K8
  `0.001790/0.003153/0.003559/0.006357`。90/90 个更新都自然小于 cap-only `0.02 sigma`，
  projection 从未触发。

因此当前 K4 的典型输出步长比自身 `0.02 sigma` cap 小约 `9.4x`，比 DBM 最终固定的 cap-only
`0.06 sigma` 小约 `28x`。两者虽都在 normalized Actor-output sigma-RMS 坐标下，Query 与 DBM 的
cost landscape 不同，不能直接把 DBM 的 `0.06` 当成 Query 已验证常量；但也不能在未校准尺度时把
当前 `2e-6` 小步长盲目延长到 160 轮。

本轮对照后，DBM 已固定且 Query 主线继续沿用的项目为：`gamma=1`；warm 只作 deterministic 外部
评价参照、不进入 Actor 输入/loss/weight；持续 `20 Critic : 少量 Actor` 更新；每轮累计 trust 必须
统一且为 cap-only；selected/latest 分开；通用指标固定为胜 warm 比例、warm-relative
median/P05/worst、聚合改善和速度/场景分层。不得重开 gamma/tempering、静态 Critic gradient-provider、
Pairwise Delta 或共享 tail-loss 扫描。

Query 尚未验证、后续必须按单变量顺序覆盖的项目为：

1. `90/160` 轮学习曲线与 selected/latest 间隙；
2. Query 自身的有效输出尺度，尤其 DBM 已检查过的 `0.005/0.01/0.02 sigma` matched-output 区间，
   以及后续 cap-only `0.02/0.04/0.06` 是否需要；
3. K16；17.11 只在 10 轮比较 K1/K4/K8，不能冻结 K4；
4. Actor LR 与后段衰减；当前固定 `2e-6`，尚未验证 DBM 的 `2e-5 -> 5e-6` 调度在 Query 上是否合适；
5. exploration 在前 90 轮完成 `0.20 -> 0.05 sigma`，而不是当前 10 轮内快速退火；
6. 65-candidate 两阶段 search-recentered bank；当前为 39 candidates；
7. gamma1 的低 ESS/batch 鲁棒性。当前 batch 64 的 ESS median 仅约 `0.05--0.07`，常由约 3--5 行
   主导；
8. 较大 Actor 步长和长轮数下的 fresh Critic ranking/bank recovery、输出饱和/action-bound 与
   episode-grouped 完整 `5 folds x 3 seeds`；
9. two-center guard、wrapper、formal/test 与闭环。这些属于训练机制通过后的下游门，继续封存。

下一步先做低成本、train/inner-only 的 **Query matched-output scale calibration**，不立即启动长跑：

- 起点固定为 17.11 每个 seed 的 K4 selected Actor/Twin Critics；Critic 全冻结；
- 只使用各 seed 的 360 fit rows 计算 `gamma=1` conservative Twin raw-cost-aware Actor 参数梯度；
- 将同一负梯度方向分别精确校准到 fit 输出 `0.005/0.01/0.02 sigma-RMS`；exact forcing 只用于一次性
  诊断，不能冒充后续 cap-only 训练；
- 只在 120 inner-selection rows 上做真实 frozen-Query direct J50 重放，报告相对 K4 起点及相对 warm
  的 mean/median/P05/worst、胜 warm、聚合改善和五档速度；outer development-OOF 不评价；
- 独立 validator 必须重载 checkpoint、重构 gradient/动作/步长并重新 Query replay；DBM 字段/label、
  Query analytic gradient、formal validation/test、wrapper 与闭环继续为零；
- 本实验只回答“当前 Query Actor/Critic 方向在多大输出位移内仍有真实 Query 改善”，不能单独冻结
  长跑 LR/cap。结果通过后，才预注册带 `10/20/40/60/90` checkpoint 的递进长曲线，避免一次性生成
  大量无效采样。

### 17.13 gamma1 matched-output scale 结果：禁止直接放大，保留小步 refreshed OAC

已按 17.12 合同完成 K4 selected checkpoint 的 `0.005/0.01/0.02 sigma-RMS` 一次性尺度诊断：

```text
scripts/model_verify/query_oac_gamma1_matched_step_config_20260902_v1.json
scripts/model_verify/run_query_oac_gamma1_matched_step.py
scripts/model_verify/validate_query_oac_gamma1_matched_step.py
outputs/query_mppi/query_oac_gamma1_matched_step_20260902_v1
```

三 seed 的 source selected round 为 `10/2/10`。每个 seed 固定自己的 K4 selected Actor/Twin Critics，
从 360 fit rows 计算同一 gamma1 full-fit 参数梯度，再将负梯度精确匹配到三个输出步长；每一步只在
120 inner rows 上重新做真实 frozen-Query J50。包含基准重放共新增 1,440 次 Query rollout，没有
训练新模型，也没有评价 outer development-OOF。

相对 K4 起点的 pooled 360-row 结果为：

| exact 输出步长 | mean gain | median | P05 | worst | mean 改善 seed 数 |
|---:|---:|---:|---:|---:|---:|
| `0.005 sigma` | `+0.01353` | `+0.00179` | `-0.90476` | `-3.20829` | 1/3 |
| `0.010 sigma` | `+0.00899` | `+0.00179` | `-1.84159` | `-6.44120` | 1/3 |
| `0.020 sigma` | `-0.04349` | `-0.00200` | `-3.75358` | `-12.99517` | 1/3 |

逐 seed mean gain 显示显著方向分歧：seed 0 在三步为
`-0.00014/-0.00998/-0.05049`，seed 1 为
`-0.01966/-0.06931/-0.24395`，只有 seed 2 为
`+0.06039/+0.10625/+0.16397`。pooled 最优虽是 `0.005 sigma`，但只 1/3 seed 的 mean 为正；
放大到 `0.02` 后 pooled mean 已转负，且相对起点 P05/worst 随尺度明显扩大。warm-relative 指标仍未
通过：三个尺度的 pooled median 为 `-0.273/-0.259/-0.269`，P05 为
`-16.426/-15.223/-18.516`。

独立 validator 重载三组 checkpoint，重新构造 gamma1 weight、Actor gradient、三档 matched Actor
并重放全部 base/step Query cost。动作、步长、cost、指标和 forbidden-input invariance 最大误差均为
0；参数梯度最大绝对误差 `1.12e-8`。资格与裁决为：

```text
QUERY_OAC_GAMMA1_MATCHED_STEP_INDEPENDENT_PASS
QUERY_GAMMA1_MATCHED_STEP_NO_RELIABLE_DIRECTION
```

该结果覆盖“依据 DBM 直接把 Query 单轮输出步长放大到 `0.02/0.04/0.06`”的建议。DBM 证明了自身
landscape 上轮数和 trust 都是瓶颈，但当前 Query K4 endpoint 的 frozen one-step gradient 在
`>=0.005 sigma` 没有跨 seed 稳定性，不能迁移 DBM 大步长。它也不否定 continuously refreshed OAC：
17.8/17.11 每次小 Actor 更新前都有 20 次 Critic 更新和 fresh Actor-visited Query bank，而本节故意
冻结 Critic、只审计一次局部方向。

下一步因此固定为 **小步 refreshed 长曲线**，不同时改变 LR、K、candidate bank 和 trust：从共同
pretrain checkpoint 重新开始，gamma1、39 candidates、每 Twin 20 updates/round、Actor LR `2e-6`、
cap-only `0.02 sigma` 全部保持；只比较 K1 cost reference 与 K4，并将 exploration 的
`0.20 -> 0.05 sigma` 退火扩展到 90 轮。观察节点为 `10/20/40/60/90`，只用 inner 选择和裁决；已消费
development-OOF 只在冻结 selected/latest 后作旁证。该实验检验“持续小步+持续重采样能否累积越过
warm”，不再把单次大步失败误读为 continuous AC 无法继续。

权威哈希：

```text
matched-step config          e5b89e21ed0461615f5a0d5c4f4872a5008e6dc5662f2c5d5bd89d05a1f7e829
matched-step runner          df191510d2776806c7b85d7ed9e41278414225a96c560e823b9edfb2d539712a
matched-step validator       686a17cdfbc163c2f9acf13f722b430d68c22549dfc08d645482a0659ee74247
matched-step manifest.json   7f405d4ed40ac93c7b23e9feaf3b36fbc0c60e162dadfb61c7153168318decda
matched-step summary.json    cc867d2c1b83104f9348612a30666bd4116d1159a37cac40d9853af6b61ab97c
matched-step validation.json 4c1f2554da77ce8406dce4a7fb2d6c0c603b909e24ff6872eee4c657e2cf5368
```

### 17.14 K1/K4 小步 90-round 曲线：K4 加速且总体占优，但仍未超过 warm

已按 17.13 冻结合同从共同 pretrain checkpoint 重启 K1/K4 配对长曲线：

```text
scripts/model_verify/query_oac_gamma1_k1_k4_90round_config_20260902_v1.json
outputs/query_mppi/query_oac_gamma1_k1_k4_90round_20260902_v1
```

两臂均为 gamma1、3 seeds、90 outer rounds、每轮20个 fit contexts、每 context 39 个 fresh Query
candidates、每个 Twin Critic 20 updates/round；只有 Actor microsteps K=`1/4` 不同。exploration radius
在完整90轮内从 `0.20` 线性退火到 `0.05 sigma`，没有沿用旧10轮快速退火。主运行共新增
432,000 次 Query rollout；formal/test、DBM 字段/label 与 Query analytic gradient 均未使用。

曲线首先确认 10 轮不足。三 seed selected rounds 为：K1 `90/90/86`，K4 `90/90/84`。K1 seed1
在 round 10 后短暂回退、到 round 31 才刷新 best；K4 seed1 在 round 2 后回退、到 round 12 才刷新
best。短跑会把这两个 seed 错误解释成平台或反向，持续 Critic refresh 和 Actor-visited sampling 后均
重新进入下降路径。

三 seed 等权的 inner checkpoint 曲线如下；表中 median/P05 是三 seed 各自统计的均值，仅用于曲线
定位，最终资格使用后面的 pooled 360-row 指标：

| arm/round | Actor mean J | warm aggregate | 胜warm | gain median | gain P05 |
|---|---:|---:|---:|---:|---:|
| K1 / 10 | 7.8101 | -45.94% | 41.11% | -0.455 | -17.753 |
| K1 / 20 | 7.7300 | -44.44% | 41.94% | -0.278 | -17.334 |
| K1 / 40 | 7.6026 | -42.06% | 44.17% | -0.222 | -16.993 |
| K1 / 60 | 7.5080 | -40.30% | 44.72% | -0.206 | -17.934 |
| K1 / 90 | 7.4151 | -38.56% | 43.61% | -0.152 | -18.442 |
| K4 / 10 | 7.6027 | -42.06% | 44.44% | -0.229 | -16.042 |
| K4 / 20 | 7.4327 | -38.89% | 45.00% | -0.183 | -16.816 |
| K4 / 40 | 7.2585 | -35.63% | 45.83% | -0.091 | -19.040 |
| K4 / 60 | 7.0699 | -32.11% | 48.61% | -0.054 | -18.076 |
| K4 / 90 | 6.8071 | -27.20% | 48.33% | -0.044 | -15.498 |

selected inner pooled 360-row 正式比较为：

| arm | selected Actor mean J | 胜warm | gain median | P05 | worst | 聚合改善 |
|---|---:|---:|---:|---:|---:|---:|
| K1 | 7.4138 | 43.33% | -0.106 | -16.575 | -63.818 | -38.54% |
| K4 | **6.8036** | **49.44%** | **-0.007** | **-15.033** | **-62.843** | **-27.13%** |

K4 在 3/3 seeds 降低 selected mean，并相对 K1 将 pooled mean J 降低 `0.6103`、aggregate 提高
`11.40pp`、胜warm提高 `6.11pp`；median/P05/worst 也分别提高约 `0.099/1.541/0.974`。因此通过
预注册的整体 K4-vs-K1 联合门。已经消费的 development-OOF 只作旁证，方向一致：K1/K4 mean J
`9.0920/8.2556`，胜warm `45.83%/49.72%`，P05 `-22.985/-17.904`，worst
`-105.710/-94.404`，aggregate `-49.22%/-35.49%`。

但 K4 仍未通过单中心 warm gate。inner pooled median 仅接近0而非转正，P05/worst和aggregate仍明显
为负；按速度的 aggregate 为40/55/70/85/100 km/h的
`-0.18%/-3.71%/-25.86%/-15.48%/-49.80%`。尤其100 km/h的 P05 从K1的 `-31.09` 变为
K4的 `-34.76`，说明总体tail改善不能掩盖最高速tail仍恶化。K4只获得“当前小步长下更有效的
mechanism arm”资格，不获得替换warm或部署资格。

实际输出步长仍远低于cap：K1 270轮的 `min/median/mean/max` 为
`0.000221/0.000374/0.000420/0.000837 sigma`，K4为
`0.000728/0.001165/0.001289/0.003220`；540/540 个round均未触发 `0.02 sigma` cap。gamma1
microstep ESS median仍约 `0.061`，低ESS问题没有因长跑消失。

旧全量validator完成了6个run、546个跨K配对检查和全部candidate/checkpoint Query重放；所有误差为0，
但因代码写死要求K列表 `[1,4,8]`，唯一失败项为 `k_arms_exact`。该原始失败记录保留为：

```text
validation_legacy_k_contract_fail.json
```

随后专用第二层validator只接受预注册 `[K1,K4]`，确认上述项是唯一失败，所有其他full-replay、run、
pair、hash和sealed-boundary检查均通过，最终资格为：

```text
QUERY_OAC_GAMMA1_K1_K4_90ROUND_INDEPENDENT_PASS
```

下一单变量是保持90轮、LR `2e-6`、cap `0.02`、39 candidates、Critic 20/round和退火完全不变，严格
配对 K4/K16。原因不是假设K16必然更好，而是K4到round 90仍有明显下降斜率，且DBM历史只证明K16
可能加快中程收敛、未证明提高最终上限。若K16不能在总体和100 km/h tail上稳定超过K4，则停止扩大K，
转向低ESS/速度冲突或sampling bank问题；不得同时放大LR/trust。

权威哈希：

```text
K1/K4 long config          d9adb243c94fc48678721b4822006766bd84969575811260de60fcb6eb589f65
K1/K4 adapter validator   40da747ac460dc999cb4fd56632bed7b8247f5953a56a4563be7c898713ffeda
K1/K4 manifest.json       c27a5a95b0042bf0a6944bc18ccd766f3f46feb4f6bccc9992223a9009e7cea8
K1/K4 summary.json        f5833026668d2d6d57490b7c0974aac5489a147be396784fa2b288abc685df90
K1/K4 validation.json     391022577ddda2402c0e0af20e5d2cb027d6724279afa8fbd24f511d9427de83
K1/K4 legacy validation   d7a5f207369c2a19cbd61c0464502c4595ba3e11c5a5b169bc9774013282a91c
```

### 17.15 K4/K16 小步 90-round 配对：K16 同时改善总体与高速 tail，进入 160-round 长曲线

已按 17.14 的单变量合同完成严格配对 K4/K16 运行：

```text
scripts/model_verify/query_oac_gamma1_k4_k16_90round_config_20260902_v1.json
outputs/query_mppi/query_oac_gamma1_k4_k16_90round_20260902_v1
```

两臂从同一 pretrain checkpoint 重新开始，均为 gamma1、3 seeds、90 rounds、每轮每个 Twin Critic
20 updates、Actor LR `2e-6`、cap-only `0.02 sigma`、39 candidates，exploration radius 在完整 90 轮内
由 `0.20` 线性退火到 `0.05 sigma`；唯一注册差异是每轮 Actor optimizer microsteps K=`4/16`。
因为共享 Actor batch schedule 以本实验的 maximum K=16 预生成，本节的 K4 只能作为本节内配对对照，
不能把它的逐轮轨迹与 17.14 maximum K=4 的 K4 运行直接作同一随机路径比较。

selected inner pooled 360-row 结果为：

| arm | selected rounds | Actor mean J | 胜warm | gain median | P05 | worst | 聚合改善 |
|---|---|---:|---:|---:|---:|---:|---:|
| K4 | `90/90/88` | 6.7837 | 49.72% | -0.0047 | -14.9557 | -67.0586 | -26.76% |
| K16 | `90/87/88` | **5.4419** | **57.50%** | **+0.2005** | **-11.2950** | **-36.0382** | **-1.69%** |

K16 在 3/3 seeds 都降低 selected mean J；相对 K4，pooled mean J 降低 `1.3418`，胜warm比例提高
`7.78pp`，median/P05/worst 分别改善约 `0.205/3.661/31.020`，聚合改善提高 `25.07pp`。已经消费的
development-OOF 只作旁证，方向仍一致：K4/K16 mean J `8.2880/6.3555`，胜warm
`48.89%/55.83%`，P05 `-17.775/-14.209`，worst `-95.940/-86.505`，聚合改善
`-36.02%/-4.31%`。

17.14 特别要求的 100 km/h tail 也同步改善，而不是用总体均值掩盖高速退化：

| arm / 100 km/h | mean J | 胜warm | median | P05 | worst | 聚合改善 |
|---|---:|---:|---:|---:|---:|---:|
| K4 | 15.3994 | 41.67% | -1.6616 | -36.1487 | -67.0586 | -49.48% |
| K16 | **11.6456** | **45.83%** | **-0.3181** | **-19.7490** | **-36.0382** | **-13.04%** |

因此 K16 通过预注册的总体 K4 比较门和显式高速 tail review，裁决为
`ADVANCE_GAMMA1_K16_TO_LONGER_GAMMA1_QUERY_PILOT`。但它仍没有获得替换 warm 的资格：总体
aggregate 仍为 `-1.69%`，100 km/h aggregate 仍为 `-13.04%`，overall/100 km/h P05 仍明显为负；
development-OOF aggregate 也仍为 `-4.31%`。

K16 的 270 个 round-update 输出步长 `min/median/max` 为
`0.002236/0.004574/0.012225 sigma-RMS`，仍全部自然小于 `0.02` cap，projection 270/270 均未触发；
K4 对应为 `0.000700/0.001147/0.003220`。gamma1 cost-weight ESS median 从 K4 的 `0.2020` 增至
K16 的 `0.2189`，原因是这里统计的是每轮 K 个 microstep 的 mean ESS，不能据此宣布低 ESS 已解决。

旧全量 validator 完成 6 个 run、2,166 个跨 K 配对检查以及全部 candidate、round0/latest/selected、
inner/OOF Query 重放，动作/raw action/cost/forbidden-input 的最大绝对误差全部为0。唯一失败仍是旧代码
写死的 `k_arms_exact=[1,4,8]`，原始失败保存在 `validation_legacy_k_contract_fail.json`；专用第二层
validator 验证预注册 `[K4,K16]`、确认旧失败集合只有该项后通过，最终资格为：

```text
QUERY_OAC_GAMMA1_K4_K16_90ROUND_INDEPENDENT_PASS
```

下一步固定为 K16 的 160-round 单臂长曲线，只增加 round budget。为保证因果可解释性，前 90 轮必须
逐轮复现本节 K16：相同 seed/context/response/batch schedules、相同 Critic/Actor 配置和
`0.20 -> 0.05 sigma` 退火；round 91--160 把 radius 保持在 `0.05 sigma`。不得在同一实验中加入
DBM 的大 LR、LR decay、65-candidate recentering 或更大 trust；它们均须在轮数效应确认后另做单变量
扫描。观察节点扩为 `10/20/40/60/90/120/140/160`，round 0 仍可参与 selected checkpoint，资格仍以
inner 的 warm-relative 指标和分速度 tail 为主，已消费 OOF 只作冻结后的旁证。

权威哈希：

```text
K4/K16 long config          d77e200bd867a185f99fa4186fd5c3cc4e53cce5204b4c8f6f796a30ce8fe94b
K4/K16 adapter validator   10ba387852c37d1b85aabbe20a718c413e4f81109b6028b61fbad3e55aae099e
K4/K16 manifest.json       cc1128c3f9b3fbc54c4ac716f03cef0627b6794363e32125e768d1687dc81c10
K4/K16 summary.json        5b6e6e63bc921a7c0ea9c97d0f432edadf2c540b222e16aa98d990033db7aa18
K4/K16 validation.json     2c219802d152b7e7dc6c503bda9df54ca0401cac03337aa85b19ef7f52b4267e
K4/K16 legacy validation   e42ec284a99cdfcc264e76df4bebdff9e9dfe7f3c39a7f672f15875b678976b0
```

### 17.16 K16 160-round 曲线：性能跨过 pooled warm，但严格跨运行前缀门失败

已按 17.15 的轮数单变量合同完成 K16 160-round 运行与全量独立 Query 回放：

```text
scripts/model_verify/query_oac_gamma1_k16_160round_config_20260902_v1.json
scripts/model_verify/run_query_oac_gamma1_k16_160round.py
scripts/model_verify/validate_query_oac_gamma1_k16_160round.py
outputs/query_mppi/query_oac_gamma1_k16_160round_20260902_v1
```

本实验为单臂 gamma1 K16，3 seeds、160 rounds；Critic 20 updates/twin/round、Actor LR `2e-6`、
cap-only `0.02 sigma`、39 candidates 与此前相同。round 1--90 的 radius 使用与 17.15 完全相同的
`0.20 -> 0.05 sigma` 数列，round 91--160 固定 `0.05 sigma`。共新增 379,800 次 Query rollout；
formal/test、DBM 字段/label 与 Query analytic gradient 继续为零。

曲线直接确认 90 轮仍过少。三 seed selected rounds 为 `146/158/160`，三者都在 round 90 后继续刷新
最低 inner mean J。三 seed 等权曲线如下；median/P05 是逐 seed metric 的均值，仅用于曲线定位：

| round | Actor mean J | warm aggregate | 胜warm | gain median | gain P05 |
|---:|---:|---:|---:|---:|---:|
| 10 | 7.2730 | -35.90% | 45.28% | -0.099 | -17.903 |
| 20 | 6.9595 | -30.05% | 50.00% | -0.025 | -17.864 |
| 40 | 6.4467 | -20.46% | 51.11% | +0.036 | -13.780 |
| 60 | 5.9032 | -10.31% | 53.89% | +0.120 | -11.947 |
| 90 | 5.4649 | -2.12% | 57.50% | +0.195 | -10.861 |
| 120 | 5.1147 | +4.43% | 59.17% | +0.233 | -9.384 |
| 140 | 5.0209 | +6.18% | 60.00% | +0.220 | -10.747 |
| 160/latest | 4.9600 | +7.32% | 59.72% | +0.217 | -10.149 |

最终 selected inner pooled 360-row 指标为 mean J `4.9023`、胜warm `58.61%`、gain median
`+0.2087`、P05 `-10.5295`、worst `-26.7126`、aggregate `+8.40%`。相对 17.15 的独立 90-round
K16 selected，mean J 从 `5.4419` 降低 `0.5396`，aggregate 从 `-1.69%` 转为 `+8.40%`，P05 从
`-11.2950` 改善为 `-10.5295`，worst 从 `-36.0382` 改善为 `-26.7126`。已消费的 development-OOF
只作旁证，其 selected mean J/胜warm/median/P05/worst/aggregate 为
`5.4044/62.22%/+0.3797/-12.8081/-59.6558/+11.30%`。

按 seed 的 selected inner aggregate 为 seed0 `-0.44%`、seed1 `+17.40%`、seed2 `+8.22%`；因此
pooled 转正不是 3/3 seed 都超过 warm，但 3/3 seed 的 160 selected mean 都低于各自在独立 90-round
artifact 的 selected mean。selected/latest 分离仍必要：seed0/1 latest 分别从 selected 的
`5.3749/4.4203` 回退到 `5.4701/4.4983`，seed2 的 selected/latest 均在 round160、J `4.9116`。

literal 预注册 inner warm gate 的三个条件均通过：overall aggregate/median 与 100 km/h aggregate 均为
正。但该门不等价于“所有分布稳定超过 warm”，不能作为替换 warm 或部署资格：

| speed | aggregate | 胜warm | median | P05 | worst |
|---:|---:|---:|---:|---:|---:|
| 40 | +5.68% | 73.61% | +0.305 | -1.362 | -3.638 |
| 55 | **-15.49%** | 61.11% | +0.131 | -4.088 | -15.423 |
| 70 | +5.59% | 55.56% | +0.236 | -8.130 | -16.000 |
| 85 | +28.59% | 59.72% | +0.311 | -8.825 | -26.279 |
| 100 | +5.03% | **43.06%** | **-0.366** | -13.436 | -26.713 |

100 km/h aggregate 转正主要受少数高 warm-cost 行的较大 gain 影响，胜率和median仍为负向，不能只看
aggregate。更明确的系统冲突为 `70 km/h:variant0` 与 `55 km/h:variant2`：pooled aggregate 分别
`-158.79%/-56.27%`；前者三 seed 胜率为 `0%/16.67%/0%`，后者三 seed aggregate 为
`-63.80%/-60.07%/-44.95%`。另外 `100 km/h:variant2` aggregate 为 `-50.88%`。这些失败跨 seed
重复，不能解释为单个 seed 的偶然 outlier。

实际 480 个 round-update 的输出步长 `min/median/mean/max` 为
`0.001944/0.004766/0.004868/0.012225 sigma-RMS`，480/480 个 projection factor 都为1，即从未触发
`0.02` cap；ESS median `0.2195`。长曲线收益仍来自 refreshed Critic + 小步 Actor 累积，不是 cap
截断或一次大步。

需要保留一个方法学失败。v1 在运行前要求新的 160-run 前 90 轮逐数值复现 17.15 的独立 K16 artifact，
tolerance `1e-6`。实际 probe radius、state visitation 与 Actor batch row 前缀最大误差均为0，但随着
GPU 重跑累计，candidate action/raw action 最大差 `0.2323`、candidate cost最大差 `48.7543`、
selection action/cost最大差 `0.01710/6.9073`。运行期间 PyTorch 报出 cuDNN execution-plan
non-determinism/fallback warning；这与跨进程浮点轨迹漂移一致，但当前证据不能把原因唯一归因于
cuDNN。因此不得事后放宽门，也不得把外部 90-vs-160 比较裁决改写成通过。正式实验裁决保留为：

```text
GAMMA1_K16_160_NO_RELIABLE_EXTENSION
```

同时，独立 validator 对当前 160 artifact 自身完成全部 374,400 个 fresh candidates 的重新生成与
Query cost 重放，并重载 round0/latest/selected Actor；candidate action/raw action/cost、checkpoint
action/cost、summary metrics 和 forbidden-input 最大误差全部为0，三个 run 的 split、episode隔离、
候选结构、radius、microsteps、cap与hash均通过。因此“artifact 与其报告一致”的资格为：

```text
QUERY_OAC_GAMMA1_K16_160ROUND_INDEPENDENT_PASS
```

这两个结论不矛盾：前者否决 v1 过强的跨运行 bitwise-like 前缀因果门，后者确认当前长曲线的数值与
边界没有算错。下一步不应直接进入 wrapper/闭环或宣称 Actor 已全面超过 warm。先做一个确定性
reproducibility/within-run round-budget 实验：启用明确的 deterministic CUDA 合同，从同一运行保存
round90 的完整 Actor/Critic/optimizer/RNG 状态，再原位继续到 round160；以该运行自身的 frozen
round90 为基准比较 91--160，并重复至少一次确认 deterministic replay。只有轮数效应在该合同下通过，
才对 `70:0`、`55:2`、`100:2` 做单变量的 stratum-aware sampling/critic-bank 诊断；不得依据当前总体
aggregate 直接调 tail loss、改多头结构或解封 formal/test。

权威哈希：

```text
K16/160 config          cad1012989c1bbaa132830dc8e70dccf5464f950378fbb1683a608edd376763b
K16/160 runner          b3fef5443cd1a0cd9dcfcc3d4579d261c10d10f7207e0bb99b6d2f71441ffe7f
K16/160 validator       d2c2eb1453e6c2b55e24795136c14986907a0f913bdb87cf974b5007ee6fb5b3
K16/160 manifest.json   a24fbc22439f24116b5b6e30c6b1112e950e5a3cbece314df0932afad94cbacf
K16/160 summary.json    ec9ed53018b22c5f4611c8f7ad7879a5d0b3cfcbaf7cddc428fd21fede3c6f83
K16/160 validation.json 05f3b97cb8af5d330085a6bd73f94abc95bbce27246770f6301f4b659a6792db
```

### 17.17 K16 Actor-LR 60-round 配对：高 LR 显著加速，但固定升 LR 未通过困难分层保护门

为回答“当前 `2e-6` 是否过低、是否有必要提高 LR”，完成了三档严格单变量筛选：

```text
scripts/model_verify/query_oac_gamma1_k16_lr_scan_60round_config_20260903_v1.json
scripts/model_verify/run_query_oac_gamma1_k16_lr_scan_60round.py
scripts/model_verify/validate_query_oac_gamma1_k16_lr_scan_60round.py
outputs/query_mppi/query_oac_gamma1_k16_lr_scan_60round_20260903_v1
```

三臂均从同一 pretrain checkpoint 开始，固定 gamma1、K16、3 seeds、60 rounds、每轮20个分层fit
contexts、39 candidates/context、每个 Twin Critic 20 updates、Critic LR `1e-4`、Actor cumulative
cap-only `0.02 sigma` 与相同随机访问/response basis/Actor batch schedule；唯一注册差异是 Actor AdamW
microstep LR=`2e-6/5e-6/1e-5`。radius 使用既有90-round `0.20 -> 0.05` 调度的精确前60轮，即
round60约 `0.10056 sigma`。共新增437,400次 Query rollout；formal/test、DBM字段/label和Query
analytic gradient继续为零。

严格 deterministic CUDA 首次尝试在任何一轮完成、任何LR结果产生之前失败：PyTorch 2.3.0+cu121
报告 `adaptive_avg_pool2d_backward_cuda` 没有 deterministic implementation。该空/部分输出保存在
`outputs/query_mppi/query_oac_gamma1_k16_lr_scan_60round_20260903_v1_strict_determinism_failed`。
随后在观察LR结果前将合同显式修订为 deterministic `warn_only`；cuDNN/attention/adaptive-pool
non-determinism warning均进入运行事实。因此本节是同进程同随机调度的配对统计，不声称跨进程bitwise
可复现。

selected inner pooled 360-row 结果为：

| Actor LR | selected rounds | mean J | 胜warm | gain median | P05 | worst | aggregate |
|---:|---|---:|---:|---:|---:|---:|---:|
| `2e-6` | `59/60/60` | 5.9123 | 53.33% | +0.1015 | -12.4281 | -51.8248 | -10.48% |
| `5e-6` | `55/60/58` | 5.0474 | 60.00% | +0.2750 | -9.4041 | -33.2307 | +5.68% |
| `1e-5` | `56/59/58` | **4.5109** | **64.17%** | **+0.3611** | **-9.1862** | **-22.4074** | **+15.71%** |

固定轮数下提高LR的加速是真实且跨seed一致的：`5e-6`和`1e-5`均在3/3 seed降低selected mean，
并同时通过总体aggregate/median/P05/worst门。三seed等权的 round10/20/40/60 mean J 为：

| LR | r10 | r20 | r40 | r60/latest |
|---:|---:|---:|---:|---:|
| `2e-6` | 7.2736 | 6.9590 | 6.4639 | 5.9156 |
| `5e-6` | 6.8784 | 6.4111 | 5.4580 | 5.0838 |
| `1e-5` | 6.4742 | 5.6894 | 4.8766 | 4.5505 |

已经消费的development-OOF仍只作旁证；三档selected mean J/胜warm/median/P05/worst/aggregate分别为：

```text
2e-6: 7.1650 / 52.78% / +0.1616 / -15.3943 / -76.7005 / -17.59%
5e-6: 5.6993 / 59.17% / +0.3309 / -11.5758 / -88.3699 /  +6.46%
1e-5: 4.7606 / 63.61% / +0.4055 /  -8.0005 / -57.9522 / +21.87%
```

100 km/h整体也随LR提高而改善，不是拒绝升LR的原因：

| LR / 100 km/h | mean J | 胜warm | median | P05 | worst | aggregate |
|---:|---:|---:|---:|---:|---:|---:|
| `2e-6` | 12.9283 | 43.06% | -0.4613 | -26.3394 | -51.8248 | -25.50% |
| `5e-6` | 10.5246 | 44.44% | -0.2048 | -15.7519 | -33.2307 | -2.16% |
| `1e-5` | **8.9115** | **52.78%** | **+0.1963** | **-14.1610** | **-19.0786** | **+13.50%** |

但预注册门要求总体提升不能通过牺牲已知困难分层获得。该门没有通过：

| slice / aggregate | `2e-6` | `5e-6` | `1e-5` |
|---|---:|---:|---:|
| `70:0` | -157.07% | -149.40% | -159.25% |
| `55:2` | **-30.77%** | -40.27% | -56.43% |
| `100:2` | -115.28% | -61.53% | -11.66% |

两档高LR的 `70:0` aggregate/P05大体改善，但worst均超出容许退化；`5e-6`的worst由
`-13.5589`变为`-15.3037`，`1e-5`为`-16.3551`。更关键的是`55:2`随LR单调恶化：P05从
`-11.4651`变为`-12.4308/-14.7554`，`1e-5`胜warm比例也从66.67%降到44.44%。因此两档候选都未
通过完整overall+tail gate，预注册建议为：

```text
KEEP_ACTOR_LR_2E6
```

这个裁决不能简化成“高LR无效”。零rollout的事后matched-overall-cost诊断显示，当高LR尚处于与
`2e-6` selected相近的总体mean J时，困难分层并未系统更差：`5e-6` round29 mean J `5.8921`、
`1e-5` round16 mean J `5.9217`，对比control `5.9123`；此时 `55:2` aggregate分别为
`-30.54%/-29.33%`，control为`-30.77%`，`70:0`与`100:2`也大体改善。说明高LR没有证据走入不同的
错误路径，而是更快推进到当前总体mean目标下会继续牺牲`55:2`的同一策略前沿。该matched-cost分析
是结果后的机制解释，不是预注册资格门。

输出步长进一步区分三个档位：

| LR | step median/mean/max | cap projection rounds | ESS median |
|---:|---|---:|---:|
| `2e-6` | `0.00457/0.00482/0.01223` | 0/180 | 0.2098 |
| `5e-6` | `0.01042/0.01040/0.02000` | 2/180 | 0.2240 |
| `1e-5` | `0.01579/0.01572/0.02000` | 26/180 | 0.2360 |

`5e-6`仍主要是自然更新放大；`1e-5`已有14.4%的round受cap主导，且训练日志出现明显round间摆动，
不建议固定使用。当前结论是：**提高LR对减少训练轮数很有价值，但不是解决`55:2/70:0`差距的必要或
充分手段；不能把固定高LR直接升级为主线。** 若后续优先优化计算效率，唯一合理的下一LR实验是
`5e-6`前段、随后降回`2e-6`的单变量schedule，并保留困难分层no-regression checkpoint gate；若优先
解决最终质量，应先诊断/修复`55:2`的sampling/Critic/共享Actor冲突，而不是继续加LR或trust。

旧全量validator完成9个run、11,532个配对检查和全部candidate/checkpoint Query重放，所有动作、cost、
forbidden-input与round1配对误差均为0；唯一失败是其硬编码K列表`[1,4,8]`，原始报告保留为
`validation_legacy_k_contract_fail.json`。专用验证层确认其余检查、LR checkpoint合同和tail裁决全部
通过，最终资格为：

```text
QUERY_OAC_GAMMA1_K16_LR_SCAN_60ROUND_INDEPENDENT_PASS
```

权威哈希：

```text
LR-scan config              84f0f78596d1f16c18bdfea2dea4632deee25e64abcfbe0ecf217061e94d8efd
LR-scan runner              26cf82281f4db59afbe533c7af488900172c97e3ec2ba1c1b616b4c819d33967
LR-scan validator           b75ea36fb434ca02d083e7eb18622a54703c143321c32a61b8cb041081b9338d
LR-scan manifest.json       21b5cb9d42ea9481f2916e8839e0fde68e71a131638e260425b8d17fa2ac2467
LR-scan summary.json        0978d35d111f335e4c4edd1c6c3aa1145662f430c97e7ec12a6bd0f82252c202
LR-scan validation.json     37acd270083f2d266cdb03d06aeca2684ee9dff970ecbfc635c985e506ded36e
LR-scan legacy validation   9b95eb83400a3950d08bc499a1dd6eae96641da3a271b81078b48288a3ca484a
```

### 17.18 共享 round-60 fork 的 160-round LR schedule：延长高 LR 有效，但 61--120 提前衰减失败

为检验“`1e-5` 高 LR 起步、随后连续衰减并增加轮数”是否优于固定 LR，完成了同一次运行内的完整状态
分叉实验：

```text
scripts/model_verify/query_oac_gamma1_k16_lr_decay_160round_config_20260903_v1.json
scripts/model_verify/run_query_oac_gamma1_k16_lr_decay_160round.py
scripts/model_verify/validate_query_oac_gamma1_k16_lr_decay_160round.py
outputs/query_mppi/query_oac_gamma1_k16_lr_decay_160round_20260903_v1
```

每个 seed 的 round 1--60 以 Actor LR `1e-5` **只运行一次**；预先固定 160 轮的分层 context schedule
和 K16 Actor batch schedule，并在 round60 保存 Actor、Twin Critic、Actor/Critic optimizer、在线 Replay、
Critic NumPy RNG、Torch CPU/CUDA RNG、selected Actor/Critic 与 selection 曲线。三条 continuation 从这一份
内存快照恢复：

```text
fixed_lr1e5:          round 61--160 固定 1e-5
cosine_lr1e5_to_2e6: round 61--120 cosine 1e-5 -> 2e-6，round 121--160 固定 2e-6
switch_lr2e6:        round 61--160 立即固定 2e-6
```

其余合同完全一致：gamma1、K16、20 Critic updates/twin/round、cap-only `0.02 sigma`、每轮20个fit
contexts、39 candidates/context；radius 在 round1--90 由 `0.20 -> 0.05 sigma`，随后固定`0.05`。
formal/test、DBM字段/label和Query analytic gradient继续为零。实际唯一在线response候选为842,400条；
计入selection/OOF checkpoint evaluation后实际执行975,960次Query trajectory evaluation。若把共享前缀
为三臂各重跑一次，逻辑数组对应1,301,400次，因此本设计既消除了伪前缀比较，也节省了重复计算。

严格CUDA仍不可用：PyTorch 2.3的attention/adaptive-pool backward继续报告non-deterministic warning。
round60起始模型、optimizer、Replay与RNG均逐字节共享，但即使fixed与cosine在round61都用`1e-5`，其
round61输出也会因CUDA kernel非确定性出现小漂移（例如seed0 inner mean J `4.8093` vs `4.7759`）。
因此“同一fork起点”成立，“后续逐轮bitwise配对”不成立；结论按3 seeds统计解释。

三臂 selected inner pooled 360-row 结果：

| schedule | selected rounds | mean J | 胜warm | gain median | P05 | worst | aggregate |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed `1e-5` | `156/158/160` | **3.8769** | **68.89%** | **+0.4271** | **-4.4364** | -31.7610 | **+27.56%** |
| cosine `1e-5 -> 2e-6` | `154/160/160` | 4.1972 | 67.22% | +0.3992 | -6.4600 | -29.3732 | +21.57% |
| switch `2e-6` | `134/143/131` | 4.3734 | 63.61% | +0.3324 | -6.6647 | **-27.9053** | +18.28% |

固定高LR不仅比两种降LR方案均值更低，而且相对本次共享round60 selected基线
（rounds `60/48/58`，mean J `4.4773`、胜warm `62.50%`、median `+0.3007`、P05 `-9.2857`、
worst `-20.6926`、aggregate `+16.34%`）在同一次运行内继续显著改善：mean J再降`0.6004`，
aggregate增加`11.22pp`，P05改善`4.85 J`。代价是严格worst恶化`11.07 J`，所以selected checkpoint
仍不能替代warm guard。三个fixed seed直到round `156/158/160`才选中，也直接说明60轮未到瓶颈，且
本次从round61开始衰减过早。

latest与selected仍需分开报告：fixed latest mean/aggregate/P05为
`3.9485/+26.22%/-5.1269`，相对selected只回退mean `0.0716`；cosine latest为
`4.2109/+21.31%/-6.5114`，相对selected回退mean `0.0137`。衰减确实降低振荡，但没有转化为更好的
selected、latest或P05。continuation 300个round-update的step median/mean/max、projection为：

```text
fixed 1e-5: 0.014924 / 0.014954 / 0.020000，27/300触发cap
cosine:     0.007498 / 0.009059 / 0.020000，12/300触发cap
switch2e-6: 0.005174 / 0.005419 / 0.011426， 0/300触发cap
```

已经消费的development-OOF只作旁证，但排序一致：

```text
fixed 1e-5: mean 3.9684 / 胜warm 69.17% / median +0.5155 / P05 -7.5395 / worst -29.4944 / aggregate +34.87%
cosine:     mean 4.3318 / 胜warm 68.61% / median +0.5301 / P05 -7.9002 / worst -45.8134 / aggregate +28.91%
switch2e-6: mean 4.5236 / 胜warm 67.22% / median +0.5050 / P05 -9.0574 / worst -58.7545 / aggregate +25.76%
```

fixed `1e-5` 的speed slice整体也优于cosine：40/55/70/85/100 km/h aggregate分别为
`+12.25/-16.93/+28.84/+50.63/+30.77%`，cosine为
`+12.64/-20.27/+15.10/+46.61/+23.88%`。但已知困难分层仍揭示同一共享Actor目标冲突：

| selected inner slice | shared r60 | fixed `1e-5` | cosine | switch `2e-6` |
|---|---:|---:|---:|---:|
| `70:0` aggregate | -136.49% | **-59.13%** | -123.03% | -142.59% |
| `55:2` aggregate | **-51.82%** | -74.72% | -81.22% | -68.63% |
| `100:2` aggregate | -11.80% | **+34.72%** | +23.23% | +5.77% |

延长固定高LR显著修复`70:0`和`100:2`，同时进一步牺牲`55:2`。所以本轮不支持“降LR可以自动修复
困难分层”；三种continuation都恶化`55:2`，而cosine的`55:2`最差。预注册cosine promotion门只有
selected median和latest-close-to-selected等少数项通过；mean/aggregate/P05/worst与多个hard-slice门
失败，正式裁决为：

```text
DO_NOT_PROMOTE_LR_DECAY_SCHEDULE
```

窄结论是：**增加到160轮并保持`1e-5`有明确价值；本次round61即开始、round120即降到`2e-6`的衰减
过早且不应推广。** 不能由此推出所有衰减都无效，但fixed三seed的selected rounds都接近预算末端，
当前也没有证据支持立即再扫另一条衰减曲线。下一步应保留fixed `1e-5`作为train-side mean/P05领先
参考，并优先处理`55:2`的sampling/critic/shared-Actor冲突；若以后只为降低latest振荡测试LR schedule，
衰减应晚于已观察到的持续改进区间，且必须继续保留inner checkpoint selection与warm guard。

独立validator重放实际唯一的21,600个response group，candidate action/raw action/cost以及
round0/latest/selected checkpoint action/cost最大绝对误差全部为0；9个run的hash、split、episode隔离、
候选结构、160轮LR日程、cap、fork字段和三臂共享前缀逐数组相等检查全部通过。最终资格为：

```text
QUERY_OAC_GAMMA1_K16_LR_DECAY_160ROUND_INDEPENDENT_PASS
```

权威哈希：

```text
LR-decay config          3cad8ef1e67875d42750eaa43a0ecc076bfd7715e6d1b0ec421e56031433a7e2
LR-decay runner          0cd2bdf3671c5ed9591f7ae949002a1443617b4ba369b3a7bd5b375e8759509a
LR-decay validator       b622cce3b2367730a6d3fd40adc70dd6ff4847afa3f7cd3063b49d4c3e0df5c8
LR-decay manifest.json   6f89d4e67156d221ab830fe0793b7c6260130cde318b6cf174a3e40afe45614e
LR-decay summary.json    62a74c9863bfb73116b7f10dea7ba60691ef41d3190aa3eb835d778ea5cb8293
LR-decay validation.json 69a6184010b1c93b056a003693505c1b53069bf1c84c69a05d7af780c97b7771
```

### 17.19 fixed `1e-5` 困难分层诊断：主因是跨 episode/reference 分布外推后的 Actor basin 失配

针对17.18中fixed `1e-5`的`55:2`退化，完成了fit与已消费inner-selection范围内的专项诊断：

```text
scripts/model_verify/query_oac_fixed_lr1e5_hard_slice_diagnostic_config_20260903_v1.json
scripts/model_verify/analyze_query_oac_fixed_lr1e5_hard_slice.py
scripts/model_verify/validate_query_oac_fixed_lr1e5_hard_slice.py
outputs/query_mppi/query_oac_fixed_lr1e5_hard_slice_diagnostic_20260903_v1
```

诊断使用fixed `1e-5`三个selected checkpoint（round `156/158/160`），目标slice为`55:2`，对照为
`70:0/100:2/55:1/55:3/100:3`。在Actor与warm两个中心分别建立`0.025/0.05/0.1 sigma`
的39候选新鲜response bank，并重算fit/inner直接cost、selected Twin Critic局部排序、有限差分梯度、
fit上的分层Actor参数梯度及deployable Actor输入/encoder表征最近邻距离。共新增26,712次Query rollout；
outer fold/formal/test未求值，DBM字段/label与Query analytic gradient均未消费。

首先排除“该slice数据少或没有低cost解”：每个fold的20个speed×variant分层均恰好6个state，fit中
每层18个state；每轮在线访问每层恰好一次，`55:2`截至selected round的访问数/分层中位数在3 seeds
均为1.0。原始fit replay中`55:2`每state有效候选数min/median/max=`132/132/897`；直接取已有候选
最低cost的mean/median/max=`1.8167/1.8234/2.1042`，18/18胜warm，aggregate `+59.26%`。因此问题
不是“搜索不到低cost区域”或简单的样本计数不足。

核心证据是明显的fit到inner episode泛化断层：

| seed | selected Actor fit mean / 胜warm / aggregate | inner mean / 胜warm / aggregate |
|---:|---:|---:|
| 0 | `2.0246 / 83.33% / +54.60%` | `8.6540 / 16.67% / -89.84%` |
| 1 | `1.9488 / 94.44% / +56.30%` | `9.4458 / 33.33% / -107.21%` |
| 2 | `1.8376 / 88.89% / +58.79%` | `5.7943 / 66.67% / -27.11%` |

`55:2` inner六个state全部来自未进入fit的`episode_056/varying_left_recovery`。其normalized reference
到同slice fit最近邻的RMS中位距离为`0.63951`，在20个分层中从大到小排第2；selected Actor encoder
feature按全fit逐维标准化后的最近邻距离中位数为`0.43764/0.45977/0.37165`，三个seed也都排第2。
排名第1的`100:3` reference距离为`0.64040`，它同样表现出跨seed尾部退化，说明这是道路/reference
episode覆盖问题，不是`55:2`独有的标签异常。

`55:2`退化又高度集中在row339/step325：warm cost=`4.8049`，三个Actor cost分别为
`23.5749/24.3471/13.8580`，单点占各seed该slice全部正回退幅度的`73.82%/62.96%/91.04%`。
该点Actor相对warm已移动`0.859/0.796/0.684 sigma RMS`；从错误Actor中心做`0.1 sigma`局部搜索，
最好也只能到`9.177/9.889/5.543`，仍无法回到warm所在低cost basin。与此同时该点Critic在Actor
局部并没有给错方向：三个seed的log-cost Pearson=`0.960/0.992/0.986`、FD梯度cosine=
`0.951/0.982/0.930`、bank gain recovery均为1。故主要失败发生在Actor把新episode映射到错误basin，
而非Critic在该灾难点附近反向指引。

整体Critic结论更细：`55:2` Actor-centered `0.1 sigma`的18个bank中，局部best全部改善Actor，15/18
达到或优于warm；best gain vs Actor中位数`+1.6824`。Critic log-cost Pearson、FD gradient cosine与
bank recovery中位数分别为`0.9134/0.9426/0.9120`，不显著弱于对照层。但P10分别降到
`0.1335/-0.1956/-2.7019`，且warm-centered bank呈明显双峰，说明Critic仍有少量state局部失真；
不过`100:2/100:3/55:3`等对照也存在相当或更差的P10，不能用它解释`55:2`特有的aggregate崩塌。
它应作为次要的tail鲁棒性问题修复，而不是当前主因。

共享Actor目标冲突也未形成跨seed稳定证据：`55:2`分层梯度与其余fit梯度的参数空间cosine为
`-0.1612/+0.0107/+0.1295`，仅1/3为负。因此不支持现在就加多头、冻结Actor或单独重权该slice。

独立validator重新加载checkpoint，重算全部Actor action/cost、encoder feature、Actor/Critic梯度，
并逐一重放648个fresh bank（25,272个候选）；所有最优候选索引保持一致。action最大误差
`5.96e-8`、Critic最大误差`7.15e-7`、梯度最大误差为0。最敏感的一项是`100:3` row577的一个
response proposal：1个float32 action ULP使cost由`40.38445`重放为`40.37069`，绝对/相对误差
`0.01376/0.0341%`；validator显式采用绝对`0.02`且相对`0.05%`双门，候选argmin不变。最终资格：

```text
QUERY_OAC_FIXED_LR1E5_HARD_SLICE_DIAGNOSTIC_INDEPENDENT_PASS
```

当前诊断裁决是：**fixed `1e-5`的总体提升真实，但`55:2`主要瓶颈不是继续加轮数、调LR或扩大同一
state的候选数，而是fit只有3条独立episode时对新reference形状覆盖不足，导致Direct Actor选错低cost
basin；Critic tail是次要问题。** 下一步保持现有单头结构和`20 Critic updates : 少量Actor updates`
框架，先增加独立期望道路/episode，而不是复制当前state。数据生成应按normalized reference/encoder
距离补齐`55:2`与`100:3`这类覆盖缺口；每个新state采用多轮、小批量、依赖上一轮response结果的自适应
搜索，并同时保留Actor中心与一个已有低cost/安全中心作为搜索起点。后者只用于训练期探索和guard，
不作为Actor输入或部署依赖。新replay先严格训练并检查Critic bank-recovery tail，再恢复小步Actor更新；
继续只用inner gate比较mean、P05/worst及困难分层，formal/test保持封闭。不建议在补齐episode覆盖前
再扫LR schedule、增加多头或对`55:2`做强行loss重权。

本轮中间产物`..._pre_representation`、`..._pre_deterministic`和
`..._interrupted_without_cublas`仅保留审计，不是权威结果；上面的canonical目录使用
`CUBLAS_WORKSPACE_CONFIG=:4096:8`及deterministic warn-only合同。

权威哈希：

```text
hard-slice config          0870beaa535f9e399ebaff00d83fa8b91fe3e541a27b14d36f89ca92430adfbb
hard-slice analyzer        a898a5abbd2149acd4229e7cdd9a390c8b2e64a9590beacf22b3a2465c27df8f
hard-slice validator       5d6505e75b67e0c91727f0f72c0d06933b72ec99f872c11aa831722df8760bc9
hard-slice manifest.json   44893b4a874b192b97b43f478efbaf8692cebddd3009bfbe8f97448569e12b89
hard-slice summary.json    aa7811a77acdacc319b4e6564e6930d410b4fbbec4425d32f084a635a8d44929
hard-slice validation.json db148d1328ee74345a891fb9c24480fa0bd6cb6edd278367c78703160eb7e51f
```

### 17.20 基于重新生成期望道路的 target-coverage 扩展：修复 55:2 reference 缺口，100:3 剩余为 history 缺口

按17.19裁决重新生成期望道路并在其上闭环采集，而不是把原始真实轨迹当作reference。第一轮仅写入
fit folds `2/3/4`，针对`55:2`与`100:3`生成12条独立episode、72个成熟snapshot（每个fold 24条）：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/query_expected_road_target_coverage_expansion_20260903_v1
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/query_expected_road_target_coverage_t0_20260903_v1
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/query_expected_road_target_coverage_fullrank_20260903_v1
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/query_target_coverage_absolute_replay_20260903_v1
```

新episode到旧同slice fit的normalized-reference最近邻距离中位数比例降到`55:2=0.103`、
`100:3=0.142`；`55:2`三个fixed-1e5 Actor encoder比例也降到`0.159--0.256`，说明reference覆盖已
实质补齐。history比例仍为`55:2≈0.809`、`100:3≈0.952`，后者几乎没有补齐。T0在67/72个state
严格胜warm，median gain `2.387`；fullrank在72/72个state继续改善T0，median extra gain
`0.1915`，每state至少16个rank候选。合并Replay为672 rows/165,204 valid candidates，旧600 rows
保持逐字节前缀，新增72条只进入fit，inner/outer各120条完全不变；独立资格为
`QUERY_TARGET_COVERAGE_ABSOLUTE_REPLAY_INDEPENDENT_PASS`。

在672-row replay上完成固定split的scratch Actor + Twin Critic预训练：Actor 50 epochs仅作粗初始化，
Twin Critic各240 epochs、LR `2e-4`。scratch Actor因多解BC平均化仍差，不作为AC准入门。Critic在inner
的三个seed结果为：

```text
seed0 Pearson 0.8366 / pair 0.7363 / landscape recovery 0.7966
seed1 Pearson 0.8574 / pair 0.7366 / landscape recovery 0.8302
seed2 Pearson 0.8270 / pair 0.7244 / landscape recovery 0.7707
```

依据17.7--17.8，pooled未分层pair-sign仅作flat/小差值诊断，不重新设为`0.75`一票否决；
`|delta logcost|>=0.05`时总体pair已约`0.803/0.818/0.809`。独立重载的Actor/Critic数值误差均为0，
全局价值与landscape信号3/3 seed通过。round1训练脚本随后为round2增加了兼容逻辑，当前文件哈希与
round1 manifest记录的运行时哈希不同；校验器如实给出
`...PASS_PAIR_DIAGNOSTIC_WITH_TRAINER_SCRIPT_DRIFT`，但checkpoint、config、Replay、summary、split和
全部重算数值哈希/结果均精确匹配。这是provenance warning，不是模型数值失败。

由于round1 Critic在`100:3`原inner状态上的排序仍弱，又做了只针对`100:3`的第二轮自适应覆盖：6条
episode、36个snapshot，使用6套不相交seed history和phase `0.890--0.950`，再次只进入fit：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/query_expected_road_target_coverage_round2_20260903_v1
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/query_target_coverage_round2_absolute_replay_20260903_v1
outputs/query_mppi/query_target_coverage_round2_fixed_split_pretrain_20260903_v1
```

708-row replay仍保持inner/outer各120不变。新状态到round1当前输入的距离降为`0.631`，encoder再降到
`0.771/0.836/0.777`，reference降到`0.88`，但history仍为`0.997`，说明250-step burn-in后改变seed/
phase不能产生新的动态历史。round2 Critic总体仍过全局门（Pearson `0.8101/0.8436/0.8365`，recovery
`0.8537/0.8623/0.8549`），但`100:3`inner pair由round1的`0.596/0.555/0.466`变为
`0.502/0.531/0.473`，没有一致改善。因此round2保留为负对照，停止第三轮同分布采集；AC选择672-row
round1 replay与其Critic。

关键哈希：

```text
round1 replay.npz          e9efcd037abcd2dd0e56a11c8999142ed269a30624106c4c0710e02966e20bf9
round1 replay validation   c11993b5bb93613283bf8042039a9e7547892c3eecc5c58bd225965ce07d24f7
round2 replay.npz          a339f6c0cd140a2f0531bb4a6828d8215993a5edaee4b0880b0426677be1bcf6
round2 replay validation   83ddf77c23f017fcf71de63d60bffe91d7ecc4ee87b2e6868d15d150cbaf16a9
pretrain validator         a322ba1caec0ceea11bdd6dbf0e287864394b9e99527047c59d7a268ed70f8b3
round1 pretrain validation 6d532b754cb0bd76d8fc5983f5991b52826fabc889ee519e50e27f04fa737cf8
round2 pretrain validation 1cbfaecc7de305a34c97efb484eaee6aa9bea6ef7c7ba92796c1e6331d33a2ca
```

### 17.21 旧高质量 Actor + 新 target-coverage Critic 的连续 OAC：55:2 已修复，100:3 剩动态历史 tail

为避免scratch BC Actor的多解平均化，从17.18 fixed `1e-5`三个selected Actor（round
`156/158/160`）启动；Critic使用17.20 round1三个新预训练checkpoint。Actor与Critic分别使用各自训练
时的输入归一化，但动作空间与网络结构完全不变。完成3 seeds × 160 rounds：每轮20个分层fit context、
39个Query response候选、每个Twin Critic更新20次、Actor做16个microstep；Actor LR固定`1e-5`、
gamma1、weight cap 2048、selected-output trust 10、每轮cap-only `0.02 sigma RMS`。radius在round1--90
由`0.20 -> 0.05 sigma`，随后保持`0.05`。round0参加inner真实Query mean选优；outer fold从头到尾未
求值。

```text
scripts/model_verify/query_target_coverage_mixed_init_oac_config_20260903_v1.json
scripts/model_verify/run_query_target_coverage_mixed_init_oac.py
scripts/model_verify/validate_query_target_coverage_mixed_init_oac.py
scripts/model_verify/analyze_query_target_coverage_mixed_init_oac.py
outputs/query_mppi/query_target_coverage_mixed_init_fixed_lr1e5_oac_160round_20260903_v1
```

三个selected rounds为`154/99/138`，说明100轮是必要预算，但100轮后是否继续改善有seed差异；应保留
inner checkpoint selection，不能固定取last。pooled inner结果：

| 指标 | round0 fixed-1e5 Actor | target-coverage OAC selected |
|---|---:|---:|
| mean J | 3.8769 | **3.0888** |
| 胜warm | 68.89% | **76.67%** |
| gain median | +0.4271 | **+0.5873** |
| gain P05 | -4.4364 | **-2.0177** |
| aggregate | +27.56% | **+42.28%** |

目标slice变化更关键：

| split/slice | round0 mean / 胜warm | selected mean / 胜warm | selected median gain / P05 |
|---|---:|---:|---:|
| inner `55:2` | `7.9647 / 38.89%` | **`3.6425 / 83.33%`** | `+1.0575 / -0.3649` |
| inner `100:3` | `15.4099 / 22.22%` | **`7.4178 / 50.00%`** | `-0.3863 / -10.5175` |
| fit `55:2` | `5.1594 / 59.26%` | **`2.4248 / 88.89%`** | `+1.2949 / -0.2324` |
| fit `100:3` | `8.5149 / 75.93%` | **`3.3720 / 91.36%`** | `+9.1705 / -0.2552` |

新增72条coverage rows跨3 seeds pooled warm胜率`90.74%`、gain median `+2.5107`、P05 `-0.2912`，
相对round0 mean再改善`5.8862 J`。这说明新数据与连续OAC确实修复了17.19的`55:2`跨episode
reference basin失配，而不是仅改善训练标签loss。独立validator检查全部源/结果哈希、split、online
shape及outer封存，并重载三个selected Actor重算inner动作和Query cost，最大误差全部为0，资格为：

```text
QUERY_TARGET_COVERAGE_MIXED_INIT_OAC_INDEPENDENT_TRAIN_SIDE_PASS
```

剩余`100:3`失败集中于inner的单个`episode_096`、control steps `250--375`，尤其后半段350/375。
该slice现有candidate bank上的selected Critic log-cost Pearson/pair中位数只有`0.2397/0.5047`，跨seed
明显不稳定；而新增fit `100:3`已经学得很好。因此它不是“100:3整体不可学”，而是250-step burn-in
后未被phase/seed扩展覆盖的动态history外推。继续复制同一种collection没有价值，checkpoint重打分也
无法消除该tail。fit中约`1e3`的最大cost另来自旧`episode_028/40:1`，其warm本身约`982`，与本次新增
coverage无关。

三个候选中seed2是当前更稳健的train-side候选：inner mean `3.1026`仅比最小的seed1高`0.0331`，但
inner worst gain为`-4.8650`，显著优于seed0/1的`-14.2750/-11.8228`；其fit mean、胜warm和P05也在
三者中最好。当前仍不消费outer，不能称为最终部署模型。下一步保持单头结构及`20 Critic : K16 Actor`
不变，若继续补数据，只做一次fit-only的`100:3`动态history多样化：改变初始动态状态/控制扰动或引入
道路/控制器瞬态，而不是只改phase/seed；随后从当前seed2 Actor warm-start并刷新Critic。若没有能力
产生不同history，应停止追加同分布数据，保留seed2为候选并进入预先冻结的后续评估流程。

权威哈希：

```text
mixed OAC config       2f3325c93c3c06596b95b5cc85750269c72f661daf1643565d0f462decba7152
mixed OAC runner       cfe57324cf9328c2eddd676fce0f2d8204b6022f31aaf9e83cb0370257409aed
mixed OAC validator    0d072702fdd98247c943f309965179c11ea99c83667df778eb843b6378108e85
mixed OAC analyzer     8caa0da4d1f8943d0fb1989579045775d2da1b8172fcbc55cf96903ab32a5e62
mixed OAC manifest     84522455bb4d133fa0519778c907c284a984e89bc30baa84b950644f1e82d124
mixed OAC summary      59288de5510edf879aa5bb59abd1eb44bf7ec46f3552d2cc47a6b6113de34e12
mixed OAC validation   e80665e7715795b8789288b1f1136f59034676cd9f5bfac362e27017cc82afae
mixed OAC diagnosis    62ccd8375c5f9f7a03ca587191005dd0eedfe52a200a84d5d4e27a7c918599d9
```

### 17.22 Query direct-cost 跨 context 批处理：保留逐轮选优与历史精确重放路径

针对17.21约40分钟运行中每轮逐条重放120个inner state的开销，完成了不减少验证覆盖、不降低选优频率
的工程加速：

```text
car_foundation/car_foundation/query_deployment.py
car_dynamics/car_dynamics/controllers_torch/mppi.py
scripts/model_verify/query_batched_direct_cost.py
scripts/model_verify/validate_query_batched_direct_cost.py
outputs/query_mppi/query_batched_direct_cost_validation_20260903_v1
```

兼容合同显式区分两种语义：原`QueryDeploymentModel.forward()`和
`TorchMPPIController.evaluate_action_sequences()`继续保持“一条physical context、N个candidate actions”，
代码路径未修改；新增`forward_context_batch()`、backend `evaluate_context_batch()`和controller
`evaluate_context_action_sequences()`只接受“B条独立physical contexts、每条恰好一个对齐action sequence”。
新实验可以将`batched_direct_cost`直接作为同前五个参数的cost helper使用，默认按120 contexts分块；需要
历史bitwise重放的旧validator继续使用原sequential helper。因此没有回改17.21已经产出结果的runner，
也没有让旧artifact的runner hash发生漂移。

CPU fake-model/controller单元测试覆盖：batch与逐context rollout/cost一致、错位shape拒绝、无batch能力的
backend拒绝；当前环境没有安装pytest，所以按相同断言直接载入并执行6个test function，全部通过。
真实Query checkpoint上又重算17.21完整的3 seeds × 161 rounds × 120 inner states：

```text
stored selected rounds: 154 / 99 / 138
batch selected rounds:  154 / 99 / 138
all costs mean absolute difference: 5.8482e-6
all costs max absolute difference:  8.2207e-3
selected mean delta: -2.8044e-6 / -2.9008e-7 / +1.2666e-6
```

最大单点差来自GPU换batch shape后的浮点kernel路径，不能宣称batch120逐值bitwise相同；但三个seed的
所有checkpoint排序与最终选择完全不变，selected mean差均小于`1e-5`。batch size 1通过新接口时与原
sequential cost逐值完全相等；另外再次用原sequential路径重算三个selected checkpoint，最大误差均为0。
故独立资格为：

```text
QUERY_BATCHED_DIRECT_COST_RESULT_PRESERVING_PASS
```

120个inner states的单次实测（RTX 3080 Ti、同一进程warm后）为：sequential `1.8490s`，batch1
`1.8170s`，batch8 `0.2306s`，batch32 `0.0617s`，batch120 `0.0158s`；batch120约`117.1x`。
完整161轮inner曲线每seed批量重算约`2.6--2.8s`，而sequential估计约298s；3 seeds预计可直接减少
约15分钟。该加速只针对one-action-per-context的inner/fit/checkpoint验证。online response bank是
one-context/multi-candidate语义，仍走原路径；若以后批量化online bank，必须另行实现B×C context-action
映射，不能把history简单重复后误称等价。

review按`summary.json 15:35:47`到`validation.json 15:37:42`推断最新validator耗时1分55秒不准确；
这段间隔还包含结果读取、指标诊断及validator文件编写。实际validator命令wall time约10.1秒，符合
3×120 sequential states约5.8秒加模型加载/重算开销。两级validator仍适合更早的全candidate实验，但
17.21的selected-only validator已经是fast validator，不应为此再把覆盖缩到20个state。

当前后续合同：保持每轮完整120-row inner选优，因为selected Actor也作为下一轮trust anchor，降低频率
会改变训练轨迹；优先在所有新runner中显式导入`batched_direct_cost`。若需要稀疏选优，必须作为新的
算法A/B注册，不能仅称为无影响的性能优化。新方向可先做预注册单seed筛选，过门后再扩3 seeds；Query
cost缓存只对重复validator有明显价值，优先级低于batching。

权威哈希：

```text
Query deployment        459466778c3fc8aab48b7730effdbea34469cb6e2da7624e41e5bad1845067b4
Torch MPPI controller   94a902e32c07cc9d427de9ecd93a0431cb9e92bab483ac348d4df7ec29c345ab
batched cost helper     01f8123bd76a0607788f2d489326a9ed25de0a48b9984f22cc9471e92e266e04
batch validator         8592382d99ce757f1ac8efa555c56d5d72c3688373e4156f98b96ab85996712a
batch validation.json   4b949ca81c38e98f52b16d61ec0d1ab225c65336c61abe066025dfb845e9f19b
batch manifest.json     a791057cfdaaca2a5912df60c9002af2d2b25a3d27e26ba8c36108320f3199ed
```

### 17.23 100:3 动态 history 单 seed 筛选：均值小幅改善，但晚段 tail 未修复，不扩 3 seeds

按17.21的唯一剩余数据方向完成了一次fit-only动态history试验。没有再只改seed/道路phase，因为250步
burn-in会把初始seed history完全移出快照；改为在burn-in步骤`[40,240)`对程序化期望道路施加一个平滑
raised-sine横向reference pulse，六条episode的幅值为`-0.75/-0.50/-0.25/+0.25/+0.50/+0.75 m`。
step250起pulse严格归零，因此六个快照`250/275/300/325/350/375`仍使用未修改的`100:3`目标道路；
变化只保留在此前250-token因果history中。冻结Query作为plant、MPPI控制器、cost、动作空间和因果执行链
均未改变，也没有读取记录future trajectory、DBM字段/label或Query analytic gradient。

新增脚本、配置及数据为：

```text
scripts/model_verify/collect_query_expected_road_dynamic_history_coverage.py
scripts/model_verify/query_expected_road_dynamic_history_coverage_config_20260903_v1.json
/disk/collect_data_from_anycar/mppi_rl_closed_loop/query_expected_road_dynamic_history_coverage_20260903_v1
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/query_expected_road_dynamic_history_t0_20260903_v1
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/query_expected_road_dynamic_history_fullrank_20260903_v1
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/query_dynamic_history_absolute_replay_20260903_v1
```

共6条独立episode、36个snapshot，fold `2/3/4`各12行，inner/outer均未增加。闭环独立validator全部
通过：快照最大道路位置误差`0.2106 m`、最大速度bin误差`0.8374 km/h`、每条history中`|z|>3`比例
最大`3.84%`、candidate context最大`2.310 z`；因果history重建、state/action chain、warm recede、
Query rollout重放和所有hash均通过。T0在31/36行严格胜warm，median gain `6.645`；fullrank的132候选
在36/36行继续严格改善T0，额外gain median/mean=`0.3410/0.4228`，fullrank cost median/mean=
`2.5870/2.6098`。合并replay以17.20 round1的672行为不可变前缀，只追加这36行，得到708 rows、
169,956个有效candidate；旧前缀、coverage context/bank及split均被独立逐数组验证，inner/outer仍各120行。

分布筛选显示这不是round2的重复副本：新增history到round1+round2 coverage的最近normalized-history RMS
中位数为`1.2843`，新增组内pairwise RMS中位数为`1.5014`（round2为`1.5290`）。但它没有整体更贴近
inner `episode_096`：沿当前seed2 Actor encoder、以旧fit统计标准化后的最近距离中位数为`0.1639`，
round2反而为`0.1243`；只有step350附近明显降到`0.0581`。因此从采集结束起就把它限定为单seed诊断
增量，而不是直接扩成三seed正式训练。

在708-row replay上只刷新注册seed2的Twin Critic，各240 epochs、LR `2e-4`。相对round1/round2 seed2，
inner Critic总体排序变化为：

```text
                 Pearson median   pair-sign median   landscape recovery
round1 seed2         0.8270            0.7244              0.7707
round2 seed2         0.8365            0.7343              0.8549
dynamic seed2        0.8588            0.7459              0.7659
```

dynamic结果的Pearson/pair是三者最好，但pair比预注册诊断阈值`0.75`低`0.0041`。通用pretrain validator
硬编码要求至少2个seed通过全局门，因此单seed输出按设计显示independent fail；这不是重放失败：checkpoint、
Actor/Critic指标、invariance、split及所有artifact hash重算误差均为0，且唯一seed通过其global Pearson+
landscape signal门。后续单seed OAC runner只在这些精确性检查和`global_value_seed_pass_count=1`成立时放行，
没有把该预训练结果误标成三seed资格。

单seed OAC保持17.21的算法设置不变：从旧fixed-`1e-5` seed2 selected Actor warm-start，20次Critic update:
K16 Actor、Actor LR `1e-5`、gamma1、cap-only `0.02 sigma RMS`，完整跑160轮并保留每轮全部120-row inner
Query checkpoint selection。唯一工程变化是显式使用17.22已证明选优等价的multi-context batch cost；
online one-context/multi-candidate response bank仍走原路径。新增运行与验证为：

```text
scripts/model_verify/query_dynamic_history_seed2_oac_config_20260903_v1.json
scripts/model_verify/run_query_dynamic_history_seed2_oac.py
scripts/model_verify/validate_query_dynamic_history_seed2_oac.py
scripts/model_verify/analyze_query_dynamic_history_seed2_oac.py
outputs/query_mppi/query_dynamic_history_seed2_fixed_lr1e5_oac_160round_20260903_v1
```

selected round为149，独立validator重新加载Actor并批量重算全部120个inner action/cost，action与cost最大误差
均为0，checkpoint argmin仍为149；split、160轮online shape、源文件/hash及outer封存全部通过，资格为：

```text
QUERY_DYNAMIC_HISTORY_SEED2_OAC_INDEPENDENT_TRAIN_SIDE_SCREEN_PASS
```

相对17.21旧seed2，结果是明确的均值改善、但tail退化：

| 指标 | 旧seed2 | dynamic-history seed2 | 变化 |
|---|---:|---:|---:|
| inner mean J | `3.1026` | **`3.0783`** | `-0.0242` |
| inner median J | **`2.2510`** | `2.3394` | `+0.0884` |
| inner胜warm | **`73.33%`** | `71.67%` | `-1.67pp` |
| inner worst gain | **`-4.8650`** | `-6.7263` | `-1.8613` |
| inner 100:3 mean J | `5.9224` | **`5.4593`** | `-0.4631` |
| inner 100:3 median gain | `+0.0393` | **`+0.5749`** | `+0.5355` |
| inner 100:3胜warm | `50.00%` | `50.00%` | `0` |
| inner 100:3 worst gain | **`-4.8650`** | `-6.7263` | `-1.8613` |

逐点看，step275 cost从`13.7716`改善到`9.8277`，step300从`5.4268`改善到`4.3199`；但目标tail的
step350由`7.5886`退化到`9.4499`，step375由`5.0637`退化到`5.4004`。selected Critic在inner
`100:3`已有candidate bank上的Pearson/pair中位数从`0.2859/0.4837`升到`0.3546/0.5034`，仍不足以
稳定处理晚段尾部。新增36个fit row本身学得很好：mean/median=`2.8778/2.3852`，胜warm `94.44%`；
这再次说明“训练分布可学”不等于跨episode tail被修复。

当前裁决为：

```text
DO_NOT_EXPAND_DYNAMIC_HISTORY_SCREEN_TO_THREE_SEEDS
```

即保留这批合法数据与candidate bank作为诊断产物，但不把相同lateral-prelude设计扩到3 seeds，也不继续
追加phase/seed-only或同形态数据。当前更稳健的train-side候选仍是17.21旧seed2，因为它的总体mean只差
`0.0242`，但median、胜warm和worst tail更好。若继续优化，应改变目标而非堆数据：在仍只用fit与已消费
inner的前提下，注册一个明确的tail/risk-sensitive checkpoint或Actor update试验，并以`100:3`最差
warm regression必须改善为晋级门；或者设计与本次不同的真实因果瞬态。未过该门前不启用outer fold。

权威哈希：

```text
dynamic collector                 6bd94aea582a2c89444e155878f65ed8e8877cfdd9f034824212e3941986d6e5
dynamic collection config        80f284f120b62ad98978eb12ee274eedb0df4139c31452342918fea870441986
dynamic collection manifest      15b37bcfa0a5a08a070af852e6e7fc98f57a26855e60c35ddd1d4114fee7d5dc
dynamic collection validation    b90df223d3efa53c94302d325f598808d492e0db31960cbf85cc26d126960ac4
dynamic T0 validation            527582c6ebc83063d629570aa891c6deffd68fd0174b9aa65a9577d68a983d0f
dynamic fullrank validation      f8b45d073b95256e80d031e46e03f2f60475676b4a75c819fb610e21b18ea2e1
dynamic merged replay.npz        cd74af86fbaff2adfc70e7320a4ba352ed49dc713689d3198575317dad1486af
dynamic replay validation        0c28888919e562781c2324cf8f8afb9466d452022c55827499cb72e1c1684ee2
seed2 pretrain validation        8823c2a43cff46f4de3e056c8523ad393a560d40733cb26380ee04c036e59d41
seed2 OAC config                 c5409fadee31d5baa5f2b1374b8313227acd0925dc9b25a1667d5a965574d425
seed2 OAC runner                 4fa8c3f3bafaec0e79832f105d6282beca73162dda11b87fb832e5a24769f891
seed2 OAC validator              f64c5e3e3885892c6bc3195382ad7dacfed3d01f46cc0d5c071976be217a9e9a
seed2 OAC analyzer               ed6185abd01455379d9a4d4b4427a300823c989e0563bcef544b442cd9ac5048
seed2 OAC manifest               865727fd07f227d7771c09445332ef978123d911ec26d3871c404d2de423710f
seed2 OAC summary                d0c2c00062698068c4340697b143858e13b49b505212a5c76cf8b3a60d634888
seed2 OAC validation             e53842f20142c1f21ee9fb0ebe059420ff63a01d6ea7030894929bfca2370c1c
seed2 OAC diagnosis              86d31fa87149875f3d9d90fb8496d2880118a664dbe6404a241c82eff5b87c06
```

### 17.24 风险感知 checkpoint 选择：旧 seed2 round157 可恢复，100:3 tail 明显优于 mean-min round138

17.23说明继续堆同形态动态history不能稳定修复tail后，对17.21旧seed2已经保存的161轮inner曲线做了
零训练成本扫描。先保留原规则的inner mean最小值，再检查其附近是否存在tail更稳健的checkpoint。冻结的
train-side选择规则为：inner mean不高于全曲线最小值`+0.10 J`的checkpoint进入eligible集合，然后最大化
inner `100:3`六个状态中的最差`warm_cost - actor_cost`，精确并列时取最早round。该规则只使用已消费的
inner-selection fold；outer/formal/test仍未求值。

旧seed2共有30个eligible round，mean-min仍是round138的`3.102578`，风险规则自动选出round157：

| 指标 | mean-min round138 | risk-selected round157 |
|---|---:|---:|
| inner mean J | **`3.1026`** | `3.1700` |
| inner median J | `2.2510` | `2.2876` |
| inner胜warm | `73.33%` | **`75.00%`** |
| inner gain median | `+0.5934` | **`+0.6000`** |
| inner P05 gain | **`-1.9763`** | `-2.3588` |
| inner worst gain | `-4.8650` | **`-4.8140`** |
| inner 100:3 mean J | `5.9224` | **`5.3744`** |
| inner 100:3 median gain | `+0.0393` | **`+0.4419`** |
| inner 100:3 worst gain | `-4.8650` | **`-2.5838`** |

round157在总体mean上付出`+0.0674 J`，但总体胜warm更高，`100:3` mean同时下降`0.5480 J`，且目标
worst regression缩小`2.2812 J`；`55:2`六行则100%胜warm，gain median/P05/worst=
`+1.4134/+0.6701/+0.6042`。因此17.21的剩余困难不只来自训练能力，也明显受到“纯mean checkpoint
selection”影响。round157的全inner最差点已转移到`70:0 episode_011 step250`，gain=`-4.8140`；
`100:3`不再是全局最差点。

原17.21 checkpoint只保存round138 selected和round160 latest，没有round157权重。为避免用inner action
做反向蒸馏，新增严格历史恢复：使用相同seed、fit schedule、20:K16更新和原逐state Query cost路径重新跑
160轮，并在冻结规则选出的round157捕获Actor及同轮Twin-Critic。第一次尝试multi-context batch虽能保持
固定checkpoint排名，但到round10已因selected-anchor接受轨迹产生约`0.006 J`偏移，立即中止；第二次原
sequential尝试缺少旧运行记录的`CUBLAS_WORKSPACE_CONFIG=:4096:8`，同样在发现偏移后中止。两次目录均为空，
最终恢复显式设置`:4096:8`并使用历史sequential路径，round1/10/60/140等曲线均精确对齐。

正式恢复产物：

```text
scripts/model_verify/query_seed2_robust_checkpoint_recovery_config_20260903_v1.json
scripts/model_verify/recover_query_seed2_robust_checkpoint.py
scripts/model_verify/validate_query_seed2_robust_checkpoint_recovery.py
outputs/query_mppi/query_seed2_robust_checkpoint_recovery_20260903_v1/robust_checkpoint.pt
```

最终round157恢复相对历史保存数组的action最大误差、Query cost最大误差、mean cost误差均为0。独立validator
再次从源曲线和规则推导round157，核对source run的既有独立资格与全部hash，加载恢复Actor和Twin-Critic，
并用原sequential Query路径重算全部120个inner state；action/cost仍为逐值0误差。资格为：

```text
QUERY_SEED2_ROBUST_CHECKPOINT_RECOVERY_INDEPENDENT_TRAIN_SIDE_PASS
```

当前train-side候选应分成两种而不是覆盖旧结果：round138仍是纯mean最优；round157是已独立恢复的风险感知
候选，推荐优先用于下一阶段，因为它在几乎相同的总体水平下显著降低`100:3` tail。选择规则现在已冻结，
但它是在查看inner曲线后形成的探索性规则，不能反复改阈值；若要启用outer，应同时、一次性比较预先明确的
round138 mean候选与round157 risk候选，并如实报告这是两个train-side finalist，而不是继续根据outer调参。

权威哈希：

```text
robust recovery config      b258886b576b0ba30206e57c5221e9eabab055bd2a866742c12476482d3d1cf8
robust recovery runner      4f04edc48c9f47e40b3bc5ef76385568761b1d03418e966358721dd2ebd4fd9f
robust recovery validator   42751860fd2c1cad80194ec0dc3589d8f59687b8d4d7db9c3b09eac2279fc10d
robust checkpoint.pt        1ef53d877167f20265e123a54e76cb7ed0f01abf7beb76b1eb6875ea6634bb19
robust manifest.json        a01ffc2de3bda0c8a466ccf4a56120c2cf6b8280031ff60d26d79e550abced49
robust summary.json         1fcbeacc552a13465919877b4c60aa9c37f4543f4fc8dbaa9a05c8914299610e
robust validation.json      0e298323eb5d67afe45795026b0d32660ac1885af3245035f4943743d57fa26b
```

### 17.25 主体优先评价合同修正与 Actor-centered Query headroom audit 预注册

用户再次明确本阶段目标不是消除长尾，而是衡量和继续提高**策略网络单次直接输出相对同状态 warm 的
总体 cost 收益**。因此从本节起，后续 train-side 路由统一使用与 DBM 主线相同的主体口径：

1. 主指标为 `mean(J_actor)`、`mean(J_warm-J_actor)` 与
   `sum(J_warm-J_actor)/sum(J_warm)`；同一固定评价集合上，checkpoint 仍按最低 Actor mean cost 选择；
2. 胜/平 warm 比例和 warm-relative median 为辅助主体指标；P05、worst、困难 slice 与逐状态回归继续
   保存，但只作诊断，不再否决 mean/aggregate 更好的候选；
3. warm 仍只是冻结的外部 comparator，不进入 Actor 输入、loss、weight、teacher 或搜索方向；
4. 该修正不授权部署。若未来进入真实 wrapper/闭环，是否需要 warm guard 另行决定，不能混入当前
   direct-center 训练质量判断；
5. outer fold、formal validation/test、DBM 字段/label 和 Query analytic gradient 继续封存。

按这一口径，17.24 的 round157 只能保留为 risk-shadow，不再推荐覆盖 mean-min round138：round138/157
的 inner mean J 为 `3.1026/3.1700`，聚合 warm 改善为 `42.02%/40.77%`。17.23 dynamic-history seed2
的 mean J `3.0783`、聚合改善 `42.48%`，相对旧 seed2 的 `3.1026/42.02%` 是小幅主体改善；此前
`DO_NOT_EXPAND_DYNAMIC_HISTORY_SCREEN_TO_THREE_SEEDS` 中由 tail 触发的否决不再有效，但其主体增量仅
`0.0242 J`且仍是单 seed，因此暂不直接升级为主线。

当前三 seed 主线 selected Actor 在固定 inner 120 states 上的 pooled direct 指标为：warm mean J
`5.3516`、Actor mean J `3.0888`、mean gain `+2.2628`、聚合改善 `+42.28%`、胜/平 warm
`76.67%`。这些是当前必须保留的主体基线。

在继续扫 LR、trust、candidate 数或追加数据前，先复用已独立验证的同状态多轮
forward-response 方法，做一次**当前 Actor-centered Query headroom audit**：

- source 固定为 17.21 三个 selected Actor 与相同 inner-selection fold 的120个状态；每个 seed 独立
  从自己的 Actor center 开始，禁止加入 warm/T0/teacher/旧 canonical 作为搜索起点；
- Actor center 首次真实 Query 重放并在全部后续轮次保留；做4轮自适应搜索，每轮使用16维满秩基的
  32个 antithetic probes，加6个由 empirical scalar-cost / weighted-trajectory response 生成并由真实
  Query 评分的 proposals；后一轮 center 必须是前一轮 incumbent/probe/proposal 的真实 Query argmin；
- 沿用已验证的 basis 顺序 Hadamard/DCT/QR260902/QR260903、半径
  `1.0/0.5/0.25/0.1 sigma`、ridge/damping=`0.1/0.1`、line factors=`0.5/1.0`；每个
  seed-state 共 `1+4*(32+6)=153` 次 Query rollout，三 seed ×120 states 共55,080次；
- 逐轮报告 Actor→best 的 absolute mean gain、相对 Actor 的 aggregate residual reduction、paired
  median reduction、改善状态比例、动作 sigma-RMS movement，以及速度/variant 分层；tail字段只记录；
- 主体 routing：最终 aggregate residual reduction `<2%` 时停止通用 Actor 超参/数据扩张；`2%--5%`
  只允许低成本单变量 A/B；`>=5%` 且多数状态可改善时，说明仍有通用 headroom，下一步依次比较
  39→65 search-recentered candidates、continuous cap-only `0.02/0.04/0.06`，再测试 Actor LR
  `2e-5`。不得把多个变量合并成一次无法归因的实验；
- 必须独立重载 source Actor、复算 Actor action/cost、重放所有保存的 probe/proposal Query cost、重构
  winner chain 与汇总指标。结果落盘前不得修改上述阈值。

### 17.26 Actor-centered Query headroom：主体空间显著，进入单变量训练 A/B

17.25预注册的审计已完成：

```text
scripts/model_verify/query_actor_centered_headroom_config_20260903_v1.json
scripts/model_verify/run_query_actor_centered_headroom.py
scripts/model_verify/validate_query_actor_centered_headroom.py
outputs/query_mppi/query_actor_centered_headroom_20260903_v1
```

三 seed ×120 inner states 均从17.21各自 selected Actor 的唯一 direct center 开始；没有加入 warm、T0、
teacher或旧 canonical 起点。四轮分别使用 Hadamard/DCT/QR260902/QR260903 的32个 antithetic probes
和6个真实 Query 复评的 response proposals，严格执行后一轮依赖前一轮真实 argmin 的搜索链。实际完成
55,080次 Query rollout；没有训练 Actor/Critic，没有读取 outer/formal/test、DBM字段/label或任何
Query analytic gradient。

pooled 360个 seed-state 的逐轮主体结果为：

| round / 累计每state预算 | best mean J | Actor→best mean gain | aggregate residual reduction | paired median reduction | 改善比例 |
|---:|---:|---:|---:|---:|---:|
| 0 / 1 | 3.0888 | 0 | 0 | 0 | 0 |
| 1 / 39 | 2.1567 | +0.9321 | 30.18% | 10.02% | 81.39% |
| 2 / 77 | 1.9610 | +1.1277 | 36.51% | 15.32% | 97.22% |
| 3 / 115 | 1.8919 | +1.1968 | 38.75% | 17.46% | 98.89% |
| 4 / 153 | **1.8574** | **+1.2314** | **39.87%** | **18.71%** | **100%** |

最终每seed aggregate reduction为`39.98/39.42/40.19%`，不是单seed偶然。最终搜索动作相对 Actor
的 sigma-RMS movement均值为`0.316`。速度分层 aggregate reduction 为：40/55/70/85/100 km/h
分别`10.89/16.04/37.71/48.23/62.83%`，五档均为正且每档改善比例均100%。这里速度差异只用于定位
headroom，不能把高速度收益替代总体主体指标。

1440个 seed-state-round 的 winner来源为 incumbent/probe/response=`266/168/1006`；response proposal
进一步优于 incumbent+probe best 的比例为`69.79%`，说明收益不是单纯来自扩大随机候选数量，多轮
trajectory-response机制仍在当前 Actor 邻域有效。

独立 validator 重载三个 source Actor，复算 Actor action/cost，逐条重放全部 probe/proposal Query
cost和weighted residual，重构四轮basis、clipping、response fit、proposal、winner chain与全部汇总。
Actor/action/cost/residual/winner的最大误差均为0；仅JSON中float32 fit-error字段相对float64复算有
约`2.98e-8`舍入差，全部检查通过：

```text
QUERY_ACTOR_CENTERED_HEADROOM_INDEPENDENT_PASS
```

按17.25事先冻结的主体阈值，`39.87% >= 5%`且100%状态可改善，裁决为：

```text
QUERY_ACTOR_HAS_MATERIAL_HEADROOM_ADVANCE_SINGLE_VARIABLE_AB
```

这说明当前 Actor 虽相对 warm 已有`42.28%`聚合改善，但仍没有把同状态多轮搜索发现的低cost区域充分
吸收到单次策略输出中；现在不应停止于tail处理，也不应把问题解释为Query landscape已饱和。下一步按
预注册顺序先比较39与65 candidate 的 search-recentered online bank，保持 Actor/Critic结构、gamma1、
20:K16、LR、cap、context schedule和评价集合不变。先做单seed配对筛选；主体 mean/aggregate有稳定正
增量才扩3 seeds。随后才单独比较continuous cap-only `0.02/0.04/0.06`，最后检查Actor LR `2e-5`。

权威哈希：

```text
headroom config          7e2a0c2cc060e9b31a2237a125ac9b0db239cc7b6f01b3c021dc4c99e17298b1
headroom runner          359204953c2c28797d3490c53e3ce7bdd4554eb23ebd6b72c3ea543b204082b8
headroom validator       7ce5ade5f511073ad6c110c50e99b6dcb6376a0bca76c49724c208a75f982dac
headroom manifest        22c301331573a77d30d735bb67e98999359106aa8404ed5e1bbcce97f8b68a7e
headroom summary         f27f1aef4b5e5b8b962a1edcd68397f78123d057aa427a4c0f1f2e288ee86afb
headroom validation      277163deaa6f28633345a7c60da9d38dad20092fed9ab4eec29af2f858ddae3c
```

### 17.27 39/65-candidate 持续 AC 单 seed 配对预注册

根据17.26的显著主体headroom，下一步不是继续离线search，也不是把search endpoint做BC，而是恢复完整
continuous AC。为先控制计算成本，固定17.21当前inner mean最低的seed1作为单seed机制筛选；两臂均从
同一个seed1 selected Actor及同轮Twin Critics开始，重新初始化相同optimizer，并使用相同fit/inner split、
40-round分层context schedule、Actor batch schedule、随机seed和前90轮`0.20→0.05 sigma`退火的前40轮。

两臂公共合同：gamma1、每轮每Twin Critic更新20次、K16 Actor microsteps、Actor LR=`1e-5`、Critic
LR=`1e-4`、cap-only `0.02 sigma RMS`、每轮20个fit contexts、所有好坏online candidates进入累计
Replay、每轮用完整120-row inner真实Query mean J更新selected checkpoint。Critic不冻结，Actor不冻结；
warm不进入输入/loss/weight；outer/formal/test继续封存。

唯一实验变量是每次Actor visit写入Replay的online bank：

- `response39`：现行 `1 Actor + 32 full-rank antithetic probes + 6 empirical response proposals`；
- `response39_recenter26`：严格保留前39项逐数组相同，再取前39项真实Query argmin为incumbent，以当前
  radius的`0.70x`和固定QR seed260904的前13个方向生成26个antithetic second-stage candidates，得到
  `39+26=65`。新增好坏candidate全部进入Replay，不把best action当监督target。

选择与路由只看主体指标：比较两臂selected inner Actor mean J、相对共同round0的mean gain以及相对warm
的aggregate improvement；P05/worst/困难slice只记录、不否决。如果65臂selected mean严格更低且aggregate
更高，则进入3-seed确认；否则保留39。40轮只用于机制筛选，不得据此宣称已达到最终收敛上限。独立验证
必须确认两臂共同round0、每组65臂的前39项与重构response39一致、second-stage recenter chain、全部online
Query cost、持续20:K16更新形状、selected Actor重载和inner Query cost；不得用outer选择结果。

### 17.28 39/65-candidate 持续 AC 单 seed 结果与三 seed 扩展预注册

17.27的配对筛选已完成：

```text
scripts/model_verify/query_continuous_ac_candidate_bank_ab_config_20260903_v1.json
scripts/model_verify/run_query_continuous_ac_candidate_bank_ab.py
scripts/model_verify/validate_query_continuous_ac_candidate_bank_ab.py
outputs/query_mppi/query_continuous_ac_candidate_bank_ab_20260903_v1
```

两臂共同round0 inner mean J均为`3.069512`。40轮持续AC后，39臂selected为round39，mean J
`3.035950`、相对warm aggregate improvement `43.2699%`；65臂selected/latest均为round40，mean J
`3.016632`、aggregate improvement `43.6309%`。因此65臂相对39臂进一步降低`0.019318 J`
（相对39臂Actor cost约`0.636%`），warm-relative aggregate提高`0.3610`个百分点。候选bank本身的
Actor→bank-best mean gain为39臂`0.679705`、65臂`0.682947`，正改善visit比例分别`96.375%/97.125%`。

tail继续只作诊断：selected Actor的warm win rate为`80.83%/78.33%`，P05 gain为
`-2.3024/-1.6211`，worst gain为`-11.8750/-14.7830`（39/65）；这些字段不覆盖主体mean/aggregate
的正向结果。按17.27冻结门槛，裁决为：

```text
PROMOTE_RECENTER65_TO_THREE_SEED_CONTINUOUS_AC
```

独立validator已重放全部`83,200`个online candidates，重构两种bank及second-stage recenter chain；
online action/cost/raw/clipped/role、source Actor与Twin-Critic adapter、selected Actor重载与inner cost的
最大误差均为0，两臂actor/context/round schedule逐值一致，资格为：

```text
QUERY_CONTINUOUS_AC_CANDIDATE_BANK_AB_INDEPENDENT_PASS
```

第一次启动在任何训练round前发现旧runner期待`normalization`而当前source checkpoint保存
`actor_normalization`，因此立即中止并保留到
`query_continuous_ac_candidate_bank_ab_20260903_v1_failed_source_schema_adapter`；正式run使用只改checkpoint
字段名、不改tensor的显式Actor adapter。失败目录没有训练结果，不进入比较。

下一步扩展规则在查看seed0/2结果前冻结：复用已独立通过的seed1结果，只新增seed0和seed2的相同40轮
配对实验；所有训练/候选/选择合同保持17.27不变。三seed确认通过需同时满足：65臂pooled selected mean J
低于39臂、pooled warm-relative aggregate更高、且至少2/3 seeds的65臂selected mean更低。P05/worst/
困难slice仍只诊断。若通过，保留65-bank进入下一项单变量continuous cap-only `0.02/0.04/0.06`；若
不通过则保留39-bank。不得因seed0/2结果修改该门槛。

权威哈希：

```text
single-seed config        c64d269ad73bf4b6a0848cee0c97c28b7fac38535461319fe25f9ad1d0a83c7e
single-seed runner        d3a57b7f067a7c6f17d3444fbd1b0d84289d858f1fe59bf8314260e437fbca8a
single-seed validator     6c85a78f78db903c4dbb5f5a4684a2b987619d9f41ab2050a586ee5b0fe6209c
single-seed manifest      90b40a6abfdf5a53ad4cf1e99ad02c20b8b4f4fe1505253e0900b2bb97a06249
single-seed summary       351cfdd67ac460b44948ba8c1f9568dc0aa72eab49fb846a6ca95d92da354045
single-seed validation    d8f7aef6b6ed359019c19659368b2338b2355b787ccb49ee57560c4256409162
```

### 17.29 65-candidate 三 seed 确认：小幅通过，进入continuous cap单变量扫描

按17.28冻结合同，复用已独立验证的seed1，只新增seed0/2的39/65配对持续AC：

```text
scripts/model_verify/query_continuous_ac_candidate_bank_ab_3seed_config_20260903_v1.json
scripts/model_verify/run_query_continuous_ac_candidate_bank_ab_3seed.py
scripts/model_verify/validate_query_continuous_ac_candidate_bank_ab_3seed.py
outputs/query_mppi/query_continuous_ac_candidate_bank_ab_3seed_20260903_v1
```

selected inner Actor mean J逐seed结果如下：

| seed | 39-bank mean / round | 65-bank mean / round | `39-65` mean reduction |
|---:|---:|---:|---:|
| 0 | 2.987673 / 38 | 3.003967 / 29 | -0.016294 |
| 1 | 3.035950 / 39 | 3.016632 / 40 | +0.019318 |
| 2 | 3.102580 / 0 | 3.095979 / 2 | +0.006600 |

三seed pooled mean J为`3.042068→3.038860`，降低`0.003208 J`（约baseline Actor cost的`0.105%`）；
相对warm aggregate improvement为`43.1556%→43.2156%`，增加`0.0600`个百分点。65-bank在2/3 seeds
降低selected mean，因此三个预注册条件均通过，裁决为：

```text
PROMOTE_RECENTER65_TO_CONTINUOUS_CAP_SCAN
```

应明确这是方向一致但量级很小的改进，不能解释为65候选已经解决17.26发现的显著headroom。诊断项中，
warm win/tie为`77.50%→77.22%`，P05 gain为`-1.8890→-1.4864`，worst gain为
`-11.8750→-14.7830`；仍不参与主体裁决。

独立validator复用seed1既有83,200-candidate独立资格，并全量重放新增seed0/2的`166,400`个online
candidates。新增四个arm的action/cost/raw/clipped/role、adapter、selected Actor/inner cost，以及三seed
共同round0、paired actor/context/round schedule和pooled指标均为0误差；资格为：

```text
QUERY_CONTINUOUS_AC_CANDIDATE_BANK_AB_THREE_SEED_INDEPENDENT_PASS
```

第一次扩展启动同样在训练前因旧helper不允许第二个seed复用Critic adapter目录而退出；保留到
`query_continuous_ac_candidate_bank_ab_3seed_20260903_v1_failed_adapter_root_collision`。正式run为每个新增
seed使用独立adapter root，tensor内容未变。

下一项单变量实验在看结果前固定：以三seed的65-bank、40-round、cap=`0.02`为control，新增完全相同的
cap=`0.04`与`0.06`两臂；Actor/Critic初始checkpoint、optimizer reset、candidate bank、20:K16、LR、
gamma、fit/inner划分、随机schedule和每轮完整inner选择均不变。先在三seed直接比较，避免再把单seed小增量
误认为稳定效果。候选cap必须同时满足pooled selected mean低于0.02、warm-relative aggregate更高且至少
2/3 seeds mean更低，才可替代0.02；若0.04与0.06都满足，按最低pooled mean选一个（first minimum解
完全相等）。tail仍只诊断。独立验证后才决定是否进入Actor LR=`2e-5`。

权威哈希：

```text
three-seed config        255019105e6ca3f61ef66adf59f84da9d4f2884c58371b9ba6499c59cf0f8a8d
three-seed runner        cbb642934814f64fe372a44a0ef4930b992932e921361b30dd324540c65f7f46
three-seed validator     83167bedae9951798894e8b687f83dbd5660b6dbbc0885cb81b30a31f5056013
three-seed manifest      145b61b6425a21d65baabb4679710e3e8904055e9da60473c5a4f64600a355f5
three-seed summary       abb9fdcbbb2f00a3c3ede2d3fe05f2a7d4c1c85fb5bdf8d96af3aa12657f3643
three-seed validation    8e2983ed883d940c4d4930eb3356aab78efedc8c643564ea864516e61e2dd739
```

### 17.30 Continuous Actor step-cap三seed扫描：保留`0.02`

17.29预注册的65-bank cap-only扫描已完成：

```text
scripts/model_verify/query_continuous_ac_cap_scan_3seed_config_20260903_v1.json
scripts/model_verify/run_query_continuous_ac_cap_scan_3seed.py
scripts/model_verify/validate_query_continuous_ac_cap_scan_3seed.py
outputs/query_mppi/query_continuous_ac_cap_scan_3seed_20260903_v1
```

cap=`0.02`直接复用17.29已独立验证的三seed 65-bank结果；`0.04/0.06`均从同一原始Actor/Twin-Critic
checkpoint重新开始40轮持续AC，只改变每轮Actor累计sigma-RMS step cap。主体结果为：

| cap | selected pooled mean J | warm aggregate | 胜/平warm | gain median | P05（诊断） | worst（诊断） |
|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | **3.038860** | **43.2156%** | 77.22% | +0.6169 | -1.4864 | -14.7830 |
| 0.04 | 3.049880 | 43.0097% | 77.22% | +0.6117 | -1.5516 | -16.9872 |
| 0.06 | 3.045270 | 43.0958% | 78.61% | +0.6132 | -1.6552 | -14.7523 |

相对0.02，0.04的pooled mean回退`0.011021 J`、aggregate回退`0.2059`个百分点，仅1/3 seeds改善；
0.06的pooled mean回退`0.006410 J`、aggregate回退`0.1198`个百分点，虽有2/3 seeds数值改善，但seed1
从`3.016632`回退至`3.052041`，覆盖了seed0的收益和seed2几乎可忽略的`0.000093 J`收益。因此两档
均未同时满足冻结的pooled mean/aggregate门，裁决为：

```text
RETAIN_CAP002_FOR_ACTOR_LR2E5_AB
```

独立validator全量重放新增`312,000`个online candidates；六个新增seed-arm的action/cost/raw/clipped/
role、source adapter与selected Actor重载均为0误差。三档cap的round0、Actor batch、visited context与round
schedule逐值相同，pooled指标和裁决独立复算为0误差，资格为：

```text
QUERY_CONTINUOUS_AC_CAP_SCAN_THREE_SEED_INDEPENDENT_PASS
```

下一项按17.25顺序只检查Actor LR：固定65-bank、cap=`0.02`、40 rounds和其余全部合同，以已验证
LR=`1e-5`三seed结果为control，新跑LR=`2e-5`三seed。替代条件继续固定为candidate pooled selected
mean更低、warm-relative aggregate更高、且至少2/3 seeds selected mean更低；tail只诊断。这里检验的是
当前更强65-bank下的短程持续AC学习速度/质量，不能覆盖17.18对旧39-bank 160-round fixed `1e-5`优于
过早衰减的结论，也不能把40-round结果解释为最终收敛上限。

权威哈希：

```text
cap-scan config        d62dd4a94957ca7c0af2cd7b81bce93d461cacabf888ae0111a7c4ccba660397
cap-scan runner        3cc7ea82c3e6da4a04ec9c7eb306c2004fce6fd66baa275440f25237aa25de5c
cap-scan validator     78b9adffa4f83b30cd2d7b2a88f087607718533faa7072c1e86ef8dde2976de2
cap-scan manifest      d649da7c08f76ebabcac0d21cd57b777e59f8636364a563491dd8bd28c039c40
cap-scan summary       269b283e207a7fa505185d9a33ace203179d0e5e8fd907b29e4dcd5a9b9c9f27
cap-scan validation    03d36e2689a3202e1489514266e13738517296a7ec04a49d3b83f963a7232154
```

### 17.31 65-bank / cap0.02下Actor LR=`2e-5`：三seed主体一致改善

17.30预注册的Actor LR单变量A/B已完成：

```text
scripts/model_verify/query_continuous_ac_actor_lr2e5_ab_3seed_config_20260903_v1.json
scripts/model_verify/run_query_continuous_ac_actor_lr2e5_ab_3seed.py
scripts/model_verify/validate_query_continuous_ac_actor_lr2e5_ab_3seed.py
outputs/query_mppi/query_continuous_ac_actor_lr2e5_ab_3seed_20260903_v1
```

LR=`1e-5`直接复用已独立验证的三seed 65-bank/cap0.02结果；LR=`2e-5`从相同原始Actor/Twin-Critic
checkpoint重新开始40轮持续AC，candidate、Replay、20:K16、Critic LR、gamma、cap、随机schedule与每轮
完整inner checkpoint选择均不变。selected mean逐seed为：

| seed | LR1e-5 mean / round | LR2e-5 mean / round | `1e-5 - 2e-5` reduction |
|---:|---:|---:|---:|
| 0 | 3.003967 / 29 | 2.925813 / 40 | +0.078154 |
| 1 | 3.016632 / 40 | 3.005385 / 35 | +0.011248 |
| 2 | 3.095979 / 2 | 3.030515 / 23 | +0.065464 |

三seed pooled mean J由`3.038860`降至`2.987238`，降低`0.051622 J`（约control Actor cost的
`1.70%`）；相对warm aggregate improvement由`43.2156%`升至`44.1802%`，提高`0.9646`个百分点。
3/3 seeds均改善，三个冻结门槛全部通过，裁决为：

```text
PROMOTE_ACTOR_LR2E5
```

诊断项不参与裁决：win/tie warm保持`77.22%`；gain median从`+0.6169`变为`+0.6045`，P05从
`-1.4864`变为`-1.7172`，worst从`-14.7830`改善至`-13.7675`。LR2e-5 pooled raw per-round
sigma-RMS step median/mean/max为`0.01626/0.01663/0.02574`，120个seed-round中21次触发cap0.02；
LR1e-5对应`0.01225/0.01266/0.02206`且仅2次触发。说明2e-5的收益伴随更多cap约束，但没有通过
增大cap获得同样收益：17.30已经证明0.04/0.06更差，因此应保留2e-5+cap0.02这个组合。

独立validator全量重放LR2e-5新增`156,000`个online candidates；action/cost/raw/clipped/role、source
adapter、selected Actor/inner cost、共同round0、paired schedule、pooled指标和裁决均为0误差，资格为：

```text
QUERY_CONTINUOUS_AC_ACTOR_LR2E5_AB_THREE_SEED_INDEPENDENT_PASS
```

这仍是40-round机制筛选，不是最终收敛声明。下一步不再继续组合扫candidate/cap/LR，而应在冻结的
65-bank、LR2e-5、cap0.02下做更长的continuous AC预算确认。由于当前checkpoint只保存selected/latest
Actor与selected Twin-Critic，没有保存完整optimizer、累计Replay和RNG状态，不能把现有round40伪装成
严格无缝resume；正式长程run应从相同原始source重新跑完整预算，并保存可恢复训练状态。优先采用160
rounds与每10轮（末段可更密）inner评估的训练侧筛选，最终selected再做一次完整120-row独立重放；
outer/formal/test继续封存。若要对“2e-5是否在长程仍优于1e-5”作因果结论，则需相同65-bank下的配对
长程control，不能直接拿旧39-bank 160-round结果代替。

权威哈希：

```text
LR2e-5 config        0beebfaf2eb774260ea2bc2ba9a0d3481678c9d72a089d6757f2f6e3fd70dd62
LR2e-5 runner        01a23289d9dd1215fe5012dc8fb234ef7af85fca03c49e2642f0982d02b6cb3e
LR2e-5 validator     f63be7b78603a0d0732092a4eb0158c5b9eec3a226506a9985a40a4e4855bf75
LR2e-5 manifest      48141fbba0e013fee697b17a3c9be4e86a0512d7851ded28c4150c9a365e08c3
LR2e-5 summary       a4bf0b6fb5f858aaa55901b23a494b3cd69a5660e9658fdba58ab2841cff11e0
LR2e-5 validation    feb1d0c3bc8435ff330d3201ef37ce42e77442c4fb486ec48058b0e318599cd1
```

### 17.32 65-bank / LR2e-5 / cap0.02 的160-round长程持续AC预注册

17.31确认的下一步是扩大**持续AC round预算**，不再同时改变candidate、cap、LR、gamma或网络结构。
正式合同固定为三seed、160 rounds、65 search-recentered candidates、Actor LR=`2e-5`、cap-only
`0.02 sigma RMS`、K16、每轮每个Twin Critic 20次更新、Critic LR=`1e-4`、gamma1、每轮20个fit
contexts，以及原90轮`0.20→0.05 sigma`候选半径退火；仍从17.21相同原始Actor/Twin-Critic checkpoint
重新开始，不能把17.31缺少optimizer/累计Replay/RNG的round40 checkpoint伪装成无缝resume。

这里修正17.31末段关于“每10轮（末段更密）inner评估”的建议：现有算法每轮把最低inner checkpoint的
Actor作为下一轮`selected_actor` trust anchor，因此降低inner评估频率会改变后续Actor更新，并非只减少
验证开销。为保持算法和前40轮严格可比，正式160-round run继续每轮在完整120-row inner上运行真实Query
并更新selected anchor；加速仅使用已验证的multi-context batched direct cost，不改变评价覆盖。

训练侧门槛在看长程结果前固定：

1. 新run的round0--40 actor batch schedule、visited contexts、全部65-bank online candidates、inner actions/
   costs必须与17.31 LR2e-5三seed逐值一致；否则不能把差异归因于追加round预算；
2. 每seed从round0--160中按最低完整inner mean J选checkpoint，first minimum解完全相等；主指标仍为
   pooled selected mean J与warm-relative aggregate，tail只诊断；
3. 若长程相对各自已验证40-round selected在至少2/3 seeds严格降低mean，且pooled mean更低、aggregate
   更高，则判定追加预算有效；否则保留40-round配置，不再盲目加round；
4. runner在每个seed完成后立即保存独立record，使进程故障时至少可按seed恢复；当前基础训练函数仍未
   保存单seed内部optimizer/累计Replay/RNG，故不得声称支持round内精确resume；
5. outer/formal/test、DBM字段/label和Query analytic gradient继续封存，warm仍只作外部comparator。

独立验证必须先确认17.31源资格与哈希、三seed前40轮精确前缀，再重放round41--160新增的全部online
candidates，重载selected Actor复算完整inner cost与pooled裁决。任何前缀误差均直接失败。

执行修正（2026-09-04，发生在正式结果产生前）：第一次长程启动在seed0 round10/20即观察到inner轨迹
与17.31旧40-round run明显不同，因此在seed0 round30后人工中止，未生成任何seed record/summary/manifest，
目录保留为`query_continuous_ac_longrun_160_20260904_v1_failed_nondeterministic_prefix`。检查确认旧A/B wrapper
没有执行基础runner `main()`中的`cudnn.deterministic=True`和
`torch.use_deterministic_algorithms(..., warn_only=True)`，所以旧artifact的Query重放虽逐值确定，跨进程
训练轨迹却没有exact保证。原“动作/cost/candidate前40轮逐值相等”门槛不可执行，不能通过放宽数值阈值
伪装成通过。

在查看任何40轮后的正式结果前，长程合同改为显式启用deterministic warn-only runtime，并在**同一次
160-round轨迹内部**比较round0--40最低checkpoint与round0--160最低checkpoint；这对“新增120轮预算是否
有用”形成严格嵌套比较。与旧17.31 artifact只要求source、配置、前40轮Actor batch schedule、visited
contexts、online round编号和round0 action/cost精确一致；旧40-round cost轨迹只作provenance诊断，不进入
长程晋级门。独立validator相应重放正式run的全部candidate，并从正式run自身重构round40 baseline。

### 17.33 160-round持续AC结果：长程预算在3/3 seeds继续改善

17.32修正后的正式长程实验与独立验证已完成：

```text
scripts/model_verify/query_continuous_ac_longrun_160_config_20260904_v1.json
scripts/model_verify/run_query_continuous_ac_longrun_160.py
scripts/model_verify/validate_query_continuous_ac_longrun_160.py
outputs/query_mppi/query_continuous_ac_longrun_160_20260904_v1
```

本次使用65-bank、Actor LR=`2e-5`、cap=`0.02`、20:K16、gamma1连续运行160 rounds，并在同一
训练轨迹内比较前40轮最低checkpoint与全160轮最低checkpoint：

| seed | 同轨迹round0--40 mean / round | round0--160 mean / round | 长程reduction |
|---:|---:|---:|---:|
| 0 | 2.908672 / 29 | 2.839481 / 139 | +0.069190 |
| 1 | 3.000405 / 36 | 2.910272 / 131 | +0.090133 |
| 2 | 3.074870 / 29 | 2.953761 / 139 | +0.121109 |

pooled selected mean J由`2.994649`降至`2.901171`，降低`0.093477 J`（相对前40 Actor cost约
`3.12%`）；warm-relative aggregate由`44.0417%`升至`45.7884%`，增加`1.7467`个百分点。3/3 seeds
均严格改善，全部冻结门槛通过，裁决为：

```text
PROMOTE_160ROUND_CONTINUOUS_AC
```

最终主体指标：warm mean J=`5.351573`、Actor mean J=`2.901171`、mean gain=`+2.450402`、aggregate
improvement=`45.7884%`、胜/平warm=`79.72%`、gain median=`+0.70655`。诊断P05=`-1.46651`、
worst=`-10.52162`。40/55/70/85/100 km/h aggregate分别为`20.11/25.34/36.75/55.87/56.80%`，
所有速度层均为正。latest mean J为`2.910829/3.046055/3.089791`，明显不等于selected结果，进一步说明
不能强制使用最后一轮。

相对17.31旧的非严格跨进程40-round artifact，pooled mean也从`2.987238`降至`2.901171`；该数值只作
辅助参考，正式因果口径仍是同轨迹嵌套比较。正式runtime记录为deterministic warn-only；PyTorch仍明确
警告memory-efficient attention和adaptive-pool backward没有严格deterministic实现，因此只能保证本run
内部比较和保存产物的Query复放，不能重新宣称任意跨进程训练轨迹bitwise一致。

独立validator重载三seed source adapter和selected Actor，复算完整120-row inner cost，并重放全部
`624,000`个online candidates。action/cost/raw/clipped/role、selected round、round40/full pooled指标和
裁决最大误差均为0；与17.31旧artifact的Actor batch schedule、visited contexts、online round与round0
action/cost也逐值一致。资格为：

```text
QUERY_CONTINUOUS_AC_LONGRUN_160_THREE_SEED_INDEPENDENT_PASS
```

runner支持在进程重启时按hash复用已完成seed，但仍不支持单seed内部optimizer/累计Replay/RNG的精确
round级resume；该限制已写入summary，不能夸大。outer/formal/test、DBM字段/label和Query analytic
gradient均未消费。

下一步不建议直接把round预算继续加到160以上：三个selected虽都较晚，但当前预注册上限已经完成，且
训练曲线存在明显振荡。应先以这三个round139/131/139 selected Actor为唯一center，重复17.26相同的
4-round、153 Query/state actor-centered headroom audit。该审计回答160-round AC究竟吸收了多少可搜索
headroom；warm/T0/teacher仍不得作为起点。若aggregate residual reduction仍`>=5%`且多数状态改善，
再针对Actor吸收机制设计单变量实验；若降至`2%--5%`只允许低成本A/B，低于`2%`则停止广泛扩张。

权威哈希：

```text
longrun config        5d325fc2e59d8695f72a977ef848bde67dc9e9905b9877f84a20b427238001b1
longrun runner        6dd7ebb5567ddb8b7e0167559daa96cc0a4a93b5f5fa7397ca565bf2cb5d6c75
longrun validator     a1d063491f654f13e9713ef173ce4ca5fadc3b6a9fbb2bea63c4bf4bea7e66c2
longrun manifest      29ed37e4729a0cef7dd61641bf4436f56fad1850d3f909082221d1e8f9aa5ef8
longrun summary       ded8667ae1decd6cb5f4ff5e07c6bc7cb62e320ef235a06734f0021f9fd477c2
longrun validation    10564dbf7052cfdd17cc03be15289ca8d9debdb8458e45b8313acf92992d23ad
```

### 17.34 当前160-round selected Actor的数值headroom审计预注册

在继续增加AC轮数、改Actor结构或引入搜索teacher前，先冻结17.33独立通过的三个selected Actor
（seed0/1/2分别为round `139/131/139`），只测量其单中心在当前Query目标下仍可被局部数值搜索降低多少。
该实验复用17.26已经独立验证的actor-centered协议，不重新选择搜索超参：每个Actor/状态从Actor自身动作
开始，依次使用四套固定满秩正交基，在`1.0/0.5/0.25/0.1 sigma`半径各评估32个antithetic probes，
再从实际clipped displacement拟合forward response并评估6个proposal；每轮在incumbent、probe和proposal
间用真实Query cost确定性取最小，完全相等时取第一个。Actor初始点计1次、每轮新增38次，因此预算固定为
`3 seeds * 120 states * (1 + 4 * 38) = 55,080`次Query rollout。

本审计严格不使用warm、T0、旧teacher或其他搜索解作为起点；它们也不进入拟合、损失、权重或筛选。
outer/formal/test继续封存，不使用DBM字段/label或任何Query/DBM解析梯度，不训练Actor/Critic。源Actor必须
保持17.33的checkpoint/arrays哈希以及
`QUERY_CONTINUOUS_AC_LONGRUN_160_THREE_SEED_INDEPENDENT_PASS`资格。旧17.26脚本不修改，以保留其已登记
哈希；本次使用薄封装指向长程run的`longrun/seed_*`布局，并由新validator逐项重放全部55,080个Query
结果、response fit和winner chain。

在查看结果前固定主指标与路由：以当前Actor自身cost为分母，报告pooled mean actor/best-found cost、mean
gain、aggregate residual reduction、paired median relative reduction、改善状态比例，以及逐速度/场景诊断。
这里的best-found只是固定预算下的局部数值下界，不是理论全局最优；也不能单独代表网络结构上限。

1. aggregate `<2%`：停止广泛搜索和Actor超参/数据扩张；
2. aggregate在`2%--5%`：只允许一个低成本、单变量Actor吸收A/B；
3. aggregate `>=5%`且超过半数状态改善：说明可搜索区域尚未被Actor充分吸收，优先诊断/改进Actor吸收机制，
   不直接把更多搜索样本当作teacher回归目标；
4. tail指标只记录，不否决更好的mean/aggregate结论；任何“Actor结构上限”结论需另做严格的搜索标签
   train-only overfit与episode-grouped cross-fit。

执行合同补充（正式headroom结果产生前）：首次seed0搜索已完成计算但在写出任何seed结果前，被旧审计
硬编码的source-cost `1e-6`门槛中止。原因是17.33长训保存的multi-context batched Query selected cost与
本审计逐state Query路径的最大差为`3.890991e-4 J`，而selected Actor action逐位一致。该差异只暴露了
两种Query执行粒度的浮点路径差，不能直接忽略，也不能修改旧审计脚本而破坏其既有哈希。正式封装因此
增加可审计的source execution adapter：原checkpoint、原arrays、selection rows和selected action必须保持
哈希/逐值一致；只用审计自身的逐state Query路径重算comparator cost，同时保存原cost、重算cost和差值，
并预先固定最大允许差为`1e-3 J`。新validator除全量重放搜索外还需复核原源与adapter双重哈希、action零
误差和cost差上限。首次运行未生成summary/manifest/headroom arrays，正式结果从三seed头重新计算。

adapter首次预门进一步在seed1发现单个状态的绝对差`0.0102901 J`，超过原`1e-3 J`上限，所以仍在
headroom搜索前中止。三seed全量只读诊断表明差异高度稀疏：各seed绝对差P95均为0；最大绝对差依次为
`3.89099e-4/1.02901e-2/9.53674e-7 J`，以`max(|saved cost|, 1 J)`缩放的最大相对差为
`1.51360e-4/1.57815e-3/1.07948e-7`，seed1最大差点为`6.52038→6.51009 J`。由于action仍逐位一致，
这继续符合batch execution path差异，但绝对门不适合跨cost量级。正式结果产生前将adapter门固定为同时
满足绝对差`<=0.02 J`与scaled relative差`<=0.002`，并保留三seed原始逐点差分。该门只允许构造同执行
路径comparator，绝不放宽后续所有saved search cost/residual的独立复放门（后者仍为`1e-6`）。第二次
中止同样未执行headroom搜索、未生成summary/manifest/headroom arrays；失败adapter已删除后从头运行。

### 17.35 当前160-round Actor headroom结果：仍有36.15%主体余量

17.34正式实验与独立验证已完成：

```text
scripts/model_verify/query_actor_centered_headroom_longrun160_config_20260904_v1.json
scripts/model_verify/run_query_actor_centered_headroom_longrun160.py
scripts/model_verify/validate_query_actor_centered_headroom_longrun160.py
outputs/query_mppi/query_actor_centered_headroom_longrun160_20260904_v1
```

pooled 360个seed-state从各自selected Actor开始的逐轮结果为：

| round / 累计每state预算 | best mean J | mean gain | aggregate reduction | paired median reduction | 改善比例 |
|---:|---:|---:|---:|---:|---:|
| 0 / 1 | 2.901143 | 0 | 0 | 0 | 0 |
| 1 / 39 | 2.114558 | +0.786585 | 27.1129% | 5.9968% | 76.11% |
| 2 / 77 | 1.946095 | +0.955048 | 32.9197% | 11.6682% | 94.44% |
| 3 / 115 | 1.884844 | +1.016299 | 35.0310% | 14.1222% | 97.78% |
| 4 / 153 | **1.852384** | **+1.048759** | **36.1499%** | **15.2860%** | **99.44%** |

逐seed aggregate为`34.8332/36.5535/37.0179%`，改善比例为`99.17/99.17/100%`；不是单seed
偶然。最终gain median=`0.32196 J`、P05=`+0.05605 J`、min=`0`，搜索链保留incumbent，所以没有
搜索回退。最终动作相对Actor的sigma-RMS movement mean/median为`0.2967/0.2069`。逐速度aggregate为
40/55/70/85/100 km/h=`8.05/10.04/38.20/43.34/59.40%`，五档均为正，且每档改善比例至少98.61%。

第一轮39-query广域response已取得最终mean gain的约75.0%；后续三轮仍从27.11%继续提高到36.15%。
1440个seed-state-round winner来源为incumbent/probe/response=`291/136/1013`，response proposal占
70.35%。这说明两个机制都存在：大半径首轮是最大增益来源，而依赖前轮winner的后续缩半径response仍
贡献约9.04个百分点，不能把结果简化为“一次多抽随机样本”。

零新增rollout的跨seed诊断进一步限制了解释。三个pair的初始Actor action sigma-RMS distance median为
`0.157/0.174/0.173`，搜索终点反而为`0.180/0.266/0.215`；但终点cost绝对差median仅
`0.0263/0.0346/0.0247 J`，Actor→best correction cosine median为`0.718/0.666/0.718`。即多数修正方向
大体一致，但低cost动作终点并不唯一。该结果支持用完整搜索candidate/response训练Critic并持续AC，
不支持把每个状态一个任意best endpoint重新当作MSE teacher；后者仍有多解平均化风险。

独立validator逐项复放全部55,080次Query、weighted residual、四套basis、clipping、response fit、
proposal和winner chain；Actor/action/cost/residual/winner误差为0，fit-error仅有约`2.98e-8`的float32
存储舍入。source execution adapter的三seed action误差为0，原batched/per-state cost差统计也独立复算。
资格与裁决为：

```text
QUERY_ACTOR_CENTERED_HEADROOM_LONGRUN160_INDEPENDENT_PASS
QUERY_ACTOR_HAS_MATERIAL_HEADROOM_DIAGNOSE_ABSORPTION
```

这里的`1.852384`与17.26旧Actor搜索到的`1.8574`相近，只能称当前固定预算下的数值best-found；它不是
理论最优。重要结论是Actor mean虽由`3.0888`改善到`2.9011`，搜索下界几乎没变，因此现有65-bank、
cap0.02、LR2e-5和160轮并未吸收大部分可达headroom。按17.34预注册门槛，禁止直接再堆round或做endpoint
BC；先做单变量探索半径机制A/B。

权威哈希：

```text
headroom160 config       e61abd5a2fac722adb9e560a45b8a2b61dc3505e69aa3e82849e51fb9102a8ab
headroom160 runner       026e539cde29a436738c6cf1afe6534b9a2e9ac72f9d0f46108311ebec91297b
headroom160 validator    84753ae2b4c5a87b1130545a9f91ba127354850e967c870b932ccba8913887ac
headroom160 manifest     87e21112b60136068fa420f5e8f5f8046e9f73bd042dd29fbcc5accdfc61ed62
headroom160 summary      6ad37af00ce13d8d4fc40f494faa4e02e276127da2b329bbdb09ffbdd967f230
headroom160 validation   d57cc1684d85533827e0e6cb9bbc2ad358fba56dcbb647fe4c6b9985d93f449d
```

### 17.36 当前Actor的65-bank探索半径单seed A/B预注册

17.35显示第一轮`1.0 sigma` response贡献了最终约75%的mean gain，而现行continuous AC的65-bank半径
只从`0.20`退火到`0.05 sigma`；此前39→65、cap和LR实验均没有隔离验证半径尺度。因此下一步固定当前
mean最低的seed0 selected Actor/Twin Critics（round139）为共同source，重新初始化两臂相同optimizer与
空的新增online Replay，做40轮单seed机制筛选。共同合同保持65 candidates、20 fit contexts/round、
20 Critic updates/Twin、K16、Actor LR2e-5、Critic LR1e-4、cap0.02、gamma1、相同context/Actor batch/
random schedule及每轮完整120-row inner Query checkpoint选择；所有好坏candidate进入Replay。

唯一变量是65-bank的第一阶段response半径schedule：control沿用`0.20→0.05`的90轮线性schedule前40轮；
treatment使用`1.00→0.10`的90轮线性schedule前40轮。两臂仍是同一39 response加26个`0.7x` recenter
probe，不增加一次访问的candidate数量，也不引入warm/T0/teacher或endpoint监督。选择门固定为treatment
selected inner mean更低且warm-relative aggregate更高；tail只诊断。若通过，才扩三seed或再隔离多轮
缩半径bank；若不通过，不能用headroom audit证明大半径Replay必然可被AC吸收。

独立验证须检查源长程资格/哈希、Actor/Critic adapter逐值一致、两臂round0与随机schedule一致，并重放
两臂共`2 * 40 * 20 * 65 = 104,000`个online Query candidates及selected Actor完整inner cost。outer/
formal/test、DBM字段/label和Query解析梯度继续封存。

执行记录：首轮独立validator已完成两臂全部104,000个candidate复放，但在最终写JSON时因内部报告仍包含
`numpy.ndarray` schedule字段而序列化失败，未生成`validation.json`且没有修改训练产物。修复仅从输出
报告中移除这些已单独比较的内部数组；candidate重放、阈值和裁决逻辑均不改变，随后从头重跑validator。

### 17.37 65-bank探索半径A/B结果：整体放宽失败，需coarse-to-fine而非单尺度

17.36预注册的单seed配对实验和修正后的独立validator已经完成：

```text
scripts/model_verify/query_continuous_ac_radius_ab_config_20260904_v1.json
scripts/model_verify/run_query_continuous_ac_radius_ab.py
scripts/model_verify/validate_query_continuous_ac_radius_ab.py
outputs/query_mppi/query_continuous_ac_radius_ab_20260904_v1
```

两臂共同round0 mean J=`2.839481`、warm aggregate=`46.9412%`。40轮结果为：

| radius arm | selected round | selected mean J | warm aggregate | latest mean J |
|---|---:|---:|---:|---:|
| control `0.20→0.05` | 40 | **2.793708** | **47.7965%** | 2.793708 |
| broad `1.00→0.10` | 0 | 2.839481 | 46.9412% | 2.959619 |

broad相对control回退`0.045774 J`，aggregate少`0.8553`个百分点，两个预注册主体门均失败，裁决为：

```text
RETAIN_CURRENT_RADIUS_SCHEDULE
```

这不是“大半径找不到低cost”的证据。相反，broad 800个访问组的Actor→bank-best mean gain为
`0.63971 J`，高于control的`0.43581 J`，且recent-bank pair accuracy日志通常约`0.97--0.99`，明显高于
control约`0.84--0.88`。问题是单尺度65点过宽后覆盖不均：broad bank-best median gain只有`0.09791 J`，
低于control的`0.17368 J`，可改善访问比例也由`96.375%`降到`79.875%`。因此它找到少数更大的收益，
却漏掉更多常见近邻收益；在相同20:K16/cap0.02下，Critic容易排序也没有自动转成Actor主体改善。

独立validator重载selected Actor与两臂checkpoint，重放两臂共104,000个online candidates；action/cost/
raw/clipped/role、selected Actor action/cost和指标误差均为0。source Actor/Critic adapter、两臂round0、
Actor batch、visited rows和online round也全部逐值一致。资格为：

```text
QUERY_CONTINUOUS_AC_RADIUS_AB_INDEPENDENT_PASS
```

这项负结果收窄了17.35的解释：大headroom真实存在，但不能把所有online点简单改成广域单尺度。下一步
若继续，应严格按用户此前提出的“每次依赖前一次结果”做固定预算coarse-to-fine bank：保留Actor，先用
广域满秩response得到真实Query incumbent，再在该incumbent附近用小半径满秩response；所有好坏点进入
Replay，不把best endpoint作监督label。为了把两次满秩response都保留，最小自然预算是
`1 + 38 + 38 = 77` candidates/visit，仅比65增加18.5%，而不是一次堆大量独立随机点。应先做单seed
40-round配对筛选；若仍不能改善Actor mean，则停止改搜索bank，转向Actor objective/update geometry的
诊断性上限试验。不要再扫单一radius、cap、LR、K或round。

权威哈希：

```text
radius-ab config       62e3ae17b175fee45224f5b172e1d6a8b4e99a02300f9fc7893e22d38e0fc2fe
radius-ab runner       ff3e808057bbc9b483fd33f4c107f8a85ee6ecad5e81cdd84d0107864827433d
radius-ab validator    3e776265787608ee3082133a1a2ebaf98108f7b887f3cb310d06f364d1efe0d5
radius-ab manifest     0569bbdb837165256d33391ce3bdbc04b5b757a91ef57e7603ebdc241076737d
radius-ab summary      c3c60c96662584a71266b937e0d7ab1768504f4d7569c3e74f5def81002bdf9c
radius-ab validation   6982077af123dff0b4bf543d3e45a35b9aad41e1585cd32b39c38f34950f5d22
```

### 17.38 coarse-to-fine 77-bank单seed配对预注册

按17.37路由，下一项只改变online bank construction，不改Actor/Critic结构、loss、更新次数、LR、cap、
gamma或状态分布。共同source仍固定为17.33长程seed0 round139 selected Actor/Twin Critics；两臂重新初始化
相同optimizer与空的新增online Replay，使用相同seed、40轮context/Actor batch schedule、20 fit contexts/
round、20 Critic updates/Twin、K16、Actor LR2e-5、Critic LR1e-4、cap0.02、gamma1和完整120-row inner
Query checkpoint选择。所有好坏candidate进入累计Replay，warm只作外部comparator。

两臂为：

- control：保持17.37的65-bank，第一阶段39 response使用现行`0.20→0.05`的90轮线性schedule前40轮，
  再围绕真实Query incumbent使用`0.7x`半径、固定QR260904前13方向的26个antithetic probes；
- treatment：每次访问固定77个候选。第一阶段用`1.00→0.10`的90轮线性schedule前40轮做完整39 response；
  取这39项的真实Query argmin为incumbent，并复用已算cost/residual，不重复rollout；第二阶段用
  `0.20→0.05`的对应local schedule和独立满秩QR260905基，再做32个antithetic probes与6个response
  proposals。总数严格为`1 + 32 + 6 + 32 + 6 = 77`，后一阶段完全依赖前一阶段winner。

第二阶段必须使用实际clipped displacement拟合，保留stage1全部candidate而非只存winner；不得把winner
作为BC label。两臂online预算分别为52,000与61,600，合计113,600个Query candidates。treatment仅比
control增加18.5%，用于同时保留广域定位和局部覆盖，不做大批独立随机采样。

结果前固定裁决：treatment selected inner mean J严格低于paired control，且warm-relative aggregate更高，
才允许进入三seed确认；否则保留65-bank。tail、bank-best分布和recent Critic ranking只诊断。独立validator
必须验证source/adapter、两臂round0与随机schedule一致，重建coarse winner到fine probes/proposals的完整
依赖链，重放全部113,600个online candidates及selected Actor inner cost。outer/formal/test、DBM字段/
label和Query解析梯度继续封存。

### 17.39 coarse-to-fine 77-bank单seed结果：搜索端明显增强，Actor端仅弱阳性

17.38实验与独立验证已完成：

```text
scripts/model_verify/query_continuous_ac_coarse_to_fine77_ab_config_20260904_v1.json
scripts/model_verify/run_query_continuous_ac_coarse_to_fine77_ab.py
scripts/model_verify/validate_query_continuous_ac_coarse_to_fine77_ab.py
outputs/query_mppi/query_continuous_ac_coarse_to_fine77_ab_20260904_v1
```

两臂共同round0 mean J=`2.839481`。主体结果为：

| bank | selected round | selected mean J | warm aggregate | latest mean J |
|---|---:|---:|---:|---:|
| recenter65 control | 40 | 2.793708 | 47.7965% | 2.793708 |
| coarse-to-fine77 | 28 | **2.793137** | **47.8072%** | 2.882826 |

treatment相对control只降低`0.000571 J`（约`0.0204%`），warm aggregate只增加`0.0107`个百分点。它按
预注册的严格数值门同时通过mean与aggregate，裁决为
`PROMOTE_COARSE_TO_FINE77_TO_THREE_SEEDS`；但必须称为**弱阳性**，不得解释为已经吸收17.35的
36.15% headroom。latest明显回退也再次证明必须保留完整inner checkpoint selection。

search bank自身的改善更明确。control的bank-best mean/median gain与正收益访问比例为
`0.43581/0.17368 J/96.375%`；77-bank为`0.76281/0.19277 J/98.5%`。在77-bank内部，第一阶段
Actor→coarse-best mean gain=`0.57673 J`、正收益率76.875%；第二阶段在coarse-best上额外贡献mean
`0.18609 J`、median `0.04967 J`，89.875%的访问进一步改善。这证明依赖前一结果的fine response同时
修复了单尺度broad的覆盖缺口并保留大收益；但更好的Replay候选仍几乎没有转成更好的Actor输出，剩余
瓶颈继续指向Actor objective/update geometry，而不是低cost候选缺失或Critic完全无法排序。

独立validator全量重放control 52,000与treatment 61,600个online candidates，重建每组coarse真实
winner、复用cost/residual、fine满秩probes/response与role；全部action/cost/raw/clipped、selected Actor
action/cost、source adapters、round0和随机schedule误差均为0，资格为：

```text
QUERY_CONTINUOUS_AC_COARSE_TO_FINE77_AB_INDEPENDENT_PASS
```

权威哈希：

```text
coarse77 config       6d4b2b55230761ae37eeb1c5f815b3fbea1d6f5338bfb2c36412dcd222c715f8
coarse77 runner       3fa7be86e8dd1cd1adbfa838964d97a266b084dc8bff9c1270280e6ac36603cf
coarse77 validator    825fdfdb728fd7a5b80d928ac362e7ca52def5ac67c00ef60bb8a07944be9997
coarse77 manifest     e681c48e50959af1d0d234560ff3c3e44717b4ac7d1d477bba47c785d3d7de94
coarse77 summary      b7d4d84b9e351a2565e10eed0d9650719003c2d6e2ffa137b276db3f80ea1bf4
coarse77 validation   8d3ee8765c1a262d99ab70aa5d94e6f824872cfe534b81ba2c87db81e55bfe24
```

### 17.40 coarse-to-fine 77-bank三seed确认预注册

因17.39效应很小，不能凭seed0直接替换65-bank。三seed确认完全保留17.38合同，分别从17.33三个selected
Actor/Twin Critics（round139/131/139）启动40轮paired control/treatment。seed0直接复用17.39已经独立
验证的不可变record/checkpoint/arrays；仅新增seed1/2训练。复用前必须验证17.39 summary/validation哈希、
资格、source checkpoint和两臂artifact哈希；不得因为节省计算而降低证据门槛。

最终门在结果前固定为：coarse-to-fine77 pooled selected mean低于recenter65、pooled warm aggregate更高，
且至少2/3 seeds的selected mean严格降低，三项同时满足才正式promote；否则认定seed0微增益不可复现，
保留65-bank并停止candidate/radius扩展，转向Actor objective/update geometry诊断。tail仍只诊断。独立
validator对复用seed0检查既有独立资格与哈希，对新增seed1/2重放每seed 113,600个online candidates、
coarse→fine链和selected Actor；然后从三seed全部arrays独立复算pooled裁决。outer/formal/test、DBM字段/
label和Query解析梯度继续封存。

### 17.41 coarse-to-fine 77-bank三seed确认：可复现但增益小，停止扩bank

17.40三seed确认与独立验证已完成：

```text
scripts/model_verify/query_continuous_ac_coarse_to_fine77_3seed_config_20260904_v1.json
scripts/model_verify/run_query_continuous_ac_coarse_to_fine77_3seed.py
scripts/model_verify/validate_query_continuous_ac_coarse_to_fine77_3seed.py
outputs/query_mppi/query_continuous_ac_coarse_to_fine77_3seed_20260904_v1
```

seed0直接复用17.39已独立验证产物，seed1/2重新运行paired 40 rounds。结果为：

| seed | recenter65 mean / round | coarse-to-fine77 mean / round | reduction |
|---:|---:|---:|---:|
| 0 | 2.793708 / 40 | 2.793137 / 28 | +0.000571 |
| 1 | 2.910272 / 0 | 2.902242 / 39 | +0.008030 |
| 2 | 2.908747 / 39 | 2.895027 / 20 | +0.013721 |

pooled mean J由`2.870909`降至`2.863468`，降低`0.007441 J`（约control的`0.259%`）；warm-relative
aggregate由`46.3539%`升至`46.4930%`，增加`0.1390`个百分点。3/3 seeds均严格改善，预注册三门全部
通过，正式裁决为：

```text
PROMOTE_COARSE_TO_FINE77
```

辅助指标：胜/平warm由`81.94%`升至`82.50%`，gain median由`0.70144`升至`0.71897 J`；P05几乎持平
但略差（`-1.44595→-1.44699 J`），worst由`-9.2699`变为`-10.0250 J`，仍只作诊断。速度层并不一致：
77相对65的mean J变化在40/55/70/85/100 km/h分别为`+0.00171/+0.01486/-0.00150/-0.07471/
+0.02245 J`（正数为回退）。pooled收益主要由85 km/h贡献，不能声称所有速度都受益。

搜索端改善则在三seed一致：77-bank Actor→best mean gain为`0.7628/0.8096/0.7042 J`，65-bank为
`0.4358/0.4701/0.3940 J`；77-bank正收益访问比例为`98.5/97.5/97.0%`，65-bank为
`96.38/96.75/95.88%`。因此coarse-to-fine构造有效，但Actor只吸收了其中很小一部分。

独立validator复核seed0既有资格和artifact哈希，并对新增seed1/2重放`227,200`个online candidates、
两阶段winner/response链与selected Actor；source adapters、两臂schedule/round0、全部action/cost/raw/
clipped/role和三seed pooled指标误差均为0。资格为：

```text
QUERY_CONTINUOUS_AC_COARSE_TO_FINE77_THREE_SEED_INDEPENDENT_PASS
```

77-bank现在可作为后续candidate baseline，但`0.259%`增益远小于17.35的36.15%数值headroom，且速度层
呈明显trade-off。停止继续增加candidate数、stage数或扫radius；也不因这个小增益直接再堆160轮。
下一步做Actor objective/update geometry诊断：在当前三个77 selected Actor/Twin Critics上，复算gamma1
总体及40/55/70/85/100速度组的参数梯度cosine/norm，并比较Critic Actor梯度与同一online bank的经验
best-improvement参数方向；只在发现稳定负冲突时测试PCGrad/CAGrad。该诊断不得把bank-best变成训练
teacher，也不得使用outer/formal/test或Query解析梯度。

权威哈希：

```text
coarse77-3seed config       858b7d3873ea2d4255b270573fdc11d8e1f7ed8014115718b7463c1ad11da063
coarse77-3seed runner       b423bdbf206f2c57cd6451df6cfca9cbdfac58602f01884ed9d9a1f650b4729d
coarse77-3seed validator    092dc7fec58f038d1dc314c6914ad18d669b689cfcdb7a680c6fbe642f6a2b1e
coarse77-3seed manifest     6bf29a0ec55e96046dfa76332783e954455023f83fa871f1c597f644c6f85558
coarse77-3seed summary      a91d3532128a85e321b1da8d0b04ad66eac83732474c0b72267086e450cb82d2
coarse77-3seed validation   3e339d43cc8f2b9fc72e6720843513313f5a4c3352a05807eb9bd9c729c859cc
```

### 17.42 coarse-to-fine77 Actor更新几何诊断预注册

按17.41路由，下一步只做train/internal-selection机制诊断，不继续扩大bank、扫描radius或追加AC轮次。
source固定为17.41已经独立通过的三个`coarse_to_fine77` selected Actor/Twin Critics及各自40轮
77-candidate online arrays；运行前校验三seed summary/validation资格、hash和每个checkpoint/arrays hash。
每seed只消费432个fit状态、其800次online访问以及120个inner-selection状态，outer/formal/test继续封存；
不得使用DBM字段/label、Query解析梯度、warm输入或bank-best监督训练，Actor/Critic权重始终不落盘修改。

梯度口径固定为gamma1 raw-cost-aware Actor objective：在当前selected Actor输出上取Twin Absolute Critics的
conservative maximum，使用截断到2048后均值归一化的`exp(detached log1p J)`权重。报告全fit与
40/55/70/85/100 km/h五个速度组的参数**更新方向**（即负loss gradient）范数、总体对各组cosine及10个
速度对的cosine。所有速度组均各自重新归一化gamma1权重，避免样本数改变目标尺度。

同一online bank的经验方向不作为teacher，而定义为只读secant oracle。对每次访问，candidate 0必须是
保存的Actor center，真实Query cost最低项为bank best；令归一化动作位移
`d=(a_best-a_center)/sigma`、收益`g=J_center-J_best>=0`，经验raw-cost下降响应为
`q=g*d/(||d||^2+1e-8)`。先在同一物理fit row内跨访问平均q以恢复state-balanced口径，再在当前selected
Actor Jacobian上计算`J_pi^T q`，得到经验参数更新方向；同时报告不乘secant斜率的raw displacement方向
作为尺度敏感性辅助项。复算总体/速度组经验方向、速度冲突以及Critic-update对经验方向cosine。不得将
bank best做MSE、BC或任何参数更新。

为避免只凭高维cosine下结论，将Critic gamma1更新方向和经验secant更新方向分别以确定性二分标定到
full-fit Actor输出`0.001 sigma RMS`，仅在120个inner-selection状态各做一次Query direct-cost复放；报告
相对source selected Actor的mean、median、P05、worst及各速度gain。该微步只存在内存和诊断数组中，
不成为checkpoint，也不参与训练。独立validator必须重载三seed source，逐项复算bank winner/secant、
全部参数向量、标定动作及720个Query costs，误差门为action/gradient `<=1e-6`、cost `<=1e-5`。

结果前冻结路由：把速度对cosine `<=-0.05`定义为实质负冲突；某一速度对在至少2/3 seeds满足才称为
stable pair。只有当至少两个stable pairs存在、且至少2/3 seeds的Critic总体方向与经验secant方向cosine
为正、Critic微步inner mean不劣于source时，才允许进入PCGrad/CAGrad A/B。若Critic与经验方向在至少
2/3 seeds非正，优先诊断Critic objective/locality，PCGrad无资格；若没有至少两个stable pairs，则否定
“速度梯度冲突是当前主要瓶颈”，不得为了尝试而上PCGrad。inner tail只作诊断，不覆盖mean主门。

### 17.43 Actor更新几何结果：无稳定速度冲突，PCGrad/CAGrad不准入

17.42预注册诊断及独立复核已完成：

```text
scripts/model_verify/query_actor_update_geometry_config_20260904_v1.json
scripts/model_verify/analyze_query_actor_update_geometry.py
scripts/model_verify/validate_query_actor_update_geometry.py
outputs/query_mppi/query_actor_update_geometry_20260904_v1
```

三个seed各有800次77-bank访问，state-balanced后均覆盖404/432个fit状态。Critic gamma1更新方向与经验
secant参数方向的总体cosine分别为`-0.4576/+0.1880/+0.1493`；只有2/3为正且正值较弱，说明Critic与
历史bank下降方向存在明显seed差异，但未达到“至少2/3非正”的Critic失配否决门。raw displacement
辅助cosine为`-0.1148/+0.0259/+0.3721`，同样不支持把bank endpoint直接当成稳定共享Actor方向。

速度梯度没有形成跨seed稳定的系统冲突。10个速度对中只有`55--85 km/h`在2/3 seeds的Critic cosine
低于预注册`-0.05`门（`-0.1224/+0.0534/-0.0875`）；其余9对均不稳定，因此stable pair只有1个，低于
PCGrad准入所需的2个。Critic速度对cosine median按seed为`0.0793/0.0445/0.1619`。经验secant各速度
方向反而更兼容：速度对minimum为`+0.0011/+0.0052/-0.0234`，median为
`0.1852/0.2762/0.3021`。裁决为：

```text
NO_STABLE_SPEED_CONFLICT_DO_NOT_PCGRAD
```

确定性标定到full-fit输出`0.001 sigma RMS`后，Critic gamma1方向在inner上的mean gain逐seed为
`+0.000233/+0.009457/+0.003533 J`，pooled为`+0.004408 J`；median仅`+0.000109 J`，P05为
`-0.02968 J`。经验secant方向逐seed为`+0.000383/-0.000255/+0.003083 J`，pooled仅
`+0.001070 J`。因此当前Critic小步总体确有下降信号，且比把bank响应直接映射到共享Actor更有效；
但效应很小并伴随状态回退，不能解释为已解决17.35的36.15% headroom。bank单次访问mean gain仍为
`0.7628/0.8096/0.7042 J`，搜索收益到共享Actor收益之间的巨大差距继续存在。

独立validator第二次重载三seed selected Actor/Twin Critics，逐项复算bank winner、state-balanced
secant、全部参数向量、二分标定动作及720个Query costs；所有array/report/routing最大误差均为0，资格为：

```text
QUERY_ACTOR_UPDATE_GEOMETRY_INDEPENDENT_PASS
```

本轮否定的是**速度组**PCGrad，不是否定持续AC。不得接着做PCGrad/CAGrad、bank-best BC、多头、cap/LR/
K/radius或无依据的更多轮次扫描。值得继续的零训练诊断是把同一口径细化到20个
`speed_kph × variant_index`道路分层：当前full-fit总体参数方向范数仅为`3.17/1.37/2.06`，若干单速度
方向范数却达到`13.78/15.62/8.75`，说明抵消可能发生在同一速度内部的不同道路形态。下一步先复算20
分层的Critic与经验secant更新cosine、找跨seed稳定负冲突；只有该层出现稳定冲突才设计分层梯度组合。
若仍无冲突，则把瓶颈归为共享Actor对多状态局部方向的表示/非线性投影与在线优化效率，而不是继续做
gradient surgery。

权威哈希：

```text
geometry config       52da0d1943d46f4bf5dc48fd15e2bfa317e5ace274edd671e2514df9eb5a4c7b
geometry runner       425a147991eb04bd760e8f9a533bcf54ae75c3b7e999138b6dc79511c3fca72f
geometry validator    def9bd4c0f907d94b835358c70ccbd94d712bdfbf32f48c27e875175787b2246
geometry manifest     9edf5646f67dd6e2b9aa7dfac25182c0d1f7b55811f46a6c253f476b72aa64fc
geometry summary      db652707281829dc5a43ebbd6c06813a1c6cb50b75427c3bb42614ae40a53b29
geometry validation   753aab54e5d748cd30a9215496899f46c44883b39d58aab5b5a4947e3181a8e0
```

### 17.44 speed×road-variant 20分层更新几何预注册

按17.43路由，下一步仍为零训练、零新增Query rollout的机制诊断。source固定为17.42--17.43独立通过的
三seed Actor更新几何工件及其对应coarse-to-fine77 selected Actor/Twin Critics。运行前校验geometry
summary/validation、coarse77 source及每seed checkpoint/online arrays hash；只消费fit与已经消费过的
online bank，inner只保留索引而不做新评价，outer/formal/test、DBM字段/label和Query解析梯度继续封存。

20个cell定义为五个`speed_kph`乘四个`variant_index`。variant语义按已验证collection manifest固定为：
`0=mild_left_nominal`、`1=moderate_right_nominal`、`2=varying_left_recovery`、
`3=varying_right_recovery`。普通cell各有18个fit状态，target-coverage扩展后的`55:2`与`100:3`各54个；
三个seed的online bank在每cell覆盖至少11个独立状态。每个cell分别复算：

1. 当前selected Actor上、该cell内部重新归一化gamma1权重的Critic参数更新方向；
2. 17.42同一定义的state-balanced经验secant参数方向；
3. 两者norm与cosine、各自20×20 pairwise cosine，以及等cell平均方向相对原生全fit方向的cosine。

主判定只使用**同速度、不同variant**的`5×C(4,2)=30`对，避免把17.43已否定的跨速度冲突重新混入。
参数cosine `<=-0.05`仍定义为实质负冲突；同一cell pair在至少2/3 seeds实质为负才称stable。为区分
真实道路目标冲突与Critic伪冲突，只有同时在Critic和经验secant两套方向中stable的pair才称supported。
跨速度的160对只作诊断。

结果前冻结路由：全局20-task PCGrad/CAGrad只有在supported pair至少4个、且覆盖至少3/5速度时才准入；
若Critic stable pair至少6个且覆盖至少3个速度、但supported门未通过，则判为Critic-only分层冲突，转向
Critic locality/ranking诊断而非gradient surgery；若两门都不满足，则否定“道路分层梯度冲突是当前主要
瓶颈”，后续转向共享Actor对逐状态局部改进的表示/投影容量诊断。不得因某一个cell或单seed负cosine
启动PCGrad、重权、BC、多头、LR/cap/K/radius或round扫描。独立validator须重载source并逐值复算
三个seed全部`2×20`个591250维参数方向、矩阵和裁决，gradient误差门`<=1e-6`。

### 17.45 20分层结果：道路冲突不系统，发现fit/inner cell权重错配

17.44的20分层零rollout诊断与独立复核已完成：

```text
scripts/model_verify/query_actor_stratum_geometry_config_20260904_v1.json
scripts/model_verify/analyze_query_actor_stratum_geometry.py
scripts/model_verify/validate_query_actor_stratum_geometry.py
outputs/query_mppi/query_actor_stratum_geometry_20260904_v1
```

主判定的30个同速度道路对中，Critic stable pair有5个，覆盖70/85/100 km/h；经验secant stable pair也有
5个，但两者交集仅2个、只覆盖85/100 km/h：

```text
supported: 85:1|85:2, 100:2|100:3
```

这低于预注册“至少4对且覆盖3个速度”的PCGrad门。Critic stable数量5也低于Critic-only门的6，最终裁决：

```text
NO_SYSTEMATIC_ROAD_STRATUM_CONFLICT_DO_NOT_PCGRAD
```

存在局部但不能全局化的Critic伪冲突。最明显的`70:1|70:3` Critic cosine在三seed为
`-0.619/-0.934/-0.775`，而真实bank经验secant为`+0.922/+0.651/+0.956`；因此在20任务上直接做
gradient surgery可能保护错误的Critic边界。逐cell Critic与经验方向同号比例也只有
`60%/65%/70%`。这些现象可作为以后Critic locality诊断的定位信息，但没有达到替换当前持续AC机制的门。

本轮更重要的新线索是Actor训练分布与inner评价分布的权重错配。20个inner cell严格等权，各6行；fit中
18个普通cell各18行，而补覆盖的`55:2`、`100:3`各54行。当前Actor uniform-row batch因此让这两个cell
各获得普通cell三倍权重，总计占fit目标25%，而inner目标只占10%。20个Critic cell方向等权平均与当前
native uniform-row方向的cosine仅为`0.609/0.871/0.762`。相比之下，经验secant的等cell与native cosine
为`0.998/0.989/0.992`；所以错配首先体现在Critic Actor objective，而不是bank搜索方向本身。该结果仅
说明两个objective方向有实质差异，还没有证明等cell方向的真实Query cost更优。

独立validator重载全部source，复算三seed共120个591250维参数方向、20×20矩阵、道路语义与裁决；所有
array/report/routing最大误差为0，且没有新增任何Query rollout。资格为：

```text
QUERY_ACTOR_STRATUM_GEOMETRY_INDEPENDENT_PASS
```

下一步应先做低成本matched-step objective A/B，而不是直接40轮训练：同一selected Actor/Twin Critics上
比较当前uniform-row gamma1方向与20-cell等权gamma1方向，标定到相同full-fit输出RMS，在同一120-row
inner上复放Query。若等cell方向在至少2/3 seeds与pooled mean均优于native，且较大一步不反转，再进入
40-round continuous-AC单变量A/B；否则否定cell weighting，转向逐状态共享Actor的表示/投影容量诊断。
仍不得使用bank-best BC、多头、PCGrad或重新扫描已有LR/cap/K/radius。

权威哈希：

```text
stratum config       714f4eaba356640091522e3d169660315271df77b63667d3b0a85c6cacd24adc
stratum runner       4a329eefa7643f35e90deea12e55f7a6168f7b8cf7464eb94a637ae8e9592b82
stratum validator    0e65d60384a39d5d0812618641202d8b08c361acfb9933857103575f25da447c
stratum manifest     216974024f927b004fa9d45edc5a8e0e9cdb0cc512d9ecfbe4ca8ffffd679827
stratum summary      481c61949eab14c77565e4e3784686062d18c63b88b199df716663b32766eff1
stratum validation   58f8a7e7dd05d5a1e6f689eaf9936c4bb59696c646bb315c235e67c97899873a
```

### 17.46 equal-cell Actor objective matched-step A/B预注册

按17.45路由，下一步先做低成本方向A/B，不直接启动40轮训练。source固定为17.44--17.45独立通过的
20分层参数工件、17.42--17.43 selected Actor/Twin Critics及同一120-row inner。两臂为：

- `native_uniform_row`：17.42当前gamma1 full-fit uniform-row Critic更新方向；
- `equal_20_cell`：17.44中20个独立cell gamma1更新方向的算术平均，每个cell在自身内部重新归一化
  raw-cost-aware权重，因此20个cell严格等权，不再让`55:2`、`100:3`因行数得到三倍权重。

两臂均只作为内存参数方向，先单位化，再用确定性二分分别标定到full-fit Actor输出
`0.001/0.002 sigma RMS`；每个seed、arm、步长在同一120-row inner上运行Query direct-cost，共新增
`3×2×2×120=1440`次rollout。Actor/Critic权重不得写回checkpoint，不改变网络、optimizer、Replay、bank、
gamma、LR、cap、K或训练轮数。主指标为相对source的inner mean gain及两臂paired mean cost；median/P05/
worst、速度和variant只诊断。outer/formal/test、DBM字段/label、warm输入与Query解析梯度继续封存。

结果前冻结晋级门：在主步长`0.001`与较大步长`0.002`上，equal-cell都必须满足pooled mean cost严格低于
native、至少2/3 seeds mean cost严格低于native、且自身相对source pooled mean gain严格为正；六项同时
成立才允许进入40-round continuous-AC单变量A/B。任何一项失败即判定matched-step证据不足，保留
uniform-row objective并转向逐状态共享Actor表示/投影容量诊断；不得事后选择单个步长或仅按tail晋级。
独立validator必须重载source、复算两臂参数方向/二分倍率/动作并独立重放全部1440个Query costs，
gradient/action误差门`<=1e-6`、cost误差门`<=1e-5`。

### 17.47 equal-cell matched-step结果：弱阳性，进入40-round单变量A/B

17.46双步长配对诊断及独立Query重放已完成：

```text
scripts/model_verify/query_actor_equal_cell_matched_step_config_20260904_v1.json
scripts/model_verify/run_query_actor_equal_cell_matched_step.py
scripts/model_verify/validate_query_actor_equal_cell_matched_step.py
outputs/query_mppi/query_actor_equal_cell_matched_step_20260904_v1
```

两臂参数方向cosine逐seed为17.45已报告的`0.6089/0.8713/0.7617`。相同步长结果为：

| full-fit输出步长 | native pooled gain | equal-cell pooled gain | equal相对native mean优势 | equal胜seed数 |
|---:|---:|---:|---:|---:|
| 0.001 sigma RMS | +0.004408 J | **+0.004944 J** | **+0.000536 J** | 2/3 |
| 0.002 sigma RMS | +0.009137 J | **+0.009664 J** | **+0.000527 J** | 2/3 |

逐seed equal-cell gain在0.001步长为`-0.001033/+0.011690/+0.004174 J`，native为
`+0.000233/+0.009457/+0.003533 J`；0.002步长分别为
`-0.001983/+0.022009/+0.008965 J`与`+0.000883/+0.018445/+0.008083 J`。因此seed0两档都回退，
但两个步长均满足pooled更优、2/3 seeds更优及equal自身pooled gain为正，预注册六门全部通过，裁决为：

```text
ADVANCE_EQUAL_CELL_TO_40ROUND_OAC_AB
```

必须称为**弱阳性机制证据**，不能直接promote objective。equal相对native的逐状态paired median cost差在
0.001/0.002步长为`+0.000040/+0.000119 J`（正数表示equal略差），mean改善来自非均匀状态。按速度，
equal相对native cost差在0.001步长的40/55/70/85/100 km/h为
`+0.00052/+0.00034/+0.00336/-0.00747/+0.00056 J`，0.002步长为
`+0.00146/+0.00097/+0.00637/-0.01427/+0.00284 J`；主体优势主要由85 km/h贡献。该异质性不覆盖
mean主门，但要求后续训练继续完整inner checkpoint选择，不能用latest或单速度替代。

独立validator重载三seed source，复算两臂方向、二分倍率、动作并再次重放全部1440个Query costs；
所有array/report/routing最大误差为0，资格为：

```text
QUERY_ACTOR_EQUAL_CELL_MATCHED_STEP_INDEPENDENT_PASS
```

下一步只允许做一次40-round continuous-AC配对A/B：共同从17.41同一三个longrun selected Actor/Twin
Critics重新开始，使用同一coarse-to-fine77 bank、Replay、20 Critic updates/Twin、K16、Actor LR2e-5、
Critic LR1e-4、cap0.02、gamma1、context/Actor batch schedule与逐轮完整inner选择；唯一变量是在Actor
loss中把gamma1 sample weight再乘以`1 / fit-cell-count`并归一化，使20个cell期望等权。Critic训练与
online context访问保持原样，不能同时改成cell-balanced。control可复用17.41已独立验证产物，但必须核对
全部source/config/schedule hash。最终替代门仍为equal pooled selected mean更低、warm aggregate更高、
且至少2/3 seed mean更低；否则保留uniform-row。tail与速度只诊断。

权威哈希：

```text
equal-cell config       e25cd45b1766cee2dd357dfec1a485af750c04d5f7e4a924e13699b24208ec2f
equal-cell runner       233603a215ee4a90fe5f693bf7be9a8c8f11e44f7e2492cbebb627f6ece96c7d
equal-cell validator    f4e1eb75a62eee0c911b9d31e418f0521d631c3ca1b23a62c0c9d197f1a5c47b
equal-cell manifest     124bdbe8bb7e0035359f341778d55fc5801ef09cf6e59946345f3c57b4107e19
equal-cell summary      109f0d91a4118f5ac28f69bdd8abf83adb9ea06626758fe4773b21f00c4a1a8c
equal-cell validation   b5dde96f2439f9f40f097f6659534a82dcd73b9b2fc6e6214a719355ca0e766e
```

### 17.48 equal-cell 40-round continuous-AC 单变量A/B预注册

按17.47冻结路由执行正式训练A/B。control固定复用17.41中已经独立验证的三seed
`coarse_to_fine77`记录；treatment从完全相同的三个longrun selected Actor/Twin Critics重新开始，继续使用
同一77-candidate coarse-to-fine bank、absolute Replay、每轮20次Critic更新/Twin、K16 Actor microsteps、
Actor LR `2e-5`、Critic LR `1e-4`、cap `0.02 sigma RMS`、gamma1、40轮、context/Actor batch随机种子和
逐轮完整120-row inner Query checkpoint选择。不得消费outer/formal/test、DBM字段/label、warm输入或Query
解析梯度。

唯一变量是Actor loss的样本权重：先按原口径计算并截断`exp(gamma * conservative_log_cost)`，再乘以
`432 / (20 * fit_cell_count)`并在每个Actor minibatch内归一化。因而普通18-row cell的cell factor为`1.2`，
`55:2`和`100:3`两个54-row cell为`0.4`，使20个`speed_kph × variant_index` cell在全fit抽样期望中等权。
Critic训练、online context访问和候选bank不得做cell balance，也不得同时改变网络、LR、cap、K、gamma、
bank或训练轮数。control产物允许复用，但runner必须锁定其summary/validation及每个arrays/checkpoint hash，
并逐seed核对Actor batch schedule、visited context/round及round0 action/cost完全一致。

主判定只看完整inner selected checkpoint：treatment必须同时满足pooled mean cost严格低于control、pooled
warm-relative aggregate严格高于control、且至少2/3 seed selected mean严格更低，才可替换uniform-row
objective；否则保留uniform-row。tail、median、速度和variant只作诊断，不覆盖mean主门。独立validator须
验证所有source/config/artifact hash与cell factor，重载三个selected Actor复算360个inner costs，并逐组
重建treatment全部`3 × 40 × 20 × 77 = 184800`个online Query candidates；action/cost/report误差门沿用
`1e-6`，最后独立复算pooled指标和裁决。

### 17.49 equal-cell 40-round结果：主门通过，替换uniform-row Actor objective

17.48预注册三seed单变量训练与独立复放均已完成：

```text
scripts/model_verify/query_continuous_ac_equal_cell_ab_config_20260904_v1.json
scripts/model_verify/run_query_continuous_ac_equal_cell_ab.py
scripts/model_verify/validate_query_continuous_ac_equal_cell_ab.py
outputs/query_mppi/query_continuous_ac_equal_cell_ab_20260904_v1
```

20-cell权重独立确认：18个普通18-row cell的factor为`1.2`，扩展后的`55:2`与`100:3`两个54-row
cell为`0.4`，各cell在full-fit上的期望总mass均为`21.6`。Critic与online context仍保持原始uniform-row
口径。三seed全部Actor batch schedule、visited rows、online round及round0 action/cost与复用control完全一致。

主指标如下：

| arm | pooled selected mean J | warm mean gain | warm aggregate | win/tie fraction | P95 J |
|---|---:|---:|---:|---:|---:|
| coarse77 uniform-row control | 2.863468 | 2.488105 | 0.464930 | 82.50% | 6.933807 |
| coarse77 equal-20-cell | **2.849886** | **2.501687** | **0.467468** | 80.56% | **6.742386** |

equal-cell把pooled mean降低`0.013583 J`（相对control约`0.474%`），warm aggregate增加`0.002538`。
逐seed control→treatment为：seed0 `2.793137→2.781486`（改善`0.011651`，selected round 18），seed1
`2.902242→2.910272`（回退`0.008030`，selected round 0），seed2 `2.895027→2.857899`（改善
`0.037127`，selected round 20）。因此pooled mean、warm aggregate、2/3 seed三项预注册主门全部通过，裁决：

```text
PROMOTE_EQUAL_20_CELL_ACTOR_OBJECTIVE
```

该结论只替换后续持续AC中的Actor sample weighting，不改Critic、bank或模型结构。效果仍有明显异质性：
分速度mean优势（control J减treatment J）在40/55/70/85/100 km/h分别为
`+0.00031/-0.00060/+0.02677/+0.04787/-0.00644 J`，主要来自70和85 km/h；中位gain从
`0.71897`轻微降至`0.70499`且win/tie从82.50%降至80.56%，所以它是mean目标下的有效改进，不能解释成
全面逐状态占优。P90/P95从`4.97765/6.93381`改善到`4.92117/6.74239`，长尾不是晋级依据。

独立validator重载三个source adapter与selected Actor，复算360个inner costs，重新构造并Query复放全部
184800个treatment candidates；所有adapter/cell/schedule/action/cost/report/pooled误差为0，资格为：

```text
QUERY_CONTINUOUS_AC_EQUAL_CELL_THREE_SEED_INDEPENDENT_PASS
```

权威哈希：

```text
equal-cell OAC config       c1376c768cb8f57d2cc100cb1141293e168ddaa46333d89f9516d604ab694d94
equal-cell OAC runner       32795c61464f836245a05d2f928c3a18e72a89acfbe3b38d658119e7d64195a1
equal-cell OAC validator    0e7fcadd65fc5bc8da7920b787d55522dc958ab3812a36302548effa4688befd
equal-cell OAC manifest     ab76abdab199a13ee0135040c77c729fa8476ba0be6a20e000ac2a7bd4009969
equal-cell OAC summary      1cb6a64429f831d156a86cfc004302ae9fac410fd7cb2ce226bc011cde9f78ab
equal-cell OAC validation   79497ad9694dd385599abf4b082f6260ef4b8d93c58b85d35196db99f92fa83c
```

### 17.50 equal-cell 160-round持续AC预注册

17.49只证明equal-cell在40轮配对A/B中优于uniform-row，不证明40轮已收敛。三seed treatment的selected
round为18/0/20，但round 22/27/33/38仍出现次优低谷，seed1第15轮仅比round0差约`0.0010 J`；因此下一步
允许把同一equal-cell objective从同一三个longrun selected Actor/Twin Critics运行到160轮，检查后120轮
能否形成新的完整inner低点。除round数外，77-bank、Replay、Critic 20/Twin、K16、Actor LR2e-5、Critic
LR1e-4、cap0.02、gamma1、半径前90轮`1.0→0.1`后固定0.1、随机种子及逐轮120-row inner选择全部不变。

前40轮必须与17.49 treatment逐seed完全相同：Actor batch schedule、visited rows、online actions/costs、
selection-round actions/costs逐项误差为0；否则长跑无资格。主门冻结为160-round selected pooled mean低于
17.49的`2.8498857747`、warm aggregate更高，且至少2/3 seed的selected mean严格低于对应40-round
selected；全部通过才以160-round替换40-round promoted baseline，否则保留17.49。latest、tail、速度和
单个seed只诊断。独立validator可复用17.49已复放且前缀完全相同的184800 candidates，只需重新Query复放
round 41--160的`3 × 120 × 20 × 77 = 554400`个suffix candidates，同时重载最终selected Actor并复算
360个inner costs及pooled裁决。outer/formal/test、DBM字段/label、warm输入和Query解析梯度继续封存。

### 17.51 equal-cell 160-round结果：3/3 seed继续改善，正式替换40-round baseline

17.50预注册长跑及独立suffix复放已完成：

```text
scripts/model_verify/query_continuous_ac_equal_cell_longrun160_config_20260904_v1.json
scripts/model_verify/run_query_continuous_ac_equal_cell_longrun160.py
scripts/model_verify/validate_query_continuous_ac_equal_cell_longrun160.py
outputs/query_mppi/query_continuous_ac_equal_cell_longrun160_20260904_v1
```

三seed的前40轮Actor schedule、全部online action/cost/role、selection action/cost与17.49 treatment逐项误差
均为0，证明唯一变化确实只是继续运行到160轮。长跑结果为：

| seed | 40-round selected J / round | 160-round selected J / round | 进一步降低 |
|---:|---:|---:|---:|
| 0 | 2.781486 / 18 | **2.578983 / 160** | **0.202503** |
| 1 | 2.910272 / 0 | **2.891265 / 58** | **0.019008** |
| 2 | 2.857899 / 20 | **2.779129 / 150** | **0.078770** |

pooled mean从`2.8498858`降至`2.7497921`，进一步降低`0.1000936 J`（约3.51%）；warm-relative aggregate
从`0.4674676`升至`0.4861712`。3/3 seed都在round 41--160出现新低点，三项预注册门全部通过，裁决：

```text
PROMOTE_EQUAL_CELL_LONGRUN160
```

新Actor相对warm的pooled mean gain为`2.601781 J`，median gain `0.764661 J`，win/tie为83.89%；Actor J的
median/P90/P95为`2.06808/4.63219/6.72188`。按速度40/55/70/85/100 km/h的mean J分别为
`2.42005/2.27778/2.31220/2.73298/4.00596`，warm aggregate分别为
`0.2041/0.2658/0.4175/0.5692/0.6111`。相对17.49，主要新增收益来自70与100 km/h；tail仍不参与主门。

独立validator校验前40轮零误差前缀后，重新Query复放round 41--160全部554400个suffix candidates，重载
三个selected Actor并复算360个inner costs；adapter、cell、action、cost、report、pooled和decision误差均
为0，资格为：

```text
QUERY_CONTINUOUS_AC_EQUAL_CELL_LONGRUN160_INDEPENDENT_PASS
```

权威哈希：

```text
equal-cell 160 config       c1d37bd502aaa9248f19d6a625f9e8cb7557c0d8a05fccd26131d14cfee8a11c
equal-cell 160 runner       eacc9214a824af7674c926a2f5b7a52a890b0fd2caade64af76bef3323547832
equal-cell 160 validator    168242992fcb758703181fd483a43cf50355eaabb8065d088ddcdfc003b1d065
equal-cell 160 manifest     84e5cc07afc08e4ea7a4f479573751447504063f474df45e76bc95858851519a
equal-cell 160 summary      ff05917b7a0880194feeff0fe9e20ed1b462487fa877c420ee445998ae7012e2
equal-cell 160 validation   87429419049dd664fe135481c3977c2292b8ff5293fb52ca991749981f1db45f
```

### 17.52 promoted equal-cell160 Actor-centered headroom复查预注册

seed0/2的selected round位于160/150，说明不能从训练轮数直接宣称已收敛；但也不能据此盲目扩到320轮。
下一步复用17.35的冻结Actor-centered四轮数值搜索，在17.51三个selected Actor的同一120-row inner上重新
量化剩余Query-cost headroom。每个state从Actor direct output开始，依次使用`1.0/0.5/0.25/0.1 sigma`
的16维full-rank antithetic probes及forward-only response proposals，每seed-state总153次Query，合计
55080次；不训练Actor/Critic，不写回checkpoint，不使用warm作搜索起点，不消费outer/formal/test、DBM
字段/label或Query解析梯度。

主指标及路由沿用17.35：最终数值best相对Actor的aggregate residual reduction `<2%`则停止广泛扩轮/扩
搜索；`2%--5%`或未过半状态改善只允许低成本单变量A/B；`>=5%`且多数状态改善则说明仍有material
headroom，应继续诊断“局部搜索收益为何没有被共享Actor吸收”，但不能把数值best改成BC teacher或上多头。
该复查首先回答是否值得继续AC，不把tail当否决门。独立validator必须重载source并重新执行全部55080次
Query rollout、response fit/winner chain及pooled裁决。

### 17.53 equal-cell160 headroom结果：仍有32.89%可搜索空间，尚未收敛

17.52的冻结四轮Actor-centered审计及独立全量复放已完成：

```text
scripts/model_verify/query_actor_centered_headroom_equal_cell160_config_20260904_v1.json
scripts/model_verify/run_query_actor_centered_headroom_equal_cell160.py
scripts/model_verify/validate_query_actor_centered_headroom_longrun160.py
outputs/query_mppi/query_actor_centered_headroom_equal_cell160_20260904_v1
```

因训练时batched Query与审计per-state Query执行路径存在极小数值差，runner先建立只用于执行口径桥接的
source adapter；三个seed最大cost差为`6.35e-5/4.46e-5/2.66e-4 J`，远低于预注册`0.02 J`与0.2%
相对门，Actor action逐值完全相同。审计的Actor起点mean `2.7497905`与17.51的`2.7497921`只相差该执行
路径数值量级。

四轮局部数值搜索把pooled mean从`2.7497905`降至`1.8453285`，mean gain `0.9044620 J`，aggregate
residual reduction为`32.892%`；360/360状态均至少改善`1e-5 J`，paired median relative reduction为
`12.956%`，median gain `0.285055 J`。这比17.35旧Actor的36.15% headroom有所收窄，但仍远高于5%的
material门，裁决：

```text
QUERY_ACTOR_HAS_MATERIAL_HEADROOM_DIAGNOSE_ABSORPTION
```

分速度40/55/70/85/100 km/h的aggregate headroom分别为
`7.60%/8.71%/33.43%/42.32%/55.18%`；低速已相对接近局部搜索，高速仍是主要未吸收空间。数值best的
pooled P95仅`3.06318`，而Actor P95为`6.72188`，但tail仍只作诊断。best动作相对Actor的movement median
为`0.1900 sigma RMS`、mean为`0.2850`，不是仅靠数值微扰产生的虚假改进。

独立validator重载全部source，重新执行55080次Query、response fit与winner chain；除浮点重算的fit-error
约`3e-8`外，其余array/Query/metrics误差均为0，资格为：

```text
QUERY_ACTOR_CENTERED_HEADROOM_LONGRUN160_INDEPENDENT_PASS
```

不得把inner上的数值best当BC teacher或改成多头；它只证明继续改善Actor仍有价值。权威哈希：

```text
headroom equal160 config       924f0ef5702f9c6d02741842aa3ec0be525d1806d389b10f66e1b5682e5ad4f7
headroom equal160 runner       51a5d42c6c2c99028620dd6a4fb14cf1fffe8313a9e4996b406dbe27addbfd4f
headroom generic validator     84753ae2b4c5a87b1130545a9f91ba127354850e967c870b932ccba8913887ac
headroom equal160 manifest     d08aeaf0bb597da6768d35192a3cb21ef4ea09cc8438c299848eb7314932647d
headroom equal160 summary      0d7ce02f537cb0cab1466cc645957f419251472b63d1dfc6e88d78f4c4b00412
headroom equal160 validation   2ecc5d9747ef019db98075190899970c933131fc0c8f22bf0b522931053605b9
```

### 17.54 下一步冻结路由：同配方扩到320轮，不新增变量

17.51中seed0/2的selected位于160/150，17.53又确认高速度仍有33%--55%的局部Query headroom，因此允许
做一次equal-cell continuous-AC 320-round长跑；这属于检验现有Actor吸收机制随更多交互是否继续有效，
不是增加teacher数据。必须从17.50相同三个longrun selected Actor/Twin Critics重新开始，保持77-bank、
Replay、20 Critic updates/Twin、K16、Actor LR2e-5、Critic LR1e-4、cap0.02、gamma1、equal-20-cell Actor
权重、前90轮半径退火后固定0.1及逐轮完整inner选择全部不变。

前160轮必须与17.51逐seed全部schedule、online candidates及selection action/cost误差为0。320轮只有在
selected pooled mean低于`2.7497921411`、warm aggregate高于`0.4861712378`、且至少2/3 seed低于各自
17.51 selected时才promote；否则保留equal-cell160并停止纯round扩展，转向Critic locality/Actor共享
投影的吸收诊断。独立validator可复用已验证前缀，只重放round 161--320的
`3 × 160 × 20 × 77 = 739200`个suffix candidates并复算最终360个inner costs。该大实验应作为下一次
明确执行步骤，不能同时改LR/cap/K/gamma/bank、模型结构或评价集。
