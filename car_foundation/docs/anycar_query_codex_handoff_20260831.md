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
