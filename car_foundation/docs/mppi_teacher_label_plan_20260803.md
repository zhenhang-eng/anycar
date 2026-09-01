# MPPI sampling-center teacher 标签方案与实现状态

更新时间：2026-08-04

## 1. 为什么先做 teacher

无论后续 actor 选择 BC、TD3 还是 SAC，都需要先定义“在给定状态下，怎样的 MPPI
sampling center 更好”。teacher sidecar 是公共数据层，不绑定某一种网络或 RL 算法：

- BC 直接回归 teacher center residual，用于得到可工作的初始化；
- TD3/SAC 可用 teacher 初始化 actor，并在后续训练中加入可衰减的 behavior constraint；
- critic/ranking 网络可用同一批 candidate cost、regret 和 ESS 标签；
- 不同 cost 权重和 temperature 可从原始 feature 重新标注，不修改原 collection。

teacher 不是必须永久约束策略，也不等于最终最优解。它的第一作用是给网络一个有物理
意义、可验证的起点，并为后续 RL 提供 warm-start 基线和离线质量检查。

## 2. 分阶段计划

### T0：现有候选重标注（已实现）

对每个 snapshot 复用已经保存的 256 条 DBM rollout，按配置重新计算：

\[
C_i = w_p\sum_t e^2_{p,i,t}+w_\psi\sum_t e^2_{\psi,i,t}
      +w_v\sum_t e^2_{v,i,t}
      +w_a\sum_t\Delta a^2_{i,t}+w_\delta\sum_t\Delta\delta^2_{i,t}
\]

\[
q_i=\frac{\exp(-(C_i-C_{min})/\lambda)}
          {\sum_j\exp(-(C_j-C_{min})/\lambda)}
\]

同时保存两种监督目标：

\[
\Delta U_{best}=U_{\arg\min C_i}-U_{warm}
\]

\[
\Delta U_{soft}=\sum_iq_iU_i-U_{warm}
\]

第一版 BC 默认使用 `soft_teacher_delta_knots`，另保留
`best_teacher_delta_knots` 做消融。T0 不重新 rollout soft center，因此
`soft_weighted_candidate_cost=\sum_iq_iC_i` 只是已有候选 cost 的期望，不能当作 soft
center 自身的 DBM cost。

### T1：高预算/多中心 DBM teacher（已实现）

从每个 snapshot 恢复 state/history/reference/warm knots，围绕 warm、T0 best、T0 soft
以及多尺度扰动中心生成 center bank，再使用固定 DBM rollout。至少记录：

- center bank 的构造、随机 seed、每中心候选预算和总 rollout 预算；
- 每个中心的 best/P10/soft cost、ESS、clipping 和 boundary 指标；
- 最终 teacher center 的选择规则及相对 warm/T0 的 regret；
- cost 配置、DBM/MPPI 参数、源码 commit/hash。

T1 应真正 rollout 每个新中心，用 held-out snapshot 检查 teacher 是否稳定优于 warm。
如果不同 seed 下 teacher 方向不稳定，应保存多模态 elite，而不是强制回归单个均值。

### T2：BC 与离线验证（第一版已实现）

按 episode 切分输入与 label，训练输出 `[8,2]` bounded delta knots 的轻量网络。比较
best-label、soft-label 和必要时的多模态 label；验证 warm、network、teacher 在相同 DBM
预算下的 cost、P10、ESS、clip 和推理延迟。

### T3：TD3/SAC 与闭环

从 BC actor 初始化 TD3 或 SAC；teacher constraint 从较强逐步衰减，RL reward 使用固定
DBM 对 proposal center 的真实 rollout 结果。先做单步 contextual bandit，再决定是否
加入多步 return。通过 held-out 单步后才进行固定场景/seed 的 DBM 闭环 A/B，最后才切
Query 重新 rollout/relabel。

## 3. T0 实现

代码与配置：

```text
scripts/model_verify/dbm_teacher_cost_configs_20260803_v1.json
scripts/model_verify/generate_dbm_proposal_teacher.py
scripts/model_verify/validate_dbm_proposal_teacher.py
```

正式 sidecar：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
dbm_teacher_t0_20260803_v1
```

布局：

```text
manifest.json                 # source/config/code provenance 与局限
splits.json                   # episode-level 6/1/1 pipeline split
cost_configs.json             # 生成时使用的不可变配置副本
labels.csv                    # 每帧/每 config 的聚合标签
summary.json
COLLECTION_SUMMARY.md
episode_000/step_000250.npz   # 数组标签
episode_000/step_000250.json  # 标签语义与 source hash
...
```

原始 `fixed_dbm_train_seed_20260802_v2` 未修改；每个 sidecar 文件记录 source snapshot
SHA256，manifest 记录整个 source index fingerprint。输出目录存在时生成器会拒绝覆盖。

当前配置只包含与采集完全一致的 `collection_default`：position/yaw/vx 权重为
`5/5/1`，acceleration/steering rate 权重为 `0.05/0.1`，temperature 为 `1.0`。
配置文件支持继续添加其他组合；schema-v2 没有独立 yaw-rate error feature，因此 T0
明确拒绝非零 yaw-rate 权重。

生成和验证：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate anycar
cd /home/plusai/anycar

python scripts/model_verify/generate_dbm_proposal_teacher.py
python scripts/model_verify/validate_dbm_proposal_teacher.py \
  /disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/\
dbm_teacher_t0_20260803_v1
```

## 4. T0 结果与解释

- 96 个 snapshot、96 行 `collection_default` 标签全部通过独立复算；
- 6/1/1 split 为 `episode_000..005 / episode_006 / episode_007`，只用于打通 pipeline；
- collection cost 最大绝对复算误差 `6.57e-4`，weight 最大误差 `8.48e-7`；
- warm candidate 已是 best：`44/96`（45.8%）；
- warm-to-best regret：mean `4.3477`、median `0.4126`、P90 `11.9583`；
- ESS：mean `1.543`、median `1.351`、范围 `[1.000, 3.215]`；
- best candidate 有任意 knot clipping 的 snapshot 占 `20.8%`。

这些结果表示 T0 数据接口和 cost replay 已可靠，但标签搜索预算仍有限。低 ESS 说明
temperature=1 下权重常被极少数候选支配；44 帧 warm 已 best，说明单中心 256 候选在
不少状态没有找到更好方向；20.8% 的 best clipping 也提示部分 optimum 可能贴近当前
bank/action 边界。因此 T0 适合训练管线、critic 和初版 BC，不应被称为最终 teacher 或
性能上限；下一项应做 T1 高预算/多中心 DBM rollout。

## 5. T1 实现与预算

代码与配置：

```text
scripts/model_verify/dbm_teacher_t1_config_20260803_v1.json
scripts/model_verify/generate_dbm_multicenter_teacher.py
scripts/model_verify/validate_dbm_multicenter_teacher.py
```

正式 sidecar：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
dbm_teacher_t1_20260803_v1
```

每个状态的搜索过程：

1. 从 `warm`、`T0 best`、`T0 soft` 三个中心分别启动；
2. 使用两个 search seed，每个起点执行三轮 CEM，sigma scale 为
   `1.0 / 0.4 / 0.15`，每轮 128 条 antithetic candidates；
3. 将 21 个初始/搜索中心用 DBM 计算 direct cost，保留三个初始中心和 direct cost
   最低的中心，共 8 个 shortlist centers；
4. 对 8 个中心使用三个相同的 selection seeds，每个 seed 采样 256 条，重新 rollout
   MPPI weighted output；
5. 以 weighted-output cost mean 为主，辅以其标准差、P10、soft-min、标准化 center
   shift 和 boundary fraction 组成显式 selection score；warm 始终在 shortlist 中；
6. 再使用三个与 selection 完全不重合的 audit seeds，只比较 warm 和 teacher，不参与
   标签选择。

每帧包含 2,304 条 CEM search rollout、6,144 条 shortlist proposal probe、1,536 条
独立 audit probe，以及少量 center/weighted-output direct rollout。全量 96 帧耗时
`418.65 s`，sidecar 约 `12 MiB`。

T1 保存 center bank、shortlist、每个 probe 的 cost/weight、P10/soft-min/ESS/clipping、
weighted-output action/trajectory/cost、selection score、teacher center/delta、source/T0
hash 和全部 seed/config。validator 从保存的动作、轨迹和 reference 独立复算 direct 与
weighted-output cost、候选分布指标、score 和 teacher argmin。

## 6. T1 全量结果

96 帧全部通过 validator。使用不参与选择的 audit seeds：

- warm weighted-output cost 均值 `13.210`，teacher 为 `7.313`；
- 96/96 帧 weighted-output cost 改善；平均降低 `5.897`，中位数降低 `5.029`，平均
  相对降低约 `43.6%`；
- P10 在 86/96 帧改善，平均降低 `5.392`；
- train/validation/test 的 weighted-output cost 平均分别降低
  `5.573 / 7.367 / 6.371`，三个 split 均未出现 weighted-output 退化；
- teacher 相对 warm 的标准化 delta RMS 均值 `0.436`、中位数 `0.318`、最大 `1.105`；
- teacher 中心本身的 boundary fraction 为 0；proposal candidate element clipping
  fraction 从 warm 的 `4.54%` 降至 teacher 的 `4.37%`；
- 最终中心的搜索起点来源：T0 soft 48 帧、T0 best 29 帧、warm 19 帧；没有直接回退到
  未优化 warm 的帧。

ESS 从 warm 的平均 `1.376` 降至 teacher 的 `1.123`，表示 temperature=1 下 teacher
周围仍由少数低 cost 候选主导。它不否定 weighted output 的改善，但说明后续 BC/RL
不能把“ESS 更大”单独当成质量目标，且仍需用新 seed 和闭环验证稳定性。

当前结论是：T1 已提供比 T0 更适合作为第一版 BC 监督目标的 `teacher_delta_knots`，
但它仍是固定 DBM、单赛道、96 个状态和有限 center bank 下的近似 teacher，不是全局
最优。第一版 T2 BC 和 held-out proposal evaluator 的结果见下一节；暂不开始 TD3/SAC。

## 7. T2 轻量 BC 基线

实现：

```text
car_foundation/car_foundation/mppi_proposal_policy.py
scripts/model_verify/train_mppi_proposal_bc.py
scripts/model_verify/evaluate_mppi_proposal_bc.py
scripts/model_verify/plot_mppi_proposal_bc.py
```

网络读取 `[250,7]` history、由 `reference_ego` 构造的 `[50,5]` reference、当前
`[vx,yawrate,acceleration,steering]` 和 `[8,2]` warm knots。history/reference 分别用
三层 Conv1d 编码，current/warm 用 MLP 编码，融合后输出 `[8,2]` tanh residual，并在
加到 warm 后限制到动作边界。总参数量 `397,840`。归一化只拟合 train episodes。

使用 T1 的 6/1/1 episode split，在 1-sigma trust
`[0.25,0.35]` 和 2-sigma trust `[0.50,0.70]` 下各训练三个 seed。以 validation episode
的标准化 delta MSE 选出 `trust2_seed2.pt`；其最佳 epoch 为 11。1-sigma 与 2-sigma
最优 validation MSE 分别为 `0.35374 / 0.35262`，差异很小，说明 trust range 不是当前
主要瓶颈。

标签拟合结果：

| Split | warm/零 residual MSE | BC MSE | BC residual RMS | teacher residual RMS |
|---|---:|---:|---:|---:|
| train | 0.2391 | 0.1908 | 0.0529 | 0.1448 |
| validation | 0.3923 | 0.3526 | 0.0489 | 0.1825 |
| test | 0.2880 | 0.2921 | 0.0494 | 0.1598 |

2-sigma 网络在训练后期能把 train batch MSE 压到约 `0.004`，所以 39.8 万参数足以记住
当前 72 个训练状态；但 validation 很早达到最佳并随后退化。选出的网络 residual RMS
只有 teacher 的约三分之一，明显收缩回 warm start。test 标签 MSE 还略差于零 residual。
因此当前首先暴露的是状态覆盖不足以及逐帧 teacher 标签的多解/随机性，而不是已证明的
网络容量上限。

另用完全不参与 T1 搜索或 audit 的 DBM seeds `14001/14002/14003`，对 warm、network、
teacher 在每帧使用相同 `256` candidates 做 proposal rollout：

| Split | warm weighted cost | network | teacher | network 相对 warm | network 胜/负 |
|---|---:|---:|---:|---:|---:|
| train (72) | 13.141 | 11.778 | 7.481 | 8.27% | 56/16 |
| validation (12) | 13.591 | 12.120 | 7.011 | 9.43% | 8/4 |
| test (12) | 12.522 | 12.153 | 7.494 | 1.25% | 6/6 |
| all (96) | 13.120 | 11.868 | 7.424 | 7.54% | 70/26 |

test 上 weighted-output cost 有小幅正收益，但样本只有 12 帧且胜负各半；P10 从
`54.400` 变为 `54.552`，没有改善，direct-center cost 也从 `18.559` 退化到 `19.554`。
因此这版 BC 可作为接线和训练基线，尚不能称为可靠的 proposal policy。batch-1 前向
延迟约为 CPU `0.65 ms`、RTX 3080 Ti CUDA `0.33 ms`，计算量不是当前问题。

正式输出：

```text
outputs/mppi_proposal/bc_t1_conv_v1/
  training_summary.json
  trust1_seed{0,1,2}.pt
  trust2_seed{0,1,2}.pt
  offline_dbm_eval_v1/summary.json
  offline_dbm_eval_v1/per_snapshot.csv
  offline_dbm_eval_v1/evaluation_arrays.npz
  offline_dbm_eval_v1/bc_t1_result.png
```

下一步不应直接用更大网络覆盖问题。先增加独立 episode、速度/误差/曲率/恢复状态覆盖，
并保存同一状态的多个稳定 elite center 或改为 cost-aware ranking/critic，避免 MSE 对
多解标签取均值后退回 warm。数据扩充后再用相同 split/evaluation seeds 做小网络与更大
temporal encoder 的受控消融；只有 held-out 单步稳定改善后才进入 DBM 闭环 A/B。

## 8. 2026-08-04 数据扩充状态

多速度采集入口与修正后的 pilot 已完成。有效 pilot
`fixed_dbm_expansion_pilot_20260804_v2` 覆盖 `1.4/2.0/2.6 m/s`，包含 3 个 episode、60
帧，全部通过 validator。旧 v1 pilot 的速度列和参考 x/y 推进不一致，已标记 rejected，
不得生成 teacher。

正式场景表 `fixed_dbm_policy_expansion_20260804_v3.json` 的 30 个 episode 已全部采集并
逐场通过 validator，共 600 帧、153,600 条 candidates 和 14,370 条连续 trace；冻结
split 为 18/6/6，每个 split 均包含全部三档速度。实际 `vx=1.053--3.092 m/s`，横向误差
`-0.291--0.269 m`，航向误差 `-0.391--0.301 rad`；绝对航向误差中位数/P95 为
`0.066/0.215 rad`，没有超过 `0.40 rad` 的 snapshot。

上述 T0/T1、固定结构重训和公平复评已完成，结果见下一节。

## 9. 扩充数据 T0/T1 与固定结构 BC 结果

### 9.1 新 sidecar 与校验

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
  dbm_teacher_t0_expansion_20260804_v2/
  dbm_teacher_t1_expansion_20260804_v1/
```

T0 共 600 帧，全部通过 cost/weight replay；最大 cost 绝对误差为
`9.84e-4`，最大 weight 绝对误差为 `8.23e-7`。warm candidate 为 best 的比例为
315/600，warm regret 的均值/中位/P90 为 `4.894/0/14.553`，ESS 均值为 `1.489`。

T1 沿用第 5 节的全部搜索、selection、audit 参数。结果：

- selection score 600/600 改善，平均/中位/P10/P90 为
  `6.035/4.734/1.685/11.889`；
- selection weighted-output cost 600/600 改善，平均/中位/P10/P90 为
  `4.938/3.815/1.401/9.659`；
- disjoint audit seeds 上 592/600 改善，平均/中位/P10/P90 为
  `4.711/3.687/1.238/8.776`；
- teacher delta standardized RMS 的均值/中位/最大为 `0.401/0.287/1.471`；
- T0-soft 起点产生最终中心 301 帧，warm 起点 219 帧，T0-best 起点 78 帧；另有 2 帧
  三个起点收敛到同一中心，1 帧直接选择 `t0_soft`；多起点搜索仍具有互补性；
- 598 帧保存 8 个 shortlist centers，2 帧因完全重复中心去重后保存 7 个。proposal
  数组按实际 center 数保存，validator 允许 `1..configured_shortlist_count`，但仍严格
  校验所有必需起点、shape、cost、weight、score 和 teacher argmin。

### 9.2 公平的新旧 checkpoint 对比

先用旧 checkpoint 在新 test 的 120 帧上评估，再训练完全相同的 Conv1d+MLP。网络仍为
397,840 参数、`[8,2]` bounded residual，1/2-sigma trust 各 3 个 seed，loss、optimizer
和 early stopping 不变。所有 proposal 评估使用相同 fresh seeds
`14001/14002/14003` 和每中心每 seed 256 candidates。

| Checkpoint / 新 test | warm | network | teacher | network-warm | 胜/负 |
|---|---:|---:|---:|---:|---:|
| 旧 `bc_t1_conv_v1/trust2_seed2.pt` | 11.760 | 11.776 | 7.174 | -0.016 | 56/64 |
| 新 `bc_t1_conv_expansion_20260804_v1/trust1_seed1.pt` | 11.760 | 11.128 | 7.174 | +0.632 | 73/47 |

新网络相对 warm 的 P10 cost 平均改善 `1.502`，82/120 帧改善；weighted-output cost
相对改善均值约 `4.91%`。新 checkpoint 回放旧 12 帧 test 时，warm/network 为
`12.522/11.827`，平均改善 `0.695`，没有观察到明显旧分布回归，但 12 帧只作为 smoke
regression 使用。

新网络 test normalized delta RMSE 为 `0.426`，比旧实验的约 `0.567` 更好；但 test
prediction delta RMS 仍只有 `0.041`，teacher 为 `0.133`。teacher 在新 test 对 warm
可提供平均 `4.586` 的 weighted-output cost 改善，新网络只实现 `0.632`，约 14%。这说明
数据扩充有效，但单目标 MSE 仍把多解或不稳定方向平均为小 residual；不能据此把容量
作为首要瓶颈。

完整输出和图：

```text
/home/plusai/anycar/outputs/mppi_proposal/
bc_t1_conv_expansion_20260804_v1/
  training_summary.json
  old_checkpoint_new_test_eval/summary.json
  new_checkpoint_new_test_eval/summary.json
  new_checkpoint_new_test_eval/bc_t1_expansion_result.png
  new_checkpoint_old_test_eval/summary.json
```

### 9.3 2026-08-05 120-episode teacher 与 1800-state BC

#### 标签预算与验证

当前首选 raw collection、T0 和 T1 为：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/
  fixed_dbm_policy_diverse_20260805_v1/
  labels/dbm_teacher_t0_diverse_20260805_v1/
  labels/dbm_teacher_t1_diverse_20260805_v1/
```

T0 复用每帧 64 个 collection candidates；2400/2400 标签通过 source hash、candidate
cost 和 weight replay，最大 cost/weight 误差为 `1.176e-3/4.308e-6`。warm candidate
在 1287/2400 帧为 best；warm-to-best regret 均值 `5.972`。

T1 配置为 3 个起点、2 search seeds、3 个 sigma stages、每 stage 64 samples；shortlist
最多 5 个中心，selection 和 audit 各 2 seeds × 64 candidates。总预算为每状态 2048
rollouts，较旧 9984 减少 79.5%；2400 状态总预算约 4.92M，仍低于旧 600 状态约
5.99M。全量严格 validator 已通过。结果：

- selection score 2398/2400 改善，均值/中位 `10.758/7.913`；
- selection weighted-output cost 2398/2400 改善，均值/中位 `8.980/6.635`；
- disjoint audit seeds 上 2392/2400 改善，均值/中位 `8.896/6.845`，P10/P90
  `2.273/17.777`；
- audit P10 cost 2081/2400 改善，均值/中位 `7.567/4.982`；
- teacher delta standardized RMS 均值/中位/最大 `0.456/0.349/1.372`；
- 仅 2 帧保留未优化 warm，其余由 warm/T0-best/T0-soft 多起点 CEM 产生。

validator 原先要求重算 score 的 float64 `argmin` 索引与生成时 float32 `argmin` 完全
相同；一帧两个近重复中心只差 `3.16e-7` 而索引翻转。现改为在既定 `3e-4` 数值容差
内接受并列最小值，同时仍严格验证 teacher center/source/actions/trajectory 与所存索引
一致。

#### 固定网络重训与公平对比

保持 397,840 参数 Conv1d+MLP、输入、bounded `[8,2]` residual、optimizer 和 trust
配置不变，用 1800/300/300 状态训练 1/2-sigma trust × 3 seeds：

```text
outputs/mppi_proposal/bc_t1_conv_diverse_20260805_v1/
```

validation 选中 `trust1_seed1.pt`，best epoch 9；其 train/validation/test normalized
delta MSE 为 `0.2235/0.2228/0.2284`。六个 run 的 best epoch 均为 5--9，继续训练到
350-epoch patience 后 validation 没有恢复；训练集和 heldout MSE 已非常接近，因此上一
版“小训练集记忆/泛化差距”明显减轻，但逐帧 teacher 映射仍有较强不可预测成分。

用全新 seeds `20001/20002/20003`、test 300 帧、每 center/seed 64 candidates 做固定
DBM proposal 评估：

| checkpoint | warm | network | teacher | network-warm | 胜/负 |
|---|---:|---:|---:|---:|---:|
| 旧 360-state `bc_t1_conv_expansion_20260804_v1/trust1_seed1.pt` | 21.622 | 20.296 | 12.136 | +1.326 | 201/99 |
| 新 1800-state `bc_t1_conv_diverse_20260805_v1/trust1_seed1.pt` | 21.622 | 18.611 | 12.136 | +3.011 | 234/66 |

paired 旧→新 weighted-output cost 平均降低 `1.685`（旧网络的 8.30%），中位降低
`1.332`，211/300 帧新网络更好。best/P10/softmin/median/direct/selection-score 均值也
分别降低 `1.804/2.356/1.802/7.126/3.608/2.115`。

该改善不是单一子集造成：五档速度的旧→新平均收益为
`0.904/1.767/1.862/1.806/2.086`；heldout nominal/mixed/recovery 为
`1.038/1.581/2.437`，各组都为正。新网络相对 warm 的 recovery 胜率为 84/100。

结论：增加独立场景和 teacher 状态有效，新网络实现 teacher 相对 warm 可用收益的约
`3.011/9.486=31.7%`，显著高于上一版约 14%，但仍有 66/300 退化帧且与 teacher 相差
`6.475`。当前不应再把“继续堆同类帧”或“K=2 在线均匀采样”作为默认主线。下一步先
把新 checkpoint 接入固定 DBM 闭环 A/B；若闭环收益稳定，再对网络退化/分布漂移状态做
DAgger 式采集和 cost-aware 重标注，并研究直接预测中心质量/排序而非只做 teacher MSE。

完整 fresh-seed 输出：

```text
outputs/mppi_proposal/bc_t1_conv_diverse_20260805_v1/
  fresh_test_eval/{summary.json,per_snapshot.csv,evaluation_arrays.npz}
  old_checkpoint_fresh_test_eval/{summary.json,per_snapshot.csv,evaluation_arrays.npz}
```

#### BC 初始化准入结论与下一训练目标

当前 BC 已足够作为 actor 初始化，不需要把参数回归误差压到 teacher 附近才开始
reward/cost 微调。fresh test 的 66 个退化状态中，34/12/1 帧分别比 warm 差超过
2/5/10 cost，最坏差 `17.680`；因此下一阶段仍需 BC anchor、center trust region 和
warm fallback，不能让 actor 无约束利用 Critic 外推误差。

第一版 Critic 直接复用 T1 `shortlist_centers` 和
`proposal_weighted_output_cost.mean(axis=seed)`，以 episode split 训练
`Q(s, center)`。主要准入指标不是只看全局 MSE，而是：同状态 pairwise ranking
accuracy、每状态 argmin top-1、选择 regret、warm-vs-teacher 判别，以及 test cost
Spearman。Critic 通过后 actor 目标采用 predicted cost + teacher BC penalty + boundary/
trust penalty；新 actor center 必须由固定 DBM 真值重算后才能加入下一轮训练。

第一版 Critic 已完成，但只通过了 shortlist 内的门槛。3-seed ensemble 在 held-out
shortlist 上 pairwise accuracy `0.801`、选择 cost `20.865 -> 13.406`；在 fresh seeds
20001--20003 的 warm/BC/teacher 三中心测试上 pairwise 为 `0.757`、选择 cost
`21.622 -> 15.271`。关键未通过项是 teacher-vs-BC 排序仅 `0.503`。这说明当前 T1
每状态 5 个离散 shortlist center 对连续 actor 输出区域覆盖不足，而不是应立即增加
SAC/TD3 训练步数。

下一标签阶段定义为局部 proposal relabel：每状态保存 warm、当前 BC、T1 teacher、
BC--teacher 插值和 BC 邻域扰动中心，全部用 common fixed-DBM rollout seeds 评估，且
validation/test 使用不相交 audit seeds。它的目的不是重做 teacher 搜索，而是补足
`Q(s, center)` 在 actor 可能访问的局部动作分布；只有独立 seed 的 teacher-vs-BC 与
局部 pairwise 排序通过后，才冻结 Critic 并做小步 actor 优化。

该阶段现已完成。局部 sidecar 为 `dbm_critic_local_diverse_20260805_v1`，2400 帧每帧
11 center，selection/audit 各 2 seeds，并通过独立 validator。Critic v2 的 audit-test
teacher-vs-BC 排序达到 `0.917`，但直接 Critic-gradient actor 在真实 DBM validation
全部严重退化，证明排序模型的离散准确率不能作为连续动作梯度准入条件。

支持集内 AWR 是当前唯一通过 mean-cost 复评的 actor 提取方法。temperature=2 在
validation 相对 BC 改善 `0.140`，在全新 test seeds 上改善 `0.084`（161/139），但尾部
未改善，尚不替换默认 BC。下一轮标签不再均匀扩场景，而应优先覆盖 BC/AWR 分歧、AWR
退化和 Critic 误排序状态，训练 pairwise delta-cost/risk head；在可靠 gate 形成前不做
无约束 actor gradient、SAC 或 TD3。

### 9.4 固定 2400 状态的全秩局部 reward sidecar

在决定继续采集 hard states 前，先确认原 11-center 局部数据每状态只有约 4 个独立动作
方向，不能识别 16 维 center 的完整局部梯度。因此不增加任何新场景或 snapshot，复用
`fixed_dbm_policy_diverse_20260805_v1` 和原 episode split，生成：

```text
labels/dbm_critic_fullrank_diverse_20260805_v1
```

每状态固定冻结 BC 为 base，用 16 个正交 Hadamard directions 的 `±0.15 sigma` 中心和
base 构成 33-center bank。selection/audit 各 4 个互不重叠 seeds，每 center/seed 64
candidates，总 fixed-DBM candidate rollout 数为 40,550,400。训练标签仍是 MPPI
weighted-output cost（reward 为其相反数），不是 teacher 筛选结果，也不读取 DBM 梯度，
所以同样的数据形式可以由 Query rollout relabel。

独立验证确认 2400/2400 状态 rank=16，平均 clipping `0.423%`。selection/audit 的方向
符号一致率为 `75.74%`（忽略两侧差异不超过 0.1 的近 tie）；方向差绝对值超过
`1/2/4` 时一致率提高为 `81.33%/84.65%/88.42%`。这说明原来的局部信息缺失已补齐，
但训练时仍需显式处理 Monte-Carlo label noise。下一 Critic 应以 paired delta-cost、
seed-mean 与 uncertainty weighting 训练，并用 audit 方向梯度一致性作为 actor-gradient
准入指标；audit seeds 不得混入训练。

#### 全秩 Critic 训练结论

已完成通用 MLP 与显式 local-quadratic 两种 3-seed Critic。训练均使用 selection reward
的 paired delta-cost，并对 33-center 满秩设计最小二乘得到的 16 维局部斜率加噪声权重；
audit labels 从未用于 optimizer 或 checkpoint 选择。

test 上 selection-gradient 对 audit-gradient 的 cosine 中位为 `0.617`，代表当前
4-seed 对 4-seed 标签可重复上限。通用 Critic 只有 `0.166`，结构化 Critic只有
`0.193`；绝对 audit gradient 大于 1 的分量符号准确率分别为 `55.51%/56.95%`，而数据
上限为 `72.08%`。结构化模型按预测从 33 个中心中选择会相对 base 平均退化 `3.322`；
通用模型仅约 `+0.003`，没有实际收益。两个模型都不准入 actor 更新，原 BC 不变。

失败不是因为 16 维方向仍缺失，而是 state-only 网络没有从 1800 个状态泛化出快速变化
的局部 reward 斜率：同一状态跨 seed 的 cosine 中位约 `0.630`，相邻保存帧交叉 cosine
中位仅 `0.044`。下一轮若继续 state-only RL，需要更多独立状态和更低噪声的 seed-mean；
更优先考虑把第一次 MPPI 的 candidate cost/error/trajectory response 压缩为网络输入，
由当前帧采样反馈引导第二次 center。不能因为已生成 full-rank 标签就直接进入 SAC/TD3。

### 9.5 历史下一步（已由 2026-08-05 状态覆盖）

暂不直接增大 encoder，也不进入 TD3/SAC。先从已保存的 T1 pool/shortlist 和 held-out
proposal cost 构造多中心监督：每个状态保留多个稳定 elite center，再训练一个共享
encoder 的 center cost/ranking critic，或使用 K 个 bounded residual heads 加 critic
选择。受控比较单头 BC、K-head oracle 和 K-head+critic；仍固定 DBM、split、cost、
candidate 数和 fresh seeds。只有 held-out 单步 cost 与胜率进一步稳定改善后，才做
fixed-DBM 闭环 A/B 和 DAgger 式失败状态追加。

## 10. Multi-elite 标签与 oracle 诊断

2026-08-04 已从扩充 T1 shortlist 生成独立 multi-elite sidecar，不修改 raw collection
或 T1：

```text
/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/
dbm_multi_elite_expansion_20260804_v1
```

筛选规则冻结在
`scripts/model_verify/dbm_multi_elite_config_20260804_v1.json`：T1 teacher 固定为第 0
个 elite；其他中心必须至少在 2/3 个 selection seed 上优于 warm，mean weighted-output
cost 不超过 teacher `+2.0`，并与已选中心保持至少 `0.20 sigma` 的标准化 RMS 距离。
最多保存 4 个中心，不足部分复制 warm 但必须由 `elite_valid_mask` 屏蔽。

600 帧全部通过 source/T1 SHA256、筛选顺序、cost、距离、shape 和 padding validator。
有效中心数分布为：1/2/3/4 个分别 `106/300/114/80` 帧，平均 `2.28`；494/600 至少
两个，194/600 至少三个。test 为 120 帧，其中 97 帧至少两个、42 帧至少三个、19 帧
有四个。这确认 T1 bank 中广泛存在多个方向不同且 selection cost 良好的中心。

oracle 使用完全未参与 T1 selection/audit 或先前 BC 评估的 DBM seeds
`15001/15002/15003`，只评估冻结 test 120 帧。实现和结果：

```text
scripts/model_verify/analyze_dbm_multicenter_oracle.py
scripts/model_verify/plot_dbm_multicenter_oracle.py

outputs/mppi_proposal/dbm_multi_elite_oracle_20260804_v2/
  summary.json
  per_snapshot.csv
  oracle_arrays.npz
  multicenter_oracle_result.png
```

主要结果：

| 方法 | 全 test mean weighted-output cost | 相对单 T1 teacher |
|---|---:|---:|
| warm | 11.660 | -4.495 |
| 单 T1 teacher / K=1 | 7.165 | 0 |
| K=2 center 算术平均 | 7.382 | -0.217 |
| K=3 center 算术平均 | 7.630 | -0.465 |
| K=2 full-budget state oracle | 7.139 | +0.026 |
| K=3 full-budget state oracle | 7.131 | +0.034 |
| K=2 固定总 256 mixture | 7.383 | -0.218 |
| K=3 固定总 256 mixture | 7.269 | -0.104 |

只看确实有至少 K 个 elite 的状态，center 算术平均相对 teacher 的损失为：K=2
`0.269`（97 帧）、K=3 `1.068`（42 帧）、K=4 `0.595`（19 帧）；完美 full-budget
selector 的额外收益仅为 `0.032/0.075/0.104`。因此多个 mode 之间做中心平均确实会落入
高 cost 区域，但单 T1 teacher 已捕获几乎全部可用中心选择收益；把总 256 candidates
均匀分给多个中心并全局 soft-weight 也没有改善。

另将 T1 teacher residual 缩到 warm→teacher 距离的 25%/50%/75%，fresh-seed cost 为
`10.678/9.281/8.036`，完整 teacher 为 `7.165`；120 帧中分别 `0/2/6` 帧优于完整
teacher。这与 BC test residual RMS `0.041`、teacher `0.133` 的明显收缩一致。

当前结论需要区分两件事：

1. **训练表示层面存在多模态/非凸平均问题。** K-head、winner-take-all 或 mode
   classification 可能帮助网络避免回归到中心平均，目标是重新接近单 T1 teacher。
2. **在线 MPPI 不需要简单混合多个中心。** 现有 oracle 没有显示 K-center mixture 或
   perfect selector 能显著超过单 teacher；因此不能把多中心解释成新的巨大性能上限。

下一实验若训练 K-head，应先做 K=2 小规模受控实验，使用 mask/set loss，并让 gating
选择一个 head，不能把多个 head 再求平均或把候选简单均匀混合。通过标准是相对当前
单头 BC 明显追回 teacher gap，同时保持 fresh-seed 退化帧下降；暂不使用 Diffusion。
