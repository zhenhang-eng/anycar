# MPPI 在线 Actor--Critic Pilot 计划（2026-08-20）

> 状态：`HIGHSPEED_TRUST006_MECHANISM_LEAD_CAP_ONLY_DECAY_NEXT`
>
> 目标：在固定DBM、train-only状态上验证“Actor产生新动作→DBM返回真实cost→Replay持续扩展→
> Critic多次更新→Actor低频更新”的闭环学习，判断持续交互能否修复离线Critic在actor-visited
> 分布上的方向尾部，并提升one-shot absolute-action Actor。
>
> 2026-08-30路线覆盖：§54--55的search-path蒸馏只证明Actor可表达性和标签质量，不替代本文件的
> 持续Actor--Critic主合同。最终切换Query/PyTorch或Query/ONNX时，动力学后端只需返回候选轨迹与
> 标量cost；Actor更新仍必须来自持续训练的Twin Critic。DBM解析梯度只作机制oracle。

## 1. 永久关闭的方案

以下两条路线标记为`PERMANENTLY_FROZEN`，不得通过改名、换loss、换坐标或增加训练epoch重新开启：

1. **冻结Critic更新Actor**：用一个离线训练完成后不再接收新真实reward的Critic，连续或单次更新
   Actor；
2. **单次Critic梯度更新Actor**：每个状态只从Critic取一次`dQ/da`，不对新Actor动作查询
   DBM/Query真实cost、不写Replay、不继续更新Critic。

关闭依据：两类方法都假设离线Critic在Actor访问的新action上已经具有可靠局部导数。§11.83.6
证明warm尾部是真实大梯度下的Critic符号错误，而bank-best尾部是近零真梯度下的虚假高norm；
单纯缩步只能减小损失，不能补充缺失监督。以后若在线pilot失败，路线回到Search/ranking与
two-center guard，禁止退回上述方案。

## 2. 问题定义

本阶段仍是单步固定状态优化，不推进车辆状态：

```text
s -> Actor -> a_abs -> frozen DBM -> J(s,a_abs)
```

所以它是SAC-style的持续交互contextual bandit，而不是带Bellman bootstrap的车辆多步SAC：

```text
Q*(s,a) = -log(1 + J_direct(s,a))
```

不需要`next_state`、target Critic或时序bootstrap。若以后进入车辆状态递推闭环，那是独立阶段，
不得把本pilot的结论直接外推过去。

## 3. 固定输入输出合同

### Actor

- 输入：当前clean/no-anchor状态合同；不输入warm center、first-pass feedback、gradient context；
- 输出：完整absolute `8x2` knots，通道顺序`[acceleration, steering]`；
- 训练分布：tanh Gaussian Actor输出mean与log-std；
- 部署候选：只使用Actor mean一次前向，Critic和随机探索不进入部署；
- warm仍可作为MPPI候选集里的独立中心，但不是Actor输入。

### Twin Critic

- 输入：`state + candidate absolute action`；
- 输出：标量cost `C_i(s,a)=log(1+J_direct)`；
- 两个Critic独立初始化；Actor使用`max(C1,C2)`作为保守cost；
- Critic主要以真实value、same-state ranking和actor-visited新样本训练，不以解析DBM梯度为输入；
- flat/stay单独建头，不再用强stationarity把共享scalar-Q整体压平。

Actor目标为：

```text
L_actor = E[max(C1,C2) + alpha * log pi(a|s)]
          + beta_KL * KL(pi_old || pi)
```

其中KL只约束相邻训练迭代的策略变化，不引入warm/anchor动作。Actor梯度不得归一化成固定长度。

## 4. Replay与在线采样

### 初始Replay

使用已验证的43200条train-only absolute-action bank：每状态8个raw、8个v1 optimized、8个v2
refined候选。每个outer fold只能读取该fold的训练episode；heldout episode只评价不训练。

### 每轮新增交互

从训练episode按`speed x scenario`均衡抽256个状态。每状态默认查询6个动作：

1. Actor deterministic mean；
2. 两对Actor-centered antithetic stochastic动作；
3. 一个较宽但有界的exploration动作。

合计1536次DBM rollout/轮。所有动作都记录：好动作、坏动作、被clip动作、Critic分歧动作均不得
丢弃。探索std从现有MPPI sigma的0.25倍起步，根据actor-visited覆盖与实际gain衰减；不使用
随机seed改变Actor mean的语义。

### Replay抽样

每个Critic batch按状态平衡，建议初始比例：

- 50% recent actor-visited；
- 25%历史absolute candidate bank；
- 25% hard-priority，包括真实回归、Twin Critic分歧和高速状态。

raw/warm/near-best anchor样本设不可淘汰保留位，避免Replay扩张后失去value尺度与flat参考。

## 5. 执行阶段

### OAC-0：合同与基线冻结

1. 固化episode-grouped 3-fold；机制pilot先跑fold 0 × 3 seed，通过后再扩3-fold × 3-seed；
2. 为每fold训练两套full-train Twin Critic，初始化自§11.83 V0 absolute value/ranking合同；
3. 固化当前最佳clean/no-anchor absolute Actor起点、DBM/weights/MPPI参数与全部SHA256；
4. 建立固定internal OOF evaluator，报告真实DBM而非Critic预测；
5. formal validation/test继续封存。

### OAC-1：Actor冻结的在线Critic burn-in

- 10轮在线采样，共约15360条新actor-visited transition；
- Actor参数冻结；
- 每轮20次Critic update；
- 监控actor-visited value Pearson、same-state pair accuracy、Twin差异、warm方向、flat/stay与高速层。

进入Actor阶段的门：

1. actor-visited material-pair accuracy ≥0.85；
2. 两Critic均无value/norm坍缩；
3. flat/stay在bank-best recall ≥0.80，warm false-stay ≤0.10；
4. Critic对本轮新增坏动作的排序在后续两轮得到修正，而不是只记住旧bank。

### OAC-2：持续Actor--Critic循环

- 最多100轮；
- 每轮先采真实DBM数据，再做20次Critic update、1次Actor update；
- Actor使用非归一化梯度、tanh动作边界、KL trust和独立flat/stay门；
- Critic LR高于Actor LR，Actor update ratio固定为`1:20`，pilot期间不扫比例；
- 每5轮在固定internal OOF状态上做真实DBM评估并保存shadow/latest/selected三套Actor。

训练中的坏动作必须进入Replay。所谓accept/reject只用于**Actor checkpoint晋级**：exploration/
latest Actor可以暂时变差，selected Actor只有在真实DBM evaluator通过时才替换；连续发散可从
selected恢复，但不能删除导致发散的Replay样本。

### OAC-3：内部episode-heldout判定

所有指标以真实DBM cost为准，至少报告：

- Actor mean相对起点、warm与bank-best的聚合headroom recovery；
- episode-bootstrap 95% CI、mean/median、P05/worst；
- 1.2/1.6/2.0/2.4/2.8m/s与六类scenario；
- stay误动率、mover漏动率、动作饱和率、相邻knot平滑性；
- actor-visited Critic ranking、Twin分歧及错误动作被后续修正的轮数；
- selected/latest之间的真实cost差，禁止只报selected。

fold 0机制门：3 seed中至少2个满足：

1. selected Actor聚合真实gain >0且episode-bootstrap CI下界 >0；
2. median gain >0；
3. 2.4和2.8m/s层聚合gain均非负；
4. actor-visited material-pair accuracy保持≥0.85；
5. 无NaN、动作全维饱和或Replay/label合同错误。

机制门通过才扩完整3-fold×3-seed。正式内部门还要求：至少2/3 seed、至少2/3 fold通过，且
two-center guarded P05≥0、worst无回归。direct Actor的P05/worst仍必须原样报告，不得用guard
遮蔽训练不稳定。

### OAC-4：短闭环传导

只有OAC-3通过后，才使用固定DBM nominal/high-speed/recovery三场景做短闭环A/B：

```text
warm-only
vs
warm + online-AC Actor two-center candidates
```

报告累计cost、失败率、控制抖动、warm-return次数和MPPI ESS。Critic不进入在线部署图。

## 6. 停止条件与失败路由

出现任一条件即停止当前run并保留完整artifact：

1. 连续20轮selected Actor真实DBM指标无改善；
2. 连续两次评估actor-visited ranking <0.80；
3. Actor动作饱和率>10%、出现NaN或真实cost显著发散；
4. 三个pilot seed中少于两个得到正聚合gain；
5. Critic只提高训练bank指标、actor-visited新动作不改善。

失败后允许：扩大actor-visited采样、调整状态平衡、改独立flat/stay头，或回Search/ranking主线。
失败后禁止：恢复冻结Critic、单次梯度Actor、只加epoch、重新扫描旧gradient loss/坐标/Hessian。

## 7. 预期产物

```text
scripts/model_verify/train_mppi_online_absolute_sac.py
scripts/model_verify/validate_mppi_online_absolute_sac.py
outputs/mppi_proposal/online_absolute_sac_<date>_v1/
  contract.json
  replay_manifest.json
  iteration_metrics.jsonl
  checkpoints/{shadow,latest,selected}/
  actor_visited_replay.npz
  internal_oof_summary.json
  validator_report.json
```

每轮必须保存Actor/Critic optimizer、temperature、Replay游标、RNG状态和source hashes，支持精确
断点恢复。任何“selected最好结果”必须同时附带完整latest轨迹，防止selection偏差。

## 8. 当前授权边界

当前只授权实现OAC-0和OAC-1，并在fold 0 × 3 seed完成Critic burn-in门。Actor仍保持冻结，直到
OAC-1四项门全部通过。该授权不打开formal validation/test，不授权ROS/实车部署，也不改变当前
two-center guard默认配置。

## 9. OAC-0/OAC-1执行结果（2026-08-20）

已按本合同完成fold 0 × 3 seed。每seed执行10轮、每轮256状态×6动作，共15360条新增
actor-visited样本；三seed合计46080次固定DBM评价。每轮20次Twin Critic update，Actor optimizer
不存在，三个Actor的运行前后module SHA256逐项相同。formal validation/test未加载。

OAC-0通过：episode fold、输入/输出、DBM/weights、Actor/Critic checkpoint、归一化来源和source
hash均已冻结；独立validator对每seed抽查96条DBM replay，最大绝对误差均为0，并复算全部summary
指标与gate完全一致。

OAC-1联合门为`0/3 seed`，所以**禁止进入OAC-2**：

| 指标 | seed 0 | seed 1 | seed 2 | 门 |
| --- | ---: | ---: | ---: | ---: |
| actor-visited material-pair accuracy | 0.877 | 0.876 | 0.880 | ≥0.85，均通过 |
| actor-visited Pearson（两Critic范围） | 0.944--0.947 | 0.939--0.943 | 0.943--0.946 | 无坍缩，均通过 |
| flat bank-best recall，阈值0.5 | 0.779 | 0.811 | 0.751 | ≥0.80，仅1/3通过 |
| warm false-stay，阈值0.5 | 0.246 | 0.196 | 0.275 | ≤0.10，0/3通过 |
| lag-2坏动作总体排序 | 0.937 | 0.941 | 0.933 | ≥0.85，均通过 |
| 初始排错坏动作的最终纠正率 | 0.370 | 0.446 | 0.332 | ≥0.50，0/3通过 |

计划原文未固定flat概率阈值，因此保留0.5原结果后，又只用fold-train选择“warm false-stay≤0.10
时recall最大”的阈值，再到fold-0 heldout检验。heldout recall仍只有`0.537/0.565/0.320`，
false-stay为`0.102/0.092/0.068`；所以flat失败不是阈值校准问题。

OAC-0 heldout bank evaluator也已补齐。在线适配后material-pair accuracy仍为
`0.942/0.949/0.937`，但相对初始化下降`0.021/0.017/0.025`；actor-visited的2.8m/s层只有
`0.825/0.826/0.825`，低于总体门。这说明主体value/ranking成立，但出现轻微旧bank遗忘和明确
高速薄弱层，不能据总体均值提前解冻Actor。

下一步若继续，只授权`OAC-1B`且Actor继续冻结：

1. flat/stay改为以local value span/成对bank-best--material-warm为目标的独立头，不再把不校准
   的普通候选BCE直接当部署stay语义；
2. Replay hard部分显式加入“最初排错且两轮后仍未纠正”的actor-visited pair与2.8m/s层；
3. 提高历史bank rehearsal或加不退化约束，要求heldout bank不再继续下降；
4. 沿用同一fold和原gate，不降低`0.80/0.10/0.50`门，不以更多epoch直接重开OAC-2。

复现链：

```text
scripts/model_verify/train_mppi_online_absolute_sac.py
scripts/model_verify/validate_mppi_online_absolute_sac.py
scripts/model_verify/analyze_mppi_online_absolute_sac_oac1.py
outputs/mppi_proposal/online_absolute_sac_oac01_20260820_v1/
```

## 10. 固定Replay Critic学习曲线（2026-08-20）

按§9后续第一优先级，只使用OAC-1每seed固定的15360条Replay继续训练；不新增DBM rollout、不
构造Actor模块或optimizer、不修改value/ranking loss、flat标签或Replay采样配比。检查点为总
update标签`200/400/800/1600`。父OAC-1未保存optimizer state，因此200步后以相同LR重新初始化
AdamW；本实验可判断“额外优化是否有用”，但不是原optimizer轨迹的bitwise续训。

核心曲线：

| updates | actor-visited pair acc（3 seed） | 初始排错纠正率 | 2.8m/s pair acc | heldout bank pair acc |
| ---: | --- | --- | --- | --- |
| 200 | 0.877/0.876/0.880 | 0.370/0.446/0.332 | 0.825/0.826/0.825 | 0.942/0.949/0.937 |
| 400 | 0.894/0.895/0.895 | 0.447/0.509/0.435 | 0.852/0.846/0.844 | 0.944/0.945/0.941 |
| 800 | 0.904/0.907/0.909 | 0.572/0.617/0.529 | 0.867/0.865/0.869 | 0.946/0.949/0.946 |
| 1600 | 0.918/0.924/0.923 | 0.634/0.666/0.584 | 0.881/0.885/0.892 | 0.947/0.949/0.944 |

所以原“纠正率<0.50”和2.8m/s<0.85主要是训练预算不足：800步后三seed均过这两个门；1600步
进一步提高。heldout bank从200到1600还改善`+0.005/+0.001/+0.007`，没有继续遗忘。

不修改flat目标的情况下，flat也随训练显著改善。1600步、原0.5阈值的全1800状态口径中seed 0/2
通过完整联合gate，seed 1因warm false-stay=`0.114`未过；即旧实现会得到2/3机制通过。但更严格
的fold-train阈值选择→fold-0 heldout结果为：recall=`0.935/0.940/0.962`，false-stay=
`0.100/0.105/0.117`，严格仅1/3通过。为避免用训练状态掩盖边界误差，Actor仍冻结，OAC-2仍不
启动。第二步“是否修改flat监督”留待单独讨论，不能由本轮自动授权。

本轮qualification为`MORE_CRITIC_OPTIMIZATION_CAN_FIX_ERROR_CORRECTION`。独立validator确认
12个checkpoint点全部零误差复算，Replay hash未变，Actor update=0，formal validation/test封存。

```text
scripts/model_verify/run_mppi_oac1_fixed_replay_curve.py
scripts/model_verify/validate_mppi_oac1_fixed_replay_curve.py
outputs/mppi_proposal/online_absolute_sac_oac1_fixed_replay_curve_20260820_v1/
```

## 11. 固定Replay扩展训练到6400步（2026-08-20）

在§10之后继续使用同一三seed Replay，将Critic总update扩到3200/6400；网络、loss、标签、配比、
评估集合均不变，Actor保持未构造。200步边界只重置一次optimizer，之后连续训练，并从本轮开始
在每个checkpoint保存Critic/flat optimizer state。

结果表明主Critic在6400步仍未平台：初始排错纠正率由1600步
`0.634/0.666/0.584`升至6400步`0.762/0.780/0.734`；2.8m/s pair accuracy由
`0.881/0.885/0.892`升至`0.921/0.925/0.927`；总体pair accuracy达到
`0.945/0.948/0.948`。历史heldout bank在两个seed改善、一个seed轻微退化，故后续不能只按训练
步数或最新checkpoint选模。

训练预算合同据此修订：OAC-1主Critic burn-in最低采用1600步，默认候选为3200步；可训练到6400
作为上界候选，但必须同时满足actor-visited纠错、高速层和heldout bank不退化。不得再使用200步
结果判断Replay信息不足。

flat头不共享这个放行结论。train阈值到heldout的false-stay在1600/3200/6400步分别为
`0.100/0.105/0.117`、`0.122/0.115/0.138`、`0.142/0.145/0.158`；随着训练继续反而恶化。
因此OAC-2仍冻结，flat监督或独立校准仍须作为第二步单独预注册，不能用全量固定0.5阈值的通过
结果替代episode-heldout gate。

独立validator对18个checkpoint零误差复算，并确认本轮新DBM rollout为0、Actor update为0、
formal validation/test封存。产物：

```text
outputs/mppi_proposal/online_absolute_sac_oac1_fixed_replay_extended_20260820_v1/
```

## 12. OAC-1B联合Value--移动系数结果（2026-08-20）

独立BCE flat头被连续移动系数辅助头替代。正式目标为
`c_move=max(delta J,0)/(max(delta J,0)+0.1)`；辅助头读取Twin Value输出、不一致度和动作差，
loss以0.05权重联合反传Twin Critic。固定`c_move<=0.5`对应原raw-gap 0.1，不做概率阈值搜索。

3 seed、400--3200步共12个post-training点全部通过固定门。3200步的bank-best recall为
`0.923/0.908/0.905`，warm false-stay为`0.008/0.008/0.013`；actor-visited pair约0.953，
2.8m/s约`0.932--0.941`，初始排错纠正率`0.776--0.810`。heldout bank排序相对联合训练前无
实质退化。

因此OAC-1B机制门在已消费fold-0上通过，但OAC-2仍不自动启动。先在新的episode outer split复核
同一模型、0.05辅助权重和固定0.5操作点；禁止重新调阈值。复核通过后，OAC-2中的系数用途为连续
缩放Actor更新，不作为硬开关：`delta_a_applied=c_move*delta_a_actor`。训练期仍由真实DBM cost
accept/reject控制selected晋级，坏样本全部进入Replay。

## 13. OAC-1B outer fold-1复核与OAC-2放行边界（2026-08-20）

新episode outer fold-1已经按§12完全冻结的合同完成。三seed各重新生成15360条Actor-visited
Replay；200步burn-in为0/3，但固定Replay续训到6400步后，actor-visited排序达到
`0.941--0.948`，heldout bank达到`0.942--0.948`，2.8m/s达到`0.918--0.923`。联合Value与
连续移动系数再训练3200步后，固定0.5门的bank-best recall为
`0.922/0.922/0.898`，warm false-stay为`0.017/0.017/0.010`；3/3 seed以及全部12个训练后
checkpoint通过。独立validator qualification为`JOINT_MOVE_COEFFICIENT_VALIDATION_PASS`。

因此OAC-2入口门现已通过，不再以“等待新split复核”为阻塞项。OAC-2实施时必须保留以下已冻结
参数，不能把fold-1再次用于调参：

1. Value burn-in使用已验证的6400步checkpoint；联合系数头使用0.05辅助权重和固定0.5物理门；
2. Actor更新低频、小步，应用量为`c_move * delta_a_actor`，系数连续使用而非二分类开关；
3. 每轮新Actor动作必须取得真实DBM cost并全部写Replay，Critic持续更新；坏动作不得过滤；
4. DBM accept/reject只决定selected Actor checkpoint是否晋级，不阻断latest Actor探索分布；
5. OAC-2只使用训练episodes及内部carve-out，formal validation/test继续封存；
6. OAC-2通过后仍需two-center guard下的短闭环A/B，不能由本节直接宣称部署收益。

复现产物：

```text
outputs/mppi_proposal/online_absolute_sac_oac01_fold1_20260820_v1/
outputs/mppi_proposal/online_absolute_sac_oac1_fixed_replay_fold1_20260820_v1/
outputs/mppi_proposal/online_absolute_sac_joint_move_coefficient_fold1_20260820_v1/
```

## 14. OAC-2 fold-1机制pilot结果（2026-08-24）

OAC-2已按固定合同执行20轮×3 seed：每轮1536次DBM交互、20次联合Critic/系数更新、1次Actor
更新；每5轮在600个episode隔离的internal-selection状态上做确定性DBM评估。三seed的四次
latest评估全部晋级，最终selected均为第20轮。

最终平均真实gain为`1.764/1.321/0.832`，episode-bootstrap CI下界均严格大于0，median与
2.4/2.8m/s分层gain也全部为正；headroom recovery为`2.01%/1.58%/0.94%`。Critic pair保持
`0.930--0.936`，连续系数固定门recall=`0.878--0.900`、false-stay=`0.012--0.018`。3/3 seed
通过，独立validator qualification为`OAC2_CONTINUOUS_ACTOR_VALIDATION_PASS`。

机制门通过不改变尾部纪律。direct gain P05=`-0.700/-1.929/-2.321`，worst=
`-27.449/-12.583/-13.627`，所以无保护Actor仍不允许部署；two-center guard下P05/worst为0。
下一步按原OAC-3计划，在其余outer folds复刻**完全相同**的20轮合同。不得用fold-1调Actor LR、
temperature、系数阈值或trust半径。至少2/3 fold、每fold至少2/3 seed通过后，才进入短闭环A/B。

复现产物：

```text
outputs/mppi_proposal/online_absolute_sac_oac2_fold1_20260824_v1/
```

## 15. OAC-3完整3-fold结果与OAC-4授权（2026-08-24）

fold-0/1/2均完成同一20轮×3 seed OAC-2合同，各自独立validator通过。进一步在每个fold的
outer-heldout 600状态上评估selected Actor，3 fold均为3/3 seed通过：九个mean gain为
`0.687--1.938`，全部bootstrap CI下界、median、2.4和2.8m/s门均为正。全九seed平均mean gain
为`1.322`，平均headroom recovery为`1.48%`。

OAC-3资格为`OAC3_OUTER_HELDOUT_PASS_READY_FOR_SHORT_CLOSED_LOOP`。因此现在只授权OAC-4短闭环：

```text
warm-only
vs
warm + fold-selected Actor two-center candidates
```

闭环必须覆盖nominal/high-speed/recovery、多seed，并报告累计cost、失败率、steering/acceleration
抖动、warm-return次数和ESS。direct Actor尾部仍明显为负（P05最低`-3.524`、worst最低
`-35.290`），所以Actor-only闭环、移除warm候选或提前部署均未授权。formal validation/test仍
封存；本阶段的outer-heldout属于训练池内部CV，不得称为最终正式测试。

```text
outputs/mppi_proposal/online_absolute_sac_oac2_full_3fold_20260824_v1/
```

## 16. 20/100/200轮训练预算复核（2026-08-24）

OAC-4按用户要求暂停。fold-1在不提高Actor LR、不修改Critic:Actor更新比和探索合同的条件下，
扩展到100与200轮×3 seed，两个新run均通过独立validator。三seed平均teacher headroom recovery
从20轮`1.51%`提高到100轮`6.79%`、200轮`11.51%`；Critic pair同步保持
`0.934/0.937/0.940`，说明20轮是明显欠训练，当前局部范围内Critic不是第一阻塞。

但预算增长同时把direct gain P05从`-1.65`推到`-5.94/-9.57`，worst从`-17.89`推到
`-64.52/-136.89`。200轮Actor mean J=`81.84`，仍远差于warm `29.17`与bank-best teacher
`5.16`；median J=`17.86`虽优于warm median `23.36`，说明主体改善和尾部发散并存。guard gain
仅从`9.80`缓慢升到`10.48`。

因此下一步不直接提高LR，也不继续把“均值持续下降”作为唯一晋级门。若继续Actor训练，必须先
预注册并配对验证：

1. state-wise regression/tail-aware Actor loss，不让少数高杠杆状态的错误更新被batch均值抵消；
2. selected checkpoint除mean/median/高速层外，加入direct P05与worst相对父selected的退化门；
3. 保持所有坏动作进入Replay、Critic持续更新和two-center部署守卫；
4. 新目标通过固定状态预算曲线后，再决定是否恢复OAC-4。

```text
outputs/mppi_proposal/online_absolute_sac_oac2_budget_20260824_v1/analysis.json
```

## 17. 200轮负尾逐状态归因与下一训练合同（2026-08-24）

逐状态DBM复算确认`17.861`是三seed median的平均，1800样本pooled median为`17.842`。Actor mean
并未随预算变坏，而是从初始`91.796`改善到`81.844`；其绝对值仍高主要来自初始就存在的重尾。
另一方面，回归比例稳定在约31%，但100→200轮最差5%有0.58--0.82 Jaccard重合，固定坏case
的损失被继续放大。按seed平均有137/600状态既差于初始又在100→200轮继续变差；严重18状态
中16个属于2.4/2.8m/s。

严重回归的83个seed-state中81个由position项主导，early steering移动中位只有`0.032sigma`，
表明问题是高杠杆维的小偏移累积，而非动作饱和或rate惩罚。最终Twin Value对全部样本端点改善
符号准确率`90.3%`，严重坏动作只有`9.6%`被误判为改善，因此下一步不再优先扩大Critic预算，
而是修Actor的跨状态风险分配和selected gate。

下一配对实验预注册为OAC-2T：

1. 保持200轮、LR=`1e-6`、`20 Critic : 1 Actor`、全部坏动作入Replay及continuous coefficient；
2. 在现有平均value loss之外，增加相对当前selected center的保守Twin Value soft-regression/CVaR
   项，禁止用真实DBM标签直接反传；
3. selected晋级除原mean/median/高速均值门外，增加direct P05不退化及2.4/2.8m/s尾部不退化门，
   worst只记录，不单独作为高方差硬门；
4. 主判据同时要求mean gain保持、P05明显优于当前200轮`-9.57`、严重回归数下降；three-seed逐项
   报告，不选择最好seed；
5. two-center guard、formal validation/test封存和OAC-4暂停状态不变。

```text
outputs/mppi_proposal/online_absolute_sac_oac2_budget_tail_20260824_v1/
```

## 18. OAC-2T固定尾部CVaR结果与修订（2026-08-24）

已完成固定权重`lambda=10/100`的top-10%预测回归CVaR两臂，各200轮×3 seed且独立验证通过。
相对原200轮，`lambda=10`将P05从`-9.57`改善到`-0.50`、worst从`-136.89`改善到`-9.81`、
回归比例从31.28%降到19.11%；但mean gain从`9.95`降到`1.39`，仅保留14.0%，guard J也从
`18.69`退到`19.36`。`lambda=100`几乎相同且稍差。所有tail floor在全部周期评估中均通过，
所以过度保守来自loss，不来自checkpoint gate；Critic pair仍约0.94。

两臂均标记`NOT_SELECTED`。下一OAC-2T2不再扫固定权重，改为：

1. 对预测回归设置非零material margin，避免惩罚Critic噪声尺度内的微小gap；
2. 将tail项写成预算约束`CVaR_beta(r) <= tau`，用dual变量自适应更新lambda；
3. `tau`由原20轮可接受尾部预注册，不在同一selection结果上择优；
4. checkpoint的P05/高速floor保留，因为本轮证明它们不会自动导致训练停滞；
5. 主门要求mean gain显著高于本轮`1.39`，同时P05显著优于原始`-9.57`，并报告guard J。

```text
outputs/mppi_proposal/online_absolute_sac_oac2_tailcvar_ab_20260824_v1/analysis.json
```

## 19. 多候选与真实task-loss容量裁决（2026-08-24）

为判断OAC收益上限来自多模态还是Actor本体，已完成两个train-only旁路诊断：

1. 固定K=4、多J16 elite集合监督相对同结构K=1只有小幅改善；median-seed OOF oracle recovery
   从`-3.120`到`-2.953`，仍显著小于0。Critic选择几乎跟随best-of-K，故多模态/selector不是
   当前主阻塞，不启动Diffusion。
2. 同一K=1 G-X直接反传DBM J50后，32/128 tiny recovery为`0.827/0.898`；完整fold-0 train为
   `0.834`，episode-heldout为`0.779`。但heldout J仍为`23.61`，离J16 `4.46`很远，且P05/worst
   仍负。

因此OAC-2T2暂不以“继续增加轮数”作为第一优先。新的结构前置门为：在真实DBM task-loss下严格
配对当前`center +/- 1 std`输出盒与更宽/全范围absolute-action输出，保持state encoder、split、
预算不变。只有输出支撑显著降低train和episode-heldout J，才把相同修改带回OAC Actor；否则再
考虑共享参数条件化/风险分配。Critic持续在线路线不被否定，但它不应承担现阶段结构上限诊断。

formal validation/test、闭环OAC-4继续封存；DBM task-loss仅为诊断/预训练，Query路线不得使用
DBM内部梯度。

## 20. 输出支撑tiny-128结构门通过（2026-08-25）

§19预注册的第一步严格A/B已完成。在同一train-only 128状态、同一初始G-X Actor和1200次真实
DBM task-loss更新下，`center +/- 1std`、`center +/- 3std`和physical `[-1,1]`的mean J分别为
`15.568/5.483/5.498`，对应headroom recovery为`0.8977/0.9924/0.9922`，J16参照为`4.668`。
三臂初始action最大差`1.19e-7`，独立DBM replay的分布指标误差为`0`。

`box1`有77.3%状态触及至少一个95%支撑边界，而`box3`为0；`full`不优于`box3`。因此当前结构
前置门判为`EXPANDED_ACTION_SUPPORT_MATERIAL_TINY128_GAIN`：`+/-1std`及其tanh饱和是已确认的
主要瓶颈之一，`+/-3std`在该子集已经足够。下一步只授权完整train-1200/episode-heldout-600的
`box1`--`box3`配对复核；通过后才修改OAC Actor输出映射。OAC-4、formal validation/test继续
封存，不能用本轮DBM梯度训练结果直接声称Query或闭环收益。

```text
outputs/mppi_proposal/dbm_task_loss_support_ab_20260825_v1/
```

## 21. `box3`完整train/heldout复核与OAC集成边界（2026-08-25）

完整fold-0复核已通过。相同2400次DBM task-loss更新下，`box1 -> box3`使train mean J
`19.792 -> 7.282`、episode-heldout mean J `23.580 -> 10.300`；recovery分别
`0.8312 -> 0.9741`和`0.7792 -> 0.9326`。heldout P95/worst从`91.11/817.54`降到
`33.06/455.34`，gain P05从`-0.622`微升至`-0.588`；回归比例增加`1.17pp`，未超过预注册
`+2pp`边界。`box3`在train/heldout均无95%支撑边界占用，`box1`分别为60.2%/62.5%。

独立DBM replay的全部指标和联合判定误差均为0，qualification为
`BOX3_SUPPORT_FULL_TRAIN_AND_HELDOUT_PASS_READY_FOR_OAC_INTEGRATION`。下一阶段授权将Actor输出映射
从`center +/- 1std`严格配对改为`center +/- 3std`，在固定状态OAC pilot中复核；不得同时修改
Critic、Replay配比、continuous coefficient、tail gate或训练预算，以保持单变量归因。初始Actor
action必须通过zero-adapter保持逐元素一致。只有OAC direct mean/median/P05/高速层与guard均通过，
才讨论恢复OAC-4。formal validation/test继续封存。

```text
outputs/mppi_proposal/dbm_task_loss_support_full_ab_20260825_v1/
```

## 22. `box3` OAC support-only配对结果（2026-08-25）

已完成fold-1、200轮、3 seed的`box1/box3`严格配对。旧固定CVaR loss未启用
（`tail_regression_weight=0`），但overall/2.4/2.8m/s P05 floor固定为`-2.5/-4.0/-4.1`，其余
Replay、Critic、continuous coefficient、探索和更新预算不变。初始action最大误差`4.17e-7`，
两臂独立validator均3/3通过且DBM Replay/selected指标误差为0。

`box3`将三seed平均tail-safe selected mean gain从`0.995`提高到`1.514`（+52.1%），三seed逐项
均提高；200轮latest mean gain从`10.171`提高到`15.543`（+52.8%）。但selected round两臂同为
`20/10/10`，selected P05从`-1.060`变为`-1.407`、worst从`-14.72`变为`-20.44`，说明更宽
支撑提高了可实现收益，却没有延长tail-safe训练时域。two-center guard mean J仅改善`0.032`。

`box1`约八成状态触及其95%输出边界，`box3`无触边；Critic material-pair accuracy均约0.94。
因此输出合同正式切换为`center +/- 3std`，支撑不再是下一阻塞。下一步只允许在`box3`基线上做
state-wise tail控制，优先nonzero material margin加自适应Lagrange/CVaR预算；固定`lambda=10/100`
继续禁用。所有坏动作继续进入Replay，P05/高速floor与two-center guard保留。OAC-4、formal
validation/test继续封存。

```text
outputs/mppi_proposal/online_absolute_sac_oac2_support_ab_20260825_v1/analysis.json
```

## 23. `+/-3std`默认强合同（2026-08-25）

从本节起，所有新OAC G-X run必须满足：

1. Actor输出支撑默认且固定为`center +/- 3std`；不传CLI参数也必须得到3std。
2. `+/-1std`仅作为显式A/B消融，必须同时传`--allow-box1-ablation`；普通训练不得静默退回box1。
3. 新run在`contract.json`记录default/active multiplier、消融授权及checkpoint-buffer要求；latest与
   selected checkpoint均保存`output_support_multiplier`，validator逐一核对。
4. `None`只用于读取历史无support-adapter checkpoint；full physical域不作为默认，启用前需另行
   预注册。
5. 3std只改变可达动作域，不放宽安全纪律：P05/2.4/2.8m/s floor、two-center guard、坏动作Replay、
   formal validation/test封存继续生效。

当前J状态采用review §11.99的统一口径。fold-1 internal-selection 600状态上，tail-safe selected
box3 Actor的direct/guard mean J为`90.283/19.387`；200轮latest为`76.254/18.538`，但其direct
P05/worst gain为`-10.572/-198.239`，不能晋级。warm mean J=`29.174`，candidate-bank best参考
J=`5.161`。所以后续基线是“3std selected Actor + two-center guard”，并继续处理state-wise tail；
不得用latest较低的mean/median掩盖尾部未过门。

## 24. OAC-2A自适应tail预算预注册（2026-08-25）

下一实验固定§23的`+/-3std`、fold-1、3 seed、200轮、Critic/Replay/continuous coefficient、
`20 Critic : 1 Actor`、探索、`0.02 sigma RMS` trust projection及现有overall/2.4/2.8m/s P05
checkpoint floor，只改变Actor tail项。历史`lambda=10/100`固定权重继续禁用。

新约束在保守Twin Value的未标准化`log1p(J)`空间定义：

```text
r_i = relu(log1p(J_actor,i) - log1p(J_selected,i) - 0.05)
c = mean(top 10% r_i)
L_actor = L_base + lambda * (c - 0.02)
lambda <- clip(lambda + 1.0 * (EMA_0.9(c) - 0.02), 0, 10)
```

`0.05`允许约5%的预测相对cost变化后才计入尾部，`0.02`是最差10%超额预算；lambda从0开始，
由数据自动升降。每轮必须落盘`c/EMA/lambda before/after`，validator重放完整dual递推。先做20轮×3
seed机制smoke，确认初始action、Replay、DBM复算、dual合同后再跑完整200轮。

完整实验与既有box3严格配对，判定分三层：

1. 机制有效：lambda曾激活、最终不固定饱和在10，递推复算误差不超过`1e-7`；
2. Pareto有效：200轮latest相对box3的P05与平均worst都改善，同时保留至少70%的box3 latest mean
   gain（阈值`10.88`）；
3. 安全时域有效：至少2/3 seed的selected round晚于box3的`20/10/10`对应值，且selected mean gain
   不低于box3三seed平均`1.514`。

若仅第1层通过，说明dual实现正确但预算口径无效；若1+2通过而3失败，保留为tail诊断但不授权
OAC-4；三层通过才讨论短闭环。formal validation/test保持封存，所有坏动作继续写Replay。

## 25. OAC-2A结果与下一路由（2026-08-25）

完整200轮结果为`ADAPTIVE_TAIL_PARETO_IMPROVES_BUT_MISSES_RETENTION_AND_SAFE_HORIZON_GATES`。
dual在3/3 seed按需激活且不饱和；latest P05/worst相对box3改善`8.446/86.391`，回归比例减少
`9.06pp`，明显优于固定CVaR。但mean gain retention=`69.16%`，略低于预注册70%；selected时域
只在seed1从10延长到80，seed0/2仍为20/10，所以OAC-4继续冻结。

下一步先做零新增rollout的state-wise dual校准诊断，逐状态比较预测超额、真实gain、速度与是否被
warm guard接管；报告tail recall、false-positive、分速度阈值和seed一致性。只有证据显示同一风险
信号可通过状态条件化稳定区分尾部，才设计下一长训；不得直接微调本轮`0.05/0.02/LR=1.0`数值。
默认3std、P05高速floor、two-center guard及坏动作Replay合同不变。

## 26. OAC-2A逐状态归因与后续边界（2026-08-25）

§25的零rollout诊断已完成。最终Twin Value的真实变化相关性`0.782`，material-tail ROC-AUC
`0.941`，所以Critic整体排序不是当前首要故障；但现有margin只召回`69.3%` material回归，且top-10%
均值budget没有逐状态non-regression保证。selected的495个回归样本中，latest修复193个、持续回归
302个，并新增82个回归。持续/新增回归的initial J中位仅`9.159/8.406`，而被修复及持续改善样本为
`21.531/28.085`：当前Actor对大headroom状态有效，对低headroom的stay/小步判断不足。

严重回归共17个seed-state，全部位于2.4/2.8m/s，16/17由position cost主导；early steering仅
移动约`0.027 sigma`中位也能被高速长horizon放大。默认3std不是该尾部来源。下一次OAC修改必须
保持Value/Replay/3std等合同，只允许一次单变量pilot比较：

1. 基线：当前全局adaptive tail；
2. 连续state-wise缩放：由保守Twin Value advantage/gap输出`[0,1]` move coefficient，连续缩放
   Actor更新，不做二分类stay开关；
3. 高速early-steering trust cap仅作为第二层消融，不能与第2项同时首轮引入。

判定必须同时报告selected→latest的persistent/recovered/new三类、material/severe recall、2.4/2.8
P05/worst及mean retention。若state-wise连续缩放仍不能使严重recall与P05改善且不保留至少70% mean
gain，则停止继续tail-loss工程，OAC只保留为候选生成训练实验；部署路径维持two-center guard。

## 27. 连续state-wise风险回拉结果（2026-08-25）

完整200轮×3 seed已完成并独立验证。该臂提高latest mean/median gain与guard J，也把selected round
从`20/80/10`延长到`20/80/60`，selected mean gain从`2.758`升到`4.019`。但latest P05/worst
从`-2.126/-111.848`恶化到`-3.088/-114.599`，2.4/2.8m/s P05分别恶化`2.966/2.086`，严重
回归从17增至29。按§26门判为`STATEWISE_RISK_SHRINK_FAIL_NO_FURTHER_WEIGHT_SWEEP`。

结论：连续风险回拉可作为checkpoint收益辅助，但共享loss权重不是逐状态安全投影，不能再通过扫
weight/temperature处理高速tail。保留selected checkpoint与产物做内部候选，OAC-4/formal
validation/test继续冻结。下一次tail实验必须改变约束作用点；优先候选是离线DBM/bank逐状态安全
投影标签，再将安全中心蒸馏进Actor，部署仍由two-center guard兜底。

## 28. 非均匀8-knot布局判定（2026-08-25）

100个train-only状态的matched-budget真实搜索与可微容量oracle均完成。mild前密布局
`[0,3,7,12,18,26,35,49]`的容量oracle mean J为`4.728`，略优于uniform的`4.793`；但128次
局部搜索mean J为`8.528`，明显差于uniform的`7.641`，且53/100状态不能恢复common anchor。
strong前密容量也不优于uniform。

因此当前不修改MPPI runtime的uniform knot times。前密布局只保留为未来原生参数化支线；重启条件
是同时重做该布局的warm-start、搜索协方差与Actor标签，并在相同预算下通过零baseline回归、P05/
worst及mean J配对门。不能把uniform teacher投影或1%的capacity oracle优势直接当作部署授权。

## 29. Critic容量与Pairwise Delta冻结Actor pilot（2026-08-25）

fold-1固定长期Replay、outer-heldout 600状态、3 seed×6400步严格配对已完成并独立验证。近2×参数
Critic将bank top-1 regret从`2.366`降至`1.456`、latest回归AUC从`0.881`升至`0.902`，证明容量
扩展有小幅真实收益，但seed极值方差仍大。

更有效的是Absolute Value + 反对称Pairwise Delta双头：参数量仅从467,585增至544,706，直接
Pairwise评分将bank top-1 regret降至`0.585`、Actor三候选有害选择从`9.00%`降至`6.83%`、严重
回归召回从`0.556`升至`0.742`，三项均在3/3配对seed改善。下一次若重启OAC训练，应优先把该头
作为同状态`new vs selected`风险/排序辅助；扩大Critic作为次级配对臂，不同时修改Actor结构。

边界：2.8m/s严重召回仍只有`0.50--0.75`，本轮Actor没有更新，所以不授权直接替换现有Twin
Value、不授权移除真实DBM accept/reject或two-center guard。正式接入前至少还需其它outer fold
复核，并预注册Pairwise评分如何进入Actor loss/候选裁决；不得把本轮固定Replay识别收益写成在线
Actor收益。

## 30. Pairwise Delta在线接入结果与路由（2026-08-25）

已按§29完成单变量接入：Pair Twin独立于Absolute Twin，前者只替换Actor adaptive-tail风险信号，
后者继续提供主value gradient；Pair先冻结Actor预训练1600步，再随online Replay按`20:1`合同持续
更新。默认代码开关保持关闭。

原dual从0开始的20轮×3 seed中，Pair准确率约`0.90--0.92`，但dual全程为0，故Pair/Absolute两臂
Actor指标逐位相同（最大差`3.1e-4`）。追加`dual initial=10`的机制压力A/B后，Pair相对Absolute
使mean/median gain提高`+0.171/+0.064`，但P05/worst下降`-0.283/-4.054`，2.4/2.8m/s P05
下降`-0.644/-0.114`，回归比例增加`1.11pp`。3/3 seed的主体收益均提高，3/3 seed的P05与worst
均未提高。

判定：Pairwise Delta作为局部候选排序器有价值，但单独接管tail risk不放行。停止200轮、更多fold
和权重扫描；若后续复用，只允许两个预注册方向之一：（1）训练期候选预排序，真实DBM仍作最终标签；
（2）Pair与Absolute风险的保守并集，且必须在短20轮先同时改善P05/worst和高速层。two-center guard、
坏动作Replay、3std、formal validation/test封存和OAC-4暂停保持不变。

## 31. Critic--DBM梯度差距归因与下一实验顺序（2026-08-25）

§11.107对当前200轮最终Critic完成600 internal-selection状态×3 seed×selected/latest的真实DBM
梯度审计。当前latest动作上的value Pearson/action cosine median/P10/norm ratio为
`0.975/0.979/0.726/1.128`，early steering P10为`0.835`。当前平均差距不再由Critic局部梯度
完全失效主导；但固定`0.002σ`动作步下Critic仍比真方向多约4.17%的回归，安全门不能移除。

主差距改判为两项：（1）OAC最小化`mean log1p(J)`，DBM容量实验最小化`mean raw J`，对应共享
Actor参数梯度cosine只有`0.718/0.359/0.325`；（2）OAC每轮实际仅移动约
`0.00021--0.00028σ`、共200次，而容量实验为2400次、LR `2e-4`。相同当前Adam moments和输出
步长下，raw-J一步mean gain约为Critic两倍，但回归比例也更高。

下一严格顺序：

1. **OAC-2C目标权重短pilot**：Critic、Replay、Actor LR、20:1、3std和tail gate不变，只改变
   Actor跨状态聚合，比较`gamma=0`（现log）、`0.5`和`1`（近raw），高cost权重必须clip并落盘；
2. 只有目标权重提高mean且P05/worst仍由现gate控制，才单独提高Actor更新RMS/轮数；
3. 不先扩大Critic、不增加gradient loss、不同时修改Pairwise/tail/Actor结构。

该pilot仍只用train-side固定DBM及internal selection；formal validation/test与OAC-4继续冻结。

## 32. OAC-2C raw-cost-aware聚合短pilot结果（2026-08-25）

§31第1步已完成。20轮×3 seed严格配对比较`gamma=0/0.5/1`，Actor weight cap固定2048并按batch
均值归一；其余OAC合同不变。latest mean gain为`2.049/2.764/3.059`，selected mean gain为
`1.514/1.946/2.610`，两种口径gamma=1均为3/3 seed优于gamma=0。说明改回近似mean raw J的
跨状态权重确实能更快压低平均cost。

tail没有同步单调通过：latest P05平均为`-2.225/-1.659/-1.446`，但worst为
`-24.67/-26.87/-28.61`；selected P05以gamma=0.5最好，gamma=1的selected worst仅1/3 seed改善。
gamma=1有效样本比例约0.249，cap仅0.143%触发，结果是raw权重集中本身而非cap主导。

下一步顺序调整为：

1. 若继续扩大Actor预算，保留gamma=1主体臂和gamma=0.5 tail对照，不再带gamma=0长跑；
2. 只改变Actor更新RMS/轮数，Critic、Replay、tail门、3std和两中心guard不变；
3. 只有mean继续提高且P05/worst/2.4/2.8m/s floor不退，才扩outer fold；否则采用gamma=0.5或回到
   selected checkpoint，不通过调大raw权重继续追mean。

formal validation/test与OAC-4仍冻结。

## 33. OAC-2C 200轮预算判定与下一顺序（2026-08-27）

gamma=0.5/1已扩到200轮，gamma=0复用历史200轮。20轮确实低估mean容量：latest mean gain最终为
`10.749/13.740/14.828`，gamma=1的headroom recovery为0.171。但gamma=1的median/P05/worst为
`1.188/-2.430/-185.51`，均不优于gamma=0的`1.622/-2.126/-111.85`；2.8m/s P05三臂3/3 seed
仍失败。

更关键的是最终selected two-center guard J为`19.296/19.325/19.314`，raw-aware direct mean收益
没有传到部署guard。原样增加轮数分支关闭。

下一顺序：

1. 零训练归因：在selection状态上复算gamma权重、Actor-vs-warm胜负、direct gain贡献和tail成员，
   验证高权重是否集中在guard拒绝状态；
2. 单变量方差实验：gamma=1保持目标不变，只把Actor batch从128增到512或做4次梯度累积，保持每轮
   一个optimizer step和相同LR；检查worst/seed方差能否下降；
3. checkpoint合同补记worst floor，但不得把它写成训练修复；
4. 若部署目标优先，预注册warm-relative、margin+clip的advantage权重，使训练预算落到可能击败warm的
   状态；若无guard direct Actor优先，则继续用bank-best headroom，但需要median/P05/worst联合门。

在上述归因前不启动400/800轮，不扩Critic，不开放formal validation/test。

## 34. 跨历史实验复核后的优先级修正（2026-08-27）

§33只从raw聚合长训曲线出发，遗漏了已经完成的DBM task-loss容量、3std支撑、tail-loss和最终Critic
梯度审计。完整证据显示：同一G-X类Actor在box3下使用真实DBM梯度，train/episode-heldout恢复率可达
`97.41%/93.26%`；当前OAC gamma=1 latest只有`17.1%`。与此同时，当前Actor-visited动作处Critic
梯度cosine median/P10已经为`0.979/0.726`，而fixed CVaR、adaptive dual和statewise shrink均已
验证不能通过共享权重稳定修复tail。

因此覆盖§33的执行优先级如下（§33原始记录保留）：

1. 首先做**matched gradient-source trajectory A/B**：同一box3初始Actor、raw-mean目标、状态batch、
   optimizer、LR、update数与投影，只切换DBM真实梯度和持续更新Twin Critic梯度；统一报到J16的恢复率、
   mean/median/P05/worst、速度分层和Actor击败warm比例。
2. 若DBM臂在相同预算下也明显低于历史93%容量，下一变量才是Actor update数/LR/全batch；若仅Critic臂
   落后，才修改Critic交互频率、target/ensemble或局部数据合同。
3. `batch 128 -> 512`保留为gamma=1的二级ESS/方差消融，不先于第1项。
4. warm-relative或guard-aware目标降为部署分支，只在主体逼近能力确认后处理；不得把guard J无变化等同于
   Actor direct优化无效，也不得重复扫描全局tail权重。

本节不开放formal validation/test和闭环；只用train-side固定DBM/internal-selection，并继续保留3std、
坏动作Replay、P05/高速记录与two-center guard。

## 35. 初始大LR衰减实验结果与后续边界（2026-08-27）

已完成与历史CUDA gamma=1基线严格配对的200轮×3 seed实验。唯一变化是Actor LR从固定`1e-6`
改为`1e-5`保持20轮、余弦衰减到round 80的`3e-6`、再衰减到round 200的`1e-6`；box3、raw
聚合、adaptive-tail、Replay、Critic `20:1`和trust均不变。

latest mean J从`76.968`降到`69.409`，median从`18.825`降到`17.953`，headroom recovery从
`17.09%`提高到`25.83%`，回归比例从`24.72%`降到`20.50%`，3/3 seed主体均改善。Critic pair
保持约`0.92--0.95`，trust projection为0，确认固定Actor LR过小且Critic没有因提速立即失效。

但2.4/2.8m/s P05在3/3 seed全部恶化，2.8m/s P05从`-8.456`降到`-12.913`；平均worst从
`-185.515`降到`-212.895`。所有checkpoint都被高速tail门拒绝，selected round为`0/0/0`。
因此本轮只通过mean-capacity机制门，不通过训练晋级门。

下一顺序：

1. matched DBM-vs-Critic梯度源A/B继续作为剩余主体差距的首要判决；
2. LR调度最多再做一次收紧消融：`1e-5`只保持10轮，或初始`5e-6`，二者只选其一预注册；
3. 若高速P05仍不过门，停止LR形状扫描，把安全约束移到逐状态DBM/bank safe projection，不再改共享
   tail loss；
4. 当前LR衰减latest不得作为部署Actor，two-center guard、formal validation/test和闭环封存不变。

## 36. Actor学习率边界扫描后的计划更新（2026-08-27）

已完成round-80严格配对的初始LR扫描：`1e-5/2e-5/4e-5/8e-5/1.6e-4/3.2e-4`。恢复率依次为
`19.66/26.99/35.80/46.49/53.35/56.17%`，说明Actor更新尺度是已证实的主体瓶颈。`1.6e-4`
已有76.7%早期更新触发`0.02sigma` trust，`3.2e-4`为100%触发，且后者只增加2.82pp恢复、median
略退，因此LR边界扫描到此关闭。

更新后的执行顺序：

1. 主体诊断固定使用`1.6e-4`初始LR和`0.02sigma` trust，不再增加名义LR；
2. 在该有效步长合同下执行matched DBM-vs-Critic gradient-source trajectory A/B，解释剩余约44%
   headroom；
3. tail作为独立安全分支，优先做逐状态DBM/bank safe projection，再通过two-center guard复算；不再做
   共享tail-loss权重扫描；
4. 只有主体梯度源结论和逐状态安全投影同时完成，才讨论训练晋级；当前全部LR臂性能门0/3，selected均为
   round 0；
5. formal validation/test和闭环继续封存。

边界artifact：

```text
outputs/mppi_proposal/oac2_lr_boundary_scan_20260827_v1/analysis.json
```

## 37. 全局trust边界完成后的计划更新（2026-08-27）

固定初始LR=`3.2e-4`后，trust `0.02/0.04/0.06/0.08sigma`的round-80恢复率为
`56.17/64.21/63.89/64.73%`。`0.04sigma`带来决定性+8.05pp，后两档只在不足1pp范围内波动，
因此全局trust扫描关闭。主体研究默认工作点更新为`3.2e-4 + 0.04sigma trust`；它是进入平台的
最小上限，并给出最好的median J=`12.415`。

下一顺序：

1. 在该工作点执行matched DBM-vs-Critic gradient-source trajectory A/B；
2. 若DBM梯度臂明显越过当前64.2%，优先处理Critic梯度累计/更新时序；若两臂都停在平台，处理共享Actor
   优化和目标聚合；
3. tail安全单独实现逐状态DBM/bank safe projection，再与two-center guard组合评价；
4. 不再提高LR或全局trust，不再扫描共享tail loss；
5. 当前全部臂性能门0/3，formal validation/test及闭环继续封存。

```text
outputs/mppi_proposal/oac2_trust_boundary_scan_20260827_v1/analysis.json
```

## 38. `6.4e-4 × 0.06sigma`交叉臂后的边界修正（2026-08-27）

补充交叉臂把round-80恢复率从`63.89%`提高到`66.75%`、mean J从`36.461`降到`33.942`，证明
`3.2e-4`没有完全利用`0.06sigma`。但改善仅2/3 seed成立，median从`12.613`退到`13.359`，回归比例
从`24.22%`升到`28.67%`；前20轮75%更新已触发trust。

因此计划不把该臂替换为稳定默认：

1. 稳定主合同仍为`3.2e-4 + 0.04sigma`；
2. matched DBM-vs-Critic梯度源A/B以稳定合同为主，`6.4e-4 + 0.06sigma`只作激进容量旁证；
3. 不把66.75%写成严格硬上限；只有明确需要pure-mean ceiling时，才允许再做一个100%投影饱和确认臂；
4. tail仍走逐状态safe projection + two-center guard，formal validation/test和闭环继续封存。

```text
outputs/mppi_proposal/oac2_lr_trust_cross_ab_20260827_v1/analysis.json
```

## 39. 通用采样中心评价合同改为warm-relative direct center（2026-08-27）

为支持后续DBM/Query横向比较，策略评价不再要求可用oracle。主合同只评价未加噪的单一Actor center：
在同一状态上分别计算`J_warm`与`J_actor`，不混入采样分布、MPPI wrapper、softmax或two-center最终
执行结果。主报告固定为：

1. `P(J_actor < J_warm)`；
2. `G_w = J_warm - J_actor`的median、P05、worst；
3. `sum(G_w) / sum(J_warm)`聚合改善；
4. 速度×场景分层；
5. DBM/Query之间的改善符号一致率与冲突帧。

`J16/headroom recovery`降为DBM专项容量诊断，不能作为Query通用门。旧OAC日志的
`gain_vs_initial/regression_fraction/P05/worst`以初始Actor为参考，后续禁止把它们解释成相对warm。

现有高更新臂`6.4e-4 + 0.06sigma`的round-90三seed重算显示：Actor超过warm比例`69.28%`，
warm-relative gain median `+6.726`，但gain mean `-4.675`，聚合相对改善`-16.0%`；2.8m/s的
mean J为`77.733`，显著差于warm的`47.301`。与稳定`3.2e-4 + 0.04sigma`相比，胜率只增加
`1.45pp`、median gain只增加`0.115`，而Actor mean J改善`2.269`。因此高LR/trust只作为主体容量
旁证，不能根据旧initial-relative tail字段晋级。

下一次评测实现必须新增逐状态落盘：`warm_cost`、`actor_cost`、未截断`gain_vs_warm`、episode、speed、
scenario与action hash。只有这些字段齐全后，才计算warm-relative P05/worst和DBM/Query一致性；
现有round-80/90 summary不足以事后恢复这两个尾部量。

## 40. warm只作评价参照后的正式重评分与gamma决策（2026-08-27）

已补齐§39要求的逐状态重评分产物。固定600个train-side internal-selection状态，对warm和Actor latest
分别做未加噪DBM J50 rollout，并落盘完整的cost、gain、episode、speed、scenario和action。warm仅是
外部配对参照，没有进入Actor输入、loss、reward、anchor或权重；训练仍优化绝对cost。

稳定`3.2e-4 + 0.04sigma`与高尺度`6.4e-4 + 0.06sigma`的正式pooled结果分别为：胜warm
`67.83%/69.28%`，gain median `+6.602/+6.748`，P05 `-114.410/-99.976`，worst
`-904.064/-606.039`，聚合改善`-23.8%/-16.0%`。因此高尺度臂在同一warm口径下的主体和尾部
都优于稳定臂，但聚合结果仍未超过warm，不能授权单中心替换或部署。

随后固定高尺度全部合同，只将绝对cost聚合`gamma=1 -> 0.5`。gamma0.5的胜率、median、P05、worst、
Actor mean J和聚合改善全部退化：`67.67%/+6.101/-127.115/-821.581/38.983/-33.6%`，对照
gamma1为`69.28%/+6.748/-99.976/-606.039/33.849/-16.0%`。逐seed median/P05/worst均0/3改善。

计划据此收紧：

1. gamma固定为1，停止gamma/tempering扫描；
2. 禁止把随机实现的`J_warm`引入训练，warm只保留为跨DBM/Query可复用的外部中心比较基准；
3. 现有Actor排序以高尺度gamma1为首，但仍需保留two-center guard，formal validation/test和闭环不开放；
4. 后续若继续优化Actor，应针对共享state-to-action映射和失败状态的绝对cost学习，不再通过warm-relative
   loss或继续降低gamma处理；
5. Query横向评测复用同样的逐状态落盘合同，各模型用自身rollout cost，不比较跨模型绝对cost尺度。

```text
outputs/mppi_proposal/oac_warm_relative_centers_20260827_v1/
outputs/mppi_proposal/oac_warm_relative_gamma05_high_ab_20260827_v1/
```

## 41. K=1/4/8 Actor microstep结果与下一训练合同（2026-08-27）

已完成严格配对的90 outer-round、3 seed机制实验。每轮Critic更新始终为20次；Actor分别更新
`K=1/4/8`次，调度LR与单步trust除以K，每轮累计trust保持`0.06sigma RMS`，temperature/tail dual
各更新一次。三组validator的工程、Replay、DBM复算、更新计数、LR与trust回放均逐seed通过；
formal validation/test未加载。

round-90内部headroom recovery为`0.6562/0.6627/0.7101`，seed标准差为
`0.0454/0.0044/0.0011`。warm-relative direct-center胜率为`67.39%/70.78%/74.56%`，median gain为
`6.355/7.089/7.638`，P05为`-102.272/-100.815/-84.558`。K=8明确改善主体、收敛速度和seed
稳定性，但mean gain仍为`-1.101`、worst为`-832.075`，性能选择门仍0/3，不能授权部署。

计划更新如下：

1. OAC主训练默认从K=1改为K=8；Critic仍20次/outer round，保持交错顺序与独立随机流；
2. 下一单变量实验增加outer Actor-visited refresh轮数，保持gamma=1、`+-3std`、累计trust
   `0.06sigma RMS`、LR调度、Replay与K=8不变；主要观察round-90后的平台是否继续下降；
3. 暂停K>8、gamma、LR和全局trust扫描，避免重新混入有效总步长变量；
4. 继续同时报告内部headroom与warm-relative direct-center胜率/median/P05/worst/聚合改善；warm只作
   外部评价参照，不进入训练；
5. 若更长K=8仍稳定在约0.71，才转向Actor-visited状态刷新频率/Replay新鲜度或matched DBM梯度源；
6. two-center guard、formal validation/test与闭环仍不开放。旧性能gate仍选round0，latest只作机制容量。

```text
outputs/mppi_proposal/oac2_multiupdate_ab_20260827_v1/analysis.json
outputs/mppi_proposal/oac2_multiupdate_warm_relative_20260827_v1/
```

## 42. K=16/32边界与后续默认合同（2026-08-28）

已按§41继续完成K=16和K=32严格配对边界实验。round-90 recovery从K=8的`0.7101`提高到
K=16的`0.7411`，再到K=32的`0.7492`；后一增量只有`0.81pp`。warm-relative聚合改善依次为
`-3.78%/+5.39%/+7.86%`，胜warm比例为`74.56%/76.11%/75.72%`。K=32继续改善mean/P05，
但不再提高胜率，主体收益已经进入边际区。

§41“默认K=8、暂不扫更大K”被本节覆盖。新计划合同为：

1. 默认训练使用K=16；需要最大化离线direct-center质量时允许K=32；
2. K扫描在32停止，不运行K=64；
3. 下一单变量转向Actor-visited refresh/Replay新鲜度：保持K=16或配对K=32、20 Critic updates、
   gamma=1、`+-3std`与累计trust不变，改变每轮新状态/新动作进入Replay后的更新组织；
4. 继续以warm-relative胜率、median、P05/worst、聚合改善和2.4/2.8m/s分层为主报告；
5. pooled warm改善转正不替代旧安全门。K=16/32仍selected round 0、性能门0/3，formal/test、闭环和
   two-center guard边界不变。

```text
outputs/mppi_proposal/oac2_multiupdate_boundary_20260828_v2/analysis.json
```

## 43. K=20严格1:1对照后的最终K合同（2026-08-28）

K=20严格执行每轮20次`Critic -> Actor`一一交错，但没有支配K=16：recovery
`0.7400 vs 0.7411`，胜warm `75.83% vs 76.11%`，聚合改善`5.08% vs 5.39%`。它只在median gain
和P05上有约`+0.22/+1.07`的小幅改善，seed方差与worst反而略差。

因此§42计划保持并进一步收紧：

1. K=16是正式默认，K=20不晋级；
2. K=32只用于不计训练成本、优先mean/P05的离线质量版本；
3. K扫描结束，不运行K=64；
4. 下一实验只研究Actor-visited refresh/Replay新鲜度，禁止同时改K、gamma、LR、trust；
5. 所有性能门、two-center guard、formal/test与闭环封存规则不变。

```text
outputs/mppi_proposal/oac2_multiupdate_boundary_20260828_v3/analysis.json
```

## 44. Actor-visited刷新频率A/B后的训练组织合同（2026-08-28）

已完成严格等预算的2x刷新A/B。K=16基线`90×256×20C×16A`与候选
`180×128×10C×8A`拥有相同的context访问、Critic/Actor更新、辅助更新、评估次数、每microstep LR和
trust；候选只把Actor动作进入Replay并反馈给Critic的周期减半。

候选没有通过预注册机制门：recovery下降`0.51pp`，warm-relative胜率下降`0.56pp`，聚合改善下降
`1.51pp`，P05和worst分别退化`2.13/53.48`，只有median改善`0.18`。因此：

1. 训练组织保持K=16、90 outer rounds、每轮20 Critic/16 Actor；不采用2x刷新；
2. 停止单独扫描Replay刷新频率。现有50% recent采样已足以排除“反馈延迟是主瓶颈”；
3. 下一轮必须改变新数据的信息含量，例如按当前Actor失败/高不确定状态定向生成Actor-visited rows，
   或改变可泛化的state-action训练目标；不得只重复同分布状态并缩短block；
4. K=32仍只是离线mean/P05质量候选，不与刷新变量混扫；
5. warm-relative direct-center报告、2.4/2.8m/s分层、two-center guard及sealed split规则保持；
6. 所有latest结果仍是机制容量。性能选择门0/3、selected round 0，不授权部署或闭环。

```text
outputs/mppi_proposal/oac2_refresh_cadence_ab_20260828_v1/analysis.json
```

## 45. Matched gradient-source裁决后的主线调整（2026-08-28）

K16 OAC的Critic与可微DBM Actor梯度源已完成严格配对。真实DBM梯度在相同90轮、1440 Actor更新、
1800 Critic更新、Replay、LR/trust及风险合同下，将recovery从`0.7411`提高到`0.7599`，三seed全部
改善；warm聚合改善从`5.39%`提高到`10.98%`。因此Critic误差有真实累计代价。

但该提升只有`1.88pp`，相对历史DBM task-loss heldout容量参考的`19.15pp`差距只关闭约`9.8%`。
matched DBM本身仍平台在约0.76，且worst未改善，所以计划调整为：

1. 冻结Critic结构/loss扫描；保留现有Twin Critic作为可部署OAC梯度源和候选估值器；
2. 不再把“换更准Critic”作为逼近J16的首要路线；DBM真实梯度只保留为机制oracle；
3. 下一优先诊断共享Actor参数梯度冲突：按速度×场景和高/低cost状态计算batch/分层参数梯度cosine、
   norm及一步真实DBM gain，确认哪些状态组相互抵消；
4. 在该诊断之后只做一个Actor目标单变量A/B：当前stochastic sampled-action + coefficient目标，对照
   deterministic mean-action raw-J surrogate；保持K16、box3、LR/trust和状态batch不变；
5. 若deterministic surrogate明显逼近matched DBM task-loss容量，再将它蒸馏/近似到Query可用训练；
   若仍平台，则转向逐状态proposal/search监督而不是继续全局共享梯度；
6. worst更差说明tail继续独立处理；two-center guard、warm-relative报告、formal/test和闭环封存不变。

```text
outputs/mppi_proposal/oac2_matched_gradient_source_ab_20260828_v1/analysis.json
```

## 46. exact-DBM共享Actor参数梯度冲突确认与下一合同（2026-08-28）

§45之后已继续完成两个单变量结果：确定性center DBM目标只把recovery从`0.7599`提高到`0.7701`；
固定Actor探索bank的guided-2x只在约5%状态上提高best-of-6，不能解释剩余主体差距。随后按计划完成
3 seed×600状态逐状态共享参数梯度审计，独立validator全通过。

三seed全参数cancellation ratio为`0.1085/0.1201/0.0788`，逐状态参数梯度与batch全局方向负对齐比例
为`42.0%/44.5%/45.0%`。五个速度组两两cosine最小值为`-0.818/-0.704/-0.517`；最高cost四分位
对全局方向cosine为`0.919/0.985/0.974`，说明raw mean更新主要由高cost状态支配，同时与其他速度/
cost组发生稳定冲突。

真实一步验证同样成立：`0.002sigma`逐状态独立DBM动作步mean gain为`0.902`、P05为`+0.0446`、
回归为0；共享saved-Adam参数步只有`0.327`、P05 `-0.352`、回归`40.2%`，动作更新对真实下降方向
cosine中位仅`0.129`。正式判定为
`SHARED_ACTOR_PARAMETER_GRADIENT_CONFLICT_CONFIRMED`。

计划更新：

1. 停止把Critic、Replay刷新、探索bank、LR/trust或K扫描作为主体；K16、box3和现有数据合同保持；
2. 下一项先做**冻结Actor的方向级零训练A/B**，固定相同DBM raw J与输出步RMS，比较plain mean、
   5速度组PCGrad、速度/成本组CAGrad（或MGDA型方向）；不得同时改网络结构或数据；
3. 方向级门：3/3 seed提高共享步相对独立动作步的mean-gain retention，降低负动作对齐/回归比例，
   且最高cost与2.4/2.8m/s收益不出现决定性倒退；
4. 只有方向级门通过才做20--30轮exact-DBM deterministic-center短训练A/B；该训练仍是DBM机制oracle，
   不是Query部署合同；
5. 若梯度手术方向本身失败，下一分支为speed/regime条件化head或轻量专家，验证共享Jacobian耦合能否
   被结构隔离；不先盲目扩大统一MLP；
6. 另保留同fold固定状态direct-task-loss容量标定，用于把历史`0.9326`从跨合同参考变成严格上限；
   它不阻塞方向级冲突实验；
7. formal validation/test、闭环与单中心部署继续封存。warm只作外部中心评价，探索bank不进入本轮
   exact-DBM Actor梯度。

```text
scripts/model_verify/analyze_mppi_oac2_actor_parameter_conflict.py
scripts/model_verify/validate_mppi_oac2_actor_parameter_conflict.py
outputs/mppi_proposal/oac2_actor_parameter_conflict_audit_20260828_v1/
```

## 47. 目标速度域切换与独立数据生成合同（2026-08-28）

目标部署速度改为`40--100 km/h`（`11.11--27.78 m/s`），与当前OAC数据的
`1.2--2.8 m/s`完全不重叠。因而§46预定的低速PCGrad/CAGrad实验暂停；§46数据仅保留为共享
Actor参数冲突的机制证据，不能外推目标域收益。

当前执行项改为独立train-only高速采集，不先审计DBM模型：

- 速度：`40/55/70/85/100 km/h`；
- 工况：steady、under/overspeed recovery、lateral、heading、combined recovery；
- 规模：`5×6×1 episode×20 snapshots = 600 states`；
- snapshot从control step 250起、stride 12、每步64个MPPI候选；
- 新collection、seed段和plan hash完全独立，不混入历史split；
- 不创建validation/test，不训练Actor/Critic，不运行闭环性能A/B；
- 默认ROS上限保持10 m/s，仅该plan显式启用30 m/s reference ceiling。

数据落盘并逐episode验证后，再在目标速度train-only池中重新定义state normalization、OAC Replay
和后续episode-grouped split；在此之前不继续低速梯度手术、LR/trust、K或Critic扫描。

### 47.1 实际落盘结果与训练入口修正

30 episode / 600 snapshots / 38,400 candidates已全部生成并逐episode验证通过，但step 250后的实际
`vx`中位仅`2.06 m/s`、P95 `11.17 m/s`，没有实现实际状态40--100 km/h覆盖。该集合降级为
`HIGH_REFERENCE_SPEED_STRESS_COLLECTION_COMPLETE_VALIDATED_ACTUAL_SPEED_TARGET_MISSED`，只能作为
高reference压力和失稳尾部数据，不直接启动目标域OAC。

已从同批连续trace的step 0--4生成150-context、9,600-candidate的实际高速DBM replay；实际速度中位
`65.60 km/h`，范围`29.99--104.84 km/h`，82%严格位于目标区间。它是当前唯一允许用于目标域训练
pipeline smoke和后续teacher/search sidecar的入口；仍是train-only，不创建formal validation/test。
在扩大独立高速episode覆盖之前，不恢复低速PCGrad/CAGrad，也不把600帧压力集与150帧初始状态集按
同分布训练或汇总。

```text
outputs/mppi_proposal/highspeed_collection_summary_20260828_v1/summary.json
outputs/mppi_proposal/highspeed_initial_dbm_replay_20260828_v1/summary.json
```

## 48. 高速proximal teacher完成与Actor入口（2026-08-30）

150个实际高速初始context已生成固定DBM proximal teacher。每状态129次中心评价，warm恒为候选0；
150/150 teacher严格优于warm，零基线违规。聚合direct cost相对warm下降`2.9295%`，episode-bootstrap
95% CI为`[2.7251%,3.2156%]`；只看实际40--100 km/h的123帧为`2.9182%`。独立DBM重放误差为0。

teacher residual为中等有界移动（标准化RMS中位`0.553 sigma`），且全部best来自两轮re-centering，
说明单轮bank不足。所有150帧都需要move，当前数据不能训练有代表性的flat/stay判别器。下一步合同为：

1. 仅在这150帧上做episode-grouped cross-fit Actor可学性pilot；30个episode整体分折，禁止逐帧随机切；
2. Actor输入使用现有history/reference/current/warm合同，输出bounded absolute center或warm residual做严格
   单变量对照；不引入低速checkpoint作为anchor；
3. 先报train fit，再报OOF warm-relative direct cost、胜warm比例、P05/worst及实际速度分层；
4. teacher只有约3%相对headroom，因此Actor至少应稳定保留50%的teacher gain且P05不低于warm，才值得
   扩大高速独立episode；
5. 600帧高reference失稳压力集继续隔离，formal validation/test继续封存；
6. 即使Actor通过，部署仍保留warm+Actor two-center guard，因为本数据没有stay负例。

```text
outputs/mppi_proposal/highspeed_proximal_teacher_20260830_v1/
```

## 49. 高速Actor与Twin Critic预训练结果（2026-08-30）

已完成`5 fold x 3 seed`、episode-grouped的双网络预训练。这里的“两个网络”是Actor与Twin Critic
两类，而不是两个Actor：

- Actor：strict no-anchor clean G-X，直接输出absolute 8x2 center；warm仅作为DBM评价基线和未来
  two-center guard候选，不进入Actor输入。输出支持为fit折teacher的`+/-3 std`；`current`为
  `[vx,yaw_rate,acceleration,steering]`，不使用实车不可直接获取的`vy/beta`；
- Twin Critic：absolute-action scalar value，使用每状态全部129个search center及其真实DBM J50做
  `log1p(J)`回归+同状态排序预训练。预训练无TD、无bootstrap，之后不得冻结；
- 分折：90 fit / 30 internal selection / 30 OOF，全部按episode隔离；formal validation/test未读取。

主结果：Actor fit/OOF teacher-gain recovery中位分别为`0.723/0.613`，但OOF范围为
`-0.783--1.149`，warm回归比例中位`0.333`且所有run的gain P05为负，故“主体中位门通过、tail门
失败”。Twin conservative Critic的OOF log-cost Pearson中位`0.992`、warm-teacher排序准确率中位
`1.0`、129候选bank recovery中位`0.588`。高Pearson含速度层绝对cost信号，局部bank recovery才是
后续OAC更重要的口径。

独立重载15个checkpoint并fresh DBM复算通过：warm/teacher cost误差0、episode leakage 0。下一阶段
只授权train-only、同fold的Actor-visited持续OAC：从本checkpoint初始化Actor/Twin Critic；每轮由Actor
产生新center和局部候选，DBM计算真实reward写Replay，Twin Critic与Actor均持续更新。禁止把预训练
Critic冻结后做一次性梯度更新，也不训练缺少负例的flat/stay head。正式闭环前保留：

1. warm-relative recovery与胜warm比例；
2. P05/worst及速度xscenario分层；
3. Critic同状态bank ranking/gain recovery与twin disagreement；
4. warm+Actor two-center guard；
5. formal validation/test继续封存。

```text
outputs/mppi_proposal/highspeed_actor_twin_critic_pretrain_20260830_v2/
```

记录：`v1`复用了legacy `current=[vx,vy,acceleration,steering]`，违反实车输入合同，已降级为诊断；
后续OAC只允许加载v2 checkpoint。

## 50. 高速Actor-visited持续OAC首轮结果（2026-08-30）

已按§49合同完成20轮train-only terminal OAC。每轮使用当前Actor exact center加16对全秩antithetic
probe，DBM J50真实反馈全部进入Replay；Twin Critic:Actor更新比为`20:1`。不存在TD/bootstrap、target
Critic、entropy或解析DBM梯度；warm不进入Actor输入/loss，只作评价和未来two-center guard。每run由
internal-selection真实DBM mean选checkpoint，OOF不参与选择。

结果把Actor OOF teacher-gain recovery中位从`0.613`提高到`0.941`，胜warm比例中位从`0.667`提高到
`0.800`，回归比例中位从`0.333`降到`0.200`。每fold至少2/3 seed相对预训练改善；15 run中12个选择
round 20，另有19/13/0各一个，round 0回退证明checkpoint gate有效。正式独立重放全部通过且误差为0。

但是联合门仅通过2/4：主体mean improvement门与fold一致性门通过；14/15 run的warm-relative P05仍负，
所以tail门失败；fresh OOF局部Critic sign accuracy中位`0.698`，略低于`0.70`门。后续计划因此为：

1. 当前selected Actor只作为two-center proposal，不作为单中心控制命令；warm floor不得删除；
2. 不因主体recovery接近teacher就提前打开formal/test或闭环；先在train-only口径解决局部排序与尾部；
3. 若继续在线轮次，必须保留internal DBM checkpoint gate和round-0 fallback，并单列latest与selected；
4. 下一消融应改变Actor附近Replay的信息或保守更新口径，不能只延长相同20:1训练；Critic局部sign和
   warm-relative P05必须联合改善；
5. 600帧高reference失稳压力集继续隔离，30-episode数据量带来的fold差异仍需如实报告。

```text
outputs/mppi_proposal/highspeed_actor_visited_oac_20260830_v1/
```

当前资格：train-only机制改善成立，部署/formal/closed-loop资格未通过。

## 51. 高速状态扩张后的预训练与OAC复核（2026-08-30）

已把目标域数据从30 episode / 150 early contexts扩为120 episode / 600 early contexts。五档速度、六类
scenario各有4个独立episode，每条取step 0--4；实际速度中位`63.2 km/h`，`78.3%`位于40--100 km/h。
全部数据为train-only，formal validation/test与闭环未打开。600状态的129-candidate proximal teacher
聚合降低J50 `3.1386%`，600/600严格改善warm。

为公平比较规模，预训练固定近似optimizer-step预算：旧30-episode为Actor/Critic 500/240 epochs，
120-episode为125/60 epochs。5-fold x 3-seed的Actor OOF recovery中位为`0.573`，但仍有fold 2负迁移；
Twin Critic OOF Pearson中位`0.984`并通过初始化门。随后相同20轮OAC把selected OOF recovery提高到
中位`1.031`、最差run`0.363`，15/15 run的OOF mean均优于预训练；胜warm比例中位`0.808`。独立
checkpoint/replay/DBM复算通过，episode leakage为0。

裁决与下一步：扩大独立状态数量已被证明有价值，尤其改善OAC主体与最差run；但不能继续把“更多同类
数据”当作尾部唯一解。当前warm-relative P05中位仍`-10.1k`，Critic局部sign中位`0.691`，联合门仍
`2/4`。因此保留以下顺序：

1. 当前Actor只进入two-center候选，warm floor不变；
2. 下一分析优先定位expanded OOF中仍输warm的10.8%--31.7%状态，按速度xscenario、动作支撑边界和
   local-sign错误交叉分解；
3. 只在该归因显示覆盖缺口时再扩状态；若回归与覆盖无关，则改Actor风险/更新或候选guard，不再盲目
   增加episode；
4. 30/60/120严格嵌套学习曲线可作为记录性补充，但full-data OAC已足以放行“数据扩张有效、尾部未解”
   的机制结论；不得用小档曲线提前开启formal/test；
5. formal validation/test、闭环与单中心部署继续冻结。

权威产物：

```text
outputs/mppi_proposal/highspeed_actor_twin_critic_pretrain_expansion_e4_20260830_v2/
outputs/mppi_proposal/highspeed_actor_visited_oac_expansion_e4_20260830_v1/
```

## 52. Strong-search上界裁决与迭代蒸馏入口（2026-08-30）

120个独立train-only step-0状态的965-candidate多起点strong search已完成并独立全候选重放通过。
strong oracle相对warm降J `19.73%`，而旧proximal teacher只降`4.77%`；当前OAC Actor单次中心降
`7.98%--8.32%`，仅恢复strong headroom `40.4%--42.2%`。因此停止Actor优化的80%门未通过，不能
转入“Actor已到上限”分支。

同时，Actor不是错误路线：从warm/teacher/Actor起点完成相同搜索分别恢复strong headroom
`77.6%/95.5%/96.5%--97.0%`，106/120个最终winning chain来自三个Actor起点。Actor已经学到高价值
搜索初始化，但到搜索终点仍需约`2.13--2.23 sigma RMS`中位移动，缺口属于多步refinement，不是一次
前向的小精度校准。

后续计划调整为：

1. 从现有artifact提取每个Actor seed自己的6轮incumbent path，构造`一步中间/两步中间/最终终点`
   三档配对标签；禁止直接把跨起点全局argmin作为普通MSE目标；
2. 先在120状态做episode-grouped cross-fit，比较一步和两步迭代蒸馏能否保留真实DBM improvement；
   每一步都限制移动并保留stay/warm fallback；
3. 若中间标签在OOF可迁移，再把相同搜索path采集扩到全部600状态并做1--2轮Actor-visited重蒸馏；
4. 若连一步中间标签都不能OOF迁移，则把Actor定位为离线/在线candidate-search初始化，而不是继续扩
   网络或直接蒸馏远终点；
5. two-center、P05/worst、100 km/h尾部、formal/test封存规则不变。strong oracle仅是965个已评中心
   中的数值下界，不宣称全局最优。

权威产物：`outputs/mppi_proposal/highspeed_strong_search_oracle_20260830_v2/`。v1数值正确但未保存逐轮
incumbent path，只保留为上界诊断；迭代蒸馏必须使用v2。

## 53. 高速代价尺度与along/cross部署口径（2026-08-30）

§52的120个train-only独立状态已完成原始J50的固定时间索引along/cross正交分解并通过独立DBM重放。
warm聚合J中position占`87.12%`，position内部along/cross为`48.61%/51.39%`；40→100 km/h时
warm along RMS中位从`9.04`升至`24.68 m`，cross RMS从`10.05`升至`27.40 m`。因此高速J变大
对应真实预测轨迹误差和长距离积分效应，不作为普通数据尺度bug处理。

该审计同时发现当前OAC改善结构与strong-search headroom不同：三个OAC Actor的warm-relative收益约
`74%`来自vx、仅`26%`来自position；strong oracle则约`35%`来自vx、`64%`来自position，且along/cross
各贡献约`32%`。这把下一阶段目标收紧为：逐轮search-path蒸馏不仅要改善总J，还应恢复空间轨迹项，
并按`position-along/position-cross/vx`报告收益分解。只改善vx但不提高position恢复的模型不能解释为
逼近strong-search上限。

保持以下边界：正式J50及cost权重不变；输入继续使用训练split归一化，Critic继续使用标准化
`log1p(J)`；不按速度或warm cost重标任务；40/55/70/85/100 km/h逐层报告；two-center、P05/worst、
formal validation/test封存规则不变。任何along/cross重新加权都属于新的控制目标counterfactual，不能
与当前J50结果混报。

复现产物：`outputs/mppi_proposal/highspeed_along_cross_cost_20260830_v1/`。

## 54. 中间search-path蒸馏放行与600-state扩展（2026-08-30）

§52登记的120-state中间标签cross-fit已完成并独立验证。三个OOF OAC起点/状态合成360个样本，同状态
严格同fold；只使用显式anchor条件下的action-transition MSE，不用Critic或DBM梯度。第一轮search
标签恢复strong headroom `40.70%`，第二轮累计`63.34%`。

学习结果通过机制门与raw tail门：one-step保留其标签收益`92.8%--97.7%`，一次前向直接预测两步终点
保留`93.7%--95.1%`，两网络级联保留`95.3%--98.1%`；全部seed的P05 gain为正，360个OOF样本零
回归。direct/cascade分别恢复strong headroom约`59.4%--60.2%`和`60.4%--62.1%`，相对warm降J约
`15.0%--15.3%`，明显高于原OAC start的`8.10%`。五个速度层均为正，且两步收益约79%来自position，
along/cross近似对称，说明改进进入空间轨迹项而非只修vx。

裁决：放行600-state train-only扩展，但不直接进入formal/闭环。执行合同：

1. 对§11.129全部600个context、三个OOF OAC起点仅采前两轮确定性search path；保留start/path1/path2、
   逐轮真实J50和along/cross/vx分项；不再为本阶段支付完整6轮965-candidate预算；
2. 保持原120-episode episode-grouped 5-fold x 3-seed，不允许同episode不同control step跨fold；
3. 主臂为一次前向`direct_two_step`，保留`stage1+stage2 cascade`作为100 km/h稳定性对照；one-step为
   机制基线；
4. 扩展成功门：OOF target-gain recovery至少2/3 seed≥0.70，aggregate warm-relative J优于原OAC，
   P05≥0、100 km/h P05≥0，position收益占比不得退回原OAC约26%的水平；
5. 若600-state direct与cascade都过门，优先一次前向direct；若只有cascade在later-step/100 km/h过门，
   接受两次轻量Actor前向。若两者均掉回低恢复，则定位为step-0小子集偏差，不扩网络、不消费formal；
6. two-center warm guard、formal validation/test封存以及闭环后置规则不变。

权威产物：`outputs/mppi_proposal/highspeed_iterative_path_distillation_20260830_v1/`。

## 55. 全600-context放大结果与部署前置项（2026-08-30）

600-context、三个OOF OAC起点的前两轮path已完成：每context 195候选，共117,000次DBM评价，独立全候选
重放误差0、零单调性/基线违规。path2 teacher将OAC start相对warm的`3.04%`改善扩到`8.48%`。

原episode-grouped 5-fold x 3-seed复训结果通过§54全部主门：

- direct-two-step：target-gain recovery `94.79%--95.23%`，相对warm降J `8.19%--8.22%`，总体与
  100 km/h P05均为正，position收益占`85.69%--87.01%`；
- cascade：recovery `93.06%--93.97%`，相对warm降J `8.10%--8.15%`，各tail门也通过，但没有
  超过direct；
- direct raw回归率仍为`1.94%--2.78%`、worst start-relative gain约`-25k--34k`。two-center warm
  guard后逐状态warm-relative下界由构造守住，聚合降J约`8.35%--8.40%`。

因此主Actor定为一次前向direct-two-step，cascade只保留为机制/高速度对照。下一执行顺序：

1. 生成全600 direct raw回归tail manifest，按速度、scenario、control step、clip和along/cross/vx
   分解，确认2%--3%回归是否集中在可观测层；不为修tail重训主Actor；
2. 将direct checkpoint接入现有two-center候选合同，单测候选集中warm和Actor两个未加噪中心均存在，
   MPPI最终选择不劣于warm候选的真实DBM cost；
3. 做train-only短时递推smoke，检查warm-shift、控制平滑和Actor重复调用；
4. 以上通过后再申请固定DBM短闭环A/B；formal validation/test仍不提前消费。

主线不再优先做Critic梯度、更多Actor结构扫描或更长search teacher。权威产物：
`outputs/mppi_proposal/highspeed_iterative_path_distillation_expansion_20260830_v1/`。

## 56. Search蒸馏结果的路线纠正与Query兼容OAC下一步（2026-08-30）

§55的数值结论保留，但“direct-two-step直接进入部署准备、主线不再回Critic”的路线结论被本节覆盖。
原因是最终动力学将切换到Query，算法不能要求DBM解析梯度；同时纯BC蒸馏不会在Actor访问新动作后继续
校正价值面。正式算法仍为：`Actor候选 -> DBM/Query标量cost -> Replay -> 持续Twin Critic -> Actor`。

direct-two-step的正确定位是：

- 证明当前Actor能够保留约`95%`的两轮局部search标签收益，网络容量不是第一阻塞；
- 给出当前OAC可以追赶的容量参考：OAC start相对warm约`3.04%`，两轮teacher约`8.48%`，蒸馏Actor
  约`8.2%`；
- 指出下一轮Replay应包含“第一次response后重定位”的高信息量局部action--cost，而不是只增加同分布
  Gaussian行；
- 不允许用蒸馏checkpoint替代Critic资格，也不允许据此提前打开formal/test或闭环。

低速经验在高速域只迁移算法组织，不迁移性能数字：冻结/单次Critic永久关闭；K16是Actor microstep
效率拐点、K32边际；2倍同分布刷新无收益；matched真DBM梯度只增加`1.88pp`，Critic不是唯一瓶颈；
共享Actor参数冲突会让正确逐状态动作梯度进入网络后发生约40%的回归。因此下一步不能只扩Critic，也
不能只做BC，而应验证search数据能否先改善Critic，再经持续OAC传给Actor。

下一主实验注册为`HIGHSPEED_SEARCH_INFORMED_REPLAY_OAC_AB`：

1. 从同一§51 Actor+Twin-Critic预训练checkpoint开始，不恢复已经走过不同Replay轨迹的terminal OAC；
   固定相同episode folds、seed、rollout、Critic/Actor update、LR、trust和Replay采样合同。
2. `R0`使用两组均围绕当前Actor center的匹配候选；`R1`第一组相同，第二组围绕第一组真实cost选出的
   incumbent重定位。建议统一为每context `1+32+32=65`个唯一中心。
3. DBM只返回J50；所有好/坏候选写Replay。禁止DBM梯度、禁止把best endpoint直接作为Actor MSE、禁止
   删除高cost动作。
4. 先冻结Actor做等预算Critic吸收，报告fresh path排序、best regret、局部sign和twin disagreement；
   随后解除Actor冻结，继续Actor-visited采集与Critic/Actor交错更新。
5. Actor评价只用确定性center：warm-relative聚合降J、胜率、median/P05/worst、strong-headroom
   recovery、五档速度和position/along/cross/vx分解；two-center结果另列，不替代raw指标。
6. 若R1提高Critic排序且Actor优于R0，才扩大训练并进行Query/PyTorch、Query/ONNX黑盒cost接口复核；
   若Critic提高但Actor不动，转共享参数冲突/PCGrad-CAGrad或regime head；若Critic不提高，先修
   value/ranking/pair-delta学习，Actor保持冻结。

为保持单变量归因，首轮两臂均沿用高速当前`20 Critic : 1 Actor`、20 rounds，不同时引入低速K16。
只有R1通过，才把K16作为下一轮独立更新预算A/B。fold数据边界同样固定：只有fit episode进入Replay和
梯度更新，internal-selection只选checkpoint，OOF episode只做fresh评价；禁止把现有600-state path
artifact不分fold地整体灌入Critic。

在本A/B完成前，formal validation/test、MPPI wrapper、短闭环和单中心部署继续封存。

## 57. Search-informed Replay OAC结果与Critic-only裁决（2026-08-30）

§56正式A/B已完成并独立验证。R1重定位显著提高真实候选质量：20轮平均best-of-bank gain相对R0的
配对增量中位`+2145.59`，15/15 run为正；第二阶段自身增量中位`+2161.81`。因此search信息源成立。

但最终Critic在fresh两轮65-candidate bank上的增量接近零：Pearson/sign/bank-recovery配对中位分别
`+0.0046/+0.0003/+0.0087`，其中bank recovery配对均值为负。0.05sigma local sign配对中位还
`-0.0026`。Actor只有小量传导：recovery `+1.07pp`、warm mean gain `+64.36`、P05 `+124.7`；全部run
的raw warm-relative P05仍为负。

裁决为`SEARCH_INFORMED_BANK_STRONG_CRITIC_TRANSFER_WEAK_ACTOR_SMALL_GAIN`。下一步执行顺序：

1. Actor冻结，直接复用R1最终Replay和已保存fresh path bank；不新增DBM rollout。
2. 每个fold/seed从当前Twin Critic与optimizer状态继续训练，至少记录`0/400/1600`额外update预算点；
   checkpoint选择只用fit/internal，fresh OOF path bank只评价不选点。
3. 主门是fresh path Pearson、sign、bank recovery与best regret是否随预算单调改善，同时0.05sigma local
   sign不退化；不得只看train loss。
4. 若Critic-only预算曲线明显上升，说明R1数据有效但20:1更新不足，再恢复Actor并单独比较当前1次与
   低速K16更新合同。
5. 若1600 update仍平台，则关闭“纯预算不足”，在同一Replay上做value+ranking对照pair-delta/相对
   value监督；Actor继续冻结。只有Critic fresh门改善后才能恢复持续OAC。

formal validation/test、Query切换、wrapper与闭环继续封存。

## 58. Search Replay的Critic吸收预算通过，下一门转为Actor传导（2026-08-30）

已完成§57的冻结Actor预算曲线。复用R1最终Replay和fresh OOF两阶段bank，零新rollout、零DBM解析
梯度；每Twin额外训练`0/400/1600`次。两阶段path的Pearson中位`0.639 -> 0.720 -> 0.786`，sign
`0.792 -> 0.817 -> 0.865`，bank recovery `0.772 -> 0.830 -> 0.920`，其P05由`0.528`升至
`0.848`。0.05sigma local sign也由`0.713`升至`0.761`。path三项在0到1600上全部15/15 run正改善。

独立重载15个最终checkpoint复算误差为0，episode leakage为0，hash链完整。裁决：R1新Replay不是
不可学；原20轮`20 Critic : 1 Actor`随65候选/状态/轮持续增长时，Critic更新预算不足。当前阻塞从
“Replay信息是否有效”收窄为“更成熟Critic能否通过持续OAC转成共享Actor收益”。

下一步合同：

1. 从同一R1最终Actor开始做配对传导A/B：一臂使用原R1 Critic，另一臂使用本节`+1600` Critic；
2. 两臂继续相同search-informed Actor-visited采集、全部候选写Replay、Twin Critic持续更新、每轮仅
   1次Actor更新，禁止冻结Critic后连续推Actor；
3. 主看OOF Actor相对warm聚合收益、胜warm比例、P05/worst和五档速度；Critic继续报告fresh path/local
   指标，确认Actor移动后没有再次掉出已学习区域；
4. 若成熟Critic显著提高Actor且不恶化tail，再调整稳态Critic更新频率；若Critic指标高但Actor仍不
   传导，阻塞落到共享Actor参数冲突，才进入PCGrad/CAGrad/regime head；
5. 低速K16不在本轮引入。它是Actor microstep效率结论，不是Critic吸收补丁；必须与Critic readiness
   分开验证。

formal validation/test、Query切换、MPPI wrapper与闭环继续封存。

## 59. 成熟Critic未加速Actor：下一阻塞为共享Actor更新几何（2026-08-31）

§58配对传导A/B已完成：相同R1 Actor/Replay起点，只替换原Critic或`+1600` Critic，然后两臂都继续
10轮`search-recentered65 + 20 Critic : 1 Actor`持续交互。成熟Critic在新Actor附近仍把fresh path
sign提高中位`+6.81pp`、bank recovery提高`+12.23pp`，15/15正改善；但Actor teacher-recovery配对
中位`-0.24pp`、均值`-2.51pp`，仅5/15 run提高，tail仍不通过。

两臂继续训练本身都15/15优于共同起点，因此持续OAC有效；失败仅指“额外Critic readiness没有变成
额外Actor收益”。成熟Critic臂的每轮Actor输出步长均值比原臂小约8.6%（`0.01311` vs `0.01434sigma`），
trust投影次数也从30/150降到7/150，而raw-J ESS相同。当前剩余分支是：

1. Critic更准但Actor梯度幅值/Adam状态偏保守；
2. Critic给出的各状态动作方向更准，但映射到共享Actor参数后仍相互抵消或负迁移。

下一唯一诊断为matched-output-step梯度几何审计：固定相同Actor、batch和输出RMS，比较两套Critic的
参数梯度、速度/场景分组cosine/cancellation，并以真实DBM标量cost只作更新后评价。若等步长成熟Critic
恢复优势，修Actor步长/optimizer；若仍无优势，进入持续Critic兼容的speed-group PCGrad/CAGrad短A/B。
低速K16、更多Critic-only update、formal/test、Query、wrapper和闭环继续后置。

## 60. 匹配输出步长审计完成：放行受控K扫描（2026-08-31）

同起点、同fit batch下，原Critic与`+1600` Critic的Actor参数梯度cosine中位仅`0.869`，但把两套更新
严格校准到相同的`0.005/0.01/0.02sigma`输出RMS后，成熟Critic在raw-SGD或保存Adam路径均无稳定OOF
优势。六个method/radius组合中配对mean delta中位仅约`-1.34`到`+5.63`，胜run为`7/15`到`10/15`，
没有一致方向。由此排除“成熟Critic只是走得小，所以Actor没受益”为主解释。

两套方向仍具有继续优化空间：saved-Adam `0.02sigma`下两臂均15/15 OOF mean gain为正，且mean收益随
半径增大。因此下一实验正式改为Actor预算K扫描，而不是再训Critic：

1. 使用当前持续训练Critic与search-informed Replay，比较`K=1/4/8/16`；
2. rollout、Critic update、初始Actor、batch与随机流配对；
3. 每轮多个Actor microstep后的**累计**输出RMS统一投影到同一trust上限，禁止每步各放`0.02sigma`；
4. 主门为OOF mean/median、P05/worst、胜warm比例与selected/latest差；
5. 若同累计trust下K提升，再单独探索更大trust；若K无效，进入speed-group PCGrad/CAGrad。

formal/test、Query、wrapper和闭环继续封存。

## 61. 受控Actor K扫描完成：停止增加microstep，转共享梯度冲突审计（2026-08-31）

已按§60固定单轮累计Actor输出移动为`0.02 sigma`。fold0先扫`K=1/4/8/16`，K4/K8均无优势；正式
端点比较扩展为`5 fold x 3 seed`的K1/K16。两臂每轮使用相同65个search-recentered候选、相同20次
Critic update/每Twin、相同初始Actor/Replay/Critic，只有Actor microstep K不同。

K16相对K1的OOF paired结果为：teacher-recovery中位`+0.84pp`（10/15正）、共同起点mean gain
`+51.18 J`（10/15正）、共同起点P05 `+191.18 J`（12/15正）。但最终部署相关的warm-relative P05
配对中位为`-236.88 J`（仅6/15正），胜warm比例配对中位`-0.83pp`（仅3/15正）。K16有边际主体
收益，但没有稳定改善单中心相对warm的tail。

固定trust确实生效：K1/K16投影后逐轮RMS中位为`0.019999/0.020000 sigma`，全部150次零违规；K16
投影前累计移动中位`0.195 sigma`，约为K1 `0.014 sigma`的14倍。大量额外microstep最终只改变小位移
球内的优化路径，收益远小于计算增加，故不再扫描K32或继续用K补偿Actor上限。

执行顺序改为：

1. K1作为成本基线，先对相同Critic目标的每速度/场景Actor参数梯度做零rollout冲突矩阵，报告
   cosine、norm、sum-gradient cancellation和各组真实收益占比；
2. 仅当审计确认跨速度/场景稳定负冲突时，比较普通mean、PCGrad、CAGrad/MGDA；保持相同K1、累计
   `0.02 sigma`trust、Replay/rollout/Critic预算与DBM checkpoint gate；
3. 若冲突处理也只有边际收益，则共享Actor更新几何分支关闭，主线回到可部署two-center guard及
   Query黑盒cost接口准备，不再继续Actor microstep/loss扫描；
4. formal validation/test、MPPI wrapper和闭环在本机制阶段仍封存。

权威产物：`outputs/mppi_proposal/highspeed_actor_k1_k16_scan_20260831_v1/`；独立validator对30个
checkpoint、全部selected center和local bank完成DBM重放，最大误差0、episode leakage 0。

## 62. 90轮预算与步长边界：K16加速中程，`0.06 sigma`机制领先（2026-08-31）

§61的五轮停止裁决已被长曲线覆盖，原结果只保留为短预算pilot。fold0三seed下，把K1/K16延长到
90轮并固定每轮精确累计输出移动`0.02 sigma`后，两者相对共同起点mean gain都超过`10200 J`，P05
均转正，胜warm比例约`0.994`；第90轮仍被全部run选中，训练预算明显不足。K16在第10--60轮比K1快
约`124--465 J`，但第90轮mean差收敛到`+26 J`，故当前把K16解释为优化加速器，而非已经证明能提高
最终mean上限。

固定K16/90轮后，精确输出步长`0.02/0.04/0.06 sigma`的selected OOF mean gain分别为
`10283/12758/13702 J`，P05为`298/1031/1193 J`，warm-relative P05为`3026/3996/4238 J`。
这证明步长和轮数都是高速Actor训练的真实约束，`0.06`为当前机制领先臂。不过该实现会在后期把小于
目标的自然更新重新放大，且只覆盖fold0；`0.06`还有一个seed的1/120状态输warm，而`0.04`为3/3 seed
全胜warm。

下一步合同因此改为：保持K16与90轮，使用cap-only `0.06 sigma`（只截大步，不放大小步）并加入后期
LR/step衰减；先过fold0 mean/P05/worst/胜warm门，再扩完整fold。PCGrad/CAGrad后置，因为长预算与
trust已解释五轮pilot的主体不足。formal/test、Query、wrapper和闭环继续封存，且
`recovery>1`不得解释为超过理论最优，它只以当前proximal teacher为分母。

权威汇总：`outputs/mppi_proposal/highspeed_actor_k16_90round_trust_scan_fold0_20260831_v1/`；独立判定为
`HIGHSPEED_ACTOR_TRUST_CURVE_INDEPENDENT_SUMMARY_PASS`。

## 63. cap-only `0.06 sigma` + 后段LR衰减完成完整OOF（2026-08-31）

§62的下一步已经完成。最终训练合同为K16、160轮、每轮Actor输出移动仅做`0.06 sigma RMS`上限截断、
exploration在前90轮退火、Actor LR在第120--160轮从`2e-5`余弦衰减到`5e-6`。fold0消融确认：

- cap-only 90轮与exact 90轮mean几乎相同（差约`-12.6 J`），但不再把自然小步放大，strict worst更好；
- 160轮常数LR相对90轮增加约`+365 J` mean，轮数仍有价值；
- 后段LR衰减相对常数LR只少约`25 J` mean，却把warm P05提高约`159 J`并降低late回落，故作为领先合同。

完整`5 fold x 3 seed` selected OOF结果：Actor mean J约`169579.5`，预训练Actor/warm/proximal teacher的
对应mean J约为`185123.4/191104.2/185106.1`；相对预训练Actor mean gain平均`15543.9 J`，P05
平均`1915.7 J`且最差run仍为`+1032.4 J`；warm-relative mean/P05平均为`+21524.7/+3293.8 J`，
15/15 run的warm P05为正。胜warm比例跨run平均98.61%，最差run仍有95.83%。五档速度的warm P05也
全部为正。

机制门通过，但strict worst未通过无条件单中心安全：最差run中的最差状态可比warm高`34743.9 J`，
平均约1.39%状态输warm。因而下一阶段必须保留warm/current与Actor center双候选，two-center guard不能
降级。selected轮为97--160、中位154，训练取内部selected checkpoint，不强制最后一轮。

执行边界更新：

1. 本合同取代exact-forcing `0.06`，作为高速度Actor训练的机制基线；
2. 允许下一阶段接Query黑盒cost接口或验证two-center wrapper传导，但两者必须单变量进行；
3. Actor center离线主指标继续用相对warm的mean/median/P05/worst/胜率，wrapper指标不得混入center训练门；
4. two-center guard要求warm候选和Actor候选同时进入真实模型rollout，禁止Actor center替换warm后删除退路；
5. formal validation/test和闭环仍封存，不能把本节train-only episode-grouped OOF当部署结论；
6. 如进入Query对照，禁止依赖DBM解析梯度，保持相同Actor步长/LR/Replay预算与真实checkpoint gate。

权威汇总：`outputs/mppi_proposal/highspeed_actor_k16_cap_schedule_20260831_v1/`；两层validator均通过，
扩展12 checkpoint的Actor/DBM重放误差为0，合并判定为
`HIGHSPEED_ACTOR_CAP_SCHEDULE_INDEPENDENT_SUMMARY_PASS`。

## 64. 冻结高速度Actor训练合同，转入two-center传导（2026-08-31）

正式冻结K16/160轮/cap-only `0.06 sigma`/120轮后LR衰减合同。选择K16的理由是其完整OOF证据和当前
最优实测指标，不是已证明K16提高最终上限；K1/K16在`0.02 sigma`的90轮fold0 mean终点只差约`26 J`。
不再补K1同合同对照，也不继续扫描K32、训练轮数、LR、trust、Critic loss或Actor结构。

下一阶段顺序：

1. 现有15个OOF Actor上比较warm-only-64与warm-32 + Actor-32；总rollout预算、噪声和随机种子严格配对；
2. direct center指标与MPPI wrapper指标分开；候选集中保留warm不等于最终softmax center有硬下界，需
   单列unguarded结果，并用同一次模型评价实现最终warm fallback，guard后违规必须为0；
3. 通过mean/median/P05/分速度wrapper门后，才按固定合同训练全训练集3-seed候选Actor；保留internal
   selection carve-out，formal validation/test继续封存；
4. Actor/candidate bank冻结后做Query PyTorch/ONNX shadow：先验证Torch/ONNX等价，再用DBM离线真值审计
   排序、regret和guard选择；Query路径禁止DBM解析梯度；
5. 最后一次性使用formal validation，并依次做固定DBM短闭环与Query shadow/闭环。

固定合同参数与完整边界见review §11.142。当前执行项为two-center固定64预算OOF A/B。

## 65. 当前主问题改为单中心DBM剩余headroom（2026-08-31）

用户明确要求当前不讨论MPPI如何采样或执行，只判断策略网络输出中心本身的质量，以及是否还值得继续
DBM探索。因此§64的two-center/Query/闭环顺序保留为后续计划，但当前暂停。

同状态零rollout连接显示：旧120状态strong-search reference mean J为`154546.1`，当前最终Actor三seed
平均mean J为`150894.8`，低约`3651.3 J`。旧reference是965-query、五个旧起点的best-found，不是理论
最优；它已被当前Actor超过，无法继续充当headroom上界。

当前唯一下一实验是Actor-centered post-search audit：冻结网络，以当前OOF Actor center为起点，在同
120状态做`65/193/385/965`嵌套确定性DBM搜索，直接统计相对Actor center的剩余cost降低比例和移动量。
若中位降低小于1%、聚合降低小于2%且速度层一致，则停止DBM全局探索；若聚合降低至少5%且多数状态
稳定改善，则继续Actor-visited Replay/OAC；若只有少数tail有明显headroom，只做定向hard-state探索。

在该裁决完成前，不进行MPPI wrapper、two-center采样分配、Query切换、formal validation/test或闭环。

## 66. 单中心DBM数值headroom裁决完成：停止全量Actor探索（2026-08-31）

§65的上限问题已用更强合同完成，而不是继续引用已失效的旧965-query reference。120个train-only平衡
状态上，从warm/teacher/旧oracle/最终Actor三seed/随机起点联合出发，使用物理动作盒内的多学习率投影
DBM autograd优化到800步，并接三套正交方向的无梯度fine polish。400→800步mean best只下降`0.11 J`，
无梯度polish再改善`0.025%`，最终numerical best-found mean J为`150181.7`。

最终固定Actor三seed期望mean J为`150894.8`，相对数值参考聚合差距`0.473%`（bootstrap 95% CI
`[0.377%,0.606%]`），逐状态差距中位`0.449%`；Actor已经回收warm到该参考之间`98.32%`的headroom。
79.2%状态在1%内、93.3%在2%内，但最差个例仍有8.88%差距。独立DBM replay误差为0，判定
`HIGHSPEED_FINAL_ACTOR_NUMERICAL_ORACLE_REPLAY_PASS`。

因此§65停止门已通过：暂停全量Actor/Critic超参、轮数、K、trust、结构和一般数据覆盖探索。若继续DBM
研究，只允许针对超过2%的少数tail状态做定向分析，并先检验该tail是否具有跨episode可学规律；不能为
约0.47%的主体聚合空间继续扩大通用训练。这里的best-found不是数学认证全局最优，DBM梯度也仅用于离线
上限审计。formal validation/test、Query、MPPI与闭环结论仍未获得授权，后续是否恢复§64传导计划由
用户另行决定。

若后续批准tail分析，样本合同固定为`analysis.json/tail_over_2pct`中的8帧（7个40 km/h、1个70 km/h），
不得重新消费formal/test或按结果修改阈值；先做跨episode可学性与共同机制检查，再决定是否值得定向补
Replay。若无一致规律，接受warm/current等后续执行层保护，而不为个例重启全量Actor训练。

权威产物：`outputs/mppi_proposal/highspeed_final_actor_numerical_oracle_20260831_v1/`。

## 67. Query切换前训练域审计：先过高速度数值环境Q0门（2026-08-31）

已回查冻结Query的实际训练数据、checkpoint、ONNX和当前600状态高速度replay。车辆/时间/动作接口合同
一致，但训练实际`vx`最高仅`3.098 m/s`，当前为`7.796--30.626 m/s`且100%越界；固定训练统计下的
context-vx/history-dx/warm-nominal-dx绝对z-score中位均约26、P95约43--44。当前`vy`有50.5%超训练
范围且Query不显式观测`vy`。此外高速度replay只取每episode前5步，250步history几乎全是step-0恒速
prime，历史动作精确为0的比例99.2%，与训练时完整12.5 s历史不同。

因此下一步不得直接假定旧Query在高速度域是已验证环境，也不需要先比较其与DBM/真实车辆误差。先执行
Q0自洽门：冻结checkpoint和输入合同，检查高速度PyTorch rollout、轨迹与cost有限且确定性可复算；同一
输入做PyTorch/ONNX高速度等价；分开报告early-prime与mature-history。只有Q0通过，才允许用Query真实
forward重新生成/标注Replay并按既有OAC合同训练；不得把DBM cost、DBM梯度或旧DBM标签混入Query环境
目标。Q0之前formal validation/test与闭环继续封存。

权威审计：`outputs/query_mppi/highspeed_query_domain_audit_20260831_v3/`，独立validator为PASS；完整条件
与边界见review §11.145。
