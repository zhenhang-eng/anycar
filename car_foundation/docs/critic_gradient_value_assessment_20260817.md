# Critic Action Gradient 路线价值评估（2026-08-17）

> 结论等级：`COORDINATE_CRITIC_FAIL_CLOSE_GRADIENT_MAINLINE`
> 依据：`car_foundation/docs/mppi_sampling_center_review_archive_20260812.md` §11.14–§11.37，及
> `car_foundation/docs/mppi_sampling_center_review_20260812.md` §11.38（六轮统一
> grouped CV、三轮审计、两轮零成本诊断，约200个可溯训练run）及“§11.50 Critic最后坐标机会”（最终全秩
> sensitivity/DCT坐标配对实验，18个训练run）

---

## 1. 一句话结论

**Critic对action gradient并非"完全学不会"，而是分层失败：easy与H_ONLY层可学可迁移、
且已经学到可用；但约30%的g0-hard状态（JOINT+G0_ONLY）的梯度方向在当前数据+当前输入下
跨状态不可预测——"补数据/换输入/换目标/换结构"四条投入途径已逐一证伪。**
Actor因gate六轮从未通过而全程冻结；残值在H与easy层排序，不在action gradient。

---

## 2. 判定标准（预注册，六轮统一）

| Gate | 门槛 | 六轮通过记录 |
| --- | --- | --- |
| fresh cosine median | ≥ 0.70 | 最好单run 0.74，分层后g0-hard全部 < 0.2 |
| fresh cosine P10 | ≥ 0 | 恒在 −0.92 ~ −0.98 |
| norm ratio | [0.5, 2.0] | 可满足（非瓶颈） |
| fold/seed 联合 | ≥4/5 fold 且 ≥2/3 seed | **0/5 fold、0/3 seed，无一例外** |

---

## 3. 证据一：四类优化手段的投入与回报

### 3.1 训练技巧与工程修复——已穷尽

| 尝试 | 结果 | 含义 |
| --- | --- | --- |
| 两层范数坍缩定位与修复（15.7→4.9→0.12，cosine自锁） | 修复后P10仍 −0.92~−0.96 | 尾部问题不在loss（§11.14-11.16） |
| action坐标/Jacobian/角度wrap排查 | 全部排除 | 非工程bug（§11.10, 11.19） |
| 输入语义清理三臂×3seed | 全0/3 gate，P10不变 | 输入清洗关闭（§11.26） |
| 16-knot设计、网络容量（3次tiny overfit 0.97/0.958/0.958） | seen全部可拟合 | **容量不是瓶颈，加大网络无收益**（§11.16/11.34/11.37） |

### 3.2 结构改造——天花板0.7量级

| 尝试 | 结果 | 含义 |
| --- | --- | --- |
| Structured Local-Q（显式带符号Hessian，4臂×3seed） | 最好D+R2 median 0.715，0/3 gate | 结构改造够不到门（§11.22） |
| state×action显式双线性交互encoder | 只救H_ONLY（0.69→0.88，不稳定），g0-hard无效 | 交互结构非主因（§11.29） |
| 显式输入三臂E1/E2（真实六维状态skip / 纯Markov充分输入，45run严格配对） | G0_ONLY仍 −0.74/−0.84，配对+0.06/−0.02 | **已记录数据内不存在可补的缺失输入**（§11.37） |

### 3.3 数据投入——最大单笔定向投入，边际收益为负

| 尝试 | 结果 | 含义 |
| --- | --- | --- |
| 定向probe采集：254,100次DBM rollout，54状态×77位置 | seen **+0.66**，unseen **−0.10**，未接触230帧配对无改善 | 记忆不泛化——**钱买不到泛化**（§11.26-11.27） |
| 训练配比1:1:1 + hard-validation checkpoint | JOINT配对+0.45（有效）但G0_ONLY 15/15仍负（−0.81） | 配比只救一层，另一层纹丝不动（§11.31） |

### 3.4 监督目标重设——换目标无用

| 尝试 | 结果 | 含义 |
| --- | --- | --- |
| 相对value-delta（真标量Q+autograd，修复对齐bug后30run） | seen收敛0.67，heldout零信息（corr≈−0.1、sign 0.40**低于随机**） | 连续value目标也学不出heldout结构（§11.35） |

---

## 4. 证据二：结构性诊断——为什么剩下的部分买不到

| # | 诊断 | 关键数字 | 含义 | § |
| --- | --- | --- | --- | --- |
| 1 | 近邻翻转率 | **43%**的kNN近邻梯度反向；翻转对\|cos\|中位**0.838**（强反向非噪声）；跨范数四分位持平 | 近邻状态给相反标签——梯度场在当前状态空间不可插值 | 11.30 |
| 2 | 可观测分离性 | 速度/曲率/clip/scenario/steering等特征AUC **0.53**（随机0.5）；label-aware聚类仅0.58 | 无任何可观测规则可区分方向 | 11.30 |
| 3 | 距离-翻转曲线 | 最近桶(0.3)→最远桶(2.0)翻转率0.43→0.46**平坦** | 盲目加密状态采样无收益依据 | 11.36 |
| 4 | local oracle vs 固定KNN | k=5邻域内**90%**存在正确标签（oracle 0.92），固定KNN **−0.26** | 正确答案在邻域里，但无机制能选出 | 11.36 |
| 5 | 缺失条件变量排查 | dbm(14项)/cost(6项)/mppi(15项)参数全部episode**恒定** | 不存在"参数变了没输入"的缺失变量 | 11.35 |
| 6 | g0 vs H反事实归因 | H隔离项**0.99**，g0隔离项**−0.09**（JOINT层） | 能学的（H）已学到；学不到的（g0锚点）是独立阻塞 | 11.27.1 |
| 7 | 600状态逐cost项autograd归因 | 单项归因翻转/同向**68.0%/4.3%**；position解释全部翻转的**59.6%**；机制族**83.2%/41.7%** | position复合项是主导直接来源，多项对消放大尾部；不是标签、FD或autograd故障 | 11.41-11.42 |

第7项的口径必须保持精确：position占可归因翻转的149/170=87.6%，但占全部250个翻转对为
149/250=59.6%，不能写成“全部翻转的88%”。FD交叉验证cosine中位0.997/P10 0.988；默认
阈值在3x3x3扫描下结论稳定。60-episode有放回cluster bootstrap的95% CI为：单项归因翻转
[60.0%,75.5%]、同向[2.0%,7.0%]，机制族翻转[77.1%,89.1%]、同向[32.8%,50.7%]。
复现入口为`analyze_mppi_cost_term_flip_attribution.py`及
`cost_term_flip_attribution_20260817_v1` format-v2 artifact。

---

## 5. 投入产出总账

| 项 | 数值 |
| --- | --- |
| 实验轮次 | 6轮统一grouped CV + 3轮审计 + 2轮零成本诊断 |
| 训练run | ~200个（checkpoint/label全hash可溯） |
| 新增rollout | 254,100次定向DBM rollout（已消费） |
| **Gate通过记录** | **0/5 fold、0/3 seed，六轮无一例外** |
| 过程纠错 | 2次结论作废（value v1对齐bug等）均标记INVALIDATED并重跑，修正后结论不变 |

### 已收割的残值（这部分不要丢）

| 产出 | 数值 | 用途 |
| --- | --- | --- |
| easy层Critic | fresh cosine 0.85~0.93、strict-20 control 0.63+ | 离线候选粗排序器（§11.38新定位） |
| H跨状态迁移 | centered响应0.83~0.99、H隔离反事实0.99 | 局部曲面几何可用 |
| 部署路径改进 | center+H代数外推0.655 vs 逐点重估0.332 | 评估/部署立即受益，无需重训 |

---

## 6. 剩余解释与路线决策

零成本检验已穷尽后，g0-hard层不可预测只剩三个解释，且**都需要新增信息才能区分**：

1. 状态覆盖密度（§11.36的平坦结论仅在当前度量下成立，未关闭）；
2. 函数内在复杂度（当前采样密度下本质不可从有限邻居插值）；
3. 尚未记录的条件变量（不在任何现有数组中）。

**路线决策（§11.38，2026-08-17）**：Critic-gradient主线降级；Critic保留为离线候选粗排序器，
不再承担Actor gradient provider；主线转向**近端Offline Search Distillation**（零rollout
coherence → 小预算曲线 → 近端蒸馏迭代 → 部署传导四步合同）。默认/部署Actor继续冻结，
formal validation/test继续封存。

§11.41--§11.42使这一决策从“模型在hard层不泛化”推进到“目标几何存在明确机制”：position
平方误差本身对轨迹位置平滑，但经过非线性DBM、有限时域与参考几何映射到action空间后，近邻
action-gradient容易分支翻向。Huber或s-d/Frenet只能作为新cost定义的counterfactual，不能
宣称必然修复，也不是当前Search路线的前置条件。Search直接比较完整`J(s,a)`，避免把脆弱的
逐项梯度符号作为监督目标。

---

## 7. 证据索引（可溯）

| 主题 | 章节 | 关键产物 |
| --- | --- | --- |
| norm坍缩与训练消融 | §11.14-11.16 | direct_critic_fresh_fd_* |
| repeat对action-location审计 | §11.20 | cross-anchor对照 |
| Structured Local-Q | §11.21-11.22 | structured_local_q_full_20260814_v1 |
| chord几何/Local-H oracle/g0审计 | §11.23-11.24 | chord_geometry_local_oracle_*、g0_learnability_audit_* |
| 三轨（语义清理/定向probe/重训） | §11.25-11.26 | targeted_local_response_labels_*、targeted_semantic_* |
| targeted grouped CV+反事实分解 | §11.27-11.27.1 | targeted_grouped_cv_20260814_v1 |
| g0专项CV（含G0_ONLY首测） | §11.28-11.29 | g0_grouped_cv_20260814_v1 |
| 平衡重跑+结构诊断 | §11.30-11.31 | g0_balanced_grouped_cv_*、g0_reversal_separability_* |
| value-delta v1作废/v2修复 | §11.32-11.35 | g0_value_delta_cv_20260817_v1(作废)/v2 |
| 邻域诊断 | §11.36 | g0_neighbor_oracle_20260817_v1 |
| 显式输入三臂 | §11.37 | g0_explicit_input_cv_20260817_v1 |
| 路线决策 | §11.38 | — |
| 逐cost项翻转归因与稳健性 | §11.41-11.42 | cost_term_flip_attribution_20260817_v1 |

---

## 8. 2026-08-18 最终坐标实验与主线关闭

为排除“原始16维action坐标condition差，导致Critic梯度失败”这一最后假设，完成了严格配对的
三臂解释：原物理坐标A、同一A checkpoint仅做`B^T`坐标变换的AT、以及真正以`a=Bz`重训的
Z。`B`使用fold-train-only sensitivity、固定MPPI sigma和每通道完整DCT，保持16维满秩；训练
沿用修复后的scalar-Q/value-delta v2，不改变状态输入、loss、数据、优化器或checkpoint选择。
协议为3 episode-grouped folds × 3 seeds，formal validation/test未加载，Actor冻结。

最终严格pooled结果：

| 指标 | A | AT（仅换度量） | Z（重训） |
| --- | --- | --- | --- |
| coordinate cosine P10（3 seed） | -0.972/-0.968/-0.967 | -0.832/-0.833/-0.810 | -0.612/-0.694/-0.805 |
| physical-step cosine P10 | 同上 | -0.250/-0.256/-0.240 | -0.316/-0.349/-0.393 |
| early-steering P10 | 约-0.997 | -0.701/-0.602/-0.683 | -0.684/-0.656/-0.694 |
| DBM J50 gain P05，0.05σ | -84.03/-87.70/-88.43 | -7.10/-7.48/-6.95 | -5.30/-6.04/-11.00 |

新坐标改善了数值condition，但没有得到可靠的物理更新方向。Z的seen value-delta correlation
达到`0.56--0.79`，证明标量value可拟合；同一seen集合的autograd gradient仍不稳定，说明
value拟合没有约束出可用action derivative。实际DBM line search中，Z在0.05σ的gain中位和
P05对全部seed均为负，不能用于Actor更新。

因此最终qualification为`COORDINATE_CRITIC_FAIL_CLOSE_GRADIENT_MAINLINE`。后续不再进行
Critic gradient loss/encoder/normalization/temporal-basis扫描；Critic仅可保留为离线candidate
ranking或value近似辅助。主路线固定为bounded proximal Offline Search Distillation与Phase 2
episode-grouped Actor learnability。

复现入口：

```text
scripts/model_verify/run_mppi_g0_sensitivity_coordinate_cv.py
scripts/model_verify/validate_mppi_g0_sensitivity_coordinate_cv.py
outputs/mppi_proposal/g0_sensitivity_coordinate_cv_20260818_v1/
```

动作通道合同同时修正为`[acceleration, steering]`，early-steering flat indices为`[1,3,5]`。
旧position-horizon产物的`early_steering_map_t10/t30`误取channel 0，实际为acceleration；源码
已修正。该问题不影响全16维horizon质量、position时域总梯度、cost项翻转归因或本节结果。

---

## 9. 2026-08-20 完整absolute-action value/ranking复核

§1和§8关闭的是“Critic作为Actor gradient provider”，并不等于标量value/ranking一定无用。
为消除旧实验只覆盖anchor附近残差扰动的疑问，新增完整绝对动作复核：1800个train状态，每状态
使用8个原始起点+8个v1优化中心+8个v2继续优化中心，共43200条完整
`(state, absolute 8x2 knots, J_direct)`；输入不含warm/anchor/residual，按episode做
3-fold×3-seed。

标量侧得到一致正结果：heldout log-value Pearson `0.945-0.951`、material-pair accuracy
`0.956-0.964`、top-1 headroom recovery `0.950-0.994`、top-4 `0.9985-0.9986`，harmful
top-1仅`0.11%-0.56%`。raw/v1-opt/v2-refined内部排序仍有`0.87-0.91`，并非只识别候选stage。
因此此前“Critic仅可保留candidate ranking辅助”现在有了直接实证支持，qualification为
`ABSOLUTE_ACTION_VALUE_CRITIC_USABLE_FOR_RANKING`。

同一checkpoint的独立action-gradient审计没有恢复主线：raw随机起点处cosine中位
`0.883-0.897`且幅值正确；warm处中位仅`0.620-0.630`、P10为`-0.297~-0.392`；bank-best
附近真梯度norm中位仅0.057，而网络幅值高估约49-54倍、cosine中位`0.016-0.041`。所以全局
高信号value场能学，近优局部导数仍未被离散value/ranking监督约束。最终用途边界更新为：

- candidate bank粗排/shortlist：可用，但仍需真实DBM/Query cost裁决top候选；
- 通过Critic autograd更新Actor：不可用，维持
  `ABSOLUTE_VALUE_CRITIC_GRADIENT_NOT_USABLE`；
- 不恢复gradient loss/坐标/encoder扫描，Search仍是局部改善主线。

完整合同、分速度指标、随机top-k对照与复现链见主review §11.83；产物位于
`outputs/mppi_proposal/absolute_action_value_critic_20260820_v1/`。

## 10. 近优stationarity修正（2026-08-20）

对§9的梯度结论作一项重要细化：bank-best真梯度norm中位仅0.057，cosine方向本身不是合理
主门；真正需要修复的是Critic约50倍的虚假幅值。保持absolute value/ranking合同不变、只在
每状态bank-best增加`lambda_s * ||d log(1+Q)/da||^2`后，`lambda_s=0.1`三seed将预测norm
中位从约2.82降至0.369/0.558/0.330，value/ranking仍通过。`lambda_s=1.0`可进一步接近真实
幅值，但会压弱raw/warm响应，因此判为过强且不扩seed。

冻结DBM的非归一化受限小步验证确认该机制有实际意义。在`eta=0.001`、逐分量最大0.05sigma
下，S1把bank-best的mean regression从V0的约`-0.038~-0.058`降到
`-0.0014~-0.0093`，P05从`-0.139~-0.203`降到`-0.0028~-0.0068`；warm仍保持正平均gain。
固定`||g_pred||<1` stay门可让约79-81%的bank-best停止，而warm仅停止约1-2%。这验证了“平坦
区不必找方向，只需输出小梯度并限制Actor更新”的用户假设。

但S1仍有seed 1的raw/warm响应退化，且少数bank-best高norm离群点保留负worst。因此结论是
stationarity可作为训练辅助，不足以恢复无保护SAC：Actor-gradient pilot若重启，必须使用
非归一化极小步、训练期DBM/Query真实cost accept/reject；部署继续one-shot Actor + two-center
warm guard。完整逐seed表和复现链见主review §11.83.4-§11.83.5。

## 11. Stationarity尾部case归因（2026-08-20）

逐帧冻结DBM复算排除了“DBM自身不稳定”：step gain最大重算误差`2.26e-5`。warm与bank-best
尾部不是同一问题。warm tail真梯度norm约10-11、Critic方向cosine中位约-0.52~-0.64；实际
cost increase与真实一阶项相关0.97-0.99，说明是明确的Critic跨episode符号错误。bank-best
tail真梯度norm约0.09-0.12，实际增加主要来自二阶曲率，Critic却仍输出2.5-3.6的tail中位norm。
两类top case都以position项为主（warm 15/15；bank-best 13/15，另2个yaw）。高速及bank-best
action边界有所富集，但跨seed三者共同tail只有10/90与11/90，说明同时存在少量系统困难帧和较多
训练seed误差，不是某个DBM场景随机失效。

因此后续不可再把二者统一称作“近优梯度尾部”：bank-best应由flat/stay头直接置零；warm方向
错误只能通过真实DBM/Query accept/reject形成新监督，单纯缩步不能修符号。详见主review
§11.83.6及`absolute_action_value_critic_stationarity_step_ab_20260820_v1/tail_case_analysis.json`。

## 12. 永久关闭与在线重启边界（2026-08-20）

正式永久关闭两类方案：`frozen-Critic -> Actor`以及“每状态一次Critic梯度、无新真实cost、无
Replay、无Critic持续更新”的single-shot Actor update。它们的共同缺陷是把离线Critic误差当成
固定可接受近似，无法利用DBM确定性reward纠正actor-visited尾部。此项是路线冻结，不再接受
loss/坐标/epoch形式的重开。

允许的唯一Actor--Critic重启是持续在线contextual-bandit循环：Actor生成新absolute action，
DBM返回真实cost，所有样本写Replay，Twin Critic多次更新，Actor低频更新，再以新策略继续交互。
坏动作必须保留为监督；真实DBM accept/reject只控制selected checkpoint晋级，不删除探索或
阻止latest Actor形成新分布。部署不加载Critic。

详细OAC-0至OAC-4计划、`20:1`更新比、内部episode-heldout门、停止条件和artifact合同见
`car_foundation/docs/mppi_online_actor_critic_pilot_plan_20260820.md`。当前只授权Actor冻结的
OAC-0/OAC-1；通过burn-in门前不得开始Actor更新，formal validation/test继续封存。

## 13. OAC-0/OAC-1实测结果（2026-08-20）

前两阶段已经执行，不再是待实施计划。三seed各采15360条actor-visited absolute-action样本并
更新Twin Critic 200步；Actor参数零更新且hash前后相同。value/ranking主体通过：actor-visited
pair accuracy `0.876--0.880`、Pearson `0.939--0.947`，lag-2坏动作总体排序`0.933--0.941`。

阻塞来自两个更严格条件：独立flat/stay无法同时满足bank-best recall≥0.80和warm false-stay≤0.10；
train-only阈值校准后heldout recall仍只有`0.320--0.565`。另外最初被Critic排错的坏动作只有
`0.332--0.446`在最终被纠正，未达到0.50。在线适配还使历史heldout bank排序下降约
`1.7--2.5pp`，2.8m/s actor-visited排序约0.825。

因此在线持续更新比冻结/单次梯度路线更有信息，但本轮尚不足以授权Actor梯度更新。Actor继续
冻结，OAC-2未启动。允许的下一轮仅为OAC-1B Critic侧补救：local-span flat监督、未纠正pair与
高速优先回放、历史bank rehearsal；不降低原门。独立复算与DBM replay由
`validate_mppi_online_absolute_sac.py`通过。

## 14. 固定Replay额外训练归因（2026-08-20）

在完全固定OAC-1 Replay与监督的条件下，将Critic总update标签从200扩到400/800/1600。800步
后三seed的初始错误纠正率已全部超过0.50，2.8m/s排序全部超过0.85；1600步纠正率达到
`0.634/0.666/0.584`，总体pair accuracy达到`0.918/0.924/0.923`。历史heldout bank没有退化，
反而比200步改善0.1--0.7pp。

因此§13的错误纠正阻塞主要是Critic训练预算不足，不是固定Replay缺少有用信息。flat/stay也随
训练改善，但严格train-calibration→heldout false-stay仍为`0.100/0.105/0.117`，只有1/3严格
通过；所以暂不修改flat标签，也不解冻Actor。父checkpoint未保存optimizer state，本轮使用同LR
AdamW重启，结论限定为“额外优化有效”，不能声称原optimizer连续训练必然得到完全相同轨迹。

## 15. 固定Replay扩展到6400步（2026-08-20）

沿§14同一Replay和同一次200步边界AdamW重置继续训练到3200/6400。主Critic仍持续改善：1600到
6400步，三seed的初始错误纠正率分别增加`0.128/0.114/0.150`，2.8m/s pair accuracy增加约
`0.035--0.040`，总体pair accuracy最终达到`0.945--0.948`。heldout历史bank相对200步的变化为
`+0.008/-0.003/+0.013`，说明没有统一的旧分布遗忘，但seed 1存在小幅退化，选模仍需heldout
约束。

与主value/ranking不同，flat头严格episode-heldout false-stay在1600步约`0.10--0.12`，到6400
步升为`0.142--0.158`；高recall伴随误报恶化。结论因此拆分为：更多训练步足以继续改善value、
ranking、错误纠偏和高速层，但不能替代flat/stay的监督或校准修复。Actor继续冻结，不能据主
Critic曲线恢复单次梯度更新方案。

复现产物：`outputs/mppi_proposal/online_absolute_sac_oac1_fixed_replay_extended_20260820_v1/`。

## 16. 联合Value与连续移动系数（2026-08-20）

独立BCE flat头的问题被定位为监督接口而非Value容量：改为让辅助头读取Twin Value与动作对信息，
并以连续`c_move=max(delta J,0)/(max(delta J,0)+0.1)`联合反传后，固定0.5物理操作点在
3 seed×4 checkpoint全部通过。最终recall `0.905--0.923`、false-stay `0.008--0.013`，系数MAE
约`0.103--0.108`；同时Value actor-visited排序约0.953、heldout bank约`0.948--0.951`，没有
出现辅助任务损伤。

这关闭了“flat必须是独立二分类概率头”的设计；二分类只保留为固定物理操作点下的派生安全指标。
当前系数表达的是可改善空间，不是局部曲率；若以后需要步长风险，还应另设sensitivity头，不能把
两者重新合成一个flat标签。Actor在新outer split复核前继续冻结。

## 17. 新outer split复核：Value与连续系数机制可迁移（2026-08-20）

在独立episode outer fold-1上重新生成三seed Actor-visited Replay，并沿完全相同的Value与连续系数
合同训练。主Value在6400步达到actor-visited pair `0.941--0.948`、heldout bank
`0.942--0.948`和2.8m/s `0.918--0.923`；继续联合训练后为
`0.948--0.953`、`0.945--0.951`和`0.928--0.934`。这说明Value/ranking改善不是fold-0记忆。

固定`c_move<=0.5`门在fold-1的recall为`0.898--0.922`，false-stay仅
`0.010--0.017`，3/3 seed和12/12训练后checkpoint通过；相关性`0.829--0.852`、MAE
`0.107--0.111`。因此“独立BCE flat头不能泛化”不应外推成“Value无法支持连续动作尺度”：两者
共享Value表征并联合训练后，连续尺度能跨split复现。

本结果只放行持续在线OAC-2 pilot，不恢复被冻结的单次Critic梯度方案。Actor在本节仍零更新；
后续Actor必须通过新真实cost、Replay与持续Critic更新闭环学习，且保留DBM accept/reject与部署
two-center guard。formal validation/test仍封存。

## 18. 持续在线更新后的Critic状态（2026-08-24）

OAC-2在fold-1执行20轮真实DBM交互后，Twin Value与连续系数没有因Actor分布移动而失效。
actor-visited pair accuracy为`0.930--0.936`；固定0.5系数门的recall为`0.878--0.900`、
false-stay为`0.012--0.018`。同时selected Actor在3 seed均获得正mean/median与正bootstrap CI
下界，说明持续Replay和Critic更新能够支持小步策略改进。

这仍不恢复“Critic梯度本身已成为可靠oracle”的结论：direct Actor在24%--37%状态回归，P05和
worst为负。通过来自低频极小步、真实DBM持续反馈、selected accept/reject与连续系数共同作用；
冻结Critic或单次梯度更新路线继续永久关闭。下一步仅允许跨fold同合同复核，并保留two-center
guard。

## 19. 完整3-fold持续在线结果（2026-08-24）

OAC-2已扩到fold-0/1/2各3 seed，并在相应outer-heldout episodes上取得9/9正mean、正median、
正bootstrap CI下界和非负高速gain。九seed平均真实gain为`1.322`，说明持续Critic更新产生的Actor
小步收益能跨episode迁移，而不是单fold内部选择伪影。

但这不改变Critic定位：它是持续交互中的可更新value/ranking近似器，不是可冻结的全局梯度
oracle。direct尾部仍有25.7%--36.7%逐状态回归，worst最低`-35.290`；只有two-center guard把
部署候选的P05/worst钳到0。下一阶段是闭环传导验证，不是恢复单次梯度路线。

## 20. 200轮预算曲线对Critic归因的修正（2026-08-24）

fold-1三seed将持续在线训练从20轮扩到100/200轮后，Actor真实mean gain从`1.306`提高到
`5.875/9.953`，同时actor-visited Critic pair从`0.934`稳定到`0.937/0.940`。三个seed到200轮
均仍选择最后checkpoint。这证明在当前Actor访问的局部动作范围内，Critic已经足以支持持续改进；
20轮收益小不能再主要归因于Critic失效。

但Critic“局部可用”不等于Actor整体目标正确。direct P05/worst随预算从
`-1.65/-17.89`恶化到`-9.57/-136.89`，而regression fraction保持约31%。这更符合Actor的
batch均值目标持续放大一批错误状态，而不是Critic主体排序坍缩。下一阻塞转为Actor的state-wise
风险分配与checkpoint gate；Critic仍必须持续接收新真实DBM数据，不能冻结，也尚未证明能在通往
teacher的全动作路径上充当全局oracle。

## 21. 200轮尾部不是最终Value端点排序盲区（2026-08-24）

对fold-1 200轮全部1800个seed-state重新计算初始Actor与最终Actor端点的保守Twin Value差，并与
真实DBM gain逐项配对。符号准确率为`90.3%`；563个真实回归样本中Critic仅将`14.6%`误判为改善，
83个`gain<=-10`严重回归样本中误判率仅`9.6%`。因此§20“Critic局部可用”的判断得到更直接
支持：当前负尾不是因为最终Value对坏端点整体失明。

这也限定了结论边界。端点排序正确不等于沿200轮训练路径的每次action gradient都正确，也不
等于共享Actor能同时满足所有状态。当前Actor最小化batch平均value，selected gate没有P05/CVaR
约束；因此可以在多数状态降cost的同时牺牲固定少数状态。严重尾部81/83由position增量主导，
early steering只移动约`0.032sigma`也能被长horizon放大。后续优先给Actor增加相对selected端点
的soft regression/CVaR项及tail checkpoint门；Critic继续在线更新，但暂不再增加单独训练预算。

```text
outputs/mppi_proposal/online_absolute_sac_oac2_budget_tail_20260824_v1/analysis.json
```

## 22. 固定尾部CVaR实验再次排除Critic主体退化（2026-08-24）

原目标、`lambda_tail=10`和`lambda_tail=100`三臂在200轮后的actor-visited Critic pair分别为
`0.9402/0.9399/0.9406`，独立validator全部通过。尾部臂显著改善P05/worst，却把mean gain压到
原来的12.7%--14.0%；因此该trade-off来自Actor loss约束强度，而不是Critic在新训练中退化。

这也证明最终Value端点信号确实能用来约束尾部，但zero-margin top-k正回归并不是合适的最终
接口。后续若继续，应使用material margin与自适应dual预算；不要通过增加Critic训练步或继续
修改Value loss来解决本轮的mean收益损失。

```text
outputs/mppi_proposal/online_absolute_sac_oac2_tailcvar_ab_20260824_v1/analysis.json
```

## 23. 多候选Actor与DBM task-loss对Critic定位的更新（2026-08-24）

固定K=4 Actor在J16多elite集合监督下，DBM best-of-K只相对K=1获得约`+0.17` OOF recovery，
两者仍为`-2.95/-3.12`；已有absolute-value Critic选择仅再损失约`0.05--0.06`。所以Critic作为
候选排序器是可用的，多模态输出也有小收益，但二者都不能修复Actor当前主体误差。

更关键的反事实是：完全绕过Critic，直接用DBM J50反传同一G-X Actor，full train与episode-heldout
recovery达到`0.834/0.779`。这进一步确认Critic不是Actor函数类的必要组成，也说明当前OAC的低
收益不能简单归因于“Actor网络无表达能力”。但task-loss最终heldout J=`23.61`仍远离J16
`4.46`，且有8.5%状态回归；真实梯度也没有自动解决共享策略的tail与输出支撑问题。

Critic角色因此保持不变：持续在线Value/ranking与多候选selector可继续使用；冻结/单次梯度
provider仍永久关闭。下一结构诊断优先扩大absolute-action输出支撑，而不是重开Critic gradient
loss或直接上Diffusion。
