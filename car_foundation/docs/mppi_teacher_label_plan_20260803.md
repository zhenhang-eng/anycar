# MPPI sampling-center teacher 标签方案与实现状态

更新时间：2026-08-03

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

### T1：高预算/多中心 DBM teacher（下一项）

从每个 snapshot 恢复 state/history/reference/warm knots，围绕 warm、T0 best、T0 soft
以及多尺度扰动中心生成 center bank，再使用固定 DBM rollout。至少记录：

- center bank 的构造、随机 seed、每中心候选预算和总 rollout 预算；
- 每个中心的 best/P10/soft cost、ESS、clipping 和 boundary 指标；
- 最终 teacher center 的选择规则及相对 warm/T0 的 regret；
- cost 配置、DBM/MPPI 参数、源码 commit/hash。

T1 应真正 rollout 每个新中心，用 held-out snapshot 检查 teacher 是否稳定优于 warm。
如果不同 seed 下 teacher 方向不稳定，应保存多模态 elite，而不是强制回归单个均值。

### T2：BC 与离线验证

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
