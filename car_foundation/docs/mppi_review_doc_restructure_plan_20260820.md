# MPPI Review 文档重构计划（2026-08-20，review 修订版）

> 状态：**已于 2026-08-20 执行并完成重构后验收**。
> 执行快照为 5008 行 / 337820 bytes，最新原始内容到 §11.79，Git blob hash 为
> `e58c145d4eca9d3246b807e142573c36355ce2e7`。重构结果为主文档 §11.38–§11.80 与历史归档
> §1–§11.37/附录 A；后续再次重构仍必须重新冻结快照，不能依赖这里记录的行号或文件尾。

---

## 1. Review 结论

重构方向通过，但原计划不能直接执行。§11.38 是合理的语义切分点：此前为初版评审、部署
two-center、100kph 契约及 Critic-gradient 完整历史；此后为 Search Distillation、Phase 0/1/2、
guard 闭环和 direct/no-anchor 精度路线。

执行前必须解决以下阻塞项：

1. **禁止按旧的 4889 行文件尾截取。** 当前文件已到 §11.78；主文档必须从 `### 11.38` 保留到
   EOF，后续再新增内容也必须自动包含。
2. **tombstone 原文必须真正进入归档。** 原计划只归档 §1–§11.37，却又删除主文档中四个作废
   block，会造成实际信息损失。
3. **§11.57 不再是当前权威状态。** §11.60/§11.61 已完成其中的 guard 待办，§11.75–§11.78
   又更新了 clean/no-anchor 输入合同和 G-X 结构基线。拆分前应 append §11.80 最新
   consolidation，并让所有入口指向它。
4. **必须迁移跨文档引用。** 其他文档仍裸指向原主文档 §9、§11.9、§11.14–§11.38、
   §11.35–§11.38；拆分后需要显式区分 archive 与 main。
5. **先解决 Git 索引状态。** review 时 `car_foundation/docs/` 被 `.gitignore` 忽略，主 review 文档
   及多份历史文档处于 staged deletion、但工作区文件仍存在。未解决前不得执行或 commit，否则
   可能把整批文档从仓库删除。

---

## 2. 目标与原则

- 主文档成为唯一工作入口：当前状态、活跃路线、当前证据链和权威索引。
- 历史正文完整进入归档；主文档中的已作废结果只保留可导航 tombstone。
- 保持 append-only 的审计语义：不改写历史实验结论，通过新 consolidation、覆盖声明和归档来
  表达后续裁决。
- 以 heading 和内容 hash 为边界，不以易漂移的固定行号为边界。
- 重构不改变任何 qualification、artifact 路径、实验数值或仍生效的合同。

---

## 3. 执行前冻结与 Git 前置检查

### 3.1 冻结源快照

执行时记录：

```text
source path
source byte count / line count
git hash-object result
最后一个 heading 及其编号
```

同时确认以下唯一语义切分标记只出现一次：

```markdown
### 11.38 路线决策：Critic-gradient 降级，转向近端 Offline Search Distillation（2026-08-17）
```

如果执行时最后一节已超过 §11.78，应更新 §11.80 consolidation 和本计划中的范围说明，但仍保留
`§11.38 -> EOF`，不得退回固定结束行号。

### 3.2 Git 阻塞门

执行重构前必须满足：

- `car_foundation/docs/` 不再因新增 ignore 规则而无法正常跟踪；
- 主文档和本计划不处于 staged deletion；
- 明确恢复哪些现存 docs/assets，且只操作本计划涉及的显式文件；
- 保存当前 staged/unstaged 状态，不覆盖其他人的代码、文档或 asset 改动；
- 未通过本门时停止，不创建重构 commit。

---

## 4. 拆分方案

### 4.1 语义边界

| 文件 | 内容范围 | 说明 |
| --- | --- | --- |
| `mppi_sampling_center_review_archive_20260812.md` | 原始 preamble + §1–§11.37；另加四个已作废 block 的完整历史附录 | 历史审计记录，只允许增加勘误/归档说明 |
| `mppi_sampling_center_review_20260812.md` | §11.38 到 EOF，包括新增的 §11.80 consolidation | 当前路线与最新证据链 |

切分必须通过完整 heading 文本定位。review 时 §11.38 位于原文件第 2972 行，但该行号仅用于人工
复核，不得作为执行脚本的唯一边界。

### 4.2 拆分前追加 §11.80 最新 consolidation

在原主文档末尾 append §11.80（§11.79 已被 reference token cross-attention 实验占用），至少固化以下当前状态：

- Critic-gradient provider 路线保持关闭；Actor、formal validation/test 继续冻结。
- two-center guard 的当前默认合同为 `m1.0whr_d5`：不对称即时 warm 回退，恢复逐步 warm
  model-cost 硬下界；对称 hysteresis 仅是显式 opt-in 性能档。
- Actor 使用严格 clean/no-anchor 输入合同；anchor、first-pass feedback 和 gradient context 均不得
  进入部署 Actor。
- 当前结构精度基线升级为 no-anchor J16 G-X；`x_ref_ego` 与对应 control knot 显式对齐。
- G-X 只获得稳定的小幅精度改善，rollout tail/formal gate 仍未通过，不授权部署。
- 明确最新的下一步及已经关闭的 canonical teacher、TCN、current skip、no-attention、Frenet 必需性
  等分支。

§11.80 应成为新的唯一权威入口。§11.57 保留为 2026-08-18 consolidation 快照，不再称为“当前
唯一权威索引”。

### 4.3 归档文件头部

```markdown
# MPPI 采样中心策略训练评审：归档（历史截至 2026-08-18）

> 本文件保存原 `mppi_sampling_center_review_20260812.md` 的 preamble、§1–§11.37，
> 以及主文档中被 tombstone 化的四个早停 bug 历史 block。
> 当前主文档从 §11.38 起继续；最新权威结论见主文档 §11.80。
> 本文件仅作审计与复现参考，不再承载活跃计划。
```

不能简单在原始第 1 行之前再加一个 H1 后原样复制，避免出现两个并列 H1。原标题应保留为归档
metadata/引用块，或将其降为归档中的“原始文档头部”小节，同时确保文字本身仍被保存。

### 4.4 主文档头部

```markdown
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
```

---

## 5. Tombstone 与历史附录

主文档中的以下四个 block 压缩为 3–6 行 tombstone；其**完整原文**按原出现顺序复制到归档的
“附录 A：主路线中的已作废实验记录”中：

| 原节 | review 时原行号 | 状态 | 权威替代 | 必须保留 |
| --- | ---: | --- | --- | --- |
| §11.51 第一处：A0 首次结果 | 3669–3709 | 早停 bug 作废 | §11.53 第一处：修复重跑 | 原始状态修正与全部数值 |
| §11.49 第二处：A0 语义清理 MLP | 3710–3748 | 早停 bug 污染的负对照 | §11.53 第一处 | 完整审计记录 |
| §11.52 第一处：A0b 2×2 | 3749–3788 | 训练结果部分作废 | §11.53 第一处 | `consensus64_labels_20260818_v1/`、`b0_sensitivity_20260818_v1/` 及有效标签/B0 结论 |
| §11.54 第一处：A1 旧执行结果 | 3839–3853 | 早停 bug 作废 | §11.50 第二处：A1 修复重跑 | 完整原文 |

注意：原计划把“§11.53 第二处”写成修复重跑，实际按行序修复重跑是 §11.53 **第一处**；第二处是
early/late splice oracle。所有 tombstone 应使用“编号 + 描述”双重消歧，禁止只写裸编号。

tombstone 示例：

```markdown
### 11.51 [已作废] Phase 2 A0 首次结果（早停 bug 污染）

> 本节原始正文已移至归档附录 A。train 恢复 0.044 及其归因被早停修复重跑推翻；
> 当前权威结果见“§11.53 A0/A0.1 早停修复与 2×2 重跑”。
```

---

## 6. 编号消歧

review 时实际重复编号为 **9 对**，不是 10 对：§11.48–§11.55 共 8 对，另有 §11.61 一对。
§11.63 本身没有重复，只包含对两个 §11.61 的文字说明。

更新索引时：

1. 保留 §11.57.1 原有 8 对记录；
2. 追加 §11.61：hysteresis 网格/warm hard return 与 KNN 完整动作更正/直接估 `a*` 预注册，
   两者都有效但主题不同；
3. 将范围说明更新为“§11.62–§11.80 按行序唯一”，执行时若新增章节则自动扩展；
4. 在 §11.80 再给出当前引用规范，避免后续继续裸引冲突编号。

---

## 7. 其他文档与引用迁移

### 7.1 历史状态标注

以下四份文档可标注为 `CLOSED-HISTORY`，但仍须保留其有效审计价值：

- `mppi_sampling_center_handoff_20260802.md`
- `mppi_direct_actor_trpo_like_design_20260807.md`
- `mppi_sequential_probe_execution_plan_20260806.md`
- `rl_mppi_sampling_center_design_20260731.md`

标注指向主文档 §11.80，而不是 §11.57。

`rl_sampling_center_overview_20260807.md` 在 2026-08-18 仍有更新，不能未经内容复核就整体标成
“不再更新”。对它使用 `HISTORICAL-OVERVIEW / PARTIALLY-SUPERSEDED`，列出哪些节仍有效、哪些被
§11.80 覆盖。

### 7.2 已知引用迁移

至少处理以下引用：

| 来源 | 当前引用 | 拆分后 |
| --- | --- | --- |
| `mppi_sampling_center_handoff_20260802.md` | 主文档 §9、§11.9 | archive §9、archive §11.9 |
| `critic_gradient_value_assessment_20260817.md` | 主文档 §11.14–§11.38 | archive §11.14–§11.37 + main §11.38 |
| `rl_sampling_center_overview_20260807.md` | 主文档 §11.35–§11.38 | archive §11.35–§11.37 + main §11.38 |
| `mppi_sampling_design_and_current_plan_20260807.md` | 主文档 §11.38–§11.50 | main，仍有效，但应补描述消歧 |

执行时必须用全仓库搜索补全清单，不能只验证主文档自身。

---

## 8. 执行步骤

1. 通过 §3 Git 阻塞门，冻结原文 hash、行数、字节数和最后 heading。
2. 在原文末尾 append §11.80 最新 consolidation，并复核其中所有当前结论。
3. 通过完整 `### 11.38 ...` heading 定位语义边界。
4. 创建归档：归档头部 + 原 preamble/§1–§11.37 + 附录 A 四个完整作废 block。
5. 创建主文档：新头部 + §11.38–EOF；将四个作废 block 原位置替换为 tombstone。
6. 更新 §11.57.1 和 §11.80 的编号消歧/引用规范。
7. 更新四份 closed-history 文档及一份 partially-superseded overview。
8. 迁移所有跨文档 section 引用。
9. 执行 §9 的内容守恒、引用、路径、编号和 Git 验证。
10. 仅在 diff 人工复核通过后提交；显式 stage 文件，禁止 `git add -A`。

若需要拆 commit，建议：

1. `archive split + latest consolidation`；
2. `tombstones + cross-document references + history labels`。

---

## 9. 验证步骤

### 9.1 内容守恒

- 保留执行前原文的只读快照和 `git hash-object`。
- 对以下 block 分别计算 hash/逐字节比较，而不是仅检查总行数：
  - 原 preamble + §1–§11.37 ↔ archive 主体；
  - 四个作废 block ↔ archive 附录 A；
  - §11.38–EOF（排除 tombstone 替换点和新增 §11.80）↔ 新主文档对应正文。
- 行数仅作为诊断信息，不作为通过标准。
- 禁止使用原计划的 `grep -v '^>'` diff；blockquotes 中包含实质状态修正，过滤它们会掩盖丢失。

### 9.2 标识符与 artifact

- 原文中的 qualification 字符串集合必须是 archive + main 的子集。
- 原文中的完整反引号路径集合必须被保留；artifact 检查需支持 `-`、`.`、大写和文件名，不能使用
  只接受 `[a-z0-9_/]` 的旧正则。
- 对四个 tombstone 同时验证归档原文位置和主文档权威替代指针。

### 9.3 Heading 与引用

- 解析 archive/main 的 heading 集，确认 §11.38 只在 main，§1–§11.37 只在 archive 主体。
- 重复 heading 应恰为已登记的 9 对；新增重复编号一律失败。
- 全仓库搜索 `mppi_sampling_center_review_20260812.md` 和裸 `§11.*` 引用，逐条确认目标文件和描述。
- 检查所有相对 Markdown 文件链接存在；仅靠主文档头部的归档说明不能视为引用已修复。

### 9.4 Git 与人工复核

- `git status --short` 中涉及文件必须与计划清单一致；不得残留 staged deletion。
- 归档为新增、主文档及引用来源为修改；无无关代码/asset 被 stage。
- 人工阅读新主文档头部、§11.57、§11.80、四个 tombstone、归档头尾和所有历史状态标注。
- 验证通过前不 commit。

---

## 10. 明确不改动

- §11.38–EOF 的有效实验正文，除四个明确列出的 tombstone block 和新增 §11.80 外不改写。
- 任何 qualification、artifact 路径、实验数字和原始审计记录。
- §8.4 rollout 预算、§10 100kph 数据隔离、warm hard-return 合同、strict clean/no-anchor 输入合同。
- `critic_gradient_value_assessment_20260817.md` 的结论正文；只允许迁移其跨文件引用。
- Actor、formal validation/test、训练代码、模型和实验产物。

---

## 11. 预期效果

精确行数应在执行快照冻结后重新计算，不再预设 1790/1900 行。预期结果是：

- 主文档打开即进入 §11.38 当前路线，并在头部明确指向最新 §11.80；
- 历史 Critic/早期部署证据和四个 bug 污染结果均可完整审计；
- 不再把已完成的 §11.57.3 待办误报为当前计划；
- 9 对重复编号全部可导航；
- 仓库内 section 引用明确区分 archive/main；
- 重构前后 qualification、artifact 和历史正文零信息损失。
