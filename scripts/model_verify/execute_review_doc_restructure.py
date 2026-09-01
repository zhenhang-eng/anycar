#!/usr/bin/env python3
"""One-shot execution of the review document restructure per plan 20260820.

The script is guarded against stale-snapshot reuse and an already restructured
source. It verifies generated content before atomically replacing output files.
"""

import hashlib
import re
import sys
from pathlib import Path

SOURCE = Path("car_foundation/docs/mppi_sampling_center_review_20260812.md")
ARCHIVE = Path("car_foundation/docs/mppi_sampling_center_review_archive_20260812.md")
SNAPSHOT = Path("/tmp/review_doc_snapshot.md")
EXPECTED_SNAPSHOT_SHA256 = "7f64d7c3a4936ba931f7cf13add85681144e3d9796376f6b51c464af3231e33b"
RESTRUCTURED_HEADER = "# MPPI 采样中心策略训练评审（当前路线，2026-08-12 起）"

if not SNAPSHOT.is_file():
    print(f"FATAL: frozen snapshot not found: {SNAPSHOT}")
    sys.exit(1)

snapshot_bytes = SNAPSHOT.read_bytes()
snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
if snapshot_sha256 != EXPECTED_SNAPSHOT_SHA256:
    print(
        "FATAL: frozen snapshot hash mismatch; refusing stale or unknown input "
        f"({snapshot_sha256})"
    )
    sys.exit(1)

if SOURCE.is_file() and SOURCE.read_text().splitlines()[:1] == [RESTRUCTURED_HEADER]:
    print("FATAL: source is already restructured; refusing to overwrite it")
    sys.exit(1)
if ARCHIVE.exists():
    print(f"FATAL: archive already exists; refusing to overwrite it: {ARCHIVE}")
    sys.exit(1)

# Read from snapshot (frozen copy)
lines = snapshot_bytes.decode().splitlines()
total = len(lines)

# --- Step 1: Locate boundaries by heading text ---
SPLIT_HEADING = "### 11.38 路线决策：Critic-gradient 降级，转向近端 Offline Search Distillation（2026-08-17）"
split_idx = None
for i, l in enumerate(lines):
    if l.strip() == SPLIT_HEADING:
        split_idx = i
        break
if split_idx is None:
    print(f"FATAL: split heading not found"); sys.exit(1)
print(f"split point: line {split_idx+1} (0-indexed {split_idx})")

# --- Step 2: Identify four tombstone blocks (by heading text) ---
TOMBSTONE_HEADINGS = [
    "### 11.51 Phase 2 A0执行结果：train恢复<50%，判定树落入拟合/标签参数化分支（2026-08-18）",
    "### 11.49 Phase 2 A0 执行结果：Actor在训练状态上也仅恢复4%——蒸馏层失败，分支路由到标签噪声（2026-08-1",
    "### 11.52 A0b 2x2消融与B0敏感度：标签形式是train失败主因，泛化成为新主阻塞（2026-08-18）",
    "### 11.54 A1旧执行结果（已作废）：S归一化坐标实验受早停bug污染（2026-08-18）",
]
# Find each heading (use prefix matching since we might have truncation)
tombstone_ranges = []  # list of (start_idx, end_idx) 0-indexed, inclusive
all_headings = [(i, l) for i, l in enumerate(lines) if l.startswith("#")]
for th in TOMBSTONE_HEADINGS:
    found = None
    for i, l in all_headings:
        if l.strip().startswith(th[:60]):
            found = i
            break
    if found is None:
        # Try shorter prefix
        for i, l in all_headings:
            if th[:40] in l:
                found = i
                break
    if found is None:
        print(f"FATAL: tombstone heading not found: {th[:60]}"); sys.exit(1)
    # Find end (next heading of same or higher level)
    end = total
    for j, (hi, hl) in enumerate(all_headings):
        if hi > found and (hl.startswith("# ") or hl.startswith("## ") or hl.startswith("### ")):
            end = hi
            break
    tombstone_ranges.append((found, end))
    print(f"tombstone: lines {found+1}-{end} | {lines[found][:60]}")

# --- Step 3: Append §11.80 consolidation to the source (before split) ---
# We'll insert it at the end of the main doc content
consolidation = """
### 11.80 当前consolidation：权威结论与活跃计划（2026-08-20）

本节为当前唯一权威入口；此前§11.57（2026-08-18）为中间快照。与历史节冲突时以本节为准。

#### 已关闭路线汇总

**Critic-gradient provider**：正式关闭（§11.38降级→`COORDINATE_CRITIC_FAIL_CLOSE_GRADIENT_MAINLINE`）。
六轮统一grouped CV均0/5 fold、0/3 seed；机制定位为position项横向修正符号歧义（§11.41-11.42：
position占已归因翻转`149/170=87.6%`，占全部翻转`149/250=59.6%`；约90%翻转涉及cross分量）。
Critic仅保留candidate ranking辅助。坐标工程（S归一化/DCT/
AT/Z）只改condition不改可用性，全部关闭。

**Search Distillation离线Actor**：主阻塞=跨episode泛化+tail。已关闭：损失工程（S归一化、stay
加权、gain阈值stay化）、KNN检索（-0.43/-0.12）、覆盖扩张（600→1800 OOF平坦非单调）、大幅标签
SNR、A2结构三臂（G/T单独持平、GT +0.091未过≥0.10机制门）。

**Direct/no-anchor路线**：corr 0.860→0.867（G-X knot-x对齐）、绝对误差~0.051-0.054；与residual
（0.971）的差距来自信息量缺口（无anchor输入），不是架构/训练/标签问题。已关闭：canonical
teacher（consensus落入无人区+109%）、TCN、current skip、no-attention、RefCross（弱正向未过门）。

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
2. **No-anchor结构精度**：G-X为当前基线；下一步取决于新信息源引入或转向guard使用方式研究。
   RefCross弱正向可作候选旁路复验。
3. **Pending非阻塞项**：horizon加权候选的原始J50重rollout验证；rho与Phase 1b状态join；
   expansion 1350状态的anchor策略决策。

#### 编号消歧与引用规范

重复编号共9对：§11.48-11.55（8对）+ §11.61（1对），详见§11.57.1消歧表。§11.62-§11.80按
行序唯一。引用一律用"编号+描述"双重消歧（如"§11.53 A0/A0.1早停修复与2x2重跑"），禁止裸引
冲突编号。

Actor、formal validation和test保持冻结。
"""

# --- Step 4: Build archive file ---
archive_header = """# MPPI 采样中心策略训练评审：归档（历史截至 2026-08-18）

> 本文件保存原 `mppi_sampling_center_review_20260812.md` 的 preamble、§1–§11.37，
> 以及主文档中被 tombstone 化的四个早停 bug 历史 block（附录 A）。
> 当前主文档从 §11.38 起继续；最新权威结论见主文档 §11.80。
> 本文件仅作审计与复现参考，不再承载活跃计划。
>
> ---
>
> **原始文档头部**（保留供引用）：
"""

# Archive: header + lines[0:split_idx] (preamble through §11.37) + appendix
archive_parts = [archive_header]
archive_original = lines[0:split_idx].copy()
archive_original[0] = "> " + archive_original[0]  # preserve old title without a second top-level H1
archive_parts.extend(archive_original)  # preamble + §1-§11.37 (0-indexed 0 to split_idx-1)

# Appendix A: four tombstone blocks in original order
archive_parts.append("\n\n---\n\n## 附录 A：主路线中的已作废实验记录\n")
archive_parts.append("> 以下四个 block 已从主文档 tombstone 化，完整原文按原出现顺序保存于此。\n")
for (start, end) in sorted(tombstone_ranges):
    archive_parts.append("\n" + "\n".join(lines[start:end]) + "\n")

archive_content = "\n".join(archive_parts)

# --- Step 5: Build new main document ---
main_header = """# MPPI 采样中心策略训练评审（当前路线，2026-08-12 起）

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
"""

# Main: new header + §11.38 to EOF (with tombstones replaced + §11.80 appended)
main_parts = [main_header]
main_parts.append("")

# Track which lines to skip (tombstone ranges)
tombstone_set = set()
for start, end in tombstone_ranges:
    for i in range(start, end):
        tombstone_set.add(i)

# Tombstone replacement texts (keyed by start index)
tombstone_replacements = {}
for idx, (start, end) in enumerate(tombstone_ranges):
    heading_line = lines[start].strip()
    if "11.51" in heading_line:
        replacement = f"""{heading_line.replace('### 11.51 ', '### 11.51 [已作废] ', 1)}

> 本节原始正文已移至归档附录 A。train 恢复 0.044 及其归因被早停修复重跑推翻；
> 当前权威结果见"§11.53 A0/A0.1早停修复与2x2重跑"。
"""
    elif "11.49" in heading_line and "Phase 2 A0 执行结果" in heading_line:
        replacement = f"""{heading_line.replace('### 11.49 ', '### 11.49 [已作废] ', 1)}

> 本节原始正文已移至归档附录 A。train 恢复率 0.044 的"蒸馏层失败"归因被早停修复
> 推翻（修复后 train 0.77）；当前权威结果见"§11.53 A0/A0.1早停修复与2x2重跑"。
"""
    elif "11.52" in heading_line:
        replacement = f"""{heading_line.replace('### 11.52 ', '### 11.52 [部分作废] ', 1)}

> 本节原始正文已移至归档附录 A。训练结果部分作废；有效产物：
> `consensus64_labels_20260818_v1/`、`b0_sensitivity_20260818_v1/`。
> 权威训练结果见"§11.53 A0/A0.1早停修复与2x2重跑"。
"""
    elif "11.54" in heading_line:
        replacement = f"""{heading_line}

> 本节原始正文已移至归档附录 A。权威结果见"§11.50 A1修复版重跑与KNN残差检索基线"。
"""
    else:
        replacement = f"{heading_line}\n\n> 本节原始正文已移至归档附录 A。\n"
    tombstone_replacements[start] = replacement

# Build main body: §11.38 to EOF, replacing tombstones
for i in range(split_idx, total):
    if i in tombstone_set:
        # Check if this is the start of a tombstone block
        for start, end in tombstone_ranges:
            if i == start:
                main_parts.append(tombstone_replacements[start])
                # Skip to end of this block (the loop will continue past end)
                break
        # Skip non-start tombstone lines (they're covered by the replacement)
        continue
    else:
        main_parts.append(lines[i])

# Append §11.80 consolidation
main_parts.append(consolidation)

main_content = "\n".join(main_parts)

# Update the historical consolidation's navigation without renumbering it.
main_content = main_content.replace(
    "本节为当前唯一权威索引；与历史节冲突时以本节为准。历史节保持append-only不删除、不改号。",
    "> **覆盖状态（2026-08-20）**：本节是 2026-08-18 的中间快照，已被 §11.80 覆盖；\n"
    "> 当前权威结论与活跃计划以 §11.80 为准。本节保留历史口径与原始消歧记录。\n\n"
    "本节原为 2026-08-18 时点的唯一权威索引；当时与更早历史节冲突时以本节为准。"
    "当前以 §11.80 为准，历史节保持append-only不删除、不改号。",
    1,
)
main_content = main_content.replace(
    "#### 11.57.1 编号消歧索引（八对冲突）",
    "#### 11.57.1 编号消歧索引（九对冲突）",
    1,
)
index_row = (
    "| §11.61 | hysteresis网格与不对称warm回退（Pareto钉死） | "
    "KNN完整动作更正与直接估`a*`预注册 | 都有效，主题不同 |"
)
main_content = main_content.replace(
    '| §11.55 | A0.2 gain阈值stay化（**已关闭**） | **A2三臂执行结果**（GT+0.091未达机制门） | 后者权威 |',
    '| §11.55 | A0.2 gain阈值stay化（**已关闭**） | **A2三臂执行结果**（GT+0.091未达机制门） | 后者权威 |\n'
    + index_row,
    1,
)
main_content = main_content.replace(
    "机制链完整：position项近邻翻转主导（§11.41-11.42，88%归因position）、56%整体反向+晚段",
    "机制链完整：position项近邻翻转主导（§11.41-11.42：已归因翻转中`149/170=87.6%`，"
    "全部翻转中`149/250=59.6%`）、56%整体反向+晚段",
    1,
)

# --- Step 6: Verify content conservation before writing ---
orig_text = "\n".join(lines)
verification_failed = False

# Check qualification strings
orig_quals = set(re.findall(r'[A-Z][A-Z_]{15,}', orig_text))
arch_quals = set(re.findall(r'[A-Z][A-Z_]{15,}', archive_content))
main_quals = set(re.findall(r'[A-Z][A-Z_]{15,}', main_content))
missing_quals = orig_quals - (arch_quals | main_quals)
print(f"\nQualification strings: orig={len(orig_quals)} arch+main={len(arch_quals|main_quals)} missing={len(missing_quals)}")
if missing_quals:
    print(f"  MISSING: {missing_quals}")
    verification_failed = True

# Check artifact paths
orig_paths = set(re.findall(r'`(?:outputs|scripts|/disk)[^\`\n]+`', orig_text))
arch_paths = set(re.findall(r'`(?:outputs|scripts|/disk)[^\`\n]+`', archive_content))
main_paths = set(re.findall(r'`(?:outputs|scripts|/disk)[^\`\n]+`', main_content))
missing_paths = orig_paths - (arch_paths | main_paths)
print(f"Artifact paths: orig={len(orig_paths)} arch+main={len(arch_paths|main_paths)} missing={len(missing_paths)}")
if missing_paths:
    print(f"  MISSING: {missing_paths}")
    verification_failed = True

# Check key content blocks survived (sample lines from each range)
sample_checks = [
    ("§1 结论", 20, "方向与工程纪律"),
    ("§9.7 闭环", 990, "realized stage cost"),
    ("§11.38 路线决策", 2972, "Critic-gradient 降级"),
    ("§11.60 guard A/B", 4320, "hysteresis同时改善"),
    ("§11.80 consolidation", None, "当前consolidation"),
]
for name, line_num, expected in sample_checks:
    if expected in archive_content or expected in main_content:
        print(f"  ✓ {name}: '{expected}' found")
    else:
        print(f"  ✗ {name}: '{expected}' NOT FOUND")
        verification_failed = True

if verification_failed:
    print("\nFATAL: verification failed; output files were not written")
    sys.exit(1)

# --- Step 7: Write sibling temp files, then atomically replace outputs ---
archive_tmp = ARCHIVE.with_name(f".{ARCHIVE.name}.tmp")
source_tmp = SOURCE.with_name(f".{SOURCE.name}.tmp")
archive_tmp.write_text(archive_content)
source_tmp.write_text(main_content)
archive_tmp.replace(ARCHIVE)
source_tmp.replace(SOURCE)

print(f"\nArchive: {len(archive_content.splitlines())} lines, {len(archive_content)} bytes")
print(f"Main: {len(main_content.splitlines())} lines, {len(main_content)} bytes")

print("\n=== RESTRUCTURE COMPLETE ===")
