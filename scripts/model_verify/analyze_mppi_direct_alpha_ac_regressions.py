#!/usr/bin/env python3
"""Deep diagnosis of the 27 frozen TR3 Alpha-AC regression contexts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_mppi_direct_alpha_ac_tr3 import DEFAULT_OUTPUT, distribution
from generate_dbm_direct_trust_region_labels import actor_inputs
from train_mppi_direct_trust_region_actor import load_actor_payload, load_dataset, tensorize
from train_mppi_direct_trust_alpha_policy import extra_tensors, make_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def model_embedding(policy, inputs, direction, rho, scale):
    context = policy.encoder(*inputs)
    encoded_direction = policy.direction_encoder(torch.cat((
        direction.flatten(1), rho, scale,
    ), dim=1))
    return torch.cat((context, encoded_direction), dim=1)


def raw_tracking_features(source, reference_speed: float) -> dict[str, float]:
    state = np.asarray(source["initial_state_six"], np.float32)
    reference = np.asarray(source["reference"], np.float32)
    first = reference[1] if len(reference) == 51 else reference[0]
    delta = state[:2] - first[:2]
    lateral = -np.sin(first[2]) * delta[0] + np.cos(first[2]) * delta[1]
    heading = np.arctan2(np.sin(state[2] - first[2]), np.cos(state[2] - first[2]))
    action = np.asarray(source["current_action"], np.float32)
    return {
        "vx_mps": float(state[3]),
        "vy_mps": float(state[4]),
        "yaw_rate_rad_s": float(state[5]),
        "overspeed_mps": float(state[3] - reference_speed),
        "lateral_error_m": float(lateral),
        "heading_error_rad": float(heading),
        "current_acceleration": float(action[0]),
        "current_steering": float(action[1]),
    }


def stats(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    value = np.asarray([row[name] for row in rows], np.float64)
    return distribution(value)


def main() -> None:
    args = parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    config = summary["config"]
    device = torch.device(args.device)
    old_payload = load_actor_payload(Path(config["old_actor"]))
    ac_payload = torch.load(Path(config["alpha_ac"]), map_location="cpu")
    policy = make_policy(old_payload, device, dropout=0.0)
    policy.load_state_dict(ac_payload["policy_state_dict"], strict=True)
    policy.eval()

    rows: list[dict[str, Any]] = []
    validation_embeddings = []
    validation_line_cost = []
    validation_directions = []
    for label_path in sorted(args.run.glob("episode_*/*.npz")):
        with np.load(label_path, allow_pickle=False) as label:
            source_path = Path(str(label["source_snapshot"]))
            context_path = Path(str(label["context_label"]))
            risk_path = Path(str(label["risk_label"]))
            with np.load(source_path, allow_pickle=False) as source, np.load(
                context_path, allow_pickle=False
            ) as context, np.load(risk_path, allow_pickle=False) as risk:
                gradient_mean = np.asarray(risk["critic_gradient_mean"], np.float32)
                gradient_std = np.asarray(risk["critic_gradient_std"], np.float32)
                feedback_names = [str(value) for value in context["feedback_names"]]
                scalar_names = feedback_names[-10:]
                for repeat in range(len(label["old_center"])):
                    old = float(label["line_cost"][repeat, 0])
                    alpha_ac_cost = float(label["alpha_ac_cost"][repeat])
                    line_cost = np.asarray(label["line_cost"][repeat], np.float32)
                    safe_index = int(label["safe_index"][repeat])
                    argmin_index = int(label["argmin_index"][repeat])
                    inputs = actor_inputs(
                        source, context, gradient_mean, gradient_std,
                        repeat, old_payload, device,
                    )
                    direction = torch.from_numpy(
                        np.asarray(label["projected_direction"][repeat], np.float32)[None]
                    ).to(device)
                    rho = torch.tensor(
                        [[float(label["requested_rho"][repeat])]],
                        dtype=torch.float32, device=device,
                    )
                    scale = torch.tensor(
                        [[float(label["trust_scale"][repeat])]],
                        dtype=torch.float32, device=device,
                    )
                    with torch.no_grad():
                        embedding = model_embedding(
                            policy, inputs, direction, rho, scale
                        )[0].cpu().numpy()
                    raw = raw_tracking_features(
                        source, float(label["reference_speed_mps"])
                    )
                    feedback = np.asarray(
                        context["first_pass_feedback"][repeat], np.float32
                    )
                    scalars = dict(zip(scalar_names, feedback[-10:].tolist()))
                    nonregression = np.flatnonzero(line_cost <= old + 1e-6)
                    max_nonregression_alpha = (
                        float(label["alpha_grid"][nonregression[-1]])
                        if len(nonregression) else 0.0
                    )
                    first_gain = float(old - line_cost[1])
                    endpoint_gain = float(old - line_cost[-1])
                    row = {
                        "row_index": len(rows),
                        "key": f"{label_path.parent.name}/{label_path.stem}/context_{repeat}",
                        "episode": label_path.parent.name,
                        "step": int(label_path.stem.split("_")[-1]),
                        "context_index": repeat,
                        "scenario": str(label["scenario"]),
                        "reference_speed_mps": float(label["reference_speed_mps"]),
                        **raw,
                        "requested_rho": float(label["requested_rho"][repeat]),
                        "trust_scale": float(label["trust_scale"][repeat]),
                        "old_cost": old,
                        "alpha_ac_cost": alpha_ac_cost,
                        "gain_vs_old": old - alpha_ac_cost,
                        "safe_index": safe_index,
                        "safe_alpha": float(label["alpha_grid"][safe_index]),
                        "argmin_alpha": float(label["alpha_grid"][argmin_index]),
                        "max_nonregression_alpha": max_nonregression_alpha,
                        "alpha_005_gain": first_gain,
                        "endpoint_gain": endpoint_gain,
                        "alpha_ac_probability": float(label["alpha_ac_probability"][repeat]),
                        "alpha_ac_conditional_alpha": float(label["alpha_ac_conditional_alpha"][repeat]),
                        "alpha_ac_alpha": float(label["alpha_ac_alpha"][repeat]),
                        "tr2b_probability": float(label["tr2b_probability"][repeat]),
                        "tr2b_alpha": float(label["tr2b_alpha"][repeat]),
                        "first_base_cost": float(scalars["first_base_cost"]),
                        "first_best_cost": float(scalars["first_best_cost"]),
                        "first_effective_sample_fraction": float(scalars["first_effective_sample_fraction"]),
                        "first_clip_fraction": float(scalars["first_clip_fraction"]),
                        "relative_weighted_fit_error": float(scalars["relative_weighted_fit_error"]),
                    }
                    row["regression"] = row["gain_vs_old"] < 0.0
                    row["failure_type"] = (
                        "false_move_safe_zero"
                        if row["regression"] and safe_index == 0
                        else "overshoot_small_safe_alpha"
                        if row["regression"] and row["safe_alpha"] < row["alpha_ac_alpha"]
                        else "other_regression"
                        if row["regression"] else "non_regression"
                    )
                    rows.append(row)
                    validation_embeddings.append(embedding)
                    validation_line_cost.append(line_cost)
                    validation_directions.append(direction[0].cpu().numpy())

    regression = [row for row in rows if row["regression"]]
    control = [row for row in rows if not row["regression"]]
    if len(regression) != 27:
        raise AssertionError(f"expected 27 regression contexts, found {len(regression)}")

    # Compare the validation contexts with the exact network representation of
    # train-only internal-fit contexts. This tests state/context coverage without
    # inventing a hand-selected raw-feature metric.
    train_data, _, splits = load_dataset(Path(ac_payload["labels"]), old_payload)
    train_tensors = tensorize(train_data, device)
    train_extra = extra_tensors(train_data, device)
    fit_index = np.flatnonzero(np.isin(train_data.episodes, splits["internal_fit"]))
    train_embeddings = []
    with torch.no_grad():
        for start in range(0, len(fit_index), 128):
            absolute = torch.from_numpy(fit_index[start:start + 128]).to(device)
            inputs = tuple(value[absolute] for value in train_tensors["inputs"])
            train_embeddings.append(model_embedding(
                policy, inputs,
                train_extra["direction"][absolute],
                train_extra["rho"][absolute],
                train_extra["scale"][absolute],
            ).cpu())
    train_embedding = torch.cat(train_embeddings).to(device)
    validation_embedding = torch.from_numpy(
        np.asarray(validation_embeddings, np.float32)
    ).to(device)
    mean = train_embedding.mean(0, keepdim=True)
    std = train_embedding.std(0, keepdim=True).clamp_min(1e-4)
    train_standard = (train_embedding - mean) / std
    validation_standard = (validation_embedding - mean) / std
    nearest_distance, nearest_index = [], []
    for start in range(0, len(rows), 50):
        distance = torch.cdist(
            validation_standard[start:start + 50], train_standard
        ) / np.sqrt(train_standard.shape[1])
        value, index = torch.topk(distance, k=5, dim=1, largest=False)
        nearest_distance.append(value.cpu().numpy())
        nearest_index.append(index.cpu().numpy())
    nearest_distance = np.concatenate(nearest_distance)
    nearest_index = np.concatenate(nearest_index)
    fit_safe_alpha = train_data.safe_alpha[fit_index]
    fit_safe_index = train_data.safe_index[fit_index]
    for index, row in enumerate(rows):
        neighbors = nearest_index[index]
        row["embedding_nn1_distance"] = float(nearest_distance[index, 0])
        row["embedding_nn5_mean_distance"] = float(np.mean(nearest_distance[index]))
        row["nearest5_train_safe_alpha_mean"] = float(np.mean(fit_safe_alpha[neighbors]))
        row["nearest5_train_safe_zero_fraction"] = float(np.mean(fit_safe_index[neighbors] == 0))

    regression = [row for row in rows if row["regression"]]
    control = [row for row in rows if not row["regression"]]
    numeric_features = (
        "reference_speed_mps", "vx_mps", "overspeed_mps", "lateral_error_m",
        "heading_error_rad", "yaw_rate_rad_s", "requested_rho", "trust_scale",
        "old_cost", "safe_alpha", "max_nonregression_alpha", "alpha_005_gain",
        "endpoint_gain", "alpha_ac_probability", "alpha_ac_conditional_alpha",
        "first_effective_sample_fraction", "first_clip_fraction",
        "relative_weighted_fit_error", "embedding_nn1_distance",
        "nearest5_train_safe_alpha_mean", "nearest5_train_safe_zero_fraction",
    )
    comparison = {
        name: {"regression": stats(regression, name), "non_regression": stats(control, name)}
        for name in numeric_features
    }
    failure_counts = {
        name: sum(row["failure_type"] == name for row in regression)
        for name in sorted({row["failure_type"] for row in regression})
    }
    by_speed = {
        f"{float(speed):.1f}": int(sum(np.isclose(row["reference_speed_mps"], speed) for row in regression))
        for speed in sorted({row["reference_speed_mps"] for row in rows})
    }
    by_scenario = {
        scenario: sum(row["scenario"] == scenario for row in regression)
        for scenario in sorted({row["scenario"] for row in rows})
    }
    by_episode = {}
    for row in regression:
        by_episode[row["episode"]] = by_episode.get(row["episode"], 0) + 1
    both_contexts = 0
    one_context = 0
    for key in {row["key"].rsplit("/", 1)[0] for row in regression}:
        count = sum(row["key"].rsplit("/", 1)[0] == key for row in regression)
        both_contexts += count == 2
        one_context += count == 1

    # Each frozen snapshot has two independently sampled first-pass contexts but
    # exactly the same physical state and reference.  Single-regression pairs are
    # therefore a controlled test of whether the policy can distinguish a locally
    # safe response direction from a locally unsafe one.
    snapshot_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        snapshot_rows.setdefault(row["key"].rsplit("/", 1)[0], []).append(row)
    paired_contexts = []
    for snapshot, pair in snapshot_rows.items():
        bad = [row for row in pair if row["regression"]]
        good = [row for row in pair if not row["regression"]]
        if len(bad) != 1 or len(good) != 1:
            continue
        bad, companion = bad[0], good[0]
        bad_direction = validation_directions[bad["row_index"]].reshape(-1)
        companion_direction = validation_directions[companion["row_index"]].reshape(-1)
        denominator = np.linalg.norm(bad_direction) * np.linalg.norm(companion_direction)
        cosine = float(np.dot(bad_direction, companion_direction) / max(denominator, 1e-12))
        paired_contexts.append({
            "snapshot": snapshot,
            "direction_cosine": cosine,
            "bad_key": bad["key"],
            "bad_gain_vs_old": bad["gain_vs_old"],
            "bad_safe_alpha": bad["safe_alpha"],
            "bad_endpoint_gain": bad["endpoint_gain"],
            "bad_probability": bad["alpha_ac_probability"],
            "bad_alpha": bad["alpha_ac_alpha"],
            "companion_key": companion["key"],
            "companion_gain_vs_old": companion["gain_vs_old"],
            "companion_safe_alpha": companion["safe_alpha"],
            "companion_endpoint_gain": companion["endpoint_gain"],
            "companion_probability": companion["alpha_ac_probability"],
            "companion_alpha": companion["alpha_ac_alpha"],
        })

    def paired_stats(prefix: str, name: str) -> dict[str, float]:
        return distribution(np.asarray([
            pair[f"{prefix}_{name}"] for pair in paired_contexts
        ], np.float64))

    validation_label_distribution = {}
    for speed in sorted({row["reference_speed_mps"] for row in rows}):
        subset = [row for row in rows if np.isclose(row["reference_speed_mps"], speed)]
        safe_zero = [row for row in subset if row["safe_index"] == 0]
        validation_label_distribution[f"{float(speed):.1f}"] = {
            "context_count": len(subset),
            "safe_zero_count": len(safe_zero),
            "safe_zero_fraction": len(safe_zero) / len(subset),
            "actor_move_count": sum(row["alpha_ac_alpha"] > 0 for row in subset),
            "actor_move_on_safe_zero_count": sum(
                row["alpha_ac_alpha"] > 0 and row["safe_index"] == 0 for row in subset
            ),
            "regression_count": sum(row["regression"] for row in subset),
            "false_move_on_safe_zero_count": sum(
                row["regression"] and row["safe_index"] == 0 for row in subset
            ),
        }

    def dataset_label_distribution(index: np.ndarray) -> dict[str, Any]:
        result = {}
        for speed in sorted(set(train_data.reference_speed[index].tolist())):
            selected = index[np.isclose(train_data.reference_speed[index], speed)]
            result[f"{float(speed):.1f}"] = {
                "context_count": int(len(selected)),
                "safe_zero_count": int(np.sum(train_data.safe_index[selected] == 0)),
                "safe_zero_fraction": float(np.mean(train_data.safe_index[selected] == 0)),
                "safe_alpha_mean": float(np.mean(train_data.safe_alpha[selected])),
            }
        result["all"] = {
            "context_count": int(len(index)),
            "safe_zero_count": int(np.sum(train_data.safe_index[index] == 0)),
            "safe_zero_fraction": float(np.mean(train_data.safe_index[index] == 0)),
            "safe_alpha_mean": float(np.mean(train_data.safe_alpha[index])),
        }
        return result

    selection_index = np.flatnonzero(
        np.isin(train_data.episodes, splits["internal_selection"])
    )
    validation_label_distribution["all"] = {
        "context_count": len(rows),
        "safe_zero_count": sum(row["safe_index"] == 0 for row in rows),
        "safe_zero_fraction": float(np.mean([row["safe_index"] == 0 for row in rows])),
        "actor_move_count": sum(row["alpha_ac_alpha"] > 0 for row in rows),
        "actor_move_on_safe_zero_count": sum(
            row["alpha_ac_alpha"] > 0 and row["safe_index"] == 0 for row in rows
        ),
        "regression_count": len(regression),
        "false_move_on_safe_zero_count": failure_counts.get("false_move_safe_zero", 0),
    }

    csv_fields = [name for name in regression[0] if name != "row_index"]
    with (args.run / "regression_contexts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in sorted(regression, key=lambda value: value["gain_vs_old"]):
            writer.writerow({name: row[name] for name in csv_fields})
    regression_indices = np.asarray([row["row_index"] for row in regression], np.int64)
    np.savez_compressed(
        args.run / "regression_contexts.npz",
        keys=np.asarray([row["key"] for row in regression]),
        line_cost=np.asarray(validation_line_cost, np.float32)[regression_indices],
        alpha_grid=np.linspace(0.0, 1.0, 21, dtype=np.float32),
        validation_embedding=np.asarray(validation_embeddings, np.float32)[regression_indices],
        nearest_train_fit_index=fit_index[nearest_index[regression_indices]],
        nearest_distance=nearest_distance[regression_indices],
    )

    report = {
        "format_version": 2,
        "run": str(args.run.resolve()),
        "context_count": len(rows),
        "regression_context_count": len(regression),
        "non_regression_context_count": len(control),
        "failure_type_count": failure_counts,
        "regression_by_speed": by_speed,
        "regression_by_scenario": by_scenario,
        "regression_by_episode": dict(sorted(by_episode.items(), key=lambda item: (-item[1], item[0]))),
        "regression_snapshot_pattern": {
            "snapshots_with_both_contexts_regressing": both_contexts,
            "snapshots_with_one_context_regressing": one_context,
        },
        "paired_context_analysis": {
            "pair_count": len(paired_contexts),
            "direction_cosine": distribution(np.asarray([
                pair["direction_cosine"] for pair in paired_contexts
            ], np.float64)),
            "bad": {
                name: paired_stats("bad", name)
                for name in ("gain_vs_old", "safe_alpha", "endpoint_gain", "probability", "alpha")
            },
            "nonregressing_companion": {
                name: paired_stats("companion", name)
                for name in ("gain_vs_old", "safe_alpha", "endpoint_gain", "probability", "alpha")
            },
            "companion_move_count": sum(pair["companion_alpha"] > 0 for pair in paired_contexts),
            "companion_stay_count": sum(pair["companion_alpha"] == 0 for pair in paired_contexts),
            "examples_sorted_by_bad_gain": sorted(
                paired_contexts, key=lambda pair: pair["bad_gain_vs_old"]
            )[:10],
        },
        "label_distribution": {
            "formal_validation_by_speed": validation_label_distribution,
            "train_internal_fit_by_speed": dataset_label_distribution(fit_index),
            "train_internal_selection_by_speed": dataset_label_distribution(selection_index),
        },
        "feature_comparison": comparison,
        "worst_contexts": sorted(regression, key=lambda row: row["gain_vs_old"])[:10],
        "diagnosis": {
            "primary": "move gate admits endpoint-like alpha on contexts whose line optimum is stay or a much smaller step",
            "safe_zero_false_move_count": failure_counts.get("false_move_safe_zero", 0),
            "overshoot_count": failure_counts.get("overshoot_small_safe_alpha", 0),
            "coverage_interpretation": "reported from learned-embedding nearest-neighbor statistics; compare regression and non-regression distances before calling the cases OOD",
        },
        "test_policy": "test episodes 105--119 not opened",
    }
    (args.run / "regression_analysis.json").write_text(json.dumps(report, indent=2) + "\n")

    markdown = [
        "# Alpha-AC TR3：27 个正式验证回退上下文的独立分析",
        "",
        "本报告只使用正式验证集 episode 090--104；test episode 105--119 未打开。",
        "",
        "## 结论",
        "",
        f"- 27 个回退全部属于步长决策错误：{failure_counts.get('false_move_safe_zero', 0)} 个本应保持 `alpha=0`，"
        f"{failure_counts.get('overshoot_small_safe_alpha', 0)} 个只允许小步；策略却都输出接近 1 的步长。",
        f"- 回退集中在较高参考速度：2.0/2.4/2.8 m/s 分别为 {by_speed['2.0']}/{by_speed['2.4']}/{by_speed['2.8']} 个，"
        "1.2 和 1.6 m/s 均为 0。",
        f"- {both_contexts} 个快照的两个上下文都回退，{one_context} 个快照只回退其中一个上下文。"
        f"后者的方向余弦相似度中位数为 {np.median([pair['direction_cosine'] for pair in paired_contexts]):.3f}，"
        "证明相近方向仍可能有相反的局部 cost 响应。",
        f"- 19 个成对样本中，坏方向端点收益中位数为 {np.median([pair['bad_endpoint_gain'] for pair in paired_contexts]):.3f}，"
        f"非回退配对方向为 {np.median([pair['companion_endpoint_gain'] for pair in paired_contexts]):.3f}；"
        f"但坏方向移动概率反而更高（均值 {np.mean([pair['bad_probability'] for pair in paired_contexts]):.3f} 对 "
        f"{np.mean([pair['companion_probability'] for pair in paired_contexts]):.3f}）。",
        "- 学到的 embedding 并未把这些样本标成明显 OOD；其最近 5 个训练近邻的 `safe_alpha` 中位数均值为 "
        f"{np.median([row['nearest5_train_safe_alpha_mean'] for row in regression]):.3f}，而 27 个样本真实 `safe_alpha` 中位数为 "
        f"{np.median([row['safe_alpha'] for row in regression]):.3f}。这是表示混叠，而不只是近邻距离过大。",
        "- 阈值选择集明显偏容易：其 `safe_alpha=0` 占比仅 "
        f"{100 * dataset_label_distribution(selection_index)['all']['safe_zero_fraction']:.1f}%，训练 fit 为 "
        f"{100 * dataset_label_distribution(fit_index)['all']['safe_zero_fraction']:.1f}%，正式验证为 "
        f"{100 * validation_label_distribution['all']['safe_zero_fraction']:.1f}%。因此 0.88 门限在内部选择集上过于乐观。",
        "",
        "## 速度分布与标签偏移",
        "",
        "|参考速度 (m/s)|正式验证 safe=0|训练 fit safe=0|内部选择 safe=0|回退数|",
        "|---:|---:|---:|---:|---:|",
    ]
    fit_distribution = dataset_label_distribution(fit_index)
    selection_distribution = dataset_label_distribution(selection_index)
    for speed in ("1.2", "1.6", "2.0", "2.4", "2.8"):
        markdown.append(
            f"|{speed}|{100 * validation_label_distribution[speed]['safe_zero_fraction']:.1f}%|"
            f"{100 * fit_distribution[speed]['safe_zero_fraction']:.1f}%|"
            f"{100 * selection_distribution[speed]['safe_zero_fraction']:.1f}%|{by_speed[speed]}|"
        )
    markdown.extend([
        "",
        "## 27 个上下文",
        "",
        "`gain = old_cost - Alpha-AC cost`，负数表示回退。",
        "",
        "|#|上下文|速度|场景|类型|old|AC|gain|safe alpha|P(move)|Actor alpha|fit error|NN5 train alpha|",
        "|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for rank, row in enumerate(sorted(regression, key=lambda value: value["gain_vs_old"]), 1):
        failure = "应保持" if row["failure_type"] == "false_move_safe_zero" else "步长过冲"
        markdown.append(
            f"|{rank}|`{row['key']}`|{row['reference_speed_mps']:.1f}|{row['scenario']}|{failure}|"
            f"{row['old_cost']:.3f}|{row['alpha_ac_cost']:.3f}|{row['gain_vs_old']:.3f}|"
            f"{row['safe_alpha']:.2f}|{row['alpha_ac_probability']:.3f}|{row['alpha_ac_alpha']:.3f}|"
            f"{row['relative_weighted_fit_error']:.3f}|{row['nearest5_train_safe_alpha_mean']:.3f}|"
        )
    markdown.extend([
        "",
        "## 根因判断",
        "",
        "当前 Actor 实际退化成了近似二值决策：门控不通过就是 0，通过后 conditional alpha 几乎恒为 1。"
        "27 个回退的 `Actor alpha` 最小值仍为 "
        f"{min(row['alpha_ac_alpha'] for row in regression):.3f}。真正失效的是两层保护："
        "门控没有识别局部坏方向，连续步长头也没有在不确定时缩小步长。",
        "",
        "首轮响应拟合误差在回退集更高（中位数 "
        f"{np.median([row['relative_weighted_fit_error'] for row in regression]):.3f} 对 "
        f"{np.median([row['relative_weighted_fit_error'] for row in control]):.3f}），但当前策略没有把它可靠地转化为保守动作。"
        "结合训练/验证 safe-zero 比例偏移和近邻标签反转，首要问题是局部响应覆盖与风险校准，"
        "不是简单增加 Actor 宽度，也不是把阈值整体调高就能根治。",
        "",
        "建议下一步仅在 train-only 数据中补充这类 hard negative：优先 2.0--2.8 m/s、较高拟合误差、"
        "相近方向但 line-cost 响应翻转的成对上下文；训练 gate 预测安全步长上界或 cost 增量置信下界，"
        "并用独立且难度匹配的内部校准集选择阈值。正式 test 仍保持封存。",
        "",
    ])
    (args.run / "regression_analysis.md").write_text("\n".join(markdown))

    # Compact evidence figure focused exclusively on the 27 regressions and
    # their matched/train-only controls.
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    alpha = np.linspace(0.0, 1.0, 21)
    for row in regression:
        curve = np.asarray(validation_line_cost[row["row_index"]]) - row["old_cost"]
        axes[0, 0].plot(alpha, curve, alpha=0.35, color="tab:red")
        axes[0, 0].scatter(row["alpha_ac_alpha"], row["alpha_ac_cost"] - row["old_cost"], s=10, color="black")
    axes[0, 0].axhline(0, color="gray", linewidth=1)
    axes[0, 0].set(title="27 regression line-cost curves", xlabel="alpha", ylabel="cost(alpha) - old cost")

    axes[0, 1].bar(
        ["safe alpha = 0", "small safe alpha"],
        [failure_counts["false_move_safe_zero"], failure_counts["overshoot_small_safe_alpha"]],
        color=["#d73027", "#fc8d59"],
    )
    axes[0, 1].set(title="Failure mechanism", ylabel="regression contexts")

    speed_labels = sorted(by_speed, key=float)
    axes[0, 2].bar(speed_labels, [by_speed[name] for name in speed_labels], color="tab:orange")
    axes[0, 2].set(title="Regression count by reference speed", xlabel="m/s", ylabel="contexts")

    paired_scatter = axes[1, 0].scatter(
        [pair["bad_endpoint_gain"] for pair in paired_contexts],
        [pair["companion_endpoint_gain"] for pair in paired_contexts],
        c=[pair["direction_cosine"] for pair in paired_contexts],
        cmap="viridis", vmin=0.8, vmax=1.0, s=42,
    )
    axes[1, 0].axhline(0, color="gray", linewidth=1)
    axes[1, 0].axvline(0, color="gray", linewidth=1)
    axes[1, 0].set(
        title="Same-state paired response flip",
        xlabel="bad direction endpoint gain", ylabel="companion endpoint gain",
    )
    figure.colorbar(paired_scatter, ax=axes[1, 0], label="direction cosine")

    type_color = {
        "false_move_safe_zero": "#d73027",
        "overshoot_small_safe_alpha": "#fc8d59",
    }
    axes[1, 1].scatter(
        [row["safe_alpha"] for row in regression],
        [row["alpha_ac_probability"] for row in regression],
        c=[type_color[row["failure_type"]] for row in regression], s=40,
    )
    axes[1, 1].axhline(float(config["alpha_ac_threshold"]), color="black", linestyle="--")
    axes[1, 1].set(
        title="Gate is confident on unsafe steps", xlabel="formal safe alpha",
        ylabel="Actor P(move)", xlim=(-0.02, 0.38),
    )

    axes[1, 2].scatter(
        [row["safe_alpha"] for row in regression],
        [row["nearest5_train_safe_alpha_mean"] for row in regression],
        c=[type_color[row["failure_type"]] for row in regression], s=40,
    )
    axes[1, 2].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[1, 2].set(
        title="Learned-neighbor label reversal", xlabel="formal safe alpha",
        ylabel="nearest-5 train safe alpha", xlim=(-0.02, 0.38), ylim=(0, 1.02),
    )
    figure.suptitle("Alpha-AC TR3: isolated analysis of 27 regressions")
    figure.savefig(args.run / "regression_analysis.png", dpi=170)
    plt.close(figure)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
