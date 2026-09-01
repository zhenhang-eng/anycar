#!/usr/bin/env python3
"""Analyze the frozen cost-sensitive J16 Actor on formal validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_mppi_j16_cost_sensitive_distillation import (
    DEFAULT_CHECKPOINT,
    OLD_FEEDBACK,
    OLD_RISK,
    OLD_SOURCE,
    VALIDATION_GT,
    actor_centers,
    evaluate_direct,
    load_actor,
    load_partition,
)


DEFAULT_ACTOR = Path(
    "outputs/mppi_proposal/j16_cost_sensitive_expansion_20260807_v2/actor_seed1.pt"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/j16_cost_sensitive_expansion_20260807_v2/"
    "validation_analysis.json"
)
DEFAULT_T1 = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_teacher_t1_diverse_20260805_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def distribution(value: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "p95": float(np.quantile(value, 0.95)),
        "maximum": float(np.max(value)),
    }


def gain_metrics(base: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    gain = base - candidate
    return {
        "mean": float(np.mean(gain)),
        "median": float(np.median(gain)),
        "p05": float(np.quantile(gain, 0.05)),
        "worst": float(np.min(gain)),
        "win_fraction": float(np.mean(gain > 0)),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    parent = torch.load(DEFAULT_CHECKPOINT, map_location="cpu")
    actor_payload = torch.load(args.actor, map_location="cpu")
    splits = json.loads((OLD_FEEDBACK / "splits.json").read_text())
    data, gt_summary = load_partition(
        OLD_SOURCE,
        OLD_FEEDBACK,
        OLD_RISK,
        VALIDATION_GT,
        None,
        splits["validation"],
        parent,
        6.0,
        "validation",
    )
    current_actor = load_actor(parent, 2.0, device)
    new_actor = load_actor(actor_payload, 6.0, device)
    current = actor_centers(current_actor, data, 100, device)
    new = actor_centers(new_actor, data, 100, device)
    warm, teacher = [], []
    for source_path in data.source_paths:
        with np.load(source_path, allow_pickle=False) as source, np.load(
            DEFAULT_T1 / source_path.parents[1].name / source_path.name,
            allow_pickle=False,
        ) as label:
            warm.append(np.asarray(source["sampling_mean_knots"], np.float32))
            teacher.append(np.asarray(label["teacher_center_knots"], np.float32))
    alphas = np.linspace(0.0, 1.0, 21)
    methods = {
        "warm": np.asarray(warm),
        "current_actor": current,
        "new_actor": new,
        "teacher": np.asarray(teacher),
        "projected_j16": data.target_center,
        **{
            f"blend_{alpha:.2f}": np.clip(
                current + alpha * (new - current), -1.0, 1.0
            ).astype(np.float32)
            for alpha in alphas
        },
    }
    cost = evaluate_direct(methods, data, 100, device)
    base = cost["current_actor"]
    trust_scan = []
    for alpha in alphas:
        name = f"blend_{alpha:.2f}"
        trust_scan.append({
            "alpha": float(alpha),
            "cost": distribution(cost[name]),
            "gain_vs_current": gain_metrics(base, cost[name]),
        })
    best = min(trust_scan, key=lambda row: row["cost"]["mean"])
    plan = json.loads((OLD_SOURCE / "scenario_plan.json").read_text())
    metadata = {
        row["episode_id"]: (
            float(row["reference_speed_mps"]), str(row["scenario_class"])
        )
        for row in plan["episodes"]
    }
    grouped = {"speed": {}, "scenario": {}}
    for value in sorted({metadata[episode][0] for episode in set(data.episodes)}):
        mask = np.asarray([metadata[episode][0] == value for episode in data.episodes])
        grouped["speed"][str(value)] = {
            name: distribution(one[mask]) for name, one in cost.items()
            if name in ("warm", "current_actor", "new_actor", "teacher", "projected_j16")
        }
        grouped["speed"][str(value)]["new_vs_current_gain"] = gain_metrics(
            base[mask], cost["new_actor"][mask]
        )
    for value in sorted({metadata[episode][1] for episode in set(data.episodes)}):
        mask = np.asarray([metadata[episode][1] == value for episode in data.episodes])
        grouped["scenario"][value] = {
            name: distribution(one[mask]) for name, one in cost.items()
            if name in ("warm", "current_actor", "new_actor", "teacher", "projected_j16")
        }
        grouped["scenario"][value]["new_vs_current_gain"] = gain_metrics(
            base[mask], cost["new_actor"][mask]
        )
    result = {
        "format_version": 1,
        "qualification": "VALIDATION_ONLY_TEST_SEALED",
        "actor": str(args.actor.resolve()),
        "validation_snapshots": gt_summary["snapshot_count"],
        "validation_contexts": len(data.target_action),
        "methods": {
            name: distribution(cost[name])
            for name in ("warm", "current_actor", "new_actor", "teacher", "projected_j16")
        },
        "new_vs_current_gain": gain_metrics(base, cost["new_actor"]),
        "trust_scan": trust_scan,
        "best_mean_trust_step": best,
        "grouped": grouped,
        "conclusion": (
            "The learned direction improves low/mid speed and supports a mean-cost "
            "trust step, but P05/worst tail gates fail; reject deployment."
        ),
        "test_policy": "test split not loaded or evaluated",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "methods": result["methods"],
        "new_vs_current_gain": result["new_vs_current_gain"],
        "best_mean_trust_step": result["best_mean_trust_step"],
        "test_policy": result["test_policy"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
