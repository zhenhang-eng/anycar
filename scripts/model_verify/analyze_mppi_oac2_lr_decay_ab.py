#!/usr/bin/env python3
"""Analyze the paired fixed versus staged-decay Actor LR OAC experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


DEFAULT_BASELINE = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_200round_20260825_v1"
)
DEFAULT_SCHEDULED = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_lrdecay_cuda_200round_20260827_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/oac2_lr_decay_ab_20260827_v1"
)
CHECKPOINTS = (10, 20, 30, 50, 100, 150, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--scheduled", type=Path, default=DEFAULT_SCHEDULED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get(data: dict, *keys: str) -> float:
    value = data
    for key in keys:
        value = value[key]
    return float(value)


def summarize_metrics(metrics: list[dict]) -> dict:
    fields = {
        "mean_cost": ("cost", "mean"),
        "median_cost": ("cost", "median"),
        "mean_gain": ("gain_vs_initial", "mean"),
        "median_gain": ("gain_vs_initial", "median"),
        "p05_gain": ("gain_vs_initial", "p05"),
        "worst_gain": ("gain_vs_initial", "minimum"),
        "regression_fraction": ("regression_fraction",),
        "headroom_recovery": ("headroom_recovery_vs_bank_best",),
        "speed_2_4_mean_gain": ("by_speed", "2.4", "mean_gain"),
        "speed_2_4_p05_gain": ("by_speed", "2.4", "p05_gain"),
        "speed_2_8_mean_gain": ("by_speed", "2.8", "mean_gain"),
        "speed_2_8_p05_gain": ("by_speed", "2.8", "p05_gain"),
        "guard_mean_cost": ("two_center_guard", "cost", "mean"),
        "guard_gain_vs_warm": ("two_center_guard", "gain_vs_warm", "mean"),
        "guard_warm_selected_fraction": (
            "two_center_guard", "warm_selected_fraction",
        ),
    }
    result = {
        name: mean(get(row, *path) for row in metrics)
        for name, path in fields.items()
    }
    result["per_seed"] = [
        {name: get(row, *path) for name, path in fields.items()}
        for row in metrics
    ]
    return result


def evaluation(record: dict, checkpoint: int) -> dict:
    return next(
        row for row in record["evaluations"]
        if int(row["round"]) == checkpoint
    )


def load_run(path: Path) -> dict:
    contract = json.loads((path / "contract.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    records = summary["records"]
    curve = {
        str(checkpoint): summarize_metrics([
            evaluation(record, checkpoint)["metrics"] for record in records
        ])
        for checkpoint in CHECKPOINTS
    }
    gate_failure_counts = {}
    for checkpoint in CHECKPOINTS:
        rows = [evaluation(record, checkpoint) for record in records]
        keys = sorted({key for row in rows for key in row["acceptance_gate"]})
        gate_failure_counts[str(checkpoint)] = {
            key: sum(not bool(row["acceptance_gate"][key]) for row in rows)
            for key in keys
        }
    windows = {}
    for start, stop in ((1, 20), (21, 80), (81, 200)):
        actor_rows = [
            row["actor"] for record in records for row in record["rounds"]
            if start <= int(row["round"]) <= stop
        ]
        windows[f"rounds_{start}_{stop}"] = {
            "learning_rate": mean(float(row.get(
                "learning_rate", contract["arguments"]["actor_learning_rate"]
            )) for row in actor_rows),
            "step_sigma_rms": mean(float(row["step_sigma_rms"]) for row in actor_rows),
            "projection_fraction": mean(
                float(row["trust_projection"] < 1.0) for row in actor_rows
            ),
            "critic_weight_ess_fraction": mean(
                float(row["actor_cost_weight_ess_fraction"]) for row in actor_rows
            ),
        }
    return {
        "path": str(path.resolve()),
        "contract_sha256": sha256(path / "contract.json"),
        "summary_sha256": sha256(path / "summary.json"),
        "formal_validation_loaded": bool(summary["formal_validation_loaded"]),
        "test_loaded": bool(summary["test_loaded"]),
        "selected_rounds": [int(row["selected_round"]) for row in records],
        "initial_actor_hashes": [row["initial_actor_module_sha256"] for row in records],
        "curve": curve,
        "latest": summarize_metrics([row["latest_metrics"] for row in records]),
        "selected": summarize_metrics([row["selected_metrics"] for row in records]),
        "gate_failure_seed_counts": gate_failure_counts,
        "actor_windows": windows,
    }


def paired_delta(candidate: dict, baseline: dict, role: str) -> dict:
    fields = (
        "mean_cost", "median_cost", "mean_gain", "median_gain", "p05_gain",
        "worst_gain", "regression_fraction", "headroom_recovery",
        "speed_2_4_mean_gain", "speed_2_4_p05_gain",
        "speed_2_8_mean_gain", "speed_2_8_p05_gain", "guard_mean_cost",
        "guard_warm_selected_fraction",
    )
    result = {
        field: candidate[role][field] - baseline[role][field]
        for field in fields
    }
    higher_is_better = (
        "mean_gain", "median_gain", "p05_gain", "worst_gain",
        "headroom_recovery", "speed_2_4_mean_gain", "speed_2_4_p05_gain",
        "speed_2_8_mean_gain", "speed_2_8_p05_gain",
    )
    lower_is_better = (
        "mean_cost", "median_cost", "regression_fraction", "guard_mean_cost",
        "guard_warm_selected_fraction",
    )
    for field in higher_is_better:
        result[f"{field}_improved_seed_count"] = sum(
            candidate[role]["per_seed"][seed][field]
            > baseline[role]["per_seed"][seed][field]
            for seed in range(3)
        )
    for field in lower_is_better:
        result[f"{field}_improved_seed_count"] = sum(
            candidate[role]["per_seed"][seed][field]
            < baseline[role]["per_seed"][seed][field]
            for seed in range(3)
        )
    return result


def main() -> None:
    args = parse_args()
    baseline_contract = json.loads((args.baseline / "contract.json").read_text())
    scheduled_contract = json.loads((args.scheduled / "contract.json").read_text())
    baseline_args = baseline_contract["arguments"]
    scheduled_args = scheduled_contract["arguments"]
    ignored = {
        "output_dir", "actor_learning_rate_schedule",
        "actor_learning_rate_initial", "actor_learning_rate_middle",
        "actor_learning_rate_initial_rounds", "actor_learning_rate_middle_round",
    }
    paired = all(
        scheduled_args.get(key) == value
        for key, value in baseline_args.items()
        if key not in ignored
    )
    baseline = load_run(args.baseline)
    scheduled = load_run(args.scheduled)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "LR_DECAY_MEAN_ACCELERATION_PASS_TAIL_SAFE_SELECTION_FAIL",
        "scope": (
            "fold-1 train-side internal-selection, gamma=1 box3 adaptive-tail, "
            "200 Actor rounds x 3 seeds; formal validation/test sealed"
        ),
        "checks": {
            "paired_except_actor_lr_schedule_and_output": paired,
            "initial_actor_hashes_match": (
                baseline["initial_actor_hashes"] == scheduled["initial_actor_hashes"]
            ),
            "formal_validation_sealed": not scheduled["formal_validation_loaded"],
            "test_sealed": not scheduled["test_loaded"],
            "scheduled_selected_none": scheduled["selected_rounds"] == [0, 0, 0],
            "latest_mean_improves_3_of_3": all(
                scheduled["latest"]["per_seed"][seed]["mean_gain"]
                > baseline["latest"]["per_seed"][seed]["mean_gain"]
                for seed in range(3)
            ),
        },
        "runs": {"fixed_lr_1e_6": baseline, "staged_cosine_lr": scheduled},
        "paired_delta": {
            "latest": paired_delta(scheduled, baseline, "latest"),
            "selected": paired_delta(scheduled, baseline, "selected"),
        },
        "decision": {
            "mean_capacity": (
                "initial large LR with decay materially accelerates direct mean and median "
                "optimization; fixed 1e-6 was too conservative"
            ),
            "critic_tracking": (
                "actor-visited pair accuracy remains near the historical level and no trust "
                "projection activates; rapid Actor movement does not immediately break Critic tracking"
            ),
            "tail": (
                "fewer states regress, but high-speed P05 and average worst severity deteriorate; "
                "all scheduled checkpoints fail the registered tail-safe selection contract"
            ),
            "next": (
                "retain LR decay as a mean-capacity mechanism, but do not authorize this schedule "
                "as the selected Actor; next isolate early schedule severity or apply per-state "
                "DBM/bank safe projection rather than another shared tail-weight sweep"
            ),
        },
    }
    if not all(result["checks"].values()):
        raise AssertionError(result["checks"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "qualification": result["qualification"],
        "fixed_latest": baseline["latest"],
        "scheduled_latest": scheduled["latest"],
        "paired_latest": result["paired_delta"]["latest"],
        "scheduled_selected_rounds": scheduled["selected_rounds"],
    }, indent=2))


if __name__ == "__main__":
    main()
