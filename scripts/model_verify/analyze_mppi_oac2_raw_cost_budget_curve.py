#!/usr/bin/env python3
"""Analyze the paired 20/100/200-round OAC raw-cost aggregation curve."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


DEFAULT_RUNS = {
    "gamma_0": Path(
        "outputs/mppi_proposal/online_absolute_sac_oac2_box3_adaptive_tail_200round_20260825_v1"
    ),
    "gamma_0_5": Path(
        "outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma05_200round_20260825_v1"
    ),
    "gamma_1": Path(
        "outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_200round_20260825_v1"
    ),
}
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/oac2_raw_cost_budget_curve_20260827_v1"
)
CHECKPOINTS = (20, 100, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma-0", type=Path, default=DEFAULT_RUNS["gamma_0"])
    parser.add_argument("--gamma-0-5", type=Path, default=DEFAULT_RUNS["gamma_0_5"])
    parser.add_argument("--gamma-1", type=Path, default=DEFAULT_RUNS["gamma_1"])
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
    result = {
        "mean_cost": mean(get(row, "cost", "mean") for row in metrics),
        "median_cost": mean(get(row, "cost", "median") for row in metrics),
        "mean_gain": mean(get(row, "gain_vs_initial", "mean") for row in metrics),
        "median_gain": mean(get(row, "gain_vs_initial", "median") for row in metrics),
        "p05_gain": mean(get(row, "gain_vs_initial", "p05") for row in metrics),
        "worst_gain": mean(get(row, "gain_vs_initial", "minimum") for row in metrics),
        "regression_fraction": mean(get(row, "regression_fraction") for row in metrics),
        "headroom_recovery": mean(
            get(row, "headroom_recovery_vs_bank_best") for row in metrics
        ),
        "speed_2_4_mean_gain": mean(
            get(row, "by_speed", "2.4", "mean_gain") for row in metrics
        ),
        "speed_2_4_p05_gain": mean(
            get(row, "by_speed", "2.4", "p05_gain") for row in metrics
        ),
        "speed_2_8_mean_gain": mean(
            get(row, "by_speed", "2.8", "mean_gain") for row in metrics
        ),
        "speed_2_8_p05_gain": mean(
            get(row, "by_speed", "2.8", "p05_gain") for row in metrics
        ),
        "guard_mean_cost": mean(
            get(row, "two_center_guard", "cost", "mean") for row in metrics
        ),
        "guard_gain_vs_warm": mean(
            get(row, "two_center_guard", "gain_vs_warm", "mean") for row in metrics
        ),
        "guard_warm_selected_fraction": mean(
            get(row, "two_center_guard", "warm_selected_fraction") for row in metrics
        ),
    }
    result["per_seed"] = [
        {
            "mean_gain": get(row, "gain_vs_initial", "mean"),
            "median_gain": get(row, "gain_vs_initial", "median"),
            "p05_gain": get(row, "gain_vs_initial", "p05"),
            "worst_gain": get(row, "gain_vs_initial", "minimum"),
            "speed_2_4_p05_gain": get(row, "by_speed", "2.4", "p05_gain"),
            "speed_2_8_p05_gain": get(row, "by_speed", "2.8", "p05_gain"),
            "guard_mean_cost": get(row, "two_center_guard", "cost", "mean"),
        }
        for row in metrics
    ]
    return result


def load_run(path: Path, gamma: float) -> dict:
    contract = json.loads((path / "contract.json").read_text())
    validator = json.loads((path / "validator_report.json").read_text())
    records = [
        json.loads((path / f"seed_{seed}" / "summary.json").read_text())
        for seed in range(3)
    ]
    curve = {}
    gate_failures = {}
    for checkpoint in CHECKPOINTS:
        evaluations = [
            next(row for row in record["evaluations"] if int(row["round"]) == checkpoint)
            for record in records
        ]
        curve[str(checkpoint)] = summarize_metrics([row["metrics"] for row in evaluations])
        gate_failures[str(checkpoint)] = {
            key: sum(not bool(row["acceptance_gate"].get(key, True)) for row in evaluations)
            for key in sorted({
                key for row in evaluations for key in row["acceptance_gate"]
            })
        }
    selected = summarize_metrics([record["selected_metrics"] for record in records])
    latest = summarize_metrics([record["latest_metrics"] for record in records])
    initial = summarize_metrics([record["initial_metrics"] for record in records])
    actor_diagnostics = {}
    keys = (
        "actor_cost_weight_ess_fraction", "actor_cost_weight_cap_fraction",
        "step_sigma_rms", "tail_regression_cvar", "tail_lagrange_after",
    )
    for start, stop in ((1, 20), (81, 100), (181, 200)):
        window = []
        for record in records:
            window.extend(
                row["actor"] for row in record["rounds"]
                if start <= int(row["round"]) <= stop
            )
        actor_diagnostics[f"rounds_{start}_{stop}"] = {
            key: mean(float(row.get(
                key,
                1.0 if key == "actor_cost_weight_ess_fraction" else 0.0,
            )) for row in window)
            for key in keys
        }
    return {
        "path": str(path.resolve()),
        "gamma": gamma,
        "contract_sha256": sha256(path / "contract.json"),
        "validator_sha256": sha256(path / "validator_report.json"),
        "validator_passed": bool(validator["passed"]),
        "formal_validation_loaded": bool(contract.get("formal_validation_loaded", False)),
        "test_loaded": bool(contract.get("test_loaded", False)),
        "selected_rounds": [int(record["selected_round"]) for record in records],
        "initial": initial,
        "curve": curve,
        "latest": latest,
        "selected": selected,
        "gate_failure_seed_counts": gate_failures,
        "actor_diagnostics": actor_diagnostics,
    }


def delta(candidate: dict, baseline: dict, role: str) -> dict:
    fields = (
        "mean_cost", "median_cost", "mean_gain", "median_gain", "p05_gain",
        "worst_gain", "regression_fraction", "headroom_recovery",
        "speed_2_4_mean_gain", "speed_2_4_p05_gain",
        "speed_2_8_mean_gain", "speed_2_8_p05_gain", "guard_mean_cost",
    )
    result = {key: candidate[role][key] - baseline[role][key] for key in fields}
    for field in ("mean_gain", "p05_gain", "worst_gain", "guard_mean_cost"):
        if field == "guard_mean_cost":
            result[f"{field}_improved_seed_count"] = sum(
                candidate[role]["per_seed"][seed][field]
                < baseline[role]["per_seed"][seed][field]
                for seed in range(3)
            )
        else:
            result[f"{field}_improved_seed_count"] = sum(
                candidate[role]["per_seed"][seed][field]
                > baseline[role]["per_seed"][seed][field]
                for seed in range(3)
            )
    return result


def main() -> None:
    args = parse_args()
    paths = {
        "gamma_0": args.gamma_0,
        "gamma_0_5": args.gamma_0_5,
        "gamma_1": args.gamma_1,
    }
    gammas = {"gamma_0": 0.0, "gamma_0_5": 0.5, "gamma_1": 1.0}
    runs = {name: load_run(path, gammas[name]) for name, path in paths.items()}
    baseline_args = json.loads((args.gamma_0 / "contract.json").read_text())["arguments"]
    ignored = {"output_dir", "actor_cost_weight_gamma", "actor_cost_weight_maximum"}
    paired = True
    for name in ("gamma_0_5", "gamma_1"):
        candidate_args = json.loads((paths[name] / "contract.json").read_text())["arguments"]
        paired &= all(
            candidate_args.get(key) == value
            for key, value in baseline_args.items()
            if key not in ignored
        )
    gamma0 = runs["gamma_0"]
    gamma1 = runs["gamma_1"]
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "RAW_MEAN_LONG_RUN_DIRECT_GAIN_UP_GUARDED_GAIN_FLAT_TAIL_WORSE",
        "scope": (
            "fold-1 train-side internal-selection OAC, 200 rounds x 3 seeds; "
            "formal validation and test sealed"
        ),
        "checks": {
            "paired_except_gamma_output_and_legacy_missing_fields": paired,
            "all_validators_passed": all(run["validator_passed"] for run in runs.values()),
            "all_have_20_100_200": all(
                set(run["curve"]) == {"20", "100", "200"} for run in runs.values()
            ),
        },
        "runs": runs,
        "paired_vs_gamma_0": {
            "gamma_0_5_latest": delta(runs["gamma_0_5"], gamma0, "latest"),
            "gamma_0_5_selected": delta(runs["gamma_0_5"], gamma0, "selected"),
            "gamma_1_latest": delta(gamma1, gamma0, "latest"),
            "gamma_1_selected": delta(gamma1, gamma0, "selected"),
        },
        "decision": {
            "twenty_rounds": "insufficient for capacity or safe-checkpoint conclusions",
            "direct_objective": (
                "gamma=1 improves 200-round latest mean direct gain, but median/tail regress"
            ),
            "deployment_objective": (
                "two-center guarded mean cost is effectively flat across gamma; raw weighting "
                "spends capacity on high-cost Actor states that warm often rejects"
            ),
            "next": (
                "do not add more rounds or raw weight first; audit guard-rejected high-weight "
                "states, then test a warm-relative/guard-aware clipped advantage objective"
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "qualification": result["qualification"],
        "latest_mean_gain": {
            name: run["latest"]["mean_gain"] for name, run in runs.items()
        },
        "latest_p05_gain": {
            name: run["latest"]["p05_gain"] for name, run in runs.items()
        },
        "latest_worst_gain": {
            name: run["latest"]["worst_gain"] for name, run in runs.items()
        },
        "selected_mean_gain": {
            name: run["selected"]["mean_gain"] for name, run in runs.items()
        },
        "selected_guard_mean_cost": {
            name: run["selected"]["guard_mean_cost"] for name, run in runs.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
