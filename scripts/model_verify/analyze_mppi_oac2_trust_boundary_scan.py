#!/usr/bin/env python3
"""Summarize the matched OAC-2 global action-space trust-boundary scan."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


DEFAULT_RUNS = {
    "0.02": Path(
        "outputs/mppi_proposal/"
        "online_absolute_sac_oac2_rawcost_gamma1_lrscan_32e5_90round_20260827_v1"
    ),
    "0.04": Path(
        "outputs/mppi_proposal/"
        "online_absolute_sac_oac2_rawcost_gamma1_lr32e5_trust004_90round_20260827_v1"
    ),
    "0.06": Path(
        "outputs/mppi_proposal/"
        "online_absolute_sac_oac2_rawcost_gamma1_lr32e5_trust006_90round_20260827_v1"
    ),
    "0.08": Path(
        "outputs/mppi_proposal/"
        "online_absolute_sac_oac2_rawcost_gamma1_lr32e5_trust008_90round_20260827_v1"
    ),
}
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/oac2_trust_boundary_scan_20260827_v1")
PRIMARY_ROUND = 80

METRICS = {
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
    "guard_warm_selected_fraction": (
        "two_center_guard", "warm_selected_fraction",
    ),
    "action_saturation_fraction": ("action_saturation_fraction",),
    "state_any_saturation_fraction": ("state_any_saturation_fraction",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    for label, path in DEFAULT_RUNS.items():
        parser.add_argument(f"--run-{label.replace('.', 'p')}", type=Path, default=path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested(data: dict, *keys: str) -> float:
    value = data
    for key in keys:
        value = value[key]
    return float(value)


def summarize(rows: list[dict]) -> dict:
    per_seed = [
        {name: nested(row, *keys) for name, keys in METRICS.items()}
        for row in rows
    ]
    return {
        **{name: mean(row[name] for row in per_seed) for name in METRICS},
        "per_seed": per_seed,
    }


def evaluation(record: dict, round_index: int) -> dict:
    return next(
        row for row in record["evaluations"]
        if int(row["round"]) == round_index
    )


def engineering_validation_pass(report: dict) -> bool:
    return (
        all(bool(report["checks"][f"seed_{seed}"]) for seed in range(3))
        and all(
            all(bool(value) for value in record["checks"].values())
            for record in report["records"]
        )
    )


def load_run(label: str, path: Path) -> dict:
    contract_path = path / "contract.json"
    summary_path = path / "summary.json"
    validator_path = path / "validator_report.json"
    contract = json.loads(contract_path.read_text())
    summary = json.loads(summary_path.read_text())
    validator = json.loads(validator_path.read_text())
    records = summary["records"]
    round_indices = list(range(10, PRIMARY_ROUND + 1, 10))
    curve = {
        str(round_index): summarize([
            evaluation(record, round_index)["metrics"] for record in records
        ])
        for round_index in round_indices
    }
    primary = curve[str(PRIMARY_ROUND)]
    actor_rows = [
        row["actor"] for record in records for row in record["rounds"]
        if 1 <= int(row["round"]) <= 20
    ]
    average_recovery_curve = {
        round_index: curve[str(round_index)]["headroom_recovery"]
        for round_index in round_indices
    }
    peak_round = max(average_recovery_curve, key=average_recovery_curve.get)
    return {
        "label": label,
        "path": str(path.resolve()),
        "contract_sha256": sha256(contract_path),
        "summary_sha256": sha256(summary_path),
        "validator_sha256": sha256(validator_path),
        "trust_sigma_rms": float(contract["arguments"]["max_step_sigma_rms"]),
        "initial_learning_rate": float(
            contract["arguments"]["actor_learning_rate_initial"]
        ),
        "primary_round": PRIMARY_ROUND,
        "primary": primary,
        "curve": curve,
        "peak_average_recovery": {
            "round": int(peak_round),
            "value": float(average_recovery_curve[peak_round]),
            "round80_minus_peak": float(
                primary["headroom_recovery"] - average_recovery_curve[peak_round]
            ),
        },
        "early_action_step": {
            "mean_sigma_rms": mean(float(row["step_sigma_rms"]) for row in actor_rows),
            "maximum_sigma_rms": max(float(row["step_sigma_rms"]) for row in actor_rows),
            "projection_fraction": mean(
                float(row["trust_projection"] < 1.0) for row in actor_rows
            ),
            "mean_projection_factor": mean(
                float(row["trust_projection"]) for row in actor_rows
            ),
            "mean_critic_pair_accuracy": mean(
                float(record["rounds"][index]["critic_pair_accuracy"])
                for record in records for index in range(20)
            ),
        },
        "selected_rounds": [int(record["selected_round"]) for record in records],
        "formal_validation_loaded": bool(summary["formal_validation_loaded"]),
        "test_loaded": bool(summary["test_loaded"]),
        "engineering_validation_pass": engineering_validation_pass(validator),
        "performance_gate_passed_seed_count": int(validator["passed_seed_count"]),
    }


def main() -> None:
    args = parse_args()
    paths = {
        label: getattr(args, f"run_{label.replace('.', 'p')}")
        for label in DEFAULT_RUNS
    }
    runs = {label: load_run(label, path) for label, path in paths.items()}
    baseline = runs["0.02"]["primary"]
    deltas = {
        label: {
            "mean_cost": run["primary"]["mean_cost"] - baseline["mean_cost"],
            "median_cost": run["primary"]["median_cost"] - baseline["median_cost"],
            "headroom_recovery_pp": 100.0 * (
                run["primary"]["headroom_recovery"]
                - baseline["headroom_recovery"]
            ),
            "regression_fraction_pp": 100.0 * (
                run["primary"]["regression_fraction"]
                - baseline["regression_fraction"]
            ),
            "p05_gain": run["primary"]["p05_gain"] - baseline["p05_gain"],
            "worst_gain": run["primary"]["worst_gain"] - baseline["worst_gain"],
            "speed_2_8_p05_gain": (
                run["primary"]["speed_2_8_p05_gain"]
                - baseline["speed_2_8_p05_gain"]
            ),
        }
        for label, run in runs.items() if label != "0.02"
    }
    best_mean = min(runs, key=lambda label: runs[label]["primary"]["mean_cost"])
    best_median = min(runs, key=lambda label: runs[label]["primary"]["median_cost"])
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "TRUST_004_TO_008_MEAN_PLATEAU_TAIL_SEPARATE_FAIL",
        "scope": (
            "fold-1 train-side internal-selection; gamma=1 box3 adaptive-tail; "
            "initial LR 3.2e-4; round-80 matched; three seeds; formal validation/test sealed"
        ),
        "checks": {
            "all_engineering_validators_pass": all(
                run["engineering_validation_pass"] for run in runs.values()
            ),
            "formal_validation_sealed": all(
                not run["formal_validation_loaded"] for run in runs.values()
            ),
            "test_sealed": all(not run["test_loaded"] for run in runs.values()),
            "all_selected_rounds_zero": all(
                run["selected_rounds"] == [0, 0, 0] for run in runs.values()
            ),
            "all_performance_gate_counts_zero": all(
                run["performance_gate_passed_seed_count"] == 0
                for run in runs.values()
            ),
            "initial_lr_fixed": len({
                run["initial_learning_rate"] for run in runs.values()
            }) == 1,
        },
        "runs": runs,
        "delta_vs_trust_002": deltas,
        "best_round80": {
            "mean_cost": best_mean,
            "median_cost": best_median,
        },
        "decision": {
            "mean": (
                "relaxing the global trust cap from 0.02 to 0.04 materially improves "
                "mean capacity, but 0.04/0.06/0.08 form a narrow plateau rather than "
                "a continuing monotonic gain curve"
            ),
            "boundary": (
                "0.04 sigma is the smallest cap reaching the plateau; 0.08 gives the "
                "best average mean by a small margin but adds path oscillation and no "
                "clear cross-seed advantage over 0.04"
            ),
            "critic": (
                "actor-visited Critic pair accuracy remains near 0.90-0.93, so the plateau "
                "is not explained by immediate Critic tracking collapse"
            ),
            "tail": (
                "direct tail severity remains a separate deployment blocker and no "
                "checkpoint passes the registered selection gate"
            ),
            "next": (
                "stop widening the global trust cap; use 0.04 as the efficient mean-capacity "
                "operating point and run the matched DBM-versus-Critic gradient-source A/B; "
                "handle safety with per-state projection and the two-center guard"
            ),
        },
    }
    if not all(result["checks"].values()):
        raise AssertionError(result["checks"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "analysis.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "qualification": result["qualification"],
        "round80": {
            label: {
                "mean_cost": run["primary"]["mean_cost"],
                "median_cost": run["primary"]["median_cost"],
                "headroom_recovery": run["primary"]["headroom_recovery"],
                "projection_fraction": run["early_action_step"]["projection_fraction"],
            }
            for label, run in runs.items()
        },
        "output": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()
