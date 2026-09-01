#!/usr/bin/env python3
"""Summarize the matched OAC-2 Actor learning-rate boundary scan."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


DEFAULT_RUNS = {
    "1e-5": Path(
        "outputs/mppi_proposal/"
        "online_absolute_sac_oac2_rawcost_gamma1_lrdecay_cuda_200round_20260827_v1"
    ),
    "2e-5": Path(
        "outputs/mppi_proposal/"
        "online_absolute_sac_oac2_rawcost_gamma1_lrscan_2e5_90round_20260827_v1"
    ),
    "4e-5": Path(
        "outputs/mppi_proposal/"
        "online_absolute_sac_oac2_rawcost_gamma1_lrscan_4e5_90round_20260827_v1"
    ),
    "8e-5": Path(
        "outputs/mppi_proposal/"
        "online_absolute_sac_oac2_rawcost_gamma1_lrscan_8e5_90round_20260827_v1"
    ),
    "1.6e-4": Path(
        "outputs/mppi_proposal/"
        "online_absolute_sac_oac2_rawcost_gamma1_lrscan_16e5_90round_20260827_v1"
    ),
    "3.2e-4": Path(
        "outputs/mppi_proposal/"
        "online_absolute_sac_oac2_rawcost_gamma1_lrscan_32e5_90round_20260827_v1"
    ),
}
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/oac2_lr_boundary_scan_20260827_v1")
PRIMARY_ROUND = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    for label, path in DEFAULT_RUNS.items():
        parser.add_argument(
            f"--run-{label.replace('.', 'p')}", type=Path, default=path,
        )
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
}


def summarize(rows: list[dict]) -> dict:
    per_seed = [
        {name: nested(row, *keys) for name, keys in METRICS.items()}
        for row in rows
    ]
    return {
        **{
            name: mean(seed[name] for seed in per_seed)
            for name in METRICS
        },
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
    primary_rows = [evaluation(record, PRIMARY_ROUND) for record in records]
    actor_rows = [
        row["actor"] for record in records for row in record["rounds"]
        if 1 <= int(row["round"]) <= 20
    ]
    curve_rounds = sorted({
        int(row["round"]) for record in records for row in record["evaluations"]
        if int(row["round"]) <= PRIMARY_ROUND
    })
    curve = {
        str(round_index): summarize([
            evaluation(record, round_index)["metrics"] for record in records
        ])
        for round_index in curve_rounds
    }
    return {
        "label": label,
        "path": str(path.resolve()),
        "contract_sha256": sha256(contract_path),
        "summary_sha256": sha256(summary_path),
        "validator_sha256": sha256(validator_path),
        "initial_learning_rate": float(
            contract["arguments"]["actor_learning_rate_initial"]
        ),
        "primary_round": PRIMARY_ROUND,
        "primary": summarize([row["metrics"] for row in primary_rows]),
        "curve": curve,
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
    run_paths = {
        label: getattr(args, f"run_{label.replace('.', 'p').replace('-', '_')}")
        for label in DEFAULT_RUNS
    }
    runs = {label: load_run(label, path) for label, path in run_paths.items()}
    ordered = list(DEFAULT_RUNS)
    increments = {}
    for previous, current in zip(ordered, ordered[1:]):
        increments[f"{previous}_to_{current}"] = {
            "headroom_recovery_pp": 100.0 * (
                runs[current]["primary"]["headroom_recovery"]
                - runs[previous]["primary"]["headroom_recovery"]
            ),
            "mean_cost": (
                runs[current]["primary"]["mean_cost"]
                - runs[previous]["primary"]["mean_cost"]
            ),
            "median_cost": (
                runs[current]["primary"]["median_cost"]
                - runs[previous]["primary"]["median_cost"]
            ),
            "p05_gain": (
                runs[current]["primary"]["p05_gain"]
                - runs[previous]["primary"]["p05_gain"]
            ),
            "speed_2_8_p05_gain": (
                runs[current]["primary"]["speed_2_8_p05_gain"]
                - runs[previous]["primary"]["speed_2_8_p05_gain"]
            ),
        }
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "LR_BOUNDARY_TRUST_SATURATION_MEAN_PASS_TAIL_FAIL",
        "scope": (
            "fold-1 train-side internal-selection; gamma=1 box3 adaptive-tail; "
            "round-80 matched schedule; three seeds; formal validation/test sealed"
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
        },
        "runs": runs,
        "paired_increments": increments,
        "decision": {
            "mean_boundary": (
                "raising the initial LR monotonically improves round-80 mean/median and "
                "headroom recovery through 3.2e-4, but gains diminish once the 0.02-sigma "
                "action-space trust projection is active on most early updates"
            ),
            "operating_region": (
                "8e-5 is the largest mostly unsaturated arm; 1.6e-4 is the practical "
                "trust-limited boundary; 3.2e-4 is a saturation confirmation, not a "
                "recommended optimizer setting"
            ),
            "tail": (
                "tail severity, especially 2.4/2.8 m/s P05, worsens independently of "
                "the monotonic mean improvement; no checkpoint is deployable"
            ),
            "next": (
                "stop increasing nominal LR under the fixed 0.02-sigma trust contract; "
                "use the 1.6e-4/trust-limited trajectory as a mean-capacity probe and "
                "separate per-state safety projection from the optimization-scale study"
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
        "primary_round": PRIMARY_ROUND,
        "headroom_recovery": {
            label: run["primary"]["headroom_recovery"]
            for label, run in runs.items()
        },
        "early_projection_fraction": {
            label: run["early_action_step"]["projection_fraction"]
            for label, run in runs.items()
        },
        "output": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()
