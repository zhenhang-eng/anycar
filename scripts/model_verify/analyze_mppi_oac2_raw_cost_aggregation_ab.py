#!/usr/bin/env python3
"""Summarize the paired OAC-2C tempered raw-cost Actor aggregation pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


DEFAULT_RUNS = {
    "gamma_0": Path(
        "outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma0_smoke_20260825_v1"
    ),
    "gamma_0_5": Path(
        "outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma05_smoke_20260825_v1"
    ),
    "gamma_1": Path(
        "outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_smoke_20260825_v1"
    ),
}
DEFAULT_BASELINE = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_box3_adaptive_tail_smoke_20260825_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/oac2_raw_cost_aggregation_ab_20260825_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma-0", type=Path, default=DEFAULT_RUNS["gamma_0"])
    parser.add_argument("--gamma-0-5", type=Path, default=DEFAULT_RUNS["gamma_0_5"])
    parser.add_argument("--gamma-1", type=Path, default=DEFAULT_RUNS["gamma_1"])
    parser.add_argument("--historical-baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric(row: dict, role: str, *path: str) -> float:
    value = row[role]
    for key in path:
        value = value[key]
    return float(value)


def summarize_run(path: Path) -> dict:
    contract = json.loads((path / "contract.json").read_text())
    validator = json.loads((path / "validator_report.json").read_text())
    rows = [
        json.loads((path / f"seed_{seed}" / "summary.json").read_text())
        for seed in range(3)
    ]
    result = {
        "path": str(path.resolve()),
        "contract_sha256": sha256(path / "contract.json"),
        "validator_sha256": sha256(path / "validator_report.json"),
        "validator_passed": bool(validator["passed"]),
        "gamma": float(contract["arguments"].get("actor_cost_weight_gamma", 0.0)),
        "weight_cap": float(
            contract["arguments"].get("actor_cost_weight_maximum", 2048.0)
        ),
        "selected_rounds": [int(row["selected_round"]) for row in rows],
        "roles": {},
    }
    for role in ("latest_metrics", "selected_metrics"):
        result["roles"][role] = {
            "headroom_recovery": mean(
                metric(row, role, "headroom_recovery_vs_bank_best") for row in rows
            ),
            "mean_gain": mean(metric(row, role, "gain_vs_initial", "mean") for row in rows),
            "median_gain": mean(
                metric(row, role, "gain_vs_initial", "median") for row in rows
            ),
            "p05_gain": mean(metric(row, role, "gain_vs_initial", "p05") for row in rows),
            "worst_gain": mean(
                metric(row, role, "gain_vs_initial", "minimum") for row in rows
            ),
            "regression_fraction": mean(
                metric(row, role, "regression_fraction") for row in rows
            ),
            "mean_cost": mean(metric(row, role, "cost", "mean") for row in rows),
            "median_cost": mean(metric(row, role, "cost", "median") for row in rows),
            "speed_2_4": {
                key: mean(metric(row, role, "by_speed", "2.4", key) for row in rows)
                for key in ("mean_gain", "median_gain", "p05_gain")
            },
            "speed_2_8": {
                key: mean(metric(row, role, "by_speed", "2.8", key) for row in rows)
                for key in ("mean_gain", "median_gain", "p05_gain")
            },
            "per_seed": [
                {
                    "seed": int(row["seed"]),
                    "mean_gain": metric(row, role, "gain_vs_initial", "mean"),
                    "median_gain": metric(row, role, "gain_vs_initial", "median"),
                    "p05_gain": metric(row, role, "gain_vs_initial", "p05"),
                    "worst_gain": metric(row, role, "gain_vs_initial", "minimum"),
                    "regression_fraction": metric(row, role, "regression_fraction"),
                    "speed_2_4_p05": metric(row, role, "by_speed", "2.4", "p05_gain"),
                    "speed_2_8_p05": metric(row, role, "by_speed", "2.8", "p05_gain"),
                }
                for row in rows
            ],
        }
    actor_keys = (
        "actor_cost_weight_p50",
        "actor_cost_weight_p90",
        "actor_cost_weight_max",
        "actor_cost_weight_ess_fraction",
        "actor_cost_weight_cap_fraction",
        "step_sigma_rms",
    )
    result["actor_update_diagnostics"] = {
        key: mean(
            mean(
                float(round_row["actor"].get(
                    key,
                    1.0 if key in {
                        "actor_cost_weight_p50", "actor_cost_weight_p90",
                        "actor_cost_weight_max", "actor_cost_weight_ess_fraction",
                    } else 0.0,
                ))
                for round_row in row["rounds"]
            )
            for row in rows
        )
        for key in actor_keys
    }
    return result


def paired_delta(candidate: dict, baseline: dict, role: str) -> dict:
    fields = (
        "headroom_recovery", "mean_gain", "median_gain", "p05_gain",
        "worst_gain", "regression_fraction", "mean_cost", "median_cost",
    )
    delta = {
        key: candidate["roles"][role][key] - baseline["roles"][role][key]
        for key in fields
    }
    delta["mean_gain_improved_seed_count"] = sum(
        candidate["roles"][role]["per_seed"][seed]["mean_gain"]
        > baseline["roles"][role]["per_seed"][seed]["mean_gain"]
        for seed in range(3)
    )
    for field in ("p05_gain", "worst_gain", "speed_2_4_p05", "speed_2_8_p05"):
        delta[f"{field}_improved_seed_count"] = sum(
            candidate["roles"][role]["per_seed"][seed][field]
            > baseline["roles"][role]["per_seed"][seed][field]
            for seed in range(3)
        )
    return delta


def main() -> None:
    args = parse_args()
    runs = {
        "gamma_0": summarize_run(args.gamma_0),
        "gamma_0_5": summarize_run(args.gamma_0_5),
        "gamma_1": summarize_run(args.gamma_1),
    }
    gamma0 = runs["gamma_0"]
    common_args = json.loads((args.gamma_0 / "contract.json").read_text())["arguments"]
    ignored = {"output_dir", "actor_cost_weight_gamma"}
    paired_contract = True
    for name, path in (("gamma_0_5", args.gamma_0_5), ("gamma_1", args.gamma_1)):
        candidate_args = json.loads((path / "contract.json").read_text())["arguments"]
        paired_contract &= all(
            candidate_args[key] == value
            for key, value in common_args.items()
            if key not in ignored
        )
        paired_contract &= runs[name]["validator_passed"]
    historical = summarize_run(args.historical_baseline)
    reproduction_deltas = []
    for role in ("latest_metrics", "selected_metrics"):
        for key in ("mean_gain", "median_gain", "p05_gain", "worst_gain"):
            reproduction_deltas.append(
                abs(gamma0["roles"][role][key] - historical["roles"][role][key])
            )
    gamma1_selected = paired_delta(runs["gamma_1"], gamma0, "selected_metrics")
    gamma1_latest = paired_delta(runs["gamma_1"], gamma0, "latest_metrics")
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            "RAW_COST_AGGREGATION_IMPROVES_MEAN_BUT_TAIL_MIXED_SHORT_PILOT"
        ),
        "scope": (
            "train-side fold-1 internal-selection, 20-round paired OAC-2C pilot; "
            "formal validation and test remain sealed"
        ),
        "contract_checks": {
            "paired_except_gamma_and_output": paired_contract,
            "all_run_validators_passed": all(run["validator_passed"] for run in runs.values()),
            "gamma_values": [runs[name]["gamma"] for name in runs],
            "weight_caps": [runs[name]["weight_cap"] for name in runs],
            "gamma0_operational_reproduction_max_metric_delta": max(reproduction_deltas),
            "gamma0_operational_reproduction_le_5e_4": max(reproduction_deltas) <= 5e-4,
        },
        "runs": runs,
        "paired_vs_gamma_0": {
            "gamma_0_5_latest": paired_delta(runs["gamma_0_5"], gamma0, "latest_metrics"),
            "gamma_0_5_selected": paired_delta(runs["gamma_0_5"], gamma0, "selected_metrics"),
            "gamma_1_latest": gamma1_latest,
            "gamma_1_selected": gamma1_selected,
        },
        "decision": {
            "mean_objective_signal": (
                "positive: gamma=1 improves latest and selected mean gain in all three seeds"
            ),
            "tail_signal": (
                "mixed: average P05 improves, but P05/worst do not improve in every seed"
            ),
            "weight_concentration": (
                "gamma=1 effective-sample fraction is about one quarter; retain clipping, "
                "DBM checkpoint floors, and the two-center guard"
            ),
            "authorization": (
                "gamma=1 is the leading Actor objective for the next isolated budget/scale "
                "pilot, but this 20-round internal result does not authorize deployment or "
                "formal validation/test"
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "qualification": result["qualification"],
        "gamma0_latest_mean_gain": gamma0["roles"]["latest_metrics"]["mean_gain"],
        "gamma05_latest_mean_gain": runs["gamma_0_5"]["roles"]["latest_metrics"]["mean_gain"],
        "gamma1_latest_mean_gain": runs["gamma_1"]["roles"]["latest_metrics"]["mean_gain"],
        "gamma1_latest_p05_delta": gamma1_latest["p05_gain"],
        "gamma1_latest_worst_delta": gamma1_latest["worst_gain"],
        "gamma1_mean_gain_improved_seed_count": gamma1_latest["mean_gain_improved_seed_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
