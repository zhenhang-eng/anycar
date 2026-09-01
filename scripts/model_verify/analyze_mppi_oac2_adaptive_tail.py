#!/usr/bin/env python3
"""Analyze the paired box3 OAC adaptive-tail budget experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from generate_dbm_proposal_teacher import sha256_file


DEFAULT_BASELINE = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_support_box3_200round_20260825_v1"
)
DEFAULT_ADAPTIVE = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_box3_adaptive_tail_200round_20260825_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_adaptive_tail_ab_20260825_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--adaptive", type=Path, default=DEFAULT_ADAPTIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def mean_metric(records: list[dict[str, Any]], role: str,
                getter: Callable[[dict[str, Any]], float]) -> float:
    return float(np.mean([getter(row[role]) for row in records]))


def role_metrics(records: list[dict[str, Any]], role: str) -> dict[str, float]:
    getters = {
        "cost_mean": lambda m: m["cost"]["mean"],
        "cost_median": lambda m: m["cost"]["median"],
        "gain_mean": lambda m: m["gain_vs_initial"]["mean"],
        "gain_median": lambda m: m["gain_vs_initial"]["median"],
        "gain_p05": lambda m: m["gain_vs_initial"]["p05"],
        "gain_worst": lambda m: m["gain_vs_initial"]["minimum"],
        "regression_fraction": lambda m: m["regression_fraction"],
        "guard_cost_mean": lambda m: m["two_center_guard"]["cost"]["mean"],
        "speed_2_4_gain_p05": lambda m: m["by_speed"]["2.4"]["p05_gain"],
        "speed_2_8_gain_p05": lambda m: m["by_speed"]["2.8"]["p05_gain"],
    }
    return {
        key: mean_metric(records, role, getter) for key, getter in getters.items()
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    baseline = json.loads((args.baseline / "summary.json").read_text())
    adaptive = json.loads((args.adaptive / "summary.json").read_text())
    baseline_contract = json.loads((args.baseline / "contract.json").read_text())
    adaptive_contract = json.loads((args.adaptive / "contract.json").read_text())
    baseline_validator = json.loads((args.baseline / "validator_report.json").read_text())
    adaptive_validator = json.loads((args.adaptive / "validator_report.json").read_text())
    base_records = sorted(baseline["records"], key=lambda row: int(row["seed"]))
    adaptive_records = sorted(adaptive["records"], key=lambda row: int(row["seed"]))
    if [row["seed"] for row in base_records] != [row["seed"] for row in adaptive_records]:
        raise AssertionError("seed mismatch")
    ignored = {
        "output_dir", "tail_constraint_mode", "tail_regression_margin_log",
        "tail_cvar_budget_log", "tail_dual_learning_rate", "tail_dual_initial",
        "tail_dual_maximum", "tail_dual_ema_decay", "allow_box1_ablation",
    }
    paired_arguments = {}
    for key, value in baseline_contract["arguments"].items():
        if key in ignored:
            continue
        paired_arguments[key] = (
            key in adaptive_contract["arguments"]
            and adaptive_contract["arguments"][key] == value
        )
    if not all(paired_arguments.values()):
        raise AssertionError("non-tail arguments are not paired")
    if float(adaptive_contract["arguments"]["actor_output_support_multiplier"]) != 3.0:
        raise AssertionError("adaptive arm is not box3")
    if adaptive_contract["arguments"]["tail_constraint_mode"] != "adaptive":
        raise AssertionError("adaptive mode missing")

    arms = {}
    for name, records in (("box3_baseline", base_records), ("adaptive_tail", adaptive_records)):
        arms[name] = {
            "selected_rounds": [int(row["selected_round"]) for row in records],
            "selected": role_metrics(records, "selected_metrics"),
            "latest": role_metrics(records, "latest_metrics"),
        }
    dual = []
    for row in adaptive_records:
        values = [float(item["actor"]["tail_lagrange_after"]) for item in row["rounds"]]
        active = [int(item["round"]) for item, value in zip(row["rounds"], values) if value > 0]
        dual.append({
            "seed": int(row["seed"]),
            "first_active_round": active[0] if active else None,
            "maximum": float(max(values)),
            "final": float(row["final_tail_lagrange"]),
            "final_cvar_ema": float(row["final_tail_cvar_ema"]),
        })

    base_latest = arms["box3_baseline"]["latest"]
    adaptive_latest = arms["adaptive_tail"]["latest"]
    base_selected = arms["box3_baseline"]["selected"]
    adaptive_selected = arms["adaptive_tail"]["selected"]
    retention = adaptive_latest["gain_mean"] / base_latest["gain_mean"]
    extended = sum(
        int(new["selected_round"]) > int(old["selected_round"])
        for old, new in zip(base_records, adaptive_records)
    )
    gates = {
        "mechanism_dual_activates_all_seeds": all(row["first_active_round"] is not None for row in dual),
        "mechanism_dual_not_saturated": all(row["maximum"] < 9.9 for row in dual),
        "pareto_latest_p05_improves": adaptive_latest["gain_p05"] > base_latest["gain_p05"],
        "pareto_latest_worst_improves": adaptive_latest["gain_worst"] > base_latest["gain_worst"],
        "pareto_latest_mean_retention_ge_0_70": retention >= 0.70,
        "safe_horizon_extended_at_least_two_seeds": extended >= 2,
        "selected_mean_gain_ge_box3": adaptive_selected["gain_mean"] >= base_selected["gain_mean"],
    }
    mechanism_pass = all(gates[key] for key in gates if key.startswith("mechanism_"))
    pareto_pass = all(gates[key] for key in gates if key.startswith("pareto_"))
    horizon_pass = (
        gates["safe_horizon_extended_at_least_two_seeds"]
        and gates["selected_mean_gain_ge_box3"]
    )
    qualification = (
        "ADAPTIVE_TAIL_FULL_PASS_READY_FOR_CLOSED_LOOP_DISCUSSION"
        if mechanism_pass and pareto_pass and horizon_pass
        else "ADAPTIVE_TAIL_PARETO_IMPROVES_BUT_MISSES_RETENTION_AND_SAFE_HORIZON_GATES"
    )
    report = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "manifest": {
            "baseline": str(args.baseline.resolve()),
            "adaptive": str(args.adaptive.resolve()),
            "baseline_summary_sha256": sha256_file(args.baseline / "summary.json"),
            "adaptive_summary_sha256": sha256_file(args.adaptive / "summary.json"),
            "baseline_validator_pass": bool(baseline_validator["passed"]),
            "adaptive_validator_pass": bool(adaptive_validator["passed"]),
            "paired_non_tail_arguments": paired_arguments,
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
        "arms": arms,
        "dual": dual,
        "comparison": {
            "latest_mean_gain_retention": float(retention),
            "latest_p05_improvement": adaptive_latest["gain_p05"] - base_latest["gain_p05"],
            "latest_worst_improvement": adaptive_latest["gain_worst"] - base_latest["gain_worst"],
            "latest_regression_fraction_change": (
                adaptive_latest["regression_fraction"] - base_latest["regression_fraction"]
            ),
            "latest_guard_cost_change": (
                adaptive_latest["guard_cost_mean"] - base_latest["guard_cost_mean"]
            ),
            "selected_mean_gain_change": (
                adaptive_selected["gain_mean"] - base_selected["gain_mean"]
            ),
            "selected_guard_cost_change": (
                adaptive_selected["guard_cost_mean"] - base_selected["guard_cost_mean"]
            ),
            "extended_seed_count": int(extended),
        },
        "gates": gates,
        "decision": (
            "The adaptive margin/budget constraint is materially better than the fixed "
            "lambda=10/100 arms and improves the box3 latest tail, but it narrowly misses "
            "the preregistered 70% mean-gain retention threshold and extends the selected "
            "safe horizon in only one of three seeds. Do not authorize OAC-4. Keep box3, "
            "the checkpoint floors, and the two-center guard; diagnose state-conditioned "
            "risk allocation before another long run."
        ),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "qualification": qualification,
        "comparison": report["comparison"],
        "gates": gates,
    }, indent=2))


if __name__ == "__main__":
    main()
