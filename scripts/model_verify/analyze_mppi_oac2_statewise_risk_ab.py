#!/usr/bin/env python3
"""Compare adaptive-tail OAC against continuous state-wise risk shrink."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from analyze_mppi_oac2_adaptive_tail import role_metrics
from generate_dbm_proposal_teacher import sha256_file


DEFAULT_BASELINE = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_box3_adaptive_tail_200round_20260825_v1"
)
DEFAULT_STATEWISE = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_box3_statewise_risk_200round_20260825_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_statewise_risk_ab_20260825_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--statewise", type=Path, default=DEFAULT_STATEWISE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    summaries = {
        name: json.loads((path / "summary.json").read_text())
        for name, path in (("adaptive", args.baseline), ("statewise", args.statewise))
    }
    contracts = {
        name: json.loads((path / "contract.json").read_text())
        for name, path in (("adaptive", args.baseline), ("statewise", args.statewise))
    }
    validators = {
        name: json.loads((path / "validator_report.json").read_text())
        for name, path in (("adaptive", args.baseline), ("statewise", args.statewise))
    }
    ignored = {
        "output_dir", "statewise_risk_shrink_weight",
        "statewise_risk_temperature_log", "statewise_risk_shrink_form",
    }
    paired = {
        key: contracts["statewise"]["arguments"].get(key) == value
        for key, value in contracts["adaptive"]["arguments"].items()
        if key not in ignored
    }
    if not all(paired.values()):
        raise AssertionError("non-statewise arguments differ")
    arms = {}
    for name in ("adaptive", "statewise"):
        records = sorted(summaries[name]["records"], key=lambda row: int(row["seed"]))
        arms[name] = {
            "selected_rounds": [int(row["selected_round"]) for row in records],
            "selected": role_metrics(records, "selected_metrics"),
            "latest": role_metrics(records, "latest_metrics"),
        }
    base = arms["adaptive"]["latest"]
    statewise = arms["statewise"]["latest"]
    retention = statewise["gain_mean"] / base["gain_mean"]
    comparison = {
        "latest_mean_gain_retention": retention,
        "latest_median_gain_change": statewise["gain_median"] - base["gain_median"],
        "latest_p05_improvement": statewise["gain_p05"] - base["gain_p05"],
        "latest_worst_improvement": statewise["gain_worst"] - base["gain_worst"],
        "latest_regression_fraction_change": (
            statewise["regression_fraction"] - base["regression_fraction"]
        ),
        "latest_guard_cost_change": statewise["guard_cost_mean"] - base["guard_cost_mean"],
        "speed_2_4_p05_improvement": (
            statewise["speed_2_4_gain_p05"] - base["speed_2_4_gain_p05"]
        ),
        "speed_2_8_p05_improvement": (
            statewise["speed_2_8_gain_p05"] - base["speed_2_8_gain_p05"]
        ),
    }
    gates = {
        "mean_gain_retention_ge_0_70": retention >= 0.70,
        "p05_improves": comparison["latest_p05_improvement"] > 0,
        "worst_improves": comparison["latest_worst_improvement"] > 0,
        "regression_fraction_not_worse": comparison["latest_regression_fraction_change"] <= 0,
        "both_high_speed_p05_improve": (
            comparison["speed_2_4_p05_improvement"] > 0
            and comparison["speed_2_8_p05_improvement"] > 0
        ),
    }
    qualification = (
        "STATEWISE_RISK_SHRINK_PARETO_PASS"
        if all(gates.values())
        else "STATEWISE_RISK_SHRINK_FAIL_NO_FURTHER_WEIGHT_SWEEP"
    )
    report = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "manifest": {
            "adaptive": str(args.baseline.resolve()),
            "statewise": str(args.statewise.resolve()),
            "adaptive_summary_sha256": sha256_file(args.baseline / "summary.json"),
            "statewise_summary_sha256": sha256_file(args.statewise / "summary.json"),
            "adaptive_validator_pass": bool(validators["adaptive"]["passed"]),
            "statewise_validator_pass": bool(validators["statewise"]["passed"]),
            "paired_arguments": paired,
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
        "arms": arms,
        "comparison": comparison,
        "gates": gates,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "qualification": qualification,
        "comparison": comparison,
        "gates": gates,
    }, indent=2))


if __name__ == "__main__":
    main()
