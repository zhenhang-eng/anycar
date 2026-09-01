#!/usr/bin/env python3
"""Compare baseline OAC-2 with two fixed tail-CVaR Actor objectives."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from generate_dbm_proposal_teacher import sha256_file


DEFAULT_BASELINE = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_fold1_200round_20260824_v1"
)
DEFAULT_W10 = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_tailcvar10_200round_20260824_v1"
)
DEFAULT_W100 = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_tailcvar_200round_20260824_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_tailcvar_ab_20260824_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--weight10", type=Path, default=DEFAULT_W10)
    parser.add_argument("--weight100", type=Path, default=DEFAULT_W100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def mean(rows: list[dict[str, Any]], fn) -> float:
    return float(np.mean([fn(row) for row in rows]))


def summarize(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = json.loads((root / "contract.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    validator = json.loads((root / "validator_report.json").read_text())
    if validator["qualification"] != "OAC2_CONTINUOUS_ACTOR_VALIDATION_PASS":
        raise AssertionError(f"source validation failed: {root}")
    if summary["formal_validation_loaded"] or summary["test_loaded"]:
        raise AssertionError(f"sealed split violation: {root}")
    rows = summary["records"]
    if len(rows) != 3 or sorted(row["seed"] for row in rows) != [0, 1, 2]:
        raise AssertionError(f"expected three seeds: {root}")
    item = {
        "tail_regression_weight": float(contract["arguments"].get("tail_regression_weight", 0.0)),
        "tail_cvar_fraction": float(contract["arguments"].get("tail_cvar_fraction", 0.10)),
        "selected_rounds": [int(row["selected_round"]) for row in rows],
        "cost_mean": mean(rows, lambda r: r["selected_metrics"]["cost"]["mean"]),
        "cost_median_average": mean(rows, lambda r: r["selected_metrics"]["cost"]["median"]),
        "gain_mean": mean(rows, lambda r: r["selected_metrics"]["gain_vs_initial"]["mean"]),
        "gain_median_average": mean(rows, lambda r: r["selected_metrics"]["gain_vs_initial"]["median"]),
        "gain_p05_average": mean(rows, lambda r: r["selected_metrics"]["gain_vs_initial"]["p05"]),
        "gain_worst_average": mean(rows, lambda r: r["selected_metrics"]["gain_vs_initial"]["minimum"]),
        "regression_fraction_average": mean(rows, lambda r: r["selected_metrics"]["regression_fraction"]),
        "headroom_recovery_average": mean(rows, lambda r: r["selected_metrics"]["headroom_recovery_vs_bank_best"]),
        "guard_cost_mean": mean(rows, lambda r: r["selected_metrics"]["two_center_guard"]["cost"]["mean"]),
        "critic_pair_average": mean(rows, lambda r: r["critic_metrics"]["actor_visited_material_pair_accuracy"]),
        "accepted_evaluations": [
            int(sum(e["accepted"] for e in row["evaluations"][1:])) for row in rows
        ],
        "evaluation_counts": [len(row["evaluations"]) - 1 for row in rows],
        "per_seed": [
            {
                "seed": int(row["seed"]),
                "gain_mean": float(row["selected_metrics"]["gain_vs_initial"]["mean"]),
                "gain_p05": float(row["selected_metrics"]["gain_vs_initial"]["p05"]),
                "gain_worst": float(row["selected_metrics"]["gain_vs_initial"]["minimum"]),
                "regression_fraction": float(row["selected_metrics"]["regression_fraction"]),
            }
            for row in rows
        ],
    }
    manifest = {
        "path": str(root.resolve()),
        "contract_sha256": sha256_file(root / "contract.json"),
        "summary_sha256": sha256_file(root / "summary.json"),
        "validator_sha256": sha256_file(root / "validator_report.json"),
    }
    return item, manifest


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    roots = {
        "baseline": args.baseline,
        "tail_cvar_weight10": args.weight10,
        "tail_cvar_weight100": args.weight100,
    }
    arms = {}
    manifests = {}
    invariant = None
    for name, root in roots.items():
        item, manifest = summarize(root)
        contract = json.loads((root / "contract.json").read_text())
        current = {
            key: contract[key]
            for key in (
                "outer_fold", "fit_episodes", "internal_selection_episodes",
                "candidate_bank_sha256", "parent_summary_sha256",
            )
        }
        current.update({
            key: contract["arguments"][key]
            for key in (
                "rounds", "actor_learning_rate", "critic_updates_per_round",
                "actor_updates_per_round", "max_step_sigma_rms",
            )
        })
        if invariant is None:
            invariant = current
        elif current != invariant:
            raise AssertionError(f"non-tail contract mismatch: {name}")
        arms[name] = item
        manifests[name] = manifest

    base = arms["baseline"]
    paired = {}
    for name in ("tail_cvar_weight10", "tail_cvar_weight100"):
        arm = arms[name]
        paired[name] = {
            "mean_gain_retention": float(arm["gain_mean"] / base["gain_mean"]),
            "gain_p05_improvement": float(arm["gain_p05_average"] - base["gain_p05_average"]),
            "gain_worst_improvement": float(arm["gain_worst_average"] - base["gain_worst_average"]),
            "regression_fraction_change": float(arm["regression_fraction_average"] - base["regression_fraction_average"]),
            "guard_cost_change": float(arm["guard_cost_mean"] - base["guard_cost_mean"]),
        }
    checks = {
        "all_selected_round200": all(
            arm["selected_rounds"] == [200, 200, 200] for arm in arms.values()
        ),
        "all_tail_evaluations_accepted": all(
            arms[name]["accepted_evaluations"] == arms[name]["evaluation_counts"]
            for name in ("tail_cvar_weight10", "tail_cvar_weight100")
        ),
        "tail_p05_improves": all(
            arms[name]["gain_p05_average"] > base["gain_p05_average"]
            for name in ("tail_cvar_weight10", "tail_cvar_weight100")
        ),
        "tail_regression_fraction_decreases": all(
            arms[name]["regression_fraction_average"] < base["regression_fraction_average"]
            for name in ("tail_cvar_weight10", "tail_cvar_weight100")
        ),
        "both_tail_weights_retain_less_than_20pct_mean_gain": all(
            paired[name]["mean_gain_retention"] < 0.20
            for name in ("tail_cvar_weight10", "tail_cvar_weight100")
        ),
        "critic_pair_stable": all(
            abs(arms[name]["critic_pair_average"] - base["critic_pair_average"]) < 0.01
            for name in ("tail_cvar_weight10", "tail_cvar_weight100")
        ),
        "formal_validation_sealed": True,
        "test_sealed": True,
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "OAC2_TAIL_CVAR_EFFECTIVE_BUT_OVERREGULARIZED_NOT_SELECTED",
        "invariant_contract": invariant,
        "manifest": manifests,
        "arms": arms,
        "paired_vs_baseline": paired,
        "checks": checks,
        "decision": (
            "A top-10% positive predicted-regression CVaR is an effective tail "
            "mechanism, but weights 10 and 100 both behave like a hard local "
            "constraint and retain less than 20% of baseline mean gain.  Do not "
            "select either Actor.  Keep the checkpoint tail floors; redesign the "
            "Actor penalty as a softer budgeted/dual constraint before another "
            "long run."
        ),
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
