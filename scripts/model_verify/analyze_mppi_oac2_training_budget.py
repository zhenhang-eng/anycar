#!/usr/bin/env python3
"""Compare the frozen 20/100/200-round OAC-2 training budgets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from generate_dbm_proposal_teacher import sha256_file


DEFAULT_RUNS = (
    Path("outputs/mppi_proposal/online_absolute_sac_oac2_fold1_20260824_v1"),
    Path("outputs/mppi_proposal/online_absolute_sac_oac2_fold1_100round_20260824_v1"),
    Path("outputs/mppi_proposal/online_absolute_sac_oac2_fold1_200round_20260824_v1"),
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_budget_20260824_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=3, type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def average(records: list[dict[str, Any]], fn: Callable[[dict[str, Any]], float]) -> float:
    return float(np.mean([fn(row) for row in records]))


def aggregate(run: Path, contract: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    records = summary["records"]
    if len(records) != 3 or sorted(row["seed"] for row in records) != [0, 1, 2]:
        raise AssertionError(f"{run}: expected three seeds")
    selected = lambda row: row["selected_metrics"]
    return {
        "rounds": int(contract["arguments"]["rounds"]),
        "selected_rounds": [int(row["selected_round"]) for row in records],
        "initial_actor_cost_mean": average(records, lambda row: row["initial_metrics"]["cost"]["mean"]),
        "selected_actor_cost_mean": average(records, lambda row: selected(row)["cost"]["mean"]),
        "selected_actor_cost_median_average": average(records, lambda row: selected(row)["cost"]["median"]),
        "gain_vs_initial_mean": average(records, lambda row: selected(row)["gain_vs_initial"]["mean"]),
        "gain_vs_initial_median_average": average(records, lambda row: selected(row)["gain_vs_initial"]["median"]),
        "gain_vs_initial_p05_average": average(records, lambda row: selected(row)["gain_vs_initial"]["p05"]),
        "gain_vs_initial_worst_average": average(records, lambda row: selected(row)["gain_vs_initial"]["minimum"]),
        "regression_fraction_average": average(records, lambda row: selected(row)["regression_fraction"]),
        "teacher_headroom_recovery_average": average(records, lambda row: selected(row)["headroom_recovery_vs_bank_best"]),
        "guard_cost_mean": average(records, lambda row: selected(row)["two_center_guard"]["cost"]["mean"]),
        "guard_gain_vs_warm_mean": average(records, lambda row: selected(row)["two_center_guard"]["gain_vs_warm"]["mean"]),
        "actor_beats_warm_fraction_average": average(
            records,
            lambda row: 1.0 - selected(row)["two_center_guard"]["warm_selected_fraction"],
        ),
        "critic_actor_visited_pair_accuracy_average": average(
            records, lambda row: row["critic_metrics"]["actor_visited_material_pair_accuracy"]
        ),
        "speed_2_4_gain_average": average(records, lambda row: selected(row)["by_speed"]["2.4"]["mean_gain"]),
        "speed_2_8_gain_average": average(records, lambda row: selected(row)["by_speed"]["2.8"]["mean_gain"]),
        "per_seed": [
            {
                "seed": int(row["seed"]),
                "gain_mean": float(selected(row)["gain_vs_initial"]["mean"]),
                "headroom_recovery": float(selected(row)["headroom_recovery_vs_bank_best"]),
                "p05": float(selected(row)["gain_vs_initial"]["p05"]),
                "worst": float(selected(row)["gain_vs_initial"]["minimum"]),
                "critic_pair": float(row["critic_metrics"]["actor_visited_material_pair_accuracy"]),
            }
            for row in records
        ],
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    manifests = []
    aggregates = []
    reference_contract = None
    bank = None
    selection_mask = None
    for run in args.runs:
        contract_path = run / "contract.json"
        summary_path = run / "summary.json"
        validator_path = run / "validator_report.json"
        contract = json.loads(contract_path.read_text())
        summary = json.loads(summary_path.read_text())
        validator = json.loads(validator_path.read_text())
        if validator["qualification"] != "OAC2_CONTINUOUS_ACTOR_VALIDATION_PASS":
            raise AssertionError(f"{run}: source validator did not pass")
        if summary["formal_validation_loaded"] or summary["test_loaded"]:
            raise AssertionError(f"{run}: sealed split violation")
        invariant = {
            "outer_fold": contract["outer_fold"],
            "fit_episodes": contract["fit_episodes"],
            "internal_selection_episodes": contract["internal_selection_episodes"],
            "candidate_bank_sha256": contract["candidate_bank_sha256"],
            "parent_summary_sha256": contract["parent_summary_sha256"],
            "actor_learning_rate": contract["arguments"]["actor_learning_rate"],
            "critic_updates_per_round": contract["arguments"]["critic_updates_per_round"],
            "actor_updates_per_round": contract["arguments"]["actor_updates_per_round"],
            "max_step_sigma_rms": contract["arguments"]["max_step_sigma_rms"],
        }
        if reference_contract is None:
            reference_contract = invariant
        elif invariant != reference_contract:
            raise AssertionError(f"{run}: non-budget contract mismatch")
        if bank is None:
            bank_path = Path(contract["arguments"]["bank_root"]) / "candidate_bank.npz"
            with np.load(bank_path, allow_pickle=False) as loaded:
                bank = {key: np.asarray(loaded[key]) for key in loaded.files}
            selection_mask = np.isin(
                bank["episode"], contract["internal_selection_episodes"]
            )
        aggregates.append(aggregate(run, contract, summary))
        manifests.append({
            "run": str(run.resolve()),
            "rounds": int(contract["arguments"]["rounds"]),
            "summary_sha256": sha256_file(summary_path),
            "contract_sha256": sha256_file(contract_path),
            "validator_sha256": sha256_file(validator_path),
        })

    assert bank is not None and selection_mask is not None
    costs = bank["costs"][selection_mask].astype(np.float64)
    warm = costs[:, 0]
    teacher = costs.min(axis=1)
    warm_teacher = {
        "state_count": int(selection_mask.sum()),
        "warm_cost_mean": float(warm.mean()),
        "warm_cost_median": float(np.median(warm)),
        "bank_best_teacher_cost_mean": float(teacher.mean()),
        "bank_best_teacher_cost_median": float(np.median(teacher)),
    }
    last = aggregates[-1]
    warm_teacher["round200_guard_warm_to_teacher_recovery"] = float(
        (warm.mean() - last["guard_cost_mean"])
        / max(warm.mean() - teacher.mean(), 1e-12)
    )

    checks = {
        "mean_gain_monotonic": all(
            b["gain_vs_initial_mean"] > a["gain_vs_initial_mean"]
            for a, b in zip(aggregates, aggregates[1:])
        ),
        "teacher_recovery_monotonic": all(
            b["teacher_headroom_recovery_average"]
            > a["teacher_headroom_recovery_average"]
            for a, b in zip(aggregates, aggregates[1:])
        ),
        "critic_pair_stays_above_0_90": all(
            row["critic_actor_visited_pair_accuracy_average"] >= 0.90
            for row in aggregates
        ),
        "p05_worsens_with_budget": all(
            b["gain_vs_initial_p05_average"] < a["gain_vs_initial_p05_average"]
            for a, b in zip(aggregates, aggregates[1:])
        ),
        "worst_worsens_with_budget": all(
            b["gain_vs_initial_worst_average"] < a["gain_vs_initial_worst_average"]
            for a, b in zip(aggregates, aggregates[1:])
        ),
        "round200_actor_mean_still_worse_than_warm": bool(
            last["selected_actor_cost_mean"] > warm.mean()
        ),
        "round200_actor_median_better_than_warm": bool(
            last["selected_actor_cost_median_average"] < np.median(warm)
        ),
        "formal_validation_sealed": True,
        "test_sealed": True,
    }
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "OAC2_BUDGET_IMPROVES_CENTER_BUT_DIRECT_TAIL_DIVERGES",
        "manifest": manifests,
        "invariant_contract": reference_contract,
        "warm_teacher_reference": warm_teacher,
        "budgets": aggregates,
        "checks": checks,
        "decision": (
            "Training budget was an immediate bottleneck: 20 rounds was not an "
            "Actor limit and Critic ranking stayed stable through 200 rounds. "
            "However, mean-only checkpoint advancement trades progressively worse "
            "direct P05/worst for center improvement. Do not raise Actor LR or "
            "authorize Actor-only use before adding a tail-aware Actor objective and "
            "selection gate."
        ),
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    if not all(checks.values()):
        raise AssertionError(f"unexpected budget-curve result: {checks}")
    (args.output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({
        "qualification": result["qualification"],
        "warm_teacher_reference": warm_teacher,
        "budgets": aggregates,
        "checks": checks,
    }, indent=2))


if __name__ == "__main__":
    main()
