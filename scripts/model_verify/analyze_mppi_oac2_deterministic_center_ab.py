#!/usr/bin/env python3
"""Analyze stochastic-DBM versus deterministic-center DBM Actor objectives."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path("outputs/mppi_proposal")
STOCHASTIC = ROOT / "online_absolute_sac_oac2_matched_dbm_gradient_k16_90round_20260828_v1"
DETERMINISTIC = ROOT / "online_absolute_sac_oac2_deterministic_center_dbm_k16_90round_20260828_v1"
WARM = ROOT / "oac2_deterministic_center_warm_relative_20260828_v1"
CAPACITY = ROOT / "dbm_task_loss_support_full_ab_20260825_v1"
OUTPUT = ROOT / "oac2_deterministic_center_ab_20260828_v1"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "mean": float(mean(values)),
        "population_std": float(pstdev(values)),
        "per_seed": [float(value) for value in values],
    }


def normalized_arguments(contract: dict[str, Any]) -> dict[str, Any]:
    arguments = dict(contract["arguments"])
    arguments.setdefault("actor_objective_mode", "stochastic_sac")
    for key in ("output_dir", "actor_objective_mode"):
        arguments.pop(key, None)
    return arguments


def latest(summary: dict[str, Any]) -> dict[str, Any]:
    records = summary["records"]
    field = lambda fn: stats([float(fn(row["latest_metrics"])) for row in records])
    return {
        "headroom_recovery": field(lambda value: value["headroom_recovery_vs_bank_best"]),
        "cost_mean": field(lambda value: value["cost"]["mean"]),
        "cost_median": field(lambda value: value["cost"]["median"]),
        "gain_p05_vs_initial": field(lambda value: value["gain_vs_initial"]["p05"]),
        "gain_worst_vs_initial": field(lambda value: value["gain_vs_initial"]["minimum"]),
        "regression_fraction": field(lambda value: value["regression_fraction"]),
    }


def best_recovery(summary: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for record in summary["records"]:
        rows = [row for row in record["evaluations"] if int(row["round"]) > 0]
        best = max(rows, key=lambda row: row["metrics"]["headroom_recovery_vs_bank_best"])
        metrics = best["metrics"]
        result.append({
            "seed": int(record["seed"]),
            "round": int(best["round"]),
            "accepted": bool(best["accepted"]),
            "headroom_recovery": float(metrics["headroom_recovery_vs_bank_best"]),
            "gain_mean_vs_initial": float(metrics["gain_vs_initial"]["mean"]),
            "gain_p05_vs_initial": float(metrics["gain_vs_initial"]["p05"]),
            "speed_2_4_gain_p05": float(metrics["by_speed"]["2.4"]["p05_gain"]),
            "speed_2_8_gain_p05": float(metrics["by_speed"]["2.8"]["p05_gain"]),
        })
    return result


def warm_metrics(summary: dict[str, Any], name: str) -> dict[str, Any]:
    pooled = summary["runs"][name]["pooled"]
    return {
        "win_fraction": float(pooled["actor_strict_win_fraction"]),
        "gain_mean": float(pooled["gain_vs_warm"]["mean"]),
        "gain_median": float(pooled["gain_vs_warm"]["median"]),
        "gain_p05": float(pooled["gain_vs_warm"]["p05"]),
        "gain_worst": float(pooled["gain_vs_warm"]["minimum"]),
        "aggregate_relative_gain": float(pooled["aggregate_relative_gain"]),
        "speed_2_4": pooled["by_speed"]["2.4"],
        "speed_2_8": pooled["by_speed"]["2.8"],
    }


def engineering_pass(validator: dict[str, Any]) -> bool:
    return all(bool(record["passed"]) for record in validator["records"])


def main() -> None:
    stochastic_contract = load(STOCHASTIC / "contract.json")
    deterministic_contract = load(DETERMINISTIC / "contract.json")
    stochastic_summary = load(STOCHASTIC / "summary.json")
    deterministic_summary = load(DETERMINISTIC / "summary.json")
    stochastic_validator = load(STOCHASTIC / "validator_report.json")
    deterministic_validator = load(DETERMINISTIC / "validator_report.json")
    warm = load(WARM / "summary.json")
    warm_validator = load(WARM / "validator_report.json")
    capacity = load(CAPACITY / "summary.json")

    direct_contract = deterministic_contract["actor_gradient_source_contract"]
    checks = {
        "only_objective_mode_argument_differs": (
            normalized_arguments(stochastic_contract)
            == normalized_arguments(deterministic_contract)
        ),
        "both_use_exact_dbm_gradient_source": (
            stochastic_contract["arguments"]["actor_gradient_source"] == "dbm"
            and deterministic_contract["arguments"]["actor_gradient_source"] == "dbm"
        ),
        "stochastic_mode_registered": (
            stochastic_contract["arguments"].get("actor_objective_mode", "stochastic_sac")
            == "stochastic_sac"
        ),
        "deterministic_mode_registered": (
            deterministic_contract["arguments"]["actor_objective_mode"]
            == "deterministic_center_dbm"
            and direct_contract["deterministic_center_dbm"]["sampled_action_affects_actor_loss"] is False
            and direct_contract["deterministic_center_dbm"]["move_coefficient_affects_actor_loss"] is False
            and direct_contract["deterministic_center_dbm"]["entropy_affects_actor_loss"] is False
            and direct_contract["deterministic_center_dbm"]["tail_affects_actor_loss"] is False
        ),
        "stochastic_engineering_all_seed_pass": engineering_pass(stochastic_validator),
        "deterministic_engineering_all_seed_pass": engineering_pass(deterministic_validator),
        "warm_relative_validator_pass": warm_validator["qualification"]
        == "OAC_WARM_RELATIVE_DIRECT_CENTER_VALIDATION_PASS",
        "formal_validation_sealed": (
            not stochastic_summary["formal_validation_loaded"]
            and not deterministic_summary["formal_validation_loaded"]
        ),
        "test_sealed": (
            not stochastic_summary["test_loaded"]
            and not deterministic_summary["test_loaded"]
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)

    stochastic_latest = latest(stochastic_summary)
    deterministic_latest = latest(deterministic_summary)
    stochastic_warm = warm_metrics(warm, "matched_dbm")
    deterministic_warm = warm_metrics(warm, "deterministic_center")
    recovery_delta = (
        deterministic_latest["headroom_recovery"]["mean"]
        - stochastic_latest["headroom_recovery"]["mean"]
    )
    warm_delta = {
        "win_fraction_pp": 100.0 * (
            deterministic_warm["win_fraction"] - stochastic_warm["win_fraction"]
        ),
        "gain_mean": deterministic_warm["gain_mean"] - stochastic_warm["gain_mean"],
        "gain_median": deterministic_warm["gain_median"] - stochastic_warm["gain_median"],
        "gain_p05": deterministic_warm["gain_p05"] - stochastic_warm["gain_p05"],
        "gain_worst": deterministic_warm["gain_worst"] - stochastic_warm["gain_worst"],
        "aggregate_relative_gain_pp": 100.0 * (
            deterministic_warm["aggregate_relative_gain"]
            - stochastic_warm["aggregate_relative_gain"]
        ),
        "speed_2_4_gain_mean": (
            deterministic_warm["speed_2_4"]["gain_mean"]
            - stochastic_warm["speed_2_4"]["gain_mean"]
        ),
        "speed_2_8_gain_mean": (
            deterministic_warm["speed_2_8"]["gain_mean"]
            - stochastic_warm["speed_2_8"]["gain_mean"]
        ),
        "speed_2_8_gain_p05": (
            deterministic_warm["speed_2_8"]["gain_p05"]
            - stochastic_warm["speed_2_8"]["gain_p05"]
        ),
    }
    capacity_recovery = float(
        capacity["records"]["box3"]["episode_heldout"]["headroom_recovery"]
    )
    deterministic_recovery = deterministic_latest["headroom_recovery"]["mean"]
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "DETERMINISTIC_CENTER_OBJECTIVE_SMALL_GAIN_TAIL_GATE_STILL_BLOCKS",
        "scope": (
            "fold-1 train-side internal-selection latest checkpoints; exact DBM in both arms; "
            "K16/90 rounds; no formal validation, test, MPPI wrapper, or closed loop"
        ),
        "checks": checks,
        "latest": {
            "stochastic_exact_dbm": stochastic_latest,
            "deterministic_center_dbm": deterministic_latest,
            "recovery_delta_pp": 100.0 * recovery_delta,
            "cost_mean_delta": (
                deterministic_latest["cost_mean"]["mean"]
                - stochastic_latest["cost_mean"]["mean"]
            ),
        },
        "best_recovery_evaluation": best_recovery(deterministic_summary),
        "warm_relative": {
            "stochastic_exact_dbm": stochastic_warm,
            "deterministic_center_dbm": deterministic_warm,
            "deterministic_minus_stochastic": warm_delta,
        },
        "checkpoint_gate": {
            "deterministic_selected_rounds": [
                int(record["selected_round"]) for record in deterministic_summary["records"]
            ],
            "accepted_nonzero_evaluations": sum(
                bool(row["accepted"])
                for record in deterministic_summary["records"]
                for row in record["evaluations"] if int(row["round"]) > 0
            ),
            "candidate_p05_floors": {
                "all": float(deterministic_contract["arguments"]["selection_gain_p05_floor"]),
                "speed_2_4": float(deterministic_contract["arguments"]["selection_speed_2_4_gain_p05_floor"]),
                "speed_2_8": float(deterministic_contract["arguments"]["selection_speed_2_8_gain_p05_floor"]),
            },
        },
        "historical_capacity_reference": {
            "recovery": capacity_recovery,
            "remaining_gap_pp": 100.0 * (capacity_recovery - deterministic_recovery),
            "caveat": (
                "The 0.9326 run used fold-0 episode-heldout, 2400 direct-task-loss updates, "
                "and a different training contract; it is a scale reference, not a paired gate."
            ),
        },
        "decision": {
            "objective_mismatch": (
                "Removing sampled-action, entropy, move-coefficient, and tail gradients gives a "
                "small consistent central improvement, but does not approach the historical capacity. "
                "The stochastic deployment-mean mismatch is real but not the main remaining gap."
            ),
            "tail": (
                "Pooled warm-relative P05/worst improve slightly, but speed-2.8 mean and P05 regress, "
                "and every nonzero candidate violates the registered single-center P05 floors."
            ),
            "next": (
                "Prioritize a per-state shared-Actor parameter-gradient conflict audit under exact DBM. "
                "Do not expand data or change Actor architecture before distinguishing gradient "
                "cancellation from optimization budget/trajectory effects."
            ),
            "authorization": (
                "Engineering and replay contracts pass for all seeds, but 0/3 performance gates pass. "
                "No deployment or closed-loop authorization; retain the two-center warm guard."
            ),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "qualification": result["qualification"],
        "latest": result["latest"],
        "warm_delta": warm_delta,
        "checkpoint_gate": result["checkpoint_gate"],
        "historical_capacity_reference": result["historical_capacity_reference"],
    }, indent=2))


if __name__ == "__main__":
    main()
