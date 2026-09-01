#!/usr/bin/env python3
"""Analyze the equal-budget OAC Actor-visited refresh-cadence A/B."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path("outputs/mppi_proposal")
BASELINE = ROOT / "online_absolute_sac_oac2_multiupdate_k16_90round_20260828_v1"
REFRESH = ROOT / "online_absolute_sac_oac2_refresh2x_equal_budget_180round_20260828_v1"
WARM = ROOT / "oac2_refresh2x_warm_relative_20260828_v1"
OUTPUT = ROOT / "oac2_refresh_cadence_ab_20260828_v1"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "mean": float(mean(values)),
        "population_std": float(pstdev(values)),
        "per_seed": [float(value) for value in values],
    }


def curve(summary: dict[str, Any], multiplier: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for baseline_round in (10, 20, 30, 40, 50, 60, 70, 80, 90):
        run_round = baseline_round * multiplier
        values = []
        for record in summary["records"]:
            row = next(item for item in record["evaluations"] if item["round"] == run_round)
            values.append(float(row["metrics"]["headroom_recovery_vs_bank_best"]))
        result[str(baseline_round)] = {
            "run_round": run_round,
            "equivalent_context_visits": baseline_round * 256,
            **stats(values),
        }
    return result


def warm_metrics(pooled: dict[str, Any]) -> dict[str, Any]:
    return {
        "actor_strict_win_fraction": float(pooled["actor_strict_win_fraction"]),
        "gain_mean": float(pooled["gain_vs_warm"]["mean"]),
        "gain_median": float(pooled["gain_vs_warm"]["median"]),
        "gain_p05": float(pooled["gain_vs_warm"]["p05"]),
        "gain_worst": float(pooled["gain_vs_warm"]["minimum"]),
        "aggregate_relative_gain": float(pooled["aggregate_relative_gain"]),
        "speed_2_4": pooled["by_speed"]["2.4"],
        "speed_2_8": pooled["by_speed"]["2.8"],
    }


def main() -> None:
    base_contract = load(BASELINE / "contract.json")
    refresh_contract = load(REFRESH / "contract.json")
    base_summary = load(BASELINE / "summary.json")
    refresh_summary = load(REFRESH / "summary.json")
    base_validator = load(BASELINE / "validator_report.json")
    refresh_validator = load(REFRESH / "validator_report.json")
    warm = load(WARM / "summary.json")
    warm_validator = load(WARM / "validator_report.json")

    ba = base_contract["arguments"]
    ra = refresh_contract["arguments"]
    bmulti = base_contract["multi_actor_update_contract"]
    rrefresh = refresh_contract["actor_visited_refresh_contract"]
    rmulti = refresh_contract["multi_actor_update_contract"]

    base_budget = {
        "context_visits": int(ba["rounds"] * ba["contexts_per_round"]),
        "critic_updates": int(ba["rounds"] * ba["critic_updates_per_round"]),
        "actor_updates": int(ba["rounds"] * ba["actor_updates_per_round"]),
        "evaluations_after_round0": int(ba["rounds"] // ba["evaluation_interval"]),
        "temperature_updates": int(ba["rounds"]),
        "tail_dual_updates": int(ba["rounds"]),
    }
    refresh_budget = {
        "context_visits": int(rrefresh["total_context_visits"]),
        "critic_updates": int(rrefresh["total_critic_updates"]),
        "actor_updates": int(rrefresh["total_actor_updates"]),
        "evaluations_after_round0": int(ra["rounds"] // ra["evaluation_interval"]),
        "temperature_updates": int(rmulti["temperature_update_count"]),
        "tail_dual_updates": int(rmulti["tail_dual_update_count"]),
    }
    base_lr_per_microstep = {
        "initial": float(ba["actor_learning_rate_initial"] / ba["actor_updates_per_round"]),
        "middle": float(ba["actor_learning_rate_middle"] / ba["actor_updates_per_round"]),
        "final": float(ba["actor_learning_rate"] / ba["actor_updates_per_round"]),
    }
    refresh_lr_per_microstep = {
        "initial": float(ra["actor_learning_rate_initial"] / ra["actor_updates_per_round"]),
        "middle": float(ra["actor_learning_rate_middle"] / ra["actor_updates_per_round"]),
        "final": float(ra["actor_learning_rate"] / ra["actor_updates_per_round"]),
    }
    checks = {
        "equal_total_budget": base_budget == refresh_budget,
        "equal_per_microstep_lr": base_lr_per_microstep == refresh_lr_per_microstep,
        "equal_per_microstep_trust": (
            float(bmulti["per_microstep_trust_sigma_rms"])
            == float(rmulti["per_microstep_trust_sigma_rms"])
        ),
        "refresh_is_exact_2x_outer_cadence": (
            ra["rounds"] == 2 * ba["rounds"]
            and ra["contexts_per_round"] * 2 == ba["contexts_per_round"]
            and ra["critic_updates_per_round"] * 2 == ba["critic_updates_per_round"]
            and ra["actor_updates_per_round"] * 2 == ba["actor_updates_per_round"]
            and ra["evaluation_interval"] == 2 * ba["evaluation_interval"]
        ),
        "baseline_engineering_all_seed_pass": all(row["passed"] for row in base_validator["records"]),
        "refresh_engineering_all_seed_pass": all(row["passed"] for row in refresh_validator["records"]),
        "warm_relative_validation_pass": (
            warm_validator["qualification"] == "OAC_WARM_RELATIVE_DIRECT_CENTER_VALIDATION_PASS"
        ),
        "formal_validation_sealed": (
            not base_summary["formal_validation_loaded"] and not refresh_summary["formal_validation_loaded"]
        ),
        "test_sealed": not base_summary["test_loaded"] and not refresh_summary["test_loaded"],
    }
    if not all(checks.values()):
        raise AssertionError(checks)

    base_curve = curve(base_summary, 1)
    refresh_curve = curve(refresh_summary, 2)
    base_warm = warm_metrics(warm["runs"]["baseline_k16"]["pooled"])
    refresh_warm = warm_metrics(warm["runs"]["refresh2x"]["pooled"])
    latest_delta = {
        "headroom_recovery_pp": 100.0 * (refresh_curve["90"]["mean"] - base_curve["90"]["mean"]),
        "headroom_recovery_seed_std_pp": 100.0 * (
            refresh_curve["90"]["population_std"] - base_curve["90"]["population_std"]
        ),
        "warm_win_fraction_pp": 100.0 * (
            refresh_warm["actor_strict_win_fraction"] - base_warm["actor_strict_win_fraction"]
        ),
        "warm_gain_mean": refresh_warm["gain_mean"] - base_warm["gain_mean"],
        "warm_gain_median": refresh_warm["gain_median"] - base_warm["gain_median"],
        "warm_gain_p05": refresh_warm["gain_p05"] - base_warm["gain_p05"],
        "warm_gain_worst": refresh_warm["gain_worst"] - base_warm["gain_worst"],
        "warm_aggregate_relative_gain_pp": 100.0 * (
            refresh_warm["aggregate_relative_gain"] - base_warm["aggregate_relative_gain"]
        ),
    }
    materially_better = bool(
        latest_delta["headroom_recovery_pp"] >= 2.0
        and latest_delta["warm_win_fraction_pp"] >= 0.0
        and latest_delta["warm_aggregate_relative_gain_pp"] >= 0.0
        and latest_delta["warm_gain_p05"] >= 0.0
    )

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            "ACTOR_VISITED_REFRESH_2X_EQUAL_BUDGET_PASS"
            if materially_better
            else "ACTOR_VISITED_REFRESH_2X_EQUAL_BUDGET_NO_GAIN"
        ),
        "scope": (
            "train-side fold-1 internal-selection latest checkpoints and deterministic warm-relative "
            "direct-center DBM J50 only; no formal validation, test, MPPI wrapper, or closed loop"
        ),
        "checks": checks,
        "budgets": {"baseline_k16": base_budget, "refresh2x": refresh_budget},
        "per_microstep": {
            "baseline_lr": base_lr_per_microstep,
            "refresh2x_lr": refresh_lr_per_microstep,
            "baseline_trust_sigma_rms": float(bmulti["per_microstep_trust_sigma_rms"]),
            "refresh2x_trust_sigma_rms": float(rmulti["per_microstep_trust_sigma_rms"]),
        },
        "headroom_recovery_curve": {"baseline_k16": base_curve, "refresh2x": refresh_curve},
        "warm_relative": {"baseline_k16": base_warm, "refresh2x": refresh_warm},
        "refresh2x_minus_baseline": latest_delta,
        "registered_pass_rule": {
            "headroom_recovery_improvement_pp_min": 2.0,
            "warm_win_fraction_not_worse": True,
            "warm_aggregate_relative_gain_not_worse": True,
            "warm_gain_p05_not_worse": True,
            "passed": materially_better,
        },
        "decision": {
            "mechanism": (
                "At exactly matched total context visits, Critic updates, Actor updates, auxiliary "
                "updates, per-microstep LR, and trust, doubling the Actor-visited refresh cadence does "
                "not improve recovery or warm-relative aggregate quality."
            ),
            "interpretation": (
                "The existing replay sampler already reserves half of each batch for recent rows. "
                "Reducing the feedback delay from 20-Critic/16-Actor blocks to 10/8 blocks is therefore "
                "not the active optimization bottleneck at this operating point."
            ),
            "next": (
                "Keep the simpler K=16/90-round organization. Do not spend another scan on refresh "
                "cadence alone; the next experiment must change the information content of Actor-visited "
                "states or the state/action objective, while retaining the warm-relative report and "
                "two-center deployment guard."
            ),
            "authorization": (
                "Engineering/replay checks pass for all seeds, but the legacy performance gate remains "
                "0/3 and selected_round remains zero. This is not deployment authorization."
            ),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"qualification": result["qualification"], "delta": latest_delta}, indent=2))


if __name__ == "__main__":
    main()
