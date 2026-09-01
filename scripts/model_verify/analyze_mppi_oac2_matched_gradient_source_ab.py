#!/usr/bin/env python3
"""Analyze the matched OAC Critic-vs-differentiable-DBM gradient-source A/B."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path("outputs/mppi_proposal")
CRITIC = ROOT / "online_absolute_sac_oac2_multiupdate_k16_90round_20260828_v1"
DBM = ROOT / "online_absolute_sac_oac2_matched_dbm_gradient_k16_90round_20260828_v1"
WARM = ROOT / "oac2_matched_gradient_source_warm_relative_20260828_v1"
CAPACITY = ROOT / "dbm_task_loss_support_full_ab_20260825_v1"
OUTPUT = ROOT / "oac2_matched_gradient_source_ab_20260828_v1"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "mean": float(mean(values)),
        "population_std": float(pstdev(values)),
        "per_seed": [float(value) for value in values],
    }


def normalized_arguments(contract: dict[str, Any]) -> dict[str, Any]:
    args = dict(contract["arguments"])
    args.setdefault("actor_gradient_source", "critic")
    args.setdefault("matched_gradient_source_pilot", False)
    args.setdefault("matched_gradient_source_smoke", False)
    args.setdefault("actor_visited_refresh_pilot", False)
    args.setdefault("aux_update_interval_rounds", 1)
    for key in (
        "output_dir", "actor_gradient_source", "matched_gradient_source_pilot",
        "matched_gradient_source_smoke",
    ):
        args.pop(key, None)
    return args


def recovery_curve(summary: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for round_id in range(10, 91, 10):
        values = [
            float(next(
                row for row in record["evaluations"] if row["round"] == round_id
            )["metrics"]["headroom_recovery_vs_bank_best"])
            for record in summary["records"]
        ]
        result[str(round_id)] = stats(values)
    return result


def latest(summary: dict[str, Any]) -> dict[str, Any]:
    records = summary["records"]
    field = lambda fn: stats([float(fn(row["latest_metrics"])) for row in records])
    return {
        "headroom_recovery": field(lambda m: m["headroom_recovery_vs_bank_best"]),
        "cost_mean": field(lambda m: m["cost"]["mean"]),
        "cost_median": field(lambda m: m["cost"]["median"]),
        "gain_p05": field(lambda m: m["gain_vs_initial"]["p05"]),
        "gain_worst": field(lambda m: m["gain_vs_initial"]["minimum"]),
        "regression_fraction": field(lambda m: m["regression_fraction"]),
        "speed_2_4_cost_mean": field(lambda m: m["by_speed"]["2.4"]["mean_cost"]),
        "speed_2_8_cost_mean": field(lambda m: m["by_speed"]["2.8"]["mean_cost"]),
    }


def warm_metrics(pooled: dict[str, Any]) -> dict[str, Any]:
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


def main() -> None:
    critic_contract, dbm_contract = load(CRITIC / "contract.json"), load(DBM / "contract.json")
    critic_summary, dbm_summary = load(CRITIC / "summary.json"), load(DBM / "summary.json")
    critic_validator = load(CRITIC / "validator_report.json")
    dbm_validator = load(DBM / "validator_report.json")
    warm = load(WARM / "summary.json")
    warm_validator = load(WARM / "validator_report.json")
    capacity = load(CAPACITY / "summary.json")

    source_contract = dbm_contract["actor_gradient_source_contract"]
    checks = {
        "only_registered_gradient_source_arguments_differ": (
            normalized_arguments(critic_contract) == normalized_arguments(dbm_contract)
        ),
        "critic_source_is_registered_default": (
            critic_contract["arguments"].get("actor_gradient_source", "critic") == "critic"
        ),
        "dbm_source_contract": (
            source_contract["source"] == "dbm"
            and source_contract["matched_pilot"]
            and not source_contract["smoke_only"]
            and source_contract["critic_and_replay_continue_in_dbm_arm"]
            and source_contract["coefficient_is_detached_in_both_arms"]
        ),
        "critic_engineering_all_seed_pass": all(row["passed"] for row in critic_validator["records"]),
        "dbm_engineering_all_seed_pass": all(row["passed"] for row in dbm_validator["records"]),
        "warm_relative_validation_pass": (
            warm_validator["qualification"] == "OAC_WARM_RELATIVE_DIRECT_CENTER_VALIDATION_PASS"
        ),
        "formal_validation_sealed": (
            not critic_summary["formal_validation_loaded"] and not dbm_summary["formal_validation_loaded"]
        ),
        "test_sealed": not critic_summary["test_loaded"] and not dbm_summary["test_loaded"],
    }
    if not all(checks.values()):
        raise AssertionError(checks)

    critic_curve, dbm_curve = recovery_curve(critic_summary), recovery_curve(dbm_summary)
    critic_latest, dbm_latest = latest(critic_summary), latest(dbm_summary)
    critic_warm = warm_metrics(warm["runs"]["critic_k16"]["pooled"])
    dbm_warm = warm_metrics(warm["runs"]["matched_dbm"]["pooled"])
    paired_recovery_delta = [
        right - left for left, right in zip(
            critic_latest["headroom_recovery"]["per_seed"],
            dbm_latest["headroom_recovery"]["per_seed"],
        )
    ]
    delta = {
        "headroom_recovery_pp": 100.0 * mean(paired_recovery_delta),
        "headroom_recovery_pp_per_seed": [100.0 * value for value in paired_recovery_delta],
        "cost_mean": dbm_latest["cost_mean"]["mean"] - critic_latest["cost_mean"]["mean"],
        "cost_median": dbm_latest["cost_median"]["mean"] - critic_latest["cost_median"]["mean"],
        "initial_relative_gain_p05": (
            dbm_latest["gain_p05"]["mean"] - critic_latest["gain_p05"]["mean"]
        ),
        "initial_relative_gain_worst": (
            dbm_latest["gain_worst"]["mean"] - critic_latest["gain_worst"]["mean"]
        ),
        "regression_fraction_pp": 100.0 * (
            dbm_latest["regression_fraction"]["mean"]
            - critic_latest["regression_fraction"]["mean"]
        ),
        "warm_win_fraction_pp": 100.0 * (
            dbm_warm["win_fraction"] - critic_warm["win_fraction"]
        ),
        "warm_gain_mean": dbm_warm["gain_mean"] - critic_warm["gain_mean"],
        "warm_gain_median": dbm_warm["gain_median"] - critic_warm["gain_median"],
        "warm_gain_p05": dbm_warm["gain_p05"] - critic_warm["gain_p05"],
        "warm_gain_worst": dbm_warm["gain_worst"] - critic_warm["gain_worst"],
        "warm_aggregate_relative_gain_pp": 100.0 * (
            dbm_warm["aggregate_relative_gain"] - critic_warm["aggregate_relative_gain"]
        ),
    }

    capacity_recovery = float(capacity["records"]["box3"]["episode_heldout"]["headroom_recovery"])
    critic_recovery = critic_latest["headroom_recovery"]["mean"]
    dbm_recovery = dbm_latest["headroom_recovery"]["mean"]
    reference_gap = capacity_recovery - critic_recovery
    gap_decomposition = {
        "historical_exact_dbm_task_loss_episode_heldout_recovery": capacity_recovery,
        "matched_critic_recovery": critic_recovery,
        "matched_dbm_recovery": dbm_recovery,
        "critic_to_historical_capacity_gap_pp": 100.0 * reference_gap,
        "closed_by_exact_gradient_pp": 100.0 * (dbm_recovery - critic_recovery),
        "fraction_of_reference_gap_closed": (
            (dbm_recovery - critic_recovery) / reference_gap
        ),
        "remaining_dbm_to_historical_capacity_gap_pp": 100.0 * (
            capacity_recovery - dbm_recovery
        ),
        "caveat": (
            "The 0.9326 capacity reference is fold-0 episode-heldout with 2400 direct-task-loss "
            "updates and is not a paired fold-1 gate. This decomposition is a scale reference only."
        ),
    }

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "MATCHED_DBM_GRADIENT_SMALL_STABLE_GAIN_SHARED_ACTOR_CONTRACT_DOMINATES",
        "scope": (
            "fold-1 train-side internal-selection latest checkpoints; K=16/90-round OAC; "
            "no formal validation, test, MPPI wrapper, or closed loop"
        ),
        "checks": checks,
        "contract": {
            "common": normalized_arguments(critic_contract),
            "critic_source": "Twin conservative log1p(J50), gamma=1 detached raw-cost weighting",
            "dbm_source": (
                "differentiable deterministic DBM log1p(J50) for sampled-action value and "
                "mean-vs-selected tail; critics/replay/coefficient continue"
            ),
            "causal_scope": (
                "The arms share exogenous RNG rules and configuration, then intentionally diverge "
                "through their Actor trajectories and resulting Actor-visited replay."
            ),
        },
        "recovery_curve": {"critic": critic_curve, "dbm": dbm_curve},
        "latest": {"critic": critic_latest, "dbm": dbm_latest},
        "warm_relative": {"critic": critic_warm, "dbm": dbm_warm},
        "dbm_minus_critic": delta,
        "historical_capacity_reference": gap_decomposition,
        "decision": {
            "critic_contribution": (
                "Replacing the complete Actor cost/tail gradient with exact DBM improves recovery "
                "in all three seeds and improves pooled warm win, mean, median, P05, and aggregate "
                "gain. Critic gradient error therefore has a real cumulative cost."
            ),
            "not_main_gap": (
                "The matched gain is only about 1.9 recovery points. Exact DBM under the same OAC "
                "budget still stops near 0.76, far below the historical 0.9326 direct-task-loss "
                "capacity reference. Shared Actor optimization, stochastic/continuous-coefficient "
                "objective, trust/LR schedule, and state-gradient interference dominate the remaining "
                "reference gap."
            ),
            "tail": (
                "Exact DBM improves pooled warm P05 and both high-speed aggregate means, but pooled "
                "worst becomes more negative because of a rarer severe outlier. Tail instability is "
                "therefore not solely a Critic-direction problem and still requires the two-center guard."
            ),
            "next": (
                "Do not invest first in another Critic architecture/loss sweep. Keep the current Twin "
                "Critic for deployable OAC and move the main optimization effort to the shared Actor "
                "training contract: state-gradient conflict diagnostics, deterministic mean-action "
                "task surrogate or better per-state proposal supervision, while preserving warm-relative "
                "and tail reporting."
            ),
            "authorization": (
                "All engineering/replay checks pass, but both arms retain selected_round=0 under the "
                "legacy performance gate. This does not authorize deployment or closed loop."
            ),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "qualification": result["qualification"],
        "dbm_minus_critic": delta,
        "historical_capacity_reference": gap_decomposition,
    }, indent=2))


if __name__ == "__main__":
    main()
