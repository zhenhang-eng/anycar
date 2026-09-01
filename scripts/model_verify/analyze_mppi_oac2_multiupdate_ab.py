#!/usr/bin/env python3
"""Analyze the registered K=1/4/8 OAC Actor-microstep mechanism A/B."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable


ROOT = Path("outputs/mppi_proposal")
RUNS = {
    1: ROOT / "online_absolute_sac_oac2_multiupdate_k1_90round_20260827_v1",
    4: ROOT / "online_absolute_sac_oac2_multiupdate_k4_90round_20260827_v1",
    8: ROOT / "online_absolute_sac_oac2_multiupdate_k8_90round_20260827_v1",
    16: ROOT / "online_absolute_sac_oac2_multiupdate_k16_90round_20260828_v1",
    20: ROOT / "online_absolute_sac_oac2_multiupdate_k20_90round_20260828_v1",
    32: ROOT / "online_absolute_sac_oac2_multiupdate_k32_90round_20260828_v1",
}
WARM_RELATIVE = ROOT / "oac2_multiupdate_warm_relative_20260828_v4"
OUTPUT = ROOT / "oac2_multiupdate_boundary_20260828_v3"
ROUNDS = (10, 20, 40, 80, 90)


def avg_sd(values: list[float]) -> dict[str, Any]:
    return {
        "mean": float(mean(values)),
        "population_std": float(pstdev(values)),
        "per_seed": [float(value) for value in values],
    }


def metric(records: list[dict[str, Any]], getter: Callable[[dict[str, Any]], float]) -> dict[str, Any]:
    return avg_sd([float(getter(record["latest_metrics"])) for record in records])


def main() -> None:
    summaries: dict[int, dict[str, Any]] = {}
    contracts: dict[int, dict[str, Any]] = {}
    validators: dict[int, dict[str, Any]] = {}
    for k, root in RUNS.items():
        summaries[k] = json.loads((root / "summary.json").read_text())
        contracts[k] = json.loads((root / "contract.json").read_text())
        validators[k] = json.loads((root / "validator_report.json").read_text())

    base_args = dict(contracts[1]["arguments"])
    for field in ("output_dir", "actor_updates_per_round"):
        base_args.pop(field)
    only_registered_k_differs = True
    for k in (4, 8, 16, 20, 32):
        args = dict(contracts[k]["arguments"])
        for field in ("output_dir", "actor_updates_per_round"):
            args.pop(field)
        only_registered_k_differs &= args == base_args

    engineering_pass = {
        str(k): bool(
            all(record["passed"] for record in validators[k]["records"])
            and validators[k]["checks"]["formal_validation_sealed"]
            and validators[k]["checks"]["test_sealed"]
        )
        for k in RUNS
    }
    if not only_registered_k_differs or not all(engineering_pass.values()):
        raise AssertionError("registered A/B or engineering/replay contract failed")

    warm = json.loads((WARM_RELATIVE / "summary.json").read_text())
    warm_validator = json.loads((WARM_RELATIVE / "validator_report.json").read_text())
    if warm_validator["qualification"] != "OAC_WARM_RELATIVE_DIRECT_CENTER_VALIDATION_PASS":
        raise AssertionError("warm-relative evaluator did not validate")

    arms: dict[str, Any] = {}
    for k in RUNS:
        records = summaries[k]["records"]
        curve: dict[str, Any] = {}
        for round_id in ROUNDS:
            values = []
            for record in records:
                row = next(item for item in record["evaluations"] if item["round"] == round_id)
                values.append(float(row["metrics"]["headroom_recovery_vs_bank_best"]))
            curve[str(round_id)] = avg_sd(values)

        warm_pooled = warm["runs"][f"k{k}"]["pooled"]
        arms[f"k{k}"] = {
            "actor_microsteps_per_round": k,
            "actor_updates_per_seed": int(summaries[k]["actor_update_count_per_seed"]),
            "critic_updates_per_round": 20,
            "headroom_recovery_curve": curve,
            "latest_initial_relative": {
                "headroom_recovery": metric(records, lambda m: m["headroom_recovery_vs_bank_best"]),
                "cost_mean": metric(records, lambda m: m["cost"]["mean"]),
                "cost_median": metric(records, lambda m: m["cost"]["median"]),
                "gain_mean": metric(records, lambda m: m["gain_vs_initial"]["mean"]),
                "gain_median": metric(records, lambda m: m["gain_vs_initial"]["median"]),
                "gain_p05": metric(records, lambda m: m["gain_vs_initial"]["p05"]),
                "gain_worst": metric(records, lambda m: m["gain_vs_initial"]["minimum"]),
                "regression_fraction": metric(records, lambda m: m["regression_fraction"]),
            },
            "latest_warm_relative_direct_center": {
                "actor_strict_win_fraction": float(warm_pooled["actor_strict_win_fraction"]),
                "gain_mean": float(warm_pooled["gain_vs_warm"]["mean"]),
                "gain_median": float(warm_pooled["gain_vs_warm"]["median"]),
                "gain_p05": float(warm_pooled["gain_vs_warm"]["p05"]),
                "gain_worst": float(warm_pooled["gain_vs_warm"]["minimum"]),
                "aggregate_relative_gain": float(warm_pooled["aggregate_relative_gain"]),
                "actor_cost_mean": float(warm_pooled["actor_cost"]["mean"]),
                "actor_cost_median": float(warm_pooled["actor_cost"]["median"]),
            },
            "selected_rounds": [int(record["selected_round"]) for record in records],
            "engineering_replay_seed_pass": [bool(record["passed"]) for record in validators[k]["records"]],
            "formal_performance_gate_passed_seed_count": int(validators[k]["passed_seed_count"]),
        }

    k1 = arms["k1"]
    k8 = arms["k8"]
    delta = {
        "round10_headroom_recovery_pp": 100.0 * (
            k8["headroom_recovery_curve"]["10"]["mean"]
            - k1["headroom_recovery_curve"]["10"]["mean"]
        ),
        "round90_headroom_recovery_pp": 100.0 * (
            k8["headroom_recovery_curve"]["90"]["mean"]
            - k1["headroom_recovery_curve"]["90"]["mean"]
        ),
        "round90_seed_std_pp": 100.0 * (
            k8["headroom_recovery_curve"]["90"]["population_std"]
            - k1["headroom_recovery_curve"]["90"]["population_std"]
        ),
        "warm_win_fraction_pp": 100.0 * (
            k8["latest_warm_relative_direct_center"]["actor_strict_win_fraction"]
            - k1["latest_warm_relative_direct_center"]["actor_strict_win_fraction"]
        ),
        "warm_gain_median": (
            k8["latest_warm_relative_direct_center"]["gain_median"]
            - k1["latest_warm_relative_direct_center"]["gain_median"]
        ),
        "warm_gain_p05": (
            k8["latest_warm_relative_direct_center"]["gain_p05"]
            - k1["latest_warm_relative_direct_center"]["gain_p05"]
        ),
        "warm_aggregate_relative_gain_pp": 100.0 * (
            k8["latest_warm_relative_direct_center"]["aggregate_relative_gain"]
            - k1["latest_warm_relative_direct_center"]["aggregate_relative_gain"]
        ),
    }
    boundary_deltas = {}
    for lower, upper in ((8, 16), (16, 20), (20, 32), (16, 32)):
        left = arms[f"k{lower}"]
        right = arms[f"k{upper}"]
        boundary_deltas[f"k{upper}_minus_k{lower}"] = {
            "round90_headroom_recovery_pp": 100.0 * (
                right["headroom_recovery_curve"]["90"]["mean"]
                - left["headroom_recovery_curve"]["90"]["mean"]
            ),
            "warm_win_fraction_pp": 100.0 * (
                right["latest_warm_relative_direct_center"]["actor_strict_win_fraction"]
                - left["latest_warm_relative_direct_center"]["actor_strict_win_fraction"]
            ),
            "warm_gain_mean": (
                right["latest_warm_relative_direct_center"]["gain_mean"]
                - left["latest_warm_relative_direct_center"]["gain_mean"]
            ),
            "warm_gain_median": (
                right["latest_warm_relative_direct_center"]["gain_median"]
                - left["latest_warm_relative_direct_center"]["gain_median"]
            ),
            "warm_gain_p05": (
                right["latest_warm_relative_direct_center"]["gain_p05"]
                - left["latest_warm_relative_direct_center"]["gain_p05"]
            ),
            "warm_gain_worst": (
                right["latest_warm_relative_direct_center"]["gain_worst"]
                - left["latest_warm_relative_direct_center"]["gain_worst"]
            ),
            "warm_aggregate_relative_gain_pp": 100.0 * (
                right["latest_warm_relative_direct_center"]["aggregate_relative_gain"]
                - left["latest_warm_relative_direct_center"]["aggregate_relative_gain"]
            ),
        }

    analysis = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "K20_ONE_TO_ONE_NO_DOMINANT_GAIN_KEEP_K16_K32_BOUNDARY",
        "scope": (
            "train-side fold-1 internal-selection only; latest checkpoint capacity; "
            "no formal validation, test, wrapper, sampling, or closed loop"
        ),
        "checks": {
            "only_output_directory_and_registered_k_argument_differ": only_registered_k_differs,
            "engineering_replay_all_seed_pass": engineering_pass,
            "warm_relative_validator_pass": True,
            "formal_validation_sealed": all(not summaries[k]["formal_validation_loaded"] for k in RUNS),
            "test_sealed": all(not summaries[k]["test_loaded"] for k in RUNS),
        },
        "arms": arms,
        "k8_minus_k1": delta,
        "boundary_deltas": boundary_deltas,
        "decision": {
            "mechanism": (
                "Increasing K through 16 materially improves Actor optimization under the same twenty "
                "Critic updates and 0.06-sigma cumulative round trust. Exact one-Critic/one-Actor "
                "interleaving at K=20 does not dominate K=16: it slightly lowers recovery, warm win "
                "fraction, mean gain, and pooled worst, while improving only median and P05 marginally."
            ),
            "remaining_gap": (
                "K=32 reaches 0.749 mean internal headroom recovery, still below the historical "
                "exact-DBM task-loss heldout capacity reference 0.9326; this historical number is a "
                "cross-split capacity reference, not a strict paired gate."
            ),
            "tail": (
                "K=16 is the first arm with positive pooled warm-relative mean gain; K=32 further "
                "improves mean, P05, and pooled worst, but 2.8-m/s and per-seed worst tails remain open. "
                "Neither arm is a deployment authorization."
            ),
            "selection": (
                "The legacy initial-relative tail/high-speed selection gate keeps selected_round=0 for "
                "all arms. Latest is reported only for mechanism capacity; engineering/replay passes "
                "must not be confused with formal performance-gate passage."
            ),
            "next": (
                "Keep K=16 as the compute-efficient default; exact K=20 one-to-one scheduling has no "
                "aggregate advantage. Keep K=32 only when prioritizing offline center mean/P05 over "
                "training cost and strict win fraction. Stop K scans before K=64 and move next to "
                "Actor-visited refresh/Replay freshness while retaining two-center guard and sealed splits."
            ),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
