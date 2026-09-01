#!/usr/bin/env python3
"""Create the decision summary for the OAC-2 Critic-vs-DBM gap audit."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def mean(values):
    return float(statistics.mean(float(value) for value in values))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audit_dir", type=Path,
        default=Path("outputs/mppi_proposal/oac2_critic_dbm_gradient_gap_20260825_v3"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analysis = json.loads((args.audit_dir / "analysis.json").read_text())
    validator = json.loads((args.audit_dir / "validator_report.json").read_text())
    rows = analysis["records"]
    run_dir = Path(analysis["contract"]["run_dir"])
    latest = [row["roles"]["latest"] for row in rows]
    gradient = [row["gradient"]["twin_conservative"] for row in latest]
    parameter = [row["parameter_gradient"]["latest"] for row in rows]
    parameter_steps = [row["parameter_step"] for row in rows]
    update_scale = []
    for seed in range(3):
        records = [
            json.loads(line) for line in
            (run_dir / f"seed_{seed}" / "iteration_metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        values = [float(row["actor"]["step_sigma_rms"]) for row in records]
        update_scale.append({
            "seed": seed,
            "median_per_round_sigma_rms": float(statistics.median(values)),
            "mean_per_round_sigma_rms": mean(values),
            "maximum_per_round_sigma_rms": max(values),
            "sum_path_length_sigma_rms": sum(values),
            "trust_projection_count": sum(
                float(row["actor"]["trust_projection"]) < 1.0 for row in records
            ),
        })

    def fixed_step(name):
        values = [row["steps"][name]["all"] for row in latest]
        return {
            "gain_mean": mean(value["gain"]["mean"] for value in values),
            "gain_median": mean(value["gain"]["median"] for value in values),
            "gain_p05": mean(value["gain"]["p05"] for value in values),
            "regressed_fraction": mean(value["regressed_fraction"] for value in values),
        }

    def parameter_step(name):
        values = [row["metrics"][name] for row in parameter_steps]
        return {
            "gain_mean": mean(value["gain"]["mean"] for value in values),
            "gain_median": mean(value["gain"]["median"] for value in values),
            "gain_p05": mean(value["gain"]["p05"] for value in values),
            "worst_gain": mean(value["gain"]["minimum"] for value in values),
            "regressed_fraction": mean(value["regressed_fraction"] for value in values),
        }

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "CRITIC_LOCAL_GRADIENT_MOSTLY_RECOVERED_OBJECTIVE_AND_ACTOR_UPDATE_CONTRACT_DOMINATE_GAP",
        "validator_qualification": validator["qualification"],
        "latest_actor_action_space": {
            "conservative_value_pearson_log_mean": mean(
                row["value"]["conservative_pearson_log"] for row in latest
            ),
            "gradient_cosine_median_mean": mean(
                row["cosine"]["median"] for row in gradient
            ),
            "gradient_cosine_p10_mean": mean(
                row["cosine"]["p10"] for row in gradient
            ),
            "gradient_norm_ratio_median_mean": mean(
                row["norm_ratio"]["median"] for row in gradient
            ),
            "early_steering_cosine_median_mean": mean(
                row["component_cosine"]["early_steering_0_2"]["median"] for row in gradient
            ),
            "early_steering_cosine_p10_mean": mean(
                row["component_cosine"]["early_steering_0_2"]["p10"] for row in gradient
            ),
            "flat_q25_cosine_median_mean": mean(
                row["gradient"]["twin_conservative_by_regime"]["flat_true_gradient_q25"]["cosine"]["median"]
                for row in latest
            ),
            "flat_q25_cosine_p10_mean": mean(
                row["gradient"]["twin_conservative_by_regime"]["flat_true_gradient_q25"]["cosine"]["p10"]
                for row in latest
            ),
            "twin_mean_p10_mean": mean(
                row["gradient"]["twin_mean"]["cosine"]["p10"] for row in latest
            ),
            "conservative_twin_p10_mean": mean(
                row["gradient"]["twin_conservative"]["cosine"]["p10"] for row in latest
            ),
        },
        "latest_actor_fixed_action_step": {
            name: fixed_step(name) for name in (
                "true_trust_0.002", "critic_trust_0.002",
                "true_trust_0.005", "critic_trust_0.005",
                "true_trust_0.02", "critic_trust_0.02",
            )
        },
        "latest_actor_parameter_gradient": {
            "critic_vs_exact_log_cosine": [
                row["critic_vs_dbm_log"]["all"]["cosine"] for row in parameter
            ],
            "exact_log_vs_exact_raw_cosine": [
                row["dbm_log_vs_dbm_raw"]["all"]["cosine"] for row in parameter
            ],
            "critic_vs_exact_raw_cosine": [
                row["critic_vs_dbm_raw"]["all"]["cosine"] for row in parameter
            ],
        },
        "matched_current_adam_parameter_step": {
            name: parameter_step(name) for name in (
                "dbm_raw_rms_0.00025", "dbm_log_rms_0.00025", "critic_rms_0.00025",
                "dbm_raw_rms_0.002", "dbm_log_rms_0.002", "critic_rms_0.002",
            )
        },
        "observed_online_actor_update_scale": update_scale,
        "attribution": {
            "critic_local_gradient": (
                "largely recovered on the final Actor-visited distribution; it adds a small-step tail "
                "penalty but does not explain the order-of-magnitude mean gap to direct DBM training"
            ),
            "objective": (
                "dominant measured mismatch: OAC minimizes per-state log1p(J), whereas the capacity "
                "run minimized batch mean raw J; shared-Actor parameter gradients diverge strongly"
            ),
            "actor_update_budget": (
                "also material: OAC uses 200 Actor steps at 1e-6 and observed about 0.00021--0.00028 "
                "sigma RMS per round; direct DBM capacity used 2400 updates at 2e-4"
            ),
            "remaining_tail": (
                "not eliminated: even a 0.002-sigma normalized Critic action step regresses about 4% "
                "of states versus near-zero for the exact direction; conservative max is slightly worse "
                "than Twin mean at P10"
            ),
        },
        "next_registered_test": (
            "Keep the trained Critic fixed and change only the Actor aggregation from log-risk-neutral "
            "toward raw-cost-aware weighting (tempered gamma 0/0.5/1 with clipping), then separately "
            "increase Actor update RMS/budget. Do not enlarge the Critic or add another gradient loss first."
        ),
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.audit_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
