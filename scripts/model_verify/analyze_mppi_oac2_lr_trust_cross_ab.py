#!/usr/bin/env python3
"""Analyze the 3.2e-4 versus 6.4e-4 Actor LR A/B at 0.06-sigma trust."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from analyze_mppi_oac2_trust_boundary_scan import load_run


DEFAULT_BASELINE = Path(
    "outputs/mppi_proposal/"
    "online_absolute_sac_oac2_rawcost_gamma1_lr32e5_trust006_90round_20260827_v1"
)
DEFAULT_CANDIDATE = Path(
    "outputs/mppi_proposal/"
    "online_absolute_sac_oac2_rawcost_gamma1_lr64e5_trust006_90round_20260827_v1"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/oac2_lr_trust_cross_ab_20260827_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def paired_delta(candidate: dict, baseline: dict) -> dict:
    higher_is_better = (
        "mean_gain", "median_gain", "p05_gain", "worst_gain",
        "headroom_recovery", "speed_2_4_mean_gain", "speed_2_4_p05_gain",
        "speed_2_8_mean_gain", "speed_2_8_p05_gain",
    )
    lower_is_better = (
        "mean_cost", "median_cost", "regression_fraction", "guard_mean_cost",
        "guard_warm_selected_fraction", "action_saturation_fraction",
        "state_any_saturation_fraction",
    )
    fields = higher_is_better + lower_is_better
    result = {
        field: candidate[field] - baseline[field]
        for field in fields
    }
    for field in higher_is_better:
        result[f"{field}_improved_seed_count"] = sum(
            candidate["per_seed"][seed][field]
            > baseline["per_seed"][seed][field]
            for seed in range(3)
        )
    for field in lower_is_better:
        result[f"{field}_improved_seed_count"] = sum(
            candidate["per_seed"][seed][field]
            < baseline["per_seed"][seed][field]
            for seed in range(3)
        )
    result["headroom_recovery_pp"] = 100.0 * result["headroom_recovery"]
    result["regression_fraction_pp"] = 100.0 * result["regression_fraction"]
    return result


def main() -> None:
    args = parse_args()
    baseline = load_run("lr3.2e-4_trust0.06", args.baseline)
    candidate = load_run("lr6.4e-4_trust0.06", args.candidate)
    delta = paired_delta(candidate["primary"], baseline["primary"])
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "LR64E5_TRUST006_MEAN_MIXED_GAIN_NEAR_JOINT_BOUNDARY",
        "scope": (
            "fold-1 train-side internal-selection; gamma=1 box3 adaptive-tail; "
            "trust 0.06 sigma; round-80 matched; three seeds; formal validation/test sealed"
        ),
        "checks": {
            "baseline_engineering_validation_pass": baseline[
                "engineering_validation_pass"
            ],
            "candidate_engineering_validation_pass": candidate[
                "engineering_validation_pass"
            ],
            "formal_validation_sealed": (
                not baseline["formal_validation_loaded"]
                and not candidate["formal_validation_loaded"]
            ),
            "test_sealed": (
                not baseline["test_loaded"] and not candidate["test_loaded"]
            ),
            "trust_fixed": baseline["trust_sigma_rms"] == candidate["trust_sigma_rms"] == 0.06,
            "candidate_lr_doubled": (
                candidate["initial_learning_rate"]
                == 2.0 * baseline["initial_learning_rate"]
            ),
            "selected_rounds_zero": (
                baseline["selected_rounds"] == [0, 0, 0]
                and candidate["selected_rounds"] == [0, 0, 0]
            ),
            "performance_gate_counts_zero": (
                baseline["performance_gate_passed_seed_count"] == 0
                and candidate["performance_gate_passed_seed_count"] == 0
            ),
        },
        "runs": {
            "lr3.2e-4_trust0.06": baseline,
            "lr6.4e-4_trust0.06": candidate,
        },
        "paired_delta_round80": delta,
        "decision": {
            "mean": (
                "doubling the initial LR improves average mean cost and headroom recovery, "
                "but only two of three seeds improve and the median cost worsens overall"
            ),
            "update_scale": (
                "the candidate uses most of the 0.06-sigma allowance: early mean step is "
                "near 0.059 sigma and 75% of early updates are projected"
            ),
            "boundary": (
                "the result proves the previous 3.2e-4 arm underused the 0.06 trust cap, "
                "but the mixed seed/median result indicates proximity to the joint LR-trust "
                "boundary rather than opening another clearly scalable gain regime"
            ),
            "tail": (
                "regression fraction and P05 deteriorate materially; no checkpoint passes "
                "the deployment selection gate"
            ),
            "next": (
                "keep 3.2e-4 plus 0.04 sigma as the stable operating point, and use the "
                "6.4e-4 plus 0.06 arm only as a mean-capacity probe for the matched DBM-"
                "versus-Critic gradient-source A/B; a fully saturated higher-LR arm would "
                "still be required to claim a strict pure-mean ceiling"
            ),
        },
    }
    if not all(result["checks"].values()):
        raise AssertionError(result["checks"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "analysis.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "qualification": result["qualification"],
        "round80": {
            key: {
                "mean_cost": run["primary"]["mean_cost"],
                "median_cost": run["primary"]["median_cost"],
                "headroom_recovery": run["primary"]["headroom_recovery"],
                "regression_fraction": run["primary"]["regression_fraction"],
                "p05_gain": run["primary"]["p05_gain"],
                "early_step": run["early_action_step"],
            }
            for key, run in result["runs"].items()
        },
        "paired_delta": delta,
        "output": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()
