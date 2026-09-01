#!/usr/bin/env python3
"""Summarize the high-LR/trust gamma=1 versus gamma=0.5 warm-relative A/B."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


DEFAULT_INPUT = Path("outputs/mppi_proposal/oac_warm_relative_gamma05_high_ab_20260827_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads((args.input_dir / "summary.json").read_text())
    validator = json.loads((args.input_dir / "validator_report.json").read_text())
    if validator["qualification"] != "OAC_WARM_RELATIVE_DIRECT_CENTER_VALIDATION_PASS":
        raise AssertionError("warm-relative rescore validator did not pass")
    baseline = summary["runs"]["gamma1_high"]
    candidate = summary["runs"]["gamma05_high"]
    baseline_manifest = summary["manifest"]["runs"]["gamma1_high"]
    candidate_manifest = summary["manifest"]["runs"]["gamma05_high"]
    baseline_contract = json.loads(
        (Path(baseline_manifest["path"]) / "contract.json").read_text()
    )
    candidate_contract = json.loads(
        (Path(candidate_manifest["path"]) / "contract.json").read_text()
    )
    ignored = {"output_dir", "actor_cost_weight_gamma"}
    left = {
        key: value for key, value in baseline_contract["arguments"].items()
        if key not in ignored
    }
    right = {
        key: value for key, value in candidate_contract["arguments"].items()
        if key not in ignored
    }
    if left != right:
        raise AssertionError("A/B contracts differ beyond gamma/output directory")
    base = baseline["pooled"]
    cand = candidate["pooled"]
    fields = {
        "actor_strict_win_fraction": (
            cand["actor_strict_win_fraction"], base["actor_strict_win_fraction"]
        ),
        "gain_mean": (
            cand["gain_vs_warm"]["mean"], base["gain_vs_warm"]["mean"]
        ),
        "gain_median": (
            cand["gain_vs_warm"]["median"], base["gain_vs_warm"]["median"]
        ),
        "gain_p05": (
            cand["gain_vs_warm"]["p05"], base["gain_vs_warm"]["p05"]
        ),
        "gain_worst": (
            cand["gain_vs_warm"]["minimum"], base["gain_vs_warm"]["minimum"]
        ),
        "actor_cost_mean": (
            cand["actor_cost"]["mean"], base["actor_cost"]["mean"]
        ),
        "aggregate_relative_gain": (
            cand["aggregate_relative_gain"], base["aggregate_relative_gain"]
        ),
    }
    delta = {key: value[0] - value[1] for key, value in fields.items()}
    higher_is_better = (
        "actor_strict_win_fraction", "gain_mean", "gain_median", "gain_p05",
        "gain_worst", "aggregate_relative_gain",
    )
    per_seed_improved = {}
    for field in ("actor_strict_win_fraction", "gain_median", "gain_p05", "gain_worst"):
        count = 0
        for seed in range(3):
            candidate_seed = candidate["per_seed"][seed]
            baseline_seed = baseline["per_seed"][seed]
            if field == "actor_strict_win_fraction":
                candidate_value = candidate_seed[field]
                baseline_value = baseline_seed[field]
            else:
                key = {"gain_median": "median", "gain_p05": "p05", "gain_worst": "minimum"}[field]
                candidate_value = candidate_seed["gain_vs_warm"][key]
                baseline_value = baseline_seed["gain_vs_warm"][key]
            count += candidate_value > baseline_value
        per_seed_improved[field] = count
    if any(delta[field] >= 0 for field in higher_is_better):
        raise AssertionError("gamma=0.5 unexpectedly improved a registered pooled gate")
    analysis = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "GAMMA05_HIGH_WARM_RELATIVE_DECISIVE_FAIL_KEEP_GAMMA1",
        "scope": summary["scope"],
        "checks": {
            "warm_relative_validator_pass": True,
            "only_gamma_and_output_dir_differ": True,
            "formal_validation_sealed": summary["checks"]["formal_validation_sealed"],
            "test_sealed": summary["checks"]["test_sealed"],
        },
        "baseline_gamma1": {key: value[1] for key, value in fields.items()},
        "candidate_gamma05": {key: value[0] for key, value in fields.items()},
        "candidate_minus_baseline": delta,
        "candidate_improved_seed_count": per_seed_improved,
        "decision": {
            "objective": (
                "tempering gamma from 1.0 to 0.5 at the effective high-LR/trust "
                "operating point worsens every pooled warm-relative gate"
            ),
            "seed_stability": (
                "median/P05/worst worsen in all three seeds; win fraction improves "
                "in only one seed"
            ),
            "next": (
                "keep gamma=1; do not introduce realized warm into Actor input/loss/"
                "reward; stop gamma/tempering scans and retain warm only as an "
                "external paired evaluation comparator"
            ),
        },
    }
    (args.input_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n"
    )
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
