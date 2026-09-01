#!/usr/bin/env python3
"""Independently recompute the full-fold OAC-2 outer-heldout analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_mppi_oac2_full_folds import run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path, nargs="?",
        default=Path(
            "outputs/mppi_proposal/online_absolute_sac_oac2_full_3fold_20260824_v1"
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def max_metric_error(left: dict, right: dict) -> float:
    errors = []
    for left_fold, right_fold in zip(left["folds"], right["folds"]):
        for left_seed, right_seed in zip(left_fold["seeds"], right_fold["seeds"]):
            for role in ("initial_metrics", "selected_metrics"):
                for group, key in (
                    ("cost", "mean"), ("gain_vs_initial", "mean"),
                    ("gain_vs_initial", "p05"), ("gain_vs_initial", "median"),
                    ("gain_vs_initial", "minimum"),
                ):
                    errors.append(abs(
                        float(left_seed[role][group][key])
                        - float(right_seed[role][group][key])
                    ))
            errors.append(abs(
                float(left_seed["selected_metrics"]["headroom_recovery_vs_bank_best"])
                - float(right_seed["selected_metrics"]["headroom_recovery_vs_bank_best"])
            ))
    return max(errors)


def main() -> None:
    cli = parse_args()
    stored = json.loads((cli.run_dir / "analysis.json").read_text())
    args = argparse.Namespace(
        run_dirs=[Path(row["run_dir"]) for row in stored["folds"]],
        output_dir=cli.run_dir,
        bootstrap_samples=2000,
        device=cli.device,
    )
    recomputed = run_analysis(args)
    error = max_metric_error(recomputed, stored)
    source_hashes = all(
        row["contract_sha256"] == recomputed_row["contract_sha256"]
        and row["summary_sha256"] == recomputed_row["summary_sha256"]
        and row["validator_sha256"] == recomputed_row["validator_sha256"]
        for row, recomputed_row in zip(stored["folds"], recomputed["folds"])
    )
    checks = {
        "metric_max_abs_error_le_1e_7": error <= 1e-7,
        "qualification_match": recomputed["qualification"] == stored["qualification"],
        "passed_fold_count_match": recomputed["passed_fold_count"] == stored["passed_fold_count"],
        "source_hashes_match": source_hashes,
        "all_source_validators_passed": recomputed["all_source_validators_passed"],
        "formal_validation_sealed": not recomputed["formal_validation_loaded"],
        "test_sealed": not recomputed["test_loaded"],
    }
    passed = bool(all(checks.values()))
    report = {
        "qualification": (
            "OAC3_OUTER_HELDOUT_VALIDATION_PASS"
            if passed else "OAC3_OUTER_HELDOUT_VALIDATION_FAIL"
        ),
        "passed": passed,
        "metric_max_abs_error": error,
        "checks": checks,
    }
    (cli.run_dir / "validator_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
