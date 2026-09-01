#!/usr/bin/env python3
"""Summarize the consumed-split local-Critic retraining ablation.

This script is read-only with respect to models and datasets.  It combines the
recorded training summaries and, where available, independently replayed fresh-FD
audits into one compact manifest.  It does not load formal validation/test data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_local_critic_retraining_ablation_20260813_v1"
)


EXPERIMENTS = (
    (
        "A0_original",
        Path("outputs/mppi_proposal/direct_local_gradient_critic_20260812_v1"),
        Path("outputs/mppi_proposal/direct_critic_fresh_fd_20260813_v2"),
        "old selection; combined 0.05/0.10/0.20 gradient target",
    ),
    (
        "B1_norm_selection",
        Path("outputs/mppi_proposal/direct_local_gradient_critic_b1_norm_selection_20260813_v1"),
        Path("outputs/mppi_proposal/direct_critic_fresh_fd_b1_norm_selection_20260813_v1"),
        "norm-aware score only; combined target",
    ),
    (
        "B2_magnitude_schedule",
        Path("outputs/mppi_proposal/direct_local_gradient_critic_b2_magnitude_schedule_20260813_v2"),
        None,
        "log-norm warmup plus delayed/reduced cosine; combined target",
    ),
    (
        "B3_bank_off",
        Path("outputs/mppi_proposal/direct_local_gradient_critic_b3_bank_off_20260813_v1"),
        None,
        "bank reconstruction disabled; otherwise B1",
    ),
    (
        "B4_smallest_target",
        Path("outputs/mppi_proposal/direct_local_gradient_critic_b4_smallest_target_20260813_v2"),
        Path("outputs/mppi_proposal/direct_critic_fresh_fd_b4_smallest_target_20260813_v2"),
        "0.05-sigma derivative target plus hard norm eligibility gate",
    ),
    (
        "B5_small_pair",
        Path("outputs/mppi_proposal/direct_local_gradient_critic_b5_small_pair_20260813_v1"),
        Path("outputs/mppi_proposal/direct_critic_fresh_fd_b5_small_pair_20260813_v1"),
        "B4 plus smallest-radius bank and pair delta/ranking",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_metrics(summary: dict[str, Any], key: str) -> dict[str, float]:
    metrics = summary[key]
    names = (
        "gradient_cosine_median",
        "gradient_cosine_p10",
        "gradient_cosine_positive_fraction",
        "gradient_norm_ratio_median",
        "gradient_norm_correlation",
        "probe_delta_sign_accuracy",
        "probe_mean_argmax_regret",
    )
    return {name: float(metrics[name]) for name in names if name in metrics}


def fresh_metrics(summary: dict[str, Any]) -> dict[str, float]:
    metrics = summary["critic_vs_fresh_fd"]["ensemble_mean"]
    return {
        "gradient_cosine_median": float(metrics["cosine"]["median"]),
        "gradient_cosine_p10": float(metrics["cosine"]["p10"]),
        "gradient_cosine_positive_fraction": float(
            metrics["cosine_positive_fraction"]
        ),
        "gradient_correlation": float(metrics["correlation"]),
        "predicted_norm_median": float(metrics["prediction_norm"]["median"]),
        "target_norm_median": float(metrics["target_norm"]["median"]),
        "gradient_norm_ratio_median": float(metrics["norm_ratio"]["median"]),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    rows = []
    for name, training_dir, fresh_dir, change in EXPERIMENTS:
        training_path = training_dir / "summary.json"
        training = json.loads(training_path.read_text())
        row: dict[str, Any] = {
            "name": name,
            "change": change,
            "training_summary": str(training_path.resolve()),
            "training_summary_sha256": sha256(training_path),
            "qualification": training["qualification"],
            "training": selected_metrics(training, "train_metrics"),
            "internal_validation": selected_metrics(
                training, "internal_validation_metrics"
            ),
            "internal_selection": selected_metrics(training, "heldout_metrics"),
            "best_epochs": [int(value["best_epoch"]) for value in training["training"]],
        }
        if fresh_dir is not None:
            fresh_path = fresh_dir / "summary.json"
            validation_path = fresh_dir / "validation_summary.json"
            fresh = json.loads(fresh_path.read_text())
            validation = json.loads(validation_path.read_text())
            if validation["qualification"] != "PASS":
                raise AssertionError(f"fresh audit validation failed: {fresh_dir}")
            row.update({
                "fresh_fd_summary": str(fresh_path.resolve()),
                "fresh_fd_summary_sha256": sha256(fresh_path),
                "fresh_fd_validation": str(validation_path.resolve()),
                "fresh_fd_validation_sha256": sha256(validation_path),
                "fresh_fd_validation_qualification": validation["qualification"],
                "fresh_fd": fresh_metrics(fresh),
            })
        rows.append(row)

    by_name = {row["name"]: row for row in rows}
    original = by_name["A0_original"]["fresh_fd"]
    b1 = by_name["B1_norm_selection"]["fresh_fd"]
    b4 = by_name["B4_smallest_target"]["fresh_fd"]
    b5 = by_name["B5_small_pair"]["fresh_fd"]
    conclusions = {
        "checkpoint_selection_exposed_hidden_amplitude_learning": (
            b1["gradient_norm_ratio_median"]
            > 20.0 * original["gradient_norm_ratio_median"]
        ),
        "smallest_radius_target_improved_median_direction": (
            b4["gradient_cosine_median"]
            > b1["gradient_cosine_median"] + 0.15
        ),
        "smallest_radius_target_fixed_negative_tail": (
            b4["gradient_cosine_p10"] >= 0.0
        ),
        "pair_losses_improved_fresh_median_direction": (
            b5["gradient_cosine_median"]
            > b4["gradient_cosine_median"] + 0.02
        ),
        "actor_update_authorized": (
            b5["gradient_cosine_median"] >= 0.70
            and b5["gradient_cosine_p10"] >= 0.0
            and b5["gradient_norm_ratio_median"] >= 0.50
        ),
    }
    if conclusions["actor_update_authorized"]:
        qualification = "CRITIC_RETRAINING_GATE_PASS"
    else:
        qualification = "AMPLITUDE_AND_MEDIAN_IMPROVED_NEGATIVE_TAIL_FAIL"
    output = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "scope": "consumed train/internal-validation/internal-selection only",
        "formal_validation_loaded": False,
        "test_loaded": False,
        "actor_updated": False,
        "experiments": rows,
        "conclusions": conclusions,
        "recommended_next_step": (
            "Collect same-state small-radius local perturbations on more diverse "
            "internal-fit episodes with episode/speed-balanced training; preserve "
            "the 0.05-sigma derivative target and norm eligibility gate. Do not "
            "update the Actor until fresh-FD median>=0.70, P10>=0 and norm ratio>=0.50."
        ),
    }
    path = args.output_dir / "analysis.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Local-Critic retraining ablation\n\n"
        f"Qualification: `{qualification}`. Actor remained frozen.\n\n"
        "Checkpoint selection and the 0.05-sigma derivative target restored "
        "gradient amplitude and median direction, but fresh-FD P10 remained "
        "negative. Pair losses did not repair the tail.\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
