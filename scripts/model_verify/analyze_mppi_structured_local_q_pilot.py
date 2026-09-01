#!/usr/bin/env python3
"""Audit a structured local-Q pilot summary and compact the 3-seed result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any


METRICS = {
    "fresh_cosine_median": ("fresh_fd", "gradient_cosine_median"),
    "fresh_cosine_p10": ("fresh_fd", "gradient_cosine_p10"),
    "fresh_norm_ratio_median": ("fresh_fd", "gradient_norm_ratio_median"),
    "fresh_norm_ratio_p90": ("fresh_fd", "gradient_norm_ratio_p90"),
    "small_chord_flip_recall": (
        "fresh_fd", "chord_bins", "small_le_0_15", "reversal_flip_recall"
    ),
    "all_chord_flip_recall": ("fresh_fd", "true_reversal_flip_recall"),
    "same_center_p10": (
        "fresh_fd", "same_center_nuisance_cosine", "p10"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested(record: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = record
    for key in path:
        value = value[key]
    return float(value)


def expected_gates(record: dict[str, Any]) -> dict[str, bool]:
    fresh = record["fresh_fd"]
    return {
        "fresh_cosine_median_ge_0_70": (
            fresh["gradient_cosine_median"] >= 0.70
        ),
        "fresh_cosine_p10_ge_0": fresh["gradient_cosine_p10"] >= 0.0,
        "fresh_norm_median_ge_0_50": (
            fresh["gradient_norm_ratio_median"] >= 0.50
        ),
        "fresh_norm_median_le_2": (
            fresh["gradient_norm_ratio_median"] <= 2.0
        ),
        "small_chord_flip_recall_ge_0_50": (
            fresh["chord_bins"]["small_le_0_15"]["reversal_flip_recall"]
            >= 0.50
        ),
        "same_center_p10_ge_0": (
            fresh["same_center_nuisance_cosine"]["p10"] >= 0.0
        ),
        "symmetry_error_le_1e_6": (
            record["heldout_0_05_labels"][
                "hessian_symmetry_max_abs_error"
            ] <= 1e-6
        ),
    }


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "median": median(values),
        "maximum": max(values),
    }


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    hash_checks: dict[str, Any] = {}
    for name, source in summary["sources"].items():
        if name.endswith("_sha256"):
            continue
        if source is None:
            continue
        path = Path(source)
        expected = summary["sources"][f"{name}_sha256"]
        actual = sha256_file(path)
        hash_checks[name] = {
            "path": str(path), "expected": expected, "actual": actual,
            "match": actual == expected,
        }
    for checkpoint in summary["checkpoints"]:
        path = Path(checkpoint)
        records = [
            record
            for arm in summary["arms"].values()
            for record in arm["records"]
            if record["checkpoint"] == checkpoint
        ]
        if len(records) != 1:
            raise ValueError(f"checkpoint record count != 1: {checkpoint}")
        expected = records[0]["checkpoint_sha256"]
        actual = sha256_file(path)
        hash_checks[f"checkpoint:{path.name}"] = {
            "path": str(path), "expected": expected, "actual": actual,
            "match": actual == expected,
        }

    gate_mismatches = []
    arm_result: dict[str, Any] = {}
    h0_by_seed = {
        int(record["seed"]): record
        for record in summary.get("arms", {}).get("H0", {}).get("records", [])
    }
    for arm, payload in summary["arms"].items():
        records = payload["records"]
        complete = 0
        for record in records:
            expected = expected_gates(record)
            if expected != record["gates"]:
                gate_mismatches.append({
                    "arm": arm, "seed": record["seed"],
                    "stored": record["gates"], "recomputed": expected,
                })
            passed = all(expected.values())
            if passed != bool(record["all_gates_passed"]):
                gate_mismatches.append({
                    "arm": arm, "seed": record["seed"],
                    "stored_all": record["all_gates_passed"],
                    "recomputed_all": passed,
                })
            complete += int(passed)
        if complete != int(payload["complete_gate_pass_count"]):
            gate_mismatches.append({
                "arm": arm, "stored_pass_count": payload[
                    "complete_gate_pass_count"
                ], "recomputed_pass_count": complete,
            })
        metrics = {
            name: distribution([nested(record, path) for record in records])
            for name, path in METRICS.items()
        }
        paired_h0_delta = {}
        if arm != "H0" and h0_by_seed:
            for name, path in METRICS.items():
                delta = [
                    nested(record, path)
                    - nested(h0_by_seed[int(record["seed"])], path)
                    for record in records
                ]
                paired_h0_delta[name] = distribution(delta)
        arm_result[arm] = {
            "complete_gate_pass_count": complete,
            "metrics_across_seed": metrics,
            "paired_delta_vs_same_seed_H0": paired_h0_delta,
        }

    passed = any(
        row["complete_gate_pass_count"] >= 2 for row in arm_result.values()
    )
    targeted_run = bool(
        summary.get("contract", {}).get("targeted_response_training", False)
    )
    if targeted_run:
        recomputed_qualification = (
            "TARGETED_STRUCTURED_LOCAL_Q_MECHANISM_GATE_PASS_ACTOR_STILL_FROZEN"
            if passed else "TARGETED_STRUCTURED_LOCAL_Q_CRITIC_FAIL_ACTOR_FROZEN"
        )
    else:
        recomputed_qualification = (
            "STRUCTURED_LOCAL_Q_MECHANISM_GATE_PASS_ACTOR_STILL_FROZEN"
            if passed else "STRUCTURED_LOCAL_Q_CRITIC_FAIL_ACTOR_FROZEN"
        )
    analysis = {
        "format_version": 1,
        "source_summary": str(args.summary.resolve()),
        "source_summary_sha256": sha256_file(args.summary),
        "hash_checks": hash_checks,
        "gate_mismatches": gate_mismatches,
        "stored_qualification": summary["qualification"],
        "recomputed_qualification": recomputed_qualification,
        "qualification_match": recomputed_qualification == summary["qualification"],
        "actor_update_performed": summary["actor_update_performed"],
        "paired_bootstrap": {
            "status": "NOT_TRIGGERED",
            "reason": "no arm reached the prerequisite 2/3 complete per-seed gates",
        },
        "arms": arm_result,
    }
    if not all(row["match"] for row in hash_checks.values()):
        raise RuntimeError("source or checkpoint SHA256 mismatch")
    if gate_mismatches or not analysis["qualification_match"]:
        raise RuntimeError("stored gate result does not reproduce")
    if summary["actor_update_performed"]:
        raise RuntimeError("pilot contract violation: Actor was updated")
    output = args.output or args.summary.with_name("analysis.json")
    output.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({
        "output": str(output.resolve()),
        "qualification": recomputed_qualification,
        "gate_mismatch_count": len(gate_mismatches),
        "all_hashes_match": True,
        "paired_bootstrap": "NOT_TRIGGERED",
    }, indent=2))


if __name__ == "__main__":
    main()
