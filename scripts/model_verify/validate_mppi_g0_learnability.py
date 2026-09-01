#!/usr/bin/env python3
"""Independently validate the g0 learnability audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left, right = np.asarray(left, np.float64), np.asarray(right, np.float64)
    return np.sum(left * right, axis=-1) / (
        np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1) + 1e-12
    )


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    value = cosine(prediction, target)
    ratio = np.linalg.norm(prediction, axis=1) / (
        np.linalg.norm(target, axis=1) + 1e-12
    )
    return {
        "cosine_median": float(np.median(value)),
        "cosine_p10": float(np.quantile(value, 0.10)),
        "positive_fraction": float(np.mean(value > 0.0)),
        "norm_ratio_median": float(np.median(ratio)),
    }


def maximum_error(actual: dict[str, float], expected: dict[str, float]) -> float:
    return max(abs(actual[key] - expected[key]) for key in actual)


def main() -> None:
    args = parse_args()
    analysis_path = args.output_dir / "analysis.json"
    arrays_path = args.output_dir / "g0_learnability_audit.npz"
    manifest_path = args.output_dir / "g0_priority_manifest.json"
    analysis = json.loads(analysis_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    hash_errors = {}
    for name, path_text in analysis["sources"].items():
        if name.endswith("_sha256"):
            continue
        path = Path(path_text)
        hash_errors[name] = (
            sha256_file(path) != analysis["sources"][f"{name}_sha256"]
        )
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    target = arrays["heldout_target"]
    hard = arrays["current_critic_hard"].astype(bool)
    metric_errors = {}
    recomputed = {}
    for space, key in (
        ("physical_only", "physical_only_prediction"),
        ("physical_plus_absolute_action", "action_aware_prediction"),
    ):
        actual = metrics(arrays[key], target)
        stored = analysis["spaces"][space]["heldout"]["all"]
        expected = {
            "cosine_median": stored["cosine"]["median"],
            "cosine_p10": stored["cosine"]["p10"],
            "positive_fraction": stored["positive_fraction"],
            "norm_ratio_median": stored["norm_ratio"]["median"],
        }
        metric_errors[space] = maximum_error(actual, expected)
        recomputed[space] = actual

    order = arrays["action_aware_neighbor_index"][:, :20]
    fit_unit = arrays["fit_gradient"] / (
        np.linalg.norm(arrays["fit_gradient"], axis=1, keepdims=True) + 1e-12
    )
    target_unit = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-12)
    neighbor = fit_unit[order]
    individual = np.sum(neighbor * target_unit[:, None], axis=2)
    oracle_best = np.max(individual, axis=1)
    coherence = np.linalg.norm(np.mean(neighbor, axis=1), axis=1)
    diagnostic_errors = {
        "oracle_best20": float(np.max(np.abs(
            oracle_best - arrays["action_aware_oracle_best20"]
        ))),
        "coherence20": float(np.max(np.abs(
            coherence - arrays["action_aware_coherence20"]
        ))),
    }
    action_cosine = cosine(arrays["action_aware_prediction"], target)
    h_bad = arrays["h_oracle_cross_negative"].astype(bool)
    routing = {
        "g0_knn_negative_count": int(np.sum(action_cosine < 0.0)),
        "g0_oracle_best20_negative_count": int(np.sum(oracle_best < 0.0)),
        "g0_mixed_neighbor_count": int(np.sum(coherence < 0.30)),
        "h_oracle_cross_negative_count": int(np.sum(h_bad)),
        "joint_g0_knn_and_h_oracle_negative_count": int(np.sum(
            (action_cosine < 0.0) & h_bad
        )),
        "current_critic_hard_and_g0_knn_negative_count": int(np.sum(
            hard & (action_cosine < 0.0)
        )),
        "current_critic_hard_but_g0_oracle_best20_positive_count": int(np.sum(
            hard & (oracle_best >= 0.0)
        )),
    }
    routing_error = routing != analysis["collection_priority_routing"]
    count_errors = {
        "heldout": len(target) != analysis["counts"]["heldout_context"],
        "hard": int(np.sum(hard)) != analysis["counts"]["current_critic_hard_context"],
        "manifest": len(manifest["rows"]) != len(target),
        "manifest_source": manifest["source_analysis"] != str(analysis_path.resolve()),
    }
    expected_qualification = (
        "G0_KNN_HELDOUT_TAIL_PASS_EXISTING_DATA_LEARNABLE"
        if recomputed["physical_plus_absolute_action"]["cosine_p10"] >= 0.0
        else (
            "G0_INFORMATION_PRESENT_BUT_FIXED_NEIGHBOR_RULE_FAILS_TAIL"
            if float(np.quantile(oracle_best, 0.10)) >= 0.0
            else "G0_LOCAL_LABEL_COVERAGE_INSUFFICIENT"
        )
    )
    qualification_error = expected_qualification != analysis["qualification"]
    contract = analysis["contract"]
    passed = (
        not any(hash_errors.values())
        and not any(count_errors.values())
        and max(metric_errors.values()) <= 1e-7
        and max(diagnostic_errors.values()) <= 1e-6
        and not routing_error
        and not qualification_error
        and contract["new_dbm_rollouts"] == 0
        and contract["actor_frozen"]
        and not contract["formal_validation_loaded"]
        and not contract["test_loaded"]
        and contract["oracle_best_uses_target_label"]
        and not contract["oracle_best_is_deployable"]
    )
    result = {
        "format_version": 1,
        "qualification": "PASS" if passed else "FAIL",
        "analysis": str(analysis_path.resolve()),
        "analysis_sha256": sha256_file(analysis_path),
        "arrays": str(arrays_path.resolve()),
        "arrays_sha256": sha256_file(arrays_path),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "hash_errors": hash_errors,
        "count_errors": count_errors,
        "metric_max_abs_errors": metric_errors,
        "diagnostic_max_abs_errors": diagnostic_errors,
        "routing_error": routing_error,
        "qualification_error": qualification_error,
        "recomputed_qualification": expected_qualification,
    }
    path = args.output_dir / "validation_summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
