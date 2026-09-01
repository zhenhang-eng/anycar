#!/usr/bin/env python3
"""Independently validate a chord-geometry/local-H oracle artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

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


def subset_metrics(
    left: np.ndarray,
    right: np.ndarray,
    true_delta: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    left, right = left[mask], right[mask]
    true_delta, prediction = true_delta[mask], prediction[mask]
    cross = np.concatenate((
        cosine(left + prediction, right),
        cosine(right - prediction, left),
    ))
    reversal = cosine(left, right) < 0.0
    predicted_response = np.concatenate((
        cosine(left, left + prediction),
        cosine(right, right - prediction),
    ))
    reversal = np.concatenate((reversal, reversal))
    delta_ratio = np.linalg.norm(prediction, axis=1) / (
        np.linalg.norm(true_delta, axis=1) + 1e-12
    )
    return {
        "cross_median": float(np.median(cross)),
        "cross_p10": float(np.quantile(cross, 0.10)),
        "reversal_recall": float(np.mean(predicted_response[reversal] < 0.0)),
        "delta_cosine_median": float(np.median(cosine(prediction, true_delta))),
        "delta_norm_ratio_median": float(np.median(delta_ratio)),
    }


def maximum_error(actual: dict[str, float], expected: dict[str, float]) -> float:
    return max(abs(actual[key] - expected[key]) for key in actual)


def main() -> None:
    args = parse_args()
    analysis_path = args.output_dir / "analysis.json"
    arrays_path = args.output_dir / "local_oracle_audit.npz"
    analysis = json.loads(analysis_path.read_text())
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
    count_error = {
        "heldout_pair": len(arrays["heldout_context"])
        != analysis["counts"]["heldout_pair"],
        "action_dimension": arrays["heldout_delta_action"].shape[1:] != (16,),
        "primary_neighbor_count": arrays["primary_neighbor_index"].shape[1]
        != analysis["local_geometry"]["thresholds"]["primary_neighbor_count"],
    }
    left = arrays["heldout_left_gradient"]
    right = arrays["heldout_right_gradient"]
    true_delta = arrays["heldout_delta_gradient"]
    distance = arrays["heldout_distance_sigma"]
    metric_errors: dict[str, float] = {}
    recomputed: dict[str, Any] = {}
    for method in ("diagonal", "symmetric", "unconstrained"):
        prediction = arrays[f"{method}_predicted_delta"]
        actual = subset_metrics(
            left, right, true_delta, prediction, np.ones(len(left), bool)
        )
        small = subset_metrics(
            left, right, true_delta, prediction, distance <= 0.15
        )
        stored = analysis["oracle"]["selected_and_heldout"][method][
            "heldout_fresh_fd"
        ]
        expected = {
            "cross_median": stored["cross_target_cosine"]["median"],
            "cross_p10": stored["cross_target_cosine"]["p10"],
            "reversal_recall": stored["true_reversal_flip_recall"],
            "delta_cosine_median": stored["delta_gradient_cosine"]["median"],
            "delta_norm_ratio_median": stored["delta_gradient_norm_ratio"]["median"],
        }
        stored_small = stored["by_chord_distance"]["small_le_0_15"]
        expected_small = {
            "cross_median": stored_small["cross_target_cosine"]["median"],
            "cross_p10": stored_small["cross_target_cosine"]["p10"],
            "reversal_recall": stored_small["true_reversal_flip_recall"],
            "delta_cosine_median": stored_small["delta_gradient_cosine"]["median"],
            "delta_norm_ratio_median": stored_small["delta_gradient_norm_ratio"]["median"],
        }
        metric_errors[f"{method}_all"] = maximum_error(actual, expected)
        metric_errors[f"{method}_small"] = maximum_error(small, expected_small)
        recomputed[method] = {"all": actual, "small": small}

    neighbor = arrays["primary_neighbor_index"]
    bank = arrays["fit_delta_action"]
    smallest_ratio, rank_5pct = [], []
    for row in neighbor:
        action = bank[row]
        action = action / (np.linalg.norm(action, axis=1, keepdims=True) + 1e-12)
        singular = np.linalg.svd(action, compute_uv=False)
        ratio = singular / singular[0]
        smallest_ratio.append(float(ratio[-1]))
        rank_5pct.append(int(np.sum(ratio > 0.05)))
    stored_geometry = analysis["local_geometry"]["heldout"][str(
        analysis["local_geometry"]["thresholds"]["primary_neighbor_count"]
    )]
    geometry_errors = {
        "smallest_ratio_median": abs(
            float(np.median(smallest_ratio))
            - stored_geometry["smallest_to_largest_singular_ratio"]["median"]
        ),
        "rank_5pct_median": abs(
            float(np.median(rank_5pct))
            - stored_geometry["rank_singular_ratio_gt_0_05"]["median"]
        ),
    }
    small_symmetric = recomputed["symmetric"]["small"]
    gate = {
        "median_ge_0_90": small_symmetric["cross_median"] >= 0.90,
        "p10_ge_0": small_symmetric["cross_p10"] >= 0.0,
        "reversal_recall_ge_0_50": small_symmetric["reversal_recall"] >= 0.50,
    }
    stored_gate = analysis["oracle"]["symmetric_oracle_gate"]
    gate_error = any(gate[key] != stored_gate[key] for key in gate)
    passed = (
        not any(hash_errors.values())
        and not any(count_error.values())
        and max(metric_errors.values()) <= 1e-7
        and max(geometry_errors.values()) <= 1e-7
        and not gate_error
        and analysis["contract"]["new_dbm_rollouts"] == 0
        and analysis["contract"]["actor_frozen"]
        and not analysis["contract"]["formal_validation_loaded"]
        and not analysis["contract"]["test_loaded"]
    )
    result = {
        "format_version": 1,
        "qualification": "PASS" if passed else "FAIL",
        "analysis": str(analysis_path.resolve()),
        "analysis_sha256": sha256_file(analysis_path),
        "arrays": str(arrays_path.resolve()),
        "arrays_sha256": sha256_file(arrays_path),
        "hash_errors": hash_errors,
        "count_errors": count_error,
        "metric_max_abs_errors": metric_errors,
        "geometry_abs_errors": geometry_errors,
        "gate_recomputed": gate,
        "gate_error": gate_error,
    }
    path = args.output_dir / "validation_summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
