#!/usr/bin/env python3
"""Independently validate the zero-rollout canonical tie-break diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / (
    "outputs/query_mppi/query_noanchor_canonical_tiebreak_diagnostic_20260901_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def robust(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    middle = np.median(values, axis=0)
    lower, upper = np.quantile(values, (0.25, 0.75), axis=0)
    scale = upper - lower
    standard = np.std(values, axis=0)
    fallback = np.where(standard > 1e-8, standard, 1.0)
    return middle, np.where(scale > 1e-8, scale, fallback)


def normalized(blocks: list[np.ndarray], rows: np.ndarray, fit: np.ndarray) -> np.ndarray:
    output = []
    for block in blocks:
        center, scale = robust(block[fit])
        output.append((block[rows] - center) / scale / math.sqrt(block.shape[1]))
    return np.concatenate(output, axis=1)


def euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    value = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None]
        - 2.0 * left @ right.T
    )
    return np.sqrt(np.maximum(value, 0.0))


def target_metrics(
    data: dict[str, np.ndarray], target: np.ndarray, sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(target)
    blocks = [
        data["history"].reshape(count, -1).astype(np.float64),
        data["reference_ego"].reshape(count, -1).astype(np.float64),
        np.concatenate((data["state"][:, 3:5], data["current_action"]), axis=1)
        .astype(np.float64),
    ]
    scaled = target.reshape(count, -1).astype(np.float64) / sigma
    folds = data["fold_id"].astype(np.int64)
    nearest = np.empty(count, np.int64)
    nearest_distance = np.empty(count, np.float64)
    rank = np.empty(count, np.float64)
    for fold in range(5):
        query = np.flatnonzero(folds == fold)
        fit = np.flatnonzero((folds != fold) & (folds != (fold + 1) % 5))
        input_distance = euclidean(
            normalized(blocks, query, fit), normalized(blocks, fit, fit)
        )
        target_distance = np.sqrt(
            np.mean(np.square(scaled[query, None] - scaled[fit][None]), axis=2)
        )
        local_nearest = np.argmin(input_distance, axis=1)
        nearest[query] = fit[local_nearest]
        nearest_distance[query] = target_distance[
            np.arange(len(query)), local_nearest
        ]
        for local, row in enumerate(query):
            rank[row] = spearmanr(
                input_distance[local], target_distance[local]
            ).statistic
    return nearest, nearest_distance, rank


def max_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    analysis = json.loads((source / "analysis.json").read_text())
    diagnostics_path = source / "diagnostics.npz"
    audit = Path(manifest["source"])
    checks = {
        "analysis_hash": sha256(source / "analysis.json") == manifest["analysis_sha256"],
        "diagnostics_hash": sha256(diagnostics_path) == manifest["diagnostics_sha256"],
        "source_manifest_hash": sha256(audit / "manifest.json") == manifest["source_manifest_sha256"],
        "source_summary_hash": sha256(audit / "summary.json") == manifest["source_summary_sha256"],
        "source_validation_hash": sha256(audit / "validation.json") == manifest["source_validation_sha256"],
        "source_audit_hash": sha256(audit / "audit.npz") == manifest["source_audit_sha256"],
        "zero_rollout_contract": (
            manifest["new_query_rollouts"] == 0
            and manifest["new_dbm_rollouts"] == 0
            and not manifest["formal_validation_or_test_consumed"]
            and not manifest["dbm_fields_or_labels_consumed"]
        ),
    }
    with np.load(audit / "audit.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(diagnostics_path, allow_pickle=False) as archive:
        saved = {name: np.asarray(archive[name]) for name in archive.files}
    config = json.loads((audit / "config.json").read_text())
    count, candidate_count = data["candidate_cost"].shape
    sigma = np.tile(np.asarray(config["noise_sigma"], np.float64), 8)
    candidates = data["candidate_knots"].reshape(count, candidate_count, -1)
    hold = data["canonical_hold_knots"].reshape(count, -1)
    distance_from_hold = np.sqrt(
        np.mean(np.square((candidates - hold[:, None]) / sigma), axis=2)
    )
    candidate_cost = data["candidate_cost"].astype(np.float64)
    best = np.min(candidate_cost, axis=1)
    tolerances = saved["relative_cost_tolerances"]
    reconstructed_index = np.empty_like(saved["selection_index"])
    reconstructed_cost = np.empty_like(saved["selected_cost"])
    reconstructed_knots = np.empty_like(saved["selected_knots"])
    reconstructed_nn = np.empty_like(saved["selected_nearest_target_sigma_rms"])
    reconstructed_rank = np.empty_like(saved["selected_input_target_spearman"])
    nearest_reference = None
    for tolerance_index, tolerance in enumerate(tolerances):
        eligible = candidate_cost <= best[:, None] * (1.0 + tolerance) + 1e-10
        index = np.argmin(np.where(eligible, distance_from_hold, np.inf), axis=1)
        target = data["candidate_knots"][np.arange(count), index]
        nearest, nn_distance, rank = target_metrics(data, target, sigma)
        if nearest_reference is None:
            nearest_reference = nearest
        elif not np.array_equal(nearest_reference, nearest):
            raise AssertionError("strict input nearest row changed with target")
        reconstructed_index[tolerance_index] = index
        reconstructed_cost[tolerance_index] = candidate_cost[np.arange(count), index]
        reconstructed_knots[tolerance_index] = target
        reconstructed_nn[tolerance_index] = nn_distance
        reconstructed_rank[tolerance_index] = rank
    old_nearest, old_nn, old_rank = target_metrics(
        data, data["old_fullrank_knots"], sigma
    )
    errors = {
        "nearest_fit_row": max_error(nearest_reference, saved["nearest_fit_row"]),
        "old_nearest_row": max_error(old_nearest, saved["nearest_fit_row"]),
        "old_fullrank_nearest_target": max_error(
            old_nn, saved["old_fullrank_nearest_target_sigma_rms"]
        ),
        "old_fullrank_spearman": max_error(
            old_rank, saved["old_fullrank_input_target_spearman"]
        ),
        "selection_index": max_error(reconstructed_index, saved["selection_index"]),
        "selected_cost": max_error(reconstructed_cost, saved["selected_cost"]),
        "selected_knots": max_error(reconstructed_knots, saved["selected_knots"]),
        "selected_nearest_target": max_error(
            reconstructed_nn, saved["selected_nearest_target_sigma_rms"]
        ),
        "selected_spearman": max_error(
            reconstructed_rank, saved["selected_input_target_spearman"]
        ),
    }
    checks["diagnostic_array_reconstruction"] = max(errors.values()) < 1e-9
    feasible = [
        record["relative_cost_tolerance"]
        for record in analysis["records"]
        if record["all_cost_and_coherence_gates_pass"]
    ]
    reconstructed_decision = (
        "CANONICAL_BANK_TIEBREAK_FEASIBLE_DIAGNOSTIC_ONLY"
        if feasible
        else "NO_TIEBREAK_WITHIN_CANONICAL_BANK_MEETS_COST_AND_COHERENCE"
    )
    checks["decision_reconstruction"] = (
        feasible == analysis["feasible_relative_cost_tolerances"]
        and reconstructed_decision == analysis["decision"] == manifest["decision"]
    )
    qualification = (
        "QUERY_CANONICAL_TIEBREAK_INDEPENDENT_PASS"
        if all(checks.values())
        else "QUERY_CANONICAL_TIEBREAK_INDEPENDENT_FAIL"
    )
    validation = {
        "qualification": qualification,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "reconstruction_max_errors": errors,
        "decision": reconstructed_decision,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256(Path(__file__).resolve()),
    }
    dump_json(source / "validation.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))
    if qualification.endswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
