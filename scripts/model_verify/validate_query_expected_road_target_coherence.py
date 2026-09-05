#!/usr/bin/env python3
"""Independently validate the Query target/coherence diagnostic artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = REPO_ROOT / (
    "outputs/query_mppi/query_expected_road_target_coherence_20260901_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, nargs="?", default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    quarter = np.quantile(values, (0.25, 0.75), axis=0)
    scale = quarter[1] - quarter[0]
    standard = np.std(values, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(standard > 1e-8, standard, 1.0))
    return center, scale


def embed(blocks: list[np.ndarray], rows: np.ndarray, fit: np.ndarray) -> np.ndarray:
    pieces = []
    for block in blocks:
        center, scale = robust_scale(block[fit])
        pieces.append(
            ((block[rows] - center) / scale) / np.sqrt(block.shape[1])
        )
    return np.concatenate(pieces, axis=1).astype(np.float64)


def distances(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(query * query, axis=1)[:, None]
        + np.sum(bank * bank, axis=1)[None, :]
        - 2.0 * query @ bank.T
    )
    return np.sqrt(np.maximum(squared, 0.0))


def recompute_neighbors(
    data: dict[str, np.ndarray], include_warm: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(data["state"])
    order = np.empty((count, 360), np.int64)
    nearest_distance = np.empty(count, np.float64)
    correlations = np.empty(count, np.float64)
    sigma = np.tile(np.asarray((0.25, 0.35), np.float64), 8)
    target = data["fullrank_teacher_knots"].reshape(count, -1) / sigma
    blocks = [
        data["history"].reshape(count, -1),
        data["reference_ego"].reshape(count, -1),
        np.concatenate((data["state"][:, 3:5], data["current_action"]), axis=1),
    ]
    if include_warm:
        blocks.append(data["mean_knots_before"].reshape(count, -1))
    folds = np.asarray(data["fold_id"], np.int64)
    for fold in range(5):
        query = np.flatnonzero(folds == fold)
        bank = np.flatnonzero((folds != fold) & (folds != (fold + 1) % 5))
        input_distance = distances(embed(blocks, query, bank), embed(blocks, bank, bank))
        local_order = np.argsort(input_distance, axis=1)
        order[query] = bank[local_order]
        nearest_distance[query] = input_distance[np.arange(len(query)), local_order[:, 0]]
        target_distance = np.sqrt(
            np.mean(np.square(target[query, None] - target[bank][None]), axis=2)
        )
        for local, row in enumerate(query):
            correlations[row] = float(
                spearmanr(input_distance[local], target_distance[local]).statistic
            )
    return order, nearest_distance, correlations


def headline_recompute(
    data: dict[str, np.ndarray], strict_neighbor: np.ndarray
) -> dict[str, float]:
    count = len(strict_neighbor)
    sigma = np.tile(np.asarray((0.25, 0.35), np.float64), 8)
    target = data["fullrank_teacher_knots"].reshape(count, -1) / sigma
    hard_distance = np.sqrt(
        np.mean(np.square(target - target[strict_neighbor]), axis=1)
    )
    cost = np.asarray(data["candidate_cost"], np.float64)
    best = np.min(cost, axis=1)
    knots = data["candidate_knots"].reshape(count, 132, -1) / sigma
    set_minimum = []
    for row, other in enumerate(strict_neighbor):
        current = np.flatnonzero(cost[row] <= best[row] * 1.02 + 1e-10)
        neighbor = np.flatnonzero(cost[other] <= best[other] * 1.02 + 1e-10)
        current_knots = knots[row][current]
        neighbor_knots = knots[other][neighbor]
        pair_distance = np.sqrt(
            np.mean(
                np.square(current_knots[:, None] - neighbor_knots[None]),
                axis=2,
            )
        )
        set_minimum.append(float(np.min(pair_distance)))
    pair_cost = np.asarray(data["pair_cost"], np.float64)
    sign = np.argmin(pair_cost, axis=3)
    margin = np.abs(pair_cost[..., 0] - pair_cost[..., 1]) / np.maximum(
        np.minimum(pair_cost[..., 0], pair_cost[..., 1]), 1.0
    )
    eligible = (margin >= 0.05) & (margin[strict_neighbor] >= 0.05)
    return {
        "hard": float(np.median(hard_distance)),
        "elite2": float(np.median(set_minimum)),
        "robust_pair": float(np.mean((sign == sign[strict_neighbor])[eligible])),
    }


def main() -> None:
    args = parse_args()
    artifact = args.artifact.resolve()
    manifest = json.loads((artifact / "manifest.json").read_text())
    analysis = json.loads((artifact / "analysis.json").read_text())
    source = Path(manifest["source"])
    pretrain = Path(manifest["pretrain"])
    hash_checks = {
        "analysis": sha256(artifact / "analysis.json") == manifest["analysis_sha256"],
        "diagnostics": sha256(artifact / "diagnostics.npz")
        == manifest["diagnostics_sha256"],
        "source_manifest": sha256(source / "manifest.json")
        == manifest["source_manifest_sha256"],
        "source_validation": sha256(source / "validation.json")
        == manifest["source_validation_sha256"],
        "source_bank": sha256(source / "bank.npz") == manifest["source_bank_sha256"],
        "pretrain_manifest": sha256(pretrain / "manifest.json")
        == manifest["pretrain_manifest_sha256"],
        "pretrain_validation": sha256(pretrain / "validation.json")
        == manifest["pretrain_validation_sha256"],
        "pretrain_oof": sha256(pretrain / "oof_predictions.npz")
        == manifest["pretrain_oof_sha256"],
    }
    with np.load(source / "bank.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(artifact / "diagnostics.npz", allow_pickle=False) as archive:
        saved = {name: np.asarray(archive[name]) for name in archive.files}
    strict_order, strict_distance, strict_spearman = recompute_neighbors(data, False)
    warm_order, warm_distance, warm_spearman = recompute_neighbors(data, True)
    errors = {
        "strict_neighbor_index": float(
            np.max(np.abs(strict_order - saved["strict_no_anchor_neighbor_index"]))
        ),
        "strict_neighbor_distance": float(
            np.max(
                np.abs(
                    strict_distance
                    - saved["strict_no_anchor_neighbor_distance"][:, 0]
                )
            )
        ),
        "strict_spearman": float(
            np.max(
                np.abs(
                    strict_spearman
                    - saved["strict_no_anchor_input_target_spearman"]
                )
            )
        ),
        "warm_neighbor_index": float(
            np.max(
                np.abs(
                    warm_order
                    - saved["diagnostic_plus_exact_warm_neighbor_index"]
                )
            )
        ),
        "warm_neighbor_distance": float(
            np.max(
                np.abs(
                    warm_distance
                    - saved["diagnostic_plus_exact_warm_neighbor_distance"][:, 0]
                )
            )
        ),
        "warm_spearman": float(
            np.max(
                np.abs(
                    warm_spearman
                    - saved["diagnostic_plus_exact_warm_input_target_spearman"]
                )
            )
        ),
    }
    strict_neighbor = strict_order[:, 0]
    for row, neighbor in enumerate(strict_neighbor):
        fold = int(data["fold_id"][row])
        if int(data["fold_id"][neighbor]) in (fold, (fold + 1) % 5):
            raise AssertionError("neighbor leaked outer or selection fold")
        if data["episode_id"][row] == data["episode_id"][neighbor]:
            raise AssertionError("neighbor reused query episode")
    headline = headline_recompute(data, strict_neighbor)
    expected = analysis["routing_evidence"]
    headline_errors = {
        "hard": abs(
            headline["hard"]
            - expected["hard_argmin_absolute_distance_median_sigma_rms"]
        ),
        "elite2": abs(
            headline["elite2"]
            - expected["relative_2pct_set_minimum_pair_median_sigma_rms"]
        ),
        "strict_spearman_median": abs(
            float(np.median(strict_spearman))
            - expected["strict_input_target_spearman_median"]
        ),
        "warm_spearman_median": abs(
            float(np.median(warm_spearman))
            - expected["plus_warm_input_target_spearman_median"]
        ),
        "robust_pair": abs(
            headline["robust_pair"]
            - expected["robust_pair_sign_agreement_margin_ge_5pct"]
        ),
    }
    checks = {
        "artifact_hashes": all(hash_checks.values()),
        "source_and_pretrain_qualified": (
            json.loads((source / "validation.json").read_text())["qualification"]
            == "QUERY_EXPECTED_ROAD_FULLRANK_PASS"
            and json.loads((pretrain / "validation.json").read_text())["qualification"]
            == "QUERY_ABSOLUTE_PRETRAIN_INDEPENDENT_PASS"
        ),
        "formal_validation_test_sealed": (
            not manifest["formal_validation_or_test_consumed"]
            and manifest["new_query_or_dbm_rollouts"] == 0
        ),
        "nested_neighbor_reconstruction": max(errors.values()) <= 1e-5,
        "headline_reconstruction": max(headline_errors.values()) <= 1e-8,
        "decision_reconstruction": (
            expected["relative_2pct_set_distance_reduction_fraction"] < 0.10
            and expected["relative_2pct_set_minimum_pair_median_sigma_rms"] > 0.50
            and expected["robust_pair_sign_agreement_margin_ge_5pct"] < 0.60
            and analysis["decision"]
            == "NO_JUSTIFICATION_FOR_SET_VALUED_ABSOLUTE_BC_RETRY"
        ),
    }
    report = {
        "qualification": (
            "QUERY_TARGET_COHERENCE_INDEPENDENT_PASS"
            if all(checks.values())
            else "QUERY_TARGET_COHERENCE_INDEPENDENT_FAIL"
        ),
        "artifact": str(artifact),
        "checks": checks,
        "hash_checks": hash_checks,
        "maximum_errors": errors,
        "headline_errors": headline_errors,
        "recomputed_headline": headline,
        "formal_validation_or_test_consumed": False,
        "new_query_or_dbm_rollouts": 0,
    }
    (artifact / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
