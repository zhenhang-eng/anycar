#!/usr/bin/env python3
"""Zero-rollout tie-break diagnosis for the canonical Query audit bank."""

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
DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "query_noanchor_canonical_teacher_audit_20260901_v1"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "outputs/query_mppi/query_noanchor_canonical_tiebreak_diagnostic_20260901_v1"
)
RELATIVE_TOLERANCES = (0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    scale = q75 - q25
    standard = np.std(values, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(standard > 1e-8, standard, 1.0))
    return center, scale


def embed(blocks: list[np.ndarray], rows: np.ndarray, fit: np.ndarray) -> np.ndarray:
    pieces = []
    for block in blocks:
        center, scale = robust_scale(block[fit])
        pieces.append((block[rows] - center) / scale / math.sqrt(block.shape[1]))
    return np.concatenate(pieces, axis=1)


def distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None]
        - 2.0 * left @ right.T
    )
    return np.sqrt(np.maximum(squared, 0.0))


def input_contract(data: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    count = len(data["state"])
    blocks = [
        data["history"].reshape(count, -1).astype(np.float64),
        data["reference_ego"].reshape(count, -1).astype(np.float64),
        np.concatenate((data["state"][:, 3:5], data["current_action"]), axis=1)
        .astype(np.float64),
    ]
    nearest = np.empty(count, np.int64)
    fold_contract = {}
    folds = data["fold_id"].astype(np.int64)
    for fold in range(5):
        query = np.flatnonzero(folds == fold)
        fit = np.flatnonzero((folds != fold) & (folds != (fold + 1) % 5))
        input_distance = distance(embed(blocks, query, fit), embed(blocks, fit, fit))
        nearest[query] = fit[np.argmin(input_distance, axis=1)]
        fold_contract[fold] = (query, fit, input_distance)
    return nearest, fold_contract


def target_metrics(
    target: np.ndarray,
    nearest: np.ndarray,
    fold_contract: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaled = target.reshape(len(target), -1).astype(np.float64) / sigma
    nearest_distance = np.sqrt(
        np.mean(np.square(scaled - scaled[nearest]), axis=1)
    )
    spearman = np.empty(len(target), np.float64)
    for query, fit, input_distance in fold_contract.values():
        target_distance = np.sqrt(
            np.mean(
                np.square(scaled[query, None] - scaled[fit][None]), axis=2
            )
        )
        for local, row in enumerate(query):
            spearman[row] = spearmanr(
                input_distance[local], target_distance[local]
            ).statistic
    return nearest_distance, spearman


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    validation = json.loads((source / "validation.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    config = json.loads((source / "config.json").read_text())
    if validation["qualification"] != "QUERY_NOANCHOR_CANONICAL_TEACHER_AUDIT_INDEPENDENT_PASS":
        raise AssertionError("canonical audit did not independently pass")
    if manifest.get("formal_validation_or_test_consumed", True):
        raise AssertionError("source consumed sealed splits")
    if manifest.get("dbm_fields_or_labels_consumed"):
        raise AssertionError("source consumed DBM fields/labels")
    with np.load(source / "audit.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}

    count, candidate_count = data["candidate_cost"].shape
    sigma = np.tile(np.asarray(config["noise_sigma"], np.float64), 8)
    candidates = data["candidate_knots"].reshape(count, candidate_count, -1)
    hold = data["canonical_hold_knots"].reshape(count, -1)
    hold_distance = np.sqrt(
        np.mean(np.square((candidates - hold[:, None]) / sigma), axis=2)
    )
    candidate_cost = data["candidate_cost"].astype(np.float64)
    best_cost = np.min(candidate_cost, axis=1)
    nearest, fold_contract = input_contract(data)
    old_nn, old_spearman = target_metrics(
        data["old_fullrank_knots"], nearest, fold_contract, sigma
    )

    selection_index = np.empty((len(RELATIVE_TOLERANCES), count), np.int64)
    selected_cost = np.empty((len(RELATIVE_TOLERANCES), count), np.float64)
    selected_knots = np.empty((len(RELATIVE_TOLERANCES), count, 8, 2), np.float32)
    selected_nn = np.empty((len(RELATIVE_TOLERANCES), count), np.float64)
    selected_spearman = np.empty_like(selected_nn)
    records = []
    gate_config = config["pre_registered_gates"]
    warm = data["old_warm_cost"].astype(np.float64)
    for tolerance_index, tolerance in enumerate(RELATIVE_TOLERANCES):
        eligible = candidate_cost <= best_cost[:, None] * (1.0 + tolerance) + 1e-10
        choice = np.argmin(np.where(eligible, hold_distance, np.inf), axis=1)
        target = data["candidate_knots"][np.arange(count), choice]
        cost = candidate_cost[np.arange(count), choice]
        nn_distance, rank = target_metrics(target, nearest, fold_contract, sigma)
        selection_index[tolerance_index] = choice
        selected_cost[tolerance_index] = cost
        selected_knots[tolerance_index] = target
        selected_nn[tolerance_index] = nn_distance
        selected_spearman[tolerance_index] = rank
        nn_median = float(np.median(nn_distance))
        old_nn_median = float(np.median(old_nn))
        distance_reduction = float(1.0 - nn_median / old_nn_median)
        gates = {
            "mean_cost_no_greater_than_warm": bool(np.mean(cost) <= np.mean(warm) + 1e-8),
            "warm_regression_fraction_le_0p20": bool(
                np.mean(cost > warm + 1e-5)
                <= gate_config["canonical_warm_regression_fraction_maximum"]
            ),
            "nn_target_sigma_rms_median_le_0p80": bool(
                nn_median
                <= gate_config["canonical_nn_target_sigma_rms_median_maximum"]
            ),
            "distance_reduction_vs_old_fullrank_ge_0p15": bool(
                distance_reduction
                >= gate_config[
                    "nn_target_distance_reduction_vs_old_fullrank_minimum"
                ]
            ),
            "input_target_spearman_median_ge_0p58": bool(
                np.median(rank)
                >= gate_config["canonical_input_target_spearman_median_minimum"]
            ),
        }
        records.append(
            {
                "relative_cost_tolerance": tolerance,
                "selected_cost": stats(cost),
                "relative_cost_above_best": stats(cost / best_cost - 1.0),
                "warm_gain": stats(warm - cost),
                "warm_regression_fraction": float(np.mean(cost > warm + 1e-5)),
                "distance_from_observable_hold_sigma_rms": stats(
                    hold_distance[np.arange(count), choice]
                ),
                "nearest_target_sigma_rms": stats(nn_distance),
                "input_target_spearman": stats(rank),
                "nn_target_distance_reduction_vs_old_fullrank": distance_reduction,
                "selection_changed_from_hard_argmin_fraction": float(
                    np.mean(choice != data["canonical_teacher_index"])
                ),
                "gates": gates,
                "all_cost_and_coherence_gates_pass": bool(all(gates.values())),
            }
        )
    feasible = [
        record["relative_cost_tolerance"]
        for record in records
        if record["all_cost_and_coherence_gates_pass"]
    ]
    decision = (
        "CANONICAL_BANK_TIEBREAK_FEASIBLE_DIAGNOSTIC_ONLY"
        if feasible
        else "NO_TIEBREAK_WITHIN_CANONICAL_BANK_MEETS_COST_AND_COHERENCE"
    )
    analysis = {
        "qualification": "QUERY_CANONICAL_TIEBREAK_DIAGNOSTIC_COMPLETE",
        "decision": decision,
        "source_experimental_decision": summary["decision"],
        "row_count": count,
        "candidate_count_per_state": candidate_count,
        "relative_cost_tolerances": list(RELATIVE_TOLERANCES),
        "old_fullrank_same_subset": {
            "cost": stats(data["old_fullrank_cost"]),
            "nearest_target_sigma_rms": stats(old_nn),
            "input_target_spearman": stats(old_spearman),
        },
        "records": records,
        "feasible_relative_cost_tolerances": feasible,
        "interpretation": [
            "The scan is post-hoc routing evidence and does not alter the pre-registered audit gates.",
            "Near-optimal tie-breaking by distance to the observable hold cannot repair this bank if no row passes both cost and coherence gates.",
            "No new Query/DBM rollout and no model training were performed.",
        ],
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
    }
    output.mkdir(parents=True)
    diagnostics_path = output / "diagnostics.npz"
    np.savez_compressed(
        diagnostics_path,
        relative_cost_tolerances=np.asarray(RELATIVE_TOLERANCES, np.float64),
        nearest_fit_row=nearest,
        old_fullrank_nearest_target_sigma_rms=old_nn,
        old_fullrank_input_target_spearman=old_spearman,
        selection_index=selection_index,
        selected_cost=selected_cost,
        selected_knots=selected_knots,
        selected_nearest_target_sigma_rms=selected_nn,
        selected_input_target_spearman=selected_spearman,
    )
    analysis_path = output / "analysis.json"
    dump_json(analysis_path, analysis)
    output_manifest = {
        "schema_version": "query-noanchor-canonical-tiebreak-diagnostic-v1",
        "dataset_type": "zero-rollout-canonical-bank-tiebreak-diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "source": str(source),
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "source_summary_sha256": sha256(source / "summary.json"),
        "source_validation_sha256": sha256(source / "validation.json"),
        "source_audit_sha256": sha256(source / "audit.npz"),
        "analysis_sha256": sha256(analysis_path),
        "diagnostics_sha256": sha256(diagnostics_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "new_query_rollouts": 0,
        "new_dbm_rollouts": 0,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
    }
    dump_json(output / "manifest.json", output_manifest)
    print(json.dumps({"qualification": analysis["qualification"], "decision": decision, "records": records}, indent=2))


if __name__ == "__main__":
    main()
