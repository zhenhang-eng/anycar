#!/usr/bin/env python3
"""Independently reproduce the single-center Critic ranking diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ARTIFACT = Path("outputs/query_mppi/query_single_center_critic_ranking_diagnostic_20260902_v1")
PAIR_SAMPLES = 4096


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def accuracy(truth: np.ndarray, estimate: np.ndarray, left: np.ndarray,
             right: np.ndarray, gap: float = 0.0) -> float:
    delta = truth[left] - truth[right]
    predicted = estimate[left] - estimate[right]
    keep = np.abs(delta) > max(gap, 1e-7)
    return float(np.mean(np.sign(delta[keep]) == np.sign(predicted[keep])))


def main() -> None:
    artifact = ARTIFACT.resolve()
    manifest = json.loads((artifact / "manifest.json").read_text())
    summary = json.loads((artifact / "summary.json").read_text())
    replay = Path(manifest["replay"])
    pretrain = Path(manifest["pretrain"])
    fullrank = Path(manifest["fullrank"])
    checks = {
        "source_hashes": (
            sha256(replay / "replay.npz") == manifest["replay_sha256"]
            and sha256(pretrain / "oof_predictions.npz") == manifest["oof_predictions_sha256"]
            and sha256(pretrain / "validation.json") == manifest["pretrain_validation_sha256"]
            and sha256(fullrank / "bank.npz") == manifest["fullrank_bank_sha256"]
        ),
        "artifact_hashes": (
            sha256(artifact / "summary.json") == manifest["summary_sha256"]
            and sha256(artifact / "diagnostics.npz") == manifest["diagnostics_sha256"]
            and sha256(Path(manifest["script"])) == manifest["script_sha256"]
        ),
        "sealed_boundaries": (
            not manifest["formal_validation_or_test_consumed"]
            and not manifest["dbm_fields_or_labels_consumed"]
            and not manifest["query_analytic_gradient_consumed"]
            and manifest["new_query_rollouts"] == 0
        ),
    }
    with np.load(replay / "replay.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(pretrain / "oof_predictions.npz", allow_pickle=False) as archive:
        prediction = np.asarray(archive["twin_conservative_log_cost"])
    with np.load(fullrank / "bank.npz", allow_pickle=False) as archive:
        fullrank_pairs = np.asarray(archive["pair_candidate_indices"])
    with np.load(artifact / "diagnostics.npz", allow_pickle=False) as archive:
        saved = {name: np.asarray(archive[name]) for name in archive.files}
    state = np.empty_like(saved["state_accuracy"])
    quantile = np.empty_like(saved["low_quantile_accuracy"])
    gap_values = np.empty_like(saved["gap_accuracy"])
    source = np.empty_like(saved["source_accuracy"])
    full_pair = np.empty_like(saved["fullrank_antithetic"])
    land_pair = np.empty_like(saved["landscape_antithetic"])
    land_gap = np.empty_like(saved["landscape_pair_median_gap"])
    for seed in range(3):
        for row in range(600):
            valid = np.flatnonzero(data["candidate_valid_mask"][row])
            truth = np.log1p(data["candidate_cost"][row, valid].astype(np.float64))
            estimate = prediction[seed, row, valid].astype(np.float64)
            rng = np.random.default_rng(1_234_000 + int(data["row_index"][row]))
            left, right = rng.integers(0, len(valid), PAIR_SAMPLES), rng.integers(0, len(valid), PAIR_SAMPLES)
            state[seed, row] = accuracy(truth, estimate, left, right)
            order = np.argsort(truth)
            for index, value in enumerate(saved["quantiles"]):
                count = max(2, int(np.ceil(len(valid) * value)))
                local = order[:count]
                rng = np.random.default_rng(2_200_000 + int(data["row_index"][row]))
                left, right = rng.choice(local, PAIR_SAMPLES), rng.choice(local, PAIR_SAMPLES)
                quantile[seed, index, row] = accuracy(truth, estimate, left, right)
        for index, gap in enumerate(saved["gaps"]):
            correct, total = 0, 0
            for row in range(600):
                valid = np.flatnonzero(data["candidate_valid_mask"][row])
                truth = np.log1p(data["candidate_cost"][row].astype(np.float64))
                estimate = prediction[seed, row].astype(np.float64)
                rng = np.random.default_rng(900_000 + int(data["row_index"][row]))
                left, right = rng.choice(valid, PAIR_SAMPLES), rng.choice(valid, PAIR_SAMPLES)
                delta = truth[left] - truth[right]
                keep = np.abs(delta) > max(float(gap), 1e-7)
                correct += int(np.sum(np.sign(delta[keep]) == np.sign(
                    estimate[left][keep] - estimate[right][keep]
                )))
                total += int(np.sum(keep))
            gap_values[seed, index] = correct / total
        for source_index, mask in enumerate((data["candidate_source"] == 0, data["candidate_source"] > 0)):
            correct, total = 0, 0
            for row in range(600):
                valid = np.flatnonzero(data["candidate_valid_mask"][row] & mask[row])
                if len(valid) < 2:
                    continue
                truth = np.log1p(data["candidate_cost"][row].astype(np.float64))
                estimate = prediction[seed, row].astype(np.float64)
                rng = np.random.default_rng(900_000 + int(data["row_index"][row]))
                left, right = rng.choice(valid, PAIR_SAMPLES), rng.choice(valid, PAIR_SAMPLES)
                delta = truth[left] - truth[right]
                keep = np.abs(delta) > 1e-7
                correct += int(np.sum(np.sign(delta[keep]) == np.sign(
                    estimate[left][keep] - estimate[right][keep]
                )))
                total += int(np.sum(keep))
            source[seed, source_index] = correct / total
        for radius, pairs in enumerate(fullrank_pairs):
            left, right = pairs[:, 0], pairs[:, 1]
            truth = np.log1p(data["candidate_cost"][:, left]) - np.log1p(data["candidate_cost"][:, right])
            estimate = prediction[seed][:, left] - prediction[seed][:, right]
            keep = np.abs(truth) > 1e-7
            full_pair[seed, radius] = np.mean(np.sign(truth[keep]) == np.sign(estimate[keep]))
        for round_index in range(4):
            correct, gaps = [], []
            for row in np.flatnonzero(data["landscape_context_mask"]):
                for branch in range(5):
                    ids = np.flatnonzero(
                        (data["candidate_source"][row] == 2)
                        & (data["candidate_round"][row] == round_index)
                        & (data["candidate_branch"][row] == branch)
                    )
                    ids = ids[np.argsort(data["candidate_local_index"][row, ids])]
                    left, right = ids[0::2], ids[1::2]
                    truth = np.log1p(data["candidate_cost"][row, left]) - np.log1p(data["candidate_cost"][row, right])
                    estimate = prediction[seed, row, left] - prediction[seed, row, right]
                    correct.extend((np.sign(truth) == np.sign(estimate)).tolist())
                    gaps.extend(np.abs(truth).tolist())
            land_pair[seed, round_index] = np.mean(correct)
            land_gap[round_index] = np.median(gaps)
    errors = {
        "state_accuracy": float(np.max(np.abs(state - saved["state_accuracy"]))),
        "low_quantile_accuracy": float(np.max(np.abs(quantile - saved["low_quantile_accuracy"]))),
        "gap_accuracy": float(np.max(np.abs(gap_values - saved["gap_accuracy"]))),
        "source_accuracy": float(np.max(np.abs(source - saved["source_accuracy"]))),
        "fullrank_antithetic": float(np.max(np.abs(full_pair - saved["fullrank_antithetic"]))),
        "landscape_antithetic": float(np.max(np.abs(land_pair - saved["landscape_antithetic"]))),
        "landscape_pair_gap": float(np.max(np.abs(land_gap - saved["landscape_pair_median_gap"]))),
    }
    checks["all_arrays_exact"] = max(errors.values()) == 0.0
    checks["decision_consistent"] = (
        summary["decision"] == "GLOBAL_VALUE_SIGNAL_PRESENT_BUT_LOW_COST_LOCAL_ORDERING_NOT_READY_FOR_AC"
        and np.max(np.median(quantile[:, 1], axis=1)) < 0.60
        and np.max(land_pair[:, -1]) < 0.70
        and np.max(full_pair[:, 0]) < 0.70
    )
    qualification = (
        "QUERY_SINGLE_CENTER_CRITIC_RANKING_INDEPENDENT_PASS"
        if all(checks.values()) else "QUERY_SINGLE_CENTER_CRITIC_RANKING_INDEPENDENT_FAIL"
    )
    validation = {
        "qualification": qualification, "checks": checks, "maximum_errors": errors,
        "decision": summary["decision"], "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
        "new_query_rollouts": 0, "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256(Path(__file__).resolve()),
    }
    (artifact / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True, default=json_default) + "\n"
    )
    print(json.dumps(validation, indent=2, default=json_default))
    if qualification.endswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
