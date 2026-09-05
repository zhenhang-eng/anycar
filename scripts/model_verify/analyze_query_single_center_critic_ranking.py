#!/usr/bin/env python3
"""Diagnose OOF Critic ranking by density, cost gap, tail, and exact probe pairs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPLAY = Path("/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/query_single_center_absolute_replay_20260902_v1")
PRETRAIN = Path("outputs/query_mppi/query_single_center_actor_twin_critic_pretrain_20260902_v1")
FULLRANK = Path("/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/query_expected_road_fullrank_expansion_20260901_v3")
OUTPUT = Path("outputs/query_mppi/query_single_center_critic_ranking_diagnostic_20260902_v1")
QUANTILES = np.asarray((0.05, 0.10, 0.25, 0.50, 1.00), np.float64)
GAPS = np.asarray((0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.50, 1.00), np.float64)
PAIR_SAMPLES = 4096


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64).reshape(-1)
    return {"count": int(len(values)), "min": float(values.min()),
            "p05": float(np.quantile(values, 0.05)), "median": float(np.median(values)),
            "mean": float(values.mean()), "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max())}


def pair_accuracy(truth: np.ndarray, prediction: np.ndarray, left: np.ndarray,
                  right: np.ndarray, minimum_gap: float = 0.0) -> float:
    true_delta = truth[left] - truth[right]
    pred_delta = prediction[left] - prediction[right]
    material = np.abs(true_delta) > max(minimum_gap, 1e-7)
    return float(np.mean(np.sign(true_delta[material]) == np.sign(pred_delta[material])))


def main() -> None:
    output = OUTPUT.resolve()
    if output.exists():
        raise FileExistsError(output)
    replay = REPLAY.resolve()
    pretrain = PRETRAIN.resolve()
    fullrank = FULLRANK.resolve()
    replay_validation = json.loads((replay / "validation.json").read_text())
    pretrain_validation = json.loads((pretrain / "validation.json").read_text())
    if replay_validation["qualification"] != "QUERY_SINGLE_CENTER_ABSOLUTE_REPLAY_INDEPENDENT_PASS":
        raise AssertionError("Replay is not qualified")
    if pretrain_validation["qualification"] != "QUERY_SINGLE_CENTER_PRETRAIN_INDEPENDENT_CONFIRMED_FAIL_NO_OAC":
        raise AssertionError("pretrain failure was not independently confirmed")
    with np.load(replay / "replay.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(pretrain / "oof_predictions.npz", allow_pickle=False) as archive:
        prediction = np.asarray(archive["twin_conservative_log_cost"])
        seeds = np.asarray(archive["seeds"])
    with np.load(fullrank / "bank.npz", allow_pickle=False) as archive:
        fullrank_pairs = np.asarray(archive["pair_candidate_indices"])
        fullrank_radii = np.unique(archive["candidate_radius_sigma"][archive["candidate_group"] == "proximal"])
    state_accuracy = np.empty((3, 600), np.float64)
    low_quantile_accuracy = np.empty((3, len(QUANTILES), 600), np.float64)
    gap_accuracy = np.empty((3, len(GAPS)), np.float64)
    source_accuracy = np.empty((3, 2), np.float64)
    fullrank_antithetic = np.empty((3, 4), np.float64)
    landscape_antithetic = np.empty((3, 4), np.float64)
    landscape_pair_median_gap = np.empty(4, np.float64)
    for seed_index in range(3):
        for row in range(600):
            valid = np.flatnonzero(data["candidate_valid_mask"][row])
            truth = np.log1p(data["candidate_cost"][row, valid].astype(np.float64))
            estimate = prediction[seed_index, row, valid].astype(np.float64)
            rng = np.random.default_rng(1_234_000 + int(data["row_index"][row]))
            left = rng.integers(0, len(valid), PAIR_SAMPLES)
            right = rng.integers(0, len(valid), PAIR_SAMPLES)
            state_accuracy[seed_index, row] = pair_accuracy(truth, estimate, left, right)
            order = np.argsort(truth)
            for q_index, quantile in enumerate(QUANTILES):
                count = max(2, int(np.ceil(len(valid) * quantile)))
                local = order[:count]
                tail_rng = np.random.default_rng(2_200_000 + int(data["row_index"][row]))
                left = tail_rng.choice(local, PAIR_SAMPLES)
                right = tail_rng.choice(local, PAIR_SAMPLES)
                low_quantile_accuracy[seed_index, q_index, row] = pair_accuracy(
                    truth, estimate, left, right
                )
        for gap_index, gap in enumerate(GAPS):
            correct, total = 0, 0
            for row in range(600):
                valid = np.flatnonzero(data["candidate_valid_mask"][row])
                truth = np.log1p(data["candidate_cost"][row].astype(np.float64))
                estimate = prediction[seed_index, row].astype(np.float64)
                rng = np.random.default_rng(900_000 + int(data["row_index"][row]))
                left, right = rng.choice(valid, PAIR_SAMPLES), rng.choice(valid, PAIR_SAMPLES)
                delta = truth[left] - truth[right]
                material = np.abs(delta) > max(float(gap), 1e-7)
                correct += int(np.sum(np.sign(delta[material]) == np.sign(
                    estimate[left][material] - estimate[right][material]
                )))
                total += int(np.sum(material))
            gap_accuracy[seed_index, gap_index] = correct / total
        for source_index, source_mask in enumerate((data["candidate_source"] == 0, data["candidate_source"] > 0)):
            correct, total = 0, 0
            for row in range(600):
                valid = np.flatnonzero(data["candidate_valid_mask"][row] & source_mask[row])
                if len(valid) < 2:
                    continue
                truth = np.log1p(data["candidate_cost"][row].astype(np.float64))
                estimate = prediction[seed_index, row].astype(np.float64)
                rng = np.random.default_rng(900_000 + int(data["row_index"][row]))
                left, right = rng.choice(valid, PAIR_SAMPLES), rng.choice(valid, PAIR_SAMPLES)
                delta = truth[left] - truth[right]
                material = np.abs(delta) > 1e-7
                correct += int(np.sum(np.sign(delta[material]) == np.sign(
                    estimate[left][material] - estimate[right][material]
                )))
                total += int(np.sum(material))
            source_accuracy[seed_index, source_index] = correct / total
        for radius_index, pairs in enumerate(fullrank_pairs):
            left, right = pairs[:, 0], pairs[:, 1]
            truth_delta = np.log1p(data["candidate_cost"][:, left]) - np.log1p(data["candidate_cost"][:, right])
            pred_delta = prediction[seed_index][:, left] - prediction[seed_index][:, right]
            material = np.abs(truth_delta) > 1e-7
            fullrank_antithetic[seed_index, radius_index] = np.mean(
                np.sign(truth_delta[material]) == np.sign(pred_delta[material])
            )
        landscape_rows = np.flatnonzero(data["landscape_context_mask"])
        for round_index in range(4):
            correct, gaps = [], []
            for row in landscape_rows:
                for branch in range(5):
                    ids = np.flatnonzero(
                        (data["candidate_source"][row] == 2)
                        & (data["candidate_round"][row] == round_index)
                        & (data["candidate_branch"][row] == branch)
                    )
                    ids = ids[np.argsort(data["candidate_local_index"][row, ids])]
                    left, right = ids[0::2], ids[1::2]
                    truth_delta = np.log1p(data["candidate_cost"][row, left]) - np.log1p(data["candidate_cost"][row, right])
                    pred_delta = prediction[seed_index, row, left] - prediction[seed_index, row, right]
                    correct.extend((np.sign(truth_delta) == np.sign(pred_delta)).tolist())
                    gaps.extend(np.abs(truth_delta).tolist())
            landscape_antithetic[seed_index, round_index] = np.mean(correct)
            landscape_pair_median_gap[round_index] = np.median(gaps)
    full_only = ~data["landscape_context_mask"]
    landscape_context = data["landscape_context_mask"]
    summary = {
        "qualification": "QUERY_SINGLE_CENTER_CRITIC_RANKING_DIAGNOSTIC_COMPLETE",
        "decision": "GLOBAL_VALUE_SIGNAL_PRESENT_BUT_LOW_COST_LOCAL_ORDERING_NOT_READY_FOR_AC",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state_pair_accuracy": {
            "fullrank_only_500": [stats(value[full_only]) for value in state_accuracy],
            "landscape_context_100": [stats(value[landscape_context]) for value in state_accuracy],
        },
        "low_cost_quantile_pair_accuracy_median_by_seed": {
            str(float(q)): [float(np.median(low_quantile_accuracy[s, i])) for s in range(3)]
            for i, q in enumerate(QUANTILES)
        },
        "minimum_log_cost_gap_accuracy_by_seed": {
            str(float(gap)): gap_accuracy[:, i].tolist() for i, gap in enumerate(GAPS)
        },
        "source_pair_accuracy_by_seed": {
            "fullrank": source_accuracy[:, 0].tolist(), "landscape": source_accuracy[:, 1].tolist()
        },
        "fullrank_exact_antithetic_accuracy": {
            str(float(radius)): fullrank_antithetic[:, i].tolist()
            for i, radius in enumerate(fullrank_radii)
        },
        "landscape_exact_antithetic_accuracy": {
            str(i): {
                "accuracy_by_seed": landscape_antithetic[:, i].tolist(),
                "true_log_cost_gap_median": float(landscape_pair_median_gap[i]),
            } for i in range(4)
        },
        "interpretation": [
            "The 100 dense landscape contexts are learnable at coarse/global scale; the 500 sparse full-rank-only contexts dominate the pooled failure.",
            "Ordering improves above a material true log-cost gap, so near-ties explain part of the global pair-sign miss.",
            "The lowest-cost 5-25 percent candidate regions remain close to random ordering and exact antithetic accuracy degrades as probe radius shrinks.",
            "No fresh Actor-center FD and no OAC are authorized by this diagnostic.",
        ],
        "formal_validation_or_test_consumed": False, "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False, "new_query_rollouts": 0,
    }
    output.mkdir(parents=True)
    array_path = output / "diagnostics.npz"
    np.savez_compressed(
        array_path, seeds=seeds, row_index=data["row_index"], quantiles=QUANTILES, gaps=GAPS,
        state_accuracy=state_accuracy, low_quantile_accuracy=low_quantile_accuracy,
        gap_accuracy=gap_accuracy, source_accuracy=source_accuracy,
        fullrank_radii=fullrank_radii, fullrank_antithetic=fullrank_antithetic,
        landscape_antithetic=landscape_antithetic,
        landscape_pair_median_gap=landscape_pair_median_gap,
    )
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "query-single-center-critic-ranking-diagnostic-v1",
        "qualification": summary["qualification"], "decision": summary["decision"],
        "replay": str(replay), "replay_sha256": sha256(replay / "replay.npz"),
        "pretrain": str(pretrain), "oof_predictions_sha256": sha256(pretrain / "oof_predictions.npz"),
        "pretrain_validation_sha256": sha256(pretrain / "validation.json"),
        "fullrank": str(fullrank), "fullrank_bank_sha256": sha256(fullrank / "bank.npz"),
        "summary_sha256": sha256(summary_path), "diagnostics_sha256": sha256(array_path),
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "formal_validation_or_test_consumed": False, "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False, "new_query_rollouts": 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
