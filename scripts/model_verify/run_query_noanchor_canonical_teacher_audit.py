#!/usr/bin/env python3
"""Run the frozen train-only no-anchor canonical Query teacher audit.

The search starts from repeated observable current action and may only recenter
on its own frozen-Query argmin.  Stored warm/T0/full-rank centers are never read
by candidate construction or teacher selection; they are comparison fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package))

from car_dynamics.controllers_torch.mppi import (  # noqa: E402
    TorchMPPIController,
    TorchMPPIParams,
)
from car_foundation.query_deployment import (  # noqa: E402
    QueryDeploymentModel,
    TorchQueryRolloutBackend,
)


DEFAULT_CONFIG = REPO_ROOT / (
    "scripts/model_verify/query_noanchor_canonical_teacher_audit_config_20260901_v1.json"
)
CONTEXT_FIELDS = (
    "row_index",
    "episode_id",
    "episode_index",
    "row_in_episode",
    "control_step",
    "speed_kph",
    "speed_index",
    "variant_index",
    "road_name",
    "fold_id",
    "source_snapshot_sha256",
    "state",
    "current_action",
    "history",
    "reference",
    "reference_ego",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
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
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def hadamard_16() -> np.ndarray:
    matrix = np.ones((1, 1), dtype=np.float64)
    while matrix.shape[0] < 16:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def dct_16() -> np.ndarray:
    index = np.arange(16, dtype=np.float64)
    frequency = np.arange(16, dtype=np.float64)[:, None]
    matrix = np.cos(math.pi * (index[None, :] + 0.5) * frequency / 16.0)
    matrix[0] *= math.sqrt(1.0 / 16.0)
    matrix[1:] *= math.sqrt(2.0 / 16.0)
    return matrix * math.sqrt(16.0)


def direction_bank() -> np.ndarray:
    directions = np.concatenate((hadamard_16(), dct_16()), axis=0)
    rms = np.sqrt(np.mean(np.square(directions), axis=1))
    if not np.allclose(rms, 1.0, atol=1e-12):
        raise AssertionError("direction RMS contract failed")
    if np.linalg.matrix_rank(directions[:16]) != 16:
        raise AssertionError("Hadamard basis is not full rank")
    if np.linalg.matrix_rank(directions[16:]) != 16:
        raise AssertionError("DCT basis is not full rank")
    return directions.reshape(32, 8, 2).astype(np.float32)


def interpolate_knots(knots: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(knots, dtype=torch.float32)
    return (
        F.interpolate(
            tensor.transpose(1, 2), size=50, mode="linear", align_corners=True
        )
        .transpose(1, 2)
        .numpy()
    )


def robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    quartile = np.quantile(values, (0.25, 0.75), axis=0)
    scale = quartile[1] - quartile[0]
    standard = np.std(values, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(standard > 1e-8, standard, 1.0))
    return center, scale


def feature_blocks(data: dict[str, np.ndarray]) -> list[np.ndarray]:
    count = len(data["state"])
    return [
        data["history"].reshape(count, -1).astype(np.float64),
        data["reference_ego"].reshape(count, -1).astype(np.float64),
        np.concatenate(
            (data["state"][:, 3:5], data["current_action"]), axis=1
        ).astype(np.float64),
    ]


def embed(
    blocks: list[np.ndarray], rows: np.ndarray, fit_rows: np.ndarray
) -> np.ndarray:
    pieces = []
    for block in blocks:
        center, scale = robust_scale(block[fit_rows])
        pieces.append((block[rows] - center) / scale / math.sqrt(block.shape[1]))
    return np.concatenate(pieces, axis=1)


def pairwise_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(np.square(left), axis=1)[:, None]
        + np.sum(np.square(right), axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.sqrt(np.maximum(squared, 0.0))


def action_distance(
    query: np.ndarray, bank: np.ndarray, sigma_flat: np.ndarray
) -> np.ndarray:
    query_scaled = query.reshape(len(query), -1) / sigma_flat
    bank_scaled = bank.reshape(len(bank), -1) / sigma_flat
    squared = np.square(query_scaled[:, None] - bank_scaled[None])
    return np.sqrt(np.mean(squared, axis=2))


def nested_coherence(
    data: dict[str, np.ndarray],
    canonical: np.ndarray,
    old_fullrank: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, np.ndarray]:
    count = len(canonical)
    nearest = np.empty(count, dtype=np.int64)
    nearest_input_distance = np.empty(count, dtype=np.float64)
    canonical_nearest_target_distance = np.empty(count, dtype=np.float64)
    old_nearest_target_distance = np.empty(count, dtype=np.float64)
    canonical_spearman = np.empty(count, dtype=np.float64)
    old_spearman = np.empty(count, dtype=np.float64)
    folds = data["fold_id"].astype(np.int64)
    blocks = feature_blocks(data)
    sigma_flat = np.tile(sigma.astype(np.float64), 8)
    for fold in range(5):
        query_rows = np.flatnonzero(folds == fold)
        excluded = (fold + 1) % 5
        fit_rows = np.flatnonzero((folds != fold) & (folds != excluded))
        if (len(query_rows), len(fit_rows)) != (20, 60):
            raise AssertionError("expected nested 20-query/60-fit audit split")
        query_embedding = embed(blocks, query_rows, fit_rows)
        fit_embedding = embed(blocks, fit_rows, fit_rows)
        input_distance = pairwise_distance(query_embedding, fit_embedding)
        canonical_distance = action_distance(
            canonical[query_rows], canonical[fit_rows], sigma_flat
        )
        old_distance = action_distance(
            old_fullrank[query_rows], old_fullrank[fit_rows], sigma_flat
        )
        local_nearest = np.argmin(input_distance, axis=1)
        nearest[query_rows] = fit_rows[local_nearest]
        nearest_input_distance[query_rows] = input_distance[
            np.arange(len(query_rows)), local_nearest
        ]
        canonical_nearest_target_distance[query_rows] = canonical_distance[
            np.arange(len(query_rows)), local_nearest
        ]
        old_nearest_target_distance[query_rows] = old_distance[
            np.arange(len(query_rows)), local_nearest
        ]
        for local, row in enumerate(query_rows):
            canonical_spearman[row] = float(
                spearmanr(input_distance[local], canonical_distance[local]).statistic
            )
            old_spearman[row] = float(
                spearmanr(input_distance[local], old_distance[local]).statistic
            )
    if np.any(~np.isfinite(canonical_spearman)) or np.any(~np.isfinite(old_spearman)):
        raise AssertionError("non-finite input-target Spearman")
    return {
        "nearest_fit_row": nearest,
        "nearest_input_distance": nearest_input_distance,
        "canonical_nearest_target_sigma_rms": canonical_nearest_target_distance,
        "old_fullrank_nearest_target_sigma_rms": old_nearest_target_distance,
        "canonical_input_target_spearman": canonical_spearman,
        "old_fullrank_input_target_spearman": old_spearman,
    }


def grouped_metrics(data: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    hold_cost = data["canonical_hold_cost"][mask].astype(np.float64)
    canonical_cost = data["canonical_teacher_cost"][mask].astype(np.float64)
    warm_cost = data["old_warm_cost"][mask].astype(np.float64)
    transfer_cost = data["nearest_target_transfer_cost"][mask].astype(np.float64)
    denominator = float(np.sum(hold_cost - canonical_cost))
    transfer_recovery = (
        float(np.sum(hold_cost - transfer_cost) / denominator)
        if denominator > 1e-12
        else 0.0
    )
    return {
        "rows": int(mask.sum()),
        "canonical_hold_cost": stats(hold_cost),
        "canonical_teacher_cost": stats(canonical_cost),
        "old_warm_cost": stats(warm_cost),
        "old_t0_cost": stats(data["old_t0_cost"][mask]),
        "old_fullrank_cost": stats(data["old_fullrank_cost"][mask]),
        "canonical_gain_vs_hold": stats(hold_cost - canonical_cost),
        "canonical_gain_vs_warm": stats(warm_cost - canonical_cost),
        "canonical_warm_regression_fraction": float(
            np.mean(canonical_cost > warm_cost + 1e-5)
        ),
        "nearest_target_transfer_cost": stats(transfer_cost),
        "nearest_target_transfer_recovery_vs_hold": transfer_recovery,
        "canonical_nearest_target_sigma_rms": stats(
            data["canonical_nearest_target_sigma_rms"][mask]
        ),
        "old_fullrank_nearest_target_sigma_rms": stats(
            data["old_fullrank_nearest_target_sigma_rms"][mask]
        ),
        "canonical_input_target_spearman": stats(
            data["canonical_input_target_spearman"][mask]
        ),
        "old_fullrank_input_target_spearman": stats(
            data["old_fullrank_input_target_spearman"][mask]
        ),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    source = Path(config["source_sidecar"]).resolve()
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if config["formal_validation_or_test_consumed"]:
        raise AssertionError("formal validation/test must remain sealed")
    if config.get("dbm_fields_or_labels_consumed"):
        raise AssertionError("DBM fields or labels are forbidden")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    source_manifest_path = source / "manifest.json"
    source_validation_path = source / "validation.json"
    source_bank_path = source / "bank.npz"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_validation = json.loads(source_validation_path.read_text())
    if source_validation["qualification"] != "QUERY_EXPECTED_ROAD_FULLRANK_PASS":
        raise AssertionError("source full-rank sidecar did not pass")
    if source_manifest.get("formal_validation_or_test_consumed", True):
        raise AssertionError("source consumed formal validation/test")
    if sha256(source_bank_path) != source_manifest["bank_sha256"]:
        raise AssertionError("source bank hash mismatch")

    with np.load(source_bank_path, allow_pickle=False) as archive:
        selection = (
            (archive["row_in_episode"] == config["state_selection"]["row_in_episode"])
            & (archive["control_step"] == config["state_selection"]["control_step"])
        )
        source_rows = np.flatnonzero(selection)
        expected = int(config["state_selection"]["expected_episode_count"])
        if len(source_rows) != expected:
            raise AssertionError(f"expected {expected} audit rows, got {len(source_rows)}")
        data = {name: np.asarray(archive[name][source_rows]) for name in CONTEXT_FIELDS}
        comparators = {
            "old_warm_knots": np.asarray(archive["mean_knots_before"][source_rows]),
            "old_t0_knots": np.asarray(archive["teacher_knots"][source_rows]),
            "old_fullrank_knots": np.asarray(
                archive["fullrank_teacher_knots"][source_rows]
            ),
            "old_warm_cost": np.asarray(
                archive["warm_direct_cost_replayed"][source_rows]
            ),
            "old_t0_cost": np.asarray(
                archive["t0_teacher_direct_cost_replayed"][source_rows]
            ),
            "old_fullrank_cost": np.asarray(
                archive["fullrank_teacher_direct_cost"][source_rows]
            ),
        }
    data["source_row_index"] = source_rows.astype(np.int64)
    data.update(comparators)
    if len(np.unique(data["episode_id"])) != expected:
        raise AssertionError("audit rows are not episode independent")
    per_fold = [int(np.sum(data["fold_id"] == fold)) for fold in range(5)]
    if per_fold != [int(config["state_selection"]["expected_rows_per_fold"])] * 5:
        raise AssertionError(f"unbalanced folds: {per_fold}")
    if sorted(np.unique(data["speed_kph"]).tolist()) != [40, 55, 70, 85, 100]:
        raise AssertionError("unexpected speed coverage")
    if sorted(np.unique(data["variant_index"]).tolist()) != [0, 1, 2, 3]:
        raise AssertionError("unexpected road-variant coverage")

    parent_t0 = Path(source_manifest["parent_t0"])
    parent_manifest = json.loads((parent_t0 / "manifest.json").read_text())
    collection = Path(parent_manifest["source_collection"])
    collection_manifest = json.loads((collection / "manifest.json").read_text())
    params = TorchMPPIParams(**collection_manifest["collection"]["mppi"])
    device = torch.device(args.device)
    query_model = QueryDeploymentModel.from_checkpoint(
        Path(source_manifest["query_checkpoint"]), device
    )
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query_model), params, device=str(device)
    )

    sigma = np.asarray(config["noise_sigma"], dtype=np.float32)
    low = np.asarray(config["action_bounds"]["minimum"], dtype=np.float32)
    high = np.asarray(config["action_bounds"]["maximum"], dtype=np.float32)
    directions = direction_bank()
    radii = [float(value) for value in config["stage_radii_sigma"]]
    candidates_per_stage = int(config["candidates_per_stage"])
    if candidates_per_stage != 2 * len(directions):
        raise AssertionError("candidate/direction contract mismatch")
    candidate_count = 1 + len(radii) * candidates_per_stage
    if candidate_count != int(config["total_candidates_per_state"]):
        raise AssertionError("total candidate contract mismatch")
    row_count = len(data["state"])
    raw_candidates = np.empty((row_count, candidate_count, 8, 2), np.float32)
    candidates = np.empty_like(raw_candidates)
    costs = np.empty((row_count, candidate_count), np.float32)
    clipped = np.empty_like(raw_candidates, dtype=bool)
    component_names = ("position", "yaw", "vx", "acceleration_rate", "steering_rate")
    components = {
        name: np.empty((row_count, candidate_count), np.float32)
        for name in component_names
    }
    stage_best_global = np.empty((row_count, len(radii) + 1), np.int64)
    stage_best_cost = np.empty((row_count, len(radii) + 1), np.float32)
    for row in range(row_count):
        initial = np.broadcast_to(data["current_action"][row], (8, 2)).copy()
        initial = np.clip(initial, low, high).astype(np.float32)
        raw_candidates[row, 0] = initial
        candidates[row, 0] = initial
        clipped[row, 0] = False
        result = controller.evaluate_action_sequences(
            data["state"][row],
            data["current_action"][row],
            data["history"][row : row + 1],
            data["reference"][row],
            interpolate_knots(initial[None]),
        )
        costs[row, 0] = float(result["cost"][0].cpu())
        for name in component_names:
            components[name][row, 0] = float(
                result["cost_components"][name][0].cpu()
            )
        incumbent = initial
        incumbent_index = 0
        incumbent_cost = float(costs[row, 0])
        stage_best_global[row, 0] = incumbent_index
        stage_best_cost[row, 0] = incumbent_cost
        for stage, radius in enumerate(radii):
            start = 1 + stage * candidates_per_stage
            raw = np.stack(
                [
                    incumbent + sign * radius * direction * sigma[None]
                    for direction in directions
                    for sign in (1.0, -1.0)
                ]
            ).astype(np.float32)
            local = np.clip(raw, low, high)
            raw_candidates[row, start : start + candidates_per_stage] = raw
            candidates[row, start : start + candidates_per_stage] = local
            clipped[row, start : start + candidates_per_stage] = (
                np.abs(raw - local) > 1e-7
            )
            result = controller.evaluate_action_sequences(
                data["state"][row],
                data["current_action"][row],
                data["history"][row : row + 1],
                data["reference"][row],
                interpolate_knots(local),
            )
            local_cost = result["cost"].cpu().numpy().astype(np.float32)
            costs[row, start : start + candidates_per_stage] = local_cost
            for name in component_names:
                components[name][row, start : start + candidates_per_stage] = (
                    result["cost_components"][name].cpu().numpy().astype(np.float32)
                )
            local_best = int(np.argmin(local_cost))
            if float(local_cost[local_best]) < incumbent_cost:
                incumbent_index = start + local_best
                incumbent = local[local_best]
                incumbent_cost = float(local_cost[local_best])
            stage_best_global[row, stage + 1] = incumbent_index
            stage_best_cost[row, stage + 1] = incumbent_cost
        if (row + 1) % 10 == 0:
            print(f"canonical search {row + 1}/{row_count}", flush=True)

    teacher_index = stage_best_global[:, -1]
    rows = np.arange(row_count)
    canonical_teacher = candidates[rows, teacher_index]
    canonical_cost = costs[rows, teacher_index]
    data.update(
        {
            "candidate_raw_knots": raw_candidates,
            "candidate_knots": candidates,
            "candidate_cost": costs,
            "candidate_clipped_mask": clipped,
            "stage_best_global_index": stage_best_global,
            "stage_best_cost": stage_best_cost,
            "canonical_hold_knots": candidates[:, 0],
            "canonical_hold_cost": costs[:, 0],
            "canonical_teacher_index": teacher_index,
            "canonical_teacher_knots": canonical_teacher,
            "canonical_teacher_cost": canonical_cost,
            "candidate_clip_fraction": np.mean(clipped, axis=(2, 3)),
        }
    )
    for name, value in components.items():
        data[f"candidate_cost_component_{name}"] = value

    coherence = nested_coherence(
        data, canonical_teacher, data["old_fullrank_knots"], sigma
    )
    data.update(coherence)
    transfer_knots = canonical_teacher[coherence["nearest_fit_row"]]
    transfer_cost = np.empty(row_count, np.float32)
    for row in range(row_count):
        result = controller.evaluate_action_sequences(
            data["state"][row],
            data["current_action"][row],
            data["history"][row : row + 1],
            data["reference"][row],
            interpolate_knots(transfer_knots[row : row + 1]),
        )
        transfer_cost[row] = float(result["cost"][0].cpu())
    data["nearest_target_transfer_knots"] = transfer_knots
    data["nearest_target_transfer_cost"] = transfer_cost

    overall = grouped_metrics(data, np.ones(row_count, dtype=bool))
    hold_gain = np.sum(
        data["canonical_hold_cost"].astype(np.float64)
        - data["canonical_teacher_cost"].astype(np.float64)
    )
    final_stage_gain = np.sum(
        data["stage_best_cost"][:, -2].astype(np.float64)
        - data["stage_best_cost"][:, -1].astype(np.float64)
    )
    final_stage_share = float(final_stage_gain / hold_gain) if hold_gain > 1e-12 else 1.0
    canonical_nn_median = float(
        np.median(data["canonical_nearest_target_sigma_rms"])
    )
    old_nn_median = float(
        np.median(data["old_fullrank_nearest_target_sigma_rms"])
    )
    distance_reduction = float(1.0 - canonical_nn_median / old_nn_median)
    gates_config = config["pre_registered_gates"]
    gates = {
        "canonical_mean_cost_no_greater_than_warm": bool(
            np.mean(data["canonical_teacher_cost"])
            <= np.mean(data["old_warm_cost"]) + 1e-8
        ),
        "canonical_warm_regression_fraction_le_0p20": bool(
            overall["canonical_warm_regression_fraction"]
            <= float(gates_config["canonical_warm_regression_fraction_maximum"])
        ),
        "canonical_nn_target_sigma_rms_median_le_0p80": bool(
            canonical_nn_median
            <= float(gates_config["canonical_nn_target_sigma_rms_median_maximum"])
        ),
        "nn_target_distance_reduction_vs_old_fullrank_ge_0p15": bool(
            distance_reduction
            >= float(
                gates_config[
                    "nn_target_distance_reduction_vs_old_fullrank_minimum"
                ]
            )
        ),
        "canonical_input_target_spearman_median_ge_0p58": bool(
            np.median(data["canonical_input_target_spearman"])
            >= float(
                gates_config["canonical_input_target_spearman_median_minimum"]
            )
        ),
        "final_stage_share_of_total_search_gain_le_0p10": bool(
            final_stage_share
            <= float(gates_config["final_stage_share_of_total_search_gain_maximum"])
        ),
        "nearest_neighbor_target_transfer_recovery_vs_hold_ge_0p25": bool(
            overall["nearest_target_transfer_recovery_vs_hold"]
            >= float(
                gates_config[
                    "nearest_neighbor_target_transfer_recovery_vs_hold_minimum"
                ]
            )
        ),
    }
    decision = (
        "CANONICAL_TEACHER_AUDIT_READY_FOR_SMALL_ACTOR"
        if all(gates.values())
        else "CANONICAL_TEACHER_AUDIT_FAIL_DO_NOT_SCALE"
    )
    summary = {
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "decision": decision,
        "row_count": row_count,
        "episode_count": int(len(np.unique(data["episode_id"]))),
        "candidate_count_per_state": candidate_count,
        "canonical_search_query_rollouts": row_count * candidate_count,
        "nearest_target_transfer_query_rollouts": row_count,
        "total_new_query_rollouts": row_count * (candidate_count + 1),
        "overall": overall,
        "stage_best_cost": {
            str(stage): stats(data["stage_best_cost"][:, stage])
            for stage in range(len(radii) + 1)
        },
        "final_stage_share_of_total_search_gain": final_stage_share,
        "nn_target_distance_reduction_vs_old_fullrank": distance_reduction,
        "pre_registered_gates": gates,
        "by_speed_kph": {
            str(speed): grouped_metrics(data, data["speed_kph"] == speed)
            for speed in sorted(np.unique(data["speed_kph"]))
        },
        "by_variant_index": {
            str(variant): grouped_metrics(data, data["variant_index"] == variant)
            for variant in sorted(np.unique(data["variant_index"]))
        },
        "by_fold": {
            str(fold): grouped_metrics(data, data["fold_id"] == fold)
            for fold in range(5)
        },
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
    }

    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.json")
    audit_path = output / "audit.npz"
    np.savez_compressed(audit_path, **data)
    summary_path = output / "summary.json"
    dump_json(summary_path, summary)
    with (output / "rows.csv").open("w", newline="") as stream:
        fields = [
            "source_row_index",
            "episode_id",
            "control_step",
            "speed_kph",
            "variant_index",
            "fold_id",
            "canonical_hold_cost",
            "canonical_teacher_index",
            "canonical_teacher_cost",
            "old_warm_cost",
            "old_t0_cost",
            "old_fullrank_cost",
            "nearest_fit_row",
            "canonical_nearest_target_sigma_rms",
            "old_fullrank_nearest_target_sigma_rms",
            "canonical_input_target_spearman",
            "nearest_target_transfer_cost",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in range(row_count):
            writer.writerow(
                {name: np.asarray(data[name][row]).item() for name in fields}
            )
    manifest = {
        "schema_version": "query-noanchor-canonical-teacher-audit-v1",
        "dataset_type": "train-only-pure-query-noanchor-canonical-search-audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "config_sha256": sha256(config_path),
        "source_sidecar": str(source),
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_validation_sha256": sha256(source_validation_path),
        "source_bank_sha256": sha256(source_bank_path),
        "query_checkpoint": source_manifest["query_checkpoint"],
        "query_checkpoint_sha256": source_manifest["query_checkpoint_sha256"],
        "audit_sha256": sha256(audit_path),
        "summary_sha256": sha256(summary_path),
        "rows_sha256": sha256(output / "rows.csv"),
        "row_count": row_count,
        "candidate_count_per_state": candidate_count,
        "total_new_query_rollouts": row_count * (candidate_count + 1),
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "forbidden_center_attestation": (
            "Candidate construction used only current_action, fixed direction banks, "
            "fixed radii, action bounds, and prior canonical Query argmin."
        ),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "limitations": [
            "This is a 100-state mechanism audit, not a full teacher dataset.",
            "The finite staged search is a best-found canonical teacher, not a global optimum.",
            "Old warm/T0/full-rank values are comparison-only and never enter the canonical argmin.",
            "No Actor is trained unless every pre-registered search/coherence/transfer gate passes.",
        ],
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({"decision": decision, "gates": gates, "overall": overall}, indent=2))
    print(f"output: {output}")


if __name__ == "__main__":
    main()
