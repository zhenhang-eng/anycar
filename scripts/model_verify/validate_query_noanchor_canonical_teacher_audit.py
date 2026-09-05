#!/usr/bin/env python3
"""Independently validate the no-anchor canonical Query teacher audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "query_noanchor_canonical_teacher_audit_20260901_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=DEFAULT_SOURCE)
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


def hadamard() -> np.ndarray:
    value = np.ones((1, 1), np.float64)
    while value.shape[0] != 16:
        value = np.block([[value, value], [value, -value]])
    return value


def dct() -> np.ndarray:
    n = np.arange(16, dtype=np.float64)
    k = np.arange(16, dtype=np.float64)[:, None]
    value = np.cos(np.pi * (n[None] + 0.5) * k / 16.0)
    value[0] /= 4.0
    value[1:] *= math.sqrt(1.0 / 8.0)
    return value * 4.0


def directions() -> np.ndarray:
    value = np.concatenate((hadamard(), dct())).reshape(32, 8, 2)
    return value.astype(np.float32)


def interpolate(knots: np.ndarray) -> np.ndarray:
    value = torch.from_numpy(np.asarray(knots, np.float32))
    return (
        F.interpolate(
            value.transpose(1, 2), size=50, mode="linear", align_corners=True
        )
        .transpose(1, 2)
        .numpy()
    )


def robust(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    scale = q75 - q25
    standard = np.std(values, axis=0)
    scale[(scale <= 1e-8) & (standard > 1e-8)] = standard[
        (scale <= 1e-8) & (standard > 1e-8)
    ]
    scale[scale <= 1e-8] = 1.0
    return center, scale


def embed(blocks: list[np.ndarray], rows: np.ndarray, fit: np.ndarray) -> np.ndarray:
    output = []
    for block in blocks:
        center, scale = robust(block[fit])
        output.append((block[rows] - center) / scale / np.sqrt(block.shape[1]))
    return np.concatenate(output, axis=1)


def distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None]
        - 2.0 * left @ right.T
    )
    return np.sqrt(np.maximum(squared, 0.0))


def recompute_coherence(
    data: dict[str, np.ndarray], sigma: np.ndarray
) -> dict[str, np.ndarray]:
    count = len(data["state"])
    blocks = [
        data["history"].reshape(count, -1).astype(np.float64),
        data["reference_ego"].reshape(count, -1).astype(np.float64),
        np.concatenate((data["state"][:, 3:5], data["current_action"]), axis=1)
        .astype(np.float64),
    ]
    nearest = np.empty(count, np.int64)
    input_nearest = np.empty(count, np.float64)
    canonical_nearest = np.empty(count, np.float64)
    old_nearest = np.empty(count, np.float64)
    canonical_rank = np.empty(count, np.float64)
    old_rank = np.empty(count, np.float64)
    sigma_flat = np.tile(sigma.astype(np.float64), 8)
    canonical = data["canonical_teacher_knots"].reshape(count, 16) / sigma_flat
    old = data["old_fullrank_knots"].reshape(count, 16) / sigma_flat
    fold_id = data["fold_id"].astype(np.int64)
    for fold in range(5):
        query = np.flatnonzero(fold_id == fold)
        fit = np.flatnonzero((fold_id != fold) & (fold_id != (fold + 1) % 5))
        input_distance = distances(embed(blocks, query, fit), embed(blocks, fit, fit))
        canonical_distance = np.sqrt(
            np.mean(np.square(canonical[query, None] - canonical[fit][None]), axis=2)
        )
        old_distance = np.sqrt(
            np.mean(np.square(old[query, None] - old[fit][None]), axis=2)
        )
        local_nearest = np.argmin(input_distance, axis=1)
        nearest[query] = fit[local_nearest]
        input_nearest[query] = input_distance[np.arange(len(query)), local_nearest]
        canonical_nearest[query] = canonical_distance[
            np.arange(len(query)), local_nearest
        ]
        old_nearest[query] = old_distance[np.arange(len(query)), local_nearest]
        for local, row in enumerate(query):
            canonical_rank[row] = spearmanr(
                input_distance[local], canonical_distance[local]
            ).statistic
            old_rank[row] = spearmanr(
                input_distance[local], old_distance[local]
            ).statistic
    return {
        "nearest_fit_row": nearest,
        "nearest_input_distance": input_nearest,
        "canonical_nearest_target_sigma_rms": canonical_nearest,
        "old_fullrank_nearest_target_sigma_rms": old_nearest,
        "canonical_input_target_spearman": canonical_rank,
        "old_fullrank_input_target_spearman": old_rank,
    }


def max_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    manifest_path = source / "manifest.json"
    summary_path = source / "summary.json"
    config_path = source / "config.json"
    audit_path = source / "audit.npz"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    config = json.loads(config_path.read_text())
    if manifest["schema_version"] != "query-noanchor-canonical-teacher-audit-v1":
        raise AssertionError("unexpected schema")
    if manifest.get("formal_validation_or_test_consumed", True):
        raise AssertionError("audit consumed sealed splits")
    if manifest.get("dbm_fields_or_labels_consumed"):
        raise AssertionError("audit reports DBM fields/labels")
    checks: dict[str, bool] = {
        "config_hash": sha256(config_path) == manifest["config_sha256"],
        "audit_hash": sha256(audit_path) == manifest["audit_sha256"],
        "summary_hash": sha256(summary_path) == manifest["summary_sha256"],
        "rows_hash": sha256(source / "rows.csv") == manifest["rows_sha256"],
        "query_checkpoint_hash": (
            sha256(Path(manifest["query_checkpoint"]))
            == manifest["query_checkpoint_sha256"]
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"artifact hash failure: {checks}")
    source_sidecar = Path(manifest["source_sidecar"])
    checks["source_manifest_hash"] = (
        sha256(source_sidecar / "manifest.json") == manifest["source_manifest_sha256"]
    )
    checks["source_validation_hash"] = (
        sha256(source_sidecar / "validation.json")
        == manifest["source_validation_sha256"]
    )
    checks["source_bank_hash"] = (
        sha256(source_sidecar / "bank.npz") == manifest["source_bank_sha256"]
    )
    with np.load(audit_path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(source_sidecar / "bank.npz", allow_pickle=False) as archive:
        source_rows = data["source_row_index"].astype(np.int64)
        source_field_errors = {
            name: max_error(data[name], archive[name][source_rows])
            for name in (
                "row_index",
                "row_in_episode",
                "control_step",
                "speed_kph",
                "variant_index",
                "fold_id",
                "state",
                "current_action",
                "history",
                "reference",
                "reference_ego",
            )
        }
        source_exact_strings = all(
            np.array_equal(data[name], archive[name][source_rows])
            for name in ("episode_id", "road_name", "source_snapshot_sha256")
        )
        comparator_errors = {
            "old_warm_knots": max_error(
                data["old_warm_knots"], archive["mean_knots_before"][source_rows]
            ),
            "old_t0_knots": max_error(
                data["old_t0_knots"], archive["teacher_knots"][source_rows]
            ),
            "old_fullrank_knots": max_error(
                data["old_fullrank_knots"],
                archive["fullrank_teacher_knots"][source_rows],
            ),
            "old_warm_cost": max_error(
                data["old_warm_cost"], archive["warm_direct_cost_replayed"][source_rows]
            ),
            "old_t0_cost": max_error(
                data["old_t0_cost"], archive["t0_teacher_direct_cost_replayed"][source_rows]
            ),
            "old_fullrank_cost": max_error(
                data["old_fullrank_cost"],
                archive["fullrank_teacher_direct_cost"][source_rows],
            ),
        }
    checks["source_context_reconstruction"] = (
        max(source_field_errors.values()) == 0.0 and source_exact_strings
    )
    checks["comparison_field_reconstruction"] = max(comparator_errors.values()) == 0.0

    sigma = np.asarray(config["noise_sigma"], np.float32)
    low = np.asarray(config["action_bounds"]["minimum"], np.float32)
    high = np.asarray(config["action_bounds"]["maximum"], np.float32)
    bank = directions()
    raw_reconstructed = np.empty_like(data["candidate_raw_knots"])
    clipped_reconstructed = np.empty_like(data["candidate_knots"])
    initial = np.broadcast_to(data["current_action"][:, None], (len(data["state"]), 8, 2))
    raw_reconstructed[:, 0] = initial
    clipped_reconstructed[:, 0] = np.clip(initial, low, high)
    per_stage = int(config["candidates_per_stage"])
    radii = [float(value) for value in config["stage_radii_sigma"]]
    stage_index_reconstructed = np.empty_like(data["stage_best_global_index"])
    stage_cost_reconstructed = np.empty_like(data["stage_best_cost"])
    stage_index_reconstructed[:, 0] = 0
    stage_cost_reconstructed[:, 0] = data["candidate_cost"][:, 0]
    for row in range(len(data["state"])):
        incumbent_index = 0
        incumbent_cost = float(data["candidate_cost"][row, 0])
        for stage, radius in enumerate(radii):
            start = 1 + stage * per_stage
            incumbent = clipped_reconstructed[row, incumbent_index]
            raw = np.stack(
                [
                    incumbent + sign * radius * direction * sigma[None]
                    for direction in bank
                    for sign in (1.0, -1.0)
                ]
            ).astype(np.float32)
            local = np.clip(raw, low, high)
            raw_reconstructed[row, start : start + per_stage] = raw
            clipped_reconstructed[row, start : start + per_stage] = local
            local_cost = data["candidate_cost"][row, start : start + per_stage]
            local_best = int(np.argmin(local_cost))
            if float(local_cost[local_best]) < incumbent_cost:
                incumbent_index = start + local_best
                incumbent_cost = float(local_cost[local_best])
            stage_index_reconstructed[row, stage + 1] = incumbent_index
            stage_cost_reconstructed[row, stage + 1] = incumbent_cost
    candidate_errors = {
        "raw": max_error(raw_reconstructed, data["candidate_raw_knots"]),
        "clipped": max_error(clipped_reconstructed, data["candidate_knots"]),
        "clip_mask": float(
            np.max(
                (np.abs(raw_reconstructed - clipped_reconstructed) > 1e-7)
                != data["candidate_clipped_mask"]
            )
        ),
        "stage_index": max_error(
            stage_index_reconstructed, data["stage_best_global_index"]
        ),
        "stage_cost": max_error(stage_cost_reconstructed, data["stage_best_cost"]),
    }
    checks["allowed_input_candidate_reconstruction"] = max(candidate_errors.values()) == 0.0
    rows = np.arange(len(data["state"]))
    teacher_index = data["stage_best_global_index"][:, -1]
    checks["canonical_argmin_reconstruction"] = bool(
        np.array_equal(teacher_index, data["canonical_teacher_index"])
        and np.array_equal(
            data["canonical_teacher_knots"],
            data["candidate_knots"][rows, teacher_index],
        )
        and np.array_equal(
            data["canonical_teacher_cost"],
            data["candidate_cost"][rows, teacher_index],
        )
    )
    checks["stage_monotonicity"] = bool(
        np.all(np.diff(data["stage_best_cost"], axis=1) <= 1e-7)
    )

    coherence = recompute_coherence(data, sigma)
    coherence_errors = {
        name: max_error(value, data[name]) for name, value in coherence.items()
    }
    checks["nested_coherence_reconstruction"] = max(coherence_errors.values()) < 1e-9

    parent_t0 = Path(json.loads((source_sidecar / "manifest.json").read_text())["parent_t0"])
    parent_manifest = json.loads((parent_t0 / "manifest.json").read_text())
    collection_manifest = json.loads(
        (Path(parent_manifest["source_collection"]) / "manifest.json").read_text()
    )
    params = TorchMPPIParams(**collection_manifest["collection"]["mppi"])
    device = torch.device(args.device)
    model = QueryDeploymentModel.from_checkpoint(Path(manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(model), params, device=str(device)
    )
    # One row for every speed/fold cell: 25 rows, all 257 candidates each.
    replay_rows = []
    for speed in sorted(np.unique(data["speed_kph"])):
        for fold in range(5):
            choices = np.flatnonzero(
                (data["speed_kph"] == speed) & (data["fold_id"] == fold)
            )
            replay_rows.append(int(choices[0]))
    replay_rows = np.asarray(replay_rows, np.int64)
    replay_candidate_error = 0.0
    replay_teacher_error = 0.0
    for position, row in enumerate(replay_rows):
        # Preserve the generator's frozen 1+64+64+64+64 batch boundary.
        # Query/CUDA reductions can differ slightly if all 257 candidates are
        # fused into one inference batch, especially for very large tail costs.
        replayed = np.empty(data["candidate_cost"].shape[1], np.float32)
        boundaries = [(0, 1)] + [
            (1 + stage * per_stage, 1 + (stage + 1) * per_stage)
            for stage in range(len(radii))
        ]
        for start, stop in boundaries:
            result = controller.evaluate_action_sequences(
                data["state"][row],
                data["current_action"][row],
                data["history"][row : row + 1],
                data["reference"][row],
                interpolate(data["candidate_knots"][row, start:stop]),
            )
            replayed[start:stop] = result["cost"].cpu().numpy()
        replay_candidate_error = max(
            replay_candidate_error,
            max_error(replayed, data["candidate_cost"][row]),
        )
        replay_teacher_error = max(
            replay_teacher_error,
            abs(
                float(replayed[data["canonical_teacher_index"][row]])
                - float(data["canonical_teacher_cost"][row])
            ),
        )
        if (position + 1) % 5 == 0:
            print(f"candidate replay {position + 1}/{len(replay_rows)}", flush=True)
    replay_transfer_error = 0.0
    for row in range(len(data["state"])):
        result = controller.evaluate_action_sequences(
            data["state"][row],
            data["current_action"][row],
            data["history"][row : row + 1],
            data["reference"][row],
            interpolate(data["nearest_target_transfer_knots"][row : row + 1]),
        )
        replay_transfer_error = max(
            replay_transfer_error,
            abs(float(result["cost"][0].cpu()) - float(data["nearest_target_transfer_cost"][row])),
        )
    numerical_tolerance = 5e-4
    checks["candidate_query_replay"] = replay_candidate_error <= numerical_tolerance
    checks["teacher_query_replay"] = replay_teacher_error <= numerical_tolerance
    checks["nearest_target_transfer_query_replay"] = (
        replay_transfer_error <= numerical_tolerance
    )
    checks["decision_reconstruction"] = (
        summary["decision"]
        == (
            "CANONICAL_TEACHER_AUDIT_READY_FOR_SMALL_ACTOR"
            if all(summary["pre_registered_gates"].values())
            else "CANONICAL_TEACHER_AUDIT_FAIL_DO_NOT_SCALE"
        )
        == manifest["decision"]
    )
    qualification = (
        "QUERY_NOANCHOR_CANONICAL_TEACHER_AUDIT_INDEPENDENT_PASS"
        if all(checks.values())
        else "QUERY_NOANCHOR_CANONICAL_TEACHER_AUDIT_INDEPENDENT_FAIL"
    )
    validation = {
        "qualification": qualification,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "source_field_max_errors": source_field_errors,
        "comparison_field_max_errors": comparator_errors,
        "candidate_reconstruction_max_errors": candidate_errors,
        "coherence_reconstruction_max_errors": coherence_errors,
        "query_replay": {
            "candidate_rows": replay_rows.tolist(),
            "candidate_rollouts": int(len(replay_rows) * data["candidate_cost"].shape[1]),
            "candidate_cost_max_error": replay_candidate_error,
            "teacher_cost_max_error": replay_teacher_error,
            "nearest_target_transfer_rollouts": int(len(data["state"])),
            "nearest_target_transfer_cost_max_error": replay_transfer_error,
            "tolerance": numerical_tolerance,
        },
        "experimental_decision": summary["decision"],
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
