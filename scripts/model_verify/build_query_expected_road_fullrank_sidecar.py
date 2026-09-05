#!/usr/bin/env python3
"""Build a T0-centered full-rank proximal direct-cost Query sidecar."""

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
    "scripts/model_verify/query_expected_road_fullrank_config_20260901_v1.json"
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
    "seed_row",
    "seed_source",
    "seed_source_file",
    "seed_source_window_index",
    "source_snapshot_sha256",
    "state",
    "current_action",
    "history",
    "reference",
    "reference_ego",
    "mean_knots_before",
    "best_candidate_index",
    "warm_direct_cost",
    "best_sampled_direct_cost",
    "weighted_output_direct_cost",
    "teacher_source_index",
    "teacher_source_name",
    "teacher_knots",
    "teacher_action_sequence",
    "teacher_trajectory",
    "teacher_direct_cost",
    "teacher_delta_knots",
    "teacher_gain_vs_warm",
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
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def hadamard_16() -> np.ndarray:
    matrix = np.ones((1, 1), dtype=np.float32)
    while len(matrix) < 16:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix.reshape(16, 8, 2)


def interpolate_knots(knots: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(knots, dtype=torch.float32)
    sequence = F.interpolate(
        tensor.transpose(1, 2), size=50, mode="linear", align_corners=True
    ).transpose(1, 2)
    return sequence.numpy()


def wrapped(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def raw_features(
    trajectories: np.ndarray,
    actions: np.ndarray,
    reference: np.ndarray,
    current_action: np.ndarray,
) -> dict[str, np.ndarray]:
    target = reference[1:]
    previous = np.concatenate(
        (np.broadcast_to(current_action, (len(actions), 1, 2)), actions[:, :-1]),
        axis=1,
    )
    return {
        "feature_position_error_sq": np.square(
            trajectories[..., :2] - target[None, :, :2]
        ).sum(axis=-1),
        "feature_yaw_error_sq": np.square(
            wrapped(trajectories[..., 2] - target[None, :, 2])
        ),
        "feature_vx_error_sq": np.square(
            trajectories[..., 3] - target[None, :, 3]
        ),
        "feature_action_rate_sq": np.square(actions - previous),
    }


def candidate_contract(config: dict) -> dict[str, np.ndarray]:
    names = list(config["base_candidates"])
    group = ["baseline"] * len(names)
    radius = [0.0] * len(names)
    direction = [-1] * len(names)
    sign = [0] * len(names)
    pair_indices = np.empty(
        (len(config["probe_radii_sigma"]), config["num_directions"], 2),
        dtype=np.int64,
    )
    for radius_index, value in enumerate(config["probe_radii_sigma"]):
        for direction_index in range(config["num_directions"]):
            plus = len(names)
            minus = plus + 1
            pair_indices[radius_index, direction_index] = (plus, minus)
            tag = str(value).replace(".", "p")
            names.extend(
                (
                    f"prox_r{tag}_d{direction_index:02d}_plus",
                    f"prox_r{tag}_d{direction_index:02d}_minus",
                )
            )
            group.extend(("proximal", "proximal"))
            radius.extend((value, value))
            direction.extend((direction_index, direction_index))
            sign.extend((1, -1))
    if len(names) != config["total_candidates"]:
        raise AssertionError("candidate count disagrees with frozen config")
    return {
        "candidate_name": np.asarray(names, dtype=str),
        "candidate_group": np.asarray(group, dtype=str),
        "candidate_radius_sigma": np.asarray(radius, dtype=np.float32),
        "candidate_direction_index": np.asarray(direction, dtype=np.int16),
        "candidate_sign": np.asarray(sign, dtype=np.int8),
        "pair_candidate_indices": pair_indices,
    }


def grouped_metrics(data: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    proximal = data["candidate_group"] == "proximal"
    gains = data["candidate_gain_vs_t0"][:, proximal][mask]
    return {
        "rows": int(mask.sum()),
        "warm_direct_cost": stats(data["warm_direct_cost_replayed"][mask]),
        "t0_teacher_direct_cost": stats(data["t0_teacher_direct_cost_replayed"][mask]),
        "fullrank_teacher_direct_cost": stats(data["fullrank_teacher_direct_cost"][mask]),
        "fullrank_gain_vs_t0": stats(data["fullrank_gain_vs_t0"][mask]),
        "fullrank_gain_vs_warm": stats(data["fullrank_gain_vs_warm"][mask]),
        "fullrank_strict_win_vs_t0_fraction": float(
            np.mean(data["fullrank_gain_vs_t0"][mask] > 0)
        ),
        "fullrank_warm_floor_fraction": float(
            np.mean(data["fullrank_teacher_index"][mask] == 0)
        ),
        "fullrank_t0_floor_fraction": float(
            np.mean(data["fullrank_teacher_index"][mask] == 1)
        ),
        "proximal_candidate_positive_fraction": float(np.mean(gains > 0)),
        "proximal_candidate_negative_fraction": float(np.mean(gains < 0)),
        "candidate_clip_fraction": stats(data["candidate_clip_fraction"][mask]),
        "fullrank_delta_from_t0_sigma_rms": stats(
            data["fullrank_delta_from_t0_sigma_rms"][mask]
        ),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    parent = Path(config["parent_t0"]).resolve()
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    parent_manifest_path = parent / "manifest.json"
    parent_validation_path = parent / "validation.json"
    parent_replay_path = parent / "replay.npz"
    parent_manifest = json.loads(parent_manifest_path.read_text())
    parent_validation = json.loads(parent_validation_path.read_text())
    if parent_validation["qualification"] != "QUERY_EXPECTED_ROAD_T0_REPLAY_PASS":
        raise AssertionError("parent T0 sidecar did not pass")
    if parent_manifest.get("formal_validation_or_test_consumed", True):
        raise AssertionError("parent T0 consumed formal validation/test")
    if config["formal_validation_or_test_consumed"]:
        raise AssertionError("full-rank config may not consume sealed splits")
    if config.get("query_analytic_gradient_consumed", False):
        raise AssertionError("full-rank config may not consume Query analytic gradients")
    if config["total_candidates"] != 4 + 2 * 16 * len(config["probe_radii_sigma"]):
        raise AssertionError("invalid candidate-count contract")

    contract = candidate_contract(config)
    sigma = np.asarray(config["noise_sigma"], dtype=np.float32)
    directions = hadamard_16() * sigma.reshape(1, 1, 2)
    candidate_count = int(config["total_candidates"])
    with np.load(parent_replay_path, allow_pickle=False) as replay:
        row_count = len(replay["state"])
        data = {name: np.asarray(replay[name]) for name in CONTEXT_FIELDS}
        raw_knots = np.empty((row_count, candidate_count, 8, 2), dtype=np.float32)
        knots = np.empty_like(raw_knots)
        actions = np.empty((row_count, candidate_count, 50, 2), dtype=np.float32)
        trajectories = np.empty((row_count, candidate_count, 50, 5), dtype=np.float32)
        costs = np.empty((row_count, candidate_count), dtype=np.float32)
        components = {
            name: np.empty((row_count, candidate_count), dtype=np.float32)
            for name in ("position", "yaw", "vx", "acceleration_rate", "steering_rate")
        }
        features = {
            "feature_position_error_sq": np.empty(
                (row_count, candidate_count, 50), dtype=np.float32
            ),
            "feature_yaw_error_sq": np.empty(
                (row_count, candidate_count, 50), dtype=np.float32
            ),
            "feature_vx_error_sq": np.empty(
                (row_count, candidate_count, 50), dtype=np.float32
            ),
            "feature_action_rate_sq": np.empty(
                (row_count, candidate_count, 50, 2), dtype=np.float32
            ),
        }

        source_manifest = json.loads(
            (Path(parent_manifest["source_collection"]) / "manifest.json").read_text()
        )
        params = TorchMPPIParams(**source_manifest["collection"]["mppi"])
        model = QueryDeploymentModel.from_checkpoint(
            Path(parent_manifest["query_checkpoint"]), args.device
        )
        controller = TorchMPPIController(
            TorchQueryRolloutBackend(model), params, device=args.device
        )
        for row in range(row_count):
            base = np.stack(
                (
                    replay["mean_knots_before"][row],
                    replay["teacher_knots"][row],
                    replay["candidate_center_knots"][row, 1],
                    replay["candidate_center_knots"][row, 2],
                )
            )
            values = [base]
            anchor = replay["teacher_knots"][row]
            for radius in config["probe_radii_sigma"]:
                paired = np.stack(
                    [
                        anchor + sign * float(radius) * direction
                        for direction in directions
                        for sign in (1.0, -1.0)
                    ]
                )
                values.append(paired)
            raw = np.concatenate(values, axis=0).astype(np.float32)
            clipped = np.clip(raw, config["action_bounds"]["minimum"], config["action_bounds"]["maximum"])
            action = interpolate_knots(clipped)
            result = controller.evaluate_action_sequences(
                replay["state"][row],
                replay["current_action"][row],
                replay["history"][row : row + 1],
                replay["reference"][row],
                action,
            )
            trajectory = result["trajectories"].cpu().numpy()
            raw_knots[row] = raw
            knots[row] = clipped
            actions[row] = action
            trajectories[row] = trajectory
            costs[row] = result["cost"].cpu().numpy()
            for name in components:
                components[name][row] = result["cost_components"][name].cpu().numpy()
            for name, value in raw_features(
                trajectory,
                action,
                replay["reference"][row],
                replay["current_action"][row],
            ).items():
                features[name][row] = value

    data.update(contract)
    data.update(
        {
            "raw_candidate_knots": raw_knots,
            "candidate_knots": knots,
            "candidate_action_sequences": actions,
            "candidate_trajectories": trajectories,
            "candidate_cost": costs,
            "candidate_clipped_mask": np.abs(raw_knots - knots) > 1e-7,
        }
    )
    for name, value in components.items():
        data[f"cost_component_{name}"] = value
    data.update(features)

    pair_indices = contract["pair_candidate_indices"]
    pair_cost = costs[:, pair_indices]
    pair_clipped = data["candidate_clipped_mask"][:, pair_indices]
    data["pair_cost"] = pair_cost
    data["pair_cost_difference_minus_minus_plus"] = pair_cost[..., 1] - pair_cost[..., 0]
    data["pair_symmetric_mask"] = ~np.any(pair_clipped, axis=(3, 4, 5))
    t0_cost = costs[:, 1].astype(np.float64)
    candidate_gain = t0_cost[:, None] - costs.astype(np.float64)
    data["candidate_gain_vs_t0"] = candidate_gain
    data["candidate_clip_fraction"] = np.mean(
        data["candidate_clipped_mask"], axis=(1, 2, 3)
    )

    local_rank = np.empty((row_count, len(config["probe_radii_sigma"])), dtype=np.int16)
    local_condition = np.empty_like(local_rank, dtype=np.float32)
    for row in range(row_count):
        anchor = knots[row, 1]
        for radius_index in range(len(config["probe_radii_sigma"])):
            indices = pair_indices[radius_index].reshape(-1)
            matrix = ((knots[row, indices] - anchor) / sigma).reshape(32, 16)
            singular = np.linalg.svd(matrix.astype(np.float64), compute_uv=False)
            local_rank[row, radius_index] = np.linalg.matrix_rank(matrix, tol=1e-7)
            local_condition[row, radius_index] = float(
                singular[0] / singular[-1] if singular[-1] > 1e-12 else np.inf
            )
    data["local_rank_by_radius"] = local_rank
    data["local_condition_by_radius"] = local_condition

    teacher_index = np.argmin(costs.astype(np.float64), axis=1).astype(np.int64)
    row = np.arange(row_count)
    teacher_knots = knots[row, teacher_index]
    teacher_cost = costs[row, teacher_index].astype(np.float64)
    t0_knots = knots[:, 1]
    warm_cost = costs[:, 0].astype(np.float64)
    data.update(
        {
            "warm_direct_cost_replayed": warm_cost,
            "t0_teacher_direct_cost_replayed": t0_cost,
            "fullrank_teacher_index": teacher_index,
            "fullrank_teacher_name": contract["candidate_name"][teacher_index],
            "fullrank_teacher_knots": teacher_knots,
            "fullrank_teacher_action_sequence": actions[row, teacher_index],
            "fullrank_teacher_trajectory": trajectories[row, teacher_index],
            "fullrank_teacher_direct_cost": teacher_cost,
            "fullrank_gain_vs_t0": t0_cost - teacher_cost,
            "fullrank_gain_vs_warm": warm_cost - teacher_cost,
            "fullrank_delta_from_t0_knots": teacher_knots - t0_knots,
            "fullrank_delta_from_t0_sigma_rms": np.sqrt(
                np.mean(np.square((teacher_knots - t0_knots) / sigma), axis=(1, 2))
            ),
            "fullrank_delta_from_warm_knots": teacher_knots - knots[:, 0],
        }
    )
    if np.any(data["fullrank_gain_vs_t0"] < -1e-10):
        raise AssertionError("full-rank teacher regressed against T0")
    if np.any(data["fullrank_gain_vs_warm"] < -1e-10):
        raise AssertionError("full-rank teacher regressed against warm")

    mask_all = np.ones(row_count, dtype=bool)
    summary = {
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "row_count": row_count,
        "candidate_count_per_row": candidate_count,
        "new_query_rollouts": row_count * candidate_count,
        "overall": grouped_metrics(data, mask_all),
        "local_rank_min": int(local_rank.min()),
        "local_condition": stats(local_condition),
        "pair_symmetric_fraction": float(np.mean(data["pair_symmetric_mask"])),
        "teacher_source_counts": {
            str(name): int(np.sum(data["fullrank_teacher_name"] == name))
            for name in np.unique(data["fullrank_teacher_name"])
        },
        "by_speed_kph": {
            str(value): grouped_metrics(data, data["speed_kph"] == value)
            for value in sorted(np.unique(data["speed_kph"]))
        },
        "by_variant_index": {
            str(value): grouped_metrics(data, data["variant_index"] == value)
            for value in sorted(np.unique(data["variant_index"]))
        },
        "by_fold": {
            str(value): grouped_metrics(data, data["fold_id"] == value)
            for value in sorted(np.unique(data["fold_id"]).tolist())
        },
        "formal_validation_or_test_consumed": False,
        "query_analytic_gradient_consumed": False,
    }

    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.json")
    bank_path = output / "bank.npz"
    np.savez_compressed(bank_path, **data)
    shutil.copy2(parent / "splits.json", output / "splits.json")
    dump_json(output / "summary.json", summary)
    with (output / "rows.csv").open("w", newline="") as stream:
        fields = [
            "row_index",
            "episode_id",
            "control_step",
            "speed_kph",
            "variant_index",
            "fold_id",
            "warm_direct_cost_replayed",
            "t0_teacher_direct_cost_replayed",
            "fullrank_teacher_index",
            "fullrank_teacher_name",
            "fullrank_teacher_direct_cost",
            "fullrank_gain_vs_t0",
            "fullrank_gain_vs_warm",
            "fullrank_delta_from_t0_sigma_rms",
            "candidate_clip_fraction",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(row_count):
            writer.writerow({name: np.asarray(data[name][index]).item() for name in fields})

    manifest = {
        "schema_version": "query-expected-road-fullrank-v1",
        "dataset_type": "pure-query-t0-centered-fullrank-direct-cost-sidecar",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256(config_path),
        "bank_contract": config,
        "candidate_contract": {
            name: value.tolist() for name, value in contract.items()
        },
        "parent_t0": str(parent),
        "parent_manifest_sha256": sha256(parent_manifest_path),
        "parent_validation_sha256": sha256(parent_validation_path),
        "parent_replay_sha256": sha256(parent_replay_path),
        "query_checkpoint": parent_manifest["query_checkpoint"],
        "query_checkpoint_sha256": parent_manifest["query_checkpoint_sha256"],
        "bank_sha256": sha256(bank_path),
        "rows_sha256": sha256(output / "rows.csv"),
        "splits_sha256": sha256(output / "splits.json"),
        "row_count": row_count,
        "candidate_count_per_row": candidate_count,
        "new_query_rollouts": row_count * candidate_count,
        "dbm_fields_or_labels_consumed": [],
        "formal_validation_or_test_consumed": False,
        "query_analytic_gradient_consumed": False,
        "limitations": [
            "This is a T0-centered proximal bank, not actor-visited Replay.",
            "The full-rank teacher is the best saved finite-bank candidate, not a global optimum.",
            "The 40-kph source retains the documented left-turn substitutions and is not direction-balanced.",
        ],
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"output: {output}")


if __name__ == "__main__":
    main()
