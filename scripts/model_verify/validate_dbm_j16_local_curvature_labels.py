#!/usr/bin/env python3
"""Independently reconstruct and replay J16 local forward-cost labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots, sha256_file
from generate_dbm_j16_local_curvature_labels import (
    ACTION_DIMENSION,
    FORMAT_VERSION,
    hadamard_directions,
    local_statistics,
    make_centers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.labels.resolve()
    summary = json.loads((root / "summary.json").read_text())
    if summary["split"] != "train" or "test" not in summary["test_policy"]:
        raise AssertionError("labels do not preserve train-only/test-sealed policy")
    source_root = Path(summary["source"])
    gt_root = Path(summary["gt_root"])
    if sha256_file(gt_root / "summary.json") != summary["gt_summary_sha256"]:
        raise AssertionError("GT summary hash mismatch")
    split_data = json.loads((root / "splits.json").read_text())
    allowed_episodes = set(split_data["train"])
    forbidden_episodes = set(split_data.get("validation", ())) | set(
        split_data.get("test", ())
    )
    if allowed_episodes & forbidden_episodes:
        raise AssertionError("train overlaps validation/test in split manifest")
    directions = hadamard_directions()
    radii = np.asarray(summary["radii_sigma"], np.float32)
    paths = sorted(root.glob("episode_*/*.npz"))
    if len(paths) != int(summary["snapshot_count"]):
        raise AssertionError("snapshot count mismatch")
    if not paths:
        raise ValueError("no local labels")

    records = []
    for path in paths:
        if path.parent.name not in allowed_episodes:
            raise AssertionError(f"unknown episode: {path.parent.name}")
        if path.parent.name in forbidden_episodes:
            raise AssertionError(f"held-out episode was generated: {path.parent.name}")
        with np.load(path, allow_pickle=False) as label:
            if int(label["format_version"]) != FORMAT_VERSION:
                raise AssertionError(f"format version: {path}")
            source_path = Path(str(label["source_snapshot"]))
            gt_path = Path(str(label["gt_result"]))
            if source_path.parent.parent.name != path.parent.name or source_path.name != path.name:
                raise AssertionError(f"source identity mismatch: {path}")
            if gt_path != gt_root / path.parent.name / path.name:
                raise AssertionError(f"GT identity mismatch: {path}")
            if sha256_file(source_path) != str(label["source_snapshot_sha256"]):
                raise AssertionError(f"source hash mismatch: {path}")
            if sha256_file(gt_path) != str(label["gt_result_sha256"]):
                raise AssertionError(f"GT hash mismatch: {path}")
            with np.load(source_path, allow_pickle=False) as source, np.load(
                gt_path, allow_pickle=False
            ) as gt:
                best = int(gt["knot_best_index"])
                base = np.asarray(label["base_j16_knots"], np.float32)
                if not np.array_equal(base, np.asarray(gt["optimized_knots"][best], np.float32)):
                    raise AssertionError(f"J16 base mismatch: {path}")
                if not np.array_equal(label["radii_sigma"], radii):
                    raise AssertionError(f"radii mismatch: {path}")
                if not np.array_equal(label["normalized_directions"], directions):
                    raise AssertionError(f"direction mismatch: {path}")
                params = json.loads(str(source["mppi_params_json"]))
                sigma = np.asarray(params["noise_sigma"], np.float32)
                raw, centers = make_centers(
                    base,
                    sigma,
                    radii,
                    directions,
                    np.asarray(params["action_min"], np.float32),
                    np.asarray(params["action_max"], np.float32),
                )
                if not np.array_equal(raw, label["raw_centers"]):
                    raise AssertionError(f"raw center mismatch: {path}")
                if not np.array_equal(centers, label["centers"]):
                    raise AssertionError(f"clipped center mismatch: {path}")
                reference = np.asarray(source["reference"], np.float32)
                if len(reference) == int(params["horizon"]) + 1:
                    reference = reference[1:]
                records.append({
                    "path": path,
                    "centers": centers,
                    "stored_cost": np.asarray(label["direct_cost"], np.float32),
                    "stored_slope": np.asarray(label["directional_slope"], np.float32),
                    "stored_curvature": np.asarray(label["directional_curvature"], np.float32),
                    "stored_symmetry": np.asarray(label["symmetric_pair_mask"], bool),
                    "stored_rank": np.asarray(label["local_direction_rank"], np.int32),
                    "base": base,
                    "sigma": sigma,
                    "initial_state_six": np.asarray(source["initial_state_six"], np.float32),
                    "current_action": np.asarray(source["current_action"], np.float32),
                    "reference": reference,
                    "mppi_params_json": str(source["mppi_params_json"]),
                    "cost_weights_json": str(source["cost_weights_json"]),
                    "dbm_params_json": str(source["dbm_params_json"]),
                    "clip": float(np.mean(raw != centers)),
                })
    for key in ("cost_weights_json", "dbm_params_json"):
        if len({record[key] for record in records}) != 1:
            raise AssertionError(f"{key} differs")
    frozen_mppi = []
    for record in records:
        value = json.loads(record["mppi_params_json"])
        value.pop("seed", None)
        frozen_mppi.append(value)
    if any(value != frozen_mppi[0] for value in frozen_mppi[1:]):
        raise AssertionError("non-seed MPPI parameters differ")

    device = torch.device(args.device)
    params = TorchMPPIParams(**json.loads(records[0]["mppi_params_json"]))
    weights = TorchMPPICostWeights(**json.loads(records[0]["cost_weights_json"]))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(records[0]["dbm_params_json"]))
    )
    maximum_cost_error = 0.0
    maximum_statistic_error = 0.0
    slopes_all, curvature_all, symmetry_all, ranks_all, clips = [], [], [], [], []
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        centers = torch.from_numpy(np.stack([record["centers"] for record in batch])).to(device)
        actions = interpolate_knots(centers, params.horizon)
        initial = torch.from_numpy(np.stack([record["initial_state_six"] for record in batch])).to(device)
        current = torch.from_numpy(np.stack([record["current_action"] for record in batch])).to(device)
        reference = torch.from_numpy(np.stack([record["reference"] for record in batch])).to(device)
        with torch.no_grad():
            replay = batched_cost(
                backend, weights, actions, initial, current, reference
            ).cpu().numpy().astype(np.float32)
        for local, record in enumerate(batch):
            maximum_cost_error = max(
                maximum_cost_error,
                float(np.max(np.abs(replay[local] - record["stored_cost"]))),
            )
            slope, curvature, symmetry, rank = local_statistics(
                record["base"], record["centers"], replay[local],
                record["sigma"], radii,
            )
            maximum_statistic_error = max(
                maximum_statistic_error,
                float(np.max(np.abs(slope - record["stored_slope"]))),
                float(np.max(np.abs(curvature - record["stored_curvature"]))),
            )
            if not np.array_equal(symmetry, record["stored_symmetry"]):
                raise AssertionError(f"symmetry mismatch: {record['path']}")
            if not np.array_equal(rank, record["stored_rank"]):
                raise AssertionError(f"rank mismatch: {record['path']}")
            slopes_all.append(slope)
            curvature_all.append(curvature)
            symmetry_all.append(symmetry)
            ranks_all.append(rank)
            clips.append(record["clip"])
        done = min(start + len(batch), len(records))
        print(f"[{done:04d}/{len(records):04d}] replay max_err={maximum_cost_error:.3g}", flush=True)
    slopes_array = np.asarray(slopes_all)
    curvature_array = np.asarray(curvature_all)
    symmetry_array = np.asarray(symmetry_all)
    ranks_array = np.asarray(ranks_all)
    checks = {
        "center_clip_fraction_mean": float(np.mean(clips)),
        "symmetric_pair_fraction": float(np.mean(symmetry_array)),
        "full_rank_fraction": float(np.mean(ranks_array == ACTION_DIMENSION)),
        "absolute_directional_slope_mean": float(np.mean(np.abs(slopes_array))),
        "positive_directional_curvature_fraction": float(np.mean(curvature_array > 0)),
    }
    for name, value in checks.items():
        if not np.isclose(value, float(summary[name]), rtol=1e-6, atol=1e-6):
            raise AssertionError(f"summary mismatch: {name}")
    result = {
        "status": "ok",
        "snapshots": len(records),
        "maximum_cost_replay_error": maximum_cost_error,
        "maximum_local_statistic_error": maximum_statistic_error,
        **checks,
        "test_policy": "validation and test splits not loaded or evaluated",
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
