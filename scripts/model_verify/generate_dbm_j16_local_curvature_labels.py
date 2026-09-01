#!/usr/bin/env python3
"""Generate train-only forward-cost neighborhoods around frozen J16 oracles.

The labels use a deterministic 16-D Hadamard design at one or more radii.  They
contain only centers and fixed-DBM forward costs; analytic DBM gradients are
never exported.  Antithetic cost differences provide directional slope and
curvature supervision for a later cost-sensitive Actor objective.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from generate_dbm_direct_gt_validation import (
    batched_cost,
    interpolate_knots,
    sha256_file,
)


FORMAT_VERSION = 1
GENERATOR_ID = "anycar-dbm-j16-local-forward-cost-v1"
ACTION_DIMENSION = 16
DIRECTION_COUNT = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radii-sigma", default="0.05,0.15")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_radii(value: str) -> np.ndarray:
    radii = np.asarray([float(item) for item in value.split(",") if item.strip()])
    if not len(radii) or np.any(~np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("--radii-sigma must contain positive finite values")
    if len(np.unique(radii)) != len(radii):
        raise ValueError("--radii-sigma values must be distinct")
    return radii.astype(np.float32)


def hadamard_directions() -> np.ndarray:
    matrix = np.ones((1, 1), dtype=np.float32)
    while len(matrix) < ACTION_DIMENSION:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    if matrix.shape != (ACTION_DIMENSION, ACTION_DIMENSION):
        raise AssertionError("invalid Hadamard construction")
    if not np.array_equal(
        matrix @ matrix.T, ACTION_DIMENSION * np.eye(ACTION_DIMENSION)
    ):
        raise AssertionError("Hadamard design is not orthogonal")
    return matrix.reshape(DIRECTION_COUNT, 8, 2)


def make_centers(
    base: np.ndarray,
    sigma: np.ndarray,
    radii: np.ndarray,
    directions: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw = [base]
    for radius in radii:
        for direction in directions:
            delta = radius * sigma.reshape(1, 2) * direction
            raw.extend((base + delta, base - delta))
    raw_array = np.asarray(raw, np.float32)
    return raw_array, np.clip(raw_array, low, high).astype(np.float32)


def local_statistics(
    base: np.ndarray,
    centers: np.ndarray,
    cost: np.ndarray,
    sigma: np.ndarray,
    radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    slopes, curvatures, symmetry, ranks = [], [], [], []
    standardized = (centers - base[None]) / sigma.reshape(1, 1, 2)
    cursor = 1
    for radius in radii:
        one_slope, one_curvature, one_symmetry = [], [], []
        displacement = []
        for _ in range(DIRECTION_COUNT):
            positive, negative = cursor, cursor + 1
            one_slope.append((cost[positive] - cost[negative]) / (2.0 * radius))
            one_curvature.append(
                (cost[positive] + cost[negative] - 2.0 * cost[0])
                / (radius * radius)
            )
            one_symmetry.append(bool(np.allclose(
                standardized[positive], -standardized[negative], rtol=0, atol=1e-6
            )))
            displacement.extend((standardized[positive], standardized[negative]))
            cursor += 2
        slopes.append(one_slope)
        curvatures.append(one_curvature)
        symmetry.append(one_symmetry)
        ranks.append(np.linalg.matrix_rank(np.asarray(displacement).reshape(-1, 16)))
    return (
        np.asarray(slopes, np.float32),
        np.asarray(curvatures, np.float32),
        np.asarray(symmetry, bool),
        np.asarray(ranks, np.int32),
    )


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    gt_root = args.gt_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    radii = parse_radii(args.radii_sigma)
    directions = hadamard_directions()
    gt_summary = json.loads((gt_root / "summary.json").read_text())
    if gt_summary["split"] != "train":
        raise AssertionError("local labels may only be generated from train GT")
    if "test" not in gt_summary["test_policy"]:
        raise AssertionError("GT summary does not seal test")
    if Path(gt_summary["source"]).resolve() != source_root:
        raise AssertionError("GT/source roots differ")
    rows = list(gt_summary["rows"])
    if args.max_snapshots:
        rows = rows[: args.max_snapshots]
    if not rows:
        raise ValueError("no GT rows selected")

    records: list[dict[str, Any]] = []
    for row in rows:
        source_path = Path(row["source"])
        gt_path = gt_root / row["episode"] / row["snapshot"]
        with np.load(source_path, allow_pickle=False) as source, np.load(
            gt_path, allow_pickle=False
        ) as gt:
            source_hash = sha256_file(source_path)
            if source_hash != row["source_sha256"] or str(gt["source_sha256"]) != source_hash:
                raise AssertionError(f"source hash mismatch: {source_path}")
            best = int(gt["knot_best_index"])
            reference = np.asarray(source["reference"], np.float32)
            params = json.loads(str(source["mppi_params_json"]))
            if len(reference) == int(params["horizon"]) + 1:
                reference = reference[1:]
            records.append({
                "episode": row["episode"],
                "snapshot": row["snapshot"],
                "source_path": source_path,
                "source_hash": source_hash,
                "gt_path": gt_path,
                "gt_hash": sha256_file(gt_path),
                "base": np.asarray(gt["optimized_knots"][best], np.float32),
                "j16_cost": float(row["j16_best_found"]),
                "initial_state_six": np.asarray(source["initial_state_six"], np.float32),
                "current_action": np.asarray(source["current_action"], np.float32),
                "reference": reference,
                "mppi_params_json": str(source["mppi_params_json"]),
                "cost_weights_json": str(source["cost_weights_json"]),
                "dbm_params_json": str(source["dbm_params_json"]),
            })
    for key in ("cost_weights_json", "dbm_params_json"):
        if len({record[key] for record in records}) != 1:
            raise ValueError(f"{key} differs across snapshots")
    frozen_mppi = []
    for record in records:
        value = json.loads(record["mppi_params_json"])
        value.pop("seed", None)
        frozen_mppi.append(value)
    if any(value != frozen_mppi[0] for value in frozen_mppi[1:]):
        raise ValueError("non-seed MPPI parameters differ across snapshots")

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    params = TorchMPPIParams(**json.loads(records[0]["mppi_params_json"]))
    weights = TorchMPPICostWeights(**json.loads(records[0]["cost_weights_json"]))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(records[0]["dbm_params_json"]))
    )
    sigma = np.asarray(params.noise_sigma, np.float32)
    low = np.asarray(params.action_min, np.float32)
    high = np.asarray(params.action_max, np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    started = time.perf_counter()
    base_errors, clips, slopes_all, curvatures_all, symmetry_all, ranks_all = (
        [], [], [], [], [], []
    )
    try:
        for batch_start in range(0, len(records), args.batch_size):
            batch = records[batch_start:batch_start + args.batch_size]
            raw_and_centers = [
                make_centers(record["base"], sigma, radii, directions, low, high)
                for record in batch
            ]
            raw = np.stack([value[0] for value in raw_and_centers])
            centers = np.stack([value[1] for value in raw_and_centers])
            center_tensor = torch.as_tensor(centers, dtype=dtype, device=device)
            actions = interpolate_knots(center_tensor, params.horizon)
            initial = torch.as_tensor(
                np.stack([record["initial_state_six"] for record in batch]),
                dtype=dtype, device=device,
            )
            current = torch.as_tensor(
                np.stack([record["current_action"] for record in batch]),
                dtype=dtype, device=device,
            )
            reference = torch.as_tensor(
                np.stack([record["reference"] for record in batch]),
                dtype=dtype, device=device,
            )
            with torch.no_grad():
                costs = batched_cost(
                    backend, weights, actions, initial, current, reference
                ).cpu().numpy().astype(np.float32)
            for local, record in enumerate(batch):
                slopes, curvatures, symmetry, ranks = local_statistics(
                    record["base"], centers[local], costs[local], sigma, radii
                )
                target_dir = staging / record["episode"]
                target_dir.mkdir(exist_ok=True)
                target = target_dir / record["snapshot"]
                np.savez_compressed(
                    target,
                    format_version=np.asarray(FORMAT_VERSION, np.int32),
                    source_snapshot=np.asarray(str(record["source_path"])),
                    source_snapshot_sha256=np.asarray(record["source_hash"]),
                    gt_result=np.asarray(str(record["gt_path"])),
                    gt_result_sha256=np.asarray(record["gt_hash"]),
                    base_j16_knots=record["base"],
                    base_j16_cost=np.asarray(costs[local, 0]),
                    radii_sigma=radii,
                    normalized_directions=directions,
                    raw_centers=raw[local],
                    centers=centers[local],
                    direct_cost=costs[local],
                    directional_slope=slopes,
                    directional_curvature=curvatures,
                    symmetric_pair_mask=symmetry,
                    local_direction_rank=ranks,
                )
                base_errors.append(abs(float(costs[local, 0]) - record["j16_cost"]))
                clips.append(float(np.mean(raw[local] != centers[local])))
                slopes_all.append(slopes)
                curvatures_all.append(curvatures)
                symmetry_all.append(symmetry)
                ranks_all.append(ranks)
            done = min(batch_start + len(batch), len(records))
            print(
                f"[{done:04d}/{len(records):04d}] local forward-cost labels "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
        split_manifest = Path(gt_summary["split_manifest"])
        shutil.copy2(split_manifest, staging / "splits.json")
        slopes_array = np.asarray(slopes_all, np.float32)
        curvature_array = np.asarray(curvatures_all, np.float32)
        symmetry_array = np.asarray(symmetry_all, bool)
        ranks_array = np.asarray(ranks_all, np.int32)
        summary = {
            "format_version": FORMAT_VERSION,
            "generator_id": GENERATOR_ID,
            "semantics": "train-only deterministic forward DBM costs; no exported DBM gradient",
            "source": str(source_root),
            "gt_root": str(gt_root),
            "gt_summary_sha256": sha256_file(gt_root / "summary.json"),
            "split": "train",
            "snapshot_count": len(records),
            "center_count": int(1 + 2 * DIRECTION_COUNT * len(radii)),
            "radii_sigma": radii.tolist(),
            "direction_design": "16x16 Sylvester Hadamard with antithetic pairs",
            "maximum_base_cost_replay_error": float(np.max(base_errors)),
            "center_clip_fraction_mean": float(np.mean(clips)),
            "symmetric_pair_fraction": float(np.mean(symmetry_array)),
            "full_rank_fraction": float(np.mean(ranks_array == ACTION_DIMENSION)),
            "absolute_directional_slope_mean": float(np.mean(np.abs(slopes_array))),
            "positive_directional_curvature_fraction": float(np.mean(curvature_array > 0)),
            "elapsed_seconds": time.perf_counter() - started,
            "test_policy": "validation and test splits not loaded or evaluated",
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": "ok", **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
