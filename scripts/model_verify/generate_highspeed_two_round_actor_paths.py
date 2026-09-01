#!/usr/bin/env python3
"""Generate two-round deterministic search paths for all high-speed OOF Actors."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from generate_dbm_j16_local_curvature_labels import hadamard_directions
from run_highspeed_strong_search_oracle import load_actor_oof_centers
from run_mppi_proximal_search_phase1a import ring_candidates, seed_bank_directions


DEFAULT_REPLAY = Path(
    "outputs/mppi_proposal/highspeed_initial_dbm_replay_expansion_20260830_v1"
)
DEFAULT_TEACHER = Path(
    "outputs/mppi_proposal/highspeed_proximal_teacher_expansion_20260830_v1"
)
DEFAULT_PRETRAIN = Path(
    "outputs/mppi_proposal/highspeed_actor_twin_critic_pretrain_expansion_e4_20260830_v2"
)
DEFAULT_OAC = Path(
    "outputs/mppi_proposal/highspeed_actor_visited_oac_expansion_e4_20260830_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/highspeed_two_round_actor_paths_20260830_v1"
)
SIGMA = np.asarray((0.25, 0.35), np.float32)
LOW = np.asarray((-1.0, -1.0), np.float32)
HIGH = np.asarray((1.0, 1.0), np.float32)
RADII = (1.0, 0.70)
SOURCE_NAMES = ("oac_seed0", "oac_seed1", "oac_seed2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--pretrain-dir", type=Path, default=DEFAULT_PRETRAIN)
    parser.add_argument("--oac-dir", type=Path, default=DEFAULT_OAC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-chunk", type=int, default=256)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64)
    return {
        "count": int(values.size), "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)), "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)), "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    replay_path = (args.replay_dir / "replay.npz").resolve()
    replay_summary_path = (args.replay_dir / "summary.json").resolve()
    teacher_path = (args.teacher_dir / "labels.npz").resolve()
    teacher_summary_path = (args.teacher_dir / "summary.json").resolve()
    pretrain_summary_path = (args.pretrain_dir / "summary.json").resolve()
    oac_summary_path = (args.oac_dir / "summary.json").resolve()
    oac_validator_path = (args.oac_dir / "validator_report.json").resolve()
    replay_summary = json.loads(replay_summary_path.read_text())
    teacher_summary = json.loads(teacher_summary_path.read_text())
    pretrain_summary = json.loads(pretrain_summary_path.read_text())
    oac_summary = json.loads(oac_summary_path.read_text())
    oac_validator = json.loads(oac_validator_path.read_text())
    if replay_summary["formal_validation_or_test_created"]:
        raise AssertionError("replay is not train-only")
    if teacher_summary["protocol"]["formal_validation_or_test_created"]:
        raise AssertionError("teacher is not train-only")
    if oac_validator["qualification"] != "HIGHSPEED_ACTOR_VISITED_OAC_INDEPENDENT_REPLAY_PASS":
        raise AssertionError("OAC source did not pass independent replay")
    if sha256(replay_path) != replay_summary["archive_sha256"]:
        raise AssertionError("replay hash mismatch")
    if sha256(teacher_path) != teacher_summary["labels_sha256"]:
        raise AssertionError("teacher hash mismatch")
    with np.load(replay_path, allow_pickle=False) as loaded:
        source = {name: np.asarray(loaded[name]) for name in loaded.files}
    count = len(source["state_six"])
    if count != 600:
        raise AssertionError("expected complete 600-state expansion replay")
    starts = load_actor_oof_centers(oac_summary, pretrain_summary, count).transpose(1, 0, 2, 3)
    directions = (hadamard_directions().astype(np.float32), seed_bank_directions(2))
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    params = TorchMPPIParams(num_samples=64)
    device = torch.device(args.device)
    path_centers_all, path_costs_all = [], []
    evaluated_centers_all, evaluated_costs_all, evaluation_counts = [], [], []
    for row in range(count):
        state = torch.as_tensor(source["state_six"][row : row + 1], device=device)
        current = torch.as_tensor(source["current_action"][row : row + 1], device=device)
        reference = torch.as_tensor(source["reference"][row : row + 1, 1:], device=device)
        cache: dict[bytes, float] = {}
        cache_centers: dict[bytes, np.ndarray] = {}

        def evaluate(candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            clipped = np.clip(candidates, LOW, HIGH).astype(np.float32)
            keys = [np.round(value, 7).tobytes() for value in clipped]
            pending_keys, pending = [], []
            for key, value in zip(keys, clipped):
                if key not in cache and key not in pending_keys:
                    pending_keys.append(key); pending.append(value)
            if pending:
                array = np.asarray(pending, np.float32)
                values = []
                with torch.no_grad():
                    for start in range(0, len(array), args.eval_chunk):
                        knots = torch.as_tensor(array[start : start + args.eval_chunk][None], device=device)
                        actions = interpolate_knots(knots, params.horizon)
                        values.append(batched_cost(
                            backend, weights, actions, state, current, reference
                        )[0].cpu().numpy())
                for key, center, cost in zip(pending_keys, pending, np.concatenate(values)):
                    cache[key] = float(cost); cache_centers[key] = center.copy()
            return clipped, np.asarray([cache[key] for key in keys], np.float32)

        row_starts, start_costs = evaluate(starts[row])
        row_paths, row_path_costs = [], []
        for initial_center, initial_cost in zip(row_starts, start_costs):
            incumbent = initial_center.copy(); incumbent_cost = float(initial_cost)
            centers_path, costs_path = [incumbent.copy()], [incumbent_cost]
            for radius, basis in zip(RADII, directions):
                candidates = ring_candidates(incumbent, SIGMA, [radius], basis)
                candidates, costs = evaluate(candidates)
                best = int(np.argmin(costs))
                if float(costs[best]) < incumbent_cost:
                    incumbent = candidates[best].copy(); incumbent_cost = float(costs[best])
                centers_path.append(incumbent.copy()); costs_path.append(incumbent_cost)
            row_paths.append(centers_path); row_path_costs.append(costs_path)
        keys = list(cache.keys())
        evaluated_centers_all.append(np.stack([cache_centers[key] for key in keys]))
        evaluated_costs_all.append(np.asarray([cache[key] for key in keys], np.float32))
        evaluation_counts.append(len(keys))
        path_centers_all.append(row_paths); path_costs_all.append(row_path_costs)
        if (row + 1) % 25 == 0 or row == 0:
            before = float(np.mean(start_costs)); after = float(np.mean(np.asarray(row_path_costs)[:, -1]))
            print(f"[{row + 1:03d}/600] {source['episode_id'][row]} J={before:.1f}->{after:.1f} budget={len(keys)}", flush=True)

    maximum = max(evaluation_counts)
    evaluated_centers = np.full((count, maximum, 8, 2), np.nan, np.float32)
    evaluated_costs = np.full((count, maximum), np.nan, np.float32)
    for row, (centers, costs) in enumerate(zip(evaluated_centers_all, evaluated_costs_all)):
        evaluated_centers[row, : len(costs)] = centers
        evaluated_costs[row, : len(costs)] = costs
    path_centers = np.asarray(path_centers_all, np.float32)
    path_costs = np.asarray(path_costs_all, np.float32)
    start_cost = path_costs[..., 0]
    path1_gain = start_cost - path_costs[..., 1]
    path2_gain = start_cost - path_costs[..., 2]
    output.mkdir(parents=True)
    artifact_path = output / "paths.npz"
    np.savez_compressed(
        artifact_path,
        source_indices=np.arange(count, dtype=np.int64),
        episode_id=source["episode_id"], scenario_class=source["scenario_class"],
        nominal_speed_kph=source["speed_kph"], control_step=source["control_step"],
        actual_vx_mps=source["state_six"][:, 3], source_actor_names=np.asarray(SOURCE_NAMES),
        path_centers=path_centers, path_costs=path_costs,
        evaluation_counts=np.asarray(evaluation_counts, np.int32),
        evaluated_centers=evaluated_centers, evaluated_costs=evaluated_costs,
    )
    summary = {
        "format": "highspeed_two_round_actor_paths_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_TWO_ROUND_ACTOR_PATHS_TRAIN_ONLY",
        "contract": {
            "contexts": count, "source_actor_paths_per_context": 3,
            "radii_sigma_rms": list(RADII),
            "directions": "base Hadamard then deterministic Givens bank2",
            "update": "strict incumbent improvement; no elite averaging",
            "objective": "deterministic fixed-DBM J50",
            "dbm_gradient_used": False, "formal_validation_or_test_created": False,
        },
        "sources": {
            "replay": str(replay_path), "replay_sha256": sha256(replay_path),
            "teacher": str(teacher_path), "teacher_sha256": sha256(teacher_path),
            "pretrain_summary": str(pretrain_summary_path),
            "pretrain_summary_sha256": sha256(pretrain_summary_path),
            "oac_summary": str(oac_summary_path), "oac_summary_sha256": sha256(oac_summary_path),
            "oac_validator": str(oac_validator_path),
            "oac_validator_sha256": sha256(oac_validator_path),
        },
        "artifact": str(artifact_path.resolve()), "artifact_sha256": sha256(artifact_path),
        "results": {
            "evaluation_count": distribution(np.asarray(evaluation_counts)),
            "start_cost": distribution(start_cost),
            "path1_cost": distribution(path_costs[..., 1]),
            "path2_cost": distribution(path_costs[..., 2]),
            "path1_gain": distribution(path1_gain), "path2_gain": distribution(path2_gain),
            "path1_strict_improvement_fraction": float(np.mean(path1_gain > 1e-5)),
            "path2_strict_improvement_fraction": float(np.mean(path2_gain > 1e-5)),
            "path_baseline_violations": int(np.sum(path2_gain < -1e-5)),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    main()
