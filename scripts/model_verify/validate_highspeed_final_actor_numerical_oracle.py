#!/usr/bin/env python3
"""Independently replay the final-Actor numerical DBM reference."""

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
from run_highspeed_final_actor_numerical_oracle import (
    DEFAULT_FOLD0,
    DEFAULT_FOLD1TO4,
    DEFAULT_OLD_ORACLE,
    DEFAULT_REPLAY,
    DEFAULT_TEACHER,
    load_final_actor_centers,
    select_rows,
)


DEFAULT_ARTIFACT = Path(
    "outputs/mppi_proposal/highspeed_final_actor_numerical_oracle_20260831_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--old-oracle-dir", type=Path, default=DEFAULT_OLD_ORACLE)
    parser.add_argument("--actor-fold0-dir", type=Path, default=DEFAULT_FOLD0)
    parser.add_argument("--actor-fold1to4-dir", type=Path, default=DEFAULT_FOLD1TO4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_cost(
    centers: np.ndarray,
    source: dict[str, np.ndarray],
    rows: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    params = TorchMPPIParams(num_samples=64)
    with torch.no_grad():
        knots = torch.as_tensor(centers, dtype=torch.float32, device=device)
        state = torch.as_tensor(source["state_six"][rows], device=device)
        current = torch.as_tensor(source["current_action"][rows], device=device)
        reference = torch.as_tensor(source["reference"][rows, 1:], device=device)
        return batched_cost(
            backend, weights, interpolate_knots(knots, params.horizon),
            state, current, reference,
        ).cpu().numpy()


def main() -> None:
    args = parse_args()
    artifact = args.artifact_dir.resolve()
    solutions_path = artifact / "solutions.npz"
    summary_path = artifact / "summary.json"
    replay_path = (args.replay_dir / "replay.npz").resolve()
    teacher_path = (args.teacher_dir / "labels.npz").resolve()
    old_oracle_path = (args.old_oracle_dir / "oracle.npz").resolve()
    summary = json.loads(summary_path.read_text())
    with np.load(solutions_path, allow_pickle=False) as loaded:
        result = {name: np.asarray(loaded[name]) for name in loaded.files}
    with np.load(replay_path, allow_pickle=False) as loaded:
        source = {name: np.asarray(loaded[name]) for name in loaded.files}
    with np.load(teacher_path, allow_pickle=False) as loaded:
        teacher = {name: np.asarray(loaded[name]) for name in loaded.files}
    with np.load(old_oracle_path, allow_pickle=False) as loaded:
        old_oracle = {name: np.asarray(loaded[name]) for name in loaded.files}

    repeats = int(summary["contract"]["repeats_per_speed_scenario_cell"])
    expected_rows = select_rows(source, repeats)
    if not np.array_equal(result["source_indices"], expected_rows):
        raise AssertionError("source row contract mismatch")
    rows = expected_rows
    count = len(source["state_six"])
    actor = load_final_actor_centers(
        (args.actor_fold0_dir, args.actor_fold1to4_dir), count,
    )
    lookup = {int(row): index for index, row in enumerate(old_oracle["source_indices"])}
    old_index = np.asarray([lookup[int(row)] for row in rows], np.int64)
    expected_first_six = np.stack((
        source["mean_knots_before"][rows], teacher["teacher_knots"][rows],
        old_oracle["oracle_centers"][old_index], actor[0, rows], actor[1, rows], actor[2, rows],
    ), axis=1)
    source_center_error = float(np.max(np.abs(result["starts"][:, :6] - expected_first_six)))

    device = torch.device(args.device)
    fresh_initial = replay_cost(result["starts"], source, rows, device)
    fresh_best = replay_cost(result["best_center"][:, None], source, rows, device)[:, 0]
    initial_cost_error = float(np.max(np.abs(fresh_initial - result["initial_cost"])))
    best_cost_error = float(np.max(np.abs(fresh_best - result["best_cost"])))
    polish_path = artifact / "polish.npz"
    polish_cost_error = None
    final_best_cost = result["best_cost"]
    if polish_path.exists():
        with np.load(polish_path, allow_pickle=False) as loaded:
            polish = {name: np.asarray(loaded[name]) for name in loaded.files}
        if not np.array_equal(polish["source_indices"], rows):
            raise AssertionError("polish source row contract mismatch")
        fresh_polish = replay_cost(polish["polished_center"][:, None], source, rows, device)[:, 0]
        polish_cost_error = float(np.max(np.abs(fresh_polish - polish["polished_cost"])))
        final_best_cost = polish["polished_cost"]
    trace = result["trace_best_cost"]
    trace_nonincreasing = bool(np.all(np.diff(trace, axis=0) <= 1e-4))
    start_floor = bool(np.all(result["best_cost"] <= result["initial_cost"].min(axis=1) + 1e-4))
    bounds_ok = bool(
        np.min(result["best_center"]) >= -1.000001
        and np.max(result["best_center"]) <= 1.000001
    )
    actor_seed = result["initial_cost"][:, 3:6]
    per_seed = {}
    for seed in range(3):
        gap = actor_seed[:, seed] - final_best_cost
        ratio = gap / actor_seed[:, seed]
        per_seed[str(seed)] = {
            "mean_cost": float(actor_seed[:, seed].mean()),
            "aggregate_gap_fraction": float(gap.sum() / actor_seed[:, seed].sum()),
            "paired_gap_fraction_median": float(np.median(ratio)),
            "paired_gap_fraction_p95": float(np.quantile(ratio, 0.95)),
            "paired_gap_fraction_max": float(ratio.max()),
        }
    checks = {
        "source_rows_exact": True,
        "source_center_max_abs_error": source_center_error,
        "fresh_initial_cost_max_abs_error": initial_cost_error,
        "fresh_best_cost_max_abs_error": best_cost_error,
        "fresh_polish_cost_max_abs_error": polish_cost_error,
        "trace_nonincreasing": trace_nonincreasing,
        "best_never_worse_than_initial_bank": start_floor,
        "physical_bounds_ok": bounds_ok,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    passed = (
        source_center_error <= 1e-7 and initial_cost_error <= 1e-4
        and best_cost_error <= 1e-4
        and (polish_cost_error is None or polish_cost_error <= 1e-4)
        and trace_nonincreasing and start_floor and bounds_ok
    )
    report = {
        "format": "highspeed_final_actor_numerical_oracle_validator_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            "HIGHSPEED_FINAL_ACTOR_NUMERICAL_ORACLE_REPLAY_PASS"
            if passed else "HIGHSPEED_FINAL_ACTOR_NUMERICAL_ORACLE_REPLAY_FAIL"
        ),
        "checks": checks,
        "per_fixed_actor_seed": per_seed,
        "artifacts": {
            "solutions": str(solutions_path), "solutions_sha256": sha256(solutions_path),
            "summary": str(summary_path), "summary_sha256": sha256(summary_path),
        },
    }
    (artifact / "validator_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
