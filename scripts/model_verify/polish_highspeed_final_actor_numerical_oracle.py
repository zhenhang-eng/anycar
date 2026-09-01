#!/usr/bin/env python3
"""Gradient-free fine-radius polish of the high-speed numerical reference."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from generate_dbm_j16_local_curvature_labels import hadamard_directions
from run_highspeed_final_actor_numerical_oracle import DEFAULT_REPLAY, distribution
from run_mppi_proximal_search_phase1a import ring_candidates, seed_bank_directions


DEFAULT_ARTIFACT = Path(
    "outputs/mppi_proposal/highspeed_final_actor_numerical_oracle_20260831_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--radii-sigma", type=float, nargs="+", default=(0.10, 0.05, 0.02, 0.01))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = args.artifact_dir.resolve()
    output_path = artifact / "polish.npz"
    summary_path = artifact / "polish_summary.json"
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to replace existing polish artifact")
    with np.load(artifact / "solutions.npz", allow_pickle=False) as loaded:
        result = {name: np.asarray(loaded[name]) for name in loaded.files}
    with np.load(args.replay_dir / "replay.npz", allow_pickle=False) as loaded:
        source = {name: np.asarray(loaded[name]) for name in loaded.files}
    rows = result["source_indices"].astype(np.int64)
    center = result["best_center"].astype(np.float32).copy()
    cost = result["best_cost"].astype(np.float32).copy()
    initial = cost.copy()
    sigma = np.asarray((0.25, 0.35), np.float32)
    bases = (
        hadamard_directions().astype(np.float32),
        seed_bank_directions(2), seed_bank_directions(3),
    )
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    params = TorchMPPIParams(num_samples=64)
    device = torch.device(args.device)
    state = torch.as_tensor(source["state_six"][rows], device=device)
    current = torch.as_tensor(source["current_action"][rows], device=device)
    reference = torch.as_tensor(source["reference"][rows, 1:], device=device)
    trace = [cost.copy()]
    stages = ["initial"]
    evaluations = 0
    moves = 0
    for radius in args.radii_sigma:
        for bank_id, basis in enumerate(bases, start=1):
            candidates = np.stack([
                ring_candidates(value, sigma, [radius], basis) for value in center
            ])
            candidates = np.clip(candidates, -1.0, 1.0).astype(np.float32)
            with torch.no_grad():
                candidate_cost = batched_cost(
                    backend, weights,
                    interpolate_knots(torch.as_tensor(candidates, device=device), params.horizon),
                    state, current, reference,
                ).cpu().numpy()
            best = np.argmin(candidate_cost, axis=1)
            row = np.arange(len(rows))
            proposed = candidate_cost[row, best]
            improved = proposed < cost
            center[improved] = candidates[row[improved], best[improved]]
            cost[improved] = proposed[improved]
            moves += int(improved.sum())
            evaluations += int(np.prod(candidate_cost.shape))
            trace.append(cost.copy())
            stages.append(f"radius{radius:g}_bank{bank_id}")
            print(
                f"{stages[-1]} mean={cost.mean():.6f} moved={int(improved.sum())}",
                flush=True,
            )
    improvement = initial - cost
    np.savez_compressed(
        output_path, source_indices=rows, initial_center=result["best_center"],
        initial_cost=initial, polished_center=center, polished_cost=cost,
        stage=np.asarray(stages), trace_cost=np.asarray(trace),
    )
    summary = {
        "format": "highspeed_final_actor_numerical_oracle_polish_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "GRADIENT_FREE_FINE_POLISH_COMPLETE",
        "contract": {
            "radii_sigma": args.radii_sigma, "orthogonal_banks_per_radius": 3,
            "candidates_per_bank": 32, "total_dbm_evaluations": evaluations,
            "offline_deterministic_dbm_only": True,
        },
        "results": {
            "pre_polish_cost": distribution(initial),
            "post_polish_cost": distribution(cost),
            "improvement": distribution(improvement),
            "aggregate_improvement_fraction": float(improvement.sum() / initial.sum()),
            "states_improved": int(np.sum(improvement > 1e-4)),
            "accepted_moves": moves,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
