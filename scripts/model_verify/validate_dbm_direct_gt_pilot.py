#!/usr/bin/env python3
"""Independently replay saved fixed-DBM direct-oracle pilot actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDBMParams, TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPICostWeights, TorchMPPIParams


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checked = 0
    maximum_error = 0.0
    for result_path in sorted(args.result_dir.glob("episode_*_step_*.npz")):
        with np.load(result_path, allow_pickle=False) as result:
            source_path = Path(str(result["source_snapshot"]))
            if sha256_file(source_path) != str(result["source_sha256"]):
                raise AssertionError(f"{result_path}: source hash mismatch")
            with np.load(source_path, allow_pickle=False) as source:
                params = TorchMPPIParams(**json.loads(str(source["mppi_params_json"])))
                weights = TorchMPPICostWeights(**json.loads(str(source["cost_weights_json"])))
                backend = TorchDynamicBicycleRolloutBackend(
                    TorchDBMParams(**json.loads(str(source["dbm_params_json"])))
                )
                backend.set_initial_lateral_velocity(float(source["initial_lateral_velocity"]))
                controller = TorchMPPIController(backend, params, weights, args.device)
                dtype = torch.float64
                initial = torch.as_tensor(source["initial_state"], dtype=dtype, device=args.device).reshape(1, 5)
                current = torch.as_tensor(source["current_action"], dtype=dtype, device=args.device).reshape(1, 2)
                reference_np = np.asarray(source["reference"])[-params.horizon :]
                reference = torch.as_tensor(reference_np, dtype=dtype, device=args.device)
                for prefix, action_key in (
                    ("knot", "knot_actions"), ("action", "optimized_actions")
                ):
                    actions = torch.as_tensor(result[action_key], dtype=dtype, device=args.device)
                    full = backend.rollout_full_state(
                        torch.empty(0, device=args.device), initial, current, actions
                    )
                    trajectory = full[..., [0, 1, 2, 3, 5]]
                    cost = controller.trajectory_cost(trajectory, actions, reference, current).cpu().numpy()
                    saved = np.asarray(result[f"{prefix}_cost_replay"])
                    error = float(np.max(np.abs(cost - saved)))
                    maximum_error = max(maximum_error, error)
                    if not np.allclose(cost, saved, atol=args.atol, rtol=args.rtol):
                        raise AssertionError(f"{result_path}: {prefix} replay error {error}")
            checked += 1
    if checked == 0:
        raise FileNotFoundError(f"no result NPZ files in {args.result_dir}")
    print(json.dumps({"checked": checked, "maximum_cost_error": maximum_error, "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
