#!/usr/bin/env python3
"""Evaluate arbitrary candidate action sequences on a fixed MPPI snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.mppi import (
    TorchMPPIController,
    TorchMPPICostWeights,
    TorchMPPIParams,
)
from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_foundation.query_deployment import (
    QueryDeploymentModel,
    TorchQueryRolloutBackend,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = (
    REPOSITORY_ROOT
    / "outputs/mppi_sampling_snapshot/live_dbm_clean_step0340/snapshot.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Optional .npy/.npz [N, 50, 2]; defaults to saved baseline samples.",
    )
    parser.add_argument("--candidate-key", default="candidate_actions")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_candidates(path: Path, key: str) -> np.ndarray:
    loaded = np.load(path)
    if isinstance(loaded, np.ndarray):
        return loaded
    if key in loaded:
        return loaded[key]
    if "sampled_action_sequences" in loaded:
        return loaded["sampled_action_sequences"]
    raise KeyError(f"{path} contains neither {key!r} nor sampled_action_sequences")


def main() -> None:
    args = parse_args()
    snapshot_dir = args.snapshot.resolve().parent
    metadata_path = snapshot_dir / "summary.json"
    metadata = json.loads(metadata_path.read_text())
    output_dir = args.output_dir or snapshot_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = np.load(args.snapshot)
    if args.candidates is None:
        candidates = snapshot["sampled_action_sequences"]
        source = "snapshot baseline samples"
    else:
        candidates = load_candidates(args.candidates, args.candidate_key)
        source = str(args.candidates.resolve())
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[1:] != (50, 2):
        raise ValueError(f"candidate actions must have shape [N, 50, 2], got {candidates.shape}")
    if not np.isfinite(candidates).all():
        raise ValueError("candidate actions contain non-finite values")
    if np.max(np.abs(candidates)) > 1.0 + 1e-6:
        raise ValueError("candidate actions exceed normalized bounds [-1, 1]")

    device = torch.device(args.device)
    params = TorchMPPIParams(**metadata["mppi_params"])
    cost_weights = TorchMPPICostWeights(**metadata["cost_weights"])
    backend_name = metadata["model"]["backend"]
    if backend_name == "dbm":
        backend = TorchDynamicBicycleRolloutBackend(
            TorchDBMParams(**metadata["model"]["dbm_params"])
        )
        if "initial_lateral_velocity" not in snapshot:
            raise KeyError(
                "DBM snapshot must contain initial_lateral_velocity for exact rollout"
            )
        backend.set_initial_lateral_velocity(
            float(snapshot["initial_lateral_velocity"])
        )
    elif backend_name == "pytorch":
        checkpoint_value = args.checkpoint or metadata["model"].get("checkpoint")
        if not checkpoint_value:
            raise ValueError("PyTorch Query snapshot requires a checkpoint")
        model = QueryDeploymentModel.from_checkpoint(
            Path(checkpoint_value), device
        )
        backend = TorchQueryRolloutBackend(model)
    else:
        raise ValueError(
            f"snapshot backend must be 'dbm' or 'pytorch', got {backend_name!r}"
        )
    controller = TorchMPPIController(
        backend, params=params, cost_weights=cost_weights, device=device
    )

    history = torch.from_numpy(snapshot["history"]).to(device)
    initial_state = torch.from_numpy(snapshot["initial_state"]).to(device).reshape(1, 5)
    current_action = torch.from_numpy(snapshot["current_action"]).to(device).reshape(1, 2)
    reference = controller._prepare_reference(snapshot["reference"])
    action_tensor = torch.from_numpy(candidates).to(device)
    trajectory = backend(history, initial_state, current_action, action_tensor).to(device)
    components = controller.trajectory_cost_components(
        trajectory, action_tensor, reference, current_action
    )
    cost = sum(components.values())
    weight = torch.softmax(-(cost - cost.min()) / params.temperature, dim=0)

    cost_np = cost.cpu().numpy()
    weight_np = weight.cpu().numpy()
    component_np = {name: value.cpu().numpy() for name, value in components.items()}
    best_index = int(np.argmin(cost_np))
    baseline_max_abs_difference = None
    if args.candidates is None:
        baseline_max_abs_difference = float(
            np.max(np.abs(cost_np - snapshot["cost"]))
        )

    evaluation_path = output_dir / "evaluation.npz"
    np.savez_compressed(
        evaluation_path,
        candidate_actions=candidates,
        predicted_trajectories=trajectory.cpu().numpy(),
        cost=cost_np,
        weight=weight_np,
        **{f"cost_{name}": value for name, value in component_np.items()},
    )
    summary = {
        "snapshot": str(args.snapshot.resolve()),
        "backend": backend_name,
        "candidate_source": source,
        "candidate_count": len(candidates),
        "best_candidate_index": best_index,
        "best_cost": float(cost_np[best_index]),
        "mean_cost": float(cost_np.mean()),
        "median_cost": float(np.median(cost_np)),
        "p95_cost": float(np.percentile(cost_np, 95)),
        "max_cost": float(cost_np.max()),
        "effective_sample_size": float(1.0 / np.square(weight_np).sum()),
        "best_cost_components": {
            name: float(values[best_index])
            for name, values in component_np.items()
        },
        "baseline_cost_max_abs_difference": baseline_max_abs_difference,
        "evaluation": str(evaluation_path.resolve()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
