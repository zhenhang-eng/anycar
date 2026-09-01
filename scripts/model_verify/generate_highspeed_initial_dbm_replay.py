#!/usr/bin/env python3
"""Replay early high-speed trace states into a compact train-only DBM bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import (
    TorchMPPIController,
    TorchMPPIParams,
    TorchMPPIRunningState,
)
from car_foundation.query_deployment import QueryHistoryBuffer


DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_highspeed_train_20260828_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/highspeed_initial_dbm_replay_20260828_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps-per-episode", type=int, default=5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_ego(reference: np.ndarray, state: np.ndarray) -> np.ndarray:
    result = np.asarray(reference, dtype=np.float32).copy()
    delta = result[:, :2] - state[None, :2]
    cosine = np.cos(float(state[2]))
    sine = np.sin(float(state[2]))
    result[:, 0] = cosine * delta[:, 0] + sine * delta[:, 1]
    result[:, 1] = -sine * delta[:, 0] + cosine * delta[:, 1]
    result[:, 2] = np.arctan2(
        np.sin(result[:, 2] - float(state[2])),
        np.cos(result[:, 2] - float(state[2])),
    )
    return result


def stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    if args.steps_per_episode < 1:
        raise ValueError("--steps-per-episode must be positive")
    source = args.source.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    plan_path = source / "scenario_plan.json"
    plan = json.loads(plan_path.read_text())
    episode_spec = {row["episode_id"]: row for row in plan["episodes"]}

    stored: dict[str, list[np.ndarray | str | int | float]] = {
        name: []
        for name in (
            "episode_id", "scenario_class", "speed_kph", "control_step",
            "state_six", "current_action", "history", "reference",
            "reference_ego", "frenet_pose", "mean_knots_before",
            "sampled_knots", "sampled_action_sequences",
            "predicted_trajectories_full", "cost", "weight",
            "optimized_action_sequence", "trace_best_cost",
            "replay_best_cost",
        )
    }
    maximum_best_cost_error = 0.0
    for episode_dir in sorted(source.glob("episode_*")):
        spec = episode_spec[episode_dir.name]
        with (episode_dir / "closed_loop_trace.jsonl").open() as stream:
            trace = [json.loads(line) for line in stream if line.strip()]
        trace = trace[: args.steps_per_episode]
        if len(trace) != args.steps_per_episode:
            raise AssertionError(f"short trace: {episode_dir}")

        backend = TorchDynamicBicycleRolloutBackend()
        controller = TorchMPPIController(
            backend,
            TorchMPPIParams(
                num_samples=int(plan["collection"]["num_samples"]),
                num_iterations=int(plan["collection"]["num_iterations"]),
                seed=int(spec["mppi_seed"]),
            ),
            device="cpu",
        )
        history = QueryHistoryBuffer(dt=controller.params.dt)
        for row in trace:
            state = np.asarray(row["state"], dtype=np.float32)
            query = state[[0, 1, 2, 3, 5]]
            current_action = np.asarray(row["current_action"], dtype=np.float32)
            if int(row["control_step"]) == 0:
                history.prime_constant_motion(query, current_action)
            history_tensor = history.tensor()
            reference = np.asarray(row["reference"], dtype=np.float32)
            mean = torch.as_tensor(row["mean_knots_before"], dtype=torch.float32)
            backend.set_initial_lateral_velocity(float(state[4]))
            _, _, info = controller(
                query,
                current_action,
                history_tensor,
                reference,
                TorchMPPIRunningState(mean_knots=mean),
            )
            replay_best = float(info["cost"].min())
            trace_best = float(row["cost_summary_at_collection"]["best"])
            maximum_best_cost_error = max(
                maximum_best_cost_error, abs(replay_best - trace_best)
            )
            stored["episode_id"].append(episode_dir.name)
            stored["scenario_class"].append(str(spec["scenario_class"]))
            stored["speed_kph"].append(float(spec["speed_kph"]))
            stored["control_step"].append(int(row["control_step"]))
            stored["state_six"].append(state)
            stored["current_action"].append(current_action)
            stored["history"].append(history_tensor.squeeze(0).numpy())
            stored["reference"].append(reference)
            stored["reference_ego"].append(reference_ego(reference, state))
            stored["frenet_pose"].append(
                np.asarray(row["frenet_pose"], dtype=np.float32)
            )
            stored["mean_knots_before"].append(mean.numpy())
            stored["sampled_knots"].append(info["sampled_knots"].numpy())
            stored["sampled_action_sequences"].append(
                info["sampled_action_sequences"].numpy()
            )
            stored["predicted_trajectories_full"].append(
                info["sampled_trajectories_full"].numpy()
            )
            stored["cost"].append(info["cost"].numpy())
            stored["weight"].append(info["weight"].numpy())
            stored["optimized_action_sequence"].append(
                info["optimized_action_sequence"].numpy()
            )
            stored["trace_best_cost"].append(trace_best)
            stored["replay_best_cost"].append(replay_best)
            history.append(query, np.asarray(row["executed_action"], np.float32))

    arrays = {
        key: np.asarray(value)
        for key, value in stored.items()
    }
    state_vx = arrays["state_six"][:, 3]
    output.mkdir(parents=True)
    archive_path = output / "replay.npz"
    np.savez_compressed(archive_path, **arrays)
    in_target = (state_vx >= 40.0 / 3.6) & (state_vx <= 100.0 / 3.6)
    summary = {
        "qualification": "HIGHSPEED_INITIAL_TRAIN_REPLAY_COMPLETE",
        "role": "train-only DBM candidate replay; not formal validation/test",
        "source_collection": str(source),
        "source_plan_sha256": sha256(plan_path),
        "archive": str(archive_path),
        "archive_sha256": sha256(archive_path),
        "episode_count": len(list(source.glob("episode_*"))),
        "steps_per_episode": args.steps_per_episode,
        "context_count": int(len(state_vx)),
        "candidate_rollout_count": int(
            len(state_vx) * int(plan["collection"]["num_samples"])
        ),
        "actual_vx_mps": stats(state_vx),
        "actual_vx_kph": stats(state_vx * 3.6),
        "target_40_100_kph_fraction": float(in_target.mean()),
        "best_candidate_cost": stats(arrays["replay_best_cost"]),
        "maximum_replay_vs_trace_best_cost_abs_error": maximum_best_cost_error,
        "formal_validation_or_test_created": False,
        "notes": [
            "Contexts are the first five physical states of every v1 episode.",
            "The 250-step history is reconstructed with the runtime QueryHistoryBuffer contract.",
            "Underspeed-recovery contexts below 40 kph are retained and explicitly counted.",
            "The long-horizon v1 snapshots remain a separate high-reference-speed stress set.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
