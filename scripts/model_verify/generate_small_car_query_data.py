#!/usr/bin/env python3
"""Generate deterministic 20 Hz small-car DBM data for Query training.

The generated PKLs use the repository's historical ``CarDataset`` schema, but
the dynamics and timebase exactly match the current Quick Start numeric plant:
``dt=0.05``, wheelbase 0.21 m, mass 4 kg, friction 0.8, and normalized actions.
States and actions are logged before each transition, so Query training must
use ``--steer-shift 0``.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for package in ("car_dataset", "car_dynamics"):
    sys.path.insert(0, str(REPOSITORY_ROOT / package))

from car_dataset import CarDataset
from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/disk/collect_data_from_anycar/generated_small_car_query_dt005"),
    )
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--episodes-per-file", type=int, default=4)
    parser.add_argument("--control-knot-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def interpolate_knots(knots: torch.Tensor, steps: int) -> torch.Tensor:
    return F.interpolate(
        knots[:, None, :], size=steps, mode="linear", align_corners=True
    )[:, 0, :]


def make_profiles(args, device, generator):
    knot_count = math.ceil(args.steps / args.control_knot_steps) + 1
    target_speed_knots = 0.2 + 3.3 * torch.rand(
        args.episodes, knot_count, generator=generator, device=device
    )
    steer_knots = 1.6 * torch.rand(
        args.episodes, knot_count, generator=generator, device=device
    ) - 0.8
    excitation_knots = 0.5 * torch.randn(
        args.episodes, knot_count, generator=generator, device=device
    )

    # Reserve deterministic subsets for zero-speed starts and stronger action
    # excitation, both of which occur in Quick Start closed-loop operation.
    target_speed_knots[::8, :2] = 0.0
    steer_knots[::8] = torch.clamp(steer_knots[::8] * 1.25, -1.0, 1.0)
    return (
        interpolate_knots(target_speed_knots, args.steps),
        interpolate_knots(steer_knots, args.steps),
        interpolate_knots(excitation_knots, args.steps),
    )


@torch.no_grad()
def simulate(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    dynamics = TorchDynamicBicycleRolloutBackend(TorchDBMParams())

    target_speed, steer_profile, excitation = make_profiles(
        args, device, generator
    )
    state = torch.zeros(args.episodes, 6, dtype=torch.float32, device=device)
    state[:, 2] = 2.0 * math.pi * torch.rand(
        args.episodes, generator=generator, device=device
    ) - math.pi
    state[:, 3] = 2.5 * torch.rand(
        args.episodes, generator=generator, device=device
    )
    state[::4, 3] = 0.0

    states = torch.empty(
        args.episodes, args.steps, 6, dtype=torch.float32, device="cpu"
    )
    actions = torch.empty(
        args.episodes, args.steps, 2, dtype=torch.float32, device="cpu"
    )
    previous_error = target_speed[:, 0] - state[:, 3]
    for step in range(args.steps):
        error = target_speed[:, step] - state[:, 3]
        derivative = error - previous_error
        acceleration = torch.clamp(
            0.75 * error + 0.15 * derivative + 0.12 * excitation[:, step],
            -1.0,
            1.0,
        )
        action = torch.stack((acceleration, steer_profile[:, step]), dim=-1)
        states[:, step] = state.cpu()
        actions[:, step] = action.cpu()
        state = dynamics.step_full_state(state, action)
        if not torch.isfinite(state).all():
            raise RuntimeError(f"non-finite DBM state at step {step}")
        previous_error = error
    return states.numpy(), actions.numpy(), target_speed.cpu().numpy()


def save_dataset(args, states, actions, target_speed):
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    if args.episodes % args.episodes_per_file != 0:
        raise ValueError("episodes must be divisible by episodes-per-file")

    for file_index, begin in enumerate(
        range(0, args.episodes, args.episodes_per_file)
    ):
        end = begin + args.episodes_per_file
        state = states[begin:end].reshape(-1, 6)
        action = actions[begin:end].reshape(-1, 2)
        yaw = state[:, 2]
        lap_end = np.zeros(len(state), dtype=np.int8)
        lap_end[args.steps - 1 :: args.steps] = 1
        zero = np.zeros(len(state), dtype=np.float32)

        dataset = CarDataset()
        dataset.car_params.update(
            {
                "wheelbase": 0.21,
                "mass": 4.0,
                "com": 0.48,
                "friction": 0.8,
                "delay": 0,
                "max_throttle": 8.0,
                "max_steer": 0.36,
                "steer_bias": 0.025,
                "sim": "car-numeric-torch-dbm",
                "dt": 0.05,
                "seed_begin": args.seed + begin,
            }
        )
        dataset.data_logs = {
            "steer": action[:, 1],
            "throttle": action[:, 0],
            "xpos_x": state[:, 0],
            "xpos_y": state[:, 1],
            "xpos_z": zero.copy(),
            "xori_w": np.cos(0.5 * yaw).astype(np.float32),
            "xori_x": zero.copy(),
            "xori_y": zero.copy(),
            "xori_z": np.sin(0.5 * yaw).astype(np.float32),
            "xvel_x": state[:, 3],
            "xvel_y": state[:, 4],
            "xvel_z": zero.copy(),
            "xacc_x": zero.copy(),
            "xacc_y": zero.copy(),
            "xacc_z": zero.copy(),
            "avel_x": zero.copy(),
            "avel_y": zero.copy(),
            "avel_z": state[:, 5],
            "traj_x": zero.copy(),
            "traj_y": zero.copy(),
            "lap_end": lap_end,
        }
        path = output_dir / f"small_car_{file_index:04d}.pkl"
        with path.open("wb") as stream:
            pickle.dump(dataset, stream, pickle.HIGHEST_PROTOCOL)

    metadata = {
        "generator": "TorchDynamicBicycleRolloutBackend",
        "state_action_logging": "pre_transition",
        "required_query_steer_shift": 0,
        "args": {
            **vars(args),
            "output_dir": str(args.output_dir),
        },
        "vehicle": {
            "dt": 0.05,
            "wheelbase": 0.21,
            "lf": 0.1008,
            "lr": 0.1092,
            "mass": 4.0,
            "friction": 0.8,
            "max_throttle": 8.0,
            "max_steer": 0.36,
            "steer_bias": 0.025,
        },
        "nominal_query_steering": {
            "ratio": 1.0 / 0.36,
            "offset_rad": -0.025 / 0.36,
            "offset_deg": math.degrees(-0.025 / 0.36),
        },
        "ranges": {
            "state_min": states.min(axis=(0, 1)).tolist(),
            "state_max": states.max(axis=(0, 1)).tolist(),
            "action_min": actions.min(axis=(0, 1)).tolist(),
            "action_max": actions.max(axis=(0, 1)).tolist(),
            "target_speed_min": float(target_speed.min()),
            "target_speed_max": float(target_speed.max()),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"dataset: {output_dir}")
    return output_dir


def main():
    args = parse_args()
    if args.episodes < 3 or args.steps < 302:
        raise ValueError("need at least 3 episodes and 302 steps")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    states, actions, target_speed = simulate(args)
    save_dataset(args, states, actions, target_speed)


if __name__ == "__main__":
    main()
