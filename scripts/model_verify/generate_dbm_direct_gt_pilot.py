#!/usr/bin/env python3
"""Generate best-found direct-action DBM oracles on frozen MPPI snapshots.

This is an offline diagnostic.  It deliberately uses analytic DBM gradients to
separate the 8x2-knot parameterization gap from later MPPI-center and policy
gaps.  The gradients are not exposed to a proposal policy or deployment path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import (
    TorchMPPIController,
    TorchMPPICostWeights,
    TorchMPPIParams,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1"
)
DEFAULT_T0 = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_teacher_t0_diverse_20260805_v1"
)
DEFAULT_T1 = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_teacher_t1_diverse_20260805_v1"
)
DEFAULT_OUTPUT = ROOT / "outputs/mppi_proposal/dbm_direct_gt_pilot_20260806_v1"
DEFAULT_EPISODES = (
    "episode_000",  # 1.2 m/s, steady
    "episode_021",  # 1.6 m/s, cold start
    "episode_042",  # 2.0 m/s, lateral recovery
    "episode_063",  # 2.4 m/s, heading recovery
    "episode_084",  # 2.8 m/s, dynamic recovery
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--t0-labels", type=Path, default=DEFAULT_T0)
    parser.add_argument("--t1-labels", type=Path, default=DEFAULT_T1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resume-dir",
        type=Path,
        help="Refine optimized knots/actions from an earlier pilot directory.",
    )
    parser.add_argument("--episodes", nargs="+", default=list(DEFAULT_EPISODES))
    parser.add_argument("--control-step", type=int, default=250)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed", type=int, default=29401)
    parser.add_argument("--random-starts", type=int, default=3)
    parser.add_argument("--knot-steps", type=int, default=500)
    parser.add_argument("--action-steps", type=int, default=700)
    parser.add_argument("--knot-lr", type=float, default=0.03)
    parser.add_argument("--action-lr", type=float, default=0.02)
    parser.add_argument("--trace-stride", type=int, default=10)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_t0(label: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    values = np.asarray(label[key], dtype=np.float64)
    if values.shape == (8, 2):
        return values
    ids = list(np.asarray(label["config_ids"]).astype(str))
    return values[ids.index("collection_default")]


def load_starts(
    snapshot: np.lib.npyio.NpzFile,
    t0_path: Path,
    t1_path: Path,
    random_starts: int,
    rng: np.random.Generator,
) -> tuple[list[str], np.ndarray]:
    warm = np.asarray(snapshot["sampling_mean_knots"], dtype=np.float64)
    names = ["warm", "zero"]
    starts = [warm, np.zeros_like(warm)]
    if t0_path.is_file():
        with np.load(t0_path, allow_pickle=False) as label:
            names.extend(("t0_best", "t0_soft"))
            starts.extend(
                (
                    select_t0(label, "best_candidate_knots"),
                    select_t0(label, "soft_teacher_center_knots"),
                )
            )
    if t1_path.is_file():
        with np.load(t1_path, allow_pickle=False) as label:
            names.append("t1_teacher")
            starts.append(np.asarray(label["teacher_center_knots"], dtype=np.float64))
    sigma = np.asarray(json.loads(str(snapshot["mppi_params_json"]))["noise_sigma"])
    for index in range(random_starts):
        names.append(f"warm_random_{index}")
        starts.append(warm + rng.standard_normal(warm.shape) * sigma)
    return names, np.clip(np.asarray(starts), -0.999, 0.999)


def bounded_raw(initial: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    normalized = ((initial - lower) / (upper - lower) * 2.0 - 1.0).clamp(-0.999, 0.999)
    return torch.atanh(normalized).detach().requires_grad_(True)


def bounded_value(raw: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return lower + 0.5 * (torch.tanh(raw) + 1.0) * (upper - lower)


def direct_cost(
    controller: TorchMPPIController,
    backend: TorchDynamicBicycleRolloutBackend,
    actions: torch.Tensor,
    initial_state: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    full = backend.rollout_full_state_differentiable(initial_state, actions)
    trajectory = full[..., [0, 1, 2, 3, 5]]
    cost = controller.trajectory_cost(trajectory, actions, reference, current_action)
    return cost, trajectory


def optimize_actions(
    controller: TorchMPPIController,
    backend: TorchDynamicBicycleRolloutBackend,
    initial: torch.Tensor,
    initial_state: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
    steps: int,
    lr: float,
    trace_stride: int,
    knots: bool,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    action_min = torch.tensor(controller.params.action_min, dtype=initial.dtype, device=initial.device)
    action_max = torch.tensor(controller.params.action_max, dtype=initial.dtype, device=initial.device)
    raw = bounded_raw(initial, action_min, action_max)
    optimizer = torch.optim.Adam((raw,), lr=lr)
    best_cost = torch.full((len(initial),), torch.inf, dtype=initial.dtype, device=initial.device)
    best_value = initial.detach().clone()
    trace: list[np.ndarray] = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        value = bounded_value(raw, action_min, action_max)
        actions = controller._interpolate_knots(value) if knots else value
        cost, _ = direct_cost(
            controller, backend, actions, initial_state, current_action, reference
        )
        finite = torch.isfinite(cost)
        improved = finite & (cost < best_cost)
        best_cost = torch.where(improved, cost.detach(), best_cost)
        best_value = torch.where(
            improved.reshape(-1, 1, 1), value.detach(), best_value
        )
        if step % trace_stride == 0 or step == steps:
            trace.append(best_cost.detach().cpu().numpy().copy())
        if step == steps:
            break
        finite_cost = cost[finite]
        if len(finite_cost) == 0:
            break
        finite_cost.sum().backward()
        torch.nn.utils.clip_grad_norm_((raw,), max_norm=1000.0)
        optimizer.step()
    return best_value, best_cost, np.asarray(trace)


@torch.no_grad()
def replay_cost(
    controller: TorchMPPIController,
    backend: TorchDynamicBicycleRolloutBackend,
    actions: torch.Tensor,
    initial_state: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    full = backend.rollout_full_state(
        torch.empty(0, device=actions.device), initial_state, current_action, actions
    )
    trajectory = full[..., [0, 1, 2, 3, 5]]
    return controller.trajectory_cost(trajectory, actions, reference, current_action), trajectory


def process_snapshot(args: argparse.Namespace, episode: str, dtype: torch.dtype) -> dict[str, Any]:
    snapshot_path = args.source / episode / "snapshots" / f"step_{args.control_step:06d}.npz"
    t0_path = args.t0_labels / episode / f"step_{args.control_step:06d}.npz"
    t1_path = args.t1_labels / episode / f"step_{args.control_step:06d}.npz"
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed + int(episode.split("_")[-1]))
    with np.load(snapshot_path, allow_pickle=False) as data:
        params = TorchMPPIParams(**json.loads(str(data["mppi_params_json"])))
        weights = TorchMPPICostWeights(**json.loads(str(data["cost_weights_json"])))
        backend = TorchDynamicBicycleRolloutBackend(
            TorchDBMParams(**json.loads(str(data["dbm_params_json"])))
        )
        backend.set_initial_lateral_velocity(float(data["initial_lateral_velocity"]))
        controller = TorchMPPIController(backend, params, weights, device)
        resume_path = None
        resume_actions_np = None
        if args.resume_dir is not None:
            resume_path = args.resume_dir / f"{episode}_step_{args.control_step:06d}.npz"
        if resume_path is not None and resume_path.is_file():
            with np.load(resume_path, allow_pickle=False) as previous:
                if str(previous["source_sha256"]) != sha256_file(snapshot_path):
                    raise AssertionError(f"{resume_path}: source hash mismatch")
                names = list(previous["start_names"].astype(str))
                starts_np = np.asarray(previous["optimized_knots"], dtype=np.float64)
                resume_actions_np = np.asarray(previous["optimized_actions"], dtype=np.float64)
        else:
            names, starts_np = load_starts(data, t0_path, t1_path, args.random_starts, rng)
        starts = torch.as_tensor(starts_np, dtype=dtype, device=device)
        initial_state = torch.as_tensor(data["initial_state"], dtype=dtype, device=device).reshape(1, 5)
        current_action = torch.as_tensor(data["current_action"], dtype=dtype, device=device).reshape(1, 2)
        reference_np = np.asarray(data["reference"])
        if len(reference_np) == params.horizon + 1:
            reference_np = reference_np[1:]
        reference = torch.as_tensor(reference_np, dtype=dtype, device=device)
        knot_value, knot_cost, knot_trace = optimize_actions(
            controller, backend, starts, initial_state, current_action, reference,
            args.knot_steps, args.knot_lr, args.trace_stride, True,
        )
        knot_actions = controller._interpolate_knots(knot_value)
        # Every converged knot solution becomes an independent initialization
        # for the less constrained 50x2 direct-action search.  On refinement,
        # retain the lower-cost choice between the previous direct solution and
        # the newly refined knot interpolation for every start.
        action_initial = knot_actions
        if resume_actions_np is not None:
            previous_actions = torch.as_tensor(
                resume_actions_np, dtype=dtype, device=device
            )
            with torch.no_grad():
                previous_cost, _ = direct_cost(
                    controller, backend, previous_actions, initial_state,
                    current_action, reference,
                )
                refined_knot_cost, _ = direct_cost(
                    controller, backend, knot_actions, initial_state,
                    current_action, reference,
                )
            use_previous = (previous_cost <= refined_knot_cost).reshape(-1, 1, 1)
            action_initial = torch.where(use_previous, previous_actions, knot_actions)
        action_value, action_cost, action_trace = optimize_actions(
            controller, backend, action_initial, initial_state, current_action, reference,
            args.action_steps, args.action_lr, args.trace_stride, False,
        )
        knot_replay, knot_trajectory = replay_cost(
            controller, backend, knot_actions, initial_state, current_action, reference
        )
        action_replay, action_trajectory = replay_cost(
            controller, backend, action_value, initial_state, current_action, reference
        )
        warm_knots = torch.as_tensor(
            np.asarray(data["sampling_mean_knots"])[None], dtype=dtype, device=device
        )
        warm_actions = controller._interpolate_knots(warm_knots)
        warm_cost, _ = replay_cost(
            controller, backend, warm_actions, initial_state, current_action, reference
        )

    knot_best = int(torch.argmin(knot_replay))
    action_best = int(torch.argmin(action_replay))
    top_count = min(3, len(names))
    knot_sorted = np.sort(knot_replay.detach().cpu().numpy())[:top_count]
    action_sorted = np.sort(action_replay.detach().cpu().numpy())[:top_count]
    result_path = args.output_dir / f"{episode}_step_{args.control_step:06d}.npz"
    np.savez_compressed(
        result_path,
        source_snapshot=np.asarray(str(snapshot_path)),
        source_sha256=np.asarray(sha256_file(snapshot_path)),
        resume_parent=np.asarray("" if resume_path is None else str(resume_path)),
        start_names=np.asarray(names),
        initial_knots=starts.detach().cpu().numpy(),
        optimized_knots=knot_value.detach().cpu().numpy(),
        knot_actions=knot_actions.detach().cpu().numpy(),
        knot_cost_optimization=knot_cost.detach().cpu().numpy(),
        knot_cost_replay=knot_replay.detach().cpu().numpy(),
        knot_trajectories=knot_trajectory.detach().cpu().numpy(),
        knot_best_trace=knot_trace,
        optimized_actions=action_value.detach().cpu().numpy(),
        action_cost_optimization=action_cost.detach().cpu().numpy(),
        action_cost_replay=action_replay.detach().cpu().numpy(),
        action_trajectories=action_trajectory.detach().cpu().numpy(),
        action_best_trace=action_trace,
        knot_best_index=np.asarray(knot_best),
        action_best_index=np.asarray(action_best),
    )
    return {
        "episode": episode,
        "control_step": args.control_step,
        "source": str(snapshot_path),
        "source_sha256": sha256_file(snapshot_path),
        "num_starts": len(names),
        "warm_cost": float(warm_cost[0]),
        "j16_best_found": float(knot_replay[knot_best]),
        "j100_best_found": float(action_replay[action_best]),
        "parameterization_gap": float(knot_replay[knot_best] - action_replay[action_best]),
        "recoverable_from_warm_j16": float(warm_cost[0] - knot_replay[knot_best]),
        "recoverable_from_warm_j100": float(warm_cost[0] - action_replay[action_best]),
        "j16_top3_relative_spread": float((knot_sorted[-1] - knot_sorted[0]) / max(abs(knot_sorted[0]), 1e-12)),
        "j100_top3_relative_spread": float((action_sorted[-1] - action_sorted[0]) / max(abs(action_sorted[0]), 1e-12)),
        "j16_replay_abs_error": float(abs(knot_replay[knot_best] - knot_cost[knot_best])),
        "j100_replay_abs_error": float(abs(action_replay[action_best] - action_cost[action_best])),
        "result": str(result_path),
    }


def plot_summary(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [row["episode"].replace("episode_", "e") for row in rows]
    x = np.arange(len(rows))
    warm = np.asarray([row["warm_cost"] for row in rows])
    j16 = np.asarray([row["j16_best_found"] for row in rows])
    j100 = np.asarray([row["j100_best_found"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(x, warm, "o-", label="warm direct")
    axes[0].plot(x, j16, "o-", label="best-found J16")
    axes[0].plot(x, j100, "o-", label="best-found J100")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("same DBM direct cost (lower is better)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].bar(x - 0.18, warm - j16, 0.36, label="warm - J16")
    axes[1].bar(x + 0.18, j16 - j100, 0.36, label="J16 - J100")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("cost gap")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.suptitle("Fixed-DBM direct oracle pilot (diagnostic gradients only)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    rows = [process_snapshot(args, episode, dtype) for episode in args.episodes]
    fieldnames = list(rows[0])
    with (args.output_dir / "per_snapshot.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "format_version": 1,
        "semantics": "best-found numerical oracle; not a proven global optimum",
        "gradient_scope": "offline fixed-DBM diagnosis only; never a policy input",
        "source": str(args.source),
        "episodes": list(args.episodes),
        "control_step": args.control_step,
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "random_starts": args.random_starts,
        "resume_dir": None if args.resume_dir is None else str(args.resume_dir),
        "knot_steps": args.knot_steps,
        "action_steps": args.action_steps,
        "mean": {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "warm_cost", "j16_best_found", "j100_best_found",
                "parameterization_gap", "recoverable_from_warm_j16",
                "recoverable_from_warm_j100", "j16_top3_relative_spread",
                "j100_top3_relative_spread",
            )
        },
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_summary(rows, args.output_dir / "direct_gt_pilot.png")
    print(json.dumps(summary["mean"], indent=2))


if __name__ == "__main__":
    main()
