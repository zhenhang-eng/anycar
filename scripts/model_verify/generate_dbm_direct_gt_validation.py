#!/usr/bin/env python3
"""Generate batched best-found J16/J100 DBM oracles for an episode split.

This is an offline diagnostic.  It uses analytic gradients only inside the
fixed DBM optimizer; no gradient is stored as a policy input or training label.
Multiple independent snapshots are batched without sharing state, reference,
current action, or optimization variables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from generate_dbm_direct_gt_pilot import bounded_raw, bounded_value, select_t0


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
DEFAULT_SPLITS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_direct_center_replay_diverse_20260806_v2/splits.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/dbm_direct_gt_validation_20260806_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--t0-labels", type=Path, default=DEFAULT_T0)
    parser.add_argument("--t1-labels", type=Path, default=DEFAULT_T1)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resume-dir", type=Path,
        help="Continue every per-start J16/J100 solution from a prior full run.",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument("--seed", type=int, default=29601)
    parser.add_argument("--random-starts", type=int, default=3)
    parser.add_argument("--knot-steps", type=int, default=150)
    parser.add_argument("--action-steps", type=int, default=200)
    parser.add_argument("--knot-lr", type=float, default=0.03)
    parser.add_argument("--action-lr", type=float, default=0.02)
    parser.add_argument("--trace-stride", type=int, default=10)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interpolate_knots(knots: torch.Tensor, horizon: int) -> torch.Tensor:
    shape = knots.shape[:-2]
    flat = knots.reshape(-1, knots.shape[-2], 2)
    actions = F.interpolate(
        flat.transpose(1, 2), size=horizon, mode="linear", align_corners=True
    ).transpose(1, 2)
    return actions.reshape(*shape, horizon, 2)


def batched_cost(
    backend: TorchDynamicBicycleRolloutBackend,
    weights: TorchMPPICostWeights,
    actions: torch.Tensor,
    initial_state_six: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Evaluate independent ``[B,S,H,2]`` actions and return ``[B,S]`` cost."""
    batch, starts, horizon, action_dim = actions.shape
    if action_dim != 2 or horizon != backend.horizon:
        raise ValueError("invalid batched action shape")
    flat_actions = actions.reshape(batch * starts, horizon, action_dim)
    initial = initial_state_six[:, None].expand(-1, starts, -1).reshape(
        batch * starts, 6
    )
    full = backend.rollout_full_state_differentiable(initial, flat_actions)
    trajectory = full[..., [0, 1, 2, 3, 5]].reshape(batch, starts, horizon, 5)
    position = (
        trajectory[..., :2] - reference[:, None, :, :2]
    ).square().sum(-1).sum(-1)
    yaw_delta = trajectory[..., 2] - reference[:, None, :, 2]
    yaw = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)).square().sum(-1)
    vx = (trajectory[..., 3] - reference[:, None, :, 3]).square().sum(-1)
    previous = torch.cat((
        current_action[:, None, None, :].expand(-1, starts, 1, -1),
        actions[:, :, :-1],
    ), dim=2)
    rate = actions - previous
    cost = weights.position * position + weights.yaw * yaw + weights.vx * vx
    if reference.shape[-1] == 5 and weights.yawrate != 0:
        cost = cost + weights.yawrate * (
            trajectory[..., 4] - reference[:, None, :, 4]
        ).square().sum(-1)
    cost = cost + weights.acceleration_rate * rate[..., 0].square().sum(-1)
    cost = cost + weights.steering_rate * rate[..., 1].square().sum(-1)
    return cost


def optimize(
    backend: TorchDynamicBicycleRolloutBackend,
    weights: TorchMPPICostWeights,
    initial: torch.Tensor,
    initial_state_six: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
    steps: int,
    learning_rate: float,
    trace_stride: int,
    knots: bool,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    raw = bounded_raw(initial, action_min, action_max)
    optimizer = torch.optim.Adam((raw,), lr=learning_rate)
    best_cost = torch.full(
        initial.shape[:2], torch.inf, dtype=initial.dtype, device=initial.device
    )
    best_value = initial.detach().clone()
    trace = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        value = bounded_value(raw, action_min, action_max)
        actions = interpolate_knots(value, backend.horizon) if knots else value
        cost = batched_cost(
            backend, weights, actions, initial_state_six, current_action, reference
        )
        finite = torch.isfinite(cost)
        improved = finite & (cost < best_cost)
        best_cost = torch.where(improved, cost.detach(), best_cost)
        best_value = torch.where(
            improved[..., None, None], value.detach(), best_value
        )
        if step % trace_stride == 0 or step == steps:
            trace.append(best_cost.min(1).values.detach().cpu().numpy())
        if step == steps:
            break
        if not torch.any(finite):
            break
        cost[finite].sum().backward()
        torch.nn.utils.clip_grad_norm_((raw,), 1000.0)
        optimizer.step()
    return best_value, best_cost, np.stack(trace, axis=1)


def load_record(
    source_path: Path,
    t0_root: Path,
    t1_root: Path,
    random_starts: int,
    seed: int,
) -> dict[str, Any]:
    episode = source_path.parents[1].name
    t0_path = t0_root / episode / source_path.name
    t1_path = t1_root / episode / source_path.name
    with np.load(source_path, allow_pickle=False) as source:
        warm = np.asarray(source["sampling_mean_knots"], np.float32)
        names = ["warm", "zero"]
        starts = [warm, np.zeros_like(warm)]
        if t0_path.is_file():
            with np.load(t0_path, allow_pickle=False) as t0:
                names.extend(("t0_best", "t0_soft"))
                starts.extend((
                    select_t0(t0, "best_candidate_knots"),
                    select_t0(t0, "soft_teacher_center_knots"),
                ))
        if t1_path.is_file():
            with np.load(t1_path, allow_pickle=False) as t1:
                names.append("t1_teacher")
                starts.append(np.asarray(t1["teacher_center_knots"], np.float32))
        params = json.loads(str(source["mppi_params_json"]))
        sigma = np.asarray(params["noise_sigma"], np.float32)
        rng = np.random.default_rng(seed)
        for index in range(random_starts):
            names.append(f"warm_random_{index}")
            starts.append(warm + rng.standard_normal(warm.shape) * sigma)
        reference = np.asarray(source["reference"], np.float32)
        if len(reference) == int(params["horizon"]) + 1:
            reference = reference[1:]
        return {
            "episode": episode,
            "snapshot": source_path.name,
            "source_path": source_path,
            "source_sha256": sha256_file(source_path),
            "start_names": names,
            "starts": np.clip(np.asarray(starts), -0.999, 0.999).astype(np.float32),
            "initial_state_six": np.asarray(source["initial_state_six"], np.float32),
            "current_action": np.asarray(source["current_action"], np.float32),
            "reference": reference,
            "mppi_params_json": str(source["mppi_params_json"]),
            "cost_weights_json": str(source["cost_weights_json"]),
            "dbm_params_json": str(source["dbm_params_json"]),
        }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    split_data = json.loads(args.splits.read_text())
    episodes = list(split_data[args.split])
    paths = [
        path
        for episode in episodes
        for path in sorted((args.source / episode / "snapshots").glob("*.npz"))
    ]
    if args.max_snapshots:
        paths = paths[:args.max_snapshots]
    if not paths:
        raise ValueError("no snapshots selected")
    random.seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    records = [
        load_record(
            path, args.t0_labels, args.t1_labels, args.random_starts,
            args.seed + index,
        )
        for index, path in enumerate(paths)
    ]
    resume_rows: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume_dir is not None:
        resume_summary = json.loads((args.resume_dir / "summary.json").read_text())
        resume_rows = {
            (row["episode"], row["snapshot"]): row
            for row in resume_summary["rows"]
        }
        for record in records:
            key = (record["episode"], record["snapshot"])
            if key not in resume_rows:
                raise ValueError(f"resume result missing {key}")
            resume_path = args.resume_dir / record["episode"] / record["snapshot"]
            with np.load(resume_path, allow_pickle=False) as previous:
                if str(previous["source_sha256"]) != record["source_sha256"]:
                    raise AssertionError(f"resume source hash mismatch: {resume_path}")
                if tuple(previous["start_names"].astype(str)) != tuple(record["start_names"]):
                    raise AssertionError(f"resume start names differ: {resume_path}")
                record["starts"] = np.asarray(previous["optimized_knots"], np.float32)
                record["previous_actions"] = np.asarray(
                    previous["optimized_actions"], np.float32
                )
    for key in ("cost_weights_json", "dbm_params_json"):
        if len({record[key] for record in records}) != 1:
            raise ValueError(f"{key} differs across selected snapshots")
    frozen_mppi = []
    for record in records:
        value = json.loads(record["mppi_params_json"])
        value.pop("seed", None)  # Direct optimization performs no MPPI sampling.
        frozen_mppi.append(value)
    if any(value != frozen_mppi[0] for value in frozen_mppi[1:]):
        raise ValueError("non-seed MPPI parameters differ across selected snapshots")
    params = TorchMPPIParams(**json.loads(records[0]["mppi_params_json"]))
    weights = TorchMPPICostWeights(**json.loads(records[0]["cost_weights_json"]))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(records[0]["dbm_params_json"]))
    )
    action_min = torch.tensor(params.action_min, dtype=dtype, device=device)
    action_max = torch.tensor(params.action_max, dtype=dtype, device=device)
    rows = []
    started = time.time()
    for batch_start in range(0, len(records), args.batch_size):
        batch_records = records[batch_start:batch_start + args.batch_size]
        starts = torch.as_tensor(
            np.stack([record["starts"] for record in batch_records]),
            dtype=dtype, device=device,
        )
        initial = torch.as_tensor(
            np.stack([record["initial_state_six"] for record in batch_records]),
            dtype=dtype, device=device,
        )
        current = torch.as_tensor(
            np.stack([record["current_action"] for record in batch_records]),
            dtype=dtype, device=device,
        )
        reference = torch.as_tensor(
            np.stack([record["reference"] for record in batch_records]),
            dtype=dtype, device=device,
        )
        with torch.no_grad():
            initial_actions = interpolate_knots(starts, params.horizon)
            initial_cost = batched_cost(
                backend, weights, initial_actions, initial, current, reference
            )
        knots, knot_cost, knot_trace = optimize(
            backend, weights, starts, initial, current, reference,
            action_min, action_max, args.knot_steps, args.knot_lr,
            args.trace_stride, True,
        )
        knot_actions = interpolate_knots(knots, params.horizon)
        action_initial = knot_actions
        if args.resume_dir is not None:
            previous_actions = torch.as_tensor(
                np.stack([record["previous_actions"] for record in batch_records]),
                dtype=dtype, device=device,
            )
            with torch.no_grad():
                previous_action_cost = batched_cost(
                    backend, weights, previous_actions, initial, current, reference
                )
                refined_knot_cost = batched_cost(
                    backend, weights, knot_actions, initial, current, reference
                )
            use_previous = previous_action_cost <= refined_knot_cost
            action_initial = torch.where(
                use_previous[..., None, None], previous_actions, knot_actions
            )
        if args.action_steps == 0:
            # J16-only mode: preserve the interpolated knot action exactly.
            # Passing it through bounded_raw/bounded_value would add a small,
            # irrelevant round-trip error and make the J100 placeholder differ.
            actions = knot_actions.detach().clone()
            with torch.no_grad():
                action_cost = batched_cost(
                    backend, weights, actions, initial, current, reference
                )
            action_trace = action_cost.min(1).values.detach().cpu().numpy()[:, None]
        else:
            actions, action_cost, action_trace = optimize(
                backend, weights, action_initial, initial, current, reference,
                action_min, action_max, args.action_steps, args.action_lr,
                args.trace_stride, False,
            )
        with torch.no_grad():
            knot_replay = batched_cost(
                backend, weights, knot_actions, initial, current, reference
            )
            action_replay = batched_cost(
                backend, weights, actions, initial, current, reference
            )
        for local, record in enumerate(batch_records):
            knot_best = int(torch.argmin(knot_replay[local]))
            action_best = int(torch.argmin(action_replay[local]))
            knot_sorted = torch.sort(knot_replay[local]).values[:3]
            action_sorted = torch.sort(action_replay[local]).values[:3]
            resume_row = resume_rows.get((record["episode"], record["snapshot"]))
            warm_cost = (
                float(resume_row["warm_cost"])
                if resume_row is not None else float(initial_cost[local, 0])
            )
            teacher_index = (
                record["start_names"].index("t1_teacher")
                if "t1_teacher" in record["start_names"] else None
            )
            if (
                resume_row is not None
                and resume_row.get("teacher_cost") is not None
            ):
                teacher_cost = float(resume_row["teacher_cost"])
            elif teacher_index is not None:
                teacher_cost = float(initial_cost[local, teacher_index])
            else:
                teacher_cost = None
            j16 = float(knot_replay[local, knot_best])
            j100 = float(action_replay[local, action_best])
            output_dir = args.output_dir / record["episode"]
            output_dir.mkdir(exist_ok=True)
            result_path = output_dir / record["snapshot"]
            np.savez_compressed(
                result_path,
                source_snapshot=np.asarray(str(record["source_path"])),
                source_sha256=np.asarray(record["source_sha256"]),
                resume_parent=np.asarray(
                    "" if args.resume_dir is None
                    else str(args.resume_dir / record["episode"] / record["snapshot"])
                ),
                start_names=np.asarray(record["start_names"]),
                initial_knots=starts[local].detach().cpu().numpy(),
                initial_cost=initial_cost[local].detach().cpu().numpy(),
                optimized_knots=knots[local].detach().cpu().numpy(),
                knot_actions=knot_actions[local].detach().cpu().numpy(),
                knot_cost_optimization=knot_cost[local].detach().cpu().numpy(),
                knot_cost_replay=knot_replay[local].detach().cpu().numpy(),
                knot_best_trace=knot_trace[local],
                optimized_actions=actions[local].detach().cpu().numpy(),
                action_cost_optimization=action_cost[local].detach().cpu().numpy(),
                action_cost_replay=action_replay[local].detach().cpu().numpy(),
                action_best_trace=action_trace[local],
                knot_best_index=np.asarray(knot_best),
                action_best_index=np.asarray(action_best),
            )
            rows.append({
                "episode": record["episode"],
                "snapshot": record["snapshot"],
                "source": str(record["source_path"]),
                "source_sha256": record["source_sha256"],
                "warm_cost": warm_cost,
                "teacher_cost": teacher_cost,
                "j16_best_found": j16,
                "j100_best_found": j100,
                "parameterization_gap": j16 - j100,
                "actor_gap_reference": "computed separately on two feedback contexts",
                "j16_top3_relative_spread": float(
                    (knot_sorted[-1] - knot_sorted[0])
                    / torch.clamp(knot_sorted[0].abs(), min=1e-12)
                ),
                "j100_top3_relative_spread": float(
                    (action_sorted[-1] - action_sorted[0])
                    / torch.clamp(action_sorted[0].abs(), min=1e-12)
                ),
                "j16_replay_abs_error": float(
                    torch.abs(knot_replay[local, knot_best] - knot_cost[local, knot_best])
                ),
                "j100_replay_abs_error": float(
                    torch.abs(action_replay[local, action_best] - action_cost[local, action_best])
                ),
                "result": str(result_path),
            })
        print(
            f"[{min(batch_start + len(batch_records), len(records)):04d}/{len(records):04d}] "
            f"J16={np.mean([row['j16_best_found'] for row in rows]):.3f} "
            f"J100={np.mean([row['j100_best_found'] for row in rows]):.3f} "
            f"elapsed={time.time()-started:.1f}s",
            flush=True,
        )
    rows.sort(key=lambda row: (row["episode"], row["snapshot"]))
    with (args.output_dir / "per_snapshot.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = (
        "warm_cost", "teacher_cost", "j16_best_found", "j100_best_found",
        "parameterization_gap", "j16_top3_relative_spread",
        "j100_top3_relative_spread", "j16_replay_abs_error",
        "j100_replay_abs_error",
    )
    summary = {
        "format_version": 1,
        "semantics": "best-found numerical oracle; not a proven global optimum",
        "gradient_scope": "offline fixed-DBM diagnosis only; never a policy input",
        "source": str(args.source),
        "split_manifest": str(args.splits),
        "split": args.split,
        "episodes": episodes,
        "snapshot_count": len(rows),
        "feedback_context_count_for_actor_comparison": 2 * len(rows),
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "resume_dir": None if args.resume_dir is None else str(args.resume_dir),
        "random_starts": args.random_starts,
        "knot_steps": args.knot_steps,
        "action_steps": args.action_steps,
        "action_scope": (
            "J16 interpolation placeholder; no J100 optimization"
            if args.action_steps == 0 else "independent full-action optimization"
        ),
        "batch_size": args.batch_size,
        "mean": {
            metric: (
                float(np.mean([row[metric] for row in rows if row[metric] is not None]))
                if any(row[metric] is not None for row in rows) else None
            )
            for metric in metrics
        },
        "p95": {
            metric: (
                float(np.quantile(
                    [row[metric] for row in rows if row[metric] is not None], 0.95
                ))
                if any(row[metric] is not None for row in rows) else None
            )
            for metric in metrics
        },
        "maximum": {
            metric: (
                float(np.max([row[metric] for row in rows if row[metric] is not None]))
                if any(row[metric] is not None for row in rows) else None
            )
            for metric in metrics
        },
        "rows": rows,
        "test_policy": "test split not loaded or evaluated",
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["mean"], indent=2), flush=True)


if __name__ == "__main__":
    main()
