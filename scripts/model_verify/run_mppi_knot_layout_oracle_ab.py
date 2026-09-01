#!/usr/bin/env python3
"""Differentiable DBM capacity oracle for alternative 8-knot time layouts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from generate_dbm_direct_gt_validation import batched_cost, bounded_raw, bounded_value, interpolate_knots
from generate_dbm_proposal_teacher import sha256_file
from run_mppi_knot_layout_search_ab import (
    HORIZON,
    LAYOUTS,
    distribution,
    interpolation_matrix,
    sample_layout_knots,
)
from run_mppi_proximal_search_phase1a import (
    DEFAULT_GT_TRAIN,
    DEFAULT_REPLAY_LABELS,
    DEFAULT_SCENARIO_PLAN,
    load_states,
    select_states,
)


DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/knot_layout_capacity_oracle_20260825_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-labels", type=Path, default=DEFAULT_REPLAY_LABELS)
    parser.add_argument("--gt-train", type=Path, default=DEFAULT_GT_TRAIN)
    parser.add_argument("--scenario-plan", type=Path, default=DEFAULT_SCENARIO_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=0, choices=(0, 1))
    parser.add_argument("--random-starts", type=int, default=3)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=26082571)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    gt_summary = json.loads((args.gt_train / "summary.json").read_text())
    if gt_summary.get("split") != "train":
        raise AssertionError("layout oracle must use train-only states")
    loader = argparse.Namespace(
        replay_labels=args.replay_labels,
        gt_train=args.gt_train,
        scenario_plan=args.scenario_plan,
        repeat=args.repeat,
    )
    states = select_states(load_states(loader), args.states)
    params = TorchMPPIParams(**json.loads(states[0]["mppi_params"]))
    weights = TorchMPPICostWeights(**json.loads(states[0]["cost_weights"]))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(states[0]["dbm_params"]))
    )
    device = torch.device(args.device)
    initial_state = torch.as_tensor(
        np.stack([state["initial_state_six"] for state in states]),
        dtype=torch.float32, device=device,
    )
    current_action = torch.as_tensor(
        np.stack([state["current_action"] for state in states]),
        dtype=torch.float32, device=device,
    )
    reference = torch.as_tensor(
        np.stack([state["reference"] for state in states]),
        dtype=torch.float32, device=device,
    )
    uniform_anchor_knots = np.stack([
        np.stack((state["a0"], state["warm"])) for state in states
    ])
    with torch.no_grad():
        uniform_anchor_sequences = interpolate_knots(
            torch.as_tensor(uniform_anchor_knots, dtype=torch.float32, device=device),
            HORIZON,
        ).cpu().numpy()
    rng = np.random.default_rng(args.seed)
    action_min = torch.as_tensor(params.action_min, dtype=torch.float32, device=device)
    action_max = torch.as_tensor(params.action_max, dtype=torch.float32, device=device)
    sigma = np.stack([state["sigma"] for state in states]).astype(np.float32)
    # Strictly pair the normalized random starts across layouts.  Layouts
    # differ only through their knot times and projected deterministic anchors.
    paired_random_noise = rng.normal(
        size=(len(states), args.random_starts, 8, 2)
    ).astype(np.float32)
    layout_costs: dict[str, np.ndarray] = {}
    layout_knots: dict[str, np.ndarray] = {}
    trace_by_layout: dict[str, list[float]] = {}

    for name, times_list in LAYOUTS.items():
        times = np.asarray(times_list, np.float64)
        matrix = torch.as_tensor(
            interpolation_matrix(times), dtype=torch.float32, device=device
        )
        projected = np.stack([
            np.stack([
                sample_layout_knots(uniform_anchor_sequences[row, start], times)
                for start in range(2)
            ])
            for row in range(len(states))
        ])
        random_knots = (
            projected[:, :1]
            + paired_random_noise * sigma[:, None, None, :]
        )
        starts = np.clip(
            np.concatenate((projected, random_knots), axis=1),
            np.asarray(params.action_min), np.asarray(params.action_max),
        ).astype(np.float32)
        raw = bounded_raw(
            torch.as_tensor(starts, dtype=torch.float32, device=device),
            action_min, action_max,
        )
        optimizer = torch.optim.Adam((raw,), lr=args.learning_rate)
        best_cost = torch.full(
            starts.shape[:2], torch.inf, dtype=torch.float32, device=device
        )
        best_knots = torch.as_tensor(starts, device=device).clone()
        trace = []
        for step in range(args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            knots = bounded_value(raw, action_min, action_max)
            actions = torch.einsum("hk,bskc->bshc", matrix, knots)
            cost = batched_cost(
                backend, weights, actions, initial_state, current_action, reference
            )
            improved = torch.isfinite(cost) & (cost < best_cost)
            best_cost = torch.where(improved, cost.detach(), best_cost)
            best_knots = torch.where(
                improved[..., None, None], knots.detach(), best_knots
            )
            if step % 10 == 0 or step == args.steps:
                trace.append(float(best_cost.min(dim=1).values.mean().detach()))
            if step == args.steps:
                break
            cost[torch.isfinite(cost)].sum().backward()
            torch.nn.utils.clip_grad_norm_((raw,), 1000.0)
            optimizer.step()
        best_index = best_cost.argmin(dim=1)
        row = torch.arange(len(states), device=device)
        layout_costs[name] = best_cost[row, best_index].cpu().numpy()
        layout_knots[name] = best_knots[row, best_index].cpu().numpy()
        trace_by_layout[name] = trace
        print(f"{name}: mean={layout_costs[name].mean():.6f}", flush=True)

    uniform = layout_costs["uniform"]
    results = {}
    for name, costs in layout_costs.items():
        delta = costs - uniform
        results[name] = {
            "cost": distribution(costs),
            "paired_cost_minus_uniform": distribution(delta),
            "wins_ties_losses_vs_uniform": [
                int(np.sum(delta < -1e-5)),
                int(np.sum(np.abs(delta) <= 1e-5)),
                int(np.sum(delta > 1e-5)),
            ],
            "optimization_trace_mean_best": trace_by_layout[name],
        }
    front = min(
        (name for name in LAYOUTS if name != "uniform"),
        key=lambda name: results[name]["cost"]["mean"],
    )
    delta = layout_costs[front] - uniform
    qualification = (
        "FRONT_DENSE_CAPACITY_ORACLE_PASS"
        if float(np.mean(delta)) < 0 and float(np.median(delta)) <= 0
        else "FRONT_DENSE_CAPACITY_ORACLE_FAIL_KEEP_UNIFORM"
    )
    checks = {
        "train_only_gt": gt_summary.get("split") == "train",
        "finite": bool(all(np.isfinite(value).all() for value in layout_costs.values())),
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "manifest": {
            "gt_train": str(args.gt_train.resolve()),
            "gt_train_summary_sha256": sha256_file(args.gt_train / "summary.json"),
            "scenario_plan_sha256": sha256_file(args.scenario_plan),
            "layouts": LAYOUTS,
            "states": args.states,
            "starts_per_state": 2 + args.random_starts,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "random_starts_paired_across_layouts": True,
            "objective": "deterministic differentiable DBM J50",
        },
        "results": results,
        "best_front_dense": front,
        "checks": checks,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "solutions.npz",
        episode=np.asarray([state["episode"] for state in states]),
        snapshot=np.asarray([state["snapshot"] for state in states]),
        speed=np.asarray([state["speed"] for state in states], np.float32),
        scenario=np.asarray([state["scenario"] for state in states]),
        **{f"{name}_cost": value for name, value in layout_costs.items()},
        **{f"{name}_knots": value for name, value in layout_knots.items()},
    )
    print(json.dumps({
        "qualification": qualification,
        "best_front_dense": front,
        "results": results,
        "checks": checks,
    }, indent=2))


if __name__ == "__main__":
    main()
