#!/usr/bin/env python3
"""B0: per-state action-time sensitivity audit (pure dynamics, no cost).

For each of the 600 Phase 1b states, differentiable DBM rollout from the
a0 anchor computes:
- Jacobians of the terminal state components [x, y, yaw, vx] w.r.t. the 16
  knot dimensions (steering and acceleration separated);
- Jacobians of the terminal position (and yaw / vx) w.r.t. each of the 50
  interpolated action time steps, giving the temporal influence profile.

Outputs:
- S: influence-normalization diagonal (RMS across states of terminal
  position sensitivity per knot dim) plus yaw/vx variants, both raw and
  per-fold stability tables (train-only states only enter the estimate);
- early/middle/late cumulative influence thirds;
- endpoint (half-support) contrast of the last action steps;
- non-uniform knot candidate positions via quantiles of the cumulative
  per-time influence;
- speed x scenario strata tables.

No cost gradients are used anywhere: the objective is terminal state
displacement, immune to cost branch flips.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPIParams
from generate_dbm_direct_gt_validation import interpolate_knots
from generate_dbm_proposal_teacher import sha256_file
from run_mppi_proximal_search_phase1a import (
    DEFAULT_GT_TRAIN,
    DEFAULT_REPLAY_LABELS,
    DEFAULT_SCENARIO_PLAN,
    load_states,
    select_states,
)


DEFAULT_LABELS = Path(
    "outputs/mppi_proposal/proximal_search_phase1b_20260818_v1/labels.npz"
)
DEFAULT_MANIFEST = Path(
    "outputs/mppi_proposal/proximal_search_phase1b_20260818_v1/manifest.json"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/b0_sensitivity_20260818_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--replay-labels", type=Path, default=DEFAULT_REPLAY_LABELS)
    parser.add_argument("--scenario-plan", type=Path, default=DEFAULT_SCENARIO_PLAN)
    parser.add_argument("--gt-train", type=Path, default=DEFAULT_GT_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    device = torch.device(args.device)
    labels = dict(np.load(args.labels, allow_pickle=False))
    manifest = json.loads(args.manifest.read_text())
    states_meta = manifest["states"]
    loader_args = SimpleNamespace(
        replay_labels=args.replay_labels,
        gt_train=args.gt_train,
        scenario_plan=args.scenario_plan,
        repeat=args.repeat,
    )
    loaded = select_states(load_states(loader_args), len(states_meta))
    if [s["episode"] for s in loaded] != [s["episode"] for s in states_meta]:
        raise AssertionError("state selection order mismatch")

    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(str(loaded[0]["dbm_params"])))
    )
    horizon = backend.horizon

    count = len(loaded)
    knot_jacobians = np.zeros((count, 4, 16), np.float32)
    time_position = np.zeros((count, horizon), np.float32)
    time_yaw = np.zeros((count, horizon), np.float32)
    time_vx = np.zeros((count, horizon), np.float32)
    for position, state in enumerate(loaded):
        knots = torch.from_numpy(state["a0"].astype(np.float32)).to(device)[None]
        knots.requires_grad_(True)
        actions = interpolate_knots(knots, horizon)
        actions.retain_grad()
        initial = torch.from_numpy(
            state["initial_state_six"].astype(np.float32)
        ).to(device)[None]
        full = backend.rollout_full_state_differentiable(initial, actions)
        terminal = full[0, -1]
        for output_index in range(4):
            gradient = torch.autograd.grad(
                terminal[output_index], knots, retain_graph=True
            )[0]
            knot_jacobians[position, output_index] = (
                gradient.detach().cpu().numpy().flatten()
            )
        for name, store in (
            ("position", time_position),
            ("yaw", time_yaw),
            ("vx", time_vx),
        ):
            indices = {"position": (0, 1), "yaw": (2,), "vx": (3,)}[name]
            rows = [terminal[index] for index in indices]
            per_time = []
            for row in rows:
                gradient = torch.autograd.grad(
                    row, actions, retain_graph=True,
                )[0][0]
                per_time.append(gradient.detach())
            magnitude = torch.stack(per_time).norm(dim=0)
            store[position] = (
                (magnitude[:, 0] ** 2 + magnitude[:, 1] ** 2).sqrt()
            ).cpu().numpy()
        if (position + 1) % 100 == 0:
            print(f"[{position + 1:03d}/{count:03d}]", flush=True)

    # Knot-space normalization matrix S (position objective default).
    position_knot = np.linalg.norm(knot_jacobians[:, :2, :], axis=1)  # [n,16]
    yaw_knot = np.abs(knot_jacobians[:, 2, :])
    vx_knot = np.abs(knot_jacobians[:, 3, :])
    s_position = np.sqrt((position_knot ** 2).mean(axis=0))
    s_yaw = np.sqrt((yaw_knot ** 2).mean(axis=0))
    s_vx = np.sqrt((vx_knot ** 2).mean(axis=0))
    # Fold stability: split episodes into 3 folds as in A0, refit S per fold.
    episodes = np.asarray([
        value.split("#")[0] for value in labels["episodes"].astype(str)
    ])
    speeds = labels["speeds"].astype(np.float32)
    scenarios = labels["scenarios"].astype(str)
    fold_of_state = np.zeros(count, np.int64)
    cell_episodes = {}
    for episode in np.unique(episodes):
        mask = episodes == episode
        key = (round(float(speeds[mask][0]), 2), str(scenarios[mask][0]))
        cell_episodes.setdefault(key, []).append(episode)
    for key in sorted(cell_episodes):
        for fold, episode in enumerate(sorted(cell_episodes[key])):
            fold_of_state[episodes == episode] = fold
    fold_s = []
    for fold in range(3):
        mask = fold_of_state != fold
        fold_s.append(np.sqrt((position_knot[mask] ** 2).mean(axis=0)))
    fold_s = np.stack(fold_s)
    relative_spread = (
        fold_s.max(axis=0) - fold_s.min(axis=0)
    ) / (fold_s.mean(axis=0) + 1e-9)

    thirds = (slice(0, 17), slice(17, 34), slice(34, horizon))
    time_mean = time_position.mean(axis=0)
    third_mass = [
        float(time_mean[third].sum() / (time_mean.sum() + 1e-12))
        for third in thirds
    ]
    endpoint_contrast = {
        "t49_over_median_late": float(
            time_mean[-1] / (np.median(time_mean[34:]) + 1e-12)
        ),
        "t48_over_median_late": float(
            time_mean[-2] / (np.median(time_mean[34:]) + 1e-12)
        ),
    }
    cumulative = np.cumsum(time_mean) / (time_mean.sum() + 1e-12)
    knot_positions = [
        int(np.searchsorted(cumulative, quantile))
        for quantile in np.linspace(0, 1, 9)[1:-1]
    ]
    strata = {}
    for speed in np.unique(speeds):
        mask = speeds == speed
        strata[f"speed_{round(float(speed), 2)}"] = {
            "count": int(mask.sum()),
            "third_mass": [
                float(time_position[mask][:, third].sum()
                      / (time_position[mask].sum() + 1e-12))
                for third in thirds
            ],
        }

    np.savez_compressed(
        args.output / "sensitivity.npz",
        knot_jacobians=knot_jacobians,
        time_position=time_position,
        time_yaw=time_yaw,
        time_vx=time_vx,
        fold_of_state=fold_of_state,
    )
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "B0_SENSITIVITY_AUDITED_ACTOR_FROZEN",
        "sources": {
            "labels": str(args.labels.resolve()),
            "labels_sha256": sha256_file(args.labels),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
        },
        "S_position_knot": s_position.tolist(),
        "S_yaw_knot": s_yaw.tolist(),
        "S_vx_knot": s_vx.tolist(),
        "fold_S_relative_spread_max": float(relative_spread.max()),
        "thirds_position_mass": third_mass,
        "endpoint_contrast": endpoint_contrast,
        "nonuniform_knot_positions_t": knot_positions,
        "strata": strata,
        "contract": {
            "objective": "terminal state displacement; no cost gradients",
            "actor_frozen": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "output": str((args.output / "summary.json").resolve()),
        "S_position_knot": [round(v, 4) for v in s_position.tolist()],
        "fold_S_relative_spread_max": summary[
            "fold_S_relative_spread_max"
        ],
        "thirds_position_mass": third_mass,
        "endpoint_contrast": endpoint_contrast,
        "nonuniform_knot_positions_t": knot_positions,
    }, indent=2))


if __name__ == "__main__":
    main()
