#!/usr/bin/env python3
"""B1 knot-layout projection oracle (pre-architecture check).

Before the A2 token/temporal-head arms, test whether the knot layout itself
loses teacher signal. For a stratified sample of the 1800-state pool, take
the guarded consensus-64 teacher knots, expand them to the 50-step action
sequence under the current uniform layout, then re-project that sequence
onto candidate 8-knot layouts (knot value = sequence value at the knot time,
reconstruction = linear interpolation between knot times with head/tail hold)
and re-roll out. The uniform layout round-trips to identity, so its cost
equals the teacher cost and any other layout's excess is pure layout
projection loss. Non-uniform candidates define explicit endpoints/tail-hold;
the historical 7-point grid is not a complete 8-knot deployment definition
and is excluded.
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
from car_dynamics.controllers_torch.mppi import (
    TorchMPPICostWeights,
    TorchMPPIParams,
)
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from run_mppi_proximal_search_phase1a import (
    DEFAULT_GT_TRAIN,
    DEFAULT_REPLAY_LABELS,
    DEFAULT_SCENARIO_PLAN,
    load_states,
)

HORIZON = 50
LAYOUTS = {
    "uniform": [0.0, 7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0],
    "front_dense": [0.0, 3.0, 7.0, 12.0, 18.0, 26.0, 35.0, 49.0],
    "rear_dense": [0.0, 14.0, 23.0, 31.0, 38.0, 43.0, 46.0, 49.0],
    "mid_dense": [0.0, 10.0, 17.0, 23.0, 28.0, 33.0, 40.0, 49.0],
}
DEFAULT_POOL = Path("outputs/mppi_proposal/consensus64_labels_pool_20260818_v1")
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/knot_layout_oracle_20260818_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def reproject(sequence: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Sample the 50-step sequence at knot times, rebuild by linear
    interpolation with head/tail hold, return the [50, 2] sequence."""
    index = np.arange(HORIZON, dtype=np.float64)
    knots = np.stack([
        np.interp(times, index, sequence[:, dim]) for dim in range(2)
    ], axis=1).astype(np.float32)
    return np.stack([
        np.interp(index, times, knots[:, dim]) for dim in range(2)
    ], axis=1).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    labels = np.load(args.pool / "labels.npz", allow_pickle=False)
    manifest = json.loads((args.pool / "manifest.json").read_text())
    keys = [f"{row['episode']}#{row['snapshot']}" for row in manifest["states"]]
    if [str(v) for v in labels["episodes"]] != keys:
        raise AssertionError("pool labels/manifest order mismatch")

    loader_args = SimpleNamespace(
        replay_labels=DEFAULT_REPLAY_LABELS,
        gt_train=DEFAULT_GT_TRAIN,
        scenario_plan=DEFAULT_SCENARIO_PLAN,
        repeat=args.repeat,
    )
    by_key = {f"{s['episode']}#{s['snapshot']}": s for s in load_states(loader_args)}
    states_all = [by_key[key] for key in keys]
    stride = len(states_all) / args.sample
    rows_idx = [int(i * stride) for i in range(args.sample)]
    states = [states_all[i] for i in rows_idx]
    teacher_knots = labels["label_knots"][rows_idx]

    device = torch.device(args.device)
    params = TorchMPPIParams(**json.loads(states[0]["mppi_params"]))
    weights = TorchMPPICostWeights(**json.loads(states[0]["cost_weights"]))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(states[0]["dbm_params"]))
    )

    def rollout(sequences: np.ndarray) -> np.ndarray:
        actions = torch.as_tensor(
            sequences[:, None], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            return batched_cost(
                backend, weights, actions,
                torch.as_tensor(
                    np.stack([s["initial_state_six"] for s in states]),
                    dtype=torch.float32, device=device,
                ),
                torch.as_tensor(
                    np.stack([s["current_action"] for s in states]),
                    dtype=torch.float32, device=device,
                ),
                torch.as_tensor(
                    np.stack([s["reference"] for s in states]),
                    dtype=torch.float32, device=device,
                ),
            ).squeeze(-1).cpu().numpy().astype(np.float64)

    with torch.no_grad():
        teacher_seq = interpolate_knots(
            torch.as_tensor(
                teacher_knots.reshape(-1, 8, 2)[:, None],
                dtype=torch.float32, device=device,
            ),
            HORIZON,
        )[:, 0].cpu().numpy()

    costs = {}
    for name, times in LAYOUTS.items():
        times_arr = np.asarray(times, np.float64)
        rebuilt = np.stack([
            reproject(teacher_seq[i], times_arr) for i in range(len(states))
        ])
        costs[name] = rollout(rebuilt)
    teacher_cost = rollout(teacher_seq.astype(np.float32))

    results = {}
    for name, cost in costs.items():
        excess = cost - teacher_cost
        results[name] = {
            "cost_mean": float(np.mean(cost)),
            "excess_mean": float(np.mean(excess)),
            "excess_median": float(np.median(excess)),
            "excess_p95": float(np.quantile(excess, 0.95)),
            "worse_than_teacher_fraction": float(np.mean(excess > 1e-6)),
            "uniform_identity_check": (
                float(np.max(np.abs(excess))) if name == "uniform" else None
            ),
        }
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "B1_KNOT_LAYOUT_PROJECTION_ORACLE_ACTOR_FROZEN",
        "sources": {"pool": str(args.pool)},
        "protocol": {
            "sample": len(states),
            "layouts": LAYOUTS,
            "projection": (
                "knot value = teacher sequence value at knot time; rebuild "
                "by linear interpolation with head/tail hold; uniform layout "
                "round-trips to identity"
            ),
        },
        "teacher_cost_mean": float(np.mean(teacher_cost)),
        "results": results,
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({
        name: {
            "excess_mean": round(value["excess_mean"], 4),
            "excess_p95": round(value["excess_p95"], 4),
            "identity": value["uniform_identity_check"],
        }
        for name, value in results.items()
    }, indent=1))


if __name__ == "__main__":
    main()
