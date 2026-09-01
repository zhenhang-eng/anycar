#!/usr/bin/env python3
"""Generate guarded consensus-64 labels for the 600 Phase 1b states.

Mirrors the 100-state consensus budget study at budget 64: three direction
banks (Hadamard + two seeded Givens bases), each 2 ring radii x 16
directions x antipodal = 64 candidates, plus the a0/warm anchors; the
consensus center is the elementwise mean of the three bank-best knots,
re-evaluated by a fresh DBM rollout, and falls back to the anchor label
whenever it does not beat min(J_a0, J_warm) + 1e-6. Label knots fall back
to a0 (stay) on fallback, matching the Phase 1b convention.
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
from generate_dbm_j16_local_curvature_labels import hadamard_directions
from generate_dbm_proposal_teacher import sha256_file
from run_mppi_proximal_search_phase1a import (
    DEFAULT_GT_TRAIN,
    DEFAULT_REPLAY_LABELS,
    DEFAULT_SCENARIO_PLAN,
    RING_RADII,
    load_states,
    ring_candidates,
    seed_bank_directions,
)


DEFAULT_LABELS = Path(
    "outputs/mppi_proposal/proximal_search_phase1b_20260818_v1/labels.npz"
)
DEFAULT_MANIFEST = Path(
    "outputs/mppi_proposal/proximal_search_phase1b_20260818_v1/manifest.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/consensus64_labels_20260818_v1"
)
GUARD = 1e-6


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
    labels = dict(np.load(args.labels, allow_pickle=False))
    manifest = json.loads(args.manifest.read_text())
    states_meta = manifest["states"]

    loader_args = SimpleNamespace(
        replay_labels=args.replay_labels,
        gt_train=args.gt_train,
        scenario_plan=args.scenario_plan,
        repeat=args.repeat,
    )
    from run_mppi_proximal_search_phase1a import select_states

    loaded = select_states(load_states(loader_args), len(states_meta))
    if [s["episode"] for s in loaded] != [s["episode"] for s in states_meta]:
        raise AssertionError("state selection order mismatch")

    params = TorchMPPIParams(**json.loads(str(loaded[0]["mppi_params"])))
    weights = TorchMPPICostWeights(**json.loads(str(loaded[0]["cost_weights"])))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(str(loaded[0]["dbm_params"])))
    )
    device = torch.device(args.device)
    low = np.asarray(params.action_min, np.float32)
    high = np.asarray(params.action_max, np.float32)
    bases = {
        1: hadamard_directions(),
        2: seed_bank_directions(2),
        3: seed_bank_directions(3),
    }

    def evaluate(state, candidates, cache):
        clipped = np.clip(candidates, low, high).astype(np.float32)
        keys_local = [np.round(value, 6).tobytes() for value in clipped]
        pending = {}
        for row, key in enumerate(keys_local):
            if key not in cache:
                pending.setdefault(key, row)
        if pending:
            pending_keys = list(pending)
            pending_array = np.asarray(
                [clipped[row] for row in pending.values()], np.float32
            )
            count = len(pending_array)
            with torch.no_grad():
                chunk = torch.as_tensor(
                    pending_array[:, None], dtype=torch.float32, device=device
                )
                actions = interpolate_knots(chunk, params.horizon)
                costs = batched_cost(
                    backend, weights, actions,
                    torch.as_tensor(
                        np.repeat(
                            state["initial_state_six"][None], count, axis=0
                        ),
                        dtype=torch.float32, device=device,
                    ),
                    torch.as_tensor(
                        np.repeat(state["current_action"][None], count, axis=0),
                        dtype=torch.float32, device=device,
                    ),
                    torch.as_tensor(
                        np.repeat(state["reference"][None], count, axis=0),
                        dtype=torch.float32, device=device,
                    ),
                ).squeeze(-1).cpu().numpy()
            for key, cost in zip(pending_keys, costs):
                cache[key] = float(cost)
        return np.asarray([cache[key] for key in keys_local], np.float32)

    consensus_rows = []
    new_evals = 0
    for position, state in enumerate(loaded):
        cache = {}
        anchor_costs = evaluate(
            state, np.stack([state["a0"], state["warm"]]), cache
        )
        new_evals += 2
        teachers = []
        for bank_id, basis in bases.items():
            bank = ring_candidates(
                state["a0"], state["sigma"], RING_RADII[:2], basis
            )
            costs = evaluate(state, bank, cache)
            new_evals += int(np.sum(np.isfinite(costs)))
            pool = np.concatenate([state["a0"][None], state["warm"][None], bank])
            pool_costs = np.concatenate([anchor_costs, costs])
            teachers.append(pool[int(np.argmin(pool_costs))])
        consensus = np.clip(
            np.stack(teachers).mean(axis=0), low, high
        ).astype(np.float32)
        cost = float(evaluate(state, consensus[None], cache)[0])
        new_evals += 1
        anchor_best = min(float(anchor_costs[0]), float(anchor_costs[1]))
        fallback = cost > anchor_best + GUARD
        label = (
            state["a0"].astype(np.float32)
            if fallback else consensus
        )
        consensus_rows.append({
            "consensus_knots": consensus,
            "label_knots": label,
            "j_consensus": cost,
            "j_teacher": anchor_best if fallback else cost,
            "fallback": bool(fallback),
        })
        if (position + 1) % 25 == 0:
            print(f"[{position + 1:03d}/{len(loaded):03d}] evals={new_evals}",
                  flush=True)

    output = {
        "episodes": labels["episodes"],
        "speeds": labels["speeds"],
        "scenarios": labels["scenarios"],
        "j_a0": labels["j_a0"],
        "j_warm": labels["j_warm"],
        "j16": labels["j16"],
        "consensus_knots": np.stack([
            row["consensus_knots"] for row in consensus_rows
        ]).astype(np.float32),
        "label_knots": np.stack([
            row["label_knots"] for row in consensus_rows
        ]).astype(np.float32),
        "j_teacher": np.asarray([
            row["j_teacher"] for row in consensus_rows
        ], np.float32),
        "fallback": np.asarray([
            row["fallback"] for row in consensus_rows
        ], bool),
        "stay": np.asarray([
            row["fallback"] for row in consensus_rows
        ], bool),
    }
    np.savez_compressed(args.output / "labels.npz", **output)
    fallback_rate = float(np.mean(output["fallback"]))
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "CONSENSUS64_LABELS_GENERATED_GUARDED",
        "sources": {
            "labels": str(args.labels.resolve()),
            "labels_sha256": sha256_file(args.labels),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
        },
        "counts": {
            "states": len(loaded),
            "new_evaluations": new_evals,
            "budget_per_state": 195,
        },
        "j_teacher_mean": float(np.mean(output["j_teacher"])),
        "fallback_rate": fallback_rate,
        "contract": {
            "guard": "consensus fallback to a0 when not beating min(J_a0,J_warm)+1e-6",
            "labels": "three-bank consensus at budget 64 (2 radii x 16 dirs)",
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "output": str((args.output / "summary.json").resolve()),
        "qualification": summary["qualification"],
        "j_teacher_mean": summary["j_teacher_mean"],
        "fallback_rate": fallback_rate,
        "new_evaluations": new_evals,
    }, indent=2))


if __name__ == "__main__":
    main()
