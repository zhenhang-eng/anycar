#!/usr/bin/env python3
"""Generate guarded consensus-64 labels for the full diverse-train pool.

Coverage-round label generation (review 11.50): one pass over all 1800
diverse-train states so nested learning-curve subsets (600/1200/1800) can be
built by slicing without regenerating labels. Same contract as the 600-state
generator: three direction banks (Hadamard + two Givens bases) at budget 64,
consensus = elementwise mean of bank-best knots, re-rolled out, guarded
fallback to the a0 stay label whenever it does not beat
min(J_a0, J_warm) + 1e-6. Anchor policy is the 2026-08-06 replay bootstrap
actor (checkpoint hash recorded), identical to Phase 1b. The expansion pool
(1350 states) is deliberately excluded this round: its only available anchor
is the frozen BC center, a different policy, which would confound the
learning curve; extending coverage beyond diverse requires an anchor-policy
decision first.
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

DEFAULT_OUTPUT = Path("outputs/mppi_proposal/consensus64_labels_pool_20260818_v1")
GUARD = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-labels", type=Path, default=DEFAULT_REPLAY_LABELS)
    parser.add_argument("--gt-train", type=Path, default=DEFAULT_GT_TRAIN)
    parser.add_argument("--scenario-plan", type=Path, default=DEFAULT_SCENARIO_PLAN)
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
    loader_args = SimpleNamespace(
        replay_labels=args.replay_labels,
        gt_train=args.gt_train,
        scenario_plan=args.scenario_plan,
        repeat=args.repeat,
    )
    states = sorted(
        load_states(loader_args),
        key=lambda item: (item["speed"], item["scenario"], item["residual_norm"]),
    )
    count = len(states)
    params = TorchMPPIParams(**json.loads(states[0]["mppi_params"]))
    weights = TorchMPPICostWeights(**json.loads(states[0]["cost_weights"]))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(states[0]["dbm_params"]))
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
            size = len(pending_array)
            with torch.no_grad():
                chunk = torch.as_tensor(
                    pending_array[:, None], dtype=torch.float32, device=device
                )
                actions = interpolate_knots(chunk, params.horizon)
                costs = batched_cost(
                    backend, weights, actions,
                    torch.as_tensor(
                        np.repeat(state["initial_state_six"][None], size, axis=0),
                        dtype=torch.float32, device=device,
                    ),
                    torch.as_tensor(
                        np.repeat(state["current_action"][None], size, axis=0),
                        dtype=torch.float32, device=device,
                    ),
                    torch.as_tensor(
                        np.repeat(state["reference"][None], size, axis=0),
                        dtype=torch.float32, device=device,
                    ),
                ).squeeze(-1).cpu().numpy()
            for key, cost in zip(pending_keys, costs):
                cache[key] = float(cost)
        return np.asarray([cache[key] for key in keys_local], np.float32)

    rows = []
    new_evals = 0
    for position, state in enumerate(states):
        cache = {}
        anchor_costs = evaluate(
            state, np.stack([state["a0"], state["warm"]]), cache
        )
        new_evals += 2
        teachers = []
        for basis in bases.values():
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
        consensus_cost = float(evaluate(state, consensus[None], cache)[0])
        new_evals += 1
        anchor_best = min(float(anchor_costs[0]), float(anchor_costs[1]))
        fallback = consensus_cost > anchor_best + GUARD
        rows.append({
            "episode": state["episode"],
            "snapshot": state["snapshot"],
            "scenario": state["scenario"],
            "speed": state["speed"],
            "source": state["source_path"],
            "source_sha256": state["source_hash"],
            "anchor_checkpoint_sha256": state["anchor_checkpoint_sha256"],
            "j_a0": float(anchor_costs[0]),
            "j_warm": float(anchor_costs[1]),
            "j16": state["j16_cost"],
            "consensus_knots": consensus,
            "label_knots": (
                state["a0"].astype(np.float32) if fallback else consensus
            ),
            "j_teacher": anchor_best if fallback else consensus_cost,
            "fallback": bool(fallback),
        })
        if (position + 1) % 50 == 0:
            print(
                f"[{position + 1:04d}/{count:04d}] evals={new_evals}",
                flush=True,
            )

    output = {
        "episodes": np.asarray([
            f"{row['episode']}#{row['snapshot']}" for row in rows
        ]),
        "speeds": np.asarray([row["speed"] for row in rows], np.float32),
        "scenarios": np.asarray([row["scenario"] for row in rows]),
        "j_a0": np.asarray([row["j_a0"] for row in rows], np.float32),
        "j_warm": np.asarray([row["j_warm"] for row in rows], np.float32),
        "j16": np.asarray([row["j16"] for row in rows], np.float32),
        "consensus_knots": np.stack([
            row["consensus_knots"] for row in rows
        ]).astype(np.float32),
        "label_knots": np.stack([
            row["label_knots"] for row in rows
        ]).astype(np.float32),
        "j_teacher": np.asarray([
            row["j_teacher"] for row in rows
        ], np.float32),
        "fallback": np.asarray([row["fallback"] for row in rows], bool),
        "stay": np.asarray([row["fallback"] for row in rows], bool),
    }
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "labels.npz", **output)
    fallback_rate = float(np.mean(output["fallback"]))
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "CONSENSUS64_POOL_LABELS_GENERATED_GUARDED",
        "sources": {
            "replay_labels": str(args.replay_labels),
            "gt_train": str(args.gt_train),
            "gt_train_summary_sha256": sha256_file(
                args.gt_train / "summary.json"
            ),
            "scenario_plan_sha256": sha256_file(args.scenario_plan),
            "anchor_checkpoint_sha256": rows[0]["anchor_checkpoint_sha256"],
        },
        "counts": {
            "states": count,
            "new_evaluations": new_evals,
            "budget_per_state": 195,
        },
        "j_teacher_mean": float(np.mean(output["j_teacher"])),
        "fallback_rate": fallback_rate,
        "note": (
            "pool ordered by (speed, scenario, residual_norm); learning-curve "
            "subsets via even-spacing slices are nested by construction; "
            "expansion pool excluded pending an anchor-policy decision"
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    (args.output / "manifest.json").write_text(json.dumps({
        "states": [
            {
                "episode": row["episode"], "snapshot": row["snapshot"],
                "scenario": row["scenario"], "speed": row["speed"],
                "source": row["source"], "source_sha256": row["source_sha256"],
                "anchor_checkpoint_sha256": row["anchor_checkpoint_sha256"],
            }
            for row in rows
        ],
    }, indent=1))
    print(json.dumps({
        "states": count,
        "new_evaluations": new_evals,
        "j_teacher_mean": summary["j_teacher_mean"],
        "fallback_rate": fallback_rate,
    }, indent=1))


if __name__ == "__main__":
    main()
