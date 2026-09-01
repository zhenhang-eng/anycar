#!/usr/bin/env python3
"""Low-budget multi-bank consensus study on the Phase 1a states.

The 128-prefix three-bank consensus costs about 384 evaluations per state.
Before Phase 2 adopts consensus labels, test whether 32/64-budget consensus
(with the guarded stay fallback from 11.45/11.46) retains the value: for each
budget the consensus is the elementwise mean of the three bank teachers, is
re-evaluated by a fresh DBM rollout, and falls back to the anchor whenever it
does not beat the anchor cost. Uses the Phase 1a manifest states and the
already-stored bank1 teachers; only bank2/bank3 prefixes and the consensus
re-rollouts are new evaluations.
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
    RING_RADII,
    load_states,
    ring_candidates,
    seed_bank_directions,
)

DEFAULT_RUN = Path("outputs/mppi_proposal/proximal_search_phase1a_20260817_v1")
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/proximal_consensus_budget_20260818_v1"
)
GUARD = 1e-6
BUDGETS = (32, 64, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
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
    labels = np.load(args.run / "labels.npz", allow_pickle=False)
    manifest = json.loads((args.run / "manifest.json").read_text())
    keys = [f"{row['episode']}#{row['snapshot']}" for row in manifest["states"]]
    if [str(value) for value in labels["episodes"]] != keys:
        raise AssertionError("labels/manifest order mismatch")

    loader_args = SimpleNamespace(
        replay_labels=DEFAULT_REPLAY_LABELS,
        gt_train=DEFAULT_GT_TRAIN,
        scenario_plan=DEFAULT_SCENARIO_PLAN,
        repeat=args.repeat,
    )
    by_key = {
        f"{state['episode']}#{state['snapshot']}": state
        for state in load_states(loader_args)
    }
    states = [by_key[key] for key in keys]
    params = TorchMPPIParams(**json.loads(states[0]["mppi_params"]))
    weights = TorchMPPICostWeights(**json.loads(states[0]["cost_weights"]))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(states[0]["dbm_params"]))
    )
    device = torch.device(args.device)
    low = np.asarray(params.action_min, np.float32)
    high = np.asarray(params.action_max, np.float32)
    from generate_dbm_j16_local_curvature_labels import hadamard_directions

    bases = {1: hadamard_directions(), 2: seed_bank_directions(2), 3: seed_bank_directions(3)}

    def evaluate(state: dict, candidates: np.ndarray, cache: dict) -> np.ndarray:
        clipped = np.clip(candidates, low, high).astype(np.float32)
        keys_local = [np.round(value, 6).tobytes() for value in clipped]
        pending: dict[bytes, int] = {}
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
                        np.repeat(state["initial_state_six"][None], count, axis=0),
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

    j_a0_all = labels["j_a0"].astype(np.float64)
    j_warm_all = labels["j_warm"].astype(np.float64)
    j16_all = labels["j16"].astype(np.float64)
    anchor = np.minimum(j_a0_all, j_warm_all)

    results = {f"consensus_{budget}": [] for budget in BUDGETS}
    new_evals = 0
    for position, state in enumerate(states):
        cache: dict[bytes, float] = {}
        anchor_costs = evaluate(
            state, np.stack([state["a0"], state["warm"]]), cache
        )
        new_evals += 2
        teachers = {budget: {} for budget in BUDGETS}
        for bank_id, basis in bases.items():
            for budget in BUDGETS:
                if bank_id == 1:
                    teachers[budget][bank_id] = labels[
                        f"c_prefix_{budget}__knots"
                    ][position]
                    continue
                ring_count = budget // 32
                bank = ring_candidates(
                    state["a0"], state["sigma"], RING_RADII[:ring_count], basis
                )
                costs = evaluate(state, bank, cache)
                new_evals += int(np.sum(np.isfinite(costs)))
                pool = np.concatenate(
                    [state["a0"][None], state["warm"][None], bank]
                )
                pool_costs = np.concatenate([anchor_costs, costs])
                teachers[budget][bank_id] = pool[int(np.argmin(pool_costs))]
        for budget in BUDGETS:
            consensus = np.clip(
                np.stack(list(teachers[budget].values())).mean(axis=0),
                low, high,
            ).astype(np.float32)
            cost = float(evaluate(state, consensus[None], cache)[0])
            new_evals += 1
            fallback = cost > min(float(anchor_costs[0]), float(anchor_costs[1])) + GUARD
            results[f"consensus_{budget}"].append({
                "episode": state["episode"],
                "snapshot": state["snapshot"],
                "cost": cost,
                "fallback": bool(fallback),
                "guarded_cost": (
                    min(float(anchor_costs[0]), float(anchor_costs[1]))
                    if fallback
                    else cost
                ),
            })
        print(f"[{position + 1:03d}/{len(states):03d}] done", flush=True)

    summary = {"per_budget": {}}
    for budget in BUDGETS:
        rows = results[f"consensus_{budget}"]
        raw = np.asarray([row["cost"] for row in rows], np.float64)
        guarded = np.asarray([row["guarded_cost"] for row in rows], np.float64)
        summary["per_budget"][str(budget)] = {
            "states": len(rows),
            "raw_cost_mean": float(np.mean(raw)),
            "guarded_cost_mean": float(np.mean(guarded)),
            "r_a0_guarded": float(np.sum(j_a0_all - guarded) / np.sum(j_a0_all - j16_all)),
            "fallback_fraction": float(np.mean([row["fallback"] for row in rows])),
            "worse_than_anchor_before_fallback": int(np.sum(raw > anchor + GUARD)),
            "worst_regression_after_fallback": float(np.max(guarded - anchor)),
        }
    summary.update({
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "CONSENSUS_BUDGET_STUDY_COMPLETED_ACTOR_FROZEN",
        "sources": {"phase1a_run": str(args.run), "repeat": args.repeat},
        "protocol": {
            "consensus": "elementwise mean of the three bank teachers, re-rolled out, guarded stay fallback at 1e-6",
            "bank_evaluations_per_state": "bank2/bank3 at 32/64/128 prefixes; bank1 teachers reused from Phase 1a labels",
            "new_evaluations_total": new_evals,
        },
    })
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(
        args.output / "consensus_labels.npz",
        episodes=np.asarray(keys),
        **{
            f"guarded_cost_{budget}": np.asarray(
                [row["guarded_cost"] for row in results[f"consensus_{budget}"]],
                np.float32,
            )
            for budget in BUDGETS
        },
    )
    print(json.dumps(summary["per_budget"], indent=1))


if __name__ == "__main__":
    main()
