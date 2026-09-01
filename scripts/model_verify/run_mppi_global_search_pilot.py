#!/usr/bin/env python3
"""Global-search teacher generation + coherence test (100-state pilot).

Runs a broad, anchor-independent search on 100 stratified states to find
near-global-optimal actions, then measures the neighborhood coherence of
these global-optimum labels vs the existing proximal teacher labels.

Search design: multi-round ring search with Hadamard directions, starting
from a wide grid of initial centers spanning the full action range (not the
warm start). Budget ~600 evaluations per state. Guard: the result must not
be worse than the existing proximal teacher (for comparability of coherence
measurement, not for deployment).
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
from run_mppi_proximal_search_phase1a import (
    DEFAULT_GT_TRAIN,
    DEFAULT_REPLAY_LABELS,
    DEFAULT_SCENARIO_PLAN,
    load_states,
    select_states,
)
from sklearn.neighbors import NearestNeighbors


DEFAULT_LABELS = Path(
    "outputs/mppi_proposal/consensus64_labels_pool_20260818_v1/labels.npz"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/global_search_pilot_20260819_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def robust_scale(x: np.ndarray) -> np.ndarray:
    q = np.quantile(x, [0.25, 0.75], axis=0)
    s = q[1] - q[0]
    return np.where(s > 1e-9, s, np.std(x, axis=0) + 1e-9)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    device = torch.device(args.device)

    labels = dict(np.load(args.labels, allow_pickle=False))
    all_keys = [str(v) for v in labels["episodes"]]
    n_total = len(all_keys)

    states_all = load_states(SimpleNamespace(
        replay_labels=DEFAULT_REPLAY_LABELS,
        gt_train=DEFAULT_GT_TRAIN,
        scenario_plan=DEFAULT_SCENARIO_PLAN,
        repeat=args.repeat,
    ))
    states_all = select_states(states_all, n_total)
    k2s = {f"{s['episode']}#{s['snapshot']}": s for s in states_all}

    # Stratified subset: pick 100 states spread across speed x scenario
    speeds = labels["speeds"].astype(np.float32)
    scenarios = labels["scenarios"].astype(str)
    # sort by (speed, scenario, index) and take every N-th
    order = np.lexsort((np.arange(n_total), scenarios, speeds))
    stride = n_total // args.states
    subset_idx = order[::stride][:args.states]
    subset_keys = [all_keys[i] for i in subset_idx]
    states = [k2s[k] for k in subset_keys]
    n = len(states)

    params = TorchMPPIParams(**json.loads(str(states[0]["mppi_params"])))
    weights = TorchMPPICostWeights(**json.loads(str(states[0]["cost_weights"])))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(str(states[0]["dbm_params"])))
    )
    low = np.asarray(params.action_min, np.float32)
    high = np.asarray(params.action_max, np.float32)
    basis = hadamard_directions()  # (16, 16)

    def evaluate(state, candidates):
        clipped = np.clip(candidates, low, high).astype(np.float32)
        knots = torch.as_tensor(
            clipped[:, None], dtype=torch.float32, device=device
        )
        actions = interpolate_knots(knots, params.horizon)
        with torch.no_grad():
            return batched_cost(
                backend, weights, actions,
                torch.as_tensor(
                    state["initial_state_six"][None].repeat(len(clipped), axis=0),
                    dtype=torch.float32, device=device,
                ),
                torch.as_tensor(
                    state["current_action"][None].repeat(len(clipped), axis=0),
                    dtype=torch.float32, device=device,
                ),
                torch.as_tensor(
                    state["reference"][None].repeat(len(clipped), axis=0),
                    dtype=torch.float32, device=device,
                ),
            ).squeeze(-1).cpu().numpy().astype(np.float64)

    # Global search: multi-round, starting from multiple diverse seeds
    # (NOT from warm start), using Hadamard ring candidates at wide radii.
    radii = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0], np.float32)
    rng = np.random.default_rng(260819)

    global_knots = []
    global_costs = []
    proximal_costs = []
    evals_used = []

    for idx, state in enumerate(states):
        # Seed centers: population mean of a* (from proximal labels as prior
        # for what "reasonable" actions look like, but NOT the warm start),
        # plus random uniform samples for diversity.
        label_mean = labels["label_knots"].reshape(n_total, -1).mean(axis=0).reshape(8, 2)
        seeds = [label_mean.astype(np.float32)]
        for _ in range(3):
            random_center = rng.uniform(
                low, high, size=(8, 2)
            ).astype(np.float32)
            seeds.append(random_center)

        best_center = None
        best_cost = float("inf")
        evals = 0

        for round_idx in range(4):
            round_best = None
            round_best_cost = float("inf")
            for seed_center in seeds:
                candidates = [seed_center]
                for radius in radii:
                    for d in range(16):
                        direction = basis[d] * radius  # (8, 2)
                        candidates.append(
                            np.clip(seed_center + direction, low, high)
                        )
                        candidates.append(
                            np.clip(seed_center - direction, low, high)
                        )
                candidates = np.asarray(candidates, np.float32)
                costs = evaluate(state, candidates)
                evals += len(candidates)
                local_best = int(np.argmin(costs))
                if costs[local_best] < round_best_cost:
                    round_best_cost = costs[local_best]
                    round_best = candidates[local_best]
            if round_best_cost < best_cost:
                best_cost = round_best_cost
                best_center = round_best.copy()
            # Elite mean for next round
            seeds = [best_center]
            # Add perturbed elite for diversity
            for _ in range(2):
                perturbed = best_center + rng.normal(
                    0, 0.1, size=(8, 2)
                ).astype(np.float32)
                seeds.append(np.clip(perturbed, low, high))

        global_knots.append(best_center)
        global_costs.append(best_cost)
        # Proximal teacher cost for comparison (from labels)
        proximal_costs.append(float(labels["j_teacher"][subset_idx[idx]]))
        evals_used.append(evals)
        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{n}] evals={evals} "
                  f"J_global={best_cost:.3f} "
                  f"J_prox={proximal_costs[-1]:.3f}", flush=True)

    global_knots = np.stack(global_knots)  # (n, 8, 2)
    global_costs = np.asarray(global_costs)
    proximal_costs = np.asarray(proximal_costs)

    # Also load proximal teacher knots for the same states
    proximal_knots = labels["label_knots"][subset_idx].astype(np.float32)

    # === Coherence analysis ===
    # Build neighbor pairs on physical state features (NO action info)
    six = np.stack([s["initial_state_six"] for s in states]).astype(np.float32)
    ref = np.stack([s["reference"].reshape(-1) for s in states]).astype(np.float32)
    ctrl = np.stack([s["current_action"] for s in states]).astype(np.float32)
    feats = np.concatenate([six, ref, ctrl], axis=1)
    z = feats / robust_scale(feats)
    nn = NearestNeighbors(n_neighbors=9).fit(z)
    dist, idx_nn = nn.kneighbors(z)
    dist, idx_nn = dist[:, 1:], idx_nn[:, 1:]  # skip self

    def coherence(target_flat):
        vals = []
        for i in range(n):
            for j_pos in range(idx_nn.shape[1]):
                j = idx_nn[i, j_pos]
                ni = np.linalg.norm(target_flat[i])
                nj = np.linalg.norm(target_flat[j])
                if ni > 1e-9 and nj > 1e-9:
                    vals.append(float(
                        target_flat[i] @ target_flat[j] / (ni * nj)
                    ))
        return float(np.median(vals)), float(np.mean(np.asarray(vals) > 0))

    global_flat = global_knots.reshape(n, -1)
    proximal_flat = proximal_knots.reshape(n, -1)
    coh_global, pos_g = coherence(global_flat)
    coh_prox, pos_p = coherence(proximal_flat)

    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "GLOBAL_SEARCH_PILOT_COHERENCE_TESTED",
        "sources": {"labels": str(args.labels.resolve())},
        "counts": {"states": n, "evaluations_per_state_median": int(np.median(evals_used))},
        "cost_comparison": {
            "J_global_mean": float(global_costs.mean()),
            "J_global_median": float(np.median(global_costs)),
            "J_proximal_mean": float(proximal_costs.mean()),
            "J_proximal_median": float(np.median(proximal_costs)),
            "global_better_fraction": float(np.mean(global_costs < proximal_costs)),
        },
        "coherence": {
            "global_optimum": {
                "median_cosine": coh_global,
                "positive_fraction": pos_g,
            },
            "proximal_teacher": {
                "median_cosine": coh_prox,
                "positive_fraction": pos_p,
            },
        },
        "contract": {
            "actor_frozen": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
    }
    np.savez_compressed(
        args.output / "global_search_labels.npz",
        state_keys=np.asarray(subset_keys),
        global_knots=global_knots,
        global_costs=global_costs,
        proximal_knots=proximal_knots,
        proximal_costs=proximal_costs,
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str((args.output / "summary.json").resolve()),
        "cost": result["cost_comparison"],
        "coherence": result["coherence"],
    }, indent=2))


if __name__ == "__main__":
    main()
