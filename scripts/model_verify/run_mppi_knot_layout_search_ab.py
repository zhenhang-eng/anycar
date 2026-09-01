#!/usr/bin/env python3
"""Matched-budget DBM search in uniform and front-dense 8-knot layouts.

Unlike the earlier projection oracle, every layout is re-optimized in its own
parameter space.  The experiment stays on stratified train-only states and does
not load formal validation or test data.
"""

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
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from generate_dbm_j16_local_curvature_labels import hadamard_directions
from generate_dbm_proposal_teacher import sha256_file
from run_mppi_proximal_search_phase1a import (
    DEFAULT_GT_TRAIN,
    DEFAULT_REPLAY_LABELS,
    DEFAULT_SCENARIO_PLAN,
    elite_mean,
    load_states,
    ring_candidates,
    select_states,
)


HORIZON = 50
LAYOUTS = {
    "uniform": [0.0, 7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0],
    "front_dense_mild": [0.0, 3.0, 7.0, 12.0, 18.0, 26.0, 35.0, 49.0],
    "front_dense_strong": [0.0, 2.0, 5.0, 9.0, 14.0, 21.0, 31.0, 49.0],
}
ROUND_RADII = (0.25, 0.50)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/knot_layout_search_ab_20260825_v3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-labels", type=Path, default=DEFAULT_REPLAY_LABELS)
    parser.add_argument("--gt-train", type=Path, default=DEFAULT_GT_TRAIN)
    parser.add_argument("--scenario-plan", type=Path, default=DEFAULT_SCENARIO_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=0, choices=(0, 1))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def interpolation_matrix(times: np.ndarray) -> np.ndarray:
    """Return W such that full_sequence[t] = W[t] @ knots."""
    times = np.asarray(times, np.float64)
    if len(times) != 8 or times[0] != 0 or times[-1] != HORIZON - 1:
        raise ValueError("layout must contain 8 ordered knots including 0 and 49")
    if np.any(np.diff(times) <= 0):
        raise ValueError("layout times must be strictly increasing")
    matrix = np.zeros((HORIZON, 8), np.float32)
    for step in range(HORIZON):
        right = int(np.searchsorted(times, step, side="right"))
        if right == 0:
            matrix[step, 0] = 1.0
        elif right == len(times):
            matrix[step, -1] = 1.0
        else:
            left = right - 1
            alpha = float((step - times[left]) / (times[right] - times[left]))
            matrix[step, left] = 1.0 - alpha
            matrix[step, right] = alpha
    return matrix


def sample_layout_knots(sequence: np.ndarray, times: np.ndarray) -> np.ndarray:
    index = np.arange(HORIZON, dtype=np.float64)
    return np.stack(
        [np.interp(times, index, sequence[:, channel]) for channel in range(2)],
        axis=1,
    ).astype(np.float32)


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    loader = argparse.Namespace(
        replay_labels=args.replay_labels,
        gt_train=args.gt_train,
        scenario_plan=args.scenario_plan,
        repeat=args.repeat,
    )
    gt_summary = json.loads((args.gt_train / "summary.json").read_text())
    if gt_summary.get("split") != "train":
        raise AssertionError("knot-layout search must use the sealed train split")
    states = select_states(load_states(loader), args.states)
    params = TorchMPPIParams(**json.loads(states[0]["mppi_params"]))
    weights = TorchMPPICostWeights(**json.loads(states[0]["cost_weights"]))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(states[0]["dbm_params"]))
    )
    device = torch.device(args.device)
    directions = hadamard_directions()
    low = np.asarray(params.action_min, np.float32)
    high = np.asarray(params.action_max, np.float32)
    matrices = {
        name: interpolation_matrix(np.asarray(times, np.float64))
        for name, times in LAYOUTS.items()
    }

    rows: list[dict] = []
    actions_by_layout: dict[str, list[np.ndarray]] = {name: [] for name in LAYOUTS}
    for position, state in enumerate(states):
        initial = torch.as_tensor(
            state["initial_state_six"][None], dtype=torch.float32, device=device
        )
        current = torch.as_tensor(
            state["current_action"][None], dtype=torch.float32, device=device
        )
        reference = torch.as_tensor(
            state["reference"][None], dtype=torch.float32, device=device
        )

        with torch.no_grad():
            uniform_anchor_knots = np.stack((state["a0"], state["warm"]))
            uniform_anchor_sequences = interpolate_knots(
                torch.as_tensor(
                    uniform_anchor_knots[:, None], dtype=torch.float32, device=device
                ),
                HORIZON,
            )[:, 0].cpu().numpy()
            common_anchor_cost = batched_cost(
                backend, weights,
                torch.as_tensor(
                    uniform_anchor_sequences[None], dtype=torch.float32, device=device
                ),
                initial, current, reference,
            )[0].cpu().numpy()

        row = {
            "episode": state["episode"],
            "snapshot": state["snapshot"],
            "speed": state["speed"],
            "scenario": state["scenario"],
            "j16": state["j16_cost"],
            "common_a0_cost": float(common_anchor_cost[0]),
            "common_warm_cost": float(common_anchor_cost[1]),
            "layouts": {},
        }
        for name, times_list in LAYOUTS.items():
            times = np.asarray(times_list, np.float64)
            matrix = torch.as_tensor(matrices[name], device=device)
            anchors = np.stack([
                sample_layout_knots(sequence, times)
                for sequence in uniform_anchor_sequences
            ])
            cache: dict[bytes, float] = {}

            def evaluate(candidates: np.ndarray) -> np.ndarray:
                candidates = np.clip(candidates, low, high).astype(np.float32)
                keys = [np.round(item, 7).tobytes() for item in candidates]
                pending = {key: index for index, key in enumerate(keys) if key not in cache}
                if pending:
                    values = np.stack([candidates[index] for index in pending.values()])
                    parts = []
                    with torch.no_grad():
                        for start in range(0, len(values), args.batch_size):
                            knots = torch.as_tensor(
                                values[start:start + args.batch_size],
                                dtype=torch.float32, device=device,
                            )
                            sequences = torch.einsum("hk,nkc->nhc", matrix, knots)
                            parts.append(batched_cost(
                                backend, weights, sequences[None],
                                initial, current, reference,
                            )[0].cpu().numpy())
                    for key, cost in zip(pending, np.concatenate(parts)):
                        cache[key] = float(cost)
                return np.asarray([cache[key] for key in keys], np.float32)

            anchor_cost = evaluate(anchors)
            round1 = ring_candidates(
                anchors[0], state["sigma"], ROUND_RADII, directions
            )
            round1_cost = evaluate(round1)
            center1 = elite_mean(
                np.concatenate((anchors, round1)),
                np.concatenate((anchor_cost, round1_cost)),
            )
            round2 = ring_candidates(center1, state["sigma"], [0.50], directions)
            round2_cost = evaluate(round2)
            center2 = elite_mean(
                np.concatenate((anchors, round1, round2)),
                np.concatenate((anchor_cost, round1_cost, round2_cost)),
            )
            round3 = ring_candidates(center2, state["sigma"], [0.25], directions)
            round3_cost = evaluate(round3)
            all_centers = np.concatenate((anchors, round1, round2, round3))
            all_costs = np.concatenate((anchor_cost, round1_cost, round2_cost, round3_cost))
            best = int(np.argmin(all_costs))
            # evaluate() clips every candidate before rollout; serialize the
            # exact evaluated action rather than the pre-clip proposal.
            evaluated_best = np.clip(all_centers[best], low, high).astype(np.float32)
            actions_by_layout[name].append(evaluated_best)
            row["layouts"][name] = {
                "projected_a0_cost": float(anchor_cost[0]),
                "projected_warm_cost": float(anchor_cost[1]),
                "search_cost": float(all_costs[best]),
                "search_candidate_index": best,
                "new_candidate_budget": 128,
                "common_baseline_violation": bool(
                    all_costs[best] > min(common_anchor_cost) + 1e-4
                ),
            }
        rows.append(row)
        print(
            f"[{position + 1:03d}/{len(states):03d}] "
            f"{state['episode']}/{state['snapshot']}", flush=True,
        )

    common_base = np.asarray([
        min(row["common_a0_cost"], row["common_warm_cost"]) for row in rows
    ])
    j16 = np.asarray([row["j16"] for row in rows])
    layout_costs = {
        name: np.asarray([row["layouts"][name]["search_cost"] for row in rows])
        for name in LAYOUTS
    }
    uniform = layout_costs["uniform"]
    results = {}
    for name, costs in layout_costs.items():
        paired = costs - uniform
        recovery = float(
            np.sum(common_base - costs) / max(np.sum(common_base - j16), 1e-12)
        )
        results[name] = {
            "search_cost": distribution(costs),
            "gain_vs_common_anchor": distribution(common_base - costs),
            "headroom_recovery_vs_common_anchor_to_j16": recovery,
            "paired_cost_minus_uniform": distribution(paired),
            "wins_ties_losses_vs_uniform": [
                int(np.sum(paired < -1e-6)),
                int(np.sum(np.abs(paired) <= 1e-6)),
                int(np.sum(paired > 1e-6)),
            ],
            "common_baseline_violation_count": int(np.sum(costs > common_base + 1e-4)),
            "projected_anchor_excess": distribution(np.asarray([
                min(row["layouts"][name]["projected_a0_cost"], row["layouts"][name]["projected_warm_cost"])
                - common_base[index]
                for index, row in enumerate(rows)
            ])),
        }

    front_names = [name for name in LAYOUTS if name != "uniform"]
    best_front = min(front_names, key=lambda name: results[name]["search_cost"]["mean"])
    paired_best = layout_costs[best_front] - uniform
    qualification = (
        "FRONT_DENSE_REOPTIMIZATION_PASS_RUNTIME_INTEGRATION_JUSTIFIED"
        if float(np.mean(paired_best)) < 0
        and float(np.quantile(paired_best, 0.95)) <= 0
        and int(np.sum(layout_costs[best_front] > common_base + 1e-4)) == 0
        else "FRONT_DENSE_REOPTIMIZATION_FAIL_KEEP_UNIFORM_LAYOUT"
    )
    checks = {
        "train_only_gt": gt_summary.get("split") == "train",
        "state_count": len(states) == args.states,
        "uniform_interpolation_identity": bool(np.allclose(
            matrices["uniform"], interpolation_matrix(np.linspace(0, 49, 8)), atol=1e-6
        )),
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "manifest": {
            "replay_labels": str(args.replay_labels.resolve()),
            "gt_train": str(args.gt_train.resolve()),
            "gt_train_summary_sha256": sha256_file(args.gt_train / "summary.json"),
            "scenario_plan": str(args.scenario_plan.resolve()),
            "scenario_plan_sha256": sha256_file(args.scenario_plan),
            "states": args.states,
            "repeat": args.repeat,
            "layouts": LAYOUTS,
            "objective": "deterministic DBM J_direct",
            "search": "matched multi-round antithetic Hadamard, 128 candidates/layout/state",
        },
        "common_baseline": {
            "cost": distribution(common_base),
            "j16_cost": distribution(j16),
        },
        "results": results,
        "best_front_dense": best_front,
        "checks": checks,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "per_state.json").write_text(json.dumps(rows, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "labels.npz",
        episode=np.asarray([row["episode"] for row in rows]),
        snapshot=np.asarray([row["snapshot"] for row in rows]),
        speed=np.asarray([row["speed"] for row in rows], np.float32),
        scenario=np.asarray([row["scenario"] for row in rows]),
        common_anchor_cost=common_base.astype(np.float32),
        j16_cost=j16.astype(np.float32),
        **{
            f"{name}_knots": np.asarray(actions_by_layout[name], np.float32)
            for name in LAYOUTS
        },
        **{
            f"{name}_cost": layout_costs[name].astype(np.float32)
            for name in LAYOUTS
        },
    )
    print(json.dumps({
        "qualification": qualification,
        "best_front_dense": best_front,
        "results": results,
        "checks": checks,
    }, indent=2))


if __name__ == "__main__":
    main()
