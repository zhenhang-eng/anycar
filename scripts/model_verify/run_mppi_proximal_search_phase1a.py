#!/usr/bin/env python3
"""Phase 1a proximal search budget screen (train-only, deterministic).

Screening question: how many unique DBM candidate evaluations per state does
a proximal teacher need, and which structure (FD + descent ladder vs
antithetic ring CEM) converts budget into direct-cost headroom most
efficiently? This run only measures teacher quality and label geometry; no
Actor is trained and no formal validation or test data is read.

Contract (review doc 11.39.2):
- Budget counts new unique DBM candidate evaluations per arm (shared-cache
  accounting: a candidate already evaluated for another arm is not counted
  again); FD probes and ladder points count; the two anchor evaluations
  (actor center a0 and warm center) are reported separately and excluded.
- Candidate sets always include a0 and warm, so J_teacher <= J_baseline by
  construction; violations are reported and must be zero.
- Objective is deterministic J_direct(s, a); no reward seeds are involved.
- Arm C single round uses one nested 256-candidate bank (16 Hadamard
  directions x 8 radii x antithetic pairs, inner rings first); prefixes give
  32/64/128/256. Multi-round runs only the 128 total budget (64 bank prefix
  + 32 + 32 re-centred rounds around elite means). Arm B evaluation blocks
  are FD@0.5sigma, coarse ladder, FD@0.25, ladder stride-2 extra, FD@0.125,
  ladder remainder, FD@1.0 (cumulative 32/64/96/128/160/224/256); the
  descent direction is always the FD@0.5 estimate so prefixes stay nested.
- Seed stability: two additional deterministic 128-prefix banks (fixed
  derangement + sign pattern) at the 128 budget only.
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
from car_dynamics.controllers_torch.mppi import (
    TorchMPPICostWeights,
    TorchMPPIParams,
)
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from generate_dbm_j16_local_curvature_labels import hadamard_directions
from generate_dbm_proposal_teacher import sha256_file

DEFAULT_REPLAY_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_direct_center_replay_diverse_20260806_v2"
)
DEFAULT_GT_TRAIN = Path("outputs/mppi_proposal/dbm_direct_gt_train_20260807_v2")
DEFAULT_SCENARIO_PLAN = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1/scenario_plan.json"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/proximal_search_phase1a_20260817_v1")
RING_RADII = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0)
FD_BLOCKS = (("fd", 0.5), ("ladder", 32), ("fd", 0.25), ("ladder", 32),
             ("fd", 0.125), ("ladder", 64), ("fd", 1.0))
LADDER_COUNT = 128
ARM_C_ROUND_RADII = {2: 0.5, 3: 0.25}
ELITE_COUNT = 16
REPORT_BUDGETS = (32, 64, 128, 256)
STEERING_DIMS = (1, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-labels", type=Path, default=DEFAULT_REPLAY_LABELS)
    parser.add_argument("--gt-train", type=Path, default=DEFAULT_GT_TRAIN)
    parser.add_argument("--scenario-plan", type=Path, default=DEFAULT_SCENARIO_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=0, choices=(0, 1))
    parser.add_argument("--eval-chunk", type=int, default=512)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_states(args: argparse.Namespace) -> list[dict]:
    splits = json.loads((args.replay_labels / "splits.json").read_text())
    train_episodes = sorted(splits["train"])
    plan = {
        row["episode_id"]: row
        for row in json.loads(args.scenario_plan.read_text())["episodes"]
    }
    gt_summary = json.loads((args.gt_train / "summary.json").read_text())
    if gt_summary["split"] != "train" or "test" not in gt_summary["test_policy"]:
        raise AssertionError("GT train summary must be sealed train split")
    gt_rows = {(row["episode"], row["snapshot"]): row for row in gt_summary["rows"]}

    states = []
    for episode in train_episodes:
        for path in sorted((args.replay_labels / episode).glob("step_*.npz")):
            row = gt_rows[(episode, path.name)]
            with np.load(path, allow_pickle=False) as label:
                source_path = Path(str(label["source_snapshot"]))
                sigma = np.asarray(label["sigma"], np.float32)
                a0 = np.asarray(label["bootstrap_actor_center"][args.repeat], np.float32)
                warm = np.asarray(label["anchor_center"][args.repeat], np.float32)
                anchor_checkpoint_sha256 = str(label["checkpoint_sha256"])
            with np.load(
                args.gt_train / episode / path.name, allow_pickle=False
            ) as gt, np.load(source_path, allow_pickle=False) as source:
                best = int(gt["knot_best_index"])
                astar = np.asarray(gt["optimized_knots"][best], np.float32)
                reference = np.asarray(source["reference"], np.float32)
                params = json.loads(str(source["mppi_params_json"]))
                if len(reference) == int(params["horizon"]) + 1:
                    reference = reference[1:]
                residual = (astar - a0) / sigma.reshape(1, 2)
                states.append({
                    "episode": episode,
                    "snapshot": path.name,
                    "scenario": plan[episode]["scenario_class"],
                    "speed": float(plan[episode]["reference_speed_mps"]),
                    "source_path": str(source_path),
                    "source_hash": sha256_file(source_path),
                    "anchor_checkpoint_sha256": anchor_checkpoint_sha256,
                    "sigma": sigma,
                    "a0": a0,
                    "warm": warm,
                    "astar": astar,
                    "j16_cost": float(row["j16_best_found"]),
                    "residual_norm": float(np.sqrt(np.mean(residual ** 2))),
                    "initial_state_six": np.asarray(source["initial_state_six"], np.float32),
                    "current_action": np.asarray(source["current_action"], np.float32),
                    "reference": reference,
                    "mppi_params": str(source["mppi_params_json"]),
                    "cost_weights": str(source["cost_weights_json"]),
                    "dbm_params": str(source["dbm_params_json"]),
                })
    return states


def select_states(states: list[dict], count: int) -> list[dict]:
    ordered = sorted(
        states,
        key=lambda item: (item["speed"], item["scenario"], item["residual_norm"]),
    )
    stride = len(ordered) / count
    return [ordered[int(index * stride)] for index in range(count)]


def ring_candidates(
    base: np.ndarray, sigma: np.ndarray, radii, directions: np.ndarray
) -> np.ndarray:
    out = []
    for radius in radii:
        for direction in directions:
            delta = radius * sigma.reshape(1, 2) * direction
            out.append(base + delta)
            out.append(base - delta)
    return np.asarray(out, np.float32)


def ladder_candidates(
    base: np.ndarray,
    sigma: np.ndarray,
    steps: np.ndarray,
    direction_knot: np.ndarray,
) -> np.ndarray:
    delta = steps.reshape(-1, 1, 1) * (
        sigma.reshape(1, 1, 2) * direction_knot.reshape(8, 2)
    )
    return (base[None] + delta).astype(np.float32)


def descent_direction(fd_costs: np.ndarray, radius: float) -> np.ndarray:
    slope = np.asarray(
        [
            (fd_costs[2 * i] - fd_costs[2 * i + 1]) / (2.0 * radius)
            for i in range(16)
        ],
        np.float64,
    )
    norm = np.linalg.norm(slope)
    if norm <= 1e-12:
        return np.zeros((8, 2), np.float32)
    descent = -slope / norm
    return (hadamard_directions().reshape(16, -1).T @ descent).reshape(8, 2).astype(
        np.float32
    )


def rotated_hadamard(pairs: list[tuple[int, int]], angles: list[float]) -> np.ndarray:
    """Deterministic orthogonal basis: Hadamard composed with fixed Givens
    rotations; rows are no longer +/-1 so the antithetic point set differs
    from the base bank while staying an orthonormal design."""
    matrix = hadamard_directions().reshape(16, -1).astype(np.float64)
    rotation = np.eye(16)
    for (i, j), angle in zip(pairs, angles):
        cosine, sine = np.cos(angle), np.sin(angle)
        givens = np.eye(16)
        givens[i, i] = cosine
        givens[j, j] = cosine
        givens[i, j] = -sine
        givens[j, i] = sine
        rotation = rotation @ givens
    rotated = matrix @ rotation
    if not np.allclose(rotated @ rotated.T, 16 * np.eye(16), atol=1e-5):
        raise AssertionError("rotated Hadamard lost orthogonality")
    if np.allclose(np.abs(rotated), 1.0):
        raise AssertionError("rotation did not change the direction set")
    return rotated.astype(np.float32).reshape(16, 8, 2)


def seed_bank_directions(bank_id: int) -> np.ndarray:
    if bank_id == 2:
        return rotated_hadamard(
            [(2 * k, 2 * k + 1) for k in range(8)],
            [0.3 + 0.05 * k for k in range(8)],
        )
    if bank_id == 3:
        return rotated_hadamard(
            [(2 * k + 1, 2 * k + 2) for k in range(7)] + [(0, 15)],
            [0.4 + 0.07 * k for k in range(8)],
        )
    raise ValueError(bank_id)


def elite_mean(centers: np.ndarray, costs: np.ndarray) -> np.ndarray:
    elite = np.argsort(costs)[:ELITE_COUNT]
    return centers[elite].mean(axis=0).astype(np.float32)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / max(norm, 1e-12)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    directions = hadamard_directions()
    states = select_states(load_states(args), args.states)
    for key in ("cost_weights", "dbm_params"):
        if len({state[key] for state in states}) != 1:
            raise ValueError(f"{key} differs across states")
    frozen_mppi = []
    for state in states:
        value = json.loads(state["mppi_params"])
        value.pop("seed", None)
        frozen_mppi.append(value)
    if any(value != frozen_mppi[0] for value in frozen_mppi[1:]):
        raise ValueError("non-seed MPPI parameters differ across states")

    params = TorchMPPIParams(**json.loads(states[0]["mppi_params"]))
    weights = TorchMPPICostWeights(**json.loads(states[0]["cost_weights"]))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(states[0]["dbm_params"]))
    )
    device = torch.device(args.device)
    low = np.asarray(params.action_min, np.float32)
    high = np.asarray(params.action_max, np.float32)
    ladder_full = np.sort(np.geomspace(0.05, 3.0, LADDER_COUNT)).astype(np.float32)
    ladder_levels = {
        1: ladder_full[::4],
        3: ladder_full[2::4],
        5: ladder_full[1::2],
    }

    arm_names = (
        [f"arm_c_prefix_{budget}" for budget in REPORT_BUDGETS]
        + ["arm_c_multi_128"]
        + [f"arm_b_{budget}" for budget in REPORT_BUDGETS]
        + ["arm_c_bank2_128", "arm_c_bank3_128"]
    )
    records: dict[str, list[dict]] = {name: [] for name in arm_names}
    anchor_rows, budget_rows, clip_rows = [], [], []

    args.output.mkdir(parents=True)
    for position, state in enumerate(states):
        cache: dict[bytes, float] = {}
        total_budget = 0
        clipped_total, evaluated_total = 0, 0

        def evaluate(candidates: np.ndarray) -> np.ndarray:
            nonlocal total_budget, clipped_total, evaluated_total
            clipped = np.clip(candidates, low, high).astype(np.float32)
            clipped_total += int(np.sum(np.any(clipped != candidates, axis=(1, 2))))
            evaluated_total += len(clipped)
            keys = [np.round(value, 6).tobytes() for value in clipped]
            pending_by_key: dict[bytes, int] = {}
            for row, key in enumerate(keys):
                if key not in cache:
                    pending_by_key.setdefault(key, row)
            if pending_by_key:
                pending_keys = list(pending_by_key)
                pending_array = np.asarray(
                    [clipped[row] for row in pending_by_key.values()], np.float32
                )
                flat = []
                with torch.no_grad():
                    for start in range(0, len(pending_array), args.eval_chunk):
                        chunk = torch.as_tensor(
                            pending_array[start:start + args.eval_chunk][None],
                            dtype=torch.float32, device=device,
                        )
                        actions = interpolate_knots(chunk, params.horizon)
                        flat.append(
                            batched_cost(
                                backend, weights, actions,
                                torch.as_tensor(state["initial_state_six"][None], dtype=torch.float32, device=device),
                                torch.as_tensor(state["current_action"][None], dtype=torch.float32, device=device),
                                torch.as_tensor(state["reference"][None], dtype=torch.float32, device=device),
                            )[0].cpu().numpy()
                        )
                total_budget += len(pending_keys)
                for key, cost in zip(pending_keys, np.concatenate(flat)):
                    cache[key] = float(cost)
            return np.asarray([cache[key] for key in keys], np.float32)

        def teacher(prefix_centers: list[np.ndarray], prefix_costs: list[np.ndarray]):
            centers = np.concatenate([state["a0"][None], state["warm"][None]] + prefix_centers)
            costs = np.concatenate([anchor_costs] + prefix_costs)
            best = int(np.argmin(costs))
            return centers[best], float(costs[best])

        anchor_costs = evaluate(np.stack([state["a0"], state["warm"]]))
        anchor_budget = total_budget
        anchor_rows.append({
            "episode": state["episode"], "snapshot": state["snapshot"],
            "j_a0": float(anchor_costs[0]), "j_warm": float(anchor_costs[1]),
            "j16": state["j16_cost"],
        })

        # Arm C: nested ring bank, then multi-round, then seed banks.
        arm_c_budget_start = total_budget
        bank1 = ring_candidates(state["a0"], state["sigma"], RING_RADII, directions)
        bank1_costs = evaluate(bank1)
        arm_c_budget = total_budget - arm_c_budget_start
        for budget in REPORT_BUDGETS:
            center, cost = teacher([bank1[:budget]], [bank1_costs[:budget]])
            records[f"arm_c_prefix_{budget}"].append({
                "episode": state["episode"], "snapshot": state["snapshot"],
                "knots": center, "cost": cost,
                "budget": min(budget, int(bank1_costs.shape[0])),
            })

        multi_budget_start = total_budget
        round1 = bank1[:64]
        round1_costs = bank1_costs[:64]
        m1 = elite_mean(
            np.concatenate([round1, state["a0"][None], state["warm"][None]]),
            np.concatenate([round1_costs, anchor_costs]),
        )
        round2 = ring_candidates(m1, state["sigma"], [ARM_C_ROUND_RADII[2]], directions)
        round2_costs = evaluate(round2)
        m2 = elite_mean(
            np.concatenate([round1, round2, state["a0"][None], state["warm"][None]]),
            np.concatenate([round1_costs, round2_costs, anchor_costs]),
        )
        round3 = ring_candidates(m2, state["sigma"], [ARM_C_ROUND_RADII[3]], directions)
        round3_costs = evaluate(round3)
        # Capture the multi-round budget immediately: 64 bank-prefix reuses
        # plus the rounds 2-3 evaluations; later arms must not contaminate it.
        arm_c_multi_budget = multi_round_budget(
            total_budget - multi_budget_start
        )
        center, cost = teacher(
            [round1, round2, round3],
            [round1_costs, round2_costs, round3_costs],
        )
        records["arm_c_multi_128"].append({
            "episode": state["episode"], "snapshot": state["snapshot"],
            "knots": center, "cost": cost, "budget": arm_c_multi_budget,
        })

        # Arm B: FD blocks and nested ladder along the fixed FD@0.5 direction.
        arm_b_budget_start = total_budget
        g_direction = None
        cumulative_centers, cumulative_costs = [], []
        block_index = 0
        for kind, value in FD_BLOCKS:
            if kind == "fd":
                block = ring_candidates(state["a0"], state["sigma"], [value], directions)
                costs = evaluate(block)
                if value == 0.5:
                    g_direction = descent_direction(costs, 0.5)
            else:
                steps = ladder_levels[block_index]
                block = ladder_candidates(state["a0"], state["sigma"], steps, g_direction)
                costs = evaluate(block)
            cumulative_centers.append(block)
            cumulative_costs.append(costs)
            block_index += 1
        arm_b_budget = total_budget - arm_b_budget_start
        sizes = [len(value) for value in cumulative_centers]
        offsets = np.cumsum([0] + sizes)
        stacked_centers = np.concatenate(cumulative_centers)
        stacked_costs = np.concatenate(cumulative_costs)
        for budget in REPORT_BUDGETS:
            take = int(min(budget, offsets[-1]))
            center, cost = teacher(
                [stacked_centers[:take]], [stacked_costs[:take]]
            )
            records[f"arm_b_{budget}"].append({
                "episode": state["episode"], "snapshot": state["snapshot"],
                "knots": center, "cost": cost, "budget": take,
            })

        # Seed-stability banks at 128: Hadamard composed with two fixed
        # Givens-rotation sets; row permutations/sign flips are invariant
        # under antithetic pairing and Sylvester Hadamard is closed under
        # column permutations, so neither provides a valid seed.
        for bank_id in (2, 3):
            bank_directions = seed_bank_directions(bank_id)
            bank = ring_candidates(
                state["a0"], state["sigma"], RING_RADII[:4], bank_directions
            )
            costs = evaluate(bank)
            center, cost = teacher([bank], [costs])
            records[f"arm_c_bank{bank_id}_128"].append({
                "episode": state["episode"], "snapshot": state["snapshot"],
                "knots": center, "cost": cost, "budget": int(costs.shape[0]),
            })

        budget_rows.append({
            "episode": state["episode"], "snapshot": state["snapshot"],
            "anchor_evaluations": anchor_budget,
            "arm_c_unique": arm_c_budget,
            "arm_c_multi_unique": arm_c_multi_budget,
            "arm_b_unique": arm_b_budget,
            "total_unique": total_budget,
        })
        clip_rows.append(clipped_total / max(evaluated_total, 1))
        print(
            f"[{position + 1:03d}/{len(states):03d}] {state['episode']}/"
            f"{state['snapshot']} budget={total_budget} "
            f"clip={clip_rows[-1]:.3f}",
            flush=True,
        )

    # Aggregate metrics.
    sigma_by_state = {
        (state["episode"], state["snapshot"]): state["sigma"] for state in states
    }
    a0_by_state = {
        (state["episode"], state["snapshot"]): state["a0"] for state in states
    }
    warm_by_state = {
        (state["episode"], state["snapshot"]): state["warm"] for state in states
    }
    astar_by_state = {
        (state["episode"], state["snapshot"]): state["astar"] for state in states
    }
    anchor_by_state = {
        (row["episode"], row["snapshot"]): row for row in anchor_rows
    }

    def aggregate(name: str) -> dict:
        costs = np.asarray([row["cost"] for row in records[name]], np.float64)
        j_a0 = np.asarray(
            [anchor_by_state[(row["episode"], row["snapshot"])]["j_a0"] for row in records[name]]
        )
        j_warm = np.asarray(
            [anchor_by_state[(row["episode"], row["snapshot"])]["j_warm"] for row in records[name]]
        )
        j16 = np.asarray(
            [anchor_by_state[(row["episode"], row["snapshot"])]["j16"] for row in records[name]]
        )
        deltas, cosines, steer_cosines, norms, anchor_picks = [], [], [], [], []
        for row in records[name]:
            sigma = sigma_by_state[(row["episode"], row["snapshot"])]
            a0 = a0_by_state[(row["episode"], row["snapshot"])]
            astar = astar_by_state[(row["episode"], row["snapshot"])]
            tiled = np.repeat(sigma, 8)
            delta = (row["knots"].reshape(-1) - a0.reshape(-1)) / tiled
            j16_delta = (astar.reshape(-1) - a0.reshape(-1)) / tiled
            deltas.append(delta)
            cosines.append(float(unit(delta) @ unit(j16_delta)))
            steer_cosines.append(
                float(unit(delta[list(STEERING_DIMS)]) @ unit(j16_delta[list(STEERING_DIMS)]))
            )
            norms.append(float(np.sqrt(np.mean(delta ** 2))))
            anchor_picks.append(
                bool(
                    np.allclose(row["knots"], a0)
                    or np.allclose(
                        row["knots"],
                        warm_by_state[(row["episode"], row["snapshot"])],
                    )
                )
            )
        cosines = np.asarray(cosines)
        steer_cosines = np.asarray(steer_cosines)
        return {
            "count": len(records[name]),
            "teacher_cost_mean": float(np.mean(costs)),
            "teacher_cost_median": float(np.median(costs)),
            "headroom_recovery_vs_warm": float(
                np.sum(j_warm - costs) / np.sum(j_warm - j16)
            ),
            "headroom_recovery_vs_a0": float(
                np.sum(j_a0 - costs) / np.sum(j_a0 - j16)
            ),
            "baseline_violations": int(np.sum(costs > np.minimum(j_a0, j_warm))),
            "teacher_is_anchor_fraction": float(np.mean(anchor_picks)),
            "residual_cosine_to_j16_median": float(np.median(cosines)),
            "residual_cosine_to_j16_p10": float(np.quantile(cosines, 0.10)),
            "steering_cosine_to_j16_median": float(np.median(steer_cosines)),
            "residual_norm_sigma_rms_median": float(np.median(norms)),
        }

    summary_arms = {name: aggregate(name) for name in arm_names}
    seed_stability = []
    for row1, row2 in zip(
        records["arm_c_prefix_128"], records["arm_c_bank2_128"]
    ):
        for row3 in records["arm_c_bank3_128"]:
            if row3["episode"] == row1["episode"] and row3["snapshot"] == row1["snapshot"]:
                sigma = sigma_by_state[(row1["episode"], row1["snapshot"])]
                tiled = np.repeat(sigma, 8)
                d1 = (row1["knots"].reshape(-1) - a0_by_state[(row1["episode"], row1["snapshot"])].reshape(-1)) / tiled
                d2 = (row2["knots"].reshape(-1) - a0_by_state[(row1["episode"], row1["snapshot"])].reshape(-1)) / tiled
                d3 = (row3["knots"].reshape(-1) - a0_by_state[(row1["episode"], row1["snapshot"])].reshape(-1)) / tiled
                seed_stability.append({
                    "episode": row1["episode"], "snapshot": row1["snapshot"],
                    "bank12_cosine": float(unit(d1) @ unit(d2)),
                    "bank13_cosine": float(unit(d1) @ unit(d3)),
                    "cost_gap_12": abs(row1["cost"] - row2["cost"]),
                    "cost_gap_13": abs(row1["cost"] - row3["cost"]),
                    "center_distance_sigma_12": float(np.sqrt(np.mean((d1 - d2) ** 2))),
                    "center_distance_sigma_13": float(np.sqrt(np.mean((d1 - d3) ** 2))),
                })
                break
    bank12 = np.asarray([row["bank12_cosine"] for row in seed_stability])
    bank13 = np.asarray([row["bank13_cosine"] for row in seed_stability])
    gap12 = np.asarray([row["cost_gap_12"] for row in seed_stability])
    gap13 = np.asarray([row["cost_gap_13"] for row in seed_stability])
    prefix_costs = {
        (row["episode"], row["snapshot"]): row["cost"]
        for row in records["arm_c_prefix_128"]
    }
    anchor_costs_by_state = {
        (row["episode"], row["snapshot"]): min(row["j_a0"], row["j_warm"])
        for row in anchor_rows
    }
    moved = np.asarray([
        (
            prefix_costs[(row["episode"], row["snapshot"])]
            < anchor_costs_by_state[(row["episode"], row["snapshot"])] - 1e-6
        )
        for row in seed_stability
    ])

    j_a0_all = np.asarray([row["j_a0"] for row in anchor_rows], np.float64)
    j_warm_all = np.asarray([row["j_warm"] for row in anchor_rows], np.float64)
    j16_all = np.asarray([row["j16"] for row in anchor_rows], np.float64)
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "PHASE1A_SEARCH_BUDGET_SCREEN_COMPLETED_ACTOR_FROZEN",
        "sources": {
            "replay_labels": str(args.replay_labels),
            "gt_train": str(args.gt_train),
            "gt_train_summary_sha256": sha256_file(args.gt_train / "summary.json"),
            "scenario_plan": str(args.scenario_plan),
            "scenario_plan_sha256": sha256_file(args.scenario_plan),
        },
        "protocol": {
            "states": len(states),
            "repeat": args.repeat,
            "objective": "deterministic J_direct",
            "budget_accounting": (
                "new unique evaluations under a shared per-state cache; "
                "anchor evaluations excluded and reported separately"
            ),
            "ring_radii_sigma": list(RING_RADII),
            "fd_block_order": [list(item) for item in FD_BLOCKS],
            "ladder_range_sigma": [0.05, 3.0],
            "elite_count": ELITE_COUNT,
            "arm_c_round_radii_sigma": ARM_C_ROUND_RADII,
            "seed_banks": "Hadamard x fixed Givens rotations (two variants), 128 prefix",
        },
        "baselines": {
            "j_a0_mean": float(np.mean(j_a0_all)),
            "j_warm_mean": float(np.mean(j_warm_all)),
            "j16_mean": float(np.mean(j16_all)),
        },
        "budget": {
            "mean_total_unique_per_state": float(np.mean(
                [row["total_unique"] for row in budget_rows]
            )),
            "mean_clip_fraction": float(np.mean(clip_rows)),
        },
        "arms": summary_arms,
        "seed_stability_128": {
            "count": len(seed_stability),
            "moved_count": int(np.sum(moved)),
            "bank12_cosine_median": float(np.median(bank12)),
            "bank13_cosine_median": float(np.median(bank13)),
            "cost_gap_median": float(np.median(np.concatenate([gap12, gap13]))),
            "bank12_cosine_median_moved_only": (
                float(np.median(bank12[moved])) if np.any(moved) else None
            ),
            "bank13_cosine_median_moved_only": (
                float(np.median(bank13[moved])) if np.any(moved) else None
            ),
            "cost_gap_median_moved_only": (
                float(np.median(np.concatenate([gap12[moved], gap13[moved]])))
                if np.any(moved)
                else None
            ),
            "note": (
                "anchor-picking states agree trivially; moved-only medians "
                "are the meaningful stability readout"
            ),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(
        args.output / "labels.npz",
        episodes=np.asarray([f"{row['episode']}#{row['snapshot']}" for row in anchor_rows]),
        speeds=np.asarray([state["speed"] for state in states], np.float32),
        scenarios=np.asarray([state["scenario"] for state in states]),
        j_a0=j_a0_all.astype(np.float32),
        j_warm=j_warm_all.astype(np.float32),
        j16=j16_all.astype(np.float32),
        **{
            f"{name.replace('arm_', '')}__knots": np.stack(
                [row["knots"] for row in records[name]]
            )
            for name in arm_names
        },
        **{
            f"{name.replace('arm_', '')}__cost": np.asarray(
                [row["cost"] for row in records[name]], np.float32
            )
            for name in arm_names
        },
    )
    (args.output / "manifest.json").write_text(json.dumps({
        "states": [
            {
                "episode": state["episode"], "snapshot": state["snapshot"],
                "scenario": state["scenario"], "speed": state["speed"],
                "source": state["source_path"], "source_sha256": state["source_hash"],
                "anchor_checkpoint_sha256": state["anchor_checkpoint_sha256"],
                "residual_norm": state["residual_norm"],
            }
            for state in states
        ],
        "budget_rows": budget_rows,
    }, indent=1))
    print(json.dumps({
        "states": len(states),
        "mean_budget_per_state": summary["budget"]["mean_total_unique_per_state"],
        "best_recovery_vs_warm": {
            name: round(value["headroom_recovery_vs_warm"], 3)
            for name, value in summary_arms.items()
        },
        "violations": sum(value["baseline_violations"] for value in summary_arms.values()),
        "seed_stability_bank12_cosine": summary["seed_stability_128"]["bank12_cosine_median"],
    }, indent=1))


def multi_round_budget(unique_new: int) -> int:
    return 64 + unique_new


if __name__ == "__main__":
    main()
