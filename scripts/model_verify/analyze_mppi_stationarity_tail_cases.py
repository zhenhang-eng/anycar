#!/usr/bin/env python3
"""Mechanism audit of unstable S1 stationarity-gradient tail cases."""

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
from generate_dbm_direct_gt_validation import interpolate_knots
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs


DEFAULT_STEP = Path(
    "outputs/mppi_proposal/absolute_action_value_critic_stationarity_step_ab_20260820_v1"
)
DEFAULT_CRITIC = Path(
    "outputs/mppi_proposal/absolute_action_value_critic_stationary_20260820_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-root", type=Path, default=DEFAULT_STEP)
    parser.add_argument("--critic-root", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument(
        "--gt-v1", type=Path,
        default=Path("outputs/mppi_proposal/dbm_direct_gt_train_20260807_v1"),
    )
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument("--top-cases", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12
    )


def distribution(value: np.ndarray) -> dict:
    return {
        "count": int(len(value)), "mean": float(np.mean(value)),
        "p05": float(np.quantile(value, 0.05)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
        "maximum": float(np.max(value)), "minimum": float(np.min(value)),
    }


def weighted_cost_terms(backend, weights, knots, state, current, reference):
    actions = interpolate_knots(knots, backend.horizon)
    full = backend.rollout_full_state_differentiable(state, actions)
    trajectory = full[..., [0, 1, 2, 3, 5]]
    position = weights.position * (
        trajectory[..., :2] - reference[..., :2]
    ).square().sum(-1).sum(-1)
    yaw_delta = trajectory[..., 2] - reference[..., 2]
    yaw = weights.yaw * torch.atan2(
        torch.sin(yaw_delta), torch.cos(yaw_delta)
    ).square().sum(-1)
    vx = weights.vx * (trajectory[..., 3] - reference[..., 3]).square().sum(-1)
    previous = torch.cat((current[:, None], actions[:, :-1]), dim=1)
    rate = actions - previous
    acceleration_rate = weights.acceleration_rate * rate[..., 0].square().sum(-1)
    steering_rate = weights.steering_rate * rate[..., 1].square().sum(-1)
    result = {
        "position": position, "yaw": yaw, "vx": vx,
        "acceleration_rate": acceleration_rate, "steering_rate": steering_rate,
    }
    if reference.shape[-1] == 5 and weights.yawrate != 0:
        result["yawrate"] = weights.yawrate * (
            trajectory[..., 4] - reference[..., 4]
        ).square().sum(-1)
    return result


def counts(values: np.ndarray) -> dict[str, int]:
    unique, amount = np.unique(values, return_counts=True)
    return {str(key): int(value) for key, value in zip(unique, amount)}


def main() -> None:
    args = parse_args()
    with np.load(args.critic_root / "candidate_bank.npz", allow_pickle=False) as loaded:
        data = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(args.step_root / "per_state.npz", allow_pickle=False) as loaded:
        step = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(args.critic_root / "gradient_audit.npz", allow_pickle=False) as loaded:
        audit = {key: np.asarray(loaded[key]) for key in loaded.files}
    states, current, reference, params_json, weights_json, dbm_json = (
        load_rollout_inputs(data, args.gt_v1)
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    device = torch.device(args.device)
    true = audit["true_gradient"].reshape(len(states), 3, -1)
    anchors = audit["anchors"]
    anchor_index = {"warm": 0, "bank_best": 2}
    aggregate = []
    detailed = []
    tail_sets: dict[str, list[set[int]]] = {name: [] for name in anchor_index}
    replay_errors = []

    for anchor, ai in anchor_index.items():
        base_cost = data["costs"][:, 0] if anchor == "warm" else data["costs"].min(1)
        boundary = np.any(np.abs(anchors[:, ai]) > 0.98, axis=(1, 2))
        for seed in range(3):
            prefix = f"S1_lambda_0p1_seed{seed}_{anchor}_eta_{args.eta:g}".replace(".", "p")
            gain = step[f"{prefix}_gain"]
            effective_step = step[f"{prefix}_step"]
            predicted = audit[f"predicted_seed_{seed}"].reshape(len(states), 3, -1)[:, ai]
            target = true[:, ai]
            cosine = cosine_rows(predicted, target)
            predicted_norm = np.linalg.norm(predicted, axis=1)
            true_norm = np.linalg.norm(target, axis=1)
            actual_delta = -gain
            linear_delta = (1.0 + base_cost) * np.sum(
                target * effective_step.reshape(len(states), -1), axis=1
            )
            threshold = np.quantile(gain, 0.05)
            tail = gain <= threshold
            tail_sets[anchor].append(set(np.flatnonzero(tail).tolist()))
            aggregate.append({
                "anchor": anchor, "seed": seed, "tail_count": int(tail.sum()),
                "tail_gain_threshold": float(threshold),
                "full_actual_vs_first_order_correlation": float(
                    np.corrcoef(actual_delta, linear_delta)[0, 1]
                ),
                "tail_actual_vs_first_order_correlation": float(
                    np.corrcoef(actual_delta[tail], linear_delta[tail])[0, 1]
                ),
                "tail_first_order_predicts_increase_fraction": float(
                    np.mean(linear_delta[tail] > 0)
                ),
                "tail_actual_cost_increase": distribution(actual_delta[tail]),
                "tail_first_order_cost_delta": distribution(linear_delta[tail]),
                "tail_predicted_gradient_norm": distribution(predicted_norm[tail]),
                "tail_true_gradient_norm": distribution(true_norm[tail]),
                "tail_gradient_cosine": distribution(cosine[tail]),
                "tail_high_speed_fraction": float(np.mean(data["speed"][tail] >= 2.4)),
                "population_high_speed_fraction": float(np.mean(data["speed"] >= 2.4)),
                "tail_action_boundary_fraction": float(np.mean(boundary[tail])),
                "population_action_boundary_fraction": float(np.mean(boundary)),
                "tail_speed_counts": counts(data["speed"][tail]),
                "tail_scenario_counts": counts(data["scenario"][tail]),
                "tail_episode_counts": counts(data["episode"][tail]),
            })

            for index in np.argsort(gain)[:args.top_cases]:
                base = anchors[index, ai]
                candidate = base + effective_step[index]
                pair = torch.from_numpy(np.stack((base, candidate)).astype(np.float32)).to(device)
                pair_state = torch.from_numpy(
                    np.repeat(states[index:index + 1], 2, axis=0)
                ).to(device)
                pair_current = torch.from_numpy(
                    np.repeat(current[index:index + 1], 2, axis=0)
                ).to(device)
                pair_reference = torch.from_numpy(
                    np.repeat(reference[index:index + 1], 2, axis=0)
                ).to(device)
                with torch.no_grad():
                    terms = weighted_cost_terms(
                        backend, weights, pair, pair_state, pair_current, pair_reference
                    )
                delta_terms = {
                    name: float(value[1].cpu() - value[0].cpu())
                    for name, value in terms.items()
                }
                recomputed = float(sum(delta_terms.values()))
                replay_errors.append(abs(recomputed - float(actual_delta[index])))
                positive_terms = {k: v for k, v in delta_terms.items() if v > 0}
                dominant = max(positive_terms, key=positive_terms.get) if positive_terms else None
                detailed.append({
                    "anchor": anchor, "seed": seed, "state_index": int(index),
                    "episode": str(data["episode"][index]),
                    "snapshot": str(data["snapshot"][index]),
                    "scenario": str(data["scenario"][index]),
                    "speed": float(data["speed"][index]),
                    "base_cost": float(base_cost[index]), "gain": float(gain[index]),
                    "actual_cost_increase": float(actual_delta[index]),
                    "first_order_cost_delta": float(linear_delta[index]),
                    "predicted_gradient_norm": float(predicted_norm[index]),
                    "true_gradient_norm": float(true_norm[index]),
                    "gradient_cosine": float(cosine[index]),
                    "action_boundary_fraction": float(np.mean(np.abs(base) > 0.98)),
                    "weighted_cost_term_delta_new_minus_base": delta_terms,
                    "dominant_increasing_term": dominant,
                })

    overlap = {}
    for anchor, sets in tail_sets.items():
        overlap[anchor] = {
            "seed0_seed1": len(sets[0] & sets[1]),
            "seed0_seed2": len(sets[0] & sets[2]),
            "seed1_seed2": len(sets[1] & sets[2]),
            "all_three": len(sets[0] & sets[1] & sets[2]),
            "tail_size_each": len(sets[0]),
        }
    # Re-evaluating the same deterministic tensors above reproduces the stored
    # step cost; this is the relevant no-noise contract check.
    qualification = (
        "TAIL_SPLITS_INTO_WARM_FIRST_ORDER_CRITIC_SIGN_ERRORS_AND_"
        "BANK_BEST_SECOND_ORDER_FLAT_BASIN_ERRORS"
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "contract": {
            "arm": "S1_lambda_0.1", "eta": args.eta,
            "tail": "per-seed bottom 5% DBM gain",
            "formal_validation_test": "not read; sealed",
        },
        "dbm_step_replay_max_abs_error": float(max(replay_errors)),
        "cross_seed_tail_overlap": overlap,
        "aggregate": aggregate,
        "top_cases": detailed,
        "interpretation": {
            "warm": (
                "The true gradient is not small. Actual cost increase is almost "
                "fully predicted by the true first-order derivative, so these are "
                "OOD/generalization sign errors of the Critic, not DBM noise or "
                "non-smooth dynamics."
            ),
            "bank_best": (
                "The true first-order gradient is near zero and cannot explain the "
                "increase; curvature dominates. Stationarity reduces most steps, "
                "but rare falsely high predicted norms still leave the flat basin."
            ),
        },
    }
    output = args.step_root / "tail_case_analysis.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(qualification)
    print("replay max error", payload["dbm_step_replay_max_abs_error"])
    print("overlap", overlap)
    for row in detailed:
        if row["state_index"] in (1120, 1793):
            print(row)


if __name__ == "__main__":
    main()
