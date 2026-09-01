#!/usr/bin/env python3
"""Audit equal-budget action exploration banks around frozen OAC Actors.

The audit is train-side mechanism evidence only.  It freezes the three latest
deterministic-center DBM Actors and evaluates six centers per internal-selection
state under the deterministic DBM J50 objective:

* current: registered Gaussian/tanh bank;
* orthogonal: state-cycled dense temporal-DCT orthogonal directions;
* guided: current bank's shared center and two antithetic pairs, replacing only
  the final blind wide sample with a cost-response fitted descent direction.

No Actor/Critic update, MPPI wrapper, formal validation, test, or closed loop is
performed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from evaluate_mppi_oac_warm_relative_centers import checkpoint_actor
from generate_dbm_proposal_teacher import sha256_file
from run_mppi_absolute_action_value_critic_cv import make_folds
from train_mppi_oac2_continuous_actor import (
    actor_exploration_bank,
    differentiable_dbm_cost,
    internal_split,
)
from train_mppi_online_absolute_sac import (
    BASE_SIGMA,
    actor_mean,
    load_actor_normalization,
    load_bank,
    make_actor_inputs,
    rollout_bank,
)


DEFAULT_RUN = Path(
    "outputs/mppi_proposal/"
    "online_absolute_sac_oac2_deterministic_center_dbm_k16_90round_20260828_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/oac2_actor_neighborhood_bank_audit_20260828_v1"
)
METHODS = ("current", "orthogonal", "guided_1x", "guided_2x")
ROLES = (
    "actor_mean", "pair0_minus", "pair0_plus", "pair1_minus", "pair1_plus", "sixth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--guided-ridge", type=float, default=1e-4)
    parser.add_argument(
        "--max-states", type=int, default=0,
        help="Optional contract smoke limit; zero evaluates all selection states.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def dct_mixed_directions() -> np.ndarray:
    """Return a dense, full-rank 16-D temporal/channel orthonormal basis.

    Rows are scaled to RMS one so their standardized perturbation magnitude
    matches a standard-normal exploration direction in expectation.
    """
    length = 8
    time = np.arange(length, dtype=np.float64) + 0.5
    dct = np.empty((length, length), np.float64)
    for mode in range(length):
        scale = math.sqrt(1.0 / length) if mode == 0 else math.sqrt(2.0 / length)
        dct[mode] = scale * np.cos(math.pi * mode * time / length)
    basis = np.zeros((16, 8, 2), np.float64)
    for mode in range(length):
        basis[mode, :, 0] = dct[mode] / math.sqrt(2.0)
        basis[mode, :, 1] = dct[mode] / math.sqrt(2.0)
        basis[length + mode, :, 0] = dct[mode] / math.sqrt(2.0)
        basis[length + mode, :, 1] = -dct[mode] / math.sqrt(2.0)
    basis *= math.sqrt(16.0)
    flat = basis.reshape(16, 16)
    gram = flat @ flat.T / 16.0
    if float(np.max(np.abs(gram - np.eye(16)))) > 1e-6:
        raise AssertionError("temporal DCT exploration basis is not RMS-orthonormal")
    return basis.astype(np.float32)


def current_bank_with_latents(
    mean: np.ndarray,
    log_scale: torch.Tensor,
    rng: np.random.Generator,
    minimum: float,
    maximum: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = np.clip(
        np.exp(log_scale.detach().cpu().numpy()), minimum, maximum
    ).astype(np.float32)
    std = BASE_SIGMA.reshape(1, 1, 2) * scale.reshape(1, 8, 2)
    latent_mean = np.arctanh(np.clip(mean, -0.999, 0.999))
    eps0 = rng.normal(size=mean.shape).astype(np.float32)
    eps1 = rng.normal(size=mean.shape).astype(np.float32)
    wide = rng.normal(size=mean.shape).astype(np.float32)
    wide_std = np.minimum(
        2.0 * std, BASE_SIGMA.reshape(1, 1, 2) * maximum
    )
    bank = np.stack((
        mean,
        np.tanh(latent_mean - std * eps0),
        np.tanh(latent_mean + std * eps0),
        np.tanh(latent_mean - std * eps1),
        np.tanh(latent_mean + std * eps1),
        np.tanh(latent_mean + wide_std * wide),
    ), axis=1).astype(np.float32)
    return bank, std.astype(np.float32), np.stack((eps0, eps1, wide), axis=1)


def orthogonal_bank(
    mean: np.ndarray,
    std: np.ndarray,
    maximum: float,
    seed: int,
    state_index: np.ndarray,
) -> np.ndarray:
    basis = dct_mixed_directions()
    local = np.arange(len(mean), dtype=np.int64)
    offset = (state_index.astype(np.int64) * 5 + local * 3 + seed * 7) % 16
    ids = np.stack((offset, (offset + 5) % 16, (offset + 11) % 16), axis=1)
    directions = basis[ids]
    latent = np.arctanh(np.clip(mean, -0.999, 0.999))
    wide_std = np.minimum(
        2.0 * std,
        BASE_SIGMA.reshape(1, 1, 2) * maximum,
    )
    return np.stack((
        mean,
        np.tanh(latent - std * directions[:, 0]),
        np.tanh(latent + std * directions[:, 0]),
        np.tanh(latent - std * directions[:, 1]),
        np.tanh(latent + std * directions[:, 1]),
        np.tanh(latent + wide_std * directions[:, 2]),
    ), axis=1).astype(np.float32)


def guided_bank(
    current: np.ndarray,
    current_cost: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: np.ndarray,
    maximum: float,
    ridge: float,
    radius_multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    result = current.copy()
    # The first two antithetic pairs give two central directional derivatives
    # in standardized latent coordinates.  Solve the minimum-norm projection
    # and replace only the blind sixth sample with a bounded descent proposal.
    slopes = np.stack((
        0.5 * (current_cost[:, 2] - current_cost[:, 1]),
        0.5 * (current_cost[:, 4] - current_cost[:, 3]),
    ), axis=1).astype(np.float64)
    directions = eps[:, :2].reshape(len(mean), 2, 16).astype(np.float64)
    gram = directions @ np.swapaxes(directions, 1, 2)
    gram += float(ridge) * np.eye(2, dtype=np.float64)[None]
    coefficients = np.linalg.solve(gram, slopes[..., None])[..., 0]
    projected_gradient = np.einsum("bi,bij->bj", coefficients, directions)
    descent = -projected_gradient
    rms = np.sqrt(np.mean(descent * descent, axis=1, keepdims=True))
    fallback = eps[:, 2].reshape(len(mean), 16).astype(np.float64)
    use_fallback = rms[:, 0] < 1e-8
    descent[use_fallback] = fallback[use_fallback]
    rms = np.sqrt(np.mean(descent * descent, axis=1, keepdims=True))
    descent = (descent / np.maximum(rms, 1e-12)).reshape(len(mean), 8, 2)
    latent = np.arctanh(np.clip(mean, -0.999, 0.999))
    guided_std = np.minimum(
        float(radius_multiplier) * std,
        BASE_SIGMA.reshape(1, 1, 2) * maximum,
    )
    result[:, 5] = np.tanh(latent + guided_std * descent).astype(np.float32)
    return result, descent.astype(np.float32)


def standardized_latent_delta(
    bank: np.ndarray, mean: np.ndarray, std: np.ndarray,
) -> np.ndarray:
    latent = np.arctanh(np.clip(bank, -0.999999, 0.999999))
    base = np.arctanh(np.clip(mean, -0.999999, 0.999999))
    return ((latent - base[:, None]) / std[:, None]).reshape(len(mean), 6, 16)


def bank_geometry_and_gradient(
    bank: np.ndarray,
    cost: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    exact_latent_gradient: np.ndarray,
) -> dict[str, np.ndarray]:
    delta = standardized_latent_delta(bank, mean, std)[:, 1:]
    target = (cost[:, 1:] - cost[:, :1]).astype(np.float64)
    fitted = np.empty_like(exact_latent_gradient, dtype=np.float64)
    ranks = np.empty(len(mean), np.int16)
    conditions = np.empty(len(mean), np.float64)
    projection_fraction = np.empty(len(mean), np.float64)
    for row in range(len(mean)):
        matrix = delta[row].astype(np.float64)
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.sum(singular > max(float(singular[0]), 1e-12) * 1e-6))
        ranks[row] = rank
        conditions[row] = (
            float(singular[0] / singular[rank - 1]) if rank else float("inf")
        )
        fitted[row] = np.linalg.lstsq(matrix, target[row], rcond=1e-6)[0]
        _, _, right = np.linalg.svd(matrix, full_matrices=False)
        span = right[:rank]
        gradient = exact_latent_gradient[row]
        projection = span.T @ (span @ gradient) if rank else np.zeros_like(gradient)
        projection_fraction[row] = float(
            np.linalg.norm(projection) / max(np.linalg.norm(gradient), 1e-12)
        )
    best = np.argmin(cost, axis=1)
    best_delta = standardized_latent_delta(bank, mean, std)[
        np.arange(len(mean)), best
    ]
    return {
        "rank": ranks,
        "condition": conditions,
        "projection_fraction": projection_fraction.astype(np.float32),
        "fitted_gradient_cosine": cosine(
            fitted, exact_latent_gradient.astype(np.float64)
        ).astype(np.float32),
        "best_step_descent_cosine": cosine(
            best_delta.astype(np.float64),
            -exact_latent_gradient.astype(np.float64),
        ).astype(np.float32),
    }


def exact_actor_gradient(
    mean: np.ndarray,
    std: np.ndarray,
    indices: np.ndarray,
    states: np.ndarray,
    current: np.ndarray,
    references: np.ndarray,
    backend: TorchDynamicBicycleRolloutBackend,
    weights: TorchMPPICostWeights,
    params: TorchMPPIParams,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    action = torch.tensor(mean, dtype=torch.float32, device=device, requires_grad=True)
    cost = differentiable_dbm_cost(
        action, indices, states, current, references,
        backend, weights, params, device,
    )
    gradient = torch.autograd.grad(cost.sum(), action)[0]
    latent_gradient = (
        gradient * (1.0 - action.square()) * torch.from_numpy(std).to(device)
    )
    return (
        cost.detach().cpu().numpy().astype(np.float32),
        latent_gradient.detach().cpu().numpy().reshape(len(mean), 16).astype(np.float32),
    )


def bootstrap_paired_mean(
    values: np.ndarray,
    episodes: np.ndarray,
    samples: int,
    seed: int,
) -> list[float]:
    unique = np.unique(episodes)
    groups = [values[episodes == episode] for episode in unique]
    rng = np.random.default_rng(seed)
    result = np.empty(samples, np.float64)
    for index in range(samples):
        choice = rng.integers(len(groups), size=len(groups))
        result[index] = np.mean(np.concatenate([groups[value] for value in choice]))
    return [float(np.quantile(result, 0.025)), float(np.quantile(result, 0.975))]


def method_summary(
    cost: np.ndarray,
    warm: np.ndarray,
    speed: np.ndarray,
    geometry: dict[str, np.ndarray],
) -> dict[str, Any]:
    base = cost[:, 0]
    best = np.min(cost, axis=1)
    gain = base - best
    actor_loses = base > warm + 1e-6
    result: dict[str, Any] = {
        "count": int(len(base)),
        "evaluation_budget_per_state": int(cost.shape[1]),
        "base_cost": distribution(base),
        "best_cost": distribution(best),
        "best_gain_vs_actor": distribution(gain),
        "strict_improvement_fraction": float(np.mean(gain > 1e-6)),
        "material_improvement_0_1_fraction": float(np.mean(gain > 0.1)),
        "material_improvement_1_0_fraction": float(np.mean(gain > 1.0)),
        "best_is_center_fraction": float(np.mean(np.argmin(cost, axis=1) == 0)),
        "actor_loses_warm_count": int(np.sum(actor_loses)),
        "actor_loses_warm_recovered_fraction": float(
            np.mean(best[actor_loses] <= warm[actor_loses] + 1e-6)
        ) if np.any(actor_loses) else 1.0,
        "actor_loses_warm_remaining_regret": distribution(
            best[actor_loses] - warm[actor_loses]
        ) if np.any(actor_loses) else None,
        "geometry": {
            "numeric_rank": distribution(geometry["rank"]),
            "condition": distribution(geometry["condition"]),
            "true_gradient_projection_fraction": distribution(
                geometry["projection_fraction"]
            ),
            "local_fit_gradient_cosine": distribution(
                geometry["fitted_gradient_cosine"]
            ),
            "best_step_descent_cosine": distribution(
                geometry["best_step_descent_cosine"]
            ),
        },
        "by_speed": {},
    }
    for value in sorted(np.unique(speed)):
        mask = np.isclose(speed, value)
        result["by_speed"][f"{value:.1f}"] = {
            "count": int(np.sum(mask)),
            "strict_improvement_fraction": float(np.mean(gain[mask] > 1e-6)),
            "gain_mean": float(np.mean(gain[mask])),
            "gain_median": float(np.median(gain[mask])),
            "gain_p05": float(np.quantile(gain[mask], 0.05)),
            "best_cost_mean": float(np.mean(best[mask])),
        }
    return result


def paired_summary(
    candidate_cost: np.ndarray,
    current_cost: np.ndarray,
    episodes: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    delta = np.min(current_cost, axis=1) - np.min(candidate_cost, axis=1)
    return {
        "best_cost_gain_over_current": distribution(delta),
        "candidate_strict_win_fraction": float(np.mean(delta > 1e-6)),
        "current_strict_win_fraction": float(np.mean(delta < -1e-6)),
        "tie_fraction": float(np.mean(np.abs(delta) <= 1e-6)),
        "episode_bootstrap_mean_ci95": bootstrap_paired_mean(
            delta, episodes, bootstrap_samples, seed,
        ),
    }


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)

    contract = json.loads((args.run / "contract.json").read_text())
    run_summary = json.loads((args.run / "summary.json").read_text())
    run_args = contract["arguments"]
    if run_args.get("actor_objective_mode") != "deterministic_center_dbm":
        raise AssertionError("bank audit requires the deterministic-center run")
    if contract.get("formal_validation_loaded") or contract.get("test_loaded"):
        raise AssertionError("sealed split violation")
    data = load_bank(Path(run_args["bank_root"]))
    folds = make_folds(data, 3)
    _, selection, _, selection_episodes = internal_split(
        data, folds, int(contract["outer_fold"])
    )
    if args.max_states > 0:
        selection = selection[:args.max_states]
        selection_episodes = sorted(np.unique(data["episode"][selection]).tolist())
    states, current, references, params_json, weights_json, dbm_json = (
        load_rollout_inputs(data, Path(run_args["gt_v1"]))
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    normalization, normalization_path = load_actor_normalization(Path(run_args["base_ac"]))
    actor_inputs = make_actor_inputs(data, normalization)
    warm = np.asarray(data["costs"][selection, 0], np.float32)
    speed = np.asarray(data["speed"][selection], np.float32)
    episode = np.asarray(data["episode"][selection])

    all_actions = []
    all_costs = []
    all_gradients = []
    all_geometry: dict[str, list[dict[str, np.ndarray]]] = {
        method: [] for method in METHODS
    }
    per_seed: dict[str, Any] = {}
    checkpoint_hashes = []
    for seed in (0, 1, 2):
        actor, checkpoint = checkpoint_actor(
            args.run, seed, float(run_args["actor_output_support_multiplier"]), device
        )
        payload = torch.load(checkpoint, map_location=device)
        log_scale = payload["log_exploration_scale"]
        mean = actor_mean(actor, actor_inputs, selection, device)
        rng_seed = 260828700 + seed
        rng = np.random.default_rng(rng_seed)
        current_bank, std, eps = current_bank_with_latents(
            mean, log_scale, rng,
            float(run_args["minimum_exploration_scale"]),
            float(run_args["maximum_exploration_scale"]),
        )
        # Bitwise guard against drifting away from the registered generator.
        expected = actor_exploration_bank(
            mean, log_scale, np.random.default_rng(rng_seed),
            float(run_args["minimum_exploration_scale"]),
            float(run_args["maximum_exploration_scale"]),
        )
        if not np.array_equal(current_bank, expected):
            raise AssertionError("current exploration generator replay mismatch")
        current_cost = rollout_bank(
            backend, weights, params, current_bank, states, current,
            references, selection, args.rollout_batch_size, device,
        )
        orth_bank = orthogonal_bank(
            mean, std, float(run_args["maximum_exploration_scale"]),
            seed, selection,
        )
        orth_cost = rollout_bank(
            backend, weights, params, orth_bank, states, current,
            references, selection, args.rollout_batch_size, device,
        )
        guided_1x, guided_direction_1x = guided_bank(
            current_bank, current_cost, mean, std, eps,
            float(run_args["maximum_exploration_scale"]), args.guided_ridge, 1.0,
        )
        guided_1x_cost = rollout_bank(
            backend, weights, params, guided_1x, states, current,
            references, selection, args.rollout_batch_size, device,
        )
        guided_2x, guided_direction_2x = guided_bank(
            current_bank, current_cost, mean, std, eps,
            float(run_args["maximum_exploration_scale"]), args.guided_ridge, 2.0,
        )
        guided_2x_cost = rollout_bank(
            backend, weights, params, guided_2x, states, current,
            references, selection, args.rollout_batch_size, device,
        )
        base_cost, exact_gradient = exact_actor_gradient(
            mean, std, selection, states, current, references,
            backend, weights, params, device,
        )
        costs = np.stack(
            (current_cost, orth_cost, guided_1x_cost, guided_2x_cost), axis=0
        )
        actions = np.stack(
            (current_bank, orth_bank, guided_1x, guided_2x), axis=0
        )
        if float(np.max(np.abs(costs[:, :, 0] - base_cost[None]))) > 1e-3:
            raise AssertionError("shared Actor center cost mismatch")
        if not (
            np.array_equal(guided_1x[:, :5], current_bank[:, :5])
            and np.array_equal(guided_2x[:, :5], current_bank[:, :5])
        ):
            raise AssertionError("guided banks do not share current first five candidates")

        seed_geometry = {}
        seed_summary = {}
        for method_index, method in enumerate(METHODS):
            geometry = bank_geometry_and_gradient(
                actions[method_index], costs[method_index], mean, std,
                exact_gradient,
            )
            all_geometry[method].append(geometry)
            seed_geometry[method] = geometry
            seed_summary[method] = method_summary(
                costs[method_index], warm, speed, geometry,
            )
        seed_summary["paired_vs_current"] = {
            method: paired_summary(
                costs[index], costs[0], episode,
                args.bootstrap_samples, 260828710 + seed * 10 + index,
            )
            for index, method in enumerate(METHODS[1:], start=1)
        }
        per_seed[str(seed)] = seed_summary
        all_actions.append(actions)
        all_costs.append(costs)
        all_gradients.append(exact_gradient)
        checkpoint_hashes.append(sha256_file(checkpoint))
        print(
            f"seed={seed} current/orth/guided1/guided2 gain="
            f"{seed_summary['current']['best_gain_vs_actor']['mean']:.3f}/"
            f"{seed_summary['orthogonal']['best_gain_vs_actor']['mean']:.3f}/"
            f"{seed_summary['guided_1x']['best_gain_vs_actor']['mean']:.3f}/"
            f"{seed_summary['guided_2x']['best_gain_vs_actor']['mean']:.3f}",
            flush=True,
        )

    actions = np.stack(all_actions, axis=0)  # seed, method, state, candidate, knot, channel
    costs = np.stack(all_costs, axis=0)
    exact_gradient = np.stack(all_gradients, axis=0)
    pooled_warm = np.tile(warm, 3)
    pooled_speed = np.tile(speed, 3)
    pooled_episode = np.tile(episode, 3)
    pooled: dict[str, Any] = {}
    for method_index, method in enumerate(METHODS):
        geometry = {
            key: np.concatenate([value[key] for value in all_geometry[method]])
            for key in all_geometry[method][0]
        }
        pooled[method] = method_summary(
            costs[:, method_index].reshape(-1, 6),
            pooled_warm, pooled_speed, geometry,
        )
    pooled["paired_vs_current"] = {
        method: paired_summary(
            costs[:, index].reshape(-1, 6),
            costs[:, 0].reshape(-1, 6),
            pooled_episode,
            args.bootstrap_samples,
            260828790 + index,
        )
        for index, method in enumerate(METHODS[1:], start=1)
    }

    arrays = {
        "method": np.asarray(METHODS),
        "role": np.asarray(ROLES),
        "seed": np.asarray((0, 1, 2), np.int16),
        "state_index": selection.astype(np.int32),
        "episode": episode,
        "speed": speed,
        "scenario": np.asarray(data["scenario"][selection]),
        "warm_cost": warm,
        "action": actions.astype(np.float32),
        "cost": costs.astype(np.float32),
        "exact_standardized_latent_gradient": exact_gradient.astype(np.float32),
    }
    np.savez_compressed(args.output_dir / "evaluation.npz", **arrays)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "ACTOR_NEIGHBORHOOD_BANK_AUDIT_COMPLETE",
        "scope": (
            "frozen deterministic-center latest Actors; fold-1 internal-selection; "
            "six deterministic DBM J50 evaluations per bank/state; no training, "
            "formal validation, test, MPPI wrapper, or closed loop"
        ),
        "contract": {
            "methods": list(METHODS),
            "evaluation_budget_per_state": 6,
            "current": "registered mean + two Gaussian antithetic pairs + blind wide sample",
            "orthogonal": (
                "mean + two state-cycled RMS-orthogonal temporal-DCT antithetic pairs + "
                "one orthogonal wide sample"
            ),
            "guided": (
                "current candidates 0:5 bitwise identical; candidate 5 replaced by a "
                "minimum-norm descent direction fitted from the two central pair slopes; "
                "1x-std and 2x-std are reported as separate equal-budget banks"
            ),
            "guided_ridge": float(args.guided_ridge),
            "warm_role": "evaluation slice only; never candidate generation or fitting",
            "exact_gradient_role": "offline diagnostic only; never candidate generation",
        },
        "checks": {
            "state_count": int(len(selection)),
            "episode_count": int(len(selection_episodes)),
            "formal_validation_loaded": False,
            "test_loaded": False,
            "current_generator_bitwise_replay_all_seed": True,
            "guided_first_five_bitwise_current_all_seed": True,
            "shared_actor_center_all_methods": True,
            "finite": bool(np.all(np.isfinite(costs))),
        },
        "manifest": {
            "run": str(args.run.resolve()),
            "run_contract_sha256": sha256_file(args.run / "contract.json"),
            "run_summary_sha256": sha256_file(args.run / "summary.json"),
            "actor_latest_sha256": checkpoint_hashes,
            "normalization": str(normalization_path.resolve()),
            "normalization_sha256": sha256_file(normalization_path),
            "selection_episode": list(selection_episodes),
        },
        "per_seed": per_seed,
        "pooled": pooled,
        "array_sha256": {
            key: sha256_array(value) for key, value in arrays.items()
        },
        "run_engineering_pass": all(
            bool(record["passed"])
            for record in json.loads((args.run / "validator_report.json").read_text())["records"]
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "qualification": summary["qualification"],
        "pooled": {
            method: {
                "gain_mean": pooled[method]["best_gain_vs_actor"]["mean"],
                "hit": pooled[method]["strict_improvement_fraction"],
                "recover_warm": pooled[method]["actor_loses_warm_recovered_fraction"],
                "rank_median": pooled[method]["geometry"]["numeric_rank"]["median"],
                "projection_median": pooled[method]["geometry"]["true_gradient_projection_fraction"]["median"],
            }
            for method in METHODS
        },
        "paired_vs_current": pooled["paired_vs_current"],
    }, indent=2))


if __name__ == "__main__":
    main()
