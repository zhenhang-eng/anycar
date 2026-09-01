#!/usr/bin/env python3
"""Train-only high-speed Actor-visited terminal OAC from v2 pretraining.

Each round evaluates the deterministic Actor center plus a full-rank antithetic
bank with the real DBM J50 objective.  Every action, including regressions, is
retained in Replay.  Twin absolute-value Critics receive twenty updates before
one bounded deterministic Actor update.  There is no next state, Bellman
bootstrap, target Critic, entropy term, or analytic DBM gradient.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from car_foundation.mppi_proposal_policy import MPPIProposalNormalization
from generate_dbm_j16_local_curvature_labels import hadamard_directions
from mppi_a2_actors import DirectNoAnchorGTXActor
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic
from run_mppi_proximal_search_phase1a import seed_bank_directions
from pretrain_highspeed_actor_twin_critic import (
    actor_predict,
    distribution,
    load_data,
    rollout_cost,
    sha256,
)


DEFAULT_PRETRAIN = Path(
    "outputs/mppi_proposal/highspeed_actor_twin_critic_pretrain_20260830_v2"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/highspeed_actor_visited_oac_20260830_v1"
)
SIGMA = np.asarray((0.25, 0.35), np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-dir", type=Path, default=DEFAULT_PRETRAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--critic-updates-per-round", type=int, default=20)
    parser.add_argument("--actor-updates-per-round", type=int, default=1)
    parser.add_argument("--critic-state-batch-size", type=int, default=8)
    parser.add_argument("--critic-candidates-per-state", type=int, default=48)
    parser.add_argument("--actor-batch-size", type=int, default=64)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--actor-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--ranking-weight", type=float, default=0.35)
    parser.add_argument("--ranking-temperature", type=float, default=0.08)
    parser.add_argument("--exploration-radius-start", type=float, default=0.20)
    parser.add_argument("--exploration-radius-end", type=float, default=0.05)
    parser.add_argument(
        "--exploration-mode",
        choices=("antithetic33", "nonrecentered65", "search_recentered65"),
        default="antithetic33",
    )
    parser.add_argument("--second-radius-ratio", type=float, default=0.70)
    parser.add_argument("--critic-absorption-updates", type=int, default=0)
    parser.add_argument("--max-round-step-sigma-rms", type=float, default=0.02)
    parser.add_argument("--trust-weight", type=float, default=0.05)
    parser.add_argument("--raw-cost-weight-cap", type=float, default=8.0)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float64).ravel()
    right = np.asarray(right, np.float64).ravel()
    if left.size < 2 or left.std() < 1e-12 or right.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def build_inputs(
    data: dict[str, np.ndarray], payload: dict[str, Any],
) -> tuple[np.ndarray, ...]:
    normalizer = MPPIProposalNormalization.from_dict(payload["normalization"])
    history, reference, current = normalizer.normalize_numpy(
        data["history"], data["reference"], data["current"]
    )
    count = len(history)
    return (
        history.astype(np.float32), reference.astype(np.float32), current.astype(np.float32),
        np.zeros((count, 8, 2), np.float32), np.zeros((count, 74), np.float32),
        np.zeros((count, 32), np.float32),
    )


def load_actor(payload: dict[str, Any], device: torch.device) -> DirectNoAnchorGTXActor:
    model = DirectNoAnchorGTXActor(dropout=0.0).to(device)
    model.load_state_dict(payload["actor_state_dict"], strict=True)
    return model


def load_critic(
    payload: dict[str, Any], twin: int, device: torch.device,
) -> ConfigurableAbsoluteActionValueCritic:
    model = ConfigurableAbsoluteActionValueCritic().to(device)
    model.load_state_dict(payload[f"critic{twin}_state_dict"], strict=True)
    return model


def make_exploration_bank(center: np.ndarray, radius: float) -> np.ndarray:
    direction = hadamard_directions().astype(np.float32)
    delta = radius * SIGMA.reshape(1, 1, 1, 2) * direction.reshape(1, 16, 8, 2)
    return np.clip(
        np.concatenate((center[:, None], center[:, None] + delta, center[:, None] - delta), axis=1),
        -1.0, 1.0,
    ).astype(np.float32)


def make_second_ring(center: np.ndarray, radius: float) -> np.ndarray:
    """Return 16 deterministic antithetic pairs without repeating center."""
    direction = seed_bank_directions(2).astype(np.float32)
    delta = radius * SIGMA.reshape(1, 1, 1, 2) * direction.reshape(1, 16, 8, 2)
    return np.clip(
        np.concatenate((center[:, None] + delta, center[:, None] - delta), axis=1),
        -1.0, 1.0,
    ).astype(np.float32)


def make_two_stage_exploration_bank(
    data: dict[str, np.ndarray], center: np.ndarray, rows: np.ndarray,
    radius: float, second_radius_ratio: float, mode: str,
    backend: TorchDynamicBicycleRolloutBackend,
    weights: TorchMPPICostWeights, params: TorchMPPIParams,
    device: torch.device, batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a paired 1+32+32 bank; only stage-two recentering differs."""
    first = make_exploration_bank(center, radius)
    first_cost = rollout_bank(
        data, first, rows, backend, weights, params, device, batch_size
    )
    row = np.arange(len(rows))
    first_best = np.argmin(first_cost, axis=1)
    incumbent = first[row, first_best]
    second_base = center if mode == "nonrecentered65" else incumbent
    second = make_second_ring(second_base, radius * second_radius_ratio)
    second_cost = rollout_bank(
        data, second, rows, backend, weights, params, device, batch_size
    )
    bank = np.concatenate((first, second), axis=1)
    cost = np.concatenate((first_cost, second_cost), axis=1)
    diagnostics = {
        "first_stage_gain": distribution(first_cost[:, 0] - first_cost.min(axis=1)),
        "full_bank_gain": distribution(first_cost[:, 0] - cost.min(axis=1)),
        "first_stage_move_fraction": float(np.mean(first_best != 0)),
        "stage2_recentered": bool(mode == "search_recentered65"),
        "second_radius_sigma": float(radius * second_radius_ratio),
    }
    return bank, cost, diagnostics


def rollout_bank(
    data: dict[str, np.ndarray], bank: np.ndarray, rows: np.ndarray,
    backend: TorchDynamicBicycleRolloutBackend, weights: TorchMPPICostWeights,
    params: TorchMPPIParams, device: torch.device, batch_size: int,
) -> np.ndarray:
    state_count, candidate_count = bank.shape[:2]
    repeated_rows = np.repeat(rows, candidate_count)
    flat = bank.reshape(state_count * candidate_count, 8, 2)
    return rollout_cost(
        data, flat, repeated_rows, backend, weights, params, device, batch_size
    ).reshape(state_count, candidate_count)


def actor_metrics(
    cost: np.ndarray, rows: np.ndarray, data: dict[str, np.ndarray],
    initial_cost: np.ndarray,
) -> dict[str, Any]:
    warm = data["anchor_cost"][rows]
    teacher = data["teacher_cost"][rows]
    gain_warm = warm - cost
    gain_initial = initial_cost - cost
    teacher_gain = warm - teacher
    guard = np.minimum(warm, cost)
    by_speed = {}
    for speed in sorted(np.unique(data["speed"][rows])):
        mask = np.isclose(data["speed"][rows], speed)
        by_speed[f"{speed:.0f}"] = {
            "count": int(np.sum(mask)),
            "mean_cost": float(np.mean(cost[mask])),
            "mean_gain_vs_warm": float(np.mean(gain_warm[mask])),
            "median_gain_vs_warm": float(np.median(gain_warm[mask])),
            "p05_gain_vs_warm": float(np.quantile(gain_warm[mask], 0.05)),
        }
    by_scenario = {}
    for scenario in sorted(np.unique(data["scenario"][rows])):
        mask = data["scenario"][rows] == scenario
        by_scenario[str(scenario)] = {
            "count": int(np.sum(mask)),
            "mean_gain_vs_warm": float(np.mean(gain_warm[mask])),
            "median_gain_vs_warm": float(np.median(gain_warm[mask])),
        }
    return {
        "cost": distribution(cost),
        "gain_vs_warm": distribution(gain_warm),
        "gain_vs_pretrained_actor": distribution(gain_initial),
        "teacher_gain_vs_warm": distribution(teacher_gain),
        "teacher_gain_recovery": float(np.sum(gain_warm) / np.sum(teacher_gain)),
        "beats_or_equals_warm_fraction": float(np.mean(gain_warm >= -1e-5)),
        "regression_fraction_vs_warm": float(np.mean(gain_warm < -1e-5)),
        "beats_pretrained_fraction": float(np.mean(gain_initial > 1e-5)),
        "two_center_guard": {
            "cost": distribution(guard),
            "gain_vs_warm": distribution(warm - guard),
            "warm_selected_fraction": float(np.mean(warm <= cost)),
            "warm_floor_violation_count": int(np.sum(guard > warm + 1e-6)),
        },
        "by_speed_kph": by_speed,
        "by_scenario": by_scenario,
    }


def critic_prediction(
    critic: nn.Module, training: dict[str, Any], inputs: tuple[np.ndarray, ...],
    rows: np.ndarray, actions: np.ndarray, device: torch.device,
) -> np.ndarray:
    output = []
    critic.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 16):
            local = rows[start : start + 16]
            prediction = critic(
                torch.from_numpy(inputs[0][local]).to(device),
                torch.from_numpy(inputs[1][local]).to(device),
                torch.from_numpy(inputs[2][local]).to(device),
                torch.from_numpy(actions[start : start + len(local)]).to(device),
            )
            output.append((
                prediction * float(training["target_std"])
                + float(training["target_mean"])
            ).cpu().numpy())
    return np.concatenate(output)


def local_critic_metrics(
    critics: tuple[nn.Module, nn.Module], training: tuple[dict, dict],
    inputs: tuple[np.ndarray, ...], rows: np.ndarray, bank: np.ndarray,
    costs: np.ndarray, device: torch.device,
) -> dict[str, Any]:
    prediction = [
        critic_prediction(model, spec, inputs, rows, bank, device)
        for model, spec in zip(critics, training)
    ]
    conservative = np.maximum(prediction[0], prediction[1])
    truth = np.log1p(costs.astype(np.float64))
    predicted_delta = conservative[:, 1:] - conservative[:, :1]
    true_delta = truth[:, 1:] - truth[:, :1]
    material = np.abs(true_delta) >= 1e-5
    row = np.arange(len(rows))
    selected = np.argmin(conservative, axis=1)
    best = np.argmin(costs, axis=1)
    gain = costs[:, 0] - costs[row, selected]
    available = costs[:, 0] - costs[row, best]
    return {
        "centered_log_cost_pearson": metric_correlation(
            conservative - conservative[:, :1], truth - truth[:, :1]
        ),
        "center_relative_sign_accuracy": float(np.mean(
            np.sign(predicted_delta[material]) == np.sign(true_delta[material])
        )),
        "twin_log_disagreement": distribution(np.abs(prediction[0] - prediction[1])),
        "bank_gain_recovery": float(np.sum(gain) / max(np.sum(available), 1e-12)),
        "bank_regret": distribution(costs[row, selected] - costs[row, best]),
        "selected_beats_center_fraction": float(np.mean(gain > 1e-5)),
    }


def critic_update(
    critic: nn.Module, optimizer: torch.optim.Optimizer, training: dict[str, Any],
    inputs: tuple[np.ndarray, ...], fit: np.ndarray, replay_actions: np.ndarray,
    replay_costs: np.ndarray, rng: np.random.Generator, args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    critic.train()
    local_state = rng.integers(len(fit), size=args.critic_state_batch_size)
    rows = fit[local_state]
    candidate_count = replay_actions.shape[1]
    sample_count = min(args.critic_candidates_per_state, candidate_count)
    candidate = np.stack([
        rng.choice(candidate_count, size=sample_count, replace=False)
        for _ in local_state
    ])
    actions = torch.from_numpy(replay_actions[local_state[:, None], candidate]).to(device)
    raw_cost = replay_costs[local_state[:, None], candidate]
    target = torch.from_numpy((
        (np.log1p(raw_cost.astype(np.float64)) - float(training["target_mean"]))
        / float(training["target_std"])
    ).astype(np.float32)).to(device)
    prediction = critic(
        torch.from_numpy(inputs[0][rows]).to(device),
        torch.from_numpy(inputs[1][rows]).to(device),
        torch.from_numpy(inputs[2][rows]).to(device), actions,
    )
    value = torch.nn.functional.smooth_l1_loss(prediction, target)
    left = torch.randint(sample_count, (len(rows), sample_count), device=device)
    right = torch.randint(sample_count, (len(rows), sample_count), device=device)
    batch = torch.arange(len(rows), device=device)[:, None]
    predicted_delta = prediction[batch, left] - prediction[batch, right]
    target_delta = target[batch, left] - target[batch, right]
    material = target_delta.abs() > 1e-5
    ranking = torch.nn.functional.softplus(
        -target_delta[material].sign() * predicted_delta[material]
        / args.ranking_temperature
    ).mean()
    loss = value + args.ranking_weight * ranking
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = float(torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0))
    optimizer.step()
    return {
        "loss": float(loss.detach()), "value": float(value.detach()),
        "ranking": float(ranking.detach()), "gradient_norm": gradient,
    }


def actor_update(
    actor: nn.Module, selected_actor: nn.Module, optimizer: torch.optim.Optimizer,
    critics: tuple[nn.Module, nn.Module], training: tuple[dict, dict],
    inputs: tuple[np.ndarray, ...], fit: np.ndarray, rng: np.random.Generator,
    args: argparse.Namespace, device: torch.device,
) -> dict[str, float]:
    rows = rng.choice(fit, size=min(args.actor_batch_size, len(fit)), replace=False)
    before_parameters = {
        name: value.detach().clone() for name, value in actor.named_parameters()
    }
    before_all = actor_predict(actor, inputs, fit, device)
    actor.train()
    tensors = tuple(torch.from_numpy(value[rows]).to(device) for value in inputs)
    center = actor(*tensors)[1]
    with torch.no_grad():
        selected = selected_actor(*tensors)[1]
    for critic in critics:
        critic.eval()
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
    physical_log = []
    for critic, spec in zip(critics, training):
        prediction = critic(
            tensors[0], tensors[1], tensors[2], center[:, None]
        )[:, 0]
        physical_log.append(
            prediction * float(spec["target_std"]) + float(spec["target_mean"])
        )
    conservative = torch.maximum(physical_log[0], physical_log[1])
    # gamma=1 raw-J aggregation, stabilized by subtracting the batch log mean
    # before exponentiation.  This changes cross-state weighting only.
    weight = torch.exp(conservative.detach() - conservative.detach().mean()).clamp(
        max=args.raw_cost_weight_cap
    )
    weight = weight / weight.mean().clamp_min(1e-12)
    sigma = torch.from_numpy(SIGMA).to(device).reshape(1, 1, 2)
    trust = (((center - selected) / sigma) ** 2).mean()
    value_term = (weight * conservative).mean()
    loss = value_term + args.trust_weight * trust
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0))
    optimizer.step()
    for critic in critics:
        for parameter in critic.parameters():
            parameter.requires_grad_(True)
    after_all = actor_predict(actor, inputs, fit, device)
    step = float(np.sqrt(np.mean(((after_all - before_all) / SIGMA) ** 2)))
    projection = (
        1.0 if getattr(args, "defer_actor_projection", False)
        else min(1.0, args.max_round_step_sigma_rms / max(step, 1e-12))
    )
    if projection < 1.0:
        with torch.no_grad():
            for name, parameter in actor.named_parameters():
                parameter.copy_(
                    before_parameters[name]
                    + projection * (parameter - before_parameters[name])
                )
        after_all = actor_predict(actor, inputs, fit, device)
        step = float(np.sqrt(np.mean(((after_all - before_all) / SIGMA) ** 2)))
    return {
        "loss": float(loss.detach()), "value": float(value_term.detach()),
        "trust": float(trust.detach()), "gradient_norm": gradient,
        "step_sigma_rms": step, "trust_projection": projection,
        "raw_weight_ess_fraction": float(
            (weight.sum().square() / (weight.square().sum() * len(weight))).detach()
        ),
        "raw_weight_max": float(weight.max().detach()),
    }


def evaluate_actor(
    actor: nn.Module, inputs: tuple[np.ndarray, ...], rows: np.ndarray,
    data: dict[str, np.ndarray], initial_cost: np.ndarray,
    backend: TorchDynamicBicycleRolloutBackend, weights: TorchMPPICostWeights,
    params: TorchMPPIParams, device: torch.device, batch_size: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    center = actor_predict(actor, inputs, rows, device)
    cost = rollout_cost(data, center, rows, backend, weights, params, device, batch_size)
    return actor_metrics(cost, rows, data, initial_cost), center, cost


def run_one(
    fold: int, seed: int, args: argparse.Namespace, data: dict[str, np.ndarray],
    pretrain_summary: dict[str, Any], output: Path, device: torch.device,
) -> dict[str, Any]:
    source_path = args.pretrain_dir / "checkpoints" / f"pretrain_fold{fold}_seed{seed}.pt"
    source_record = next(
        value for value in pretrain_summary["records"]
        if value["fold"] == fold and value["seed"] == seed
    )
    if sha256(source_path) != source_record["checkpoint_sha256"]:
        raise AssertionError("pretrain checkpoint hash mismatch")
    payload = torch.load(source_path, map_location=device)
    fit = np.asarray(payload["fit_indices"], np.int64)
    selection = np.asarray(payload["selection_indices"], np.int64)
    oof = np.asarray(payload["oof_indices"], np.int64)
    if set(data["episode"][fit]) & set(data["episode"][selection]):
        raise AssertionError("fit/selection episode leakage")
    if set(data["episode"][fit]) & set(data["episode"][oof]):
        raise AssertionError("fit/OOF episode leakage")
    inputs = build_inputs(data, payload)
    set_seed(830_000 + fold * 100 + seed)
    rng = np.random.default_rng(830_000 + fold * 100 + seed)
    actor = load_actor(payload, device)
    critics = (load_critic(payload, 1, device), load_critic(payload, 2, device))
    training = (payload["critic1_training"], payload["critic2_training"])
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.actor_lr, weight_decay=args.weight_decay
    )
    critic_optimizers = tuple(torch.optim.AdamW(
        model.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay
    ) for model in critics)
    backend = TorchDynamicBicycleRolloutBackend()
    weights = TorchMPPICostWeights()
    params = TorchMPPIParams(num_samples=64)

    initial = {}
    initial_cost = {}
    for name, rows in (("fit", fit), ("selection", selection), ("oof", oof)):
        center = actor_predict(actor, inputs, rows, device)
        cost = rollout_cost(
            data, center, rows, backend, weights, params, device,
            args.rollout_batch_size,
        )
        initial[name] = actor_metrics(cost, rows, data, cost)
        initial_cost[name] = cost
    selected_actor = copy.deepcopy(actor).eval()
    selected_round = 0
    selected_selection_cost = float(np.mean(initial_cost["selection"]))
    selected_metrics = initial["selection"]

    replay_actions = data["actions"][fit].copy()
    replay_costs = data["costs"][fit].copy()
    source_replay_candidates = int(replay_actions.shape[1])

    def collect_bank(
        center: np.ndarray, rows: np.ndarray, radius: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if args.exploration_mode == "antithetic33":
            bank = make_exploration_bank(center, radius)
            cost = rollout_bank(
                data, bank, rows, backend, weights, params, device,
                args.rollout_batch_size,
            )
            return bank, cost, {
                "first_stage_gain": distribution(cost[:, 0] - cost.min(axis=1)),
                "full_bank_gain": distribution(cost[:, 0] - cost.min(axis=1)),
                "first_stage_move_fraction": float(np.mean(np.argmin(cost, axis=1) != 0)),
                "stage2_recentered": False,
                "second_radius_sigma": 0.0,
            }
        return make_two_stage_exploration_bank(
            data, center, rows, radius, args.second_radius_ratio,
            args.exploration_mode, backend, weights, params, device,
            args.rollout_batch_size,
        )

    absorption_record = None
    if args.critic_absorption_updates > 0:
        absorption_center = actor_predict(actor, inputs, fit, device)
        absorption_bank, absorption_cost, absorption_bank_diagnostics = collect_bank(
            absorption_center, fit, args.exploration_radius_start
        )
        replay_actions = np.concatenate((replay_actions, absorption_bank), axis=1)
        replay_costs = np.concatenate((replay_costs, absorption_cost), axis=1)
        absorption_logs = []
        for _ in range(args.critic_absorption_updates):
            for critic, optimizer, spec in zip(critics, critic_optimizers, training):
                absorption_logs.append(critic_update(
                    critic, optimizer, spec, inputs, fit, replay_actions,
                    replay_costs, rng, args, device,
                ))
        absorption_record = {
            "updates_per_twin": int(args.critic_absorption_updates),
            "actor_updated": False,
            "replay_candidates_per_fit_state": int(replay_actions.shape[1]),
            "bank": absorption_bank_diagnostics,
            "critic": {
                "loss_mean": float(np.mean([value["loss"] for value in absorption_logs])),
                "value_mean": float(np.mean([value["value"] for value in absorption_logs])),
                "ranking_mean": float(np.mean([value["ranking"] for value in absorption_logs])),
            },
        }
    round_records = []
    for round_index in range(1, args.rounds + 1):
        fraction = (round_index - 1) / max(args.rounds - 1, 1)
        radius = (
            args.exploration_radius_start
            + fraction * (args.exploration_radius_end - args.exploration_radius_start)
        )
        current_center = actor_predict(actor, inputs, fit, device)
        bank, cost, bank_diagnostics = collect_bank(current_center, fit, radius)
        replay_actions = np.concatenate((replay_actions, bank), axis=1)
        replay_costs = np.concatenate((replay_costs, cost), axis=1)
        critic_logs = []
        for update in range(args.critic_updates_per_round):
            for twin, (critic, optimizer, spec) in enumerate(zip(
                critics, critic_optimizers, training
            )):
                critic_logs.append(critic_update(
                    critic, optimizer, spec, inputs, fit, replay_actions,
                    replay_costs, rng, args, device,
                ))
        actor_logs = []
        for _ in range(args.actor_updates_per_round):
            actor_logs.append(actor_update(
                actor, selected_actor, actor_optimizer, critics, training,
                inputs, fit, rng, args, device,
            ))
        selection_metrics, _, selection_cost = evaluate_actor(
            actor, inputs, selection, data, initial_cost["selection"],
            backend, weights, params, device, args.rollout_batch_size,
        )
        if float(np.mean(selection_cost)) < selected_selection_cost:
            selected_selection_cost = float(np.mean(selection_cost))
            selected_round = round_index
            selected_actor = copy.deepcopy(actor).eval()
            selected_metrics = selection_metrics
        round_records.append({
            "round": round_index, "exploration_radius_sigma": radius,
            "replay_candidates_per_fit_state": int(replay_actions.shape[1]),
            "new_bank_gain_vs_actor": distribution(cost[:, 0] - cost.min(axis=1)),
            "bank": bank_diagnostics,
            "critic": {
                "loss_mean": float(np.mean([value["loss"] for value in critic_logs])),
                "value_mean": float(np.mean([value["value"] for value in critic_logs])),
                "ranking_mean": float(np.mean([value["ranking"] for value in critic_logs])),
            },
            "actor": actor_logs[-1],
            "selection": selection_metrics,
            "selected": bool(selected_round == round_index),
        })
        print(
            f"fold={fold} seed={seed} round={round_index:02d} "
            f"sel_gain={selection_metrics['gain_vs_pretrained_actor']['mean']:.1f} "
            f"selected_round={selected_round} step={actor_logs[-1]['step_sigma_rms']:.5f}",
            flush=True,
        )

    latest_actor = actor
    final = {}
    centers = {}
    costs = {}
    for role, model in (("selected", selected_actor), ("latest", latest_actor)):
        final[role] = {}
        centers[role] = {}
        costs[role] = {}
        for name, rows in (("fit", fit), ("selection", selection), ("oof", oof)):
            metrics, center, cost = evaluate_actor(
                model, inputs, rows, data, initial_cost[name], backend, weights,
                params, device, args.rollout_batch_size,
            )
            final[role][name] = metrics
            centers[role][name] = center
            costs[role][name] = cost

    # Fresh, never-trained OOF local bank checks the final Critic where Actor
    # updates actually need local ranking.  This bank is diagnostic only.
    oof_bank = make_exploration_bank(centers["selected"]["oof"], 0.05)
    oof_bank_cost = rollout_bank(
        data, oof_bank, oof, backend, weights, params, device,
        args.rollout_batch_size,
    )
    critic_oof = local_critic_metrics(
        critics, training, inputs, oof, oof_bank, oof_bank_cost, device
    )

    run_dir = output / f"fold{fold}_seed{seed}"
    run_dir.mkdir()
    replay_path = run_dir / "replay.npz"
    np.savez_compressed(
        replay_path, fit_indices=fit, actions=replay_actions, costs=replay_costs,
        source_replay_candidates=np.asarray(source_replay_candidates, np.int64),
        absorption_candidates=np.asarray(
            0 if absorption_record is None else absorption_bank.shape[1], np.int64
        ),
        selected_oof_center=centers["selected"]["oof"],
        selected_oof_cost=costs["selected"]["oof"],
        oof_probe_bank=oof_bank, oof_probe_cost=oof_bank_cost,
    )
    checkpoint = run_dir / "oac_checkpoint.pt"
    torch.save({
        "qualification": "HIGHSPEED_ACTOR_VISITED_OAC_TRAIN_ONLY",
        "fold": fold, "seed": seed, "selected_round": selected_round,
        "source_pretrain_checkpoint": str(source_path.resolve()),
        "source_pretrain_sha256": sha256(source_path),
        "normalization": payload["normalization"],
        "actor_selected_state_dict": selected_actor.state_dict(),
        "actor_latest_state_dict": latest_actor.state_dict(),
        "critic1_state_dict": critics[0].state_dict(),
        "critic2_state_dict": critics[1].state_dict(),
        "critic1_training": training[0], "critic2_training": training[1],
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic1_optimizer": critic_optimizers[0].state_dict(),
        "critic2_optimizer": critic_optimizers[1].state_dict(),
        "fit_indices": fit, "selection_indices": selection, "oof_indices": oof,
        "formal_validation_or_test_created": False,
    }, checkpoint)
    return {
        "fold": fold, "seed": seed, "selected_round": selected_round,
        "source_pretrain_checkpoint": str(source_path.resolve()),
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
        "replay": str(replay_path.resolve()), "replay_sha256": sha256(replay_path),
        "replay_candidates_per_fit_state": int(replay_actions.shape[1]),
        "source_replay_candidates_per_fit_state": source_replay_candidates,
        "absorption": absorption_record,
        "initial": initial, "selected": final["selected"], "latest": final["latest"],
        "selected_internal_selection": selected_metrics,
        "critic_oof_local_probe": critic_oof,
        "rounds": round_records,
    }


def aggregate(records: list[dict[str, Any]], role: str, split: str) -> dict[str, Any]:
    values = [record[role][split] for record in records]
    keys = (
        "teacher_gain_recovery", "beats_or_equals_warm_fraction",
        "regression_fraction_vs_warm", "beats_pretrained_fraction",
    )
    result = {key: distribution(np.asarray([value[key] for value in values])) for key in keys}
    for gain_name in ("gain_vs_warm", "gain_vs_pretrained_actor"):
        result[gain_name] = {
            statistic: distribution(np.asarray([
                value[gain_name][statistic] for value in values
            ])) for statistic in ("mean", "p05", "median", "min")
        }
    return result


def main() -> None:
    args = parse_args()
    args.pretrain_dir = args.pretrain_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    output.mkdir(parents=True)
    pretrain_summary_path = args.pretrain_dir / "summary.json"
    pretrain_validator_path = args.pretrain_dir / "validator_report.json"
    pretrain_summary = json.loads(pretrain_summary_path.read_text())
    pretrain_validator = json.loads(pretrain_validator_path.read_text())
    if pretrain_validator["qualification"] != "HIGHSPEED_ACTOR_TWIN_CRITIC_PRETRAIN_INDEPENDENT_REPLAY_PASS":
        raise AssertionError("pretraining independent replay did not pass")
    if pretrain_summary["contract"]["current_input"] != "[vx, yaw_rate, acceleration, steering]; no vy/beta":
        raise AssertionError("refusing legacy vy checkpoint")
    source = pretrain_summary["source"]
    data = load_data(Path(source["replay_dir"]), Path(source["teacher_dir"]))
    source_indices = np.asarray(pretrain_summary["contract"].get(
        "source_indices", np.arange(len(data["episode"]), dtype=np.int64)
    ), np.int64)
    full_count = len(data["episode"])
    data = {
        key: value[source_indices]
        if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == full_count
        else value
        for key, value in data.items()
    }
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    device = torch.device(args.device)
    contract = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_ACTOR_VISITED_OAC_CONTRACT_TRAIN_ONLY",
        "source_pretrain": str(args.pretrain_dir),
        "source_pretrain_summary_sha256": sha256(pretrain_summary_path),
        "source_pretrain_validator_sha256": sha256(pretrain_validator_path),
        "folds": folds, "seeds": seeds,
        "current_input": "[vx, yaw_rate, acceleration, steering]; no vy/beta",
        "action": "absolute 8x2 [acceleration, steering] center, no warm input",
        "environment": "terminal deterministic DBM J50 contextual bandit",
        "replay": "all initial 129 search centers plus every Actor/probe center, regressions retained",
        "exploration": {
            "mode": args.exploration_mode,
            "first_ring": "Actor center + 16 antithetic Hadamard pairs",
            "second_ring": (
                "none" if args.exploration_mode == "antithetic33"
                else "16 rotated-Hadamard antithetic pairs"
            ),
            "second_stage_recentered": bool(
                args.exploration_mode == "search_recentered65"
            ),
            "radius_start_end_sigma": [
                args.exploration_radius_start, args.exploration_radius_end
            ],
            "second_radius_ratio": args.second_radius_ratio,
        },
        "critic_actor_update_ratio": f"{args.critic_updates_per_round}:{args.actor_updates_per_round}",
        "actor_objective": "conservative Twin log-value gradient with detached gamma=1 raw-J cross-state weights",
        "warm_in_actor_loss": False,
        "warm_role": "evaluation baseline and two-center guard only",
        "bellman_bootstrap": False, "target_critic": False,
        "analytic_dbm_gradient": False,
        "checkpoint_selection": "lowest internal-selection deterministic DBM mean cost; OOF never selects",
        "formal_validation_or_test_created": False,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    records = []
    for fold in folds:
        for seed in seeds:
            print(f"starting fold={fold} seed={seed}", flush=True)
            records.append(run_one(fold, seed, args, data, pretrain_summary, output, device))
    critic_keys = (
        "centered_log_cost_pearson", "center_relative_sign_accuracy",
        "bank_gain_recovery", "selected_beats_center_fraction",
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_ACTOR_VISITED_OAC_COMPLETE_TRAIN_ONLY",
        "contract_sha256": sha256(output / "contract.json"),
        "record_count": len(records),
        "selected_round": distribution(np.asarray([value["selected_round"] for value in records])),
        "initial_oof": aggregate(records, "initial", "oof"),
        "selected_oof": aggregate(records, "selected", "oof"),
        "latest_oof": aggregate(records, "latest", "oof"),
        "critic_oof_local_probe": {
            key: distribution(np.asarray([
                record["critic_oof_local_probe"][key] for record in records
            ])) for key in critic_keys
        },
        "registered_gate": {
            "selected_oof_mean_gain_vs_pretrained_positive_median": bool(np.median([
                record["selected"]["oof"]["gain_vs_pretrained_actor"]["mean"]
                for record in records
            ]) > 0.0),
            "selected_oof_improves_pretrained_in_at_least_2_of_3_seeds_each_fold": all(
                sum(
                    record["selected"]["oof"]["gain_vs_pretrained_actor"]["mean"] > 0.0
                    for record in records if record["fold"] == fold
                ) >= 2 for fold in folds
            ),
            "selected_oof_gain_vs_warm_p05_nonnegative_all_runs": all(
                record["selected"]["oof"]["gain_vs_warm"]["p05"] >= 0.0
                for record in records
            ),
            "critic_local_sign_accuracy_median_at_least_0_70": bool(np.median([
                record["critic_oof_local_probe"]["center_relative_sign_accuracy"]
                for record in records
            ]) >= 0.70),
        },
        "formal_validation_or_test_created": False,
        "records": records,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "qualification": summary["qualification"],
        "selected_round": summary["selected_round"],
        "initial_oof": summary["initial_oof"],
        "selected_oof": summary["selected_oof"],
        "critic_oof_local_probe": summary["critic_oof_local_probe"],
        "registered_gate": summary["registered_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
