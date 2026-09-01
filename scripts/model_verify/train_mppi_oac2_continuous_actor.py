#!/usr/bin/env python3
"""Run the registered OAC-2 continuous contextual-bandit pilot.

This is single-step SAC-style training, not a vehicle-transition MDP.  Each
round queries deterministic DBM costs for the current Actor mean and bounded
tanh-Gaussian exploration actions, appends every result to Replay, performs 20
joint Twin-Value/move-coefficient updates, and then performs one Actor update.

The deployment Actor remains a one-shot deterministic absolute 8x2 center.
The stochastic scale is training-only.  A train-episode internal carve-out is
used only for deterministic DBM checkpoint selection.  Formal validation and
test data are never loaded.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from generate_dbm_proposal_teacher import sha256_file
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from run_mppi_absolute_action_value_critic_cv import (
    AbsoluteActionValueCritic,
    make_folds,
)
from mppi_a2_actors import (
    DirectNoAnchorGTXActor,
    DirectNoAnchorGTXSupportActor,
    OAC_DEFAULT_OUTPUT_SUPPORT_MULTIPLIER,
)
from mppi_pair_delta_critic import (
    ConfigurableAbsoluteActionValueCritic,
)
from train_mppi_joint_value_gap_pilot import (
    ContinuousGapHead,
    evaluate as evaluate_critic,
    gap_objective,
    group_best_rows,
    sample_gap_batch,
    value_objective,
)
from train_mppi_online_absolute_sac import (
    BASE_SIGMA,
    ROLE_NAMES,
    actor_mean,
    critic_state_inputs,
    distribution,
    explore_actions,
    load_actor_normalization,
    load_bank,
    make_actor_inputs,
    material_pair_accuracy,
    module_digest,
    predict_actions,
    rollout_bank,
    sample_pairs,
    sample_training_points,
    stratified_contexts,
)


DEFAULT_PARENT = Path(
    "outputs/mppi_proposal/online_absolute_sac_joint_move_coefficient_fold1_20260820_v1"
)
DEFAULT_BANK = Path("outputs/mppi_proposal/absolute_action_value_critic_20260820_v1")
DEFAULT_ACTOR = Path("outputs/mppi_proposal/j16_noanchor_gt_x_20260820_v1")
DEFAULT_BASE_AC = Path(
    "outputs/mppi_proposal/direct_residual_online_ac_20260811_v2/"
    "direct_residual_online_ac_selected.pt"
)
DEFAULT_GT = Path("outputs/mppi_proposal/dbm_direct_gt_train_20260807_v1")
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_fold1_20260824_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--actor-root", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--base-ac", type=Path, default=DEFAULT_BASE_AC)
    parser.add_argument("--gt-v1", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--contexts-per-round", type=int, default=256)
    parser.add_argument("--critic-updates-per-round", type=int, default=20)
    parser.add_argument("--actor-updates-per-round", type=int, default=1)
    parser.add_argument(
        "--actor-gradient-source", choices=("critic", "dbm"), default="critic",
        help=(
            "Source of the Actor cost gradient. critic preserves OAC; dbm uses "
            "the differentiable deterministic DBM J50 while leaving Replay and "
            "Twin-Critic training active for a matched mechanism A/B."
        ),
    )
    parser.add_argument(
        "--actor-objective-mode",
        choices=("stochastic_sac", "deterministic_center_dbm"),
        default="stochastic_sac",
        help=(
            "Actor objective contract. stochastic_sac preserves the registered "
            "sampled-action/entropy/move-coefficient/tail path. "
            "deterministic_center_dbm directly minimizes deterministic DBM J50 "
            "at the deployed Actor mean plus trust; Replay exploration and "
            "continuous Twin-Critic training remain active."
        ),
    )
    parser.add_argument(
        "--matched-gradient-source-pilot", action="store_true",
        help=(
            "Authorize the registered K=16/90-round exact-DBM gradient-source arm."
        ),
    )
    parser.add_argument(
        "--matched-gradient-source-smoke", action="store_true",
        help="Allow a one- or two-round DBM gradient-source contract smoke test.",
    )
    parser.add_argument(
        "--multi-actor-update-pilot", action="store_true",
        help=(
            "Explicitly authorize the K=1/4/8 Actor-microstep pilot. The "
            "twenty Critic updates are interleaved across K Actor updates, "
            "the scheduled Actor LR is divided by K, and the existing "
            "--max-step-sigma-rms becomes a cumulative per-round limit."
        ),
    )
    parser.add_argument(
        "--actor-visited-refresh-pilot", action="store_true",
        help=(
            "Authorize the equal-budget high-frequency refresh contract: ten "
            "Critic and eight Actor updates per half-round, with auxiliary "
            "temperature/tail updates normally gated every two refresh rounds."
        ),
    )
    parser.add_argument(
        "--aux-update-interval-rounds", type=int, default=1,
        help="Update temperature and adaptive tail dual every N outer rounds.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pair-batch-size", type=int, default=64)
    parser.add_argument("--gap-batch-size", type=int, default=128)
    parser.add_argument("--actor-batch-size", type=int, default=128)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-4)
    parser.add_argument("--gap-learning-rate", type=float, default=2e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--actor-learning-rate-schedule",
        choices=("constant", "staged_cosine"),
        default="constant",
        help=(
            "Actor optimizer schedule. staged_cosine holds the initial LR, "
            "then decays through a middle LR to --actor-learning-rate."
        ),
    )
    parser.add_argument("--actor-learning-rate-initial", type=float)
    parser.add_argument("--actor-learning-rate-middle", type=float)
    parser.add_argument("--actor-learning-rate-initial-rounds", type=int, default=20)
    parser.add_argument("--actor-learning-rate-middle-round", type=int, default=80)
    parser.add_argument("--temperature-learning-rate", type=float, default=1e-4)
    parser.add_argument("--initial-temperature", type=float, default=0.01)
    parser.add_argument("--target-entropy", type=float, default=-16.0)
    parser.add_argument("--trust-weight", type=float, default=0.25)
    parser.add_argument("--max-step-sigma-rms", type=float, default=0.02)
    parser.add_argument(
        "--actor-cost-weight-gamma", type=float, choices=(0.0, 0.5, 1.0),
        default=0.0,
        help=(
            "Tempered cross-state Actor cost aggregation. Zero preserves "
            "mean log1p(J); one uses the mean-J gradient direction when the "
            "detached weight cap is inactive."
        ),
    )
    parser.add_argument(
        "--actor-cost-weight-maximum", type=float, default=2048.0,
        help=(
            "Maximum detached exp(gamma*q) state weight before batch-mean "
            "normalization. q is the conservative Twin log1p(J) prediction."
        ),
    )
    parser.add_argument("--initial-exploration-scale", type=float, default=0.25)
    parser.add_argument("--minimum-exploration-scale", type=float, default=0.05)
    parser.add_argument("--maximum-exploration-scale", type=float, default=0.50)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--ranking-weight", type=float, default=0.35)
    parser.add_argument("--ranking-temperature", type=float, default=0.25)
    parser.add_argument("--gap-weight", type=float, default=0.05)
    parser.add_argument("--material-gap", type=float, default=0.1)
    parser.add_argument("--flat-gap", type=float, default=0.1)
    parser.add_argument("--tail-regression-weight", type=float, default=0.0)
    parser.add_argument("--tail-cvar-fraction", type=float, default=0.10)
    parser.add_argument("--tail-regression-margin-log", type=float, default=0.0)
    parser.add_argument(
        "--tail-constraint-mode", choices=("fixed", "adaptive"), default="fixed"
    )
    parser.add_argument("--tail-cvar-budget-log", type=float, default=0.0)
    parser.add_argument("--tail-dual-learning-rate", type=float, default=1.0)
    parser.add_argument("--tail-dual-initial", type=float, default=0.0)
    parser.add_argument("--tail-dual-maximum", type=float, default=10.0)
    parser.add_argument("--tail-dual-ema-decay", type=float, default=0.90)
    parser.add_argument(
        "--statewise-risk-shrink-weight", type=float, default=0.0,
        help=(
            "Continuously pull risky per-state Actor means back toward the "
            "frozen selected center. Zero preserves the registered baseline."
        ),
    )
    parser.add_argument(
        "--statewise-risk-temperature-log", type=float, default=0.02,
        help="Temperature of the continuous Twin-Value risk coefficient.",
    )
    parser.add_argument(
        "--statewise-risk-shrink-form",
        choices=("squared_raw", "normalized_rms"),
        default="normalized_rms",
        help=(
            "Risk-shrink scale. normalized_rms divides per-state movement RMS "
            "by the registered max-step RMS; squared_raw is retained only to "
            "reproduce the first scale-check smoke."
        ),
    )
    parser.add_argument(
        "--pair-delta-enabled", action="store_true",
        help=(
            "Enable independent Twin antisymmetric same-state delta Critics. "
            "The absolute Twin Critics and their optimizers remain unchanged."
        ),
    )
    parser.add_argument("--pair-delta-pretrain-updates", type=int, default=1600)
    parser.add_argument("--pair-delta-learning-rate", type=float, default=1e-4)
    parser.add_argument("--pair-delta-weight", type=float, default=1.0)
    parser.add_argument(
        "--pair-delta-disagreement-weight", type=float, default=0.5,
        help=(
            "Conservative pair risk is max(delta1,delta2) plus this multiplier "
            "times Twin absolute disagreement."
        ),
    )
    parser.add_argument(
        "--actor-output-support-multiplier",
        type=float,
        choices=(1.0, OAC_DEFAULT_OUTPUT_SUPPORT_MULTIPLIER),
        default=OAC_DEFAULT_OUTPUT_SUPPORT_MULTIPLIER,
        help=(
            "Actor output support in MPPI std units. The production/default "
            "contract is +/-3std; +/-1std is retained only for an explicit A/B."
        ),
    )
    parser.add_argument(
        "--allow-box1-ablation",
        action="store_true",
        help="Explicitly authorize the legacy +/-1std support for an A/B ablation.",
    )
    parser.add_argument("--selection-gain-p05-floor", type=float)
    parser.add_argument("--selection-speed-2-4-gain-p05-floor", type=float)
    parser.add_argument("--selection-speed-2-8-gain-p05-floor", type=float)
    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def serialized_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def actor_learning_rate_for_round(
    args: argparse.Namespace | Any, local_round: int,
) -> float:
    """Return the registered Actor LR for a one-indexed local round."""
    if args.actor_learning_rate_schedule == "constant":
        return float(args.actor_learning_rate)
    initial = float(args.actor_learning_rate_initial)
    middle = float(args.actor_learning_rate_middle)
    final = float(args.actor_learning_rate)
    initial_rounds = int(args.actor_learning_rate_initial_rounds)
    middle_round = int(args.actor_learning_rate_middle_round)
    total_rounds = int(args.rounds)
    if local_round <= initial_rounds:
        return initial
    if local_round <= middle_round:
        progress = (local_round - initial_rounds) / (middle_round - initial_rounds)
        return middle + 0.5 * (initial - middle) * (1.0 + math.cos(math.pi * progress))
    progress = (local_round - middle_round) / (total_rounds - middle_round)
    return final + 0.5 * (middle - final) * (1.0 + math.cos(math.pi * progress))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_actor(
    path: Path, fold: int, seed: int, device: torch.device,
    support_multiplier: float | None = None,
):
    payload = torch.load(path, map_location=device)
    if int(payload["fold"]) != fold or int(payload["seed"]) != seed:
        raise AssertionError("Actor fold/seed mismatch")
    if support_multiplier is None:
        actor = DirectNoAnchorGTXActor(dropout=0.0).to(device)
        actor.load_state_dict(payload["model_state_dict"], strict=True)
    else:
        actor = DirectNoAnchorGTXSupportActor(
            support_multiplier=support_multiplier, dropout=0.0
        ).to(device)
        incompatible = actor.load_state_dict(payload["model_state_dict"], strict=False)
        expected_missing = {
            "output_support_multiplier",
            "support_adapter_head.weight", "support_adapter_head.bias",
            "support_adapter_skip.weight", "support_adapter_skip.bias",
        }
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise AssertionError(
                f"unexpected support Actor load result: {incompatible}"
            )
    actor.train()
    for parameter in actor.parameters():
        parameter.requires_grad_(True)
    return actor, payload


def load_value(
    model_path: Path, training_path: Path, device: torch.device,
) -> tuple[AbsoluteActionValueCritic, dict[str, Any], dict[str, Any]]:
    model_payload = torch.load(model_path, map_location=device)
    training_payload = torch.load(training_path, map_location=device)
    model = AbsoluteActionValueCritic(dropout=0.0).to(device)
    model.load_state_dict(model_payload["model"], strict=True)
    return model, training_payload, model_payload


def initialize_pair_delta_critic(
    value_model: AbsoluteActionValueCritic, device: torch.device,
) -> ConfigurableAbsoluteActionValueCritic:
    """Clone the absolute Critic representation and add a fresh pair head."""
    model = ConfigurableAbsoluteActionValueCritic(pair_delta=True).to(device)
    incompatible = model.load_state_dict(value_model.state_dict(), strict=False)
    expected_missing = {
        "pair_head.0.weight", "pair_head.0.bias",
        "pair_head.2.weight", "pair_head.2.bias",
        "pair_head.4.weight", "pair_head.4.bias",
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise AssertionError(f"unexpected Pairwise Critic initialization: {incompatible}")
    return model


def pair_delta_objective(
    model: ConfigurableAbsoluteActionValueCritic,
    payload: dict[str, Any], inputs: tuple[np.ndarray, ...], pairs,
    args: argparse.Namespace, device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    state, action, cost = pairs
    history = torch.from_numpy(inputs[0][state]).to(device)
    reference = torch.from_numpy(inputs[1][state]).to(device)
    current = torch.from_numpy(inputs[2][state]).to(device)
    action_tensor = torch.from_numpy(action).to(device)
    cost_tensor = torch.from_numpy(cost).to(device)
    predicted = model.pair_delta(
        history, reference, current, action_tensor[:, 0], action_tensor[:, 1]
    )
    target = (
        torch.log1p(cost_tensor[:, 0]) - torch.log1p(cost_tensor[:, 1])
    ) / float(payload["training"]["target_std"])
    delta = torch.nn.functional.smooth_l1_loss(predicted, target)
    raw = cost_tensor[:, 0] - cost_tensor[:, 1]
    ranking = torch.nn.functional.softplus(
        -raw.sign() * predicted / float(args.ranking_temperature)
    ).mean()
    loss = delta + float(args.ranking_weight) * ranking
    return loss, {
        "delta": float(delta.detach()),
        "ranking": float(ranking.detach()),
    }


def physical_pair_delta(
    model: ConfigurableAbsoluteActionValueCritic,
    payload: dict[str, Any], inputs: tuple[np.ndarray, ...],
    indices: np.ndarray, candidate: torch.Tensor, reference_action: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    prediction = model.pair_delta(
        torch.from_numpy(inputs[0][indices]).to(device),
        torch.from_numpy(inputs[1][indices]).to(device),
        torch.from_numpy(inputs[2][indices]).to(device),
        candidate, reference_action,
    )
    return prediction * float(payload["training"]["target_std"])


def pair_mean_comparison_accuracy(
    model1: ConfigurableAbsoluteActionValueCritic, payload1: dict[str, Any],
    model2: ConfigurableAbsoluteActionValueCritic, payload2: dict[str, Any],
    inputs: tuple[np.ndarray, ...], state: np.ndarray, action: np.ndarray,
    cost: np.ndarray, material_gap: float, disagreement_weight: float,
    device: torch.device,
) -> tuple[float, int]:
    """Accuracy for each exploration candidate relative to Actor mean row 0."""
    correct = []
    model1.eval(); model2.eval()
    with torch.no_grad():
        for candidate_index in range(1, action.shape[1]):
            candidate = torch.from_numpy(action[:, candidate_index]).to(device)
            baseline = torch.from_numpy(action[:, 0]).to(device)
            delta1 = physical_pair_delta(
                model1, payload1, inputs, state, candidate, baseline, device
            )
            delta2 = physical_pair_delta(
                model2, payload2, inputs, state, candidate, baseline, device
            )
            score = (
                torch.maximum(delta1, delta2)
                + float(disagreement_weight) * torch.abs(delta1 - delta2)
            ).cpu().numpy()
            truth = cost[:, candidate_index] - cost[:, 0]
            material = np.abs(truth) >= material_gap
            correct.extend((np.sign(score[material]) == np.sign(truth[material])).tolist())
    model1.train(); model2.train()
    return (float(np.mean(correct)) if correct else 0.0, len(correct))


def internal_split(
    data: dict[str, np.ndarray], folds: np.ndarray, outer_fold: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Take one complete episode per speed/scenario cell for selection."""
    outer_train = folds != outer_fold
    fit_episodes: list[str] = []
    selection_episodes: list[str] = []
    for speed in sorted(np.unique(data["speed"][outer_train])):
        for scenario in sorted(np.unique(data["scenario"][outer_train])):
            mask = (
                outer_train
                & np.isclose(data["speed"], speed)
                & (data["scenario"] == scenario)
            )
            episodes = sorted(np.unique(data["episode"][mask]).tolist())
            if len(episodes) < 2:
                raise AssertionError("internal split needs at least two episodes/cell")
            selection_episodes.append(str(episodes[-1]))
            fit_episodes.extend(str(value) for value in episodes[:-1])
    fit = np.flatnonzero(np.isin(data["episode"], fit_episodes) & outer_train)
    selection = np.flatnonzero(
        np.isin(data["episode"], selection_episodes) & outer_train
    )
    if len(set(fit_episodes) & set(selection_episodes)):
        raise AssertionError("internal fit/selection episode overlap")
    if len(fit) != 600 or len(selection) != 600:
        raise AssertionError(f"unexpected internal split sizes {len(fit)}/{len(selection)}")
    return fit, selection, sorted(fit_episodes), sorted(selection_episodes)


def filter_replay(
    replay: dict[str, np.ndarray], allowed_states: np.ndarray,
) -> dict[str, np.ndarray]:
    mask = np.isin(replay["state_index"], allowed_states)
    result = {key: np.asarray(value[mask]) for key, value in replay.items()}
    if not np.all(np.isin(result["state_index"], allowed_states)):
        raise AssertionError("Replay internal-selection leakage")
    return result


def append_replay(
    replay: dict[str, np.ndarray], state: np.ndarray, action: np.ndarray,
    cost: np.ndarray, round_index: int, group: np.ndarray, role: np.ndarray,
    pre1: np.ndarray, pre2: np.ndarray,
) -> dict[str, np.ndarray]:
    new = {
        "state_index": np.asarray(state, np.int64),
        "action": np.asarray(action, np.float32),
        "cost": np.asarray(cost, np.float32),
        "round": np.full(len(state), round_index, np.int16),
        "interaction_group": np.asarray(group, np.int32),
        "role": np.asarray(role),
        "pre_critic1": np.asarray(pre1, np.float32),
        "pre_critic2": np.asarray(pre2, np.float32),
    }
    return {
        key: np.concatenate((replay[key], new[key]))
        for key in new
    }


def actor_tensor(
    actor: nn.Module, actor_inputs: tuple[np.ndarray, ...], indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    tensors = tuple(
        torch.from_numpy(value[indices]).to(device) for value in actor_inputs
    )
    _, center = actor(*tensors)
    return center


def physical_value(
    model: AbsoluteActionValueCritic, payload: dict[str, Any],
    inputs: tuple[np.ndarray, ...], indices: np.ndarray,
    actions: torch.Tensor, device: torch.device,
) -> torch.Tensor:
    prediction = model(
        torch.from_numpy(inputs[0][indices]).to(device),
        torch.from_numpy(inputs[1][indices]).to(device),
        torch.from_numpy(inputs[2][indices]).to(device),
        actions,
    )
    return (
        prediction * float(payload["training"]["target_std"])
        + float(payload["training"]["target_mean"])
    )


def differentiable_dbm_cost(
    actions: torch.Tensor, indices: np.ndarray, states: np.ndarray,
    current: np.ndarray, references: np.ndarray,
    backend: TorchDynamicBicycleRolloutBackend,
    weights: TorchMPPICostWeights, params: TorchMPPIParams,
    device: torch.device,
) -> torch.Tensor:
    """Return deterministic J50 with an intact gradient to 8x2 knots."""
    full_actions = interpolate_knots(actions, params.horizon).unsqueeze(1)
    return batched_cost(
        backend, weights, full_actions,
        torch.from_numpy(states[indices]).to(device),
        torch.from_numpy(current[indices]).to(device),
        torch.from_numpy(references[indices]).to(device),
    )[:, 0]


def actor_exploration_bank(
    mean: np.ndarray, log_scale: torch.Tensor, rng: np.random.Generator,
    minimum: float, maximum: float,
) -> np.ndarray:
    scale = np.clip(
        np.exp(log_scale.detach().cpu().numpy()), minimum, maximum
    ).astype(np.float32)
    std = BASE_SIGMA.reshape(1, 1, 2) * scale.reshape(1, 8, 2)
    latent_mean = np.arctanh(np.clip(mean, -0.999, 0.999))
    eps0 = rng.normal(size=mean.shape).astype(np.float32)
    eps1 = rng.normal(size=mean.shape).astype(np.float32)
    wide = rng.normal(size=mean.shape).astype(np.float32)
    wide_std = np.minimum(2.0 * std, BASE_SIGMA.reshape(1, 1, 2) * maximum)
    return np.stack((
        mean,
        np.tanh(latent_mean - std * eps0),
        np.tanh(latent_mean + std * eps0),
        np.tanh(latent_mean - std * eps1),
        np.tanh(latent_mean + std * eps1),
        np.tanh(latent_mean + wide_std * wide),
    ), axis=1).astype(np.float32)


def actor_update(
    args: argparse.Namespace, actor: nn.Module, selected_actor: nn.Module,
    actor_optimizer: torch.optim.Optimizer, log_scale: nn.Parameter,
    log_alpha: nn.Parameter, temperature_optimizer: torch.optim.Optimizer,
    actor_inputs: tuple[np.ndarray, ...], critic_inputs: tuple[np.ndarray, ...],
    critic1: nn.Module, payload1: dict[str, Any], critic2: nn.Module,
    payload2: dict[str, Any], gap_head: ContinuousGapHead,
    data: dict[str, np.ndarray], fit: np.ndarray, rng: np.random.Generator,
    device: torch.device, tail_lagrange: float = 0.0,
    pair_critic1: ConfigurableAbsoluteActionValueCritic | None = None,
    pair_critic2: ConfigurableAbsoluteActionValueCritic | None = None,
    step_limit_sigma_rms: float | None = None,
    update_temperature: bool = True,
    dbm_context: tuple[
        np.ndarray, np.ndarray, np.ndarray,
        TorchDynamicBicycleRolloutBackend, TorchMPPICostWeights, TorchMPPIParams,
    ] | None = None,
) -> dict[str, float]:
    deterministic_center_objective = (
        args.actor_objective_mode == "deterministic_center_dbm"
    )
    if (pair_critic1 is None) != (pair_critic2 is None):
        raise ValueError("Twin Pairwise Critics must be both enabled or both disabled")
    if args.actor_gradient_source == "dbm" and dbm_context is None:
        raise ValueError("DBM Actor gradient source requires rollout context")
    if args.actor_gradient_source == "dbm" and pair_critic1 is not None:
        raise ValueError("matched DBM gradient-source pilot forbids Pair Critics")
    if deterministic_center_objective and args.actor_gradient_source != "dbm":
        raise ValueError("deterministic-center objective requires DBM gradients")
    index = stratified_contexts(data, fit, args.actor_batch_size, rng)
    before_state = {
        key: value.detach().clone() for key, value in actor.state_dict().items()
    }
    before_scale = log_scale.detach().clone()
    actor.train()
    mean = actor_tensor(actor, actor_inputs, index, device)
    with torch.no_grad():
        selected_mean = actor_tensor(
            selected_actor, actor_inputs, index, device
        )
        bank = torch.from_numpy(data["actions"][index]).to(device)
        q1_bank = physical_value(
            critic1, payload1, critic_inputs, index, bank, device
        )
        q2_bank = physical_value(
            critic2, payload2, critic_inputs, index, bank, device
        )
        reference_index = torch.maximum(q1_bank, q2_bank).argmin(dim=1)
        row = torch.arange(len(index), device=device)
        reference = bank[row, reference_index]
        pair = torch.stack((mean.detach(), reference), dim=1)
        pair1 = physical_value(
            critic1, payload1, critic_inputs, index, pair, device
        )
        pair2 = physical_value(
            critic2, payload2, critic_inputs, index, pair, device
        )
        coefficient = gap_head(
            pair1, pair2, mean.detach(), reference
        ).clamp(0.0, 1.0)
        selected_q1 = physical_value(
            critic1, payload1, critic_inputs, index,
            selected_mean[:, None], device,
        )[:, 0]
        selected_q2 = physical_value(
            critic2, payload2, critic_inputs, index,
            selected_mean[:, None], device,
        )[:, 0]
        selected_conservative = torch.maximum(selected_q1, selected_q2)

    sigma = torch.from_numpy(BASE_SIGMA).to(device).reshape(1, 1, 2)
    scale = log_scale.exp().reshape(1, 8, 2)
    std = sigma * scale
    latent_mean = torch.atanh(mean.clamp(-0.999, 0.999))
    epsilon = torch.randn_like(mean)
    latent_action = latent_mean + std * epsilon
    action = torch.tanh(latent_action)
    log_prob = (
        -0.5 * (epsilon.square() + 2.0 * std.log() + math.log(2.0 * math.pi))
        - torch.log(1.0 - action.square() + 1e-6)
    ).sum(dim=(1, 2))
    frozen_models = [critic1, critic2, gap_head]
    if pair_critic1 is not None:
        frozen_models.extend((pair_critic1, pair_critic2))
    for model in frozen_models:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    q1 = physical_value(
        critic1, payload1, critic_inputs, index, action[:, None], device
    )[:, 0]
    q2 = physical_value(
        critic2, payload2, critic_inputs, index, action[:, None], device
    )[:, 0]
    conservative = torch.maximum(q1, q2)
    mean_q1 = physical_value(
        critic1, payload1, critic_inputs, index, mean[:, None], device,
    )[:, 0]
    mean_q2 = physical_value(
        critic2, payload2, critic_inputs, index, mean[:, None], device,
    )[:, 0]
    mean_conservative = torch.maximum(mean_q1, mean_q2)
    objective_value = conservative
    objective_mean = mean_conservative
    objective_selected = selected_conservative
    dbm_action_cost = conservative.new_full((), float("nan"))
    dbm_mean_cost = conservative.new_full((), float("nan"))
    if args.actor_gradient_source == "dbm":
        assert dbm_context is not None
        states, current, references, backend, weights, params = dbm_context
        if deterministic_center_objective:
            # The sampled action remains available to the actor-visited Replay
            # collection outside this loss, but it must not provide an Actor
            # gradient in the deterministic deployment-center contract.
            with torch.no_grad():
                exact_action_cost = differentiable_dbm_cost(
                    action.detach(), index, states, current, references,
                    backend, weights, params, device,
                )
        else:
            exact_action_cost = differentiable_dbm_cost(
                action, index, states, current, references,
                backend, weights, params, device,
            )
        exact_mean_cost = differentiable_dbm_cost(
            mean, index, states, current, references,
            backend, weights, params, device,
        )
        with torch.no_grad():
            exact_selected_cost = differentiable_dbm_cost(
                selected_mean, index, states, current, references,
                backend, weights, params, device,
            )
        objective_value = torch.log1p(exact_action_cost.clamp_min(0.0))
        objective_mean = torch.log1p(exact_mean_cost.clamp_min(0.0))
        objective_selected = torch.log1p(exact_selected_cost.clamp_min(0.0))
        dbm_action_cost = exact_action_cost.mean()
        dbm_mean_cost = exact_mean_cost.mean()
    pair_delta_mean = mean_conservative.new_zeros(())
    pair_delta_p90 = mean_conservative.new_zeros(())
    pair_disagreement_mean = mean_conservative.new_zeros(())
    if pair_critic1 is not None:
        delta1 = physical_pair_delta(
            pair_critic1, payload1, critic_inputs, index,
            mean, selected_mean, device,
        )
        delta2 = physical_pair_delta(
            pair_critic2, payload2, critic_inputs, index,
            mean, selected_mean, device,
        )
        pair_disagreement = torch.abs(delta1 - delta2)
        conservative_pair_delta = (
            torch.maximum(delta1, delta2)
            + float(args.pair_delta_disagreement_weight) * pair_disagreement
        )
        raw_regression_excess = (
            conservative_pair_delta - float(args.tail_regression_margin_log)
        )
        pair_delta_mean = conservative_pair_delta.mean()
        pair_delta_p90 = torch.quantile(conservative_pair_delta, 0.90)
        pair_disagreement_mean = pair_disagreement.mean()
    else:
        raw_regression_excess = (
            objective_mean - objective_selected
            - float(args.tail_regression_margin_log)
        )
    predicted_regression = torch.relu(raw_regression_excess)
    statewise_risk_coefficient = torch.sigmoid(
        raw_regression_excess / float(args.statewise_risk_temperature_log)
    )
    tail_count = max(
        1, int(math.ceil(float(args.tail_cvar_fraction) * len(index)))
    )
    tail_cvar = torch.topk(predicted_regression, tail_count).values.mean()
    per_state_trust = (((mean - selected_mean) / sigma) ** 2).mean(dim=(1, 2))
    trust = per_state_trust.mean()
    effective_step_limit = (
        float(args.max_step_sigma_rms)
        if step_limit_sigma_rms is None else float(step_limit_sigma_rms)
    )
    if args.statewise_risk_shrink_form == "normalized_rms":
        shrink_distance = torch.sqrt(per_state_trust + 1e-12) / float(
            args.max_step_sigma_rms
        )
    else:
        shrink_distance = per_state_trust
    statewise_risk_shrink = (
        statewise_risk_coefficient.detach() * shrink_distance
    ).mean()
    temperature = log_alpha.exp().detach()
    if deterministic_center_objective:
        actor_cost_weight = torch.ones_like(objective_mean)
    elif args.actor_cost_weight_gamma == 0.0:
        # Preserve the registered baseline path bit-for-bit.
        actor_cost_weight = torch.ones_like(objective_value)
    else:
        actor_cost_weight = torch.exp(
            float(args.actor_cost_weight_gamma) * objective_value.detach()
        ).clamp(max=float(args.actor_cost_weight_maximum))
        # A positive batch scalar does not alter the objective direction.  This
        # normalization isolates cross-state weighting from the global Actor LR.
        actor_cost_weight = actor_cost_weight / actor_cost_weight.mean().clamp_min(1e-12)
    if deterministic_center_objective:
        # This is intentionally the deployment quantity itself: one
        # deterministic Actor center evaluated by raw DBM J50.  No learned
        # move coefficient, sampled-action value, or transformed cost can
        # alter its parameter-gradient direction.
        value_term = exact_mean_cost.mean()
        entropy_term = log_prob.mean() * 0.0
    else:
        value_term = (
            coefficient.detach() * actor_cost_weight * objective_value
        ).mean()
        entropy_term = (temperature * log_prob).mean()
    tail_weight = (
        float(tail_lagrange)
        if args.tail_constraint_mode == "adaptive"
        else float(args.tail_regression_weight)
    )
    tail_constraint = tail_cvar - float(args.tail_cvar_budget_log)
    if deterministic_center_objective:
        effective_tail_weight = 0.0
        loss = value_term + args.trust_weight * trust
    else:
        effective_tail_weight = tail_weight
        loss = (
            value_term + entropy_term + args.trust_weight * trust
            + tail_weight * tail_constraint
            + args.statewise_risk_shrink_weight * statewise_risk_shrink
        )

    actor_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(
        list(actor.parameters()) + [log_scale], 1.0
    ))
    actor_optimizer.step()
    with torch.no_grad():
        log_scale.clamp_(
            math.log(args.minimum_exploration_scale),
            math.log(args.maximum_exploration_scale),
        )
    for model in frozen_models:
        for parameter in model.parameters():
            parameter.requires_grad_(True)

    temperature_loss = -(
        log_alpha * (log_prob.detach() + args.target_entropy)
    ).mean()
    if update_temperature and not deterministic_center_objective:
        temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        temperature_optimizer.step()

    with torch.no_grad():
        after = actor_tensor(actor, actor_inputs, index, device)
        step_rms = float(torch.sqrt((((after - mean) / sigma) ** 2).mean()))
        projection = min(1.0, effective_step_limit / max(step_rms, 1e-12))
        if projection < 1.0:
            for name, value in actor.state_dict().items():
                value.copy_(before_state[name] + projection * (value - before_state[name]))
            log_scale.copy_(before_scale + projection * (log_scale - before_scale))
            after = actor_tensor(actor, actor_inputs, index, device)
            step_rms = float(torch.sqrt((((after - mean) / sigma) ** 2).mean()))
    return {
        "loss": float(loss.detach()),
        "value_term": float(value_term.detach()),
        "actor_cost_weight_gamma": float(args.actor_cost_weight_gamma),
        "actor_cost_weight_mean": float(actor_cost_weight.mean().detach()),
        "actor_cost_weight_p50": float(
            torch.quantile(actor_cost_weight, 0.50).detach()
        ),
        "actor_cost_weight_p90": float(
            torch.quantile(actor_cost_weight, 0.90).detach()
        ),
        "actor_cost_weight_max": float(actor_cost_weight.max().detach()),
        "actor_cost_weight_ess_fraction": float(
            (
                actor_cost_weight.sum().square()
                / (actor_cost_weight.square().sum().clamp_min(1e-12) * len(index))
            ).detach()
        ),
        "actor_cost_weight_cap_fraction": float(
            0.0 if deterministic_center_objective else
            (
                torch.exp(
                    float(args.actor_cost_weight_gamma) * objective_value.detach()
                ) >= float(args.actor_cost_weight_maximum)
            ).float().mean().detach()
        ),
        "entropy_term": float(entropy_term.detach()),
        "trust": float(trust.detach()),
        "statewise_risk_shrink": float(statewise_risk_shrink.detach()),
        "statewise_risk_coefficient_mean": float(
            statewise_risk_coefficient.mean().detach()
        ),
        "statewise_risk_coefficient_p90": float(
            torch.quantile(statewise_risk_coefficient, 0.90).detach()
        ),
        "tail_regression_cvar": float(tail_cvar.detach()),
        "tail_constraint_violation": float(tail_constraint.detach()),
        "tail_lagrange_used": effective_tail_weight,
        "tail_regression_active_fraction": float(
            (predicted_regression > 0).float().mean().detach()
        ),
        "tail_regression_log_gap_median": float(
            torch.median(objective_mean - objective_selected).detach()
        ),
        "tail_regression_log_gap_p90": float(
            torch.quantile(objective_mean - objective_selected, 0.90).detach()
        ),
        "actor_gradient_source": str(args.actor_gradient_source),
        "actor_objective_mode": str(args.actor_objective_mode),
        "dbm_sampled_action_cost_mean": float(dbm_action_cost.detach()),
        "dbm_mean_action_cost_mean": float(dbm_mean_cost.detach()),
        "tail_risk_source": (
            "monitor_only_dbm_mean_vs_selected"
            if deterministic_center_objective else
            ("twin_pair_delta" if pair_critic1 is not None else "twin_absolute_difference")
        ),
        "pair_delta_conservative_mean": float(pair_delta_mean.detach()),
        "pair_delta_conservative_p90": float(pair_delta_p90.detach()),
        "pair_delta_disagreement_mean": float(pair_disagreement_mean.detach()),
        "temperature": float(log_alpha.exp().detach()),
        "temperature_loss": float(temperature_loss.detach()),
        "coefficient_mean": float(coefficient.mean()),
        "coefficient_p10": float(torch.quantile(coefficient, 0.10)),
        "coefficient_p90": float(torch.quantile(coefficient, 0.90)),
        "gradient_norm_before_clip": gradient_norm,
        "step_sigma_rms": step_rms,
        "step_limit_sigma_rms": effective_step_limit,
        "trust_projection": projection,
        "temperature_updated": bool(
            update_temperature and not deterministic_center_objective
        ),
        "tail_affects_actor_loss": bool(not deterministic_center_objective),
        "move_coefficient_affects_actor_loss": bool(
            not deterministic_center_objective
        ),
        "effective_tail_weight": float(effective_tail_weight),
        "exploration_scale_mean": float(log_scale.exp().mean()),
    }


def project_actor_to_round_trust(
    actor: nn.Module, log_scale: nn.Parameter,
    round_start_state: dict[str, torch.Tensor], round_start_scale: torch.Tensor,
    round_start_action: torch.Tensor, actor_inputs: tuple[np.ndarray, ...],
    indices: np.ndarray, device: torch.device, limit_sigma_rms: float,
) -> tuple[float, float]:
    """Project the cumulative round displacement, preserving legacy K=1 logic."""
    sigma = torch.from_numpy(BASE_SIGMA).to(device).reshape(1, 1, 2)
    with torch.no_grad():
        after = actor_tensor(actor, actor_inputs, indices, device)
        cumulative_rms = float(torch.sqrt(
            (((after - round_start_action) / sigma) ** 2).mean()
        ))
        projection = min(1.0, float(limit_sigma_rms) / max(cumulative_rms, 1e-12))
        if projection < 1.0:
            for name, value in actor.state_dict().items():
                start = round_start_state[name]
                value.copy_(start + projection * (value - start))
            log_scale.copy_(
                round_start_scale + projection * (log_scale - round_start_scale)
            )
            after = actor_tensor(actor, actor_inputs, indices, device)
            cumulative_rms = float(torch.sqrt(
                (((after - round_start_action) / sigma) ** 2).mean()
            ))
    return cumulative_rms, projection


def bootstrap_mean_ci(
    gain: np.ndarray, episodes: np.ndarray, samples: int, seed: int,
) -> list[float]:
    unique = np.unique(episodes)
    by_episode = [gain[episodes == value] for value in unique]
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, np.float64)
    for sample in range(samples):
        choice = rng.integers(len(unique), size=len(unique))
        boot[sample] = np.mean(np.concatenate([by_episode[index] for index in choice]))
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def evaluate_actor(
    args: argparse.Namespace, actor: nn.Module,
    actor_inputs: tuple[np.ndarray, ...], indices: np.ndarray,
    initial_cost: np.ndarray | None, data: dict[str, np.ndarray],
    states: np.ndarray, current: np.ndarray, references: np.ndarray,
    backend: TorchDynamicBicycleRolloutBackend, weights: TorchMPPICostWeights,
    params: TorchMPPIParams, device: torch.device, bootstrap_seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    action = actor_mean(actor, actor_inputs, indices, device)
    cost = rollout_bank(
        backend, weights, params, action[:, None], states, current,
        references, indices, args.rollout_batch_size, device,
    )[:, 0]
    if initial_cost is None:
        initial_cost = cost.copy()
    gain = initial_cost - cost
    warm = data["costs"][indices, 0]
    bank_best = data["costs"][indices].min(axis=1)
    headroom = float(
        np.sum(initial_cost - cost)
        / max(float(np.sum(initial_cost - bank_best)), 1e-9)
    )
    guarded_cost = np.minimum(warm, cost)
    by_speed = {}
    for speed in sorted(np.unique(data["speed"][indices])):
        mask = np.isclose(data["speed"][indices], speed)
        by_speed[f"{speed:.1f}"] = {
            "mean_gain": float(np.mean(gain[mask])),
            "median_gain": float(np.median(gain[mask])),
            "p05_gain": float(np.quantile(gain[mask], 0.05)),
            "mean_cost": float(np.mean(cost[mask])),
        }
    by_scenario = {}
    for scenario in sorted(np.unique(data["scenario"][indices])):
        mask = data["scenario"][indices] == scenario
        by_scenario[str(scenario)] = {
            "mean_gain": float(np.mean(gain[mask])),
            "median_gain": float(np.median(gain[mask])),
        }
    result = {
        "count": int(len(indices)),
        "cost": distribution(cost),
        "gain_vs_initial": distribution(gain),
        "gain_episode_bootstrap_ci95": bootstrap_mean_ci(
            gain, data["episode"][indices], args.bootstrap_samples, bootstrap_seed
        ),
        "headroom_recovery_vs_bank_best": headroom,
        "regression_fraction": float(np.mean(gain < -1e-6)),
        "action_saturation_fraction": float(np.mean(np.abs(action) >= 0.999)),
        "state_any_saturation_fraction": float(np.mean(
            np.any(np.abs(action) >= 0.999, axis=(1, 2))
        )),
        "mean_adjacent_acceleration_delta": float(np.mean(np.abs(np.diff(action[:, :, 0], axis=1)))),
        "mean_adjacent_steering_delta": float(np.mean(np.abs(np.diff(action[:, :, 1], axis=1)))),
        "by_speed": by_speed,
        "by_scenario": by_scenario,
        "two_center_guard": {
            "cost": distribution(guarded_cost),
            "gain_vs_warm": distribution(warm - guarded_cost),
            "warm_selected_fraction": float(np.mean(warm <= cost)),
        },
        "action_sha256": __import__("hashlib").sha256(action.tobytes()).hexdigest(),
        "cost_sha256": __import__("hashlib").sha256(cost.tobytes()).hexdigest(),
    }
    return result, cost


def acceptance_gate(
    candidate: dict[str, Any], candidate_cost: np.ndarray,
    selected_cost: np.ndarray, data: dict[str, np.ndarray], indices: np.ndarray,
    args: argparse.Namespace | None = None,
) -> dict[str, bool]:
    gain = selected_cost - candidate_cost
    speed = data["speed"][indices]
    result = {
        "mean_gain_vs_selected_nonnegative": float(np.mean(gain)) >= 0.0,
        "median_gain_vs_selected_nonnegative": float(np.median(gain)) >= 0.0,
        "speed_2_4_gain_nonnegative": float(np.mean(gain[np.isclose(speed, 2.4)])) >= 0.0,
        "speed_2_8_gain_nonnegative": float(np.mean(gain[np.isclose(speed, 2.8)])) >= 0.0,
        "state_saturation_le_0_10": candidate["state_any_saturation_fraction"] <= 0.10,
        "finite": bool(np.all(np.isfinite(candidate_cost))),
    }
    if args is not None and args.selection_gain_p05_floor is not None:
        result["gain_p05_above_floor"] = (
            candidate["gain_vs_initial"]["p05"]
            >= float(args.selection_gain_p05_floor)
        )
    if args is not None and args.selection_speed_2_4_gain_p05_floor is not None:
        result["speed_2_4_gain_p05_above_floor"] = (
            candidate["by_speed"]["2.4"]["p05_gain"]
            >= float(args.selection_speed_2_4_gain_p05_floor)
        )
    if args is not None and args.selection_speed_2_8_gain_p05_floor is not None:
        result["speed_2_8_gain_p05_above_floor"] = (
            candidate["by_speed"]["2.8"]["p05_gain"]
            >= float(args.selection_speed_2_8_gain_p05_floor)
        )
    return result


def save_seed_checkpoint(
    seed_dir: Path, actor: nn.Module, selected_actor: nn.Module,
    shadow_state: dict[str, torch.Tensor], critic1: nn.Module, payload1: dict,
    critic2: nn.Module, payload2: dict, gap_head: nn.Module,
    actor_optimizer, optimizer1, optimizer2, gap_optimizer,
    log_scale: nn.Parameter, log_alpha: nn.Parameter, temperature_optimizer,
    actor_updates: int, critic_updates: int, selected_round: int,
    tail_lagrange: float, tail_cvar_ema: float | None,
    pair_critic1: ConfigurableAbsoluteActionValueCritic | None = None,
    pair_critic2: ConfigurableAbsoluteActionValueCritic | None = None,
    pair_optimizer1: torch.optim.Optimizer | None = None,
    pair_optimizer2: torch.optim.Optimizer | None = None,
) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "actor_update_count": actor_updates,
        "critic_additional_update_count": critic_updates,
        "selected_round": selected_round,
        "formal_validation_loaded": False,
        "test_loaded": False,
        "tail_lagrange": float(tail_lagrange),
        "tail_cvar_ema": (
            None if tail_cvar_ema is None else float(tail_cvar_ema)
        ),
    }
    torch.save({
        **common, "model_state_dict": actor.state_dict(),
        "optimizer": actor_optimizer.state_dict(),
        "log_exploration_scale": log_scale.detach().cpu(),
        "temperature_log_alpha": log_alpha.detach().cpu(),
        "temperature_optimizer": temperature_optimizer.state_dict(),
    }, seed_dir / "actor_latest.pt")
    torch.save({
        **common, "model_state_dict": selected_actor.state_dict(),
    }, seed_dir / "actor_selected.pt")
    torch.save({
        **common, "model_state_dict": shadow_state,
    }, seed_dir / "actor_shadow.pt")
    torch.save({
        **common, "model": critic1.state_dict(), "training": payload1["training"],
        "optimizer": optimizer1.state_dict(),
    }, seed_dir / "critic1.pt")
    torch.save({
        **common, "model": critic2.state_dict(), "training": payload2["training"],
        "optimizer": optimizer2.state_dict(),
    }, seed_dir / "critic2.pt")
    torch.save({
        **common, "model": gap_head.state_dict(), "optimizer": gap_optimizer.state_dict(),
        "output_mode": "move_coefficient",
    }, seed_dir / "move_coefficient_head.pt")
    if pair_critic1 is not None:
        if pair_critic2 is None or pair_optimizer1 is None or pair_optimizer2 is None:
            raise ValueError("incomplete Twin Pairwise checkpoint state")
        torch.save({
            **common,
            "model_class": "ConfigurableAbsoluteActionValueCriticPairDelta",
            "model": pair_critic1.state_dict(),
            "model_config": pair_critic1.config,
            "training": payload1["training"],
            "optimizer": pair_optimizer1.state_dict(),
        }, seed_dir / "pair_critic1.pt")
        torch.save({
            **common,
            "model_class": "ConfigurableAbsoluteActionValueCriticPairDelta",
            "model": pair_critic2.state_dict(),
            "model_config": pair_critic2.config,
            "training": payload2["training"],
            "optimizer": pair_optimizer2.state_dict(),
        }, seed_dir / "pair_critic2.pt")


def main() -> None:
    args = parse_args()
    args.target_mode = "move_coefficient"
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.actor_gradient_source == "dbm":
        if not args.matched_gradient_source_pilot:
            raise ValueError("DBM Actor gradients require --matched-gradient-source-pilot")
        if not args.multi_actor_update_pilot:
            raise ValueError("matched gradient-source pilot requires multi-Actor mode")
        if args.matched_gradient_source_smoke:
            if args.rounds not in {1, 2} or args.evaluation_interval != 1:
                raise ValueError("matched gradient-source smoke requires one/two rounds, eval every round")
        else:
            if args.actor_updates_per_round != 16 or args.critic_updates_per_round != 20:
                raise ValueError("matched gradient-source pilot requires K=16 and 20 Critic updates")
            if args.rounds != 90 or args.evaluation_interval != 10:
                raise ValueError("matched gradient-source pilot requires 90 rounds, eval every 10")
        if args.actor_cost_weight_gamma != 1.0:
            raise ValueError("matched gradient-source pilot requires gamma=1 raw-cost aggregation")
        if args.pair_delta_enabled:
            raise ValueError("matched gradient-source pilot forbids Pair Critics")
    elif args.matched_gradient_source_pilot:
        raise ValueError("matched gradient-source pilot flag is only valid for DBM gradients")
    elif args.matched_gradient_source_smoke:
        raise ValueError("matched gradient-source smoke is only valid for DBM gradients")
    if args.actor_objective_mode == "deterministic_center_dbm":
        if args.actor_gradient_source != "dbm":
            raise ValueError(
                "deterministic-center objective requires --actor-gradient-source dbm"
            )
        if not args.matched_gradient_source_pilot:
            raise ValueError(
                "deterministic-center objective requires the matched pilot contract"
            )
        if args.pair_delta_enabled:
            raise ValueError("deterministic-center objective forbids Pair Critics")
    if args.actor_visited_refresh_pilot:
        if not args.multi_actor_update_pilot:
            raise ValueError("Actor-visited refresh pilot requires multi-Actor mode")
        if (args.critic_updates_per_round, args.actor_updates_per_round) != (10, 8):
            raise ValueError("refresh pilot requires 10 Critic and 8 Actor updates per round")
        if args.aux_update_interval_rounds != 2:
            raise ValueError("refresh pilot requires auxiliary updates every two rounds")
    elif args.critic_updates_per_round != 20:
        raise ValueError("OAC-2 and the microstep pilot require 20 Critic updates per round")
    if args.multi_actor_update_pilot:
        if args.actor_updates_per_round not in {1, 4, 8, 16, 20, 32}:
            raise ValueError("multi-Actor pilot requires K in {1, 4, 8, 16, 20, 32}")
    elif args.actor_updates_per_round != 1:
        raise ValueError(
            "legacy OAC-2 freezes the Actor:Critic update ratio at 1:20; "
            "use --multi-actor-update-pilot for the registered K=1/4/8 study"
        )
    if args.aux_update_interval_rounds <= 0:
        raise ValueError("auxiliary update interval must be positive")
    if args.rounds % args.aux_update_interval_rounds != 0:
        raise ValueError("round count must be divisible by auxiliary update interval")
    if args.gap_weight != 0.05:
        raise ValueError("OAC-2 requires the validated 0.05 coefficient weight")
    if args.tail_regression_weight < 0:
        raise ValueError("tail regression weight must be nonnegative")
    if args.statewise_risk_shrink_weight < 0:
        raise ValueError("statewise risk shrink weight must be nonnegative")
    if args.statewise_risk_temperature_log <= 0:
        raise ValueError("statewise risk temperature must be positive")
    if args.actor_cost_weight_gamma < 0 or args.actor_cost_weight_gamma > 1:
        raise ValueError("Actor cost-weight gamma must be within [0, 1]")
    if args.actor_cost_weight_maximum <= 1:
        raise ValueError("Actor cost-weight maximum must exceed one")
    if args.actor_learning_rate <= 0:
        raise ValueError("Actor learning rate must be positive")
    if args.actor_learning_rate_schedule == "staged_cosine":
        if args.actor_learning_rate_initial is None or args.actor_learning_rate_middle is None:
            raise ValueError("staged_cosine requires initial and middle Actor LRs")
        if not (
            args.actor_learning_rate_initial >= args.actor_learning_rate_middle
            >= args.actor_learning_rate > 0
        ):
            raise ValueError("staged Actor LRs must satisfy initial >= middle >= final > 0")
        if not (
            1 <= args.actor_learning_rate_initial_rounds
            < args.actor_learning_rate_middle_round < args.rounds
        ):
            raise ValueError(
                "staged Actor LR rounds must satisfy 1 <= initial_rounds "
                "< middle_round < total rounds"
            )
    if args.pair_delta_enabled:
        if args.tail_constraint_mode != "adaptive":
            raise ValueError("Pairwise OAC pilot requires the adaptive tail contract")
        if args.pair_delta_pretrain_updates <= 0:
            raise ValueError("Pairwise OAC requires positive Actor-frozen pretraining")
        if args.pair_delta_learning_rate <= 0 or args.pair_delta_weight <= 0:
            raise ValueError("Pairwise learning rate/weight must be positive")
        if args.pair_delta_disagreement_weight < 0:
            raise ValueError("Pairwise disagreement weight must be nonnegative")
    if args.tail_regression_margin_log < 0 or args.tail_cvar_budget_log < 0:
        raise ValueError("tail margin and budget must be nonnegative")
    if args.tail_constraint_mode == "adaptive":
        if args.tail_regression_weight != 0:
            raise ValueError("adaptive tail mode forbids a fixed tail weight")
        if args.tail_regression_margin_log <= 0:
            raise ValueError("adaptive tail mode requires a nonzero material margin")
        if args.tail_dual_learning_rate <= 0 or args.tail_dual_maximum <= 0:
            raise ValueError("adaptive tail dual learning rate/maximum must be positive")
        if not 0 <= args.tail_dual_initial <= args.tail_dual_maximum:
            raise ValueError("adaptive tail dual initial value is out of bounds")
        if not 0 <= args.tail_dual_ema_decay < 1:
            raise ValueError("adaptive tail EMA decay must be within [0, 1)")
    if (
        args.actor_output_support_multiplier
        != OAC_DEFAULT_OUTPUT_SUPPORT_MULTIPLIER
        and not args.allow_box1_ablation
    ):
        raise ValueError(
            "The OAC default output contract is +/-3std. Use "
            "--allow-box1-ablation only for an explicitly named legacy box1 A/B."
        )
    if not 0 < args.tail_cvar_fraction <= 1:
        raise ValueError("tail CVaR fraction must be within (0, 1]")
    if args.rounds % args.evaluation_interval != 0:
        raise ValueError("registered OAC-2 requires the final round to be evaluated")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    data = load_bank(args.bank_root)
    folds = make_folds(data, 3)
    parent_contract = json.loads((args.parent_run / "contract.json").read_text())
    parent_summary = json.loads((args.parent_run / "summary.json").read_text())
    parent_validator = json.loads((args.parent_run / "validator_report.json").read_text())
    outer_fold = int(parent_contract.get("outer_fold", 0))
    if parent_validator["qualification"] != "JOINT_MOVE_COEFFICIENT_VALIDATION_PASS":
        raise AssertionError("OAC-2 parent mechanism gate did not pass")
    fit, selection, fit_episodes, selection_episodes = internal_split(
        data, folds, outer_fold
    )
    states, current, references, params_json, weights_json, dbm_json = (
        load_rollout_inputs(data, args.gt_v1)
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    normalization, _ = load_actor_normalization(args.base_ac)
    actor_inputs = make_actor_inputs(data, normalization)
    source_fixed = Path(parent_contract["arguments"]["parent_run"])
    fixed_contract = json.loads((source_fixed / "contract.json").read_text())
    source_oac1 = Path(parent_contract.get(
        "source_oac1_run", fixed_contract["parent_run"]
    ))
    contract = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "OAC2_CONTINUOUS_CONTEXTUAL_BANDIT_CONTRACT",
        "arguments": serialized_args(args),
        "outer_fold": outer_fold,
        "fit_episodes": fit_episodes,
        "internal_selection_episodes": selection_episodes,
        "fit_state_count": int(len(fit)),
        "internal_selection_state_count": int(len(selection)),
        "parent_run": str(args.parent_run.resolve()),
        "parent_summary_sha256": sha256_file(args.parent_run / "summary.json"),
        "parent_validator_sha256": sha256_file(args.parent_run / "validator_report.json"),
        "candidate_bank_sha256": sha256_file(args.bank_root / "candidate_bank.npz"),
        "source_oac1_run": str(source_oac1.resolve()),
        "source_fixed_run": str(source_fixed.resolve()),
        "actor_update_contract": (
            (
                f"twenty joint Twin-Value/coefficient updates interleaved with "
                f"K={args.actor_updates_per_round} tanh-Gaussian Actor microsteps; "
                "scheduled LR divided by K; per-microstep trust divided by K; "
                "per-round cumulative action-space trust fixed at "
                f"{args.max_step_sigma_rms} sigma RMS; temperature and tail dual "
                f"updated every {args.aux_update_interval_rounds} outer round(s); "
                "independently partitioned RNG streams; "
            )
            if args.multi_actor_update_pilot else
            "one tanh-Gaussian reparameterized mean-policy update after twenty "
            "joint Twin-Value/coefficient updates; coefficient is continuous; "
            f"post-step action-space trust projection is fixed at "
            f"{args.max_step_sigma_rms} sigma RMS; "
        ) + f"output-support multiplier={args.actor_output_support_multiplier}",
        "multi_actor_update_contract": {
            "enabled": bool(args.multi_actor_update_pilot),
            "actor_updates_per_round": int(args.actor_updates_per_round),
            "critic_updates_per_round": int(args.critic_updates_per_round),
            "critic_partition": "balanced integer partition before each Actor microstep",
            "scheduled_lr_divisor": (
                int(args.actor_updates_per_round)
                if args.multi_actor_update_pilot else 1
            ),
            "per_microstep_trust_sigma_rms": float(
                args.max_step_sigma_rms / (
                    args.actor_updates_per_round
                    if args.multi_actor_update_pilot else 1
                )
            ),
            "per_round_cumulative_trust_sigma_rms": float(args.max_step_sigma_rms),
            "cumulative_trust_context": "all fit contexts",
            "aux_update_interval_rounds": int(args.aux_update_interval_rounds),
            "temperature_update_count": int(
                args.rounds // args.aux_update_interval_rounds
            ),
            "tail_dual_update_count": int(
                args.rounds // args.aux_update_interval_rounds
            ),
            "warm_used_in_training": False,
        },
        "actor_visited_refresh_contract": {
            "enabled": bool(args.actor_visited_refresh_pilot),
            "contexts_per_round": int(args.contexts_per_round),
            "rounds": int(args.rounds),
            "total_context_visits": int(args.contexts_per_round * args.rounds),
            "total_critic_updates": int(
                args.critic_updates_per_round * args.rounds
            ),
            "total_actor_updates": int(
                args.actor_updates_per_round * args.rounds
            ),
            "aux_update_interval_rounds": int(args.aux_update_interval_rounds),
        },
        "actor_gradient_source_contract": {
            "source": str(args.actor_gradient_source),
            "objective_mode": str(args.actor_objective_mode),
            "matched_pilot": bool(args.matched_gradient_source_pilot),
            "smoke_only": bool(args.matched_gradient_source_smoke),
            "critic_arm": (
                "detached exp(q) weighting times Twin log1p(J50) value gradient"
            ),
            "dbm_arm": (
                "the same detached weighting form evaluated with differentiable "
                "deterministic DBM log1p(J50); mean/selected tail costs also DBM"
            ),
            "critic_and_replay_continue_in_dbm_arm": True,
            "coefficient_is_detached_in_both_arms": True,
            "deterministic_center_dbm": {
                "actor_value_term": "mean deterministic raw DBM J50 at Actor mean",
                "sampled_action_affects_actor_loss": False,
                "move_coefficient_affects_actor_loss": False,
                "entropy_affects_actor_loss": False,
                "tail_affects_actor_loss": False,
                "trust_affects_actor_loss": True,
                "replay_exploration_continues": True,
                "twin_critic_training_continues": True,
            },
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
        "actor_cost_aggregation_contract": {
            "critic_target": "log1p(J50)",
            "gamma": float(args.actor_cost_weight_gamma),
            "detached_state_weight": "clip(exp(gamma*q_conservative), max=cap)",
            "weight_cap": float(args.actor_cost_weight_maximum),
            "batch_normalization": "divide detached weights by batch mean",
            "gamma_zero": "registered mean-log baseline",
            "gamma_one": (
                "mean-raw-J parameter-gradient direction while the cap is inactive"
            ),
        },
        "actor_learning_rate_contract": {
            "schedule": args.actor_learning_rate_schedule,
            "initial": (
                float(args.actor_learning_rate_initial)
                if args.actor_learning_rate_initial is not None else
                float(args.actor_learning_rate)
            ),
            "middle": (
                float(args.actor_learning_rate_middle)
                if args.actor_learning_rate_middle is not None else
                float(args.actor_learning_rate)
            ),
            "final": float(args.actor_learning_rate),
            "initial_rounds": int(args.actor_learning_rate_initial_rounds),
            "middle_round": int(args.actor_learning_rate_middle_round),
            "total_rounds": int(args.rounds),
            "interpolation": (
                "piecewise cosine" if args.actor_learning_rate_schedule == "staged_cosine"
                else "constant"
            ),
            "action_space_projection_sigma_rms": float(args.max_step_sigma_rms),
            "microstep_divisor": (
                int(args.actor_updates_per_round)
                if args.multi_actor_update_pilot else 1
            ),
        },
        "output_support_contract": {
            "units": "per-channel MPPI sampling standard deviation",
            "default_multiplier": OAC_DEFAULT_OUTPUT_SUPPORT_MULTIPLIER,
            "active_multiplier": float(args.actor_output_support_multiplier),
            "box1_requires_explicit_ablation": True,
            "box1_ablation_authorized": bool(args.allow_box1_ablation),
            "full_physical_support_is_not_default": True,
            "checkpoint_multiplier_buffer_required": True,
        },
        "tail_constraint_contract": {
            "mode": args.tail_constraint_mode,
            "quantity": (
                "top-tail mean relu(conservative_twin_pair_delta - "
                "material_margin_log)"
                if args.pair_delta_enabled else
                "top-tail mean relu(log1p_J_actor_mean - "
                "log1p_J_selected_center - material_margin_log)"
            ),
            "fraction": float(args.tail_cvar_fraction),
            "material_margin_log": float(args.tail_regression_margin_log),
            "budget_log": float(args.tail_cvar_budget_log),
            "fixed_weight": float(args.tail_regression_weight),
            "dual_learning_rate": float(args.tail_dual_learning_rate),
            "dual_initial": float(args.tail_dual_initial),
            "dual_maximum": float(args.tail_dual_maximum),
            "dual_ema_decay": float(args.tail_dual_ema_decay),
        },
        "statewise_risk_shrink_contract": {
            "enabled": bool(args.statewise_risk_shrink_weight > 0),
            "weight": float(args.statewise_risk_shrink_weight),
            "temperature_log": float(args.statewise_risk_temperature_log),
            "form": args.statewise_risk_shrink_form,
            "coefficient": (
                "sigmoid((conservative_twin_log1p_J_actor_mean - "
                "conservative_twin_log1p_J_selected - material_margin_log) / "
                "temperature_log)"
            ),
            "penalty": (
                "mean(stopgrad(coefficient) * "
                "rms16((actor-selected)/sigma)/max_step_sigma_rms)"
                if args.statewise_risk_shrink_form == "normalized_rms"
                else "mean(stopgrad(coefficient) * mean16(((actor-selected)/sigma)^2))"
            ),
            "deployment": "distilled into Actor parameters; no online Critic",
        },
        "pair_delta_contract": {
            "enabled": bool(args.pair_delta_enabled),
            "model": (
                "independent Twin ConfigurableAbsoluteActionValueCritic pair heads; "
                "absolute Twin Value models and optimizer states unchanged"
            ),
            "target": (
                "log1p(J_candidate)-log1p(J_reference), standardized only by "
                "the corresponding absolute Critic target std"
            ),
            "antisymmetry": "0.5*(f(a,b)-f(b,a)); delta(a,a)=0 by construction",
            "pretrain_updates": int(args.pair_delta_pretrain_updates),
            "online_updates_per_round": int(args.critic_updates_per_round),
            "loss_weight": float(args.pair_delta_weight),
            "risk": (
                "max(delta1,delta2) + disagreement_weight*abs(delta1-delta2)"
            ),
            "disagreement_weight": float(args.pair_delta_disagreement_weight),
            "actor_use": (
                "replaces only the adaptive-tail risk signal; absolute Twin Value "
                "still provides the main Actor value gradient"
            ),
            "deployment": "training-only; deployed Actor still one-shot with two-center guard",
        },
        "selection_contract": (
            "deterministic DBM on one held-out train episode per speed/scenario; "
            "selected advances only when mean, median, 2.4/2.8 m/s, saturation, "
            "and any explicitly configured direct-tail floors pass"
        ),
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.output_dir / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")

    all_records = []
    for seed in [int(value) for value in args.seeds.split(",")]:
        set_seed(26082400 + seed)
        rng = np.random.default_rng(26082400 + seed)
        if args.multi_actor_update_pilot:
            interaction_rng = rng
            critic_rng = np.random.default_rng(26082600 + seed)
            actor_rng = np.random.default_rng(26082700 + seed)
        else:
            # Preserve completed legacy runs bit-for-bit.
            interaction_rng = critic_rng = actor_rng = rng
        actor_path = args.actor_root / f"a0_fold{outer_fold}_seed{seed}.pt"
        actor, _ = load_actor(
            actor_path, outer_fold, seed, device,
            args.actor_output_support_multiplier,
        )
        initial_actor_hash = module_digest(actor)
        selected_actor = copy.deepcopy(actor).to(device).eval()
        for parameter in selected_actor.parameters():
            parameter.requires_grad_(False)
        joint_dir = args.parent_run / f"seed_{seed}" / "updates_3200"
        fixed_dir = source_fixed / f"seed_{seed}" / "updates_6400"
        critic1, payload1, joint1 = load_value(
            joint_dir / "critic1.pt", fixed_dir / "critic1.pt", device
        )
        critic2, payload2, joint2 = load_value(
            joint_dir / "critic2.pt", fixed_dir / "critic2.pt", device
        )
        critic_inputs = critic_state_inputs(data, payload1)
        gap_payload = torch.load(joint_dir / "gap_head.pt", map_location=device)
        gap_head = ContinuousGapHead(output_mode="move_coefficient").to(device)
        gap_head.load_state_dict(gap_payload["model"], strict=True)
        optimizer1 = torch.optim.AdamW(
            critic1.parameters(), lr=args.critic_learning_rate, weight_decay=args.weight_decay
        )
        optimizer2 = torch.optim.AdamW(
            critic2.parameters(), lr=args.critic_learning_rate, weight_decay=args.weight_decay
        )
        optimizer1.load_state_dict(joint1["optimizer"])
        optimizer2.load_state_dict(joint2["optimizer"])
        pair_critic1 = pair_critic2 = None
        pair_optimizer1 = pair_optimizer2 = None
        if args.pair_delta_enabled:
            # Preserve the exact Torch RNG stream used by the paired baseline
            # Actor exploration/update.  Pair-model initialization is isolated.
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_state = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            pair_critic1 = initialize_pair_delta_critic(critic1, device)
            pair_critic2 = initialize_pair_delta_critic(critic2, device)
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            pair_optimizer1 = torch.optim.AdamW(
                pair_critic1.parameters(), lr=args.pair_delta_learning_rate,
                weight_decay=args.weight_decay,
            )
            pair_optimizer2 = torch.optim.AdamW(
                pair_critic2.parameters(), lr=args.pair_delta_learning_rate,
                weight_decay=args.weight_decay,
            )
        gap_optimizer = torch.optim.AdamW(
            gap_head.parameters(), lr=args.gap_learning_rate, weight_decay=args.weight_decay
        )
        gap_optimizer.load_state_dict(gap_payload["optimizer"])
        log_scale = nn.Parameter(torch.full((8, 2), math.log(args.initial_exploration_scale), device=device))
        log_alpha = nn.Parameter(torch.tensor(math.log(args.initial_temperature), device=device))
        actor_optimizer = torch.optim.AdamW(
            list(actor.parameters()) + [log_scale], lr=args.actor_learning_rate,
            weight_decay=args.weight_decay,
        )
        temperature_optimizer = torch.optim.Adam(
            [log_alpha], lr=args.temperature_learning_rate
        )
        replay_path = source_oac1 / f"seed_{seed}" / "actor_visited_replay.npz"
        with np.load(replay_path, allow_pickle=False) as loaded:
            source_replay = {key: np.asarray(loaded[key]) for key in loaded.files}
        replay = filter_replay(source_replay, fit)
        replay = {key: replay[key] for key in (
            "state_index", "action", "cost", "round", "interaction_group",
            "role", "pre_critic1", "pre_critic2",
        )}
        initial_replay_rows = len(replay["cost"])
        pair_pretrain_records = []
        if args.pair_delta_enabled:
            pair_rng = np.random.default_rng(260825600 + seed)
            for pair_update in range(1, args.pair_delta_pretrain_updates + 1):
                pair_batch = sample_pairs(
                    data, fit, replay, args.pair_batch_size,
                    args.material_gap, pair_rng,
                )
                pair_optimizer1.zero_grad(set_to_none=True)
                pair_optimizer2.zero_grad(set_to_none=True)
                pair_loss1, pair_info1 = pair_delta_objective(
                    pair_critic1, payload1, critic_inputs, pair_batch,
                    args, device,
                )
                pair_loss2, pair_info2 = pair_delta_objective(
                    pair_critic2, payload2, critic_inputs, pair_batch,
                    args, device,
                )
                pair_total = args.pair_delta_weight * (pair_loss1 + pair_loss2)
                pair_total.backward()
                torch.nn.utils.clip_grad_norm_(pair_critic1.parameters(), 10.0)
                torch.nn.utils.clip_grad_norm_(pair_critic2.parameters(), 10.0)
                pair_optimizer1.step(); pair_optimizer2.step()
                pair_pretrain_records.append({
                    "total": float(pair_total.detach()),
                    "delta1": pair_info1["delta"], "delta2": pair_info2["delta"],
                    "ranking1": pair_info1["ranking"],
                    "ranking2": pair_info2["ranking"],
                })
                if pair_update % 400 == 0:
                    recent_pair = pair_pretrain_records[-400:]
                    print(
                        f"seed={seed} pair-pretrain={pair_update}/"
                        f"{args.pair_delta_pretrain_updates} "
                        f"loss={np.mean([row['total'] for row in recent_pair]):.4f}",
                        flush=True,
                    )
        next_group = int(replay["interaction_group"].max()) + 1
        initial_metrics, initial_cost = evaluate_actor(
            args, actor, actor_inputs, selection, None,
            data, states, current, references, backend, weights, params, device,
            26082420 + seed,
        )
        selected_cost = initial_cost.copy()
        selected_metrics = initial_metrics
        selected_round = 0
        round_records = []
        evaluation_records = [{
            "round": 0, "role": "initial_selected", "accepted": True,
            "metrics": initial_metrics,
        }]
        shadow_state = copy.deepcopy(actor.state_dict())
        tail_lagrange = float(args.tail_dual_initial)
        tail_cvar_ema: float | None = None
        tail_cvar_window: list[float] = []

        for local_round in range(1, args.rounds + 1):
            aux_update_due = (
                local_round % args.aux_update_interval_rounds == 0
            )
            absolute_round = 10 + local_round
            chosen = stratified_contexts(
                data, fit, args.contexts_per_round, interaction_rng
            )
            mean = actor_mean(actor, actor_inputs, chosen, device)
            action_bank = actor_exploration_bank(
                mean, log_scale, interaction_rng, args.minimum_exploration_scale,
                args.maximum_exploration_scale,
            )
            costs = rollout_bank(
                backend, weights, params, action_bank, states, current,
                references, chosen, args.rollout_batch_size, device,
            )
            flat_state = np.repeat(chosen, len(ROLE_NAMES))
            flat_action = action_bank.reshape(-1, 8, 2)
            flat_cost = costs.reshape(-1)
            roles = np.tile(np.asarray(ROLE_NAMES), len(chosen))
            groups = np.repeat(
                next_group + np.arange(len(chosen)), len(ROLE_NAMES)
            ).astype(np.int32)
            next_group += len(chosen)
            pre1 = predict_actions(
                critic1, critic_inputs, payload1, flat_state, flat_action, device
            )
            pre2 = predict_actions(
                critic2, critic_inputs, payload2, flat_state, flat_action, device
            )
            replay = append_replay(
                replay, flat_state, flat_action, flat_cost, absolute_round,
                groups, roles, pre1, pre2,
            )
            replay_best = group_best_rows(replay)
            critic_losses = []
            shadow_state = copy.deepcopy(actor.state_dict())
            round_start_scale = log_scale.detach().clone()
            with torch.no_grad():
                round_start_action = actor_tensor(
                    actor, actor_inputs, fit, device
                ).detach().clone()
            actor_updates = int(args.actor_updates_per_round)
            critic_partition = [
                args.critic_updates_per_round // actor_updates
                + int(step < args.critic_updates_per_round % actor_updates)
                for step in range(actor_updates)
            ]
            scheduled_actor_learning_rate = actor_learning_rate_for_round(
                args, local_round
            )
            microstep_learning_rate = scheduled_actor_learning_rate / (
                actor_updates if args.multi_actor_update_pilot else 1
            )
            microstep_trust = args.max_step_sigma_rms / (
                actor_updates if args.multi_actor_update_pilot else 1
            )
            actor_microsteps = []
            round_projection_min = 1.0

            for actor_step, critic_step_count in enumerate(critic_partition):
                for _ in range(critic_step_count):
                    points = sample_training_points(
                        data, fit, replay, absolute_round, args.batch_size,
                        critic_rng,
                    )
                    pairs = sample_pairs(
                        data, fit, replay, args.pair_batch_size,
                        args.material_gap, critic_rng,
                    )
                    gap_batch = sample_gap_batch(
                        data, fit, replay, replay_best, args.gap_batch_size,
                        critic_rng, "move_coefficient",
                    )
                    optimizer1.zero_grad(set_to_none=True)
                    optimizer2.zero_grad(set_to_none=True)
                    gap_optimizer.zero_grad(set_to_none=True)
                    if args.pair_delta_enabled:
                        pair_optimizer1.zero_grad(set_to_none=True)
                        pair_optimizer2.zero_grad(set_to_none=True)
                    value1, info1 = value_objective(
                        critic1, payload1, critic_inputs, points, pairs,
                        args, device,
                    )
                    value2, info2 = value_objective(
                        critic2, payload2, critic_inputs, points, pairs,
                        args, device,
                    )
                    gap_loss, gap_info = gap_objective(
                        critic1, payload1, critic_inputs, critic2, payload2,
                        critic_inputs, gap_head, gap_batch, device,
                        "move_coefficient",
                    )
                    total = value1 + value2 + args.gap_weight * gap_loss
                    pair_info1 = {"delta": 0.0, "ranking": 0.0}
                    pair_info2 = {"delta": 0.0, "ranking": 0.0}
                    if args.pair_delta_enabled:
                        pair_loss1, pair_info1 = pair_delta_objective(
                            pair_critic1, payload1, critic_inputs, pairs,
                            args, device,
                        )
                        pair_loss2, pair_info2 = pair_delta_objective(
                            pair_critic2, payload2, critic_inputs, pairs,
                            args, device,
                        )
                        total = total + args.pair_delta_weight * (
                            pair_loss1 + pair_loss2
                        )
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(critic1.parameters(), 10.0)
                    torch.nn.utils.clip_grad_norm_(critic2.parameters(), 10.0)
                    torch.nn.utils.clip_grad_norm_(gap_head.parameters(), 10.0)
                    if args.pair_delta_enabled:
                        torch.nn.utils.clip_grad_norm_(
                            pair_critic1.parameters(), 10.0
                        )
                        torch.nn.utils.clip_grad_norm_(
                            pair_critic2.parameters(), 10.0
                        )
                    optimizer1.step(); optimizer2.step(); gap_optimizer.step()
                    if args.pair_delta_enabled:
                        pair_optimizer1.step(); pair_optimizer2.step()
                    critic_losses.append({
                        "total": float(total.detach()),
                        "critic1_value": info1["value"],
                        "critic2_value": info2["value"],
                        "gap": gap_info["gap"],
                        "pair_delta1": pair_info1["delta"],
                        "pair_delta2": pair_info2["delta"],
                    })

                for parameter_group in actor_optimizer.param_groups:
                    parameter_group["lr"] = microstep_learning_rate
                cpu_rng_state = torch.get_rng_state()
                cuda_rng_state = (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None
                )
                torch.manual_seed(
                    260828000 + seed * 100000 + local_round * 100 + actor_step
                )
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(
                        260828000 + seed * 100000
                        + local_round * 100 + actor_step
                    )
                actor_info_step = actor_update(
                    args, actor, selected_actor, actor_optimizer, log_scale,
                    log_alpha, temperature_optimizer, actor_inputs,
                    critic_inputs, critic1, payload1, critic2, payload2,
                    gap_head, data, fit, actor_rng, device, tail_lagrange,
                    pair_critic1, pair_critic2,
                    step_limit_sigma_rms=microstep_trust,
                    update_temperature=(
                        actor_step == actor_updates - 1 and aux_update_due
                    ),
                    dbm_context=(
                        states, current, references, backend, weights, params
                    ),
                )
                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_state)
                cumulative_rms, round_projection = project_actor_to_round_trust(
                    actor, log_scale, shadow_state, round_start_scale,
                    round_start_action, actor_inputs, fit, device,
                    args.max_step_sigma_rms,
                )
                round_projection_min = min(
                    round_projection_min, round_projection
                )
                actor_info_step.update({
                    "microstep": actor_step + 1,
                    "critic_updates_before": critic_step_count,
                    "learning_rate": float(microstep_learning_rate),
                    "scheduled_round_learning_rate": float(
                        scheduled_actor_learning_rate
                    ),
                    "round_cumulative_step_sigma_rms": cumulative_rms,
                    "round_trust_projection": round_projection,
                })
                actor_microsteps.append(actor_info_step)

            actor_info = dict(actor_microsteps[-1])
            actor_info.update({
                "microstep_count": actor_updates,
                "critic_update_partition": critic_partition,
                "microstep_learning_rate": float(microstep_learning_rate),
                "scheduled_round_learning_rate": float(
                    scheduled_actor_learning_rate
                ),
                "microstep_trust_sigma_rms": float(microstep_trust),
                "round_cumulative_step_sigma_rms": float(
                    actor_microsteps[-1]["round_cumulative_step_sigma_rms"]
                ),
                "round_trust_projection_min": float(round_projection_min),
                "step_sigma_rms": float(
                    actor_microsteps[-1]["round_cumulative_step_sigma_rms"]
                ),
                "trust_projection": float(round_projection_min),
                "tail_lagrange_used": (
                    0.0 if args.actor_objective_mode == "deterministic_center_dbm"
                    else float(tail_lagrange)
                ),
                "tail_regression_cvar": float(np.mean([
                    row["tail_regression_cvar"] for row in actor_microsteps
                ])),
                "aux_update_due": bool(aux_update_due),
                "microsteps": actor_microsteps,
            })
            tail_cvar_window.append(float(actor_info["tail_regression_cvar"]))
            tail_dual_observation = None
            if args.tail_constraint_mode == "adaptive":
                if aux_update_due:
                    tail_dual_observation = float(np.mean(tail_cvar_window))
                    tail_cvar_window.clear()
                    if tail_cvar_ema is None:
                        tail_cvar_ema = tail_dual_observation
                    else:
                        tail_cvar_ema = (
                            args.tail_dual_ema_decay * tail_cvar_ema
                            + (1.0 - args.tail_dual_ema_decay)
                            * tail_dual_observation
                        )
                    tail_lagrange = float(np.clip(
                        tail_lagrange
                        + args.tail_dual_learning_rate
                        * (tail_cvar_ema - args.tail_cvar_budget_log),
                        0.0, args.tail_dual_maximum,
                    ))
            actor_info["tail_dual_observation"] = tail_dual_observation
            actor_info["tail_cvar_window_count_after"] = len(tail_cvar_window)
            actor_info["tail_cvar_ema_after"] = (
                None if tail_cvar_ema is None else float(tail_cvar_ema)
            )
            actor_info["tail_lagrange_after"] = float(tail_lagrange)
            post1 = predict_actions(
                critic1, critic_inputs, payload1, flat_state, flat_action, device
            )
            post2 = predict_actions(
                critic2, critic_inputs, payload2, flat_state, flat_action, device
            )
            accuracy, pair_count = material_pair_accuracy(
                np.maximum(post1, post2), flat_cost,
                groups - groups.min(), args.material_gap,
            )
            pair_accuracy = pair_accuracy_count = 0
            if args.pair_delta_enabled:
                pair_accuracy, pair_accuracy_count = pair_mean_comparison_accuracy(
                    pair_critic1, payload1, pair_critic2, payload2,
                    critic_inputs, chosen, action_bank, costs,
                    args.material_gap, args.pair_delta_disagreement_weight, device,
                )
            record = {
                "round": local_round,
                "new_rows": int(len(flat_cost)),
                "mean_new_cost": float(np.mean(flat_cost)),
                "critic_pair_accuracy": accuracy,
                "critic_pair_count": pair_count,
                "pair_delta_mean_comparison_accuracy": pair_accuracy,
                "pair_delta_mean_comparison_count": pair_accuracy_count,
                "critic_loss": {
                    key: float(np.mean([row[key] for row in critic_losses]))
                    for key in critic_losses[0]
                },
                "actor": actor_info,
            }
            round_records.append(record)

            if local_round % args.evaluation_interval == 0 or local_round == args.rounds:
                latest_metrics, latest_cost = evaluate_actor(
                    args, actor, actor_inputs, selection, initial_cost, data,
                    states, current, references, backend, weights, params, device,
                    26082500 + seed * 100 + local_round,
                )
                gate = acceptance_gate(
                    latest_metrics, latest_cost, selected_cost, data, selection,
                    args,
                )
                accepted = bool(all(gate.values()))
                if accepted:
                    selected_actor.load_state_dict(actor.state_dict(), strict=True)
                    selected_cost = latest_cost.copy()
                    selected_metrics = latest_metrics
                    selected_round = local_round
                evaluation_records.append({
                    "round": local_round, "role": "latest", "accepted": accepted,
                    "acceptance_gate": gate, "metrics": latest_metrics,
                    "selected_round_after": selected_round,
                })
                print(
                    f"seed={seed} round={local_round} "
                    f"latestR={latest_metrics['headroom_recovery_vs_bank_best']:.4f} "
                    f"gain={latest_metrics['gain_vs_initial']['mean']:.4f} "
                    f"selected={selected_round} value_pair={accuracy:.3f} "
                    f"delta_pair={pair_accuracy:.3f}",
                    flush=True,
                )

        critic_metrics = evaluate_critic(
            args, data, folds, replay, critic1, payload1, critic2, payload2,
            gap_head, device, outer_fold,
        )
        replay.update({
            "final_critic1": predict_actions(
                critic1, critic_inputs, payload1, replay["state_index"], replay["action"], device
            ),
            "final_critic2": predict_actions(
                critic2, critic_inputs, payload2, replay["state_index"], replay["action"], device
            ),
        })
        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir()
        np.savez_compressed(seed_dir / "actor_visited_replay.npz", **replay)
        save_seed_checkpoint(
            seed_dir, actor, selected_actor, shadow_state, critic1, payload1,
            critic2, payload2, gap_head, actor_optimizer, optimizer1, optimizer2,
            gap_optimizer, log_scale, log_alpha, temperature_optimizer,
            args.rounds * args.actor_updates_per_round,
            args.rounds * args.critic_updates_per_round,
            selected_round, tail_lagrange, tail_cvar_ema,
            pair_critic1, pair_critic2, pair_optimizer1, pair_optimizer2,
        )
        record = {
            "seed": seed,
            "initial_actor_checkpoint": str(actor_path.resolve()),
            "initial_actor_checkpoint_sha256": sha256_file(actor_path),
            "initial_actor_module_sha256": initial_actor_hash,
            "latest_actor_module_sha256": module_digest(actor),
            "selected_actor_module_sha256": module_digest(selected_actor),
            "initial_replay_rows_after_internal_filter": initial_replay_rows,
            "final_replay_rows": int(len(replay["cost"])),
            "interaction_dbm_rollouts": int(
                args.rounds * args.contexts_per_round * len(ROLE_NAMES)
            ),
            "internal_evaluation_dbm_rollouts": int(
                (1 + args.rounds // args.evaluation_interval) * len(selection)
            ),
            "new_dbm_rollouts": int(
                args.rounds * args.contexts_per_round * len(ROLE_NAMES)
                + (1 + args.rounds // args.evaluation_interval) * len(selection)
            ),
            "actor_update_count": args.rounds * args.actor_updates_per_round,
            "critic_additional_update_count": args.rounds * args.critic_updates_per_round,
            "pair_delta_enabled": bool(args.pair_delta_enabled),
            "pair_delta_pretrain_update_count": (
                args.pair_delta_pretrain_updates if args.pair_delta_enabled else 0
            ),
            "pair_delta_online_update_count": (
                args.rounds * args.critic_updates_per_round
                if args.pair_delta_enabled else 0
            ),
            "pair_delta_pretrain_final_400": ({
                key: float(np.mean([row[key] for row in pair_pretrain_records[-400:]]))
                for key in pair_pretrain_records[-1]
            } if pair_pretrain_records else None),
            "selected_round": selected_round,
            "final_tail_lagrange": float(tail_lagrange),
            "final_tail_cvar_ema": (
                None if tail_cvar_ema is None else float(tail_cvar_ema)
            ),
            "initial_metrics": initial_metrics,
            "latest_metrics": latest_metrics,
            "selected_metrics": selected_metrics,
            "critic_metrics": critic_metrics,
            "rounds": round_records,
            "evaluations": evaluation_records,
            "source_replay": str(replay_path.resolve()),
            "source_replay_sha256": sha256_file(replay_path),
        }
        (seed_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
        with (seed_dir / "iteration_metrics.jsonl").open("w") as stream:
            for row in round_records:
                stream.write(json.dumps(row) + "\n")
        all_records.append(record)

    seed_pass = []
    for row in all_records:
        selected = row["selected_metrics"]
        ci = selected["gain_episode_bootstrap_ci95"]
        critic_pair = selected_pair = row["critic_metrics"][
            "actor_visited_material_pair_accuracy"
        ]
        gates = {
            "selected_mean_gain_positive": selected["gain_vs_initial"]["mean"] > 0.0,
            "selected_ci_lower_positive": ci[0] > 0.0,
            "selected_median_gain_positive": selected["gain_vs_initial"]["median"] > 0.0,
            "speed_2_4_nonnegative": selected["by_speed"]["2.4"]["mean_gain"] >= 0.0,
            "speed_2_8_nonnegative": selected["by_speed"]["2.8"]["mean_gain"] >= 0.0,
            "critic_pair_ge_0_85": critic_pair >= 0.85,
            "finite_and_not_saturated": selected["state_any_saturation_fraction"] <= 0.10,
        }
        if args.selection_gain_p05_floor is not None:
            gates["selected_gain_p05_above_floor"] = (
                selected["gain_vs_initial"]["p05"]
                >= args.selection_gain_p05_floor
            )
        if args.selection_speed_2_4_gain_p05_floor is not None:
            gates["selected_speed_2_4_gain_p05_above_floor"] = (
                selected["by_speed"]["2.4"]["p05_gain"]
                >= args.selection_speed_2_4_gain_p05_floor
            )
        if args.selection_speed_2_8_gain_p05_floor is not None:
            gates["selected_speed_2_8_gain_p05_above_floor"] = (
                selected["by_speed"]["2.8"]["p05_gain"]
                >= args.selection_speed_2_8_gain_p05_floor
            )
        row["oac2_seed_gates"] = gates
        row["oac2_seed_pass"] = bool(all(gates.values()))
        seed_pass.append(row["oac2_seed_pass"])
        (args.output_dir / f"seed_{row['seed']}" / "summary.json").write_text(
            json.dumps(row, indent=2) + "\n"
        )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            "OAC2_MECHANISM_PASS_READY_FOR_FULL_FOLD_REPLICATION"
            if sum(seed_pass) >= 2
            else "OAC2_MECHANISM_FAIL_ACTOR_UPDATE_NOT_AUTHORIZED_FORWARD"
        ),
        "passed_seed_count": int(sum(seed_pass)),
        "required_seed_count": 2,
        "records": all_records,
        "actor_update_count_per_seed": args.rounds * args.actor_updates_per_round,
        "critic_update_ratio": (
            f"{args.critic_updates_per_round}:{args.actor_updates_per_round}"
        ),
        "new_dbm_rollouts": int(sum(row["new_dbm_rollouts"] for row in all_records)),
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "qualification": summary["qualification"],
        "passed_seed_count": summary["passed_seed_count"],
        "selected_round": {str(row["seed"]): row["selected_round"] for row in all_records},
        "selected_headroom_recovery": {
            str(row["seed"]): row["selected_metrics"]["headroom_recovery_vs_bank_best"]
            for row in all_records
        },
        "selected_gain_mean": {
            str(row["seed"]): row["selected_metrics"]["gain_vs_initial"]["mean"]
            for row in all_records
        },
    }, indent=2))


if __name__ == "__main__":
    main()
