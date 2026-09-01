#!/usr/bin/env python3
"""Train a continuously updated train-only single-step Alpha SAC.

This is an entropy-regularized contextual bandit, not a vehicle-transition MDP.
Each interaction samples a train-only frozen context, lets the hybrid Actor choose
stay/move and a continuous alpha, evaluates that unique center with one direct DBM
rollout, appends the result to actor-visited replay, and then updates both main
Critics and the Actor.  Since every transition is terminal, the exact SAC target
is the observed transformed reward; no Bellman target network is required.

The original 21-point TR1 line replay remains a coverage anchor.  Actor-visited
replay is sampled separately so it cannot be drowned out by the offline grid.
Internal-selection episodes select checkpoints but never enter replay.  Formal
validation and test episodes are not loaded.
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
import torch.nn.functional as F

from car_foundation.mppi_proposal_policy import (
    TorchMPPITrustAlphaCritic,
    TorchMPPITrustAlphaSACPolicy,
)
from generate_dbm_proposal_teacher import repository_state, sha256_file
from train_mppi_direct_alpha_actor_critic import (
    critic_bank,
    critic_metrics,
    critic_one,
    gate_metrics,
    transform_reward,
)
from train_mppi_direct_trust_region_actor import (
    DEFAULT_LABELS,
    direct_cost,
    distribution,
    load_actor_payload,
    load_dataset,
    tensorize,
)
from train_mppi_direct_trust_alpha_policy import center_from_alpha, extra_tensors


DEFAULT_INITIAL_AC = Path(
    "outputs/mppi_proposal/direct_alpha_actor_critic_20260810_v1/"
    "alpha_actor_critic_selected.pt"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_alpha_online_sac_20260810_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--initial-ac", type=Path, default=DEFAULT_INITIAL_AC)
    parser.add_argument(
        "--resume-run", type=Path,
        help="Continue from a prior online-SAC run and its actor-visited replay.",
    )
    parser.add_argument(
        "--resume-checkpoint-role", choices=("latest", "selected"),
        default="latest",
        help="Resume the latest training state when available, or the selected Actor state.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--training-mode", choices=("actor_critic", "critic_only"),
        default="actor_critic",
    )
    parser.add_argument(
        "--collect-transitions", action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable to isolate Critic optimization on a fixed restored replay.",
    )
    parser.add_argument(
        "--selection-mode", choices=("safe_gate", "aggressive_mean"),
        default="safe_gate",
        help=(
            "safe_gate preserves the historical P05/worst/stay gates; "
            "aggressive_mean selects the threshold/checkpoint with minimum mean "
            "direct cost and records tail risk without blocking training"
        ),
    )
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--contexts-per-iteration", type=int, default=320)
    parser.add_argument("--policy-samples-per-context", type=int, default=2)
    parser.add_argument("--uniform-samples-per-context", type=int, default=1)
    parser.add_argument("--replay-capacity", type=int, default=200000)
    parser.add_argument(
        "--context-sampling-mode",
        choices=("stratified", "critic_priority_mixed"),
        default="stratified",
        help=(
            "Select train-only rollout contexts uniformly by speed, or mix that "
            "coverage with Critic-disagreement/gate/cost/boundary priority."
        ),
    )
    parser.add_argument("--priority-context-fraction", type=float, default=0.5)
    parser.add_argument("--priority-candidate-multiplier", type=int, default=4)
    parser.add_argument("--priority-disagreement-weight", type=float, default=0.4)
    parser.add_argument("--priority-gate-weight", type=float, default=0.2)
    parser.add_argument("--priority-cost-weight", type=float, default=0.3)
    parser.add_argument("--priority-boundary-weight", type=float, default=0.1)
    parser.add_argument(
        "--local-probe-radii", default="",
        help="Comma-separated paired alpha offsets around the deterministic Actor action.",
    )
    parser.add_argument(
        "--local-probe-boundary-mode", choices=("clip", "shifted_window"),
        default="clip",
        help=(
            "clip reproduces symmetric probes with boundary duplicates; "
            "shifted_window keeps three ordered, radius-spaced actions by moving "
            "the probe window inward when the Actor is near alpha 0 or 1."
        ),
    )
    parser.add_argument(
        "--reset-local-probe-replay", action=argparse.BooleanOptionalAction,
        default=False,
        help="Start a fresh local-probe replay when changing its geometry.",
    )
    parser.add_argument("--local-probe-capacity", type=int, default=100000)
    parser.add_argument("--local-probe-batch-size", type=int, default=256)
    parser.add_argument("--critic-updates-per-iteration", type=int, default=40)
    parser.add_argument("--actor-updates-per-iteration", type=int, default=4)
    parser.add_argument("--critic-warmup-iterations", type=int, default=0)
    parser.add_argument("--actor-samples-per-update", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--online-replay-fraction", type=float, default=0.25)
    parser.add_argument("--negative-replay-boost", type=float, default=2.0)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-5)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--reward-scale", type=float, default=5.0)
    parser.add_argument("--critic-huber-beta", type=float, default=0.05)
    parser.add_argument("--critic-anchor-weight", type=float, default=0.10)
    parser.add_argument("--critic-delta-weight", type=float, default=0.5)
    parser.add_argument("--critic-negative-loss-weight", type=float, default=2.0)
    parser.add_argument("--critic-sign-weight", type=float, default=0.05)
    parser.add_argument("--critic-sign-temperature", type=float, default=0.10)
    parser.add_argument(
        "--critic-objective", choices=("mixed_points", "separated_curve"),
        default="mixed_points",
    )
    parser.add_argument(
        "--critic-trainable-scope", choices=("all", "action_heads"),
        default="all",
        help=(
            "action_heads freezes the pretrained context/direction/state fusion "
            "and updates only alpha_encoder plus q_head"
        ),
    )
    parser.add_argument("--critic-base-context-batch-size", type=int, default=64)
    parser.add_argument("--critic-base-value-weight", type=float, default=1.0)
    parser.add_argument("--critic-base-delta-weight", type=float, default=1.0)
    parser.add_argument("--critic-online-value-weight", type=float, default=0.5)
    parser.add_argument("--critic-online-delta-weight", type=float, default=0.5)
    parser.add_argument("--critic-local-delta-weight", type=float, default=0.0)
    parser.add_argument("--critic-local-sign-weight", type=float, default=0.0)
    parser.add_argument("--critic-local-sign-threshold", type=float, default=0.002)
    parser.add_argument("--critic-local-sign-temperature", type=float, default=0.02)
    parser.add_argument(
        "--critic-selection-objective",
        choices=("full_line", "local_probe_guarded"), default="full_line",
    )
    parser.add_argument("--critic-full-line-score-tolerance", type=float, default=0.01)
    parser.add_argument("--critic-min-selection-improvement", type=float, default=1e-4)
    parser.add_argument("--gate-entropy-temperature", type=float, default=0.01)
    parser.add_argument("--alpha-entropy-temperature", type=float, default=0.01)
    parser.add_argument("--critic-disagreement-weight", type=float, default=0.05)
    parser.add_argument("--actor-bc-weight-start", type=float, default=0.20)
    parser.add_argument("--actor-bc-weight-end", type=float, default=0.02)
    parser.add_argument("--initial-log-std", type=float, default=1.0)
    parser.add_argument("--alpha-logit-scale", type=float, default=1.0)
    parser.add_argument("--evaluation-interval", type=int, default=2)
    parser.add_argument("--critic-rank-evaluation-interval", type=int, default=10)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--freeze-actor-context-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--regression-mean-penalty", type=float, default=0.25)
    parser.add_argument("--regression-p95-penalty", type=float, default=0.05)
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_local_probe_radii(value: str | list[float] | tuple[float, ...]) -> tuple[float, ...]:
    if isinstance(value, str):
        result = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    else:
        result = tuple(float(item) for item in value)
    if any(not 0.0 < item <= 1.0 for item in result):
        raise ValueError("local probe radii must be within (0,1]")
    if len(set(result)) != len(result):
        raise ValueError("local probe radii must be unique")
    return result


def assert_finite_module(module: torch.nn.Module, name: str) -> None:
    for parameter_name, parameter in module.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(
                f"{name}.{parameter_name} contains non-finite values"
            )


def serialized_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


class OnlineReplay:
    """Bounded actor-visited replay with tail-aware sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.context = np.empty(self.capacity, np.int64)
        self.alpha = np.empty(self.capacity, np.float32)
        self.reward = np.empty(self.capacity, np.float32)
        self.iteration = np.empty(self.capacity, np.int32)
        self.size = 0
        self.position = 0

    def add(
        self,
        context: np.ndarray,
        alpha: np.ndarray,
        reward: np.ndarray,
        iteration: int,
    ) -> None:
        context = np.asarray(context, np.int64).reshape(-1)
        alpha = np.asarray(alpha, np.float32).reshape(-1)
        reward = np.asarray(reward, np.float32).reshape(-1)
        if not (len(context) == len(alpha) == len(reward)):
            raise ValueError("replay arrays have different lengths")
        if len(context) > self.capacity:
            context = context[-self.capacity:]
            alpha = alpha[-self.capacity:]
            reward = reward[-self.capacity:]
        index = (np.arange(len(context)) + self.position) % self.capacity
        self.context[index] = context
        self.alpha[index] = alpha
        self.reward[index] = reward
        self.iteration[index] = int(iteration)
        self.position = int((self.position + len(context)) % self.capacity)
        self.size = min(self.capacity, self.size + len(context))

    def sample(
        self,
        count: int,
        rng: np.random.Generator,
        negative_boost: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.size == 0 or count <= 0:
            return (
                np.empty(0, np.int64), np.empty(0, np.float32),
                np.empty(0, np.float32),
            )
        reward = self.reward[:self.size]
        priority = 1.0 + float(negative_boost) * (reward < 0.0)
        priority += np.minimum(np.abs(reward) / 20.0, 2.0)
        probability = priority / np.sum(priority)
        selected = rng.choice(self.size, size=count, replace=True, p=probability)
        return self.context[selected], self.alpha[selected], self.reward[selected]

    def arrays(self) -> dict[str, np.ndarray]:
        if self.size < self.capacity:
            index = np.arange(self.size)
        else:
            index = np.concatenate((
                np.arange(self.position, self.capacity), np.arange(self.position)
            ))
        return {
            "context_index": self.context[index].copy(),
            "alpha": self.alpha[index].copy(),
            "reward": self.reward[index].copy(),
            "iteration": self.iteration[index].copy(),
        }

    def load_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        """Restore chronological replay arrays into a fresh replay buffer."""
        context = np.asarray(arrays["context_index"], np.int64).reshape(-1)
        alpha = np.asarray(arrays["alpha"], np.float32).reshape(-1)
        reward = np.asarray(arrays["reward"], np.float32).reshape(-1)
        iteration = np.asarray(arrays["iteration"], np.int32).reshape(-1)
        if not (len(context) == len(alpha) == len(reward) == len(iteration)):
            raise ValueError("restored replay arrays have different lengths")
        if len(context) > self.capacity:
            context = context[-self.capacity:]
            alpha = alpha[-self.capacity:]
            reward = reward[-self.capacity:]
            iteration = iteration[-self.capacity:]
        self.size = len(context)
        self.position = self.size % self.capacity
        self.context[:self.size] = context
        self.alpha[:self.size] = alpha
        self.reward[:self.size] = reward
        self.iteration[:self.size] = iteration


class LocalProbeReplay:
    """Paired left/center/right Actor-neighborhood rewards."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.context = np.empty(self.capacity, np.int64)
        self.alpha = np.empty((self.capacity, 3), np.float32)
        self.reward = np.empty((self.capacity, 3), np.float32)
        self.radius = np.empty(self.capacity, np.float32)
        self.anchor_alpha = np.empty(self.capacity, np.float32)
        self.iteration = np.empty(self.capacity, np.int32)
        self.size = 0
        self.position = 0

    def add(
        self, context, alpha, reward, radius, anchor_alpha, iteration: int
    ) -> None:
        context = np.asarray(context, np.int64).reshape(-1)
        alpha = np.asarray(alpha, np.float32).reshape(-1, 3)
        reward = np.asarray(reward, np.float32).reshape(-1, 3)
        radius = np.asarray(radius, np.float32).reshape(-1)
        anchor_alpha = np.asarray(anchor_alpha, np.float32).reshape(-1)
        if not (
            len(context) == len(alpha) == len(reward)
            == len(radius) == len(anchor_alpha)
        ):
            raise ValueError("local-probe replay arrays have different lengths")
        if len(context) > self.capacity:
            context, alpha, reward, radius, anchor_alpha = (
                value[-self.capacity:]
                for value in (context, alpha, reward, radius, anchor_alpha)
            )
        index = (np.arange(len(context)) + self.position) % self.capacity
        self.context[index] = context
        self.alpha[index] = alpha
        self.reward[index] = reward
        self.radius[index] = radius
        self.anchor_alpha[index] = anchor_alpha
        self.iteration[index] = int(iteration)
        self.position = int((self.position + len(context)) % self.capacity)
        self.size = min(self.capacity, self.size + len(context))

    def sample(self, count: int, rng: np.random.Generator):
        if self.size == 0 or count <= 0:
            return (
                np.empty(0, np.int64), np.empty((0, 3), np.float32),
                np.empty((0, 3), np.float32),
            )
        selected = rng.choice(self.size, size=count, replace=True)
        return (
            self.context[selected], self.alpha[selected], self.reward[selected]
        )

    def arrays(self) -> dict[str, np.ndarray]:
        if self.size < self.capacity:
            index = np.arange(self.size)
        else:
            index = np.concatenate((
                np.arange(self.position, self.capacity), np.arange(self.position)
            ))
        return {
            "context_index": self.context[index].copy(),
            "alpha": self.alpha[index].copy(),
            "reward": self.reward[index].copy(),
            "radius": self.radius[index].copy(),
            "anchor_alpha": self.anchor_alpha[index].copy(),
            "iteration": self.iteration[index].copy(),
        }

    def load_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        context = np.asarray(arrays["context_index"], np.int64).reshape(-1)
        alpha = np.asarray(arrays["alpha"], np.float32).reshape(-1, 3)
        reward = np.asarray(arrays["reward"], np.float32).reshape(-1, 3)
        radius = np.asarray(arrays["radius"], np.float32).reshape(-1)
        anchor_alpha = np.asarray(
            arrays.get("anchor_alpha", alpha[:, 1]), np.float32
        ).reshape(-1)
        iteration = np.asarray(arrays["iteration"], np.int32).reshape(-1)
        if not (
            len(context) == len(alpha) == len(reward) == len(radius)
            == len(anchor_alpha) == len(iteration)
        ):
            raise ValueError("restored local-probe arrays have different lengths")
        if len(context) > self.capacity:
            context, alpha, reward, radius, anchor_alpha, iteration = (
                value[-self.capacity:]
                for value in (
                    context, alpha, reward, radius, anchor_alpha, iteration
                )
            )
        self.size = len(context)
        self.position = self.size % self.capacity
        self.context[:self.size] = context
        self.alpha[:self.size] = alpha
        self.reward[:self.size] = reward
        self.radius[:self.size] = radius
        self.anchor_alpha[:self.size] = anchor_alpha
        self.iteration[:self.size] = iteration


def policy_batch(policy, tensors, extra, index: torch.Tensor):
    return policy(
        *(value[index] for value in tensors["inputs"]),
        extra["direction"][index], extra["rho"][index], extra["scale"][index],
    )


def make_policy(
    deterministic_state: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    dropout: float,
) -> TorchMPPITrustAlphaSACPolicy:
    policy = TorchMPPITrustAlphaSACPolicy(
        dropout=dropout, initial_log_std=args.initial_log_std,
        alpha_logit_scale=args.alpha_logit_scale,
    ).to(device)
    policy.load_deterministic_policy_state_dict(deterministic_state)
    return policy


@torch.no_grad()
def deterministic_outputs(
    policy, tensors, extra, index: np.ndarray, threshold: float,
    batch_size: int, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probability, conditional, alpha, centers = [], [], [], []
    policy.eval()
    for start in range(0, len(index), batch_size):
        absolute = torch.from_numpy(index[start:start + batch_size]).to(device)
        _, p, mean, _ = policy_batch(policy, tensors, extra, absolute)
        one_conditional = torch.sigmoid(mean)
        one_alpha = policy.deterministic_alpha(p, mean, threshold)
        one_center = center_from_alpha(extra, absolute, one_alpha)
        probability.append(p.cpu().numpy())
        conditional.append(one_conditional.cpu().numpy())
        alpha.append(one_alpha.cpu().numpy())
        centers.append(one_center.cpu().numpy())
    return tuple(np.concatenate(value) for value in (
        probability, conditional, alpha, centers
    ))


def metrics_from_cache(
    probability: np.ndarray,
    conditional: np.ndarray,
    move_cost: np.ndarray,
    data,
    index: np.ndarray,
    threshold: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    predicted_move = probability > threshold
    alpha = np.where(predicted_move, conditional, 0.0)
    cost = np.where(predicted_move, move_cost, data.old_cost[index])
    old = data.old_cost[index]
    gain = old - cost
    regression = np.maximum(-gain, 0.0)
    target_move = data.safe_index[index] > 0
    stay_recall = (
        float(np.mean(~predicted_move[~target_move]))
        if np.any(~target_move) else 1.0
    )
    move_recall = (
        float(np.mean(predicted_move[target_move]))
        if np.any(target_move) else 1.0
    )
    move_precision = (
        float(np.mean(target_move[predicted_move]))
        if np.any(predicted_move) else 1.0
    )
    score = (
        float(np.mean(cost))
        + args.regression_mean_penalty * float(np.mean(regression))
        + args.regression_p95_penalty * float(np.quantile(regression, 0.95))
    )
    result = {
        "selection_score": score,
        "direct_cost": distribution(cost),
        "old_cost": distribution(old),
        "safe_cost": distribution(data.safe_cost[index]),
        "line_argmin_cost": distribution(np.min(
            data.direct_line_cost[index], axis=1
        )),
        "gap_vs_safe_teacher": distribution(cost - data.safe_cost[index]),
        "gap_vs_line_argmin": distribution(
            cost - np.min(data.direct_line_cost[index], axis=1)
        ),
        "safe_teacher_beaten_fraction": float(np.mean(
            cost < data.safe_cost[index] - 1e-6
        )),
        "safe_teacher_tied_or_better_fraction": float(np.mean(
            cost <= data.safe_cost[index] + 1e-6
        )),
        "gain_vs_old": distribution(gain),
        "regression_fraction": float(np.mean(gain < 0.0)),
        "move_fraction": float(np.mean(predicted_move)),
        "move_accuracy": float(np.mean(predicted_move == target_move)),
        "move_recall": move_recall,
        "stay_recall": stay_recall,
        "move_precision": move_precision,
        "conditional_alpha_mae_on_move": float(np.mean(np.abs(
            conditional[target_move] - data.safe_alpha[index][target_move]
        ))) if np.any(target_move) else 0.0,
        "hard_alpha_mean": float(np.mean(alpha)),
        "hard_alpha_p95": float(np.quantile(alpha, 0.95)),
        "move_threshold": float(threshold),
    }
    result["gates"] = gate_metrics(result)
    result["pass"] = all(result["gates"].values())
    return result


@torch.no_grad()
def calibrate_and_evaluate(
    policy, data, tensors, extra, index: np.ndarray,
    args: argparse.Namespace, device: torch.device,
) -> dict[str, Any] | None:
    probability, conditional, _, centers = deterministic_outputs(
        policy, tensors, extra, index, 0.0,
        args.evaluation_batch_size, device,
    )
    move_cost = direct_cost(
        centers, data, tensors, index,
        args.evaluation_batch_size, device,
    )
    thresholds = (
        np.concatenate((
            np.arange(0.0, 1.00, 0.01),
            np.asarray((0.995, 0.999, 1.0)),
        ))
        if args.selection_mode == "aggressive_mean"
        else np.concatenate((
            np.arange(0.50, 1.00, 0.01), np.asarray((0.995, 0.999))
        ))
    )
    candidates = [
        metrics_from_cache(
            probability, conditional, move_cost, data, index,
            float(threshold), args,
        )
        for threshold in thresholds
    ]
    if args.selection_mode == "aggressive_mean":
        return min(candidates, key=lambda row: row["direct_cost"]["mean"])
    passing = [row for row in candidates if row["pass"]]
    return min(passing, key=lambda row: row["selection_score"]) if passing else None


def stratified_context_sample(
    data, fit_index: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    speeds = sorted(np.unique(data.reference_speed[fit_index]))
    groups = [fit_index[np.isclose(data.reference_speed[fit_index], speed)] for speed in speeds]
    per_group = int(math.ceil(count / len(groups)))
    selected = np.concatenate([
        rng.choice(group, size=per_group, replace=len(group) < per_group)
        for group in groups
    ])[:count]
    rng.shuffle(selected)
    return selected.astype(np.int64)


def local_probe_triplet(
    anchor_alpha: np.ndarray,
    radius: float | np.ndarray,
    boundary_mode: str,
) -> np.ndarray:
    """Build ordered probe actions while preserving the Actor anchor explicitly."""
    anchor = np.asarray(anchor_alpha, np.float32).reshape(-1)
    offset = np.broadcast_to(
        np.asarray(radius, np.float32), anchor.shape
    ).astype(np.float32)
    if boundary_mode == "clip":
        return np.stack((
            np.clip(anchor - offset, 0.0, 1.0),
            anchor,
            np.clip(anchor + offset, 0.0, 1.0),
        ), axis=1).astype(np.float32)
    if boundary_mode != "shifted_window":
        raise ValueError(f"unknown local-probe boundary mode: {boundary_mode}")

    lower = anchor - offset < 0.0
    upper = anchor + offset > 1.0
    left = np.where(
        lower, anchor,
        np.where(upper, anchor - 2.0 * offset, anchor - offset),
    )
    middle = np.where(
        lower, anchor + offset,
        np.where(upper, anchor - offset, anchor),
    )
    right = np.where(
        lower, anchor + 2.0 * offset,
        np.where(upper, anchor, anchor + offset),
    )
    return np.clip(
        np.stack((left, middle, right), axis=1), 0.0, 1.0
    ).astype(np.float32)


def rank_fraction(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float64).reshape(-1)
    if len(value) <= 1:
        return np.zeros(len(value), np.float64)
    order = np.argsort(value, kind="stable")
    result = np.empty(len(value), np.float64)
    result[order] = np.linspace(0.0, 1.0, len(value))
    return result


@torch.no_grad()
def sample_rollout_contexts(
    policy, q1, q2, data, tensors, extra, fit_index: np.ndarray,
    threshold: float, args: argparse.Namespace, rng: np.random.Generator,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    count = int(args.contexts_per_iteration)
    if args.context_sampling_mode == "stratified":
        context = stratified_context_sample(data, fit_index, count, rng)
        return context, {
            "mode": "stratified",
            "base_context_count": len(context),
            "priority_context_count": 0,
        }

    priority_count = int(round(count * args.priority_context_fraction))
    priority_count = min(max(priority_count, 0), count)
    base_count = count - priority_count
    base = stratified_context_sample(data, fit_index, base_count, rng)
    candidate_count = min(
        len(fit_index),
        max(priority_count, count * args.priority_candidate_multiplier),
    )
    candidate = stratified_context_sample(
        data, fit_index, candidate_count, rng
    )
    if len(base):
        candidate = candidate[~np.isin(candidate, base)]
    if len(candidate) < priority_count:
        remaining = fit_index[~np.isin(fit_index, base)]
        candidate = remaining
    if priority_count == 0:
        context = base
        rng.shuffle(context)
        return context.astype(np.int64), {
            "mode": "critic_priority_mixed",
            "base_context_count": len(base),
            "priority_context_count": 0,
        }

    absolute = torch.from_numpy(candidate).to(device)
    policy.eval()
    q1.eval()
    q2.eval()
    _, probability, mean, _ = policy_batch(policy, tensors, extra, absolute)
    anchor = policy.deterministic_alpha(
        probability, mean, threshold
    ).cpu().numpy()
    radius = max(args.local_probe_radii, default=0.05)
    probe = local_probe_triplet(
        anchor, radius, args.local_probe_boundary_mode
    )
    probe_tensor = torch.from_numpy(probe).to(device)
    prediction1 = critic_bank(q1, tensors, extra, absolute, probe_tensor).cpu().numpy()
    prediction2 = critic_bank(q2, tensors, extra, absolute, probe_tensor).cpu().numpy()
    disagreement = (
        np.mean(np.abs(prediction1 - prediction2), axis=1)
        + np.abs(
            (prediction1[:, -1] - prediction1[:, 0])
            - (prediction2[:, -1] - prediction2[:, 0])
        )
    )
    gate_closeness = -np.abs(probability.cpu().numpy() - float(threshold))
    old_cost = np.log1p(np.maximum(data.old_cost[candidate], 0.0))
    boundary = np.abs(anchor - 0.5) * 2.0
    score = (
        args.priority_disagreement_weight * rank_fraction(disagreement)
        + args.priority_gate_weight * rank_fraction(gate_closeness)
        + args.priority_cost_weight * rank_fraction(old_cost)
        + args.priority_boundary_weight * rank_fraction(boundary)
    )
    probability_weight = np.maximum(score, 0.0) ** 2 + 0.05
    probability_weight /= np.sum(probability_weight)
    chosen = rng.choice(
        len(candidate), size=priority_count, replace=False,
        p=probability_weight,
    )
    priority = candidate[chosen]
    context = np.concatenate((base, priority)).astype(np.int64)
    rng.shuffle(context)
    return context, {
        "mode": "critic_priority_mixed",
        "base_context_count": len(base),
        "priority_context_count": len(priority),
        "candidate_context_count": len(candidate),
        "priority_score_candidate": distribution(score),
        "priority_score_selected": distribution(score[chosen]),
        "candidate_old_cost": distribution(data.old_cost[candidate]),
        "priority_selected_old_cost": distribution(data.old_cost[priority]),
        "candidate_q_disagreement": distribution(disagreement),
        "priority_selected_q_disagreement": distribution(disagreement[chosen]),
        "candidate_gate_distance": distribution(-gate_closeness),
        "priority_selected_gate_distance": distribution((-gate_closeness)[chosen]),
    }


@torch.no_grad()
def collect_actor_visited(
    policy, q1, q2,
    replay: OnlineReplay,
    local_replay: LocalProbeReplay,
    data,
    tensors,
    extra,
    fit_index: np.ndarray,
    threshold: float,
    iteration: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    context, context_sampling = sample_rollout_contexts(
        policy, q1, q2, data, tensors, extra, fit_index,
        threshold, args, rng, device,
    )
    absolute = torch.from_numpy(context).to(device)
    policy.eval()
    _, probability, mean, log_std = policy_batch(
        policy, tensors, extra, absolute
    )
    deterministic = policy.deterministic_alpha(
        probability, mean, threshold
    ).cpu().numpy()[:, None]
    actions = [deterministic]
    if args.policy_samples_per_context:
        epsilon = torch.from_numpy(rng.standard_normal(
            (len(context), args.policy_samples_per_context)
        ).astype(np.float32)).to(device)
        conditional = torch.sigmoid(
            mean[:, None] + log_std.exp()[:, None] * epsilon
        )
        move = torch.from_numpy(
            rng.random((len(context), args.policy_samples_per_context)).astype(np.float32)
        ).to(device) < probability[:, None]
        actions.append(torch.where(
            move, conditional, torch.zeros_like(conditional)
        ).cpu().numpy())
    if args.uniform_samples_per_context:
        actions.append(rng.uniform(
            0.0, 1.0,
            size=(len(context), args.uniform_samples_per_context),
        ).astype(np.float32))
    local_columns = []
    for radius in args.local_probe_radii:
        if args.local_probe_boundary_mode == "clip":
            left_column = sum(value.shape[1] for value in actions)
            actions.append(np.clip(deterministic - radius, 0.0, 1.0))
            right_column = sum(value.shape[1] for value in actions)
            actions.append(np.clip(deterministic + radius, 0.0, 1.0))
            local_columns.append((
                float(radius), (left_column, 0, right_column)
            ))
        else:
            start = sum(value.shape[1] for value in actions)
            actions.append(local_probe_triplet(
                deterministic[:, 0], radius, args.local_probe_boundary_mode
            ))
            local_columns.append((
                float(radius), (start, start + 1, start + 2)
            ))
    alpha_matrix = np.concatenate(actions, axis=1).astype(np.float32)
    repeated_context = np.repeat(context, alpha_matrix.shape[1])
    alpha = alpha_matrix.reshape(-1)
    raw = (
        data.old_center[repeated_context]
        + alpha[:, None, None]
        * data.sigma[repeated_context, None, :]
        * data.projected_direction[repeated_context]
    )
    centers = np.clip(raw, -1.0, 1.0).astype(np.float32)
    cost = direct_cost(
        centers, data, tensors, repeated_context,
        args.evaluation_batch_size, device,
    )
    reward = data.old_cost[repeated_context] - cost
    if not all(np.all(np.isfinite(value)) for value in (
        alpha, centers, cost, reward
    )):
        raise FloatingPointError("actor-visited rollout contains non-finite values")
    replay.add(repeated_context, alpha, reward, iteration)
    reward_matrix = reward.reshape(len(context), -1)
    deterministic_reward = reward_matrix[:, 0]
    if local_columns:
        local_context, local_alpha, local_reward = [], [], []
        local_radius, local_anchor = [], []
        for radius, columns in local_columns:
            local_context.append(context)
            local_alpha.append(alpha_matrix[:, columns])
            local_reward.append(reward_matrix[:, columns])
            local_radius.append(np.full(len(context), radius, np.float32))
            local_anchor.append(deterministic[:, 0])
        local_replay.add(
            np.concatenate(local_context), np.concatenate(local_alpha),
            np.concatenate(local_reward), np.concatenate(local_radius),
            np.concatenate(local_anchor), iteration,
        )
    summary = {
        "iteration": iteration,
        "unique_context_count": len(context),
        "transition_count": len(alpha),
        "alpha": distribution(alpha),
        "reward": distribution(reward),
        "deterministic_reward": distribution(deterministic_reward),
        "negative_reward_fraction": float(np.mean(reward < 0.0)),
        "deterministic_negative_fraction": float(np.mean(deterministic_reward < 0.0)),
        "zero_action_fraction": float(np.mean(alpha == 0.0)),
        "local_probe_group_count": len(context) * len(local_columns),
        "local_probe_radii": list(args.local_probe_radii),
        "local_probe_boundary_mode": args.local_probe_boundary_mode,
        "context_sampling": context_sampling,
        "local_probe_replay_size": local_replay.size,
        "replay_size": replay.size,
    }
    return summary, (repeated_context, alpha, reward)


@torch.no_grad()
def critic_batch_metrics(
    q1, q2, tensors, extra,
    context: np.ndarray, alpha: np.ndarray, reward: np.ndarray,
    args: argparse.Namespace, device: torch.device,
) -> dict[str, float]:
    q1.eval()
    q2.eval()
    prediction = []
    for start in range(0, len(context), args.evaluation_batch_size):
        absolute = torch.from_numpy(
            context[start:start + args.evaluation_batch_size]
        ).to(device)
        one_alpha = torch.from_numpy(
            alpha[start:start + args.evaluation_batch_size]
        ).to(device)
        prediction.append(torch.minimum(
            critic_one(q1, tensors, extra, absolute, one_alpha),
            critic_one(q2, tensors, extra, absolute, one_alpha),
        ).cpu().numpy())
    prediction = np.concatenate(prediction)
    target = np.arcsinh(reward / args.reward_scale)
    negative = reward < 0.0
    return {
        "transformed_reward_rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "transformed_reward_mae": float(np.mean(np.abs(prediction - target))),
        "reward_sign_accuracy": float(np.mean(np.sign(prediction) == np.sign(target))),
        "negative_sign_accuracy": (
            float(np.mean(prediction[negative] < 0.0)) if np.any(negative) else 1.0
        ),
        "predicted_negative_mean": (
            float(np.mean(prediction[negative])) if np.any(negative) else 0.0
        ),
        "target_negative_mean": (
            float(np.mean(target[negative])) if np.any(negative) else 0.0
        ),
    }


@torch.no_grad()
def critic_local_probe_metrics(
    policy, q1, q2, data, tensors, extra, index: np.ndarray,
    threshold: float, args: argparse.Namespace, device: torch.device,
) -> dict[str, Any] | None:
    """Evaluate Q direction around the frozen Actor using direct rollout labels."""
    local_probe_radii = tuple(getattr(args, "local_probe_radii", ()))
    if not local_probe_radii:
        return None
    _, _, center_alpha, _ = deterministic_outputs(
        policy, tensors, extra, index, threshold,
        args.evaluation_batch_size, device,
    )
    context = np.repeat(index, len(local_probe_radii))
    center = np.repeat(center_alpha, len(local_probe_radii))
    radius = np.tile(np.asarray(local_probe_radii, np.float32), len(index))
    alpha = local_probe_triplet(
        center, radius,
        getattr(args, "local_probe_boundary_mode", "clip"),
    )
    flat_context = np.repeat(context, 3)
    flat_alpha = alpha.reshape(-1)
    raw = (
        data.old_center[flat_context]
        + flat_alpha[:, None, None]
        * data.sigma[flat_context, None, :]
        * data.projected_direction[flat_context]
    )
    centers = np.clip(raw, -1.0, 1.0).astype(np.float32)
    cost = direct_cost(
        centers, data, tensors, flat_context,
        args.evaluation_batch_size, device,
    ).reshape(-1, 3)
    reward = data.old_cost[context, None] - cost
    target = np.arcsinh(reward / args.reward_scale)
    prediction = []
    q1.eval()
    q2.eval()
    for start in range(0, len(context), args.evaluation_batch_size):
        absolute = torch.from_numpy(
            context[start:start + args.evaluation_batch_size]
        ).to(device)
        one_alpha = torch.from_numpy(
            alpha[start:start + args.evaluation_batch_size]
        ).to(device)
        prediction.append(torch.minimum(
            critic_bank(q1, tensors, extra, absolute, one_alpha),
            critic_bank(q2, tensors, extra, absolute, one_alpha),
        ).cpu().numpy())
    prediction = np.concatenate(prediction)
    target_delta = np.diff(target, axis=1)
    prediction_delta = np.diff(prediction, axis=1)
    valid = np.diff(alpha, axis=1) > 1e-7
    meaningful = valid & (
        np.abs(target_delta) >= args.critic_local_sign_threshold
    )
    sign_accuracy = (
        float(np.mean(
            np.sign(prediction_delta[meaningful])
            == np.sign(target_delta[meaningful])
        )) if np.any(meaningful) else 1.0
    )
    central_target = target[:, 2] - target[:, 0]
    central_prediction = prediction[:, 2] - prediction[:, 0]
    central_meaningful = np.abs(central_target) >= args.critic_local_sign_threshold
    central_sign_accuracy = (
        float(np.mean(
            np.sign(central_prediction[central_meaningful])
            == np.sign(central_target[central_meaningful])
        )) if np.any(central_meaningful) else 1.0
    )
    predicted_index = np.argmax(prediction, axis=1)
    oracle_index = np.argmax(reward, axis=1)
    row = np.arange(len(context))
    regret = reward[row, oracle_index] - reward[row, predicted_index]
    rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
    return {
        "selection_score": float(
            np.mean(regret) + 0.1 * rmse
            + 0.1 * (1.0 - central_sign_accuracy)
            + 0.05 * (1.0 - sign_accuracy)
        ),
        "group_count": len(context),
        "radii": list(local_probe_radii),
        "transformed_reward_rmse": rmse,
        "transformed_reward_mae": float(np.mean(np.abs(prediction - target))),
        "adjacent_sign_accuracy": sign_accuracy,
        "central_sign_accuracy": central_sign_accuracy,
        "predicted_argmax_regret": distribution(regret),
    }


def update_critics(
    q1, q2, optimizer, replay: OnlineReplay, local_replay: LocalProbeReplay,
    data, tensors, extra, fit_index: np.ndarray,
    args: argparse.Namespace, rng: np.random.Generator,
    device: torch.device,
) -> dict[str, float]:
    q1.train()
    q2.train()
    if args.critic_trainable_scope == "action_heads":
        for critic in (q1, q2):
            critic.encoder.eval()
            critic.direction_encoder.eval()
            critic.state_fusion.eval()
    history: dict[str, list[float]] = {}

    def record(**values: torch.Tensor) -> None:
        for name, value in values.items():
            history.setdefault(name, []).append(float(value.detach()))

    online_count = min(
        int(round(args.batch_size * args.online_replay_fraction)), args.batch_size
    )
    for _ in range(args.critic_updates_per_iteration):
        if args.critic_objective == "separated_curve":
            base_context = rng.choice(
                fit_index, size=args.critic_base_context_batch_size,
                replace=True,
            )
            base_absolute = torch.from_numpy(base_context).to(device)
            base_alpha = tensors["alpha_grid"][base_absolute]
            base_target = transform_reward(
                tensors["old_cost"][base_absolute, None]
                - tensors["direct_line_cost"][base_absolute],
                args.reward_scale,
            )
            base_prediction1 = critic_bank(
                q1, tensors, extra, base_absolute, base_alpha
            )
            base_prediction2 = critic_bank(
                q2, tensors, extra, base_absolute, base_alpha
            )
            base_value_loss = (
                F.smooth_l1_loss(
                    base_prediction1, base_target,
                    beta=args.critic_huber_beta,
                )
                + F.smooth_l1_loss(
                    base_prediction2, base_target,
                    beta=args.critic_huber_beta,
                )
            )
            base_delta = base_target[:, 1:] - base_target[:, :-1]
            base_delta_loss = (
                F.smooth_l1_loss(
                    base_prediction1[:, 1:] - base_prediction1[:, :-1],
                    base_delta, beta=args.critic_huber_beta,
                )
                + F.smooth_l1_loss(
                    base_prediction2[:, 1:] - base_prediction2[:, :-1],
                    base_delta, beta=args.critic_huber_beta,
                )
            )
            anchor_loss = (
                base_prediction1[:, 0].square().mean()
                + base_prediction2[:, 0].square().mean()
            )

            online_context, online_alpha, online_reward = replay.sample(
                args.batch_size, rng, args.negative_replay_boost
            )
            if len(online_context):
                online_absolute = torch.from_numpy(online_context).to(device)
                online_action = torch.from_numpy(online_alpha).to(device)
                online_target = transform_reward(
                    torch.from_numpy(online_reward).to(device),
                    args.reward_scale,
                )
                online_prediction1 = critic_one(
                    q1, tensors, extra, online_absolute, online_action
                )
                online_prediction2 = critic_one(
                    q2, tensors, extra, online_absolute, online_action
                )
                online_zero = torch.zeros_like(online_action)
                online_anchor1 = critic_one(
                    q1, tensors, extra, online_absolute, online_zero
                )
                online_anchor2 = critic_one(
                    q2, tensors, extra, online_absolute, online_zero
                )
                online_weight = torch.where(
                    online_target < 0.0,
                    torch.full_like(
                        online_target, args.critic_negative_loss_weight
                    ),
                    torch.ones_like(online_target),
                )
                online_value = (
                    F.smooth_l1_loss(
                        online_prediction1, online_target,
                        beta=args.critic_huber_beta, reduction="none",
                    )
                    + F.smooth_l1_loss(
                        online_prediction2, online_target,
                        beta=args.critic_huber_beta, reduction="none",
                    )
                )
                online_value_loss = (
                    torch.sum(online_value * online_weight)
                    / torch.sum(online_weight)
                )
                online_delta = (
                    F.smooth_l1_loss(
                        online_prediction1 - online_anchor1, online_target,
                        beta=args.critic_huber_beta, reduction="none",
                    )
                    + F.smooth_l1_loss(
                        online_prediction2 - online_anchor2, online_target,
                        beta=args.critic_huber_beta, reduction="none",
                    )
                )
                online_delta_loss = (
                    torch.sum(online_delta * online_weight)
                    / torch.sum(online_weight)
                )
            else:
                online_value_loss = torch.zeros((), device=device)
                online_delta_loss = torch.zeros((), device=device)

            local_context, local_alpha, local_reward = local_replay.sample(
                args.local_probe_batch_size, rng
            )
            if len(local_context):
                local_absolute = torch.from_numpy(local_context).to(device)
                local_action = torch.from_numpy(local_alpha).to(device)
                local_target = transform_reward(
                    torch.from_numpy(local_reward).to(device), args.reward_scale
                )
                local_prediction1 = critic_bank(
                    q1, tensors, extra, local_absolute, local_action
                )
                local_prediction2 = critic_bank(
                    q2, tensors, extra, local_absolute, local_action
                )
                local_target_delta = local_target[:, 1:] - local_target[:, :-1]
                local_prediction_delta1 = (
                    local_prediction1[:, 1:] - local_prediction1[:, :-1]
                )
                local_prediction_delta2 = (
                    local_prediction2[:, 1:] - local_prediction2[:, :-1]
                )
                local_valid = (
                    local_action[:, 1:] - local_action[:, :-1]
                ).abs() > 1e-7
                local_delta_raw = (
                    F.smooth_l1_loss(
                        local_prediction_delta1, local_target_delta,
                        beta=args.critic_huber_beta, reduction="none",
                    )
                    + F.smooth_l1_loss(
                        local_prediction_delta2, local_target_delta,
                        beta=args.critic_huber_beta, reduction="none",
                    )
                )
                local_delta_loss = local_delta_raw[local_valid].mean()
                meaningful = (
                    local_valid
                    & (local_target_delta.abs() >= args.critic_local_sign_threshold)
                )
                if torch.any(meaningful):
                    local_sign = torch.sign(local_target_delta)
                    local_sign_raw = (
                        F.softplus(
                            -local_sign * local_prediction_delta1
                            / args.critic_local_sign_temperature
                        )
                        + F.softplus(
                            -local_sign * local_prediction_delta2
                            / args.critic_local_sign_temperature
                        )
                    )
                    local_sign_loss = local_sign_raw[meaningful].mean()
                else:
                    local_sign_loss = torch.zeros((), device=device)
            else:
                local_delta_loss = torch.zeros((), device=device)
                local_sign_loss = torch.zeros((), device=device)

            loss = (
                args.critic_base_value_weight * base_value_loss
                + args.critic_base_delta_weight * base_delta_loss
                + args.critic_anchor_weight * anchor_loss
                + args.critic_online_value_weight * online_value_loss
                + args.critic_online_delta_weight * online_delta_loss
                + args.critic_local_delta_weight * local_delta_loss
                + args.critic_local_sign_weight * local_sign_loss
            )
            components = {
                "total": loss,
                "base_value": base_value_loss,
                "base_delta": base_delta_loss,
                "anchor": anchor_loss,
                "online_value": online_value_loss,
                "online_delta": online_delta_loss,
                "local_delta": local_delta_loss,
                "local_sign": local_sign_loss,
            }
        else:
            online_context, online_alpha, online_reward = replay.sample(
                online_count, rng, args.negative_replay_boost
            )
            base_count = args.batch_size - len(online_context)
            base_context = rng.choice(fit_index, size=base_count, replace=True)
            grid_index = rng.integers(0, data.alpha_grid.shape[1], size=base_count)
            base_alpha = data.alpha_grid[base_context, grid_index]
            base_reward = (
                data.old_cost[base_context]
                - data.direct_line_cost[base_context, grid_index]
            )
            context = np.concatenate((base_context, online_context)).astype(np.int64)
            alpha = np.concatenate((base_alpha, online_alpha)).astype(np.float32)
            reward = np.concatenate((base_reward, online_reward)).astype(np.float32)
            order = rng.permutation(len(context))
            absolute = torch.from_numpy(context[order]).to(device)
            action = torch.from_numpy(alpha[order]).to(device)
            target = transform_reward(
                torch.from_numpy(reward[order]).to(device), args.reward_scale
            )
            prediction1 = critic_one(q1, tensors, extra, absolute, action)
            prediction2 = critic_one(q2, tensors, extra, absolute, action)
            zero = torch.zeros_like(action)
            anchor1 = critic_one(q1, tensors, extra, absolute, zero)
            anchor2 = critic_one(q2, tensors, extra, absolute, zero)
            sample_weight = torch.where(
                target < 0.0,
                torch.full_like(target, args.critic_negative_loss_weight),
                torch.ones_like(target),
            )
            value_loss = (
                F.smooth_l1_loss(
                    prediction1, target, beta=args.critic_huber_beta,
                    reduction="none",
                )
                + F.smooth_l1_loss(
                    prediction2, target, beta=args.critic_huber_beta,
                    reduction="none",
                )
            )
            value_loss = torch.sum(value_loss * sample_weight) / torch.sum(sample_weight)
            delta_loss = (
                F.smooth_l1_loss(
                    prediction1 - anchor1, target,
                    beta=args.critic_huber_beta, reduction="none",
                )
                + F.smooth_l1_loss(
                    prediction2 - anchor2, target,
                    beta=args.critic_huber_beta, reduction="none",
                )
            )
            delta_loss = torch.sum(delta_loss * sample_weight) / torch.sum(sample_weight)
            meaningful = target.abs() >= 0.01
            sign = torch.sign(target)
            sign_loss = (
                F.softplus(
                    -sign * (prediction1 - anchor1)
                    / args.critic_sign_temperature
                )
                + F.softplus(
                    -sign * (prediction2 - anchor2)
                    / args.critic_sign_temperature
                )
            )
            sign_loss = (
                torch.sum(sign_loss[meaningful] * sample_weight[meaningful])
                / torch.sum(sample_weight[meaningful])
                if torch.any(meaningful) else torch.zeros((), device=device)
            )
            anchor_loss = anchor1.square().mean() + anchor2.square().mean()
            loss = (
                value_loss
                + args.critic_delta_weight * delta_loss
                + args.critic_sign_weight * sign_loss
                + args.critic_anchor_weight * anchor_loss
            )
            components = {
                "total": loss,
                "mixed_value": value_loss,
                "mixed_delta": delta_loss,
                "mixed_sign": sign_loss,
                "anchor": anchor_loss,
            }
        if not torch.isfinite(loss):
            raise FloatingPointError("Critic loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(q1.parameters()) + list(q2.parameters()), 5.0
        )
        optimizer.step()
        record(**components)
    return {
        name: float(np.mean(values)) for name, values in history.items()
    }


def logit_normal_sample(
    mean: torch.Tensor, log_std: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    epsilon = torch.randn_like(mean)
    std = log_std.exp()
    latent = mean + std * epsilon
    alpha = torch.sigmoid(latent)
    normal_log_prob = (
        -0.5 * epsilon.square() - log_std - 0.5 * math.log(2.0 * math.pi)
    )
    log_probability = normal_log_prob - torch.log(
        alpha * (1.0 - alpha) + 1e-6
    )
    return alpha, log_probability


def update_actor(
    policy, q1, q2, optimizer,
    tensors, extra, fit_index: np.ndarray,
    initial_move: torch.Tensor, initial_conditional: torch.Tensor,
    bc_weight: float, args: argparse.Namespace,
    rng: np.random.Generator, device: torch.device,
) -> dict[str, float]:
    critic_parameters = list(q1.parameters()) + list(q2.parameters())
    critic_requires_grad = [
        parameter.requires_grad for parameter in critic_parameters
    ]
    for parameter in critic_parameters:
        parameter.requires_grad_(False)
    q1.eval()
    q2.eval()
    policy.train()
    if args.freeze_actor_context_encoder:
        policy.encoder.eval()
    losses, q_values, entropy_values = [], [], []
    for _ in range(args.actor_updates_per_iteration):
        context = rng.choice(fit_index, size=args.batch_size, replace=True)
        absolute = torch.from_numpy(context).to(device)
        move_logit, probability, mean, log_std = policy_batch(
            policy, tensors, extra, absolute
        )
        sample_mean = mean[:, None].expand(-1, args.actor_samples_per_update)
        sample_log_std = log_std[:, None].expand_as(sample_mean)
        alpha, log_probability = logit_normal_sample(
            sample_mean, sample_log_std
        )
        q_move1 = critic_bank(q1, tensors, extra, absolute, alpha)
        q_move2 = critic_bank(q2, tensors, extra, absolute, alpha)
        q_move = torch.minimum(q_move1, q_move2)
        zero = torch.zeros_like(mean)
        q_stay = torch.minimum(
            critic_one(q1, tensors, extra, absolute, zero),
            critic_one(q2, tensors, extra, absolute, zero),
        )
        log_move = torch.log(probability.clamp_min(1e-6))
        log_stay = torch.log((1.0 - probability).clamp_min(1e-6))
        stay_objective = (1.0 - probability) * (
            args.gate_entropy_temperature * log_stay - q_stay
        )
        conditional_objective = (
            args.alpha_entropy_temperature * log_probability
            - q_move
            + args.critic_disagreement_weight * (q_move1 - q_move2).abs()
        ).mean(1)
        move_objective = probability * (
            args.gate_entropy_temperature * log_move + conditional_objective
        )
        gate_bc = F.binary_cross_entropy_with_logits(
            move_logit, initial_move[absolute], reduction="none"
        )
        alpha_bc = F.smooth_l1_loss(
            torch.sigmoid(mean), initial_conditional[absolute],
            beta=0.02, reduction="none",
        )
        loss = (stay_objective + move_objective).mean()
        loss = loss + bc_weight * (gate_bc + alpha_bc).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Actor loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in policy.parameters() if parameter.requires_grad],
            2.0,
        )
        optimizer.step()
        losses.append(float(loss.detach()))
        q_values.append(float((
            probability * q_move.mean(1) + (1 - probability) * q_stay
        ).mean().detach()))
        entropy_values.append(float((-probability * log_move - (1 - probability) * log_stay).mean().detach()))
    for parameter, requires_grad in zip(
        critic_parameters, critic_requires_grad
    ):
        parameter.requires_grad_(requires_grad)
    return {
        "actor_loss": float(np.mean(losses)),
        "actor_expected_q": float(np.mean(q_values)),
        "gate_entropy": float(np.mean(entropy_values)),
    }


@torch.no_grad()
def initial_targets(
    policy, tensors, extra, threshold: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    probability, conditional = [], []
    policy.eval()
    for start in range(0, len(tensors["old_cost"]), 256):
        absolute = torch.arange(
            start, min(start + 256, len(tensors["old_cost"])), device=device
        )
        _, one_probability, mean, _ = policy_batch(
            policy, tensors, extra, absolute
        )
        probability.append(one_probability)
        conditional.append(torch.sigmoid(mean))
    probability = torch.cat(probability)
    return (probability > threshold).float(), torch.cat(conditional)


def grouped_metrics(
    policy, data, tensors, extra, index: np.ndarray,
    threshold: float, args: argparse.Namespace, device: torch.device,
) -> dict[str, Any]:
    result = {}
    for speed in sorted(np.unique(data.reference_speed[index])):
        selected = index[np.isclose(data.reference_speed[index], speed)]
        probability, conditional, _, centers = deterministic_outputs(
            policy, tensors, extra, selected, threshold,
            args.evaluation_batch_size, device,
        )
        move_cost = direct_cost(
            centers, data, tensors, selected,
            args.evaluation_batch_size, device,
        )
        metrics = metrics_from_cache(
            probability, conditional, move_cost, data, selected,
            threshold, args,
        )
        result[f"{float(speed):.1f}"] = {
            "context_count": len(selected),
            "policy_cost_mean": metrics["direct_cost"]["mean"],
            "gain_mean": metrics["gain_vs_old"]["mean"],
            "gain_p05": metrics["gain_vs_old"]["p05"],
            "gain_worst": metrics["gain_vs_old"]["minimum"],
            "move_fraction": metrics["move_fraction"],
        }
    return result


def checkpoint_payload(
    policy, q1, q2, initial_payload, args,
    iteration: int, threshold: float, metrics: dict[str, Any],
    fit_episodes: list[str], selection_episodes: list[str],
    replay: OnlineReplay, qualification: str,
    actor_optimizer_state: dict[str, Any],
    critic_optimizer_state: dict[str, Any],
    resume_metadata: dict[str, Any],
    posthoc_safe_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "method": "train-only continuously updated single-step hybrid Alpha SAC",
        "qualification": qualification,
        "policy_class": "TorchMPPITrustAlphaSACPolicy",
        "critic_class": "TorchMPPITrustAlphaCritic",
        "policy_state_dict": copy.deepcopy(policy.cpu().state_dict()),
        "critic1_state_dict": copy.deepcopy(q1.cpu().state_dict()),
        "critic2_state_dict": copy.deepcopy(q2.cpu().state_dict()),
        "actor_optimizer_state_dict": copy.deepcopy(actor_optimizer_state),
        "critic_optimizer_state_dict": copy.deepcopy(critic_optimizer_state),
        "state_normalization": initial_payload["state_normalization"],
        "feedback_mean": initial_payload["feedback_mean"],
        "feedback_std": initial_payload["feedback_std"],
        "gradient_mean": initial_payload["gradient_mean"],
        "gradient_std": initial_payload["gradient_std"],
        "old_actor": initial_payload["old_actor"],
        "old_actor_sha256": initial_payload["old_actor_sha256"],
        "proposal_actor": initial_payload["proposal_actor"],
        "proposal_actor_sha256": initial_payload["proposal_actor_sha256"],
        "initial_ac": str(args.initial_ac.resolve()),
        "initial_ac_sha256": sha256_file(args.initial_ac),
        "labels": str(args.labels.resolve()),
        "labels_hashes": {
            name: sha256_file(args.labels / name)
            for name in ("config.json", "splits.json", "summary.json")
        },
        "move_threshold": float(threshold),
        "selection_mode": args.selection_mode,
        "reward_transform": f"asinh(reward/{args.reward_scale})",
        "terminal_transition": True,
        "alpha_logit_scale": float(args.alpha_logit_scale),
        "selected_iteration": int(iteration),
        "fit_episodes": fit_episodes,
        "selection_episodes": selection_episodes,
        "online_replay_size": replay.size,
        **resume_metadata,
        "internal_selection_metrics": metrics,
        "posthoc_safe_internal_selection_metrics": posthoc_safe_metrics,
        "training_arguments": serialized_arguments(args),
        "test_policy": "formal validation and test not loaded or evaluated",
    }


def main() -> None:
    args = parse_args()
    args.local_probe_radii = parse_local_probe_radii(args.local_probe_radii)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 0.0 <= args.online_replay_fraction <= 1.0:
        raise ValueError("online replay fraction must be within [0,1]")
    if not 0.0 <= args.priority_context_fraction <= 1.0:
        raise ValueError("priority context fraction must be within [0,1]")
    if args.priority_candidate_multiplier < 1:
        raise ValueError("priority candidate multiplier must be at least one")
    priority_weights = (
        args.priority_disagreement_weight, args.priority_gate_weight,
        args.priority_cost_weight, args.priority_boundary_weight,
    )
    if any(value < 0.0 for value in priority_weights):
        raise ValueError("priority weights must be nonnegative")
    if args.context_sampling_mode == "critic_priority_mixed" and not any(
        priority_weights
    ):
        raise ValueError("critic-priority sampling requires a positive weight")
    if args.critic_selection_objective == "local_probe_guarded" and not args.local_probe_radii:
        raise ValueError("local_probe_guarded selection requires --local-probe-radii")
    if (
        (args.critic_local_delta_weight or args.critic_local_sign_weight)
        and args.critic_objective != "separated_curve"
    ):
        raise ValueError("local probe losses require --critic-objective separated_curve")
    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed + 510013)
    device = torch.device(args.device)

    resume_payload = None
    resume_summary = None
    resume_checkpoint = None
    resume_run = args.resume_run.resolve() if args.resume_run else None
    if resume_run is not None:
        latest_training_checkpoint = (
            resume_run / "online_alpha_sac_training_state.pt"
        )
        resume_checkpoint = (
            latest_training_checkpoint
            if (
                args.resume_checkpoint_role == "latest"
                and latest_training_checkpoint.is_file()
            )
            else resume_run / "online_alpha_sac_selected.pt"
        )
        resume_replay = resume_run / "actor_visited_replay.npz"
        if not resume_checkpoint.is_file() or not resume_replay.is_file():
            raise FileNotFoundError(
                "resume run must contain online_alpha_sac_selected.pt and "
                "actor_visited_replay.npz"
            )
        resume_payload = torch.load(resume_checkpoint, map_location="cpu")
        if resume_payload.get("policy_class") != "TorchMPPITrustAlphaSACPolicy":
            raise AssertionError("resume checkpoint is not an online Alpha SAC policy")
        args.initial_ac = Path(resume_payload["initial_ac"])
        if abs(float(resume_payload.get("alpha_logit_scale", 1.0)) - args.alpha_logit_scale) > 1e-12:
            raise AssertionError(
                "--alpha-logit-scale must match the resumed policy checkpoint"
            )
        summary_path = resume_run / "training_summary.json"
        if summary_path.is_file():
            resume_summary = json.loads(summary_path.read_text())

    initial_payload = torch.load(args.initial_ac, map_location="cpu")
    if initial_payload.get("policy_class") != "TorchMPPITrustAlphaPolicy":
        raise AssertionError("initial checkpoint is not a deterministic Alpha policy")
    if args.labels.resolve() != Path(initial_payload["labels"]).resolve():
        raise AssertionError("initial Alpha AC and requested TR1 labels differ")
    if sha256_file(args.labels / "summary.json") != initial_payload["labels_hashes"]["summary.json"]:
        raise AssertionError("TR1 label hash changed")
    old_payload = load_actor_payload(Path(initial_payload["old_actor"]))
    data, _, splits = load_dataset(args.labels, old_payload, args.max_snapshots)
    if args.max_snapshots:
        present = sorted(set(data.episodes.tolist()))
        fit_episodes = selection_episodes = present
    else:
        fit_episodes = list(splits["internal_fit"])
        selection_episodes = list(splits["internal_selection"])
    fit_index = np.flatnonzero(np.isin(data.episodes, fit_episodes))
    selection_index = np.flatnonzero(np.isin(data.episodes, selection_episodes))
    if not args.max_snapshots and len(fit_index) + len(selection_index) != len(data.episodes):
        raise AssertionError("internal split does not cover TR1 contexts")
    if not args.max_snapshots and set(fit_episodes) & set(selection_episodes):
        raise AssertionError("fit and internal-selection episodes overlap")

    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    policy = make_policy(
        initial_payload["policy_state_dict"], args, device, dropout=0.05
    )
    q1 = TorchMPPITrustAlphaCritic(dropout=0.05).to(device)
    q2 = TorchMPPITrustAlphaCritic(dropout=0.05).to(device)
    q1.load_state_dict(initial_payload["critic1_state_dict"], strict=True)
    q2.load_state_dict(initial_payload["critic2_state_dict"], strict=True)
    if resume_payload is not None:
        policy.load_state_dict(resume_payload["policy_state_dict"], strict=True)
        q1.load_state_dict(resume_payload["critic1_state_dict"], strict=True)
        q2.load_state_dict(resume_payload["critic2_state_dict"], strict=True)
    if args.critic_trainable_scope == "action_heads":
        for critic in (q1, q2):
            for module in (
                critic.encoder, critic.direction_encoder, critic.state_fusion
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
    if args.freeze_actor_context_encoder:
        for parameter in policy.encoder.parameters():
            parameter.requires_grad_(False)
    actor_parameters = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    actor_optimizer = torch.optim.AdamW(
        actor_parameters, lr=args.actor_learning_rate,
        weight_decay=args.weight_decay,
    )
    critic_optimizer = torch.optim.AdamW(
        list(q1.parameters()) + list(q2.parameters()),
        lr=args.critic_learning_rate, weight_decay=args.weight_decay,
    )
    optimizer_state_restored = False
    if resume_payload is not None and all(
        key in resume_payload for key in (
            "actor_optimizer_state_dict", "critic_optimizer_state_dict"
        )
    ):
        actor_optimizer.load_state_dict(resume_payload["actor_optimizer_state_dict"])
        critic_optimizer.load_state_dict(resume_payload["critic_optimizer_state_dict"])
        optimizer_state_restored = True
    # Optimizer.load_state_dict also restores historical param-group hyperparameters.
    # Reapply the explicit command-line contract so resumed experiments use the
    # requested learning rates instead of silently inheriting an older run.
    for group in actor_optimizer.param_groups:
        group["lr"] = args.actor_learning_rate
        group["weight_decay"] = args.weight_decay
    for group in critic_optimizer.param_groups:
        group["lr"] = args.critic_learning_rate
        group["weight_decay"] = args.weight_decay
    initial_threshold = float(initial_payload["move_threshold"])
    base_policy = make_policy(
        initial_payload["policy_state_dict"], args, device, dropout=0.0
    )
    initial_move, initial_conditional = initial_targets(
        base_policy, tensors, extra, initial_threshold, device
    )
    del base_policy
    replay = OnlineReplay(args.replay_capacity)
    local_replay = LocalProbeReplay(args.local_probe_capacity)
    start_iteration = 0
    if resume_run is not None:
        with np.load(resume_run / "actor_visited_replay.npz", allow_pickle=False) as saved:
            replay.load_arrays({name: saved[name] for name in (
                "context_index", "alpha", "reward", "iteration"
            )})
        if replay.size:
            start_iteration = int(np.max(replay.arrays()["iteration"]))
        local_replay_path = resume_run / "actor_local_probe_replay.npz"
        if local_replay_path.is_file() and not args.reset_local_probe_replay:
            with np.load(local_replay_path, allow_pickle=False) as saved:
                local_arrays = {
                    name: saved[name] for name in (
                        "context_index", "alpha", "reward", "radius", "iteration"
                    )
                }
                if "anchor_alpha" in saved.files:
                    local_arrays["anchor_alpha"] = saved["anchor_alpha"]
                local_replay.load_arrays(local_arrays)

    initial_metrics = calibrate_and_evaluate(
        policy, data, tensors, extra, selection_index, args, device
    )
    if initial_metrics is None:
        raise AssertionError("initial Alpha AC no longer passes internal selection")
    initial_metrics.update({"iteration": start_iteration})
    initial_critic_metrics = critic_metrics(
        q1, q2, data, tensors, extra,
        selection_index, args, device,
    )
    initial_local_critic_metrics = critic_local_probe_metrics(
        policy, q1, q2, data, tensors, extra, selection_index,
        float(initial_metrics["move_threshold"]), args, device,
    )
    history = [initial_metrics]
    latest_metrics = copy.deepcopy(initial_metrics)
    best_iteration = start_iteration
    best_metrics = copy.deepcopy(initial_metrics)
    best_policy = copy.deepcopy(policy.state_dict())
    best_q1 = copy.deepcopy(q1.state_dict())
    best_q2 = copy.deepcopy(q2.state_dict())
    best_actor_optimizer = copy.deepcopy(actor_optimizer.state_dict())
    best_critic_optimizer = copy.deepcopy(critic_optimizer.state_dict())
    best_critic_iteration = start_iteration
    best_critic_metrics = copy.deepcopy(initial_critic_metrics)
    best_local_critic_metrics = copy.deepcopy(initial_local_critic_metrics)
    best_critic_score = float(
        initial_local_critic_metrics["selection_score"]
        if args.critic_selection_objective == "local_probe_guarded"
        else initial_critic_metrics["selection_score"]
    )
    best_cost = float(initial_metrics["direct_cost"]["mean"])
    collection_history = []
    critic_update_history = []
    critic_rank_history = [{
        "iteration": start_iteration,
        **copy.deepcopy(initial_critic_metrics),
    }]
    critic_local_rank_history = ([{
        "iteration": start_iteration,
        **copy.deepcopy(initial_local_critic_metrics),
    }] if initial_local_critic_metrics is not None else [])
    print(
        f"[iteration={start_iteration:03d}] cost={best_cost:.4f} "
        f"threshold={initial_metrics['move_threshold']:.3f} "
        f"move={initial_metrics['move_fraction']:.3f}",
        flush=True,
    )

    for local_iteration in range(1, args.iterations + 1):
        iteration = start_iteration + local_iteration
        active_threshold = float(best_metrics["move_threshold"])
        if args.collect_transitions:
            collection, new_batch = collect_actor_visited(
                policy, q1, q2, replay, local_replay,
                data, tensors, extra, fit_index,
                active_threshold, iteration, args, rng, device,
            )
        else:
            if replay.size == 0:
                raise AssertionError(
                    "fixed-replay training requires a nonempty resumed replay"
                )
            diagnostic_count = max(args.batch_size, 1024)
            new_batch = replay.sample(
                diagnostic_count, rng, args.negative_replay_boost
            )
            collection = {
                "iteration": iteration,
                "collection_disabled": True,
                "transition_count": 0,
                "replay_size": replay.size,
            }
        before = critic_batch_metrics(
            q1, q2, tensors, extra, *new_batch, args, device
        )
        critic_losses = update_critics(
            q1, q2, critic_optimizer, replay, local_replay,
            data, tensors, extra, fit_index, args, rng, device,
        )
        critic_loss = critic_losses["total"]
        after = critic_batch_metrics(
            q1, q2, tensors, extra, *new_batch, args, device
        )
        progress = (local_iteration - 1) / max(args.iterations - 1, 1)
        bc_weight = (
            args.actor_bc_weight_start * (1.0 - progress)
            + args.actor_bc_weight_end * progress
        )
        if (
            args.training_mode == "critic_only"
            or local_iteration <= args.critic_warmup_iterations
        ):
            actor_update = {
                "actor_updated": False,
                "actor_loss": None,
                "actor_expected_q": None,
                "gate_entropy": None,
            }
        else:
            actor_update = {
                "actor_updated": True,
                **update_actor(
                    policy, q1, q2, actor_optimizer,
                    tensors, extra, fit_index,
                    initial_move, initial_conditional,
                    bc_weight, args, rng, device,
                ),
            }
        assert_finite_module(policy, "policy")
        assert_finite_module(q1, "critic1")
        assert_finite_module(q2, "critic2")
        collection_history.append(collection)
        critic_update_history.append({
            "iteration": iteration,
            "critic_loss": critic_loss,
            "critic_loss_components": critic_losses,
            "new_batch_before_update": before,
            "new_batch_after_update": after,
            "bc_weight": bc_weight,
            **actor_update,
        })
        if (
            args.critic_rank_evaluation_interval > 0
            and (
                local_iteration % args.critic_rank_evaluation_interval == 0
                or local_iteration == args.iterations
            )
        ):
            rank_metrics = critic_metrics(
                q1, q2, data, tensors, extra,
                selection_index, args, device,
            )
            critic_rank_history.append({
                "iteration": iteration,
                **rank_metrics,
            })
            local_rank_metrics = critic_local_probe_metrics(
                policy, q1, q2, data, tensors, extra, selection_index,
                float(best_metrics["move_threshold"]), args, device,
            )
            if local_rank_metrics is not None:
                critic_local_rank_history.append({
                    "iteration": iteration,
                    **local_rank_metrics,
                })
            if args.critic_selection_objective == "local_probe_guarded":
                rank_score = float(local_rank_metrics["selection_score"])
                rank_eligible = (
                    float(rank_metrics["selection_score"])
                    <= float(initial_critic_metrics["selection_score"])
                    + args.critic_full_line_score_tolerance
                )
            else:
                rank_score = float(rank_metrics["selection_score"])
                rank_eligible = True
            if (
                rank_eligible
                and rank_score
                < best_critic_score - args.critic_min_selection_improvement
            ):
                best_critic_score = rank_score
                best_critic_iteration = iteration
                best_critic_metrics = copy.deepcopy(rank_metrics)
                best_local_critic_metrics = copy.deepcopy(local_rank_metrics)
                best_q1 = copy.deepcopy(q1.state_dict())
                best_q2 = copy.deepcopy(q2.state_dict())
                best_critic_optimizer = copy.deepcopy(
                    critic_optimizer.state_dict()
                )

        if iteration % args.evaluation_interval == 0 or iteration == args.iterations:
            metrics = calibrate_and_evaluate(
                policy, data, tensors, extra, selection_index, args, device
            )
            if metrics is None:
                print(
                    f"[iteration={iteration:03d}] no internal threshold passes; "
                    f"replay={replay.size}", flush=True,
                )
                continue
            metrics.update({
                "iteration": iteration,
                "replay_size": replay.size,
                "critic_loss": critic_loss,
                "critic_loss_components": critic_losses,
                "new_batch_before_update": before,
                "new_batch_after_update": after,
                "bc_weight": bc_weight,
                **actor_update,
            })
            latest_metrics = copy.deepcopy(metrics)
            history.append(metrics)
            cost = float(metrics["direct_cost"]["mean"])
            eligible = (
                metrics["pass"]
                if args.selection_mode == "safe_gate"
                else np.isfinite(cost)
            )
            if (
                args.training_mode == "actor_critic"
                and eligible and cost < best_cost
            ):
                best_cost = cost
                best_iteration = iteration
                best_metrics = copy.deepcopy(metrics)
                best_policy = copy.deepcopy(policy.state_dict())
                best_q1 = copy.deepcopy(q1.state_dict())
                best_q2 = copy.deepcopy(q2.state_dict())
                best_actor_optimizer = copy.deepcopy(actor_optimizer.state_dict())
                best_critic_optimizer = copy.deepcopy(critic_optimizer.state_dict())
            gain = metrics["gain_vs_old"]
            print(
                f"[iteration={iteration:03d}] cost={cost:.4f} "
                f"gain={gain['mean']:.3f} p05={gain['p05']:.3f} "
                f"worst={gain['minimum']:.3f} threshold={metrics['move_threshold']:.3f} "
                f"move={metrics['move_fraction']:.3f} replay={replay.size} "
                f"new-rmse={before['transformed_reward_rmse']:.3f}->"
                f"{after['transformed_reward_rmse']:.3f}",
                flush=True,
            )

    latest_iteration = start_iteration + args.iterations
    latest_policy = copy.deepcopy(policy.state_dict())
    latest_q1 = copy.deepcopy(q1.state_dict())
    latest_q2 = copy.deepcopy(q2.state_dict())
    latest_actor_optimizer = copy.deepcopy(actor_optimizer.state_dict())
    latest_critic_optimizer = copy.deepcopy(critic_optimizer.state_dict())
    latest_critic_selection = critic_metrics(
        q1, q2, data, tensors, extra,
        selection_index, args, device,
    )
    latest_local_critic_selection = critic_local_probe_metrics(
        policy, q1, q2, data, tensors, extra, selection_index,
        float(latest_metrics["move_threshold"]), args, device,
    )
    safe_args = copy.copy(args)
    safe_args.selection_mode = "safe_gate"
    latest_posthoc_safe_metrics = calibrate_and_evaluate(
        policy, data, tensors, extra, selection_index, safe_args, device
    )

    policy.load_state_dict(best_policy, strict=True)
    q1.load_state_dict(best_q1, strict=True)
    q2.load_state_dict(best_q2, strict=True)
    actor_optimizer.load_state_dict(best_actor_optimizer)
    critic_optimizer.load_state_dict(best_critic_optimizer)
    policy.eval()
    q1.eval()
    q2.eval()
    best_threshold = float(best_metrics["move_threshold"])
    speed = grouped_metrics(
        policy, data, tensors, extra, selection_index,
        best_threshold, args, device,
    )
    critic_selection = critic_metrics(
        q1, q2, data, tensors, extra,
        selection_index, args, device,
    )
    local_critic_selection = critic_local_probe_metrics(
        policy, q1, q2, data, tensors, extra, selection_index,
        best_threshold, args, device,
    )
    improved = best_cost < float(initial_metrics["direct_cost"]["mean"]) - 1e-6
    speed_pass = all(row["gain_mean"] >= 0.0 for row in speed.values())
    if args.training_mode == "critic_only":
        initial_selection_score = float(
            initial_local_critic_metrics["selection_score"]
            if args.critic_selection_objective == "local_probe_guarded"
            else initial_critic_metrics["selection_score"]
        )
        improved = (
            best_critic_score
            < initial_selection_score - args.critic_min_selection_improvement
        )
        qualification = (
            "ONLINE_ALPHA_CRITIC_ONLY_LOCAL_RANK_IMPROVED"
            if improved and args.critic_selection_objective == "local_probe_guarded"
            else "ONLINE_ALPHA_CRITIC_ONLY_RANK_IMPROVED"
            if improved else "ONLINE_ALPHA_CRITIC_ONLY_RANK_PLATEAU"
        )
        best_iteration = best_critic_iteration
    elif args.selection_mode == "aggressive_mean":
        qualification = (
            "ONLINE_ALPHA_SAC_AGGRESSIVE_MEAN_IMPROVED"
            if improved else "ONLINE_ALPHA_SAC_AGGRESSIVE_MEAN_PLATEAU"
        )
    else:
        qualification = (
            "ONLINE_ALPHA_SAC_INTERNAL_PASS"
            if best_metrics["pass"] and improved and speed_pass
            else "ONLINE_ALPHA_SAC_MECHANISM_ONLY"
        )
    posthoc_safe_metrics = calibrate_and_evaluate(
        policy, data, tensors, extra, selection_index, safe_args, device
    )
    resume_metadata = {
        "resumed_from_run": str(resume_run) if resume_run is not None else None,
        "resumed_from_checkpoint_sha256": (
            sha256_file(resume_checkpoint)
            if resume_checkpoint is not None else None
        ),
        "resumed_from_checkpoint": (
            str(resume_checkpoint) if resume_checkpoint is not None else None
        ),
        "resume_start_iteration": start_iteration,
        "optimizer_state_restored": optimizer_state_restored,
    }
    checkpoint = checkpoint_payload(
        policy, q1, q2, initial_payload, args,
        best_iteration, best_threshold, best_metrics,
        fit_episodes, selection_episodes, replay, qualification,
        actor_optimizer.state_dict(), critic_optimizer.state_dict(),
        resume_metadata, posthoc_safe_metrics,
    )
    checkpoint["checkpoint_role"] = (
        "selected_critic_with_frozen_actor"
        if args.training_mode == "critic_only"
        else "selected_policy"
    )
    checkpoint.update({
        "training_mode": args.training_mode,
        "selected_critic_iteration": best_critic_iteration,
        "initial_critic_internal_selection": initial_critic_metrics,
        "selected_critic_internal_selection": best_critic_metrics,
        "initial_local_critic_internal_selection": initial_local_critic_metrics,
        "selected_local_critic_internal_selection": best_local_critic_metrics,
        "effective_actor_learning_rate": actor_optimizer.param_groups[0]["lr"],
        "effective_critic_learning_rate": critic_optimizer.param_groups[0]["lr"],
    })
    checkpoint_path = args.output_dir / "online_alpha_sac_selected.pt"
    torch.save(checkpoint, checkpoint_path)
    training_state = copy.deepcopy(checkpoint)
    training_state.update({
        "checkpoint_role": "latest_training_state",
        "qualification": "ONLINE_ALPHA_SAC_RESUME_STATE_NOT_POLICY_SELECTION",
        "policy_state_dict": latest_policy,
        "critic1_state_dict": latest_q1,
        "critic2_state_dict": latest_q2,
        "actor_optimizer_state_dict": latest_actor_optimizer,
        "critic_optimizer_state_dict": latest_critic_optimizer,
        "move_threshold": float(latest_metrics["move_threshold"]),
        "selected_iteration": int(latest_iteration),
        "internal_selection_metrics": latest_metrics,
        "posthoc_safe_internal_selection_metrics": latest_posthoc_safe_metrics,
    })
    training_state_path = args.output_dir / "online_alpha_sac_training_state.pt"
    torch.save(training_state, training_state_path)
    replay_arrays = replay.arrays()
    np.savez_compressed(
        args.output_dir / "actor_visited_replay.npz",
        **replay_arrays,
        episode=data.episodes[replay_arrays["context_index"]],
        reference_speed=data.reference_speed[replay_arrays["context_index"]],
    )
    local_replay_arrays = local_replay.arrays()
    if local_replay.size:
        np.savez_compressed(
            args.output_dir / "actor_local_probe_replay.npz",
            **local_replay_arrays,
            episode=data.episodes[local_replay_arrays["context_index"]],
            reference_speed=data.reference_speed[
                local_replay_arrays["context_index"]
            ],
        )
    summary = {
        "format_version": 1,
        "method": "train-only continuously updated single-step hybrid Alpha SAC",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "labels": str(args.labels.resolve()),
        "initial_ac": str(args.initial_ac.resolve()),
        "fit_episode_count": len(fit_episodes),
        "selection_episode_count": len(selection_episodes),
        "fit_context_count": len(fit_index),
        "selection_context_count": len(selection_index),
        **resume_metadata,
        "resume_source_selected_metrics": (
            resume_summary.get("selected_metrics")
            if resume_summary is not None else None
        ),
        "initial_metrics": initial_metrics,
        "initial_critic_internal_selection": initial_critic_metrics,
        "initial_local_critic_internal_selection": initial_local_critic_metrics,
        "selected_critic_iteration": best_critic_iteration,
        "selected_critic_internal_selection": best_critic_metrics,
        "selected_local_critic_internal_selection": best_local_critic_metrics,
        "selected_iteration": best_iteration,
        "selected_metrics": best_metrics,
        "latest_training_iteration": latest_iteration,
        "latest_training_metrics": latest_metrics,
        "posthoc_safe_selected_actor_metrics": posthoc_safe_metrics,
        "posthoc_safe_latest_actor_metrics": latest_posthoc_safe_metrics,
        "by_reference_speed_mps": speed,
        "critic_internal_selection": critic_selection,
        "latest_critic_internal_selection": latest_critic_selection,
        "local_critic_internal_selection": local_critic_selection,
        "latest_local_critic_internal_selection": latest_local_critic_selection,
        "online_replay_size": replay.size,
        "online_negative_fraction": float(np.mean(replay_arrays["reward"] < 0.0)),
        "local_probe_replay_size": local_replay.size,
        "collection_history": collection_history,
        "update_history": critic_update_history,
        "critic_rank_history": critic_rank_history,
        "critic_local_rank_history": critic_local_rank_history,
        "evaluation_history": history,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_state_checkpoint": str(training_state_path.resolve()),
        "training_state_checkpoint_sha256": sha256_file(training_state_path),
        "qualification": qualification,
        "effective_actor_learning_rate": actor_optimizer.param_groups[0]["lr"],
        "effective_critic_learning_rate": critic_optimizer.param_groups[0]["lr"],
        "training_arguments": serialized_arguments(args),
        "test_policy": "formal validation and test not loaded or evaluated",
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({
        "initial_cost": initial_metrics["direct_cost"]["mean"],
        "selected_cost": best_metrics["direct_cost"]["mean"],
        "safe_teacher_cost": best_metrics["safe_cost"]["mean"],
        "line_argmin_cost": best_metrics["line_argmin_cost"]["mean"],
        "gap_vs_safe_teacher": best_metrics["gap_vs_safe_teacher"]["mean"],
        "safe_teacher_beaten_fraction": best_metrics["safe_teacher_beaten_fraction"],
        "selection_mode": args.selection_mode,
        "posthoc_safe_cost": (
            posthoc_safe_metrics["direct_cost"]["mean"]
            if posthoc_safe_metrics is not None else None
        ),
        "selected_iteration": best_iteration,
        "selected_critic_iteration": best_critic_iteration,
        "initial_critic_score": initial_critic_metrics["selection_score"],
        "selected_critic_score": best_critic_metrics["selection_score"],
        "online_replay_size": replay.size,
        "online_negative_fraction": summary["online_negative_fraction"],
        "critic_internal_selection": critic_selection,
        "by_reference_speed_mps": speed,
        "checkpoint": str(checkpoint_path),
        "qualification": qualification,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
