#!/usr/bin/env python3
"""Train a train-only 16-D residual center Actor--Critic with real DBM rewards.

The frozen scalar-alpha Actor supplies one deterministic base center.  A new
deterministic Actor emits a unique 8x2 residual around that center.  Exploration
is external: every collected context evaluates all 16 antithetic Hadamard
directions with one deterministic direct DBM rollout per center.  Twin Critics
and the residual Actor remain trainable throughout the repeated one-step loop.

J16 best-found labels are used on internal-fit episodes as broad, forward-cost
replay only.  Internal-selection J16 labels are loaded only for reporting the
same-split numerical upper-bound diagnostic.  No analytic DBM gradient is used.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from car_dynamics.controllers_torch.mppi import fixed_hadamard_knot_noise
from car_foundation.mppi_proposal_policy import (
    TorchMPPIContinuousCenterCritic,
    TorchMPPIDeterministicCenterActor,
    TorchMPPITrustAlphaSACPolicy,
)
from generate_dbm_proposal_teacher import repository_state, sha256_file
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    DEFAULT_LABELS,
    direct_cost,
    distribution,
    load_actor_payload,
    load_dataset,
    tensorize,
)


DEFAULT_BASE = Path(
    "outputs/mppi_proposal/direct_alpha_online_sac_local_probe_20260811_v3_lr2e5/"
    "online_alpha_sac_selected.pt"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_residual_online_ac_20260811_v1"
)
DEFAULT_J16 = (
    Path("outputs/mppi_proposal/dbm_direct_gt_train_20260807_v2/summary.json"),
    Path("outputs/mppi_proposal/dbm_direct_gt_train_expansion_20260807_v2/summary.json"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--base-alpha", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--j16-summary", type=Path, nargs="+", default=DEFAULT_J16)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--contexts-per-iteration", type=int, default=320)
    parser.add_argument("--direction-pairs", type=int, default=16)
    parser.add_argument("--exploration-radius-start", type=float, default=0.40)
    parser.add_argument("--exploration-radius-end", type=float, default=0.08)
    parser.add_argument(
        "--verified-trust-step-sigma", type=float, default=0.0,
        help=(
            "If positive, use a broad probe only to choose a direction, then "
            "forward-evaluate a bounded source-sigma step before making it an "
            "Actor measured target. Zero reproduces direct probe cloning."
        ),
    )
    parser.add_argument("--maximum-residual-sigma", type=float, default=2.0)
    parser.add_argument(
        "--replay-capacity", type=int, default=350000,
        help=(
            "Must retain the fit base/J16 anchors plus all probes for the "
            "declared run; the 30x320x33 default needs 328,200 slots."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--critic-updates-per-iteration", type=int, default=60)
    parser.add_argument("--actor-updates-per-iteration", type=int, default=4)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-5)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--reward-scale", type=float, default=5.0)
    parser.add_argument("--critic-huber-beta", type=float, default=0.10)
    parser.add_argument("--critic-sign-weight", type=float, default=0.05)
    parser.add_argument("--actor-q-weight", type=float, default=0.20)
    parser.add_argument("--measured-target-weight", type=float, default=1.0)
    parser.add_argument("--j16-target-weight", type=float, default=0.02)
    parser.add_argument("--residual-trust-weight", type=float, default=0.01)
    parser.add_argument("--freeze-actor-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluation-interval", type=int, default=2)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--max-fit-contexts", type=int, default=0)
    parser.add_argument("--max-selection-contexts", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serialized_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: ([str(item) for item in value] if isinstance(value, (list, tuple))
              and value and isinstance(value[0], Path)
              else str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }


class Replay:
    """Bounded deterministic direct-reward replay."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.context = np.empty(capacity, np.int64)
        self.action = np.empty((capacity, 8, 2), np.float32)
        self.reward = np.empty(capacity, np.float32)
        self.source = np.empty(capacity, "U24")
        self.size = 0
        self.position = 0

    def add(self, context: np.ndarray, action: np.ndarray, reward: np.ndarray, source: str) -> None:
        context = np.asarray(context, np.int64).reshape(-1)
        action = np.asarray(action, np.float32).reshape(-1, 8, 2)
        reward = np.asarray(reward, np.float32).reshape(-1)
        if not (len(context) == len(action) == len(reward)):
            raise ValueError("replay arrays have different lengths")
        for one_context, one_action, one_reward in zip(context, action, reward):
            self.context[self.position] = one_context
            self.action[self.position] = one_action
            self.reward[self.position] = one_reward
            self.source[self.position] = source
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.size < self.capacity:
            order = np.arange(self.size)
        else:
            order = np.concatenate((
                np.arange(self.position, self.capacity), np.arange(self.position)
            ))
        return (
            self.context[order].copy(), self.action[order].copy(),
            self.reward[order].copy(), self.source[order].copy(),
        )

    def sample(self, count: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        if self.size == 0:
            raise ValueError("cannot sample empty replay")
        pool = np.arange(self.size if self.size < self.capacity else self.capacity)
        negative = pool[self.reward[pool] < 0.0]
        negative_count = min(len(negative), count // 4)
        chosen = rng.choice(pool, size=count - negative_count, replace=True)
        if negative_count:
            chosen = np.concatenate((
                chosen, rng.choice(negative, size=negative_count, replace=True)
            ))
            rng.shuffle(chosen)
        return self.context[chosen], self.action[chosen], self.reward[chosen]


def make_base_policy(payload: dict[str, Any], device: torch.device) -> TorchMPPITrustAlphaSACPolicy:
    policy = TorchMPPITrustAlphaSACPolicy(
        dropout=0.0,
        alpha_logit_scale=float(payload.get("alpha_logit_scale", 1.0)),
    ).to(device)
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    policy.eval()
    return policy


def actor_inputs(
    tensors: dict[str, Any], base_center: np.ndarray, device: torch.device,
) -> tuple[torch.Tensor, ...]:
    values = list(tensors["inputs"])
    values[3] = torch.from_numpy(np.asarray(base_center, np.float32)).to(device)
    return tuple(values)


def module_batch(
    module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], index: torch.Tensor,
    action: torch.Tensor | None = None,
):
    state = tuple(value[index] for value in inputs)
    return module(*state) if action is None else module(*state, action)


@torch.no_grad()
def residual_outputs(
    actor: TorchMPPIDeterministicCenterActor,
    inputs: tuple[torch.Tensor, ...], index: np.ndarray,
    batch_size: int, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    actor.eval()
    actions, centers = [], []
    for start in range(0, len(index), batch_size):
        one = torch.from_numpy(index[start:start + batch_size]).to(device)
        action, center = module_batch(actor, inputs, one)
        actions.append(action.cpu().numpy())
        centers.append(center.cpu().numpy())
    return np.concatenate(actions), np.concatenate(centers)


def hadamard_directions() -> np.ndarray:
    bank = fixed_hadamard_knot_noise(
        (1.0, 1.0), radii=(1.0, 2.0), device="cpu"
    ).numpy()
    result = np.asarray(bank[2:34:2], np.float32)
    if result.shape != (16, 8, 2) or np.linalg.matrix_rank(result.reshape(16, 16)) != 16:
        raise AssertionError("Hadamard exploration is not full rank")
    return result


def center_from_residual(
    base_center: np.ndarray, sigma: np.ndarray, action: np.ndarray,
    maximum_residual_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    raw = base_center + action * maximum_residual_sigma * sigma[:, None, :]
    center = np.clip(raw, -1.0, 1.0).astype(np.float32)
    effective = (center - base_center) / (
        maximum_residual_sigma * sigma[:, None, :]
    )
    return center, effective.astype(np.float32)


def load_j16(
    labels: Path, summary_paths: list[Path] | tuple[Path, ...], context_count: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    by_source: dict[str, tuple[np.ndarray, float]] = {}
    hashes = []
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text())
        hashes.append(sha256_file(summary_path))
        for row in summary["rows"]:
            result_path = Path(row["result"])
            with np.load(result_path, allow_pickle=False) as result:
                best = int(result["knot_best_index"])
                by_source[str(Path(str(result["source_snapshot"])).resolve())] = (
                    np.asarray(result["optimized_knots"][best], np.float32),
                    float(result["knot_cost_replay"][best]),
                )
    knots, costs = [], []
    for label_path in sorted(labels.glob("episode_*/*.npz")):
        with np.load(label_path, allow_pickle=False) as label:
            key = str(Path(str(label["source_snapshot"])).resolve())
            if key not in by_source:
                raise KeyError(f"missing J16 result for {key}")
            knot, cost = by_source[key]
            repeat = len(label["safe_index"])
            knots.extend([knot] * repeat)
            costs.extend([cost] * repeat)
    if len(knots) != context_count:
        raise AssertionError("J16/TR1 context count mismatch")
    return np.asarray(knots, np.float32), np.asarray(costs, np.float32), hashes


def stratified_sample(data, index: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    groups = [
        index[np.isclose(data.reference_speed[index], speed)]
        for speed in sorted(np.unique(data.reference_speed[index]))
    ]
    per_group = int(np.ceil(count / len(groups)))
    result = np.concatenate([
        rng.choice(group, size=per_group, replace=len(group) < per_group)
        for group in groups
    ])[:count]
    rng.shuffle(result)
    return result.astype(np.int64)


@torch.no_grad()
def evaluate(
    actor, inputs, data, tensors, index, base_cost, j16_cost,
    args, device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    action, center = residual_outputs(
        actor, inputs, index, args.evaluation_batch_size, device
    )
    cost = direct_cost(
        center, data, tensors, index, args.evaluation_batch_size, device
    )
    gain_base = base_cost[index] - cost
    gain_old = data.old_cost[index] - cost
    safe = data.safe_cost[index]
    line = np.min(data.direct_line_cost[index], axis=1)
    rho = np.sqrt(np.mean((
        (center - inputs[3][index].detach().cpu().numpy())
        / data.sigma[index, None, :]
    ) ** 2, axis=(1, 2)))
    result = {
        "direct_cost": distribution(cost),
        "base_alpha_cost": distribution(base_cost[index]),
        "old_cost": distribution(data.old_cost[index]),
        "safe_teacher_cost": distribution(safe),
        "line_argmin_cost": distribution(line),
        "j16_best_found_cost": distribution(j16_cost[index]),
        "gain_vs_base_alpha": distribution(gain_base),
        "gain_vs_old": distribution(gain_old),
        "gap_vs_safe_teacher": distribution(cost - safe),
        "gap_vs_line_argmin": distribution(cost - line),
        "gap_vs_j16_best_found": distribution(cost - j16_cost[index]),
        "base_beaten_fraction": float(np.mean(gain_base > 1e-6)),
        "base_regression_fraction": float(np.mean(gain_base < -1e-6)),
        "safe_teacher_beaten_fraction": float(np.mean(cost < safe - 1e-6)),
        "residual_rho_source_sigma": distribution(rho),
        "action_saturation_fraction": float(np.mean(np.abs(action) >= 0.999)),
    }
    return result, action, cost


def grouped_metrics(cost: np.ndarray, base_cost: np.ndarray, data, index: np.ndarray) -> dict[str, Any]:
    result = {}
    for speed in sorted(np.unique(data.reference_speed[index])):
        local = np.isclose(data.reference_speed[index], speed)
        absolute = index[local]
        result[f"{float(speed):.1f}"] = {
            "context_count": int(len(absolute)),
            "base_alpha_cost_mean": float(np.mean(base_cost[absolute])),
            "residual_actor_cost_mean": float(np.mean(cost[local])),
            "gain_vs_base_mean": float(np.mean(base_cost[absolute] - cost[local])),
            "gain_vs_old_mean": float(np.mean(data.old_cost[absolute] - cost[local])),
        }
    return result


def update_critics(q1, q2, optimizer, replay, inputs, args, rng, device) -> dict[str, float]:
    losses, sign_values = [], []
    for _ in range(args.critic_updates_per_iteration):
        context, action, reward = replay.sample(args.batch_size, rng)
        absolute = torch.from_numpy(context).to(device)
        action_tensor = torch.from_numpy(action).to(device)
        target = torch.from_numpy(np.arcsinh(reward / args.reward_scale)).to(device)
        pred1 = module_batch(q1, inputs, absolute, action_tensor)
        pred2 = module_batch(q2, inputs, absolute, action_tensor)
        value = F.smooth_l1_loss(pred1, target, beta=args.critic_huber_beta)
        value = value + F.smooth_l1_loss(pred2, target, beta=args.critic_huber_beta)
        meaningful = target.abs() >= 0.01
        if torch.any(meaningful):
            sign = torch.sign(target[meaningful])
            sign_loss = F.softplus(-sign * pred1[meaningful] / 0.05).mean()
            sign_loss = sign_loss + F.softplus(-sign * pred2[meaningful] / 0.05).mean()
        else:
            sign_loss = torch.zeros((), device=device)
        loss = value + args.critic_sign_weight * sign_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(q1.parameters()) + list(q2.parameters()), 5.0
        )
        optimizer.step()
        losses.append(float(value.detach()))
        sign_values.append(float(sign_loss.detach()))
    return {
        "critic_value_loss": float(np.mean(losses)),
        "critic_sign_loss": float(np.mean(sign_values)),
    }


def update_actor(
    actor, q1, q2, optimizer, inputs, fit_index,
    measured_action, measured_valid, j16_action,
    args, rng, device,
) -> dict[str, float]:
    critic_parameters = list(q1.parameters()) + list(q2.parameters())
    for parameter in critic_parameters:
        parameter.requires_grad_(False)
    q1.eval()
    q2.eval()
    losses, q_values, measured_losses = [], [], []
    valid_index = fit_index[measured_valid[fit_index]]
    for _ in range(args.actor_updates_per_iteration):
        pool = valid_index if len(valid_index) >= args.batch_size // 2 else fit_index
        context = rng.choice(pool, size=args.batch_size, replace=True)
        absolute = torch.from_numpy(context).to(device)
        action, _ = module_batch(actor, inputs, absolute)
        value = torch.minimum(
            module_batch(q1, inputs, absolute, action),
            module_batch(q2, inputs, absolute, action),
        )
        measured = torch.from_numpy(measured_action[context]).to(device)
        target_loss = F.smooth_l1_loss(action, measured, beta=0.05)
        oracle = torch.from_numpy(j16_action[context]).to(device)
        oracle_loss = F.smooth_l1_loss(action, oracle, beta=0.10)
        trust = action.square().mean()
        loss = (
            -args.actor_q_weight * value.mean()
            + args.measured_target_weight * target_loss
            + args.j16_target_weight * oracle_loss
            + args.residual_trust_weight * trust
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in actor.parameters() if p.requires_grad], 2.0
        )
        optimizer.step()
        losses.append(float(loss.detach()))
        q_values.append(float(value.mean().detach()))
        measured_losses.append(float(target_loss.detach()))
    for parameter in critic_parameters:
        parameter.requires_grad_(True)
    return {
        "actor_loss": float(np.mean(losses)),
        "actor_predicted_q": float(np.mean(q_values)),
        "actor_measured_target_loss": float(np.mean(measured_losses)),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 1 <= args.direction_pairs <= 16:
        raise ValueError("direction-pairs must be within [1,16]")
    if not 0 < args.exploration_radius_end <= args.exploration_radius_start:
        raise ValueError("exploration radii must satisfy 0 < end <= start")
    if args.verified_trust_step_sigma < 0.0:
        raise ValueError("verified-trust-step-sigma must be nonnegative")
    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed + 160811)
    device = torch.device(args.device)

    base_payload = torch.load(args.base_alpha, map_location="cpu")
    if base_payload.get("policy_class") != "TorchMPPITrustAlphaSACPolicy":
        raise AssertionError("base checkpoint is not an online Alpha policy")
    if Path(base_payload["labels"]).resolve() != args.labels.resolve():
        raise AssertionError("base Alpha checkpoint and requested labels differ")
    old_payload = load_actor_payload(Path(base_payload["old_actor"]))
    data, _, splits = load_dataset(args.labels, old_payload)
    fit_episodes = list(splits["internal_fit"])
    selection_episodes = list(splits["internal_selection"])
    fit_index = np.flatnonzero(np.isin(data.episodes, fit_episodes))
    selection_index = np.flatnonzero(np.isin(data.episodes, selection_episodes))
    if args.max_fit_contexts:
        fit_index = fit_index[:args.max_fit_contexts]
    if args.max_selection_contexts:
        selection_index = selection_index[:args.max_selection_contexts]
    if set(data.episodes[fit_index]) & set(data.episodes[selection_index]):
        raise AssertionError("fit and internal-selection episodes overlap")
    collection_contexts = min(args.contexts_per_iteration, len(fit_index))
    required_replay_capacity = (
        2 * len(fit_index)
        + args.iterations * collection_contexts * (
            1 + 2 * args.direction_pairs
            + int(args.verified_trust_step_sigma > 0.0)
        )
    )
    if args.replay_capacity < required_replay_capacity:
        raise ValueError(
            f"replay-capacity {args.replay_capacity} would evict frozen anchors; "
            f"the declared run requires at least {required_replay_capacity}"
        )
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    base_policy = make_base_policy(base_payload, device)
    _, _, base_alpha, base_center = deterministic_outputs(
        base_policy, tensors, extra, np.arange(len(data.episodes)),
        float(base_payload["move_threshold"]), args.evaluation_batch_size, device,
    )
    base_cost = direct_cost(
        base_center, data, tensors, np.arange(len(data.episodes)),
        args.evaluation_batch_size, device,
    )
    j16_center, j16_cost, j16_hashes = load_j16(
        args.labels, args.j16_summary, len(data.episodes)
    )
    inputs = actor_inputs(tensors, base_center, device)

    actor = TorchMPPIDeterministicCenterActor(
        args.maximum_residual_sigma, dropout=0.05
    ).to(device)
    encoder_state = {
        key[len("encoder."):]: value
        for key, value in old_payload["actor_state_dict"].items()
        if key.startswith("encoder.")
    }
    actor.encoder.load_state_dict(encoder_state, strict=True)
    if args.freeze_actor_encoder:
        for parameter in actor.encoder.parameters():
            parameter.requires_grad_(False)
    q1 = TorchMPPIContinuousCenterCritic(dropout=0.05).to(device)
    q2 = TorchMPPIContinuousCenterCritic(dropout=0.05).to(device)
    q1.encoder.load_state_dict(actor.encoder.state_dict(), strict=True)
    q2.encoder.load_state_dict(actor.encoder.state_dict(), strict=True)
    actor_optimizer = torch.optim.AdamW(
        [p for p in actor.parameters() if p.requires_grad],
        lr=args.actor_learning_rate, weight_decay=args.weight_decay,
    )
    critic_optimizer = torch.optim.AdamW(
        list(q1.parameters()) + list(q2.parameters()),
        lr=args.critic_learning_rate, weight_decay=args.weight_decay,
    )

    replay = Replay(args.replay_capacity)
    zero_action = np.zeros((len(fit_index), 8, 2), np.float32)
    replay.add(fit_index, zero_action, np.zeros(len(fit_index), np.float32), "base_alpha")
    projected_j16_action = np.clip(
        (j16_center - base_center)
        / (args.maximum_residual_sigma * data.sigma[:, None, :]),
        -1.0, 1.0,
    ).astype(np.float32)
    projected_center, projected_j16_action = center_from_residual(
        base_center, data.sigma, projected_j16_action,
        args.maximum_residual_sigma,
    )
    projected_fit_cost = direct_cost(
        projected_center[fit_index], data, tensors, fit_index,
        args.evaluation_batch_size, device,
    )
    replay.add(
        fit_index, projected_j16_action[fit_index],
        base_cost[fit_index] - projected_fit_cost, "projected_j16_fit",
    )

    measured_action = np.zeros_like(projected_j16_action)
    measured_reward = np.zeros(len(data.episodes), np.float32)
    measured_valid = np.zeros(len(data.episodes), bool)
    directions = hadamard_directions()
    initial_metrics, _, initial_cost = evaluate(
        actor, inputs, data, tensors, selection_index, base_cost, j16_cost,
        args, device,
    )
    best_metrics = copy.deepcopy(initial_metrics)
    best_cost = float(initial_metrics["direct_cost"]["mean"])
    best_iteration = 0
    best_actor = copy.deepcopy(actor.state_dict())
    best_q1 = copy.deepcopy(q1.state_dict())
    best_q2 = copy.deepcopy(q2.state_dict())
    history = [{"iteration": 0, **initial_metrics}]
    collection_history = []
    print(
        f"[iteration=000] base/residual={base_cost[selection_index].mean():.6f}/"
        f"{best_cost:.6f} J16={j16_cost[selection_index].mean():.6f}",
        flush=True,
    )

    for iteration in range(1, args.iterations + 1):
        fraction = (iteration - 1) / max(args.iterations - 1, 1)
        radius_sigma = (
            args.exploration_radius_start
            + fraction * (args.exploration_radius_end - args.exploration_radius_start)
        )
        context = stratified_sample(
            data, fit_index, min(args.contexts_per_iteration, len(fit_index)), rng
        )
        actor_action, _ = residual_outputs(
            actor, inputs, context, args.evaluation_batch_size, device
        )
        start = ((iteration - 1) * args.direction_pairs) % 16
        direction_index = np.asarray([
            (start + offset) % 16 for offset in range(args.direction_pairs)
        ], np.int64)
        normalized_radius = radius_sigma / args.maximum_residual_sigma
        bank = [actor_action]
        for direction in directions[direction_index]:
            bank.extend((
                np.clip(actor_action + normalized_radius * direction, -1.0, 1.0),
                np.clip(actor_action - normalized_radius * direction, -1.0, 1.0),
            ))
        action_bank = np.stack(bank, axis=1).astype(np.float32)
        flat_context = np.repeat(context, action_bank.shape[1])
        flat_action = action_bank.reshape(-1, 8, 2)
        flat_center, flat_action = center_from_residual(
            base_center[flat_context], data.sigma[flat_context], flat_action,
            args.maximum_residual_sigma,
        )
        cost = direct_cost(
            flat_center, data, tensors, flat_context,
            args.evaluation_batch_size, device,
        )
        reward = base_cost[flat_context] - cost
        replay.add(flat_context, flat_action, reward, f"probe_{iteration:03d}")
        reward_bank = reward.reshape(len(context), -1)
        effective_bank = flat_action.reshape(len(context), -1, 8, 2)
        best = np.argmax(reward_bank, axis=1)
        row = np.arange(len(context))
        broad_best_reward = reward_bank[row, best]
        broad_best_action = effective_bank[row, best]
        trust_summary = None
        if args.verified_trust_step_sigma > 0.0:
            delta = broad_best_action - actor_action
            delta_rho = np.sqrt(np.mean(
                (delta * args.maximum_residual_sigma) ** 2,
                axis=(1, 2),
            ))
            trust_scale = np.minimum(
                1.0,
                args.verified_trust_step_sigma / np.maximum(delta_rho, 1e-12),
            ).astype(np.float32)
            trust_action = (
                actor_action + trust_scale[:, None, None] * delta
            ).astype(np.float32)
            trust_center, trust_action = center_from_residual(
                base_center[context], data.sigma[context], trust_action,
                args.maximum_residual_sigma,
            )
            trust_cost = direct_cost(
                trust_center, data, tensors, context,
                args.evaluation_batch_size, device,
            )
            candidate_reward = base_cost[context] - trust_cost
            candidate_action = trust_action
            replay.add(
                context, trust_action, candidate_reward,
                f"trust_{iteration:03d}",
            )
            trust_summary = {
                "maximum_step_source_sigma": args.verified_trust_step_sigma,
                "requested_direction_rho_source_sigma": distribution(delta_rho),
                "applied_scale": distribution(trust_scale),
                "verified_reward": distribution(candidate_reward),
                "positive_verified_fraction": float(np.mean(candidate_reward > 0.0)),
            }
        else:
            candidate_reward = broad_best_reward
            candidate_action = broad_best_action
        improve = candidate_reward > measured_reward[context] + 1e-6
        update_context = context[improve]
        measured_action[update_context] = candidate_action[improve]
        measured_reward[update_context] = candidate_reward[improve]
        measured_valid[update_context] = True
        critic_update = update_critics(
            q1, q2, critic_optimizer, replay, inputs, args, rng, device
        )
        actor_update = update_actor(
            actor, q1, q2, actor_optimizer, inputs, fit_index,
            measured_action, measured_valid, projected_j16_action,
            args, rng, device,
        )
        collection = {
            "iteration": iteration,
            "context_count": int(len(context)),
            "centers_per_context": int(action_bank.shape[1]),
            "direction_indices": direction_index.tolist(),
            "exploration_radius_source_sigma": float(radius_sigma),
            "reward": distribution(reward),
            "negative_reward_fraction": float(np.mean(reward < 0.0)),
            "broad_probe_best_reward": distribution(broad_best_reward),
            "actor_target_reward": distribution(candidate_reward),
            "verified_trust_step": trust_summary,
            "measured_target_context_count": int(np.sum(measured_valid[fit_index])),
            "replay_size": replay.size,
            **critic_update,
            **actor_update,
        }
        collection_history.append(collection)
        if iteration % args.evaluation_interval == 0 or iteration == args.iterations:
            metrics, _, selection_cost = evaluate(
                actor, inputs, data, tensors, selection_index, base_cost, j16_cost,
                args, device,
            )
            row_metrics = {"iteration": iteration, **collection, **metrics}
            history.append(row_metrics)
            mean_cost = float(metrics["direct_cost"]["mean"])
            if mean_cost < best_cost - 1e-6:
                best_cost = mean_cost
                best_iteration = iteration
                best_metrics = copy.deepcopy(metrics)
                best_actor = copy.deepcopy(actor.state_dict())
                best_q1 = copy.deepcopy(q1.state_dict())
                best_q2 = copy.deepcopy(q2.state_dict())
            print(
                f"[iteration={iteration:03d}] cost={mean_cost:.6f} "
                f"gain_base={metrics['gain_vs_base_alpha']['mean']:.6f} "
                f"p05={metrics['gain_vs_base_alpha']['p05']:.6f} "
                f"replay={replay.size}", flush=True,
            )

    actor.load_state_dict(best_actor, strict=True)
    q1.load_state_dict(best_q1, strict=True)
    q2.load_state_dict(best_q2, strict=True)
    selected_metrics, selected_action, selected_cost = evaluate(
        actor, inputs, data, tensors, selection_index, base_cost, j16_cost,
        args, device,
    )
    repeated_action, repeated_center = residual_outputs(
        actor, inputs, selection_index, args.evaluation_batch_size, device
    )
    _, repeated_center_2 = residual_outputs(
        actor, inputs, selection_index, args.evaluation_batch_size, device
    )
    deterministic_error = float(np.max(np.abs(repeated_center - repeated_center_2)))
    selected_metrics["grouped_by_reference_speed"] = grouped_metrics(
        selected_cost, base_cost, data, selection_index
    )
    selected_metrics["deterministic_repeat_max_abs_error"] = deterministic_error
    selected_metrics["old_to_j16_potential_recovered_fraction"] = float(
        (np.mean(data.old_cost[selection_index]) - np.mean(selected_cost))
        / max(np.mean(data.old_cost[selection_index] - j16_cost[selection_index]), 1e-12)
    )
    improved = best_cost < float(initial_metrics["direct_cost"]["mean"]) - 1e-6
    qualification = (
        "DIRECT_RESIDUAL_16D_INTERNAL_MEAN_IMPROVED"
        if improved else "DIRECT_RESIDUAL_16D_INTERNAL_PLATEAU"
    )
    checkpoint_path = args.output_dir / "direct_residual_online_ac_selected.pt"
    torch.save({
        "format_version": 1,
        "method": "train-only continuous 16-D residual direct Actor-Critic",
        "qualification": qualification,
        "actor_class": "TorchMPPIDeterministicCenterActor",
        "critic_class": "TorchMPPIContinuousCenterCritic",
        "actor_state_dict": best_actor,
        "critic1_state_dict": best_q1,
        "critic2_state_dict": best_q2,
        "base_alpha_checkpoint": str(args.base_alpha.resolve()),
        "base_alpha_sha256": sha256_file(args.base_alpha),
        "labels": str(args.labels.resolve()),
        "labels_hashes": {
            name: sha256_file(args.labels / name)
            for name in ("config.json", "splits.json", "summary.json")
        },
        "j16_summaries": [str(path.resolve()) for path in args.j16_summary],
        "j16_summary_sha256": j16_hashes,
        "maximum_residual_sigma": args.maximum_residual_sigma,
        "base_move_threshold": float(base_payload["move_threshold"]),
        "selected_iteration": best_iteration,
        "fit_episodes": fit_episodes,
        "selection_episodes": selection_episodes,
        "training_arguments": serialized_args(args),
        "internal_selection_metrics": selected_metrics,
        "test_policy": "formal validation and test not loaded or evaluated",
    }, checkpoint_path)
    replay_context, replay_action, replay_reward, replay_source = replay.arrays()
    np.savez_compressed(
        args.output_dir / "direct_residual_replay.npz",
        context_index=replay_context,
        action=replay_action,
        reward=replay_reward,
        source=replay_source,
        episode=data.episodes[replay_context],
        reference_speed=data.reference_speed[replay_context],
    )
    summary = {
        "format_version": 1,
        "method": "train-only continuous 16-D residual direct Actor-Critic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "checkpoint": str(checkpoint_path.resolve()),
        "base_alpha_checkpoint": str(args.base_alpha.resolve()),
        "fit_context_count": int(len(fit_index)),
        "selection_context_count": int(len(selection_index)),
        "replay_size": replay.size,
        "required_replay_capacity": required_replay_capacity,
        "selected_iteration": best_iteration,
        "initial_internal_selection": initial_metrics,
        "selected_internal_selection": selected_metrics,
        "projected_j16_fit_cost": distribution(projected_fit_cost),
        "actor_output": "unique deterministic 8x2 residual center",
        "exploration": {
            "training_only": True,
            "full_rank": args.direction_pairs == 16,
            "direction_pairs": args.direction_pairs,
            "verified_trust_step_source_sigma": args.verified_trust_step_sigma,
            "radius_source_sigma": [
                args.exploration_radius_start, args.exploration_radius_end,
            ],
            "reward": "base Alpha direct cost - residual center direct cost",
            "analytic_dbm_gradient": False,
        },
        "history": history,
        "collection_history": collection_history,
        "test_policy": "formal validation and test not loaded or evaluated",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("history", "collection_history")}, indent=2))


if __name__ == "__main__":
    main()
