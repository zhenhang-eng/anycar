#!/usr/bin/env python3
"""Train the unique-output 16-D Actor from response-guided Critic supervision.

Probe feedback is training-only.  It never enters Actor or Critic inputs.  Around
the current Actor action, forward DBM rollouts provide full-rank antithetic pairs;
trajectory-response fitting proposes additional useful actions.  Twin Critics fit
point rewards plus within-state finite-difference delta/sign targets, and the Actor
is updated only through the Critics with a trust penalty.  No winner cloning and no
analytic DBM gradient are used.
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

from car_foundation.mppi_proposal_policy import (
    TorchMPPIContinuousCenterCritic,
    TorchMPPIDeterministicCenterActor,
)
from evaluate_mppi_direct_feedback_guided_exploration import (
    first_pass_bank,
    fit_response_directions,
    line_bank,
    rollout_cost_residuals,
)
from generate_dbm_proposal_teacher import repository_state, sha256_file
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_residual_online_ac import (
    Replay,
    actor_inputs,
    distribution,
    evaluate,
    grouped_metrics,
    hadamard_directions,
    load_j16,
    make_base_policy,
    module_batch,
    residual_outputs,
    stratified_sample,
)
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    direct_cost,
    load_actor_payload,
    load_dataset,
    tensorize,
)


DEFAULT_INITIAL = Path(
    "outputs/mppi_proposal/direct_residual_online_ac_20260811_v2/"
    "direct_residual_online_ac_selected.pt"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_response_slope_ac_20260811_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-actor", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--contexts-per-iteration", type=int, default=320)
    parser.add_argument("--first-radius-sigma", type=float, default=0.10)
    parser.add_argument(
        "--response-radii-sigma", default="0.025,0.05,0.10,0.15,0.25,0.40"
    )
    parser.add_argument("--fit-ridge", type=float, default=0.10)
    parser.add_argument("--step-damping", type=float, default=0.10)
    parser.add_argument("--maximum-gn-step-sigma", type=float, default=0.40)
    parser.add_argument("--replay-capacity", type=int, default=300000)
    parser.add_argument("--pair-capacity", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rollout-batch-size", type=int, default=24)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--critic-updates-per-iteration", type=int, default=80)
    parser.add_argument("--actor-updates-per-iteration", type=int, default=2)
    parser.add_argument("--critic-warmup-iterations", type=int, default=4)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-5)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--reward-scale", type=float, default=5.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--pair-delta-weight", type=float, default=2.0)
    parser.add_argument("--pair-sign-weight", type=float, default=0.10)
    parser.add_argument("--actor-trust-weight", type=float, default=0.05)
    parser.add_argument(
        "--freeze-actor-encoder", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--evaluation-interval", type=int, default=2)
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


class PairReplay:
    """Within-state action pairs carrying finite-difference reward information."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.context = np.empty(capacity, np.int64)
        self.action_a = np.empty((capacity, 8, 2), np.float32)
        self.action_b = np.empty((capacity, 8, 2), np.float32)
        self.reward_a = np.empty(capacity, np.float32)
        self.reward_b = np.empty(capacity, np.float32)
        self.source = np.empty(capacity, "U16")
        self.size = 0
        self.position = 0

    def add(
        self, context: np.ndarray, action_a: np.ndarray, action_b: np.ndarray,
        reward_a: np.ndarray, reward_b: np.ndarray, source: str,
    ) -> None:
        arrays = (
            np.asarray(context, np.int64).reshape(-1),
            np.asarray(action_a, np.float32).reshape(-1, 8, 2),
            np.asarray(action_b, np.float32).reshape(-1, 8, 2),
            np.asarray(reward_a, np.float32).reshape(-1),
            np.asarray(reward_b, np.float32).reshape(-1),
        )
        if len({len(value) for value in arrays}) != 1:
            raise ValueError("pair replay arrays have different lengths")
        for values in zip(*arrays):
            i = self.position
            self.context[i], self.action_a[i], self.action_b[i] = values[:3]
            self.reward_a[i], self.reward_b[i], self.source[i] = values[3], values[4], source
            self.position = (i + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, count: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        chosen = rng.choice(self.size, size=count, replace=True)
        return (
            self.context[chosen], self.action_a[chosen], self.action_b[chosen],
            self.reward_a[chosen], self.reward_b[chosen],
        )

    def arrays(self) -> tuple[np.ndarray, ...]:
        if self.size < self.capacity:
            order = np.arange(self.size)
        else:
            order = np.concatenate((np.arange(self.position, self.capacity), np.arange(self.position)))
        return (
            self.context[order].copy(), self.action_a[order].copy(),
            self.action_b[order].copy(), self.reward_a[order].copy(),
            self.reward_b[order].copy(), self.source[order].copy(),
        )


def centers_to_actions(
    centers: np.ndarray, alpha_center: np.ndarray, sigma: np.ndarray,
    maximum_residual_sigma: float,
) -> np.ndarray:
    return np.clip(
        (centers - alpha_center[:, None])
        / (maximum_residual_sigma * sigma[:, None, None, :]),
        -1.0, 1.0,
    ).astype(np.float32)


def transformed(reward: np.ndarray, scale: float) -> np.ndarray:
    return np.arcsinh(np.asarray(reward, np.float32) / scale).astype(np.float32)


def update_critics(
    q1: torch.nn.Module, q2: torch.nn.Module, optimizer: torch.optim.Optimizer,
    replay: Replay, pairs: PairReplay, inputs: tuple[torch.Tensor, ...],
    args: argparse.Namespace, rng: np.random.Generator, device: torch.device,
) -> dict[str, float]:
    q1.train()
    q2.train()
    value_losses, delta_losses, sign_losses = [], [], []
    for _ in range(args.critic_updates_per_iteration):
        context, action, reward = replay.sample(args.batch_size, rng)
        index = torch.from_numpy(context).to(device)
        action_t = torch.from_numpy(action).to(device)
        target = torch.from_numpy(transformed(reward, args.reward_scale)).to(device)
        p1 = module_batch(q1, inputs, index, action_t)
        p2 = module_batch(q2, inputs, index, action_t)
        value_loss = F.smooth_l1_loss(p1, target, beta=0.10)
        value_loss = value_loss + F.smooth_l1_loss(p2, target, beta=0.10)

        pc, aa, ab, ra, rb = pairs.sample(args.batch_size, rng)
        pi = torch.from_numpy(pc).to(device)
        aa_t, ab_t = torch.from_numpy(aa).to(device), torch.from_numpy(ab).to(device)
        pair_target = torch.from_numpy(
            transformed(ra, args.reward_scale) - transformed(rb, args.reward_scale)
        ).to(device)
        d1 = module_batch(q1, inputs, pi, aa_t) - module_batch(q1, inputs, pi, ab_t)
        d2 = module_batch(q2, inputs, pi, aa_t) - module_batch(q2, inputs, pi, ab_t)
        delta_loss = F.smooth_l1_loss(d1, pair_target, beta=0.05)
        delta_loss = delta_loss + F.smooth_l1_loss(d2, pair_target, beta=0.05)
        meaningful = pair_target.abs() >= 0.005
        if torch.any(meaningful):
            sign = torch.sign(pair_target[meaningful])
            sign_loss = F.softplus(-sign * d1[meaningful] / 0.05).mean()
            sign_loss = sign_loss + F.softplus(-sign * d2[meaningful] / 0.05).mean()
        else:
            sign_loss = torch.zeros((), device=device)
        loss = (
            args.value_weight * value_loss
            + args.pair_delta_weight * delta_loss
            + args.pair_sign_weight * sign_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 5.0)
        optimizer.step()
        value_losses.append(float(value_loss.detach()))
        delta_losses.append(float(delta_loss.detach()))
        sign_losses.append(float(sign_loss.detach()))
    return {
        "critic_value_loss": float(np.mean(value_losses)),
        "critic_pair_delta_loss": float(np.mean(delta_losses)),
        "critic_pair_sign_loss": float(np.mean(sign_losses)),
    }


def update_actor(
    actor: torch.nn.Module, q1: torch.nn.Module, q2: torch.nn.Module,
    optimizer: torch.optim.Optimizer, inputs: tuple[torch.Tensor, ...],
    fit_index: np.ndarray, initial_action: np.ndarray, args: argparse.Namespace,
    rng: np.random.Generator, device: torch.device,
) -> dict[str, float]:
    actor.train()
    q1.eval()
    q2.eval()
    for parameter in list(q1.parameters()) + list(q2.parameters()):
        parameter.requires_grad_(False)
    losses, values, trusts = [], [], []
    for _ in range(args.actor_updates_per_iteration):
        context = rng.choice(fit_index, size=args.batch_size, replace=True)
        index = torch.from_numpy(context).to(device)
        action, _ = module_batch(actor, inputs, index)
        value = torch.minimum(
            module_batch(q1, inputs, index, action),
            module_batch(q2, inputs, index, action),
        )
        anchor = torch.from_numpy(initial_action[context]).to(device)
        trust = F.smooth_l1_loss(action, anchor, beta=0.05)
        loss = -value.mean() + args.actor_trust_weight * trust
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in actor.parameters() if p.requires_grad], 2.0
        )
        optimizer.step()
        losses.append(float(loss.detach()))
        values.append(float(value.mean().detach()))
        trusts.append(float(trust.detach()))
    for parameter in list(q1.parameters()) + list(q2.parameters()):
        parameter.requires_grad_(True)
    return {
        "actor_loss": float(np.mean(losses)),
        "actor_predicted_q": float(np.mean(values)),
        "actor_trust_loss": float(np.mean(trusts)),
    }


@torch.no_grad()
def critic_probe_metrics(
    actor: torch.nn.Module, q1: torch.nn.Module, q2: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...], data: Any, tensors: dict[str, Any],
    index: np.ndarray, alpha_center: np.ndarray, base_cost: np.ndarray,
    maximum_residual_sigma: float, directions: np.ndarray,
    args: argparse.Namespace, device: torch.device,
) -> dict[str, float]:
    actor.eval()
    q1.eval()
    q2.eval()
    true_delta, predicted_delta, regrets = [], [], []
    for start in range(0, len(index), args.rollout_batch_size):
        one = index[start:start + args.rollout_batch_size]
        action, center = residual_outputs(actor, inputs, one, args.evaluation_batch_size, device)
        bank = first_pass_bank(center, data.sigma[one], directions, args.first_radius_sigma)
        cost, _ = rollout_cost_residuals(bank, data, tensors, one, device)
        actions = centers_to_actions(
            bank, alpha_center[one], data.sigma[one], maximum_residual_sigma
        )
        flat_index = np.repeat(one, 33)
        flat_action = torch.from_numpy(actions.reshape(-1, 8, 2)).to(device)
        absolute = torch.from_numpy(flat_index).to(device)
        prediction = torch.minimum(
            module_batch(q1, inputs, absolute, flat_action),
            module_batch(q2, inputs, absolute, flat_action),
        ).reshape(len(one), 33).cpu().numpy()
        reward = base_cost[one, None] - cost
        z = transformed(reward, args.reward_scale)
        true_delta.append(z[:, 1:17] - z[:, 17:33])
        predicted_delta.append(prediction[:, 1:17] - prediction[:, 17:33])
        selected = np.argmax(prediction, axis=1)
        regrets.append(cost[np.arange(len(one)), selected] - cost.min(axis=1))
    truth, prediction = np.concatenate(true_delta).reshape(-1), np.concatenate(predicted_delta).reshape(-1)
    meaningful = np.abs(truth) >= 0.005
    correlation = float(np.corrcoef(truth, prediction)[0, 1]) if np.std(prediction) > 0 else 0.0
    return {
        "probe_delta_correlation": correlation,
        "probe_delta_sign_accuracy": float(np.mean(np.sign(truth[meaningful]) == np.sign(prediction[meaningful]))),
        "probe_mean_argmax_regret": float(np.mean(np.concatenate(regrets))),
        "probe_meaningful_pair_count": int(np.sum(meaningful)),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed + 260811)
    device = torch.device(args.device)
    radii = np.asarray([float(v) for v in args.response_radii_sigma.split(",")], np.float32)

    initial_payload = torch.load(args.initial_actor, map_location="cpu")
    alpha_path = Path(initial_payload["base_alpha_checkpoint"])
    alpha_payload = torch.load(alpha_path, map_location="cpu")
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    labels = Path(initial_payload["labels"])
    data, _, splits = load_dataset(labels, old_payload)
    fit_index = np.flatnonzero(np.isin(data.episodes, splits["internal_fit"]))
    selection_index = np.flatnonzero(np.isin(data.episodes, splits["internal_selection"]))
    if args.max_fit_contexts:
        fit_index = fit_index[:args.max_fit_contexts]
    if args.max_selection_contexts:
        selection_index = selection_index[:args.max_selection_contexts]
    if set(data.episodes[fit_index]) & set(data.episodes[selection_index]):
        raise AssertionError("fit/selection leakage")

    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    alpha_policy = make_base_policy(alpha_payload, device)
    _, _, _, alpha_center = deterministic_outputs(
        alpha_policy, tensors, extra, np.arange(len(data.episodes)),
        float(initial_payload["base_move_threshold"]), args.evaluation_batch_size, device,
    )
    inputs = actor_inputs(tensors, alpha_center, device)
    maximum_residual_sigma = float(initial_payload["maximum_residual_sigma"])
    actor = TorchMPPIDeterministicCenterActor(maximum_residual_sigma, dropout=0.05).to(device)
    actor.load_state_dict(initial_payload["actor_state_dict"], strict=True)
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

    all_index = np.arange(len(data.episodes))
    initial_action, initial_center = residual_outputs(
        actor, inputs, all_index, args.evaluation_batch_size, device
    )
    base_cost = direct_cost(alpha_center, data, tensors, all_index, args.evaluation_batch_size, device)
    initial_cost_all = direct_cost(initial_center, data, tensors, all_index, args.evaluation_batch_size, device)
    j16_center, j16_cost, j16_hashes = load_j16(
        labels, [Path(path) for path in initial_payload["j16_summaries"]], len(data.episodes)
    )
    del j16_center
    replay = Replay(args.replay_capacity)
    replay.add(
        fit_index, initial_action[fit_index],
        base_cost[fit_index] - initial_cost_all[fit_index], "initial_actor",
    )
    pairs = PairReplay(args.pair_capacity)
    directions = hadamard_directions()

    initial_metrics, _, _ = evaluate(
        actor, inputs, data, tensors, selection_index, base_cost, j16_cost, args, device
    )
    initial_probe = critic_probe_metrics(
        actor, q1, q2, inputs, data, tensors, selection_index, alpha_center,
        base_cost, maximum_residual_sigma, directions, args, device,
    )
    best_cost = float(initial_metrics["direct_cost"]["mean"])
    best_iteration = 0
    best_actor, best_q1, best_q2 = (
        copy.deepcopy(actor.state_dict()), copy.deepcopy(q1.state_dict()), copy.deepcopy(q2.state_dict())
    )
    best_metrics = copy.deepcopy(initial_metrics)
    history = [{"iteration": 0, **initial_metrics, "critic_probe": initial_probe}]
    print(f"[iteration=000] actor={best_cost:.6f} sign={initial_probe['probe_delta_sign_accuracy']:.4f}", flush=True)

    for iteration in range(1, args.iterations + 1):
        context = stratified_sample(
            data, fit_index, min(args.contexts_per_iteration, len(fit_index)), rng
        )
        actor_action, actor_center = residual_outputs(
            actor, inputs, context, args.evaluation_batch_size, device
        )
        first_centers = first_pass_bank(
            actor_center, data.sigma[context], directions, args.first_radius_sigma
        )
        first_cost, first_residual = rollout_cost_residuals(
            first_centers, data, tensors, context, device
        )
        first_action = centers_to_actions(
            first_centers, alpha_center[context], data.sigma[context], maximum_residual_sigma
        )
        first_reward = base_cost[context, None] - first_cost
        replay.add(
            np.repeat(context, 33), first_action.reshape(-1, 8, 2),
            first_reward.reshape(-1), f"probe_{iteration:03d}",
        )
        pair_context = np.repeat(context, 16)
        pairs.add(
            pair_context, first_action[:, 1:17].reshape(-1, 8, 2),
            first_action[:, 17:33].reshape(-1, 8, 2),
            first_reward[:, 1:17].reshape(-1), first_reward[:, 17:33].reshape(-1),
            "antithetic",
        )

        cost_direction, trajectory_direction, blended_direction, fit = fit_response_directions(
            first_centers, actor_center, data.sigma[context], first_cost, first_residual,
            args.fit_ridge, args.step_damping, args.maximum_gn_step_sigma,
        )
        response_centers = np.concatenate((
            line_bank(actor_center, data.sigma[context], cost_direction, radii),
            line_bank(actor_center, data.sigma[context], trajectory_direction, radii),
            line_bank(actor_center, data.sigma[context], blended_direction, radii),
        ), axis=1)
        response_cost, _ = rollout_cost_residuals(
            response_centers, data, tensors, context, device
        )
        response_action = centers_to_actions(
            response_centers, alpha_center[context], data.sigma[context], maximum_residual_sigma
        )
        response_reward = base_cost[context, None] - response_cost
        response_count = response_action.shape[1]
        replay.add(
            np.repeat(context, response_count), response_action.reshape(-1, 8, 2),
            response_reward.reshape(-1), f"response_{iteration:03d}",
        )
        pairs.add(
            np.repeat(context, response_count), response_action.reshape(-1, 8, 2),
            np.repeat(actor_action[:, None], response_count, axis=1).reshape(-1, 8, 2),
            response_reward.reshape(-1), np.repeat(first_reward[:, :1], response_count, axis=1).reshape(-1),
            "response",
        )

        critic_update = update_critics(
            q1, q2, critic_optimizer, replay, pairs, inputs, args, rng, device
        )
        if iteration > args.critic_warmup_iterations:
            actor_update = update_actor(
                actor, q1, q2, actor_optimizer, inputs, fit_index,
                initial_action, args, rng, device,
            )
        else:
            actor_update = {
                "actor_loss": 0.0, "actor_predicted_q": 0.0,
                "actor_trust_loss": 0.0,
            }
        collection = {
            "iteration": iteration,
            "context_count": int(len(context)),
            "value_replay_size": replay.size,
            "pair_replay_size": pairs.size,
            "probe_reward": distribution(first_reward),
            "response_reward": distribution(response_reward),
            "trajectory_fit_error": distribution(fit["trajectory_relative_fit_error"]),
            **critic_update, **actor_update,
        }
        if iteration % args.evaluation_interval == 0 or iteration == args.iterations:
            metrics, _, _ = evaluate(
                actor, inputs, data, tensors, selection_index, base_cost, j16_cost, args, device
            )
            probe = critic_probe_metrics(
                actor, q1, q2, inputs, data, tensors, selection_index, alpha_center,
                base_cost, maximum_residual_sigma, directions, args, device,
            )
            history.append({**collection, **metrics, "critic_probe": probe})
            mean_cost = float(metrics["direct_cost"]["mean"])
            if mean_cost < best_cost - 1e-6:
                best_cost, best_iteration, best_metrics = mean_cost, iteration, copy.deepcopy(metrics)
                best_actor, best_q1, best_q2 = (
                    copy.deepcopy(actor.state_dict()), copy.deepcopy(q1.state_dict()), copy.deepcopy(q2.state_dict())
                )
            print(
                f"[iteration={iteration:03d}] actor={mean_cost:.6f} best={best_cost:.6f} "
                f"sign={probe['probe_delta_sign_accuracy']:.4f} regret={probe['probe_mean_argmax_regret']:.4f}",
                flush=True,
            )

    actor.load_state_dict(best_actor, strict=True)
    q1.load_state_dict(best_q1, strict=True)
    q2.load_state_dict(best_q2, strict=True)
    selected_metrics, selected_action, selected_cost = evaluate(
        actor, inputs, data, tensors, selection_index, base_cost, j16_cost, args, device
    )
    selected_metrics["grouped_by_reference_speed"] = grouped_metrics(
        selected_cost, base_cost, data, selection_index
    )
    selected_probe = critic_probe_metrics(
        actor, q1, q2, inputs, data, tensors, selection_index, alpha_center,
        base_cost, maximum_residual_sigma, directions, args, device,
    )
    _, repeated_center = residual_outputs(actor, inputs, selection_index, args.evaluation_batch_size, device)
    _, repeated_center2 = residual_outputs(actor, inputs, selection_index, args.evaluation_batch_size, device)
    selected_metrics["deterministic_repeat_max_abs_error"] = float(np.max(np.abs(repeated_center - repeated_center2)))
    selected_metrics["critic_probe"] = selected_probe
    qualification = (
        "RESPONSE_SLOPE_AC_INTERNAL_MEAN_IMPROVED"
        if best_iteration > 0 else "RESPONSE_SLOPE_AC_INTERNAL_PLATEAU"
    )
    checkpoint = args.output_dir / "direct_response_slope_ac_selected.pt"
    torch.save({
        "format_version": 1,
        "method": "unique-output 16-D Actor-Critic with training-only response slope supervision",
        "qualification": qualification,
        "actor_class": "TorchMPPIDeterministicCenterActor",
        "critic_class": "TorchMPPIContinuousCenterCritic",
        "actor_state_dict": best_actor,
        "critic1_state_dict": best_q1,
        "critic2_state_dict": best_q2,
        "initial_actor": str(args.initial_actor.resolve()),
        "initial_actor_sha256": sha256_file(args.initial_actor),
        "base_alpha_checkpoint": str(alpha_path.resolve()),
        "base_move_threshold": float(initial_payload["base_move_threshold"]),
        "maximum_residual_sigma": maximum_residual_sigma,
        "labels": str(labels.resolve()),
        "selected_iteration": best_iteration,
        "fit_episodes": list(splits["internal_fit"]),
        "selection_episodes": list(splits["internal_selection"]),
        "training_arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "internal_selection_metrics": selected_metrics,
        "actor_new_probe_input": False,
        "test_policy": "formal validation and test not loaded or evaluated",
    }, checkpoint)
    rc, ra, rr, rs = replay.arrays()
    np.savez_compressed(args.output_dir / "value_replay.npz", context_index=rc, action=ra, reward=rr, source=rs)
    pc, paa, pab, pra, prb, ps = pairs.arrays()
    np.savez_compressed(
        args.output_dir / "pair_replay.npz", context_index=pc, action_a=paa,
        action_b=pab, reward_a=pra, reward_b=prb, source=ps,
    )
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "unique-output 16-D Actor-Critic with training-only response slope supervision",
        "qualification": qualification,
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "checkpoint": str(checkpoint.resolve()),
        "initial_actor": str(args.initial_actor.resolve()),
        "fit_context_count": int(len(fit_index)),
        "selection_context_count": int(len(selection_index)),
        "selected_iteration": best_iteration,
        "value_replay_size": replay.size,
        "pair_replay_size": pairs.size,
        "initial_internal_selection": initial_metrics,
        "initial_critic_probe": initial_probe,
        "selected_internal_selection": selected_metrics,
        "history": history,
        "contract": {
            "actor_new_probe_input": False,
            "actor_output": "one unique deterministic 8x2 residual center",
            "probe_usage": "training-only exploration and Critic delta/sign supervision",
            "actor_update": "twin-Critic gradient plus trust; no winner/response BC",
            "analytic_dbm_gradient": False,
            "terminal_contextual_bandit": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "j16_summary_sha256": j16_hashes,
        },
        "test_policy": "formal validation and test remain sealed",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
