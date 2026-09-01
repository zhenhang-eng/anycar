#!/usr/bin/env python3
"""Train a train-only continuous Alpha Actor--Critic on verified TR1 lines.

This is a contextual-bandit Actor--Critic: the vehicle state is frozen and there
is no Bellman bootstrap.  Twin Critics learn the complete 21-point deterministic
``alpha -> old_cost - direct_cost`` curve from internal-fit episodes.  A TR2-B
Actor is then updated through the conservative twin-Critic minimum while retaining
its exact hard stay fallback.  Epoch zero competes and only internal-selection
direct DBM rollout may select a checkpoint.  Formal validation and test are never
loaded.
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
    TorchMPPITrustAlphaCritic,
    TorchMPPITrustAlphaPolicy,
)
from generate_dbm_proposal_teacher import repository_state, sha256_file
from train_mppi_direct_trust_region_actor import (
    DEFAULT_LABELS,
    direct_cost,
    distribution,
    load_actor_payload,
    load_dataset,
    tensorize,
)
from train_mppi_direct_trust_alpha_policy import (
    center_from_alpha,
    evaluate,
    extra_tensors,
    make_policy,
    outputs,
    policy_batch,
)


DEFAULT_INITIAL_POLICY = Path(
    "outputs/mppi_proposal/direct_trust_alpha_policy_20260810_v1/"
    "trust_alpha_policy_tail_calibrated.pt"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_alpha_actor_critic_20260810_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--initial-policy", type=Path, default=DEFAULT_INITIAL_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--critic-epochs", type=int, default=100)
    parser.add_argument("--actor-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--critic-evaluation-interval", type=int, default=5)
    parser.add_argument("--actor-evaluation-interval", type=int, default=5)
    parser.add_argument("--critic-learning-rate", type=float, default=2e-4)
    parser.add_argument("--critic-minimum-learning-rate", type=float, default=2e-6)
    parser.add_argument("--actor-learning-rate", type=float, default=2e-5)
    parser.add_argument("--actor-minimum-learning-rate", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--reward-scale", type=float, default=5.0)
    parser.add_argument("--critic-delta-weight", type=float, default=1.0)
    parser.add_argument("--critic-anchor-weight", type=float, default=0.25)
    parser.add_argument("--critic-huber-beta", type=float, default=0.05)
    parser.add_argument("--actor-disagreement-weight", type=float, default=0.10)
    parser.add_argument("--actor-regression-weight", type=float, default=0.50)
    parser.add_argument("--actor-q-weight", type=float, default=0.10)
    parser.add_argument("--actor-policy-improvement-weight", type=float, default=1.0)
    parser.add_argument("--actor-bc-weight-start", type=float, default=0.50)
    parser.add_argument("--actor-bc-weight-end", type=float, default=0.10)
    parser.add_argument("--actor-bc-alpha-weight", type=float, default=1.0)
    parser.add_argument("--policy-improvement-margin", type=float, default=0.02)
    parser.add_argument("--policy-improvement-max-disagreement", type=float, default=0.25)
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


def transform_reward(reward: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.asinh(reward / scale)


def make_critic(
    initial_policy: TorchMPPITrustAlphaPolicy,
    device: torch.device,
    dropout: float,
) -> TorchMPPITrustAlphaCritic:
    critic = TorchMPPITrustAlphaCritic(dropout=dropout).to(device)
    critic.encoder.load_state_dict(initial_policy.encoder.state_dict(), strict=True)
    critic.direction_encoder.load_state_dict(
        initial_policy.direction_encoder.state_dict(), strict=True
    )
    return critic


def critic_bank(
    critic: TorchMPPITrustAlphaCritic,
    tensors: dict[str, Any],
    extra: dict[str, torch.Tensor],
    index: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return critic.forward_alpha_bank(
        *(value[index] for value in tensors["inputs"]),
        extra["direction"][index], extra["rho"][index], extra["scale"][index],
        alpha,
    )


def critic_one(
    critic: TorchMPPITrustAlphaCritic,
    tensors: dict[str, Any],
    extra: dict[str, torch.Tensor],
    index: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return critic(
        *(value[index] for value in tensors["inputs"]),
        extra["direction"][index], extra["rho"][index], extra["scale"][index],
        alpha,
    )


@torch.no_grad()
def critic_metrics(
    q1: TorchMPPITrustAlphaCritic,
    q2: TorchMPPITrustAlphaCritic,
    data,
    tensors,
    extra,
    index: np.ndarray,
    args,
    device,
) -> dict[str, Any]:
    q1.eval()
    q2.eval()
    predictions = []
    for start in range(0, len(index), args.evaluation_batch_size):
        absolute = torch.from_numpy(index[start:start + args.evaluation_batch_size]).to(device)
        alpha = tensors["alpha_grid"][absolute]
        predictions.append(torch.minimum(
            critic_bank(q1, tensors, extra, absolute, alpha),
            critic_bank(q2, tensors, extra, absolute, alpha),
        ).cpu().numpy())
    prediction = np.concatenate(predictions)
    reward = data.old_cost[index, None] - data.direct_line_cost[index]
    target = np.arcsinh(reward / args.reward_scale)
    predicted_index = np.argmax(prediction, axis=1)
    oracle_index = np.argmax(reward, axis=1)
    row = np.arange(len(index))
    chosen_reward = reward[row, predicted_index]
    oracle_reward = reward[row, oracle_index]
    regret = oracle_reward - chosen_reward
    target_delta = np.diff(target, axis=1)
    prediction_delta = np.diff(prediction, axis=1)
    meaningful = np.abs(target_delta) >= 0.01
    sign_accuracy = (
        float(np.mean(np.sign(prediction_delta[meaningful]) == np.sign(target_delta[meaningful])))
        if np.any(meaningful) else 1.0
    )
    return {
        "selection_score": float(np.mean(regret) + 0.10 * np.sqrt(np.mean((prediction - target) ** 2))),
        "transformed_reward_rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "transformed_reward_mae": float(np.mean(np.abs(prediction - target))),
        "adjacent_sign_accuracy": sign_accuracy,
        "predicted_argmax_exact_fraction": float(np.mean(predicted_index == oracle_index)),
        "predicted_argmax_alpha_mae": float(np.mean(np.abs(
            data.alpha_grid[index][row, predicted_index]
            - data.alpha_grid[index][row, oracle_index]
        ))),
        "predicted_argmax_regret": distribution(regret),
        "predicted_argmax_true_reward": distribution(chosen_reward),
        "oracle_reward": distribution(oracle_reward),
    }


def context_weights(data, index: np.ndarray) -> torch.Tensor:
    value = np.asarray(data.episode_weight[index], np.float32)
    return torch.from_numpy(value / np.mean(value))


def train_critics(
    seed,
    initial_policy,
    data,
    tensors,
    extra,
    fit_index,
    selection_index,
    args,
    device,
):
    set_seed(seed)
    q1 = make_critic(initial_policy, device, dropout=0.05)
    # Offset the second initialization while preserving the requested run seed.
    torch.manual_seed(seed + 100003)
    q2 = make_critic(initial_policy, device, dropout=0.05)
    optimizer = torch.optim.AdamW(
        list(q1.parameters()) + list(q2.parameters()),
        lr=args.critic_learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.critic_epochs, eta_min=args.critic_minimum_learning_rate
    )
    weights = context_weights(data, fit_index).to(device)
    rng = np.random.default_rng(seed)
    history = []
    best_state = None
    best_epoch = -1
    best_score = float("inf")
    for epoch in range(1, args.critic_epochs + 1):
        q1.train()
        q2.train()
        order = rng.permutation(len(fit_index))
        losses = []
        for start in range(0, len(order), args.batch_size):
            local_np = order[start:start + args.batch_size]
            local = torch.from_numpy(local_np).to(device)
            absolute = torch.from_numpy(fit_index[local_np]).to(device)
            alpha = tensors["alpha_grid"][absolute]
            reward = tensors["old_cost"][absolute, None] - tensors["direct_line_cost"][absolute]
            target = transform_reward(reward, args.reward_scale)
            pred1 = critic_bank(q1, tensors, extra, absolute, alpha)
            pred2 = critic_bank(q2, tensors, extra, absolute, alpha)
            per_value = (
                F.smooth_l1_loss(pred1, target, beta=args.critic_huber_beta, reduction="none").mean(1)
                + F.smooth_l1_loss(pred2, target, beta=args.critic_huber_beta, reduction="none").mean(1)
            )
            target_delta = target[:, 1:] - target[:, :-1]
            per_delta = (
                F.smooth_l1_loss(
                    pred1[:, 1:] - pred1[:, :-1], target_delta,
                    beta=args.critic_huber_beta, reduction="none",
                ).mean(1)
                + F.smooth_l1_loss(
                    pred2[:, 1:] - pred2[:, :-1], target_delta,
                    beta=args.critic_huber_beta, reduction="none",
                ).mean(1)
            )
            per_anchor = pred1[:, 0].square() + pred2[:, 0].square()
            per = (
                per_value
                + args.critic_delta_weight * per_delta
                + args.critic_anchor_weight * per_anchor
            )
            loss = torch.sum(per * weights[local]) / torch.sum(weights[local])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(q1.parameters()) + list(q2.parameters()), 5.0
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        if epoch % args.critic_evaluation_interval == 0 or epoch == args.critic_epochs:
            metrics = critic_metrics(
                q1, q2, data, tensors, extra, selection_index, args, device
            )
            metrics.update({
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            })
            history.append(metrics)
            score = float(metrics["selection_score"])
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = (copy.deepcopy(q1.state_dict()), copy.deepcopy(q2.state_dict()))
            print(
                f"[seed={seed} critic={epoch:03d}] score={score:.4f} "
                f"regret={metrics['predicted_argmax_regret']['mean']:.4f} "
                f"rmse={metrics['transformed_reward_rmse']:.4f} "
                f"sign={metrics['adjacent_sign_accuracy']:.3f}",
                flush=True,
            )
    if best_state is None:
        raise AssertionError("critic checkpoint selection did not run")
    q1.load_state_dict(best_state[0], strict=True)
    q2.load_state_dict(best_state[1], strict=True)
    return q1, q2, best_epoch, history


def policy_eval_args(args, initial_payload) -> argparse.Namespace:
    return argparse.Namespace(
        evaluation_batch_size=args.evaluation_batch_size,
        move_threshold=float(initial_payload["move_threshold"]),
        regression_mean_penalty=args.regression_mean_penalty,
        regression_p95_penalty=args.regression_p95_penalty,
    )


def gate_metrics(metrics: dict[str, Any]) -> dict[str, bool]:
    gain = metrics["gain_vs_old"]
    return {
        "mean_positive": gain["mean"] > 0.0,
        "median_nonnegative": gain["median"] >= -1e-6,
        "p05_nonnegative": gain["p05"] >= -1e-6,
        "worst_at_least_minus_5": gain["minimum"] >= -5.0,
        "stay_recall_at_least_half": metrics["stay_recall"] >= 0.5,
    }


@torch.no_grad()
def calibrate_policy_threshold(
    policy, data, tensors, extra, index, eval_args, device
):
    # Every sigmoid probability is positive, so threshold zero evaluates the
    # conditional-alpha center exactly once.  All threshold candidates then
    # reuse that deterministic direct cost and the old-alpha-zero cost.
    always_args = copy.copy(eval_args)
    always_args.move_threshold = 0.0
    probability, _, _, centers = outputs(
        policy, tensors, extra, index, always_args, device
    )
    move_cost = direct_cost(
        centers, data, tensors, index,
        eval_args.evaluation_batch_size, device,
    )
    old = data.old_cost[index]
    target_move = data.safe_index[index] > 0
    thresholds = np.concatenate((
        np.arange(0.50, 1.00, 0.01), np.asarray((0.995, 0.999))
    ))
    passing = []
    for threshold in thresholds:
        predicted_move = probability > threshold
        cost = np.where(predicted_move, move_cost, old)
        gain = old - cost
        stay_recall = (
            float(np.mean(~predicted_move[~target_move]))
            if np.any(~target_move) else 1.0
        )
        gates = {
            "mean_positive": float(np.mean(gain)) > 0.0,
            "median_nonnegative": float(np.median(gain)) >= -1e-6,
            "p05_nonnegative": float(np.quantile(gain, 0.05)) >= -1e-6,
            "worst_at_least_minus_5": float(np.min(gain)) >= -5.0,
            "stay_recall_at_least_half": stay_recall >= 0.5,
        }
        if all(gates.values()):
            passing.append((float(np.mean(cost)), float(threshold)))
    if not passing:
        return None
    threshold = min(passing)[1]
    selected_args = copy.copy(eval_args)
    selected_args.move_threshold = threshold
    metrics = evaluate(
        policy, data, tensors, extra, index, selected_args, device
    )
    metrics["move_threshold"] = threshold
    metrics["gates"] = gate_metrics(metrics)
    metrics["pass"] = all(metrics["gates"].values())
    return metrics


@torch.no_grad()
def initial_actor_targets(policy, tensors, extra, index: np.ndarray, threshold, device):
    probability, conditional, hard = [], [], []
    policy.eval()
    for start in range(0, len(index), 256):
        absolute = torch.from_numpy(index[start:start + 256]).to(device)
        _, p, a = policy_batch(policy, tensors, extra, absolute)
        probability.append(p.cpu())
        conditional.append(a.cpu())
        hard.append(policy.hard_alpha(p, a, threshold).cpu())
    return tuple(torch.cat(value).to(device) for value in (probability, conditional, hard))


@torch.no_grad()
def critic_policy_targets(q1, q2, tensors, extra, index, args, device):
    target_alpha, target_move, confidence = [], [], []
    q1.eval()
    q2.eval()
    for start in range(0, len(index), 256):
        absolute = torch.from_numpy(index[start:start + 256]).to(device)
        alpha = tensors["alpha_grid"][absolute]
        value1 = critic_bank(q1, tensors, extra, absolute, alpha)
        value2 = critic_bank(q2, tensors, extra, absolute, alpha)
        conservative = torch.minimum(value1, value2)
        best_index = torch.argmax(conservative, dim=1)
        row = torch.arange(len(absolute), device=device)
        best_alpha = alpha[row, best_index]
        advantage = conservative[row, best_index] - conservative[:, 0]
        disagreement = (value1[row, best_index] - value2[row, best_index]).abs()
        move = (
            (best_index > 0)
            & (advantage >= args.policy_improvement_margin)
            & (disagreement <= args.policy_improvement_max_disagreement)
        )
        target_alpha.append(torch.where(move, best_alpha, torch.zeros_like(best_alpha)))
        target_move.append(move.to(best_alpha.dtype))
        confidence.append(torch.clamp(
            advantage / max(args.policy_improvement_margin * 5.0, 1e-6), 0.25, 2.0
        ))
    return tuple(torch.cat(value) for value in (target_alpha, target_move, confidence))


def train_actor(
    seed,
    initial_payload,
    old_payload,
    initial_policy,
    q1,
    q2,
    data,
    tensors,
    extra,
    fit_index,
    selection_index,
    args,
    device,
):
    policy = make_policy(old_payload, device, dropout=0.05)
    policy.load_state_dict(initial_payload["policy_state_dict"], strict=True)
    if args.freeze_actor_context_encoder:
        for parameter in policy.encoder.parameters():
            parameter.requires_grad_(False)
    for critic in (q1, q2):
        critic.eval()
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.actor_learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.actor_epochs, eta_min=args.actor_minimum_learning_rate
    )
    threshold = float(initial_payload["move_threshold"])
    init_probability, init_conditional, init_hard = initial_actor_targets(
        initial_policy, tensors, extra, fit_index, threshold, device
    )
    improve_alpha, improve_move, improve_confidence = critic_policy_targets(
        q1, q2, tensors, extra, fit_index, args, device
    )
    print(
        f"[seed={seed} actor-target] critic_move={float(improve_move.mean()):.3f} "
        f"critic_alpha={float(improve_alpha.mean()):.3f} "
        f"initial_move={float((init_probability > threshold).float().mean()):.3f}",
        flush=True,
    )
    rng = np.random.default_rng(seed + 200003)
    eval_args = policy_eval_args(args, initial_payload)
    history = []
    initial = calibrate_policy_threshold(
        policy, data, tensors, extra, selection_index, eval_args, device
    )
    if initial is None:
        raise AssertionError("calibrated TR2-B epoch zero no longer passes its gate")
    initial_gate = initial["gates"]
    initial.update({
        "epoch": 0,
        "learning_rate": args.actor_learning_rate,
        "gates": initial_gate,
        "pass": all(initial_gate.values()),
    })
    history.append(initial)
    best_state = copy.deepcopy(policy.state_dict())
    best_epoch = 0
    best_metrics = copy.deepcopy(initial)
    best_cost = float(initial["direct_cost"]["mean"])
    fit_weights = context_weights(data, fit_index).to(device)
    for epoch in range(1, args.actor_epochs + 1):
        policy.train()
        # Frozen encoders must remain deterministic even while the policy heads train.
        if args.freeze_actor_context_encoder:
            policy.encoder.eval()
        order = rng.permutation(len(fit_index))
        losses = []
        progress = (epoch - 1) / max(args.actor_epochs - 1, 1)
        bc_weight = (
            args.actor_bc_weight_start * (1.0 - progress)
            + args.actor_bc_weight_end * progress
        )
        for start in range(0, len(order), args.batch_size):
            local_np = order[start:start + args.batch_size]
            local = torch.from_numpy(local_np).to(device)
            absolute = torch.from_numpy(fit_index[local_np]).to(device)
            move_logit, probability, conditional = policy_batch(
                policy, tensors, extra, absolute
            )
            hard = policy.hard_alpha(probability, conditional, threshold)
            soft = probability * conditional
            # Exact deployed alpha in the forward pass, smooth gate in backward.
            alpha_st = hard.detach() + soft - soft.detach()
            value1 = critic_one(q1, tensors, extra, absolute, alpha_st)
            value2 = critic_one(q2, tensors, extra, absolute, alpha_st)
            conservative_value = torch.minimum(value1, value2)
            zero = torch.zeros_like(alpha_st)
            base_value = torch.minimum(
                critic_one(q1, tensors, extra, absolute, zero),
                critic_one(q2, tensors, extra, absolute, zero),
            )
            disagreement = (value1 - value2).abs()
            regression = F.relu(base_value - conservative_value)
            init_move = (init_probability[local] > threshold).to(move_logit.dtype)
            gate_bc = F.binary_cross_entropy_with_logits(
                move_logit, init_move, reduction="none"
            )
            alpha_bc = F.smooth_l1_loss(
                soft, init_hard[local], beta=0.02, reduction="none"
            )
            conditional_bc = torch.zeros_like(alpha_bc)
            moving = init_move > 0.5
            if torch.any(moving):
                conditional_bc[moving] = F.smooth_l1_loss(
                    conditional[moving], init_conditional[local][moving],
                    beta=0.02, reduction="none",
                )
            improve_gate = F.binary_cross_entropy_with_logits(
                move_logit, improve_move[local], reduction="none"
            )
            improve_alpha_loss = F.smooth_l1_loss(
                soft, improve_alpha[local], beta=0.02, reduction="none"
            )
            improve_conditional = torch.zeros_like(improve_alpha_loss)
            improve_moving = improve_move[local] > 0.5
            if torch.any(improve_moving):
                improve_conditional[improve_moving] = F.smooth_l1_loss(
                    conditional[improve_moving], improve_alpha[local][improve_moving],
                    beta=0.02, reduction="none",
                )
            policy_improvement = improve_confidence[local] * (
                improve_gate + improve_alpha_loss + improve_conditional
            )
            per = (
                args.actor_q_weight * (
                    -conservative_value
                    + args.actor_disagreement_weight * disagreement
                    + args.actor_regression_weight * regression
                )
                + args.actor_policy_improvement_weight * policy_improvement
                + bc_weight * (
                    gate_bc
                    + args.actor_bc_alpha_weight * (alpha_bc + conditional_bc)
                )
            )
            loss = torch.sum(per * fit_weights[local]) / torch.sum(fit_weights[local])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        if epoch % args.actor_evaluation_interval == 0 or epoch == args.actor_epochs:
            metrics = calibrate_policy_threshold(
                policy, data, tensors, extra, selection_index, eval_args, device
            )
            if metrics is None:
                print(
                    f"[seed={seed} actor={epoch:03d}] no calibrated threshold passes",
                    flush=True,
                )
                continue
            gates = metrics["gates"]
            metrics.update({
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "bc_weight": float(bc_weight),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "gates": gates,
                "pass": all(gates.values()),
            })
            history.append(metrics)
            cost = float(metrics["direct_cost"]["mean"])
            if metrics["pass"] and cost < best_cost:
                best_cost = cost
                best_epoch = epoch
                best_state = copy.deepcopy(policy.state_dict())
                best_metrics = copy.deepcopy(metrics)
            gain = metrics["gain_vs_old"]
            print(
                f"[seed={seed} actor={epoch:03d}] cost={cost:.4f} "
                f"gain={gain['mean']:.4f} p05={gain['p05']:.3f} "
                f"worst={gain['minimum']:.3f} move={metrics['move_fraction']:.3f} "
                f"threshold={metrics['move_threshold']:.3f} "
                f"pass={metrics['pass']}",
                flush=True,
            )
    policy.load_state_dict(best_state, strict=True)
    policy.eval()
    return policy, best_epoch, best_metrics, history


def grouped_metrics(policy, data, tensors, extra, index, eval_args, device):
    result = {}
    for speed in sorted(np.unique(data.reference_speed[index])):
        one = index[np.isclose(data.reference_speed[index], speed)]
        metrics = evaluate(policy, data, tensors, extra, one, eval_args, device)
        result[str(float(speed))] = {
            "context_count": len(one),
            "old_cost_mean": metrics["old_cost"]["mean"],
            "policy_cost_mean": metrics["direct_cost"]["mean"],
            "gain_mean": metrics["gain_vs_old"]["mean"],
            "gain_p05": metrics["gain_vs_old"]["p05"],
            "gain_worst": metrics["gain_vs_old"]["minimum"],
            "move_fraction": metrics["move_fraction"],
        }
    return result


def checkpoint_payload(
    policy,
    q1,
    q2,
    initial_payload,
    args,
    seed,
    critic_epoch,
    actor_epoch,
    fit_episodes,
    selection_episodes,
    metrics,
    qualification,
):
    return {
        "format_version": 1,
        "method": "train-only continuous Alpha Actor-Critic contextual bandit",
        "qualification": qualification,
        "policy_class": "TorchMPPITrustAlphaPolicy",
        "critic_class": "TorchMPPITrustAlphaCritic",
        "policy_state_dict": copy.deepcopy(policy.cpu().state_dict()),
        "critic1_state_dict": copy.deepcopy(q1.cpu().state_dict()),
        "critic2_state_dict": copy.deepcopy(q2.cpu().state_dict()),
        "state_normalization": initial_payload["state_normalization"],
        "feedback_mean": initial_payload["feedback_mean"],
        "feedback_std": initial_payload["feedback_std"],
        "gradient_mean": initial_payload["gradient_mean"],
        "gradient_std": initial_payload["gradient_std"],
        "old_actor": initial_payload["old_actor"],
        "old_actor_sha256": initial_payload["old_actor_sha256"],
        "proposal_actor": initial_payload["proposal_actor"],
        "proposal_actor_sha256": initial_payload["proposal_actor_sha256"],
        "initial_policy": str(args.initial_policy.resolve()),
        "initial_policy_sha256": sha256_file(args.initial_policy),
        "labels": str(args.labels.resolve()),
        "labels_hashes": {
            name: sha256_file(args.labels / name)
            for name in ("config.json", "splits.json", "summary.json")
        },
        "move_threshold": float(metrics["move_threshold"]),
        "reward_transform": f"asinh(reward/{args.reward_scale})",
        "seed": seed,
        "selected_critic_epoch": critic_epoch,
        "selected_actor_epoch": actor_epoch,
        "fit_episodes": fit_episodes,
        "selection_episodes": selection_episodes,
        "internal_selection_metrics": metrics,
        "training_arguments": vars(args),
        "test_policy": "formal validation and test not loaded or evaluated",
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    initial_payload = torch.load(args.initial_policy, map_location="cpu")
    if initial_payload.get("qualification") != "TR2B_PASS_ALPHA_AC_READY":
        raise AssertionError("initial policy did not pass the TR2-B Alpha AC gate")
    if sha256_file(Path(initial_payload["labels"]) / "summary.json") != initial_payload["labels_hashes"]["summary.json"]:
        raise AssertionError("initial policy TR1 label hash mismatch")
    if args.labels.resolve() != Path(initial_payload["labels"]).resolve():
        raise AssertionError("requested labels differ from calibrated TR2-B source")
    old_payload = load_actor_payload(Path(initial_payload["old_actor"]))
    data, _, splits = load_dataset(args.labels, old_payload, args.max_snapshots)
    device = torch.device(args.device)
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    if args.max_snapshots:
        fit_episodes = selection_episodes = sorted(set(data.episodes.tolist()))
    else:
        fit_episodes = list(splits["internal_fit"])
        selection_episodes = list(splits["internal_selection"])
    fit_index = np.flatnonzero(np.isin(data.episodes, fit_episodes))
    selection_index = np.flatnonzero(np.isin(data.episodes, selection_episodes))
    if not args.max_snapshots and len(fit_index) + len(selection_index) != len(data.episodes):
        raise AssertionError("internal fit/selection do not cover TR1 contexts")
    initial_policy = make_policy(old_payload, device, dropout=0.0)
    initial_policy.load_state_dict(initial_payload["policy_state_dict"], strict=True)
    runs = []
    for seed in args.seeds:
        q1, q2, critic_epoch, critic_history = train_critics(
            seed, initial_policy, data, tensors, extra,
            fit_index, selection_index, args, device,
        )
        policy, actor_epoch, metrics, actor_history = train_actor(
            seed, initial_payload, old_payload, initial_policy, q1, q2,
            data, tensors, extra, fit_index, selection_index, args, device,
        )
        eval_args = policy_eval_args(args, initial_payload)
        eval_args.move_threshold = float(metrics["move_threshold"])
        speed = grouped_metrics(
            policy, data, tensors, extra, selection_index, eval_args, device
        )
        speed_pass = all(row["gain_mean"] >= 0.0 for row in speed.values())
        improved = metrics["direct_cost"]["mean"] < actor_history[0]["direct_cost"]["mean"] - 1e-6
        qualification = (
            "ALPHA_AC_INTERNAL_PASS"
            if metrics["pass"] and speed_pass and improved
            else "ALPHA_AC_RETAIN_TR2B"
        )
        path = args.output_dir / f"alpha_actor_critic_seed{seed}.pt"
        torch.save(checkpoint_payload(
            policy, q1, q2, initial_payload, args, seed, critic_epoch, actor_epoch,
            fit_episodes, selection_episodes, metrics, qualification,
        ), path)
        runs.append({
            "seed": seed,
            "selected_critic_epoch": critic_epoch,
            "selected_actor_epoch": actor_epoch,
            "critic_metrics": critic_metrics(
                q1.to(device), q2.to(device), data, tensors, extra,
                selection_index, args, device,
            ),
            "actor_metrics": metrics,
            "initial_actor_metrics": actor_history[0],
            "by_reference_speed_mps": speed,
            "all_speed_mean_nonregression": speed_pass,
            "qualification": qualification,
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": sha256_file(path),
            "critic_history": critic_history,
            "actor_history": actor_history,
        })
        # Move the frozen models back for the next run's GPU memory.
        q1.cpu()
        q2.cpu()
        policy.cpu()
        torch.cuda.empty_cache()
    passing = [row for row in runs if row["qualification"] == "ALPHA_AC_INTERNAL_PASS"]
    winner = min(
        passing if passing else runs,
        key=lambda row: row["actor_metrics"]["direct_cost"]["mean"],
    )
    selected_source = Path(winner["checkpoint"])
    selected_payload = torch.load(selected_source, map_location="cpu")
    final_qualification = (
        "ALPHA_AC_PASS_READY_TO_FREEZE"
        if passing else "ALPHA_AC_FAIL_RETAIN_TR2B"
    )
    selected_payload["qualification"] = final_qualification
    selected_payload["selection_rule"] = (
        "lowest internal-selection direct DBM cost among mean/median/P05/worst/"
        "stay and all-speed gates; epoch zero competes"
    )
    selected_path = args.output_dir / "alpha_actor_critic_selected.pt"
    torch.save(selected_payload, selected_path)
    summary = {
        "format_version": 1,
        "method": "train-only continuous Alpha Actor-Critic contextual bandit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "labels": str(args.labels.resolve()),
        "initial_policy": str(args.initial_policy.resolve()),
        "fit_episode_count": len(fit_episodes),
        "selection_episode_count": len(selection_episodes),
        "fit_context_count": len(fit_index),
        "selection_context_count": len(selection_index),
        "fit_transition_count": int(len(fit_index) * data.alpha_grid.shape[1]),
        "selection_transition_count": int(len(selection_index) * data.alpha_grid.shape[1]),
        "alpha_grid": data.alpha_grid[0].tolist(),
        "runs": runs,
        "selected_seed": int(winner["seed"]),
        "selected_critic_epoch": int(winner["selected_critic_epoch"]),
        "selected_actor_epoch": int(winner["selected_actor_epoch"]),
        "selected_checkpoint": str(selected_path.resolve()),
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "qualification": final_qualification,
        "test_policy": "formal validation and test not loaded or evaluated",
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({
        "selected_seed": summary["selected_seed"],
        "selected_critic_epoch": summary["selected_critic_epoch"],
        "selected_actor_epoch": summary["selected_actor_epoch"],
        "initial_metrics": winner["initial_actor_metrics"],
        "selected_metrics": winner["actor_metrics"],
        "critic_metrics": winner["critic_metrics"],
        "by_reference_speed_mps": winner["by_reference_speed_mps"],
        "checkpoint": str(selected_path),
        "qualification": final_qualification,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
