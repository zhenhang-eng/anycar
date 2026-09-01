#!/usr/bin/env python3
"""Train an episode-heldout deterministic center Actor--Critic on direct DBM labels."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TorchMPPIContinuousCenterCritic,
    TorchMPPIDeterministicCenterActor,
)
from generate_dbm_multicenter_teacher import evaluate_knots, make_controller
from train_mppi_two_pass_feedback_critic import load_partition as load_state_partition


DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1"
)
DEFAULT_PARENT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_feedback_diverse_20260805_v1"
)
DEFAULT_RISK = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_risk_replay_diverse_20260805_v1"
)
DEFAULT_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_direct_center_replay_diverse_20260806_v1"
)
DEFAULT_BOOTSTRAP = Path(
    "outputs/mppi_proposal/continuous_center_sac_bootstrap_20260806_v2/"
    "continuous_center_sac_bootstrap.pt"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_center_actor_critic_diverse_20260806_v1"
)


@dataclass
class DirectPartition:
    action: np.ndarray
    reward: np.ndarray
    direct_cost: np.ndarray
    support: np.ndarray
    feedback: np.ndarray
    gradient: np.ndarray
    center_names: tuple[str, ...]
    label_paths: list[Path]
    context_in_file: np.ndarray
    sigma: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--risk-labels", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--bootstrap", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument(
        "--critic-bootstrap", type=Path, default=None,
        help="Optional prior Direct Actor--Critic checkpoint for continued Q learning.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--critic-epochs", type=int, default=50)
    parser.add_argument("--actor-epochs", type=int, default=40)
    parser.add_argument("--critic-patience", type=int, default=10)
    parser.add_argument("--critic-batch-contexts", type=int, default=16)
    parser.add_argument("--actor-batch-contexts", type=int, default=64)
    parser.add_argument("--critic-learning-rate", type=float, default=2e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--ranking-weight", type=float, default=0.10)
    parser.add_argument(
        "--directional-weight", type=float, default=0.0,
        help="Finite-difference slope loss weight for local antithetic pairs.",
    )
    parser.add_argument("--directional-selection-weight", type=float, default=10.0)
    parser.add_argument("--actor-q-weight", type=float, default=1.0)
    parser.add_argument("--actor-trust-weight", type=float, default=0.0)
    parser.add_argument(
        "--actor-bc-scope", choices=("all", "local"), default="all"
    )
    parser.add_argument("--bc-weight-start", type=float, default=2.0)
    parser.add_argument("--bc-weight-end", type=float, default=0.50)
    parser.add_argument("--actor-eval-interval", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_initial_actor_state(
    actor: TorchMPPIDeterministicCenterActor, checkpoint: dict[str, Any]
) -> None:
    """Load either the historical stochastic bootstrap or a Direct Actor."""
    if checkpoint.get("actor_class") == "TorchMPPIDeterministicCenterActor":
        actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    else:
        actor.load_stochastic_actor_state_dict(checkpoint["actor_state_dict"])


def load_direct_partition(
    label_root: Path,
    parent_root: Path,
    risk_root: Path,
    episodes: list[str],
    maximum_delta_sigma: float,
) -> DirectPartition:
    episode_set = set(episodes)
    actions, rewards, costs, masks = [], [], [], []
    feedbacks, gradients, sigmas = [], [], []
    label_paths: list[Path] = []
    contexts = []
    names: tuple[str, ...] | None = None
    for path in sorted(label_root.glob("episode_*/*.npz")):
        if path.parent.name not in episode_set:
            continue
        parent_path = parent_root / path.parent.name / path.name
        risk_path = risk_root / path.parent.name / path.name
        with np.load(path, allow_pickle=False) as label, np.load(
            parent_path, allow_pickle=False
        ) as parent, np.load(risk_path, allow_pickle=False) as risk:
            one_names = tuple(label["center_names"].astype(str))
            if names is None:
                names = one_names
            elif names != one_names:
                raise ValueError(f"center names differ: {path}")
            centers = np.asarray(label["centers"], np.float32)
            anchors = np.asarray(label["anchor_center"], np.float32)
            sigma = np.asarray(label["sigma"], np.float32)
            normalized = (
                (centers - anchors[:, None])
                / (float(maximum_delta_sigma) * sigma.reshape(1, 1, 1, 2))
            )
            support = np.all(np.abs(normalized) <= 1.0 + 1e-6, axis=(2, 3))
            action = np.clip(normalized, -1.0, 1.0).astype(np.float32)
            direct_cost = np.asarray(label["direct_cost"], np.float32)
            reward = direct_cost[:, :1] - direct_cost
            feedback = np.asarray(parent["first_pass_feedback"], np.float32)
            gradient = np.concatenate((
                np.asarray(risk["critic_gradient_mean"], np.float32),
                np.asarray(risk["critic_gradient_std"], np.float32),
            ), axis=1)
            for context in range(len(anchors)):
                actions.append(action[context])
                rewards.append(reward[context])
                costs.append(direct_cost[context])
                masks.append(support[context])
                feedbacks.append(feedback[context])
                gradients.append(gradient[context])
                sigmas.append(sigma)
                label_paths.append(path)
                contexts.append(context)
    if not actions or names is None:
        raise ValueError("no direct replay labels for requested episodes")
    return DirectPartition(
        action=np.asarray(actions, np.float32),
        reward=np.asarray(rewards, np.float32),
        direct_cost=np.asarray(costs, np.float32),
        support=np.asarray(masks, bool),
        feedback=np.asarray(feedbacks, np.float32),
        gradient=np.asarray(gradients, np.float32),
        center_names=names,
        label_paths=label_paths,
        context_in_file=np.asarray(contexts, np.int32),
        sigma=np.asarray(sigmas, np.float32),
    )


def build_inputs(state: Any, replay: DirectPartition, checkpoint: dict) -> tuple[np.ndarray, ...]:
    if len(state.history) != len(replay.action):
        raise ValueError("state/direct replay context counts differ")
    normalization = MPPIProposalNormalization.from_dict(
        checkpoint["state_normalization"]
    )
    history, reference, current = normalization.normalize_numpy(
        state.history, state.reference, state.current
    )
    return (
        history.astype(np.float32),
        reference.astype(np.float32),
        current.astype(np.float32),
        state.anchor.astype(np.float32),
        ((replay.feedback - checkpoint["feedback_mean"]) / checkpoint["feedback_std"]).astype(np.float32),
        ((replay.gradient - checkpoint["gradient_mean"]) / checkpoint["gradient_std"]).astype(np.float32),
    )


def state_batch(
    inputs: tuple[np.ndarray, ...], index: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(value[index]).to(device) for value in inputs)


def critic_bank_metrics(
    q1: TorchMPPIContinuousCenterCritic,
    q2: TorchMPPIContinuousCenterCritic,
    inputs: tuple[np.ndarray, ...],
    replay: DirectPartition,
    reward_scale: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    q1.eval()
    q2.eval()
    prediction = []
    with torch.no_grad():
        for start in range(0, len(replay.action), batch_size):
            index = np.arange(start, min(start + batch_size, len(replay.action)))
            action = torch.from_numpy(replay.action[index]).to(device)
            state = state_batch(inputs, index, device)
            prediction.append(torch.minimum(
                q1.forward_action_bank(*state, action),
                q2.forward_action_bank(*state, action),
            ).cpu().numpy())
    predicted = np.concatenate(prediction) * reward_scale
    mask = replay.support
    target = replay.reward
    selected = np.where(mask, predicted, -np.inf).argmax(1)
    oracle = np.where(mask, target, -np.inf).argmax(1)
    chosen = target[np.arange(len(selected)), selected]
    oracle_value = target[np.arange(len(oracle)), oracle]
    valid_prediction = predicted[mask]
    valid_target = target[mask]
    correlation = float(np.corrcoef(valid_prediction, valid_target)[0, 1])
    positive, negative = local_pair_indices(replay.center_names)
    action_delta = np.linalg.norm(
        (replay.action[:, positive] - replay.action[:, negative]).reshape(
            len(replay.action), len(positive), -1
        ),
        axis=2,
    )
    pair_mask = mask[:, positive] & mask[:, negative] & (action_delta > 1e-6)
    target_slope = (
        target[:, positive] - target[:, negative]
    ) / np.maximum(action_delta, 1e-6)
    predicted_slope = (
        predicted[:, positive] - predicted[:, negative]
    ) / np.maximum(action_delta, 1e-6)
    slope_correlation = float(np.corrcoef(
        predicted_slope[pair_mask], target_slope[pair_mask]
    )[0, 1])
    direction_agreement = float(np.mean(
        np.sign(predicted_slope[pair_mask]) == np.sign(target_slope[pair_mask])
    ))
    return {
        "mae": float(np.mean(np.abs(valid_prediction - valid_target))),
        "correlation": correlation,
        "selected_true_advantage_mean": float(chosen.mean()),
        "oracle_advantage_mean": float(oracle_value.mean()),
        "argmax_regret_mean": float(np.mean(oracle_value - chosen)),
        "selected_win_fraction": float(np.mean(chosen > 0)),
        "local_slope_correlation": slope_correlation,
        "local_direction_agreement": direction_agreement,
    }


def local_pair_indices(center_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    lookup = {name: index for index, name in enumerate(center_names)}
    positive = []
    negative = []
    for name, index in lookup.items():
        if name.startswith("actor_h") and name.endswith("_pos"):
            partner = name[:-4] + "_neg"
            if partner not in lookup:
                raise ValueError(f"missing antithetic partner for {name}")
            positive.append(index)
            negative.append(lookup[partner])
    if not positive:
        raise ValueError("no local antithetic pairs found")
    return np.asarray(positive, np.int64), np.asarray(negative, np.int64)


def train_critics(
    q1: TorchMPPIContinuousCenterCritic,
    q2: TorchMPPIContinuousCenterCritic,
    train_inputs: tuple[np.ndarray, ...],
    train: DirectPartition,
    validation_inputs: tuple[np.ndarray, ...],
    validation: DirectPartition,
    reward_scale: float,
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[dict[str, Any]], int]:
    optimizer = torch.optim.AdamW(
        list(q1.parameters()) + list(q2.parameters()),
        args.critic_learning_rate,
        weight_decay=args.weight_decay,
    )
    initial_metrics = critic_bank_metrics(
        q1, q2, validation_inputs, validation, reward_scale,
        args.critic_batch_contexts, device,
    )
    def selection_score(metrics: dict[str, float]) -> float:
        directional_penalty = (
            args.directional_selection_weight
            * (1.0 - metrics["local_direction_agreement"])
            if args.directional_weight > 0.0 else 0.0
        )
        return (
            metrics["argmax_regret_mean"] + 0.05 * metrics["mae"]
            + directional_penalty
        )

    best_score = selection_score(initial_metrics)
    best_epoch = 0
    best_state = (copy.deepcopy(q1.state_dict()), copy.deepcopy(q2.state_dict()))
    stale = 0
    history = [{"epoch": 0, "train_loss": None, **initial_metrics}]
    pair_positive_np, pair_negative_np = local_pair_indices(train.center_names)
    pair_positive = torch.from_numpy(pair_positive_np).to(device)
    pair_negative = torch.from_numpy(pair_negative_np).to(device)
    for epoch in range(1, args.critic_epochs + 1):
        q1.train()
        q2.train()
        order = rng.permutation(len(train.action))
        losses = []
        for start in range(0, len(order), args.critic_batch_contexts):
            index = order[start:start + args.critic_batch_contexts]
            state = state_batch(train_inputs, index, device)
            action = torch.from_numpy(train.action[index]).to(device)
            target = torch.from_numpy(train.reward[index] / reward_scale).to(device)
            mask = torch.from_numpy(train.support[index]).to(device)
            pred1 = q1.forward_action_bank(*state, action)
            pred2 = q2.forward_action_bank(*state, action)
            regression = F.smooth_l1_loss(pred1[mask], target[mask], beta=0.25)
            regression = regression + F.smooth_l1_loss(
                pred2[mask], target[mask], beta=0.25
            )
            masked_target = target.masked_fill(~mask, -torch.inf)
            oracle = masked_target.argmax(1)
            ranking = F.cross_entropy(pred1.masked_fill(~mask, -1e9), oracle)
            ranking = ranking + F.cross_entropy(pred2.masked_fill(~mask, -1e9), oracle)
            loss = regression + args.ranking_weight * ranking
            if args.directional_weight > 0.0:
                delta_norm = torch.linalg.vector_norm(
                    (action[:, pair_positive] - action[:, pair_negative]).flatten(2),
                    dim=2,
                ).clamp_min(1e-6)
                target_slope = (
                    target[:, pair_positive] - target[:, pair_negative]
                ) / delta_norm
                slope1 = (
                    pred1[:, pair_positive] - pred1[:, pair_negative]
                ) / delta_norm
                slope2 = (
                    pred2[:, pair_positive] - pred2[:, pair_negative]
                ) / delta_norm
                pair_mask = mask[:, pair_positive] & mask[:, pair_negative]
                directional = F.smooth_l1_loss(
                    slope1[pair_mask], target_slope[pair_mask], beta=0.10
                )
                directional = directional + F.smooth_l1_loss(
                    slope2[pair_mask], target_slope[pair_mask], beta=0.10
                )
                loss = loss + args.directional_weight * directional
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(q1.parameters()) + list(q2.parameters()), 5.0
            )
            optimizer.step()
            losses.append(float(loss.item()))
        metrics = critic_bank_metrics(
            q1, q2, validation_inputs, validation, reward_scale,
            args.critic_batch_contexts, device,
        )
        score = selection_score(metrics)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(row)
        print(
            f"[critic {epoch:03d}] loss={row['train_loss']:.4f} "
            f"corr={metrics['correlation']:.3f} regret={metrics['argmax_regret_mean']:.3f} "
            f"dir={metrics['local_direction_agreement']:.3f}",
            flush=True,
        )
        if score < best_score - 1e-4:
            best_score = score
            best_epoch = epoch
            best_state = (
                copy.deepcopy(q1.state_dict()), copy.deepcopy(q2.state_dict())
            )
            stale = 0
        else:
            stale += 1
        if stale >= args.critic_patience:
            break
    q1.load_state_dict(best_state[0])
    q2.load_state_dict(best_state[1])
    return best_state[0], best_state[1], history, best_epoch


@torch.no_grad()
def actor_outputs(
    actor: TorchMPPIDeterministicCenterActor,
    inputs: tuple[np.ndarray, ...],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    actor.eval()
    actions, centers = [], []
    for start in range(0, len(inputs[0]), batch_size):
        index = np.arange(start, min(start + batch_size, len(inputs[0])))
        action, center = actor(*state_batch(inputs, index, device))
        actions.append(action.cpu().numpy())
        centers.append(center.cpu().numpy())
    return np.concatenate(actions).astype(np.float32), np.concatenate(centers).astype(np.float32)


@torch.no_grad()
def evaluate_actor_direct_costs(
    actor: TorchMPPIDeterministicCenterActor,
    inputs: tuple[np.ndarray, ...],
    replay: DirectPartition,
    source_root: Path,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    _, actor_center = actor_outputs(actor, inputs, batch_size, device)
    costs = np.empty(len(actor_center), np.float32)
    cursor = 0
    while cursor < len(replay.label_paths):
        path = replay.label_paths[cursor]
        end = cursor
        while end < len(replay.label_paths) and replay.label_paths[end] == path:
            end += 1
        episode = path.parent.name
        source_path = source_root / episode / "snapshots" / path.name
        with np.load(source_path, allow_pickle=False) as source:
            config = {
                "objective": {
                    "cost_weights": json.loads(str(source["cost_weights_json"]))
                }
            }
            controller, backend = make_controller(source, config, device)
            direct_cost, _, _ = evaluate_knots(
                controller,
                backend,
                actor_center[cursor:end],
                torch.from_numpy(source["history"]).to(device),
                torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5),
                torch.from_numpy(source["current_action"]).to(device).reshape(1, 2),
                controller._prepare_reference(source["reference"]),
            )
            costs[cursor:end] = direct_cost
        cursor = end
    return costs


@torch.no_grad()
def evaluate_actor_direct(
    actor: TorchMPPIDeterministicCenterActor,
    inputs: tuple[np.ndarray, ...],
    replay: DirectPartition,
    source_root: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    costs = evaluate_actor_direct_costs(
        actor, inputs, replay, source_root, batch_size, device
    )
    actor_index = replay.center_names.index("bootstrap_actor")
    initial = replay.direct_cost[:, actor_index]
    masked = np.where(replay.support, replay.direct_cost, np.inf)
    oracle = masked.min(1)
    gain = initial - costs
    return {
        "actor_direct_cost_mean": float(costs.mean()),
        "bootstrap_actor_direct_cost_mean": float(initial.mean()),
        "stored_oracle_direct_cost_mean": float(oracle.mean()),
        "gain_vs_bootstrap_mean": float(gain.mean()),
        "gain_vs_bootstrap_median": float(np.median(gain)),
        "gain_vs_bootstrap_p05": float(np.quantile(gain, 0.05)),
        "win_fraction": float(np.mean(gain > 0)),
        "loss_fraction": float(np.mean(gain < 0)),
        "worst_gain": float(gain.min()),
        "oracle_regret_mean": float(np.mean(costs - oracle)),
    }


def train_actor(
    actor: TorchMPPIDeterministicCenterActor,
    q1: TorchMPPIContinuousCenterCritic,
    q2: TorchMPPIContinuousCenterCritic,
    train_inputs: tuple[np.ndarray, ...],
    train: DirectPartition,
    validation_inputs: tuple[np.ndarray, ...],
    validation: DirectPartition,
    reward_scale: float,
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int]:
    for parameter in list(q1.parameters()) + list(q2.parameters()):
        parameter.requires_grad_(False)
    q1.eval()
    q2.eval()
    optimizer = torch.optim.AdamW(
        actor.parameters(), args.actor_learning_rate, weight_decay=args.weight_decay
    )
    bc_support = train.support.copy()
    if args.actor_bc_scope == "local":
        local = np.asarray(
            [name.startswith("actor_h") for name in train.center_names], bool
        )
        bc_support &= local[None]
    train_oracle = np.where(bc_support, train.reward, -np.inf).argmax(1)
    train_target = train.action[np.arange(len(train.action)), train_oracle]
    initial_train_action, _ = actor_outputs(
        actor, train_inputs, args.actor_batch_contexts, device
    )
    initial_metrics = evaluate_actor_direct(
        actor, validation_inputs, validation, args.source,
        args.actor_batch_contexts, device,
    )
    best_cost = initial_metrics["actor_direct_cost_mean"]
    best_epoch = 0
    best_state = copy.deepcopy(actor.state_dict())
    history = [{"epoch": 0, **initial_metrics}]
    for epoch in range(1, args.actor_epochs + 1):
        actor.train()
        fraction = (epoch - 1) / max(args.actor_epochs - 1, 1)
        bc_weight = args.bc_weight_start + fraction * (
            args.bc_weight_end - args.bc_weight_start
        )
        order = rng.permutation(len(train.action))
        losses = []
        for start in range(0, len(order), args.actor_batch_contexts):
            index = order[start:start + args.actor_batch_contexts]
            state = state_batch(train_inputs, index, device)
            action, _ = actor(*state)
            value = torch.minimum(q1(*state, action), q2(*state, action))
            target = torch.from_numpy(train_target[index]).to(device)
            bc = F.smooth_l1_loss(action, target, beta=0.05)
            initial = torch.from_numpy(initial_train_action[index]).to(device)
            trust = F.mse_loss(action, initial)
            loss = (
                -args.actor_q_weight * value.mean()
                + bc_weight * bc
                + args.actor_trust_weight * trust
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "bc_weight": bc_weight,
            "actor_q_weight": args.actor_q_weight,
            "actor_trust_weight": args.actor_trust_weight,
        }
        if epoch % args.actor_eval_interval == 0 or epoch == args.actor_epochs:
            metrics = evaluate_actor_direct(
                actor, validation_inputs, validation, args.source,
                args.actor_batch_contexts, device,
            )
            row.update(metrics)
            print(
                f"[actor {epoch:03d}] cost={metrics['actor_direct_cost_mean']:.3f} "
                f"gain={metrics['gain_vs_bootstrap_mean']:.3f} "
                f"p05={metrics['gain_vs_bootstrap_p05']:.3f}",
                flush=True,
            )
            if metrics["actor_direct_cost_mean"] < best_cost - 1e-4:
                best_cost = metrics["actor_direct_cost_mean"]
                best_epoch = epoch
                best_state = copy.deepcopy(actor.state_dict())
        history.append(row)
    actor.load_state_dict(best_state)
    return best_state, history, best_epoch


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    splits = json.loads((args.labels / "splits.json").read_text())
    bootstrap = torch.load(args.bootstrap, map_location="cpu")
    maximum = float(bootstrap["maximum_delta_sigma"])
    episodes = {"train": splits["train"], "validation": splits["validation"]}
    state = {
        split: load_state_partition(
            args.source, args.parent_labels, episodes[split], "selection"
        ) for split in episodes
    }
    replay = {
        split: load_direct_partition(
            args.labels, args.parent_labels, args.risk_labels,
            episodes[split], maximum,
        ) for split in episodes
    }
    inputs = {
        split: build_inputs(state[split], replay[split], bootstrap)
        for split in episodes
    }
    critic_bootstrap = (
        torch.load(args.critic_bootstrap, map_location="cpu")
        if args.critic_bootstrap is not None else None
    )
    reward_scale = (
        float(critic_bootstrap["reward_scale"])
        if critic_bootstrap is not None else float(max(
            np.quantile(
                np.abs(replay["train"].reward[replay["train"].support]), 0.90
            ),
            1.0,
        ))
    )
    q1 = TorchMPPIContinuousCenterCritic(args.dropout).to(device)
    q2 = TorchMPPIContinuousCenterCritic(args.dropout).to(device)
    if critic_bootstrap is not None:
        q1.load_state_dict(critic_bootstrap["q1_state_dict"], strict=True)
        q2.load_state_dict(critic_bootstrap["q2_state_dict"], strict=True)
    q1_state, q2_state, critic_history, critic_epoch = train_critics(
        q1, q2, inputs["train"], replay["train"],
        inputs["validation"], replay["validation"], reward_scale,
        args, rng, device,
    )
    critic_metrics = {
        split: critic_bank_metrics(
            q1, q2, inputs[split], replay[split], reward_scale,
            args.critic_batch_contexts, device,
        ) for split in episodes
    }
    actor = TorchMPPIDeterministicCenterActor(maximum, args.dropout).to(device)
    load_initial_actor_state(actor, bootstrap)
    actor_state, actor_history, actor_epoch = train_actor(
        actor, q1, q2, inputs["train"], replay["train"],
        inputs["validation"], replay["validation"], reward_scale,
        args, rng, device,
    )
    validation_actor = evaluate_actor_direct(
        actor, inputs["validation"], replay["validation"], args.source,
        args.actor_batch_contexts, device,
    )
    checkpoint_path = (args.output_dir / "direct_center_actor_critic.pt").resolve()
    torch.save({
        "format_version": 1,
        "method": "episode-heldout deterministic direct-center Actor-Critic",
        "actor_class": "TorchMPPIDeterministicCenterActor",
        "actor_state_dict": actor_state,
        "q1_state_dict": q1_state,
        "q2_state_dict": q2_state,
        "state_normalization": bootstrap["state_normalization"],
        "feedback_mean": bootstrap["feedback_mean"],
        "feedback_std": bootstrap["feedback_std"],
        "gradient_mean": bootstrap["gradient_mean"],
        "gradient_std": bootstrap["gradient_std"],
        "maximum_delta_sigma": maximum,
        "reward_scale": reward_scale,
        "training_args": vars(args),
        "qualification": "validation_only_test_sealed",
    }, checkpoint_path)
    summary = {
        "format_version": 1,
        "method": "episode-heldout deterministic direct-center Actor-Critic",
        "qualification": "VALIDATION_ONLY_TEST_SEALED",
        "checkpoint": str(checkpoint_path),
        "train_contexts": len(replay["train"].action),
        "validation_contexts": len(replay["validation"].action),
        "centers_per_context": replay["train"].action.shape[1],
        "out_of_actor_support_fraction": float(1.0 - replay["train"].support.mean()),
        "reward_scale": reward_scale,
        "critic_best_epoch": critic_epoch,
        "critic_metrics": critic_metrics,
        "actor_best_epoch": actor_epoch,
        "validation_actor": validation_actor,
        "test_policy": "episode_105..119 not loaded or evaluated",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "history.json").write_text(json.dumps({
        "critic": critic_history,
        "actor": actor_history,
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
