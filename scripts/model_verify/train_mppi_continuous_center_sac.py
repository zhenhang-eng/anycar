#!/usr/bin/env python3
"""Bootstrap a continuous 16-D MPPI-center SAC Actor and twin Critics.

This stage deliberately does not claim an RL result.  The squashed-Gaussian
Actor is behavior-cloned from the frozen T1 center, while the twin Critics are
initialized only on centers with stored fixed-DBM rewards.  The saved model is
the starting point for actor-visited DBM replay; offline Q extrapolation is not
used to qualify or deploy the Actor.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TorchMPPIContinuousCenterActor,
    TorchMPPIContinuousCenterCritic,
)
from train_mppi_multidirection_actor_critic import load_rewards
from train_mppi_single_step_state_actor_critic import build_inputs
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
    "dbm_two_pass_multidirection_replay_diverse_20260805_v1"
)
DEFAULT_T1 = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_teacher_t1_diverse_20260805_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/continuous_center_sac_bootstrap_20260806_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--risk-labels", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--t1-labels", type=Path, default=DEFAULT_T1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=70)
    parser.add_argument("--maximum-delta-sigma", type=float, default=2.0)
    parser.add_argument(
        "--bc-target", choices=("bank_oracle", "t1"), default="bank_oracle",
        help="Initialization target only; neither target constrains later continuous actions.",
    )
    parser.add_argument("--bc-epochs", type=int, default=60)
    parser.add_argument("--critic-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--critic-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value).to(device)


def load_center_targets(
    source_root: Path,
    parent_root: Path,
    label_root: Path,
    t1_root: Path,
    episodes: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return T1 clone targets, stored continuous replay centers, and sigma."""
    episode_set = set(episodes)
    teacher_targets: list[np.ndarray] = []
    replay_centers: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    for path in sorted(label_root.glob("episode_*/*.npz")):
        if path.parent.name not in episode_set:
            continue
        episode = path.parent.name
        source_path = source_root / episode / "snapshots" / path.name
        parent_path = parent_root / episode / path.name
        t1_path = t1_root / episode / path.name
        with np.load(path, allow_pickle=False) as label, np.load(
            parent_path, allow_pickle=False
        ) as parent, np.load(t1_path, allow_pickle=False) as t1, np.load(
            source_path, allow_pickle=False
        ) as source:
            centers = np.asarray(label["centers"], np.float32)
            anchors = np.asarray(parent["guided_center_knots"], np.float32)
            if not np.allclose(anchors, label["guided_center_knots"], atol=1e-6):
                raise ValueError(f"anchor mismatch: {path}")
            params = json.loads(str(source["mppi_params_json"]))
            sigma = np.asarray(params["noise_sigma"], np.float32)
            teacher = np.asarray(t1["teacher_center_knots"], np.float32)
            for repeat in range(len(anchors)):
                teacher_targets.append(teacher)
                replay_centers.append(centers[repeat])
                sigmas.append(sigma)
    if not teacher_targets:
        raise ValueError("no continuous center targets for requested episodes")
    return (
        np.asarray(teacher_targets, np.float32),
        np.asarray(replay_centers, np.float32),
        np.asarray(sigmas, np.float32),
    )


def center_to_action(
    center: np.ndarray,
    anchor: np.ndarray,
    sigma: np.ndarray,
    maximum_delta_sigma: float,
) -> np.ndarray:
    denominator = sigma[:, None, None, :] * float(maximum_delta_sigma)
    return np.clip((center - anchor[:, None, :, :]) / denominator, -1.0, 1.0).astype(
        np.float32
    )


def state_batch(
    inputs: tuple[np.ndarray, ...], indices: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(tensor(value[indices], device) for value in inputs)


@torch.no_grad()
def deterministic_action(
    actor: TorchMPPIContinuousCenterActor,
    inputs: tuple[np.ndarray, ...],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    actor.eval()
    rows = []
    for start in range(0, len(inputs[0]), batch_size):
        index = np.arange(start, min(start + batch_size, len(inputs[0])))
        mean, _ = actor(*state_batch(inputs, index, device))
        rows.append(torch.tanh(mean).cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def train_actor_bc(
    actor: TorchMPPIContinuousCenterActor,
    train_inputs: tuple[np.ndarray, ...],
    train_target: np.ndarray,
    validation_inputs: tuple[np.ndarray, ...],
    validation_target: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], int]:
    optimizer = torch.optim.AdamW(
        actor.parameters(), args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.bc_epochs + 1):
        actor.train()
        order = rng.permutation(len(train_target))
        train_losses = []
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            mean, log_std = actor(*state_batch(train_inputs, index, device))
            prediction = torch.tanh(mean)
            target = tensor(train_target[index], device)
            # Clone the mean only.  Keep a moderate stochastic policy for the
            # later actor-visited DBM collection instead of collapsing std.
            loss = F.smooth_l1_loss(prediction, target, beta=0.10)
            loss = loss + 1e-4 * log_std.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        prediction = deterministic_action(
            actor, validation_inputs, args.batch_size, device
        )
        validation_loss = float(np.mean((prediction - validation_target) ** 2))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_action_mse": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in actor.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise AssertionError("BC initialization produced no checkpoint")
    actor.load_state_dict(best_state)
    return best_state, history, best_epoch


@torch.no_grad()
def critic_metrics(
    q1: TorchMPPIContinuousCenterCritic,
    q2: TorchMPPIContinuousCenterCritic,
    inputs: tuple[np.ndarray, ...],
    action: np.ndarray,
    reward: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    q1.eval(); q2.eval()
    prediction = []
    context_count, center_count = action.shape[:2]
    flat_context = np.repeat(np.arange(context_count), center_count)
    flat_action = action.reshape(-1, 8, 2)
    for start in range(0, len(flat_context), batch_size):
        index = np.arange(start, min(start + batch_size, len(flat_context)))
        state = state_batch(inputs, flat_context[index], device)
        a = tensor(flat_action[index], device)
        prediction.append(torch.minimum(q1(*state, a), q2(*state, a)).cpu().numpy())
    predicted = np.concatenate(prediction).reshape(context_count, center_count)
    oracle = reward.argmax(1)
    selected = predicted.argmax(1)
    selected_reward = reward[np.arange(context_count), selected]
    oracle_reward = reward[np.arange(context_count), oracle]
    correlation = float(np.corrcoef(predicted.ravel(), reward.ravel())[0, 1])
    return {
        "mae": float(np.mean(np.abs(predicted - reward))),
        "correlation": correlation,
        "selected_reward_mean": float(selected_reward.mean()),
        "oracle_reward_mean": float(oracle_reward.mean()),
        "argmax_regret_mean": float(np.mean(oracle_reward - selected_reward)),
        "top1_fraction": float(np.mean(selected == oracle)),
    }


def train_critics(
    q1: TorchMPPIContinuousCenterCritic,
    q2: TorchMPPIContinuousCenterCritic,
    train_inputs: tuple[np.ndarray, ...],
    train_action: np.ndarray,
    train_reward: np.ndarray,
    validation_inputs: tuple[np.ndarray, ...],
    validation_action: np.ndarray,
    validation_reward: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[dict[str, Any]], int]:
    parameters = list(q1.parameters()) + list(q2.parameters())
    optimizer = torch.optim.AdamW(
        parameters, args.learning_rate, weight_decay=args.weight_decay
    )
    context_count, center_count = train_action.shape[:2]
    flat_context = np.repeat(np.arange(context_count), center_count)
    flat_action = train_action.reshape(-1, 8, 2)
    flat_reward = train_reward.reshape(-1)
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_state: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] | None = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.critic_epochs + 1):
        q1.train(); q2.train()
        order = rng.permutation(len(flat_context))
        losses = []
        for start in range(0, len(order), args.critic_batch_size):
            index = order[start : start + args.critic_batch_size]
            state = state_batch(train_inputs, flat_context[index], device)
            action = tensor(flat_action[index], device)
            target = tensor(flat_reward[index], device)
            prediction1 = q1(*state, action)
            prediction2 = q2(*state, action)
            loss = F.smooth_l1_loss(prediction1, target, beta=0.25)
            loss = loss + F.smooth_l1_loss(prediction2, target, beta=0.25)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        metrics = critic_metrics(
            q1, q2, validation_inputs, validation_action, validation_reward,
            args.critic_batch_size, device,
        )
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        if metrics["argmax_regret_mean"] < best_score - 1e-5:
            best_score = metrics["argmax_regret_mean"]
            best_epoch = epoch
            best_state = (
                {name: value.detach().cpu().clone() for name, value in q1.state_dict().items()},
                {name: value.detach().cpu().clone() for name, value in q2.state_dict().items()},
            )
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise AssertionError("Critic initialization produced no checkpoint")
    q1.load_state_dict(best_state[0]); q2.load_state_dict(best_state[1])
    return best_state[0], best_state[1], history, best_epoch


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    splits = json.loads((args.labels / "splits.json").read_text())
    state = {
        split: load_state_partition(
            args.source, args.parent_labels, splits[split], "selection"
        )
        for split in ("train", "validation", "test")
    }
    reward = {
        split: load_rewards(
            args.parent_labels, args.risk_labels, args.labels,
            splits[split], "selection",
        )
        for split in ("train", "validation", "test")
    }
    target = {
        split: load_center_targets(
            args.source, args.parent_labels, args.labels, args.t1_labels,
            splits[split],
        )
        for split in ("train", "validation", "test")
    }
    train_state = state["train"]
    state_norm = MPPIProposalNormalization.fit(
        train_state.history, train_state.reference, train_state.current
    )
    feedback_mean = reward["train"].context[:, :74].mean(0).astype(np.float32)
    feedback_std = np.maximum(
        reward["train"].context[:, :74].std(0), 1e-4
    ).astype(np.float32)
    gradient_mean = reward["train"].context[:, 74:].mean(0).astype(np.float32)
    gradient_std = np.maximum(
        reward["train"].context[:, 74:].std(0), 1e-4
    ).astype(np.float32)
    inputs = {
        split: build_inputs(
            state[split], reward[split], state_norm,
            feedback_mean, feedback_std, gradient_mean, gradient_std,
        )
        for split in ("train", "validation", "test")
    }
    clone_action: dict[str, np.ndarray] = {}
    replay_action: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        teacher, centers, sigma = target[split]
        if len(teacher) != len(state[split].anchor):
            raise ValueError(f"target alignment failed for {split}")
        if args.bc_target == "bank_oracle":
            best_index = reward[split].advantage_mean.argmax(1)
            clone_center = centers[np.arange(len(centers)), best_index]
        else:
            clone_center = teacher
        clone_action[split] = center_to_action(
            clone_center[:, None], state[split].anchor, sigma,
            args.maximum_delta_sigma,
        )[:, 0]
        replay_action[split] = center_to_action(
            centers, state[split].anchor, sigma,
            args.maximum_delta_sigma,
        )
    reward_scale = float(max(
        np.quantile(np.abs(reward["train"].advantage_mean), 0.90), 1.0
    ))
    normalized_reward = {
        split: (reward[split].advantage_mean / reward_scale).astype(np.float32)
        for split in ("train", "validation", "test")
    }
    actor = TorchMPPIContinuousCenterActor(
        args.maximum_delta_sigma, args.dropout
    ).to(device)
    q1 = TorchMPPIContinuousCenterCritic(args.dropout).to(device)
    q2 = TorchMPPIContinuousCenterCritic(args.dropout).to(device)
    actor_state, actor_history, actor_epoch = train_actor_bc(
        actor, inputs["train"], clone_action["train"],
        inputs["validation"], clone_action["validation"],
        args, rng, device,
    )
    q1_state, q2_state, critic_history, critic_epoch = train_critics(
        q1, q2, inputs["train"], replay_action["train"], normalized_reward["train"],
        inputs["validation"], replay_action["validation"], normalized_reward["validation"],
        args, rng, device,
    )
    target_q1 = copy.deepcopy(q1).eval()
    target_q2 = copy.deepcopy(q2).eval()
    split_metrics: dict[str, Any] = {}
    for split in ("validation", "test"):
        prediction = deterministic_action(actor, inputs[split], args.batch_size, device)
        split_metrics[split] = {
            "clone_action_rmse": float(
                np.sqrt(np.mean((prediction - clone_action[split]) ** 2))
            ),
            "actor_action_abs_mean": float(np.mean(np.abs(prediction))),
            "actor_action_saturation_fraction": float(np.mean(np.abs(prediction) > 0.98)),
            "stored_critic": critic_metrics(
                q1, q2, inputs[split], replay_action[split], normalized_reward[split],
                args.critic_batch_size, device,
            ),
        }
    checkpoint_path = (args.output_dir / "continuous_center_sac_bootstrap.pt").resolve()
    torch.save(
        {
            "format_version": 1,
            "method": "continuous 16-D center SAC bootstrap",
            "actor_state_dict": actor_state,
            "q1_state_dict": q1_state,
            "q2_state_dict": q2_state,
            "target_q1_state_dict": target_q1.state_dict(),
            "target_q2_state_dict": target_q2.state_dict(),
            "state_normalization": state_norm.to_dict(),
            "feedback_mean": feedback_mean,
            "feedback_std": feedback_std,
            "gradient_mean": gradient_mean,
            "gradient_std": gradient_std,
            "reward_scale": reward_scale,
            "maximum_delta_sigma": args.maximum_delta_sigma,
            "bc_target": args.bc_target,
            "training_args": vars(args),
            "qualification": "bootstrap_only_not_rl_qualified",
        },
        checkpoint_path,
    )
    summary = {
        "format_version": 1,
        "method": "continuous 16-D center SAC bootstrap",
        "checkpoint": str(checkpoint_path),
        "semantics": (
            f"{args.bc_target} behavior-cloned stochastic Actor plus twin Q "
            "initialization on stored DBM-evaluated 33-center replay; no "
            "offline-Q Actor update."
        ),
        "qualification": "BOOTSTRAP_ONLY",
        "actor_parameters": actor.parameter_count,
        "critic_parameters_each": q1.parameter_count,
        "maximum_delta_sigma": args.maximum_delta_sigma,
        "bc_target": args.bc_target,
        "reward_scale": reward_scale,
        "actor_best_epoch": actor_epoch,
        "critic_best_epoch": critic_epoch,
        "metrics": split_metrics,
        "next_gate": (
            "Collect actor-visited arbitrary continuous centers with real fixed-DBM "
            "reward, update twin Critics and Actor between batches, then audit on "
            "disjoint seeds."
        ),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps({"actor": actor_history, "critic": critic_history}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
