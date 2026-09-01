#!/usr/bin/env python3
"""Train a repeated same-state MPPI probe policy with discrete SAC.

The environment is an internal search episode, not a vehicle transition.  The
first action (guided anchor) is observed at reset.  Each Actor inference selects
one unobserved feedback-derived center, a stored fixed-DBM rollout reveals its
cost, and the best observed center is retained.  Actor-visited transitions are
added to a replay buffer; Actor and twin Critics are updated between collection
batches rather than during deployment inference.
"""

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
    TorchMPPISequentialProbeActorCritic,
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
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/sequential_probe_sac_20260805_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--risk-labels", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--probe-budget", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--contexts-per-collect", type=int, default=1200)
    parser.add_argument("--updates-per-iteration", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=250000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--entropy-temperature", type=float, default=0.03)
    parser.add_argument("--actor-loss-weight", type=float, default=0.25)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--reward-clip", type=float, default=10.0)
    parser.add_argument(
        "--reward-mode",
        choices=("immediate_probe", "terminal_independent"),
        default="terminal_independent",
    )
    parser.add_argument("--epsilon-start", type=float, default=0.8)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--evaluation-budgets", default="1,2,3,4,8")
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


@dataclass
class ReplayBatch:
    context: np.ndarray
    value: np.ndarray
    mask: np.ndarray
    remaining: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_value: np.ndarray
    next_mask: np.ndarray
    next_remaining: np.ndarray
    done: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, action_count: int) -> None:
        self.capacity = int(capacity)
        self.action_count = int(action_count)
        self.position = 0
        self.size = 0
        self.context = np.empty(self.capacity, np.int32)
        self.value = np.empty((self.capacity, self.action_count), np.float32)
        self.mask = np.empty((self.capacity, self.action_count), np.bool_)
        self.remaining = np.empty((self.capacity, 1), np.float32)
        self.action = np.empty(self.capacity, np.int64)
        self.reward = np.empty(self.capacity, np.float32)
        self.next_value = np.empty_like(self.value)
        self.next_mask = np.empty_like(self.mask)
        self.next_remaining = np.empty_like(self.remaining)
        self.done = np.empty(self.capacity, np.bool_)

    def add(self, batch: ReplayBatch) -> None:
        count = len(batch.context)
        indices = (np.arange(count) + self.position) % self.capacity
        for name in (
            "context", "value", "mask", "remaining", "action", "reward",
            "next_value", "next_mask", "next_remaining", "done",
        ):
            getattr(self, name)[indices] = getattr(batch, name)
        self.position = int((self.position + count) % self.capacity)
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> ReplayBatch:
        indices = rng.integers(0, self.size, size=batch_size)
        return ReplayBatch(
            **{
                name: getattr(self, name)[indices]
                for name in ReplayBatch.__dataclass_fields__
            }
        )


def tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value).to(device)


def model_forward(
    model: TorchMPPISequentialProbeActorCritic,
    inputs: tuple[np.ndarray, ...],
    context: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray,
    remaining: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base = tuple(tensor(item[context], device) for item in inputs)
    return model(
        *base,
        tensor(value, device),
        tensor(mask, device),
        tensor(remaining, device),
    )


@torch.no_grad()
def choose_action(
    model: TorchMPPISequentialProbeActorCritic,
    inputs: tuple[np.ndarray, ...],
    context: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray,
    remaining: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    epsilon: float,
) -> np.ndarray:
    model.eval()
    logits, _, _ = model_forward(
        model, inputs, context, value, mask, remaining, device
    )
    logits = logits.masked_fill(tensor(mask, device), -1e9)
    action = logits.argmax(1).cpu().numpy()
    explore = rng.random(len(action)) < epsilon
    for index in np.flatnonzero(explore):
        action[index] = rng.choice(np.flatnonzero(~mask[index]))
    return action.astype(np.int64)


def collect_actor_replay(
    model: TorchMPPISequentialProbeActorCritic,
    inputs: tuple[np.ndarray, ...],
    advantage_by_seed: np.ndarray,
    reward_scale: float,
    probe_budget: int,
    context_count: int,
    device: torch.device,
    rng: np.random.Generator,
    epsilon: float,
    reward_mode: str,
    reward_clip: float,
) -> ReplayBatch:
    total_context = len(advantage_by_seed)
    context = rng.choice(
        total_context,
        size=min(context_count, total_context),
        replace=False,
    ).astype(np.int32)
    count = len(context)
    probe_seed = rng.integers(0, advantage_by_seed.shape[2], size=count)
    value = np.zeros((count, advantage_by_seed.shape[1]), np.float32)
    mask = np.zeros_like(value, dtype=np.bool_)
    mask[:, 0] = True
    rows: dict[str, list[np.ndarray]] = {
        name: [] for name in ReplayBatch.__dataclass_fields__
    }
    denominator = max(probe_budget - 1, 1)
    for step in range(probe_budget - 1):
        remaining = np.full(
            (count, 1), (probe_budget - 1 - step) / denominator, np.float32
        )
        action = choose_action(
            model, inputs, context, value, mask, remaining, device, rng, epsilon
        )
        next_value = value.copy()
        next_mask = mask.copy()
        observed = advantage_by_seed[context, action, probe_seed]
        next_value[np.arange(count), action] = np.clip(
            observed / reward_scale, -10.0, 10.0
        )
        next_mask[np.arange(count), action] = True
        best_before = np.where(mask, value, -np.inf).max(1)
        best_after = np.where(next_mask, next_value, -np.inf).max(1)
        done = np.full(count, step == probe_budget - 2, np.bool_)
        if reward_mode == "immediate_probe":
            reward = np.clip(
                best_after - best_before, 0.0, reward_clip
            ).astype(np.float32)
        else:
            reward = np.zeros(count, np.float32)
            if done[0]:
                selected = np.where(
                    next_mask, next_value, -np.inf
                ).argmax(1)
                selected_by_seed = advantage_by_seed[context, selected]
                selected_probe = selected_by_seed[np.arange(count), probe_seed]
                independent = (
                    selected_by_seed.sum(1) - selected_probe
                ) / (advantage_by_seed.shape[2] - 1)
                reward = np.clip(
                    independent / reward_scale, -reward_clip, reward_clip
                ).astype(np.float32)
        next_remaining = np.full(
            (count, 1), (probe_budget - 2 - step) / denominator, np.float32
        )
        next_remaining = np.maximum(next_remaining, 0.0)
        for name, item in (
            ("context", context), ("value", value), ("mask", mask),
            ("remaining", remaining), ("action", action), ("reward", reward),
            ("next_value", next_value), ("next_mask", next_mask),
            ("next_remaining", next_remaining), ("done", done),
        ):
            rows[name].append(item.copy())
        value, mask = next_value, next_mask
    return ReplayBatch(
        **{name: np.concatenate(items, axis=0) for name, items in rows.items()}
    )


def masked_policy(
    logits: torch.Tensor,
    observed_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_logits = logits.masked_fill(observed_mask, -1e9)
    log_probability = F.log_softmax(masked_logits, dim=1)
    return log_probability.exp(), log_probability


def update_network(
    model: TorchMPPISequentialProbeActorCritic,
    target: TorchMPPISequentialProbeActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: ReplayBatch,
    inputs: tuple[np.ndarray, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    logits, q1, q2 = model_forward(
        model, inputs, batch.context, batch.value, batch.mask,
        batch.remaining, device,
    )
    action = tensor(batch.action, device).long()[:, None]
    predicted_q1 = q1.gather(1, action).squeeze(1)
    predicted_q2 = q2.gather(1, action).squeeze(1)
    with torch.no_grad():
        next_logits, _, _ = model_forward(
            model, inputs, batch.context, batch.next_value, batch.next_mask,
            batch.next_remaining, device,
        )
        _, target_q1, target_q2 = model_forward(
            target, inputs, batch.context, batch.next_value, batch.next_mask,
            batch.next_remaining, device,
        )
        probability, log_probability = masked_policy(
            next_logits, tensor(batch.next_mask, device)
        )
        soft_value = (
            probability
            * (
                torch.minimum(target_q1, target_q2)
                - args.entropy_temperature * log_probability
            )
        ).sum(1)
        target_q = tensor(batch.reward, device) + args.gamma * (
            1.0 - tensor(batch.done.astype(np.float32), device)
        ) * soft_value
    critic_loss = F.smooth_l1_loss(
        predicted_q1, target_q, beta=0.25
    ) + F.smooth_l1_loss(predicted_q2, target_q, beta=0.25)
    probability, log_probability = masked_policy(logits, tensor(batch.mask, device))
    actor_loss = (
        probability
        * (
            args.entropy_temperature * log_probability
            - torch.minimum(q1, q2).detach()
        )
    ).sum(1).mean()
    loss = critic_loss + args.actor_loss_weight * actor_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    with torch.no_grad():
        for target_parameter, parameter in zip(target.parameters(), model.parameters()):
            target_parameter.lerp_(parameter, args.target_tau)
    entropy = -(probability * log_probability).sum(1).mean()
    return {
        "loss": float(loss.item()),
        "critic_loss": float(critic_loss.item()),
        "actor_loss": float(actor_loss.item()),
        "entropy": float(entropy.item()),
        "q_mean": float(torch.minimum(predicted_q1, predicted_q2).mean().item()),
        "target_q_mean": float(target_q.mean().item()),
    }


@torch.no_grad()
def evaluate_budgets(
    model: TorchMPPISequentialProbeActorCritic,
    inputs: tuple[np.ndarray, ...],
    advantage_by_seed: np.ndarray,
    reward_scale: float,
    budgets: list[int],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    context_count, action_count, seed_count = advantage_by_seed.shape
    context = np.repeat(np.arange(context_count, dtype=np.int32), seed_count)
    probe_seed = np.tile(np.arange(seed_count), context_count)
    count = len(context)
    requested = sorted(set(budgets))
    result: dict[str, Any] = {}

    def simulate(budget: int) -> None:
        value = np.zeros((count, action_count), np.float32)
        mask = np.zeros_like(value, dtype=np.bool_)
        mask[:, 0] = True
        denominator = max(budget - 1, 1)
        for step in range(1, budget):
            remaining = np.full(
                (count, 1), (budget - step) / denominator, np.float32
            )
            logits, _, _ = model_forward(
                model, inputs, context, value, mask, remaining, device
            )
            action = logits.masked_fill(
                tensor(mask, device), -1e9
            ).argmax(1).cpu().numpy()
            observed = advantage_by_seed[context, action, probe_seed]
            value[np.arange(count), action] = np.clip(
                observed / reward_scale, -10.0, 10.0
            )
            mask[np.arange(count), action] = True
        selected = np.where(mask, value, -np.inf).argmax(1)
        seed_sum = advantage_by_seed[context, selected].sum(1)
        selected_probe = advantage_by_seed[context, selected, probe_seed]
        independent_gain = (seed_sum - selected_probe) / (seed_count - 1)
        result[str(budget)] = {
            "probe_budget": budget,
            "candidate_rollout_budget": int(budget * 64),
            "mean_independent_seed_advantage": float(independent_gain.mean()),
            "median_independent_seed_advantage": float(np.median(independent_gain)),
            "p05_independent_seed_advantage": float(np.quantile(independent_gain, 0.05)),
            "p10_independent_seed_advantage": float(np.quantile(independent_gain, 0.10)),
            "win_fraction": float(np.mean(independent_gain > 0.0)),
            "loss_fraction": float(np.mean(independent_gain < 0.0)),
            "worst_independent_seed_advantage": float(independent_gain.min()),
            "selected_action_histogram": np.bincount(
                selected, minlength=action_count
            ).tolist(),
        }

    for budget in requested:
        simulate(budget)
    probe_table = advantage_by_seed.transpose(0, 2, 1).reshape(
        context_count * seed_count, action_count
    )
    full_selected = probe_table.argmax(1)
    full_context = np.repeat(np.arange(context_count), seed_count)
    full_probe_seed = np.tile(np.arange(seed_count), context_count)
    full_seed_sum = advantage_by_seed[full_context, full_selected].sum(1)
    full_probe = advantage_by_seed[full_context, full_selected, full_probe_seed]
    full_gain = (full_seed_sum - full_probe) / (seed_count - 1)
    result["full_33_probe"] = {
        "probe_budget": action_count,
        "candidate_rollout_budget": int(action_count * 64),
        "mean_independent_seed_advantage": float(full_gain.mean()),
        "median_independent_seed_advantage": float(np.median(full_gain)),
        "p10_independent_seed_advantage": float(np.quantile(full_gain, 0.10)),
        "win_fraction": float(np.mean(full_gain > 0.0)),
        "worst_independent_seed_advantage": float(full_gain.min()),
    }
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.probe_budget < 2 or args.probe_budget > 33:
        raise ValueError("probe budget must be between 2 and 33")
    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    label_summary = json.loads((args.labels / "summary.json").read_text())
    action_names = list(label_summary["action_names"])
    action_count = len(action_names)
    splits = json.loads((args.labels / "splits.json").read_text())
    selection_state = {
        split: load_state_partition(
            args.source, args.parent_labels, splits[split], "selection"
        )
        for split in ("train", "validation", "test")
    }
    selection_reward = {
        split: load_rewards(
            args.parent_labels, args.risk_labels, args.labels,
            splits[split], "selection",
        )
        for split in ("train", "validation", "test")
    }
    audit_state = load_state_partition(
        args.source, args.parent_labels, splits["test"], "audit"
    )
    audit_reward = load_rewards(
        args.parent_labels, args.risk_labels, args.labels,
        splits["test"], "audit",
    )
    train_state = selection_state["train"]
    train_reward = selection_reward["train"]
    state_norm = MPPIProposalNormalization.fit(
        train_state.history, train_state.reference, train_state.current
    )
    feedback_mean = train_reward.context[:, :74].mean(0).astype(np.float32)
    feedback_std = np.maximum(
        train_reward.context[:, :74].std(0), 1e-4
    ).astype(np.float32)
    gradient_mean = train_reward.context[:, 74:].mean(0).astype(np.float32)
    gradient_std = np.maximum(
        train_reward.context[:, 74:].std(0), 1e-4
    ).astype(np.float32)
    selection_inputs = {
        split: build_inputs(
            selection_state[split], selection_reward[split], state_norm,
            feedback_mean, feedback_std, gradient_mean, gradient_std,
        )
        for split in ("train", "validation", "test")
    }
    audit_inputs = build_inputs(
        audit_state, audit_reward, state_norm, feedback_mean, feedback_std,
        gradient_mean, gradient_std,
    )
    reward_scale = float(max(
        np.quantile(np.abs(train_reward.advantage_by_seed), 0.90), 1.0
    ))
    model = TorchMPPISequentialProbeActorCritic(
        action_count, args.dropout
    ).to(device)
    target = copy.deepcopy(model).to(device).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(), args.learning_rate, weight_decay=args.weight_decay
    )
    replay = ReplayBuffer(args.replay_capacity, action_count)
    budgets = [int(value) for value in args.evaluation_budgets.split(",")]
    if max(budgets) > action_count:
        raise ValueError("evaluation budget exceeds action count")
    history: list[dict[str, Any]] = []
    best_score = -np.inf
    best_iteration = 0
    best_state = None
    stale = 0
    for iteration in range(1, args.iterations + 1):
        fraction = (iteration - 1) / max(args.iterations - 1, 1)
        epsilon = args.epsilon_start + fraction * (
            args.epsilon_end - args.epsilon_start
        )
        collected = collect_actor_replay(
            model, selection_inputs["train"], train_reward.advantage_by_seed,
            reward_scale, args.probe_budget, args.contexts_per_collect,
            device, rng, epsilon,
            args.reward_mode,
            args.reward_clip,
        )
        replay.add(collected)
        update_rows = []
        if replay.size >= args.batch_size:
            for _ in range(args.updates_per_iteration):
                update_rows.append(
                    update_network(
                        model, target, optimizer,
                        replay.sample(args.batch_size, rng),
                        selection_inputs["train"], args, device,
                    )
                )
        validation = evaluate_budgets(
            model, selection_inputs["validation"],
            selection_reward["validation"].advantage_by_seed,
            reward_scale, [args.probe_budget], device,
        )[str(args.probe_budget)]
        score = validation["mean_independent_seed_advantage"]
        row = {
            "iteration": iteration,
            "epsilon": epsilon,
            "replay_size": replay.size,
            "validation": validation,
            "updates": {
                key: float(np.mean([value[key] for value in update_rows]))
                for key in update_rows[0]
            } if update_rows else {},
        }
        history.append(row)
        print(
            f"[{iteration:03d}/{args.iterations:03d}] replay={replay.size} "
            f"eps={epsilon:.3f} val_gain={score:.4f}",
            flush=True,
        )
        if score > best_score + 1e-4:
            best_score = score
            best_iteration = iteration
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise AssertionError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    selection_test = evaluate_budgets(
        model, selection_inputs["test"],
        selection_reward["test"].advantage_by_seed,
        reward_scale, budgets, device,
    )
    audit_test = evaluate_budgets(
        model, audit_inputs, audit_reward.advantage_by_seed,
        reward_scale, budgets, device,
    )
    checkpoint_path = (args.output_dir / "sequential_probe_sac.pt").resolve()
    torch.save(
        {
            "format_version": 1,
            "model_type": type(model).__name__,
            "model_state_dict": best_state,
            "state_normalization": state_norm.to_dict(),
            "feedback_mean": feedback_mean,
            "feedback_std": feedback_std,
            "gradient_mean": gradient_mean,
            "gradient_std": gradient_std,
            "reward_scale": reward_scale,
            "action_names": action_names,
            "direction_names": label_summary["direction_names"],
            "radii_sigma": label_summary["radii_sigma"],
            "training_probe_budget": args.probe_budget,
            "training_args": vars(args),
        },
        checkpoint_path,
    )
    summary = {
        "format_version": 1,
        "method": "same-state repeated-probe discrete SAC",
        "checkpoint": str(checkpoint_path),
        "labels": str(args.labels.resolve()),
        "parameter_count": model.parameter_count,
        "action_count": action_count,
        "reward_scale": reward_scale,
        "probe_semantics": (
            "One 64-candidate DBM proposal-output evaluation per observed center; "
            "anchor is observed first and best-so-far is retained."
        ),
        "update_semantics": (
            "Actor/Critics are frozen within deployment search; actor-visited "
            "transitions enter replay and network weights update between batches."
        ),
        "best_iteration": best_iteration,
        "best_validation_mean_independent_seed_advantage": best_score,
        "iterations_run": len(history),
        "training_history": history,
        "selection_test": selection_test,
        "audit_test": audit_test,
        "audit_protocol": (
            "Training and checkpoint selection use selection train/validation only. "
            "Audit/test is read once after the selected checkpoint is frozen; each "
            "reported action is selected by one probe seed and scored on the other three."
        ),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(json.dumps({
        "status": "ok",
        "output": str(args.output_dir),
        "best_iteration": best_iteration,
        "audit_test": audit_test,
    }, indent=2))


if __name__ == "__main__":
    main()
