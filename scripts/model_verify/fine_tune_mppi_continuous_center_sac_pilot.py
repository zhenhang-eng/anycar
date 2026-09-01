#!/usr/bin/env python3
"""Historical stochastic-MPPI-reward pilot for continuous center SAC.

The five frozen train snapshots are a mechanism/upper-bound pilot, not a
generalization result.  Every new continuous Actor action is evaluated by real
forward-only DBM+MPPI rollout before it enters replay.  Critics and Actor update
between collection batches; audit seeds are read only after model selection.

This entry point is retained to reproduce the 2026-08-06 v1--v3 diagnostics.
Its seed-dependent MPPI weighted-output reward is no longer the primary Actor
objective.  New work must use ``fine_tune_mppi_direct_center_sac_pilot.py``.
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
    TorchMPPIContinuousCenterActor,
    TorchMPPIContinuousCenterCritic,
)
from evaluate_mppi_continuous_center_bootstrap import actor_inputs
from generate_dbm_multicenter_teacher import make_controller
from generate_dbm_sampling_center_gt_pilot import evaluate_centers


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
DEFAULT_BANK = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_multidirection_replay_diverse_20260805_v1"
)
DEFAULT_T1 = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_teacher_t1_diverse_20260805_v1"
)
DEFAULT_CHECKPOINT = Path(
    "outputs/mppi_proposal/continuous_center_sac_bootstrap_20260806_v2/"
    "continuous_center_sac_bootstrap.pt"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/continuous_center_sac_fixed5_pilot_20260806_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--risk-labels", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--bank-labels", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--t1-labels", type=Path, default=DEFAULT_T1)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", nargs="+", default=(
        "episode_000", "episode_021", "episode_042", "episode_063", "episode_084"
    ))
    parser.add_argument("--control-step", type=int, default=250)
    parser.add_argument("--context-index", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--actor-samples-per-context", type=int, default=8)
    parser.add_argument("--updates-per-iteration", type=int, default=80)
    parser.add_argument("--actor-updates-per-iteration", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-5)
    parser.add_argument("--entropy-temperature", type=float, default=0.001)
    parser.add_argument("--bc-weight-start", type=float, default=1.0)
    parser.add_argument("--bc-weight-end", type=float, default=0.05)
    parser.add_argument("--candidate-noise-scale", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--collection-seed-base", type=int, default=31000)
    parser.add_argument("--selection-seeds", type=int, nargs="+", default=(31201, 31202))
    parser.add_argument("--audit-seeds", type=int, nargs="+", default=(31211, 31212, 31213, 31214))
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--allow-historical-stochastic-objective",
        action="store_true",
        help=(
            "Explicitly reproduce the superseded seed-dependent MPPI reward "
            "pilot. New Actor training must use the direct-center script."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Environment:
    episode: str
    model_input: tuple[np.ndarray, ...]
    controller: Any
    backend: Any
    history: torch.Tensor
    initial: torch.Tensor
    current_action: torch.Tensor
    reference: torch.Tensor
    anchor: np.ndarray
    teacher: np.ndarray
    bank_centers: np.ndarray
    bank_reward: np.ndarray
    sigma: np.ndarray
    candidate_sigma: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray


class Replay:
    def __init__(self) -> None:
        self.context: list[int] = []
        self.action: list[np.ndarray] = []
        self.reward: list[float] = []
        self.source: list[str] = []

    def add(
        self, context: int, action: np.ndarray, reward: float, source: str
    ) -> None:
        self.context.append(int(context))
        self.action.append(np.asarray(action, np.float32))
        self.reward.append(float(reward))
        self.source.append(source)

    def sample(
        self, count: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        index = rng.integers(0, len(self.context), size=count)
        return (
            np.asarray(self.context, np.int64)[index],
            np.asarray(self.action, np.float32)[index],
            np.asarray(self.reward, np.float32)[index],
        )

    def __len__(self) -> int:
        return len(self.context)


def load_environments(
    args: argparse.Namespace, checkpoint: dict, device: torch.device
) -> list[Environment]:
    environments = []
    for episode in args.episodes:
        name = f"step_{args.control_step:06d}.npz"
        source_path = args.source / episode / "snapshots" / name
        parent_path = args.parent_labels / episode / name
        risk_path = args.risk_labels / episode / name
        bank_path = args.bank_labels / episode / name
        t1_path = args.t1_labels / episode / name
        with np.load(source_path, allow_pickle=False) as source, np.load(
            parent_path, allow_pickle=False
        ) as parent, np.load(risk_path, allow_pickle=False) as risk, np.load(
            bank_path, allow_pickle=False
        ) as bank, np.load(t1_path, allow_pickle=False) as t1:
            config = {"objective": {"cost_weights": json.loads(str(source["cost_weights_json"]))}}
            controller, backend = make_controller(source, config, device)
            prepared = actor_inputs(
                source, parent, risk, args.context_index, checkpoint, device
            )
            params = json.loads(str(source["mppi_params_json"]))
            sigma = np.asarray(params["noise_sigma"], np.float32)
            environments.append(Environment(
                episode=episode,
                model_input=tuple(value.cpu().numpy()[0].astype(np.float32) for value in prepared),
                controller=controller,
                backend=backend,
                history=torch.from_numpy(source["history"]).to(device),
                initial=torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5),
                current_action=torch.from_numpy(source["current_action"]).to(device).reshape(1, 2),
                reference=controller._prepare_reference(source["reference"]),
                anchor=np.asarray(parent["guided_center_knots"][args.context_index], np.float32),
                teacher=np.asarray(t1["teacher_center_knots"], np.float32),
                bank_centers=np.asarray(bank["centers"][args.context_index], np.float32),
                bank_reward=np.asarray(bank["paired_advantage_mean"][args.context_index], np.float32),
                sigma=sigma,
                candidate_sigma=sigma * float(args.candidate_noise_scale),
                action_min=np.asarray(params["action_min"], np.float32),
                action_max=np.asarray(params["action_max"], np.float32),
            ))
    return environments


def stacked_inputs(environments: list[Environment]) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray([environment.model_input[index] for environment in environments], np.float32)
        for index in range(6)
    )


def state_batch(
    inputs: tuple[np.ndarray, ...], context: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(value[context]).to(device) for value in inputs)


def center_to_action(environment: Environment, center: np.ndarray, maximum: float) -> np.ndarray:
    return np.clip(
        (center - environment.anchor) / (environment.sigma * maximum), -1.0, 1.0
    ).astype(np.float32)


@torch.no_grad()
def actor_centers(
    actor: TorchMPPIContinuousCenterActor,
    inputs: tuple[np.ndarray, ...],
    environments: list[Environment],
    samples_per_context: int,
    device: torch.device,
    deterministic: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    actor.eval()
    centers: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for context, environment in enumerate(environments):
        repeated = tuple(
            torch.from_numpy(np.repeat(value[context:context + 1], samples_per_context, axis=0)).to(device)
            for value in inputs
        )
        action, _, center = actor.sample(*repeated, deterministic=deterministic)
        centers.append(center.cpu().numpy().astype(np.float32))
        actions.append(action.cpu().numpy().astype(np.float32))
    return centers, actions


def evaluate_center_sets(
    environments: list[Environment], center_sets: list[np.ndarray], seeds: list[int]
) -> list[np.ndarray]:
    results = []
    for environment, centers in zip(environments, center_sets):
        evaluation = evaluate_centers(
            centers, seeds, environment.controller, environment.backend,
            environment.history, environment.initial, environment.current_action,
            environment.reference, environment.candidate_sigma,
            environment.action_min, environment.action_max,
        )
        results.append(evaluation["mean_cost"])
    return results


def update_networks(
    actor: TorchMPPIContinuousCenterActor,
    q1: TorchMPPIContinuousCenterCritic,
    q2: TorchMPPIContinuousCenterCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    replay: Replay,
    inputs: tuple[np.ndarray, ...],
    clone_action: np.ndarray,
    bc_weight: float,
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, float]:
    critic_losses = []
    actor_losses = []
    q_values = []
    for _ in range(args.updates_per_iteration):
        context, action, reward = replay.sample(args.batch_size, rng)
        state = state_batch(inputs, context, device)
        action_tensor = torch.from_numpy(action).to(device)
        reward_tensor = torch.from_numpy(reward).to(device)
        q1.train(); q2.train()
        predicted1 = q1(*state, action_tensor)
        predicted2 = q2(*state, action_tensor)
        critic_loss = F.smooth_l1_loss(predicted1, reward_tensor, beta=0.25)
        critic_loss = critic_loss + F.smooth_l1_loss(predicted2, reward_tensor, beta=0.25)
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 5.0)
        critic_optimizer.step()

        critic_losses.append(float(critic_loss.item()))

    # Keep policy-to-new-data ratio deliberately low.  Earlier 1:1 Critic/Actor
    # updates let the Actor exploit transient Q error before the next DBM batch.
    for _ in range(args.actor_updates_per_iteration):
        actor_context = rng.integers(0, len(inputs[0]), size=args.batch_size)
        actor_state = state_batch(inputs, actor_context, device)
        for parameter in list(q1.parameters()) + list(q2.parameters()):
            parameter.requires_grad_(False)
        sampled_action, log_probability, _ = actor.sample(*actor_state)
        value = torch.minimum(
            q1(*actor_state, sampled_action), q2(*actor_state, sampled_action)
        )
        mean, _ = actor(*actor_state)
        deterministic_action = torch.tanh(mean)
        target = torch.from_numpy(clone_action[actor_context]).to(device)
        actor_loss = (
            args.entropy_temperature * log_probability - value
        ).mean() + bc_weight * F.smooth_l1_loss(
            deterministic_action, target, beta=0.05
        )
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        actor_optimizer.step()
        for parameter in list(q1.parameters()) + list(q2.parameters()):
            parameter.requires_grad_(True)
        actor_losses.append(float(actor_loss.item()))
        q_values.append(float(value.mean().item()))
    return {
        "critic_loss": float(np.mean(critic_losses)),
        "actor_loss": float(np.mean(actor_losses)),
        "actor_q": float(np.mean(q_values)),
    }


def main() -> None:
    args = parse_args()
    if not args.allow_historical_stochastic_objective:
        raise RuntimeError(
            "This historical pilot uses a seed-dependent MPPI weighted-output "
            "reward and is invalid for the current Direct Actor gate. Pass "
            "--allow-historical-stochastic-objective only for exact reproduction; "
            "otherwise run fine_tune_mppi_direct_center_sac_pilot.py."
        )
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if set(args.selection_seeds) & set(args.audit_seeds):
        raise ValueError("selection and audit seeds must be disjoint")
    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    actor = TorchMPPIContinuousCenterActor(
        float(checkpoint["maximum_delta_sigma"]), dropout=0.0
    ).to(device)
    q1 = TorchMPPIContinuousCenterCritic(dropout=0.0).to(device)
    q2 = TorchMPPIContinuousCenterCritic(dropout=0.0).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    q1.load_state_dict(checkpoint["q1_state_dict"])
    q2.load_state_dict(checkpoint["q2_state_dict"])
    initial_actor_state = copy.deepcopy(actor.state_dict())
    environments = load_environments(args, checkpoint, device)
    inputs = stacked_inputs(environments)
    maximum = float(checkpoint["maximum_delta_sigma"])
    reward_scale = float(checkpoint["reward_scale"])
    replay = Replay()
    clone_action = []
    for context, environment in enumerate(environments):
        best = int(np.argmax(environment.bank_reward))
        clone_action.append(center_to_action(environment, environment.bank_centers[best], maximum))
        for center, reward in zip(environment.bank_centers, environment.bank_reward):
            replay.add(
                context, center_to_action(environment, center, maximum),
                float(reward / reward_scale), "stored_bank",
            )
    clone_action_array = np.asarray(clone_action, np.float32)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), args.actor_learning_rate)
    critic_optimizer = torch.optim.AdamW(
        list(q1.parameters()) + list(q2.parameters()), args.learning_rate
    )

    initial_centers, _ = actor_centers(
        actor, inputs, environments, 1, device, deterministic=True
    )
    initial_centers = [value[0] for value in initial_centers]
    initial_validation = evaluate_center_sets(
        environments, [value[None] for value in initial_centers], list(args.selection_seeds)
    )
    best_validation_cost = float(np.mean([value[0] for value in initial_validation]))
    best_iteration = 0
    best_actor_state = copy.deepcopy(actor.state_dict())
    history = [{
        "iteration": 0,
        "validation_actor_cost": best_validation_cost,
        "replay_size": len(replay),
    }]
    stale = 0
    for iteration in range(1, args.iterations + 1):
        centers, actions = actor_centers(
            actor, inputs, environments, args.actor_samples_per_context,
            device, deterministic=False,
        )
        # Common-random-number anchor evaluation makes each newly collected
        # reward paired with the actions generated in this batch.
        collection_seeds = [
            args.collection_seed_base + 2 * iteration - 1,
            args.collection_seed_base + 2 * iteration,
        ]
        center_sets = [
            np.concatenate((environment.anchor[None], proposed), axis=0)
            for environment, proposed in zip(environments, centers)
        ]
        costs = evaluate_center_sets(environments, center_sets, collection_seeds)
        for context, (cost, proposed_actions) in enumerate(zip(costs, actions)):
            advantage = cost[0] - cost[1:]
            for action, reward in zip(proposed_actions, advantage):
                replay.add(
                    context, action, float(reward / reward_scale),
                    f"actor_iteration_{iteration}",
                )
        fraction = (iteration - 1) / max(args.iterations - 1, 1)
        bc_weight = args.bc_weight_start + fraction * (
            args.bc_weight_end - args.bc_weight_start
        )
        update = update_networks(
            actor, q1, q2, actor_optimizer, critic_optimizer, replay,
            inputs, clone_action_array, bc_weight, args, rng, device,
        )
        deterministic_centers, _ = actor_centers(
            actor, inputs, environments, 1, device, deterministic=True
        )
        validation = evaluate_center_sets(
            environments, deterministic_centers, list(args.selection_seeds)
        )
        validation_cost = float(np.mean([value[0] for value in validation]))
        row = {
            "iteration": iteration,
            "collection_seeds": collection_seeds,
            "validation_actor_cost": validation_cost,
            "bc_weight": bc_weight,
            "replay_size": len(replay),
            **update,
        }
        history.append(row)
        print(
            f"[{iteration:02d}/{args.iterations:02d}] replay={len(replay)} "
            f"val_cost={validation_cost:.4f} q={update['actor_q']:.4f}",
            flush=True,
        )
        if validation_cost < best_validation_cost - 1e-4:
            best_validation_cost = validation_cost
            best_iteration = iteration
            best_actor_state = copy.deepcopy(actor.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    actor.load_state_dict(best_actor_state); actor.eval()
    final_centers, final_actions = actor_centers(
        actor, inputs, environments, 1, device, deterministic=True
    )
    rows = []
    for context, environment in enumerate(environments):
        actor_replay_indices = [
            index for index, (stored_context, source) in enumerate(
                zip(replay.context, replay.source)
            )
            if stored_context == context and source.startswith("actor_iteration")
        ]
        replay_actions = np.asarray(
            [replay.action[index] for index in actor_replay_indices], np.float32
        )
        replay_centers = np.clip(
            environment.anchor[None]
            + replay_actions * maximum * environment.sigma[None, None, :],
            environment.action_min,
            environment.action_max,
        ).astype(np.float32)
        replay_selection = evaluate_centers(
            replay_centers, list(args.selection_seeds), environment.controller,
            environment.backend, environment.history, environment.initial,
            environment.current_action, environment.reference,
            environment.candidate_sigma, environment.action_min, environment.action_max,
        )
        replay_selected = replay_centers[int(np.argmin(replay_selection["mean_cost"]))]
        comparison = np.concatenate((
            environment.anchor[None], initial_centers[context][None],
            final_centers[context], replay_selected[None], environment.teacher[None],
            environment.bank_centers,
        ))
        selection = evaluate_centers(
            comparison, list(args.selection_seeds), environment.controller,
            environment.backend, environment.history, environment.initial,
            environment.current_action, environment.reference,
            environment.candidate_sigma, environment.action_min, environment.action_max,
        )
        audit = evaluate_centers(
            comparison, list(args.audit_seeds), environment.controller,
            environment.backend, environment.history, environment.initial,
            environment.current_action, environment.reference,
            environment.candidate_sigma, environment.action_min, environment.action_max,
        )
        bank_selected = 5 + int(np.argmin(selection["mean_cost"][5:]))
        bank_clairvoyant = 5 + int(np.argmin(audit["mean_cost"][5:]))
        nearest = np.sqrt(np.mean(
            ((environment.bank_centers - final_centers[context][0]) / environment.sigma) ** 2,
            axis=(1, 2),
        )).min()
        rows.append({
            "episode": environment.episode,
            "anchor_audit_cost": float(audit["mean_cost"][0]),
            "initial_actor_audit_cost": float(audit["mean_cost"][1]),
            "sac_actor_audit_cost": float(audit["mean_cost"][2]),
            "replay_selected_audit_cost": float(audit["mean_cost"][3]),
            "teacher_audit_cost": float(audit["mean_cost"][4]),
            "bank_selected_audit_cost": float(audit["mean_cost"][bank_selected]),
            "bank_clairvoyant_audit_cost": float(audit["mean_cost"][bank_clairvoyant]),
            "sac_gain_vs_initial": float(audit["mean_cost"][1] - audit["mean_cost"][2]),
            "replay_selected_gain_vs_initial": float(
                audit["mean_cost"][1] - audit["mean_cost"][3]
            ),
            "sac_nearest_bank_sigma_rms": float(nearest),
        })
    keys = [key for key in rows[0] if key != "episode"]
    mean = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    checkpoint_path = (args.output_dir / "continuous_center_sac_fixed5.pt").resolve()
    torch.save({
        **checkpoint,
        "format_version": 2,
        "method": "actor-visited continuous center SAC fixed-five pilot",
        "actor_state_dict": best_actor_state,
        "q1_state_dict": q1.state_dict(),
        "q2_state_dict": q2.state_dict(),
        "initial_actor_state_dict": initial_actor_state,
        "best_iteration": best_iteration,
        "training_args": vars(args),
        "qualification": "fixed_five_mechanism_pilot_only",
    }, checkpoint_path)
    summary = {
        "format_version": 1,
        "method": "actor-visited continuous center SAC fixed-five pilot",
        "qualification": "MECHANISM_PILOT_ONLY",
        "checkpoint": str(checkpoint_path),
        "best_iteration": best_iteration,
        "best_selection_cost": best_validation_cost,
        "candidate_budget_per_center": 64,
        "candidate_noise_scale": args.candidate_noise_scale,
        "candidate_noise_design": "zero_extra_antithetic_pairs",
        "selection_seeds": list(args.selection_seeds),
        "audit_seeds": list(args.audit_seeds),
        "actor_visited_new_centers": int(
            sum(source.startswith("actor_iteration") for source in replay.source)
        ),
        "mean": mean,
        "rows": rows,
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "actor_visited_replay.npz",
        context=np.asarray(replay.context, np.int32),
        action=np.asarray(replay.action, np.float32),
        normalized_reward=np.asarray(replay.reward, np.float32),
        source=np.asarray(replay.source),
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
