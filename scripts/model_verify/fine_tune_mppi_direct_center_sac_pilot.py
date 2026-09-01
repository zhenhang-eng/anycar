#!/usr/bin/env python3
"""Deterministic center Actor with decaying structured direct-cost exploration.

The Actor emits one 8x2 knot center.  The existing MPPI linear interpolation
maps it to one 50x2 action sequence and a single forward DBM rollout supplies
its reward.  The Actor itself has one unique output.  Training-only Hadamard
antithetic perturbations create replay actions with a linearly decaying radius;
they are external to the deployed policy and never change an existing label.

A frozen 64-candidate Hadamard neighborhood is evaluated only as a secondary
"is this a useful MPPI center?" metric.  It is not the Actor's primary reward.
The five default snapshots are a mechanism pilot, not a generalization result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from car_dynamics.controllers_torch.mppi import (
    FIXED_HADAMARD_BANK_VERSION,
    fixed_hadamard_knot_noise,
)
from car_foundation.mppi_proposal_policy import (
    TorchMPPIContinuousCenterCritic,
    TorchMPPIDeterministicCenterActor,
)
from fine_tune_mppi_continuous_center_sac_pilot import (
    Replay,
    center_to_action,
    load_environments,
    state_batch,
    stacked_inputs,
)
from generate_dbm_multicenter_teacher import (
    evaluate_actions,
    evaluate_knots,
    stable_weight,
)


DEFAULT_CHECKPOINT = Path(
    "outputs/mppi_proposal/continuous_center_sac_bootstrap_20260806_v2/"
    "continuous_center_sac_bootstrap.pt"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/continuous_center_direct_fixed5_pilot_20260806_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # load_environments consumes these paths and fields.
    parser.add_argument("--source", type=Path, default=Path(
        "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
        "fixed_dbm_policy_diverse_20260805_v1"
    ))
    parser.add_argument("--parent-labels", type=Path, default=Path(
        "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
        "dbm_two_pass_feedback_diverse_20260805_v1"
    ))
    parser.add_argument("--risk-labels", type=Path, default=Path(
        "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
        "dbm_two_pass_risk_replay_diverse_20260805_v1"
    ))
    parser.add_argument("--bank-labels", type=Path, default=Path(
        "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
        "dbm_two_pass_multidirection_replay_diverse_20260805_v1"
    ))
    parser.add_argument("--t1-labels", type=Path, default=Path(
        "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
        "dbm_teacher_t1_diverse_20260805_v1"
    ))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", nargs="+", default=(
        "episode_000", "episode_021", "episode_042", "episode_063", "episode_084"
    ))
    parser.add_argument("--control-step", type=int, default=250)
    parser.add_argument("--context-index", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--exploration-direction-pairs-per-context", type=int, default=4)
    parser.add_argument("--exploration-radius-start", type=float, default=0.30)
    parser.add_argument("--exploration-radius-end", type=float, default=0.05)
    parser.add_argument("--updates-per-iteration", type=int, default=80)
    parser.add_argument("--actor-updates-per-iteration", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-5)
    parser.add_argument("--bc-weight-start", type=float, default=1.0)
    parser.add_argument("--bc-weight-end", type=float, default=0.05)
    parser.add_argument("--fixed-bank-radii", type=float, nargs=2, default=(0.10, 0.30))
    # Compatibility field consumed by the shared snapshot loader.  The direct
    # objective does not use this stochastic candidate scale.
    parser.add_argument("--candidate-noise-scale", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def direct_evaluate(environment: Any, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cost, actions, _ = evaluate_knots(
        environment.controller,
        environment.backend,
        np.asarray(centers, np.float32),
        environment.history,
        environment.initial,
        environment.current_action,
        environment.reference,
    )
    return cost.astype(np.float32), actions.astype(np.float32)


def fixed_bank_contract_hash(radii: tuple[float, float]) -> str:
    normalized = fixed_hadamard_knot_noise(
        (1.0, 1.0), radii=radii, device="cpu"
    ).numpy()
    digest = hashlib.sha256()
    digest.update(FIXED_HADAMARD_BANK_VERSION.encode())
    digest.update(np.asarray(radii, np.float32).tobytes())
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def fixed_neighborhood_evaluate(
    environment: Any, center: np.ndarray, radii: tuple[float, float]
) -> dict[str, Any]:
    noise = fixed_hadamard_knot_noise(
        environment.sigma,
        radii=radii,
        device="cpu",
    ).numpy()
    raw = np.asarray(center, np.float32)[None] + noise
    centers = np.clip(raw, environment.action_min, environment.action_max).astype(np.float32)
    cost, actions = direct_evaluate(environment, centers)
    weight = stable_weight(cost, environment.controller.params.temperature)
    weighted_action = np.sum(weight[:, None, None] * actions, axis=0).astype(np.float32)
    weighted_cost, _ = evaluate_actions(
        environment.controller,
        environment.backend,
        weighted_action[None],
        environment.history,
        environment.initial,
        environment.current_action,
        environment.reference,
    )
    return {
        "candidate_cost": cost,
        "weight": weight.astype(np.float32),
        "weighted_output_cost": float(weighted_cost[0]),
        "best_cost": float(np.min(cost)),
        "p10_cost": float(np.quantile(cost, 0.10)),
        "median_cost": float(np.median(cost)),
        "ess": float(1.0 / np.sum(np.square(weight))),
        "clip_fraction": float(np.mean(raw != centers)),
    }


@torch.no_grad()
def deterministic_actor_centers(
    actor: TorchMPPIDeterministicCenterActor,
    inputs: tuple[np.ndarray, ...],
    environments: list[Any],
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    actor.eval()
    context = np.arange(len(environments), dtype=np.int64)
    action, center = actor(*state_batch(inputs, context, device))
    action_np = action.cpu().numpy().astype(np.float32)
    center_np = center.cpu().numpy().astype(np.float32)
    return (
        [center_np[index] for index in range(len(environments))],
        [action_np[index] for index in range(len(environments))],
    )


def hadamard_action_directions() -> np.ndarray:
    # The positive inner entries are exactly the 16 full-rank Hadamard rows.
    bank = fixed_hadamard_knot_noise(
        (1.0, 1.0), radii=(1.0, 2.0), device="cpu"
    ).numpy()
    directions = bank[2:34:2]
    if np.linalg.matrix_rank(directions.reshape(16, 16)) != 16:
        raise AssertionError("structured exploration directions are not full rank")
    return directions.astype(np.float32)


@torch.no_grad()
def structured_exploration_centers(
    actor: TorchMPPIDeterministicCenterActor,
    inputs: tuple[np.ndarray, ...],
    environments: list[Any],
    iteration: int,
    radius_sigma: float,
    direction_pairs: int,
    maximum_delta_sigma: float,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    if not 1 <= direction_pairs <= 16:
        raise ValueError("exploration direction pairs must be within [1,16]")
    base_centers, base_actions = deterministic_actor_centers(
        actor, inputs, environments, device
    )
    directions = hadamard_action_directions()
    start = ((iteration - 1) * direction_pairs) % len(directions)
    indices = [
        int((start + offset) % len(directions)) for offset in range(direction_pairs)
    ]
    center_sets: list[np.ndarray] = []
    action_sets: list[np.ndarray] = []
    normalized_radius = float(radius_sigma) / float(maximum_delta_sigma)
    for environment, base_center, base_action in zip(
        environments, base_centers, base_actions
    ):
        actions = [base_action]
        for index in indices:
            perturbation = normalized_radius * directions[index]
            actions.extend((
                np.clip(base_action + perturbation, -1.0, 1.0),
                np.clip(base_action - perturbation, -1.0, 1.0),
            ))
        requested = np.asarray(actions, np.float32)
        centers = np.clip(
            environment.anchor[None]
            + requested * maximum_delta_sigma * environment.sigma.reshape(1, 1, 2),
            environment.action_min,
            environment.action_max,
        ).astype(np.float32)
        effective = np.asarray([
            center_to_action(environment, center, maximum_delta_sigma)
            for center in centers
        ], np.float32)
        center_sets.append(centers)
        action_sets.append(effective)
    return center_sets, action_sets, indices


def update_deterministic_networks(
    actor: TorchMPPIDeterministicCenterActor,
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
    for _ in range(args.updates_per_iteration):
        context, action, reward = replay.sample(args.batch_size, rng)
        state = state_batch(inputs, context, device)
        action_tensor = torch.from_numpy(action).to(device)
        reward_tensor = torch.from_numpy(reward).to(device)
        q1.train()
        q2.train()
        predicted1 = q1(*state, action_tensor)
        predicted2 = q2(*state, action_tensor)
        critic_loss = F.smooth_l1_loss(predicted1, reward_tensor, beta=0.25)
        critic_loss = critic_loss + F.smooth_l1_loss(
            predicted2, reward_tensor, beta=0.25
        )
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(q1.parameters()) + list(q2.parameters()), 5.0
        )
        critic_optimizer.step()
        critic_losses.append(float(critic_loss.item()))

    actor_losses = []
    q_values = []
    for _ in range(args.actor_updates_per_iteration):
        actor_context = rng.integers(0, len(inputs[0]), size=args.batch_size)
        actor_state = state_batch(inputs, actor_context, device)
        for parameter in list(q1.parameters()) + list(q2.parameters()):
            parameter.requires_grad_(False)
        action, _ = actor(*actor_state)
        value = torch.minimum(
            q1(*actor_state, action), q2(*actor_state, action)
        )
        target = torch.from_numpy(clone_action[actor_context]).to(device)
        actor_loss = -value.mean() + bc_weight * F.smooth_l1_loss(
            action, target, beta=0.05
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
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 0 < args.exploration_radius_end <= args.exploration_radius_start:
        raise ValueError(
            "exploration radii must satisfy 0 < end <= start"
        )
    radii = (float(args.fixed_bank_radii[0]), float(args.fixed_bank_radii[1]))
    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    actor = TorchMPPIDeterministicCenterActor(
        float(checkpoint["maximum_delta_sigma"]), dropout=0.0
    ).to(device)
    actor.load_stochastic_actor_state_dict(checkpoint["actor_state_dict"])
    initial_actor_state = copy.deepcopy(actor.state_dict())
    # The old Critics predict stochastic MPPI-wrapper reward and are therefore
    # deliberately not loaded into this deterministic direct-cost experiment.
    q1 = TorchMPPIContinuousCenterCritic(dropout=0.0).to(device)
    q2 = TorchMPPIContinuousCenterCritic(dropout=0.0).to(device)
    environments = load_environments(args, checkpoint, device)
    inputs = stacked_inputs(environments)
    maximum = float(checkpoint["maximum_delta_sigma"])

    replay = Replay()
    clone_action = []
    raw_advantages: list[float] = []
    for context, environment in enumerate(environments):
        labeled_centers = np.concatenate(
            (environment.anchor[None], environment.bank_centers, environment.teacher[None]),
            axis=0,
        )
        cost, _ = direct_evaluate(environment, labeled_centers)
        advantage = cost[0] - cost
        best_bank = 1 + int(np.argmin(cost[1:-1]))
        clone_action.append(center_to_action(
            environment, labeled_centers[best_bank], maximum
        ))
        for center, one_advantage in zip(labeled_centers, advantage):
            raw_advantages.append(float(one_advantage))
            replay.add(
                context,
                center_to_action(environment, center, maximum),
                float(one_advantage),
                "direct_relabel",
            )
    reward_scale = float(max(np.quantile(np.abs(raw_advantages), 0.90), 1.0))
    replay.reward = [value / reward_scale for value in replay.reward]
    clone_action_array = np.asarray(clone_action, np.float32)

    actor_optimizer = torch.optim.AdamW(actor.parameters(), args.actor_learning_rate)
    critic_optimizer = torch.optim.AdamW(
        list(q1.parameters()) + list(q2.parameters()), args.learning_rate
    )
    initial_centers, _ = deterministic_actor_centers(
        actor, inputs, environments, device
    )
    initial_cost = [
        float(direct_evaluate(environment, center[None])[0][0])
        for environment, center in zip(environments, initial_centers)
    ]
    best_validation_cost = float(np.mean(initial_cost))
    best_iteration = 0
    best_actor_state = copy.deepcopy(actor.state_dict())
    history = [{
        "iteration": 0,
        "validation_direct_actor_cost": best_validation_cost,
        "replay_size": len(replay),
    }]
    stale = 0
    for iteration in range(1, args.iterations + 1):
        fraction = (iteration - 1) / max(args.iterations - 1, 1)
        exploration_radius = args.exploration_radius_start + fraction * (
            args.exploration_radius_end - args.exploration_radius_start
        )
        centers, actions, direction_indices = structured_exploration_centers(
            actor,
            inputs,
            environments,
            iteration,
            exploration_radius,
            args.exploration_direction_pairs_per_context,
            maximum,
            device,
        )
        for context, (environment, proposed_centers, proposed_actions) in enumerate(
            zip(environments, centers, actions)
        ):
            center_set = np.concatenate((environment.anchor[None], proposed_centers), axis=0)
            cost, _ = direct_evaluate(environment, center_set)
            for action, advantage in zip(proposed_actions, cost[0] - cost[1:]):
                replay.add(
                    context,
                    action,
                    float(advantage / reward_scale),
                    f"actor_iteration_{iteration}",
                )
        bc_weight = args.bc_weight_start + fraction * (
            args.bc_weight_end - args.bc_weight_start
        )
        update = update_deterministic_networks(
            actor,
            q1,
            q2,
            actor_optimizer,
            critic_optimizer,
            replay,
            inputs,
            clone_action_array,
            bc_weight,
            args,
            rng,
            device,
        )
        deterministic_centers, _ = deterministic_actor_centers(
            actor, inputs, environments, device
        )
        validation_cost = float(np.mean([
            direct_evaluate(environment, center[None])[0][0]
            for environment, center in zip(environments, deterministic_centers)
        ]))
        history.append({
            "iteration": iteration,
            "validation_direct_actor_cost": validation_cost,
            "exploration_radius_sigma": exploration_radius,
            "exploration_direction_indices": direction_indices,
            "bc_weight": bc_weight,
            "replay_size": len(replay),
            **update,
        })
        print(
            f"[{iteration:02d}/{args.iterations:02d}] replay={len(replay)} "
            f"direct_cost={validation_cost:.4f} q={update['actor_q']:.4f}",
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

    actor.load_state_dict(best_actor_state)
    actor.eval()
    final_centers, _ = deterministic_actor_centers(
        actor, inputs, environments, device
    )
    rows = []
    fixed_candidate_cost = []
    for context, environment in enumerate(environments):
        visited = [
            index for index, (stored_context, source) in enumerate(
                zip(replay.context, replay.source)
            )
            if stored_context == context and source.startswith("actor_iteration")
        ]
        visited_actions = np.asarray([replay.action[index] for index in visited], np.float32)
        visited_centers = np.clip(
            environment.anchor[None]
            + visited_actions * maximum * environment.sigma.reshape(1, 1, 2),
            environment.action_min,
            environment.action_max,
        ).astype(np.float32)
        visited_cost, _ = direct_evaluate(environment, visited_centers)
        replay_best = visited_centers[int(np.argmin(visited_cost))]
        comparison = np.concatenate((
            environment.anchor[None],
            initial_centers[context][None],
            final_centers[context][None],
            replay_best[None],
            environment.teacher[None],
            environment.bank_centers,
        ))
        cost, _ = direct_evaluate(environment, comparison)
        repeat_cost, _ = direct_evaluate(environment, comparison)
        bank_best = float(np.min(cost[5:]))
        local = fixed_neighborhood_evaluate(environment, final_centers[context], radii)
        fixed_candidate_cost.append(local["candidate_cost"])
        rows.append({
            "episode": environment.episode,
            "anchor_direct_cost": float(cost[0]),
            "initial_actor_direct_cost": float(cost[1]),
            "deterministic_actor_direct_cost": float(cost[2]),
            "replay_best_direct_cost": float(cost[3]),
            "teacher_direct_cost": float(cost[4]),
            "stored_bank_best_direct_cost": bank_best,
            "actor_gain_vs_initial": float(cost[1] - cost[2]),
            "replay_best_gain_vs_initial": float(cost[1] - cost[3]),
            "direct_repeat_max_abs_error": float(np.max(np.abs(cost - repeat_cost))),
            "fixed_neighborhood_weighted_output_cost": local["weighted_output_cost"],
            "fixed_neighborhood_best_cost": local["best_cost"],
            "fixed_neighborhood_p10_cost": local["p10_cost"],
            "fixed_neighborhood_median_cost": local["median_cost"],
            "fixed_neighborhood_ess": local["ess"],
            "fixed_neighborhood_clip_fraction": local["clip_fraction"],
        })

    numeric_keys = [key for key in rows[0] if key != "episode"]
    mean = {
        key: float(np.mean([row[key] for row in rows])) for key in numeric_keys
    }
    contract_hash = fixed_bank_contract_hash(radii)
    checkpoint_path = (args.output_dir / "continuous_center_direct_fixed5.pt").resolve()
    torch.save({
        **checkpoint,
        "format_version": 4,
        "method": "deterministic direct-center Actor-Critic pilot",
        "actor_class": "TorchMPPIDeterministicCenterActor",
        "primary_reward_objective": "anchor_direct_cost - actor_center_direct_cost",
        "actor_state_dict": best_actor_state,
        "q1_state_dict": q1.state_dict(),
        "q2_state_dict": q2.state_dict(),
        "initial_actor_state_dict": initial_actor_state,
        "reward_scale": reward_scale,
        "best_iteration": best_iteration,
        "fixed_candidate_bank_version": FIXED_HADAMARD_BANK_VERSION,
        "fixed_candidate_bank_hash": contract_hash,
        "training_args": vars(args),
        "exploration": {
            "type": "external_fullrank_hadamard_antithetic",
            "radius_schedule_source_sigma": [
                args.exploration_radius_start,
                args.exploration_radius_end,
            ],
            "direction_pairs_per_context": args.exploration_direction_pairs_per_context,
            "part_of_actor_output": False,
        },
        "qualification": "fixed_five_direct_objective_mechanism_pilot_only",
    }, checkpoint_path)
    summary = {
        "format_version": 1,
        "method": "deterministic direct-center Actor-Critic pilot",
        "qualification": "MECHANISM_PILOT_ONLY",
        "primary_objective": {
            "definition": "one 8x2 center -> fixed linear interpolation -> one 50x2 action sequence -> one DBM cost",
            "actor_output_is_unique": True,
            "uses_mppi_candidates": False,
            "uses_reward_seed": False,
            "reward": "anchor_direct_cost - actor_center_direct_cost",
        },
        "training_exploration": {
            "type": "external_fullrank_hadamard_antithetic",
            "part_of_actor_output": False,
            "radius_schedule_source_sigma": [
                args.exploration_radius_start,
                args.exploration_radius_end,
            ],
            "direction_pairs_per_context": args.exploration_direction_pairs_per_context,
            "directions_cycle_every_iterations": int(np.ceil(
                16 / args.exploration_direction_pairs_per_context
            )),
        },
        "secondary_center_quality": {
            "candidate_count": 64,
            "version": FIXED_HADAMARD_BANK_VERSION,
            "contract_hash": contract_hash,
            "radii_source_sigma": list(radii),
            "role": "deterministic auxiliary MPPI-center metric; not Actor reward",
        },
        "checkpoint": str(checkpoint_path),
        "best_iteration": best_iteration,
        "best_direct_cost": best_validation_cost,
        "reward_scale": reward_scale,
        "actor_visited_new_centers": int(sum(
            source.startswith("actor_iteration") for source in replay.source
        )),
        "mean": mean,
        "rows": rows,
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "direct_objective_replay.npz",
        context=np.asarray(replay.context, np.int32),
        action=np.asarray(replay.action, np.float32),
        normalized_direct_reward=np.asarray(replay.reward, np.float32),
        source=np.asarray(replay.source),
        fixed_candidate_cost=np.asarray(fixed_candidate_cost, np.float32),
        fixed_candidate_bank_hash=np.asarray(contract_hash),
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
