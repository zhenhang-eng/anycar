#!/usr/bin/env python3
"""Run a shared-prefix Query OAC Actor-LR decay comparison through round 160."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import run_query_oac_gamma1_k_scan as base
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic
from run_query_single_center_oac20to1 import (
    StratifiedQueues,
    actor_from_payload,
    actor_predict,
    candidate_ranking_metrics,
    direct_cost,
    distribution,
    load_inputs,
    response_bank,
    set_seed,
    update_critics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_oac_gamma1_k16_lr_decay_160round_config_20260903_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def deterministic_contract(config: dict) -> dict:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    mode = str(config["pairing"]["deterministic_cuda_mode"])
    if mode not in ("strict", "warn_only"):
        raise AssertionError(f"unsupported deterministic CUDA mode: {mode}")
    torch.use_deterministic_algorithms(True, warn_only=mode == "warn_only")
    return {
        "mode": mode,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def cpu_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in state.items()}


def make_models(
    payload: dict,
    config: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, list[ConfigurableAbsoluteActionValueCritic], list[dict], torch.optim.Optimizer, list[torch.optim.Optimizer]]:
    actor = actor_from_payload(payload, "actor_state_dict", device)
    critics: list[ConfigurableAbsoluteActionValueCritic] = []
    trainings: list[dict] = []
    critic_optimizers: list[torch.optim.Optimizer] = []
    for twin in (1, 2):
        critic = ConfigurableAbsoluteActionValueCritic().to(device)
        critic.load_state_dict(payload[f"critic{twin}_state_dict"], strict=True)
        critics.append(critic)
        trainings.append(payload[f"critic{twin}_training"])
        critic_optimizers.append(torch.optim.AdamW(
            critic.parameters(),
            lr=float(config["critic_updates"]["learning_rate"]),
            weight_decay=float(config["critic_updates"]["weight_decay"]),
        ))
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(config["common_prefix"]["learning_rate_per_microstep"]),
        weight_decay=float(config["actor_updates"]["weight_decay"]),
    )
    return actor, critics, trainings, actor_optimizer, critic_optimizers


def lr_for_round(config: dict, arm: dict, round_index: int) -> float:
    prefix_rounds = int(config["common_prefix"]["rounds"])
    if round_index <= prefix_rounds:
        return float(config["common_prefix"]["learning_rate_per_microstep"])
    if arm["continuation_schedule"] == "fixed":
        return float(arm["continuation_learning_rate"])
    if arm["continuation_schedule"] != "cosine":
        raise AssertionError(f"unsupported schedule {arm['continuation_schedule']}")
    start_round = int(arm["decay_start_round"])
    end_round = int(arm["decay_end_round"])
    if round_index > end_round:
        return float(arm["post_decay_learning_rate"])
    fraction = (round_index - start_round) / max(end_round - start_round, 1)
    start = float(arm["decay_start_learning_rate"])
    end = float(arm["decay_end_learning_rate"])
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * fraction))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


def concatenate_lists(lists: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate(values) for name, values in lists.items()}


def snapshot_training_state(
    actor: torch.nn.Module,
    critics: list[ConfigurableAbsoluteActionValueCritic],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizers: list[torch.optim.Optimizer],
    critic_rng: np.random.Generator,
    selected_state: dict[str, torch.Tensor],
    selected_critic_states: list[dict[str, torch.Tensor]],
    selected_round: int,
    selected_mean: float,
    group_cursor: int,
) -> dict[str, Any]:
    return {
        "actor_state": copy.deepcopy(actor.state_dict()),
        "critic_states": [copy.deepcopy(model.state_dict()) for model in critics],
        "actor_optimizer_state": copy.deepcopy(actor_optimizer.state_dict()),
        "critic_optimizer_states": [copy.deepcopy(item.state_dict()) for item in critic_optimizers],
        "critic_rng_state": copy.deepcopy(critic_rng.bit_generator.state),
        "torch_cpu_rng_state": torch.get_rng_state().clone(),
        "torch_cuda_rng_states": [item.clone() for item in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        "selected_state": copy.deepcopy(selected_state),
        "selected_critic_states": copy.deepcopy(selected_critic_states),
        "selected_round": int(selected_round),
        "selected_mean": float(selected_mean),
        "group_cursor": int(group_cursor),
    }


def restore_training_state(
    snapshot: dict[str, Any],
    payload: dict,
    config: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, list[ConfigurableAbsoluteActionValueCritic], list[dict], torch.optim.Optimizer, list[torch.optim.Optimizer], np.random.Generator]:
    actor, critics, trainings, actor_optimizer, critic_optimizers = make_models(payload, config, device)
    actor.load_state_dict(snapshot["actor_state"], strict=True)
    actor_optimizer.load_state_dict(snapshot["actor_optimizer_state"])
    for critic, state, optimizer, optimizer_state in zip(
        critics,
        snapshot["critic_states"],
        critic_optimizers,
        snapshot["critic_optimizer_states"],
    ):
        critic.load_state_dict(state, strict=True)
        optimizer.load_state_dict(optimizer_state)
    critic_rng = np.random.default_rng()
    critic_rng.bit_generator.state = copy.deepcopy(snapshot["critic_rng_state"])
    torch.set_rng_state(snapshot["torch_cpu_rng_state"])
    if torch.cuda.is_available() and snapshot["torch_cuda_rng_states"]:
        torch.cuda.set_rng_state_all(snapshot["torch_cuda_rng_states"])
    return actor, critics, trainings, actor_optimizer, critic_optimizers, critic_rng


def evaluation_block(
    selection_cost: np.ndarray,
    oof_cost: np.ndarray,
    initial_selection_cost: np.ndarray,
    initial_oof_cost: np.ndarray,
    data: dict[str, np.ndarray],
    selection: np.ndarray,
    oof: np.ndarray,
) -> dict[str, Any]:
    return {
        "inner_warm_relative": base.warm_relative_metrics(
            selection_cost, data["warm_cost"][selection],
            data["speed_kph"][selection], data["variant_index"][selection],
        ),
        "development_oof_warm_relative": base.warm_relative_metrics(
            oof_cost, data["warm_cost"][oof],
            data["speed_kph"][oof], data["variant_index"][oof],
        ),
        "inner_round0_relative": base.metrics(
            selection_cost, initial_selection_cost,
            data["warm_cost"][selection], data["speed_kph"][selection],
        ),
        "development_oof_round0_relative": base.metrics(
            oof_cost, initial_oof_cost,
            data["warm_cost"][oof], data["speed_kph"][oof],
        ),
    }


def execute_round(
    *,
    round_index: int,
    arm: dict,
    config: dict,
    data: dict[str, np.ndarray],
    controller: base.TorchMPPIController,
    weights: dict[str, float],
    bases: np.ndarray,
    radii: np.ndarray,
    chosen_schedule: np.ndarray,
    actor_schedule: np.ndarray,
    fit: np.ndarray,
    selection: np.ndarray,
    inputs: tuple[np.ndarray, ...],
    sigma: np.ndarray,
    device: torch.device,
    actor: torch.nn.Module,
    critics: list[ConfigurableAbsoluteActionValueCritic],
    trainings: list[dict],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizers: list[torch.optim.Optimizer],
    critic_rng: np.random.Generator,
    online_lists: dict[str, list[np.ndarray]],
    actor_batch_lists: dict[str, list[np.ndarray]],
    selection_actions: list[np.ndarray],
    selection_costs: list[np.ndarray],
    selected_state: dict[str, torch.Tensor],
    selected_critic_states: list[dict[str, torch.Tensor]],
    selected_round: int,
    selected_mean: float,
    group_cursor: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], list[dict[str, torch.Tensor]], int, float, int]:
    chosen = chosen_schedule[round_index - 1]
    centers = actor_predict(actor, inputs, chosen, device)
    local_costs = []
    for position, row in enumerate(chosen):
        actions, costs, raw, clipped = response_bank(
            controller, data, int(row), centers[position],
            float(radii[round_index - 1]), bases[(round_index - 1) % len(bases)],
            sigma, weights, config,
        )
        local_costs.append(costs)
        online_lists["state_index"].append(np.full(len(actions), row, np.int64))
        online_lists["action"].append(actions)
        online_lists["cost"].append(costs)
        online_lists["raw_action"].append(raw)
        online_lists["clipped"].append(clipped)
        online_lists["round"].append(np.full(len(actions), round_index, np.int16))
        online_lists["group"].append(np.full(len(actions), group_cursor, np.int32))
        online_lists["role"].append(np.asarray(["actor"] + ["probe"] * 32 + ["response"] * 6))
        group_cursor += 1
    online = concatenate_lists(online_lists)
    critic_history = [
        update_critics(
            critics, critic_optimizers, trainings, inputs, data, fit,
            online, config, critic_rng, device,
        )
        for _ in range(int(config["critic_updates"]["updates_per_round_per_twin"]))
    ]
    batch_rows = actor_schedule[round_index - 1]
    actor_batch_lists["actor_batch_state_index"].append(batch_rows.reshape(-1))
    actor_batch_lists["actor_batch_round"].append(np.full(batch_rows.size, round_index, np.int16))
    actor_batch_lists["actor_batch_microstep"].append(np.repeat(
        np.arange(1, len(batch_rows) + 1, dtype=np.int16), batch_rows.shape[1]
    ))
    selected_actor = copy.deepcopy(actor)
    selected_actor.load_state_dict(selected_state, strict=True)
    lr = lr_for_round(config, arm, round_index)
    set_optimizer_lr(actor_optimizer, lr)
    actor_info = base.gamma1_actor_microsteps(
        actor, selected_actor, actor_optimizer, critics, trainings,
        inputs, fit, batch_rows, config, sigma, device,
    )
    actor_info["learning_rate_per_microstep"] = lr
    selection_action = actor_predict(actor, inputs, selection, device)
    selection_cost = direct_cost(controller, data, selection, selection_action, weights)
    selection_actions.append(selection_action)
    selection_costs.append(selection_cost)
    accepted = float(selection_cost.mean()) < selected_mean
    if accepted:
        selected_round = round_index
        selected_mean = float(selection_cost.mean())
        selected_state = copy.deepcopy(actor.state_dict())
        selected_critic_states = [copy.deepcopy(model.state_dict()) for model in critics]
    recent_count = len(chosen) * int(config["pilot"]["candidates_per_visit"])
    ranking = candidate_ranking_metrics(
        critics, trainings, inputs,
        online["state_index"][-recent_count:], online["action"][-recent_count:],
        online["cost"][-recent_count:], online["group"][-recent_count:], device,
    )
    record = {
        "round": round_index,
        "probe_radius_sigma": float(radii[round_index - 1]),
        "visited_rows": chosen.tolist(),
        "new_candidate_cost": distribution(np.concatenate(local_costs)),
        "critic_loss": {
            "mean": float(np.mean([value["loss_mean"] for value in critic_history])),
            "value": float(np.mean([value["value_loss_mean"] for value in critic_history])),
            "ranking": float(np.mean([value["ranking_loss_mean"] for value in critic_history])),
        },
        "actor_update": actor_info,
        "selection_warm_relative": base.warm_relative_metrics(
            selection_cost, data["warm_cost"][selection],
            data["speed_kph"][selection], data["variant_index"][selection],
        ),
        "selected": bool(accepted),
        "selected_round_after_evaluation": int(selected_round),
        "fresh_candidate_critic": ranking,
    }
    print(
        f"{arm['name']} round={round_index}/160 "
        f"lr={lr:.8g} Jsel={selection_cost.mean():.5f} "
        f"best={selected_mean:.5f}@{selected_round} "
        f"warmAgg={record['selection_warm_relative']['aggregate_improvement']:.4f} "
        f"step={actor_info['final_cumulative_output_step_sigma_rms']:.6f}",
        flush=True,
    )
    return record, selected_state, selected_critic_states, selected_round, selected_mean, group_cursor


def empty_online_lists() -> dict[str, list[np.ndarray]]:
    return {name: [] for name in (
        "state_index", "action", "cost", "raw_action", "clipped", "round", "group", "role"
    )}


def empty_actor_batch_lists() -> dict[str, list[np.ndarray]]:
    return {name: [] for name in (
        "actor_batch_state_index", "actor_batch_round", "actor_batch_microstep"
    )}


def run_seed(
    config: dict,
    seed: int,
    data: dict[str, np.ndarray],
    replay_manifest: dict,
    pretrain_dir: Path,
    controller: base.TorchMPPIController,
    weights: dict[str, float],
    bases: np.ndarray,
    radii: np.ndarray,
    fit: np.ndarray,
    selection: np.ndarray,
    oof: np.ndarray,
    output: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    base_seed = 2_609_120 + seed
    set_seed(base_seed)
    context_rng = np.random.default_rng(base_seed + 11)
    critic_rng = np.random.default_rng(base_seed + 29)
    actor_rng = np.random.default_rng(base_seed + 47)
    rounds = int(config["pilot"]["rounds"])
    prefix_rounds = int(config["common_prefix"]["rounds"])
    k = 16
    queues = StratifiedQueues(data, fit, context_rng)
    chosen_schedule = np.stack([queues.take() for _ in range(rounds)])
    actor_schedule = np.stack([
        np.stack([
            actor_rng.choice(fit, int(config["actor_updates"]["batch_size"]), replace=False).astype(np.int64)
            for _ in range(k)
        ])
        for _ in range(rounds)
    ])
    checkpoint_source = pretrain_dir / "checkpoints" / f"pretrain_fold0_seed{seed}.pt"
    payload = torch.load(checkpoint_source, map_location=device, weights_only=False)
    inputs = load_inputs(data, payload["normalization"])
    actor, critics, trainings, actor_optimizer, critic_optimizers = make_models(payload, config, device)
    sigma = np.asarray(config["pilot"]["noise_sigma"], np.float32).reshape(1, 2)
    online_lists = empty_online_lists()
    actor_batch_lists = empty_actor_batch_lists()
    initial_selection_action = actor_predict(actor, inputs, selection, device)
    initial_selection_cost = direct_cost(controller, data, selection, initial_selection_action, weights)
    initial_oof_action = actor_predict(actor, inputs, oof, device)
    initial_oof_cost = direct_cost(controller, data, oof, initial_oof_action, weights)
    selection_actions = [initial_selection_action]
    selection_costs = [initial_selection_cost]
    selected_round = 0
    selected_mean = float(initial_selection_cost.mean())
    selected_state = copy.deepcopy(actor.state_dict())
    selected_critic_states = [copy.deepcopy(model.state_dict()) for model in critics]
    group_cursor = 0
    prefix_records = []
    prefix_arm = config["arms"][0]
    for round_index in range(1, prefix_rounds + 1):
        result = execute_round(
            round_index=round_index, arm=prefix_arm, config=config, data=data,
            controller=controller, weights=weights, bases=bases, radii=radii,
            chosen_schedule=chosen_schedule, actor_schedule=actor_schedule,
            fit=fit, selection=selection, inputs=inputs, sigma=sigma, device=device,
            actor=actor, critics=critics, trainings=trainings,
            actor_optimizer=actor_optimizer, critic_optimizers=critic_optimizers,
            critic_rng=critic_rng, online_lists=online_lists,
            actor_batch_lists=actor_batch_lists, selection_actions=selection_actions,
            selection_costs=selection_costs, selected_state=selected_state,
            selected_critic_states=selected_critic_states, selected_round=selected_round,
            selected_mean=selected_mean, group_cursor=group_cursor,
        )
        record, selected_state, selected_critic_states, selected_round, selected_mean, group_cursor = result
        prefix_records.append(record)
    fork = snapshot_training_state(
        actor, critics, actor_optimizer, critic_optimizers, critic_rng,
        selected_state, selected_critic_states, selected_round, selected_mean, group_cursor,
    )
    prefix_online = concatenate_lists(online_lists)
    prefix_batches = concatenate_lists(actor_batch_lists)
    prefix_dir = output / "shared_prefix" / f"seed_{seed}"
    prefix_dir.mkdir(parents=True)
    prefix_arrays_path = prefix_dir / "prefix_arrays.npz"
    np.savez_compressed(
        prefix_arrays_path,
        **prefix_online,
        **prefix_batches,
        fit_indices=fit,
        selection_indices=selection,
        oof_indices=oof,
        chosen_schedule=chosen_schedule,
        actor_schedule=actor_schedule,
        probe_radius_by_round=radii,
        selection_round_action=np.stack(selection_actions),
        selection_round_cost=np.stack(selection_costs),
        round0_oof_action=initial_oof_action,
        round0_oof_cost=initial_oof_cost,
    )
    fork_path = prefix_dir / "fork_state.pt"
    torch.save({
        "qualification": "QUERY_OAC_SHARED_PREFIX_FORK_STATE_TRAIN_ONLY",
        "seed": seed,
        "fork_round": prefix_rounds,
        "actor_state_dict": cpu_state(fork["actor_state"]),
        "critic1_state_dict": cpu_state(fork["critic_states"][0]),
        "critic2_state_dict": cpu_state(fork["critic_states"][1]),
        "actor_optimizer_state_dict": fork["actor_optimizer_state"],
        "critic1_optimizer_state_dict": fork["critic_optimizer_states"][0],
        "critic2_optimizer_state_dict": fork["critic_optimizer_states"][1],
        "critic_rng_state": fork["critic_rng_state"],
        "torch_cpu_rng_state": fork["torch_cpu_rng_state"],
        "torch_cuda_rng_states": fork["torch_cuda_rng_states"],
        "selected_actor_state_dict": cpu_state(fork["selected_state"]),
        "selected_critic1_state_dict": cpu_state(fork["selected_critic_states"][0]),
        "selected_critic2_state_dict": cpu_state(fork["selected_critic_states"][1]),
        "selected_round": fork["selected_round"],
        "selected_mean": fork["selected_mean"],
        "group_cursor": fork["group_cursor"],
        "prefix_arrays": str(prefix_arrays_path),
        "prefix_arrays_sha256": base.sha256(prefix_arrays_path),
        "source_pretrain_checkpoint": str(checkpoint_source),
        "source_pretrain_checkpoint_sha256": base.sha256(checkpoint_source),
        "source_replay_sha256": replay_manifest["replay_sha256"],
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }, fork_path)

    results = []
    for arm in config["arms"]:
        restored = restore_training_state(fork, payload, config, device)
        actor, critics, trainings, actor_optimizer, critic_optimizers, critic_rng = restored
        online_lists = copy.deepcopy(online_lists if False else {
            name: [np.asarray(value).copy()] for name, value in prefix_online.items()
        })
        actor_batch_lists = {
            name: [np.asarray(value).copy()] for name, value in prefix_batches.items()
        }
        branch_selection_actions = [np.asarray(value).copy() for value in selection_actions]
        branch_selection_costs = [np.asarray(value).copy() for value in selection_costs]
        branch_records = copy.deepcopy(prefix_records)
        branch_selected_state = copy.deepcopy(fork["selected_state"])
        branch_selected_critic_states = copy.deepcopy(fork["selected_critic_states"])
        branch_selected_round = int(fork["selected_round"])
        branch_selected_mean = float(fork["selected_mean"])
        branch_group_cursor = int(fork["group_cursor"])
        for round_index in range(prefix_rounds + 1, rounds + 1):
            result = execute_round(
                round_index=round_index, arm=arm, config=config, data=data,
                controller=controller, weights=weights, bases=bases, radii=radii,
                chosen_schedule=chosen_schedule, actor_schedule=actor_schedule,
                fit=fit, selection=selection, inputs=inputs, sigma=sigma, device=device,
                actor=actor, critics=critics, trainings=trainings,
                actor_optimizer=actor_optimizer, critic_optimizers=critic_optimizers,
                critic_rng=critic_rng, online_lists=online_lists,
                actor_batch_lists=actor_batch_lists,
                selection_actions=branch_selection_actions,
                selection_costs=branch_selection_costs,
                selected_state=branch_selected_state,
                selected_critic_states=branch_selected_critic_states,
                selected_round=branch_selected_round, selected_mean=branch_selected_mean,
                group_cursor=branch_group_cursor,
            )
            record, branch_selected_state, branch_selected_critic_states, branch_selected_round, branch_selected_mean, branch_group_cursor = result
            branch_records.append(record)
        online = concatenate_lists(online_lists)
        actor_batches = concatenate_lists(actor_batch_lists)
        latest_state = copy.deepcopy(actor.state_dict())
        latest_critic_states = [copy.deepcopy(model.state_dict()) for model in critics]
        latest_selection_action = branch_selection_actions[-1]
        latest_selection_cost = branch_selection_costs[-1]
        latest_oof_action = actor_predict(actor, inputs, oof, device)
        latest_oof_cost = direct_cost(controller, data, oof, latest_oof_action, weights)
        actor.load_state_dict(branch_selected_state, strict=True)
        selected_selection_action = actor_predict(actor, inputs, selection, device)
        selected_selection_cost = direct_cost(controller, data, selection, selected_selection_action, weights)
        selected_oof_action = actor_predict(actor, inputs, oof, device)
        selected_oof_cost = direct_cost(controller, data, oof, selected_oof_action, weights)
        initial_eval = evaluation_block(
            initial_selection_cost, initial_oof_cost, initial_selection_cost, initial_oof_cost,
            data, selection, oof,
        )
        latest_eval = evaluation_block(
            latest_selection_cost, latest_oof_cost, initial_selection_cost, initial_oof_cost,
            data, selection, oof,
        )
        selected_eval = evaluation_block(
            selected_selection_cost, selected_oof_cost, initial_selection_cost, initial_oof_cost,
            data, selection, oof,
        )
        run_dir = output / arm["name"] / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        arrays_path = run_dir / "pilot_arrays.npz"
        np.savez_compressed(
            arrays_path,
            **online,
            **actor_batches,
            fit_indices=fit,
            selection_indices=selection,
            oof_indices=oof,
            probe_radius_by_round=radii,
            selection_round_action=np.stack(branch_selection_actions),
            selection_round_cost=np.stack(branch_selection_costs),
            round0_oof_action=initial_oof_action,
            round0_oof_cost=initial_oof_cost,
            latest_selection_action=latest_selection_action,
            latest_selection_cost=latest_selection_cost,
            latest_oof_action=latest_oof_action,
            latest_oof_cost=latest_oof_cost,
            selected_selection_action=selected_selection_action,
            selected_selection_cost=selected_selection_cost,
            selected_oof_action=selected_oof_action,
            selected_oof_cost=selected_oof_cost,
        )
        checkpoint_path = run_dir / "checkpoint.pt"
        torch.save({
            "qualification": "QUERY_OAC_GAMMA1_K16_LR_DECAY_TRAIN_ONLY",
            "arm": arm,
            "seed": seed,
            "selected_round": branch_selected_round,
            "actor_update_count": rounds * k,
            "critic_update_count_per_twin": rounds * int(config["critic_updates"]["updates_per_round_per_twin"]),
            "actor_training": payload["actor_training"],
            "normalization": payload["normalization"],
            "critic1_training": trainings[0],
            "critic2_training": trainings[1],
            "selected_actor_state_dict": cpu_state(branch_selected_state),
            "latest_actor_state_dict": cpu_state(latest_state),
            "selected_critic1_state_dict": cpu_state(branch_selected_critic_states[0]),
            "selected_critic2_state_dict": cpu_state(branch_selected_critic_states[1]),
            "latest_critic1_state_dict": cpu_state(latest_critic_states[0]),
            "latest_critic2_state_dict": cpu_state(latest_critic_states[1]),
            "fork_state": str(fork_path),
            "fork_state_sha256": base.sha256(fork_path),
            "prefix_arrays": str(prefix_arrays_path),
            "prefix_arrays_sha256": base.sha256(prefix_arrays_path),
            "fit_indices": fit,
            "selection_indices": selection,
            "oof_indices": oof,
            "source_pretrain_checkpoint": str(checkpoint_source),
            "source_pretrain_checkpoint_sha256": base.sha256(checkpoint_source),
            "source_replay_sha256": replay_manifest["replay_sha256"],
            "formal_validation_or_test_consumed": False,
            "dbm_fields_or_labels_consumed": [],
            "query_analytic_gradient_consumed": False,
        }, checkpoint_path)
        results.append({
            "arm": arm["name"],
            "actor_updates_per_round": k,
            "seed": seed,
            "selected_round": branch_selected_round,
            "initial": initial_eval,
            "latest": latest_eval,
            "selected": selected_eval,
            "rounds": branch_records,
            "online_replay_rows": int(len(online["cost"])),
            "actor_batch_rows": int(len(actor_batches["actor_batch_state_index"])),
            "logical_query_rollouts": int(len(online["cost"]) + (rounds + 5) * len(selection)),
            "arrays": str(arrays_path),
            "arrays_sha256": base.sha256(arrays_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": base.sha256(checkpoint_path),
            "fork_state": str(fork_path),
            "fork_state_sha256": base.sha256(fork_path),
            "prefix_arrays": str(prefix_arrays_path),
            "prefix_arrays_sha256": base.sha256(prefix_arrays_path),
        })
    return results


def schedule_checks(summary_pooled: dict, records: list[dict], config: dict) -> tuple[dict, str]:
    candidate = str(config["decision_gate"]["candidate"])
    controls = list(config["decision_gate"]["controls"])
    gate = config["decision_gate"]
    primary = summary_pooled[candidate]["selected"]["inner"]
    control_metrics = [summary_pooled[name]["selected"]["inner"] for name in controls]
    best_mean = min(item["actor_cost"]["mean"] for item in control_metrics)
    best_aggregate = max(item["aggregate_improvement"] for item in control_metrics)
    best_median = max(item["gain"]["median"] for item in control_metrics)
    best_p05 = max(item["gain"]["p05"] for item in control_metrics)
    best_worst = max(item["gain"]["min"] for item in control_metrics)
    by_key = {(item["arm"], int(item["seed"])): item for item in records}
    seed_count = sum(
        by_key[(candidate, seed)]["selected"]["inner_warm_relative"]["actor_cost"]["mean"]
        <= min(by_key[(control, seed)]["selected"]["inner_warm_relative"]["actor_cost"]["mean"] for control in controls)
        + float(gate["pooled_mean_cost_not_worse_than_best_control_tolerance"])
        for seed in config["pilot"]["seeds"]
    )
    latest = summary_pooled[candidate]["latest"]["inner"]
    checks = {
        "seed_mean_cost_advantage_count": seed_count >= int(gate["actor_mean_cost_advantage_seed_count_minimum"]),
        "selected_mean_cost": primary["actor_cost"]["mean"] <= best_mean + float(gate["pooled_mean_cost_not_worse_than_best_control_tolerance"]),
        "selected_aggregate": primary["aggregate_improvement"] >= best_aggregate - float(gate["pooled_aggregate_improvement_not_worse_than_best_control_tolerance"]),
        "selected_median": primary["gain"]["median"] >= best_median - float(gate["pooled_median_not_worse_than_best_control_tolerance"]),
        "selected_p05": primary["gain"]["p05"] >= best_p05 - float(gate["pooled_p05_not_worse_than_best_control_tolerance"]),
        "selected_worst": primary["gain"]["min"] >= best_worst - float(gate["pooled_worst_not_worse_than_best_control_tolerance"]),
        "latest_close_to_selected": latest["actor_cost"]["mean"] <= primary["actor_cost"]["mean"] + float(gate["latest_to_selected_mean_cost_gap_maximum"]),
    }
    fixed = summary_pooled["fixed_lr1e5"]["selected"]["inner"]
    for slice_name in gate["hard_slices"]:
        local = primary["by_speed_variant"][slice_name]
        reference = fixed["by_speed_variant"][slice_name]
        checks[f"hard_{slice_name}_aggregate"] = local["aggregate_improvement"] >= reference["aggregate_improvement"] - float(gate["hard_slice_aggregate_not_worse_than_fixed_lr1e5_tolerance"])
        checks[f"hard_{slice_name}_p05"] = local["gain"]["p05"] >= reference["gain"]["p05"] - float(gate["hard_slice_p05_not_worse_than_fixed_lr1e5_tolerance"])
        checks[f"hard_{slice_name}_worst"] = local["gain"]["min"] >= reference["gain"]["min"] - float(gate["hard_slice_worst_not_worse_than_fixed_lr1e5_tolerance"])
    decision = "PROMOTE_HIGH_INITIAL_LR_COSINE_DECAY" if all(checks.values()) else "DO_NOT_PROMOTE_LR_DECAY_SCHEDULE"
    return {"seed_mean_cost_advantage_count": int(seed_count), "checks": checks, "passes": bool(all(checks.values()))}, decision


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    if config["formal_validation_or_test_consumed"] or config["dbm_fields_or_labels_consumed"]:
        raise AssertionError("sealed boundary violation")
    if int(config["common_prefix"]["rounds"]) != 60 or int(config["pilot"]["rounds"]) != 160:
        raise AssertionError("expected shared round-60 fork and round-160 endpoint")
    if [arm["name"] for arm in config["arms"]] != ["fixed_lr1e5", "cosine_lr1e5_to_2e6", "switch_lr2e6"]:
        raise AssertionError("unexpected LR schedule arms")
    deterministic = deterministic_contract(config)
    lr_scan_dir = Path(config["sources"]["lr_scan_60round"]).resolve()
    lr_scan_validation = json.loads((lr_scan_dir / "validation.json").read_text())
    if lr_scan_validation["qualification"] != "QUERY_OAC_GAMMA1_K16_LR_SCAN_60ROUND_INDEPENDENT_PASS":
        raise AssertionError("source LR scan did not independently pass")
    prior_dir = Path(config["sources"]["k16_160round"]).resolve()
    prior_validation = json.loads((prior_dir / "validation.json").read_text())
    if prior_validation["qualification"] != "QUERY_OAC_GAMMA1_K16_160ROUND_INDEPENDENT_PASS":
        raise AssertionError("source K16/160 did not independently pass")
    replay_dir = Path(config["sources"]["absolute_replay"]).resolve()
    pretrain_dir = Path(config["sources"]["pretrain"]).resolve()
    replay_manifest = json.loads((replay_dir / "manifest.json").read_text())
    with np.load(replay_dir / "replay.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    fullrank_manifest = json.loads((Path(replay_manifest["fullrank_source"]) / "manifest.json").read_text())
    parent_manifest = json.loads((Path(fullrank_manifest["parent_t0"]) / "manifest.json").read_text())
    collection_manifest = json.loads((Path(parent_manifest["source_collection"]) / "manifest.json").read_text())
    device = torch.device(args.device)
    query = base.QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = base.TorchMPPIController(
        base.TorchQueryRolloutBackend(query),
        base.TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    fit = np.flatnonzero(np.isin(data["fold_id"], config["split_contract"]["fit_folds"]))
    selection = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    oof = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
    anneal = np.linspace(
        float(config["pilot"]["probe_radius_sigma_start"]),
        float(config["pilot"]["probe_radius_sigma_end"]),
        int(config["pilot"]["probe_radius_sigma_anneal_rounds"]),
    ).astype(np.float32)
    radii = np.concatenate((anneal, np.full(int(config["pilot"]["rounds"]) - len(anneal), anneal[-1], np.float32)))
    output.mkdir(parents=True)
    records = []
    bases = base.basis_bank()
    for seed in config["pilot"]["seeds"]:
        records.extend(run_seed(
            config, int(seed), data, replay_manifest, pretrain_dir,
            controller, weights, bases, radii, fit, selection, oof, output, device,
        ))
    by_arm = {
        arm["name"]: [record for record in records if record["arm"] == arm["name"]]
        for arm in config["arms"]
    }
    pooled = {
        arm: {
            stage: {
                "inner": base.pooled_warm(local, stage, "inner", data),
                "development_oof": base.pooled_warm(local, stage, "oof", data),
            }
            for stage in ("round0", "latest", "selected")
        }
        for arm, local in by_arm.items()
    }
    decision_checks, decision = schedule_checks(pooled, records, config)
    prefix_online_per_seed = int(config["common_prefix"]["rounds"]) * int(config["pilot"]["fit_contexts_visited_per_round"]) * int(config["pilot"]["candidates_per_visit"])
    continuation_online_per_arm_seed = (int(config["pilot"]["rounds"]) - int(config["common_prefix"]["rounds"])) * int(config["pilot"]["fit_contexts_visited_per_round"]) * int(config["pilot"]["candidates_per_visit"])
    actual_online = len(config["pilot"]["seeds"]) * (prefix_online_per_seed + len(config["arms"]) * continuation_online_per_arm_seed)
    summary = {
        "qualification": "QUERY_OAC_GAMMA1_K16_LR_DECAY_PENDING_INDEPENDENT_VALIDATION",
        "decision": decision,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": config,
        "deterministic_runtime_contract": deterministic,
        "records": records,
        "pooled_warm_relative": pooled,
        "decision_checks": decision_checks,
        "decision_population": "inner selected checkpoints only",
        "development_oof_role": "corroborating already-consumed evidence; not LR schedule selection or untouched validation",
        "actual_unique_online_query_rollouts": int(actual_online),
        "logical_online_query_rollouts_if_prefix_duplicated": int(sum(item["online_replay_rows"] for item in records)),
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    summary_path = output / "summary.json"
    base.dump_json(summary_path, summary)
    manifest = {
        "schema_version": "query-oac-gamma1-k16-shared-prefix-lr-decay-v1",
        "qualification": summary["qualification"],
        "decision": decision,
        "config": str(config_path),
        "config_sha256": base.sha256(config_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": base.sha256(Path(__file__).resolve()),
        "shared_runner": str(Path(base.__file__).resolve()),
        "shared_runner_sha256": base.sha256(Path(base.__file__).resolve()),
        "absolute_replay": str(replay_dir),
        "absolute_replay_sha256": replay_manifest["replay_sha256"],
        "pretrain": str(pretrain_dir),
        "pretrain_manifest_sha256": base.sha256(pretrain_dir / "manifest.json"),
        "lr_scan_60round": str(lr_scan_dir),
        "lr_scan_validation_sha256": base.sha256(lr_scan_dir / "validation.json"),
        "k16_160round": str(prior_dir),
        "k16_160round_validation_sha256": base.sha256(prior_dir / "validation.json"),
        "query_checkpoint": replay_manifest["query_checkpoint"],
        "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
        "summary_sha256": base.sha256(summary_path),
        "run_artifacts": {
            f"{record['arm']}_seed{record['seed']}": {
                "arrays_sha256": record["arrays_sha256"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "fork_state_sha256": record["fork_state_sha256"],
                "prefix_arrays_sha256": record["prefix_arrays_sha256"],
            }
            for record in records
        },
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    base.dump_json(output / "manifest.json", manifest)
    print(json.dumps({
        "decision": decision,
        "decision_checks": decision_checks,
        "selected_inner": {arm: pooled[arm]["selected"]["inner"] for arm in pooled},
        "latest_inner": {arm: pooled[arm]["latest"]["inner"] for arm in pooled},
    }, indent=2))


if __name__ == "__main__":
    main()
