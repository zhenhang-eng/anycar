#!/usr/bin/env python3
"""Run train-side Query OAC with an established Actor and target-coverage Critics."""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

import pretrain_query_single_center_actor_twin_critic as pretrain
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic
from run_query_forward_response_landscape_pilot import basis_bank
from run_query_oac_gamma1_k_scan import warm_relative_metrics
from run_query_single_center_oac20to1 import (
    StratifiedQueues,
    actor_from_payload,
    actor_output_step_rms,
    actor_predict,
    actor_tensor,
    candidate_ranking_metrics,
    direct_cost,
    distribution,
    interpolate_state,
    load_inputs,
    physical_critic_value,
    response_bank,
    set_seed,
    sha256,
    update_critics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_target_coverage_mixed_init_oac_config_20260903_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def mixed_actor_microsteps(
    actor: torch.nn.Module,
    selected_actor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    critics: list[ConfigurableAbsoluteActionValueCritic],
    trainings: list[dict],
    actor_inputs: tuple[np.ndarray, ...],
    critic_inputs: tuple[np.ndarray, ...],
    fit: np.ndarray,
    batch_schedule: np.ndarray,
    config: dict,
    sigma: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """Take Actor steps while respecting each source model's own input normalization."""
    spec = config["actor_updates"]
    gamma = float(config["actor_cost_weight_gamma"])
    maximum = float(config["actor_cost_weight_maximum"])
    before_state = copy.deepcopy(actor.state_dict())
    before_output = actor_predict(actor, actor_inputs, fit, device)
    microsteps = []
    for critic in critics:
        critic.eval()
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
    for microstep, rows in enumerate(batch_schedule, start=1):
        actor.train()
        action = actor_tensor(actor, actor_inputs, rows, device)
        with torch.no_grad():
            selected = actor_tensor(selected_actor, actor_inputs, rows, device)
        values = [
            physical_critic_value(critic, training, critic_inputs, rows, action, device)
            for critic, training in zip(critics, trainings)
        ]
        conservative = torch.maximum(values[0], values[1])
        unnormalized = torch.exp(gamma * conservative.detach())
        weight = unnormalized.clamp(max=maximum)
        weight = weight / weight.mean().clamp_min(1e-12)
        sigma_tensor = torch.from_numpy(sigma).to(device).reshape(1, 1, 2)
        trust = torch.mean(torch.square((action - selected) / sigma_tensor))
        loss = torch.mean(weight * conservative) + float(
            spec["selected_checkpoint_output_trust_weight"]
        ) * trust
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            actor.parameters(), float(spec["gradient_clip_norm"])
        ))
        optimizer.step()
        effective_fraction = float(
            torch.square(weight.sum()).div(len(weight) * torch.square(weight).sum()).cpu()
        )
        microsteps.append({
            "microstep": microstep,
            "loss": float(loss.detach().cpu()),
            "conservative_log_cost": float(conservative.mean().detach().cpu()),
            "selected_output_trust": float(trust.detach().cpu()),
            "gradient_norm_before_clip": gradient_norm,
            "cost_weight_effective_sample_fraction": effective_fraction,
            "cost_weight_maximum": float(weight.max().detach().cpu()),
            "cost_weight_minimum": float(weight.min().detach().cpu()),
            "cost_weight_clip_fraction": float(
                torch.mean((unnormalized >= maximum).float()).cpu()
            ),
        })
    for critic in critics:
        for parameter in critic.parameters():
            parameter.requires_grad_(True)
    end_state = copy.deepcopy(actor.state_dict())
    raw_rms = actor_output_step_rms(actor, before_output, actor_inputs, fit, sigma, device)
    cap = float(spec["cumulative_per_round_output_step_cap_sigma_rms"])
    projection = 1.0
    if raw_rms > cap:
        lower, upper = 0.0, 1.0
        for _ in range(16):
            midpoint = 0.5 * (lower + upper)
            actor.load_state_dict(interpolate_state(before_state, end_state, midpoint), strict=True)
            current = actor_output_step_rms(
                actor, before_output, actor_inputs, fit, sigma, device
            )
            if current <= cap:
                lower = midpoint
            else:
                upper = midpoint
        projection = lower
        actor.load_state_dict(interpolate_state(before_state, end_state, projection), strict=True)
    final_rms = actor_output_step_rms(actor, before_output, actor_inputs, fit, sigma, device)
    return {
        "microstep_count": int(len(batch_schedule)),
        "loss_mean": float(np.mean([item["loss"] for item in microsteps])),
        "cost_weight_effective_sample_fraction_mean": float(np.mean([
            item["cost_weight_effective_sample_fraction"] for item in microsteps
        ])),
        "cost_weight_effective_sample_fraction_minimum": float(np.min([
            item["cost_weight_effective_sample_fraction"] for item in microsteps
        ])),
        "raw_cumulative_output_step_sigma_rms": raw_rms,
        "final_cumulative_output_step_sigma_rms": final_rms,
        "cumulative_trust_projection": projection,
        "cost_weight_gamma": gamma,
        "microsteps": microsteps,
    }


def evaluate_actor(
    actor: torch.nn.Module,
    actor_inputs: tuple[np.ndarray, ...],
    rows: np.ndarray,
    controller: TorchMPPIController,
    data: dict[str, np.ndarray],
    weights: dict[str, float],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    action = actor_predict(actor, actor_inputs, rows, device)
    cost = direct_cost(controller, data, rows, action, weights)
    report = warm_relative_metrics(
        cost, data["warm_cost"][rows], data["speed_kph"][rows],
        data["variant_index"][rows],
    )
    return action, cost, report


def run_seed(
    seed: int,
    config: dict,
    data: dict[str, np.ndarray],
    replay_manifest: dict,
    controller: TorchMPPIController,
    fit: np.ndarray,
    selection: np.ndarray,
    outer: np.ndarray,
    radii: np.ndarray,
    bases: np.ndarray,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    base_seed = 2_609_300 + seed
    set_seed(base_seed)
    context_rng = np.random.default_rng(base_seed + 11)
    critic_rng = np.random.default_rng(base_seed + 29)
    actor_rng = np.random.default_rng(base_seed + 47)
    rounds = int(config["pilot"]["rounds"])
    microstep_count = int(config["actor_updates"]["updates_per_round"])
    actor_schedule = np.stack([
        np.stack([
            actor_rng.choice(
                fit, int(config["actor_updates"]["batch_size"]), replace=False
            ).astype(np.int64)
            for _ in range(microstep_count)
        ])
        for _ in range(rounds)
    ])

    actor_source = Path(config["sources"]["actor_oac"]) / f"seed_{seed}" / "checkpoint.pt"
    critic_source = Path(config["sources"]["critic_pretrain"]) / "checkpoints" / f"seed_{seed}.pt"
    actor_payload = torch.load(actor_source, map_location=device, weights_only=False)
    critic_payload = torch.load(critic_source, map_location=device, weights_only=False)
    actor = actor_from_payload(actor_payload, "selected_actor_state_dict", device)
    selected_actor = copy.deepcopy(actor)
    actor_inputs = load_inputs(data, actor_payload["normalization"])
    critic_inputs = load_inputs(data, critic_payload["normalization"])
    critics, trainings, critic_optimizers = [], [], []
    for twin in (1, 2):
        critic = ConfigurableAbsoluteActionValueCritic().to(device)
        critic.load_state_dict(critic_payload[f"critic{twin}_state_dict"], strict=True)
        critics.append(critic)
        trainings.append(critic_payload[f"critic{twin}_training"])
        critic_optimizers.append(torch.optim.AdamW(
            critic.parameters(), lr=float(config["critic_updates"]["learning_rate"]),
            weight_decay=float(config["critic_updates"]["weight_decay"]),
        ))
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=float(config["actor_updates"]["learning_rate_per_microstep"]),
        weight_decay=float(config["actor_updates"]["weight_decay"]),
    )
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    sigma = np.asarray(config["pilot"]["noise_sigma"], np.float32).reshape(1, 2)
    queues = StratifiedQueues(data, fit, context_rng)

    round0_selection_action, round0_selection_cost, round0_selection = evaluate_actor(
        actor, actor_inputs, selection, controller, data, weights, device
    )
    round0_fit_action, round0_fit_cost, round0_fit = evaluate_actor(
        actor, actor_inputs, fit, controller, data, weights, device
    )
    selected_round = 0
    selected_mean = float(round0_selection_cost.mean())
    selected_state = copy.deepcopy(actor.state_dict())
    selected_critic_states = [copy.deepcopy(model.state_dict()) for model in critics]
    selection_actions = [round0_selection_action]
    selection_costs = [round0_selection_cost]
    online_lists: dict[str, list[np.ndarray]] = {
        "state_index": [], "action": [], "cost": [], "raw_action": [],
        "clipped": [], "round": [], "group": [], "role": [],
    }
    group_cursor = 0
    round_records = []
    for round_index in range(1, rounds + 1):
        chosen = queues.take()
        if len(chosen) != int(config["pilot"]["fit_contexts_visited_per_round"]):
            raise AssertionError("stratified context count changed")
        centers = actor_predict(actor, actor_inputs, chosen, device)
        local_costs = []
        for position, row in enumerate(chosen):
            actions, costs, raw, clipped = response_bank(
                controller, data, int(row), centers[position],
                float(radii[round_index - 1]),
                bases[(round_index - 1) % len(bases)], sigma, weights, config,
            )
            local_costs.append(costs)
            online_lists["state_index"].append(np.full(len(actions), row, np.int64))
            online_lists["action"].append(actions)
            online_lists["cost"].append(costs)
            online_lists["raw_action"].append(raw)
            online_lists["clipped"].append(clipped)
            online_lists["round"].append(np.full(len(actions), round_index, np.int16))
            online_lists["group"].append(np.full(len(actions), group_cursor, np.int32))
            online_lists["role"].append(np.asarray(
                ["actor"] + ["probe"] * 32 + ["response"] * 6
            ))
            group_cursor += 1
        online = {name: np.concatenate(values) for name, values in online_lists.items()}
        critic_history = [
            update_critics(
                critics, critic_optimizers, trainings, critic_inputs, data, fit,
                online, config, critic_rng, device,
            )
            for _ in range(int(config["critic_updates"]["updates_per_round_per_twin"]))
        ]
        selected_actor.load_state_dict(selected_state, strict=True)
        actor_info = mixed_actor_microsteps(
            actor, selected_actor, actor_optimizer, critics, trainings,
            actor_inputs, critic_inputs, fit, actor_schedule[round_index - 1],
            config, sigma, device,
        )
        selection_action, selection_cost, selection_report = evaluate_actor(
            actor, actor_inputs, selection, controller, data, weights, device
        )
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
            critics, trainings, critic_inputs,
            online["state_index"][-recent_count:], online["action"][-recent_count:],
            online["cost"][-recent_count:], online["group"][-recent_count:], device,
        )
        round_records.append({
            "round": round_index,
            "probe_radius_sigma": float(radii[round_index - 1]),
            "visited_rows": chosen.tolist(),
            "new_candidate_cost": distribution(np.concatenate(local_costs)),
            "critic_loss": {
                "mean": float(np.mean([item["loss_mean"] for item in critic_history])),
                "value": float(np.mean([item["value_loss_mean"] for item in critic_history])),
                "ranking": float(np.mean([item["ranking_loss_mean"] for item in critic_history])),
            },
            "critic_recent_bank": ranking,
            "actor_update": actor_info,
            "inner_warm_relative": selection_report,
            "inner_cost_mean": float(selection_cost.mean()),
            "accepted_as_selected": accepted,
            "selected_round_after_update": selected_round,
        })
        if round_index == 1 or round_index % 10 == 0:
            print(
                f"seed={seed} round={round_index}/{rounds} "
                f"inner_mean={selection_cost.mean():.6f} selected={selected_round} "
                f"pair={ranking['center_relative_sign_accuracy']['median']:.4f}",
                flush=True,
            )

    latest_state = copy.deepcopy(actor.state_dict())
    latest_selection_action = selection_actions[-1]
    latest_selection_cost = selection_costs[-1]
    actor.load_state_dict(selected_state, strict=True)
    for critic, state in zip(critics, selected_critic_states):
        critic.load_state_dict(state, strict=True)
    selected_selection_action, selected_selection_cost, selected_selection = evaluate_actor(
        actor, actor_inputs, selection, controller, data, weights, device
    )
    selected_fit_action, selected_fit_cost, selected_fit = evaluate_actor(
        actor, actor_inputs, fit, controller, data, weights, device
    )
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir()
    arrays_path = seed_dir / "oac_arrays.npz"
    online = {name: np.concatenate(values) for name, values in online_lists.items()}
    np.savez_compressed(
        arrays_path,
        fit_indices=fit,
        selection_indices=selection,
        outer_indices_unevaluated=outer,
        actor_batch_schedule=actor_schedule,
        selection_round_action=np.stack(selection_actions),
        selection_round_cost=np.stack(selection_costs),
        round0_fit_action=round0_fit_action,
        round0_fit_cost=round0_fit_cost,
        selected_fit_action=selected_fit_action,
        selected_fit_cost=selected_fit_cost,
        online_state_index=online["state_index"],
        online_action=online["action"],
        online_cost=online["cost"],
        online_raw_action=online["raw_action"],
        online_clipped=online["clipped"],
        online_round=online["round"],
        online_group=online["group"],
        online_role=online["role"],
    )
    checkpoint_path = seed_dir / "checkpoint.pt"
    torch.save({
        "qualification": "QUERY_TARGET_COVERAGE_MIXED_INIT_OAC_TRAIN_SIDE_COMPLETE",
        "seed": seed,
        "selected_round": selected_round,
        "actor_training": actor_payload["actor_training"],
        "actor_normalization": actor_payload["normalization"],
        "critic_normalization": critic_payload["normalization"],
        "critic1_training": trainings[0],
        "critic2_training": trainings[1],
        "selected_actor_state_dict": {
            name: value.detach().cpu() for name, value in selected_state.items()
        },
        "latest_actor_state_dict": {
            name: value.detach().cpu() for name, value in latest_state.items()
        },
        "selected_critic1_state_dict": {
            name: value.detach().cpu() for name, value in selected_critic_states[0].items()
        },
        "selected_critic2_state_dict": {
            name: value.detach().cpu() for name, value in selected_critic_states[1].items()
        },
        "fit_indices": fit,
        "selection_indices": selection,
        "outer_indices_unevaluated": outer,
        "source_actor_checkpoint": str(actor_source.resolve()),
        "source_actor_checkpoint_sha256": sha256(actor_source),
        "source_critic_checkpoint": str(critic_source.resolve()),
        "source_critic_checkpoint_sha256": sha256(critic_source),
        "source_replay_sha256": replay_manifest["replay_sha256"],
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }, checkpoint_path)
    return {
        "seed": seed,
        "selected_round": selected_round,
        "round0": {"inner": round0_selection, "fit": round0_fit},
        "latest": {
            "inner": warm_relative_metrics(
                latest_selection_cost, data["warm_cost"][selection],
                data["speed_kph"][selection], data["variant_index"][selection],
            )
        },
        "selected": {"inner": selected_selection, "fit": selected_fit},
        "rounds": round_records,
        "online_query_rollouts": int(len(online["cost"])),
        "inner_checkpoint_query_rollouts": int((rounds + 1) * len(selection)),
        "fit_report_query_rollouts": int(2 * len(fit)),
        "arrays": str(arrays_path),
        "arrays_sha256": sha256(arrays_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
    }


def pooled(records: list[dict[str, Any]], stage: str, split: str,
           data: dict[str, np.ndarray]) -> dict[str, Any]:
    costs, rows = [], []
    for record in records:
        with np.load(record["arrays"], allow_pickle=False) as archive:
            local_rows = archive["selection_indices"] if split == "inner" else archive["fit_indices"]
            if split == "inner":
                all_cost = archive["selection_round_cost"]
                local_cost = all_cost[0] if stage == "round0" else all_cost[int(record["selected_round"])]
            else:
                local_cost = archive[f"{stage}_fit_cost"]
            costs.append(np.asarray(local_cost))
            rows.append(np.asarray(local_rows))
    joined_rows = np.concatenate(rows)
    return warm_relative_metrics(
        np.concatenate(costs), data["warm_cost"][joined_rows],
        data["speed_kph"][joined_rows], data["variant_index"][joined_rows],
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    if (config["formal_validation_or_test_consumed"]
            or config["dbm_fields_or_labels_consumed"]
            or config["query_analytic_gradient_consumed"]):
        raise AssertionError("sealed-boundary contract violated")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    loader_config = {"outputs": {"absolute_replay": config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader_config)
    replay_validation = json.loads(
        (Path(config["sources"]["absolute_replay"]) / "validation.json").read_text()
    )
    if replay_validation["qualification"] != "QUERY_TARGET_COVERAGE_ABSOLUTE_REPLAY_INDEPENDENT_PASS":
        raise AssertionError("target-coverage Replay did not independently pass")
    critic_validation = json.loads(
        (Path(config["sources"]["critic_pretrain"]) / "validation.json").read_text()
    )
    if not critic_validation["checks"]["checkpoint_reload_exact"]:
        raise AssertionError("target-coverage Critic checkpoints did not reproduce")
    if not critic_validation["checks"]["global_value_and_landscape_signal"]:
        raise AssertionError("target-coverage Critic global signal failed")
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    fit = np.flatnonzero(np.isin(data["fold_id"], config["split_contract"]["fit_folds"]))
    selection = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    outer = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
    if (len(fit), len(selection), len(outer)) != (432, 120, 120):
        raise AssertionError(f"unexpected split sizes {(len(fit), len(selection), len(outer))}")
    episode_sets = [set(data["episode_id"][rows].tolist()) for rows in (fit, selection, outer)]
    if episode_sets[0] & episode_sets[1] or episode_sets[0] & episode_sets[2] or episode_sets[1] & episode_sets[2]:
        raise AssertionError("episode leakage")
    rounds = int(config["pilot"]["rounds"])
    anneal = int(config["pilot"]["probe_radius_sigma_anneal_rounds"])
    radii = np.concatenate((
        np.linspace(
            float(config["pilot"]["probe_radius_sigma_start"]),
            float(config["pilot"]["probe_radius_sigma_end"]), anneal,
        ),
        np.full(rounds - anneal, float(config["pilot"]["probe_radius_sigma_end"])),
    )).astype(np.float32)
    output.mkdir(parents=True)
    records = [
        run_seed(
            int(seed), config, data, replay_manifest, controller, fit, selection,
            outer, radii, basis_bank(), output, device,
        )
        for seed in config["pilot"]["seeds"]
    ]
    pooled_metrics = {
        "round0_inner": pooled(records, "round0", "inner", data),
        "selected_inner": pooled(records, "selected", "inner", data),
        "round0_fit": pooled(records, "round0", "fit", data),
        "selected_fit": pooled(records, "selected", "fit", data),
    }
    selected_means = [record["selected"]["inner"]["actor_cost"]["mean"] for record in records]
    round0_means = [record["round0"]["inner"]["actor_cost"]["mean"] for record in records]
    decision_checks = {
        "selected_inner_mean_not_worse_each_seed": all(
            selected <= initial + 1e-9 for selected, initial in zip(selected_means, round0_means)
        ),
        "selected_after_round0_seed_count_at_least_two": sum(
            int(record["selected_round"] > 0) for record in records
        ) >= 2,
        "pooled_inner_mean_improved": (
            pooled_metrics["selected_inner"]["actor_cost"]["mean"]
            < pooled_metrics["round0_inner"]["actor_cost"]["mean"]
        ),
    }
    decision = (
        "MIXED_INIT_OAC_HAS_TRAIN_SIDE_PROGRESS"
        if all(decision_checks.values())
        else "MIXED_INIT_OAC_NO_RELIABLE_TRAIN_SIDE_PROGRESS"
    )
    summary = {
        "qualification": "QUERY_TARGET_COVERAGE_MIXED_INIT_OAC_TRAIN_SIDE_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": config,
        "records": records,
        "pooled": pooled_metrics,
        "decision": decision,
        "decision_checks": decision_checks,
        "deterministic_runtime_contract": {
            "mode": "warn_only",
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        },
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "query-target-coverage-mixed-init-oac-v1",
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "source_replay": config["sources"]["absolute_replay"],
        "source_replay_sha256": replay_manifest["replay_sha256"],
        "source_critic_pretrain": config["sources"]["critic_pretrain"],
        "source_actor_oac": config["sources"]["actor_oac"],
        "summary_sha256": sha256(output / "summary.json"),
        "arrays_sha256": {Path(r["arrays"]).name + f"_seed{r['seed']}": r["arrays_sha256"] for r in records},
        "checkpoint_sha256": {Path(r["checkpoint"]).name + f"_seed{r['seed']}": r["checkpoint_sha256"] for r in records},
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({
        "output": str(output), "decision": decision,
        "selected_rounds": [record["selected_round"] for record in records],
        "round0_inner_mean": pooled_metrics["round0_inner"]["actor_cost"]["mean"],
        "selected_inner_mean": pooled_metrics["selected_inner"]["actor_cost"]["mean"],
        "outer_fold_evaluated": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
