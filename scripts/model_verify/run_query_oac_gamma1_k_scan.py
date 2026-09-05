#!/usr/bin/env python3
"""Run a paired gamma-1 Query OAC Actor-microstep K=1/4/8 scan."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package))

from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic  # noqa: E402
from run_query_forward_response_landscape_pilot import basis_bank  # noqa: E402
from run_query_single_center_oac20to1 import (  # noqa: E402
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
    metrics,
    physical_critic_value,
    response_bank,
    set_seed,
    update_critics,
)


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_oac_gamma1_k_scan_config_20260902_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n")


def warm_relative_metrics(
    cost: np.ndarray,
    warm: np.ndarray,
    speed: np.ndarray,
    variant: np.ndarray,
) -> dict[str, Any]:
    cost64 = np.asarray(cost, np.float64)
    warm64 = np.asarray(warm, np.float64)
    gain = warm64 - cost64

    def local(mask: np.ndarray) -> dict[str, Any]:
        local_gain = gain[mask]
        local_warm = warm64[mask]
        local_cost = cost64[mask]
        return {
            "count": int(mask.sum()),
            "warm_cost": distribution(local_warm),
            "actor_cost": distribution(local_cost),
            "gain": distribution(local_gain),
            "win_or_tie_fraction": float(np.mean(local_cost <= local_warm)),
            "aggregate_improvement": float(local_gain.sum() / local_warm.sum()),
        }

    all_mask = np.ones(len(gain), dtype=bool)
    output = local(all_mask)
    output["by_speed_kph"] = {
        str(int(value)): local(speed == value)
        for value in sorted(np.unique(speed).tolist())
    }
    output["by_variant_index"] = {
        str(int(value)): local(variant == value)
        for value in sorted(np.unique(variant).tolist())
    }
    output["by_speed_variant"] = {
        f"{int(speed_value)}:{int(variant_value)}": local(
            (speed == speed_value) & (variant == variant_value)
        )
        for speed_value in sorted(np.unique(speed).tolist())
        for variant_value in sorted(np.unique(variant).tolist())
        if np.any((speed == speed_value) & (variant == variant_value))
    }
    return output


def gamma1_actor_microsteps(
    actor: torch.nn.Module,
    selected_actor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    critics: list[ConfigurableAbsoluteActionValueCritic],
    trainings: list[dict],
    inputs: tuple[np.ndarray, ...],
    fit: np.ndarray,
    batch_schedule: np.ndarray,
    config: dict,
    sigma: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    spec = config["actor_updates"]
    gamma = float(config["actor_cost_weight_gamma"])
    maximum = float(config["actor_cost_weight_maximum"])
    before_state = copy.deepcopy(actor.state_dict())
    before_output = actor_predict(actor, inputs, fit, device)
    microsteps = []
    for critic in critics:
        critic.eval()
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
    for microstep, rows in enumerate(batch_schedule, start=1):
        actor.train()
        action = actor_tensor(actor, inputs, rows, device)
        with torch.no_grad():
            selected = actor_tensor(selected_actor, inputs, rows, device)
        values = [
            physical_critic_value(critic, training, inputs, rows, action, device)
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
            "cost_weight_clip_fraction": float(torch.mean((unnormalized >= maximum).float()).cpu()),
        })
    for critic in critics:
        for parameter in critic.parameters():
            parameter.requires_grad_(True)
    end_state = copy.deepcopy(actor.state_dict())
    raw_rms = actor_output_step_rms(actor, before_output, inputs, fit, sigma, device)
    cap = float(spec["cumulative_per_round_output_step_cap_sigma_rms"])
    projection = 1.0
    if raw_rms > cap:
        lower, upper = 0.0, 1.0
        for _ in range(16):
            midpoint = 0.5 * (lower + upper)
            actor.load_state_dict(interpolate_state(before_state, end_state, midpoint), strict=True)
            current = actor_output_step_rms(actor, before_output, inputs, fit, sigma, device)
            if current <= cap:
                lower = midpoint
            else:
                upper = midpoint
        projection = lower
        actor.load_state_dict(interpolate_state(before_state, end_state, projection), strict=True)
    final_rms = actor_output_step_rms(actor, before_output, inputs, fit, sigma, device)
    return {
        "microstep_count": int(len(batch_schedule)),
        "microsteps": microsteps,
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
    }


def run_arm_seed(
    config: dict,
    arm: dict,
    seed: int,
    data: dict[str, np.ndarray],
    replay_manifest: dict,
    pretrain_dir: Path,
    controller: TorchMPPIController,
    weights: dict[str, float],
    bases: np.ndarray,
    radii: np.ndarray,
    fit: np.ndarray,
    selection: np.ndarray,
    oof: np.ndarray,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    base_seed = 2_609_120 + seed
    set_seed(base_seed)
    context_rng = np.random.default_rng(base_seed + 11)
    critic_rng = np.random.default_rng(base_seed + 29)
    actor_rng = np.random.default_rng(base_seed + 47)
    max_k = max(int(item["actor_updates_per_round"]) for item in config["arms"])
    actor_schedule = np.stack([
        np.stack([
            actor_rng.choice(fit, int(config["actor_updates"]["batch_size"]), replace=False).astype(np.int64)
            for _ in range(max_k)
        ])
        for _ in range(int(config["pilot"]["rounds"]))
    ])
    k = int(arm["actor_updates_per_round"])
    checkpoint_source = pretrain_dir / "checkpoints" / f"pretrain_fold0_seed{seed}.pt"
    payload = torch.load(checkpoint_source, map_location=device, weights_only=False)
    inputs = load_inputs(data, payload["normalization"])
    actor = actor_from_payload(payload, "actor_state_dict", device)
    selected_actor = copy.deepcopy(actor)
    critics, trainings, critic_optimizers = [], [], []
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
        lr=float(config["actor_updates"]["learning_rate_per_microstep"]),
        weight_decay=float(config["actor_updates"]["weight_decay"]),
    )
    queues = StratifiedQueues(data, fit, context_rng)
    sigma = np.asarray(config["pilot"]["noise_sigma"], np.float32).reshape(1, 2)
    online_lists: dict[str, list[np.ndarray]] = {
        "state_index": [], "action": [], "cost": [], "raw_action": [],
        "clipped": [], "round": [], "group": [], "role": [],
    }
    actor_batch_lists: dict[str, list[np.ndarray]] = {
        "actor_batch_state_index": [], "actor_batch_round": [], "actor_batch_microstep": [],
    }
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
    round_records = []
    for round_index in range(1, int(config["pilot"]["rounds"]) + 1):
        chosen = queues.take()
        centers = actor_predict(actor, inputs, chosen, device)
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
            online_lists["role"].append(np.asarray(["actor"] + ["probe"] * 32 + ["response"] * 6))
            group_cursor += 1
        online = {name: np.concatenate(value) for name, value in online_lists.items()}
        critic_history = [
            update_critics(
                critics, critic_optimizers, trainings, inputs, data, fit,
                online, config, critic_rng, device,
            )
            for _ in range(int(config["critic_updates"]["updates_per_round_per_twin"]))
        ]
        batch_rows = actor_schedule[round_index - 1, :k]
        actor_batch_lists["actor_batch_state_index"].append(batch_rows.reshape(-1))
        actor_batch_lists["actor_batch_round"].append(np.full(batch_rows.size, round_index, np.int16))
        actor_batch_lists["actor_batch_microstep"].append(np.repeat(
            np.arange(1, k + 1, dtype=np.int16), batch_rows.shape[1]
        ))
        selected_actor.load_state_dict(selected_state, strict=True)
        actor_info = gamma1_actor_microsteps(
            actor, selected_actor, actor_optimizer, critics, trainings,
            inputs, fit, batch_rows, config, sigma, device,
        )
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
        round_records.append({
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
            "selection_round0_relative": metrics(
                selection_cost, initial_selection_cost,
                data["warm_cost"][selection], data["speed_kph"][selection],
            ),
            "selection_warm_relative": warm_relative_metrics(
                selection_cost, data["warm_cost"][selection],
                data["speed_kph"][selection], data["variant_index"][selection],
            ),
            "selected": bool(accepted),
            "selected_round_after_evaluation": selected_round,
            "fresh_candidate_critic": ranking,
        })
        print(
            f"{arm['name']} seed={seed} round={round_index}/10 "
            f"Jsel={selection_cost.mean():.5f} best={selected_mean:.5f}@{selected_round} "
            f"warmAgg={round_records[-1]['selection_warm_relative']['aggregate_improvement']:.4f} "
            f"step={actor_info['final_cumulative_output_step_sigma_rms']:.6f}",
            flush=True,
        )
    online = {name: np.concatenate(value) for name, value in online_lists.items()}
    actor_batches = {name: np.concatenate(value) for name, value in actor_batch_lists.items()}
    latest_state = copy.deepcopy(actor.state_dict())
    latest_critic_states = [copy.deepcopy(model.state_dict()) for model in critics]
    latest_selection_action = selection_actions[-1]
    latest_selection_cost = selection_costs[-1]
    latest_oof_action = actor_predict(actor, inputs, oof, device)
    latest_oof_cost = direct_cost(controller, data, oof, latest_oof_action, weights)
    actor.load_state_dict(selected_state, strict=True)
    selected_selection_action = actor_predict(actor, inputs, selection, device)
    selected_selection_cost = direct_cost(controller, data, selection, selected_selection_action, weights)
    selected_oof_action = actor_predict(actor, inputs, oof, device)
    selected_oof_cost = direct_cost(controller, data, oof, selected_oof_action, weights)

    def evaluation_block(
        selection_cost_value: np.ndarray,
        oof_cost_value: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "inner_warm_relative": warm_relative_metrics(
                selection_cost_value, data["warm_cost"][selection],
                data["speed_kph"][selection], data["variant_index"][selection],
            ),
            "development_oof_warm_relative": warm_relative_metrics(
                oof_cost_value, data["warm_cost"][oof],
                data["speed_kph"][oof], data["variant_index"][oof],
            ),
            "inner_round0_relative": metrics(
                selection_cost_value, initial_selection_cost,
                data["warm_cost"][selection], data["speed_kph"][selection],
            ),
            "development_oof_round0_relative": metrics(
                oof_cost_value, initial_oof_cost,
                data["warm_cost"][oof], data["speed_kph"][oof],
            ),
        }

    initial_eval = evaluation_block(initial_selection_cost, initial_oof_cost)
    latest_eval = evaluation_block(latest_selection_cost, latest_oof_cost)
    selected_eval = evaluation_block(selected_selection_cost, selected_oof_cost)
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
        selection_round_action=np.stack(selection_actions),
        selection_round_cost=np.stack(selection_costs),
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
        "qualification": "QUERY_OAC_GAMMA1_K_SCAN_TRAIN_ONLY",
        "arm": arm,
        "seed": seed,
        "selected_round": selected_round,
        "actor_update_count": int(config["pilot"]["rounds"]) * k,
        "critic_update_count_per_twin": int(config["pilot"]["rounds"]) * int(config["critic_updates"]["updates_per_round_per_twin"]),
        "actor_training": payload["actor_training"],
        "normalization": payload["normalization"],
        "critic1_training": trainings[0],
        "critic2_training": trainings[1],
        "selected_actor_state_dict": {name: value.detach().cpu() for name, value in selected_state.items()},
        "latest_actor_state_dict": {name: value.detach().cpu() for name, value in latest_state.items()},
        "selected_critic1_state_dict": {name: value.detach().cpu() for name, value in selected_critic_states[0].items()},
        "selected_critic2_state_dict": {name: value.detach().cpu() for name, value in selected_critic_states[1].items()},
        "latest_critic1_state_dict": {name: value.detach().cpu() for name, value in latest_critic_states[0].items()},
        "latest_critic2_state_dict": {name: value.detach().cpu() for name, value in latest_critic_states[1].items()},
        "fit_indices": fit,
        "selection_indices": selection,
        "oof_indices": oof,
        "source_pretrain_checkpoint": str(checkpoint_source),
        "source_pretrain_checkpoint_sha256": sha256(checkpoint_source),
        "source_replay_sha256": replay_manifest["replay_sha256"],
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }, checkpoint_path)
    return {
        "arm": arm["name"],
        "actor_updates_per_round": k,
        "seed": seed,
        "selected_round": selected_round,
        "initial": initial_eval,
        "latest": latest_eval,
        "selected": selected_eval,
        "rounds": round_records,
        "online_replay_rows": int(len(online["cost"])),
        "actor_batch_rows": int(len(actor_batches["actor_batch_state_index"])),
        "new_query_rollouts": int(len(online["cost"]) + 15 * len(selection)),
        "arrays": str(arrays_path),
        "arrays_sha256": sha256(arrays_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
    }


def pooled_warm(
    records: list[dict[str, Any]],
    stage: str,
    split: str,
    data: dict[str, np.ndarray],
) -> dict[str, Any]:
    costs, warms, speeds, variants = [], [], [], []
    for record in records:
        with np.load(record["arrays"], allow_pickle=False) as archive:
            rows = archive["selection_indices"] if split == "inner" else archive["oof_indices"]
            key = f"{stage}_selection_cost" if split == "inner" else f"{stage}_oof_cost"
            if stage == "round0":
                key = "selection_round_cost" if split == "inner" else "round0_oof_cost"
                cost = archive[key][0] if split == "inner" else archive[key]
            else:
                cost = archive[key]
            costs.append(np.asarray(cost))
            warms.append(data["warm_cost"][rows])
            speeds.append(data["speed_kph"][rows])
            variants.append(data["variant_index"][rows])
    return warm_relative_metrics(
        np.concatenate(costs), np.concatenate(warms),
        np.concatenate(speeds), np.concatenate(variants),
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    if config["formal_validation_or_test_consumed"] or config["dbm_fields_or_labels_consumed"]:
        raise AssertionError("sealed boundary violation")
    aggregation_dir = Path(config["sources"]["aggregation_ab"]).resolve()
    aggregation_validation = json.loads((aggregation_dir / "validation.json").read_text())
    if aggregation_validation["qualification"] != "QUERY_OAC_AGGREGATION_AB_INDEPENDENT_PASS":
        raise AssertionError("source aggregation A/B did not independently pass")
    replay_dir = Path(config["sources"]["absolute_replay"]).resolve()
    pretrain_dir = Path(config["sources"]["pretrain"]).resolve()
    replay_manifest = json.loads((replay_dir / "manifest.json").read_text())
    with np.load(replay_dir / "replay.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    fullrank_manifest = json.loads((Path(replay_manifest["fullrank_source"]) / "manifest.json").read_text())
    parent_manifest = json.loads((Path(fullrank_manifest["parent_t0"]) / "manifest.json").read_text())
    collection_manifest = json.loads((Path(parent_manifest["source_collection"]) / "manifest.json").read_text())
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    fit = np.flatnonzero(np.isin(data["fold_id"], config["split_contract"]["fit_folds"]))
    selection = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    oof = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
    bases = basis_bank()
    radii = np.linspace(
        float(config["pilot"]["probe_radius_sigma_start"]),
        float(config["pilot"]["probe_radius_sigma_end"]),
        int(config["pilot"]["rounds"]),
    ).astype(np.float32)
    output.mkdir(parents=True)
    records = []
    for arm in config["arms"]:
        for seed in config["pilot"]["seeds"]:
            records.append(run_arm_seed(
                config, arm, int(seed), data, replay_manifest, pretrain_dir,
                controller, weights, bases, radii, fit, selection, oof,
                output, device,
            ))
    by_arm = {
        arm["name"]: [record for record in records if record["arm"] == arm["name"]]
        for arm in config["arms"]
    }
    pooled = {
        arm: {
            stage: {
                "inner": pooled_warm(local, stage, "inner", data),
                "development_oof": pooled_warm(local, stage, "oof", data),
            }
            for stage in ("round0", "latest", "selected")
        }
        for arm, local in by_arm.items()
    }
    baseline = config["arms"][0]["name"]
    baseline_records = {int(record["seed"]): record for record in by_arm[baseline]}
    gate = config["decision_gate"]
    candidate_checks = {}
    decision = "GAMMA1_K_SCAN_FAIL_RETAIN_K1_COST_REFERENCE"
    for candidate in gate["candidate_order"]:
        candidate_records = {int(record["seed"]): record for record in by_arm[candidate]}
        seed_count = sum(
            candidate_records[seed]["selected"]["inner_warm_relative"]["actor_cost"]["mean"]
            < baseline_records[seed]["selected"]["inner_warm_relative"]["actor_cost"]["mean"]
            for seed in baseline_records
        )
        candidate_primary = pooled[candidate]["selected"]["inner"]
        baseline_primary = pooled[baseline]["selected"]["inner"]
        checks = {
            "actor_mean_cost_advantage_seed_count": seed_count >= int(gate["candidate_actor_mean_cost_advantage_seed_count_minimum"]),
            "pooled_aggregate_improvement_greater_than_k1": candidate_primary["aggregate_improvement"] > baseline_primary["aggregate_improvement"],
            "pooled_median_within_tolerance": candidate_primary["gain"]["median"] >= baseline_primary["gain"]["median"] - float(gate["candidate_pooled_warm_relative_median_no_worse_tolerance"]),
            "pooled_p05_within_tolerance": candidate_primary["gain"]["p05"] >= baseline_primary["gain"]["p05"] - float(gate["candidate_pooled_warm_relative_p05_no_worse_tolerance"]),
            "pooled_worst_within_tolerance": candidate_primary["gain"]["min"] >= baseline_primary["gain"]["min"] - float(gate["candidate_pooled_warm_relative_worst_no_worse_tolerance"]),
        }
        candidate_checks[candidate] = {
            "mean_cost_advantage_seed_count": int(seed_count),
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
        if decision.endswith("K1_COST_REFERENCE") and all(checks.values()):
            decision = f"ADVANCE_{candidate.upper()}_TO_LONGER_GAMMA1_QUERY_PILOT"
    summary = {
        "qualification": "QUERY_OAC_GAMMA1_K_SCAN_PENDING_INDEPENDENT_VALIDATION",
        "decision": decision,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": config,
        "records": records,
        "pooled_warm_relative": pooled,
        "candidate_checks": candidate_checks,
        "decision_population": "inner selected checkpoints only",
        "development_oof_role": "corroborating already-consumed evidence; not arm selection or untouched validation",
        "new_query_rollouts": int(sum(record["new_query_rollouts"] for record in records)),
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    summary_path = output / "summary.json"
    dump_json(summary_path, summary)
    manifest = {
        "schema_version": "query-oac-gamma1-k-scan-v1",
        "qualification": summary["qualification"],
        "decision": decision,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "absolute_replay": str(replay_dir),
        "absolute_replay_sha256": replay_manifest["replay_sha256"],
        "pretrain": str(pretrain_dir),
        "pretrain_manifest_sha256": sha256(pretrain_dir / "manifest.json"),
        "aggregation_ab": str(aggregation_dir),
        "aggregation_validation_sha256": sha256(aggregation_dir / "validation.json"),
        "query_checkpoint": replay_manifest["query_checkpoint"],
        "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
        "summary_sha256": sha256(summary_path),
        "run_artifacts": {
            f"{record['arm']}_seed{record['seed']}": {
                "arrays_sha256": record["arrays_sha256"],
                "checkpoint_sha256": record["checkpoint_sha256"],
            }
            for record in records
        },
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({
        "decision": decision,
        "candidate_checks": candidate_checks,
        "selected_inner": {
            arm: pooled[arm]["selected"]["inner"] for arm in pooled
        },
    }, indent=2, default=json_default))


if __name__ == "__main__":
    main()
