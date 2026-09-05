#!/usr/bin/env python3
"""Compare gamma-0 and gamma-1 Query Actor gradients at matched output steps."""

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
from run_query_single_center_oac20to1 import (  # noqa: E402
    actor_from_payload,
    actor_predict,
    direct_cost,
    distribution,
    load_inputs,
    metrics,
    physical_critic_value,
)


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_actor_aggregation_microstep_config_20260902_v1.json"


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


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def vector_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / max(denominator, 1e-30))


def predict_conservative_log(
    actor: torch.nn.Module,
    critics: list[ConfigurableAbsoluteActionValueCritic],
    trainings: list[dict],
    inputs: tuple[np.ndarray, ...],
    rows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = np.empty(len(rows), np.float32)
    actor.eval()
    for critic in critics:
        critic.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            local = rows[start : start + batch_size]
            action = torch.from_numpy(actor_predict(actor, inputs, local, device)).to(device)
            values = [
                physical_critic_value(critic, training, inputs, local, action, device)
                for critic, training in zip(critics, trainings)
            ]
            output[start : start + len(local)] = torch.maximum(values[0], values[1]).cpu().numpy()
    return output


def objective_weights(log_cost: np.ndarray, gamma: float, maximum: float) -> np.ndarray:
    if gamma == 0.0:
        return np.ones_like(log_cost, dtype=np.float32)
    value = np.minimum(np.exp(gamma * log_cost.astype(np.float64)), maximum)
    return (value / max(float(value.mean()), 1e-12)).astype(np.float32)


def actor_gradient(
    actor: torch.nn.Module,
    critics: list[ConfigurableAbsoluteActionValueCritic],
    trainings: list[dict],
    inputs: tuple[np.ndarray, ...],
    rows: np.ndarray,
    weights: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    actor.zero_grad(set_to_none=True)
    actor.eval()
    for critic in critics:
        critic.eval()
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
    for start in range(0, len(rows), batch_size):
        local = rows[start : start + batch_size]
        tensors = tuple(torch.from_numpy(value[local]).to(device) for value in inputs)
        action = actor(*tensors)[1]
        values = [
            physical_critic_value(critic, training, inputs, local, action, device)
            for critic, training in zip(critics, trainings)
        ]
        conservative = torch.maximum(values[0], values[1])
        local_weight = torch.from_numpy(weights[start : start + len(local)]).to(device)
        (local_weight * conservative).sum().div(len(rows)).backward()
    for critic in critics:
        for parameter in critic.parameters():
            parameter.requires_grad_(True)
    names, offsets, values = [], [0], []
    for name, parameter in actor.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"missing Actor gradient for {name}")
        names.append(name)
        values.append(parameter.grad.detach().reshape(-1).cpu().numpy().astype(np.float32))
        offsets.append(offsets[-1] + parameter.numel())
    return np.concatenate(values), names, np.asarray(offsets, np.int64)


def state_from_gradient(
    base_state: dict[str, torch.Tensor],
    names: list[str],
    offsets: np.ndarray,
    gradient: np.ndarray,
    multiplier: float,
) -> dict[str, torch.Tensor]:
    output = copy.deepcopy(base_state)
    for index, name in enumerate(names):
        shape = base_state[name].shape
        local = torch.from_numpy(
            gradient[offsets[index] : offsets[index + 1]].reshape(shape)
        ).to(base_state[name].device)
        output[name] = base_state[name] - float(multiplier) * local
    return output


def output_rms(
    actor: torch.nn.Module,
    base_output: np.ndarray,
    inputs: tuple[np.ndarray, ...],
    rows: np.ndarray,
    sigma: np.ndarray,
    device: torch.device,
) -> float:
    current = actor_predict(actor, inputs, rows, device)
    return float(np.sqrt(np.mean(np.square((current - base_output) / sigma))))


def calibrate_step(
    actor: torch.nn.Module,
    base_state: dict[str, torch.Tensor],
    base_output: np.ndarray,
    inputs: tuple[np.ndarray, ...],
    rows: np.ndarray,
    sigma: np.ndarray,
    gradient: np.ndarray,
    names: list[str],
    offsets: np.ndarray,
    target: float,
    device: torch.device,
) -> tuple[float, float, dict[str, torch.Tensor]]:
    norm = float(np.linalg.norm(gradient.astype(np.float64)))
    if norm <= 1e-20:
        raise AssertionError("zero Actor parameter gradient")
    direction = gradient / norm
    lower, upper = 0.0, 1e-7
    while True:
        actor.load_state_dict(state_from_gradient(base_state, names, offsets, direction, upper), strict=True)
        value = output_rms(actor, base_output, inputs, rows, sigma, device)
        if value >= target:
            break
        upper *= 2.0
        if upper > 10.0:
            raise AssertionError("failed to bracket matched output step")
    for _ in range(32):
        midpoint = 0.5 * (lower + upper)
        actor.load_state_dict(state_from_gradient(base_state, names, offsets, direction, midpoint), strict=True)
        value = output_rms(actor, base_output, inputs, rows, sigma, device)
        if value < target:
            lower = midpoint
        else:
            upper = midpoint
    multiplier = 0.5 * (lower + upper)
    state = state_from_gradient(base_state, names, offsets, direction, multiplier)
    actor.load_state_dict(state, strict=True)
    achieved = output_rms(actor, base_output, inputs, rows, sigma, device)
    return multiplier, achieved, state


def gradient_report(
    aggregate: np.ndarray,
    speed_gradients: np.ndarray,
    speed_values: np.ndarray,
) -> dict[str, Any]:
    matrix = np.empty((len(speed_values), len(speed_values)), np.float64)
    for left in range(len(speed_values)):
        for right in range(len(speed_values)):
            matrix[left, right] = vector_cosine(speed_gradients[left], speed_gradients[right])
    off_diagonal = matrix[np.triu_indices(len(speed_values), 1)]
    return {
        "aggregate_norm": float(np.linalg.norm(aggregate.astype(np.float64))),
        "speed_gradient_norm": {
            str(int(speed)): float(np.linalg.norm(speed_gradients[index].astype(np.float64)))
            for index, speed in enumerate(speed_values)
        },
        "aggregate_to_speed_cosine": {
            str(int(speed)): vector_cosine(aggregate, speed_gradients[index])
            for index, speed in enumerate(speed_values)
        },
        "speed_pair_cosine_matrix": matrix.tolist(),
        "speed_pair_negative_fraction": float(np.mean(off_diagonal < 0.0)),
        "speed_pair_cosine_minimum": float(off_diagonal.min()),
        "speed_pair_cosine_median": float(np.median(off_diagonal)),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    if config["formal_validation_or_test_consumed"] or config["dbm_fields_or_labels_consumed"]:
        raise AssertionError("sealed boundary violation")
    if config["query_analytic_gradient_consumed"]:
        raise AssertionError("Query analytic gradient is forbidden")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    replay_dir = Path(config["sources"]["absolute_replay"]).resolve()
    oac_dir = Path(config["sources"]["oac20to1"]).resolve()
    replay_manifest = json.loads((replay_dir / "manifest.json").read_text())
    oac_manifest = json.loads((oac_dir / "manifest.json").read_text())
    oac_validation = json.loads((oac_dir / "validation.json").read_text())
    if oac_validation["qualification"] != "QUERY_SINGLE_CENTER_OAC20TO1_INDEPENDENT_PASS":
        raise AssertionError("source OAC20:1 artifact did not independently pass")
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
    output.mkdir(parents=True)
    sigma = np.asarray(config["step_contract"]["noise_sigma"], np.float32).reshape(1, 2)
    steps = np.asarray(config["matched_output_steps_sigma_rms"], np.float64)
    speed_values = np.asarray(sorted(np.unique(data["speed_kph"])), np.int64)
    arm_names = [value["name"] for value in config["arms"]]
    records = []
    pooled_gain: dict[str, list[np.ndarray]] = {name: [] for name in arm_names}
    for seed in config["population"]["seeds"]:
        seed = int(seed)
        seed_dir = oac_dir / f"seed_{seed}"
        checkpoint_path = seed_dir / "checkpoint.pt"
        arrays_path = seed_dir / "pilot_arrays.npz"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        with np.load(arrays_path, allow_pickle=False) as archive:
            source_arrays = {name: np.asarray(archive[name]) for name in archive.files}
        fit = checkpoint["fit_indices"].astype(np.int64)
        selection = checkpoint["selection_indices"].astype(np.int64)
        oof = checkpoint["oof_indices"].astype(np.int64)
        inputs = load_inputs(data, checkpoint["normalization"])
        payload = {
            "actor_training": checkpoint["actor_training"],
            "selected_actor_state_dict": checkpoint["selected_actor_state_dict"],
        }
        actor = actor_from_payload(payload, "selected_actor_state_dict", device)
        critics, trainings = [], []
        for twin in (1, 2):
            critic = ConfigurableAbsoluteActionValueCritic().to(device)
            critic.load_state_dict(checkpoint[f"selected_critic{twin}_state_dict"], strict=True)
            critic.eval()
            critics.append(critic)
            trainings.append(checkpoint[f"critic{twin}_training"])
        base_state = copy.deepcopy(actor.state_dict())
        base_fit_action = actor_predict(actor, inputs, fit, device)
        base_selection_action = actor_predict(actor, inputs, selection, device)
        base_selection_cost = source_arrays["selected_selection_cost"].astype(np.float32)
        if np.max(np.abs(base_selection_action - source_arrays["selected_selection_action"])) > 1e-6:
            raise AssertionError("selected Actor reload mismatch")
        conservative_log = predict_conservative_log(
            actor, critics, trainings, inputs, fit, device,
            int(config["gradient_contract"]["batch_size"]),
        )
        arm_gradients, arm_speed_gradients, arm_weights = [], [], []
        arm_step_actions, arm_step_costs = [], []
        arm_multipliers, arm_achieved = [], []
        arm_reports = []
        parameter_names: list[str] | None = None
        parameter_offsets: np.ndarray | None = None
        for arm in config["arms"]:
            gamma = float(arm["gamma"])
            weights = objective_weights(conservative_log, gamma, float(config["weight_maximum"]))
            actor.load_state_dict(base_state, strict=True)
            gradient, names, offsets = actor_gradient(
                actor, critics, trainings, inputs, fit, weights, device,
                int(config["gradient_contract"]["batch_size"]),
            )
            if parameter_names is None:
                parameter_names, parameter_offsets = names, offsets
            elif names != parameter_names or not np.array_equal(offsets, parameter_offsets):
                raise AssertionError("Actor parameter layout changed")
            speed_gradients = []
            for speed in speed_values:
                rows = fit[data["speed_kph"][fit] == speed]
                local_log = predict_conservative_log(
                    actor, critics, trainings, inputs, rows, device,
                    int(config["gradient_contract"]["batch_size"]),
                )
                local_weights = objective_weights(local_log, gamma, float(config["weight_maximum"]))
                local_gradient, local_names, local_offsets = actor_gradient(
                    actor, critics, trainings, inputs, rows, local_weights, device,
                    int(config["gradient_contract"]["batch_size"]),
                )
                if local_names != names or not np.array_equal(local_offsets, offsets):
                    raise AssertionError("speed-gradient parameter layout changed")
                speed_gradients.append(local_gradient)
            speed_gradients_np = np.stack(speed_gradients)
            local_actions, local_costs, multipliers, achieved, step_reports = [], [], [], [], []
            for target in steps:
                actor.load_state_dict(base_state, strict=True)
                multiplier, actual, _ = calibrate_step(
                    actor, base_state, base_fit_action, inputs, fit, sigma,
                    gradient, names, offsets, float(target), device,
                )
                action = actor_predict(actor, inputs, selection, device)
                cost = direct_cost(
                    controller, data, selection, action,
                    {name: float(value) for name, value in config["cost_weights"].items()},
                )
                report = metrics(
                    cost, base_selection_cost, data["warm_cost"][selection],
                    data["speed_kph"][selection],
                )
                local_actions.append(action)
                local_costs.append(cost)
                multipliers.append(multiplier)
                achieved.append(actual)
                step_reports.append({
                    "target_output_step_sigma_rms": float(target),
                    "achieved_output_step_sigma_rms": actual,
                    "normalized_parameter_step": multiplier,
                    "inner": report,
                })
            effective_fraction = float(
                np.square(weights.sum()) / (len(weights) * np.square(weights).sum())
            )
            arm_reports.append({
                "name": arm["name"],
                "gamma": gamma,
                "weight": distribution(weights),
                "weight_effective_sample_fraction": effective_fraction,
                "weight_clip_fraction": float(np.mean(np.exp(gamma * conservative_log.astype(np.float64)) >= float(config["weight_maximum"]))) if gamma else 0.0,
                "gradient": gradient_report(gradient, speed_gradients_np, speed_values),
                "steps": step_reports,
            })
            arm_gradients.append(gradient)
            arm_speed_gradients.append(speed_gradients_np)
            arm_weights.append(weights)
            arm_step_actions.append(np.stack(local_actions))
            arm_step_costs.append(np.stack(local_costs))
            arm_multipliers.append(np.asarray(multipliers, np.float64))
            arm_achieved.append(np.asarray(achieved, np.float64))
            primary = int(np.flatnonzero(np.isclose(steps, float(config["primary_step_sigma_rms"])))[0])
            pooled_gain[arm["name"]].append(base_selection_cost - local_costs[primary])
        cross_arm_cosine = vector_cosine(arm_gradients[0], arm_gradients[1])
        arrays_output = output / f"seed_{seed}.npz"
        np.savez_compressed(
            arrays_output,
            fit_indices=fit,
            selection_indices=selection,
            oof_indices_untouched=oof,
            speed_values=speed_values,
            arm_names=np.asarray(arm_names),
            parameter_names=np.asarray(parameter_names),
            parameter_offsets=parameter_offsets,
            base_fit_action=base_fit_action,
            base_selection_action=base_selection_action,
            base_selection_cost=base_selection_cost,
            fit_conservative_log_cost=conservative_log,
            arm_weight=np.stack(arm_weights),
            arm_parameter_gradient=np.stack(arm_gradients),
            arm_speed_parameter_gradient=np.stack(arm_speed_gradients),
            matched_step_target=steps,
            matched_step_parameter_multiplier=np.stack(arm_multipliers),
            matched_step_achieved=np.stack(arm_achieved),
            matched_step_selection_action=np.stack(arm_step_actions),
            matched_step_selection_cost=np.stack(arm_step_costs),
        )
        record = {
            "seed": seed,
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": sha256(checkpoint_path),
            "source_arrays": str(arrays_path),
            "source_arrays_sha256": sha256(arrays_path),
            "output_arrays": str(arrays_output),
            "output_arrays_sha256": sha256(arrays_output),
            "cross_arm_parameter_gradient_cosine": cross_arm_cosine,
            "arms": arm_reports,
            "oof_evaluated": False,
            "new_query_rollouts": int(len(selection) * len(steps) * len(config["arms"])),
        }
        records.append(record)
        primary = int(np.flatnonzero(np.isclose(steps, float(config["primary_step_sigma_rms"])))[0])
        print(
            f"seed={seed} grad_cos={cross_arm_cosine:.3f} "
            + " ".join(
                f"{arm_names[index]} Jgain={arm_reports[index]['steps'][primary]['inner']['gain_vs_round0']['mean']:.6f}"
                for index in range(len(arm_names))
            ),
            flush=True,
        )
    primary = int(np.flatnonzero(np.isclose(steps, float(config["primary_step_sigma_rms"])))[0])
    gamma0 = config["arms"][0]["name"]
    gamma1 = config["arms"][1]["name"]
    highspeed_counts = {}
    for speed in (85, 100):
        highspeed_counts[str(speed)] = int(sum(
            record["arms"][1]["steps"][primary]["inner"]["by_speed_kph"][str(speed)]["gain_vs_round0"]["mean"]
            > record["arms"][0]["steps"][primary]["inner"]["by_speed_kph"][str(speed)]["gain_vs_round0"]["mean"]
            for record in records
        ))
    pooled0 = np.concatenate(pooled_gain[gamma0]).astype(np.float64)
    pooled1 = np.concatenate(pooled_gain[gamma1]).astype(np.float64)
    routing_checks = {
        "gamma1_85kmh_advantage_at_least_2_of_3": highspeed_counts["85"] >= 2,
        "gamma1_100kmh_advantage_at_least_2_of_3": highspeed_counts["100"] >= 2,
        "gamma1_pooled_mean_no_worse": float(pooled1.mean()) >= float(pooled0.mean()),
        "gamma1_pooled_p05_within_0p25": float(np.quantile(pooled1, 0.05)) >= float(np.quantile(pooled0, 0.05)) - 0.25,
    }
    highspeed_pass = routing_checks["gamma1_85kmh_advantage_at_least_2_of_3"] and routing_checks["gamma1_100kmh_advantage_at_least_2_of_3"]
    if all(routing_checks.values()):
        decision = "ADVANCE_GAMMA1_TO_PAIRED_CONTINUOUS_OAC20TO1"
    elif highspeed_pass:
        decision = "HIGHSPEED_MEAN_IMPROVES_BUT_TAIL_MIXED_TEST_GAMMA0P5"
    else:
        decision = "GAMMA1_DOES_NOT_FIX_HIGHSPEED_RETAIN_GAMMA0_AUDIT_SPEED_CONFLICT"
    summary = {
        "qualification": "QUERY_ACTOR_AGGREGATION_MICROSTEP_PENDING_INDEPENDENT_VALIDATION",
        "decision": decision,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": config,
        "records": records,
        "primary_step_sigma_rms": float(config["primary_step_sigma_rms"]),
        "highspeed_gamma1_advantage_seed_count": highspeed_counts,
        "routing_checks": routing_checks,
        "pooled_primary": {
            gamma0: distribution(pooled0),
            gamma1: distribution(pooled1),
            "gamma1_minus_gamma0_gain": distribution(pooled1 - pooled0),
        },
        "new_query_rollouts": int(sum(record["new_query_rollouts"] for record in records)),
        "oof_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    summary_path = output / "summary.json"
    dump_json(summary_path, summary)
    manifest = {
        "schema_version": "query-actor-aggregation-microstep-v1",
        "qualification": summary["qualification"],
        "decision": decision,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "absolute_replay": str(replay_dir),
        "absolute_replay_sha256": replay_manifest["replay_sha256"],
        "source_oac20to1": str(oac_dir),
        "source_oac20to1_manifest_sha256": sha256(oac_dir / "manifest.json"),
        "source_oac20to1_validation_sha256": sha256(oac_dir / "validation.json"),
        "query_checkpoint": replay_manifest["query_checkpoint"],
        "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
        "summary_sha256": sha256(summary_path),
        "seed_arrays_sha256": {
            str(record["seed"]): record["output_arrays_sha256"] for record in records
        },
        "oof_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({
        "decision": decision,
        "routing_checks": routing_checks,
        "pooled_primary": summary["pooled_primary"],
    }, indent=2))


if __name__ == "__main__":
    main()
