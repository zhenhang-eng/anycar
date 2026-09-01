#!/usr/bin/env python3
"""Audit shared-Actor parameter-gradient conflict under exact DBM J50.

This is a train-side mechanism audit for the latest deterministic-center DBM
Actors.  It does not train Actor/Critic, generate Replay, load formal/test
splits, run an MPPI wrapper, or run closed loop.

For each of three frozen Actors and the same 600 internal-selection states it
computes:

* the exact per-state action gradient and Actor parameter gradient;
* exact cancellation and alignment with the batch-mean parameter gradient;
* exact speed/scenario/cost-quartile/warm-outcome group gradients;
* independent action-space, shared raw-SGD, and saved-Adam parameter steps at
  matched output RMS, followed by deterministic DBM J50 rollout.

The independent action step is a mechanism reference, not a deployable policy.
Warm is used only to define reporting strata and never enters a gradient.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from analyze_mppi_oac2_critic_dbm_gradient_gap import (
    calibrated_adam_objective_step,
)
from evaluate_mppi_oac_warm_relative_centers import checkpoint_actor
from generate_dbm_proposal_teacher import sha256_file
from run_mppi_absolute_action_value_critic_cv import make_folds
from train_mppi_oac2_continuous_actor import (
    actor_tensor,
    differentiable_dbm_cost,
    internal_split,
)
from train_mppi_online_absolute_sac import (
    BASE_SIGMA,
    actor_mean,
    load_actor_normalization,
    load_bank,
    make_actor_inputs,
    rollout_bank,
)


DEFAULT_RUN = Path(
    "outputs/mppi_proposal/"
    "online_absolute_sac_oac2_deterministic_center_dbm_k16_90round_20260828_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/oac2_actor_parameter_conflict_audit_20260828_v1"
)
SEEDS = (0, 1, 2)
PARAMETER_GROUPS = ("all", "encoder", "decoder", "support_adapter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gradient-batch-size", type=int, default=32)
    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--step-radii", default="0.002,0.02")
    parser.add_argument(
        "--max-states", type=int, default=0,
        help="Contract smoke limit. Zero uses all internal-selection states.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def parameter_group(name: str) -> str:
    if name.startswith("encoder."):
        return "encoder"
    if name.startswith("decoder."):
        return "decoder"
    if name.startswith("support_adapter"):
        return "support_adapter"
    raise AssertionError(f"unregistered Actor parameter group: {name}")


def named_trainable_parameters(actor: torch.nn.Module):
    return [(name, value) for name, value in actor.named_parameters() if value.requires_grad]


def zero_gradient_like(parameters) -> list[torch.Tensor]:
    return [torch.zeros_like(parameter, device="cpu") for _, parameter in parameters]


def add_gradient_(total: list[torch.Tensor], gradient, parameters) -> None:
    for target, value, (_, parameter) in zip(total, gradient, parameters):
        source = torch.zeros_like(parameter) if value is None else value
        target.add_(source.detach().cpu())


def flat_gradient(gradient: list[torch.Tensor]) -> np.ndarray:
    return np.concatenate([
        value.detach().reshape(-1).cpu().numpy().astype(np.float32)
        for value in gradient
    ])


def gradient_norm(gradient: list[torch.Tensor], names: list[str], group: str) -> float:
    total = 0.0
    for name, value in zip(names, gradient):
        if group == "all" or parameter_group(name) == group:
            total += float(torch.sum(value.double() * value.double()))
    return float(np.sqrt(total))


def gradient_dot(
    left: list[torch.Tensor], right: list[torch.Tensor], names: list[str], group: str,
) -> float:
    total = 0.0
    for name, a, b in zip(names, left, right):
        if group == "all" or parameter_group(name) == group:
            total += float(torch.sum(a.double() * b.double()))
    return total


def vector_comparison(
    left: list[torch.Tensor], right: list[torch.Tensor], names: list[str], group: str,
) -> dict[str, float]:
    left_norm = gradient_norm(left, names, group)
    right_norm = gradient_norm(right, names, group)
    return {
        "cosine": float(
            gradient_dot(left, right, names, group)
            / max(left_norm * right_norm, 1e-30)
        ),
        "left_norm": left_norm,
        "right_norm": right_norm,
        "norm_ratio": left_norm / max(right_norm, 1e-30),
    }


def exact_parameter_gradient(
    actor: torch.nn.Module,
    actor_inputs: tuple[np.ndarray, ...],
    indices: np.ndarray,
    states: np.ndarray,
    current: np.ndarray,
    references: np.ndarray,
    backend: TorchDynamicBicycleRolloutBackend,
    weights: TorchMPPICostWeights,
    params: TorchMPPIParams,
    device: torch.device,
    batch_size: int,
) -> tuple[list[torch.Tensor], np.ndarray]:
    """Return the exact gradient of mean raw J and per-state costs."""
    parameters = named_trainable_parameters(actor)
    total = zero_gradient_like(parameters)
    costs = []
    count = len(indices)
    for begin in range(0, count, batch_size):
        local = indices[begin:begin + batch_size]
        action = actor_tensor(actor, actor_inputs, local, device)
        cost = differentiable_dbm_cost(
            action, local, states, current, references,
            backend, weights, params, device,
        )
        gradient = torch.autograd.grad(
            cost.sum() / float(count),
            [parameter for _, parameter in parameters],
            allow_unused=True,
        )
        add_gradient_(total, gradient, parameters)
        costs.append(cost.detach().cpu().numpy())
    return total, np.concatenate(costs).astype(np.float32)


def per_state_gradient_metrics(
    actor: torch.nn.Module,
    actor_inputs: tuple[np.ndarray, ...],
    indices: np.ndarray,
    states: np.ndarray,
    current: np.ndarray,
    references: np.ndarray,
    backend: TorchDynamicBicycleRolloutBackend,
    weights: TorchMPPICostWeights,
    params: TorchMPPIParams,
    global_gradient: list[torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    parameters = named_trainable_parameters(actor)
    names = [name for name, _ in parameters]
    global_device = [value.to(device) for value in global_gradient]
    count = len(indices)
    action_gradient = np.empty((count, 8, 2), np.float32)
    action_gradient_norm = np.empty(count, np.float32)
    norms = np.empty((count, len(PARAMETER_GROUPS)), np.float32)
    cosine_global = np.empty_like(norms)
    dot_global = np.empty_like(norms)
    cursor = 0
    for begin in range(0, count, batch_size):
        local = indices[begin:begin + batch_size]
        action = actor_tensor(actor, actor_inputs, local, device)
        cost = differentiable_dbm_cost(
            action, local, states, current, references,
            backend, weights, params, device,
        )
        action_grad = torch.autograd.grad(cost.sum(), action, retain_graph=True)[0]
        action_gradient[cursor:cursor + len(local)] = action_grad.detach().cpu().numpy()
        action_gradient_norm[cursor:cursor + len(local)] = (
            action_grad.detach().reshape(len(local), -1).norm(dim=1).cpu().numpy()
        )
        for offset in range(len(local)):
            scalar = torch.sum(action[offset] * action_grad[offset].detach())
            gradient_raw = torch.autograd.grad(
                scalar,
                [parameter for _, parameter in parameters],
                retain_graph=offset + 1 < len(local),
                allow_unused=True,
            )
            gradient = [
                torch.zeros_like(parameter) if value is None else value
                for value, (_, parameter) in zip(gradient_raw, parameters)
            ]
            for group_index, group in enumerate(PARAMETER_GROUPS):
                norm_sq = torch.zeros((), dtype=torch.float64, device=device)
                dot = torch.zeros((), dtype=torch.float64, device=device)
                global_sq = torch.zeros((), dtype=torch.float64, device=device)
                for name, value, reference in zip(names, gradient, global_device):
                    if group == "all" or parameter_group(name) == group:
                        norm_sq += torch.sum(value.double() * value.double())
                        dot += torch.sum(value.double() * reference.double())
                        global_sq += torch.sum(reference.double() * reference.double())
                norm = torch.sqrt(norm_sq)
                global_norm = torch.sqrt(global_sq)
                row = cursor + offset
                norms[row, group_index] = float(norm)
                dot_global[row, group_index] = float(dot)
                cosine_global[row, group_index] = float(
                    dot / torch.clamp(norm * global_norm, min=1e-30)
                )
        cursor += len(local)
        print(f"  per-state gradients {cursor}/{count}", flush=True)
    return {
        "action_gradient": action_gradient,
        "action_gradient_norm": action_gradient_norm,
        "parameter_norm": norms,
        "parameter_cosine_global": cosine_global,
        "parameter_dot_global": dot_global,
    }


def strata_masks(
    speed: np.ndarray,
    scenario: np.ndarray,
    actor_cost: np.ndarray,
    warm_cost: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {
        "speed": {}, "scenario": {}, "cost_quartile": {}, "warm_outcome": {},
    }
    for value in sorted(np.unique(speed)):
        result["speed"][f"{float(value):.1f}"] = np.isclose(speed, value)
    for value in sorted(np.unique(scenario)):
        result["scenario"][str(value)] = scenario == value
    order = np.argsort(actor_cost, kind="stable")
    for quartile, local in enumerate(np.array_split(order, 4)):
        mask = np.zeros(len(actor_cost), bool)
        mask[local] = True
        result["cost_quartile"][f"q{quartile}"] = mask
    gain = warm_cost - actor_cost
    result["warm_outcome"]["actor_beats_warm"] = gain > 1e-6
    result["warm_outcome"]["actor_not_better_warm"] = gain <= 1e-6
    return result


def group_gradient_audit(
    actor: torch.nn.Module,
    actor_inputs: tuple[np.ndarray, ...],
    selection: np.ndarray,
    masks: dict[str, dict[str, np.ndarray]],
    states: np.ndarray,
    current: np.ndarray,
    references: np.ndarray,
    backend: TorchDynamicBicycleRolloutBackend,
    weights: TorchMPPICostWeights,
    params: TorchMPPIParams,
    global_gradient: list[torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    names = [name for name, _ in named_trainable_parameters(actor)]
    result: dict[str, Any] = {}
    for family, entries in masks.items():
        gradients = {}
        records = {}
        for label, mask in entries.items():
            if not np.any(mask):
                continue
            gradient, cost = exact_parameter_gradient(
                actor, actor_inputs, selection[mask], states, current, references,
                backend, weights, params, device, batch_size,
            )
            gradients[label] = gradient
            records[label] = {
                "count": int(np.sum(mask)),
                "cost": distribution(cost),
                "cosine_to_global": {
                    group: vector_comparison(
                        gradient, global_gradient, names, group,
                    )["cosine"] for group in PARAMETER_GROUPS
                },
                "norm": {
                    group: gradient_norm(gradient, names, group)
                    for group in PARAMETER_GROUPS
                },
            }
        labels = list(gradients)
        matrices = {}
        for group in PARAMETER_GROUPS:
            matrices[group] = [
                [
                    vector_comparison(
                        gradients[left], gradients[right], names, group,
                    )["cosine"]
                    for right in labels
                ] for left in labels
            ]
        result[family] = {
            "labels": labels,
            "records": records,
            "pairwise_cosine": matrices,
        }
    return result


def actor_actions(
    actor: torch.nn.Module,
    actor_inputs: tuple[np.ndarray, ...],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    return actor_mean(actor, actor_inputs, indices, device, batch_size=batch_size)


def raw_parameter_step(
    actor: torch.nn.Module,
    gradient: list[torch.Tensor],
    actor_inputs: tuple[np.ndarray, ...],
    selection: np.ndarray,
    base_action: np.ndarray,
    target_rms: float,
    device: torch.device,
) -> tuple[np.ndarray, float, float]:
    sigma = BASE_SIGMA.reshape(1, 1, 2)
    learning_rate = 1e-6
    updated = base_action
    actual = 0.0
    for _ in range(10):
        candidate = copy.deepcopy(actor).to(device)
        with torch.no_grad():
            for parameter, value in zip(candidate.parameters(), gradient):
                parameter.add_(value.to(device), alpha=-float(learning_rate))
        updated = actor_actions(candidate, actor_inputs, selection, device)
        actual = float(np.sqrt(np.mean(((updated - base_action) / sigma) ** 2)))
        if actual <= 1e-14:
            learning_rate *= 10.0
            continue
        ratio = float(target_rms) / actual
        if abs(ratio - 1.0) <= 0.01:
            break
        learning_rate *= float(np.clip(ratio, 0.1, 10.0))
    return updated.astype(np.float32), actual, float(learning_rate)


def independent_action_step(
    action: np.ndarray, action_gradient: np.ndarray, target_rms: float,
) -> np.ndarray:
    sigma = BASE_SIGMA.reshape(1, 1, 2).astype(np.float64)
    grad_z = np.asarray(action_gradient, np.float64) * sigma
    rms = np.sqrt(np.mean(grad_z * grad_z, axis=(1, 2), keepdims=True))
    delta_z = -float(target_rms) * grad_z / np.maximum(rms, 1e-12)
    return np.clip(action + delta_z * sigma, -1.0, 1.0).astype(np.float32)


def step_metrics(
    base_action: np.ndarray,
    candidate_action: np.ndarray,
    base_cost: np.ndarray,
    candidate_cost: np.ndarray,
    action_gradient: np.ndarray,
    speed: np.ndarray,
) -> dict[str, Any]:
    sigma = BASE_SIGMA.reshape(1, 1, 2).astype(np.float64)
    delta_z = (candidate_action - base_action) / sigma
    descent = -np.asarray(action_gradient, np.float64) * sigma
    numerator = np.sum(delta_z * descent, axis=(1, 2))
    denominator = (
        np.linalg.norm(delta_z.reshape(len(delta_z), -1), axis=1)
        * np.linalg.norm(descent.reshape(len(descent), -1), axis=1)
    )
    cosine = numerator / np.maximum(denominator, 1e-12)
    move_rms = np.sqrt(np.mean(delta_z * delta_z, axis=(1, 2)))
    gain = np.asarray(base_cost, np.float64) - np.asarray(candidate_cost, np.float64)
    result: dict[str, Any] = {
        "cost": distribution(candidate_cost),
        "gain": distribution(gain),
        "regression_fraction": float(np.mean(gain < -1e-6)),
        "material_regression_0_1_fraction": float(np.mean(gain < -0.1)),
        "move_sigma_rms": distribution(move_rms),
        "action_descent_cosine": distribution(cosine),
        "negative_descent_alignment_fraction": float(np.mean(cosine < 0.0)),
        "by_speed": {},
    }
    for value in sorted(np.unique(speed)):
        mask = np.isclose(speed, value)
        result["by_speed"][f"{float(value):.1f}"] = {
            "count": int(np.sum(mask)),
            "gain_mean": float(np.mean(gain[mask])),
            "gain_median": float(np.median(gain[mask])),
            "gain_p05": float(np.quantile(gain[mask], 0.05)),
            "regression_fraction": float(np.mean(gain[mask] < -1e-6)),
            "descent_cosine_median": float(np.median(cosine[mask])),
        }
    return result


def per_state_conflict_summary(
    metrics: dict[str, np.ndarray],
    global_gradient: list[torch.Tensor],
    parameter_names: list[str],
) -> dict[str, Any]:
    result = {}
    count = len(metrics["parameter_norm"])
    for group_index, group in enumerate(PARAMETER_GROUPS):
        norms = metrics["parameter_norm"][:, group_index]
        cosine = metrics["parameter_cosine_global"][:, group_index]
        global_norm = gradient_norm(global_gradient, parameter_names, group)
        result[group] = {
            "global_mean_gradient_norm": global_norm,
            "per_state_gradient_norm": distribution(norms),
            "cosine_to_global": distribution(cosine),
            "negative_alignment_fraction": float(np.mean(cosine < 0.0)),
            "strong_negative_alignment_fraction": float(np.mean(cosine < -0.25)),
            "cancellation_ratio": float(
                count * global_norm / max(float(np.sum(norms)), 1e-30)
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    radii = tuple(float(value) for value in args.step_radii.split(","))
    if not radii or any(value <= 0.0 for value in radii):
        raise ValueError(radii)

    contract = json.loads((args.run / "contract.json").read_text())
    run_args = contract["arguments"]
    if run_args.get("actor_objective_mode") != "deterministic_center_dbm":
        raise AssertionError("audit requires deterministic-center DBM run")
    if contract.get("formal_validation_loaded") or contract.get("test_loaded"):
        raise AssertionError("sealed split violation")

    data = load_bank(Path(run_args["bank_root"]))
    folds = make_folds(data, 3)
    _, selection, _, selection_episodes = internal_split(
        data, folds, int(contract["outer_fold"])
    )
    if args.max_states > 0:
        selection = selection[:args.max_states]
        selection_episodes = sorted(np.unique(data["episode"][selection]).tolist())
    states, current, references, params_json, weights_json, dbm_json = (
        load_rollout_inputs(data, Path(run_args["gt_v1"]))
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    normalization, normalization_path = load_actor_normalization(Path(run_args["base_ac"]))
    actor_inputs = make_actor_inputs(data, normalization)
    speed = np.asarray(data["speed"][selection], np.float32)
    scenario = np.asarray(data["scenario"][selection])
    episode = np.asarray(data["episode"][selection])
    warm_cost = np.asarray(data["costs"][selection, 0], np.float32)

    all_actor_action = []
    all_actor_cost = []
    all_action_gradient = []
    all_param_norm = []
    all_param_cosine = []
    all_param_dot = []
    all_global_flat = []
    all_step_action = []
    all_step_cost = []
    all_step_cosine = []
    per_seed = {}
    checkpoint_hashes = []
    parameter_names = None
    parameter_numels = None
    parameter_groups = None
    step_names = ["base"]
    for radius in radii:
        token = str(radius).replace(".", "p")
        step_names.extend((f"independent_{token}", f"shared_sgd_{token}", f"shared_adam_{token}"))

    for seed in SEEDS:
        print(f"seed={seed}: load Actor and compute global mean-J gradient", flush=True)
        actor, checkpoint = checkpoint_actor(
            args.run, seed, float(run_args["actor_output_support_multiplier"]), device
        )
        payload = torch.load(checkpoint, map_location=device)
        actor.eval()
        parameters = named_trainable_parameters(actor)
        names = [name for name, _ in parameters]
        numels = [parameter.numel() for _, parameter in parameters]
        groups = [parameter_group(name) for name in names]
        if parameter_names is None:
            parameter_names = names
            parameter_numels = numels
            parameter_groups = groups
        elif names != parameter_names or numels != parameter_numels:
            raise AssertionError("Actor parameter contract differs across seeds")

        global_gradient, global_cost = exact_parameter_gradient(
            actor, actor_inputs, selection, states, current, references,
            backend, weights, params, device, args.gradient_batch_size,
        )
        base_action = actor_actions(actor, actor_inputs, selection, device)
        replay_error = float(np.max(np.abs(
            global_cost - rollout_bank(
                backend, weights, params, base_action[:, None], states, current,
                references, selection, args.rollout_batch_size, device,
            )[:, 0]
        )))
        print(f"seed={seed}: per-state parameter gradients", flush=True)
        metrics = per_state_gradient_metrics(
            actor, actor_inputs, selection, states, current, references,
            backend, weights, params, global_gradient, device,
            args.gradient_batch_size,
        )
        conflict = per_state_conflict_summary(metrics, global_gradient, names)
        masks = strata_masks(speed, scenario, global_cost, warm_cost)
        print(f"seed={seed}: exact group gradients", flush=True)
        grouped = group_gradient_audit(
            actor, actor_inputs, selection, masks, states, current, references,
            backend, weights, params, global_gradient, device,
            args.gradient_batch_size,
        )

        actions = [base_action]
        actual_step = {"base": 0.0}
        step_learning_rate = {"base": 0.0}
        for radius in radii:
            token = str(radius).replace(".", "p")
            independent = independent_action_step(
                base_action, metrics["action_gradient"], radius,
            )
            shared_sgd, sgd_rms, sgd_lr = raw_parameter_step(
                actor, global_gradient, actor_inputs, selection, base_action,
                radius, device,
            )
            shared_adam, adam_rms, adam_lr = calibrated_adam_objective_step(
                actor, payload["optimizer"], global_gradient, radius,
                actor_inputs, selection, base_action, device,
            )
            actions.extend((independent, shared_sgd, shared_adam))
            actual_step[f"independent_{token}"] = float(np.sqrt(np.mean(
                ((independent - base_action) / BASE_SIGMA.reshape(1, 1, 2)) ** 2
            )))
            actual_step[f"shared_sgd_{token}"] = sgd_rms
            actual_step[f"shared_adam_{token}"] = adam_rms
            step_learning_rate[f"independent_{token}"] = 0.0
            step_learning_rate[f"shared_sgd_{token}"] = sgd_lr
            step_learning_rate[f"shared_adam_{token}"] = adam_lr
        actions_np = np.stack(actions, axis=0)
        costs_np = np.stack([
            rollout_bank(
                backend, weights, params, value[:, None], states, current,
                references, selection, args.rollout_batch_size, device,
            )[:, 0]
            for value in actions_np
        ], axis=0).astype(np.float32)
        base_step_cost_error = float(np.max(np.abs(costs_np[0] - global_cost)))
        # The differentiable path and rollout helper use different batch
        # shapes.  DBM float32 replay is deterministic within each path but
        # can differ by a few 1e-3 for the full 600-state evaluation.
        if base_step_cost_error > 1e-2:
            raise AssertionError(
                f"base cost mismatch: max_abs_error={base_step_cost_error}"
            )
        step_summary = {"base": {"cost": distribution(global_cost)}}
        step_cosine = np.ones((len(step_names), len(selection)), np.float32)
        for index, name in enumerate(step_names[1:], start=1):
            record = step_metrics(
                base_action, actions_np[index], global_cost, costs_np[index],
                metrics["action_gradient"], speed,
            )
            record["actual_global_sigma_rms"] = actual_step[name]
            record["calibrated_parameter_learning_rate"] = step_learning_rate[name]
            step_summary[name] = record
            sigma = BASE_SIGMA.reshape(1, 1, 2)
            delta_z = (actions_np[index] - base_action) / sigma
            descent = -metrics["action_gradient"] * sigma
            step_cosine[index] = (
                np.sum(delta_z * descent, axis=(1, 2))
                / np.maximum(
                    np.linalg.norm(delta_z.reshape(len(delta_z), -1), axis=1)
                    * np.linalg.norm(descent.reshape(len(descent), -1), axis=1),
                    1e-12,
                )
            )

        per_seed[str(seed)] = {
            "actor_cost": distribution(global_cost),
            "global_gradient": {
                group: {"norm": gradient_norm(global_gradient, names, group)}
                for group in PARAMETER_GROUPS
            },
            "per_state_conflict": conflict,
            "group_conflict": grouped,
            "steps": step_summary,
            "checks": {
                "actor_cost_rollout_max_abs_error": replay_error,
                "base_step_cost_max_abs_error": base_step_cost_error,
            },
        }
        all_actor_action.append(base_action)
        all_actor_cost.append(global_cost)
        all_action_gradient.append(metrics["action_gradient"])
        all_param_norm.append(metrics["parameter_norm"])
        all_param_cosine.append(metrics["parameter_cosine_global"])
        all_param_dot.append(metrics["parameter_dot_global"])
        all_global_flat.append(flat_gradient(global_gradient))
        all_step_action.append(actions_np)
        all_step_cost.append(costs_np)
        all_step_cosine.append(step_cosine)
        checkpoint_hashes.append(sha256_file(checkpoint))
        print(
            f"seed={seed}: cancellation={conflict['all']['cancellation_ratio']:.4f} "
            f"negative={conflict['all']['negative_alignment_fraction']:.3f} "
            f"adam002_gain={step_summary.get('shared_adam_0p002', {}).get('gain', {}).get('mean', float('nan')):.3f}",
            flush=True,
        )

    actor_action = np.stack(all_actor_action)
    actor_cost = np.stack(all_actor_cost)
    action_gradient = np.stack(all_action_gradient)
    param_norm = np.stack(all_param_norm)
    param_cosine = np.stack(all_param_cosine)
    param_dot = np.stack(all_param_dot)
    global_flat = np.stack(all_global_flat)
    step_action = np.stack(all_step_action)
    step_cost = np.stack(all_step_cost)
    step_cosine = np.stack(all_step_cosine)

    pooled_conflict = {}
    for group_index, group in enumerate(PARAMETER_GROUPS):
        cosine = param_cosine[:, :, group_index].reshape(-1)
        pooled_conflict[group] = {
            "per_state_gradient_norm": distribution(
                param_norm[:, :, group_index].reshape(-1)
            ),
            "cosine_to_seed_global": distribution(cosine),
            "negative_alignment_fraction": float(np.mean(cosine < 0.0)),
            "strong_negative_alignment_fraction": float(np.mean(cosine < -0.25)),
            "seed_cancellation_ratio": [
                per_seed[str(seed)]["per_state_conflict"][group]["cancellation_ratio"]
                for seed in SEEDS
            ],
        }
    pooled_steps = {}
    for index, name in enumerate(step_names):
        if name == "base":
            pooled_steps[name] = {"cost": distribution(step_cost[:, index].reshape(-1))}
            continue
        pooled_steps[name] = step_metrics(
            actor_action.reshape(-1, 8, 2),
            step_action[:, index].reshape(-1, 8, 2),
            actor_cost.reshape(-1), step_cost[:, index].reshape(-1),
            action_gradient.reshape(-1, 8, 2), np.tile(speed, len(SEEDS)),
        )

    all_cancellation = np.asarray(
        [per_seed[str(seed)]["per_state_conflict"]["all"]["cancellation_ratio"] for seed in SEEDS]
    )
    all_negative = np.asarray(
        [per_seed[str(seed)]["per_state_conflict"]["all"]["negative_alignment_fraction"] for seed in SEEDS]
    )
    speed_negative = []
    for seed in SEEDS:
        speed_records = per_seed[str(seed)]["group_conflict"]["speed"]["records"]
        speed_negative.append(min(
            float(record["cosine_to_global"]["all"])
            for record in speed_records.values()
        ))
    conflict_confirmed = bool(
        np.median(all_cancellation) < 0.35
        or np.median(all_negative) > 0.20
        or np.median(speed_negative) < 0.0
    )
    qualification = (
        "SHARED_ACTOR_PARAMETER_GRADIENT_CONFLICT_CONFIRMED"
        if conflict_confirmed else
        "SHARED_ACTOR_PARAMETER_GRADIENT_CONFLICT_NOT_CONFIRMED"
    )

    arrays = {
        "seed": np.asarray(SEEDS, np.int16),
        "state_index": selection.astype(np.int32),
        "episode": episode,
        "speed": speed,
        "scenario": scenario,
        "warm_cost": warm_cost,
        "parameter_group_name": np.asarray(PARAMETER_GROUPS),
        "parameter_name": np.asarray(parameter_names),
        "parameter_numel": np.asarray(parameter_numels, np.int32),
        "parameter_group": np.asarray(parameter_groups),
        "global_parameter_gradient": global_flat.astype(np.float32),
        "actor_action": actor_action.astype(np.float32),
        "actor_cost": actor_cost.astype(np.float32),
        "action_gradient": action_gradient.astype(np.float32),
        "per_state_parameter_norm": param_norm.astype(np.float32),
        "per_state_parameter_cosine_global": param_cosine.astype(np.float32),
        "per_state_parameter_dot_global": param_dot.astype(np.float32),
        "step_name": np.asarray(step_names),
        "step_action": step_action.astype(np.float32),
        "step_cost": step_cost.astype(np.float32),
        "step_action_descent_cosine": step_cosine.astype(np.float32),
    }
    np.savez_compressed(args.output_dir / "evaluation.npz", **arrays)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "scope": (
            "three frozen latest deterministic-center DBM Actors; outer-fold-1 "
            "internal-selection only; exact deterministic raw J50; no training, "
            "Replay generation, formal validation, test, wrapper, or closed loop"
        ),
        "contract": {
            "objective": "per-state and batch-mean deterministic DBM raw J50",
            "warm_role": "reporting stratum only; excluded from gradients",
            "parameter_groups": list(PARAMETER_GROUPS),
            "group_families": ["speed", "scenario", "cost_quartile", "warm_outcome"],
            "step_radii_sigma_rms": list(radii),
            "steps": {
                "independent": "per-state exact action-space standardized steepest descent",
                "shared_sgd": "one shared negative batch-mean parameter-gradient step",
                "shared_adam": "saved Actor AdamW moments with the same batch-mean DBM gradient",
            },
            "registered_conflict_signal": (
                "median all-parameter cancellation ratio <0.35 OR median per-state "
                "negative alignment fraction >0.20 OR median worst-speed group cosine <0"
            ),
            "historical_0_9326_role": (
                "cross-fold/cross-contract capacity reference only; not an audit pass gate"
            ),
        },
        "checks": {
            "state_count": int(len(selection)),
            "episode_count": int(len(selection_episodes)),
            "seed_count": len(SEEDS),
            "formal_validation_loaded": False,
            "test_loaded": False,
            "actor_cost_rollout_max_abs_error": max(
                value["checks"]["actor_cost_rollout_max_abs_error"]
                for value in per_seed.values()
            ),
            "finite": bool(all(np.all(np.isfinite(value)) for value in arrays.values() if np.issubdtype(value.dtype, np.number))),
        },
        "manifest": {
            "run": str(args.run.resolve()),
            "run_contract_sha256": sha256_file(args.run / "contract.json"),
            "run_summary_sha256": sha256_file(args.run / "summary.json"),
            "checkpoint_sha256": checkpoint_hashes,
            "bank_root": str(Path(run_args["bank_root"]).resolve()),
            "gt_v1": str(Path(run_args["gt_v1"]).resolve()),
            "normalization_source": str(normalization_path.resolve()),
            "selection_index_sha256": sha256_array(selection.astype(np.int32)),
            "array_sha256": {key: sha256_array(value) for key, value in arrays.items()},
        },
        "per_seed": per_seed,
        "pooled": {
            "conflict": pooled_conflict,
            "steps": pooled_steps,
            "seed_all_cancellation_ratio": all_cancellation.tolist(),
            "seed_all_negative_alignment_fraction": all_negative.tolist(),
            "seed_worst_speed_group_cosine": speed_negative,
        },
        "decision": {
            "conflict_confirmed": conflict_confirmed,
            "next_if_confirmed": (
                "Run one exact-DBM grouped gradient-surgery A/B (plain mean versus "
                "speed/cost-stratified PCGrad or CAGrad) before changing Actor capacity."
            ),
            "next_if_not_confirmed": (
                "Run the same-fold fixed-state direct-task-loss capacity calibration; "
                "then inspect Actor Jacobian conditioning/optimizer trajectory."
            ),
            "exploration_bank": (
                "Not a causal explanation for this audit because the deterministic DBM "
                "Actor gradient is evaluated directly at the Actor center."
            ),
        },
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "qualification": qualification,
        "seed_cancellation_ratio": all_cancellation.tolist(),
        "seed_negative_alignment_fraction": all_negative.tolist(),
        "seed_worst_speed_group_cosine": speed_negative,
        "pooled_steps": {
            key: value.get("gain", {}).get("mean") for key, value in pooled_steps.items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
