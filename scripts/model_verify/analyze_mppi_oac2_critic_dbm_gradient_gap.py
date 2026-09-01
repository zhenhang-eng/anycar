#!/usr/bin/env python3
"""Audit the current online OAC Critic against exact differentiable DBM feedback.

The audit is intentionally read-only.  On the OAC internal-selection episodes it
compares, at identical Actor actions:

* exact d log1p(J50) / d absolute knots;
* each Twin Value gradient, their mean, and the conservative max branch;
* actual DBM improvement after matched fixed-trust and magnitude-aware steps;
* Actor-parameter gradients for mean J50, mean log1p(J50), and Twin Critic value.

Formal validation/test are not loaded and no network is updated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import TorchDBMParams, TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from run_mppi_absolute_action_value_critic_cv import AbsoluteActionValueCritic, make_folds
from train_mppi_online_absolute_sac import (
    BASE_SIGMA,
    critic_state_inputs,
    load_actor_normalization,
    load_bank,
    make_actor_inputs,
    stratified_contexts,
)
from train_mppi_oac2_continuous_actor import internal_split, physical_value
from validate_mppi_oac2_continuous_actor import load_actor_checkpoint, load_critic_checkpoint


DEFAULT_RUN = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_box3_adaptive_tail_200round_20260825_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/oac2_critic_dbm_gradient_gap_20260825_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--roles", default="selected,latest")
    parser.add_argument("--radii", default="0.002,0.005,0.01,0.02")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--parameter-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(value: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(value, np.float64)
    return {
        "count": int(len(value)),
        "mean": float(np.mean(value)),
        "p05": float(np.quantile(value, 0.05)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
        "p95": float(np.quantile(value, 0.95)),
        "minimum": float(np.min(value)),
        "maximum": float(np.max(value)),
    }


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.reshape(len(left), -1).astype(np.float64)
    right = right.reshape(len(right), -1).astype(np.float64)
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def gradient_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    flat_pred = predicted.reshape(len(predicted), -1)
    flat_true = truth.reshape(len(truth), -1)
    cosine = cosine_rows(flat_pred, flat_true)
    pred_norm = np.linalg.norm(flat_pred, axis=1)
    true_norm = np.linalg.norm(flat_true, axis=1)
    ratio = pred_norm / np.maximum(true_norm, 1e-12)
    component_groups = {
        "all": np.arange(16),
        "acceleration": np.arange(0, 16, 2),
        "steering": np.arange(1, 16, 2),
        "early_steering_0_2": np.asarray((1, 3, 5)),
        "late_steering_3_7": np.asarray((7, 9, 11, 13, 15)),
    }
    return {
        "cosine": distribution(cosine),
        "cosine_positive_fraction": float(np.mean(cosine > 0)),
        "cosine_above_0_5_fraction": float(np.mean(cosine >= 0.5)),
        "norm_ratio": distribution(ratio),
        "predicted_norm": distribution(pred_norm),
        "true_norm": distribution(true_norm),
        "component_cosine": {
            name: distribution(cosine_rows(flat_pred[:, index], flat_true[:, index]))
            for name, index in component_groups.items()
        },
    }


def load_initial_actor(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device)
    from mppi_a2_actors import DirectNoAnchorGTXSupportActor
    actor = DirectNoAnchorGTXSupportActor(support_multiplier=3.0, dropout=0.0).to(device)
    incompatible = actor.load_state_dict(payload["model_state_dict"], strict=False)
    expected = {
        "output_support_multiplier",
        "support_adapter_head.weight", "support_adapter_head.bias",
        "support_adapter_skip.weight", "support_adapter_skip.bias",
    }
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise AssertionError(f"unexpected initial Actor load: {incompatible}")
    actor.eval()
    return actor


def actor_actions(actor, actor_inputs, indices: np.ndarray, device, batch_size=256):
    result = []
    actor.eval()
    with torch.no_grad():
        for begin in range(0, len(indices), batch_size):
            local = indices[begin:begin + batch_size]
            tensors = tuple(torch.from_numpy(value[local]).to(device) for value in actor_inputs)
            _, center = actor(*tensors)
            result.append(center.cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def exact_cost_gradient(actions, indices, states, current, reference, backend,
                        weights, params, device, batch_size):
    costs, gradients = [], []
    for begin in range(0, len(indices), batch_size):
        local_indices = indices[begin:begin + batch_size]
        knots = torch.from_numpy(actions[begin:begin + batch_size]).to(device).requires_grad_(True)
        interpolated = interpolate_knots(knots, params.horizon).unsqueeze(1)
        cost = batched_cost(
            backend, weights, interpolated,
            torch.from_numpy(states[local_indices]).to(device),
            torch.from_numpy(current[local_indices]).to(device),
            torch.from_numpy(reference[local_indices]).to(device),
        )[:, 0]
        gradient = torch.autograd.grad(torch.log1p(cost).sum(), knots)[0]
        costs.append(cost.detach().cpu().numpy())
        gradients.append(gradient.detach().cpu().numpy())
    return np.concatenate(costs).astype(np.float32), np.concatenate(gradients).astype(np.float32)


def critic_cost_gradient(critic1, payload1, critic2, payload2, inputs, actions,
                         indices, device, batch_size):
    output = {key: [] for key in (
        "q1", "q2", "g1", "g2", "g_mean", "g_conservative", "selected_twin"
    )}
    critic1.eval(); critic2.eval()
    for begin in range(0, len(indices), batch_size):
        local_indices = indices[begin:begin + batch_size]
        action = torch.from_numpy(actions[begin:begin + batch_size, None]).to(device).requires_grad_(True)
        q1 = physical_value(critic1, payload1, inputs, local_indices, action, device)[:, 0]
        q2 = physical_value(critic2, payload2, inputs, local_indices, action, device)[:, 0]
        g1 = torch.autograd.grad(q1.sum(), action, retain_graph=True)[0][:, 0]
        g2 = torch.autograd.grad(q2.sum(), action, retain_graph=True)[0][:, 0]
        q_mean = 0.5 * (q1 + q2)
        g_mean = torch.autograd.grad(q_mean.sum(), action, retain_graph=True)[0][:, 0]
        conservative = torch.maximum(q1, q2)
        g_conservative = torch.autograd.grad(conservative.sum(), action)[0][:, 0]
        output["q1"].append(q1.detach().cpu().numpy())
        output["q2"].append(q2.detach().cpu().numpy())
        output["g1"].append(g1.detach().cpu().numpy())
        output["g2"].append(g2.detach().cpu().numpy())
        output["g_mean"].append(g_mean.detach().cpu().numpy())
        output["g_conservative"].append(g_conservative.detach().cpu().numpy())
        output["selected_twin"].append((q2 > q1).detach().cpu().numpy().astype(np.int8) + 1)
    return {key: np.concatenate(value) for key, value in output.items()}


def trust_step(action: np.ndarray, gradient: np.ndarray, radius: float) -> np.ndarray:
    sigma = BASE_SIGMA.reshape(1, 1, 2).astype(np.float64)
    metric_gradient = gradient.astype(np.float64) * sigma
    rms = np.sqrt(np.mean(metric_gradient ** 2, axis=(1, 2), keepdims=True))
    normalized = metric_gradient / np.maximum(rms, 1e-12)
    delta = -float(radius) * sigma * normalized
    delta[rms.reshape(-1) < 1e-10] = 0.0
    return np.clip(action.astype(np.float64) + delta, -1.0, 1.0).astype(np.float32)


def global_magnitude_step(action: np.ndarray, gradient: np.ndarray, eta: float) -> np.ndarray:
    return np.clip(action.astype(np.float64) - eta * gradient.astype(np.float64), -1.0, 1.0).astype(np.float32)


def evaluate_action_bank(action_bank, indices, states, current, reference, backend,
                         weights, params, device, batch_size):
    values = []
    for begin in range(0, len(indices), batch_size):
        local_indices = indices[begin:begin + batch_size]
        knots = torch.from_numpy(action_bank[begin:begin + batch_size]).to(device)
        cost = batched_cost(
            backend, weights, interpolate_knots(knots, params.horizon),
            torch.from_numpy(states[local_indices]).to(device),
            torch.from_numpy(current[local_indices]).to(device),
            torch.from_numpy(reference[local_indices]).to(device),
        )
        values.append(cost.detach().cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def step_metrics(base: np.ndarray, stepped: np.ndarray, mask=None):
    if mask is None:
        mask = np.ones(len(base), dtype=bool)
    gain = base[mask] - stepped[mask]
    return {
        "cost": distribution(stepped[mask]),
        "gain": distribution(gain),
        "improved_fraction": float(np.mean(gain > 0)),
        "regressed_fraction": float(np.mean(gain < 0)),
    }


def parameter_vector(actor, objective: str, actor_inputs, critic_inputs, indices,
                     states, current, reference, backend, weights, params,
                     critic1, payload1, critic2, payload2, device):
    actor.zero_grad(set_to_none=True)
    tensors = tuple(torch.from_numpy(value[indices]).to(device) for value in actor_inputs)
    _, action = actor(*tensors)
    if objective in ("dbm_raw", "dbm_log"):
        cost = batched_cost(
            backend, weights, interpolate_knots(action, params.horizon).unsqueeze(1),
            torch.from_numpy(states[indices]).to(device),
            torch.from_numpy(current[indices]).to(device),
            torch.from_numpy(reference[indices]).to(device),
        )[:, 0]
        loss = cost.mean() if objective == "dbm_raw" else torch.log1p(cost).mean()
    elif objective == "critic":
        q1 = physical_value(critic1, payload1, critic_inputs, indices, action[:, None], device)[:, 0]
        q2 = physical_value(critic2, payload2, critic_inputs, indices, action[:, None], device)[:, 0]
        loss = torch.maximum(q1, q2).mean()
    else:
        raise ValueError(objective)
    named = [(name, parameter) for name, parameter in actor.named_parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(loss, [parameter for _, parameter in named], allow_unused=True)
    groups: dict[str, list[np.ndarray]] = {"all": [], "encoder": [], "decoder": [], "support_adapter": []}
    for (name, parameter), gradient in zip(named, gradients):
        value = torch.zeros_like(parameter) if gradient is None else gradient
        array = value.detach().reshape(-1).cpu().numpy().astype(np.float64)
        groups["all"].append(array)
        if name.startswith("encoder."):
            groups["encoder"].append(array)
        elif name.startswith("decoder."):
            groups["decoder"].append(array)
        elif name.startswith("support_adapter"):
            groups["support_adapter"].append(array)
    return {key: np.concatenate(value) for key, value in groups.items()}, float(loss.detach())


def named_parameter_gradient(actor, objective: str, actor_inputs, critic_inputs,
                             indices, states, current, reference, backend, weights,
                             params, critic1, payload1, critic2, payload2, device):
    """Return Actor gradients in optimizer parameter order for a single objective."""
    actor.zero_grad(set_to_none=True)
    tensors = tuple(torch.from_numpy(value[indices]).to(device) for value in actor_inputs)
    _, action = actor(*tensors)
    if objective in ("dbm_raw", "dbm_log"):
        cost = batched_cost(
            backend, weights, interpolate_knots(action, params.horizon).unsqueeze(1),
            torch.from_numpy(states[indices]).to(device),
            torch.from_numpy(current[indices]).to(device),
            torch.from_numpy(reference[indices]).to(device),
        )[:, 0]
        loss = cost.mean() if objective == "dbm_raw" else torch.log1p(cost).mean()
    elif objective == "critic":
        q1 = physical_value(critic1, payload1, critic_inputs, indices, action[:, None], device)[:, 0]
        q2 = physical_value(critic2, payload2, critic_inputs, indices, action[:, None], device)[:, 0]
        loss = torch.maximum(q1, q2).mean()
    else:
        raise ValueError(objective)
    parameters = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return [
        (torch.zeros_like(parameter) if gradient is None else gradient).detach().clone()
        for parameter, gradient in zip(parameters, gradients)
    ]


def adam_objective_step(actor, actor_optimizer_state, gradients, learning_rate,
                        actor_inputs, selection, base_actions, device):
    """Apply one AdamW step using the saved OAC moments and report output RMS."""
    candidate = copy.deepcopy(actor).to(device)
    dummy_scale = torch.nn.Parameter(torch.zeros(8, 2, device=device))
    optimizer = torch.optim.AdamW(
        list(candidate.parameters()) + [dummy_scale], lr=float(learning_rate),
        weight_decay=1e-6,
    )
    optimizer.load_state_dict(copy.deepcopy(actor_optimizer_state))
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
    for parameter, gradient in zip(candidate.parameters(), gradients):
        parameter.grad = gradient.to(device).clone()
    dummy_scale.grad = torch.zeros_like(dummy_scale)
    optimizer.step()
    updated = actor_actions(candidate, actor_inputs, selection, device)
    sigma = BASE_SIGMA.reshape(1, 1, 2)
    rms = float(np.sqrt(np.mean(((updated - base_actions) / sigma) ** 2)))
    return updated, rms


def calibrated_adam_objective_step(actor, actor_optimizer_state, gradients,
                                   target_rms, actor_inputs, selection,
                                   base_actions, device):
    learning_rate = 1e-6 * float(target_rms) / 0.00025
    updated = base_actions
    actual = 0.0
    for _ in range(5):
        updated, actual = adam_objective_step(
            actor, actor_optimizer_state, gradients, learning_rate,
            actor_inputs, selection, base_actions, device,
        )
        if actual <= 1e-12:
            break
        ratio = float(target_rms) / actual
        if abs(ratio - 1.0) <= 0.01:
            break
        learning_rate *= float(np.clip(ratio, 0.2, 5.0))
    return updated, actual, learning_rate


def vector_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_norm = float(np.linalg.norm(left)); right_norm = float(np.linalg.norm(right))
    return {
        "cosine": float(np.dot(left, right) / max(left_norm * right_norm, 1e-12)),
        "norm_ratio": left_norm / max(right_norm, 1e-12),
        "left_norm": left_norm,
        "right_norm": right_norm,
    }


def stored_metric_error(recomputed: np.ndarray, stored: dict) -> float:
    pairs = (
        (float(np.mean(recomputed)), float(stored["mean"])),
        (float(np.median(recomputed)), float(stored["median"])),
        (float(np.max(recomputed)), float(stored["maximum"])),
    )
    return max(abs(left - right) for left, right in pairs)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    roles = tuple(value.strip() for value in args.roles.split(",") if value.strip())
    if not set(roles) <= {"initial", "selected", "latest"}:
        raise ValueError(roles)
    radii = tuple(float(value) for value in args.radii.split(","))
    contract = json.loads((args.run_dir / "contract.json").read_text())
    source_summary = json.loads((args.run_dir / "summary.json").read_text())
    run_args = contract["arguments"]
    bank_root = Path(run_args["bank_root"])
    gt_v1 = Path(run_args["gt_v1"])
    base_ac = Path(run_args["base_ac"])
    data = load_bank(bank_root)
    folds = make_folds(data, 3)
    fit, selection, fit_episodes, selection_episodes = internal_split(
        data, folds, int(contract["outer_fold"])
    )
    states, current, reference, params_json, weights_json, dbm_json = load_rollout_inputs(data, gt_v1)
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    actor_normalization, _ = load_actor_normalization(base_ac)
    actor_inputs = make_actor_inputs(data, actor_normalization)
    seeds = [int(value) for value in str(run_args["seeds"]).split(",")]
    source_by_seed = {int(row["seed"]): row for row in source_summary["records"]}
    artifact: dict[str, np.ndarray] = {
        "selection_indices": selection.astype(np.int64),
        "speed": data["speed"][selection],
        "scenario": data["scenario"][selection],
        "episode": data["episode"][selection],
    }
    records = []
    max_source_metric_error = 0.0
    for seed in seeds:
        seed_dir = args.run_dir / f"seed_{seed}"
        critic1, payload1 = load_critic_checkpoint(seed_dir / "critic1.pt", device)
        critic2, payload2 = load_critic_checkpoint(seed_dir / "critic2.pt", device)
        if payload1["training"]["normalization"] != payload2["training"]["normalization"]:
            raise AssertionError("Twin Critic normalization mismatch")
        critic_inputs = critic_state_inputs(data, payload1)
        role_actors = {}
        role_payloads = {}
        if "initial" in roles:
            role_actors["initial"] = load_initial_actor(
                Path(source_by_seed[seed]["initial_actor_checkpoint"]), device
            )
        if "selected" in roles:
            role_actors["selected"], role_payloads["selected"] = load_actor_checkpoint(seed_dir / "actor_selected.pt", device)
        if "latest" in roles:
            role_actors["latest"], role_payloads["latest"] = load_actor_checkpoint(seed_dir / "actor_latest.pt", device)
        role_records = {}
        actions_by_role = {}
        cost_by_role = {}
        for role, actor in role_actors.items():
            actions = actor_actions(actor, actor_inputs, selection, device)
            true_cost, true_gradient = exact_cost_gradient(
                actions, selection, states, current, reference, backend, weights,
                params, device, args.batch_size,
            )
            actions_by_role[role] = actions
            cost_by_role[role] = true_cost
            critic = critic_cost_gradient(
                critic1, payload1, critic2, payload2, critic_inputs, actions,
                selection, device, args.batch_size,
            )
            true_log = np.log1p(true_cost)
            q1, q2 = critic["q1"], critic["q2"]
            q_conservative = np.maximum(q1, q2)
            candidate_best = data["costs"][selection].min(axis=1)
            headroom = true_cost - candidate_best
            true_norm = np.linalg.norm(true_gradient.reshape(len(selection), -1), axis=1)
            flat_threshold = float(np.quantile(true_norm, 0.25))
            masks = {
                "all": np.ones(len(selection), dtype=bool),
                "flat_true_gradient_q25": true_norm <= flat_threshold,
                "nonflat_true_gradient": true_norm > flat_threshold,
                "near_bank_best_gap_le_0_1": headroom <= 0.1,
                "headroom_gt_1": headroom > 1.0,
            }
            gradients = {
                "twin1": critic["g1"],
                "twin2": critic["g2"],
                "twin_mean": critic["g_mean"],
                "twin_conservative": critic["g_conservative"],
            }
            gradient_record = {name: gradient_metrics(value, true_gradient) for name, value in gradients.items()}
            gradient_record["twin_conservative_by_speed"] = {
                f"{speed:.1f}": gradient_metrics(
                    critic["g_conservative"][np.isclose(data["speed"][selection], speed)],
                    true_gradient[np.isclose(data["speed"][selection], speed)],
                ) for speed in sorted(np.unique(data["speed"][selection]))
            }
            gradient_record["twin_conservative_by_regime"] = {
                name: gradient_metrics(critic["g_conservative"][mask], true_gradient[mask])
                for name, mask in masks.items() if np.any(mask)
            }
            step_bank = [actions]
            step_names = ["base"]
            for radius in radii:
                step_bank.append(trust_step(actions, true_gradient, radius))
                step_names.append(f"true_trust_{radius:g}")
                step_bank.append(trust_step(actions, critic["g_conservative"], radius))
                step_names.append(f"critic_trust_{radius:g}")
            radius = radii[-1]
            for name in ("g1", "g2", "g_mean"):
                step_bank.append(trust_step(actions, critic[name], radius))
                step_names.append(f"{name}_trust_{radius:g}")
            sigma = BASE_SIGMA.reshape(1, 1, 2)
            true_rms = np.sqrt(np.mean((true_gradient / sigma) ** 2, axis=(1, 2)))
            eta = float(radii[-1] / max(float(np.median(true_rms)), 1e-12))
            step_bank.append(global_magnitude_step(actions, true_gradient, eta))
            step_names.append("true_global_eta")
            step_bank.append(global_magnitude_step(actions, critic["g_conservative"], eta))
            step_names.append("critic_global_eta")
            stacked = np.stack(step_bank, axis=1)
            step_cost = evaluate_action_bank(
                stacked, selection, states, current, reference, backend, weights,
                params, device, args.batch_size,
            )
            step_record = {
                name: {
                    "all": step_metrics(true_cost, step_cost[:, index]),
                    "flat_true_gradient_q25": step_metrics(true_cost, step_cost[:, index], masks["flat_true_gradient_q25"]),
                    "nonflat_true_gradient": step_metrics(true_cost, step_cost[:, index], masks["nonflat_true_gradient"]),
                    "by_speed": {
                        f"{speed:.1f}": step_metrics(
                            true_cost, step_cost[:, index],
                            np.isclose(data["speed"][selection], speed),
                        ) for speed in sorted(np.unique(data["speed"][selection]))
                    },
                } for index, name in enumerate(step_names) if name != "base"
            }
            expected = source_by_seed[seed][f"{role}_metrics"]["cost"]
            source_error = stored_metric_error(true_cost, expected)
            max_source_metric_error = max(max_source_metric_error, source_error)
            role_records[role] = {
                "source_cost_metric_max_abs_error": source_error,
                "true_cost": distribution(true_cost),
                "bank_headroom": distribution(headroom),
                "flat_true_gradient_threshold": flat_threshold,
                "value": {
                    "twin1_pearson_log": correlation(q1, true_log),
                    "twin2_pearson_log": correlation(q2, true_log),
                    "conservative_pearson_log": correlation(q_conservative, true_log),
                    "twin1_mae_log": float(np.mean(np.abs(q1 - true_log))),
                    "twin2_mae_log": float(np.mean(np.abs(q2 - true_log))),
                    "conservative_mae_log": float(np.mean(np.abs(q_conservative - true_log))),
                    "twin_disagreement": distribution(np.abs(q1 - q2)),
                },
                "gradient": gradient_record,
                "steps": step_record,
                "global_eta": eta,
            }
            prefix = f"seed{seed}_{role}"
            artifact[f"{prefix}_actions"] = actions
            artifact[f"{prefix}_true_cost"] = true_cost
            artifact[f"{prefix}_true_gradient_log"] = true_gradient
            artifact[f"{prefix}_q1"] = q1
            artifact[f"{prefix}_q2"] = q2
            artifact[f"{prefix}_g1"] = critic["g1"]
            artifact[f"{prefix}_g2"] = critic["g2"]
            artifact[f"{prefix}_g_conservative"] = critic["g_conservative"]
            artifact[f"{prefix}_step_names"] = np.asarray(step_names)
            artifact[f"{prefix}_step_cost"] = step_cost

        # Actor parameter-gradient audit on one deterministic, stratified heldout batch.
        param_indices = stratified_contexts(
            data, selection, min(args.parameter_batch_size, len(selection)),
            np.random.default_rng(260825800 + seed),
        )
        parameter_records = {}
        for role, actor in role_actors.items():
            objectives = {}
            losses = {}
            for objective in ("dbm_raw", "dbm_log", "critic"):
                objectives[objective], losses[objective] = parameter_vector(
                    actor, objective, actor_inputs, critic_inputs, param_indices,
                    states, current, reference, backend, weights, params,
                    critic1, payload1, critic2, payload2, device,
                )
            parameter_records[role] = {
                "count": int(len(param_indices)),
                "losses": losses,
                "critic_vs_dbm_log": {
                    group: vector_comparison(objectives["critic"][group], objectives["dbm_log"][group])
                    for group in objectives["critic"]
                },
                "dbm_log_vs_dbm_raw": {
                    group: vector_comparison(objectives["dbm_log"][group], objectives["dbm_raw"][group])
                    for group in objectives["critic"]
                },
                "critic_vs_dbm_raw": {
                    group: vector_comparison(objectives["critic"][group], objectives["dbm_raw"][group])
                    for group in objectives["critic"]
                },
            }
        # Directly compare one current-Adam step from identical latest Actor
        # parameters and moments.  Output RMS is calibrated, so this isolates
        # objective/gradient direction rather than nominal learning-rate scale.
        parameter_step_record = None
        if "latest" in role_actors and "optimizer" in role_payloads["latest"]:
            actor = role_actors["latest"]
            gradients = {
                objective: named_parameter_gradient(
                    actor, objective, actor_inputs, critic_inputs, param_indices,
                    states, current, reference, backend, weights, params,
                    critic1, payload1, critic2, payload2, device,
                ) for objective in ("dbm_raw", "dbm_log", "critic")
            }
            candidates = [actions_by_role["latest"]]
            candidate_names = ["base"]
            calibration = {}
            for target_rms in (0.00025, 0.001, 0.002):
                for objective in ("dbm_raw", "dbm_log", "critic"):
                    updated, actual_rms, learning_rate = calibrated_adam_objective_step(
                        actor, role_payloads["latest"]["optimizer"], gradients[objective],
                        target_rms, actor_inputs, selection, actions_by_role["latest"], device,
                    )
                    name = f"{objective}_rms_{target_rms:g}"
                    candidates.append(updated)
                    candidate_names.append(name)
                    calibration[name] = {
                        "target_output_sigma_rms": target_rms,
                        "actual_output_sigma_rms": actual_rms,
                        "calibrated_learning_rate": learning_rate,
                    }
            candidate_cost = evaluate_action_bank(
                np.stack(candidates, axis=1), selection, states, current,
                reference, backend, weights, params, device, args.batch_size,
            )
            parameter_step_record = {
                "gradient_batch_count": int(len(param_indices)),
                "optimizer": "saved current OAC AdamW moments; one step; learning rate calibrated to output RMS",
                "calibration": calibration,
                "metrics": {
                    name: step_metrics(cost_by_role["latest"], candidate_cost[:, index])
                    for index, name in enumerate(candidate_names) if name != "base"
                },
            }
            artifact[f"seed{seed}_parameter_step_names"] = np.asarray(candidate_names)
            artifact[f"seed{seed}_parameter_step_actions"] = np.stack(candidates, axis=1)
            artifact[f"seed{seed}_parameter_step_cost"] = candidate_cost
        records.append({
            "seed": seed,
            "roles": role_records,
            "parameter_gradient": parameter_records,
            "parameter_step": parameter_step_record,
        })
    artifact_path = args.output_dir / "audit.npz"
    np.savez_compressed(artifact_path, **artifact)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "OAC2_CRITIC_DBM_GRADIENT_GAP_AUDIT_COMPLETE",
        "contract": {
            "run_dir": str(args.run_dir.resolve()),
            "run_summary_sha256": sha256_file(args.run_dir / "summary.json"),
            "run_validator_sha256": sha256_file(args.run_dir / "validator_report.json"),
            "split": "fold-1 internal-selection episodes; absent from online Replay",
            "fit_episodes": fit_episodes,
            "selection_episodes": selection_episodes,
            "state_count": int(len(selection)),
            "roles": list(roles),
            "coordinate": "d log1p(J50) / d physical absolute 8x2 knots",
            "fixed_trust_step": "steepest descent under RMS(delta/sigma)=radius",
            "parameter_objectives": ["mean J50", "mean log1p(J50)", "mean conservative Twin log1p(J)"],
            "formal_validation_loaded": False,
            "test_loaded": False,
            "networks_updated": False,
        },
        "dbm_source_cost_metric_max_abs_error": max_source_metric_error,
        "records": records,
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["qualification"])
    print("source metric error", max_source_metric_error)
    for row in records:
        for role in roles:
            value = row["roles"][role]
            gradient = value["gradient"]["twin_conservative"]
            step = value["steps"][f"critic_trust_{radii[-1]:g}"]["all"]
            parameter = row["parameter_gradient"][role]
            print({
                "seed": row["seed"], "role": role,
                "value_corr": round(value["value"]["conservative_pearson_log"], 3),
                "action_cos_med": round(gradient["cosine"]["median"], 3),
                "action_cos_p10": round(gradient["cosine"]["p10"], 3),
                "norm_med": round(gradient["norm_ratio"]["median"], 3),
                "step_gain_med": round(step["gain"]["median"], 3),
                "step_regression": round(step["regressed_fraction"], 3),
                "param_critic_vs_log": round(parameter["critic_vs_dbm_log"]["all"]["cosine"], 3),
                "param_log_vs_raw": round(parameter["dbm_log_vs_dbm_raw"]["all"]["cosine"], 3),
                "param_critic_vs_raw": round(parameter["critic_vs_dbm_raw"]["all"]["cosine"], 3),
            })


if __name__ == "__main__":
    main()
