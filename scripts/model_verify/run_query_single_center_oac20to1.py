#!/usr/bin/env python3
"""Run the train-only strict-no-anchor Query OAC 20-Critic:1-Actor pilot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
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
from car_foundation.mppi_proposal_policy import MPPIProposalNormalization  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from mppi_a2_actors import DirectNoAnchorGTXActor  # noqa: E402
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic  # noqa: E402
from run_query_forward_response_landscape_pilot import (  # noqa: E402
    basis_bank,
    evaluate,
    fit_response,
)


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_single_center_oac20to1_config_20260902_v1.json"


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float64).reshape(-1)
    right = np.asarray(right, np.float64).reshape(-1)
    if len(left) < 2 or left.std() < 1e-12 or right.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def load_inputs(data: dict[str, np.ndarray], normalization: dict) -> tuple[np.ndarray, ...]:
    normalizer = MPPIProposalNormalization.from_dict(normalization)
    history, reference, current = normalizer.normalize_numpy(
        data["history"], data["critic_reference"], data["critic_current"]
    )
    count = len(history)
    return (
        history.astype(np.float32),
        reference.astype(np.float32),
        current.astype(np.float32),
        np.zeros((count, 8, 2), np.float32),
        np.zeros((count, 74), np.float32),
        np.zeros((count, 32), np.float32),
    )


def actor_from_payload(payload: dict, state_key: str, device: torch.device) -> DirectNoAnchorGTXActor:
    training = payload["actor_training"]
    actor = DirectNoAnchorGTXActor(
        dropout=0.0,
        center=torch.tensor(training["out_center"], dtype=torch.float32, device=device),
        scale=torch.tensor(training["out_scale"], dtype=torch.float32, device=device),
    ).to(device)
    actor.load_state_dict(payload[state_key], strict=True)
    return actor


def actor_predict(
    actor: torch.nn.Module,
    inputs: tuple[np.ndarray, ...],
    rows: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    actor.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            local = rows[start : start + batch_size]
            tensors = tuple(torch.from_numpy(value[local]).to(device) for value in inputs)
            output.append(actor(*tensors)[1].cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def actor_tensor(
    actor: torch.nn.Module,
    inputs: tuple[np.ndarray, ...],
    rows: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    tensors = tuple(torch.from_numpy(value[rows]).to(device) for value in inputs)
    return actor(*tensors)[1]


def direct_cost(
    controller: TorchMPPIController,
    data: dict[str, np.ndarray],
    rows: np.ndarray,
    knots: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    output = np.empty(len(rows), np.float32)
    for position, row in enumerate(rows):
        output[position] = evaluate(
            controller, data, int(row), knots[position : position + 1], weights
        )[0][0]
    return output


def metrics(
    cost: np.ndarray,
    initial: np.ndarray,
    warm: np.ndarray,
    speed: np.ndarray,
) -> dict[str, Any]:
    gain = initial.astype(np.float64) - cost.astype(np.float64)
    warm_gain = warm.astype(np.float64) - cost.astype(np.float64)
    by_speed = {}
    for value in sorted(np.unique(speed).tolist()):
        mask = speed == value
        by_speed[str(int(value))] = {
            "count": int(mask.sum()),
            "gain_vs_round0": distribution(gain[mask]),
            "gain_vs_warm": distribution(warm_gain[mask]),
        }
    return {
        "cost": distribution(cost),
        "gain_vs_round0": distribution(gain),
        "gain_vs_warm": distribution(warm_gain),
        "regression_vs_round0_fraction": float(np.mean(gain < -1e-5)),
        "regression_vs_warm_fraction": float(np.mean(warm_gain < -1e-5)),
        "by_speed_kph": by_speed,
    }


class StratifiedQueues:
    def __init__(self, data: dict[str, np.ndarray], fit: np.ndarray, rng: np.random.Generator) -> None:
        self.rng = rng
        self.values: dict[tuple[int, int], np.ndarray] = {}
        self.cursor: dict[tuple[int, int], int] = {}
        for speed in sorted(np.unique(data["speed_index"][fit]).tolist()):
            for variant in sorted(np.unique(data["variant_index"][fit]).tolist()):
                key = (int(speed), int(variant))
                rows = fit[
                    (data["speed_index"][fit] == speed)
                    & (data["variant_index"][fit] == variant)
                ]
                if len(rows) == 0:
                    raise AssertionError(f"empty fit stratum {key}")
                self.values[key] = self.rng.permutation(rows)
                self.cursor[key] = 0

    def take(self) -> np.ndarray:
        output = []
        for key in sorted(self.values):
            if self.cursor[key] == len(self.values[key]):
                self.values[key] = self.rng.permutation(self.values[key])
                self.cursor[key] = 0
            output.append(int(self.values[key][self.cursor[key]]))
            self.cursor[key] += 1
        return np.asarray(output, np.int64)


def response_bank(
    controller: TorchMPPIController,
    data: dict[str, np.ndarray],
    row: int,
    center: np.ndarray,
    radius: float,
    basis: np.ndarray,
    sigma: np.ndarray,
    weights: dict[str, float],
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center_cost, center_residual = evaluate(
        controller, data, row, center[None], weights
    )
    directions = basis.reshape(16, 8, 2)
    raw_probe = np.stack(
        [center + sign * radius * direction * sigma for direction in directions for sign in (1.0, -1.0)]
    ).astype(np.float32)
    probes = np.clip(raw_probe, -1.0, 1.0).astype(np.float32)
    probe_cost, probe_residual = evaluate(controller, data, row, probes, weights)
    fitted = fit_response(
        center,
        float(center_cost[0]),
        center_residual[0],
        probes,
        probe_cost,
        probe_residual,
        sigma,
        float(config["pilot"]["response_fit_ridge"]),
        float(config["pilot"]["response_gauss_newton_damping"]),
        radius,
    )
    line = config["pilot"]["response_line_factors"]
    steps = []
    for name in ("cost_direction", "trajectory_direction", "blend_direction"):
        base = fitted["gn_step"] if name == "trajectory_direction" else radius * fitted[name]
        for factor in line:
            steps.append(float(factor) * base)
    steps = np.stack(steps).reshape(6, 8, 2).astype(np.float32)
    raw_proposal = center[None] + steps * sigma
    proposals = np.clip(raw_proposal, -1.0, 1.0).astype(np.float32)
    proposal_cost, _ = evaluate(controller, data, row, proposals, weights)
    actions = np.concatenate((center[None], probes, proposals)).astype(np.float32)
    costs = np.concatenate((center_cost, probe_cost, proposal_cost)).astype(np.float32)
    raw = np.concatenate((center[None], raw_probe, raw_proposal)).astype(np.float32)
    clipped = np.any(np.abs(raw - actions) > 1e-7, axis=(1, 2))
    return actions, costs, raw, clipped


def combined_candidates(
    data: dict[str, np.ndarray],
    row: int,
    online_state: np.ndarray,
    online_action: np.ndarray,
    online_cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = data["candidate_valid_mask"][row]
    actions = data["candidate_knots"][row, valid]
    costs = data["candidate_cost"][row, valid]
    source = np.zeros(len(actions), np.int8)
    positions = np.flatnonzero(online_state == row)
    if len(positions):
        actions = np.concatenate((actions, online_action[positions]))
        costs = np.concatenate((costs, online_cost[positions]))
        source = np.concatenate((source, np.ones(len(positions), np.int8)))
    return actions, costs, source


def select_training_candidates(
    data: dict[str, np.ndarray],
    rows: np.ndarray,
    online: dict[str, np.ndarray],
    count: int,
    maximum_online: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    batch_actions = np.empty((len(rows), count, 8, 2), np.float32)
    batch_cost = np.empty((len(rows), count), np.float32)
    for position, row in enumerate(rows):
        actions, costs, source = combined_candidates(
            data, int(row), online["state_index"], online["action"], online["cost"]
        )
        chosen = [0, int(np.argmin(costs))]
        online_indices = np.flatnonzero(source == 1)
        online_indices = online_indices[~np.isin(online_indices, chosen)]
        take_online = min(maximum_online, len(online_indices), count - len(chosen))
        if take_online:
            chosen.extend(rng.choice(online_indices, take_online, replace=False).tolist())
        remaining = np.flatnonzero(~np.isin(np.arange(len(costs)), chosen))
        need = count - len(chosen)
        if need:
            ordered = remaining[np.argsort(costs[remaining])]
            positions = np.linspace(0, len(ordered) - 1, need + 2)[1:-1]
            candidates = ordered[np.rint(positions).astype(int)].tolist()
            for candidate in candidates:
                if candidate not in chosen:
                    chosen.append(int(candidate))
            if len(chosen) < count:
                remaining = np.flatnonzero(~np.isin(np.arange(len(costs)), chosen))
                chosen.extend(rng.choice(remaining, count - len(chosen), replace=False).tolist())
        selected = np.asarray(chosen[:count], np.int64)
        batch_actions[position] = actions[selected]
        batch_cost[position] = costs[selected]
    return batch_actions, batch_cost


def update_critics(
    critics: list[ConfigurableAbsoluteActionValueCritic],
    optimizers: list[torch.optim.Optimizer],
    trainings: list[dict],
    inputs: tuple[np.ndarray, ...],
    data: dict[str, np.ndarray],
    fit: np.ndarray,
    online: dict[str, np.ndarray],
    config: dict,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, float]:
    spec = config["critic_updates"]
    visited = np.unique(online["state_index"])
    batch_size = int(spec["state_batch_size"])
    focused = int(round(batch_size * float(spec["online_focused_state_fraction"])))
    if len(visited) == 0:
        focused = 0
    focused_rows = rng.choice(visited, focused, replace=len(visited) < focused) if focused else np.empty(0, np.int64)
    other = rng.choice(fit, batch_size - focused, replace=len(fit) < batch_size - focused)
    rows = np.concatenate((focused_rows, other)).astype(np.int64)
    actions_np, costs_np = select_training_candidates(
        data,
        rows,
        online,
        int(spec["candidates_per_state"]),
        int(spec["maximum_online_candidates_per_state_batch"]),
        rng,
    )
    actions = torch.from_numpy(actions_np).to(device)
    losses = []
    value_losses = []
    ranking_losses = []
    for twin, (critic, optimizer, training) in enumerate(zip(critics, optimizers, trainings)):
        critic.train()
        truth_np = (
            (np.log1p(costs_np.astype(np.float64)) - float(training["target_mean"]))
            / float(training["target_std"])
        ).astype(np.float32)
        truth = torch.from_numpy(truth_np).to(device)
        prediction = critic(
            torch.from_numpy(inputs[0][rows]).to(device),
            torch.from_numpy(inputs[1][rows]).to(device),
            torch.from_numpy(inputs[2][rows]).to(device),
            actions,
        )
        value = torch.nn.functional.smooth_l1_loss(prediction, truth)
        candidate_count = actions.shape[1]
        left = torch.randint(candidate_count, (len(rows), candidate_count), device=device)
        right = torch.randint(candidate_count, (len(rows), candidate_count), device=device)
        batch = torch.arange(len(rows), device=device)[:, None]
        true_delta = truth[batch, left] - truth[batch, right]
        pred_delta = prediction[batch, left] - prediction[batch, right]
        material = true_delta.abs() > 1e-5
        ranking = torch.nn.functional.softplus(
            -true_delta[material].sign() * pred_delta[material]
            / float(spec["ranking_temperature"])
        ).mean() if torch.any(material) else prediction.sum() * 0.0
        loss = value + float(spec["ranking_weight"]) * ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        value_losses.append(float(value.detach().cpu()))
        ranking_losses.append(float(ranking.detach().cpu()))
    return {
        "loss_mean": float(np.mean(losses)),
        "value_loss_mean": float(np.mean(value_losses)),
        "ranking_loss_mean": float(np.mean(ranking_losses)),
    }


def physical_critic_value(
    critic: ConfigurableAbsoluteActionValueCritic,
    training: dict,
    inputs: tuple[np.ndarray, ...],
    rows: np.ndarray,
    action: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    value = critic(
        torch.from_numpy(inputs[0][rows]).to(device),
        torch.from_numpy(inputs[1][rows]).to(device),
        torch.from_numpy(inputs[2][rows]).to(device),
        action[:, None],
    )[:, 0]
    return value * float(training["target_std"]) + float(training["target_mean"])


def interpolate_state(
    start: dict[str, torch.Tensor], end: dict[str, torch.Tensor], alpha: float
) -> dict[str, torch.Tensor]:
    output = {}
    for name in start:
        if torch.is_floating_point(start[name]):
            output[name] = start[name] + float(alpha) * (end[name] - start[name])
        else:
            output[name] = end[name]
    return output


def actor_output_step_rms(
    actor: torch.nn.Module,
    before: np.ndarray,
    inputs: tuple[np.ndarray, ...],
    fit: np.ndarray,
    sigma: np.ndarray,
    device: torch.device,
) -> float:
    after = actor_predict(actor, inputs, fit, device)
    return float(np.sqrt(np.mean(np.square((after - before) / sigma))))


def actor_update(
    actor: DirectNoAnchorGTXActor,
    selected_actor: DirectNoAnchorGTXActor,
    optimizer: torch.optim.Optimizer,
    critics: list[ConfigurableAbsoluteActionValueCritic],
    trainings: list[dict],
    inputs: tuple[np.ndarray, ...],
    fit: np.ndarray,
    config: dict,
    rng: np.random.Generator,
    sigma: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    spec = config["actor_updates"]
    rows = rng.choice(fit, int(spec["batch_size"]), replace=False).astype(np.int64)
    before_state = copy.deepcopy(actor.state_dict())
    before_output = actor_predict(actor, inputs, fit, device)
    actor.train()
    action = actor_tensor(actor, inputs, rows, device)
    with torch.no_grad():
        selected = actor_tensor(selected_actor, inputs, rows, device)
    for critic in critics:
        critic.eval()
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
    values = [
        physical_critic_value(critic, training, inputs, rows, action, device)
        for critic, training in zip(critics, trainings)
    ]
    conservative = torch.maximum(values[0], values[1])
    sigma_tensor = torch.from_numpy(sigma).to(device).reshape(1, 1, 2)
    trust = torch.mean(torch.square((action - selected) / sigma_tensor))
    loss = conservative.mean() + float(spec["selected_checkpoint_output_trust_weight"]) * trust
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(
        actor.parameters(), float(spec["gradient_clip_norm"])
    ))
    optimizer.step()
    for critic in critics:
        for parameter in critic.parameters():
            parameter.requires_grad_(True)
    end_state = copy.deepcopy(actor.state_dict())
    cap = float(spec["per_round_output_step_cap_sigma_rms"])
    raw_rms = actor_output_step_rms(actor, before_output, inputs, fit, sigma, device)
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
        "loss": float(loss.detach().cpu()),
        "conservative_log_cost": float(conservative.mean().detach().cpu()),
        "selected_output_trust": float(trust.detach().cpu()),
        "gradient_norm_before_clip": gradient_norm,
        "raw_output_step_sigma_rms": raw_rms,
        "final_output_step_sigma_rms": final_rms,
        "trust_projection": projection,
    }


def candidate_ranking_metrics(
    critics: list[ConfigurableAbsoluteActionValueCritic],
    trainings: list[dict],
    inputs: tuple[np.ndarray, ...],
    state_index: np.ndarray,
    action: np.ndarray,
    cost: np.ndarray,
    group: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    predictions = []
    for critic, training in zip(critics, trainings):
        critic.eval()
        output = np.empty(len(action), np.float32)
        with torch.no_grad():
            for start in range(0, len(action), 512):
                rows = state_index[start : start + 512]
                local_action = torch.from_numpy(action[start : start + 512, None]).to(device)
                value = critic(
                    torch.from_numpy(inputs[0][rows]).to(device),
                    torch.from_numpy(inputs[1][rows]).to(device),
                    torch.from_numpy(inputs[2][rows]).to(device),
                    local_action,
                )[:, 0]
                output[start : start + len(rows)] = (
                    value * float(training["target_std"]) + float(training["target_mean"])
                ).cpu().numpy()
        predictions.append(output)
    conservative = np.maximum(predictions[0], predictions[1])
    correlations, sign_accuracy, recovery = [], [], []
    for value in np.unique(group):
        mask = group == value
        truth = np.log1p(cost[mask].astype(np.float64))
        predicted = conservative[mask].astype(np.float64)
        correlations.append(correlation(predicted, truth))
        true_delta = truth[1:] - truth[0]
        pred_delta = predicted[1:] - predicted[0]
        material = np.abs(true_delta) > 1e-7
        sign_accuracy.append(float(np.mean(np.sign(true_delta[material]) == np.sign(pred_delta[material]))))
        base = float(cost[mask][0])
        selected = float(cost[mask][np.argmin(predicted)])
        best = float(np.min(cost[mask]))
        recovery.append((base - selected) / max(base - best, 1e-8) if base > best + 1e-8 else 0.0)
    return {
        "centered_log_cost_pearson": distribution(np.asarray(correlations)),
        "center_relative_sign_accuracy": distribution(np.asarray(sign_accuracy)),
        "bank_gain_recovery": distribution(np.asarray(recovery)),
        "twin_disagreement": distribution(np.abs(predictions[0] - predictions[1])),
    }


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left.astype(np.float64) * right.astype(np.float64), axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def fresh_fd_diagnostic(
    controller: TorchMPPIController,
    data: dict[str, np.ndarray],
    oof: np.ndarray,
    centers: np.ndarray,
    center_cost: np.ndarray,
    critics: list[ConfigurableAbsoluteActionValueCritic],
    trainings: list[dict],
    inputs: tuple[np.ndarray, ...],
    config: dict,
    weights: dict[str, float],
    sigma: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    spec = config["fresh_fd_diagnostic"]
    radius = float(spec["one_sided_radius_sigma"])
    basis = basis_bank()[0].reshape(16, 8, 2)
    signed = np.stack([sign * direction for direction in basis for sign in (1.0, -1.0)])
    probe = np.empty((len(oof), 32, 8, 2), np.float32)
    probe_cost = np.empty((len(oof), 32), np.float32)
    true_gradient = np.empty((len(oof), 16), np.float32)
    for position, row in enumerate(oof):
        local = np.clip(centers[position, None] + radius * signed * sigma, -1.0, 1.0).astype(np.float32)
        costs = evaluate(controller, data, int(row), local, weights)[0]
        x = ((local - centers[position, None]) / sigma).reshape(32, 16).astype(np.float64)
        y = np.log1p(costs.astype(np.float64)) - np.log1p(float(center_cost[position]))
        true_gradient[position] = np.linalg.solve(
            x.T @ x + float(spec["ridge"]) * np.eye(16), x.T @ y
        ).astype(np.float32)
        probe[position], probe_cost[position] = local, costs
    action = torch.from_numpy(centers).to(device)
    action.requires_grad_(True)
    predicted_values = [
        physical_critic_value(critic, training, inputs, oof, action, device)
        for critic, training in zip(critics, trainings)
    ]
    conservative = torch.maximum(predicted_values[0], predicted_values[1])
    gradient_abs = torch.autograd.grad(conservative.sum(), action)[0]
    predicted_gradient = (gradient_abs * torch.from_numpy(sigma).to(device)).detach().cpu().numpy().reshape(len(oof), 16)
    true_norm = np.linalg.norm(true_gradient, axis=1)
    predicted_norm = np.linalg.norm(predicted_gradient, axis=1)
    flat_threshold = float(np.quantile(true_norm, 0.25))
    flat = true_norm <= flat_threshold
    nonflat = ~flat
    cosines = cosine(predicted_gradient, true_gradient)
    norm_ratio = predicted_norm / np.maximum(true_norm, 1e-12)
    report = {
        "flat_true_norm_q25_threshold": flat_threshold,
        "flat_q25": {
            "true_norm": distribution(true_norm[flat]),
            "predicted_norm": distribution(predicted_norm[flat]),
        },
        "nonflat_q75": {
            "true_norm": distribution(true_norm[nonflat]),
            "predicted_norm": distribution(predicted_norm[nonflat]),
            "cosine": distribution(cosines[nonflat]),
            "norm_ratio": distribution(norm_ratio[nonflat]),
        },
    }
    arrays = {
        "fd_probe_knots": probe,
        "fd_probe_cost": probe_cost,
        "fd_true_gradient_z": true_gradient,
        "fd_predicted_gradient_z": predicted_gradient.astype(np.float32),
        "fd_true_gradient_norm": true_norm.astype(np.float32),
        "fd_predicted_gradient_norm": predicted_norm.astype(np.float32),
        "fd_gradient_cosine": cosines.astype(np.float32),
        "fd_gradient_norm_ratio": norm_ratio.astype(np.float32),
        "fd_flat_q25_mask": flat,
    }
    return report, arrays


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    if config["formal_validation_or_test_consumed"] or config["dbm_fields_or_labels_consumed"]:
        raise AssertionError("sealed-data contract violation")
    if config["query_analytic_gradient_consumed"]:
        raise AssertionError("analytic Query gradient is forbidden")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    replay_dir = Path(config["sources"]["absolute_replay"]).resolve()
    pretrain_dir = Path(config["sources"]["pretrain"]).resolve()
    stationarity_dir = Path(config["sources"]["stationarity_reassessment"]).resolve()
    replay_manifest = json.loads((replay_dir / "manifest.json").read_text())
    replay_validation = json.loads((replay_dir / "validation.json").read_text())
    pretrain_manifest = json.loads((pretrain_dir / "manifest.json").read_text())
    pretrain_validation = json.loads((pretrain_dir / "validation.json").read_text())
    stationarity_validation = json.loads((stationarity_dir / "validation.json").read_text())
    if replay_validation["qualification"] != "QUERY_SINGLE_CENTER_ABSOLUTE_REPLAY_INDEPENDENT_PASS":
        raise AssertionError("absolute Replay did not independently pass")
    if pretrain_validation["qualification"] != "QUERY_SINGLE_CENTER_PRETRAIN_INDEPENDENT_CONFIRMED_FAIL_NO_OAC":
        raise AssertionError("unexpected pretrain validation qualification")
    if stationarity_validation["qualification"] != "QUERY_SINGLE_CENTER_STATIONARITY_REASSESSMENT_INDEPENDENT_PASS":
        raise AssertionError("stationarity reassessment did not independently pass")
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
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    sigma = np.asarray(config["pilot"]["noise_sigma"], np.float32).reshape(1, 2)
    bases = basis_bank()
    radii = np.linspace(
        float(config["pilot"]["probe_radius_sigma_start"]),
        float(config["pilot"]["probe_radius_sigma_end"]),
        int(config["pilot"]["rounds"]),
    ).astype(np.float32)
    fold = int(config["split_contract"]["outer_fold"])
    selection_fold = int(config["split_contract"]["inner_selection_fold"])
    fit = np.flatnonzero(np.isin(data["fold_id"], config["split_contract"]["fit_folds"]))
    selection = np.flatnonzero(data["fold_id"] == selection_fold)
    oof = np.flatnonzero(data["fold_id"] == fold)
    if (len(fit), len(selection), len(oof)) != (360, 120, 120):
        raise AssertionError("unexpected nested split")
    episode_sets = [set(data["episode_id"][rows].tolist()) for rows in (fit, selection, oof)]
    if episode_sets[0] & episode_sets[1] or episode_sets[0] & episode_sets[2] or episode_sets[1] & episode_sets[2]:
        raise AssertionError("episode leakage")
    records = []
    for seed in config["pilot"]["seeds"]:
        seed = int(seed)
        set_seed(2_609_020 + seed)
        rng = np.random.default_rng(2_609_020 + seed)
        checkpoint_source = pretrain_dir / "checkpoints" / f"pretrain_fold{fold}_seed{seed}.pt"
        payload = torch.load(checkpoint_source, map_location=device, weights_only=False)
        if not np.array_equal(payload["fit_indices"], fit):
            raise AssertionError("fit indices differ from pretrained checkpoint")
        inputs = load_inputs(data, payload["normalization"])
        actor = actor_from_payload(payload, "actor_state_dict", device)
        selected_actor = copy.deepcopy(actor)
        critics = []
        trainings = []
        critic_optimizers = []
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
            lr=float(config["actor_updates"]["learning_rate"]),
            weight_decay=float(config["actor_updates"]["weight_decay"]),
        )
        queues = StratifiedQueues(data, fit, rng)
        online_lists: dict[str, list[np.ndarray]] = {
            "state_index": [], "action": [], "cost": [], "raw_action": [],
            "clipped": [], "round": [], "group": [], "role": [],
        }
        initial_selection_action = actor_predict(actor, inputs, selection, device)
        initial_selection_cost = direct_cost(
            controller, data, selection, initial_selection_action, weights
        )
        initial_oof_action = actor_predict(actor, inputs, oof, device)
        initial_oof_cost = direct_cost(controller, data, oof, initial_oof_action, weights)
        selection_actions = [initial_selection_action]
        selection_costs = [initial_selection_cost]
        round_records = []
        selected_round = 0
        selected_mean = float(initial_selection_cost.mean())
        selected_state = copy.deepcopy(actor.state_dict())
        selected_critic_states = [copy.deepcopy(model.state_dict()) for model in critics]
        group_cursor = 0
        for round_index in range(1, int(config["pilot"]["rounds"]) + 1):
            chosen = queues.take()
            if len(chosen) != int(config["pilot"]["fit_contexts_visited_per_round"]):
                raise AssertionError("stratified context count mismatch")
            centers = actor_predict(actor, inputs, chosen, device)
            local_actions, local_costs = [], []
            for position, row in enumerate(chosen):
                actions, costs, raw, clipped = response_bank(
                    controller,
                    data,
                    int(row),
                    centers[position],
                    float(radii[round_index - 1]),
                    bases[(round_index - 1) % len(bases)],
                    sigma,
                    weights,
                    config,
                )
                if len(actions) != int(config["pilot"]["candidates_per_visit"]):
                    raise AssertionError("candidate-count contract failed")
                local_actions.append(actions)
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
            critic_history = []
            for _ in range(int(config["critic_updates"]["updates_per_actor_update_per_twin"])):
                critic_history.append(update_critics(
                    critics, critic_optimizers, trainings, inputs, data, fit,
                    online, config, rng, device,
                ))
            selected_actor.load_state_dict(selected_state, strict=True)
            actor_info = actor_update(
                actor, selected_actor, actor_optimizer, critics, trainings,
                inputs, fit, config, rng, sigma, device,
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
            ranking = candidate_ranking_metrics(
                critics, trainings, inputs,
                online["state_index"][-len(chosen) * 39 :],
                online["action"][-len(chosen) * 39 :],
                online["cost"][-len(chosen) * 39 :],
                online["group"][-len(chosen) * 39 :],
                device,
            )
            record = {
                "round": round_index,
                "probe_radius_sigma": float(radii[round_index - 1]),
                "visited_rows": chosen.tolist(),
                "new_query_candidates": int(len(chosen) * 39),
                "total_online_replay_rows": int(len(online["cost"])),
                "new_candidate_cost": distribution(np.concatenate(local_costs)),
                "critic_update_count_per_twin": int(config["critic_updates"]["updates_per_actor_update_per_twin"]),
                "critic_loss": {
                    "mean": float(np.mean([value["loss_mean"] for value in critic_history])),
                    "value": float(np.mean([value["value_loss_mean"] for value in critic_history])),
                    "ranking": float(np.mean([value["ranking_loss_mean"] for value in critic_history])),
                },
                "actor_update": actor_info,
                "selection": metrics(
                    selection_cost, initial_selection_cost,
                    data["warm_cost"][selection], data["speed_kph"][selection]
                ),
                "selected": bool(accepted),
                "selected_round_after_evaluation": selected_round,
                "fresh_candidate_critic": ranking,
            }
            round_records.append(record)
            print(
                f"seed={seed} round={round_index}/10 Jsel={selection_cost.mean():.5f} "
                f"best={selected_mean:.5f}@{selected_round} step={actor_info['final_output_step_sigma_rms']:.6f} "
                f"pair={ranking['center_relative_sign_accuracy']['mean']:.3f}",
                flush=True,
            )
        online = {name: np.concatenate(value) for name, value in online_lists.items()}
        latest_state = copy.deepcopy(actor.state_dict())
        latest_critic_states = [copy.deepcopy(model.state_dict()) for model in critics]
        actor.load_state_dict(selected_state, strict=True)
        for critic, state in zip(critics, selected_critic_states):
            critic.load_state_dict(state, strict=True)
        selected_actor.load_state_dict(selected_state, strict=True)
        selected_selection_action = actor_predict(actor, inputs, selection, device)
        selected_selection_cost = direct_cost(
            controller, data, selection, selected_selection_action, weights
        )
        selected_oof_action = actor_predict(actor, inputs, oof, device)
        selected_oof_cost = direct_cost(controller, data, oof, selected_oof_action, weights)
        fd_report, fd_arrays = fresh_fd_diagnostic(
            controller, data, oof, selected_oof_action, selected_oof_cost,
            critics, trainings, inputs, config, weights, sigma, device,
        )
        inner_metrics = metrics(
            selected_selection_cost, initial_selection_cost,
            data["warm_cost"][selection], data["speed_kph"][selection]
        )
        oof_metrics = metrics(
            selected_oof_cost, initial_oof_cost,
            data["warm_cost"][oof], data["speed_kph"][oof]
        )
        inner_gate = {
            "mean_cost_no_greater_than_round0": inner_metrics["gain_vs_round0"]["mean"] >= -1e-6,
            "median_gain_vs_round0_nonnegative": inner_metrics["gain_vs_round0"]["median"] >= -1e-6,
            "p05_gain_vs_round0_nonnegative": inner_metrics["gain_vs_round0"]["p05"] >= -1e-6,
        }
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir()
        arrays = {
            **online,
            "fit_indices": fit,
            "selection_indices": selection,
            "oof_indices": oof,
            "probe_radius_by_round": radii,
            "selection_round_action": np.stack(selection_actions),
            "selection_round_cost": np.stack(selection_costs),
            "round0_oof_action": initial_oof_action,
            "round0_oof_cost": initial_oof_cost,
            "selected_selection_action": selected_selection_action,
            "selected_selection_cost": selected_selection_cost,
            "selected_oof_action": selected_oof_action,
            "selected_oof_cost": selected_oof_cost,
            **fd_arrays,
        }
        replay_path = seed_dir / "pilot_arrays.npz"
        np.savez_compressed(replay_path, **arrays)
        checkpoint_path = seed_dir / "checkpoint.pt"
        torch.save({
            "qualification": "QUERY_SINGLE_CENTER_OAC20TO1_TRAIN_ONLY",
            "fold": fold,
            "selection_fold": selection_fold,
            "seed": seed,
            "selected_round": selected_round,
            "actor_update_count": int(config["pilot"]["rounds"]),
            "critic_update_count_per_twin": int(config["pilot"]["rounds"]) * int(config["critic_updates"]["updates_per_actor_update_per_twin"]),
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
            "actor_optimizer_state_dict": actor_optimizer.state_dict(),
            "critic1_optimizer_state_dict": critic_optimizers[0].state_dict(),
            "critic2_optimizer_state_dict": critic_optimizers[1].state_dict(),
            "fit_indices": fit,
            "selection_indices": selection,
            "oof_indices": oof,
            "source_pretrain_checkpoint": str(checkpoint_source),
            "source_pretrain_checkpoint_sha256": sha256(checkpoint_source),
            "source_replay_sha256": replay_manifest["replay_sha256"],
            "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
            "formal_validation_or_test_consumed": False,
            "dbm_fields_or_labels_consumed": [],
            "query_analytic_gradient_consumed": False,
        }, checkpoint_path)
        seed_summary = {
            "seed": seed,
            "selected_round": selected_round,
            "selected_inner": inner_metrics,
            "selected_oof": oof_metrics,
            "inner_outcome_gates": inner_gate,
            "inner_outcome_gates_pass": bool(all(inner_gate.values())),
            "fresh_fd_diagnostic": fd_report,
            "rounds": round_records,
            "online_replay_rows": int(len(online["cost"])),
            "query_rollouts": int(
                len(online["cost"]) + len(selection) * 12 + len(oof) * 34
            ),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "arrays": str(replay_path),
            "arrays_sha256": sha256(replay_path),
        }
        dump_json(seed_dir / "summary.json", seed_summary)
        records.append(seed_summary)
        print(
            f"seed={seed} selected={selected_round} inner_gain={inner_metrics['gain_vs_round0']['mean']:.6f} "
            f"OOF_gain={oof_metrics['gain_vs_round0']['mean']:.6f} "
            f"FD_P10={fd_report['nonflat_q75']['cosine']['p10']:.3f}",
            flush=True,
        )
    pass_count = int(sum(record["inner_outcome_gates_pass"] for record in records))
    required = int(config["outcome_gates"]["minimum_seeds_passing_all_inner_gates"])
    qualification = (
        "QUERY_SINGLE_CENTER_OAC20TO1_PENDING_INDEPENDENT_VALIDATION"
        if pass_count >= required
        else "QUERY_SINGLE_CENTER_OAC20TO1_INNER_GATE_FAIL_PENDING_INDEPENDENT_VALIDATION"
    )
    summary = {
        "qualification": qualification,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": config,
        "inner_seed_pass_count": pass_count,
        "records": records,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    summary_path = output / "summary.json"
    dump_json(summary_path, summary)
    manifest = {
        "schema_version": "query-single-center-oac20to1-v1",
        "qualification": qualification,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "absolute_replay": str(replay_dir),
        "absolute_replay_sha256": replay_manifest["replay_sha256"],
        "pretrain": str(pretrain_dir),
        "pretrain_manifest_sha256": sha256(pretrain_dir / "manifest.json"),
        "stationarity_reassessment": str(stationarity_dir),
        "stationarity_validation_sha256": sha256(stationarity_dir / "validation.json"),
        "query_checkpoint": replay_manifest["query_checkpoint"],
        "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
        "summary_sha256": sha256(summary_path),
        "seed_artifacts": {
            str(record["seed"]): {
                "checkpoint_sha256": record["checkpoint_sha256"],
                "arrays_sha256": record["arrays_sha256"],
                "summary_sha256": sha256(output / f"seed_{record['seed']}" / "summary.json"),
            }
            for record in records
        },
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({
        "qualification": qualification,
        "inner_seed_pass_count": pass_count,
        "selected_round": [record["selected_round"] for record in records],
        "inner_gain": [record["selected_inner"]["gain_vs_round0"]["mean"] for record in records],
        "oof_gain": [record["selected_oof"]["gain_vs_round0"]["mean"] for record in records],
    }, indent=2))


if __name__ == "__main__":
    main()
