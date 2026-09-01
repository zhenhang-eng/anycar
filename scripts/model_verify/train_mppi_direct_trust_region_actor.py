#!/usr/bin/env python3
"""Fit the unchanged deterministic Actor to verified FR-TRPI safe targets.

Only TR1 ``internal_fit`` episodes update parameters.  Frozen
``internal_selection`` episodes select seed and epoch with the Actor's unique-output
direct DBM cost; epoch zero always competes.  Formal validation and test episodes are
never loaded.  No Critic or analytic DBM gradient is used.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TorchMPPIDeterministicCenterActor,
    ego_reference_features,
)
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from generate_dbm_proposal_teacher import repository_state, sha256_file


DEFAULT_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_direct_trust_region_train_20260807_v1"
)
DEFAULT_OLD_ACTOR = Path(
    "outputs/mppi_proposal/direct_center_actor_trust_step_20260806_v2/"
    "direct_center_actor_trust_selected.pt"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_actor_trust_region_20260807_v1"
)


@dataclass
class TrustDataset:
    inputs: tuple[np.ndarray, ...]
    safe_center: np.ndarray
    old_center: np.ndarray
    endpoint_center: np.ndarray
    projected_direction: np.ndarray
    requested_rho: np.ndarray
    trust_scale: np.ndarray
    sigma: np.ndarray
    safe_index: np.ndarray
    safe_alpha: np.ndarray
    safe_cost: np.ndarray
    old_cost: np.ndarray
    alpha_grid: np.ndarray
    direct_line_cost: np.ndarray
    episode_weight: np.ndarray
    episodes: np.ndarray
    reference_speed: np.ndarray
    scenario: np.ndarray
    initial_state_six: np.ndarray
    current_action: np.ndarray
    direct_reference: np.ndarray
    mppi_params: dict[str, Any]
    cost_weights: dict[str, Any]
    dbm_params: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--old-actor", type=Path, default=DEFAULT_OLD_ACTOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--huber-beta", type=float, default=0.05)
    parser.add_argument("--trust-weight", type=float, default=2.0)
    parser.add_argument("--bound-weight", type=float, default=0.05)
    parser.add_argument("--improvement-weight", type=float, default=1.0)
    parser.add_argument("--improvement-cap", type=float, default=20.0)
    parser.add_argument("--stay-weight", type=float, default=2.0)
    parser.add_argument("--stay-rho-threshold", type=float, default=0.025)
    parser.add_argument("--regression-mean-penalty", type=float, default=0.25)
    parser.add_argument("--regression-p95-penalty", type=float, default=0.05)
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_actor_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("actor_class") != "TorchMPPIDeterministicCenterActor":
        raise AssertionError("old checkpoint is not the deterministic Actor")
    return payload


def make_actor(payload: dict[str, Any], device: torch.device, dropout: float) -> TorchMPPIDeterministicCenterActor:
    actor = TorchMPPIDeterministicCenterActor(
        maximum_delta_sigma=float(payload["maximum_delta_sigma"]), dropout=dropout
    ).to(device)
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    return actor


def _normalization_inputs(
    source: np.lib.npyio.NpzFile,
    context: np.lib.npyio.NpzFile,
    gradient_mean: np.ndarray,
    gradient_std: np.ndarray,
    repeat: int,
    payload: dict[str, Any],
) -> tuple[np.ndarray, ...]:
    state = np.asarray(source["initial_state"], np.float32)
    action = np.asarray(source["current_action"], np.float32)
    history = np.asarray(source["history"][0], np.float32)[None]
    reference = ego_reference_features(source["reference_ego"], float(state[3]))[None]
    current = np.asarray((state[3], state[4], *action), np.float32)[None]
    normalization = MPPIProposalNormalization.from_dict(payload["state_normalization"])
    history, reference, current = normalization.normalize_numpy(history, reference, current)
    anchor = np.asarray(context["guided_center_knots"][repeat], np.float32)[None]
    feedback = np.asarray(context["first_pass_feedback"][repeat], np.float32)
    feedback = (feedback - payload["feedback_mean"]) / payload["feedback_std"]
    gradient = np.concatenate((gradient_mean[repeat], gradient_std[repeat]))
    gradient = (gradient - payload["gradient_mean"]) / payload["gradient_std"]
    return tuple(np.asarray(value, np.float32) for value in (
        history[0], reference[0], current[0], anchor[0], feedback, gradient
    ))


def load_dataset(labels: Path, payload: dict[str, Any], max_snapshots: int = 0) -> tuple[TrustDataset, dict, dict]:
    config = json.loads((labels / "config.json").read_text())
    splits = json.loads((labels / "splits.json").read_text())
    if sha256_file(Path(config["old_actor"])) != config["old_actor_sha256"]:
        raise AssertionError("TR1 old Actor hash no longer matches")
    fields: dict[str, list] = {name: [] for name in (
        "history", "reference", "current", "anchor", "feedback", "gradient",
        "safe_center", "old_center", "endpoint_center", "projected_direction",
        "requested_rho", "trust_scale", "sigma", "safe_index", "safe_alpha",
        "safe_cost", "old_cost",
        "alpha_grid", "direct_line_cost",
        "episodes", "reference_speed", "scenario", "initial_state_six",
        "current_action", "direct_reference",
    )}
    fixed_config: tuple[str, str, str] | None = None
    label_paths = sorted(labels.glob("episode_*/*.npz"))
    if max_snapshots:
        label_paths = label_paths[:max_snapshots]
    for label_path in label_paths:
        episode = label_path.parent.name
        if episode not in set(splits["train"]):
            raise AssertionError(f"non-train label found: {label_path}")
        with np.load(label_path, allow_pickle=False) as label:
            source_path = Path(str(label["source_snapshot"]))
            context_path = Path(str(label["context_label"]))
            risk_text = str(label["risk_label"])
            with np.load(source_path, allow_pickle=False) as source, np.load(
                context_path, allow_pickle=False
            ) as context:
                if risk_text:
                    with np.load(Path(risk_text), allow_pickle=False) as risk:
                        gradient_mean = np.asarray(risk["critic_gradient_mean"], np.float32)
                        gradient_std = np.asarray(risk["critic_gradient_std"], np.float32)
                else:
                    gradient_mean = np.asarray(context["critic_gradient_mean"], np.float32)
                    gradient_std = np.asarray(context["critic_gradient_std"], np.float32)
                mppi_config = json.loads(str(source["mppi_params_json"]))
                # Collection seed varies by episode but has no role in the
                # deterministic one-center direct rollout used by TR2.  Older
                # snapshots also omit newer MPPI fields whose dataclass defaults
                # equal the explicit values in the expansion collection.
                comparable_mppi = asdict(TorchMPPIParams(**mppi_config))
                comparable_mppi.pop("seed", None)
                one_config = (
                    json.dumps(comparable_mppi, sort_keys=True),
                    json.dumps(json.loads(str(source["cost_weights_json"])), sort_keys=True),
                    json.dumps(json.loads(str(source["dbm_params_json"])), sort_keys=True),
                )
                if fixed_config is None:
                    fixed_config = one_config
                elif one_config != fixed_config:
                    raise AssertionError("mixed DBM/MPPI/cost configuration in TR1 labels")
                reference = np.asarray(source["reference"], np.float32)
                horizon = int(mppi_config["horizon"])
                if len(reference) == horizon + 1:
                    reference = reference[1:]
                for repeat in range(len(label["safe_index"])):
                    values = _normalization_inputs(
                        source, context, gradient_mean, gradient_std, repeat, payload
                    )
                    for name, value in zip(
                        ("history", "reference", "current", "anchor", "feedback", "gradient"),
                        values,
                    ):
                        fields[name].append(value)
                    safe = int(label["safe_index"][repeat])
                    fields["safe_center"].append(np.asarray(label["safe_center"][repeat], np.float32))
                    fields["old_center"].append(np.asarray(label["old_actor_center"][repeat], np.float32))
                    fields["endpoint_center"].append(np.asarray(label["line_centers"][repeat, -1], np.float32))
                    fields["projected_direction"].append(np.asarray(label["projected_normalized_direction"][repeat], np.float32))
                    fields["requested_rho"].append(float(label["requested_rho"][repeat]))
                    fields["trust_scale"].append(float(label["trust_scale"][repeat]))
                    fields["sigma"].append(np.asarray(label["source_sigma"], np.float32))
                    fields["safe_index"].append(safe)
                    fields["safe_alpha"].append(float(label["safe_alpha"][repeat]))
                    fields["safe_cost"].append(float(label["direct_cost"][repeat, safe]))
                    fields["old_cost"].append(float(label["direct_cost"][repeat, 0]))
                    fields["alpha_grid"].append(np.asarray(label["alpha_grid"], np.float32))
                    fields["direct_line_cost"].append(
                        np.asarray(label["direct_cost"][repeat], np.float32)
                    )
                    fields["episodes"].append(episode)
                    fields["reference_speed"].append(float(label["reference_speed_mps"]))
                    fields["scenario"].append(str(label["scenario_class"]))
                    fields["initial_state_six"].append(np.asarray(source["initial_state_six"], np.float32))
                    fields["current_action"].append(np.asarray(source["current_action"], np.float32))
                    fields["direct_reference"].append(reference)
    if not fields["safe_center"] or fixed_config is None:
        raise ValueError("no TR1 labels loaded")
    episodes = np.asarray(fields["episodes"])
    unique, counts = np.unique(episodes, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    episode_weight = np.asarray([1.0 / count_map[value] for value in episodes], np.float32)
    episode_weight /= episode_weight.mean()
    dataset = TrustDataset(
        inputs=tuple(np.asarray(fields[name], np.float32) for name in (
            "history", "reference", "current", "anchor", "feedback", "gradient"
        )),
        safe_center=np.asarray(fields["safe_center"], np.float32),
        old_center=np.asarray(fields["old_center"], np.float32),
        endpoint_center=np.asarray(fields["endpoint_center"], np.float32),
        projected_direction=np.asarray(fields["projected_direction"], np.float32),
        requested_rho=np.asarray(fields["requested_rho"], np.float32),
        trust_scale=np.asarray(fields["trust_scale"], np.float32),
        sigma=np.asarray(fields["sigma"], np.float32),
        safe_index=np.asarray(fields["safe_index"], np.int64),
        safe_alpha=np.asarray(fields["safe_alpha"], np.float32),
        safe_cost=np.asarray(fields["safe_cost"], np.float32),
        old_cost=np.asarray(fields["old_cost"], np.float32),
        alpha_grid=np.asarray(fields["alpha_grid"], np.float32),
        direct_line_cost=np.asarray(fields["direct_line_cost"], np.float32),
        episode_weight=episode_weight,
        episodes=episodes,
        reference_speed=np.asarray(fields["reference_speed"], np.float32),
        scenario=np.asarray(fields["scenario"]),
        initial_state_six=np.asarray(fields["initial_state_six"], np.float32),
        current_action=np.asarray(fields["current_action"], np.float32),
        direct_reference=np.asarray(fields["direct_reference"], np.float32),
        mppi_params=json.loads(fixed_config[0]),
        cost_weights=json.loads(fixed_config[1]),
        dbm_params=json.loads(fixed_config[2]),
    )
    return dataset, config, splits


def tensorize(data: TrustDataset, device: torch.device) -> dict[str, Any]:
    return {
        "inputs": tuple(torch.from_numpy(value).to(device) for value in data.inputs),
        "safe_center": torch.from_numpy(data.safe_center).to(device),
        "old_center": torch.from_numpy(data.old_center).to(device),
        "sigma": torch.from_numpy(data.sigma).to(device),
        "safe_index": torch.from_numpy(data.safe_index).to(device),
        "old_cost": torch.from_numpy(data.old_cost).to(device),
        "alpha_grid": torch.from_numpy(data.alpha_grid).to(device),
        "direct_line_cost": torch.from_numpy(data.direct_line_cost).to(device),
        "initial_state_six": torch.from_numpy(data.initial_state_six).to(device),
        "current_action": torch.from_numpy(data.current_action).to(device),
        "direct_reference": torch.from_numpy(data.direct_reference).to(device),
    }


def actor_batch(actor: TorchMPPIDeterministicCenterActor, tensors: dict[str, Any], index: torch.Tensor):
    return actor(*(value[index] for value in tensors["inputs"]))


@torch.no_grad()
def actor_outputs(
    actor: TorchMPPIDeterministicCenterActor,
    tensors: dict[str, Any],
    index: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    actions, centers = [], []
    actor.eval()
    for start in range(0, len(index), batch_size):
        one = torch.from_numpy(index[start:start + batch_size]).to(device)
        action, center = actor_batch(actor, tensors, one)
        actions.append(action.cpu().numpy())
        centers.append(center.cpu().numpy())
    return np.concatenate(actions), np.concatenate(centers)


@torch.no_grad()
def direct_cost(
    centers: np.ndarray,
    data: TrustDataset,
    tensors: dict[str, Any],
    index: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    params = TorchMPPIParams(**data.mppi_params)
    weights = TorchMPPICostWeights(**data.cost_weights)
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**data.dbm_params))
    values = []
    for start in range(0, len(index), batch_size):
        one_np = index[start:start + batch_size]
        one = torch.from_numpy(one_np).to(device)
        center = torch.from_numpy(centers[start:start + batch_size]).to(device)
        actions = interpolate_knots(center, params.horizon).unsqueeze(1)
        value = batched_cost(
            backend, weights, actions,
            tensors["initial_state_six"][one],
            tensors["current_action"][one],
            tensors["direct_reference"][one],
        )
        values.append(value[:, 0].cpu().numpy())
    return np.concatenate(values)


def distribution(value: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "p05": float(np.quantile(value, 0.05)),
        "p95": float(np.quantile(value, 0.95)),
        "minimum": float(np.min(value)),
        "maximum": float(np.max(value)),
    }


def evaluate_actor(
    actor: TorchMPPIDeterministicCenterActor,
    data: TrustDataset,
    tensors: dict[str, Any],
    index: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    action, center = actor_outputs(
        actor, tensors, index, args.evaluation_batch_size, device
    )
    cost = direct_cost(
        center, data, tensors, index, args.evaluation_batch_size, device
    )
    sigma = data.sigma[index, None, :]
    target_error = (center - data.safe_center[index]) / sigma
    old_delta = (center - data.old_center[index]) / sigma
    rho = np.sqrt(np.mean(old_delta ** 2, axis=(1, 2)))
    gain = data.old_cost[index] - cost
    regression = np.maximum(-gain, 0.0)
    target_stay = data.safe_index[index] == 0
    predicted_stay = rho <= args.stay_rho_threshold
    stay_accuracy = float(np.mean(predicted_stay == target_stay))
    stay_recall = float(np.mean(predicted_stay[target_stay])) if np.any(target_stay) else 1.0
    move_recall = float(np.mean(~predicted_stay[~target_stay])) if np.any(~target_stay) else 1.0
    score = (
        float(np.mean(cost))
        + args.regression_mean_penalty * float(np.mean(regression))
        + args.regression_p95_penalty * float(np.quantile(regression, 0.95))
    )
    return {
        "selection_score": score,
        "direct_cost": distribution(cost),
        "old_cost": distribution(data.old_cost[index]),
        "safe_cost": distribution(data.safe_cost[index]),
        "gain_vs_old": distribution(gain),
        "regression_fraction": float(np.mean(gain < 0.0)),
        "target_sigma_rmse": float(np.sqrt(np.mean(target_error ** 2))),
        "target_sigma_mae": float(np.mean(np.abs(target_error))),
        "rho_from_old": distribution(rho),
        "trust_violation_fraction": float(np.mean(rho > float(data.mppi_params.get("trust_radius_sigma_rms", 0.5)) + 1e-6)),
        "stay_accuracy": stay_accuracy,
        "stay_recall": stay_recall,
        "move_recall": move_recall,
        "action_saturation_fraction": float(np.mean(np.abs(action) >= 0.999)),
    }


def sample_weights(data: TrustDataset, index: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    gain = np.maximum(data.old_cost[index] - data.safe_cost[index], 0.0)
    improvement = 1.0 + args.improvement_weight * np.minimum(
        gain / max(args.improvement_cap, 1e-6), 1.0
    )
    stay = np.where(data.safe_index[index] == 0, args.stay_weight, 1.0)
    result = data.episode_weight[index] * improvement.astype(np.float32) * stay.astype(np.float32)
    return (result / np.mean(result)).astype(np.float32)


def optimize_epochs(
    actor: TorchMPPIDeterministicCenterActor,
    data: TrustDataset,
    tensors: dict[str, Any],
    index: np.ndarray,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> None:
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.minimum_learning_rate
    )
    weights = torch.from_numpy(sample_weights(data, index, args)).to(device)
    old_lookup = torch.from_numpy(data.old_center[index]).to(device)
    safe_lookup = torch.from_numpy(data.safe_center[index]).to(device)
    sigma_lookup = torch.from_numpy(data.sigma[index]).to(device)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        actor.train()
        order = rng.permutation(len(index))
        for start in range(0, len(order), args.batch_size):
            local_np = order[start:start + args.batch_size]
            local = torch.from_numpy(local_np).to(device)
            absolute = torch.from_numpy(index[local_np]).to(device)
            action, center = actor_batch(actor, tensors, absolute)
            sigma = sigma_lookup[local].unsqueeze(1)
            target_error = (center - safe_lookup[local]) / sigma
            target_loss = F.smooth_l1_loss(
                target_error, torch.zeros_like(target_error),
                beta=args.huber_beta, reduction="none",
            ).mean(dim=(1, 2))
            old_delta = (center - old_lookup[local]) / sigma
            rho = torch.sqrt(torch.mean(old_delta.square(), dim=(1, 2)) + 1e-12)
            trust_loss = F.relu(rho - 0.5).square()
            bound_loss = F.relu(action.abs() - 0.98).square().mean(dim=(1, 2))
            per_sample = target_loss + args.trust_weight * trust_loss + args.bound_weight * bound_loss
            loss = torch.sum(per_sample * weights[local]) / torch.sum(weights[local])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            optimizer.step()
        scheduler.step()


def fit_seed(
    seed: int,
    payload: dict[str, Any],
    data: TrustDataset,
    tensors: dict[str, Any],
    fit_index: np.ndarray,
    selection_index: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, list[dict[str, Any]]]:
    set_seed(seed)
    actor = make_actor(payload, device, dropout=0.05)
    history: list[dict[str, Any]] = []
    initial_old_error = float(np.max(np.abs(
        actor_outputs(actor, tensors, np.arange(len(data.episodes)), args.evaluation_batch_size, device)[1]
        - data.old_center
    )))
    if initial_old_error > 2e-6:
        raise AssertionError(f"old Actor reconstruction error {initial_old_error}")
    best_state = copy.deepcopy(actor.state_dict())
    best_epoch = 0
    initial = evaluate_actor(actor, data, tensors, selection_index, args, device)
    initial["epoch"] = 0
    initial["learning_rate"] = args.learning_rate
    history.append(initial)
    best_score = float(initial["selection_score"])
    print(
        f"[seed={seed} epoch=000] score={best_score:.4f} "
        f"cost={initial['direct_cost']['mean']:.4f} "
        f"target={initial['target_sigma_rmse']:.4f}", flush=True,
    )
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.minimum_learning_rate
    )
    fit_weights = torch.from_numpy(sample_weights(data, fit_index, args)).to(device)
    old_fit = torch.from_numpy(data.old_center[fit_index]).to(device)
    safe_fit = torch.from_numpy(data.safe_center[fit_index]).to(device)
    sigma_fit = torch.from_numpy(data.sigma[fit_index]).to(device)
    rng = np.random.default_rng(seed)
    for epoch in range(1, args.epochs + 1):
        actor.train()
        order = rng.permutation(len(fit_index))
        losses = []
        for start in range(0, len(order), args.batch_size):
            local_np = order[start:start + args.batch_size]
            local = torch.from_numpy(local_np).to(device)
            absolute = torch.from_numpy(fit_index[local_np]).to(device)
            action, center = actor_batch(actor, tensors, absolute)
            sigma = sigma_fit[local].unsqueeze(1)
            target_error = (center - safe_fit[local]) / sigma
            target_loss = F.smooth_l1_loss(
                target_error, torch.zeros_like(target_error),
                beta=args.huber_beta, reduction="none",
            ).mean(dim=(1, 2))
            old_delta = (center - old_fit[local]) / sigma
            rho = torch.sqrt(torch.mean(old_delta.square(), dim=(1, 2)) + 1e-12)
            trust_loss = F.relu(rho - 0.5).square()
            bound_loss = F.relu(action.abs() - 0.98).square().mean(dim=(1, 2))
            per_sample = target_loss + args.trust_weight * trust_loss + args.bound_weight * bound_loss
            loss = torch.sum(per_sample * fit_weights[local]) / torch.sum(fit_weights[local])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        if epoch % args.evaluation_interval == 0 or epoch == args.epochs:
            metrics = evaluate_actor(actor, data, tensors, selection_index, args, device)
            metrics.update({
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            })
            history.append(metrics)
            score = float(metrics["selection_score"])
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(actor.state_dict())
            print(
                f"[seed={seed} epoch={epoch:03d}] score={score:.4f} "
                f"cost={metrics['direct_cost']['mean']:.4f} "
                f"target={metrics['target_sigma_rmse']:.4f} "
                f"p05={metrics['gain_vs_old']['p05']:.3f} "
                f"stay={metrics['stay_recall']:.3f}", flush=True,
            )
    return best_state, best_epoch, history


def checkpoint_payload(
    actor: TorchMPPIDeterministicCenterActor,
    old_payload: dict[str, Any],
    labels: Path,
    old_actor: Path,
    seed: int,
    epoch: int,
    fit_episodes: list[str],
    selection_episodes: list[str],
    args: argparse.Namespace,
    qualification: str,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "method": "FR-TRPI safe-target deterministic Actor fit",
        "qualification": qualification,
        "actor_class": "TorchMPPIDeterministicCenterActor",
        "actor_state_dict": copy.deepcopy(actor.cpu().state_dict()),
        "state_normalization": old_payload["state_normalization"],
        "feedback_mean": old_payload["feedback_mean"],
        "feedback_std": old_payload["feedback_std"],
        "gradient_mean": old_payload["gradient_mean"],
        "gradient_std": old_payload["gradient_std"],
        "maximum_delta_sigma": float(old_payload["maximum_delta_sigma"]),
        "source_old_actor": str(old_actor.resolve()),
        "source_old_actor_sha256": sha256_file(old_actor),
        "source_labels": str(labels.resolve()),
        "source_labels_config_sha256": sha256_file(labels / "config.json"),
        "source_labels_splits_sha256": sha256_file(labels / "splits.json"),
        "source_labels_summary_sha256": sha256_file(labels / "summary.json"),
        "seed": seed,
        "selected_epoch": epoch,
        "fit_episodes": fit_episodes,
        "selection_episodes": selection_episodes,
        "training_arguments": vars(args),
        "test_policy": "formal validation and test not loaded or evaluated",
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.epochs < 1 or args.evaluation_interval < 1:
        raise ValueError("epochs and evaluation interval must be positive")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    old_payload = load_actor_payload(args.old_actor)
    data, label_config, splits = load_dataset(args.labels, old_payload, args.max_snapshots)
    if label_config["old_actor_sha256"] != sha256_file(args.old_actor):
        raise AssertionError("requested old Actor differs from TR1 source")
    fit_episodes = list(splits["internal_fit"])
    selection_episodes = list(splits["internal_selection"])
    if args.max_snapshots:
        present = set(data.episodes.tolist())
        fit_episodes = sorted(present)
        selection_episodes = sorted(present)
    fit_index = np.flatnonzero(np.isin(data.episodes, fit_episodes))
    selection_index = np.flatnonzero(np.isin(data.episodes, selection_episodes))
    if not args.max_snapshots and (
        len(fit_index) + len(selection_index) != len(data.episodes)
        or set(fit_episodes) & set(selection_episodes)
    ):
        raise AssertionError("invalid internal episode split")
    tensors = tensorize(data, device)
    seed_runs = []
    seed_states: dict[int, dict[str, torch.Tensor]] = {}
    for seed in args.seeds:
        state, epoch, history = fit_seed(
            seed, old_payload, data, tensors, fit_index, selection_index, args, device
        )
        actor = make_actor(old_payload, device, dropout=0.0)
        actor.load_state_dict(state, strict=True)
        selected_metrics = evaluate_actor(
            actor, data, tensors, selection_index, args, device
        )
        seed_states[seed] = state
        torch.save(
            checkpoint_payload(
                actor, old_payload, args.labels, args.old_actor, seed, epoch,
                fit_episodes, selection_episodes, args,
                "TR2_INTERNAL_SELECTION_ONLY",
            ),
            args.output_dir / f"selection_actor_seed{seed}.pt",
        )
        seed_runs.append({
            "seed": seed,
            "selected_epoch": epoch,
            "selected_metrics": selected_metrics,
            "history": history,
        })
    winner = min(seed_runs, key=lambda row: row["selected_metrics"]["selection_score"])
    selected_seed = int(winner["seed"])
    selected_epoch = int(winner["selected_epoch"])
    set_seed(selected_seed)
    final_actor = make_actor(old_payload, device, dropout=0.05)
    all_index = np.arange(len(data.episodes), dtype=np.int64)
    optimize_epochs(
        final_actor, data, tensors, all_index, selected_epoch, args, device, selected_seed
    )
    final_actor.eval()
    final_all_metrics = evaluate_actor(
        final_actor, data, tensors, all_index, args, device
    )
    final_payload = checkpoint_payload(
        final_actor, old_payload, args.labels, args.old_actor,
        selected_seed, selected_epoch, fit_episodes, selection_episodes, args,
        "TR2_FROZEN_TRAIN_DOMAIN_ONLY",
    )
    final_path = args.output_dir / "direct_actor_trust_region_selected.pt"
    torch.save(final_payload, final_path)
    summary = {
        "format_version": 1,
        "method": "FR-TRPI safe-target deterministic Actor fit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "labels": str(args.labels.resolve()),
        "labels_hashes": {
            name: sha256_file(args.labels / name)
            for name in ("config.json", "splits.json", "summary.json")
        },
        "old_actor": str(args.old_actor.resolve()),
        "old_actor_sha256": sha256_file(args.old_actor),
        "snapshot_count": int(len(data.episodes) // 2),
        "context_count": int(len(data.episodes)),
        "fit_episode_count": len(fit_episodes),
        "selection_episode_count": len(selection_episodes),
        "fit_context_count": int(len(fit_index)),
        "selection_context_count": int(len(selection_index)),
        "seeds": seed_runs,
        "selected_seed": selected_seed,
        "selected_epoch": selected_epoch,
        "selected_internal_metrics": winner["selected_metrics"],
        "refit_all_train_episodes": True,
        "final_all_train_metrics": final_all_metrics,
        "checkpoint": str(final_path.resolve()),
        "checkpoint_sha256": sha256_file(final_path),
        "qualification": "TR2_FROZEN_TRAIN_DOMAIN_ONLY",
        "test_policy": "formal validation and test not loaded or evaluated",
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({
        "selected_seed": selected_seed,
        "selected_epoch": selected_epoch,
        "selected_internal_metrics": winner["selected_metrics"],
        "final_all_train_metrics": final_all_metrics,
        "checkpoint": str(final_path),
        "qualification": summary["qualification"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
