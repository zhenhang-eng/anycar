#!/usr/bin/env python3
"""Episode-grouped high-speed Actor BC and twin absolute-value Critic pretrain.

This is deliberately a train-only initialization stage.  The Actor is the
strict no-anchor G-X model and learns the proximal-search teacher as an
absolute 8x2 center.  Each Critic learns scalar log1p(DBM J50) from all 129
search candidates per state; no TD target, bootstrap, or Actor update is used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from car_dynamics.controllers_torch.dbm import TorchDynamicBicycleRolloutBackend
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    ego_reference_features,
)
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from mppi_a2_actors import DirectNoAnchorGTXActor
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic


DEFAULT_REPLAY = Path(
    "outputs/mppi_proposal/highspeed_initial_dbm_replay_20260828_v1"
)
DEFAULT_TEACHER = Path(
    "outputs/mppi_proposal/highspeed_proximal_teacher_20260830_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/highspeed_actor_twin_critic_pretrain_20260830_v2"
)
SUPPORT_STD = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--episodes-per-stratum", type=int, choices=(1, 2, 4), default=1,
        help="Use the first N independent episodes in every speed/scenario stratum.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--actor-epochs", type=int, default=500)
    parser.add_argument("--critic-epochs", type=int, default=240)
    parser.add_argument("--selection-stride", type=int, default=10)
    parser.add_argument("--actor-batch-size", type=int, default=32)
    parser.add_argument("--critic-state-batch-size", type=int, default=8)
    parser.add_argument("--critic-candidates-per-state", type=int, default=48)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--ranking-weight", type=float, default=0.35)
    parser.add_argument("--ranking-temperature", type=float, default=0.08)
    parser.add_argument("--rollout-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float64).ravel()
    right = np.asarray(right, np.float64).ravel()
    if left.size < 2 or left.std() < 1e-12 or right.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return correlation(rankdata(np.asarray(left).ravel()), rankdata(np.asarray(right).ravel()))


def make_folds(episodes: np.ndarray, speeds: np.ndarray, scenarios: np.ndarray) -> np.ndarray:
    speed_values = sorted(np.unique(speeds).tolist())
    scenario_values = sorted(np.unique(scenarios).tolist())
    speed_index = {float(value): index for index, value in enumerate(speed_values)}
    scenario_index = {str(value): index for index, value in enumerate(scenario_values)}
    episode_fold: dict[str, int] = {}
    for speed in speed_values:
        for scenario in scenario_values:
            mask = (speeds == speed) & (scenarios == scenario)
            local_episodes = sorted(np.unique(episodes[mask]).tolist())
            for repeat_index, episode in enumerate(local_episodes):
                episode_fold[str(episode)] = (
                    speed_index[float(speed)]
                    + scenario_index[str(scenario)]
                    + repeat_index
                ) % 5
    folds = np.asarray([episode_fold[str(value)] for value in episodes], np.int64)
    expected = len(episodes) // 5
    if len(episodes) % 5 or [int(np.sum(folds == fold)) for fold in range(5)] != [expected] * 5:
        raise AssertionError("expected balanced contexts in each outer fold")
    return folds


def select_episode_subset(
    data: dict[str, np.ndarray], episodes_per_stratum: int,
) -> dict[str, np.ndarray]:
    """Select a deterministic nested, stratum-balanced episode subset."""
    selected_episodes: list[str] = []
    for speed in sorted(np.unique(data["speed"]).tolist()):
        for scenario in sorted(np.unique(data["scenario"]).tolist()):
            mask = (data["speed"] == speed) & (data["scenario"] == scenario)
            local = sorted(np.unique(data["episode"][mask]).tolist())
            if len(local) < episodes_per_stratum:
                raise AssertionError(
                    f"stratum {(speed, scenario)} has only {len(local)} episodes"
                )
            selected_episodes.extend(local[:episodes_per_stratum])
    rows = np.flatnonzero(np.isin(data["episode"], selected_episodes))
    expected_episodes = 30 * episodes_per_stratum
    if len(np.unique(data["episode"][rows])) != expected_episodes:
        raise AssertionError("unexpected selected episode count")
    output: dict[str, np.ndarray] = {}
    full_count = len(data["episode"])
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == full_count:
            output[key] = value[rows]
        else:
            output[key] = value
    output["source_indices"] = rows.astype(np.int64)
    return output


def load_data(replay_dir: Path, teacher_dir: Path) -> dict[str, np.ndarray]:
    replay_path = replay_dir / "replay.npz"
    replay_summary = json.loads((replay_dir / "summary.json").read_text())
    teacher_path = teacher_dir / "labels.npz"
    teacher_summary = json.loads((teacher_dir / "summary.json").read_text())
    if replay_summary.get("formal_validation_or_test_created"):
        raise AssertionError("replay is not train-only")
    if teacher_summary.get("formal_validation_or_test_created"):
        raise AssertionError("teacher is not train-only")
    if sha256(replay_path) != replay_summary["archive_sha256"]:
        raise AssertionError("replay hash mismatch")
    if sha256(teacher_path) != teacher_summary["labels_sha256"]:
        raise AssertionError("teacher hash mismatch")
    with np.load(replay_path, allow_pickle=False) as loaded:
        replay = {name: np.asarray(loaded[name]) for name in loaded.files}
    with np.load(teacher_path, allow_pickle=False) as loaded:
        teacher = {name: np.asarray(loaded[name]) for name in loaded.files}
    count = len(replay["state_six"])
    if teacher["search_centers"].shape != (count, 129, 8, 2):
        raise AssertionError("unexpected high-speed data shape")
    for name in ("episode_id", "scenario_class", "control_step"):
        if not np.array_equal(replay[name], teacher[name]):
            raise AssertionError(f"row contract mismatch: {name}")
    if not np.allclose(replay["mean_knots_before"], teacher["anchor_knots"], atol=1e-7):
        raise AssertionError("teacher anchor mismatch")
    if not np.allclose(teacher["search_costs"][:, 0], teacher["anchor_cost"], atol=1e-4):
        raise AssertionError("candidate zero is not warm")
    reference = np.stack([
        ego_reference_features(value, float(state[3]))
        for value, state in zip(replay["reference_ego"], replay["state_six"])
    ]).astype(np.float32)
    # Deployable 4-D contract: vx, yaw rate, acceleration, steering.  Lateral
    # velocity/beta are intentionally excluded because the real vehicle does
    # not provide them as direct observations.
    current = np.stack([
        np.asarray((state[3], state[5], *action), np.float32)
        for state, action in zip(replay["state_six"], replay["current_action"])
    ])
    return {
        "history": replay["history"].astype(np.float32),
        "reference": reference,
        "current": current,
        "state_six": replay["state_six"].astype(np.float32),
        "current_action": replay["current_action"].astype(np.float32),
        "rollout_reference": replay["reference"][:, 1:].astype(np.float32),
        "anchor": teacher["anchor_knots"].astype(np.float32),
        "teacher": teacher["teacher_knots"].astype(np.float32),
        "anchor_cost": teacher["anchor_cost"].astype(np.float32),
        "teacher_cost": teacher["teacher_cost"].astype(np.float32),
        "actions": teacher["search_centers"].astype(np.float32),
        "costs": teacher["search_costs"].astype(np.float32),
        "episode": teacher["episode_id"].astype(str),
        "scenario": teacher["scenario_class"].astype(str),
        "speed": teacher["nominal_speed_kph"].astype(np.float32),
        "actual_vx": teacher["actual_vx_mps"].astype(np.float32),
        "replay_sha256": np.asarray(sha256(replay_path)),
        "teacher_sha256": np.asarray(sha256(teacher_path)),
    }


def normalized_inputs(
    data: dict[str, np.ndarray], fit: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], MPPIProposalNormalization]:
    normalizer = MPPIProposalNormalization.fit(
        data["history"][fit], data["reference"][fit], data["current"][fit]
    )
    history, reference, current = normalizer.normalize_numpy(
        data["history"], data["reference"], data["current"]
    )
    count = len(history)
    return (
        history.astype(np.float32), reference.astype(np.float32),
        current.astype(np.float32), np.zeros((count, 8, 2), np.float32),
        np.zeros((count, 74), np.float32), np.zeros((count, 32), np.float32),
    ), normalizer


def actor_center_scale(labels: np.ndarray, fit: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    center = labels[fit].mean(axis=0).astype(np.float32)
    scale = (SUPPORT_STD * labels[fit].std(axis=0) + 1e-4).astype(np.float32)
    # Preserve enough physical support even when one fold has a nearly constant dimension.
    scale = np.maximum(scale, np.asarray((0.12, 0.12), np.float32)[None])
    return torch.from_numpy(center), torch.from_numpy(scale)


def actor_predict(
    actor: nn.Module, inputs: tuple[np.ndarray, ...], indices: np.ndarray,
    device: torch.device, batch_size: int = 256,
) -> np.ndarray:
    output = []
    actor.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            local = indices[start : start + batch_size]
            tensors = tuple(torch.from_numpy(value[local]).to(device) for value in inputs)
            _, center = actor(*tensors)
            output.append(center.cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def rollout_cost(
    data: dict[str, np.ndarray], knots: np.ndarray, indices: np.ndarray,
    backend: TorchDynamicBicycleRolloutBackend, weights: TorchMPPICostWeights,
    params: TorchMPPIParams, device: torch.device, batch_size: int,
) -> np.ndarray:
    result = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            local = indices[start : start + batch_size]
            action = interpolate_knots(
                torch.from_numpy(knots[start : start + len(local)]).to(device),
                params.horizon,
            ).unsqueeze(1)
            value = batched_cost(
                backend, weights, action,
                torch.from_numpy(data["state_six"][local]).to(device),
                torch.from_numpy(data["current_action"][local]).to(device),
                torch.from_numpy(data["rollout_reference"][local]).to(device),
            )
            result.append(value[:, 0].cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def actor_metrics(
    prediction_cost: np.ndarray, anchor_cost: np.ndarray, teacher_cost: np.ndarray,
) -> dict[str, Any]:
    gain = anchor_cost - prediction_cost
    teacher_gain = anchor_cost - teacher_cost
    denominator = float(np.sum(teacher_gain))
    recovery = float(np.sum(gain) / denominator) if denominator > 0 else 0.0
    return {
        "aggregate_teacher_gain_recovery": recovery,
        "actor_cost": distribution(prediction_cost),
        "warm_relative_gain": distribution(gain),
        "teacher_gain": distribution(teacher_gain),
        "beats_or_equals_warm_fraction": float(np.mean(gain >= -1e-5)),
        "strictly_beats_warm_fraction": float(np.mean(gain > 1e-5)),
        "regression_fraction": float(np.mean(gain < -1e-5)),
    }


def train_actor(
    data: dict[str, np.ndarray], inputs: tuple[np.ndarray, ...], fit: np.ndarray,
    selection: np.ndarray, seed: int, args: argparse.Namespace,
    backend: TorchDynamicBicycleRolloutBackend, weights: TorchMPPICostWeights,
    params: TorchMPPIParams, device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(seed)
    center, scale = actor_center_scale(data["teacher"], fit)
    actor = DirectNoAnchorGTXActor(
        dropout=0.0, center=center.to(device), scale=scale.to(device)
    ).to(device)
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.actor_lr, weight_decay=args.weight_decay
    )
    target = torch.from_numpy(data["teacher"]).to(device)
    rng = np.random.default_rng(806_301 + seed)
    best_score = math.inf
    best_state = None
    best_epoch = 0
    history = []
    for epoch in range(1, args.actor_epochs + 1):
        actor.train()
        order = rng.permutation(fit)
        losses = []
        for start in range(0, len(order), args.actor_batch_size):
            rows = order[start : start + args.actor_batch_size]
            tensors = tuple(torch.from_numpy(value[rows]).to(device) for value in inputs)
            _, predicted = actor(*tensors)
            loss = (predicted - target[rows]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % args.selection_stride == 0 or epoch == args.actor_epochs:
            predicted = actor_predict(actor, inputs, selection, device)
            cost = rollout_cost(
                data, predicted, selection, backend, weights, params, device,
                args.rollout_batch_size,
            )
            score = float(np.mean(cost))
            record = {
                "epoch": epoch, "fit_mse": float(np.mean(losses)),
                "selection_cost_mean": score,
                "selection_recovery": actor_metrics(
                    cost, data["anchor_cost"][selection], data["teacher_cost"][selection]
                )["aggregate_teacher_gain_recovery"],
            }
            history.append(record)
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(actor.state_dict())
    if best_state is None:
        raise AssertionError("Actor checkpoint selection failed")
    actor.load_state_dict(best_state, strict=True)
    return actor, {
        "best_epoch": best_epoch,
        "best_selection_cost_mean": best_score,
        "selection_history": history,
        "out_center": center.numpy().tolist(),
        "out_scale": scale.numpy().tolist(),
        "support_std": SUPPORT_STD,
    }


def critic_predict(
    critic: nn.Module, inputs: tuple[np.ndarray, ...], actions: np.ndarray,
    indices: np.ndarray, device: torch.device, state_batch: int = 16,
) -> np.ndarray:
    output = []
    critic.eval()
    with torch.no_grad():
        for start in range(0, len(indices), state_batch):
            rows = indices[start : start + state_batch]
            output.append(critic(
                torch.from_numpy(inputs[0][rows]).to(device),
                torch.from_numpy(inputs[1][rows]).to(device),
                torch.from_numpy(inputs[2][rows]).to(device),
                torch.from_numpy(actions[rows]).to(device),
            ).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def critic_metrics(
    predicted_z: np.ndarray, true_cost: np.ndarray, target_mean: float,
    target_std: float,
) -> dict[str, Any]:
    true_log = np.log1p(true_cost.astype(np.float64))
    predicted_log = predicted_z.astype(np.float64) * target_std + target_mean
    true_best = np.argmin(true_cost, axis=1)
    selected = np.argmin(predicted_z, axis=1)
    rows = np.arange(len(true_cost))
    regret = true_cost[rows, selected] - true_cost[rows, true_best]
    available = true_cost[:, 0] - true_cost[rows, true_best]
    chosen_gain = true_cost[:, 0] - true_cost[rows, selected]
    material = available > 1e-5
    return {
        "pearson_log_cost": correlation(predicted_log, true_log),
        "spearman_log_cost": spearman(predicted_log, true_log),
        "standardized_rmse": float(np.sqrt(np.mean(
            (predicted_z - (true_log - target_mean) / target_std) ** 2
        ))),
        "warm_teacher_order_accuracy": float(np.mean(
            predicted_z[:, 0] > predicted_z[rows, true_best]
        )),
        "bank_top1_exact_fraction": float(np.mean(selected == true_best)),
        "bank_regret": distribution(regret),
        "bank_chosen_warm_relative_gain": distribution(chosen_gain),
        "bank_gain_recovery": float(
            np.sum(chosen_gain[material]) / np.sum(available[material])
        ) if np.any(material) else 0.0,
    }


def critic_selection_score(metrics: dict[str, Any]) -> float:
    return (
        metrics["standardized_rmse"]
        + 0.5 * (1.0 - metrics["pearson_log_cost"])
        + 0.5 * (1.0 - metrics["warm_teacher_order_accuracy"])
        + max(0.0, -metrics["bank_gain_recovery"])
    )


def train_critic(
    data: dict[str, np.ndarray], inputs: tuple[np.ndarray, ...], fit: np.ndarray,
    selection: np.ndarray, seed: int, args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(seed)
    critic = ConfigurableAbsoluteActionValueCritic().to(device)
    optimizer = torch.optim.AdamW(
        critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay
    )
    fit_log = np.log1p(data["costs"][fit].astype(np.float64))
    target_mean = float(fit_log.mean())
    target_std = float(max(fit_log.std(), 1e-6))
    target_z = ((np.log1p(data["costs"].astype(np.float64)) - target_mean) / target_std).astype(np.float32)
    rng = np.random.default_rng(907_501 + seed)
    candidates = data["actions"].shape[1]
    sample_count = min(args.critic_candidates_per_state, candidates)
    best_score = math.inf
    best_state = None
    best_epoch = 0
    history = []
    for epoch in range(1, args.critic_epochs + 1):
        critic.train()
        order = rng.permutation(fit)
        losses = []
        for start in range(0, len(order), args.critic_state_batch_size):
            rows = order[start : start + args.critic_state_batch_size]
            chosen = np.stack([
                np.concatenate((np.asarray((0,), np.int64), rng.choice(
                    np.arange(1, candidates), size=sample_count - 1, replace=False
                ))) for _ in rows
            ])
            action = torch.from_numpy(data["actions"][rows[:, None], chosen]).to(device)
            truth = torch.from_numpy(target_z[rows[:, None], chosen]).to(device)
            prediction = critic(
                torch.from_numpy(inputs[0][rows]).to(device),
                torch.from_numpy(inputs[1][rows]).to(device),
                torch.from_numpy(inputs[2][rows]).to(device), action,
            )
            value_loss = torch.nn.functional.smooth_l1_loss(prediction, truth)
            # Same-state random pairs constrain local ordering without changing
            # the scalar target or requiring an auxiliary pair head.
            left = torch.randint(sample_count, (len(rows), sample_count), device=device)
            right = torch.randint(sample_count, (len(rows), sample_count), device=device)
            batch_row = torch.arange(len(rows), device=device)[:, None]
            pred_delta = prediction[batch_row, left] - prediction[batch_row, right]
            true_delta = truth[batch_row, left] - truth[batch_row, right]
            material = true_delta.abs() > 1e-5
            if torch.any(material):
                ranking = torch.nn.functional.softplus(
                    -true_delta[material].sign() * pred_delta[material]
                    / args.ranking_temperature
                ).mean()
            else:
                ranking = prediction.sum() * 0.0
            loss = value_loss + args.ranking_weight * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % args.selection_stride == 0 or epoch == args.critic_epochs:
            predicted = critic_predict(critic, inputs, data["actions"], selection, device)
            metrics = critic_metrics(
                predicted, data["costs"][selection], target_mean, target_std
            )
            score = critic_selection_score(metrics)
            history.append({
                "epoch": epoch, "fit_loss": float(np.mean(losses)),
                "selection_score": score, "selection": metrics,
            })
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(critic.state_dict())
    if best_state is None:
        raise AssertionError("Critic checkpoint selection failed")
    critic.load_state_dict(best_state, strict=True)
    return critic, {
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "selection_history": history,
        "target_mean": target_mean,
        "target_std": target_std,
        "target_transform": "standardized_log1p_dbm_j50",
    }


def aggregate(records: list[dict[str, Any]], section: str, split: str) -> dict[str, Any]:
    values = [record[section][split] for record in records]
    if section == "actor":
        keys = (
            "aggregate_teacher_gain_recovery", "beats_or_equals_warm_fraction",
            "regression_fraction",
        )
    else:
        keys = (
            "pearson_log_cost", "spearman_log_cost", "standardized_rmse",
            "warm_teacher_order_accuracy", "bank_gain_recovery",
        )
    return {key: distribution(np.asarray([value[key] for value in values])) for key in keys}


def main() -> None:
    args = parse_args()
    replay_dir = args.replay_dir.resolve()
    teacher_dir = args.teacher_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    output.mkdir(parents=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    data = select_episode_subset(
        load_data(replay_dir, teacher_dir), args.episodes_per_stratum
    )
    folds = make_folds(data["episode"], data["speed"], data["scenario"])
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    device = torch.device(args.device)
    params = TorchMPPIParams(num_samples=64)
    weights = TorchMPPICostWeights()
    backend = TorchDynamicBicycleRolloutBackend()
    records: list[dict[str, Any]] = []
    for fold in range(args.folds):
        oof = np.flatnonzero(folds == fold)
        selection = np.flatnonzero(folds == ((fold + 1) % args.folds))
        fit = np.flatnonzero((folds != fold) & (folds != ((fold + 1) % args.folds)))
        fold_size = len(data["episode"]) // args.folds
        if (len(fit), len(selection), len(oof)) != (
            3 * fold_size, fold_size, fold_size,
        ):
            raise AssertionError("unexpected nested episode split")
        inputs, normalizer = normalized_inputs(data, fit)
        for seed in seeds:
            print(f"fold={fold} seed={seed}: Actor", flush=True)
            actor, actor_training = train_actor(
                data, inputs, fit, selection, seed, args,
                backend, weights, params, device,
            )
            actor_splits = {}
            for name, rows in (("fit", fit), ("selection", selection), ("oof", oof)):
                predicted = actor_predict(actor, inputs, rows, device)
                cost = rollout_cost(
                    data, predicted, rows, backend, weights, params, device,
                    args.rollout_batch_size,
                )
                actor_splits[name] = actor_metrics(
                    cost, data["anchor_cost"][rows], data["teacher_cost"][rows]
                )
            critics = []
            critic_training = []
            critic_splits = []
            for twin in range(2):
                critic_seed = seed * 100 + fold * 10 + twin + 10_000
                print(f"fold={fold} seed={seed}: Critic {twin + 1}", flush=True)
                critic, training = train_critic(
                    data, inputs, fit, selection, critic_seed, args, device
                )
                splits = {}
                for name, rows in (("fit", fit), ("selection", selection), ("oof", oof)):
                    predicted = critic_predict(
                        critic, inputs, data["actions"], rows, device
                    )
                    splits[name] = critic_metrics(
                        predicted, data["costs"][rows],
                        training["target_mean"], training["target_std"],
                    )
                critics.append(critic)
                critic_training.append(training)
                critic_splits.append(splits)
            # Twin conservative ranking uses max predicted cost after mapping
            # each model back to physical log-cost units.
            twin_splits = {}
            for name, rows in (("fit", fit), ("selection", selection), ("oof", oof)):
                physical = []
                for critic, training in zip(critics, critic_training):
                    predicted = critic_predict(
                        critic, inputs, data["actions"], rows, device
                    )
                    physical.append(
                        predicted * training["target_std"] + training["target_mean"]
                    )
                conservative_log = np.maximum(physical[0], physical[1])
                # critic_metrics accepts z plus affine; here z is already log.
                twin_splits[name] = critic_metrics(
                    conservative_log, data["costs"][rows], 0.0, 1.0
                )
            checkpoint = checkpoint_dir / f"pretrain_fold{fold}_seed{seed}.pt"
            torch.save({
                "qualification": "HIGHSPEED_ACTOR_TWIN_CRITIC_PRETRAIN_TRAIN_ONLY",
                "fold": fold, "seed": seed,
                "actor_architecture": "DirectNoAnchorGTXActor",
                "actor_state_dict": actor.state_dict(),
                "actor_training": actor_training,
                "critic_architecture": "ConfigurableAbsoluteActionValueCritic",
                "critic1_state_dict": critics[0].state_dict(),
                "critic2_state_dict": critics[1].state_dict(),
                "critic1_training": critic_training[0],
                "critic2_training": critic_training[1],
                "normalization": normalizer.to_dict(),
                "fit_indices": fit, "selection_indices": selection, "oof_indices": oof,
                "source_indices": data["source_indices"],
                "source_replay_sha256": str(data["replay_sha256"]),
                "source_teacher_sha256": str(data["teacher_sha256"]),
                "formal_validation_or_test_created": False,
            }, checkpoint)
            record = {
                "fold": fold, "seed": seed, "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256(checkpoint),
                "fit_episodes": sorted(np.unique(data["episode"][fit]).tolist()),
                "selection_episodes": sorted(np.unique(data["episode"][selection]).tolist()),
                "oof_episodes": sorted(np.unique(data["episode"][oof]).tolist()),
                "actor": actor_splits,
                "critic1": critic_splits[0], "critic2": critic_splits[1],
                "critic_twin_conservative": twin_splits,
            }
            records.append(record)
            print(
                f"fold={fold} seed={seed} OOF Actor R="
                f"{actor_splits['oof']['aggregate_teacher_gain_recovery']:.3f}; "
                f"Twin corr={twin_splits['oof']['pearson_log_cost']:.3f} "
                f"gainR={twin_splits['oof']['bank_gain_recovery']:.3f}",
                flush=True,
            )
    actor_oof_recovery = np.asarray([
        record["actor"]["oof"]["aggregate_teacher_gain_recovery"] for record in records
    ])
    actor_oof_p05 = np.asarray([
        record["actor"]["oof"]["warm_relative_gain"]["p05"] for record in records
    ])
    critic_corr = np.asarray([
        record["critic_twin_conservative"]["oof"]["pearson_log_cost"] for record in records
    ])
    critic_order = np.asarray([
        record["critic_twin_conservative"]["oof"]["warm_teacher_order_accuracy"]
        for record in records
    ])
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_ACTOR_TWIN_CRITIC_PRETRAIN_COMPLETE_TRAIN_ONLY",
        "source": {
            "replay_dir": str(replay_dir), "teacher_dir": str(teacher_dir),
            "replay_sha256": str(data["replay_sha256"]),
            "teacher_sha256": str(data["teacher_sha256"]),
        },
        "contract": {
            "state_count": int(len(data["episode"])),
            "episode_count": int(len(np.unique(data["episode"]))),
            "episodes_per_stratum": args.episodes_per_stratum,
            "source_indices": data["source_indices"].tolist(),
            "folds": args.folds, "seeds": seeds,
            "nested_split": (
                f"{3 * (len(data['episode']) // args.folds)} fit / "
                f"{len(data['episode']) // args.folds} selection / "
                f"{len(data['episode']) // args.folds} OOF, episode grouped"
            ),
            "actor": "no-anchor clean G-X absolute 8x2 center BC, +/-3 train-fold std support",
            "current_input": "[vx, yaw_rate, acceleration, steering]; no vy/beta",
            "critic": "twin absolute-action scalar value, standardized log1p deterministic DBM J50",
            "critic_candidate_count_per_state": 129,
            "td_bootstrap": False, "actor_critic_joint_updates": False,
            "formal_validation_or_test_created": False,
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "actor": {
            "fit": aggregate(records, "actor", "fit"),
            "oof": aggregate(records, "actor", "oof"),
            "registered_gate": {
                "recovery_median_at_least_0_50": bool(np.median(actor_oof_recovery) >= 0.50),
                "all_run_p05_nonnegative": bool(np.all(actor_oof_p05 >= 0.0)),
            },
        },
        "critic_twin_conservative": {
            "fit": aggregate(records, "critic_twin_conservative", "fit"),
            "oof": aggregate(records, "critic_twin_conservative", "oof"),
            "initialization_gate": {
                "pearson_median_at_least_0_50": bool(np.median(critic_corr) >= 0.50),
                "warm_teacher_order_median_at_least_0_80": bool(np.median(critic_order) >= 0.80),
            },
        },
        "records": records,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "qualification": summary["qualification"],
        "actor_oof_recovery": summary["actor"]["oof"]["aggregate_teacher_gain_recovery"],
        "critic_oof_pearson": summary["critic_twin_conservative"]["oof"]["pearson_log_cost"],
        "actor_gate": summary["actor"]["registered_gate"],
        "critic_gate": summary["critic_twin_conservative"]["initialization_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
