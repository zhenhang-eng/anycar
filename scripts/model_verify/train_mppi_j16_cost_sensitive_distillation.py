#!/usr/bin/env python3
"""Train an episode-balanced Actor from J16 targets and forward-cost curvature.

Training uses only train episodes.  Every source state contributes the same two
frozen first-pass contexts as the current Actor contract.  The target is the
reachable six-sigma projection of the best-found J16 center.  A Hadamard-basis
curvature loss is derived solely from saved forward costs; DBM gradients are not
used by the trainer.  Formal validation is loaded only after all Actors freeze.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
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
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots, sha256_file
from generate_dbm_j16_local_curvature_labels import hadamard_directions


DEFAULT_CHECKPOINT = Path(
    "outputs/mppi_proposal/direct_center_actor_trust_step_20260806_v2/"
    "direct_center_actor_trust_selected.pt"
)
OLD_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/fixed_dbm_policy_diverse_20260805_v1"
)
OLD_FEEDBACK = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_feedback_diverse_20260805_v1"
)
OLD_RISK = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_risk_replay_diverse_20260805_v1"
)
OLD_GT = Path("outputs/mppi_proposal/dbm_direct_gt_train_20260807_v2")
OLD_LOCAL = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_j16_local_curvature_train_diverse_20260807_v1"
)
OLD_PLAN = OLD_SOURCE / "scenario_plan.json"
NEW_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_train_expansion_20260807_v1"
)
NEW_CONTEXT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "mppi_first_pass_actor_context_train_expansion_20260807_v1"
)
NEW_GT = Path("outputs/mppi_proposal/dbm_direct_gt_train_expansion_20260807_v2")
NEW_LOCAL = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_j16_local_curvature_train_expansion_20260807_v1"
)
NEW_PLAN = Path("scripts/model_verify/fixed_dbm_policy_train_expansion_20260807_v1.json")
VALIDATION_GT = Path("outputs/mppi_proposal/dbm_direct_gt_validation_20260806_v2")
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/j16_cost_sensitive_expansion_20260807_v1"
)


@dataclass
class Dataset:
    inputs: tuple[np.ndarray, ...]
    target_action: np.ndarray
    target_center: np.ndarray
    target_cost: np.ndarray
    curvature_weight: np.ndarray | None
    episode_weight: np.ndarray
    episodes: np.ndarray
    snapshots: np.ndarray
    source_paths: list[Path]
    initial_state_six: np.ndarray
    current_action: np.ndarray
    direct_reference: np.ndarray
    sigma: np.ndarray
    reachable: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-delta-sigma", type=float, default=6.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--huber-beta", type=float, default=0.05)
    parser.add_argument("--curvature-weight", type=float, default=0.5)
    parser.add_argument("--selection-episodes-per-stratum", type=int, default=2)
    parser.add_argument("--evaluation-batch-size", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_actor(
    checkpoint: dict[str, Any], scale: float, device: torch.device
) -> TorchMPPIDeterministicCenterActor:
    actor = TorchMPPIDeterministicCenterActor(
        maximum_delta_sigma=scale, dropout=0.05
    ).to(device)
    state = copy.deepcopy(checkpoint["actor_state_dict"])
    state.pop("maximum_delta_sigma", None)
    incompatible = actor.load_state_dict(state, strict=False)
    if set(incompatible.missing_keys) != {"maximum_delta_sigma"}:
        raise AssertionError(f"unexpected missing Actor state: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        raise AssertionError(f"unexpected Actor state: {incompatible.unexpected_keys}")
    return actor


def gt_lookup(root: Path, expected_split: str) -> tuple[dict[tuple[str, str], dict], dict]:
    summary = json.loads((root / "summary.json").read_text())
    if summary["split"] != expected_split or "test" not in summary["test_policy"]:
        raise AssertionError(f"invalid GT split/policy: {root}")
    return {(row["episode"], row["snapshot"]): row for row in summary["rows"]}, summary


def load_partition(
    source_root: Path,
    context_root: Path,
    risk_root: Path | None,
    gt_root: Path,
    local_root: Path | None,
    episodes: list[str],
    checkpoint: dict[str, Any],
    scale: float,
    expected_split: str,
) -> tuple[Dataset, dict[str, Any]]:
    lookup, gt_summary = gt_lookup(gt_root, expected_split)
    normalization = MPPIProposalNormalization.from_dict(checkpoint["state_normalization"])
    episode_set = set(episodes)
    directions = hadamard_directions().reshape(16, 16).astype(np.float32) / 4.0
    fields: dict[str, list] = {name: [] for name in (
        "history", "reference", "current", "anchor", "feedback", "gradient",
        "target_action", "target_center", "target_cost", "curvature_weight",
        "episodes", "snapshots", "source_paths", "initial_state_six",
        "current_action", "direct_reference", "sigma", "reachable",
    )}
    for context_path in sorted(context_root.glob("episode_*/*.npz")):
        episode = context_path.parent.name
        if episode not in episode_set:
            continue
        snapshot = context_path.name
        key = (episode, snapshot)
        if key not in lookup:
            raise AssertionError(f"GT row missing {key}")
        source_path = source_root / episode / "snapshots" / snapshot
        gt_path = gt_root / episode / snapshot
        local_path = None if local_root is None else local_root / episode / snapshot
        risk_path = None if risk_root is None else risk_root / episode / snapshot
        with np.load(source_path, allow_pickle=False) as source, np.load(
            context_path, allow_pickle=False
        ) as context, np.load(gt_path, allow_pickle=False) as gt:
            source_hash = sha256_file(source_path)
            if str(context["source_snapshot_sha256"]) != source_hash:
                raise AssertionError(f"context/source hash mismatch: {context_path}")
            best = int(gt["knot_best_index"])
            oracle = np.asarray(gt["optimized_knots"][best], np.float32)
            anchors = np.asarray(context["guided_center_knots"], np.float32)
            feedback = np.asarray(context["first_pass_feedback"], np.float32)
            if risk_path is None:
                gradient = np.concatenate((
                    np.asarray(context["critic_gradient_mean"], np.float32),
                    np.asarray(context["critic_gradient_std"], np.float32),
                ), axis=1)
            else:
                with np.load(risk_path, allow_pickle=False) as risk:
                    gradient = np.concatenate((
                        np.asarray(risk["critic_gradient_mean"], np.float32),
                        np.asarray(risk["critic_gradient_std"], np.float32),
                    ), axis=1)
            params = json.loads(str(source["mppi_params_json"]))
            sigma = np.asarray(params["noise_sigma"], np.float32)
            low = np.asarray(params["action_min"], np.float32)
            high = np.asarray(params["action_max"], np.float32)
            requested = (oracle[None] - anchors) / (scale * sigma.reshape(1, 1, 2))
            target_action = np.clip(requested, -1.0, 1.0).astype(np.float32)
            target_center = np.clip(
                anchors + target_action * scale * sigma.reshape(1, 1, 2), low, high
            ).astype(np.float32)
            target_action = ((target_center - anchors) / (
                scale * sigma.reshape(1, 1, 2)
            )).astype(np.float32)
            reachable = bool(np.all(np.abs(requested) <= 1.0 + 1e-6))
            if local_path is not None:
                with np.load(local_path, allow_pickle=False) as local:
                    curvature = np.maximum(
                        np.asarray(local["directional_curvature"], np.float32)[0], 0.0
                    )
                    symmetric = np.asarray(local["symmetric_pair_mask"], bool)[0]
                weight = np.log1p(curvature) * symmetric.astype(np.float32) + 0.05
                weight /= np.maximum(np.mean(weight), 1e-6)
            else:
                weight = np.ones(16, np.float32)
            state = np.asarray(source["initial_state"], np.float32)
            action = np.asarray(source["current_action"], np.float32)
            history = np.asarray(source["history"][0], np.float32)
            reference = ego_reference_features(source["reference_ego"], float(state[3]))
            current = np.asarray((state[3], state[4], *action), np.float32)
            direct_reference = np.asarray(source["reference"], np.float32)
            if len(direct_reference) == int(params["horizon"]) + 1:
                direct_reference = direct_reference[1:]
            for repeat in range(len(anchors)):
                fields["history"].append(history)
                fields["reference"].append(reference)
                fields["current"].append(current)
                fields["anchor"].append(anchors[repeat])
                fields["feedback"].append(feedback[repeat])
                fields["gradient"].append(gradient[repeat])
                fields["target_action"].append(target_action[repeat])
                fields["target_center"].append(target_center[repeat])
                fields["target_cost"].append(float(lookup[key]["j16_best_found"]))
                fields["curvature_weight"].append(weight)
                fields["episodes"].append(episode)
                fields["snapshots"].append(snapshot)
                fields["source_paths"].append(source_path)
                fields["initial_state_six"].append(np.asarray(source["initial_state_six"], np.float32))
                fields["current_action"].append(action)
                fields["direct_reference"].append(direct_reference)
                fields["sigma"].append(sigma)
                fields["reachable"].append(reachable)
    if not fields["history"]:
        raise ValueError("empty partition")
    history, reference, current = normalization.normalize_numpy(
        np.asarray(fields["history"], np.float32),
        np.asarray(fields["reference"], np.float32),
        np.asarray(fields["current"], np.float32),
    )
    inputs = (
        history.astype(np.float32),
        reference.astype(np.float32),
        current.astype(np.float32),
        np.asarray(fields["anchor"], np.float32),
        ((np.asarray(fields["feedback"], np.float32) - checkpoint["feedback_mean"])
         / checkpoint["feedback_std"]).astype(np.float32),
        ((np.asarray(fields["gradient"], np.float32) - checkpoint["gradient_mean"])
         / checkpoint["gradient_std"]).astype(np.float32),
    )
    episode_array = np.asarray(fields["episodes"])
    unique, counts = np.unique(episode_array, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    episode_weight = np.asarray([1.0 / count_map[value] for value in episode_array], np.float32)
    episode_weight /= episode_weight.mean()
    dataset = Dataset(
        inputs=inputs,
        target_action=np.asarray(fields["target_action"], np.float32),
        target_center=np.asarray(fields["target_center"], np.float32),
        target_cost=np.asarray(fields["target_cost"], np.float32),
        curvature_weight=np.asarray(fields["curvature_weight"], np.float32),
        episode_weight=episode_weight,
        episodes=episode_array,
        snapshots=np.asarray(fields["snapshots"]),
        source_paths=list(fields["source_paths"]),
        initial_state_six=np.asarray(fields["initial_state_six"], np.float32),
        current_action=np.asarray(fields["current_action"], np.float32),
        direct_reference=np.asarray(fields["direct_reference"], np.float32),
        sigma=np.asarray(fields["sigma"], np.float32),
        reachable=np.asarray(fields["reachable"], bool),
    )
    return dataset, gt_summary


def concatenate(parts: list[Dataset]) -> Dataset:
    inputs = tuple(np.concatenate([part.inputs[i] for part in parts]) for i in range(6))
    episodes = np.concatenate([part.episodes for part in parts])
    unique, counts = np.unique(episodes, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    episode_weight = np.asarray(
        [1.0 / count_map[value] for value in episodes], np.float32
    )
    episode_weight /= episode_weight.mean()
    return Dataset(
        inputs=inputs,
        target_action=np.concatenate([part.target_action for part in parts]),
        target_center=np.concatenate([part.target_center for part in parts]),
        target_cost=np.concatenate([part.target_cost for part in parts]),
        curvature_weight=np.concatenate([part.curvature_weight for part in parts]),
        episode_weight=episode_weight,
        episodes=episodes,
        snapshots=np.concatenate([part.snapshots for part in parts]),
        source_paths=sum((part.source_paths for part in parts), []),
        initial_state_six=np.concatenate([part.initial_state_six for part in parts]),
        current_action=np.concatenate([part.current_action for part in parts]),
        direct_reference=np.concatenate([part.direct_reference for part in parts]),
        sigma=np.concatenate([part.sigma for part in parts]),
        reachable=np.concatenate([part.reachable for part in parts]),
    )


def combined_episode_split(count: int) -> tuple[list[str], list[str]]:
    groups: dict[tuple[float, str], list[str]] = {}
    for plan_path in (OLD_PLAN, NEW_PLAN):
        plan = json.loads(plan_path.read_text())
        for row in plan["episodes"]:
            if row["split"] != "train":
                continue
            key = (float(row["reference_speed_mps"]), str(row["scenario_class"]))
            groups.setdefault(key, []).append(str(row["episode_id"]))
    fit, selection = [], []
    for key, episodes in sorted(groups.items()):
        episodes.sort()
        if len(episodes) != 12:
            raise AssertionError(f"combined stratum {key} has {len(episodes)} episodes")
        fit.extend(episodes[:-count])
        selection.extend(episodes[-count:])
    return fit, selection


def batch_inputs(inputs: tuple[np.ndarray, ...], index: np.ndarray, device: torch.device):
    return tuple(torch.from_numpy(value[index]).to(device) for value in inputs)


@torch.no_grad()
def fit_metrics(
    actor: TorchMPPIDeterministicCenterActor,
    data: Dataset,
    index: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    predictions = []
    actor.eval()
    for start in range(0, len(index), batch_size):
        one = index[start:start + batch_size]
        action, _ = actor(*batch_inputs(data.inputs, one, device))
        predictions.append(action.cpu().numpy())
    prediction = np.concatenate(predictions)
    target = data.target_action[index]
    error = prediction - target
    weight = data.episode_weight[index]
    mse = np.mean(error ** 2, axis=(1, 2))
    weighted_mse = float(np.sum(weight * mse) / np.sum(weight))
    directions = hadamard_directions().reshape(16, 16).astype(np.float32) / 4.0
    projection = error.reshape(len(error), 16) @ directions.T
    curvature = data.curvature_weight[index]
    curvature_mse = np.mean(projection ** 2 * curvature, axis=1)
    weighted_curvature_mse = float(
        np.sum(weight * curvature_mse) / np.sum(weight)
    )
    return {
        "action_rmse": float(np.sqrt(np.mean(error ** 2))),
        "episode_balanced_action_rmse": float(np.sqrt(weighted_mse)),
        "episode_balanced_curvature_rmse": float(
            np.sqrt(weighted_curvature_mse)
        ),
        "action_mae": float(np.mean(np.abs(error))),
    }


def train_one(
    seed: int,
    checkpoint: dict[str, Any],
    data: Dataset,
    fit_index: np.ndarray,
    selection_index: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TorchMPPIDeterministicCenterActor, list[dict[str, float]], int]:
    set_seed(seed)
    actor = load_actor(checkpoint, args.maximum_delta_sigma, device)
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.minimum_learning_rate
    )
    directions = torch.from_numpy(hadamard_directions().reshape(16, 16) / 4.0).to(device)
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        actor.train()
        order = rng.permutation(fit_index)
        losses = []
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            prediction, _ = actor(*batch_inputs(data.inputs, index, device))
            target = torch.from_numpy(data.target_action[index]).to(device)
            sample_weight = torch.from_numpy(data.episode_weight[index]).to(device)
            base = F.smooth_l1_loss(
                prediction, target, beta=args.huber_beta, reduction="none"
            ).mean(dim=(1, 2))
            error = (prediction - target).flatten(1)
            projection = error @ directions.T
            curvature = torch.from_numpy(data.curvature_weight[index]).to(device)
            local = F.smooth_l1_loss(
                projection, torch.zeros_like(projection),
                beta=args.huber_beta, reduction="none",
            )
            local = (local * curvature).mean(1)
            per_sample = base + args.curvature_weight * local
            loss = torch.sum(per_sample * sample_weight) / torch.sum(sample_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        selection = fit_metrics(
            actor, data, selection_index, args.evaluation_batch_size, device
        )
        score = selection["episode_balanced_curvature_rmse"]
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(actor.state_dict())
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            train = fit_metrics(actor, data, fit_index, args.evaluation_batch_size, device)
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **{f"fit_{k}": v for k, v in train.items()},
                **{f"selection_{k}": v for k, v in selection.items()},
            }
            history.append(row)
            print(
                f"[seed={seed} epoch={epoch:03d}] loss={row['train_loss']:.5f} "
                f"fit={train['action_rmse']:.4f} select={selection['action_rmse']:.4f} "
                f"lr={row['learning_rate']:.2g}",
                flush=True,
            )
    if best_state is None:
        raise AssertionError("no selected Actor state")
    # The held-out episodes select only the epoch.  Refit from the same frozen
    # parent on all train episodes for that many updates so the final Actor does
    # not discard 60 independent trajectories merely to perform model selection.
    set_seed(seed)
    actor = load_actor(checkpoint, args.maximum_delta_sigma, device)
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.minimum_learning_rate
    )
    all_index = np.arange(len(data.target_action), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for epoch in range(1, best_epoch + 1):
        actor.train()
        order = rng.permutation(all_index)
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            prediction, _ = actor(*batch_inputs(data.inputs, index, device))
            target = torch.from_numpy(data.target_action[index]).to(device)
            sample_weight = torch.from_numpy(data.episode_weight[index]).to(device)
            base = F.smooth_l1_loss(
                prediction, target, beta=args.huber_beta, reduction="none"
            ).mean(dim=(1, 2))
            error = (prediction - target).flatten(1)
            projection = error @ directions.T
            curvature = torch.from_numpy(data.curvature_weight[index]).to(device)
            local = F.smooth_l1_loss(
                projection, torch.zeros_like(projection),
                beta=args.huber_beta, reduction="none",
            )
            local = (local * curvature).mean(1)
            per_sample = base + args.curvature_weight * local
            loss = torch.sum(per_sample * sample_weight) / torch.sum(sample_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % 50 == 0 or epoch == best_epoch:
            metrics = fit_metrics(
                actor, data, all_index, args.evaluation_batch_size, device
            )
            print(
                f"[seed={seed} refit={epoch:03d}/{best_epoch:03d}] "
                f"all_train={metrics['action_rmse']:.4f}",
                flush=True,
            )
    actor.eval()
    return actor, history, best_epoch


@torch.no_grad()
def actor_centers(
    actor: TorchMPPIDeterministicCenterActor,
    data: Dataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    values = []
    actor.eval()
    for start in range(0, len(data.target_action), batch_size):
        index = np.arange(start, min(start + batch_size, len(data.target_action)))
        _, center = actor(*batch_inputs(data.inputs, index, device))
        values.append(center.cpu().numpy())
    return np.concatenate(values)


def evaluate_direct(
    methods: dict[str, np.ndarray], data: Dataset, batch_size: int, device: torch.device
) -> dict[str, np.ndarray]:
    first_source = data.source_paths[0]
    with np.load(first_source, allow_pickle=False) as source:
        params = TorchMPPIParams(**json.loads(str(source["mppi_params_json"])))
        weights = TorchMPPICostWeights(**json.loads(str(source["cost_weights_json"])))
        backend = TorchDynamicBicycleRolloutBackend(
            TorchDBMParams(**json.loads(str(source["dbm_params_json"])))
        )
    names = tuple(methods)
    output = {name: [] for name in names}
    for start in range(0, len(data.target_action), batch_size):
        stop = min(start + batch_size, len(data.target_action))
        centers = np.stack([methods[name][start:stop] for name in names], axis=1)
        center_t = torch.from_numpy(centers.astype(np.float32)).to(device)
        actions = interpolate_knots(center_t, params.horizon)
        initial = torch.from_numpy(data.initial_state_six[start:stop]).to(device)
        current = torch.from_numpy(data.current_action[start:stop]).to(device)
        reference = torch.from_numpy(data.direct_reference[start:stop]).to(device)
        cost = batched_cost(
            backend, weights, actions, initial, current, reference
        ).cpu().numpy()
        for index, name in enumerate(names):
            output[name].append(cost[:, index])
    return {name: np.concatenate(value) for name, value in output.items()}


def distribution(cost: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(cost)),
        "median": float(np.median(cost)),
        "p95": float(np.quantile(cost, 0.95)),
        "maximum": float(np.max(cost)),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    old_splits = json.loads((OLD_FEEDBACK / "splits.json").read_text())
    new_splits = json.loads((NEW_CONTEXT / "splits.json").read_text())
    old_train, _ = load_partition(
        OLD_SOURCE, OLD_FEEDBACK, OLD_RISK, OLD_GT, OLD_LOCAL,
        old_splits["train"], checkpoint, args.maximum_delta_sigma, "train",
    )
    new_train, _ = load_partition(
        NEW_SOURCE, NEW_CONTEXT, None, NEW_GT, NEW_LOCAL,
        new_splits["train"], checkpoint, args.maximum_delta_sigma, "train",
    )
    train = concatenate([old_train, new_train])
    fit_episodes, selection_episodes = combined_episode_split(
        args.selection_episodes_per_stratum
    )
    fit_index = np.flatnonzero(np.isin(train.episodes, fit_episodes))
    selection_index = np.flatnonzero(np.isin(train.episodes, selection_episodes))
    if len(fit_index) + len(selection_index) != len(train.target_action):
        raise AssertionError("internal episode split does not cover training contexts")
    trained = {}
    training_summary = []
    for seed in args.seeds:
        actor, history, best_epoch = train_one(
            seed, checkpoint, train, fit_index, selection_index, args, device
        )
        trained[f"actor_seed{seed}"] = copy.deepcopy(actor).cpu()
        payload = {
            "format_version": 1,
            "method": "episode-balanced cost-sensitive J16 distillation",
            "qualification": "VALIDATION_ONLY_TEST_SEALED",
            "maximum_delta_sigma": args.maximum_delta_sigma,
            "seed": seed,
            "selected_epoch": best_epoch,
            "refit_all_train_episodes": True,
            "actor_state_dict": trained[f"actor_seed{seed}"].state_dict(),
            "state_normalization": checkpoint["state_normalization"],
            "feedback_mean": checkpoint["feedback_mean"],
            "feedback_std": checkpoint["feedback_std"],
            "gradient_mean": checkpoint["gradient_mean"],
            "gradient_std": checkpoint["gradient_std"],
            "source_checkpoint": str(args.checkpoint),
            "source_checkpoint_sha256": sha256_file(args.checkpoint),
            "fit_episodes": fit_episodes,
            "selection_episodes": selection_episodes,
            "training_history": history,
            "test_policy": "test split not loaded or evaluated",
        }
        torch.save(payload, args.output_dir / f"actor_seed{seed}.pt")
        training_summary.append({
            "seed": seed,
            "selected_epoch": best_epoch,
            "refit_all_train_episodes": True,
            "history": history,
        })

    # Formal validation is opened only after every trained Actor is frozen.
    validation, validation_gt = load_partition(
        OLD_SOURCE, OLD_FEEDBACK, OLD_RISK, VALIDATION_GT, None,
        old_splits["validation"], checkpoint, args.maximum_delta_sigma, "validation",
    )
    current_actor = load_actor(checkpoint, float(checkpoint["maximum_delta_sigma"]), device)
    methods = {
        "current_actor": actor_centers(
            current_actor, validation, args.evaluation_batch_size, device
        ),
        "projected_j16": validation.target_center,
    }
    for name, actor_cpu in trained.items():
        methods[name] = actor_centers(
            actor_cpu.to(device), validation, args.evaluation_batch_size, device
        )
        actor_cpu.cpu()
    direct = evaluate_direct(methods, validation, args.evaluation_batch_size, device)
    results = {}
    for name, cost in direct.items():
        results[name] = {
            **distribution(cost),
            "gap_to_j16_mean": float(np.mean(cost - validation.target_cost)),
            **(
                fit_metrics(
                    trained[name].to(device), validation,
                    np.arange(len(validation.target_action)),
                    args.evaluation_batch_size, device,
                )
                if name in trained else {}
            ),
        }
        if name in trained:
            trained[name].cpu()
    summary = {
        "format_version": 1,
        "method": "episode-balanced forward-cost-sensitive J16 Actor distillation",
        "qualification": "VALIDATION_ONLY_TEST_SEALED",
        "train_snapshots": int(len(train.target_action) // 2),
        "train_contexts": int(len(train.target_action)),
        "fit_episodes": len(fit_episodes),
        "selection_episodes": len(selection_episodes),
        "fit_contexts": len(fit_index),
        "selection_contexts": len(selection_index),
        "target_reachable_fraction": float(np.mean(train.reachable)),
        "curvature_weight": args.curvature_weight,
        "epochs": args.epochs,
        "seeds": args.seeds,
        "training": training_summary,
        "validation": {
            "episodes": old_splits["validation"],
            "snapshots": validation_gt["snapshot_count"],
            "contexts": len(validation.target_action),
            "j16": distribution(validation.target_cost),
            "methods": results,
        },
        "test_policy": "test split not loaded or evaluated",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["validation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
