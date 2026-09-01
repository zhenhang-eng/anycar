#!/usr/bin/env python3
"""Train an episode-heldout MPPI proposal cost/ranking critic from T1 centers."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TorchMPPIProposalCritic,
    ego_reference_features,
)
from evaluate_mppi_proposal_bc import load_policy, predict_center


DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1"
)
DEFAULT_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_teacher_t1_diverse_20260805_v1"
)
DEFAULT_ACTOR = Path(
    "outputs/mppi_proposal/bc_t1_conv_diverse_20260805_v1/trust1_seed1.pt"
)
DEFAULT_FRESH_EVAL = Path(
    "outputs/mppi_proposal/bc_t1_conv_diverse_20260805_v1/"
    "fresh_test_eval/per_snapshot.csv"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/critic_t1_diverse_20260805_v1")


@dataclass
class CriticArrays:
    history: np.ndarray
    reference: np.ndarray
    current: np.ndarray
    warm: np.ndarray
    centers: np.ndarray
    center_mask: np.ndarray
    costs: np.ndarray
    cost_seed_std: np.ndarray
    warm_index: np.ndarray
    network_index: np.ndarray
    teacher_index: np.ndarray
    episode_ids: list[str]
    control_steps: np.ndarray

    @property
    def advantage(self) -> np.ndarray:
        warm_cost = np.take_along_axis(
            self.costs, self.warm_index[:, None], axis=1
        )
        return warm_cost - self.costs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--actor-checkpoint", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--fresh-eval-csv", type=Path, default=DEFAULT_FRESH_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cost-key",
        default="proposal_weighted_output_cost",
        help="Per-center/per-seed NPZ field used as the training target.",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--rank-weight", type=float, default=0.25)
    parser.add_argument("--anchor-weight", type=float, default=0.10)
    parser.add_argument("--rank-tie-cost", type=float, default=0.10)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(
    source_root: Path,
    label_root: Path,
    episodes: list[str],
    cost_key: str = "proposal_weighted_output_cost",
) -> CriticArrays:
    records: list[dict[str, Any]] = []
    maximum_centers = 0
    episode_set = set(episodes)
    for label_path in sorted(label_root.glob("episode_*/*.npz")):
        episode_id = label_path.parent.name
        if episode_id not in episode_set:
            continue
        source_path = source_root / episode_id / "snapshots" / label_path.name
        with np.load(source_path, allow_pickle=False) as source, np.load(
            label_path, allow_pickle=False
        ) as label:
            state = np.asarray(source["initial_state"], dtype=np.float32)
            action = np.asarray(source["current_action"], dtype=np.float32)
            centers = np.asarray(label["shortlist_centers"], dtype=np.float32)
            if cost_key not in label:
                raise KeyError(f"{label_path}: missing {cost_key}")
            seed_cost = np.asarray(label[cost_key], dtype=np.float32)
            if seed_cost.shape[0] != centers.shape[0] or seed_cost.ndim != 2:
                raise ValueError(f"{label_path}: invalid center/cost shape")
            maximum_centers = max(maximum_centers, len(centers))
            records.append(
                {
                    "history": np.asarray(source["history"][0], dtype=np.float32),
                    "reference": ego_reference_features(
                        source["reference_ego"], float(state[3])
                    ),
                    "current": np.asarray(
                        (state[3], state[4], *action), dtype=np.float32
                    ),
                    "warm": np.asarray(
                        source["sampling_mean_knots"], dtype=np.float32
                    ),
                    "centers": centers,
                    "costs": seed_cost.mean(axis=1),
                    "cost_seed_std": seed_cost.std(axis=1),
                    "warm_index": int(label["warm_shortlist_index"]),
                    "network_index": int(label["network_shortlist_index"])
                    if "network_shortlist_index" in label
                    else -1,
                    "teacher_index": int(label["teacher_shortlist_index"]),
                    "episode_id": episode_id,
                    "control_step": int(source["control_step"]),
                }
            )
    if not records:
        raise ValueError(f"no critic records found for episodes {episodes}")

    count = len(records)
    centers = np.zeros((count, maximum_centers, 8, 2), dtype=np.float32)
    costs = np.zeros((count, maximum_centers), dtype=np.float32)
    cost_seed_std = np.zeros_like(costs)
    mask = np.zeros((count, maximum_centers), dtype=bool)
    for index, record in enumerate(records):
        center_count = len(record["centers"])
        centers[index, :center_count] = record["centers"]
        costs[index, :center_count] = record["costs"]
        cost_seed_std[index, :center_count] = record["cost_seed_std"]
        mask[index, :center_count] = True
        if center_count < maximum_centers:
            centers[index, center_count:] = record["warm"]
            costs[index, center_count:] = record["costs"][record["warm_index"]]

    return CriticArrays(
        history=np.asarray([row["history"] for row in records], dtype=np.float32),
        reference=np.asarray(
            [row["reference"] for row in records], dtype=np.float32
        ),
        current=np.asarray([row["current"] for row in records], dtype=np.float32),
        warm=np.asarray([row["warm"] for row in records], dtype=np.float32),
        centers=centers,
        center_mask=mask,
        costs=costs,
        cost_seed_std=cost_seed_std,
        warm_index=np.asarray(
            [row["warm_index"] for row in records], dtype=np.int64
        ),
        network_index=np.asarray(
            [row["network_index"] for row in records], dtype=np.int64
        ),
        teacher_index=np.asarray(
            [row["teacher_index"] for row in records], dtype=np.int64
        ),
        episode_ids=[row["episode_id"] for row in records],
        control_steps=np.asarray(
            [row["control_step"] for row in records], dtype=np.int64
        ),
    )


class CriticDataset(Dataset):
    def __init__(
        self,
        arrays: CriticArrays,
        normalization: MPPIProposalNormalization,
        advantage_scale: float,
    ) -> None:
        history, reference, current = normalization.normalize_numpy(
            arrays.history, arrays.reference, arrays.current
        )
        self.history = torch.from_numpy(history.astype(np.float32))
        self.reference = torch.from_numpy(reference.astype(np.float32))
        self.current = torch.from_numpy(current.astype(np.float32))
        self.warm = torch.from_numpy(arrays.warm)
        self.centers = torch.from_numpy(arrays.centers)
        self.mask = torch.from_numpy(arrays.center_mask)
        self.target = torch.from_numpy(
            (arrays.advantage / advantage_scale).astype(np.float32)
        )
        self.warm_index = torch.from_numpy(arrays.warm_index)

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.history[index],
            self.reference[index],
            self.current[index],
            self.warm[index],
            self.centers[index],
            self.mask[index],
            self.target[index],
            self.warm_index[index],
        )


def predict_center_bank(
    model: TorchMPPIProposalCritic,
    history: torch.Tensor,
    reference: torch.Tensor,
    current: torch.Tensor,
    warm: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    batch, center_count = centers.shape[:2]
    return model(
        history[:, None].expand(-1, center_count, -1, -1).reshape(
            batch * center_count, *history.shape[1:]
        ),
        reference[:, None].expand(-1, center_count, -1, -1).reshape(
            batch * center_count, *reference.shape[1:]
        ),
        current[:, None].expand(-1, center_count, -1).reshape(
            batch * center_count, current.shape[-1]
        ),
        warm[:, None].expand(-1, center_count, -1, -1).reshape(
            batch * center_count, *warm.shape[1:]
        ),
        centers.reshape(batch * center_count, *centers.shape[2:]),
    ).reshape(batch, center_count)


def critic_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    warm_index: torch.Tensor,
    rank_weight: float,
    anchor_weight: float,
    normalized_tie: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    regression = F.smooth_l1_loss(
        prediction[mask], target[mask], beta=0.5, reduction="mean"
    )
    target_difference = target[:, :, None] - target[:, None, :]
    prediction_difference = prediction[:, :, None] - prediction[:, None, :]
    center_count = prediction.shape[1]
    upper = torch.triu(
        torch.ones(center_count, center_count, dtype=torch.bool, device=mask.device),
        diagonal=1,
    )
    pair_mask = (
        mask[:, :, None]
        & mask[:, None, :]
        & upper[None]
        & (target_difference.abs() > normalized_tie)
    )
    if pair_mask.any():
        sign = target_difference[pair_mask].sign()
        difference = prediction_difference[pair_mask]
        pair_weight = target_difference[pair_mask].abs().clamp(max=2.0)
        ranking = (F.softplus(-sign * difference) * pair_weight).sum() / pair_weight.sum()
    else:
        ranking = prediction.new_zeros(())
    warm_prediction = prediction.gather(1, warm_index[:, None]).squeeze(1)
    anchor = warm_prediction.square().mean()
    total = regression + rank_weight * ranking + anchor_weight * anchor
    return total, {
        "regression": float(regression.detach()),
        "ranking": float(ranking.detach()),
        "anchor": float(anchor.detach()),
    }


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def correlation(x: np.ndarray, y: np.ndarray, ranked: bool = False) -> float:
    if ranked:
        x, y = rank_values(x), rank_values(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_predictions(
    arrays: CriticArrays, predicted_advantage: np.ndarray, tie_cost: float
) -> dict[str, Any]:
    target = arrays.advantage
    mask = arrays.center_mask
    error = predicted_advantage[mask] - target[mask]
    target_flat = target[mask]
    prediction_flat = predicted_advantage[mask]
    correct = 0
    pairs = 0
    for state in range(len(target)):
        indices = np.flatnonzero(mask[state])
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                target_delta = target[state, left] - target[state, right]
                if abs(target_delta) <= tie_cost:
                    continue
                prediction_delta = (
                    predicted_advantage[state, left]
                    - predicted_advantage[state, right]
                )
                correct += int(np.sign(target_delta) == np.sign(prediction_delta))
                pairs += 1
    masked_prediction = np.where(mask, predicted_advantage, -np.inf)
    masked_target = np.where(mask, target, -np.inf)
    selected = np.argmax(masked_prediction, axis=1)
    oracle = np.argmax(masked_target, axis=1)
    selected_advantage = target[np.arange(len(target)), selected]
    best_advantage = target[np.arange(len(target)), oracle]
    regret = best_advantage - selected_advantage
    teacher_prediction = predicted_advantage[
        np.arange(len(target)), arrays.teacher_index
    ]
    warm_prediction = predicted_advantage[
        np.arange(len(target)), arrays.warm_index
    ]
    teacher_target = target[np.arange(len(target)), arrays.teacher_index]
    metrics = {
        "state_count": len(target),
        "center_count": int(mask.sum()),
        "advantage_mae": float(np.mean(np.abs(error))),
        "advantage_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pearson": correlation(target_flat, prediction_flat),
        "spearman": correlation(target_flat, prediction_flat, ranked=True),
        "pairwise_accuracy": float(correct / max(pairs, 1)),
        "pairwise_count": pairs,
        "top1_accuracy": float(np.mean(selected == oracle)),
        "selection_regret": {
            "mean": float(np.mean(regret)),
            "median": float(np.median(regret)),
            "p90": float(np.quantile(regret, 0.9)),
            "maximum": float(np.max(regret)),
        },
        "selected_true_advantage": {
            "mean": float(np.mean(selected_advantage)),
            "median": float(np.median(selected_advantage)),
            "wins": int(np.sum(selected_advantage > 1e-6)),
            "ties": int(np.sum(np.abs(selected_advantage) <= 1e-6)),
            "losses": int(np.sum(selected_advantage < -1e-6)),
        },
        "teacher_vs_warm_accuracy": float(
            np.mean(
                np.sign(teacher_prediction - warm_prediction)
                == np.sign(teacher_target)
            )
        ),
        "teacher_preferred_count": int(
            np.sum(teacher_prediction > warm_prediction)
        ),
        "selected_teacher_count": int(np.sum(selected == arrays.teacher_index)),
        "selected_warm_count": int(np.sum(selected == arrays.warm_index)),
        "oracle_advantage_mean": float(np.mean(best_advantage)),
        "warm_cost_mean": float(
            np.mean(
                arrays.costs[np.arange(len(target)), arrays.warm_index]
            )
        ),
        "selected_cost_mean": float(
            np.mean(
                arrays.costs[np.arange(len(target)), arrays.warm_index]
                - selected_advantage
            )
        ),
    }
    if np.all(arrays.network_index >= 0):
        network_prediction = predicted_advantage[
            np.arange(len(target)), arrays.network_index
        ]
        network_target = target[np.arange(len(target)), arrays.network_index]
        metrics.update(
            {
                "network_vs_warm_accuracy": float(
                    np.mean(
                        np.sign(network_prediction - warm_prediction)
                        == np.sign(network_target)
                    )
                ),
                "teacher_vs_network_accuracy": float(
                    np.mean(
                        np.sign(teacher_prediction - network_prediction)
                        == np.sign(teacher_target - network_target)
                    )
                ),
                "selected_network_count": int(
                    np.sum(selected == arrays.network_index)
                ),
            }
        )
    return metrics


@torch.no_grad()
def predict_dataset(
    models: list[TorchMPPIProposalCritic],
    dataset: CriticDataset,
    device: torch.device,
    advantage_scale: float,
    batch_size: int = 64,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_predictions = []
    for batch in loader:
        history, reference, current, warm, centers, _, _, _ = (
            value.to(device) for value in batch
        )
        prediction = torch.stack(
            [
                predict_center_bank(
                    model, history, reference, current, warm, centers
                )
                for model in models
            ]
        ).mean(dim=0)
        all_predictions.append(prediction.cpu().numpy() * advantage_scale)
    return np.concatenate(all_predictions)


def initialize_from_actor(
    model: TorchMPPIProposalCritic, actor_checkpoint: dict[str, Any]
) -> list[str]:
    actor_state = actor_checkpoint["model_state_dict"]
    critic_state = model.state_dict()
    prefixes = (
        "history_encoder.",
        "reference_encoder.",
        "current_encoder.",
        "warm_encoder.",
    )
    loaded = {
        name: value
        for name, value in actor_state.items()
        if name.startswith(prefixes)
        and name in critic_state
        and critic_state[name].shape == value.shape
    }
    model.load_state_dict(loaded, strict=False)
    return sorted(loaded)


def train_one(
    seed: int,
    args: argparse.Namespace,
    datasets: dict[str, CriticDataset],
    arrays: dict[str, CriticArrays],
    actor_checkpoint: dict[str, Any],
    advantage_scale: float,
    normalization: MPPIProposalNormalization,
) -> tuple[dict[str, Any], dict[str, Any]]:
    set_seed(seed)
    device = torch.device(args.device)
    model = TorchMPPIProposalCritic(dropout=args.dropout).to(device)
    loaded_encoder_keys = initialize_from_actor(model, actor_checkpoint)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        datasets["validation"], batch_size=args.batch_size, shuffle=False
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=35, min_lr=1e-6
    )
    normalized_tie = args.rank_tie_cost / advantage_scale
    best_validation = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            history, reference, current, warm, centers, mask, target, warm_index = (
                value.to(device) for value in batch
            )
            prediction = predict_center_bank(
                model, history, reference, current, warm, centers
            )
            loss, _ = critic_loss(
                prediction,
                target,
                mask,
                warm_index,
                args.rank_weight,
                args.anchor_weight,
                normalized_tie,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in validation_loader:
                history, reference, current, warm, centers, mask, target, warm_index = (
                    value.to(device) for value in batch
                )
                prediction = predict_center_bank(
                    model, history, reference, current, warm, centers
                )
                loss, _ = critic_loss(
                    prediction,
                    target,
                    mask,
                    warm_index,
                    args.rank_weight,
                    args.anchor_weight,
                    normalized_tie,
                )
                validation_losses.append(float(loss))
        validation_loss = float(np.mean(validation_losses))
        scheduler.step(validation_loss)
        if validation_loss < best_validation - 1e-6:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0:
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(train_losses)),
                    "validation_loss": validation_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
        if stale >= args.patience:
            break
    if best_state is None:
        raise AssertionError("critic training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    predictions = {
        split: predict_dataset(
            [model], datasets[split], device, advantage_scale, args.batch_size
        )
        for split in ("train", "validation", "test")
    }
    metrics = {
        "seed": seed,
        "parameter_count": model.parameter_count,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_validation_loss": best_validation,
        "loaded_actor_encoder_tensor_count": len(loaded_encoder_keys),
        "split_metrics": {
            split: evaluate_predictions(
                arrays[split], predictions[split], args.rank_tie_cost
            )
            for split in ("train", "validation", "test")
        },
        "history": history_rows,
    }
    checkpoint = {
        "format_version": 1,
        "model_type": "TorchMPPIProposalCritic",
        "architecture": {
            "dropout": args.dropout,
            "parameter_count": model.parameter_count,
            "target": "warm_weighted_output_cost_mean - center_weighted_output_cost_mean",
        },
        "normalization": normalization.to_dict(),
        "advantage_scale": advantage_scale,
        "model_state_dict": best_state,
        "source_collection": str(args.source.resolve()),
        "teacher_labels": str(args.labels.resolve()),
        "actor_checkpoint": str(args.actor_checkpoint.resolve()),
        "training": {
            "seed": seed,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "rank_weight": args.rank_weight,
            "anchor_weight": args.anchor_weight,
            "rank_tie_cost": args.rank_tie_cost,
        },
        "metrics": metrics,
    }
    return metrics, checkpoint


def load_critic_checkpoint(
    path: Path, device: torch.device
) -> tuple[TorchMPPIProposalCritic, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("model_type") != "TorchMPPIProposalCritic":
        raise ValueError(f"{path}: invalid critic checkpoint")
    model = TorchMPPIProposalCritic(
        dropout=float(checkpoint["architecture"]["dropout"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def evaluate_fresh_centers(
    models: list[TorchMPPIProposalCritic],
    normalization: MPPIProposalNormalization,
    advantage_scale: float,
    source_root: Path,
    label_root: Path,
    actor_checkpoint_path: Path,
    csv_path: Path,
    device: torch.device,
    tie_cost: float,
) -> dict[str, Any]:
    actor, actor_normalization, _ = load_policy(actor_checkpoint_path, device)
    if normalization.to_dict() != actor_normalization.to_dict():
        raise AssertionError("critic and actor normalization differ")
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    true_advantage = []
    prediction = []
    method_names = ("warm", "network", "teacher")
    for row in rows:
        episode_id = row["episode_id"]
        step = int(row["control_step"])
        source_path = source_root / episode_id / "snapshots" / f"step_{step:06d}.npz"
        label_path = label_root / episode_id / f"step_{step:06d}.npz"
        with np.load(source_path, allow_pickle=False) as source, np.load(
            label_path, allow_pickle=False
        ) as label:
            _, network_center = predict_center(
                actor, actor_normalization, source, device
            )
            warm = np.asarray(source["sampling_mean_knots"], dtype=np.float32)
            teacher = np.asarray(label["teacher_center_knots"], dtype=np.float32)
            centers = np.asarray((warm, network_center, teacher), dtype=np.float32)
            state = np.asarray(source["initial_state"], dtype=np.float32)
            action = np.asarray(source["current_action"], dtype=np.float32)
            history = np.asarray(source["history"][0], dtype=np.float32)
            reference = ego_reference_features(source["reference_ego"], float(state[3]))
            current = np.asarray((state[3], state[4], *action), dtype=np.float32)
            history, reference, current = normalization.normalize_numpy(
                history, reference, current
            )
            tensors = (
                torch.from_numpy(history[None]).to(device),
                torch.from_numpy(reference[None]).to(device),
                torch.from_numpy(current[None]).to(device),
                torch.from_numpy(warm[None]).to(device),
                torch.from_numpy(centers[None]).to(device),
            )
            with torch.no_grad():
                predicted = torch.stack(
                    [predict_center_bank(model, *tensors) for model in models]
                ).mean(dim=0)[0]
            prediction.append(predicted.cpu().numpy() * advantage_scale)
        costs = np.asarray(
            [float(row[f"{method}_weighted_output_cost"]) for method in method_names]
        )
        true_advantage.append(costs[0] - costs)
    target = np.asarray(true_advantage)
    predicted = np.asarray(prediction)
    mask = np.ones_like(target, dtype=bool)
    dummy = CriticArrays(
        history=np.empty((len(rows), 0, 0), dtype=np.float32),
        reference=np.empty((len(rows), 0, 0), dtype=np.float32),
        current=np.empty((len(rows), 0), dtype=np.float32),
        warm=np.empty((len(rows), 0, 0), dtype=np.float32),
        centers=np.empty((len(rows), 3, 0, 0), dtype=np.float32),
        center_mask=mask,
        costs=np.asarray(
            [
                [float(row[f"{method}_weighted_output_cost"]) for method in method_names]
                for row in rows
            ]
        ),
        cost_seed_std=np.zeros_like(target),
        warm_index=np.zeros(len(rows), dtype=np.int64),
        network_index=np.ones(len(rows), dtype=np.int64),
        teacher_index=np.full(len(rows), 2, dtype=np.int64),
        episode_ids=[row["episode_id"] for row in rows],
        control_steps=np.asarray([int(row["control_step"]) for row in rows]),
    )
    summary = evaluate_predictions(dummy, predicted, tie_cost)
    selected = np.argmax(predicted, axis=1)
    summary["method_names"] = list(method_names)
    summary["selected_method_counts"] = {
        method: int(np.sum(selected == index))
        for index, method in enumerate(method_names)
    }
    summary["network_vs_warm_ranking_accuracy"] = float(
        np.mean(np.sign(predicted[:, 1] - predicted[:, 0]) == np.sign(target[:, 1]))
    )
    summary["teacher_vs_network_ranking_accuracy"] = float(
        np.mean(
            np.sign(predicted[:, 2] - predicted[:, 1])
            == np.sign(target[:, 2] - target[:, 1])
        )
    )
    return summary


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one critic seed is required")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    source_root = args.source.resolve()
    label_root = args.labels.resolve()
    actor_checkpoint_path = args.actor_checkpoint.resolve()
    actor_checkpoint = torch.load(actor_checkpoint_path, map_location="cpu")
    normalization = MPPIProposalNormalization.from_dict(
        actor_checkpoint["normalization"]
    )
    splits = json.loads((label_root / "splits.json").read_text())
    arrays = {
        split: load_split(source_root, label_root, splits[split], args.cost_key)
        for split in ("train", "validation", "test")
    }
    audit_cost_key = f"audit_{args.cost_key}"
    first_label = next(label_root.glob("episode_*/*.npz"))
    with np.load(first_label, allow_pickle=False) as first:
        has_audit_cost = audit_cost_key in first
    audit_arrays = (
        {
            split: load_split(
                source_root, label_root, splits[split], audit_cost_key
            )
            for split in ("validation", "test")
        }
        if has_audit_cost
        else None
    )
    train_advantage = arrays["train"].advantage[arrays["train"].center_mask]
    advantage_scale = float(max(np.std(train_advantage), 1.0))
    datasets = {
        split: CriticDataset(values, normalization, advantage_scale)
        for split, values in arrays.items()
    }
    run_metrics = []
    checkpoint_paths = []
    for seed in seeds:
        metrics, checkpoint = train_one(
            seed,
            args,
            datasets,
            arrays,
            actor_checkpoint,
            advantage_scale,
            normalization,
        )
        checkpoint_path = (args.output_dir / f"critic_seed{seed}.pt").resolve()
        torch.save(checkpoint, checkpoint_path)
        metrics["checkpoint"] = str(checkpoint_path)
        run_metrics.append(metrics)
        checkpoint_paths.append(checkpoint_path)
        print(json.dumps({"run": f"critic_seed{seed}", **metrics}, indent=2))

    device = torch.device(args.device)
    models = [load_critic_checkpoint(path, device)[0] for path in checkpoint_paths]
    ensemble_metrics = {
        split: evaluate_predictions(
            arrays[split],
            predict_dataset(
                models, datasets[split], device, advantage_scale, args.batch_size
            ),
            args.rank_tie_cost,
        )
        for split in ("train", "validation", "test")
    }
    ensemble_audit_metrics = None
    if audit_arrays is not None:
        ensemble_audit_metrics = {
            split: evaluate_predictions(
                audit_arrays[split],
                predict_dataset(
                    models,
                    datasets[split],
                    device,
                    advantage_scale,
                    args.batch_size,
                ),
                args.rank_tie_cost,
            )
            for split in ("validation", "test")
        }
    fresh_metrics = None
    if args.fresh_eval_csv.is_file():
        fresh_metrics = evaluate_fresh_centers(
            models,
            normalization,
            advantage_scale,
            source_root,
            label_root,
            actor_checkpoint_path,
            args.fresh_eval_csv.resolve(),
            device,
            args.rank_tie_cost,
        )
    selected_run = min(
        run_metrics,
        key=lambda row: (
            row["split_metrics"]["validation"]["selection_regret"]["mean"],
            -row["split_metrics"]["validation"]["pairwise_accuracy"],
        ),
    )
    summary = {
        "format_version": 1,
        "model_type": "TorchMPPIProposalCritic",
        "source_collection": str(source_root),
        "teacher_labels": str(label_root),
        "actor_checkpoint": str(actor_checkpoint_path),
        "split_counts": {split: len(value.history) for split, value in arrays.items()},
        "center_counts": {
            split: int(value.center_mask.sum()) for split, value in arrays.items()
        },
        "advantage_scale": advantage_scale,
        "target_semantics": "warm weighted-output cost mean minus center weighted-output cost mean",
        "training_cost_key": args.cost_key,
        "training_config": {
            key: getattr(args, key)
            for key in (
                "epochs",
                "patience",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "dropout",
                "rank_weight",
                "anchor_weight",
                "rank_tie_cost",
            )
        },
        "runs": run_metrics,
        "selected_single_checkpoint": selected_run["checkpoint"],
        "ensemble_checkpoints": [str(path) for path in checkpoint_paths],
        "ensemble_metrics": ensemble_metrics,
        "ensemble_audit_metrics": ensemble_audit_metrics,
        "fresh_seed_test_metrics": fresh_metrics,
        "interpretation": (
            "The critic predicts one-step warm-relative proposal quality. It is not "
            "a temporal Q function and must not be used outside the evaluated center "
            "trust region without fixed-DBM relabeling."
        ),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({"status": "ok", "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
