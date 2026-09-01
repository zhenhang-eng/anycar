#!/usr/bin/env python3
"""Train and qualify a local MPPI critic from full-rank reward differences."""

from __future__ import annotations

import argparse
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
    TorchMPPILocalQuadraticCritic,
    TorchMPPIProposalCritic,
    ego_reference_features,
)
from train_mppi_proposal_critic import initialize_from_actor


DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1"
)
DEFAULT_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_critic_fullrank_diverse_20260805_v1"
)
DEFAULT_ACTOR = Path(
    "outputs/mppi_proposal/bc_t1_conv_diverse_20260805_v1/trust1_seed1.pt"
)
DEFAULT_INITIAL_CRITIC = Path(
    "outputs/mppi_proposal/critic_local_diverse_20260805_v2/training_summary.json"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/critic_fullrank_diverse_20260805_v1")
ACTION_DIMENSION = 16
DIRECTION_COUNT = 16
CENTER_COUNT = 33


@dataclass
class FullRankArrays:
    history: np.ndarray
    reference: np.ndarray
    current: np.ndarray
    warm: np.ndarray
    base: np.ndarray
    centers: np.ndarray
    sigma: np.ndarray
    selection_cost: np.ndarray
    audit_cost: np.ndarray
    selection_advantage: np.ndarray
    audit_advantage: np.ndarray
    selection_advantage_std: np.ndarray
    selection_gradient: np.ndarray
    audit_gradient: np.ndarray
    selection_gradient_std: np.ndarray
    episode_ids: list[str]
    control_steps: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--actor-checkpoint", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument(
        "--initial-critic-summary", type=Path, default=DEFAULT_INITIAL_CRITIC
    )
    parser.add_argument(
        "--initialization",
        choices=("critic", "actor"),
        default="actor",
        help="Initialize the full critic or only actor-compatible state encoders.",
    )
    parser.add_argument(
        "--model-type", choices=("quadratic", "generic"), default="quadratic"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--gradient-weight", type=float, default=0.75)
    parser.add_argument("--direction-rank-weight", type=float, default=0.25)
    parser.add_argument("--anchor-weight", type=float, default=0.10)
    parser.add_argument("--rank-tie-cost", type=float, default=0.10)
    parser.add_argument("--minimum-uncertainty-weight", type=float, default=0.15)
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


def local_gradients(
    centers: np.ndarray,
    base: np.ndarray,
    sigma: np.ndarray,
    costs: np.ndarray,
) -> np.ndarray:
    """Fit d(base_cost-center_cost)/d standardized-center for each seed."""
    design = ((centers - base[None]) / sigma.reshape(1, 1, 2)).reshape(
        len(centers), ACTION_DIMENSION
    )
    if np.linalg.matrix_rank(design) != ACTION_DIMENSION:
        raise ValueError("local center design is not full rank")
    advantage_by_seed = costs[0][None, :] - costs
    return np.stack(
        [
            np.linalg.lstsq(design, advantage_by_seed[:, index], rcond=None)[0]
            for index in range(costs.shape[1])
        ],
        axis=0,
    ).astype(np.float32)


def load_split(
    source_root: Path, label_root: Path, episodes: list[str]
) -> FullRankArrays:
    fields = {
        name: []
        for name in (
            "history",
            "reference",
            "current",
            "warm",
            "base",
            "centers",
            "sigma",
            "selection_cost",
            "audit_cost",
            "selection_advantage",
            "audit_advantage",
            "selection_advantage_std",
            "selection_gradient",
            "audit_gradient",
            "selection_gradient_std",
            "control_steps",
        )
    }
    episode_ids = []
    episode_set = set(episodes)
    for label_path in sorted(label_root.glob("episode_*/*.npz")):
        episode_id = label_path.parent.name
        if episode_id not in episode_set:
            continue
        source_path = source_root / episode_id / "snapshots" / label_path.name
        with np.load(source_path, allow_pickle=False) as source, np.load(
            label_path, allow_pickle=False
        ) as label:
            centers = np.asarray(label["centers"], dtype=np.float32)
            if centers.shape != (CENTER_COUNT, 8, 2):
                raise ValueError(f"{label_path}: invalid centers")
            if int(label["local_direction_rank"]) != ACTION_DIMENSION:
                raise ValueError(f"{label_path}: local design is not full rank")
            base = np.asarray(label["base_center_knots"], dtype=np.float32)
            if not np.array_equal(centers[0], base):
                raise ValueError(f"{label_path}: center zero is not the base")
            params = json.loads(str(source["mppi_params_json"]))
            sigma = np.asarray(params["noise_sigma"], dtype=np.float32)
            selection_cost_by_seed = np.asarray(
                label["proposal_weighted_output_cost"], dtype=np.float32
            )
            audit_cost_by_seed = np.asarray(
                label["audit_proposal_weighted_output_cost"], dtype=np.float32
            )
            selection_advantage_by_seed = (
                selection_cost_by_seed[0][None, :] - selection_cost_by_seed
            )
            audit_advantage_by_seed = (
                audit_cost_by_seed[0][None, :] - audit_cost_by_seed
            )
            selection_gradient_by_seed = local_gradients(
                centers, base, sigma, selection_cost_by_seed
            )
            audit_gradient_by_seed = local_gradients(
                centers, base, sigma, audit_cost_by_seed
            )
            state = np.asarray(source["initial_state"], dtype=np.float32)
            action = np.asarray(source["current_action"], dtype=np.float32)
            fields["history"].append(
                np.asarray(source["history"][0], dtype=np.float32)
            )
            fields["reference"].append(
                ego_reference_features(source["reference_ego"], float(state[3]))
            )
            fields["current"].append(
                np.asarray((state[3], state[4], *action), dtype=np.float32)
            )
            fields["warm"].append(
                np.asarray(source["sampling_mean_knots"], dtype=np.float32)
            )
            fields["base"].append(base)
            fields["centers"].append(centers)
            fields["sigma"].append(sigma)
            fields["selection_cost"].append(selection_cost_by_seed.mean(axis=1))
            fields["audit_cost"].append(audit_cost_by_seed.mean(axis=1))
            fields["selection_advantage"].append(
                selection_advantage_by_seed.mean(axis=1)
            )
            fields["audit_advantage"].append(audit_advantage_by_seed.mean(axis=1))
            fields["selection_advantage_std"].append(
                selection_advantage_by_seed.std(axis=1)
            )
            fields["selection_gradient"].append(
                selection_gradient_by_seed.mean(axis=0)
            )
            fields["audit_gradient"].append(audit_gradient_by_seed.mean(axis=0))
            fields["selection_gradient_std"].append(
                selection_gradient_by_seed.std(axis=0)
            )
            fields["control_steps"].append(int(source["control_step"]))
            episode_ids.append(episode_id)
    if not episode_ids:
        raise ValueError(f"no full-rank critic records for {episodes}")
    return FullRankArrays(
        **{
            name: np.asarray(value, dtype=np.float32)
            if name != "control_steps"
            else np.asarray(value, dtype=np.int64)
            for name, value in fields.items()
        },
        episode_ids=episode_ids,
    )


def uncertainty_weight(
    standard_deviation: np.ndarray, scale: float, minimum: float
) -> np.ndarray:
    weight = 1.0 / (1.0 + np.square(standard_deviation / max(scale, 1e-6)))
    return np.maximum(weight, minimum).astype(np.float32)


class FullRankDataset(Dataset):
    def __init__(
        self,
        arrays: FullRankArrays,
        normalization: MPPIProposalNormalization,
        advantage_scale: float,
        advantage_noise_scale: float,
        gradient_noise_scale: float,
        minimum_uncertainty_weight: float,
    ) -> None:
        history, reference, current = normalization.normalize_numpy(
            arrays.history, arrays.reference, arrays.current
        )
        self.history = torch.from_numpy(history.astype(np.float32))
        self.reference = torch.from_numpy(reference.astype(np.float32))
        self.current = torch.from_numpy(current.astype(np.float32))
        self.warm = torch.from_numpy(arrays.warm)
        self.base = torch.from_numpy(arrays.base)
        self.centers = torch.from_numpy(arrays.centers)
        self.sigma = torch.from_numpy(arrays.sigma)
        self.target = torch.from_numpy(
            (arrays.selection_advantage / advantage_scale).astype(np.float32)
        )
        self.target_weight = torch.from_numpy(
            uncertainty_weight(
                arrays.selection_advantage_std,
                advantage_noise_scale,
                minimum_uncertainty_weight,
            )
        )
        self.gradient_target = torch.from_numpy(
            (arrays.selection_gradient / advantage_scale).astype(np.float32)
        )
        self.gradient_weight = torch.from_numpy(
            uncertainty_weight(
                arrays.selection_gradient_std,
                gradient_noise_scale,
                minimum_uncertainty_weight,
            )
        )

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.history[index],
            self.reference[index],
            self.current[index],
            self.warm[index],
            self.base[index],
            self.centers[index],
            self.sigma[index],
            self.target[index],
            self.target_weight[index],
            self.gradient_target[index],
            self.gradient_weight[index],
        )


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1e-6)


def loss_and_gradient(
    model: TorchMPPIProposalCritic | TorchMPPILocalQuadraticCritic,
    batch: tuple[torch.Tensor, ...],
    gradient_weight: float,
    direction_rank_weight: float,
    anchor_weight: float,
    normalized_tie: float,
    create_graph: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    (
        history,
        reference,
        current,
        warm,
        base,
        centers,
        sigma,
        target,
        target_weight,
        gradient_target,
        gradient_confidence,
    ) = batch
    centers_for_gradient = centers.detach().requires_grad_(gradient_weight > 0.0)
    anchor = base if isinstance(model, TorchMPPILocalQuadraticCritic) else warm
    prediction = model.forward_center_bank(
        history, reference, current, anchor, centers_for_gradient
    )
    regression = weighted_mean(
        F.smooth_l1_loss(prediction, target, beta=0.5, reduction="none"),
        target_weight,
    )
    positive = torch.arange(1, CENTER_COUNT, 2, device=prediction.device)
    negative = positive + 1
    predicted_difference = prediction[:, positive] - prediction[:, negative]
    target_difference = target[:, positive] - target[:, negative]
    direction_confidence = torch.minimum(
        target_weight[:, positive], target_weight[:, negative]
    )
    valid = target_difference.abs() > normalized_tie
    if valid.any():
        rank_importance = (
            target_difference[valid].abs().clamp(max=2.0)
            * direction_confidence[valid]
        )
        directional_rank = weighted_mean(
            F.softplus(
                -target_difference[valid].sign() * predicted_difference[valid]
            ),
            rank_importance,
        )
    else:
        directional_rank = prediction.new_zeros(())
    anchor = prediction[:, 0].square().mean()
    if gradient_weight > 0.0:
        physical_gradient = torch.autograd.grad(
            prediction[:, 0].sum(),
            centers_for_gradient,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0][:, 0]
        standardized_gradient = (
            physical_gradient * sigma[:, None, :]
        ).reshape(-1, ACTION_DIMENSION)
        gradient_regression = weighted_mean(
            F.smooth_l1_loss(
                standardized_gradient,
                gradient_target,
                beta=0.5,
                reduction="none",
            ),
            gradient_confidence,
        )
    else:
        standardized_gradient = torch.zeros_like(gradient_target)
        gradient_regression = prediction.new_zeros(())
    total = (
        regression
        + direction_rank_weight * directional_rank
        + anchor_weight * anchor
        + gradient_weight * gradient_regression
    )
    return total, {
        "total": float(total.detach()),
        "regression": float(regression.detach()),
        "directional_rank": float(directional_rank.detach()),
        "anchor": float(anchor.detach()),
        "gradient": float(gradient_regression.detach()),
    }


def move_batch(
    batch: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device) for value in batch)


def initialize_model(
    model: TorchMPPIProposalCritic | TorchMPPILocalQuadraticCritic,
    seed: int,
    args: argparse.Namespace,
    actor_checkpoint: dict[str, Any],
    advantage_scale: float,
) -> dict[str, Any]:
    if isinstance(model, TorchMPPILocalQuadraticCritic):
        actor_state = actor_checkpoint["model_state_dict"]
        model_state = model.state_dict()
        prefixes = (
            "history_encoder.",
            "reference_encoder.",
            "current_encoder.",
            "warm_encoder.",
            "fusion.",
        )
        loaded = {
            name: value
            for name, value in actor_state.items()
            if name.startswith(prefixes)
            and name in model_state
            and model_state[name].shape == value.shape
        }
        model.load_state_dict(loaded, strict=False)
        return {
            "method": "actor_state_and_fusion",
            "loaded_tensor_count": len(loaded),
        }
    if args.initialization == "actor" or isinstance(
        model, TorchMPPILocalQuadraticCritic
    ):
        loaded = initialize_from_actor(model, actor_checkpoint)
        return {"method": "actor_encoders", "loaded_tensor_count": len(loaded)}
    summary = json.loads(args.initial_critic_summary.resolve().read_text())
    candidates = [Path(value) for value in summary["ensemble_checkpoints"]]
    match = next(
        (path for path in candidates if path.stem == f"critic_seed{seed}"),
        Path(summary["selected_single_checkpoint"]),
    )
    checkpoint = torch.load(match, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    old_scale = float(checkpoint["advantage_scale"])
    output_scale = old_scale / advantage_scale
    with torch.no_grad():
        model.fusion[-1].weight.mul_(output_scale)
        model.fusion[-1].bias.mul_(output_scale)
    return {
        "method": "previous_local_critic",
        "checkpoint": str(match.resolve()),
        "old_advantage_scale": old_scale,
        "output_rescale": output_scale,
    }


def train_one(
    seed: int,
    args: argparse.Namespace,
    datasets: dict[str, FullRankDataset],
    actor_checkpoint: dict[str, Any],
    advantage_scale: float,
    normalization: MPPIProposalNormalization,
) -> tuple[
    TorchMPPIProposalCritic | TorchMPPILocalQuadraticCritic,
    dict[str, Any],
    dict[str, Any],
]:
    set_seed(seed)
    device = torch.device(args.device)
    model_class = (
        TorchMPPILocalQuadraticCritic
        if args.model_type == "quadratic"
        else TorchMPPIProposalCritic
    )
    model = model_class(dropout=args.dropout).to(device)
    initialization = initialize_model(
        model, seed, args, actor_checkpoint, advantage_scale
    )
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
        optimizer, mode="min", factor=0.35, patience=18, min_lr=3e-7
    )
    normalized_tie = args.rank_tie_cost / advantage_scale
    best_validation = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        training = []
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            loss, components = loss_and_gradient(
                model,
                batch,
                args.gradient_weight,
                args.direction_rank_weight,
                args.anchor_weight,
                normalized_tie,
                create_graph=True,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            training.append(components)
        model.eval()
        validation = []
        for raw_batch in validation_loader:
            batch = move_batch(raw_batch, device)
            loss, components = loss_and_gradient(
                model,
                batch,
                args.gradient_weight,
                args.direction_rank_weight,
                args.anchor_weight,
                normalized_tie,
                create_graph=False,
            )
            validation.append(components)
        validation_loss = float(np.mean([row["total"] for row in validation]))
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
        if epoch == 1 or epoch % 10 == 0:
            row = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{
                    f"train_{name}": float(np.mean([item[name] for item in training]))
                    for name in training[0]
                },
                **{
                    f"validation_{name}": float(
                        np.mean([item[name] for item in validation])
                    )
                    for name in validation[0]
                },
            }
            history_rows.append(row)
            print(json.dumps({"seed": seed, **row}), flush=True)
        if stale >= args.patience:
            break
    if best_state is None:
        raise AssertionError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    metrics = {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_validation_loss": best_validation,
        "initialization": initialization,
        "history": history_rows,
    }
    checkpoint = {
        "format_version": 1,
        "model_type": type(model).__name__,
        "architecture": {
            "dropout": args.dropout,
            "parameter_count": model.parameter_count,
            "target": "frozen-BC-relative weighted-output cost advantage",
        },
        "normalization": normalization.to_dict(),
        "advantage_scale": advantage_scale,
        "model_state_dict": best_state,
        "source_collection": str(args.source.resolve()),
        "labels": str(args.labels.resolve()),
        "actor_checkpoint": str(args.actor_checkpoint.resolve()),
        "training": {
            "seed": seed,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_weight": args.gradient_weight,
            "direction_rank_weight": args.direction_rank_weight,
            "anchor_weight": args.anchor_weight,
            "rank_tie_cost": args.rank_tie_cost,
            "minimum_uncertainty_weight": args.minimum_uncertainty_weight,
            "initialization": initialization,
        },
        "metrics": metrics,
    }
    return model, metrics, checkpoint


def predict_values_and_gradients(
    models: list[TorchMPPIProposalCritic | TorchMPPILocalQuadraticCritic],
    dataset: FullRankDataset,
    device: torch.device,
    advantage_scale: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    values = []
    gradients = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        history, reference, current, warm, base, centers, sigma = batch[:7]
        anchor = (
            base
            if isinstance(models[0], TorchMPPILocalQuadraticCritic)
            else warm
        )
        with torch.no_grad():
            prediction = torch.stack(
                [
                    model.forward_center_bank(
                        history, reference, current, anchor, centers
                    )
                    for model in models
                ]
            ).mean(dim=0)
        base_for_gradient = base.detach().requires_grad_(True)
        base_prediction = torch.stack(
            [
                model(history, reference, current, anchor, base_for_gradient)
                for model in models
            ]
        ).mean(dim=0)
        physical_gradient = torch.autograd.grad(
            base_prediction.sum(), base_for_gradient
        )[0]
        standardized_gradient = (
            physical_gradient * sigma[:, None, :]
        ).reshape(-1, ACTION_DIMENSION)
        values.append(prediction.detach().cpu().numpy() * advantage_scale)
        gradients.append(
            standardized_gradient.detach().cpu().numpy() * advantage_scale
        )
    return np.concatenate(values), np.concatenate(gradients)


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def value_metrics(
    arrays: FullRankArrays, prediction: np.ndarray, tie_cost: float
) -> dict[str, Any]:
    target = arrays.audit_advantage
    error = prediction - target
    target_difference = target[:, :, None] - target[:, None, :]
    prediction_difference = prediction[:, :, None] - prediction[:, None, :]
    upper = np.triu(np.ones((CENTER_COUNT, CENTER_COUNT), dtype=bool), k=1)
    valid = upper[None] & (np.abs(target_difference) > tie_cost)
    pairwise = np.mean(
        np.sign(target_difference[valid]) == np.sign(prediction_difference[valid])
    )
    positive = np.arange(1, CENTER_COUNT, 2)
    negative = positive + 1
    target_direction = target[:, positive] - target[:, negative]
    predicted_direction = prediction[:, positive] - prediction[:, negative]
    direction_valid = np.abs(target_direction) > tie_cost
    selected = np.argmax(prediction, axis=1)
    oracle = np.argmax(target, axis=1)
    selected_advantage = target[np.arange(len(target)), selected]
    oracle_advantage = target[np.arange(len(target)), oracle]
    return {
        "advantage_mae": float(np.mean(np.abs(error))),
        "advantage_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "advantage_correlation": correlation(target.reshape(-1), prediction.reshape(-1)),
        "pairwise_accuracy": float(pairwise),
        "pairwise_count": int(valid.sum()),
        "directional_pair_accuracy": float(
            np.mean(
                np.sign(target_direction[direction_valid])
                == np.sign(predicted_direction[direction_valid])
            )
        ),
        "directional_pair_count": int(direction_valid.sum()),
        "top1_accuracy": float(np.mean(selected == oracle)),
        "selection_regret_mean": float(
            np.mean(oracle_advantage - selected_advantage)
        ),
        "base_cost_mean": float(arrays.audit_cost[:, 0].mean()),
        "selected_cost_mean": float(
            np.mean(arrays.audit_cost[:, 0] - selected_advantage)
        ),
        "selected_improvement_mean": float(np.mean(selected_advantage)),
        "selected_wins": int(np.sum(selected_advantage > 0.0)),
        "selected_losses": int(np.sum(selected_advantage < 0.0)),
    }


def gradient_pair_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, Any]:
    cosine = np.sum(prediction * target, axis=1) / (
        np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1) + 1e-12
    )
    quantiles = {
        name: float(np.quantile(cosine, value))
        for name, value in (
            ("p05", 0.05),
            ("p10", 0.10),
            ("p25", 0.25),
            ("median", 0.50),
            ("p75", 0.75),
            ("p90", 0.90),
            ("p95", 0.95),
        )
    }
    sign_accuracy = {}
    for threshold in (0.0, 0.5, 1.0, 2.0, 4.0):
        valid = np.abs(target) > threshold
        sign_accuracy[str(threshold)] = {
            "count": int(valid.sum()),
            "accuracy": float(
                np.mean(np.sign(prediction[valid]) == np.sign(target[valid]))
            ),
        }
    return {
        "cosine": quantiles,
        "cosine_positive_fraction": float(np.mean(cosine > 0.0)),
        "cosine_above_0_5_fraction": float(np.mean(cosine > 0.5)),
        "rmse": float(np.sqrt(np.mean(np.square(prediction - target)))),
        "mae": float(np.mean(np.abs(prediction - target))),
        "correlation": correlation(prediction.reshape(-1), target.reshape(-1)),
        "component_sign_accuracy": sign_accuracy,
    }


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain distinct integers")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.gradient_weight < 0.0:
        raise ValueError("--gradient-weight must be nonnegative")
    source_root = args.source.resolve()
    label_root = args.labels.resolve()
    actor_checkpoint = torch.load(args.actor_checkpoint.resolve(), map_location="cpu")
    normalization = MPPIProposalNormalization.from_dict(
        actor_checkpoint["normalization"]
    )
    splits = json.loads((label_root / "splits.json").read_text())
    arrays = {
        split: load_split(source_root, label_root, splits[split])
        for split in ("train", "validation", "test")
    }
    train_advantage = arrays["train"].selection_advantage[:, 1:]
    advantage_scale = float(max(np.std(train_advantage), 1.0))
    advantage_noise_scale = float(
        max(np.median(arrays["train"].selection_advantage_std[:, 1:]), 0.1)
    )
    gradient_noise_scale = float(
        max(np.median(arrays["train"].selection_gradient_std), 0.1)
    )
    datasets = {
        split: FullRankDataset(
            values,
            normalization,
            advantage_scale,
            advantage_noise_scale,
            gradient_noise_scale,
            args.minimum_uncertainty_weight,
        )
        for split, values in arrays.items()
    }
    args.output_dir.mkdir(parents=True)
    models = []
    runs = []
    checkpoints = []
    for seed in seeds:
        model, metrics, checkpoint = train_one(
            seed,
            args,
            datasets,
            actor_checkpoint,
            advantage_scale,
            normalization,
        )
        checkpoint_path = (args.output_dir / f"critic_seed{seed}.pt").resolve()
        checkpoint["metrics"] = metrics
        torch.save(checkpoint, checkpoint_path)
        metrics["checkpoint"] = str(checkpoint_path)
        runs.append(metrics)
        checkpoints.append(checkpoint_path)
        models.append(model)
    device = torch.device(args.device)
    ensemble = {}
    for split in ("train", "validation", "test"):
        prediction, gradient = predict_values_and_gradients(
            models,
            datasets[split],
            device,
            advantage_scale,
            args.batch_size,
        )
        ensemble[split] = {
            "audit_value": value_metrics(
                arrays[split], prediction, args.rank_tie_cost
            ),
            "predicted_vs_selection_gradient": gradient_pair_metrics(
                gradient, arrays[split].selection_gradient
            ),
            "predicted_vs_audit_gradient": gradient_pair_metrics(
                gradient, arrays[split].audit_gradient
            ),
            "selection_vs_audit_gradient_ceiling": gradient_pair_metrics(
                arrays[split].selection_gradient, arrays[split].audit_gradient
            ),
        }
    selected_run = min(runs, key=lambda row: row["best_validation_loss"])
    summary = {
        "format_version": 1,
        "model_type": type(models[0]).__name__,
        "source_collection": str(source_root),
        "labels": str(label_root),
        "actor_checkpoint": str(args.actor_checkpoint.resolve()),
        "split_counts": {
            split: len(values.history) for split, values in arrays.items()
        },
        "center_count": CENTER_COUNT,
        "action_dimension": ACTION_DIMENSION,
        "advantage_scale": advantage_scale,
        "advantage_noise_scale": advantage_noise_scale,
        "gradient_noise_scale": gradient_noise_scale,
        "target_semantics": (
            "base weighted-output cost minus center weighted-output cost; local "
            "gradient is least-squares fitted from rollout rewards in standardized "
            "center coordinates"
        ),
        "training_config": {
            key: getattr(args, key)
            for key in (
                "initialization",
                "model_type",
                "epochs",
                "patience",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "dropout",
                "gradient_weight",
                "direction_rank_weight",
                "anchor_weight",
                "rank_tie_cost",
                "minimum_uncertainty_weight",
            )
        },
        "runs": runs,
        "selected_single_checkpoint": selected_run["checkpoint"],
        "ensemble_checkpoints": [str(path) for path in checkpoints],
        "ensemble_metrics": ensemble,
        "qualification_note": (
            "Audit seeds are never used for optimization or checkpoint selection. "
            "Actor-gradient use requires predicted-vs-audit local gradient agreement "
            "near the selection-vs-audit reward-noise ceiling and real DBM reevaluation."
        ),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({"status": "ok", "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
