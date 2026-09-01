#!/usr/bin/env python3
"""Train and audit a feedback-conditioned critic for pass-two MPPI centers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TorchMPPIFeedbackQuadraticCritic,
    ego_reference_features,
)


DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1"
)
DEFAULT_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_feedback_diverse_20260805_v1"
)
DEFAULT_ACTOR = Path(
    "outputs/mppi_proposal/bc_t1_conv_diverse_20260805_v1/trust1_seed1.pt"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/critic_two_pass_feedback_20260805_v1")
ACTION_DIMENSION = 16
CENTER_COUNT = 33


@dataclass
class Arrays:
    history: np.ndarray
    reference: np.ndarray
    current: np.ndarray
    anchor: np.ndarray
    feedback: np.ndarray
    centers: np.ndarray
    sigma: np.ndarray
    cost: np.ndarray
    advantage: np.ndarray
    gradient: np.ndarray
    state_index: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--actor-checkpoint", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=55)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--gradient-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.35)
    parser.add_argument("--rank-weight", type=float, default=0.20)
    parser.add_argument("--rank-tie-cost", type=float, default=0.10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_gradient(
    centers: np.ndarray, anchor: np.ndarray, sigma: np.ndarray, cost: np.ndarray
) -> np.ndarray:
    design = ((centers - anchor[None]) / sigma.reshape(1, 1, 2)).reshape(
        CENTER_COUNT, ACTION_DIMENSION
    )
    if np.linalg.matrix_rank(design) != ACTION_DIMENSION:
        raise ValueError("second-pass local center design is not full rank")
    advantage = cost[0] - cost
    return np.linalg.lstsq(design, advantage, rcond=None)[0].astype(np.float32)


def load_partition(
    source_root: Path,
    label_root: Path,
    episodes: list[str],
    partition: str,
) -> Arrays:
    prefix = "" if partition == "selection" else "audit_"
    fields: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "history", "reference", "current", "anchor", "feedback",
            "centers", "sigma", "cost", "advantage", "gradient", "state_index",
        )
    }
    episode_set = set(episodes)
    state_index = 0
    for label_path in sorted(label_root.glob("episode_*/*.npz")):
        if label_path.parent.name not in episode_set:
            continue
        source_path = source_root / label_path.parent.name / "snapshots" / label_path.name
        with np.load(source_path, allow_pickle=False) as source, np.load(label_path, allow_pickle=False) as label:
            anchors = np.asarray(label[f"{prefix}guided_center_knots"], np.float32)
            feedback = np.asarray(label[f"{prefix}first_pass_feedback"], np.float32)
            centers = np.asarray(label[f"{prefix}centers"], np.float32)
            costs = np.asarray(label[f"{prefix}proposal_weighted_output_cost"], np.float32)
            params = json.loads(str(source["mppi_params_json"]))
            sigma = np.asarray(params["noise_sigma"], np.float32)
            state = np.asarray(source["initial_state"], np.float32)
            action = np.asarray(source["current_action"], np.float32)
            history = np.asarray(source["history"][0], np.float32)
            reference = ego_reference_features(source["reference_ego"], float(state[3]))
            current = np.asarray((state[3], state[4], *action), np.float32)
            for repeat in range(len(anchors)):
                cost = costs[repeat]
                fields["history"].append(history)
                fields["reference"].append(reference)
                fields["current"].append(current)
                fields["anchor"].append(anchors[repeat])
                fields["feedback"].append(feedback[repeat])
                fields["centers"].append(centers[repeat])
                fields["sigma"].append(sigma)
                fields["cost"].append(cost)
                fields["advantage"].append(cost[0] - cost)
                fields["gradient"].append(
                    fit_gradient(centers[repeat], anchors[repeat], sigma, cost)
                )
                fields["state_index"].append(np.asarray(state_index, np.int64))
        state_index += 1
    if not fields["history"]:
        raise ValueError(f"no {partition} labels for requested episodes")
    return Arrays(
        **{
            name: np.asarray(value, dtype=np.int64 if name == "state_index" else np.float32)
            for name, value in fields.items()
        }
    )


class CriticDataset(Dataset):
    def __init__(
        self,
        arrays: Arrays,
        state_normalization: MPPIProposalNormalization,
        feedback_mean: np.ndarray,
        feedback_std: np.ndarray,
        advantage_scale: float,
    ) -> None:
        history, reference, current = state_normalization.normalize_numpy(
            arrays.history, arrays.reference, arrays.current
        )
        self.values = tuple(
            torch.from_numpy(value.astype(np.float32)) for value in (
                history, reference, current, arrays.anchor,
                (arrays.feedback - feedback_mean) / feedback_std,
                arrays.centers, arrays.advantage / advantage_scale,
                arrays.gradient / advantage_scale,
            )
        )

    def __len__(self) -> int:
        return len(self.values[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return tuple(value[index] for value in self.values)


def move(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device) for value in batch)


def loss_terms(
    model: TorchMPPIFeedbackQuadraticCritic,
    batch: tuple[torch.Tensor, ...],
    args: argparse.Namespace,
    advantage_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    history, reference, current, anchor, feedback, centers, target, gradient_target = batch
    prediction = model.forward_center_bank(
        history, reference, current, anchor, feedback, centers
    )
    gradient, _ = model.local_parameters(history, reference, current, anchor, feedback)
    gradient = gradient.flatten(1)
    regression = F.smooth_l1_loss(prediction, target, beta=0.5)
    gradient_regression = F.smooth_l1_loss(gradient, gradient_target, beta=0.5)
    cosine = (1.0 - F.cosine_similarity(gradient, gradient_target, dim=1)).mean()
    positive = torch.arange(1, CENTER_COUNT, 2, device=prediction.device)
    negative = positive + 1
    target_difference = target[:, positive] - target[:, negative]
    prediction_difference = prediction[:, positive] - prediction[:, negative]
    valid = target_difference.abs() > args.rank_tie_cost / advantage_scale
    rank = (
        F.softplus(-target_difference[valid].sign() * prediction_difference[valid]).mean()
        if valid.any() else prediction.new_zeros(())
    )
    total = regression + args.gradient_weight * gradient_regression + args.cosine_weight * cosine + args.rank_weight * rank
    return total, {
        "total": float(total.detach()), "regression": float(regression.detach()),
        "gradient": float(gradient_regression.detach()), "cosine": float(cosine.detach()),
        "rank": float(rank.detach()),
    }


def initialize_from_actor(
    model: TorchMPPIFeedbackQuadraticCritic, checkpoint: dict[str, Any]
) -> int:
    current = model.state_dict()
    prefixes = ("history_encoder.", "reference_encoder.", "current_encoder.", "warm_encoder.")
    compatible = {
        name: value for name, value in checkpoint["model_state_dict"].items()
        if name.startswith(prefixes) and name in current and current[name].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    return len(compatible)


def train_one(
    seed: int,
    args: argparse.Namespace,
    datasets: dict[str, CriticDataset],
    actor_checkpoint: dict[str, Any],
    advantage_scale: float,
) -> tuple[TorchMPPIFeedbackQuadraticCritic, dict[str, Any]]:
    set_seed(seed)
    device = torch.device(args.device)
    model = TorchMPPIFeedbackQuadraticCritic(dropout=args.dropout).to(device)
    loaded = initialize_from_actor(model, actor_checkpoint)
    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        datasets["validation"], batch_size=args.batch_size, shuffle=False
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=15, min_lr=3e-7
    )
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        training = []
        for batch in train_loader:
            loss, terms = loss_terms(model, move(batch, device), args, advantage_scale)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            training.append(terms)
        model.eval()
        validation = []
        with torch.no_grad():
            for batch in validation_loader:
                _, terms = loss_terms(model, move(batch, device), args, advantage_scale)
                validation.append(terms)
        validation_loss = float(np.mean([row["total"] for row in validation]))
        scheduler.step(validation_loss)
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            row = {
                "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train_{name}": float(np.mean([x[name] for x in training])) for name in training[0]},
                **{f"validation_{name}": float(np.mean([x[name] for x in validation])) for name in validation[0]},
            }
            history_rows.append(row)
            print(json.dumps({"seed": seed, **row}), flush=True)
        if stale >= args.patience:
            break
    if best_state is None:
        raise AssertionError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
        "best_validation_loss": best_loss, "loaded_actor_tensors": loaded,
        "history": history_rows,
    }


def predict(
    models: list[TorchMPPIFeedbackQuadraticCritic],
    dataset: CriticDataset,
    device: torch.device,
    scale: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    values, gradients = [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        history, reference, current, anchor, feedback, centers, _, _ = move(batch, device)
        with torch.no_grad():
            local = [model.local_parameters(history, reference, current, anchor, feedback)[0].flatten(1) for model in models]
            bank = [model.forward_center_bank(history, reference, current, anchor, feedback, centers) for model in models]
        gradients.append(torch.stack(local).mean(0).cpu().numpy() * scale)
        values.append(torch.stack(bank).mean(0).cpu().numpy() * scale)
    return np.concatenate(values), np.concatenate(gradients)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left.reshape(-1), right.reshape(-1))[0, 1]) if np.std(left) > 1e-12 and np.std(right) > 1e-12 else 0.0


def gradient_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    cosine = np.sum(prediction * target, axis=1) / (
        np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1) + 1e-12
    )
    return {
        "cosine_p10": float(np.quantile(cosine, 0.10)),
        "cosine_p25": float(np.quantile(cosine, 0.25)),
        "cosine_median": float(np.median(cosine)),
        "cosine_p75": float(np.quantile(cosine, 0.75)),
        "cosine_positive_fraction": float(np.mean(cosine > 0)),
        "cosine_above_0_5_fraction": float(np.mean(cosine > 0.5)),
        "correlation": correlation(prediction, target),
        "rmse": float(np.sqrt(np.mean((prediction-target)**2))),
    }


def value_metrics(prediction: np.ndarray, arrays: Arrays, tie: float) -> dict[str, Any]:
    target = arrays.advantage
    selected = np.argmax(prediction, axis=1)
    actual = target[np.arange(len(target)), selected]
    oracle = target.max(axis=1)
    positive, negative = np.arange(1, CENTER_COUNT, 2), np.arange(2, CENTER_COUNT, 2)
    td = target[:, positive] - target[:, negative]
    pd = prediction[:, positive] - prediction[:, negative]
    valid = np.abs(td) > tie
    return {
        "advantage_mae": float(np.mean(np.abs(prediction-target))),
        "advantage_correlation": correlation(prediction, target),
        "directional_pair_accuracy": float(np.mean(np.sign(td[valid]) == np.sign(pd[valid]))),
        "directional_pair_count": int(valid.sum()),
        "selected_improvement_mean": float(actual.mean()),
        "selected_improvement_median": float(np.median(actual)),
        "selected_wins": int(np.sum(actual > 0)), "selected_losses": int(np.sum(actual < 0)),
        "selection_regret_mean": float(np.mean(oracle-actual)),
        "guided_base_cost_mean": float(arrays.cost[:, 0].mean()),
        "selected_cost_mean": float(np.mean(arrays.cost[:, 0]-actual)),
    }


def state_mean_gradient(arrays: Arrays) -> np.ndarray:
    return np.stack([
        arrays.gradient[arrays.state_index == index].mean(axis=0)
        for index in np.unique(arrays.state_index)
    ])


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    source_root, label_root = args.source.resolve(), args.labels.resolve()
    splits = json.loads((label_root / "splits.json").read_text())
    selection = {split: load_partition(source_root, label_root, splits[split], "selection") for split in ("train", "validation", "test")}
    audit = {split: load_partition(source_root, label_root, splits[split], "audit") for split in ("train", "validation", "test")}
    actor_checkpoint = torch.load(args.actor_checkpoint.resolve(), map_location="cpu")
    normalization = MPPIProposalNormalization.from_dict(actor_checkpoint["normalization"])
    feedback_mean = selection["train"].feedback.mean(axis=0).astype(np.float32)
    feedback_std = np.maximum(selection["train"].feedback.std(axis=0), 1e-4).astype(np.float32)
    advantage_scale = float(max(np.std(selection["train"].advantage[:, 1:]), 1.0))
    linear_input = (selection["train"].feedback - feedback_mean) / feedback_std
    linear_input = np.concatenate(
        (np.ones((len(linear_input), 1), np.float32), linear_input), axis=1
    )
    linear_weights = np.linalg.solve(
        linear_input.T @ linear_input + 0.1 * np.eye(linear_input.shape[1]),
        linear_input.T @ selection["train"].gradient,
    ).astype(np.float32)
    selection_datasets = {
        split: CriticDataset(values, normalization, feedback_mean, feedback_std, advantage_scale)
        for split, values in selection.items()
    }
    audit_datasets = {
        split: CriticDataset(values, normalization, feedback_mean, feedback_std, advantage_scale)
        for split, values in audit.items()
    }
    args.output_dir.mkdir(parents=True)
    models, runs, checkpoint_paths = [], [], []
    for seed in seeds:
        model, run = train_one(seed, args, selection_datasets, actor_checkpoint, advantage_scale)
        checkpoint_path = (args.output_dir / f"critic_seed{seed}.pt").resolve()
        torch.save(
            {
                "format_version": 1, "model_type": type(model).__name__,
                "model_state_dict": model.state_dict(), "feedback_mean": feedback_mean,
                "feedback_std": feedback_std, "state_normalization": normalization.to_dict(),
                "advantage_scale": advantage_scale, "source_collection": str(source_root),
                "labels": str(label_root), "training": run,
            }, checkpoint_path,
        )
        run["checkpoint"] = str(checkpoint_path)
        models.append(model)
        runs.append(run)
        checkpoint_paths.append(str(checkpoint_path))
    device = torch.device(args.device)
    metrics = {}
    for split in ("train", "validation", "test"):
        selection_value, selection_gradient = predict(models, selection_datasets[split], device, advantage_scale, args.batch_size)
        audit_value, audit_gradient = predict(models, audit_datasets[split], device, advantage_scale, args.batch_size)
        feedback_baseline = -audit[split].feedback[:, 16:32]
        audit_linear_input = (audit[split].feedback - feedback_mean) / feedback_std
        audit_linear_input = np.concatenate(
            (np.ones((len(audit_linear_input), 1), np.float32), audit_linear_input), axis=1
        )
        linear_prediction = audit_linear_input @ linear_weights
        metrics[split] = {
            "selection_value": value_metrics(selection_value, selection[split], args.rank_tie_cost),
            "audit_value": value_metrics(audit_value, audit[split], args.rank_tie_cost),
            "predicted_vs_selection_gradient": gradient_metrics(selection_gradient, selection[split].gradient),
            "predicted_vs_audit_gradient": gradient_metrics(audit_gradient, audit[split].gradient),
            "first_pass_empirical_gradient_vs_audit": gradient_metrics(feedback_baseline, audit[split].gradient),
            "feedback_linear_ridge_vs_audit_gradient": gradient_metrics(
                linear_prediction, audit[split].gradient
            ),
            "selection_vs_audit_state_mean_ceiling": gradient_metrics(
                state_mean_gradient(selection[split]), state_mean_gradient(audit[split])
            ),
        }
    summary = {
        "format_version": 1, "model_type": type(models[0]).__name__,
        "source_collection": str(source_root), "labels": str(label_root),
        "split_context_counts": {split: len(value.history) for split, value in selection.items()},
        "feedback_dimension": int(selection["train"].feedback.shape[1]),
        "advantage_scale": advantage_scale, "runs": runs,
        "feedback_linear_ridge": {
            "ridge": 0.1,
            "training_partition": "selection/train only",
            "weights": linear_weights.tolist(),
        },
        "ensemble_checkpoints": checkpoint_paths, "ensemble_metrics": metrics,
        "audit_policy": "Audit first-pass feedback and second-pass rollout seeds are never used for optimization or checkpoint selection.",
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output_dir), "test": metrics["test"]}, indent=2))


if __name__ == "__main__":
    main()
