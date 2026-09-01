#!/usr/bin/env python3
"""Train a mean/lower-tail/win critic on repeated two-pass step rewards."""

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

from car_foundation.mppi_proposal_policy import TorchMPPIFeedbackStepRiskCritic


DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1"
)
DEFAULT_PARENT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_feedback_diverse_20260805_v1"
)
DEFAULT_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_risk_replay_diverse_20260805_v1"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/critic_two_pass_step_risk_20260805_v3")


@dataclass
class Arrays:
    context: np.ndarray
    signed_radius: np.ndarray
    old_linear: np.ndarray
    advantage_mean: np.ndarray
    advantage_p10: np.ndarray
    win_probability: np.ndarray
    advantage_by_seed: np.ndarray
    state_index: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=55)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--tail-weight", type=float, default=1.0)
    parser.add_argument("--win-weight", type=float, default=0.30)
    parser.add_argument("--safety-weight", type=float, default=0.40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def signed_radii(radii: list[float]) -> np.ndarray:
    result = [0.0]
    for radius in radii:
        result.extend((radius, -radius))
    return np.asarray(result, np.float32)


def load_partition(
    source_root: Path,
    parent_root: Path,
    label_root: Path,
    episodes: list[str],
    partition: str,
) -> Arrays:
    prefix = "" if partition == "selection" else "audit_"
    fields: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "context", "signed_radius", "old_linear", "advantage_mean",
            "advantage_p10", "win_probability", "advantage_by_seed", "state_index",
        )
    }
    episode_set = set(episodes)
    summary = json.loads((label_root / "summary.json").read_text())
    radii = signed_radii(summary["radii_sigma"])
    state_index = 0
    for label_path in sorted(label_root.glob("episode_*/*.npz")):
        if label_path.parent.name not in episode_set:
            continue
        parent_path = parent_root / label_path.parent.name / label_path.name
        source_path = source_root / label_path.parent.name / "snapshots" / label_path.name
        with np.load(source_path, allow_pickle=False) as source, np.load(
            parent_path, allow_pickle=False
        ) as parent, np.load(label_path, allow_pickle=False) as label:
            params = json.loads(str(source["mppi_params_json"]))
            sigma = np.asarray(params["noise_sigma"], np.float32)
            feedback = np.asarray(parent[f"{prefix}first_pass_feedback"], np.float32)
            gradient_mean = np.asarray(label[f"{prefix}critic_gradient_mean"], np.float32)
            gradient_std = np.asarray(label[f"{prefix}critic_gradient_std"], np.float32)
            anchors = np.asarray(label[f"{prefix}guided_center_knots"], np.float32)
            centers = np.asarray(label[f"{prefix}centers"], np.float32)
            mean = np.asarray(label[f"{prefix}paired_advantage_mean"], np.float32)
            p10 = np.asarray(label[f"{prefix}paired_advantage_p10"], np.float32)
            win = np.asarray(label[f"{prefix}paired_win_probability"], np.float32)
            by_seed = np.asarray(label[f"{prefix}paired_advantage_by_seed"], np.float32)
            standardized_delta = (
                (centers - anchors[:, None]) / sigma.reshape(1, 1, 1, 2)
            ).reshape(len(anchors), len(radii), -1)
            old_linear = np.sum(gradient_mean[:, None] * standardized_delta, axis=2)
            context = np.concatenate((feedback, gradient_mean, gradient_std), axis=1)
            for repeat in range(len(anchors)):
                fields["context"].append(context[repeat])
                fields["signed_radius"].append(radii)
                fields["old_linear"].append(old_linear[repeat])
                fields["advantage_mean"].append(mean[repeat])
                fields["advantage_p10"].append(p10[repeat])
                fields["win_probability"].append(win[repeat])
                fields["advantage_by_seed"].append(by_seed[repeat])
                fields["state_index"].append(np.asarray(state_index, np.int64))
        state_index += 1
    if not fields["context"]:
        raise ValueError(f"no {partition} risk replay labels")
    return Arrays(
        **{
            name: np.asarray(value, np.int64 if name == "state_index" else np.float32)
            for name, value in fields.items()
        }
    )


class RiskDataset(Dataset):
    def __init__(
        self,
        arrays: Arrays,
        context_mean: np.ndarray,
        context_std: np.ndarray,
        linear_mean: float,
        linear_std: float,
        advantage_scale: float,
    ) -> None:
        self.values = tuple(
            torch.from_numpy(value.astype(np.float32))
            for value in (
                (arrays.context - context_mean) / context_std,
                arrays.signed_radius,
                (arrays.old_linear - linear_mean) / linear_std,
                arrays.advantage_mean / advantage_scale,
                arrays.advantage_p10 / advantage_scale,
                arrays.win_probability,
                (arrays.advantage_by_seed.min(axis=2) >= -1.0).astype(np.float32),
            )
        )

    def __len__(self) -> int:
        return len(self.values[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return tuple(value[index] for value in self.values)


def move(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device) for value in batch)


def loss_terms(
    model: TorchMPPIFeedbackStepRiskCritic,
    batch: tuple[torch.Tensor, ...],
    tail_weight: float,
    win_weight: float,
    safety_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    (
        context, radius, old_linear, target_mean, target_p10, target_win, target_safe
    ) = batch
    mean, p10, win_logit, safe_logit = model.forward_center_bank(
        context, radius, old_linear
    )
    moving = radius != 0
    mean_loss = F.smooth_l1_loss(mean[moving], target_mean[moving], beta=0.5)
    tail_loss = F.smooth_l1_loss(p10[moving], target_p10[moving], beta=0.5)
    win_loss = F.binary_cross_entropy_with_logits(win_logit[moving], target_win[moving])
    safety_loss = F.binary_cross_entropy_with_logits(
        safe_logit[moving], target_safe[moving]
    )
    total = (
        mean_loss + tail_weight * tail_loss + win_weight * win_loss
        + safety_weight * safety_loss
    )
    return total, {
        "total": float(total.detach()),
        "mean": float(mean_loss.detach()),
        "tail": float(tail_loss.detach()),
        "win": float(win_loss.detach()),
        "safety": float(safety_loss.detach()),
    }


def train_one(
    seed: int,
    args: argparse.Namespace,
    datasets: dict[str, RiskDataset],
) -> tuple[TorchMPPIFeedbackStepRiskCritic, dict[str, Any]]:
    set_seed(seed)
    device = torch.device(args.device)
    model = TorchMPPIFeedbackStepRiskCritic(dropout=args.dropout).to(device)
    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        datasets["validation"], batch_size=args.batch_size, shuffle=False
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=15, min_lr=3e-7
    )
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        training = []
        for batch in train_loader:
            loss, terms = loss_terms(
                model, move(batch, device), args.tail_weight, args.win_weight,
                args.safety_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            training.append(terms)
        model.eval()
        validation = []
        with torch.no_grad():
            for batch in validation_loader:
                _, terms = loss_terms(
                    model, move(batch, device), args.tail_weight, args.win_weight,
                    args.safety_weight,
                )
                validation.append(terms)
        validation_loss = float(np.mean([row["total"] for row in validation]))
        scheduler.step(validation_loss)
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch = validation_loss, epoch
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
                    f"train_{name}": float(np.mean([value[name] for value in training]))
                    for name in training[0]
                },
                **{
                    f"validation_{name}": float(
                        np.mean([value[name] for value in validation])
                    )
                    for name in validation[0]
                },
            }
            history.append(row)
            print(json.dumps({"seed": seed, **row}), flush=True)
        if stale >= args.patience:
            break
    if best_state is None:
        raise AssertionError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_validation_loss": best_loss,
        "history": history,
    }


def predict(
    models: list[TorchMPPIFeedbackStepRiskCritic],
    dataset: RiskDataset,
    device: torch.device,
    advantage_scale: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means, tails, probabilities, safety_probabilities, disagreements = [], [], [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        context, radius, old_linear = move(batch[:3], device)
        with torch.no_grad():
            result = [
                model.forward_center_bank(context, radius, old_linear)
                for model in models
            ]
            model_mean = torch.stack([value[0] for value in result])
            model_tail = torch.stack([value[1] for value in result])
            model_probability = torch.sigmoid(
                torch.stack([value[2] for value in result])
            )
            model_safety = torch.sigmoid(
                torch.stack([value[3] for value in result])
            )
        means.append(model_mean.mean(0).cpu().numpy() * advantage_scale)
        tails.append(model_tail.mean(0).cpu().numpy() * advantage_scale)
        probabilities.append(model_probability.mean(0).cpu().numpy())
        safety_probabilities.append(model_safety.mean(0).cpu().numpy())
        disagreements.append(model_mean.std(0, unbiased=False).cpu().numpy() * advantage_scale)
    return tuple(
        map(
            np.concatenate,
            (means, tails, probabilities, safety_probabilities, disagreements),
        )
    )


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    return (
        float(np.corrcoef(left.reshape(-1), right.reshape(-1))[0, 1])
        if np.std(left) > 1e-12 and np.std(right) > 1e-12
        else 0.0
    )


def prediction_metrics(
    arrays: Arrays,
    mean: np.ndarray,
    tail: np.ndarray,
    probability: np.ndarray,
    safety_probability: np.ndarray,
) -> dict[str, float]:
    moving = arrays.signed_radius != 0
    safe = arrays.advantage_by_seed.min(axis=2) >= -1.0
    return {
        "mean_advantage_mae": float(
            np.mean(np.abs(mean[moving] - arrays.advantage_mean[moving]))
        ),
        "mean_advantage_correlation": correlation(
            mean[moving], arrays.advantage_mean[moving]
        ),
        "p10_advantage_mae": float(
            np.mean(np.abs(tail[moving] - arrays.advantage_p10[moving]))
        ),
        "p10_advantage_correlation": correlation(
            tail[moving], arrays.advantage_p10[moving]
        ),
        "win_probability_brier": float(
            np.mean(np.square(probability[moving] - arrays.win_probability[moving]))
        ),
        "win_probability_correlation": correlation(
            probability[moving], arrays.win_probability[moving]
        ),
        "safety_probability_brier": float(
            np.mean(np.square(safety_probability[moving] - safe[moving]))
        ),
        "safety_probability_correlation": correlation(
            safety_probability[moving], safe[moving]
        ),
    }


def selected_metrics(
    arrays: Arrays, selected: np.ndarray
) -> dict[str, Any]:
    row = np.arange(len(selected))
    by_seed = arrays.advantage_by_seed[row, selected]
    context_mean = by_seed.mean(axis=1)
    moved = selected != 0
    moved_by_seed = by_seed[moved]
    moved_context = context_mean[moved]
    if moved_by_seed.size:
        moved_metrics = {
            "moved_context_mean_advantage": float(moved_context.mean()),
            "moved_context_median_advantage": float(np.median(moved_context)),
            "moved_seed_advantage_p05": float(np.quantile(moved_by_seed, 0.05)),
            "moved_seed_advantage_p10": float(np.quantile(moved_by_seed, 0.10)),
            "moved_seed_win_fraction": float(np.mean(moved_by_seed > 0)),
            "moved_seed_worst": float(moved_by_seed.min()),
        }
    else:
        moved_metrics = {
            "moved_context_mean_advantage": 0.0,
            "moved_context_median_advantage": 0.0,
            "moved_seed_advantage_p05": 0.0,
            "moved_seed_advantage_p10": 0.0,
            "moved_seed_win_fraction": 0.0,
            "moved_seed_worst": 0.0,
        }
    return {
        "selected_contexts": int(np.sum(moved)),
        "selection_fraction": float(np.mean(moved)),
        "context_mean_advantage": float(context_mean.mean()),
        "context_median_advantage": float(np.median(context_mean)),
        "context_wins": int(np.sum(context_mean > 0)),
        "context_losses": int(np.sum(context_mean < 0)),
        "seed_advantage_p05": float(np.quantile(by_seed, 0.05)),
        "seed_advantage_p10": float(np.quantile(by_seed, 0.10)),
        "seed_advantage_mean": float(by_seed.mean()),
        "seed_win_fraction": float(np.mean(by_seed > 0)),
        "seed_worst": float(by_seed.min()),
        "center_histogram": np.bincount(
            selected, minlength=arrays.advantage_mean.shape[1]
        ).tolist(),
        **moved_metrics,
    }


def select_with_gate(
    mean: np.ndarray,
    tail: np.ndarray,
    probability: np.ndarray,
    safety_probability: np.ndarray,
    disagreement: np.ndarray,
    tail_threshold: float,
    win_threshold: float,
    safety_threshold: float,
    disagreement_threshold: float,
) -> np.ndarray:
    eligible = (
        (tail >= tail_threshold)
        & (probability >= win_threshold)
        & (safety_probability >= safety_threshold)
        & (disagreement <= disagreement_threshold)
    )
    eligible[:, 0] = False
    # Deployment is deliberately restricted to the smallest positive trust step.
    # Larger steps have positive means driven by rare large gains but fail tail gates.
    eligible[:, 2:] = False
    score = np.where(eligible, mean, -np.inf)
    selected = np.argmax(score, axis=1)
    selected[~np.any(eligible, axis=1)] = 0
    return selected


def choose_gate(
    arrays: Arrays,
    mean: np.ndarray,
    tail: np.ndarray,
    probability: np.ndarray,
    safety_probability: np.ndarray,
    disagreement: np.ndarray,
) -> tuple[dict[str, float], dict[str, Any]]:
    candidates = []
    disagreement_grid = np.quantile(
        disagreement[:, 1], (0.25, 0.50, 0.75, 0.90, 1.0)
    )
    for tail_threshold in (-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0):
        for win_threshold in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
            for safety_threshold in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
                for disagreement_threshold in disagreement_grid:
                    selected = select_with_gate(
                        mean, tail, probability, safety_probability, disagreement,
                        tail_threshold, win_threshold, safety_threshold,
                        float(disagreement_threshold),
                    )
                    metrics = selected_metrics(arrays, selected)
                    metrics["score"] = metrics["seed_advantage_mean"]
                    candidates.append(
                        (
                            {
                                "tail_threshold": tail_threshold,
                                "win_threshold": win_threshold,
                                "safety_threshold": safety_threshold,
                                "disagreement_threshold": float(disagreement_threshold),
                            },
                            metrics,
                        )
                    )
    feasible = [
        value
        for value in candidates
        if value[1]["selected_contexts"] >= 30
        and value[1]["moved_seed_advantage_p05"] >= -0.5
        and value[1]["context_losses"] <= max(2, int(0.20 * value[1]["selected_contexts"]))
    ]
    return max(feasible or candidates, key=lambda value: value[1]["score"])


def fixed_center_metrics(arrays: Arrays) -> dict[str, Any]:
    return {
        str(index): selected_metrics(
            arrays, np.full(len(arrays.context), index, np.int64)
        )
        for index in range(arrays.advantage_mean.shape[1])
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    source_root, parent_root, label_root = (
        args.source.resolve(), args.parent_labels.resolve(), args.labels.resolve()
    )
    splits = json.loads((label_root / "splits.json").read_text())
    selection = {
        split: load_partition(
            source_root, parent_root, label_root, splits[split], "selection"
        )
        for split in ("train", "validation", "test")
    }
    audit = {
        split: load_partition(
            source_root, parent_root, label_root, splits[split], "audit"
        )
        for split in ("train", "validation", "test")
    }
    context_mean = selection["train"].context.mean(axis=0).astype(np.float32)
    context_std = np.maximum(
        selection["train"].context.std(axis=0), 1e-4
    ).astype(np.float32)
    linear_mean = float(selection["train"].old_linear.mean())
    linear_std = float(max(selection["train"].old_linear.std(), 1e-4))
    advantage_scale = float(
        max(np.std(selection["train"].advantage_mean[:, 1:]), 1.0)
    )
    selection_datasets = {
        split: RiskDataset(
            values, context_mean, context_std, linear_mean, linear_std, advantage_scale
        )
        for split, values in selection.items()
    }
    audit_datasets = {
        split: RiskDataset(
            values, context_mean, context_std, linear_mean, linear_std, advantage_scale
        )
        for split, values in audit.items()
    }
    args.output_dir.mkdir(parents=True)
    models, runs, checkpoint_paths = [], [], []
    for seed in seeds:
        model, run = train_one(seed, args, selection_datasets)
        checkpoint_path = (args.output_dir / f"risk_critic_seed{seed}.pt").resolve()
        torch.save(
            {
                "format_version": 1,
                "model_type": type(model).__name__,
                "model_state_dict": model.state_dict(),
                "context_mean": context_mean,
                "context_std": context_std,
                "linear_mean": linear_mean,
                "linear_std": linear_std,
                "advantage_scale": advantage_scale,
                "labels": str(label_root),
                "training": run,
            },
            checkpoint_path,
        )
        run["checkpoint"] = str(checkpoint_path)
        models.append(model)
        runs.append(run)
        checkpoint_paths.append(str(checkpoint_path))
    device = torch.device(args.device)
    predictions: dict[str, dict[str, tuple[np.ndarray, ...]]] = {
        "selection": {}, "audit": {}
    }
    metrics: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        predictions["selection"][split] = predict(
            models, selection_datasets[split], device, advantage_scale, args.batch_size
        )
        predictions["audit"][split] = predict(
            models, audit_datasets[split], device, advantage_scale, args.batch_size
        )
        metrics[split] = {}
        for partition, arrays in (
            ("selection", selection[split]), ("audit", audit[split])
        ):
            mean, tail, probability, safety_probability, disagreement = (
                predictions[partition][split]
            )
            metrics[split][partition] = prediction_metrics(
                arrays, mean, tail, probability, safety_probability
            )
    validation_prediction = predictions["selection"]["validation"]
    gate, validation_gate_metrics = choose_gate(
        selection["validation"], *validation_prediction
    )
    test_prediction = predictions["audit"]["test"]
    test_selected = select_with_gate(*test_prediction, **gate)
    test_mean_selected = np.argmax(test_prediction[0], axis=1)
    test_oracle = np.argmax(audit["test"].advantage_mean, axis=1)
    policy_metrics = {
        "validation_selected_gate": gate,
        "validation_gate_metrics": validation_gate_metrics,
        "test_audit_gate_metrics": selected_metrics(audit["test"], test_selected),
        "test_audit_risk_neutral_argmax": selected_metrics(
            audit["test"], test_mean_selected
        ),
        "test_audit_oracle_mean": selected_metrics(audit["test"], test_oracle),
        "test_audit_fixed_centers": fixed_center_metrics(audit["test"]),
    }
    summary = {
        "format_version": 1,
        "model_type": type(models[0]).__name__,
        "labels": str(label_root),
        "split_context_counts": {
            split: len(values.context) for split, values in selection.items()
        },
        "advantage_scale": advantage_scale,
        "runs": runs,
        "ensemble_checkpoints": checkpoint_paths,
        "prediction_metrics": metrics,
        "policy_metrics": policy_metrics,
        "audit_policy": (
            "Training and checkpoint selection use selection labels only. Gate "
            "thresholds use selection/validation only; audit/test is evaluated once. "
            "The deployable gate is restricted to the +0.03-sigma center because "
            "larger steps fail validation tail constraints."
        ),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(args.output_dir),
                "test_audit": metrics["test"]["audit"],
                "policy": policy_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
