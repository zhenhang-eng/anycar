#!/usr/bin/env python3
"""Train lightweight MPPI proposal policies on T1 DBM teacher labels."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TorchMPPIProposalPolicy,
    ego_reference_features,
)


DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_train_seed_20260802_v2"
)
DEFAULT_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_teacher_t1_20260803_v1"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/bc_t1_conv_v1")


@dataclass
class ProposalArrays:
    history: np.ndarray
    reference: np.ndarray
    current: np.ndarray
    warm: np.ndarray
    teacher_center: np.ndarray
    teacher_delta: np.ndarray
    episode_ids: list[str]
    control_steps: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--trust-multipliers", default="1.0,2.0")
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--patience", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=80)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_split(
    source_root: Path, label_root: Path, episodes: list[str]
) -> ProposalArrays:
    values: dict[str, list[Any]] = {
        "history": [],
        "reference": [],
        "current": [],
        "warm": [],
        "teacher_center": [],
        "teacher_delta": [],
        "episode_ids": [],
        "control_steps": [],
    }
    for label_path in sorted(label_root.glob("episode_*/*.npz")):
        episode_id = label_path.parent.name
        if episode_id not in episodes:
            continue
        source_path = source_root / episode_id / "snapshots" / label_path.name
        with np.load(source_path, allow_pickle=False) as source, np.load(
            label_path, allow_pickle=False
        ) as label:
            state = np.asarray(source["initial_state"], dtype=np.float32)
            current_action = np.asarray(source["current_action"], dtype=np.float32)
            values["history"].append(
                np.asarray(source["history"][0], dtype=np.float32)
            )
            values["reference"].append(
                ego_reference_features(source["reference_ego"], float(state[3]))
            )
            values["current"].append(
                np.asarray((state[3], state[4], *current_action), dtype=np.float32)
            )
            values["warm"].append(
                np.asarray(source["sampling_mean_knots"], dtype=np.float32)
            )
            values["teacher_center"].append(
                np.asarray(label["teacher_center_knots"], dtype=np.float32)
            )
            values["teacher_delta"].append(
                np.asarray(label["teacher_delta_knots"], dtype=np.float32)
            )
            values["episode_ids"].append(episode_id)
            values["control_steps"].append(int(source["control_step"]))
    if not values["history"]:
        raise ValueError(f"no snapshots found for episodes {episodes}")
    return ProposalArrays(
        history=np.asarray(values["history"], dtype=np.float32),
        reference=np.asarray(values["reference"], dtype=np.float32),
        current=np.asarray(values["current"], dtype=np.float32),
        warm=np.asarray(values["warm"], dtype=np.float32),
        teacher_center=np.asarray(values["teacher_center"], dtype=np.float32),
        teacher_delta=np.asarray(values["teacher_delta"], dtype=np.float32),
        episode_ids=list(values["episode_ids"]),
        control_steps=np.asarray(values["control_steps"], dtype=np.int64),
    )


class ProposalDataset(Dataset):
    def __init__(
        self, arrays: ProposalArrays, normalization: MPPIProposalNormalization
    ) -> None:
        history, reference, current = normalization.normalize_numpy(
            arrays.history, arrays.reference, arrays.current
        )
        self.history = torch.from_numpy(history.astype(np.float32))
        self.reference = torch.from_numpy(reference.astype(np.float32))
        self.current = torch.from_numpy(current.astype(np.float32))
        self.warm = torch.from_numpy(arrays.warm)
        self.teacher_center = torch.from_numpy(arrays.teacher_center)
        self.teacher_delta = torch.from_numpy(arrays.teacher_delta)

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.history[index],
            self.reference[index],
            self.current[index],
            self.warm[index],
            self.teacher_center[index],
            self.teacher_delta[index],
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_label_fit(
    model: TorchMPPIProposalPolicy,
    loader: DataLoader,
    device: torch.device,
    base_sigma: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    center_error = []
    delta_error = []
    teacher_values = []
    predictions = []
    for history, reference, current, warm, teacher_center, teacher_delta in loader:
        history = history.to(device)
        reference = reference.to(device)
        current = current.to(device)
        warm = warm.to(device)
        teacher_center = teacher_center.to(device)
        teacher_delta = teacher_delta.to(device)
        predicted_delta, predicted_center = model(history, reference, current, warm)
        center_error.append((predicted_center - teacher_center).cpu())
        delta_error.append(((predicted_delta - teacher_delta) / base_sigma).cpu())
        teacher_values.append(teacher_delta.cpu())
        predictions.append(predicted_delta.cpu())
    center_error_tensor = torch.cat(center_error)
    delta_error_tensor = torch.cat(delta_error)
    teacher_tensor = torch.cat(teacher_values)
    prediction_tensor = torch.cat(predictions)
    return {
        "normalized_delta_mse": float(delta_error_tensor.square().mean()),
        "normalized_delta_rmse": float(
            torch.sqrt(delta_error_tensor.square().mean())
        ),
        "center_mae": float(center_error_tensor.abs().mean()),
        "center_rmse": float(torch.sqrt(center_error_tensor.square().mean())),
        "acceleration_delta_mae": float(
            (prediction_tensor[..., 0] - teacher_tensor[..., 0]).abs().mean()
        ),
        "steering_delta_mae": float(
            (prediction_tensor[..., 1] - teacher_tensor[..., 1]).abs().mean()
        ),
        "prediction_delta_rms": float(torch.sqrt(prediction_tensor.square().mean())),
        "teacher_delta_rms": float(torch.sqrt(teacher_tensor.square().mean())),
    }


def train_one(
    seed: int,
    trust_multiplier: float,
    args: argparse.Namespace,
    train_dataset: ProposalDataset,
    validation_dataset: ProposalDataset,
    test_dataset: ProposalDataset,
    normalization: MPPIProposalNormalization,
    splits: dict[str, list[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    set_seed(seed)
    device = torch.device(args.device)
    trust_scale = (0.25 * trust_multiplier, 0.35 * trust_multiplier)
    model = TorchMPPIProposalPolicy(
        trust_scale=trust_scale, dropout=args.dropout
    ).to(device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    train_eval_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        min_lr=args.min_learning_rate,
    )
    base_sigma = torch.tensor((0.25, 0.35), device=device).reshape(1, 1, 2)
    best_validation = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for batch in train_loader:
            history, reference, current, warm, teacher_center, _ = (
                value.to(device) for value in batch
            )
            _, predicted_center = model(history, reference, current, warm)
            error = (predicted_center - teacher_center) / base_sigma
            loss = error.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        validation = evaluate_label_fit(
            model, validation_loader, device, base_sigma
        )
        validation_loss = validation["normalized_delta_mse"]
        scheduler.step(validation_loss)
        if validation_loss < best_validation - 1e-7:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 25 == 0:
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_batch_mse": float(np.mean(batch_losses)),
                    "validation_mse": validation_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
        if epochs_without_improvement >= args.patience:
            break
    if best_state is None:
        raise AssertionError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    metrics = {
        "seed": seed,
        "trust_multiplier": trust_multiplier,
        "trust_scale": list(trust_scale),
        "parameter_count": model.parameter_count,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_validation_mse": best_validation,
        "label_fit": {
            "train": evaluate_label_fit(
                model, train_eval_loader, device, base_sigma
            ),
            "validation": evaluate_label_fit(
                model, validation_loader, device, base_sigma
            ),
            "test": evaluate_label_fit(model, test_loader, device, base_sigma),
        },
        "history": history_rows,
    }
    checkpoint = {
        "format_version": 1,
        "model_type": "TorchMPPIProposalPolicy",
        "architecture": {
            "trust_scale": list(trust_scale),
            "dropout": args.dropout,
            "parameter_count": model.parameter_count,
        },
        "normalization": normalization.to_dict(),
        "model_state_dict": best_state,
        "source_collection": str(args.source.resolve()),
        "teacher_labels": str(args.labels.resolve()),
        "splits": splits,
        "training": {
            "seed": seed,
            "trust_multiplier": trust_multiplier,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "lr_scheduler_factor": args.lr_scheduler_factor,
            "lr_scheduler_patience": args.lr_scheduler_patience,
            "min_learning_rate": args.min_learning_rate,
        },
        "metrics": metrics,
    }
    return checkpoint, metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = json.loads((args.labels / "splits.json").read_text())
    train_arrays = load_split(args.source, args.labels, splits["train"])
    validation_arrays = load_split(args.source, args.labels, splits["validation"])
    test_arrays = load_split(args.source, args.labels, splits["test"])
    normalization = MPPIProposalNormalization.fit(
        train_arrays.history, train_arrays.reference, train_arrays.current
    )
    train_dataset = ProposalDataset(train_arrays, normalization)
    validation_dataset = ProposalDataset(validation_arrays, normalization)
    test_dataset = ProposalDataset(test_arrays, normalization)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    trust_multipliers = [
        float(value) for value in args.trust_multipliers.split(",") if value.strip()
    ]
    if not seeds or not trust_multipliers:
        raise ValueError("at least one seed and trust multiplier are required")
    runs = []
    for trust_multiplier in trust_multipliers:
        for seed in seeds:
            checkpoint, metrics = train_one(
                seed,
                trust_multiplier,
                args,
                train_dataset,
                validation_dataset,
                test_dataset,
                normalization,
                splits,
            )
            run_name = f"trust{trust_multiplier:g}_seed{seed}"
            checkpoint_path = args.output_dir / f"{run_name}.pt"
            torch.save(checkpoint, checkpoint_path)
            metrics["checkpoint"] = str(checkpoint_path.resolve())
            runs.append(metrics)
            print(json.dumps({"run": run_name, **metrics["label_fit"]}, indent=2))
    selected = min(runs, key=lambda run: run["best_validation_mse"])
    summary = {
        "format_version": 1,
        "source_collection": str(args.source.resolve()),
        "teacher_labels": str(args.labels.resolve()),
        "split_counts": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "test": len(test_dataset),
        },
        "training_config": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "lr_scheduler_factor": args.lr_scheduler_factor,
            "lr_scheduler_patience": args.lr_scheduler_patience,
            "min_learning_rate": args.min_learning_rate,
            "dropout": args.dropout,
        },
        "normalization": normalization.to_dict(),
        "runs": runs,
        "selected_checkpoint": selected["checkpoint"],
        "selection_rule": "minimum episode-heldout validation normalized delta MSE",
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({"status": "ok", "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
