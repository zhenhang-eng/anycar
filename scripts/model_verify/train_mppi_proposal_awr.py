#!/usr/bin/env python3
"""Reward-weighted on-support actor regression from fixed-DBM proposal labels."""

from __future__ import annotations

import argparse
import json
import random
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
    "fixed_dbm_policy_diverse_20260805_v1"
)
DEFAULT_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_critic_local_diverse_20260805_v1"
)
DEFAULT_ACTOR = Path(
    "outputs/mppi_proposal/bc_t1_conv_diverse_20260805_v1/trust1_seed1.pt"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/awr_local_20260805_v1")
SUPPORTED_CENTER_COUNT = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--actor-checkpoint", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temperatures", default="2.0,4.0,8.0")
    parser.add_argument("--bc-anchor-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
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


class AWRDataset(Dataset):
    def __init__(
        self,
        source_root: Path,
        label_root: Path,
        episodes: list[str],
        normalization: MPPIProposalNormalization,
    ) -> None:
        values = {name: [] for name in (
            "history", "reference", "current", "warm", "baseline", "teacher",
            "centers", "selection_cost", "audit_cost",
        )}
        episode_set = set(episodes)
        for label_path in sorted(label_root.glob("episode_*/*.npz")):
            if label_path.parent.name not in episode_set:
                continue
            source_path = source_root / label_path.parent.name / "snapshots" / label_path.name
            with np.load(source_path, allow_pickle=False) as source, np.load(
                label_path, allow_pickle=False
            ) as label:
                state = np.asarray(source["initial_state"], dtype=np.float32)
                action = np.asarray(source["current_action"], dtype=np.float32)
                values["history"].append(source["history"][0])
                values["reference"].append(
                    ego_reference_features(source["reference_ego"], float(state[3]))
                )
                values["current"].append((state[3], state[4], *action))
                centers = np.asarray(
                    label["shortlist_centers"][:SUPPORTED_CENTER_COUNT],
                    dtype=np.float32,
                )
                values["warm"].append(centers[0])
                values["baseline"].append(centers[1])
                values["teacher"].append(centers[2])
                values["centers"].append(centers)
                values["selection_cost"].append(
                    label["proposal_weighted_output_cost"][:SUPPORTED_CENTER_COUNT].mean(1)
                )
                values["audit_cost"].append(
                    label["audit_proposal_weighted_output_cost"][:SUPPORTED_CENTER_COUNT].mean(1)
                )
        if not values["history"]:
            raise ValueError(f"no AWR data for {episodes}")
        history = np.asarray(values["history"], dtype=np.float32)
        reference = np.asarray(values["reference"], dtype=np.float32)
        current = np.asarray(values["current"], dtype=np.float32)
        history, reference, current = normalization.normalize_numpy(
            history, reference, current
        )
        self.values = tuple(
            torch.from_numpy(np.asarray(value, dtype=np.float32))
            for value in (
                history,
                reference,
                current,
                values["warm"],
                values["baseline"],
                values["teacher"],
                values["centers"],
                values["selection_cost"],
                values["audit_cost"],
            )
        )

    def __len__(self) -> int:
        return len(self.values[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return tuple(value[index] for value in self.values)


def load_actor(
    checkpoint: dict[str, Any], device: torch.device
) -> TorchMPPIProposalPolicy:
    architecture = checkpoint["architecture"]
    model = TorchMPPIProposalPolicy(
        trust_scale=tuple(architecture["trust_scale"]),
        dropout=float(architecture["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def awr_loss(
    actor: TorchMPPIProposalPolicy,
    batch: tuple[torch.Tensor, ...],
    temperature: float,
    anchor_weight: float,
    use_audit_cost: bool,
    base_sigma: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    history, reference, current, warm, baseline, teacher, centers, selection, audit = batch
    costs = audit if use_audit_cost else selection
    weights = torch.softmax(-(costs - costs.min(dim=1, keepdim=True).values) / temperature, dim=1)
    _, prediction = actor(history, reference, current, warm)
    error = (prediction[:, None] - centers) / base_sigma[:, None]
    reward_weighted = (weights * error.square().mean(dim=(2, 3))).sum(dim=1).mean()
    update = (prediction - baseline) / base_sigma
    anchor = update.square().mean()
    loss = reward_weighted + anchor_weight * anchor
    weighted_target = (weights[..., None, None] * centers).sum(dim=1)
    target_update = torch.sqrt(
        (((weighted_target - baseline) / base_sigma).square().mean(dim=(1, 2))) + 1e-12
    )
    actual_update = torch.sqrt(update.square().mean(dim=(1, 2)) + 1e-12)
    teacher_distance = torch.sqrt(
        (((prediction - teacher) / base_sigma).square().mean(dim=(1, 2))) + 1e-12
    )
    baseline_teacher = torch.sqrt(
        (((baseline - teacher) / base_sigma).square().mean(dim=(1, 2))) + 1e-12
    )
    return loss, {
        "reward_weighted_mse": reward_weighted,
        "bc_anchor_mse": anchor,
        "target_update_rms": target_update.mean(),
        "actor_update_rms": actual_update.mean(),
        "actor_update_rms_max": actual_update.max(),
        "teacher_distance": teacher_distance.mean(),
        "baseline_teacher_distance": baseline_teacher.mean(),
        "closer_to_teacher": (teacher_distance < baseline_teacher).float().mean(),
        "teacher_weight": weights[:, 2].mean(),
        "network_weight": weights[:, 1].mean(),
    }


@torch.no_grad()
def evaluate(
    actor: TorchMPPIProposalPolicy,
    dataset: AWRDataset,
    device: torch.device,
    args: argparse.Namespace,
    temperature: float,
    use_audit_cost: bool,
) -> dict[str, float]:
    actor.eval()
    base_sigma = torch.tensor((0.25, 0.35), device=device).reshape(1, 1, 2)
    totals: dict[str, float] = {}
    count = 0
    for raw in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
        batch = tuple(value.to(device) for value in raw)
        loss, metrics = awr_loss(
            actor,
            batch,
            temperature,
            args.bc_anchor_weight,
            use_audit_cost,
            base_sigma,
        )
        metrics = {"loss": loss, **metrics}
        batch_count = len(batch[0])
        count += batch_count
        for name, value in metrics.items():
            if name == "actor_update_rms_max":
                totals[name] = max(totals.get(name, 0.0), float(value))
            else:
                totals[name] = totals.get(name, 0.0) + float(value) * batch_count
    return {
        name: value if name == "actor_update_rms_max" else value / count
        for name, value in totals.items()
    }


def train_one(
    temperature: float,
    args: argparse.Namespace,
    datasets: dict[str, AWRDataset],
    parent: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    set_seed(args.seed)
    device = torch.device(args.device)
    actor = load_actor(parent, device)
    for name, parameter in actor.named_parameters():
        parameter.requires_grad_(name.startswith(("fusion.", "output.")))
    trainable = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=40, min_lr=1e-6
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True, generator=generator
    )
    base_sigma = torch.tensor((0.25, 0.35), device=device).reshape(1, 1, 2)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        actor.eval()
        for raw in loader:
            batch = tuple(value.to(device) for value in raw)
            loss, _ = awr_loss(
                actor,
                batch,
                temperature,
                args.bc_anchor_weight,
                False,
                base_sigma,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
        validation = evaluate(
            actor, datasets["validation"], device, args, temperature, True
        )
        scheduler.step(validation["loss"])
        if validation["loss"] < best_loss - 1e-7:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in actor.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            history.append(
                {
                    "epoch": epoch,
                    "validation": validation,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
        if stale >= args.patience:
            break
    if best_state is None:
        raise AssertionError("AWR produced no checkpoint")
    actor.load_state_dict(best_state)
    metrics = {
        "temperature": temperature,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_validation_loss": best_loss,
        "trainable_parameter_count": sum(value.numel() for value in trainable),
        "splits": {
            split: evaluate(actor, dataset, device, args, temperature, split != "train")
            for split, dataset in datasets.items()
        },
        "history": history,
    }
    checkpoint = dict(parent)
    checkpoint.update(
        {
            "format_version": 2,
            "model_state_dict": best_state,
            "parent_actor_checkpoint": str(args.actor_checkpoint.resolve()),
            "local_critic_labels": str(args.labels.resolve()),
            "fine_tuning": {
                "method": "fixed_dbm_reward_weighted_regression",
                "temperature": temperature,
                "bc_anchor_weight": args.bc_anchor_weight,
                "supported_center_count": SUPPORTED_CENTER_COUNT,
                "seed": args.seed,
                "best_epoch": best_epoch,
                "trainable_modules": ["fusion", "output"],
            },
            "metrics": metrics,
        }
    )
    return metrics, checkpoint


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    parent = torch.load(args.actor_checkpoint, map_location="cpu")
    normalization = MPPIProposalNormalization.from_dict(parent["normalization"])
    split_payload = json.loads((args.labels / "splits.json").read_text())
    datasets = {
        split: AWRDataset(args.source, args.labels, split_payload[split], normalization)
        for split in ("train", "validation", "test")
    }
    temperatures = [
        float(value) for value in args.temperatures.split(",") if value.strip()
    ]
    runs = []
    for temperature in temperatures:
        metrics, checkpoint = train_one(temperature, args, datasets, parent)
        path = (args.output_dir / f"temperature{temperature:g}.pt").resolve()
        torch.save(checkpoint, path)
        metrics["checkpoint"] = str(path)
        runs.append(metrics)
        print(json.dumps({"run": path.stem, **metrics}, indent=2))
    selected = min(runs, key=lambda row: row["best_validation_loss"])
    summary = {
        "format_version": 1,
        "method": "fixed_dbm_reward_weighted_regression",
        "source_collection": str(args.source.resolve()),
        "local_labels": str(args.labels.resolve()),
        "parent_actor_checkpoint": str(args.actor_checkpoint.resolve()),
        "split_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "bc_anchor_weight": args.bc_anchor_weight,
        "supported_center_count": SUPPORTED_CENTER_COUNT,
        "runs": runs,
        "label_fit_selected_checkpoint": selected["checkpoint"],
        "selection_warning": "Select the deployable checkpoint with real DBM validation costs.",
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({"status": "ok", "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
