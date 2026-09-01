#!/usr/bin/env python3
"""Conservatively fine-tune a BC proposal actor against a frozen critic ensemble."""

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
from train_mppi_proposal_critic import load_critic_checkpoint


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
DEFAULT_CRITIC = Path(
    "outputs/mppi_proposal/critic_local_diverse_20260805_v2/training_summary.json"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/actor_critic_local_20260805_v1")


@dataclass
class ActorArrays:
    history: np.ndarray
    reference: np.ndarray
    current: np.ndarray
    warm: np.ndarray
    baseline: np.ndarray
    teacher: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--actor-checkpoint", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--critic-summary", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--anchor-weights", default="0.25,1.0,4.0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--uncertainty-weight", type=float, default=0.25)
    parser.add_argument("--maximum-update-rms", type=float, default=0.40)
    parser.add_argument("--trust-penalty", type=float, default=20.0)
    parser.add_argument("--boundary-penalty", type=float, default=2.0)
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


def load_arrays(
    source_root: Path, label_root: Path, episodes: list[str]
) -> ActorArrays:
    rows: dict[str, list[np.ndarray]] = {
        name: []
        for name in ("history", "reference", "current", "warm", "baseline", "teacher")
    }
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
            rows["history"].append(np.asarray(source["history"][0], dtype=np.float32))
            rows["reference"].append(
                ego_reference_features(source["reference_ego"], float(state[3]))
            )
            rows["current"].append(
                np.asarray((state[3], state[4], *action), dtype=np.float32)
            )
            centers = np.asarray(label["shortlist_centers"], dtype=np.float32)
            rows["warm"].append(centers[int(label["warm_shortlist_index"])])
            rows["baseline"].append(centers[int(label["network_shortlist_index"])])
            rows["teacher"].append(centers[int(label["teacher_shortlist_index"])])
    if not rows["history"]:
        raise ValueError(f"no actor data for {episodes}")
    return ActorArrays(**{name: np.asarray(value, dtype=np.float32) for name, value in rows.items()})


class ActorDataset(Dataset):
    def __init__(
        self, arrays: ActorArrays, normalization: MPPIProposalNormalization
    ) -> None:
        history, reference, current = normalization.normalize_numpy(
            arrays.history, arrays.reference, arrays.current
        )
        self.values = tuple(
            torch.from_numpy(value.astype(np.float32))
            for value in (
                history,
                reference,
                current,
                arrays.warm,
                arrays.baseline,
                arrays.teacher,
            )
        )

    def __len__(self) -> int:
        return len(self.values[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return tuple(value[index] for value in self.values)


def load_actor(path: Path, device: torch.device) -> tuple[TorchMPPIProposalPolicy, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    architecture = checkpoint["architecture"]
    model = TorchMPPIProposalPolicy(
        trust_scale=tuple(architecture["trust_scale"]),
        dropout=float(architecture["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), checkpoint


def critic_values(
    critics: list[torch.nn.Module],
    history: torch.Tensor,
    reference: torch.Tensor,
    current: torch.Tensor,
    warm: torch.Tensor,
    center: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        [critic(history, reference, current, warm, center) for critic in critics]
    )


def objective(
    actor: TorchMPPIProposalPolicy,
    critics: list[torch.nn.Module],
    batch: tuple[torch.Tensor, ...],
    base_sigma: torch.Tensor,
    anchor_weight: float,
    uncertainty_weight: float,
    maximum_update_rms: float,
    trust_penalty: float,
    boundary_penalty: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    history, reference, current, warm, baseline, teacher = batch
    _, center = actor(history, reference, current, warm)
    values = critic_values(critics, history, reference, current, warm, center)
    mean = values.mean(0)
    uncertainty = values.std(0, unbiased=False)
    robust = mean - uncertainty_weight * uncertainty
    standardized_update = (center - baseline) / base_sigma
    update_rms = torch.sqrt(standardized_update.square().mean(dim=(1, 2)) + 1e-12)
    anchor = standardized_update.square().mean()
    excess = torch.relu(update_rms - maximum_update_rms).square().mean()
    boundary = torch.relu(center.abs() - 0.98).square().mean()
    loss = (
        -robust.mean()
        + anchor_weight * anchor
        + trust_penalty * excess
        + boundary_penalty * boundary
    )
    teacher_distance = torch.sqrt(
        (((center - teacher) / base_sigma).square().mean(dim=(1, 2))) + 1e-12
    )
    baseline_teacher_distance = torch.sqrt(
        (((baseline - teacher) / base_sigma).square().mean(dim=(1, 2))) + 1e-12
    )
    return loss, {
        "robust_advantage": robust.mean(),
        "critic_mean": mean.mean(),
        "uncertainty": uncertainty.mean(),
        "anchor_mse": anchor,
        "update_rms": update_rms.mean(),
        "update_rms_max": update_rms.max(),
        "trust_excess": excess,
        "boundary": boundary,
        "teacher_distance": teacher_distance.mean(),
        "baseline_teacher_distance": baseline_teacher_distance.mean(),
        "closer_to_teacher": (teacher_distance < baseline_teacher_distance).float().mean(),
    }


@torch.no_grad()
def evaluate(
    actor: TorchMPPIProposalPolicy,
    critics: list[torch.nn.Module],
    dataset: ActorDataset,
    device: torch.device,
    args: argparse.Namespace,
    anchor_weight: float,
) -> dict[str, float]:
    actor.eval()
    sums: dict[str, float] = {}
    count = 0
    base_sigma = torch.tensor((0.25, 0.35), device=device).reshape(1, 1, 2)
    for raw in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
        batch = tuple(value.to(device) for value in raw)
        loss, metrics = objective(
            actor,
            critics,
            batch,
            base_sigma,
            anchor_weight,
            args.uncertainty_weight,
            args.maximum_update_rms,
            args.trust_penalty,
            args.boundary_penalty,
        )
        metrics = {"loss": loss, **metrics}
        batch_count = len(batch[0])
        count += batch_count
        for name, value in metrics.items():
            # update_rms_max is intentionally reduced as a maximum below.
            if name == "update_rms_max":
                sums[name] = max(sums.get(name, 0.0), float(value))
            else:
                sums[name] = sums.get(name, 0.0) + float(value) * batch_count
    return {
        name: value if name == "update_rms_max" else value / count
        for name, value in sums.items()
    }


def train_one(
    anchor_weight: float,
    args: argparse.Namespace,
    datasets: dict[str, ActorDataset],
    critics: list[torch.nn.Module],
    actor_checkpoint: dict[str, Any],
    normalization: MPPIProposalNormalization,
) -> tuple[dict[str, Any], dict[str, Any]]:
    set_seed(args.seed)
    device = torch.device(args.device)
    actor, _ = load_actor(args.actor_checkpoint, device)
    for name, parameter in actor.named_parameters():
        parameter.requires_grad_(name.startswith(("fusion.", "output.")))
    trainable = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=25, min_lr=1e-6
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    base_sigma = torch.tensor((0.25, 0.35), device=device).reshape(1, 1, 2)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history_rows = []
    for epoch in range(1, args.epochs + 1):
        # Deterministic dropout behavior is deliberate for a small policy update.
        actor.eval()
        for raw in loader:
            batch = tuple(value.to(device) for value in raw)
            loss, _ = objective(
                actor,
                critics,
                batch,
                base_sigma,
                anchor_weight,
                args.uncertainty_weight,
                args.maximum_update_rms,
                args.trust_penalty,
                args.boundary_penalty,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
        validation = evaluate(
            actor, critics, datasets["validation"], device, args, anchor_weight
        )
        scheduler.step(validation["loss"])
        if validation["loss"] < best_loss - 1e-6:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in actor.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0:
            history_rows.append(
                {
                    "epoch": epoch,
                    "validation": validation,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
        if stale >= args.patience:
            break
    if best_state is None:
        raise AssertionError("actor fine-tuning produced no checkpoint")
    actor.load_state_dict(best_state)
    metrics = {
        "anchor_weight": anchor_weight,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_validation_loss": best_loss,
        "trainable_parameter_count": sum(value.numel() for value in trainable),
        "splits": {
            split: evaluate(actor, critics, dataset, device, args, anchor_weight)
            for split, dataset in datasets.items()
        },
        "history": history_rows,
    }
    checkpoint = dict(actor_checkpoint)
    checkpoint.update(
        {
            "format_version": 2,
            "model_state_dict": best_state,
            "parent_actor_checkpoint": str(args.actor_checkpoint.resolve()),
            "critic_summary": str(args.critic_summary.resolve()),
            "local_critic_labels": str(args.labels.resolve()),
            "fine_tuning": {
                "method": "frozen_critic_ensemble_with_bc_anchor",
                "seed": args.seed,
                "anchor_weight": anchor_weight,
                "uncertainty_weight": args.uncertainty_weight,
                "maximum_update_rms": args.maximum_update_rms,
                "trust_penalty": args.trust_penalty,
                "boundary_penalty": args.boundary_penalty,
                "learning_rate": args.learning_rate,
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
    device = torch.device(args.device)
    actor_checkpoint = torch.load(args.actor_checkpoint, map_location="cpu")
    normalization = MPPIProposalNormalization.from_dict(actor_checkpoint["normalization"])
    critic_summary = json.loads(args.critic_summary.read_text())
    critics = []
    scales = []
    for checkpoint_path in critic_summary["ensemble_checkpoints"]:
        critic, checkpoint = load_critic_checkpoint(Path(checkpoint_path), device)
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
        critics.append(critic)
        scales.append(float(checkpoint["advantage_scale"]))
        if checkpoint["normalization"] != normalization.to_dict():
            raise AssertionError("actor and critic normalization differ")
    if not np.allclose(scales, scales[0], atol=0, rtol=0):
        raise AssertionError("critic target scales differ")
    split_payload = json.loads((args.labels / "splits.json").read_text())
    arrays = {
        split: load_arrays(args.source, args.labels, split_payload[split])
        for split in ("train", "validation", "test")
    }
    datasets = {
        split: ActorDataset(values, normalization) for split, values in arrays.items()
    }
    anchor_weights = [
        float(value) for value in args.anchor_weights.split(",") if value.strip()
    ]
    runs = []
    for anchor_weight in anchor_weights:
        metrics, checkpoint = train_one(
            anchor_weight,
            args,
            datasets,
            critics,
            actor_checkpoint,
            normalization,
        )
        path = (args.output_dir / f"anchor{anchor_weight:g}.pt").resolve()
        torch.save(checkpoint, path)
        metrics["checkpoint"] = str(path)
        runs.append(metrics)
        print(json.dumps({"run": path.stem, **metrics}, indent=2))
    selected = min(runs, key=lambda row: row["best_validation_loss"])
    summary = {
        "format_version": 1,
        "method": "frozen_critic_ensemble_with_bc_anchor",
        "source_collection": str(args.source.resolve()),
        "local_critic_labels": str(args.labels.resolve()),
        "parent_actor_checkpoint": str(args.actor_checkpoint.resolve()),
        "critic_summary": str(args.critic_summary.resolve()),
        "split_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "configuration": {
            name: getattr(args, name)
            for name in (
                "seed",
                "epochs",
                "patience",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "uncertainty_weight",
                "maximum_update_rms",
                "trust_penalty",
                "boundary_penalty",
            )
        },
        "runs": runs,
        "critic_selected_checkpoint": selected["checkpoint"],
        "selection_warning": (
            "Final checkpoint must be selected by real fixed-DBM validation rollouts, "
            "not by this critic-estimated validation objective alone."
        ),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({"status": "ok", "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
