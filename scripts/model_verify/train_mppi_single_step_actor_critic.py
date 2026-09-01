#!/usr/bin/env python3
"""Train a fully on-support one-step actor-critic for two-pass MPPI.

The action is a categorical positive trust radius along the feedback Critic's
normalized direction: 0, 0.03, 0.06, 0.10, or 0.15 sigma.  Every action is
already evaluated with repeated DBM rollouts for every replay context, so the Q
update never relies on an unevaluated action.  Episode and reward-seed audit
splits remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from car_foundation.mppi_proposal_policy import (
    TorchMPPIFeedbackDiscreteStepActor,
    TorchMPPIFeedbackDiscreteStepCritic,
)
from train_mppi_two_pass_step_risk_critic import load_partition


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
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/single_step_actor_critic_20260805_v1"
)
ACTION_INDICES = np.asarray((0, 1, 3, 5, 7), np.int64)
ACTION_RADII = np.asarray((0.0, 0.03, 0.06, 0.10, 0.15), np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--critic-seeds", default="0,1,2")
    parser.add_argument("--actor-seeds", default="10,11,12")
    parser.add_argument("--critic-epochs", type=int, default=160)
    parser.add_argument("--actor-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--uncertainty-grid", default="0,0.25,0.5,1,2")
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


def normalize_context(
    context: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return ((context - mean) / std).astype(np.float32)


def policy_metrics(advantage: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    row = np.arange(len(selected))
    chosen = advantage[row, selected]
    return {
        "context_count": int(len(selected)),
        "mean_advantage": float(chosen.mean()),
        "median_advantage": float(np.median(chosen)),
        "p05_advantage": float(np.quantile(chosen, 0.05)),
        "p10_advantage": float(np.quantile(chosen, 0.10)),
        "win_fraction": float(np.mean(chosen > 0.0)),
        "loss_fraction": float(np.mean(chosen < 0.0)),
        "worst_advantage": float(chosen.min()),
        "action_histogram": np.bincount(
            selected, minlength=len(ACTION_RADII)
        ).tolist(),
        "mean_radius_sigma": float(ACTION_RADII[selected].mean()),
    }


def train_critic(
    seed: int,
    train_context: np.ndarray,
    train_target: np.ndarray,
    validation_context: np.ndarray,
    validation_target: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TorchMPPIFeedbackDiscreteStepCritic, dict[str, Any]]:
    set_seed(seed)
    model = TorchMPPIFeedbackDiscreteStepCritic(
        len(ACTION_RADII), args.dropout
    ).to(device)
    train_dataset = TensorDataset(
        torch.from_numpy(train_context), torch.from_numpy(train_target)
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_context), torch.from_numpy(validation_target)
    )
    loader = DataLoader(
        train_dataset, args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(validation_dataset, args.batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=10, min_lr=3e-7
    )
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    for epoch in range(1, args.critic_epochs + 1):
        model.train()
        for context, target in loader:
            prediction = model(context.to(device))
            target = target.to(device)
            regression = F.smooth_l1_loss(prediction, target, beta=0.25)
            target_difference = target[:, :, None] - target[:, None, :]
            prediction_difference = prediction[:, :, None] - prediction[:, None, :]
            valid = target_difference.abs() > 0.02
            ranking = F.softplus(
                -target_difference[valid].sign() * prediction_difference[valid]
            ).mean()
            loss = regression + 0.15 * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for context, target in validation_loader:
                losses.append(
                    F.smooth_l1_loss(
                        model(context.to(device)), target.to(device), beta=0.25
                    ).item()
                )
        validation_loss = float(np.mean(losses))
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
        if stale >= args.patience:
            break
    if best_state is None:
        raise AssertionError("critic training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_validation_loss": best_loss,
    }


@torch.no_grad()
def predict_critics(
    models: list[TorchMPPIFeedbackDiscreteStepCritic],
    context: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for begin in range(0, len(context), batch_size):
        batch = torch.from_numpy(context[begin : begin + batch_size]).to(device)
        values.append(
            torch.stack([model(batch) for model in models]).cpu().numpy()
        )
    prediction = np.concatenate(values, axis=1)
    return prediction.mean(axis=0), prediction.std(axis=0)


def train_actor(
    seed: int,
    train_context: np.ndarray,
    train_action: np.ndarray,
    validation_context: np.ndarray,
    validation_advantage: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TorchMPPIFeedbackDiscreteStepActor, dict[str, Any]]:
    set_seed(seed)
    model = TorchMPPIFeedbackDiscreteStepActor(
        len(ACTION_RADII), args.dropout
    ).to(device)
    dataset = TensorDataset(
        torch.from_numpy(train_context), torch.from_numpy(train_action)
    )
    loader = DataLoader(
        dataset, args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_score, best_epoch, best_state, stale = -float("inf"), 0, None, 0
    for epoch in range(1, args.actor_epochs + 1):
        model.train()
        for context, target in loader:
            loss = F.cross_entropy(model(context.to(device)), target.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            selected = model(
                torch.from_numpy(validation_context).to(device)
            ).argmax(dim=1).cpu().numpy()
        score = policy_metrics(validation_advantage, selected)["mean_advantage"]
        if score > best_score + 1e-6:
            best_score, best_epoch = score, epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise AssertionError("actor training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_validation_mean_advantage": best_score,
    }


@torch.no_grad()
def actor_actions(
    model: TorchMPPIFeedbackDiscreteStepActor,
    context: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    return model(torch.from_numpy(context).to(device)).argmax(1).cpu().numpy()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    splits = json.loads((args.labels / "splits.json").read_text())
    selection = {
        split: load_partition(
            args.source, args.parent_labels, args.labels, splits[split], "selection"
        )
        for split in ("train", "validation", "test")
    }
    audit_test = load_partition(
        args.source, args.parent_labels, args.labels, splits["test"], "audit"
    )
    context_mean = selection["train"].context.mean(0).astype(np.float32)
    context_std = np.maximum(
        selection["train"].context.std(0), 1e-4
    ).astype(np.float32)
    advantage_scale = float(
        max(selection["train"].advantage_mean[:, ACTION_INDICES].std(), 1.0)
    )
    contexts = {
        split: normalize_context(values.context, context_mean, context_std)
        for split, values in selection.items()
    }
    audit_context = normalize_context(audit_test.context, context_mean, context_std)
    advantages = {
        split: values.advantage_mean[:, ACTION_INDICES]
        for split, values in selection.items()
    }
    audit_advantage = audit_test.advantage_mean[:, ACTION_INDICES]
    targets = {
        split: (value / advantage_scale).astype(np.float32)
        for split, value in advantages.items()
    }

    critic_models, critic_runs, critic_paths = [], [], []
    critic_seeds = [int(value) for value in args.critic_seeds.split(",")]
    for seed in critic_seeds:
        model, run = train_critic(
            seed, contexts["train"], targets["train"], contexts["validation"],
            targets["validation"], args, device,
        )
        path = (args.output_dir / f"critic_seed{seed}.pt").resolve()
        torch.save({
            "format_version": 1,
            "model_type": type(model).__name__,
            "model_state_dict": model.state_dict(),
            "context_mean": context_mean,
            "context_std": context_std,
            "advantage_scale": advantage_scale,
            "action_radii_sigma": ACTION_RADII,
            "action_source_indices": ACTION_INDICES,
            "training": run,
        }, path)
        run["checkpoint"] = str(path)
        critic_models.append(model)
        critic_runs.append(run)
        critic_paths.append(str(path))

    q_prediction: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "validation", "test"):
        mean, std = predict_critics(
            critic_models, contexts[split], device, args.batch_size
        )
        q_prediction[split] = (mean * advantage_scale, std * advantage_scale)
    audit_q_mean, audit_q_std = predict_critics(
        critic_models, audit_context, device, args.batch_size
    )
    audit_q_mean *= advantage_scale
    audit_q_std *= advantage_scale

    uncertainty_grid = [float(value) for value in args.uncertainty_grid.split(",")]
    beta_rows = []
    for beta in uncertainty_grid:
        selected = np.argmax(
            q_prediction["validation"][0] - beta * q_prediction["validation"][1],
            axis=1,
        )
        beta_rows.append({
            "uncertainty_penalty": beta,
            **policy_metrics(advantages["validation"], selected),
        })
    selected_beta_row = max(beta_rows, key=lambda value: value["mean_advantage"])
    selected_beta = float(selected_beta_row["uncertainty_penalty"])
    train_policy_target = np.argmax(
        q_prediction["train"][0] - selected_beta * q_prediction["train"][1],
        axis=1,
    ).astype(np.int64)

    actor_models, actor_runs, actor_paths = [], [], []
    actor_seeds = [int(value) for value in args.actor_seeds.split(",")]
    for seed in actor_seeds:
        model, run = train_actor(
            seed, contexts["train"], train_policy_target,
            contexts["validation"], advantages["validation"], args, device,
        )
        validation_selected = actor_actions(
            model, contexts["validation"], device
        )
        run["validation_metrics"] = policy_metrics(
            advantages["validation"], validation_selected
        )
        path = (args.output_dir / f"actor_seed{seed}.pt").resolve()
        torch.save({
            "format_version": 1,
            "model_type": type(model).__name__,
            "model_state_dict": model.state_dict(),
            "context_mean": context_mean,
            "context_std": context_std,
            "advantage_scale": advantage_scale,
            "action_radii_sigma": ACTION_RADII,
            "action_source_indices": ACTION_INDICES,
            "uncertainty_penalty": selected_beta,
            "critic_checkpoints": critic_paths,
            "training": run,
        }, path)
        run["checkpoint"] = str(path)
        actor_models.append(model)
        actor_runs.append(run)
        actor_paths.append(str(path))
    best_actor_index = int(np.argmax([
        value["validation_metrics"]["mean_advantage"] for value in actor_runs
    ]))
    best_actor = actor_models[best_actor_index]
    test_selected = actor_actions(best_actor, contexts["test"], device)
    audit_selected = actor_actions(best_actor, audit_context, device)
    q_audit_selected = np.argmax(
        audit_q_mean - selected_beta * audit_q_std, axis=1
    )
    oracle_selected = np.argmax(audit_advantage, axis=1)
    fixed_metrics = {
        str(radius): policy_metrics(
            audit_advantage,
            np.full(len(audit_advantage), index, dtype=np.int64),
        )
        for index, radius in enumerate(ACTION_RADII.tolist())
    }
    summary = {
        "format_version": 1,
        "method": "fully-covered discrete contextual actor-critic",
        "source": str(args.source.resolve()),
        "parent_labels": str(args.parent_labels.resolve()),
        "replay_labels": str(args.labels.resolve()),
        "action_radii_sigma": ACTION_RADII.tolist(),
        "action_source_indices": ACTION_INDICES.tolist(),
        "split_context_counts": {
            split: len(value.context) for split, value in selection.items()
        },
        "advantage_scale": advantage_scale,
        "critic_runs": critic_runs,
        "critic_validation_penalty_search": beta_rows,
        "selected_uncertainty_penalty": selected_beta,
        "actor_runs": actor_runs,
        "selected_actor_checkpoint": actor_paths[best_actor_index],
        "selection_test_actor": policy_metrics(advantages["test"], test_selected),
        "audit_test_actor": policy_metrics(audit_advantage, audit_selected),
        "audit_test_direct_q": policy_metrics(audit_advantage, q_audit_selected),
        "audit_test_fixed_actions": fixed_metrics,
        "audit_test_oracle": policy_metrics(audit_advantage, oracle_selected),
        "audit_protocol": (
            "Critic/actor training and checkpoint selection use selection rewards. "
            "Audit first-pass feedback and reward seeds are used once for final test."
        ),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({
        "status": "ok", "output": str(args.output_dir),
        "selected_beta": selected_beta,
        "validation": actor_runs[best_actor_index]["validation_metrics"],
        "audit_test": summary["audit_test_actor"],
        "audit_oracle": summary["audit_test_oracle"],
    }, indent=2))


if __name__ == "__main__":
    main()
