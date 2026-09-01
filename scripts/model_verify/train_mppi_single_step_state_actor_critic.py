#!/usr/bin/env python3
"""Train a state+feedback one-step Actor-Critic with mean/tail DBM rewards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TorchMPPIFeedbackDiscreteStateNetwork,
)
from train_mppi_single_step_actor_critic import (
    ACTION_INDICES,
    ACTION_RADII,
    policy_metrics,
    set_seed,
)
from train_mppi_two_pass_feedback_critic import load_partition as load_state_partition
from train_mppi_two_pass_step_risk_critic import load_partition as load_reward_partition


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
    "outputs/mppi_proposal/single_step_state_actor_critic_20260805_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--critic-seeds", default="20,21,22")
    parser.add_argument("--actor-seeds", default="30,31,32")
    parser.add_argument("--critic-epochs", type=int, default=180)
    parser.add_argument("--actor-epochs", type=int, default=140)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--tail-mix-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_inputs(
    state: Any,
    reward: Any,
    state_norm: MPPIProposalNormalization,
    feedback_mean: np.ndarray,
    feedback_std: np.ndarray,
    gradient_mean: np.ndarray,
    gradient_std: np.ndarray,
) -> tuple[np.ndarray, ...]:
    if len(state.history) != len(reward.context):
        raise ValueError("state and reward context counts differ")
    feedback = reward.context[:, :74]
    if not np.allclose(feedback, state.feedback, atol=1e-5):
        raise ValueError("state and risk-replay feedback ordering differs")
    history, reference, current = state_norm.normalize_numpy(
        state.history, state.reference, state.current
    )
    return (
        history.astype(np.float32),
        reference.astype(np.float32),
        current.astype(np.float32),
        state.anchor.astype(np.float32),
        ((feedback - feedback_mean) / feedback_std).astype(np.float32),
        ((reward.context[:, 74:] - gradient_mean) / gradient_std).astype(np.float32),
    )


def dataset(inputs: tuple[np.ndarray, ...], target: np.ndarray) -> TensorDataset:
    return TensorDataset(
        *(torch.from_numpy(value) for value in inputs), torch.from_numpy(target)
    )


def forward(model: TorchMPPIFeedbackDiscreteStateNetwork, batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    values = tuple(value.to(device) for value in batch)
    return model(*values[:-1]), values[-1]


def train_model(
    seed: int,
    output_count: int,
    train_data: TensorDataset,
    validation_data: TensorDataset,
    args: argparse.Namespace,
    actor: bool,
) -> tuple[TorchMPPIFeedbackDiscreteStateNetwork, dict[str, Any]]:
    set_seed(seed)
    device = torch.device(args.device)
    model = TorchMPPIFeedbackDiscreteStateNetwork(output_count, args.dropout).to(device)
    loader = DataLoader(
        train_data, args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(validation_data, args.batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=12, min_lr=3e-7
    )
    epochs = args.actor_epochs if actor else args.critic_epochs
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            prediction, target = forward(model, batch, device)
            if actor:
                loss = F.cross_entropy(prediction, target.long())
            else:
                predicted_mean, predicted_tail = prediction.chunk(2, dim=1)
                target_mean, target_tail = target.chunk(2, dim=1)
                loss = F.smooth_l1_loss(
                    predicted_mean, target_mean, beta=0.25
                ) + F.smooth_l1_loss(predicted_tail, target_tail, beta=0.25)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in validation_loader:
                prediction, target = forward(model, batch, device)
                if actor:
                    loss = F.cross_entropy(prediction, target.long())
                else:
                    pm, pt = prediction.chunk(2, 1)
                    tm, tt = target.chunk(2, 1)
                    loss = F.smooth_l1_loss(pm, tm, beta=0.25) + F.smooth_l1_loss(pt, tt, beta=0.25)
                losses.append(loss.item())
        validation_loss = float(np.mean(losses))
        scheduler.step(validation_loss)
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise AssertionError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
        "best_validation_loss": best_loss,
    }


@torch.no_grad()
def predict(
    models: list[TorchMPPIFeedbackDiscreteStateNetwork],
    inputs: tuple[np.ndarray, ...],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    per_model = []
    for model in models:
        output = []
        data = TensorDataset(*(torch.from_numpy(value) for value in inputs))
        for batch in DataLoader(data, batch_size):
            output.append(model(*(value.to(device) for value in batch)).cpu().numpy())
        per_model.append(np.concatenate(output))
    values = np.asarray(per_model)
    return values.mean(0), values.std(0)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    splits = json.loads((args.labels / "splits.json").read_text())
    state_selection = {
        split: load_state_partition(
            args.source, args.parent_labels, splits[split], "selection"
        ) for split in ("train", "validation", "test")
    }
    reward_selection = {
        split: load_reward_partition(
            args.source, args.parent_labels, args.labels, splits[split], "selection"
        ) for split in ("train", "validation", "test")
    }
    state_audit = load_state_partition(
        args.source, args.parent_labels, splits["test"], "audit"
    )
    reward_audit = load_reward_partition(
        args.source, args.parent_labels, args.labels, splits["test"], "audit"
    )
    train_state, train_reward = state_selection["train"], reward_selection["train"]
    state_norm = MPPIProposalNormalization.fit(
        train_state.history, train_state.reference, train_state.current
    )
    feedback_mean = train_reward.context[:, :74].mean(0).astype(np.float32)
    feedback_std = np.maximum(train_reward.context[:, :74].std(0), 1e-4).astype(np.float32)
    gradient_mean = train_reward.context[:, 74:].mean(0).astype(np.float32)
    gradient_std = np.maximum(train_reward.context[:, 74:].std(0), 1e-4).astype(np.float32)
    selection_inputs = {
        split: build_inputs(
            state_selection[split], reward_selection[split], state_norm,
            feedback_mean, feedback_std, gradient_mean, gradient_std,
        ) for split in ("train", "validation", "test")
    }
    audit_inputs = build_inputs(
        state_audit, reward_audit, state_norm, feedback_mean, feedback_std,
        gradient_mean, gradient_std,
    )
    selection_mean = {
        split: value.advantage_mean[:, ACTION_INDICES]
        for split, value in reward_selection.items()
    }
    selection_tail = {
        split: np.quantile(
            value.advantage_by_seed[:, ACTION_INDICES], 0.10, axis=2
        ).astype(np.float32)
        for split, value in reward_selection.items()
    }
    audit_mean = reward_audit.advantage_mean[:, ACTION_INDICES]
    audit_tail = np.quantile(
        reward_audit.advantage_by_seed[:, ACTION_INDICES], 0.10, axis=2
    ).astype(np.float32)
    advantage_scale = float(max(selection_mean["train"].std(), 1.0))
    critic_targets = {
        split: np.concatenate((selection_mean[split], selection_tail[split]), 1).astype(np.float32) / advantage_scale
        for split in ("train", "validation", "test")
    }
    critic_train = dataset(selection_inputs["train"], critic_targets["train"])
    critic_validation = dataset(selection_inputs["validation"], critic_targets["validation"])
    critic_models, critic_runs, critic_paths = [], [], []
    for seed in (int(value) for value in args.critic_seeds.split(",")):
        model, run = train_model(seed, 10, critic_train, critic_validation, args, False)
        path = (args.output_dir / f"critic_seed{seed}.pt").resolve()
        torch.save({"model_state_dict": model.state_dict(), "training": run}, path)
        run["checkpoint"] = str(path)
        critic_models.append(model); critic_runs.append(run); critic_paths.append(str(path))
    prediction = {}
    for split in ("train", "validation", "test"):
        mean, std = predict(critic_models, selection_inputs[split], args.batch_size, device)
        prediction[split] = (mean * advantage_scale, std * advantage_scale)
    audit_prediction = predict(critic_models, audit_inputs, args.batch_size, device)
    audit_prediction = tuple(value * advantage_scale for value in audit_prediction)
    mix_rows = []
    for mix in (float(value) for value in args.tail_mix_grid.split(",")):
        predicted_mean, predicted_tail = prediction["validation"][0].reshape(-1, 2, len(ACTION_RADII)).transpose(1, 0, 2)
        score = (1.0 - mix) * predicted_mean + mix * predicted_tail
        selected = score.argmax(1)
        metrics = policy_metrics(selection_mean["validation"], selected)
        true_tail = selection_tail["validation"][np.arange(len(selected)), selected]
        mix_rows.append({"tail_mix": mix, **metrics, "selected_true_p10_mean": float(true_tail.mean())})
    # Mean cost remains the primary objective; validation P10 breaks near ties.
    best_mix_row = max(mix_rows, key=lambda value: (value["mean_advantage"], value["selected_true_p10_mean"]))
    best_mix = float(best_mix_row["tail_mix"])
    predicted_mean, predicted_tail = prediction["train"][0].reshape(-1, 2, len(ACTION_RADII)).transpose(1, 0, 2)
    actor_target = ((1.0 - best_mix) * predicted_mean + best_mix * predicted_tail).argmax(1).astype(np.int64)
    actor_train = dataset(selection_inputs["train"], actor_target)
    # Actor checkpoint selection is pure distillation validation loss.  The risk
    # mixture itself was selected on real DBM validation rewards above.
    validation_q_mean, validation_q_tail = prediction["validation"][0].reshape(-1, 2, len(ACTION_RADII)).transpose(1, 0, 2)
    validation_actor_target = ((1.0 - best_mix) * validation_q_mean + best_mix * validation_q_tail).argmax(1).astype(np.int64)
    actor_validation = dataset(selection_inputs["validation"], validation_actor_target)
    actor_models, actor_runs, actor_paths = [], [], []
    for seed in (int(value) for value in args.actor_seeds.split(",")):
        model, run = train_model(seed, 5, actor_train, actor_validation, args, True)
        selected = predict([model], selection_inputs["validation"], args.batch_size, device)[0].argmax(1)
        run["validation_metrics"] = policy_metrics(selection_mean["validation"], selected)
        path = (args.output_dir / f"actor_seed{seed}.pt").resolve()
        torch.save({
            "format_version": 1, "model_type": type(model).__name__,
            "input_type": "state_feedback", "model_state_dict": model.state_dict(),
            "state_normalization": state_norm.to_dict(),
            "feedback_mean": feedback_mean, "feedback_std": feedback_std,
            "gradient_mean": gradient_mean, "gradient_std": gradient_std,
            "action_radii_sigma": ACTION_RADII,
            "action_source_indices": ACTION_INDICES,
            "tail_mix": best_mix, "critic_checkpoints": critic_paths,
            "training": run,
        }, path)
        run["checkpoint"] = str(path)
        actor_models.append(model); actor_runs.append(run); actor_paths.append(str(path))
    best_actor_index = int(np.argmax([run["validation_metrics"]["mean_advantage"] for run in actor_runs]))
    best_actor = actor_models[best_actor_index]
    audit_selected = predict([best_actor], audit_inputs, args.batch_size, device)[0].argmax(1)
    test_selected = predict([best_actor], selection_inputs["test"], args.batch_size, device)[0].argmax(1)
    summary = {
        "format_version": 1, "method": "state+feedback discrete mean/tail actor-critic",
        "action_radii_sigma": ACTION_RADII.tolist(), "action_source_indices": ACTION_INDICES.tolist(),
        "advantage_scale": advantage_scale, "critic_runs": critic_runs,
        "validation_tail_mix_search": mix_rows, "selected_tail_mix": best_mix,
        "actor_runs": actor_runs, "selected_actor_checkpoint": actor_paths[best_actor_index],
        "selection_test_actor": policy_metrics(selection_mean["test"], test_selected),
        "audit_test_actor": policy_metrics(audit_mean, audit_selected),
        "audit_selected_p10_mean": float(audit_tail[np.arange(len(audit_selected)), audit_selected].mean()),
        "audit_protocol": "Training/model selection use selection train/validation only; audit/test is final replay qualification.",
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output_dir), "tail_mix": best_mix, "audit_test": summary["audit_test_actor"]}, indent=2))


if __name__ == "__main__":
    main()
