#!/usr/bin/env python3
"""Train a one-step Actor-Critic over the feedback-derived direction bank."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import MPPIProposalNormalization
from train_mppi_single_step_state_actor_critic import (
    build_inputs,
    dataset,
    predict,
    train_model,
)
from train_mppi_two_pass_feedback_critic import load_partition as load_state_partition


DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1"
)
DEFAULT_PARENT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_feedback_diverse_20260805_v1"
)
DEFAULT_RISK = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_risk_replay_diverse_20260805_v1"
)
DEFAULT_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_multidirection_replay_diverse_20260805_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/multidirection_actor_critic_20260805_v1"
)


@dataclass
class RewardArrays:
    context: np.ndarray
    advantage_mean: np.ndarray
    advantage_by_seed: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--risk-labels", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--critic-seeds", default="40,41,42")
    parser.add_argument("--actor-seeds", default="50,51,52")
    parser.add_argument("--critic-epochs", type=int, default=180)
    parser.add_argument("--actor-epochs", type=int, default=140)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--tail-mix-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument(
        "--actor-target", choices=("critic", "reward_oracle"), default="critic"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_rewards(
    parent_root: Path,
    risk_root: Path,
    label_root: Path,
    episodes: list[str],
    partition: str,
) -> RewardArrays:
    prefix = "" if partition == "selection" else "audit_"
    parent_prefix = "" if partition == "selection" else "audit_"
    episode_set = set(episodes)
    contexts, means, seeds = [], [], []
    for path in sorted(label_root.glob("episode_*/*.npz")):
        if path.parent.name not in episode_set:
            continue
        parent_path = parent_root / path.parent.name / path.name
        risk_path = risk_root / path.parent.name / path.name
        with np.load(path, allow_pickle=False) as label, np.load(
            parent_path, allow_pickle=False
        ) as parent, np.load(risk_path, allow_pickle=False) as risk:
            feedback = np.asarray(
                parent[f"{parent_prefix}first_pass_feedback"], np.float32
            )
            gradient_mean = np.asarray(
                risk[f"{prefix}critic_gradient_mean"], np.float32
            )
            gradient_std = np.asarray(
                risk[f"{prefix}critic_gradient_std"], np.float32
            )
            context = np.concatenate((feedback, gradient_mean, gradient_std), axis=1)
            cost = np.asarray(
                label[f"{prefix}proposal_output_cost_by_seed"], np.float32
            )
            advantage = cost[:, :1] - cost
            contexts.extend(context)
            means.extend(advantage.mean(axis=2))
            seeds.extend(advantage)
    if not contexts:
        raise ValueError(f"no {partition} multidirection rewards")
    return RewardArrays(
        np.asarray(contexts, np.float32),
        np.asarray(means, np.float32),
        np.asarray(seeds, np.float32),
    )


def policy_metrics(
    advantage: np.ndarray,
    selected: np.ndarray,
    action_names: list[str],
) -> dict[str, Any]:
    chosen = advantage[np.arange(len(selected)), selected]
    return {
        "context_count": len(selected),
        "mean_advantage": float(chosen.mean()),
        "median_advantage": float(np.median(chosen)),
        "p05_advantage": float(np.quantile(chosen, 0.05)),
        "p10_advantage": float(np.quantile(chosen, 0.10)),
        "win_fraction": float(np.mean(chosen > 0.0)),
        "loss_fraction": float(np.mean(chosen < 0.0)),
        "worst_advantage": float(chosen.min()),
        "action_histogram": np.bincount(
            selected, minlength=len(action_names)
        ).tolist(),
        "top_actions": [
            {"name": action_names[index], "count": int(count)}
            for index, count in sorted(
                enumerate(np.bincount(selected, minlength=len(action_names))),
                key=lambda value: value[1], reverse=True,
            )[:8]
        ],
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    summary = json.loads((args.labels / "summary.json").read_text())
    action_names = list(summary["action_names"])
    action_count = len(action_names)
    splits = json.loads((args.labels / "splits.json").read_text())
    device = torch.device(args.device)
    state_selection = {
        split: load_state_partition(
            args.source, args.parent_labels, splits[split], "selection"
        ) for split in ("train", "validation", "test")
    }
    reward_selection = {
        split: load_rewards(
            args.parent_labels, args.risk_labels, args.labels,
            splits[split], "selection",
        ) for split in ("train", "validation", "test")
    }
    state_audit = load_state_partition(
        args.source, args.parent_labels, splits["test"], "audit"
    )
    reward_audit = load_rewards(
        args.parent_labels, args.risk_labels, args.labels,
        splits["test"], "audit",
    )
    train_state = state_selection["train"]
    train_reward = reward_selection["train"]
    state_norm = MPPIProposalNormalization.fit(
        train_state.history, train_state.reference, train_state.current
    )
    feedback_mean = train_reward.context[:, :74].mean(0).astype(np.float32)
    feedback_std = np.maximum(
        train_reward.context[:, :74].std(0), 1e-4
    ).astype(np.float32)
    gradient_mean = train_reward.context[:, 74:].mean(0).astype(np.float32)
    gradient_std = np.maximum(
        train_reward.context[:, 74:].std(0), 1e-4
    ).astype(np.float32)
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
        split: value.advantage_mean for split, value in reward_selection.items()
    }
    selection_tail = {
        split: np.quantile(value.advantage_by_seed, 0.10, axis=2).astype(np.float32)
        for split, value in reward_selection.items()
    }
    audit_mean = reward_audit.advantage_mean
    audit_tail = np.quantile(
        reward_audit.advantage_by_seed, 0.10, axis=2
    ).astype(np.float32)
    advantage_scale = float(max(selection_mean["train"].std(), 1.0))
    critic_targets = {
        split: np.concatenate(
            (selection_mean[split], selection_tail[split]), axis=1
        ).astype(np.float32) / advantage_scale
        for split in ("train", "validation", "test")
    }
    critic_models, critic_runs, critic_paths = [], [], []
    for seed in (int(value) for value in args.critic_seeds.split(",")):
        model, run = train_model(
            seed, 2 * action_count,
            dataset(selection_inputs["train"], critic_targets["train"]),
            dataset(selection_inputs["validation"], critic_targets["validation"]),
            args, False,
        )
        path = (args.output_dir / f"critic_seed{seed}.pt").resolve()
        torch.save({"model_state_dict": model.state_dict(), "training": run}, path)
        run["checkpoint"] = str(path)
        critic_models.append(model); critic_runs.append(run); critic_paths.append(str(path))
    predictions = {}
    for split in ("train", "validation", "test"):
        mean, std = predict(
            critic_models, selection_inputs[split], args.batch_size, device
        )
        predictions[split] = (mean * advantage_scale, std * advantage_scale)
    mix_rows = []
    for mix in (float(value) for value in args.tail_mix_grid.split(",")):
        mean, tail = predictions["validation"][0].reshape(
            -1, 2, action_count
        ).transpose(1, 0, 2)
        if args.actor_target == "reward_oracle":
            selected = (
                (1.0 - mix) * selection_mean["validation"]
                + mix * selection_tail["validation"]
            ).argmax(1)
        else:
            selected = ((1.0 - mix) * mean + mix * tail).argmax(1)
        chosen_tail = selection_tail["validation"][np.arange(len(selected)), selected]
        mix_rows.append({
            "tail_mix": mix,
            **policy_metrics(selection_mean["validation"], selected, action_names),
            "selected_true_p10_mean": float(chosen_tail.mean()),
        })
    best_mix_row = max(
        mix_rows,
        key=lambda value: (
            value["mean_advantage"], value["selected_true_p10_mean"]
        ),
    )
    best_mix = float(best_mix_row["tail_mix"])
    train_mean, train_tail = predictions["train"][0].reshape(
        -1, 2, action_count
    ).transpose(1, 0, 2)
    if args.actor_target == "reward_oracle":
        actor_target = (
            (1.0 - best_mix) * selection_mean["train"]
            + best_mix * selection_tail["train"]
        ).argmax(1).astype(np.int64)
    else:
        actor_target = (
            (1.0 - best_mix) * train_mean + best_mix * train_tail
        ).argmax(1).astype(np.int64)
    validation_mean, validation_tail = predictions["validation"][0].reshape(
        -1, 2, action_count
    ).transpose(1, 0, 2)
    if args.actor_target == "reward_oracle":
        validation_target = (
            (1.0 - best_mix) * selection_mean["validation"]
            + best_mix * selection_tail["validation"]
        ).argmax(1).astype(np.int64)
    else:
        validation_target = (
            (1.0 - best_mix) * validation_mean + best_mix * validation_tail
        ).argmax(1).astype(np.int64)
    actor_models, actor_runs, actor_paths = [], [], []
    for seed in (int(value) for value in args.actor_seeds.split(",")):
        model, run = train_model(
            seed, action_count,
            dataset(selection_inputs["train"], actor_target),
            dataset(selection_inputs["validation"], validation_target),
            args, True,
        )
        selected = predict(
            [model], selection_inputs["validation"], args.batch_size, device
        )[0].argmax(1)
        run["validation_metrics"] = policy_metrics(
            selection_mean["validation"], selected, action_names
        )
        path = (args.output_dir / f"actor_seed{seed}.pt").resolve()
        torch.save({
            "format_version": 1,
            "model_type": type(model).__name__,
            "input_type": "multidirection_state_feedback",
            "model_state_dict": model.state_dict(),
            "state_normalization": state_norm.to_dict(),
            "feedback_mean": feedback_mean,
            "feedback_std": feedback_std,
            "gradient_mean": gradient_mean,
            "gradient_std": gradient_std,
            "action_names": action_names,
            "direction_names": summary["direction_names"],
            "radii_sigma": summary["radii_sigma"],
            "tail_mix": best_mix,
            "actor_target": args.actor_target,
            "critic_checkpoints": critic_paths,
            "training": run,
        }, path)
        run["checkpoint"] = str(path)
        actor_models.append(model); actor_runs.append(run); actor_paths.append(str(path))
    best_actor_index = int(np.argmax([
        run["validation_metrics"]["mean_advantage"] for run in actor_runs
    ]))
    best_actor = actor_models[best_actor_index]
    test_selected = predict(
        [best_actor], selection_inputs["test"], args.batch_size, device
    )[0].argmax(1)
    audit_selected = predict(
        [best_actor], audit_inputs, args.batch_size, device
    )[0].argmax(1)
    audit_chosen_tail = audit_tail[np.arange(len(audit_selected)), audit_selected]
    audit_oracle = audit_mean.argmax(1)
    result = {
        "format_version": 1,
        "method": "state+feedback multidirection mean/tail actor-critic",
        "actor_target": args.actor_target,
        "labels": str(args.labels.resolve()),
        "action_names": action_names,
        "advantage_scale": advantage_scale,
        "critic_runs": critic_runs,
        "validation_tail_mix_search": mix_rows,
        "selected_tail_mix": best_mix,
        "actor_runs": actor_runs,
        "selected_actor_checkpoint": actor_paths[best_actor_index],
        "selection_test_actor": policy_metrics(
            selection_mean["test"], test_selected, action_names
        ),
        "audit_test_actor": policy_metrics(
            audit_mean, audit_selected, action_names
        ),
        "audit_selected_p10_mean": float(audit_chosen_tail.mean()),
        "audit_test_oracle": policy_metrics(
            audit_mean, audit_oracle, action_names
        ),
        "audit_protocol": (
            "Training/model selection use selection train/validation only. "
            "Audit/test rewards are read once for final replay qualification."
        ),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({
        "status": "ok", "output": str(args.output_dir),
        "tail_mix": best_mix, "audit_test": result["audit_test_actor"],
        "audit_oracle": result["audit_test_oracle"],
    }, indent=2))


if __name__ == "__main__":
    main()
