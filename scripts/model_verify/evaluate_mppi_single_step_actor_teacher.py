#!/usr/bin/env python3
"""Fresh-seed DBM comparison of a one-step actor and the T1 teacher center."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TorchMPPIFeedbackDiscreteStateNetwork,
    TorchMPPIFeedbackDiscreteStepActor,
    ego_reference_features,
)
from generate_dbm_multicenter_teacher import load_config, make_controller
from generate_dbm_two_pass_risk_replay_labels import evaluate_center_bank


DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1"
)
DEFAULT_PARENT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_feedback_diverse_20260805_v1"
)
DEFAULT_REPLAY = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_risk_replay_diverse_20260805_v1"
)
DEFAULT_TEACHER = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_teacher_t1_diverse_20260805_v1"
)
DEFAULT_ACTOR_DIR = Path(
    "outputs/mppi_proposal/single_step_actor_critic_20260805_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/single_step_actor_teacher_eval_20260805_v1"
)
METHODS = ("guided", "fixed_pos_0p10", "actor", "t1_teacher")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--replay-labels", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--teacher-labels", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--actor-dir", type=Path, default=DEFAULT_ACTOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--evaluation-seeds",
        default="27001,27002,27003,27004,27005,27006,27007,27008",
    )
    parser.add_argument("--samples-per-center-seed", type=int, default=64)
    parser.add_argument("--second-noise-scale", type=float, default=0.10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Return positive gains when right has lower cost than left."""
    seed_gain = left - right
    context_gain = seed_gain.mean(axis=2).reshape(-1)
    return {
        "mean_cost_reduction": float(seed_gain.mean()),
        "median_context_cost_reduction": float(np.median(context_gain)),
        "p05_context_cost_reduction": float(np.quantile(context_gain, 0.05)),
        "p10_context_cost_reduction": float(np.quantile(context_gain, 0.10)),
        "context_wins": int(np.sum(context_gain > 0.0)),
        "context_losses": int(np.sum(context_gain < 0.0)),
        "context_ties": int(np.sum(context_gain == 0.0)),
        "seed_win_fraction": float(np.mean(seed_gain > 0.0)),
        "worst_context_cost_reduction": float(context_gain.min()),
    }


def cost_metrics(cost: np.ndarray) -> dict[str, Any]:
    context = cost.mean(axis=2).reshape(-1)
    return {
        "mean": float(cost.mean()),
        "median_context": float(np.median(context)),
        "p90_context": float(np.quantile(context, 0.90)),
        "p95_context": float(np.quantile(context, 0.95)),
        "maximum_context": float(context.max()),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    seeds = [int(value) for value in args.evaluation_seeds.split(",")]
    if len(seeds) != len(set(seeds)):
        raise ValueError("evaluation seeds must be distinct")
    forbidden = set(range(24001, 24400)) | set(range(26001, 26200))
    if forbidden & set(seeds):
        raise ValueError("fresh evaluation seeds overlap training/audit replay seeds")

    training_summary = json.loads((args.actor_dir / "training_summary.json").read_text())
    actor_path = Path(training_summary["selected_actor_checkpoint"])
    checkpoint = torch.load(actor_path, map_location="cpu")
    device = torch.device(args.device)
    input_type = checkpoint.get("input_type", "feedback_only")
    if input_type == "state_feedback":
        actor = TorchMPPIFeedbackDiscreteStateNetwork(
            len(checkpoint["action_radii_sigma"]), dropout=0.0
        ).to(device)
        state_normalization = MPPIProposalNormalization.from_dict(
            checkpoint["state_normalization"]
        )
        feedback_mean = np.asarray(checkpoint["feedback_mean"], np.float32)
        feedback_std = np.asarray(checkpoint["feedback_std"], np.float32)
        gradient_mean_normalization = np.asarray(
            checkpoint["gradient_mean"], np.float32
        )
        gradient_std_normalization = np.asarray(
            checkpoint["gradient_std"], np.float32
        )
    else:
        actor = TorchMPPIFeedbackDiscreteStepActor(
            len(checkpoint["action_radii_sigma"]), dropout=0.0
        ).to(device)
        context_mean = np.asarray(checkpoint["context_mean"], np.float32)
        context_std = np.asarray(checkpoint["context_std"], np.float32)
    actor.load_state_dict(checkpoint["model_state_dict"])
    actor.eval()
    source_indices = np.asarray(checkpoint["action_source_indices"], np.int64)
    action_radii = np.asarray(checkpoint["action_radii_sigma"], np.float32)
    splits = json.loads((args.replay_labels / "splits.json").read_text())
    test_episodes = set(splits["test"])
    config = load_config(args.parent_labels / "teacher_config.json")
    rows: list[dict[str, Any]] = []
    all_costs: list[np.ndarray] = []
    actor_actions: list[np.ndarray] = []

    paths = [
        path for path in sorted(args.replay_labels.glob("episode_*/*.npz"))
        if path.parent.name in test_episodes
    ]
    for index, replay_path in enumerate(paths, 1):
        episode = replay_path.parent.name
        source_path = args.source / episode / "snapshots" / replay_path.name
        parent_path = args.parent_labels / episode / replay_path.name
        teacher_path = args.teacher_labels / episode / replay_path.name
        with np.load(source_path, allow_pickle=False) as source, np.load(
            parent_path, allow_pickle=False
        ) as parent, np.load(replay_path, allow_pickle=False) as replay, np.load(
            teacher_path, allow_pickle=False
        ) as teacher:
            params = json.loads(str(source["mppi_params_json"]))
            sigma = np.asarray(params["noise_sigma"], np.float32)
            action_min = np.asarray(params["action_min"], np.float32)
            action_max = np.asarray(params["action_max"], np.float32)
            feedback = np.asarray(parent["audit_first_pass_feedback"], np.float32)
            gradient_mean = np.asarray(replay["audit_critic_gradient_mean"], np.float32)
            gradient_std = np.asarray(replay["audit_critic_gradient_std"], np.float32)
            context = np.concatenate((feedback, gradient_mean, gradient_std), axis=1)
            with torch.no_grad():
                if input_type == "state_feedback":
                    state = np.asarray(source["initial_state"], np.float32)
                    action = np.asarray(source["current_action"], np.float32)
                    repeat = len(feedback)
                    history = np.repeat(
                        np.asarray(source["history"], np.float32), repeat, axis=0
                    )
                    reference = np.repeat(
                        ego_reference_features(
                            source["reference_ego"], float(state[3])
                        )[None], repeat, axis=0,
                    )
                    current = np.repeat(
                        np.asarray((state[3], state[4], *action), np.float32)[None],
                        repeat, axis=0,
                    )
                    history, reference, current = state_normalization.normalize_numpy(
                        history, reference, current
                    )
                    selected = actor(
                        torch.from_numpy(history).to(device),
                        torch.from_numpy(reference).to(device),
                        torch.from_numpy(current).to(device),
                        torch.from_numpy(
                            np.asarray(replay["audit_guided_center_knots"], np.float32)
                        ).to(device),
                        torch.from_numpy(
                            (feedback - feedback_mean) / feedback_std
                        ).to(device),
                        torch.from_numpy(
                            (context[:, 74:] - gradient_mean_normalization)
                            / gradient_std_normalization
                        ).to(device),
                    ).argmax(1)
                else:
                    normalized = ((context - context_mean) / context_std).astype(np.float32)
                    selected = actor(
                        torch.from_numpy(normalized).to(device)
                    ).argmax(1)
            selected = selected.cpu().numpy()
            actor_actions.append(selected)
            replay_centers = np.asarray(replay["audit_centers"], np.float32)
            repeat = len(replay_centers)
            actor_centers = replay_centers[
                np.arange(repeat), source_indices[selected]
            ]
            teacher_centers = np.repeat(
                np.asarray(teacher["teacher_center_knots"], np.float32)[None],
                repeat, axis=0,
            )
            centers = np.stack(
                (
                    replay_centers[:, 0],
                    replay_centers[:, 5],
                    actor_centers,
                    teacher_centers,
                ),
                axis=1,
            )
            controller, backend = make_controller(source, config, device)
            result = evaluate_center_bank(
                centers=centers,
                seeds=seeds,
                sample_count=args.samples_per_center_seed,
                local_sigma=sigma * args.second_noise_scale,
                action_min=action_min,
                action_max=action_max,
                temperature=float(config["objective"]["temperature"]),
                controller=controller,
                backend=backend,
                history=torch.from_numpy(source["history"]).to(device),
                initial_state=torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5),
                current_action=torch.from_numpy(source["current_action"]).to(device).reshape(1, 2),
                reference=controller._prepare_reference(source["reference"]),
            )
            cost = result["proposal_output_cost_by_seed"]
            all_costs.append(cost)
            for repeat_index in range(repeat):
                rows.append({
                    "episode_id": episode,
                    "snapshot": replay_path.stem,
                    "repeat": repeat_index,
                    "first_pass_seed": int(replay["audit_first_pass_seed"][repeat_index]),
                    "actor_action_index": int(selected[repeat_index]),
                    "actor_radius_sigma": float(action_radii[selected[repeat_index]]),
                    **{
                        f"{name}_cost_mean": float(cost[repeat_index, method].mean())
                        for method, name in enumerate(METHODS)
                    },
                })
        if index % 25 == 0 or index == len(paths):
            print(f"[{index:03d}/{len(paths):03d}] fresh actor/teacher DBM evaluation", flush=True)

    costs = np.asarray(all_costs, np.float32)
    selected_actions = np.concatenate(actor_actions)
    method_metrics = {
        name: cost_metrics(costs[:, :, method, :])
        for method, name in enumerate(METHODS)
    }
    comparisons = {
        "actor_vs_guided": comparison(costs[:, :, 0], costs[:, :, 2]),
        "actor_vs_fixed_pos_0p10": comparison(costs[:, :, 1], costs[:, :, 2]),
        "actor_vs_t1_teacher": comparison(costs[:, :, 3], costs[:, :, 2]),
        "t1_teacher_vs_guided": comparison(costs[:, :, 0], costs[:, :, 3]),
    }
    summary = {
        "format_version": 1,
        "actor_checkpoint": str(actor_path.resolve()),
        "actor_input_type": input_type,
        "source": str(args.source.resolve()),
        "parent_labels": str(args.parent_labels.resolve()),
        "replay_labels": str(args.replay_labels.resolve()),
        "teacher_labels": str(args.teacher_labels.resolve()),
        "test_snapshot_count": len(paths),
        "context_count": int(costs.shape[0] * costs.shape[1]),
        "evaluation_seeds": seeds,
        "samples_per_center_seed": args.samples_per_center_seed,
        "second_noise_scale": args.second_noise_scale,
        "common_random_numbers": True,
        "method_cost": method_metrics,
        "comparisons": comparisons,
        "actor_action_histogram": np.bincount(
            selected_actions, minlength=len(action_radii)
        ).tolist(),
        "actor_mean_radius_sigma": float(action_radii[selected_actions].mean()),
        "interpretation": (
            "All four centers use identical fresh second-pass noise, sample budget, "
            "DBM, objective, states, and first-pass audit contexts. Positive comparison "
            "cost reduction means the method named after 'vs' has lower DBM cost."
        ),
    }
    with (args.output_dir / "per_context.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output_dir / "fresh_eval.npz",
        method_names=np.asarray(METHODS),
        evaluation_seeds=np.asarray(seeds, np.int64),
        costs=costs,
        actor_action_indices=selected_actions,
        actor_action_radii=action_radii[selected_actions],
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({
        "status": "ok", "output": str(args.output_dir),
        "method_cost": method_metrics,
        "actor_vs_t1_teacher": comparisons["actor_vs_t1_teacher"],
    }, indent=2))


if __name__ == "__main__":
    main()
