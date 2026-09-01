#!/usr/bin/env python3
"""Fresh-seed DBM comparison of the multidirection Actor and T1 teacher."""

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
    TorchMPPISequentialProbeActorCritic,
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
DEFAULT_RISK = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_risk_replay_diverse_20260805_v1"
)
DEFAULT_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_multidirection_replay_diverse_20260805_v1"
)
DEFAULT_TEACHER = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_teacher_t1_diverse_20260805_v1"
)
DEFAULT_ACTOR = Path(
    "outputs/mppi_proposal/multidirection_actor_critic_20260805_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/multidirection_actor_teacher_eval_20260805_v1"
)
DEFAULT_SEQUENTIAL = Path(
    "outputs/mppi_proposal/sequential_probe_sac_20260805_v1/"
    "sequential_probe_sac.pt"
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--risk-labels", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--teacher-labels", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--actor-dir", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument(
        "--policy", choices=("actor", "probe", "sequential"), default="actor"
    )
    parser.add_argument(
        "--sequential-checkpoint", type=Path, default=DEFAULT_SEQUENTIAL
    )
    parser.add_argument("--probe-seed", type=int, default=28401)
    parser.add_argument("--probe-budget", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--evaluation-seeds",
        default="28201,28202,28203,28204,28205,28206,28207,28208",
    )
    parser.add_argument("--samples-per-center-seed", type=int, default=64)
    parser.add_argument("--second-noise-scale", type=float, default=0.10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def cost_metrics(cost: np.ndarray) -> dict[str, float]:
    context = cost.mean(axis=2).reshape(-1)
    return {
        "mean": float(cost.mean()),
        "median_context": float(np.median(context)),
        "p90_context": float(np.quantile(context, 0.90)),
        "p95_context": float(np.quantile(context, 0.95)),
        "maximum_context": float(context.max()),
    }


def comparison(baseline: np.ndarray, method: np.ndarray) -> dict[str, Any]:
    seed_gain = baseline - method
    context_gain = seed_gain.mean(axis=2).reshape(-1)
    return {
        "mean_cost_reduction": float(seed_gain.mean()),
        "median_context_cost_reduction": float(np.median(context_gain)),
        "p05_context_cost_reduction": float(np.quantile(context_gain, 0.05)),
        "p10_context_cost_reduction": float(np.quantile(context_gain, 0.10)),
        "context_wins": int(np.sum(context_gain > 0.0)),
        "context_losses": int(np.sum(context_gain < 0.0)),
        "seed_win_fraction": float(np.mean(seed_gain > 0.0)),
        "worst_context_cost_reduction": float(context_gain.min()),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    seeds = [int(value) for value in args.evaluation_seeds.split(",")]
    if len(seeds) != len(set(seeds)):
        raise ValueError("evaluation seeds must be distinct")
    forbidden = set(range(24001, 28200))
    if forbidden & set(seeds):
        raise ValueError("fresh evaluation seeds overlap label/training seeds")
    if args.policy == "sequential":
        if not 2 <= args.probe_budget <= 33:
            raise ValueError("sequential probe budget must be between 2 and 33")
        if args.probe_seed in forbidden or args.probe_seed in seeds:
            raise ValueError("sequential probe seed must be fresh and disjoint")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    policy_method = {
        "actor": "multidirection_actor",
        "probe": "full_bank_probe",
        "sequential": "sequential_probe",
    }[args.policy]
    methods = (
        (
            "guided", "critic_pos_0p10", "fixed_priority_probe",
            policy_method, "t1_teacher",
        )
        if args.policy == "sequential"
        else ("guided", "critic_pos_0p10", policy_method, "t1_teacher")
    )
    label_summary = json.loads((args.labels / "summary.json").read_text())
    action_names = list(label_summary["action_names"])
    actor_path = None
    if args.policy == "actor":
        training_summary = json.loads(
            (args.actor_dir / "training_summary.json").read_text()
        )
        actor_path = Path(training_summary["selected_actor_checkpoint"])
        checkpoint = torch.load(actor_path, map_location="cpu")
        actor = TorchMPPIFeedbackDiscreteStateNetwork(
            len(action_names), dropout=0.0
        ).to(device)
        actor.load_state_dict(checkpoint["model_state_dict"])
        actor.eval()
        state_norm = MPPIProposalNormalization.from_dict(
            checkpoint["state_normalization"]
        )
        feedback_mean = np.asarray(checkpoint["feedback_mean"], np.float32)
        feedback_std = np.asarray(checkpoint["feedback_std"], np.float32)
        gradient_mean_norm = np.asarray(checkpoint["gradient_mean"], np.float32)
        gradient_std_norm = np.asarray(checkpoint["gradient_std"], np.float32)
    elif args.policy == "sequential":
        actor_path = args.sequential_checkpoint.resolve()
        checkpoint = torch.load(actor_path, map_location="cpu")
        actor = TorchMPPISequentialProbeActorCritic(
            len(action_names), dropout=0.0
        ).to(device)
        actor.load_state_dict(checkpoint["model_state_dict"])
        actor.eval()
        state_norm = MPPIProposalNormalization.from_dict(
            checkpoint["state_normalization"]
        )
        feedback_mean = np.asarray(checkpoint["feedback_mean"], np.float32)
        feedback_std = np.asarray(checkpoint["feedback_std"], np.float32)
        gradient_mean_norm = np.asarray(checkpoint["gradient_mean"], np.float32)
        gradient_std_norm = np.asarray(checkpoint["gradient_std"], np.float32)
        reward_scale = float(checkpoint["reward_scale"])
    splits = json.loads((args.labels / "splits.json").read_text())
    test_episodes = set(splits["test"])
    config = load_config(args.parent_labels / "teacher_config.json")
    paths = [
        path for path in sorted(args.labels.glob("episode_*/*.npz"))
        if path.parent.name in test_episodes
    ]
    costs, selected_actions, probe_sequences, rows = [], [], [], []
    for index, label_path in enumerate(paths, 1):
        episode = label_path.parent.name
        source_path = args.source / episode / "snapshots" / label_path.name
        parent_path = args.parent_labels / episode / label_path.name
        risk_path = args.risk_labels / episode / label_path.name
        teacher_path = args.teacher_labels / episode / label_path.name
        with np.load(source_path, allow_pickle=False) as source, np.load(
            parent_path, allow_pickle=False
        ) as parent, np.load(risk_path, allow_pickle=False) as risk, np.load(
            label_path, allow_pickle=False
        ) as label, np.load(teacher_path, allow_pickle=False) as teacher:
            anchors = np.asarray(label["audit_guided_center_knots"], np.float32)
            repeat = len(anchors)
            one_probe_sequence = np.empty((repeat, 0), np.int64)
            fixed_priority_centers = None
            params = json.loads(str(source["mppi_params_json"]))
            sigma = np.asarray(params["noise_sigma"], np.float32)
            controller, backend = make_controller(source, config, device)
            if args.policy == "probe":
                # The first stored audit reward seed is an explicit online probe;
                # every fresh evaluation seed below remains unseen.
                selected = np.asarray(
                    label["audit_proposal_output_cost_by_seed"], np.float32
                )[:, :, 0].argmin(axis=1)
            elif args.policy == "actor":
                feedback = np.asarray(
                    parent["audit_first_pass_feedback"], np.float32
                )
                gradient_context = np.concatenate((
                    np.asarray(risk["audit_critic_gradient_mean"], np.float32),
                    np.asarray(risk["audit_critic_gradient_std"], np.float32),
                ), axis=1)
                state = np.asarray(source["initial_state"], np.float32)
                current_action = np.asarray(source["current_action"], np.float32)
                history = np.repeat(
                    np.asarray(source["history"], np.float32), repeat, axis=0
                )
                reference = np.repeat(
                    ego_reference_features(
                        source["reference_ego"], float(state[3])
                    )[None], repeat, axis=0,
                )
                current = np.repeat(
                    np.asarray(
                        (state[3], state[4], *current_action), np.float32
                    )[None], repeat, axis=0,
                )
                history, reference, current = state_norm.normalize_numpy(
                    history, reference, current
                )
                with torch.no_grad():
                    selected = actor(
                        torch.from_numpy(history).to(device),
                        torch.from_numpy(reference).to(device),
                        torch.from_numpy(current).to(device),
                        torch.from_numpy(anchors).to(device),
                        torch.from_numpy(
                            (feedback - feedback_mean) / feedback_std
                        ).to(device),
                        torch.from_numpy(
                            (gradient_context - gradient_mean_norm)
                            / gradient_std_norm
                        ).to(device),
                    ).argmax(1).cpu().numpy()
            else:
                feedback = np.asarray(
                    parent["audit_first_pass_feedback"], np.float32
                )
                gradient_context = np.concatenate((
                    np.asarray(risk["audit_critic_gradient_mean"], np.float32),
                    np.asarray(risk["audit_critic_gradient_std"], np.float32),
                ), axis=1)
                state = np.asarray(source["initial_state"], np.float32)
                current_action = np.asarray(source["current_action"], np.float32)
                history = np.repeat(
                    np.asarray(source["history"], np.float32), repeat, axis=0
                )
                reference = np.repeat(
                    ego_reference_features(
                        source["reference_ego"], float(state[3])
                    )[None], repeat, axis=0,
                )
                current = np.repeat(
                    np.asarray((state[3], state[4], *current_action), np.float32)[None],
                    repeat, axis=0,
                )
                history, reference, current = state_norm.normalize_numpy(
                    history, reference, current
                )
                normalized_feedback = (feedback - feedback_mean) / feedback_std
                normalized_gradient = (
                    gradient_context - gradient_mean_norm
                ) / gradient_std_norm
                bank = np.asarray(label["audit_centers"], np.float32)
                anchor_probe = evaluate_center_bank(
                    centers=anchors[:, None], seeds=[args.probe_seed],
                    sample_count=args.samples_per_center_seed,
                    local_sigma=sigma * args.second_noise_scale,
                    action_min=np.asarray(params["action_min"], np.float32),
                    action_max=np.asarray(params["action_max"], np.float32),
                    temperature=float(config["objective"]["temperature"]),
                    controller=controller, backend=backend,
                    history=torch.from_numpy(source["history"]).to(device),
                    initial_state=torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5),
                    current_action=torch.from_numpy(source["current_action"]).to(device).reshape(1, 2),
                    reference=controller._prepare_reference(source["reference"]),
                )["proposal_output_cost_by_seed"][:, 0, 0]
                probe_value = np.zeros((repeat, len(action_names)), np.float32)
                probe_mask = np.zeros_like(probe_value, dtype=np.bool_)
                probe_mask[:, 0] = True
                one_probe_sequence = np.empty(
                    (repeat, args.probe_budget - 1), np.int64
                )
                denominator = max(args.probe_budget - 1, 1)
                for probe_step in range(1, args.probe_budget):
                    remaining = np.full(
                        (repeat, 1),
                        (args.probe_budget - probe_step) / denominator,
                        np.float32,
                    )
                    with torch.no_grad():
                        logits, _, _ = actor(
                            torch.from_numpy(history).to(device),
                            torch.from_numpy(reference).to(device),
                            torch.from_numpy(current).to(device),
                            torch.from_numpy(anchors).to(device),
                            torch.from_numpy(normalized_feedback).to(device),
                            torch.from_numpy(normalized_gradient).to(device),
                            torch.from_numpy(probe_value).to(device),
                            torch.from_numpy(probe_mask).to(device),
                            torch.from_numpy(remaining).to(device),
                        )
                        action = logits.masked_fill(
                            torch.from_numpy(probe_mask).to(device), -1e9
                        ).argmax(1).cpu().numpy()
                    one_probe_sequence[:, probe_step - 1] = action
                    selected_centers = bank[np.arange(repeat), action]
                    action_probe = evaluate_center_bank(
                        centers=selected_centers[:, None], seeds=[args.probe_seed],
                        sample_count=args.samples_per_center_seed,
                        local_sigma=sigma * args.second_noise_scale,
                        action_min=np.asarray(params["action_min"], np.float32),
                        action_max=np.asarray(params["action_max"], np.float32),
                        temperature=float(config["objective"]["temperature"]),
                        controller=controller, backend=backend,
                        history=torch.from_numpy(source["history"]).to(device),
                        initial_state=torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5),
                        current_action=torch.from_numpy(source["current_action"]).to(device).reshape(1, 2),
                        reference=controller._prepare_reference(source["reference"]),
                    )["proposal_output_cost_by_seed"][:, 0, 0]
                    probe_value[np.arange(repeat), action] = np.clip(
                        (anchor_probe - action_probe) / reward_scale, -10.0, 10.0
                    )
                    probe_mask[np.arange(repeat), action] = True
                selected = np.where(
                    probe_mask, probe_value, -np.inf
                ).argmax(1)
                fixed_names = (
                    "critic_plus_preconditioned_pos_0p150",
                    "critic_pos_0p150",
                    "negative_gradient_pos_0p150",
                )
                fixed_indices = np.asarray(
                    [action_names.index(name) for name in fixed_names], np.int64
                )
                fixed_probe = evaluate_center_bank(
                    centers=bank[:, fixed_indices], seeds=[args.probe_seed],
                    sample_count=args.samples_per_center_seed,
                    local_sigma=sigma * args.second_noise_scale,
                    action_min=np.asarray(params["action_min"], np.float32),
                    action_max=np.asarray(params["action_max"], np.float32),
                    temperature=float(config["objective"]["temperature"]),
                    controller=controller, backend=backend,
                    history=torch.from_numpy(source["history"]).to(device),
                    initial_state=torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5),
                    current_action=torch.from_numpy(source["current_action"]).to(device).reshape(1, 2),
                    reference=controller._prepare_reference(source["reference"]),
                )["proposal_output_cost_by_seed"][:, :, 0]
                fixed_cost = np.concatenate((anchor_probe[:, None], fixed_probe), 1)
                fixed_bank_indices = np.concatenate((np.asarray([0]), fixed_indices))
                fixed_selected = fixed_bank_indices[fixed_cost.argmin(1)]
                fixed_priority_centers = bank[np.arange(repeat), fixed_selected]
            selected_actions.append(selected)
            probe_sequences.append(one_probe_sequence)
            bank = np.asarray(label["audit_centers"], np.float32)
            actor_centers = bank[np.arange(repeat), selected]
            teacher_centers = np.repeat(
                np.asarray(teacher["teacher_center_knots"], np.float32)[None],
                repeat, axis=0,
            )
            center_list = [anchors, bank[:, 3]]
            if fixed_priority_centers is not None:
                center_list.append(fixed_priority_centers)
            center_list.extend((actor_centers, teacher_centers))
            centers = np.stack(center_list, axis=1)
            result = evaluate_center_bank(
                centers=centers,
                seeds=seeds,
                sample_count=args.samples_per_center_seed,
                local_sigma=sigma * args.second_noise_scale,
                action_min=np.asarray(params["action_min"], np.float32),
                action_max=np.asarray(params["action_max"], np.float32),
                temperature=float(config["objective"]["temperature"]),
                controller=controller,
                backend=backend,
                history=torch.from_numpy(source["history"]).to(device),
                initial_state=torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5),
                current_action=torch.from_numpy(source["current_action"]).to(device).reshape(1, 2),
                reference=controller._prepare_reference(source["reference"]),
            )
            one_cost = result["proposal_output_cost_by_seed"]
            costs.append(one_cost)
            for repeat_index in range(repeat):
                rows.append({
                    "episode_id": episode,
                    "snapshot": label_path.stem,
                    "repeat": repeat_index,
                    "actor_action_index": int(selected[repeat_index]),
                    "actor_action_name": action_names[selected[repeat_index]],
                    "probe_action_sequence": ";".join(
                        action_names[action]
                        for action in one_probe_sequence[repeat_index]
                    ),
                    **{
                        f"{name}_cost_mean": float(one_cost[repeat_index, method].mean())
                        for method, name in enumerate(methods)
                    },
                })
        if index % 25 == 0 or index == len(paths):
            print(f"[{index:03d}/{len(paths):03d}] fresh multidirection evaluation", flush=True)
    cost = np.asarray(costs, np.float32)
    selected = np.concatenate(selected_actions)
    probe_sequence = np.concatenate(probe_sequences)
    policy_index = methods.index(policy_method)
    guided_index = methods.index("guided")
    critic_index = methods.index("critic_pos_0p10")
    teacher_index = methods.index("t1_teacher")
    summary = {
        "format_version": 1,
        "policy": args.policy,
        "actor_checkpoint": str(actor_path.resolve()) if actor_path else None,
        "probe_seed": (
            int(label_summary["audit_seeds"][0]) if args.policy == "probe"
            else args.probe_seed if args.policy == "sequential" else None
        ),
        "probe_candidate_rollouts_per_context": (
            len(action_names) * int(label_summary["samples_per_center_seed"])
            if args.policy == "probe"
            else args.probe_budget * args.samples_per_center_seed
            if args.policy == "sequential" else 0
        ),
        "test_snapshot_count": len(paths),
        "context_count": int(cost.shape[0] * cost.shape[1]),
        "evaluation_seeds": seeds,
        "samples_per_center_seed": args.samples_per_center_seed,
        "second_noise_scale": args.second_noise_scale,
        "common_random_numbers": True,
        "method_cost": {
            name: cost_metrics(cost[:, :, method, :])
            for method, name in enumerate(methods)
        },
        "comparisons": {
            f"{policy_method}_vs_guided": comparison(
                cost[:, :, guided_index], cost[:, :, policy_index]
            ),
            f"{policy_method}_vs_critic_pos_0p10": comparison(
                cost[:, :, critic_index], cost[:, :, policy_index]
            ),
            f"{policy_method}_vs_t1_teacher": comparison(
                cost[:, :, teacher_index], cost[:, :, policy_index]
            ),
            **({
                "sequential_probe_vs_fixed_priority_probe": comparison(
                    cost[:, :, methods.index("fixed_priority_probe")],
                    cost[:, :, policy_index],
                )
            } if "fixed_priority_probe" in methods else {}),
        },
        "actor_action_histogram": np.bincount(
            selected, minlength=len(action_names)
        ).tolist(),
        "top_actor_actions": [
            {"name": action_names[action], "count": int(count)}
            for action, count in sorted(
                enumerate(np.bincount(selected, minlength=len(action_names))),
                key=lambda value: value[1], reverse=True,
            )[:10]
        ],
        "probe_action_histogram_by_round": [
            np.bincount(
                probe_sequence[:, round_index], minlength=len(action_names)
            ).tolist()
            for round_index in range(probe_sequence.shape[1])
        ],
    }
    with (args.output_dir / "per_context.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(
        args.output_dir / "fresh_eval.npz",
        method_names=np.asarray(methods), evaluation_seeds=np.asarray(seeds),
        costs=cost, actor_action_indices=selected,
        probe_action_sequences=probe_sequence,
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({
        "status": "ok", "output": str(args.output_dir),
        "method_cost": summary["method_cost"],
        f"{policy_method}_vs_t1_teacher": summary["comparisons"][
            f"{policy_method}_vs_t1_teacher"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
