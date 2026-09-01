#!/usr/bin/env python3
"""Independently validate train-only online Alpha-SAC replay and checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import (
    TorchMPPITrustAlphaCritic,
    TorchMPPITrustAlphaSACPolicy,
)
from train_mppi_direct_alpha_actor_critic import critic_metrics
from train_mppi_direct_alpha_online_sac import (
    calibrate_and_evaluate,
    critic_local_probe_metrics,
    deterministic_outputs,
    local_probe_triplet,
    metrics_from_cache,
)
from train_mppi_direct_trust_region_actor import (
    direct_cost,
    load_actor_payload,
    load_dataset,
    tensorize,
)
from train_mppi_direct_trust_alpha_policy import extra_tensors


DEFAULT_RUN = Path(
    "outputs/mppi_proposal/direct_alpha_online_sac_trainonly_20260810_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=DEFAULT_RUN)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def maximum_metric_error(saved: Any, replayed: Any) -> float:
    errors: list[float] = []

    def visit(left: Any, right: Any) -> None:
        if isinstance(left, dict):
            for key, value in left.items():
                if key in right and key not in {
                    "iteration", "replay_size", "critic_loss", "bc_weight",
                    "new_batch_before_update", "new_batch_after_update",
                    "actor_loss", "actor_expected_q", "gate_entropy",
                }:
                    visit(value, right[key])
        elif isinstance(left, bool):
            if bool(left) != bool(right):
                errors.append(1.0)
        elif isinstance(left, (int, float)):
            errors.append(abs(float(left) - float(right)))

    visit(saved, replayed)
    return max(errors, default=0.0)


def main() -> None:
    args = parse_args()
    summary = json.loads((args.run / "training_summary.json").read_text())
    checkpoint = torch.load(args.run / "online_alpha_sac_selected.pt", map_location="cpu")
    training = argparse.Namespace(**checkpoint["training_arguments"])
    # Replay rewards are intentionally recomputed with the validator batch size.
    # Policy/Critic metrics use the frozen training contract's evaluation batch
    # size so DBM floating-point reduction order does not create a false metric
    # mismatch for larger shifted-window probe banks.
    training.evaluation_batch_size = int(training.evaluation_batch_size)
    device = torch.device(args.device)
    old_payload = load_actor_payload(Path(checkpoint["old_actor"]))
    data, _, splits = load_dataset(Path(checkpoint["labels"]), old_payload)
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    fit_episodes = set(checkpoint["fit_episodes"])
    selection_episodes = set(checkpoint["selection_episodes"])
    if fit_episodes & selection_episodes:
        raise AssertionError("checkpoint fit/selection episodes overlap")
    if fit_episodes | selection_episodes != set(splits["train"]):
        raise AssertionError("checkpoint internal split differs from TR1 train split")

    with np.load(args.run / "actor_visited_replay.npz", allow_pickle=False) as replay:
        context = np.asarray(replay["context_index"], np.int64)
        alpha = np.asarray(replay["alpha"], np.float32)
        reward = np.asarray(replay["reward"], np.float32)
        replay_episode = np.asarray(replay["episode"])
    if len(context) != int(checkpoint["online_replay_size"]):
        raise AssertionError("online replay size differs from checkpoint")
    if not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(reward)):
        raise AssertionError("online replay contains non-finite values")
    if np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise AssertionError("online replay alpha is outside [0,1]")
    if not set(replay_episode.tolist()) <= fit_episodes:
        raise AssertionError("online replay contains non-fit episodes")
    if set(replay_episode.tolist()) & selection_episodes:
        raise AssertionError("internal-selection episode leaked into replay")
    raw = (
        data.old_center[context]
        + alpha[:, None, None]
        * data.sigma[context, None, :]
        * data.projected_direction[context]
    )
    centers = np.clip(raw, -1.0, 1.0).astype(np.float32)
    replay_cost = direct_cost(
        centers, data, tensors, context, args.batch_size, device
    )
    replay_reward = data.old_cost[context] - replay_cost
    reward_error = float(np.max(np.abs(reward - replay_reward)))

    local_reward_error = 0.0
    local_semantic_error = 0.0
    local_replay_count = 0
    local_replay_path = args.run / "actor_local_probe_replay.npz"
    if local_replay_path.is_file():
        with np.load(local_replay_path, allow_pickle=False) as local:
            local_context = np.asarray(local["context_index"], np.int64)
            local_alpha = np.asarray(local["alpha"], np.float32)
            local_reward = np.asarray(local["reward"], np.float32)
            local_radius = np.asarray(local["radius"], np.float32)
            local_anchor = np.asarray(
                local["anchor_alpha"]
                if "anchor_alpha" in local.files else local_alpha[:, 1],
                np.float32,
            )
            local_episode = np.asarray(local["episode"])
        local_replay_count = len(local_context)
        if local_alpha.shape != (local_replay_count, 3):
            raise AssertionError("local-probe alpha shape is invalid")
        if local_reward.shape != (local_replay_count, 3):
            raise AssertionError("local-probe reward shape is invalid")
        if not set(local_episode.tolist()) <= fit_episodes:
            raise AssertionError("local-probe replay contains non-fit episodes")
        if np.any(local_alpha < 0.0) or np.any(local_alpha > 1.0):
            raise AssertionError("local-probe alpha is outside [0,1]")
        if np.any(np.diff(local_alpha, axis=1) < 0.0):
            raise AssertionError("local-probe alpha triplet is not ordered")
        expected_alpha = local_probe_triplet(
            local_anchor, local_radius,
            getattr(training, "local_probe_boundary_mode", "clip"),
        )
        local_semantic_error = float(np.max(np.abs(
            local_alpha - expected_alpha
        )))
        flat_context = np.repeat(local_context, 3)
        flat_alpha = local_alpha.reshape(-1)
        local_raw = (
            data.old_center[flat_context]
            + flat_alpha[:, None, None]
            * data.sigma[flat_context, None, :]
            * data.projected_direction[flat_context]
        )
        local_centers = np.clip(local_raw, -1.0, 1.0).astype(np.float32)
        local_cost = direct_cost(
            local_centers, data, tensors, flat_context, args.batch_size, device
        ).reshape(-1, 3)
        replayed_local_reward = data.old_cost[local_context, None] - local_cost
        local_reward_error = float(np.max(np.abs(
            local_reward - replayed_local_reward
        )))

    policy = TorchMPPITrustAlphaSACPolicy(
        dropout=0.0,
        initial_log_std=float(training.initial_log_std),
        alpha_logit_scale=float(checkpoint.get("alpha_logit_scale", 1.0)),
    ).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    q1 = TorchMPPITrustAlphaCritic(dropout=0.0).to(device)
    q2 = TorchMPPITrustAlphaCritic(dropout=0.0).to(device)
    q1.load_state_dict(checkpoint["critic1_state_dict"], strict=True)
    q2.load_state_dict(checkpoint["critic2_state_dict"], strict=True)
    selection_index = np.flatnonzero(np.isin(data.episodes, list(selection_episodes)))
    probability, conditional, _, policy_centers = deterministic_outputs(
        policy, tensors, extra, selection_index,
        0.0, args.batch_size, device,
    )
    move_cost = direct_cost(
        policy_centers, data, tensors, selection_index, args.batch_size, device
    )
    policy_metrics = metrics_from_cache(
        probability, conditional, move_cost, data, selection_index,
        float(checkpoint["move_threshold"]), training,
    )
    threshold_rows = []
    for threshold in np.concatenate((
        np.arange(0.0, 1.0, 0.01), np.asarray((0.995, 0.999, 1.0))
    )):
        one = metrics_from_cache(
            probability, conditional, move_cost, data, selection_index,
            float(threshold), training,
        )
        threshold_rows.append({
            "threshold": float(threshold),
            "direct_cost_mean": one["direct_cost"]["mean"],
            "move_fraction": one["move_fraction"],
            "regression_fraction": one["regression_fraction"],
            "gain_p05": one["gain_vs_old"]["p05"],
            "gain_worst": one["gain_vs_old"]["minimum"],
            "safe_teacher_beaten_fraction": one["safe_teacher_beaten_fraction"],
            "pass": one["pass"],
        })
    mean_best = min(threshold_rows, key=lambda row: row["direct_cost_mean"])
    passing_rows = [row for row in threshold_rows if row["pass"]]
    passing_best = min(
        passing_rows, key=lambda row: row["direct_cost_mean"]
    ) if passing_rows else None
    policy_error = maximum_metric_error(
        summary["selected_metrics"], policy_metrics
    )
    posthoc_safe_error = 0.0
    saved_posthoc_safe = summary.get("posthoc_safe_selected_actor_metrics")
    if saved_posthoc_safe is not None:
        safe_training = copy.copy(training)
        safe_training.selection_mode = "safe_gate"
        replayed_posthoc_safe = calibrate_and_evaluate(
            policy, data, tensors, extra, selection_index,
            safe_training, device,
        )
        if replayed_posthoc_safe is None:
            posthoc_safe_error = float("inf")
        else:
            posthoc_safe_error = maximum_metric_error(
                saved_posthoc_safe, replayed_posthoc_safe
            )
    critic_result = critic_metrics(
        q1, q2, data, tensors, extra, selection_index,
        training, device,
    )
    critic_error = maximum_metric_error(
        summary["critic_internal_selection"], critic_result
    )
    local_critic_error = 0.0
    local_critic_result = critic_local_probe_metrics(
        policy, q1, q2, data, tensors, extra, selection_index,
        float(checkpoint["move_threshold"]), training, device,
    )
    if summary.get("local_critic_internal_selection") is not None:
        local_critic_error = maximum_metric_error(
            summary["local_critic_internal_selection"], local_critic_result
        )
    training_state_policy_error = 0.0
    training_state_critic_error = 0.0
    training_state_local_critic_error = 0.0
    training_state_path = args.run / "online_alpha_sac_training_state.pt"
    if training_state_path.is_file():
        training_state = torch.load(training_state_path, map_location="cpu")
        if training_state.get("checkpoint_role") != "latest_training_state":
            raise AssertionError("online training-state checkpoint role is invalid")
        latest_policy = TorchMPPITrustAlphaSACPolicy(
            dropout=0.0,
            initial_log_std=float(training.initial_log_std),
            alpha_logit_scale=float(
                training_state.get("alpha_logit_scale", 1.0)
            ),
        ).to(device)
        latest_policy.load_state_dict(
            training_state["policy_state_dict"], strict=True
        )
        latest_probability, latest_conditional, _, latest_centers = (
            deterministic_outputs(
                latest_policy, tensors, extra, selection_index,
                0.0, args.batch_size, device,
            )
        )
        latest_move_cost = direct_cost(
            latest_centers, data, tensors, selection_index,
            args.batch_size, device,
        )
        latest_policy_metrics = metrics_from_cache(
            latest_probability, latest_conditional, latest_move_cost,
            data, selection_index,
            float(training_state["move_threshold"]), training,
        )
        training_state_policy_error = maximum_metric_error(
            summary["latest_training_metrics"], latest_policy_metrics
        )
        latest_q1 = TorchMPPITrustAlphaCritic(dropout=0.0).to(device)
        latest_q2 = TorchMPPITrustAlphaCritic(dropout=0.0).to(device)
        latest_q1.load_state_dict(
            training_state["critic1_state_dict"], strict=True
        )
        latest_q2.load_state_dict(
            training_state["critic2_state_dict"], strict=True
        )
        latest_critic_result = critic_metrics(
            latest_q1, latest_q2, data, tensors, extra,
            selection_index, training, device,
        )
        training_state_critic_error = maximum_metric_error(
            summary["latest_critic_internal_selection"],
            latest_critic_result,
        )
        latest_local_critic_result = critic_local_probe_metrics(
            latest_policy, latest_q1, latest_q2, data, tensors, extra,
            selection_index, float(training_state["move_threshold"]),
            training, device,
        )
        if summary.get("latest_local_critic_internal_selection") is not None:
            training_state_local_critic_error = maximum_metric_error(
                summary["latest_local_critic_internal_selection"],
                latest_local_critic_result,
            )
    result = {
        "format_version": 1,
        "run": str(args.run.resolve()),
        "replay_transition_count": len(context),
        "replay_fit_episode_count": len(set(replay_episode.tolist())),
        "replay_reward_max_abs_error": reward_error,
        "local_probe_replay_group_count": local_replay_count,
        "local_probe_reward_max_abs_error": local_reward_error,
        "local_probe_semantic_max_abs_error": local_semantic_error,
        "policy_metric_max_abs_error": policy_error,
        "posthoc_safe_metric_max_abs_error": posthoc_safe_error,
        "critic_metric_max_abs_error": critic_error,
        "local_critic_metric_max_abs_error": local_critic_error,
        "training_state_policy_metric_max_abs_error": training_state_policy_error,
        "training_state_critic_metric_max_abs_error": training_state_critic_error,
        "training_state_local_critic_metric_max_abs_error": (
            training_state_local_critic_error
        ),
        "threshold_diagnostic": {
            "mean_cost_best": mean_best,
            "safety_passing_best": passing_best,
        },
        "selection_replay_overlap": sorted(
            set(replay_episode.tolist()) & selection_episodes
        ),
        "formal_validation_or_test_loaded": False,
        "qualification": (
            "ONLINE_ALPHA_SAC_INDEPENDENTLY_VALIDATED"
            if max(
                reward_error, policy_error, posthoc_safe_error, critic_error,
                local_reward_error, local_semantic_error, local_critic_error,
                training_state_policy_error, training_state_critic_error,
                training_state_local_critic_error,
            ) <= 1e-5
            else "ONLINE_ALPHA_SAC_VALIDATION_FAILED"
        ),
    }
    (args.run / "independent_validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)
    if result["qualification"] != "ONLINE_ALPHA_SAC_INDEPENDENTLY_VALIDATED":
        raise AssertionError("online Alpha-SAC independent validation failed")


if __name__ == "__main__":
    main()
