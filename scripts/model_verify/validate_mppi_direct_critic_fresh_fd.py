#!/usr/bin/env python3
"""Independently replay the fresh finite-difference Critic audit artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import (
    TorchMPPIActorCenteredLocalCritic,
    TorchMPPIDeterministicCenterActor,
)
from generate_dbm_proposal_teacher import sha256_file
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_residual_online_ac import (
    actor_inputs,
    make_base_policy,
    residual_outputs,
)
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    direct_cost,
    load_actor_payload,
    load_dataset,
    tensorize,
)
import analyze_mppi_direct_critic_fresh_fd as audit
import train_mppi_direct_local_gradient_critic as local_train


DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_critic_fresh_fd_20260813_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--replay-count", type=int, default=1024)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def numeric_error(reference: Any, actual: Any) -> float:
    errors: list[float] = []
    if isinstance(reference, dict) and isinstance(actual, dict):
        for key, value in reference.items():
            if key in actual:
                errors.append(numeric_error(value, actual[key]))
    elif isinstance(reference, (int, float)) and isinstance(actual, (int, float)):
        errors.append(abs(float(reference) - float(actual)))
    return max(errors, default=0.0)


def action_coordinate_check(
    actor: TorchMPPIDeterministicCenterActor,
    models: list[TorchMPPIActorCenteredLocalCritic],
    inputs: tuple[torch.Tensor, ...],
    context_index: np.ndarray,
    alpha_center: np.ndarray,
    actor_action: np.ndarray,
    actor_center: np.ndarray,
    arrays: dict[str, np.ndarray],
    maximum_residual_sigma: float,
    device: torch.device,
    count: int = 32,
) -> dict[str, Any]:
    """Verify normalized/physical/pre-tanh Jacobians used by the audit."""
    index = torch.from_numpy(context_index).to(device)
    with torch.no_grad():
        feature = actor.encoder(*(one[index] for one in inputs))
        pre_tanh = actor.action_head(feature).reshape(-1, 8, 2)
        requested_action = torch.tanh(pre_tanh)
        effective_action, reconstructed_center = actor.center_from_action(
            torch.from_numpy(alpha_center[context_index]).to(device),
            requested_action,
        )
    pre_tanh_np = pre_tanh.cpu().numpy()
    requested_np = requested_action.cpu().numpy()
    effective_np = effective_action.cpu().numpy()
    center_np = reconstructed_center.cpu().numpy()
    actor_base_sigma = actor.base_sigma.detach().cpu().numpy().reshape(2)
    stored_sigma = arrays["sigma"]
    sigma_error = float(np.max(np.abs(stored_sigma - actor_base_sigma)))
    maximum_sigma_error = abs(
        float(actor.maximum_delta_sigma.detach().cpu()) - maximum_residual_sigma
    )

    # The fresh FD label is fit in normalized-action coordinates.  Refit the
    # central differences independently in center coordinates and verify the
    # exact diagonal coordinate transform g_center = g_norm / (M * sigma).
    normalized = arrays["actions"][:, 0]
    center = arrays["centers"][:, 0]
    reward = arrays["transformed_reward"][:, 0]
    normalized_design = (
        normalized[:, 1:17] - normalized[:, 17:33]
    ).reshape(len(context_index), 16, 16)
    center_design = (
        center[:, 1:17] - center[:, 17:33]
    ).reshape(len(context_index), 16, 16)
    difference = reward[:, 1:17] - reward[:, 17:33]
    normalized_gradient = np.asarray([
        np.linalg.solve(design, target)
        for design, target in zip(normalized_design, difference)
    ], np.float32)
    center_gradient = np.asarray([
        np.linalg.solve(design, target)
        for design, target in zip(center_design, difference)
    ], np.float32)
    diagonal_scale = np.broadcast_to(
        maximum_residual_sigma * stored_sigma[:, None, :],
        (len(context_index), 8, 2),
    ).reshape(len(context_index), 16)
    expected_center_gradient = normalized_gradient / diagonal_scale
    fd_coordinate_max_abs_error = float(np.max(np.abs(
        center_gradient - expected_center_gradient
    )))
    fd_coordinate_vector_relative_error = np.linalg.norm(
        center_gradient - expected_center_gradient, axis=1
    ) / np.maximum(np.linalg.norm(expected_center_gradient, axis=1), 1e-8)

    chosen = context_index[: min(count, len(context_index))]
    chosen_index = torch.from_numpy(chosen).to(device)
    chosen_count = len(chosen)
    anchor = torch.from_numpy(actor_action[:chosen_count]).to(device)
    center_anchor = torch.from_numpy(actor_center[:chosen_count]).to(device)
    alpha_anchor = torch.from_numpy(alpha_center[chosen]).to(device)
    channel_scale = (
        actor.maximum_delta_sigma * actor.base_sigma
    ).detach()
    pre_tanh_errors, center_errors = [], []
    for model in models:
        _, analytic_gradient, _ = model.local_parameters(
            *(one[chosen_index] for one in inputs), anchor
        )

        raw_query = torch.from_numpy(pre_tanh_np[:chosen_count]).to(device)
        raw_query.requires_grad_(True)
        requested_query = torch.tanh(raw_query)
        normalized_query, _ = actor.center_from_action(
            alpha_anchor, requested_query
        )
        raw_prediction = model(
            *(one[chosen_index] for one in inputs), anchor, normalized_query
        )
        raw_autograd = torch.autograd.grad(raw_prediction.sum(), raw_query)[0]
        requested_center = alpha_anchor + requested_query * channel_scale
        unclipped = (
            (requested_center >= -1.0) & (requested_center <= 1.0)
        ).to(requested_query.dtype)
        expected_raw = (
            analytic_gradient * (1.0 - requested_query.square()) * unclipped
        )
        pre_tanh_errors.append(float(torch.max(torch.abs(
            raw_autograd - expected_raw
        )).detach().cpu()))

        center_query = center_anchor.detach().clone().requires_grad_(True)
        normalized_from_center = (center_query - alpha_anchor) / channel_scale
        center_prediction = model(
            *(one[chosen_index] for one in inputs), anchor,
            normalized_from_center,
        )
        center_autograd = torch.autograd.grad(
            center_prediction.sum(), center_query
        )[0]
        expected_center = analytic_gradient / channel_scale
        center_errors.append(float(torch.max(torch.abs(
            center_autograd - expected_center
        )).detach().cpu()))

    tanh_jacobian = 1.0 - np.square(requested_np)
    actor_clipped_element = np.abs(requested_np - effective_np) > 1e-6
    actor_clipped_context = np.any(actor_clipped_element, axis=(1, 2))
    physical_jacobian = 1.0 / (
        maximum_residual_sigma * actor_base_sigma
    )
    full_channel_scale = np.broadcast_to(
        maximum_residual_sigma * actor_base_sigma.reshape(1, 1, 2),
        actor_action.shape,
    )
    requested_center_np = (
        alpha_center[context_index]
        + requested_np * full_channel_scale
    )
    unclipped_np = (
        (requested_center_np >= -1.0) & (requested_center_np <= 1.0)
    ).astype(np.float32)
    pre_tanh_jacobian = tanh_jacobian * unclipped_np
    ensemble_gradient = np.mean(arrays["critic_gradient"], axis=0).reshape(
        len(context_index), 8, 2
    )
    fresh_gradient = arrays["gradient"].reshape(len(context_index), 8, 2)

    def comparison_summary(
        prediction: np.ndarray, target: np.ndarray
    ) -> dict[str, float]:
        flat_prediction = prediction.reshape(len(context_index), 16)
        flat_target = target.reshape(len(context_index), 16)
        cosine = audit.cosine_rows(flat_prediction, flat_target)
        norm_ratio = np.linalg.norm(flat_prediction, axis=1) / np.maximum(
            np.linalg.norm(flat_target, axis=1), 1e-8
        )
        return {
            "cosine_median": float(np.median(cosine)),
            "cosine_p10": float(np.quantile(cosine, 0.10)),
            "cosine_positive_fraction": float(np.mean(cosine > 0.0)),
            "norm_ratio_median": float(np.median(norm_ratio)),
        }

    coordinate_comparison = {
        "effective_normalized_action": comparison_summary(
            ensemble_gradient, fresh_gradient
        ),
        "physical_center": comparison_summary(
            ensemble_gradient / full_channel_scale,
            fresh_gradient / full_channel_scale,
        ),
        "actor_pre_tanh_with_clamp": comparison_summary(
            ensemble_gradient * pre_tanh_jacobian,
            fresh_gradient * pre_tanh_jacobian,
        ),
    }
    return {
        "coordinate_definitions": {
            "critic_action": "effective normalized residual u in [-1,1]",
            "center": "c = clip(alpha_center + u * M * sigma, -1, 1)",
            "finite_difference_physical_offset": "delta_c = radius_sigma * sigma * direction",
            "finite_difference_fitted_coordinate": "u = (c-alpha_center)/(M*sigma)",
            "pre_tanh": "u_requested = tanh(z)",
            "critic_autograd_tensor": "normalized query action u, not center c or pre-tanh z",
        },
        "maximum_residual_sigma": maximum_residual_sigma,
        "actor_buffer_maximum_delta_sigma": float(
            actor.maximum_delta_sigma.detach().cpu()
        ),
        "maximum_residual_sigma_error": maximum_sigma_error,
        "actor_base_sigma": actor_base_sigma.tolist(),
        "stored_sigma_min": np.min(stored_sigma, axis=0).tolist(),
        "stored_sigma_max": np.max(stored_sigma, axis=0).tolist(),
        "stored_vs_actor_sigma_max_abs_error": sigma_error,
        "physical_center_to_normalized_jacobian_by_channel": physical_jacobian.tolist(),
        "physical_jacobian_condition_number": float(
            np.max(physical_jacobian) / np.min(physical_jacobian)
        ),
        "actor_pre_tanh_abs": audit.distribution(np.abs(pre_tanh_np)),
        "actor_requested_action_abs": audit.distribution(np.abs(requested_np)),
        "tanh_jacobian": audit.distribution(tanh_jacobian),
        "requested_vs_effective_action_max_abs_error": float(np.max(np.abs(
            requested_np - effective_np
        ))),
        "actor_center_clipped_element_fraction": float(np.mean(actor_clipped_element)),
        "actor_center_clipped_context_fraction": float(np.mean(actor_clipped_context)),
        "effective_vs_saved_actor_action_max_abs_error": float(np.max(np.abs(
            effective_np - actor_action
        ))),
        "reconstructed_vs_saved_center_max_abs_error": float(np.max(np.abs(
            center_np - actor_center
        ))),
        "fresh_fd_normalized_to_center_gradient_max_abs_error": (
            fd_coordinate_max_abs_error
        ),
        "fresh_fd_normalized_to_center_gradient_vector_relative_error": (
            audit.distribution(fd_coordinate_vector_relative_error)
        ),
        "pre_tanh_chain_autograd_context_count": chosen_count,
        "pre_tanh_chain_per_critic_max_abs_error": pre_tanh_errors,
        "pre_tanh_chain_max_abs_error": float(max(pre_tanh_errors, default=0.0)),
        "center_chain_per_critic_max_abs_error": center_errors,
        "center_chain_max_abs_error": float(max(center_errors, default=0.0)),
        "critic_vs_fresh_fd_by_coordinate": coordinate_comparison,
        "reward_coordinate_note": (
            "Critic and FD both differentiate z=asinh(raw_reward/5); converting "
            "both to raw reward multiplies every action dimension in one context "
            "by the same scalar and does not change cosine or norm ratio."
        ),
    }


def main() -> None:
    args = parse_args()
    summary_path = args.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    artifact_path = Path(summary["artifact"])
    arrays = load_npz(artifact_path)
    device = torch.device(args.device)

    hash_errors = {
        "artifact": sha256_file(artifact_path) != summary["artifact_sha256"],
        "critic_summary": sha256_file(Path(summary["critic_summary"]))
        != summary["critic_summary_sha256"],
        "initial_actor": sha256_file(Path(summary["initial_actor"]))
        != summary["initial_actor_sha256"],
        "critic_checkpoints": any(
            sha256_file(Path(path)) != digest
            for path, digest in zip(
                summary["critic_checkpoints"], summary["critic_checkpoint_sha256"]
            )
        ),
    }

    directions = arrays["directions"].reshape(16, 16)
    orthogonality_error = float(np.max(np.abs(
        directions @ directions.T - np.eye(16)
    )))
    direction_hash = hashlib.sha256(
        arrays["directions"].astype(np.float32).tobytes()
    ).hexdigest()

    critic_summary = json.loads(Path(summary["critic_summary"]).read_text())
    initial_actor_path = Path(summary["initial_actor"])
    initial_payload = torch.load(initial_actor_path, map_location="cpu")
    alpha_payload = torch.load(
        Path(initial_payload["base_alpha_checkpoint"]), map_location="cpu"
    )
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, splits = load_dataset(Path(initial_payload["labels"]), old_payload)
    context_index = np.flatnonzero(
        np.isin(data.episodes, splits["internal_selection"])
    )
    split_mapping_error = int(not np.array_equal(
        context_index, arrays["context_index"]
    ))
    episode_mapping_error = int(not np.array_equal(
        data.episodes[context_index].astype(str), arrays["episode"].astype(str)
    ))

    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    alpha_policy = make_base_policy(alpha_payload, device)
    _, _, _, alpha_center = deterministic_outputs(
        alpha_policy, tensors, extra, np.arange(len(data.episodes)),
        float(initial_payload["base_move_threshold"]),
        args.evaluation_batch_size, device,
    )
    inputs = actor_inputs(tensors, alpha_center, device)
    actor = TorchMPPIDeterministicCenterActor(
        float(initial_payload["maximum_residual_sigma"]), dropout=0.0
    ).to(device)
    actor.load_state_dict(initial_payload["actor_state_dict"], strict=True)
    actor.eval()
    actor_action, actor_center = residual_outputs(
        actor, inputs, context_index, args.evaluation_batch_size, device
    )
    actor_action_error = float(np.max(np.abs(actor_action - arrays["actor_action"])))
    actor_center_error = float(np.max(np.abs(actor_center - arrays["actor_center"])))
    alpha_center_error = float(np.max(np.abs(
        alpha_center[context_index] - arrays["alpha_center"]
    )))

    recalculated_gradient, recalculated_curvature, recalculated_rank = (
        local_train.fit_local_parameters(
            arrays["actions"].reshape(len(context_index), -1, 8, 2),
            arrays["transformed_reward"].reshape(len(context_index), -1),
            arrays["actor_action"],
        )
    )
    gradient_reconstruction_error = float(np.max(np.abs(
        recalculated_gradient - arrays["gradient"]
    )))
    curvature_reconstruction_error = float(np.max(np.abs(
        recalculated_curvature - arrays["curvature"]
    )))
    rank_reconstruction_error = int(np.max(np.abs(
        recalculated_rank - arrays["combined_design_rank"]
    )))

    models = []
    for checkpoint_path in summary["critic_checkpoints"]:
        payload = torch.load(checkpoint_path, map_location="cpu")
        model = TorchMPPIActorCenteredLocalCritic(dropout=0.0).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        models.append(model)
    critic_value, critic_gradient, critic_curvature = audit.critic_gradients(
        models, inputs, context_index, actor_action,
        args.evaluation_batch_size, device,
    )
    critic_value_error = float(np.max(np.abs(
        critic_value - arrays["critic_value"]
    )))
    critic_gradient_error = float(np.max(np.abs(
        critic_gradient - arrays["critic_gradient"]
    )))
    critic_curvature_error = float(np.max(np.abs(
        critic_curvature - arrays["critic_curvature"]
    )))

    rng = np.random.default_rng(260813)
    flat_centers = arrays["centers"].reshape(-1, 8, 2)
    flat_cost = arrays["cost"].reshape(-1)
    repeated_index = np.repeat(
        context_index,
        len(arrays["probe_radii_sigma"]) * arrays["centers"].shape[2],
    )
    count = min(args.replay_count, len(flat_centers))
    chosen = rng.choice(len(flat_centers), size=count, replace=False)
    replay_cost = direct_cost(
        flat_centers[chosen], data, tensors, repeated_index[chosen],
        args.evaluation_batch_size, device,
    )
    rollout_replay_error = float(np.max(np.abs(
        replay_cost - flat_cost[chosen]
    )))

    meaningful_norm = float(
        summary["critic_vs_fresh_fd"]["ensemble_mean"]
        ["meaningful_norm_threshold"]
    )
    primary = audit.gradient_metrics(
        np.mean(critic_gradient, axis=0), arrays["gradient"], meaningful_norm
    )
    primary_metric_error = numeric_error(
        summary["critic_vs_fresh_fd"]["ensemble_mean"], primary
    )
    autograd_check = audit.autograd_gradient_check(
        models, inputs, context_index, actor_action, device
    )
    coordinate_check = action_coordinate_check(
        actor, models, inputs, context_index, alpha_center, actor_action,
        actor_center, arrays, float(initial_payload["maximum_residual_sigma"]),
        device,
    )

    qualification = "PASS"
    if any(hash_errors.values()):
        qualification = "FAIL_INPUT_HASH"
    elif split_mapping_error or episode_mapping_error:
        qualification = "FAIL_SPLIT_MAPPING"
    elif max(actor_action_error, actor_center_error, alpha_center_error) > 1e-6:
        qualification = "FAIL_ACTOR_RECONSTRUCTION"
    elif max(gradient_reconstruction_error, curvature_reconstruction_error) > 1e-4:
        qualification = "FAIL_FD_RECONSTRUCTION"
    elif rank_reconstruction_error:
        qualification = "FAIL_DESIGN_RANK_RECONSTRUCTION"
    elif max(critic_value_error, critic_gradient_error, critic_curvature_error) > 1e-6:
        qualification = "FAIL_CRITIC_RECONSTRUCTION"
    elif rollout_replay_error > 5e-4:
        qualification = "FAIL_DBM_ROLLOUT_REPLAY"
    elif primary_metric_error > 1e-6:
        qualification = "FAIL_METRIC_REPLAY"
    elif autograd_check["maximum_abs_error"] > 1e-6:
        qualification = "FAIL_ACTION_AUTOGRAD"
    elif max(
        coordinate_check["maximum_residual_sigma_error"],
        coordinate_check["stored_vs_actor_sigma_max_abs_error"],
        coordinate_check["effective_vs_saved_actor_action_max_abs_error"],
        coordinate_check["reconstructed_vs_saved_center_max_abs_error"],
        coordinate_check["pre_tanh_chain_max_abs_error"],
        coordinate_check["center_chain_max_abs_error"],
    ) > 5e-6:
        qualification = "FAIL_ACTION_COORDINATE_CONTRACT"
    elif coordinate_check[
        "fresh_fd_normalized_to_center_gradient_max_abs_error"
    ] > 5e-5:
        qualification = "FAIL_FD_COORDINATE_JACOBIAN"
    elif orthogonality_error > 1e-5 or direction_hash != summary["direction_sha256"]:
        qualification = "FAIL_DIRECTION_BANK"

    result: dict[str, Any] = {
        "format_version": 1,
        "qualification": qualification,
        "output_dir": str(args.output_dir.resolve()),
        "hash_errors": hash_errors,
        "split_mapping_error": split_mapping_error,
        "episode_mapping_error": episode_mapping_error,
        "direction_orthogonality_max_abs_error": orthogonality_error,
        "direction_hash_match": direction_hash == summary["direction_sha256"],
        "actor_action_max_abs_error": actor_action_error,
        "actor_center_max_abs_error": actor_center_error,
        "alpha_center_max_abs_error": alpha_center_error,
        "gradient_reconstruction_max_abs_error": gradient_reconstruction_error,
        "curvature_reconstruction_max_abs_error": curvature_reconstruction_error,
        "design_rank_reconstruction_max_abs_error": rank_reconstruction_error,
        "critic_value_max_abs_error": critic_value_error,
        "critic_gradient_max_abs_error": critic_gradient_error,
        "critic_curvature_max_abs_error": critic_curvature_error,
        "rollout_replay_count": int(count),
        "rollout_replay_max_abs_error": rollout_replay_error,
        "primary_metric_max_abs_error": primary_metric_error,
        "autograd_gradient_check": autograd_check,
        "action_coordinate_check": coordinate_check,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    if qualification != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
