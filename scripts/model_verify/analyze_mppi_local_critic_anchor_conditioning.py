#!/usr/bin/env python3
"""Audit action-anchor conditioning of the frozen 16-D local Critic.

This is a zero-rollout consumed-split diagnostic.  It pairs the two first-pass
contexts belonging to each physical snapshot, verifies that physical state and
reference are identical, and cross-evaluates each Critic context at both absolute
Actor centers.  The audit distinguishes nuisance feedback changes from legitimate
gradient changes along the deterministic direct-cost action surface.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import (
    TorchMPPIAbsoluteCenterLocalCritic,
    TorchMPPIActorCenteredLocalCritic,
)
from generate_dbm_proposal_teacher import repository_state, sha256_file
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_residual_online_ac import actor_inputs, make_base_policy
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    load_actor_payload,
    load_dataset,
    tensorize,
)


DEFAULT_CRITIC = Path(
    "outputs/mppi_proposal/direct_local_gradient_critic_b4_smallest_target_20260813_v2"
)
DEFAULT_FRESH = Path(
    "outputs/mppi_proposal/direct_critic_fresh_fd_b4_smallest_target_20260813_v2"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_local_critic_anchor_conditioning_20260813_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--fresh-dir", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, np.float64).reshape(len(left), -1)
    right = np.asarray(right, np.float64).reshape(len(right), -1)
    return np.sum(left * right, axis=1) / (
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-12
    )


def distribution(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, np.float64)
    return {
        "mean": float(np.mean(value)),
        "minimum": float(np.min(value)),
        "p05": float(np.quantile(value, 0.05)),
        "p10": float(np.quantile(value, 0.10)),
        "p25": float(np.quantile(value, 0.25)),
        "median": float(np.median(value)),
        "p75": float(np.quantile(value, 0.75)),
        "p90": float(np.quantile(value, 0.90)),
        "p95": float(np.quantile(value, 0.95)),
        "maximum": float(np.max(value)),
    }


def pair_indices(episode: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, right = [], []
    for value in np.unique(episode):
        local = np.flatnonzero(episode == value)
        if len(local) != 10:
            raise AssertionError(f"expected ten contexts in {value}, got {len(local)}")
        for start in range(0, len(local), 2):
            left.append(int(local[start]))
            right.append(int(local[start + 1]))
    return np.asarray(left, np.int64), np.asarray(right, np.int64)


@torch.no_grad()
def predict_gradient(
    models: list[TorchMPPIActorCenteredLocalCritic],
    inputs: tuple[torch.Tensor, ...],
    context: np.ndarray,
    action: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    result = []
    for start in range(0, len(context), batch_size):
        local_context = context[start : start + batch_size]
        index = torch.from_numpy(local_context).to(device)
        anchor = torch.from_numpy(action[start : start + len(index)]).to(device)
        local = []
        for model in models:
            _, gradient, _ = model.local_parameters(
                *(value[index] for value in inputs), anchor
            )
            local.append(gradient.flatten(1))
        result.append(torch.stack(local).mean(0).cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)

    critic_summary_path = args.critic_dir / "summary.json"
    fresh_summary_path = args.fresh_dir / "summary.json"
    fresh_validation_path = args.fresh_dir / "validation_summary.json"
    fresh_npz_path = args.fresh_dir / "fresh_fd_audit.npz"
    critic_summary = json.loads(critic_summary_path.read_text())
    fresh_validation = json.loads(fresh_validation_path.read_text())
    if fresh_validation["qualification"] != "PASS":
        raise AssertionError("fresh FD source is not independently validated")

    initial_actor_path = Path(critic_summary["initial_actor"])
    initial_payload = torch.load(initial_actor_path, map_location="cpu")
    alpha_payload = torch.load(
        initial_payload["base_alpha_checkpoint"], map_location="cpu"
    )
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, _ = load_dataset(Path(initial_payload["labels"]), old_payload)
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    alpha_policy = make_base_policy(alpha_payload, device)
    _, _, _, alpha_center = deterministic_outputs(
        alpha_policy,
        tensors,
        extra,
        np.arange(len(data.episodes)),
        float(initial_payload["base_move_threshold"]),
        args.batch_size,
        device,
    )
    inputs = actor_inputs(tensors, alpha_center, device)

    with np.load(fresh_npz_path, allow_pickle=False) as archive:
        fresh = {key: np.asarray(archive[key]) for key in archive.files}
    context = np.asarray(fresh["context_index"], np.int64)
    episode = np.asarray(fresh["episode"])
    left, right = pair_indices(episode)
    left_context, right_context = context[left], context[right]

    # The direct DBM objective inputs must be identical inside each repeat pair.
    physical_errors = {
        "initial_state_max_abs": float(np.max(np.abs(
            data.initial_state_six[left_context]
            - data.initial_state_six[right_context]
        ))),
        "current_action_max_abs": float(np.max(np.abs(
            data.current_action[left_context] - data.current_action[right_context]
        ))),
        "direct_reference_max_abs": float(np.max(np.abs(
            data.direct_reference[left_context]
            - data.direct_reference[right_context]
        ))),
        "history_input_max_abs": float(np.max(np.abs(
            data.inputs[0][left_context] - data.inputs[0][right_context]
        ))),
        "reference_input_max_abs": float(np.max(np.abs(
            data.inputs[1][left_context] - data.inputs[1][right_context]
        ))),
        "current_input_max_abs": float(np.max(np.abs(
            data.inputs[2][left_context] - data.inputs[2][right_context]
        ))),
    }
    if max(physical_errors.values()) != 0.0:
        raise AssertionError(f"repeat physical inputs changed: {physical_errors}")

    normalization = old_payload["state_normalization"]
    raw_history = (
        data.inputs[0] * np.asarray(normalization["history_std"])
        + np.asarray(normalization["history_mean"])
    )
    raw_reference = (
        data.inputs[1] * np.asarray(normalization["reference_std"])
        + np.asarray(normalization["reference_mean"])
    )
    reference_unit_error = np.abs(
        np.square(raw_reference[..., 2]) + np.square(raw_reference[..., 3]) - 1.0
    )
    angle_audit = {
        "history_dyaw_minimum_rad": float(np.min(raw_history[..., 2])),
        "history_dyaw_maximum_rad": float(np.max(raw_history[..., 2])),
        "history_dyaw_abs_p99_rad": float(np.quantile(
            np.abs(raw_history[..., 2]), 0.99
        )),
        "history_dyaw_abs_gt_3_count": int(np.sum(
            np.abs(raw_history[..., 2]) > 3.0
        )),
        "reference_sincos_unit_max_abs_error": float(np.max(reference_unit_error)),
        "all_finite": bool(
            np.all(np.isfinite(raw_history)) and np.all(np.isfinite(raw_reference))
        ),
    }

    sigma = np.asarray(fresh["sigma"], np.float32)
    actor_center = np.asarray(fresh["actor_center"], np.float32)
    actor_action = np.asarray(fresh["actor_action"], np.float32)
    target = np.asarray(fresh["gradient"], np.float32)
    center_delta_sigma_rms = np.sqrt(np.mean(np.square(
        (actor_center[right] - actor_center[left]) / sigma[left, None]
    ), axis=(1, 2)))
    alpha_delta_sigma_rms = np.sqrt(np.mean(np.square(
        (fresh["alpha_center"][right] - fresh["alpha_center"][left])
        / sigma[left, None]
    ), axis=(1, 2)))
    residual_action_rms = np.sqrt(np.mean(np.square(
        actor_action[right] - actor_action[left]
    ), axis=(1, 2)))
    feedback_rms = np.sqrt(np.mean(np.square(
        data.inputs[4][right_context] - data.inputs[4][left_context]
    ), axis=1))
    gradient_context_rms = np.sqrt(np.mean(np.square(
        data.inputs[5][right_context] - data.inputs[5][left_context]
    ), axis=1))

    true_response_cosine = cosine_rows(target[left], target[right])
    true_reversal = true_response_cosine < 0.0
    action_step = (
        (actor_center[right] - actor_center[left]) / (2.0 * sigma[left, None])
    )
    left_directional = np.sum(
        target[left].reshape(-1, 8, 2) * action_step, axis=(1, 2)
    )
    right_directional = np.sum(
        target[right].reshape(-1, 8, 2) * action_step, axis=(1, 2)
    )

    models = []
    for checkpoint in critic_summary["checkpoints"]:
        payload = torch.load(checkpoint, map_location="cpu")
        if payload["model_class"] == "TorchMPPIActorCenteredLocalCritic":
            model = TorchMPPIActorCenteredLocalCritic(dropout=0.0).to(device)
        elif payload["model_class"] == "TorchMPPIAbsoluteCenterLocalCritic":
            model = TorchMPPIAbsoluteCenterLocalCritic(
                dropout=0.0,
                maximum_residual_sigma=float(payload["maximum_residual_sigma"]),
            ).to(device)
        else:
            raise AssertionError("unexpected Critic class")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        models.append(model.eval())

    maximum_residual_sigma = float(initial_payload["maximum_residual_sigma"])
    def action_at(rows: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return (
            (centers - alpha_center[rows])
            / (maximum_residual_sigma * data.sigma[rows, None])
        ).astype(np.float32)

    action_00 = action_at(left_context, actor_center[left])
    action_01 = action_at(left_context, actor_center[right])
    action_10 = action_at(right_context, actor_center[left])
    action_11 = action_at(right_context, actor_center[right])
    if max(
        float(np.max(np.abs(action_00 - actor_action[left]))),
        float(np.max(np.abs(action_11 - actor_action[right]))),
    ) > 2e-6:
        raise AssertionError("stored own action does not reconstruct absolute center")

    prediction_00 = predict_gradient(
        models, inputs, left_context, action_00, args.batch_size, device
    )
    prediction_01 = predict_gradient(
        models, inputs, left_context, action_01, args.batch_size, device
    )
    prediction_10 = predict_gradient(
        models, inputs, right_context, action_10, args.batch_size, device
    )
    prediction_11 = predict_gradient(
        models, inputs, right_context, action_11, args.batch_size, device
    )
    predicted_response_left = cosine_rows(prediction_00, prediction_01)
    predicted_response_right = cosine_rows(prediction_10, prediction_11)
    nuisance_same_center_left = cosine_rows(prediction_00, prediction_10)
    nuisance_same_center_right = cosine_rows(prediction_01, prediction_11)
    own_prediction = np.empty_like(target)
    own_prediction[left] = prediction_00
    own_prediction[right] = prediction_11
    own_target_cosine = cosine_rows(own_prediction, target)
    own_norm_ratio = np.linalg.norm(own_prediction, axis=1) / (
        np.linalg.norm(target, axis=1) + 1e-12
    )

    arrays_path = args.output_dir / "anchor_conditioning_audit.npz"
    np.savez_compressed(
        arrays_path,
        context_index=context,
        episode=episode,
        pair_left=left,
        pair_right=right,
        actor_center=actor_center,
        alpha_center=fresh["alpha_center"],
        actor_action=actor_action,
        sigma=sigma,
        target_gradient=target,
        center_delta_sigma_rms=center_delta_sigma_rms,
        alpha_delta_sigma_rms=alpha_delta_sigma_rms,
        residual_action_rms=residual_action_rms,
        feedback_rms=feedback_rms,
        gradient_context_rms=gradient_context_rms,
        true_response_cosine=true_response_cosine,
        true_reversal=true_reversal,
        left_directional_derivative=left_directional,
        right_directional_derivative=right_directional,
        action_00=action_00,
        action_01=action_01,
        action_10=action_10,
        action_11=action_11,
        prediction_00=prediction_00,
        prediction_01=prediction_01,
        prediction_10=prediction_10,
        prediction_11=prediction_11,
        own_prediction=own_prediction,
        own_target_cosine=own_target_cosine,
        own_norm_ratio=own_norm_ratio,
        predicted_response_left=predicted_response_left,
        predicted_response_right=predicted_response_right,
        nuisance_same_center_left=nuisance_same_center_left,
        nuisance_same_center_right=nuisance_same_center_right,
    )

    reversal_count = int(np.sum(true_reversal))
    gates = {
        "fresh_gradient_cosine_median_ge_0_70": (
            float(np.median(own_target_cosine)) >= 0.70
        ),
        "fresh_gradient_cosine_p10_ge_0": (
            float(np.quantile(own_target_cosine, 0.10)) >= 0.0
        ),
        "fresh_gradient_norm_ratio_median_ge_0_50": (
            float(np.median(own_norm_ratio)) >= 0.50
        ),
        "fresh_gradient_norm_ratio_median_le_2_00": (
            float(np.median(own_norm_ratio)) <= 2.00
        ),
        "true_reversal_flip_recall_mean_ge_0_50": (
            float(np.mean(np.concatenate((
                predicted_response_left[true_reversal] < 0.0,
                predicted_response_right[true_reversal] < 0.0,
            )))) >= 0.50
        ),
        "same_center_nuisance_cosine_median_ge_0_90": (
            float(np.median(np.concatenate((
                nuisance_same_center_left, nuisance_same_center_right,
            )))) >= 0.90
        ),
    }
    qualification = (
        "ACTION_LOCATION_CONDITIONING_PASS"
        if all(gates.values())
        else "ACTION_LOCATION_CONDITIONING_FAIL"
    )
    summary: dict[str, Any] = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "sources": {
            "critic_summary": str(critic_summary_path.resolve()),
            "critic_summary_sha256": sha256_file(critic_summary_path),
            "fresh_summary": str(fresh_summary_path.resolve()),
            "fresh_summary_sha256": sha256_file(fresh_summary_path),
            "fresh_validation": str(fresh_validation_path.resolve()),
            "fresh_validation_sha256": sha256_file(fresh_validation_path),
            "fresh_fd_audit": str(fresh_npz_path.resolve()),
            "fresh_fd_audit_sha256": sha256_file(fresh_npz_path),
            "critic_checkpoints": [str(Path(x).resolve()) for x in critic_summary["checkpoints"]],
            "critic_checkpoint_sha256": [
                sha256_file(Path(x)) for x in critic_summary["checkpoints"]
            ],
        },
        "counts": {
            "context": int(len(context)),
            "physical_pair": int(len(left)),
            "true_gradient_reversal_pair": reversal_count,
        },
        "physical_input_identity": physical_errors,
        "angle_encoding_audit": angle_audit,
        "repeat_input_change": {
            "alpha_center_sigma_rms": distribution(alpha_delta_sigma_rms),
            "actor_center_sigma_rms": distribution(center_delta_sigma_rms),
            "residual_actor_action_rms": distribution(residual_action_rms),
            "normalized_feedback_rms": distribution(feedback_rms),
            "normalized_gradient_context_rms": distribution(gradient_context_rms),
        },
        "true_action_response": {
            "gradient_cosine": distribution(true_response_cosine),
            "reversal_fraction": float(np.mean(true_reversal)),
            "reversal_center_delta_sigma_rms": distribution(
                center_delta_sigma_rms[true_reversal]
            ),
            "reversal_directional_derivative_positive_to_negative_fraction": float(
                np.mean(
                    (left_directional[true_reversal] > 0.0)
                    & (right_directional[true_reversal] < 0.0)
                )
            ),
        },
        "critic_action_response": {
            "own_fresh_target_cosine": distribution(own_target_cosine),
            "own_fresh_target_norm_ratio": distribution(own_norm_ratio),
            "fixed_left_context_cosine": distribution(predicted_response_left),
            "fixed_right_context_cosine": distribution(predicted_response_right),
            "true_reversal_flip_recall_left": float(np.mean(
                predicted_response_left[true_reversal] < 0.0
            )),
            "true_reversal_flip_recall_right": float(np.mean(
                predicted_response_right[true_reversal] < 0.0
            )),
            "same_absolute_center_nuisance_cosine_left": distribution(
                nuisance_same_center_left
            ),
            "same_absolute_center_nuisance_cosine_right": distribution(
                nuisance_same_center_right
            ),
            "cross_anchor_target_cosine_left_context": distribution(
                cosine_rows(prediction_01, target[right])
            ),
            "cross_anchor_target_cosine_right_context": distribution(
                cosine_rows(prediction_10, target[left])
            ),
        },
        "gates": gates,
        "artifacts": {
            "audit_npz": str(arrays_path.resolve()),
            "audit_npz_sha256": sha256_file(arrays_path),
        },
        "interpretation": [
            "The deterministic objective is smooth locally; a reversal across two distinct centers is not an angle-wrap discontinuity.",
            "Repeat contexts are not same-target gradient duplicates because their absolute Actor centers differ.",
            "The current Critic is almost invariant to cross-anchor action changes and therefore does not model g(s,a) adequately.",
        ],
        "contract": {
            "new_dbm_rollouts": 0,
            "actor_updated": False,
            "critic_updated": False,
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
    }
    summary_path = args.output_dir / "analysis.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Local-Critic anchor-conditioning audit\n\n"
        f"Qualification: `{qualification}`.\n\n"
        "This consumed-split audit performs no rollout or training.  See analysis.json "
        "and the independent validation summary before changing Critic supervision.\n"
    )
    print(json.dumps({
        "qualification": summary["qualification"],
        "counts": summary["counts"],
        "angle_encoding_audit": angle_audit,
        "repeat_input_change": summary["repeat_input_change"],
        "true_action_response": summary["true_action_response"],
        "critic_action_response": summary["critic_action_response"],
    }, indent=2))


if __name__ == "__main__":
    main()
