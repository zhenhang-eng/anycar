#!/usr/bin/env python3
"""Tiny-set memorization test for the actor-centered local MPPI Critic.

This is a zero-new-rollout diagnostic.  It reuses the stored 0.05-sigma finite-
difference labels, freezes the Actor, and repeatedly fits the exact same 32
internal-fit contexts.  Two arms separate basic gradient memorization from the
production multi-task objective used by the B4 Critic.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from car_foundation.mppi_proposal_policy import (
    TorchMPPIActorCenteredLocalCritic,
    TorchMPPIDeterministicCenterActor,
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


DEFAULT_SOURCE = Path(
    "outputs/mppi_proposal/direct_local_gradient_critic_b4_smallest_target_20260813_v2"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_local_critic_tiny_overfit_20260813_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context-count", type=int, default=32)
    parser.add_argument("--selection-seed", type=int, default=260813)
    parser.add_argument("--training-seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--evaluation-interval", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gradient-log-norm-weight", type=float, default=0.20)
    parser.add_argument("--safe-center-limit", type=float, default=0.95)
    parser.add_argument("--minimum-target-norm", type=float, default=1e-3)
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


def distribution(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, np.float64)
    return {
        "minimum": float(np.min(value)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
        "maximum": float(np.max(value)),
        "mean": float(np.mean(value)),
    }


def gradient_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, np.float64).reshape(len(prediction), -1)
    target = np.asarray(target, np.float64).reshape(len(target), -1)
    pred_norm = np.linalg.norm(prediction, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    cosine = np.sum(prediction * target, axis=1) / (
        pred_norm * target_norm + 1e-12
    )
    ratio = (pred_norm + 1e-8) / (target_norm + 1e-8)
    return {
        "count": int(len(target)),
        "cosine": distribution(cosine),
        "positive_fraction": float(np.mean(cosine > 0.0)),
        "above_0_98_fraction": float(np.mean(cosine > 0.98)),
        "norm_ratio": distribution(ratio),
        "component_rmse": float(np.sqrt(np.mean(np.square(prediction - target)))),
        "component_max_abs_error": float(np.max(np.abs(prediction - target))),
    }


def memorization_gate(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["cosine"]["median"] > 0.99
        and metrics["cosine"]["p10"] > 0.98
        and metrics["cosine"]["minimum"] > 0.98
        and 0.95 <= metrics["norm_ratio"]["median"] <= 1.05
        and metrics["norm_ratio"]["minimum"] >= 0.80
        and metrics["norm_ratio"]["maximum"] <= 1.20
    )


def select_contexts(
    labels: dict[str, np.ndarray],
    data: Any,
    train_episodes: set[str],
    count: int,
    seed: int,
    safe_center_limit: float,
    minimum_target_norm: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target = labels["gradient_by_radius"][:, 0]
    target_norm = np.linalg.norm(target, axis=1)
    safe_center = np.max(np.abs(labels["actor_center"]), axis=(1, 2))
    candidate = np.flatnonzero(
        np.asarray([episode in train_episodes for episode in labels["episode"]])
        & (safe_center <= safe_center_limit)
        & (target_norm >= minimum_target_norm)
    )
    if len(candidate) < count:
        raise AssertionError(f"only {len(candidate)} eligible contexts for {count}")

    context_index = labels["context_index"]
    speed = data.reference_speed[context_index]
    scenario = data.scenario[context_index]
    groups: dict[tuple[str, str], list[int]] = {}
    for position in candidate:
        key = (f"{float(speed[position]):.1f}", str(scenario[position]))
        groups.setdefault(key, []).append(int(position))
    for values in groups.values():
        rng.shuffle(values)

    selected: list[int] = []
    used_episode: set[str] = set()
    for key in sorted(groups):
        for position in groups[key]:
            episode = str(labels["episode"][position])
            if episode not in used_episode:
                selected.append(position)
                used_episode.add(episode)
                break
        if len(selected) >= count:
            break

    remainder = candidate.copy()
    rng.shuffle(remainder)
    for position in remainder:
        episode = str(labels["episode"][position])
        if episode in used_episode or int(position) in selected:
            continue
        selected.append(int(position))
        used_episode.add(episode)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise AssertionError("could not select one context per independent episode")
    return np.asarray(selected, np.int64)


def model_parameters(
    model: TorchMPPIActorCenteredLocalCritic,
    inputs: tuple[torch.Tensor, ...],
    global_index: torch.Tensor,
    anchor_action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return model.local_parameters(
        *(value[global_index] for value in inputs), anchor_action
    )


@torch.no_grad()
def evaluate(
    model: TorchMPPIActorCenteredLocalCritic,
    inputs: tuple[torch.Tensor, ...],
    global_index: torch.Tensor,
    anchor_action: torch.Tensor,
    target_gradient: torch.Tensor,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    _, prediction, _ = model_parameters(model, inputs, global_index, anchor_action)
    prediction_np = prediction.flatten(1).cpu().numpy()
    return (
        gradient_metrics(prediction_np, target_gradient.cpu().numpy()),
        prediction_np,
    )


def train_arm(
    arm: str,
    seed: int,
    actor: TorchMPPIDeterministicCenterActor,
    inputs: tuple[torch.Tensor, ...],
    labels: dict[str, np.ndarray],
    selected: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TorchMPPIActorCenteredLocalCritic, dict[str, Any], np.ndarray]:
    if arm not in {"gradient_only", "b4_multitask"}:
        raise ValueError(arm)
    set_seed(seed)
    model = TorchMPPIActorCenteredLocalCritic(dropout=0.0).to(device)
    model.encoder.load_state_dict(actor.encoder.state_dict(), strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )

    global_index = torch.from_numpy(labels["context_index"][selected]).to(device)
    anchor = torch.from_numpy(labels["actor_action"][selected]).to(device)
    target_gradient = torch.from_numpy(
        labels["gradient_by_radius"][selected, 0]
    ).to(device)
    target_value = torch.from_numpy(labels["value"][selected]).to(device)
    target_curvature = torch.from_numpy(labels["curvature"][selected]).to(device)
    action_bank = torch.from_numpy(labels["actions"][selected]).to(device).reshape(
        len(selected), -1, 8, 2
    )
    target_bank = torch.from_numpy(
        labels["transformed_reward"][selected]
    ).to(device).reshape(len(selected), -1)

    gradient_scale = torch.from_numpy(
        np.maximum(
            np.std(labels["gradient_by_radius"][selected, 0], axis=0), 0.05
        ).astype(np.float32)
    ).to(device)
    value_scale = max(float(np.std(labels["value"][selected])), 0.05)
    curvature_scale = max(float(np.std(labels["curvature"][selected])), 0.05)
    bank_scale = max(
        float(np.std(labels["transformed_reward"][selected])), 0.05
    )

    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_prediction: np.ndarray | None = None
    best_passed = False
    consecutive_passes = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        value, gradient, curvature = model_parameters(
            model, inputs, global_index, anchor
        )
        flat_gradient = gradient.flatten(1)
        gradient_loss = F.smooth_l1_loss(
            (flat_gradient - target_gradient) / gradient_scale,
            torch.zeros_like(flat_gradient),
            beta=0.5,
        )
        prediction_norm = torch.linalg.vector_norm(flat_gradient, dim=1)
        target_norm = torch.linalg.vector_norm(target_gradient, dim=1)
        log_norm_loss = F.smooth_l1_loss(
            torch.log(prediction_norm + 1e-4) - torch.log(target_norm + 1e-4),
            torch.zeros_like(prediction_norm),
            beta=0.5,
        )
        cosine_loss = (
            1.0 - F.cosine_similarity(flat_gradient, target_gradient, dim=1)
        ).mean()
        value_loss = F.smooth_l1_loss(
            (value - target_value) / value_scale,
            torch.zeros_like(value),
            beta=0.5,
        )
        curvature_loss = F.smooth_l1_loss(
            (curvature - target_curvature) / curvature_scale,
            torch.zeros_like(curvature),
            beta=0.5,
        )
        action_delta = action_bank - anchor[:, None]
        predicted_bank = model.local_value(
            value[:, None], gradient[:, None], curvature[:, None], action_delta
        )
        bank_loss = F.smooth_l1_loss(
            (predicted_bank - target_bank) / bank_scale,
            torch.zeros_like(predicted_bank),
            beta=0.5,
        )
        if arm == "gradient_only":
            loss = gradient_loss + args.gradient_log_norm_weight * log_norm_loss
        else:
            loss = (
                0.20 * value_loss
                + gradient_loss
                + 0.50 * cosine_loss
                + 0.05 * curvature_loss
                + 0.25 * bank_loss
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if epoch == 1 or epoch % args.evaluation_interval == 0:
            metrics, prediction = evaluate(
                model, inputs, global_index, anchor, target_gradient
            )
            score = (
                1.0 - metrics["cosine"]["median"]
                + 1.0 - metrics["cosine"]["p10"]
                + abs(math.log(max(metrics["norm_ratio"]["median"], 1e-8)))
                + metrics["component_rmse"] / (
                    float(torch.linalg.vector_norm(target_gradient, dim=1).median())
                    + 1e-8
                )
            )
            passed = memorization_gate(metrics)
            if (passed and not best_passed) or (passed == best_passed and score < best_score):
                best_score = float(score)
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_prediction = prediction.copy()
                best_passed = passed
            consecutive_passes = consecutive_passes + 1 if passed else 0
            history.append({
                "epoch": epoch,
                "total_loss": float(loss.detach()),
                "gradient_loss": float(gradient_loss.detach()),
                "log_norm_loss": float(log_norm_loss.detach()),
                "cosine_loss": float(cosine_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "curvature_loss": float(curvature_loss.detach()),
                "bank_loss": float(bank_loss.detach()),
                "metrics": metrics,
                "memorization_gate": passed,
            })
            if epoch == 1 or epoch % 250 == 0 or passed:
                print(json.dumps({
                    "arm": arm,
                    "seed": seed,
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "cosine_median": metrics["cosine"]["median"],
                    "cosine_p10": metrics["cosine"]["p10"],
                    "norm_ratio_median": metrics["norm_ratio"]["median"],
                    "norm_ratio_p10": metrics["norm_ratio"]["p10"],
                    "norm_ratio_p90": metrics["norm_ratio"]["p90"],
                    "gate": passed,
                }), flush=True)
            if consecutive_passes >= 3:
                break

    final_metrics, final_prediction = evaluate(
        model, inputs, global_index, anchor, target_gradient
    )
    if best_state is None or best_prediction is None:
        raise AssertionError("no evaluated tiny-set checkpoint")
    model.load_state_dict(best_state, strict=True)
    best_metrics, replayed_best_prediction = evaluate(
        model, inputs, global_index, anchor, target_gradient
    )
    if not np.array_equal(best_prediction, replayed_best_prediction):
        maximum_error = float(
            np.max(np.abs(best_prediction - replayed_best_prediction))
        )
        if maximum_error > 1e-6:
            raise AssertionError(f"best checkpoint replay changed by {maximum_error}")
    record = {
        "arm": arm,
        "seed": seed,
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "best_memorization_gate": memorization_gate(best_metrics),
        "final_memorization_gate": memorization_gate(final_metrics),
        "history": history,
        "regularization": {
            "dropout": 0.0,
            "weight_decay": 0.0,
            "scheduler": False,
            "early_stopping": False,
            "actor_frozen": True,
        },
        "loss_contract": (
            "gradient component Smooth-L1 + 0.20 log-norm Smooth-L1"
            if arm == "gradient_only"
            else "B4 value/gradient/cosine/curvature/all-radius-bank weights"
        ),
    }
    # Return the demonstrably best-capacity checkpoint while preserving final
    # metrics to expose any late multi-task degradation.
    model.load_state_dict(best_state, strict=True)
    return model, record, replayed_best_prediction


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.context_count < 8:
        raise ValueError("context-count must be at least 8")
    if args.epochs < 1 or args.evaluation_interval < 1:
        raise ValueError("epochs/evaluation-interval must be positive")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)

    source_summary_path = args.source_dir / "summary.json"
    source_labels_path = args.source_dir / "local_forward_labels.npz"
    source_summary = json.loads(source_summary_path.read_text())
    if source_summary["contract"]["gradient_target"] != "smallest":
        raise AssertionError("source Critic did not use the smallest-radius target")
    initial_actor_path = Path(source_summary["initial_actor"])
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
        256,
        device,
    )
    inputs = actor_inputs(tensors, alpha_center, device)
    actor = TorchMPPIDeterministicCenterActor(
        float(initial_payload["maximum_residual_sigma"]), dropout=0.0
    ).to(device)
    actor.load_state_dict(initial_payload["actor_state_dict"], strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)

    with np.load(source_labels_path, allow_pickle=False) as archive:
        labels = {key: np.asarray(archive[key]) for key in archive.files}
    labels["gradient"] = np.asarray(labels["gradient_by_radius"][:, 0], np.float32)
    train_episodes = set(source_summary["split"]["train_episodes"])
    selected = select_contexts(
        labels,
        data,
        train_episodes,
        args.context_count,
        args.selection_seed,
        args.safe_center_limit,
        args.minimum_target_norm,
    )

    selected_global = labels["context_index"][selected]
    selected_index = torch.from_numpy(selected_global).to(device)
    with torch.no_grad():
        actor_action, actor_center = actor(*(value[selected_index] for value in inputs))
    actor_action_error = float(np.max(np.abs(
        actor_action.cpu().numpy() - labels["actor_action"][selected]
    )))
    actor_center_error = float(np.max(np.abs(
        actor_center.cpu().numpy() - labels["actor_center"][selected]
    )))
    if actor_action_error > 5e-5 or actor_center_error > 5e-5:
        raise AssertionError("frozen Actor no longer reconstructs source labels")

    target = labels["gradient_by_radius"][selected, 0]
    selection_rows = []
    for position in selected:
        index = int(labels["context_index"][position])
        selection_rows.append({
            "label_position": int(position),
            "context_index": index,
            "episode": str(labels["episode"][position]),
            "reference_speed_mps": float(data.reference_speed[index]),
            "scenario": str(data.scenario[index]),
            "target_gradient_norm": float(
                np.linalg.norm(labels["gradient_by_radius"][position, 0])
            ),
            "actor_center_max_abs": float(
                np.max(np.abs(labels["actor_center"][position]))
            ),
        })
    (args.output_dir / "tiny_set_manifest.json").write_text(
        json.dumps({
            "selection_seed": args.selection_seed,
            "context_count": args.context_count,
            "one_context_per_episode": len({x["episode"] for x in selection_rows})
            == args.context_count,
            "rows": selection_rows,
        }, indent=2) + "\n"
    )
    np.savez_compressed(
        args.output_dir / "tiny_set.npz",
        label_position=selected,
        context_index=selected_global,
        target_gradient=target,
        episode=labels["episode"][selected],
    )

    runs = []
    predictions: dict[str, np.ndarray] = {}
    for arm in ("gradient_only", "b4_multitask"):
        for seed in [int(value) for value in args.training_seeds.split(",")]:
            model, record, prediction = train_arm(
                arm, seed, actor, inputs, labels, selected, args, device
            )
            checkpoint_path = args.output_dir / f"{arm}_seed{seed}.pt"
            torch.save({
                "format_version": 1,
                "model_class": "TorchMPPIActorCenteredLocalCritic",
                "model_state_dict": model.state_dict(),
                "arm": arm,
                "seed": seed,
                "source_summary": str(source_summary_path.resolve()),
                "source_summary_sha256": sha256_file(source_summary_path),
                "source_labels": str(source_labels_path.resolve()),
                "source_labels_sha256": sha256_file(source_labels_path),
                "initial_actor": str(initial_actor_path.resolve()),
                "initial_actor_sha256": sha256_file(initial_actor_path),
                "selected_label_positions": selected,
                "selected_context_index": selected_global,
            }, checkpoint_path)
            record["checkpoint"] = str(checkpoint_path.resolve())
            record["checkpoint_sha256"] = sha256_file(checkpoint_path)
            runs.append(record)
            predictions[f"{arm}_seed{seed}"] = prediction

    np.savez_compressed(
        args.output_dir / "predictions.npz",
        target_gradient=target,
        **predictions,
    )
    by_arm = {}
    for arm in ("gradient_only", "b4_multitask"):
        local = [record for record in runs if record["arm"] == arm]
        by_arm[arm] = {
            "run_count": len(local),
            "best_gate_pass_count": int(sum(
                record["best_memorization_gate"] for record in local
            )),
            "final_gate_pass_count": int(sum(
                record["final_memorization_gate"] for record in local
            )),
            "cosine_median": distribution(np.asarray([
                record["best_metrics"]["cosine"]["median"] for record in local
            ])),
            "cosine_p10": distribution(np.asarray([
                record["best_metrics"]["cosine"]["p10"] for record in local
            ])),
            "norm_ratio_median": distribution(np.asarray([
                record["best_metrics"]["norm_ratio"]["median"] for record in local
            ])),
        }
    gradient_pass = by_arm["gradient_only"]["best_gate_pass_count"] == len(
        [record for record in runs if record["arm"] == "gradient_only"]
    )
    multitask_pass = by_arm["b4_multitask"]["best_gate_pass_count"] == len(
        [record for record in runs if record["arm"] == "b4_multitask"]
    )
    if gradient_pass and multitask_pass:
        qualification = "TINY_OVERFIT_PASS_BASIC_AND_MULTITASK"
        conclusion = (
            "The Critic can memorize exact 0.05-sigma gradients under both the "
            "minimal and B4 multi-task objectives. Remaining full-data failure is "
            "not a basic architecture/optimizer capacity failure."
        )
    elif gradient_pass:
        qualification = "TINY_OVERFIT_PASS_BASIC_MULTITASK_CONFLICT"
        conclusion = (
            "The Critic can memorize gradients in isolation but not under the B4 "
            "multi-task loss; prioritize loss conflict before dataset expansion."
        )
    else:
        qualification = "TINY_OVERFIT_BASIC_FAILURE"
        conclusion = (
            "The Critic cannot memorize 32 fixed gradients even in isolation; "
            "prioritize implementation, optimizer, scaling, or architecture checks."
        )

    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "repository": repository_state(Path(__file__).resolve().parents[2]),
        "source_summary": str(source_summary_path.resolve()),
        "source_summary_sha256": sha256_file(source_summary_path),
        "source_labels": str(source_labels_path.resolve()),
        "source_labels_sha256": sha256_file(source_labels_path),
        "initial_actor": str(initial_actor_path.resolve()),
        "initial_actor_sha256": sha256_file(initial_actor_path),
        "tiny_set_manifest_sha256": sha256_file(
            args.output_dir / "tiny_set_manifest.json"
        ),
        "tiny_set_npz_sha256": sha256_file(args.output_dir / "tiny_set.npz"),
        "context_count": args.context_count,
        "target": "stored smallest-radius 0.05-sigma finite-difference gradient",
        "target_gradient_norm": distribution(np.linalg.norm(target, axis=1)),
        "frozen_actor_reconstruction": {
            "maximum_action_error": actor_action_error,
            "maximum_center_error": actor_center_error,
        },
        "memorization_gate": {
            "cosine_median_gt": 0.99,
            "cosine_p10_gt": 0.98,
            "cosine_minimum_gt": 0.98,
            "norm_ratio_median_range": [0.95, 1.05],
            "norm_ratio_minimum": 0.80,
            "norm_ratio_maximum": 1.20,
        },
        "by_arm": by_arm,
        "runs": runs,
        "conclusion": conclusion,
        "contract": {
            "new_dbm_rollouts": 0,
            "actor_frozen": True,
            "critic_updated": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "selected_from": "internal-fit train episodes only",
            "dropout": 0.0,
            "weight_decay": 0.0,
            "scheduler": False,
            "early_stopping": False,
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (args.output_dir / "README.md").write_text(
        "# Local-Critic tiny-set overfit diagnostic\n\n"
        f"Qualification: `{qualification}`.\n\n{conclusion}\n"
    )
    print(json.dumps({
        "qualification": qualification,
        "by_arm": by_arm,
        "conclusion": conclusion,
    }, indent=2))


if __name__ == "__main__":
    main()
