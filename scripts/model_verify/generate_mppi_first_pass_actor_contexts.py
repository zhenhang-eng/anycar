#!/usr/bin/env python3
"""Generate train-only first-pass contexts for the deterministic center Actor.

The frozen BC policy produces the pass-one center.  Fixed antithetic forward
rollouts then produce the existing 74-D feedback vector and guided anchor.  A
frozen feedback-Critic ensemble supplies the same 16-D mean and 16-D standard
deviation context used by the current Actor.  No DBM analytic gradient is used.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from compare_guided_mppi_stage_counts import make_device_independent_antithetic_candidates
from evaluate_mppi_proposal_bc import load_policy, predict_center
from generate_dbm_multicenter_teacher import (
    evaluate_actions,
    load_config,
    make_controller,
    stable_weight,
)
from generate_dbm_proposal_teacher import repository_state, sha256_file
from generate_dbm_two_pass_feedback_labels import (
    ACTION_DIMENSION,
    FEEDBACK_NAMES,
    antithetic_noise,
)
from generate_dbm_two_pass_risk_replay_labels import (
    critic_directions,
    load_critic_ensemble,
)
from guide_mppi_sampling_from_trajectory_error import (
    evaluate_knots as evaluate_knots_torch,
    fit_guided_center,
)


FORMAT_VERSION = 1
GENERATOR_ID = "anycar-first-pass-actor-context-v1"
DEFAULT_ACTOR = Path(
    "outputs/mppi_proposal/bc_t1_conv_diverse_20260805_v1/trust1_seed1.pt"
)
DEFAULT_CRITIC = Path("outputs/mppi_proposal/critic_two_pass_feedback_20260805_v1")
DEFAULT_CONFIG = Path(__file__).with_name("dbm_teacher_t1_diverse_20260805_v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-seeds", default="24001,24002")
    parser.add_argument("--first-samples", type=int, default=128)
    parser.add_argument("--fit-ridge", type=float, default=0.10)
    parser.add_argument("--step-damping", type=float, default=0.10)
    parser.add_argument("--max-standardized-step", type=float, default=1.0)
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument("--reference-feedback", type=Path)
    parser.add_argument("--reference-risk", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--first-seeds must contain distinct integers")
    return seeds


def first_pass(
    base: np.ndarray,
    seed: int,
    source: np.lib.npyio.NpzFile,
    controller: Any,
    backend: Any,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    params = json.loads(str(source["mppi_params_json"]))
    sigma = np.asarray(params["noise_sigma"], np.float32)
    low = np.asarray(params["action_min"], np.float32)
    high = np.asarray(params["action_max"], np.float32)
    center_t = torch.from_numpy(base).to(controller.device)
    sigma_t = torch.from_numpy(sigma).to(controller.device)
    history = torch.from_numpy(source["history"]).to(controller.device)
    initial = torch.from_numpy(source["initial_state"]).to(controller.device).reshape(1, 5)
    current = torch.from_numpy(source["current_action"]).to(controller.device).reshape(1, 2)
    reference = controller._prepare_reference(source["reference"])
    knots = make_device_independent_antithetic_candidates(
        center_t,
        sigma_t,
        args.first_samples,
        np.random.default_rng(seed),
        None,
        tuple(low),
        tuple(high),
    )
    evaluated = evaluate_knots_torch(
        controller, backend, knots, history, initial, current, reference
    )
    guided, guided_step, response, fit = fit_guided_center(
        evaluated,
        center_t,
        sigma_t,
        args.fit_ridge,
        args.step_damping,
        args.max_standardized_step,
    )
    cost = evaluated.cost.detach().cpu().numpy().astype(np.float32)
    knots_np = evaluated.knots.detach().cpu().numpy().astype(np.float32)
    weight = stable_weight(cost, float(controller.params.temperature))
    weighted_actions = np.sum(
        weight[:, None, None] * evaluated.actions.detach().cpu().numpy(), axis=0
    ).astype(np.float32)
    weighted_output_cost, _ = evaluate_actions(
        controller,
        backend,
        weighted_actions[None],
        history,
        initial,
        current,
        reference,
    )
    normalized_delta = ((knots_np - base) / sigma.reshape(1, 1, 2)).reshape(
        args.first_samples, ACTION_DIMENSION
    )
    weighted_shift = np.sum(weight[:, None] * normalized_delta, axis=0)
    baseline_residual = evaluated.residuals[0]
    empirical_gradient = (2.0 * (response @ baseline_residual)).detach().cpu().numpy()
    empirical_hessian_diagonal = (
        2.0 * torch.sum(response.square(), dim=1)
    ).detach().cpu().numpy()
    raw = base[None] + antithetic_noise(
        seed, args.first_samples, tuple(base.shape), sigma
    )
    temperature = float(controller.params.temperature)
    softmin = float(
        cost.min()
        - temperature
        * np.log(np.mean(np.exp(-(cost - cost.min()) / temperature)))
    )
    scalar = np.asarray(
        [
            cost[0],
            cost.min(),
            np.quantile(cost, 0.10),
            np.median(cost),
            cost.mean(),
            weighted_output_cost[0],
            softmin,
            1.0 / np.sum(weight * weight) / args.first_samples,
            np.mean(raw != knots_np),
            fit["relative_weighted_fit_error"],
        ],
        np.float32,
    )
    feedback = np.concatenate(
        (
            guided_step.detach().cpu().numpy().reshape(-1),
            empirical_gradient.reshape(-1),
            empirical_hessian_diagonal.reshape(-1),
            weighted_shift.reshape(-1),
            scalar,
        )
    ).astype(np.float32)
    if feedback.shape != (len(FEEDBACK_NAMES),):
        raise AssertionError("feedback dimension mismatch")
    return {
        "first_pass_seed": np.asarray(seed, np.int64),
        "first_pass_knots": knots_np,
        "first_pass_cost": cost,
        "first_pass_feedback": feedback,
        "guided_center_knots": guided.detach().cpu().numpy().astype(np.float32),
    }


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.first_seeds)
    if args.first_samples < 4 or args.first_samples % 2:
        raise ValueError("--first-samples must be an even integer >=4")
    source_root = args.source.resolve()
    split_path = args.splits.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    split_data = json.loads(split_path.read_text())
    train_episodes = list(split_data["train"])
    forbidden = set(split_data.get("validation", ())) | set(split_data.get("test", ()))
    if set(train_episodes) & forbidden:
        raise AssertionError("train overlaps held-out episodes")
    paths = [
        path
        for episode in train_episodes
        for path in sorted((source_root / episode / "snapshots").glob("*.npz"))
    ]
    if args.max_snapshots:
        paths = paths[: args.max_snapshots]
    if not paths:
        raise ValueError("no train snapshots selected")
    device = torch.device(args.device)
    actor, actor_normalization, _ = load_policy(args.actor_checkpoint.resolve(), device)
    models, critic_normalization, feedback_mean, feedback_std, critic_paths = (
        load_critic_ensemble(args.critic_dir.resolve(), device)
    )
    config = load_config(args.config.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    started = time.perf_counter()
    feedback_regression, anchor_regression = [], []
    gradient_mean_regression, gradient_std_regression = [], []
    first_base_cost, first_best_cost = [], []
    try:
        for index, source_path in enumerate(paths, 1):
            source_hash = sha256_file(source_path)
            with np.load(source_path, allow_pickle=False) as source:
                _, base = predict_center(actor, actor_normalization, source, device)
                controller, backend = make_controller(source, config, device)
                repeats = [first_pass(base, seed, source, controller, backend, args) for seed in seeds]
                anchors = np.stack([value["guided_center_knots"] for value in repeats])
                feedback = np.stack([value["first_pass_feedback"] for value in repeats])
                _, gradient_mean, gradient_std = critic_directions(
                    models,
                    critic_normalization,
                    feedback_mean,
                    feedback_std,
                    source,
                    anchors,
                    feedback,
                    device,
                )
            arrays = {
                "format_version": np.asarray(FORMAT_VERSION, np.int32),
                "source_snapshot_sha256": np.asarray(source_hash),
                "base_actor_checkpoint_sha256": np.asarray(
                    sha256_file(args.actor_checkpoint.resolve())
                ),
                "base_center_knots": base,
                "feedback_names": np.asarray(FEEDBACK_NAMES),
                "first_pass_seed": np.stack([value["first_pass_seed"] for value in repeats]),
                "first_pass_knots": np.stack([value["first_pass_knots"] for value in repeats]),
                "first_pass_cost": np.stack([value["first_pass_cost"] for value in repeats]),
                "first_pass_feedback": feedback,
                "guided_center_knots": anchors,
                "critic_gradient_mean": gradient_mean,
                "critic_gradient_std": gradient_std,
            }
            episode_dir = staging / source_path.parents[1].name
            episode_dir.mkdir(exist_ok=True)
            np.savez_compressed(episode_dir / source_path.name, **arrays)
            first_base_cost.extend(arrays["first_pass_cost"][:, 0].tolist())
            first_best_cost.extend(arrays["first_pass_cost"].min(axis=1).tolist())

            if args.reference_feedback is not None:
                reference_path = args.reference_feedback / source_path.parents[1].name / source_path.name
                with np.load(reference_path, allow_pickle=False) as reference:
                    feedback_regression.append(float(np.max(np.abs(
                        feedback - np.asarray(reference["first_pass_feedback"], np.float32)
                    ))))
                    anchor_regression.append(float(np.max(np.abs(
                        anchors - np.asarray(reference["guided_center_knots"], np.float32)
                    ))))
            if args.reference_risk is not None:
                risk_path = args.reference_risk / source_path.parents[1].name / source_path.name
                with np.load(risk_path, allow_pickle=False) as risk:
                    gradient_mean_regression.append(float(np.max(np.abs(
                        gradient_mean - np.asarray(risk["critic_gradient_mean"], np.float32)
                    ))))
                    gradient_std_regression.append(float(np.max(np.abs(
                        gradient_std - np.asarray(risk["critic_gradient_std"], np.float32)
                    ))))
            if index % 20 == 0 or index == len(paths):
                print(
                    f"[{index:04d}/{len(paths):04d}] first-pass contexts "
                    f"elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )
        shutil.copy2(split_path, staging / "splits.json")
        shutil.copy2(args.config.resolve(), staging / "teacher_config.json")
        summary = {
            "format_version": FORMAT_VERSION,
            "generator_id": GENERATOR_ID,
            "semantics": "train-only frozen-policy and forward-rollout Actor inputs; no DBM analytic gradient",
            "source": str(source_root),
            "snapshot_count": len(paths),
            "contexts_per_snapshot": len(seeds),
            "first_seeds": seeds,
            "first_samples": args.first_samples,
            "fit_ridge": args.fit_ridge,
            "step_damping": args.step_damping,
            "max_standardized_step": args.max_standardized_step,
            "actor_checkpoint": str(args.actor_checkpoint.resolve()),
            "actor_checkpoint_sha256": sha256_file(args.actor_checkpoint.resolve()),
            "critic_dir": str(args.critic_dir.resolve()),
            "critic_checkpoints": critic_paths,
            "critic_checkpoint_sha256": [sha256_file(Path(path)) for path in critic_paths],
            "first_base_cost_mean": float(np.mean(first_base_cost)),
            "first_best_cost_mean": float(np.mean(first_best_cost)),
            "reference_feedback_max_abs_error": (
                float(np.max(feedback_regression)) if feedback_regression else None
            ),
            "reference_anchor_max_abs_error": (
                float(np.max(anchor_regression)) if anchor_regression else None
            ),
            "reference_gradient_mean_max_abs_error": (
                float(np.max(gradient_mean_regression)) if gradient_mean_regression else None
            ),
            "reference_gradient_std_max_abs_error": (
                float(np.max(gradient_std_regression)) if gradient_std_regression else None
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "repository": repository_state(Path(__file__).resolve().parents[2]),
            "test_policy": "validation and test splits not loaded or evaluated",
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": "ok", **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
