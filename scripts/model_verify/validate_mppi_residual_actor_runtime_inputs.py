#!/usr/bin/env python3
"""Rebuild the residual Actor's online context from raw fixed-DBM snapshots.

This is the gate between cached fixed-state evaluation and closed-loop use.  It
re-runs one or both frozen 128-rollout first-pass seeds, reconstructs the guided
anchor, 74-D feedback and feedback-Critic gradient context, then executes the
complete old/proposal/Alpha/residual Actor chain.  Cached label fields are used
only as validation targets, never as runtime inputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TorchMPPIDeterministicCenterActor,
    ego_reference_features,
)
from evaluate_mppi_proposal_bc import load_policy, predict_center
from generate_dbm_direct_trust_region_labels import line_centers
from generate_dbm_multicenter_teacher import load_config, make_controller
from generate_dbm_proposal_teacher import repository_state, sha256_file
from generate_mppi_first_pass_actor_contexts import first_pass
from generate_dbm_two_pass_risk_replay_labels import (
    critic_directions,
    load_critic_ensemble,
)
from train_mppi_direct_residual_online_ac import make_base_policy
from train_mppi_direct_trust_region_actor import load_actor_payload, make_actor


DEFAULT_CHECKPOINT = Path(
    "outputs/mppi_proposal/direct_residual_online_ac_20260811_v2/"
    "direct_residual_online_ac_selected.pt"
)
DEFAULT_REFERENCE = Path(
    "outputs/mppi_proposal/two_center_integration_20260813_v2/evaluation.npz"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/residual_actor_runtime_input_validation_20260813_v2"
)
DEFAULT_BC = Path(
    "outputs/mppi_proposal/bc_t1_conv_diverse_20260805_v1/trust1_seed1.pt"
)
DEFAULT_FEEDBACK_CRITIC = Path(
    "outputs/mppi_proposal/critic_two_pass_feedback_20260805_v1"
)
DEFAULT_CONFIG = Path(__file__).with_name("dbm_teacher_t1_diverse_20260805_v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC)
    parser.add_argument("--feedback-critic-dir", type=Path, default=DEFAULT_FEEDBACK_CRITIC)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--first-seeds", type=int, nargs="+", default=(24001, 24002))
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def normalized_inputs(
    source: np.lib.npyio.NpzFile,
    anchor: np.ndarray,
    feedback: np.ndarray,
    gradient_mean: np.ndarray,
    gradient_std: np.ndarray,
    payload: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    state = np.asarray(source["initial_state"], np.float32)
    action = np.asarray(source["current_action"], np.float32)
    history = np.asarray(source["history"][0], np.float32)[None]
    reference = ego_reference_features(
        source["reference_ego"], float(state[3])
    )[None]
    current = np.asarray((state[3], state[4], *action), np.float32)[None]
    normalization = MPPIProposalNormalization.from_dict(payload["state_normalization"])
    history, reference, current = normalization.normalize_numpy(
        history, reference, current
    )
    feedback_value = (
        np.asarray(feedback, np.float32) - np.asarray(payload["feedback_mean"], np.float32)
    ) / np.asarray(payload["feedback_std"], np.float32)
    gradient = np.concatenate((gradient_mean, gradient_std)).astype(np.float32)
    gradient = (
        gradient - np.asarray(payload["gradient_mean"], np.float32)
    ) / np.asarray(payload["gradient_std"], np.float32)
    arrays = (history[0], reference[0], current[0], anchor, feedback_value, gradient)
    return tuple(torch.from_numpy(np.asarray(value, np.float32)[None]).to(device) for value in arrays)


@torch.no_grad()
def actor_chain(
    inputs: tuple[torch.Tensor, ...],
    sigma: np.ndarray,
    trust_radius: float,
    old_actor: TorchMPPIDeterministicCenterActor,
    proposal_actor: TorchMPPIDeterministicCenterActor,
    base_policy: torch.nn.Module,
    base_threshold: float,
    residual_actor: TorchMPPIDeterministicCenterActor,
) -> dict[str, np.ndarray | float]:
    _, old_center_t = old_actor(*inputs)
    _, proposal_center_t = proposal_actor(*inputs)
    old_center = old_center_t[0].cpu().numpy()
    proposal_center = proposal_center_t[0].cpu().numpy()
    line = line_centers(
        old_center, proposal_center, sigma,
        np.full(2, -1.0, np.float32), np.full(2, 1.0, np.float32),
        trust_radius, np.asarray((0.0, 1.0), np.float32),
    )
    direction = torch.from_numpy(
        np.asarray(line["projected_direction"], np.float32)[None]
    ).to(old_center_t.device)
    rho = torch.tensor([[float(line["requested_rho"])]], device=old_center_t.device)
    scale = torch.tensor([[float(line["trust_scale"])]], device=old_center_t.device)
    _, probability, mean, _ = base_policy(*inputs, direction, rho, scale)
    alpha_t = base_policy.deterministic_alpha(probability, mean, base_threshold)
    sigma_t = torch.from_numpy(sigma).to(old_center_t.device).reshape(1, 1, 2)
    base_center_t = torch.clamp(
        old_center_t + alpha_t[:, None, None] * sigma_t * direction,
        -1.0, 1.0,
    )
    residual_inputs = list(inputs)
    residual_inputs[3] = base_center_t
    _, residual_center_t = residual_actor(*residual_inputs)
    return {
        "old_center": old_center,
        "proposal_center": proposal_center,
        "requested_rho": float(line["requested_rho"]),
        "trust_scale": float(line["trust_scale"]),
        "projected_direction": np.asarray(line["projected_direction"], np.float32),
        "move_probability": float(probability[0]),
        "base_alpha": float(alpha_t[0]),
        "base_center": base_center_t[0].cpu().numpy(),
        "residual_center": residual_center_t[0].cpu().numpy(),
    }


def maximum_error(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.first_seeds or len(set(args.first_seeds)) != len(args.first_seeds):
        raise ValueError("first-pass seeds must be nonempty and unique")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    started = time.perf_counter()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    base_path = Path(checkpoint["base_alpha_checkpoint"])
    base_payload = torch.load(base_path, map_location="cpu")
    old_payload = load_actor_payload(Path(base_payload["old_actor"]))
    proposal_payload = torch.load(Path(base_payload["proposal_actor"]), map_location="cpu")
    if "actor_state_dict" not in proposal_payload or "maximum_delta_sigma" not in proposal_payload:
        raise AssertionError("proposal checkpoint is not a deterministic center Actor")
    labels = Path(checkpoint["labels"])
    split = json.loads((labels / "splits.json").read_text())
    selection = set(split["internal_selection"])
    records = []
    for label_path in sorted(labels.glob("episode_*/*.npz")):
        if label_path.parent.name not in selection:
            continue
        with np.load(label_path, allow_pickle=False) as label:
            records.append((
                label_path,
                Path(str(label["source_snapshot"])),
                Path(str(label["context_label"])),
            ))
    if args.max_snapshots:
        records = records[:args.max_snapshots]
    if not records:
        raise ValueError("no internal-selection snapshots selected")

    reference: dict[tuple[str, int], np.ndarray] = {}
    with np.load(args.reference, allow_pickle=False) as values:
        for source, repeat, center in zip(
            values["source_path"], values["context_in_file"], values["actor_center"]
        ):
            reference[(str(source), int(repeat))] = np.asarray(center, np.float32)

    bc_actor, bc_normalization, _ = load_policy(args.bc_checkpoint, device)
    critics, critic_normalization, feedback_mean, feedback_std, critic_paths = (
        load_critic_ensemble(args.feedback_critic_dir, device)
    )
    config = load_config(args.config)
    old_actor = make_actor(old_payload, device, dropout=0.0).eval()
    proposal_actor = make_actor(proposal_payload, device, dropout=0.0).eval()
    base_policy = make_base_policy(base_payload, device)
    residual_actor = TorchMPPIDeterministicCenterActor(
        float(checkpoint["maximum_residual_sigma"]), dropout=0.0
    ).to(device)
    residual_actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    residual_actor.eval()
    trust_radius = float(json.loads((labels / "config.json").read_text())["trust_radius_sigma_rms"])
    runtime_args = SimpleNamespace(
        first_samples=128,
        fit_ridge=0.10,
        step_damping=0.10,
        max_standardized_step=1.0,
    )

    errors: dict[str, list[float]] = {name: [] for name in (
        "base_center", "first_knots", "first_cost", "guided_center", "feedback",
        "gradient_mean", "gradient_std", "old_center", "proposal_center",
        "residual_center",
    )}
    rows = []
    for record_index, (label_path, source_path, context_path) in enumerate(records, 1):
        with np.load(source_path, allow_pickle=False) as source, np.load(
            context_path, allow_pickle=False
        ) as cached, np.load(label_path, allow_pickle=False) as label:
            cached_seeds = np.asarray(cached["first_pass_seed"], np.int64).tolist()
            requested = [seed for seed in args.first_seeds if seed in cached_seeds]
            if requested != list(args.first_seeds):
                raise AssertionError(f"{context_path}: requested seed missing")
            _, base = predict_center(bc_actor, bc_normalization, source, device)
            errors["base_center"].append(float(np.max(np.abs(
                base - np.asarray(cached["base_center_knots"], np.float32)
            ))))
            controller, backend = make_controller(source, config, device)
            regenerated = [
                first_pass(base, seed, source, controller, backend, runtime_args)
                for seed in requested
            ]
            anchors = np.stack([value["guided_center_knots"] for value in regenerated])
            feedback = np.stack([value["first_pass_feedback"] for value in regenerated])
            _, gradient_mean, gradient_std = critic_directions(
                critics, critic_normalization, feedback_mean, feedback_std,
                source, anchors, feedback, device,
            )
            for local, seed in enumerate(requested):
                cached_index = cached_seeds.index(seed)
                fields = {
                    "first_knots": (regenerated[local]["first_pass_knots"], cached["first_pass_knots"][cached_index]),
                    "first_cost": (regenerated[local]["first_pass_cost"], cached["first_pass_cost"][cached_index]),
                    "guided_center": (anchors[local], cached["guided_center_knots"][cached_index]),
                    "feedback": (feedback[local], cached["first_pass_feedback"][cached_index]),
                    "gradient_mean": (gradient_mean[local], cached["critic_gradient_mean"][cached_index]),
                    "gradient_std": (gradient_std[local], cached["critic_gradient_std"][cached_index]),
                }
                for name, (actual, expected) in fields.items():
                    errors[name].append(float(np.max(np.abs(
                        np.asarray(actual) - np.asarray(expected)
                    ))))
                inputs = normalized_inputs(
                    source, anchors[local], feedback[local], gradient_mean[local],
                    gradient_std[local], old_payload, device,
                )
                sigma = np.asarray(json.loads(str(source["mppi_params_json"]))["noise_sigma"], np.float32)
                output = actor_chain(
                    inputs, sigma, trust_radius, old_actor, proposal_actor,
                    base_policy, float(base_payload["move_threshold"]), residual_actor,
                )
                errors["old_center"].append(float(np.max(np.abs(
                    output["old_center"] - np.asarray(label["old_actor_center"][cached_index], np.float32)
                ))))
                errors["proposal_center"].append(float(np.max(np.abs(
                    output["proposal_center"] - np.asarray(label["proposal_actor_center"][cached_index], np.float32)
                ))))
                target = reference[(str(source_path), cached_index)]
                center_error = float(np.max(np.abs(output["residual_center"] - target)))
                errors["residual_center"].append(center_error)
                rows.append({
                    "episode": label_path.parent.name,
                    "snapshot": label_path.name,
                    "first_seed": seed,
                    "context_in_file": cached_index,
                    "residual_center_max_abs_error": center_error,
                    "base_alpha": output["base_alpha"],
                    "move_probability": output["move_probability"],
                })
        if record_index % 25 == 0 or record_index == len(records):
            print(
                f"[{record_index:03d}/{len(records):03d}] "
                f"residual_center_error={maximum_error(errors['residual_center']):.3e}",
                flush=True,
            )

    maxima = {name: maximum_error(value) for name, value in errors.items()}
    tolerance = {
        "first_cost": 5e-4,
        "feedback": 5e-4,
        "gradient_mean": 5e-4,
        "gradient_std": 5e-4,
        "default": 5e-5,
    }
    failures = {
        name: value for name, value in maxima.items()
        if value > tolerance.get(name, tolerance["default"])
    }
    summary = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "qualification": "PASS_RUNTIME_INPUT_RECONSTRUCTION" if not failures else "FAIL_RUNTIME_INPUT_RECONSTRUCTION",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "partition": "consumed internal_selection mechanism diagnostic",
        "snapshot_count": len(records),
        "context_count": len(rows),
        "first_pass_seeds": list(args.first_seeds),
        "first_pass_candidate_rollouts_per_runtime_decision": 128,
        "first_pass_weighted_output_rollouts_per_runtime_decision": 1,
        "strict_guard_rollout_budget": {
            "actor_context_first_pass_candidates": 128,
            "actor_context_first_pass_weighted_output": 1,
            "warm_mppi": 256,
            "warm_weighted_output_cost": 1,
            "actor_direct_cost": 1,
            "total": 387,
        },
        "maximum_absolute_error": maxima,
        "tolerance": tolerance,
        "failures": failures,
        "feedback_critic_checkpoints": critic_paths,
        "elapsed_seconds": time.perf_counter() - started,
        "repository_state": repository_state(Path.cwd()),
        "caveats": [
            "This validates cached-state online-input reconstruction, not closed-loop behavior.",
            "The 258 count applies only after an Actor center already exists.",
            "The 74-D feedback includes one separately rolled-out first-pass weighted-output cost.",
            "Reusing warm-MPPI candidates as Actor feedback is not yet qualified.",
            "Formal validation/test episodes are not loaded.",
        ],
        "test_policy": "formal validation and test remain sealed",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
