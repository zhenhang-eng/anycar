#!/usr/bin/env python3
"""Independently validate FR-TRPI hashes, centers, safe labels, and DBM costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TorchMPPIDeterministicCenterActor,
    ego_reference_features,
)
from generate_dbm_direct_trust_region_labels import (
    DEFAULT_OUTPUT,
    NEW_CONTEXT,
    NEW_SOURCE,
    OLD_ACTOR,
    OLD_CONTEXT,
    OLD_RISK,
    OLD_SOURCE,
    PROPOSAL_ACTOR,
)
from generate_dbm_multicenter_teacher import evaluate_knots, make_controller
from generate_dbm_proposal_teacher import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--old-source", type=Path, default=OLD_SOURCE)
    parser.add_argument("--old-context", type=Path, default=OLD_CONTEXT)
    parser.add_argument("--old-risk", type=Path, default=OLD_RISK)
    parser.add_argument("--new-source", type=Path, default=NEW_SOURCE)
    parser.add_argument("--new-context", type=Path, default=NEW_CONTEXT)
    parser.add_argument("--old-actor", type=Path, default=OLD_ACTOR)
    parser.add_argument("--proposal-actor", type=Path, default=PROPOSAL_ACTOR)
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_actor(path: Path, device: torch.device) -> tuple[dict[str, Any], TorchMPPIDeterministicCenterActor]:
    payload = torch.load(path, map_location="cpu")
    actor = TorchMPPIDeterministicCenterActor(
        maximum_delta_sigma=float(payload["maximum_delta_sigma"]), dropout=0.0
    ).to(device)
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    actor.eval()
    return payload, actor


def inputs(
    source: np.lib.npyio.NpzFile,
    context: np.lib.npyio.NpzFile,
    gradient_mean: np.ndarray,
    gradient_std: np.ndarray,
    repeat: int,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    state = np.asarray(source["initial_state"], np.float32)
    action = np.asarray(source["current_action"], np.float32)
    history = np.asarray(source["history"][0], np.float32)[None]
    reference = ego_reference_features(source["reference_ego"], float(state[3]))[None]
    current = np.asarray((state[3], state[4], *action), np.float32)[None]
    normalization = MPPIProposalNormalization.from_dict(checkpoint["state_normalization"])
    history, reference, current = normalization.normalize_numpy(history, reference, current)
    anchor = np.asarray(context["guided_center_knots"][repeat], np.float32)[None]
    feedback = np.asarray(context["first_pass_feedback"][repeat], np.float32)
    feedback = (feedback - checkpoint["feedback_mean"]) / checkpoint["feedback_std"]
    gradient = np.concatenate((gradient_mean[repeat], gradient_std[repeat]))
    gradient = (gradient - checkpoint["gradient_mean"]) / checkpoint["gradient_std"]
    return tuple(
        torch.from_numpy(np.asarray(value, np.float32)).to(device)
        for value in (history, reference, current, anchor, feedback[None], gradient[None])
    )


def independently_reconstruct_line(
    old: np.ndarray,
    proposal: np.ndarray,
    sigma: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    radius: float,
    alphas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
    direction = (proposal - old) / sigma[None]
    rho = float(np.sqrt(np.mean(direction * direction)))
    scale = min(1.0, radius / (rho + 1e-8))
    projected = direction * scale
    raw = old[None] + alphas[:, None, None] * sigma[None, None] * projected[None]
    centers = np.minimum(np.maximum(raw, low), high).astype(np.float32)
    realized = (centers - old[None]) / sigma[None, None]
    realized_rho = np.sqrt(np.mean(realized * realized, axis=(1, 2))).astype(np.float32)
    return direction.astype(np.float32), raw.astype(np.float32), centers, rho, scale, realized_rho


def independently_select_safe(
    cost: np.ndarray,
    minimum_improvement: float,
    tie_absolute: float,
    tie_fraction: float,
) -> tuple[int, int, float]:
    best = int(np.argmin(cost))
    gain = float(cost[0] - cost[best])
    tolerance = max(tie_absolute, tie_fraction * max(gain, 0.0))
    if gain < minimum_improvement:
        return best, 0, tolerance
    allowed = [
        index for index, value in enumerate(cost)
        if value <= cost[best] + tolerance and value < cost[0] - minimum_improvement
    ]
    return best, (allowed[0] if allowed else 0), tolerance


def update_max(current: float, *values: np.ndarray) -> float:
    return max(current, *(float(np.max(np.abs(value))) for value in values))


def main() -> None:
    args = parse_args()
    summary = json.loads((args.labels / "summary.json").read_text())
    config = json.loads((args.labels / "config.json").read_text())
    splits = json.loads((args.labels / "splits.json").read_text())
    if sha256_file(args.old_actor) != config["old_actor_sha256"]:
        raise AssertionError("old Actor hash mismatch")
    if sha256_file(args.proposal_actor) != config["proposal_actor_sha256"]:
        raise AssertionError("proposal Actor hash mismatch")
    sealed = set(splits["formal_validation_sealed_not_generated"]) | set(
        splits["test_sealed_not_generated"]
    )
    present = {path.name for path in args.labels.glob("episode_*") if path.is_dir()}
    if present & sealed:
        raise AssertionError("held-out episode labels are present")
    if not present <= set(splits["train"]):
        raise AssertionError("unexpected episode directory")

    device = torch.device(args.device)
    old_payload, old_actor = load_actor(args.old_actor, device)
    proposal_payload, proposal_actor = load_actor(args.proposal_actor, device)
    paths = sorted(args.labels.glob("episode_*/*.npz"))
    if args.max_snapshots:
        paths = paths[: args.max_snapshots]
    if not paths:
        raise ValueError("no label records")
    max_center_error = 0.0
    max_cost_error = 0.0
    max_metadata_error = 0.0
    context_count = 0
    safe_zero = 0
    argmin_zero = 0
    for index, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=False) as label:
            source_kind = str(label["source_kind"])
            episode = path.parent.name
            if source_kind == "diverse":
                source_path = args.old_source / episode / "snapshots" / path.name
                context_path = args.old_context / episode / path.name
                risk_path = args.old_risk / episode / path.name
            elif source_kind == "expansion":
                source_path = args.new_source / episode / "snapshots" / path.name
                context_path = args.new_context / episode / path.name
                risk_path = None
            else:
                raise AssertionError(f"{path}: unknown source kind")
            if str(label["source_snapshot_sha256"]) != sha256_file(source_path):
                raise AssertionError(f"{path}: source hash mismatch")
            if str(label["context_label_sha256"]) != sha256_file(context_path):
                raise AssertionError(f"{path}: context hash mismatch")
            if risk_path is not None and str(label["risk_label_sha256"]) != sha256_file(risk_path):
                raise AssertionError(f"{path}: risk hash mismatch")
            if str(label["old_actor_checkpoint_sha256"]) != config["old_actor_sha256"]:
                raise AssertionError(f"{path}: old Actor record hash mismatch")
            if str(label["proposal_actor_checkpoint_sha256"]) != config["proposal_actor_sha256"]:
                raise AssertionError(f"{path}: proposal Actor record hash mismatch")
            with np.load(source_path, allow_pickle=False) as source, np.load(
                context_path, allow_pickle=False
            ) as context:
                if risk_path is None:
                    gradient_mean = np.asarray(context["critic_gradient_mean"], np.float32)
                    gradient_std = np.asarray(context["critic_gradient_std"], np.float32)
                else:
                    with np.load(risk_path, allow_pickle=False) as risk:
                        gradient_mean = np.asarray(risk["critic_gradient_mean"], np.float32)
                        gradient_std = np.asarray(risk["critic_gradient_std"], np.float32)
                params = json.loads(str(source["mppi_params_json"]))
                sigma = np.asarray(params["noise_sigma"], np.float32)
                low = np.asarray(params["action_min"], np.float32)
                high = np.asarray(params["action_max"], np.float32)
                alphas = np.asarray(label["alpha_grid"], np.float32)
                radius = float(label["trust_radius_sigma_rms"])
                if not np.array_equal(alphas, np.asarray(config["alpha_grid"], np.float32)):
                    raise AssertionError(f"{path}: alpha grid mismatch")
                objective = {"objective": {"cost_weights": json.loads(str(source["cost_weights_json"]))}}
                controller, backend = make_controller(source, objective, device)
                history = torch.from_numpy(source["history"]).to(device)
                initial = torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5)
                current_action = torch.from_numpy(source["current_action"]).to(device).reshape(1, 2)
                reference = controller._prepare_reference(source["reference"])
                repeat_count = len(context["guided_center_knots"])
                if label["direct_cost"].shape != (repeat_count, len(alphas)):
                    raise AssertionError(f"{path}: cost shape mismatch")
                for repeat in range(repeat_count):
                    old_input = inputs(
                        source, context, gradient_mean, gradient_std,
                        repeat, old_payload, device,
                    )
                    proposal_input = inputs(
                        source, context, gradient_mean, gradient_std,
                        repeat, proposal_payload, device,
                    )
                    with torch.no_grad():
                        old_action_t, old_center_t = old_actor(*old_input)
                        proposal_action_t, proposal_center_t = proposal_actor(*proposal_input)
                    old_center = old_center_t[0].cpu().numpy().astype(np.float32)
                    proposal_center = proposal_center_t[0].cpu().numpy().astype(np.float32)
                    direction, raw, centers, rho, scale, realized_rho = independently_reconstruct_line(
                        old_center, proposal_center, sigma, low, high, radius, alphas
                    )
                    max_center_error = update_max(
                        max_center_error,
                        old_center - label["old_actor_center"][repeat],
                        proposal_center - label["proposal_actor_center"][repeat],
                        old_action_t[0].cpu().numpy() - label["old_actor_action"][repeat],
                        proposal_action_t[0].cpu().numpy() - label["proposal_actor_action"][repeat],
                        direction - label["requested_normalized_direction"][repeat],
                        raw - label["raw_line_centers"][repeat],
                        centers - label["line_centers"][repeat],
                        realized_rho - label["realized_rho"][repeat],
                    )
                    max_metadata_error = max(
                        max_metadata_error,
                        abs(rho - float(label["requested_rho"][repeat])),
                        abs(scale - float(label["trust_scale"][repeat])),
                    )
                    cost, _, _ = evaluate_knots(
                        controller, backend, centers, history, initial,
                        current_action, reference,
                    )
                    max_cost_error = update_max(
                        max_cost_error,
                        cost - label["direct_cost"][repeat],
                        (cost[0] - cost) - label["advantage_vs_old"][repeat],
                    )
                    best, safe, tolerance = independently_select_safe(
                        cost,
                        float(label["minimum_improvement"]),
                        float(label["tie_absolute"]),
                        float(label["tie_fraction"]),
                    )
                    if best != int(label["argmin_index"][repeat]):
                        raise AssertionError(f"{path}: argmin mismatch")
                    if safe != int(label["safe_index"][repeat]):
                        raise AssertionError(f"{path}: safe index mismatch")
                    max_metadata_error = max(
                        max_metadata_error,
                        abs(tolerance - float(label["safe_tolerance"][repeat])),
                    )
                    max_center_error = update_max(
                        max_center_error, centers[safe] - label["safe_center"][repeat]
                    )
                    safe_zero += int(safe == 0)
                    argmin_zero += int(best == 0)
                    context_count += 1
            if not all(np.all(np.isfinite(label[key])) for key in (
                "old_actor_center", "proposal_actor_center", "line_centers",
                "direct_cost", "advantage_vs_old", "safe_center",
            )):
                raise AssertionError(f"{path}: non-finite label")
        if index == 1 or index % 50 == 0 or index == len(paths):
            print(
                f"[{index:04d}/{len(paths):04d}] contexts={context_count} "
                f"center_err={max_center_error:.3g} cost_err={max_cost_error:.3g}",
                flush=True,
            )

    if not args.max_snapshots:
        if len(paths) != int(summary["snapshot_count"]):
            raise AssertionError("snapshot count mismatch")
        if context_count != int(summary["context_count"]):
            raise AssertionError("context count mismatch")
    if max_center_error > 1e-6 or max_cost_error > 1e-5 or max_metadata_error > 1e-6:
        raise AssertionError("validation tolerance exceeded")
    result = {
        "validated_snapshots": len(paths),
        "validated_contexts": context_count,
        "maximum_center_error": max_center_error,
        "maximum_cost_error": max_cost_error,
        "maximum_metadata_error": max_metadata_error,
        "argmin_alpha_zero_fraction": argmin_zero / context_count,
        "safe_alpha_zero_fraction": safe_zero / context_count,
        "sealed_episode_count": len(sealed),
        "qualification": "TR1_VALIDATED" if not args.max_snapshots else "SMOKE_VALIDATED",
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
