#!/usr/bin/env python3
"""Evaluate forward-only dynamic 33-slot center generators on frozen DBM states."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from generate_dbm_multicenter_teacher import make_controller
from generate_dbm_sampling_center_gt_pilot import evaluate_centers
from generate_dbm_two_pass_multidirection_replay import (
    DIRECTION_NAMES,
    direction_bank as current_direction_bank,
    normalize_direction,
)
from guide_mppi_sampling_from_trajectory_error import evaluate_knots


DEFAULT_PARENT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_feedback_diverse_20260805_v1"
)
DEFAULT_RISK = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_risk_replay_diverse_20260805_v1"
)
DEFAULT_CURRENT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_multidirection_replay_diverse_20260805_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/dbm_dynamic_generator_oracle_20260806_v1"
)
BANK_NAMES = (
    "B0_current",
    "B1_segmented_fixed",
    "B2_current_adaptive",
    "B3_combined_adaptive",
    "B4_first_pass_elites",
    "B5_hybrid_current_elites",
    "B6_residual_response",
    "B7_hybrid_current_response",
)
SEGMENT_DIRECTION_NAMES = (
    "total",
    "early_tracking",
    "middle_tracking",
    "late_tracking",
    "position",
    "yaw",
    "vx",
    "action_rate",
)
CURRENT_RADII = np.asarray((0.03, 0.06, 0.10, 0.15), np.float32)
SEGMENT_RADII = np.asarray((0.15, 0.35, 0.65, 1.00), np.float32)
RADIUS_FRACTIONS = np.asarray((0.25, 0.50, 0.75, 1.00), np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("center_result_dirs", type=Path, nargs="+")
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--risk-labels", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--current-labels", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-seeds", type=int, nargs="+", default=(29511, 29512))
    parser.add_argument("--audit-seeds", type=int, nargs="+", default=(29521, 29522, 29523, 29524))
    parser.add_argument("--candidate-noise-scale", type=float, default=0.10)
    parser.add_argument("--fit-ridge", type=float, default=0.10)
    parser.add_argument("--step-damping", type=float, default=0.10)
    parser.add_argument("--minimum-step", type=float, default=0.20)
    parser.add_argument("--maximum-step", type=float, default=1.50)
    parser.add_argument("--response-maximum-step", type=float, default=2.00)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def trajectory_objectives(
    controller: Any,
    trajectories: torch.Tensor,
    actions: torch.Tensor,
    reference: torch.Tensor,
    current_action: torch.Tensor,
) -> np.ndarray:
    weights = controller.cost_weights
    position = weights.position * (
        trajectories[..., :2] - reference[None, :, :2]
    ).square().sum(dim=-1)
    yaw = weights.yaw * controller._wrapped_angle_difference(
        trajectories[..., 2], reference[None, :, 2]
    ).square()
    vx = weights.vx * (trajectories[..., 3] - reference[None, :, 3]).square()
    previous = torch.cat(
        (current_action.expand(actions.shape[0], 1, -1), actions[:, :-1]), dim=1
    )
    rate = actions - previous
    rate = (
        weights.acceleration_rate * rate[..., 0].square()
        + weights.steering_rate * rate[..., 1].square()
    )
    tracking = position + yaw + vx
    objectives = torch.stack(
        (
            (tracking + rate).sum(dim=1),
            tracking[:, :17].sum(dim=1),
            tracking[:, 17:34].sum(dim=1),
            tracking[:, 34:].sum(dim=1),
            position.sum(dim=1),
            yaw.sum(dim=1),
            vx.sum(dim=1),
            rate.sum(dim=1),
        ),
        dim=1,
    )
    return objectives.detach().cpu().numpy().astype(np.float64)


def fit_segment_directions(
    first_knots: np.ndarray,
    objectives: np.ndarray,
    base: np.ndarray,
    sigma: np.ndarray,
    ridge: float,
    fallback: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    x = ((first_knots - base) / sigma.reshape(1, 1, 2)).reshape(len(first_knots), -1)
    y = objectives - objectives[0:1]
    total = objectives[:, 0]
    scale = max(float(np.quantile(total, 0.50) - total.min()), 1e-3)
    weight = np.exp(-(total - total.min()) / scale)
    sqrt_weight = np.sqrt(weight / max(weight.mean(), 1e-12))[:, None]
    xw = x * sqrt_weight
    matrix = xw.T @ xw + float(ridge) * np.eye(x.shape[1])
    gradients = []
    directions = []
    for index in range(objectives.shape[1]):
        gradient = np.linalg.solve(matrix, xw.T @ (y[:, index : index + 1] * sqrt_weight))[:, 0]
        gradients.append(gradient)
        directions.append(normalize_direction(-gradient, fallback))
    return (
        np.asarray(directions, np.float32),
        np.asarray(gradients, np.float32),
        float(np.linalg.cond(matrix)),
    )


def orient_and_adapt(
    directions: np.ndarray,
    gradient: np.ndarray,
    hessian_diagonal: np.ndarray,
    damping: float,
    minimum_step: float,
    maximum_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oriented = []
    radii = []
    scores = []
    positive_hessian = np.maximum(hessian_diagonal, 0.0)
    for value in directions:
        flat = np.asarray(value, np.float32).reshape(-1)
        if float(gradient @ flat) > 0:
            flat = -flat
        slope = float(gradient @ flat)
        curvature = float(np.sum(positive_hessian * flat * flat) + damping)
        step = float(np.clip(-slope / curvature, minimum_step, maximum_step))
        one_radii = step * RADIUS_FRACTIONS
        predicted = slope * step + 0.5 * curvature * step * step
        oriented.append(flat.reshape(8, 2))
        radii.append(one_radii)
        scores.append(-predicted)
    return (
        np.asarray(oriented, np.float32),
        np.asarray(radii, np.float32),
        np.asarray(scores, np.float32),
    )


def select_combined_directions(
    names: list[str],
    directions: np.ndarray,
    radii: np.ndarray,
    scores: np.ndarray,
    count: int = 8,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    selected: list[int] = []
    normalized = directions.reshape(len(directions), -1).astype(np.float64)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True).clip(1e-12)
    for index in np.argsort(-scores):
        if all(float(normalized[index] @ normalized[old]) < 0.97 for old in selected):
            selected.append(int(index))
        if len(selected) == count:
            break
    if len(selected) < count:
        for index in np.argsort(-scores):
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) == count:
                break
    return [names[index] for index in selected], directions[selected], radii[selected]


def make_bank(
    anchor: np.ndarray,
    directions: np.ndarray,
    radii: np.ndarray,
    sigma: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> np.ndarray:
    if radii.ndim == 1:
        radii = np.repeat(radii[None], len(directions), axis=0)
    centers = [anchor]
    for direction, one_radii in zip(directions, radii):
        for radius in one_radii:
            centers.append(anchor + float(radius) * direction * sigma.reshape(1, 2))
    if len(centers) != 33:
        raise AssertionError("every dynamic bank must retain exactly 33 slots")
    return np.clip(np.asarray(centers, np.float32), action_min, action_max)


def select_diverse_elite_directions(
    first_knots: np.ndarray,
    first_cost: np.ndarray,
    anchor: np.ndarray,
    sigma: np.ndarray,
    count: int,
) -> tuple[list[str], np.ndarray]:
    deltas = (first_knots - anchor) / sigma.reshape(1, 1, 2)
    flat = deltas.reshape(len(deltas), -1).astype(np.float64)
    norm = np.linalg.norm(flat, axis=1)
    unit = flat / norm[:, None].clip(1e-12)
    selected: list[int] = []
    for index in np.argsort(first_cost):
        if norm[index] < 0.10:
            continue
        if all(float(unit[index] @ unit[old]) < 0.95 for old in selected):
            selected.append(int(index))
        if len(selected) == count:
            break
    if len(selected) < count:
        for index in np.argsort(first_cost):
            if norm[index] >= 0.10 and int(index) not in selected:
                selected.append(int(index))
            if len(selected) == count:
                break
    if len(selected) != count:
        raise AssertionError("first pass did not provide enough elite directions")
    return [f"first_elite_{index:03d}" for index in selected], deltas[selected].astype(np.float32)


def fit_residual_response_directions(
    first_knots: np.ndarray,
    residuals: np.ndarray,
    first_cost: np.ndarray,
    base: np.ndarray,
    anchor: np.ndarray,
    sigma: np.ndarray,
    ridge: float,
    damping: float,
    maximum_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = ((first_knots - base) / sigma.reshape(1, 1, 2)).reshape(len(first_knots), -1)
    residual = residuals.reshape(len(first_knots), 50, -1).astype(np.float64)
    if residual.shape[2] != 6:
        raise ValueError("expected six weighted residual fields per horizon step")
    scale = max(float(np.quantile(first_cost, 0.50) - first_cost.min()), 1e-3)
    weight = np.exp(-(first_cost - first_cost.min()) / scale)
    sqrt_weight = np.sqrt(weight / max(weight.mean(), 1e-12))[:, None]
    xw = x * sqrt_weight
    normal = xw.T @ xw + float(ridge) * np.eye(x.shape[1])
    identity = np.eye(x.shape[1])
    masks = (
        (slice(None), slice(None)),
        (slice(0, 17), slice(None)),
        (slice(17, 34), slice(None)),
        (slice(34, 50), slice(None)),
        (slice(None), slice(0, 2)),
        (slice(None), slice(2, 3)),
        (slice(None), slice(3, 4)),
        (slice(None), slice(4, 6)),
    )
    directions = []
    standardized_steps = []
    scores = []
    for time_mask, field_mask in masks:
        selected = residual[:, time_mask, field_mask].reshape(len(first_knots), -1)
        delta = selected - selected[0:1]
        response = np.linalg.solve(normal, xw.T @ (delta * sqrt_weight))
        baseline = selected[0]
        step = -np.linalg.solve(
            response @ response.T + float(damping) * identity,
            response @ baseline,
        )
        step = np.clip(step, -maximum_step, maximum_step)
        predicted_before = float(baseline @ baseline)
        predicted_after = float((baseline + step @ response) @ (baseline + step @ response))
        proposal = np.clip(
            base + step.reshape(8, 2) * sigma.reshape(1, 2), -1.0, 1.0
        )
        direction = (proposal - anchor) / sigma.reshape(1, 2)
        directions.append(direction)
        standardized_steps.append(step.reshape(8, 2))
        scores.append(predicted_before - predicted_after)
    return (
        np.asarray(directions, np.float32),
        np.asarray(standardized_steps, np.float32),
        np.asarray(scores, np.float32),
    )


@torch.no_grad()
def construct_banks(
    source: np.lib.npyio.NpzFile,
    parent: np.lib.npyio.NpzFile,
    risk: np.lib.npyio.NpzFile,
    current: np.lib.npyio.NpzFile,
    context: int,
    controller: Any,
    backend: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    params = json.loads(str(source["mppi_params_json"]))
    sigma = np.asarray(params["noise_sigma"], np.float32)
    action_min = np.asarray(params["action_min"], np.float32)
    action_max = np.asarray(params["action_max"], np.float32)
    base = np.asarray(parent["base_center_knots"], np.float32)
    anchor = np.asarray(parent["guided_center_knots"][context], np.float32)
    first_knots = np.asarray(parent["first_pass_knots"][context], np.float32)
    history = torch.from_numpy(source["history"]).to(device)
    initial = torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5)
    current_action = torch.from_numpy(source["current_action"]).to(device).reshape(1, 2)
    reference = controller._prepare_reference(source["reference"])
    first = evaluate_knots(
        controller, backend, torch.from_numpy(first_knots).to(device),
        history, initial, current_action, reference,
    )
    objectives = trajectory_objectives(
        controller, first.trajectories, first.actions, reference, current_action
    )
    feedback = np.asarray(parent["first_pass_feedback"][context], np.float32)
    critic = np.asarray(risk["normalized_critic_direction"][context], np.float32)
    current_directions = current_direction_bank(
        feedback[None], first_knots[None],
        np.asarray(parent["first_pass_cost"][context : context + 1], np.float32),
        base, sigma, critic[None], args.step_damping,
    )[0]
    segment_directions, segment_gradients, fit_condition = fit_segment_directions(
        first_knots, objectives, base, sigma, args.fit_ridge,
        current_directions[0],
    )
    total_gradient = np.asarray(feedback[16:32], np.float32)
    total_hessian = np.asarray(feedback[32:48], np.float32)
    current_oriented, current_radii, current_scores = orient_and_adapt(
        current_directions, total_gradient, total_hessian,
        args.step_damping, args.minimum_step, args.maximum_step,
    )
    segment_oriented, segment_radii, segment_scores = orient_and_adapt(
        segment_directions, total_gradient, total_hessian,
        args.step_damping, args.minimum_step, args.maximum_step,
    )
    combined_names, combined_directions, combined_radii = select_combined_directions(
        [f"current:{name}" for name in DIRECTION_NAMES]
        + [f"segment:{name}" for name in SEGMENT_DIRECTION_NAMES],
        np.concatenate((current_oriented, segment_oriented), axis=0),
        np.concatenate((current_radii, segment_radii), axis=0),
        np.concatenate((current_scores, segment_scores), axis=0),
    )
    elite_names, elite_directions = select_diverse_elite_directions(
        first_knots,
        np.asarray(parent["first_pass_cost"][context], np.float32),
        anchor,
        sigma,
        8,
    )
    current_top_names, current_top_directions, current_top_radii = select_combined_directions(
        list(DIRECTION_NAMES), current_oriented, current_radii, current_scores, count=4
    )
    hybrid_elite_names = elite_names[:4]
    hybrid_directions = np.concatenate(
        (current_top_directions, elite_directions[:4]), axis=0
    )
    hybrid_radii = np.concatenate(
        (
            current_top_radii,
            np.repeat(RADIUS_FRACTIONS[None], 4, axis=0),
        ),
        axis=0,
    )
    hybrid_names = [f"current:{name}" for name in current_top_names] + [
        f"elite:{name}" for name in hybrid_elite_names
    ]
    response_directions, response_steps, response_scores = fit_residual_response_directions(
        first_knots,
        first.residuals.detach().cpu().numpy(),
        np.asarray(parent["first_pass_cost"][context], np.float32),
        base,
        anchor,
        sigma,
        args.fit_ridge,
        args.step_damping,
        args.response_maximum_step,
    )
    response_radii = np.repeat(
        np.asarray((0.25, 0.50, 1.00, 1.50), np.float32)[None], 8, axis=0
    )
    response_top = np.argsort(-response_scores)[:4]
    hybrid_response_directions = np.concatenate(
        (current_top_directions, response_directions[response_top]), axis=0
    )
    hybrid_response_radii = np.concatenate(
        (current_top_radii, response_radii[response_top]), axis=0
    )
    hybrid_response_names = [f"current:{name}" for name in current_top_names] + [
        f"response:{SEGMENT_DIRECTION_NAMES[index]}" for index in response_top
    ]
    banks = np.stack(
        (
            np.asarray(current["centers"][context], np.float32),
            make_bank(anchor, segment_directions, SEGMENT_RADII, sigma, action_min, action_max),
            make_bank(anchor, current_oriented, current_radii, sigma, action_min, action_max),
            make_bank(anchor, combined_directions, combined_radii, sigma, action_min, action_max),
            make_bank(anchor, elite_directions, RADIUS_FRACTIONS, sigma, action_min, action_max),
            make_bank(anchor, hybrid_directions, hybrid_radii, sigma, action_min, action_max),
            make_bank(anchor, response_directions, response_radii, sigma, action_min, action_max),
            make_bank(
                anchor,
                hybrid_response_directions,
                hybrid_response_radii,
                sigma,
                action_min,
                action_max,
            ),
        )
    )
    directions = np.stack(
        (
            current_directions,
            segment_directions,
            current_oriented,
            combined_directions,
            elite_directions,
            hybrid_directions,
            response_directions,
            hybrid_response_directions,
        )
    )
    radii = np.stack(
        (
            np.repeat(CURRENT_RADII[None], 8, axis=0),
            np.repeat(SEGMENT_RADII[None], 8, axis=0),
            current_radii,
            combined_radii,
            np.repeat(RADIUS_FRACTIONS[None], 8, axis=0),
            hybrid_radii,
            response_radii,
            hybrid_response_radii,
        )
    )
    slot_names = np.asarray(
        [
            list(current["action_names"].astype(str)),
            ["guided_anchor"] + [f"{name}_{radius_index}" for name in SEGMENT_DIRECTION_NAMES for radius_index in range(4)],
            ["guided_anchor"] + [f"{name}_adaptive_{radius_index}" for name in DIRECTION_NAMES for radius_index in range(4)],
            ["guided_anchor"] + [f"{name}_adaptive_{radius_index}" for name in combined_names for radius_index in range(4)],
            ["guided_anchor"] + [f"{name}_fraction_{radius_index}" for name in elite_names for radius_index in range(4)],
            ["guided_anchor"] + [f"{name}_fraction_{radius_index}" for name in hybrid_names for radius_index in range(4)],
            ["guided_anchor"] + [f"response:{name}_fraction_{radius_index}" for name in SEGMENT_DIRECTION_NAMES for radius_index in range(4)],
            ["guided_anchor"] + [f"{name}_fraction_{radius_index}" for name in hybrid_response_names for radius_index in range(4)],
        ]
    )
    return {
        "banks": banks,
        "directions": directions,
        "radii": radii,
        "slot_names": slot_names,
        "segment_gradients": segment_gradients,
        "response_standardized_steps": response_steps,
        "response_scores": response_scores,
        "segment_objectives": objectives,
        "fit_condition": fit_condition,
        "combined_direction_names": np.asarray(combined_names),
        "history": history,
        "initial": initial,
        "current_action": current_action,
        "reference": reference,
        "sigma": sigma,
        "action_min": action_min,
        "action_max": action_max,
    }


def process_one(args: argparse.Namespace, result_dir: Path, device: torch.device) -> dict[str, Any]:
    with np.load(result_dir / "center_oracle.npz", allow_pickle=False) as oracle:
        source_path = Path(str(oracle["source_snapshot"]))
        context = int(oracle["context_index"])
        unrestricted_center = np.asarray(oracle["comparison_centers"][3], np.float32)
    episode = source_path.parents[1].name
    parent_path = args.parent_labels / episode / source_path.name
    risk_path = args.risk_labels / episode / source_path.name
    current_path = args.current_labels / episode / source_path.name
    with np.load(source_path, allow_pickle=False) as source, np.load(
        parent_path, allow_pickle=False
    ) as parent, np.load(risk_path, allow_pickle=False) as risk, np.load(
        current_path, allow_pickle=False
    ) as current:
        config = {"objective": {"cost_weights": json.loads(str(source["cost_weights_json"]))}}
        controller, backend = make_controller(source, config, device)
        generated = construct_banks(
            source, parent, risk, current, context, controller, backend, device, args
        )
        candidate_sigma = generated["sigma"] * float(args.candidate_noise_scale)
        selection_cost = []
        audit_cost = []
        for centers in generated["banks"]:
            selection_cost.append(
                evaluate_centers(
                    centers, list(args.selection_seeds), controller, backend,
                    generated["history"], generated["initial"],
                    generated["current_action"], generated["reference"],
                    candidate_sigma, generated["action_min"], generated["action_max"],
                )["output_cost"]
            )
            audit_cost.append(
                evaluate_centers(
                    centers, list(args.audit_seeds), controller, backend,
                    generated["history"], generated["initial"],
                    generated["current_action"], generated["reference"],
                    candidate_sigma, generated["action_min"], generated["action_max"],
                )["output_cost"]
            )
        unrestricted_audit = evaluate_centers(
            unrestricted_center[None], list(args.audit_seeds), controller, backend,
            generated["history"], generated["initial"], generated["current_action"],
            generated["reference"], candidate_sigma, generated["action_min"],
            generated["action_max"],
        )["output_cost"][0]
    selection_cost_array = np.asarray(selection_cost, np.float32)
    audit_cost_array = np.asarray(audit_cost, np.float32)
    selected_index = selection_cost_array.mean(axis=2).argmin(axis=1)
    clairvoyant_index = audit_cost_array.mean(axis=2).argmin(axis=1)
    selected_audit = audit_cost_array[
        np.arange(len(BANK_NAMES)), selected_index
    ].mean(axis=1)
    clairvoyant_audit = audit_cost_array[
        np.arange(len(BANK_NAMES)), clairvoyant_index
    ].mean(axis=1)
    baseline_gap = float(clairvoyant_audit[0] - unrestricted_audit.mean())
    recovery = (clairvoyant_audit[0] - clairvoyant_audit) / max(baseline_gap, 1e-9)
    output_path = args.output_dir / f"{episode}_{source_path.stem}.npz"
    np.savez_compressed(
        output_path,
        source_snapshot=np.asarray(str(source_path)),
        context_index=np.asarray(context),
        bank_names=np.asarray(BANK_NAMES),
        slot_names=generated["slot_names"],
        centers=generated["banks"],
        directions=generated["directions"],
        radii_sigma=generated["radii"],
        combined_direction_names=generated["combined_direction_names"],
        segment_gradients=generated["segment_gradients"],
        response_standardized_steps=generated["response_standardized_steps"],
        response_scores=generated["response_scores"],
        segment_objectives=generated["segment_objectives"],
        fit_condition=np.asarray(generated["fit_condition"]),
        selection_seeds=np.asarray(args.selection_seeds),
        audit_seeds=np.asarray(args.audit_seeds),
        candidate_noise_scale=np.asarray(args.candidate_noise_scale),
        candidate_noise_design=np.asarray("zero_extra_antithetic_pairs"),
        selection_output_cost=selection_cost_array,
        audit_output_cost=audit_cost_array,
        selected_indices=selected_index,
        clairvoyant_indices=clairvoyant_index,
        selected_audit_cost=selected_audit,
        clairvoyant_audit_cost=clairvoyant_audit,
        unrestricted_center=unrestricted_center,
        unrestricted_audit_cost=unrestricted_audit,
        coverage_recovery_fraction=recovery,
    )
    row: dict[str, Any] = {
        "episode": episode,
        "snapshot": source_path.stem,
        "unrestricted_audit_cost": float(unrestricted_audit.mean()),
        "fit_condition": generated["fit_condition"],
        "result": str(output_path),
    }
    for index, name in enumerate(BANK_NAMES):
        row[f"{name}_selected_audit"] = float(selected_audit[index])
        row[f"{name}_clairvoyant_audit"] = float(clairvoyant_audit[index])
        row[f"{name}_coverage_recovery"] = float(recovery[index])
        row[f"{name}_selected_slot"] = str(generated["slot_names"][index, selected_index[index]])
        row[f"{name}_clairvoyant_slot"] = str(generated["slot_names"][index, clairvoyant_index[index]])
    return row


def main() -> None:
    args = parse_args()
    if set(args.selection_seeds) & set(args.audit_seeds):
        raise ValueError("selection and audit seeds overlap")
    if args.candidate_noise_scale <= 0:
        raise ValueError("candidate-noise-scale must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    rows = [process_one(args, path, device) for path in args.center_result_dirs]
    with (args.output_dir / "per_snapshot.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "semantics": "forward-only dynamic 33-slot bank; no DBM analytic gradient",
        "snapshot_count": len(rows),
        "selection_seeds": list(args.selection_seeds),
        "audit_seeds": list(args.audit_seeds),
        "candidate_budget_per_center": 64,
        "candidate_noise_scale": args.candidate_noise_scale,
        "candidate_noise_design": "zero_extra_antithetic_pairs",
        "unrestricted_audit_mean": float(np.mean([row["unrestricted_audit_cost"] for row in rows])),
        "banks": {},
    }
    for name in BANK_NAMES:
        selected = np.asarray([row[f"{name}_selected_audit"] for row in rows])
        clairvoyant = np.asarray([row[f"{name}_clairvoyant_audit"] for row in rows])
        recovery = np.asarray([row[f"{name}_coverage_recovery"] for row in rows])
        summary["banks"][name] = {
            "selected_audit_mean": float(selected.mean()),
            "clairvoyant_audit_mean": float(clairvoyant.mean()),
            "clairvoyant_audit_max": float(clairvoyant.max()),
            "coverage_recovery_mean": float(recovery.mean()),
            "snapshot_wins_vs_B0": int(np.sum(clairvoyant < np.asarray([row["B0_current_clairvoyant_audit"] for row in rows]))),
        }
    baseline = summary["banks"]["B0_current"]["clairvoyant_audit_mean"]
    unrestricted = summary["unrestricted_audit_mean"]
    target = baseline - 0.5 * (baseline - unrestricted)
    summary["gate"] = {
        "target_clairvoyant_mean": target,
        "requires_recovery_fraction": 0.5,
        "passing_banks": [
            name for name in BANK_NAMES[1:]
            if summary["banks"][name]["clairvoyant_audit_mean"] <= target
        ],
    }
    summary["rows"] = rows
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
