#!/usr/bin/env python3
"""Best-found fixed-DBM MPPI sampling-center oracle on frozen snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from generate_dbm_multicenter_teacher import (
    evaluate_actions,
    evaluate_knots,
    make_controller,
    stable_weight,
)
from generate_dbm_two_pass_feedback_labels import antithetic_noise


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1"
)
DEFAULT_T1 = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_teacher_t1_diverse_20260805_v1"
)
DEFAULT_BANK = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_multidirection_replay_diverse_20260805_v1"
)
DEFAULT_OUTPUT = ROOT / "outputs/mppi_proposal/dbm_sampling_center_gt_pilot_20260806_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--t1-labels", type=Path, default=DEFAULT_T1)
    parser.add_argument("--bank-labels", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--direct-result", type=Path, required=True)
    parser.add_argument("--episode", default="episode_042")
    parser.add_argument("--control-step", type=int, default=250)
    parser.add_argument("--context-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--scales", type=float, nargs="+", default=(1.0, 0.5, 0.25, 0.10))
    parser.add_argument(
        "--candidate-noise-scale",
        type=float,
        default=0.10,
        help="Probe-candidate sigma relative to the source MPPI sigma.",
    )
    parser.add_argument("--selection-seeds", type=int, nargs="+", default=(29411, 29412))
    parser.add_argument("--audit-seeds", type=int, nargs="+", default=(29421, 29422, 29423, 29424))
    parser.add_argument("--search-seed", type=int, default=29431)
    return parser.parse_args()


@torch.no_grad()
def evaluate_centers(
    centers: np.ndarray,
    seeds: list[int],
    controller,
    backend,
    history: torch.Tensor,
    initial_state: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
    sigma: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> dict[str, np.ndarray]:
    center_count = len(centers)
    sample_count = controller.params.num_samples
    output_cost = np.empty((center_count, len(seeds)), dtype=np.float32)
    ess = np.empty_like(output_cost)
    weighted_actions = np.empty(
        (center_count, len(seeds), controller.params.horizon, 2), dtype=np.float32
    )
    for seed_index, seed in enumerate(seeds):
        noise = antithetic_noise(seed, sample_count, (8, 2), sigma)
        raw = centers[:, None] + noise[None]
        knots = np.clip(raw, action_min, action_max).astype(np.float32)
        cost, actions, _ = evaluate_knots(
            controller, backend, knots.reshape(-1, 8, 2), history,
            initial_state, current_action, reference,
        )
        cost = cost.reshape(center_count, sample_count)
        actions = actions.reshape(center_count, sample_count, controller.params.horizon, 2)
        for center_index in range(center_count):
            weight = stable_weight(cost[center_index], controller.params.temperature)
            weighted = np.sum(weight[:, None, None] * actions[center_index], axis=0).astype(np.float32)
            weighted_actions[center_index, seed_index] = weighted
            ess[center_index, seed_index] = float(1.0 / np.sum(weight * weight))
        one_cost, _ = evaluate_actions(
            controller, backend,
            weighted_actions[:, seed_index], history, initial_state,
            current_action, reference,
        )
        output_cost[:, seed_index] = one_cost
    return {
        "output_cost": output_cost,
        "mean_cost": output_cost.mean(axis=1),
        "ess": ess,
        "weighted_actions": weighted_actions,
    }


def main() -> None:
    args = parse_args()
    if set(args.selection_seeds) & set(args.audit_seeds):
        raise ValueError("selection and audit seeds must be disjoint")
    if args.population <= args.elite_count or args.elite_count < 2:
        raise ValueError("population must exceed elite-count >= 2")
    if args.candidate_noise_scale <= 0:
        raise ValueError("candidate-noise-scale must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshot_path = (
        args.source / args.episode / "snapshots" / f"step_{args.control_step:06d}.npz"
    )
    t1_path = args.t1_labels / args.episode / f"step_{args.control_step:06d}.npz"
    bank_path = args.bank_labels / args.episode / f"step_{args.control_step:06d}.npz"
    device = torch.device(args.device)
    with np.load(snapshot_path, allow_pickle=False) as source:
        config = {
            "objective": {"cost_weights": json.loads(str(source["cost_weights_json"]))}
        }
        controller, backend = make_controller(source, config, device)
        history = torch.from_numpy(source["history"]).to(device)
        initial_state = torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5)
        current_action = torch.from_numpy(source["current_action"]).to(device).reshape(1, 2)
        reference = controller._prepare_reference(source["reference"])
        params = json.loads(str(source["mppi_params_json"]))
        warm = np.asarray(source["sampling_mean_knots"], np.float32)
        sigma = np.asarray(params["noise_sigma"], np.float32)
        candidate_sigma = sigma * float(args.candidate_noise_scale)
        action_min = np.asarray(params["action_min"], np.float32)
        action_max = np.asarray(params["action_max"], np.float32)
        with np.load(t1_path, allow_pickle=False) as t1:
            teacher = np.asarray(t1["teacher_center_knots"], np.float32)
        with np.load(args.direct_result, allow_pickle=False) as direct:
            direct_index = int(direct["knot_best_index"])
            j16_center = np.asarray(direct["optimized_knots"][direct_index], np.float32)
            j16_cost = float(direct["knot_cost_replay"][direct_index])
        with np.load(bank_path, allow_pickle=False) as bank:
            action_names = np.asarray(bank["action_names"]).astype(str)
            bank_centers = np.asarray(bank["centers"][args.context_index], np.float32)

        start_names = ["warm", "t1_teacher", "j16_direct"]
        starts = [warm, teacher, j16_center]
        rng = np.random.default_rng(args.search_seed)
        candidates: list[np.ndarray] = []
        candidate_names: list[str] = []
        candidate_selection_cost: list[float] = []
        for start_name, start in zip(start_names, starts):
            center = start.copy()
            for stage, scale in enumerate(args.scales):
                perturbation = rng.standard_normal((args.population - 1, 8, 2)).astype(np.float32)
                population = np.concatenate(
                    (center[None], center[None] + perturbation * sigma * float(scale)), axis=0
                )
                population = np.clip(population, action_min, action_max).astype(np.float32)
                evaluation = evaluate_centers(
                    population, list(args.selection_seeds), controller, backend,
                    history, initial_state, current_action, reference,
                    candidate_sigma, action_min, action_max,
                )
                order = np.argsort(evaluation["mean_cost"])
                for index in order[: args.elite_count]:
                    candidates.append(population[index].copy())
                    candidate_names.append(f"{start_name}_stage{stage}_rank{len(candidates)}")
                    candidate_selection_cost.append(float(evaluation["mean_cost"][index]))
                elite = population[order[: args.elite_count]]
                elite_cost = evaluation["mean_cost"][order[: args.elite_count]].astype(np.float64)
                scale_cost = max(float(np.quantile(elite_cost, 0.75) - elite_cost.min()), 0.05)
                weight = np.exp(-(elite_cost - elite_cost.min()) / scale_cost)
                weight /= weight.sum()
                center = np.sum(weight[:, None, None] * elite, axis=0).astype(np.float32)

        # Deduplicate the retained CEM elites, then choose only on selection seeds.
        unique: list[np.ndarray] = []
        unique_names: list[str] = []
        for name, center in zip(candidate_names, candidates):
            if not any(np.allclose(center, old, rtol=0, atol=1e-7) for old in unique):
                unique.append(center)
                unique_names.append(name)
        search_centers = np.asarray(unique, np.float32)
        selection = evaluate_centers(
            search_centers, list(args.selection_seeds), controller, backend,
            history, initial_state, current_action, reference,
            candidate_sigma, action_min, action_max,
        )
        selected_index = int(np.argmin(selection["mean_cost"]))
        selected_center = search_centers[selected_index]
        comparison_names = np.concatenate(
            (np.asarray(("unrestricted_selected", "warm", "t1_teacher", "j16_direct")), action_names)
        )
        comparison_centers = np.concatenate(
            (selected_center[None], warm[None], teacher[None], j16_center[None], bank_centers), axis=0
        )
        audit = evaluate_centers(
            comparison_centers, list(args.audit_seeds), controller, backend,
            history, initial_state, current_action, reference,
            candidate_sigma, action_min, action_max,
        )
        bank_slice = slice(4, 4 + len(bank_centers))
        bank_selection = evaluate_centers(
            bank_centers, list(args.selection_seeds), controller, backend,
            history, initial_state, current_action, reference,
            candidate_sigma, action_min, action_max,
        )
        bank_selected_index = int(np.argmin(bank_selection["mean_cost"]))
        bank_audit_clairvoyant = int(np.argmin(audit["mean_cost"][bank_slice]))

    np.savez_compressed(
        args.output_dir / "center_oracle.npz",
        source_snapshot=np.asarray(str(snapshot_path)),
        context_index=np.asarray(args.context_index),
        selection_seeds=np.asarray(args.selection_seeds),
        audit_seeds=np.asarray(args.audit_seeds),
        candidate_noise_scale=np.asarray(args.candidate_noise_scale, np.float32),
        candidate_noise_design=np.asarray("zero_extra_antithetic_pairs"),
        search_center_names=np.asarray(unique_names),
        search_centers=search_centers,
        search_selection_output_cost=selection["output_cost"],
        selected_search_index=np.asarray(selected_index),
        comparison_names=comparison_names,
        comparison_centers=comparison_centers,
        audit_output_cost=audit["output_cost"],
        audit_ess=audit["ess"],
        bank_selected_index=np.asarray(bank_selected_index),
        bank_audit_clairvoyant_index=np.asarray(bank_audit_clairvoyant),
    )
    bank_selected_audit_index = 4 + bank_selected_index
    bank_clairvoyant_audit_index = 4 + bank_audit_clairvoyant
    summary = {
        "semantics": "best-found center oracle selected without audit-seed access",
        "episode": args.episode,
        "control_step": args.control_step,
        "context_index": args.context_index,
        "candidate_budget_per_center": controller.params.num_samples,
        "candidate_noise_scale": args.candidate_noise_scale,
        "candidate_noise_design": "zero_extra_antithetic_pairs",
        "selection_seeds": list(args.selection_seeds),
        "audit_seeds": list(args.audit_seeds),
        "j16_direct_cost": j16_cost,
        "audit_mean_cost": {
            "unrestricted_selected": float(audit["mean_cost"][0]),
            "warm": float(audit["mean_cost"][1]),
            "t1_teacher": float(audit["mean_cost"][2]),
            "j16_as_center": float(audit["mean_cost"][3]),
            "bank_selected_on_selection": float(audit["mean_cost"][bank_selected_audit_index]),
            "bank_clairvoyant_on_audit": float(audit["mean_cost"][bank_clairvoyant_audit_index]),
        },
        "selected_center_source": unique_names[selected_index],
        "bank_selected_name": str(action_names[bank_selected_index]),
        "bank_audit_clairvoyant_name": str(action_names[bank_audit_clairvoyant]),
        "search_center_count": len(search_centers),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
