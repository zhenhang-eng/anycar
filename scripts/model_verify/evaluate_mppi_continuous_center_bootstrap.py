#!/usr/bin/env python3
"""Fresh-seed fixed-DBM gate for the continuous-center SAC bootstrap Actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TorchMPPIContinuousCenterActor,
    ego_reference_features,
)
from generate_dbm_multicenter_teacher import make_controller
from generate_dbm_sampling_center_gt_pilot import evaluate_centers


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
DEFAULT_BANK = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_multidirection_replay_diverse_20260805_v1"
)
DEFAULT_T1 = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_teacher_t1_diverse_20260805_v1"
)
DEFAULT_CHECKPOINT = Path(
    "outputs/mppi_proposal/continuous_center_sac_bootstrap_20260806_v1/"
    "continuous_center_sac_bootstrap.pt"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/continuous_center_sac_bootstrap_eval_20260806_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--risk-labels", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--bank-labels", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--t1-labels", type=Path, default=DEFAULT_T1)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--episodes", nargs="+", default=(
            "episode_000", "episode_021", "episode_042", "episode_063", "episode_084"
        )
    )
    parser.add_argument("--control-step", type=int, default=250)
    parser.add_argument("--context-index", type=int, default=0)
    parser.add_argument("--candidate-noise-scale", type=float, default=0.10)
    parser.add_argument("--selection-seeds", type=int, nargs="+", default=(29711, 29712))
    parser.add_argument("--audit-seeds", type=int, nargs="+", default=(29721, 29722, 29723, 29724))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def actor_inputs(
    source: np.lib.npyio.NpzFile,
    parent: np.lib.npyio.NpzFile,
    risk: np.lib.npyio.NpzFile,
    context: int,
    checkpoint: dict,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    state = np.asarray(source["initial_state"], np.float32)
    action = np.asarray(source["current_action"], np.float32)
    history = np.asarray(source["history"][0], np.float32)[None]
    reference = ego_reference_features(source["reference_ego"], float(state[3]))[None]
    current = np.asarray((state[3], state[4], *action), np.float32)[None]
    norm = MPPIProposalNormalization.from_dict(checkpoint["state_normalization"])
    history, reference, current = norm.normalize_numpy(history, reference, current)
    anchor = np.asarray(parent["guided_center_knots"][context], np.float32)[None]
    feedback = np.asarray(parent["first_pass_feedback"][context], np.float32)
    feedback = (feedback - checkpoint["feedback_mean"]) / checkpoint["feedback_std"]
    gradient = np.concatenate((
        np.asarray(risk["critic_gradient_mean"][context], np.float32),
        np.asarray(risk["critic_gradient_std"][context], np.float32),
    ))
    gradient = (gradient - checkpoint["gradient_mean"]) / checkpoint["gradient_std"]
    return tuple(
        torch.from_numpy(np.asarray(value, np.float32)).to(device)
        for value in (
            history, reference, current, anchor,
            feedback[None], gradient[None],
        )
    )


def main() -> None:
    args = parse_args()
    if set(args.selection_seeds) & set(args.audit_seeds):
        raise ValueError("selection and audit seeds must be disjoint")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    actor = TorchMPPIContinuousCenterActor(
        float(checkpoint["maximum_delta_sigma"]), dropout=0.0
    ).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    rows = []
    for episode in args.episodes:
        name = f"step_{args.control_step:06d}.npz"
        source_path = args.source / episode / "snapshots" / name
        parent_path = args.parent_labels / episode / name
        risk_path = args.risk_labels / episode / name
        bank_path = args.bank_labels / episode / name
        t1_path = args.t1_labels / episode / name
        with np.load(source_path, allow_pickle=False) as source, np.load(
            parent_path, allow_pickle=False
        ) as parent, np.load(risk_path, allow_pickle=False) as risk, np.load(
            bank_path, allow_pickle=False
        ) as bank, np.load(t1_path, allow_pickle=False) as t1:
            config = {"objective": {"cost_weights": json.loads(str(source["cost_weights_json"]))}}
            controller, backend = make_controller(source, config, device)
            model_input = actor_inputs(
                source, parent, risk, args.context_index, checkpoint, device
            )
            with torch.no_grad():
                normalized_action, _, actor_center_tensor = actor.sample(
                    *model_input, deterministic=True
                )
            actor_center = actor_center_tensor[0].cpu().numpy().astype(np.float32)
            normalized_action_numpy = normalized_action[0].cpu().numpy().astype(np.float32)
            anchor = np.asarray(
                parent["guided_center_knots"][args.context_index], np.float32
            )
            teacher = np.asarray(t1["teacher_center_knots"], np.float32)
            bank_centers = np.asarray(
                bank["centers"][args.context_index], np.float32
            )
            params = json.loads(str(source["mppi_params_json"]))
            sigma = np.asarray(params["noise_sigma"], np.float32)
            candidate_sigma = sigma * float(args.candidate_noise_scale)
            action_min = np.asarray(params["action_min"], np.float32)
            action_max = np.asarray(params["action_max"], np.float32)
            history = torch.from_numpy(source["history"]).to(device)
            initial = torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5)
            current_action = torch.from_numpy(source["current_action"]).to(device).reshape(1, 2)
            reference = controller._prepare_reference(source["reference"])
            centers = np.concatenate((anchor[None], actor_center[None], teacher[None], bank_centers))
            selection = evaluate_centers(
                centers, list(args.selection_seeds), controller, backend,
                history, initial, current_action, reference,
                candidate_sigma, action_min, action_max,
            )
            audit = evaluate_centers(
                centers, list(args.audit_seeds), controller, backend,
                history, initial, current_action, reference,
                candidate_sigma, action_min, action_max,
            )
            bank_selected = 3 + int(np.argmin(selection["mean_cost"][3:]))
            bank_clairvoyant = 3 + int(np.argmin(audit["mean_cost"][3:]))
            standardized_distance = np.sqrt(np.mean(
                ((bank_centers - actor_center[None]) / sigma) ** 2, axis=(1, 2)
            ))
            row = {
                "episode": episode,
                "anchor_audit_cost": float(audit["mean_cost"][0]),
                "actor_audit_cost": float(audit["mean_cost"][1]),
                "teacher_audit_cost": float(audit["mean_cost"][2]),
                "bank_selected_audit_cost": float(audit["mean_cost"][bank_selected]),
                "bank_clairvoyant_audit_cost": float(audit["mean_cost"][bank_clairvoyant]),
                "actor_gain_vs_anchor": float(audit["mean_cost"][0] - audit["mean_cost"][1]),
                "actor_nearest_bank_sigma_rms": float(standardized_distance.min()),
                "actor_action_abs_mean": float(np.mean(np.abs(normalized_action_numpy))),
            }
            rows.append(row)
            np.savez_compressed(
                args.output_dir / f"{episode}_{name}",
                centers=centers,
                selection_output_cost=selection["output_cost"],
                audit_output_cost=audit["output_cost"],
                normalized_actor_action=normalized_action_numpy,
                bank_selected_index=np.asarray(bank_selected),
                bank_clairvoyant_index=np.asarray(bank_clairvoyant),
            )
    keys = (
        "anchor_audit_cost", "actor_audit_cost", "teacher_audit_cost",
        "bank_selected_audit_cost", "bank_clairvoyant_audit_cost",
        "actor_gain_vs_anchor", "actor_nearest_bank_sigma_rms",
        "actor_action_abs_mean",
    )
    mean = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    summary = {
        "format_version": 1,
        "semantics": "fresh-seed fixed-DBM gate for BC-only continuous Actor initialization",
        "qualification": "BOOTSTRAP_EVALUATION_NOT_SAC",
        "checkpoint": str(args.checkpoint.resolve()),
        "candidate_budget_per_center": 64,
        "candidate_noise_scale": args.candidate_noise_scale,
        "candidate_noise_design": "zero_extra_antithetic_pairs",
        "selection_seeds": list(args.selection_seeds),
        "audit_seeds": list(args.audit_seeds),
        "mean": mean,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
