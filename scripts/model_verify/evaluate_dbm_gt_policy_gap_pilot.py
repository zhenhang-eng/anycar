#!/usr/bin/env python3
"""Decompose fixed/feedback policy gaps on GT-first frozen DBM pilots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TorchMPPISequentialProbeActorCritic,
    ego_reference_features,
)
from generate_dbm_multicenter_teacher import make_controller
from generate_dbm_sampling_center_gt_pilot import evaluate_centers


DEFAULT_PARENT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_feedback_diverse_20260805_v1"
)
DEFAULT_RISK = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_risk_replay_diverse_20260805_v1"
)
DEFAULT_CHECKPOINT = Path(
    "outputs/mppi_proposal/sequential_probe_sac_20260805_v2/sequential_probe_sac.pt"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/dbm_gt_policy_gap_pilot_20260806_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", type=Path, nargs="+")
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--risk-labels", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--probe-budget", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_action_names = list(checkpoint["action_names"])
    action_count = len(checkpoint_action_names)
    actor = TorchMPPISequentialProbeActorCritic(action_count, dropout=0.0).to(device)
    actor.load_state_dict(checkpoint["model_state_dict"])
    actor.eval()
    state_norm = MPPIProposalNormalization.from_dict(checkpoint["state_normalization"])
    feedback_mean = np.asarray(checkpoint["feedback_mean"], np.float32)
    feedback_std = np.asarray(checkpoint["feedback_std"], np.float32)
    gradient_mean = np.asarray(checkpoint["gradient_mean"], np.float32)
    gradient_std = np.asarray(checkpoint["gradient_std"], np.float32)
    reward_scale = float(checkpoint["reward_scale"])
    rows: list[dict[str, object]] = []
    method_values = {
        name: [] for name in (
            "unrestricted_center", "bank_audit_clairvoyant", "full_bank_probe",
            "fixed_priority_4probe", "feedback_4probe", "guided_anchor",
        )
    }
    fixed_names = (
        "guided_anchor",
        "critic_plus_preconditioned_pos_0p150",
        "critic_pos_0p150",
        "negative_gradient_pos_0p150",
    )
    for result_dir in args.result_dirs:
        with np.load(result_dir / "center_oracle.npz", allow_pickle=False) as result:
            if not np.isclose(float(result["candidate_noise_scale"]), 0.10):
                raise ValueError(f"{result_dir}: policy gap requires 0.10 sigma")
            source_path = Path(str(result["source_snapshot"]))
            episode = source_path.parents[1].name
            context = int(result["context_index"])
            selection_seeds = list(np.asarray(result["selection_seeds"], dtype=int))
            audit_cost = np.asarray(result["audit_output_cost"], np.float32)
            comparison_names = list(result["comparison_names"].astype(str))
            bank_names = comparison_names[4:]
            bank_centers = np.asarray(result["comparison_centers"][4:], np.float32)
            if len(bank_names) != action_count:
                raise ValueError(f"{result_dir}: checkpoint action count mismatch")
            if bank_names != checkpoint_action_names:
                raise ValueError(f"{result_dir}: checkpoint action names mismatch")
            with np.load(source_path, allow_pickle=False) as source:
                config = {"objective": {"cost_weights": json.loads(str(source["cost_weights_json"]))}}
                controller, backend = make_controller(source, config, device)
                history_tensor = torch.from_numpy(source["history"]).to(device)
                initial = torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5)
                current_tensor = torch.from_numpy(source["current_action"]).to(device).reshape(1, 2)
                reference_tensor = controller._prepare_reference(source["reference"])
                params = json.loads(str(source["mppi_params_json"]))
                sigma = np.asarray(params["noise_sigma"], np.float32) * 0.10
                selection = evaluate_centers(
                    bank_centers, selection_seeds, controller, backend,
                    history_tensor, initial, current_tensor, reference_tensor,
                    sigma, np.asarray(params["action_min"], np.float32),
                    np.asarray(params["action_max"], np.float32),
                )["output_cost"]
                state = np.asarray(source["initial_state"], np.float32)
                current_action = np.asarray(source["current_action"], np.float32)
                history = np.asarray(source["history"], np.float32)
                reference = ego_reference_features(source["reference_ego"], float(state[3]))[None]
                current = np.asarray((state[3], state[4], *current_action), np.float32)[None]
                history, reference, current = state_norm.normalize_numpy(history, reference, current)
            parent_path = args.parent_labels / episode / source_path.name
            risk_path = args.risk_labels / episode / source_path.name
            with np.load(parent_path, allow_pickle=False) as parent, np.load(
                risk_path, allow_pickle=False
            ) as risk:
                anchor = np.asarray(parent["guided_center_knots"][context : context + 1], np.float32)
                feedback = (
                    np.asarray(parent["first_pass_feedback"][context], np.float32)
                    - feedback_mean
                ) / feedback_std
                gradient = np.concatenate(
                    (
                        np.asarray(risk["critic_gradient_mean"][context], np.float32),
                        np.asarray(risk["critic_gradient_std"][context], np.float32),
                    )
                )
                gradient = (gradient - gradient_mean) / gradient_std
            fixed_indices = np.asarray([bank_names.index(name) for name in fixed_names])
            anchor_index = int(fixed_indices[0])
            audit_bank_mean = audit_cost[4:].mean(axis=1)
            clairvoyant_index = int(np.argmin(audit_bank_mean))
            for seed_index, probe_seed in enumerate(selection_seeds):
                probe_cost = selection[:, seed_index]
                full_index = int(np.argmin(probe_cost))
                fixed_index = int(fixed_indices[np.argmin(probe_cost[fixed_indices])])
                probe_value = np.zeros((1, action_count), np.float32)
                probe_mask = np.zeros((1, action_count), np.bool_)
                probe_mask[0, anchor_index] = True
                sequence: list[int] = [anchor_index]
                for probe_step in range(1, args.probe_budget):
                    remaining = np.asarray(
                        [[(args.probe_budget - probe_step) / (args.probe_budget - 1)]],
                        np.float32,
                    )
                    logits, _, _ = actor(
                        torch.from_numpy(history).to(device),
                        torch.from_numpy(reference).to(device),
                        torch.from_numpy(current).to(device),
                        torch.from_numpy(anchor).to(device),
                        torch.from_numpy(feedback[None]).to(device),
                        torch.from_numpy(gradient[None]).to(device),
                        torch.from_numpy(probe_value).to(device),
                        torch.from_numpy(probe_mask).to(device),
                        torch.from_numpy(remaining).to(device),
                    )
                    action = int(
                        logits.masked_fill(torch.from_numpy(probe_mask).to(device), -1e9)
                        .argmax(1).item()
                    )
                    sequence.append(action)
                    probe_value[0, action] = np.clip(
                        (probe_cost[anchor_index] - probe_cost[action]) / reward_scale,
                        -10.0, 10.0,
                    )
                    probe_mask[0, action] = True
                feedback_index = int(np.where(probe_mask[0], probe_value[0], -np.inf).argmax())
                values = {
                    "unrestricted_center": float(audit_cost[0].mean()),
                    "bank_audit_clairvoyant": float(audit_bank_mean[clairvoyant_index]),
                    "full_bank_probe": float(audit_bank_mean[full_index]),
                    "fixed_priority_4probe": float(audit_bank_mean[fixed_index]),
                    "feedback_4probe": float(audit_bank_mean[feedback_index]),
                    "guided_anchor": float(audit_bank_mean[anchor_index]),
                }
                for name, value in values.items():
                    method_values[name].append(value)
                rows.append(
                    {
                        "episode": episode,
                        "context_index": context,
                        "probe_seed": probe_seed,
                        **values,
                        "full_bank_action": bank_names[full_index],
                        "fixed_action": bank_names[fixed_index],
                        "feedback_action": bank_names[feedback_index],
                        "feedback_sequence": ";".join(bank_names[index] for index in sequence),
                    }
                )
    summary = {
        "semantics": "same frozen states, 0.10-sigma probes, independent four-seed audit costs",
        "checkpoint": str(args.checkpoint.resolve()),
        "snapshot_count": len(args.result_dirs),
        "probe_context_count": len(rows),
        "method_cost": {name: summarize(values) for name, values in method_values.items()},
        "mean_gap_to_unrestricted": {
            name: float(np.mean(method_values[name]) - np.mean(method_values["unrestricted_center"]))
            for name in method_values if name != "unrestricted_center"
        },
        "state_only_status": "PENDING: no frozen sequential state-only checkpoint exists",
    }
    with (args.output_dir / "per_probe_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
