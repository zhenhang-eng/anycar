#!/usr/bin/env python3
"""Audit the state-level negative tail of the 20/100/200-round OAC-2 Actors.

This analysis replays the frozen internal-selection split only.  It separates
states that were already expensive from states whose deterministic DBM cost is
made worse by longer Actor training, and attributes the latter by speed,
scenario, cost component, action channel, seed agreement, and final Twin-Value
prediction.  Formal validation and test data remain sealed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from generate_dbm_direct_gt_validation import interpolate_knots
from generate_dbm_proposal_teacher import sha256_file
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from mppi_a2_actors import DirectNoAnchorGTXActor
from train_mppi_oac2_continuous_actor import load_value
from train_mppi_online_absolute_sac import (
    actor_mean,
    critic_state_inputs,
    load_actor_normalization,
    load_bank,
    make_actor_inputs,
    predict_actions,
    rollout_bank,
)


DEFAULT_RUNS = {
    20: Path("outputs/mppi_proposal/online_absolute_sac_oac2_fold1_20260824_v1"),
    100: Path("outputs/mppi_proposal/online_absolute_sac_oac2_fold1_100round_20260824_v1"),
    200: Path("outputs/mppi_proposal/online_absolute_sac_oac2_fold1_200round_20260824_v1"),
}
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_budget_tail_20260824_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run20", type=Path, default=DEFAULT_RUNS[20])
    parser.add_argument("--run100", type=Path, default=DEFAULT_RUNS[100])
    parser.add_argument("--run200", type=Path, default=DEFAULT_RUNS[200])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def distribution(value: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(value, np.float64)
    return {
        "count": int(len(value)),
        "mean": float(np.mean(value)),
        "p05": float(np.quantile(value, 0.05)),
        "p10": float(np.quantile(value, 0.10)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
        "p95": float(np.quantile(value, 0.95)),
        "minimum": float(np.min(value)),
        "maximum": float(np.max(value)),
    }


def load_actor(path: Path, device: torch.device) -> DirectNoAnchorGTXActor:
    payload = torch.load(path, map_location=device)
    if payload.get("formal_validation_loaded") or payload.get("test_loaded"):
        raise AssertionError(f"sealed split violation in {path}")
    actor = DirectNoAnchorGTXActor(dropout=0.0).to(device)
    actor.load_state_dict(payload["model_state_dict"], strict=True)
    actor.eval()
    return actor


def action_cost_components(
    backend: TorchDynamicBicycleRolloutBackend,
    weights: TorchMPPICostWeights,
    params: TorchMPPIParams,
    actions: np.ndarray,
    states: np.ndarray,
    current: np.ndarray,
    references: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    names = (
        "position", "yaw", "vx", "yawrate", "acceleration_rate",
        "steering_rate", "total",
    )
    result: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for begin in range(0, len(indices), batch_size):
        local = indices[begin:begin + batch_size]
        knots = torch.from_numpy(actions[begin:begin + batch_size, None]).to(device)
        full_actions = interpolate_knots(knots, params.horizon)
        with torch.no_grad():
            batch, starts, horizon, _ = full_actions.shape
            flat_actions = full_actions.reshape(batch * starts, horizon, 2)
            initial = torch.from_numpy(states[local]).to(device)
            initial = initial[:, None].expand(-1, starts, -1).reshape(batch * starts, 6)
            full = backend.rollout_full_state_differentiable(initial, flat_actions)
            trajectory = full[..., [0, 1, 2, 3, 5]].reshape(batch, starts, horizon, 5)
            reference = torch.from_numpy(references[local]).to(device)[:, None]
            position = weights.position * (
                trajectory[..., :2] - reference[..., :2]
            ).square().sum(-1).sum(-1)
            yaw_delta = trajectory[..., 2] - reference[..., 2]
            yaw = weights.yaw * torch.atan2(
                torch.sin(yaw_delta), torch.cos(yaw_delta)
            ).square().sum(-1)
            vx = weights.vx * (
                trajectory[..., 3] - reference[..., 3]
            ).square().sum(-1)
            if references.shape[-1] == 5 and weights.yawrate != 0:
                yawrate = weights.yawrate * (
                    trajectory[..., 4] - reference[..., 4]
                ).square().sum(-1)
            else:
                yawrate = torch.zeros_like(position)
            current_tensor = torch.from_numpy(current[local]).to(device)
            previous = torch.cat((
                current_tensor[:, None, None].expand(-1, starts, 1, -1),
                full_actions[:, :, :-1],
            ), dim=2)
            rate = full_actions - previous
            acceleration_rate = weights.acceleration_rate * rate[..., 0].square().sum(-1)
            steering_rate = weights.steering_rate * rate[..., 1].square().sum(-1)
            values = {
                "position": position,
                "yaw": yaw,
                "vx": vx,
                "yawrate": yawrate,
                "acceleration_rate": acceleration_rate,
                "steering_rate": steering_rate,
            }
            values["total"] = sum(values.values())
        for name, value in values.items():
            result[name].append(value[:, 0].cpu().numpy())
    return {
        name: np.concatenate(chunks).astype(np.float32)
        for name, chunks in result.items()
    }


def subset_breakdown(
    mask: np.ndarray, data: dict[str, np.ndarray], indices: np.ndarray,
) -> dict[str, Any]:
    speed = data["speed"][indices]
    scenario = data["scenario"][indices]
    count = int(mask.sum())
    return {
        "count": count,
        "fraction": float(mask.mean()),
        "by_speed": {
            str(float(value)): {
                "count": int(np.sum(mask & np.isclose(speed, value))),
                "rate_within_slice": float(np.mean(mask[np.isclose(speed, value)])),
            }
            for value in sorted(np.unique(speed))
        },
        "by_scenario": {
            str(value): {
                "count": int(np.sum(mask & (scenario == value))),
                "rate_within_slice": float(np.mean(mask[scenario == value])),
            }
            for value in sorted(np.unique(scenario))
        },
    }


def main() -> None:
    args = parse_args()
    runs = {20: args.run20, 100: args.run100, 200: args.run200}
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)

    contracts = {budget: json.loads((run / "contract.json").read_text()) for budget, run in runs.items()}
    reference = contracts[200]
    for budget, contract in contracts.items():
        if int(contract["arguments"]["rounds"]) != budget:
            raise AssertionError(f"budget mismatch for {runs[budget]}")
        for key in (
            "outer_fold", "fit_episodes", "internal_selection_episodes",
            "candidate_bank_sha256", "parent_summary_sha256",
        ):
            if contract[key] != reference[key]:
                raise AssertionError(f"non-budget contract mismatch: {key}")

    bank_root = Path(reference["arguments"]["bank_root"])
    data = load_bank(bank_root)
    selection = np.flatnonzero(np.isin(data["episode"], reference["internal_selection_episodes"]))
    if len(selection) != 600:
        raise AssertionError(f"expected 600 selection states, got {len(selection)}")
    states, current, references, params_json, weights_json, dbm_json = load_rollout_inputs(
        data, Path(reference["arguments"]["gt_v1"])
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    normalization, normalization_source = load_actor_normalization(
        Path(reference["arguments"]["base_ac"])
    )
    inputs = make_actor_inputs(data, normalization)

    action = np.empty((4, 3, len(selection), 8, 2), np.float32)
    cost = np.empty((4, 3, len(selection)), np.float32)
    labels = (0, 20, 100, 200)
    action_paths: list[dict[str, Any]] = []
    for seed in range(3):
        initial_path = Path(reference["arguments"]["actor_root"]) / f"a0_fold1_seed{seed}.pt"
        actor = load_actor(initial_path, device)
        action[0, seed] = actor_mean(actor, inputs, selection, device)
        cost[0, seed] = rollout_bank(
            backend, weights, params, action[0, seed, :, None], states, current,
            references, selection, args.batch_size, device,
        )[:, 0]
        action_paths.append({"budget": 0, "seed": seed, "path": str(initial_path.resolve()), "sha256": sha256_file(initial_path)})
        del actor
        for budget_index, budget in enumerate(labels[1:], start=1):
            path = runs[budget] / f"seed_{seed}" / "actor_selected.pt"
            actor = load_actor(path, device)
            action[budget_index, seed] = actor_mean(actor, inputs, selection, device)
            cost[budget_index, seed] = rollout_bank(
                backend, weights, params, action[budget_index, seed, :, None],
                states, current, references, selection, args.batch_size, device,
            )[:, 0]
            action_paths.append({"budget": budget, "seed": seed, "path": str(path.resolve()), "sha256": sha256_file(path)})
            del actor

    bank_cost = data["costs"][selection].astype(np.float64)
    warm_cost = bank_cost[:, 0]
    teacher_cost = bank_cost.min(axis=1)
    mean_cost = cost.mean(axis=1)
    mean_gain = cost[0].mean(axis=0)[None] - mean_cost
    # Rewrite row zero explicitly for readability: initial relative gain is zero.
    mean_gain[0] = 0.0
    per_seed_gain = cost[0, None] - cost

    budget_summary = {}
    for budget_index, budget in enumerate(labels):
        gain = per_seed_gain[budget_index]
        budget_summary[str(budget)] = {
            "cost_all_seed_state": distribution(cost[budget_index].reshape(-1)),
            "cost_seed_mean_per_state": distribution(mean_cost[budget_index]),
            "gain_all_seed_state": distribution(gain.reshape(-1)),
            "regression_fraction": float(np.mean(gain < 0)),
            "state_regression_seed_count": {
                str(count): int(np.sum(np.sum(gain < 0, axis=0) == count))
                for count in range(4)
            },
        }

    # State-level trends are evaluated after averaging paired seed trajectories.
    c0, c20, c100, c200 = mean_cost
    g20, g100, g200 = c0 - c20, c0 - c100, c0 - c200
    masks = {
        "regress_at_200": g200 < 0,
        "severe_regress_at_200_gain_le_minus10": g200 <= -10,
        "worsens_100_to_200": c200 > c100,
        "regress_at_200_and_worsens_100_to_200": (g200 < 0) & (c200 > c100),
        "monotonic_improvement_0_20_100_200": (c20 < c0) & (c100 < c20) & (c200 < c100),
        "top5pct_negative_gain_200": g200 <= np.quantile(g200, 0.05),
        "top5pct_absolute_cost_200": c200 >= np.quantile(c200, 0.95),
    }
    groups = {
        name: {
            **subset_breakdown(mask, data, selection),
            "initial_cost": distribution(c0[mask]),
            "round20_cost": distribution(c20[mask]),
            "round100_cost": distribution(c100[mask]),
            "round200_cost": distribution(c200[mask]),
            "round200_gain": distribution(g200[mask]),
            "warm_cost": distribution(warm_cost[mask]),
            "teacher_cost": distribution(teacher_cost[mask]),
        }
        for name, mask in masks.items()
    }

    # Tail persistence within each seed and agreement across seeds.
    tail_masks = np.zeros((4, 3, len(selection)), bool)
    for budget_index in range(1, 4):
        for seed in range(3):
            threshold = np.quantile(per_seed_gain[budget_index, seed], 0.05)
            tail_masks[budget_index, seed] = per_seed_gain[budget_index, seed] <= threshold
    overlaps = {}
    for seed in range(3):
        overlaps[str(seed)] = {
            "tail20_tail100_jaccard": float(np.sum(tail_masks[1, seed] & tail_masks[2, seed]) / np.sum(tail_masks[1, seed] | tail_masks[2, seed])),
            "tail100_tail200_jaccard": float(np.sum(tail_masks[2, seed] & tail_masks[3, seed]) / np.sum(tail_masks[2, seed] | tail_masks[3, seed])),
            "tail20_tail200_jaccard": float(np.sum(tail_masks[1, seed] & tail_masks[3, seed]) / np.sum(tail_masks[1, seed] | tail_masks[3, seed])),
            "in_all_three_tails": int(np.sum(tail_masks[1, seed] & tail_masks[2, seed] & tail_masks[3, seed])),
        }

    # Component attribution for the 200-round negative tail.
    component_names = (
        "position", "yaw", "vx", "yawrate", "acceleration_rate", "steering_rate", "total"
    )
    component_delta = {name: np.empty((3, len(selection)), np.float32) for name in component_names}
    component_replay_error = []
    for seed in range(3):
        initial_components = action_cost_components(
            backend, weights, params, action[0, seed], states, current,
            references, selection, args.batch_size, device,
        )
        final_components = action_cost_components(
            backend, weights, params, action[3, seed], states, current,
            references, selection, args.batch_size, device,
        )
        component_replay_error.extend((
            float(np.max(np.abs(initial_components["total"] - cost[0, seed]))),
            float(np.max(np.abs(final_components["total"] - cost[3, seed]))),
        ))
        for name in component_names:
            component_delta[name][seed] = final_components[name] - initial_components[name]

    true_gain_200 = per_seed_gain[3]
    negative = true_gain_200 < 0
    severe = true_gain_200 <= -10
    component_summary = {}
    for group_name, mask in (("all_regressions", negative), ("severe_regressions", severe)):
        component_summary[group_name] = {
            name: distribution(component_delta[name][mask])
            for name in component_names
        }
        stacked = np.stack([component_delta[name] for name in component_names[:-1]], axis=-1)
        dominant = np.argmax(stacked, axis=-1)
        component_summary[group_name]["dominant_positive_component_counts"] = {
            name: int(np.sum(mask & (dominant == index)))
            for index, name in enumerate(component_names[:-1])
        }

    # Action displacement: physical channel order is [acceleration, steering].
    sigma = np.asarray(params.noise_sigma, np.float32).reshape(1, 1, 1, 2)
    delta_sigma = (action[3] - action[0]) / sigma
    action_summary = {}
    for group_name, mask in (("all", np.ones_like(negative)), ("regressions", negative), ("severe_regressions", severe)):
        action_summary[group_name] = {
            "rms_sigma": distribution(np.sqrt(np.mean(delta_sigma[mask].reshape(-1, 16) ** 2, axis=1))),
            "early_steering_abs_sigma": distribution(np.mean(np.abs(delta_sigma[:, :, :3, 1][mask[..., None].repeat(3, axis=-1)].reshape(-1, 3)), axis=1)),
            "late_steering_abs_sigma": distribution(np.mean(np.abs(delta_sigma[:, :, 3:, 1][mask[..., None].repeat(5, axis=-1)].reshape(-1, 5)), axis=1)),
            "acceleration_abs_sigma": distribution(np.mean(np.abs(delta_sigma[..., 0][mask[..., None].repeat(8, axis=-1)].reshape(-1, 8)), axis=1)),
        }

    # Does the final Critic think the harmful Actor movement is beneficial?
    critic_predicted_gain = np.empty((3, len(selection)), np.float32)
    for seed in range(3):
        root = runs[200] / f"seed_{seed}"
        critic1, payload1, _ = load_value(root / "critic1.pt", root / "critic1.pt", device)
        critic2, payload2, _ = load_value(root / "critic2.pt", root / "critic2.pt", device)
        critic_inputs1 = critic_state_inputs(data, payload1)
        critic_inputs2 = critic_state_inputs(data, payload2)
        q0 = np.maximum(
            predict_actions(critic1, critic_inputs1, payload1, selection, action[0, seed], device),
            predict_actions(critic2, critic_inputs2, payload2, selection, action[0, seed], device),
        )
        q200 = np.maximum(
            predict_actions(critic1, critic_inputs1, payload1, selection, action[3, seed], device),
            predict_actions(critic2, critic_inputs2, payload2, selection, action[3, seed], device),
        )
        critic_predicted_gain[seed] = q0 - q200

    critic_tail = {
        "all": {
            "predicted_gain": distribution(critic_predicted_gain.reshape(-1)),
            "sign_agreement": float(np.mean((critic_predicted_gain > 0) == (true_gain_200 > 0))),
        },
        "true_regressions": {
            "count": int(negative.sum()),
            "predicted_gain": distribution(critic_predicted_gain[negative]),
            "critic_predicts_improvement_fraction": float(np.mean(critic_predicted_gain[negative] > 0)),
        },
        "true_severe_regressions": {
            "count": int(severe.sum()),
            "predicted_gain": distribution(critic_predicted_gain[severe]),
            "critic_predicts_improvement_fraction": float(np.mean(critic_predicted_gain[severe] > 0)),
        },
    }

    # Top state cases use seed-mean paired costs; retain exact row identity.
    top_rows = np.argsort(g200)[:30]
    cases = []
    for local in top_rows:
        cases.append({
            "bank_index": int(selection[local]),
            "episode": str(data["episode"][selection[local]]),
            "snapshot": str(data["snapshot"][selection[local]]),
            "speed": float(data["speed"][selection[local]]),
            "scenario": str(data["scenario"][selection[local]]),
            "initial_cost_seed_mean": float(c0[local]),
            "round20_cost_seed_mean": float(c20[local]),
            "round100_cost_seed_mean": float(c100[local]),
            "round200_cost_seed_mean": float(c200[local]),
            "round200_gain_seed_mean": float(g200[local]),
            "warm_cost": float(warm_cost[local]),
            "teacher_cost": float(teacher_cost[local]),
            "regressing_seed_count": int(np.sum(true_gain_200[:, local] < 0)),
            "critic_predicts_improvement_seed_count": int(np.sum(critic_predicted_gain[:, local] > 0)),
        })

    negative_loss = np.maximum(-true_gain_200, 0)
    flat_loss = np.sort(negative_loss.reshape(-1))[::-1]
    total_negative_loss = float(flat_loss.sum())
    concentration = {
        "negative_gain_sum": total_negative_loss,
        "top_1pct_share": float(flat_loss[:18].sum() / max(total_negative_loss, 1e-12)),
        "top_5pct_share": float(flat_loss[:90].sum() / max(total_negative_loss, 1e-12)),
        "top_10pct_share": float(flat_loss[:180].sum() / max(total_negative_loss, 1e-12)),
        "positive_gain_sum": float(np.maximum(true_gain_200, 0).sum()),
    }

    checks = {
        "selection_count_600": len(selection) == 600,
        "component_replay_max_error_le_1e_3": max(component_replay_error) <= 1e-3,
        "formal_validation_sealed": all(not c.get("formal_validation_loaded") for c in contracts.values()),
        "test_sealed": all(not c.get("test_loaded") for c in contracts.values()),
    }
    if not all(checks.values()):
        raise AssertionError(checks)

    manifest = {
        "runs": {
            str(budget): {
                "path": str(run.resolve()),
                "contract_sha256": sha256_file(run / "contract.json"),
                "summary_sha256": sha256_file(run / "summary.json"),
                "validator_sha256": sha256_file(run / "validator_report.json"),
            }
            for budget, run in runs.items()
        },
        "candidate_bank": str((bank_root / "candidate_bank.npz").resolve()),
        "candidate_bank_sha256": sha256_file(bank_root / "candidate_bank.npz"),
        "normalization_source": str(normalization_source.resolve()),
        "actor_checkpoints": action_paths,
        "selection_episodes": reference["internal_selection_episodes"],
        "cost_weights": asdict(weights),
        "mppi_params": asdict(params),
    }
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "OAC2_200ROUND_MEDIAN_IMPROVES_BUT_PERSISTENT_POSITION_TAIL_WORSENS",
        "manifest": manifest,
        "budget_summary": budget_summary,
        "state_level_groups": groups,
        "tail_overlap_by_seed": overlaps,
        "negative_loss_concentration": concentration,
        "cost_component_delta_final_minus_initial": component_summary,
        "action_displacement_final_minus_initial": action_summary,
        "final_critic_tail_diagnostic": critic_tail,
        "worst_30_state_mean_cases": cases,
        "checks": checks,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    np.savez_compressed(
        args.output_dir / "evaluation.npz",
        selection_index=selection,
        episode=data["episode"][selection],
        snapshot=data["snapshot"][selection],
        speed=data["speed"][selection],
        scenario=data["scenario"][selection],
        action=action,
        cost=cost,
        warm_cost=warm_cost.astype(np.float32),
        teacher_cost=teacher_cost.astype(np.float32),
        critic_predicted_gain=critic_predicted_gain,
        **{f"component_delta_{name}": value for name, value in component_delta.items()},
    )
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "qualification": result["qualification"],
        "budget_summary": budget_summary,
        "state_level_groups": groups,
        "tail_overlap_by_seed": overlaps,
        "negative_loss_concentration": concentration,
        "cost_component_delta": component_summary,
        "action_displacement": action_summary,
        "critic_tail": critic_tail,
        "worst_10": cases[:10],
        "checks": checks,
    }, indent=2))


if __name__ == "__main__":
    main()
