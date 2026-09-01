#!/usr/bin/env python3
"""State-wise diagnosis of unresolved OAC-2A adaptive-tail regressions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from analyze_mppi_oac2_budget_tail_cases import action_cost_components, distribution
from generate_dbm_proposal_teacher import sha256_file
from run_mppi_absolute_action_value_critic_cv import make_folds
from train_mppi_oac2_continuous_actor import load_actor, load_value
from train_mppi_online_absolute_sac import (
    actor_mean,
    critic_state_inputs,
    load_actor_normalization,
    load_bank,
    make_actor_inputs,
    predict_actions,
    rollout_bank,
)
from validate_mppi_oac2_continuous_actor import load_actor_checkpoint


DEFAULT_RUN = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_box3_adaptive_tail_200round_20260825_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_adaptive_tail_states_20260825_v2"
)
ROLES = ("initial", "selected", "latest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float64).reshape(-1)
    right = np.asarray(right, np.float64).reshape(-1)
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def classification(score: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    score = np.asarray(score).reshape(-1)
    target = np.asarray(target, bool).reshape(-1)
    predicted = score > 0
    return {
        "positive_count": int(target.sum()),
        "active_count": int(predicted.sum()),
        "recall": float(np.mean(predicted[target])) if np.any(target) else 0.0,
        "false_positive_rate": float(np.mean(predicted[~target])) if np.any(~target) else 0.0,
        "precision": float(np.mean(target[predicted])) if np.any(predicted) else 0.0,
        "roc_auc": float(roc_auc_score(target, score)) if len(np.unique(target)) == 2 else 0.5,
    }


def slice_rates(mask: np.ndarray, data: dict[str, np.ndarray], selection: np.ndarray) -> dict[str, Any]:
    speed = data["speed"][selection]
    scenario = data["scenario"][selection]
    # A state is counted once if any seed has the condition.
    state_mask = np.any(mask, axis=0)
    return {
        "seed_state_count": int(mask.sum()),
        "state_any_seed_count": int(state_mask.sum()),
        "by_speed": {
            str(float(value)): {
                "state_count": int(np.sum(state_mask & np.isclose(speed, value))),
                "rate": float(np.mean(state_mask[np.isclose(speed, value)])),
            }
            for value in sorted(np.unique(speed))
        },
        "by_scenario": {
            str(value): {
                "state_count": int(np.sum(state_mask & (scenario == value))),
                "rate": float(np.mean(state_mask[scenario == value])),
            }
            for value in sorted(np.unique(scenario))
        },
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    device = torch.device(args.device)
    contract = json.loads((args.run_dir / "contract.json").read_text())
    summary = json.loads((args.run_dir / "summary.json").read_text())
    validator = json.loads((args.run_dir / "validator_report.json").read_text())
    arguments = contract["arguments"]
    if arguments["tail_constraint_mode"] != "adaptive":
        raise AssertionError("run is not adaptive-tail")
    margin = float(arguments["tail_regression_margin_log"])
    bank_root = Path(arguments["bank_root"])
    actor_root = Path(arguments["actor_root"])
    data = load_bank(bank_root)
    folds = make_folds(data, 3)
    outer_fold = int(contract["outer_fold"])
    selection = folds[outer_fold]
    if sorted(np.unique(data["episode"][selection]).tolist()) != sorted(
        contract["internal_selection_episodes"]
    ):
        # make_folds returns the outer fold, while OAC internal selection is a
        # separate subset of the complement; use the registered episodes.
        selection = np.flatnonzero(np.isin(
            data["episode"], contract["internal_selection_episodes"]
        ))
    if len(selection) != 600:
        raise AssertionError(f"expected 600 states, got {len(selection)}")
    states, current, references, params_json, weights_json, dbm_json = load_rollout_inputs(
        data, Path(arguments["gt_v1"])
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    normalization, _ = load_actor_normalization(Path(arguments["base_ac"]))
    actor_inputs = make_actor_inputs(data, normalization)

    actions = np.empty((3, 3, len(selection), 8, 2), np.float32)
    costs = np.empty((3, 3, len(selection)), np.float32)
    predicted_delta = np.empty((3, len(selection)), np.float32)
    checkpoints = []
    component_delta = {
        name: np.empty((3, len(selection)), np.float32)
        for name in ("position", "yaw", "vx", "yawrate", "acceleration_rate", "steering_rate", "total")
    }
    component_errors = []
    for seed in range(3):
        seed_dir = args.run_dir / f"seed_{seed}"
        initial_path = actor_root / f"a0_fold{outer_fold}_seed{seed}.pt"
        actors = [
            load_actor(initial_path, outer_fold, seed, device, 3.0)[0],
            load_actor_checkpoint(seed_dir / "actor_selected.pt", device)[0],
            load_actor_checkpoint(seed_dir / "actor_latest.pt", device)[0],
        ]
        paths = [initial_path, seed_dir / "actor_selected.pt", seed_dir / "actor_latest.pt"]
        for role_index, (actor, path) in enumerate(zip(actors, paths)):
            actor.eval()
            actions[role_index, seed] = actor_mean(actor, actor_inputs, selection, device)
            costs[role_index, seed] = rollout_bank(
                backend, weights, params, actions[role_index, seed, :, None],
                states, current, references, selection, args.batch_size, device,
            )[:, 0]
            checkpoints.append({
                "seed": seed, "role": ROLES[role_index],
                "path": str(path.resolve()), "sha256": sha256_file(path),
            })
        critic1, payload1, _ = load_value(
            seed_dir / "critic1.pt", seed_dir / "critic1.pt", device
        )
        critic2, payload2, _ = load_value(
            seed_dir / "critic2.pt", seed_dir / "critic2.pt", device
        )
        inputs1 = critic_state_inputs(data, payload1)
        inputs2 = critic_state_inputs(data, payload2)
        q = []
        for role_index in (1, 2):
            q.append(np.maximum(
                predict_actions(critic1, inputs1, payload1, selection, actions[role_index, seed], device),
                predict_actions(critic2, inputs2, payload2, selection, actions[role_index, seed], device),
            ))
        predicted_delta[seed] = q[1] - q[0]
        initial_components = action_cost_components(
            backend, weights, params, actions[0, seed], states, current,
            references, selection, args.batch_size, device,
        )
        latest_components = action_cost_components(
            backend, weights, params, actions[2, seed], states, current,
            references, selection, args.batch_size, device,
        )
        component_errors.extend((
            float(np.max(np.abs(initial_components["total"] - costs[0, seed]))),
            float(np.max(np.abs(latest_components["total"] - costs[2, seed]))),
        ))
        for name in component_delta:
            component_delta[name][seed] = latest_components[name] - initial_components[name]

    gain_initial = costs[0] - costs[2]
    gain_selected = costs[1] - costs[2]
    true_log_delta = np.log1p(costs[2]) - np.log1p(costs[1])
    predicted_excess = predicted_delta - margin
    regress_initial = gain_initial < 0
    severe_initial = gain_initial <= -10
    regress_selected = gain_selected < 0
    material_selected = true_log_delta > margin
    score = predicted_excess

    selected_bad = costs[1] > costs[0]
    latest_bad = costs[2] > costs[0]
    transition_masks = {
        "persistent_bad": selected_bad & latest_bad,
        "new_bad_after_selected": (~selected_bad) & latest_bad,
        "recovered_by_latest": selected_bad & (~latest_bad),
        "good_at_selected_and_latest": (~selected_bad) & (~latest_bad),
    }

    sigma = np.asarray(params.noise_sigma, np.float32).reshape(1, 1, 1, 2)
    displacement = (actions[2] - actions[0]) / sigma
    bad = regress_initial
    good = ~bad
    component_attribution = {}
    for label, mask in (("regress_initial", bad), ("severe_initial", severe_initial), ("improve_initial", good)):
        component_attribution[label] = {
            name: distribution(value[mask]) for name, value in component_delta.items()
        }
        positive = np.stack([
            component_delta[name] for name in (
                "position", "yaw", "vx", "yawrate", "acceleration_rate", "steering_rate"
            )
        ], axis=-1)
        dominant = np.argmax(positive, axis=-1)
        names = ("position", "yaw", "vx", "yawrate", "acceleration_rate", "steering_rate")
        component_attribution[label]["dominant_counts"] = {
            name: int(np.sum(mask & (dominant == index))) for index, name in enumerate(names)
        }

    movement = {}
    for label, mask in (("regress_initial", bad), ("severe_initial", severe_initial), ("improve_initial", good)):
        flat = displacement[mask].reshape(-1, 8, 2)
        movement[label] = {
            "full_rms_sigma": distribution(np.sqrt(np.mean(flat.reshape(-1, 16) ** 2, axis=1))),
            "early_steering_abs_sigma": distribution(np.mean(np.abs(flat[:, :3, 1]), axis=1)),
            "late_steering_abs_sigma": distribution(np.mean(np.abs(flat[:, 3:, 1]), axis=1)),
            "acceleration_abs_sigma": distribution(np.mean(np.abs(flat[:, :, 0]), axis=1)),
        }

    transition_analysis = {}
    for label, mask in transition_masks.items():
        normalized_movement = displacement[mask].reshape(-1, 8, 2)
        latest_minus_initial = (costs[2] - costs[0])[mask]
        transition_analysis[label] = {
            "seed_state_count": int(mask.sum()),
            "state_any_seed_count": int(np.any(mask, axis=0).sum()),
            "initial_cost": distribution(costs[0][mask]),
            "latest_minus_initial_cost": distribution(latest_minus_initial),
            "critic_tail_active_fraction": float(np.mean(score[mask] > 0)),
            "critic_excess_score": distribution(score[mask]),
            "position_delta": distribution(component_delta["position"][mask]),
            "early_steering_abs_sigma": distribution(
                np.mean(np.abs(normalized_movement[:, :3, 1]), axis=1)
            ),
        }

    seed_consistency = {
        "regress_vs_initial_state_count_by_seed_count": {
            str(count): int(np.sum(np.sum(regress_initial, axis=0) == count))
            for count in range(4)
        },
        "severe_vs_initial_state_count_by_seed_count": {
            str(count): int(np.sum(np.sum(severe_initial, axis=0) == count))
            for count in range(4)
        },
    }

    signal = {
        "predicted_vs_true_log_delta_pearson": correlation(predicted_delta, true_log_delta),
        "regress_vs_selected": classification(score, regress_selected),
        "material_regress_vs_selected": classification(score, material_selected),
        "regress_vs_initial": classification(score, regress_initial),
        "severe_regress_vs_initial": classification(score, severe_initial),
        "predicted_excess": distribution(score.reshape(-1)),
        "true_log_delta": distribution(true_log_delta.reshape(-1)),
    }
    unresolved = {
        "regress_vs_initial": slice_rates(regress_initial, data, selection),
        "severe_vs_initial": slice_rates(severe_initial, data, selection),
        "regress_vs_selected": slice_rates(regress_selected, data, selection),
    }
    mean_gain = gain_initial.mean(axis=0)
    worst_rows = np.argsort(mean_gain)[:30]
    cases = []
    for local in worst_rows:
        cases.append({
            "bank_index": int(selection[local]),
            "episode": str(data["episode"][selection[local]]),
            "snapshot": str(data["snapshot"][selection[local]]),
            "speed": float(data["speed"][selection[local]]),
            "scenario": str(data["scenario"][selection[local]]),
            "gain_vs_initial_seed_mean": float(mean_gain[local]),
            "regressing_seed_count": int(regress_initial[:, local].sum()),
            "severe_seed_count": int(severe_initial[:, local].sum()),
            "critic_tail_active_seed_count": int((score[:, local] > 0).sum()),
            "position_delta_seed_mean": float(component_delta["position"][:, local].mean()),
            "yaw_delta_seed_mean": float(component_delta["yaw"][:, local].mean()),
            "vx_delta_seed_mean": float(component_delta["vx"][:, local].mean()),
            "early_steering_abs_sigma_seed_mean": float(np.mean(np.abs(displacement[:, local, :3, 1]))),
        })

    checks = {
        "source_validator_pass": bool(validator["passed"]),
        "selection_count_600": len(selection) == 600,
        "component_replay_error_le_1e_3": max(component_errors) <= 1e-3,
        "formal_validation_sealed": not bool(summary["formal_validation_loaded"]),
        "test_sealed": not bool(summary["test_loaded"]),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    result = {
        "format_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "ADAPTIVE_TAIL_PARTIAL_REPAIR_GLOBAL_AGGREGATE_MISSES_LOW_HEADROOM_AND_HIGH_SPEED_LEVERAGE_TAIL",
        "manifest": {
            "run": str(args.run_dir.resolve()),
            "contract_sha256": sha256_file(args.run_dir / "contract.json"),
            "summary_sha256": sha256_file(args.run_dir / "summary.json"),
            "validator_sha256": sha256_file(args.run_dir / "validator_report.json"),
            "checkpoints": checkpoints,
            "selection_episodes": contract["internal_selection_episodes"],
            "tail_margin_log": margin,
        },
        "signal": signal,
        "unresolved_tail": unresolved,
        "component_attribution": component_attribution,
        "action_displacement": movement,
        "selected_to_latest_transitions": transition_analysis,
        "cross_seed_consistency": seed_consistency,
        "mechanism_decision": [
            "critic_ranking_is_informative_but_the_fixed_margin_only_activates_for_a_subset_of_regressions",
            "continued_training_repairs_many_selected_checkpoint_regressions_but_also_creates_new_regressions",
            "persistent_and_new_regressions_are_low_headroom_states_where_small_moves_are_not_reliably_beneficial",
            "catastrophic_severity_is_high_speed_and_position_cost_dominated_with_early_steering_leverage",
            "global_topk_mean_constraint_does_not_provide_a_per_state_non_regression_guarantee",
        ],
        "worst_30_seed_mean_cases": cases,
        "checks": checks,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(
        args.output_dir / "evaluation.npz",
        selection_index=selection,
        episode=data["episode"][selection], snapshot=data["snapshot"][selection],
        speed=data["speed"][selection], scenario=data["scenario"][selection],
        action=actions, cost=costs, predicted_log_delta=predicted_delta,
        true_log_delta=true_log_delta,
        **{f"component_delta_{name}": value for name, value in component_delta.items()},
    )
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "qualification": result["qualification"],
        "signal": signal,
        "unresolved_tail": unresolved,
        "component_attribution": component_attribution,
        "action_displacement": movement,
        "selected_to_latest_transitions": transition_analysis,
        "cross_seed_consistency": seed_consistency,
        "worst_10": cases[:10],
        "checks": checks,
    }, indent=2))


if __name__ == "__main__":
    main()
