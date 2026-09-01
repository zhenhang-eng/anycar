#!/usr/bin/env python3
"""Independently validate an OAC-2 continuous contextual-bandit run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from generate_dbm_proposal_teacher import sha256_file
from mppi_a2_actors import (
    DirectNoAnchorGTXActor,
    DirectNoAnchorGTXSupportActor,
    OAC_DEFAULT_OUTPUT_SUPPORT_MULTIPLIER,
)
from run_mppi_absolute_action_value_critic_cv import (
    AbsoluteActionValueCritic,
    make_folds,
)
from train_mppi_joint_value_gap_pilot import (
    ContinuousGapHead,
    evaluate as evaluate_critic,
)
from train_mppi_oac2_continuous_actor import (
    acceptance_gate,
    actor_learning_rate_for_round,
    evaluate_actor,
    internal_split,
    load_actor,
)
from train_mppi_online_absolute_sac import (
    ROLE_NAMES,
    critic_state_inputs,
    load_actor_normalization,
    load_bank,
    make_actor_inputs,
    module_digest,
    rollout_bank,
)


METRIC_RECOMPUTE_TOLERANCE = 2e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path, nargs="?",
        default=Path(
            "outputs/mppi_proposal/online_absolute_sac_oac2_fold1_20260824_v1"
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_actor_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device)
    state = payload["model_state_dict"]
    if "output_support_multiplier" in state:
        actor = DirectNoAnchorGTXSupportActor(
            support_multiplier=float(state["output_support_multiplier"]),
            dropout=0.0,
        ).to(device)
    else:
        actor = DirectNoAnchorGTXActor(dropout=0.0).to(device)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor, payload


def load_critic_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device)
    critic = AbsoluteActionValueCritic(dropout=0.0).to(device)
    critic.load_state_dict(payload["model"], strict=True)
    critic.eval()
    return critic, payload


def scalar_error(recomputed: dict, stored: dict) -> float:
    pairs = (
        (recomputed["cost"]["mean"], stored["cost"]["mean"]),
        (recomputed["cost"]["maximum"], stored["cost"]["maximum"]),
        (recomputed["gain_vs_initial"]["mean"], stored["gain_vs_initial"]["mean"]),
        (recomputed["gain_vs_initial"]["p05"], stored["gain_vs_initial"]["p05"]),
        (recomputed["gain_vs_initial"]["median"], stored["gain_vs_initial"]["median"]),
        (recomputed["gain_vs_initial"]["minimum"], stored["gain_vs_initial"]["minimum"]),
        (recomputed["headroom_recovery_vs_bank_best"], stored["headroom_recovery_vs_bank_best"]),
        (recomputed["by_speed"]["2.4"]["mean_gain"], stored["by_speed"]["2.4"]["mean_gain"]),
        (recomputed["by_speed"]["2.8"]["mean_gain"], stored["by_speed"]["2.8"]["mean_gain"]),
        (recomputed["two_center_guard"]["cost"]["mean"], stored["two_center_guard"]["cost"]["mean"]),
    )
    return max(abs(float(left) - float(right)) for left, right in pairs)


def main() -> None:
    cli = parse_args()
    device = torch.device(cli.device)
    contract = json.loads((cli.run_dir / "contract.json").read_text())
    summary = json.loads((cli.run_dir / "summary.json").read_text())
    arguments = contract["arguments"]
    args = SimpleNamespace(
        rollout_batch_size=int(arguments["rollout_batch_size"]),
        bootstrap_samples=int(arguments["bootstrap_samples"]),
        material_gap=float(arguments["material_gap"]),
        flat_gap=float(arguments["flat_gap"]),
        target_mode="move_coefficient",
    )
    bank_root = Path(arguments["bank_root"])
    actor_root = Path(arguments["actor_root"])
    base_ac = Path(arguments["base_ac"])
    gt_v1 = Path(arguments["gt_v1"])
    parent = Path(contract["parent_run"])
    outer_fold = int(contract["outer_fold"])
    data = load_bank(bank_root)
    folds = make_folds(data, 3)
    fit, selection, fit_episodes, selection_episodes = internal_split(
        data, folds, outer_fold
    )
    states, current, references, params_json, weights_json, dbm_json = (
        load_rollout_inputs(data, gt_v1)
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    normalization, _ = load_actor_normalization(base_ac)
    actor_inputs = make_actor_inputs(data, normalization)
    checks = {
        "contract": contract["qualification"] == "OAC2_CONTINUOUS_CONTEXTUAL_BANDIT_CONTRACT",
        "parent_summary_hash": sha256_file(parent / "summary.json") == contract["parent_summary_sha256"],
        "parent_validator_hash": sha256_file(parent / "validator_report.json") == contract["parent_validator_sha256"],
        "candidate_bank_hash": sha256_file(bank_root / "candidate_bank.npz") == contract["candidate_bank_sha256"],
        "internal_fit_episodes_match": fit_episodes == contract["fit_episodes"],
        "internal_selection_episodes_match": selection_episodes == contract["internal_selection_episodes"],
        "internal_split_disjoint": not bool(set(fit_episodes) & set(selection_episodes)),
        "formal_validation_sealed": not bool(summary["formal_validation_loaded"]),
        "test_sealed": not bool(summary["test_loaded"]),
        "registered_update_ratio": summary["critic_update_ratio"] == (
            f"{int(arguments['critic_updates_per_round'])}:"
            f"{int(arguments['actor_updates_per_round'])}"
        ),
    }
    multi_actor_contract = contract.get("multi_actor_update_contract")
    multi_actor_enabled = bool(arguments.get("multi_actor_update_pilot", False))
    actor_updates_per_round = int(arguments.get("actor_updates_per_round", 1))
    critic_updates_per_round = int(arguments.get("critic_updates_per_round", 20))
    aux_update_interval = int(arguments.get("aux_update_interval_rounds", 1))
    if multi_actor_contract is not None:
        checks["multi_actor_contract_matches_arguments"] = (
            bool(multi_actor_contract["enabled"]) == multi_actor_enabled
            and int(multi_actor_contract["actor_updates_per_round"])
            == actor_updates_per_round
            and int(multi_actor_contract["critic_updates_per_round"])
            == critic_updates_per_round
            and int(multi_actor_contract.get("aux_update_interval_rounds", 1))
            == aux_update_interval
            and bool(multi_actor_contract["warm_used_in_training"]) is False
        )
    refresh_contract = contract.get("actor_visited_refresh_contract")
    if refresh_contract is not None:
        checks["actor_visited_refresh_contract_matches_arguments"] = (
            bool(refresh_contract["enabled"])
            == bool(arguments.get("actor_visited_refresh_pilot", False))
            and int(refresh_contract["contexts_per_round"])
            == int(arguments["contexts_per_round"])
            and int(refresh_contract["rounds"]) == int(arguments["rounds"])
            and int(refresh_contract["total_context_visits"])
            == int(arguments["contexts_per_round"]) * int(arguments["rounds"])
            and int(refresh_contract["total_critic_updates"])
            == critic_updates_per_round * int(arguments["rounds"])
            and int(refresh_contract["total_actor_updates"])
            == actor_updates_per_round * int(arguments["rounds"])
            and int(refresh_contract["aux_update_interval_rounds"])
            == aux_update_interval
        )
    support_multiplier = arguments.get("actor_output_support_multiplier")
    support_contract = contract.get("output_support_contract")
    if support_multiplier is not None:
        support_multiplier = float(support_multiplier)
        checks["support_multiplier_registered"] = support_multiplier in {
            1.0, OAC_DEFAULT_OUTPUT_SUPPORT_MULTIPLIER,
        }
        checks["box1_explicit_ablation"] = (
            support_multiplier == OAC_DEFAULT_OUTPUT_SUPPORT_MULTIPLIER
            or bool(arguments.get("allow_box1_ablation", False))
            or support_contract is None  # Compatibility with completed pre-contract A/B runs.
        )
    if support_contract is not None:
        checks["support_contract_default_box3"] = (
            float(support_contract["default_multiplier"])
            == OAC_DEFAULT_OUTPUT_SUPPORT_MULTIPLIER
        )
        checks["support_contract_matches_arguments"] = (
            float(support_contract["active_multiplier"])
            == float(support_multiplier)
        )
        checks["support_checkpoint_buffer_required"] = bool(
            support_contract["checkpoint_multiplier_buffer_required"]
        )
    tail_mode = arguments.get("tail_constraint_mode", "fixed")
    tail_contract = contract.get("tail_constraint_contract")
    if tail_contract is not None:
        checks["tail_contract_matches_arguments"] = (
            tail_contract["mode"] == tail_mode
            and float(tail_contract["material_margin_log"])
            == float(arguments.get("tail_regression_margin_log", 0.0))
            and float(tail_contract["budget_log"])
            == float(arguments.get("tail_cvar_budget_log", 0.0))
        )
    risk_contract = contract.get("statewise_risk_shrink_contract")
    if risk_contract is not None:
        checks["statewise_risk_shrink_contract_matches_arguments"] = (
            bool(risk_contract["enabled"])
            == (float(arguments.get("statewise_risk_shrink_weight", 0.0)) > 0.0)
            and float(risk_contract["weight"])
            == float(arguments.get("statewise_risk_shrink_weight", 0.0))
            and float(risk_contract["temperature_log"])
            == float(arguments.get("statewise_risk_temperature_log", 0.02))
            and risk_contract.get("form", "squared_raw")
            == arguments.get("statewise_risk_shrink_form", "squared_raw")
        )
    aggregation_contract = contract.get("actor_cost_aggregation_contract")
    if aggregation_contract is not None:
        checks["actor_cost_aggregation_contract_matches_arguments"] = (
            float(aggregation_contract["gamma"])
            == float(arguments.get("actor_cost_weight_gamma", 0.0))
            and float(aggregation_contract["weight_cap"])
            == float(arguments.get("actor_cost_weight_maximum", 2048.0))
            and aggregation_contract["critic_target"] == "log1p(J50)"
            and aggregation_contract["batch_normalization"]
            == "divide detached weights by batch mean"
        )
    gradient_source = arguments.get("actor_gradient_source", "critic")
    objective_mode = arguments.get("actor_objective_mode", "stochastic_sac")
    gradient_source_contract = contract.get("actor_gradient_source_contract")
    if gradient_source_contract is not None:
        checks["actor_gradient_source_contract_matches_arguments"] = (
            gradient_source in {"critic", "dbm"}
            and gradient_source_contract["source"] == gradient_source
            and gradient_source_contract.get("objective_mode", "stochastic_sac")
            == objective_mode
            and bool(gradient_source_contract["matched_pilot"])
            == bool(arguments.get("matched_gradient_source_pilot", False))
            and bool(gradient_source_contract.get("smoke_only", False))
            == bool(arguments.get("matched_gradient_source_smoke", False))
            and bool(gradient_source_contract["critic_and_replay_continue_in_dbm_arm"])
            and bool(gradient_source_contract["coefficient_is_detached_in_both_arms"])
            and not bool(gradient_source_contract["formal_validation_loaded"])
            and not bool(gradient_source_contract["test_loaded"])
        )
        if objective_mode == "deterministic_center_dbm":
            direct = gradient_source_contract.get("deterministic_center_dbm", {})
            checks["deterministic_center_objective_contract"] = (
                gradient_source == "dbm"
                and bool(arguments.get("matched_gradient_source_pilot", False))
                and direct.get("actor_value_term")
                == "mean deterministic raw DBM J50 at Actor mean"
                and direct.get("sampled_action_affects_actor_loss") is False
                and direct.get("move_coefficient_affects_actor_loss") is False
                and direct.get("entropy_affects_actor_loss") is False
                and direct.get("tail_affects_actor_loss") is False
                and direct.get("trust_affects_actor_loss") is True
                and direct.get("replay_exploration_continues") is True
                and direct.get("twin_critic_training_continues") is True
            )
    learning_rate_contract = contract.get("actor_learning_rate_contract")
    if learning_rate_contract is not None:
        schedule = arguments.get("actor_learning_rate_schedule", "constant")
        checks["actor_learning_rate_contract_matches_arguments"] = (
            learning_rate_contract["schedule"] == schedule
            and float(learning_rate_contract["final"])
            == float(arguments["actor_learning_rate"])
            and int(learning_rate_contract["total_rounds"])
            == int(arguments["rounds"])
            and float(learning_rate_contract["action_space_projection_sigma_rms"])
            == float(arguments["max_step_sigma_rms"])
        )
    records = []
    for source in summary["records"]:
        seed = int(source["seed"])
        seed_dir = cli.run_dir / f"seed_{seed}"
        if support_multiplier is None:
            initial_actor, _ = load_actor_checkpoint(
                actor_root / f"a0_fold{outer_fold}_seed{seed}.pt", device
            )
        else:
            initial_actor, _ = load_actor(
                actor_root / f"a0_fold{outer_fold}_seed{seed}.pt",
                outer_fold, seed, device, float(support_multiplier),
            )
        latest_actor, latest_payload = load_actor_checkpoint(
            seed_dir / "actor_latest.pt", device
        )
        selected_actor, selected_payload = load_actor_checkpoint(
            seed_dir / "actor_selected.pt", device
        )
        initial_metrics, initial_cost = evaluate_actor(
            args, initial_actor, actor_inputs, selection, None, data, states,
            current, references, backend, weights, params, device, 26082420 + seed,
        )
        latest_metrics, latest_cost = evaluate_actor(
            args, latest_actor, actor_inputs, selection, initial_cost, data,
            states, current, references, backend, weights, params, device,
            26082500 + seed * 100 + int(arguments["rounds"]),
        )
        selected_round = int(source["selected_round"])
        selected_metrics, selected_cost = evaluate_actor(
            args, selected_actor, actor_inputs, selection, initial_cost, data,
            states, current, references, backend, weights, params, device,
            26082500 + seed * 100 + selected_round,
        )
        with np.load(seed_dir / "actor_visited_replay.npz", allow_pickle=False) as loaded:
            replay = {key: np.asarray(loaded[key]) for key in loaded.files}
        new_mask = replay["round"] >= 11
        expected_new = (
            int(arguments["rounds"])
            * int(arguments["contexts_per_round"])
            * len(ROLE_NAMES)
        )
        new_roles = replay["role"][new_mask]
        audit_rows = np.flatnonzero(new_mask)
        audit_rows = audit_rows[np.linspace(0, len(audit_rows) - 1, 96, dtype=np.int64)]
        audit_cost = rollout_bank(
            backend, weights, params, replay["action"][audit_rows, None],
            states, current, references, replay["state_index"][audit_rows],
            args.rollout_batch_size, device,
        )[:, 0]
        critic1, payload1 = load_critic_checkpoint(seed_dir / "critic1.pt", device)
        critic2, payload2 = load_critic_checkpoint(seed_dir / "critic2.pt", device)
        gap_payload = torch.load(seed_dir / "move_coefficient_head.pt", map_location=device)
        gap_head = ContinuousGapHead(output_mode="move_coefficient").to(device)
        gap_head.load_state_dict(gap_payload["model"], strict=True)
        critic_recomputed = evaluate_critic(
            args, data, folds, replay, critic1, payload1, critic2, payload2,
            gap_head, device, outer_fold,
        )
        # When no candidate checkpoint was accepted, selected is exactly the
        # initial Actor.  CUDA reduction jitter around zero must not flip the
        # semantic gate during independent replay.
        gate_metrics = source["selected_metrics"] if selected_round == 0 else selected_metrics
        final_gate = {
            "selected_mean_gain_positive": gate_metrics["gain_vs_initial"]["mean"] > 0.0,
            "selected_ci_lower_positive": gate_metrics["gain_episode_bootstrap_ci95"][0] > 0.0,
            "selected_median_gain_positive": gate_metrics["gain_vs_initial"]["median"] > 0.0,
            "speed_2_4_nonnegative": gate_metrics["by_speed"]["2.4"]["mean_gain"] >= 0.0,
            "speed_2_8_nonnegative": gate_metrics["by_speed"]["2.8"]["mean_gain"] >= 0.0,
            "critic_pair_ge_0_85": critic_recomputed["actor_visited_material_pair_accuracy"] >= 0.85,
            "finite_and_not_saturated": selected_metrics["state_any_saturation_fraction"] <= 0.10,
        }
        p05_floor = arguments.get("selection_gain_p05_floor")
        speed24_floor = arguments.get("selection_speed_2_4_gain_p05_floor")
        speed28_floor = arguments.get("selection_speed_2_8_gain_p05_floor")
        if p05_floor is not None:
            final_gate["selected_gain_p05_above_floor"] = (
                gate_metrics["gain_vs_initial"]["p05"] >= float(p05_floor)
            )
        if speed24_floor is not None:
            final_gate["selected_speed_2_4_gain_p05_above_floor"] = (
                gate_metrics["by_speed"]["2.4"]["p05_gain"]
                >= float(speed24_floor)
            )
        if speed28_floor is not None:
            final_gate["selected_speed_2_8_gain_p05_above_floor"] = (
                gate_metrics["by_speed"]["2.8"]["p05_gain"]
                >= float(speed28_floor)
            )
        seed_checks = {
            "initial_actor_source_hash": sha256_file(
                actor_root / f"a0_fold{outer_fold}_seed{seed}.pt"
            ) == source["initial_actor_checkpoint_sha256"],
            "initial_actor_module_hash": module_digest(initial_actor) == source["initial_actor_module_sha256"],
            "latest_actor_module_hash": module_digest(latest_actor) == source["latest_actor_module_sha256"],
            "selected_actor_module_hash": module_digest(selected_actor) == source["selected_actor_module_sha256"],
            "actor_update_count": int(latest_payload["actor_update_count"]) == (
                int(arguments["rounds"]) * actor_updates_per_round
            ),
            "critic_update_count": int(latest_payload["critic_additional_update_count"]) == (
                int(arguments["rounds"]) * critic_updates_per_round
            ),
            "selected_round_checkpoint": int(selected_payload["selected_round"]) == selected_round,
            "new_replay_rows": int(new_mask.sum()) == expected_new,
            "new_replay_fit_only": bool(np.all(np.isin(replay["state_index"][new_mask], fit))),
            "new_replay_selection_excluded": not bool(np.any(np.isin(replay["state_index"][new_mask], selection))),
            "all_roles_complete": all(int(np.sum(new_roles == role)) == expected_new // len(ROLE_NAMES) for role in ROLE_NAMES),
            "dbm_replay_error_le_1e_4": float(np.max(np.abs(audit_cost - replay["cost"][audit_rows]))) <= 1e-4,
            "initial_metrics_recompute": scalar_error(initial_metrics, source["initial_metrics"]) <= METRIC_RECOMPUTE_TOLERANCE,
            "latest_metrics_recompute": scalar_error(latest_metrics, source["latest_metrics"]) <= METRIC_RECOMPUTE_TOLERANCE,
            "selected_metrics_recompute": scalar_error(selected_metrics, source["selected_metrics"]) <= METRIC_RECOMPUTE_TOLERANCE,
            "critic_pair_recompute": abs(
                critic_recomputed["actor_visited_material_pair_accuracy"]
                - source["critic_metrics"]["actor_visited_material_pair_accuracy"]
            ) <= 1e-7,
            "coefficient_gate_recompute": critic_recomputed["gap_heldout"]["fixed_gap_0_1"] == source["critic_metrics"]["gap_heldout"]["fixed_gap_0_1"],
            "seed_gates_recompute": final_gate == source["oac2_seed_gates"],
            "optimizers_serialized": all(
                "optimizer" in torch.load(seed_dir / name, map_location="cpu")
                for name in (
                    "actor_latest.pt", "critic1.pt", "critic2.pt",
                    "move_coefficient_head.pt",
                )
            ),
            "temperature_optimizer_serialized": "temperature_optimizer" in latest_payload,
            "formal_validation_sealed": not bool(latest_payload["formal_validation_loaded"]),
            "test_sealed": not bool(latest_payload["test_loaded"]),
        }
        if learning_rate_contract is not None:
            schedule_args = SimpleNamespace(**arguments)
            lr_divisor = (
                actor_updates_per_round if multi_actor_enabled else 1
            )
            seed_checks["actor_learning_rate_schedule_replay"] = all(
                abs(
                    float(round_record["actor"]["learning_rate"])
                    - actor_learning_rate_for_round(
                        schedule_args, int(round_record["round"])
                    ) / lr_divisor
                ) <= 1e-12
                for round_record in source["rounds"]
            )
            expected_final_lr = actor_learning_rate_for_round(
                schedule_args, int(arguments["rounds"])
            ) / lr_divisor
            actor_optimizer_groups = latest_payload["optimizer"]["param_groups"]
            seed_checks["actor_optimizer_final_lr_matches_schedule"] = all(
                abs(float(group["lr"]) - expected_final_lr) <= 1e-12
                for group in actor_optimizer_groups
            )
        if multi_actor_enabled:
            cumulative_limit = float(arguments["max_step_sigma_rms"])
            microstep_limit = cumulative_limit / actor_updates_per_round
            seed_checks["multi_actor_microstep_count"] = all(
                int(row["actor"]["microstep_count"]) == actor_updates_per_round
                and len(row["actor"]["microsteps"]) == actor_updates_per_round
                for row in source["rounds"]
            )
            seed_checks["critic_partition_per_round"] = all(
                sum(row["actor"]["critic_update_partition"])
                == critic_updates_per_round
                for row in source["rounds"]
            )
            seed_checks["microstep_trust_registered"] = all(
                abs(
                    float(row["actor"]["microstep_trust_sigma_rms"])
                    - microstep_limit
                ) <= 1e-12
                for row in source["rounds"]
            )
            seed_checks["round_cumulative_trust_respected"] = all(
                float(row["actor"]["round_cumulative_step_sigma_rms"])
                <= cumulative_limit + 5e-4
                for row in source["rounds"]
            )
            seed_checks["temperature_update_interval"] = all(
                sum(
                    bool(step["temperature_updated"])
                    for step in row["actor"]["microsteps"]
                ) == (
                    0 if objective_mode == "deterministic_center_dbm" else
                    (1 if int(row["round"]) % aux_update_interval == 0 else 0)
                )
                and bool(row["actor"].get("aux_update_due", True))
                == (int(row["round"]) % aux_update_interval == 0)
                for row in source["rounds"]
            )
            if gradient_source_contract is not None:
                seed_checks["actor_gradient_source_per_microstep"] = all(
                    step.get("actor_gradient_source") == gradient_source
                    for row in source["rounds"]
                    for step in row["actor"]["microsteps"]
                )
                seed_checks["dbm_gradient_diagnostics_finite"] = all(
                    (
                        np.isfinite(step["dbm_sampled_action_cost_mean"])
                        and np.isfinite(step["dbm_mean_action_cost_mean"])
                    ) if gradient_source == "dbm" else (
                        np.isnan(step["dbm_sampled_action_cost_mean"])
                        and np.isnan(step["dbm_mean_action_cost_mean"])
                    )
                    for row in source["rounds"]
                    for step in row["actor"]["microsteps"]
                )
                if objective_mode == "deterministic_center_dbm":
                    seed_checks["deterministic_center_loss_path_per_microstep"] = all(
                        step.get("actor_objective_mode")
                        == "deterministic_center_dbm"
                        and step.get("tail_affects_actor_loss") is False
                        and step.get("move_coefficient_affects_actor_loss") is False
                        and abs(float(step.get("effective_tail_weight", float("nan"))))
                        <= 1e-12
                        and bool(step["temperature_updated"]) is False
                        for row in source["rounds"]
                        for step in row["actor"]["microsteps"]
                    )
        if support_multiplier is not None:
            latest_state = latest_payload["model_state_dict"]
            selected_state = selected_payload["model_state_dict"]
            seed_checks["latest_support_buffer_matches_contract"] = (
                "output_support_multiplier" in latest_state
                and abs(
                    float(latest_state["output_support_multiplier"])
                    - float(support_multiplier)
                ) <= 1e-7
            )
            seed_checks["selected_support_buffer_matches_contract"] = (
                "output_support_multiplier" in selected_state
                and abs(
                    float(selected_state["output_support_multiplier"])
                    - float(support_multiplier)
                ) <= 1e-7
            )
        if tail_mode == "adaptive":
            decay = float(arguments["tail_dual_ema_decay"])
            learning_rate = float(arguments["tail_dual_learning_rate"])
            budget = float(arguments["tail_cvar_budget_log"])
            maximum = float(arguments["tail_dual_maximum"])
            expected_lagrange = float(arguments["tail_dual_initial"])
            expected_ema = None
            pending_observations = []
            recurrence_ok = True
            for round_record in source["rounds"]:
                actor_record = round_record["actor"]
                expected_used = (
                    0.0 if objective_mode == "deterministic_center_dbm"
                    else expected_lagrange
                )
                recurrence_ok &= abs(
                    float(actor_record["tail_lagrange_used"])
                    - expected_used
                ) <= 1e-7
                pending_observations.append(
                    float(actor_record["tail_regression_cvar"])
                )
                due = int(round_record["round"]) % aux_update_interval == 0
                recurrence_ok &= bool(actor_record.get("aux_update_due", True)) == due
                if due:
                    observed = float(np.mean(pending_observations))
                    pending_observations.clear()
                    recurrence_ok &= abs(
                        float(actor_record["tail_dual_observation"]) - observed
                    ) <= 1e-7
                    expected_ema = (
                        observed
                        if expected_ema is None
                        else decay * expected_ema + (1.0 - decay) * observed
                    )
                    expected_lagrange = float(np.clip(
                        expected_lagrange
                        + learning_rate * (expected_ema - budget),
                        0.0, maximum,
                    ))
                else:
                    recurrence_ok &= actor_record.get("tail_dual_observation") is None
                recurrence_ok &= abs(
                    float(actor_record["tail_lagrange_after"])
                    - expected_lagrange
                ) <= 1e-7
                recurrence_ok &= int(
                    actor_record.get("tail_cvar_window_count_after", 0)
                ) == len(pending_observations)
            seed_checks["tail_dual_recurrence"] = recurrence_ok
            seed_checks["tail_dual_checkpoint_matches"] = (
                abs(float(latest_payload["tail_lagrange"]) - expected_lagrange)
                <= 1e-7
                and abs(float(source["final_tail_lagrange"]) - expected_lagrange)
                <= 1e-7
                and abs(float(latest_payload["tail_cvar_ema"]) - expected_ema)
                <= 1e-7
            )
        passed = bool(all(seed_checks.values()))
        checks[f"seed_{seed}"] = passed
        records.append({
            "seed": seed,
            "passed": passed,
            "checks": seed_checks,
            "dbm_replay_max_abs_error": float(np.max(np.abs(audit_cost - replay["cost"][audit_rows]))),
            "initial_metric_max_abs_error": scalar_error(initial_metrics, source["initial_metrics"]),
            "latest_metric_max_abs_error": scalar_error(latest_metrics, source["latest_metrics"]),
            "selected_metric_max_abs_error": scalar_error(selected_metrics, source["selected_metrics"]),
            "oac2_seed_pass": bool(all(final_gate.values())),
            "selected_gate": final_gate,
        })
    passed_seed_count = sum(row["oac2_seed_pass"] for row in records)
    checks["passed_seed_count"] = passed_seed_count == int(summary["passed_seed_count"])
    required_validation_seeds = (
        1 if bool(arguments.get("matched_gradient_source_smoke", False)) else 2
    )
    checks["validator_required_seed_count"] = (
        passed_seed_count >= required_validation_seeds
    )
    passed = bool(all(checks.values()))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            "OAC2_CONTINUOUS_ACTOR_VALIDATION_PASS"
            if passed else "OAC2_CONTINUOUS_ACTOR_VALIDATION_FAIL"
        ),
        "passed": passed,
        "passed_seed_count": int(passed_seed_count),
        "checks": checks,
        "records": records,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (cli.run_dir / "validator_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
