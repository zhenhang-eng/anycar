#!/usr/bin/env python3
"""Independently validate the three-seed 39/65 continuous Query AC comparison."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package))

import pretrain_query_single_center_actor_twin_critic as pretrain  # noqa: E402
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from run_query_continuous_ac_candidate_bank_ab import response_bank39, response_bank65  # noqa: F401,E402
from run_query_target_coverage_mixed_init_oac import warm_relative_metrics  # noqa: E402
from validate_query_continuous_ac_candidate_bank_ab import (  # noqa: E402
    array_error,
    sha256,
    validate_adapter,
    validate_arm,
)


DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_continuous_ac_candidate_bank_ab_3seed_20260903_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def pooled(records: dict[str, dict[str, Any]], arm: str,
           data: dict[str, np.ndarray]) -> dict[str, Any]:
    costs, rows = [], []
    for seed in (0, 1, 2):
        record = records[str(seed)][arm]
        with np.load(record["arrays"], allow_pickle=False) as archive:
            local_rows = np.asarray(archive["selection_indices"])
            local_cost = np.asarray(archive["selection_round_cost"])[int(record["selected_round"])]
        rows.append(local_rows)
        costs.append(local_cost)
    joined_rows, joined_costs = np.concatenate(rows), np.concatenate(costs)
    return warm_relative_metrics(
        joined_costs, data["warm_cost"][joined_rows], data["speed_kph"][joined_rows],
        data["variant_index"][joined_rows],
    )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    manifest_path, summary_path = output / "manifest.json", output / "summary.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    if sha256(config_path) != manifest["config_sha256"]:
        raise AssertionError("config hash mismatch")
    if sha256(Path(manifest["script"])) != manifest["script_sha256"]:
        raise AssertionError("runner hash mismatch")
    if sha256(summary_path) != manifest["summary_sha256"]:
        raise AssertionError("summary hash mismatch")
    if any((summary["outer_fold_evaluated"], summary["formal_validation_or_test_consumed"],
            summary["dbm_fields_or_labels_consumed"], summary["query_analytic_gradient_consumed"])):
        raise AssertionError("sealed boundary violated")

    source = Path(config["sources"]["actor_oac"])
    source_validation_path = source / "validation.json"
    if sha256(source_validation_path) != manifest["source_actor_validation_sha256"]:
        raise AssertionError("source OAC validation hash mismatch")
    if json.loads(source_validation_path.read_text())["qualification"] != "QUERY_TARGET_COVERAGE_MIXED_INIT_OAC_INDEPENDENT_TRAIN_SIDE_PASS":
        raise AssertionError("source OAC qualification changed")

    reused = Path(config["sources"]["validated_seed1_ab"])
    reused_summary_path, reused_validation_path = reused / "summary.json", reused / "validation.json"
    if sha256(reused_summary_path) != manifest["reused_seed1_summary_sha256"]:
        raise AssertionError("reused seed1 summary changed")
    if sha256(reused_validation_path) != manifest["reused_seed1_validation_sha256"]:
        raise AssertionError("reused seed1 validation changed")
    reused_summary = json.loads(reused_summary_path.read_text())
    reused_validation = json.loads(reused_validation_path.read_text())
    if reused_validation["qualification"] != "QUERY_CONTINUOUS_AC_CANDIDATE_BANK_AB_INDEPENDENT_PASS":
        raise AssertionError("reused seed1 validation no longer passes")
    for arm in config["pilot"]["candidate_arms"]:
        for field in ("arrays", "arrays_sha256", "checkpoint", "checkpoint_sha256", "selected_round"):
            if summary["records"]["1"][arm][field] != reused_summary["records"][arm][field]:
                raise AssertionError(f"reused seed1 {arm} {field} changed")
    reused_contract = reused_summary["contract"]
    for field in ("split_contract", "critic_updates", "actor_updates", "actor_cost_weight_gamma",
                  "actor_cost_weight_maximum", "cost_weights", "checkpoint_selection", "tail_metrics_role"):
        if config[field] != reused_contract[field]:
            raise AssertionError(f"three-seed contract changed {field}")
    for field in ("rounds", "fit_contexts_visited_per_round", "candidate_arms", "noise_sigma",
                  "probe_radius_sigma_start", "probe_radius_sigma_end",
                  "probe_radius_sigma_anneal_rounds", "response_fit_ridge",
                  "response_gauss_newton_damping", "response_line_factors",
                  "recenter_radius_ratio", "recenter_direction_basis", "curve_checkpoints"):
        if config["pilot"][field] != reused_contract["pilot"][field]:
            raise AssertionError(f"three-seed pilot contract changed {field}")

    loader_config = {"outputs": {"absolute_replay": config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader_config)
    if replay_manifest["replay_sha256"] != manifest["source_replay_sha256"]:
        raise AssertionError("source Replay hash mismatch")
    selection = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    outer = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
    if (len(selection), len(outer)) != (120, 120):
        raise AssertionError("split size changed")
    if set(data["episode_id"][selection]) & set(data["episode_id"][outer]):
        raise AssertionError("inner/outer episode leakage")

    adapter_errors = {}
    for seed in config["pilot"]["new_seeds"]:
        item = summary["new_seed_adapters"][str(seed)]
        if item != manifest["new_seed_adapters"][str(seed)]:
            raise AssertionError("adapter manifest mismatch")
        source_checkpoint = source / f"seed_{seed}" / "checkpoint.pt"
        if sha256(source_checkpoint) != item["source_checkpoint_sha256"]:
            raise AssertionError("source checkpoint changed")
        adapter_errors[str(seed)] = validate_adapter({
            "source_actor_adapter": item["actor"],
            "source_actor_adapter_sha256": item["actor_sha256"],
            "source_adapter": item["critic"],
            "source_adapter_sha256": item["critic_sha256"],
        }, source_checkpoint)

    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    arm_reports: dict[str, dict[str, Any]] = {}
    paired_errors: dict[str, dict[str, float]] = {}
    round0_errors: dict[str, dict[str, float]] = {}
    artifact_hashes = True
    for seed in (0, 1, 2):
        for arm, candidate_count in config["pilot"]["candidate_arms"].items():
            record = summary["records"][str(seed)][arm]
            key = f"seed{seed}_{arm}"
            artifact_hashes &= sha256(Path(record["arrays"])) == manifest["result_arrays_sha256"][key]
            artifact_hashes &= sha256(Path(record["checkpoint"])) == manifest["result_checkpoint_sha256"][key]
            if seed in config["pilot"]["new_seeds"]:
                arm_reports[key] = validate_arm(
                    arm, record, int(candidate_count), config,
                    data, controller, selection, device,
                )
        left = summary["records"][str(seed)]["response39"]
        right = summary["records"][str(seed)]["response39_recenter26"]
        with np.load(left["arrays"], allow_pickle=False) as la, np.load(right["arrays"], allow_pickle=False) as ra:
            paired_errors[str(seed)] = {
                "actor_batch_schedule": array_error(la["actor_batch_schedule"], ra["actor_batch_schedule"]),
                "visited_rows": array_error(
                    la["online_state_index"].reshape(800, 39)[:, 0],
                    ra["online_state_index"].reshape(800, 65)[:, 0],
                ),
                "online_round": array_error(
                    la["online_round"].reshape(800, 39)[:, 0],
                    ra["online_round"].reshape(800, 65)[:, 0],
                ),
            }
            round0_errors[str(seed)] = {
                "action": array_error(la["selection_round_action"][0], ra["selection_round_action"][0]),
                "cost": array_error(la["selection_round_cost"][0], ra["selection_round_cost"][0]),
            }

    recomputed_pooled = {
        arm: pooled(summary["records"], arm, data) for arm in config["pilot"]["candidate_arms"]
    }
    pooled_errors = {
        arm: {
            "actor_mean": abs(float(recomputed_pooled[arm]["actor_cost"]["mean"])
                              - float(summary["pooled"][arm]["actor_cost"]["mean"])),
            "aggregate": abs(float(recomputed_pooled[arm]["aggregate_improvement"])
                             - float(summary["pooled"][arm]["aggregate_improvement"])),
        } for arm in config["pilot"]["candidate_arms"]
    }
    baseline_mean = float(recomputed_pooled["response39"]["actor_cost"]["mean"])
    treatment_mean = float(recomputed_pooled["response39_recenter26"]["actor_cost"]["mean"])
    baseline_aggregate = float(recomputed_pooled["response39"]["aggregate_improvement"])
    treatment_aggregate = float(recomputed_pooled["response39_recenter26"]["aggregate_improvement"])
    seed_improvements = []
    for seed in (0, 1, 2):
        baseline = summary["records"][str(seed)]["response39"]
        treatment = summary["records"][str(seed)]["response39_recenter26"]
        seed_improvements.append(
            float(treatment["selected"]["inner"]["actor_cost"]["mean"])
            < float(baseline["selected"]["inner"]["actor_cost"]["mean"])
        )
    expected_checks = {
        "treatment_pooled_mean_lower": treatment_mean < baseline_mean,
        "treatment_pooled_warm_aggregate_higher": treatment_aggregate > baseline_aggregate,
        "treatment_mean_lower_in_at_least_two_seeds": sum(seed_improvements) >= 2,
    }
    expected_decision = (
        "PROMOTE_RECENTER65_TO_CONTINUOUS_CAP_SCAN"
        if all(expected_checks.values()) else "RETAIN_RESPONSE39_CONTINUOUS_AC"
    )
    checks = {
        "artifact_hashes": bool(artifact_hashes),
        "source_and_new_adapters_exact": all(
            value == 0.0 for report in adapter_errors.values() for value in report.values()
        ),
        "reused_seed1_independently_qualified_and_exact": True,
        "training_contract_matches_validated_seed1": True,
        "split_and_sealed_boundary": True,
        "paired_actor_and_context_schedules": all(
            value == 0.0 for report in paired_errors.values() for value in report.values()
        ),
        "common_round0_exact": all(
            value == 0.0 for report in round0_errors.values() for value in report.values()
        ),
        "all_new_online_query_candidates_replayed": all(
            report["all_checks_pass"] for report in arm_reports.values()
        ),
        "new_selected_actor_and_inner_cost_reloaded": all(
            report["errors"]["selected_action_reload"] == 0.0
            and report["errors"]["selected_cost_reload"] <= 1e-6
            for report in arm_reports.values()
        ),
        "pooled_metrics_recomputed": all(
            value <= 1e-12 for report in pooled_errors.values() for value in report.values()
        ),
        "decision_recomputed": summary["decision_checks"] == expected_checks
        and summary["decision"] == expected_decision,
    }
    validation = {
        "qualification": (
            "QUERY_CONTINUOUS_AC_CANDIDATE_BANK_AB_THREE_SEED_INDEPENDENT_PASS"
            if all(checks.values()) else "QUERY_CONTINUOUS_AC_CANDIDATE_BANK_AB_THREE_SEED_INDEPENDENT_FAIL"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "new_candidate_replay_count": int(sum(
            summary["records"][str(seed)][arm]["online_query_rollouts"]
            for seed in config["pilot"]["new_seeds"]
            for arm in config["pilot"]["candidate_arms"]
        )),
        "adapter_errors": adapter_errors,
        "paired_schedule_errors": paired_errors,
        "round0_errors": round0_errors,
        "new_arm_reports": {
            key: {"errors": value["errors"], "all_checks_pass": value["all_checks_pass"]}
            for key, value in arm_reports.items()
        },
        "pooled_errors": pooled_errors,
        "recomputed_decision_checks": expected_checks,
        "recomputed_decision": expected_decision,
        "recomputed_comparison": {
            "baseline_pooled_selected_mean": baseline_mean,
            "treatment_pooled_selected_mean": treatment_mean,
            "treatment_pooled_mean_reduction": baseline_mean - treatment_mean,
            "baseline_pooled_warm_aggregate": baseline_aggregate,
            "treatment_pooled_warm_aggregate": treatment_aggregate,
            "treatment_pooled_aggregate_delta": treatment_aggregate - baseline_aggregate,
            "improved_seed_count": int(sum(seed_improvements)),
        },
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "validation.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
