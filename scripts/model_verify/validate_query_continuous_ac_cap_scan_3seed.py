#!/usr/bin/env python3
"""Independently validate the three-seed continuous Query AC step-cap scan."""

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
from run_query_target_coverage_mixed_init_oac import warm_relative_metrics  # noqa: E402
from validate_query_continuous_ac_candidate_bank_ab import (  # noqa: E402
    array_error,
    sha256,
    validate_adapter,
    validate_arm,
)


DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_continuous_ac_cap_scan_3seed_20260903_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def pooled(records: dict[str, dict[str, Any]], cap: str,
           data: dict[str, np.ndarray]) -> dict[str, Any]:
    costs, rows = [], []
    for seed in (0, 1, 2):
        record = records[cap][str(seed)]
        with np.load(record["arrays"], allow_pickle=False) as archive:
            rows.append(np.asarray(archive["selection_indices"]))
            costs.append(np.asarray(archive["selection_round_cost"])[int(record["selected_round"])])
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

    reused = Path(config["sources"]["validated_cap002"])
    reused_summary_path, reused_validation_path = reused / "summary.json", reused / "validation.json"
    if sha256(reused_summary_path) != manifest["reused_cap002_summary_sha256"]:
        raise AssertionError("reused cap002 summary changed")
    if sha256(reused_validation_path) != manifest["reused_cap002_validation_sha256"]:
        raise AssertionError("reused cap002 validation changed")
    reused_summary = json.loads(reused_summary_path.read_text())
    reused_validation = json.loads(reused_validation_path.read_text())
    if reused_validation["qualification"] != "QUERY_CONTINUOUS_AC_CANDIDATE_BANK_AB_THREE_SEED_INDEPENDENT_PASS":
        raise AssertionError("reused cap002 no longer independently passes")
    for seed in config["pilot"]["seeds"]:
        old = reused_summary["records"][str(seed)]["response39_recenter26"]
        new = summary["records"]["cap002"][str(seed)]
        for field in ("arrays", "arrays_sha256", "checkpoint", "checkpoint_sha256", "selected_round"):
            if new[field] != old[field]:
                raise AssertionError(f"reused cap002 seed{seed} {field} changed")
    old_contract = reused_summary["contract"]
    for field in ("split_contract", "critic_updates", "actor_updates", "actor_cost_weight_gamma",
                  "actor_cost_weight_maximum", "cost_weights", "checkpoint_selection", "tail_metrics_role"):
        if config[field] != old_contract[field]:
            raise AssertionError(f"cap scan contract changed {field}")
    for field in ("rounds", "fit_contexts_visited_per_round", "noise_sigma",
                  "probe_radius_sigma_start", "probe_radius_sigma_end",
                  "probe_radius_sigma_anneal_rounds", "response_fit_ridge",
                  "response_gauss_newton_damping", "response_line_factors",
                  "recenter_radius_ratio", "recenter_direction_basis", "curve_checkpoints"):
        if config["pilot"][field] != old_contract["pilot"][field]:
            raise AssertionError(f"cap scan pilot contract changed {field}")
    if config["pilot"]["candidates_per_visit"] != 65:
        raise AssertionError("cap scan did not retain the 65-candidate bank")

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
    for seed in config["pilot"]["seeds"]:
        item = summary["source_adapters"][str(seed)]
        if item != manifest["source_adapters"][str(seed)]:
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
    artifact_hashes = True
    arm_reports: dict[str, dict[str, Any]] = {}
    for cap in config["pilot"]["cap_arms"]:
        for seed in config["pilot"]["seeds"]:
            record = summary["records"][cap][str(seed)]
            key = f"{cap}_seed{seed}"
            artifact_hashes &= sha256(Path(record["arrays"])) == manifest["result_arrays_sha256"][key]
            artifact_hashes &= sha256(Path(record["checkpoint"])) == manifest["result_checkpoint_sha256"][key]
            if cap in config["pilot"]["new_cap_arms"]:
                arm_reports[key] = validate_arm(
                    cap, record, 65, config, data, controller, selection, device,
                )

    paired_errors: dict[str, dict[str, dict[str, float]]] = {}
    round0_errors: dict[str, dict[str, dict[str, float]]] = {}
    for seed in config["pilot"]["seeds"]:
        control = summary["records"]["cap002"][str(seed)]
        with np.load(control["arrays"], allow_pickle=False) as control_arrays:
            for cap in config["pilot"]["new_cap_arms"]:
                candidate = summary["records"][cap][str(seed)]
                with np.load(candidate["arrays"], allow_pickle=False) as arrays:
                    paired_errors.setdefault(str(seed), {})[cap] = {
                        "actor_batch_schedule": array_error(control_arrays["actor_batch_schedule"], arrays["actor_batch_schedule"]),
                        "visited_rows": array_error(
                            control_arrays["online_state_index"].reshape(800, 65)[:, 0],
                            arrays["online_state_index"].reshape(800, 65)[:, 0],
                        ),
                        "online_round": array_error(
                            control_arrays["online_round"].reshape(800, 65)[:, 0],
                            arrays["online_round"].reshape(800, 65)[:, 0],
                        ),
                    }
                    round0_errors.setdefault(str(seed), {})[cap] = {
                        "action": array_error(control_arrays["selection_round_action"][0], arrays["selection_round_action"][0]),
                        "cost": array_error(control_arrays["selection_round_cost"][0], arrays["selection_round_cost"][0]),
                    }

    recomputed_pooled = {
        cap: pooled(summary["records"], cap, data) for cap in config["pilot"]["cap_arms"]
    }
    pooled_errors = {
        cap: {
            "actor_mean": abs(float(report["actor_cost"]["mean"])
                              - float(summary["pooled"][cap]["actor_cost"]["mean"])),
            "aggregate": abs(float(report["aggregate_improvement"])
                             - float(summary["pooled"][cap]["aggregate_improvement"])),
        } for cap, report in recomputed_pooled.items()
    }
    control_mean = float(recomputed_pooled["cap002"]["actor_cost"]["mean"])
    control_aggregate = float(recomputed_pooled["cap002"]["aggregate_improvement"])
    expected_candidates = {}
    eligible = []
    for cap in config["pilot"]["new_cap_arms"]:
        improved = []
        for seed in config["pilot"]["seeds"]:
            control_record = summary["records"]["cap002"][str(seed)]
            candidate_record = summary["records"][cap][str(seed)]
            improved.append(
                float(candidate_record["selected"]["inner"]["actor_cost"]["mean"])
                < float(control_record["selected"]["inner"]["actor_cost"]["mean"])
            )
        candidate_mean = float(recomputed_pooled[cap]["actor_cost"]["mean"])
        candidate_aggregate = float(recomputed_pooled[cap]["aggregate_improvement"])
        candidate_checks = {
            "pooled_mean_lower_than_cap002": candidate_mean < control_mean,
            "pooled_warm_aggregate_higher_than_cap002": candidate_aggregate > control_aggregate,
            "mean_lower_in_at_least_two_seeds": sum(improved) >= 2,
        }
        if all(candidate_checks.values()):
            eligible.append(cap)
        expected_candidates[cap] = {
            "checks": candidate_checks,
            "improved_seed_count": int(sum(improved)),
            "pooled_selected_mean": candidate_mean,
            "pooled_warm_aggregate": candidate_aggregate,
        }
    selected_cap = min(eligible, key=lambda name: float(recomputed_pooled[name]["actor_cost"]["mean"])) if eligible else "cap002"
    expected_decision = (
        f"PROMOTE_{selected_cap.upper()}_TO_ACTOR_LR2E5_AB"
        if selected_cap != "cap002" else "RETAIN_CAP002_FOR_ACTOR_LR2E5_AB"
    )
    decision_exact = summary["comparison"]["eligible_caps"] == eligible
    decision_exact &= summary["comparison"]["selected_cap"] == selected_cap
    decision_exact &= summary["decision"] == expected_decision
    for cap in config["pilot"]["new_cap_arms"]:
        stored = summary["comparison"]["candidates"][cap]
        expected = expected_candidates[cap]
        decision_exact &= stored["decision_checks"] == expected["checks"]
        decision_exact &= stored["improved_seed_count"] == expected["improved_seed_count"]
        decision_exact &= abs(stored["pooled_selected_mean"] - expected["pooled_selected_mean"]) <= 1e-12
        decision_exact &= abs(stored["pooled_warm_aggregate"] - expected["pooled_warm_aggregate"]) <= 1e-12

    checks = {
        "artifact_hashes": bool(artifact_hashes),
        "source_and_adapters_exact": all(
            value == 0.0 for report in adapter_errors.values() for value in report.values()
        ),
        "reused_cap002_independently_qualified_and_exact": True,
        "training_contract_matches_cap002": True,
        "split_and_sealed_boundary": True,
        "paired_actor_and_context_schedules": all(
            value == 0.0 for seed_report in paired_errors.values()
            for cap_report in seed_report.values() for value in cap_report.values()
        ),
        "common_round0_exact": all(
            value == 0.0 for seed_report in round0_errors.values()
            for cap_report in seed_report.values() for value in cap_report.values()
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
        "decision_recomputed": bool(decision_exact),
    }
    validation = {
        "qualification": (
            "QUERY_CONTINUOUS_AC_CAP_SCAN_THREE_SEED_INDEPENDENT_PASS"
            if all(checks.values()) else "QUERY_CONTINUOUS_AC_CAP_SCAN_THREE_SEED_INDEPENDENT_FAIL"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "new_candidate_replay_count": int(sum(
            summary["records"][cap][str(seed)]["online_query_rollouts"]
            for cap in config["pilot"]["new_cap_arms"] for seed in config["pilot"]["seeds"]
        )),
        "adapter_errors": adapter_errors,
        "paired_schedule_errors": paired_errors,
        "round0_errors": round0_errors,
        "new_arm_reports": {
            key: {"errors": value["errors"], "all_checks_pass": value["all_checks_pass"]}
            for key, value in arm_reports.items()
        },
        "pooled_errors": pooled_errors,
        "recomputed_candidates": expected_candidates,
        "recomputed_selected_cap": selected_cap,
        "recomputed_decision": expected_decision,
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
