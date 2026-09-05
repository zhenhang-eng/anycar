#!/usr/bin/env python3
"""Independently validate the three-seed 160-round Query continuous-AC run."""

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
from query_batched_direct_cost import batched_direct_cost  # noqa: E402
from run_query_continuous_ac_candidate_bank_ab import response_bank65  # noqa: E402
from run_query_forward_response_landscape_pilot import basis_bank  # noqa: E402
from run_query_single_center_oac20to1 import actor_from_payload, actor_predict, load_inputs, sha256  # noqa: E402
from run_query_target_coverage_mixed_init_oac import warm_relative_metrics  # noqa: E402
from validate_query_continuous_ac_candidate_bank_ab import array_error, validate_adapter  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_continuous_ac_longrun_160_20260904_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def best_stage(costs: np.ndarray, rows: np.ndarray, final_round: int,
               data: dict[str, np.ndarray]) -> dict[str, Any]:
    local = np.asarray(costs[: final_round + 1], np.float32)
    means = local.astype(np.float64).mean(axis=1)
    selected_round = int(np.argmin(means))
    report = warm_relative_metrics(
        local[selected_round], data["warm_cost"][rows], data["speed_kph"][rows],
        data["variant_index"][rows],
    )
    return {
        "selected_round": selected_round,
        "selected_cost": local[selected_round],
        "inner": report,
    }


def pooled(stages: list[dict[str, Any]], rows: np.ndarray,
           data: dict[str, np.ndarray]) -> dict[str, Any]:
    joined_rows = np.tile(rows, len(stages))
    joined_cost = np.concatenate([stage["selected_cost"] for stage in stages])
    return warm_relative_metrics(
        joined_cost, data["warm_cost"][joined_rows], data["speed_kph"][joined_rows],
        data["variant_index"][joined_rows],
    )


def validate_seed(
    seed: int,
    record: dict[str, Any],
    config: dict,
    data: dict[str, np.ndarray],
    controller: TorchMPPIController,
    selection: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    arrays_path, checkpoint_path = Path(record["arrays"]), Path(record["checkpoint"])
    if sha256(arrays_path) != record["arrays_sha256"]:
        raise AssertionError(f"seed{seed} arrays hash mismatch")
    if sha256(checkpoint_path) != record["checkpoint_sha256"]:
        raise AssertionError(f"seed{seed} checkpoint hash mismatch")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor = actor_from_payload(payload, "selected_actor_state_dict", device)
    inputs = load_inputs(data, payload["actor_normalization"])
    selected_action = actor_predict(actor, inputs, selection, device)
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    selected_cost = batched_direct_cost(controller, data, selection, selected_action, weights)
    errors: dict[str, float] = {}
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if arrays["actor_batch_schedule"].shape != (160, 16, 64):
            raise AssertionError("Actor batch schedule shape changed")
        if arrays["selection_round_cost"].shape != (161, 120):
            raise AssertionError("selection cost shape changed")
        group_ids = np.unique(arrays["online_group"])
        if len(group_ids) != 3200:
            raise AssertionError("online group count changed")
        if len(arrays["online_cost"]) != 208000:
            raise AssertionError("online candidate count changed")
        if not np.array_equal(np.unique(arrays["online_round"]), np.arange(1, 161)):
            raise AssertionError("online round coverage changed")
        selected_round = int(record["selected_round"])
        errors["selected_action_reload"] = array_error(
            selected_action, arrays["selection_round_action"][selected_round]
        )
        errors["selected_cost_reload"] = array_error(
            selected_cost, arrays["selection_round_cost"][selected_round]
        )
        reconstructed_full = best_stage(
            np.asarray(arrays["selection_round_cost"]), selection, 160, data
        )
        reconstructed_40 = best_stage(
            np.asarray(arrays["selection_round_cost"]), selection, 40, data
        )
        errors["selected_round_recomputed"] = float(
            reconstructed_full["selected_round"] != selected_round
        )
        errors["selected_metric_actor_mean"] = abs(
            float(reconstructed_full["inner"]["actor_cost"]["mean"])
            - float(record["selected"]["inner"]["actor_cost"]["mean"])
        )
        errors["selected_metric_aggregate"] = abs(
            float(reconstructed_full["inner"]["aggregate_improvement"])
            - float(record["selected"]["inner"]["aggregate_improvement"])
        )
        sigma = np.asarray(config["pilot"]["noise_sigma"], np.float32).reshape(1, 2)
        anneal = int(config["pilot"]["probe_radius_sigma_anneal_rounds"])
        radii = np.concatenate((
            np.linspace(float(config["pilot"]["probe_radius_sigma_start"]),
                        float(config["pilot"]["probe_radius_sigma_end"]), anneal),
            np.full(160 - anneal, float(config["pilot"]["probe_radius_sigma_end"])),
        )).astype(np.float32)
        bases = basis_bank()
        expected_roles = np.asarray(
            ["actor"] + ["probe"] * 32 + ["response"] * 6 + ["recenter"] * 26
        )
        maxima = {"action": 0.0, "cost": 0.0, "raw": 0.0, "clipped": 0.0, "role": 0.0}
        for group in group_ids:
            mask = arrays["online_group"] == group
            if int(mask.sum()) != 65:
                raise AssertionError("variable-width online group")
            stored_rows = arrays["online_state_index"][mask]
            stored_rounds = arrays["online_round"][mask]
            if not np.all(stored_rows == stored_rows[0]) or not np.all(stored_rounds == stored_rounds[0]):
                raise AssertionError("online group mixes rows or rounds")
            row, round_index = int(stored_rows[0]), int(stored_rounds[0])
            center = np.asarray(arrays["online_action"][mask][0], np.float32)
            expected = response_bank65(
                controller, data, row, center, float(radii[round_index - 1]),
                bases[(round_index - 1) % len(bases)], sigma, weights, config,
            )
            maxima["action"] = max(maxima["action"], array_error(arrays["online_action"][mask], expected[0]))
            maxima["cost"] = max(maxima["cost"], array_error(arrays["online_cost"][mask], expected[1]))
            maxima["raw"] = max(maxima["raw"], array_error(arrays["online_raw_action"][mask], expected[2]))
            maxima["clipped"] = max(maxima["clipped"], array_error(arrays["online_clipped"][mask], expected[3]))
            maxima["role"] = max(maxima["role"], array_error(arrays["online_role"][mask], expected_roles))
            if int(group) == 0 or (int(group) + 1) % 400 == 0:
                print(f"validate seed={seed} group {int(group) + 1}/3200", flush=True)
        schedule = {
            "actor_batch_schedule": np.asarray(arrays["actor_batch_schedule"][:40]),
            "visited_rows": np.asarray(arrays["online_state_index"]).reshape(3200, 65)[:800, 0],
            "online_round": np.asarray(arrays["online_round"]).reshape(3200, 65)[:800, 0],
            "round0_action": np.asarray(arrays["selection_round_action"][0]),
            "round0_cost": np.asarray(arrays["selection_round_cost"][0]),
        }
    errors.update({f"online_{name}": value for name, value in maxima.items()})
    return {
        "errors": errors,
        "round40": reconstructed_40,
        "full": reconstructed_full,
        "schedule": schedule,
        "all_checks_pass": all(value <= 1e-6 for value in errors.values()),
    }


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
    previous = Path(config["sources"]["validated_40round_lr2e5"])
    previous_summary_path, previous_validation_path = previous / "summary.json", previous / "validation.json"
    if sha256(previous_summary_path) != manifest["previous_summary_sha256"]:
        raise AssertionError("historical 40-round summary changed")
    if sha256(previous_validation_path) != manifest["previous_validation_sha256"]:
        raise AssertionError("historical 40-round validation changed")
    previous_summary = json.loads(previous_summary_path.read_text())
    if json.loads(previous_validation_path.read_text())["qualification"] != "QUERY_CONTINUOUS_AC_ACTOR_LR2E5_AB_THREE_SEED_INDEPENDENT_PASS":
        raise AssertionError("historical 40-round source no longer passes")

    old_contract = previous_summary["contract"]
    for field in ("split_contract", "critic_updates", "actor_cost_weight_gamma",
                  "actor_cost_weight_maximum", "cost_weights", "tail_metrics_role"):
        if config[field] != old_contract[field]:
            raise AssertionError(f"longrun contract changed {field}")
    for field in config["actor_updates"]:
        if field == "learning_rate_per_microstep":
            if config["actor_updates"][field] != old_contract["pilot"]["learning_rate_arms"]["lr2e5"]:
                raise AssertionError("longrun Actor LR changed")
        elif config["actor_updates"][field] != old_contract["actor_updates"][field]:
            raise AssertionError(f"longrun actor contract changed {field}")
    for field in ("fit_contexts_visited_per_round", "candidates_per_visit", "noise_sigma",
                  "probe_radius_sigma_start", "probe_radius_sigma_end", "probe_radius_sigma_anneal_rounds",
                  "response_fit_ridge", "response_gauss_newton_damping", "response_line_factors",
                  "recenter_radius_ratio", "recenter_direction_basis"):
        if config["pilot"][field] != old_contract["pilot"][field]:
            raise AssertionError(f"longrun pilot contract changed {field}")
    if config["pilot"]["rounds"] != 160:
        raise AssertionError("longrun round budget changed")
    runtime = summary["deterministic_runtime_contract"]
    if not (runtime["cudnn_deterministic"] and runtime["deterministic_algorithms"]
            and not runtime["cudnn_benchmark"] and runtime["cublas_workspace_config"] == ":4096:8"):
        raise AssertionError("deterministic warn-only runtime was not enabled")

    loader_config = {"outputs": {"absolute_replay": config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader_config)
    if replay_manifest["replay_sha256"] != manifest["source_replay_sha256"]:
        raise AssertionError("source Replay hash mismatch")
    selection = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    outer = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
    if (len(selection), len(outer)) != (120, 120) or set(data["episode_id"][selection]) & set(data["episode_id"][outer]):
        raise AssertionError("split size or isolation changed")

    adapter_errors = {}
    for seed in config["pilot"]["seeds"]:
        item = summary["source_adapters"][str(seed)]
        if item != manifest["source_adapters"][str(seed)]:
            raise AssertionError("adapter manifest mismatch")
        source_checkpoint = source / f"seed_{seed}" / "checkpoint.pt"
        adapter_errors[str(seed)] = validate_adapter({
            "source_actor_adapter": item["actor"], "source_actor_adapter_sha256": item["actor_sha256"],
            "source_adapter": item["critic"], "source_adapter_sha256": item["critic_sha256"],
        }, source_checkpoint)

    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query), TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    reports = {}
    artifact_hashes = True
    seed_record_hashes = True
    historical_prefix_errors = {}
    for seed in config["pilot"]["seeds"]:
        record = summary["records"][str(seed)]
        artifact_hashes &= sha256(Path(record["arrays"])) == manifest["result_arrays_sha256"][str(seed)]
        artifact_hashes &= sha256(Path(record["checkpoint"])) == manifest["result_checkpoint_sha256"][str(seed)]
        seed_record_path = output / f"seed_{seed}_record.json"
        seed_record_hashes &= sha256(seed_record_path) == manifest["seed_record_sha256"][str(seed)]
        reports[str(seed)] = validate_seed(
            int(seed), record, config, data, controller, selection, device,
        )
        historical = previous_summary["records"]["lr2e5"][str(seed)]
        with np.load(historical["arrays"], allow_pickle=False) as old:
            expected = {
                "actor_batch_schedule": np.asarray(old["actor_batch_schedule"]),
                "visited_rows": np.asarray(old["online_state_index"]).reshape(800, 65)[:, 0],
                "online_round": np.asarray(old["online_round"]).reshape(800, 65)[:, 0],
                "round0_action": np.asarray(old["selection_round_action"][0]),
                "round0_cost": np.asarray(old["selection_round_cost"][0]),
            }
        historical_prefix_errors[str(seed)] = {
            name: array_error(reports[str(seed)]["schedule"][name], value)
            for name, value in expected.items()
        }

    round40_stages = [reports[str(seed)]["round40"] for seed in config["pilot"]["seeds"]]
    full_stages = [reports[str(seed)]["full"] for seed in config["pilot"]["seeds"]]
    round40_pooled = pooled(round40_stages, selection, data)
    full_pooled = pooled(full_stages, selection, data)
    pooled_errors = {
        "round40_actor_mean": abs(float(round40_pooled["actor_cost"]["mean"])
                                  - float(summary["within_run_round40_pooled"]["actor_cost"]["mean"])),
        "round40_aggregate": abs(float(round40_pooled["aggregate_improvement"])
                                 - float(summary["within_run_round40_pooled"]["aggregate_improvement"])),
        "full_actor_mean": abs(float(full_pooled["actor_cost"]["mean"])
                               - float(summary["pooled"]["actor_cost"]["mean"])),
        "full_aggregate": abs(float(full_pooled["aggregate_improvement"])
                              - float(summary["pooled"]["aggregate_improvement"])),
    }
    improved = [
        full["inner"]["actor_cost"]["mean"] < early["inner"]["actor_cost"]["mean"]
        for early, full in zip(round40_stages, full_stages)
    ]
    expected_decision_checks = {
        "all_schedule_and_round0_prefixes_exact": all(
            value == 0.0 for report in historical_prefix_errors.values() for value in report.values()
        ),
        "longrun_pooled_mean_lower": full_pooled["actor_cost"]["mean"] < round40_pooled["actor_cost"]["mean"],
        "longrun_pooled_warm_aggregate_higher": full_pooled["aggregate_improvement"] > round40_pooled["aggregate_improvement"],
        "longrun_mean_lower_in_at_least_two_seeds": sum(improved) >= 2,
    }
    expected_decision = "PROMOTE_160ROUND_CONTINUOUS_AC" if all(expected_decision_checks.values()) else "RETAIN_40ROUND_CONTINUOUS_AC"
    checks = {
        "artifact_hashes": bool(artifact_hashes and seed_record_hashes),
        "source_and_adapters_exact": all(value == 0.0 for report in adapter_errors.values() for value in report.values()),
        "historical_source_independently_qualified": True,
        "single_variable_longrun_contract": True,
        "deterministic_warn_only_runtime_recorded": True,
        "split_and_sealed_boundary": True,
        "historical_schedule_and_round0_prefix_exact": all(
            value == 0.0 for report in historical_prefix_errors.values() for value in report.values()
        ),
        "all_online_query_candidates_replayed": all(report["all_checks_pass"] for report in reports.values()),
        "selected_actor_and_inner_cost_reloaded": all(
            report["errors"]["selected_action_reload"] == 0.0
            and report["errors"]["selected_cost_reload"] <= 1e-6 for report in reports.values()
        ),
        "round40_and_full_metrics_recomputed": all(value <= 1e-12 for value in pooled_errors.values()),
        "decision_recomputed": summary["decision_checks"] == expected_decision_checks
        and summary["decision"] == expected_decision,
    }
    validation = {
        "qualification": "QUERY_CONTINUOUS_AC_LONGRUN_160_THREE_SEED_INDEPENDENT_PASS"
                         if all(checks.values()) else "QUERY_CONTINUOUS_AC_LONGRUN_160_THREE_SEED_INDEPENDENT_FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "online_candidate_replay_count": int(sum(summary["records"][str(seed)]["online_query_rollouts"]
                                                 for seed in config["pilot"]["seeds"])),
        "adapter_errors": adapter_errors, "historical_prefix_errors": historical_prefix_errors,
        "seed_reports": {seed: {"errors": report["errors"], "all_checks_pass": report["all_checks_pass"]}
                         for seed, report in reports.items()},
        "pooled_errors": pooled_errors, "recomputed_decision_checks": expected_decision_checks,
        "recomputed_decision": expected_decision,
        "recomputed_comparison": {
            "round40_pooled_selected_mean": float(round40_pooled["actor_cost"]["mean"]),
            "longrun_pooled_selected_mean": float(full_pooled["actor_cost"]["mean"]),
            "longrun_pooled_mean_reduction": float(round40_pooled["actor_cost"]["mean"]-full_pooled["actor_cost"]["mean"]),
            "round40_pooled_warm_aggregate": float(round40_pooled["aggregate_improvement"]),
            "longrun_pooled_warm_aggregate": float(full_pooled["aggregate_improvement"]),
            "longrun_pooled_aggregate_delta": float(full_pooled["aggregate_improvement"]-round40_pooled["aggregate_improvement"]),
            "improved_seed_count": int(sum(improved)),
        },
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "validation.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
