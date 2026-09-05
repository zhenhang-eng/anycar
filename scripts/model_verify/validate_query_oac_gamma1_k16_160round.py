#!/usr/bin/env python3
"""Independently replay and qualify the Query gamma1 K16 160-round curve."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_query_oac_gamma1_k16_160round as runner
import run_query_oac_gamma1_k_scan as base


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_oac_gamma1_k16_160round_20260902_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def max_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.dtype.kind in "OUS" or right.dtype.kind in "OUS":
        return 0.0 if np.array_equal(left, right) else float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def recursive_error(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return float("inf")
        return max((recursive_error(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return float("inf")
        return max((recursive_error(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    summary_path = output / "summary.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    replay_dir = Path(manifest["absolute_replay"])
    pretrain_dir = Path(manifest["pretrain"])
    prior_dir = Path(manifest["k4_k16_90round"])
    prior_summary = json.loads((prior_dir / "summary.json").read_text())
    prior_validation = json.loads((prior_dir / "validation.json").read_text())
    replay_manifest = json.loads((replay_dir / "manifest.json").read_text())
    with np.load(replay_dir / "replay.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    fullrank_manifest = json.loads((Path(replay_manifest["fullrank_source"]) / "manifest.json").read_text())
    parent_manifest = json.loads((Path(fullrank_manifest["parent_t0"]) / "manifest.json").read_text())
    collection_manifest = json.loads((Path(parent_manifest["source_collection"]) / "manifest.json").read_text())
    device = torch.device(args.device)
    query = base.QueryDeploymentModel.from_checkpoint(Path(manifest["query_checkpoint"]), device)
    controller = base.TorchMPPIController(
        base.TorchQueryRolloutBackend(query),
        base.TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    sigma = np.asarray(config["pilot"]["noise_sigma"], np.float32).reshape(1, 2)
    radii = runner.piecewise_radii(config)
    bases = base.basis_bank()
    rounds = int(config["pilot"]["rounds"])
    contexts = int(config["pilot"]["fit_contexts_visited_per_round"])
    candidates = int(config["pilot"]["candidates_per_visit"])
    prefix_rounds = int(config["prefix_reproduction"]["rounds"])
    tolerance = float(config["prefix_reproduction"]["required_maximum_absolute_error"])
    expected_online = rounds * contexts * candidates
    expected_arms = [{"name": "gamma1_k16", "actor_updates_per_round": 16}]
    checks: dict[str, bool] = {
        "config_hash": base.sha256(config_path) == manifest["config_sha256"],
        "runner_hash": base.sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "shared_runner_hash": base.sha256(Path(manifest["shared_runner"])) == manifest["shared_runner_sha256"],
        "summary_hash": base.sha256(summary_path) == manifest["summary_sha256"],
        "replay_hash": base.sha256(replay_dir / "replay.npz") == manifest["absolute_replay_sha256"],
        "pretrain_manifest_hash": base.sha256(pretrain_dir / "manifest.json") == manifest["pretrain_manifest_sha256"],
        "aggregation_validation_hash": base.sha256(Path(manifest["aggregation_ab"]) / "validation.json") == manifest["aggregation_validation_sha256"],
        "prior_validation_hash": base.sha256(prior_dir / "validation.json") == manifest["k4_k16_90round_validation_sha256"],
        "prior_qualified": prior_validation["qualification"] == "QUERY_OAC_GAMMA1_K4_K16_90ROUND_INDEPENDENT_PASS",
        "query_checkpoint_hash": base.sha256(Path(manifest["query_checkpoint"])) == manifest["query_checkpoint_sha256"],
        "registered_single_k16": config["arms"] == expected_arms,
        "registered_rounds_160": rounds == 160,
        "registered_anneal_rounds_90": int(config["pilot"]["probe_radius_sigma_anneal_rounds"]) == 90,
        "registered_curve_checkpoints": config["pilot"]["curve_checkpoints"] == [10, 20, 40, 60, 90, 120, 140, 160],
        "radius_tail_fixed": bool(np.all(radii[90:] == np.float32(0.05))),
        "gamma_fixed_one": float(config["actor_cost_weight_gamma"]) == 1.0,
        "critic_updates_fixed_20": int(config["critic_updates"]["updates_per_round_per_twin"]) == 20,
        "warm_external_only": config["evaluation_contract"]["warm_role"].startswith("external deterministic"),
        "decision_inner_only": summary["decision_population"] == "inner selected checkpoints only",
        "development_oof_not_untouched": "already-consumed" in summary["development_oof_role"],
        "formal_test_sealed": not bool(summary["formal_validation_or_test_consumed"]) and not bool(manifest["formal_validation_or_test_consumed"]),
        "dbm_fields_absent": not bool(summary["dbm_fields_or_labels_consumed"]) and not bool(manifest["dbm_fields_or_labels_consumed"]),
        "analytic_query_gradient_absent": not bool(summary["query_analytic_gradient_consumed"]) and not bool(manifest["query_analytic_gradient_consumed"]),
    }
    errors = {
        "candidate_action": 0.0,
        "candidate_raw_action": 0.0,
        "candidate_cost": 0.0,
        "round0_selection_action": 0.0,
        "round0_selection_cost": 0.0,
        "round0_oof_action": 0.0,
        "round0_oof_cost": 0.0,
        "latest_selection_action": 0.0,
        "latest_selection_cost": 0.0,
        "latest_oof_action": 0.0,
        "latest_oof_cost": 0.0,
        "selected_selection_action": 0.0,
        "selected_selection_cost": 0.0,
        "selected_oof_action": 0.0,
        "selected_oof_cost": 0.0,
        "summary_metrics": 0.0,
        "forbidden_actor_input": 0.0,
    }
    prefix_errors = {name: 0.0 for name in summary["prefix_reproduction_maximum_absolute_errors"]}
    prior_records = {
        int(record["seed"]): record
        for record in prior_summary["records"]
        if record["arm"] == "gamma1_k16"
    }
    reports = []
    arrays_by_seed: dict[int, dict[str, np.ndarray]] = {}
    records_by_seed = {int(record["seed"]): record for record in summary["records"]}
    for record in summary["records"]:
        seed = int(record["seed"])
        k = int(record["actor_updates_per_round"])
        arrays_path = Path(record["arrays"])
        checkpoint_path = Path(record["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        with np.load(arrays_path, allow_pickle=False) as archive:
            saved = {name: np.asarray(archive[name]) for name in archive.files}
        arrays_by_seed[seed] = saved
        fit, selection, oof = saved["fit_indices"], saved["selection_indices"], saved["oof_indices"]
        source_path = pretrain_dir / "checkpoints" / f"pretrain_fold0_seed{seed}.pt"
        run_checks: dict[str, bool] = {
            "checkpoint_hash": base.sha256(checkpoint_path) == record["checkpoint_sha256"] == manifest["run_artifacts"][f"gamma1_k16_seed{seed}"]["checkpoint_sha256"],
            "arrays_hash": base.sha256(arrays_path) == record["arrays_sha256"] == manifest["run_artifacts"][f"gamma1_k16_seed{seed}"]["arrays_sha256"],
            "source_checkpoint_exact": Path(checkpoint["source_pretrain_checkpoint"]) == source_path,
            "source_checkpoint_hash": base.sha256(source_path) == checkpoint["source_pretrain_checkpoint_sha256"],
            "source_replay_hash": checkpoint["source_replay_sha256"] == manifest["absolute_replay_sha256"],
            "split_sizes": (len(fit), len(selection), len(oof)) == (360, 120, 120),
            "split_folds_exact": bool(np.all(np.isin(data["fold_id"][fit], [2, 3, 4])) and np.all(data["fold_id"][selection] == 1) and np.all(data["fold_id"][oof] == 0)),
            "online_count": len(saved["cost"]) == expected_online == int(record["online_replay_rows"]),
            "online_fit_only": bool(np.all(np.isin(saved["state_index"], fit))),
            "online_excludes_selection_oof": bool(not np.any(np.isin(saved["state_index"], selection)) and not np.any(np.isin(saved["state_index"], oof))),
            "actor_batch_count": len(saved["actor_batch_state_index"]) == rounds * k * int(config["actor_updates"]["batch_size"]) == int(record["actor_batch_rows"]),
            "actor_batches_fit_only": bool(np.all(np.isin(saved["actor_batch_state_index"], fit))),
            "actor_update_count": int(checkpoint["actor_update_count"]) == rounds * k,
            "critic_update_count": int(checkpoint["critic_update_count_per_twin"]) == rounds * 20,
            "candidate_finite_bounded": bool(np.all(np.isfinite(saved["action"])) and np.all(np.isfinite(saved["cost"])) and np.all(np.abs(saved["action"]) <= 1.0 + 1e-7)),
            "probe_radius_exact": max_error(saved["probe_radius_by_round"], radii) == 0.0,
            "cumulative_cap_respected": all(float(item["actor_update"]["final_cumulative_output_step_sigma_rms"]) <= 0.020000001 and float(item["actor_update"]["final_cumulative_output_step_sigma_rms"]) <= float(item["actor_update"]["raw_cumulative_output_step_sigma_rms"]) + 1e-9 for item in record["rounds"]),
            "microsteps_exact": all(int(item["actor_update"]["microstep_count"]) == 16 and len(item["actor_update"]["microsteps"]) == 16 for item in record["rounds"]),
            "formal_test_sealed": not bool(checkpoint["formal_validation_or_test_consumed"]),
            "dbm_fields_absent": not bool(checkpoint["dbm_fields_or_labels_consumed"]),
            "analytic_query_gradient_absent": not bool(checkpoint["query_analytic_gradient_consumed"]),
        }
        episode_sets = [set(data["episode_id"][rows].tolist()) for rows in (fit, selection, oof)]
        run_checks["episode_disjoint"] = not bool(episode_sets[0] & episode_sets[1] or episode_sets[0] & episode_sets[2] or episode_sets[1] & episode_sets[2])
        for round_index in range(1, rounds + 1):
            mask = saved["round"] == round_index
            groups = np.unique(saved["group"][mask])
            local_ok = mask.sum() == contexts * candidates and len(groups) == contexts
            for group in groups:
                positions = np.flatnonzero(saved["group"] == group)
                roles = saved["role"][positions]
                local_ok = local_ok and len(positions) == 39 and np.sum(roles == "actor") == 1 and np.sum(roles == "probe") == 32 and np.sum(roles == "response") == 6
                row = int(saved["state_index"][positions[0]])
                actions, costs, raw, clipped = base.response_bank(
                    controller, data, row, saved["action"][positions[0]],
                    float(radii[round_index - 1]), bases[(round_index - 1) % len(bases)],
                    sigma, weights, config,
                )
                errors["candidate_action"] = max(errors["candidate_action"], max_error(actions, saved["action"][positions]))
                errors["candidate_raw_action"] = max(errors["candidate_raw_action"], max_error(raw, saved["raw_action"][positions]))
                errors["candidate_cost"] = max(errors["candidate_cost"], max_error(costs, saved["cost"][positions]))
                local_ok = local_ok and np.array_equal(clipped, saved["clipped"][positions])
            batch_mask = saved["actor_batch_round"] == round_index
            local_ok = local_ok and all(np.sum(batch_mask & (saved["actor_batch_microstep"] == microstep)) == 64 for microstep in range(1, 17))
            run_checks[f"round_{round_index}_structure_and_replay"] = bool(local_ok)

        source = torch.load(source_path, map_location=device, weights_only=False)
        inputs = base.load_inputs(data, checkpoint["normalization"])
        actors = {}
        for stage, state_key in (("round0", "actor_state_dict"), ("latest", "latest_actor_state_dict"), ("selected", "selected_actor_state_dict")):
            payload = dict(source)
            if stage != "round0":
                payload[state_key] = checkpoint[state_key]
            actor = base.actor_from_payload(payload, state_key, device)
            actors[stage] = actor
            selection_action = base.actor_predict(actor, inputs, selection, device)
            oof_action = base.actor_predict(actor, inputs, oof, device)
            selection_cost = base.direct_cost(controller, data, selection, selection_action, weights)
            oof_cost = base.direct_cost(controller, data, oof, oof_action, weights)
            if stage == "round0":
                saved_selection_action = saved["selection_round_action"][0]
                saved_selection_cost = saved["selection_round_cost"][0]
            else:
                saved_selection_action = saved[f"{stage}_selection_action"]
                saved_selection_cost = saved[f"{stage}_selection_cost"]
            errors[f"{stage}_selection_action"] = max(errors[f"{stage}_selection_action"], max_error(selection_action, saved_selection_action))
            errors[f"{stage}_selection_cost"] = max(errors[f"{stage}_selection_cost"], max_error(selection_cost, saved_selection_cost))
            errors[f"{stage}_oof_action"] = max(errors[f"{stage}_oof_action"], max_error(oof_action, saved[f"{stage}_oof_action"]))
            errors[f"{stage}_oof_cost"] = max(errors[f"{stage}_oof_cost"], max_error(oof_cost, saved[f"{stage}_oof_cost"]))
            expected_metrics = base.warm_relative_metrics(selection_cost, data["warm_cost"][selection], data["speed_kph"][selection], data["variant_index"][selection])
            record_stage = record["initial"] if stage == "round0" else record[stage]
            errors["summary_metrics"] = max(errors["summary_metrics"], recursive_error(expected_metrics, record_stage["inner_warm_relative"]))
        selected_round = int(np.argmin(saved["selection_round_cost"].mean(axis=1)))
        run_checks["selected_round_reconstructed"] = selected_round == int(record["selected_round"]) == int(checkpoint["selected_round"])
        run_checks["selected_matches_round"] = max_error(saved["selected_selection_action"], saved["selection_round_action"][selected_round]) <= tolerance and max_error(saved["selected_selection_cost"], saved["selection_round_cost"][selected_round]) <= tolerance
        local = selection[:20]
        tensors = [torch.from_numpy(value[local]).to(device) for value in inputs]
        actor = actors["selected"]
        actor.eval()
        with torch.no_grad():
            reference = actor(*tensors)[1]
            for index in (3, 4, 5):
                changed = list(tensors)
                changed[index] = torch.randn_like(changed[index])
                errors["forbidden_actor_input"] = max(errors["forbidden_actor_input"], float(torch.max(torch.abs(actor(*changed)[1] - reference)).cpu()))

        with np.load(prior_records[seed]["arrays"], allow_pickle=False) as archive:
            prior = {name: np.asarray(archive[name]) for name in archive.files}
        candidate_prefix_count = prefix_rounds * contexts * candidates
        batch_prefix_count = prefix_rounds * 16 * int(config["actor_updates"]["batch_size"])
        comparisons = {
            "probe_radius": (saved["probe_radius_by_round"][:prefix_rounds], prior["probe_radius_by_round"]),
            "state_index": (saved["state_index"][:candidate_prefix_count], prior["state_index"]),
            "action": (saved["action"][:candidate_prefix_count], prior["action"]),
            "raw_action": (saved["raw_action"][:candidate_prefix_count], prior["raw_action"]),
            "cost": (saved["cost"][:candidate_prefix_count], prior["cost"]),
            "selection_round_action": (saved["selection_round_action"][:prefix_rounds + 1], prior["selection_round_action"]),
            "selection_round_cost": (saved["selection_round_cost"][:prefix_rounds + 1], prior["selection_round_cost"]),
            "actor_batch_state_index": (saved["actor_batch_state_index"][:batch_prefix_count], prior["actor_batch_state_index"]),
        }
        for name, (left, right) in comparisons.items():
            prefix_errors[name] = max(prefix_errors[name], max_error(left, right))
        reports.append({"seed": seed, "selected_round": selected_round, "checks": run_checks, "all_checks_pass": bool(all(run_checks.values()))})

    records = list(records_by_seed.values())
    pooled = {
        stage: {
            split: base.pooled_warm(records, stage, "inner" if split == "inner" else "oof", data)
            for split in ("inner", "development_oof")
        }
        for stage in ("round0", "latest", "selected")
    }
    errors["summary_metrics"] = max(errors["summary_metrics"], recursive_error(pooled, summary["pooled_warm_relative"]["gamma1_k16"]))
    prior_metric = prior_summary["pooled_warm_relative"]["gamma1_k16"]["selected"]["inner"]
    current_metric = pooled["selected"]["inner"]
    prior_speed100 = prior_metric["by_speed_kph"]["100"]
    current_speed100 = current_metric["by_speed_kph"]["100"]
    gate = config["decision_gate"]
    selected_after_90 = sum(int(record["selected_round"]) > prefix_rounds for record in records)
    mean_advantage = sum(record["selected"]["inner_warm_relative"]["actor_cost"]["mean"] < prior_records[int(record["seed"])]["selected"]["inner_warm_relative"]["actor_cost"]["mean"] for record in records)
    extension_checks = {
        "prefix_reproduced": max(prefix_errors.values()) <= tolerance,
        "selected_after_round90_seed_count": selected_after_90 >= int(gate["selected_round_after_90_seed_count_minimum"]),
        "actor_mean_cost_advantage_seed_count": mean_advantage >= int(gate["actor_mean_cost_advantage_seed_count_minimum"]),
        "pooled_aggregate_improvement": current_metric["aggregate_improvement"] > prior_metric["aggregate_improvement"],
        "pooled_median": current_metric["gain"]["median"] >= prior_metric["gain"]["median"] - float(gate["pooled_warm_relative_median_no_worse_tolerance"]),
        "pooled_p05": current_metric["gain"]["p05"] >= prior_metric["gain"]["p05"] - float(gate["pooled_warm_relative_p05_no_worse_tolerance"]),
        "pooled_worst": current_metric["gain"]["min"] >= prior_metric["gain"]["min"] - float(gate["pooled_warm_relative_worst_no_worse_tolerance"]),
        "speed100_aggregate_improvement": current_speed100["aggregate_improvement"] > prior_speed100["aggregate_improvement"],
        "speed100_p05": current_speed100["gain"]["p05"] >= prior_speed100["gain"]["p05"] - float(gate["speed100_p05_no_worse_tolerance"]),
        "speed100_worst": current_speed100["gain"]["min"] >= prior_speed100["gain"]["min"] - float(gate["speed100_worst_no_worse_tolerance"]),
    }
    decision = "GAMMA1_K16_160_EXTENDS_90ROUND_LEARNING" if all(extension_checks.values()) else "GAMMA1_K16_160_NO_RELIABLE_EXTENSION"
    warm_spec = config["inner_warm_gate"]
    warm_checks = {
        "pooled_aggregate_improvement": current_metric["aggregate_improvement"] >= float(warm_spec["pooled_aggregate_improvement_minimum"]),
        "pooled_gain_median": current_metric["gain"]["median"] >= float(warm_spec["pooled_gain_median_minimum"]),
        "speed100_aggregate_improvement": current_speed100["aggregate_improvement"] >= float(warm_spec["speed100_aggregate_improvement_minimum"]),
    }
    checks.update({
        "all_run_checks": len(reports) == 3 and all(report["all_checks_pass"] for report in reports),
        "all_replay_errors_zero": all(value <= tolerance for value in errors.values()),
        "prefix_errors_reconstructed": recursive_error(prefix_errors, summary["prefix_reproduction_maximum_absolute_errors"]) <= 1e-12,
        "extension_checks_reconstructed": extension_checks == summary["extension_checks"],
        "warm_gate_reconstructed": warm_checks == summary["inner_warm_gate"]["checks"] and bool(all(warm_checks.values())) == bool(summary["inner_warm_gate"]["passes"]),
        "decision_reconstructed": decision == summary["decision"] == manifest["decision"],
    })
    passed = bool(all(checks.values()))
    qualification = "QUERY_OAC_GAMMA1_K16_160ROUND_INDEPENDENT_PASS" if passed else "QUERY_OAC_GAMMA1_K16_160ROUND_INDEPENDENT_FAIL"
    validation = {
        "qualification": qualification,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "decision": decision,
        "checks": checks,
        "run_reports": reports,
        "maximum_absolute_errors": errors,
        "prefix_reproduction_maximum_absolute_errors": prefix_errors,
        "recomputed_extension_checks": extension_checks,
        "recomputed_inner_warm_gate": {"checks": warm_checks, "passes": bool(all(warm_checks.values()))},
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    validation_path = output / "validation.json"
    base.dump_json(validation_path, validation)
    if passed:
        manifest["qualification"] = qualification
        manifest["validation"] = str(validation_path)
        manifest["validation_sha256"] = base.sha256(validation_path)
        manifest["validator"] = str(Path(__file__).resolve())
        manifest["validator_sha256"] = base.sha256(Path(__file__).resolve())
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "qualification": qualification,
        "decision": decision,
        "failed_checks": [name for name, value in checks.items() if not value],
        "failed_runs": [f"seed{report['seed']}" for report in reports if not report["all_checks_pass"]],
        "maximum_absolute_errors": errors,
        "prefix_errors": prefix_errors,
        "extension_checks": extension_checks,
        "inner_warm_gate": validation["recomputed_inner_warm_gate"],
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
