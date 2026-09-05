#!/usr/bin/env python3
"""Independently validate the shared-prefix Query OAC LR-decay comparison."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_query_oac_gamma1_k16_lr_decay_160round as runner
import run_query_oac_gamma1_k_scan as base
from run_query_single_center_oac20to1 import (
    actor_from_payload,
    actor_predict,
    direct_cost,
    load_inputs,
    response_bank,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_oac_gamma1_k16_lr_decay_160round_20260903_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def max_error(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, np.float64)
    right64 = np.asarray(right, np.float64)
    return 0.0 if left64.size == 0 else float(np.max(np.abs(left64 - right64)))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def equal_prefix(branch: dict[str, np.ndarray], prefix: dict[str, np.ndarray], prefix_rounds: int) -> bool:
    online_mask = branch["round"] <= prefix_rounds
    batch_mask = branch["actor_batch_round"] <= prefix_rounds
    online_names = ("state_index", "action", "cost", "raw_action", "clipped", "round", "group", "role")
    batch_names = ("actor_batch_state_index", "actor_batch_round", "actor_batch_microstep")
    return bool(
        all(np.array_equal(branch[name][online_mask], prefix[name]) for name in online_names)
        and all(np.array_equal(branch[name][batch_mask], prefix[name]) for name in batch_names)
        and np.array_equal(branch["selection_round_action"][: prefix_rounds + 1], prefix["selection_round_action"])
        and np.array_equal(branch["selection_round_cost"][: prefix_rounds + 1], prefix["selection_round_cost"])
        and np.array_equal(branch["round0_oof_action"], prefix["round0_oof_action"])
        and np.array_equal(branch["round0_oof_cost"], prefix["round0_oof_cost"])
    )


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
    replay_manifest = json.loads((replay_dir / "manifest.json").read_text())
    data = load_npz(replay_dir / "replay.npz")
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
    rounds = int(config["pilot"]["rounds"])
    prefix_rounds = int(config["common_prefix"]["rounds"])
    contexts = int(config["pilot"]["fit_contexts_visited_per_round"])
    candidates = int(config["pilot"]["candidates_per_visit"])
    anneal = np.linspace(
        float(config["pilot"]["probe_radius_sigma_start"]),
        float(config["pilot"]["probe_radius_sigma_end"]),
        int(config["pilot"]["probe_radius_sigma_anneal_rounds"]),
    ).astype(np.float32)
    radii = np.concatenate((anneal, np.full(rounds - len(anneal), anneal[-1], np.float32)))
    bases = base.basis_bank()
    checks: dict[str, bool] = {
        "config_hash": base.sha256(config_path) == manifest["config_sha256"],
        "runner_hash": base.sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "shared_runner_hash": base.sha256(Path(manifest["shared_runner"])) == manifest["shared_runner_sha256"],
        "summary_hash": base.sha256(summary_path) == manifest["summary_sha256"],
        "replay_hash": base.sha256(replay_dir / "replay.npz") == manifest["absolute_replay_sha256"],
        "pretrain_manifest_hash": base.sha256(pretrain_dir / "manifest.json") == manifest["pretrain_manifest_sha256"],
        "query_checkpoint_hash": base.sha256(Path(manifest["query_checkpoint"])) == manifest["query_checkpoint_sha256"],
        "source_lr_scan_hash": base.sha256(Path(manifest["lr_scan_60round"]) / "validation.json") == manifest["lr_scan_validation_sha256"],
        "source_k16_160_hash": base.sha256(Path(manifest["k16_160round"]) / "validation.json") == manifest["k16_160round_validation_sha256"],
        "shared_prefix_registered": prefix_rounds == 60 and bool(config["pairing"]["shared_prefix_exact_by_construction"]),
        "rounds_registered": rounds == 160,
        "arms_registered": [arm["name"] for arm in config["arms"]] == ["fixed_lr1e5", "cosine_lr1e5_to_2e6", "switch_lr2e6"],
        "gamma_fixed_one": float(config["actor_cost_weight_gamma"]) == 1.0,
        "critic_updates_fixed_20": int(config["critic_updates"]["updates_per_round_per_twin"]) == 20,
        "deterministic_warn_only": summary["deterministic_runtime_contract"]["mode"] == "warn_only",
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
    }
    arrays_by_key: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    records_by_key = {(str(item["arm"]), int(item["seed"])): item for item in summary["records"]}
    reports: list[dict[str, Any]] = []
    replayed_groups: set[tuple[int, int]] = set()
    for record in summary["records"]:
        arm = str(record["arm"])
        seed = int(record["seed"])
        key = f"{arm}_seed{seed}"
        arrays_path = Path(record["arrays"])
        checkpoint_path = Path(record["checkpoint"])
        prefix_path = Path(record["prefix_arrays"])
        fork_path = Path(record["fork_state"])
        saved = load_npz(arrays_path)
        prefix = load_npz(prefix_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        fork = torch.load(fork_path, map_location="cpu", weights_only=False)
        arrays_by_key[(arm, seed)] = saved
        fit, selection, oof = saved["fit_indices"], saved["selection_indices"], saved["oof_indices"]
        source_path = pretrain_dir / "checkpoints" / f"pretrain_fold0_seed{seed}.pt"
        run_checks = {
            "arrays_hash": base.sha256(arrays_path) == record["arrays_sha256"] == manifest["run_artifacts"][key]["arrays_sha256"],
            "checkpoint_hash": base.sha256(checkpoint_path) == record["checkpoint_sha256"] == manifest["run_artifacts"][key]["checkpoint_sha256"],
            "prefix_hash": base.sha256(prefix_path) == record["prefix_arrays_sha256"] == manifest["run_artifacts"][key]["prefix_arrays_sha256"],
            "fork_hash": base.sha256(fork_path) == record["fork_state_sha256"] == manifest["run_artifacts"][key]["fork_state_sha256"],
            "source_checkpoint": Path(checkpoint["source_pretrain_checkpoint"]) == source_path and base.sha256(source_path) == checkpoint["source_pretrain_checkpoint_sha256"],
            "source_replay": checkpoint["source_replay_sha256"] == manifest["absolute_replay_sha256"],
            "split_sizes": (len(fit), len(selection), len(oof)) == (360, 120, 120),
            "fit_only": bool(np.all(np.isin(saved["state_index"], fit)) and not np.any(np.isin(saved["state_index"], selection)) and not np.any(np.isin(saved["state_index"], oof))),
            "online_count": len(saved["cost"]) == rounds * contexts * candidates == int(record["online_replay_rows"]),
            "actor_batch_count": len(saved["actor_batch_state_index"]) == rounds * 16 * int(config["actor_updates"]["batch_size"]) == int(record["actor_batch_rows"]),
            "prefix_online_count": len(prefix["cost"]) == prefix_rounds * contexts * candidates,
            "prefix_batch_count": len(prefix["actor_batch_state_index"]) == prefix_rounds * 16 * int(config["actor_updates"]["batch_size"]),
            "prefix_exact": equal_prefix(saved, prefix, prefix_rounds),
            "fork_complete": all(name in fork for name in (
                "actor_state_dict", "critic1_state_dict", "critic2_state_dict",
                "actor_optimizer_state_dict", "critic1_optimizer_state_dict", "critic2_optimizer_state_dict",
                "critic_rng_state", "torch_cpu_rng_state", "torch_cuda_rng_states",
                "selected_actor_state_dict", "selected_critic1_state_dict", "selected_critic2_state_dict",
            )),
            "fork_round": int(fork["fork_round"]) == prefix_rounds,
            "counts": int(checkpoint["actor_update_count"]) == rounds * 16 and int(checkpoint["critic_update_count_per_twin"]) == rounds * 20,
            "bounds": bool(np.all(np.isfinite(saved["action"])) and np.all(np.isfinite(saved["cost"])) and np.all(np.abs(saved["action"]) <= 1.0 + 1e-7)),
            "round_records": len(record["rounds"]) == rounds,
            "lr_schedule": all(
                abs(float(item["actor_update"]["learning_rate_per_microstep"]) - runner.lr_for_round(config, next(value for value in config["arms"] if value["name"] == arm), int(item["round"]))) <= 1e-15
                for item in record["rounds"]
            ),
            "cap_respected": all(
                float(item["actor_update"]["final_cumulative_output_step_sigma_rms"]) <= float(config["actor_updates"]["cumulative_per_round_output_step_cap_sigma_rms"]) + 1e-9
                and float(item["actor_update"]["final_cumulative_output_step_sigma_rms"]) <= float(item["actor_update"]["raw_cumulative_output_step_sigma_rms"]) + 1e-9
                for item in record["rounds"]
            ),
            "microsteps": all(int(item["actor_update"]["microstep_count"]) == 16 for item in record["rounds"]),
            "sealed_checkpoint": not bool(checkpoint["formal_validation_or_test_consumed"]) and not bool(checkpoint["dbm_fields_or_labels_consumed"]) and not bool(checkpoint["query_analytic_gradient_consumed"]),
        }
        episodes = [set(data["episode_id"][rows].tolist()) for rows in (fit, selection, oof)]
        run_checks["episode_disjoint"] = not bool(episodes[0] & episodes[1] or episodes[0] & episodes[2] or episodes[1] & episodes[2])
        for round_index in range(1, rounds + 1):
            mask = saved["round"] == round_index
            groups = np.unique(saved["group"][mask])
            run_checks[f"round_{round_index}_shape"] = bool(mask.sum() == contexts * candidates and len(groups) == contexts)
            rows = np.asarray([saved["state_index"][np.flatnonzero(saved["group"] == group)[0]] for group in groups])
            run_checks[f"round_{round_index}_stratified"] = len(set(zip(data["speed_index"][rows].tolist(), data["variant_index"][rows].tolist()))) == contexts
            roles_ok = True
            for group in groups:
                positions = np.flatnonzero(saved["group"] == group)
                roles = saved["role"][positions]
                roles_ok = roles_ok and bool(len(positions) == candidates and np.sum(roles == "actor") == 1 and np.sum(roles == "probe") == 32 and np.sum(roles == "response") == 6)
                replay_key = (seed, int(group)) if round_index <= prefix_rounds else (seed + 1000 * (1 + [value["name"] for value in config["arms"]].index(arm)), int(group))
                if replay_key not in replayed_groups:
                    row = int(saved["state_index"][positions[0]])
                    actions, costs, raw, clipped = response_bank(
                        controller, data, row, saved["action"][positions[0]],
                        float(radii[round_index - 1]), bases[(round_index - 1) % len(bases)],
                        sigma, weights, config,
                    )
                    errors["candidate_action"] = max(errors["candidate_action"], max_error(actions, saved["action"][positions]))
                    errors["candidate_raw_action"] = max(errors["candidate_raw_action"], max_error(raw, saved["raw_action"][positions]))
                    errors["candidate_cost"] = max(errors["candidate_cost"], max_error(costs, saved["cost"][positions]))
                    roles_ok = roles_ok and bool(np.array_equal(clipped, saved["clipped"][positions]))
                    replayed_groups.add(replay_key)
            run_checks[f"round_{round_index}_roles_replay"] = roles_ok
        source = torch.load(source_path, map_location=device, weights_only=False)
        inputs = load_inputs(data, checkpoint["normalization"])
        for stage, state_key in (("round0", "actor_state_dict"), ("latest", "latest_actor_state_dict"), ("selected", "selected_actor_state_dict")):
            payload = dict(source)
            if stage != "round0":
                payload[state_key] = checkpoint[state_key]
            actor = actor_from_payload(payload, state_key, device)
            selection_action = actor_predict(actor, inputs, selection, device)
            oof_action = actor_predict(actor, inputs, oof, device)
            selection_cost = direct_cost(controller, data, selection, selection_action, weights)
            oof_cost = direct_cost(controller, data, oof, oof_action, weights)
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
        selected_round = int(np.argmin(saved["selection_round_cost"].mean(axis=1)))
        run_checks["selected_round"] = selected_round == int(record["selected_round"]) == int(checkpoint["selected_round"])
        run_checks["selected_matches_round"] = max_error(saved["selected_selection_action"], saved["selection_round_action"][selected_round]) <= 1e-6 and max_error(saved["selected_selection_cost"], saved["selection_round_cost"][selected_round]) <= 1e-6
        reports.append({"arm": arm, "seed": seed, "selected_round": selected_round, "checks": run_checks, "all_checks_pass": bool(all(run_checks.values()))})

    arms = [arm["name"] for arm in config["arms"]]
    pair_checks = {}
    for seed in config["pilot"]["seeds"]:
        reference = arrays_by_key[(arms[0], int(seed))]
        for arm in arms[1:]:
            candidate = arrays_by_key[(arm, int(seed))]
            pair_checks[f"seed{seed}_{arm}_splits"] = all(np.array_equal(reference[name], candidate[name]) for name in ("fit_indices", "selection_indices", "oof_indices"))
            pair_checks[f"seed{seed}_{arm}_visit_schedule"] = all(np.array_equal(reference[name], candidate[name]) for name in ("state_index", "round", "group", "role"))
            pair_checks[f"seed{seed}_{arm}_actor_batches"] = all(np.array_equal(reference[name], candidate[name]) for name in ("actor_batch_state_index", "actor_batch_round", "actor_batch_microstep"))
            left_prefix = reference["round"] <= prefix_rounds
            right_prefix = candidate["round"] <= prefix_rounds
            pair_checks[f"seed{seed}_{arm}_exact_online_prefix"] = all(np.array_equal(reference[name][left_prefix], candidate[name][right_prefix]) for name in ("action", "raw_action", "cost", "clipped"))
            pair_checks[f"seed{seed}_{arm}_exact_selection_prefix"] = np.array_equal(reference["selection_round_action"][: prefix_rounds + 1], candidate["selection_round_action"][: prefix_rounds + 1]) and np.array_equal(reference["selection_round_cost"][: prefix_rounds + 1], candidate["selection_round_cost"][: prefix_rounds + 1])
    pooled = {
        arm: {
            stage: {
                "inner": base.pooled_warm([records_by_key[(arm, int(seed))] for seed in config["pilot"]["seeds"]], stage, "inner", data),
                "development_oof": base.pooled_warm([records_by_key[(arm, int(seed))] for seed in config["pilot"]["seeds"]], stage, "oof", data),
            }
            for stage in ("round0", "latest", "selected")
        }
        for arm in arms
    }
    decision_checks, decision = runner.schedule_checks(pooled, summary["records"], config)
    checks["all_run_checks"] = all(item["all_checks_pass"] for item in reports)
    checks["all_pair_checks"] = all(pair_checks.values())
    checks["pooled_reconstructed"] = pooled == summary["pooled_warm_relative"]
    checks["decision_reconstructed"] = decision == summary["decision"] == manifest["decision"] and decision_checks == summary["decision_checks"]
    checks["replay_errors_zero"] = all(value <= 1e-6 for value in errors.values())
    passed = bool(all(checks.values()))
    qualification = "QUERY_OAC_GAMMA1_K16_LR_DECAY_160ROUND_INDEPENDENT_PASS" if passed else "QUERY_OAC_GAMMA1_K16_LR_DECAY_160ROUND_INDEPENDENT_FAIL"
    report = {
        "qualification": qualification,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "decision": decision,
        "checks": checks,
        "decision_checks": decision_checks,
        "pair_checks": pair_checks,
        "run_reports": reports,
        "maximum_absolute_errors": errors,
        "unique_replayed_response_groups": len(replayed_groups),
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    validation_path = output / "validation.json"
    base.dump_json(validation_path, report)
    if passed:
        manifest["qualification"] = qualification
        manifest["validation"] = str(validation_path)
        manifest["validation_sha256"] = base.sha256(validation_path)
        manifest["validator"] = str(Path(__file__).resolve())
        manifest["validator_sha256"] = base.sha256(Path(__file__).resolve())
        base.dump_json(manifest_path, manifest)
    print(json.dumps({
        "qualification": qualification,
        "decision": decision,
        "failed_checks": [name for name, value in checks.items() if not value],
        "failed_runs": [f"{item['arm']}_seed{item['seed']}" for item in reports if not item["all_checks_pass"]],
        "maximum_absolute_errors": errors,
        "unique_replayed_response_groups": len(replayed_groups),
        "decision_checks": decision_checks,
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
