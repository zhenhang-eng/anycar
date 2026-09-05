#!/usr/bin/env python3
"""Independently validate the paired gamma-1 Query OAC K scan."""

from __future__ import annotations

import argparse
import hashlib
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

from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from run_query_forward_response_landscape_pilot import basis_bank  # noqa: E402
from run_query_single_center_oac20to1 import (  # noqa: E402
    actor_from_payload,
    actor_predict,
    direct_cost,
    distribution,
    load_inputs,
    response_bank,
)


DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_oac_gamma1_k_scan_20260902_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def max_error(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, np.float64)
    right64 = np.asarray(right, np.float64)
    return 0.0 if left64.size == 0 else float(np.max(np.abs(left64 - right64)))


def warm_metrics(cost: np.ndarray, warm: np.ndarray) -> dict[str, Any]:
    cost64 = np.asarray(cost, np.float64).reshape(-1)
    warm64 = np.asarray(warm, np.float64).reshape(-1)
    gain = warm64 - cost64
    return {
        "actor_cost": distribution(cost64),
        "warm_cost": distribution(warm64),
        "gain": distribution(gain),
        "win_or_tie_fraction": float(np.mean(cost64 <= warm64)),
        "aggregate_improvement": float(gain.sum() / warm64.sum()),
    }


def compare_metrics(left: dict[str, Any], right: dict[str, Any], tolerance: float = 1e-10) -> bool:
    for group in ("actor_cost", "warm_cost", "gain"):
        for name in ("min", "p05", "p10", "median", "mean", "p90", "p95", "max"):
            if abs(float(left[group][name]) - float(right[group][name])) > tolerance:
                return False
    return (
        abs(float(left["win_or_tie_fraction"]) - float(right["win_or_tie_fraction"])) <= tolerance
        and abs(float(left["aggregate_improvement"]) - float(right["aggregate_improvement"])) <= tolerance
    )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    replay_dir = Path(manifest["absolute_replay"])
    pretrain_dir = Path(manifest["pretrain"])
    replay_manifest = json.loads((replay_dir / "manifest.json").read_text())
    with np.load(replay_dir / "replay.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    fullrank_manifest = json.loads((Path(replay_manifest["fullrank_source"]) / "manifest.json").read_text())
    parent_manifest = json.loads((Path(fullrank_manifest["parent_t0"]) / "manifest.json").read_text())
    collection_manifest = json.loads((Path(parent_manifest["source_collection"]) / "manifest.json").read_text())
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    sigma = np.asarray(config["pilot"]["noise_sigma"], np.float32).reshape(1, 2)
    rounds = int(config["pilot"]["rounds"])
    contexts = int(config["pilot"]["fit_contexts_visited_per_round"])
    candidates = int(config["pilot"]["candidates_per_visit"])
    expected_online = rounds * contexts * candidates
    radii = np.linspace(
        float(config["pilot"]["probe_radius_sigma_start"]),
        float(config["pilot"]["probe_radius_sigma_end"]),
        rounds,
    ).astype(np.float32)
    bases = basis_bank()
    checks: dict[str, bool] = {
        "config_hash": sha256(config_path) == manifest["config_sha256"],
        "runner_hash": sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "summary_hash": sha256(output / "summary.json") == manifest["summary_sha256"],
        "replay_hash": sha256(replay_dir / "replay.npz") == manifest["absolute_replay_sha256"],
        "pretrain_manifest_hash": sha256(pretrain_dir / "manifest.json") == manifest["pretrain_manifest_sha256"],
        "aggregation_validation_hash": sha256(Path(manifest["aggregation_ab"]) / "validation.json") == manifest["aggregation_validation_sha256"],
        "query_checkpoint_hash": sha256(Path(manifest["query_checkpoint"])) == manifest["query_checkpoint_sha256"],
        "gamma_fixed_one": float(config["actor_cost_weight_gamma"]) == 1.0,
        "k_arms_exact": [int(arm["actor_updates_per_round"]) for arm in config["arms"]] == [1, 4, 8],
        "critic_updates_fixed_20": int(config["critic_updates"]["updates_per_round_per_twin"]) == 20,
        "warm_is_external_only": config["evaluation_contract"]["warm_role"].startswith("external deterministic"),
        "decision_is_inner_only": summary["decision_population"] == "inner selected checkpoints only",
        "development_oof_not_untouched": "already-consumed" in summary["development_oof_role"],
        "formal_test_sealed": not bool(summary["formal_validation_or_test_consumed"])
        and not bool(manifest["formal_validation_or_test_consumed"]),
        "dbm_fields_absent": not bool(summary["dbm_fields_or_labels_consumed"])
        and not bool(manifest["dbm_fields_or_labels_consumed"]),
        "analytic_query_gradient_absent": not bool(summary["query_analytic_gradient_consumed"])
        and not bool(manifest["query_analytic_gradient_consumed"]),
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
        "forbidden_actor_input": 0.0,
        "paired_round1_action": 0.0,
        "paired_round1_cost": 0.0,
    }
    reports: list[dict[str, Any]] = []
    arrays_by_key: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    records_by_key = {(str(record["arm"]), int(record["seed"])): record for record in summary["records"]}

    for record in summary["records"]:
        arm = str(record["arm"])
        seed = int(record["seed"])
        k = int(record["actor_updates_per_round"])
        key = f"{arm}_seed{seed}"
        arrays_path = Path(record["arrays"])
        checkpoint_path = Path(record["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        with np.load(arrays_path, allow_pickle=False) as archive:
            saved = {name: np.asarray(archive[name]) for name in archive.files}
        arrays_by_key[(arm, seed)] = saved
        fit, selection, oof = saved["fit_indices"], saved["selection_indices"], saved["oof_indices"]
        source_path = pretrain_dir / "checkpoints" / f"pretrain_fold0_seed{seed}.pt"
        run_checks: dict[str, bool] = {
            "checkpoint_hash": sha256(checkpoint_path) == record["checkpoint_sha256"] == manifest["run_artifacts"][key]["checkpoint_sha256"],
            "arrays_hash": sha256(arrays_path) == record["arrays_sha256"] == manifest["run_artifacts"][key]["arrays_sha256"],
            "source_checkpoint_exact": Path(checkpoint["source_pretrain_checkpoint"]) == source_path,
            "source_checkpoint_hash": sha256(source_path) == checkpoint["source_pretrain_checkpoint_sha256"],
            "source_replay_hash": checkpoint["source_replay_sha256"] == manifest["absolute_replay_sha256"],
            "nested_split_sizes": (len(fit), len(selection), len(oof)) == (360, 120, 120),
            "nested_split_fold_exact": bool(
                np.all(np.isin(data["fold_id"][fit], config["split_contract"]["fit_folds"]))
                and np.all(data["fold_id"][selection] == config["split_contract"]["inner_selection_fold"])
                and np.all(data["fold_id"][oof] == config["split_contract"]["outer_fold"])
            ),
            "online_count": len(saved["cost"]) == expected_online == int(record["online_replay_rows"]),
            "online_fit_only": bool(np.all(np.isin(saved["state_index"], fit))),
            "online_excludes_selection_oof": bool(
                not np.any(np.isin(saved["state_index"], selection))
                and not np.any(np.isin(saved["state_index"], oof))
            ),
            "actor_batch_count": len(saved["actor_batch_state_index"]) == rounds * k * int(config["actor_updates"]["batch_size"]) == int(record["actor_batch_rows"]),
            "actor_batches_fit_only": bool(np.all(np.isin(saved["actor_batch_state_index"], fit))),
            "actor_update_count": int(checkpoint["actor_update_count"]) == rounds * k,
            "critic_update_count": int(checkpoint["critic_update_count_per_twin"]) == rounds * 20,
            "candidate_finite_and_bounded": bool(
                np.all(np.isfinite(saved["action"])) and np.all(np.isfinite(saved["cost"]))
                and np.all(np.abs(saved["action"]) <= 1.0 + 1e-7)
            ),
            "cumulative_cap_respected": all(
                float(item["actor_update"]["final_cumulative_output_step_sigma_rms"])
                <= float(config["actor_updates"]["cumulative_per_round_output_step_cap_sigma_rms"]) + 1e-9
                and float(item["actor_update"]["final_cumulative_output_step_sigma_rms"])
                <= float(item["actor_update"]["raw_cumulative_output_step_sigma_rms"]) + 1e-9
                for item in record["rounds"]
            ),
            "microstep_count_exact": all(
                int(item["actor_update"]["microstep_count"]) == k
                and len(item["actor_update"]["microsteps"]) == k
                and float(item["actor_update"]["cost_weight_gamma"]) == 1.0
                for item in record["rounds"]
            ),
            "new_query_rollout_count": int(record["new_query_rollouts"]) == expected_online + 15 * len(selection),
            "formal_test_sealed": not bool(checkpoint["formal_validation_or_test_consumed"]),
            "dbm_fields_absent": not bool(checkpoint["dbm_fields_or_labels_consumed"]),
            "analytic_query_gradient_absent": not bool(checkpoint["query_analytic_gradient_consumed"]),
        }
        episode_sets = [set(data["episode_id"][rows].tolist()) for rows in (fit, selection, oof)]
        run_checks["episode_disjoint"] = not bool(
            episode_sets[0] & episode_sets[1]
            or episode_sets[0] & episode_sets[2]
            or episode_sets[1] & episode_sets[2]
        )
        for round_index in range(1, rounds + 1):
            mask = saved["round"] == round_index
            groups = np.unique(saved["group"][mask])
            run_checks[f"round_{round_index}_shape"] = bool(
                mask.sum() == contexts * candidates and len(groups) == contexts
            )
            rows = np.asarray([
                saved["state_index"][np.flatnonzero(saved["group"] == group)[0]] for group in groups
            ])
            run_checks[f"round_{round_index}_stratified"] = len(set(zip(
                data["speed_index"][rows].tolist(), data["variant_index"][rows].tolist()
            ))) == contexts
            roles_ok, clipping_ok = True, True
            for group in groups:
                positions = np.flatnonzero(saved["group"] == group)
                roles = saved["role"][positions]
                roles_ok = roles_ok and bool(
                    len(positions) == candidates
                    and np.sum(roles == "actor") == 1
                    and np.sum(roles == "probe") == 32
                    and np.sum(roles == "response") == 6
                )
                row = int(saved["state_index"][positions[0]])
                actions, costs, raw, clipped = response_bank(
                    controller, data, row, saved["action"][positions[0]],
                    float(radii[round_index - 1]),
                    bases[(round_index - 1) % len(bases)], sigma, weights, config,
                )
                errors["candidate_action"] = max(errors["candidate_action"], max_error(actions, saved["action"][positions]))
                errors["candidate_raw_action"] = max(errors["candidate_raw_action"], max_error(raw, saved["raw_action"][positions]))
                errors["candidate_cost"] = max(errors["candidate_cost"], max_error(costs, saved["cost"][positions]))
                clipping_ok = clipping_ok and bool(np.array_equal(clipped, saved["clipped"][positions]))
            run_checks[f"round_{round_index}_roles"] = roles_ok
            run_checks[f"round_{round_index}_clipping"] = clipping_ok
            local_batch = saved["actor_batch_round"] == round_index
            counts = [
                np.sum(local_batch & (saved["actor_batch_microstep"] == microstep))
                for microstep in range(1, k + 1)
            ]
            run_checks[f"round_{round_index}_actor_batches"] = counts == [int(config["actor_updates"]["batch_size"])] * k

        source = torch.load(source_path, map_location=device, weights_only=False)
        inputs = load_inputs(data, checkpoint["normalization"])
        actors = {}
        for stage, state_key in (
            ("round0", "actor_state_dict"),
            ("latest", "latest_actor_state_dict"),
            ("selected", "selected_actor_state_dict"),
        ):
            payload = dict(source)
            if stage != "round0":
                payload[state_key] = checkpoint[state_key]
            actor = actor_from_payload(payload, state_key, device)
            actors[stage] = actor
            selection_action = actor_predict(actor, inputs, selection, device)
            oof_action = actor_predict(actor, inputs, oof, device)
            selection_cost = direct_cost(controller, data, selection, selection_action, weights)
            oof_cost = direct_cost(controller, data, oof, oof_action, weights)
            if stage == "round0":
                selection_action_saved = saved["selection_round_action"][0]
                selection_cost_saved = saved["selection_round_cost"][0]
            else:
                selection_action_saved = saved[f"{stage}_selection_action"]
                selection_cost_saved = saved[f"{stage}_selection_cost"]
            errors[f"{stage}_selection_action"] = max(errors[f"{stage}_selection_action"], max_error(selection_action, selection_action_saved))
            errors[f"{stage}_selection_cost"] = max(errors[f"{stage}_selection_cost"], max_error(selection_cost, selection_cost_saved))
            errors[f"{stage}_oof_action"] = max(errors[f"{stage}_oof_action"], max_error(oof_action, saved[f"{stage}_oof_action"]))
            errors[f"{stage}_oof_cost"] = max(errors[f"{stage}_oof_cost"], max_error(oof_cost, saved[f"{stage}_oof_cost"]))
            record_stage = record["initial"] if stage == "round0" else record[stage]
            run_checks[f"{stage}_inner_warm_metrics"] = compare_metrics(
                warm_metrics(selection_cost_saved, data["warm_cost"][selection]),
                record_stage["inner_warm_relative"],
            )
            run_checks[f"{stage}_oof_warm_metrics"] = compare_metrics(
                warm_metrics(saved[f"{stage}_oof_cost"], data["warm_cost"][oof]),
                record_stage["development_oof_warm_relative"],
            )
        selected_round = int(np.argmin(saved["selection_round_cost"].mean(axis=1)))
        run_checks["selected_round_reconstructed"] = selected_round == int(record["selected_round"]) == int(checkpoint["selected_round"])
        run_checks["selected_selection_matches_round"] = (
            max_error(saved["selected_selection_action"], saved["selection_round_action"][selected_round]) <= 1e-6
            and max_error(saved["selected_selection_cost"], saved["selection_round_cost"][selected_round]) <= 1e-6
        )
        actor = actors["selected"]
        local = selection[:20]
        tensors = [torch.from_numpy(value[local]).to(device) for value in inputs]
        actor.eval()
        with torch.no_grad():
            reference = actor(*tensors)[1]
            for index in (3, 4, 5):
                changed = list(tensors)
                changed[index] = torch.randn_like(changed[index])
                errors["forbidden_actor_input"] = max(
                    errors["forbidden_actor_input"],
                    float(torch.max(torch.abs(actor(*changed)[1] - reference)).cpu()),
                )
        reports.append({
            "arm": arm,
            "seed": seed,
            "selected_round": selected_round,
            "checks": run_checks,
            "all_checks_pass": all(run_checks.values()),
        })

    arms = [str(arm["name"]) for arm in config["arms"]]
    seeds = [int(seed) for seed in config["pilot"]["seeds"]]
    pair_checks: dict[str, bool] = {}
    for seed in seeds:
        reference = arrays_by_key[(arms[0], seed)]
        for arm in arms[1:]:
            candidate = arrays_by_key[(arm, seed)]
            pair_checks[f"seed_{seed}_{arm}_splits"] = all(np.array_equal(
                reference[name], candidate[name]
            ) for name in ("fit_indices", "selection_indices", "oof_indices"))
            pair_checks[f"seed_{seed}_{arm}_visit_schedule"] = all(np.array_equal(
                reference[name], candidate[name]
            ) for name in ("state_index", "round", "group", "role"))
            round1 = reference["round"] == 1
            errors["paired_round1_action"] = max(errors["paired_round1_action"], max_error(
                reference["action"][round1], candidate["action"][round1]
            ))
            errors["paired_round1_cost"] = max(errors["paired_round1_cost"], max_error(
                reference["cost"][round1], candidate["cost"][round1]
            ))
            for round_index in range(1, rounds + 1):
                for microstep in range(1, int(records_by_key[(arms[0], seed)]["actor_updates_per_round"]) + 1):
                    left_mask = (reference["actor_batch_round"] == round_index) & (reference["actor_batch_microstep"] == microstep)
                    right_mask = (candidate["actor_batch_round"] == round_index) & (candidate["actor_batch_microstep"] == microstep)
                    pair_checks[f"seed_{seed}_{arm}_round_{round_index}_batch_prefix_{microstep}"] = np.array_equal(
                        reference["actor_batch_state_index"][left_mask], candidate["actor_batch_state_index"][right_mask]
                    )
        for lower_arm, higher_arm in zip(arms[:-1], arms[1:]):
            lower = arrays_by_key[(lower_arm, seed)]
            higher = arrays_by_key[(higher_arm, seed)]
            lower_k = int(records_by_key[(lower_arm, seed)]["actor_updates_per_round"])
            for round_index in range(1, rounds + 1):
                for microstep in range(1, lower_k + 1):
                    lower_mask = (lower["actor_batch_round"] == round_index) & (lower["actor_batch_microstep"] == microstep)
                    higher_mask = (higher["actor_batch_round"] == round_index) & (higher["actor_batch_microstep"] == microstep)
                    pair_checks[f"seed_{seed}_{lower_arm}_{higher_arm}_round_{round_index}_batch_prefix_{microstep}"] = np.array_equal(
                        lower["actor_batch_state_index"][lower_mask], higher["actor_batch_state_index"][higher_mask]
                    )

    pooled: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in arms:
        pooled[arm] = {}
        local_records = [records_by_key[(arm, seed)] for seed in seeds]
        for stage in ("round0", "latest", "selected"):
            pooled[arm][stage] = {}
            for split, rows_key, cost_key in (
                ("inner", "selection_indices", f"{stage}_selection_cost"),
                ("development_oof", "oof_indices", f"{stage}_oof_cost"),
            ):
                costs, warms = [], []
                for record in local_records:
                    saved = arrays_by_key[(arm, int(record["seed"]))]
                    rows = saved[rows_key]
                    if stage == "round0" and split == "inner":
                        cost = saved["selection_round_cost"][0]
                    else:
                        cost = saved[cost_key]
                    costs.append(cost)
                    warms.append(data["warm_cost"][rows])
                pooled[arm][stage][split] = warm_metrics(np.concatenate(costs), np.concatenate(warms))
                checks[f"pooled_{arm}_{stage}_{split}"] = compare_metrics(
                    pooled[arm][stage][split],
                    summary["pooled_warm_relative"][arm][stage][split],
                )

    baseline = arms[0]
    gate = config["decision_gate"]
    candidate_checks = {}
    decision = "GAMMA1_K_SCAN_FAIL_RETAIN_K1_COST_REFERENCE"
    for candidate in gate["candidate_order"]:
        seed_count = sum(
            records_by_key[(candidate, seed)]["selected"]["inner_warm_relative"]["actor_cost"]["mean"]
            < records_by_key[(baseline, seed)]["selected"]["inner_warm_relative"]["actor_cost"]["mean"]
            for seed in seeds
        )
        candidate_primary = pooled[candidate]["selected"]["inner"]
        baseline_primary = pooled[baseline]["selected"]["inner"]
        local_checks = {
            "actor_mean_cost_advantage_seed_count": seed_count >= int(gate["candidate_actor_mean_cost_advantage_seed_count_minimum"]),
            "pooled_aggregate_improvement_greater_than_k1": candidate_primary["aggregate_improvement"] > baseline_primary["aggregate_improvement"],
            "pooled_median_within_tolerance": candidate_primary["gain"]["median"] >= baseline_primary["gain"]["median"] - float(gate["candidate_pooled_warm_relative_median_no_worse_tolerance"]),
            "pooled_p05_within_tolerance": candidate_primary["gain"]["p05"] >= baseline_primary["gain"]["p05"] - float(gate["candidate_pooled_warm_relative_p05_no_worse_tolerance"]),
            "pooled_worst_within_tolerance": candidate_primary["gain"]["min"] >= baseline_primary["gain"]["min"] - float(gate["candidate_pooled_warm_relative_worst_no_worse_tolerance"]),
        }
        candidate_checks[candidate] = {
            "mean_cost_advantage_seed_count": int(seed_count),
            "checks": local_checks,
            "passes": bool(all(local_checks.values())),
        }
        if decision.endswith("K1_COST_REFERENCE") and all(local_checks.values()):
            decision = f"ADVANCE_{candidate.upper()}_TO_LONGER_GAMMA1_QUERY_PILOT"
    checks["all_run_checks"] = all(report["all_checks_pass"] for report in reports)
    checks["all_pair_checks"] = all(pair_checks.values())
    checks["candidate_checks_reconstructed"] = candidate_checks == summary["candidate_checks"]
    checks["decision_reconstructed"] = decision == summary["decision"] == manifest["decision"]
    checks["replay_errors_zero"] = all(value <= 1e-6 for value in errors.values())
    passed = all(checks.values())
    validation = {
        "qualification": "QUERY_OAC_GAMMA1_K_SCAN_INDEPENDENT_PASS" if passed else "QUERY_OAC_GAMMA1_K_SCAN_INDEPENDENT_FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "checks": checks,
        "pair_checks": pair_checks,
        "run_reports": reports,
        "maximum_absolute_errors": errors,
        "recomputed_candidate_checks": candidate_checks,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    validation_path = output / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "qualification": validation["qualification"],
        "decision": decision,
        "failed_checks": [name for name, value in checks.items() if not value],
        "failed_runs": [f"{item['arm']}_seed{item['seed']}" for item in reports if not item["all_checks_pass"]],
        "maximum_absolute_errors": errors,
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
