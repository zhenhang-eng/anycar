#!/usr/bin/env python3
"""Independently validate the paired gamma-0/gamma-1 continuous Query OAC pilot."""

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


DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_oac_aggregation_ab_20260902_v1"


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


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def max_error(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, np.float64)
    right64 = np.asarray(right, np.float64)
    if left64.size == 0:
        return 0.0
    return float(np.max(np.abs(left64 - right64)))


def by_speed_mean_gain(
    data: dict[str, np.ndarray], rows: np.ndarray, gain: np.ndarray
) -> dict[str, float]:
    return {
        str(int(speed)): float(np.mean(gain[data["speed_kph"][rows] == speed]))
        for speed in sorted(np.unique(data["speed_kph"][rows]).tolist())
    }


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
    expected_rows = rounds * contexts * candidates
    radii = np.linspace(
        float(config["pilot"]["probe_radius_sigma_start"]),
        float(config["pilot"]["probe_radius_sigma_end"]),
        rounds,
    ).astype(np.float32)
    bases = basis_bank()

    checks: dict[str, bool] = {
        "config_hash": sha256(config_path) == manifest["config_sha256"],
        "runner_hash": sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "summary_hash": sha256(summary_path) == manifest["summary_sha256"],
        "replay_hash": sha256(replay_dir / "replay.npz") == manifest["absolute_replay_sha256"],
        "pretrain_manifest_hash": sha256(pretrain_dir / "manifest.json") == manifest["pretrain_manifest_sha256"],
        "microstep_validation_hash": sha256(Path(manifest["microstep_diagnostic"]) / "validation.json") == manifest["microstep_validation_sha256"],
        "query_checkpoint_hash": sha256(Path(manifest["query_checkpoint"])) == manifest["query_checkpoint_sha256"],
        "ratio_is_20_to_1": int(config["critic_updates"]["updates_per_actor_update_per_twin"]) == 20
        and int(config["actor_updates"]["updates_per_round"]) == 1,
        "pairing_contract_registered": all(bool(config["pairing"][name]) for name in (
            "same_pretrain_checkpoint", "same_context_queue_and_basis_stream",
            "same_critic_and_actor_update_budget", "same_learning_rates_and_trust",
        )),
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
        "selected_selection_action": 0.0,
        "selected_selection_cost": 0.0,
        "selected_oof_action": 0.0,
        "selected_oof_cost": 0.0,
        "forbidden_actor_input": 0.0,
        "paired_round1_action": 0.0,
        "paired_round1_cost": 0.0,
    }
    reports: list[dict[str, Any]] = []
    saved_by_key: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    record_by_key = {(record["arm"], int(record["seed"])): record for record in summary["records"]}

    for record in summary["records"]:
        arm = str(record["arm"])
        seed = int(record["seed"])
        key = f"{arm}_seed{seed}"
        arrays_path = Path(record["arrays"])
        checkpoint_path = Path(record["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        with np.load(arrays_path, allow_pickle=False) as archive:
            saved = {name: np.asarray(archive[name]) for name in archive.files}
        saved_by_key[(arm, seed)] = saved
        fit = saved["fit_indices"]
        selection = saved["selection_indices"]
        oof = saved["oof_indices"]
        expected_source = pretrain_dir / "checkpoints" / f"pretrain_fold0_seed{seed}.pt"
        run_checks: dict[str, bool] = {
            "checkpoint_hash": sha256(checkpoint_path) == record["checkpoint_sha256"] == manifest["run_artifacts"][key]["checkpoint_sha256"],
            "arrays_hash": sha256(arrays_path) == record["arrays_sha256"] == manifest["run_artifacts"][key]["arrays_sha256"],
            "source_checkpoint_exact": Path(checkpoint["source_pretrain_checkpoint"]) == expected_source,
            "source_checkpoint_hash": sha256(expected_source) == checkpoint["source_pretrain_checkpoint_sha256"],
            "source_replay_hash": checkpoint["source_replay_sha256"] == manifest["absolute_replay_sha256"],
            "nested_split_sizes": (len(fit), len(selection), len(oof)) == (360, 120, 120),
            "nested_split_fold_exact": bool(
                np.all(np.isin(data["fold_id"][fit], config["split_contract"]["fit_folds"]))
                and np.all(data["fold_id"][selection] == config["split_contract"]["inner_selection_fold"])
                and np.all(data["fold_id"][oof] == config["split_contract"]["outer_fold"])
            ),
            "online_row_count": len(saved["cost"]) == expected_rows == int(record["online_replay_rows"]),
            "online_state_fit_only": bool(np.all(np.isin(saved["state_index"], fit))),
            "online_state_excludes_selection_oof": bool(
                not np.any(np.isin(saved["state_index"], selection))
                and not np.any(np.isin(saved["state_index"], oof))
            ),
            "candidate_finite": bool(np.all(np.isfinite(saved["action"])) and np.all(np.isfinite(saved["cost"]))),
            "candidate_bounds": bool(np.all(np.abs(saved["action"]) <= 1.0 + 1e-7)),
            "update_counts": int(checkpoint["actor_update_count"]) == rounds
            and int(checkpoint["critic_update_count_per_twin"]) == rounds * 20,
            "selected_round_bounds": 0 <= int(checkpoint["selected_round"]) <= rounds,
            "formal_test_sealed": not bool(checkpoint["formal_validation_or_test_consumed"]),
            "dbm_fields_absent": not bool(checkpoint["dbm_fields_or_labels_consumed"]),
            "analytic_query_gradient_absent": not bool(checkpoint["query_analytic_gradient_consumed"]),
            "actor_step_cap_respected": all(
                float(item["actor_update"]["final_output_step_sigma_rms"])
                <= float(config["actor_updates"]["per_round_output_step_cap_sigma_rms"]) + 1e-9
                and float(item["actor_update"]["final_output_step_sigma_rms"])
                <= float(item["actor_update"]["raw_output_step_sigma_rms"]) + 1e-9
                for item in record["rounds"]
            ),
            "actor_weight_gamma_exact": all(
                abs(float(item["actor_update"]["cost_weight_gamma"]) - float(record["gamma"])) <= 1e-12
                for item in record["rounds"]
            ),
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
                saved["state_index"][np.flatnonzero(saved["group"] == group)[0]]
                for group in groups
            ])
            cells = set(zip(data["speed_index"][rows].tolist(), data["variant_index"][rows].tolist()))
            run_checks[f"round_{round_index}_stratified"] = len(cells) == contexts
            roles_ok = True
            clipping_ok = True
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

        source_payload = torch.load(expected_source, map_location=device, weights_only=False)
        actor_payload = dict(source_payload)
        actor_payload["selected_actor_state_dict"] = checkpoint["selected_actor_state_dict"]
        inputs = load_inputs(data, checkpoint["normalization"])
        actor = actor_from_payload(actor_payload, "selected_actor_state_dict", device)
        selection_action = actor_predict(actor, inputs, selection, device)
        oof_action = actor_predict(actor, inputs, oof, device)
        selection_cost = direct_cost(controller, data, selection, selection_action, weights)
        oof_cost = direct_cost(controller, data, oof, oof_action, weights)
        errors["selected_selection_action"] = max(errors["selected_selection_action"], max_error(selection_action, saved["selected_selection_action"]))
        errors["selected_selection_cost"] = max(errors["selected_selection_cost"], max_error(selection_cost, saved["selected_selection_cost"]))
        errors["selected_oof_action"] = max(errors["selected_oof_action"], max_error(oof_action, saved["selected_oof_action"]))
        errors["selected_oof_cost"] = max(errors["selected_oof_cost"], max_error(oof_cost, saved["selected_oof_cost"]))
        selected_round = int(np.argmin(saved["selection_round_cost"].mean(axis=1)))
        run_checks["selected_round_reconstructed"] = selected_round == int(checkpoint["selected_round"]) == int(record["selected_round"])
        run_checks["selected_selection_matches_selected_round"] = (
            max_error(saved["selected_selection_action"], saved["selection_round_action"][selected_round]) <= 1e-6
            and max_error(saved["selected_selection_cost"], saved["selection_round_cost"][selected_round]) <= 1e-6
        )
        oof_gain = saved["round0_oof_cost"].astype(np.float64) - saved["selected_oof_cost"].astype(np.float64)
        run_checks["record_oof_mean_reconstructed"] = abs(
            float(record["selected_oof"]["gain_vs_round0"]["mean"]) - float(oof_gain.mean())
        ) <= 1e-10
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
            "checks": run_checks,
            "all_checks_pass": all(run_checks.values()),
            "selected_round": selected_round,
            "oof_gain_mean": float(oof_gain.mean()),
            "oof_gain_by_speed_mean": by_speed_mean_gain(data, oof, oof_gain),
        })

    arm0 = config["arms"][0]["name"]
    arm1 = config["arms"][1]["name"]
    seeds = [int(seed) for seed in config["pilot"]["seeds"]]
    paired_checks: dict[str, bool] = {}
    for seed in seeds:
        left = saved_by_key[(arm0, seed)]
        right = saved_by_key[(arm1, seed)]
        paired_checks[f"seed_{seed}_split_exact"] = all(
            np.array_equal(left[name], right[name])
            for name in ("fit_indices", "selection_indices", "oof_indices")
        )
        paired_checks[f"seed_{seed}_visited_schedule_exact"] = all(
            np.array_equal(left[name], right[name])
            for name in ("state_index", "round", "group", "role")
        )
        round1 = left["round"] == 1
        errors["paired_round1_action"] = max(errors["paired_round1_action"], max_error(left["action"][round1], right["action"][round1]))
        errors["paired_round1_cost"] = max(errors["paired_round1_cost"], max_error(left["cost"][round1], right["cost"][round1]))
        paired_checks[f"seed_{seed}_round0_selection_exact"] = np.array_equal(
            left["selection_round_action"][0], right["selection_round_action"][0]
        ) and np.array_equal(left["selection_round_cost"][0], right["selection_round_cost"][0])
        paired_checks[f"seed_{seed}_round0_oof_exact"] = np.array_equal(
            left["round0_oof_action"], right["round0_oof_action"]
        ) and np.array_equal(left["round0_oof_cost"], right["round0_oof_cost"])

    pooled_gain: dict[str, np.ndarray] = {}
    seed_gain: dict[tuple[str, int], np.ndarray] = {}
    by_speed: dict[tuple[str, int], dict[str, float]] = {}
    for arm in (arm0, arm1):
        pooled = []
        for seed in seeds:
            saved = saved_by_key[(arm, seed)]
            gain = saved["round0_oof_cost"].astype(np.float64) - saved["selected_oof_cost"].astype(np.float64)
            seed_gain[(arm, seed)] = gain
            by_speed[(arm, seed)] = by_speed_mean_gain(data, saved["oof_indices"], gain)
            pooled.append(gain)
        pooled_gain[arm] = np.concatenate(pooled)
    gate = config["decision_gate"]
    speed_counts = {
        str(speed): int(sum(
            by_speed[(arm1, seed)][str(speed)] > by_speed[(arm0, seed)][str(speed)]
            for seed in seeds
        ))
        for speed in (85, 100)
    }
    decision_checks = {
        "gamma1_oof_mean_advantage_seed_count": int(sum(
            seed_gain[(arm1, seed)].mean() > seed_gain[(arm0, seed)].mean()
            for seed in seeds
        )) >= int(gate["gamma1_oof_mean_advantage_seed_count_minimum"]),
        "gamma1_pooled_oof_mean_no_worse": pooled_gain[arm1].mean() >= pooled_gain[arm0].mean(),
        "gamma1_85kmh_mean_advantage_seed_count": speed_counts["85"] >= int(gate["gamma1_85kmh_mean_advantage_seed_count_minimum"]),
        "gamma1_100kmh_mean_advantage_seed_count": speed_counts["100"] >= int(gate["gamma1_100kmh_mean_advantage_seed_count_minimum"]),
        "gamma1_pooled_oof_p05_within_tolerance": np.quantile(pooled_gain[arm1], 0.05)
        >= np.quantile(pooled_gain[arm0], 0.05) - float(gate["gamma1_pooled_oof_p05_tolerance"]),
    }
    mean_checks = all(value for name, value in decision_checks.items() if "p05" not in name)
    if all(decision_checks.values()):
        decision = "ADOPT_GAMMA1_FOR_NEXT_SAME_CONTRACT_OAC_EXPANSION"
    elif mean_checks:
        decision = "GAMMA1_MEAN_PASS_TAIL_MIXED_TEST_GAMMA0P5"
    else:
        decision = "GAMMA1_CONTINUOUS_OAC_FAIL_RETAIN_GAMMA0"
    recomputed = {
        "decision": decision,
        "decision_checks": decision_checks,
        "highspeed_gamma1_advantage_seed_count": speed_counts,
        "paired_oof": {
            arm0: distribution(pooled_gain[arm0]),
            arm1: distribution(pooled_gain[arm1]),
            "gamma1_minus_gamma0_gain": distribution(pooled_gain[arm1] - pooled_gain[arm0]),
        },
    }
    checks["all_run_checks"] = all(report["all_checks_pass"] for report in reports)
    checks["all_pairing_checks"] = all(paired_checks.values())
    checks["decision_reconstructed"] = decision == summary["decision"] == manifest["decision"]
    checks["decision_checks_reconstructed"] = decision_checks == summary["decision_checks"]
    checks["highspeed_counts_reconstructed"] = speed_counts == summary["highspeed_gamma1_advantage_seed_count"]
    checks["paired_pooled_metrics_reconstructed"] = all(
        abs(float(recomputed["paired_oof"][arm][stat]) - float(summary["paired_oof"][arm][stat])) <= 1e-10
        for arm in (arm0, arm1, "gamma1_minus_gamma0_gain")
        for stat in ("mean", "median", "p05", "p10", "p90", "p95", "min", "max")
    )
    checks["replay_errors_zero"] = all(value <= 1e-6 for value in errors.values())
    all_checks_pass = all(checks.values())
    validation = {
        "qualification": "QUERY_OAC_AGGREGATION_AB_INDEPENDENT_PASS" if all_checks_pass else "QUERY_OAC_AGGREGATION_AB_INDEPENDENT_FAIL",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "pairing_checks": paired_checks,
        "maximum_absolute_errors": errors,
        "run_reports": reports,
        "recomputed": recomputed,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    validation_path = output / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True, default=json_default) + "\n")
    print(json.dumps({
        "qualification": validation["qualification"],
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "failed_runs": [f"{item['arm']}_seed{item['seed']}" for item in reports if not item["all_checks_pass"]],
        "maximum_absolute_errors": errors,
        "decision": decision,
    }, indent=2, default=json_default))
    if not all_checks_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
