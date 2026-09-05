#!/usr/bin/env python3
"""Independently validate equal-cell 160-round AC, replaying its post-pilot suffix."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package))

import pretrain_query_single_center_actor_twin_critic as pretrain  # noqa: E402
import run_query_target_coverage_mixed_init_oac as base  # noqa: E402
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from query_batched_direct_cost import batched_direct_cost  # noqa: E402
from run_query_continuous_ac_coarse_to_fine77_ab import response_bank77  # noqa: E402
from run_query_forward_response_landscape_pilot import basis_bank  # noqa: E402
from run_query_single_center_oac20to1 import actor_from_payload, actor_predict, load_inputs, sha256  # noqa: E402
from validate_query_continuous_ac_candidate_bank_ab import array_error, nested_error, validate_adapter  # noqa: E402
from validate_query_continuous_ac_equal_cell_ab import derive_cell_arrays  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_continuous_ac_equal_cell_longrun160_20260904_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_seed(
    record: dict, pilot_record: dict, config: dict, training: dict,
    data: dict[str, np.ndarray], fit: np.ndarray, selection: np.ndarray,
    expected_weight: np.ndarray, expected_cell_id: np.ndarray,
    controller: TorchMPPIController, device: torch.device,
) -> dict:
    arrays_path, checkpoint_path = Path(record["arrays"]), Path(record["checkpoint"])
    if sha256(arrays_path) != record["arrays_sha256"] or sha256(checkpoint_path) != record["checkpoint_sha256"]:
        raise AssertionError("longrun result hash mismatch")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor = actor_from_payload(payload, "selected_actor_state_dict", device)
    inputs = load_inputs(data, payload["actor_normalization"])
    selected_action = actor_predict(actor, inputs, selection, device)
    weights = {name: float(value) for name, value in training["cost_weights"].items()}
    selected_cost = batched_direct_cost(controller, data, selection, selected_action, weights)
    errors = {}
    prefix_errors = {}
    cell_errors = {}
    with np.load(arrays_path, allow_pickle=False) as arrays, np.load(pilot_record["arrays"], allow_pickle=False) as pilot:
        if arrays["actor_batch_schedule"].shape != (160, 16, 64):
            raise AssertionError("longrun Actor schedule shape changed")
        groups = np.unique(arrays["online_group"])
        if len(groups) != 3200 or len(arrays["online_cost"]) != 3200 * 77:
            raise AssertionError("longrun candidate budget changed")
        if not np.array_equal(np.unique(arrays["online_round"]), np.arange(1, 161)):
            raise AssertionError("longrun round coverage changed")
        selected_round = int(record["selected_round"])
        errors["selected_action_reload"] = array_error(selected_action, arrays["selection_round_action"][selected_round])
        errors["selected_cost_reload"] = array_error(selected_cost, arrays["selection_round_cost"][selected_round])
        candidate_prefix = int(config["prefix_rounds"]) * 20 * 77
        prefix_pairs = {
            "actor_batch_schedule": (pilot["actor_batch_schedule"], arrays["actor_batch_schedule"][:40]),
            "selection_round_action": (pilot["selection_round_action"], arrays["selection_round_action"][:41]),
            "selection_round_cost": (pilot["selection_round_cost"], arrays["selection_round_cost"][:41]),
            "online_state_index": (pilot["online_state_index"], arrays["online_state_index"][:candidate_prefix]),
            "online_action": (pilot["online_action"], arrays["online_action"][:candidate_prefix]),
            "online_cost": (pilot["online_cost"], arrays["online_cost"][:candidate_prefix]),
            "online_raw_action": (pilot["online_raw_action"], arrays["online_raw_action"][:candidate_prefix]),
            "online_clipped": (pilot["online_clipped"], arrays["online_clipped"][:candidate_prefix]),
            "online_round": (pilot["online_round"], arrays["online_round"][:candidate_prefix]),
            "online_group": (pilot["online_group"], arrays["online_group"][:candidate_prefix]),
            "online_role": (pilot["online_role"], arrays["online_role"][:candidate_prefix]),
        }
        prefix_errors = {name: array_error(a, b) for name, (a, b) in prefix_pairs.items()}
        cell_errors = {
            "fit_row_weight": array_error(arrays["actor_fit_row_cell_weight"], expected_weight),
            "fit_cell_id": array_error(arrays["fit_cell_id"], expected_cell_id[fit]),
            "selection_cell_id": array_error(arrays["selection_cell_id"], expected_cell_id[selection]),
        }
        anneal = int(training["pilot"]["probe_radius_sigma_anneal_rounds"])
        radii = np.concatenate((
            np.linspace(float(training["pilot"]["coarse_radius_start"]), float(training["pilot"]["coarse_radius_end"]), anneal),
            np.full(160 - anneal, float(training["pilot"]["coarse_radius_end"])),
        )).astype(np.float32)
        bases = basis_bank()
        sigma = np.asarray(training["pilot"]["noise_sigma"], np.float32).reshape(1, 2)
        expected_roles = np.asarray(
            ["actor"] + ["coarse_probe"] * 32 + ["coarse_response"] * 6
            + ["fine_probe"] * 32 + ["fine_response"] * 6
        )
        maxima = {"action": 0.0, "cost": 0.0, "raw": 0.0, "clipped": 0.0, "role": 0.0}
        suffix_groups = groups[int(config["prefix_rounds"]) * 20:]
        for offset, group in enumerate(suffix_groups, start=1):
            mask = arrays["online_group"] == group
            if int(mask.sum()) != 77:
                raise AssertionError("longrun response group width changed")
            rows = arrays["online_state_index"][mask]
            rounds = arrays["online_round"][mask]
            if not np.all(rows == rows[0]) or not np.all(rounds == rounds[0]):
                raise AssertionError("longrun response group mixes states/rounds")
            row, round_index = int(rows[0]), int(rounds[0])
            center = np.asarray(arrays["online_action"][mask][0], np.float32)
            expected = response_bank77(
                controller, data, row, center, float(radii[round_index - 1]),
                bases[(round_index - 1) % len(bases)], sigma, weights, training,
            )
            maxima["action"] = max(maxima["action"], array_error(arrays["online_action"][mask], expected[0]))
            maxima["cost"] = max(maxima["cost"], array_error(arrays["online_cost"][mask], expected[1]))
            maxima["raw"] = max(maxima["raw"], array_error(arrays["online_raw_action"][mask], expected[2]))
            maxima["clipped"] = max(maxima["clipped"], array_error(arrays["online_clipped"][mask], expected[3]))
            maxima["role"] = max(maxima["role"], array_error(arrays["online_role"][mask], expected_roles))
            if offset == 1 or offset % 400 == 0:
                print(f"validate suffix group {offset}/{len(suffix_groups)}", flush=True)
        report = base.warm_relative_metrics(
            arrays["selection_round_cost"][selected_round], data["warm_cost"][selection],
            data["speed_kph"][selection], data["variant_index"][selection],
        )
        errors["selected_metric_actor_mean"] = abs(float(report["actor_cost"]["mean"]) - float(record["selected"]["inner"]["actor_cost"]["mean"]))
        errors["selected_metric_aggregate"] = abs(float(report["aggregate_improvement"]) - float(record["selected"]["inner"]["aggregate_improvement"]))
    errors.update({f"suffix_online_{name}": value for name, value in maxima.items()})
    return {
        "errors": errors, "prefix_errors": prefix_errors, "cell_errors": cell_errors,
        "all_checks_pass": all(value <= 1e-6 for value in errors.values())
        and all(value == 0.0 for value in prefix_errors.values())
        and all(value <= 1e-7 for value in cell_errors.values()),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    if sha256(config_path) != manifest["config_sha256"] or sha256(Path(manifest["script"])) != manifest["script_sha256"]:
        raise AssertionError("config or runner hash mismatch")
    if sha256(Path(manifest["equal_cell_runner"])) != manifest["equal_cell_runner_sha256"]:
        raise AssertionError("equal-cell implementation hash changed")
    if sha256(summary_path) != manifest["summary_sha256"]:
        raise AssertionError("summary hash mismatch")
    if any((summary["outer_fold_evaluated"], summary["formal_validation_or_test_consumed"],
            summary["dbm_fields_or_labels_consumed"], summary["query_analytic_gradient_consumed"])):
        raise AssertionError("sealed boundary violated")

    pilot_root = Path(config["sources"]["pilot40_root"])
    pilot_summary_path, pilot_validation_path = pilot_root / "summary.json", pilot_root / "validation.json"
    if not (sha256(pilot_summary_path) == manifest["pilot40_summary_sha256"] == config["sources"]["pilot40_summary_sha256"]):
        raise AssertionError("pilot summary hash changed")
    if not (sha256(pilot_validation_path) == manifest["pilot40_validation_sha256"] == config["sources"]["pilot40_validation_sha256"]):
        raise AssertionError("pilot validation hash changed")
    if json.loads(pilot_validation_path.read_text())["qualification"] != config["sources"]["pilot40_qualification"]:
        raise AssertionError("pilot qualification changed")
    pilot = json.loads(pilot_summary_path.read_text())
    training = summary["inherited_training_contract"]
    replay_root = Path(training["sources"]["absolute_replay"])
    data, replay_manifest, collection_manifest = pretrain.load_data({"outputs": {"absolute_replay": str(replay_root)}})
    if replay_manifest["replay_sha256"] != manifest["source_replay_sha256"] or replay_manifest["query_checkpoint_sha256"] != manifest["query_checkpoint_sha256"]:
        raise AssertionError("Replay or Query checkpoint hash changed")
    fit = np.flatnonzero(np.isin(data["fold_id"], training["split_contract"]["fit_folds"]))
    selection = np.flatnonzero(data["fold_id"] == training["split_contract"]["inner_selection_fold"])
    outer = np.flatnonzero(data["fold_id"] == training["split_contract"]["outer_fold"])
    if (len(fit), len(selection), len(outer)) != (432, 120, 120):
        raise AssertionError("split sizes changed")
    episode_sets = [set(data["episode_id"][rows].tolist()) for rows in (fit, selection, outer)]
    if any(episode_sets[a] & episode_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("episode leakage")
    expected_weight, expected_cell_id, cell_report = derive_cell_arrays(data, fit)
    cell_report_error = nested_error(summary["cell_balance_contract"], cell_report)
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query), TorchMPPIParams(**collection_manifest["collection"]["mppi"]), device=str(device)
    )

    adapter_errors, reports = {}, {}
    for seed_value in training["pilot"]["seeds"]:
        key = str(seed_value)
        adapter = summary["adapters"][key]
        adapter_view = {
            "source_actor_adapter": adapter["actor"], "source_actor_adapter_sha256": adapter["actor_sha256"],
            "source_adapter": adapter["critic"], "source_adapter_sha256": adapter["critic_sha256"],
        }
        adapter_errors[key] = validate_adapter(adapter_view, Path(adapter["source_checkpoint"]))
        reports[key] = validate_seed(
            summary["records"][key], pilot["records"]["equal_20_cell"][key], config, training,
            data, fit, selection, expected_weight, expected_cell_id, controller, device,
        )
    pooled = base.pooled([summary["records"][str(seed)] for seed in training["pilot"]["seeds"]], "selected", "inner", data)
    pooled_error = nested_error(summary["pooled"], pooled)
    pilot_pooled_error = nested_error(summary["pilot40_pooled"], pilot["pooled"]["equal_20_cell"])
    per_seed, improved = {}, 0
    for seed_value in training["pilot"]["seeds"]:
        key = str(seed_value)
        pilot_mean = float(pilot["records"]["equal_20_cell"][key]["selected"]["inner"]["actor_cost"]["mean"])
        longrun_mean = float(summary["records"][key]["selected"]["inner"]["actor_cost"]["mean"])
        lower = longrun_mean < pilot_mean
        improved += int(lower)
        per_seed[key] = {"pilot40_mean": pilot_mean, "longrun160_mean": longrun_mean,
                         "longrun_mean_reduction": pilot_mean - longrun_mean, "longrun_lower": lower}
    pilot_mean = float(pilot["pooled"]["equal_20_cell"]["actor_cost"]["mean"])
    longrun_mean = float(pooled["actor_cost"]["mean"])
    pilot_aggregate = float(pilot["pooled"]["equal_20_cell"]["aggregate_improvement"])
    longrun_aggregate = float(pooled["aggregate_improvement"])
    expected_decision = (
        "PROMOTE_EQUAL_CELL_LONGRUN160"
        if longrun_mean < pilot_mean and longrun_aggregate > pilot_aggregate and improved >= 2
        else "RETAIN_EQUAL_CELL_PILOT40"
    )
    checks = {
        "artifact_config_and_source_hashes": True,
        "pilot40_independently_qualified": True,
        "source_adapters_exact": all(value == 0.0 for report in adapter_errors.values() for value in report.values()),
        "equal_cell_weights_independently_recomputed": cell_report_error <= 1e-7 and all(value <= 1e-7 for report in reports.values() for value in report["cell_errors"].values()),
        "first_40_round_prefix_exact": all(value == 0.0 for report in reports.values() for value in report["prefix_errors"].values()),
        "all_554400_suffix_candidates_replayed": all(report["all_checks_pass"] for report in reports.values()),
        "selected_actors_and_360_inner_costs_reloaded": all(report["all_checks_pass"] for report in reports.values()),
        "pooled_metrics_recomputed": pooled_error <= 1e-12 and pilot_pooled_error <= 1e-12,
        "decision_recomputed": summary["decision"] == expected_decision,
        "split_and_sealed_boundary": True,
    }
    passed = all(checks.values())
    result = {
        "qualification": "QUERY_CONTINUOUS_AC_EQUAL_CELL_LONGRUN160_INDEPENDENT_PASS" if passed else "QUERY_CONTINUOUS_AC_EQUAL_CELL_LONGRUN160_INDEPENDENT_FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "adapter_errors": adapter_errors, "seed_reports": reports,
        "cell_report_max_abs_error": cell_report_error,
        "pooled_metric_max_abs_error": pooled_error, "pilot_pooled_max_abs_error": pilot_pooled_error,
        "recomputed_decision": expected_decision,
        "recomputed_comparison": {
            "pilot40_pooled_mean": pilot_mean, "longrun160_pooled_mean": longrun_mean,
            "longrun_pooled_mean_reduction": pilot_mean - longrun_mean,
            "pilot40_warm_aggregate": pilot_aggregate, "longrun160_warm_aggregate": longrun_aggregate,
            "longrun_aggregate_delta": longrun_aggregate - pilot_aggregate,
            "improved_seed_count": improved, "per_seed": per_seed,
        },
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "validation.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
