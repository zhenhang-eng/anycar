#!/usr/bin/env python3
"""Independently replay recenter65 versus coarse-to-fine77 Query AC."""

from __future__ import annotations

import argparse
import copy
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
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from query_batched_direct_cost import batched_direct_cost  # noqa: E402
from run_query_continuous_ac_coarse_to_fine77_ab import response_bank77  # noqa: E402
from run_query_forward_response_landscape_pilot import basis_bank  # noqa: E402
from run_query_single_center_oac20to1 import actor_from_payload, actor_predict, load_inputs, sha256  # noqa: E402
from run_query_target_coverage_mixed_init_oac import warm_relative_metrics  # noqa: E402
from validate_query_continuous_ac_candidate_bank_ab import (  # noqa: E402
    array_error, validate_adapter, validate_arm,
)


DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_continuous_ac_coarse_to_fine77_ab_20260904_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_treatment(
    record: dict,
    config: dict,
    data: dict[str, np.ndarray],
    controller: TorchMPPIController,
    selection: np.ndarray,
    device: torch.device,
) -> dict:
    arrays_path, checkpoint_path = Path(record["arrays"]), Path(record["checkpoint"])
    if sha256(arrays_path) != record["arrays_sha256"] or sha256(checkpoint_path) != record["checkpoint_sha256"]:
        raise AssertionError("treatment result hash mismatch")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor = actor_from_payload(payload, "selected_actor_state_dict", device)
    inputs = load_inputs(data, payload["actor_normalization"])
    selected_action = actor_predict(actor, inputs, selection, device)
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    selected_cost = batched_direct_cost(controller, data, selection, selected_action, weights)
    errors = {}
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if arrays["actor_batch_schedule"].shape != (40, 16, 64):
            raise AssertionError("Actor batch schedule shape changed")
        groups = np.unique(arrays["online_group"])
        if len(groups) != 800 or len(arrays["online_cost"]) != 800 * 77:
            raise AssertionError("treatment group/budget contract changed")
        if not np.array_equal(np.unique(arrays["online_round"]), np.arange(1, 41)):
            raise AssertionError("treatment round coverage changed")
        selected_round = int(record["selected_round"])
        errors["selected_action_reload"] = array_error(
            selected_action, arrays["selection_round_action"][selected_round]
        )
        errors["selected_cost_reload"] = array_error(
            selected_cost, arrays["selection_round_cost"][selected_round]
        )
        radii = np.linspace(
            float(config["pilot"]["coarse_radius_start"]),
            float(config["pilot"]["coarse_radius_end"]),
            int(config["pilot"]["probe_radius_sigma_anneal_rounds"]),
        ).astype(np.float32)[:40]
        bases = basis_bank()
        sigma = np.asarray(config["pilot"]["noise_sigma"], np.float32).reshape(1, 2)
        expected_roles = np.asarray(
            ["actor"] + ["coarse_probe"] * 32 + ["coarse_response"] * 6
            + ["fine_probe"] * 32 + ["fine_response"] * 6
        )
        maxima = {"action": 0.0, "cost": 0.0, "raw": 0.0, "clipped": 0.0, "role": 0.0}
        for group in groups:
            mask = arrays["online_group"] == group
            if int(mask.sum()) != 77:
                raise AssertionError("treatment group width changed")
            rows = arrays["online_state_index"][mask]
            rounds = arrays["online_round"][mask]
            if not np.all(rows == rows[0]) or not np.all(rounds == rounds[0]):
                raise AssertionError("treatment group mixes states or rounds")
            row, round_index = int(rows[0]), int(rounds[0])
            center = np.asarray(arrays["online_action"][mask][0], np.float32)
            expected = response_bank77(
                controller, data, row, center, float(radii[round_index - 1]),
                bases[(round_index - 1) % len(bases)], sigma, weights, config,
            )
            maxima["action"] = max(maxima["action"], array_error(arrays["online_action"][mask], expected[0]))
            maxima["cost"] = max(maxima["cost"], array_error(arrays["online_cost"][mask], expected[1]))
            maxima["raw"] = max(maxima["raw"], array_error(arrays["online_raw_action"][mask], expected[2]))
            maxima["clipped"] = max(maxima["clipped"], array_error(arrays["online_clipped"][mask], expected[3]))
            maxima["role"] = max(maxima["role"], array_error(arrays["online_role"][mask], expected_roles))
            if int(group) == 0 or (int(group) + 1) % 200 == 0:
                print(f"validate coarse_to_fine77 group {int(group) + 1}/800", flush=True)
        report = warm_relative_metrics(
            arrays["selection_round_cost"][selected_round], data["warm_cost"][selection],
            data["speed_kph"][selection], data["variant_index"][selection],
        )
        errors["selected_metric_actor_mean"] = abs(
            float(report["actor_cost"]["mean"]) - float(record["selected"]["inner"]["actor_cost"]["mean"])
        )
        errors["selected_metric_aggregate"] = abs(
            float(report["aggregate_improvement"]) - float(record["selected"]["inner"]["aggregate_improvement"])
        )
        actor_schedule = np.asarray(arrays["actor_batch_schedule"])
        visited = np.asarray(arrays["online_state_index"]).reshape(800, 77)[:, 0]
        online_round = np.asarray(arrays["online_round"]).reshape(800, 77)[:, 0]
    errors.update({f"online_{name}": value for name, value in maxima.items()})
    return {
        "arm": "coarse_to_fine77", "errors": errors,
        "actor_batch_schedule": actor_schedule, "visited_rows": visited,
        "online_round": online_round,
        "all_checks_pass": all(value <= 1e-6 for value in errors.values()),
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
    if sha256(summary_path) != manifest["summary_sha256"]:
        raise AssertionError("summary hash mismatch")
    if any((summary["outer_fold_evaluated"], summary["formal_validation_or_test_consumed"],
            summary["dbm_fields_or_labels_consumed"], summary["query_analytic_gradient_consumed"])):
        raise AssertionError("sealed boundary violated")
    source_root = Path(config["sources"]["actor_oac_root"])
    source_summary_path, source_validation_path = source_root / "summary.json", source_root / "validation.json"
    if sha256(source_summary_path) != manifest["source_actor_summary_sha256"]:
        raise AssertionError("source summary hash mismatch")
    if sha256(source_validation_path) != manifest["source_actor_validation_sha256"]:
        raise AssertionError("source validation hash mismatch")
    if json.loads(source_validation_path.read_text())["qualification"] != config["sources"]["actor_oac_qualification"]:
        raise AssertionError("source qualification changed")
    source_checkpoint = Path(summary["source_checkpoint"])
    if sha256(source_checkpoint) != manifest["source_checkpoint_sha256"]:
        raise AssertionError("source checkpoint hash mismatch")
    adapter_errors = validate_adapter(summary, source_checkpoint)

    loader = {"outputs": {"absolute_replay": config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader)
    if replay_manifest["replay_sha256"] != manifest["source_replay_sha256"]:
        raise AssertionError("source Replay hash mismatch")
    selection = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    outer = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
    if (len(selection), len(outer)) != (120, 120) or set(data["episode_id"][selection]) & set(data["episode_id"][outer]):
        raise AssertionError("split contract failed")
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query), TorchMPPIParams(**collection_manifest["collection"]["mppi"]), device=str(device)
    )
    control_config = copy.deepcopy(config)
    control_config["pilot"]["probe_radius_sigma_start"] = float(config["pilot"]["control_radius_start"])
    control_config["pilot"]["probe_radius_sigma_end"] = float(config["pilot"]["control_radius_end"])
    reports = {
        "control_recenter65": validate_arm(
            "control_recenter65", summary["records"]["control_recenter65"], 65,
            control_config, data, controller, selection, device,
        ),
        "coarse_to_fine77": validate_treatment(
            summary["records"]["coarse_to_fine77"], config, data, controller, selection, device,
        ),
    }
    control, treatment = reports["control_recenter65"], reports["coarse_to_fine77"]
    paired_errors = {
        "actor_batch_schedule": array_error(control["actor_batch_schedule"], treatment["actor_batch_schedule"]),
        "visited_rows": array_error(control["visited_rows"], treatment["visited_rows"]),
        "online_round": array_error(control["online_round"], treatment["online_round"]),
    }
    with np.load(summary["records"]["control_recenter65"]["arrays"], allow_pickle=False) as a, np.load(
            summary["records"]["coarse_to_fine77"]["arrays"], allow_pickle=False) as b:
        paired_errors["round0_action"] = array_error(a["selection_round_action"][0], b["selection_round_action"][0])
        paired_errors["round0_cost"] = array_error(a["selection_round_cost"][0], b["selection_round_cost"][0])
    control_record, treatment_record = summary["records"]["control_recenter65"], summary["records"]["coarse_to_fine77"]
    control_mean = float(control_record["selected"]["inner"]["actor_cost"]["mean"])
    treatment_mean = float(treatment_record["selected"]["inner"]["actor_cost"]["mean"])
    control_aggregate = float(control_record["selected"]["inner"]["aggregate_improvement"])
    treatment_aggregate = float(treatment_record["selected"]["inner"]["aggregate_improvement"])
    expected_decision = (
        "PROMOTE_COARSE_TO_FINE77_TO_THREE_SEEDS"
        if treatment_mean < control_mean and treatment_aggregate > control_aggregate
        else "RETAIN_RECENTER65"
    )
    checks = {
        "artifact_and_source_hashes": True,
        "source_longrun_independently_qualified": True,
        "source_adapters_exact": all(value == 0.0 for value in adapter_errors.values()),
        "paired_schedule_and_round0_exact": all(value == 0.0 for value in paired_errors.values()),
        "all_113600_online_candidates_replayed": all(report["all_checks_pass"] for report in reports.values()),
        "coarse_winner_to_fine_response_chain_reconstructed": treatment["all_checks_pass"],
        "selected_actors_and_inner_costs_reloaded": all(report["all_checks_pass"] for report in reports.values()),
        "decision_recomputed": summary["decision"] == expected_decision,
        "split_and_sealed_boundary": True,
    }
    passed = all(checks.values())
    serializable_reports = {
        arm: {key: value for key, value in report.items()
              if key not in ("actor_batch_schedule", "visited_rows", "online_round")}
        for arm, report in reports.items()
    }
    result = {
        "qualification": "QUERY_CONTINUOUS_AC_COARSE_TO_FINE77_AB_INDEPENDENT_PASS" if passed else "QUERY_CONTINUOUS_AC_COARSE_TO_FINE77_AB_INDEPENDENT_FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "adapter_errors": adapter_errors, "paired_errors": paired_errors,
        "arm_reports": serializable_reports, "recomputed_decision": expected_decision,
        "recomputed_comparison": {
            "control_selected_mean": control_mean, "treatment_selected_mean": treatment_mean,
            "treatment_mean_reduction": control_mean - treatment_mean,
            "control_warm_aggregate": control_aggregate, "treatment_warm_aggregate": treatment_aggregate,
            "treatment_aggregate_delta": treatment_aggregate - control_aggregate,
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
