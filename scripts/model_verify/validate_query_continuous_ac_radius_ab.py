#!/usr/bin/env python3
"""Independently replay the paired current-Actor Query AC radius A/B."""

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
from run_query_single_center_oac20to1 import sha256  # noqa: E402
from validate_query_continuous_ac_candidate_bank_ab import (  # noqa: E402
    array_error, validate_adapter, validate_arm,
)


DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_continuous_ac_radius_ab_20260904_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


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
    if (len(selection), len(outer)) != (120, 120):
        raise AssertionError("split size changed")
    if set(data["episode_id"][selection]) & set(data["episode_id"][outer]):
        raise AssertionError("inner/outer episode leakage")
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query), TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    reports = {}
    for arm, radius in config["pilot"]["radius_arms"].items():
        arm_config = copy.deepcopy(config)
        arm_config["pilot"]["probe_radius_sigma_start"] = float(radius["start"])
        arm_config["pilot"]["probe_radius_sigma_end"] = float(radius["end"])
        reports[arm] = validate_arm(
            arm, summary["records"][arm], 65, arm_config, data, controller, selection, device
        )
    control, broad = reports["control_020_to_005"], reports["broad_100_to_010"]
    paired_errors = {
        "actor_batch_schedule": array_error(control["actor_batch_schedule"], broad["actor_batch_schedule"]),
        "visited_rows": array_error(control["visited_rows"], broad["visited_rows"]),
        "online_round": array_error(control["online_round"], broad["online_round"]),
    }
    with np.load(summary["records"]["control_020_to_005"]["arrays"], allow_pickle=False) as a, np.load(
            summary["records"]["broad_100_to_010"]["arrays"], allow_pickle=False) as b:
        paired_errors["round0_action"] = array_error(a["selection_round_action"][0], b["selection_round_action"][0])
        paired_errors["round0_cost"] = array_error(a["selection_round_cost"][0], b["selection_round_cost"][0])
    control_record = summary["records"]["control_020_to_005"]
    broad_record = summary["records"]["broad_100_to_010"]
    control_mean = float(control_record["selected"]["inner"]["actor_cost"]["mean"])
    broad_mean = float(broad_record["selected"]["inner"]["actor_cost"]["mean"])
    control_aggregate = float(control_record["selected"]["inner"]["aggregate_improvement"])
    broad_aggregate = float(broad_record["selected"]["inner"]["aggregate_improvement"])
    expected_decision = (
        "PROMOTE_BROAD_RADIUS_TO_THREE_SEED_CONFIRMATION"
        if broad_mean < control_mean and broad_aggregate > control_aggregate
        else "RETAIN_CURRENT_RADIUS_SCHEDULE"
    )
    checks = {
        "artifact_and_source_hashes": True,
        "source_longrun_independently_qualified": True,
        "source_adapters_exact": all(value == 0.0 for value in adapter_errors.values()),
        "paired_schedule_and_round0_exact": all(value == 0.0 for value in paired_errors.values()),
        "all_104000_online_candidates_replayed": all(report["all_checks_pass"] for report in reports.values()),
        "selected_actors_and_inner_costs_reloaded": all(report["all_checks_pass"] for report in reports.values()),
        "decision_recomputed": summary["decision"] == expected_decision,
        "split_and_sealed_boundary": True,
    }
    passed = all(checks.values())
    serializable_reports = {
        arm: {
            key: value for key, value in report.items()
            if key not in ("actor_batch_schedule", "visited_rows", "online_round")
        }
        for arm, report in reports.items()
    }
    result = {
        "qualification": "QUERY_CONTINUOUS_AC_RADIUS_AB_INDEPENDENT_PASS" if passed else "QUERY_CONTINUOUS_AC_RADIUS_AB_INDEPENDENT_FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "adapter_errors": adapter_errors, "paired_errors": paired_errors,
        "arm_reports": serializable_reports, "recomputed_decision": expected_decision,
        "recomputed_comparison": {
            "control_selected_mean": control_mean, "broad_selected_mean": broad_mean,
            "broad_mean_reduction": control_mean - broad_mean,
            "control_warm_aggregate": control_aggregate, "broad_warm_aggregate": broad_aggregate,
            "broad_aggregate_delta": broad_aggregate - control_aggregate,
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
