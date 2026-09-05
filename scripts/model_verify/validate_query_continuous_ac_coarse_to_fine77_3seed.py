#!/usr/bin/env python3
"""Independently validate the three-seed coarse-to-fine77 confirmation."""

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
import run_query_target_coverage_mixed_init_oac as base  # noqa: E402
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from run_query_single_center_oac20to1 import sha256  # noqa: E402
from validate_query_continuous_ac_candidate_bank_ab import array_error, nested_error, validate_adapter, validate_arm  # noqa: E402
from validate_query_continuous_ac_coarse_to_fine77_ab import validate_treatment  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_continuous_ac_coarse_to_fine77_3seed_20260904_v1"


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
    if sha256(source_summary_path) != manifest["source_actor_summary_sha256"] or sha256(source_validation_path) != manifest["source_actor_validation_sha256"]:
        raise AssertionError("longrun source hash mismatch")
    if json.loads(source_validation_path.read_text())["qualification"] != config["sources"]["actor_oac_qualification"]:
        raise AssertionError("longrun source qualification changed")
    single_root = Path(config["sources"]["validated_single_seed"])
    single_summary_path, single_validation_path = single_root / "summary.json", single_root / "validation.json"
    if sha256(single_summary_path) != manifest["single_seed_summary_sha256"] or sha256(single_validation_path) != manifest["single_seed_validation_sha256"]:
        raise AssertionError("single-seed source hash mismatch")
    single_summary = json.loads(single_summary_path.read_text())
    if json.loads(single_validation_path.read_text())["qualification"] != config["sources"]["validated_single_seed_qualification"]:
        raise AssertionError("single-seed source qualification changed")
    for arm in config["pilot"]["candidate_arms"]:
        reused = summary["records"][arm]["0"]
        original = single_summary["records"][arm]
        if reused["arrays_sha256"] != original["arrays_sha256"] or reused["checkpoint_sha256"] != original["checkpoint_sha256"]:
            raise AssertionError(f"reused seed0 record changed for {arm}")
        if sha256(Path(reused["arrays"])) != reused["arrays_sha256"] or sha256(Path(reused["checkpoint"])) != reused["checkpoint_sha256"]:
            raise AssertionError(f"reused seed0 artifact hash mismatch for {arm}")

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
    seed_reports = {}
    adapter_errors = {}
    paired_errors = {}
    for seed in config["pilot"]["new_seeds"]:
        key = str(seed)
        adapter = summary["adapters"][key]
        adapter_view = {
            "source_actor_adapter": adapter["actor"], "source_actor_adapter_sha256": adapter["actor_sha256"],
            "source_adapter": adapter["critic"], "source_adapter_sha256": adapter["critic_sha256"],
        }
        adapter_errors[key] = validate_adapter(adapter_view, Path(adapter["source_checkpoint"]))
        reports = {
            "control_recenter65": validate_arm(
                "control_recenter65", summary["records"]["control_recenter65"][key], 65,
                control_config, data, controller, selection, device,
            ),
            "coarse_to_fine77": validate_treatment(
                summary["records"]["coarse_to_fine77"][key], config, data, controller, selection, device,
            ),
        }
        control, treatment = reports["control_recenter65"], reports["coarse_to_fine77"]
        paired_errors[key] = {
            "actor_batch_schedule": array_error(control["actor_batch_schedule"], treatment["actor_batch_schedule"]),
            "visited_rows": array_error(control["visited_rows"], treatment["visited_rows"]),
            "online_round": array_error(control["online_round"], treatment["online_round"]),
        }
        with np.load(summary["records"]["control_recenter65"][key]["arrays"], allow_pickle=False) as a, np.load(
                summary["records"]["coarse_to_fine77"][key]["arrays"], allow_pickle=False) as b:
            paired_errors[key]["round0_action"] = array_error(a["selection_round_action"][0], b["selection_round_action"][0])
            paired_errors[key]["round0_cost"] = array_error(a["selection_round_cost"][0], b["selection_round_cost"][0])
        seed_reports[key] = {
            arm: {name: value for name, value in report.items()
                  if name not in ("actor_batch_schedule", "visited_rows", "online_round")}
            for arm, report in reports.items()
        }

    pooled = {
        arm: base.pooled([summary["records"][arm][str(seed)] for seed in config["pilot"]["seeds"]], "selected", "inner", data)
        for arm in config["pilot"]["candidate_arms"]
    }
    pooled_error = nested_error(summary["pooled"], pooled)
    per_seed = {}
    improved = 0
    for seed in config["pilot"]["seeds"]:
        key = str(seed)
        control_mean = float(summary["records"]["control_recenter65"][key]["selected"]["inner"]["actor_cost"]["mean"])
        treatment_mean = float(summary["records"]["coarse_to_fine77"][key]["selected"]["inner"]["actor_cost"]["mean"])
        lower = treatment_mean < control_mean
        improved += int(lower)
        per_seed[key] = {"control_mean": control_mean, "treatment_mean": treatment_mean, "treatment_mean_reduction": control_mean - treatment_mean, "treatment_lower": lower}
    control_mean = float(pooled["control_recenter65"]["actor_cost"]["mean"])
    treatment_mean = float(pooled["coarse_to_fine77"]["actor_cost"]["mean"])
    control_aggregate = float(pooled["control_recenter65"]["aggregate_improvement"])
    treatment_aggregate = float(pooled["coarse_to_fine77"]["aggregate_improvement"])
    expected_decision = (
        "PROMOTE_COARSE_TO_FINE77"
        if treatment_mean < control_mean and treatment_aggregate > control_aggregate and improved >= 2
        else "RETAIN_RECENTER65_STOP_BANK_EXPANSION"
    )
    checks = {
        "artifact_and_source_hashes": True,
        "reused_seed0_independent_qualification_and_hashes": True,
        "new_source_adapters_exact": all(value == 0.0 for report in adapter_errors.values() for value in report.values()),
        "new_seed_paired_schedule_and_round0_exact": all(value == 0.0 for report in paired_errors.values() for value in report.values()),
        "all_227200_new_online_candidates_replayed": all(report[arm]["all_checks_pass"] for report in seed_reports.values() for arm in report),
        "new_coarse_to_fine_chains_reconstructed": all(report["coarse_to_fine77"]["all_checks_pass"] for report in seed_reports.values()),
        "pooled_metrics_recomputed": pooled_error <= 1e-12,
        "decision_recomputed": summary["decision"] == expected_decision,
        "split_and_sealed_boundary": True,
    }
    passed = all(checks.values())
    result = {
        "qualification": "QUERY_CONTINUOUS_AC_COARSE_TO_FINE77_THREE_SEED_INDEPENDENT_PASS" if passed else "QUERY_CONTINUOUS_AC_COARSE_TO_FINE77_THREE_SEED_INDEPENDENT_FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "adapter_errors": adapter_errors, "paired_errors": paired_errors, "seed_reports": seed_reports,
        "pooled_metric_max_abs_error": pooled_error, "recomputed_decision": expected_decision,
        "recomputed_comparison": {
            "control_pooled_mean": control_mean, "treatment_pooled_mean": treatment_mean,
            "treatment_pooled_mean_reduction": control_mean - treatment_mean,
            "control_warm_aggregate": control_aggregate, "treatment_warm_aggregate": treatment_aggregate,
            "treatment_aggregate_delta": treatment_aggregate - control_aggregate,
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
