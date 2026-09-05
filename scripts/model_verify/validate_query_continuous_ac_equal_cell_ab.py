#!/usr/bin/env python3
"""Independently validate the three-seed equal-cell continuous-AC A/B."""

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
from run_query_single_center_oac20to1 import sha256  # noqa: E402
from validate_query_continuous_ac_candidate_bank_ab import array_error, nested_error, validate_adapter  # noqa: E402
from validate_query_continuous_ac_coarse_to_fine77_ab import validate_treatment  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_continuous_ac_equal_cell_ab_20260904_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def derive_cell_arrays(data: dict[str, np.ndarray], fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    cells = sorted({(int(data["speed_kph"][row]), int(data["variant_index"][row])) for row in fit})
    if len(cells) != 20:
        raise AssertionError("fit no longer contains exactly 20 cells")
    mapping = {cell: index for index, cell in enumerate(cells)}
    cell_id = np.asarray([
        mapping[(int(speed), int(variant))]
        for speed, variant in zip(data["speed_kph"], data["variant_index"])
    ], dtype=np.int16)
    counts = np.bincount(cell_id[fit], minlength=20)
    if sorted(counts.tolist()) != [18] * 18 + [54] * 2:
        raise AssertionError("fit cell counts changed")
    weights = np.asarray([len(fit) / (20 * counts[index]) for index in cell_id[fit]], np.float32)
    masses = np.bincount(cell_id[fit], weights=weights, minlength=20)
    report = {
        "formula": "fit_count / (cell_count * fit_cell_count)",
        "fit_count": int(len(fit)), "cell_count": 20,
        "cells": [
            {"cell_id": index, "speed_kph": speed, "variant_index": variant,
             "fit_count": int(counts[index]), "row_weight": float(len(fit) / (20 * counts[index])),
             "total_expected_mass": float(masses[index])}
            for index, (speed, variant) in enumerate(cells)
        ],
        "fit_row_weight_mean": float(weights.mean()),
        "fit_row_weight_minimum": float(weights.min()),
        "fit_row_weight_maximum": float(weights.max()),
    }
    return weights, cell_id, report


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

    control_root = Path(config["sources"]["control_root"])
    control_summary_path, control_validation_path = control_root / "summary.json", control_root / "validation.json"
    matched_root = Path(config["sources"]["matched_step_root"])
    matched_summary_path, matched_validation_path = matched_root / "summary.json", matched_root / "validation.json"
    source_checks = {
        "control_summary": sha256(control_summary_path) == manifest["control_summary_sha256"] == config["sources"]["control_summary_sha256"],
        "control_validation": sha256(control_validation_path) == manifest["control_validation_sha256"] == config["sources"]["control_validation_sha256"],
        "matched_summary": sha256(matched_summary_path) == manifest["matched_step_summary_sha256"] == config["sources"]["matched_step_summary_sha256"],
        "matched_validation": sha256(matched_validation_path) == manifest["matched_step_validation_sha256"] == config["sources"]["matched_step_validation_sha256"],
        "control_qualification": json.loads(control_validation_path.read_text())["qualification"] == config["sources"]["control_qualification"],
        "matched_qualification": json.loads(matched_validation_path.read_text())["qualification"] == config["sources"]["matched_step_qualification"],
    }
    if not all(source_checks.values()):
        raise AssertionError(f"source lock failed: {source_checks}")
    source_summary = json.loads(control_summary_path.read_text())
    for seed_value in config["pilot"]["seeds"]:
        key = str(seed_value)
        reused = summary["records"]["control_uniform_row"][key]
        original = source_summary["records"]["coarse_to_fine77"][key]
        if reused["arrays_sha256"] != original["arrays_sha256"] or reused["checkpoint_sha256"] != original["checkpoint_sha256"]:
            raise AssertionError(f"reused control record changed for seed {key}")
        if sha256(Path(reused["arrays"])) != reused["arrays_sha256"] or sha256(Path(reused["checkpoint"])) != reused["checkpoint_sha256"]:
            raise AssertionError(f"reused control artifact hash mismatch for seed {key}")

    loader = {"outputs": {"absolute_replay": config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader)
    if replay_manifest["replay_sha256"] != manifest["source_replay_sha256"] or replay_manifest["query_checkpoint_sha256"] != manifest["query_checkpoint_sha256"]:
        raise AssertionError("Replay or Query checkpoint hash changed")
    fit = np.flatnonzero(np.isin(data["fold_id"], config["split_contract"]["fit_folds"]))
    selection = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    outer = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
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
    adapter_errors, paired_errors, cell_errors, seed_reports = {}, {}, {}, {}
    for seed_value in config["pilot"]["seeds"]:
        key = str(seed_value)
        adapter = summary["adapters"][key]
        if sha256(Path(adapter["source_checkpoint"])) != adapter["source_checkpoint_sha256"]:
            raise AssertionError(f"source checkpoint changed for seed {key}")
        adapter_view = {
            "source_actor_adapter": adapter["actor"], "source_actor_adapter_sha256": adapter["actor_sha256"],
            "source_adapter": adapter["critic"], "source_adapter_sha256": adapter["critic_sha256"],
        }
        adapter_errors[key] = validate_adapter(adapter_view, Path(adapter["source_checkpoint"]))
        treatment = summary["records"]["equal_20_cell"][key]
        report = validate_treatment(treatment, config, data, controller, selection, device)
        with np.load(summary["records"]["control_uniform_row"][key]["arrays"], allow_pickle=False) as left, np.load(
            treatment["arrays"], allow_pickle=False
        ) as right:
            control_visited = left["online_state_index"].reshape(800, 77)[:, 0]
            treatment_visited = right["online_state_index"].reshape(800, 77)[:, 0]
            control_round = left["online_round"].reshape(800, 77)[:, 0]
            treatment_round = right["online_round"].reshape(800, 77)[:, 0]
            paired_errors[key] = {
                "actor_batch_schedule": array_error(left["actor_batch_schedule"], right["actor_batch_schedule"]),
                "visited_rows": array_error(control_visited, treatment_visited),
                "online_round": array_error(control_round, treatment_round),
                "round0_action": array_error(left["selection_round_action"][0], right["selection_round_action"][0]),
                "round0_cost": array_error(left["selection_round_cost"][0], right["selection_round_cost"][0]),
            }
            cell_errors[key] = {
                "fit_row_weight": array_error(right["actor_fit_row_cell_weight"], expected_weight),
                "fit_cell_id": array_error(right["fit_cell_id"], expected_cell_id[fit]),
                "selection_cell_id": array_error(right["selection_cell_id"], expected_cell_id[selection]),
            }
        payload = torch.load(treatment["checkpoint"], map_location="cpu", weights_only=False)
        if payload["source_actor_checkpoint_sha256"] != adapter["actor_sha256"] or payload["source_critic_checkpoint_sha256"] != adapter["critic_sha256"]:
            raise AssertionError(f"treatment adapter provenance changed for seed {key}")
        seed_reports[key] = {name: value for name, value in report.items()
                             if name not in ("actor_batch_schedule", "visited_rows", "online_round")}

    pooled = {
        arm: base.pooled([summary["records"][arm][str(seed)] for seed in config["pilot"]["seeds"]], "selected", "inner", data)
        for arm in ("control_uniform_row", "equal_20_cell")
    }
    pooled_error = nested_error(summary["pooled"], pooled)
    per_seed, improved = {}, 0
    for seed_value in config["pilot"]["seeds"]:
        key = str(seed_value)
        control_mean = float(summary["records"]["control_uniform_row"][key]["selected"]["inner"]["actor_cost"]["mean"])
        treatment_mean = float(summary["records"]["equal_20_cell"][key]["selected"]["inner"]["actor_cost"]["mean"])
        lower = treatment_mean < control_mean
        improved += int(lower)
        per_seed[key] = {"control_mean": control_mean, "treatment_mean": treatment_mean,
                         "treatment_mean_reduction": control_mean - treatment_mean, "treatment_lower": lower}
    control_mean = float(pooled["control_uniform_row"]["actor_cost"]["mean"])
    treatment_mean = float(pooled["equal_20_cell"]["actor_cost"]["mean"])
    control_aggregate = float(pooled["control_uniform_row"]["aggregate_improvement"])
    treatment_aggregate = float(pooled["equal_20_cell"]["aggregate_improvement"])
    expected_decision = (
        "PROMOTE_EQUAL_20_CELL_ACTOR_OBJECTIVE"
        if treatment_mean < control_mean and treatment_aggregate > control_aggregate and improved >= 2
        else "RETAIN_UNIFORM_ROW_ACTOR_OBJECTIVE"
    )
    checks = {
        "artifact_config_and_source_hashes": True,
        "source_qualifications_locked": True,
        "reused_control_artifacts_unchanged": True,
        "source_adapters_exact": all(value == 0.0 for report in adapter_errors.values() for value in report.values()),
        "equal_cell_weights_independently_recomputed": cell_report_error <= 1e-7 and all(value <= 1e-7 for report in cell_errors.values() for value in report.values()),
        "paired_schedule_context_and_round0_exact": all(value == 0.0 for report in paired_errors.values() for value in report.values()),
        "all_184800_treatment_candidates_replayed": all(report["all_checks_pass"] for report in seed_reports.values()),
        "selected_actors_and_360_inner_costs_reloaded": all(report["all_checks_pass"] for report in seed_reports.values()),
        "pooled_metrics_recomputed": pooled_error <= 1e-12,
        "decision_recomputed": summary["decision"] == expected_decision,
        "split_and_sealed_boundary": True,
    }
    passed = all(checks.values())
    result = {
        "qualification": "QUERY_CONTINUOUS_AC_EQUAL_CELL_THREE_SEED_INDEPENDENT_PASS" if passed else "QUERY_CONTINUOUS_AC_EQUAL_CELL_THREE_SEED_INDEPENDENT_FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "source_checks": source_checks, "adapter_errors": adapter_errors, "paired_errors": paired_errors,
        "cell_errors": cell_errors, "cell_report_max_abs_error": cell_report_error,
        "seed_reports": seed_reports, "pooled_metric_max_abs_error": pooled_error,
        "recomputed_decision": expected_decision,
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
