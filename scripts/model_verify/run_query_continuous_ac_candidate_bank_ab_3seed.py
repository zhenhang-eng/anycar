#!/usr/bin/env python3
"""Expand the validated 39/65 continuous Query AC comparison to three seeds."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
from run_query_continuous_ac_candidate_bank_ab import (  # noqa: E402
    bank_report,
    make_actor_adapter,
    make_critic_adapter,
    response_bank39,
    response_bank65,
    rewrite_roles,
)


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_continuous_ac_candidate_bank_ab_3seed_config_20260903_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def selected_cost(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(record["arrays"], allow_pickle=False) as archive:
        rows = np.asarray(archive["selection_indices"])
        costs = np.asarray(archive["selection_round_cost"])[int(record["selected_round"])]
    return rows, costs


def pooled(records: dict[str, dict[str, Any]], arm: str,
           data: dict[str, np.ndarray]) -> dict[str, Any]:
    rows, costs = zip(*(selected_cost(records[str(seed)][arm]) for seed in (0, 1, 2)))
    joined_rows, joined_costs = np.concatenate(rows), np.concatenate(costs)
    return base.warm_relative_metrics(
        joined_costs, data["warm_cost"][joined_rows], data["speed_kph"][joined_rows],
        data["variant_index"][joined_rows],
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if (config["formal_validation_or_test_consumed"]
            or config["dbm_fields_or_labels_consumed"]
            or config["query_analytic_gradient_consumed"]):
        raise AssertionError("sealed-boundary contract violated")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    source = Path(config["sources"]["actor_oac"])
    source_validation_path = source / "validation.json"
    source_validation = json.loads(source_validation_path.read_text())
    if source_validation["qualification"] != "QUERY_TARGET_COVERAGE_MIXED_INIT_OAC_INDEPENDENT_TRAIN_SIDE_PASS":
        raise AssertionError("source OAC did not independently pass")
    reused = Path(config["sources"]["validated_seed1_ab"])
    reused_validation_path = reused / "validation.json"
    reused_validation = json.loads(reused_validation_path.read_text())
    if reused_validation["qualification"] != "QUERY_CONTINUOUS_AC_CANDIDATE_BANK_AB_INDEPENDENT_PASS":
        raise AssertionError("reused seed1 A/B did not independently pass")
    reused_summary_path = reused / "summary.json"
    reused_summary = json.loads(reused_summary_path.read_text())

    loader_config = {"outputs": {"absolute_replay": config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader_config)
    replay_validation = json.loads(
        (Path(config["sources"]["absolute_replay"]) / "validation.json").read_text()
    )
    if replay_validation["qualification"] != "QUERY_TARGET_COVERAGE_ABSOLUTE_REPLAY_INDEPENDENT_PASS":
        raise AssertionError("source Replay did not independently pass")
    fit = np.flatnonzero(np.isin(data["fold_id"], config["split_contract"]["fit_folds"]))
    selection = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    outer = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
    if (len(fit), len(selection), len(outer)) != (432, 120, 120):
        raise AssertionError("split sizes changed")
    episode_sets = [set(data["episode_id"][rows].tolist()) for rows in (fit, selection, outer)]
    if any(episode_sets[a] & episode_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("episode leakage")

    output.mkdir(parents=True)
    adapters: dict[str, dict[str, str]] = {}
    for seed in config["pilot"]["new_seeds"]:
        source_checkpoint = source / f"seed_{seed}" / "checkpoint.pt"
        critic = make_critic_adapter(
            source_checkpoint, output / f"source_adapter_seed{seed}", seed
        )
        actor = make_actor_adapter(source_checkpoint, output / "source_actor_adapter", seed)
        adapters[str(seed)] = {
            "source_checkpoint": str(source_checkpoint.resolve()),
            "source_checkpoint_sha256": base.sha256(source_checkpoint),
            "critic": str(critic.resolve()),
            "critic_sha256": base.sha256(critic),
            "actor": str(actor.resolve()),
            "actor_sha256": base.sha256(actor),
        }

    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    rounds = int(config["pilot"]["rounds"])
    radii = np.linspace(
        float(config["pilot"]["probe_radius_sigma_start"]),
        float(config["pilot"]["probe_radius_sigma_end"]),
        int(config["pilot"]["probe_radius_sigma_anneal_rounds"]),
    ).astype(np.float32)[:rounds]
    bases = base.basis_bank()
    base.direct_cost = batched_direct_cost
    records: dict[str, dict[str, Any]] = {
        "1": copy.deepcopy(reused_summary["records"]),
    }
    for seed in config["pilot"]["new_seeds"]:
        seed_records: dict[str, Any] = {}
        for arm, candidate_count in config["pilot"]["candidate_arms"].items():
            arm_config = copy.deepcopy(config)
            arm_config["pilot"]["candidates_per_visit"] = int(candidate_count)
            arm_config["sources"]["critic_pretrain"] = str(
                Path(adapters[str(seed)]["critic"]).parents[1]
            )
            arm_config["sources"]["actor_oac"] = str((output / "source_actor_adapter").resolve())
            arm_output = output / arm
            arm_output.mkdir(exist_ok=True)
            base.response_bank = response_bank39 if arm == "response39" else response_bank65
            record = base.run_seed(
                int(seed), arm_config, data, replay_manifest, controller,
                fit, selection, outer, radii, bases, arm_output, device,
            )
            rewrite_roles(record, arm, int(candidate_count))
            record["bank"] = bank_report(record, int(candidate_count))
            seed_records[arm] = record
        records[str(seed)] = seed_records
    records = {str(seed): records[str(seed)] for seed in (0, 1, 2)}

    pooled_reports = {
        arm: pooled(records, arm, data) for arm in config["pilot"]["candidate_arms"]
    }
    per_seed = {}
    improved_seed_count = 0
    for seed in (0, 1, 2):
        baseline = records[str(seed)]["response39"]
        treatment = records[str(seed)]["response39_recenter26"]
        baseline_mean = float(baseline["selected"]["inner"]["actor_cost"]["mean"])
        treatment_mean = float(treatment["selected"]["inner"]["actor_cost"]["mean"])
        improved = treatment_mean < baseline_mean
        improved_seed_count += int(improved)
        per_seed[str(seed)] = {
            "baseline_selected_round": int(baseline["selected_round"]),
            "treatment_selected_round": int(treatment["selected_round"]),
            "baseline_selected_mean": baseline_mean,
            "treatment_selected_mean": treatment_mean,
            "treatment_mean_reduction": baseline_mean - treatment_mean,
            "baseline_warm_aggregate": float(baseline["selected"]["inner"]["aggregate_improvement"]),
            "treatment_warm_aggregate": float(treatment["selected"]["inner"]["aggregate_improvement"]),
            "treatment_mean_lower": improved,
        }
    baseline_pooled_mean = float(pooled_reports["response39"]["actor_cost"]["mean"])
    treatment_pooled_mean = float(pooled_reports["response39_recenter26"]["actor_cost"]["mean"])
    baseline_pooled_aggregate = float(pooled_reports["response39"]["aggregate_improvement"])
    treatment_pooled_aggregate = float(pooled_reports["response39_recenter26"]["aggregate_improvement"])
    decision_checks = {
        "treatment_pooled_mean_lower": treatment_pooled_mean < baseline_pooled_mean,
        "treatment_pooled_warm_aggregate_higher": treatment_pooled_aggregate > baseline_pooled_aggregate,
        "treatment_mean_lower_in_at_least_two_seeds": improved_seed_count >= 2,
    }
    decision = (
        "PROMOTE_RECENTER65_TO_CONTINUOUS_CAP_SCAN"
        if all(decision_checks.values()) else "RETAIN_RESPONSE39_CONTINUOUS_AC"
    )
    comparison = {
        "per_seed": per_seed,
        "improved_seed_count": improved_seed_count,
        "baseline_pooled_selected_mean": baseline_pooled_mean,
        "treatment_pooled_selected_mean": treatment_pooled_mean,
        "treatment_pooled_mean_reduction": baseline_pooled_mean - treatment_pooled_mean,
        "baseline_pooled_warm_aggregate": baseline_pooled_aggregate,
        "treatment_pooled_warm_aggregate": treatment_pooled_aggregate,
        "treatment_pooled_aggregate_delta": treatment_pooled_aggregate - baseline_pooled_aggregate,
    }
    summary = {
        "qualification": "QUERY_CONTINUOUS_AC_CANDIDATE_BANK_AB_THREE_SEED_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": config,
        "records": records,
        "pooled": pooled_reports,
        "comparison": comparison,
        "decision_checks": decision_checks,
        "decision": decision,
        "new_seed_adapters": adapters,
        "reused_seed1_summary": str(reused_summary_path.resolve()),
        "reused_seed1_summary_sha256": base.sha256(reused_summary_path),
        "reused_seed1_validation": str(reused_validation_path.resolve()),
        "reused_seed1_validation_sha256": base.sha256(reused_validation_path),
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "query-continuous-ac-candidate-bank-ab-three-seed-v1",
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": base.sha256(config_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": base.sha256(Path(__file__).resolve()),
        "source_actor_oac": str(source.resolve()),
        "source_actor_validation_sha256": base.sha256(source_validation_path),
        "source_replay": config["sources"]["absolute_replay"],
        "source_replay_sha256": replay_manifest["replay_sha256"],
        "reused_seed1_summary_sha256": base.sha256(reused_summary_path),
        "reused_seed1_validation_sha256": base.sha256(reused_validation_path),
        "new_seed_adapters": adapters,
        "summary_sha256": base.sha256(output / "summary.json"),
        "result_arrays_sha256": {
            f"seed{seed}_{arm}": records[str(seed)][arm]["arrays_sha256"]
            for seed in (0, 1, 2) for arm in config["pilot"]["candidate_arms"]
        },
        "result_checkpoint_sha256": {
            f"seed{seed}_{arm}": records[str(seed)][arm]["checkpoint_sha256"]
            for seed in (0, 1, 2) for arm in config["pilot"]["candidate_arms"]
        },
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({
        "output": str(output), "decision": decision,
        "comparison": comparison,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
