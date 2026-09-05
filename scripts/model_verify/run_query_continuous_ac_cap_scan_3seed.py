#!/usr/bin/env python3
"""Run a paired three-seed continuous Query AC Actor step-cap scan."""

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
    response_bank65,
    rewrite_roles,
)


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_continuous_ac_cap_scan_3seed_config_20260903_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def pooled(records: dict[str, dict[str, Any]], cap: str,
           data: dict[str, np.ndarray]) -> dict[str, Any]:
    costs, rows = [], []
    for seed in (0, 1, 2):
        record = records[cap][str(seed)]
        with np.load(record["arrays"], allow_pickle=False) as archive:
            local_rows = np.asarray(archive["selection_indices"])
            local_cost = np.asarray(archive["selection_round_cost"])[int(record["selected_round"])]
        rows.append(local_rows)
        costs.append(local_cost)
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
    if json.loads(source_validation_path.read_text())["qualification"] != "QUERY_TARGET_COVERAGE_MIXED_INIT_OAC_INDEPENDENT_TRAIN_SIDE_PASS":
        raise AssertionError("source OAC did not independently pass")
    reused = Path(config["sources"]["validated_cap002"])
    reused_summary_path, reused_validation_path = reused / "summary.json", reused / "validation.json"
    reused_summary = json.loads(reused_summary_path.read_text())
    reused_validation = json.loads(reused_validation_path.read_text())
    if reused_validation["qualification"] != "QUERY_CONTINUOUS_AC_CANDIDATE_BANK_AB_THREE_SEED_INDEPENDENT_PASS":
        raise AssertionError("reused cap002 source did not independently pass")

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
    for seed in config["pilot"]["seeds"]:
        source_checkpoint = source / f"seed_{seed}" / "checkpoint.pt"
        critic = make_critic_adapter(
            source_checkpoint, output / f"source_adapter_seed{seed}", seed
        )
        actor = make_actor_adapter(
            source_checkpoint, output / f"source_actor_adapter_seed{seed}", seed
        )
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
    radii = np.linspace(
        float(config["pilot"]["probe_radius_sigma_start"]),
        float(config["pilot"]["probe_radius_sigma_end"]),
        int(config["pilot"]["probe_radius_sigma_anneal_rounds"]),
    ).astype(np.float32)[:int(config["pilot"]["rounds"])]
    bases = base.basis_bank()
    base.direct_cost = batched_direct_cost
    base.response_bank = response_bank65
    records: dict[str, dict[str, Any]] = {
        "cap002": {
            str(seed): copy.deepcopy(
                reused_summary["records"][str(seed)]["response39_recenter26"]
            ) for seed in config["pilot"]["seeds"]
        }
    }
    for cap_name in config["pilot"]["new_cap_arms"]:
        cap_records = {}
        for seed in config["pilot"]["seeds"]:
            arm_config = copy.deepcopy(config)
            arm_config["actor_updates"]["cumulative_per_round_output_step_cap_sigma_rms"] = float(
                config["pilot"]["cap_arms"][cap_name]
            )
            arm_config["sources"]["critic_pretrain"] = str(
                Path(adapters[str(seed)]["critic"]).parents[1]
            )
            arm_config["sources"]["actor_oac"] = str(
                Path(adapters[str(seed)]["actor"]).parents[1]
            )
            cap_output = output / cap_name
            cap_output.mkdir(exist_ok=True)
            record = base.run_seed(
                int(seed), arm_config, data, replay_manifest, controller,
                fit, selection, outer, radii, bases, cap_output, device,
            )
            rewrite_roles(record, "response39_recenter26", 65)
            record["bank"] = bank_report(record, 65)
            cap_records[str(seed)] = record
        records[cap_name] = cap_records

    pooled_reports = {cap: pooled(records, cap, data) for cap in config["pilot"]["cap_arms"]}
    control = "cap002"
    per_cap: dict[str, Any] = {}
    eligible: list[str] = []
    control_mean = float(pooled_reports[control]["actor_cost"]["mean"])
    control_aggregate = float(pooled_reports[control]["aggregate_improvement"])
    for cap in config["pilot"]["new_cap_arms"]:
        improved_seeds = 0
        per_seed = {}
        for seed in config["pilot"]["seeds"]:
            control_record = records[control][str(seed)]
            candidate_record = records[cap][str(seed)]
            local_control = float(control_record["selected"]["inner"]["actor_cost"]["mean"])
            local_candidate = float(candidate_record["selected"]["inner"]["actor_cost"]["mean"])
            improved = local_candidate < local_control
            improved_seeds += int(improved)
            per_seed[str(seed)] = {
                "control_selected_mean": local_control,
                "candidate_selected_mean": local_candidate,
                "candidate_mean_reduction": local_control - local_candidate,
                "control_selected_round": int(control_record["selected_round"]),
                "candidate_selected_round": int(candidate_record["selected_round"]),
                "candidate_mean_lower": improved,
            }
        candidate_mean = float(pooled_reports[cap]["actor_cost"]["mean"])
        candidate_aggregate = float(pooled_reports[cap]["aggregate_improvement"])
        checks = {
            "pooled_mean_lower_than_cap002": candidate_mean < control_mean,
            "pooled_warm_aggregate_higher_than_cap002": candidate_aggregate > control_aggregate,
            "mean_lower_in_at_least_two_seeds": improved_seeds >= 2,
        }
        if all(checks.values()):
            eligible.append(cap)
        per_cap[cap] = {
            "cap": float(config["pilot"]["cap_arms"][cap]),
            "pooled_selected_mean": candidate_mean,
            "pooled_warm_aggregate": candidate_aggregate,
            "pooled_mean_reduction_vs_cap002": control_mean - candidate_mean,
            "pooled_aggregate_delta_vs_cap002": candidate_aggregate - control_aggregate,
            "improved_seed_count": improved_seeds,
            "per_seed": per_seed,
            "decision_checks": checks,
        }
    selected_cap = min(eligible, key=lambda name: float(pooled_reports[name]["actor_cost"]["mean"])) if eligible else control
    decision = (
        f"PROMOTE_{selected_cap.upper()}_TO_ACTOR_LR2E5_AB"
        if selected_cap != control else "RETAIN_CAP002_FOR_ACTOR_LR2E5_AB"
    )
    comparison = {
        "control": {
            "cap": 0.02,
            "pooled_selected_mean": control_mean,
            "pooled_warm_aggregate": control_aggregate,
        },
        "candidates": per_cap,
        "eligible_caps": eligible,
        "selected_cap": selected_cap,
    }
    summary = {
        "qualification": "QUERY_CONTINUOUS_AC_CAP_SCAN_THREE_SEED_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": config,
        "records": records,
        "pooled": pooled_reports,
        "comparison": comparison,
        "decision": decision,
        "source_adapters": adapters,
        "reused_cap002_summary": str(reused_summary_path.resolve()),
        "reused_cap002_summary_sha256": base.sha256(reused_summary_path),
        "reused_cap002_validation": str(reused_validation_path.resolve()),
        "reused_cap002_validation_sha256": base.sha256(reused_validation_path),
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "query-continuous-ac-cap-scan-three-seed-v1",
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
        "reused_cap002_summary_sha256": base.sha256(reused_summary_path),
        "reused_cap002_validation_sha256": base.sha256(reused_validation_path),
        "source_adapters": adapters,
        "summary_sha256": base.sha256(output / "summary.json"),
        "result_arrays_sha256": {
            f"{cap}_seed{seed}": records[cap][str(seed)]["arrays_sha256"]
            for cap in config["pilot"]["cap_arms"] for seed in config["pilot"]["seeds"]
        },
        "result_checkpoint_sha256": {
            f"{cap}_seed{seed}": records[cap][str(seed)]["checkpoint_sha256"]
            for cap in config["pilot"]["cap_arms"] for seed in config["pilot"]["seeds"]
        },
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "decision": decision, "comparison": comparison}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
