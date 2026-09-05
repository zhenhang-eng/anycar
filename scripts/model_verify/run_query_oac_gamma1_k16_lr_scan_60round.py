#!/usr/bin/env python3
"""Run a deterministic paired Query gamma1 K16 Actor-LR screen."""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import run_query_oac_gamma1_k_scan as base


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_oac_gamma1_k16_lr_scan_60round_config_20260903_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def deterministic_contract(config: dict) -> dict:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    mode = str(config["pairing"]["deterministic_cuda_mode"])
    if mode not in ("strict", "warn_only"):
        raise AssertionError(f"unsupported deterministic CUDA mode: {mode}")
    torch.use_deterministic_algorithms(True, warn_only=mode == "warn_only")
    return {
        "mode": mode,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def tail_checks(candidate: dict, baseline: dict, config: dict) -> dict:
    spec = config["lr_tail_gate"]
    candidate100 = candidate["by_speed_kph"]["100"]
    baseline100 = baseline["by_speed_kph"]["100"]
    checks = {
        "speed100_aggregate_improvement": candidate100["aggregate_improvement"] > baseline100["aggregate_improvement"],
        "speed100_median": candidate100["gain"]["median"] >= baseline100["gain"]["median"] - float(spec["speed100_median_no_worse_tolerance"]),
        "speed100_p05": candidate100["gain"]["p05"] >= baseline100["gain"]["p05"] - float(spec["speed100_p05_no_worse_tolerance"]),
        "speed100_worst": candidate100["gain"]["min"] >= baseline100["gain"]["min"] - float(spec["speed100_worst_no_worse_tolerance"]),
    }
    for slice_name in spec["hard_slices"]:
        local = candidate["by_speed_variant"][slice_name]
        reference = baseline["by_speed_variant"][slice_name]
        checks[f"hard_{slice_name}_aggregate"] = local["aggregate_improvement"] >= reference["aggregate_improvement"] - float(spec["hard_slice_aggregate_no_worse_tolerance"])
        checks[f"hard_{slice_name}_p05"] = local["gain"]["p05"] >= reference["gain"]["p05"] - float(spec["hard_slice_p05_no_worse_tolerance"])
        checks[f"hard_{slice_name}_worst"] = local["gain"]["min"] >= reference["gain"]["min"] - float(spec["hard_slice_worst_no_worse_tolerance"])
    return checks


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    if config["formal_validation_or_test_consumed"] or config["dbm_fields_or_labels_consumed"]:
        raise AssertionError("sealed boundary violation")
    deterministic = deterministic_contract(config)
    expected_lrs = [2e-6, 5e-6, 1e-5]
    if [float(arm["learning_rate_per_microstep"]) for arm in config["arms"]] != expected_lrs:
        raise AssertionError("unexpected Actor-LR arms")
    if any(int(arm["actor_updates_per_round"]) != 16 for arm in config["arms"]):
        raise AssertionError("all LR arms must use K16")

    aggregation_dir = Path(config["sources"]["aggregation_ab"]).resolve()
    aggregation_validation = json.loads((aggregation_dir / "validation.json").read_text())
    if aggregation_validation["qualification"] != "QUERY_OAC_AGGREGATION_AB_INDEPENDENT_PASS":
        raise AssertionError("source aggregation A/B did not independently pass")
    prior_dir = Path(config["sources"]["k16_160round"]).resolve()
    prior_validation = json.loads((prior_dir / "validation.json").read_text())
    if prior_validation["qualification"] != "QUERY_OAC_GAMMA1_K16_160ROUND_INDEPENDENT_PASS":
        raise AssertionError("source K16/160 artifact did not independently pass")
    replay_dir = Path(config["sources"]["absolute_replay"]).resolve()
    pretrain_dir = Path(config["sources"]["pretrain"]).resolve()
    replay_manifest = json.loads((replay_dir / "manifest.json").read_text())
    with np.load(replay_dir / "replay.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    fullrank_manifest = json.loads((Path(replay_manifest["fullrank_source"]) / "manifest.json").read_text())
    parent_manifest = json.loads((Path(fullrank_manifest["parent_t0"]) / "manifest.json").read_text())
    collection_manifest = json.loads((Path(parent_manifest["source_collection"]) / "manifest.json").read_text())
    device = torch.device(args.device)
    query = base.QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = base.TorchMPPIController(
        base.TorchQueryRolloutBackend(query),
        base.TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    weights = {name: float(value) for name, value in config["cost_weights"].items()}
    fit = np.flatnonzero(np.isin(data["fold_id"], config["split_contract"]["fit_folds"]))
    selection = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    oof = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
    radii = np.linspace(
        float(config["pilot"]["probe_radius_sigma_start"]),
        float(config["pilot"]["probe_radius_sigma_end"]),
        int(config["pilot"]["rounds"]),
    ).astype(np.float32)
    output.mkdir(parents=True)
    records = []
    bases = base.basis_bank()
    for arm in config["arms"]:
        local_config = copy.deepcopy(config)
        local_config["actor_updates"]["learning_rate_per_microstep"] = float(arm["learning_rate_per_microstep"])
        for seed in config["pilot"]["seeds"]:
            record = base.run_arm_seed(
                local_config, arm, int(seed), data, replay_manifest, pretrain_dir,
                controller, weights, bases, radii, fit, selection, oof, output, device,
            )
            record["learning_rate_per_microstep"] = float(arm["learning_rate_per_microstep"])
            records.append(record)
    by_arm = {
        arm["name"]: [record for record in records if record["arm"] == arm["name"]]
        for arm in config["arms"]
    }
    pooled = {
        arm: {
            stage: {
                "inner": base.pooled_warm(local, stage, "inner", data),
                "development_oof": base.pooled_warm(local, stage, "oof", data),
            }
            for stage in ("round0", "latest", "selected")
        }
        for arm, local in by_arm.items()
    }
    baseline = config["arms"][0]["name"]
    baseline_records = {int(record["seed"]): record for record in by_arm[baseline]}
    gate = config["decision_gate"]
    candidate_checks = {}
    decision = "GAMMA1_K_SCAN_FAIL_RETAIN_K1_COST_REFERENCE"
    for candidate in gate["candidate_order"]:
        candidate_records = {int(record["seed"]): record for record in by_arm[candidate]}
        seed_count = sum(
            candidate_records[seed]["selected"]["inner_warm_relative"]["actor_cost"]["mean"]
            < baseline_records[seed]["selected"]["inner_warm_relative"]["actor_cost"]["mean"]
            for seed in baseline_records
        )
        candidate_primary = pooled[candidate]["selected"]["inner"]
        baseline_primary = pooled[baseline]["selected"]["inner"]
        checks = {
            "actor_mean_cost_advantage_seed_count": seed_count >= int(gate["candidate_actor_mean_cost_advantage_seed_count_minimum"]),
            "pooled_aggregate_improvement_greater_than_k1": candidate_primary["aggregate_improvement"] > baseline_primary["aggregate_improvement"],
            "pooled_median_within_tolerance": candidate_primary["gain"]["median"] >= baseline_primary["gain"]["median"] - float(gate["candidate_pooled_warm_relative_median_no_worse_tolerance"]),
            "pooled_p05_within_tolerance": candidate_primary["gain"]["p05"] >= baseline_primary["gain"]["p05"] - float(gate["candidate_pooled_warm_relative_p05_no_worse_tolerance"]),
            "pooled_worst_within_tolerance": candidate_primary["gain"]["min"] >= baseline_primary["gain"]["min"] - float(gate["candidate_pooled_warm_relative_worst_no_worse_tolerance"]),
        }
        candidate_checks[candidate] = {
            "mean_cost_advantage_seed_count": int(seed_count),
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
        if decision.endswith("K1_COST_REFERENCE") and all(checks.values()):
            decision = f"ADVANCE_{candidate.upper()}_TO_LONGER_GAMMA1_QUERY_PILOT"

    tail_review = {}
    recommendation = "KEEP_ACTOR_LR_2E6"
    for candidate in gate["candidate_order"]:
        checks = tail_checks(
            pooled[candidate]["selected"]["inner"],
            pooled[baseline]["selected"]["inner"],
            config,
        )
        overall_passes = bool(candidate_checks[candidate]["passes"])
        tail_review[candidate] = {
            "checks": checks,
            "overall_gate_passes": overall_passes,
            "passes": bool(overall_passes and all(checks.values())),
        }
        if recommendation == "KEEP_ACTOR_LR_2E6" and overall_passes and all(checks.values()):
            recommendation = f"RAISE_ACTOR_LR_TO_{candidate.rsplit('_lr', 1)[-1].upper()}"

    summary = {
        "qualification": "QUERY_OAC_GAMMA1_K16_LR_SCAN_PENDING_INDEPENDENT_VALIDATION",
        "decision": decision,
        "lr_recommendation": recommendation,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": config,
        "deterministic_runtime_contract": deterministic,
        "records": records,
        "pooled_warm_relative": pooled,
        "candidate_checks": candidate_checks,
        "lr_tail_review": tail_review,
        "decision_population": "inner selected checkpoints only",
        "development_oof_role": "corroborating already-consumed evidence; not LR selection or untouched validation",
        "new_query_rollouts": int(sum(record["new_query_rollouts"] for record in records)),
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    summary_path = output / "summary.json"
    base.dump_json(summary_path, summary)
    manifest = {
        "schema_version": "query-oac-gamma1-k16-lr-scan-v1",
        "qualification": summary["qualification"],
        "decision": decision,
        "lr_recommendation": recommendation,
        "config": str(config_path),
        "config_sha256": base.sha256(config_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": base.sha256(Path(__file__).resolve()),
        "shared_runner": str(Path(base.__file__).resolve()),
        "shared_runner_sha256": base.sha256(Path(base.__file__).resolve()),
        "absolute_replay": str(replay_dir),
        "absolute_replay_sha256": replay_manifest["replay_sha256"],
        "pretrain": str(pretrain_dir),
        "pretrain_manifest_sha256": base.sha256(pretrain_dir / "manifest.json"),
        "aggregation_ab": str(aggregation_dir),
        "aggregation_validation_sha256": base.sha256(aggregation_dir / "validation.json"),
        "k16_160round": str(prior_dir),
        "k16_160round_validation_sha256": base.sha256(prior_dir / "validation.json"),
        "query_checkpoint": replay_manifest["query_checkpoint"],
        "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
        "summary_sha256": base.sha256(summary_path),
        "run_artifacts": {
            f"{record['arm']}_seed{record['seed']}": {
                "arrays_sha256": record["arrays_sha256"],
                "checkpoint_sha256": record["checkpoint_sha256"],
            }
            for record in records
        },
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    base.dump_json(output / "manifest.json", manifest)
    print(json.dumps({
        "decision": decision,
        "lr_recommendation": recommendation,
        "candidate_checks": candidate_checks,
        "lr_tail_review": tail_review,
        "selected_inner": {arm: pooled[arm]["selected"]["inner"] for arm in pooled},
    }, indent=2))


if __name__ == "__main__":
    main()
