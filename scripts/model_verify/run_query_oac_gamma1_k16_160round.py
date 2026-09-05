#!/usr/bin/env python3
"""Run the preregistered Query gamma1 K16 160-round extension."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import run_query_oac_gamma1_k_scan as base


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_oac_gamma1_k16_160round_config_20260902_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def piecewise_radii(config: dict) -> np.ndarray:
    pilot = config["pilot"]
    rounds = int(pilot["rounds"])
    anneal = int(pilot["probe_radius_sigma_anneal_rounds"])
    if not 1 <= anneal <= rounds:
        raise AssertionError("invalid probe anneal round count")
    prefix = np.linspace(
        float(pilot["probe_radius_sigma_start"]),
        float(pilot["probe_radius_sigma_end"]),
        anneal,
    ).astype(np.float32)
    return np.concatenate([
        prefix,
        np.full(rounds - anneal, float(pilot["probe_radius_sigma_end"]), np.float32),
    ])


def compact(metric: dict) -> dict:
    return {
        "mean_cost": float(metric["actor_cost"]["mean"]),
        "win_or_tie_fraction": float(metric["win_or_tie_fraction"]),
        "gain_median": float(metric["gain"]["median"]),
        "gain_p05": float(metric["gain"]["p05"]),
        "gain_worst": float(metric["gain"]["min"]),
        "aggregate_improvement": float(metric["aggregate_improvement"]),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    if config["formal_validation_or_test_consumed"] or config["dbm_fields_or_labels_consumed"]:
        raise AssertionError("sealed boundary violation")
    if len(config["arms"]) != 1 or int(config["arms"][0]["actor_updates_per_round"]) != 16:
        raise AssertionError("this runner requires the single preregistered K16 arm")

    aggregation_dir = Path(config["sources"]["aggregation_ab"]).resolve()
    aggregation_validation = json.loads((aggregation_dir / "validation.json").read_text())
    if aggregation_validation["qualification"] != "QUERY_OAC_AGGREGATION_AB_INDEPENDENT_PASS":
        raise AssertionError("source aggregation A/B did not independently pass")
    prior_dir = Path(config["sources"]["k4_k16_90round"]).resolve()
    prior_validation = json.loads((prior_dir / "validation.json").read_text())
    if prior_validation["qualification"] != "QUERY_OAC_GAMMA1_K4_K16_90ROUND_INDEPENDENT_PASS":
        raise AssertionError("source K4/K16 90-round run did not independently pass")
    prior_summary = json.loads((prior_dir / "summary.json").read_text())

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
    radii = piecewise_radii(config)
    output.mkdir(parents=True)
    arm = config["arms"][0]
    records = [
        base.run_arm_seed(
            config, arm, int(seed), data, replay_manifest, pretrain_dir,
            controller, weights, base.basis_bank(), radii, fit, selection, oof,
            output, device,
        )
        for seed in config["pilot"]["seeds"]
    ]
    pooled = {
        stage: {
            "inner": base.pooled_warm(records, stage, "inner", data),
            "development_oof": base.pooled_warm(records, stage, "oof", data),
        }
        for stage in ("round0", "latest", "selected")
    }

    prefix_errors = {
        "probe_radius": 0.0,
        "state_index": 0.0,
        "action": 0.0,
        "raw_action": 0.0,
        "cost": 0.0,
        "selection_round_action": 0.0,
        "selection_round_cost": 0.0,
        "actor_batch_state_index": 0.0,
    }
    prior_records = {
        int(record["seed"]): record
        for record in prior_summary["records"]
        if record["arm"] == config["prefix_reproduction"]["source_arm"]
    }
    prefix_rounds = int(config["prefix_reproduction"]["rounds"])
    for record in records:
        seed = int(record["seed"])
        with np.load(record["arrays"], allow_pickle=False) as current_archive, np.load(
            prior_records[seed]["arrays"], allow_pickle=False
        ) as prior_archive:
            current = {name: np.asarray(current_archive[name]) for name in current_archive.files}
            prior = {name: np.asarray(prior_archive[name]) for name in prior_archive.files}
        candidate_count = prefix_rounds * int(config["pilot"]["fit_contexts_visited_per_round"]) * int(config["pilot"]["candidates_per_visit"])
        actor_batch_count = prefix_rounds * 16 * int(config["actor_updates"]["batch_size"])
        comparisons = {
            "probe_radius": (current["probe_radius_by_round"][:prefix_rounds], prior["probe_radius_by_round"]),
            "state_index": (current["state_index"][:candidate_count], prior["state_index"]),
            "action": (current["action"][:candidate_count], prior["action"]),
            "raw_action": (current["raw_action"][:candidate_count], prior["raw_action"]),
            "cost": (current["cost"][:candidate_count], prior["cost"]),
            "selection_round_action": (current["selection_round_action"][:prefix_rounds + 1], prior["selection_round_action"]),
            "selection_round_cost": (current["selection_round_cost"][:prefix_rounds + 1], prior["selection_round_cost"]),
            "actor_batch_state_index": (current["actor_batch_state_index"][:actor_batch_count], prior["actor_batch_state_index"]),
        }
        for name, (left, right) in comparisons.items():
            prefix_errors[name] = max(prefix_errors[name], base.max_error(left, right) if hasattr(base, "max_error") else float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))))

    prior_metric = prior_summary["pooled_warm_relative"]["gamma1_k16"]["selected"]["inner"]
    current_metric = pooled["selected"]["inner"]
    prior_speed100 = prior_metric["by_speed_kph"]["100"]
    current_speed100 = current_metric["by_speed_kph"]["100"]
    gate = config["decision_gate"]
    prior_by_seed = {int(record["seed"]): record for record in prior_records.values()}
    selected_after_90 = sum(int(record["selected_round"]) > prefix_rounds for record in records)
    mean_advantage = sum(
        record["selected"]["inner_warm_relative"]["actor_cost"]["mean"]
        < prior_by_seed[int(record["seed"])]["selected"]["inner_warm_relative"]["actor_cost"]["mean"]
        for record in records
    )
    checks = {
        "prefix_reproduced": max(prefix_errors.values()) <= float(config["prefix_reproduction"]["required_maximum_absolute_error"]),
        "selected_after_round90_seed_count": selected_after_90 >= int(gate["selected_round_after_90_seed_count_minimum"]),
        "actor_mean_cost_advantage_seed_count": mean_advantage >= int(gate["actor_mean_cost_advantage_seed_count_minimum"]),
        "pooled_aggregate_improvement": current_metric["aggregate_improvement"] > prior_metric["aggregate_improvement"],
        "pooled_median": current_metric["gain"]["median"] >= prior_metric["gain"]["median"] - float(gate["pooled_warm_relative_median_no_worse_tolerance"]),
        "pooled_p05": current_metric["gain"]["p05"] >= prior_metric["gain"]["p05"] - float(gate["pooled_warm_relative_p05_no_worse_tolerance"]),
        "pooled_worst": current_metric["gain"]["min"] >= prior_metric["gain"]["min"] - float(gate["pooled_warm_relative_worst_no_worse_tolerance"]),
        "speed100_aggregate_improvement": current_speed100["aggregate_improvement"] > prior_speed100["aggregate_improvement"],
        "speed100_p05": current_speed100["gain"]["p05"] >= prior_speed100["gain"]["p05"] - float(gate["speed100_p05_no_worse_tolerance"]),
        "speed100_worst": current_speed100["gain"]["min"] >= prior_speed100["gain"]["min"] - float(gate["speed100_worst_no_worse_tolerance"]),
    }
    decision = "GAMMA1_K16_160_EXTENDS_90ROUND_LEARNING" if all(checks.values()) else "GAMMA1_K16_160_NO_RELIABLE_EXTENSION"
    warm_gate = {
        "pooled_aggregate_improvement": current_metric["aggregate_improvement"] >= float(config["inner_warm_gate"]["pooled_aggregate_improvement_minimum"]),
        "pooled_gain_median": current_metric["gain"]["median"] >= float(config["inner_warm_gate"]["pooled_gain_median_minimum"]),
        "speed100_aggregate_improvement": current_speed100["aggregate_improvement"] >= float(config["inner_warm_gate"]["speed100_aggregate_improvement_minimum"]),
    }
    summary = {
        "qualification": "QUERY_OAC_GAMMA1_K16_160ROUND_PENDING_INDEPENDENT_VALIDATION",
        "decision": decision,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": config,
        "records": records,
        "pooled_warm_relative": {"gamma1_k16": pooled},
        "prior_90round_selected_inner": prior_metric,
        "prefix_reproduction_maximum_absolute_errors": prefix_errors,
        "extension_checks": checks,
        "selected_after_round90_seed_count": int(selected_after_90),
        "mean_cost_advantage_seed_count": int(mean_advantage),
        "inner_warm_gate": {"checks": warm_gate, "passes": bool(all(warm_gate.values()))},
        "decision_population": "inner selected checkpoints only",
        "development_oof_role": "corroborating already-consumed evidence; not round-budget selection or untouched validation",
        "new_query_rollouts": int(sum(record["new_query_rollouts"] for record in records)),
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    summary_path = output / "summary.json"
    base.dump_json(summary_path, summary)
    manifest = {
        "schema_version": "query-oac-gamma1-k16-160round-v1",
        "qualification": summary["qualification"],
        "decision": decision,
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
        "k4_k16_90round": str(prior_dir),
        "k4_k16_90round_validation_sha256": base.sha256(prior_dir / "validation.json"),
        "query_checkpoint": replay_manifest["query_checkpoint"],
        "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
        "summary_sha256": base.sha256(summary_path),
        "run_artifacts": {
            f"gamma1_k16_seed{record['seed']}": {
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
        "extension_checks": checks,
        "inner_warm_gate": summary["inner_warm_gate"],
        "prior_90round": compact(prior_metric),
        "selected_160round": compact(current_metric),
        "selected_rounds": [record["selected_round"] for record in records],
        "prefix_errors": prefix_errors,
    }, indent=2))


if __name__ == "__main__":
    main()
