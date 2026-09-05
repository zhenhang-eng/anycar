#!/usr/bin/env python3
"""Run the dynamic-history seed-2 OAC screen with batched Query evaluation."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import pretrain_query_single_center_actor_twin_critic as pretrain
import run_query_target_coverage_mixed_init_oac as base
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend
from query_batched_direct_cost import batched_direct_cost


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_dynamic_history_seed2_oac_config_20260903_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    if any((
        config["formal_validation_or_test_consumed"],
        bool(config["dbm_fields_or_labels_consumed"]),
        config["query_analytic_gradient_consumed"],
    )):
        raise AssertionError("sealed-boundary contract violated")
    if config["pilot"]["seeds"] != [2]:
        raise AssertionError("this screening runner is registered for seed 2 only")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    loader_config = {"outputs": {"absolute_replay": config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader_config)
    replay_validation = json.loads(
        (Path(config["sources"]["absolute_replay"]) / "validation.json").read_text()
    )
    if replay_validation["qualification"] != "QUERY_TARGET_COVERAGE_ABSOLUTE_REPLAY_INDEPENDENT_PASS":
        raise AssertionError("dynamic-history replay did not independently pass")
    critic_validation = json.loads(
        (Path(config["sources"]["critic_pretrain"]) / "validation.json").read_text()
    )
    if not critic_validation["checks"]["checkpoint_reload_exact"]:
        raise AssertionError("seed-2 Critic checkpoint did not reproduce")
    if int(critic_validation["global_value_seed_pass_count"]) != 1:
        raise AssertionError("seed-2 Critic lacks the global value signal required for screening")

    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    fit = np.flatnonzero(np.isin(data["fold_id"], config["split_contract"]["fit_folds"]))
    selection = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    outer = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
    if len(selection) != 120 or len(outer) != 120 or len(fit) != len(data["state"]) - 240:
        raise AssertionError(f"unexpected split sizes {(len(fit), len(selection), len(outer))}")
    episode_sets = [set(data["episode_id"][rows].tolist()) for rows in (fit, selection, outer)]
    if episode_sets[0] & episode_sets[1] or episode_sets[0] & episode_sets[2] or episode_sets[1] & episode_sets[2]:
        raise AssertionError("episode leakage")

    rounds = int(config["pilot"]["rounds"])
    anneal = int(config["pilot"]["probe_radius_sigma_anneal_rounds"])
    radii = np.concatenate((
        np.linspace(
            float(config["pilot"]["probe_radius_sigma_start"]),
            float(config["pilot"]["probe_radius_sigma_end"]),
            anneal,
        ),
        np.full(rounds - anneal, float(config["pilot"]["probe_radius_sigma_end"])),
    )).astype(np.float32)

    # The shared training implementation resolves this module global at call
    # time.  Replace only the result-equivalent direct-cost evaluator; all
    # optimizer schedules and checkpoint selection remain unchanged.
    base.direct_cost = batched_direct_cost
    output.mkdir(parents=True)
    records = [
        base.run_seed(
            2, config, data, replay_manifest, controller, fit, selection,
            outer, radii, base.basis_bank(), output, device,
        )
    ]
    pooled_metrics = {
        "round0_inner": base.pooled(records, "round0", "inner", data),
        "selected_inner": base.pooled(records, "selected", "inner", data),
        "round0_fit": base.pooled(records, "round0", "fit", data),
        "selected_fit": base.pooled(records, "selected", "fit", data),
    }
    record = records[0]
    decision_checks = {
        "selected_inner_mean_not_worse": (
            record["selected"]["inner"]["actor_cost"]["mean"]
            <= record["round0"]["inner"]["actor_cost"]["mean"] + 1e-9
        ),
        "selected_after_round0": int(record["selected_round"]) > 0,
        "inner_mean_improved": (
            pooled_metrics["selected_inner"]["actor_cost"]["mean"]
            < pooled_metrics["round0_inner"]["actor_cost"]["mean"]
        ),
    }
    decision = (
        "DYNAMIC_HISTORY_SEED2_SCREEN_PROGRESS"
        if all(decision_checks.values())
        else "DYNAMIC_HISTORY_SEED2_SCREEN_NO_PROGRESS"
    )
    summary = {
        "qualification": "QUERY_DYNAMIC_HISTORY_SEED2_OAC_TRAIN_SIDE_SCREEN_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": config,
        "records": records,
        "pooled": pooled_metrics,
        "decision": decision,
        "decision_checks": decision_checks,
        "query_direct_cost_evaluator": {
            "mode": "multi-context batch",
            "helper": str((REPO_ROOT / "scripts/model_verify/query_batched_direct_cost.py").resolve()),
            "helper_sha256": base.sha256(REPO_ROOT / "scripts/model_verify/query_batched_direct_cost.py"),
            "semantic_validation": str((REPO_ROOT / "outputs/query_mppi/query_batched_direct_cost_validation_20260903_v1/validation.json").resolve()),
        },
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    base.dump_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "query-dynamic-history-seed2-oac-screen-v1",
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": base.sha256(config_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": base.sha256(Path(__file__).resolve()),
        "shared_training_implementation": str(Path(base.__file__).resolve()),
        "shared_training_implementation_sha256": base.sha256(Path(base.__file__).resolve()),
        "batched_direct_cost_helper_sha256": base.sha256(REPO_ROOT / "scripts/model_verify/query_batched_direct_cost.py"),
        "source_replay": config["sources"]["absolute_replay"],
        "source_replay_sha256": replay_manifest["replay_sha256"],
        "source_critic_pretrain": config["sources"]["critic_pretrain"],
        "source_actor_oac": config["sources"]["actor_oac"],
        "summary_sha256": base.sha256(output / "summary.json"),
        "arrays_sha256": {Path(record["arrays"]).name: record["arrays_sha256"]},
        "checkpoint_sha256": {Path(record["checkpoint"]).name: record["checkpoint_sha256"]},
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    base.dump_json(output / "manifest.json", manifest)
    print(json.dumps({
        "output": str(output),
        "decision": decision,
        "selected_round": record["selected_round"],
        "round0_inner_mean": pooled_metrics["round0_inner"]["actor_cost"]["mean"],
        "selected_inner_mean": pooled_metrics["selected_inner"]["actor_cost"]["mean"],
        "outer_fold_evaluated": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
