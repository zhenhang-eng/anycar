#!/usr/bin/env python3
"""Coarsely initialize the no-anchor Actor and strictly fit twin Critics on expanded fit folds."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import pretrain_query_single_center_actor_twin_critic as base
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_target_coverage_fixed_split_pretrain_config_20260903_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config["formal_validation_or_test_consumed"] or config["dbm_fields_or_labels_consumed"] or config["query_analytic_gradient_consumed"]:
        raise AssertionError("sealed-boundary contract violated")
    output = Path(config["outputs"]["pretrain"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    data, source_manifest, collection_manifest = base.load_data(config)
    fit = np.flatnonzero(np.isin(data["fold_id"], config["split_contract"]["fit_folds"]))
    selection = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    outer = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
    if len(selection) != 120 or len(outer) != 120 or len(fit) != len(data["state"]) - 240:
        raise AssertionError(f"unexpected split sizes {(len(fit), len(selection), len(outer))}")
    episode_sets = [set(data["episode_id"][rows].tolist()) for rows in (fit, selection, outer)]
    if episode_sets[0] & episode_sets[1] or episode_sets[0] & episode_sets[2] or episode_sets[1] & episode_sets[2]:
        raise AssertionError("episode leakage")

    device = torch.device(args.device)
    query_model = QueryDeploymentModel.from_checkpoint(Path(source_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query_model),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    inputs, normalizer = base.normalized_inputs(data, fit)
    output.mkdir(parents=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    records = []
    gates = config["critic_inner_gates"]
    for seed in [int(value) for value in config["critic"]["seeds"]]:
        print(f"seed={seed}: Actor coarse initialization", flush=True)
        actor, actor_training = base.train_actor(
            data, inputs, fit, selection, 31_000 + seed, config, controller, device
        )
        actor_splits = {}
        for name, rows in (("fit", fit), ("inner", selection)):
            prediction = base.actor_predict(actor, inputs, rows, device)
            cost = base.query_cost(controller, data, prediction, rows)
            actor_splits[name] = base.actor_metrics(
                cost, data["warm_cost"][rows], data["actor_target_cost"][rows]
            )

        critics, trainings, critic_splits = [], [], []
        for twin in range(2):
            print(f"seed={seed}: Critic {twin + 1}/2 strict 240 epochs", flush=True)
            critic, training = base.train_critic(
                data, inputs, fit, selection, 41_000 + seed * 10 + twin,
                config, device,
            )
            splits = {}
            for name, rows in (("fit", fit), ("inner", selection)):
                prediction = base.critic_predict(critic, inputs, data, rows, device)
                physical = prediction * training["target_std"] + training["target_mean"]
                splits[name] = {
                    "metrics": base.state_metrics(physical, data, rows),
                    "landscape_gain_recovery": base.landscape_gain_recovery(physical, data, rows),
                }
            critics.append(critic)
            trainings.append(training)
            critic_splits.append(splits)

        twin_splits = {}
        for name, rows in (("fit", fit), ("inner", selection)):
            physical = []
            for critic, training in zip(critics, trainings):
                prediction = base.critic_predict(critic, inputs, data, rows, device)
                physical.append(prediction * training["target_std"] + training["target_mean"])
            conservative = np.fmax(physical[0], physical[1])
            twin_splits[name] = {
                "metrics": base.state_metrics(conservative, data, rows),
                "landscape_gain_recovery": base.landscape_gain_recovery(conservative, data, rows),
            }
        inner = twin_splits["inner"]
        inner_pass = (
            inner["metrics"]["state_pearson_log_cost"]["median"] >= gates["log_cost_pearson_median_minimum"]
            and inner["metrics"]["state_pair_sign_accuracy"]["median"] >= gates["same_state_pair_sign_accuracy_median_minimum"]
            and inner["landscape_gain_recovery"] >= gates["landscape_bank_gain_recovery_minimum"]
        )
        checkpoint = checkpoint_dir / f"seed_{seed}.pt"
        invariance = base.no_anchor_invariance(actor, inputs, selection, device)
        torch.save(
            {
                "qualification": "QUERY_TARGET_COVERAGE_FIXED_SPLIT_PRETRAIN_TRAIN_ONLY",
                "seed": seed,
                "actor_architecture": "DirectNoAnchorGTXActor",
                "actor_state_dict": {name: value.detach().cpu() for name, value in actor.state_dict().items()},
                "actor_training": actor_training,
                "critic_architecture": "ConfigurableAbsoluteActionValueCritic(base,pair_delta=false)",
                "critic1_state_dict": {name: value.detach().cpu() for name, value in critics[0].state_dict().items()},
                "critic2_state_dict": {name: value.detach().cpu() for name, value in critics[1].state_dict().items()},
                "critic1_training": trainings[0],
                "critic2_training": trainings[1],
                "normalization": normalizer.to_dict(),
                "fit_indices": fit,
                "selection_indices": selection,
                "outer_indices_unevaluated": outer,
                "source_replay_sha256": source_manifest["replay_sha256"],
                "query_checkpoint_sha256": source_manifest["query_checkpoint_sha256"],
                "no_anchor_invariance": invariance,
                "formal_validation_or_test_consumed": False,
                "dbm_fields_or_labels_consumed": [],
                "query_analytic_gradient_consumed": False,
            },
            checkpoint,
        )
        records.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": base.sha256(checkpoint),
                "actor": actor_splits,
                "critic1": critic_splits[0],
                "critic2": critic_splits[1],
                "critic_twin_conservative": twin_splits,
                "inner_gates_pass": bool(inner_pass),
                "no_anchor_invariance": invariance,
            }
        )
        print(
            f"seed={seed}: inner corr={inner['metrics']['state_pearson_log_cost']['median']:.4f} "
            f"pair={inner['metrics']['state_pair_sign_accuracy']['median']:.4f} "
            f"landR={inner['landscape_gain_recovery']:.4f} pass={inner_pass}",
            flush=True,
        )

    pass_count = int(sum(record["inner_gates_pass"] for record in records))
    passed = pass_count >= int(gates["minimum_seeds_passing_all_gates"])
    qualification = (
        "QUERY_TARGET_COVERAGE_FIXED_SPLIT_PRETRAIN_INNER_PASS_PENDING_INDEPENDENT_VALIDATION"
        if passed
        else "QUERY_TARGET_COVERAGE_FIXED_SPLIT_PRETRAIN_INNER_FAIL"
    )
    summary = {
        "qualification": qualification,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": config,
        "split_counts": {"fit": len(fit), "inner": len(selection), "outer_unevaluated": len(outer)},
        "inner_seed_pass_count": pass_count,
        "records": records,
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    summary_path = output / "summary.json"
    base.dump_json(summary_path, summary)
    manifest = {
        "schema_version": "query-target-coverage-fixed-split-pretrain-v1",
        "qualification": qualification,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": base.sha256(config_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": base.sha256(Path(__file__).resolve()),
        "source_replay": str(config["outputs"]["absolute_replay"]),
        "source_replay_sha256": source_manifest["replay_sha256"],
        "query_checkpoint": source_manifest["query_checkpoint"],
        "query_checkpoint_sha256": source_manifest["query_checkpoint_sha256"],
        "summary_sha256": base.sha256(summary_path),
        "checkpoint_sha256": {Path(record["checkpoint"]).name: record["checkpoint_sha256"] for record in records},
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    base.dump_json(output / "manifest.json", manifest)
    print(json.dumps({"qualification": qualification, "inner_seed_pass_count": pass_count}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
