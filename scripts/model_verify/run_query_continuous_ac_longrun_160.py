#!/usr/bin/env python3
"""Run the promoted three-seed Query continuous-AC contract for 160 rounds."""

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


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_continuous_ac_longrun_160_config_20260904_v1.json"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume-completed-seeds", action="store_true")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def schedule_prefix_errors(record: dict[str, Any], previous: dict[str, Any]) -> dict[str, float]:
    errors = {}
    with np.load(record["arrays"], allow_pickle=False) as current, np.load(
        previous["arrays"], allow_pickle=False
    ) as old:
        pairs = {
            "actor_batch_schedule": (
                np.asarray(current["actor_batch_schedule"][:40]),
                np.asarray(old["actor_batch_schedule"]),
            ),
            "visited_rows": (
                np.asarray(current["online_state_index"]).reshape(-1, 65)[:800, 0],
                np.asarray(old["online_state_index"]).reshape(-1, 65)[:, 0],
            ),
            "online_round": (
                np.asarray(current["online_round"]).reshape(-1, 65)[:800, 0],
                np.asarray(old["online_round"]).reshape(-1, 65)[:, 0],
            ),
            "round0_action": (
                np.asarray(current["selection_round_action"][0]),
                np.asarray(old["selection_round_action"][0]),
            ),
            "round0_cost": (
                np.asarray(current["selection_round_cost"][0]),
                np.asarray(old["selection_round_cost"][0]),
            ),
        }
        for field, (actual, expected) in pairs.items():
            if actual.shape != expected.shape:
                errors[field] = float("inf")
            else:
                errors[field] = float(np.max(np.abs(
                    actual.astype(np.float64) - expected.astype(np.float64)
                )))
    return errors


def best_through_round(record: dict[str, Any], final_round: int,
                       data: dict[str, np.ndarray]) -> dict[str, Any]:
    with np.load(record["arrays"], allow_pickle=False) as archive:
        costs = np.asarray(archive["selection_round_cost"][: final_round + 1])
        rows = np.asarray(archive["selection_indices"])
    means = costs.astype(np.float64).mean(axis=1)
    selected_round = int(np.argmin(means))
    report = base.warm_relative_metrics(
        costs[selected_round], data["warm_cost"][rows], data["speed_kph"][rows],
        data["variant_index"][rows],
    )
    return {"selected_round": selected_round, "inner": report}


def pooled_stage(stages: list[dict[str, Any]], data: dict[str, np.ndarray],
                 selection: np.ndarray) -> dict[str, Any]:
    costs = []
    for stage in stages:
        costs.append(np.asarray(stage["selected_cost"], np.float32))
    rows = np.tile(selection, len(stages))
    return base.warm_relative_metrics(
        np.concatenate(costs), data["warm_cost"][rows], data["speed_kph"][rows],
        data["variant_index"][rows],
    )


def record_valid(record: dict[str, Any]) -> bool:
    return (
        Path(record["arrays"]).is_file()
        and Path(record["checkpoint"]).is_file()
        and base.sha256(Path(record["arrays"])) == record["arrays_sha256"]
        and base.sha256(Path(record["checkpoint"])) == record["checkpoint_sha256"]
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists() and not args.resume_completed_seeds:
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if (config["formal_validation_or_test_consumed"]
            or config["dbm_fields_or_labels_consumed"]
            or config["query_analytic_gradient_consumed"]):
        raise AssertionError("sealed-boundary contract violated")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    source = Path(config["sources"]["actor_oac"])
    source_validation_path = source / "validation.json"
    if json.loads(source_validation_path.read_text())["qualification"] != "QUERY_TARGET_COVERAGE_MIXED_INIT_OAC_INDEPENDENT_TRAIN_SIDE_PASS":
        raise AssertionError("source OAC did not independently pass")
    previous = Path(config["sources"]["validated_40round_lr2e5"])
    previous_summary_path, previous_validation_path = previous / "summary.json", previous / "validation.json"
    previous_summary = json.loads(previous_summary_path.read_text())
    previous_validation = json.loads(previous_validation_path.read_text())
    if previous_validation["qualification"] != "QUERY_CONTINUOUS_AC_ACTOR_LR2E5_AB_THREE_SEED_INDEPENDENT_PASS":
        raise AssertionError("40-round LR2e-5 source did not independently pass")

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

    output.mkdir(parents=True, exist_ok=args.resume_completed_seeds)
    adapters: dict[str, dict[str, str]] = {}
    for seed in config["pilot"]["seeds"]:
        source_checkpoint = source / f"seed_{seed}" / "checkpoint.pt"
        critic_root = output / f"source_adapter_seed{seed}"
        actor_root = output / f"source_actor_adapter_seed{seed}"
        critic = critic_root / "checkpoints" / f"seed_{seed}.pt"
        actor = actor_root / f"seed_{seed}" / "checkpoint.pt"
        if not critic.exists():
            critic = make_critic_adapter(source_checkpoint, critic_root, seed)
        if not actor.exists():
            actor = make_actor_adapter(source_checkpoint, actor_root, seed)
        adapters[str(seed)] = {
            "source_checkpoint": str(source_checkpoint.resolve()),
            "source_checkpoint_sha256": base.sha256(source_checkpoint),
            "critic": str(critic.resolve()), "critic_sha256": base.sha256(critic),
            "actor": str(actor.resolve()), "actor_sha256": base.sha256(actor),
        }

    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query), TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    rounds = int(config["pilot"]["rounds"])
    anneal = int(config["pilot"]["probe_radius_sigma_anneal_rounds"])
    radii = np.concatenate((
        np.linspace(float(config["pilot"]["probe_radius_sigma_start"]),
                    float(config["pilot"]["probe_radius_sigma_end"]), anneal),
        np.full(rounds - anneal, float(config["pilot"]["probe_radius_sigma_end"])),
    )).astype(np.float32)
    bases = base.basis_bank()
    base.direct_cost = batched_direct_cost
    base.response_bank = response_bank65
    records = []
    prefix_reports = {}
    for seed in config["pilot"]["seeds"]:
        record_path = output / f"seed_{seed}_record.json"
        if record_path.exists():
            if not args.resume_completed_seeds:
                raise FileExistsError(record_path)
            record = json.loads(record_path.read_text())
            if not record_valid(record):
                raise AssertionError(f"completed seed{seed} record failed hash check")
            print(f"seed={seed} reusing completed record", flush=True)
        else:
            arm_config = copy.deepcopy(config)
            arm_config["sources"]["critic_pretrain"] = str(
                Path(adapters[str(seed)]["critic"]).parents[1]
            )
            arm_config["sources"]["actor_oac"] = str(
                Path(adapters[str(seed)]["actor"]).parents[1]
            )
            run_root = output / "longrun"
            run_root.mkdir(exist_ok=True)
            record = base.run_seed(
                int(seed), arm_config, data, replay_manifest, controller,
                fit, selection, outer, radii, bases, run_root, device,
            )
            rewrite_roles(record, "response39_recenter26", 65)
            record["bank"] = bank_report(record, 65)
            dump_json(record_path, record)
        previous_record = previous_summary["records"]["lr2e5"][str(seed)]
        errors = schedule_prefix_errors(record, previous_record)
        if not all(value == 0.0 for value in errors.values()):
            raise AssertionError(f"seed{seed} schedule/round0 prefix mismatch: {errors}")
        prefix_reports[str(seed)] = errors
        records.append(record)

    pooled_report = base.pooled(records, "selected", "inner", data)
    historical_records = [previous_summary["records"]["lr2e5"][str(seed)] for seed in config["pilot"]["seeds"]]
    historical_pooled = base.pooled(historical_records, "selected", "inner", data)
    round40_stages = []
    for record in records:
        stage = best_through_round(record, 40, data)
        with np.load(record["arrays"], allow_pickle=False) as archive:
            stage["selected_cost"] = np.asarray(
                archive["selection_round_cost"][stage["selected_round"]]
            ).tolist()
        round40_stages.append(stage)
    round40_pooled = pooled_stage(round40_stages, data, selection)
    per_seed = {}
    improved_count = 0
    for seed, record, stage, historical in zip(
        config["pilot"]["seeds"], records, round40_stages, historical_records
    ):
        old_mean = float(stage["inner"]["actor_cost"]["mean"])
        new_mean = float(record["selected"]["inner"]["actor_cost"]["mean"])
        improved = new_mean < old_mean
        improved_count += int(improved)
        per_seed[str(seed)] = {
            "round40_selected_mean": old_mean,
            "round40_selected_round": int(stage["selected_round"]),
            "longrun_selected_mean": new_mean,
            "longrun_selected_round": int(record["selected_round"]),
            "longrun_mean_reduction": old_mean - new_mean,
            "strictly_improved": improved,
            "historical_40round_selected_mean": float(
                historical["selected"]["inner"]["actor_cost"]["mean"]
            ),
            "historical_40round_selected_round": int(historical["selected_round"]),
        }
    old_mean = float(round40_pooled["actor_cost"]["mean"])
    new_mean = float(pooled_report["actor_cost"]["mean"])
    old_aggregate = float(round40_pooled["aggregate_improvement"])
    new_aggregate = float(pooled_report["aggregate_improvement"])
    checks = {
        "all_schedule_and_round0_prefixes_exact": all(
            value == 0.0 for report in prefix_reports.values() for value in report.values()
        ),
        "longrun_pooled_mean_lower": new_mean < old_mean,
        "longrun_pooled_warm_aggregate_higher": new_aggregate > old_aggregate,
        "longrun_mean_lower_in_at_least_two_seeds": improved_count >= 2,
    }
    decision = "PROMOTE_160ROUND_CONTINUOUS_AC" if all(checks.values()) else "RETAIN_40ROUND_CONTINUOUS_AC"
    comparison = {
        "round40_pooled_selected_mean": old_mean,
        "longrun_pooled_selected_mean": new_mean,
        "longrun_pooled_mean_reduction": old_mean - new_mean,
        "round40_pooled_warm_aggregate": old_aggregate,
        "longrun_pooled_warm_aggregate": new_aggregate,
        "longrun_pooled_aggregate_delta": new_aggregate - old_aggregate,
        "improved_seed_count": improved_count,
        "per_seed": per_seed,
        "historical_40round_pooled_selected_mean": float(
            historical_pooled["actor_cost"]["mean"]
        ),
        "historical_40round_pooled_warm_aggregate": float(
            historical_pooled["aggregate_improvement"]
        ),
    }
    summary = {
        "qualification": "QUERY_CONTINUOUS_AC_LONGRUN_160_THREE_SEED_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(), "contract": config,
        "records": {str(seed): record for seed, record in zip(config["pilot"]["seeds"], records)},
        "pooled": pooled_report, "within_run_round40_pooled": round40_pooled,
        "within_run_round40_stages": round40_stages,
        "historical_round40_pooled": historical_pooled,
        "prefix_errors": prefix_reports, "comparison": comparison,
        "decision_checks": checks, "decision": decision, "source_adapters": adapters,
        "completed_seed_records": [str((output / f"seed_{seed}_record.json").resolve())
                                   for seed in config["pilot"]["seeds"]],
        "resume_scope": "completed seeds only; no within-seed optimizer/Replay/RNG resume",
        "historical_40round_role": "provenance and schedule/round0 contract only; old wrapper did not enable deterministic CUDA training",
        "deterministic_runtime_contract": {
            "mode": "warn_only",
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        },
        "previous_summary": str(previous_summary_path.resolve()),
        "previous_summary_sha256": base.sha256(previous_summary_path),
        "previous_validation": str(previous_validation_path.resolve()),
        "previous_validation_sha256": base.sha256(previous_validation_path),
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "query-continuous-ac-longrun-160-three-seed-v1",
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path), "config_sha256": base.sha256(config_path),
        "script": str(Path(__file__).resolve()), "script_sha256": base.sha256(Path(__file__).resolve()),
        "source_actor_oac": str(source.resolve()),
        "source_actor_validation_sha256": base.sha256(source_validation_path),
        "source_replay": config["sources"]["absolute_replay"],
        "source_replay_sha256": replay_manifest["replay_sha256"],
        "previous_summary_sha256": base.sha256(previous_summary_path),
        "previous_validation_sha256": base.sha256(previous_validation_path),
        "source_adapters": adapters, "summary_sha256": base.sha256(output / "summary.json"),
        "result_arrays_sha256": {str(seed): record["arrays_sha256"]
                                 for seed, record in zip(config["pilot"]["seeds"], records)},
        "result_checkpoint_sha256": {str(seed): record["checkpoint_sha256"]
                                     for seed, record in zip(config["pilot"]["seeds"], records)},
        "seed_record_sha256": {str(seed): base.sha256(output / f"seed_{seed}_record.json")
                               for seed in config["pilot"]["seeds"]},
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "decision": decision, "comparison": comparison}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
