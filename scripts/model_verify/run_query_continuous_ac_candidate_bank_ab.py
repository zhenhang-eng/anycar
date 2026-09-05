#!/usr/bin/env python3
"""Run paired continuous Query AC with 39 versus 65 online candidates."""

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
from run_query_forward_response_landscape_pilot import evaluate, qr_16  # noqa: E402
from run_query_single_center_oac20to1 import response_bank as response_bank39  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_continuous_ac_candidate_bank_ab_config_20260903_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def response_bank65(
    controller: TorchMPPIController,
    data: dict[str, np.ndarray],
    row: int,
    center: np.ndarray,
    radius: float,
    basis: np.ndarray,
    sigma: np.ndarray,
    weights: dict[str, float],
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actions, costs, raw, clipped = response_bank39(
        controller, data, row, center, radius, basis, sigma, weights, config
    )
    if len(actions) != 39:
        raise AssertionError("response39 prefix changed")
    incumbent = actions[int(np.argmin(costs))]
    directions = qr_16(260904)[:13].reshape(13, 8, 2).astype(np.float32)
    second_radius = radius * float(config["pilot"]["recenter_radius_ratio"])
    raw_second = np.stack([
        incumbent + sign * second_radius * direction * sigma
        for direction in directions for sign in (1.0, -1.0)
    ]).astype(np.float32)
    second = np.clip(raw_second, -1.0, 1.0).astype(np.float32)
    second_cost, _ = evaluate(controller, data, row, second, weights)
    return (
        np.concatenate((actions, second)).astype(np.float32),
        np.concatenate((costs, second_cost)).astype(np.float32),
        np.concatenate((raw, raw_second)).astype(np.float32),
        np.concatenate((clipped, np.any(np.abs(raw_second - second) > 1e-7, axis=(1, 2)))),
    )


def make_critic_adapter(source_checkpoint: Path, adapter_root: Path, seed: int) -> Path:
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_dir = adapter_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    target = checkpoint_dir / f"seed_{seed}.pt"
    torch.save({
        "qualification": "QUERY_CONTINUOUS_AC_CANDIDATE_AB_SOURCE_ADAPTER",
        "seed": seed,
        "normalization": payload["critic_normalization"],
        "critic1_training": payload["critic1_training"],
        "critic2_training": payload["critic2_training"],
        "critic1_state_dict": payload["selected_critic1_state_dict"],
        "critic2_state_dict": payload["selected_critic2_state_dict"],
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": base.sha256(source_checkpoint),
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }, target)
    return target


def make_actor_adapter(source_checkpoint: Path, adapter_root: Path, seed: int) -> Path:
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    target_dir = adapter_root / f"seed_{seed}"
    target_dir.mkdir(parents=True)
    target = target_dir / "checkpoint.pt"
    adapted = dict(payload)
    adapted["normalization"] = payload["actor_normalization"]
    adapted["qualification"] = "QUERY_CONTINUOUS_AC_CANDIDATE_AB_ACTOR_SOURCE_ADAPTER"
    adapted["adapted_source_checkpoint"] = str(source_checkpoint.resolve())
    adapted["adapted_source_checkpoint_sha256"] = base.sha256(source_checkpoint)
    torch.save(adapted, target)
    return target


def rewrite_roles(record: dict[str, Any], arm: str, candidate_count: int) -> None:
    path = Path(record["arrays"])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    group_count = len(np.unique(arrays["online_group"]))
    if len(arrays["online_cost"]) != group_count * candidate_count:
        raise AssertionError("online candidate count does not match fixed group width")
    local_roles = ["actor"] + ["probe"] * 32 + ["response"] * 6
    if arm == "response39_recenter26":
        local_roles += ["recenter"] * 26
    roles = np.tile(np.asarray(local_roles), group_count)
    if len(roles) != len(arrays["online_cost"]):
        raise AssertionError("role rewrite length mismatch")
    arrays["online_role"] = roles
    temporary = path.with_suffix(".rewrite.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    record["arrays_sha256"] = base.sha256(path)


def bank_report(record: dict[str, Any], candidate_count: int) -> dict[str, Any]:
    with np.load(record["arrays"], allow_pickle=False) as archive:
        costs = np.asarray(archive["online_cost"], np.float64).reshape(-1, candidate_count)
    gain = costs[:, 0] - costs.min(axis=1)
    return {
        "group_count": int(len(costs)),
        "candidate_count": candidate_count,
        "mean_actor_to_bank_best_gain": float(gain.mean()),
        "median_actor_to_bank_best_gain": float(np.median(gain)),
        "positive_gain_fraction": float(np.mean(gain > 1e-5)),
    }


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
    seed = int(config["pilot"]["seeds"][0])
    source_checkpoint = source / f"seed_{seed}" / "checkpoint.pt"
    output.mkdir(parents=True)
    adapter_path = make_critic_adapter(source_checkpoint, output / "source_adapter", seed)
    actor_adapter_path = make_actor_adapter(
        source_checkpoint, output / "source_actor_adapter", seed
    )

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
    if episode_sets[0] & episode_sets[1] or episode_sets[0] & episode_sets[2] or episode_sets[1] & episode_sets[2]:
        raise AssertionError("episode leakage")

    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    rounds = int(config["pilot"]["rounds"])
    anneal = int(config["pilot"]["probe_radius_sigma_anneal_rounds"])
    full_radii = np.linspace(
        float(config["pilot"]["probe_radius_sigma_start"]),
        float(config["pilot"]["probe_radius_sigma_end"]),
        anneal,
    ).astype(np.float32)
    radii = full_radii[:rounds]
    bases = base.basis_bank()
    base.direct_cost = batched_direct_cost
    records = {}
    for arm, candidate_count in config["pilot"]["candidate_arms"].items():
        arm_config = copy.deepcopy(config)
        arm_config["pilot"]["candidates_per_visit"] = int(candidate_count)
        arm_config["sources"]["critic_pretrain"] = str((output / "source_adapter").resolve())
        arm_config["sources"]["actor_oac"] = str((output / "source_actor_adapter").resolve())
        arm_output = output / arm
        arm_output.mkdir()
        base.response_bank = response_bank39 if arm == "response39" else response_bank65
        record = base.run_seed(
            seed, arm_config, data, replay_manifest, controller,
            fit, selection, outer, radii, bases, arm_output, device,
        )
        rewrite_roles(record, arm, int(candidate_count))
        record["bank"] = bank_report(record, int(candidate_count))
        records[arm] = record

    baseline = records["response39"]
    treatment = records["response39_recenter26"]
    baseline_mean = float(baseline["selected"]["inner"]["actor_cost"]["mean"])
    treatment_mean = float(treatment["selected"]["inner"]["actor_cost"]["mean"])
    baseline_aggregate = float(baseline["selected"]["inner"]["aggregate_improvement"])
    treatment_aggregate = float(treatment["selected"]["inner"]["aggregate_improvement"])
    common_round0_error = abs(
        float(baseline["round0"]["inner"]["actor_cost"]["mean"])
        - float(treatment["round0"]["inner"]["actor_cost"]["mean"])
    )
    decision_checks = {
        "common_round0_mean_exact": common_round0_error == 0.0,
        "treatment_selected_mean_lower": treatment_mean < baseline_mean,
        "treatment_warm_aggregate_higher": treatment_aggregate > baseline_aggregate,
    }
    decision = (
        "PROMOTE_RECENTER65_TO_THREE_SEED_CONTINUOUS_AC"
        if all(decision_checks.values())
        else "RETAIN_RESPONSE39_CONTINUOUS_AC"
    )
    comparison = {
        "common_round0_mean_abs_error": common_round0_error,
        "baseline_selected_mean": baseline_mean,
        "treatment_selected_mean": treatment_mean,
        "treatment_mean_reduction": baseline_mean - treatment_mean,
        "baseline_warm_aggregate": baseline_aggregate,
        "treatment_warm_aggregate": treatment_aggregate,
        "treatment_aggregate_delta": treatment_aggregate - baseline_aggregate,
        "baseline_selected_round": int(baseline["selected_round"]),
        "treatment_selected_round": int(treatment["selected_round"]),
    }
    summary = {
        "qualification": "QUERY_CONTINUOUS_AC_CANDIDATE_BANK_AB_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": config,
        "records": records,
        "comparison": comparison,
        "decision_checks": decision_checks,
        "decision": decision,
        "source_adapter": str(adapter_path.resolve()),
        "source_adapter_sha256": base.sha256(adapter_path),
        "source_actor_adapter": str(actor_adapter_path.resolve()),
        "source_actor_adapter_sha256": base.sha256(actor_adapter_path),
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "query-continuous-ac-candidate-bank-ab-v1",
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
        "source_checkpoint_sha256": base.sha256(source_checkpoint),
        "source_adapter_sha256": base.sha256(adapter_path),
        "source_actor_adapter_sha256": base.sha256(actor_adapter_path),
        "summary_sha256": base.sha256(output / "summary.json"),
        "result_arrays_sha256": {arm: record["arrays_sha256"] for arm, record in records.items()},
        "result_checkpoint_sha256": {arm: record["checkpoint_sha256"] for arm, record in records.items()},
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({
        "output": str(output),
        "decision": decision,
        "comparison": comparison,
        "bank": {arm: record["bank"] for arm, record in records.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
