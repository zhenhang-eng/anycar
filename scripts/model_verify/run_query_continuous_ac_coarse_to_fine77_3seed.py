#!/usr/bin/env python3
"""Confirm recenter65 versus coarse-to-fine77 Query AC over three seeds."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

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
from run_query_continuous_ac_candidate_bank_ab import make_actor_adapter, make_critic_adapter, response_bank65  # noqa: E402
from run_query_continuous_ac_coarse_to_fine77_ab import bank_report, response_bank77, rewrite_roles  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_continuous_ac_coarse_to_fine77_3seed_config_20260904_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if (config["formal_validation_or_test_consumed"] or config["dbm_fields_or_labels_consumed"]
            or config["query_analytic_gradient_consumed"]):
        raise AssertionError("sealed-boundary contract violated")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    source_root = Path(config["sources"]["actor_oac_root"])
    source_summary_path, source_validation_path = source_root / "summary.json", source_root / "validation.json"
    if base.sha256(source_summary_path) != config["sources"]["actor_oac_summary_sha256"]:
        raise AssertionError("source summary hash mismatch")
    if base.sha256(source_validation_path) != config["sources"]["actor_oac_validation_sha256"]:
        raise AssertionError("source validation hash mismatch")
    if json.loads(source_validation_path.read_text())["qualification"] != config["sources"]["actor_oac_qualification"]:
        raise AssertionError("source longrun qualification changed")
    source_summary = json.loads(source_summary_path.read_text())
    single_root = Path(config["sources"]["validated_single_seed"])
    single_summary_path, single_validation_path = single_root / "summary.json", single_root / "validation.json"
    if base.sha256(single_summary_path) != config["sources"]["validated_single_seed_summary_sha256"]:
        raise AssertionError("single-seed summary hash mismatch")
    if base.sha256(single_validation_path) != config["sources"]["validated_single_seed_validation_sha256"]:
        raise AssertionError("single-seed validation hash mismatch")
    if json.loads(single_validation_path.read_text())["qualification"] != config["sources"]["validated_single_seed_qualification"]:
        raise AssertionError("single-seed source did not independently pass")
    single_summary = json.loads(single_summary_path.read_text())
    for record in single_summary["records"].values():
        if base.sha256(Path(record["arrays"])) != record["arrays_sha256"] or base.sha256(Path(record["checkpoint"])) != record["checkpoint_sha256"]:
            raise AssertionError("reused seed0 artifact hash mismatch")

    loader = {"outputs": {"absolute_replay": config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader)
    fit = np.flatnonzero(np.isin(data["fold_id"], config["split_contract"]["fit_folds"]))
    selection = np.flatnonzero(data["fold_id"] == config["split_contract"]["inner_selection_fold"])
    outer = np.flatnonzero(data["fold_id"] == config["split_contract"]["outer_fold"])
    if (len(fit), len(selection), len(outer)) != (432, 120, 120):
        raise AssertionError("split sizes changed")
    episode_sets = [set(data["episode_id"][rows].tolist()) for rows in (fit, selection, outer)]
    if any(episode_sets[a] & episode_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("episode leakage")

    output.mkdir(parents=True)
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query), TorchMPPIParams(**collection_manifest["collection"]["mppi"]), device=str(device)
    )
    base.direct_cost = batched_direct_cost
    bases = base.basis_bank()
    records = {arm: {"0": single_summary["records"][arm]} for arm in config["pilot"]["candidate_arms"]}
    adapters = {"0": {
        "source_checkpoint": single_summary["source_checkpoint"],
        "source_checkpoint_sha256": single_summary["source_checkpoint_sha256"],
        "actor": single_summary["source_actor_adapter"],
        "actor_sha256": single_summary["source_actor_adapter_sha256"],
        "critic": single_summary["source_adapter"],
        "critic_sha256": single_summary["source_adapter_sha256"],
        "reused_from_validated_single_seed": True,
    }}
    for seed in config["pilot"]["new_seeds"]:
        seed = int(seed)
        source_record = source_summary["records"][str(seed)]
        if int(source_record["selected_round"]) != int(config["pilot"]["source_selected_round_by_seed"][str(seed)]):
            raise AssertionError(f"source selected round changed for seed {seed}")
        source_checkpoint = Path(source_record["checkpoint"])
        if base.sha256(source_checkpoint) != source_record["checkpoint_sha256"]:
            raise AssertionError(f"source checkpoint changed for seed {seed}")
        critic_adapter = make_critic_adapter(source_checkpoint, output / f"source_critic_adapter_seed{seed}", seed)
        actor_adapter = make_actor_adapter(source_checkpoint, output / f"source_actor_adapter_seed{seed}", seed)
        adapters[str(seed)] = {
            "source_checkpoint": str(source_checkpoint.resolve()), "source_checkpoint_sha256": base.sha256(source_checkpoint),
            "actor": str(actor_adapter.resolve()), "actor_sha256": base.sha256(actor_adapter),
            "critic": str(critic_adapter.resolve()), "critic_sha256": base.sha256(critic_adapter),
            "reused_from_validated_single_seed": False,
        }
        for arm, candidate_count in config["pilot"]["candidate_arms"].items():
            arm_config = copy.deepcopy(config)
            arm_config["pilot"]["candidates_per_visit"] = int(candidate_count)
            arm_config["sources"]["critic_pretrain"] = str(critic_adapter.parents[1])
            arm_config["sources"]["actor_oac"] = str(actor_adapter.parents[1])
            if arm == "control_recenter65":
                start, end, function = float(config["pilot"]["control_radius_start"]), float(config["pilot"]["control_radius_end"]), response_bank65
            else:
                start, end, function = float(config["pilot"]["coarse_radius_start"]), float(config["pilot"]["coarse_radius_end"]), response_bank77
            arm_config["pilot"]["probe_radius_sigma_start"] = start
            arm_config["pilot"]["probe_radius_sigma_end"] = end
            radii = np.linspace(start, end, int(config["pilot"]["probe_radius_sigma_anneal_rounds"])).astype(np.float32)
            base.response_bank = function
            arm_output = output / arm / f"seed_{seed}_run"
            arm_output.mkdir(parents=True)
            record = base.run_seed(
                seed, arm_config, data, replay_manifest, controller, fit, selection, outer,
                radii[: int(config["pilot"]["rounds"])], bases, arm_output, device,
            )
            rewrite_roles(record, arm, int(candidate_count))
            record["bank"] = bank_report(record, int(candidate_count))
            records[arm][str(seed)] = record

    pooled = {
        arm: base.pooled([records[arm][str(seed)] for seed in config["pilot"]["seeds"]], "selected", "inner", data)
        for arm in records
    }
    per_seed = {}
    improved = 0
    for seed in config["pilot"]["seeds"]:
        seed = str(seed)
        control_mean = float(records["control_recenter65"][seed]["selected"]["inner"]["actor_cost"]["mean"])
        treatment_mean = float(records["coarse_to_fine77"][seed]["selected"]["inner"]["actor_cost"]["mean"])
        is_improved = treatment_mean < control_mean
        improved += int(is_improved)
        per_seed[seed] = {
            "control_mean": control_mean, "treatment_mean": treatment_mean,
            "treatment_mean_reduction": control_mean - treatment_mean,
            "treatment_lower": is_improved,
            "control_selected_round": int(records["control_recenter65"][seed]["selected_round"]),
            "treatment_selected_round": int(records["coarse_to_fine77"][seed]["selected_round"]),
        }
    control_mean = float(pooled["control_recenter65"]["actor_cost"]["mean"])
    treatment_mean = float(pooled["coarse_to_fine77"]["actor_cost"]["mean"])
    control_aggregate = float(pooled["control_recenter65"]["aggregate_improvement"])
    treatment_aggregate = float(pooled["coarse_to_fine77"]["aggregate_improvement"])
    checks = {
        "treatment_pooled_mean_lower": treatment_mean < control_mean,
        "treatment_pooled_warm_aggregate_higher": treatment_aggregate > control_aggregate,
        "at_least_two_of_three_seed_means_lower": improved >= 2,
    }
    decision = "PROMOTE_COARSE_TO_FINE77" if all(checks.values()) else "RETAIN_RECENTER65_STOP_BANK_EXPANSION"
    comparison = {
        "control_pooled_mean": control_mean, "treatment_pooled_mean": treatment_mean,
        "treatment_pooled_mean_reduction": control_mean - treatment_mean,
        "control_warm_aggregate": control_aggregate, "treatment_warm_aggregate": treatment_aggregate,
        "treatment_aggregate_delta": treatment_aggregate - control_aggregate,
        "improved_seed_count": improved, "per_seed": per_seed, "checks": checks,
    }
    summary = {
        "qualification": "QUERY_CONTINUOUS_AC_COARSE_TO_FINE77_THREE_SEED_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(), "contract": config,
        "records": records, "adapters": adapters, "pooled": pooled,
        "comparison": comparison, "decision": decision,
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "query-continuous-ac-coarse-to-fine77-three-seed-v1",
        "qualification": "PENDING_INDEPENDENT_VALIDATION", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path), "config_sha256": base.sha256(config_path),
        "script": str(Path(__file__).resolve()), "script_sha256": base.sha256(Path(__file__).resolve()),
        "source_actor_summary_sha256": base.sha256(source_summary_path),
        "source_actor_validation_sha256": base.sha256(source_validation_path),
        "single_seed_summary_sha256": base.sha256(single_summary_path),
        "single_seed_validation_sha256": base.sha256(single_validation_path),
        "source_replay_sha256": replay_manifest["replay_sha256"],
        "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
        "summary_sha256": base.sha256(output / "summary.json"),
        "result_arrays_sha256": {arm: {seed: record["arrays_sha256"] for seed, record in values.items()} for arm, values in records.items()},
        "result_checkpoint_sha256": {arm: {seed: record["checkpoint_sha256"] for seed, record in values.items()} for arm, values in records.items()},
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "decision": decision, **comparison}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
