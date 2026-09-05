#!/usr/bin/env python3
"""Extend the promoted equal-cell Query AC treatment from 40 to 160 rounds."""

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
import run_query_continuous_ac_equal_cell_ab as equal  # noqa: E402
import run_query_target_coverage_mixed_init_oac as base  # noqa: E402
from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from query_batched_direct_cost import batched_direct_cost  # noqa: E402
from run_query_continuous_ac_coarse_to_fine77_ab import bank_report, response_bank77, rewrite_roles  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_continuous_ac_equal_cell_longrun160_config_20260904_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prefix_errors(pilot: dict, longrun: dict, prefix_rounds: int) -> dict[str, float]:
    group_count, candidate_count = prefix_rounds * 20, prefix_rounds * 20 * 77
    with np.load(pilot["arrays"], allow_pickle=False) as left, np.load(longrun["arrays"], allow_pickle=False) as right:
        pairs = {
            "actor_batch_schedule": (left["actor_batch_schedule"], right["actor_batch_schedule"][:prefix_rounds]),
            "selection_round_action": (left["selection_round_action"], right["selection_round_action"][:prefix_rounds + 1]),
            "selection_round_cost": (left["selection_round_cost"], right["selection_round_cost"][:prefix_rounds + 1]),
            "online_state_index": (left["online_state_index"], right["online_state_index"][:candidate_count]),
            "online_action": (left["online_action"], right["online_action"][:candidate_count]),
            "online_cost": (left["online_cost"], right["online_cost"][:candidate_count]),
            "online_raw_action": (left["online_raw_action"], right["online_raw_action"][:candidate_count]),
            "online_clipped": (left["online_clipped"], right["online_clipped"][:candidate_count]),
            "online_round": (left["online_round"], right["online_round"][:candidate_count]),
            "online_group": (left["online_group"], right["online_group"][:candidate_count]),
            "online_role": (left["online_role"], right["online_role"][:candidate_count]),
        }
        if len(np.unique(right["online_group"][:candidate_count])) != group_count:
            raise AssertionError("longrun prefix group count changed")
        return {name: float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))
                if np.asarray(a).dtype.kind not in "OUSb" else float(not np.array_equal(a, b))
                for name, (a, b) in pairs.items()}


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if any((config["formal_validation_or_test_consumed"], config["dbm_fields_or_labels_consumed"],
            config["query_analytic_gradient_consumed"])):
        raise AssertionError("sealed-boundary contract violated")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    pilot_root = Path(config["sources"]["pilot40_root"])
    pilot_summary_path, pilot_validation_path = pilot_root / "summary.json", pilot_root / "validation.json"
    if base.sha256(pilot_summary_path) != config["sources"]["pilot40_summary_sha256"] or base.sha256(pilot_validation_path) != config["sources"]["pilot40_validation_sha256"]:
        raise AssertionError("pilot40 source hash changed")
    if json.loads(pilot_validation_path.read_text())["qualification"] != config["sources"]["pilot40_qualification"]:
        raise AssertionError("pilot40 source qualification changed")
    pilot = json.loads(pilot_summary_path.read_text())
    run_config = copy.deepcopy(pilot["contract"])
    run_config["output"] = str(output)
    run_config["pilot"]["rounds"] = int(config["rounds"])
    if int(config["prefix_rounds"]) != int(pilot["contract"]["pilot"]["rounds"]):
        raise AssertionError("prefix contract changed")

    replay_root = Path(run_config["sources"]["absolute_replay"])
    data, replay_manifest, collection_manifest = pretrain.load_data({"outputs": {"absolute_replay": str(replay_root)}})
    if json.loads((replay_root / "validation.json").read_text())["qualification"] != "QUERY_TARGET_COVERAGE_ABSOLUTE_REPLAY_INDEPENDENT_PASS":
        raise AssertionError("source Replay did not independently pass")
    fit = np.flatnonzero(np.isin(data["fold_id"], run_config["split_contract"]["fit_folds"]))
    selection = np.flatnonzero(data["fold_id"] == run_config["split_contract"]["inner_selection_fold"])
    outer = np.flatnonzero(data["fold_id"] == run_config["split_contract"]["outer_fold"])
    if (len(fit), len(selection), len(outer)) != (432, 120, 120):
        raise AssertionError("split sizes changed")
    episode_sets = [set(data["episode_id"][rows].tolist()) for rows in (fit, selection, outer)]
    if any(episode_sets[a] & episode_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("episode leakage")
    equal.ACTOR_ROW_WEIGHT, cell_id, cell_contract = equal.derive_cell_contract(data, fit)

    output.mkdir(parents=True)
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query), TorchMPPIParams(**collection_manifest["collection"]["mppi"]), device=str(device)
    )
    base.direct_cost = batched_direct_cost
    base.response_bank = response_bank77
    base.mixed_actor_microsteps = equal.equal_cell_actor_microsteps
    anneal = int(run_config["pilot"]["probe_radius_sigma_anneal_rounds"])
    rounds = int(run_config["pilot"]["rounds"])
    radii = np.concatenate((
        np.linspace(float(run_config["pilot"]["coarse_radius_start"]), float(run_config["pilot"]["coarse_radius_end"]), anneal),
        np.full(rounds - anneal, float(run_config["pilot"]["coarse_radius_end"])),
    )).astype(np.float32)
    bases = base.basis_bank()
    records, prefixes = {}, {}
    for seed_value in run_config["pilot"]["seeds"]:
        seed, key = int(seed_value), str(seed_value)
        adapter = pilot["adapters"][key]
        if base.sha256(Path(adapter["actor"])) != adapter["actor_sha256"] or base.sha256(Path(adapter["critic"])) != adapter["critic_sha256"]:
            raise AssertionError(f"adapter hash changed for seed {key}")
        seed_config = copy.deepcopy(run_config)
        seed_config["sources"]["actor_oac"] = str(Path(adapter["actor"]).parents[1])
        seed_config["sources"]["critic_pretrain"] = str(Path(adapter["critic"]).parents[1])
        seed_output = output / "equal_20_cell_longrun160" / f"seed_{seed}_run"
        seed_output.mkdir(parents=True)
        record = base.run_seed(
            seed, seed_config, data, replay_manifest, controller, fit, selection, outer,
            radii, bases, seed_output, device,
        )
        rewrite_roles(record, "coarse_to_fine77", 77)
        equal.attach_cell_arrays(record, equal.ACTOR_ROW_WEIGHT, cell_id)
        record["bank"] = bank_report(record, 77)
        record["actor_cell_weighting"] = cell_contract
        prefixes[key] = prefix_errors(pilot["records"]["equal_20_cell"][key], record, int(config["prefix_rounds"]))
        if any(value != 0.0 for value in prefixes[key].values()):
            raise AssertionError(f"seed {key} first-40 prefix changed: {prefixes[key]}")
        records[key] = record

    pooled = base.pooled([records[str(seed)] for seed in run_config["pilot"]["seeds"]], "selected", "inner", data)
    baseline = pilot["pooled"]["equal_20_cell"]
    per_seed, improved = {}, 0
    for seed_value in run_config["pilot"]["seeds"]:
        key = str(seed_value)
        baseline_mean = float(pilot["records"]["equal_20_cell"][key]["selected"]["inner"]["actor_cost"]["mean"])
        longrun_mean = float(records[key]["selected"]["inner"]["actor_cost"]["mean"])
        lower = longrun_mean < baseline_mean
        improved += int(lower)
        per_seed[key] = {
            "pilot40_mean": baseline_mean, "longrun160_mean": longrun_mean,
            "longrun_mean_reduction": baseline_mean - longrun_mean, "longrun_lower": lower,
            "pilot40_selected_round": int(pilot["records"]["equal_20_cell"][key]["selected_round"]),
            "longrun160_selected_round": int(records[key]["selected_round"]),
        }
    baseline_mean = float(baseline["actor_cost"]["mean"])
    longrun_mean = float(pooled["actor_cost"]["mean"])
    baseline_aggregate = float(baseline["aggregate_improvement"])
    longrun_aggregate = float(pooled["aggregate_improvement"])
    checks = {
        "longrun_pooled_mean_lower": longrun_mean < baseline_mean,
        "longrun_pooled_warm_aggregate_higher": longrun_aggregate > baseline_aggregate,
        "at_least_two_of_three_seed_means_lower": improved >= 2,
    }
    decision = "PROMOTE_EQUAL_CELL_LONGRUN160" if all(checks.values()) else "RETAIN_EQUAL_CELL_PILOT40"
    comparison = {
        "pilot40_pooled_mean": baseline_mean, "longrun160_pooled_mean": longrun_mean,
        "longrun_pooled_mean_reduction": baseline_mean - longrun_mean,
        "pilot40_warm_aggregate": baseline_aggregate, "longrun160_warm_aggregate": longrun_aggregate,
        "longrun_aggregate_delta": longrun_aggregate - baseline_aggregate,
        "improved_seed_count": improved, "per_seed": per_seed, "checks": checks,
    }
    summary = {
        "qualification": "QUERY_CONTINUOUS_AC_EQUAL_CELL_LONGRUN160_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(), "contract": config,
        "inherited_training_contract": run_config, "records": records, "adapters": pilot["adapters"],
        "cell_balance_contract": cell_contract, "prefix_errors": prefixes,
        "pilot40_pooled": baseline, "pooled": pooled, "comparison": comparison, "decision": decision,
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "query-continuous-ac-equal-cell-longrun160-v1",
        "qualification": "PENDING_INDEPENDENT_VALIDATION", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path), "config_sha256": base.sha256(config_path),
        "script": str(Path(__file__).resolve()), "script_sha256": base.sha256(Path(__file__).resolve()),
        "equal_cell_runner": str(Path(equal.__file__).resolve()), "equal_cell_runner_sha256": base.sha256(Path(equal.__file__).resolve()),
        "pilot40_summary_sha256": base.sha256(pilot_summary_path),
        "pilot40_validation_sha256": base.sha256(pilot_validation_path),
        "source_replay_sha256": replay_manifest["replay_sha256"],
        "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
        "summary_sha256": base.sha256(output / "summary.json"),
        "arrays_sha256": {seed: record["arrays_sha256"] for seed, record in records.items()},
        "checkpoint_sha256": {seed: record["checkpoint_sha256"] for seed, record in records.items()},
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "decision": decision, **comparison}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
