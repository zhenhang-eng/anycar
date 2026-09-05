#!/usr/bin/env python3
"""Run a paired current-Actor Query AC exploration-radius A/B."""

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
from run_query_continuous_ac_candidate_bank_ab import (  # noqa: E402
    bank_report, make_actor_adapter, make_critic_adapter, response_bank65, rewrite_roles,
)


DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_continuous_ac_radius_ab_config_20260904_v1.json"


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
    source_summary_path = source_root / "summary.json"
    source_validation_path = source_root / "validation.json"
    if base.sha256(source_summary_path) != config["sources"]["actor_oac_summary_sha256"]:
        raise AssertionError("source summary hash mismatch")
    if base.sha256(source_validation_path) != config["sources"]["actor_oac_validation_sha256"]:
        raise AssertionError("source validation hash mismatch")
    if json.loads(source_validation_path.read_text())["qualification"] != config["sources"]["actor_oac_qualification"]:
        raise AssertionError("source longrun Actor did not independently pass")
    source_summary = json.loads(source_summary_path.read_text())
    seed = int(config["pilot"]["seeds"][0])
    source_record = source_summary["records"][str(seed)]
    if int(source_record["selected_round"]) != int(config["pilot"]["source_selected_round"]):
        raise AssertionError("source selected round changed")
    source_checkpoint = Path(source_record["checkpoint"])
    if base.sha256(source_checkpoint) != source_record["checkpoint_sha256"]:
        raise AssertionError("source checkpoint hash mismatch")

    loader = {"outputs": {"absolute_replay": config["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader)
    replay_validation = json.loads((Path(config["sources"]["absolute_replay"]) / "validation.json").read_text())
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
    critic_adapter = make_critic_adapter(source_checkpoint, output / "source_critic_adapter", seed)
    actor_adapter = make_actor_adapter(source_checkpoint, output / "source_actor_adapter", seed)
    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query), TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    base.direct_cost = batched_direct_cost
    base.response_bank = response_bank65
    bases = base.basis_bank()
    records = {}
    for arm, radius in config["pilot"]["radius_arms"].items():
        arm_config = copy.deepcopy(config)
        arm_config["pilot"]["probe_radius_sigma_start"] = float(radius["start"])
        arm_config["pilot"]["probe_radius_sigma_end"] = float(radius["end"])
        arm_config["sources"]["critic_pretrain"] = str(critic_adapter.parents[1])
        arm_config["sources"]["actor_oac"] = str(actor_adapter.parents[1])
        full_radii = np.linspace(
            float(radius["start"]), float(radius["end"]),
            int(config["pilot"]["probe_radius_sigma_anneal_rounds"]),
        ).astype(np.float32)
        arm_output = output / arm
        arm_output.mkdir()
        record = base.run_seed(
            seed, arm_config, data, replay_manifest, controller, fit, selection, outer,
            full_radii[: int(config["pilot"]["rounds"])], bases, arm_output, device,
        )
        rewrite_roles(record, "response39_recenter26", 65)
        record["bank"] = bank_report(record, 65)
        records[arm] = record

    control = records["control_020_to_005"]
    broad = records["broad_100_to_010"]
    control_mean = float(control["selected"]["inner"]["actor_cost"]["mean"])
    broad_mean = float(broad["selected"]["inner"]["actor_cost"]["mean"])
    control_aggregate = float(control["selected"]["inner"]["aggregate_improvement"])
    broad_aggregate = float(broad["selected"]["inner"]["aggregate_improvement"])
    checks = {
        "broad_selected_mean_lower": broad_mean < control_mean,
        "broad_warm_aggregate_higher": broad_aggregate > control_aggregate,
    }
    decision = (
        "PROMOTE_BROAD_RADIUS_TO_THREE_SEED_CONFIRMATION"
        if all(checks.values()) else "RETAIN_CURRENT_RADIUS_SCHEDULE"
    )
    comparison = {
        "control_selected_mean": control_mean,
        "broad_selected_mean": broad_mean,
        "broad_mean_reduction": control_mean - broad_mean,
        "control_warm_aggregate": control_aggregate,
        "broad_warm_aggregate": broad_aggregate,
        "broad_aggregate_delta": broad_aggregate - control_aggregate,
        "control_selected_round": int(control["selected_round"]),
        "broad_selected_round": int(broad["selected_round"]),
        "checks": checks,
    }
    summary = {
        "qualification": "QUERY_CONTINUOUS_AC_RADIUS_AB_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": config,
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": base.sha256(source_checkpoint),
        "source_actor_adapter": str(actor_adapter.resolve()),
        "source_actor_adapter_sha256": base.sha256(actor_adapter),
        "source_adapter": str(critic_adapter.resolve()),
        "source_adapter_sha256": base.sha256(critic_adapter),
        "records": records,
        "comparison": comparison,
        "decision": decision,
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "query-continuous-ac-radius-ab-v1",
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path), "config_sha256": base.sha256(config_path),
        "script": str(Path(__file__).resolve()), "script_sha256": base.sha256(Path(__file__).resolve()),
        "source_actor_summary_sha256": base.sha256(source_summary_path),
        "source_actor_validation_sha256": base.sha256(source_validation_path),
        "source_checkpoint_sha256": base.sha256(source_checkpoint),
        "source_replay_sha256": replay_manifest["replay_sha256"],
        "query_checkpoint_sha256": replay_manifest["query_checkpoint_sha256"],
        "summary_sha256": base.sha256(output / "summary.json"),
        "result_arrays_sha256": {arm: record["arrays_sha256"] for arm, record in records.items()},
        "result_checkpoint_sha256": {arm: record["checkpoint_sha256"] for arm, record in records.items()},
        "outer_fold_evaluated": False, "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [], "query_analytic_gradient_consumed": False,
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "decision": decision, **comparison}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
