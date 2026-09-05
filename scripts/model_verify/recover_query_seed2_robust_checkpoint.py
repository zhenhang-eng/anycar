#!/usr/bin/env python3
"""Replay seed-2 OAC and recover a risk-selected intermediate checkpoint."""

from __future__ import annotations

import argparse
import copy
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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "scripts/model_verify/query_seed2_robust_checkpoint_recovery_config_20260903_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def select_round(
    costs: np.ndarray,
    warm: np.ndarray,
    target_mask: np.ndarray,
    tolerance: float,
) -> tuple[int, dict[str, float]]:
    means = costs.mean(axis=1)
    eligible = np.flatnonzero(means <= float(means.min()) + tolerance)
    worst_gain = (warm[None, target_mask] - costs[:, target_mask]).min(axis=1)
    selected = int(eligible[np.argmax(worst_gain[eligible])])
    return selected, {
        "minimum_inner_mean": float(means.min()),
        "selected_inner_mean": float(means[selected]),
        "selected_target_worst_gain": float(worst_gain[selected]),
        "eligible_round_count": int(len(eligible)),
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if any((
        config["formal_validation_or_test_consumed"],
        bool(config["dbm_fields_or_labels_consumed"]),
        config["query_analytic_gradient_consumed"],
    )):
        raise AssertionError("sealed-boundary contract violated")
    source_root = Path(config["source_run"]).resolve()
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    source_summary = json.loads((source_root / "summary.json").read_text())
    source_manifest = json.loads((source_root / "manifest.json").read_text())
    algorithm = copy.deepcopy(source_summary["contract"])
    seed = int(config["seed"])
    algorithm["pilot"]["seeds"] = [seed]
    algorithm["output"] = str(output)

    loader_config = {"outputs": {"absolute_replay": algorithm["sources"]["absolute_replay"]}}
    data, replay_manifest, collection_manifest = pretrain.load_data(loader_config)
    fit = np.flatnonzero(np.isin(data["fold_id"], algorithm["split_contract"]["fit_folds"]))
    inner = np.flatnonzero(data["fold_id"] == algorithm["split_contract"]["inner_selection_fold"])
    outer = np.flatnonzero(data["fold_id"] == algorithm["split_contract"]["outer_fold"])
    source_record = next(value for value in source_summary["records"] if int(value["seed"]) == seed)
    with np.load(source_record["arrays"], allow_pickle=False) as archive:
        source_arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if not np.array_equal(source_arrays["selection_indices"], inner):
        raise AssertionError("source inner indices changed")
    rule = config["selection_rule"]
    target_mask = (
        (data["speed_kph"][inner] == int(rule["target_speed_kph"]))
        & (data["variant_index"][inner] == int(rule["target_variant_index"]))
    )
    target_round, selection_report = select_round(
        source_arrays["selection_round_cost"], data["warm_cost"][inner],
        target_mask, float(rule["mean_cost_tolerance_above_minimum"]),
    )

    device = torch.device(args.device)
    query = QueryDeploymentModel.from_checkpoint(Path(replay_manifest["query_checkpoint"]), device)
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query),
        TorchMPPIParams(**collection_manifest["collection"]["mppi"]),
        device=str(device),
    )
    rounds = int(algorithm["pilot"]["rounds"])
    anneal = int(algorithm["pilot"]["probe_radius_sigma_anneal_rounds"])
    radii = np.concatenate((
        np.linspace(
            float(algorithm["pilot"]["probe_radius_sigma_start"]),
            float(algorithm["pilot"]["probe_radius_sigma_end"]), anneal,
        ),
        np.full(rounds - anneal, float(algorithm["pilot"]["probe_radius_sigma_end"])),
    )).astype(np.float32)

    captured: dict[str, object] = {}
    live_critics: list[torch.nn.Module] = []
    selection_call = -1
    original_evaluate = base.evaluate_actor
    original_update = base.update_critics

    def capture_update(critics, *update_args, **update_kwargs):
        live_critics[:] = critics
        return original_update(critics, *update_args, **update_kwargs)

    def capture_evaluate(actor, actor_inputs, rows, *evaluate_args, **evaluate_kwargs):
        nonlocal selection_call
        result = original_evaluate(actor, actor_inputs, rows, *evaluate_args, **evaluate_kwargs)
        if np.array_equal(rows, inner):
            selection_call += 1
            if selection_call == target_round:
                captured["actor_state_dict"] = {
                    name: value.detach().cpu().clone() for name, value in actor.state_dict().items()
                }
                captured["critic_state_dicts"] = [
                    {name: value.detach().cpu().clone() for name, value in critic.state_dict().items()}
                    for critic in live_critics
                ]
                captured["action"] = result[0].copy()
                captured["cost"] = result[1].copy()
        return result

    base.update_critics = capture_update
    base.evaluate_actor = capture_evaluate
    output.mkdir(parents=True)
    try:
        replay_record = base.run_seed(
            seed, algorithm, data, replay_manifest, controller, fit, inner,
            outer, radii, base.basis_bank(), output, device,
        )
    finally:
        base.evaluate_actor = original_evaluate
        base.update_critics = original_update
    if "actor_state_dict" not in captured or len(captured["critic_state_dicts"]) != 2:
        raise AssertionError("target checkpoint was not captured")

    action_error = float(np.max(np.abs(captured["action"] - source_arrays["selection_round_action"][target_round])))
    cost_error = float(np.max(np.abs(captured["cost"] - source_arrays["selection_round_cost"][target_round])))
    mean_cost_error = abs(float(np.mean(captured["cost"])) - float(np.mean(source_arrays["selection_round_cost"][target_round])))
    if action_error > 1e-7 or cost_error > 0.01 or mean_cost_error > 1e-5:
        raise AssertionError(
            f"recovery diverged from source curve: action={action_error} cost={cost_error} mean={mean_cost_error}"
        )

    source_actor = torch.load(
        Path(algorithm["sources"]["actor_oac"]) / f"seed_{seed}" / "checkpoint.pt",
        map_location="cpu", weights_only=False,
    )
    source_critic = torch.load(
        Path(algorithm["sources"]["critic_pretrain"]) / "checkpoints" / f"seed_{seed}.pt",
        map_location="cpu", weights_only=False,
    )
    robust_checkpoint = output / "robust_checkpoint.pt"
    torch.save({
        "qualification": "QUERY_SEED2_ROBUST_CHECKPOINT_RECOVERED_TRAIN_SIDE",
        "seed": seed,
        "selected_round": target_round,
        "selection_rule": rule,
        "selection_report": selection_report,
        "actor_training": source_actor["actor_training"],
        "actor_normalization": source_actor["normalization"],
        "critic_normalization": source_critic["normalization"],
        "critic1_training": source_critic["critic1_training"],
        "critic2_training": source_critic["critic2_training"],
        "selected_actor_state_dict": captured["actor_state_dict"],
        "selected_critic1_state_dict": captured["critic_state_dicts"][0],
        "selected_critic2_state_dict": captured["critic_state_dicts"][1],
        "fit_indices": fit,
        "selection_indices": inner,
        "outer_indices_unevaluated": outer,
        "source_run": str(source_root),
        "source_run_manifest_sha256": base.sha256(source_root / "manifest.json"),
        "source_arrays_sha256": base.sha256(Path(source_record["arrays"])),
        "source_replay_sha256": replay_manifest["replay_sha256"],
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }, robust_checkpoint)
    np.savez_compressed(
        output / "robust_recovery_arrays.npz",
        source_inner_action=source_arrays["selection_round_action"][target_round],
        source_inner_cost=source_arrays["selection_round_cost"][target_round],
        recovered_inner_action=captured["action"],
        recovered_inner_cost=captured["cost"],
        target_mask=target_mask,
    )
    summary = {
        "qualification": "QUERY_SEED2_ROBUST_CHECKPOINT_RECOVERY_PENDING_INDEPENDENT_VALIDATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_round": target_round,
        "selection_report": selection_report,
        "source_mean_selected_round": int(source_record["selected_round"]),
        "recovery_errors": {
            "action_max_abs": action_error,
            "cost_max_abs": cost_error,
            "cost_mean_abs": mean_cost_error,
        },
        "replay_record": replay_record,
        "robust_checkpoint": str(robust_checkpoint),
        "robust_checkpoint_sha256": base.sha256(robust_checkpoint),
        "arrays": str(output / "robust_recovery_arrays.npz"),
        "arrays_sha256": base.sha256(output / "robust_recovery_arrays.npz"),
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    base.dump_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "query-seed2-robust-checkpoint-recovery-v1",
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": base.sha256(config_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": base.sha256(Path(__file__).resolve()),
        "shared_training_implementation": str(Path(base.__file__).resolve()),
        "shared_training_implementation_sha256": base.sha256(Path(base.__file__).resolve()),
        "query_evaluation": "historical sequential direct-cost path",
        "source_run": str(source_root),
        "source_manifest_sha256": base.sha256(source_root / "manifest.json"),
        "source_summary_sha256": base.sha256(source_root / "summary.json"),
        "source_arrays_sha256": base.sha256(Path(source_record["arrays"])),
        "summary_sha256": base.sha256(output / "summary.json"),
        "robust_checkpoint_sha256": summary["robust_checkpoint_sha256"],
        "arrays_sha256": summary["arrays_sha256"],
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    base.dump_json(output / "manifest.json", manifest)
    print(json.dumps({
        "target_round": target_round,
        "selection_report": selection_report,
        "recovery_errors": summary["recovery_errors"],
        "robust_checkpoint": str(robust_checkpoint),
        "outer_fold_evaluated": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
