#!/usr/bin/env python3
"""Independently validate the train-only Query Actor nested OOF artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for package in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package))

from car_dynamics.controllers_torch.mppi import (  # noqa: E402
    TorchMPPIController,
    TorchMPPIParams,
)
from car_foundation.mppi_proposal_policy import (  # noqa: E402
    MPPIProposalNormalization,
    TorchMPPIProposalPolicy,
    ego_reference_features,
)
from car_foundation.query_deployment import (  # noqa: E402
    QueryDeploymentModel,
    TorchQueryRolloutBackend,
)
from build_query_expected_road_fullrank_sidecar import interpolate_knots  # noqa: E402


DEFAULT_ARTIFACT = REPO_ROOT / (
    "outputs/query_mppi/query_expected_road_actor_oof_20260901_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, nargs="?", default=DEFAULT_ARTIFACT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def maximum_dictionary_error(left: dict, right: dict) -> float:
    errors = []
    for name in MPPIProposalNormalization.__dataclass_fields__:
        errors.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(left[name], np.float64)
                        - np.asarray(right[name], np.float64)
                    )
                )
            )
        )
    return max(errors, default=0.0)


@torch.no_grad()
def predict(
    model: TorchMPPIProposalPolicy,
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    rows: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(rows), 256):
        local = rows[start : start + 256]
        _, center = model(
            torch.from_numpy(inputs[0][local]).to(device),
            torch.from_numpy(inputs[1][local]).to(device),
            torch.from_numpy(inputs[2][local]).to(device),
            torch.from_numpy(inputs[3][local]).to(device),
        )
        output.append(center.cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def query_costs(controller, data, rows, knots) -> np.ndarray:
    actions = interpolate_knots(knots)
    result = np.empty(len(rows), dtype=np.float64)
    for local, row in enumerate(rows):
        evaluated = controller.evaluate_action_sequences(
            data["state"][row],
            data["current_action"][row],
            data["history"][row : row + 1],
            data["reference"][row],
            actions[local : local + 1],
        )
        result[local] = float(evaluated["cost"][0].cpu())
    return result


def main() -> None:
    args = parse_args()
    artifact = args.artifact.resolve()
    manifest = json.loads((artifact / "manifest.json").read_text())
    summary = json.loads((artifact / "summary.json").read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    source = Path(config["source_sidecar"])
    source_manifest = json.loads((source / "manifest.json").read_text())
    source_validation = json.loads((source / "validation.json").read_text())

    hash_checks = {
        "config": sha256(config_path) == manifest["config_sha256"],
        "script": sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "source_manifest": sha256(source / "manifest.json")
        == manifest["source_manifest_sha256"],
        "source_validation": sha256(source / "validation.json")
        == manifest["source_validation_sha256"],
        "source_bank": sha256(source / "bank.npz")
        == manifest["source_bank_sha256"],
        "query_checkpoint": sha256(Path(source_manifest["query_checkpoint"]))
        == manifest["query_checkpoint_sha256"],
        "projected_teacher": sha256(artifact / "projected_teacher.npz")
        == manifest["projected_teacher_sha256"],
        "oof_predictions": sha256(artifact / "oof_predictions.npz")
        == manifest["oof_predictions_sha256"],
        "summary": sha256(artifact / "summary.json") == manifest["summary_sha256"],
    }
    if not all(hash_checks.values()):
        raise AssertionError(f"artifact hash failure: {hash_checks}")
    if source_validation["qualification"] != "QUERY_EXPECTED_ROAD_FULLRANK_PASS":
        raise AssertionError("source validation qualification changed")

    with np.load(source / "bank.npz", allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(artifact / "projected_teacher.npz", allow_pickle=False) as archive:
        projected = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(artifact / "oof_predictions.npz", allow_pickle=False) as archive:
        oof = {name: np.asarray(archive[name]) for name in archive.files}

    reference = np.stack(
        [
            ego_reference_features(value, float(state[3]))
            for value, state in zip(data["reference_ego"], data["state"])
        ]
    ).astype(np.float32)
    current = np.stack(
        [
            np.asarray((state[3], state[4], *action), dtype=np.float32)
            for state, action in zip(data["state"], data["current_action"])
        ]
    )
    history = data["history"].astype(np.float32)
    warm = data["mean_knots_before"].astype(np.float32)
    sigma = np.asarray(config["noise_sigma"], np.float32).reshape(1, 1, 2)
    expected_projected = np.clip(
        warm
        + np.clip(
            (data["fullrank_teacher_knots"] - warm) / sigma,
            -float(config["maximum_delta_sigma"]),
            float(config["maximum_delta_sigma"]),
        )
        * sigma,
        -1.0,
        1.0,
    )
    projection_error = float(
        np.max(np.abs(expected_projected - projected["projected_teacher_knots"]))
    )
    row_contract_error = float(
        max(
            np.max(np.abs(oof["row_index"] - data["row_index"])),
            np.max(np.abs(oof["fold_id"] - data["fold_id"])),
            0 if np.array_equal(oof["episode_id"], data["episode_id"]) else 1,
        )
    )

    device = torch.device(args.device)
    parent_manifest = json.loads(
        (Path(source_manifest["parent_t0"]) / "manifest.json").read_text()
    )
    collection_manifest = json.loads(
        (Path(parent_manifest["source_collection"]) / "manifest.json").read_text()
    )
    params = TorchMPPIParams(**collection_manifest["collection"]["mppi"])
    query_model = QueryDeploymentModel.from_checkpoint(
        Path(source_manifest["query_checkpoint"]), device
    )
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query_model), params, device=str(device)
    )

    seeds = [int(value) for value in config["seeds"]]
    seed_to_index = {seed: index for index, seed in enumerate(seeds)}
    prediction_error = 0.0
    normalization_error = 0.0
    max_delta_sigma = 0.0
    bound_violation = 0.0
    split_error = 0.0
    selection_error = 0.0
    formal_error = 0.0
    checkpoint_hashes = {}
    for record in summary["records"]:
        fold = int(record["fold"])
        seed = int(record["seed"])
        seed_index = seed_to_index[seed]
        checkpoint_path = Path(record["checkpoint"])
        checkpoint_hashes[checkpoint_path.name] = (
            sha256(checkpoint_path) == manifest["checkpoint_sha256"][checkpoint_path.name]
            == record["checkpoint_sha256"]
        )
        payload = torch.load(checkpoint_path, map_location="cpu")
        expected_oof = np.flatnonzero(data["fold_id"] == fold)
        selection_fold = (
            fold + int(config["inner_selection_fold_offset_by_seed"][seed_index])
        ) % int(config["folds"])
        expected_selection = np.flatnonzero(data["fold_id"] == selection_fold)
        expected_fit = np.flatnonzero(
            (data["fold_id"] != fold) & (data["fold_id"] != selection_fold)
        )
        for actual, expected in (
            (payload["fit_indices"], expected_fit),
            (payload["selection_indices"], expected_selection),
            (payload["oof_indices"], expected_oof),
        ):
            split_error = max(
                split_error,
                0.0 if np.array_equal(actual, expected) else 1.0,
            )
        episode_sets = [
            set(data["episode_id"][rows].tolist())
            for rows in (expected_fit, expected_selection, expected_oof)
        ]
        if episode_sets[0] & episode_sets[1] or episode_sets[0] & episode_sets[2] or episode_sets[1] & episode_sets[2]:
            split_error = 1.0

        expected_normalizer = MPPIProposalNormalization.fit(
            history[expected_fit], reference[expected_fit], current[expected_fit]
        )
        normalization_error = max(
            normalization_error,
            maximum_dictionary_error(
                payload["normalization"], expected_normalizer.to_dict()
            ),
        )
        normalizer = MPPIProposalNormalization.from_dict(payload["normalization"])
        norm_history, norm_reference, norm_current = normalizer.normalize_numpy(
            history, reference, current
        )
        trust = tuple(
            (
                float(config["maximum_delta_sigma"])
                * np.asarray(config["noise_sigma"], np.float32)
            ).tolist()
        )
        model = TorchMPPIProposalPolicy(
            trust_scale=trust, dropout=float(config["dropout"])
        ).to(device)
        model.load_state_dict(payload["actor_state_dict"], strict=True)
        predicted = predict(
            model,
            (
                norm_history.astype(np.float32),
                norm_reference.astype(np.float32),
                norm_current.astype(np.float32),
                warm,
            ),
            expected_oof,
            device,
        )
        stored = oof["actor_knots"][seed_index, expected_oof]
        prediction_error = max(
            prediction_error, float(np.max(np.abs(predicted - stored)))
        )
        normalized_delta = np.abs((predicted - warm[expected_oof]) / sigma)
        max_delta_sigma = max(max_delta_sigma, float(normalized_delta.max()))
        bound_violation = max(
            bound_violation,
            float(max(np.max(predicted - 1.0), np.max(-1.0 - predicted), 0.0)),
        )
        history_rows = payload["selection_history"]
        best = min(history_rows, key=lambda row: row["selection_direct_cost_mean"])
        selection_error = max(
            selection_error,
            abs(float(best["selection_direct_cost_mean"]) - float(payload["best_selection_direct_cost_mean"])),
            abs(int(best["epoch"]) - int(payload["best_epoch"])),
            abs(int(record["best_epoch"]) - int(payload["best_epoch"])),
        )
        formal_error = max(
            formal_error,
            float(bool(payload.get("formal_validation_or_test_consumed", True))),
        )

    if not all(checkpoint_hashes.values()):
        raise AssertionError("checkpoint hash failure")

    replay_cost = np.empty_like(oof["actor_direct_cost"], dtype=np.float64)
    for seed_index in range(len(seeds)):
        rows = np.arange(len(data["row_index"]), dtype=np.int64)
        replay_cost[seed_index] = query_costs(
            controller, data, rows, oof["actor_knots"][seed_index]
        )
        print(f"replayed OOF seed {seeds[seed_index]}", flush=True)
    cost_error = float(np.max(np.abs(replay_cost - oof["actor_direct_cost"])))
    cost_relative_error = float(
        np.max(
            np.abs(replay_cost - oof["actor_direct_cost"])
            / np.maximum(np.abs(oof["actor_direct_cost"]), 1.0)
        )
    )
    cap_rows = np.arange(15, len(data["row_index"]), 30, dtype=np.int64)
    cap_replay = query_costs(
        controller,
        data,
        cap_rows,
        projected["projected_teacher_knots"][cap_rows],
    )
    projected_cost_error = float(
        np.max(
            np.abs(
                cap_replay - projected["projected_teacher_direct_cost"][cap_rows]
            )
        )
    )

    warm_cost = data["warm_direct_cost_replayed"].astype(np.float64)
    fullrank_cost = data["fullrank_teacher_direct_cost"].astype(np.float64)
    recomputed_recoveries = []
    summary_metric_error = 0.0
    for seed_index, seed_report in enumerate(summary["seed_reports"]):
        recovery = float(
            np.sum(warm_cost - replay_cost[seed_index])
            / np.sum(warm_cost - fullrank_cost)
        )
        recomputed_recoveries.append(recovery)
        summary_metric_error = max(
            summary_metric_error,
            abs(recovery - float(seed_report["overall"]["fullrank_headroom_recovery"])),
            abs(
                float(np.mean(replay_cost[seed_index]))
                - float(seed_report["overall"]["actor_direct_cost"]["mean"])
            ),
        )

    checks = {
        "artifact_hashes": all(hash_checks.values()) and all(checkpoint_hashes.values()),
        "source_qualified_and_sealed": (
            source_validation["qualification"] == "QUERY_EXPECTED_ROAD_FULLRANK_PASS"
            and not source_manifest.get("formal_validation_or_test_consumed", True)
        ),
        "row_contract": row_contract_error == 0.0,
        "projected_teacher_reconstruction": projection_error <= 1e-7,
        "projected_teacher_query_shadow": projected_cost_error <= 1e-4,
        "episode_nested_split": split_error == 0.0,
        "fit_only_normalization": normalization_error <= 1e-7,
        "checkpoint_selection": selection_error == 0.0,
        "checkpoint_prediction_replay": prediction_error <= 1e-7,
        "bounded_actor_output": (
            max_delta_sigma <= float(config["maximum_delta_sigma"]) + 1e-5
            and bound_violation == 0.0
        ),
        "pytorch_query_oof_replay": cost_error <= 1e-4 and cost_relative_error <= 1e-5,
        "summary_metric_reconstruction": summary_metric_error <= 1e-10,
        "finite": all(
            np.isfinite(value).all()
            for value in (
                projected["projected_teacher_knots"],
                projected["projected_teacher_direct_cost"],
                oof["actor_knots"],
                oof["actor_direct_cost"],
                replay_cost,
            )
        ),
        "formal_validation_test_sealed": (
            formal_error == 0.0
            and not manifest.get("formal_validation_or_test_consumed", True)
            and not summary.get("formal_validation_or_test_consumed", True)
            and not config.get("formal_validation_or_test_consumed", True)
        ),
        "no_dbm_contract": (
            manifest.get("dbm_fields_or_labels_consumed") == []
            and source_manifest.get("dbm_fields_or_labels_consumed") == []
        ),
    }
    qualification = (
        "QUERY_ACTOR_OOF_INDEPENDENT_PASS" if all(checks.values())
        else "QUERY_ACTOR_OOF_INDEPENDENT_FAIL"
    )
    validation = {
        "qualification": qualification,
        "actor_qualification": summary["qualification"],
        "artifact": str(artifact),
        "checks": checks,
        "hash_checks": hash_checks,
        "checkpoint_hash_checks": checkpoint_hashes,
        "maximum_errors": {
            "row_contract": row_contract_error,
            "projected_teacher_reconstruction": projection_error,
            "projected_teacher_query_cost": projected_cost_error,
            "nested_split": split_error,
            "normalization": normalization_error,
            "checkpoint_selection": selection_error,
            "checkpoint_prediction": prediction_error,
            "actor_delta_sigma": max_delta_sigma,
            "actor_bound_violation": bound_violation,
            "pytorch_query_oof_cost": cost_error,
            "pytorch_query_oof_cost_relative": cost_relative_error,
            "summary_metrics": summary_metric_error,
            "formal": formal_error,
        },
        "recomputed_oof_fullrank_headroom_recovery": recomputed_recoveries,
        "projected_teacher_shadow_rows": cap_rows.tolist(),
        "formal_validation_or_test_consumed": False,
    }
    dump_json(artifact / "validation.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))
    if qualification.endswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
