#!/usr/bin/env python3
"""Independently validate Query absolute/no-anchor Actor + Twin Critic OOF."""

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

from car_dynamics.controllers_torch.mppi import TorchMPPIController, TorchMPPIParams  # noqa: E402
from car_foundation.mppi_proposal_policy import (  # noqa: E402
    MPPIProposalNormalization,
    ego_reference_features,
)
from car_foundation.query_deployment import QueryDeploymentModel, TorchQueryRolloutBackend  # noqa: E402
from build_query_expected_road_fullrank_sidecar import interpolate_knots  # noqa: E402
from mppi_a2_actors import DirectNoAnchorGTXActor  # noqa: E402
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic  # noqa: E402


DEFAULT_ARTIFACT = REPO_ROOT / (
    "outputs/query_mppi/query_expected_road_absolute_pretrain_20260901_v1"
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
    return max(
        float(
            np.max(
                np.abs(
                    np.asarray(left[name], np.float64)
                    - np.asarray(right[name], np.float64)
                )
            )
        )
        for name in MPPIProposalNormalization.__dataclass_fields__
    )


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float64).ravel()
    right = np.asarray(right, np.float64).ravel()
    if left.std() < 1e-12 or right.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def critic_metrics(predicted_log: np.ndarray, true_cost: np.ndarray) -> dict[str, float]:
    true_log = np.log1p(np.asarray(true_cost, np.float64))
    predicted_log = np.asarray(predicted_log, np.float64)
    rows = np.arange(len(true_cost))
    true_best = np.argmin(true_cost, axis=1)
    predicted_best = np.argmin(predicted_log, axis=1)
    available = true_cost[:, 0] - true_cost[rows, true_best]
    chosen_gain = true_cost[:, 0] - true_cost[rows, predicted_best]
    material = available > 1e-5
    return {
        "pearson_log_cost": correlation(predicted_log, true_log),
        "warm_teacher_order_accuracy": float(
            np.mean(predicted_log[:, 0] > predicted_log[rows, true_best])
        ),
        "bank_gain_recovery": float(
            np.sum(chosen_gain[material]) / np.sum(available[material])
        ) if np.any(material) else 0.0,
    }


@torch.no_grad()
def actor_predict(
    actor: DirectNoAnchorGTXActor,
    inputs: tuple[np.ndarray, ...],
    rows: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    actor.eval()
    output = []
    for start in range(0, len(rows), 256):
        local = rows[start : start + 256]
        tensors = tuple(torch.from_numpy(value[local]).to(device) for value in inputs)
        output.append(actor(*tensors)[1].cpu().numpy())
    return np.concatenate(output).astype(np.float32)


@torch.no_grad()
def critic_predict(
    critic: ConfigurableAbsoluteActionValueCritic,
    inputs: tuple[np.ndarray, ...],
    actions: np.ndarray,
    rows: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    critic.eval()
    output = []
    for start in range(0, len(rows), 16):
        local = rows[start : start + 16]
        output.append(
            critic(
                torch.from_numpy(inputs[0][local]).to(device),
                torch.from_numpy(inputs[1][local]).to(device),
                torch.from_numpy(inputs[2][local]).to(device),
                torch.from_numpy(actions[local]).to(device),
            ).cpu().numpy()
        )
    return np.concatenate(output).astype(np.float32)


def query_costs(
    controller: TorchMPPIController,
    data: dict[str, np.ndarray],
    rows: np.ndarray,
    knots: np.ndarray,
) -> np.ndarray:
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
        "trainer": sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "source_manifest": sha256(source / "manifest.json")
        == manifest["source_manifest_sha256"],
        "source_validation": sha256(source / "validation.json")
        == manifest["source_validation_sha256"],
        "source_bank": sha256(source / "bank.npz") == manifest["source_bank_sha256"],
        "query_checkpoint": sha256(Path(source_manifest["query_checkpoint"]))
        == manifest["query_checkpoint_sha256"],
        "summary": sha256(artifact / "summary.json") == manifest["summary_sha256"],
        "oof_predictions": sha256(artifact / "oof_predictions.npz")
        == manifest["oof_predictions_sha256"],
    }
    if not all(hash_checks.values()):
        raise AssertionError(f"artifact hash failure: {hash_checks}")
    if source_validation["qualification"] != "QUERY_EXPECTED_ROAD_FULLRANK_PASS":
        raise AssertionError("source validation qualification changed")

    with np.load(source / "bank.npz", allow_pickle=False) as archive:
        raw = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(artifact / "oof_predictions.npz", allow_pickle=False) as archive:
        oof = {name: np.asarray(archive[name]) for name in archive.files}

    data = {
        "history": raw["history"].astype(np.float32),
        "reference_features": np.stack(
            [
                ego_reference_features(value, float(state[3]))
                for value, state in zip(raw["reference_ego"], raw["state"])
            ]
        ).astype(np.float32),
        "current": np.stack(
            [
                np.asarray((state[3], state[4], *action), np.float32)
                for state, action in zip(raw["state"], raw["current_action"])
            ]
        ),
        "state": raw["state"].astype(np.float32),
        "current_action": raw["current_action"].astype(np.float32),
        "reference": raw["reference"].astype(np.float32),
        "actions": raw["candidate_knots"].astype(np.float32),
        "costs": raw["candidate_cost"].astype(np.float32),
        "teacher": raw["fullrank_teacher_knots"].astype(np.float32),
        "warm_cost": raw["warm_direct_cost_replayed"].astype(np.float64),
        "teacher_cost": raw["fullrank_teacher_direct_cost"].astype(np.float64),
        "episode": raw["episode_id"].astype(str),
        "fold": raw["fold_id"].astype(np.int64),
        "row": raw["row_index"].astype(np.int64),
    }
    row_contract_error = float(
        max(
            np.max(np.abs(oof["row_index"] - data["row"])),
            np.max(np.abs(oof["fold_id"] - data["fold"])),
            0 if np.array_equal(oof["episode_id"], data["episode"]) else 1,
        )
    )

    parent_manifest = json.loads(
        (Path(source_manifest["parent_t0"]) / "manifest.json").read_text()
    )
    collection_manifest = json.loads(
        (Path(parent_manifest["source_collection"]) / "manifest.json").read_text()
    )
    device = torch.device(args.device)
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
    critic_prediction_error = 0.0
    normalization_error = 0.0
    split_error = 0.0
    selection_error = 0.0
    invariance_error = 0.0
    formal_error = 0.0
    bound_violation = 0.0
    checkpoint_hashes = {}
    recomputed_records = []
    for record in summary["records"]:
        fold = int(record["fold"])
        seed = int(record["seed"])
        seed_index = seed_to_index[seed]
        checkpoint = Path(record["checkpoint"])
        checkpoint_hashes[checkpoint.name] = (
            sha256(checkpoint) == manifest["checkpoint_sha256"][checkpoint.name]
            == record["checkpoint_sha256"]
        )
        payload = torch.load(checkpoint, map_location="cpu")
        expected_oof = np.flatnonzero(data["fold"] == fold)
        selection_fold = (fold + int(config["selection_fold_offset"])) % int(config["folds"])
        expected_selection = np.flatnonzero(data["fold"] == selection_fold)
        expected_fit = np.flatnonzero(
            (data["fold"] != fold) & (data["fold"] != selection_fold)
        )
        for actual, expected in (
            (payload["fit_indices"], expected_fit),
            (payload["selection_indices"], expected_selection),
            (payload["oof_indices"], expected_oof),
        ):
            split_error = max(split_error, 0.0 if np.array_equal(actual, expected) else 1.0)
        groups = [set(data["episode"][rows]) for rows in (expected_fit, expected_selection, expected_oof)]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            split_error = 1.0

        expected_normalizer = MPPIProposalNormalization.fit(
            data["history"][expected_fit],
            data["reference_features"][expected_fit],
            data["current"][expected_fit],
        )
        normalization_error = max(
            normalization_error,
            maximum_dictionary_error(payload["normalization"], expected_normalizer.to_dict()),
        )
        normalizer = MPPIProposalNormalization.from_dict(payload["normalization"])
        normalized = normalizer.normalize_numpy(
            data["history"], data["reference_features"], data["current"]
        )
        count = len(data["row"])
        inputs = (
            *(value.astype(np.float32) for value in normalized),
            np.zeros((count, 8, 2), np.float32),
            np.zeros((count, 74), np.float32),
            np.zeros((count, 32), np.float32),
        )

        actor = DirectNoAnchorGTXActor(dropout=0.0).to(device)
        actor.load_state_dict(payload["actor_state_dict"], strict=True)
        predicted = actor_predict(actor, inputs, expected_oof, device)
        prediction_error = max(
            prediction_error,
            float(np.max(np.abs(predicted - oof["actor_knots"][seed_index, expected_oof]))),
        )
        bound_violation = max(
            bound_violation,
            float(max(np.max(predicted - 1.0), np.max(-1.0 - predicted), 0.0)),
        )
        local = expected_oof[:20]
        baseline = actor_predict(actor, inputs, local, device)
        changed = list(inputs)
        changed[3] = np.random.default_rng(1000 + fold).normal(size=changed[3].shape).astype(np.float32)
        invariance_error = max(
            invariance_error,
            float(np.max(np.abs(actor_predict(actor, tuple(changed), local, device) - baseline))),
        )
        changed = list(inputs)
        changed[4] = np.random.default_rng(2000 + fold).normal(size=changed[4].shape).astype(np.float32)
        invariance_error = max(
            invariance_error,
            float(np.max(np.abs(actor_predict(actor, tuple(changed), local, device) - baseline))),
        )
        changed = list(inputs)
        changed[5] = np.random.default_rng(3000 + fold).normal(size=changed[5].shape).astype(np.float32)
        invariance_error = max(
            invariance_error,
            float(np.max(np.abs(actor_predict(actor, tuple(changed), local, device) - baseline))),
        )

        physical = []
        for twin in (1, 2):
            critic = ConfigurableAbsoluteActionValueCritic().to(device)
            critic.load_state_dict(payload[f"critic{twin}_state_dict"], strict=True)
            standardized = critic_predict(
                critic, inputs, data["actions"], expected_oof, device
            )
            training = payload[f"critic{twin}_training"]
            physical.append(
                standardized * float(training["target_std"])
                + float(training["target_mean"])
            )
        conservative = np.maximum(physical[0], physical[1])
        critic_prediction_error = max(
            critic_prediction_error,
            float(
                np.max(
                    np.abs(
                        conservative
                        - oof["twin_conservative_log_cost"][seed_index, expected_oof]
                    )
                )
            ),
        )
        metrics = critic_metrics(conservative, data["costs"][expected_oof])
        stored_metrics = record["critic_twin_conservative"]["oof"]
        recomputed_records.append(metrics)
        selection_error = max(
            selection_error,
            abs(
                float(payload["actor_training"]["best_selection_cost_mean"])
                - min(
                    float(row["selection_cost_mean"])
                    for row in payload["actor_training"]["selection_history"]
                )
            ),
            abs(
                int(payload["actor_training"]["best_epoch"])
                - int(
                    min(
                        payload["actor_training"]["selection_history"],
                        key=lambda row: row["selection_cost_mean"],
                    )["epoch"]
                )
            ),
            *(abs(metrics[name] - float(stored_metrics[name])) for name in metrics),
        )
        formal_error = max(
            formal_error,
            float(bool(payload.get("formal_validation_or_test_consumed", True))),
        )

    if not all(checkpoint_hashes.values()):
        raise AssertionError("checkpoint hash failure")

    all_rows = np.arange(len(data["row"]), dtype=np.int64)
    replay_cost = np.empty_like(oof["actor_direct_cost"], dtype=np.float64)
    for seed_index, seed in enumerate(seeds):
        replay_cost[seed_index] = query_costs(
            controller, data, all_rows, oof["actor_knots"][seed_index]
        )
        print(f"replayed OOF seed {seed}", flush=True)
    cost_error = float(np.max(np.abs(replay_cost - oof["actor_direct_cost"])))
    cost_relative_error = float(
        np.max(
            np.abs(replay_cost - oof["actor_direct_cost"])
            / np.maximum(np.abs(oof["actor_direct_cost"]), 1.0)
        )
    )
    support_rows = np.arange(15, len(data["row"]), 30, dtype=np.int64)
    support_replay = query_costs(
        controller,
        data,
        support_rows,
        oof["support_projected_teacher_knots"][support_rows],
    )
    support_cost_error = float(
        np.max(
            np.abs(
                support_replay
                - oof["support_projected_teacher_direct_cost"][support_rows]
            )
        )
    )

    record_corr = np.asarray([row["pearson_log_cost"] for row in recomputed_records])
    record_order = np.asarray([row["warm_teacher_order_accuracy"] for row in recomputed_records])
    record_gain = np.asarray([row["bank_gain_recovery"] for row in recomputed_records])
    actor_recovery = []
    actor_p05 = []
    for record in summary["records"]:
        actor_recovery.append(float(record["actor"]["oof"]["aggregate_teacher_gain_recovery"]))
        actor_p05.append(float(record["actor"]["oof"]["warm_relative_gain"]["p05"]))
    recomputed_gates = {
        "actor_oof_recovery_median_ge_0p50": bool(np.median(actor_recovery) >= 0.5),
        "critic_oof_pearson_median_ge_0p50": bool(np.median(record_corr) >= 0.5),
        "critic_oof_warm_teacher_order_median_ge_0p80": bool(np.median(record_order) >= 0.8),
        "critic_oof_bank_gain_recovery_median_ge_0p50": bool(np.median(record_gain) >= 0.5),
    }
    expected_qualification = (
        "QUERY_ABSOLUTE_PRETRAIN_READY_FOR_ACTOR_VISITED_OAC"
        if all(recomputed_gates.values())
        else "QUERY_ABSOLUTE_PRETRAIN_FAIL_NO_OAC"
    )
    gate_error = 0.0 if (
        recomputed_gates == summary["initialization_gates"]
        and expected_qualification == summary["qualification"] == manifest["qualification"]
    ) else 1.0

    checks = {
        "artifact_hashes": all(hash_checks.values()) and all(checkpoint_hashes.values()),
        "source_qualified_and_sealed": (
            source_validation["qualification"] == "QUERY_EXPECTED_ROAD_FULLRANK_PASS"
            and not source_manifest.get("formal_validation_or_test_consumed", True)
        ),
        "row_contract": row_contract_error == 0.0,
        "episode_nested_split": split_error == 0.0,
        "fit_only_normalization": normalization_error <= 1e-7,
        "checkpoint_selection_and_metrics": selection_error <= 1e-10,
        "actor_checkpoint_prediction_replay": prediction_error <= 1e-7,
        "strict_no_anchor_invariance": invariance_error == 0.0,
        "bounded_absolute_actor_output": bound_violation == 0.0,
        "twin_critic_prediction_replay": critic_prediction_error <= 1e-7,
        "pytorch_query_actor_oof_replay": cost_error <= 1e-4 and cost_relative_error <= 1e-5,
        "support_oracle_query_shadow": support_cost_error <= 1e-4,
        "initialization_gate_reconstruction": gate_error == 0.0,
        "finite": all(
            np.isfinite(value).all()
            for value in (
                oof["actor_knots"],
                oof["actor_direct_cost"],
                oof["twin_conservative_log_cost"],
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
        "QUERY_ABSOLUTE_PRETRAIN_INDEPENDENT_PASS"
        if all(checks.values())
        else "QUERY_ABSOLUTE_PRETRAIN_INDEPENDENT_FAIL"
    )
    validation = {
        "qualification": qualification,
        "initialization_qualification": summary["qualification"],
        "artifact": str(artifact),
        "checks": checks,
        "hash_checks": hash_checks,
        "checkpoint_hash_checks": checkpoint_hashes,
        "recomputed_initialization_gates": recomputed_gates,
        "direct_tail_gate": {
            "all_run_warm_relative_p05_nonnegative": bool(np.all(np.asarray(actor_p05) >= 0.0)),
            "minimum_run_p05": float(np.min(actor_p05)),
        },
        "maximum_errors": {
            "row_contract": row_contract_error,
            "nested_split": split_error,
            "normalization": normalization_error,
            "checkpoint_selection_and_metrics": selection_error,
            "actor_checkpoint_prediction": prediction_error,
            "actor_no_anchor_invariance": invariance_error,
            "actor_bound_violation": bound_violation,
            "twin_critic_prediction": critic_prediction_error,
            "pytorch_query_actor_oof_cost": cost_error,
            "pytorch_query_actor_oof_cost_relative": cost_relative_error,
            "support_oracle_query_cost": support_cost_error,
            "initialization_gate": gate_error,
            "formal": formal_error,
        },
        "support_oracle_shadow_rows": support_rows.tolist(),
        "formal_validation_or_test_consumed": False,
    }
    dump_json(artifact / "validation.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))
    if qualification.endswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
