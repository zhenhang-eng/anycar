#!/usr/bin/env python3
"""Nested-OOF absolute/no-anchor Query Actor and Twin Critic pretraining.

This is the frozen high-speed OAC initialization contract ported to the validated
expected-road Query environment.  The Actor directly predicts an absolute 8x2
sampling center and cannot observe warm/anchor, feedback, gradients, teacher ids,
or costs.  Twin scalar Critics learn all saved absolute candidate Query costs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
from pretrain_highspeed_actor_twin_critic import (  # noqa: E402
    actor_center_scale,
    actor_metrics,
    actor_predict,
    critic_metrics,
    critic_predict,
    distribution,
    train_critic,
)


DEFAULT_CONFIG = REPO_ROOT / (
    "scripts/model_verify/query_expected_road_absolute_pretrain_config_20260901_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(source: Path) -> tuple[dict[str, np.ndarray], dict, dict]:
    manifest = json.loads((source / "manifest.json").read_text())
    validation = json.loads((source / "validation.json").read_text())
    if validation["qualification"] != "QUERY_EXPECTED_ROAD_FULLRANK_PASS":
        raise AssertionError("source full-rank sidecar did not pass")
    if manifest.get("formal_validation_or_test_consumed", True):
        raise AssertionError("source consumed formal validation/test")
    if manifest.get("dbm_fields_or_labels_consumed"):
        raise AssertionError("source reports DBM fields or labels")
    bank_path = source / "bank.npz"
    if sha256(bank_path) != manifest["bank_sha256"]:
        raise AssertionError("source bank hash mismatch")
    with np.load(bank_path, allow_pickle=False) as archive:
        raw = {name: np.asarray(archive[name]) for name in archive.files}
    parent_manifest = json.loads(
        (Path(manifest["parent_t0"]) / "manifest.json").read_text()
    )
    collection_manifest = json.loads(
        (Path(parent_manifest["source_collection"]) / "manifest.json").read_text()
    )
    reference = np.stack(
        [
            ego_reference_features(value, float(state[3]))
            for value, state in zip(raw["reference_ego"], raw["state"])
        ]
    ).astype(np.float32)
    current = np.stack(
        [
            np.asarray((state[3], state[4], *action), np.float32)
            for state, action in zip(raw["state"], raw["current_action"])
        ]
    )
    data = {
        "history": raw["history"].astype(np.float32),
        "reference": reference,
        "current": current,
        "state": raw["state"].astype(np.float32),
        "current_action": raw["current_action"].astype(np.float32),
        "rollout_reference": raw["reference"].astype(np.float32),
        "anchor": raw["mean_knots_before"].astype(np.float32),
        "teacher": raw["fullrank_teacher_knots"].astype(np.float32),
        "anchor_cost": raw["warm_direct_cost_replayed"].astype(np.float32),
        "teacher_cost": raw["fullrank_teacher_direct_cost"].astype(np.float32),
        "t0_cost": raw["t0_teacher_direct_cost_replayed"].astype(np.float32),
        "actions": raw["candidate_knots"].astype(np.float32),
        "costs": raw["candidate_cost"].astype(np.float32),
        "episode": raw["episode_id"].astype(str),
        "fold": raw["fold_id"].astype(np.int64),
        "speed": raw["speed_kph"].astype(np.float32),
        "variant": raw["variant_index"].astype(np.int64),
        "road": raw["road_name"].astype(str),
        "row_index": raw["row_index"].astype(np.int64),
    }
    if data["actions"].shape != (600, 132, 8, 2):
        raise AssertionError("unexpected candidate bank shape")
    if not np.allclose(data["costs"][:, 0], data["anchor_cost"], atol=1e-6):
        raise AssertionError("candidate zero is not exact warm")
    rows = np.arange(600)
    if not np.array_equal(np.argmin(data["costs"], axis=1), raw["fullrank_teacher_index"]):
        raise AssertionError("teacher argmin mismatch")
    if not np.allclose(
        data["actions"][rows, np.argmin(data["costs"], axis=1)], data["teacher"], atol=0
    ):
        raise AssertionError("teacher center mismatch")
    data["source_bank_sha256"] = np.asarray(manifest["bank_sha256"])
    data["source_manifest_sha256"] = np.asarray(sha256(source / "manifest.json"))
    return data, manifest, collection_manifest


def normalized_inputs(
    data: dict[str, np.ndarray], fit: np.ndarray
) -> tuple[tuple[np.ndarray, ...], MPPIProposalNormalization]:
    normalizer = MPPIProposalNormalization.fit(
        data["history"][fit], data["reference"][fit], data["current"][fit]
    )
    history, reference, current = normalizer.normalize_numpy(
        data["history"], data["reference"], data["current"]
    )
    count = len(history)
    return (
        history.astype(np.float32),
        reference.astype(np.float32),
        current.astype(np.float32),
        np.zeros((count, 8, 2), np.float32),
        np.zeros((count, 74), np.float32),
        np.zeros((count, 32), np.float32),
    ), normalizer


def query_cost(
    controller: TorchMPPIController,
    data: dict[str, np.ndarray],
    knots: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    actions = interpolate_knots(knots)
    output = np.empty(len(rows), dtype=np.float32)
    for local, row in enumerate(rows):
        result = controller.evaluate_action_sequences(
            data["state"][row],
            data["current_action"][row],
            data["history"][row : row + 1],
            data["rollout_reference"][row],
            actions[local : local + 1],
        )
        output[local] = float(result["cost"][0].cpu())
    return output


def train_actor(
    data: dict[str, np.ndarray],
    inputs: tuple[np.ndarray, ...],
    fit: np.ndarray,
    selection: np.ndarray,
    seed: int,
    config: dict,
    controller: TorchMPPIController,
    device: torch.device,
) -> tuple[DirectNoAnchorGTXActor, dict[str, Any]]:
    set_seed(seed)
    center, scale = actor_center_scale(data["teacher"], fit)
    actor = DirectNoAnchorGTXActor(
        dropout=0.0, center=center.to(device), scale=scale.to(device)
    ).to(device)
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(config["actor_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    target = torch.from_numpy(data["teacher"]).to(device)
    rng = np.random.default_rng(806_301 + seed)
    best_score = math.inf
    best_state = None
    best_epoch = 0
    history = []
    for epoch in range(1, int(config["actor_epochs"]) + 1):
        actor.train()
        order = rng.permutation(fit)
        losses = []
        for start in range(0, len(order), int(config["actor_batch_size"])):
            rows = order[start : start + int(config["actor_batch_size"])]
            tensors = tuple(torch.from_numpy(value[rows]).to(device) for value in inputs)
            _, predicted = actor(*tensors)
            loss = (predicted - target[rows]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % int(config["actor_selection_stride"]) == 0:
            predicted = actor_predict(actor, inputs, selection, device)
            cost = query_cost(controller, data, predicted, selection)
            score = float(np.mean(cost))
            report = actor_metrics(
                cost, data["anchor_cost"][selection], data["teacher_cost"][selection]
            )
            history.append(
                {
                    "epoch": epoch,
                    "fit_mse": float(np.mean(losses)),
                    "selection_cost_mean": score,
                    "selection_recovery": report["aggregate_teacher_gain_recovery"],
                }
            )
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(actor.state_dict())
    if best_state is None:
        raise AssertionError("Actor checkpoint selection failed")
    actor.load_state_dict(best_state, strict=True)
    return actor, {
        "best_epoch": best_epoch,
        "best_selection_cost_mean": best_score,
        "selection_history": history,
        "out_center": center.numpy().tolist(),
        "out_scale": scale.numpy().tolist(),
        "support_std": float(config["output_support_std"]),
    }


def no_anchor_invariance(
    actor: DirectNoAnchorGTXActor,
    inputs: tuple[np.ndarray, ...],
    rows: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    actor.eval()
    local = rows[: min(20, len(rows))]
    base = [torch.from_numpy(value[local]).to(device) for value in inputs]
    with torch.no_grad():
        reference = actor(*base)[1]
        changed_anchor = list(base)
        changed_anchor[3] = torch.linspace(-1, 1, 16, device=device).reshape(1, 8, 2).expand(len(local), -1, -1)
        changed_feedback = list(base)
        changed_feedback[4] = torch.randn_like(base[4])
        changed_gradient = list(base)
        changed_gradient[5] = torch.randn_like(base[5])
        return {
            "anchor": float(torch.max(torch.abs(actor(*changed_anchor)[1] - reference)).cpu()),
            "feedback": float(torch.max(torch.abs(actor(*changed_feedback)[1] - reference)).cpu()),
            "gradient": float(torch.max(torch.abs(actor(*changed_gradient)[1] - reference)).cpu()),
        }


def pooled_actor_report(
    data: dict[str, np.ndarray], costs: np.ndarray
) -> dict[str, Any]:
    report = actor_metrics(costs, data["anchor_cost"], data["teacher_cost"])
    t0_gain = data["t0_cost"].astype(np.float64) - costs.astype(np.float64)
    report.update(
        {
            "t0_relative_gain": distribution(t0_gain),
            "beats_or_equals_t0_fraction": float(np.mean(t0_gain >= -1e-5)),
            "t0_regression_fraction": float(np.mean(t0_gain < -1e-5)),
        }
    )
    return report


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    source = Path(config["source_sidecar"]).resolve()
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    if config["formal_validation_or_test_consumed"]:
        raise AssertionError("formal validation/test must remain sealed")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data, source_manifest, collection_manifest = load_data(source)
    device = torch.device(args.device)
    params = TorchMPPIParams(**collection_manifest["collection"]["mppi"])
    query_model = QueryDeploymentModel.from_checkpoint(
        Path(source_manifest["query_checkpoint"]), device
    )
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query_model), params, device=str(device)
    )
    output.mkdir(parents=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    seeds = [int(value) for value in config["seeds"]]
    count = len(data["episode"])
    oof_actor_knots = np.empty((len(seeds), count, 8, 2), np.float32)
    oof_actor_cost = np.empty((len(seeds), count), np.float32)
    oof_twin_log = np.empty((len(seeds), count, data["actions"].shape[1]), np.float32)
    support_knots = np.empty((count, 8, 2), np.float32)
    records = []
    critic_args = SimpleNamespace(
        critic_lr=float(config["critic_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        critic_state_batch_size=int(config["critic_state_batch_size"]),
        critic_candidates_per_state=int(config["critic_candidates_per_state"]),
        ranking_temperature=float(config["ranking_temperature"]),
        ranking_weight=float(config["ranking_weight"]),
        critic_epochs=int(config["critic_epochs"]),
        selection_stride=int(config["critic_selection_stride"]),
    )
    for fold in range(int(config["folds"])):
        oof = np.flatnonzero(data["fold"] == fold)
        selection_fold = (fold + int(config["selection_fold_offset"])) % int(config["folds"])
        selection = np.flatnonzero(data["fold"] == selection_fold)
        fit = np.flatnonzero((data["fold"] != fold) & (data["fold"] != selection_fold))
        if (len(fit), len(selection), len(oof)) != (360, 120, 120):
            raise AssertionError("unexpected nested split")
        groups = [set(data["episode"][rows].tolist()) for rows in (fit, selection, oof)]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise AssertionError("episode leakage")
        inputs, normalizer = normalized_inputs(data, fit)
        support_center, support_scale = actor_center_scale(data["teacher"], fit)
        support_knots[oof] = np.clip(
            data["teacher"][oof],
            support_center.numpy() - support_scale.numpy(),
            support_center.numpy() + support_scale.numpy(),
        )
        for seed_index, seed in enumerate(seeds):
            run_seed = 10_000 + fold * 100 + seed
            print(f"fold={fold} seed={seed}: Actor", flush=True)
            actor, actor_training = train_actor(
                data, inputs, fit, selection, run_seed, config, controller, device
            )
            actor_splits = {}
            actor_predictions = {}
            for name, rows in (("fit", fit), ("selection", selection), ("oof", oof)):
                prediction = actor_predict(actor, inputs, rows, device)
                cost = query_cost(controller, data, prediction, rows)
                actor_predictions[name] = prediction
                actor_splits[name] = actor_metrics(
                    cost, data["anchor_cost"][rows], data["teacher_cost"][rows]
                )
                if name == "oof":
                    oof_actor_knots[seed_index, rows] = prediction
                    oof_actor_cost[seed_index, rows] = cost

            critics = []
            critic_training = []
            critic_splits = []
            for twin in range(2):
                critic_seed = 20_000 + fold * 100 + seed * 10 + twin
                print(f"fold={fold} seed={seed}: Critic {twin + 1}", flush=True)
                critic, training = train_critic(
                    data, inputs, fit, selection, critic_seed, critic_args, device
                )
                split_report = {}
                for name, rows in (("fit", fit), ("selection", selection), ("oof", oof)):
                    prediction = critic_predict(
                        critic, inputs, data["actions"], rows, device
                    )
                    split_report[name] = critic_metrics(
                        prediction,
                        data["costs"][rows],
                        training["target_mean"],
                        training["target_std"],
                    )
                critics.append(critic)
                critic_training.append(training)
                critic_splits.append(split_report)

            twin_splits = {}
            for name, rows in (("fit", fit), ("selection", selection), ("oof", oof)):
                physical = []
                for critic, training in zip(critics, critic_training):
                    prediction = critic_predict(
                        critic, inputs, data["actions"], rows, device
                    )
                    physical.append(
                        prediction * training["target_std"] + training["target_mean"]
                    )
                conservative = np.maximum(physical[0], physical[1])
                twin_splits[name] = critic_metrics(
                    conservative, data["costs"][rows], 0.0, 1.0
                )
                if name == "oof":
                    oof_twin_log[seed_index, rows] = conservative

            invariance = no_anchor_invariance(actor, inputs, oof, device)
            checkpoint = checkpoint_dir / f"pretrain_fold{fold}_seed{seed}.pt"
            torch.save(
                {
                    "qualification": "QUERY_ABSOLUTE_ACTOR_TWIN_CRITIC_PRETRAIN_TRAIN_ONLY",
                    "fold": fold,
                    "seed": seed,
                    "run_seed": run_seed,
                    "selection_fold": selection_fold,
                    "actor_architecture": config["actor_architecture"],
                    "actor_state_dict": {
                        name: value.detach().cpu() for name, value in actor.state_dict().items()
                    },
                    "actor_training": actor_training,
                    "critic_architecture": config["critic_architecture"],
                    "critic1_state_dict": {
                        name: value.detach().cpu() for name, value in critics[0].state_dict().items()
                    },
                    "critic2_state_dict": {
                        name: value.detach().cpu() for name, value in critics[1].state_dict().items()
                    },
                    "critic1_training": critic_training[0],
                    "critic2_training": critic_training[1],
                    "normalization": normalizer.to_dict(),
                    "fit_indices": fit,
                    "selection_indices": selection,
                    "oof_indices": oof,
                    "source_bank_sha256": str(data["source_bank_sha256"]),
                    "query_checkpoint_sha256": source_manifest["query_checkpoint_sha256"],
                    "no_anchor_invariance": invariance,
                    "formal_validation_or_test_consumed": False,
                },
                checkpoint,
            )
            record = {
                "fold": fold,
                "seed": seed,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256(checkpoint),
                "fit_episodes": sorted(groups[0]),
                "selection_episodes": sorted(groups[1]),
                "oof_episodes": sorted(groups[2]),
                "no_anchor_invariance": invariance,
                "actor": actor_splits,
                "critic1": critic_splits[0],
                "critic2": critic_splits[1],
                "critic_twin_conservative": twin_splits,
            }
            records.append(record)
            print(
                f"fold={fold} seed={seed} OOF Actor R="
                f"{actor_splits['oof']['aggregate_teacher_gain_recovery']:.3f}; "
                f"Twin corr={twin_splits['oof']['pearson_log_cost']:.3f} "
                f"gainR={twin_splits['oof']['bank_gain_recovery']:.3f}",
                flush=True,
            )

    all_rows = np.arange(count, dtype=np.int64)
    support_cost = query_cost(controller, data, support_knots, all_rows)
    support_report = actor_metrics(
        support_cost, data["anchor_cost"], data["teacher_cost"]
    )
    oof_path = output / "oof_predictions.npz"
    np.savez_compressed(
        oof_path,
        row_index=data["row_index"],
        episode_id=data["episode"],
        fold_id=data["fold"],
        seeds=np.asarray(seeds, np.int64),
        actor_knots=oof_actor_knots,
        actor_direct_cost=oof_actor_cost,
        twin_conservative_log_cost=oof_twin_log,
        support_projected_teacher_knots=support_knots,
        support_projected_teacher_direct_cost=support_cost,
    )

    pooled_actor = []
    pooled_critic = []
    for seed_index, seed in enumerate(seeds):
        actor_report = pooled_actor_report(data, oof_actor_cost[seed_index])
        critic_report = critic_metrics(
            oof_twin_log[seed_index], data["costs"], 0.0, 1.0
        )
        pooled_actor.append(
            {
                "seed": seed,
                "overall": actor_report,
                "by_speed_kph": {
                    str(speed): actor_metrics(
                        oof_actor_cost[seed_index, data["speed"] == speed],
                        data["anchor_cost"][data["speed"] == speed],
                        data["teacher_cost"][data["speed"] == speed],
                    )
                    for speed in sorted(np.unique(data["speed"]))
                },
            }
        )
        pooled_critic.append({"seed": seed, "overall": critic_report})

    actor_record_recovery = np.asarray(
        [row["actor"]["oof"]["aggregate_teacher_gain_recovery"] for row in records]
    )
    actor_record_p05 = np.asarray(
        [row["actor"]["oof"]["warm_relative_gain"]["p05"] for row in records]
    )
    critic_record_corr = np.asarray(
        [row["critic_twin_conservative"]["oof"]["pearson_log_cost"] for row in records]
    )
    critic_record_order = np.asarray(
        [row["critic_twin_conservative"]["oof"]["warm_teacher_order_accuracy"] for row in records]
    )
    critic_record_recovery = np.asarray(
        [row["critic_twin_conservative"]["oof"]["bank_gain_recovery"] for row in records]
    )
    gates = {
        "actor_oof_recovery_median_ge_0p50": bool(np.median(actor_record_recovery) >= 0.5),
        "critic_oof_pearson_median_ge_0p50": bool(np.median(critic_record_corr) >= 0.5),
        "critic_oof_warm_teacher_order_median_ge_0p80": bool(np.median(critic_record_order) >= 0.8),
        "critic_oof_bank_gain_recovery_median_ge_0p50": bool(np.median(critic_record_recovery) >= 0.5),
    }
    tail_gate = {
        "all_run_warm_relative_p05_nonnegative": bool(np.all(actor_record_p05 >= 0.0)),
        "minimum_run_p05": float(actor_record_p05.min()),
    }
    qualification = (
        "QUERY_ABSOLUTE_PRETRAIN_READY_FOR_ACTOR_VISITED_OAC"
        if all(gates.values())
        else "QUERY_ABSOLUTE_PRETRAIN_FAIL_NO_OAC"
    )
    summary = {
        "qualification": qualification,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "sidecar": str(source),
            "manifest_sha256": sha256(source / "manifest.json"),
            "validation_sha256": sha256(source / "validation.json"),
            "bank_sha256": source_manifest["bank_sha256"],
            "query_checkpoint": source_manifest["query_checkpoint"],
            "query_checkpoint_sha256": source_manifest["query_checkpoint_sha256"],
        },
        "contract": config,
        "support_projection_oracle": support_report,
        "initialization_gates": gates,
        "direct_tail_gate": tail_gate,
        "actor_oof_by_record": {
            "recovery": distribution(actor_record_recovery),
            "warm_gain_p05": distribution(actor_record_p05),
        },
        "critic_oof_by_record": {
            "pearson_log_cost": distribution(critic_record_corr),
            "warm_teacher_order_accuracy": distribution(critic_record_order),
            "bank_gain_recovery": distribution(critic_record_recovery),
        },
        "pooled_actor_oof_by_seed": pooled_actor,
        "pooled_critic_oof_by_seed": pooled_critic,
        "records": records,
        "formal_validation_or_test_consumed": False,
    }
    summary_path = output / "summary.json"
    dump_json(summary_path, summary)
    manifest = {
        "schema_version": "query-absolute-actor-twin-critic-pretrain-v1",
        "dataset_type": "train-only-query-absolute-noanchor-nested-oof-pretrain",
        "qualification": qualification,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "source_validation_sha256": sha256(source / "validation.json"),
        "source_bank_sha256": source_manifest["bank_sha256"],
        "query_checkpoint_sha256": source_manifest["query_checkpoint_sha256"],
        "summary_sha256": sha256(summary_path),
        "oof_predictions_sha256": sha256(oof_path),
        "checkpoint_sha256": {
            Path(record["checkpoint"]).name: record["checkpoint_sha256"] for record in records
        },
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "limitations": [
            "This is offline initialization, not actor-visited continuous OAC.",
            "Direct Actor tail failure does not disappear without the external exact-warm guard.",
            "The 40-kph source is not direction-balanced.",
        ],
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({"qualification": qualification, "gates": gates, "tail": tail_gate}, indent=2))


if __name__ == "__main__":
    main()
