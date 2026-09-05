#!/usr/bin/env python3
"""Train-only nested-CV Query Actor learnability measurement.

The Actor predicts one bounded 8x2 residual around the exact causal entry warm.
Checkpoints are selected only by frozen-Query direct J50 on an inner whole-episode
fold.  The outer fold is evaluated once after selection and is never used for
checkpoint or hyperparameter selection.  No DBM field, label, or gradient is read.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


DEFAULT_CONFIG = REPO_ROOT / (
    "scripts/model_verify/query_expected_road_actor_oof_config_20260901_v1.json"
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


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


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
        raise AssertionError("source reports DBM inputs")
    bank_path = source / "bank.npz"
    if sha256(bank_path) != manifest["bank_sha256"]:
        raise AssertionError("source bank hash mismatch")
    with np.load(bank_path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    if len(data["row_index"]) != 600 or not np.array_equal(
        data["row_index"], np.arange(600)
    ):
        raise AssertionError("unexpected source row contract")
    parent_manifest = json.loads(
        (Path(manifest["parent_t0"]) / "manifest.json").read_text()
    )
    source_manifest = json.loads(
        (Path(parent_manifest["source_collection"]) / "manifest.json").read_text()
    )
    return data, manifest, source_manifest


def build_actor_arrays(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
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
    return {
        "history": data["history"].astype(np.float32),
        "reference": reference,
        "current": current,
        "warm": data["mean_knots_before"].astype(np.float32),
    }


def normalize_inputs(
    arrays: dict[str, np.ndarray], fit: np.ndarray
) -> tuple[dict[str, np.ndarray], MPPIProposalNormalization]:
    normalizer = MPPIProposalNormalization.fit(
        arrays["history"][fit], arrays["reference"][fit], arrays["current"][fit]
    )
    history, reference, current = normalizer.normalize_numpy(
        arrays["history"], arrays["reference"], arrays["current"]
    )
    return {
        "history": history.astype(np.float32),
        "reference": reference.astype(np.float32),
        "current": current.astype(np.float32),
        "warm": arrays["warm"],
    }, normalizer


def make_actor(config: dict, device: torch.device) -> TorchMPPIProposalPolicy:
    sigma = np.asarray(config["noise_sigma"], dtype=np.float32)
    trust = tuple((float(config["maximum_delta_sigma"]) * sigma).tolist())
    return TorchMPPIProposalPolicy(
        trust_scale=trust, dropout=float(config["dropout"])
    ).to(device)


@torch.no_grad()
def actor_predict(
    actor: TorchMPPIProposalPolicy,
    inputs: dict[str, np.ndarray],
    rows: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    actor.eval()
    output = []
    for start in range(0, len(rows), batch_size):
        local = rows[start : start + batch_size]
        _, center = actor(
            torch.from_numpy(inputs["history"][local]).to(device),
            torch.from_numpy(inputs["reference"][local]).to(device),
            torch.from_numpy(inputs["current"][local]).to(device),
            torch.from_numpy(inputs["warm"][local]).to(device),
        )
        output.append(center.cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def query_costs(
    controller: TorchMPPIController,
    data: dict[str, np.ndarray],
    rows: np.ndarray,
    knots: np.ndarray,
) -> np.ndarray:
    result = np.empty(len(rows), dtype=np.float64)
    actions = interpolate_knots(knots)
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


def metrics(
    rows: np.ndarray,
    actor_cost: np.ndarray,
    data: dict[str, np.ndarray],
    projected_teacher_cost: np.ndarray,
) -> dict[str, Any]:
    warm = data["warm_direct_cost_replayed"][rows].astype(np.float64)
    t0 = data["t0_teacher_direct_cost_replayed"][rows].astype(np.float64)
    fullrank = data["fullrank_teacher_direct_cost"][rows].astype(np.float64)
    projected = projected_teacher_cost[rows]
    actor_cost = np.asarray(actor_cost, np.float64)
    warm_gain = warm - actor_cost
    t0_gain = t0 - actor_cost

    def recovery(baseline: np.ndarray, target: np.ndarray) -> float:
        denominator = float(np.sum(baseline - target))
        return float(np.sum(baseline - actor_cost) / denominator)

    return {
        "rows": int(len(rows)),
        "actor_direct_cost": distribution(actor_cost),
        "warm_direct_cost": distribution(warm),
        "t0_direct_cost": distribution(t0),
        "fullrank_teacher_direct_cost": distribution(fullrank),
        "projected_teacher_direct_cost": distribution(projected),
        "warm_relative_gain": distribution(warm_gain),
        "t0_relative_gain": distribution(t0_gain),
        "fullrank_headroom_recovery": recovery(warm, fullrank),
        "projected_headroom_recovery": recovery(warm, projected),
        "t0_headroom_recovery": recovery(warm, t0),
        "beats_or_equals_warm_fraction": float(np.mean(warm_gain >= -1e-5)),
        "strictly_beats_warm_fraction": float(np.mean(warm_gain > 1e-5)),
        "warm_regression_fraction": float(np.mean(warm_gain < -1e-5)),
        "beats_or_equals_t0_fraction": float(np.mean(t0_gain >= -1e-5)),
        "strictly_beats_t0_fraction": float(np.mean(t0_gain > 1e-5)),
        "t0_regression_fraction": float(np.mean(t0_gain < -1e-5)),
    }


def episode_bootstrap_recovery(
    data: dict[str, np.ndarray], actor_cost: np.ndarray, seed: int
) -> dict[str, float | int]:
    episodes = np.unique(data["episode_id"])
    rng = np.random.default_rng(91_000 + seed)
    numerators = {}
    denominators = {}
    for episode in episodes:
        mask = data["episode_id"] == episode
        numerators[str(episode)] = float(
            np.sum(data["warm_direct_cost_replayed"][mask] - actor_cost[mask])
        )
        denominators[str(episode)] = float(
            np.sum(
                data["warm_direct_cost_replayed"][mask]
                - data["fullrank_teacher_direct_cost"][mask]
            )
        )
    draws = np.empty(5000, dtype=np.float64)
    for index in range(len(draws)):
        sampled = rng.choice(episodes, size=len(episodes), replace=True)
        numerator = sum(numerators[str(value)] for value in sampled)
        denominator = sum(denominators[str(value)] for value in sampled)
        draws[index] = numerator / denominator
    return {
        "draws": int(len(draws)),
        "lower": float(np.quantile(draws, 0.025)),
        "median": float(np.quantile(draws, 0.5)),
        "upper": float(np.quantile(draws, 0.975)),
    }


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
    arrays = build_actor_arrays(data)
    device = torch.device(args.device)
    params = TorchMPPIParams(**collection_manifest["collection"]["mppi"])
    query_model = QueryDeploymentModel.from_checkpoint(
        Path(source_manifest["query_checkpoint"]), device
    )
    controller = TorchMPPIController(
        TorchQueryRolloutBackend(query_model), params, device=str(device)
    )
    sigma = np.asarray(config["noise_sigma"], dtype=np.float32).reshape(1, 1, 2)
    maximum_delta = float(config["maximum_delta_sigma"])
    normalized_teacher = (
        data["fullrank_teacher_knots"] - arrays["warm"]
    ) / sigma
    projected_teacher = np.clip(
        arrays["warm"] + np.clip(normalized_teacher, -maximum_delta, maximum_delta) * sigma,
        -1.0,
        1.0,
    ).astype(np.float32)
    all_rows = np.arange(len(arrays["warm"]), dtype=np.int64)
    projected_teacher_cost = query_costs(
        controller, data, all_rows, projected_teacher
    )

    output.mkdir(parents=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    projected_path = output / "projected_teacher.npz"
    np.savez_compressed(
        projected_path,
        row_index=data["row_index"],
        projected_teacher_knots=projected_teacher,
        projected_teacher_direct_cost=projected_teacher_cost,
    )

    seeds = [int(value) for value in config["seeds"]]
    oof_prediction = np.empty((len(seeds), len(all_rows), 8, 2), dtype=np.float32)
    oof_cost = np.empty((len(seeds), len(all_rows)), dtype=np.float64)
    records: list[dict[str, Any]] = []
    for fold in range(int(config["folds"])):
        oof = np.flatnonzero(data["fold_id"] == fold)
        for seed_index, seed in enumerate(seeds):
            offset = int(config["inner_selection_fold_offset_by_seed"][seed_index])
            selection_fold = (fold + offset) % int(config["folds"])
            selection = np.flatnonzero(data["fold_id"] == selection_fold)
            fit = np.flatnonzero(
                (data["fold_id"] != fold) & (data["fold_id"] != selection_fold)
            )
            fit_episodes = set(data["episode_id"][fit].tolist())
            selection_episodes = set(data["episode_id"][selection].tolist())
            oof_episodes = set(data["episode_id"][oof].tolist())
            if fit_episodes & selection_episodes or fit_episodes & oof_episodes or selection_episodes & oof_episodes:
                raise AssertionError("episode leakage in nested split")
            if (len(fit), len(selection), len(oof)) != (360, 120, 120):
                raise AssertionError("unexpected nested split sizes")

            inputs, normalizer = normalize_inputs(arrays, fit)
            run_seed = 31_000 + fold * 100 + seed
            set_seed(run_seed)
            actor = make_actor(config, device)
            optimizer = torch.optim.AdamW(
                actor.parameters(),
                lr=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]),
            )
            target = torch.from_numpy(projected_teacher).to(device)
            rng = np.random.default_rng(71_000 + fold * 100 + seed)
            best_score = float("inf")
            best_epoch = 0
            best_state = None
            selection_history = []
            for epoch in range(1, int(config["epochs"]) + 1):
                actor.train()
                order = rng.permutation(fit)
                losses = []
                for start in range(0, len(order), int(config["batch_size"])):
                    rows = order[start : start + int(config["batch_size"])]
                    _, center = actor(
                        torch.from_numpy(inputs["history"][rows]).to(device),
                        torch.from_numpy(inputs["reference"][rows]).to(device),
                        torch.from_numpy(inputs["current"][rows]).to(device),
                        torch.from_numpy(inputs["warm"][rows]).to(device),
                    )
                    loss = torch.mean(((center - target[rows]) / torch.from_numpy(sigma).to(device)) ** 2)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                if epoch % int(config["selection_stride"]) == 0:
                    prediction = actor_predict(actor, inputs, selection, device)
                    direct_cost = query_costs(
                        controller, data, selection, prediction
                    )
                    score = float(np.mean(direct_cost))
                    row = {
                        "epoch": epoch,
                        "fit_normalized_mse": float(np.mean(losses)),
                        "selection_direct_cost_mean": score,
                        "selection_fullrank_headroom_recovery": metrics(
                            selection, direct_cost, data, projected_teacher_cost
                        )["fullrank_headroom_recovery"],
                    }
                    selection_history.append(row)
                    if score < best_score:
                        best_score = score
                        best_epoch = epoch
                        best_state = copy.deepcopy(actor.state_dict())
            if best_state is None:
                raise AssertionError("checkpoint selection failed")
            actor.load_state_dict(best_state, strict=True)

            split_metrics = {}
            split_predictions = {}
            split_costs = {}
            for name, rows in (("fit", fit), ("selection", selection), ("oof", oof)):
                prediction = actor_predict(actor, inputs, rows, device)
                direct_cost = query_costs(controller, data, rows, prediction)
                split_predictions[name] = prediction
                split_costs[name] = direct_cost
                split_metrics[name] = metrics(
                    rows, direct_cost, data, projected_teacher_cost
                )
            oof_prediction[seed_index, oof] = split_predictions["oof"]
            oof_cost[seed_index, oof] = split_costs["oof"]

            checkpoint = checkpoint_dir / f"actor_fold{fold}_seed{seed}.pt"
            torch.save(
                {
                    "qualification": "QUERY_EXPECTED_ROAD_ACTOR_TRAIN_ONLY",
                    "architecture": config["architecture"],
                    "actor_state_dict": {
                        name: value.detach().cpu()
                        for name, value in actor.state_dict().items()
                    },
                    "normalization": normalizer.to_dict(),
                    "fold": fold,
                    "seed": seed,
                    "run_seed": run_seed,
                    "selection_fold": selection_fold,
                    "fit_indices": fit,
                    "selection_indices": selection,
                    "oof_indices": oof,
                    "best_epoch": best_epoch,
                    "best_selection_direct_cost_mean": best_score,
                    "selection_history": selection_history,
                    "source_bank_sha256": source_manifest["bank_sha256"],
                    "query_checkpoint_sha256": source_manifest["query_checkpoint_sha256"],
                    "formal_validation_or_test_consumed": False,
                },
                checkpoint,
            )
            record = {
                "fold": fold,
                "seed": seed,
                "selection_fold": selection_fold,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256(checkpoint),
                "best_epoch": best_epoch,
                "best_selection_direct_cost_mean": best_score,
                "fit_episodes": sorted(fit_episodes),
                "selection_episodes": sorted(selection_episodes),
                "oof_episodes": sorted(oof_episodes),
                "metrics": split_metrics,
            }
            records.append(record)
            print(
                f"fold={fold} seed={seed} epoch={best_epoch} "
                f"OOF J={split_metrics['oof']['actor_direct_cost']['mean']:.6f} "
                f"H={split_metrics['oof']['fullrank_headroom_recovery']:.4f} "
                f"P05={split_metrics['oof']['warm_relative_gain']['p05']:.6f}",
                flush=True,
            )

    oof_path = output / "oof_predictions.npz"
    np.savez_compressed(
        oof_path,
        row_index=data["row_index"],
        episode_id=data["episode_id"],
        fold_id=data["fold_id"],
        seeds=np.asarray(seeds, dtype=np.int64),
        actor_knots=oof_prediction,
        actor_direct_cost=oof_cost,
    )
    seed_reports = []
    for seed_index, seed in enumerate(seeds):
        overall = metrics(all_rows, oof_cost[seed_index], data, projected_teacher_cost)
        overall["episode_bootstrap_fullrank_recovery_ci95"] = episode_bootstrap_recovery(
            data, oof_cost[seed_index], seed
        )
        seed_reports.append(
            {
                "seed": seed,
                "overall": overall,
                "by_speed_kph": {
                    str(speed): metrics(
                        np.flatnonzero(data["speed_kph"] == speed),
                        oof_cost[seed_index, data["speed_kph"] == speed],
                        data,
                        projected_teacher_cost,
                    )
                    for speed in sorted(np.unique(data["speed_kph"]))
                },
                "by_fold": {
                    str(fold): metrics(
                        np.flatnonzero(data["fold_id"] == fold),
                        oof_cost[seed_index, data["fold_id"] == fold],
                        data,
                        projected_teacher_cost,
                    )
                    for fold in range(int(config["folds"]))
                },
            }
        )

    recoveries = np.asarray(
        [row["overall"]["fullrank_headroom_recovery"] for row in seed_reports]
    )
    median_seed_index = int(np.argsort(recoveries)[len(recoveries) // 2])
    median_report = seed_reports[median_seed_index]
    primary = {
        "at_least_two_seeds_recovery_ge_0p50": bool(np.sum(recoveries >= 0.5) >= 2),
        "all_seeds_recovery_positive": bool(np.all(recoveries > 0.0)),
        "median_seed_bootstrap_lower_positive": bool(
            median_report["overall"]["episode_bootstrap_fullrank_recovery_ci95"]["lower"] > 0.0
        ),
    }
    tail = {
        "median_seed_warm_gain_p05_nonnegative": bool(
            median_report["overall"]["warm_relative_gain"]["p05"] >= 0.0
        ),
        "median_seed_worst_reported": median_report["overall"]["warm_relative_gain"]["min"],
    }
    primary_pass = all(primary.values())
    tail_pass = bool(tail["median_seed_warm_gain_p05_nonnegative"])
    if primary_pass and tail_pass:
        qualification = "QUERY_ACTOR_OOF_LEARNABILITY_PASS_DIRECT_TAIL_PASS"
    elif primary_pass:
        qualification = "QUERY_ACTOR_OOF_LEARNABILITY_PASS_WARM_GUARD_REQUIRED"
    else:
        qualification = "QUERY_ACTOR_OOF_LEARNABILITY_FAIL"

    cap_metrics = metrics(
        all_rows, projected_teacher_cost, data, projected_teacher_cost
    )
    summary = {
        "qualification": qualification,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": config,
        "source": {
            "sidecar": str(source),
            "manifest_sha256": sha256(source / "manifest.json"),
            "validation_sha256": sha256(source / "validation.json"),
            "bank_sha256": source_manifest["bank_sha256"],
            "query_checkpoint": source_manifest["query_checkpoint"],
            "query_checkpoint_sha256": source_manifest["query_checkpoint_sha256"],
        },
        "projected_teacher_cap_oracle": cap_metrics,
        "seed_reports": seed_reports,
        "median_seed": int(seeds[median_seed_index]),
        "primary_gate": primary,
        "tail_gate": tail,
        "records": records,
        "formal_validation_or_test_consumed": False,
    }
    summary_path = output / "summary.json"
    dump_json(summary_path, summary)
    manifest = {
        "schema_version": "query-expected-road-actor-oof-v1",
        "dataset_type": "train-only-query-actor-nested-episode-oof",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "source_validation_sha256": sha256(source / "validation.json"),
        "source_bank_sha256": source_manifest["bank_sha256"],
        "query_checkpoint_sha256": source_manifest["query_checkpoint_sha256"],
        "projected_teacher_sha256": sha256(projected_path),
        "oof_predictions_sha256": sha256(oof_path),
        "summary_sha256": sha256(summary_path),
        "checkpoint_sha256": {
            Path(record["checkpoint"]).name: record["checkpoint_sha256"]
            for record in records
        },
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "limitations": [
            "This is open-loop direct-center OOF, not actor-visited Replay or closed loop.",
            "The 40-kph source is not direction-balanced.",
            "A single Actor center never removes the mandatory exact-warm two-center guard.",
        ],
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps({"qualification": qualification, "recoveries": recoveries.tolist(), "tail": tail}, indent=2))


if __name__ == "__main__":
    main()
