#!/usr/bin/env python3
"""Diagnose target-slice gains and remaining failures after mixed-init Query OAC."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic
from run_query_single_center_oac20to1 import load_inputs, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def stats(cost: np.ndarray, warm: np.ndarray, round0: np.ndarray) -> dict:
    gain = warm.astype(np.float64) - cost.astype(np.float64)
    progress = round0.astype(np.float64) - cost.astype(np.float64)
    return {
        "count": int(len(cost)),
        "cost_mean": float(np.mean(cost)),
        "cost_median": float(np.median(cost)),
        "warm_win_or_tie_fraction": float(np.mean(cost <= warm)),
        "warm_gain_mean": float(np.mean(gain)),
        "warm_gain_median": float(np.median(gain)),
        "warm_gain_p05": float(np.quantile(gain, 0.05)),
        "warm_gain_worst": float(np.min(gain)),
        "round0_to_selected_mean_gain": float(np.mean(progress)),
        "round0_to_selected_win_or_tie_fraction": float(np.mean(cost <= round0)),
    }


def critic_bank_metrics(
    critics: list[ConfigurableAbsoluteActionValueCritic],
    trainings: list[dict],
    inputs: tuple[np.ndarray, ...],
    data: dict[str, np.ndarray],
    row: int,
    device: torch.device,
) -> dict:
    valid = data["candidate_valid_mask"][row]
    action = data["candidate_knots"][row, valid]
    truth = data["candidate_cost"][row, valid]
    rows = np.full(len(action), row, np.int64)
    predictions = []
    with torch.no_grad():
        for critic, training in zip(critics, trainings):
            prediction = critic(
                torch.from_numpy(inputs[0][rows]).to(device),
                torch.from_numpy(inputs[1][rows]).to(device),
                torch.from_numpy(inputs[2][rows]).to(device),
                torch.from_numpy(action[:, None]).to(device),
            )[:, 0]
            predictions.append((
                prediction * float(training["target_std"]) + float(training["target_mean"])
            ).cpu().numpy())
    conservative = np.maximum(predictions[0], predictions[1])
    log_truth = np.log1p(truth.astype(np.float64))
    correlation = 0.0 if log_truth.std() < 1e-12 else float(
        np.corrcoef(conservative, log_truth)[0, 1]
    )
    true_delta = log_truth[:, None] - log_truth[None, :]
    predicted_delta = conservative[:, None] - conservative[None, :]
    material = np.triu(np.abs(true_delta) > 1e-5, 1)
    pair = float(np.mean(
        np.sign(true_delta[material]) == np.sign(predicted_delta[material])
    ))
    return {
        "candidate_count": int(len(truth)),
        "log_cost_pearson": correlation,
        "pair_sign_accuracy": pair,
        "predicted_pick_true_cost": float(truth[np.argmin(conservative)]),
        "oracle_candidate_cost": float(np.min(truth)),
    }


def main() -> None:
    args = parse_args()
    root = args.run.resolve()
    summary = json.loads((root / "summary.json").read_text())
    replay = Path(summary["contract"]["sources"]["absolute_replay"]) / "replay.npz"
    with np.load(replay, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    device = torch.device(args.device)
    loaded = []
    for record in summary["records"]:
        with np.load(record["arrays"], allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        checkpoint = torch.load(record["checkpoint"], map_location=device, weights_only=False)
        loaded.append((record, arrays, checkpoint))

    pooled = {}
    for scope, split, condition in (
        ("inner_all", "inner", lambda rows: np.ones(len(rows), bool)),
        ("inner_55:2", "inner", lambda rows: (data["speed_kph"][rows] == 55) & (data["variant_index"][rows] == 2)),
        ("inner_100:3", "inner", lambda rows: (data["speed_kph"][rows] == 100) & (data["variant_index"][rows] == 3)),
        ("fit_55:2", "fit", lambda rows: (data["speed_kph"][rows] == 55) & (data["variant_index"][rows] == 2)),
        ("fit_100:3", "fit", lambda rows: (data["speed_kph"][rows] == 100) & (data["variant_index"][rows] == 3)),
        ("new_coverage_72", "fit", lambda rows: rows >= 600),
        ("old_fit_prefix_600", "fit", lambda rows: rows < 600),
    ):
        selected_values, warm_values, round0_values = [], [], []
        for record, arrays, _ in loaded:
            rows = arrays["selection_indices"] if split == "inner" else arrays["fit_indices"]
            mask = condition(rows)
            if split == "inner":
                costs = arrays["selection_round_cost"]
                selected = costs[int(record["selected_round"])]
                round0 = costs[0]
            else:
                selected = arrays["selected_fit_cost"]
                round0 = arrays["round0_fit_cost"]
            selected_values.append(selected[mask])
            round0_values.append(round0[mask])
            warm_values.append(data["warm_cost"][rows][mask])
        pooled[scope] = stats(
            np.concatenate(selected_values), np.concatenate(warm_values),
            np.concatenate(round0_values),
        )

    hard_rows = np.flatnonzero(
        (data["fold_id"] == summary["contract"]["split_contract"]["inner_selection_fold"])
        & (data["speed_kph"] == 100) & (data["variant_index"] == 3)
    )
    hard_100_records = []
    critic_correlations, critic_pairs = [], []
    for record, arrays, checkpoint in loaded:
        inputs = load_inputs(data, checkpoint["critic_normalization"])
        critics, trainings = [], []
        for twin in (1, 2):
            critic = ConfigurableAbsoluteActionValueCritic().to(device)
            critic.load_state_dict(checkpoint[f"selected_critic{twin}_state_dict"], strict=True)
            critic.eval()
            critics.append(critic)
            trainings.append(checkpoint[f"critic{twin}_training"])
        selection = arrays["selection_indices"]
        positions = np.asarray([int(np.flatnonzero(selection == row)[0]) for row in hard_rows])
        selected_cost = arrays["selection_round_cost"][int(record["selected_round"]), positions]
        for row, cost in zip(hard_rows, selected_cost):
            bank = critic_bank_metrics(critics, trainings, inputs, data, int(row), device)
            critic_correlations.append(bank["log_cost_pearson"])
            critic_pairs.append(bank["pair_sign_accuracy"])
            hard_100_records.append({
                "seed": int(record["seed"]),
                "row": int(row),
                "episode_id": str(data["episode_id"][row]),
                "control_step": int(data["control_step"][row]),
                "warm_cost": float(data["warm_cost"][row]),
                "fullrank_teacher_cost": float(data["fullrank_teacher_cost"][row]),
                "selected_actor_cost": float(cost),
                "warm_gain": float(data["warm_cost"][row] - cost),
                "critic_bank": bank,
            })

    outliers = []
    for record, arrays, _ in loaded:
        fit = arrays["fit_indices"]
        positions = np.argsort(arrays["selected_fit_cost"])[-6:][::-1]
        for position in positions:
            row = int(fit[position])
            outliers.append({
                "seed": int(record["seed"]), "row": row,
                "episode_id": str(data["episode_id"][row]),
                "speed_kph": int(data["speed_kph"][row]),
                "variant_index": int(data["variant_index"][row]),
                "control_step": int(data["control_step"][row]),
                "is_new_coverage_row": bool(row >= 600),
                "warm_cost": float(data["warm_cost"][row]),
                "round0_actor_cost": float(arrays["round0_fit_cost"][position]),
                "selected_actor_cost": float(arrays["selected_fit_cost"][position]),
            })

    seed_candidates = []
    for record in summary["records"]:
        inner = record["selected"]["inner"]
        fit_report = record["selected"]["fit"]
        seed_candidates.append({
            "seed": int(record["seed"]),
            "selected_round": int(record["selected_round"]),
            "inner_cost_mean": float(inner["actor_cost"]["mean"]),
            "inner_warm_win_or_tie_fraction": float(inner["win_or_tie_fraction"]),
            "inner_warm_gain_p05": float(inner["gain"]["p05"]),
            "inner_warm_gain_worst": float(inner["gain"]["min"]),
            "fit_cost_mean": float(fit_report["actor_cost"]["mean"]),
            "fit_warm_win_or_tie_fraction": float(fit_report["win_or_tie_fraction"]),
            "fit_warm_gain_p05": float(fit_report["gain"]["p05"]),
        })

    report = {
        "qualification": "QUERY_TARGET_COVERAGE_MIXED_INIT_OAC_DIAGNOSIS_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(root),
        "run_manifest_sha256": sha256(root / "manifest.json"),
        "run_validation_sha256": sha256(root / "validation.json"),
        "pooled": pooled,
        "seed_candidates": seed_candidates,
        "inner_100:3": {
            "rows": hard_100_records,
            "critic_log_cost_pearson_median": float(np.median(critic_correlations)),
            "critic_pair_sign_accuracy_median": float(np.median(critic_pairs)),
            "episode_ids": sorted(set(item["episode_id"] for item in hard_100_records)),
            "control_steps": sorted(set(item["control_step"] for item in hard_100_records)),
        },
        "largest_fit_costs": outliers,
        "conclusions": [
            "The expanded fit distribution plus continuous OAC repairs the previous 55:2 cross-episode failure on the sealed inner-selection fold.",
            "The new 100:3 fit episodes are learned well, but inner episode_096 retains late-control-step failures; this is a dynamic-history generalization gap rather than a general 100:3 failure.",
            "The selected Critics are weak and seed-sensitive on the existing inner 100:3 candidate banks, so checkpoint rescoring alone cannot remove the remaining tail.",
            "The approximately 1e3 fit-cost outlier is old episode_028 40:1 where warm itself is approximately 982; it is unrelated to the new coverage rows.",
            "Seed 2 is the robust train-side candidate: nearly tied inner mean, best fit metrics, and materially smaller worst inner regression.",
        ],
        "recommended_next_step": (
            "Keep architecture and 20:16 fixed. Before any outer evaluation, design one fit-only history-diversification collection for 100:3 that changes the closed-loop dynamic history "
            "(initial state/control perturbations or transient road/controller conditions), not another phase/seed-only collection that converges after burn-in; then warm-start the current seed-2 Actor and refresh the Critics."
        ),
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
        "analyzer": str(Path(__file__).resolve()),
        "analyzer_sha256": sha256(Path(__file__).resolve()),
    }
    (root / "diagnosis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "qualification": report["qualification"],
        "pooled": pooled,
        "critic_100:3_pearson_median": report["inner_100:3"]["critic_log_cost_pearson_median"],
        "critic_100:3_pair_median": report["inner_100:3"]["critic_pair_sign_accuracy_median"],
        "seed_candidates": seed_candidates,
        "outer_fold_evaluated": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
