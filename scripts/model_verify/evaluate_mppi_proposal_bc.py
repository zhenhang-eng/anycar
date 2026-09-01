#!/usr/bin/env python3
"""Evaluate a BC proposal policy against warm and T1 teacher with fresh DBM seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TorchMPPIProposalPolicy,
    ego_reference_features,
)
from generate_dbm_multicenter_teacher import (
    evaluate_knots,
    load_config,
    make_controller,
    proposal_evaluation,
)


DEFAULT_TRAINING = Path("outputs/mppi_proposal/bc_t1_conv_v1/training_summary.json")
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/bc_t1_conv_v1/offline_dbm_eval_v1")
METHODS = ("warm", "network", "teacher")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "Optional source-collection override. This allows a frozen older "
            "checkpoint to be evaluated on a newer held-out collection."
        ),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help=(
            "Optional T1-label override paired with --source. Episode splits are "
            "always read from the selected label sidecar."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="14001,14002,14003")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help=(
            "Optional candidate count per center and seed; zero preserves the "
            "T1 teacher configuration."
        ),
    )
    parser.add_argument(
        "--splits",
        default="train,validation,test",
        help="Comma-separated episode splits to evaluate.",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=0,
        help="Optional limit after split filtering; zero evaluates all snapshots.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_policy(
    checkpoint_path: Path, device: torch.device
) -> tuple[TorchMPPIProposalPolicy, MPPIProposalNormalization, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("model_type") != "TorchMPPIProposalPolicy":
        raise ValueError("checkpoint is not a TorchMPPIProposalPolicy")
    architecture = checkpoint["architecture"]
    model = TorchMPPIProposalPolicy(
        trust_scale=tuple(architecture["trust_scale"]),
        dropout=float(architecture["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    normalization = MPPIProposalNormalization.from_dict(
        checkpoint["normalization"]
    )
    return model, normalization, checkpoint


@torch.no_grad()
def predict_center(
    model: TorchMPPIProposalPolicy,
    normalization: MPPIProposalNormalization,
    source: np.lib.npyio.NpzFile,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(source["initial_state"], dtype=np.float32)
    action = np.asarray(source["current_action"], dtype=np.float32)
    history = np.asarray(source["history"][0], dtype=np.float32)
    reference = ego_reference_features(source["reference_ego"], float(state[3]))
    current = np.asarray((state[3], state[4], *action), dtype=np.float32)
    history, reference, current = normalization.normalize_numpy(
        history, reference, current
    )
    warm = np.asarray(source["sampling_mean_knots"], dtype=np.float32)
    predicted_delta, predicted_center = model(
        torch.from_numpy(history[None]).to(device),
        torch.from_numpy(reference[None]).to(device),
        torch.from_numpy(current[None]).to(device),
        torch.from_numpy(warm[None]).to(device),
    )
    return (
        predicted_delta[0].cpu().numpy(),
        predicted_center[0].cpu().numpy(),
    )


def comparison(base: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    gain = base - candidate
    return {
        "gain_mean": float(gain.mean()),
        "gain_median": float(np.median(gain)),
        "gain_p10": float(np.quantile(gain, 0.10)),
        "gain_p90": float(np.quantile(gain, 0.90)),
        "relative_gain_mean": float(np.mean(gain / np.maximum(np.abs(base), 1e-6))),
        "wins": int(np.sum(gain > 1e-6)),
        "ties": int(np.sum(np.abs(gain) <= 1e-6)),
        "losses": int(np.sum(gain < -1e-6)),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"snapshot_count": len(rows), "methods": {}}
    metric_names = (
        "weighted_output_cost",
        "best_cost",
        "p10_cost",
        "softmin_cost",
        "median_cost",
        "effective_sample_size",
        "clip_fraction",
        "direct_cost",
        "selection_score",
    )
    values: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        values[method] = {
            metric: np.asarray([row[f"{method}_{metric}"] for row in rows])
            for metric in metric_names
        }
        summary["methods"][method] = {
            metric: {
                "mean": float(array.mean()),
                "median": float(np.median(array)),
            }
            for metric, array in values[method].items()
        }
    lower_better = (
        "weighted_output_cost",
        "best_cost",
        "p10_cost",
        "softmin_cost",
        "median_cost",
        "clip_fraction",
        "direct_cost",
        "selection_score",
    )
    summary["comparisons"] = {}
    for candidate, base in (
        ("network", "warm"),
        ("network", "teacher"),
        ("teacher", "warm"),
    ):
        name = f"{candidate}_vs_{base}"
        summary["comparisons"][name] = {
            metric: comparison(values[base][metric], values[candidate][metric])
            for metric in lower_better
        }
    return summary


def main() -> None:
    args = parse_args()
    training_summary = json.loads(args.training_summary.read_text())
    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else Path(training_summary["selected_checkpoint"])
    )
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    model, normalization, checkpoint = load_policy(checkpoint_path, device)
    if (args.source is None) != (args.labels is None):
        raise ValueError("--source and --labels must be provided together")
    source_root = (
        args.source.resolve()
        if args.source is not None
        else Path(checkpoint["source_collection"]).resolve()
    )
    label_root = (
        args.labels.resolve()
        if args.labels is not None
        else Path(checkpoint["teacher_labels"]).resolve()
    )
    config = load_config(label_root / "teacher_config.json")
    teacher_label_num_samples = int(config["proposal_evaluation"]["num_samples"])
    if args.num_samples:
        if args.num_samples < 4 or args.num_samples % 2:
            raise ValueError("--num-samples must be an even integer >= 4")
        config["proposal_evaluation"]["num_samples"] = args.num_samples
    existing_seeds = set(config["proposal_evaluation"]["seeds"]) | set(
        config["proposal_evaluation"]["audit_seeds"]
    )
    if existing_seeds & set(seeds):
        raise ValueError("evaluation seeds must be disjoint from T1 selection/audit")
    label_splits = json.loads((label_root / "splits.json").read_text())
    splits = {
        name: episodes
        for name, episodes in label_splits.items()
        if name in ("train", "validation", "test")
    }
    split_by_episode = {
        episode: split for split, episodes in splits.items() for episode in episodes
    }
    requested_splits = {
        value.strip() for value in args.splits.split(",") if value.strip()
    }
    unknown_splits = requested_splits - set(splits)
    if unknown_splits:
        raise ValueError(f"unknown splits: {sorted(unknown_splits)}")
    label_paths = [
        path
        for path in sorted(label_root.glob("episode_*/*.npz"))
        if split_by_episode[path.parent.name] in requested_splits
    ]
    if args.max_snapshots > 0:
        label_paths = label_paths[: args.max_snapshots]
    if not label_paths:
        raise ValueError("no snapshots selected for evaluation")
    rows: list[dict[str, Any]] = []
    saved: dict[str, list[np.ndarray]] = {
        "centers": [],
        "predicted_delta": [],
        "weighted_output_cost": [],
        "p10_cost": [],
        "softmin_cost": [],
        "best_cost": [],
        "median_cost": [],
        "effective_sample_size": [],
        "clip_fraction": [],
        "direct_cost": [],
        "selection_score": [],
    }
    for index, label_path in enumerate(label_paths, start=1):
        episode_id = label_path.parent.name
        source_path = source_root / episode_id / "snapshots" / label_path.name
        with np.load(source_path, allow_pickle=False) as source, np.load(
            label_path, allow_pickle=False
        ) as label:
            predicted_delta, network_center = predict_center(
                model, normalization, source, device
            )
            warm = np.asarray(source["sampling_mean_knots"], dtype=np.float32)
            teacher = np.asarray(label["teacher_center_knots"], dtype=np.float32)
            centers = np.asarray((warm, network_center, teacher), dtype=np.float32)
            controller, backend = make_controller(source, config, device)
            history = torch.from_numpy(source["history"]).to(device)
            state = torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5)
            current_action = torch.from_numpy(source["current_action"]).to(device).reshape(
                1, 2
            )
            reference = controller._prepare_reference(source["reference"])
            params = json.loads(str(source["mppi_params_json"]))
            sigma = np.asarray(params["noise_sigma"], dtype=np.float32)
            action_min = np.asarray(params["action_min"], dtype=np.float32)
            action_max = np.asarray(params["action_max"], dtype=np.float32)
            evaluation = proposal_evaluation(
                centers,
                config,
                controller,
                backend,
                history,
                state,
                current_action,
                reference,
                warm,
                sigma,
                action_min,
                action_max,
                evaluation_seeds=seeds,
            )
            direct_cost, _, _ = evaluate_knots(
                controller,
                backend,
                centers,
                history,
                state,
                current_action,
                reference,
            )
            row: dict[str, Any] = {
                "episode_id": episode_id,
                "split": split_by_episode[episode_id],
                "control_step": int(source["control_step"]),
            }
            metric_map = {
                "weighted_output_cost": evaluation[
                    "proposal_weighted_output_cost"
                ].mean(axis=1),
                "best_cost": evaluation["proposal_best_cost"].mean(axis=1),
                "p10_cost": evaluation["proposal_p10_cost"].mean(axis=1),
                "softmin_cost": evaluation["proposal_softmin_cost"].mean(axis=1),
                "median_cost": evaluation["proposal_median_cost"].mean(axis=1),
                "effective_sample_size": evaluation[
                    "proposal_effective_sample_size"
                ].mean(axis=1),
                "clip_fraction": evaluation["proposal_clip_fraction"].mean(axis=1),
                "direct_cost": direct_cost,
                "selection_score": evaluation["selection_score"],
            }
            for method_index, method in enumerate(METHODS):
                for metric_name, metric_values in metric_map.items():
                    row[f"{method}_{metric_name}"] = float(
                        metric_values[method_index]
                    )
            rows.append(row)
            saved["centers"].append(centers)
            saved["predicted_delta"].append(predicted_delta)
            for metric_name, metric_values in metric_map.items():
                saved[metric_name].append(np.asarray(metric_values))
        if index % 12 == 0:
            print(f"[{index:03d}/{len(label_paths):03d}] evaluated")
    with (args.output_dir / "per_snapshot.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output_dir / "evaluation_arrays.npz",
        method_names=np.asarray(METHODS),
        evaluation_seeds=np.asarray(seeds, dtype=np.int64),
        **{name: np.asarray(values) for name, values in saved.items()},
    )
    by_split = {
        split: aggregate(split_rows)
        for split in ("train", "validation", "test")
        if (split_rows := [row for row in rows if row["split"] == split])
    }
    summary = {
        "format_version": 1,
        "checkpoint": str(checkpoint_path),
        "model_parameter_count": model.parameter_count,
        "evaluation_seeds": seeds,
        "candidate_count_per_center_seed": config["proposal_evaluation"][
            "num_samples"
        ],
        "teacher_label_candidate_count_per_center_seed": teacher_label_num_samples,
        "source_collection": str(source_root),
        "teacher_labels": str(label_root),
        "checkpoint_source_collection": checkpoint["source_collection"],
        "checkpoint_teacher_labels": checkpoint["teacher_labels"],
        "overall": aggregate(rows),
        "by_split": by_split,
        "interpretation_warning": (
            "Teacher labels are optimized independently per snapshot; only the network "
            "tests cross-episode generalization."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
