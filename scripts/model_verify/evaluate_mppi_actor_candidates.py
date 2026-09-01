#!/usr/bin/env python3
"""Evaluate several proposal actors together with common fixed-DBM noise."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_mppi_proposal_bc import load_policy, predict_center
from generate_dbm_multicenter_teacher import load_config, make_controller
from generate_dbm_proposal_critic_labels import batched_proposal_evaluation


DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1"
)
DEFAULT_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_critic_local_diverse_20260805_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--checkpoints",
        required=True,
        help="Comma-separated name=checkpoint entries; names must be unique.",
    )
    parser.add_argument("--splits", default="validation")
    parser.add_argument("--seeds", default="22001,22002")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def parse_checkpoints(text: str) -> list[tuple[str, Path]]:
    values = []
    for item in text.split(","):
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            raise ValueError("each checkpoint must use name=path")
        values.append((name, Path(path).resolve()))
    if len({name for name, _ in values}) != len(values):
        raise ValueError("checkpoint names must be unique")
    if {name for name, _ in values} & {"warm", "teacher"}:
        raise ValueError("warm and teacher are reserved method names")
    return values


def comparison(base: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    gain = base - candidate
    return {
        "base_cost_mean": float(base.mean()),
        "candidate_cost_mean": float(candidate.mean()),
        "gain_mean": float(gain.mean()),
        "gain_median": float(np.median(gain)),
        "gain_p10": float(np.quantile(gain, 0.10)),
        "gain_p90": float(np.quantile(gain, 0.90)),
        "wins": int(np.sum(gain > 1e-6)),
        "ties": int(np.sum(np.abs(gain) <= 1e-6)),
        "losses": int(np.sum(gain < -1e-6)),
        "loss_over_2": int(np.sum(gain < -2.0)),
        "loss_over_5": int(np.sum(gain < -5.0)),
        "worst_gain": float(gain.min()),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    checkpoint_specs = parse_checkpoints(args.checkpoints)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    requested_splits = {value for value in args.splits.split(",") if value}
    device = torch.device(args.device)
    policies = []
    checkpoint_metadata = {}
    for name, path in checkpoint_specs:
        policy, normalization, checkpoint = load_policy(path, device)
        policies.append((name, policy, normalization))
        checkpoint_metadata[name] = {
            "path": str(path),
            "parent_actor_checkpoint": checkpoint.get("parent_actor_checkpoint"),
            "fine_tuning": checkpoint.get("fine_tuning"),
        }
    source_root = args.source.resolve()
    label_root = args.labels.resolve()
    split_payload = json.loads((label_root / "splits.json").read_text())
    unknown = requested_splits - {"train", "validation", "test"}
    if unknown:
        raise ValueError(f"unknown splits {sorted(unknown)}")
    split_by_episode = {
        episode: split
        for split in ("train", "validation", "test")
        for episode in split_payload[split]
    }
    label_paths = [
        path
        for path in sorted(label_root.glob("episode_*/*.npz"))
        if split_by_episode[path.parent.name] in requested_splits
    ]
    if not label_paths:
        raise ValueError("no snapshots selected")
    config = load_config(label_root / "teacher_config.json")
    stored_seeds = set(config["proposal_evaluation"]["seeds"]) | set(
        config["proposal_evaluation"]["audit_seeds"]
    )
    local_summary = json.loads((label_root / "summary.json").read_text())
    stored_seeds |= set(local_summary["selection_seeds"]) | set(
        local_summary["audit_seeds"]
    )
    if stored_seeds & set(seeds):
        raise ValueError("evaluation seeds overlap stored teacher/local label seeds")
    method_names = ["warm", *[name for name, _ in checkpoint_specs], "teacher"]
    rows = []
    center_values = []
    metric_values: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "weighted_output_cost",
            "best_cost",
            "p10_cost",
            "median_cost",
            "softmin_cost",
            "effective_sample_size",
            "clip_fraction",
        )
    }
    for index, label_path in enumerate(label_paths, start=1):
        episode_id = label_path.parent.name
        source_path = source_root / episode_id / "snapshots" / label_path.name
        with np.load(source_path, allow_pickle=False) as source, np.load(
            label_path, allow_pickle=False
        ) as label:
            warm = np.asarray(source["sampling_mean_knots"], dtype=np.float32)
            centers = [warm]
            for _, policy, normalization in policies:
                centers.append(predict_center(policy, normalization, source, device)[1])
            centers.append(np.asarray(label["teacher_center_knots"], dtype=np.float32))
            centers_array = np.asarray(centers, dtype=np.float32)
            params = json.loads(str(source["mppi_params_json"]))
            sigma = np.asarray(params["noise_sigma"], dtype=np.float32)
            action_min = np.asarray(params["action_min"], dtype=np.float32)
            action_max = np.asarray(params["action_max"], dtype=np.float32)
            controller, backend = make_controller(source, config, device)
            evaluation = batched_proposal_evaluation(
                centers_array,
                config,
                controller,
                backend,
                torch.from_numpy(source["history"]).to(device),
                torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5),
                torch.from_numpy(source["current_action"]).to(device).reshape(1, 2),
                controller._prepare_reference(source["reference"]),
                warm,
                sigma,
                action_min,
                action_max,
                seeds,
            )
            row: dict[str, Any] = {
                "episode_id": episode_id,
                "split": split_by_episode[episode_id],
                "control_step": int(source["control_step"]),
            }
            field_map = {
                "weighted_output_cost": "proposal_weighted_output_cost",
                "best_cost": "proposal_best_cost",
                "p10_cost": "proposal_p10_cost",
                "median_cost": "proposal_median_cost",
                "softmin_cost": "proposal_softmin_cost",
                "effective_sample_size": "proposal_effective_sample_size",
                "clip_fraction": "proposal_clip_fraction",
            }
            for metric_name, field in field_map.items():
                values = evaluation[field].mean(axis=1)
                metric_values[metric_name].append(values)
                for method_index, method in enumerate(method_names):
                    row[f"{method}_{metric_name}"] = float(values[method_index])
            rows.append(row)
            center_values.append(centers_array)
        if index % 20 == 0 or index == len(label_paths):
            print(f"[{index:04d}/{len(label_paths):04d}] actor candidates evaluated")
    with (args.output_dir / "per_snapshot.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    arrays = {name: np.asarray(value) for name, value in metric_values.items()}
    np.savez_compressed(
        args.output_dir / "evaluation_arrays.npz",
        method_names=np.asarray(method_names),
        evaluation_seeds=np.asarray(seeds, dtype=np.int64),
        centers=np.asarray(center_values),
        **arrays,
    )
    costs = arrays["weighted_output_cost"]
    method_index = {name: index for index, name in enumerate(method_names)}
    summary = {
        "format_version": 1,
        "source_collection": str(source_root),
        "labels": str(label_root),
        "splits": sorted(requested_splits),
        "snapshot_count": len(rows),
        "evaluation_seeds": seeds,
        "candidate_count_per_center_seed": int(
            config["proposal_evaluation"]["num_samples"]
        ),
        "method_names": method_names,
        "checkpoints": checkpoint_metadata,
        "method_cost_mean": {
            name: float(costs[:, index].mean())
            for index, name in enumerate(method_names)
        },
        "comparisons_vs_warm": {
            name: comparison(costs[:, method_index["warm"]], costs[:, index])
            for index, name in enumerate(method_names)
            if name != "warm"
        },
        "comparisons_vs_bc": {
            name: comparison(costs[:, method_index["bc"]], costs[:, index])
            for index, name in enumerate(method_names)
            if "bc" in method_index and name not in {"warm", "bc"}
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
