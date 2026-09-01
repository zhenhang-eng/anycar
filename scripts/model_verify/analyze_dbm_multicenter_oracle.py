#!/usr/bin/env python3
"""Evaluate multi-elite DBM oracle bounds and fixed-budget center mixtures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from generate_dbm_multicenter_teacher import (
    evaluate_actions,
    evaluate_knots,
    load_config,
    make_controller,
    proposal_evaluation,
    stable_seeded_noise,
    stable_weight,
)


DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_expansion_20260804_v3"
)
DEFAULT_T1 = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_teacher_t1_expansion_20260804_v1"
)
DEFAULT_MULTI = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_multi_elite_expansion_20260804_v1"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/dbm_multi_elite_oracle_20260804_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--t1-labels", type=Path, default=DEFAULT_T1)
    parser.add_argument("--multi-labels", type=Path, default=DEFAULT_MULTI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="15001,15002,15003")
    parser.add_argument("--splits", default="test")
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def fixed_budget_evaluation(
    centers: np.ndarray,
    seed: int,
    total_samples: int,
    temperature: float,
    controller,
    backend,
    history: torch.Tensor,
    state: torch.Tensor,
    current_action: torch.Tensor,
    reference: torch.Tensor,
    sigma: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    center_count = len(centers)
    base = total_samples // center_count
    remainder = total_samples % center_count
    allocations = [base + (index < remainder) for index in range(center_count)]
    noise = stable_seeded_noise(seed, total_samples, centers[0].shape, sigma)
    offset = 0
    all_cost = []
    all_actions = []
    center_weighted_actions = []
    for center, count in zip(centers, allocations):
        center_noise = noise[offset : offset + count].copy()
        center_noise[0] = 0.0
        offset += count
        raw_knots = center[None] + center_noise
        knots = np.clip(raw_knots, action_min, action_max).astype(np.float32)
        cost, actions, _ = evaluate_knots(
            controller,
            backend,
            knots,
            history,
            state,
            current_action,
            reference,
        )
        weight = stable_weight(cost, temperature)
        center_weighted_actions.append(
            np.sum(weight[:, None, None] * actions, axis=0).astype(np.float32)
        )
        all_cost.append(cost)
        all_actions.append(actions)
    center_output_cost, _ = evaluate_actions(
        controller,
        backend,
        np.asarray(center_weighted_actions, dtype=np.float32),
        history,
        state,
        current_action,
        reference,
    )
    combined_cost = np.concatenate(all_cost)
    combined_actions = np.concatenate(all_actions)
    combined_weight = stable_weight(combined_cost, temperature)
    mixture_action = np.sum(
        combined_weight[:, None, None] * combined_actions, axis=0
    ).astype(np.float32)
    mixture_cost, _ = evaluate_actions(
        controller,
        backend,
        mixture_action[None],
        history,
        state,
        current_action,
        reference,
    )
    return (
        center_output_cost.astype(np.float32),
        float(mixture_cost[0]),
        float(np.min(combined_cost)),
        float(np.quantile(combined_cost, 0.10)),
    )


def comparison(base: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    gain = base - candidate
    return {
        "gain_mean": float(gain.mean()),
        "gain_median": float(np.median(gain)),
        "gain_p10": float(np.quantile(gain, 0.10)),
        "gain_p90": float(np.quantile(gain, 0.90)),
        "wins": int(np.sum(gain > 1e-6)),
        "ties": int(np.sum(np.abs(gain) <= 1e-6)),
        "losses": int(np.sum(gain < -1e-6)),
    }


def aggregate(rows: list[dict[str, Any]], methods: tuple[str, ...]) -> dict[str, Any]:
    values = {
        method: np.asarray([row[f"{method}_cost"] for row in rows], dtype=np.float64)
        for method in methods
    }
    return {
        "snapshot_count": len(rows),
        "methods": {
            method: {
                "mean": float(cost.mean()),
                "median": float(np.median(cost)),
                "p90": float(np.quantile(cost, 0.90)),
            }
            for method, cost in values.items()
        },
        "versus_warm": {
            method: comparison(values["warm"], values[method])
            for method in methods
            if method != "warm"
        },
        "versus_teacher": {
            method: comparison(values["teacher"], values[method])
            for method in methods
            if method not in ("warm", "teacher")
        },
    }


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    t1_root = args.t1_labels.resolve()
    multi_root = args.multi_labels.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one fresh evaluation seed is required")
    t1_config = load_config(t1_root / "teacher_config.json")
    prior_seeds = set(t1_config["proposal_evaluation"]["seeds"]) | set(
        t1_config["proposal_evaluation"]["audit_seeds"]
    ) | {14001, 14002, 14003}
    if prior_seeds & set(seeds):
        raise ValueError("oracle seeds must be disjoint from T1 and prior BC evaluation")
    split_data = json.loads((multi_root / "splits.json").read_text())
    split_by_episode = {
        episode: split
        for split, episodes in split_data.items()
        if split in ("train", "validation", "test")
        for episode in episodes
    }
    requested_splits = {
        value.strip() for value in args.splits.split(",") if value.strip()
    }
    unknown = requested_splits - {"train", "validation", "test"}
    if unknown:
        raise ValueError(f"unknown splits: {sorted(unknown)}")
    label_paths = [
        path
        for path in sorted(multi_root.glob("episode_*/*.npz"))
        if split_by_episode[path.parent.name] in requested_splits
    ]
    if args.max_snapshots > 0:
        label_paths = label_paths[: args.max_snapshots]
    if not label_paths:
        raise ValueError("no multi-elite labels selected")
    max_elites = int(
        json.loads((multi_root / "multi_elite_config.json").read_text())["max_elites"]
    )
    methods = ["warm", "teacher"]
    for count in range(1, max_elites + 1):
        methods.extend(
            (
                f"elite_mean_k{count}",
                f"full_budget_state_oracle_k{count}",
                f"full_budget_seed_oracle_k{count}",
                f"fixed_budget_state_oracle_k{count}",
                f"fixed_budget_seed_oracle_k{count}",
                f"fixed_budget_mixture_k{count}",
            )
        )
    methods.extend(("teacher_scale_25", "teacher_scale_50", "teacher_scale_75"))
    method_names = tuple(methods)
    rows: list[dict[str, Any]] = []
    saved_costs: dict[str, list[np.ndarray]] = {name: [] for name in method_names}
    device = torch.device(args.device)
    total_samples = int(t1_config["proposal_evaluation"]["num_samples"])
    temperature = float(t1_config["objective"]["temperature"])
    for snapshot_index, multi_path in enumerate(label_paths, start=1):
        episode_id = multi_path.parent.name
        source_path = source_root / episode_id / "snapshots" / multi_path.name
        t1_path = t1_root / episode_id / multi_path.name
        with np.load(source_path, allow_pickle=False) as source, np.load(
            t1_path, allow_pickle=False
        ) as t1, np.load(multi_path, allow_pickle=False) as multi:
            valid_count = int(multi["elite_count"])
            elites = np.asarray(multi["elite_centers"][:valid_count], dtype=np.float32)
            teacher = np.asarray(t1["teacher_center_knots"], dtype=np.float32)
            if not np.allclose(elites[0], teacher, rtol=0, atol=1e-7):
                raise AssertionError(f"{multi_path}: elite zero differs from T1 teacher")
            warm = np.asarray(source["sampling_mean_knots"], dtype=np.float32)
            controller, backend = make_controller(source, t1_config, device)
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
            elite_means = np.asarray(
                [
                    elites[: min(count, valid_count)].mean(axis=0)
                    for count in range(1, max_elites + 1)
                ],
                dtype=np.float32,
            )
            scaled_teacher = np.asarray(
                [warm + scale * (teacher - warm) for scale in (0.25, 0.50, 0.75)],
                dtype=np.float32,
            )
            full_centers = np.concatenate(
                (warm[None], elites, elite_means, scaled_teacher), axis=0
            )
            full = proposal_evaluation(
                full_centers,
                t1_config,
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
            full_cost = np.asarray(full["proposal_weighted_output_cost"], dtype=np.float64)
            per_method_seed_cost: dict[str, np.ndarray] = {
                "warm": full_cost[0],
                "teacher": full_cost[1],
            }
            mean_offset = 1 + valid_count
            scale_offset = mean_offset + max_elites
            for requested_count in range(1, max_elites + 1):
                per_method_seed_cost[f"elite_mean_k{requested_count}"] = full_cost[
                    mean_offset + requested_count - 1
                ]
            for scale_index, scale_name in enumerate(
                ("teacher_scale_25", "teacher_scale_50", "teacher_scale_75")
            ):
                per_method_seed_cost[scale_name] = full_cost[scale_offset + scale_index]
            fixed_center_costs: dict[int, list[np.ndarray]] = {
                count: [] for count in range(1, max_elites + 1)
            }
            fixed_mixture_costs: dict[int, list[float]] = {
                count: [] for count in range(1, max_elites + 1)
            }
            for seed_index, seed in enumerate(seeds):
                fixed_cache: dict[int, tuple[np.ndarray, float]] = {
                    1: (full_cost[1:2, seed_index].astype(np.float32), float(full_cost[1, seed_index]))
                }
                for requested_count in range(1, max_elites + 1):
                    effective_count = min(requested_count, valid_count)
                    if effective_count not in fixed_cache:
                        center_cost, mixture_cost, _, _ = fixed_budget_evaluation(
                            elites[:effective_count],
                            seed,
                            total_samples,
                            temperature,
                            controller,
                            backend,
                            history,
                            state,
                            current_action,
                            reference,
                            sigma,
                            action_min,
                            action_max,
                        )
                        fixed_cache[effective_count] = (center_cost, mixture_cost)
                    center_cost, mixture_cost = fixed_cache[effective_count]
                    fixed_center_costs[requested_count].append(center_cost)
                    fixed_mixture_costs[requested_count].append(mixture_cost)
            for requested_count in range(1, max_elites + 1):
                effective_count = min(requested_count, valid_count)
                full_elite_cost = full_cost[1 : 1 + effective_count]
                full_state_index = int(np.argmin(full_elite_cost.mean(axis=1)))
                per_method_seed_cost[
                    f"full_budget_state_oracle_k{requested_count}"
                ] = full_elite_cost[full_state_index]
                per_method_seed_cost[
                    f"full_budget_seed_oracle_k{requested_count}"
                ] = np.min(full_elite_cost, axis=0)
                fixed_array = np.stack(
                    fixed_center_costs[requested_count], axis=1
                ).astype(np.float64)
                fixed_state_index = int(np.argmin(fixed_array.mean(axis=1)))
                per_method_seed_cost[
                    f"fixed_budget_state_oracle_k{requested_count}"
                ] = fixed_array[fixed_state_index]
                per_method_seed_cost[
                    f"fixed_budget_seed_oracle_k{requested_count}"
                ] = np.min(fixed_array, axis=0)
                per_method_seed_cost[
                    f"fixed_budget_mixture_k{requested_count}"
                ] = np.asarray(fixed_mixture_costs[requested_count], dtype=np.float64)
            for name in method_names:
                saved_costs[name].append(per_method_seed_cost[name])
            if not np.allclose(
                per_method_seed_cost["teacher"],
                per_method_seed_cost["fixed_budget_mixture_k1"],
                rtol=3e-4,
                atol=3e-4,
            ):
                raise AssertionError(f"{multi_path}: K=1 mixture differs from teacher")
            row: dict[str, Any] = {
                "episode_id": episode_id,
                "split": split_by_episode[episode_id],
                "control_step": int(source["control_step"]),
                "elite_count": valid_count,
            }
            for name in method_names:
                row[f"{name}_cost"] = float(per_method_seed_cost[name].mean())
            rows.append(row)
        if snapshot_index % 12 == 0:
            print(f"[{snapshot_index:03d}/{len(label_paths):03d}] evaluated")
    with (args.output_dir / "per_snapshot.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output_dir / "oracle_arrays.npz",
        method_names=np.asarray(method_names),
        evaluation_seeds=np.asarray(seeds, dtype=np.int64),
        **{name: np.asarray(costs) for name, costs in saved_costs.items()},
    )
    by_split = {
        split: aggregate(split_rows, method_names)
        for split in ("train", "validation", "test")
        if (split_rows := [row for row in rows if row["split"] == split])
    }
    summary = {
        "format_version": 1,
        "source_collection": str(source_root),
        "t1_labels": str(t1_root),
        "multi_elite_labels": str(multi_root),
        "evaluation_seeds": seeds,
        "requested_splits": sorted(requested_splits),
        "single_center_candidate_budget": total_samples,
        "fixed_mixture_total_candidate_budget": total_samples,
        "full_budget_oracle_candidate_multiplier": {
            f"k{count}": count for count in range(1, max_elites + 1)
        },
        "method_semantics": {
            "elite_mean": (
                "Arithmetic mean of the first K diverse elite centers, evaluated with "
                "a full 256-candidate proposal budget; diagnoses mode averaging."
            ),
            "teacher_scale": (
                "Warm plus 25/50/75 percent of the T1 teacher residual, each evaluated "
                "with a full 256-candidate proposal budget; diagnoses residual shrinkage."
            ),
            "full_budget_state_oracle": (
                "Each elite receives 256 candidates; a perfect state-level selector "
                "chooses the center with the lowest mean fresh-seed output cost."
            ),
            "full_budget_seed_oracle": (
                "Optimistic post-randomness bound choosing the best elite separately per seed."
            ),
            "fixed_budget_state_oracle": (
                "A total of 256 candidates is split across elites; a perfect state-level "
                "selector chooses the best center output."
            ),
            "fixed_budget_seed_oracle": (
                "Optimistic post-randomness bound after splitting 256 candidates."
            ),
            "fixed_budget_mixture": (
                "A feasible uniform multi-center proposal: split 256 candidates across "
                "elites, globally soft-weight all candidates, and emit one MPPI action sequence."
            ),
        },
        "overall": aggregate(rows, method_names),
        "by_split": by_split,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
