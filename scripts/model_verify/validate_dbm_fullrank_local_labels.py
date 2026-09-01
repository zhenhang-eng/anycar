#!/usr/bin/env python3
"""Validate full-rank local fixed-DBM reward labels without new rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_dbm_fullrank_local_labels import (
    ACTION_DIMENSION,
    CENTER_COUNT,
    DIRECTION_COUNT,
    FORMAT_VERSION,
    center_names,
    hadamard_directions,
    make_centers,
    summarize_costs,
)
from generate_dbm_proposal_teacher import sha256_file


DEFAULT_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_critic_fullrank_diverse_20260805_v1"
)
COST_FIELDS = (
    "proposal_weighted_output_cost",
    "proposal_best_cost",
    "proposal_p10_cost",
    "proposal_median_cost",
    "proposal_softmin_cost",
    "proposal_effective_sample_size",
    "proposal_clip_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", nargs="?", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a pilot subset that does not cover every episode in splits.json.",
    )
    return parser.parse_args()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_close(actual: float, expected: float, name: str) -> None:
    check(np.isclose(actual, expected, rtol=1e-6, atol=1e-5), name)


def main() -> None:
    args = parse_args()
    root = args.labels.resolve()
    summary = json.loads((root / "summary.json").read_text())
    source_root = Path(summary["source_collection"])
    parent_root = Path(summary["parent_labels"])
    selection_seeds = np.asarray(summary["selection_seeds"], dtype=np.int64)
    audit_seeds = np.asarray(summary["audit_seeds"], dtype=np.int64)
    check(not set(selection_seeds) & set(audit_seeds), "seed sets overlap")
    check(int(summary["action_dimension"]) == ACTION_DIMENSION, "action dimension")
    check(int(summary["direction_count"]) == DIRECTION_COUNT, "direction count")
    check(int(summary["center_count"]) == CENTER_COUNT, "center count")
    radius_sigma = float(summary["radius_sigma"])

    expected_directions = hadamard_directions()
    flattened = expected_directions.reshape(DIRECTION_COUNT, ACTION_DIMENSION)
    check(
        np.array_equal(
            flattened @ flattened.T,
            ACTION_DIMENSION * np.eye(ACTION_DIMENSION),
        ),
        "Hadamard design is not orthogonal",
    )

    split_payload = json.loads((root / "splits.json").read_text())
    splits = {name: split_payload[name] for name in ("train", "validation", "test")}
    expected_episodes = {episode for members in splits.values() for episode in members}
    check(
        sum(len(members) for members in splits.values()) == len(expected_episodes),
        "episode split overlap",
    )

    paths = sorted(root.glob("episode_*/*.npz"))
    check(len(paths) == int(summary["snapshot_count"]), "snapshot count mismatch")
    check(bool(paths), "no label snapshots")
    selection_costs = []
    audit_costs = []
    ranks = []
    clip_fractions = []
    seen_episodes = set()
    names_expected = center_names()
    for index, path in enumerate(paths, start=1):
        episode_id = path.parent.name
        seen_episodes.add(episode_id)
        source_path = source_root / episode_id / "snapshots" / path.name
        parent_path = parent_root / episode_id / path.name
        check(source_path.is_file(), f"{path}: source missing")
        check(parent_path.is_file(), f"{path}: parent missing")
        with np.load(path, allow_pickle=False) as label, np.load(
            source_path, allow_pickle=False
        ) as source, np.load(parent_path, allow_pickle=False) as parent:
            check(int(label["format_version"]) == FORMAT_VERSION, f"{path}: version")
            check(
                str(label["source_snapshot_sha256"]) == sha256_file(source_path),
                f"{path}: source hash",
            )
            check(
                str(label["parent_label_sha256"]) == sha256_file(parent_path),
                f"{path}: parent hash",
            )
            names = tuple(label["center_names"].astype(str))
            check(names == names_expected, f"{path}: center names")
            base = np.asarray(label["base_center_knots"], dtype=np.float32)
            check(base.shape == (8, 2), f"{path}: base shape")
            check(
                np.allclose(base, parent["network_center_knots"], atol=1e-7),
                f"{path}: base differs from frozen BC",
            )
            directions = np.asarray(label["normalized_directions"], dtype=np.float32)
            check(
                np.array_equal(directions, expected_directions),
                f"{path}: direction design",
            )
            check_close(float(label["radius_sigma"]), radius_sigma, f"{path}: radius")
            params = json.loads(str(source["mppi_params_json"]))
            sigma = np.asarray(params["noise_sigma"], dtype=np.float32)
            low = np.asarray(params["action_min"], dtype=np.float32)
            high = np.asarray(params["action_max"], dtype=np.float32)
            raw_expected, centers_expected = make_centers(
                base, directions, sigma, radius_sigma, low, high
            )
            raw = np.asarray(label["raw_centers"], dtype=np.float32)
            centers = np.asarray(label["centers"], dtype=np.float32)
            check(raw.shape == (CENTER_COUNT, 8, 2), f"{path}: raw center shape")
            check(centers.shape == (CENTER_COUNT, 8, 2), f"{path}: center shape")
            check(np.array_equal(raw, raw_expected), f"{path}: raw centers")
            check(np.array_equal(centers, centers_expected), f"{path}: clipped centers")
            check(np.all(np.isfinite(centers)), f"{path}: nonfinite centers")
            check(np.all(centers >= low) and np.all(centers <= high), f"{path}: bounds")
            standardized = (centers - base[None]) / sigma.reshape(1, 1, 2)
            rank = int(np.linalg.matrix_rank(standardized.reshape(CENTER_COUNT, -1)))
            check(rank == int(label["local_direction_rank"]), f"{path}: rank value")
            check(rank == ACTION_DIMENSION, f"{path}: local coverage is not full rank")
            ranks.append(rank)
            clip_fractions.append(float(np.mean(raw != centers)))
            check(
                np.array_equal(label["proposal_evaluation_seeds"], selection_seeds),
                f"{path}: selection seeds",
            )
            check(
                np.array_equal(label["audit_evaluation_seeds"], audit_seeds),
                f"{path}: audit seeds",
            )
            for prefix, seeds in (("", selection_seeds), ("audit_", audit_seeds)):
                for field in COST_FIELDS:
                    values = np.asarray(label[f"{prefix}{field}"])
                    check(
                        values.shape == (CENTER_COUNT, len(seeds)),
                        f"{path}: {prefix}{field} shape",
                    )
                    check(np.all(np.isfinite(values)), f"{path}: {prefix}{field}")
            selection_costs.append(label["proposal_weighted_output_cost"])
            audit_costs.append(label["audit_proposal_weighted_output_cost"])
        check(path.with_suffix(".json").is_file(), f"{path}: metadata missing")
        if index % 400 == 0:
            print(f"[{index:04d}/{len(paths):04d}] validated", flush=True)

    check(seen_episodes <= expected_episodes, "unknown episode outside splits")
    if not args.allow_partial:
        check(seen_episodes == expected_episodes, "episode coverage differs from splits")
    selection = np.asarray(selection_costs, dtype=np.float32)
    audit = np.asarray(audit_costs, dtype=np.float32)
    diagnostics = summarize_costs(selection, audit)
    recorded = summary["reward_diagnostics"]
    for name in (
        "selection_audit_center_cost_mae",
        "directional_sign_stability",
        "directional_difference_mae",
    ):
        check_close(float(diagnostics[name]), float(recorded[name]), f"summary {name}")
    check(
        int(diagnostics["directional_valid_comparison_count"])
        == int(recorded["directional_valid_comparison_count"]),
        "summary directional valid count",
    )
    check(int(np.min(ranks)) == int(summary["local_rank"]["minimum"]), "rank min")
    check(int(np.max(ranks)) == int(summary["local_rank"]["maximum"]), "rank max")
    check(
        int(np.sum(np.asarray(ranks) == ACTION_DIMENSION))
        == int(summary["local_rank"]["full_rank_count"]),
        "full rank count",
    )
    check_close(
        float(np.mean(clip_fractions)),
        float(summary["center_clip_fraction"]["mean"]),
        "clip fraction mean",
    )
    check_close(
        float(np.max(clip_fractions)),
        float(summary["center_clip_fraction"]["maximum"]),
        "clip fraction maximum",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "labels": str(root),
                "snapshots": len(paths),
                "episodes": len(seen_episodes),
                "centers_per_snapshot": CENTER_COUNT,
                "full_rank_snapshots": int(np.sum(np.asarray(ranks) == ACTION_DIMENSION)),
                "directional_sign_stability": diagnostics[
                    "directional_sign_stability"
                ],
                "selection_audit_center_cost_mae": diagnostics[
                    "selection_audit_center_cost_mae"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
