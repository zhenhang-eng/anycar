#!/usr/bin/env python3
"""Validate local fixed-DBM proposal-critic labels without new rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_dbm_proposal_critic_labels import CENTER_NAMES, FORMAT_VERSION
from generate_dbm_proposal_teacher import sha256_file


DEFAULT_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_critic_local_diverse_20260805_v1"
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
    return parser.parse_args()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    args = parse_args()
    root = args.labels.resolve()
    summary = json.loads((root / "summary.json").read_text())
    source_root = Path(summary["source_collection"])
    t1_root = Path(summary["t1_labels"])
    selection_seeds = np.asarray(summary["selection_seeds"], dtype=np.int64)
    audit_seeds = np.asarray(summary["audit_seeds"], dtype=np.int64)
    check(not set(selection_seeds) & set(audit_seeds), "seed sets overlap")
    check(summary["center_names"] == list(CENTER_NAMES), "summary center names")
    split_payload = json.loads((root / "splits.json").read_text())
    splits = {
        name: split_payload[name] for name in ("train", "validation", "test")
    }
    expected_episodes = {
        episode for members in splits.values() for episode in members
    }
    check(
        sum(len(members) for members in splits.values()) == len(expected_episodes),
        "episode split overlap",
    )
    paths = sorted(root.glob("episode_*/*.npz"))
    check(len(paths) == int(summary["snapshot_count"]), "snapshot count mismatch")
    selection_costs = []
    audit_costs = []
    seen_episodes = set()
    for index, path in enumerate(paths, start=1):
        episode_id = path.parent.name
        seen_episodes.add(episode_id)
        source_path = source_root / episode_id / "snapshots" / path.name
        t1_path = t1_root / episode_id / path.name
        check(source_path.is_file(), f"{path}: source missing")
        check(t1_path.is_file(), f"{path}: T1 missing")
        with np.load(path, allow_pickle=False) as label, np.load(
            source_path, allow_pickle=False
        ) as source, np.load(t1_path, allow_pickle=False) as t1:
            check(int(label["format_version"]) == FORMAT_VERSION, f"{path}: version")
            check(
                str(label["source_snapshot_sha256"]) == sha256_file(source_path),
                f"{path}: source hash",
            )
            check(
                str(label["t1_label_sha256"]) == sha256_file(t1_path),
                f"{path}: T1 hash",
            )
            names = tuple(label["center_names"].astype(str))
            check(names == CENTER_NAMES, f"{path}: center names")
            centers = np.asarray(label["shortlist_centers"], dtype=np.float32)
            check(centers.shape == (len(CENTER_NAMES), 8, 2), f"{path}: centers")
            check(np.all(np.isfinite(centers)), f"{path}: nonfinite centers")
            params = json.loads(str(source["mppi_params_json"]))
            low = np.asarray(params["action_min"], dtype=np.float32)
            high = np.asarray(params["action_max"], dtype=np.float32)
            check(np.all(centers >= low) and np.all(centers <= high), f"{path}: bounds")
            check(
                np.allclose(centers[0], source["sampling_mean_knots"], atol=1e-7),
                f"{path}: warm center",
            )
            check(
                np.allclose(centers[1], label["network_center_knots"], atol=1e-7),
                f"{path}: network center",
            )
            check(
                np.allclose(centers[2], t1["teacher_center_knots"], atol=1e-7),
                f"{path}: teacher center",
            )
            check(int(label["warm_shortlist_index"]) == 0, f"{path}: warm index")
            check(int(label["network_shortlist_index"]) == 1, f"{path}: network index")
            check(int(label["teacher_shortlist_index"]) == 2, f"{path}: teacher index")
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
                        values.shape == (len(CENTER_NAMES), len(seeds)),
                        f"{path}: {prefix}{field} shape",
                    )
                    check(
                        np.all(np.isfinite(values)),
                        f"{path}: {prefix}{field} nonfinite",
                    )
            selection_costs.append(label["proposal_weighted_output_cost"].mean(1))
            audit_costs.append(label["audit_proposal_weighted_output_cost"].mean(1))
        check(path.with_suffix(".json").is_file(), f"{path}: metadata missing")
        if index % 400 == 0:
            print(f"[{index:04d}/{len(paths):04d}] validated")
    check(seen_episodes == expected_episodes, "episode coverage differs from splits")
    selection = np.asarray(selection_costs)
    audit = np.asarray(audit_costs)
    for name, values in (("selection", selection), ("audit", audit)):
        recorded = summary[name]["mean_cost"]
        for center_index, center_name in enumerate(CENTER_NAMES):
            check(
                np.isclose(
                    values[:, center_index].mean(), recorded[center_name], atol=1e-5
                ),
                f"summary {name} mean mismatch: {center_name}",
            )
    print(
        json.dumps(
            {
                "status": "ok",
                "labels": str(root),
                "snapshots": len(paths),
                "episodes": len(seen_episodes),
                "centers_per_snapshot": len(CENTER_NAMES),
                "selection_teacher_vs_network_gain": float(
                    np.mean(selection[:, 1] - selection[:, 2])
                ),
                "audit_teacher_vs_network_gain": float(
                    np.mean(audit[:, 1] - audit[:, 2])
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
