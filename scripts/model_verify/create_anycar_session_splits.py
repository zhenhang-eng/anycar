#!/usr/bin/env python3
"""Create leakage-resistant AnyCar split manifests for sim/real experiments."""

import argparse
import random
import re
from datetime import datetime
from pathlib import Path


TIMESTAMP = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)")


def timestamp(path):
    match = TIMESTAMP.search(path.name)
    if match is None:
        raise ValueError(f"No ISO timestamp in {path}")
    return datetime.fromisoformat(match.group(1))


def temporal_clusters(paths, maximum_gap_seconds=1.0):
    ordered = sorted(paths, key=lambda path: (timestamp(path), path.name))
    clusters = []
    for path in ordered:
        current_time = timestamp(path)
        if (
            not clusters
            or (current_time - timestamp(clusters[-1][-1])).total_seconds()
            > maximum_gap_seconds
        ):
            clusters.append([])
        clusters[-1].append(path)
    return clusters


def seeded_sample(paths, count, seed):
    paths = list(paths)
    generator = random.Random(seed)
    generator.shuffle(paths)
    if count > len(paths):
        raise ValueError(f"Requested {count} files from a population of {len(paths)}")
    return sorted(paths[:count], key=lambda path: (timestamp(path), path.name))


def simulation_split(paths, seed):
    clusters = temporal_clusters(paths)
    if len(clusters) != 6:
        raise ValueError(f"Expected six simulation sessions, found {len(clusters)}")
    train_population = [path for cluster in clusters[:4] for path in cluster]
    return {
        "train": seeded_sample(train_population, 16000, seed),
        "val": seeded_sample(clusters[4], 3000, seed + 1),
        "test": seeded_sample(clusters[5], 1000, seed + 2),
        "discarded": [],
    }, clusters


def real_split(paths):
    clusters = temporal_clusters(paths)
    march = [cluster for cluster in clusters if timestamp(cluster[0]).date().isoformat() == "2025-03-31"]
    april = [cluster for cluster in clusters if timestamp(cluster[0]).date().isoformat() == "2025-04-01"]
    if len(march) + len(april) != len(clusters):
        raise ValueError("Real dataset contains dates outside 2025-03-31 and 2025-04-01")
    boundary = int(len(april) * 0.70)
    guard = 20
    val_clusters = april[:boundary]
    discarded = april[boundary : boundary + guard]
    test_clusters = april[boundary + guard :]
    return {
        "train": [path for cluster in march for path in cluster],
        "val": [path for cluster in val_clusters for path in cluster],
        "test": [path for cluster in test_clusters for path in cluster],
        "discarded": [path for cluster in discarded for path in cluster],
    }, clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("simulation", "real"), required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    paths = list(args.dataset_dir.glob("*.pkl"))
    split, clusters = (
        simulation_split(paths, args.seed)
        if args.domain == "simulation"
        else real_split(paths)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test", "discarded"):
        (args.output_dir / f"{name}_files.txt").write_text(
            "".join(f"{path.resolve()}\n" for path in split[name])
        )
    summary = [
        f"domain: {args.domain}",
        f"dataset_dir: {args.dataset_dir.resolve()}",
        f"temporal_clusters: {len(clusters)}",
        *(f"{name}_files: {len(split[name])}" for name in ("train", "val", "test", "discarded")),
    ]
    (args.output_dir / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
