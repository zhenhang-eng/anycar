#!/usr/bin/env python3
"""Build nested learning-curve subsets from the consensus-64 pool labels.

Slices the full diverse-train pool (ordered by speed/scenario/residual-norm,
the same order select_states produces) at even-spacing indices so that
600 ⊂ 1200 ⊂ 1800 are nested by construction. Sanity checks: the 600-point
subset must equal the Phase 1b manifest states (the existing A0b c64 run is
therefore the first point of the curve) and nesting must hold exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_POOL = Path("outputs/mppi_proposal/consensus64_labels_pool_20260818_v1")
DEFAULT_PHASE1B_MANIFEST = Path(
    "outputs/mppi_proposal/proximal_search_phase1b_20260818_v1/manifest.json"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/learning_curve_subsets_20260818_v1")
SIZES = (600, 1200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--phase1b-manifest", type=Path, default=DEFAULT_PHASE1B_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    labels = dict(np.load(args.pool / "labels.npz", allow_pickle=False))
    manifest = json.loads((args.pool / "manifest.json").read_text())
    states = manifest["states"]
    count = len(states)
    episodes = [str(value) for value in labels["episodes"]]

    phase1b = [
        f"{row['episode']}#{row['snapshot']}"
        for row in json.loads(args.phase1b_manifest.read_text())["states"]
    ]
    index_sets = {}
    for size in args.sizes:
        if size > count:
            raise ValueError(f"size {size} exceeds pool {count}")
        indices = [int(index * count / size) for index in range(size)]
        index_sets[size] = indices
    ordered = sorted(args.sizes)
    for smaller, larger in zip(ordered[:-1], ordered[1:]):
        if not set(index_sets[smaller]) <= set(index_sets[larger]):
            raise AssertionError(f"nesting violated for {smaller} in {larger}")
    if count in index_sets:
        raise ValueError("full pool needs no subset file")
    if min(ordered) == 600:
        subset600 = [episodes[i] for i in index_sets[600]]
        if subset600 != phase1b:
            mismatch = sum(a != b for a, b in zip(subset600, phase1b))
            raise AssertionError(
                f"600-subset differs from Phase 1b manifest ({mismatch} rows)"
            )

    args.output.mkdir(parents=True)
    for size in ordered:
        indices = index_sets[size]
        target = args.output / f"n{size}"
        target.mkdir()
        np.savez_compressed(
            target / "labels.npz",
            **{name: value[indices] for name, value in labels.items()},
        )
        (target / "manifest.json").write_text(json.dumps({
            "states": [states[i] for i in indices],
            "pool": str(args.pool),
            "indices": indices,
        }, indent=1))
        print(f"n{size}: {len(indices)} states written")


if __name__ == "__main__":
    main()
