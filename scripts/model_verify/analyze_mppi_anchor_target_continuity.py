#!/usr/bin/env python3
"""Anchor-stability / target-space continuity diagnostic (zero rollout).

Tests the hypothesis that the full optimal action a* is more stable across
neighbor states than the residual da* = a* - a0, i.e. that the instability
lives in the subtraction against the mediocre anchor rather than in a* itself.

For each neighbor state pair (block-equal IQR distance on physical state +
reference + current + anchor), compare:
  - neighbor cosine and normalized L2 of the FULL action a*
  - neighbor cosine and normalized L2 of the RESIDUAL da* = a* - a0
plus magnitudes, to see whether a* is larger / better conditioned.
A substantially higher neighbor continuity for a* supports estimating the
full action directly; comparable continuity means the instability is in the
landscape/target, not the anchor subtraction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from run_mppi_proximal_search_phase1a import (
    DEFAULT_GT_TRAIN,
    DEFAULT_REPLAY_LABELS,
    DEFAULT_SCENARIO_PLAN,
    load_states,
    select_states,
)


DEFAULT_LABELS = Path(
    "outputs/mppi_proposal/consensus64_labels_pool_20260818_v1/labels.npz"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/anchor_target_continuity_20260818_v1"
)
BLOCKS = ("initial_state_six", "reference", "current_action", "anchor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--neighbors", type=int, default=8)
    return parser.parse_args()


def robust_scale(x: np.ndarray) -> np.ndarray:
    q = np.quantile(x, [0.25, 0.75], axis=0)
    scale = q[1] - q[0]
    return np.where(scale > 1e-9, scale, np.std(x, axis=0) + 1e-9)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    labels = dict(np.load(args.labels, allow_pickle=False))
    a_star = labels["label_knots"].astype(np.float32)
    count = len(a_star)
    teacher_keys = [str(v) for v in labels["episodes"]]

    states = select_states(load_states(SimpleNamespace(
        replay_labels=DEFAULT_REPLAY_LABELS,
        gt_train=DEFAULT_GT_TRAIN,
        scenario_plan=DEFAULT_SCENARIO_PLAN,
        repeat=0,
    )), count)
    key_to_state = {
        f"{s['episode']}#{s['snapshot']}": s for s in states
    }
    ordered_states = [key_to_state[k] for k in teacher_keys]

    a0 = np.stack([s["a0"] for s in ordered_states]).astype(np.float32)
    residual = a_star - a0

    feats = np.concatenate([
        np.stack([np.asarray(s["initial_state_six"], np.float32)
                  for s in ordered_states]),
        np.stack([np.asarray(s["reference"], np.float32).reshape(-1)
                  for s in ordered_states]),
        np.stack([np.asarray(s["current_action"], np.float32)
                  for s in ordered_states]),
        a0.reshape(count, -1),
    ], axis=1)
    z = feats / robust_scale(feats)

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=args.neighbors + 1).fit(z)
    dist, idx = nn.kneighbors(z)
    dist = dist[:, 1:]
    idx = idx[:, 1:]

    flat_star = a_star.reshape(count, -1)
    flat_res = residual.reshape(count, -1)
    star_norm = np.linalg.norm(flat_star, axis=1)
    res_norm = np.linalg.norm(flat_res, axis=1)

    def neighbor_stats(flat: np.ndarray, norms: np.ndarray):
        cos_list, rel_list = [], []
        for i in range(count):
            for j_pos in range(idx.shape[1]):
                j = idx[i, j_pos]
                ni, nj = norms[i], norms[j]
                if ni < 1e-8 or nj < 1e-8:
                    continue
                cos_list.append(float(flat[i] @ flat[j] / (ni * nj)))
                rel_list.append(float(
                    np.linalg.norm(flat[i] - flat[j]) / (0.5 * (ni + nj) + 1e-12)
                ))
        return np.asarray(cos_list), np.asarray(rel_list)

    star_cos, star_rel = neighbor_stats(flat_star, star_norm)
    res_cos, res_rel = neighbor_stats(flat_res, res_norm)

    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "ANCHOR_TARGET_CONTINUITY_DIAGNOSED_ACTOR_FROZEN",
        "sources": {"labels": str(args.labels.resolve())},
        "counts": {"states": count, "pairs": int(len(star_cos))},
        "magnitudes": {
            "full_action_norm_median": float(np.median(star_norm)),
            "residual_norm_median": float(np.median(res_norm)),
            "residual_over_full_norm_median": float(np.median(
                res_norm / (star_norm + 1e-12)
            )),
        },
        "full_action_neighbor": {
            "cosine_median": float(np.median(star_cos)),
            "cosine_p10": float(np.quantile(star_cos, 0.10)),
            "rel_l2_median": float(np.median(star_rel)),
            "cos_positive_fraction": float(np.mean(star_cos > 0)),
        },
        "residual_neighbor": {
            "cosine_median": float(np.median(res_cos)),
            "cosine_p10": float(np.quantile(res_cos, 0.10)),
            "rel_l2_median": float(np.median(res_rel)),
            "cos_positive_fraction": float(np.mean(res_cos > 0)),
        },
        "contract": {
            "actor_frozen": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "zero_rollout": True,
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    m = result["magnitudes"]
    fa, rs = result["full_action_neighbor"], result["residual_neighbor"]
    print("magnitudes: full %.3f  residual %.3f  ratio %.3f" % (
        m["full_action_norm_median"], m["residual_norm_median"],
        m["residual_over_full_norm_median"]))
    print("full   neighbor: cos med %.3f p10 %.3f relL2 %.3f posfrac %.3f" % (
        fa["cosine_median"], fa["cosine_p10"], fa["rel_l2_median"],
        fa["cos_positive_fraction"]))
    print("resid  neighbor: cos med %.3f p10 %.3f relL2 %.3f posfrac %.3f" % (
        rs["cosine_median"], rs["cosine_p10"], rs["rel_l2_median"],
        rs["cos_positive_fraction"]))


if __name__ == "__main__":
    main()
