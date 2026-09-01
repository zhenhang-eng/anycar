#!/usr/bin/env python3
"""Merged zero-rollout neighborhood diagnosis: distance-bucketed flip rates,
fixed KNN, and local label oracle.

Protocol (review-tightened):
- Physical distance uses the full DBM Markov inputs: initial_state_six,
  raw reference trajectory (direct_reference), current control, and the
  absolute Actor center. Every dimension is standardized by its own robust
  scale over the 600 internal-selection contexts; blocks contribute equally.
- Neighbor candidates exclude the context itself, its repeats (same physical
  snapshot), and anything from the same episode.
- Buckets are global deciles of the nearest-neighbor distance. Per bucket
  and stratum we report the gradient flip rate against the nearest neighbor
  with Wilson 95% intervals and sample counts.
- Fixed KNN (k in 1..10) predicts the mean neighbor label; the local label
  oracle picks the best-cosine label inside the same k-neighborhood. No
  global target-aware oracle is used.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from generate_dbm_proposal_teacher import sha256_file
from train_mppi_direct_local_gradient_critic import cosine_rows
from train_mppi_direct_trust_region_actor import (
    load_actor_payload,
    load_dataset,
)


DEFAULT_INITIAL = Path(
    "outputs/mppi_proposal/direct_residual_online_ac_20260811_v2/"
    "direct_residual_online_ac_selected.pt"
)
DEFAULT_FRESH = Path(
    "outputs/mppi_proposal/direct_critic_fresh_fd_b4_smallest_target_20260813_v2/"
    "fresh_fd_audit.npz"
)
DEFAULT_MANIFEST = Path(
    "outputs/mppi_proposal/targeted_local_probe_manifest_20260814_v1/manifest.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/g0_neighbor_oracle_20260817_v1/analysis.json"
)
STRATA = (
    "G0_ONLY_CRITIC_HARD", "JOINT_G0_H", "H_ONLY",
    "G0_ONLY_CRITIC_OK", "CRITIC_ONLY", "ALL_GOOD",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-actor", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--fresh-npz", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--buckets", type=int, default=10)
    parser.add_argument("--max-k", type=int, default=10)
    return parser.parse_args()


def robust_scale(block: np.ndarray) -> np.ndarray:
    quarter = np.quantile(block, [0.25, 0.75], axis=0)
    scale = quarter[1] - quarter[0]
    scale = np.where(scale > 1e-9, scale, np.std(block, axis=0) + 1e-9)
    return scale


def block_distance(
    left: np.ndarray, right: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    z = (left - right) / scale
    return np.mean(np.square(z), axis=-1)


def wilson_interval(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    ) / denominator
    return [float(center - half), float(center + half)]


def main() -> None:
    args = parse_args()
    import torch

    initial = torch.load(args.initial_actor, map_location="cpu")
    alpha_payload = torch.load(initial["base_alpha_checkpoint"], map_location="cpu")
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, _ = load_dataset(Path(initial["labels"]), old_payload)
    fresh = dict(np.load(args.fresh_npz, allow_pickle=False))
    manifest = json.loads(args.manifest.read_text())
    row_by_context = {
        int(row["context_index"]): row for row in manifest["rows"]
    }
    contexts = fresh["context_index"].astype(np.int64)
    gradient = fresh["gradient"].astype(np.float32)
    episode = fresh["episode"].astype(str)
    state_key = np.asarray([
        f"{row_by_context[int(value)]['episode']}#"
        f"{int(row_by_context[int(value)]['physical_snapshot_ordinal'])}"
        for value in contexts
    ])
    stratum = np.asarray([
        row_by_context[int(value)]["stratum_id"] for value in contexts
    ])
    center = fresh["actor_center"].astype(np.float32).reshape(len(contexts), -1)

    six = data.initial_state_six[contexts].astype(np.float32)
    reference = data.direct_reference[contexts].astype(np.float32).reshape(
        len(contexts), -1
    )
    control = data.current_action[contexts].astype(np.float32)
    blocks = {
        "initial_state_six": (six, robust_scale(six)),
        "reference": (reference, robust_scale(reference)),
        "current_control": (control, robust_scale(control)),
        "actor_center": (center, robust_scale(center)),
    }

    count = len(contexts)
    distances = np.zeros((count, count), np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            total = 0.0
            for block, scale in blocks.values():
                total += block_distance(
                    block[i][None], block[j][None], scale
                )[0]
            distances[i, j] = distances[j, i] = np.sqrt(total)

    same_episode = episode[:, None] == episode[None, :]
    same_state = state_key[:, None] == state_key[None, :]
    excluded = same_episode | same_state | np.eye(count, dtype=bool)
    valid = ~excluded
    masked = np.where(valid, distances, np.inf)
    nearest_index = np.argmin(masked, axis=1)
    nearest_distance = masked[np.arange(count), nearest_index]

    order = np.argsort(nearest_distance)
    bucket_edges = np.quantile(
        nearest_distance, np.linspace(0, 1, args.buckets + 1)
    )
    bucket_of = np.clip(
        np.searchsorted(bucket_edges, nearest_distance, side="right") - 1,
        0, args.buckets - 1,
    )

    flip = np.asarray([
        float(
            (gradient[i] @ gradient[nearest_index[i]])
            / (np.linalg.norm(gradient[i])
               * np.linalg.norm(gradient[nearest_index[i]]) + 1e-12)
        ) < 0.0
        for i in range(count)
    ])

    buckets_report = {}
    for bucket in range(args.buckets):
        edge_low = float(bucket_edges[bucket])
        edge_high = float(bucket_edges[bucket + 1])
        per_stratum = {}
        for name in STRATA:
            mask = (bucket_of == bucket) & (stratum == name)
            total = int(np.sum(mask))
            successes = int(np.sum(flip[mask]))
            per_stratum[name] = {
                "count": total,
                "flip_rate": successes / total if total else None,
                "wilson_95": wilson_interval(successes, total),
            }
        mask_any = bucket_of == bucket
        per_stratum["ALL"] = {
            "count": int(np.sum(mask_any)),
            "flip_rate": float(np.mean(flip[mask_any])),
            "wilson_95": wilson_interval(
                int(np.sum(flip[mask_any])), int(np.sum(mask_any))
            ),
        }
        buckets_report[str(bucket)] = {
            "distance_low": edge_low,
            "distance_high": edge_high,
            "by_stratum": per_stratum,
        }

    topk_indices = np.argsort(masked, axis=1)[:, : args.max_k]
    knn_report = {}
    for k in range(1, args.max_k + 1):
        per_stratum = {}
        for name in STRATA:
            rows = np.flatnonzero(stratum == name)
            if len(rows) == 0:
                continue
            fixed, oracle = [], []
            for row in rows:
                neighbors = topk_indices[row, :k]
                labels = gradient[neighbors]
                norms = np.linalg.norm(labels, axis=1, keepdims=True) + 1e-12
                mean_label = (labels / norms).mean(axis=0)
                fixed.append(float(
                    (mean_label @ gradient[row])
                    / (np.linalg.norm(mean_label)
                       * np.linalg.norm(gradient[row]) + 1e-12)
                ))
                best = max(float(
                    (label @ gradient[row])
                    / (np.linalg.norm(label)
                       * np.linalg.norm(gradient[row]) + 1e-12)
                ) for label in labels)
                oracle.append(best)
            per_stratum[name] = {
                "count": int(len(rows)),
                "fixed_knn_cosine_median": float(np.median(fixed)),
                "local_oracle_cosine_median": float(np.median(oracle)),
                "local_oracle_positive_fraction": float(np.mean(
                    np.asarray(oracle) > 0
                )),
            }
        knn_report[str(k)] = per_stratum

    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "G0_NEIGHBOR_DISTANCE_ORACLE_DIAGNOSIS_ACTOR_FROZEN",
        "sources": {
            "initial_actor": str(args.initial_actor.resolve()),
            "initial_actor_sha256": sha256_file(args.initial_actor),
            "fresh_fd": str(args.fresh_npz.resolve()),
            "fresh_fd_sha256": sha256_file(args.fresh_npz),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
        },
        "protocol": {
            "distance_blocks": sorted(blocks),
            "normalization": "per-dimension IQR over the 600 contexts",
            "exclusions": "same episode, same physical snapshot, self",
            "buckets": args.buckets,
            "max_k": args.max_k,
            "note": (
                "local label oracle picks the best-cosine label within the "
                "same k-neighborhood; global target-aware oracles excluded"
            ),
        },
        "nearest_neighbor_distance": {
            "median": float(np.median(nearest_distance)),
            "p10": float(np.quantile(nearest_distance, 0.10)),
            "p90": float(np.quantile(nearest_distance, 0.90)),
            "bucket_edges": bucket_edges.tolist(),
        },
        "distance_bucketed_flip": buckets_report,
        "knn_and_local_oracle": knn_report,
        "contract": {
            "actor_frozen": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "zero_rollout": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = {
        "output": str(args.output.resolve()),
        "nn_distance_median": result["nearest_neighbor_distance"]["median"],
        "bucket_edges": [
            round(edge, 2) for edge in bucket_edges.tolist()
        ],
        "flip_all_by_bucket": [
            round(buckets_report[str(b)]["by_stratum"]["ALL"]["flip_rate"], 3)
            for b in range(args.buckets)
        ],
    }
    for name in ("G0_ONLY_CRITIC_HARD", "JOINT_G0_H", "H_ONLY", "ALL_GOOD"):
        summary[f"flip_{name}_by_bucket"] = [
            (
                round(buckets_report[str(b)]["by_stratum"][name]["flip_rate"], 3)
                if buckets_report[str(b)]["by_stratum"][name]["flip_rate"]
                is not None else None
            )
            for b in range(args.buckets)
        ]
        k1 = knn_report["1"].get(name)
        k5 = knn_report["5"].get(name)
        summary[f"knn_{name}"] = {
            "k1_fixed": round(k1["fixed_knn_cosine_median"], 3),
            "k1_local_oracle": round(k1["local_oracle_cosine_median"], 3),
            "k5_fixed": round(k5["fixed_knn_cosine_median"], 3),
            "k5_local_oracle": round(k5["local_oracle_cosine_median"], 3),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
