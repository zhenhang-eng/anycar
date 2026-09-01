#!/usr/bin/env python3
"""Phase 0.1 zero-rollout residual-coherence diagnosis for the proximal
search-distillation route.

Question (pre-registered): is the a0-conditioned residual target
Delta_a* = (a* - a0) / sigma more neighborhood-coherent than the already
measured own-center gradient labels (43-46% nearest-neighbor flips), and
does conditioning the distance metric on a0 itself reduce branch mixing?

Data: the diverse replay-label snapshots (train+validation splits, 105
episodes x 20 snapshots = 2100 snapshots, 2 first-pass contexts each) joined
with the frozen J16 best-found centers from the direct GT artifacts. No new
DBM rollouts are performed; no actor is updated; formal validation/test stay
sealed. The validation split here is the historically consumed oracle set and
is used only as consumed-data mechanism analysis.

Protocol:
- Feature blocks follow the 11.36 convention: initial_state_six, raw
  reference[1:] (50x4), current_action, plus the absolute anchor block a0.
  Each dimension is standardized by its own IQR scale (std fallback) over all
  contexts; blocks contribute equally (per-dimension weight 1/sqrt(block dim)).
- Neighbor candidates exclude the context itself, its repeat (same physical
  snapshot), and anything from the same episode.
- Residual labels use sigma-normalized knot space; the flip criterion is
  cosine(Delta_i, Delta_j) < 0.
- Two anchors are reported: bootstrap_actor_center (frozen deterministic
  actor stored in the replay labels) and anchor_center (warm/MPPI anchor).
- The primary distance metric includes the a0 block; a without-a0 variant
  isolates the conditioning contribution. Both use identical exclusions.

Pre-registered routing:
- RESIDUAL_COHERENCE_VIABLE: nearest-neighbor residual flip rate <= 0.30 AND
  fixed k=5 residual cosine median >= 0.40 AND removing the a0 block raises
  the flip rate by >= 0.05.
- Otherwise CONSERVATIVE_SUPERVISION_REQUIRED.

Label-space coherence is necessary but not sufficient: whether translated
neighbor residuals actually improve cost requires rollouts and is deferred to
Phase 1. The local availability oracle below is a label-space proxy only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from generate_dbm_proposal_teacher import sha256_file

DEFAULT_REPLAY_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_direct_center_replay_diverse_20260806_v2"
)
DEFAULT_GT_TRAIN = Path(
    "outputs/mppi_proposal/dbm_direct_gt_train_20260807_v2"
)
DEFAULT_GT_VALIDATION = Path(
    "outputs/mppi_proposal/dbm_direct_gt_validation_20260806_v2"
)
DEFAULT_SCENARIO_PLAN = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1/scenario_plan.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/residual_coherence_phase0_20260817_v1/analysis.json"
)
SPEED_LEVELS = (1.2, 1.6, 2.0, 2.4, 2.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-labels", type=Path, default=DEFAULT_REPLAY_LABELS)
    parser.add_argument("--gt-train", type=Path, default=DEFAULT_GT_TRAIN)
    parser.add_argument("--gt-validation", type=Path, default=DEFAULT_GT_VALIDATION)
    parser.add_argument("--scenario-plan", type=Path, default=DEFAULT_SCENARIO_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--buckets", type=int, default=10)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument(
        "--query-split",
        choices=("train", "validation"),
        default=None,
        help=(
            "restricted mode: train contexts form the only neighbor bank "
            "(normalization fitted on train) and the chosen split supplies "
            "queries; omitted keeps the original pooled mode"
        ),
    )
    return parser.parse_args()


def robust_scale(block: np.ndarray) -> np.ndarray:
    quarter = np.quantile(block, [0.25, 0.75], axis=0)
    scale = quarter[1] - quarter[0]
    scale = np.where(scale > 1e-9, scale, np.std(block, axis=0) + 1e-9)
    return scale


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


def load_gt_lookup(gt_root: Path, expected_split: str) -> dict[tuple[str, str], dict]:
    summary = json.loads((gt_root / "summary.json").read_text())
    if summary["split"] != expected_split:
        raise AssertionError(f"{gt_root} is split {summary['split']}")
    if "test" not in summary["test_policy"]:
        raise AssertionError("GT result does not seal test")
    return {
        (row["episode"], row["snapshot"]): row for row in summary["rows"]
    }


def load_gt_centers(
    gt_root: Path,
    lookup: dict[tuple[str, str], dict],
) -> dict[tuple[str, str], tuple[np.ndarray, float]]:
    cache: dict[tuple[str, str], tuple[np.ndarray, float]] = {}
    for key in lookup:
        if key in cache:
            continue
        with np.load(gt_root / key[0] / key[1], allow_pickle=False) as result:
            index = int(result["knot_best_index"])
            cache[key] = (
                np.asarray(result["optimized_knots"][index], np.float32),
                float(lookup[key]["j16_best_found"]),
            )
    return cache


def build_dataset(args: argparse.Namespace) -> dict[str, np.ndarray]:
    splits = json.loads((args.replay_labels / "splits.json").read_text())
    episode_split = {}
    for split in ("train", "validation"):
        for episode in splits[split]:
            episode_split[episode] = split
    plan = json.loads(args.scenario_plan.read_text())
    episode_plan = {row["episode_id"]: row for row in plan["episodes"]}

    lookups = {
        "train": load_gt_lookup(args.gt_train, "train"),
        "validation": load_gt_lookup(args.gt_validation, "validation"),
    }
    gt_cache = {
        "train": load_gt_centers(args.gt_train, lookups["train"]),
        "validation": load_gt_centers(
            args.gt_validation, lookups["validation"]
        ),
    }

    fields: dict[str, list] = {
        "episode": [], "snapshot": [], "split": [], "scenario": [],
        "speed": [], "ordinal": [], "context": [],
        "six": [], "reference": [], "control": [],
        "actor_anchor": [], "warm_anchor": [], "sigma": [],
        "astar": [], "j16_cost": [],
    }
    episodes = sorted(episode_split)
    for episode in episodes:
        split = episode_split[episode]
        row_plan = episode_plan[episode]
        paths = sorted((args.replay_labels / episode).glob("step_*.npz"))
        for ordinal, path in enumerate(paths):
            key = (episode, path.name)
            if key not in lookups[split]:
                raise AssertionError(f"missing GT row for {key}")
            astar, j16_cost = gt_cache[split][key]
            with np.load(path, allow_pickle=False) as label:
                source_path = str(label["source_snapshot"])
                sigma = np.asarray(label["sigma"], np.float32)
                actor_anchor = np.asarray(label["bootstrap_actor_center"], np.float32)
                warm_anchor = np.asarray(label["anchor_center"], np.float32)
            with np.load(source_path, allow_pickle=False) as source:
                six = np.asarray(source["initial_state_six"], np.float32)
                reference = np.asarray(source["reference"], np.float32)[1:]
                control = np.asarray(source["current_action"], np.float32)
            if np.any(sigma <= 0):
                raise AssertionError(f"non-positive sigma at {key}")
            if reference.shape != (50, 4):
                raise AssertionError(f"unexpected reference shape {reference.shape}")
            for repeat in range(actor_anchor.shape[0]):
                fields["episode"].append(episode)
                fields["snapshot"].append(path.name)
                fields["split"].append(split)
                fields["scenario"].append(row_plan["scenario_class"])
                fields["speed"].append(row_plan["reference_speed_mps"])
                fields["ordinal"].append(ordinal)
                fields["context"].append(repeat)
                fields["six"].append(six)
                fields["reference"].append(reference.reshape(-1))
                fields["control"].append(control)
                fields["actor_anchor"].append(actor_anchor[repeat].reshape(-1))
                fields["warm_anchor"].append(warm_anchor[repeat].reshape(-1))
                fields["sigma"].append(sigma)
                fields["astar"].append(astar.reshape(-1))
                fields["j16_cost"].append(j16_cost)
    return {name: np.asarray(value) for name, value in fields.items()}


def pairwise_distances(
    blocks: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Block-equal IQR-standardized euclidean distance over given blocks."""
    scales = {name: robust_scale(value) for name, value in blocks.items()}
    pieces = []
    for name, value in blocks.items():
        weight = 1.0 / np.sqrt(value.shape[1])
        pieces.append((value / scales[name]) * weight)
    matrix = np.concatenate(pieces, axis=1).astype(np.float64)
    squared = (
        np.sum(matrix * matrix, axis=1)[:, None]
        + np.sum(matrix * matrix, axis=1)[None, :]
        - 2.0 * (matrix @ matrix.T)
    )
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared), scales


def sigma_rms(delta: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """RMS of sigma-normalized knot deltas, per context.

    delta/sigma are flattened (N, 16) with column order [knot0_d0, knot0_d1,
    ...], so sigma is tiled per knot before normalization.
    """
    tiled = np.repeat(sigma, 8, axis=1) if sigma.shape[1] == 2 else sigma
    normalized = delta / tiled
    return np.sqrt(np.mean(normalized * normalized, axis=1))


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def knn_report(
    labels: np.ndarray,
    topk_indices: np.ndarray,
    ks: range,
    strata: np.ndarray,
    stratum_names: tuple[str, ...],
) -> dict[str, dict]:
    unit = normalized_rows(labels)
    report: dict[str, dict] = {}
    for k in ks:
        per_stratum: dict[str, dict] = {}
        for name in stratum_names:
            rows = np.flatnonzero(strata == name)
            if not len(rows):
                continue
            fixed, oracle, positive = [], [], []
            for row in rows:
                neighbors = topk_indices[row, :k]
                mean_label = unit[neighbors].mean(axis=0)
                fixed.append(float(mean_label @ unit[row]))
                best = max(float(unit[n] @ unit[row]) for n in neighbors)
                oracle.append(best)
                positive.append(best > 0.0)
            per_stratum[name] = {
                "count": int(len(rows)),
                "fixed_knn_cosine_median": float(np.median(fixed)),
                "local_availability_oracle_median": float(np.median(oracle)),
                "local_availability_positive_fraction": float(
                    np.mean(positive)
                ),
            }
        report[str(k)] = per_stratum
    return report


def run_train_bank_mode(args: argparse.Namespace, data: dict[str, np.ndarray]) -> None:
    """Phase 0.1b: train-only neighbor bank, train-fitted normalization.

    Validation queries never see validation neighbors, so the reported
    coherence is the train->validation transfer of the residual labels. The
    validation split remains a historically consumed set: this is a
    historical diagnostic, not an unbiased qualification gate.
    """
    count = len(data["episode"])
    sigma = data["sigma"]
    sigma_tiled = np.repeat(sigma, 8, axis=1)
    bank_mask = data["split"] == "train"
    query_mask = data["split"] == args.query_split
    if not np.any(bank_mask) or not np.any(query_mask):
        raise AssertionError("empty bank or query split")
    bank_episodes = set(data["episode"][bank_mask].tolist())
    if bank_episodes & set(data["episode"][query_mask].tolist()):
        raise AssertionError("query and bank episodes overlap")

    feature_names = ("initial_state_six", "reference", "current_control")
    anchors = {
        "actor": (data["actor_anchor"], (data["astar"] - data["actor_anchor"]) / sigma_tiled),
        "warm": (data["warm_anchor"], (data["astar"] - data["warm_anchor"]) / sigma_tiled),
    }
    speed_strata = np.asarray([
        f"{min(SPEED_LEVELS, key=lambda value: abs(value - speed)):.1f}"
        for speed in data["speed"][query_mask]
    ])
    speed_names = tuple(f"{value:.1f}" for value in SPEED_LEVELS)

    results: dict[str, dict] = {}
    for anchor_name, (anchor, residual) in anchors.items():
        for metric_name, block_names in (
            ("with_a0", feature_names + ("anchor",)),
            ("without_a0", feature_names),
        ):
            blocks = {
                "initial_state_six": data["six"],
                "reference": data["reference"],
                "current_control": data["control"],
                "anchor": anchor,
            }
            scales = {
                name: robust_scale(value[bank_mask])
                for name, value in blocks.items()
                if name in block_names
            }

            def embed(mask: np.ndarray) -> np.ndarray:
                pieces = []
                for name in block_names:
                    value = blocks[name][mask] / scales[name]
                    pieces.append(value * (1.0 / np.sqrt(value.shape[1])))
                return np.concatenate(pieces, axis=1).astype(np.float64)

            query_matrix, bank_matrix = embed(query_mask), embed(bank_mask)
            squared = (
                np.sum(query_matrix * query_matrix, axis=1)[:, None]
                + np.sum(bank_matrix * bank_matrix, axis=1)[None, :]
                - 2.0 * (query_matrix @ bank_matrix.T)
            )
            np.maximum(squared, 0.0, out=squared)
            distances = np.sqrt(squared)

            nearest_index = np.argmin(distances, axis=1)
            nearest_distance = distances[np.arange(len(nearest_index)), nearest_index]
            bank_rows = np.flatnonzero(bank_mask)
            local_of_bank = np.full(count, -1, dtype=np.int64)
            local_of_bank[bank_rows] = np.arange(len(bank_rows))
            nearest_local = local_of_bank[bank_rows[nearest_index]]

            residual_q = residual[query_mask]
            residual_b = residual[bank_mask]
            unit_q = normalized_rows(residual_q)
            unit_b = normalized_rows(residual_b)
            nearest_cosine = np.asarray([
                float(unit_q[i] @ unit_b[local])
                for i, local in enumerate(nearest_local)
            ])
            flip = nearest_cosine < 0.0
            centered_mean = residual[bank_mask].mean(axis=0, keepdims=True)
            unit_qc = normalized_rows(residual_q - centered_mean)
            unit_bc = normalized_rows(residual_b - centered_mean)
            nearest_cosine_centered = np.asarray([
                float(unit_qc[i] @ unit_bc[local])
                for i, local in enumerate(nearest_local)
            ])
            topk = np.argsort(distances, axis=1)[:, : args.max_k]
            topk_global = bank_rows[topk]
            knn_fixed = {str(k): [] for k in range(1, args.max_k + 1)}
            knn_oracle = {str(k): [] for k in range(1, args.max_k + 1)}
            for i in range(len(topk_global)):
                for k in range(1, args.max_k + 1):
                    neighbors = topk_global[i, :k]
                    mean_label = unit_b[local_of_bank[neighbors]].mean(axis=0)
                    knn_fixed[str(k)].append(float(mean_label @ unit_q[i]))
                    knn_oracle[str(k)].append(max(
                        float(unit_b[local] @ unit_q[i])
                        for local in local_of_bank[neighbors]
                    ))
            per_speed = {}
            for name in speed_names:
                mask = speed_strata == name
                per_speed[name] = {
                    "count": int(np.sum(mask)),
                    "nearest_residual_flip_rate": float(np.mean(flip[mask])),
                    "nearest_residual_cosine_median": float(
                        np.median(nearest_cosine[mask])
                    ),
                }
            results[f"{anchor_name}__{metric_name}"] = {
                "queries": int(np.sum(query_mask)),
                "bank_contexts": int(np.sum(bank_mask)),
                "nearest_distance_median": float(np.median(nearest_distance)),
                "nearest_residual_flip_rate": float(np.mean(flip)),
                "nearest_residual_cosine_median": float(
                    np.median(nearest_cosine)
                ),
                "fixed_knn_cosine_median": {
                    k: float(np.median(values))
                    for k, values in knn_fixed.items()
                },
                "local_availability_oracle": {
                    k: {
                        "median": float(np.median(values)),
                        "positive_fraction": float(np.mean(
                            np.asarray(values) > 0
                        )),
                    }
                    for k, values in knn_oracle.items()
                },
                "nearest_flip_by_speed": per_speed,
                "mean_removed": {
                    "flip_rate": float(np.mean(nearest_cosine_centered < 0.0)),
                    "cosine_median": float(
                        np.median(nearest_cosine_centered)
                    ),
                },
            }

    actor_with = results["actor__with_a0"]
    actor_without = results["actor__without_a0"]
    knn5 = actor_with["fixed_knn_cosine_median"]["5"]
    viable = (
        actor_with["nearest_residual_flip_rate"] <= 0.30
        and knn5 >= 0.40
        and (
            actor_without["nearest_residual_flip_rate"]
            - actor_with["nearest_residual_flip_rate"]
        ) >= 0.05
    )
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            "RESIDUAL_COHERENCE_TRAIN_BANK_HISTORICAL_DIAGNOSTIC_"
            "ACTOR_FROZEN"
        ),
        "routing": (
            "RESIDUAL_COHERENCE_VIABLE"
            if viable
            else "CONSERVATIVE_SUPERVISION_REQUIRED"
        ),
        "sources": {
            "replay_labels": str(args.replay_labels),
            "gt_train": str(args.gt_train),
            "gt_validation": str(args.gt_validation),
            "scenario_plan": str(args.scenario_plan),
        },
        "protocol": {
            "mode": "train-only neighbor bank, train-fitted normalization",
            "query_split": args.query_split,
            "bank_split": "train",
            "queries": int(np.sum(query_mask)),
            "bank_contexts": int(np.sum(bank_mask)),
            "note": (
                "validation split is historically consumed; this is a "
                "historical diagnostic, not an unbiased qualification gate"
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=1))
    print(json.dumps({
        "routing": summary["routing"],
        "flip_with_a0": actor_with["nearest_residual_flip_rate"],
        "flip_without_a0": actor_without["nearest_residual_flip_rate"],
        "knn5_cosine_with_a0": knn5,
    }, indent=1))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = build_dataset(args)
    if args.query_split is not None:
        run_train_bank_mode(args, data)
        return
    count = len(data["episode"])
    if count != data["astar"].shape[0]:
        raise AssertionError("context count mismatch")

    sigma = data["sigma"]
    actor_anchor = data["actor_anchor"]
    warm_anchor = data["warm_anchor"]
    astar = data["astar"]
    sigma_tiled = np.repeat(sigma, 8, axis=1)
    residual_actor = (astar - actor_anchor) / sigma_tiled
    residual_warm = (astar - warm_anchor) / sigma_tiled
    residual_norm_actor = np.sqrt(
        np.mean(residual_actor * residual_actor, axis=1)
    )
    residual_norm_warm = np.sqrt(
        np.mean(residual_warm * residual_warm, axis=1)
    )

    episode_codes, episode_index = np.unique(data["episode"], return_inverse=True)
    state_key = np.asarray([
        f"{data['episode'][i]}#{int(data['ordinal'][i])}"
        for i in range(count)
    ])
    state_codes, state_index = np.unique(state_key, return_inverse=True)
    same_episode = episode_index[:, None] == episode_index[None, :]
    same_state = state_index[:, None] == state_index[None, :]
    excluded = same_episode | same_state | np.eye(count, dtype=bool)
    valid = ~excluded

    feature_blocks = {
        "initial_state_six": data["six"],
        "reference": data["reference"],
        "current_control": data["control"],
    }
    speed_strata = np.asarray([
        f"{min(SPEED_LEVELS, key=lambda value: abs(value - speed)):.1f}"
        for speed in data["speed"]
    ])
    split_names = ("train", "validation")
    speed_names = tuple(f"{value:.1f}" for value in SPEED_LEVELS)

    results: dict[str, dict] = {}

    for anchor_name, residual, res_norm in (
        ("actor", residual_actor, residual_norm_actor),
        ("warm", residual_warm, residual_norm_warm),
    ):
        anchor_block = actor_anchor if anchor_name == "actor" else warm_anchor
        for metric_name, blocks in (
            ("with_a0", {**feature_blocks, "anchor": anchor_block}),
            ("without_a0", feature_blocks),
        ):
            distances, scales = pairwise_distances(blocks)
            masked = np.where(valid, distances, np.inf)
            nearest_index = np.argmin(masked, axis=1)
            nearest_distance = masked[np.arange(count), nearest_index]
            finite = np.isfinite(nearest_distance)
            if not np.all(finite):
                raise AssertionError("context without any valid neighbor")

            order = np.argsort(nearest_distance)
            bucket_edges = np.quantile(
                nearest_distance, np.linspace(0, 1, args.buckets + 1)
            )
            bucket_of = np.clip(
                np.searchsorted(bucket_edges, nearest_distance, side="right") - 1,
                0, args.buckets - 1,
            )
            unit = normalized_rows(residual)
            centered = residual - residual.mean(axis=0, keepdims=True)
            unit_centered = normalized_rows(centered)
            nearest_cosine = np.asarray([
                float(unit[i] @ unit[nearest_index[i]]) for i in range(count)
            ])
            nearest_cosine_centered = np.asarray([
                float(
                    unit_centered[i] @ unit_centered[nearest_index[i]]
                )
                for i in range(count)
            ])
            flip = nearest_cosine < 0.0
            flip_centered = nearest_cosine_centered < 0.0
            steering_dims = [1, 3, 5]
            steer_i = normalized_rows(residual[:, steering_dims])
            nearest_steer_cosine = np.asarray([
                float(steer_i[i] @ steer_i[nearest_index[i]])
                for i in range(count)
            ])
            steer_flip = nearest_steer_cosine < 0.0
            astar_distance = sigma_rms(
                astar[nearest_index] - astar, sigma
            )
            jump = astar_distance > 1.0

            buckets_report = {}
            for bucket in range(args.buckets):
                mask = bucket_of == bucket
                total = int(np.sum(mask))
                buckets_report[str(bucket)] = {
                    "distance_low": float(bucket_edges[bucket]),
                    "distance_high": float(bucket_edges[bucket + 1]),
                    "count": total,
                    "flip_rate": float(np.mean(flip[mask])) if total else None,
                    "wilson_95": wilson_interval(int(np.sum(flip[mask])), total),
                    "residual_cosine_median": (
                        float(np.median(nearest_cosine[mask])) if total else None
                    ),
                    "absolute_center_jump_over_1sigma_fraction": (
                        float(np.mean(jump[mask])) if total else None
                    ),
                }

            topk_indices = np.argsort(masked, axis=1)[:, : args.max_k]
            per_speed = knn_report(
                residual, topk_indices, range(1, args.max_k + 1),
                speed_strata, speed_names,
            )
            per_split = knn_report(
                residual, topk_indices, range(1, args.max_k + 1),
                data["split"], split_names,
            )
            overall = knn_report(
                residual, topk_indices, range(1, args.max_k + 1),
                np.asarray(["ALL"] * count), ("ALL",),
            )
            speed_flip = {}
            for name in speed_names:
                mask = speed_strata == name
                speed_flip[name] = {
                    "count": int(np.sum(mask)),
                    "nearest_residual_flip_rate": float(np.mean(flip[mask])),
                    "nearest_residual_cosine_median": float(
                        np.median(nearest_cosine[mask])
                    ),
                }
            norm_quartiles = np.quantile(res_norm, [0.25, 0.5, 0.75])
            norm_bucket = np.digitize(res_norm, norm_quartiles)
            norm_quartile_report = {}
            for bucket in range(4):
                mask = norm_bucket == bucket
                norm_quartile_report[str(bucket)] = {
                    "norm_range": [
                        float(norm_quartiles[bucket - 1]) if bucket else 0.0,
                        float(norm_quartiles[bucket]) if bucket < 3 else None,
                    ],
                    "count": int(np.sum(mask)),
                    "nearest_residual_flip_rate": float(np.mean(flip[mask])),
                    "nearest_abs_cosine_median": float(
                        np.median(np.abs(nearest_cosine[mask]))
                    ),
                }

            results[f"{anchor_name}__{metric_name}"] = {
                "nearest_distance_median": float(np.median(nearest_distance)),
                "nearest_residual_flip_rate": float(np.mean(flip)),
                "nearest_residual_cosine_median": float(
                    np.median(nearest_cosine)
                ),
                "nearest_absolute_center_jump_over_1sigma_fraction": float(
                    np.mean(jump)
                ),
                "nearest_absolute_center_distance_sigma_rms_median": float(
                    np.median(astar_distance)
                ),
                "buckets": buckets_report,
                "knn_by_speed": per_speed,
                "knn_by_split": per_split,
                "knn_overall": overall,
                "nearest_flip_by_speed": speed_flip,
                "post_hoc_robustness": {
                    "note": (
                        "computed after headline numbers; guards the "
                        "small-norm-noise, marginal-cosine, and global-bias "
                        "readings"
                    ),
                    "flip_by_residual_norm_quartile": norm_quartile_report,
                    "nearest_abs_cosine_median": float(
                        np.median(np.abs(nearest_cosine))
                    ),
                    "mean_removed": {
                        "flip_rate": float(np.mean(flip_centered)),
                        "cosine_median": float(
                            np.median(nearest_cosine_centered)
                        ),
                        "purpose": (
                            "removes any dataset-common residual direction; "
                            "low flip here means state-specific coherence"
                        ),
                    },
                    "front_steering_knots_0_2": {
                        "dims": steering_dims,
                        "flip_rate": float(np.mean(steer_flip)),
                        "cosine_median": float(
                            np.median(nearest_steer_cosine)
                        ),
                    },
                },
            }

    # Adjacent within-episode J16 transitions (historical 68.8% analog).
    adjacent = []
    episode_of = data["episode"]
    ordinal_of = data["ordinal"].astype(int)
    context_of = data["context"].astype(int)
    index_by_key = {
        (episode_of[i], ordinal_of[i], context_of[i]): i for i in range(count)
    }
    for i in range(count):
        partner = index_by_key.get(
            (episode_of[i], ordinal_of[i] + 1, context_of[i])
        )
        if partner is None:
            continue
        adjacent.append((i, partner))
    adjacent_jumps, adjacent_residual_shift = [], []
    for i, partner in adjacent:
        adjacent_jumps.append(
            float(sigma_rms(astar[partner] - astar[i], sigma[i:i + 1])[0])
        )
        shift = (residual_actor[partner] - residual_actor[i]) / np.sqrt(2.0)
        adjacent_residual_shift.append(float(np.sqrt(np.mean(shift * shift))))
    adjacent_jumps = np.asarray(adjacent_jumps)
    adjacent_residual_shift = np.asarray(adjacent_residual_shift)

    with_a0 = results["actor__with_a0"]
    without_a0 = results["actor__without_a0"]
    flip_with = with_a0["nearest_residual_flip_rate"]
    flip_without = without_a0["nearest_residual_flip_rate"]
    knn5_cosine = with_a0["knn_overall"]["5"]["ALL"][
        "fixed_knn_cosine_median"
    ]
    viable = (
        flip_with <= 0.30
        and knn5_cosine >= 0.40
        and (flip_without - flip_with) >= 0.05
    )
    routing = (
        "RESIDUAL_COHERENCE_VIABLE"
        if viable
        else "CONSERVATIVE_SUPERVISION_REQUIRED"
    )

    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "RESIDUAL_COHERENCE_PHASE0_DIAGNOSIS_ACTOR_FROZEN",
        "routing": routing,
        "routing_rule": {
            "viable_if": (
                "nearest flip <= 0.30 AND k5 fixed-KNN cosine median >= 0.40 "
                "AND removing the a0 block raises flip >= 0.05"
            ),
            "flip_with_a0": flip_with,
            "flip_without_a0": flip_without,
            "knn5_fixed_cosine_median_with_a0": knn5_cosine,
        },
        "sources": {
            "replay_labels": str(args.replay_labels),
            "replay_splits_sha256": sha256_file(
                args.replay_labels / "splits.json"
            ),
            "gt_train": str(args.gt_train),
            "gt_train_summary_sha256": sha256_file(
                args.gt_train / "summary.json"
            ),
            "gt_validation": str(args.gt_validation),
            "gt_validation_summary_sha256": sha256_file(
                args.gt_validation / "summary.json"
            ),
            "scenario_plan": str(args.scenario_plan),
            "scenario_plan_sha256": sha256_file(args.scenario_plan),
        },
        "protocol": {
            "contexts": int(count),
            "snapshots": int(len(set(state_key.tolist()))),
            "episodes": int(len(set(data["episode"].tolist()))),
            "splits": {
                name: int(np.sum(data["split"] == name))
                for name in split_names
            },
            "feature_blocks": [
                "initial_state_six", "reference[1:]_50x4", "current_control",
                "anchor(conditional)",
            ],
            "normalization": (
                "per-dimension IQR (std fallback), blocks weighted equally"
            ),
            "exclusions": "same episode, same physical snapshot, self",
            "anchor_definitions": (
                "actor=bootstrap_actor_center from replay labels; "
                "warm=anchor_center; residuals sigma-normalized per knot dim"
            ),
            "buckets": args.buckets,
            "max_k": args.max_k,
            "reference_flip_rate_baseline": (
                "own-center gradient nearest-neighbor flip 0.43-0.46 "
                "(g0_neighbor_oracle_20260817_v1)"
            ),
            "note": (
                "label-space coherence only; translated-candidate cost "
                "verification requires rollouts and is deferred to Phase 1"
            ),
        },
        "residual_magnitude_sigma_rms": {
            "actor_anchor": {
                "median": float(np.median(residual_norm_actor)),
                "p10": float(np.quantile(residual_norm_actor, 0.10)),
                "p90": float(np.quantile(residual_norm_actor, 0.90)),
            },
            "warm_anchor": {
                "median": float(np.median(residual_norm_warm)),
                "p10": float(np.quantile(residual_norm_warm, 0.10)),
                "p90": float(np.quantile(residual_norm_warm, 0.90)),
            },
        },
        "adjacent_transitions": {
            "count": int(len(adjacent_jumps)),
            "absolute_center_jump_over_1sigma_fraction": float(
                np.mean(adjacent_jumps > 1.0)
            ),
            "absolute_center_distance_sigma_rms_median": float(
                np.median(adjacent_jumps)
            ),
            "actor_residual_shift_sigma_rms_median": float(
                np.median(adjacent_residual_shift)
            ),
            "historical_reference": (
                "68.8% of validation transitions exceeded one sigma "
                "(2026-08-07 J16 distillation handoff)"
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=1))
    print(json.dumps({
        "routing": routing,
        "flip_with_a0": flip_with,
        "flip_without_a0": flip_without,
        "knn5_cosine_with_a0": knn5_cosine,
        "contexts": count,
    }, indent=1))


if __name__ == "__main__":
    main()
