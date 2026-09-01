#!/usr/bin/env python3
"""Cost-term flip attribution with threshold sensitivity and bootstrap CIs.

Solidifies the §11.41 inline audit into a reproducible artifact:
- Per-term autograd action gradients (position/yaw/vx/rates) for all 600
  internal-selection contexts, cross-checked against fresh-FD labels.
- Nearest-neighbor pairs under the §11.36 full-Markov distance (same-episode
  excluded), deterministic.
- Attribution at a 3x3x3 threshold grid (term cosine -0.1/-0.3/-0.5, share
  0.3/0.4/0.5, cancellation gate 0.4/0.6/0.8): single-term flip attribution
  and combined-mechanism coverage, each with a same-direction baseline.
- Episode-level bootstrap CIs for the headline rates (adjacent frames from
  one episode are not independent samples).
- All term gradients, per-pair flags, weights, and input hashes stored.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights
from generate_dbm_direct_gt_validation import interpolate_knots
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
    "outputs/mppi_proposal/cost_term_flip_attribution_20260817_v1"
)
TERMS = ("position", "yaw", "vx", "steer_rate", "accel_rate")
MAIN3 = ("position", "yaw", "vx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-actor", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--fresh-npz", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=260817)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    initial = torch.load(args.initial_actor, map_location="cpu")
    alpha_payload = torch.load(
        initial["base_alpha_checkpoint"], map_location="cpu"
    )
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, _ = load_dataset(Path(initial["labels"]), old_payload)
    fresh = dict(np.load(args.fresh_npz, allow_pickle=False))
    manifest = json.loads(args.manifest.read_text())
    row_by_context = {
        int(row["context_index"]): row for row in manifest["rows"]
    }
    contexts = fresh["context_index"].astype(np.int64)
    count = len(contexts)
    fd_gradient = fresh["gradient"].astype(np.float32)
    episode = fresh["episode"].astype(str)
    stratum = np.asarray([
        row_by_context[int(value)]["stratum_id"] for value in contexts
    ])

    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**data.dbm_params)
    )
    weights = data.cost_weights
    horizon = backend.horizon

    term_grads = {name: np.zeros((count, 16), np.float32) for name in TERMS}
    for position in range(count):
        knots = torch.from_numpy(
            fresh["actor_center"][position].astype(np.float32)
        ).to(device)[None]
        knots.requires_grad_(True)
        actions = interpolate_knots(knots, horizon)
        state = torch.from_numpy(
            data.initial_state_six[contexts[position]].astype(np.float32)
        ).to(device)[None]
        full = backend.rollout_full_state_differentiable(state, actions)
        trajectory = full[0][:, [0, 1, 2, 3, 5]]
        if trajectory.shape[0] != horizon:
            trajectory = trajectory[:horizon]
        reference = torch.from_numpy(
            data.direct_reference[contexts[position]].astype(np.float32)
        ).to(device)
        if reference.shape[0] == trajectory.shape[0] + 1:
            reference = reference[1:]
        pos = weights["position"] * (
            (trajectory[:, :2] - reference[:, :2]) ** 2
        ).sum()
        yaw_delta = trajectory[:, 2] - reference[:, 2]
        yaw = weights["yaw"] * (
            torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)) ** 2
        ).sum()
        vx = weights["vx"] * (
            (trajectory[:, 3] - reference[:, 3]) ** 2
        ).sum()
        current = torch.from_numpy(
            data.current_action[contexts[position]].astype(np.float32)
        ).to(device)
        previous = torch.cat((current[None], actions[0, :-1]), dim=0)
        steer_rate = weights["steering_rate"] * (
            (actions[0, :, 1] - previous[:, 1]) ** 2
        ).sum()
        accel_rate = weights["acceleration_rate"] * (
            (actions[0, :, 0] - previous[:, 0]) ** 2
        ).sum()
        for name, value in (
            ("position", pos), ("yaw", yaw), ("vx", vx),
            ("steer_rate", steer_rate), ("accel_rate", accel_rate),
        ):
            gradient = torch.autograd.grad(value, knots, retain_graph=True)[0]
            term_grads[name][position] = gradient.detach().cpu().numpy().flatten()

    net_reward_gradient = -sum(term_grads.values())
    cross_check = cosine_rows(net_reward_gradient, fd_gradient)

    magnitude = {name: np.linalg.norm(term_grads[name], axis=1) for name in MAIN3}
    cancellation = np.linalg.norm(net_reward_gradient, axis=1) / (
        sum(magnitude.values()) + 1e-12
    )

    features = np.concatenate((
        data.initial_state_six[contexts].astype(np.float32),
        data.direct_reference[contexts].astype(np.float32).reshape(count, -1),
        data.current_action[contexts].astype(np.float32),
        fresh["actor_center"].astype(np.float32).reshape(count, -1),
    ), axis=1)
    quarter = np.quantile(features, [0.25, 0.75], axis=0)
    scale = quarter[1] - quarter[0]
    scale = np.where(
        scale > 1e-9, scale, np.std(features, axis=0) + 1e-9
    )
    standardized = features / scale
    distance = np.linalg.norm(
        standardized[:, None, :] - standardized[None, :, :], axis=2
    )
    same = (episode[:, None] == episode[None, :]) | np.eye(count, dtype=bool)
    distance = np.where(same, np.inf, distance)
    nearest = np.argmin(distance, axis=1)

    normalized = fd_gradient / (
        np.linalg.norm(fd_gradient, axis=1, keepdims=True) + 1e-12
    )
    flip = np.asarray([
        float(normalized[i] @ normalized[nearest[i]]) < 0.0
        for i in range(count)
    ])
    dominant = np.argmax(
        np.stack([magnitude[name] for name in MAIN3]), axis=0
    )

    def attribution(cos_gate: float, share_gate: float):
        flags = np.zeros(count, bool)
        which = [None] * count
        for i in range(count):
            j = nearest[i]
            cross = [
                float(cosine_rows(
                    term_grads[name][i : i + 1], term_grads[name][j : j + 1]
                )[0])
                for name in MAIN3
            ]
            worst = int(np.argmin(cross))
            share = max(
                magnitude[MAIN3[worst]][i]
                / (sum(magnitude[name][i] for name in MAIN3) + 1e-9),
                magnitude[MAIN3[worst]][j]
                / (sum(magnitude[name][j] for name in MAIN3) + 1e-9),
            )
            if cross[worst] < cos_gate and share > share_gate:
                flags[i] = True
                which[i] = MAIN3[worst]
        return flags, which

    grid = {
        "cosine_gate": [-0.1, -0.3, -0.5],
        "share_gate": [0.3, 0.4, 0.5],
        "cancellation_gate": [0.4, 0.6, 0.8],
    }
    default_flags, default_which = attribution(-0.3, 0.4)
    dom_differs = dominant != dominant[nearest]
    sensitivity = []
    for cos_gate in grid["cosine_gate"]:
        for share_gate in grid["share_gate"]:
            flags, _ = attribution(cos_gate, share_gate)
            for cancel_gate in grid["cancellation_gate"]:
                combined = (
                    flags | dom_differs
                    | (np.minimum(cancellation, cancellation[nearest]) < cancel_gate)
                )
                sensitivity.append({
                    "cosine_gate": cos_gate,
                    "share_gate": share_gate,
                    "cancellation_gate": cancel_gate,
                    "single_term_flip_rate_flipped": float(flags[flip].mean()),
                    "single_term_flip_rate_same": float(flags[~flip].mean()),
                    "combined_rate_flipped": float(combined[flip].mean()),
                    "combined_rate_same": float(combined[~flip].mean()),
                })

    unique_episodes = np.unique(episode)
    episode_of = {name: index for index, name in enumerate(unique_episodes)}
    episode_index = np.asarray([episode_of[name] for name in episode])
    rng = np.random.default_rng(args.seed)
    headline = {
        "single_term_flip_rate_flipped": float(default_flags[flip].mean()),
        "single_term_flip_rate_same": float(default_flags[~flip].mean()),
    }
    combined_default = (
        default_flags | dom_differs
        | (np.minimum(cancellation, cancellation[nearest]) < 0.6)
    )
    headline["combined_rate_flipped"] = float(combined_default[flip].mean())
    headline["combined_rate_same"] = float(combined_default[~flip].mean())
    bootstrap = {}
    for key in headline:
        if key.endswith("same"):
            mask = ~flip
        else:
            mask = flip
        values = []
        for _ in range(args.bootstrap):
            draw = rng.integers(len(unique_episodes), size=len(unique_episodes))
            # Cluster bootstrap with replacement.  Preserve how many times each
            # episode was drawn; np.isin(draw) would silently turn this into
            # subsampling without replacement whenever an episode repeats.
            cluster_count = np.bincount(
                draw, minlength=len(unique_episodes)
            ).astype(np.float64)
            row_weight = cluster_count[episode_index]
            local_weight = row_weight * mask.astype(np.float64)
            if local_weight.sum() == 0:
                continue
            source = default_flags if key.startswith("single") else combined_default
            values.append(float(np.sum(local_weight * source) / local_weight.sum()))
        bootstrap[key] = {
            "replicates": len(values),
            "method": "episode_cluster_bootstrap_with_replacement",
            "ci_95": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
        }

    def attribution_counts(mask: np.ndarray) -> dict[str, int]:
        counts: dict[str, int] = {}
        for keep, name in zip(mask, default_which):
            if keep and name is not None:
                counts[name] = counts.get(name, 0) + 1
        return counts

    attribution_code = np.asarray([
        -1 if name is None else MAIN3.index(name) for name in default_which
    ], dtype=np.int8)
    attribution_by_term = {
        "flipped_pairs": attribution_counts(flip),
        "same_direction_pairs": attribution_counts(~flip),
        "all_pairs": attribution_counts(np.ones(count, dtype=bool)),
    }

    np.savez_compressed(
        args.output_dir / "term_gradients.npz",
        **term_grads,
        net_reward_gradient=net_reward_gradient,
        cancellation_ratio=cancellation,
        fd_cross_check_cosine=cross_check,
        nearest_index=nearest,
        flip=flip,
        single_term_flag=default_flags,
        single_term_attribution_code=attribution_code,
    )
    result = {
        "format_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "COST_TERM_FLIP_ATTRIBUTION_AUDITED_ACTOR_FROZEN",
        "sources": {
            "initial_actor": str(args.initial_actor.resolve()),
            "initial_actor_sha256": sha256_file(args.initial_actor),
            "fresh_fd": str(args.fresh_npz.resolve()),
            "fresh_fd_sha256": sha256_file(args.fresh_npz),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
        },
        "cost_weights": weights,
        "fd_cross_check": {
            "cosine_median": float(np.median(cross_check)),
            "cosine_p10": float(np.quantile(cross_check, 0.10)),
        },
        "counts": {
            "contexts": int(count),
            "flipped_pairs": int(flip.sum()),
            "same_pairs": int((~flip).sum()),
            "episodes": int(len(unique_episodes)),
        },
        "strata": {
            name: {
                "count": int(np.sum(stratum == name)),
                "flip_rate": float(flip[stratum == name].mean()),
                "cancellation_median": float(
                    np.median(cancellation[stratum == name])
                ),
            }
            for name in np.unique(stratum)
        },
        "defaults": {
            "cosine_gate": -0.3, "share_gate": 0.4,
            "cancellation_gate": 0.6,
        },
        "headline_rates": headline,
        "bootstrap_ci_95": bootstrap,
        "attribution_by_term": attribution_by_term,
        "threshold_sensitivity": sensitivity,
        "contract": {
            "actor_frozen": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "artifact_term_gradients": "term_gradients.npz",
        },
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({
        "output": str((args.output_dir / "analysis.json").resolve()),
        "fd_cross_check_median": result["fd_cross_check"]["cosine_median"],
        "headline": headline,
        "bootstrap_ci_95": bootstrap,
        "attribution_by_term": attribution_by_term,
        "sensitivity_corners": [
            entry for entry in sensitivity
            if entry["cosine_gate"] in (-0.1, -0.5)
            and entry["share_gate"] in (0.3, 0.5)
            and entry["cancellation_gate"] in (0.4, 0.8)
        ][:4],
    }, indent=2))


if __name__ == "__main__":
    main()
