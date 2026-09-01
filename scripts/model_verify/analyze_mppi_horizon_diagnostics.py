#!/usr/bin/env python3
"""Horizon-level gradient diagnostics (parallel to Phase 2, non-blocking).

Part 1 - along/cross decomposition (review priority): the position term
w_pos * sum_t |p_t - p_ref,t|^2 splits exactly into along/cross components
using the reference heading at the SAME time index,
|dp|^2 = (dp.u)^2 + (dp.v)^2, with no nearest-point matching or Frenet index
switching. Among the position-attributed gradient flips of the 11.42 audit,
classify whether the flip comes from the cross component itself, the along
component itself, or their competition.

Part 2 - horizon weighting sweep: per-step action gradients g_t allow
zero-rollout recomposition grad(J_w) = sum_t w_t g_t for candidate temporal
weightings. Screening reports flip rate and cosine under the SAME nearest-
neighbor pairs as the audit. Per the review, this is screening only: any
candidate must later be re-rolled under the original uniform J50 (gain,
P05, worst, stay) before it can be considered; lowering the flip rate alone
may just change the objective.

Part 3 - gradient margin rho = ||sum_t g_t|| / (sum_t ||g_t|| + eps): a
bounded temporal-cancellation diagnostic. Low rho marks heavy horizon
cancellation. Diagnostic only: it never becomes an Actor input and Query
models take no gradient dependency.

Actor stays frozen; no formal validation or test data is read.
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
from generate_dbm_direct_gt_validation import interpolate_knots
from generate_dbm_proposal_teacher import sha256_file
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
DEFAULT_ATTRIBUTION = Path(
    "outputs/mppi_proposal/cost_term_flip_attribution_20260817_v1"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/horizon_diagnostics_20260818_v1"
)
FLIP_COS_GATE = -0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-actor", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--fresh-npz", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--attribution", type=Path, default=DEFAULT_ATTRIBUTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)


def weighting(name: str, horizon: int) -> np.ndarray:
    index = np.arange(horizon, dtype=np.float64)
    if name == "uniform":
        values = np.ones(horizon)
    elif name == "ramp_up":
        values = index / max(horizon - 1, 1)
    elif name == "ramp_down":
        values = 1.0 - index / max(horizon - 1, 1)
    elif name == "exp_decay_0.95":
        values = 0.95 ** index
    elif name == "exp_decay_0.90":
        values = 0.90 ** index
    elif name == "exp_growth_1.02":
        values = 1.02 ** index
    elif name == "late_half":
        values = (index >= horizon // 2).astype(np.float64)
    elif name == "front_half":
        values = (index < horizon // 2).astype(np.float64)
    elif name == "drop_first_10":
        values = (index >= 10).astype(np.float64)
    else:
        raise ValueError(name)
    return (values / np.mean(values)).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    initial = torch.load(args.initial_actor, map_location="cpu")
    alpha_payload = torch.load(initial["base_alpha_checkpoint"], map_location="cpu")
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, _ = load_dataset(Path(initial["labels"]), old_payload)
    fresh = dict(np.load(args.fresh_npz, allow_pickle=False))
    manifest = json.loads(args.manifest.read_text())
    row_by_context = {int(row["context_index"]): row for row in manifest["rows"]}
    contexts = fresh["context_index"].astype(np.int64)
    count = len(contexts)
    stratum = np.asarray([
        row_by_context[int(value)]["stratum_id"] for value in contexts
    ])

    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**data.dbm_params)
    )
    weights = data.cost_weights
    horizon = backend.horizon

    attribution = np.load(args.attribution / "term_gradients.npz", allow_pickle=False)
    nearest_index = attribution["nearest_index"]
    flip = attribution["flip"]
    attribution_code = attribution["single_term_attribution_code"]
    net_reference = attribution["net_reward_gradient"].astype(np.float32)

    step_grads = np.zeros((count, horizon, 16), np.float32)
    along_grads = np.zeros((count, 16), np.float32)
    cross_grads = np.zeros((count, 16), np.float32)
    identity_errors = []
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
        delta = trajectory[:, :2] - reference[:, :2]
        heading = reference[:, 2]
        along = torch.cos(heading)
        normal = torch.sin(heading)
        along_component = (delta[:, 0] * along + delta[:, 1] * normal) ** 2
        cross_component = (delta[:, 0] * -normal + delta[:, 1] * along) ** 2
        position_square = (delta ** 2).sum(dim=1)
        identity_errors.append(float(torch.max(
            torch.abs(along_component + cross_component - position_square)
        )))
        if identity_errors[-1] > 1e-4:
            raise AssertionError("along/cross identity violated")
        yaw_delta = trajectory[:, 2] - reference[:, 2]
        yaw = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)) ** 2
        vx = (trajectory[:, 3] - reference[:, 3]) ** 2
        current = torch.from_numpy(
            data.current_action[contexts[position]].astype(np.float32)
        ).to(device)
        previous = torch.cat((current[None], actions[0, :-1]), dim=0)
        steer_rate = (actions[0, :, 1] - previous[:, 1]) ** 2
        accel_rate = (actions[0, :, 0] - previous[:, 0]) ** 2

        along_loss = weights["position"] * along_component.sum()
        cross_loss = weights["position"] * cross_component.sum()
        along_grads[position] = torch.autograd.grad(
            along_loss, knots, retain_graph=True
        )[0].detach().cpu().numpy().flatten()
        cross_grads[position] = torch.autograd.grad(
            cross_loss, knots, retain_graph=True
        )[0].detach().cpu().numpy().flatten()

        per_step = (
            weights["position"] * position_square
            + weights["yaw"] * yaw
            + weights["vx"] * vx
            + weights["steering_rate"] * steer_rate
            + weights["acceleration_rate"] * accel_rate
        )
        for step in range(horizon):
            step_grads[position, step] = torch.autograd.grad(
                per_step[step], knots, retain_graph=True
            )[0].detach().cpu().numpy().flatten()
        if position % 100 == 0:
            print(f"[{position:03d}/{count:03d}] gradients done", flush=True)

    recomposed = step_grads.sum(axis=1)
    recomposed_error = np.asarray([
        float(unit(recomposed[i]) @ unit(-net_reference[i]))
        for i in range(count)
    ])
    if float(np.median(recomposed_error)) < 0.99:
        raise AssertionError("per-step recomposition disagrees with stored net")

    # Part 1: along/cross classification of position-attributed flips.
    position_flips = np.flatnonzero(flip & (attribution_code == 0))
    classification = {
        "cross_only": 0, "along_only": 0, "both": 0, "neither": 0,
    }
    for i in position_flips:
        j = int(nearest_index[i])
        cross_cos = float(unit(cross_grads[i]) @ unit(cross_grads[j]))
        along_cos = float(unit(along_grads[i]) @ unit(along_grads[j]))
        cross_flip = cross_cos < FLIP_COS_GATE
        along_flip = along_cos < FLIP_COS_GATE
        if cross_flip and along_flip:
            classification["both"] += 1
        elif cross_flip:
            classification["cross_only"] += 1
        elif along_flip:
            classification["along_only"] += 1
        else:
            classification["neither"] += 1
    position_flips_count = len(position_flips)

    # Part 2: weighting sweep screening (same neighbor pairs as the audit).
    sweep = {}
    for name in (
        "uniform", "ramp_up", "ramp_down", "exp_decay_0.95", "exp_decay_0.90",
        "exp_growth_1.02", "late_half", "front_half", "drop_first_10",
    ):
        values = weighting(name, horizon)
        weighted = (step_grads * values[None, :, None]).sum(axis=1)
        unit_weighted = np.stack([unit(row) for row in weighted])
        cosines = np.asarray([
            float(unit_weighted[i] @ unit_weighted[int(nearest_index[i])])
            for i in range(count)
        ])
        flip_rate = float(np.mean(cosines < 0.0))
        per_stratum = {}
        for name_stratum in np.unique(stratum):
            mask = stratum == name_stratum
            per_stratum[str(name_stratum)] = {
                "count": int(np.sum(mask)),
                "flip_rate": float(np.mean(cosines[mask] < 0.0)),
            }
        sweep[name] = {
            "flip_rate_all": flip_rate,
            "cosine_median": float(np.median(cosines)),
            "by_stratum": per_stratum,
        }

    # Part 3: gradient margin rho.
    norms = np.linalg.norm(step_grads, axis=2)
    rho = np.linalg.norm(step_grads.sum(axis=1), axis=1) / (
        norms.sum(axis=1) + 1e-12
    )
    rho_flip = rho[flip]
    rho_same = rho[~flip]

    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HORIZON_DIAGNOSTICS_ACTOR_FROZEN",
        "sources": {
            "initial_actor": str(args.initial_actor),
            "fresh_npz": str(args.fresh_npz),
            "attribution_npz": str(args.attribution / "term_gradients.npz"),
            "attribution_sha256": sha256_file(
                args.attribution / "term_gradients.npz"
            ),
        },
        "validation": {
            "recomposition_vs_net_cosine_median": float(
                np.median(recomposed_error)
            ),
            "along_cross_identity_max_error": float(np.max(identity_errors)),
        },
        "sources_sha256": {
            "fresh_npz": sha256_file(args.fresh_npz),
            "initial_actor": sha256_file(args.initial_actor),
            "manifest": sha256_file(args.manifest),
        },
        "part1_along_cross": {
            "position_attributed_flips": position_flips_count,
            "classification": classification,
            "flip_cos_gate": FLIP_COS_GATE,
            "interpretation_note": (
                "cross-driven flips indicate lateral correction ambiguity "
                "(physical semantics); along-driven or competition flips "
                "indicate horizon bookkeeping, relevant to whether an s-d "
                "reparameterization would be a semantic improvement or "
                "merely an easier-to-learn objective"
            ),
        },
        "part2_weighting_sweep": {
            "screening_only": True,
            "verification_pending": (
                "candidates must be re-rolled under the original uniform "
                "J50 (real gain, P05, worst, stay) before any consideration; "
                "lower flip rate alone may only change the objective"
            ),
            "results": sweep,
        },
        "part3_gradient_margin": {
            "definition": "rho = ||sum_t g_t|| / (sum_t ||g_t|| + eps)",
            "median": float(np.median(rho)),
            "p10": float(np.quantile(rho, 0.10)),
            "median_flip_states": float(np.median(rho_flip)) if len(rho_flip) else None,
            "median_same_states": float(np.median(rho_same)) if len(rho_same) else None,
            "usage_note": (
                "confidence diagnostic for stay/small-step/fallback "
                "identification only; never an Actor input; Query models "
                "take no gradient dependency"
            ),
        },
        "boundary": (
            "parallel to Phase 2 and non-blocking: the formal Actor still "
            "learns guarded teachers under the original J50; the formal cost "
            "changes only if a new definition wins on original J50, "
            "actor-centered MPPI, and the short closed loop"
        ),
    }
    args.output.mkdir(parents=True)
    (args.output / "analysis.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(
        args.output / "per_state.npz",
        step_grads=step_grads,
        along_grads=along_grads,
        cross_grads=cross_grads,
        rho=rho.astype(np.float32),
    )
    print(json.dumps({
        "part1": summary["part1_along_cross"],
        "part2_flip_rates": {
            name: round(value["flip_rate_all"], 3)
            for name, value in sweep.items()
        },
        "part3": {
            "rho_median": round(summary["part3_gradient_margin"]["median"], 3),
            "rho_flip": round(summary["part3_gradient_margin"]["median_flip_states"], 3) if summary["part3_gradient_margin"]["median_flip_states"] is not None else None,
            "rho_same": round(summary["part3_gradient_margin"]["median_same_states"], 3) if summary["part3_gradient_margin"]["median_same_states"] is not None else None,
        },
    }, indent=1))


if __name__ == "__main__":
    main()
