#!/usr/bin/env python3
"""Independently validate routed same-state multidirectional DBM labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from generate_dbm_proposal_teacher import sha256_file
from train_mppi_direct_trust_region_actor import (
    direct_cost,
    load_actor_payload,
    load_dataset,
    tensorize,
)


DEFAULT_INPUT = Path(
    "outputs/mppi_proposal/targeted_local_response_labels_20260814_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--replay-count", type=int, default=1024)
    parser.add_argument("--cost-atol", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def independent_fit(
    action: np.ndarray, reward: np.ndarray, anchor: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gradients, curvatures, ranks = [], [], []
    for local_action, local_reward, local_anchor in zip(action, reward, anchor):
        delta = (local_action - local_anchor[None]).reshape(len(local_action), 16)
        design = np.concatenate(
            (delta, 0.5 * np.sum(np.square(delta), axis=1, keepdims=True)),
            axis=1,
        )[1:]
        target = local_reward[1:] - local_reward[0]
        ranks.append(int(np.linalg.matrix_rank(design)))
        coefficient = np.linalg.lstsq(design, target, rcond=None)[0]
        gradients.append(coefficient[:16])
        curvatures.append(coefficient[16])
    return (
        np.asarray(gradients, np.float32),
        np.asarray(curvatures, np.float32),
        np.asarray(ranks, np.int64),
    )


def maximum_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def main() -> None:
    args = parse_args()
    summary_path = args.input_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    arrays_path = Path(summary["arrays"])
    routing_path = Path(summary["sources"]["routing_manifest"])
    initial_path = Path(summary["sources"]["initial_actor"])
    local_labels_path = Path(summary["sources"]["local_labels"])
    source_hash_checks = {
        "arrays": sha256_file(arrays_path) == summary["arrays_sha256"],
        "routing_manifest": sha256_file(routing_path)
        == summary["sources"]["routing_manifest_sha256"],
        "initial_actor": sha256_file(initial_path)
        == summary["sources"]["initial_actor_sha256"],
        "local_labels": sha256_file(local_labels_path)
        == summary["sources"]["local_labels_sha256"],
    }
    routing = json.loads(routing_path.read_text())
    selected_rows = [row for row in routing["rows"] if row["selected_for_probe_pilot"]]

    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    context_index = arrays["context_index"].astype(np.int64)
    expected_context = np.asarray(
        [row["context_index"] for row in selected_rows], np.int64
    )
    context_order_exact = bool(np.array_equal(context_index, expected_context))
    roles_exact = bool(
        np.array_equal(
            arrays["pilot_role"],
            np.asarray([row["pilot_role"] for row in selected_rows]),
        )
    )
    strata_exact = bool(
        np.array_equal(
            arrays["stratum_id"],
            np.asarray([row["stratum_id"] for row in selected_rows]),
        )
    )

    actor_payload = torch.load(initial_path, map_location="cpu")
    alpha_payload = torch.load(actor_payload["base_alpha_checkpoint"], map_location="cpu")
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, splits = load_dataset(Path(actor_payload["labels"]), old_payload)
    source_split_exact = bool(
        np.all(np.isin(data.episodes[context_index], splits["internal_selection"]))
        and not np.any(
            np.isin(
                data.episodes[context_index],
                splits["formal_validation_sealed_not_generated"],
            )
        )
        and not np.any(
            np.isin(
                data.episodes[context_index],
                splits["test_sealed_not_generated"],
            )
        )
    )
    maximum_residual_sigma = float(actor_payload["maximum_residual_sigma"])
    alpha = arrays["alpha_center"].astype(np.float32)
    sigma = arrays["sigma"].astype(np.float32)
    low = np.maximum(
        -1.0, alpha - maximum_residual_sigma * sigma[:, None, :]
    ).astype(np.float32)
    high = np.minimum(
        1.0, alpha + maximum_residual_sigma * sigma[:, None, :]
    ).astype(np.float32)
    low_error = maximum_abs(low, arrays["action_low"])
    high_error = maximum_abs(high, arrays["action_high"])

    directions = arrays["response_directions"].astype(np.float32)
    radii = arrays["outer_radii_sigma"].astype(np.float32)
    actor_center = arrays["actor_center"].astype(np.float32)
    raw_outer_blocks = [actor_center[:, None]]
    for radius in radii:
        offset = float(radius) * sigma[:, None, None, :] * directions[None]
        raw_outer_blocks.extend(
            (actor_center[:, None] + offset, actor_center[:, None] - offset)
        )
    raw_outer = np.concatenate(raw_outer_blocks, axis=1).astype(np.float32)
    outer = np.clip(raw_outer, low[:, None], high[:, None]).astype(np.float32)
    raw_outer_error = maximum_abs(raw_outer, arrays["raw_outer_centers"])
    outer_error = maximum_abs(outer, arrays["outer_centers"])

    fd_directions = arrays["fd_directions"].astype(np.float32)
    fd_radius = float(arrays["fd_radius_sigma"])
    fd_offset = fd_radius * sigma[:, None, None, :] * fd_directions[None]
    raw_inner = np.concatenate(
        (
            outer[:, :, None],
            outer[:, :, None] + fd_offset[:, None],
            outer[:, :, None] - fd_offset[:, None],
        ),
        axis=2,
    ).astype(np.float32)
    inner = np.clip(
        raw_inner, low[:, None, None], high[:, None, None]
    ).astype(np.float32)
    raw_inner_error = maximum_abs(raw_inner, arrays["raw_inner_centers"])
    inner_error = maximum_abs(inner, arrays["inner_centers"])

    reconstructed_outer_action = np.clip(
        (outer - alpha[:, None])
        / (maximum_residual_sigma * sigma[:, None, None, :]),
        -1.0,
        1.0,
    ).astype(np.float32)
    reconstructed_inner_action = np.clip(
        (inner - alpha[:, None, None])
        / (maximum_residual_sigma * sigma[:, None, None, None, :]),
        -1.0,
        1.0,
    ).astype(np.float32)
    outer_action_error = maximum_abs(
        reconstructed_outer_action, arrays["outer_actions"]
    )
    inner_action_error = maximum_abs(
        reconstructed_inner_action, arrays["inner_actions"]
    )

    flat_action = reconstructed_inner_action.reshape(-1, 33, 8, 2)
    flat_reward = arrays["transformed_reward"].reshape(-1, 33)
    flat_anchor = np.repeat(reconstructed_outer_action, 1, axis=0).reshape(-1, 8, 2)
    gradient, curvature, inner_rank = independent_fit(
        flat_action, flat_reward, flat_anchor
    )
    gradient_error = maximum_abs(
        gradient.reshape(100, 77, 16), arrays["local_gradient"]
    )
    curvature_error = maximum_abs(
        curvature.reshape(100, 77), arrays["local_curvature"]
    )
    rank_exact = bool(
        np.array_equal(inner_rank.reshape(100, 77), arrays["inner_design_rank"])
    )

    total = int(np.prod(arrays["direct_cost"].shape))
    replay_count = min(args.replay_count, total)
    flat_index = np.linspace(0, total - 1, replay_count, dtype=np.int64)
    local_context, location_index, probe_index = np.unravel_index(
        flat_index, arrays["direct_cost"].shape
    )
    replay_centers = inner[local_context, location_index, probe_index]
    replay_context = context_index[local_context]
    device = torch.device(args.device)
    tensors = tensorize(data, device)
    replay_cost = direct_cost(
        replay_centers,
        data,
        tensors,
        replay_context,
        args.batch_size,
        device,
    )
    stored_replay_cost = arrays["direct_cost"][
        local_context, location_index, probe_index
    ]
    replay_cost_error = maximum_abs(replay_cost, stored_replay_cost)
    alpha_cost = direct_cost(
        alpha,
        data,
        tensors,
        context_index,
        args.batch_size,
        device,
    )
    reward_scale = 5.0
    transformed_reward = np.arcsinh(
        (alpha_cost[:, None, None] - arrays["direct_cost"]) / reward_scale
    ).astype(np.float32)
    reward_error = maximum_abs(transformed_reward, arrays["transformed_reward"])

    outer_delta = (
        reconstructed_outer_action[:, 1:] - reconstructed_outer_action[:, :1]
    ).reshape(100, 76, 16)
    outer_ranks = np.asarray(
        [np.linalg.matrix_rank(value) for value in outer_delta], np.int64
    )
    finite = bool(
        all(
            np.all(np.isfinite(value))
            for value in arrays.values()
            if np.issubdtype(value.dtype, np.number)
        )
    )
    contract = summary["contract"]
    contract_exact = bool(
        contract["actor_frozen"]
        and not contract["analytic_dbm_gradient"]
        and not contract["formal_validation_loaded"]
        and not contract["test_loaded"]
    )
    checks = {
        "source_hashes": all(source_hash_checks.values()),
        "context_order": context_order_exact,
        "roles": roles_exact,
        "strata": strata_exact,
        "source_split": source_split_exact,
        "shape": arrays["direct_cost"].shape == (100, 77, 33),
        "finite": finite,
        "bounds": max(low_error, high_error) <= 1e-7,
        "outer_geometry": max(raw_outer_error, outer_error) <= 1e-6,
        "inner_geometry": max(raw_inner_error, inner_error) <= 1e-6,
        "action_coordinate": max(outer_action_error, inner_action_error) <= 1e-6,
        "independent_fit": max(gradient_error, curvature_error) <= 5e-4,
        "inner_rank": rank_exact and int(np.min(inner_rank)) >= 17,
        "outer_rank": int(np.min(outer_ranks)) >= 12,
        "dbm_cost_replay": replay_cost_error <= args.cost_atol,
        "reward_transform": reward_error <= 5e-5,
        "contract": contract_exact,
    }
    qualification = (
        "TARGETED_LOCAL_RESPONSE_LABELS_VALIDATED"
        if all(checks.values())
        else "TARGETED_LOCAL_RESPONSE_LABELS_VALIDATION_FAILED"
    )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "summary": str(summary_path.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "arrays_sha256": sha256_file(arrays_path),
        "checks": checks,
        "source_hash_checks": source_hash_checks,
        "errors": {
            "low": low_error,
            "high": high_error,
            "raw_outer": raw_outer_error,
            "outer": outer_error,
            "raw_inner": raw_inner_error,
            "inner": inner_error,
            "outer_action": outer_action_error,
            "inner_action": inner_action_error,
            "gradient": gradient_error,
            "curvature": curvature_error,
            "dbm_cost_replay": replay_cost_error,
            "reward_transform": reward_error,
        },
        "counts": {
            "context": len(context_index),
            "replayed_dbm_cost": replay_count,
            "outer_rank_minimum": int(np.min(outer_ranks)),
            "inner_rank_minimum": int(np.min(inner_rank)),
        },
    }
    output = args.input_dir / "validation_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
