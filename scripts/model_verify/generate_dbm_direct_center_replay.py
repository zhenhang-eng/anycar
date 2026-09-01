#!/usr/bin/env python3
"""Generate immutable episode-level direct-cost replay around a deterministic Actor."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import TorchMPPIDeterministicCenterActor
from evaluate_mppi_continuous_center_bootstrap import actor_inputs
from generate_dbm_fullrank_local_labels import hadamard_directions
from generate_dbm_multicenter_teacher import evaluate_knots, make_controller
from generate_dbm_proposal_teacher import repository_state, sha256_file


FORMAT_VERSION = 1
GENERATOR_ID = "anycar-dbm-direct-center-replay-v1"
DEFAULT_SOURCE = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/"
    "fixed_dbm_policy_diverse_20260805_v1"
)
DEFAULT_PARENT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_feedback_diverse_20260805_v1"
)
DEFAULT_RISK = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_risk_replay_diverse_20260805_v1"
)
DEFAULT_BANK = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_two_pass_multidirection_replay_diverse_20260805_v1"
)
DEFAULT_T1 = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_teacher_t1_diverse_20260805_v1"
)
DEFAULT_CHECKPOINT = Path(
    "outputs/mppi_proposal/continuous_center_sac_bootstrap_20260806_v2/"
    "continuous_center_sac_bootstrap.pt"
)
DEFAULT_OUTPUT = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_direct_center_replay_diverse_20260806_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--risk-labels", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--bank-labels", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--t1-labels", type=Path, default=DEFAULT_T1)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--local-radii-sigma", type=float, nargs="+", default=(0.03, 0.06, 0.10, 0.15)
    )
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def center_names(bank_count: int, radii: tuple[float, ...]) -> tuple[str, ...]:
    names = ["anchor"]
    names.extend(f"stored_bank_{index:02d}" for index in range(bank_count))
    names.extend(("t1_teacher", "bootstrap_actor"))
    for radius in radii:
        for direction in range(16):
            names.extend((
                f"actor_h{direction:02d}_r{radius:.3f}_pos",
                f"actor_h{direction:02d}_r{radius:.3f}_neg",
            ))
    return tuple(names)


def local_centers(
    actor_center: np.ndarray,
    sigma: np.ndarray,
    directions: np.ndarray,
    radii: tuple[float, ...],
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for radius in radii:
        delta = float(radius) * directions * sigma.reshape(1, 1, 2)
        for one in delta:
            values.extend((actor_center + one, actor_center - one))
    raw = np.asarray(values, np.float32)
    return raw, np.clip(raw, action_min, action_max).astype(np.float32)


def local_rank(
    centers: np.ndarray, actor_center: np.ndarray, sigma: np.ndarray
) -> int:
    design = ((centers - actor_center[None]) / sigma.reshape(1, 1, 2)).reshape(
        len(centers), 16
    )
    return int(np.linalg.matrix_rank(design))


def main() -> None:
    args = parse_args()
    radii = tuple(float(value) for value in args.local_radii_sigma)
    if not radii or any(value <= 0 for value in radii) or tuple(sorted(set(radii))) != radii:
        raise ValueError("local radii must be distinct increasing positive values")
    if args.output.exists():
        raise FileExistsError(args.output)
    splits = json.loads((args.parent_labels / "splits.json").read_text())
    allowed_episodes = set(splits["train"]) | set(splits["validation"])
    paths = [
        path for path in sorted(args.bank_labels.glob("episode_*/*.npz"))
        if path.parent.name in allowed_episodes
    ]
    if args.max_snapshots:
        paths = paths[: args.max_snapshots]
    if not paths:
        raise ValueError("no train/validation snapshots found")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    actor = TorchMPPIDeterministicCenterActor(
        float(checkpoint["maximum_delta_sigma"]), dropout=0.0
    ).to(device)
    if checkpoint.get("actor_class") == "TorchMPPIDeterministicCenterActor":
        actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    else:
        actor.load_stochastic_actor_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    directions = hadamard_directions().astype(np.float32)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    started = time.time()
    context_count = 0
    center_count = None
    rank_min = 16
    clip_sum = 0.0
    cost_sum = {name: 0.0 for name in ("anchor", "actor", "teacher", "bank_best", "local_best")}
    try:
        for index, bank_path in enumerate(paths, start=1):
            episode = bank_path.parent.name
            source_path = args.source / episode / "snapshots" / bank_path.name
            parent_path = args.parent_labels / episode / bank_path.name
            risk_path = args.risk_labels / episode / bank_path.name
            t1_path = args.t1_labels / episode / bank_path.name
            with np.load(source_path, allow_pickle=False) as source, np.load(
                parent_path, allow_pickle=False
            ) as parent, np.load(risk_path, allow_pickle=False) as risk, np.load(
                bank_path, allow_pickle=False
            ) as bank, np.load(t1_path, allow_pickle=False) as t1:
                config = {
                    "objective": {
                        "cost_weights": json.loads(str(source["cost_weights_json"]))
                    }
                }
                controller, backend = make_controller(source, config, device)
                params = json.loads(str(source["mppi_params_json"]))
                sigma = np.asarray(params["noise_sigma"], np.float32)
                action_min = np.asarray(params["action_min"], np.float32)
                action_max = np.asarray(params["action_max"], np.float32)
                anchors = np.asarray(parent["guided_center_knots"], np.float32)
                stored_bank = np.asarray(bank["centers"], np.float32)
                teacher = np.asarray(t1["teacher_center_knots"], np.float32)
                history = torch.from_numpy(source["history"]).to(device)
                initial = torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5)
                current_action = torch.from_numpy(source["current_action"]).to(device).reshape(1, 2)
                reference = controller._prepare_reference(source["reference"])
                per_context_centers = []
                per_context_raw_local = []
                per_context_actor = []
                per_context_action = []
                per_context_rank = []
                per_context_clip = []
                per_context_cost = []
                for context in range(len(anchors)):
                    model_input = actor_inputs(
                        source, parent, risk, context, checkpoint, device
                    )
                    with torch.no_grad():
                        actor_action, actor_center_tensor = actor(*model_input)
                    actor_center = actor_center_tensor[0].cpu().numpy().astype(np.float32)
                    actor_action_np = actor_action[0].cpu().numpy().astype(np.float32)
                    raw_local, clipped_local = local_centers(
                        actor_center, sigma, directions, radii, action_min, action_max
                    )
                    centers = np.concatenate((
                        anchors[context][None], stored_bank[context], teacher[None],
                        actor_center[None], clipped_local,
                    )).astype(np.float32)
                    direct_cost, _, _ = evaluate_knots(
                        controller, backend, centers, history, initial,
                        current_action, reference,
                    )
                    bank_slice = slice(1, 1 + len(stored_bank[context]))
                    local_slice = slice(3 + len(stored_bank[context]), None)
                    cost_sum["anchor"] += float(direct_cost[0])
                    cost_sum["teacher"] += float(direct_cost[1 + len(stored_bank[context])])
                    cost_sum["actor"] += float(direct_cost[2 + len(stored_bank[context])])
                    cost_sum["bank_best"] += float(np.min(direct_cost[bank_slice]))
                    cost_sum["local_best"] += float(np.min(direct_cost[local_slice]))
                    one_rank = local_rank(clipped_local, actor_center, sigma)
                    one_clip = float(np.mean(raw_local != clipped_local))
                    rank_min = min(rank_min, one_rank)
                    clip_sum += one_clip
                    context_count += 1
                    per_context_centers.append(centers)
                    per_context_raw_local.append(raw_local)
                    per_context_actor.append(actor_center)
                    per_context_action.append(actor_action_np)
                    per_context_rank.append(one_rank)
                    per_context_clip.append(one_clip)
                    per_context_cost.append(direct_cost.astype(np.float32))
                    center_count = len(centers)

                output_dir = staging / episode
                output_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output_dir / bank_path.name,
                    format_version=np.asarray(FORMAT_VERSION),
                    source_snapshot=np.asarray(str(source_path)),
                    source_snapshot_sha256=np.asarray(sha256_file(source_path)),
                    parent_label_sha256=np.asarray(sha256_file(parent_path)),
                    risk_label_sha256=np.asarray(sha256_file(risk_path)),
                    bank_label_sha256=np.asarray(sha256_file(bank_path)),
                    t1_label_sha256=np.asarray(sha256_file(t1_path)),
                    checkpoint_sha256=np.asarray(sha256_file(args.checkpoint)),
                    center_names=np.asarray(center_names(len(stored_bank[0]), radii)),
                    local_radii_sigma=np.asarray(radii, np.float32),
                    normalized_directions=directions,
                    sigma=sigma,
                    anchor_center=anchors,
                    bootstrap_actor_center=np.asarray(per_context_actor, np.float32),
                    bootstrap_actor_action=np.asarray(per_context_action, np.float32),
                    raw_local_centers=np.asarray(per_context_raw_local, np.float32),
                    centers=np.asarray(per_context_centers, np.float32),
                    direct_cost=np.asarray(per_context_cost, np.float32),
                    advantage_vs_anchor=(
                        np.asarray(per_context_cost, np.float32)[:, :1]
                        - np.asarray(per_context_cost, np.float32)
                    ),
                    local_direction_rank=np.asarray(per_context_rank, np.int32),
                    local_clip_fraction=np.asarray(per_context_clip, np.float32),
                )
            if index == 1 or index % 25 == 0 or index == len(paths):
                print(
                    f"[{index:04d}/{len(paths):04d}] contexts={context_count} "
                    f"rank_min={rank_min} elapsed={time.time()-started:.1f}s",
                    flush=True,
                )

        generated_splits = {
            "format_version": 1,
            "train": splits["train"],
            "validation": splits["validation"],
            "test_sealed_not_generated": splits["test"],
        }
        (staging / "splits.json").write_text(json.dumps(generated_splits, indent=2) + "\n")
        summary = {
            "format_version": FORMAT_VERSION,
            "generator": GENERATOR_ID,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": str(args.source.resolve()),
            "parent_labels": str(args.parent_labels.resolve()),
            "risk_labels": str(args.risk_labels.resolve()),
            "bank_labels": str(args.bank_labels.resolve()),
            "t1_labels": str(args.t1_labels.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "repository": repository_state(Path(__file__).resolve().parents[2]),
            "snapshot_count": len(paths),
            "context_count": context_count,
            "center_count_per_context": center_count,
            "local_radii_sigma": list(radii),
            "local_direction_count": 16,
            "local_direction_rank_min": rank_min,
            "mean_local_clip_fraction": clip_sum / context_count,
            "mean_direct_cost": {
                name: value / context_count for name, value in cost_sum.items()
            },
            "test_policy": "episode_105..119 sealed; no label files generated",
            "elapsed_seconds": time.time() - started,
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(staging, args.output)
        print(json.dumps(summary, indent=2))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
