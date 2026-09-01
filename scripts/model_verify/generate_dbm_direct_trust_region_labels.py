#!/usr/bin/env python3
"""Generate immutable train-only FR-TRPI direct-cost line-search labels.

The old and proposal deterministic Actors are frozen.  Their center difference is
projected into a source-MPPI-sigma trust region, then evaluated at a fixed alpha
grid with one deterministic DBM direct rollout per center.  No DBM analytic
gradient, stochastic MPPI wrapper, formal-validation state, or test state is used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TorchMPPIDeterministicCenterActor,
    ego_reference_features,
)
from generate_dbm_multicenter_teacher import evaluate_knots, make_controller
from generate_dbm_proposal_teacher import repository_state, sha256_file


FORMAT_VERSION = 1
GENERATOR_ID = "anycar-dbm-direct-trust-region-v1"
ROOT = Path("/disk/collect_data_from_anycar/mppi_rl_closed_loop")
OLD_SOURCE = ROOT / "fixed_dbm_policy_diverse_20260805_v1"
OLD_CONTEXT = ROOT / "labels/dbm_two_pass_feedback_diverse_20260805_v1"
OLD_RISK = ROOT / "labels/dbm_two_pass_risk_replay_diverse_20260805_v1"
NEW_SOURCE = ROOT / "fixed_dbm_policy_train_expansion_20260807_v1"
NEW_CONTEXT = ROOT / "labels/mppi_first_pass_actor_context_train_expansion_20260807_v1"
OLD_PLAN = OLD_SOURCE / "scenario_plan.json"
NEW_PLAN = Path("scripts/model_verify/fixed_dbm_policy_train_expansion_20260807_v1.json")
OLD_ACTOR = Path(
    "outputs/mppi_proposal/direct_center_actor_trust_step_20260806_v2/"
    "direct_center_actor_trust_selected.pt"
)
PROPOSAL_ACTOR = Path(
    "outputs/mppi_proposal/j16_cost_sensitive_expansion_20260807_v2/actor_seed1.pt"
)
DEFAULT_OUTPUT = ROOT / "labels/dbm_direct_trust_region_train_20260807_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-source", type=Path, default=OLD_SOURCE)
    parser.add_argument("--old-context", type=Path, default=OLD_CONTEXT)
    parser.add_argument("--old-risk", type=Path, default=OLD_RISK)
    parser.add_argument("--new-source", type=Path, default=NEW_SOURCE)
    parser.add_argument("--new-context", type=Path, default=NEW_CONTEXT)
    parser.add_argument("--old-plan", type=Path, default=OLD_PLAN)
    parser.add_argument("--new-plan", type=Path, default=NEW_PLAN)
    parser.add_argument("--old-actor", type=Path, default=OLD_ACTOR)
    parser.add_argument("--proposal-actor", type=Path, default=PROPOSAL_ACTOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trust-radius-sigma-rms", type=float, default=0.50)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--minimum-improvement", type=float, default=0.05)
    parser.add_argument("--tie-absolute", type=float, default=0.05)
    parser.add_argument("--tie-fraction", type=float, default=0.05)
    parser.add_argument("--selection-episodes-per-stratum", type=int, default=2)
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def alpha_grid(step: float) -> np.ndarray:
    if not 0.0 < step <= 1.0:
        raise ValueError("alpha step must be in (0,1]")
    count = int(round(1.0 / step))
    if not np.isclose(count * step, 1.0, atol=1e-9):
        raise ValueError("alpha step must divide one exactly")
    return np.linspace(0.0, 1.0, count + 1, dtype=np.float32)


def load_actor(path: Path, device: torch.device) -> tuple[dict[str, Any], TorchMPPIDeterministicCenterActor]:
    payload = torch.load(path, map_location="cpu")
    actor = TorchMPPIDeterministicCenterActor(
        maximum_delta_sigma=float(payload["maximum_delta_sigma"]), dropout=0.0
    ).to(device)
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    actor.eval()
    return payload, actor


def actor_inputs(
    source: np.lib.npyio.NpzFile,
    context: np.lib.npyio.NpzFile,
    gradient_mean: np.ndarray,
    gradient_std: np.ndarray,
    repeat: int,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    state = np.asarray(source["initial_state"], np.float32)
    action = np.asarray(source["current_action"], np.float32)
    history = np.asarray(source["history"][0], np.float32)[None]
    reference = ego_reference_features(source["reference_ego"], float(state[3]))[None]
    current = np.asarray((state[3], state[4], *action), np.float32)[None]
    normalization = MPPIProposalNormalization.from_dict(checkpoint["state_normalization"])
    history, reference, current = normalization.normalize_numpy(
        history, reference, current
    )
    anchor = np.asarray(context["guided_center_knots"][repeat], np.float32)[None]
    feedback = np.asarray(context["first_pass_feedback"][repeat], np.float32)
    feedback = (feedback - checkpoint["feedback_mean"]) / checkpoint["feedback_std"]
    gradient = np.concatenate((gradient_mean[repeat], gradient_std[repeat]))
    gradient = (gradient - checkpoint["gradient_mean"]) / checkpoint["gradient_std"]
    return tuple(
        torch.from_numpy(np.asarray(value, np.float32)).to(device)
        for value in (history, reference, current, anchor, feedback[None], gradient[None])
    )


def plan_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = [row for row in data["episodes"] if row["split"] == "train"]
    if not rows:
        raise ValueError(f"no train episodes in {path}")
    return rows


def combined_split(
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    selection_count: int,
) -> tuple[list[str], list[str], list[str]]:
    groups: dict[tuple[float, str], list[str]] = defaultdict(list)
    for row in (*old_rows, *new_rows):
        key = (float(row["reference_speed_mps"]), str(row["scenario_class"]))
        groups[key].append(str(row["episode_id"]))
    fit, selection = [], []
    for key, episodes in sorted(groups.items()):
        episodes.sort()
        if len(episodes) != 12:
            raise AssertionError(f"combined stratum {key} has {len(episodes)} episodes")
        if not 0 < selection_count < len(episodes):
            raise ValueError("invalid selection episodes per stratum")
        fit.extend(episodes[:-selection_count])
        selection.extend(episodes[-selection_count:])
    return sorted(fit + selection), sorted(fit), sorted(selection)


def source_records(
    source_kind: str,
    source_root: Path,
    context_root: Path,
    risk_root: Path | None,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metadata = {str(row["episode_id"]): row for row in rows}
    records = []
    for episode in sorted(metadata):
        for path in sorted((source_root / episode / "snapshots").glob("*.npz")):
            context_path = context_root / episode / path.name
            risk_path = None if risk_root is None else risk_root / episode / path.name
            if not context_path.is_file():
                raise FileNotFoundError(context_path)
            if risk_path is not None and not risk_path.is_file():
                raise FileNotFoundError(risk_path)
            records.append({
                "source_kind": source_kind,
                "source_path": path,
                "context_path": context_path,
                "risk_path": risk_path,
                "metadata": metadata[episode],
            })
    return records


def line_centers(
    old_center: np.ndarray,
    proposal_center: np.ndarray,
    sigma: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    trust_radius: float,
    alphas: np.ndarray,
) -> dict[str, np.ndarray | float]:
    requested = (proposal_center - old_center) / sigma.reshape(1, 2)
    requested_rho = float(np.sqrt(np.mean(requested ** 2)))
    trust_scale = min(1.0, trust_radius / (requested_rho + 1e-8))
    projected = requested * trust_scale
    raw = old_center[None] + alphas[:, None, None] * sigma.reshape(1, 1, 2) * projected[None]
    centers = np.clip(raw, low, high).astype(np.float32)
    realized = (centers - old_center[None]) / sigma.reshape(1, 1, 2)
    realized_rho = np.sqrt(np.mean(realized ** 2, axis=(1, 2))).astype(np.float32)
    return {
        "requested_direction": requested.astype(np.float32),
        "requested_rho": requested_rho,
        "trust_scale": float(trust_scale),
        "projected_direction": projected.astype(np.float32),
        "raw_centers": raw.astype(np.float32),
        "centers": centers,
        "realized_rho": realized_rho,
        "clip_fraction": np.mean(raw != centers, axis=(1, 2)).astype(np.float32),
    }


def select_safe_index(
    cost: np.ndarray,
    minimum_improvement: float,
    tie_absolute: float,
    tie_fraction: float,
) -> tuple[int, int, float]:
    argmin = int(np.argmin(cost))
    best_gain = float(cost[0] - cost[argmin])
    if best_gain < minimum_improvement:
        return argmin, 0, max(tie_absolute, tie_fraction * max(best_gain, 0.0))
    tolerance = max(tie_absolute, tie_fraction * best_gain)
    eligible = np.flatnonzero(
        (cost <= cost[argmin] + tolerance)
        & (cost < cost[0] - minimum_improvement)
    )
    safe = int(eligible[0]) if len(eligible) else 0
    return argmin, safe, float(tolerance)


def distribution(value: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "p05": float(np.quantile(value, 0.05)),
        "p95": float(np.quantile(value, 0.95)),
        "minimum": float(np.min(value)),
        "maximum": float(np.max(value)),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, Any]:
    def one(values: list[dict[str, float]]) -> dict[str, Any]:
        old = np.asarray([row["old_cost"] for row in values])
        endpoint = np.asarray([row["endpoint_cost"] for row in values])
        argmin = np.asarray([row["argmin_cost"] for row in values])
        safe = np.asarray([row["safe_cost"] for row in values])
        return {
            "context_count": len(values),
            "old_cost": distribution(old),
            "trusted_endpoint_cost": distribution(endpoint),
            "argmin_cost": distribution(argmin),
            "safe_cost": distribution(safe),
            "argmin_gain": distribution(old - argmin),
            "safe_gain": distribution(old - safe),
            "argmin_alpha_zero_fraction": float(np.mean([
                row["argmin_index"] == 0 for row in values
            ])),
            "safe_alpha_zero_fraction": float(np.mean([
                row["safe_index"] == 0 for row in values
            ])),
            "requested_rho": distribution(np.asarray([
                row["requested_rho"] for row in values
            ])),
            "trust_scale": distribution(np.asarray([
                row["trust_scale"] for row in values
            ])),
        }

    grouped: dict[float, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["reference_speed_mps"])].append(row)
    return {
        "overall": one(rows),
        "by_reference_speed_mps": {
            str(speed): one(values) for speed, values in sorted(grouped.items())
        },
    }


def main() -> None:
    args = parse_args()
    if args.trust_radius_sigma_rms <= 0:
        raise ValueError("trust radius must be positive")
    if args.minimum_improvement < 0 or args.tie_absolute < 0 or args.tie_fraction < 0:
        raise ValueError("improvement and tie thresholds must be nonnegative")
    if args.output.exists():
        raise FileExistsError(args.output)
    alphas = alpha_grid(args.alpha_step)
    old_rows = plan_rows(args.old_plan)
    new_rows = plan_rows(args.new_plan)
    train_episodes, fit_episodes, selection_episodes = combined_split(
        old_rows, new_rows, args.selection_episodes_per_stratum
    )
    records = source_records(
        "diverse", args.old_source, args.old_context, args.old_risk, old_rows
    ) + source_records(
        "expansion", args.new_source, args.new_context, None, new_rows
    )
    records.sort(key=lambda row: (row["source_path"].parents[1].name, row["source_path"].name))
    if args.max_snapshots:
        records = records[: args.max_snapshots]
    if not records:
        raise ValueError("no train snapshots selected")

    old_heldout = json.loads((args.old_context / "splits.json").read_text())
    generated_episodes = {row["source_path"].parents[1].name for row in records}
    forbidden = set(old_heldout.get("validation", ())) | set(old_heldout.get("test", ()))
    if generated_episodes & forbidden:
        raise AssertionError("formal validation/test episode selected")

    device = torch.device(args.device)
    old_payload, old_actor = load_actor(args.old_actor, device)
    proposal_payload, proposal_actor = load_actor(args.proposal_actor, device)
    checkpoint_hashes = {
        "old_actor": sha256_file(args.old_actor),
        "proposal_actor": sha256_file(args.proposal_actor),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.tmp-", dir=args.output.parent))
    started = time.perf_counter()
    context_rows: list[dict[str, float]] = []
    source_counts: dict[str, int] = defaultdict(int)
    try:
        for index, record in enumerate(records, 1):
            source_path: Path = record["source_path"]
            context_path: Path = record["context_path"]
            risk_path: Path | None = record["risk_path"]
            episode = source_path.parents[1].name
            with np.load(source_path, allow_pickle=False) as source, np.load(
                context_path, allow_pickle=False
            ) as context:
                if str(context["source_snapshot_sha256"]) != sha256_file(source_path):
                    raise AssertionError(f"context/source hash mismatch: {context_path}")
                if risk_path is None:
                    gradient_mean = np.asarray(context["critic_gradient_mean"], np.float32)
                    gradient_std = np.asarray(context["critic_gradient_std"], np.float32)
                else:
                    with np.load(risk_path, allow_pickle=False) as risk:
                        if str(risk["source_snapshot_sha256"]) != sha256_file(source_path):
                            raise AssertionError(f"risk/source hash mismatch: {risk_path}")
                        gradient_mean = np.asarray(risk["critic_gradient_mean"], np.float32)
                        gradient_std = np.asarray(risk["critic_gradient_std"], np.float32)
                config = {"objective": {"cost_weights": json.loads(str(source["cost_weights_json"]))}}
                controller, backend = make_controller(source, config, device)
                params = json.loads(str(source["mppi_params_json"]))
                sigma = np.asarray(params["noise_sigma"], np.float32)
                low = np.asarray(params["action_min"], np.float32)
                high = np.asarray(params["action_max"], np.float32)
                history = torch.from_numpy(source["history"]).to(device)
                initial = torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5)
                current_action = torch.from_numpy(source["current_action"]).to(device).reshape(1, 2)
                reference = controller._prepare_reference(source["reference"])
                old_centers, proposal_centers = [], []
                old_actions, proposal_actions = [], []
                requested_direction, projected_direction = [], []
                requested_rho, trust_scale = [], []
                raw_centers, centers = [], []
                realized_rho, clip_fraction = [], []
                direct_cost, advantage = [], []
                argmin_index, safe_index, safe_tolerance = [], [], []
                safe_centers = []
                repeat_count = len(context["guided_center_knots"])
                for repeat in range(repeat_count):
                    old_input = actor_inputs(
                        source, context, gradient_mean, gradient_std,
                        repeat, old_payload, device,
                    )
                    proposal_input = actor_inputs(
                        source, context, gradient_mean, gradient_std,
                        repeat, proposal_payload, device,
                    )
                    with torch.no_grad():
                        old_action_t, old_center_t = old_actor(*old_input)
                        proposal_action_t, proposal_center_t = proposal_actor(*proposal_input)
                    old_center = old_center_t[0].cpu().numpy().astype(np.float32)
                    proposal_center = proposal_center_t[0].cpu().numpy().astype(np.float32)
                    line = line_centers(
                        old_center, proposal_center, sigma, low, high,
                        args.trust_radius_sigma_rms, alphas,
                    )
                    cost, _, _ = evaluate_knots(
                        controller, backend, line["centers"], history, initial,
                        current_action, reference,
                    )
                    argmin, safe, tolerance = select_safe_index(
                        cost, args.minimum_improvement,
                        args.tie_absolute, args.tie_fraction,
                    )
                    gain = cost[0] - cost
                    old_centers.append(old_center)
                    proposal_centers.append(proposal_center)
                    old_actions.append(old_action_t[0].cpu().numpy().astype(np.float32))
                    proposal_actions.append(proposal_action_t[0].cpu().numpy().astype(np.float32))
                    requested_direction.append(line["requested_direction"])
                    projected_direction.append(line["projected_direction"])
                    requested_rho.append(line["requested_rho"])
                    trust_scale.append(line["trust_scale"])
                    raw_centers.append(line["raw_centers"])
                    centers.append(line["centers"])
                    realized_rho.append(line["realized_rho"])
                    clip_fraction.append(line["clip_fraction"])
                    direct_cost.append(cost.astype(np.float32))
                    advantage.append(gain.astype(np.float32))
                    argmin_index.append(argmin)
                    safe_index.append(safe)
                    safe_tolerance.append(tolerance)
                    safe_centers.append(line["centers"][safe])
                    context_rows.append({
                        "reference_speed_mps": float(record["metadata"]["reference_speed_mps"]),
                        "old_cost": float(cost[0]),
                        "endpoint_cost": float(cost[-1]),
                        "argmin_cost": float(cost[argmin]),
                        "safe_cost": float(cost[safe]),
                        "argmin_index": float(argmin),
                        "safe_index": float(safe),
                        "requested_rho": float(line["requested_rho"]),
                        "trust_scale": float(line["trust_scale"]),
                    })

                initial_six = np.asarray(source["initial_state_six"], np.float32)
                source_reference = np.asarray(source["reference"], np.float32)
                heading_error = float(np.arctan2(
                    np.sin(initial_six[2] - source_reference[0, 2]),
                    np.cos(initial_six[2] - source_reference[0, 2]),
                ))
                reference_speed = float(record["metadata"]["reference_speed_mps"])
                output_dir = staging / episode
                output_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output_dir / source_path.name,
                    format_version=np.asarray(FORMAT_VERSION, np.int32),
                    source_kind=np.asarray(record["source_kind"]),
                    source_snapshot=np.asarray(str(source_path.resolve())),
                    source_snapshot_sha256=np.asarray(sha256_file(source_path)),
                    context_label=np.asarray(str(context_path.resolve())),
                    context_label_sha256=np.asarray(sha256_file(context_path)),
                    risk_label=np.asarray("" if risk_path is None else str(risk_path.resolve())),
                    risk_label_sha256=np.asarray("" if risk_path is None else sha256_file(risk_path)),
                    old_actor_checkpoint_sha256=np.asarray(checkpoint_hashes["old_actor"]),
                    proposal_actor_checkpoint_sha256=np.asarray(checkpoint_hashes["proposal_actor"]),
                    alpha_grid=alphas,
                    trust_radius_sigma_rms=np.asarray(args.trust_radius_sigma_rms, np.float32),
                    minimum_improvement=np.asarray(args.minimum_improvement, np.float32),
                    tie_absolute=np.asarray(args.tie_absolute, np.float32),
                    tie_fraction=np.asarray(args.tie_fraction, np.float32),
                    source_sigma=sigma,
                    action_min=low,
                    action_max=high,
                    old_actor_action=np.asarray(old_actions, np.float32),
                    old_actor_center=np.asarray(old_centers, np.float32),
                    proposal_actor_action=np.asarray(proposal_actions, np.float32),
                    proposal_actor_center=np.asarray(proposal_centers, np.float32),
                    requested_normalized_direction=np.asarray(requested_direction, np.float32),
                    requested_rho=np.asarray(requested_rho, np.float32),
                    trust_scale=np.asarray(trust_scale, np.float32),
                    projected_normalized_direction=np.asarray(projected_direction, np.float32),
                    raw_line_centers=np.asarray(raw_centers, np.float32),
                    line_centers=np.asarray(centers, np.float32),
                    realized_rho=np.asarray(realized_rho, np.float32),
                    line_clip_fraction=np.asarray(clip_fraction, np.float32),
                    direct_cost=np.asarray(direct_cost, np.float32),
                    advantage_vs_old=np.asarray(advantage, np.float32),
                    argmin_index=np.asarray(argmin_index, np.int32),
                    safe_index=np.asarray(safe_index, np.int32),
                    safe_tolerance=np.asarray(safe_tolerance, np.float32),
                    safe_alpha=alphas[np.asarray(safe_index, np.int32)],
                    safe_center=np.asarray(safe_centers, np.float32),
                    reference_speed_mps=np.asarray(reference_speed, np.float32),
                    actual_vx_mps=np.asarray(initial_six[3], np.float32),
                    overspeed_mps=np.asarray(initial_six[3] - reference_speed, np.float32),
                    heading_error_rad=np.asarray(heading_error, np.float32),
                    yaw_rate_rad_s=np.asarray(initial_six[5], np.float32),
                    scenario_class=np.asarray(str(record["metadata"]["scenario_class"])),
                )
                source_counts[record["source_kind"]] += 1
            if index == 1 or index % 50 == 0 or index == len(records):
                print(
                    f"[{index:04d}/{len(records):04d}] contexts={len(context_rows)} "
                    f"elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )

        old_splits = json.loads((args.old_context / "splits.json").read_text())
        new_splits = json.loads((args.new_context / "splits.json").read_text())
        split_payload = {
            "format_version": 1,
            "train": train_episodes,
            "internal_fit": fit_episodes,
            "internal_selection": selection_episodes,
            "formal_validation_sealed_not_generated": old_splits.get("validation", []),
            "test_sealed_not_generated": old_splits.get("test", []),
            "new_source_validation": new_splits.get("validation", []),
            "new_source_test": new_splits.get("test", []),
        }
        (staging / "splits.json").write_text(json.dumps(split_payload, indent=2) + "\n")
        config_payload = {
            "format_version": FORMAT_VERSION,
            "generator": GENERATOR_ID,
            "old_source": str(args.old_source.resolve()),
            "old_context": str(args.old_context.resolve()),
            "old_risk": str(args.old_risk.resolve()),
            "new_source": str(args.new_source.resolve()),
            "new_context": str(args.new_context.resolve()),
            "old_plan": str(args.old_plan.resolve()),
            "new_plan": str(args.new_plan.resolve()),
            "old_plan_sha256": sha256_file(args.old_plan),
            "new_plan_sha256": sha256_file(args.new_plan),
            "old_actor": str(args.old_actor.resolve()),
            "proposal_actor": str(args.proposal_actor.resolve()),
            "old_actor_sha256": checkpoint_hashes["old_actor"],
            "proposal_actor_sha256": checkpoint_hashes["proposal_actor"],
            "trust_radius_sigma_rms": args.trust_radius_sigma_rms,
            "alpha_grid": alphas.tolist(),
            "minimum_improvement": args.minimum_improvement,
            "tie_absolute": args.tie_absolute,
            "tie_fraction": args.tie_fraction,
            "selection_episodes_per_stratum": args.selection_episodes_per_stratum,
            "max_snapshots": args.max_snapshots,
            "objective": "deterministic direct DBM cost; no reward seed",
            "gradient_policy": "no DBM analytic gradients consumed or stored",
        }
        (staging / "config.json").write_text(json.dumps(config_payload, indent=2) + "\n")
        metrics = summarize(context_rows)
        summary = {
            "format_version": FORMAT_VERSION,
            "generator": GENERATOR_ID,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository": repository_state(Path(__file__).resolve().parents[2]),
            "snapshot_count": len(records),
            "context_count": len(context_rows),
            "line_center_count": len(alphas),
            "direct_rollout_count": len(context_rows) * len(alphas),
            "episode_count_in_plan": len(train_episodes),
            "generated_episode_count": len(generated_episodes),
            "source_snapshot_count": dict(source_counts),
            "metrics": metrics,
            "qualification": "TRAIN_ONLY_TR1_LABELS",
            "test_policy": "formal validation and test episodes not generated",
            "elapsed_seconds": time.perf_counter() - started,
        }
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(staging, args.output)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
