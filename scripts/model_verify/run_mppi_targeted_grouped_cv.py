#!/usr/bin/env python3
"""State-balanced grouped cross-validation for the targeted local-Q Critic.

Pre-registered protocol (review doc §11.26):
- 5 folds over the 54 unique physical states behind the 79 target repeat
  contexts, grouped by (episode, physical_snapshot_ordinal); all repeats and
  all 77 outer locations of a state stay in the same fold.
- Fold assignment is greedy-balanced on stratum (H_ONLY / JOINT_G0_H), then
  scenario, then per-state repeat count, with a fixed seed.
- Per epoch the targeted loss samples state -> repeat center -> a few
  locations and uses the state-level loss
  L_target = (1/|S|) sum_s (1/|A_s|) sum_a L(s,a).
- Checkpoint selection uses only the original internal-validation split;
  heldout folds never select epochs.
- 3 seeds per fold; 21 matched easy controls are never trained and act as a
  fixed heldout compared against the old (non-targeted) PA checkpoints.
- Out-of-fold metrics are reported overall and per stratum.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from car_foundation.mppi_proposal_policy import (
    TorchMPPIDeterministicCenterActor,
    TorchMPPISemanticStructuredLocalQCritic,
)
from generate_dbm_proposal_teacher import sha256_file
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_local_gradient_critic import cosine_rows
from train_mppi_direct_residual_online_ac import actor_inputs, make_base_policy
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    load_actor_payload,
    load_dataset,
    tensorize,
)
from train_mppi_structured_local_q_critic import (
    DEFAULT_FRESH,
    DEFAULT_INITIAL,
    DEFAULT_LABELS,
    batch_parameters,
    chord_distance_sigma,
    chord_weights,
    cross_actions,
    fresh_metrics,
    initialize_semantic_encoder_from_actor,
    load_npz,
    parameters_at_absolute_actions,
    partner_positions,
    prepare_targeted_training,
    set_seed,
    structured_metrics,
    subset_first_axis,
    targeted_metrics,
)

DEFAULT_TARGETED_LABELS = Path(
    "outputs/mppi_proposal/targeted_local_response_labels_20260814_v1/"
    "targeted_local_response_labels.npz"
)
DEFAULT_MANIFEST = Path(
    "outputs/mppi_proposal/targeted_local_probe_manifest_20260814_v1/manifest.json"
)
DEFAULT_BASELINE_SUMMARY = Path(
    "outputs/mppi_proposal/semantic_structured_local_q_20260814_v1/summary.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/targeted_grouped_cv_20260814_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-actor", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--labels-npz", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--fresh-npz", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--targeted-labels-npz", type=Path, default=DEFAULT_TARGETED_LABELS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--selection-min-epoch", type=int, default=80)
    parser.add_argument("--scheduler-min-epoch", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--hessian-scale", type=float, default=256.0)
    parser.add_argument("--value-weight", type=float, default=0.20)
    parser.add_argument("--gradient-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.50)
    parser.add_argument("--probe-value-weight", type=float, default=0.25)
    parser.add_argument("--chord-weight", type=float, default=5.0)
    parser.add_argument("--norm-weight", type=float, default=0.20)
    parser.add_argument("--targeted-value-weight", type=float, default=0.20)
    parser.add_argument("--targeted-gradient-weight", type=float, default=1.0)
    parser.add_argument("--targeted-cosine-weight", type=float, default=0.50)
    parser.add_argument("--targeted-chord-weight", type=float, default=5.0)
    parser.add_argument("--targeted-norm-weight", type=float, default=0.20)
    parser.add_argument("--chord-d0", type=float, default=0.15)
    parser.add_argument("--chord-w-max", type=float, default=4.0)
    parser.add_argument("--chord-epsilon", type=float, default=0.02)
    parser.add_argument("--small-chord-sigma", type=float, default=0.15)
    parser.add_argument("--medium-chord-sigma", type=float, default=0.30)
    parser.add_argument("--cv-locations-per-center", type=int, default=8)
    parser.add_argument("--cv-states-per-batch", type=int, default=8)
    parser.add_argument("--fold-seed", type=int, default=260814)
    parser.add_argument("--overfit-pairs", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def build_state_table(
    manifest: dict[str, Any], targeted: dict[str, np.ndarray]
) -> tuple[dict[str, dict[str, Any]], dict[int, str]]:
    """Map state key -> metadata, and targeted-npz target row -> state key."""
    rows = [
        row for row in manifest["rows"]
        if row["pilot_role"] in ("target", "matched_easy_control")
    ]
    by_context = {int(row["context_index"]): row for row in rows}
    states: dict[str, dict[str, Any]] = {}
    row_state: dict[int, str] = {}
    for position, context in enumerate(targeted["context_index"].astype(np.int64)):
        role = targeted["pilot_role"][position]
        if role != "target":
            continue
        row = by_context.get(int(context))
        if row is None or row["pilot_role"] != "target":
            raise AssertionError(f"target context {context} missing from manifest")
        key = f"{row['episode']}#{int(row['physical_snapshot_ordinal'])}"
        entry = states.setdefault(key, {
            "state_key": key,
            "episode": row["episode"],
            "physical_snapshot_ordinal": int(row["physical_snapshot_ordinal"]),
            "context_strata": [],
            "scenario": row["scenario"],
            "clipped": bool(row["clipped"]),
            "reference_speed_mps": float(row["reference_speed_mps"]),
            "context_rows": [],
        })
        entry["context_strata"].append(row["stratum_id"])
        entry["context_rows"].append(position)
        row_state[position] = key
    for key, entry in states.items():
        entry["repeat_count"] = len(entry["context_rows"])
        # Stratum is a per-context property; lift to state level as worst case
        # for fold balancing. Per-context strata remain the reporting unit.
        entry["stratum_id"] = (
            "JOINT_G0_H" if "JOINT_G0_H" in entry["context_strata"] else "H_ONLY"
        )
    if len(states) != 54 or len(row_state) != 79:
        raise AssertionError(
            f"state/context count mismatch: {len(states)} states, {len(row_state)} rows"
        )
    return states, row_state


def assign_folds(
    states: dict[str, dict[str, Any]], fold_count: int, seed: int
) -> dict[str, int]:
    """Greedy balanced assignment on stratum, scenario, repeat count."""
    rng = np.random.default_rng(seed)
    keys = sorted(states)
    keys = [keys[index] for index in rng.permutation(len(keys))]
    assignment: dict[str, int] = {}
    strata = sorted({states[key]["stratum_id"] for key in keys})
    # Largest stratum first for balance; within stratum, shuffled order.
    ordered: list[str] = []
    by_stratum = {
        stratum: [key for key in keys if states[key]["stratum_id"] == stratum]
        for stratum in strata
    }
    remaining = sum(len(value) for value in by_stratum.values())
    while remaining:
        for stratum in strata:
            if by_stratum[stratum]:
                ordered.append(by_stratum[stratum].pop())
                remaining -= 1
    for key in ordered:
        meta = states[key]
        scores = []
        for fold in range(fold_count):
            members = [name for name, owner in assignment.items() if owner == fold]
            same_stratum = sum(
                states[name]["stratum_id"] == meta["stratum_id"] for name in members
            )
            same_scenario = sum(
                states[name]["scenario"] == meta["scenario"] for name in members
            )
            same_repeats = sum(
                states[name]["repeat_count"] == meta["repeat_count"]
                for name in members
            )
            scores.append((same_stratum, len(members), same_scenario, same_repeats))
        assignment[key] = min(range(fold_count), key=lambda fold: scores[fold])
    return assignment


def state_balanced_targeted_pass(
    model: TorchMPPISemanticStructuredLocalQCritic,
    inputs: tuple[torch.Tensor, ...],
    targeted_train: dict[str, np.ndarray],
    train_states: list[str],
    state_locations: dict[str, dict[str, np.ndarray]],
    rng: np.random.Generator,
    scales: dict[str, np.ndarray | float],
    args: argparse.Namespace,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """One epoch of state-balanced targeted updates.

    For every train state, each repeat center contributes
    ``args.cv_locations_per_center`` sampled locations; the loss averages
    per-location losses within a state and then averages across states in a
    batch, implementing L = (1/|S|) sum_s (1/|A_s|) sum_a L(s,a).
    """
    value_scale = float(scales["value"])
    gradient_scale = torch.from_numpy(
        np.asarray(scales["gradient"], np.float32)
    ).to(device)
    chord_scale = torch.from_numpy(
        np.asarray(scales["chord"], np.float32)
    ).to(device)
    order = [train_states[index] for index in rng.permutation(len(train_states))]
    totals: dict[str, list[float]] = {key: [] for key in (
        "targeted_total", "targeted_value", "targeted_gradient",
        "targeted_cosine", "targeted_chord", "targeted_norm",
    )}
    for start in range(0, len(order), args.cv_states_per_batch):
        batch_states = order[start : start + args.cv_states_per_batch]
        per_state_loss = []
        component_logs: dict[str, list[float]] = {key: [] for key in (
            "targeted_value", "targeted_gradient", "targeted_cosine",
            "targeted_chord", "targeted_norm",
        )}
        for key in batch_states:
            groups = state_locations[key]
            sampled: list[np.ndarray] = []
            for center_rows in groups.values():
                count = min(args.cv_locations_per_center, len(center_rows))
                sampled.append(rng.choice(center_rows, size=count, replace=False))
            local_np = np.concatenate(sampled)
            local = torch.from_numpy(local_np).to(device)
            tq0, tg0, thessian = parameters_at_absolute_actions(
                model,
                inputs,
                torch.from_numpy(
                    targeted_train["context_index"][local_np]
                ).to(device),
                torch.from_numpy(
                    targeted_train["absolute_action"][local_np]
                ).to(device),
            )
            target_value = torch.from_numpy(
                targeted_train["value"][local_np]
            ).to(device)
            target_gradient = torch.from_numpy(
                targeted_train["gradient"][local_np]
            ).to(device)
            value_loss = F.smooth_l1_loss(
                (tq0 - target_value) / value_scale,
                torch.zeros_like(tq0), beta=0.5,
            )
            gradient_loss = F.smooth_l1_loss(
                (tg0 - target_gradient) / gradient_scale,
                torch.zeros_like(tg0), beta=0.5,
            )
            cosine_loss = (
                1.0 - F.cosine_similarity(tg0, target_gradient, dim=1)
            ).mean()
            delta = torch.from_numpy(
                (targeted_train["partner_action"][local_np]
                 - targeted_train["local_action"][local_np]).reshape(len(local_np), -1)
            ).to(device)
            predicted_delta = torch.einsum("bij,bj->bi", thessian, delta)
            true_delta = torch.from_numpy(
                targeted_train["partner_gradient"][local_np]
                - targeted_train["gradient"][local_np]
            ).to(device)
            chord_loss = F.smooth_l1_loss(
                (predicted_delta - true_delta) / chord_scale,
                torch.zeros_like(predicted_delta), beta=0.5,
            )
            norm_loss = F.smooth_l1_loss(
                torch.log(torch.linalg.vector_norm(tg0, dim=1) + 1e-4),
                torch.log(torch.linalg.vector_norm(target_gradient, dim=1) + 1e-4),
                beta=0.5,
            )
            state_loss = (
                args.targeted_value_weight * value_loss
                + args.targeted_gradient_weight * gradient_loss
                + args.targeted_cosine_weight * cosine_loss
                + args.targeted_chord_weight * chord_loss
                + args.targeted_norm_weight * norm_loss
            )
            per_state_loss.append(state_loss)
            for name, value in (
                ("targeted_value", value_loss),
                ("targeted_gradient", gradient_loss),
                ("targeted_cosine", cosine_loss),
                ("targeted_chord", chord_loss),
                ("targeted_norm", norm_loss),
            ):
                component_logs[name].append(float(value.detach()))
        loss = torch.stack(per_state_loss).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        totals["targeted_total"].append(float(loss.detach()))
        for name in component_logs:
            totals[name].append(float(np.mean(component_logs[name])))
    return {key: float(np.mean(value)) for key, value in totals.items()}


def train_fold(
    fold: int,
    seed: int,
    actor: TorchMPPIDeterministicCenterActor,
    inputs: tuple[torch.Tensor, ...],
    labels: dict[str, np.ndarray],
    train_positions: np.ndarray,
    validation_positions: np.ndarray,
    partner: np.ndarray,
    cross_action: np.ndarray,
    distance: np.ndarray,
    weight: np.ndarray,
    scales: dict[str, np.ndarray | float],
    targeted_train: dict[str, np.ndarray],
    train_states: list[str],
    state_locations: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TorchMPPISemanticStructuredLocalQCritic, dict[str, Any]]:
    set_seed(seed * 100 + fold)
    model = TorchMPPISemanticStructuredLocalQCritic(
        include_feedback=False,
        low_rank=2,
        dropout=args.dropout,
        hessian_scale=args.hessian_scale,
        hessian_enabled=True,
    ).to(device)
    initialize_semantic_encoder_from_actor(model, actor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.35, patience=12, min_lr=3e-7
    )
    rng = np.random.default_rng(260814 + seed * 100 + fold)
    own_action = torch.from_numpy(labels["actor_action"]).to(device)
    target_value = torch.from_numpy(labels["value"]).to(device)
    target_gradient = torch.from_numpy(labels["gradient"]).to(device)
    probe_action = torch.from_numpy(labels["actions"][:, 0]).to(device)
    probe_target = torch.from_numpy(labels["transformed_reward"][:, 0]).to(device)
    cross_action_t = torch.from_numpy(cross_action).to(device)
    partner_t = torch.from_numpy(partner).to(device)
    chord_weight_t = torch.from_numpy(weight).to(device)
    gradient_scale = torch.from_numpy(
        np.asarray(scales["gradient"], np.float32)
    ).to(device)
    chord_scale = torch.from_numpy(
        np.asarray(scales["chord"], np.float32)
    ).to(device)
    value_scale = float(scales["value"])
    probe_scale = float(scales["probe"])
    best_score, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_positions)
        base_losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            pos_np = order[start : start + args.batch_size]
            pos = torch.from_numpy(pos_np).to(device)
            context = torch.from_numpy(labels["context_index"][pos_np]).to(device)
            q0, g0, hessian = batch_parameters(model, inputs, context)
            tv, tg = target_value[pos], target_gradient[pos]
            value_loss = F.smooth_l1_loss(
                (q0 - tv) / value_scale, torch.zeros_like(q0), beta=0.5
            )
            gradient_loss = F.smooth_l1_loss(
                (g0 - tg) / gradient_scale, torch.zeros_like(g0), beta=0.5
            )
            cosine_loss = (1.0 - F.cosine_similarity(g0, tg, dim=1)).mean()
            delta_probe = probe_action[pos] - own_action[pos, None]
            predicted_probe = model.local_value(
                q0[:, None], g0[:, None], hessian[:, None], delta_probe
            )
            probe_loss = F.smooth_l1_loss(
                (predicted_probe - probe_target[pos]) / probe_scale,
                torch.zeros_like(predicted_probe), beta=0.5,
            )
            delta_chord = cross_action_t[pos] - own_action[pos]
            predicted_delta_gradient = torch.einsum(
                "bij,bj->bi", hessian, delta_chord.flatten(1)
            )
            target_delta_gradient = target_gradient[partner_t[pos]] - tg
            per_component = F.smooth_l1_loss(
                (predicted_delta_gradient - target_delta_gradient) / chord_scale,
                torch.zeros_like(predicted_delta_gradient), beta=0.5,
                reduction="none",
            ).mean(1)
            chord_loss = torch.sum(
                chord_weight_t[pos] * per_component
            ) / torch.sum(chord_weight_t[pos])
            log_norm = torch.log(torch.linalg.vector_norm(g0, dim=1) + 1e-4)
            target_log_norm = torch.log(torch.linalg.vector_norm(tg, dim=1) + 1e-4)
            norm_loss = F.smooth_l1_loss(log_norm, target_log_norm, beta=0.5)
            loss = (
                args.value_weight * value_loss
                + args.gradient_weight * gradient_loss
                + args.cosine_weight * cosine_loss
                + args.probe_value_weight * probe_loss
                + args.chord_weight * chord_loss
                + args.norm_weight * norm_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            base_losses.append(float(loss.detach()))
        targeted_losses = state_balanced_targeted_pass(
            model, inputs, targeted_train, train_states, state_locations,
            rng, scales, args, device, optimizer,
        )
        validation = structured_metrics(
            model, inputs, labels, validation_positions, partner, cross_action,
            distance, args, device,
        )
        score = (
            1.0 - validation["gradient_cosine_median"]
            + 0.5 * (1.0 - validation["gradient_cosine_p10"])
            + 0.20 * validation["gradient_abs_log_norm_ratio_median"]
            + 0.25 * (1.0 - validation["cross_target_cosine_median"])
            + 0.50 * (1.0 - validation["true_reversal_flip_recall"])
            + 0.05 * validation["probe_value_rmse"]
        )
        if epoch >= args.scheduler_min_epoch:
            scheduler.step(score)
        eligible = (
            epoch >= args.selection_min_epoch
            and 0.5 <= validation["gradient_norm_ratio_median"] <= 2.0
        )
        if eligible and score < best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        elif eligible:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            history.append({
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "checkpoint_eligible": eligible,
                "score": float(score),
                "base_loss": float(np.mean(base_losses)),
                "targeted_loss": targeted_losses,
                "validation": validation,
            })
            print(json.dumps({
                "fold": fold, "seed": seed, **history[-1],
            }), flush=True)
        if eligible and stale >= args.patience:
            break
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        best_score = score
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return model, {
        "fold": int(fold),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "epochs_run": int(epoch),
        "best_validation_score": float(best_score),
        "history": history,
    }


def gate_block(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "cosine_median_ge_0_70": metrics["gradient_cosine_median"] >= 0.70,
        "cosine_p10_ge_0": metrics["gradient_cosine_p10"] >= 0.0,
        "norm_ratio_in_0_5_2_0": (
            0.5 <= metrics["gradient_norm_ratio_median"] <= 2.0
        ),
        "reversal_recall_ge_0_50": metrics["reversal_flip_recall"] >= 0.50,
    }


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)

    initial_payload = torch.load(args.initial_actor, map_location="cpu")
    alpha_payload = torch.load(
        initial_payload["base_alpha_checkpoint"], map_location="cpu"
    )
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    data, _, splits = load_dataset(Path(initial_payload["labels"]), old_payload)
    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    alpha_policy = make_base_policy(alpha_payload, device)
    _, _, _, alpha_center = deterministic_outputs(
        alpha_policy, tensors, extra, np.arange(len(data.episodes)),
        float(initial_payload["base_move_threshold"]),
        args.evaluation_batch_size, device,
    )
    inputs = list(actor_inputs(tensors, alpha_center, device))
    actor = TorchMPPIDeterministicCenterActor(
        float(initial_payload["maximum_residual_sigma"]), dropout=0.0
    ).to(device)
    actor.load_state_dict(initial_payload["actor_state_dict"], strict=True)
    actor.eval()

    labels = load_npz(args.labels_npz)
    labels["gradient"] = labels["gradient_by_radius"][:, 0].astype(np.float32)
    labels["value"] = labels["transformed_reward"][:, 0, 0].astype(np.float32)
    context = labels["context_index"].astype(np.int64)
    actor_center_by_context = np.zeros((len(data.episodes), 8, 2), np.float32)
    assigned = np.zeros(len(data.episodes), bool)
    for position, value in enumerate(context):
        actor_center_by_context[value] = labels["actor_center"][position]
        assigned[value] = True
    if not np.all(assigned):
        raise AssertionError("absolute action map is incomplete")
    inputs[3] = torch.from_numpy(actor_center_by_context).to(device)
    inputs = tuple(inputs)
    partner = partner_positions(data, labels)
    cross_action = cross_actions(
        data, labels, partner, alpha_center,
        float(initial_payload["maximum_residual_sigma"]),
    )
    distance = chord_distance_sigma(data, labels, partner)
    weight = chord_weights(distance, args)
    fit_positions = np.flatnonzero(np.isin(
        data.episodes[context], splits["internal_fit"]
    ))
    fit_episodes = np.unique(data.episodes[context[fit_positions]])
    split_rng = np.random.default_rng(260812)
    shuffled = fit_episodes.copy()
    split_rng.shuffle(shuffled)
    validation_episodes = set(shuffled[: max(1, len(shuffled) // 5)])
    validation_positions = fit_positions[np.asarray([
        episode in validation_episodes
        for episode in data.episodes[context[fit_positions]]
    ])]
    train_positions = fit_positions[np.asarray([
        episode not in validation_episodes
        for episode in data.episodes[context[fit_positions]]
    ])]
    scales: dict[str, np.ndarray | float] = {
        "value": max(float(np.std(labels["value"][train_positions])), 0.05),
        "gradient": np.maximum(
            np.std(labels["gradient"][train_positions], axis=0), 0.05
        ).astype(np.float32),
        "probe": max(float(np.std(
            labels["transformed_reward"][train_positions, 0]
        )), 0.05),
        "chord": np.maximum(np.std(
            labels["gradient"][partner[train_positions]]
            - labels["gradient"][train_positions], axis=0
        ), 0.05).astype(np.float32),
    }

    targeted_validation = json.loads(
        (args.targeted_labels_npz.parent / "validation_summary.json").read_text()
    )
    if targeted_validation["qualification"] != "TARGETED_LOCAL_RESPONSE_LABELS_VALIDATED":
        raise AssertionError("targeted response labels are not independently validated")
    if sha256_file(args.targeted_labels_npz) != targeted_validation["arrays_sha256"]:
        raise AssertionError("targeted response label hash changed after validation")
    targeted = load_npz(args.targeted_labels_npz)
    manifest = json.loads(args.manifest.read_text())
    states, row_state = build_state_table(manifest, targeted)
    control_rows = np.flatnonzero(targeted["pilot_role"] == "matched_easy_control")
    if len(control_rows) != 21:
        raise AssertionError("matched easy control count mismatch")
    control_states = {
        f"{row['episode']}#{int(row['physical_snapshot_ordinal'])}"
        for row in manifest["rows"] if row["pilot_role"] == "matched_easy_control"
    }
    shared_states = control_states & set(states)
    control_state_by_row = {}
    for row in manifest["rows"]:
        if row["pilot_role"] == "matched_easy_control":
            control_state_by_row[int(row["context_index"])] = (
                f"{row['episode']}#{int(row['physical_snapshot_ordinal'])}"
            )
    strict_control_rows = np.asarray([
        row for row in control_rows
        if control_state_by_row[int(targeted["context_index"][row])]
        not in shared_states
    ], np.int64)
    if len(strict_control_rows) != 20:
        raise AssertionError(
            f"unexpected state-shared control count: {len(control_rows) - len(strict_control_rows)}"
        )

    prepared = prepare_targeted_training(targeted, role="target")
    location_state = np.asarray([
        row_state[int(row)] for row in prepared["source_row"]
    ])
    location_center = prepared["source_row"]
    state_locations: dict[str, dict[str, np.ndarray]] = {}
    for key in states:
        rows = np.flatnonzero(location_state == key)
        centers: dict[str, np.ndarray] = {}
        for source_row in np.unique(location_center[rows]):
            centers[str(int(source_row))] = np.sort(
                rows[location_center[rows] == source_row]
            )
        state_locations[key] = centers
    assignment = assign_folds(states, args.folds, args.fold_seed)
    fold_states: dict[int, list[str]] = {
        fold: sorted(key for key, owner in assignment.items() if owner == fold)
        for fold in range(args.folds)
    }
    fold_composition = {
        str(fold): {
            "state_count": len(keys),
            "context_count": sum(len(states[key]["context_rows"]) for key in keys),
            "strata": {
                stratum: sum(
                    states[key]["stratum_id"] == stratum for key in keys
                )
                for stratum in sorted({s["stratum_id"] for s in states.values()})
            },
            "scenarios": {
                scenario: sum(states[key]["scenario"] == scenario for key in keys)
                for scenario in sorted({s["scenario"] for s in states.values()})
            },
            "repeat_counts": {
                str(count): sum(
                    states[key]["repeat_count"] == count for key in keys
                )
                for count in (1, 2)
            },
        }
        for fold, keys in fold_states.items()
    }

    fresh = load_npz(args.fresh_npz)
    fresh_context = fresh["context_index"].astype(np.int64)
    consumed_context = set(
        targeted["context_index"][targeted["pilot_role"] == "target"]
        .astype(np.int64).tolist()
    )
    consumed_episode = set(
        fresh["episode"][np.asarray(
            [int(value) in consumed_context for value in fresh_context], bool
        )].tolist()
    )
    unseen_positions = np.asarray([
        position for position, value in enumerate(fresh["episode"])
        if value not in consumed_episode
    ], np.int64)
    if len(unseen_positions) != 230:
        raise AssertionError("untouched complete-episode subset mismatch")
    fresh_unseen = subset_first_axis(fresh, unseen_positions)

    baseline_summary = json.loads(args.baseline_summary.read_text())
    if baseline_summary["sources"]["initial_actor_sha256"] != sha256_file(args.initial_actor):
        raise AssertionError("baseline PA used a different frozen Actor")
    baseline_records = {
        int(record["seed"]): record
        for record in baseline_summary["arms"]["PA"]["records"]
    }
    baseline_control_metrics: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        record = baseline_records[seed]
        checkpoint = Path(record["checkpoint"])
        if sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise AssertionError(f"baseline seed{seed} checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location="cpu")
        model = TorchMPPISemanticStructuredLocalQCritic(
            include_feedback=bool(payload["include_feedback"]),
            low_rank=int(payload["low_rank"]),
            dropout=0.0,
            hessian_scale=float(payload["hessian_scale"]),
            hessian_enabled=bool(payload["hessian_enabled"]),
        ).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        baseline_control_metrics[str(seed)] = {
            "control_21": targeted_metrics(
                model, inputs, targeted, control_rows,
                args.evaluation_batch_size, device,
            ),
            "control_20_state_strict": targeted_metrics(
                model, inputs, targeted, strict_control_rows,
                args.evaluation_batch_size, device,
            ),
        }
        del model

    stratum_by_context_row = {
        int(row["context_index"]): row["stratum_id"]
        for row in manifest["rows"]
        if row["pilot_role"] == "target"
    }
    stratum_by_npz_row = {
        position: stratum_by_context_row[int(targeted["context_index"][position])]
        for position in np.flatnonzero(targeted["pilot_role"] == "target")
    }
    fold_records: list[dict[str, Any]] = []
    for fold in range(args.folds):
        heldout_states = fold_states[fold]
        train_states = [
            key for key in sorted(states) if key not in set(heldout_states)
        ]
        heldout_rows = np.sort(np.concatenate([
            states[key]["context_rows"] for key in heldout_states
        ]))
        for seed in seeds:
            model, training = train_fold(
                fold, seed, actor, inputs, labels, train_positions,
                validation_positions, partner, cross_action, distance, weight,
                scales, prepared, train_states, state_locations, args, device,
            )
            checkpoint_path = args.output_dir / f"cv_fold{fold}_seed{seed}.pt"
            torch.save({
                "format_version": 1,
                "model_class": "TorchMPPISemanticStructuredLocalQCritic",
                "model_state_dict": model.state_dict(),
                "arm": "PA",
                "fold": int(fold),
                "seed": int(seed),
                "heldout_states": heldout_states,
                "hessian_enabled": True,
                "low_rank": 2,
                "include_feedback": False,
                "hessian_scale": args.hessian_scale,
                "initial_actor": str(args.initial_actor.resolve()),
                "initial_actor_sha256": sha256_file(args.initial_actor),
                "contract": {
                    "actor_frozen": True,
                    "formal_validation_loaded": False,
                    "test_loaded": False,
                    "controls_never_trained": True,
                    "checkpoint_selected_by": "original_internal_validation_only",
                },
                "training": training,
            }, checkpoint_path)
            out_of_fold = targeted_metrics(
                model, inputs, targeted, heldout_rows,
                args.evaluation_batch_size, device,
            )
            by_stratum = {}
            for stratum in sorted(set(stratum_by_npz_row.values())):
                stratum_rows = heldout_rows[np.asarray([
                    stratum_by_npz_row[int(row)] == stratum
                    for row in heldout_rows
                ], bool)]
                by_stratum[stratum] = targeted_metrics(
                    model, inputs, targeted, stratum_rows,
                    args.evaluation_batch_size, device,
                )
            control = targeted_metrics(
                model, inputs, targeted, control_rows,
                args.evaluation_batch_size, device,
            )
            control_strict = targeted_metrics(
                model, inputs, targeted, strict_control_rows,
                args.evaluation_batch_size, device,
            )
            fresh_result = fresh_metrics(
                model, inputs, fresh_unseen,
                float(initial_payload["maximum_residual_sigma"]), args, device,
            )
            paired_metric_names = (
                "gradient_cosine_median", "gradient_cosine_p10",
                "gradient_norm_ratio_median", "reversal_flip_recall",
            )
            fold_records.append({
                "fold": int(fold),
                "seed": int(seed),
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "heldout_states": heldout_states,
                "heldout_context_count": int(len(heldout_rows)),
                "training": {
                    key: training[key] for key in (
                        "fold", "seed", "best_epoch", "epochs_run",
                        "best_validation_score",
                    )
                },
                "out_of_fold": out_of_fold,
                "out_of_fold_by_stratum": by_stratum,
                "control_21": control,
                "control_20_state_strict": control_strict,
                "control_21_paired_delta_vs_baseline": {
                    metric: control[metric]
                    - baseline_control_metrics[str(seed)]["control_21"][metric]
                    for metric in paired_metric_names
                },
                "control_20_state_strict_paired_delta_vs_baseline": {
                    metric: control_strict[metric]
                    - baseline_control_metrics[str(seed)][
                        "control_20_state_strict"
                    ][metric]
                    for metric in paired_metric_names
                },
                "fresh_fd_unseen_230": fresh_result,
                "out_of_fold_gates": gate_block(out_of_fold),
            })
            del model

    # Aggregation: per-fold pass (>=2/3 seeds pass cosine-median AND norm),
    # per-seed pooled gates (each seed's folds partition the 79 contexts),
    # and the control non-degradation rule.
    fold_pass = {}
    for fold in range(args.folds):
        runs = [record for record in fold_records if record["fold"] == fold]
        passing = sum(
            record["out_of_fold_gates"]["cosine_median_ge_0_70"]
            and record["out_of_fold_gates"]["norm_ratio_in_0_5_2_0"]
            for record in runs
        )
        fold_pass[str(fold)] = {
            "seed_pass_count": int(passing),
            "fold_passes_cosine_median_and_norm": passing >= 2,
        }
    folds_passing = sum(
        entry["fold_passes_cosine_median_and_norm"] for entry in fold_pass.values()
    )
    per_seed_pooled: dict[str, Any] = {}
    for seed in seeds:
        runs = [record for record in fold_records if record["seed"] == seed]
        # Pool per-fold out-of-fold metrics weighted by heldout location count;
        # each seed's 5 folds partition the 79 target contexts exactly once.
        weights = np.asarray([
            record["out_of_fold"]["location_count"] for record in runs
        ], np.float64)
        pooled = {
            "gradient_cosine_median_weighted_mean": float(np.average(
                [record["out_of_fold"]["gradient_cosine_median"] for record in runs],
                weights=weights,
            )),
            "gradient_cosine_p10_min": float(np.min([
                record["out_of_fold"]["gradient_cosine_p10"] for record in runs
            ])),
            "gradient_norm_ratio_median_weighted_mean": float(np.average(
                [record["out_of_fold"]["gradient_norm_ratio_median"] for record in runs],
                weights=weights,
            )),
            "reversal_flip_recall_weighted_mean": float(np.average(
                [record["out_of_fold"]["reversal_flip_recall"] for record in runs],
                weights=weights,
            )),
        }
        per_seed_pooled[str(seed)] = pooled
    control_deltas = {
        metric: [
            record["control_20_state_strict_paired_delta_vs_baseline"][metric]
            for record in fold_records
        ]
        for metric in (
            "gradient_cosine_median", "gradient_cosine_p10",
            "gradient_norm_ratio_median", "reversal_flip_recall",
        )
    }
    control_deltas_21 = {
        metric: [
            record["control_21_paired_delta_vs_baseline"][metric]
            for record in fold_records
        ]
        for metric in (
            "gradient_cosine_median", "gradient_cosine_p10",
            "gradient_norm_ratio_median", "reversal_flip_recall",
        )
    }
    control_rule = {
        "operational_rule": (
            "median paired delta vs old PA >= -0.02 for cosine median and P10 "
            "on the 20 state-unseen controls (operationalizes the "
            "pre-registered 'not significantly degraded'); context 6018 shares "
            "its physical state with target context 6019 and is excluded from "
            "the strict set but reported in control_21"
        ),
        "strict_20_median_delta": {
            metric: float(np.median(values))
            for metric, values in control_deltas.items()
        },
        "strict_20_min_delta": {
            metric: float(np.min(values))
            for metric, values in control_deltas.items()
        },
        "all_21_median_delta": {
            metric: float(np.median(values))
            for metric, values in control_deltas_21.items()
        },
    }
    control_ok = (
        control_rule["strict_20_median_delta"]["gradient_cosine_median"] >= -0.02
        and control_rule["strict_20_median_delta"]["gradient_cosine_p10"] >= -0.02
    )
    seed_full_pass = {}
    for seed in seeds:
        runs = [record for record in fold_records if record["seed"] == seed]
        pooled_pass_count = 0
        for record in runs:
            gates = record["out_of_fold_gates"]
            if all(gates.values()):
                pooled_pass_count += 1
        pooled = per_seed_pooled[str(seed)]
        seed_full_pass[str(seed)] = {
            "folds_with_all_gates": int(pooled_pass_count),
            "pooled_metrics": pooled,
            "full_pass": (
                pooled_pass_count == args.folds
                and pooled["gradient_cosine_median_weighted_mean"] >= 0.70
                and pooled["gradient_cosine_p10_min"] >= 0.0
                and 0.5 <= pooled["gradient_norm_ratio_median_weighted_mean"] <= 2.0
                and pooled["reversal_flip_recall_weighted_mean"] >= 0.50
            ),
        }
    seeds_passing = sum(entry["full_pass"] for entry in seed_full_pass.values())
    decision = {
        "folds_passing_cosine_median_and_norm": int(folds_passing),
        "fold_rule_passed_requires_ge_4": bool(folds_passing >= 4),
        "seeds_fully_passing": int(seeds_passing),
        "seed_rule_passed_requires_ge_2_of_3": bool(seeds_passing >= 2),
        "control_non_degradation_passed": bool(control_ok),
    }
    overall = (
        decision["fold_rule_passed_requires_ge_4"]
        and decision["seed_rule_passed_requires_ge_2_of_3"]
        and decision["control_non_degradation_passed"]
    )
    qualification = (
        "GROUPED_CV_H_RESPONSE_TRANSFER_PASS_EXPAND_COVERAGE_ONLY"
        if overall else
        "GROUPED_CV_H_RESPONSE_TRANSFER_FAIL_REPRESENTATION_OR_CONDITIONING"
    )
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "sources": {
            "initial_actor": str(args.initial_actor.resolve()),
            "initial_actor_sha256": sha256_file(args.initial_actor),
            "labels": str(args.labels_npz.resolve()),
            "labels_sha256": sha256_file(args.labels_npz),
            "fresh_fd": str(args.fresh_npz.resolve()),
            "fresh_fd_sha256": sha256_file(args.fresh_npz),
            "targeted_labels": str(args.targeted_labels_npz.resolve()),
            "targeted_labels_sha256": sha256_file(args.targeted_labels_npz),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "baseline_summary": str(args.baseline_summary.resolve()),
            "baseline_summary_sha256": sha256_file(args.baseline_summary),
        },
        "protocol": {
            "folds": args.folds,
            "seeds": seeds,
            "states_total": len(states),
            "target_contexts_total": 79,
            "grouping_key": "(episode, physical_snapshot_ordinal)",
            "locations_per_center_per_epoch": args.cv_locations_per_center,
            "states_per_batch": args.cv_states_per_batch,
            "state_level_loss": "L = (1/|S|) sum_s (1/|A_s|) sum_a L(s,a)",
            "checkpoint_selection": "original internal-validation split only",
            "controls": (
                "21 matched easy controls never trained; degradation gate uses "
                "the 20 whose physical state is also never trained (context "
                "6018 excluded, shares a state with target 6019)"
            ),
            "stratification": [
                "stratum_id", "scenario", "repeat_count", "clipped",
                "reference_speed_mps",
            ],
        },
        "fold_assignment": assignment,
        "fold_composition": fold_composition,
        "baseline_control_metrics": baseline_control_metrics,
        "records": fold_records,
        "decision": {
            **decision,
            "fold_pass": fold_pass,
            "seed_full_pass": seed_full_pass,
            "control_rule": control_rule,
            "overall_pass": bool(overall),
            "authorization_note": (
                "Pass only authorizes expanding H-response state coverage; it "
                "does not certify the g0 tail. Actor remains frozen either way."
            ),
        },
        "contract": {
            "actor_frozen": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "output": str((args.output_dir / "summary.json").resolve()),
        "qualification": qualification,
        "decision": summary["decision"]["overall_pass"],
        "folds_passing": int(folds_passing),
        "seeds_passing": int(seeds_passing),
        "control_ok": bool(control_ok),
    }, indent=2))


if __name__ == "__main__":
    main()
