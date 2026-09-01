#!/usr/bin/env python3
"""Phase 2 A0: actor learnability baseline under the current contract.

3-fold x 3-seed episode-grouped cross-fit on the 600 Phase 1b multi-128
states. The actor is the current TorchMPPIDeterministicCenterActor used
unchanged: center = clamp(a0 + tanh(head) * 2 * sigma_base, +-1), trained
with plain action-space MSE against label_knots (stay states labeled a0).
Inputs are the exact deployment contract rebuilt per state: normalized
history / reference / current from the source snapshot, feedback and
gradient context from the trust context label, anchor slot = a0.

Pre-registered gates (review doc section 11.49): aggregated recovery
H = sum(J_a0 - J_pi) / sum(J_a0 - J_teacher); >= 2/3 seeds H_OOF >= 0.50;
median-seed episode-bootstrap 95% CI lower bound > 0; all seeds positive;
tail Delta J P05 >= 0 with worst reported; H_train - H_OOF > 0.15 flags a
generalization problem. Decision tree per section 11.49.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import (
    TorchMPPICostWeights,
    TorchMPPIParams,
)
from car_foundation.mppi_proposal_policy import (
    TorchMPPIDeterministicCenterActor,
)
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from generate_dbm_proposal_teacher import sha256_file
from run_mppi_proximal_search_phase1a import (
    DEFAULT_GT_TRAIN,
    DEFAULT_REPLAY_LABELS,
    DEFAULT_SCENARIO_PLAN,
    load_states,
    select_states,
)
from train_mppi_direct_trust_region_actor import (
    _normalization_inputs,
    load_actor_payload,
)
from mppi_a2_actors import ARCHITECTURES, FLAT_SENSITIVITY
from mppi_tcn_actors import TCNDirectNoAnchorActor


DEFAULT_LABELS = Path(
    "outputs/mppi_proposal/proximal_search_phase1b_20260818_v1/labels.npz"
)
DEFAULT_MANIFEST = Path(
    "outputs/mppi_proposal/proximal_search_phase1b_20260818_v1/manifest.json"
)
DEFAULT_INITIAL = Path(
    "outputs/mppi_proposal/direct_residual_online_ac_20260811_v2/"
    "direct_residual_online_ac_selected.pt"
)
DEFAULT_TRUST_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_direct_trust_region_train_20260807_v1"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/actor_a0_baseline_20260818_v1")
BOOTSTRAP = 2000
BLOCK_NAMES = (
    "history", "reference", "current", "anchor", "feedback", "gradient",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--replay-labels", type=Path, default=DEFAULT_REPLAY_LABELS)
    parser.add_argument("--scenario-plan", type=Path, default=DEFAULT_SCENARIO_PLAN)
    parser.add_argument("--gt-train", type=Path, default=DEFAULT_GT_TRAIN)
    parser.add_argument("--trust-labels", type=Path, default=DEFAULT_TRUST_LABELS)
    parser.add_argument("--initial-actor", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--selection-min-epoch", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--stay-move-threshold", type=float, default=0.05)
    parser.add_argument("--repeat", type=int, default=0, choices=(0, 1))
    parser.add_argument(
        "--s-from-b0", type=Path, default=None,
        help="B0 sensitivity.npz; when set, the loss is computed in "
             "influence-normalized coordinates z = S * delta_a with S "
             "refit on each fold's training episodes",
    )
    parser.add_argument(
        "--stay-flags-file", type=Path, default=None,
        help="optional labels npz whose 'stay' flags define the stay class "
             "(default: the --labels file's own stay flags)",
    )
    parser.add_argument(
        "--loss", choices=("plain", "stay_balanced"), default="plain",
        help="stay_balanced weights stay samples by mover_count/stay_count",
    )
    parser.add_argument(
        "--supervision-objective",
        choices=("action_mse", "trajectory_effect", "trajectory_hybrid"),
        default="action_mse",
        help=(
            "action_mse reproduces the historical baseline; trajectory_effect "
            "matches the frozen-DBM rollout produced by the J16 label; "
            "trajectory_hybrid adds a small normalized action-MSE regularizer"
        ),
    )
    parser.add_argument(
        "--trajectory-action-weight", type=float, default=0.05,
        help="normalized action-MSE weight for trajectory_hybrid",
    )
    parser.add_argument(
        "--trajectory-rate-weight", type=float, default=1.0,
        help="penalty on predicted control-rate cost above the teacher rate cost",
    )
    parser.add_argument(
        "--arch", choices=("base", "g", "t", "gt", "direct", "direct_noanchor",
                           "direct_noanchor_gt", "direct_noanchor_gt_clean",
                           "direct_noanchor_gt_x", "direct_noanchor_gt_frenet",
                           "direct_noanchor_gt_xf",
                           "direct_noanchor_gt_x_refcross",
                           "direct_noanchor_gt_x_refhistcross",
                           "gt_current_skip",
                           "gt_no_attention", "tcn_noanchor",
                           "tcn_gt"), default="base",
        help="actor architecture arm (11.54 A2 contract)",
    )
    parser.add_argument(
        "--gain-stay-threshold", type=float, default=None,
        help=(
            "zero-rollout relabeling: set labels with guarded deterministic "
            "gain min(J_a0,J_warm)-J_teacher below this threshold to a0"
        ),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def rollout_cost(
    backend, weights, params, knots, states_six, current_actions, references,
    batch_size, device,
) -> np.ndarray:
    values = []
    for start in range(0, len(knots), batch_size):
        stop = min(start + batch_size, len(knots))
        actions = interpolate_knots(
            torch.from_numpy(knots[start:stop]).to(device), params.horizon
        ).unsqueeze(1)
        value = batched_cost(
            backend, weights, actions,
            torch.from_numpy(states_six[start:stop]).to(device),
            torch.from_numpy(current_actions[start:stop]).to(device),
            torch.from_numpy(references[start:stop]).to(device),
        )
        values.append(value[:, 0].cpu().numpy())
    return np.concatenate(values)


def rollout_full_state(
    backend, knots, states_six, horizon, batch_size, device,
) -> torch.Tensor:
    """Frozen differentiable-DBM rollout target returned on ``device``."""
    values = []
    for start in range(0, len(knots), batch_size):
        stop = min(start + batch_size, len(knots))
        actions = interpolate_knots(
            torch.from_numpy(knots[start:stop]).to(device), horizon
        )
        initial = torch.from_numpy(states_six[start:stop]).to(device)
        with torch.no_grad():
            values.append(
                backend.rollout_full_state_differentiable(initial, actions)
            )
    return torch.cat(values, dim=0)


def trajectory_effect_loss_per_sample(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weights: TorchMPPICostWeights,
) -> torch.Tensor:
    """Task-weighted trajectory matching, averaged over the 50-step horizon."""
    position = (predicted[..., :2] - target[..., :2]).square().sum(-1)
    yaw_delta = predicted[..., 2] - target[..., 2]
    yaw = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)).square()
    vx = (predicted[..., 3] - target[..., 3]).square()
    value = weights.position * position + weights.yaw * yaw + weights.vx * vx
    if weights.yawrate != 0:
        value = value + weights.yawrate * (
            predicted[..., 5] - target[..., 5]
        ).square()
    return value.mean(dim=1)


def action_rate_cost_per_sample(
    actions: torch.Tensor,
    current_action: torch.Tensor,
    weights: TorchMPPICostWeights,
) -> torch.Tensor:
    previous = torch.cat((current_action[:, None], actions[:, :-1]), dim=1)
    rate = actions - previous
    return (
        weights.acceleration_rate * rate[..., 0].square()
        + weights.steering_rate * rate[..., 1].square()
    ).mean(dim=1)


def episode_bootstrap_ci(episodes, delta, baseline, seed, replicates=BOOTSTRAP):
    unique = np.unique(episodes)
    rng = np.random.default_rng(260818 + seed)
    delta_by_episode = np.asarray([
        delta[episodes == episode].sum() for episode in unique
    ], dtype=np.float64)
    baseline_by_episode = np.asarray([
        baseline[episodes == episode].sum() for episode in unique
    ], dtype=np.float64)
    values = []
    for _ in range(replicates):
        draw = rng.integers(len(unique), size=len(unique))
        # Preserve bootstrap multiplicity and resample numerator/denominator
        # together.  np.isin would collapse duplicate episode draws.
        values.append(float(
            delta_by_episode[draw].sum() / baseline_by_episode[draw].sum()
        ))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    args = parse_args()
    if args.trajectory_action_weight < 0 or args.trajectory_rate_weight < 0:
        raise ValueError("trajectory loss weights must be non-negative")
    if args.supervision_objective != "action_mse":
        if args.loss != "plain" or args.s_from_b0 is not None:
            raise ValueError(
                "trajectory objective A/B requires plain state weighting and "
                "no sensitivity-normalized action loss"
            )
        if args.gain_stay_threshold is not None:
            raise ValueError("trajectory objective A/B forbids gain relabeling")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    seeds = [int(value) for value in args.seeds.split(",")]

    labels = dict(np.load(args.labels, allow_pickle=False))
    manifest = json.loads(args.manifest.read_text())
    states_meta = manifest["states"]
    if len(states_meta) != len(labels["episodes"]):
        raise AssertionError("manifest/labels state count mismatch")

    loaded = select_states(load_states(args), len(states_meta))
    if [s["episode"] for s in loaded] != [s["episode"] for s in states_meta]:
        raise AssertionError("state selection order mismatch vs manifest")

    initial = torch.load(args.initial_actor, map_location="cpu")
    alpha_payload = torch.load(initial["base_alpha_checkpoint"], map_location="cpu")
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))

    blocks = {name: [] for name in BLOCK_NAMES}
    states_six, current_actions, references, frenet_features = [], [], [], []
    a0_list, sigma_list = [], []
    for meta, state in zip(states_meta, loaded):
        replay_path = args.replay_labels / meta["episode"] / meta["snapshot"]
        trust_path = args.trust_labels / meta["episode"] / meta["snapshot"]
        with np.load(replay_path, allow_pickle=False) as replay, np.load(
            trust_path, allow_pickle=False
        ) as trust:
            a0 = np.asarray(
                replay["bootstrap_actor_center"][args.repeat], np.float32
            )
            sigma = np.asarray(replay["sigma"], np.float32)
            source_path = Path(str(replay["source_snapshot"]))
            context_path = Path(str(trust["context_label"]))
            risk_text = str(trust["risk_label"])
            with np.load(source_path, allow_pickle=False) as source, np.load(
                context_path, allow_pickle=False
            ) as context:
                if risk_text:
                    with np.load(Path(risk_text), allow_pickle=False) as risk:
                        gradient_mean = np.asarray(
                            risk["critic_gradient_mean"], np.float32
                        )
                        gradient_std = np.asarray(
                            risk["critic_gradient_std"], np.float32
                        )
                else:
                    gradient_mean = np.asarray(
                        context["critic_gradient_mean"], np.float32
                    )
                    gradient_std = np.asarray(
                        context["critic_gradient_std"], np.float32
                    )
                values = _normalization_inputs(
                    source, context, gradient_mean, gradient_std,
                    args.repeat, old_payload,
                )
                for name, value in zip(BLOCK_NAMES, values):
                    blocks[name].append(value)
                six = np.asarray(source["initial_state_six"], np.float32)
                action = np.asarray(source["current_action"], np.float32)
                reference = np.asarray(source["reference"], np.float32)
                frenet = np.asarray(source["frenet_pose"], np.float32)
                frenet_features.append(np.asarray(
                    (frenet[1], np.sin(frenet[2]), np.cos(frenet[2])),
                    np.float32,
                ))
                mppi_config = json.loads(str(source["mppi_params_json"]))
                if len(reference) == int(mppi_config["horizon"]) + 1:
                    reference = reference[1:]
        blocks["anchor"][-1] = a0
        states_six.append(six)
        current_actions.append(action)
        references.append(reference)
        a0_list.append(a0)
        sigma_list.append(np.broadcast_to(sigma, (8, 2)))

    if args.arch in ("direct_noanchor_gt_frenet", "direct_noanchor_gt_xf"):
        # Reuse the fixed 32-D call slot only as a transport container.  The
        # corresponding actors zero this tensor before the legacy encoder and
        # read only these three deployable local-geometry values in the token
        # decoder; no first-pass gradient value remains visible.
        blocks["gradient"] = [
            np.pad(value, (0, 29)).astype(np.float32)
            for value in frenet_features
        ]

    inputs = tuple(
        torch.from_numpy(np.stack(blocks[name])).to(device)
        for name in BLOCK_NAMES
    )
    states_six = np.stack(states_six)
    current_actions = np.stack(current_actions)
    references = np.stack(references)
    a0_knots = np.stack(a0_list).astype(np.float32)
    label_knots = labels["label_knots"].astype(np.float32)
    if args.stay_flags_file is not None:
        stay_source = dict(
            np.load(args.stay_flags_file, allow_pickle=False)
        )
        stay = stay_source["stay"].astype(bool)
    else:
        stay = labels["stay"].astype(bool)
    original_stay = stay.copy()
    forced_stay = np.zeros(len(stay), dtype=bool)
    source_gain = (
        np.minimum(labels["j_a0"], labels["j_warm"])
        - labels["j_teacher"]
    ).astype(np.float32)
    if args.gain_stay_threshold is not None:
        if args.gain_stay_threshold < 0:
            raise ValueError("--gain-stay-threshold must be non-negative")
        forced_stay = source_gain < args.gain_stay_threshold
        label_knots = label_knots.copy()
        label_knots[forced_stay] = a0_knots[forced_stay]
        stay = stay | forced_stay
    episodes = np.asarray([
        value.split("#")[0] for value in labels["episodes"].astype(str)
    ])
    state_keys = labels["episodes"].astype(str)
    speeds = labels["speeds"].astype(np.float32)
    scenarios = labels["scenarios"].astype(str)
    count = len(states_meta)
    np.savez_compressed(
        args.output_dir / "gain_stay_relabel.npz",
        source_gain=source_gain,
        original_stay=original_stay,
        forced_stay=forced_stay,
        effective_stay=stay,
    )

    params = TorchMPPIParams(**json.loads(str(loaded[0]["mppi_params"])))
    weights = TorchMPPICostWeights(**json.loads(str(loaded[0]["cost_weights"])))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(str(loaded[0]["dbm_params"])))
    )

    j_a0 = rollout_cost(
        backend, weights, params, a0_knots, states_six, current_actions,
        references, args.evaluation_batch_size, device,
    )
    j_teacher = rollout_cost(
        backend, weights, params, label_knots, states_six, current_actions,
        references, args.evaluation_batch_size, device,
    )
    replay_gap = float(np.max(np.abs(j_a0 - labels["j_a0"].astype(np.float32))))
    if replay_gap > 1e-4:
        raise AssertionError(f"a0 replay mismatch vs labels: {replay_gap}")

    states_six_tensor = torch.from_numpy(states_six).to(device)
    current_actions_tensor = torch.from_numpy(current_actions).to(device)
    teacher_trajectory = None
    teacher_rate_cost = None
    if args.supervision_objective != "action_mse":
        teacher_trajectory = rollout_full_state(
            backend, label_knots, states_six, params.horizon,
            args.evaluation_batch_size, device,
        )
        teacher_actions = interpolate_knots(
            torch.from_numpy(label_knots).to(device), params.horizon
        )
        teacher_rate_cost = action_rate_cost_per_sample(
            teacher_actions, current_actions_tensor, weights
        ).detach()

    b0 = None
    if args.s_from_b0 is not None:
        b0 = dict(np.load(args.s_from_b0, allow_pickle=False))
        if len(b0["knot_jacobians"]) != count:
            raise AssertionError("B0 state count mismatch")
    cell_episodes = {}
    for episode in np.unique(episodes):
        mask = episodes == episode
        key = (round(float(speeds[mask][0]), 2), str(scenarios[mask][0]))
        cell_episodes.setdefault(key, []).append(episode)
    fold_of_episode = {}
    for key in sorted(cell_episodes):
        members = sorted(cell_episodes[key])
        if len(members) != args.folds:
            raise AssertionError(
                f"cell {key} has {len(members)} episodes, expected {args.folds}"
            )
        for fold, episode in enumerate(members):
            fold_of_episode[episode] = fold
    fold_of_state = np.asarray(
        [fold_of_episode[episode] for episode in episodes], np.int64
    )

    records = []
    for seed in seeds:
        for fold in range(args.folds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            heldout = fold_of_state == fold
            train = ~heldout
            if args.arch == "tcn_noanchor":
                train_mask = fold_of_state != fold
                train_labels = label_knots[train_mask]
                center_t = torch.from_numpy(
                    train_labels.reshape(-1, 8, 2).mean(axis=0)
                ).to(device)
                scale_t = torch.from_numpy(
                    train_labels.reshape(-1, 8, 2).std(axis=0) + 1e-6
                ).to(device)
                actor = TCNDirectNoAnchorActor(
                    dropout=args.dropout,
                    center=center_t, scale=scale_t,
                ).to(device)
            elif args.arch == "tcn_gt":
                train_mask = fold_of_state != fold
                train_labels = label_knots[train_mask]
                center_t = torch.from_numpy(
                    train_labels.reshape(-1, 8, 2).mean(axis=0)
                ).to(device)
                scale_t = torch.from_numpy(
                    train_labels.reshape(-1, 8, 2).std(axis=0) + 1e-6
                ).to(device)
                actor = ARCHITECTURES[args.arch](
                    float(initial["maximum_residual_sigma"]),
                    dropout=args.dropout,
                    center=center_t, scale=scale_t,
                ).to(device)
            elif args.arch in ("direct_noanchor", "direct_noanchor_gt",
                               "direct_noanchor_gt_clean",
                               "direct_noanchor_gt_x",
                               "direct_noanchor_gt_frenet",
                               "direct_noanchor_gt_xf",
                               "direct_noanchor_gt_x_refcross",
                               "direct_noanchor_gt_x_refhistcross",
                               "gt_current_skip", "gt_no_attention"):
                train_mask = fold_of_state != fold
                train_labels = label_knots[train_mask]
                center_t = torch.from_numpy(
                    train_labels.reshape(-1, 8, 2).mean(axis=0)
                ).to(device)
                scale_t = torch.from_numpy(
                    train_labels.reshape(-1, 8, 2).std(axis=0) + 1e-6
                ).to(device)
                actor = ARCHITECTURES[args.arch](
                    float(initial["maximum_residual_sigma"]),
                    dropout=args.dropout,
                    center=center_t, scale=scale_t,
                ).to(device)
            else:
                actor = ARCHITECTURES[args.arch](
                    float(initial["maximum_residual_sigma"]),
                    dropout=args.dropout,
                ).to(device)
            optimizer = torch.optim.AdamW(
                actor.parameters(), lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.4, patience=8, min_lr=1e-6
            )
            rng = np.random.default_rng(260818 + seed * 100 + fold)
            target = torch.from_numpy(label_knots).to(device)
            if b0 is not None:
                position_knot = np.linalg.norm(
                    b0["knot_jacobians"][:, :2, :], axis=1
                )
                train_mask = fold_of_state != fold
                s_fold = np.sqrt(
                    (position_knot[train_mask] ** 2).mean(axis=0)
                ).astype(np.float32).reshape(1, 8, 2)
                scale_s = torch.from_numpy(s_fold).to(device)
            else:
                scale_s = None
            if args.loss == "stay_balanced":
                stay_count = float(np.sum(stay[train]))
                mover_count = float(np.sum(~stay[train]))
                stay_weight = float(
                    mover_count / stay_count if stay_count else 1.0
                )
            else:
                stay_weight = 1.0
            sample_weight = torch.from_numpy(
                np.where(stay, stay_weight, 1.0).astype(np.float32)
            ).to(device)
            train_rows_local = np.flatnonzero(train)
            action_loss_scale = 1.0
            trajectory_loss_scale = 1.0
            if args.supervision_objective != "action_mse":
                train_context = torch.from_numpy(train_rows_local).to(device)
                mean_center = target[train_context].mean(dim=0, keepdim=True)
                mean_center = mean_center.expand(len(train_rows_local), -1, -1)
                mean_actions = interpolate_knots(mean_center, params.horizon)
                with torch.no_grad():
                    mean_trajectory = backend.rollout_full_state_differentiable(
                        states_six_tensor[train_context], mean_actions
                    )
                    mean_traj_loss = trajectory_effect_loss_per_sample(
                        mean_trajectory,
                        teacher_trajectory[train_context],
                        weights,
                    )
                    mean_rate = action_rate_cost_per_sample(
                        mean_actions,
                        current_actions_tensor[train_context],
                        weights,
                    )
                    mean_excess_rate = torch.relu(
                        mean_rate - teacher_rate_cost[train_context]
                    )
                    trajectory_loss_scale = max(float(torch.mean(
                        mean_traj_loss
                        + args.trajectory_rate_weight * mean_excess_rate
                    )), 1e-8)
                    action_loss_scale = max(float(torch.mean(
                        (mean_center - target[train_context]).square()
                    )), 1e-8)
            best_loss, best_state, best_epoch, stale = float("inf"), None, None, 0
            for epoch in range(1, args.epochs + 1):
                actor.train()
                order = rng.permutation(len(train_rows_local))
                epoch_loss = []
                for start in range(0, len(order), args.batch_size):
                    batch = train_rows_local[
                        order[start : start + args.batch_size]
                    ]
                    context = torch.from_numpy(batch).to(device)
                    state_inputs = tuple(value[context] for value in inputs)
                    _, center = actor(*state_inputs)
                    difference = center - target[context]
                    if args.supervision_objective == "action_mse":
                        if scale_s is not None:
                            difference = difference * scale_s
                        per_element = difference ** 2
                        per_sample = per_element.flatten(1).mean(dim=1)
                    else:
                        predicted_actions = interpolate_knots(
                            center, params.horizon
                        )
                        predicted_trajectory = (
                            backend.rollout_full_state_differentiable(
                                states_six_tensor[context], predicted_actions
                            )
                        )
                        trajectory_per_sample = (
                            trajectory_effect_loss_per_sample(
                                predicted_trajectory,
                                teacher_trajectory[context],
                                weights,
                            )
                        )
                        predicted_rate = action_rate_cost_per_sample(
                            predicted_actions,
                            current_actions_tensor[context],
                            weights,
                        )
                        rate_excess = torch.relu(
                            predicted_rate - teacher_rate_cost[context]
                        )
                        per_sample = (
                            trajectory_per_sample
                            + args.trajectory_rate_weight * rate_excess
                        ) / trajectory_loss_scale
                        if args.supervision_objective == "trajectory_hybrid":
                            action_per_sample = difference.square().flatten(1).mean(1)
                            per_sample = per_sample + (
                                args.trajectory_action_weight
                                * action_per_sample / action_loss_scale
                            )
                    loss = float(stay_weight) * 0.0 + torch.sum(
                        sample_weight[context] * per_sample
                    ) / torch.sum(sample_weight[context])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
                    optimizer.step()
                    epoch_loss.append(float(loss.detach()))
                mean_loss = float(np.mean(epoch_loss))
                scheduler.step(mean_loss)
                # The early-stop counter starts only once checkpoints are
                # eligible.  Counting the warm-up epochs as stale would make
                # the default patience (25) terminate training before the
                # default selection_min_epoch (40), leaving best_state unset.
                if epoch >= args.selection_min_epoch:
                    if mean_loss < best_loss:
                        best_loss = mean_loss
                        best_state = copy.deepcopy(actor.state_dict())
                        best_epoch = epoch
                        stale = 0
                    else:
                        stale += 1
                    if stale >= args.patience:
                        break
            if best_state is not None:
                actor.load_state_dict(best_state, strict=True)
            actor.eval()
            predicted_chunks = []
            with torch.no_grad():
                for start in range(0, count, 256):
                    context = torch.from_numpy(
                        np.arange(start, min(start + 256, count))
                    ).to(device)
                    state_inputs = tuple(value[context] for value in inputs)
                    _, center = actor(*state_inputs)
                    predicted_chunks.append(center.cpu().numpy())
            predicted = np.concatenate(predicted_chunks).astype(np.float32)
            j_actor = rollout_cost(
                backend, weights, params, predicted, states_six,
                current_actions, references, args.evaluation_batch_size,
                device,
            )
            baseline_delta = j_a0 - j_teacher
            actor_delta = j_a0 - j_actor
            h_train = float(
                actor_delta[train].sum() / baseline_delta[train].sum()
            )
            h_oof = float(
                actor_delta[heldout].sum() / baseline_delta[heldout].sum()
            )
            moved = (
                np.linalg.norm(
                    (predicted - a0_knots).reshape(count, -1), axis=1
                )
                > args.stay_move_threshold
            )
            knot_error = np.abs(predicted - label_knots).reshape(count, 8, 2)
            records.append({
                "seed": seed, "fold": fold,
                "h_train": h_train, "h_oof": h_oof,
                "p05_delta_heldout": float(
                    np.quantile(actor_delta[heldout], 0.05)
                ),
                "worst_delta_heldout": float(np.min(actor_delta[heldout])),
                "stay_false_move_rate": (
                    float(np.mean(moved[heldout & stay]))
                    if np.sum(heldout & stay) else None
                ),
                "mover_missed_rate": (
                    float(np.mean(~moved[heldout & ~stay]))
                    if np.sum(heldout & ~stay) else None
                ),
                "knot0_2_error_median": float(np.median(
                    knot_error[:, :3, :].reshape(-1)
                )),
                "knot3_7_error_median": float(np.median(
                    knot_error[:, 3:, :].reshape(-1)
                )),
                "knot0_2_steer_error_median": float(np.median(
                    knot_error[:, :3, 1].reshape(-1)
                )),
                "knot0_2_accel_error_median": float(np.median(
                    knot_error[:, :3, 0].reshape(-1)
                )),
                "sens_weighted_error_median": float(np.median(
                    (np.abs(predicted - label_knots).reshape(count, -1)
                     * np.asarray(FLAT_SENSITIVITY, np.float32)).sum(axis=1)
                )),
                "best_epoch_loss": best_loss,
                "best_epoch": best_epoch,
                "supervision_objective": args.supervision_objective,
                "trajectory_loss_scale": trajectory_loss_scale,
                "action_loss_scale": action_loss_scale,
                "_delta_heldout": actor_delta[heldout].tolist(),
                "_baseline_heldout": baseline_delta[heldout].tolist(),
                "_episodes_heldout": episodes[heldout].tolist(),
                "_state_keys_heldout": state_keys[heldout].tolist(),
                "_predicted_heldout": predicted[heldout].astype(np.float32).tolist(),
            })
            torch.save({
                "model_state_dict": actor.state_dict(),
                "fold": fold, "seed": seed,
                "heldout_episodes": sorted(set(episodes[heldout].tolist())),
                "initial_actor_sha256": sha256_file(args.initial_actor),
            }, args.output_dir / f"a0_fold{fold}_seed{seed}.pt")
            del actor

    per_seed = {}
    for seed in seeds:
        runs = [r for r in records if r["seed"] == seed]
        per_seed[str(seed)] = {
            "h_oof_folds": [r["h_oof"] for r in runs],
            "h_oof_pooled": float(
                np.sum([
                    np.asarray(r["_delta_heldout"]).sum() for r in runs
                ]) / np.sum([
                    np.asarray(r["_baseline_heldout"]).sum() for r in runs
                ])
            ),
            "h_train_median": float(np.median([r["h_train"] for r in runs])),
        }
    median_seed = str(seeds[int(np.argsort(
        [per_seed[str(seed)]["h_oof_pooled"] for seed in seeds]
    )[len(seeds) // 2])])
    median_runs = [r for r in records if r["seed"] == int(median_seed)]
    deltas = np.concatenate([
        np.asarray(r["_delta_heldout"]) for r in median_runs
    ])
    baseline = np.concatenate([
        np.asarray(r["_baseline_heldout"]) for r in median_runs
    ])
    eps = np.concatenate([
        np.asarray(r["_episodes_heldout"]) for r in median_runs
    ])
    ci = episode_bootstrap_ci(eps, deltas, baseline, seed=int(median_seed))
    gates = {
        "main_two_of_three_seeds_h_oof_ge_0_50": bool(
            sum(per_seed[str(seed)]["h_oof_pooled"] >= 0.50 for seed in seeds)
            >= 2
        ),
        "median_seed_ci_lower_positive": bool(ci[0] > 0),
        "all_seeds_positive": bool(all(
            per_seed[str(seed)]["h_oof_pooled"] > 0 for seed in seeds
        )),
        "tail_p05_ge_0_median_seed": bool(np.quantile(deltas, 0.05) >= 0),
        "generalization_gap_gt_0_15": bool(
            per_seed[median_seed]["h_train_median"]
            - per_seed[median_seed]["h_oof_pooled"] > 0.15
        ),
    }
    np.savez_compressed(
        args.output_dir / "oof_evaluation.npz",
        seed=np.concatenate([
            np.full(len(r["_delta_heldout"]), r["seed"], dtype=np.int64)
            for r in records
        ]),
        fold=np.concatenate([
            np.full(len(r["_delta_heldout"]), r["fold"], dtype=np.int64)
            for r in records
        ]),
        episode=np.concatenate([
            np.asarray(r["_episodes_heldout"]) for r in records
        ]),
        actor_gain=np.concatenate([
            np.asarray(r["_delta_heldout"], dtype=np.float32) for r in records
        ]),
        teacher_gain=np.concatenate([
            np.asarray(r["_baseline_heldout"], dtype=np.float32) for r in records
        ]),
        predicted_knots=np.concatenate([
            np.asarray(r["_predicted_heldout"], dtype=np.float32) for r in records
        ]),
        state_keys=np.concatenate([
            np.asarray(r["_state_keys_heldout"]) for r in records
        ]),
    )
    for record in records:
        for key in (
            "_delta_heldout", "_baseline_heldout", "_episodes_heldout",
            "_predicted_heldout", "_state_keys_heldout",
        ):
            record.pop(key, None)
    median_h_train = per_seed[median_seed]["h_train_median"]
    median_h_oof = per_seed[median_seed]["h_oof_pooled"]
    if median_h_train < 0.50:
        decision = "TRAIN_FAIL_FIT_STRUCTURE_OR_LABEL_PROBLEM"
    elif median_h_oof < 0.50 and gates["generalization_gap_gt_0_15"]:
        decision = "GENERALIZATION_PROBLEM"
    elif median_h_oof >= 0.50 and gates["tail_p05_ge_0_median_seed"]:
        decision = "ACTOR_LEARNABILITY_PASS"
    else:
        decision = "BODY_LEARNABLE_TAIL_NOT_PASSED"
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": f"A0_{decision}",
        "sources": {
            "labels": str(args.labels.resolve()),
            "labels_sha256": sha256_file(args.labels),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "initial_actor": str(args.initial_actor.resolve()),
            "initial_actor_sha256": sha256_file(args.initial_actor),
            "replay_labels": str(args.replay_labels.resolve()),
            "trust_labels": str(args.trust_labels.resolve()),
        },
        "protocol": {
            "folds": args.folds, "seeds": seeds,
            "stratification": "one episode per speed-scenario cell per fold",
            "actor": args.arch,
            "actor_input_contract": (
                "history+reference+current only; anchor/feedback/gradient blinded"
                if args.arch == "direct_noanchor_gt_clean"
                else "architecture-defined; see checkpoint and source"
            ),
            "loss": args.loss,
            "supervision_objective": args.supervision_objective,
            "trajectory_action_weight": args.trajectory_action_weight,
            "trajectory_rate_weight": args.trajectory_rate_weight,
            "trajectory_loss_normalization": (
                "fold-train mean-center rollout mismatch plus excess-rate"
                if args.supervision_objective != "action_mse" else None
            ),
            "epochs": args.epochs, "patience": args.patience,
            "selection_min_epoch": args.selection_min_epoch,
            "sensitivity_normalized_loss": args.s_from_b0 is not None,
            "gain_stay_threshold": args.gain_stay_threshold,
            "gain_stay_forced_count": int(forced_stay.sum()),
            "effective_stay_count": int(stay.sum()),
            "stay_move_threshold": args.stay_move_threshold,
            "recovery": "aggregated sum ratio; never per-frame",
        },
        "per_seed": per_seed,
        "median_seed": median_seed,
        "bootstrap_ci_95_median_seed": ci,
        "gates": gates,
        "decision": decision,
        "records": records,
        "contract": {
            "actor_frozen": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "anchor_visible_to_actor": args.arch not in (
                "direct_noanchor", "direct_noanchor_gt",
                "direct_noanchor_gt_clean", "gt_current_skip",
                "direct_noanchor_gt_x", "direct_noanchor_gt_frenet",
                "direct_noanchor_gt_xf",
                "direct_noanchor_gt_x_refcross",
                "direct_noanchor_gt_x_refhistcross",
                "gt_no_attention", "tcn_noanchor", "tcn_gt",
            ),
            "first_pass_feedback_visible_to_actor": (
                args.arch not in (
                    "direct_noanchor_gt_clean", "direct_noanchor_gt_x",
                    "direct_noanchor_gt_frenet", "direct_noanchor_gt_xf",
                    "direct_noanchor_gt_x_refcross",
                    "direct_noanchor_gt_x_refhistcross",
                )
            ),
            "labels": str(args.labels.resolve()),
            "stay_flags": (
                str(args.stay_flags_file.resolve())
                if args.stay_flags_file is not None else str(args.labels.resolve())
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({
        "output": str((args.output_dir / "summary.json").resolve()),
        "qualification": summary["qualification"],
        "per_seed": {
            seed: {
                "h_oof_pooled": round(value["h_oof_pooled"], 4),
                "h_train_median": round(value["h_train_median"], 4),
            }
            for seed, value in per_seed.items()
        },
        "ci": ci, "gates": gates, "decision": decision,
    }, indent=2))


if __name__ == "__main__":
    main()
