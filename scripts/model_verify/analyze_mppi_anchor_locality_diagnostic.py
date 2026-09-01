#!/usr/bin/env python3
"""Diagnostics 2+3: anchor locality of the output head (zero rollout).

Diagnostic 2: does the actor's output depend on the anchor in a
temporally-local (per-knot) way? Measure the Jacobian of the output center
w.r.t. the anchor knots; beyond the mechanical identity (center = a0 +
residual), the residual's anchor-Jacobian should concentrate on the 2x2
block diagonal for a model with per-knot correspondence. Token heads are
expected to show higher block-diagonal concentration than the flat MLP.
Secondary: anchor knot-reversal shift vs the mechanical shift.

Diagnostic 3: does the network allocate state-responsiveness to the
high-leverage knots? Correlate per-knot (predicted and teacher) residual
variance shares with the B0 terminal-position leverage profile.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import (
    TorchMPPIDeterministicCenterActor,
)
from mppi_a2_actors import (
    A2GeometryActor,
    A2GeometryTemporalActor,
    A2TemporalActor,
)
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


DEFAULT_INITIAL = Path(
    "outputs/mppi_proposal/direct_residual_online_ac_20260811_v2/"
    "direct_residual_online_ac_selected.pt"
)
DEFAULT_TRUST_LABELS = Path(
    "/disk/collect_data_from_anycar/mppi_rl_closed_loop/labels/"
    "dbm_direct_trust_region_train_20260807_v1"
)
DEFAULT_B0 = Path("outputs/mppi_proposal/b0_sensitivity_20260818_v1/sensitivity.npz")
DEFAULT_COUPLING = Path(
    "outputs/mppi_proposal/output_temporal_coupling_diagnostic_20260818_v1/"
    "summary.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/anchor_locality_diagnostic_20260818_v1"
)
ARCHS = {
    "base": (
        TorchMPPIDeterministicCenterActor,
        Path("outputs/mppi_proposal/actor_curve_n1800_20260818_v1/"
             "a0_fold0_seed0.pt"),
    ),
    "A2-G": (
        A2GeometryActor,
        Path("outputs/mppi_proposal/a2_g_n1800_20260818_v1/a0_fold0_seed0.pt"),
    ),
    "A2-T": (
        A2TemporalActor,
        Path("outputs/mppi_proposal/a2_t_n1800_20260818_v1/a0_fold0_seed0.pt"),
    ),
    "A2-GT": (
        A2GeometryTemporalActor,
        Path("outputs/mppi_proposal/a2_gt_n1800_20260818_v1/a0_fold0_seed0.pt"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=200)
    parser.add_argument("--initial-actor", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--trust-labels", type=Path, default=DEFAULT_TRUST_LABELS)
    parser.add_argument("--b0", type=Path, default=DEFAULT_B0)
    parser.add_argument("--coupling", type=Path, default=DEFAULT_COUPLING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def build_inputs(states, old_payload, trust_labels: Path, device):
    blocks = {name: [] for name in (
        "history", "reference", "current", "anchor", "feedback", "gradient",
    )}
    a0_list = []
    for state in states:
        replay_path = Path(state["replay_label_path"]) if "replay_label_path" in state else None
        trust_path = trust_labels / state["episode"] / state["snapshot"]
        with np.load(trust_path, allow_pickle=False) as trust:
            source_path = Path(str(trust["source_snapshot"]))
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
                    0, old_payload,
                )
                for name, value in zip(blocks, values):
                    blocks[name].append(value)
        a0_list.append(state["a0"].astype(np.float32))
    inputs = tuple(
        torch.from_numpy(np.stack(blocks[name])).to(device)
        for name in blocks
    )
    return inputs, np.stack(a0_list).astype(np.float32)


def knot_block_diagonal_concentration(jacobian: np.ndarray, valid: np.ndarray):
    """Fraction of squared residual-Jacobian mass on 2x2 knot blocks."""
    residual = jacobian.copy()
    np.fill_diagonal(residual, residual.diagonal() - 1.0)
    mask = valid[:, None] & valid[None, :]
    squared = residual ** 2
    total = float(squared[mask].sum())
    if total < 1e-12:
        return None
    block_mass = 0.0
    for knot in range(8):
        rows = slice(knot * 2, knot * 2 + 2)
        block_mask = mask[rows, rows]
        block_mass += float(squared[rows, rows][block_mask].sum())
    return block_mass / total


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    device = torch.device(args.device)

    initial = torch.load(args.initial_actor, map_location="cpu")
    alpha_payload = torch.load(
        initial["base_alpha_checkpoint"], map_location="cpu"
    )
    old_payload = load_actor_payload(Path(alpha_payload["old_actor"]))
    states = select_states(load_states(SimpleNamespace(
        replay_labels=DEFAULT_REPLAY_LABELS,
        gt_train=DEFAULT_GT_TRAIN,
        scenario_plan=DEFAULT_SCENARIO_PLAN,
        repeat=0,
    )), 1800)[: args.states]
    inputs, a0 = build_inputs(states, old_payload, args.trust_labels, device)

    results = {}
    for arch, (cls, checkpoint_path) in ARCHS.items():
        payload = torch.load(checkpoint_path, map_location="cpu")
        model = cls(maximum_delta_sigma=2.0, dropout=0.0).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        concentrations = []
        reversal_ratios = []
        for index in range(len(states)):
            context = torch.arange(index, index + 1, device=device)
            base_inputs = [value[context] for value in inputs]
            anchor = torch.from_numpy(a0[index][None]).to(device)
            anchor = anchor.clone().requires_grad_(True)
            center = model(
                base_inputs[0], base_inputs[1], base_inputs[2], anchor,
                base_inputs[4], base_inputs[5],
            )
            if isinstance(center, tuple):
                center = center[1]
            flat = center.reshape(16)
            jacobian = torch.zeros(16, 16, device=device)
            for dim in range(16):
                gradient = torch.autograd.grad(
                    flat[dim], anchor, retain_graph=True
                )[0]
                jacobian[dim] = gradient.reshape(16)
            valid = (flat.abs() < 0.999).cpu().numpy()
            concentration = knot_block_diagonal_concentration(
                jacobian.detach().cpu().numpy(), valid
            )
            if concentration is not None:
                concentrations.append(concentration)
            with torch.no_grad():
                reversed_anchor = anchor.detach().clone().reshape(8, 2)
                reversed_anchor = reversed_anchor.flip(0)[None]
                center_rev = model(
                    base_inputs[0], base_inputs[1], base_inputs[2],
                    reversed_anchor, base_inputs[4], base_inputs[5],
                )
                if isinstance(center_rev, tuple):
                    center_rev = center_rev[1]
                shift = float((center_rev - center.detach()).norm())
                mechanical = float(
                    (reversed_anchor - anchor.detach()).norm()
                )
                reversal_ratios.append(shift / (mechanical + 1e-12))
        results[arch] = {
            "block_diag_concentration_median": float(np.median(concentrations)),
            "block_diag_concentration_p25": float(np.quantile(
                concentrations, 0.25
            )),
            "reversal_shift_over_mechanical_median": float(np.median(
                reversal_ratios
            )),
            "states_used": len(concentrations),
        }
        del model

    b0 = dict(np.load(args.b0, allow_pickle=False))
    per_dim_leverage = np.linalg.norm(
        b0["knot_jacobians"][:, :2, :], axis=1
    ).mean(axis=0)
    leverage = np.linalg.norm(per_dim_leverage.reshape(8, 2), axis=1)
    leverage_share = leverage / (leverage.sum() + 1e-12)
    coupling = json.loads(args.coupling.read_text())
    variance_correlations = {}
    for arch, row in coupling["results"].items():
        seed_data = row["per_seed"]["0"]
        pred_share = np.asarray(
            seed_data["variance_distribution_predicted"]
        )
        teacher_share = np.asarray(
            seed_data["variance_distribution_teacher"]
        )
        variance_correlations[arch] = {
            "predicted_vs_leverage": float(np.corrcoef(
                pred_share, leverage_share
            )[0, 1]),
            "teacher_vs_leverage": float(np.corrcoef(
                teacher_share, leverage_share
            )[0, 1]),
        }

    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "ANCHOR_LOCALITY_DIAGNOSED_ACTOR_FROZEN",
        "sources": {
            "architectures": {
                k: str(v[1]) for k, v in ARCHS.items()
            },
            "b0": str(args.b0.resolve()),
            "coupling": str(args.coupling.resolve()),
        },
        "metrics": {
            "block_diag_concentration": (
                "fraction of squared (J_center - I) mass on per-knot 2x2 "
                "blocks; higher = more temporally-local anchor dependence"
            ),
            "reversal_shift_over_mechanical": (
                "|center(a0_reversed) - center(a0)| / |a0_reversed - a0|; "
                "1.0 = purely mechanical (head ignores anchor), <1 = head "
                "compensates"
            ),
        },
        "results": results,
        "leverage_correlations": variance_correlations,
        "leverage_share_b0": leverage_share.tolist(),
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
    print(f"{'arch':7s} {'blockdiag':>10s} {'p25':>7s} {'rev_ratio':>10s} "
          f"{'varcorr_pred':>13s} {'varcorr_teach':>14s}")
    for arch, row in results.items():
        vc = variance_correlations[arch]
        print(f"{arch:7s} {row['block_diag_concentration_median']:10.3f} "
              f"{row['block_diag_concentration_p25']:7.3f} "
              f"{row['reversal_shift_over_mechanical_median']:10.3f} "
              f"{vc['predicted_vs_leverage']:13.3f} "
              f"{vc['teacher_vs_leverage']:14.3f}")
    print("B0 leverage share:", [round(v, 3) for v in leverage_share.tolist()])


if __name__ == "__main__":
    main()
