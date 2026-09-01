#!/usr/bin/env python3
"""Zero-training structural checks for the signed symmetric local-Q module."""

from __future__ import annotations

import json

import torch

from car_foundation.mppi_proposal_policy import (
    TorchMPPISemanticStructuredLocalQCritic,
    TorchMPPIStructuredLocalQCritic,
)


def inputs(batch: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(batch, 250, 7),
        torch.randn(batch, 50, 5),
        torch.randn(batch, 4),
        torch.randn(batch, 8, 2).clamp(-1, 1),
        torch.randn(batch, 74),
        torch.randn(batch, 32),
    )


def main() -> None:
    torch.manual_seed(260814)
    batch = 3
    source = inputs(batch)
    result = {}
    for name, enabled, rank in (
        ("H0", False, 0),
        ("D", True, 0),
        ("D_R1", True, 1),
        ("D_R2", True, 2),
    ):
        model = TorchMPPIStructuredLocalQCritic(
            low_rank=rank, dropout=0.0, hessian_enabled=enabled
        ).eval()
        if enabled:
            with torch.no_grad():
                model.diagonal_head.bias.copy_(
                    torch.linspace(-3.0, 2.0, model.flat_action_dim)
                )
                if model.low_rank_value_head is not None:
                    model.low_rank_value_head.bias.copy_(
                        torch.linspace(-4.0, 3.0, rank)
                    )
        q0, g0, hessian = model.local_parameters(*source)
        symmetry_error = float(
            torch.max(torch.abs(hessian - hessian.transpose(1, 2)))
        )
        if symmetry_error > 1e-6:
            raise AssertionError(f"{name} Hessian is not symmetric")
        action_ref = torch.randn(batch, 8, 2) * 0.1
        action = (action_ref + torch.randn(batch, 8, 2) * 0.03).requires_grad_()
        value = model(*source, action_ref, action)
        autograd = torch.autograd.grad(value.sum(), action)[0].flatten(1)
        analytic = model.local_gradient(
            g0, hessian, action - action_ref
        )
        gradient_error = float(torch.max(torch.abs(autograd - analytic)))
        if gradient_error > 1e-6:
            raise AssertionError(f"{name} analytic gradient mismatch")
        eigenvalue = torch.linalg.eigvalsh(hessian)
        if name == "H0" and float(torch.max(torch.abs(hessian))) != 0.0:
            raise AssertionError("H0 produced a nonzero Hessian")
        if enabled and not bool(torch.any(eigenvalue < 0.0)):
            raise AssertionError(f"{name} cannot express negative curvature")
        result[name] = {
            "parameter_count": model.parameter_count,
            "symmetry_max_abs_error": symmetry_error,
            "autograd_max_abs_error": gradient_error,
            "eigenvalue_minimum": float(eigenvalue.min()),
            "eigenvalue_maximum": float(eigenvalue.max()),
        }
    semantic = {}
    for name, include_feedback in (("PA", False), ("PAF", True)):
        model = TorchMPPISemanticStructuredLocalQCritic(
            include_feedback=include_feedback, low_rank=2, dropout=0.0
        ).eval()
        with torch.no_grad():
            torch.nn.init.normal_(model.gradient_head.weight, std=0.02)
        base = inputs(batch)
        feedback_changed = list(base)
        feedback_changed[4] = feedback_changed[4] + 3.0
        gradient_context_changed = list(base)
        gradient_context_changed[5] = gradient_context_changed[5] + 3.0
        action_changed = list(base)
        action_changed[3] = (action_changed[3] + 0.1).clamp(-1, 1)
        base_gradient = model.local_parameters(*base)[1]
        feedback_gradient = model.local_parameters(*feedback_changed)[1]
        context_gradient = model.local_parameters(*gradient_context_changed)[1]
        action_gradient = model.local_parameters(*action_changed)[1]
        feedback_difference = float(torch.max(torch.abs(
            base_gradient - feedback_gradient
        )))
        gradient_context_difference = float(torch.max(torch.abs(
            base_gradient - context_gradient
        )))
        action_difference = float(torch.max(torch.abs(
            base_gradient - action_gradient
        )))
        if include_feedback and feedback_difference <= 1e-7:
            raise AssertionError("PAF does not consume feedback")
        if not include_feedback and feedback_difference != 0.0:
            raise AssertionError("PA unexpectedly consumes feedback")
        if gradient_context_difference != 0.0:
            raise AssertionError(f"{name} unexpectedly consumes gradient context")
        if action_difference <= 1e-7:
            raise AssertionError(f"{name} does not consume absolute action")
        semantic[name] = {
            "feedback_difference": feedback_difference,
            "gradient_context_difference": gradient_context_difference,
            "absolute_action_difference": action_difference,
        }
    result["semantic_contract"] = semantic
    print(json.dumps({"qualification": "PASS", "arms": result}, indent=2))


if __name__ == "__main__":
    main()
