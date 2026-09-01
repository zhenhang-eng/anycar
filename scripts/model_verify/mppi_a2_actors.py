#!/usr/bin/env python3
"""A2 structural actor variants (research-only, not deployment code).

Three arms under the 11.53/11.54 contract; each keeps the frozen encoder
stack and the tanh/2-sigma/clamp center semantics of
TorchMPPIDeterministicCenterActor and changes exactly one structural axis:

- A2GeometryActor (A2-G): appends 8 per-knot relative-geometry features
  (lateral offset, heading error, next-segment heading change as curvature
  proxy, reference-speed offset, all sampled from the normalized ego
  reference at the uniform knot times) to the fused trunk feature before the
  original flat linear head.
- A2TemporalActor (A2-T): replaces the flat head with an 8-token non-causal
  two-layer self-attention decoder; each token carries the anchor knot, a
  time encoding and the population per-knot position sensitivity, plus a
  projection of the global trunk feature. Bidirectional by construction so
  late knots condition on early context (splice evidence 11.53).
- A2GeometryTemporalActor (A2-GT): the token decoder with the same
  knot-aligned geometry added to the tokens.

First round deliberately excludes early-knot bound tightening, loss
weighting, history changes, feedback removal and smoothness losses.
"""

from __future__ import annotations

import torch
from torch import nn

from car_foundation.mppi_proposal_policy import TorchMPPIDeterministicCenterActor

KNOT_TIMES = (0, 7, 14, 21, 28, 35, 42, 49)
FLAT_SENSITIVITY = (
    1.031, 2.347, 1.660, 3.992, 1.431, 3.455, 1.120, 2.858,
    0.816, 2.190, 0.543, 1.478, 0.271, 0.708, 0.051, 0.121,
)


def knot_geometry(reference: torch.Tensor) -> torch.Tensor:
    """[B,50,5] normalized ego reference -> [B,8,4] per-knot geometry."""
    samples = reference[:, KNOT_TIMES, :]
    lateral = samples[..., 1]
    heading = torch.atan2(samples[..., 2], samples[..., 3])
    speed = samples[..., 4]
    yaw = torch.atan2(reference[..., 2], reference[..., 3])
    following = torch.clamp(
        torch.tensor(list(KNOT_TIMES[1:]) + [KNOT_TIMES[-1]]),
        0, reference.shape[1] - 1,
    ).to(reference.device)
    heading_change = yaw[:, following] - yaw[:, KNOT_TIMES]
    heading_change = torch.atan2(
        torch.sin(heading_change), torch.cos(heading_change)
    )
    return torch.stack((lateral, heading, heading_change, speed), dim=-1)


def knot_time_encoding() -> torch.Tensor:
    times = torch.tensor(KNOT_TIMES, dtype=torch.float32) / 49.0
    return torch.stack((times, 1.0 - times), dim=-1)


def knot_sensitivity() -> torch.Tensor:
    values = torch.tensor(FLAT_SENSITIVITY).reshape(8, 2).sum(dim=1)
    return values / values.max()


class A2GeometryActor(TorchMPPIDeterministicCenterActor):
    def __init__(self, maximum_delta_sigma: float = 2.0, dropout: float = 0.05):
        super().__init__(maximum_delta_sigma=maximum_delta_sigma, dropout=dropout)
        self.action_head = nn.Linear(self.encoder.output_dim + 32, 16)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, history, reference, current, anchor_knots, feedback,
                gradient_context):
        feature = self.encoder(
            history, reference, current, anchor_knots, feedback, gradient_context
        )
        geometry = knot_geometry(reference).reshape(reference.shape[0], -1)
        raw = self.action_head(torch.cat((feature, geometry), dim=-1))
        raw = raw.reshape(-1, self.knot_count, self.action_dim)
        return self.center_from_action(anchor_knots, torch.tanh(raw))


class _TokenDecoder(nn.Module):
    token_dim = 64

    def __init__(self, dropout: float, use_geometry: bool) -> None:
        super().__init__()
        token_static = 2 + 2 + 1 + (4 if use_geometry else 0)
        self.static_proj = nn.Linear(token_static, self.token_dim)
        self.feature_proj = nn.Linear(192, self.token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim, nhead=4, dim_feedforward=128,
            dropout=dropout, batch_first=True,
        )
        self.encoder_layers = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(self.token_dim, 2)
        # Zero-init per-knot trunk skip (192->16 reshaped to [B,8,2]): the
        # decoder starts exactly at the flat-head function class with 16
        # independent weights; the token structure can only add capacity.
        # Without it the token arm underfits within the shared 120 epoch
        # budget and the structure comparison is confounded.
        self.skip = nn.Linear(192, 16)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)
        self.use_geometry = use_geometry
        self.register_buffer("time_encoding", knot_time_encoding())
        self.register_buffer("knot_sensitivity", knot_sensitivity())

    def forward(self, feature: torch.Tensor, anchor_knots: torch.Tensor,
                reference: torch.Tensor) -> torch.Tensor:
        batch = feature.shape[0]
        times = self.time_encoding[None].expand(batch, -1, -1)
        sensitivity = self.knot_sensitivity[None, :, None].expand(
            batch, -1, -1
        )
        pieces = [anchor_knots, times, sensitivity]
        if self.use_geometry:
            pieces.append(knot_geometry(reference))
        static = torch.cat(pieces, dim=-1)
        tokens = self.static_proj(static) + self.feature_proj(feature)[:, None]
        decoded = self.encoder_layers(tokens)
        skip = self.skip(feature).reshape(batch, 8, 2)
        return self.head(decoded) + skip


class A2TemporalActor(TorchMPPIDeterministicCenterActor):
    def __init__(self, maximum_delta_sigma: float = 2.0, dropout: float = 0.05):
        super().__init__(maximum_delta_sigma=maximum_delta_sigma, dropout=dropout)
        self.action_head = nn.Identity()
        self.decoder = _TokenDecoder(dropout, use_geometry=False)

    def forward(self, history, reference, current, anchor_knots, feedback,
                gradient_context):
        feature = self.encoder(
            history, reference, current, anchor_knots, feedback, gradient_context
        )
        raw = self.decoder(feature, anchor_knots, reference)
        return self.center_from_action(anchor_knots, torch.tanh(raw))


class A2GeometryTemporalActor(TorchMPPIDeterministicCenterActor):
    def __init__(self, maximum_delta_sigma: float = 2.0, dropout: float = 0.05):
        super().__init__(maximum_delta_sigma=maximum_delta_sigma, dropout=dropout)
        self.action_head = nn.Identity()
        self.decoder = _TokenDecoder(dropout, use_geometry=True)

    def forward(self, history, reference, current, anchor_knots, feedback,
                gradient_context):
        feature = self.encoder(
            history, reference, current, anchor_knots, feedback, gradient_context
        )
        raw = self.decoder(feature, anchor_knots, reference)
        return self.center_from_action(anchor_knots, torch.tanh(raw))


class DirectActionActor(TorchMPPIDeterministicCenterActor):
    """Predicts the full action a* directly (no anchor added to the output).

    Research-only arm for the direct-vs-residual target contrast (review
    11.61). The anchor a0 remains an encoder INPUT (context), but the output
    is tanh(head) predicting a* directly instead of anchor + bounded
    correction. The head therefore learns the full action (norm ~1.48) rather
    than the small residual (norm ~0.205).
    """

    def forward(self, history, reference, current, anchor_knots, feedback,
                gradient_context):
        feature = self.encoder(
            history, reference, current, anchor_knots, feedback,
            gradient_context,
        )
        raw = self.action_head(feature).reshape(
            -1, self.knot_count, self.action_dim
        )
        center = torch.tanh(raw)
        return center, center


class DirectNoAnchorActor(TorchMPPIDeterministicCenterActor):
    """Direct full-action prediction with the anchor BLINDED from the encoder.

    The anchor input slot is zeroed before entering the encoder so the
    network receives no information about the warm start / previous center.
    The output is tanh(head) + per-dim affine centering fitted on the
    training labels (population statistics of a*), and the head is a
    two-layer MLP for extra capacity. No anchor anywhere: not in the output,
    not in the input.
    """

    def __init__(self, maximum_delta_sigma: float = 2.0, dropout: float = 0.05,
                 center: torch.Tensor | None = None,
                 scale: torch.Tensor | None = None):
        super().__init__(maximum_delta_sigma=maximum_delta_sigma, dropout=dropout)
        self.action_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 256), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 16),
        )
        nn.init.zeros_(self.action_head[-1].weight)
        nn.init.zeros_(self.action_head[-1].bias)
        if center is not None:
            self.register_buffer("out_center", center.clone().reshape(1, 8, 2))
        else:
            self.register_buffer("out_center", torch.zeros(1, 8, 2))
        if scale is not None:
            self.register_buffer("out_scale", scale.clone().reshape(1, 8, 2))
        else:
            self.register_buffer("out_scale", torch.ones(1, 8, 2))

    def forward(self, history, reference, current, anchor_knots, feedback,
                gradient_context):
        blinded = torch.zeros_like(anchor_knots)
        feature = self.encoder(
            history, reference, current, blinded, feedback, gradient_context,
        )
        raw = self.action_head(feature).reshape(
            -1, self.knot_count, self.action_dim
        )
        center = torch.tanh(raw) * self.out_scale + self.out_center
        center = torch.clamp(center, -1.0, 1.0)
        return center, center


class _NoAnchorTokenDecoder(nn.Module):
    """Temporal token decoder that uses ZERO ANCHOR tokens.

    Replaces the per-knot anchor tokens with zero vectors so the decoder
    gets no warm-start information. Geometry tokens from the reference
    trajectory provide the per-knot contextual signal instead.
    """
    token_dim = 64

    def __init__(
        self,
        dropout: float,
        use_geometry: bool,
        include_longitudinal: bool = False,
        local_pose_dim: int = 0,
    ) -> None:
        super().__init__()
        token_static = (
            2 + 2 + 1 + (4 if use_geometry else 0)
            + (1 if include_longitudinal else 0) + local_pose_dim
        )
        self.static_proj = nn.Linear(token_static, self.token_dim)
        self.feature_proj = nn.Linear(192, self.token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim, nhead=4, dim_feedforward=128,
            dropout=dropout, batch_first=True,
        )
        self.encoder_layers = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(self.token_dim, 2)
        self.skip = nn.Linear(192, 16)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)
        self.use_geometry = use_geometry
        self.include_longitudinal = include_longitudinal
        self.local_pose_dim = local_pose_dim
        self.register_buffer("time_encoding", knot_time_encoding())
        self.register_buffer("knot_sensitivity", knot_sensitivity())

    def forward_with_tokens(
        self,
        feature: torch.Tensor,
        reference: torch.Tensor,
        local_pose: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = feature.shape[0]
        times = self.time_encoding[None].expand(batch, -1, -1)
        sensitivity = self.knot_sensitivity[None, :, None].expand(
            batch, -1, -1
        )
        # Zero anchor tokens: no warm-start info.
        zero_anchor = torch.zeros(
            batch, 8, 2, device=feature.device, dtype=feature.dtype
        )
        pieces = [zero_anchor, times, sensitivity]
        if self.use_geometry:
            pieces.append(knot_geometry(reference))
        if self.include_longitudinal:
            pieces.append(reference[:, KNOT_TIMES, 0:1])
        if self.local_pose_dim:
            if local_pose is None or local_pose.shape[1:] != (self.local_pose_dim,):
                raise ValueError(
                    f"local_pose must have shape [B,{self.local_pose_dim}]"
                )
            pieces.append(local_pose[:, None, :].expand(-1, 8, -1))
        static = torch.cat(pieces, dim=-1)
        tokens = self.static_proj(static) + self.feature_proj(feature)[:, None]
        decoded = self.encoder_layers(tokens)
        skip = self.skip(feature).reshape(batch, 8, 2)
        return self.head(decoded) + skip, decoded

    def forward(
        self,
        feature: torch.Tensor,
        reference: torch.Tensor,
        local_pose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw, _ = self.forward_with_tokens(feature, reference, local_pose)
        return raw


class _NoAttentionKnotDecoder(nn.Module):
    """Shared per-knot decoder WITHOUT self-attention between knots.

    Each knot's output is produced independently by the SAME decoder applied
    to (global_feature, time_encoding, per_knot_geometry). No inter-knot
    attention: the "same control law at different time points" prior is
    enforced by weight sharing rather than learned interaction.
    """
    def __init__(self, dropout: float) -> None:
        super().__init__()
        # Input: feature(192) + time_encoding(2) + sensitivity(1) + geometry(4) = 199
        input_dim = 192 + 2 + 1 + 4
        hidden = 128
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        self.skip = nn.Linear(192, 16)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)
        self.register_buffer("time_encoding", knot_time_encoding())
        self.register_buffer("knot_sensitivity", knot_sensitivity())

    def forward(self, feature: torch.Tensor,
                reference: torch.Tensor) -> torch.Tensor:
        batch = feature.shape[0]
        geom = knot_geometry(reference)  # [B, 8, 4]
        times = self.time_encoding[None].expand(batch, -1, -1)  # [B, 8, 2]
        sens = self.knot_sensitivity[None, :, None].expand(batch, -1, -1)  # [B, 8, 1]
        # Broadcast feature to each knot: [B, 192] → [B, 8, 192]
        feat_expanded = feature[:, None, :].expand(-1, 8, -1)
        # Concatenate: [B, 8, 199]
        per_knot_input = torch.cat([feat_expanded, times, sens, geom], dim=-1)
        # Apply shared decoder independently per knot
        decoded = self.decoder(per_knot_input)  # [B, 8, 2]
        skip = self.skip(feature).reshape(batch, 8, 2)
        return decoded + skip


class DirectNoAnchorGTActor(TorchMPPIDeterministicCenterActor):
    """No-anchor direct + knot-aligned geometry tokens + temporal decoder.

    Combines three improvements in one arm:
    - Anchor is fully blinded (zeroed in both encoder and token decoder).
    - Knot-aligned geometry tokens from the reference trajectory provide
      per-knot contextual information (lateral offset, heading error,
      curvature, speed offset at each knot time).
    - Two-layer non-causal self-attention temporal decoder with per-knot
      tokens allows late knots to condition on early-knot context.
    - Per-dim affine centering on the output for target whitening.
    """

    def __init__(self, maximum_delta_sigma: float = 2.0, dropout: float = 0.05,
                 center: torch.Tensor | None = None,
                 scale: torch.Tensor | None = None):
        super().__init__(maximum_delta_sigma=maximum_delta_sigma, dropout=dropout)
        self.action_head = nn.Identity()
        self.decoder = _NoAnchorTokenDecoder(dropout, use_geometry=True)
        if center is not None:
            self.register_buffer("out_center", center.clone().reshape(1, 8, 2))
        else:
            self.register_buffer("out_center", torch.zeros(1, 8, 2))
        if scale is not None:
            self.register_buffer("out_scale", scale.clone().reshape(1, 8, 2))
        else:
            self.register_buffer("out_scale", torch.ones(1, 8, 2))

    def forward(self, history, reference, current, anchor_knots, feedback,
                gradient_context):
        blinded = torch.zeros_like(anchor_knots)
        feature = self.encoder(
            history, reference, current, blinded, feedback, gradient_context,
        )
        raw = self.decoder(feature, reference)
        center = torch.tanh(raw) * self.out_scale + self.out_center
        center = torch.clamp(center, -1.0, 1.0)
        return center, center


class DirectNoAnchorGTCleanActor(DirectNoAnchorGTActor):
    """Strict one-shot input-contract arm for the no-anchor GT actor.

    In addition to blinding ``anchor_knots``, this arm blinds the 74-D
    first-pass feedback and 32-D gradient context.  Consequently the output
    can depend only on history, ego-frame reference, and current state.  The
    zero tensors deliberately retain the exact encoder/decoder parameter
    count of ``DirectNoAnchorGTActor`` so this is an information-only
    ablation; a deployment refactor may remove the now-constant branches
    after the comparison is complete.
    """

    def forward(self, history, reference, current, anchor_knots, feedback,
                gradient_context):
        blinded_anchor = torch.zeros_like(anchor_knots)
        blinded_feedback = torch.zeros_like(feedback)
        blinded_gradient = torch.zeros_like(gradient_context)
        feature = self.encoder(
            history, reference, current, blinded_anchor, blinded_feedback,
            blinded_gradient,
        )
        raw = self.decoder(feature, reference)
        center = torch.tanh(raw) * self.out_scale + self.out_center
        center = torch.clamp(center, -1.0, 1.0)
        return center, center


class _DirectNoAnchorGTLocalGeometryActor(DirectNoAnchorGTActor):
    """Clean no-anchor GT with optional deployable local-geometry inputs."""

    include_longitudinal = False
    include_frenet = False

    def __init__(self, maximum_delta_sigma: float = 2.0, dropout: float = 0.05,
                 center: torch.Tensor | None = None,
                 scale: torch.Tensor | None = None):
        super().__init__(
            maximum_delta_sigma=maximum_delta_sigma,
            dropout=dropout,
            center=center,
            scale=scale,
        )
        self.decoder = _NoAnchorTokenDecoder(
            dropout,
            use_geometry=True,
            include_longitudinal=self.include_longitudinal,
            local_pose_dim=3 if self.include_frenet else 0,
        )

    def forward(self, history, reference, current, anchor_knots, feedback,
                gradient_context):
        blinded_anchor = torch.zeros_like(anchor_knots)
        blinded_feedback = torch.zeros_like(feedback)
        blinded_gradient = torch.zeros_like(gradient_context)
        feature = self.encoder(
            history, reference, current, blinded_anchor, blinded_feedback,
            blinded_gradient,
        )
        # The runner uses the otherwise-forbidden gradient tensor only as a
        # fixed-shape carrier for [frenet_t, sin(frenet_xi), cos(frenet_xi)].
        # No first-pass gradient value reaches either encoder or decoder.
        local_pose = gradient_context[:, :3] if self.include_frenet else None
        raw = self.decoder(feature, reference, local_pose)
        center = torch.tanh(raw) * self.out_scale + self.out_center
        center = torch.clamp(center, -1.0, 1.0)
        return center, center


class DirectNoAnchorGTXActor(_DirectNoAnchorGTLocalGeometryActor):
    """G-X: add per-knot ego-frame longitudinal reference position."""

    include_longitudinal = True


OAC_DEFAULT_OUTPUT_SUPPORT_MULTIPLIER = 3.0


class DirectNoAnchorGTXSupportActor(DirectNoAnchorGTXActor):
    """Paired output-support arm with an exactly preserved initial action.

    ``support_multiplier=1`` and ``3`` share the same parameterization and a
    zero-initialized pre-squash adapter.  At initialization both reproduce the
    source G-X Actor, while the latter can subsequently reach center +/-3std.
    """

    def __init__(
        self,
        support_multiplier: float = OAC_DEFAULT_OUTPUT_SUPPORT_MULTIPLIER,
        maximum_delta_sigma: float = 2.0,
        dropout: float = 0.05,
        center: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> None:
        if support_multiplier < 1.0:
            raise ValueError("support_multiplier must be at least one")
        super().__init__(
            maximum_delta_sigma=maximum_delta_sigma,
            dropout=dropout,
            center=center,
            scale=scale,
        )
        self.register_buffer(
            "output_support_multiplier",
            torch.tensor(float(support_multiplier), dtype=torch.float32),
        )
        self.support_adapter_head = nn.Linear(64, 2)
        self.support_adapter_skip = nn.Linear(192, 16)
        nn.init.zeros_(self.support_adapter_head.weight)
        nn.init.zeros_(self.support_adapter_head.bias)
        nn.init.zeros_(self.support_adapter_skip.weight)
        nn.init.zeros_(self.support_adapter_skip.bias)

    def forward(self, history, reference, current, anchor_knots, feedback,
                gradient_context):
        blinded_anchor = torch.zeros_like(anchor_knots)
        blinded_feedback = torch.zeros_like(feedback)
        blinded_gradient = torch.zeros_like(gradient_context)
        feature = self.encoder(
            history, reference, current, blinded_anchor, blinded_feedback,
            blinded_gradient,
        )
        raw, decoded = self.decoder.forward_with_tokens(feature, reference)
        multiplier = self.output_support_multiplier
        base_unit = torch.tanh(raw)
        origin = torch.atanh(torch.clamp(
            base_unit / multiplier, -0.999999, 0.999999
        ))
        delta = (
            self.support_adapter_head(decoded)
            + self.support_adapter_skip(feature).reshape(-1, 8, 2)
        )
        center = (
            multiplier * self.out_scale * torch.tanh(origin + delta)
            + self.out_center
        )
        center = torch.clamp(center, -1.0, 1.0)
        return center, center


class DirectNoAnchorGTFrenetActor(_DirectNoAnchorGTLocalGeometryActor):
    """G-F: add current [frenet_t, sin(xi), cos(xi)] to every knot."""

    include_frenet = True


class DirectNoAnchorGTXFActor(_DirectNoAnchorGTLocalGeometryActor):
    """G-XF: combine longitudinal reference and current Frenet geometry."""

    include_longitudinal = True
    include_frenet = True


class _NoAnchorReferenceCrossDecoder(_NoAnchorTokenDecoder):
    """G-X decoder whose control queries directly attend to 50 reference tokens."""

    def __init__(self, dropout: float, include_history_tokens: bool = False):
        super().__init__(
            dropout,
            use_geometry=True,
            include_longitudinal=True,
        )
        self.reference_token_proj = nn.Sequential(
            nn.Linear(7, self.token_dim),
            nn.SiLU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.reference_cross_attention = nn.MultiheadAttention(
            self.token_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.reference_cross_norm = nn.LayerNorm(self.token_dim)
        self.reference_cross_dropout = nn.Dropout(dropout)
        self.include_history_tokens = include_history_tokens
        if include_history_tokens:
            self.history_token_proj = nn.Linear(96, self.token_dim)
            self.history_cross_attention = nn.MultiheadAttention(
                self.token_dim, num_heads=4, dropout=dropout, batch_first=True
            )
            self.history_cross_norm = nn.LayerNorm(self.token_dim)
            self.history_cross_dropout = nn.Dropout(dropout)
        ref_time = torch.linspace(0.0, 1.0, 50)
        self.register_buffer(
            "reference_time_encoding",
            torch.stack((ref_time, 1.0 - ref_time), dim=-1),
        )

    def forward(
        self,
        feature: torch.Tensor,
        reference: torch.Tensor,
        history_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = feature.shape[0]
        times = self.time_encoding[None].expand(batch, -1, -1)
        sensitivity = self.knot_sensitivity[None, :, None].expand(
            batch, -1, -1
        )
        zero_anchor = torch.zeros(
            batch, 8, 2, device=feature.device, dtype=feature.dtype
        )
        static = torch.cat(
            (
                zero_anchor,
                times,
                sensitivity,
                knot_geometry(reference),
                reference[:, KNOT_TIMES, 0:1],
            ),
            dim=-1,
        )
        tokens = self.static_proj(static) + self.feature_proj(feature)[:, None]

        reference_time = self.reference_time_encoding[None].expand(
            batch, -1, -1
        ).to(dtype=reference.dtype)
        reference_tokens = self.reference_token_proj(
            torch.cat((reference, reference_time), dim=-1)
        )
        reference_update, _ = self.reference_cross_attention(
            tokens, reference_tokens, reference_tokens, need_weights=False
        )
        tokens = self.reference_cross_norm(
            tokens + self.reference_cross_dropout(reference_update)
        )

        if self.include_history_tokens:
            if history_tokens is None or history_tokens.shape[1:] != (8, 96):
                raise ValueError("history_tokens must have shape [B,8,96]")
            projected_history = self.history_token_proj(history_tokens)
            history_update, _ = self.history_cross_attention(
                tokens, projected_history, projected_history, need_weights=False
            )
            tokens = self.history_cross_norm(
                tokens + self.history_cross_dropout(history_update)
            )

        decoded = self.encoder_layers(tokens)
        skip = self.skip(feature).reshape(batch, 8, 2)
        return self.head(decoded) + skip


class DirectNoAnchorGTXReferenceCrossActor(DirectNoAnchorGTXActor):
    """G-X-R: add direct control-query cross-attention over reference tokens."""

    def __init__(self, maximum_delta_sigma: float = 2.0, dropout: float = 0.05,
                 center: torch.Tensor | None = None,
                 scale: torch.Tensor | None = None):
        super().__init__(
            maximum_delta_sigma=maximum_delta_sigma,
            dropout=dropout,
            center=center,
            scale=scale,
        )
        self.decoder = _NoAnchorReferenceCrossDecoder(dropout)

    def forward(self, history, reference, current, anchor_knots, feedback,
                gradient_context):
        zeros_anchor = torch.zeros_like(anchor_knots)
        zeros_feedback = torch.zeros_like(feedback)
        zeros_gradient = torch.zeros_like(gradient_context)
        feature = self.encoder(
            history, reference, current, zeros_anchor, zeros_feedback,
            zeros_gradient,
        )
        raw = self.decoder(feature, reference)
        center = torch.tanh(raw) * self.out_scale + self.out_center
        center = torch.clamp(center, -1.0, 1.0)
        return center, center


class DirectNoAnchorGTXReferenceHistoryCrossActor(
    DirectNoAnchorGTXReferenceCrossActor
):
    """G-X-RH: additionally expose eight pooled history tokens to each knot."""

    def __init__(self, maximum_delta_sigma: float = 2.0, dropout: float = 0.05,
                 center: torch.Tensor | None = None,
                 scale: torch.Tensor | None = None):
        super().__init__(
            maximum_delta_sigma=maximum_delta_sigma,
            dropout=dropout,
            center=center,
            scale=scale,
        )
        self.decoder = _NoAnchorReferenceCrossDecoder(
            dropout, include_history_tokens=True
        )

    def forward(self, history, reference, current, anchor_knots, feedback,
                gradient_context):
        zeros_anchor = torch.zeros_like(anchor_knots)
        zeros_feedback = torch.zeros_like(feedback)
        zeros_gradient = torch.zeros_like(gradient_context)
        feature = self.encoder(
            history, reference, current, zeros_anchor, zeros_feedback,
            zeros_gradient,
        )
        history_tokens = self.encoder.history_encoder.convolution(
            history.transpose(1, 2)
        ).transpose(1, 2)
        raw = self.decoder(feature, reference, history_tokens)
        center = torch.tanh(raw) * self.out_scale + self.out_center
        center = torch.clamp(center, -1.0, 1.0)
        return center, center


class GTCurrentSkipActor(TorchMPPIDeterministicCenterActor):
    """GT no-anchor with a Current skip connection to the output.

    Single-variable ablation on top of DirectNoAnchorGTActor: identical in
    every way except a zero-initialized Linear(4→16) skip from the raw
    current state directly to the output action, bypassing the entire
    encoder/fusion/trunk/decoder pipeline. At initialization the model is
    exactly the baseline; during training the skip learns what part of the
    optimal action is a direct function of the current state.
    """

    def __init__(self, maximum_delta_sigma: float = 2.0, dropout: float = 0.05,
                 center: torch.Tensor | None = None,
                 scale: torch.Tensor | None = None):
        super().__init__(maximum_delta_sigma=maximum_delta_sigma, dropout=dropout)
        self.action_head = nn.Identity()
        self.decoder = _NoAnchorTokenDecoder(dropout, use_geometry=True)
        # Zero-init skip: current(4) → raw pre-tanh output(16)
        self.current_skip = nn.Linear(4, 16)
        nn.init.zeros_(self.current_skip.weight)
        nn.init.zeros_(self.current_skip.bias)
        if center is not None:
            self.register_buffer("out_center", center.clone().reshape(1, 8, 2))
        else:
            self.register_buffer("out_center", torch.zeros(1, 8, 2))
        if scale is not None:
            self.register_buffer("out_scale", scale.clone().reshape(1, 8, 2))
        else:
            self.register_buffer("out_scale", torch.ones(1, 8, 2))

    def forward(self, history, reference, current, anchor_knots, feedback,
                gradient_context):
        blinded = torch.zeros_like(anchor_knots)
        feature = self.encoder(
            history, reference, current, blinded, feedback, gradient_context,
        )
        raw = self.decoder(feature, reference)  # [B, 8, 2]
        # Current skip: raw current → 16-dim, reshape to [B, 8, 2]
        skip = self.current_skip(current).reshape(-1, 8, 2)
        combined_raw = raw.reshape(-1, 16) + skip.reshape(-1, 16)
        combined_raw = combined_raw.reshape(-1, 8, 2)
        center = torch.tanh(combined_raw) * self.out_scale + self.out_center
        center = torch.clamp(center, -1.0, 1.0)
        return center, center


class GTNoAttentionActor(TorchMPPIDeterministicCenterActor):
    """GT no-anchor with shared per-knot decoder, NO self-attention.

    Single-variable ablation on top of DirectNoAnchorGTActor: identical
    encoder/fusion/no-anchor/centering, but the token decoder's 2-layer
    self-attention is replaced by an independent shared per-knot MLP applied
    to (feature, time, sensitivity, geometry). Each knot's output is produced
    independently by weight sharing ("same control law at different times"),
    with no inter-knot attention.
    """

    def __init__(self, maximum_delta_sigma: float = 2.0, dropout: float = 0.05,
                 center: torch.Tensor | None = None,
                 scale: torch.Tensor | None = None):
        super().__init__(maximum_delta_sigma=maximum_delta_sigma, dropout=dropout)
        self.action_head = nn.Identity()
        self.decoder = _NoAttentionKnotDecoder(dropout)
        if center is not None:
            self.register_buffer("out_center", center.clone().reshape(1, 8, 2))
        else:
            self.register_buffer("out_center", torch.zeros(1, 8, 2))
        if scale is not None:
            self.register_buffer("out_scale", scale.clone().reshape(1, 8, 2))
        else:
            self.register_buffer("out_scale", torch.ones(1, 8, 2))

    def forward(self, history, reference, current, anchor_knots, feedback,
                gradient_context):
        blinded = torch.zeros_like(anchor_knots)
        feature = self.encoder(
            history, reference, current, blinded, feedback, gradient_context,
        )
        raw = self.decoder(feature, reference)
        center = torch.tanh(raw) * self.out_scale + self.out_center
        center = torch.clamp(center, -1.0, 1.0)
        return center, center


class TCNHistoryGTActor(TorchMPPIDeterministicCenterActor):
    """Single-variable ablation: GT architecture with TCN history encoder.

    Identical to DirectNoAnchorGTActor (no-anchor, geometry tokens, temporal
    token decoder, per-dim centering) EXCEPT the history encoder is replaced
    from the original TemporalConvEncoder to a DilatedTCNEncoder. This
    isolates the effect of the history encoder change.
    """

    def __init__(self, maximum_delta_sigma: float = 2.0, dropout: float = 0.05,
                 center: torch.Tensor | None = None,
                 scale: torch.Tensor | None = None):
        super().__init__(maximum_delta_sigma=maximum_delta_sigma, dropout=dropout)
        self.action_head = nn.Identity()
        self.decoder = _NoAnchorTokenDecoder(dropout, use_geometry=True)
        # Replace the history encoder with TCN
        from mppi_tcn_actors import DilatedTCNEncoder
        self.encoder.history_encoder = DilatedTCNEncoder(
            input_channels=7, hidden=64, num_blocks=5, kernel=3,
            dropout=dropout, output_dim=128,
        )
        if center is not None:
            self.register_buffer("out_center", center.clone().reshape(1, 8, 2))
        else:
            self.register_buffer("out_center", torch.zeros(1, 8, 2))
        if scale is not None:
            self.register_buffer("out_scale", scale.clone().reshape(1, 8, 2))
        else:
            self.register_buffer("out_scale", torch.ones(1, 8, 2))

    def forward(self, history, reference, current, anchor_knots, feedback,
                gradient_context):
        blinded = torch.zeros_like(anchor_knots)
        # TCN for history, original encoders for the rest (including the
        # blinded anchor's 64-dim slot to match the fusion input dim).
        h_hist = self.encoder.history_encoder(history)  # [B, 128]
        h_ref = self.encoder.reference_encoder(reference)  # [B, 128]
        h_cur = self.encoder.current_encoder(current)  # [B, 64]
        h_anc = self.encoder.anchor_encoder(blinded.flatten(1))  # [B, 64]
        h_fb = self.encoder.feedback_encoder(feedback)  # [B, 128]
        h_grad = self.encoder.gradient_encoder(gradient_context)  # [B, 64]
        feature = self.encoder.fusion(torch.cat(
            [h_hist, h_ref, h_cur, h_anc, h_fb, h_grad], dim=-1
        ))
        raw = self.decoder(feature, reference)
        center = torch.tanh(raw) * self.out_scale + self.out_center
        center = torch.clamp(center, -1.0, 1.0)
        return center, center


ARCHITECTURES = {
    "base": TorchMPPIDeterministicCenterActor,
    "g": A2GeometryActor,
    "t": A2TemporalActor,
    "gt": A2GeometryTemporalActor,
    "direct": DirectActionActor,
    "direct_noanchor": DirectNoAnchorActor,
    "direct_noanchor_gt": DirectNoAnchorGTActor,
    "direct_noanchor_gt_clean": DirectNoAnchorGTCleanActor,
    "direct_noanchor_gt_x": DirectNoAnchorGTXActor,
    "direct_noanchor_gt_frenet": DirectNoAnchorGTFrenetActor,
    "direct_noanchor_gt_xf": DirectNoAnchorGTXFActor,
    "direct_noanchor_gt_x_refcross": DirectNoAnchorGTXReferenceCrossActor,
    "direct_noanchor_gt_x_refhistcross": (
        DirectNoAnchorGTXReferenceHistoryCrossActor
    ),
    "gt_current_skip": GTCurrentSkipActor,
    "gt_no_attention": GTNoAttentionActor,
    "tcn_gt": TCNHistoryGTActor,
}
