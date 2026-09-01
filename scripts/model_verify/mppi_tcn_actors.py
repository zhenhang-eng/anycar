#!/usr/bin/env python3
"""TCN-based no-anchor direct actor (research-only).

Architecture per the 11.70 follow-up plan:
- History [250,7] → Dilated TCN (5 residual blocks, dilation 1/2/4/8/16,
  hidden 64, kernel 3) → last-token + global-avg-pool concat → h_H [128]
- Reference [50,5] → simple Conv1D → h_R [64]
- Current [4] → passthrough
- Fusion: MLP([h_H, h_R, current]) → 256 → 16
- Output: tanh(raw) * per-dim scale + per-dim center (fitted on train labels)
- No anchor anywhere: anchor input is zeroed before the encoder.
- J16 oracle supervision target.
"""

from __future__ import annotations

import torch
from torch import nn


class TCNResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(
            channels, channels, kernel, dilation=dilation, padding=padding
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel, dilation=dilation, padding=padding
        )
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        return self.activation(residual + out)


class DilatedTCNEncoder(nn.Module):
    """Multi-scale temporal encoder for the 250-step dynamics history.

    5 residual TCN blocks with dilations 1/2/4/8/16 give a receptive field
    of ~100 steps while retaining fine-grained recent dynamics. The output
    concatenates the last-time-step token (instantaneous state) with a
    global average pool (long-term trend).
    """

    def __init__(self, input_channels: int = 7, hidden: int = 64,
                 num_blocks: int = 5, kernel: int = 3, dropout: float = 0.1,
                 output_dim: int = 128):
        super().__init__()
        self.input_proj = nn.Conv1d(input_channels, hidden, 1)
        self.blocks = nn.ModuleList([
            TCNResidualBlock(hidden, dilation=2 ** i, kernel=kernel,
                             dropout=dropout)
            for i in range(num_blocks)
        ])
        # output: last token (hidden) + global avg pool (hidden) = 2*hidden
        assert output_dim == 2 * hidden, \
            f"output_dim must be 2*hidden, got {output_dim} vs {2*hidden}"
        self.output_dim = output_dim

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: [B, T, C] → transpose to [B, C, T]
        x = history.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        # x: [B, hidden, T]
        last_token = x[:, :, -1]  # [B, hidden]
        pooled = x.mean(dim=2)  # [B, hidden]
        return torch.cat([last_token, pooled], dim=-1)  # [B, 2*hidden]


class SimpleReferenceEncoder(nn.Module):
    """Lightweight encoder for the 50-step reference trajectory."""

    def __init__(self, input_dim: int = 5, output_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(64 * 4, output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.output_dim = output_dim

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        # reference: [B, T, C] → [B, C, T]
        return self.conv(reference.transpose(1, 2))


class TCNDirectNoAnchorActor(nn.Module):
    """No-anchor direct actor with TCN history encoding.

    Combines:
    - Dilated TCN for history (multi-scale temporal dynamics)
    - Simple Conv1D for reference (task condition, only 50 steps)
    - Current state passthrough
    - MLP fusion → per-dim centered output
    - Anchor is fully blinded (zeroed).
    """

    knot_count = 8
    action_dim = 2

    def __init__(self, dropout: float = 0.1,
                 center: torch.Tensor | None = None,
                 scale: torch.Tensor | None = None,
                 hidden: int = 64, num_blocks: int = 5):
        super().__init__()
        self.history_encoder = DilatedTCNEncoder(
            input_channels=7, hidden=hidden, num_blocks=num_blocks,
            dropout=dropout, output_dim=2 * hidden,
        )
        self.reference_encoder = SimpleReferenceEncoder(
            input_dim=5, output_dim=64, dropout=dropout,
        )
        # Fusion: [h_H(2*hidden) + h_R(64) + current(4)] → 256 → 16
        fusion_input = 2 * hidden + 64 + 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 16),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        if center is not None:
            self.register_buffer(
                "out_center", center.clone().reshape(1, 8, 2)
            )
        else:
            self.register_buffer("out_center", torch.zeros(1, 8, 2))
        if scale is not None:
            self.register_buffer(
                "out_scale", scale.clone().reshape(1, 8, 2)
            )
        else:
            self.register_buffer("out_scale", torch.ones(1, 8, 2))

    def forward(self, history, reference, current, anchor_knots, feedback,
                gradient_context):
        # Blind the anchor: zero it out
        blinded = torch.zeros_like(anchor_knots)
        h_H = self.history_encoder(history)  # [B, 2*hidden]
        h_R = self.reference_encoder(reference)  # [B, 64]
        combined = torch.cat([h_H, h_R, current], dim=-1)
        raw = self.fusion(combined).reshape(-1, self.knot_count, self.action_dim)
        center = torch.tanh(raw) * self.out_scale + self.out_center
        center = torch.clamp(center, -1.0, 1.0)
        return center, center
