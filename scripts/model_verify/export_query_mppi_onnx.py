#!/usr/bin/env python3
"""Export and numerically validate the Query graph used by Torch MPPI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for package_dir in ("car_foundation", "car_dynamics"):
    sys.path.insert(0, str(REPO_ROOT / package_dir))
sys.path.insert(0, str(REPO_ROOT / "car_dynamics" / "car_dynamics"))

from car_foundation.query_deployment import (  # noqa: E402
    OnnxQueryRolloutBackend,
    QueryDeploymentModel,
    QueryHistoryBuffer,
    export_query_onnx,
)


DEFAULT_CHECKPOINT = REPO_ROOT / (
    "outputs/formal_real_finetune_query_baseline_split/"
    "20260728T143256/query_best.pt"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/anycar_query.onnx"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--validation-batch", type=int, default=32)
    parser.add_argument(
        "--validation-speed",
        type=float,
        default=None,
        help="Validation vx; defaults to the checkpoint context mean.",
    )
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-abs-tolerance", type=float, default=2e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.provider == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA provider requested but PyTorch CUDA is unavailable")

    model = QueryDeploymentModel.from_checkpoint(args.checkpoint.resolve(), device)
    output = export_query_onnx(
        model,
        args.output.resolve(),
        opset=args.opset,
        candidate_batch=2,
    )
    backend = OnnxQueryRolloutBackend(
        output, provider=args.provider, output_device=device
    )

    validation_speed = (
        float(model.context_mean[0])
        if args.validation_speed is None
        else args.validation_speed
    )
    initial_state = torch.tensor(
        [[0.0, 0.0, 0.0, validation_speed, float(model.context_mean[1])]],
        dtype=torch.float32,
        device=device,
    )
    current_action = model.context_mean[2:4].reshape(1, 2).detach().clone()
    history_buffer = QueryHistoryBuffer(dt=model.dt)
    history_buffer.prime_constant_motion(initial_state[0], current_action[0])
    history = history_buffer.tensor(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(3407)
    future_action = torch.randn(
        args.validation_batch, 50, 2, generator=generator, device=device
    )
    future_action[..., 0] *= 0.25
    future_action[..., 1] *= 0.35
    future_action.clamp_(-1.0, 1.0)

    with torch.inference_mode():
        torch_output = model(
            history, initial_state, current_action, future_action
        )
    onnx_output = backend(
        history, initial_state, current_action, future_action
    )
    difference = (torch_output - onnx_output).abs()
    maximum = float(difference.max())
    mean = float(difference.mean())
    if maximum > args.max_abs_tolerance:
        raise RuntimeError(
            f"PyTorch/ONNX max_abs={maximum:.6e} exceeds "
            f"tolerance={args.max_abs_tolerance:.6e}"
        )

    print(f"checkpoint: {args.checkpoint.resolve()}")
    print(f"onnx:       {output}")
    print(f"providers:  {backend.session.get_providers()}")
    print(f"batch:      {args.validation_batch}")
    print(f"speed:      {validation_speed:.6f} m/s")
    print(f"max_abs:    {maximum:.6e}")
    print(f"mean_abs:   {mean:.6e}")


if __name__ == "__main__":
    main()
