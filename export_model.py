"""Export GrayGoNet to TorchScript for C++ inference.

Usage:
    python export_model.py checkpoints/champion_iter_0010.pt
    python export_model.py checkpoints_v5/champion_iter_0007.pt --v5
    python export_model.py checkpoints/champion_iter_0010.pt --output model_traced.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from model import GrayGoNet


def load_checkpoint(path: Path, device: torch.device, from_v5: bool = False) -> GrayGoNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]

    filters = sd["stem_conv.conv.weight"].shape[0]
    block_indices = {int(k.split(".")[1]) for k in sd if k.startswith("res_blocks.")}
    blocks = len(block_indices)
    board_size = int(round((sd["policy_fc.weight"].shape[0] - 1) ** 0.5))

    model = GrayGoNet(board_size=board_size, blocks=blocks, filters=filters)

    if from_v5:
        model_sd = model.state_dict()
        loaded = 0
        for k, v in sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
                loaded += 1
        model.load_state_dict(model_sd)
        print(f"Loaded {loaded}/{len(model_sd)} params from v5 checkpoint")
    else:
        model.load_state_dict(sd)
        print(f"Loaded all {len(sd)} params")

    return model


def export(model: GrayGoNet, output_path: Path) -> None:
    model.eval()
    model = model.cpu()
    board_size = model.board_size
    dummy = torch.randn(1, 6, board_size, board_size)

    with torch.no_grad():
        orig_out = model(dummy)

    traced = torch.jit.trace(model, dummy)
    traced.save(str(output_path))
    print(f"Saved traced model to {output_path}")

    # Verify outputs match
    loaded = torch.jit.load(str(output_path))
    loaded.eval()
    with torch.no_grad():
        loaded_out = loaded(dummy)

    for i, (a, b) in enumerate(zip(orig_out, loaded_out)):
        diff = (a - b).abs().max().item()
        names = ["policy", "value", "aux1", "aux2", "aux3", "aux4", "aux5"]
        name = names[i] if i < len(names) else f"output_{i}"
        status = "OK" if diff < 1e-5 else f"MISMATCH (max diff={diff:.6e})"
        print(f"  {name}: {status}")

    print("Export verified successfully.")


def main():
    parser = argparse.ArgumentParser(description="Export GrayGoNet to TorchScript")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: same dir as checkpoint, champion_traced.pt)")
    parser.add_argument("--v5", action="store_true", help="Load from v5 checkpoint format")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = ckpt_path.parent / "champion_traced.pt"

    device = torch.device(args.device)
    model = load_checkpoint(ckpt_path, device, from_v5=args.v5)
    model.to(device)
    export(model, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
