"""Residual-stream capture and storage.

Imports torch lazily — only the GPU host runs this module.
"""
from __future__ import annotations

from pathlib import Path

import torch  # noqa: E402  (only imported on GPU host)


def capture_residuals(
    model,
    input_ids: "torch.Tensor",
    target_position: int,
) -> "torch.Tensor":
    """Capture residual stream at every layer at the given token position.

    Returns a tensor of shape (n_layers, hidden_dim), bf16 on CPU.
    """
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = (
                hidden[:, target_position, :].detach().cpu().to(torch.bfloat16).squeeze(0)
            )
        return hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    try:
        with torch.no_grad():
            _ = model(input_ids)
    finally:
        for h in handles:
            h.remove()

    n_layers = len(model.model.layers)
    return torch.stack([captured[i] for i in range(n_layers)], dim=0)
    # shape: (n_layers, hidden_dim)


def save_activations(tensor: "torch.Tensor", path: str | Path) -> None:
    """Atomic save: write to .tmp then rename. Survives interrupted sessions mid-write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, tmp)
    tmp.replace(path)


def target_position_from_input_ids(input_ids: "torch.Tensor") -> int:
    """Last token of the most recent tool message: tokenize the conversation up through the tool message,
    then `len(input_ids) - 1` is the target position.
    """
    return int(input_ids.shape[-1]) - 1
