"""Activation steering. project_plan.md section 10.

Two routes for the steering direction:
- Probe-derived: take the trained logistic-regression weight vector at the
  best-performing layer for Classifier B (Level 1 vs 2), normalized.
- Contrastive (mean-difference): mean(L2 activations) - mean(L1 activations)
  at the chosen layer, normalized.

Steering hook adds α·direction to the residual stream at the chosen layer
at the last token position before chain-of-thought generation.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def probe_direction_from_weights(weights: np.ndarray | list[float]) -> np.ndarray:
    """Unit-norm probe direction. Sign convention: positive scores predict class 1
    (in our setting, Level 2 = inconsistent). Adding α·direction with α>0 should
    nudge the residual toward the L2 representation."""
    w = np.asarray(weights, dtype=np.float32).flatten()
    n = np.linalg.norm(w)
    if n == 0:
        raise ValueError("Probe weight vector is zero-norm — cannot derive direction.")
    return w / n


def contrastive_direction(
    activations: np.ndarray,
    labels: np.ndarray,
    layer_idx: int,
) -> np.ndarray:
    """Mean(class 1) - Mean(class 0) at `layer_idx`, unit-normalized.

    activations: (n_samples, n_layers, hidden_dim)
    labels: (n_samples,) binary
    """
    pos = activations[labels == 1, layer_idx, :].astype(np.float32)
    neg = activations[labels == 0, layer_idx, :].astype(np.float32)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("Need at least one example of each class for contrastive direction.")
    diff = pos.mean(axis=0) - neg.mean(axis=0)
    n = np.linalg.norm(diff)
    if n == 0:
        raise ValueError("Mean-difference vector is zero — classes have identical means.")
    return diff / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def make_steering_hook(
    direction,
    alpha: float,
    position: int = -1,
    prefill_only: bool = True,
):
    """Forward hook that adds α·direction to the residual stream at `position`.

    `direction` is a 1-D array/tensor of shape (hidden_dim,). The hook expects
    the wrapped module to return a tuple whose first element is hidden states
    of shape (batch, seq, hidden) — standard layout for a HuggingFace
    transformer block.

    `prefill_only=True` (default, matches project_plan.md §10.2): only fire on
    multi-token forward passes (prefill), not on cached single-token generation
    steps. This nudges the model's representation of the input *before* CoT
    begins, rather than persistently altering every token it generates.

    Lazy torch import — only the model-running environment needs this.
    """
    import torch

    if not isinstance(direction, torch.Tensor):
        direction_t = torch.tensor(np.asarray(direction).flatten(), dtype=torch.float32)
    else:
        direction_t = direction.flatten().to(torch.float32)

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        if is_tuple:
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = ()
        # Skip cached single-token generation steps when prefill_only.
        if prefill_only and hidden.shape[1] <= 1:
            return output
        d = direction_t.to(device=hidden.device, dtype=hidden.dtype)
        hidden = hidden.clone()
        hidden[:, position, :] = hidden[:, position, :] + alpha * d
        if is_tuple:
            return (hidden,) + rest
        return hidden

    return hook


def attach_steering_hook(
    model,
    layer_idx: int,
    direction,
    alpha: float,
    position: int = -1,
    prefill_only: bool = True,
):
    """Attach the hook and return the handle (caller is responsible for `.remove()`)."""
    hook = make_steering_hook(direction, alpha, position=position, prefill_only=prefill_only)
    return model.model.layers[layer_idx].register_forward_hook(hook)
