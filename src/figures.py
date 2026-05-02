"""Plotting. project_plan.md sections 8.2 and 9.

Matplotlib only. All inputs are small numpy arrays / dicts; no model or torch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def probe_auc_by_layer(
    layer_aucs_by_classifier: dict[str, list[float] | list[list[float]]],
    out_path: str | Path,
    title: str = "Probe AUC by layer",
) -> None:
    """Plot per-layer AUC for each classifier.

    Each value can be either a list (single seed) or a list-of-lists (one per
    seed) — in which case we plot mean and a shaded ±1 std band.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, vals in layer_aucs_by_classifier.items():
        arr = np.asarray(vals, dtype=np.float64)
        if arr.ndim == 1:
            ax.plot(arr, label=name, linewidth=2)
        else:  # (n_seeds, n_layers)
            mean = arr.mean(axis=0)
            sd = arr.std(axis=0)
            xs = np.arange(arr.shape[1])
            ax.plot(xs, mean, label=name, linewidth=2)
            ax.fill_between(xs, mean - sd, mean + sd, alpha=0.2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("5-fold CV AUC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def three_curve_plot(
    levels: list[int],
    verb_rate: list[float] | list[list[float]],
    rej_rate: list[float] | list[list[float]],
    probe_auc_l0_l2: float | list[float] | None,
    probe_auc_l1_l2: float | list[float] | None,
    out_path: str | Path,
    title: str = "Faithfulness signals vs perturbation level",
) -> None:
    """Headline figure (project_plan.md §9).

    verb_rate and rej_rate are per-level rates (length == len(levels)). Each
    can be a list (single seed) or list-of-lists (per-seed, mean ± std).

    Probe AUCs are scalar or per-seed lists. They render as horizontal markers
    because the underlying probes are binary classifiers, not per-level rates.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    def _plot_curve(vals, label, marker):
        arr = np.asarray(vals, dtype=np.float64)
        if arr.ndim == 1:
            ax.plot(levels, arr, marker=marker, label=label, linewidth=2)
        else:  # (n_seeds, n_levels)
            mean = arr.mean(axis=0)
            sd = arr.std(axis=0)
            ax.errorbar(levels, mean, yerr=sd, marker=marker, label=label,
                        linewidth=2, capsize=4)

    _plot_curve(verb_rate, "CoT verbalization rate", "o")
    _plot_curve(rej_rate, "Action-rejection rate", "s")

    def _mean_sd(v):
        if v is None:
            return None, None
        a = np.asarray(v, dtype=np.float64).flatten()
        return float(a.mean()), (float(a.std()) if a.size > 1 else 0.0)

    auc_a_mean, auc_a_sd = _mean_sd(probe_auc_l0_l2)
    auc_b_mean, auc_b_sd = _mean_sd(probe_auc_l1_l2)
    if auc_a_mean is not None:
        ax.axhline(auc_a_mean, color="tab:green", linestyle=":",
                   label=f"Probe AUC (L0 vs L2)={auc_a_mean:.2f}±{auc_a_sd:.2f}")
    if auc_b_mean is not None:
        ax.axhline(auc_b_mean, color="tab:purple", linestyle=":",
                   label=f"Probe AUC (L1 vs L2)={auc_b_mean:.2f}±{auc_b_sd:.2f}")

    ax.set_xticks(levels)
    ax.set_xlabel("Perturbation level")
    ax.set_ylabel("Rate / AUC")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def steering_curve(
    alphas: list[float],
    verb_rate: list[float],
    rej_rate: list[float],
    clean_success: list[float],
    out_path: str | Path,
    title: str = "Steering effect (held-out L2)",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(alphas, verb_rate, marker="o", label="Verbalization rate (L2)", linewidth=2)
    ax.plot(alphas, rej_rate,  marker="s", label="Action-rejection rate (L2)", linewidth=2)
    ax.plot(alphas, clean_success, marker="^", label="Task success (L0)",
            color="tab:gray", linewidth=2)
    ax.set_xlabel("α (steering strength)")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
