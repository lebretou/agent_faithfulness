"""Plotting.

Matplotlib only. All inputs are small numpy arrays / dicts; no model or torch.
Presentation-ready styling: colorblind-safe palette, 300 dpi, large fonts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


# Okabe-Ito colorblind-safe palette.
PALETTE = {
    "blue":      "#0072B2",
    "orange":    "#E69F00",
    "green":     "#009E73",
    "vermilion": "#D55E00",
    "purple":    "#CC79A7",
    "sky":       "#56B4E9",
    "yellow":    "#F0E442",
    "gray":      "#5A5A5A",
    "lightgray": "#BBBBBB",
}

DPI = 300


def _apply_style(ax):
    """Clean projector-friendly style: no top/right spines, light y-grid only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#444444")
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", labelsize=12, colors="#222222",
                   length=4, width=1.0)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _set_labels(ax, *, xlabel=None, ylabel=None, title=None):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=13.5, color="#222222", labelpad=8)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=13.5, color="#222222", labelpad=8)
    if title is not None:
        ax.set_title(title, fontsize=15.5, color="#111111", pad=14, weight="semibold")


def _save(fig, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def probe_auc_by_layer(
    layer_aucs_by_classifier: dict[str, list[float] | list[list[float]]],
    out_path: str | Path,
    title: str = "Probe AUC by layer",
) -> None:
    """Plot per-layer AUC for each classifier.

    Each value can be either a list (single seed) or a list-of-lists (one per
    seed) — in which case we plot mean and a shaded ±1 std band.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE["blue"], PALETTE["vermilion"]]

    for (name, vals), color in zip(layer_aucs_by_classifier.items(), colors):
        arr = np.asarray(vals, dtype=np.float64)
        if arr.ndim == 1:
            xs = np.arange(arr.shape[0])
            ax.plot(xs, arr, label=name, color=color, linewidth=2.4, zorder=3)
            mean = arr
        else:  # (n_seeds, n_layers)
            mean = arr.mean(axis=0)
            sd = arr.std(axis=0)
            xs = np.arange(arr.shape[1])
            ax.plot(xs, mean, label=name, color=color, linewidth=2.4, zorder=3)
            ax.fill_between(xs, mean - sd, mean + sd, color=color,
                            alpha=0.18, linewidth=0, zorder=2)

    # Annotate peak of last (central) classifier.
    if mean is not None:
        peak_idx = int(np.argmax(mean))
        peak_val = float(mean[peak_idx])
        ax.scatter([peak_idx], [peak_val], s=70, color=colors[-1],
                   edgecolor="white", linewidth=1.5, zorder=5)
        ax.annotate(f"peak: layer {peak_idx}\nAUC = {peak_val:.2f}",
                    xy=(peak_idx, peak_val),
                    xytext=(peak_idx - 8, peak_val - 0.18),
                    fontsize=11, color="#222222",
                    arrowprops=dict(arrowstyle="->", color="#666666",
                                    lw=1.0, shrinkA=0, shrinkB=4))

    ax.axhline(0.5, color=PALETTE["gray"], linestyle=(0, (4, 3)),
               linewidth=1.2, label="chance", zorder=1)

    ax.set_xlim(-0.5, max(xs) + 0.5)
    ax.set_ylim(0.4, 1.02)
    _apply_style(ax)
    _set_labels(ax, xlabel="Transformer layer",
                ylabel="5-fold CV AUC", title=title)
    leg = ax.legend(loc="lower right", fontsize=11.5, frameon=True,
                    framealpha=0.95, edgecolor="#DDDDDD")
    leg.get_frame().set_linewidth(0.8)

    _save(fig, out_path)


def three_curve_plot(
    levels: list[int],
    verb_rate: list[float] | list[list[float]],
    rej_rate: list[float] | list[list[float]],
    probe_auc_l0_l2: float | list[float] | None,
    probe_auc_l1_l2: float | list[float] | None,
    out_path: str | Path,
    title: str = "Faithfulness signals vs perturbation level",
) -> None:
    """Headline figure ()."""
    fig, ax = plt.subplots(figsize=(9, 5.4))

    def _plot_curve(vals, label, marker, color):
        arr = np.asarray(vals, dtype=np.float64)
        if arr.ndim == 1:
            ax.plot(levels, arr, marker=marker, label=label, color=color,
                    linewidth=2.6, markersize=9, markeredgecolor="white",
                    markeredgewidth=1.4, zorder=4)
        else:
            mean = arr.mean(axis=0)
            sd = arr.std(axis=0)
            ax.errorbar(levels, mean, yerr=sd, marker=marker, label=label,
                        color=color, linewidth=2.6, markersize=9, capsize=5,
                        markeredgecolor="white", markeredgewidth=1.4,
                        elinewidth=1.4, zorder=4)

    _plot_curve(verb_rate, "CoT verbalizes inconsistency", "o", PALETTE["blue"])
    _plot_curve(rej_rate,  "Action rejects perturbation",  "s", PALETTE["vermilion"])

    def _mean_sd(v):
        if v is None:
            return None, None
        a = np.asarray(v, dtype=np.float64).flatten()
        return float(a.mean()), (float(a.std()) if a.size > 1 else 0.0)

    auc_a_mean, auc_a_sd = _mean_sd(probe_auc_l0_l2)
    auc_b_mean, auc_b_sd = _mean_sd(probe_auc_l1_l2)
    if auc_a_mean is not None:
        ax.axhline(auc_a_mean, color=PALETTE["green"], linestyle=(0, (5, 3)),
                   linewidth=1.8, zorder=2,
                   label=f"Probe AUC, L0 vs L2 = {auc_a_mean:.2f} ± {auc_a_sd:.2f}")
    if auc_b_mean is not None:
        ax.axhline(auc_b_mean, color=PALETTE["purple"], linestyle=(0, (5, 3)),
                   linewidth=1.8, zorder=2,
                   label=f"Probe AUC, L1 vs L2 = {auc_b_mean:.2f} ± {auc_b_sd:.2f}")

    level_names = {0: "L0\nclean", 1: "L1\nnon-constraint\nswap",
                   2: "L2\nconstraint\nviolation"}
    ax.set_xticks(levels)
    ax.set_xticklabels([level_names.get(l, str(l)) for l in levels])
    ax.set_xlim(min(levels) - 0.3, max(levels) + 0.3)
    ax.set_ylim(0, 1.05)
    _apply_style(ax)
    _set_labels(ax, xlabel="Perturbation level",
                ylabel="Rate  /  AUC", title=title)
    leg = ax.legend(loc="center left", fontsize=10.5, frameon=True,
                    framealpha=0.95, edgecolor="#DDDDDD",
                    bbox_to_anchor=(0.015, 0.55))
    leg.get_frame().set_linewidth(0.8)

    _save(fig, out_path)


def steering_curve(
    alphas: list[float],
    verb_rate: list[float],
    rej_rate: list[float],
    clean_success: list[float],
    out_path: str | Path,
    title: str = "Steering effect (held-out L2)",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))

    ax.plot(alphas, verb_rate, marker="o", color=PALETTE["blue"],
            label="CoT verbalization (L2)", linewidth=2.6, markersize=9,
            markeredgecolor="white", markeredgewidth=1.4, zorder=4)
    ax.plot(alphas, rej_rate, marker="s", color=PALETTE["vermilion"],
            label="Action rejection (L2)", linewidth=2.6, markersize=9,
            markeredgecolor="white", markeredgewidth=1.4, zorder=4)
    ax.plot(alphas, clean_success, marker="^", color=PALETTE["gray"],
            label="Task success (L0, control)", linewidth=2.2, markersize=9,
            markeredgecolor="white", markeredgewidth=1.4, linestyle="--",
            zorder=3, alpha=0.85)

    # Shade two regimes:
    #   - dissociation: verb stays ≥ baseline AND rejection < baseline
    #   - over-steering: verb has fallen back below baseline (model degrading)
    a = np.asarray(alphas, dtype=np.float64)
    base_v = verb_rate[0]
    base_r = rej_rate[0]
    in_dissoc = [(verb_rate[i] >= base_v - 0.02 and rej_rate[i] < base_r)
                 for i in range(len(a))]
    over_steer = [(verb_rate[i] < base_v - 0.02 and i > 0)
                  for i in range(len(a))]

    dissoc_idxs = [i for i, v in enumerate(in_dissoc) if v]
    if dissoc_idxs:
        lo = a[dissoc_idxs[0]] - 0.05
        hi = a[dissoc_idxs[-1]] + 0.05
        ax.axvspan(lo, hi, color=PALETTE["yellow"], alpha=0.20, zorder=0,
                   label="dissociation: verb ↑, rejection ↓")

    over_idxs = [i for i, v in enumerate(over_steer) if v]
    if over_idxs:
        lo = a[over_idxs[0]] - 0.05
        hi = a[-1] + 0.05
        ax.axvspan(lo, hi, color=PALETTE["vermilion"], alpha=0.10, zorder=0,
                   label="over-steering: model degrades")

    ax.set_xticks(alphas)
    ax.set_xlim(min(alphas) - 0.1, max(alphas) + 0.1)
    ax.set_ylim(0, 1.05)
    _apply_style(ax)
    _set_labels(ax, xlabel="Steering strength  α",
                ylabel="Rate", title=title)
    leg = ax.legend(loc="center left", fontsize=10.5, frameon=True,
                    framealpha=0.95, edgecolor="#DDDDDD",
                    bbox_to_anchor=(1.01, 0.5))
    leg.get_frame().set_linewidth(0.8)

    _save(fig, out_path)
