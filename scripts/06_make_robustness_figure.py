"""Slide 8.5: 2-panel robustness figure showing the verb↑ / rej↓ dissociation
replicates across steering vector type and layer, on a shared α grid.

Panels (left → right):
  1. Contrastive direction @ L19 (steering_contrastive_l19_highn)
  2. Probe direction @ L25       (steering_probe_l25_native_v2)

Both runs swept α ∈ {0, 0.5, 1, 1.5, 2}, so the x-axes line up.

Usage:
    python scripts/06_make_robustness_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.figures import PALETTE, _apply_style, _save


PANELS = [
    ("Contrastive direction · layer 19",
     "data/steering_summaries/steering_contrastive_l19_highn/steering_summary.json"),
    ("Probe direction · layer 25",
     "data/steering_summaries/steering_probe_l25_native_v2/steering_summary.json"),
]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)

    for ax, (title, path) in zip(axes, PANELS):
        s = json.load(open(path))
        per = sorted(s["per_alpha"], key=lambda p: p["alpha"])
        a    = [p["alpha"]        for p in per]
        verb = [p["verb_rate_l2"] for p in per]
        rej  = [p["rej_rate_l2"]  for p in per]
        succ = [p["task_success_l0"] for p in per]

        ax.plot(a, verb, marker="o", color=PALETTE["blue"],
                label="CoT verbalization (L2)", linewidth=2.6, markersize=9,
                markeredgecolor="white", markeredgewidth=1.4, zorder=4)
        ax.plot(a, rej, marker="s", color=PALETTE["vermilion"],
                label="Action rejection (L2)", linewidth=2.6, markersize=9,
                markeredgecolor="white", markeredgewidth=1.4, zorder=4)
        ax.plot(a, succ, marker="^", color=PALETTE["gray"],
                label="Task success (L0, control)", linewidth=2.0,
                markersize=8, markeredgecolor="white", markeredgewidth=1.2,
                linestyle="--", alpha=0.85, zorder=3)

        # Annotate Δrejection at α=1.
        try:
            i0, i1 = a.index(0.0), a.index(1.0)
            drop = (rej[i1] - rej[i0]) * 100
            ax.annotate(f"Δrejection @ α=1:  {drop:+.0f} pp",
                        xy=(1.0, rej[i1]),
                        xytext=(1.05, rej[i1] - 0.16),
                        fontsize=11, color=PALETTE["vermilion"],
                        weight="semibold",
                        arrowprops=dict(arrowstyle="->",
                                        color=PALETTE["vermilion"],
                                        lw=1.0, shrinkA=0, shrinkB=4))
        except ValueError:
            pass

        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.1, 2.1)
        ax.set_xticks(a)
        _apply_style(ax)
        ax.set_xlabel("Steering strength  α", fontsize=13,
                      color="#222222", labelpad=8)
        ax.set_title(title, fontsize=13.5, color="#111111", pad=10,
                     weight="semibold")

    axes[0].set_ylabel("Rate", fontsize=13.5, color="#222222", labelpad=8)

    # Shared legend below — leave room with subplots_adjust.
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=3,
                     fontsize=11.5, frameon=False,
                     bbox_to_anchor=(0.5, -0.12))

    fig.suptitle("Dissociation replicates: verbalization stays up, "
                 "action rejection drops",
                 fontsize=15.5, weight="semibold", color="#111111", y=1.02)

    fig.subplots_adjust(bottom=0.18)

    out = Path("figures/steering_robustness.png")
    _save(fig, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
