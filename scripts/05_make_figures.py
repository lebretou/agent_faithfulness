"""Build the probe-AUC-by-layer and three-curve plots from local artifacts.

Local-only. Reads the JSONL trajectories (for verb%/rej% per level) and the
probe-AUC JSON files written by 03_train_probes.py.

Usage:
    python scripts/05_make_figures.py \
        --data_dir data \
        --probes_dir data/probes \
        --out_dir figures \
        --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.figures import probe_auc_by_layer, three_curve_plot
from src.probes import load_probe_results


def _label_rates(trajs, levels=(0, 1, 2)):
    """Per-level (verb_rate, rej_rate) for one seed."""
    by = defaultdict(list)
    for t in trajs:
        by[t["perturbation_level"]].append(t["labels"])
    verb = []
    rej  = []
    for lvl in levels:
        ts = by[lvl]
        if not ts:
            verb.append(np.nan)
            rej.append(np.nan)
            continue
        verb.append(sum(int(L["cot_mentions_perturbation"]) for L in ts) / len(ts))
        rej.append(sum(int(L["action_rejects_perturbation"]) for L in ts) / len(ts))
    return verb, rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--probes_dir", default="data/probes")
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    probes_dir = Path(args.probes_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    levels = [0, 1, 2]

    # ---- Per-seed label rates ----
    verb_per_seed = []
    rej_per_seed = []
    for seed in args.seeds:
        jsonl = data_dir / f"seed_{seed}" / "trajectories.jsonl"
        trajs = [json.loads(l) for l in open(jsonl) if l.strip()]
        v, r = _label_rates(trajs, levels)
        verb_per_seed.append(v)
        rej_per_seed.append(r)
        print(f"seed {seed}: verb%={[f'{x*100:.1f}' for x in v]}  "
              f"rej%={[f'{x*100:.1f}' for x in r]}")

    # ---- Per-seed probe AUCs ----
    aucs_a_by_layer = []   # (n_seeds, n_layers)
    aucs_b_by_layer = []
    auc_a_best = []
    auc_b_best = []
    for seed in args.seeds:
        p = probes_dir / f"probes_seed{seed}.json"
        if not p.exists():
            print(f"WARNING: probe results missing for seed {seed} at {p}")
            continue
        results = load_probe_results(p)
        for r in results:
            if r.classifier == "A_L0_vs_L2":
                aucs_a_by_layer.append(r.layer_aucs)
                auc_a_best.append(r.best_auc)
            elif r.classifier == "B_L1_vs_L2":
                aucs_b_by_layer.append(r.layer_aucs)
                auc_b_best.append(r.best_auc)

    # ---- Figure 1: probe AUC by layer ----
    if aucs_a_by_layer and aucs_b_by_layer:
        probe_auc_by_layer(
            {
                "A: L0 vs L2 (sanity)": aucs_a_by_layer,
                "B: L1 vs L2 (central)": aucs_b_by_layer,
            },
            out_dir / "probe_auc_by_layer.png",
            title="Per-layer probe AUC (mean ± std across seeds)",
        )
        print(f"Wrote {out_dir / 'probe_auc_by_layer.png'}")

    # ---- Figure 2: three-curve plot ----
    three_curve_plot(
        levels=levels,
        verb_rate=verb_per_seed,
        rej_rate=rej_per_seed,
        probe_auc_l0_l2=auc_a_best if auc_a_best else None,
        probe_auc_l1_l2=auc_b_best if auc_b_best else None,
        out_path=out_dir / "three_curve_plot.png",
        title="Verbalization vs Action vs Probe by perturbation level",
    )
    print(f"Wrote {out_dir / 'three_curve_plot.png'}")


if __name__ == "__main__":
    main()
