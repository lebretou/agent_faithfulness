"""Train a probe at a specific layer index and save in the standard
probes_seed<N>.json format. Use when you want a probe direction that's
geometrically native to a layer other than the per-layer-AUC argmax.

Loads the same activations as 03_train_probes.py, but only computes
weights at one layer (much faster than the full sweep).

Usage (Colab):
    python scripts/03c_train_layer_probe.py \\
        --root /content/drive/MyDrive/agent_faithfulness/data \\
        --seed 42 --layer 25 \\
        --out /content/drive/MyDrive/agent_faithfulness/data/probes/probes_seed42_layer25.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.probes import load_activations, save_probe_results, ProbeResult  # noqa: E402


def _path_for_step(s, root, seed):
    p = s.get("activations_path")
    if not p:
        return None
    p = Path(p)
    if p.exists():
        return p
    parts = p.parts
    if "activations" in parts:
        idx = parts.index("activations")
        rel = Path(*parts[idx:])
        candidate = root / f"seed_{seed}" / rel
        if candidate.exists():
            return candidate
    candidate = root / f"seed_{seed}" / "activations" / p.name
    return candidate if candidate.exists() else None


def collect_seed(root: Path, seed: int):
    jsonl = root / f"seed_{seed}" / "trajectories.jsonl"
    paths, levels = [], []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            for s in t["steps"]:
                p = _path_for_step(s, root, seed)
                if p is None:
                    continue
                paths.append(p)
                levels.append(int(t["perturbation_level"]))
    return paths, np.array(levels, dtype=np.int64)


def train_at_layer(activations, labels, layer_idx, classifier_name, n_folds=5, seed=42):
    X = activations[:, layer_idx, :].astype(np.float32)
    y = labels.astype(np.int64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear", random_state=seed)
    cv_aucs = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")
    auc = float(cv_aucs.mean())
    # Refit on all data for the steering vector.
    final = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear", random_state=seed)
    final.fit(X, y)
    weights = final.coef_.flatten().tolist()
    return ProbeResult(
        classifier=classifier_name,
        layer_aucs=[auc],          # single-element list; index 0 = the trained layer
        n_layers=1,
        n_samples=len(y),
        n_pos=int((y == 1).sum()),
        n_neg=int((y == 0).sum()),
        best_layer=layer_idx,      # << critical: tells 04_steering_sweep where to attach
        best_auc=auc,
        best_layer_weights=weights,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[layer-probe] seed={args.seed} layer={args.layer}")
    paths, levels = collect_seed(Path(args.root), args.seed)
    if len(paths) == 0:
        raise SystemExit("No activations found.")

    print(f"  Loading {len(paths)} activations...")
    activations = load_activations(paths)
    print(f"  Activations shape: {activations.shape}")

    results = []
    # Classifier A: L0 vs L2 (sanity).
    mask_a = (levels == 0) | (levels == 2)
    if mask_a.sum() >= 20:
        y_a = (levels[mask_a] == 2).astype(np.int64)
        res_a = train_at_layer(activations[mask_a], y_a, args.layer,
                               "A_L0_vs_L2", seed=args.seed)
        print(f"  [A] L0 vs L2  AUC={res_a.best_auc:.4f}")
        results.append(res_a)

    # Classifier B: L1 vs L2 (central).
    mask_b = (levels == 1) | (levels == 2)
    if mask_b.sum() >= 20:
        y_b = (levels[mask_b] == 2).astype(np.int64)
        res_b = train_at_layer(activations[mask_b], y_b, args.layer,
                               "B_L1_vs_L2", seed=args.seed)
        print(f"  [B] L1 vs L2  AUC={res_b.best_auc:.4f}")
        results.append(res_b)

    save_probe_results(results, args.out)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
