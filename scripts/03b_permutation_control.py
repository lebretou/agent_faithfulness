"""Permutation-label control for the per-layer probes.

Shuffles the L1/L2 (and L0/L2) labels and re-trains the same per-layer
probes. AUC should collapse to chance (~0.5) at every layer. If it stays
elevated, the original probe AUC reflects a setup-side confound (e.g.,
unbalanced trajectory length, batch-position bias) rather than genuine
linear separability of the labelled classes.

Usage:
    python scripts/03b_permutation_control.py \
        --root data \
        --seeds 42 43 44 \
        --n_perms 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse helpers from the real probe script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

# We can't `from 03_train_probes import collect_seed` (numeric prefix), so
# replicate just the small helper.
from src.probes import load_activations, train_per_layer_probes  # noqa: E402


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n_perms", type=int, default=3,
                    help="Number of label-shuffles per classifier per seed.")
    ap.add_argument("--out_subdir", default="probes")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = root / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    full_results = []

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        paths, levels = collect_seed(root, seed)
        if len(paths) == 0:
            print("  no activations; skipping.")
            continue

        print(f"  Loading {len(paths)} activations...")
        activations = load_activations(paths)
        print(f"  Activations shape: {activations.shape}")

        for cls_name, mask, label_pos in [
            ("A_L0_vs_L2", (levels == 0) | (levels == 2), 2),
            ("B_L1_vs_L2", (levels == 1) | (levels == 2), 2),
        ]:
            X = activations[mask]
            y_true = (levels[mask] == label_pos).astype(np.int64)

            # True AUC for reference.
            res_true = train_per_layer_probes(X, y_true, classifier_name=cls_name, seed=seed)
            print(f"  [{cls_name}] TRUE     n={len(y_true):4d}  best_layer={res_true.best_layer:2d}  "
                  f"best_auc={res_true.best_auc:.4f}")

            # Shuffled-label AUCs.
            shuffle_aucs = []
            shuffle_best_aucs = []
            shuffle_rng = np.random.default_rng(seed * 100 + 7)
            for perm_i in range(args.n_perms):
                y_shuf = y_true.copy()
                shuffle_rng.shuffle(y_shuf)
                res_shuf = train_per_layer_probes(
                    X, y_shuf, classifier_name=f"{cls_name}_shuf{perm_i}",
                    seed=seed * 1000 + perm_i,
                )
                shuffle_aucs.append(res_shuf.layer_aucs)
                shuffle_best_aucs.append(res_shuf.best_auc)
                print(f"  [{cls_name}] SHUFFLE{perm_i}  best_layer={res_shuf.best_layer:2d}  "
                      f"best_auc={res_shuf.best_auc:.4f}")

            # Aggregate stats.
            shuffle_aucs = np.array(shuffle_aucs)  # (n_perms, n_layers)
            full_results.append({
                "seed": seed,
                "classifier": cls_name,
                "true_layer_aucs": res_true.layer_aucs,
                "true_best_layer": res_true.best_layer,
                "true_best_auc": res_true.best_auc,
                "shuffled_layer_aucs": shuffle_aucs.tolist(),
                "shuffled_best_aucs": shuffle_best_aucs,
                "shuffled_mean_layer_aucs": shuffle_aucs.mean(axis=0).tolist(),
                "shuffled_max_layer_auc": float(shuffle_aucs.max()),
                "shuffled_mean_best_auc": float(np.mean(shuffle_best_aucs)),
            })

    out_path = out_dir / "permutation_control.json"
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # --- Compact verdict ---
    print("\n=== Verdict ===")
    print(f"{'seed':<5}{'cls':<14}{'true_AUC':<10}{'shuf_mean':<11}{'shuf_max':<10}{'gap':<8}")
    pass_all = True
    for r in full_results:
        gap = r["true_best_auc"] - r["shuffled_mean_best_auc"]
        ok = (r["shuffled_mean_best_auc"] < 0.60) and (r["shuffled_max_layer_auc"] < 0.65)
        flag = "✓" if ok else "✗"
        if not ok:
            pass_all = False
        print(f"{r['seed']:<5}{r['classifier']:<14}{r['true_best_auc']:<10.4f}"
              f"{r['shuffled_mean_best_auc']:<11.4f}{r['shuffled_max_layer_auc']:<10.4f}"
              f"{gap:<8.4f}{flag}")
    if pass_all:
        print("\nPASS: shuffled-label AUC near chance everywhere; true probe AUC reflects the labels, not setup confounds.")
    else:
        print("\nFAIL: shuffled-label AUC elevated. Investigate confounds before interpreting the true probes.")


if __name__ == "__main__":
    main()
