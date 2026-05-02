"""Train per-layer probes on Drive-resident activations.

project_plan.md §8. Runs on Colab (or any host with the activations on disk).

For each seed:
  - Load all trajectories' activation tensors at every captured (traj, step).
  - Build labels: each step inherits the trajectory's perturbation_level.
  - Train Classifier A (L0 vs L2) and Classifier B (L1 vs L2) per layer.
  - Save AUC-by-layer + best-layer probe weights to a JSON next to the JSONL.

Usage (Colab):
    python scripts/03_train_probes.py --root /content/drive/MyDrive/agent_faithfulness/data \
                                      --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.probes import (
    load_activations, train_per_layer_probes, save_probe_results, ProbeResult,
)


def _path_for_step(s: dict, root: Path, seed: int, fallback_traj_id: str) -> Path | None:
    p = s.get("activations_path")
    if not p:
        return None
    p = Path(p)
    if p.exists():
        return p
    # Path stored at generation time was Colab-absolute. Map to local root.
    # Last 3 path parts: data/seed_X/activations/traj_YYYY_step_Z.pt
    parts = p.parts
    if "activations" in parts:
        idx = parts.index("activations")
        rel = Path(*parts[idx:])
        candidate = root / f"seed_{seed}" / rel
        if candidate.exists():
            return candidate
    # Last fallback: try root/seed_X/activations/<filename>.
    candidate = root / f"seed_{seed}" / "activations" / p.name
    if candidate.exists():
        return candidate
    return None


def collect_seed(root: Path, seed: int):
    """Return (paths, levels) where paths are activation files and levels are the
    perturbation_level of the parent trajectory.

    Each step that has an activation file contributes one sample.
    """
    jsonl = root / f"seed_{seed}" / "trajectories.jsonl"
    paths: list[Path] = []
    levels: list[int] = []
    n_traj_with_acts = 0
    n_traj_total = 0
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            n_traj_total += 1
            steps_with = 0
            for s in t["steps"]:
                p = _path_for_step(s, root, seed, t["trajectory_id"])
                if p is None:
                    continue
                paths.append(p)
                levels.append(int(t["perturbation_level"]))
                steps_with += 1
            if steps_with:
                n_traj_with_acts += 1
    print(f"  seed {seed}: {n_traj_with_acts}/{n_traj_total} trajectories contributed activations "
          f"({len(paths)} samples total)")
    return paths, np.array(levels, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Path containing seed_<N>/ subdirs (e.g. data/ on the host).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out_subdir", default="probes",
                    help="Output subdir under root.")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = root / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        paths, levels = collect_seed(root, seed)
        if len(paths) == 0:
            print(f"  No activations found for seed {seed}. Skipping.")
            continue

        print("  Loading activations into memory...")
        activations = load_activations(paths)
        print(f"  Activations shape: {activations.shape}")

        results: list[ProbeResult] = []

        # Classifier A: Level 0 vs Level 2.
        mask_a = (levels == 0) | (levels == 2)
        if mask_a.sum() >= 20:
            X_a = activations[mask_a]
            y_a = (levels[mask_a] == 2).astype(np.int64)  # L2 = positive class
            res_a = train_per_layer_probes(
                X_a, y_a, classifier_name="A_L0_vs_L2",
                n_folds=args.n_folds, C=args.C, seed=seed,
            )
            print(f"  [A] L0 vs L2  best layer={res_a.best_layer}  AUC={res_a.best_auc:.4f}")
            results.append(res_a)
        else:
            print(f"  [A] Skipped (only {mask_a.sum()} samples).")

        # Classifier B: Level 1 vs Level 2.
        mask_b = (levels == 1) | (levels == 2)
        if mask_b.sum() >= 20:
            X_b = activations[mask_b]
            y_b = (levels[mask_b] == 2).astype(np.int64)  # L2 = positive class
            res_b = train_per_layer_probes(
                X_b, y_b, classifier_name="B_L1_vs_L2",
                n_folds=args.n_folds, C=args.C, seed=seed,
            )
            print(f"  [B] L1 vs L2  best layer={res_b.best_layer}  AUC={res_b.best_auc:.4f}")
            results.append(res_b)
        else:
            print(f"  [B] Skipped (only {mask_b.sum()} samples).")

        out_path = out_dir / f"probes_seed{seed}.json"
        save_probe_results(results, out_path)
        print(f"  Saved -> {out_path}")

    print(f"\nAll seeds done. Probe artifacts in {out_dir}/")


if __name__ == "__main__":
    main()
