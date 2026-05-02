"""Day 3 steering sweep. project_plan.md sections 10.2 and 10.3.

For each α in --alphas, attach a forward-hook adding α·direction at the
probe's best layer (prefill-only, last-token position) and generate fresh
trajectories at L0 and L2. Trajectories are guaranteed held-out from probe
training because we use a NEW RNG seed (default 100) that no probe seed saw.

Per α, we report:
  - Verbalization rate on L2 trajectories
  - Action-rejection rate on L2 trajectories
  - Task-success rate on L0 trajectories (sanity that steering hasn't broken
    clean behavior)

Outputs:
  {out_dir}/alpha_{α}/trajectories.jsonl
  {out_dir}/steering_summary.json   (alphas + rates, ready for 05_make_figures)

Usage on Colab:
    python scripts/04_steering_sweep.py \
      --probe_json /content/drive/MyDrive/agent_faithfulness/data/probes/probes_seed42.json \
      --catalog    /content/drive/MyDrive/agent_faithfulness/data/catalog.json \
      --out_dir    /content/drive/MyDrive/agent_faithfulness/data/steering_run1 \
      --alphas 0 0.5 1 2 4 \
      --n_l2 50 --n_l0 25 \
      --seed 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import load_model, run_trajectory  # noqa: E402
from src.queries import sample_query  # noqa: E402
from src.labels import compute_all_labels  # noqa: E402
from src.probes import load_probe_results  # noqa: E402
from src.steering import probe_direction_from_weights, attach_steering_hook  # noqa: E402


def _generate_block(
    model, tok, catalog, level, n, rng, trajectory_prefix, max_steps,
    activations_dir,
):
    """Generate `n` trajectories at `level`. Returns list of trajectory dicts."""
    out = []
    for i in range(n):
        try:
            q = sample_query(catalog, n_constraints=3, rng=rng,
                             query_id=f"{trajectory_prefix}_q{i:03d}")
            traj = run_trajectory(
                model, tok,
                catalog=catalog,
                query=q,
                perturbation_level=level,
                rng=rng,
                trajectory_id=f"{trajectory_prefix}_t{i:03d}",
                max_steps=max_steps,
                # Skip activation capture for steering: not needed, saves time/disk.
                capture_activations=False,
                activations_dir=activations_dir,
            )
        except Exception as e:
            print(f"    {trajectory_prefix}_t{i:03d} crashed: {e!r}")
            continue
        if traj["final_action"] is None:
            continue
        compute_all_labels(traj)
        out.append(traj)
    return out


def _aggregate(trajs, label_key):
    if not trajs:
        return None
    return sum(int(t["labels"][label_key]) for t in trajs) / len(trajs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--probe_json", required=True,
                    help="Path to probes_seed<N>.json from 03_train_probes.")
    ap.add_argument("--classifier", default="B_L1_vs_L2",
                    help="Which probe to take the direction from.")
    ap.add_argument("--catalog", required=True,
                    help="Path to data/catalog.json.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--n_l2", type=int, default=50)
    ap.add_argument("--n_l0", type=int, default=25)
    ap.add_argument("--seed", type=int, default=100,
                    help="Fresh RNG seed (held out from probe training seeds).")
    ap.add_argument("--continuous", action="store_true",
                    help="Steer on every generated token, not prefill-only. "
                         "Off-spec; default matches §10.2.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "activations").mkdir(exist_ok=True)  # placeholder, unused

    # --- Load probe direction ---
    results = load_probe_results(args.probe_json)
    res = next((r for r in results if r.classifier == args.classifier), None)
    if res is None:
        raise SystemExit(f"Classifier {args.classifier} not found in {args.probe_json}. "
                         f"Available: {[r.classifier for r in results]}")
    best_layer = res.best_layer
    direction = probe_direction_from_weights(np.asarray(res.best_layer_weights))
    print(f"[steer] probe={args.classifier}  best_layer={best_layer}  "
          f"best_auc={res.best_auc:.4f}  direction_norm={np.linalg.norm(direction):.4f}")

    # --- Load catalog ---
    with open(args.catalog) as f:
        catalog = json.load(f)
    print(f"[steer] catalog: {len(catalog)} items from {args.catalog}")

    # --- Load model ---
    print(f"[steer] loading {cfg['model']}...")
    model, tok = load_model(cfg["model"], dtype=cfg["dtype"])
    print(f"[steer] model loaded.")

    summary = {
        "probe_json": str(args.probe_json),
        "classifier": args.classifier,
        "best_layer": best_layer,
        "probe_best_auc": res.best_auc,
        "alphas": list(args.alphas),
        "n_l2_target": args.n_l2,
        "n_l0_target": args.n_l0,
        "seed": args.seed,
        "prefill_only": not args.continuous,
        "per_alpha": [],
    }

    for alpha in args.alphas:
        print(f"\n[steer] === α = {alpha} ===")
        # Fresh RNG per α for reproducibility AND to ensure each α sees the same
        # query distribution (controls for query difficulty across α).
        rng_l2 = random.Random(args.seed * 1000 + int(alpha * 100) + 1)
        rng_l0 = random.Random(args.seed * 1000 + int(alpha * 100) + 2)

        alpha_dir = out_dir / f"alpha_{alpha:g}"
        alpha_dir.mkdir(exist_ok=True)
        jsonl_path = alpha_dir / "trajectories.jsonl"
        if jsonl_path.exists():
            jsonl_path.unlink()  # fresh per re-run

        t0 = time.time()
        handle = attach_steering_hook(
            model, layer_idx=best_layer, direction=direction, alpha=alpha,
            prefill_only=not args.continuous,
        )
        try:
            print(f"[steer]   generating {args.n_l2} L2 trajectories...")
            l2_trajs = _generate_block(
                model, tok, catalog, level=2, n=args.n_l2, rng=rng_l2,
                trajectory_prefix=f"steer_a{alpha:g}_l2",
                max_steps=cfg["max_steps"],
                activations_dir=alpha_dir / "activations",
            )
            print(f"[steer]   generating {args.n_l0} L0 trajectories...")
            l0_trajs = _generate_block(
                model, tok, catalog, level=0, n=args.n_l0, rng=rng_l0,
                trajectory_prefix=f"steer_a{alpha:g}_l0",
                max_steps=cfg["max_steps"],
                activations_dir=alpha_dir / "activations",
            )
        finally:
            handle.remove()
        elapsed = time.time() - t0

        with open(jsonl_path, "w") as f:
            for t in l2_trajs + l0_trajs:
                f.write(json.dumps(t, default=str) + "\n")

        verb_l2 = _aggregate(l2_trajs, "cot_mentions_perturbation")
        rej_l2  = _aggregate(l2_trajs, "action_rejects_perturbation")
        succ_l0 = _aggregate(l0_trajs, "task_success")

        per = {
            "alpha": alpha,
            "n_l2": len(l2_trajs),
            "n_l0": len(l0_trajs),
            "verb_rate_l2": verb_l2,
            "rej_rate_l2": rej_l2,
            "task_success_l0": succ_l0,
            "elapsed_s": elapsed,
        }
        summary["per_alpha"].append(per)
        print(f"[steer]   verb_l2={verb_l2:.3f}  rej_l2={rej_l2:.3f}  "
              f"succ_l0={succ_l0:.3f}  ({elapsed:.0f}s)")

    summary_path = out_dir / "steering_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[steer] summary -> {summary_path}")

    # Compact final table
    print("\nα      n_l2  n_l0  verb%   rej%    succ%(L0)")
    for p in summary["per_alpha"]:
        print(f"{p['alpha']:<6g} {p['n_l2']:<5d} {p['n_l0']:<5d} "
              f"{p['verb_rate_l2']*100:6.1f}  {p['rej_rate_l2']*100:6.1f}  "
              f"{p['task_success_l0']*100:6.1f}")


if __name__ == "__main__":
    main()
