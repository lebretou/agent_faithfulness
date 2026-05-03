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
from src.probes import load_probe_results, load_activations  # noqa: E402
from src.steering import (
    probe_direction_from_weights, attach_steering_hook,
)  # noqa: E402

import numpy as np  # noqa: E402


def _existing_ids(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    ids = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["trajectory_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def _read_existing(jsonl_path: Path) -> list[dict]:
    out = []
    if not jsonl_path.exists():
        return out
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _generate_block(
    model, tok, catalog, level, n, rng, trajectory_prefix, max_steps,
    jsonl_path: Path, resume: bool = True,
):
    """Generate `n` trajectories at `level`, appending each to JSONL immediately.

    Resumes from existing IDs in jsonl_path if resume=True. Returns the full
    list of trajectories for this block (including any pre-existing ones).
    """
    done_ids = _existing_ids(jsonl_path) if resume else set()
    if done_ids:
        print(f"    [{trajectory_prefix}] resume: {len(done_ids)} already on disk")

    out = _read_existing(jsonl_path) if resume else []
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "a") as fout:
        for i in range(n):
            traj_id = f"{trajectory_prefix}_t{i:03d}"
            if traj_id in done_ids:
                continue
            try:
                q = sample_query(catalog, n_constraints=3, rng=rng,
                                 query_id=f"{trajectory_prefix}_q{i:03d}")
                traj = run_trajectory(
                    model, tok,
                    catalog=catalog,
                    query=q,
                    perturbation_level=level,
                    rng=rng,
                    trajectory_id=traj_id,
                    max_steps=max_steps,
                    capture_activations=False,
                    activations_dir=None,
                )
            except Exception as e:
                print(f"    {traj_id} crashed: {e!r}")
                continue
            if traj["final_action"] is None:
                continue
            compute_all_labels(traj)
            fout.write(json.dumps(traj, default=str) + "\n")
            fout.flush()
            out.append(traj)
    return out


def _aggregate(trajs, label_key):
    if not trajs:
        return None
    return sum(int(t["labels"][label_key]) for t in trajs) / len(trajs)


def _path_for_activation(s, corpus_root: Path, seed: int) -> Path | None:
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
        candidate = corpus_root / f"seed_{seed}" / rel
        if candidate.exists():
            return candidate
    candidate = corpus_root / f"seed_{seed}" / "activations" / p.name
    return candidate if candidate.exists() else None


def load_contrastive_direction(
    corpus_root: Path,
    seed: int,
    layer_idx: int,
    classes: tuple[int, int] = (1, 2),
) -> np.ndarray:
    """Mean(class[1]) - Mean(class[0]) at `layer_idx`, normalized.

    Walks the seed's trajectories.jsonl, loads activations at every step that
    belongs to one of the two classes, and computes the difference of means
    at the requested layer.
    """
    jsonl = corpus_root / f"seed_{seed}" / "trajectories.jsonl"
    paths: list[Path] = []
    labels: list[int] = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            lvl = int(t["perturbation_level"])
            if lvl not in classes:
                continue
            for s in t["steps"]:
                p = _path_for_activation(s, corpus_root, seed)
                if p is None:
                    continue
                paths.append(p)
                labels.append(lvl)

    if not paths:
        raise RuntimeError(
            f"No activations found for seed {seed} under {corpus_root}."
        )

    print(f"[contrastive] loading {len(paths)} activations for layer {layer_idx}...")
    activations = load_activations(paths)  # (n, n_layers, hidden)
    labels_arr = np.array(labels, dtype=np.int64)
    pos = activations[labels_arr == classes[1], layer_idx, :].astype(np.float32)
    neg = activations[labels_arr == classes[0], layer_idx, :].astype(np.float32)
    if pos.size == 0 or neg.size == 0:
        raise RuntimeError(
            f"Need both classes; got n_pos={len(pos)}, n_neg={len(neg)}"
        )
    diff = pos.mean(axis=0) - neg.mean(axis=0)
    n = float(np.linalg.norm(diff))
    if n == 0:
        raise RuntimeError("Mean-difference is zero — classes have identical means.")
    print(f"[contrastive] n_pos={len(pos)}  n_neg={len(neg)}  ||diff||={n:.4f}")
    return diff / n


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
    ap.add_argument("--no_resume", action="store_true",
                    help="Disable resume; overwrite existing per-α JSONLs.")
    ap.add_argument("--direction", choices=["probe", "contrastive"], default="probe",
                    help="probe (default) = unit-normalized logistic regression "
                         "weights; contrastive = mean(L2) - mean(L1) at the chosen "
                         "layer, normalized.")
    ap.add_argument("--layer", type=int, default=None,
                    help="Override probe.best_layer with a specific layer index. "
                         "Affects both probe and contrastive directions.")
    ap.add_argument("--corpus_root", default=None,
                    help="Path containing seed_<N>/ for contrastive direction. "
                         "If omitted, inferred as parent of probe_json's parent.")
    ap.add_argument("--contrastive_seed", type=int, default=42,
                    help="Which seed's activations to use for the contrastive direction.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "activations").mkdir(exist_ok=True)  # placeholder, unused

    # --- Load probe metadata (used either for direction or just for best_layer) ---
    results = load_probe_results(args.probe_json)
    res = next((r for r in results if r.classifier == args.classifier), None)
    if res is None:
        raise SystemExit(f"Classifier {args.classifier} not found in {args.probe_json}. "
                         f"Available: {[r.classifier for r in results]}")

    # Layer choice: --layer overrides probe.best_layer.
    layer_idx = args.layer if args.layer is not None else res.best_layer

    # --- Direction ---
    if args.direction == "probe":
        if args.layer is not None and args.layer != res.best_layer:
            print(f"[steer] WARNING: probe weights came from layer {res.best_layer}, "
                  f"steering at layer {layer_idx}. Direction is still the layer-{res.best_layer} "
                  f"probe vector applied at layer {layer_idx}.")
        direction = probe_direction_from_weights(np.asarray(res.best_layer_weights))
    else:  # contrastive
        if args.corpus_root is None:
            corpus_root = Path(args.probe_json).resolve().parent.parent
            print(f"[steer] inferred corpus_root={corpus_root}")
        else:
            corpus_root = Path(args.corpus_root)
        direction = load_contrastive_direction(
            corpus_root=corpus_root,
            seed=args.contrastive_seed,
            layer_idx=layer_idx,
            classes=(1, 2),
        )

    print(f"[steer] probe={args.classifier}  layer={layer_idx}  "
          f"best_auc(probe)={res.best_auc:.4f}  direction={args.direction}  "
          f"direction_norm={np.linalg.norm(direction):.4f}")
    best_layer = layer_idx  # used downstream as the steering layer

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
        "steering_layer": layer_idx,
        "probe_best_layer": res.best_layer,
        "probe_best_auc": res.best_auc,
        "direction": args.direction,
        "contrastive_seed": args.contrastive_seed if args.direction == "contrastive" else None,
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
        l2_jsonl = alpha_dir / "trajectories_l2.jsonl"
        l0_jsonl = alpha_dir / "trajectories_l0.jsonl"
        combined_jsonl = alpha_dir / "trajectories.jsonl"

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
                jsonl_path=l2_jsonl, resume=not args.no_resume,
            )
            print(f"[steer]   generating {args.n_l0} L0 trajectories...")
            l0_trajs = _generate_block(
                model, tok, catalog, level=0, n=args.n_l0, rng=rng_l0,
                trajectory_prefix=f"steer_a{alpha:g}_l0",
                max_steps=cfg["max_steps"],
                jsonl_path=l0_jsonl, resume=not args.no_resume,
            )
        finally:
            handle.remove()
        elapsed = time.time() - t0

        # Combined view for downstream consumers.
        with open(combined_jsonl, "w") as f:
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
