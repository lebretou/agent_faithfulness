"""Generate the trajectory corpus on Colab Pro+ A100.

project_plan.md sections 5.5, 6, 7, 11 (Day 1 overnight).

Resumable: appends each finished trajectory to JSONL immediately and skips
already-done trajectory_ids on --resume.

Usage (from a Colab cell):
    !python scripts/01_generate_corpus.py \
        --config configs/default.yaml \
        --n_trajectories 250 \
        --out_dir /content/drive/MyDrive/agent_faithfulness/data
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

# Make `src` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.catalog import generate_catalog, save_catalog  # noqa: E402
from src.queries import sample_query  # noqa: E402
from src.labels import compute_all_labels  # noqa: E402
from src.agent import load_model, run_trajectory  # noqa: E402


def _allocate_levels(n: int, distribution: dict) -> list[int]:
    counts = {int(k): int(round(n * float(v))) for k, v in distribution.items()}
    # Adjust rounding so they sum to n.
    diff = n - sum(counts.values())
    if diff != 0:
        keys = sorted(counts.keys())
        counts[keys[-1]] += diff
    levels = []
    for lvl, c in counts.items():
        levels.extend([lvl] * c)
    return levels


def _existing_trajectory_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ids.add(obj["trajectory_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n_trajectories", type=int, default=None,
                    help="Override config.n_trajectories.")
    ap.add_argument("--out_dir", default=None,
                    help="Override config.out_dir. Use Drive path on Colab.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override config.seed for query+perturbation RNG. "
                         "Catalog seed is fixed at config.seed so all seeds share the world.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip trajectory_ids already present in trajectories.jsonl.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n_trajectories = args.n_trajectories or cfg["n_trajectories"]
    out_dir = Path(args.out_dir or cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectories_path = out_dir / cfg["trajectories_filename"]
    activations_dir = out_dir / cfg["activations_subdir"]
    activations_dir.mkdir(parents=True, exist_ok=True)

    catalog_seed = int(cfg["seed"])
    rng_seed = int(args.seed) if args.seed is not None else catalog_seed
    rng = random.Random(rng_seed)

    # Catalog stays deterministic across seed runs (single shared "world").
    # Lives at the parent of out_dir so all seeded runs reuse it.
    parent_dir = out_dir.parent if out_dir.parent != Path("") else out_dir
    catalog_path = parent_dir / cfg["catalog_filename"]

    if not catalog_path.exists():
        catalog = generate_catalog(n=cfg["catalog_size"], seed=catalog_seed)
        save_catalog(catalog, catalog_path)
        print(f"[gen] Wrote catalog to {catalog_path}")
    else:
        with open(catalog_path) as f:
            catalog = json.load(f)
        print(f"[gen] Loaded existing catalog from {catalog_path}")

    levels = _allocate_levels(n_trajectories, cfg["level_distribution"])
    rng.shuffle(levels)

    done = _existing_trajectory_ids(trajectories_path) if args.resume else set()
    if done:
        print(f"[gen] Resume: skipping {len(done)} existing trajectories")

    print(f"[gen] Loading model: {cfg['model']}")
    model, tok = load_model(cfg["model"], dtype=cfg["dtype"])
    print(f"[gen] Model loaded. Beginning generation of {n_trajectories} trajectories "
          f"(catalog_seed={catalog_seed}, rng_seed={rng_seed}).")

    t_start = time.time()
    n_done = 0
    n_attempted = 0
    n_dropped = 0

    with open(trajectories_path, "a") as f_out:
        for i, level in enumerate(tqdm(levels, desc="trajectories")):
            traj_id = f"traj_{i:04d}"
            if traj_id in done:
                continue

            n_attempted += 1
            try:
                query = sample_query(
                    catalog, n_constraints=cfg["n_constraints"], rng=rng,
                    query_id=f"query_{i:04d}",
                )
                traj = run_trajectory(
                    model, tok,
                    catalog=catalog,
                    query=query,
                    perturbation_level=level,
                    rng=rng,
                    trajectory_id=traj_id,
                    max_steps=cfg["max_steps"],
                    capture_activations=cfg["capture_activations"],
                    activations_dir=activations_dir,
                )
            except Exception as e:
                print(f"[gen] Trajectory {traj_id} crashed: {e!r}")
                n_dropped += 1
                continue

            # Drop trajectories that hit the step cap without purchasing (§5.4).
            if traj["final_action"] is None:
                n_dropped += 1
                continue

            compute_all_labels(traj)

            # Strip non-JSON-serializable extras (raw_assistant kept as string is fine).
            f_out.write(json.dumps(traj, default=str) + "\n")
            f_out.flush()
            n_done += 1

            # Rate snapshot every 5.
            if n_done % 5 == 0:
                elapsed = time.time() - t_start
                rate = n_done / max(elapsed, 1e-6)
                eta = (n_trajectories - n_done) / max(rate, 1e-6)
                print(
                    f"[gen] done={n_done} dropped={n_dropped} "
                    f"rate={rate*60:.2f}/min eta={eta/3600:.2f}h"
                )

    print(f"[gen] Finished. n_done={n_done} n_dropped={n_dropped} "
          f"n_attempted={n_attempted}")
    print(f"[gen] Trajectories: {trajectories_path}")
    print(f"[gen] Activations:  {activations_dir}")


if __name__ == "__main__":
    main()
