"""Re-compute the free deterministic labels on an existing trajectories.jsonl.

Useful when the verbalization regex evolves, since labels are otherwise
written at generation time.

Usage:
    python scripts/02_compute_labels.py --in data/seed_42/trajectories.jsonl \
                                        --out data/seed_42/trajectories.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.labels import compute_all_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    flips = {"verb": 0, "rej": 0, "succ": 0}

    # Read all, recompute, write atomically.
    new_lines = []
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            old = t.get("labels", {}) or {}
            new = compute_all_labels(t)
            for k_short, k_full in [
                ("verb", "cot_mentions_perturbation"),
                ("rej", "action_rejects_perturbation"),
                ("succ", "task_success"),
            ]:
                if old.get(k_full) != new.get(k_full):
                    flips[k_short] += 1
            new_lines.append(json.dumps(t, default=str))
            n += 1

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write("\n".join(new_lines) + "\n")
    tmp.replace(out_path)

    print(f"Relabeled {n} trajectories.")
    print(f"  flipped cot_mentions_perturbation: {flips['verb']}")
    print(f"  flipped action_rejects_perturbation: {flips['rej']}")
    print(f"  flipped task_success: {flips['succ']}")


if __name__ == "__main__":
    main()
