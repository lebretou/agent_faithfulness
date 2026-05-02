"""Catalog generation. project_plan.md section 5.1."""
from __future__ import annotations

import json
import random
from pathlib import Path

from .schema import CATEGORIES, COLORS, SIZES, MATERIALS, BRANDS, PRICE_MIN, PRICE_MAX


def generate_catalog(n: int = 200, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for i in range(n):
        items.append(
            {
                "item_id": f"item_{i:04d}",
                "category": rng.choice(CATEGORIES),
                "color": rng.choice(COLORS),
                "size": rng.choice(SIZES),
                "price": round(rng.uniform(PRICE_MIN, PRICE_MAX), 2),
                "material": rng.choice(MATERIALS),
                "brand": rng.choice(BRANDS),
            }
        )
    return items


def save_catalog(catalog: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(catalog, f, indent=2)


def load_catalog(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)
