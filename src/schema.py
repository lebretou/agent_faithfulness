"""Single source of truth for the synthetic shopping schema."""
from __future__ import annotations

CATEGORIES = ["shirt", "pants", "shoes", "jacket", "hat", "bag", "watch", "sunglasses"]
COLORS = ["red", "blue", "green", "black", "white", "yellow", "purple", "gray", "brown", "pink"]
SIZES = ["XS", "S", "M", "L", "XL"]
MATERIALS = ["cotton", "polyester", "leather", "wool", "denim", "nylon"]
BRANDS = [f"Brand{c}" for c in "ABCDEFGHIJKL"]

PRICE_MIN = 10.0
PRICE_MAX = 100.0

# Categorical attributes only — price is handled separately everywhere.
SCHEMA: dict[str, list[str]] = {
    "category": CATEGORIES,
    "color": COLORS,
    "size": SIZES,
    "material": MATERIALS,
    "brand": BRANDS,
}

ATTRIBUTES = list(SCHEMA.keys())  # ["category", "color", "size", "material", "brand"]
