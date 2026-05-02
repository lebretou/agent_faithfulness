"""Query templates and rejection-sampled query generation. project_plan.md 5.2."""
from __future__ import annotations

import random

from .schema import SCHEMA, ATTRIBUTES, PRICE_MIN, PRICE_MAX
from .tool import item_satisfies_constraints


# Categorical attrs eligible to be a query constraint, plus a synthetic "price_max" slot.
_CONSTRAINT_KEYS = ATTRIBUTES + ["price_max"]


def _sample_constraint_keys(n: int, rng: random.Random) -> list[str]:
    if n > len(_CONSTRAINT_KEYS):
        raise ValueError(f"n_constraints={n} > available constraint slots {len(_CONSTRAINT_KEYS)}")
    return rng.sample(_CONSTRAINT_KEYS, n)


def _sample_constraint_values(keys: list[str], rng: random.Random) -> dict:
    out = {}
    for k in keys:
        if k == "price_max":
            out[k] = round(rng.uniform(PRICE_MIN + 5, PRICE_MAX - 5), 2)
        else:
            out[k] = rng.choice(SCHEMA[k])
    return out


def _is_satisfiable(catalog: list[dict], constraints: dict) -> bool:
    return any(item_satisfies_constraints(it, constraints) for it in catalog)


def render_natural_language(constraints: dict) -> str:
    parts = []
    color = constraints.get("color")
    category = constraints.get("category")
    size = constraints.get("size")
    material = constraints.get("material")
    brand = constraints.get("brand")
    price_max = constraints.get("price_max")

    descriptors = []
    if color:
        descriptors.append(color)
    if size:
        descriptors.append(f"size {size}")
    if material:
        descriptors.append(material)

    head = " ".join(descriptors).strip()
    noun = category if category else "item"
    if head:
        phrase = f"a {head} {noun}"
    else:
        phrase = f"an {noun}" if noun[0] in "aeiou" else f"a {noun}"

    suffix = []
    if brand:
        suffix.append(f"from {brand}")
    if price_max is not None:
        suffix.append(f"under ${price_max:.0f}")

    out = f"Find me {phrase}"
    if suffix:
        out += " " + " ".join(suffix)
    return out + "."


def sample_query(
    catalog: list[dict],
    n_constraints: int,
    rng: random.Random,
    query_id: str | None = None,
    max_attempts: int = 200,
) -> dict:
    """Rejection-sampled query: guaranteed satisfiable against `catalog`."""
    for _ in range(max_attempts):
        keys = _sample_constraint_keys(n_constraints, rng)
        constraints = _sample_constraint_values(keys, rng)
        if _is_satisfiable(catalog, constraints):
            return {
                "query_id": query_id or "query_unknown",
                "constraints": constraints,
                "natural_language": render_natural_language(constraints),
            }
    raise RuntimeError(
        f"Could not sample a satisfiable query in {max_attempts} attempts "
        f"with n_constraints={n_constraints}."
    )


def sample_queries(
    catalog: list[dict],
    n_queries: int,
    n_constraints: int,
    rng: random.Random,
) -> list[dict]:
    return [
        sample_query(catalog, n_constraints, rng, query_id=f"query_{i:04d}")
        for i in range(n_queries)
    ]
