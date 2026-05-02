"""Tool-result perturbation. project_plan.md section 6."""
from __future__ import annotations

import copy
import random

from .schema import SCHEMA


def perturb_tool_result(
    result: list[dict],
    level: int,
    query_constraints: dict,
    rng: random.Random,
) -> tuple[list[dict], str | None, tuple | None]:
    """Apply a Level 0/1/2 perturbation to the top-ranked tool result.

    Returns (perturbed_result, perturbed_attribute, (original_value, new_value)).
    For Level 0 returns (result, None, None) unchanged.
    """
    if level == 0:
        return result, None, None
    if not result:
        return result, None, None

    top_item = copy.deepcopy(result[0])

    constraint_categorical = [a for a in query_constraints if a in SCHEMA]
    has_price_constraint = "price_max" in query_constraints

    if level == 1:
        non_constraint_attrs = [a for a in SCHEMA if a not in query_constraints]
        if not non_constraint_attrs:
            return result, None, None
        attr = rng.choice(non_constraint_attrs)
        original = top_item[attr]
        candidates = [v for v in SCHEMA[attr] if v != original]
        new_value = rng.choice(candidates)
        top_item[attr] = new_value

    elif level == 2:
        # Pool of constraint attributes we know how to violate.
        violatable = list(constraint_categorical)
        if has_price_constraint:
            violatable.append("price")
        if not violatable:
            return result, None, None
        attr = rng.choice(violatable)
        if attr == "price":
            original = top_item["price"]
            price_max = query_constraints["price_max"]
            new_value = round(price_max + rng.uniform(10.0, 30.0), 2)
            top_item["price"] = new_value
        else:
            original = top_item[attr]
            candidates = [v for v in SCHEMA[attr] if v != original]
            new_value = rng.choice(candidates)
            top_item[attr] = new_value
    else:
        raise ValueError(f"Unknown perturbation level: {level}")

    perturbed_result = [top_item] + [copy.deepcopy(it) for it in result[1:]]
    return perturbed_result, attr, (original, new_value)
