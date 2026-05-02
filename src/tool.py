"""Search tool. project_plan.md section 5.3."""
from __future__ import annotations


def item_satisfies_constraints(item: dict, constraints: dict) -> bool:
    for k, v in constraints.items():
        if k == "price_max":
            if item["price"] > v:
                return False
        else:
            if item.get(k) != v:
                return False
    return True


def _match_score(item: dict, constraints: dict) -> int:
    """Count of constraints this item satisfies."""
    score = 0
    for k, v in constraints.items():
        if k == "price_max":
            if item["price"] <= v:
                score += 1
        else:
            if item.get(k) == v:
                score += 1
    return score


def search(catalog: list[dict], query: dict, k: int = 5) -> list[dict]:
    """Return up to k catalog items, ranked by match score then by price ascending.

    `query` may be either a query dict (with a "constraints" field) or the
    constraints dict itself, to make this convenient for both agent calls and
    direct internal use.
    """
    constraints = query.get("constraints", query) if isinstance(query, dict) else {}
    if not constraints:
        return []

    scored = [(_match_score(it, constraints), it["price"], it) for it in catalog]
    # Higher score first; lower price first.
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [it for _, _, it in scored[:k]]
