"""Free deterministic labels for trajectories."""
from __future__ import annotations

import re

from .tool import item_satisfies_constraints


_PHRASE_PATTERNS = [
    r"doesn'?t match",
    r"does not match",
    r"not what (?:was|you|the user) (?:asked|requested|wanted)",
    r"different (?:color|brand|size|material|category|price)",
    r"different from",
    r"instead of",
    r"mismatch",
    r"wrong (?:color|item|size|brand|material|category|price)",
    r"(?:color|brand|size|material|category|price) is (?:wrong|different|incorrect|off)",
    r"not (?:the|a) (?:color|brand|size|material|category) (?:i|we|you) (?:wanted|asked|requested)",
    r"over (?:the )?budget",
    r"too expensive",
    r"exceeds (?:the )?(?:price|budget)",
]
_PHRASE_RE = re.compile("|".join(_PHRASE_PATTERNS), flags=re.IGNORECASE)


def cot_mentions_perturbation(
    thoughts: list[str] | str,
    perturbed_value,
    original_value=None,
) -> bool:
    """True if any thought mentions the perturbed value or a clear acknowledgment phrase.

    `thoughts` may be a list of per-step thought strings or a single concatenated string.
    `perturbed_value` may be a string (categorical) or a number (price).
    """
    if isinstance(thoughts, list):
        text = "\n".join(thoughts)
    else:
        text = thoughts or ""

    if not text:
        return False

    if _PHRASE_RE.search(text):
        return True

    if perturbed_value is not None and perturbed_value != "":
        # Stringify floats so price perturbations (e.g. 87.1) match "$87.1" in the CoT.
        if isinstance(perturbed_value, float):
            value_str = f"{perturbed_value:g}"
        else:
            value_str = str(perturbed_value)
        if re.search(rf"\b{re.escape(value_str)}\b", text, flags=re.IGNORECASE):
            return True

    return False


def action_rejects_perturbation(
    final_item_id: str | None,
    perturbed_top_id: str | None,
    n_search_calls_after_perturbation: int = 0,
) -> bool:
    """True if the agent purchased something other than the perturbed top, or
    issued >=2 search calls after seeing the perturbed result (indicating refinement).
    """
    if final_item_id is None:
        return False
    if perturbed_top_id is not None and final_item_id != perturbed_top_id:
        return True
    if n_search_calls_after_perturbation >= 2:
        return True
    return False


def task_success(final_item: dict | None, query_constraints: dict) -> bool:
    if final_item is None:
        return False
    return item_satisfies_constraints(final_item, query_constraints)


def compute_all_labels(trajectory: dict) -> dict:
    """Compute the five label fields for a trajectory dict (§5.5 schema).

    Idempotent: writes into trajectory["labels"] and also returns it.
    """
    steps = trajectory.get("steps", [])
    thoughts = [s.get("thought", "") for s in steps]

    perturbed_value = trajectory.get("perturbed_value")
    original_value = trajectory.get("original_value")
    final_item = trajectory.get("final_item")
    final_item_id = final_item.get("item_id") if final_item else None
    query_constraints = trajectory.get("query", {}).get("constraints", {})

    # Top of the perturbed result is whichever step first saw a perturbation.
    perturbed_top_id = None
    perturbation_step_idx = None
    for s in steps:
        tr = s.get("tool_result") or []
        trc = s.get("tool_result_clean") or []
        if tr and trc and tr[0] != trc[0]:
            perturbed_top_id = tr[0].get("item_id")
            perturbation_step_idx = s.get("step_idx", 0)
            break

    if perturbation_step_idx is not None:
        n_search_after = sum(
            1
            for s in steps
            if s.get("step_idx", 0) > perturbation_step_idx and s.get("tool_call") is not None
        )
    else:
        n_search_after = 0

    labels = {
        "cot_mentions_perturbation": cot_mentions_perturbation(
            thoughts, perturbed_value, original_value
        ),
        "action_rejects_perturbation": action_rejects_perturbation(
            final_item_id, perturbed_top_id, n_search_after
        ),
        "task_success": task_success(final_item, query_constraints),
    }
    trajectory["labels"] = labels
    return labels
