import random

from src.perturbation import perturb_tool_result
from src.tool import item_satisfies_constraints
from src.schema import SCHEMA


BASE_RESULT = [
    {"item_id": "item_0", "category": "shirt", "color": "blue", "size": "M",
     "price": 20.0, "material": "cotton", "brand": "BrandA"},
    {"item_id": "item_1", "category": "shirt", "color": "blue", "size": "M",
     "price": 22.0, "material": "cotton", "brand": "BrandB"},
]


def test_level_0_is_passthrough():
    rng = random.Random(0)
    out, attr, vals = perturb_tool_result(BASE_RESULT, 0, {"category": "shirt"}, rng)
    assert out == BASE_RESULT
    assert attr is None and vals is None


def test_level_1_still_satisfies_constraints():
    rng = random.Random(0)
    constraints = {"category": "shirt", "color": "blue", "price_max": 30.0}
    for _ in range(50):
        out, attr, vals = perturb_tool_result(BASE_RESULT, 1, constraints, rng)
        assert item_satisfies_constraints(out[0], constraints)
        assert attr not in constraints
        assert attr in SCHEMA
        assert vals[0] != vals[1]


def test_level_2_violates_exactly_one_constraint():
    rng = random.Random(0)
    constraints = {"category": "shirt", "color": "blue", "price_max": 30.0}
    for _ in range(50):
        out, attr, vals = perturb_tool_result(BASE_RESULT, 2, constraints, rng)
        assert not item_satisfies_constraints(out[0], constraints)
        # Count violated constraints — must be exactly 1.
        violated = 0
        for k, v in constraints.items():
            if k == "price_max":
                if out[0]["price"] > v:
                    violated += 1
            else:
                if out[0].get(k) != v:
                    violated += 1
        assert violated == 1, f"expected 1 violation, got {violated} (attr={attr})"


def test_level_2_price_lands_above_max():
    rng = random.Random(0)
    constraints = {"price_max": 30.0}
    saw_price = False
    for _ in range(100):
        out, attr, vals = perturb_tool_result(BASE_RESULT, 2, constraints, rng)
        if attr == "price":
            saw_price = True
            assert out[0]["price"] > 30.0
            assert vals[1] > 30.0
    assert saw_price, "Expected at least one price perturbation across runs"


def test_level_2_preserves_tail_items():
    rng = random.Random(0)
    constraints = {"color": "blue"}
    out, _, _ = perturb_tool_result(BASE_RESULT, 2, constraints, rng)
    assert out[1] == BASE_RESULT[1]
