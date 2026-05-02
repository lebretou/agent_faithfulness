from src.tool import search, item_satisfies_constraints, _match_score


CATALOG = [
    {"item_id": "item_0", "category": "shirt", "color": "blue", "size": "M",
     "price": 20.0, "material": "cotton", "brand": "BrandA"},
    {"item_id": "item_1", "category": "shirt", "color": "blue", "size": "M",
     "price": 15.0, "material": "cotton", "brand": "BrandB"},
    {"item_id": "item_2", "category": "shirt", "color": "red", "size": "M",
     "price": 12.0, "material": "cotton", "brand": "BrandC"},
    {"item_id": "item_3", "category": "pants", "color": "blue", "size": "L",
     "price": 80.0, "material": "denim", "brand": "BrandD"},
]


def test_top_result_has_max_score():
    constraints = {"category": "shirt", "color": "blue", "price_max": 30.0}
    res = search(CATALOG, {"constraints": constraints}, k=4)
    assert _match_score(res[0], constraints) == 3


def test_price_tiebreak_lower_first():
    constraints = {"category": "shirt", "color": "blue", "price_max": 30.0}
    res = search(CATALOG, {"constraints": constraints}, k=4)
    # Both item_0 ($20) and item_1 ($15) satisfy all 3; cheaper goes first.
    assert res[0]["item_id"] == "item_1"
    assert res[1]["item_id"] == "item_0"


def test_respects_k():
    constraints = {"category": "shirt"}
    res = search(CATALOG, {"constraints": constraints}, k=2)
    assert len(res) == 2


def test_item_satisfies_constraints():
    item = CATALOG[0]
    assert item_satisfies_constraints(item, {"category": "shirt", "color": "blue"})
    assert item_satisfies_constraints(item, {"price_max": 25.0})
    assert not item_satisfies_constraints(item, {"price_max": 10.0})
    assert not item_satisfies_constraints(item, {"color": "red"})


def test_search_accepts_bare_constraints_dict():
    constraints = {"category": "shirt", "color": "blue"}
    res = search(CATALOG, constraints, k=3)
    assert len(res) == 3
