from src.catalog import generate_catalog
from src.schema import CATEGORIES, COLORS, SIZES, MATERIALS, BRANDS, PRICE_MIN, PRICE_MAX


def test_reproducible_with_same_seed():
    a = generate_catalog(n=200, seed=42)
    b = generate_catalog(n=200, seed=42)
    assert a == b


def test_size_and_id_format():
    cat = generate_catalog(n=200, seed=42)
    assert len(cat) == 200
    assert cat[0]["item_id"] == "item_0000"
    assert cat[-1]["item_id"] == "item_0199"


def test_all_fields_in_vocab():
    cat = generate_catalog(n=200, seed=42)
    for item in cat:
        assert item["category"] in CATEGORIES
        assert item["color"] in COLORS
        assert item["size"] in SIZES
        assert item["material"] in MATERIALS
        assert item["brand"] in BRANDS
        assert PRICE_MIN <= item["price"] <= PRICE_MAX


def test_different_seeds_differ():
    a = generate_catalog(n=50, seed=1)
    b = generate_catalog(n=50, seed=2)
    assert a != b
