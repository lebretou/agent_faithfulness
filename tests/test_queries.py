import random

from src.catalog import generate_catalog
from src.queries import sample_query, sample_queries, render_natural_language
from src.tool import item_satisfies_constraints


def test_sampled_query_is_satisfiable():
    catalog = generate_catalog(n=200, seed=42)
    rng = random.Random(0)
    for _ in range(20):
        q = sample_query(catalog, n_constraints=3, rng=rng)
        assert any(item_satisfies_constraints(it, q["constraints"]) for it in catalog)


def test_constraint_count_matches():
    catalog = generate_catalog(n=200, seed=42)
    rng = random.Random(0)
    for n in (1, 2, 3, 4):
        q = sample_query(catalog, n_constraints=n, rng=rng)
        assert len(q["constraints"]) == n


def test_sample_queries_ten():
    catalog = generate_catalog(n=200, seed=42)
    rng = random.Random(0)
    qs = sample_queries(catalog, n_queries=10, n_constraints=3, rng=rng)
    assert len(qs) == 10
    ids = {q["query_id"] for q in qs}
    assert len(ids) == 10


def test_natural_language_nonempty():
    rng = random.Random(0)
    catalog = generate_catalog(n=200, seed=42)
    q = sample_query(catalog, n_constraints=3, rng=rng)
    nl = render_natural_language(q["constraints"])
    assert isinstance(nl, str) and len(nl) > 5
    assert nl.endswith(".")
