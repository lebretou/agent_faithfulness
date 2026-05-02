import numpy as np

from src.probes import train_per_layer_probes, save_probe_results, load_probe_results


def _synthetic(n_per_class=80, n_layers=4, hidden=32, separable_layer=2, seed=0):
    """One layer is informative; others are pure noise."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_class
    acts = rng.normal(0, 1, size=(n, n_layers, hidden)).astype(np.float32)
    labels = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int64)
    # Add a class-conditional shift only at `separable_layer`.
    direction = rng.normal(0, 1, size=hidden).astype(np.float32)
    direction /= np.linalg.norm(direction)
    acts[labels == 1, separable_layer, :] += 1.5 * direction
    # shuffle
    perm = rng.permutation(n)
    return acts[perm], labels[perm]


def test_probe_finds_separable_layer():
    acts, labels = _synthetic(separable_layer=2)
    res = train_per_layer_probes(acts, labels, classifier_name="test")
    assert res.best_layer == 2, f"Expected best_layer=2, got {res.best_layer}"
    assert res.best_auc > 0.75
    # Noise layers should be far from the separable layer's AUC.
    for noise_layer in (0, 1, 3):
        assert res.layer_aucs[noise_layer] < res.best_auc - 0.15, (
            f"layer {noise_layer} AUC={res.layer_aucs[noise_layer]:.3f} too close to best"
        )


def test_probe_returns_full_metadata():
    acts, labels = _synthetic()
    res = train_per_layer_probes(acts, labels, classifier_name="test")
    assert res.n_samples == acts.shape[0]
    assert res.n_layers == acts.shape[1]
    assert res.n_pos + res.n_neg == res.n_samples
    assert len(res.layer_aucs) == res.n_layers
    assert len(res.best_layer_weights) == acts.shape[2]


def test_probe_serialization_roundtrip(tmp_path):
    acts, labels = _synthetic()
    res = train_per_layer_probes(acts, labels, classifier_name="test")
    path = tmp_path / "probes.json"
    save_probe_results([res], path)
    loaded = load_probe_results(path)
    assert len(loaded) == 1
    assert loaded[0].best_layer == res.best_layer
    assert abs(loaded[0].best_auc - res.best_auc) < 1e-9
    assert loaded[0].layer_aucs == res.layer_aucs


def test_probe_rejects_bad_shape():
    import pytest
    acts = np.random.randn(20, 32).astype(np.float32)  # 2-D, not 3-D
    labels = np.zeros(20, dtype=np.int64)
    with pytest.raises(ValueError):
        train_per_layer_probes(acts, labels, classifier_name="test")
