import numpy as np
import pytest

from src.steering import probe_direction_from_weights, contrastive_direction, cosine


def test_probe_direction_unit_norm():
    w = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    d = probe_direction_from_weights(w)
    np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-6)
    np.testing.assert_allclose(d, np.array([0.6, 0.8, 0.0]), atol=1e-6)


def test_probe_direction_zero_raises():
    with pytest.raises(ValueError):
        probe_direction_from_weights(np.zeros(8))


def test_contrastive_direction_recovers_known_offset():
    rng = np.random.default_rng(0)
    n_per_class = 200
    n_layers = 3
    hidden = 16
    base = rng.normal(0, 1, size=(2 * n_per_class, n_layers, hidden)).astype(np.float32)
    labels = np.array([0] * n_per_class + [1] * n_per_class)
    true_dir = np.zeros(hidden, dtype=np.float32); true_dir[0] = 1.0
    base[labels == 1, 1, :] += 2.0 * true_dir
    d = contrastive_direction(base, labels, layer_idx=1)
    assert cosine(d, true_dir) > 0.9
    np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-6)


def test_contrastive_direction_other_layers_random():
    rng = np.random.default_rng(0)
    n_per_class = 100
    base = rng.normal(0, 1, size=(2 * n_per_class, 3, 16)).astype(np.float32)
    labels = np.array([0] * n_per_class + [1] * n_per_class)
    base[labels == 1, 1, 0] += 2.0  # only layer 1 has a real signal
    d_clean = contrastive_direction(base, labels, layer_idx=0)
    # Cosine with a known direction at a noise layer should be near zero.
    known = np.zeros(16, dtype=np.float32); known[0] = 1.0
    assert abs(cosine(d_clean, known)) < 0.4


def test_cosine_basic():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(cosine(a, b)) < 1e-9
    assert abs(cosine(a, a) - 1.0) < 1e-9


def test_make_steering_hook_prefill_fires():
    """Multi-token prefill input: hook fires."""
    torch = pytest.importorskip("torch")
    from src.steering import make_steering_hook
    hidden = torch.zeros(1, 4, 8)  # (batch, seq=4, hidden)
    direction = torch.ones(8)
    hook = make_steering_hook(direction, alpha=2.0, position=-1, prefill_only=True)
    out = hook(None, None, (hidden,))
    new_hidden = out[0]
    assert torch.allclose(new_hidden[0, -1, :], torch.full((8,), 2.0))
    assert torch.allclose(new_hidden[0, 0, :], torch.zeros(8))


def test_make_steering_hook_skips_generation_step_when_prefill_only():
    """Single-token cached generation step: prefill_only hook should no-op."""
    torch = pytest.importorskip("torch")
    from src.steering import make_steering_hook
    hidden = torch.zeros(1, 1, 8)  # (batch, seq=1, hidden) — autoregressive step
    direction = torch.ones(8)
    hook = make_steering_hook(direction, alpha=2.0, position=-1, prefill_only=True)
    out = hook(None, None, (hidden,))
    new_hidden = out[0]
    # Hook should leave it untouched.
    assert torch.allclose(new_hidden[0, 0, :], torch.zeros(8))


def test_make_steering_hook_continuous_mode():
    """prefill_only=False: hook fires on single-token steps too."""
    torch = pytest.importorskip("torch")
    from src.steering import make_steering_hook
    hidden = torch.zeros(1, 1, 8)
    direction = torch.ones(8)
    hook = make_steering_hook(direction, alpha=2.0, position=-1, prefill_only=False)
    out = hook(None, None, (hidden,))
    new_hidden = out[0]
    assert torch.allclose(new_hidden[0, 0, :], torch.full((8,), 2.0))
