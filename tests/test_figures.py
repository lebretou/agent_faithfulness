"""Smoke tests for figure generation. Verifies files write and don't crash."""
import matplotlib

matplotlib.use("Agg")  # no display in tests

from src.figures import probe_auc_by_layer, three_curve_plot, steering_curve


def test_probe_auc_by_layer_single_seed(tmp_path):
    out = tmp_path / "fig1.png"
    probe_auc_by_layer({"A": [0.5, 0.6, 0.8, 0.7]}, out)
    assert out.exists() and out.stat().st_size > 1000


def test_probe_auc_by_layer_multi_seed(tmp_path):
    out = tmp_path / "fig1.png"
    aucs = [
        [0.5, 0.6, 0.85, 0.7],
        [0.5, 0.62, 0.83, 0.71],
        [0.51, 0.59, 0.84, 0.72],
    ]
    probe_auc_by_layer({"B": aucs}, out)
    assert out.exists()


def test_three_curve_plot(tmp_path):
    out = tmp_path / "three.png"
    three_curve_plot(
        levels=[0, 1, 2],
        verb_rate=[[0.17, 0.33, 0.82], [0.18, 0.33, 0.83]],
        rej_rate=[[0.0, 0.17, 0.26], [0.0, 0.16, 0.27]],
        probe_auc_l0_l2=[0.92, 0.93],
        probe_auc_l1_l2=[0.71, 0.69],
        out_path=out,
    )
    assert out.exists()


def test_steering_curve(tmp_path):
    out = tmp_path / "steer.png"
    steering_curve(
        alphas=[0, 0.5, 1, 2, 4],
        verb_rate=[0.82, 0.84, 0.88, 0.93, 0.95],
        rej_rate=[0.26, 0.30, 0.40, 0.55, 0.60],
        clean_success=[0.85, 0.85, 0.84, 0.78, 0.65],
        out_path=out,
    )
    assert out.exists()
