"""Per-layer linear probes. project_plan.md sections 8.

Pure CPU. Reads bf16 activation tensors saved by src.activations and trains
a per-layer logistic regression with 5-fold cross-validated AUC.

Two classifiers per project_plan.md §8:
- Classifier A (sanity): Level 0 vs Level 2.
- Classifier B (central): Level 1 vs Level 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


@dataclass
class ProbeResult:
    classifier: str
    layer_aucs: list[float]
    n_layers: int
    n_samples: int
    n_pos: int  # samples with label 1
    n_neg: int  # samples with label 0
    best_layer: int
    best_auc: float
    best_layer_weights: list[float] = field(default_factory=list)  # (hidden_dim,)

    def to_dict(self) -> dict:
        return {
            "classifier": self.classifier,
            "layer_aucs": list(self.layer_aucs),
            "n_layers": self.n_layers,
            "n_samples": self.n_samples,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "best_layer": self.best_layer,
            "best_auc": self.best_auc,
            "best_layer_weights": list(self.best_layer_weights),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProbeResult":
        return cls(**d)


def load_activations(
    paths: list[str | Path],
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Stack a list of .pt activation files into (n_samples, n_layers, hidden_dim) float32.

    Lazy import of torch — only Colab/probe-training environment has it.
    """
    import torch  # noqa

    arrs = []
    for p in paths:
        t = torch.load(p, map_location="cpu", weights_only=True)
        # bf16 on disk; cast to float32 for sklearn.
        a = t.to(torch.float32).numpy()
        if expected_shape is not None and a.shape != expected_shape:
            raise ValueError(f"Activation {p} shape {a.shape} != expected {expected_shape}")
        arrs.append(a)
    return np.stack(arrs, axis=0)


def train_per_layer_probes(
    activations: np.ndarray,
    labels: np.ndarray,
    classifier_name: str,
    n_folds: int = 5,
    C: float = 1.0,
    max_iter: int = 1000,
    seed: int = 42,
) -> ProbeResult:
    """Train a per-layer logistic regression and return AUC by layer + best-layer weights.

    activations: (n_samples, n_layers, hidden_dim)
    labels: (n_samples,) binary {0, 1}
    """
    if activations.ndim != 3:
        raise ValueError(f"activations must be 3-D, got {activations.shape}")
    n_samples, n_layers, hidden_dim = activations.shape
    if labels.shape != (n_samples,):
        raise ValueError(
            f"labels shape {labels.shape} doesn't match n_samples={n_samples}"
        )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    layer_aucs = []
    for layer_idx in range(n_layers):
        X = activations[:, layer_idx, :].astype(np.float32)
        y = labels.astype(np.int64)
        clf = LogisticRegression(
            max_iter=max_iter, C=C, solver="liblinear", random_state=seed
        )
        scores = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")
        layer_aucs.append(float(scores.mean()))

    best_layer = int(np.argmax(layer_aucs))
    best_auc = layer_aucs[best_layer]

    # Refit at best layer on the full data to extract probe direction for steering.
    X_best = activations[:, best_layer, :].astype(np.float32)
    final = LogisticRegression(max_iter=max_iter, C=C, solver="liblinear", random_state=seed)
    final.fit(X_best, labels)
    weights = final.coef_.flatten().tolist()

    return ProbeResult(
        classifier=classifier_name,
        layer_aucs=layer_aucs,
        n_layers=n_layers,
        n_samples=n_samples,
        n_pos=int((labels == 1).sum()),
        n_neg=int((labels == 0).sum()),
        best_layer=best_layer,
        best_auc=best_auc,
        best_layer_weights=weights,
    )


def save_probe_results(results: list[ProbeResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)


def load_probe_results(path: str | Path) -> list[ProbeResult]:
    with open(path) as f:
        return [ProbeResult.from_dict(d) for d in json.load(f)]
