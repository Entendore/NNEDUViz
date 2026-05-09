"""Model factory and utility functions (PyTorch only)."""

import numpy as np
from typing import List, Tuple, Optional, Any
from models_pytorch import MLP, CNN, RNN, LSTM, Transformer, Generator, Discriminator


def make_model(architecture: str, layers: Optional[List[int]] = None, dropout: float = 0.0) -> Any:
    default_layers = layers or [2, 64, 32, 2]
    models = {
        "mlp": lambda: MLP(default_layers, dropout),
        "cnn": lambda: CNN(dropout),
        "rnn": lambda: RNN(),
        "lstm": lambda: LSTM(),
        "transformer": lambda: Transformer(),
        "gan": lambda: (Generator(), Discriminator()),
    }
    if architecture not in models:
        raise ValueError(f"Unknown architecture: {architecture}")
    return models[architecture]()


def count_parameters(model: Any) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def simple_pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    centered = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:n_components].T


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[int(true)][int(pred)] += 1
    return cm


def grid_2d(X: np.ndarray, resolution: int = 60, margin: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    return gx, gy, np.c_[gx.ravel(), gy.ravel()].astype(np.float32)