"""Synthetic dataset generators for neural network visualization."""

import numpy as np
from typing import Tuple


def train_test_split(
    X: np.ndarray, 
    y: np.ndarray, 
    test_ratio: float = 0.2,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, ...)
        Feature matrix
    y : array-like of shape (n_samples,)
        Labels or targets
    test_ratio : float, default 0.2
        Proportion of data to use for testing (0.0 to 0.5)
    stratify : bool, default True
        If True, ensure class proportions are preserved in both sets
    
    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    """
    n = len(X)
    n_test = max(1, int(n * test_ratio))
    
    if stratify and y.ndim == 1:
        # Split each class proportionally
        train_indices, test_indices = [], []
        classes = np.unique(y)
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            np.random.shuffle(cls_indices)
            n_cls_test = max(1, int(len(cls_indices) * test_ratio))
            test_indices.extend(cls_indices[:n_cls_test])
            train_indices.extend(cls_indices[n_cls_test:])
        
        train_idx = np.array(train_indices)
        test_idx = np.array(test_indices)
    else:
        indices = np.random.permutation(n)
        train_idx = indices[n_test:]
        test_idx = indices[:n_test]
    
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    
    if X.ndim == 3:
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def make_circles(n: int = 500, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    n2 = n // 2
    angles = np.random.uniform(0, 2 * np.pi, n)
    r_inner = 0.5 + np.random.randn(n2) * noise
    X_inner = np.c_[r_inner * np.cos(angles[:n2]), r_inner * np.sin(angles[:n2])]
    r_outer = 1.0 + np.random.randn(n - n2) * noise
    X_outer = np.c_[r_outer * np.cos(angles[n2:]), r_outer * np.sin(angles[n2:])]
    X = np.vstack([X_inner, X_outer])
    y = np.hstack([np.zeros(n2), np.ones(n - n2)])
    return np.float32(X), np.int64(y)


def make_spirals(n: int = 500, noise: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    n2 = n // 2
    t = np.linspace(0, 3 * np.pi, n2)
    X1 = np.c_[np.cos(t) * t / 3, np.sin(t) * t / 3] + np.random.randn(n2, 2) * noise
    X2 = np.c_[np.cos(t + np.pi) * t / 3, np.sin(t + np.pi) * t / 3] + np.random.randn(n2, 2) * noise
    return np.float32(np.vstack([X1, X2])), np.int64(np.hstack([np.zeros(n2), np.ones(n2)]))


def make_xor(n: int = 500, noise: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    corners = [(-1, -1), (1, 1), (-1, 1), (1, -1)]
    X_parts, Y_parts = [], []
    for i, (cx, cy) in enumerate(corners):
        ni = n // 4
        X_parts.append(np.c_[np.full(ni, cx) + np.random.randn(ni) * noise,
                             np.full(ni, cy) + np.random.randn(ni) * noise])
        Y_parts.append(np.full(ni, i % 2, dtype=int))
    return np.float32(np.vstack(X_parts)), np.int64(np.concatenate(Y_parts))


def make_moons(n: int = 500, noise: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    n2 = n // 2
    angles = np.linspace(0, np.pi, n2)
    X1 = np.c_[np.cos(angles), np.sin(angles)] + np.random.randn(n2, 2) * noise
    X2 = np.c_[1 - np.cos(angles), 0.5 - np.sin(angles)] + np.random.randn(n2, 2) * noise
    return np.float32(np.vstack([X1, X2])), np.int64(np.hstack([np.zeros(n2), np.ones(n2)]))


def make_checkerboard(n: int = 500, grid_size: int = 3, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.uniform(-grid_size, grid_size, (n, 2))
    cell_x = ((X[:, 0] + grid_size) / (2 * grid_size) * grid_size).astype(int)
    cell_y = ((X[:, 1] + grid_size) / (2 * grid_size) * grid_size).astype(int)
    y = (cell_x + cell_y) % 2
    X += np.random.randn(n, 2) * noise
    return np.float32(X), np.int64(y)


def make_sine(seq_len: int = 30, n: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    for _ in range(n):
        phase = np.random.uniform(0, 2 * np.pi)
        freq = np.random.uniform(0.5, 1.5)
        t = np.linspace(0, 4 * np.pi, seq_len + 1)
        signal = np.sin(freq * t + phase)
        X.append(signal[:-1])
        Y.append(signal[1:])
    return np.float32(np.array(X))[..., None], np.float32(np.array(Y))[..., None]


def make_multi_frequency(seq_len: int = 30, n: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    for _ in range(n):
        t = np.linspace(0, 4 * np.pi, seq_len + 1)
        signal = (0.5 * np.sin(np.random.uniform(0.5, 1.0) * t + np.random.uniform(0, 2 * np.pi)) +
                  0.5 * np.sin(np.random.uniform(1.0, 2.0) * t + np.random.uniform(0, 2 * np.pi)))
        X.append(signal[:-1])
        Y.append(signal[1:])
    return np.float32(np.array(X))[..., None], np.float32(np.array(Y))[..., None]


def make_signals(n: int = 500, seq_len: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    for _ in range(n):
        class_idx = np.random.randint(3)
        t = np.linspace(0, 4 * np.pi, seq_len)
        if class_idx == 0:
            signal = np.sin(t)
        elif class_idx == 1:
            signal = np.sin(2 * t)
        else:
            signal = np.sin(t) * np.cos(t)
        signal += np.random.randn(seq_len) * 0.2
        X.append(signal)
        Y.append(class_idx)
    return np.float32(np.array(X))[..., None], np.int64(np.array(Y))


def make_complex_signals(n: int = 500, seq_len: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    for _ in range(n):
        class_idx = np.random.randint(5)
        t = np.linspace(0, 4 * np.pi, seq_len)
        if class_idx == 0:
            signal = np.sin(t)
        elif class_idx == 1:
            signal = sum(np.sin((2 * k + 1) * t) / (2 * k + 1) for k in range(5))
        elif class_idx == 2:
            signal = sum((-1) ** k * np.sin((k + 1) * t) / (k + 1) for k in range(5))
        elif class_idx == 3:
            signal = np.sin(t * np.linspace(1, 3, seq_len))
        else:
            signal = np.sin(3 * t) * np.exp(-t / (4 * np.pi))
        signal += np.random.randn(seq_len) * 0.15
        X.append(signal)
        Y.append(class_idx)
    return np.float32(np.array(X))[..., None], np.int64(np.array(Y))


def make_gan_data(n: int = 500, mode: str = "ring") -> np.ndarray:
    if mode == "ring":
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        angles += np.random.randn(n) * 0.1
        radius = 1 + np.random.randn(n) * 0.08
        X = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
    elif mode == "spiral":
        t = np.linspace(0, 3 * np.pi, n)
        X = np.c_[t * np.cos(t), t * np.sin(t)] * 0.1 + np.random.randn(n, 2) * 0.05
        X = (X - X.mean(0)) / X.std(0)
    elif mode == "gaussian_mix":
        centers = [(-1, -1), (1, 1), (-1, 1), (1, -1)]
        X = np.array([centers[i % 4] + np.random.randn(2) * 0.3 for i in range(n)])
    elif mode == "grid":
        side = int(np.sqrt(n))
        x = np.linspace(-1.5, 1.5, side)
        y = np.linspace(-1.5, 1.5, side)
        xx, yy = np.meshgrid(x, y)
        X = np.c_[xx.ravel(), yy.ravel()][:n] + np.random.randn(n, 2) * 0.05
    elif mode == "figure8":
        t = np.linspace(0, 2 * np.pi, n)
        denom = 1 + np.sin(t) ** 2
        X = np.c_[np.cos(t) / denom, np.sin(t) * np.cos(t) / denom] * 2 + np.random.randn(n, 2) * 0.05
    else:
        return make_gan_data(n, "ring")
    return np.float32(X)


MLP_DATASETS = {
    "circles": make_circles,
    "spirals": make_spirals,
    "xor": make_xor,
    "moons": make_moons,
    "checkerboard": make_checkerboard,
}

CNN_DATASETS = {
    "signals_3class": make_signals,
    "signals_5class": make_complex_signals,
}

SEQ_DATASETS = {
    "sine": make_sine,
    "multi_freq": make_multi_frequency,
}