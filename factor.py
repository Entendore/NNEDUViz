"""Model factory and shared utility functions."""

import numpy as np
from models_tensorflow import TF_AVAILABLE


def make_model(ntype, framework, layers=None, dropout=0.0):
    """Instantiate a model for the given architecture and framework."""
    if framework == "pytorch":
        import models_pytorch as pt
        if ntype == "mlp":
            return pt.PT_MLP(layers or (2, 64, 32, 2), dropout)
        if ntype == "cnn":
            return pt.PT_CNN(dropout)
        if ntype == "rnn":
            return pt.PT_RNN()
        if ntype == "lstm":
            return pt.PT_LSTM()
        if ntype == "transformer":
            return pt.PT_Transformer()
        if ntype == "gan":
            return pt.PT_Gen(), pt.PT_Dis()

    elif framework == "tensorflow" and TF_AVAILABLE:
        import models_tensorflow as tf_m
        if ntype == "mlp":
            return tf_m.TF_MLP(layers or (2, 64, 32, 2), dropout)
        if ntype == "cnn":
            return tf_m.TF_CNN(dropout)
        if ntype == "rnn":
            return tf_m.TF_RNN()
        if ntype == "lstm":
            return tf_m.TF_LSTM()
        if ntype == "transformer":
            return tf_m.TF_Transformer()
        if ntype == "gan":
            return tf_m.TF_Gen(), tf_m.TF_Dis()
    return None


def count_params(model, framework):
    """Count trainable parameters."""
    if framework == "pytorch":
        return sum(p.numel() for p in model.parameters())
    elif framework == "tensorflow" and TF_AVAILABLE:
        return sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    return 0


def simple_pca(X, nc=2):
    """Simple PCA via SVD (no sklearn needed)."""
    c = X - X.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    return c @ Vt[:nc].T


def conf_matrix(yt, yp, nc):
    """Build confusion matrix."""
    cm = np.zeros((nc, nc), dtype=int)
    for t, p in zip(yt, yp):
        cm[int(t)][int(p)] += 1
    return cm


def grid_2d(X, r=60, margin=0.5):
    """Create a 2D meshgrid covering X with given resolution."""
    xn, xx = X[:, 0].min() - margin, X[:, 0].max() + margin
    yn, yx = X[:, 1].min() - margin, X[:, 1].max() + margin
    gx, gy = np.meshgrid(np.linspace(xn, xx, r), np.linspace(yn, yx, r))
    return gx, gy, np.c_[gx.ravel(), gy.ravel()].astype(np.float32)