"""Dataset generators — all return (X, y) as NumPy arrays."""

import numpy as np


def make_circles(n=500, noise=0.1):
    a = np.random.uniform(0, 2 * np.pi, n)
    n2 = n // 2
    ri = 0.5 + np.random.randn(n2) * noise
    ro = 1.0 + np.random.randn(n - n2) * noise
    Xi = np.c_[ri * np.cos(a[:n2]), ri * np.sin(a[:n2])]
    Xo = np.c_[ro * np.cos(a[n2:]), ro * np.sin(a[n2:])]
    X = np.vstack([Xi, Xo])
    y = np.hstack([np.zeros(n2), np.ones(n - n2)])
    return np.float32(X), np.int64(y)


def make_spirals(n=500, noise=0.15):
    n2 = n // 2
    t = np.linspace(0, 3 * np.pi, n2)
    X1 = np.c_[np.cos(t) * t / 3, np.sin(t) * t / 3] + np.random.randn(n2, 2) * noise
    X2 = np.c_[np.cos(t + np.pi) * t / 3, np.sin(t + np.pi) * t / 3] + np.random.randn(n2, 2) * noise
    return np.float32(np.vstack([X1, X2])), np.int64(np.hstack([np.zeros(n2), np.ones(n2)]))


def make_xor(n=500, noise=0.15):
    cs = [(-1, -1), (1, 1), (-1, 1), (1, -1)]
    X, Y = [], []
    for i, (cx, cy) in enumerate(cs):
        ni = n // 4
        X.append(np.c_[np.full(ni, cx) + np.random.randn(ni) * noise,
                        np.full(ni, cy) + np.random.randn(ni) * noise])
        Y.append(np.full(ni, i % 2, dtype=int))
    return np.float32(np.vstack(X)), np.int64(np.concatenate(Y))


def make_moons(n=500, noise=0.15):
    n2 = n // 2
    a = np.linspace(0, np.pi, n2)
    X1 = np.c_[np.cos(a), np.sin(a)] + np.random.randn(n2, 2) * noise
    X2 = np.c_[1 - np.cos(a), 0.5 - np.sin(a)] + np.random.randn(n2, 2) * noise
    return np.float32(np.vstack([X1, X2])), np.int64(np.hstack([np.zeros(n2), np.ones(n2)]))


def make_sine(seq_len=30, n=300):
    X, Y = [], []
    for _ in range(n):
        ph = np.random.uniform(0, 2 * np.pi)
        fr = np.random.uniform(0.5, 1.5)
        t = np.linspace(0, 4 * np.pi, seq_len + 1)
        s = np.sin(fr * t + ph)
        X.append(s[:-1])
        Y.append(s[1:])
    return np.float32(np.array(X))[..., None], np.float32(np.array(Y))[..., None]


def make_gan_data(n=500, mode="ring"):
    if mode == "ring":
        a = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.random.randn(n) * 0.1
        r = 1 + np.random.randn(n) * 0.08
        return np.float32(np.c_[r * np.cos(a), r * np.sin(a)])
    elif mode == "spiral":
        t = np.linspace(0, 3 * np.pi, n)
        d = np.c_[t * np.cos(t), t * np.sin(t)] * 0.1 + np.random.randn(n, 2) * 0.05
        return np.float32((d - d.mean(0)) / d.std(0))
    else:
        cs = [(-1, -1), (1, 1), (-1, 1), (1, -1)]
        X = [cs[i % 4] + np.random.randn(2) * 0.3 for i in range(n)]
        return np.float32(np.array(X))


def make_signals(n=500, sl=32):
    X, Y = [], []
    for _ in range(n):
        c = np.random.randint(3)
        t = np.linspace(0, 4 * np.pi, sl)
        if c == 0:
            s = np.sin(t)
        elif c == 1:
            s = np.sin(2 * t)
        else:
            s = np.sin(t) * np.cos(t)
        X.append(s + np.random.randn(sl) * 0.2)
        Y.append(c)
    return np.float32(np.array(X))[..., None], np.int64(np.array(Y))


MLP_DATASETS = {
    "circles": make_circles,
    "spirals": make_spirals,
    "xor": make_xor,
    "moons": make_moons,
}