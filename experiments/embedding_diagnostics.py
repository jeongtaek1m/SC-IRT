#!/usr/bin/env python3
"""Collapse diagnostics for a set of embedding vectors (used by the JEPA pre-training check).

The quantity logged during the first JEPA runs pooled the sample axis and the channel axis, which after a
LayerNorm stays near 1 even when every input produces the same vector. These four do not: the across-sample
standard deviation per channel, the total variance, the mean pairwise cosine of the UNCENTRED vectors (1 when
every vector is the same direction), and the effective rank exp(entropy of the normalised covariance spectrum),
whose ceiling is min(d, n - 1). Verified on identical, zero, rank-one and random inputs.
"""
import numpy as np


def embedding_diagnostics(H, eps=1e-12):
    """H: (number of valid samples, embedding dimension)."""
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2 or len(H) < 2 or not np.isfinite(H).all():
        raise ValueError('Need at least two finite embedding vectors.')
    n, d = H.shape
    X = H - H.mean(axis=0, keepdims=True)
    C = X.T @ X / (n - 1)
    ev = np.maximum(np.linalg.eigvalsh(C), 0.0)
    variance = float(ev.sum())
    if variance <= eps:
        effective_rank = 0.0
    else:
        p = ev[ev > 0] / variance
        effective_rank = float(np.exp(-(p * np.log(p)).sum()))
    norms = np.linalg.norm(H, axis=1)
    valid = norms > eps
    V = H[valid] / norms[valid, None]
    nv = len(V)
    mean_cosine = None
    if nv >= 2:
        value = (np.square(V.sum(axis=0)).sum() - np.square(V).sum()) / (nv * (nv - 1))
        mean_cosine = float(np.clip(value, -1.0, 1.0))
    return {'n': n, 'across_sample_std': float(H.std(axis=0, ddof=1).mean()), 'total_variance': variance,
            'mean_pairwise_cosine': mean_cosine, 'effective_rank': effective_rank,
            'max_possible_rank': min(d, n - 1)}


if __name__ == '__main__':                                   # the four cases the function must separate
    rng = np.random.default_rng(0)
    for name, H in (('identical vectors', np.tile(rng.normal(size=64), (200, 1))),
                    ('zero vectors', np.zeros((200, 64))),
                    ('rank one', rng.normal(size=(200, 1)) * rng.normal(size=(1, 64))),
                    ('random', rng.normal(size=(200, 64)))):
        d = embedding_diagnostics(H)
        print(f"{name:18s} std {d['across_sample_std']:.3f}  var {d['total_variance']:.3f}  "
              f"cos {d['mean_pairwise_cosine'] if d['mean_pairwise_cosine'] is None else round(d['mean_pairwise_cosine'], 3)}  "
              f"erank {d['effective_rank']:.2f} / {d['max_possible_rank']}")
