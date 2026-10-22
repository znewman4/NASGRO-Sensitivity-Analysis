"""CDF-based moment-independent sensitivity, used as a robustness check on the Sobol ranking.

For each input group, beta_i = E_{X_i}[ sup_y |F_Y(y) - F_{Y|X_i}(y)| ]
(the Kolmogorov-Smirnov form of Baucells & Borgonovo 2013). It works on the
whole output distribution, so a right-censored RUL can be used directly:
censored samples (inf) simply never enter the CDF. It is estimated from
given data by binning each group's standard-normal coordinates; a dummy
input with random bins gives the estimator's bias floor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KSResult:
    groups: list[str]
    index: np.ndarray    # (groups, outputs)
    dummy: np.ndarray    # (outputs,) bias floor from a random partition
    n: int

    def table(self, output: int = 0, label: str = "") -> str:
        lines = [f"{label}  KS index (dummy floor {self.dummy[output]:.3f}, N = {self.n})",
                 f"{'group':<14}{'beta':>8}"]
        for i in np.argsort(-self.index[:, output]):
            lines.append(f"{self.groups[i]:<14}{self.index[i, output]:8.3f}")
        return "\n".join(lines)


def ks_indices(y, w: np.ndarray, groups, n_bins: int = 16, n_grid: int = 64,
               rng: np.random.Generator | None = None) -> KSResult:
    """y: (n,) or (n, m) outputs (inf allowed); w: (n, d) standard normals; groups: [(label, columns)]."""
    y = np.asarray(y, dtype=float).reshape(len(w), -1)
    n = y.shape[0]
    grids = []
    for col in y.T:
        finite = col[np.isfinite(col)]
        grids.append(np.unique(np.quantile(finite, np.linspace(0, 1, n_grid))) if finite.size else np.zeros(1))
    below = [y[:, j, None] <= g[None, :] for j, g in enumerate(grids)]   # (n, G) per output

    def statistic(labels):
        _, labels = np.unique(labels, return_inverse=True)
        onehot = np.zeros((n, labels.max() + 1))
        onehot[np.arange(n), labels] = 1.0
        counts = onehot.sum(axis=0)
        out = np.empty(len(below))
        for j, ind in enumerate(below):
            conditional = (onehot.T @ ind) / counts[:, None]
            out[j] = np.sum(counts * np.abs(conditional - ind.mean(axis=0)).max(axis=1)) / n
        return out

    def bins(cols):
        per_dim = max(2, int(round(n_bins ** (1.0 / len(cols)))))
        labels = np.zeros(n, dtype=int)
        for c in cols:
            edges = np.quantile(w[:, c], np.linspace(0, 1, per_dim + 1)[1:-1])
            labels = labels * per_dim + np.searchsorted(edges, w[:, c])
        return labels

    rng = rng or np.random.default_rng()
    index = np.array([statistic(bins(cols)) for _, cols in groups])
    dummy = statistic(rng.permutation(np.arange(n) % n_bins))
    return KSResult([g for g, _ in groups], index, dummy, n)
