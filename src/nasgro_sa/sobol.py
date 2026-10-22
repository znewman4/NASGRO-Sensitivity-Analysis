"""Variance-based (Sobol) sensitivity indices by the pick-freeze method.

Groups of columns are swapped together, so a group of mutually dependent
inputs receives a single index; this is valid provided the groups are
independent of one another. Outputs may be any finite quantity, including
0/1 indicators such as failure within a horizon, which stay well defined
when trajectories arrest or are censored.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


def saltelli_blocks(w_a: np.ndarray, w_b: np.ndarray, groups) -> np.ndarray:
    """Stack A, B and one A_B^(i) matrix per group (A with group i's columns taken from B)."""
    blocks = [w_a, w_b]
    for cols in groups:
        ab = w_a.copy()
        ab[:, cols] = w_b[:, cols]
        blocks.append(ab)
    return np.vstack(blocks)


def _estimate(fa, fb, fab):
    """Saltelli (2010) first-order and Jansen (1999) total-effect estimators.

    Outputs are centred first: the first-order estimator is shift-invariant
    in expectation but its variance grows with the squared mean (e.g. log
    rates far from zero).
    """
    mean = np.mean(np.concatenate([fa, fb]), axis=0)
    fa, fb, fab = fa - mean, fb - mean, fab - mean
    var = np.var(np.concatenate([fa, fb]), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s1 = np.mean(fb * (fab - fa), axis=1) / var
        st = 0.5 * np.mean((fa - fab) ** 2, axis=1) / var
    return s1, st, var


@dataclass
class SobolResult:
    groups: list[str]
    outputs: list[str]
    first: np.ndarray        # (groups, outputs)
    total: np.ndarray
    first_ci: np.ndarray     # (2, groups, outputs); nan without bootstrap
    total_ci: np.ndarray
    mean: np.ndarray         # (outputs,)
    variance: np.ndarray
    n: int

    def table(self, output: int | str = 0) -> str:
        j = self.outputs.index(output) if isinstance(output, str) else output
        lines = [f"{self.outputs[j]}  (mean {self.mean[j]:.4g}, var {self.variance[j]:.4g}, N = {self.n})",
                 f"{'group':<14}{'S1':>8}{'95% CI':>18}{'ST':>8}{'95% CI':>18}"]
        for i in np.argsort(-np.nan_to_num(self.total[:, j], nan=-1.0)):
            lo1, hi1 = self.first_ci[:, i, j]
            lot, hit = self.total_ci[:, i, j]
            lines.append(f"{self.groups[i]:<14}{self.first[i, j]:8.3f}  [{lo1:6.3f}, {hi1:6.3f}]"
                         f"{self.total[i, j]:8.3f}  [{lot:6.3f}, {hit:6.3f}]")
        return "\n".join(lines)


def sobol_indices(y, n: int, groups: list[str], outputs: list[str] | None = None,
                  n_boot: int = 0, rng: np.random.Generator | None = None, level: float = 0.95) -> SobolResult:
    """Indices from outputs evaluated on ``saltelli_blocks`` rows: y has shape (n (k + 2),) or (n (k + 2), m)."""
    k = len(groups)
    y = np.asarray(y, dtype=float)
    y = y.reshape(k + 2, n, -1)
    fa, fb, fab = y[0], y[1], y[2:]
    s1, st, var = _estimate(fa, fb, fab)
    ci1 = cit = np.full((2,) + s1.shape, np.nan)
    if n_boot:
        rng = rng or np.random.default_rng()
        boot1, boott = np.empty((n_boot,) + s1.shape), np.empty((n_boot,) + s1.shape)
        for b in range(n_boot):
            i = rng.integers(0, n, n)
            boot1[b], boott[b], _ = _estimate(fa[i], fb[i], fab[:, i])
        q = [50 * (1 - level), 50 * (1 + level)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # zero-variance outputs give all-nan columns
            ci1, cit = np.nanpercentile(boot1, q, axis=0), np.nanpercentile(boott, q, axis=0)
    outputs = outputs or [f"y{j}" for j in range(y.shape[2])]
    return SobolResult(list(groups), list(outputs), s1, st, ci1, cit,
                       np.mean(np.concatenate([fa, fb]), axis=0), var, n)
