"""Remaining life by integrating dN = da / G(a) from the initial depth to failure.

G(a) is the growth per cycle (constant amplitude) or per repeating load block
(spectrum without interaction); either way it depends on crack depth only, so a
single quadrature over a geometric grid of depths gives the life, vectorised
over all samples. Failure is Kmax >= Kc or the geometric limit; a crack whose
growth rate is zero anywhere before failure arrests (infinite life).
"""
import numpy as np

from . import cases, nasgro


def depth_grid(a_i, a_lim, n_grid):
    """Geometric grid per sample: edges (n, n_grid+1), midpoints and widths."""
    u = np.linspace(0.0, 1.0, n_grid + 1)
    a_i = np.minimum(a_i, 0.999 * a_lim)
    edges = a_i[:, None] * (a_lim / a_i)[:, None] ** u
    mid = np.sqrt(edges[:, 1:] * edges[:, :-1])
    return edges, mid, np.diff(edges, axis=1)


def integrate(rate, kpeak, Kc, da, cycles_per_step=1.0):
    """Life from per-depth growth `rate` (per step) on midpoints with widths da.

    kpeak: the largest Kmax at each depth (for the Kc check).
    Returns life in cycles (inf where the crack arrests).
    """
    failed_here = kpeak >= Kc[:, None]
    stop = np.where(failed_here.any(axis=1), failed_here.argmax(axis=1), rate.shape[1])
    idx = np.arange(rate.shape[1])[None, :]
    live = idx < stop[:, None]
    arrested = (live & (rate <= 0.0)).any(axis=1)
    with np.errstate(divide="ignore"):
        dn = np.where(live, da / np.where(rate > 0, rate, 1.0), 0.0)
    life = np.maximum(cycles_per_step * dn.sum(axis=1), 1.0)   # >= 1 cycle if critical at start
    return np.where(arrested, np.inf, life)


def chunked(fn, s, size=1000):
    """Apply fn(sub_s) -> lives over sample chunks to bound memory."""
    n = s["a_i"].size
    return np.concatenate([fn({k: v[i:i + size] for k, v in s.items()})
                           for i in range(0, n, size)])


def ca_life(case, s, k_res_fn=None, n_grid=400):
    """Constant-amplitude remaining life, optionally with a residual K_res(a).

    k_res_fn(s, a) -> K_res on the grid (MPa sqrt(m)); it is added to both Kmax
    and Kmin, so dK is unchanged and the effective R shifts.
    """
    _, a, da = depth_grid(s["a_i"], cases.a_limit(case, s), n_grid)
    smax = cases.max_stress(case, s)[:, None]
    k1 = cases.k_unit(case, s, a)
    kmax = k1 * smax
    kmin = kmax * s["R"][:, None]
    if k_res_fn is not None:
        kr = k_res_fn(s, a)
        kmax, kmin = kmax + kr, kmin + kr
    rate = nasgro.growth_rate(kmax, kmin, a, cases.material(s))
    return integrate(rate, kmax, s["Kc"], da)


def u90(life):
    """Width in decades of the central 90 % of finite lives."""
    x = np.log10(life[np.isfinite(life)])
    if x.size < 20:
        return np.nan
    q5, q95 = np.percentile(x, [5, 95])
    return q95 - q5


def stats(life):
    fin = np.isfinite(life)
    x = np.log10(life[fin])
    return dict(mu=float(np.mean(x)) if x.size else np.nan,
                sigma=float(np.std(x)) if x.size else np.nan,
                u90=float(u90(life)), pf=float(fin.mean()),
                median=float(np.median(x)) if x.size else np.nan)


def u90_reduction(life_fn, inputs):
    """Fractional reduction of U90 when each input in turn is known exactly.

    life_fn(fixed=tuple) -> lives with common random numbers.
    Returns (base stats, {display name: reduction}).
    """
    base = life_fn(fixed=())
    u0 = u90(base)
    out = {}
    for disp, key in inputs.items():
        out[disp] = 1.0 - u90(life_fn(fixed=(key,))) / u0
    return stats(base), out
