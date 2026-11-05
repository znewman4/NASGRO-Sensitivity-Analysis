"""How RUL uncertainty falls as repeat crack-depth measurements are added (Bayesian updating).

Set-up, per case, at a 1 mm detected crack (NASGRO-equation inputs only):
  * Prior: the study's industry-typical inputs, including the detection sizing
    (depth +-0.5 mm sd). This is the "no repeat inspections" state of Figure 2.
  * Repeat inspections every `interval` cycles measure the current depth with
    Gaussian error of sd `sigma_m`. The crack has not failed when it is
    inspected, so that is also information.
  * Posterior by importance weighting of a large prior ensemble: every member
    carries its own depth history a(t), so its weight is the likelihood of the
    measurements, prod_j phi((y_j - a_i(t_j)) / sigma_m) * 1[life_i > t_j].
    Remaining life at inspection k is life_i - t_k.
  * "True" cracks are drawn from the same prior on a separate seed, so the
    posterior should be calibrated: its 90 % band should hold the true RUL
    about 90 % of the time (checked and reported).

The model has no cycle-to-cycle randomness: once the inputs are known, growth
is deterministic. Real cracks scatter about their own mean curve (Virkler
1979), so the late, very narrow bands here are lower bounds.
"""
import numpy as np

from . import cases, life, nasgro, runner

# Inputs whose posterior narrowing is reported (key in the sample dict, log scale?)
LEARNED = {
    "bar": {"growth rate at 2 mm (all inputs combined)": ("rate_2mm", True),
            "crack depth": ("a_i", False), "load amplitude": ("F", False),
            "growth-rate level (C at pivot)": ("level", False),
            "slope n": ("n", False), "threshold dKth0": ("dKth0", False), "threshold shape p": ("p", True),
            "diameter D": ("D", False), "toughness Kc": ("Kc", False), "intrinsic length a0": ("a0", True),
            "stress ratio R": ("R", False), "tilt": ("tilt", False)},
    "plate": {"growth rate at 2 mm (all inputs combined)": ("rate_2mm", True),
              "crack depth": ("a_i", False), "stress amplitude": ("S", False),
              "growth-rate level (C at pivot)": ("level", False),
              "slope n": ("n", False), "crack shape a/c": ("ac", False), "constraint alpha": ("alpha", False),
              "toughness Kc": ("Kc", False), "threshold dKth0": ("dKth0", False), "q": ("q", False),
              "p": ("p", False), "intrinsic length a0": ("a0", True), "stress ratio R": ("R", False),
              "thickness t": ("t", False), "Smax/sigma0": ("smax_sigma0", False), "tilt": ("tilt", False)},
}


def ca_profile(case, s, n_grid=400):
    """Depth grid edges, Kmax and growth rate on the midpoints, constant amplitude, mode I."""
    edges, a, da = life.depth_grid(s["a_i"], cases.a_limit(case, s), n_grid)
    kmax = cases.k_unit(case, s, a) * cases.max_stress(case, s)[:, None]
    rate = nasgro.growth_rate(kmax, kmax * s["R"][:, None], a, cases.material(s))
    return edges, kmax, rate, np.zeros(len(edges))


def histories(case, s, times, profile=ca_profile):
    """Life and crack extent at each time in `times` for every sample (nan once failed).

    A crack that arrests keeps its depth at arrest. `profile` gives the growth
    profile (default: constant-amplitude mode I; mixed_mode.mm_profile for tilted cracks).
    """
    edges, kmax, rate, offset = profile(case, s)
    edges = edges + offset[:, None]
    da = np.diff(edges, axis=1)
    failed_here = kmax >= s["Kc"][:, None]
    stop = np.where(failed_here.any(axis=1), failed_here.argmax(axis=1), rate.shape[1])
    live = np.arange(rate.shape[1])[None, :] < stop[:, None]
    with np.errstate(divide="ignore"):
        dn = np.where(live, np.where(rate > 0, da / np.where(rate > 0, rate, 1.0), np.inf), 0.0)
    cum = np.concatenate([np.zeros((dn.shape[0], 1)), np.cumsum(dn, axis=1)], axis=1)
    total = np.maximum(cum[:, -1], 1.0)
    times = np.asarray(times, dtype=float)
    out = np.full((len(total), len(times)), np.nan)
    rows = np.arange(len(total))
    for k, tk in enumerate(times):
        j = np.clip((cum < tk).sum(axis=1), 1, cum.shape[1] - 1)     # cum[j-1] < tk <= cum[j]
        c0, c1 = cum[rows, j - 1], cum[rows, j]
        with np.errstate(invalid="ignore"):
            frac = np.where(np.isfinite(c1), np.clip((tk - c0) / np.where(c1 > c0, c1 - c0, 1.0), 0.0, 1.0), 0.0)
        a_k = edges[rows, j - 1] + frac * (edges[rows, j] - edges[rows, j - 1])
        out[:, k] = np.where(tk < total, a_k, np.nan)               # nan once failed (inf total: never)
    return total, out


PIVOT = {"bar": 8.0, "plate": 10.0}     # MPa sqrt(m), the pivot of the joint C-n fit in cases.sample


def derived(case, s):
    """Growth-rate level at the fit pivot, and the growth rate of a 2 mm crack with all of the sample's inputs."""
    s["level"] = np.log(s["C"]) + s["n"] * np.log(PIVOT[case])
    a = np.full((len(s["C"]), 1), 2e-3)
    kmax = cases.k_unit(case, s, a) * cases.max_stress(case, s)[:, None]
    r = nasgro.growth_rate(kmax, kmax * s["R"][:, None], a, cases.material(s))[:, 0]
    s["rate_2mm"] = np.maximum(r, 1e-14)
    return s


def ensemble(case, n, seed, times, a_nom=1e-3, chunk=20_000):
    """Prior ensemble: inputs, life (inf if it arrests) and depth histories at `times`."""
    s = derived(case, cases.sample(case, a_nom, n, seed=seed, fixed=runner.fx(())))
    lives, hist = [], []
    for i in range(0, n, chunk):
        sub = {k: v[i:i + chunk] for k, v in s.items()}
        L = life.ca_life(case, sub)                    # inf where the crack arrests
        _, h = histories(case, sub, times)
        lives.append(L)
        hist.append(h)
    return s, np.concatenate(lives), np.concatenate(hist)


def weighted_quantiles(x, w, qs):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    cw /= cw[-1]
    return np.interp(qs, cw, x[o])


def update(prior_life, prior_hist, times, y, sigma_m, k):
    """Log-weights after the first k repeat measurements y[:k] (truth intact at each)."""
    logw = np.zeros(len(prior_life))
    for j in range(k):
        a = prior_hist[:, j]
        alive = np.isfinite(a) & (prior_life > times[j])
        r = np.where(alive, (y[j] - np.where(alive, a, 0.0)) / sigma_m, np.inf)
        logw += -0.5 * r**2
    return logw


def rul_posterior(prior_life, logw, t_now, min_rel=1e-10):
    """Weighted log10 RUL quantiles, ESS and P(arrest) at time t_now."""
    keep = np.isfinite(logw)
    w = np.zeros_like(logw)
    w[keep] = np.exp(logw[keep] - logw[keep].max())
    w[w < min_rel] = 0.0
    ess = w.sum() ** 2 / (w**2).sum()
    fin = np.isfinite(prior_life) & (w > 0)
    p_arrest = 1.0 - w[fin].sum() / w.sum()
    rul = np.log10(np.maximum(prior_life[fin] - t_now, 1.0))
    q = weighted_quantiles(rul, w[fin], [0.05, 0.5, 0.95])
    return dict(q05=q[0], q50=q[1], q95=q[2], u90=q[2] - q[0], ess=float(ess), p_arrest=float(p_arrest)), w


def transformed(s, case):
    """Reported inputs on their analysis scale (log where flagged)."""
    return {name: (np.log(s[key]) if log else s[key]) for name, (key, log) in LEARNED[case].items() if key in s}


def prior_sd(s, case):
    return {name: float(np.std(x)) for name, x in transformed(s, case).items() if np.std(x) > 0}


def posterior_sd_ratio(s, w, case, sd0):
    """Posterior sd / prior sd of each reported input; s and w may be the members with non-zero weight."""
    out = {}
    for name, x in transformed(s, case).items():
        if name not in sd0:
            continue
        m = np.sum(w * x) / w.sum()
        out[name] = float(np.sqrt(np.sum(w * (x - m) ** 2) / w.sum()) / sd0[name])
    return out
