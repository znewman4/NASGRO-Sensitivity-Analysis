"""Variable-amplitude scenario: a repeating load block, with and without load interaction.

Spectrum. Each block has n_b cycles at a constant mean stress. Amplitudes follow
a Weibull exceedance P(S > s) = exp(-(s/s0)^h) (h = 1 exponential, h = 2
Rayleigh, i.e. narrow-band Gaussian loading), clipped at the level exceeded 10
times per block (the usual truncation). The amplitudes are counted into
N_BINS levels and scaled so the block's damage-equivalent amplitude
(m-th moment, m = the case's nominal Paris slope) equals the constant-amplitude
amplitude of the base case. With no threshold and no interaction the spectrum
would therefore give the base-case life; differences come from cycles below
threshold, from the mean-stress (R) of each level, and from retardation.
Peaks are truncated at 0.9 sigma_y (gross yield) at constant mean stress.
Uncertain spectrum inputs: shape h and block length n_b (which together set the
overload ratio); its scale is the existing stress/load-amplitude input.

Stage 1 (no interaction): growth per block G(a) = sum_j n_j da/dN_j depends on
depth only, so life.integrate gives the life directly.

Stage 2 (generalised Willenborg; Willenborg et al. 1971, Gallagher 1974):
an overload with peak Kol leaves a plastic zone r = (Kol/sigma_y)^2 / (beta pi).
A later cycle with peak Kmax inside it carries a residual intensity
    K_R = phi [sigma_y sqrt(beta pi (a_ol + r_ol - a)) - Kmax],  phi = 1/(SOLR - 1),
which shifts Kmax and Kmin down (dK unchanged), exactly like a compressive
residual stress, so NASGRO's closure and threshold respond through R. Growth
stops when Kol/Kmax >= SOLR. The block is split into N_SEG segments; the 10
peak cycles sit at evenly spaced segments, the rest are spread evenly. The
zone state is carried from block to block; once growth per block is a small
fraction of the depth, identical blocks are lumped (state shifted with the crack).
Not modelled: underload/compressive-cycle acceleration, strip-yield closure
(Newman 1981), interaction with residual stress.
"""
import numpy as np
from scipy.special import ndtr

from . import cases, life, nasgro

N_BINS = 16          # amplitude levels below the clip level (+1 clip level)
N_CLIP = 10          # cycles per block at the clip level
N_SEG = 20           # segments per block (40 changes log10 N by < 0.01)
PEAK_CAP = 0.9       # peak stress limit as a fraction of yield

SPECTRUM = {   # judgement ranges; see the study document for the reasoning
    # block length chosen so a typical life spans >= ~10 blocks
    "bar": dict(h=(1.2, 2.0), nb=(1e3, 3e4), solr=(1.7, 2.7), beta=(2.0, 6.0)),
    "plate": dict(h=(1.2, 2.0), nb=(3e2, 3e3), solr=(2.5, 3.5), beta=(2.0, 6.0)),
}
STAGE1_INPUTS = {"spectrum shape h": "va_h", "block length n_b": "va_nb"}
STAGE2_INPUTS = {**STAGE1_INPUTS, "shut-off ratio SOLR": "va_solr",
                 "overload plastic-zone factor beta": "va_beta"}

_GL_U, _GL_W = np.polynomial.legendre.leggauss(12)
_GL_U, _GL_W = 0.5 * (_GL_U + 1.0), 0.5 * _GL_W


def _uniform(name, lo, hi, n, seed, fixed, log=False):
    if fixed:
        return np.full(n, np.sqrt(lo * hi) if log else 0.5 * (lo + hi))
    u = ndtr(cases._z(name, n, seed))
    return lo * (hi / lo) ** u if log else lo + (hi - lo) * u


def add_inputs(s, case, seed, fixed=()):
    n = s["a_i"].size
    sp = SPECTRUM[case]
    s["va_h"] = _uniform("va_h", *sp["h"], n, seed, "va_h" in fixed)
    s["va_nb"] = _uniform("va_nb", *sp["nb"], n, seed, "va_nb" in fixed, log=True)
    s["va_solr"] = _uniform("va_solr", *sp["solr"], n, seed, "va_solr" in fixed)
    s["va_beta"] = _uniform("va_beta", *sp["beta"], n, seed, "va_beta" in fixed)
    return s


def spectrum(h, nb, m_ref):
    """Per-sample amplitude levels (relative to the equivalent amplitude) and counts.

    Returns amp (n, N_BINS+1), counts (n, N_BINS+1); the last level is the clip level.
    """
    h, nb = h[:, None], nb[:, None]
    s_clip = np.log(nb / N_CLIP) ** (1.0 / h)                       # s0 = 1
    edges = s_clip * np.linspace(0.0, 1.0, N_BINS + 1)[None, :]
    lo, hi = edges[:, :-1, None], edges[:, 1:, None]
    x = lo + (hi - lo) * _GL_U                                      # (n, bins, gl)
    pdf = h[..., None] * x ** (h[..., None] - 1) * np.exp(-x ** h[..., None])
    w = pdf * (hi - lo) * _GL_W
    rep = (np.sum(w * x**m_ref, -1) / np.maximum(w.sum(-1), 1e-300)) ** (1.0 / m_ref)
    counts = nb * (np.exp(-lo[..., 0] ** h) - np.exp(-hi[..., 0] ** h))     # exact bin mass
    counts = np.concatenate([counts, np.full_like(nb, float(N_CLIP))], axis=1)
    amp = np.concatenate([rep, s_clip], axis=1)
    s_eq = (np.sum(counts * amp**m_ref, 1, keepdims=True) / counts.sum(1, keepdims=True)) ** (1.0 / m_ref)
    return amp / s_eq, counts


def _levels(case, s):
    """Peak and valley stress (MPa) of every level, and counts per block."""
    m_ref = cases.BAR["n"] if case == "bar" else cases.PLATE["n"]
    rel, counts = spectrum(s["va_h"], s["va_nb"], m_ref)
    smax_ca = cases.max_stress(case, s)[:, None]
    R = s["R"][:, None]
    sm, sa = smax_ca * (1 + R) / 2, smax_ca * (1 - R) / 2
    # peaks truncated at 0.9 sigma_y (gross-yield limit), keeping the mean stress
    sa_j = np.minimum(sa * rel, PEAK_CAP * s["sigma_y"][:, None] - sm)
    return sm + sa_j, sm - sa_j, counts


def stage1_life(case, s, n_grid=400):
    """Spectrum without interaction: life in cycles."""
    _, a, da = life.depth_grid(s["a_i"], cases.a_limit(case, s), n_grid)
    smax, smin, counts = _levels(case, s)
    k1 = cases.k_unit(case, s, a)[..., None]                       # (n, grid, 1)
    mat = {k: v[..., None] for k, v in cases.material(s).items()}
    rate = nasgro.growth_rate(k1 * smax[:, None, :], k1 * smin[:, None, :], a[..., None], mat)
    per_block = np.sum(rate * counts[:, None, :], axis=-1)
    kpeak = k1[..., 0] * smax[:, -1:]
    return life.integrate(per_block, kpeak, s["Kc"], da, cycles_per_step=counts.sum(1))


def _sequence():
    """Counts of each level per segment as fractions of the block (N_SEG, N_BINS+1)."""
    frac = np.full((N_SEG, N_BINS + 1), 1.0 / N_SEG)
    peak = np.zeros(N_SEG)
    peak[np.linspace(0, N_SEG, N_CLIP, endpoint=False).astype(int)] = 1.0
    frac[:, -1] = peak / N_CLIP
    return frac


def stage2_life(case, s, step_frac=0.01, max_cycles=1e10):
    """Generalised Willenborg, block by block. Life in cycles (inf = arrest/runout)."""
    n = s["a_i"].size
    smax, smin, counts = _levels(case, s)                          # (n, L)
    seq = _sequence()                                              # (N_SEG, L)
    mat = {k: v for k, v in cases.material(s).items()}             # (n, 1)
    sy, beta = s["sigma_y"][:, None], s["va_beta"][:, None]
    phi = 1.0 / (s["va_solr"][:, None] - 1.0)
    a_lim = cases.a_limit(case, s)
    nb = counts.sum(1)

    def kunit(a):
        return cases.k_unit(case, s, a[:, None])                    # (n, 1)

    def zone(k):
        return (k / sy) ** 2 / (beta * np.pi)

    a = s["a_i"].copy()
    zb = a + zone(kunit(a) * smax[:, -1:])[:, 0]                   # steady state at start
    N = np.zeros(n)
    done = np.zeros(n, bool)
    out = np.full(n, np.inf)
    for _ in range(200000):
        act = ~done
        if not act.any():
            break
        a0 = a.copy()
        for k in range(N_SEG):
            live = ~done
            k1 = kunit(a)
            kmax, kmin = k1 * smax, k1 * smin                       # (n, L)
            c = counts * seq[k]
            r = np.where(c > 0, zone(np.maximum(kmax, 0.0)), 0.0).max(1)
            zb = np.maximum(zb, a + r)
            kreq = sy[:, 0] * np.sqrt(beta[:, 0] * np.pi * np.maximum(zb - a, 0.0))
            kr = phi * np.maximum(kreq[:, None] - kmax, 0.0)
            rate = nasgro.growth_rate(kmax - kr, kmin - kr, a[:, None], mat)
            grow = np.sum(np.where(c > 0, c * rate, 0.0), 1)
            a = np.where(live, np.minimum(a + grow, a_lim), a)
            # failure within the block, resolved to the segment
            fail = live & ((kunit(a)[:, 0] * smax[:, -1] >= s["Kc"]) | (a >= a_lim))
            out = np.where(fail, N + (k + 1) / N_SEG * nb, out)
            done |= fail
        act = act & ~done
        dab = a - a0
        arrest = act & (dab <= 0.0)
        M = np.where(act & ~arrest,
                     np.clip(np.floor(step_frac * a / np.maximum(dab, 1e-30)), 1, 1e7), 1.0)
        shift = (M - 1.0) * dab
        a = np.where(act, np.minimum(a + shift, a_lim), a)
        zb = np.where(act, zb + shift, zb)
        N = np.where(act, N + M * nb, N)
        runout = act & (N > max_cycles)
        done |= arrest | runout
    return out
