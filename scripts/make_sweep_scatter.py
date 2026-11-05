"""Giannella-style per-input scatter plots for the NASGRO-only run (Appendix A).

For every NASGRO-equation input of each case, at a 1 mm crack, the input's
spread is set to twice typical (capped at the catalogue's high end), typical,
a third and a thirtieth, with every other input at typical spread (the levels
and seeds of sweeps.json / Figure A1). Each row: input value against life
(cf. Giannella 2022, Figs 12-14), the density of log10 life with normal fits,
and the mean, sd and failed share at each level.

Writes fig_scatter_bar.png, fig_scatter_plate_1.png and fig_scatter_plate_2.png
to results/2026-10-23_nasgro_only/.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make_figures as mf  # noqa: E402,F401  (shared style)
from make_figures import GRID, INK, INK2, SERIES, plt  # noqa: E402
from nasgro_scenarios import cases, life, runner  # noqa: E402
from run_nasgro_only import HIGH, SWEEP_LEVELS  # noqa: E402

N, SEED, N_SHOW = 10_000, 1, 4000
LC = [SERIES[0], SERIES[1], SERIES[2], SERIES[4]]      # x2, typical, x1/3, x1/30
LEVEL_LABEL = {"x2": "×2 typical", "typical": "typical (industry)", "x1/3": "×1/3", "x1/30": "×1/30 (near exact)"}
# key -> (axis label, function of the sample dict giving the plotted value, unit for the sd column)
PIVOT = {"bar": 8.0, "plate": 10.0}


def shown(case, key):
    p = PIVOT[case]
    table = {
        "depth": ("a at detection [mm]", lambda s: s["a_i"] * 1e3, "mm", 1.0),
        "material": (f"da/dN at ΔK = {p:g} [m/cycle]", lambda s: s["C"] * p ** s["n"], "", None),
        "Cn": (f"da/dN at ΔK = {p:g} [m/cycle]", lambda s: s["C"] * p ** s["n"], "", None),
        "F": ("F [kN]", lambda s: s["F"] / 1e3, "kN", 1.0),
        "S": ("S [MPa]", lambda s: s["S"], "MPa", 1.0),
        "D": ("D [mm]", lambda s: s["D"] * 1e3, "mm", 1.0),
        "t": ("t [mm]", lambda s: s["t"] * 1e3, "mm", 1.0),
        "W": ("W [mm]", lambda s: s["W"] * 1e3, "mm", 1.0),
        "tilt": ("tilt [deg]", lambda s: np.rad2deg(s["tilt"]), "deg", 1.0),
        "Kc": ("Kc [MPa√m]", lambda s: s["Kc"], "MPa√m", 1.0),
        "a0": ("a0 [mm]", lambda s: s["a0"] * 1e3, "mm", 1.0),
        "R": ("R", lambda s: s["R"], "", 1.0),
        "ac": ("a/c", lambda s: s["ac"], "", 1.0),
        "alpha": ("α", lambda s: s["alpha"], "", 1.0),
        "smax_sigma0": ("Smax/σ0", lambda s: s["smax_sigma0"], "", 1.0),
        "q": ("q", lambda s: s["q"], "", 1.0),
        "p": ("p", lambda s: s["p"], "", 1.0),
        "dKth0": ("ΔKth0 [MPa√m]", lambda s: s["dKth0"], "MPa√m", 1.0),
        "cth_pos": ("Cth (R ≥ 0)", lambda s: s["cth_pos"], "", 1.0),
        "cth_neg": ("Cth (R < 0)", lambda s: s["cth_neg"], "", 1.0),
    }
    return table[key]


def task(case, name, key):
    runner.set_standard()
    rows = []
    for label, m in SWEEP_LEVELS:
        mm = min(m, HIGH.get(key, 2.0)) if m > 1 else m
        s = cases.sample(case, 1e-3, N, seed=SEED, fixed=runner.NON_NASGRO, spread={key: mm})
        L = life.chunked(lambda x: life.ca_life(case, x), s)
        x = shown(case, key)[1](s)
        rows.append(dict(level=label, mult=mm, x=x, life=L, stats=life.stats(L)))
    return case, name, key, rows


def row(axes, case, name, key, rows, last):
    sax, pax, tax = axes
    lbl, _, unit, _ = shown(case, key)
    rng = np.random.default_rng(0)
    pick = rng.choice(N, N_SHOW, replace=False)
    order = list(range(len(rows)))
    for i in order:                                   # widest first, so the narrow clouds sit on top
        r = rows[i]
        fin = np.isfinite(r["life"][pick])
        sax.scatter(r["life"][pick][fin], r["x"][pick][fin], s=2 if i < 3 else 3, color=LC[i],
                    alpha=0.45 if i < 3 else 0.8, lw=0,
                    rasterized=True)
    sax.set_xscale("log")
    if key in ("material", "Cn"):
        sax.set_yscale("log")
    sax.set_xlim(3e4 if case == "bar" else 2e3, 5e6 if case == "bar" else 4e5)
    sax.set_ylabel(lbl, fontsize=9)
    sax.set_title(name, loc="left", fontsize=10)
    bins = np.linspace(*(4.5, 6.7) if case == "bar" else (3.3, 5.6), 60)
    xs = np.linspace(bins[0], bins[-1], 300)
    lines = []
    for i, r in enumerate(rows):
        fin = np.isfinite(r["life"])
        y = np.log10(r["life"][fin])
        pax.hist(y, bins=bins, density=True, histtype="step", color=LC[i], lw=1.3)
        st = r["stats"]
        pax.plot(xs, np.exp(-0.5 * ((xs - st["mu"]) / st["sigma"]) ** 2) / (st["sigma"] * np.sqrt(2 * np.pi)),
                 color=LC[i], lw=1, ls="--")
        sd = np.std(r["x"])
        sd_txt = f"{sd:.2g} {unit}".strip() if key not in ("material", "Cn") else \
            f"{np.std(np.log10(r['x'])):.2g} dec"
        lines.append((LEVEL_LABEL[r["level"]], sd_txt, st["mu"], st["sigma"], st["pf"]))
    pax.set_ylabel("PDF", fontsize=9)
    tax.axis("off")
    tax.text(0.0, 0.95, f"{'level':<19}{'input sd':>12}{'μ':>7}{'σ':>7}{'Pf':>6}", family="monospace",
             fontsize=8, color=INK2, transform=tax.transAxes, va="top")
    for j, (lv, sd, mu, sg, pf) in enumerate(lines):
        tax.text(0.0, 0.78 - 0.17 * j, f"{lv:<19}{sd:>12}{mu:>7.2f}{sg:>7.3f}{pf:>6.0%}", family="monospace",
                 fontsize=8, color=INK, transform=tax.transAxes, va="top")
        tax.plot([-0.05], [0.745 - 0.17 * j], "s", color=LC[j], markersize=6, transform=tax.transAxes,
                 clip_on=False)
    if last:
        sax.set_xlabel("Remaining life Nf from 1 mm [cycles]")
        pax.set_xlabel("log10 Nf")


def figure(case, items, results, name, title):
    fig, axes = plt.subplots(len(items), 3, figsize=(13.5, 2.25 * len(items) + 0.9),
                             gridspec_kw=dict(width_ratios=[1.15, 1, 0.95]))
    axes = np.atleast_2d(axes)
    for r, (nm, key) in enumerate(items):
        row(axes[r], case, nm, key, results[(case, key)], r == len(items) - 1)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    path = os.path.join(runner.STANDARD_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(path)


def main():
    runner.set_standard()
    tasks = [(c, nm, k) for c in ("bar", "plate") for nm, k in runner.base_inputs(c).items()]
    results = {}
    for case, nm, key, rows in runner.run_pool(task, tasks):
        results[(case, key)] = rows
    for case in ("bar", "plate"):
        items = list(runner.base_inputs(case).items())
        spread = {k: results[(case, k)][0]["stats"]["sigma"] - results[(case, k)][-1]["stats"]["sigma"]
                  for _, k in items}
        items.sort(key=lambda it: -spread[it[1]])
        title = (f"{'Round bar, 25CrMo4, R = −1, 250 MPa' if case == 'bar' else 'Plate surface crack, R = 0.1, 200 MPa'}"
                 ", 1 mm crack. One input's spread changed at a time\n"
                 "(blue ×2 typical, orange typical, green ×1/3, pink ×1/30), every other input at typical spread")
        if case == "bar":
            figure(case, items, results, "fig_scatter_bar.png", title)
        else:
            h = (len(items) + 1) // 2
            figure(case, items[:h], results, "fig_scatter_plate_1.png", title + " (part 1)")
            figure(case, items[h:], results, "fig_scatter_plate_2.png", title + " (part 2)")


if __name__ == "__main__":
    main()
