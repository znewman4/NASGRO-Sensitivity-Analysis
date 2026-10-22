"""Giannella-style figures for every input: life scatter (her Figs 12-14) and log-life density (Fig. 11).

For each case at a 1 mm crack (Giannella's nominal depth, and first detection), every input or fitted
group is swept through four spreads with all other inputs at industry-typical uncertainty
(src/nasgro_sa/sweeps.py). Each figure row is one input, most influential first: input value against
life, the density of log10 Nf with normal fits, and the paper's statistics (mean and sd of log10 Nf over
failing cracks, and Pf).

Usage:
    python scripts/run_variable_sweeps.py [--cases round_bar plate] [--a 1.0] [--n 10000] [--seed 41]
                                          [--rows 10] [--out results/variable_sweeps]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

from nasgro_sa import giannella as g
from nasgro_sa.cases import plate_case, round_bar_case
from nasgro_sa.plotting import GRID, INK_2, SLOTS, style
from nasgro_sa.sweeps import spread_sweeps

MM = 1e-3
# member: (name, axis symbol, unit, display scale)
DISPLAY = {
    "a_init": ("crack depth a", "a", "mm", 1 / MM), "aspect_init": ("crack shape a/c", "a/c", "", 1.0),
    "theta": ("crack tilt", "tilt", "deg", 1.0), "F": ("load amplitude F", "F", "kN", 1.0),
    "S_max": ("stress amplitude", "S", "MPa", 1.0), "R": ("stress ratio R", "R", "", 1.0),
    "K_factor": ("SIF solution error", "K factor", "", 1.0), "D": ("diameter D", "D", "mm", 1 / MM),
    "thickness": ("thickness t", "t", "mm", 1 / MM), "width": ("width W", "W", "mm", 1 / MM),
    "curve_level": ("growth-rate level at the pivot", "C-n level", "", 1.0),
    "n": ("growth-rate exponent", "n (m)", "", 1.0), "C": ("C", "C", "m/cycle", 1.0),
    "dK0": ("threshold ΔKth0", "ΔKth0", "MPa√m", 1.0),
    "p": ("threshold-shape exponent p", "p", "", 1.0), "q": ("critical-shape exponent q", "q", "", 1.0),
    "Kc": ("toughness Kc", "Kc", "MPa√m", 1.0), "alpha": ("constraint factor α", "α", "", 1.0),
    "smax_sigma0": ("Smax/σ0", "Smax/σ0", "", 1.0), "Cth_pos": ("Cth (R ≥ 0)", "Cth", "", 1.0),
    "Cth_neg": ("Cth (R < 0)", "Cth", "", 1.0),
    "a0_intrinsic": ("intrinsic crack length a0", "a0", "mm", 1 / MM),
    "closure_factor": ("crack opening factor κ", "κ", "", 1.0),
}
GROUPS = {"growth fit": "material fit (C, m, p, ΔKth0)", "C-n": "growth curve C–n"}
LOG_AXIS = {"C"}
PALETTE = (SLOTS[0], SLOTS[1], SLOTS[2], SLOTS[4])   # largest spread first, as Giannella's blue/red/green/magenta
MAX_POINTS = 4000
MU, SIGMA = "μ", "σ"


def level_label(factor: float, empirical: bool) -> str:
    if factor == 1.0:
        return "pool" if empirical else "typical"
    if factor < 1.0:
        return f"×1/{round(1 / factor)}"
    return f"×{factor:.3g}"


def input_sd(values, member) -> float:
    """Realised sd of the swept input in display units (decades for log-scale inputs)."""
    return float(np.std(np.log10(values))) if member in LOG_AXIS else float(np.std(values) * DISPLAY[member][3])


def input_sd_text(values, member) -> str:
    unit = "dec" if member in LOG_AXIS else DISPLAY[member][2]
    return f"{input_sd(values, member):.2g}{' ' + unit if unit else ''}"


def spread_text(spec, sw) -> str:
    if sw.group in spec.empirical:
        return f"typical = accepted pool; plotted member {DISPLAY[sw.member][1]}"
    parts = []
    for m in sw.members:
        mg, (_, symbol, unit, scale) = spec.marginals[m], DISPLAY[m]
        value = f"sd {mg.sd * scale:.3g}{' ' + unit if unit else ''}" if mg.sd is not None else f"CoV {mg.cov:.3g}"
        parts.append(f"{symbol} {value}" if len(sw.members) > 1 else value)
    return "typical " + ", ".join(parts)


def plot_rows(case, spec, sweeps, inactive, xlim, n, path, part):
    rows = len(sweeps)
    height = 1.85 * rows + 1.6
    fig, axes = plt.subplots(rows, 3, figsize=(8.8, height), squeeze=False,
                             gridspec_kw={"width_ratios": [1.15, 0.95, 0.98]})
    fig.subplots_adjust(left=0.085, right=0.99, top=1 - 1.3 / height, bottom=0.55 / height, hspace=0.55, wspace=0.28)
    lo, hi = np.log10(xlim)
    bins, grid = np.linspace(lo, hi, 61), np.linspace(lo, hi, 300)
    for (sc, pdf, tab), sw in zip(axes, sweeps):
        empirical = sw.group in spec.empirical
        colours = PALETTE[-len(sw.levels):]
        name, symbol, unit, scale = DISPLAY[sw.member]
        for lv, col in zip(sw.levels, colours):
            k = np.flatnonzero(lv.failed)[:MAX_POINTS]
            sc.scatter(lv.cycles[k], lv.values[k] * scale, s=1.5, color=col, alpha=0.55, linewidths=0,
                       rasterized=True)
            logs = np.log10(lv.cycles[lv.failed])
            mu, sd, _ = lv.stats
            pdf.hist(logs, bins=bins, density=True, histtype="step", color=col, lw=0.9)
            pdf.plot(grid, stats.norm.pdf(grid, mu, sd), color=col, lw=1.0, ls="--")
        values = np.concatenate([lv.values[lv.failed] for lv in sw.levels]) * scale
        if sw.member in LOG_AXIS:
            sc.set_yscale("log")
            sc.set_ylim(*np.percentile(values, [0.5, 99.5]) * np.array([0.8, 1.25]))
        else:
            y0, y1 = np.percentile(values, [0.5, 99.5])
            pad = 0.08 * (y1 - y0) or 0.05 * abs(y0) or 0.05
            sc.set_ylim(y0 - pad, y1 + pad)
        sc.set_xscale("log")
        sc.set_xlim(*xlim)
        sc.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        sc.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
        sc.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0e}".replace("e+0", "e")))
        sc.set_ylabel(f"{symbol} [{unit}]" if unit else symbol)
        sc.set_title(f"{GROUPS.get(sw.group, name)}  ({spread_text(spec, sw)})", loc="left", fontsize=8)
        pdf.set_xlim(lo, hi)
        pdf.set_ylabel("PDF")
        for ax in (sc, pdf):
            ax.grid(color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=7)
        tab.axis("off")
        tab.text(0.05, 0.97, f"{'':6}{'input sd':>11}{MU:>6}{SIGMA:>6}{'Pf':>5}", transform=tab.transAxes,
                 va="top", fontsize=6.5, family="monospace", color=INK_2)
        for j, (lv, col) in enumerate(zip(sw.levels, colours)):
            y = 0.76 - 0.19 * j
            mu, sd, pf = lv.stats
            tab.plot([0.0], [y], marker="s", ms=5, color=col, transform=tab.transAxes, clip_on=False)
            tab.text(0.05, y, f"{level_label(lv.factor, empirical):<6}{input_sd_text(lv.values, sw.member):>11}"
                              f"{mu:6.2f}{sd:6.3f}{pf:5.0%}",
                     transform=tab.transAxes, va="center", fontsize=6.5, family="monospace")
    axes[-1][0].set_xlabel("Fatigue life Nf [cycles]")
    axes[-1][1].set_xlabel("log10 Nf")
    key = [Line2D([], [], marker="s", ls="", color=c, label=t) for c, t in zip(
        PALETTE, ("×2 typical (capped at the catalogue high end)", "typical (industry)", "×1/3",
                  "×1/30 (near exact)"))]
    fig.legend(handles=key, loc="upper left", bbox_to_anchor=(0.01, 1 - 0.78 / height), ncol=4, fontsize=7.5,
               handletextpad=0.3, columnspacing=1.2)
    suffix = f" ({part})" if part else ""
    fig.suptitle(f"{case.title}, {spec.marginals['a_init'].nominal / MM:g} mm crack: one input at a time{suffix}\n"
                 "Left: input value vs life (cf. Giannella Figs 12-14). "
                 "Middle: density of log10 Nf with normal fits (cf. Fig. 11).\n"
                 "Right: the swept input's own sd at each level, then mean μ and sd σ of log10 Nf over failing "
                 "cracks, and the failed share Pf.",
                 x=0.01, ha="left", fontsize=8.5)
    note = (f"All other inputs at industry-typical uncertainty; {n} cracks per level with common random numbers; "
            "material pool levels are the pool and shrunk versions of it.")
    if inactive:
        note += " Not shown (no effect on life here): " + ", ".join(inactive) + "."
    fig.text(0.01, 0.12 / height, note, fontsize=6.5, color=INK_2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["round_bar", "plate"])
    ap.add_argument("--a", type=float, default=1.0, help="nominal crack depth [mm]")
    ap.add_argument("--n", type=int, default=10_000, help="Monte Carlo samples per level")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--rows", type=int, default=10, help="most inputs per figure")
    ap.add_argument("--out", default="results/variable_sweeps")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # input names use Greek letters
    style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    builders = {"round_bar": lambda: round_bar_case(g.load_or_build_pool("results/giannella")[0]),
                "plate": plate_case}
    for name in args.cases:
        case, t0 = builders[name](), time.perf_counter()
        spec = case.industry(args.a * MM, "typical")
        sweeps = spread_sweeps(case, args.a * MM, args.n, np.random.default_rng(args.seed))
        shown = sorted((s for s in sweeps if s.active), key=lambda s: -s.benefit)
        inactive = [GROUPS.get(s.group, DISPLAY[s.member][0]) for s in sweeps if not s.active]
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(("group", "plotted_member", "factor", "input_sd_display_units", "mean_log10_nf", "sd_log10_nf",
                        "pf"))
            for s in sweeps:
                for lv in s.levels:
                    w.writerow((s.group, s.member, f"{lv.factor:.4g}", f"{input_sd(lv.values, s.member):.4g}",
                                *(f"{v:.4g}" for v in lv.stats)))
        cycles = np.concatenate([lv.cycles[lv.failed] for s in shown for lv in s.levels])
        xlim = tuple(np.percentile(cycles, [0.5, 99.5]) * np.array([0.85, 1.15]))
        chunks = np.array_split(np.arange(len(shown)), -(-len(shown) // args.rows))
        for i, idx in enumerate(chunks):
            part = f"part {i + 1} of {len(chunks)}" if len(chunks) > 1 else ""
            path = out / (f"{name}_part{i + 1}.png" if part else f"{name}.png")
            plot_rows(case, spec, [shown[j] for j in idx], inactive, xlim, args.n, path, part)
        mu, sd, pf = shown[0].typical.stats
        print(f"\n{case.title}, a0 = {args.a:g} mm: typical uncertainty mu {mu:.3f}, sigma {sd:.3f}, Pf {pf:.0%}")
        print(f"  {'input':<38}{'plotted':>10}   sigma of log10 Nf at each level (largest spread first)")
        for s in shown:
            levels = "  ".join(f"{level_label(lv.factor, s.group in spec.empirical)} {lv.stats[1]:.3f}"
                               for lv in s.levels)
            print(f"  {GROUPS.get(s.group, DISPLAY[s.member][0]):<38}{DISPLAY[s.member][1]:>10}   {levels}")
        if inactive:
            print("  no effect: " + ", ".join(inactive))
        print(f"[{name}] done in {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"\nTables and figures written to {out}")


if __name__ == "__main__":
    main()
