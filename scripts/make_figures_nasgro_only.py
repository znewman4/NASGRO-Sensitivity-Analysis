"""Figures for the NASGRO-only study (reads and writes results/2026-10-23_nasgro_only/)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make_figures as mf  # noqa: E402  (shared style)
from make_figures import INK, INK2, SERIES, SURF, plt  # noqa: E402
from nasgro_scenarios import runner  # noqa: E402

D = runner.STANDARD_DIR
runner.RESULTS = D
SHORT = {
    "material fit (C, m, p, dKth0)": "material fit (C, n, p, ΔKth0)", "growth curve C-n": "growth curve C–n",
    "crack opening factor kappa": "crack-opening factor κ", "constraint factor alpha": "constraint factor α",
    "Smax/sigma0": "Smax/σ0", "threshold dKth0": "threshold ΔKth0", "Cth (R >= 0)": "Cth (R ≥ 0)",
    "intrinsic crack length a0": "intrinsic crack length a0",
}
lab = lambda k: SHORT.get(k, k)  # noqa: E731
CASE_T = {"bar": "Round bar, 25CrMo4, R = −1, 250 MPa", "plate": "Plate surface crack, R = 0.1, 200 MPa"}


def _rankings():
    rr = runner.load("rankings.json")["results"]
    out = {}
    for r in rr:
        out.setdefault((r["config"], r["case"], r["depth_mm"]), []).append(r)
    return out


def _agg(rs):
    keys = rs[0]["reduction"].keys()
    M = np.array([[r["reduction"][k] for k in keys] for r in rs])
    return list(keys), M.mean(0), M.min(0), M.max(0)


def fig_rank():
    R = _rankings()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw=dict(width_ratios=[4, 5]))
    for ax, case in zip(axes, ("bar", "plate")):
        depths = runner.DEPTHS_MM[case]
        aggs = [_agg(R[("standard", case, d)]) for d in depths]
        keys = aggs[0][0]
        mean = np.array([a[1] for a in aggs]).T
        lo = np.array([a[2] for a in aggs]).T
        hi = np.array([a[3] for a in aggs]).T
        keep = [i for i in np.argsort(-mean.max(1)) if mean[i].max() >= 0.02][:9]
        ax.imshow(np.clip(mean[keep], 0, None), cmap="Blues", vmin=0, vmax=0.4, aspect="auto")
        for i, k in enumerate(keep):
            for j in range(len(depths)):
                v = mean[k, j]
                pc = lambda x: f"{max(x, 0) if abs(x) < 0.005 else x:.0%}"  # noqa: E731  (no "-0%")
                ax.text(j, i, f"{pc(v)}\n{pc(lo[k, j])} to {pc(hi[k, j])}", ha="center", va="center",
                        fontsize=8.5, color="white" if v > 0.25 else INK, linespacing=1.3)
        ax.set_yticks(range(len(keep)))
        ax.set_yticklabels([lab(keys[k]) for k in keep])
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels([f"{d:g}" for d in depths])
        ax.set_xlabel("Initial crack depth [mm]")
        ax.set_title(CASE_T[case], loc="left", fontsize=11)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle("U90 reduction when one input is known exactly, NASGRO-equation inputs only "
                 "(mean of 5 seeds; range below)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "fig_rank.png"), dpi=150)


def fig_with_without():
    """Dumbbell: each input's U90 reduction with and without kappa and the SIF error."""
    R = _rankings()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0))
    for ax, case in zip(axes, ("bar", "plate")):
        ke, me, le, he = _agg(R[("extended", case, 1.0)])
        ks, ms, ls, hs = _agg(R[("standard", case, 1.0)])
        ext = dict(zip(ke, me))
        std = dict(zip(ks, ms))
        rng_s = {k: (a, b) for k, a, b in zip(ks, ls, hs)}
        names = sorted(ke, key=lambda k: -max(ext.get(k, 0), std.get(k, 0)))
        names = [k for k in names if max(ext.get(k, 0), std.get(k, 0)) >= 0.015][:10]
        y = np.arange(len(names))[::-1]
        for yi, k in zip(y, names):
            if k in std:
                ax.plot([ext[k], std[k]], [yi, yi], color=INK2, lw=1.2, zorder=1)
                ax.plot([rng_s[k][0], rng_s[k][1]], [yi - 0.22, yi - 0.22], color=SERIES[1], lw=1, zorder=1)
        ax.scatter([ext[k] for k in names], y, s=55, color=SERIES[0], zorder=2, edgecolor=SURF,
                   linewidth=1.5, label="with κ and SIF error uncertain (previous)")
        ax.scatter([std.get(k, np.nan) for k in names], y, s=55, color=SERIES[1], zorder=3,
                   edgecolor=SURF, linewidth=1.5, label="NASGRO inputs only (this run; bar = seed range)")
        for yi, k in zip(y, names):
            if k not in std:
                ax.annotate("fixed at 1", (ext[k], yi), xytext=(8, -3), textcoords="offset points",
                            fontsize=8.5, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([lab(k) for k in names])
        ax.xaxis.set_major_formatter(mf.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("U90 reduction when known exactly, 1 mm crack")
        ax.set_title(CASE_T[case], loc="left", fontsize=11)
        ax.grid(axis="y", visible=False)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Ranking with and without the two non-NASGRO inputs (same samples, mean of 5 seeds)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "fig_with_without.png"), dpi=150)


def fig_phases():
    ph = runner.load("phases.json")["results"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, case in zip(axes, ("bar", "plate")):
        rr = sorted([r for r in ph if r["case"] == case], key=lambda r: r["depth_mm"])
        x = np.arange(len(rr))
        parts = [("near_threshold", "near threshold", SERIES[0]), ("paris", "steady (Paris) growth", SERIES[1]),
                 ("near_critical", "near critical", SERIES[2])]
        bottom = np.zeros(len(rr))
        few = np.array([r["never_grow"] > 0.98 for r in rr])
        for key, name, col in parts:
            v = np.array([0.0 if f else np.nan_to_num(r[key]) for r, f in zip(rr, few)])
            ax.bar(x, v, 0.7, bottom=bottom, color=col, label=name, edgecolor=SURF, linewidth=1.5)
            bottom += v
        for xi, r, f in zip(x, rr, few):
            ax.text(xi, 1.03, f"{r['never_grow']:.0%}", ha="center", fontsize=9, color=INK2)
            if f:
                ax.text(xi, 0.45, "too few\ncracks\ngrow", ha="center", fontsize=9, color=INK2)
        ax.text(-0.9, 1.03, "never grow:", ha="right", fontsize=9, color=INK2)
        ax.axvline(x[[r["depth_mm"] for r in rr].index(1.0)] - 0.5, color=INK2, ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['depth_mm']:g}" for r in rr])
        ax.set_xlabel("Initial crack depth [mm]")
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(mf.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(CASE_T[case], loc="left", fontsize=11, pad=18)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of remaining life\n(cracks that fail)")
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, frameon=False, ncol=3, loc="lower center", fontsize=9.5)
    fig.suptitle("Where the remaining life is spent (NASGRO inputs only); dashed line = 1 mm detection limit",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(D, "fig_phases.png"), dpi=150)


# Typical spread and catalogue range (low, typical, high) as CoV, or sd where noted.
CATALOGUE = [
    ("crack depth, undetected (CoV)", 0.3, 0.6, 1.0, "lit"),
    ("crack depth, UT-sized (sd, mm)", 0.15, 0.5, 1.8, "lit"),
    ("stress or load amplitude", 0.015, 0.10, 0.5, "lit"),
    ("crack shape a/c", 0.1, 0.2, 0.3, "judg"),
    ("diameter, thickness, width", 0.005, 0.015, 0.03, "judg"),
    ("growth-rate level C (at pivot)", 0.05, 0.15, 0.3, "lit"),
    ("growth-rate slope n", 0.02, 0.05, 0.1, "judg"),
    ("threshold ΔKth0", 0.05, 0.10, 0.2, "lit"),
    ("threshold-shape exponent p", 0.15, 0.3, 0.5, "judg"),
    ("critical-shape exponent q", 0.15, 0.3, 0.5, "judg"),
    ("fracture toughness Kc", 0.05, 0.10, 0.3, "lit"),
    ("constraint factor α", 0.1, 0.19, 0.3, "lit"),
    ("Smax/σ0", 0.1, 0.2, 0.3, "judg"),
    ("Cth (R ≥ 0 and R < 0)", 0.1, 0.3, 0.5, "judg"),
    ("intrinsic crack length a0", 0.25, 0.5, 1.0, "judg"),
]


def fig_uncertainty():
    fig, ax = plt.subplots(figsize=(10, 6.4))
    y = np.arange(len(CATALOGUE))[::-1]
    for yi, (name, lo, typ, hi, src) in zip(y, CATALOGUE):
        col = SERIES[0] if src == "lit" else SERIES[1]
        ax.plot([lo, hi], [yi, yi], color=col, lw=2.5, solid_capstyle="round")
        ax.plot(typ, yi, "o", color=col, markersize=8, markeredgecolor=SURF, markeredgewidth=1.5)
    ax.plot([], [], "-o", color=SERIES[0], label="literature range")
    ax.plot([], [], "-o", color=SERIES[1], label="engineering judgement (to be sourced)")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([c[0] for c in CATALOGUE])
    ax.set_xlabel("Coefficient of variation (dot = typical, bar = low to high); UT-sized depth in mm sd")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    ax.grid(axis="y", visible=False)
    ax.set_title("Industry-typical uncertainty of the NASGRO-equation inputs and the crack geometry",
                 loc="left", fontsize=11.5)
    fig.text(0.01, 0.01, "Absolute: stress ratio R sd 0.02 / 0.05 / 0.15; crack tilt sd 2 / 5 / 10° once sized "
             "(5 / 15 / 30° before). κ and the SIF error are not NASGRO inputs and are fixed at 1.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(D, "fig_uncertainty.png"), dpi=150)


SWEEP_NAMES = {"depth": "crack depth", "material": "material fit (C, n, p, ΔKth0)", "F": "load amplitude",
               "S": "stress amplitude", "D": "diameter D", "tilt": "crack tilt", "Kc": "toughness Kc",
               "a0": "intrinsic crack length a0", "R": "stress ratio R", "alpha": "constraint factor α",
               "Cn": "growth curve C–n", "ac": "crack shape a/c", "q": "critical-shape exponent q",
               "p": "threshold-shape exponent p", "smax_sigma0": "Smax/σ0", "dKth0": "threshold ΔKth0",
               "t": "thickness t", "cth_pos": "Cth (R ≥ 0)", "W": "width W", "cth_neg": "Cth (R < 0)"}


def fig_sweeps():
    sw = runner.load("sweeps.json")["results"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    for ax, case in zip(axes, ("bar", "plate")):
        rr = [r for r in sw if r["case"] == case]
        rr.sort(key=lambda r: -(r["levels"][0]["sigma"] - r["levels"][3]["sigma"]))
        y = np.arange(len(rr))[::-1]
        for yi, r in zip(y, rr):
            sig = [lv["sigma"] for lv in r["levels"]]
            ax.plot([min(sig), max(sig)], [yi, yi], color=INK2, lw=1, zorder=1)
            for k, lv in enumerate(r["levels"]):
                ax.plot(lv["sigma"], yi, "o", color=SERIES[[0, 1, 2, 4][k]], markersize=7,
                        markeredgecolor=SURF, markeredgewidth=1.2, zorder=2,
                        label=["twice typical (capped)", "typical", "a third", "a thirtieth"][k] if yi == y[0] else None)
        ax.set_yticks(y)
        ax.set_yticklabels([SWEEP_NAMES.get(r["key"], r["key"]) for r in rr])
        ax.set_xlabel("σ of log10 life over cracks that fail, 1 mm crack")
        ax.set_title(CASE_T[case], loc="left", fontsize=11)
        ax.grid(axis="y", visible=False)
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, frameon=False, ncol=4, loc="lower center", fontsize=9.5, title="spread of the swept input")
    fig.suptitle("Giannella-style sweeps: life scatter at four spreads of each input, every other input typical",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(D, "fig_sweeps.png"), dpi=150)


if __name__ == "__main__":
    which = sys.argv[1:] or ["rank", "ww", "phases", "unc", "sweeps"]
    if "rank" in which:
        fig_rank()
    if "ww" in which:
        fig_with_without()
    if "phases" in which:
        fig_phases()
    if "unc" in which:
        fig_uncertainty()
    if "sweeps" in which:
        fig_sweeps()
