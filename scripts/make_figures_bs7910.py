"""Figures for the BS 7910 rerun (results/2026-11-05_bs7910/)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "2026-11-05_bs7910")
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SEQ = ["#9ec5f0", "#5d9ee6", "#2a78d6", "#14498a"]      # light -> dark, one hue
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
CASE_TITLE = {"bar": "Round bar, 25CrMo4, R = −1, 250 MPa", "plate": "Plate surface crack, R = 0.1, 200 MPa"}
SHORT = {"study 0.5 mm": "study\n0.5 mm", "conventional UT": "conv. UT\n2.3 mm",
         "focussed phased array / zonal AUT": "phased array\n1.15 mm", "TOFD (Table T.2)": "TOFD\n0.77 mm",
         "TOFD trials, +-0.5 mm": "TOFD trials\n0.38 mm", "TOFD trials, +-0.1 mm": "TOFD best\n0.08 mm"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200,
                     "figure.facecolor": "#fbfbf9", "axes.facecolor": "#fbfbf9"})


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def fig_sweep(r):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for j, case in enumerate(("bar", "plate")):
        rows = [x for x in r["sweep"] if x["case"] == case]
        for i, a in enumerate(sorted({x["depth_mm"] for x in rows})):
            pts = sorted([x for x in rows if x["depth_mm"] == a], key=lambda x: x["sd_mm"])
            sd = np.array([p["sd_mm"] for p in pts])
            for row, key, scale in ((0, "u90", 1), (1, "depth_reduction", 100)):
                ax = axes[row, j]
                y = np.array([p[key] for p in pts]) * scale
                ax.plot(sd, y, color=SEQ[i], lw=2, label=f"{a:g} mm", zorder=2)
                ok = np.array([p["sizable"] for p in pts])
                ax.scatter(sd[ok], y[ok], s=34, color=SEQ[i], edgecolor="#fbfbf9", lw=1.5, zorder=3)
                ax.scatter(sd[~ok], y[~ok], s=34, facecolor="#fbfbf9", edgecolor=SEQ[i], lw=1.5, zorder=3)
        axes[0, j].set_title(CASE_TITLE[case], fontsize=9.5, color=INK, loc="left")
        axes[1, j].set_xscale("log")
        axes[1, j].set_xticks([0.08, 0.38, 0.5, 0.77, 1.15, 2.3])
        axes[1, j].set_xticklabels(["0.08", "0.38", "0.5", "0.77", "1.15", "2.3"], fontsize=8)
        axes[1, j].minorticks_off()
        axes[1, j].set_xlabel("Detection sizing error, sd (mm)  [BS 7910 Table T.2, sd = E/1.3]")
    axes[0, 0].set_ylabel("U90 at detection (decades)")
    axes[1, 0].set_ylabel("U90 reduction, depth\nknown exactly (%)")
    axes[0, 1].legend(title="Start depth", fontsize=8, title_fontsize=8, loc="upper left")
    fig.text(0.01, 0.035, "sd 0.08 / 0.38 mm: TOFD trials (±0.1 / ±0.5 mm); 0.5 mm: the study's assumption; "
             "0.77 mm: TOFD; 1.15 mm: focussed phased array or zonal AUT; 2.3 mm: conventional UT.",
             fontsize=7.5, color=MUTED)
    fig.text(0.01, 0.005, "Hollow markers: below the technique's minimum reliably sized depth (Table T.2). "
             "Mean of 3 seeds × 10,000; NASGRO-equation inputs only.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_bs_sizing_sweep.png"))


def fig_rank(r, technique="focussed phased array / zonal AUT"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1, 1]})
    for ax, case in zip(axes, ("bar", "plate")):
        rows = sorted([x for x in r["ranking"] if x["case"] == case and x["technique"] == technique],
                      key=lambda x: x["depth_mm"])
        names = [k for k in rows[0]["reduction"] if max(x["reduction"][k] for x in rows) >= 0.02]
        names.sort(key=lambda k: -max(x["reduction"][k] for x in rows))
        M = np.array([[x["reduction"][k] * 100 for x in rows] for k in names])
        ax.imshow(np.clip(M, 0, None), cmap="Blues", vmin=0, vmax=60, aspect="auto")
        for i in range(M.shape[0]):
            for jj in range(M.shape[1]):
                lo, hi = rows[jj]["reduction_range"][names[i]]
                ax.text(jj, i, f"{M[i, jj]:.0f} %\n{lo*100:.0f}–{hi*100:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if M[i, jj] > 35 else INK)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n.replace(" (C, m, p, dKth0)", " (C, n, p, ΔKₜₕ₀)") for n in names],
                           fontsize=8)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([f"{x['depth_mm']:g}" + (" *" if x["depth_mm"] < 2 else "") for x in rows])
        ax.set_xlabel("Detected crack depth (mm)")
        ax.grid(False)
        ax.set_title(CASE_TITLE[case] + f"\nbaseline U90: " +
                     ", ".join(f"{x['u90']:.2f}" for x in rows) + " decades", fontsize=9, color=INK, loc="left")
    fig.text(0.01, 0.01, "U90 reduction when one input is known exactly; depth sized by focussed phased array "
             "(sd 1.15 mm). Mean of 3 seeds, range below; inputs under 2 % omitted.\n* Below the minimum depth "
             "phased array can reliably size (Table T.2).", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_bs_rank.png"))


def fig_laws(r):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=False)
    for ax, case in zip(axes, ("bar", "plate")):
        rows = [x for x in r["growth_laws"] if x["case"] == case]
        x = np.arange(len(rows))
        for off, key, col, lab in ((-0.15, "nasgro", CAT[0], "NASGRO (this study)"),
                                   (0.15, "annex_k", CAT[1], "Paris law, Annex K scatter")):
            med = np.array([w[key]["q50"] for w in rows])
            q5 = np.array([w[key]["q5"] for w in rows])
            q95 = np.minimum(np.array([w[key]["q95"] for w in rows]), 7)
            q23 = np.array([w[key]["q2.3"] for w in rows])
            ax.vlines(x + off, q5, q95, color=col, lw=2)
            ax.scatter(x + off, med, s=40, color=col, edgecolor="#fbfbf9", lw=1.5, zorder=3, label=lab + ": median, 5–95 %")
            ax.scatter(x + off, q23, s=40, marker="_", color=col, lw=2.5, zorder=3)
        ax.scatter(x, [w["bs_det_log10"] for w in rows], s=55, marker="D", color=CAT[2], edgecolor="#fbfbf9",
                   lw=1.5, zorder=4, label="BS 7910 deterministic (depth + E)")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[w["technique"]] for w in rows], fontsize=7.5)
        ax.set_ylabel("log₁₀ remaining life (cycles)")
        ax.set_title(CASE_TITLE[case] + ", 3 mm reported crack", fontsize=9.5, color=INK, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=MUTED, marker="_", ls="", ms=12, mew=2.5))
    l.append("2.3rd percentile (97.7 % survival, BS 7910 8.3)")
    fig.legend(h, l, fontsize=7.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.04))
    fig.text(0.01, 0.01, "Paris law: full stress range (8.2.1.4), conservative for the unwelded R = −1 bar. "
             "Plate: steel law scaled by (E_steel/E)³ (Equation 8.8).", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_bs_growth_laws.png"))


def fig_obs(o):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, case in zip(axes, ("bar", "plate")):
        for i, (name, reg) in enumerate(o[case]["regimes"].items()):
            pk = reg["per_k"]
            k = np.array([p["k"] for p in pk])
            m = np.array([p["u90_median"] for p in pk])
            lo = np.array([p["u90_q25"] for p in pk])
            hi = np.array([p["u90_q75"] for p in pk])
            col = CAT[i]
            ls = "--" if name.startswith("permanent") else "-"
            ax.fill_between(k, lo, hi, color=col, alpha=0.12, lw=0)
            ax.plot(k, m, color=col, lw=2, ls=ls, marker="o", ms=4, label=name)
        ax.set_xlabel(f"Inspection intervals since detection (every {o[case]['interval']:,.0f} cycles)")
        ax.set_title(CASE_TITLE[case] + ", detected at 2 mm", fontsize=9.5, color=INK, loc="left")
        ax.set_xticks(range(0, 11))
    axes[0].set_ylabel("U90 of remaining life (decades)")
    axes[1].legend(fontsize=7.5, loc="upper right")
    fig.text(0.01, 0.01, "Median over the true cracks; band = middle half. Detection and repeat readings use the same "
             "technique's sizing sd (BS 7910 Table T.2, T.7.5.2; sd = E/1.3).", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_bs_observations.png"))


if __name__ == "__main__":
    what = sys.argv[1:] or ["main", "obs"]
    if "main" in what:
        r = load("bs7910.json")
        fig_sweep(r)
        fig_rank(r)
        fig_laws(r)
    if "obs" in what:
        fig_obs(load("observations_bs7910.json"))
    print("figures written")
