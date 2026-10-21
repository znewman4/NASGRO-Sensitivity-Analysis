"""Figures for the scenario results (reads results/*.json, writes results/fig_*.png)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nasgro_scenarios import runner  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
})
OUT = runner.RESULTS


def _delta_plot(ax, rows, title, key="median"):
    for i, (label, xs, ys, style) in enumerate(rows):
        ax.plot(xs, ys, style, color=SERIES[i], label=label, markersize=7,
                markeredgecolor=SURF, markeredgewidth=1.5)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([0.2, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.2", "0.5", "1", "2", "4"])
    ax.set_xlabel("Initial crack depth [mm]")
    ax.set_title(title, loc="left", fontsize=11)


def _heat(ax, res_list, title, min_red=0.02, max_rows=8, always=()):
    depths = [r["depth_mm"] for r in res_list]
    names = list(res_list[0]["reduction"])
    M = np.array([[r["reduction"][k] for r in res_list] for k in names])
    keep = np.nanmax(M, axis=1) >= min_red
    order = np.argsort(-np.nanmax(np.where(keep[:, None], M, -1), axis=1))
    order = [i for i in order if keep[i]][:max_rows]
    order += [names.index(k) for k in always if k in names and names.index(k) not in order]
    M = M[order]
    ax.imshow(np.clip(M, 0, None), cmap="Blues", vmin=0, vmax=0.4, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.25 else INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i].replace("(C, m, p, dKth0)", "").replace("kappa", "κ")
                        .replace("beta", "β").replace("dKth0", "ΔKth0").strip() for i in order])
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([f"{d:g}" for d in depths])
    ax.set_xlabel("Initial crack depth [mm]")
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def residual_figures():
    rs = runner.load("residual_stress.json")["results"]
    by = {}
    for r in rs:
        by.setdefault(r["scenario"], []).append(r)
    for v in by.values():
        v.sort(key=lambda r: r["depth_mm"])
    labels = {
        "bar_tensile": "Tensile profile", "bar_tensile_sd": "Tensile, shakedown",
        "bar_compressive": "Compressive (rolled/peened)", "bar_uniform": "Uniform tensile (check C)",
        "plate_weld": "As-welded profile", "plate_weld_sd": "As-welded, shakedown",
        "plate_uniform": "Uniform tensile (check C)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, case, title in [(axes[0], "bar", "Round bar, R = −1, 250 MPa"),
                            (axes[1], "plate", "Plate surface crack, R = 0.1, 200 MPa")]:
        rows = []
        for nm in [k for k in labels if k.startswith(case)]:
            xs = [r["depth_mm"] for r in by[nm]]
            ys = [r["stats"]["median"] - r["baseline_stats"]["median"] for r in by[nm]]
            rows.append((labels[nm], xs, ys, "-o"))
        _delta_plot(ax, rows, title)
        ax.legend(frameon=False, fontsize=9, loc="lower left")
        ax.set_ylim(-1.6, 0.2)
    u4 = [r for r in by["plate_uniform"] if r["depth_mm"] == 4.0][0]
    d4 = u4["stats"]["median"] - u4["baseline_stats"]["median"]
    axes[1].annotate(f"4 mm: {d4:.1f} decades\nfirst-cycle fracture", xy=(2.3, -1.55),
                     xytext=(2.45, -0.75), fontsize=9, color=INK2,
                     arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    axes[0].set_ylabel("Change in median log10 life [decades]")
    fig.suptitle("Residual stress: change in median remaining life against the constant-amplitude baseline",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_rs_life.png"), dpi=150)

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.9))
    rs_rows = ("residual stress amplitude", "residual stress decay length")
    for ax, nm in zip(axes, ["bar_tensile", "bar_uniform", "bar_compressive", "plate_weld"]):
        _heat(ax, by[nm], ("Bar: " if nm.startswith("bar") else "Plate: ") + labels[nm][0].lower() + labels[nm][1:],
              max_rows=7, always=rs_rows)
    fig.suptitle("Residual stress: U90 reduction when one input is known exactly "
                 "(inputs below 2 % at every depth omitted)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_rs_rank.png"), dpi=150)


def va_figures():
    va = runner.load("variable_amplitude.json")["results"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, case, title in [(axes[0], "bar", "Round bar, R = −1"), (axes[1], "plate", "Plate, R = 0.1")]:
        rows = []
        for st, lab, style in [(1, "Spectrum, no interaction", "-o"), (2, "Spectrum + Willenborg", "s")]:
            rr = sorted([r for r in va if r["case"] == case and r["stage"] == st], key=lambda r: r["depth_mm"])
            rows.append((lab, [r["depth_mm"] for r in rr],
                         [r["stats"]["median"] - r["baseline_stats"]["median"] for r in rr], style))
        _delta_plot(ax, rows, title)
        ax.set_ylim(-0.05, 0.42)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=9.5, ncol=2, loc="upper left", bbox_to_anchor=(0.06, 0.93))
    axes[0].set_ylabel("Change in median log10 life [decades]")
    fig.suptitle("Variable amplitude: change in median remaining life against constant amplitude "
                 "(same damage-equivalent amplitude)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, "fig_va_life.png"), dpi=150)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.0), gridspec_kw=dict(width_ratios=[4, 5, 2, 3]))
    for ax, case, st, title in [(axes[0], "bar", 1, "Bar: spectrum, no interaction"),
                                (axes[1], "plate", 1, "Plate: spectrum, no interaction"),
                                (axes[2], "bar", 2, "Bar: + Willenborg"),
                                (axes[3], "plate", 2, "Plate: + Willenborg")]:
        rr = sorted([r for r in va if r["case"] == case and r["stage"] == st], key=lambda r: r["depth_mm"])
        extra = ["spectrum shape h", "block length n_b"]
        if st == 2:
            extra += ["shut-off ratio SOLR", "overload plastic-zone factor beta"]
        _heat(ax, rr, title, max_rows=6 if st == 1 else 5, always=extra)
    fig.suptitle("Variable amplitude: U90 reduction when one input is known exactly "
                 "(Willenborg at 3,000 samples, so about ±3 % noise)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_va_rank.png"), dpi=150)


if __name__ == "__main__":
    which = sys.argv[1:] or ["rs", "va"]
    if "rs" in which:
        residual_figures()
    if "va" in which:
        va_figures()
