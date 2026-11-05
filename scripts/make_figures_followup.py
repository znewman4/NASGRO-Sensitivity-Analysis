"""Figures for the follow-up questions (Giannella gap, theta and model choice, extra observations).

Reads giannella_bridge.json, model_discrimination.json and observations.json from
results/2026-10-23_nasgro_only/ and writes fig_giannella_bridge.png,
fig_model_discrimination.png, fig_obs_example.png, fig_obs_u90.png and
fig_obs_learned.png there.
"""
import os
import sys

import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make_figures as mf  # noqa: E402,F401  (shared style)
from make_figures import GRID, INK, INK2, SERIES, SURF, plt  # noqa: E402
from nasgro_scenarios import runner  # noqa: E402

D = runner.STANDARD_DIR
runner.RESULTS = D
CASE_T = {"bar": "Round bar, 25CrMo4, R = −1, 250 MPa", "plate": "Plate surface crack, R = 0.1, 200 MPa"}
PREC_LABEL = {"0.5": "±0.5 mm (typical UT)", "0.17": "±0.17 mm (a third of UT)", "0.05": "±0.05 mm (near exact)"}
PREC_COLOR = {"0.5": SERIES[0], "0.17": SERIES[1], "0.05": SERIES[2]}
SIGMA_1MM = {"bar": 0.36, "plate": 0.38}   # σ of log10 life at 1 mm, NASGRO-only baseline (Appendix A)


def save(fig, name):
    path = os.path.join(D, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(path)


# --- Question 1 ---------------------------------------------------------------------------------

def fig_bridge():
    b = runner.load("giannella_bridge.json")
    steps = b["bridge"]
    paper_sd = b["paper"]["stress"]["250"][1]
    vals = [paper_sd] + [s["stats"]["sigma"] for s in steps]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
    y = np.arange(len(vals))[::-1]
    for i, (yy, v) in enumerate(zip(y, vals)):
        if i < 2:
            ax.barh(yy, v, color=SERIES[0] if i == 1 else INK2, height=0.62, alpha=1 if i == 1 else 0.55)
            ax.text(v + 0.005, yy, f"{v:.3f}", va="center", fontsize=9, color=INK)
        else:
            lo, hi = vals[i - 1], v
            ax.barh(yy, hi - lo, left=lo, color=SERIES[1] if hi >= lo else SERIES[2], height=0.62)
            ax.plot([lo, lo], [yy + 0.31, yy + 1 - 0.31], color=INK2, lw=0.8)
            ax.text(max(hi, lo) + 0.005, yy, f"{hi:.3f}  ({hi - lo:+.3f})", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(["Giannella (2022), as published", "Giannella's setup in this engine",
                        "+ crack depth sd 0.15 → 0.5 mm", "+ load CoV 1.5 % → 10 %",
                        "+ Kc, a0, R, tilt uncertain", "+ this study's material model"])
    ax.set_xlim(0, max(vals) * 1.3)
    ax.set_xlabel("σ of log10 life, 1 mm crack, 250 MPa [decades]")
    ax.grid(axis="y", visible=False)
    ax.set_title("a. What widens the scatter, one change at a time", loc="left", fontsize=11)

    srcs = ["crack depth", "material", "load amplitude", "diameter D"]
    paper = {"crack depth": 1 - 0.086 / 0.118, "diameter D": 1 - 0.109 / 0.117, "load amplitude": 1 - 0.116 / 0.118}
    cols = [("Giannella (2022), Figs 12–14", INK2, paper),
            ("Giannella's setup, this engine", SERIES[0],
             {k: v["sigma_reduction"] for k, v in steps[0]["contributions"].items()}),
            ("This study", SERIES[1], {k: v["sigma_reduction"] for k, v in steps[-1]["contributions"].items()})]
    h, handles = 0.26, []
    yy = np.arange(len(srcs))[::-1]
    for j, (lbl, c, d) in enumerate(cols):
        for i, s in zip(yy, srcs):
            if s not in d:
                bx.text(0.004, i + (1 - j) * h, "not swept", va="center", fontsize=8, color=INK2)
                continue
            v = max(d[s], 0.0)
            bx.barh(i + (1 - j) * h, v, height=h - 0.03, color=c, alpha=0.55 if j == 0 else 1)
            bx.text(v + 0.006, i + (1 - j) * h, f"{v:.0%}", va="center", fontsize=8.5, color=INK)
        handles.append(Patch(color=c, alpha=0.55 if j == 0 else 1, label=lbl))
    bx.set_yticks(yy)
    bx.set_yticklabels(srcs)
    bx.set_xlim(0, 0.6)
    bx.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    bx.set_xlabel("σ reduction when the source is known exactly")
    bx.grid(axis="y", visible=False)
    bx.legend(handles=handles, loc="lower right", fontsize=8.5, frameon=False)
    bx.set_title("b. Ranking of sources", loc="left", fontsize=11)
    for a in (ax, bx):
        a.set_axisbelow(True)
    save(fig, "fig_giannella_bridge.png")


# --- Question 2 ---------------------------------------------------------------------------------

def fig_discrimination():
    d = runner.load("model_discrimination.json")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.6))
    for i, case in enumerate(("bar", "plate")):
        g = d["gap"][case]
        th = np.array(g["theta"])
        gap = np.array(g["spread"]) - np.array(g["spread_min"])
        arr = np.array(g["arrest"])
        ax.plot(th[~arr], gap[~arr], "-o", color=SERIES[i], markersize=6, markeredgecolor=SURF,
                markeredgewidth=1.5, label=f"{case}: gap between models")
        ax.axhline(SIGMA_1MM[case], color=SERIES[i], lw=1, ls=":")
        ax.text(0.5, SIGMA_1MM[case] + (0.012 if case == "plate" else -0.03),
                f"{case}: σ of life from the other inputs", fontsize=8, color=INK2)
    ax.axhspan(0, 0.1, color=GRID, alpha=0.7, lw=0)
    ax.text(31, 0.03, "below 0.1 decade: the model choice hardly matters", fontsize=8, color=INK2)
    ax.set_xlabel("Crack tilt θ from the plane normal to the stress [deg]")
    ax.set_ylabel("Spread of predicted log10 life\nacross the four models [decades]")
    ax.set_title("a. How far apart the models are (1 mm crack)", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_axisbelow(True)
    for i, case in enumerate(("bar", "plate")):
        idn = d["identification"][case]
        th = np.array([float(t) for t in idn])
        for k, ls, mk in (("3", "--", "s"), ("10", "-", "o")):
            p = np.array([idn[t][k]["mean"] if k in idn[t] else np.nan for t in idn])
            bx.plot(th + (i - 0.5) * 0.8, p, ls, marker=mk, color=SERIES[i], markersize=6, markeredgecolor=SURF,
                    markeredgewidth=1.5, label=f"{case}, {k} inspections")
    bx.axhline(0.25, color=INK2, lw=1, ls=":")
    bx.text(0.5, 0.21, "0.25 = no information (four equally likely models)", fontsize=8, color=INK2)
    bx.set_ylim(0, 1.02)
    bx.set_xlabel("Crack tilt θ, UT-sized to ±5° [deg]")
    bx.set_ylabel("Mean probability given to the true model")
    bx.set_title(f"b. Telling them apart from repeat depth inspections (±{d['sigma_m_mm']:g} mm)", loc="left",
                 fontsize=11)
    bx.legend(frameon=False, fontsize=8.5, loc="upper left")
    bx.set_axisbelow(True)
    save(fig, "fig_model_discrimination.png")


# --- Question 3 ---------------------------------------------------------------------------------

def fig_obs_example():
    o = runner.load("observations.json")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, case in zip(axes, ("bar", "plate")):
        ex = o[case]["example"]
        t = np.array(ex["t_grid"]) / 1e3
        pb = np.array(ex["prior_band"]) * 1e3
        ax.fill_between(t, pb[:, 0], pb[:, 2], color=GRID, lw=0, label="no repeat inspections (90 % band)")
        for key, c in (("0.5", SERIES[0]), ("0.05", SERIES[2])):
            p = ex["panels"][key]
            band = np.array(p["band"]) * 1e3
            lbl = f"after {p['k']} inspections, {PREC_LABEL[key]}"
            if key == "0.05":
                ax.fill_between(t, band[:, 0], band[:, 2], color=c, alpha=0.35, lw=0, label=lbl)
            else:                                  # outline, so the two bands stay distinguishable
                ax.plot(t, band[:, 0], color=c, lw=1.6, ls="--", label=lbl)
                ax.plot(t, band[:, 2], color=c, lw=1.6, ls="--")
            tk = np.array(ex["times"][:p["k"]]) / 1e3
            ax.errorbar(tk, np.array(p["y"]) * 1e3, yerr=float(key), fmt="o", color=c, markersize=6,
                        markeredgecolor=SURF, capsize=3, lw=1.2)
        ax.plot(t, np.array(ex["truth_depth"]) * 1e3, color=INK, lw=1.6, label="true crack")
        ax.axvline(ex["truth_life"] / 1e3, color=INK2, lw=1, ls=":")
        ax.set_ylim(0, min(np.nanmax(pb[:, 2]) * 1.05, 14))
        ax.set_xlim(0, t[-1])
        ax.set_xlabel("Cycles from detection [thousands]")
        ax.set_ylabel("Crack depth [mm]")
        ax.set_title(f"{case.capitalize()}: one example crack, inspected every {o[case]['interval'] / 1e3:g}k cycles",
                     loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_axisbelow(True)
    save(fig, "fig_obs_example.png")


def fig_obs_u90():
    o = runner.load("observations.json")
    rk = runner.load("rankings.json")["results"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, case in zip(axes, ("bar", "plate")):
        for key in ("0.5", "0.17", "0.05"):
            pk = o[case]["precisions"][key]["per_k"]
            k = np.array([p["k"] for p in pk])
            m = np.array([p["u90_median"] for p in pk])
            ax.fill_between(k, [p["u90_q25"] for p in pk], [p["u90_q75"] for p in pk], color=PREC_COLOR[key],
                            alpha=0.15, lw=0)
            ax.plot(k, m, "-o", color=PREC_COLOR[key], markersize=6, markeredgecolor=SURF, markeredgewidth=1.5,
                    label=PREC_LABEL[key])
            ax.text(k[-1] + 0.15, m[-1], f"{m[-1]:.2f}", va="center", fontsize=8.5, color=INK)
        base = o[case]["precisions"]["0.5"]["per_k"][0]["u90_median"]
        red = np.mean([r["reduction"]["crack depth"] for r in rk
                       if r["config"] == "standard" and r["case"] == case and r["depth_mm"] == 1.0])
        ax.axhline(base * (1 - red), color=INK2, lw=1, ls="--")
        below = case == "plate"
        ax.text(10.4 if not below else 0.3, base * (1 - red) + (-0.03 if below else 0.03),
                "crack depth known exactly, no repeats", fontsize=8, color=INK2,
                va="top" if below else "bottom", ha="left" if below else "right")
        ax.set_xticks(range(0, 11))
        ax.set_xlim(-0.3, 11)
        ax.set_ylim(0, 1.3)
        ax.set_xlabel(f"Repeat inspections after detection (every {o[case]['interval'] / 1e3:g}k cycles)")
        ax.set_title(CASE_T[case], loc="left", fontsize=11)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("U90 of remaining life [decades]\n(median over 300 true cracks; band = middle half)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left", title="Depth measurement error (sd)",
                   title_fontsize=8.5)
    save(fig, "fig_obs_u90.png")


def fig_obs_learned():
    o = runner.load("observations.json")
    key = "0.17"
    combined = "predicted life (all inputs combined)"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, case in zip(axes, ("bar", "plate")):
        sd = o[case]["precisions"][key]["sd_ratio"]
        last = max(sd, key=int)
        names = [n for n in sd["1"] if n != combined]
        top = [combined, names[0]]                             # life, then the combined growth rate
        order = top + sorted([n for n in names if n not in top], key=lambda n: sd[last][n])[:8]
        y = np.arange(len(order))[::-1]
        for k, c in (("1", SERIES[0]), ("3", SERIES[1]), ("6", SERIES[2])):
            if k not in sd:
                continue
            ax.plot([sd[k][n] for n in order], y, "o", color=c, markersize=8, markeredgecolor=SURF,
                    markeredgewidth=1.5, label=f"after {k} inspection{'s' if k != '1' else ''}")
        for yy in y:
            ax.axhline(yy, color=GRID, lw=0.8, zorder=0)
        ax.axvline(1, color=INK2, lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.set_xlim(0, 1.05)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Posterior sd / prior sd (1 = nothing learned)")
        ax.set_title(f"{case.capitalize()}: what the inspections pin down (±{key} mm)", loc="left", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_obs_learned.png")


# --- Question 3, expanded: the RUL predictions themselves, and periodic vs continuous ---------------

K_SHOW_COLORS = ["#b9c7d6", "#7fb0e8", "#2a78d6", "#0d3f80"]      # 0, 1, 3, last: light -> dark (ordinal)
REGIME_STYLE = {"periodic 0.5": (SERIES[0], "-", "periodic UT, ±0.5 mm, 1 reading per interval"),
                "periodic 0.17": (SERIES[1], "-", "periodic, ±0.17 mm, 1 reading per interval"),
                "periodic 0.05": (SERIES[2], "-", "periodic, ±0.05 mm, 1 reading per interval"),
                "continuous 0.5": (SERIES[6], "--", "continuous sensor, ±0.5 mm, 10 readings per interval"),
                "continuous 0.5 + bias 0.25": (SERIES[7], ":", "continuous, ±0.5 mm + unknown fixed offset ±0.25 mm")}


def fig_obs_rul():
    """RUL predictions after each inspection for the example crack, one panel per precision."""
    o = runner.load("inspection_regimes.json")
    keys = ("0.5", "0.17", "0.05")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), sharex="row", sharey="row")
    rng = np.random.default_rng(3)
    for r, case in enumerate(("bar", "plate")):
        ex = o[case]["example"]
        T, dt = ex["truth_life"], o[case]["interval"]
        for c, key in enumerate(keys):
            ax = axes[r, c]
            reg = ex["regimes"][f"periodic {key}"]
            col = PREC_COLOR[key]
            for smp, band in zip(reg["samples"], reg["bands"]):
                k = smp["k"]
                rul = (np.array(smp["life"][:300]) - k * dt) / 1e3
                x = k * dt / 1e3 + rng.uniform(-0.28, 0.28, len(rul)) * dt / 1e3
                ax.scatter(x, rul, s=4, color=col, alpha=0.35, lw=0, rasterized=True)
                lo, hi = 10 ** band["q05"] / 1e3, 10 ** band["q95"] / 1e3
                ax.plot([k * dt / 1e3] * 2, [lo, hi], color=INK, lw=1.2)
                for yy in (lo, hi):
                    ax.plot([k * dt / 1e3 - 0.18 * dt / 1e3, k * dt / 1e3 + 0.18 * dt / 1e3], [yy, yy], color=INK,
                            lw=1.2)
            first, last = reg["bands"][0], reg["bands"][-1]
            tt = np.linspace(0, T * 0.995, 400)
            ax.plot(tt / 1e3, (T - tt) / 1e3, color=INK, lw=1.4, ls="--")
            ks = np.array([b["k"] for b in reg["bands"]])
            ax.plot(ks * dt / 1e3, (T - ks * dt) / 1e3, "D", color=INK, markersize=5, markeredgecolor=SURF,
                    markeredgewidth=1.0, zorder=5)
            ax.set_yscale("log")
            ax.set_ylim(T / 1e3 * 0.03, T / 1e3 * 12)
            ax.set_xlim(-0.6 * dt / 1e3, T / 1e3 * 1.02)
            f0 = 10 ** (first["q95"] - first["q05"])
            f1 = 10 ** (last["q95"] - last["q05"])
            ax.text(0.98, 0.97, f"90 % band: factor {f0:.0f} wide at detection,\n"
                                f"factor {f1:.1f} after {last['k']} inspections",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK)
            ax.set_title(f"{case.capitalize()}: {PREC_LABEL[key]}", loc="left", fontsize=10.5)
            ax.set_axisbelow(True)
            if r == 1:
                ax.set_xlabel("Cycles since detection [thousands]")
        axes[r, 0].set_ylabel("Predicted remaining life [thousand cycles]")
    axes[0, 0].plot([], [], color=INK, lw=1.4, ls="--", label="true remaining life")
    axes[0, 0].plot([], [], color=INK, lw=1.2, label="90 % band of the prediction")
    axes[0, 0].scatter([], [], s=12, color=INK2, label="individual predictions (ensemble members)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_obs_rul.png")


def fig_obs_giannella(case):
    """Giannella-style: predicted RUL (relative to the true RUL) against detected depth, and its density,
    after 0, 1, 3 and the last inspection before failure, for one crack."""
    o = runner.load("inspection_regimes.json")[case]
    ex = o["example"]
    T, dt = ex["truth_life"], o["interval"]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 8.4), gridspec_kw=dict(width_ratios=[1.15, 1, 0.9]))
    for r, key in enumerate(("0.5", "0.17", "0.05")):
        reg = ex["regimes"][f"periodic {key}"]
        smp, bands = reg["samples"], reg["bands"]
        show = [0, 1, 3, len(smp) - 1]
        sax, pax, tax = axes[r]
        bins = np.linspace(-1.3, 1.3, 79)
        lines = []
        for j, i in enumerate(show):
            s, bd = smp[i], bands[i]
            tk = s["k"] * dt
            ratio = (np.array(s["life"]) - tk) / (T - tk)
            sax.scatter(ratio, np.array(s["a_i"]) * 1e3, s=3 if j < 3 else 4, color=K_SHOW_COLORS[j],
                        alpha=0.5 if j < 3 else 0.8, lw=0, rasterized=True)
            y = np.log10(ratio)
            pax.hist(y, bins=bins, density=True, histtype="step", color=K_SHOW_COLORS[j], lw=1.5)
            lines.append((s["k"], (T - tk) / 1e3, np.std(y), 10 ** (bd["q95"] - bd["q05"])))
        sax.axvline(1, color=INK, lw=1.2, ls="--")
        sax.plot([1], [ex["truth_a_i"] * 1e3], "x", color=INK, markersize=9, mew=2)
        pax.axvline(0, color=INK, lw=1.2, ls="--")
        sax.set_xscale("log")
        sax.set_xlim(0.05, 20)
        sax.set_ylim(0, 3)
        sax.set_ylabel("Depth at detection [mm]", fontsize=9)
        sax.set_title(f"Depth readings {PREC_LABEL[key]}", loc="left", fontsize=10.5)
        pax.set_ylabel("PDF", fontsize=9)
        tax.axis("off")
        tax.text(0.0, 0.95, f"{'after':<15}{'true RUL':>10}{'σ':>7}{'90 % band':>11}", family="monospace",
                 fontsize=8.5, color=INK2, transform=tax.transAxes, va="top")
        for j, (k, rul, sg, fac) in enumerate(lines):
            lbl = "detection" if k == 0 else f"{k} inspection{'s' if k > 1 else ''}"
            tax.text(0.0, 0.78 - 0.17 * j, f"{lbl:<15}{format(rul, '.0f') + 'k':>10}{sg:>7.3f}"
                     f"{'×' + format(fac, '.1f'):>11}", family="monospace", fontsize=8.5, color=INK,
                     transform=tax.transAxes, va="top")
            tax.plot([-0.05], [0.745 - 0.17 * j], "s", color=K_SHOW_COLORS[j], markersize=7,
                     transform=tax.transAxes, clip_on=False)
        for a in (sax, pax):
            a.set_axisbelow(True)
    axes[2, 0].set_xlabel("Predicted remaining life ÷ true remaining life  (dashed / ×: the true crack)")
    axes[2, 1].set_xlabel("log10 (predicted RUL ÷ true RUL)")
    fig.suptitle(f"{CASE_T[case]}: one crack detected at 1 mm, inspected every {o['interval'] / 1e3:g}k cycles. "
                 "RUL predictions at detection and after 1, 3 and the last inspection before failure\n"
                 "(light to dark). σ: sd of log10 RUL; 90 % band: ratio of its 95th to 5th percentile", x=0.01,
                 ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, f"fig_obs_giannella_{case}.png")


def fig_obs_regimes():
    """(a) the readings each regime gives on the example bar crack; (b, c) U90 against cycles since detection."""
    o = runner.load("inspection_regimes.json")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), gridspec_kw=dict(width_ratios=[1, 1, 1]))
    ax = axes[0]
    ex = o["bar"]["example"]
    T = ex["truth_life"]
    rp = ex["regimes"]["periodic 0.5"]
    rc = ex["regimes"]["continuous 0.5"]
    obs = runner.load("observations.json")["bar"]["example"]
    t_true = np.array(obs["t_grid"]) / 1e3
    ax.plot(t_true, np.array(obs["truth_depth"]) * 1e3, color=INK, lw=1.6, label="true depth")
    ok = np.array(rc["readings_t"]) < T
    ax.plot(np.array(rc["readings_t"])[ok] / 1e3, np.array(rc["readings_y"])[ok] * 1e3, "o",
            color=SERIES[6], markersize=3.5, alpha=0.8, label="continuous sensor: 10 readings per interval, ±0.5 mm")
    okp = np.array(rp["readings_t"]) < T
    ax.plot(np.array(rp["readings_t"])[okp] / 1e3, np.array(rp["readings_y"])[okp] * 1e3, "s",
            color=SERIES[0], markersize=8, markeredgecolor=SURF, markeredgewidth=1.5,
            label="periodic UT: 1 reading per interval, ±0.5 mm")
    ax.set_xlim(0, T / 1e3 * 1.02)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlabel("Cycles since detection [thousands]")
    ax.set_ylabel("Crack depth [mm]")
    ax.set_title("a. What each regime sees (example bar crack)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_axisbelow(True)
    for ax, case, tag in ((axes[1], "bar", "b"), (axes[2], "plate", "c")):
        dt = o[case]["interval"]
        for name, (col, ls, lbl) in REGIME_STYLE.items():
            pk = o[case]["regimes"][name]
            k = np.array([p["k"] for p in pk])
            m = np.array([p["u90_median"] for p in pk])
            ax.plot(k * dt / 1e3, m, ls=ls, marker="o", color=col, markersize=5, markeredgecolor=SURF,
                    markeredgewidth=1.2, label=lbl)
        ax.set_ylim(0, 1.3)
        ax.set_xlabel("Cycles since detection [thousands]")
        ax.set_title(f"{tag}. {CASE_T[case]}", loc="left", fontsize=11)
        ax.set_axisbelow(True)
    axes[1].set_ylabel("U90 of remaining life [decades]\n(median over 300 true cracks)")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.66, -0.06))
    fig.tight_layout()
    save(fig, "fig_obs_regimes.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["bridge", "disc", "example", "u90", "learned", "rul", "gstyle", "regimes"]
    for w in which:
        {"bridge": fig_bridge, "disc": fig_discrimination, "example": fig_obs_example,
         "u90": fig_obs_u90, "learned": fig_obs_learned, "rul": fig_obs_rul,
         "gstyle": lambda: [fig_obs_giannella(c) for c in ("bar", "plate")], "regimes": fig_obs_regimes}[w]()
