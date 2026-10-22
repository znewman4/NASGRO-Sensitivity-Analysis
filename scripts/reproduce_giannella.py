"""Reproduce Giannella (2022, Int J Fract 235:179-195), Figs 9 and 11-15, and compare with the published values.

Usage:
    python scripts/reproduce_giannella.py [--n 20000] [--pool 20000] [--q 1] [--seed 7] [--out results/giannella]

Prints ours-vs-paper tables and writes figures to --out. The accepted
material pool is cached there, keyed by q and seed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from nasgro_sa import giannella as g
from nasgro_sa.sobol import saltelli_blocks, sobol_indices

MM = 1e-3
# Reference palette (dataviz skill): categorical slots 1-3 and a sequential blue ramp.
OURS, PAPER, OURS_ALT = "#2a78d6", "#eb6834", "#1baf7a"
LEVELS = ["#184f95", "#3987e5", "#86b6ef", "#cde2fb"]    # largest sigma darkest
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
UNITS = {"a_init": ("Initial crack depth a", "mm", 1 / MM), "D": ("Diameter D", "mm", 1 / MM),
         "F": ("Load amplitude F", "kN", 1.0)}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "legend.frameon": False,
    })


def row(label, ours, paper):
    p = "" if paper is None else f"{paper[0]:7.3f} {paper[1]:7.3f} {paper[2]:6.0%}"
    return f"  {label:<14}{ours[0]:7.3f} {ours[1]:7.3f} {ours[2]:6.0%}   |  {p}"


HEADER = f"  {'':<14}{'mu':>7} {'sigma':>7} {'Pf':>6}   |  {'mu':>7} {'sigma':>7} {'Pf':>6}\n  {'':<14}{'--- ours ---':^22}   |  {'--- paper ---':^22}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000, help="Monte Carlo samples per case")
    ap.add_argument("--pool", type=int, default=20_000, help="accepted material sets")
    ap.add_argument("--n-sobol", type=int, default=4096)
    ap.add_argument("--q", type=float, default=g.FIXED["q"], help=f"NASGRO q (Table 2 prints {g.Q_TABLE:g})")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/giannella")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    style()

    fixed = {**g.FIXED, "q": args.q}
    mat, rate = g.load_or_build_pool(args.out, args.pool, args.seed, fixed)
    corr = np.corrcoef([np.log10(mat["C"]), mat["n"], mat["p"], mat["dK0"]])
    print(f"Material pool: {len(mat['C'])} accepted sets (acceptance rate {rate:.2e}), q = {args.q:g}")
    print("  corr  C-m {:.2f}  C-p {:.2f}  C-dK {:.2f}  m-p {:.2f}  m-dK {:.2f}  p-dK {:.2f}".format(
        corr[0, 1], corr[0, 2], corr[0, 3], corr[1, 2], corr[1, 3], corr[2, 3]))
    print("  paper -0.99       0.88        -0.47        -0.81       0.40        -0.70   (Fig. 8)")

    rng = np.random.default_rng(args.seed + 1)
    form, spec = g.formulation(), g.input_spec(mat, fixed)
    d_nom = g.NOMINAL["D"][0] * MM

    # Fig. 9 and 11: baseline and stress levels (load cv held at Table 4's 1.5 %)
    print("\nStress levels (Figs 9, 11): log10 Nf over propagating cracks\n" + HEADER)
    stress = {}
    for s in (150, 200, 250, 300, 400, 500):
        f_mean = s * np.pi * d_nom**2 / 4 * 1e6 / 1e3
        sp = spec.with_nominal(F=f_mean).scaled({"F": f_mean / g.NOMINAL["F"][0]})
        life = form.run(sp.sample(args.n, rng))
        stress[s] = (np.log10(life.cycles[life.failed]), g.log_life_stats(life))
        print(row(f"{s} MPa", stress[s][1], g.PAPER["baseline"] if s == 250 else g.PAPER["stress"][s]))

    # Figs 12-14: one-at-a-time sigma sweeps with common random numbers
    sweeps = {}
    for target in ("a_init", "D", "F"):
        print(f"\n{UNITS[target][0]} sigma sweep (Fig. {12 + list(UNITS).index(target)})\n" + HEADER)
        w = spec.standard_normal(args.n, rng)
        sweeps[target] = {}
        for level, paper in g.PAPER[target].items():
            x = spec.scaled({target: level / g.NOMINAL[target][1]}).transform(w)
            life = form.run(x)
            sweeps[target][level] = (x[target] * UNITS[target][2], life.cycles, life.failed, g.log_life_stats(life))
            print(row(f"sigma = {level:g} {UNITS[target][1]}", sweeps[target][level][3], paper))

    # Fig. 15: contributions, by variance removed when a source is fixed, and by Sobol total effect
    w = spec.standard_normal(args.n, rng)
    life = form.run(spec.transform(w))
    var_all = np.var(np.log10(life.rul[life.failed]))
    removal = {}
    for key in g.SOURCES:
        life = form.run(spec.scaled({key: 0.0}).transform(w))
        removal[key] = var_all - np.var(np.log10(life.rul[life.failed]))
    groups = spec.uncertain_groups()
    ws = saltelli_blocks(spec.standard_normal(args.n_sobol, rng), spec.standard_normal(args.n_sobol, rng),
                         [c for _, c in groups])
    life = form.run(spec.transform(ws))
    y = np.log10(np.minimum(life.rul, form.settings.horizon))
    sob = sobol_indices(y, args.n_sobol, [lbl for lbl, _ in groups])
    total = dict(zip(sob.groups, sob.total[:, 0]))
    share = lambda d: {g.SOURCES[k]: max(d[k], 0.0) / sum(max(v, 0.0) for v in d.values()) for k in g.SOURCES}
    rem_share, sob_share = share(removal), share(total)
    print("\nContributions (Fig. 15), normalised shares")
    print(f"  {'source':<12}{'paper':>8}{'ours: variance removed':>25}{'ours: Sobol total':>20}")
    for k, name in g.SOURCES.items():
        print(f"  {name:<12}{g.PAPER['contributions'][name]:8.0%}{rem_share[name]:25.0%}{sob_share[name]:20.0%}")
    print(f"  (Sobol first-order sum {sob.first[:, 0].sum():.2f}, total sum {sob.total[:, 0].sum():.2f}; "
          f"{np.mean(~life.failed):.2%} of Sobol runs did not fail)")

    plot_sweep_summary(sweeps, Path(args.out) / "sweep_summary.png")
    plot_sweep_scatter(sweeps, Path(args.out) / "sweep_scatter.png")
    plot_stress(stress, Path(args.out) / "stress_levels.png")
    plot_contributions(rem_share, sob_share, Path(args.out) / "contributions.png")
    print(f"\nFigures written to {args.out}")


def plot_sweep_summary(sweeps, path):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharey="row", constrained_layout=True)
    for j, (target, levels) in enumerate(sweeps.items()):
        lv = np.array(sorted(levels))
        ours = np.array([levels[l][3] for l in lv])
        paper = np.array([g.PAPER[target][l] for l in lv])
        for i, (k, label) in enumerate(((1, "sigma of log10 Nf"), (0, "mean of log10 Nf"))):
            ax = axes[i, j]
            ax.plot(lv, ours[:, k], color=OURS, lw=2, marker="o", ms=6, label="This reproduction")
            ax.plot(lv, paper[:, k], color=PAPER, lw=2, ls="--", marker="s", ms=6, label="Giannella (2022)")
            ax.axvline(g.NOMINAL[target][1], color=INK_2, lw=0.8, ls=":")
            ax.set_xscale("log")
            if j == 0:
                ax.set_ylabel(label)
            if i == 1:
                ax.set_xlabel(f"Input sigma: {UNITS[target][0]} [{UNITS[target][1]}]")
            else:
                ax.set_title(UNITS[target][0], loc="left", fontsize=10)
    axes[0, 0].legend(loc="upper left")
    axes[0, 0].annotate("Table 4 nominal", (g.NOMINAL["a_init"][1], axes[0, 0].get_ylim()[0]),
                        xytext=(3, 3), textcoords="offset points", color=INK_2, fontsize=8)
    fig.suptitle("Life scatter vs input uncertainty, one input at a time (cf. Figs 12-14)", x=0.01, ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweep_scatter(sweeps, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    for ax, (target, levels) in zip(axes, sweeps.items()):
        for color, level in zip(LEVELS, sorted(levels, reverse=True)):
            value, cycles, failed, _ = levels[level]
            k = np.flatnonzero(failed)[:4000]
            ax.scatter(cycles[k], value[k], s=3, color=color, alpha=0.7, linewidths=0,
                       label=f"sigma = {level:g} {UNITS[target][1]}")
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0e}".replace("e+0", "e")))
        ax.set_xlabel("Fatigue life Nf [cycles]")
        ax.set_ylabel(f"{UNITS[target][0]} [{UNITS[target][1]}]")
        ax.legend(loc="upper right", markerscale=3, fontsize=8, frameon=True, facecolor=SURFACE,
                  edgecolor=GRID, framealpha=0.9)
    fig.suptitle("Input value vs predicted life at each uncertainty level (cf. Figs 12-14)", x=0.01, ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stress(stress, path):
    levels = sorted(stress)
    fig, axes = plt.subplots(len(levels), 1, figsize=(7.5, 9), sharex=True, constrained_layout=True)
    grid = np.linspace(3.5, 9.5, 400)
    for ax, s in zip(axes, levels):
        logs, (mu, sd, pf) = stress[s]
        pm, ps, ppf = g.PAPER["baseline"] if s == 250 else g.PAPER["stress"][s]
        ax.hist(logs, bins=np.arange(3.5, 9.6, 0.05), density=True, color=OURS, alpha=0.55, edgecolor=SURFACE,
                linewidth=0.3)
        ax.plot(grid, np.exp(-0.5 * ((grid - pm) / ps) ** 2) / (ps * np.sqrt(2 * np.pi)), color=PAPER, lw=2)
        ax.set_yticks([])
        ax.text(0.01, 0.82, f"{s} MPa", transform=ax.transAxes, fontweight="bold")
        ax.text(0.99, 0.80, f"ours: mu {mu:.2f}, sigma {sd:.3f}, Pf {pf:.0%}\npaper: mu {pm:.2f}, sigma {ps:.3f}, Pf {ppf:.0%}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK_2)
    axes[0].plot([], [], color=OURS, lw=6, alpha=0.55, label="This reproduction (histogram)")
    axes[0].plot([], [], color=PAPER, lw=2, label="Giannella (2022), normal fit")
    axes[0].legend(loc="upper center", ncol=2, fontsize=8)
    axes[-1].set_xlabel("log10 Nf (propagating cracks only)")
    fig.suptitle("Life distributions at each stress level (cf. Figs 9, 11)", x=0.01, ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_contributions(removal, sobol, path):
    names = list(g.PAPER["contributions"])
    series = [("Giannella (2022), Fig. 15", PAPER, g.PAPER["contributions"]),
              ("Ours: variance removed by fixing the source", OURS, removal),
              ("Ours: Sobol total effect", OURS_ALT, sobol)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)
    h = 0.26
    for i, (label, color, vals) in enumerate(series):
        y = np.arange(len(names)) + (i - 1) * (h + 0.02)
        ax.barh(y, [vals[n] for n in names], height=h, color=color, label=label)
        for yy, n in zip(y, names):
            ax.text(vals[n] + 0.008, yy, f"{vals[n]:.0%}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(np.arange(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("Share of life scatter (normalised)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Ranking of uncertainty sources (cf. Fig. 15)", loc="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
