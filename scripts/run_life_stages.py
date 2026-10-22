"""Questions 1-3 across the crack's life, for both cases.

  Q1  Which part of life is being evaluated: where the remaining life is spent
      (near-threshold, Paris, near-critical) and whether the crack grows at all.
  Q2  What NASGRO is most sensitive to, independent of input uncertainty, and
      how that changes with crack size.
  Q3  What matters once industry-typical uncertainty is applied, and how that
      changes with crack size (crack-size knowledge depends on detection).

Usage:
    python scripts/run_life_stages.py [--cases round_bar plate] [--n 1024] [--n-levels 512]
                                      [--n-map 300] [--seed 11] [--out results/life_stages] [--quick]

Writes CSV tables and figures to --out/<case>/ and prints the answer tables.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from nasgro_sa import giannella as g
from nasgro_sa.cases import LABELS, plate_case, round_bar_case
from nasgro_sa.integrate import REGIMES
from nasgro_sa.lifestage import OUTPUTS, declared_label, regime_map, run_stage
from nasgro_sa.plotting import (CRITICAL, DIVERGING, EMPTY, GRID, INK, INK_2, SEQUENTIAL, SLOTS, SURFACE, heatmap,
                                ordered, style)
from nasgro_sa.uncertainty import CATALOGUE, CRACK, DETECTION_LIMIT, LEVELS, crack_stage, relative_size_uncertainty

MM = 1e-3
MEDIAN = OUTPUTS.index("fail by median life")


def stage_labels(case):
    return [f"{a / MM:g}" for a in case.stages]


# ---------------------------------------------------------------- tables

def sobol_matrix(stages, pick, output=MEDIAN):
    """Total indices (groups x stages) for one output; nan where not estimable."""
    groups = ordered(list(dict.fromkeys(gr for s in stages for gr in pick(s).sobol.groups)))
    m = np.full((len(groups), len(stages)), np.nan)
    for j, s in enumerate(stages):
        res = pick(s)
        if res.estimable[output]:
            for i, gr in enumerate(groups):
                if gr in res.sobol.groups:
                    m[i, j] = res.sobol.total[res.sobol.groups.index(gr), output]
    return groups, m


def write_csv(path, rows, cols, matrix, corner="input"):
    with open(path, "w", newline="") as f:   # csv quotes labels that contain commas
        w = csv.writer(f)
        w.writerow([corner, *cols])
        w.writerows([LABELS.get(r, r), *(f"{v:.4g}" for v in vals)] for r, vals in zip(rows, matrix))


def tables(case, stages, out):
    cols = [f"a0={c}mm" for c in stage_labels(case)]
    summaries = [s.summary(case.declared_horizons) for s in stages]
    keys = list(summaries[0])
    with open(out / "stages.csv", "w") as f:
        f.write("a0_mm,knowledge," + ",".join(keys) + ",screen_h05,screen_h50,industry_h05,industry_h50\n")
        for s, sm in zip(stages, summaries):
            f.write(f"{s.a / MM:g},{crack_stage(s.a)}," + ",".join(f"{sm[k]:.4g}" for k in keys)
                    + "," + ",".join(f"{h:.4g}" for h in (*s.screen.horizons, *s.industry['typical'].horizons)) + "\n")
    local = ordered(list(case.varied))
    write_csv(out / "local_effects.csv", local, cols,
              np.array([[s.local.effect[k] for s in stages] for k in local]))
    write_csv(out / "local_bounded.csv", local, cols,
              np.array([[float(s.local.bounded[k]) for s in stages] for k in local]))
    for o, oname in enumerate(stages[0].screen.sobol.outputs):
        tag = oname.replace(" ", "_").replace("%", "pc").replace("+", "")
        write_csv(out / f"screen_ST_{tag}.csv", *sobol_matrix(stages, lambda s: s.screen, o)[:1], cols,
                  sobol_matrix(stages, lambda s: s.screen, o)[1])
        for level in LEVELS:
            groups, m = sobol_matrix(stages, lambda s, lv=level: s.industry[lv], o)
            write_csv(out / f"industry_{level}_ST_{tag}.csv", groups, cols, m)
    rate_groups = ordered(stages[0].rate.groups)
    write_csv(out / "rate_ST.csv", rate_groups, cols,
              np.array([[s.rate.total[s.rate.groups.index(gr), 0] for s in stages] for gr in rate_groups]))
    return summaries


def top(groups, values, k=5):
    order = [i for i in np.argsort(-np.nan_to_num(values, nan=-1.0)) if np.isfinite(values[i])][:k]
    return ", ".join(f"{LABELS.get(groups[i], groups[i])} {values[i]:.2f}" for i in order) or "not estimable"


def kendall(stage, level):
    a, b = stage.industry["typical"], stage.industry[level]
    if not (a.estimable[MEDIAN] and b.estimable[MEDIAN]):
        return np.nan
    common = [gr for gr in a.sobol.groups if gr in b.sobol.groups]
    x = [a.sobol.total[a.sobol.groups.index(gr), MEDIAN] for gr in common]
    y = [b.sobol.total[b.sobol.groups.index(gr), MEDIAN] for gr in common]
    return float(stats.kendalltau(x, y).statistic)


def print_answers(case, stages, summaries):
    print(f"\n{'=' * 100}\n{case.title}\n{'=' * 100}")
    for note in case.notes:
        print(f"  note: {note}")
    print("\nQ1  Where is the remaining life spent? (typical industry uncertainty; shares of cycles to failure)")
    print(f"  {'a0 mm':>6} {'knowledge':<11}{'P(no prop.)':>12}{'near-thr':>10}{'Paris':>8}{'near-crit':>10}"
          f"{'5% life':>11}{'median life':>13}")
    for s, sm in zip(stages, summaries):
        if np.isfinite(sm["share_threshold"]):
            shares = f"{sm['share_threshold']:>10.2f}{sm['share_paris']:>8.2f}{sm['share_critical']:>10.2f}"
        else:
            shares = f"{'(too few cracks grow)':>28}"
        print(f"  {s.a / MM:>6g} {crack_stage(s.a):<11}{sm['p_arrest']:>12.2f}{shares}"
              f"{sm['life_q05']:>11.3g}{sm['life_q50']:>13.3g}")

    print("\nQ2  NASGRO's own sensitivity (every input +/-10 %, R +/-0.1, tilt +/-10 deg)")
    print("  (SIF solution error is not a NASGRO input and acts exactly like stress, so it appears only in Q3)")
    print("  Sobol total indices, failure by the median remaining life (top 5):")
    for s in stages:
        res = s.screen
        val = res.sobol.total[:, MEDIAN] if res.estimable[MEDIAN] else np.full(len(res.sobol.groups), np.nan)
        print(f"  {s.a / MM:>6g} mm  {top(res.sobol.groups, val)}")
    print("  Local effect at the nominal state, change in log10 life for +10 % (* = a run did not fail):")
    local = ordered(list(case.varied))
    print(f"  {'':<22}" + "".join(f"{c:>8}" for c in stage_labels(case)))
    for k in local:
        cells = "".join(f"{s.local.effect[k]:>7.3f}{'*' if s.local.bounded[k] else ' '}" for s in stages)
        print(f"  {LABELS.get(k, k):<22}{cells}")

    print("\nQ3  Importance under industry-typical uncertainty (catalogue), failure by the median remaining life")
    print("  top 5 Sobol total indices; Kendall tau of the full ranking vs the low and high catalogue ends")
    for s in stages:
        res = s.industry["typical"]
        val = res.sobol.total[:, MEDIAN] if res.estimable[MEDIAN] else np.full(len(res.sobol.groups), np.nan)
        print(f"  {s.a / MM:>6g} mm ({crack_stage(s.a):<10})  {top(res.sobol.groups, val)}"
              f"   [tau low {kendall(s, 'low'):.2f}, high {kendall(s, 'high'):.2f}]")
    print("  Non-propagation (where 1-99 % of cracks arrest): top 3 Sobol total indices")
    for s in stages:
        res = s.industry["typical"]
        if res.estimable[OUTPUTS.index("no propagation")]:
            print(f"  {s.a / MM:>6g} mm  {top(res.sobol.groups, res.sobol.total[:, 2], 3)}")
    print("  Instantaneous growth rate at the matched state (log10 da/dN), top 3 Sobol total indices:")
    for s in stages:
        print(f"  {s.a / MM:>6g} mm  {top(s.rate.groups, s.rate.total[:, 0], 3)}"
              f"   ({s.rate_below_threshold:.0%} of samples below threshold)")

    print("\nFixed decision horizons (typical industry uncertainty): P(fail within N cycles) and top 3 drivers")
    print("  (population-relative horizons above ask what drives the spread of life; these ask what drives")
    print("   the probability of failing before a declared cycle count, e.g. the next inspection)")
    for h in case.declared_horizons:
        o = stages[0].industry["typical"].sobol.outputs.index(declared_label(h))
        print(f"  N = {h:.3g} cycles")
        for s, sm in zip(stages, summaries):
            res = s.industry["typical"]
            drivers = (top(res.sobol.groups, res.sobol.total[:, o], 3) if res.estimable[o]
                       else "(probability too close to 0 or 1 for stable indices)")
            print(f"    {s.a / MM:>6g} mm  P = {sm[f'p_fail_by_{h:.3g}']:.3f}   {drivers}")


# ---------------------------------------------------------------- figures

def detection_line(ax, case):
    k = sum(a < DETECTION_LIMIT for a in case.stages)
    if 0 < k < len(case.stages):
        ax.axvline(k - 0.5, color=INK, lw=1.0, ls="--")
        ax.text(k - 0.45, -0.7, "detected and sized ->", fontsize=7, color=INK_2, va="bottom")
        ax.text(k - 0.55, -0.7, "<- undetected / EICS", fontsize=7, color=INK_2, va="bottom", ha="right")


def plot_q1(case, rmap, stages, summaries, path):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    a_mm = rmap.a / MM
    edges = np.sqrt(a_mm[:-1] * a_mm[1:])
    x_edges = np.concatenate([[a_mm[0] ** 2 / edges[0]], edges, [a_mm[-1] ** 2 / edges[-1]]])
    ds = np.diff(rmap.s).mean()
    y_edges = np.concatenate([rmap.s - ds / 2, [rmap.s[-1] + ds / 2]])
    cm = SEQUENTIAL.copy()
    cm.set_bad((0, 0, 0, 0))
    for mask, color in ((rmap.critical_at_start, CRITICAL), (np.isnan(rmap.share[0]) & ~rmap.critical_at_start, EMPTY)):
        ax.pcolormesh(x_edges, y_edges, np.where(mask, 1.0, np.nan),
                      cmap=LinearSegmentedColormap.from_list("flat", [color, color]), vmin=0, vmax=1)
    mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(rmap.share[0]), cmap=cm, vmin=0, vmax=1)
    for level, ls in zip((0.1, 0.5, 0.9), (":", "-", "--")):
        ax.contour(a_mm, rmap.s, rmap.p_propagate, levels=[level], colors=INK, linewidths=1.1, linestyles=ls)
        ax.plot([], [], color=INK, lw=1.1, ls=ls, label=f"P(grows) = {level:.0%}")
    ax.axhline(case.stress_nominal, color=INK_2, lw=0.8, ls=":")
    ax.plot(np.array(case.stages) / MM, [case.stress_nominal] * len(case.stages), "o", ms=6, mfc=SURFACE,
            mec=INK, mew=1.2, label="stages studied")
    ax.fill([], [], color=EMPTY, label="nominal crack does not grow")
    ax.fill([], [], color=CRITICAL, label="critical at start")
    ax.axvline(DETECTION_LIMIT / MM, color=INK_2, lw=0.8, ls="--")
    ax.text(DETECTION_LIMIT / MM * 1.05, rmap.s[0], "detection limit", fontsize=7, color=INK_2, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Initial crack depth a0 [mm]")
    ax.set_ylabel("Nominal stress amplitude [MPa]")
    ax.set_title("Share of remaining life spent near threshold (nominal inputs)\n"
                 "contours: probability the crack grows under typical industry uncertainty",
                 loc="left", fontsize=8.5)
    fig.colorbar(mesh, ax=ax, label="share of cycles near threshold", shrink=0.85)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=7)

    x = np.arange(len(stages))
    bottom = np.zeros(len(stages))
    for k, (name, color) in enumerate(zip(REGIMES, SLOTS)):
        vals = np.array([sm[f"share_{('threshold', 'paris', 'critical')[k]}"] for sm in summaries])
        bx.bar(x, np.nan_to_num(vals), bottom=bottom, color=color, width=0.7, edgecolor=SURFACE, linewidth=1.5,
               label=name)
        bottom += np.nan_to_num(vals)
    for xi, sm in zip(x, summaries):
        bx.text(xi, 1.03, f"{sm['p_arrest']:.0%}", ha="center", fontsize=7, color=INK_2)
        if not np.isfinite(sm["share_threshold"]):
            bx.text(xi, 0.5, "too few\ncracks\ngrow", ha="center", va="center", fontsize=7, color=INK_2)
    bx.text(-0.6, 1.10, "P(no propagation):", fontsize=7, color=INK_2)
    bx.set_xticks(x, stage_labels(case))
    bx.set_ylim(0, 1.18)
    bx.set_xlabel("Initial crack depth a0 [mm]")
    bx.set_ylabel("Share of cycles to failure (cracks that fail)")
    bx.set_title("Where the remaining life is spent (typical industry uncertainty)", loc="left", fontsize=8.5)
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=7.5)
    bx.grid(axis="y", color=GRID, lw=0.6)
    fig.suptitle(f"Q1 - {case.title}", x=0.01, ha="left", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_q2(case, stages, path):
    local = ordered(list(case.varied))
    effect = np.array([[s.local.effect[k] for s in stages] for k in local])
    bounded = np.array([[s.local.bounded[k] for s in stages] for k in local])
    groups, st = sobol_matrix(stages, lambda s: s.screen)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 0.32 * len(local) + 1.9), constrained_layout=True)
    heatmap(ax, effect, local, stage_labels(case), DIVERGING, -0.5, 0.5, "{:+.2f}", bounded)
    ax.set_title("Local effect on log10 life of +10 % in each input (R +0.1, tilt 10 deg)\n"
                 "red = shorter life; colour clipped at +/-0.5 decades; * = a run did not fail", loc="left",
                 fontsize=8.5)
    heatmap(bx, st, groups, stage_labels(case), SEQUENTIAL, 0.0, 1.0)
    bx.set_title("Sobol total index, every input +/-10 % uniform (failure by median life)\n"
                 "n/e = not estimable (too few failures or survivors)", loc="left", fontsize=8.5)
    fig.suptitle(f"Q2 - NASGRO's own sensitivity - {case.title}", x=0.01, ha="left", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_q3(case, stages, path):
    groups, st = sobol_matrix(stages, lambda s: s.industry["typical"])
    groups_a, st_a = sobol_matrix(stages, lambda s: s.industry["typical"], OUTPUTS.index("no propagation"))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 0.34 * len(groups) + 2.2), constrained_layout=True)
    heatmap(ax, st, groups, stage_labels(case), SEQUENTIAL, 0.0, 1.0)
    detection_line(ax, case)
    ax.set_title("Sobol total index under typical industry uncertainty: failure by median life\n\n", loc="left",
                 fontsize=8.5)
    heatmap(bx, st_a, groups_a, stage_labels(case), SEQUENTIAL, 0.0, 1.0)
    detection_line(bx, case)
    bx.set_title("Sobol total index under typical industry uncertainty: whether the crack grows at all\n\n",
                 loc="left", fontsize=8.5)
    fig.suptitle(f"Q3 - importance once industry uncertainty is applied - {case.title}", x=0.01, ha="left",
                 fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fixed_horizons(case, stages, summaries, path):
    horizons = case.declared_horizons
    outputs = stages[0].industry["typical"].sobol.outputs
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.2 * len(horizons), 0.34 * 11 + 2.4),
                             constrained_layout=True, sharey=True)
    for ax, h in zip(np.atleast_1d(axes), horizons):
        groups, st = sobol_matrix(stages, lambda s: s.industry["typical"], outputs.index(declared_label(h)))
        cols = [f"{c}\nP={sm[f'p_fail_by_{h:.3g}']:.2f}" for c, sm in zip(stage_labels(case), summaries)]
        heatmap(ax, st, groups, cols, SEQUENTIAL, 0.0, 1.0)
        ax.set_title(f"Failure within N = {h:.3g} cycles", loc="left", fontsize=9)
    fig.suptitle(f"Fixed decision horizons - Sobol total index under typical industry uncertainty (P = probability "
                 f"of failing within N; n/e where P is too close to 0 or 1) - {case.title}", x=0.01, ha="left",
                 fontsize=9.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_catalogue(path):
    rel = {f"crack: {e.label}": e for e in (CRACK["undetected"]["a_init"], CRACK["sized"]["aspect_init"],
                                             CRACK["undetected"]["aspect_init"])}
    rel.update({e.label: e for k, e in CATALOGUE.items() if e.kind == "cov" and k != "F"})
    names = list(rel)[::-1]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 6.4), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.17, top=0.88, wspace=0.3)
    for i, name in enumerate(names):
        e = rel[name]
        color = SLOTS[0] if e.basis == "literature" else SLOTS[1]
        ax.plot([e.low, e.high], [i, i], color=color, lw=2, solid_capstyle="round")
        ax.plot(e.typical, i, "o", ms=7, color=color, mec=SURFACE, mew=1.2)
    ax.plot([], [], "o-", color=SLOTS[0], label="literature range")
    ax.plot([], [], "o-", color=SLOTS[1], label="judgement (to be sourced)")
    ax.set_yticks(range(len(names)), names)
    ax.set_xscale("log")
    ax.set_xlim(0.003, 1.5)
    ax.set_xlabel("Coefficient of variation (dot = typical, bar = low to high)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True, facecolor=SURFACE, edgecolor=GRID)
    ax.set_title("Industry-typical relative uncertainty by input", loc="left", fontsize=9)
    sd_notes = [f"{e.label}: {e.low:g} / {e.typical:g} / {e.high:g}" for e in
                (CATALOGUE["R"], CRACK["sized"]["a_init"], CRACK["sized"]["theta"], CRACK["undetected"]["theta"])]
    fig.text(0.01, 0.025, "Absolute standard deviations (low / typical / high; mm or deg):\n" + ";  ".join(sd_notes),
             fontsize=7, color=INK_2)

    a = np.geomspace(0.02, 8.0, 300) * MM
    for level, color in zip(("high", "typical", "low"), ("#184f95", "#3987e5", "#86b6ef")):
        bx.plot(a / MM, relative_size_uncertainty(a, level), color=color, lw=2, label=f"{level} catalogue end")
    bx.axvline(DETECTION_LIMIT / MM, color=INK_2, lw=0.8, ls="--")
    bx.text(0.03, 0.97, "detection limit (dashed):\nundetected (EICS-type) left,\nUT-sized right", transform=bx.transAxes,
            fontsize=7, color=INK_2, va="top")
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlabel("Crack depth [mm]")
    bx.set_ylabel("Coefficient of variation of crack depth knowledge")
    bx.grid(color=GRID, lw=0.6)
    bx.legend(loc="lower left", fontsize=7.5)
    bx.set_title("Crack-size uncertainty changes over the crack's life", loc="left", fontsize=9)
    fig.suptitle("Industry-typical input uncertainty (sources in src/nasgro_sa/uncertainty.py)", x=0.01, ha="left",
                 fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["round_bar", "plate"])
    ap.add_argument("--n", type=int, default=1024, help="Sobol base sample size (screen, typical, rate)")
    ap.add_argument("--n-levels", type=int, default=512, help="Sobol base sample size at the low/high ends")
    ap.add_argument("--n-map", type=int, default=300, help="Monte Carlo samples per regime-map cell")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="results/life_stages")
    ap.add_argument("--quick", action="store_true", help="small samples, for checking the pipeline")
    args = ap.parse_args()
    if args.quick:
        args.n, args.n_levels, args.n_map, args.n_boot = 128, 64, 40, 20
    style()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    plot_catalogue(out_root / "uncertainty_catalogue.png")

    builders = {"round_bar": lambda: round_bar_case(g.load_or_build_pool("results/giannella")[0]),
                "plate": plate_case}
    for name in args.cases:
        case = builders[name]()
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed)
        t0 = time.perf_counter()
        rmap = regime_map(case, np.geomspace(0.03 * MM, 0.9 * case.a_limit, 12), rng, n_mc=args.n_map)
        stages = []
        for a in case.stages:
            stages.append(run_stage(case, a, rng, n_screen=args.n, n_industry=args.n, n_levels=args.n_levels,
                                    n_rate=args.n, n_boot=args.n_boot))
            print(f"[{name}] a0 = {a / MM:g} mm done ({time.perf_counter() - t0:.0f} s)", flush=True)
        summaries = tables(case, stages, out)
        print_answers(case, stages, summaries)
        plot_q1(case, rmap, stages, summaries, out / "q1_life_regimes.png")
        plot_q2(case, stages, out / "q2_sensitivity.png")
        plot_q3(case, stages, out / "q3_importance.png")
        plot_fixed_horizons(case, stages, summaries, out / "fixed_horizons.png")
    print(f"\nTables and figures written to {out_root}")


if __name__ == "__main__":
    main()
