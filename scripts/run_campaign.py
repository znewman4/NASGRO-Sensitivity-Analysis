"""Campaign runs: what better knowledge buys, and how robust and transferable the rankings are.

  reduction      variance-reduction curves: each group's spread cut progressively, others unchanged
  pairs          paired reductions: is information valuable only when another uncertainty is cut too?
  physics        separate physics from knowledge: equal +/-10 %, constant crack knowledge, staged knowledge
  scenarios      loading/scenario matrix: stress amplitude, stress ratio and geometry
  distributions  alternative shapes, widths and dependencies: are the shortlisted targets robust?
  models         consistently calibrated growth models: does model disagreement outweigh precise inputs?
Fixed decision horizons are part of scripts/run_life_stages.py.

Usage:
    python scripts/run_campaign.py [runs ...] [--cases round_bar plate] [--n 2000] [--n-sobol 512]
                                   [--seed 21] [--out results/campaign] [--quick]
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

from nasgro_sa import campaign as cp
from nasgro_sa import giannella as g
from nasgro_sa.cases import plate_case, round_bar_case
from nasgro_sa.plotting import DIVERGING, GRID, INK_2, SEQUENTIAL, SLOTS, SURFACE, heatmap, label, ordered, style

MM = 1e-3
RUNS = ("reduction", "pairs", "physics", "scenarios", "distributions", "models")


def mm(a):
    return f"{a / MM:g} mm"


def fmt_top(indices, k=3):
    return ", ".join(f"{label(gr)} {indices[gr]:.2f}" for gr in cp.top_k(indices, k)) or "not estimable"


def write_rows(path, header, rows):
    with open(path, "w", newline="") as f:   # csv quotes names that contain commas
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([f"{v:.4g}" if isinstance(v, float) else v for v in r] for r in rows)


def index_matrix(results, columns):
    """Rows: union of groups (ordered); columns: the given keys of ``results``; nan where not estimable."""
    groups = ordered(list(dict.fromkeys(gr for c in columns for gr in results[c].sobol.groups)))
    m = np.full((len(groups), len(columns)), np.nan)
    texts = [[None] * len(columns) for _ in groups]
    for j, c in enumerate(columns):
        ind = cp.total_indices(results[c])
        for i, gr in enumerate(groups):
            if gr not in results[c].sobol.groups:
                texts[i][j] = "-"
            elif gr in ind:
                m[i, j] = ind[gr]
    return groups, m, texts


# ---------------------------------------------------------------- reduction curves

def report_reduction(case, curves, out):
    print("\nVARIANCE-REDUCTION CURVES, every input/group (U90 reduction in decades; others unchanged)")
    print("  plateau = remaining spread fraction at which 80 % of the achievable U90 reduction is reached")
    rows = []
    for a, rc in curves.items():
        print(f"  {mm(a)}: U90 {rc.base['u90']:.2f} decades")
        print(f"    {'group':<24}{'halve spread':>13}{'remove':>9}{'plateau':>9}{'median':>9}{'5 %':>8}")
        half = list(rc.factors).index(0.5)
        for grp in rc.ranking():
            red = rc.reduction(grp)
            print(f"    {label(grp):<24}{red[half]:>+13.3f}{red[-1]:>+9.3f}{rc.plateau(grp):>9.2f}"
                  f"{rc.shifts(grp, 'log_q50')[-1]:>+9.3f}{rc.shifts(grp, 'log_q05')[-1]:>+8.3f}")
            rows += [(a / MM, grp, f, r, m["u90"], m["log_q50"], m["log_q05"], m["p_no_propagation"])
                     for f, r, m in zip(rc.factors, red, rc.curves[grp])]
    write_rows(out / "reduction_curves.csv",
               ("a0_mm", "group", "remaining_spread_fraction", "u90_reduction", "u90", "log_q50", "log_q05",
                "p_no_prop"), rows)

    union = sorted({grp for rc in curves.values() for grp in rc.curves},
                   key=lambda grp: -max(np.nan_to_num(rc.achievable(grp)) for rc in curves.values()))
    colors = {grp: (SLOTS[i] if i < len(SLOTS) else INK_2) for i, grp in enumerate(union)}
    fig, axes = plt.subplots(1, len(curves), figsize=(4.4 * len(curves), 4.0), constrained_layout=True)
    for ax, (a, rc) in zip(np.atleast_1d(axes), curves.items()):
        reduction = 100 * (1 - np.array(rc.factors))
        for grp in rc.ranking()[::-1]:
            ax.plot(reduction, rc.reduction(grp), "o-", color=colors[grp], lw=2 if colors[grp] != INK_2 else 1,
                    ms=3, label=label(grp))
        ax.set_title(f"a0 = {mm(a)}", loc="left", fontsize=9)
        ax.set_xlabel("Reduction of that input's spread [%]")
        ax.grid(color=GRID, lw=0.6)
    np.atleast_1d(axes)[0].set_ylabel("U90 reduction [decades]")
    handles = [plt.Line2D([], [], color=colors[grp], lw=2, marker="o", ms=3) for grp in union[:len(SLOTS)]]
    fig.legend(handles, [label(grp) for grp in union[:len(SLOTS)]] , loc="outside lower center", ncol=4,
               fontsize=7.5)
    fig.suptitle(f"Variance-reduction curves, every input/group (grey: beyond the top eight) - {case.title}", x=0.01,
                 ha="left", fontsize=10)
    fig.savefig(out / "reduction_curves.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- paired reductions

def report_pairs(case, pairs, out):
    if not pairs:
        print("\nPAIRED REDUCTIONS: fewer than two inputs at every key stage")
        return
    print(f"\nPAIRED REDUCTIONS (each spread cut to {next(iter(pairs.values())).factor:.0%}; U90 reductions, decades)")
    print("  synergy = pair reduction - sum of single reductions; > 0: worth more together, < 0: overlapping")
    rows = []
    for a, pr in pairs.items():
        s = pr.synergy()
        print(f"  {mm(a)}: single reductions " + ", ".join(f"{label(gr)} {pr.single[gr]:+.3f}" for gr in pr.groups))
        ranked = sorted(((s[i, j], i, j) for i in range(len(pr.groups)) for j in range(i + 1, len(pr.groups))),
                        reverse=True)
        for v, i, j in ranked[:3]:
            gi, gj = pr.groups[i], pr.groups[j]
            print(f"    {label(gi)} + {label(gj)}: pair {pr.pair[(gi, gj)]:+.3f} vs singles "
                  f"{pr.single[gi] + pr.single[gj]:+.3f}  -> synergy {v:+.3f}")
        rows += [(a / MM, gi, gj, pr.pair[(gi, gj)], pr.single[gi], pr.single[gj], s[i, j])
                 for i, gi in enumerate(pr.groups) for j, gj in enumerate(pr.groups) if j > i]
    write_rows(out / "paired_reductions.csv",
               ("a0_mm", "group_1", "group_2", "pair_reduction", "reduction_1", "reduction_2", "synergy"), rows)

    fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 4.2), constrained_layout=True)
    for ax, (a, pr) in zip(np.atleast_1d(axes), pairs.items()):
        s = pr.synergy()
        vmax = max(0.02, np.abs(s[~np.eye(len(s), dtype=bool)]).max())
        texts = [[f"single\n{s[i, i]:+.3f}" if i == j else None for j in range(len(s))] for i in range(len(s))]
        shown = s.copy()
        np.fill_diagonal(shown, 0.0)
        heatmap(ax, shown, pr.groups, [label(gr) for gr in pr.groups], DIVERGING, -vmax, vmax, "{:+.3f}",
                xlabel="", texts=texts)
        ax.tick_params(axis="x", rotation=40)
        ax.set_title(f"a0 = {mm(a)}", loc="left", fontsize=9)
    fig.suptitle(f"Paired reductions: synergy in U90 reduction [decades] (blue = complementary) - {case.title}",
                 x=0.01, ha="left", fontsize=10)
    fig.savefig(out / "paired_reductions.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- physics vs knowledge

def report_physics(case, stages, out):
    print("\nPHYSICS VERSUS KNOWLEDGE (Sobol total indices, failure by median life)")
    print("  screen = every input +/-10 % (physics only); constant = catalogue uncertainty with the same crack")
    print("  knowledge at every stage; staged = crack knowledge by detection stage (the Q3 analysis)")
    print(f"  {'a0':>7}  {'tau scr-con':>11} {'tau con-stg':>11}  top 3 (constant)  |  top 3 (staged)")
    variants = ("screen", "constant", "staged")
    rows, prev = [], {}
    for st in stages:
        ind = {v: cp.total_indices(getattr(st, v)) for v in variants}
        drift = {v: cp.rank_agreement(prev[v], ind[v]) if prev.get(v) else np.nan for v in variants}
        print(f"  {mm(st.a):>7}  {cp.rank_agreement(ind['screen'], ind['constant']):>11.2f} "
              f"{cp.rank_agreement(ind['constant'], ind['staged']):>11.2f}  {fmt_top(ind['constant'])}  |  "
              f"{fmt_top(ind['staged'])}")
        rows.append((st.a / MM, cp.rank_agreement(ind["screen"], ind["constant"]),
                     cp.rank_agreement(ind["constant"], ind["staged"]), drift["constant"], drift["staged"],
                     ind["constant"].get("a_init", np.nan), ind["staged"].get("a_init", np.nan)))
        prev = ind
    print("  stage-to-stage tau (constant knowledge: physics only; staged: physics + knowledge):")
    print("   " + ", ".join(f"{r[0]:g}mm {r[3]:.2f}/{r[4]:.2f}" for r in rows[1:]))
    write_rows(out / "physics_vs_knowledge.csv",
               ("a0_mm", "tau_screen_vs_constant", "tau_constant_vs_staged", "tau_constant_vs_previous_stage",
                "tau_staged_vs_previous_stage", "ST_crack_depth_constant", "ST_crack_depth_staged"), rows)

    cols = [f"{st.a / MM:g}" for st in stages]
    titles = ("Physics only: every input +/-10 %", "Industry uncertainty, crack knowledge constant",
              "Industry uncertainty, crack knowledge by detection stage")
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), constrained_layout=True)
    for ax, v, title in zip(axes, variants, titles):
        res = {c: getattr(st, v) for c, st in zip(cols, stages)}
        groups, m, texts = index_matrix(res, cols)
        heatmap(ax, m, groups, cols, SEQUENTIAL, 0.0, 1.0, texts=texts)
        ax.set_title(title, loc="left", fontsize=9)
    fig.suptitle(f"What changes the ranking - Sobol total index, failure by median life ('-' = not an uncertain "
                 f"input in that analysis) - {case.title}", x=0.01, ha="left", fontsize=10)
    fig.savefig(out / "physics_vs_knowledge.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- scenarios and distribution variants

def report_matrix(title, case, results, base_key, out, stem):
    """Shared report for scenario and variant studies: transfer of the ranking to each column."""
    print(f"\n{title} (Sobol total indices, failure by median life)")
    rows = []
    for a, res in results.items():
        base = cp.total_indices(res[base_key])
        shortlist = cp.top_k(base, 3)
        print(f"  {mm(a)}: shortlist under '{base_key}': " + ", ".join(label(gr) for gr in shortlist))
        print(f"    {'':<28}{'tau':>6}{'kept':>6}  top 3")
        for name, r in res.items():
            ind = cp.total_indices(r)
            kept = sum(gr in cp.top_k(ind, 3) for gr in shortlist) if ind else 0
            tau = cp.rank_agreement(base, ind)
            print(f"    {name:<28}{tau:>6.2f}{kept:>4}/3  {fmt_top(ind)}")
            rows.append((a / MM, name, tau, kept, *(cp.top_k(ind, 3) + [""] * 3)[:3]))
    write_rows(out / f"{stem}.csv", ("a0_mm", "column", "tau_vs_base", "shortlist_kept", "top1", "top2", "top3"),
               rows)

    fig, axes = plt.subplots(1, len(results), figsize=(5.6 * len(results), 6.6), constrained_layout=True)
    for ax, (a, res) in zip(np.atleast_1d(axes), results.items()):
        cols = list(res)
        groups, m, texts = index_matrix(res, cols)
        heatmap(ax, m, groups, cols, SEQUENTIAL, 0.0, 1.0, xlabel="", texts=texts)
        ax.tick_params(axis="x", rotation=60)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        ax.set_title(f"a0 = {mm(a)}", loc="left", fontsize=9)
    fig.suptitle(f"{title.capitalize()} - Sobol total index, failure by median life - {case.title}", x=0.01,
                 ha="left", fontsize=10)
    fig.savefig(out / f"{stem}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- model comparison

def report_models(case, comparisons, out):
    print("\nMODEL COMPARISON (consistently calibrated; typical industry uncertainty)")
    names = list(comparisons[0].models)
    rows = []
    for mc in comparisons:
        print(f"  {mm(mc.a)}")
        for name in names:
            m = mc.models[name]
            print(f"    {name:<8} P(no prop.) {m['p_no_propagation']:.2f}   5 % 10^{m['log_q05']:.2f}   "
                  f"median 10^{m['log_q50']:.2f}   U90 {m['u90']:.3f}")
            rows.append((mc.a / MM, name, m["p_no_propagation"], m["log_q05"], m["log_q50"], m["u90"]))
        print(f"    model disagreement: median {mc.disagreement['log_q50']:.3f}, 5 % {mc.disagreement['log_q05']:.3f}, "
              f"U90 {mc.disagreement['u90']:.3f} decades;  U90 reduction from exact crack characterisation "
              f"{mc.benefit['perfect crack characterisation']:+.3f}, all non-material inputs known "
              f"{mc.benefit['all non-material inputs known']:+.3f}")
    write_rows(out / "model_comparison.csv", ("a0_mm", "model", "p_no_prop", "log10_q05", "log10_q50", "u90"), rows)
    write_rows(out / "model_vs_inputs.csv",
               ("a0_mm", "disagreement_log_q05", "disagreement_log_q50", "disagreement_u90",
                "u90_reduction_crack_characterisation", "u90_reduction_all_inputs"),
               [(mc.a / MM, mc.disagreement["log_q05"], mc.disagreement["log_q50"], mc.disagreement["u90"],
                 mc.benefit["perfect crack characterisation"], mc.benefit["all non-material inputs known"])
                for mc in comparisons])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    x = np.arange(len(comparisons))
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        xs = x - 0.4 + width * (k + 0.5)
        q05 = [mc.models[name]["log_q05"] for mc in comparisons]
        q50 = [mc.models[name]["log_q50"] for mc in comparisons]
        ax.vlines(xs, q05, q50, color=SLOTS[k], lw=2)
        ax.plot(xs, q05, "o", color=SLOTS[k], ms=7, mec=SURFACE, label=f"{name}: 5 % life (dot) to median")
        ax.plot(xs, q50, "o", color=SURFACE, ms=7, mec=SLOTS[k], mew=1.6)
    ax.set_xticks(x, [mm(mc.a) for mc in comparisons])
    ax.set_ylabel("log10 remaining life [cycles]")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("Remaining life by growth model (filled: 5 %, open: median)", loc="left", fontsize=9)
    series = (("Model disagreement (median life)", SLOTS[1], [mc.disagreement["log_q50"] for mc in comparisons]),
              ("U90 reduction: crack qualities exact", SLOTS[0],
               [mc.benefit["perfect crack characterisation"] for mc in comparisons]),
              ("U90 reduction: all non-material inputs exact", SLOTS[2],
               [mc.benefit["all non-material inputs known"] for mc in comparisons]))
    for k, (name, color, vals) in enumerate(series):
        bx.bar(x - 0.3 + 0.3 * k, vals, width=0.28, color=color, label=name, edgecolor=SURFACE, linewidth=1.5)
        for xi, v in zip(x - 0.3 + 0.3 * k, vals):
            bx.text(xi, v + 0.004, f"{v:.2f}", ha="center", fontsize=7, color=INK_2)
    bx.set_xticks(x, [mm(mc.a) for mc in comparisons])
    bx.set_ylabel("Decades")
    bx.grid(axis="y", color=GRID, lw=0.6)
    bx.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    bx.set_title("Does model disagreement outweigh the benefit of precise inputs?", loc="left", fontsize=9)
    fig.suptitle(f"Model comparison - {case.title}", x=0.01, ha="left", fontsize=10)
    fig.savefig(out / "model_comparison.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=list(RUNS), help=f"any of {', '.join(RUNS)} (default: all)")
    ap.add_argument("--cases", nargs="+", default=["round_bar", "plate"])
    ap.add_argument("--n", type=int, default=2000, help="Monte Carlo samples per run (reduction, pairs, models)")
    ap.add_argument("--n-sobol", type=int, default=512, help="Sobol base sample size (physics, scenarios, distributions)")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", default="results/campaign")
    ap.add_argument("--quick", action="store_true", help="small samples, for checking the pipeline")
    args = ap.parse_args()
    unknown = set(args.runs) - set(RUNS)
    if unknown:
        ap.error(f"unknown runs: {', '.join(sorted(unknown))}")
    if args.quick:
        args.n, args.n_sobol = 200, 64
    style()
    pool = g.load_or_build_pool("results/giannella")[0]
    builders = {"round_bar": lambda: round_bar_case(pool), "plate": plate_case}

    for name in args.cases:
        case = builders[name]()
        out = Path(args.out) / name
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed)
        t0 = time.perf_counter()
        print(f"\n{'=' * 100}\n{case.title}\n{'=' * 100}")
        if "reduction" in args.runs or "pairs" in args.runs:
            curves = {a: cp.reduction_curves(case, a, args.n, rng) for a in case.key_stages}
            if "reduction" in args.runs:
                report_reduction(case, curves, out)
            if "pairs" in args.runs:
                pairs = {a: cp.paired_reductions(case, a, curves[a].ranking()[:5], args.n, rng)
                         for a in case.key_stages}
                report_pairs(case, pairs, out)
        if "physics" in args.runs:
            report_physics(case, cp.physics_vs_knowledge(case, args.n_sobol, rng), out)
        if "scenarios" in args.runs:
            report_matrix("loading / scenario matrix", case, cp.scenario_matrix(case, args.n_sobol, rng),
                          case.scenarios[0][0], out, "scenarios")
        if "distributions" in args.runs:
            report_matrix("alternative distributions and dependencies", case,
                          cp.variant_study(case, args.n_sobol, rng), "typical", out, "distributions")
        if "models" in args.runs:
            models = (cp.bar_models(case, g.load_or_build_paris_pool("results/giannella")[0])
                      if name == "round_bar" else cp.plate_models(case, rng))
            report_models(case, cp.model_comparison(case, models, args.n, rng), out)
        print(f"\n[{name}] done in {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"\nTables and figures written to {args.out}")


if __name__ == "__main__":
    main()
