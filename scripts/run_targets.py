"""Measurement-target map: which crack quality is worth measuring, and how precisely.

Main analysis (src/nasgro_sa/targets.py). For each case it reports:
  * every input/group: reduction in U90, the 90 % RUL band (decades), when its uncertainty is reduced;
  * each crack quality (size, shape, orientation, opening/closure): U90 against measurement precision,
    with material and loading uncertainty kept, at every crack state, repeated across scenarios and at
    the low and high catalogue ends;
  * Giannella-style figures: baseline versus reduced-uncertainty RUL distributions, then measurement sd
    versus RUL spread;
  * the measurement-target map: achievable reduction, required precision, diminishing-return point and
    applicable regime for each candidate.
Supporting: Sobol total indices at each state.

Usage:
    python scripts/run_targets.py [--cases round_bar plate] [--n 2000] [--sobol-n 512] [--seed 31]
                                  [--out results/targets] [--quick]
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

from nasgro_sa import giannella as g
from nasgro_sa import targets as tg
from nasgro_sa.campaign import total_indices
from nasgro_sa.cases import CATEGORY, plate_case, round_bar_case
from nasgro_sa.plotting import GRID, INK, INK_2, SEQUENTIAL, SLOTS, SURFACE, heatmap, label, style
from nasgro_sa.uncertainty import UT_CAPABILITY

MM = 1e-3
STATES = {"round_bar": (0.5, 1.0, 2.0, 4.0), "plate": (0.2, 0.5, 1.0, 2.0, 4.0)}   # mm
SCENARIOS = {"round_bar": None,   # all
             "plate": ("S=200 R=0.1 (base)", "S=120 R=0.1", "S=280 R=0.1", "S=200 R=-1", "S=200 R=0.5", "t=20mm",
                       "a/c=0.3")}
FIG_CANDIDATES = ("size (depth)", "shape (a/c)", "orientation (energy, 30 deg)", "opening / closure")
COLOURS = {"size": SLOTS[0], "shape": SLOTS[1], "orientation": SLOTS[2], "opening": SLOTS[3], "floor": SLOTS[6]}
ORIENT_COLOURS = dict(zip((c.name for c in tg.ORIENTATION), (SLOTS[2], SLOTS[5], SLOTS[6], SLOTS[7])))
GROUP_COLOURS = {"crack": SLOTS[0], "loading": SLOTS[2], "analysis": "#8a8984", "geometry": "#8a8984"}


def mm(a):
    return f"{a / MM:g} mm"


def fmt_sd(cand: tg.Candidate, sd: float) -> str:
    if not np.isfinite(sd):
        return "-"
    if cand.key == "size":
        return f"{sd:.2g} mm"
    if cand.key == "orientation":
        return f"{sd:.2g} deg"
    return f"{sd:.2g}"


def pct(x: float) -> str:
    return f"{x:+.0%}" if np.isfinite(x) else "n/e"


def write_rows(path, header, rows):
    with open(path, "w", newline="") as f:   # csv quotes names that contain commas
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([f"{v:.4g}" if isinstance(v, float) else v for v in r] for r in rows)


# ---------------------------------------------------------------- tables

def tables(run: tg.TargetRun, out: Path):
    results = [*run.base, *(r for r in run.scenarios if r not in run.base), *run.levels]
    sweeps = [(r, s) for r in results for s in r.sweeps.values()]
    write_rows(out / "sweeps.csv",
               ("scenario", "level", "a0_mm", "candidate", "sd", "u90", "log_q50", "log_q05", "p_no_prop",
                "baseline_u90", "baseline_u90_se", "floor_u90"),
               [(r.scenario, r.level, r.a / MM, s.candidate.name, sd, p["u90"], p["log_q50"], p["log_q05"],
                 p["p_no_propagation"], s.baseline["u90"], s.baseline.get("u90_se", float("nan")), s.floor["u90"])
                for r, s in sweeps for sd, p in zip(s.sds, s.points)])
    write_rows(out / "target_map.csv",
               ("scenario", "level", "a0_mm", "candidate", "baseline_u90", "achievable_dec", "achievable_share",
                "required_sd", "diminishing_sd", "substantial", "coarse_achievable_dec", "coarse_required_sd",
                "median_shift_exact", "q05_shift_exact", "p_no_prop_baseline", "p_no_prop_exact",
                "crack_removable_share"),
               [(r.scenario, r.level, r.a / MM, s.candidate.name, s.baseline["u90"], s.achievable, s.achievable_share,
                 s.required_sd, s.diminishing_sd, s.substantial, s.coarse_achievable, s.coarse_required_sd,
                 s.shift("log_q50"), s.shift("log_q05"), s.baseline["p_no_propagation"],
                 s.points[-1]["p_no_propagation"], s.crack_removable_share) for r, s in sweeps])
    rows = []
    for r in run.base:
        c = r.curves
        half = list(c.factors).index(0.5)
        for grp in c.ranking():
            rows.append((r.a / MM, grp, c.achievable(grp), c.achievable(grp) / c.base["u90"], c.reduction(grp)[half],
                         c.plateau(grp), c.shifts(grp, "log_q50")[-1], c.shifts(grp, "log_q05")[-1]))
    write_rows(out / "all_inputs_reduction.csv",
               ("a0_mm", "group", "u90_reduction_exact", "share_of_baseline", "u90_reduction_half_spread",
                "plateau_fraction", "median_shift_exact", "q05_shift_exact"), rows)
    write_rows(out / "shape_beyond_size.csv", ("a0_mm", "given", "u90_reduction_from_shape"),
               [(r.a / MM, k, v) for r in run.base for k, v in r.beyond_size.items()])
    if all(r.sobol for r in run.base):
        groups = list(dict.fromkeys(gr for r in run.base for gr in r.sobol.sobol.groups))
        write_rows(out / "supporting_sobol.csv", ("group", *[f"a0={r.a / MM:g}mm" for r in run.base]),
                   [(gr, *[total_indices(r.sobol).get(gr, float("nan")) for r in run.base]) for gr in groups])


# ---------------------------------------------------------------- printed report

def print_report(run: tg.TargetRun):
    case = run.case
    print(f"\n{'=' * 104}\n{case.title}\n{'=' * 104}")
    print("Declared measure: U90 = log10(N95/N05), the 90 % band of RUL among cracks that fail (decades).")
    print("Reference: industry-typical uncertainty of every input. Floor: all crack qualities known exactly.")

    print("\nUNCERTAINTY FLOOR (material, loading, geometry and SIF uncertainty that crack characterisation cannot "
          "remove)")
    print(f"  {'a0':>7}{'U90 now':>10}{'+/- MC':>8}{'floor':>8}{'removable':>11}{'P(no growth)':>14}")
    for r in run.base:
        s = r.sweeps["size (depth)"]
        print(f"  {mm(r.a):>7}{s.baseline['u90']:>10.3f}{s.baseline['u90_se']:>8.3f}{s.floor['u90']:>8.3f}"
              f"{s.crack_removable_share:>11.0%}{s.baseline['p_no_propagation']:>14.2f}")

    print("\nEVERY INPUT/GROUP: U90 reduction when known exactly (decades, share of U90), at half its spread, and the")
    print("shifts of the median and 5 % percentile when known exactly (decades); top 8 per state")
    for r in run.base:
        c = r.curves
        half = list(c.factors).index(0.5)
        print(f"  {mm(r.a)}")
        for grp in c.ranking()[:8]:
            print(f"    {label(grp):<26}{c.achievable(grp):>+8.3f} ({c.achievable(grp) / c.base['u90']:>4.0%})"
                  f"   half {c.reduction(grp)[half]:>+7.3f}   median {c.shifts(grp, 'log_q50')[-1]:>+7.3f}"
                  f"   5% {c.shifts(grp, 'log_q05')[-1]:>+7.3f}")

    print("\nMEASUREMENT-TARGET MAP (base scenario): achievable U90 reduction vs current knowledge [required precision for")
    print(f"{tg.REQUIRED_SHARE:.0%} of it | diminishing-return point]; '-' = not substantial (< {tg.SUBSTANTIAL:.0%} of U90)")
    names = [c.name for c in tg.CANDIDATES if c.name in run.base[0].sweeps]
    for name in names:
        cand = run.base[0].sweeps[name].candidate
        cells = []
        for r in run.base:
            s = r.sweeps[name]
            req = (f"{fmt_sd(cand, s.required_sd)} | {fmt_sd(cand, s.diminishing_sd)}" if s.substantial else "-")
            cells.append(f"{mm(r.a)}: {pct(s.achievable_share)} [{req}]")
        print(f"  {name:<34}" + ";  ".join(cells))
    print("  versus an uninformed reference (coarsest grid sd), for context:")
    for name in names:
        cand = run.base[0].sweeps[name].candidate
        cells = [f"{mm(r.a)}: {pct(r.sweeps[name].coarse_achievable / r.sweeps[name].baseline['u90'])} "
                 f"[{fmt_sd(cand, r.sweeps[name].coarse_required_sd)}]" for r in run.base]
        print(f"  {name:<34}" + ";  ".join(cells))

    print("\nROBUSTNESS at the key states: substantial in how many scenarios, and at the low / high catalogue ends")
    for name in [c.name for c in tg.SCENARIO_CANDIDATES if c.name in names]:
        cells = []
        for a in case.key_stages:
            scen = [r.sweeps[name] for r in run.scenarios if r.a == a and name in r.sweeps]
            lev = {r.level: r.sweeps[name] for r in run.levels if r.a == a and name in r.sweeps}
            shares = [s.achievable_share for s in scen if np.isfinite(s.achievable_share)]
            span = f"{min(shares):+.0%}..{max(shares):+.0%}" if shares else "n/e"
            cells.append(f"{mm(a)}: {sum(s.substantial for s in scen)}/{len(scen)} scen ({span}), "
                         f"low {pct(lev['low'].achievable_share)}, high {pct(lev['high'].achievable_share)}")
        print(f"  {name:<34}" + ";  ".join(cells))

    if run.base[0].beyond_size:
        print("\nSHAPE BEYOND SIZE: further U90 reduction from knowing a/c exactly once depth is measured")
        for r in run.base:
            u90 = r.sweeps["size (depth)"].baseline["u90"]
            print(f"  {mm(r.a)}: " + ";  ".join(f"{k}: {v:+.3f} ({v / u90:+.0%})" for k, v in r.beyond_size.items()))

    print("\nMEDIAN AND 5 % SHIFTS when the candidate is known exactly (decades; positive = longer life)")
    for name in names:
        print(f"  {name:<34}" + ";  ".join(f"{mm(r.a)}: median {r.sweeps[name].shift('log_q50'):+.3f}, "
                                           f"5% {r.sweeps[name].shift('log_q05'):+.3f}" for r in run.base))

    if all(r.sobol for r in run.base):
        print("\nSUPPORTING: Sobol total indices, failure by median life (typical uncertainty), top 5")
        for r in run.base:
            ind = total_indices(r.sobol)
            top = sorted(ind, key=lambda k: -ind[k])[:5]
            print(f"  {mm(r.a)}: " + (", ".join(f"{label(k)} {ind[k]:.2f}" for k in top) or "not estimable"))


# ---------------------------------------------------------------- figures

def plot_distributions(run: tg.TargetRun, path: Path):
    rows = [r for r in run.base if r.a in run.case.key_stages]
    cols = [n for n in FIG_CANDIDATES if n in rows[0].sweeps] + ["all crack qualities"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.3 * len(cols), 2.5 * len(rows)), constrained_layout=True,
                             squeeze=False)
    for i, r in enumerate(rows):
        for j, name in enumerate(cols):
            ax = axes[i, j]
            if name == "all crack qualities":
                base_key, red_key, colour = "baseline (projection, 0 deg)", "floor (projection, 0 deg)", COLOURS["floor"]
                s = r.sweeps["size (depth)"]
                u0, u1 = s.baseline["u90"], s.floor["u90"]
                p0, p1 = s.baseline["p_no_propagation"], s.floor["p_no_propagation"]
                dq50, dq05 = s.floor["log_q50"] - s.baseline["log_q50"], s.floor["log_q05"] - s.baseline["log_q05"]
            else:
                s = r.sweeps[name]
                cand = s.candidate
                base_key, red_key, colour = f"baseline ({cand.mechanics}, {cand.tilt:g} deg)", name, COLOURS[cand.key]
                u0, u1 = s.baseline["u90"], s.points[-1]["u90"]
                p0, p1 = s.baseline["p_no_propagation"], s.points[-1]["p_no_propagation"]
                dq50, dq05 = s.shift("log_q50"), s.shift("log_q05")
            data = [r.samples[base_key], r.samples[red_key]]
            lo, hi = np.percentile(np.concatenate(data), (0.5, 99.5))
            bins = np.linspace(lo, hi, 40)
            ax.hist(data[0], bins=bins, density=True, color="#bdbbb5", alpha=0.8, label="current knowledge")
            ax.hist(data[1], bins=bins, density=True, histtype="step", lw=2, color=colour, label="known exactly")
            ax.text(0.98, 0.97, f"U90 {u0:.2f} -> {u1:.2f} ({(u1 - u0) / u0:+.0%})\nmedian {dq50:+.2f}, 5% {dq05:+.2f}"
                    f"\nno growth {p0:.0%} -> {p1:.0%}", transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
                    color=INK_2)
            ax.set_yticks([])
            if i == 0:
                ax.set_title(name, loc="left", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"a0 = {mm(r.a)}")
            if i == len(rows) - 1:
                ax.set_xlabel("log10 RUL [cycles]")
    axes[0, 0].legend(fontsize=6.8, loc="upper left")
    fig.suptitle(f"RUL distributions: current knowledge vs each crack quality known exactly (material and loading "
                 f"uncertainty kept) - {run.case.title}", x=0.01, ha="left", fontsize=9.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _x(sds, cand_grid_min):
    """Plot positions: exact (sd = 0) sits left of the smallest finite sd on a log axis."""
    sds = np.asarray(sds, dtype=float)
    return np.where(sds > 0, sds, cand_grid_min / 2.5)


def plot_sd_vs_spread(run: tg.TargetRun, path: Path):
    names = [n for n in FIG_CANDIDATES if n in run.base[0].sweeps]
    shades = SEQUENTIAL(np.linspace(0.35, 1.0, len(run.base)))
    fig, axes = plt.subplots(3, len(names), figsize=(3.9 * len(names), 8.2), constrained_layout=True, squeeze=False)
    for j, name in enumerate(names):
        cand = run.base[0].sweeps[name].candidate
        smallest = min(min(sd for sd in r.sweeps[name].sds if sd > 0) for r in run.base)
        for k, r in enumerate(run.base):
            s = r.sweeps[name]
            x = _x(s.sds, smallest)
            axes[0, j].plot(x, s.values("u90"), "o-", color=shades[k], lw=1.8, ms=4, label=f"a0 = {mm(r.a)}")
            axes[0, j].axhline(s.baseline["u90"], color=shades[k], lw=0.8, ls=":")
            axes[0, j].axhline(s.floor["u90"], color=shades[k], lw=0.8, ls="--")
            axes[1, j].plot(x, s.values("log_q50") - s.baseline["log_q50"], "o-", color=shades[k], lw=1.8, ms=4)
            axes[2, j].plot(x, s.values("log_q05") - s.baseline["log_q05"], "o-", color=shades[k], lw=1.8, ms=4)
        ut = UT_CAPABILITY[tg.UT_KEY[cand.key]]
        for ax in axes[:, j]:
            ax.set_xscale("log")
            ax.grid(color=GRID, lw=0.6)
            ticks = sorted({*ax.get_xticks()} | {smallest / 2.5})
            if ut:
                ax.axvspan(ut[0], ut[2], color=COLOURS[cand.key], alpha=0.10, lw=0)
                ax.axvline(ut[1], color=COLOURS[cand.key], lw=0.8, alpha=0.6)
        for row in (1, 2):
            axes[row, j].axhline(0.0, color=INK_2, lw=0.6)
        axes[0, j].set_title(name + ("" if ut else "  (no established UT capability)"), loc="left", fontsize=8.5)
        axes[2, j].set_xlabel(f"measurement sd [{cand.unit}]  (leftmost = exact)")
    axes[0, 0].set_ylabel("U90, 90 % RUL band [decades]")
    axes[1, 0].set_ylabel("median shift [decades]")
    axes[2, 0].set_ylabel("5 % percentile shift [decades]")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Measurement precision vs RUL spread - {run.case.title}\n(dotted: current knowledge; dashed: floor "
                 f"with every crack quality exact; shaded: indicative UT precision)", x=0.01, ha="left", fontsize=9.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_orientation(run: tg.TargetRun, path: Path):
    fig, axes = plt.subplots(1, len(run.base), figsize=(3.4 * len(run.base), 3.4), constrained_layout=True,
                             sharey=True, squeeze=False)
    for ax, r in zip(axes[0], run.base):
        for cand in tg.ORIENTATION:
            s = r.sweeps.get(cand.name)
            if s is None:
                continue
            ax.plot(_x(s.sds, 2.0), 100 * s.reduction() / s.baseline["u90"], "o-", color=ORIENT_COLOURS[cand.name],
                    lw=1.8, ms=4, label=cand.name.replace("orientation ", ""))
        ax.axhline(0.0, color=INK_2, lw=0.6)
        ax.axhline(100 * tg.SUBSTANTIAL, color=INK_2, lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.grid(color=GRID, lw=0.6)
        ax.set_title(f"a0 = {mm(r.a)}", loc="left", fontsize=8.5)
        ax.set_xlabel("orientation sd [deg] (leftmost = exact)")
    axes[0, 0].set_ylabel("U90 reduction vs current knowledge [%]")
    axes[0, 0].legend(fontsize=6.8, loc="lower left")
    fig.suptitle(f"Orientation under different mechanics (dotted: substantial threshold) - {run.case.title}", x=0.01,
                 ha="left", fontsize=9.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_all_inputs(run: tg.TargetRun, path: Path):
    n_groups = len(run.base[0].curves.curves)
    fig, axes = plt.subplots(1, len(run.base), figsize=(3.3 * len(run.base), 0.28 * n_groups + 1.6),
                             constrained_layout=True, squeeze=False)
    for ax, r in zip(axes[0], run.base):
        c = r.curves
        order = c.ranking()[::-1]
        shares = [100 * c.achievable(grp) / c.base["u90"] for grp in order]
        colours = [GROUP_COLOURS.get(CATEGORY.get(grp, "material"), SLOTS[1]) for grp in order]
        ax.barh(range(len(order)), shares, color=colours, height=0.7)
        ax.set_yticks(range(len(order)), [label(grp) for grp in order], fontsize=7)
        for y, v in enumerate(shares):
            ax.text(max(v, 0) + 0.5, y, f"{v:.0f}%", va="center", fontsize=6.5, color=INK_2)
        ax.axvline(0.0, color=INK_2, lw=0.6)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_title(f"a0 = {mm(r.a)} (U90 {c.base['u90']:.2f})", loc="left", fontsize=8.5)
        ax.set_xlabel("U90 reduction if known exactly [%]")
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in (SLOTS[0], SLOTS[2], "#8a8984", SLOTS[1])]
    fig.legend(handles, ["crack quality", "loading", "geometry / SIF", "material"], loc="outside lower center",
               ncol=4, fontsize=7.5)
    fig.suptitle(f"Every input/group: reduction in the 90 % RUL band when known exactly - {run.case.title}", x=0.01,
                 ha="left", fontsize=9.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_map(run: tg.TargetRun, path: Path):
    names = [c.name for c in tg.CANDIDATES if c.name in run.base[0].sweeps]
    m = np.array([[run.base[k].sweeps[n].achievable_share for k in range(len(run.base))] for n in names])
    texts = []
    for n in names:
        row = []
        for r in run.base:
            s = r.sweeps[n]
            row.append(f"{s.achievable_share:.0%}\n{fmt_sd(s.candidate, s.required_sd)}" if s.substantial
                       else f"{s.achievable_share:.0%}\nn.s.")
        texts.append(row)
    fig, ax = plt.subplots(figsize=(1.6 * len(run.base) + 3.4, 0.62 * len(names) + 1.6), constrained_layout=True)
    heatmap(ax, np.clip(m, 0, None), names, [mm(r.a) for r in run.base], SEQUENTIAL, 0.0, max(0.4, np.nanmax(m)),
            texts=texts)
    ax.set_title(f"Measurement-target map: U90 reduction if known exactly, and the precision needed for "
                 f"{tg.REQUIRED_SHARE:.0%} of it\n(n.s. = below {tg.SUBSTANTIAL:.0%}) - {run.case.title}", loc="left",
                 fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scenarios(run: tg.TargetRun, path: Path):
    names = [c.name for c in tg.SCENARIO_CANDIDATES if c.name in run.base[0].sweeps]
    columns = []
    for a in run.case.key_stages:
        columns += [(a, r.scenario, r) for r in run.scenarios if r.a == a]
        columns += [(a, f"catalogue {r.level}", r) for r in run.levels if r.a == a]
    m = np.array([[col[2].sweeps[n].achievable_share if n in col[2].sweeps else np.nan for col in columns]
                  for n in names])
    fig, ax = plt.subplots(figsize=(0.62 * len(columns) + 3.6, 0.6 * len(names) + 2.6), constrained_layout=True)
    heatmap(ax, np.clip(m, 0, None), names, [f"{c[0] / MM:g} mm | {c[1]}" for c in columns], SEQUENTIAL, 0.0,
            max(0.4, np.nanmax(m)), fmt="{:.0%}", xlabel="")
    ax.tick_params(axis="x", rotation=70)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.set_title(f"Robustness: U90 reduction if known exactly, across scenarios and catalogue ends - {run.case.title}",
                 loc="left", fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["round_bar", "plate"])
    ap.add_argument("--n", type=int, default=2000, help="Monte Carlo samples per run")
    ap.add_argument("--sobol-n", type=int, default=512, help="Sobol base sample size for the supporting study")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--out", default="results/targets")
    ap.add_argument("--quick", action="store_true", help="small samples, for checking the pipeline")
    args = ap.parse_args()
    if args.quick:
        args.n, args.sobol_n = 300, 64
    style()
    pool = g.load_or_build_pool("results/giannella")[0]
    builders = {"round_bar": lambda: round_bar_case(pool), "plate": plate_case}
    for name in args.cases:
        case = builders[name]()
        out = Path(args.out) / name
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed)
        t0 = time.perf_counter()
        run = tg.run_targets(case, rng, states=tuple(v * MM for v in STATES[name]), scenario_names=SCENARIOS[name],
                             n=args.n, sobol_n=args.sobol_n)
        tables(run, out)
        print_report(run)
        plot_distributions(run, out / "rul_distributions.png")
        plot_sd_vs_spread(run, out / "sd_vs_spread.png")
        plot_orientation(run, out / "orientation_mechanics.png")
        plot_all_inputs(run, out / "all_inputs_ranking.png")
        plot_map(run, out / "target_map.png")
        plot_scenarios(run, out / "scenario_robustness.png")
        print(f"\n[{name}] done in {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"\nTables and figures written to {args.out}")


if __name__ == "__main__":
    main()
