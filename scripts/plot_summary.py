"""Summary figures for the short report, drawn from existing result tables (no new simulation).

  regimes.png          where the remaining life is spent, by initial crack depth  (from results/life_stages)
  ranking_by_size.png  U90 reduction when each input is known exactly, by crack depth  (from results/targets)

Run scripts/run_life_stages.py and scripts/run_targets.py first.

Usage:
    python scripts/plot_summary.py [--results results] [--out results/summary]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from nasgro_sa.plotting import GRID, INK_2, SEQUENTIAL, SLOTS, SURFACE, heatmap, label, style
from nasgro_sa.uncertainty import crack_stage

MM = 1e-3
CASES = {"round_bar": "Round bar, 25CrMo4, R = -1, 250 MPa", "plate": "Plate surface crack, R = 0.1, 200 MPa"}
REGIMES = (("share_threshold", "near threshold", SLOTS[0]), ("share_paris", "steady (Paris) growth", SLOTS[1]),
           ("share_critical", "near critical", SLOTS[2]))
NAMES = {"a_init": "crack depth", "growth fit": "material fit (C, m, p, ΔKth0)", "F": "load amplitude",
         "S_max": "stress amplitude", "closure_factor": "crack opening factor κ", "aspect_init": "crack shape a/c",
         "alpha": "constraint factor α", "C-n": "growth curve C-n", "q": "critical-shape exponent q",
         "p": "threshold-shape exponent p", "K_factor": "SIF solution error", "D": "diameter D", "Kc": "toughness Kc"}
MIN_SHARE = 0.02   # inputs below this U90 reduction at every crack depth are left out of the ranking


def read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def detection_split(depths_mm) -> float:
    """x position between the last undetected and the first sized crack depth."""
    return sum(crack_stage(d * MM) == "undetected" for d in depths_mm) - 0.5


def plot_regimes(results: Path, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.78, wspace=0.06)
    for ax, (case, title) in zip(axes, CASES.items()):
        rows = read(results / "life_stages" / case / "stages.csv")
        depths = [float(r["a0_mm"]) for r in rows]
        x = np.arange(len(rows))
        bottom = np.zeros(len(rows))
        for key, name, color in REGIMES:
            share = np.nan_to_num(np.array([float(r[key]) for r in rows]))
            ax.bar(x, share, 0.72, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=1, label=name)
            bottom += share
        for i, r in enumerate(rows):
            ax.text(i, 1.03, f"{float(r['p_arrest']):.0%}", ha="center", fontsize=7.5, color=INK_2)
            if not np.isfinite(float(r["share_paris"])):
                ax.text(i, 0.5, "too few\ncracks\ngrow", ha="center", va="center", fontsize=7, color=INK_2)
        split = detection_split(depths)
        ax.axvline(split, color=INK_2, lw=0.8, ls="--")
        ax.text(split - 0.12, 1.12, "undetected (EICS)", ha="right", fontsize=7.5, color=INK_2)
        ax.text(split + 0.12, 1.12, "detected, UT-sized", ha="left", fontsize=7.5, color=INK_2)
        ax.set_xticks(x, [f"{d:g}" for d in depths])
        ax.set_xlabel("Initial crack depth [mm]")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=9, pad=34)
    axes[0].set_ylabel("Share of remaining life\n(cracks that fail)")
    axes[0].text(-0.5, 1.03, "never grow:", ha="right", fontsize=7.5, color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Where the remaining life is spent, and the share of cracks that never grow", x=0.01, ha="left",
                 fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ranking(results: Path, path: Path):
    tables = {case: read(results / "targets" / case / "all_inputs_reduction.csv") for case in CASES}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [4, 5]})
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.12, top=0.78, wspace=0.58)
    for ax, (case, title) in zip(axes, CASES.items()):
        share = {(r["group"], float(r["a0_mm"])): float(r["share_of_baseline"]) for r in tables[case]}
        depths = sorted({a for _, a in share})
        groups = [g for g in dict.fromkeys(g for g, _ in share) if max(share[g, a] for a in depths) >= MIN_SHARE]
        groups.sort(key=lambda g: -np.mean([max(share[g, a], 0.0) for a in depths]))
        m = np.array([[share[g, a] for a in depths] for g in groups])
        m = np.where(np.abs(m) < 0.005, 0.0, m)   # avoid "-0 %"
        heatmap(ax, m, [NAMES.get(g, label(g)) for g in groups], [f"{a:g}" for a in depths], SEQUENTIAL, 0.0, 0.3,
                fmt="{:.0%}", xlabel="Initial crack depth [mm]")
        split = detection_split(depths)
        ax.axvline(split, color=SURFACE, lw=3)
        ax.text(split - 0.08, -0.75, "undetected", ha="right", fontsize=7.5, color=INK_2)
        ax.text(split + 0.08, -0.75, "detected", ha="left", fontsize=7.5, color=INK_2)
        ax.set_title(title, loc="left", fontsize=9, pad=20)
    fig.suptitle("How much the 90 % RUL band (U90) narrows if one input is known exactly\n"
                 "(every other input keeps its industry-typical uncertainty; inputs below 2 % everywhere omitted)",
                 x=0.01, ha="left", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/summary")
    args = ap.parse_args()
    style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_regimes(Path(args.results), out / "regimes.png")
    plot_ranking(Path(args.results), out / "ranking_by_size.png")
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
