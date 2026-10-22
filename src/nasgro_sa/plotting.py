"""Shared figure styling (reference palette from the dataviz guidance) and the annotated heatmap.

Scripts select the Agg backend before importing this module.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .cases import CATEGORY, CATEGORY_ORDER, LABELS

SURFACE, INK, INK_2, GRID, EMPTY, CRITICAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#e9e8e4", "#bdbbb5"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#b3261e", "#e34948", "#f0efec", "#3987e5", "#184f95"])


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "font.size": 8.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def ordered(names):
    """Rows grouped by category, in LABELS order within a category."""
    canon = list(LABELS)
    return sorted(names, key=lambda k: (CATEGORY_ORDER.index(CATEGORY.get(k, "closure")),
                                        canon.index(k) if k in canon else len(canon)))


def label(name: str) -> str:
    return LABELS.get(name, name)


def heatmap(ax, m, rows, cols, cmap, vmin, vmax, fmt="{:.2f}", flags=None, xlabel="Initial crack depth a0 [mm]",
            texts=None):
    """Annotated heatmap; nan cells are shown as 'n/e' (not estimable); * marks flagged cells.

    ``texts`` (same shape, None entries ignored) replaces the annotation of individual cells.
    """
    m = np.asarray(m, dtype=float)
    shown = np.ma.masked_invalid(np.clip(m, vmin, vmax))
    cm = cmap.copy()
    cm.set_bad(EMPTY)
    ax.imshow(shown, cmap=cm, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            custom = texts[i][j] if texts is not None else None
            if not np.isfinite(v):
                ax.text(j, i, custom or "n/e", ha="center", va="center", fontsize=6.5, color=INK_2)
                continue
            if custom is not None:
                ax.text(j, i, custom, ha="center", va="center", fontsize=6.5, color=INK, style="italic")
                continue
            level = (np.clip(v, vmin, vmax) - vmin) / (vmax - vmin)
            dark = level > 0.62 if vmin >= 0 else abs(level - 0.5) > 0.36
            text = fmt.format(v) + ("*" if flags is not None and flags[i, j] else "")
            ax.text(j, i, text, ha="center", va="center", fontsize=6.5, color="white" if dark else INK)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), [label(r) for r in rows])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
