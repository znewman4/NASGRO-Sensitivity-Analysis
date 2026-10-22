"""Giannella-style spread sweeps for every input: her Figs 11-14 (life scatter and log-life density
at several input spreads), extended from three inputs to all of them.

One input or fitted group at a time is swept through four spreads while every other input keeps its
industry-typical uncertainty, with common random numbers, as in Giannella (2022):
  * the top level is twice the typical spread, capped at the catalogue's high end so that every level
    stays within the documented industry range;
  * then typical, a third and a thirtieth of typical (near exact), the ratios of her Figs 12-14.
A fitted material pool cannot be widened without new data, so its levels are the pool itself, a third
and a thirtieth (rows shrunk towards the pool centre).

Statistics follow the paper: mean and sd of log10 Nf over cracks that fail, and the failed share Pf.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .cases import Case
from .giannella import log_life_stats
from .integrate import LifeResult
from .uncertainty import CATALOGUE, CRACK, crack_stage

TOP = 2.0
LOWER = (1.0, 1 / 3, 1 / 30)   # typical, a third, a thirtieth


@dataclass
class Level:
    factor: float                         # on the typical spread
    values: np.ndarray                    # the plotted input, SI units, every sample
    cycles: np.ndarray
    failed: np.ndarray
    stats: tuple[float, float, float]     # mean and sd of log10 Nf over failed samples, Pf


@dataclass
class VariableSweep:
    group: str
    members: tuple[str, ...]
    member: str            # member plotted against life: the one whose value tracks life most closely
    levels: list[Level]    # largest spread first

    @property
    def typical(self) -> Level:
        return next(lv for lv in self.levels if lv.factor == 1.0)

    @property
    def benefit(self) -> float:
        """Fall in the sd of log10 Nf from typical to near-exact knowledge."""
        return self.typical.stats[1] - self.levels[-1].stats[1]

    @property
    def active(self) -> bool:
        """False when the spread does not change any life at all (e.g. Cth for R < 0 at R > 0)."""
        return not all(np.array_equal(lv.cycles, self.levels[0].cycles) for lv in self.levels[1:])


def top_factor(members, a: float) -> float:
    """Twice typical, capped at the catalogue's high end of every member."""
    entries = [CRACK[crack_stage(a)].get(m) or CATALOGUE.get(m) for m in members]
    return min([TOP, *(e.high / e.typical for e in entries if e is not None)])


def _tracking_member(members, x, life: LifeResult) -> str:
    if len(members) == 1:
        return members[0]
    logs = np.log10(life.cycles[life.failed])
    rho = [abs(stats.spearmanr(np.asarray(x[m])[life.failed], logs)[0]) for m in members]
    return members[int(np.nanargmax(rho))]


def spread_sweeps(case: Case, a: float, n: int, rng: np.random.Generator) -> list[VariableSweep]:
    """Every uncertain input or group of ``case`` at nominal depth ``a`` (m), swept one at a time.

    All sweeps share one set of underlying normals, so the typical level is the same run everywhere.
    """
    spec = case.industry(a, "typical")
    w = spec.standard_normal(n, rng)
    x = spec.transform(w)
    life = case.formulation.run(x)
    out = []
    for group, cols in spec.uncertain_groups():
        members = tuple(spec.names[c] for c in cols)
        member = _tracking_member(members, x, life)
        factors = LOWER if group in spec.empirical else (top_factor(members, a), *LOWER)
        levels = []
        for f in factors:
            xi = x if f == 1.0 else spec.scaled({group: f}).transform(w)
            li = life if f == 1.0 else case.formulation.run(xi)
            levels.append(Level(f, np.asarray(xi[member], dtype=float), li.cycles, li.failed, log_life_stats(li)))
        out.append(VariableSweep(group, members, member, levels))
    return out
