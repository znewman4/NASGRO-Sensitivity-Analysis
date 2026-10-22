"""Probabilistic input specification.

Each input has a marginal distribution defined by a nominal value and a
spread. Dependent inputs are declared as a group and coupled through a
Gaussian copula; inputs in different groups are independent, which is what
keeps grouped Sobol indices well defined. Spreads can be rescaled while the
declared statistic (mean or median) stays fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np
from scipy import stats

KINDS = ("fixed", "normal", "lognormal", "uniform")


@dataclass(frozen=True)
class Marginal:
    """One input's marginal distribution.

    ``nominal`` is the statistic named by ``hold`` (mean or median) and is
    preserved when the spread is rescaled. The spread is ``sd`` if given,
    otherwise ``cov * |nominal|``; both describe the untruncated parent, so
    ``bounds`` truncation shifts the realised moments slightly. ``basis``
    records whether the spread is supported by data or is an exploratory range.
    """

    kind: str
    nominal: float
    cov: float | None = None
    sd: float | None = None
    hold: str = "mean"
    bounds: tuple[float, float] | None = None
    basis: str = "exploratory"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}")
        if self.hold not in ("mean", "median"):
            raise ValueError("hold must be 'mean' or 'median'")
        if self.basis not in ("data", "exploratory"):
            raise ValueError("basis must be 'data' or 'exploratory'")
        if self.kind == "lognormal" and self.nominal <= 0:
            raise ValueError("lognormal nominal must be positive")

    @property
    def spread(self) -> float:
        if self.kind == "fixed":
            return 0.0
        if self.sd is not None:
            return float(self.sd)
        return float(self.cov or 0.0) * abs(self.nominal)

    def scaled(self, factor: float) -> Marginal:
        if self.sd is not None:
            return replace(self, sd=self.sd * factor)
        if self.cov is not None:
            return replace(self, cov=self.cov * factor)
        return self

    def _parent(self):
        sd = self.spread
        if self.kind == "normal":
            return stats.norm(self.nominal, sd)
        if self.kind == "uniform":
            half = np.sqrt(3.0) * sd
            return stats.uniform(self.nominal - half, 2.0 * half)
        sigma = np.sqrt(np.log1p((sd / self.nominal) ** 2))
        mu = np.log(self.nominal) - (0.5 * sigma**2 if self.hold == "mean" else 0.0)
        return stats.lognorm(s=sigma, scale=np.exp(mu))

    def from_standard_normal(self, z) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if self.spread == 0.0:
            return np.full(z.shape, float(self.nominal))
        parent = self._parent()
        if self.bounds is None and self.kind == "normal":
            return self.nominal + self.spread * z
        if self.bounds is None and self.kind == "lognormal":
            return parent.kwds["scale"] * np.exp(parent.kwds["s"] * z)
        u = stats.norm.cdf(z)
        if self.bounds is not None:
            lo, hi = parent.cdf(self.bounds)
            u = lo + u * (hi - lo)
        return parent.ppf(u)


@dataclass(frozen=True)
class EmpiricalGroup:
    """Jointly distributed inputs given as an equally weighted sample, e.g. accepted parameter sets.

    Each draw picks a whole row, so the joint structure is kept exactly. With
    ``factor`` < 1 every row is shrunk towards ``reference`` (on a log scale for
    names in ``log``); factor 0 collapses the group onto the reference.
    """

    samples: Mapping[str, np.ndarray]
    reference: Mapping[str, float]
    log: tuple[str, ...] = ()
    factor: float = 1.0

    @property
    def size(self) -> int:
        return len(next(iter(self.samples.values())))

    def values(self, u: np.ndarray) -> dict[str, np.ndarray]:
        rows = np.minimum((np.asarray(u) * self.size).astype(int), self.size - 1)
        out = {}
        for name, s in self.samples.items():
            v, ref = np.asarray(s, dtype=float)[rows], float(self.reference[name])
            if name in self.log:
                out[name] = ref * (v / ref) ** self.factor
            else:
                out[name] = ref + self.factor * (v - ref)
        return out


@dataclass(frozen=True)
class InputSpec:
    """Joint input distribution: marginals, dependent groups and their normal-score correlations.

    ``empirical`` groups are drawn row-wise from a sample instead; their members
    still need an entry in ``marginals`` (a fixed placeholder) to own a column.
    """

    marginals: Mapping[str, Marginal]
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    correlation: Mapping[tuple[str, str], float] = field(default_factory=dict)
    empirical: Mapping[str, EmpiricalGroup] = field(default_factory=dict)

    def __post_init__(self):
        member_of = {}
        for label, group in self.empirical.items():
            for m in group.samples:
                if m not in self.marginals or m in member_of:
                    raise ValueError(f"empirical member {m!r} needs its own placeholder marginal")
                member_of[m] = label
        for label, members in self.groups.items():
            if label in self.marginals:
                raise ValueError(f"group label {label!r} clashes with an input name")
            for m in members:
                if m not in self.marginals:
                    raise ValueError(f"group {label!r} names unknown input {m!r}")
                if m in member_of:
                    raise ValueError(f"input {m!r} is in more than one group")
                member_of[m] = label
        for (x, y) in self.correlation:
            if x not in member_of or member_of.get(x) != member_of.get(y):
                raise ValueError(f"correlated inputs {x!r}, {y!r} must share a group")
        for label, members in self.groups.items():
            np.linalg.cholesky(self._group_correlation(members))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.marginals)

    def all_groups(self) -> list[tuple[str, tuple[str, ...]]]:
        """Declared groups followed by singleton groups for every ungrouped input."""
        empirical = [(label, tuple(g.samples)) for label, g in self.empirical.items()]
        grouped = {m for _, members in (*self.groups.items(), *empirical) for m in members}
        return list(self.groups.items()) + empirical + [(n, (n,)) for n in self.names if n not in grouped]

    def uncertain_groups(self) -> list[tuple[str, np.ndarray]]:
        """(label, column indices) for every group with at least one non-degenerate member."""
        col = {n: i for i, n in enumerate(self.names)}

        def uncertain(label, members):
            if label in self.empirical:
                return self.empirical[label].factor > 0 and self.empirical[label].size > 1
            return any(self.marginals[m].spread > 0 for m in members)

        return [(label, np.array([col[m] for m in members]))
                for label, members in self.all_groups() if uncertain(label, members)]

    def _group_correlation(self, members) -> np.ndarray:
        corr = np.eye(len(members))
        for i, x in enumerate(members):
            for j, y in enumerate(members):
                rho = self.correlation.get((x, y), self.correlation.get((y, x)))
                if rho is not None and i != j:
                    corr[i, j] = rho
        return corr

    def standard_normal(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Independent standard normals, one column per input; transform() maps them to inputs."""
        return rng.standard_normal((n, len(self.names)))

    def transform(self, w: np.ndarray) -> dict[str, np.ndarray]:
        col = {n: i for i, n in enumerate(self.names)}
        z = np.array(w, dtype=float, copy=True)
        for members in self.groups.values():
            cols = [col[m] for m in members]
            z[:, cols] = w[:, cols] @ np.linalg.cholesky(self._group_correlation(members)).T
        out = {n: self.marginals[n].from_standard_normal(z[:, col[n]]) for n in self.names}
        for group in self.empirical.values():
            out.update(group.values(stats.norm.cdf(w[:, col[next(iter(group.samples))]])))
        return out

    def sample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        return self.transform(self.standard_normal(n, rng))

    def scaled(self, factors: Mapping[str, float]) -> InputSpec:
        """Rescale spreads; keys are ungrouped input names or group labels.

        A declared group is one jointly fitted uncertainty (e.g. C and n), so
        its members can only be rescaled together through the group label.
        Rescaling one member alone while keeping the copula correlation would
        describe a different joint distribution that was never fitted.
        """
        grouped = {m: label for label, members in self.groups.items() for m in members}
        grouped.update({m: label for label, g in self.empirical.items() for m in g.samples})
        for key in factors:
            if key in grouped:
                raise ValueError(f"{key!r} belongs to group {grouped[key]!r}; rescale the group as a whole")
        out, empirical = dict(self.marginals), dict(self.empirical)
        for key, factor in factors.items():
            if key in empirical:
                empirical[key] = replace(empirical[key], factor=empirical[key].factor * factor)
                continue
            for m in self.groups.get(key, (key,)):
                out[m] = out[m].scaled(factor)
        return replace(self, marginals=out, empirical=empirical)

    def with_nominal(self, **values: float) -> InputSpec:
        out = dict(self.marginals)
        for name, value in values.items():
            out[name] = replace(out[name], nominal=value)
        return replace(self, marginals=out)
