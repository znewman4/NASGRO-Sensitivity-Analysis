"""A formulation binds geometry, growth model, failure criterion and numerics into one life model."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping

import numpy as np

from .integrate import FailureCriterion, IntegrationSettings, LifeResult, integrate_life

# Physical-input names every formulation reads, alongside its rate model's parameters and "Kc".
STATE_INPUTS = ("a_init", "aspect_init", "S_max", "R")


@dataclass(frozen=True)
class Formulation:
    geometry: object
    rate_model: object
    criterion: FailureCriterion = FailureCriterion()
    settings: IntegrationSettings = IntegrationSettings()
    derive: Callable[[Mapping[str, np.ndarray]], dict[str, np.ndarray]] | None = None   # e.g. S_max from load and diameter

    @property
    def inputs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*STATE_INPUTS, *self.geometry.inputs, *self.rate_model.parameters, "Kc")))

    def _complete(self, x: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        return {**x, **self.derive(x)} if self.derive else x

    def _params(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        optional = [k for k in getattr(self.rate_model, "optional", ()) if k in x]
        return {k: x[k] for k in (*self.rate_model.parameters, "Kc", *optional)}

    def _geometry_params(self, x: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: x[k] for k in self.geometry.inputs}

    def run(self, x: Mapping[str, np.ndarray]) -> LifeResult:
        """Remaining life from the sampled current state under the declared future loading."""
        x = self._complete(x)
        a0 = np.asarray(x["a_init"], dtype=float)
        return integrate_life(self.geometry, self.rate_model, self._params(x),
                              a0, a0 / x["aspect_init"], x["S_max"], x["R"],
                              self.criterion, self.settings, self._geometry_params(x))

    def rates_at(self, x: Mapping[str, np.ndarray], a: float, c: float) -> tuple[np.ndarray, np.ndarray]:
        """da/dN and dc/dN at a fixed (matched) crack geometry for the sampled material and loading.

        This is the growth-rate sensitivity quantity; it is deliberately kept
        separate from life, which depends on the whole trajectory.
        """
        x = self._complete(x)
        s_max, r = np.asarray(x["S_max"], dtype=float), np.asarray(x["R"], dtype=float)
        k_a, k_c = self.geometry.stress_intensity(np.full(s_max.shape, a), np.full(s_max.shape, c), s_max,
                                                  **self._geometry_params(x))
        p = self._params(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.rate_model(k_a * (1 - r), r, a, p), self.rate_model(k_c * (1 - r), r, c, p)

    def with_settings(self, **kwargs) -> Formulation:
        return replace(self, settings=replace(self.settings, **kwargs))
