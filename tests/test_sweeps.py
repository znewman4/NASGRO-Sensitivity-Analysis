import numpy as np
import pytest

from nasgro_sa import sweeps as sw
from nasgro_sa.cases import plate_case

MM = 1e-3


def test_top_level_stays_within_the_catalogue():
    assert sw.top_factor(("S_max",), 1 * MM) == 2.0                  # high end (cov 0.5) is beyond twice typical
    assert sw.top_factor(("alpha",), 1 * MM) == pytest.approx(1.5)   # capped at the high end, 0.3 / 0.2
    assert sw.top_factor(("a_init",), 1 * MM) == 2.0
    assert sw.top_factor(("curve_level", "n"), 1 * MM) == 2.0


def test_spread_sweeps_share_the_typical_run():
    sweeps = sw.spread_sweeps(plate_case(), 1 * MM, 200, np.random.default_rng(3))
    reference = sweeps[0].typical.stats
    for s in sweeps:
        assert [lv.factor for lv in s.levels][1:] == pytest.approx([1.0, 1 / 3, 1 / 30])
        assert s.levels[0].factor <= 2.0
        assert s.typical.stats == reference                          # common random numbers
    depth = next(s for s in sweeps if s.group == "a_init")
    assert depth.benefit > 0                                         # knowing depth narrows the life scatter
    assert np.ptp(depth.levels[-1].values) < np.ptp(depth.levels[0].values)
    assert next(s for s in sweeps if s.group == "C-n").member in ("curve_level", "n")
