"""Shared plumbing for the scenario scripts: one (case, depth) ranking task."""
import json
import os
from multiprocessing import Pool

import numpy as np

from . import cases, life

DEPTHS_MM = {"bar": [0.5, 1.0, 2.0, 4.0], "plate": [0.2, 0.5, 1.0, 2.0, 4.0]}
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def base_inputs(case):
    return dict(cases.BAR_INPUTS if case == "bar" else cases.PLATE_INPUTS)


def rank_task(case, depth_mm, life_of_samples, extra_inputs=None, n=10000, seed=1,
              with_baseline=True):
    """U90 ranking for one case and depth.

    life_of_samples(fixed) -> lives, built on cases.sample(..., fixed=fixed).
    Returns stats, reductions, and (optionally) the plain-baseline stats on the
    same random numbers.
    """
    inputs = base_inputs(case)
    inputs.update(extra_inputs or {})
    st, red = life.u90_reduction(life_of_samples, inputs)
    out = dict(case=case, depth_mm=depth_mm, n=n, stats=st, reduction=red)
    if with_baseline:
        s = cases.sample(case, depth_mm * 1e-3, n, seed=seed)
        out["baseline_stats"] = life.stats(life.chunked(lambda x: life.ca_life(case, x), s))
    return out


def run_pool(fn, tasks, processes=None):
    with Pool(processes or os.cpu_count()) as p:
        return p.starmap(fn, tasks)


def save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda x: float(x) if np.isscalar(x) else str(x))
    return path


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)
