"""Sparse permanent array read as change, with the Sun et al. (2023) value (new script).

Sun et al. (2023, SHM 22:3451) detected a 0.1 mm change in a side-drilled hole with a
permanently bonded 18-element, 5 MHz array; direct TFM did not size it absolutely.
Here the change reading error is taken as sd 0.10 mm (Sun's smallest detected step),
sharing one unknown absolute offset of sd 1.15 mm (BS 7910 Table T.2 phased array,
as no absolute error is reported). Same ensemble seeds, truths and settings as the
sparse level of run_array_spectrum.py, so only the reading error differs.

Writes results/2026-11-05_bs7910/array_spectrum_sun_<case>.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_array_spectrum as ras  # noqa: E402
from nasgro_scenarios import runner  # noqa: E402

SD_DET, SIG_CHANGE, OFFSET = 1.15, 0.10, 1.15
LI = 1                                 # the sparse level's index in ras.LEVELS (seeds)


def run_case(case):
    runner.set_standard()
    dt = ras.interval(case)
    times_f = dt / ras.FINE * np.arange(1, ras.K * ras.FINE + 1)
    ref_s = ras.bs.sample(case, ras.A0_MM * 1e-3, 100_000, 2.3, seed=11)
    ref = float(ras.life.u90(ras.life.chunked(lambda x: ras.life.ca_life(case, x), ref_s)))
    L, H = ras.ensemble(case, SD_DET, (), times_f, 3000 + 100 * LI, ras.N_ENS)
    Lt_all, Ht_all = ras.ensemble(case, SD_DET, (), times_f, 9000 + 100 * LI, 2 * ras.N_TRUTH)
    truths = np.flatnonzero(np.isfinite(Lt_all))[:ras.N_TRUTH]
    Lt, Ht = Lt_all[truths], Ht_all[truths]
    rng = np.random.default_rng(5)
    idx = np.arange(0, ras.K * ras.FINE, 1)
    rows = []
    for i in range(len(Lt)):
        y = Ht[i, idx] + SIG_CHANGE * 1e-3 * rng.standard_normal(len(idx)) + OFFSET * 1e-3 * rng.standard_normal()
        rows += ras.track(L, H, Lt[i], y, idx, SIG_CHANGE * 1e-3, OFFSET * 1e-3, times_f, dt)
    sm = ras.summarise(rows)
    sm[ras.K]["reduction"] = 1 - sm[ras.K]["u90"] / ref
    print(case, sm[ras.K], flush=True)
    return {"reference_u90_conventional_ut": ref, "sd_change_mm": SIG_CHANGE, "offset_sd_mm": OFFSET,
            "after_ten": sm[ras.K]}


def main():
    for case in sys.argv[1:] or ("bar", "plate"):
        res = {case: run_case(case)}
        with open(os.path.join(ras.OUT, f"array_spectrum_sun_{case}.json"), "w") as f:
            json.dump(res, f, indent=1)
    print("saved")


if __name__ == "__main__":
    main()
