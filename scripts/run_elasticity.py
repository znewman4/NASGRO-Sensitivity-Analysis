"""Raw sensitivity: change in nominal log10 life for a +10 % change of each input.

Every other input at nominal (no Monte Carlo); NASGRO-equation inputs only.
Writes results/2026-10-23_nasgro_only/elasticity.json.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nasgro_scenarios import cases, life, runner  # noqa: E402

KEYS = {"bar": ["F", "D", "C", "n", "p", "dKth0", "Kc", "a0", "q", "alpha", "smax_sigma0", "cth_neg"],
        "plate": ["S", "t", "W", "ac", "C", "n", "p", "q", "dKth0", "Kc", "a0", "alpha", "smax_sigma0",
                  "cth_pos", "cth_neg"]}
ALL = tuple(set(cases.BAR_INPUTS.values()) | set(cases.PLATE_INPUTS.values()) | {"material", "Cn"})


def nominal_life(case, a_mm, key=None, factor=1.0, depth_factor=1.0):
    s = cases.sample(case, a_mm * 1e-3 * depth_factor, 1, seed=1, fixed=ALL)
    if key == "C":
        s["C"] = s["C"] * factor
    elif key is not None:
        s[key] = s[key] * factor
    return float(life.ca_life(case, s)[0])


def main():
    out = []
    for case in ("bar", "plate"):
        for d in runner.DEPTHS_MM[case]:
            L0 = nominal_life(case, d)
            row = dict(case=case, depth_mm=d, log10_life=float(np.log10(L0)),
                       depth=float(np.log10(nominal_life(case, d, depth_factor=1.1) / L0)))
            for k in KEYS[case]:
                L1 = nominal_life(case, d, k, 1.1)
                row[k] = float(np.log10(L1 / L0)) if np.isfinite(L1) and np.isfinite(L0) else None
            out.append(row)
    os.makedirs(runner.STANDARD_DIR, exist_ok=True)
    runner.RESULTS = runner.STANDARD_DIR
    print(runner.save("elasticity.json", dict(results=out, note="log10(N(+10%)/N)")))
    for r in out:
        print(r["case"], r["depth_mm"], {k: round(v, 3) for k, v in r.items()
                                        if isinstance(v, float) and k not in ("depth_mm", "log10_life")})


if __name__ == "__main__":
    main()
