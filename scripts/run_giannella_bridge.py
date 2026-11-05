"""Why this study's bar results differ from Giannella (2022), and the corrected reproduction.

Part A, reproduction: Giannella's own setup in this engine, with the earlier
("original") and the corrected material pool, against the paper's stress
levels (Figs 10-11), its pool correlations (Fig. 8) and, as a hold-out check
not used in the correction, its one-at-a-time spread sweeps (Figs 12-14).

Part B, bridge at 250 MPa and a 1 mm crack: starting from Giannella's setup,
change one assumption at a time to this study's bar (crack-depth sizing
spread, load spread, the extra inputs, the material model). At each step:
sd and U90 of log10 life, and how much each source narrows them when known
exactly. Each change is also applied alone, to show the order does not
decide the answer.

Writes results/2026-10-23_nasgro_only/giannella_bridge.json.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nasgro_scenarios import giannella as g, life, runner  # noqa: E402

N, SEED = 20_000, 1
SOURCES = {"crack depth": ("depth",), "material": ("material",), "diameter D": ("D",),
           "load amplitude": ("F",), "Kc, a0, R, tilt": ("Kc", "a0", "R", "tilt")}


def run(cfg, fixed=(), stress=250.0, pool="corrected", n=N, a_sd=None):
    c = dict(cfg)
    if a_sd is not None:
        c["a_sd"] = a_sd
    s = g.bridge_sample(c, n, seed=SEED, fixed=fixed, stress=stress, pool_name=pool)
    return life.stats(life.chunked(lambda x: life.ca_life("bar", x), s))


def contributions(cfg, pool="corrected"):
    base = run(cfg, pool=pool)
    out = {}
    for name, keys in SOURCES.items():
        if keys[0] in ("Kc",) and not cfg["extras"]:
            continue
        st = run(cfg, fixed=keys, pool=pool)
        out[name] = dict(sigma_reduction=1 - st["sigma"] / base["sigma"], u90_reduction=1 - st["u90"] / base["u90"])
    return base, out


def main():
    runner.set_standard()
    res = {"paper": {"stress": g.PAPER_STRESS, "sweep": g.PAPER_SWEEP, "corr": g.PAPER_CORR}}
    g0 = g.step_configs()[0][1]

    print("Part A: reproduction of Giannella's own setup")
    for pool in ("original", "corrected"):
        m = g.pool(pool)
        corr = g.pool_corr(m)
        stress = {s: run(g0, stress=s, pool=pool) for s in g.PAPER_STRESS}
        # Figs 12-14 hold-out: sd at the Table 4 spread and at the smallest swept spread
        sweep = {"crack depth": (run(g0, pool=pool)["sigma"], run(g0, pool=pool, a_sd=0.005e-3)["sigma"])}
        sweep["diameter D"] = (sweep["crack depth"][0], run(g0, fixed=("D",), pool=pool)["sigma"])
        sweep["load amplitude"] = (sweep["crack depth"][0], run(g0, fixed=("F",), pool=pool)["sigma"])
        best = g.bridge_sample(g0, 1, fixed=("depth", "material", "D", "F"), pool_name=pool)
        best_life = float(np.log10(life.ca_life("bar", best))[0])
        res[pool] = dict(stress=stress, sweep=sweep, corr=corr, best_fit_log10N=best_life,
                         dKth0_pct=np.percentile(m["dKth0"], [5, 50, 95]).tolist(), q=g.POOLS[pool]["q"])
        print(f"\n  pool {pool}: dKth0 5/50/95 % {np.round(res[pool]['dKth0_pct'], 2)}, best-fit log10 N {best_life:.3f}")
        print("  corr  " + "  ".join(f"{k} {v:+.2f} ({g.PAPER_CORR[k]:+.2f})" for k, v in corr.items()))
        for s, st in stress.items():
            p = g.PAPER_STRESS[s]
            print(f"  {s} MPa  mu {st['mu']:.3f} ({p[0]:.3f})  sd {st['sigma']:.3f} ({p[1]:.3f})  Pf {st['pf']:.2f} ({p[2]:.2f})")
        for k, (a, b) in sweep.items():
            pa, pb = g.PAPER_SWEEP[k]
            print(f"  sweep {k:<15} sd {a:.3f} -> {b:.3f} ({pa:.3f} -> {pb:.3f}): "
                  f"{1 - b / a:.0%} ({1 - pb / pa:.0%})")

    print("\nPart B: bridge, 250 MPa, 1 mm crack, corrected pool")
    steps = []
    for label, cfg in g.step_configs():
        base, contrib = contributions(cfg)
        steps.append(dict(label=label, cfg=cfg, stats=base, contributions=contrib))
        print(f"  {label:<45} sd {base['sigma']:.3f}  U90 {base['u90']:.3f}  mu {base['mu']:.3f}  Pf {base['pf']:.2f}  | "
              + ", ".join(f"{k} {v['sigma_reduction']:.0%}/{v['u90_reduction']:.0%}" for k, v in contrib.items()))
    alone = {}
    for label, change in g.STEPS[1:]:
        st = run({**g0, **change})
        alone[label] = st
        print(f"  alone: {label:<38} sd {st['sigma']:.3f}  U90 {st['u90']:.3f}")
    res["bridge"] = steps
    res["alone"] = alone
    print("  study bar check (sd, U90):", steps[-1]["stats"]["sigma"], steps[-1]["stats"]["u90"])
    print(runner.save("giannella_bridge.json", res))


if __name__ == "__main__":
    main()
