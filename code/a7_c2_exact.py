"""A7 — C2 in exact integer form.

Exact credit c = n0(eps, delta) - n0(eps, delta'): the largest number of real events that can be
dropped while a zero-failure test still has false-certification probability <= delta' at p = eps.
Tightness: (1-eps)^{n0(delta')} <= delta' < (1-eps)^{n0(delta') - 1}.
The P0 closed form c* = floor(ln(delta'/delta) / (-ln(1-eps))) is kept as a (safe) lower bound.
Adds columns to results/c2_mc.json WITHOUT touching the P0 values; the MC for the exact n uses its
own generator (100,000 runs, seed 42).
"""
from __future__ import annotations

import json

import numpy as np

from common import RESULTS, dump, n0

N_MC = 100_000
SEED = 42
TOL_PP = 0.5


def main():
    d = json.load(open(RESULTS / "c2_mc.json", encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    for r in d["grid"]:
        eps, dl, dp = r["eps"], r["delta"], r["delta_prime"]
        c = n0(eps, dl) - n0(eps, dp)
        n = n0(eps, dp)
        closed = (1 - eps) ** n
        mc = float((rng.binomial(n, eps, size=N_MC) == 0).mean())
        r.update(c_exact=c, n_real_exact=n, closed_form_exact=closed, mc_exact=mc,
                 abs_dev_pp_exact=100 * abs(mc - closed), pass_05pp_exact=100 * abs(mc - closed) <= TOL_PP,
                 exact_le_delta_prime=closed <= dp + 1e-12,
                 exact_is_maximal=bool((1 - eps) ** (n - 1) > dp),
                 c_star_is_lower_bound=r["c_star"] <= c, saving_pct_exact=100 * c / n0(eps, dl))
        assert r["c_integer_exact"] == c
    prim = {(r["eps"], r["delta"], r["delta_prime"]): r for r in d["grid"]}
    d["primary"] = [prim[(p["eps"], p["delta"], p["delta_prime"])] for p in d["primary"]]
    d["meta"]["formula_exact"] = "c = n0(eps,delta) - n0(eps,delta')  (A7, 2026-09-27: main form)"
    d["meta"]["formula"] = ("c* = floor(ln(delta'/delta)/(-ln(1-eps)))  — P0 closed form, kept as a "
                            "LOWER bound on the exact credit")
    d["meta"]["mc_exact"] = f"{N_MC} runs, seed {SEED}, separate generator; P0 'mc' column untouched"
    d["all_pass_exact"] = all(r["pass_05pp_exact"] for r in d["grid"])
    d["max_abs_dev_pp_exact"] = max(r["abs_dev_pp_exact"] for r in d["grid"])
    d["all_exact_maximal"] = all(r["exact_is_maximal"] for r in d["grid"])
    d["all_c_star_lower_bound"] = all(r["c_star_is_lower_bound"] for r in d["grid"])
    dump(d, "c2_mc.json")
    for r in d["primary"]:
        print(f"delta'={r['delta_prime']:.3f} c*={r['c_star']} c_exact={r['c_exact']} n={r['n_real_exact']} "
              f"closed={r['closed_form_exact']:.4f} MC={r['mc_exact']:.4f} dev={r['abs_dev_pp_exact']:.3f}pp "
              f"saving={r['saving_pct_exact']:.0f}%")
    print("all pass", d["all_pass_exact"], "max dev", round(d["max_abs_dev_pp_exact"], 3),
          "maximal", d["all_exact_maximal"], "c* lower bound", d["all_c_star_lower_bound"])


if __name__ == "__main__":
    main()
