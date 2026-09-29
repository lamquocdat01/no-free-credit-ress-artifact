"""T3 — Monte Carlo check of C2 (borrowing price) and closed-form C3 numbers.

C2: allowing false-certification probability delta' > delta when the simulator is wrong buys at
most c* = floor( ln(delta'/delta) / (-ln(1-eps)) ) real events. At the worst-case truth
p_true = eps, a zero-failure test on n = n0 - c real events certifies falsely with probability
(1-eps)^(n0-c). The MC below (100,000 runs, seed 42) must match that closed form within 0.5 pp.

C3: p <= q + D, D = P(real miss AND twin catch). With q <= q_U known from abundant twin runs,
certifying D <= eps - q_U at level delta/2 by a zero-discordance success run needs
n = ceil( ln(delta/2) / ln(1 - (eps - q_U)) ) paired real events.
"""
from __future__ import annotations

import numpy as np

from common import dump, n0

N_MC = 100_000
SEED = 42
TOL_PP = 0.5


def c_star(eps, delta, delta_p):
    return int(np.floor(np.log(delta_p / delta) / (-np.log(1 - eps)) + 1e-12))


def c_exact(eps, delta, delta_p):
    """Integer-exact credit: n0(eps,delta) - n0(eps,delta')."""
    return n0(eps, delta) - n0(eps, delta_p)


def mc_false_cert(rng, eps, n):
    # each run: n i.i.d. real events with miss prob eps; certify iff zero misses
    misses = rng.binomial(n, eps, size=N_MC)
    return float((misses == 0).mean())


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    # primary: eps = delta = 0.05, delta' in {5, 7.5, 10, 15, 20}%
    grid = [(0.05, 0.05, dp) for dp in (0.05, 0.075, 0.10, 0.15, 0.20)]
    # secondary: eps in {.01,.02,.05,.10}, delta in {.01,.05}, delta'/delta in {1,1.5,2,3,4}
    for eps in (0.01, 0.02, 0.05, 0.10):
        for delta in (0.01, 0.05):
            for r in (1.0, 1.5, 2.0, 3.0, 4.0):
                cand = (eps, delta, round(delta * r, 6))
                if cand not in grid:
                    grid.append(cand)
    for i, (eps, delta, dp) in enumerate(grid):
        N0 = n0(eps, delta)
        c = c_star(eps, delta, dp)
        n = N0 - c
        closed = (1 - eps) ** n
        mc = mc_false_cert(rng, eps, n)
        se = np.sqrt(closed * (1 - closed) / N_MC)
        rows.append(dict(primary=i < 5, eps=eps, delta=delta, delta_prime=dp, n0=N0, c_star=c,
                         c_integer_exact=c_exact(eps, delta, dp), n_real=n,
                         closed_form=closed, mc=mc, mc_se=se,
                         abs_dev_pp=100 * abs(mc - closed), pass_05pp=100 * abs(mc - closed) <= TOL_PP,
                         closed_le_delta_prime=closed <= dp + 1e-12,
                         saving_pct=100 * c / N0))
    all_pass = all(r["pass_05pp"] for r in rows)
    prim = [r for r in rows if r["primary"]]
    print(f"{'eps':>5} {'delta':>5} {'delta_p':>7} {'n0':>4} {'c*':>3} {'c_int':>5} {'n':>4} "
          f"{'closed':>8} {'MC':>8} {'dev_pp':>6}")
    for r in rows:
        print(f"{r['eps']:5.2f} {r['delta']:5.2f} {r['delta_prime']:7.3f} {r['n0']:4d} {r['c_star']:3d} "
              f"{r['c_integer_exact']:5d} {r['n_real']:4d} {r['closed_form']:8.5f} {r['mc']:8.5f} "
              f"{r['abs_dev_pp']:6.3f}")
    dump(dict(meta=dict(n_mc=N_MC, seed=SEED, tol_pp=TOL_PP, p_true="eps (worst case at boundary)",
                        formula="c* = floor(ln(delta'/delta)/(-ln(1-eps)))",
                        note_c_integer_exact="n0(eps,delta)-n0(eps,delta'); can exceed c* by 1 because "
                                             "n0 is a ceiling (slack in n0)"),
              primary=prim, grid=rows, all_pass=all_pass,
              max_abs_dev_pp=max(r["abs_dev_pp"] for r in rows)), "c2_mc.json")

    # ── C3 numbers ──
    eps, delta = 0.05, 0.05
    c3 = []
    expected = {0.0: 72, 0.01: 91, 0.02: 122}
    for qU in (0.0, 0.01, 0.02):
        n = int(np.ceil(np.log(delta / 2) / np.log(1 - (eps - qU))))
        c3.append(dict(q_U=qU, target_D=eps - qU, level=delta / 2, n_real=n, expected=expected[qU],
                       match=n == expected[qU], vs_n0=n - n0(eps, delta),
                       false_cert_at_boundary=(1 - (eps - qU)) ** n))
    for r in c3:
        print(r)
    dump(dict(meta=dict(eps=eps, delta=delta, n0=n0(eps, delta),
                        rule="n = ceil(ln(delta/2)/ln(1-(eps-q_U))); p <= q + D, q <= q_U, delta split in half"),
              rows=c3, all_match=all(r["match"] for r in c3)), "c3_numbers.json")


if __name__ == "__main__":
    main()
