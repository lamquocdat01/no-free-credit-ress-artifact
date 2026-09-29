"""A2 — certifiable operating points (eps, d_min, K) for Mode G and Mode G* (CDnet, gap1 events).

Grid: d_min in {1,2,4,8,16,32} (events with duration >= d_min only) x K in K_GRID.
Cells with K <= d_min have zero TEMPORAL miss (an event of length D >= K always contains a sampled
frame — the K-1 lemma), so only detector miss remains: this is where the sim2real gap lives.
Sample size: certify iff the one-sided Clopper-Pearson (1-delta) upper bound on p is <= eps
(failures allowed, not only zero-failure). n_req(eps, p) = smallest n with
P_{Bin(n,p)}(certify) >= 0.80, evaluated at p = p_hat and at the pessimistic p = video-bootstrap
97.5% upper limit of p_hat. Compared with the zero-failure n0 = 59 (whose power at p is (1-p)^59).
Output: results/operating_points.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from common import B_BOOT, K_GRID, RESULTS, SEED, cp_ci, dump, n0

D_MIN = [1, 2, 4, 8, 16, 32]
EPS_GRID = [0.05, 0.10, 0.20]
DELTA = 0.05
POWER = 0.80
N_MAX = 20000


def crit(n, eps, delta=DELTA):
    """Largest x such that the one-sided CP upper bound U(x, n) <= eps (i.e. P_eps(X <= x) <= delta);
    -1 if even x = 0 does not certify."""
    x = int(sps.binom.ppf(delta, n, eps))
    while x >= 0 and sps.binom.cdf(x, n, eps) > delta:
        x -= 1
    while sps.binom.cdf(x + 1, n, eps) <= delta:
        x += 1
    return x


_CRIT = {}


def power(n, p, eps):
    key = (n, eps)
    if key not in _CRIT:
        _CRIT[key] = crit(n, eps)
    c = _CRIT[key]
    return 0.0 if c < 0 else float(sps.binom.cdf(c, n, p))


def n_required(p, eps):
    if p >= eps:
        return None
    lo = n0(eps, DELTA)
    for n in range(lo, N_MAX + 1):
        if power(n, p, eps) >= POWER:
            return n
    return None


def cell_stats(x, vids, rng):
    n = len(x)
    if n == 0:
        return None
    uv = np.unique(vids)
    idx = {v: np.where(vids == v)[0] for v in uv}
    bs = np.array([x[np.concatenate([idx[v] for v in rng.choice(uv, len(uv))])].mean() for _ in range(B_BOOT)])
    return dict(n_events=n, n_videos=len(uv), p_hat=float(x.mean()), cp95=cp_ci(float(x.sum()), n),
                boot95_video=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


def main():
    E = pd.read_csv(RESULTS / "miss_matrix_Gstar.csv")
    E = E[E.merge_gap == 1].reset_index(drop=True)
    variants = {"G": ("pG", E), "Gstar": ("pGs", E),
                "Gstar_Lmax_ge256px (sensitivity)": ("pGs", E[E.L_area_max >= 256])}
    rng = np.random.default_rng(SEED)
    n0s = {eps: n0(eps, DELTA) for eps in EPS_GRID}
    out = dict(meta=dict(source="miss_matrix_Gstar.csv (CDnet, gap1 events)", d_min=D_MIN, K=K_GRID,
                         eps=EPS_GRID, delta=DELTA, power=POWER,
                         test="certify iff one-sided CP (1-delta) upper bound <= eps (failures allowed)",
                         n0_zero_failure=n0s,
                         k_le_dmin="temporal miss = 0 (K-1 lemma); only detector miss remains",
                         bootstrap=f"B={B_BOOT} seed={SEED}, by video"),
               tables={})
    for name, (col, D) in variants.items():
        tab = {}
        for d in D_MIN:
            sub = D[D.duration >= d]
            row = {}
            for K in K_GRID:
                st = cell_stats(sub[f"{col}_K{K}"].values, sub.video.values, rng)
                st["K_le_dmin"] = K <= d
                st["n_required"] = {}
                for eps in EPS_GRID:
                    ph, pu = st["p_hat"], st["boot95_video"][1]
                    st["n_required"][f"eps{eps}"] = dict(
                        at_p_hat=n_required(ph, eps), at_p_upper=n_required(pu, eps),
                        power_of_zero_failure_n0=float((1 - ph) ** n0s[eps]),
                        power_of_CP_test_at_n0=power(n0s[eps], ph, eps))
                row[f"K{K}"] = st
            tab[f"d{d}"] = dict(n_events=len(sub), frac_events_kept=len(sub) / len(D), cells=row)
        out["tables"][name] = tab
        print(f"\n== {name} ==  p_hat (n_req at eps=0.10 / 0.20)")
        for d in D_MIN:
            cells = tab[f"d{d}"]["cells"]
            print(f"d{d:>2} n={tab[f'd{d}']['n_events']:3d} " + " ".join(
                f"K{K}:{cells[f'K{K}']['p_hat']*100:4.1f}{'*' if K <= d else ' '}"
                f"({cells[f'K{K}']['n_required']['eps0.1']['at_p_hat']},{cells[f'K{K}']['n_required']['eps0.2']['at_p_hat']})"
                for K in K_GRID))
    dump(out, "operating_points.json")


if __name__ == "__main__":
    main()
