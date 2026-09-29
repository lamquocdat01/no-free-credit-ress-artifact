"""A5 — feasible forms of C4 with the real event counts (theory in notes/theory_C4_variants.md).

Setting: calibration scenes s = 1..m, n_s paired (real, twin) events each; discordance
Z_si = 1{real miss, twin catch} ~ Bern(D_s) independent given the scenes. X_s = sum_i Z_si,
X = sum_s X_s, N = sum_s n_s. U_CP(x, n) = one-sided (1-delta) Clopper-Pearson upper bound.

C4a-fleet  (event-weighted, CONDITIONAL on the calibration scenes): U_CP(X, N) >= Dbar_w =
           sum n_s D_s / N with prob >= 1-delta for ANY heterogeneous D_s (Hoeffding 1956 Thm 4 +
           binomial mean-median lemma). Guarantee is about the events of THESE cameras.
C4a-pop    (camera-averaged, for a NEW camera drawn from the scene population): J = #{s: X_s >= 1};
           U_CP(J, m) >= mu = E[D_S] with prob >= 1-delta (needs only n_s >= 1).
C4b        (per camera, quantile): for a threshold t, pi_t = P(D_S > t) <= U_CP(J, m) / h_t,
           h_t = 1 - (1-t)^{n_min}. New camera: p_new <= q_new + t except on a set of cameras of
           mass <= gamma_hat = U_CP(J, m)/h_t.
Checks: (1) mean-median lemma numerically; (2) C4a-fleet validity under heterogeneous D_s — exact
Poisson-binomial tail (DP) + MC 100,000 runs, seed 42; (3) power simulation with D_s ~ Beta(mean mu,
CV tau), mu in {0, .5, 1, 2}%, tau in {0, .5, 1}, using the real n_s; (4) (m, n) needed for C4b.
Output: results/c4_power.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats as sps

from common import RESULTS, SEED, dump, n0

DELTA = 0.05
N_MC = 100_000
N_POWER = 20_000
MUS = [0.0, 0.005, 0.01, 0.02]
TAUS = [0.0, 0.5, 1.0]
TARGETS = [0.01, 0.025, 0.05, 0.10]     # bound on D wanted: eps/2 for eps = 2, 5, 10, 20 %
GAMMAS = [0.05, 0.10]


def ucp(x, n, delta=DELTA):
    x = np.asarray(x, float)
    n = np.asarray(n, float)
    u = sps.beta.ppf(1 - delta, x + 1, np.maximum(n - x, 1e-12))
    return np.where(x >= n, 1.0, u)


def pb_cdf(ps, x):
    """Exact Poisson-binomial CDF P(S <= x) by DP."""
    dp = np.zeros(len(ps) + 1)
    dp[0] = 1.0
    for p in ps:
        dp[1:] = dp[1:] * (1 - p) + dp[:-1] * p
        dp[0] *= (1 - p)
    return float(dp[: x + 1].sum())


def xstar(N, p, delta=DELTA):
    """Largest x with U_CP(x, N) < p, i.e. F_{N,p}(x) < delta; -1 if none."""
    x = int(sps.binom.ppf(delta, N, p))
    while x >= 0 and sps.binom.cdf(x, N, p) >= delta:
        x -= 1
    while sps.binom.cdf(x + 1, N, p) < delta:
        x += 1
    return x


def beta_draw(rng, mu, tau, size):
    if mu == 0:
        return np.zeros(size)
    if tau == 0:
        return np.full(size, mu)
    var = (tau * mu) ** 2
    c = mu * (1 - mu) / var - 1
    return rng.beta(mu * c, (1 - mu) * c, size=size)


def real_ns():
    """n_s of the 44 CDnet calibration scenes (p1_background GT-empty list) under Mode G* events,
    gap1 (primary) and gap0 (P0 reference, 383); + the 20 LASIESTA scenes with a background run."""
    bg = json.load(open(RESULTS / "p1_background17.json", encoding="utf-8"))  # corpus17 (raw data only)
    cal = [r["video"] for r in bg["scenes"] if r["dataset"] == "CDnet2014"]
    E = pd.read_csv(RESULTS / "miss_matrix_Gstar.csv")
    ns = {}
    for mg in (0, 1):
        c = E[(E.merge_gap == mg) & E.video.isin(cal)].groupby("video").size()
        ns[f"cdnet44_gap{mg}"] = c.reindex(cal).fillna(0).astype(int).values
    las_bg = [r["video"] for r in bg["scenes"] if r["dataset"] == "LASIESTA"]
    X = pd.read_csv(RESULTS / "miss_matrix_Gstar_ext.csv")
    lc = X[(X.dataset == "LASIESTA") & (X.level == "frame") & X.video.isin(las_bg)].groupby("video").size()
    ns["cdnet44_gap1_plus_lasiesta"] = np.r_[ns["cdnet44_gap1"], lc.reindex(las_bg).fillna(0).astype(int).values]
    return {k: v[v > 0] for k, v in ns.items()}, cal, las_bg


def lemma_check():
    """min over p in (0, 1-1/N) of F_{N,p}(floor(Np)) (claim: > 1/4, Greenberg & Mohri 2014 by symmetry)."""
    worst = (1.0, None, None)
    for N in list(range(2, 201)) + list(range(210, 2001, 10)):   # N = 1: the range (0, 1-1/N) is empty
        p = np.linspace(1e-6, 1 - 1 / N - 1e-9, 4000)
        F = sps.binom.cdf(np.floor(N * p), N, p)
        i = int(np.argmin(F))
        if F[i] < worst[0]:
            worst = (float(F[i]), N, float(p[i]))
    return dict(min_F_floor_Np=worst[0], at_N=worst[1], at_p=worst[2], above_quarter=worst[0] > 0.25,
                grid="N = 2..200 and 210..2000 step 10; 4000 p-points in (0, 1-1/N)",
                implies="for delta <= 1/4: F_{N,p}(x) < delta  =>  x <= Np - 1 (Hoeffding Thm 4 applies)")


def validity_c4a(rng, ns):
    """P(U_CP(X, N) < Dbar_w) under heterogeneous D_s: exact PB tail + MC."""
    N = int(ns.sum())
    m = len(ns)
    pats = {}
    pats["homogeneous_1pct"] = np.full(m, 0.01)
    r = np.random.default_rng(SEED)
    pats["beta_cv1_mean1pct"] = beta_draw(r, 0.01, 1.0, m)
    pats["beta_cv1_mean3pct"] = beta_draw(r, 0.03, 1.0, m)
    one = np.zeros(m)
    one[np.argmax(ns)] = 0.2
    pats["one_scene_20pct_rest_0"] = one
    half = np.where(np.arange(m) % 2 == 0, 0.04, 0.0)
    pats["alternating_4pct_0"] = half
    two = np.where(np.arange(m) < m // 4, 0.08, 0.002)
    pats["quarter_8pct_rest_0.2pct"] = two
    out = {}
    for name, D in pats.items():
        ps = np.repeat(D, ns)
        dbar = float(ps.mean())
        xs = xstar(N, dbar)
        exact_pb = pb_cdf(ps, xs) if xs >= 0 else 0.0
        exact_bin = float(sps.binom.cdf(xs, N, dbar)) if xs >= 0 else 0.0
        X = rng.binomial(ns[None, :], D[None, :], size=(N_MC, m)).sum(1)
        fail = float((ucp(X, N) < dbar).mean())
        out[name] = dict(Dbar_w=dbar, N=N, x_star=xs, x_star_le_Np_minus_1=bool(xs <= N * dbar - 1),
                         exact_fail_poisson_binomial=exact_pb, exact_fail_binomial=exact_bin,
                         mc_fail=fail, mc_se=float(np.sqrt(max(fail, 1e-12) * (1 - fail) / N_MC)),
                         valid=bool(exact_pb <= DELTA and fail <= DELTA + 3 * np.sqrt(DELTA * (1 - DELTA) / N_MC)))
    return out


def c4b_requirements():
    """(m scenes, n events per scene) such that J = 0 gives gamma_hat = U_CP(0, m) / (1-(1-t)^n) <= gamma.
    m must exceed m_min = smallest m with U_CP(0, m) <= gamma (n -> infinity); total events m*n
    decreases towards the floor ~ ln(1/delta) / (gamma * t) as m grows."""
    out = {}
    for t in TARGETS:
        for g in GAMMAS:
            m_min = next(m for m in range(1, 10000) if 1 - DELTA ** (1 / m) <= g)
            rows = []
            for m in sorted({m_min, m_min + 1, int(1.25 * m_min), int(1.5 * m_min), 2 * m_min, 3 * m_min, 4 * m_min}):
                u = 1 - DELTA ** (1 / m)
                h = u / g
                n = int(np.ceil(np.log(1 - h) / np.log(1 - t))) if h < 1 else None
                rows.append(dict(m=m, n_per_scene=n, total_events=None if n is None else m * n))
            out[f"t{t}_gamma{g}"] = dict(t=t, gamma=g, m_min_scenes=m_min,
                                         total_events_floor=float(np.log(1 / DELTA) / (g * t)), table=rows)
    return out


def power_sim(rng, ns, label):
    m, N = len(ns), int(ns.sum())
    n_min = int(ns.min())
    res = []
    for mu in MUS:
        for tau in TAUS:
            if mu == 0 and tau > 0:
                continue
            D = np.stack([beta_draw(rng, mu, tau, m) for _ in range(N_POWER)]) if (mu > 0 and tau > 0) \
                else np.full((N_POWER, m), mu)
            X = rng.binomial(ns[None, :], D)
            Ua = ucp(X.sum(1), N)
            J = (X >= 1).sum(1)
            Up = ucp(J, m)
            dbar = (D * ns).sum(1) / N
            rec = dict(n_scenes=m, N=N, mu=mu, tau=tau,
                       c4a_fleet=dict(U_median=float(np.median(Ua)),
                                      power={f"U<={t}": float((Ua <= t).mean()) for t in TARGETS},
                                      fail_rate=float((Ua < dbar).mean())),
                       c4a_pop=dict(U_median=float(np.median(Up)),
                                    power={f"U<={t}": float((Up <= t).mean()) for t in TARGETS},
                                    fail_rate_vs_mu=float((Up < mu).mean()) if mu > 0 else 0.0),
                       c4b={})
            for t in TARGETS:
                h = 1 - (1 - t) ** n_min
                gh = Up / h
                pi_t = float(np.mean(D > t))      # empirical scene mass above t (this draw's population)
                rec["c4b"][f"t{t}"] = dict(h=h, gamma_hat_median=float(np.median(gh)),
                                           power={f"gamma<={g}": float((gh <= g).mean()) for g in GAMMAS},
                                           true_mass_above_t=pi_t)
            res.append(rec)
    return {label: res}


def main():
    rng = np.random.default_rng(SEED)
    ns_all, cal, las_bg = real_ns()
    ns = ns_all["cdnet44_gap1"]
    out = dict(meta=dict(delta=DELTA, n_mc=N_MC, n_power=N_POWER, seed=SEED, mus=MUS, taus_cv=TAUS,
                         targets_on_D=TARGETS, gammas=GAMMAS,
                         beta="D_s ~ Beta with mean mu and coefficient of variation tau (tau=0: D_s = mu)",
                         calibration_scenes=cal, lasiesta_bg_scenes=las_bg),
               n_s={k: dict(n_scenes=int(len(v)), N=int(v.sum()), median=float(np.median(v)),
                            min=int(v.min()), max=int(v.max()), values=v.tolist()) for k, v in ns_all.items()})
    out["lemma_mean_median"] = lemma_check()
    print(out["lemma_mean_median"])
    out["c4a_fleet_validity"] = validity_c4a(rng, ns)
    for k, v in out["c4a_fleet_validity"].items():
        print(k, round(v["exact_fail_poisson_binomial"], 4), round(v["exact_fail_binomial"], 4), v["mc_fail"], v["valid"])
    # zero-discordance bounds with the real data size
    out["zero_discordance_bounds"] = {k: dict(c4a_fleet_U=float(ucp(0, int(v.sum()))), c4a_pop_U=float(ucp(0, len(v))),
                                              c4b_gamma_hat={f"t{t}": float(ucp(0, len(v)) / (1 - (1 - t) ** int(v.min())))
                                                             for t in TARGETS})
                                      for k, v in ns_all.items()}
    out["c4b_requirements"] = c4b_requirements()
    out["scenes_needed_c4a_pop"] = {f"U<={t}": int(np.ceil(np.log(DELTA) / np.log(1 - t))) for t in TARGETS}
    out["power"] = {}
    for k in ("cdnet44_gap1", "cdnet44_gap1_plus_lasiesta"):
        out["power"].update(power_sim(rng, ns_all[k], k))
    # real events saved for a NEW camera (operating points from A2, G*, at p_hat)
    op = json.load(open(RESULTS / "operating_points.json", encoding="utf-8"))["tables"]["Gstar"]
    sav = {}
    for d, K, eps in [(32, 1, 0.10), (32, 8, 0.20), (16, 4, 0.20), (16, 1, 0.20)]:
        c = op[f"d{d}"]["cells"][f"K{K}"]
        sav[f"d{d}_K{K}_eps{eps}"] = dict(p_hat=c["p_hat"], n_required_direct=c["n_required"][f"eps{eps}"]["at_p_hat"],
                                          n0_zero_failure=n0(eps, DELTA))
    out["new_camera_saving_reference"] = sav
    dump(out, "c4_power.json")
    for k, recs in out["power"].items():
        for r in recs:
            print(k, r["mu"], r["tau"], "C4a-fleet U med", round(r["c4a_fleet"]["U_median"], 4),
                  "pow<=2.5%", r["c4a_fleet"]["power"]["U<=0.025"], "| pop U med", round(r["c4a_pop"]["U_median"], 4),
                  "| C4b t=5% gamma med", round(r["c4b"]["t0.05"]["gamma_hat_median"], 3))
    print(json.dumps(out["c4b_requirements"], indent=0))
    print(out["zero_discordance_bounds"])


if __name__ == "__main__":
    main()
