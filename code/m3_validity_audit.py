"""P1c M3 — validity audit of the twin certificate on the main domain (76 scenes in v1.1; 78 in v1.0), at OP* (d_min 32, K 16) and at the
post-hoc challenge point (d_min 16, K 16) chosen in E0.2 (the nearest point with >= 20 real-miss events).

Unit (E0.1): event; twin catch = >= 2 of 3 sprite variants; a sampling phase uniform in {0..K-1} per event.
Per event e: x_e = P_phi(real miss AND twin catch), p_e = P_phi(real miss), q_e = P_phi(twin miss).
Per scene s: D_s = mean_e x_e, p_s = mean_e p_e, q_s = mean_e q_e (exact over the scene's events: the scene's
finite event pool is its "truth"); beta_s = 1 - prod_e (1 - x_e) = P(scene shows >= 1 discordance at a random draw).

Certification rule for a NEW camera s (C3 + C4a-pop): certify "p_s <= eps" iff q_s + U_pop <= eps, where
U_pop = U_CP(J, m) at delta (J = #calibration scenes with >= 1 discordant event) and q_s is the twin miss rate of the
camera (twin runs are cheap: q treated as known; the finite-twin variant splits delta, see M4).

(1) LOSO: for each scene s, J_{-s} from the other m - 1 scenes (worst-phase integer J, conservative);
    violation_D = D_s > U_pop(-s); violation_C3 = p_s > q_s + U_pop(-s); false certificate = certified AND p_s > eps.
(2) MC false certification (10,000 runs, seed 42): population model = the m main scenes. Each run: m calibration
    scenes drawn with replacement, each shows a discordance with prob beta_s (random phases) -> J -> U_pop;
    a new camera drawn uniformly and independently; false certificate = certified AND p_s > eps.
    Also the theorem target of C4a-pop: P(U_pop < mu), mu = population mean of D_s  (must be <= delta).
    PLAN asked for "truth" videos with >= 59 real events: no main-domain scene has that many A>=256 G* events with a
    twin (max 13 at OP*, 34 at d_min 1) -> the truth is the exact finite pool of each scene (deviation, reported).
(3) Must-fail: operating point(s) where p_s > eps in >= 3 main scenes; the rule must refuse those cameras in >= 95% of
    MC runs. Misses split into structural (duration < K) and appearance.
(4) C4a-fleet under the MEASURED heterogeneous discordance: event-level Bernoulli(x_e) (and the scene-level D_s with the
    real n_s): exact Poisson-binomial failure P(U(X, N) < Dbar) by DP + MC 100,000 (seed 42); also the same pattern
    scaled to Dbar = 2%, 5% (stress).
Output: results/validity_audit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c4_power import pb_cdf, ucp, xstar  # noqa: E402
from common import RESULTS, SEED, dump  # noqa: E402
from e0_checks import cells, event_rows  # noqa: E402

DELTA = 0.05
N_MC = 10_000
N_MC_THM = 100_000
POINTS = {"OP*": dict(eps=0.20, d_min=32, K=16),
          "challenge_16_16": dict(eps=0.20, d_min=16, K=16, note="post-hoc: chosen after E0.2 as the nearest point "
                                                                   "with >= 20 real-miss events")}
MUST_FAIL = [dict(eps=0.20, d_min=1, K=48), dict(eps=0.20, d_min=4, K=48), dict(eps=0.20, d_min=16, K=48)]


def event_table(rows, d_min, K, rule=2):
    out = []
    for r in rows:
        if r["duration"] < d_min:
            continue
        rm, tc, st = cells(r, K, rule)
        out.append(dict(video=r["video"], event_id=r["event_id"], x=float((rm & tc).mean()), p=float(rm.mean()),
                        q=float((~tc).mean()), p_app=float((rm & ~st).mean()), p_struct=float((rm & st).mean()),
                        w=float((rm & tc).any())))
    return pd.DataFrame(out)


def scene_table(ev):
    g = ev.groupby("video")
    S = pd.DataFrame(dict(n=g.size(), D=g.x.mean(), p=g.p.mean(), q=g.q.mean(), p_app=g.p_app.mean(),
                          p_struct=g.p_struct.mean(), w=g.w.max(),
                          beta=g.x.apply(lambda s: 1 - np.prod(1 - s.values))))
    return S


def loso(S, eps):
    m = len(S)
    J = int(S.w.sum())
    rows = []
    for v, r in S.iterrows():
        Jm = J - int(r.w)
        U = float(ucp(Jm, m - 1))
        cert = bool(r.q + U <= eps)
        rows.append(dict(video=v, n=int(r.n), D_s=r.D, p_s=r.p, q_s=r.q, J_minus=Jm, U_pop_minus=U,
                         viol_D=bool(r.D > U), viol_C3=bool(r.p > r.q + U), certified=cert,
                         false_cert=bool(cert and r.p > eps)))
    L = pd.DataFrame(rows)
    return dict(m=m, J_all=J, U_pop_all=float(ucp(J, m)),
                violation_rate_D=dict(num=int(L.viol_D.sum()), den=m, value=float(L.viol_D.mean())),
                violation_rate_C3=dict(num=int(L.viol_C3.sum()), den=m, value=float(L.viol_C3.mean())),
                certified=dict(num=int(L.certified.sum()), den=m),
                false_certificates=dict(num=int(L.false_cert.sum()), den=m, value=float(L.false_cert.mean())),
                scenes_p_gt_eps=int((S.p > eps).sum()), pass_le_delta=bool(L.viol_D.mean() <= DELTA),
                violating_scenes=L[L.viol_D | L.viol_C3 | L.false_cert].to_dict("records"))


def mc_false_cert(S, eps, rng, n_mc=N_MC, restrict=None):
    m = len(S)
    beta, p, q, D = S.beta.values, S.p.values, S.q.values, S.D.values
    mu = float(D.mean())
    cal = rng.integers(0, m, size=(n_mc, m))
    J = (rng.random((n_mc, m)) < beta[cal]).sum(1)
    U = ucp(J, m)
    pool = np.flatnonzero(restrict) if restrict is not None else np.arange(m)
    new = pool[rng.integers(0, len(pool), size=n_mc)]
    cert = q[new] + U <= eps
    fc = cert & (p[new] > eps)
    se = lambda x: float(np.sqrt(max(x, 1e-12) * (1 - x) / n_mc))    # noqa: E731
    return dict(n_mc=n_mc, mu_pop_D=mu, U_pop_median=float(np.median(U)),
                P_U_lt_mu=float((U < mu).mean()), P_U_lt_mu_se=se(float((U < mu).mean())),
                certify_rate=float(cert.mean()), false_cert_rate=float(fc.mean()), false_cert_se=se(float(fc.mean())),
                false_cert_given_p_gt_eps=(float(fc[p[new] > eps].mean()) if (p[new] > eps).any() else None),
                refuse_rate_given_p_gt_eps=(float(1 - cert[p[new] > eps].mean()) if (p[new] > eps).any() else None),
                share_new_with_p_gt_eps=float((p[new] > eps).mean()))


def fleet_check(ev, rng, label):
    """C4a-fleet: P(U(X, N) < Dbar) under the measured event-level x_e (heterogeneous Poisson-binomial)."""
    out = {}
    base = ev.x.values
    N = len(base)
    for tag, scale in (("measured", None), ("scaled_to_2pct", 0.02), ("scaled_to_5pct", 0.05)):
        ps = base.copy()
        if scale is not None:
            if ps.sum() == 0:
                continue
            ps = np.minimum(1.0, ps * scale / ps.mean())
        dbar = float(ps.mean())
        if dbar == 0:
            out[tag] = dict(Dbar=0.0, N=N, note="Dbar = 0: failure impossible (U >= 0)")
            continue
        xs = xstar(N, dbar, DELTA)
        exact = pb_cdf(ps, xs) if xs >= 0 else 0.0
        X = (rng.random((N_MC_THM, N)) < ps).sum(1) if N * N_MC_THM <= 4e7 else \
            np.concatenate([(rng.random((N_MC_THM // 10, N)) < ps).sum(1) for _ in range(10)])
        fail = float((ucp(X, N) < dbar).mean())
        out[tag] = dict(Dbar=dbar, N=N, n_events_x_pos=int((ps > 0).sum()), x_star=int(xs),
                        exact_fail_poisson_binomial=exact,
                        exact_fail_binomial_same_mean=float(__import__("scipy.stats", fromlist=["binom"]).binom.cdf(xs, N, dbar)) if xs >= 0 else 0.0,
                        mc_fail=fail, mc_se=float(np.sqrt(max(fail, 1e-12) * (1 - fail) / N_MC_THM)),
                        valid=bool(exact <= DELTA and fail <= DELTA + 3 * np.sqrt(DELTA * (1 - DELTA) / N_MC_THM)))
    return {label: out}


def pop_theorem_check(m, n_mc=N_MC_THM):
    """C4a-pop / C4b at their tight case (separate generator, seed 42): D_S in {0,1} with P(D_S = 1) = mu and n_s = 1
    -> J ~ Bin(m, mu) exactly; P(U(J, m) < mu) must be <= delta. C4b: D_S in {0, t'} (t' > t) with mass pi, n_s = n;
    P(U(J, m)/h_t < pi) must be <= delta."""
    rng = np.random.default_rng(SEED)
    pop = {}
    for mu in (0.01, 0.02, 0.05, 0.10, 0.20):
        J = (rng.random((n_mc, m)) < mu).sum(1)
        f = float((ucp(J, m) < mu).mean())
        pop[str(mu)] = dict(fail=f, se=float(np.sqrt(max(f, 1e-12) * (1 - f) / n_mc)))
    c4b = {}
    for t, tp, pi, n in ((0.05, 0.10, 0.10, 20), (0.10, 0.20, 0.10, 10), (0.05, 0.05001, 0.20, 30)):
        h = 1 - (1 - t) ** n
        Ds = np.where(rng.random((n_mc, m)) < pi, tp, 0.0)
        J = (rng.binomial(n, Ds) >= 1).sum(1)
        f = float((ucp(J, m) / h < pi).mean())
        c4b[f"t{t}_pi{pi}_n{n}"] = dict(h_t=h, fail=f)
    return dict(m=m, n_mc=n_mc, c4a_pop_tight=pop, c4a_pop_max_fail=max(v["fail"] for v in pop.values()),
                c4b=c4b, c4b_max_fail=max(v["fail"] for v in c4b.values()))


def main():
    rng = np.random.default_rng(SEED)
    rows = event_rows("c0_main", "main")
    out = dict(meta=dict(doc=__doc__.split("Output")[0].strip(), delta=DELTA, n_mc=N_MC, n_mc_theorem=N_MC_THM, seed=SEED,
                         deviation_truth_videos="no main-domain scene has >= 59 A>=256 events with a twin; truth = exact "
                                                "finite event pool of each scene; see max_events_per_scene"),
               points={}, must_fail={}, c4a_fleet={})
    for name, op in POINTS.items():
        ev = event_table(rows, op["d_min"], op["K"])
        S = scene_table(ev)
        rec = dict(op=op, N_events=len(ev), m_scenes=len(S), max_events_per_scene=int(S.n.max()),
                   scenes_with_D_pos=S[S.D > 0][["n", "D", "p", "q"]].reset_index().to_dict("records"),
                   loso=loso(S, op["eps"]), mc=mc_false_cert(S, op["eps"], rng))
        rec["positive_conditional_claim_kept"] = bool(rec["loso"]["violation_rate_D"]["value"] <= DELTA
                                                      and rec["loso"]["false_certificates"]["num"] == 0
                                                      and rec["mc"]["P_U_lt_mu"] <= DELTA + 0.004)
        out["points"][name] = rec
        out["c4a_fleet"].update(fleet_check(ev, rng, name))
    for op in MUST_FAIL:
        ev = event_table(rows, op["d_min"], op["K"])
        S = scene_table(ev)
        bad = (S.p > op["eps"]).values
        key = f"d{op['d_min']}_K{op['K']}"
        rec = dict(op=op, scenes_p_gt_eps=S[bad][["n", "p", "p_app", "p_struct", "q", "D"]].reset_index().to_dict("records"),
                   n_scenes_p_gt_eps=int(bad.sum()))
        if bad.sum() >= 3:
            rec["mc_restricted_to_p_gt_eps"] = mc_false_cert(S, op["eps"], rng, restrict=bad)
            rec["loso"] = loso(S, op["eps"])
            rr = rec["mc_restricted_to_p_gt_eps"]["refuse_rate_given_p_gt_eps"]
            rec["refuse_ge_95pct"] = bool(rr is not None and rr >= 0.95)
            # appearance-only variant: would the rule still refuse if structural misses were removed from BOTH sides?
            app_bad = (S.p_app > op["eps"]).values
            rec["scenes_p_app_gt_eps"] = int(app_bad.sum())
        out["must_fail"][key] = rec
        out["c4a_fleet"].update(fleet_check(ev, rng, key))
    mf = [v for v in out["must_fail"].values() if v.get("refuse_ge_95pct") is not None]
    out["summary"] = {k: dict(loso_viol_D=v["loso"]["violation_rate_D"], loso_viol_C3=v["loso"]["violation_rate_C3"],
                              loso_false_cert=v["loso"]["false_certificates"], mc_false_cert=v["mc"]["false_cert_rate"],
                              mc_P_U_lt_mu=v["mc"]["P_U_lt_mu"], positive_conditional_claim_kept=v["positive_conditional_claim_kept"])
                      for k, v in out["points"].items()}
    out["summary"]["must_fail_all_refuse_ge_95pct"] = bool(mf and all(v["refuse_ge_95pct"] for v in mf))
    out["summary"]["c4a_fleet_all_valid"] = bool(all(r.get("valid", True) for d in out["c4a_fleet"].values() for r in d.values()))
    out["pop_theorem_check"] = pop_theorem_check(out["points"]["OP*"]["m_scenes"])
    out["summary"]["c4a_pop_tight_max_fail"] = out["pop_theorem_check"]["c4a_pop_max_fail"]
    out["summary"]["c4b_max_fail"] = out["pop_theorem_check"]["c4b_max_fail"]
    dump(out, "validity_audit.json")
    print(json.dumps(out["summary"], indent=1, default=float))
    for k, v in out["must_fail"].items():
        print(k, v["n_scenes_p_gt_eps"], v.get("refuse_ge_95pct"), v.get("scenes_p_app_gt_eps"))
    for k, d in out["c4a_fleet"].items():
        print(k, {t: (round(r.get("Dbar", 0), 4), r.get("exact_fail_poisson_binomial"), r.get("mc_fail"), r.get("valid")) for t, r in d.items()})


if __name__ == "__main__":
    main()
