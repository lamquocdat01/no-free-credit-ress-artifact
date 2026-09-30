"""P1c — C2 on real simulator data (S0 = BMC-synth, S1 = the main-domain twin) and C5a (pessimistic-length twin is
conservative under periodic sampling K).

C2 (borrowing). Two-tier procedure B(delta'): run the simulator test first -- the simulator "passes" iff the one-sided
CP (1-delta) upper bound of its miss rate over its N_sim events is <= eps; if it passes, certify on real data with the
zero-failure test of size n0(eps, delta'), otherwise with n0(eps, delta). Exact credit when it passes:
c = n0(eps, delta) - n0(eps, delta') = 8/14/22/27 at eps = delta = 5% (theory_C1_C2_C3.md, C2).
Measured by MC (100,000 runs, seed 42; sim events bootstrapped from the pool with a uniform random phase; real
events Bernoulli(p)): (i) P(sim passes), realised credit = c * P(pass) (in events and % of n0);
(ii) false-certification at the worst-case truth p = eps (must be <= delta' when the sim passes, and the unconditional
rate is what C1/C2 price); (iii) power at the main-domain p (event unit, phase-uniform).
Operating points: eps = 5% (the C2 table) at (d_min 32, K 1) and eps = 20% at OP* (d_min 32, K 16).
Pools: S0 = BMC-synth Mode G* events (frame level, 31 events; all d >= 32 except one) and Mode T (261 events);
S1 = twin of the main scenes (76 in v1.1) (majority of 3 variants).

C5a. On the twin, shorten every event to its first ceil((1-x) d) frames, x in {0, 10, 25, 50}%; twin miss q (and the
real p) at each K must be non-decreasing in x (per event: residue sets shrink; pooled: mean of per-event
non-decreasing values). Exact under the twin's mod-48 early stop (once all 48 residues are covered, later frames
cannot change any residue set).
Outputs: results/c2_empirical.json, results/c5a_monotone.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c4_power import ucp  # noqa: E402
from common import RESULTS, SEED, dump, n0  # noqa: E402
from e0_checks import cells, event_rows, real_hit_frames, tier_videos  # noqa: E402
from m3_validity_audit import event_table  # noqa: E402

N_MC = 100_000
DELTA = 0.05
DPRIMES = [0.075, 0.10, 0.15, 0.20]
C_EXPECTED = {0.075: 8, 0.10: 14, 0.15: 22, 0.20: 27}          # eps = delta = 5%
OPS = {"eps5_d32_K1": dict(eps=0.05, d_min=32, K=1), "eps20_OPstar": dict(eps=0.20, d_min=32, K=16)}


def pools(rows, d_min, K):
    G = pd.read_csv(RESULTS / "miss_matrix_Gstar_ext.csv")
    B = G[(G.dataset == "BMC") & (G.duration >= d_min)]
    T = pd.read_csv(RESULTS / "miss_matrix_T.csv")
    BT = T[(T.dataset == "BMC") & (T.duration >= d_min)]
    tw = event_table(rows, d_min, K)
    return {"S0_BMC_Gstar": B[f"pGs_K{K}"].values, "S0_BMC_T": BT[f"p_miss_K{K}"].values, "S1_twin_main": tw.q.values}, \
        float(tw.p.mean()), len(tw)


def c2(rows, rng):
    out = {}
    for name, op in OPS.items():
        eps = op["eps"]
        P, p_main, n_main = pools(rows, op["d_min"], op["K"])
        N0 = n0(eps, DELTA)
        rec = dict(op=op, n0=N0, p_main=p_main, n_main_events=n_main, pools={})
        for pn, q in P.items():
            Ns = len(q)
            idx = rng.integers(0, Ns, size=(N_MC, Ns))
            miss = (rng.random((N_MC, Ns)) < q[idx]).sum(1)
            passes = ucp(miss, Ns) <= eps
            pr = dict(N_sim=Ns, q_sim_mean=float(q.mean()), P_sim_pass=float(passes.mean()), by_delta_prime={})
            for dp in DPRIMES:
                n1 = n0(eps, dp)
                c = N0 - n1
                n_used = np.where(passes, n1, N0)
                fc = (rng.random(N_MC) < (1 - eps) ** n_used)       # zero failures among n_used real events at p = eps
                pw = (rng.random(N_MC) < (1 - p_main) ** n_used)
                pr["by_delta_prime"][str(dp)] = dict(
                    c_exact=c, c_expected_table=(C_EXPECTED[dp] if eps == 0.05 else None),
                    c_matches=(c == C_EXPECTED[dp]) if eps == 0.05 else None,
                    realised_credit_events=float(c * passes.mean()), realised_credit_pct_of_n0=float(100 * c * passes.mean() / N0),
                    false_cert_at_p_eq_eps=float(fc.mean()), closed_form=float(passes.mean() * (1 - eps) ** n1 + (1 - passes.mean()) * (1 - eps) ** N0),
                    false_cert_given_pass=float(fc[passes].mean()) if passes.any() else None,
                    false_cert_given_pass_le_delta_prime=bool(fc[passes].mean() <= dp + 3 * np.sqrt(dp * (1 - dp) / max(1, passes.sum()))) if passes.any() else None,
                    power_at_p_main=float(pw.mean()))
            rec["pools"][pn] = pr
        out[name] = rec
    return out


def c5a(rows):
    T = pd.read_parquet(RESULTS / "gstar_frames_twin_c0_main.parquet")
    T = T[T.def_a256 & T.video.isin(tier_videos("main"))]
    E = pd.read_csv(RESULTS / "miss_matrix_Gstar_a256.csv")
    E = E[E.def_a256 & E.video.isin(T.video.unique())]
    RH = real_hit_frames(E, {})
    xs = [0.0, 0.10, 0.25, 0.50]
    Ks = [1, 4, 8, 16, 48]
    per = []
    for (ds, v, eid), g in T.groupby(["dataset", "video", "event_id"]):
        e = E[(E.dataset == ds) & (E.video == v) & (E.event_id == eid)].iloc[0]
        on, d = int(e.onset_frame), int(e.duration)
        rr = RH[(ds, v, int(eid))]
        for x in xs:
            L = int(np.ceil((1 - x) * d))
            hi = on + L
            fr_all = np.arange(on, hi)
            rec = dict(video=v, event_id=int(eid), duration=d, x=x, L=L)
            for K in Ks:
                rh = np.zeros(K, bool)
                rh[np.unique(rr[rr < hi] % K)] = True
                th = np.zeros((3, K), bool)
                for vi, gv in g.groupby("variant"):
                    f = gv.frame.values[gv.twin_hit.values.astype(bool)]
                    th[int(vi), np.unique(f[f < hi] % K)] = True
                tc = th.sum(0) >= 2
                rec[f"q_K{K}"] = float((~tc).mean())
                rec[f"p_K{K}"] = float((~rh).mean())
            per.append(rec)
    P = pd.DataFrame(per)
    out = dict(unit="event (majority twin), phase-uniform; events of the main domain (A>=256)", shortening=xs, K=Ks, tables={})
    for d_min in (32, 16, 1):
        sub = P[P.duration >= d_min]
        tab = {}
        for K in Ks:
            q = sub.groupby("x")[f"q_K{K}"].mean()
            p = sub.groupby("x")[f"p_K{K}"].mean()
            wide = sub.pivot_table(index=["video", "event_id"], columns="x", values=f"q_K{K}")
            mono_ev = bool((wide.diff(axis=1).iloc[:, 1:] >= -1e-12).all().all())
            tab[f"K{K}"] = dict(q_by_x={str(k): float(v) for k, v in q.items()}, p_by_x={str(k): float(v) for k, v in p.items()},
                                q_monotone=bool(np.all(np.diff(q.values) >= -1e-12)),
                                p_monotone=bool(np.all(np.diff(p.values) >= -1e-12)),
                                per_event_q_monotone=mono_ev, N=int(len(wide)))
        out["tables"][f"d_min{d_min}"] = tab
    out["all_monotone"] = bool(all(t["q_monotone"] and t["per_event_q_monotone"] for d in out["tables"].values() for t in d.values()))
    return out


def main():
    rng = np.random.default_rng(SEED)
    rows = event_rows("c0_main", "main")
    r2 = c2(rows, rng)
    dump(dict(meta=dict(doc=__doc__.split("C5a.")[0].strip(), n_mc=N_MC, seed=SEED, delta=DELTA), results=r2), "c2_empirical.json")
    r5 = c5a(rows)
    dump(dict(meta=dict(doc="C5a." + __doc__.split("C5a.")[1].split("Outputs")[0].rstrip()), **r5), "c5a_monotone.json")
    for k, v in r2.items():
        print(k, "n0", v["n0"], "p_main", round(v["p_main"], 4))
        for pn, pr in v["pools"].items():
            print("  ", pn, pr["N_sim"], round(pr["q_sim_mean"], 4), "pass", pr["P_sim_pass"],
                  {dp: (d["c_exact"], round(d["realised_credit_events"], 2), round(d["false_cert_at_p_eq_eps"], 4), d["false_cert_given_pass"] and round(d["false_cert_given_pass"], 4)) for dp, d in pr["by_delta_prime"].items()})
    print("C5a all monotone", r5["all_monotone"])
    for d, t in r5["tables"].items():
        print(d, {k: [round(x, 4) for x in v["q_by_x"].values()] for k, v in t.items()})


if __name__ == "__main__":
    main()
