"""P1c M4 — exchange rate: real events a main-domain camera needs WITH the twin vs WITHOUT it, over the grid
d_min x K, using the MAIN-DOMAIN real miss rate (not the 11.3% of OP*, which was pooled over all CDnet incl. night /
turbulence / PTZ; E0.2).

Per grid point (eps = 20%; 10% as sensitivity), main domain (78 scenes, A>=256, unit = event, majority twin):
  p_main      = mean_e P_phi(real miss)  (all main events with duration >= d_min)
  n_direct    = smallest n with P(one-sided CP upper bound on p <= eps) >= 80% at p = p_main (a2_operating_points)
  n0(eps)     = zero-failure success run (14 at eps 20%, 29 at 10%); references n0(5%,5%) = 59 and 112 (old OP* figure)
  twin route  (C3 + C4a-pop, delta split in half): certify with ZERO real events iff q_U,s + U_pop,-s <= eps, where
              U_pop,-s = U_CP(J_-s, 77) at delta/2 (J from the OTHER scenes, worst-phase integer) and
              q_U,s = (a) q_s itself ("twin runs are cheap": q known), or (b) U_CP at delta/2 on the twin events we have
              (3 sprite variants x the scene's events, fractional expected miss count) - the finite-twin reading.
              If the twin route fails the camera falls back to the direct route (saving 0).
  saving_s    = n_ref - n_real_twin (n_ref = n_direct; also vs n0(eps), 59, 112), in events and in %.
  waiting hours saved = n_saved / lambda_s, lambda_s = the scene's events (>= d_min) per hour of video.
G2 (PLAN 5): saving >= 50% of real events at >= 50% of cameras.
fps: notes/fps_sources.json has BMC only. CDnet / LASIESTA fps are NOT documented per video; fps is taken from the
video name when it states it (port_0_17fps, tramCrossroad_1fps, tunnelExit_0_35fps, turnpike_0_5fps), otherwise an
ASSUMED nominal fps (25; sensitivity 15 and 30). Video length = annotated frames in the G* tables.
Twin cost: wall seconds per scene from the C0 run logs (results/c0_main.log, results/d1_c0.out; 3 workers, 1 scene per
worker, CPU only) -> CPU-hours per scene (= wall hours of one worker).
Output: results/exchange_rate.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a2_operating_points import n_required  # noqa: E402
from c4_power import ucp  # noqa: E402
from common import RESULTS, SEED, dump, n0  # noqa: E402
from e0_checks import event_rows  # noqa: E402
from m3_validity_audit import event_table  # noqa: E402

DELTA = 0.05
D_GRID = [32, 24, 16, 12, 8, 4, 1]
K_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]
NAMED_FPS = {"port_0_17fps": 0.17, "tramCrossroad_1fps": 1.0, "tunnelExit_0_35fps": 0.35, "turnpike_0_5fps": 0.5}
FPS_ASSUMED = [25.0, 15.0, 30.0]
MARKED = {"OP*": (32, 16, 0.20), "OP-A": (32, 8, 0.20), "OP-C": (32, 1, 0.10), "challenge_16_16": (16, 16, 0.20)}


def twin_cost():
    sec = {}
    for f in ("c0_main.log", "d1_c0.out"):
        for line in open(RESULTS / f, encoding="utf-8", errors="replace"):
            m = re.search(r"twin_c0_main[\\/]frames_([^_]+)__(.+)\.parquet (\d+) rows (\d+)s", line)
            if m:
                sec[m.group(2)] = int(m.group(4))
    return sec


def video_frames():
    n = {}
    for ds in ("CDnet2014", "LASIESTA"):
        F = pd.read_parquet(RESULTS / f"gstar_frames_{ds}.parquet", columns=["video", "frame"])
        n.update(F.groupby("video").size().to_dict())
    return n


def twin_qU(rows, d_min, K):
    """Finite-twin q upper bound at delta/2: 3 variants x events, expected miss count (majority not used: each
    variant is one twin run)."""
    out = {}
    per = {}
    for r in rows:
        if r["duration"] < d_min:
            continue
        th = r[f"th{K}"]
        per.setdefault(r["video"], []).extend([(1 - th[v].mean()) for v in range(3)])
    for v, xs in per.items():
        out[v] = float(ucp(float(np.sum(xs)), len(xs), DELTA / 2))
    return out


def n_twin_needed(q, U, eps, n_max=5000):
    """Twin events (runs) needed so that U_CP(q n, n) at delta/2 + U <= eps, at the observed twin miss rate q."""
    if q + U >= eps:
        return None
    for n in range(1, n_max + 1):
        if ucp(q * n, n, DELTA / 2) + U <= eps:
            return n
    return None


def point(rows, d_min, K, eps, frames, cost):
    ev = event_table(rows, d_min, K)
    S = ev.groupby("video").agg(n=("x", "size"), p=("p", "mean"), q=("q", "mean"), w=("w", "max"))
    m = len(S)
    J = int(S.w.sum())
    p_main = float(ev.p.mean())
    nd = n_required(p_main, eps)
    qfin = twin_qU(rows, d_min, K)
    recs = []
    for v, r in S.iterrows():
        U = float(ucp(J - int(r.w), m - 1, DELTA / 2))
        ok_known = bool(r.q + U <= eps)
        ok_fin = bool(qfin[v] + U <= eps)
        fps = NAMED_FPS.get(v, None)
        hours = {f: frames[v] / (fps or f) / 3600 for f in FPS_ASSUMED}
        lam = {f: r.n / h for f, h in hours.items()}          # events (>= d_min) per hour
        rec = dict(video=v, n_events=int(r.n), p_s=float(r.p), q_s=float(r.q), q_U_finite=qfin[v], U_pop_half_delta=U,
                   twin_ok_q_known=ok_known, twin_ok_q_finite=ok_fin, fps_named=fps,
                   lambda_per_hour={str(f): float(x) for f, x in lam.items()},
                   twin_cpu_hours=(cost[v] / 3600 if v in cost else None),
                   n_twin_events_needed=n_twin_needed(float(r.q), U, eps),
                   n_twin_events_available=3 * sum(1 for rr in rows if rr["video"] == v and rr["duration"] >= d_min),
                   twin_cpu_hours_per_twin_event=(cost[v] / 3600 / (3 * sum(1 for rr in rows if rr["video"] == v))
                                                  if v in cost else None))
        for tag, ok in (("q_known", ok_known), ("q_finite", ok_fin)):
            n_real = 0 if ok else nd
            rec[tag] = dict(n_real=n_real,
                            saving_vs_direct=(None if nd is None else nd - n_real),
                            saving_pct_vs_direct=(None if nd in (None, 0) else 100 * (nd - n_real) / nd),
                            saving_vs_n0=(n0(eps, DELTA) if ok else 0), saving_vs_59=(59 if ok else 0),
                            saving_vs_112=(112 if ok else 0),
                            wait_hours_saved_vs_direct={str(f): (None if nd is None else (nd - n_real) / lam[f])
                                                        for f in FPS_ASSUMED})
        recs.append(rec)
    R = pd.DataFrame(recs)

    def agg(tag):
        s = pd.DataFrame([r[tag] for r in recs])
        sp = s.saving_pct_vs_direct.astype(float)
        wh = pd.DataFrame(list(s.wait_hours_saved_vs_direct)).astype(float)
        nt = R.n_twin_events_needed.dropna()
        ct = (R.n_twin_events_needed * R.twin_cpu_hours_per_twin_event).dropna()
        return dict(cameras_twin_ok=int(R[f"twin_ok_{tag}"].sum()), m=m,
                    n_twin_events_needed_median=(float(nt.median()) if len(nt) else None),
                    n_twin_events_needed_iqr=([float(x) for x in np.percentile(nt, [25, 75])] if len(nt) else None),
                    twin_cpu_hours_needed_median=(float(ct.median()) if len(ct) else None),
                    n_real_median=float(np.median(s.n_real.dropna())) if s.n_real.notna().any() else None,
                    n_real_iqr=([float(x) for x in np.percentile(s.n_real.dropna(), [25, 75])]
                                if s.n_real.notna().any() else None),
                    share_cameras_saving_ge_50pct=(float((sp >= 50).mean()) if sp.notna().any() else None),
                    G2_saving_met=(bool((sp >= 50).mean() >= 0.5) if sp.notna().any() else False),
                    events_saved_total_vs_direct=float(s.saving_vs_direct.fillna(0).sum()),
                    wait_hours_saved_median={k: float(wh[k].median()) for k in wh},
                    wait_hours_saved_iqr={k: [float(x) for x in np.nanpercentile(wh[k], [25, 75])] for k in wh})
    return dict(d_min=d_min, K=K, eps=eps, N_events=len(ev), p_main=p_main, J=J,
                n_direct_at_p_main=nd, n0=n0(eps, DELTA), direct_certifiable=nd is not None,
                q_known=agg("q_known"), q_finite=agg("q_finite")), recs


def main():
    rows = event_rows("c0_main", "main")
    frames, cost = video_frames(), twin_cost()
    grid, marked = [], {}
    for eps in (0.20, 0.10):
        for d in D_GRID:
            for K in K_GRID:
                g, recs = point(rows, d, K, eps, frames, cost)
                grid.append(g)
                for name, (dm, kk, ee) in MARKED.items():
                    if (dm, kk, ee) == (d, K, eps):
                        marked[name] = dict(**g, per_camera=recs)
    c = np.array([v for v in cost.values()]) / 3600
    out = dict(meta=dict(doc=__doc__.split("Output")[0].strip(), delta=DELTA, seed=SEED, power=0.80,
                         fps_note="fps from video name for 4 CDnet videos; otherwise ASSUMED 25 fps (15 / 30 sensitivity)"),
               twin_cost=dict(n_scenes=len(cost), cpu_hours_per_scene_median=float(np.median(c)),
                              cpu_hours_per_scene_iqr=[float(x) for x in np.percentile(c, [25, 75])],
                              cpu_hours_total=float(c.sum()),
                              note="wall time of one CPU worker (i7-1185G7), 3 sprite variants, early stop at 48 residues"),
               marked=marked, grid=grid)
    G = pd.DataFrame([dict(eps=g["eps"], d_min=g["d_min"], K=g["K"], p_main=g["p_main"], n_direct=g["n_direct_at_p_main"],
                           ok_known=g["q_known"]["cameras_twin_ok"], ok_fin=g["q_finite"]["cameras_twin_ok"],
                           G2_known=g["q_known"]["G2_saving_met"], G2_fin=g["q_finite"]["G2_saving_met"]) for g in grid])
    out["grid_summary"] = dict(
        eps20_points=int((G.eps == 0.2).sum()),
        eps20_direct_certifiable=int(((G.eps == 0.2) & G.n_direct.notna()).sum()),
        eps20_G2_met_q_known=int(((G.eps == 0.2) & G.G2_known).sum()),
        eps20_G2_met_q_finite=int(((G.eps == 0.2) & G.G2_fin).sum()),
        n_direct_range_eps20=[float(G[(G.eps == 0.2)].n_direct.min()), float(G[(G.eps == 0.2)].n_direct.max())])
    dump(out, "exchange_rate.json")
    pd.set_option("display.width", 200)
    print(G[G.eps == 0.2].round(4).to_string())
    for k, v in marked.items():
        print(k, v["p_main"], v["n_direct_at_p_main"], v["J"], {t: {kk: v[t][kk] for kk in ("cameras_twin_ok", "n_twin_events_needed_median", "twin_cpu_hours_needed_median", "share_cameras_saving_ge_50pct", "G2_saving_met", "events_saved_total_vs_direct", "wait_hours_saved_median")} for t in ("q_known", "q_finite")})
    print(out["twin_cost"], out["grid_summary"])


if __name__ == "__main__":
    main()
