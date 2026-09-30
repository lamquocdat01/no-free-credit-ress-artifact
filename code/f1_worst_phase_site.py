"""P4-final F1 — two additional computations from the existing per-frame tables (no detector, no new twin).

1. Worst-phase twin miss. For every event, real miss / twin miss / discordance are taken at the WORST sampling phase:
   p_w = 1[the real event is missed at >= 1 phase], q_w = 1[the twin (majority of 3 variants) misses at >= 1 phase],
   D_w = 1[real miss and twin catch at the same phase, for >= 1 phase] (the D convention of E0.1).
   Per event, p_w <= q_w + D_w (if the real event is missed at phase phi, the twin either misses or catches at phi), so the
   bound holds for any fixed deployment phase. Recomputed at OP* and (16,16): tier table (p_w, q_w, D_w, fleet and
   population bounds) and the exchange rate (direct plan at the pooled worst-phase p, twin route with worst-phase q).
2. Site-level population bound. Scenes are grouped into sites, a coarser unit of correlation than the scene:
   CDnet 2014 = challenge category; LASIESTA = sequence group (name prefix, e.g. I_SM, O_SM). J_site = sites with at least
   one discordant (worst-phase) event; U_pop-site = U_CP(J_site, m_site); leave-one-site-out audit.
3. (v1.1, post-F1 correction) cameraJitter is its own tier (code/corpus17.py). The cameras of the out-of-domain tiers
   (jitter, night) are also judged by the twin route calibrated on the MAIN scenes only (worst-phase q, transfer term
   U_CP(J_main, m_main) at delta/2): false acceptance = accepted and p_w > eps (named failure modes).
Outputs: results/worst_phase.json, results/site_bound.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a2_operating_points import n_required  # noqa: E402
from c4_power import ucp  # noqa: E402
from common import RESULTS, SEED, dump  # noqa: E402
from e0_checks import cells, event_rows  # noqa: E402
from m4_exchange_rate import n_twin_needed  # noqa: E402

DELTA = 0.05
POINTS = {"OP*": (32, 16, 0.20), "challenge_16_16": (16, 16, 0.20)}
TIERS = {"main": ["c0_main"], "jitter": ["c0_main"], "night": ["nheld_b", "ntune_b"], "turbulence": ["ttune_base"]}


def worst_table(rows, d_min, K):
    out = []
    for r in rows:
        if r["duration"] < d_min:
            continue
        rm, tc, st = cells(r, K, 2)
        out.append(dict(video=r["video"], category=r["category"], p_w=float(rm.any()), q_w=float((~tc).any()),
                        D_w=float((rm & tc).any()), p=float(rm.mean()), q=float((~tc).mean())))
    return pd.DataFrame(out)


def tier_record(ev):
    N, m = len(ev), ev.video.nunique()
    Xw = int(ev.D_w.sum())
    J = int((ev.groupby("video").D_w.max() > 0).sum())
    return dict(N=N, m=m, p_w=float(ev.p_w.mean()), q_w=float(ev.q_w.mean()), D_w=Xw, D_w_rate=Xw / N,
                U_fleet=float(ucp(Xw, N)), J=J, U_pop=float(ucp(J, m)), p_phase_avg=float(ev.p.mean()),
                q_phase_avg=float(ev.q.mean()), check_p_le_q_plus_D=bool((ev.p_w <= ev.q_w + ev.D_w + 1e-12).all()))


def exchange(ev, eps):
    S = ev.groupby("video").agg(n=("p_w", "size"), p_w=("p_w", "mean"), q_w=("q_w", "mean"), w=("D_w", "max"))
    m, J = len(S), int(S.w.sum())
    p_main = float(ev.p_w.mean())
    nd = n_required(p_main, eps)
    rec = []
    for v, r in S.iterrows():
        U = float(ucp(J - int(r.w), m - 1, DELTA / 2))
        ok = bool(r.q_w + U <= eps)
        rec.append(dict(video=v, q_w=float(r.q_w), p_w=float(r.p_w), U_half=U, accepted=ok,
                        saved=(nd if (ok and nd is not None) else 0), false_accept=bool(ok and r.p_w > eps),
                        n_twin_needed=n_twin_needed(float(r.q_w), U, eps)))
    R = pd.DataFrame(rec)
    nt = R.n_twin_needed.dropna()
    return dict(p_main_worst=p_main, n_direct=nd, cameras_accepted=int(R.accepted.sum()), m=m,
                saved_median=float(R.saved.median()), saved_iqr=[float(x) for x in np.percentile(R.saved, [25, 75])],
                false_accept=int(R.false_accept.sum()),
                n_twin_needed_median=(float(nt.median()) if len(nt) else None),
                note="twin route with worst-phase q (q known), transfer term U_CP(J_-s, m-1) at delta/2")


def out_of_domain(ev_main, ev_ood, eps):
    """Out-of-domain cameras judged with a main-only calibration (twin route, worst-phase q, q known)."""
    Sm = ev_main.groupby("video").D_w.max()
    J, m = int((Sm > 0).sum()), len(Sm)
    U = float(ucp(J, m, DELTA / 2))
    S = ev_ood.groupby("video").agg(n=("p_w", "size"), p_w=("p_w", "mean"), q_w=("q_w", "mean"), w=("D_w", "max"))
    rec = [dict(video=v, n=int(r.n), p_w=float(r.p_w), q_w=float(r.q_w), D_w_any=bool(r.w > 0), U_half=U,
                accepted=bool(r.q_w + U <= eps), false_accept=bool(r.q_w + U <= eps and r.p_w > eps)) for v, r in S.iterrows()]
    return dict(J_main=J, m_main=m, U_half=U, cameras=len(rec), accepted=int(sum(r["accepted"] for r in rec)),
                p_w_gt_eps=int(sum(r["p_w"] > eps for r in rec)), false_accept=int(sum(r["false_accept"] for r in rec)),
                per_camera=rec)


def site_of(row):
    return f"CDnet:{row['category']}" if not str(row["video"])[:2] in ("I_", "O_") else f"LASIESTA:{str(row['video'])[:4]}"


def site_bound(ev):
    ev = ev.assign(site=ev.apply(site_of, axis=1))
    per_site = ev.groupby("site").agg(n=("D_w", "size"), scenes=("video", "nunique"), disc=("D_w", "sum"),
                                      D_site=("D_w", "mean"))
    m_site = len(per_site)
    J_site = int((per_site.disc > 0).sum())
    loso = []
    for s, r in per_site.iterrows():
        Jm = J_site - int(r.disc > 0)
        U = float(ucp(Jm, m_site - 1))
        loso.append(dict(site=s, n=int(r.n), scenes=int(r.scenes), D_site=float(r.D_site), U_minus=U, viol=bool(r.D_site > U)))
    L = pd.DataFrame(loso)
    return dict(m_site=m_site, J_site=J_site, U_pop_site=float(ucp(J_site, m_site)),
                m_scene=int(ev.video.nunique()), J_scene=int((ev.groupby("video").D_w.max() > 0).sum()),
                loso_site_violations=int(L.viol.sum()),
                sites=per_site.reset_index().to_dict("records"), loso=L.to_dict("records"))


def main():
    rows = {t: [r for tag in tags for r in event_rows(tag, t)] for t, tags in TIERS.items()}
    wp = dict(meta=dict(doc=__doc__.split("2. Site")[0].strip(), seed=SEED, delta=DELTA), points={})
    sb = dict(meta=dict(doc="Site-level population bound: " + __doc__.split("2. Site-level population bound.")[1].split("Outputs")[0].strip(),
                        delta=DELTA), points={})
    for name, (d, K, eps) in POINTS.items():
        evs = {t: worst_table(r, d, K) for t, r in rows.items()}
        allev = pd.concat(evs.values(), ignore_index=True)
        wp["points"][name] = dict(op=dict(d_min=d, K=K, eps=eps),
                                  tiers={**{t: tier_record(e) for t, e in evs.items()}, "all_tiers": tier_record(allev)},
                                  exchange_main=exchange(evs["main"], eps),
                                  ood_with_main_calibration={t: out_of_domain(evs["main"], evs[t], eps)
                                                             for t in ("jitter", "night")})
        sb["points"][name] = dict(op=dict(d_min=d, K=K, eps=eps), main=site_bound(evs["main"]))
    dump(wp, "worst_phase.json")
    dump(sb, "site_bound.json")
    for name in POINTS:
        t = wp["points"][name]["tiers"]["main"]
        x = wp["points"][name]["exchange_main"]
        s = sb["points"][name]["main"]
        print(name, "main p_w", round(t["p_w"], 4), "q_w", round(t["q_w"], 4), "Dw", t["D_w"], "Uf", round(t["U_fleet"], 4),
              "J", t["J"], "Up", round(t["U_pop"], 4), "check", t["check_p_le_q_plus_D"])
        print("   exchange", {k: v for k, v in x.items() if k != "note"})
        print("   site m", s["m_site"], "J", s["J_site"], "U", round(s["U_pop_site"], 4), "loso viol", s["loso_site_violations"])
        print("   ood", {t: {k: v for k, v in o.items() if k != "per_camera"} for t, o in wp["points"][name]["ood_with_main_calibration"].items()})
        print("   night", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in wp["points"][name]["tiers"]["night"].items()})


if __name__ == "__main__":
    main()
